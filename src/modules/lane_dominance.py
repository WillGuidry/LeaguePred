"""
Lane Dominance Module
----------------------
Tracks per-lane (top, jng, mid, bot, sup) player performance to generate
team-level lane dominance features.

Features computed per lane:
    - avg_gd10: Average gold diff at 10 minutes for that lane
    - avg_gd15: Average gold diff at 15 minutes for that lane
    - avg_xpd10: Average XP diff at 10 minutes for that lane
    - avg_csd10: Average CS diff at 10 minutes for that lane

Team-level aggregates:
    - lane_dominance_score: Weighted sum of lane gold diffs (normalized)
    - best_lane_gd10: Strongest lane's avg gold diff at 10
    - worst_lane_gd10: Weakest lane's avg gold diff at 10
    - lane_spread: Spread between best and worst lane (concentration vs balanced)
    - solo_lane_avg_gd10: Average of top+mid gold diffs at 10
    - bot_lane_avg_gd10: Average of bot+sup gold diffs at 10
    - jng_proximity_gd10: Jungle gold diff at 10 (proxy for map pressure)

Player-level scores:
    - player_dominance_score per position: Composite of gold/xp/cs diffs

Leakage prevention:
    - record_game() called in UPDATE phase (after result known)
    - get_features() called in PREDICT phase (before prediction, prior games only)
"""

import numpy as np
from collections import deque
from typing import Dict, Optional, Tuple


POSITIONS = ["top", "jng", "mid", "bot", "sup"]

# Weights for lane dominance score (how much each lane matters)
LANE_WEIGHTS = {
    "top": 0.18,
    "jng": 0.22,
    "mid": 0.22,
    "bot": 0.22,
    "sup": 0.16,
}

# Normalization constants for gold diffs (typical std devs)
GD10_SCALE = 400.0   # ~400 gold std dev at 10 min
GD15_SCALE = 800.0   # ~800 gold std dev at 15 min
XPD10_SCALE = 300.0  # ~300 xp std dev at 10 min
CSD10_SCALE = 10.0   # ~10 cs std dev at 10 min


class LaneDominanceTracker:
    """
    Tracks per-lane player performance for each team using a rolling window.

    Stores per-game lane stats and computes rolling aggregates on demand.
    """

    def __init__(self, window: int = 20, min_games: int = 5, shrinkage_weight: float = 5.0):
        self.window = window
        self.min_games = min_games
        self.shrinkage_weight = shrinkage_weight

        # {team_name: deque of lane_stats dicts}
        # Each entry: {"top": {...}, "jng": {...}, "mid": {...}, "bot": {...}, "sup": {...}}
        self.team_history: Dict[str, deque] = {}

        # Global running averages for shrinkage priors
        self._global_sums = {pos: {"gd10": 0.0, "gd15": 0.0, "xpd10": 0.0, "csd10": 0.0, "count": 0}
                             for pos in POSITIONS}

    def record_game(self, team: str, lane_stats: dict) -> None:
        """
        Record per-lane stats for a completed game.

        Args:
            team: Team name.
            lane_stats: Dict keyed by position ("top", "jng", "mid", "bot", "sup"),
                        each containing:
                          - golddiffat10, golddiffat15
                          - xpdiffat10, xpdiffat15
                          - csdiffat10, csdiffat15
                          - kills, deaths, assists
                          - damageshare, earnedgoldshare
        """
        if team not in self.team_history:
            self.team_history[team] = deque(maxlen=self.window)

        record = {}
        for pos in POSITIONS:
            pstats = lane_stats.get(pos, {})
            entry = {
                "gd10": _safe_float(pstats.get("golddiffat10", 0.0)),
                "gd15": _safe_float(pstats.get("golddiffat15", 0.0)),
                "xpd10": _safe_float(pstats.get("xpdiffat10", 0.0)),
                "xpd15": _safe_float(pstats.get("xpdiffat15", 0.0)),
                "csd10": _safe_float(pstats.get("csdiffat10", 0.0)),
                "csd15": _safe_float(pstats.get("csdiffat15", 0.0)),
                "kills": _safe_float(pstats.get("kills", 0.0)),
                "deaths": _safe_float(pstats.get("deaths", 0.0)),
                "assists": _safe_float(pstats.get("assists", 0.0)),
                "damageshare": _safe_float(pstats.get("damageshare", 0.0)),
                "earnedgoldshare": _safe_float(pstats.get("earnedgoldshare", 0.0)),
            }
            record[pos] = entry

            # Update global sums
            g = self._global_sums[pos]
            g["gd10"] += entry["gd10"]
            g["gd15"] += entry["gd15"]
            g["xpd10"] += entry["xpd10"]
            g["csd10"] += entry["csd10"]
            g["count"] += 1

        self.team_history[team].append(record)

    def get_features(self, team: str) -> dict:
        """
        Compute lane dominance features from rolling history.

        Returns dict with:
            Per-lane features (prefixed by position):
                {pos}_avg_gd10, {pos}_avg_gd15, {pos}_avg_xpd10, {pos}_avg_csd10
                {pos}_dominance_score (composite)

            Team-level aggregates:
                lane_dominance_score, best_lane_gd10, worst_lane_gd10,
                lane_spread, solo_lane_avg_gd10, bot_lane_avg_gd10,
                jng_proximity_gd10, lane_consistency
        """
        history = self.team_history.get(team, deque())
        n = len(history)

        if n == 0:
            return self._default_features()

        games = list(history)
        features = {}

        lane_gd10s = {}
        lane_scores = {}

        for pos in POSITIONS:
            avg_gd10 = np.mean([g[pos]["gd10"] for g in games])
            avg_gd15 = np.mean([g[pos]["gd15"] for g in games])
            avg_xpd10 = np.mean([g[pos]["xpd10"] for g in games])
            avg_csd10 = np.mean([g[pos]["csd10"] for g in games])

            # Apply shrinkage for small samples
            if n < self.min_games:
                shrink = n / (n + self.shrinkage_weight)
                avg_gd10 *= shrink
                avg_gd15 *= shrink
                avg_xpd10 *= shrink
                avg_csd10 *= shrink

            features[f"{pos}_avg_gd10"] = avg_gd10
            features[f"{pos}_avg_gd15"] = avg_gd15
            features[f"{pos}_avg_xpd10"] = avg_xpd10
            features[f"{pos}_avg_csd10"] = avg_csd10

            # Per-lane dominance score: composite of gold + xp + cs advantages
            dom_score = (
                avg_gd10 / GD10_SCALE * 0.50
                + avg_xpd10 / XPD10_SCALE * 0.30
                + avg_csd10 / CSD10_SCALE * 0.20
            )
            features[f"{pos}_dominance_score"] = dom_score
            lane_gd10s[pos] = avg_gd10
            lane_scores[pos] = dom_score

        # Team-level aggregates
        weighted_score = sum(
            lane_scores[pos] * LANE_WEIGHTS[pos] for pos in POSITIONS
        )
        features["lane_dominance_score"] = weighted_score

        gd10_values = [lane_gd10s[pos] for pos in POSITIONS]
        features["best_lane_gd10"] = max(gd10_values)
        features["worst_lane_gd10"] = min(gd10_values)
        features["lane_spread"] = max(gd10_values) - min(gd10_values)

        # Solo lanes vs bot lane
        features["solo_lane_avg_gd10"] = (lane_gd10s["top"] + lane_gd10s["mid"]) / 2.0
        features["bot_lane_avg_gd10"] = (lane_gd10s["bot"] + lane_gd10s["sup"]) / 2.0
        features["jng_proximity_gd10"] = lane_gd10s["jng"]

        # Lane consistency: how stable are gold diffs game to game?
        if n >= 3:
            per_game_totals = []
            for g in games:
                total = sum(g[pos]["gd10"] * LANE_WEIGHTS[pos] for pos in POSITIONS)
                per_game_totals.append(total)
            features["lane_consistency"] = np.std(per_game_totals) / GD10_SCALE
        else:
            features["lane_consistency"] = 0.5  # neutral

        features["n_games"] = n
        return features

    def get_confidence(self, team: str) -> float:
        """Confidence based on number of recorded games."""
        n = len(self.team_history.get(team, deque()))
        if n == 0:
            return 0.0
        if n < self.min_games:
            return 0.3 * (n / self.min_games)
        return min(1.0, 0.3 + 0.7 * (n / self.window))

    def get_player_scores(self, team: str) -> dict:
        """
        Get per-position dominance scores for display/analysis.

        Returns:
            {pos: {"dominance_score": float, "avg_gd10": float, ...}}
        """
        features = self.get_features(team)
        scores = {}
        for pos in POSITIONS:
            scores[pos] = {
                "dominance_score": features.get(f"{pos}_dominance_score", 0.0),
                "avg_gd10": features.get(f"{pos}_avg_gd10", 0.0),
                "avg_gd15": features.get(f"{pos}_avg_gd15", 0.0),
                "avg_xpd10": features.get(f"{pos}_avg_xpd10", 0.0),
                "avg_csd10": features.get(f"{pos}_avg_csd10", 0.0),
            }
        return scores

    def _default_features(self) -> dict:
        """Return neutral features for teams with no history."""
        features = {}
        for pos in POSITIONS:
            features[f"{pos}_avg_gd10"] = 0.0
            features[f"{pos}_avg_gd15"] = 0.0
            features[f"{pos}_avg_xpd10"] = 0.0
            features[f"{pos}_avg_csd10"] = 0.0
            features[f"{pos}_dominance_score"] = 0.0
        features["lane_dominance_score"] = 0.0
        features["best_lane_gd10"] = 0.0
        features["worst_lane_gd10"] = 0.0
        features["lane_spread"] = 0.0
        features["solo_lane_avg_gd10"] = 0.0
        features["bot_lane_avg_gd10"] = 0.0
        features["jng_proximity_gd10"] = 0.0
        features["lane_consistency"] = 0.5
        features["n_games"] = 0
        return features

    def game_count(self, team: str) -> int:
        return len(self.team_history.get(team, deque()))

    def reset_team(self, team: str) -> None:
        if team in self.team_history:
            self.team_history[team].clear()


class LaneDominanceModule:
    """
    Computes lane dominance edge between two teams.

    Interface matches other modules (LeadStateModule, MomentumModule).
    """

    def __init__(self, tracker: LaneDominanceTracker):
        self.tracker = tracker
        self._last_features_a = {}
        self._last_features_b = {}
        self._last_confidence = 0.0

    def compute(self, team_a: str, team_b: str) -> float:
        """
        Compute lane dominance edge for team_a vs team_b.

        Returns:
            Positive = team_a has lane advantage, negative = team_b.
            Range roughly [-1, 1].
        """
        feats_a = self.tracker.get_features(team_a)
        feats_b = self.tracker.get_features(team_b)
        self._last_features_a = feats_a
        self._last_features_b = feats_b

        conf_a = self.tracker.get_confidence(team_a)
        conf_b = self.tracker.get_confidence(team_b)
        self._last_confidence = min(conf_a, conf_b)

        # Edge = difference in lane dominance scores
        edge = feats_a["lane_dominance_score"] - feats_b["lane_dominance_score"]

        # Scale by confidence
        edge *= self._last_confidence

        return edge

    def confidence(self) -> float:
        return self._last_confidence


def _safe_float(val, default=0.0) -> float:
    """Safely convert to float."""
    if val is None:
        return default
    try:
        v = float(val)
        if np.isnan(v):
            return default
        return v
    except (TypeError, ValueError):
        return default
