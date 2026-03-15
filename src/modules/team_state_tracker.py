"""
Team State Tracker
-------------------
Accumulates per-team post-match stats chronologically for rolling feature computation.

This is the data backbone for the Lead State Efficiency module. It maintains a sliding
window of recent games per team and computes rolling aggregates on demand.

Leakage prevention:
    - record_game() is called AFTER the match result is known (during the UPDATE phase)
    - get_features() is called BEFORE the match prediction (during the PREDICT phase)
    - Features at game N use only games 1..N-1, never game N itself
"""

import numpy as np
from collections import deque
from typing import Dict, Optional


class TeamStateTracker:
    """
    Tracks per-team game history and computes rolling features.

    Usage in the pipeline loop:
        tracker = TeamStateTracker(window=20, min_games=5)

        for each game:
            # PREDICT phase: get features from prior games only
            features_a = tracker.get_features(team_a)
            features_b = tracker.get_features(team_b)

            # ... make prediction ...

            # UPDATE phase: record this game's stats
            tracker.record_game(team_a, game_stats_a)
            tracker.record_game(team_b, game_stats_b)
    """

    def __init__(
        self,
        window: int = 20,
        min_games: int = 5,
        shrinkage_weight: float = 5.0,
    ):
        """
        Args:
            window: Number of recent games to keep per team.
            min_games: Minimum games before producing non-prior features.
            shrinkage_weight: Bayesian prior weight for conditional features.
                              A team with shrinkage_weight qualifying games
                              gets 50% shrinkage toward the prior.
        """
        self.window = window
        self.min_games = min_games
        self.shrinkage_weight = shrinkage_weight

        # {team_name: deque of game stat dicts}
        self.team_history: Dict[str, deque] = {}

        # Global priors computed across all recorded games
        self._global_counts = {
            "total_games": 0,
            "wins": 0,
            "first_dragon": 0,
            "first_herald": 0,
            "first_tower": 0,
            "first_blood": 0,
            "ahead_2k_15_count": 0,
            "ahead_2k_15_wins": 0,
            "ahead_3k_20_count": 0,
            "ahead_3k_20_wins": 0,
            "behind_2k_15_count": 0,
            "behind_2k_15_wins": 0,
            "gd15_sum": 0.0,
            "gd20_sum": 0.0,
            "gamelength_ahead_sum": 0.0,
            "gamelength_ahead_count": 0,
        }

    def record_game(self, team: str, stats: dict) -> None:
        """
        Record a completed game for a team.

        Called in the UPDATE phase, after the match result is known.

        Args:
            team: Team name.
            stats: Dict with the following keys:
                Required: result (0 or 1)
                Optional: golddiffat15, golddiffat20, firstdragon, firstherald,
                          firsttower, firstblood, dragons, barons, towers,
                          gamelength, totalgold, opp_totalgold, heralds
        """
        if team not in self.team_history:
            self.team_history[team] = deque(maxlen=self.window)

        # Compute state flags
        gd15 = stats.get("golddiffat15", 0.0) or 0.0
        gd20 = stats.get("golddiffat20", 0.0) or 0.0
        result = stats.get("result", 0)

        record = {
            "result": int(result),
            "golddiffat15": float(gd15),
            "golddiffat20": float(gd20),
            "firstdragon": int(stats.get("firstdragon", 0) or 0),
            "firstherald": int(stats.get("firstherald", 0) or 0),
            "firsttower": int(stats.get("firsttower", 0) or 0),
            "firstblood": int(stats.get("firstblood", 0) or 0),
            "dragons": int(stats.get("dragons", 0) or 0),
            "barons": int(stats.get("barons", 0) or 0),
            "towers": int(stats.get("towers", 0) or 0),
            "gamelength": float(stats.get("gamelength", 0) or 0),
            "totalgold": float(stats.get("totalgold", 0) or 0),
            "opp_totalgold": float(stats.get("opp_totalgold", 0) or 0),
            # State flags
            "ahead_2k_at_15": gd15 >= 2000,
            "ahead_3k_at_20": gd20 >= 3000,
            "behind_2k_at_15": gd15 <= -2000,
        }

        self.team_history[team].append(record)

        # Update global priors
        self._update_global_priors(record)

    def _update_global_priors(self, record: dict) -> None:
        """Update running global statistics for shrinkage priors."""
        g = self._global_counts
        g["total_games"] += 1
        g["wins"] += record["result"]
        g["first_dragon"] += record["firstdragon"]
        g["first_herald"] += record["firstherald"]
        g["first_tower"] += record["firsttower"]
        g["first_blood"] += record["firstblood"]
        g["gd15_sum"] += record["golddiffat15"]
        g["gd20_sum"] += record["golddiffat20"]

        if record["ahead_2k_at_15"]:
            g["ahead_2k_15_count"] += 1
            g["ahead_2k_15_wins"] += record["result"]
        if record["ahead_3k_at_20"]:
            g["ahead_3k_20_count"] += 1
            g["ahead_3k_20_wins"] += record["result"]
        if record["behind_2k_at_15"]:
            g["behind_2k_15_count"] += 1
            g["behind_2k_15_wins"] += record["result"]
        if record["ahead_2k_at_15"] and record["gamelength"] > 0:
            g["gamelength_ahead_sum"] += record["gamelength"]
            g["gamelength_ahead_count"] += 1

    def get_features(self, team: str) -> dict:
        """
        Compute rolling features for a team from their game history.

        Called in the PREDICT phase, before the match result is known.
        Uses only games already recorded (no leakage).

        Returns:
            Dict of feature name -> value. Returns prior-based defaults
            if the team has fewer than min_games.
        """
        history = self.team_history.get(team, deque())
        n = len(history)

        if n == 0:
            return self._default_features()

        games = list(history)

        # Family A: Lead Creation (rolling averages)
        avg_gd15 = np.mean([g["golddiffat15"] for g in games])
        avg_gd20 = np.mean([g["golddiffat20"] for g in games])
        first_dragon_rate = np.mean([g["firstdragon"] for g in games])
        first_herald_rate = np.mean([g["firstherald"] for g in games])
        first_tower_rate = np.mean([g["firsttower"] for g in games])
        first_blood_rate = np.mean([g["firstblood"] for g in games])

        # Family C: Lead Conversion (state-conditional features)
        ahead_2k_15 = [g for g in games if g["ahead_2k_at_15"]]
        ahead_3k_20 = [g for g in games if g["ahead_3k_at_20"]]
        behind_2k_15 = [g for g in games if g["behind_2k_at_15"]]

        # Win rates with shrinkage
        wr_ahead_2k_15 = self._shrunk_rate(
            ahead_2k_15, "ahead_2k_15"
        )
        wr_ahead_3k_20 = self._shrunk_rate(
            ahead_3k_20, "ahead_3k_20"
        )
        wr_behind_2k_15 = self._shrunk_rate(
            behind_2k_15, "behind_2k_15"
        )

        # Average game length when ahead at 15
        if ahead_2k_15:
            lengths = [g["gamelength"] for g in ahead_2k_15 if g["gamelength"] > 0]
            if lengths:
                avg_gl_ahead = np.mean(lengths)
            else:
                avg_gl_ahead = self._global_avg_gl_ahead()
        else:
            avg_gl_ahead = self._global_avg_gl_ahead()

        # Apply shrinkage on lead creation features too when sample is small
        if n < self.min_games:
            shrink = n / (n + self.shrinkage_weight)
            prior = self._default_features()
            avg_gd15 = shrink * avg_gd15 + (1 - shrink) * prior["avg_gd15"]
            avg_gd20 = shrink * avg_gd20 + (1 - shrink) * prior["avg_gd20"]
            first_dragon_rate = shrink * first_dragon_rate + (1 - shrink) * prior["first_dragon_rate"]
            first_herald_rate = shrink * first_herald_rate + (1 - shrink) * prior["first_herald_rate"]
            first_tower_rate = shrink * first_tower_rate + (1 - shrink) * prior["first_tower_rate"]
            first_blood_rate = shrink * first_blood_rate + (1 - shrink) * prior["first_blood_rate"]

        return {
            # Family A: Lead Creation
            "avg_gd15": avg_gd15,
            "avg_gd20": avg_gd20,
            "first_dragon_rate": first_dragon_rate,
            "first_herald_rate": first_herald_rate,
            "first_tower_rate": first_tower_rate,
            "first_blood_rate": first_blood_rate,
            # Family C: Lead Conversion
            "win_rate_when_ahead_2k_15": wr_ahead_2k_15,
            "win_rate_when_ahead_3k_20": wr_ahead_3k_20,
            "win_rate_when_behind_2k_15": wr_behind_2k_15,
            "avg_game_length_when_ahead": avg_gl_ahead,
            # Metadata
            "n_games": n,
            "n_ahead_2k_15": len(ahead_2k_15),
            "n_ahead_3k_20": len(ahead_3k_20),
            "n_behind_2k_15": len(behind_2k_15),
        }

    def get_confidence(self, team: str) -> float:
        """
        Return confidence level for this team's features.

        Scales from 0.0 (no data) to 1.0 (full window of data).
        Below min_games, confidence is proportionally low.
        """
        n = len(self.team_history.get(team, deque()))
        if n == 0:
            return 0.0
        if n < self.min_games:
            return 0.3 * (n / self.min_games)
        return min(1.0, 0.3 + 0.7 * (n / self.window))

    def _shrunk_rate(self, qualifying_games: list, state_key: str) -> float:
        """
        Compute a Bayesian-shrunk win rate for state-conditional features.

        Uses the global win rate for that state as the prior.
        """
        n = len(qualifying_games)
        if n == 0:
            return self._global_conditional_wr(state_key)

        team_rate = np.mean([g["result"] for g in qualifying_games])
        prior_rate = self._global_conditional_wr(state_key)

        # Bayesian shrinkage: weighted average of team rate and prior
        shrunk = (team_rate * n + prior_rate * self.shrinkage_weight) / (
            n + self.shrinkage_weight
        )
        return shrunk

    def _global_conditional_wr(self, state_key: str) -> float:
        """Get global win rate for a conditional state. Falls back to sensible defaults."""
        g = self._global_counts
        if state_key == "ahead_2k_15" and g["ahead_2k_15_count"] > 0:
            return g["ahead_2k_15_wins"] / g["ahead_2k_15_count"]
        elif state_key == "ahead_3k_20" and g["ahead_3k_20_count"] > 0:
            return g["ahead_3k_20_wins"] / g["ahead_3k_20_count"]
        elif state_key == "behind_2k_15" and g["behind_2k_15_count"] > 0:
            return g["behind_2k_15_wins"] / g["behind_2k_15_count"]

        # Sensible defaults before we have enough global data
        defaults = {
            "ahead_2k_15": 0.82,
            "ahead_3k_20": 0.88,
            "behind_2k_15": 0.18,
        }
        return defaults.get(state_key, 0.50)

    def _global_avg_gl_ahead(self) -> float:
        """Get global average game length when ahead at 15."""
        g = self._global_counts
        if g["gamelength_ahead_count"] > 0:
            return g["gamelength_ahead_sum"] / g["gamelength_ahead_count"]
        return 1800.0  # Default 30 min

    def _default_features(self) -> dict:
        """Return prior-based default features for teams with no history."""
        g = self._global_counts
        total = max(g["total_games"], 1)

        return {
            "avg_gd15": g["gd15_sum"] / total if total > 1 else 0.0,
            "avg_gd20": g["gd20_sum"] / total if total > 1 else 0.0,
            "first_dragon_rate": g["first_dragon"] / total if total > 1 else 0.50,
            "first_herald_rate": g["first_herald"] / total if total > 1 else 0.50,
            "first_tower_rate": g["first_tower"] / total if total > 1 else 0.50,
            "first_blood_rate": g["first_blood"] / total if total > 1 else 0.50,
            "win_rate_when_ahead_2k_15": self._global_conditional_wr("ahead_2k_15"),
            "win_rate_when_ahead_3k_20": self._global_conditional_wr("ahead_3k_20"),
            "win_rate_when_behind_2k_15": self._global_conditional_wr("behind_2k_15"),
            "avg_game_length_when_ahead": self._global_avg_gl_ahead(),
            "n_games": 0,
            "n_ahead_2k_15": 0,
            "n_ahead_3k_20": 0,
            "n_behind_2k_15": 0,
        }

    def game_count(self, team: str) -> int:
        """Return number of recorded games for a team."""
        return len(self.team_history.get(team, deque()))

    def reset_team(self, team: str) -> None:
        """Clear history for a team (e.g., after major roster change)."""
        if team in self.team_history:
            self.team_history[team].clear()
