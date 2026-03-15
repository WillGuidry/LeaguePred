"""
Team State Tracker
-------------------
Accumulates per-team post-match stats chronologically for rolling feature computation.

This is the data backbone for the Lead State Efficiency module. It maintains a sliding
window of recent games per team and computes rolling aggregates on demand across
five feature families:

    A. Lead Creation — how teams build advantages
    B. Advantage Quality — how robust/stable leads are
    C. Lead Conversion — translating leads into wins
    D. Throw Tendency — failing from winning states
    E. Comeback / Resistance — performing from losing states

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
    Tracks per-team game history and computes rolling features across
    five lead-state families.

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
            "first_baron": 0,
            # State-conditional counts
            "ahead_2k_15_count": 0,
            "ahead_2k_15_wins": 0,
            "ahead_3k_20_count": 0,
            "ahead_3k_20_wins": 0,
            "behind_2k_15_count": 0,
            "behind_2k_15_wins": 0,
            # Lead stability
            "ahead_15_count": 0,
            "ahead_15_still_ahead_20": 0,
            "ahead_15_behind_20": 0,
            # Closing times
            "gamelength_ahead_15_sum": 0.0,
            "gamelength_ahead_15_count": 0,
            "gamelength_ahead_20_sum": 0.0,
            "gamelength_ahead_20_count": 0,
            "gamelength_behind_15_sum": 0.0,
            "gamelength_behind_15_count": 0,
            # Herald → tower
            "first_herald_games": 0,
            "first_herald_and_first_tower": 0,
            # Baron conversion
            "first_baron_count": 0,
            "first_baron_wins": 0,
            # Baron throw (ahead + first baron + lost)
            "ahead_first_baron_count": 0,
            "ahead_first_baron_wins": 0,
            # Dragon soul
            "soul_count": 0,
            # Gold sums
            "gd10_sum": 0.0,
            "gd15_sum": 0.0,
            "gd20_sum": 0.0,
            # Recovery
            "behind_15_count": 0,
            "behind_15_recovered_20": 0,
            # Plate diff
            "plate_diff_sum": 0.0,
        }

    def record_game(self, team: str, stats: dict) -> None:
        """
        Record a completed game for a team.

        Called in the UPDATE phase, after the match result is known.

        Args:
            team: Team name.
            stats: Dict with keys:
                Required: result (0 or 1)
                Optional: golddiffat10, golddiffat15, golddiffat20, golddiffat25,
                          firstdragon, firstherald, firsttower, firstblood, firstbaron,
                          dragons, barons, towers, elementaldrakes, heralds, elders,
                          gamelength, totalgold, opp_totalgold,
                          turretplates, opp_turretplates
        """
        if team not in self.team_history:
            self.team_history[team] = deque(maxlen=self.window)

        def _f(key, default=0.0):
            val = stats.get(key, default)
            if val is None:
                return default
            try:
                return float(val)
            except (TypeError, ValueError):
                return default

        def _i(key, default=0):
            val = stats.get(key, default)
            if val is None:
                return default
            try:
                return int(float(val))
            except (TypeError, ValueError):
                return default

        result = _i("result", 0)
        gd10 = _f("golddiffat10")
        gd15 = _f("golddiffat15")
        gd20 = _f("golddiffat20")
        gd25 = _f("golddiffat25")
        gamelength = _f("gamelength")
        totalgold = _f("totalgold")
        opp_totalgold = _f("opp_totalgold")
        plates = _f("turretplates")
        opp_plates = _f("opp_turretplates")
        elementaldrakes = _i("elementaldrakes")

        # State flags
        ahead_2k_at_15 = gd15 >= 2000
        ahead_3k_at_20 = gd20 >= 3000
        behind_2k_at_15 = gd15 <= -2000
        ahead_at_15 = gd15 > 0
        behind_at_15 = gd15 < 0
        ahead_at_20 = gd20 > 0

        # Derived: gold diff at end (from totalgold)
        gold_diff_end = totalgold - opp_totalgold if (totalgold > 0 and opp_totalgold > 0) else 0.0

        record = {
            "result": result,
            "golddiffat10": gd10,
            "golddiffat15": gd15,
            "golddiffat20": gd20,
            "golddiffat25": gd25,
            "gold_diff_end": gold_diff_end,
            "firstdragon": _i("firstdragon"),
            "firstherald": _i("firstherald"),
            "firsttower": _i("firsttower"),
            "firstblood": _i("firstblood"),
            "firstbaron": _i("firstbaron"),
            "dragons": _i("dragons"),
            "barons": _i("barons"),
            "towers": _i("towers"),
            "elementaldrakes": elementaldrakes,
            "heralds": _i("heralds"),
            "elders": _i("elders"),
            "gamelength": gamelength,
            "totalgold": totalgold,
            "opp_totalgold": opp_totalgold,
            "turretplates": plates,
            "opp_turretplates": opp_plates,
            "plate_diff": plates - opp_plates,
            # State flags
            "ahead_2k_at_15": ahead_2k_at_15,
            "ahead_3k_at_20": ahead_3k_at_20,
            "behind_2k_at_15": behind_2k_at_15,
            "ahead_at_15": ahead_at_15,
            "behind_at_15": behind_at_15,
            "ahead_at_20": ahead_at_20,
            # Derived flags
            "lead_evaporated": ahead_at_15 and not ahead_at_20,  # ahead@15, behind/even@20
            "gold_recovered": behind_at_15 and ahead_at_20,       # behind@15, ahead@20
            "has_soul": elementaldrakes >= 4,
        }

        self.team_history[team].append(record)
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
        g["first_baron"] += record["firstbaron"]
        g["gd10_sum"] += record["golddiffat10"]
        g["gd15_sum"] += record["golddiffat15"]
        g["gd20_sum"] += record["golddiffat20"]
        g["plate_diff_sum"] += record["plate_diff"]

        if record["has_soul"]:
            g["soul_count"] += 1

        # State-conditional priors
        if record["ahead_2k_at_15"]:
            g["ahead_2k_15_count"] += 1
            g["ahead_2k_15_wins"] += record["result"]
        if record["ahead_3k_at_20"]:
            g["ahead_3k_20_count"] += 1
            g["ahead_3k_20_wins"] += record["result"]
        if record["behind_2k_at_15"]:
            g["behind_2k_15_count"] += 1
            g["behind_2k_15_wins"] += record["result"]

        # Lead stability
        if record["ahead_at_15"]:
            g["ahead_15_count"] += 1
            if record["ahead_at_20"]:
                g["ahead_15_still_ahead_20"] += 1
            else:
                g["ahead_15_behind_20"] += 1

        # Closing times
        if record["ahead_2k_at_15"] and record["gamelength"] > 0:
            g["gamelength_ahead_15_sum"] += record["gamelength"]
            g["gamelength_ahead_15_count"] += 1
        if record["ahead_3k_at_20"] and record["gamelength"] > 0:
            g["gamelength_ahead_20_sum"] += record["gamelength"]
            g["gamelength_ahead_20_count"] += 1
        if record["behind_2k_at_15"] and record["gamelength"] > 0:
            g["gamelength_behind_15_sum"] += record["gamelength"]
            g["gamelength_behind_15_count"] += 1

        # Herald → tower
        if record["firstherald"]:
            g["first_herald_games"] += 1
            if record["firsttower"]:
                g["first_herald_and_first_tower"] += 1

        # Baron conversion
        if record["firstbaron"]:
            g["first_baron_count"] += 1
            g["first_baron_wins"] += record["result"]
            if record["ahead_2k_at_15"]:
                g["ahead_first_baron_count"] += 1
                g["ahead_first_baron_wins"] += record["result"]

        # Recovery
        if record["behind_at_15"]:
            g["behind_15_count"] += 1
            if record["ahead_at_20"]:
                g["behind_15_recovered_20"] += 1

    def get_features(self, team: str) -> dict:
        """
        Compute rolling features for a team from their game history.

        Called in the PREDICT phase, before the match result is known.
        Uses only games already recorded (no leakage).

        Returns dict organized by feature family:
            A. Lead Creation (7 features)
            B. Advantage Quality (3 features)
            C. Lead Conversion (8 features)
            D. Throw Tendency (3 features)
            E. Comeback / Resistance (3 features)
            + metadata
        """
        history = self.team_history.get(team, deque())
        n = len(history)

        if n == 0:
            return self._default_features()

        games = list(history)

        # ===========================================================
        # Family A: Lead Creation
        # ===========================================================
        avg_gd10 = np.mean([g["golddiffat10"] for g in games])
        avg_gd15 = np.mean([g["golddiffat15"] for g in games])
        first_dragon_rate = np.mean([g["firstdragon"] for g in games])
        first_herald_rate = np.mean([g["firstherald"] for g in games])
        first_tower_rate = np.mean([g["firsttower"] for g in games])
        first_blood_rate = np.mean([g["firstblood"] for g in games])
        plate_diff = np.mean([g["plate_diff"] for g in games])

        # ===========================================================
        # Family B: Advantage Quality
        # ===========================================================
        ahead_15_games = [g for g in games if g["ahead_at_15"]]

        if ahead_15_games:
            lead_stability = np.mean([
                1 if g["ahead_at_20"] else 0 for g in ahead_15_games
            ])
            # Gold volatility when ahead: how much does lead change 15→20?
            vols = [abs(g["golddiffat20"] - g["golddiffat15"]) for g in ahead_15_games]
            gold_volatility_when_ahead = np.mean(vols)
        else:
            lead_stability = self._global_rate("lead_stability")
            gold_volatility_when_ahead = 500.0  # neutral default

        # Compound lead: ahead at 15 AND at least one first objective
        compound_games = [
            g for g in ahead_15_games
            if g["firstdragon"] or g["firstherald"]
        ]
        compound_lead_rate = len(compound_games) / max(n, 1)

        # ===========================================================
        # Family C: Lead Conversion
        # ===========================================================
        ahead_2k_15 = [g for g in games if g["ahead_2k_at_15"]]
        ahead_3k_20 = [g for g in games if g["ahead_3k_at_20"]]

        wr_ahead_2k_15 = self._shrunk_rate(
            [g["result"] for g in ahead_2k_15], "ahead_2k_15"
        )
        wr_ahead_3k_20 = self._shrunk_rate(
            [g["result"] for g in ahead_3k_20], "ahead_3k_20"
        )

        # Closing times
        close_time_ahead_15 = self._conditional_avg(
            ahead_2k_15, "gamelength", self._global_avg("close_time_ahead_15")
        )
        close_time_ahead_20 = self._conditional_avg(
            ahead_3k_20, "gamelength", self._global_avg("close_time_ahead_20")
        )

        # Gold snowball rate: how much lead grows from 15→end when ahead at 15
        snowball_vals = []
        for g in ahead_2k_15:
            if g["golddiffat15"] > 0 and g["gold_diff_end"] != 0:
                snowball_vals.append(g["gold_diff_end"] - g["golddiffat15"])
        gold_snowball_rate = np.mean(snowball_vals) if snowball_vals else 0.0

        # Dragon soul rate (4+ elementaldrakes)
        dragon_soul_rate = np.mean([1 if g["has_soul"] else 0 for g in games])

        # Herald → tower conversion: P(first_tower | first_herald)
        herald_games = [g for g in games if g["firstherald"]]
        if herald_games:
            herald_tower_conv = np.mean([g["firsttower"] for g in herald_games])
        else:
            herald_tower_conv = self._global_rate("herald_tower")

        # Baron conversion: win rate when got first baron
        baron_games = [g for g in games if g["firstbaron"]]
        baron_win_rate = self._shrunk_rate(
            [g["result"] for g in baron_games], "first_baron"
        )

        # ===========================================================
        # Family D: Throw Tendency
        # ===========================================================
        throw_rate_2k_15 = 1.0 - wr_ahead_2k_15

        # Lead evaporation: ahead at 15 but behind/even at 20
        if ahead_15_games:
            lead_evaporation_rate = np.mean([
                1 if g["lead_evaporated"] else 0 for g in ahead_15_games
            ])
        else:
            lead_evaporation_rate = self._global_rate("lead_evaporation")

        # Baron throw: had first baron AND ahead at 15, but lost
        ahead_baron_games = [g for g in games if g["firstbaron"] and g["ahead_2k_at_15"]]
        baron_throw_rate = 1.0 - self._shrunk_rate(
            [g["result"] for g in ahead_baron_games], "ahead_first_baron"
        )

        # ===========================================================
        # Family E: Comeback / Resistance
        # ===========================================================
        behind_2k_15 = [g for g in games if g["behind_2k_at_15"]]
        wr_behind_2k_15 = self._shrunk_rate(
            [g["result"] for g in behind_2k_15], "behind_2k_15"
        )

        # Gold recovery rate: behind at 15 but ahead at 20
        behind_15_games = [g for g in games if g["behind_at_15"]]
        if behind_15_games:
            gold_recovery_rate = np.mean([
                1 if g["gold_recovered"] else 0 for g in behind_15_games
            ])
        else:
            gold_recovery_rate = self._global_rate("gold_recovery")

        # Extension time when behind: longer = better stalling
        extend_time_behind = self._conditional_avg(
            behind_2k_15, "gamelength", self._global_avg("extend_time_behind")
        )

        # ===========================================================
        # Apply shrinkage on lead creation features when sample is small
        # ===========================================================
        if n < self.min_games:
            shrink = n / (n + self.shrinkage_weight)
            prior = self._default_features()
            avg_gd10 = shrink * avg_gd10 + (1 - shrink) * prior["avg_gd10"]
            avg_gd15 = shrink * avg_gd15 + (1 - shrink) * prior["avg_gd15"]
            first_dragon_rate = shrink * first_dragon_rate + (1 - shrink) * prior["first_dragon_rate"]
            first_herald_rate = shrink * first_herald_rate + (1 - shrink) * prior["first_herald_rate"]
            first_tower_rate = shrink * first_tower_rate + (1 - shrink) * prior["first_tower_rate"]
            first_blood_rate = shrink * first_blood_rate + (1 - shrink) * prior["first_blood_rate"]
            plate_diff = shrink * plate_diff + (1 - shrink) * prior["plate_diff"]

        return {
            # Family A: Lead Creation
            "avg_gd10": avg_gd10,
            "avg_gd15": avg_gd15,
            "first_dragon_rate": first_dragon_rate,
            "first_herald_rate": first_herald_rate,
            "first_tower_rate": first_tower_rate,
            "first_blood_rate": first_blood_rate,
            "plate_diff": plate_diff,
            # Family B: Advantage Quality
            "lead_stability": lead_stability,
            "compound_lead_rate": compound_lead_rate,
            "gold_volatility_when_ahead": gold_volatility_when_ahead,
            # Family C: Lead Conversion
            "win_rate_when_ahead_2k_15": wr_ahead_2k_15,
            "win_rate_when_ahead_3k_20": wr_ahead_3k_20,
            "close_time_ahead_15": close_time_ahead_15,
            "close_time_ahead_20": close_time_ahead_20,
            "gold_snowball_rate": gold_snowball_rate,
            "dragon_soul_rate": dragon_soul_rate,
            "herald_tower_conv": herald_tower_conv,
            "baron_win_rate": baron_win_rate,
            # Family D: Throw Tendency
            "throw_rate_2k_15": throw_rate_2k_15,
            "lead_evaporation_rate": lead_evaporation_rate,
            "baron_throw_rate": baron_throw_rate,
            # Family E: Comeback / Resistance
            "win_rate_when_behind_2k_15": wr_behind_2k_15,
            "gold_recovery_rate": gold_recovery_rate,
            "extend_time_behind": extend_time_behind,
            # Metadata
            "n_games": n,
            "n_ahead_2k_15": len(ahead_2k_15),
            "n_ahead_3k_20": len(ahead_3k_20),
            "n_behind_2k_15": len(behind_2k_15),
            "n_ahead_15": len(ahead_15_games),
            "n_behind_15": len(behind_15_games),
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

    def _shrunk_rate(self, outcomes: list, state_key: str) -> float:
        """
        Compute a Bayesian-shrunk rate toward a global prior.

        Args:
            outcomes: List of 0/1 values.
            state_key: Key to look up global prior rate.
        """
        n = len(outcomes)
        if n == 0:
            return self._global_conditional_wr(state_key)

        team_rate = np.mean(outcomes)
        prior_rate = self._global_conditional_wr(state_key)

        shrunk = (team_rate * n + prior_rate * self.shrinkage_weight) / (
            n + self.shrinkage_weight
        )
        return shrunk

    def _conditional_avg(self, qualifying_games: list, key: str, default: float) -> float:
        """Average of a field across qualifying games, with fallback."""
        vals = [g[key] for g in qualifying_games if g.get(key, 0) > 0]
        return np.mean(vals) if vals else default

    def _global_conditional_wr(self, state_key: str) -> float:
        """Get global win rate for a conditional state."""
        g = self._global_counts

        lookups = {
            "ahead_2k_15": ("ahead_2k_15_wins", "ahead_2k_15_count"),
            "ahead_3k_20": ("ahead_3k_20_wins", "ahead_3k_20_count"),
            "behind_2k_15": ("behind_2k_15_wins", "behind_2k_15_count"),
            "first_baron": ("first_baron_wins", "first_baron_count"),
            "ahead_first_baron": ("ahead_first_baron_wins", "ahead_first_baron_count"),
        }

        if state_key in lookups:
            wins_key, count_key = lookups[state_key]
            if g[count_key] > 0:
                return g[wins_key] / g[count_key]

        defaults = {
            "ahead_2k_15": 0.82,
            "ahead_3k_20": 0.88,
            "behind_2k_15": 0.18,
            "first_baron": 0.75,
            "ahead_first_baron": 0.85,
        }
        return defaults.get(state_key, 0.50)

    def _global_rate(self, stat: str) -> float:
        """Get a global rate for non-win-rate conditional features."""
        g = self._global_counts
        rates = {
            "lead_stability": (
                g["ahead_15_still_ahead_20"] / max(g["ahead_15_count"], 1)
                if g["ahead_15_count"] > 0 else 0.72
            ),
            "lead_evaporation": (
                g["ahead_15_behind_20"] / max(g["ahead_15_count"], 1)
                if g["ahead_15_count"] > 0 else 0.28
            ),
            "herald_tower": (
                g["first_herald_and_first_tower"] / max(g["first_herald_games"], 1)
                if g["first_herald_games"] > 0 else 0.55
            ),
            "gold_recovery": (
                g["behind_15_recovered_20"] / max(g["behind_15_count"], 1)
                if g["behind_15_count"] > 0 else 0.28
            ),
        }
        return rates.get(stat, 0.50)

    def _global_avg(self, stat: str) -> float:
        """Get global average for time-based features."""
        g = self._global_counts
        avgs = {
            "close_time_ahead_15": (
                g["gamelength_ahead_15_sum"] / g["gamelength_ahead_15_count"]
                if g["gamelength_ahead_15_count"] > 0 else 1800.0
            ),
            "close_time_ahead_20": (
                g["gamelength_ahead_20_sum"] / g["gamelength_ahead_20_count"]
                if g["gamelength_ahead_20_count"] > 0 else 1800.0
            ),
            "extend_time_behind": (
                g["gamelength_behind_15_sum"] / g["gamelength_behind_15_count"]
                if g["gamelength_behind_15_count"] > 0 else 2000.0
            ),
        }
        return avgs.get(stat, 1800.0)

    def _default_features(self) -> dict:
        """Return prior-based default features for teams with no history."""
        g = self._global_counts
        total = max(g["total_games"], 1)

        return {
            # Family A
            "avg_gd10": g["gd10_sum"] / total if total > 1 else 0.0,
            "avg_gd15": g["gd15_sum"] / total if total > 1 else 0.0,
            "first_dragon_rate": g["first_dragon"] / total if total > 1 else 0.50,
            "first_herald_rate": g["first_herald"] / total if total > 1 else 0.50,
            "first_tower_rate": g["first_tower"] / total if total > 1 else 0.50,
            "first_blood_rate": g["first_blood"] / total if total > 1 else 0.50,
            "plate_diff": g["plate_diff_sum"] / total if total > 1 else 0.0,
            # Family B
            "lead_stability": self._global_rate("lead_stability"),
            "compound_lead_rate": 0.20,
            "gold_volatility_when_ahead": 500.0,
            # Family C
            "win_rate_when_ahead_2k_15": self._global_conditional_wr("ahead_2k_15"),
            "win_rate_when_ahead_3k_20": self._global_conditional_wr("ahead_3k_20"),
            "close_time_ahead_15": self._global_avg("close_time_ahead_15"),
            "close_time_ahead_20": self._global_avg("close_time_ahead_20"),
            "gold_snowball_rate": 0.0,
            "dragon_soul_rate": g["soul_count"] / total if total > 1 else 0.15,
            "herald_tower_conv": self._global_rate("herald_tower"),
            "baron_win_rate": self._global_conditional_wr("first_baron"),
            # Family D
            "throw_rate_2k_15": 1.0 - self._global_conditional_wr("ahead_2k_15"),
            "lead_evaporation_rate": self._global_rate("lead_evaporation"),
            "baron_throw_rate": 1.0 - self._global_conditional_wr("ahead_first_baron"),
            # Family E
            "win_rate_when_behind_2k_15": self._global_conditional_wr("behind_2k_15"),
            "gold_recovery_rate": self._global_rate("gold_recovery"),
            "extend_time_behind": self._global_avg("extend_time_behind"),
            # Metadata
            "n_games": 0,
            "n_ahead_2k_15": 0,
            "n_ahead_3k_20": 0,
            "n_behind_2k_15": 0,
            "n_ahead_15": 0,
            "n_behind_15": 0,
        }

    def game_count(self, team: str) -> int:
        """Return number of recorded games for a team."""
        return len(self.team_history.get(team, deque()))

    def reset_team(self, team: str) -> None:
        """Clear history for a team (e.g., after major roster change)."""
        if team in self.team_history:
            self.team_history[team].clear()
