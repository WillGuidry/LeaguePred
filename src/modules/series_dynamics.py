"""
Series Dynamics Module
------------------------
Captures how teams adapt and perform under pressure in Bo3/Bo5 series.

Some teams are clutch: they elevate in elimination games, adapt after losses,
and close out series when up. Others choke under pressure or can't adjust.
This module quantifies those tendencies.

Signals tracked:
    1. Game 1 win rate — momentum setters vs slow starters
    2. Adaptation rate — how win rate changes from game 1 to later games
    3. Elimination win rate — performance in must-win games (down 0-1, 1-2)
    4. Reverse sweep rate — ability to recover from losing series start
    5. Closeout rate — converting match point opportunities

All features are computed from completed series BEFORE the current prediction.
The module outputs a signed edge score for a given series context.

Data model:
    A "series" is identified by (team_a, team_b, date) grouping games with
    the same gameid prefix or consecutive game_numbers. The pipeline feeds
    us individual game results with series context.
"""

import math
import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple

from src.modules.base_module import BaseModule
from src.config import (
    SERIES_MIN_SERIES,
    SERIES_WINDOW,
    SERIES_SIGNAL_WEIGHTS,
    SERIES_SHRINKAGE_WEIGHT,
)


class SeriesRecord:
    """A completed best-of-N series."""

    __slots__ = ["opponent", "format", "game_results", "won"]

    def __init__(self, opponent: str, bo_format: int, game_results: List[int], won: bool):
        """
        Args:
            opponent: Opposing team name.
            format: Series format (3 for Bo3, 5 for Bo5).
            game_results: List of 0/1 for each game from this team's perspective.
            won: Whether this team won the series.
        """
        self.opponent = opponent
        self.format = bo_format
        self.game_results = game_results
        self.won = won

    @property
    def games_played(self) -> int:
        return len(self.game_results)

    @property
    def won_game_1(self) -> bool:
        return self.game_results[0] == 1 if self.game_results else False

    @property
    def had_match_point(self) -> bool:
        """Did this team reach match point (1 win from series victory)?"""
        wins_needed = (self.format + 1) // 2
        # Check if at any point (before last game) team was 1 win away
        running = 0
        for i, r in enumerate(self.game_results[:-1]):
            running += r
            if running == wins_needed - 1:
                return True
        return False

    @property
    def converted_match_point(self) -> bool:
        """Won the series after reaching match point."""
        return self.had_match_point and self.won

    @property
    def faced_elimination(self) -> bool:
        """Was this team ever 1 loss from elimination?"""
        losses_to_lose = (self.format + 1) // 2
        running_losses = 0
        for r in self.game_results:
            if r == 0:
                running_losses += 1
            if running_losses == losses_to_lose - 1:
                return True
        return False

    @property
    def survived_elimination(self) -> bool:
        """Won a game while facing elimination."""
        if not self.faced_elimination:
            return False
        losses_to_lose = (self.format + 1) // 2
        running_losses = 0
        for r in self.game_results:
            if r == 0:
                running_losses += 1
            if running_losses == losses_to_lose - 1:
                # Next game(s) are elimination games
                # If we're still here, team survived
                break
        # Check if any wins came after reaching elimination threshold
        return self.won

    @property
    def is_reverse_sweep(self) -> bool:
        """Won the series after losing game 1."""
        return not self.won_game_1 and self.won

    @property
    def elimination_games(self) -> List[int]:
        """Results of games where this team faced elimination."""
        losses_to_lose = (self.format + 1) // 2
        elim_results = []
        running_losses = 0
        in_elimination = False
        for r in self.game_results:
            if in_elimination:
                elim_results.append(r)
            if r == 0:
                running_losses += 1
            if running_losses >= losses_to_lose - 1:
                in_elimination = True
                if r == 0 and running_losses == losses_to_lose - 1:
                    # This loss brought us to elimination — don't count it
                    pass
        return elim_results


class SeriesTracker:
    """
    Tracks per-team series history for dynamics computation.

    Accumulates completed series and builds game-within-series records
    from individual game results fed by the pipeline.

    Usage:
        tracker = SeriesTracker()

        # As games come in from the pipeline:
        tracker.record_game(team="T1", opponent="Gen.G",
                           game_number=1, result=1, bo_format=3)
        tracker.record_game(team="T1", opponent="Gen.G",
                           game_number=2, result=1, bo_format=3)
        # Series auto-completes when enough wins are accumulated

        # Pre-match:
        signals = tracker.get_signals("T1")
    """

    def __init__(
        self,
        window: int = SERIES_WINDOW,
        min_series: int = SERIES_MIN_SERIES,
        shrinkage_weight: float = SERIES_SHRINKAGE_WEIGHT,
    ):
        self.window = window
        self.min_series = min_series
        self.shrinkage_weight = shrinkage_weight

        # {team: deque of SeriesRecord}
        self.completed_series: Dict[str, deque] = {}

        # {team: dict tracking in-progress series}
        self._active_series: Dict[str, dict] = {}

        # Global priors
        self._global = {
            "total_series": 0,
            "game1_wins": 0,
            "adaptations": 0,       # Won game 2+ after losing game 1
            "adaptation_opps": 0,   # Lost game 1
            "elim_wins": 0,
            "elim_games": 0,
            "reverse_sweeps": 0,
            "reverse_sweep_opps": 0,
            "closeouts": 0,
            "closeout_opps": 0,
        }

    def record_game(
        self,
        team: str,
        opponent: str,
        game_number: int,
        result: int,
        bo_format: int = 1,
    ) -> None:
        """
        Record an individual game result within a series.

        For Bo1 games, the series auto-completes immediately.
        For Bo3/Bo5, the tracker accumulates games until the series ends.

        Args:
            team: Team name (from this team's perspective).
            opponent: Opposing team.
            game_number: Game number within the series (1-indexed).
            result: 1 for win, 0 for loss.
            bo_format: Series format (1, 3, or 5).
        """
        if bo_format <= 1:
            # Bo1: no series dynamics, skip
            return

        key = team

        # Start new series if needed
        if key not in self._active_series:
            self._active_series[key] = {
                "opponent": opponent,
                "format": bo_format,
                "results": [],
            }

        active = self._active_series[key]

        # Detect new series: different opponent or game_number reset
        if active["opponent"] != opponent or (
            game_number == 1 and len(active["results"]) > 0
        ):
            # Previous series was incomplete — discard it
            self._active_series[key] = {
                "opponent": opponent,
                "format": bo_format,
                "results": [],
            }
            active = self._active_series[key]

        active["results"].append(int(result))

        # Check if series is complete
        wins_needed = (bo_format + 1) // 2
        wins = sum(active["results"])
        losses = len(active["results"]) - wins

        if wins >= wins_needed or losses >= wins_needed:
            # Series complete
            series = SeriesRecord(
                opponent=opponent,
                bo_format=bo_format,
                game_results=list(active["results"]),
                won=(wins >= wins_needed),
            )
            self._store_series(team, series)
            del self._active_series[key]

    def _store_series(self, team: str, series: SeriesRecord) -> None:
        """Store a completed series and update global priors."""
        if team not in self.completed_series:
            self.completed_series[team] = deque(maxlen=self.window)

        self.completed_series[team].append(series)
        self._update_global(series)

    def _update_global(self, series: SeriesRecord) -> None:
        """Update global prior statistics."""
        g = self._global
        g["total_series"] += 1

        if series.won_game_1:
            g["game1_wins"] += 1

        if not series.won_game_1:
            g["adaptation_opps"] += 1
            if series.won:
                g["adaptations"] += 1

        elim_games = series.elimination_games
        if elim_games:
            g["elim_games"] += len(elim_games)
            g["elim_wins"] += sum(elim_games)

        if not series.won_game_1:
            g["reverse_sweep_opps"] += 1
            if series.is_reverse_sweep:
                g["reverse_sweeps"] += 1

        if series.had_match_point:
            g["closeout_opps"] += 1
            if series.converted_match_point:
                g["closeouts"] += 1

    def get_signals(self, team: str) -> dict:
        """
        Compute series dynamics signals for a team.

        Returns:
            Dict with signal values, all in [0, 1] range.
        """
        history = list(self.completed_series.get(team, deque()))
        n = len(history)

        if n == 0:
            return self._default_signals()

        # Game 1 win rate
        game1_wr = self._shrunk_rate(
            [s.won_game_1 for s in history],
            self._global_rate("game1"),
        )

        # Adaptation rate: win rate in games after game 1
        later_results = []
        for s in history:
            later_results.extend(s.game_results[1:])
        if later_results:
            adaptation_rate = self._shrunk_rate(
                later_results,
                self._global_rate("adaptation"),
            )
        else:
            adaptation_rate = self._global_rate("adaptation")

        # Elimination win rate
        elim_results = []
        for s in history:
            elim_results.extend(s.elimination_games)
        if elim_results:
            elimination_wr = self._shrunk_rate(
                elim_results,
                self._global_rate("elimination"),
            )
        else:
            elimination_wr = self._global_rate("elimination")

        # Reverse sweep rate (won after losing game 1)
        lost_game1 = [s for s in history if not s.won_game_1]
        if lost_game1:
            reverse_sweep_rate = self._shrunk_rate(
                [s.won for s in lost_game1],
                self._global_rate("reverse_sweep"),
            )
        else:
            reverse_sweep_rate = self._global_rate("reverse_sweep")

        # Closeout rate (won after reaching match point)
        had_mp = [s for s in history if s.had_match_point]
        if had_mp:
            closeout_rate = self._shrunk_rate(
                [s.converted_match_point for s in had_mp],
                self._global_rate("closeout"),
            )
        else:
            closeout_rate = self._global_rate("closeout")

        return {
            "game1_win_rate": game1_wr,
            "adaptation_rate": adaptation_rate,
            "elimination_win_rate": elimination_wr,
            "reverse_sweep_rate": reverse_sweep_rate,
            "closeout_rate": closeout_rate,
            "n_series": n,
        }

    def get_confidence(self, team: str) -> float:
        """Confidence based on number of completed series."""
        n = len(self.completed_series.get(team, deque()))
        if n == 0:
            return 0.0
        if n < self.min_series:
            return 0.2 * (n / self.min_series)
        return min(1.0, 0.2 + 0.8 * (n / self.window))

    def series_count(self, team: str) -> int:
        return len(self.completed_series.get(team, deque()))

    def _shrunk_rate(self, outcomes: list, prior: float) -> float:
        """Bayesian-shrunk rate toward a global prior."""
        n = len(outcomes)
        if n == 0:
            return prior

        observed = sum(1 for x in outcomes if x) / n
        shrunk = (observed * n + prior * self.shrinkage_weight) / (
            n + self.shrinkage_weight
        )
        return shrunk

    def _global_rate(self, stat: str) -> float:
        """Get global prior rate for a statistic."""
        g = self._global
        rates = {
            "game1": g["game1_wins"] / max(g["total_series"], 1),
            "adaptation": g["adaptations"] / max(g["adaptation_opps"], 1) if g["adaptation_opps"] > 0 else 0.45,
            "elimination": g["elim_wins"] / max(g["elim_games"], 1) if g["elim_games"] > 0 else 0.45,
            "reverse_sweep": g["reverse_sweeps"] / max(g["reverse_sweep_opps"], 1) if g["reverse_sweep_opps"] > 0 else 0.35,
            "closeout": g["closeouts"] / max(g["closeout_opps"], 1) if g["closeout_opps"] > 0 else 0.65,
        }
        # Sensible defaults when no global data yet
        defaults = {
            "game1": 0.50,
            "adaptation": 0.45,
            "elimination": 0.45,
            "reverse_sweep": 0.35,
            "closeout": 0.65,
        }
        rate = rates.get(stat, 0.5)
        if g["total_series"] == 0:
            return defaults.get(stat, 0.5)
        return rate

    def _default_signals(self) -> dict:
        return {
            "game1_win_rate": self._global_rate("game1"),
            "adaptation_rate": self._global_rate("adaptation"),
            "elimination_win_rate": self._global_rate("elimination"),
            "reverse_sweep_rate": self._global_rate("reverse_sweep"),
            "closeout_rate": self._global_rate("closeout"),
            "n_series": 0,
        }


class SeriesDynamicsModule(BaseModule):
    """
    Series Dynamics module for pre-match prediction.

    Computes an edge score based on teams' historical Bo3/Bo5 performance.
    Particularly useful for playoffs where series play differs from regular season.

    The edge captures:
        - Does team_a typically win game 1 and set the tone?
        - Can team_b adapt and improve within a series?
        - Who performs better under elimination pressure?
        - Who closes out series more efficiently?

    Usage:
        tracker = SeriesTracker()
        module = SeriesDynamicsModule(tracker)

        edge = module.compute(team_a, team_b, {"bo_format": 3, "game_number": 1})
    """

    name = "series_dynamics"

    def __init__(
        self,
        tracker: SeriesTracker,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.tracker = tracker
        self.weights = weights or SERIES_SIGNAL_WEIGHTS
        self._last_confidence = 0.0
        self._last_signals_a = {}
        self._last_signals_b = {}

    def compute(self, team_a: str, team_b: str, match_context: dict = None) -> float:
        """
        Compute series dynamics edge.

        Args:
            team_a, team_b: Team names.
            match_context: Optional dict with:
                - bo_format: Series format (1, 3, 5). If 1 or absent, returns 0.
                - game_number: Current game in series (1, 2, 3...).
                - series_score_a: Team A's current series wins.
                - series_score_b: Team B's current series wins.

        Returns:
            Float where positive favors team_a. Scaled by confidence.
        """
        ctx = match_context or {}
        bo_format = ctx.get("bo_format", 1)

        # No series dynamics for Bo1
        if bo_format <= 1:
            self._last_confidence = 0.0
            return 0.0

        signals_a = self.tracker.get_signals(team_a)
        signals_b = self.tracker.get_signals(team_b)

        self._last_signals_a = signals_a
        self._last_signals_b = signals_b

        conf_a = self.tracker.get_confidence(team_a)
        conf_b = self.tracker.get_confidence(team_b)
        self._last_confidence = min(conf_a, conf_b)

        if self._last_confidence == 0.0:
            return 0.0

        # Adjust signal weights based on game context
        effective_weights = self._context_adjusted_weights(ctx)

        # Compute edge
        edge = 0.0
        for signal, weight in effective_weights.items():
            val_a = signals_a.get(signal, 0.5)
            val_b = signals_b.get(signal, 0.5)
            # All signals are rates in [0, 1], center at 0.5
            diff = (val_a - 0.5) - (val_b - 0.5)
            edge += weight * diff

        edge *= self._last_confidence
        return edge

    def confidence(self) -> float:
        return self._last_confidence

    def _context_adjusted_weights(self, ctx: dict) -> Dict[str, float]:
        """
        Adjust signal weights based on the current series state.

        In game 1: game1_win_rate matters most.
        In elimination game: elimination_win_rate matters most.
        On match point: closeout_rate matters most.
        """
        weights = dict(self.weights)
        game_number = ctx.get("game_number", 1)
        series_score_a = ctx.get("series_score_a", 0)
        series_score_b = ctx.get("series_score_b", 0)
        bo_format = ctx.get("bo_format", 3)
        wins_needed = (bo_format + 1) // 2

        if game_number == 1:
            # Game 1: boost game1 weight, reduce pressure signals
            weights["game1_win_rate"] *= 1.5
            weights["elimination_win_rate"] *= 0.5
            weights["closeout_rate"] *= 0.5
        else:
            # Later games: adaptation matters more
            weights["adaptation_rate"] *= 1.3
            weights["game1_win_rate"] *= 0.7

        # Check for elimination scenario
        if series_score_a == wins_needed - 1 and series_score_b < wins_needed - 1:
            # Team A on match point
            weights["closeout_rate"] *= 1.5
        elif series_score_b == wins_needed - 1 and series_score_a < wins_needed - 1:
            # Team B on match point (team A faces elimination)
            weights["elimination_win_rate"] *= 1.5
            weights["reverse_sweep_rate"] *= 1.3

        # Normalize weights to sum to 1
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}

        return weights

    def get_signal_contributions(
        self, team_a: str, team_b: str, match_context: dict = None
    ) -> Dict[str, float]:
        """Break down edge into per-signal contributions."""
        ctx = match_context or {}
        effective_weights = self._context_adjusted_weights(ctx)

        signals_a = self.tracker.get_signals(team_a)
        signals_b = self.tracker.get_signals(team_b)

        contributions = {}
        for signal, weight in effective_weights.items():
            val_a = signals_a.get(signal, 0.5)
            val_b = signals_b.get(signal, 0.5)
            diff = (val_a - 0.5) - (val_b - 0.5)
            contributions[signal] = weight * diff

        return contributions
