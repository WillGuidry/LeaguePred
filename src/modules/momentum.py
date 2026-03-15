"""
Rolling Momentum Module
-------------------------
Captures recent form — hot streaks, slumps, and performance trends.

A team's Elo is a long-term estimate, but form is real: a team on a 7-game
win streak is genuinely stronger *right now* than their Elo reflects. This
module quantifies that recency signal.

Signals tracked:
    1. Win rate at short/medium windows (5/10 games)
    2. Current streak (with diminishing returns)
    3. Elo trend (are ratings rising or falling?)
    4. Dominance trend (quality of recent wins vs quality of recent losses)

All signals are computed from games BEFORE the current prediction (no leakage).
The module outputs a signed edge score: positive = team_a has better recent form.
"""

import math
import numpy as np
from collections import deque
from typing import Dict, Optional

from src.modules.base_module import BaseModule
from src.config import (
    MOMENTUM_WINDOW_SHORT,
    MOMENTUM_WINDOW_MEDIUM,
    MOMENTUM_WINDOW_LONG,
    MOMENTUM_DECAY_HALF_LIFE,
    MOMENTUM_SIGNAL_WEIGHTS,
    MOMENTUM_STREAK_CAP,
    MOMENTUM_MIN_GAMES,
)


class MomentumTracker:
    """
    Tracks per-team recent results for momentum computation.

    Records game outcomes chronologically and computes rolling form signals
    on demand. Uses exponential decay to weight recent games more heavily.

    Usage in pipeline:
        tracker = MomentumTracker()

        for each game:
            # PREDICT phase
            signals_a = tracker.get_signals(team_a)
            signals_b = tracker.get_signals(team_b)

            # UPDATE phase (after result known)
            tracker.record_game(team_a, result=1, elo_after=1600, dominance=0.7)
            tracker.record_game(team_b, result=0, elo_after=1400, dominance=0.3)
    """

    def __init__(
        self,
        window: int = MOMENTUM_WINDOW_LONG,
        min_games: int = MOMENTUM_MIN_GAMES,
        decay_half_life: float = MOMENTUM_DECAY_HALF_LIFE,
        streak_cap: int = MOMENTUM_STREAK_CAP,
    ):
        self.window = window
        self.min_games = min_games
        self.decay_half_life = decay_half_life
        self.streak_cap = streak_cap

        # {team: deque of game records}
        self.team_history: Dict[str, deque] = {}

    def record_game(
        self,
        team: str,
        result: int,
        elo_after: float = 0.0,
        dominance: float = 0.5,
    ) -> None:
        """
        Record a completed game for momentum tracking.

        Args:
            team: Team name.
            result: 1 for win, 0 for loss.
            elo_after: Team's Elo rating after this game's update.
            dominance: Dominance score from MOV module (0-1, 0.5 = neutral).
        """
        if team not in self.team_history:
            self.team_history[team] = deque(maxlen=self.window)

        self.team_history[team].append({
            "result": int(result),
            "elo_after": float(elo_after),
            "dominance": float(dominance),
        })

    def get_signals(self, team: str) -> dict:
        """
        Compute momentum signals for a team from their recent history.

        Returns dict with:
            - win_rate_short: Win rate over short window (0-1)
            - win_rate_medium: Win rate over medium window (0-1)
            - streak: Current streak score (-1 to 1, scaled with cap)
            - elo_trend: Elo trajectory (-1 to 1)
            - dominance_trend: Recent dominance quality (-1 to 1)
            - n_games: Number of games available
        """
        history = self.team_history.get(team, deque())
        n = len(history)

        if n == 0:
            return self._default_signals()

        games = list(history)

        # Win rates at different windows
        win_rate_short = self._windowed_win_rate(games, MOMENTUM_WINDOW_SHORT)
        win_rate_medium = self._windowed_win_rate(games, MOMENTUM_WINDOW_MEDIUM)

        # Current streak
        streak = self._compute_streak(games)

        # Elo trend (slope of recent Elo trajectory)
        elo_trend = self._compute_elo_trend(games)

        # Dominance trend (decay-weighted average dominance, centered at 0.5)
        dominance_trend = self._compute_dominance_trend(games)

        return {
            "win_rate_short": win_rate_short,
            "win_rate_medium": win_rate_medium,
            "streak": streak,
            "elo_trend": elo_trend,
            "dominance_trend": dominance_trend,
            "n_games": n,
        }

    def get_confidence(self, team: str) -> float:
        """
        Confidence in momentum signals, scaling from 0 to 1.

        Below min_games, confidence is low. Reaches 1.0 at full window.
        """
        n = len(self.team_history.get(team, deque()))
        if n == 0:
            return 0.0
        if n < self.min_games:
            return 0.2 * (n / self.min_games)
        return min(1.0, 0.2 + 0.8 * (n / self.window))

    def game_count(self, team: str) -> int:
        return len(self.team_history.get(team, deque()))

    def _windowed_win_rate(self, games: list, window: int) -> float:
        """Exponential-decay-weighted win rate over the last `window` games."""
        recent = games[-window:]
        if not recent:
            return 0.5

        # Decay weights: most recent game has weight 1.0
        weights = []
        for i, _ in enumerate(recent):
            age = len(recent) - 1 - i  # 0 = most recent
            w = math.pow(0.5, age / self.decay_half_life)
            weights.append(w)

        results = [g["result"] for g in recent]
        weighted_sum = sum(r * w for r, w in zip(results, weights))
        weight_total = sum(weights)

        return weighted_sum / weight_total if weight_total > 0 else 0.5

    def _compute_streak(self, games: list) -> float:
        """
        Compute current streak as a scaled score in [-1, 1].

        A 3-game win streak → ~0.5, 6+ → ~1.0 (capped).
        Loss streaks mirror into negative values.
        """
        if not games:
            return 0.0

        # Count current streak from most recent game
        last_result = games[-1]["result"]
        streak_len = 0
        for g in reversed(games):
            if g["result"] == last_result:
                streak_len += 1
            else:
                break

        # Cap and normalize to [0, 1]
        capped = min(streak_len, self.streak_cap)
        # Diminishing returns: sqrt scaling
        scaled = math.sqrt(capped / self.streak_cap)

        # Sign: positive for win streak, negative for loss streak
        return scaled if last_result == 1 else -scaled

    def _compute_elo_trend(self, games: list) -> float:
        """
        Compute Elo trajectory as a normalized slope.

        Uses the last N games' Elo values. Positive = rising, negative = falling.
        Normalized to roughly [-1, 1] where ±200 Elo change over the window = ±1.
        """
        elos = [g["elo_after"] for g in games if g["elo_after"] > 0]
        if len(elos) < 2:
            return 0.0

        # Simple linear regression slope
        n = len(elos)
        x = np.arange(n, dtype=float)
        x_mean = x.mean()
        y_mean = np.mean(elos)

        numerator = np.sum((x - x_mean) * (np.array(elos) - y_mean))
        denominator = np.sum((x - x_mean) ** 2)

        if denominator == 0:
            return 0.0

        slope = numerator / denominator  # Elo points per game

        # Normalize: ~13 Elo/game change over window ≈ 200 total = ±1.0
        normalized = slope / 13.0
        return max(-1.0, min(1.0, normalized))

    def _compute_dominance_trend(self, games: list) -> float:
        """
        Decay-weighted average dominance, centered at 0 (neutral = 0.5 raw).

        Captures whether recent wins are dominant or scrappy.
        Returns value in [-1, 1].
        """
        if not games:
            return 0.0

        weights = []
        for i, _ in enumerate(games):
            age = len(games) - 1 - i
            w = math.pow(0.5, age / self.decay_half_life)
            weights.append(w)

        dominances = [g["dominance"] for g in games]
        weighted_sum = sum(d * w for d, w in zip(dominances, weights))
        weight_total = sum(weights)

        avg_dominance = weighted_sum / weight_total if weight_total > 0 else 0.5

        # Center at 0 and scale: 0.5 raw → 0.0, 1.0 raw → 1.0, 0.0 raw → -1.0
        return (avg_dominance - 0.5) * 2.0

    def _default_signals(self) -> dict:
        return {
            "win_rate_short": 0.5,
            "win_rate_medium": 0.5,
            "streak": 0.0,
            "elo_trend": 0.0,
            "dominance_trend": 0.0,
            "n_games": 0,
        }


class MomentumModule(BaseModule):
    """
    Rolling Momentum module for pre-match prediction.

    Computes a composite momentum edge from recent form signals.
    Positive = team_a has better recent form, negative = team_b.

    Usage:
        tracker = MomentumTracker()
        module = MomentumModule(tracker)

        edge = module.compute(team_a, team_b)
        confidence = module.confidence()
    """

    name = "rolling_momentum"

    def __init__(
        self,
        tracker: MomentumTracker,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.tracker = tracker
        self.weights = weights or MOMENTUM_SIGNAL_WEIGHTS
        self._last_confidence = 0.0
        self._last_signals_a = {}
        self._last_signals_b = {}

    def compute(self, team_a: str, team_b: str, match_context: dict = None) -> float:
        """
        Compute momentum edge for team_a vs team_b.

        Returns:
            Float where positive favors team_a, negative favors team_b.
            Scaled by confidence so low-data predictions are dampened.
        """
        signals_a = self.tracker.get_signals(team_a)
        signals_b = self.tracker.get_signals(team_b)

        self._last_signals_a = signals_a
        self._last_signals_b = signals_b

        conf_a = self.tracker.get_confidence(team_a)
        conf_b = self.tracker.get_confidence(team_b)
        self._last_confidence = min(conf_a, conf_b)

        if self._last_confidence == 0.0:
            return 0.0

        # Compute weighted edge across signals
        edge = 0.0
        for signal, weight in self.weights.items():
            val_a = signals_a.get(signal, 0.0)
            val_b = signals_b.get(signal, 0.0)

            if signal in ("win_rate_short", "win_rate_medium"):
                # Win rates are [0, 1], center at 0.5
                diff = (val_a - 0.5) - (val_b - 0.5)
            else:
                # streak, elo_trend, dominance_trend already centered at 0
                diff = val_a - val_b

            edge += weight * diff

        edge *= self._last_confidence
        return edge

    def confidence(self) -> float:
        return self._last_confidence

    def get_signal_contributions(
        self, team_a: str, team_b: str
    ) -> Dict[str, float]:
        """Break down the edge into per-signal contributions."""
        signals_a = self.tracker.get_signals(team_a)
        signals_b = self.tracker.get_signals(team_b)

        contributions = {}
        for signal, weight in self.weights.items():
            val_a = signals_a.get(signal, 0.0)
            val_b = signals_b.get(signal, 0.0)

            if signal in ("win_rate_short", "win_rate_medium"):
                diff = (val_a - 0.5) - (val_b - 0.5)
            else:
                diff = val_a - val_b

            contributions[signal] = weight * diff

        return contributions
