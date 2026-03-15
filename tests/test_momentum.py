"""
Tests for the Rolling Momentum module and MomentumTracker.
"""

import pytest
import math
from src.modules.momentum import MomentumTracker, MomentumModule


class TestMomentumTracker:
    """Test the MomentumTracker rolling signal computation."""

    def test_empty_team_returns_defaults(self):
        tracker = MomentumTracker()
        signals = tracker.get_signals("Unknown")
        assert signals["n_games"] == 0
        assert signals["win_rate_short"] == 0.5
        assert signals["streak"] == 0.0

    def test_confidence_zero_no_games(self):
        tracker = MomentumTracker()
        assert tracker.get_confidence("Unknown") == 0.0

    def test_confidence_scales_with_games(self):
        tracker = MomentumTracker(min_games=5, window=15)
        for i in range(3):
            tracker.record_game("TeamA", result=1)
        conf_3 = tracker.get_confidence("TeamA")
        assert 0 < conf_3 < 0.2

        for i in range(7):
            tracker.record_game("TeamA", result=1)
        conf_10 = tracker.get_confidence("TeamA")
        assert conf_10 > conf_3

    def test_win_streak_positive(self):
        tracker = MomentumTracker(min_games=1)
        for _ in range(5):
            tracker.record_game("TeamA", result=1)
        signals = tracker.get_signals("TeamA")
        assert signals["streak"] > 0
        assert signals["win_rate_short"] > 0.5

    def test_loss_streak_negative(self):
        tracker = MomentumTracker(min_games=1)
        for _ in range(5):
            tracker.record_game("TeamA", result=0)
        signals = tracker.get_signals("TeamA")
        assert signals["streak"] < 0
        assert signals["win_rate_short"] < 0.5

    def test_streak_capped(self):
        tracker = MomentumTracker(min_games=1, streak_cap=6)
        for _ in range(10):
            tracker.record_game("TeamA", result=1)
        signals = tracker.get_signals("TeamA")
        # Streak should be capped at 1.0 (sqrt(6/6))
        assert signals["streak"] <= 1.0

    def test_recent_games_weighted_more(self):
        """Exponential decay should weight recent games more heavily."""
        tracker = MomentumTracker(min_games=1, window=10, decay_half_life=2.0)

        # 5 losses then 3 wins
        for _ in range(5):
            tracker.record_game("TeamA", result=0)
        for _ in range(3):
            tracker.record_game("TeamA", result=1)

        signals = tracker.get_signals("TeamA")
        # Despite 5L 3W overall, recent wins should push short win rate above 0.5
        assert signals["win_rate_short"] > 0.5

    def test_elo_trend_rising(self):
        tracker = MomentumTracker(min_games=1)
        # Simulate rising Elo
        for i in range(10):
            tracker.record_game("TeamA", result=1, elo_after=1500 + i * 20)
        signals = tracker.get_signals("TeamA")
        assert signals["elo_trend"] > 0

    def test_elo_trend_falling(self):
        tracker = MomentumTracker(min_games=1)
        for i in range(10):
            tracker.record_game("TeamA", result=0, elo_after=1500 - i * 20)
        signals = tracker.get_signals("TeamA")
        assert signals["elo_trend"] < 0

    def test_dominance_trend(self):
        tracker = MomentumTracker(min_games=1)
        # Record dominant wins (dominance > 0.5)
        for _ in range(5):
            tracker.record_game("TeamA", result=1, dominance=0.8)
        signals = tracker.get_signals("TeamA")
        assert signals["dominance_trend"] > 0

        # Record weak performances
        tracker2 = MomentumTracker(min_games=1)
        for _ in range(5):
            tracker2.record_game("TeamB", result=0, dominance=0.2)
        signals_b = tracker2.get_signals("TeamB")
        assert signals_b["dominance_trend"] < 0

    def test_window_respected(self):
        tracker = MomentumTracker(window=5, min_games=1)
        for _ in range(10):
            tracker.record_game("TeamA", result=1)
        assert tracker.game_count("TeamA") == 5

    def test_game_count(self):
        tracker = MomentumTracker(window=20)
        assert tracker.game_count("TeamA") == 0
        tracker.record_game("TeamA", result=1)
        assert tracker.game_count("TeamA") == 1


class TestMomentumModule:
    """Test the MomentumModule compute and confidence methods."""

    def _setup(self, **kwargs):
        tracker = MomentumTracker(**kwargs)
        module = MomentumModule(tracker)
        return tracker, module

    def test_no_data_returns_zero(self):
        _, module = self._setup()
        edge = module.compute("TeamA", "TeamB")
        assert edge == 0.0
        assert module.confidence() == 0.0

    def test_hot_team_positive_edge(self):
        tracker, module = self._setup(min_games=1)
        # TeamA on a hot streak
        for _ in range(8):
            tracker.record_game("TeamA", result=1, elo_after=1600, dominance=0.7)
            tracker.record_game("TeamB", result=0, elo_after=1400, dominance=0.3)

        edge = module.compute("TeamA", "TeamB")
        assert edge > 0

    def test_symmetry(self):
        tracker, module = self._setup(min_games=1)
        for _ in range(8):
            tracker.record_game("TeamA", result=1, elo_after=1600)
            tracker.record_game("TeamB", result=0, elo_after=1400)

        edge_ab = module.compute("TeamA", "TeamB")
        edge_ba = module.compute("TeamB", "TeamA")
        assert abs(edge_ab + edge_ba) < 0.01

    def test_equal_teams_near_zero(self):
        tracker, module = self._setup(min_games=1)
        for _ in range(10):
            tracker.record_game("TeamA", result=1, elo_after=1500)
            tracker.record_game("TeamB", result=1, elo_after=1500)

        edge = module.compute("TeamA", "TeamB")
        assert abs(edge) < 0.05

    def test_confidence_dampens_edge(self):
        tracker, module = self._setup(min_games=5, window=15)

        # Only 2 games (below min_games=5)
        for _ in range(2):
            tracker.record_game("TeamA", result=1, elo_after=1600)
            tracker.record_game("TeamB", result=0, elo_after=1400)
        edge_low = module.compute("TeamA", "TeamB")

        # Add more games
        for _ in range(13):
            tracker.record_game("TeamA", result=1, elo_after=1700)
            tracker.record_game("TeamB", result=0, elo_after=1300)
        edge_high = module.compute("TeamA", "TeamB")

        assert abs(edge_high) > abs(edge_low)

    def test_signal_contributions(self):
        tracker, module = self._setup(min_games=1)
        for _ in range(10):
            tracker.record_game("TeamA", result=1, elo_after=1600, dominance=0.8)
            tracker.record_game("TeamB", result=0, elo_after=1400, dominance=0.2)

        contributions = module.get_signal_contributions("TeamA", "TeamB")
        assert "win_rate_short" in contributions
        assert "streak" in contributions
        assert contributions["win_rate_short"] > 0
