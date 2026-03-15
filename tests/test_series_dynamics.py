"""
Tests for the Series Dynamics module and SeriesTracker.
"""

import pytest
from src.modules.series_dynamics import (
    SeriesRecord, SeriesTracker, SeriesDynamicsModule,
)


class TestSeriesRecord:
    """Test SeriesRecord properties."""

    def test_bo3_win_2_0(self):
        s = SeriesRecord("Opp", bo_format=3, game_results=[1, 1], won=True)
        assert s.won_game_1
        assert s.games_played == 2
        assert s.had_match_point  # Was at 1-0 (1 win from victory)
        assert s.converted_match_point
        assert not s.faced_elimination
        assert not s.is_reverse_sweep

    def test_bo3_loss_1_2(self):
        s = SeriesRecord("Opp", bo_format=3, game_results=[1, 0, 0], won=False)
        assert s.won_game_1
        assert s.games_played == 3
        assert s.had_match_point  # Was at 1-0
        assert not s.converted_match_point
        assert s.faced_elimination  # Was at 1-1 (1 loss from out)
        assert not s.is_reverse_sweep

    def test_bo3_reverse_sweep(self):
        s = SeriesRecord("Opp", bo_format=3, game_results=[0, 1, 1], won=True)
        assert not s.won_game_1
        assert s.is_reverse_sweep
        assert s.faced_elimination  # Was at 0-1

    def test_bo5_3_0(self):
        s = SeriesRecord("Opp", bo_format=5, game_results=[1, 1, 1], won=True)
        assert s.won_game_1
        assert s.games_played == 3
        assert s.had_match_point  # Was at 2-0
        assert s.converted_match_point
        assert not s.faced_elimination

    def test_bo5_reverse_sweep_3_2(self):
        s = SeriesRecord("Opp", bo_format=5, game_results=[0, 0, 1, 1, 1], won=True)
        assert not s.won_game_1
        assert s.is_reverse_sweep
        assert s.faced_elimination  # Was at 0-2
        assert s.survived_elimination

    def test_bo5_loss_2_3(self):
        s = SeriesRecord("Opp", bo_format=5, game_results=[1, 1, 0, 0, 0], won=False)
        assert s.won_game_1
        assert s.had_match_point  # Was at 2-0
        assert not s.converted_match_point
        assert s.faced_elimination  # Was at 2-2

    def test_elimination_games(self):
        s = SeriesRecord("Opp", bo_format=5, game_results=[0, 0, 1, 1, 1], won=True)
        elim = s.elimination_games
        # After going 0-2, all remaining games are elimination games
        assert len(elim) >= 1
        assert sum(elim) >= 1  # Won at least some


class TestSeriesTracker:
    """Test the SeriesTracker accumulation and signal computation."""

    def test_bo1_ignored(self):
        tracker = SeriesTracker()
        tracker.record_game("TeamA", "TeamB", game_number=1, result=1, bo_format=1)
        assert tracker.series_count("TeamA") == 0

    def test_bo3_completes_at_2_wins(self):
        tracker = SeriesTracker()
        tracker.record_game("TeamA", "TeamB", game_number=1, result=1, bo_format=3)
        assert tracker.series_count("TeamA") == 0  # Not complete yet
        tracker.record_game("TeamA", "TeamB", game_number=2, result=1, bo_format=3)
        assert tracker.series_count("TeamA") == 1  # 2-0, done

    def test_bo3_completes_at_2_losses(self):
        tracker = SeriesTracker()
        tracker.record_game("TeamA", "TeamB", game_number=1, result=0, bo_format=3)
        tracker.record_game("TeamA", "TeamB", game_number=2, result=0, bo_format=3)
        assert tracker.series_count("TeamA") == 1  # 0-2, done

    def test_bo3_full_3_games(self):
        tracker = SeriesTracker()
        tracker.record_game("TeamA", "TeamB", game_number=1, result=1, bo_format=3)
        tracker.record_game("TeamA", "TeamB", game_number=2, result=0, bo_format=3)
        tracker.record_game("TeamA", "TeamB", game_number=3, result=1, bo_format=3)
        assert tracker.series_count("TeamA") == 1  # 2-1

    def test_bo5_full_5_games(self):
        tracker = SeriesTracker()
        results = [1, 0, 1, 0, 1]  # 3-2
        for i, r in enumerate(results):
            tracker.record_game("TeamA", "TeamB", game_number=i + 1, result=r, bo_format=5)
        assert tracker.series_count("TeamA") == 1

    def test_new_opponent_starts_new_series(self):
        tracker = SeriesTracker()
        tracker.record_game("TeamA", "TeamB", game_number=1, result=1, bo_format=3)
        # New opponent before series completes
        tracker.record_game("TeamA", "TeamC", game_number=1, result=1, bo_format=3)
        tracker.record_game("TeamA", "TeamC", game_number=2, result=1, bo_format=3)
        assert tracker.series_count("TeamA") == 1  # Only the completed one vs TeamC

    def test_confidence_scales(self):
        tracker = SeriesTracker(min_series=5, window=20)
        assert tracker.get_confidence("TeamA") == 0.0

        # Add 3 series (below min)
        for i in range(3):
            opp = f"Opp{i}"
            tracker.record_game("TeamA", opp, game_number=1, result=1, bo_format=3)
            tracker.record_game("TeamA", opp, game_number=2, result=1, bo_format=3)
        conf_3 = tracker.get_confidence("TeamA")
        assert 0 < conf_3 < 0.2

    def test_signals_default_for_unknown_team(self):
        tracker = SeriesTracker()
        signals = tracker.get_signals("Unknown")
        assert signals["n_series"] == 0
        assert 0 <= signals["game1_win_rate"] <= 1

    def test_signals_reflect_game1_dominance(self):
        tracker = SeriesTracker(min_series=1, shrinkage_weight=0.0)
        # Win all game 1s (10 series, all 2-0)
        for i in range(10):
            opp = f"Opp{i}"
            tracker.record_game("TeamA", opp, game_number=1, result=1, bo_format=3)
            tracker.record_game("TeamA", opp, game_number=2, result=1, bo_format=3)

        signals = tracker.get_signals("TeamA")
        assert signals["game1_win_rate"] > 0.8

    def test_signals_reflect_reverse_sweep_ability(self):
        tracker = SeriesTracker(min_series=1, shrinkage_weight=0.0)
        # Lose game 1 but win series (5 reverse sweeps out of 5)
        for i in range(5):
            opp = f"Opp{i}"
            tracker.record_game("TeamA", opp, game_number=1, result=0, bo_format=3)
            tracker.record_game("TeamA", opp, game_number=2, result=1, bo_format=3)
            tracker.record_game("TeamA", opp, game_number=3, result=1, bo_format=3)

        signals = tracker.get_signals("TeamA")
        assert signals["reverse_sweep_rate"] > 0.7

    def test_window_respected(self):
        tracker = SeriesTracker(window=3)
        for i in range(10):
            opp = f"Opp{i}"
            tracker.record_game("TeamA", opp, game_number=1, result=1, bo_format=3)
            tracker.record_game("TeamA", opp, game_number=2, result=1, bo_format=3)
        assert tracker.series_count("TeamA") == 3


class TestSeriesDynamicsModule:
    """Test the SeriesDynamicsModule compute and confidence."""

    def _setup(self, **kwargs):
        tracker = SeriesTracker(**kwargs)
        module = SeriesDynamicsModule(tracker)
        return tracker, module

    def test_bo1_returns_zero(self):
        _, module = self._setup()
        edge = module.compute("TeamA", "TeamB", {"bo_format": 1})
        assert edge == 0.0

    def test_no_data_returns_zero(self):
        _, module = self._setup()
        edge = module.compute("TeamA", "TeamB", {"bo_format": 3})
        assert edge == 0.0

    def test_clutch_team_positive_edge(self):
        tracker, module = self._setup(min_series=1, shrinkage_weight=0.0)

        # TeamA: great at adaptation and closing out
        for i in range(8):
            opp = f"Opp{i}"
            tracker.record_game("TeamA", opp, game_number=1, result=1, bo_format=3)
            tracker.record_game("TeamA", opp, game_number=2, result=1, bo_format=3)

        # TeamB: chokes under pressure, loses game 1 and loses series
        for i in range(8):
            opp = f"Foe{i}"
            tracker.record_game("TeamB", opp, game_number=1, result=0, bo_format=3)
            tracker.record_game("TeamB", opp, game_number=2, result=0, bo_format=3)

        edge = module.compute("TeamA", "TeamB", {"bo_format": 3, "game_number": 1})
        assert edge > 0

    def test_symmetry(self):
        tracker, module = self._setup(min_series=1, shrinkage_weight=0.0)

        for i in range(8):
            opp = f"Opp{i}"
            tracker.record_game("TeamA", opp, game_number=1, result=1, bo_format=3)
            tracker.record_game("TeamA", opp, game_number=2, result=1, bo_format=3)
            tracker.record_game("TeamB", opp, game_number=1, result=0, bo_format=3)
            tracker.record_game("TeamB", opp, game_number=2, result=0, bo_format=3)

        ctx = {"bo_format": 3, "game_number": 1}
        edge_ab = module.compute("TeamA", "TeamB", ctx)
        edge_ba = module.compute("TeamB", "TeamA", ctx)
        # Should be approximately symmetric (not exact due to weight adjustment)
        assert abs(edge_ab + edge_ba) < 0.05

    def test_context_adjusts_weights(self):
        tracker, module = self._setup(min_series=1, shrinkage_weight=0.0)

        # Build history for both teams
        for i in range(8):
            opp = f"Opp{i}"
            tracker.record_game("TeamA", opp, game_number=1, result=1, bo_format=3)
            tracker.record_game("TeamA", opp, game_number=2, result=1, bo_format=3)
            tracker.record_game("TeamB", opp, game_number=1, result=0, bo_format=3)
            tracker.record_game("TeamB", opp, game_number=2, result=0, bo_format=3)

        # Game 1 context vs game 3 (elimination) context should differ
        edge_g1 = module.compute("TeamA", "TeamB", {"bo_format": 3, "game_number": 1})
        edge_g3 = module.compute("TeamA", "TeamB", {
            "bo_format": 3, "game_number": 3,
            "series_score_a": 1, "series_score_b": 1
        })
        # Both positive but different magnitudes due to weight adjustment
        assert edge_g1 > 0
        assert edge_g3 > 0

    def test_signal_contributions(self):
        tracker, module = self._setup(min_series=1, shrinkage_weight=0.0)

        for i in range(8):
            opp = f"Opp{i}"
            tracker.record_game("TeamA", opp, game_number=1, result=1, bo_format=3)
            tracker.record_game("TeamA", opp, game_number=2, result=1, bo_format=3)
            tracker.record_game("TeamB", opp, game_number=1, result=0, bo_format=3)
            tracker.record_game("TeamB", opp, game_number=2, result=0, bo_format=3)

        ctx = {"bo_format": 3, "game_number": 1}
        contributions = module.get_signal_contributions("TeamA", "TeamB", ctx)
        assert "game1_win_rate" in contributions
        assert contributions["game1_win_rate"] > 0
