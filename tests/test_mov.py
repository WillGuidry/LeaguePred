"""
Tests for the Margin of Victory module.
"""

import pytest
from src.modules.margin_of_victory import (
    compute_dominance_score,
    compute_margin_multiplier,
    extract_mov_from_row,
    _normalize_gold_diff_15,
    _normalize_game_duration,
    _normalize_kill_diff,
    _normalize_objective_diff,
    _normalize_gold_diff_end,
)
from src.config import MARGIN_MULTIPLIER_MIN, MARGIN_MULTIPLIER_MAX


class TestNormalization:
    """Test individual normalization functions."""

    def test_gold_diff_15_zero(self):
        """Zero gold diff → 0.5 (neutral)."""
        assert abs(_normalize_gold_diff_15(0.0) - 0.5) < 0.01

    def test_gold_diff_15_positive(self):
        """Positive gold diff → above 0.5."""
        assert _normalize_gold_diff_15(2000) > 0.5

    def test_gold_diff_15_negative(self):
        """Negative gold diff → below 0.5."""
        assert _normalize_gold_diff_15(-2000) < 0.5

    def test_game_duration_short(self):
        """Short game (20 min = 1200s) → high dominance."""
        assert _normalize_game_duration(1200) > 0.5

    def test_game_duration_long(self):
        """Long game (45 min = 2700s) → low dominance."""
        assert _normalize_game_duration(2700) < 0.5

    def test_game_duration_baseline(self):
        """25 min game → 1.0 (maximum short-game credit)."""
        assert abs(_normalize_game_duration(1500) - 1.0) < 0.01

    def test_kill_diff_zero(self):
        """Zero kill diff → 0.5."""
        assert abs(_normalize_kill_diff(0) - 0.5) < 0.01

    def test_objective_diff_positive(self):
        """Positive objective diff → above 0.5."""
        assert _normalize_objective_diff(3, 1, 4) > 0.5

    def test_gold_diff_end_zero(self):
        """Zero final gold diff → 0.5."""
        assert abs(_normalize_gold_diff_end(0) - 0.5) < 0.01


class TestDominanceScore:
    """Test the composite dominance score."""

    def test_neutral_game(self):
        """A perfectly neutral game should score ~0.5 + game_duration component."""
        score = compute_dominance_score(
            gold_diff_15=0, gamelength=2250,  # 37.5 min → 0.5 duration
            kills_a=10, kills_b=10,
            dragons_a=2, dragons_b=2,
            barons_a=0, barons_b=0,
            towers_a=5, towers_b=5,
            totalgold_a=50000, totalgold_b=50000,
        )
        # All sigmoid components at 0.5, duration at 0.5 → score ≈ 0.5
        assert 0.4 < score < 0.6

    def test_dominant_win(self):
        """A stomp should score well above 0.5."""
        score = compute_dominance_score(
            gold_diff_15=3000, gamelength=1500,  # 25 min
            kills_a=20, kills_b=5,
            dragons_a=4, dragons_b=0,
            barons_a=1, barons_b=0,
            towers_a=11, towers_b=2,
            totalgold_a=65000, totalgold_b=45000,
        )
        assert score > 0.7


class TestMarginMultiplier:
    """Test the full margin multiplier computation."""

    def test_multiplier_bounds(self):
        """Multiplier should always be within configured bounds."""
        for gd15 in [-5000, -1000, 0, 1000, 5000]:
            for gl in [1200, 1800, 2700]:
                m = compute_margin_multiplier(
                    gold_diff_15=gd15, gamelength=gl,
                    kills_a=15, kills_b=10,
                    dragons_a=3, dragons_b=1,
                    barons_a=1, barons_b=0,
                    towers_a=8, towers_b=3,
                    totalgold_a=55000, totalgold_b=45000,
                    winner_is_a=True,
                )
                assert MARGIN_MULTIPLIER_MIN <= m <= MARGIN_MULTIPLIER_MAX

    def test_dominant_win_higher_multiplier(self):
        """A dominant win should produce a higher multiplier than a close win."""
        dominant = compute_margin_multiplier(
            gold_diff_15=3000, gamelength=1500,
            kills_a=20, kills_b=5,
            dragons_a=4, dragons_b=0,
            barons_a=1, barons_b=0,
            towers_a=11, towers_b=2,
            totalgold_a=65000, totalgold_b=45000,
        )
        close = compute_margin_multiplier(
            gold_diff_15=200, gamelength=2400,
            kills_a=12, kills_b=11,
            dragons_a=2, dragons_b=2,
            barons_a=1, barons_b=1,
            towers_a=7, towers_b=6,
            totalgold_a=52000, totalgold_b=51000,
        )
        assert dominant > close

    def test_winner_orientation(self):
        """When winner_is_a=False, stats should be oriented from B's perspective."""
        # Team B won with team A having the gold lead at 15
        m = compute_margin_multiplier(
            gold_diff_15=2000,  # Team A was ahead at 15
            gamelength=2400,
            kills_a=10, kills_b=15,  # But team B got more kills
            dragons_a=1, dragons_b=3,
            barons_a=0, barons_b=1,
            towers_a=3, towers_b=8,
            totalgold_a=45000, totalgold_b=55000,
            winner_is_a=False,
        )
        # Should still be a valid multiplier
        assert MARGIN_MULTIPLIER_MIN <= m <= MARGIN_MULTIPLIER_MAX


class TestExtractMov:
    """Test the row-extraction integration function."""

    def test_with_valid_row(self):
        row = {
            "team_a": "T1", "team_b": "Gen.G",
            "a_golddiffat15": 1500, "gamelength": 1800,
            "a_kills": 15, "b_kills": 8,
            "a_dragons": 3, "b_dragons": 1,
            "a_barons": 1, "b_barons": 0,
            "a_towers": 9, "b_towers": 3,
            "a_totalgold": 58000, "b_totalgold": 46000,
        }
        m = extract_mov_from_row(row, winner="T1")
        assert MARGIN_MULTIPLIER_MIN <= m <= MARGIN_MULTIPLIER_MAX

    def test_with_missing_data(self):
        """Missing data should return neutral multiplier."""
        row = {"team_a": "T1", "team_b": "Gen.G"}
        m = extract_mov_from_row(row, winner="T1")
        assert m == 1.0

    def test_with_nan_values(self):
        """NaN values should be handled gracefully."""
        row = {
            "team_a": "T1", "team_b": "Gen.G",
            "a_golddiffat15": float("nan"),
            "gamelength": 1800,
            "a_kills": 10, "b_kills": 8,
            "a_dragons": 2, "b_dragons": 2,
            "a_barons": 0, "b_barons": 0,
            "a_towers": 6, "b_towers": 5,
            "a_totalgold": 50000, "b_totalgold": 48000,
        }
        m = extract_mov_from_row(row, winner="T1")
        assert MARGIN_MULTIPLIER_MIN <= m <= MARGIN_MULTIPLIER_MAX
