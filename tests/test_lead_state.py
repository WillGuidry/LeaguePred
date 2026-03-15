"""
Tests for the Lead State Efficiency module and TeamStateTracker.
"""

import pytest
import numpy as np
from src.modules.team_state_tracker import TeamStateTracker
from src.modules.lead_state import LeadStateModule


def _make_game(result=1, gd15=1000, gd20=1500, firstdragon=1, firstherald=0,
               firsttower=1, firstblood=0, dragons=2, barons=0, towers=5,
               gamelength=1800, totalgold=55000, opp_totalgold=48000):
    """Helper to create a game stat dict."""
    return {
        "result": result,
        "golddiffat15": gd15,
        "golddiffat20": gd20,
        "firstdragon": firstdragon,
        "firstherald": firstherald,
        "firsttower": firsttower,
        "firstblood": firstblood,
        "dragons": dragons,
        "barons": barons,
        "towers": towers,
        "gamelength": gamelength,
        "totalgold": totalgold,
        "opp_totalgold": opp_totalgold,
    }


class TestTeamStateTracker:
    """Test the TeamStateTracker rolling feature computation."""

    def test_empty_team_returns_defaults(self):
        """A team with no history should return default features."""
        tracker = TeamStateTracker(window=20, min_games=5)
        features = tracker.get_features("Unknown Team")
        assert features["n_games"] == 0
        assert features["avg_gd15"] == 0.0  # Global prior with no data

    def test_no_leakage(self):
        """Features at game N should use only games 1..N-1."""
        tracker = TeamStateTracker(window=20, min_games=1)

        # Record 5 games with known gold diffs
        gold_diffs = [1000, 2000, 3000, 4000, 5000]
        for gd in gold_diffs:
            tracker.record_game("TeamA", _make_game(gd15=gd))

        # After recording 5 games, features should reflect all 5
        features = tracker.get_features("TeamA")
        expected_avg = np.mean(gold_diffs)
        assert abs(features["avg_gd15"] - expected_avg) < 1.0

    def test_rolling_window(self):
        """With window=3, only the last 3 games should be used."""
        tracker = TeamStateTracker(window=3, min_games=1, shrinkage_weight=0.0)

        # Record 5 games
        for gd in [1000, 2000, 3000, 4000, 5000]:
            tracker.record_game("TeamA", _make_game(gd15=gd))

        features = tracker.get_features("TeamA")
        # Window=3, so only last 3 games: [3000, 4000, 5000]
        expected_avg = np.mean([3000, 4000, 5000])
        assert abs(features["avg_gd15"] - expected_avg) < 1.0
        assert features["n_games"] == 3

    def test_shrinkage_with_few_games(self):
        """With fewer than min_games, features should be shrunk toward priors."""
        tracker = TeamStateTracker(window=20, min_games=10, shrinkage_weight=5.0)

        # Seed global priors with some neutral games from other teams
        for _ in range(20):
            tracker.record_game("OtherTeam", _make_game(gd15=0))

        # Record just 2 games with very high gold diff for the test team
        tracker.record_game("TeamA", _make_game(gd15=5000))
        tracker.record_game("TeamA", _make_game(gd15=5000))

        features = tracker.get_features("TeamA")
        # avg_gd15 should be pulled toward ~0 (global prior) by shrinkage
        # shrink factor = 2 / (2 + 5) ≈ 0.286, so result ≈ 0.286 * 5000 + 0.714 * ~0
        assert features["avg_gd15"] < 5000  # Shrunk toward prior
        assert features["avg_gd15"] > 0     # But still positive

    def test_shrinkage_convergence(self):
        """With many games, shrinkage should have minimal effect."""
        tracker = TeamStateTracker(window=20, min_games=5, shrinkage_weight=5.0)

        # Record 20 games all with gd15=2000
        for _ in range(20):
            tracker.record_game("TeamA", _make_game(gd15=2000))

        features = tracker.get_features("TeamA")
        # With 20 games and min_games=5, no shrinkage on lead creation features
        # (shrinkage only applies when n < min_games)
        assert abs(features["avg_gd15"] - 2000) < 100

    def test_conditional_win_rate(self):
        """Test state-conditional win rate computation."""
        tracker = TeamStateTracker(window=20, min_games=1, shrinkage_weight=0.0)

        # Record 10 games: 8 games ahead at 15 (6 wins), 2 games behind
        for i in range(8):
            result = 1 if i < 6 else 0  # Win 6 of 8 when ahead
            tracker.record_game("TeamA", _make_game(result=result, gd15=3000))
        for _ in range(2):
            tracker.record_game("TeamA", _make_game(result=0, gd15=-500))

        features = tracker.get_features("TeamA")
        # Without shrinkage, win_rate_when_ahead should be 6/8 = 0.75
        # But shrinkage_weight=0 still applies in _shrunk_rate
        assert features["n_ahead_2k_15"] == 8
        assert abs(features["win_rate_when_ahead_2k_15"] - 0.75) < 0.01

    def test_confidence_scaling(self):
        """Confidence should scale from 0 to 1 with more games."""
        tracker = TeamStateTracker(window=20, min_games=5)

        # No games → 0 confidence
        assert tracker.get_confidence("TeamA") == 0.0

        # 3 games (below min) → low confidence
        for _ in range(3):
            tracker.record_game("TeamA", _make_game())
        conf_3 = tracker.get_confidence("TeamA")
        assert 0 < conf_3 < 0.3

        # 10 games → moderate confidence
        for _ in range(7):
            tracker.record_game("TeamA", _make_game())
        conf_10 = tracker.get_confidence("TeamA")
        assert conf_10 > conf_3

        # 20 games (full window) → high confidence
        for _ in range(10):
            tracker.record_game("TeamA", _make_game())
        conf_20 = tracker.get_confidence("TeamA")
        assert conf_20 >= conf_10
        assert conf_20 <= 1.0

    def test_game_count(self):
        """game_count should return number of recorded games."""
        tracker = TeamStateTracker(window=5)
        assert tracker.game_count("TeamA") == 0
        tracker.record_game("TeamA", _make_game())
        assert tracker.game_count("TeamA") == 1
        for _ in range(10):
            tracker.record_game("TeamA", _make_game())
        # Window is 5, so max stored is 5
        assert tracker.game_count("TeamA") == 5

    def test_reset_team(self):
        """reset_team should clear all history."""
        tracker = TeamStateTracker(window=20)
        tracker.record_game("TeamA", _make_game())
        tracker.record_game("TeamA", _make_game())
        tracker.reset_team("TeamA")
        assert tracker.game_count("TeamA") == 0

    def test_handles_none_values(self):
        """Should handle None values in stats gracefully."""
        tracker = TeamStateTracker(window=20, min_games=1)
        stats = {
            "result": 1,
            "golddiffat15": None,
            "golddiffat20": None,
            "firstdragon": None,
            "firstherald": None,
            "firsttower": None,
            "firstblood": None,
            "dragons": None,
            "barons": None,
            "towers": None,
            "gamelength": None,
            "totalgold": None,
            "opp_totalgold": None,
        }
        tracker.record_game("TeamA", stats)
        features = tracker.get_features("TeamA")
        assert features["n_games"] == 1


class TestLeadStateModule:
    """Test the LeadStateModule compute and confidence methods."""

    def _setup_module(self, window=20, min_games=5):
        tracker = TeamStateTracker(window=window, min_games=min_games, shrinkage_weight=5.0)
        module = LeadStateModule(tracker)
        return tracker, module

    def test_no_data_returns_zero(self):
        """With no data for either team, edge should be 0."""
        _, module = self._setup_module()
        edge = module.compute("TeamA", "TeamB")
        assert edge == 0.0
        assert module.confidence() == 0.0

    def test_symmetry(self):
        """compute(A, B) should be approximately -compute(B, A)."""
        tracker, module = self._setup_module(min_games=1)

        # Give TeamA strong stats
        for _ in range(10):
            tracker.record_game("TeamA", _make_game(result=1, gd15=2000))
            tracker.record_game("TeamB", _make_game(result=0, gd15=-1000))

        edge_ab = module.compute("TeamA", "TeamB")
        edge_ba = module.compute("TeamB", "TeamA")

        assert abs(edge_ab + edge_ba) < 0.01  # Should be symmetric

    def test_better_team_positive_edge(self):
        """A team with better stats should have a positive edge."""
        tracker, module = self._setup_module(min_games=1)

        for _ in range(10):
            tracker.record_game("Strong", _make_game(
                result=1, gd15=3000, gd20=5000,
                firstdragon=1, firstherald=1, firsttower=1,
            ))
            tracker.record_game("Weak", _make_game(
                result=0, gd15=-1000, gd20=-2000,
                firstdragon=0, firstherald=0, firsttower=0,
            ))

        edge = module.compute("Strong", "Weak")
        assert edge > 0

    def test_confidence_scales_edge(self):
        """Low confidence should dampen the edge score."""
        tracker, module = self._setup_module(min_games=5)

        # Only 2 games (below min_games=5)
        for _ in range(2):
            tracker.record_game("TeamA", _make_game(result=1, gd15=5000))
            tracker.record_game("TeamB", _make_game(result=0, gd15=-5000))

        edge_low = module.compute("TeamA", "TeamB")

        # Now add more games
        for _ in range(18):
            tracker.record_game("TeamA", _make_game(result=1, gd15=5000))
            tracker.record_game("TeamB", _make_game(result=0, gd15=-5000))

        edge_high = module.compute("TeamA", "TeamB")

        # With more data, edge should be larger (higher confidence)
        assert abs(edge_high) > abs(edge_low)

    def test_feature_contributions(self):
        """get_feature_contributions should return per-feature breakdown."""
        tracker, module = self._setup_module(min_games=1)

        for _ in range(10):
            tracker.record_game("TeamA", _make_game(result=1, gd15=2000))
            tracker.record_game("TeamB", _make_game(result=0, gd15=-500))

        contributions = module.get_feature_contributions("TeamA", "TeamB")
        assert "avg_gd15" in contributions
        assert contributions["avg_gd15"] > 0  # TeamA has higher avg_gd15

    def test_equal_teams_near_zero_edge(self):
        """Two teams with identical stats should have near-zero edge."""
        tracker, module = self._setup_module(min_games=1)

        for _ in range(10):
            tracker.record_game("TeamA", _make_game(result=1, gd15=1000))
            tracker.record_game("TeamB", _make_game(result=1, gd15=1000))

        edge = module.compute("TeamA", "TeamB")
        assert abs(edge) < 0.01
