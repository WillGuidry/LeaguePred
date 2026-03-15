"""
Tests for the Lead State Efficiency module and TeamStateTracker.

Covers all 5 feature families:
    A. Lead Creation
    B. Advantage Quality
    C. Lead Conversion
    D. Throw Tendency
    E. Comeback / Resistance
"""

import pytest
import numpy as np
from src.modules.team_state_tracker import TeamStateTracker
from src.modules.lead_state import LeadStateModule


def _make_game(
    result=1, gd10=500, gd15=1000, gd20=1500, gd25=2000,
    firstdragon=1, firstherald=0, firsttower=1, firstblood=0, firstbaron=0,
    dragons=2, barons=0, towers=5, elementaldrakes=2, heralds=0, elders=0,
    gamelength=1800, totalgold=55000, opp_totalgold=48000,
    turretplates=3, opp_turretplates=2,
):
    """Helper to create a game stat dict with all fields."""
    return {
        "result": result,
        "golddiffat10": gd10,
        "golddiffat15": gd15,
        "golddiffat20": gd20,
        "golddiffat25": gd25,
        "firstdragon": firstdragon,
        "firstherald": firstherald,
        "firsttower": firsttower,
        "firstblood": firstblood,
        "firstbaron": firstbaron,
        "dragons": dragons,
        "barons": barons,
        "towers": towers,
        "elementaldrakes": elementaldrakes,
        "heralds": heralds,
        "elders": elders,
        "gamelength": gamelength,
        "totalgold": totalgold,
        "opp_totalgold": opp_totalgold,
        "turretplates": turretplates,
        "opp_turretplates": opp_turretplates,
    }


class TestTeamStateTracker:
    """Test the TeamStateTracker rolling feature computation."""

    def test_empty_team_returns_defaults(self):
        tracker = TeamStateTracker(window=20, min_games=5)
        features = tracker.get_features("Unknown Team")
        assert features["n_games"] == 0
        assert features["avg_gd15"] == 0.0
        # All new features should be present
        assert "lead_stability" in features
        assert "gold_snowball_rate" in features
        assert "throw_rate_2k_15" in features
        assert "gold_recovery_rate" in features

    def test_no_leakage(self):
        tracker = TeamStateTracker(window=20, min_games=1)
        gold_diffs = [1000, 2000, 3000, 4000, 5000]
        for gd in gold_diffs:
            tracker.record_game("TeamA", _make_game(gd15=gd))
        features = tracker.get_features("TeamA")
        expected_avg = np.mean(gold_diffs)
        assert abs(features["avg_gd15"] - expected_avg) < 1.0

    def test_rolling_window(self):
        tracker = TeamStateTracker(window=3, min_games=1, shrinkage_weight=0.0)
        for gd in [1000, 2000, 3000, 4000, 5000]:
            tracker.record_game("TeamA", _make_game(gd15=gd))
        features = tracker.get_features("TeamA")
        expected_avg = np.mean([3000, 4000, 5000])
        assert abs(features["avg_gd15"] - expected_avg) < 1.0
        assert features["n_games"] == 3

    def test_shrinkage_with_few_games(self):
        tracker = TeamStateTracker(window=20, min_games=10, shrinkage_weight=5.0)
        for _ in range(20):
            tracker.record_game("OtherTeam", _make_game(gd15=0, gd10=0))
        tracker.record_game("TeamA", _make_game(gd15=5000, gd10=3000))
        tracker.record_game("TeamA", _make_game(gd15=5000, gd10=3000))
        features = tracker.get_features("TeamA")
        assert features["avg_gd15"] < 5000  # Shrunk toward prior
        assert features["avg_gd15"] > 0

    def test_shrinkage_convergence(self):
        tracker = TeamStateTracker(window=20, min_games=5, shrinkage_weight=5.0)
        for _ in range(20):
            tracker.record_game("TeamA", _make_game(gd15=2000))
        features = tracker.get_features("TeamA")
        assert abs(features["avg_gd15"] - 2000) < 100

    def test_conditional_win_rate(self):
        tracker = TeamStateTracker(window=20, min_games=1, shrinkage_weight=0.0)
        for i in range(8):
            result = 1 if i < 6 else 0
            tracker.record_game("TeamA", _make_game(result=result, gd15=3000))
        for _ in range(2):
            tracker.record_game("TeamA", _make_game(result=0, gd15=-500))
        features = tracker.get_features("TeamA")
        assert features["n_ahead_2k_15"] == 8
        assert abs(features["win_rate_when_ahead_2k_15"] - 0.75) < 0.01

    def test_confidence_scaling(self):
        tracker = TeamStateTracker(window=20, min_games=5)
        assert tracker.get_confidence("TeamA") == 0.0
        for _ in range(3):
            tracker.record_game("TeamA", _make_game())
        conf_3 = tracker.get_confidence("TeamA")
        assert 0 < conf_3 < 0.3
        for _ in range(7):
            tracker.record_game("TeamA", _make_game())
        conf_10 = tracker.get_confidence("TeamA")
        assert conf_10 > conf_3
        for _ in range(10):
            tracker.record_game("TeamA", _make_game())
        conf_20 = tracker.get_confidence("TeamA")
        assert conf_20 >= conf_10
        assert conf_20 <= 1.0

    def test_game_count(self):
        tracker = TeamStateTracker(window=5)
        assert tracker.game_count("TeamA") == 0
        tracker.record_game("TeamA", _make_game())
        assert tracker.game_count("TeamA") == 1
        for _ in range(10):
            tracker.record_game("TeamA", _make_game())
        assert tracker.game_count("TeamA") == 5

    def test_reset_team(self):
        tracker = TeamStateTracker(window=20)
        tracker.record_game("TeamA", _make_game())
        tracker.record_game("TeamA", _make_game())
        tracker.reset_team("TeamA")
        assert tracker.game_count("TeamA") == 0

    def test_handles_none_values(self):
        tracker = TeamStateTracker(window=20, min_games=1)
        stats = {k: None for k in [
            "result", "golddiffat10", "golddiffat15", "golddiffat20",
            "golddiffat25", "firstdragon", "firstherald", "firsttower",
            "firstblood", "firstbaron", "dragons", "barons", "towers",
            "elementaldrakes", "heralds", "elders", "gamelength",
            "totalgold", "opp_totalgold", "turretplates", "opp_turretplates",
        ]}
        stats["result"] = 1
        tracker.record_game("TeamA", stats)
        features = tracker.get_features("TeamA")
        assert features["n_games"] == 1

    # --- Family A tests ---

    def test_avg_gd10(self):
        tracker = TeamStateTracker(window=20, min_games=1)
        for _ in range(5):
            tracker.record_game("TeamA", _make_game(gd10=800))
        features = tracker.get_features("TeamA")
        assert abs(features["avg_gd10"] - 800) < 1.0

    def test_plate_diff(self):
        tracker = TeamStateTracker(window=20, min_games=1)
        for _ in range(5):
            tracker.record_game("TeamA", _make_game(turretplates=4, opp_turretplates=1))
        features = tracker.get_features("TeamA")
        assert abs(features["plate_diff"] - 3.0) < 0.1

    # --- Family B tests ---

    def test_lead_stability(self):
        """Team ahead at 15 should be tracked for stability at 20."""
        tracker = TeamStateTracker(window=20, min_games=1, shrinkage_weight=0.0)
        # 6 games ahead at 15, 5 of them still ahead at 20
        for i in range(5):
            tracker.record_game("TeamA", _make_game(gd15=2000, gd20=2500))
        tracker.record_game("TeamA", _make_game(gd15=2000, gd20=-500))
        features = tracker.get_features("TeamA")
        assert abs(features["lead_stability"] - 5/6) < 0.01

    def test_gold_volatility_when_ahead(self):
        tracker = TeamStateTracker(window=20, min_games=1)
        # All games ahead, small change 15→20
        for _ in range(5):
            tracker.record_game("TeamA", _make_game(gd15=2000, gd20=2100))
        features = tracker.get_features("TeamA")
        assert features["gold_volatility_when_ahead"] < 200  # Low volatility

    def test_compound_lead_rate(self):
        tracker = TeamStateTracker(window=20, min_games=1)
        # 3 games ahead at 15 with first dragon, 2 not ahead
        for _ in range(3):
            tracker.record_game("TeamA", _make_game(
                gd15=2000, firstdragon=1, firstherald=0
            ))
        for _ in range(2):
            tracker.record_game("TeamA", _make_game(
                gd15=-500, firstdragon=0, firstherald=0
            ))
        features = tracker.get_features("TeamA")
        assert abs(features["compound_lead_rate"] - 3/5) < 0.01

    # --- Family C tests ---

    def test_close_time_ahead_20(self):
        tracker = TeamStateTracker(window=20, min_games=1)
        for _ in range(5):
            tracker.record_game("TeamA", _make_game(
                gd20=4000, gamelength=1600
            ))
        features = tracker.get_features("TeamA")
        assert abs(features["close_time_ahead_20"] - 1600) < 1.0

    def test_gold_snowball_rate(self):
        """Gold snowball should measure lead growth from 15→end when ahead at 15."""
        tracker = TeamStateTracker(window=20, min_games=1)
        # Ahead at 15 with gd15=2500, end gold diff = +7000 → snowball = 4500
        for _ in range(5):
            tracker.record_game("TeamA", _make_game(
                gd15=2500, totalgold=60000, opp_totalgold=53000
            ))
        features = tracker.get_features("TeamA")
        # gold_diff_end = 7000, gd15 = 2500, snowball = 7000 - 2500 = 4500
        assert features["gold_snowball_rate"] > 0

    def test_dragon_soul_rate(self):
        tracker = TeamStateTracker(window=20, min_games=1)
        # 3 out of 5 games got soul (4+ drakes)
        for _ in range(3):
            tracker.record_game("TeamA", _make_game(elementaldrakes=4))
        for _ in range(2):
            tracker.record_game("TeamA", _make_game(elementaldrakes=2))
        features = tracker.get_features("TeamA")
        assert abs(features["dragon_soul_rate"] - 0.6) < 0.01

    def test_herald_tower_conv(self):
        """Herald→tower conversion when team got first herald."""
        tracker = TeamStateTracker(window=20, min_games=1)
        # 4 games with first herald, 3 also got first tower
        for _ in range(3):
            tracker.record_game("TeamA", _make_game(firstherald=1, firsttower=1))
        tracker.record_game("TeamA", _make_game(firstherald=1, firsttower=0))
        # 1 game without first herald
        tracker.record_game("TeamA", _make_game(firstherald=0, firsttower=0))
        features = tracker.get_features("TeamA")
        # herald_tower_conv = 3/4 = 0.75 (only counts herald games)
        assert abs(features["herald_tower_conv"] - 0.75) < 0.01

    def test_baron_win_rate(self):
        tracker = TeamStateTracker(window=20, min_games=1, shrinkage_weight=0.0)
        # 5 games with first baron, 4 wins
        for _ in range(4):
            tracker.record_game("TeamA", _make_game(result=1, firstbaron=1))
        tracker.record_game("TeamA", _make_game(result=0, firstbaron=1))
        features = tracker.get_features("TeamA")
        assert abs(features["baron_win_rate"] - 0.8) < 0.01

    # --- Family D tests ---

    def test_throw_rate(self):
        tracker = TeamStateTracker(window=20, min_games=1, shrinkage_weight=0.0)
        # 10 games ahead at 15, 8 wins 2 losses → throw rate = 0.2
        for _ in range(8):
            tracker.record_game("TeamA", _make_game(result=1, gd15=3000))
        for _ in range(2):
            tracker.record_game("TeamA", _make_game(result=0, gd15=3000))
        features = tracker.get_features("TeamA")
        assert abs(features["throw_rate_2k_15"] - 0.2) < 0.01

    def test_lead_evaporation_rate(self):
        """Ahead at 15, behind at 20 = lead evaporated."""
        tracker = TeamStateTracker(window=20, min_games=1)
        # 4 games ahead at 15: 3 still ahead at 20, 1 behind
        for _ in range(3):
            tracker.record_game("TeamA", _make_game(gd15=2000, gd20=2500))
        tracker.record_game("TeamA", _make_game(gd15=2000, gd20=-500))
        features = tracker.get_features("TeamA")
        assert abs(features["lead_evaporation_rate"] - 0.25) < 0.01

    def test_baron_throw_rate(self):
        tracker = TeamStateTracker(window=20, min_games=1, shrinkage_weight=0.0)
        # 3 games: ahead at 15 + first baron — 2 wins, 1 loss
        for _ in range(2):
            tracker.record_game("TeamA", _make_game(
                result=1, gd15=3000, firstbaron=1
            ))
        tracker.record_game("TeamA", _make_game(
            result=0, gd15=3000, firstbaron=1
        ))
        features = tracker.get_features("TeamA")
        # baron_throw = 1 - (2/3) = ~0.333
        assert abs(features["baron_throw_rate"] - 1/3) < 0.01

    # --- Family E tests ---

    def test_gold_recovery_rate(self):
        """Behind at 15 but ahead at 20 = recovered."""
        tracker = TeamStateTracker(window=20, min_games=1)
        # 4 games behind at 15: 1 recovered to ahead at 20
        tracker.record_game("TeamA", _make_game(gd15=-2000, gd20=500))
        for _ in range(3):
            tracker.record_game("TeamA", _make_game(gd15=-2000, gd20=-1500))
        features = tracker.get_features("TeamA")
        assert abs(features["gold_recovery_rate"] - 0.25) < 0.01

    def test_extend_time_behind(self):
        tracker = TeamStateTracker(window=20, min_games=1)
        for _ in range(5):
            tracker.record_game("TeamA", _make_game(
                gd15=-3000, gamelength=2200
            ))
        features = tracker.get_features("TeamA")
        assert abs(features["extend_time_behind"] - 2200) < 1.0


class TestLeadStateModule:
    """Test the LeadStateModule compute and confidence methods."""

    def _setup_module(self, window=20, min_games=5):
        tracker = TeamStateTracker(window=window, min_games=min_games, shrinkage_weight=5.0)
        module = LeadStateModule(tracker)
        return tracker, module

    def test_no_data_returns_zero(self):
        _, module = self._setup_module()
        edge = module.compute("TeamA", "TeamB")
        assert edge == 0.0
        assert module.confidence() == 0.0

    def test_symmetry(self):
        tracker, module = self._setup_module(min_games=1)
        for _ in range(10):
            tracker.record_game("TeamA", _make_game(result=1, gd15=2000, gd10=1000))
            tracker.record_game("TeamB", _make_game(result=0, gd15=-1000, gd10=-500))
        edge_ab = module.compute("TeamA", "TeamB")
        edge_ba = module.compute("TeamB", "TeamA")
        assert abs(edge_ab + edge_ba) < 0.01

    def test_better_team_positive_edge(self):
        tracker, module = self._setup_module(min_games=1)
        for _ in range(10):
            tracker.record_game("Strong", _make_game(
                result=1, gd10=1500, gd15=3000, gd20=5000,
                firstdragon=1, firstherald=1, firsttower=1, firstbaron=1,
                elementaldrakes=4, turretplates=4, opp_turretplates=1,
            ))
            tracker.record_game("Weak", _make_game(
                result=0, gd10=-1000, gd15=-1000, gd20=-2000,
                firstdragon=0, firstherald=0, firsttower=0, firstbaron=0,
                elementaldrakes=1, turretplates=1, opp_turretplates=4,
            ))
        edge = module.compute("Strong", "Weak")
        assert edge > 0

    def test_confidence_scales_edge(self):
        tracker, module = self._setup_module(min_games=5)
        for _ in range(2):
            tracker.record_game("TeamA", _make_game(result=1, gd15=5000))
            tracker.record_game("TeamB", _make_game(result=0, gd15=-5000))
        edge_low = module.compute("TeamA", "TeamB")
        for _ in range(18):
            tracker.record_game("TeamA", _make_game(result=1, gd15=5000))
            tracker.record_game("TeamB", _make_game(result=0, gd15=-5000))
        edge_high = module.compute("TeamA", "TeamB")
        assert abs(edge_high) > abs(edge_low)

    def test_feature_contributions(self):
        tracker, module = self._setup_module(min_games=1)
        for _ in range(10):
            tracker.record_game("TeamA", _make_game(result=1, gd15=2000))
            tracker.record_game("TeamB", _make_game(result=0, gd15=-500))
        contributions = module.get_feature_contributions("TeamA", "TeamB")
        assert "avg_gd15" in contributions
        assert contributions["avg_gd15"] > 0

    def test_family_contributions(self):
        tracker, module = self._setup_module(min_games=1)
        for _ in range(10):
            tracker.record_game("TeamA", _make_game(
                result=1, gd10=1500, gd15=3000, gd20=4000,
                firstdragon=1, firstherald=1, firsttower=1,
                elementaldrakes=4, turretplates=4, opp_turretplates=1,
            ))
            tracker.record_game("TeamB", _make_game(
                result=0, gd10=-500, gd15=-1000, gd20=-2000,
                firstdragon=0, firstherald=0, firsttower=0,
                elementaldrakes=1, turretplates=1, opp_turretplates=4,
            ))
        families = module.get_family_contributions("TeamA", "TeamB")
        assert "lead_creation" in families
        assert "advantage_quality" in families
        assert "lead_conversion" in families
        assert "throw_tendency" in families
        assert "comeback_resistance" in families
        # Strong team should have positive lead creation
        assert families["lead_creation"] > 0

    def test_equal_teams_near_zero_edge(self):
        tracker, module = self._setup_module(min_games=1)
        for _ in range(10):
            tracker.record_game("TeamA", _make_game(result=1, gd15=1000))
            tracker.record_game("TeamB", _make_game(result=1, gd15=1000))
        edge = module.compute("TeamA", "TeamB")
        assert abs(edge) < 0.01

    def test_inverted_features_direction(self):
        """Throw rate, evaporation, close time: lower = better for the team."""
        tracker, module = self._setup_module(min_games=1)
        # TeamA: low throw rate (almost always converts leads)
        for _ in range(10):
            tracker.record_game("TeamA", _make_game(
                result=1, gd15=3000, gd20=4000, gamelength=1500,
            ))
        # TeamB: high throw rate (often loses from ahead)
        for _ in range(5):
            tracker.record_game("TeamB", _make_game(
                result=1, gd15=3000, gd20=4000, gamelength=2200,
            ))
        for _ in range(5):
            tracker.record_game("TeamB", _make_game(
                result=0, gd15=3000, gd20=-500, gamelength=2200,
            ))
        contribs = module.get_feature_contributions("TeamA", "TeamB")
        # TeamA should benefit from lower throw rate
        assert contribs.get("throw_rate_2k_15", 0) > 0
        # TeamA should benefit from faster close time
        assert contribs.get("close_time_ahead_15", 0) > 0
