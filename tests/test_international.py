"""
Tests for PersistentInternationalElo and its integration with TournamentPredictor.
Run with: python -m pytest tests/test_international.py -v
"""

from src.elo.engine import EloEngine
from src.elo.international import PersistentInternationalElo, TournamentPredictor


def _make_domestic_engine():
    """Create a domestic engine with teams from different leagues."""
    engine = EloEngine()
    # Simulate some domestic games to establish ratings
    # BLG dominates LPL, FEARX is mid-tier LCK
    for _ in range(10):
        engine.process_match("BLG", "LPL_Team2", winner="BLG")
        engine.process_match("BLG", "LPL_Team3", winner="BLG")
        engine.process_match("LPL_Team2", "LPL_Team3", winner="LPL_Team2")
        engine.process_match("T1", "FEARX", winner="T1")
        engine.process_match("FEARX", "LCK_Team3", winner="FEARX")
        engine.process_match("T1", "LCK_Team3", winner="T1")
    return engine


def _make_team_leagues():
    return {
        "BLG": "LPL",
        "LPL_Team2": "LPL",
        "LPL_Team3": "LPL",
        "T1": "LCK",
        "FEARX": "LCK",
        "LCK_Team3": "LCK",
    }


class TestPersistentInternationalElo:
    def test_zero_games_gives_zero_weight(self):
        p = PersistentInternationalElo()
        assert p.get_intl_weight("BLG") == 0.0

    def test_weight_ramps_with_games(self):
        p = PersistentInternationalElo(games_full_trust=20, max_weight=0.85)
        p.game_counts["BLG"] = 10
        p.ratings["BLG"] = 1600
        w = p.get_intl_weight("BLG")
        assert 0.4 < w < 0.5  # 10/20 * 0.85 = 0.425

    def test_full_trust_at_threshold(self):
        p = PersistentInternationalElo(games_full_trust=20, max_weight=0.85)
        p.game_counts["BLG"] = 20
        p.ratings["BLG"] = 1600
        assert p.get_intl_weight("BLG") == 0.85

    def test_caps_at_max_weight(self):
        p = PersistentInternationalElo(games_full_trust=20, max_weight=0.85)
        p.game_counts["BLG"] = 100  # Way above threshold
        p.ratings["BLG"] = 1600
        assert p.get_intl_weight("BLG") == 0.85

    def test_blended_prior_no_intl_games(self):
        """Team with no international games gets pure regional prior."""
        p = PersistentInternationalElo()
        result = p.get_blended_prior("FEARX", regional_prior=1400, domestic_deviation=50)
        assert result == 1450  # regional_prior + deviation, unchanged

    def test_blended_prior_with_intl_games(self):
        """Team with international games blends toward intl Elo."""
        p = PersistentInternationalElo(games_full_trust=20, max_weight=0.85)
        p.game_counts["BLG"] = 20
        p.ratings["BLG"] = 1600

        regional_prior = 1300
        deviation = 100
        result = p.get_blended_prior("BLG", regional_prior, deviation)

        # blended_base = (1-0.85)*1300 + 0.85*1600 = 195 + 1360 = 1555
        # result = 1555 + 100 = 1655
        expected = (1 - 0.85) * 1300 + 0.85 * 1600 + 100
        assert abs(result - expected) < 0.01

    def test_decay_regresses_toward_regional(self):
        p = PersistentInternationalElo(decay_factor=0.3)
        p.ratings["BLG"] = 1600
        p._regional_priors["BLG"] = 1300
        p.game_counts["BLG"] = 10

        p.decay()
        # 1600 + 0.3 * (1300 - 1600) = 1600 - 90 = 1510
        assert abs(p.ratings["BLG"] - 1510) < 0.01

    def test_absorb_tournament(self):
        """After absorbing a tournament, persistent store has the team's data."""
        engine = _make_domestic_engine()
        leagues = _make_team_leagues()

        predictor = TournamentPredictor(engine, leagues)
        # Simulate BLG playing 5 international games
        predictor.tournament_engine.add_team("BLG", elo=1600)
        predictor.tournament_engine.add_team("T1", elo=1550)
        for _ in range(3):
            predictor.update("BLG", "T1", "BLG")
        predictor.update("BLG", "T1", "T1")
        predictor.update("BLG", "T1", "BLG")

        p = PersistentInternationalElo()
        p.absorb_tournament(predictor)

        assert p.game_counts["BLG"] == 5
        assert p.game_counts["T1"] == 5
        assert "BLG" in p.ratings
        assert "T1" in p.ratings


class TestTournamentPredictorWithPersistent:
    def test_persistent_intl_affects_prior(self):
        """A team with international experience gets a different prior
        than the same team without."""
        engine = _make_domestic_engine()
        leagues = _make_team_leagues()

        # Without persistent intl
        pred_no_intl = TournamentPredictor(engine, leagues)
        prior_no = pred_no_intl._compute_prior_elo("BLG")

        # With persistent intl — BLG has 20 games at high Elo
        persistent = PersistentInternationalElo(
            games_full_trust=20, max_weight=0.85
        )
        persistent.ratings["BLG"] = 1650
        persistent.game_counts["BLG"] = 20
        persistent._regional_priors["BLG"] = 1300

        pred_with_intl = TournamentPredictor(
            engine, leagues, persistent_intl=persistent
        )
        prior_with = pred_with_intl._compute_prior_elo("BLG")

        # BLG's international Elo (1650) is higher than the LPL regional
        # prior, so the persistent version should give a higher prior
        assert prior_with > prior_no

    def test_no_persistent_intl_unchanged_behavior(self):
        """Without persistent_intl, TournamentPredictor behaves as before."""
        engine = _make_domestic_engine()
        leagues = _make_team_leagues()

        pred = TournamentPredictor(engine, leagues)
        result = pred.predict("BLG", "T1")
        assert "win_prob_a" in result
        assert 0 <= result["win_prob_a"] <= 1

    def test_experienced_team_vs_newcomer(self):
        """A team with intl experience should be rated more independently
        of their regional prior than a newcomer."""
        engine = _make_domestic_engine()
        leagues = _make_team_leagues()

        persistent = PersistentInternationalElo(
            games_full_trust=20, max_weight=0.85
        )
        # BLG has proven themselves internationally at 1650
        persistent.ratings["BLG"] = 1650
        persistent.game_counts["BLG"] = 20
        persistent._regional_priors["BLG"] = 1300

        # FEARX has never played internationally — 0 games
        # Their prior should be purely regional

        pred = TournamentPredictor(
            engine, leagues, persistent_intl=persistent
        )

        blg_prior = pred._compute_prior_elo("BLG")
        fearx_prior = pred._compute_prior_elo("FEARX")

        # BLG should be notably ahead of FEARX since intl Elo is high
        assert blg_prior > fearx_prior

    def test_multi_tournament_accumulation(self):
        """Persistent Elo accumulates across multiple tournaments."""
        engine = _make_domestic_engine()
        leagues = _make_team_leagues()
        persistent = PersistentInternationalElo()

        # Tournament 1: BLG plays 5 games
        pred1 = TournamentPredictor(
            engine, leagues, persistent_intl=persistent
        )
        pred1.tournament_engine.add_team("BLG", elo=1550)
        pred1.tournament_engine.add_team("T1", elo=1550)
        for _ in range(5):
            pred1.update("BLG", "T1", "BLG")
        persistent.absorb_tournament(pred1)
        persistent.decay()

        games_after_t1 = persistent.game_counts["BLG"]
        assert games_after_t1 == 5

        # Tournament 2: BLG plays 5 more games
        pred2 = TournamentPredictor(
            engine, leagues, persistent_intl=persistent
        )
        pred2.tournament_engine.add_team("BLG", elo=1550)
        pred2.tournament_engine.add_team("T1", elo=1550)
        for _ in range(5):
            pred2.update("BLG", "T1", "BLG")
        persistent.absorb_tournament(pred2)

        # Should have accumulated 10 total games
        assert persistent.game_counts["BLG"] == 10
