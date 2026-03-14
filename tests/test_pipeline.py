"""
Smoke tests for the core pipeline.
Run with: python -m pytest tests/ -v
"""

import numpy as np
import pandas as pd

from src.elo.engine import EloEngine
from src.elo.pipeline import run_elo_pipeline
from src.evaluation.metrics import log_loss, brier_score, accuracy, calibration_bins, full_report


def make_test_data(n_games=100):
    """Generate synthetic match data for testing."""
    np.random.seed(42)
    teams = ["Team_A", "Team_B", "Team_C", "Team_D"]
    # Team_A is strongest, Team_D is weakest
    strength = {"Team_A": 0.7, "Team_B": 0.55, "Team_C": 0.45, "Team_D": 0.3}

    rows = []
    base_date = pd.Timestamp("2024-01-01")
    for i in range(n_games):
        t1, t2 = np.random.choice(teams, 2, replace=False)
        # Simulate outcome based on relative strength
        p_t1 = strength[t1] / (strength[t1] + strength[t2])
        winner = t1 if np.random.random() < p_t1 else t2
        rows.append({
            "date": base_date + pd.Timedelta(days=i // 4),
            "team_a": t1,
            "team_b": t2,
            "winner": winner,
            "league": "TEST",
            "split": "Spring" if i < 50 else "Summer",
        })
    return pd.DataFrame(rows)


class TestEloEngine:
    def test_basic_prediction(self):
        engine = EloEngine()
        engine.add_team("A")
        engine.add_team("B")
        pred = engine.predict("A", "B")
        assert pred["win_prob_a"] == 0.5  # Equal ratings = 50/50

    def test_rating_update(self):
        engine = EloEngine()
        engine.process_match("A", "B", winner="A")
        assert engine.get_rating("A") > 1500
        assert engine.get_rating("B") < 1500

    def test_probabilities_sum_to_one(self):
        engine = EloEngine()
        engine.process_match("A", "B", winner="A")
        engine.process_match("A", "B", winner="A")
        pred = engine.predict("A", "B")
        assert abs(pred["win_prob_a"] + pred["win_prob_b"] - 1.0) < 1e-10

    def test_regression_to_mean(self):
        engine = EloEngine()
        engine.process_match("A", "B", winner="A")
        rating_before = engine.get_rating("A")
        engine.regress_to_mean(0.5)
        rating_after = engine.get_rating("A")
        assert abs(rating_after - 1500) < abs(rating_before - 1500)


class TestMetrics:
    def test_log_loss_baseline(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])
        ll = log_loss(y_true, y_prob)
        assert abs(ll - 0.6931) < 0.01  # ln(2)

    def test_log_loss_perfect(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.99, 0.01, 0.99, 0.01])
        ll = log_loss(y_true, y_prob)
        assert ll < 0.1

    def test_brier_baseline(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.5, 0.5, 0.5, 0.5])
        bs = brier_score(y_true, y_prob)
        assert abs(bs - 0.25) < 0.01

    def test_accuracy(self):
        y_true = np.array([1, 0, 1, 0])
        y_prob = np.array([0.8, 0.3, 0.6, 0.2])
        assert accuracy(y_true, y_prob) == 1.0

    def test_calibration_bins(self):
        y_true = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.6, 0.4, 0.8, 0.3, 0.2])
        cal = calibration_bins(y_true, y_prob, n_bins=5)
        assert len(cal) > 0
        assert "predicted_prob" in cal.columns
        assert "actual_win_rate" in cal.columns


class TestPipeline:
    def test_run_pipeline(self):
        df = make_test_data(50)
        result = run_elo_pipeline(df)
        assert "pred_prob_a" in result.columns
        assert "actual_outcome_a" in result.columns
        assert len(result) == 50

    def test_predictions_are_valid_probabilities(self):
        df = make_test_data(100)
        result = run_elo_pipeline(df)
        assert (result["pred_prob_a"] >= 0).all()
        assert (result["pred_prob_a"] <= 1).all()

    def test_first_prediction_is_fifty_fifty(self):
        df = make_test_data(100)
        result = run_elo_pipeline(df)
        # First game between any two teams should be 50/50
        assert result.iloc[0]["pred_prob_a"] == 0.5

    def test_full_report(self):
        df = make_test_data(100)
        result = run_elo_pipeline(df)
        clean = result.dropna(subset=["actual_outcome_a"])
        report = full_report(
            clean["actual_outcome_a"].values,
            clean["pred_prob_a"].values,
            model_name="Test",
            print_report=False,
        )
        assert "log_loss" in report
        assert "brier_score" in report
        assert report["log_loss"] < 0.693  # Should beat random
