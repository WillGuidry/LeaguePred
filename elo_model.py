"""
Elo Model
---------
Wraps the existing EloEngine with batch processing, evaluation metrics
(Brier score, log-loss), and comparison against market odds.
"""

import math
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from engine import EloEngine


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class EloModel:
    """
    Extended Elo model with batch processing and evaluation.

    Processes a game-level DataFrame chronologically, maintains ratings,
    and records predictions for evaluation.
    """

    def __init__(self, config_path: str = "config.yaml"):
        cfg = load_config(config_path)
        elo_cfg = cfg.get("elo", {})
        self.engine = EloEngine(
            default_elo=elo_cfg.get("default_rating", 1500.0),
            k_factor=elo_cfg.get("k_factor", 32.0),
            scale_factor=elo_cfg.get("scale_factor", 400.0),
        )
        self.season_regression = elo_cfg.get("season_regression", 0.3)
        self.predictions = []

    def process_games(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process all games chronologically and record Elo predictions.

        Expects a game-level DataFrame with columns:
        - team_a, team_b, winner
        - date (optional, for season boundary detection)
        - league (optional)

        Returns:
            DataFrame with one row per game including:
            - elo_prob_a: pre-match Elo win probability for team_a
            - elo_a_before, elo_b_before: ratings before the match
            - actual_win_a: 1 if team_a won, 0 otherwise
        """
        self.predictions = []
        prev_year = None

        for idx, row in games_df.iterrows():
            team_a = row["team_a"]
            team_b = row["team_b"]
            winner = row["winner"]

            # Season boundary regression
            if "date" in row and pd.notna(row.get("date")):
                year = pd.Timestamp(row["date"]).year
                if prev_year is not None and year != prev_year:
                    self.engine.regress_to_mean(self.season_regression)
                prev_year = year

            # Record pre-match prediction
            elo_a = self.engine.get_rating(team_a)
            elo_b = self.engine.get_rating(team_b)
            prob_a = self.engine.expected_score(team_a, team_b)
            actual_a = 1.0 if winner == team_a else 0.0

            self.predictions.append({
                "gameid": row.get("gameid", idx),
                "date": row.get("date"),
                "team_a": team_a,
                "team_b": team_b,
                "winner": winner,
                "elo_a_before": elo_a,
                "elo_b_before": elo_b,
                "elo_prob_a": prob_a,
                "actual_win_a": actual_a,
            })

            # Update ratings
            self.engine.process_match(team_a, team_b, winner)

        return pd.DataFrame(self.predictions)

    def get_predictions_df(self) -> pd.DataFrame:
        """Return the predictions DataFrame from the last process_games call."""
        return pd.DataFrame(self.predictions)

    def win_probability(self, team_a: str, team_b: str) -> float:
        """Get current Elo-based win probability for team_a vs team_b."""
        return self.engine.expected_score(team_a, team_b)

    def get_ratings(self) -> dict:
        """Return current ratings dict."""
        return dict(self.engine.ratings)

    # =========================================================================
    # Evaluation Metrics
    # =========================================================================

    @staticmethod
    def brier_score(predicted: np.ndarray, actual: np.ndarray) -> float:
        """
        Compute Brier score: mean squared error of probability predictions.
        Lower is better. Range [0, 1]. Random baseline = 0.25.
        """
        return float(np.mean((predicted - actual) ** 2))

    @staticmethod
    def log_loss(predicted: np.ndarray, actual: np.ndarray, eps: float = 1e-15) -> float:
        """
        Compute log-loss (binary cross-entropy).
        Lower is better. Punishes confident wrong predictions heavily.
        """
        p = np.clip(predicted, eps, 1 - eps)
        return -float(np.mean(actual * np.log(p) + (1 - actual) * np.log(1 - p)))

    @staticmethod
    def accuracy(predicted: np.ndarray, actual: np.ndarray) -> float:
        """
        Compute classification accuracy (predict winner = team with prob > 0.5).
        """
        pred_class = (predicted >= 0.5).astype(float)
        return float(np.mean(pred_class == actual))

    def evaluate(self, predictions_df: Optional[pd.DataFrame] = None) -> dict:
        """
        Evaluate the Elo model's predictions.

        Args:
            predictions_df: DataFrame with elo_prob_a and actual_win_a columns.
                            If None, uses self.predictions.

        Returns:
            Dict with brier_score, log_loss, accuracy, and num_games.
        """
        if predictions_df is None:
            predictions_df = self.get_predictions_df()

        if predictions_df.empty:
            return {"brier_score": None, "log_loss": None, "accuracy": None, "num_games": 0}

        pred = predictions_df["elo_prob_a"].values
        actual = predictions_df["actual_win_a"].values

        return {
            "brier_score": self.brier_score(pred, actual),
            "log_loss": self.log_loss(pred, actual),
            "accuracy": self.accuracy(pred, actual),
            "num_games": len(predictions_df),
        }

    def evaluate_vs_market(
        self,
        predictions_df: pd.DataFrame,
        market_prob_col: str = "market_prob_a",
    ) -> dict:
        """
        Compare Elo model accuracy against market odds.

        Args:
            predictions_df: Must contain elo_prob_a, actual_win_a, and
                            a column with market-implied probabilities.

        Returns:
            Dict comparing Elo vs Market on Brier score, log-loss, and accuracy.
        """
        if market_prob_col not in predictions_df.columns:
            raise ValueError(f"Column '{market_prob_col}' not found in DataFrame")

        # Drop rows where market data is missing
        df = predictions_df.dropna(subset=[market_prob_col])
        pred_elo = df["elo_prob_a"].values
        pred_market = df[market_prob_col].values
        actual = df["actual_win_a"].values

        return {
            "elo": {
                "brier_score": self.brier_score(pred_elo, actual),
                "log_loss": self.log_loss(pred_elo, actual),
                "accuracy": self.accuracy(pred_elo, actual),
            },
            "market": {
                "brier_score": self.brier_score(pred_market, actual),
                "log_loss": self.log_loss(pred_market, actual),
                "accuracy": self.accuracy(pred_market, actual),
            },
            "num_games": len(df),
        }


if __name__ == "__main__":
    # Demo with synthetic data
    print("=" * 50)
    print("Elo Model — Evaluation Demo")
    print("=" * 50)

    # Create a synthetic games DataFrame
    games = pd.DataFrame({
        "team_a": ["T1", "Gen.G", "T1", "HLE", "Gen.G", "T1", "HLE", "Gen.G"],
        "team_b": ["Gen.G", "HLE", "HLE", "KT", "KT", "KT", "Gen.G", "T1"],
        "winner": ["T1", "Gen.G", "T1", "HLE", "Gen.G", "T1", "Gen.G", "Gen.G"],
        "date": pd.date_range("2024-01-01", periods=8, freq="7D"),
    })

    model = EloModel.__new__(EloModel)
    model.engine = EloEngine()
    model.season_regression = 0.3
    model.predictions = []

    preds = model.process_games(games)
    metrics = model.evaluate(preds)

    print(f"\nResults over {metrics['num_games']} games:")
    print(f"  Brier Score: {metrics['brier_score']:.4f}")
    print(f"  Log Loss:    {metrics['log_loss']:.4f}")
    print(f"  Accuracy:    {metrics['accuracy']:.1%}")

    print("\nFinal Ratings:")
    for team, elo in sorted(model.engine.ratings.items(), key=lambda x: -x[1]):
        print(f"  {team:<12} {elo:.1f}")
