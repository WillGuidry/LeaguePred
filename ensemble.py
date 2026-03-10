"""
Ensemble Model
--------------
LightGBM-based residual correction on top of Elo predictions.
Combines all feature layers into a final probability estimate.
"""

from typing import Optional

import numpy as np
import pandas as pd
import yaml
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class EnsembleModel:
    """
    Train a LightGBM model on residuals from Elo predictions.

    Final probability = elo_base + lgbm_residual_correction

    Features:
    - elo_win_prob: Elo-based win probability
    - archetype_similarity_delta: head-to-head archetype correction
    - comp_pca_0..7: PCA components from champion interaction matrix
    - player_mastery_delta: team mastery score difference
    - recent_form_last3: rolling win rate over last 3 games
    - series_game_number: which game in a best-of series (1, 2, 3...)
    """

    def __init__(self, config_path: str = "config.yaml"):
        cfg = load_config(config_path)
        ens_cfg = cfg.get("ensemble", {})

        self.lgbm_params = ens_cfg.get("lgbm_params", {})
        self.test_size = ens_cfg.get("test_size", 0.2)
        self.feature_names = ens_cfg.get("features", [
            "elo_win_prob", "archetype_similarity_delta",
            "comp_pca_0", "comp_pca_1", "comp_pca_2", "comp_pca_3",
            "comp_pca_4", "comp_pca_5", "comp_pca_6", "comp_pca_7",
            "player_mastery_delta", "recent_form_last3", "series_game_number",
        ])

        self.model: Optional[lgb.LGBMRegressor] = None
        self.shap_values: Optional[np.ndarray] = None
        self.train_metrics: dict = {}

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all required feature columns exist and fill missing values.

        Args:
            df: DataFrame that should contain the feature columns.

        Returns:
            DataFrame with all feature columns present (missing filled with 0).
        """
        result = df.copy()
        for col in self.feature_names:
            if col not in result.columns:
                result[col] = 0.0
            else:
                result[col] = pd.to_numeric(result[col], errors="coerce").fillna(0.0)
        return result

    def compute_recent_form(
        self,
        games_df: pd.DataFrame,
        window: int = 3,
    ) -> pd.DataFrame:
        """
        Add recent_form_last3 feature: rolling win rate delta over last N games.

        For each game, look at the last N games for each team and compute
        win rate, then take the delta (team_a_form - team_b_form).
        """
        df = games_df.copy()
        df["recent_form_last3"] = 0.0

        # Build per-team game history
        team_results = {}  # team -> list of (date, result)

        for idx, row in df.iterrows():
            team_a = row["team_a"]
            team_b = row["team_b"]
            winner = row.get("winner", "")

            # Look up recent form before this game
            form_a = self._get_recent_form(team_results, team_a, window)
            form_b = self._get_recent_form(team_results, team_b, window)
            df.at[idx, "recent_form_last3"] = form_a - form_b

            # Record results
            team_results.setdefault(team_a, []).append(1.0 if winner == team_a else 0.0)
            team_results.setdefault(team_b, []).append(1.0 if winner == team_b else 0.0)

        return df

    @staticmethod
    def _get_recent_form(team_results: dict, team: str, window: int) -> float:
        """Get recent win rate for a team."""
        history = team_results.get(team, [])
        if not history:
            return 0.5  # Default: even
        recent = history[-window:]
        return float(np.mean(recent))

    def compute_series_game_number(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add series_game_number feature.

        Infer game number within a series by looking at consecutive games
        between the same two teams on the same date.
        """
        df = games_df.copy()
        df["series_game_number"] = 1

        if "date" not in df.columns:
            return df

        prev_matchup = None
        prev_date = None
        game_num = 1

        for idx, row in df.iterrows():
            teams = frozenset([row["team_a"], row["team_b"]])
            date = row.get("date")

            if teams == prev_matchup and date == prev_date:
                game_num += 1
            else:
                game_num = 1

            df.at[idx, "series_game_number"] = game_num
            prev_matchup = teams
            prev_date = date

        return df

    def compute_residual_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the residual target: actual_win - elo_predicted_win.

        This is what the LightGBM model learns to predict.
        """
        result = df.copy()
        if "actual_win_a" not in result.columns:
            result["actual_win_a"] = (result["winner"] == result["team_a"]).astype(float)
        if "elo_win_prob" not in result.columns and "elo_prob_a" in result.columns:
            result["elo_win_prob"] = result["elo_prob_a"]

        result["residual"] = result["actual_win_a"] - result.get("elo_win_prob", 0.5)
        return result

    def train(self, df: pd.DataFrame) -> dict:
        """
        Train the LightGBM residual correction model.

        Uses chronological train/test split (no future leakage).

        Args:
            df: Full feature DataFrame with all features and residual target.

        Returns:
            Dict with train and test metrics.
        """
        df = self.prepare_features(df)
        df = self.compute_residual_target(df)

        # Drop rows with missing target
        df = df.dropna(subset=["residual"])

        # Chronological split
        split_idx = int(len(df) * (1 - self.test_size))
        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        X_train = train_df[self.feature_names].values
        y_train = train_df["residual"].values
        X_test = test_df[self.feature_names].values
        y_test = test_df["residual"].values

        # Train LightGBM
        params = {
            "n_estimators": self.lgbm_params.get("n_estimators", 200),
            "learning_rate": self.lgbm_params.get("learning_rate", 0.05),
            "max_depth": self.lgbm_params.get("max_depth", 4),
            "num_leaves": self.lgbm_params.get("num_leaves", 15),
            "min_child_samples": self.lgbm_params.get("min_child_samples", 20),
            "subsample": self.lgbm_params.get("subsample", 0.8),
            "colsample_bytree": self.lgbm_params.get("colsample_bytree", 0.8),
            "reg_alpha": self.lgbm_params.get("reg_alpha", 0.1),
            "reg_lambda": self.lgbm_params.get("reg_lambda", 1.0),
            "random_state": self.lgbm_params.get("random_state", 42),
            "verbose": -1,
        }

        self.model = lgb.LGBMRegressor(**params)
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
        )

        # Predictions
        train_pred = self.model.predict(X_train)
        test_pred = self.model.predict(X_test)

        # Evaluate
        self.train_metrics = {
            "train_rmse": float(np.sqrt(np.mean((train_pred - y_train) ** 2))),
            "test_rmse": float(np.sqrt(np.mean((test_pred - y_test) ** 2))),
            "train_mae": float(np.mean(np.abs(train_pred - y_train))),
            "test_mae": float(np.mean(np.abs(test_pred - y_test))),
            "train_size": len(train_df),
            "test_size": len(test_df),
        }

        # Compute final probabilities for test set
        elo_base = test_df["elo_win_prob"].values
        final_prob = np.clip(elo_base + test_pred, 0.01, 0.99)
        actual = test_df["actual_win_a"].values

        # Brier score comparison
        brier_elo = float(np.mean((elo_base - actual) ** 2))
        brier_ensemble = float(np.mean((final_prob - actual) ** 2))
        self.train_metrics["test_brier_elo"] = brier_elo
        self.train_metrics["test_brier_ensemble"] = brier_ensemble
        self.train_metrics["brier_improvement"] = brier_elo - brier_ensemble

        # Accuracy comparison
        acc_elo = float(np.mean((elo_base >= 0.5) == actual))
        acc_ensemble = float(np.mean((final_prob >= 0.5) == actual))
        self.train_metrics["test_accuracy_elo"] = acc_elo
        self.train_metrics["test_accuracy_ensemble"] = acc_ensemble

        return self.train_metrics

    def predict_residual(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict residual corrections for new data.

        Args:
            df: DataFrame with feature columns.

        Returns:
            Array of residual predictions.
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        df = self.prepare_features(df)
        X = df[self.feature_names].values
        return self.model.predict(X)

    def predict_probability(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predict final win probabilities: elo_base + lgbm_correction.

        Args:
            df: DataFrame with elo_win_prob and feature columns.

        Returns:
            Array of final probabilities, clipped to [0.01, 0.99].
        """
        residual = self.predict_residual(df)
        elo_base = df.get("elo_win_prob", pd.Series(0.5, index=df.index)).values
        return np.clip(elo_base + residual, 0.01, 0.99)

    def explain_with_shap(self, df: pd.DataFrame) -> dict:
        """
        Compute SHAP values for feature importance analysis.

        Args:
            df: DataFrame with feature columns.

        Returns:
            Dict with:
            - shap_values: array of SHAP values
            - feature_importance: sorted list of (feature, mean_abs_shap)
        """
        if not HAS_SHAP:
            return {"error": "shap not installed. pip install shap"}
        if self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")

        df = self.prepare_features(df)
        X = df[self.feature_names]

        explainer = shap.TreeExplainer(self.model)
        self.shap_values = explainer.shap_values(X)

        # Feature importance ranking
        mean_abs_shap = np.mean(np.abs(self.shap_values), axis=0)
        importance = sorted(
            zip(self.feature_names, mean_abs_shap),
            key=lambda x: x[1],
            reverse=True,
        )

        return {
            "shap_values": self.shap_values,
            "feature_importance": importance,
        }

    def plot_feature_importance(self, importance: list, output_path: str = "feature_importance.html"):
        """
        Plot SHAP-based feature importance using plotly.

        Args:
            importance: List of (feature_name, importance_score) tuples.
            output_path: Path to save the HTML plot.
        """
        if not HAS_PLOTLY:
            print("plotly not installed. pip install plotly")
            return

        features, scores = zip(*importance)
        fig = go.Figure(go.Bar(
            x=list(reversed(scores)),
            y=list(reversed(features)),
            orientation="h",
        ))
        fig.update_layout(
            title="Feature Importance (Mean |SHAP|)",
            xaxis_title="Mean Absolute SHAP Value",
            yaxis_title="Feature",
            height=400,
            margin=dict(l=200),
        )
        fig.write_html(output_path)
        print(f"Feature importance plot saved to {output_path}")

    def feature_importance_builtin(self) -> list:
        """
        Get LightGBM's built-in feature importance (gain-based).
        Useful as a quick check even without SHAP.

        Returns:
            Sorted list of (feature_name, importance) tuples.
        """
        if self.model is None:
            return []
        importances = self.model.feature_importances_
        ranked = sorted(
            zip(self.feature_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )
        return ranked


if __name__ == "__main__":
    print("=" * 50)
    print("Ensemble Model — Demo")
    print("=" * 50)

    np.random.seed(42)
    n = 500

    # Synthetic data
    df = pd.DataFrame({
        "team_a": [f"Team_{i % 10}" for i in range(n)],
        "team_b": [f"Team_{(i + 3) % 10}" for i in range(n)],
        "elo_win_prob": np.random.uniform(0.3, 0.7, n),
        "archetype_similarity_delta": np.random.normal(0, 0.3, n),
        "player_mastery_delta": np.random.normal(0, 0.2, n),
        "recent_form_last3": np.random.normal(0, 0.15, n),
        "series_game_number": np.random.choice([1, 2, 3], n),
        "date": pd.date_range("2023-01-01", periods=n, freq="D"),
    })
    for i in range(8):
        df[f"comp_pca_{i}"] = np.random.normal(0, 1, n)

    # Synthetic outcomes correlated with elo + some noise from features
    noise = 0.1 * df["archetype_similarity_delta"] + 0.05 * df["player_mastery_delta"]
    df["actual_win_a"] = (df["elo_win_prob"] + noise + np.random.normal(0, 0.3, n) > 0.5).astype(float)
    df["winner"] = np.where(df["actual_win_a"] == 1, df["team_a"], df["team_b"])

    model = EnsembleModel.__new__(EnsembleModel)
    cfg = load_config()
    ens_cfg = cfg.get("ensemble", {})
    model.lgbm_params = ens_cfg.get("lgbm_params", {})
    model.test_size = 0.2
    model.feature_names = ens_cfg.get("features", [])
    model.model = None
    model.shap_values = None
    model.train_metrics = {}

    metrics = model.train(df)

    print("\nTraining Results:")
    print(f"  Train RMSE:          {metrics['train_rmse']:.4f}")
    print(f"  Test RMSE:           {metrics['test_rmse']:.4f}")
    print(f"  Test Brier (Elo):    {metrics['test_brier_elo']:.4f}")
    print(f"  Test Brier (Ens):    {metrics['test_brier_ensemble']:.4f}")
    print(f"  Brier Improvement:   {metrics['brier_improvement']:.4f}")
    print(f"  Test Acc (Elo):      {metrics['test_accuracy_elo']:.1%}")
    print(f"  Test Acc (Ensemble): {metrics['test_accuracy_ensemble']:.1%}")

    print("\nBuilt-in Feature Importance:")
    for feat, imp in model.feature_importance_builtin():
        print(f"  {feat:<30} {imp}")
