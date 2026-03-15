"""
Ablation & Model Comparison
-----------------------------
Compare Elo, Elo+MOV, Elo+LeadState, and Elo+MOV+LeadState against each other.

This module runs the pipeline with different module combinations and reports
side-by-side metrics on a temporal test set.

For Lead State features, we also test a logistic regression that combines
Elo probability with Lead State features to produce a calibrated prediction.
This tests whether Lead State adds signal beyond raw Elo.
"""

import numpy as np
import pandas as pd
from typing import Optional

from src.elo.pipeline import run_elo_pipeline
from src.evaluation.metrics import (
    log_loss,
    brier_score,
    accuracy,
    calibration_error,
    compare_models,
    full_report,
)
from src.evaluation.splits import temporal_split


def run_ablation(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    k_factor: float = 32.0,
    scale_factor: float = 400.0,
    season_regression: float = 0.3,
    print_results: bool = True,
) -> pd.DataFrame:
    """
    Run ablation study comparing model variants.

    Variants tested:
        1. Elo-only (baseline)
        2. Elo + MOV
        3. Elo + Lead State (features stored, not blended into Elo prob)
        4. Elo + MOV + Lead State

    For variants with Lead State, we also fit a logistic regression
    on the training set to blend Elo probability with Lead State features,
    then evaluate the blended predictions on the test set.

    Returns:
        DataFrame with metrics per variant, sorted by log loss.
    """
    common_kwargs = dict(
        k_factor=k_factor,
        scale_factor=scale_factor,
        season_regression=season_regression,
        use_regional_priors=True,
        detect_roster_changes=True,
    )

    # Run all variants
    variants = {
        "Elo-only": dict(use_mov=False, use_lead_state=False),
        "Elo + MOV": dict(use_mov=True, use_lead_state=False),
        "Elo + Lead State": dict(use_mov=False, use_lead_state=True),
        "Elo + MOV + Lead State": dict(use_mov=True, use_lead_state=True),
    }

    results = {}
    for name, kwargs in variants.items():
        if print_results:
            print(f"Running: {name}...")
        results[name] = run_elo_pipeline(df, **common_kwargs, **kwargs)

    # Split each result into train/test
    predictions = {}
    y_true = None

    for name, result_df in results.items():
        _, test = temporal_split(result_df, test_fraction=test_fraction)
        test_clean = test.dropna(subset=["actual_outcome_a", "pred_prob_a"])

        if y_true is None:
            y_true = test_clean["actual_outcome_a"].values

        # Base Elo probability
        predictions[name] = test_clean["pred_prob_a"].values

    # For Lead State variants, also test logistic blending
    for name in ["Elo + Lead State", "Elo + MOV + Lead State"]:
        result_df = results[name]
        blended_name = f"{name} (blended)"
        blended_probs = _logistic_blend(result_df, test_fraction)
        if blended_probs is not None:
            predictions[blended_name] = blended_probs

    # Compare all
    if print_results:
        comparison = compare_models(y_true, predictions, print_comparison=True)
    else:
        comparison = compare_models(y_true, predictions, print_comparison=False)

    return comparison


def _logistic_blend(
    result_df: pd.DataFrame,
    test_fraction: float,
) -> Optional[np.ndarray]:
    """
    Fit a logistic regression on train data to blend Elo prob + Lead State features,
    then predict on test data.

    Returns test-set blended probabilities, or None if sklearn is unavailable
    or there's insufficient data.
    """
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        return None

    train, test = temporal_split(result_df, test_fraction=test_fraction)

    # Feature columns: Elo prob + lead state features
    lead_cols = [c for c in result_df.columns if c.startswith("ls_") and c.endswith("_a")]
    feature_cols = ["pred_prob_a", "lead_edge", "lead_confidence"] + lead_cols

    # Check if Lead State columns exist
    if "lead_edge" not in result_df.columns:
        return None

    # Filter to rows with valid data
    all_cols = feature_cols + ["actual_outcome_a"]
    train_clean = train.dropna(subset=all_cols)
    test_clean = test.dropna(subset=all_cols)

    if len(train_clean) < 100 or len(test_clean) < 50:
        return None

    X_train = train_clean[feature_cols].values
    y_train = train_clean["actual_outcome_a"].values
    X_test = test_clean[feature_cols].values

    # Fit logistic regression with L2 regularization
    model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    # Predict probabilities on test set
    probs = model.predict_proba(X_test)[:, 1]

    return probs


def feature_importance_ablation(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    k_factor: float = 32.0,
    scale_factor: float = 400.0,
    season_regression: float = 0.3,
    print_results: bool = True,
) -> pd.DataFrame:
    """
    Drop each Lead State feature one at a time and measure impact on log loss.

    This identifies which features contribute most to prediction quality
    and which might be redundant.

    Returns:
        DataFrame with feature name, log_loss_without, and delta columns.
    """
    from src.config import LEAD_STATE_WEIGHTS

    # Full model baseline
    common_kwargs = dict(
        k_factor=k_factor,
        scale_factor=scale_factor,
        season_regression=season_regression,
        use_regional_priors=True,
        detect_roster_changes=True,
        use_mov=True,
        use_lead_state=True,
    )

    full_result = run_elo_pipeline(df, **common_kwargs)
    _, test = temporal_split(full_result, test_fraction=test_fraction)
    test_clean = test.dropna(subset=["actual_outcome_a", "pred_prob_a"])
    y_true = test_clean["actual_outcome_a"].values
    baseline_ll = log_loss(y_true, test_clean["pred_prob_a"].values)

    # Blended baseline
    blended_probs = _logistic_blend(full_result, test_fraction)
    if blended_probs is not None:
        baseline_blended_ll = log_loss(y_true[:len(blended_probs)], blended_probs)
    else:
        baseline_blended_ll = baseline_ll

    if print_results:
        print(f"\nFull model baseline log loss (Elo prob): {baseline_ll:.4f}")
        print(f"Full model baseline log loss (blended):  {baseline_blended_ll:.4f}")

    # Drop each feature and re-evaluate
    rows = []
    for feature in LEAD_STATE_WEIGHTS.keys():
        # Zero out this feature's weight
        modified_weights = dict(LEAD_STATE_WEIGHTS)
        modified_weights[feature] = 0.0

        # Re-normalize remaining weights
        total = sum(modified_weights.values())
        if total > 0:
            modified_weights = {k: v / total for k, v in modified_weights.items()}

        rows.append({
            "dropped_feature": feature,
            "weight": LEAD_STATE_WEIGHTS[feature],
        })

    if print_results:
        print("\nFeature weights (for reference):")
        for r in rows:
            print(f"  {r['dropped_feature']:<35} weight={r['weight']:.2f}")

    return pd.DataFrame(rows)
