"""
Elo Pipeline
-------------
Connects the Elo engine to data and evaluation.

This is the backbone pipeline:
    1. Load match data
    2. Process matches chronologically through Elo engine
    3. Generate predictions for each match BEFORE updating ratings
    4. Evaluate prediction quality

Key design principle:
    For every match, we predict FIRST, then update.
    This prevents look-ahead leakage — the Elo rating used for prediction
    has never seen the match outcome.
"""

import numpy as np
import pandas as pd
from typing import Optional

from src.elo.engine import EloEngine
from src.evaluation.metrics import full_report, compare_models
from src.evaluation.splits import temporal_split, walk_forward_splits


def run_elo_pipeline(
    df: pd.DataFrame,
    k_factor: float = 32.0,
    default_elo: float = 1500.0,
    scale_factor: float = 400.0,
    season_regression: float = 0.3,
    regression_col: Optional[str] = "split",
) -> pd.DataFrame:
    """
    Run the Elo engine over a chronological sequence of matches.

    For each match:
        1. Record the pre-match Elo prediction
        2. Process the match result and update ratings
        3. Store both prediction and outcome

    Args:
        df: Match data sorted by date, with columns:
            team_a, team_b, winner (required)
            date, league, split, patch (optional)
        k_factor: Elo K-factor.
        default_elo: Starting Elo for new teams.
        scale_factor: Elo scale factor.
        season_regression: How much to regress between splits.
        regression_col: Column to detect split boundaries (None to disable).

    Returns:
        DataFrame with original data plus prediction columns:
            elo_a, elo_b, elo_diff, pred_prob_a, pred_prob_b,
            predicted_winner, actual_outcome_a (1 if team_a won)
    """
    df = df.sort_values("date").reset_index(drop=True) if "date" in df.columns else df.copy()

    engine = EloEngine(
        default_elo=default_elo,
        k_factor=k_factor,
        scale_factor=scale_factor,
    )

    predictions = []
    last_split = None

    for idx, row in df.iterrows():
        team_a = row["team_a"]
        team_b = row["team_b"]
        winner = row["winner"]

        # Handle season regression at split boundaries
        if regression_col and regression_col in df.columns:
            current_split = row[regression_col]
            if last_split is not None and current_split != last_split:
                engine.regress_to_mean(season_regression)
            last_split = current_split

        # PREDICT FIRST (before seeing the result)
        pred = engine.predict(team_a, team_b)

        # Actual outcome (1 if team_a won, 0 if team_b won)
        if winner == team_a:
            actual_a = 1
        elif winner == team_b:
            actual_a = 0
        else:
            actual_a = np.nan  # Unknown winner

        predictions.append({
            "elo_a": pred["elo_a"],
            "elo_b": pred["elo_b"],
            "elo_diff": pred["elo_diff"],
            "pred_prob_a": pred["win_prob_a"],
            "pred_prob_b": pred["win_prob_b"],
            "predicted_winner": pred["predicted_winner"],
            "actual_outcome_a": actual_a,
        })

        # NOW update ratings with the result
        if pd.notna(actual_a):
            engine.process_match(team_a, team_b, winner)

    pred_df = pd.DataFrame(predictions)
    result = pd.concat([df.reset_index(drop=True), pred_df], axis=1)

    return result


def evaluate_elo(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    k_factor: float = 32.0,
    default_elo: float = 1500.0,
    scale_factor: float = 400.0,
    season_regression: float = 0.3,
    model_name: str = "Elo Baseline",
) -> dict:
    """
    Run Elo on the full dataset and evaluate on the test portion.

    Important: Elo ratings are built from ALL data chronologically,
    but evaluation metrics are computed ONLY on the test set.
    This mirrors real usage — we've been tracking ratings since the beginning,
    and we evaluate our predictions going forward.

    Args:
        df: Match data with date, team_a, team_b, winner.
        test_fraction: Fraction of data to hold out for evaluation.
        k_factor, default_elo, scale_factor, season_regression: Elo params.
        model_name: Name for the report.

    Returns:
        Evaluation report dict.
    """
    # Run Elo over the entire dataset
    results = run_elo_pipeline(
        df,
        k_factor=k_factor,
        default_elo=default_elo,
        scale_factor=scale_factor,
        season_regression=season_regression,
    )

    # Split into train/test by time
    train, test = temporal_split(results, test_fraction=test_fraction)

    # Evaluate on test set only
    test_clean = test.dropna(subset=["actual_outcome_a", "pred_prob_a"])

    if len(test_clean) == 0:
        print("WARNING: No valid predictions in test set.")
        return {}

    y_true = test_clean["actual_outcome_a"].values
    y_prob = test_clean["pred_prob_a"].values

    report = full_report(y_true, y_prob, model_name=model_name)
    report["elo_params"] = {
        "k_factor": k_factor,
        "default_elo": default_elo,
        "scale_factor": scale_factor,
        "season_regression": season_regression,
    }
    report["results_df"] = results

    return report


def walk_forward_evaluate_elo(
    df: pd.DataFrame,
    n_splits: int = 5,
    min_train_size: int = 200,
    k_factor: float = 32.0,
    default_elo: float = 1500.0,
    scale_factor: float = 400.0,
    season_regression: float = 0.3,
    model_name: str = "Elo Baseline",
) -> dict:
    """
    Walk-forward evaluation of Elo predictions.

    For each split:
        - Run Elo from the beginning through the training period
        - Evaluate predictions on the test window

    This is more robust than a single train/test split because it
    tests across multiple time periods.

    Returns:
        Dict with per-split and aggregate metrics.
    """
    # Run Elo over entire dataset first
    results = run_elo_pipeline(
        df,
        k_factor=k_factor,
        default_elo=default_elo,
        scale_factor=scale_factor,
        season_regression=season_regression,
    )

    # Get walk-forward splits
    splits = walk_forward_splits(
        results,
        n_splits=n_splits,
        min_train_size=min_train_size,
    )

    from src.evaluation.metrics import log_loss, brier_score, accuracy, calibration_error

    split_results = []
    all_y_true = []
    all_y_prob = []

    for i, (train, test) in enumerate(splits):
        test_clean = test.dropna(subset=["actual_outcome_a", "pred_prob_a"])
        if len(test_clean) == 0:
            continue

        y_true = test_clean["actual_outcome_a"].values
        y_prob = test_clean["pred_prob_a"].values

        split_results.append({
            "split": i + 1,
            "train_size": len(train),
            "test_size": len(test_clean),
            "log_loss": round(log_loss(y_true, y_prob), 4),
            "brier_score": round(brier_score(y_true, y_prob), 4),
            "accuracy": round(accuracy(y_true, y_prob), 4),
            "calibration_error": round(calibration_error(y_true, y_prob), 4),
        })

        all_y_true.extend(y_true)
        all_y_prob.extend(y_prob)

    # Aggregate
    splits_df = pd.DataFrame(split_results)

    print(f"\n{'='*70}")
    print(f"WALK-FORWARD EVALUATION: {model_name} ({n_splits} splits)")
    print(f"{'='*70}")
    print(splits_df.to_string(index=False))
    print(f"\nMean log loss:     {splits_df['log_loss'].mean():.4f} (+/- {splits_df['log_loss'].std():.4f})")
    print(f"Mean Brier score:  {splits_df['brier_score'].mean():.4f} (+/- {splits_df['brier_score'].std():.4f})")
    print(f"Mean accuracy:     {splits_df['accuracy'].mean():.1%}")
    print()

    return {
        "model": model_name,
        "splits": splits_df,
        "aggregate_log_loss": round(log_loss(np.array(all_y_true), np.array(all_y_prob)), 4),
        "aggregate_brier": round(brier_score(np.array(all_y_true), np.array(all_y_prob)), 4),
        "aggregate_accuracy": round(accuracy(np.array(all_y_true), np.array(all_y_prob)), 4),
    }


def grid_search_elo(
    df: pd.DataFrame,
    k_factors: list = None,
    scale_factors: list = None,
    regressions: list = None,
    test_fraction: float = 0.2,
) -> pd.DataFrame:
    """
    Grid search over Elo hyperparameters.

    IMPORTANT: This uses a single train/test split.
    After finding good params, validate with walk_forward_evaluate_elo.

    Args:
        df: Match data.
        k_factors: List of K values to try.
        scale_factors: List of scale factors to try.
        regressions: List of regression fractions to try.
        test_fraction: Fraction for test set.

    Returns:
        DataFrame with results sorted by log loss.
    """
    if k_factors is None:
        k_factors = [16, 24, 32, 40, 48]
    if scale_factors is None:
        scale_factors = [400]
    if regressions is None:
        regressions = [0.0, 0.2, 0.3, 0.4, 0.5]

    from src.evaluation.metrics import log_loss as ll_fn, brier_score as bs_fn, accuracy as acc_fn

    results = []
    total = len(k_factors) * len(scale_factors) * len(regressions)
    print(f"Grid search: {total} combinations...")

    for k in k_factors:
        for s in scale_factors:
            for r in regressions:
                pipeline_result = run_elo_pipeline(
                    df, k_factor=k, scale_factor=s, season_regression=r,
                )

                # Evaluate on test portion only
                split_idx = int(len(pipeline_result) * (1 - test_fraction))
                test = pipeline_result.iloc[split_idx:]
                test_clean = test.dropna(subset=["actual_outcome_a", "pred_prob_a"])

                if len(test_clean) == 0:
                    continue

                y_true = test_clean["actual_outcome_a"].values
                y_prob = test_clean["pred_prob_a"].values

                results.append({
                    "k_factor": k,
                    "scale_factor": s,
                    "regression": r,
                    "log_loss": round(ll_fn(y_true, y_prob), 4),
                    "brier_score": round(bs_fn(y_true, y_prob), 4),
                    "accuracy": round(acc_fn(y_true, y_prob), 4),
                    "test_size": len(test_clean),
                })

    results_df = pd.DataFrame(results).sort_values("log_loss")
    print(f"\nTop 5 configurations:")
    print(results_df.head().to_string(index=False))
    print()

    return results_df
