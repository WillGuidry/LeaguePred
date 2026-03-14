"""
Evaluation Metrics
------------------
Core metrics for evaluating prediction quality and market edge.

Primary metrics:
    - Log loss: penalizes confident wrong predictions heavily
    - Brier score: mean squared error of probability predictions
    - Calibration: do 60% predictions win 60% of the time?
    - Accuracy: baseline sanity check (not sufficient alone)

Market metrics:
    - ROI: return on investment vs flat betting
    - CLV: closing line value (are we beating the closing odds?)

Leakage note:
    All evaluation uses time-ordered splits. NEVER random splits.
    The evaluator enforces this by requiring a date column.
"""

import numpy as np
import pandas as pd
from typing import Optional


def log_loss(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-15) -> float:
    """
    Compute log loss (cross-entropy) for binary predictions.

    Lower is better. Baseline (all 0.5) = 0.693.
    A model that predicts the base rate perfectly gets ~0.68 for 55/45 splits.

    Args:
        y_true: Actual outcomes (0 or 1).
        y_prob: Predicted probability of outcome = 1.

    Returns:
        Mean log loss.
    """
    y_prob = np.clip(y_prob, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Compute Brier score (mean squared error of probabilities).

    Lower is better. Baseline (all 0.5) = 0.25.
    Range: 0 (perfect) to 1 (worst).

    The Brier score decomposes into:
        Brier = Reliability - Resolution + Uncertainty
    where reliability measures calibration and resolution measures sharpness.
    """
    return np.mean((y_prob - y_true) ** 2)


def accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """
    Simple classification accuracy (predict winner = side with >50% prob).

    Warning: accuracy alone is misleading for betting. A model that's 55%
    accurate but poorly calibrated is less useful than one that's 53%
    accurate but well-calibrated.
    """
    y_pred = (y_prob >= 0.5).astype(int)
    return np.mean(y_true == y_pred)


def calibration_bins(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute calibration table: group predictions into bins and compare
    predicted probability to actual win rate.

    A well-calibrated model has predicted_prob ≈ actual_win_rate in each bin.

    Args:
        y_true: Actual outcomes (0 or 1).
        y_prob: Predicted probabilities.
        n_bins: Number of bins to divide predictions into.

    Returns:
        DataFrame with columns:
            bin_center: midpoint of the probability bin
            predicted_prob: mean predicted probability in bin
            actual_win_rate: actual win rate in bin
            count: number of predictions in bin
            deviation: actual - predicted (positive = underconfident)
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    rows = []
    for i in range(n_bins):
        mask = bin_indices == i
        count = mask.sum()
        if count == 0:
            continue

        pred_mean = y_prob[mask].mean()
        actual_mean = y_true[mask].mean()

        rows.append({
            "bin_center": (bins[i] + bins[i + 1]) / 2,
            "predicted_prob": round(pred_mean, 4),
            "actual_win_rate": round(actual_mean, 4),
            "count": int(count),
            "deviation": round(actual_mean - pred_mean, 4),
        })

    return pd.DataFrame(rows)


def calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE).

    Weighted average of |predicted - actual| across bins.
    Lower is better. 0 = perfectly calibrated.
    """
    cal = calibration_bins(y_true, y_prob, n_bins)
    if cal.empty:
        return float("nan")
    weights = cal["count"].values / cal["count"].sum()
    errors = np.abs(cal["deviation"].values)
    return float(np.sum(weights * errors))


def full_report(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    print_report: bool = True,
) -> dict:
    """
    Generate a complete evaluation report.

    Args:
        y_true: Actual outcomes.
        y_prob: Predicted probabilities.
        model_name: Label for the model.
        print_report: Whether to print results.

    Returns:
        Dict with all metrics.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    n = len(y_true)
    ll = log_loss(y_true, y_prob)
    bs = brier_score(y_true, y_prob)
    acc = accuracy(y_true, y_prob)
    ece = calibration_error(y_true, y_prob)
    cal = calibration_bins(y_true, y_prob)

    # Baseline comparison (always predict 50%)
    base_rate = y_true.mean()
    ll_baseline_50 = log_loss(y_true, np.full(n, 0.5))
    ll_baseline_rate = log_loss(y_true, np.full(n, base_rate))
    bs_baseline_50 = brier_score(y_true, np.full(n, 0.5))

    # Skill scores (positive = better than baseline)
    ll_skill_vs_50 = 1 - ll / ll_baseline_50
    bs_skill_vs_50 = 1 - bs / bs_baseline_50

    report = {
        "model": model_name,
        "n_predictions": n,
        "base_rate": round(base_rate, 4),
        "log_loss": round(ll, 4),
        "log_loss_baseline_50": round(ll_baseline_50, 4),
        "log_loss_baseline_rate": round(ll_baseline_rate, 4),
        "log_loss_skill_vs_50": round(ll_skill_vs_50, 4),
        "brier_score": round(bs, 4),
        "brier_baseline_50": round(bs_baseline_50, 4),
        "brier_skill_vs_50": round(bs_skill_vs_50, 4),
        "accuracy": round(acc, 4),
        "calibration_error": round(ece, 4),
        "calibration_table": cal,
    }

    if print_report:
        _print_report(report)

    return report


def _print_report(report: dict) -> None:
    """Pretty-print an evaluation report."""
    print(f"\n{'='*55}")
    print(f"EVALUATION REPORT: {report['model']}")
    print(f"{'='*55}")
    print(f"Predictions:          {report['n_predictions']}")
    print(f"Base rate (team_a):   {report['base_rate']:.1%}")
    print()
    print(f"{'Metric':<25} {'Model':>10} {'Baseline':>10} {'Skill':>10}")
    print(f"{'-'*55}")
    print(
        f"{'Log loss':<25} {report['log_loss']:>10.4f} "
        f"{report['log_loss_baseline_50']:>10.4f} "
        f"{report['log_loss_skill_vs_50']:>+10.4f}"
    )
    print(
        f"{'Brier score':<25} {report['brier_score']:>10.4f} "
        f"{report['brier_baseline_50']:>10.4f} "
        f"{report['brier_skill_vs_50']:>+10.4f}"
    )
    print(f"{'Accuracy':<25} {report['accuracy']:>10.1%}")
    print(f"{'Calibration error (ECE)':<25} {report['calibration_error']:>10.4f}")

    # Calibration table
    cal = report["calibration_table"]
    if not cal.empty:
        print(f"\n{'Calibration Breakdown':}")
        print(f"{'Predicted':>12} {'Actual':>12} {'Count':>8} {'Deviation':>12}")
        print(f"{'-'*46}")
        for _, row in cal.iterrows():
            dev = row['deviation']
            flag = " *" if abs(dev) > 0.05 else ""
            print(
                f"{row['predicted_prob']:>11.1%} "
                f"{row['actual_win_rate']:>11.1%} "
                f"{row['count']:>8d} "
                f"{dev:>+11.1%}{flag}"
            )
    print()


def compare_models(
    y_true: np.ndarray,
    predictions: dict,
    print_comparison: bool = True,
) -> pd.DataFrame:
    """
    Compare multiple models side by side.

    Args:
        y_true: Actual outcomes.
        predictions: Dict of {"model_name": y_prob_array, ...}
        print_comparison: Whether to print results.

    Returns:
        DataFrame with one row per model.
    """
    y_true = np.asarray(y_true, dtype=float)
    rows = []

    for name, y_prob in predictions.items():
        y_prob = np.asarray(y_prob, dtype=float)
        rows.append({
            "model": name,
            "log_loss": round(log_loss(y_true, y_prob), 4),
            "brier_score": round(brier_score(y_true, y_prob), 4),
            "accuracy": round(accuracy(y_true, y_prob), 4),
            "calibration_error": round(calibration_error(y_true, y_prob), 4),
        })

    result = pd.DataFrame(rows).sort_values("log_loss")

    if print_comparison:
        print(f"\n{'='*70}")
        print("MODEL COMPARISON (sorted by log loss)")
        print(f"{'='*70}")
        print(result.to_string(index=False))
        print()

    return result
