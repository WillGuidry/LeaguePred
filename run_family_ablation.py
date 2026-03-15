"""
Family-Level Ablation Study
------------------------------
Tests each Lead State feature family's signal individually and incrementally.

Approach:
    1. Run the full pipeline with Lead State enabled (computes all 24 features)
    2. On the test set, use logistic regression to blend Elo probability with
       different subsets of Lead State features
    3. Compare: Elo baseline → +each family alone → cumulative families → all

This avoids re-running the pipeline for each family — we just vary which
features the logistic model sees.
"""

import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.data.loader import load_oracle_csv
from src.elo.pipeline import run_elo_pipeline
from src.evaluation.splits import temporal_split
from src.evaluation.metrics import log_loss, brier_score, accuracy, calibration_error


# Feature families with their column names (as they appear in pipeline output)
FAMILIES = {
    "A: Lead Creation": [
        "ls_avg_gd10_a", "ls_avg_gd15_a",
        "ls_first_dragon_rate_a", "ls_first_herald_rate_a",
        "ls_first_tower_rate_a", "ls_first_blood_rate_a",
        "ls_plate_diff_a",
        "ls_avg_gd10_b", "ls_avg_gd15_b",
        "ls_first_dragon_rate_b", "ls_first_herald_rate_b",
        "ls_first_tower_rate_b", "ls_first_blood_rate_b",
        "ls_plate_diff_b",
    ],
    "B: Advantage Quality": [
        "ls_lead_stability_a", "ls_compound_lead_rate_a",
        "ls_gold_volatility_when_ahead_a",
        "ls_lead_stability_b", "ls_compound_lead_rate_b",
        "ls_gold_volatility_when_ahead_b",
    ],
    "C: Lead Conversion": [
        "ls_win_rate_when_ahead_2k_15_a", "ls_win_rate_when_ahead_3k_20_a",
        "ls_close_time_ahead_15_a", "ls_close_time_ahead_20_a",
        "ls_gold_snowball_rate_a", "ls_dragon_soul_rate_a",
        "ls_herald_tower_conv_a", "ls_baron_win_rate_a",
        "ls_win_rate_when_ahead_2k_15_b", "ls_win_rate_when_ahead_3k_20_b",
        "ls_close_time_ahead_15_b", "ls_close_time_ahead_20_b",
        "ls_gold_snowball_rate_b", "ls_dragon_soul_rate_b",
        "ls_herald_tower_conv_b", "ls_baron_win_rate_b",
    ],
    "D: Throw Tendency": [
        "ls_throw_rate_2k_15_a", "ls_lead_evaporation_rate_a",
        "ls_baron_throw_rate_a",
        "ls_throw_rate_2k_15_b", "ls_lead_evaporation_rate_b",
        "ls_baron_throw_rate_b",
    ],
    "E: Comeback / Resistance": [
        "ls_win_rate_when_behind_2k_15_a", "ls_gold_recovery_rate_a",
        "ls_extend_time_behind_a",
        "ls_win_rate_when_behind_2k_15_b", "ls_gold_recovery_rate_b",
        "ls_extend_time_behind_b",
    ],
}


def blend_with_logistic(
    X_train, y_train, X_test,
    C=1.0,
):
    """Fit logistic regression and return test probabilities."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = LogisticRegression(C=C, max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)
    probs = model.predict_proba(X_test_s)[:, 1]
    return probs, model


def run_family_ablation(data_path, leagues=None, test_fraction=0.2):
    """Run the full family ablation study."""

    # Load data
    print("Loading data...")
    df = load_oracle_csv(data_path, leagues=leagues)
    print(f"Loaded {len(df)} games\n")

    # Run pipeline with Lead State + MOV enabled
    print("Running pipeline with all modules...")
    result = run_elo_pipeline(
        df,
        use_mov=True,
        use_lead_state=True,
        use_momentum=False,
        use_series_dynamics=False,
    )

    # Split
    train, test = temporal_split(result, test_fraction=test_fraction)

    # Check which columns actually exist
    all_ls_cols = [c for c in result.columns if c.startswith("ls_")]
    print(f"Lead State columns found: {len(all_ls_cols)}")

    # Filter to clean rows
    base_cols = ["pred_prob_a", "actual_outcome_a", "lead_edge", "lead_confidence"]
    all_needed = base_cols + all_ls_cols
    existing = [c for c in all_needed if c in result.columns]

    train_clean = train.dropna(subset=existing)
    test_clean = test.dropna(subset=existing)

    y_train = train_clean["actual_outcome_a"].values
    y_test = test_clean["actual_outcome_a"].values
    elo_probs_test = test_clean["pred_prob_a"].values

    print(f"Train: {len(train_clean)} games, Test: {len(test_clean)} games")
    print()

    # Baseline: Elo-only
    elo_ll = log_loss(y_test, elo_probs_test)
    elo_bs = brier_score(y_test, elo_probs_test)
    elo_acc = accuracy(y_test, elo_probs_test)
    elo_cal = calibration_error(y_test, elo_probs_test)

    print("=" * 80)
    print("FAMILY-LEVEL ABLATION STUDY")
    print("=" * 80)
    print(f"\n{'Model':<45} {'LogLoss':>8} {'Brier':>8} {'Acc':>7} {'ECE':>7} {'ΔLL':>8}")
    print("-" * 80)
    print(f"{'Elo baseline':<45} {elo_ll:>8.4f} {elo_bs:>8.4f} {elo_acc:>6.1%} {elo_cal:>7.4f} {'---':>8}")

    # --- Individual family tests ---
    print(f"\n{'--- Individual Families (Elo + one family) ---'}")
    family_results = {}

    for family_name, family_cols in FAMILIES.items():
        # Filter to columns that actually exist in the data
        cols = [c for c in family_cols if c in result.columns]
        if not cols:
            print(f"  {family_name}: no columns found, skipping")
            continue

        X_train = np.column_stack([
            train_clean["pred_prob_a"].values,
            train_clean[cols].values,
        ])
        X_test = np.column_stack([
            test_clean["pred_prob_a"].values,
            test_clean[cols].values,
        ])

        try:
            probs, model = blend_with_logistic(X_train, y_train, X_test)
            ll = log_loss(y_test, probs)
            bs = brier_score(y_test, probs)
            acc = accuracy(y_test, probs)
            cal = calibration_error(y_test, probs)
            delta = ll - elo_ll

            family_results[family_name] = {
                "ll": ll, "bs": bs, "acc": acc, "cal": cal, "delta": delta,
                "cols": cols,
            }

            marker = "▼" if delta < -0.001 else ("▲" if delta > 0.001 else "~")
            print(f"  Elo + {family_name:<40} {ll:>8.4f} {bs:>8.4f} {acc:>6.1%} {cal:>7.4f} {delta:>+7.4f} {marker}")
        except Exception as e:
            print(f"  Elo + {family_name}: ERROR - {e}")

    # --- Cumulative family tests ---
    print(f"\n{'--- Cumulative Families ---'}")
    cumulative_cols = []
    family_order = list(FAMILIES.keys())

    for family_name in family_order:
        family_cols = FAMILIES[family_name]
        cols = [c for c in family_cols if c in result.columns]
        cumulative_cols.extend(cols)

        if not cumulative_cols:
            continue

        X_train = np.column_stack([
            train_clean["pred_prob_a"].values,
            train_clean[cumulative_cols].values,
        ])
        X_test = np.column_stack([
            test_clean["pred_prob_a"].values,
            test_clean[cumulative_cols].values,
        ])

        try:
            probs, model = blend_with_logistic(X_train, y_train, X_test)
            ll = log_loss(y_test, probs)
            bs = brier_score(y_test, probs)
            acc = accuracy(y_test, probs)
            cal = calibration_error(y_test, probs)
            delta = ll - elo_ll

            short = family_name.split(":")[0]
            families_so_far = "+".join(
                f.split(":")[0] for f in family_order[:family_order.index(family_name) + 1]
            )
            label = f"Elo + {families_so_far}"
            marker = "▼" if delta < -0.001 else ("▲" if delta > 0.001 else "~")
            print(f"  {label:<45} {ll:>8.4f} {bs:>8.4f} {acc:>6.1%} {cal:>7.4f} {delta:>+7.4f} {marker}")
        except Exception as e:
            print(f"  Cumulative up to {family_name}: ERROR - {e}")

    # --- Edge score only (raw lead_edge from module) ---
    print(f"\n{'--- Lead Edge Score Only ---'}")
    X_train = np.column_stack([
        train_clean["pred_prob_a"].values,
        train_clean["lead_edge"].values,
        train_clean["lead_confidence"].values,
    ])
    X_test = np.column_stack([
        test_clean["pred_prob_a"].values,
        test_clean["lead_edge"].values,
        test_clean["lead_confidence"].values,
    ])
    try:
        probs, model = blend_with_logistic(X_train, y_train, X_test)
        ll = log_loss(y_test, probs)
        bs = brier_score(y_test, probs)
        acc = accuracy(y_test, probs)
        cal = calibration_error(y_test, probs)
        delta = ll - elo_ll
        marker = "▼" if delta < -0.001 else ("▲" if delta > 0.001 else "~")
        print(f"  {'Elo + lead_edge + confidence':<45} {ll:>8.4f} {bs:>8.4f} {acc:>6.1%} {cal:>7.4f} {delta:>+7.4f} {marker}")
    except Exception as e:
        print(f"  Lead edge: ERROR - {e}")

    # --- All features ---
    print(f"\n{'--- All Features ---'}")
    all_family_cols = []
    for cols in FAMILIES.values():
        all_family_cols.extend([c for c in cols if c in result.columns])

    X_train = np.column_stack([
        train_clean["pred_prob_a"].values,
        train_clean["lead_edge"].values,
        train_clean["lead_confidence"].values,
        train_clean[all_family_cols].values,
    ])
    X_test = np.column_stack([
        test_clean["pred_prob_a"].values,
        test_clean["lead_edge"].values,
        test_clean["lead_confidence"].values,
        test_clean[all_family_cols].values,
    ])
    try:
        probs, model = blend_with_logistic(X_train, y_train, X_test)
        ll = log_loss(y_test, probs)
        bs = brier_score(y_test, probs)
        acc = accuracy(y_test, probs)
        cal = calibration_error(y_test, probs)
        delta = ll - elo_ll
        marker = "▼" if delta < -0.001 else ("▲" if delta > 0.001 else "~")
        print(f"  {'Elo + ALL lead state features':<45} {ll:>8.4f} {bs:>8.4f} {acc:>6.1%} {cal:>7.4f} {delta:>+7.4f} {marker}")

        # Print feature importances (top 10)
        feature_names = ["pred_prob_a", "lead_edge", "lead_confidence"] + all_family_cols
        coefs = model.coef_[0]
        importance = sorted(zip(feature_names, coefs), key=lambda x: abs(x[1]), reverse=True)
        print(f"\n  Top 15 features by |coefficient|:")
        for name, coef in importance[:15]:
            # Clean up name for display
            short = name.replace("ls_", "").replace("_a", " (A)").replace("_b", " (B)")
            print(f"    {short:<45} {coef:>+8.4f}")
    except Exception as e:
        print(f"  All features: ERROR - {e}")

    print(f"\n{'=' * 80}")
    print("▼ = improves on Elo baseline, ▲ = worse, ~ = neutral")
    print(f"Baseline Elo log loss: {elo_ll:.4f}")
    print()


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else "data/raw/matches_2025.csv"
    leagues = None
    if "--major-only" in sys.argv:
        leagues = ["LCK", "LPL", "LEC", "LCP", "LCS", "CBLOL", "MSI", "WLDs", "PCS"]
    run_family_ablation(data_path, leagues=leagues)
