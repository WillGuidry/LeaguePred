"""
Run Baseline Evaluation
------------------------
Entry point for running the Elo baseline on your data.

Usage:
    python run_baseline.py --data data/raw/your_matches.csv
    python run_baseline.py --data data/raw/your_matches.csv --grid-search
    python run_baseline.py --data data/raw/your_matches.csv --walk-forward

This script:
    1. Loads your match data (Oracle's Elixir or generic CSV)
    2. Validates the data quality
    3. Runs the Elo engine chronologically
    4. Evaluates predictions on held-out future data
    5. Reports log loss, Brier score, calibration, and accuracy
"""

import argparse
import sys

from src.data.loader import load_oracle_csv, validate_data, print_validation_report
from src.elo.pipeline import (
    evaluate_elo,
    walk_forward_evaluate_elo,
    grid_search_elo,
    compare_regional_vs_flat,
)


def main():
    parser = argparse.ArgumentParser(description="LeaguePred — Elo Baseline Evaluation")
    parser.add_argument(
        "--data", required=True,
        help="Path to match data CSV (Oracle's Elixir format or generic)"
    )
    parser.add_argument(
        "--min-date", default=None,
        help="Only include matches after this date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--test-fraction", type=float, default=0.2,
        help="Fraction of data to use for testing (default: 0.2)"
    )
    parser.add_argument(
        "--k-factor", type=float, default=32.0,
        help="Elo K-factor (default: 32)"
    )
    parser.add_argument(
        "--walk-forward", action="store_true",
        help="Use walk-forward validation instead of simple split"
    )
    parser.add_argument(
        "--n-splits", type=int, default=5,
        help="Number of walk-forward splits (default: 5)"
    )
    parser.add_argument(
        "--grid-search", action="store_true",
        help="Run grid search over Elo hyperparameters"
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Compare flat Elo vs regional priors vs regional+roster"
    )
    parser.add_argument(
        "--no-regional", action="store_true",
        help="Disable regional Elo priors (use flat 1500)"
    )
    parser.add_argument(
        "--leagues", nargs="+", default=None,
        help="Only include these leagues (e.g., --leagues LCK LPL LEC LCS CBLOL LCP)"
    )
    parser.add_argument(
        "--major-only", action="store_true",
        help="Shortcut: only major leagues (LCK LPL LEC LCP LCS CBLOL MSI WLDs PCS)"
    )

    args = parser.parse_args()

    # Resolve league filter
    leagues = args.leagues
    if args.major_only:
        leagues = ["LCK", "LPL", "LEC", "LCP", "LCS", "CBLOL", "MSI", "WLDs", "PCS"]

    # Step 1: Load data
    print("\n" + "=" * 60)
    print("STEP 1: Loading Data")
    print("=" * 60)
    df = load_oracle_csv(args.data, min_date=args.min_date, leagues=leagues)

    # Step 2: Validate
    print("\n" + "=" * 60)
    print("STEP 2: Validating Data")
    print("=" * 60)
    report = validate_data(df)
    print_validation_report(report)

    if report.get("errors"):
        print("Fatal errors in data. Fix before proceeding.")
        sys.exit(1)

    use_regional = not args.no_regional

    # Step 3: Run evaluation
    if args.compare:
        print("\n" + "=" * 60)
        print("STEP 3: Ablation — Regional Priors vs Flat Elo")
        print("=" * 60)
        compare_regional_vs_flat(
            df,
            k_factor=args.k_factor,
            test_fraction=args.test_fraction,
        )

    elif args.grid_search:
        print("\n" + "=" * 60)
        print("STEP 3: Grid Search")
        print("=" * 60)
        grid_search_elo(df, test_fraction=args.test_fraction, use_regional_priors=use_regional)

    elif args.walk_forward:
        print("\n" + "=" * 60)
        print("STEP 3: Walk-Forward Evaluation")
        print("=" * 60)
        walk_forward_evaluate_elo(
            df,
            n_splits=args.n_splits,
            k_factor=args.k_factor,
            use_regional_priors=use_regional,
        )

    else:
        print("\n" + "=" * 60)
        print("STEP 3: Elo Baseline Evaluation")
        print("=" * 60)
        evaluate_elo(
            df,
            test_fraction=args.test_fraction,
            k_factor=args.k_factor,
            use_regional_priors=use_regional,
        )


if __name__ == "__main__":
    main()
