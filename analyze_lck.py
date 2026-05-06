"""
LCK-only ELO model (2026 data).

Builds an ELO model using only LCK games, evaluates prediction accuracy
via walk-forward validation, and reports final rankings.

Usage:
    python analyze_lck.py
"""

import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np

from src.data.loader import load_oracle_csv, validate_data, print_validation_report
from src.elo.engine import EloEngine
from src.evaluation.metrics import log_loss, brier_score, accuracy


def main():
    # -------------------------------------------------------------------------
    # Load LCK data only
    # -------------------------------------------------------------------------
    print("Loading LCK 2026 data...")
    df = load_oracle_csv("data/raw/all_matches.csv", leagues=["LCK"])
    df = df.sort_values("date").reset_index(drop=True)

    print(f"\nLoaded {len(df)} LCK games")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Splits: {df['split'].dropna().unique().tolist()}")
    print(f"Unique teams: {df['team_a'].nunique() + df['team_b'].nunique() - len(set(df['team_a']) & set(df['team_b']))}")

    # Quick validation
    report = validate_data(df)
    print_validation_report(report)

    # -------------------------------------------------------------------------
    # Walk-forward evaluation: predict each game, then update ELO
    # -------------------------------------------------------------------------
    engine = EloEngine(default_elo=1500.0, k_factor=32.0)

    preds = []
    actuals = []
    skipped = 0

    for _, row in df.iterrows():
        if pd.isna(row["winner"]):
            skipped += 1
            continue

        team_a, team_b = row["team_a"], row["team_b"]

        # Skip first-meeting games (no information yet) for prediction quality
        # but still process them to learn ratings
        prob_a = engine.expected_score(team_a, team_b)
        actual = 1 if row["winner"] == team_a else 0

        preds.append(prob_a)
        actuals.append(actual)

        engine.process_match(team_a, team_b, row["winner"])

    preds = np.array(preds)
    actuals = np.array(actuals)

    print(f"\n{'='*60}")
    print(f"  LCK 2026 — WALK-FORWARD METRICS")
    print(f"  {len(preds)} games (skipped {skipped} unknown)")
    print(f"{'='*60}")

    print(f"\n  Log loss:  {log_loss(actuals, preds):.4f}  (lower = better; 0.693 = coin flip)")
    print(f"  Brier:     {brier_score(actuals, preds):.4f}  (lower = better; 0.250 = coin flip)")
    print(f"  Accuracy:  {accuracy(actuals, preds)*100:.1f}%")

    # Compare to coin-flip baseline
    coin = np.full(len(actuals), 0.5)
    print(f"\n  vs coin flip:")
    print(f"    Log loss improvement: {log_loss(actuals, coin) - log_loss(actuals, preds):+.4f}")
    print(f"    Brier improvement:    {brier_score(actuals, coin) - brier_score(actuals, preds):+.4f}")

    # -------------------------------------------------------------------------
    # Confidence-bucketed accuracy (calibration sanity check)
    # -------------------------------------------------------------------------
    print(f"\n  CALIBRATION (does X% confidence -> X% win rate?):")
    bins = [(0.0, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 1.0)]
    for lo, hi in bins:
        mask = (preds >= lo) & (preds < hi)
        if mask.sum() == 0:
            continue
        avg_pred = preds[mask].mean()
        avg_actual = actuals[mask].mean()
        n = mask.sum()
        print(f"    [{lo:.1f}-{hi:.1f}]  n={n:3d}  predicted={avg_pred:.3f}  actual={avg_actual:.3f}")

    # -------------------------------------------------------------------------
    # Final rankings
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  FINAL LCK RANKINGS (after all 2026 games)")
    print(f"{'='*60}")
    print(f"\n  {'Rank':<5} {'Team':<25} {'ELO':>8}  {'Games':>6}")
    print(f"  {'-'*50}")

    # Count games per team
    games_played = {}
    for _, row in df.iterrows():
        if pd.isna(row["winner"]):
            continue
        for t in [row["team_a"], row["team_b"]]:
            games_played[t] = games_played.get(t, 0) + 1

    for i, (team, elo) in enumerate(engine.get_rankings(), 1):
        n = games_played.get(team, 0)
        print(f"  {i:<5} {team:<25} {elo:>8.1f}  {n:>6}")

    # -------------------------------------------------------------------------
    # Save model state for reuse
    # -------------------------------------------------------------------------
    engine.save("data/processed/elo_engine_lck.json")
    print(f"\n  [saved LCK ELO state to data/processed/elo_engine_lck.json]")

    # -------------------------------------------------------------------------
    # Per-split breakdown (Cup vs Rounds 1-2)
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  ACCURACY BY SPLIT")
    print(f"{'='*60}")

    df_clean = df[df["winner"].notna()].reset_index(drop=True)
    df_clean["pred"] = preds
    df_clean["actual"] = actuals

    for split in df_clean["split"].dropna().unique():
        sub = df_clean[df_clean["split"] == split]
        if len(sub) == 0:
            continue
        s_preds = sub["pred"].values
        s_actuals = sub["actual"].values
        print(f"\n  {split} ({len(sub)} games):")
        print(f"    Log loss:  {log_loss(s_actuals, s_preds):.4f}")
        print(f"    Accuracy:  {accuracy(s_actuals, s_preds)*100:.1f}%")

    print()


if __name__ == "__main__":
    main()
