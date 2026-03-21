"""
Analyze red vs blue side winrate at First Stand 2026 (FST league).

Builds an ELO model from all pre-FST data, then for each FST game:
  1. Records ELO-expected win probability for each side
  2. Records actual outcome
  3. Compares: does red/blue win more than ELO expects?

Usage:
    python analyze_side_winrate.py
"""

import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from src.elo.engine import EloEngine


def main():
    # -------------------------------------------------------------------------
    # Load and prepare data
    # -------------------------------------------------------------------------
    df = pd.read_csv("data/raw/all_matches.csv", low_memory=False)
    df["date"] = pd.to_datetime(df["date"])

    # Keep only team-level rows (2 per game)
    team_df = df[df["position"] == "team"].copy()

    # Build game-level rows: one row per game with blue/red info
    games = []
    for gid, grp in team_df.groupby("gameid"):
        if len(grp) != 2:
            continue
        blue = grp[grp["side"] == "Blue"].iloc[0]
        red = grp[grp["side"] == "Red"].iloc[0]
        games.append({
            "gameid": gid,
            "date": blue["date"],
            "league": blue["league"],
            "split": blue["split"],
            "blue_team": blue["teamname"],
            "red_team": red["teamname"],
            "blue_win": int(blue["result"] == 1),
            "winner": blue["teamname"] if blue["result"] == 1 else red["teamname"],
        })

    gdf = pd.DataFrame(games).sort_values("date").reset_index(drop=True)
    print(f"Total games: {len(gdf)}")

    # -------------------------------------------------------------------------
    # Identify FST games and pre-FST games
    # -------------------------------------------------------------------------
    fst_mask = gdf["league"] == "FST"
    fst_df = gdf[fst_mask].copy()
    pre_fst_df = gdf[~fst_mask & (gdf["date"] < fst_df["date"].min())].copy()

    print(f"FST (First Stand) games: {len(fst_df)}")
    print(f"Pre-FST games for ELO training: {len(pre_fst_df)}")

    # -------------------------------------------------------------------------
    # Build ELO from all pre-FST data
    # -------------------------------------------------------------------------
    engine = EloEngine(k_factor=32.0)

    for _, row in pre_fst_df.iterrows():
        engine.process_match(row["blue_team"], row["red_team"], row["winner"])

    print(f"Teams with ratings after pre-FST training: {len(engine.ratings)}")

    # -------------------------------------------------------------------------
    # Process FST games: predict then update
    # -------------------------------------------------------------------------
    results = []
    for _, row in fst_df.iterrows():
        blue, red = row["blue_team"], row["red_team"]
        blue_win = row["blue_win"]

        # ELO expected win prob for blue side
        elo_prob_blue = engine.expected_score(blue, red)

        results.append({
            "gameid": row["gameid"],
            "date": row["date"],
            "blue_team": blue,
            "red_team": red,
            "blue_win": blue_win,
            "elo_prob_blue": elo_prob_blue,
            "elo_diff": engine.get_rating(blue) - engine.get_rating(red),
        })

        # Update ELO with result
        engine.process_match(blue, red, row["winner"])

    rdf = pd.DataFrame(results)

    # -------------------------------------------------------------------------
    # Raw side winrate
    # -------------------------------------------------------------------------
    n = len(rdf)
    blue_wins = rdf["blue_win"].sum()
    red_wins = n - blue_wins
    blue_pct = blue_wins / n * 100
    red_pct = red_wins / n * 100

    print(f"\n{'='*60}")
    print(f"  FIRST STAND 2026 — SIDE WINRATE ANALYSIS")
    print(f"  {n} games")
    print(f"{'='*60}")

    print(f"\n  RAW SIDE WINRATE:")
    print(f"    Blue side: {blue_wins}/{n}  ({blue_pct:.1f}%)")
    print(f"    Red side:  {red_wins}/{n}  ({red_pct:.1f}%)")

    # -------------------------------------------------------------------------
    # ELO-adjusted: does red/blue win more than expected?
    # -------------------------------------------------------------------------
    # Average ELO-expected blue win prob
    avg_elo_blue = rdf["elo_prob_blue"].mean()
    avg_elo_red = 1 - avg_elo_blue

    # Actual vs expected
    actual_blue_wr = rdf["blue_win"].mean()
    actual_red_wr = 1 - actual_blue_wr

    blue_above_expected = (actual_blue_wr - avg_elo_blue) * 100
    red_above_expected = (actual_red_wr - avg_elo_red) * 100

    print(f"\n  ELO-ADJUSTED SIDE ADVANTAGE:")
    print(f"    Avg ELO-expected blue WR: {avg_elo_blue*100:.1f}%")
    print(f"    Actual blue WR:           {actual_blue_wr*100:.1f}%")
    print(f"    Blue above expected:      {blue_above_expected:+.1f}pp")
    print(f"")
    print(f"    Avg ELO-expected red WR:  {avg_elo_red*100:.1f}%")
    print(f"    Actual red WR:            {actual_red_wr*100:.1f}%")
    print(f"    Red above expected:       {red_above_expected:+.1f}pp")

    # -------------------------------------------------------------------------
    # Per-game: how often does red win when ELO says they shouldn't?
    # -------------------------------------------------------------------------
    # For each game, did red win above their ELO expectation?
    rdf["red_win"] = 1 - rdf["blue_win"]
    rdf["elo_prob_red"] = 1 - rdf["elo_prob_blue"]

    # Games where red was the ELO underdog (elo_prob_red < 0.5) but won
    red_underdog = rdf[rdf["elo_prob_red"] < 0.5]
    red_underdog_wins = red_underdog["red_win"].sum()
    red_underdog_n = len(red_underdog)

    blue_underdog = rdf[rdf["elo_prob_blue"] < 0.5]
    blue_underdog_wins = blue_underdog["blue_win"].sum()
    blue_underdog_n = len(blue_underdog)

    print(f"\n  UPSET RATES BY SIDE:")
    if red_underdog_n > 0:
        print(f"    Red as underdog:  won {int(red_underdog_wins)}/{red_underdog_n} "
              f"({red_underdog_wins/red_underdog_n*100:.1f}%)")
    if blue_underdog_n > 0:
        print(f"    Blue as underdog: won {int(blue_underdog_wins)}/{blue_underdog_n} "
              f"({blue_underdog_wins/blue_underdog_n*100:.1f}%)")

    # -------------------------------------------------------------------------
    # Residual analysis: average (actual - expected) per side
    # -------------------------------------------------------------------------
    rdf["blue_residual"] = rdf["blue_win"] - rdf["elo_prob_blue"]
    rdf["red_residual"] = rdf["red_win"] - rdf["elo_prob_red"]

    avg_blue_resid = rdf["blue_residual"].mean()
    avg_red_resid = rdf["red_residual"].mean()

    print(f"\n  AVERAGE RESIDUAL (actual - ELO expected):")
    print(f"    Blue side: {avg_blue_resid:+.4f}  (positive = wins more than expected)")
    print(f"    Red side:  {avg_red_resid:+.4f}  (positive = wins more than expected)")

    # Statistical significance
    from scipy import stats
    # One-sample t-test: is the red residual significantly > 0?
    if len(rdf) > 2:
        t_stat, p_val = stats.ttest_1samp(rdf["red_residual"], 0)
        print(f"\n  SIGNIFICANCE TEST (red residual != 0):")
        print(f"    t-stat: {t_stat:.3f}, p-value: {p_val:.4f}")
        if p_val < 0.05:
            print(f"    → Statistically significant at p<0.05")
        else:
            print(f"    → NOT statistically significant (p={p_val:.3f})")

    print()


if __name__ == "__main__":
    main()
