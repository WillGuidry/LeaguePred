"""
Simulate match predictions using domestic Elo + international results.

Loads all match data, builds posterior Elo from domestic leagues,
incorporates FST international results, then predicts cross-regional matchups.

Usage:
    python simulate_matches.py
"""

import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from src.data.loader import load_oracle_csv
from src.elo.engine import EloEngine
from src.config import REGIONAL_ELO_PRIORS, REGIONAL_ELO_DEFAULT
from predict import print_prediction


DATA_PATH = "data/raw/international_matches.csv"

# International event league codes in this dataset
INTERNATIONAL_LEAGUES = {"FST"}

# Matches to predict
MATCHES = [
    ("BNK FEARX", "G2 Esports"),
    ("JD Gaming", "LYON"),
]


def main():
    # Load all match data
    df = load_oracle_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    # Split into domestic and international
    intl_mask = df["league"].isin(INTERNATIONAL_LEAGUES)
    domestic_df = df[~intl_mask].copy()
    intl_df = df[intl_mask].copy()

    print(f"\nDomestic games: {len(domestic_df)}")
    print(f"International games (FST): {len(intl_df)}")

    # ── Phase 1: Build domestic Elo ──────────────────────────────────
    engine = EloEngine(k_factor=32.0)
    team_leagues = {}

    for _, row in domestic_df.iterrows():
        team_a = row["team_a"]
        team_b = row["team_b"]
        winner = row["winner"]
        league = row.get("league", None)

        # Initialize teams at regional prior
        if team_a not in engine.ratings:
            prior = REGIONAL_ELO_PRIORS.get(league, REGIONAL_ELO_DEFAULT) if league else engine.default_elo
            engine.add_team(team_a, elo=prior)
        if team_b not in engine.ratings:
            prior = REGIONAL_ELO_PRIORS.get(league, REGIONAL_ELO_DEFAULT) if league else engine.default_elo
            engine.add_team(team_b, elo=prior)

        if league:
            team_leagues[team_a] = league
            team_leagues[team_b] = league

        if pd.notna(winner):
            engine.process_match(team_a, team_b, winner)

    # ── Phase 2: Process international (FST) matches ─────────────────
    # Use a blended K-factor: international games update at 1.5x domestic
    # to give proper weight to cross-regional results
    intl_k = 48.0

    print(f"\n{'='*60}")
    print("  PROCESSING INTERNATIONAL (FST) MATCHES")
    print(f"{'='*60}")

    for _, row in intl_df.iterrows():
        team_a = row["team_a"]
        team_b = row["team_b"]
        winner = row["winner"]

        # Ensure teams exist (they should from domestic phase)
        if team_a not in engine.ratings:
            league = team_leagues.get(team_a)
            prior = REGIONAL_ELO_PRIORS.get(league, REGIONAL_ELO_DEFAULT) if league else engine.default_elo
            engine.add_team(team_a, elo=prior)
        if team_b not in engine.ratings:
            league = team_leagues.get(team_b)
            prior = REGIONAL_ELO_PRIORS.get(league, REGIONAL_ELO_DEFAULT) if league else engine.default_elo
            engine.add_team(team_b, elo=prior)

        if pd.notna(winner):
            elo_a_before = engine.get_rating(team_a)
            elo_b_before = engine.get_rating(team_b)

            # Temporarily override K-factor for international matches
            old_k = engine.k_factor
            engine.k_factor = intl_k
            engine.process_match(team_a, team_b, winner)
            engine.k_factor = old_k

            elo_a_after = engine.get_rating(team_a)
            elo_b_after = engine.get_rating(team_b)
            print(f"  {winner:<25} beat {team_a if winner == team_b else team_b:<25}"
                  f"  [{team_a}: {elo_a_before:.0f}->{elo_a_after:.0f}]"
                  f"  [{team_b}: {elo_b_before:.0f}->{elo_b_after:.0f}]")

    # ── Phase 3: Print posterior ratings for target teams ─────────────
    target_teams = set()
    for a, b in MATCHES:
        target_teams.add(a)
        target_teams.add(b)

    print(f"\n{'='*60}")
    print("  POSTERIOR ELO RATINGS (Domestic + International)")
    print(f"{'='*60}")
    print(f"  {'Team':<28} {'League':<7} {'Elo':>7}")
    print(f"  {'-'*45}")
    for team in sorted(target_teams):
        league = team_leagues.get(team, "?")
        elo = engine.get_rating(team)
        prior = REGIONAL_ELO_PRIORS.get(league, REGIONAL_ELO_DEFAULT)
        delta = elo - prior
        print(f"  {team:<28} {league:<7} {elo:>7.0f}  (prior: {prior}, {delta:+.0f})")

    # ── Phase 4: Predict matchups + Bo5 series odds ─────────────────
    print(f"\n{'='*60}")
    print("  MATCH PREDICTIONS + Bo5 SERIES ODDS")
    print(f"{'='*60}")

    for team_a, team_b in MATCHES:
        pred = engine.predict(team_a, team_b)
        pred["league_a"] = team_leagues.get(team_a, "?")
        pred["league_b"] = team_leagues.get(team_b, "?")
        pred["cross_regional"] = False
        print_prediction(pred)

        p = pred["win_prob_a"]
        q = 1 - p
        print_bo5_odds(team_a, team_b, p, q)


def print_bo5_odds(team_a: str, team_b: str, p: float, q: float):
    """
    Print Bo5 series length probabilities.

    In a Bo5, the possible outcomes are:
      3-0:  p^3                      or  q^3
      3-1:  C(3,1) * p^3 * q        or  C(3,1) * q^3 * p
      3-2:  C(4,2) * p^3 * q^2      or  C(4,2) * q^3 * p^2

    The deciding game must be won by the series winner, so the
    combinatorics count how many ways the loser wins exactly k
    of the first (2+k) games.
    """
    # Team A wins
    a_30 = p**3
    a_31 = 3 * (p**3) * q
    a_32 = 6 * (p**3) * (q**2)

    # Team B wins
    b_30 = q**3
    b_31 = 3 * (q**3) * p
    b_32 = 6 * (q**3) * (p**2)

    # Series length probabilities
    p_3games = a_30 + b_30
    p_4games = a_31 + b_31
    p_5games = a_32 + b_32

    # Series winner probabilities
    p_a_wins = a_30 + a_31 + a_32
    p_b_wins = b_30 + b_31 + b_32

    print(f"  {'─'*56}")
    print(f"  Bo5 SERIES BREAKDOWN")
    print(f"  {'─'*56}")
    print(f"  Series winner:  {team_a}: {p_a_wins:.1%}   {team_b}: {p_b_wins:.1%}")
    print()
    print(f"  {'Score':<12} {team_a + ' wins':<20} {team_b + ' wins':<20} {'Total':<10}")
    print(f"  {'─'*56}")
    print(f"  {'3-0':<12} {a_30:>8.1%}{'':12} {b_30:>8.1%}{'':12} {p_3games:>8.1%}")
    print(f"  {'3-1':<12} {a_31:>8.1%}{'':12} {b_31:>8.1%}{'':12} {p_4games:>8.1%}")
    print(f"  {'3-2':<12} {a_32:>8.1%}{'':12} {b_32:>8.1%}{'':12} {p_5games:>8.1%}")
    print()
    print(f"  Goes to Game 4+:  {p_4games + p_5games:.1%}")
    print(f"  Goes to Game 5:   {p_5games:.1%}")
    print()


if __name__ == "__main__":
    main()
