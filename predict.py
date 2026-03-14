"""
Predict Match Outcomes
-----------------------
Generate win probability predictions for upcoming matches.

Usage:
    # Predict a single match
    python predict.py --team-a "T1" --team-b "Gen.G"

    # Predict with cross-regional adjustment (for international events)
    python predict.py --team-a "T1" --team-b "LOUD" --international

    # Show current rankings
    python predict.py --rankings

    # Show rankings for a specific league
    python predict.py --rankings --leagues LCK

    # Predict from a schedule file
    python predict.py --schedule schedule.csv
"""

import argparse
import sys
import pandas as pd
import numpy as np
from typing import Optional

from src.data.loader import load_oracle_csv
from src.elo.engine import EloEngine
from src.config import REGIONAL_ELO_PRIORS, REGIONAL_ELO_DEFAULT


# Major leagues to include when building Elo ratings
MAJOR_LEAGUES = ["LCK", "LPL", "LEC", "LCP", "LCS", "CBLOL", "MSI", "WLDs", "PCS"]


def build_engine(
    data_path: str = "data/raw/all_matches.csv",
    leagues: list = None,
    k_factor: float = 32.0,
) -> tuple:
    """
    Build an Elo engine from historical data.

    Returns:
        (engine, team_leagues) — the engine with current ratings,
        and a dict mapping team -> most recent league.
    """
    if leagues is None:
        leagues = MAJOR_LEAGUES

    df = load_oracle_csv(data_path, leagues=leagues)
    df = df.sort_values("date").reset_index(drop=True)

    engine = EloEngine(k_factor=k_factor)
    team_leagues = {}

    for _, row in df.iterrows():
        team_a = row["team_a"]
        team_b = row["team_b"]
        winner = row["winner"]
        league = row.get("league", None)

        engine.add_team(team_a)
        engine.add_team(team_b)

        if league:
            team_leagues[team_a] = league
            team_leagues[team_b] = league

        if pd.notna(winner):
            engine.process_match(team_a, team_b, winner)

    return engine, team_leagues


def cross_regional_adjustment(
    engine: EloEngine,
    team_a: str,
    team_b: str,
    team_leagues: dict,
) -> dict:
    """
    Adjust Elo predictions for cross-regional matchups.

    The problem: within-league Elo pools are separate. An LCK team at 1700
    and a CBLOL team at 1700 are NOT the same strength.

    The fix: use Lolesports regional strength scores to create a
    "global Elo" that accounts for league-pool differences.

    Method:
        1. Get each team's within-league Elo
        2. Center each league's Elo pool around its regional strength score
        3. Predict using the adjusted "global" Elo values

    This is equivalent to asking:
        "If we placed these two teams on a common scale, what would
        the Elo gap be?"
    """
    league_a = team_leagues.get(team_a)
    league_b = team_leagues.get(team_b)

    elo_a = engine.get_rating(team_a)
    elo_b = engine.get_rating(team_b)

    # If same league, no adjustment needed
    if league_a == league_b:
        return engine.predict(team_a, team_b)

    # Get the league-average Elo from our engine (empirical)
    league_avg_a = _get_league_avg_elo(engine, team_leagues, league_a)
    league_avg_b = _get_league_avg_elo(engine, team_leagues, league_b)

    # Get Lolesports regional strength as target center
    regional_a = REGIONAL_ELO_PRIORS.get(league_a, REGIONAL_ELO_DEFAULT)
    regional_b = REGIONAL_ELO_PRIORS.get(league_b, REGIONAL_ELO_DEFAULT)

    # Adjusted Elo = team's deviation from league average + regional center
    # This preserves within-league ordering while shifting the whole pool
    adjusted_a = (elo_a - league_avg_a) + regional_a
    adjusted_b = (elo_b - league_avg_b) + regional_b

    # Calculate expected score using adjusted values
    diff = adjusted_a - adjusted_b
    expected_a = 1.0 / (1.0 + 10 ** (-diff / engine.scale_factor))

    return {
        "team_a": team_a,
        "team_b": team_b,
        "league_a": league_a,
        "league_b": league_b,
        "elo_a_raw": round(elo_a, 1),
        "elo_b_raw": round(elo_b, 1),
        "elo_a_adjusted": round(adjusted_a, 1),
        "elo_b_adjusted": round(adjusted_b, 1),
        "regional_prior_a": regional_a,
        "regional_prior_b": regional_b,
        "elo_diff_raw": round(elo_a - elo_b, 1),
        "elo_diff_adjusted": round(diff, 1),
        "win_prob_a": round(expected_a, 3),
        "win_prob_b": round(1 - expected_a, 3),
        "predicted_winner": team_a if expected_a >= 0.5 else team_b,
        "cross_regional": True,
    }


def _get_league_avg_elo(engine: EloEngine, team_leagues: dict, league: str) -> float:
    """Get the average Elo of all teams in a league."""
    teams = [t for t, lg in team_leagues.items() if lg == league and t in engine.ratings]
    if not teams:
        return engine.default_elo
    return np.mean([engine.ratings[t] for t in teams])


def predict_match(
    engine: EloEngine,
    team_a: str,
    team_b: str,
    team_leagues: dict,
    international: bool = False,
) -> dict:
    """
    Predict a single match.

    Args:
        engine: Trained Elo engine.
        team_a: First team name.
        team_b: Second team name.
        team_leagues: {team: league} mapping.
        international: Whether to apply cross-regional adjustment.
    """
    # Check teams exist
    for team in [team_a, team_b]:
        if team not in engine.ratings:
            print(f"WARNING: '{team}' not found in ratings. Using default Elo.")
            print(f"  Available teams with similar names:")
            matches = [t for t in engine.ratings if team.lower() in t.lower()]
            for m in matches[:5]:
                print(f"    - {m} ({engine.ratings[m]:.0f})")
            if not matches:
                print(f"    (none found)")

    league_a = team_leagues.get(team_a)
    league_b = team_leagues.get(team_b)
    same_league = league_a == league_b

    if international and not same_league:
        return cross_regional_adjustment(engine, team_a, team_b, team_leagues)
    else:
        pred = engine.predict(team_a, team_b)
        pred["league_a"] = league_a
        pred["league_b"] = league_b
        pred["cross_regional"] = False
        return pred


def print_prediction(pred: dict) -> None:
    """Pretty-print a match prediction."""
    print(f"\n{'='*55}")
    team_a = pred["team_a"]
    team_b = pred["team_b"]
    prob_a = pred["win_prob_a"]
    prob_b = pred["win_prob_b"]

    league_a = pred.get("league_a", "?")
    league_b = pred.get("league_b", "?")

    print(f"  {team_a} ({league_a}) vs {team_b} ({league_b})")
    print(f"{'='*55}")

    if pred.get("cross_regional"):
        print(f"  Cross-regional adjustment: ON")
        print(f"  Raw Elo:      {pred['elo_a_raw']:>7.0f} vs {pred['elo_b_raw']:>7.0f}  (diff: {pred['elo_diff_raw']:>+.0f})")
        print(f"  Adjusted Elo: {pred['elo_a_adjusted']:>7.0f} vs {pred['elo_b_adjusted']:>7.0f}  (diff: {pred['elo_diff_adjusted']:>+.0f})")
        print(f"  Regional:     {pred['regional_prior_a']:>7} vs {pred['regional_prior_b']:>7}")
    else:
        elo_a = pred.get("elo_a", "?")
        elo_b = pred.get("elo_b", "?")
        diff = pred.get("elo_diff", "?")
        print(f"  Elo:          {elo_a:>7} vs {elo_b:>7}  (diff: {diff:>+})")

    print()

    # Visual bar
    bar_len = 40
    fill_a = int(prob_a * bar_len)
    bar = "█" * fill_a + "░" * (bar_len - fill_a)
    print(f"  {prob_a:>5.1%} [{bar}] {prob_b:>5.1%}")
    print(f"  {team_a:<20}  {'':>10}  {team_b:>20}")
    print()

    winner = pred["predicted_winner"]
    confidence = max(prob_a, prob_b)
    print(f"  Predicted winner: {winner} ({confidence:.1%})")
    print()


def print_rankings(engine: EloEngine, team_leagues: dict, filter_league: str = None) -> None:
    """Print current Elo rankings."""
    rankings = engine.get_rankings()

    if filter_league:
        rankings = [(t, e) for t, e in rankings if team_leagues.get(t) == filter_league]
        print(f"\n{'='*50}")
        print(f"  {filter_league} Rankings")
    else:
        print(f"\n{'='*50}")
        print(f"  Global Rankings (within-league Elo)")
    print(f"{'='*50}")
    print(f"  {'Rank':<5} {'Team':<28} {'League':<7} {'Elo':>7}")
    print(f"  {'-'*48}")

    for i, (team, elo) in enumerate(rankings[:30], 1):
        league = team_leagues.get(team, "?")
        print(f"  {i:<5} {team:<28} {league:<7} {elo:>7.0f}")
    print()


def main():
    parser = argparse.ArgumentParser(description="LeaguePred — Match Predictions")
    parser.add_argument("--data", default="data/raw/all_matches.csv", help="Path to data")
    parser.add_argument("--team-a", help="First team name")
    parser.add_argument("--team-b", help="Second team name")
    parser.add_argument("--international", action="store_true",
                        help="Apply cross-regional Elo adjustment")
    parser.add_argument("--rankings", action="store_true", help="Show current rankings")
    parser.add_argument("--leagues", nargs="+", default=None,
                        help="Filter rankings by league(s)")
    parser.add_argument("--schedule", help="CSV with team_a, team_b columns to batch predict")
    parser.add_argument("--k-factor", type=float, default=32.0)

    args = parser.parse_args()

    # Build engine
    engine, team_leagues = build_engine(args.data, k_factor=args.k_factor)

    if args.rankings:
        if args.leagues:
            for league in args.leagues:
                print_rankings(engine, team_leagues, filter_league=league)
        else:
            print_rankings(engine, team_leagues)
        return

    if args.schedule:
        schedule = pd.read_csv(args.schedule)
        for _, row in schedule.iterrows():
            pred = predict_match(engine, row["team_a"], row["team_b"],
                                 team_leagues, international=args.international)
            print_prediction(pred)
        return

    if args.team_a and args.team_b:
        pred = predict_match(engine, args.team_a, args.team_b,
                             team_leagues, international=args.international)
        print_prediction(pred)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
