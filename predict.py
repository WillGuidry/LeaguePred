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
import json
import os
import sys
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

from src.data.loader import load_oracle_csv
from src.elo.engine import EloEngine
from src.elo.international import TournamentPredictor, PersistentInternationalElo
from src.config import REGIONAL_ELO_PRIORS, REGIONAL_ELO_DEFAULT, INTERNATIONAL_EVENTS

# Default paths for cached model state
CACHE_DIR = "data/processed"
ENGINE_CACHE = os.path.join(CACHE_DIR, "elo_engine.json")
INTL_CACHE = os.path.join(CACHE_DIR, "intl_elo.json")
LEAGUES_CACHE = os.path.join(CACHE_DIR, "team_leagues.json")
PREDICTIONS_LOG = os.path.join(CACHE_DIR, "predictions.jsonl")


# Major leagues to include when building Elo ratings
MAJOR_LEAGUES = ["LCK", "LPL", "LEC", "LCP", "LCS", "CBLOL", "MSI", "WLDs", "PCS"]


def build_engine(
    data_path: str = "data/raw/all_matches.csv",
    leagues: list = None,
    k_factor: float = 32.0,
) -> tuple:
    """
    Build an Elo engine from historical data, processing international
    events through PersistentInternationalElo to build career international
    ratings.

    Returns:
        (domestic_engine, team_leagues, persistent_intl)
    """
    if leagues is None:
        leagues = MAJOR_LEAGUES

    df = load_oracle_csv(data_path, leagues=leagues)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    # Separate domestic and international games
    intl_mask = df["league"].isin(INTERNATIONAL_EVENTS)
    intl_df = df[intl_mask].copy()
    domestic_df = df[~intl_mask].copy()

    # Build domestic Elo
    domestic_engine = EloEngine(k_factor=k_factor)
    team_leagues = {}

    for _, row in domestic_df.iterrows():
        team_a = row["team_a"]
        team_b = row["team_b"]
        winner = row["winner"]
        league = row.get("league", None)

        domestic_engine.add_team(team_a)
        domestic_engine.add_team(team_b)

        if league:
            team_leagues[team_a] = league
            team_leagues[team_b] = league

        if pd.notna(winner):
            domestic_engine.process_match(team_a, team_b, winner)

    # Process international events through PersistentInternationalElo
    persistent_intl = PersistentInternationalElo()

    if len(intl_df) > 0:
        # Find tournament boundaries (gaps > 14 days)
        intl_dates = intl_df["date"].values
        gaps = pd.Series(intl_dates).diff().dt.days
        tournament_starts = [0] + list(gaps[gaps > 14].index)

        tournaments = []
        for i, start_idx in enumerate(tournament_starts):
            end_idx = tournament_starts[i + 1] if i + 1 < len(tournament_starts) else len(intl_df)
            tournaments.append(intl_df.iloc[start_idx:end_idx])

        for t_df in tournaments:
            if len(t_df) < 3:
                continue

            # Map international teams to their domestic leagues
            for _, row in t_df.iterrows():
                for team in [row["team_a"], row["team_b"]]:
                    if team not in team_leagues:
                        team_domestic = domestic_df[
                            (domestic_df["team_a"] == team) | (domestic_df["team_b"] == team)
                        ]
                        if len(team_domestic) > 0:
                            last_row = team_domestic.iloc[-1]
                            team_leagues[team] = last_row["league"]

            predictor = TournamentPredictor(
                domestic_engine=domestic_engine,
                team_leagues=team_leagues,
                persistent_intl=persistent_intl,
            )

            for _, row in t_df.iterrows():
                winner = row["winner"]
                if pd.notna(winner):
                    predictor.update(row["team_a"], row["team_b"], winner)

            tournament_date = t_df["date"].max()
            persistent_intl.absorb_tournament(predictor, tournament_date=tournament_date)
            persistent_intl.decay()

    # Set current date for recency calculations on future predictions
    if persistent_intl._current_date is None:
        persistent_intl._current_date = pd.Timestamp.now()

    # Save to cache so subsequent runs don't rebuild from scratch
    save_engine(domestic_engine, team_leagues, persistent_intl)

    return domestic_engine, team_leagues, persistent_intl


def save_engine(
    engine: EloEngine,
    team_leagues: dict,
    persistent_intl: PersistentInternationalElo,
) -> None:
    """Save engine state to disk for reuse."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    engine.save(ENGINE_CACHE)
    persistent_intl.save(INTL_CACHE)
    with open(LEAGUES_CACHE, "w") as f:
        json.dump(team_leagues, f, indent=2)
    print(f"  [saved model state to {CACHE_DIR}/]")


def load_engine() -> tuple:
    """
    Load previously saved engine state from disk.
    Returns (engine, team_leagues, persistent_intl) or None if no cache exists.
    """
    if not (os.path.exists(ENGINE_CACHE) and os.path.exists(INTL_CACHE)
            and os.path.exists(LEAGUES_CACHE)):
        return None
    try:
        engine = EloEngine.load(ENGINE_CACHE)
        persistent_intl = PersistentInternationalElo.load(INTL_CACHE)
        with open(LEAGUES_CACHE) as f:
            team_leagues = json.load(f)
        print(f"  [loaded cached model from {CACHE_DIR}/]")
        return engine, team_leagues, persistent_intl
    except Exception as e:
        print(f"  [cache load failed: {e}, rebuilding...]")
        return None


def log_prediction(pred: dict) -> None:
    """Append a prediction to the JSONL log file."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    entry = {**pred, "timestamp": datetime.now().isoformat()}
    with open(PREDICTIONS_LOG, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def predict_international(
    domestic_engine: EloEngine,
    team_a: str,
    team_b: str,
    team_leagues: dict,
    persistent_intl: PersistentInternationalElo,
) -> dict:
    """
    Predict a cross-regional match using persistent international Elo.

    Teams with extensive international history (e.g. BLG) rely mostly
    on their proven international Elo. Teams with no international
    games (e.g. newcomers) fall back to regional priors.
    """
    predictor = TournamentPredictor(
        domestic_engine=domestic_engine,
        team_leagues=team_leagues,
        persistent_intl=persistent_intl,
    )

    pred = predictor.predict(team_a, team_b)

    # Add international experience info
    info_a = persistent_intl.get_team_info(team_a)
    info_b = persistent_intl.get_team_info(team_b)
    pred["intl_games_a"] = info_a["intl_games"]
    pred["intl_games_b"] = info_b["intl_games"]
    pred["intl_weight_a"] = info_a["intl_weight"]
    pred["intl_weight_b"] = info_b["intl_weight"]
    pred["intl_elo_a"] = info_a["intl_elo"]
    pred["intl_elo_b"] = info_b["intl_elo"]
    pred["cross_regional"] = True

    return pred


def _get_league_avg_elo(engine: EloEngine, team_leagues: dict, league: str) -> float:
    """Get the average Elo of all teams in a league."""
    teams = [t for t, lg in team_leagues.items() if lg == league and t in engine.ratings]
    if not teams:
        return engine.default_elo
    return np.mean([engine.ratings[t] for t in teams])


def predict_match(
    domestic_engine: EloEngine,
    team_a: str,
    team_b: str,
    team_leagues: dict,
    international: bool = False,
    persistent_intl: PersistentInternationalElo = None,
) -> dict:
    """
    Predict a single match.
    """
    # Check teams exist
    for team in [team_a, team_b]:
        if team not in domestic_engine.ratings:
            print(f"WARNING: '{team}' not found in ratings. Using default Elo.")
            print(f"  Available teams with similar names:")
            matches = [t for t in domestic_engine.ratings if team.lower() in t.lower()]
            for m in matches[:5]:
                print(f"    - {m} ({domestic_engine.ratings[m]:.0f})")
            if not matches:
                print(f"    (none found)")

    league_a = team_leagues.get(team_a)
    league_b = team_leagues.get(team_b)
    same_league = league_a == league_b

    if international and not same_league and persistent_intl is not None:
        return predict_international(
            domestic_engine, team_a, team_b, team_leagues, persistent_intl
        )
    else:
        pred = domestic_engine.predict(team_a, team_b)
        pred["league_a"] = league_a
        pred["league_b"] = league_b
        pred["cross_regional"] = False
        return pred


def print_prediction(pred: dict) -> None:
    """Pretty-print a match prediction."""
    print(f"\n{'='*60}")
    team_a = pred["team_a"]
    team_b = pred["team_b"]
    prob_a = pred["win_prob_a"]
    prob_b = pred["win_prob_b"]

    league_a = pred.get("league_a", "?")
    league_b = pred.get("league_b", "?")

    print(f"  {team_a} ({league_a}) vs {team_b} ({league_b})")
    print(f"{'='*60}")

    if pred.get("cross_regional"):
        print(f"  Cross-regional adjustment: ON (persistent intl Elo)")
        print(f"  Prior Elo:    {pred.get('prior_elo_a', '?'):>7} vs {pred.get('prior_elo_b', '?'):>7}")
        print(f"  Blended Elo:  {pred.get('blended_elo_a', '?'):>7} vs {pred.get('blended_elo_b', '?'):>7}")

        intl_games_a = pred.get("intl_games_a", 0)
        intl_games_b = pred.get("intl_games_b", 0)
        intl_w_a = pred.get("intl_weight_a", 0)
        intl_w_b = pred.get("intl_weight_b", 0)
        intl_elo_a = pred.get("intl_elo_a", 0)
        intl_elo_b = pred.get("intl_elo_b", 0)
        print(f"  Intl exp:     {intl_games_a:>4} games ({intl_w_a:.0%} intl weight)"
              f"  vs  {intl_games_b:>4} games ({intl_w_b:.0%} intl weight)")
        if intl_elo_a > 0 or intl_elo_b > 0:
            print(f"  Intl Elo:     {intl_elo_a:>7.0f} vs {intl_elo_b:>7.0f}")
    else:
        elo_a = pred.get("elo_a", "?")
        elo_b = pred.get("elo_b", "?")
        diff = pred.get("elo_diff", "?")
        print(f"  Elo:          {elo_a:>7} vs {elo_b:>7}  (diff: {diff:>+})")

    print()

    # Visual bar
    bar_len = 40
    fill_a = int(prob_a * bar_len)
    bar = "\u2588" * fill_a + "\u2591" * (bar_len - fill_a)
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
    parser = argparse.ArgumentParser(description="LeaguePred \u2014 Match Predictions")
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

    parser.add_argument("--rebuild", action="store_true",
                        help="Force rebuild from data (ignore cache)")
    args = parser.parse_args()

    # Try loading cached engine first, rebuild only if needed
    cached = None if args.rebuild else load_engine()
    if cached is not None:
        engine, team_leagues, persistent_intl = cached
    else:
        engine, team_leagues, persistent_intl = build_engine(args.data, k_factor=args.k_factor)

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
                                 team_leagues, international=args.international,
                                 persistent_intl=persistent_intl)
            print_prediction(pred)
            log_prediction(pred)
        return

    if args.team_a and args.team_b:
        pred = predict_match(engine, args.team_a, args.team_b,
                             team_leagues, international=args.international,
                             persistent_intl=persistent_intl)
        print_prediction(pred)
        log_prediction(pred)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
