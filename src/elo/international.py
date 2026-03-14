"""
International Tournament Elo
------------------------------
Handles predictions for cross-regional events (MSI, Worlds, etc.).

Problem:
    Domestic Elo pools are isolated. When T1 (LCK) plays LOUD (CBLOL),
    their raw Elo numbers are not comparable. And past international
    results go stale between events.

Solution:
    Each international tournament gets its own isolated Elo system.
    Teams start at a "global Elo" derived from:
        1. Their regional strength prior (Lolesports scores)
        2. Their within-league standing (deviation from league average)

    As the tournament progresses, predictions blend from this prior
    toward tournament-specific results.

    Between tournaments, the tournament Elo is discarded entirely.
    Each event is a fresh start.

Leakage notes:
    - Regional priors are public pre-tournament information (safe)
    - Domestic Elo is built from pre-tournament games only (safe)
    - Tournament Elo only uses games already played in that event (safe)
"""

import numpy as np
import pandas as pd
from typing import Optional

from src.elo.engine import EloEngine
from src.config import REGIONAL_ELO_PRIORS, REGIONAL_ELO_DEFAULT


class TournamentPredictor:
    """
    Manages predictions for a single international tournament.

    Usage:
        predictor = TournamentPredictor(domestic_engine, team_leagues)
        predictor.predict("T1", "LOUD")           # Uses prior only
        predictor.update("T1", "LOUD", "T1")      # Feed in result
        predictor.predict("T1", "G2 Esports")     # Blends prior + evidence
    """

    def __init__(
        self,
        domestic_engine: EloEngine,
        team_leagues: dict,
        k_factor: float = 48.0,
        scale_factor: float = 400.0,
        blend_games: int = 8,
        max_tournament_weight: float = 0.6,
    ):
        """
        Args:
            domestic_engine: Elo engine trained on domestic games.
            team_leagues: {team_name: league_code} mapping.
            k_factor: K-factor for tournament Elo (higher = faster adaptation).
                      Use higher than domestic because few games, high signal.
            scale_factor: Elo scale factor.
            blend_games: Number of games before tournament Elo reaches max weight.
            max_tournament_weight: Maximum weight for tournament Elo in the blend.
                                  0.6 means at most 60% tournament, 40% prior.
                                  We never go full tournament because sample is small.
        """
        self.domestic_engine = domestic_engine
        self.team_leagues = team_leagues
        self.k_factor = k_factor
        self.scale_factor = scale_factor
        self.blend_games = blend_games
        self.max_tournament_weight = max_tournament_weight

        # Tournament-specific Elo engine (starts fresh)
        self.tournament_engine = EloEngine(
            default_elo=1500.0,
            k_factor=k_factor,
            scale_factor=scale_factor,
        )

        # Track games per team in this tournament
        self.games_played = {}  # {team: count}

        # Store each team's global prior Elo
        self.prior_elo = {}

        # Match log
        self.match_log = []

    def _compute_prior_elo(self, team: str) -> float:
        """
        Compute a team's global prior Elo for the tournament.

        Method: team's deviation from league average + regional strength.
        This places all teams on a common global scale.
        """
        if team in self.prior_elo:
            return self.prior_elo[team]

        league = self.team_leagues.get(team)
        regional_strength = REGIONAL_ELO_PRIORS.get(league, REGIONAL_ELO_DEFAULT)

        if team in self.domestic_engine.ratings:
            domestic_elo = self.domestic_engine.ratings[team]
            league_avg = self._league_avg(league)
            deviation = domestic_elo - league_avg
            prior = regional_strength + deviation
        else:
            # Unknown team — use regional strength as-is
            prior = regional_strength

        self.prior_elo[team] = prior
        return prior

    def _league_avg(self, league: str) -> float:
        """Average domestic Elo for a league."""
        teams = [
            t for t, lg in self.team_leagues.items()
            if lg == league and t in self.domestic_engine.ratings
        ]
        if not teams:
            return self.domestic_engine.default_elo
        return np.mean([self.domestic_engine.ratings[t] for t in teams])

    def _blend_weight(self, team: str) -> float:
        """
        How much weight to give tournament Elo vs prior.

        Returns a value between 0.0 (all prior) and max_tournament_weight.
        Ramps linearly with games played.
        """
        games = self.games_played.get(team, 0)
        raw_weight = min(games / self.blend_games, 1.0)
        return raw_weight * self.max_tournament_weight

    def _blended_elo(self, team: str) -> float:
        """
        Get the blended Elo for a team.

        blend = (1 - w) * prior + w * tournament_elo
        """
        prior = self._compute_prior_elo(team)
        w = self._blend_weight(team)

        if team in self.tournament_engine.ratings:
            tournament = self.tournament_engine.ratings[team]
        else:
            tournament = prior  # First game in tournament

        return (1 - w) * prior + w * tournament

    def predict(self, team_a: str, team_b: str) -> dict:
        """
        Predict a match outcome using blended Elo.
        """
        # Initialize tournament Elo from prior if needed
        for team in [team_a, team_b]:
            if team not in self.tournament_engine.ratings:
                prior = self._compute_prior_elo(team)
                self.tournament_engine.add_team(team, elo=prior)

        blended_a = self._blended_elo(team_a)
        blended_b = self._blended_elo(team_b)

        diff = blended_a - blended_b
        prob_a = 1.0 / (1.0 + 10 ** (-diff / self.scale_factor))

        w_a = self._blend_weight(team_a)
        w_b = self._blend_weight(team_b)

        return {
            "team_a": team_a,
            "team_b": team_b,
            "league_a": self.team_leagues.get(team_a, "?"),
            "league_b": self.team_leagues.get(team_b, "?"),
            "prior_elo_a": round(self.prior_elo.get(team_a, 1500), 1),
            "prior_elo_b": round(self.prior_elo.get(team_b, 1500), 1),
            "tournament_elo_a": round(self.tournament_engine.get_rating(team_a), 1),
            "tournament_elo_b": round(self.tournament_engine.get_rating(team_b), 1),
            "blended_elo_a": round(blended_a, 1),
            "blended_elo_b": round(blended_b, 1),
            "blend_weight_a": round(w_a, 2),
            "blend_weight_b": round(w_b, 2),
            "games_a": self.games_played.get(team_a, 0),
            "games_b": self.games_played.get(team_b, 0),
            "elo_diff": round(diff, 1),
            "win_prob_a": round(prob_a, 3),
            "win_prob_b": round(1 - prob_a, 3),
            "predicted_winner": team_a if prob_a >= 0.5 else team_b,
        }

    def update(self, team_a: str, team_b: str, winner: str) -> None:
        """
        Feed a tournament result into the system.
        Call this after each game to update tournament Elo.
        """
        # Initialize if needed
        for team in [team_a, team_b]:
            if team not in self.tournament_engine.ratings:
                prior = self._compute_prior_elo(team)
                self.tournament_engine.add_team(team, elo=prior)

        # Update tournament Elo
        self.tournament_engine.process_match(team_a, team_b, winner)

        # Track games played
        self.games_played[team_a] = self.games_played.get(team_a, 0) + 1
        self.games_played[team_b] = self.games_played.get(team_b, 0) + 1

        # Log
        self.match_log.append({
            "team_a": team_a,
            "team_b": team_b,
            "winner": winner,
            "games_a": self.games_played[team_a],
            "games_b": self.games_played[team_b],
        })

    def get_standings(self) -> list:
        """Get all teams sorted by blended Elo."""
        teams = list(self.tournament_engine.ratings.keys())
        standings = []
        for team in teams:
            standings.append({
                "team": team,
                "league": self.team_leagues.get(team, "?"),
                "blended_elo": round(self._blended_elo(team), 1),
                "prior_elo": round(self.prior_elo.get(team, 1500), 1),
                "tournament_elo": round(self.tournament_engine.ratings[team], 1),
                "blend_weight": round(self._blend_weight(team), 2),
                "games": self.games_played.get(team, 0),
            })
        return sorted(standings, key=lambda x: x["blended_elo"], reverse=True)

    def print_standings(self) -> None:
        """Pretty-print tournament standings."""
        standings = self.get_standings()
        print(f"\n{'='*75}")
        print(f"  TOURNAMENT STANDINGS (blended Elo)")
        print(f"{'='*75}")
        print(f"  {'Team':<25} {'League':<6} {'Blend':>6} {'Prior':>6} {'Tourn':>6} {'W':>5} {'GP':>3}")
        print(f"  {'-'*70}")
        for s in standings:
            print(
                f"  {s['team']:<25} {s['league']:<6} "
                f"{s['blended_elo']:>6.0f} {s['prior_elo']:>6.0f} "
                f"{s['tournament_elo']:>6.0f} {s['blend_weight']:>5.0%} "
                f"{s['games']:>3}"
            )
        print()


def backtest_international(
    all_data_path: str = "data/raw/all_matches.csv",
    k_factor_domestic: float = 32.0,
    k_factor_tournament: float = 48.0,
    blend_games: int = 8,
    max_tournament_weight: float = 0.6,
) -> dict:
    """
    Backtest the tournament predictor against historical international events.

    For each tournament:
        1. Build domestic Elo from all games BEFORE the tournament
        2. Initialize TournamentPredictor
        3. For each game in the tournament:
            a. Predict BEFORE seeing result
            b. Update with result
        4. Evaluate predictions

    Also compares against:
        - Pure regional priors (no blending)
        - Flat domestic Elo (no regional adjustment)
    """
    from src.data.loader import load_oracle_csv
    from src.evaluation.metrics import log_loss, brier_score, accuracy, full_report

    # Load all major league data
    major_leagues = ["LCK", "LPL", "LEC", "LCP", "LCS", "CBLOL", "MSI", "WLDs", "PCS"]
    all_df = load_oracle_csv(all_data_path, leagues=major_leagues)
    all_df["date"] = pd.to_datetime(all_df["date"])
    all_df = all_df.sort_values("date").reset_index(drop=True)

    # Identify international games
    intl_mask = all_df["league"].isin(["MSI", "WLDs"])
    intl_df = all_df[intl_mask].copy()
    domestic_df = all_df[~intl_mask].copy()

    # Find tournament boundaries (gaps > 14 days between international games)
    intl_dates = intl_df["date"].values
    gaps = pd.Series(intl_dates).diff().dt.days
    tournament_starts = [0] + list(gaps[gaps > 14].index)

    tournaments = []
    for i, start_idx in enumerate(tournament_starts):
        if i + 1 < len(tournament_starts):
            end_idx = tournament_starts[i + 1]
        else:
            end_idx = len(intl_df)
        tournaments.append(intl_df.iloc[start_idx:end_idx])

    print(f"Found {len(tournaments)} international tournaments")

    # Backtest results
    all_preds_blended = []
    all_preds_prior_only = []
    all_preds_flat = []
    all_actuals = []

    for t_idx, t_df in enumerate(tournaments):
        if len(t_df) < 5:
            continue

        t_start = t_df["date"].min()
        t_event = f"{t_df['league'].iloc[0]} {t_start.strftime('%Y-%m')}"
        print(f"\n  Tournament: {t_event} ({len(t_df)} games)")

        # Build domestic Elo from all domestic games BEFORE this tournament
        pre_domestic = domestic_df[domestic_df["date"] < t_start]
        domestic_engine = EloEngine(k_factor=k_factor_domestic)
        team_leagues = {}

        for _, row in pre_domestic.iterrows():
            domestic_engine.add_team(row["team_a"])
            domestic_engine.add_team(row["team_b"])
            team_leagues[row["team_a"]] = row["league"]
            team_leagues[row["team_b"]] = row["league"]
            if pd.notna(row["winner"]):
                domestic_engine.process_match(row["team_a"], row["team_b"], row["winner"])

        # Also include team_leagues from tournament teams
        for _, row in t_df.iterrows():
            # For international games, use the team's domestic league
            for team in [row["team_a"], row["team_b"]]:
                if team not in team_leagues:
                    # Try to find from domestic data
                    team_domestic = domestic_df[
                        (domestic_df["team_a"] == team) | (domestic_df["team_b"] == team)
                    ]
                    if len(team_domestic) > 0:
                        last_row = team_domestic.iloc[-1]
                        if last_row["team_a"] == team:
                            team_leagues[team] = last_row["league"]
                        else:
                            team_leagues[team] = last_row["league"]

        # Initialize tournament predictor
        predictor = TournamentPredictor(
            domestic_engine=domestic_engine,
            team_leagues=team_leagues,
            k_factor=k_factor_tournament,
            blend_games=blend_games,
            max_tournament_weight=max_tournament_weight,
        )

        # Process each game
        for _, row in t_df.iterrows():
            team_a = row["team_a"]
            team_b = row["team_b"]
            winner = row["winner"]

            if pd.isna(winner):
                continue

            actual = 1 if winner == team_a else 0

            # Blended prediction
            pred = predictor.predict(team_a, team_b)
            all_preds_blended.append(pred["win_prob_a"])

            # Prior-only prediction (no blending, just regional + domestic deviation)
            prior_a = predictor._compute_prior_elo(team_a)
            prior_b = predictor._compute_prior_elo(team_b)
            prior_diff = prior_a - prior_b
            prior_prob = 1.0 / (1.0 + 10 ** (-prior_diff / 400))
            all_preds_prior_only.append(prior_prob)

            # Flat domestic Elo (no regional adjustment)
            flat_a = domestic_engine.get_rating(team_a)
            flat_b = domestic_engine.get_rating(team_b)
            flat_diff = flat_a - flat_b
            flat_prob = 1.0 / (1.0 + 10 ** (-flat_diff / 400))
            all_preds_flat.append(flat_prob)

            all_actuals.append(actual)

            # Update tournament Elo
            predictor.update(team_a, team_b, winner)

    # Final comparison
    y_true = np.array(all_actuals)

    print(f"\n{'='*70}")
    print(f"INTERNATIONAL BACKTEST RESULTS ({len(y_true)} games)")
    print(f"{'='*70}")

    from src.evaluation.metrics import compare_models
    results = compare_models(y_true, {
        "Blended (prior + tournament)": np.array(all_preds_blended),
        "Prior only (regional + domestic)": np.array(all_preds_prior_only),
        "Flat domestic Elo (no adjustment)": np.array(all_preds_flat),
        "Coin flip (0.5)": np.full(len(y_true), 0.5),
    })

    # Also show calibration for the blended model
    blended_report = full_report(
        y_true, np.array(all_preds_blended),
        model_name="Blended Tournament Model",
    )

    return {
        "comparison": results,
        "blended_report": blended_report,
        "n_tournaments": len([t for t in tournaments if len(t) >= 5]),
        "n_games": len(y_true),
    }
