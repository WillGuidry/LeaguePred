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

import json
import os

import numpy as np
import pandas as pd
from typing import Optional

from src.elo.engine import EloEngine
from datetime import timedelta
from src.config import (
    REGIONAL_ELO_PRIORS, REGIONAL_ELO_DEFAULT,
    INTL_DECAY_BETWEEN_EVENTS, INTL_GAMES_FULL_TRUST, INTL_MAX_WEIGHT,
)


class PersistentInternationalElo:
    """
    Maintains a career international Elo for each team across tournaments.

    Problem:
        The original TournamentPredictor discards all international Elo
        between events. BLG's 20+ international games count for nothing
        when predicting the next tournament. A team like FEARX with zero
        international games gets the same treatment.

    Solution:
        Track a running international Elo and career game count per team.
        Between tournaments, decay the international Elo toward the
        regional prior (not a full reset). When computing a team's prior
        for the next event, blend between regional prior and career
        international Elo based on how many international games they've
        played.

    Usage:
        persistent = PersistentInternationalElo()
        # After each tournament:
        persistent.absorb_tournament(predictor)
        persistent.decay()
        # Before next tournament:
        predictor = TournamentPredictor(..., persistent_intl=persistent)
    """

    def __init__(
        self,
        decay_factor: float = INTL_DECAY_BETWEEN_EVENTS,
        games_full_trust: int = INTL_GAMES_FULL_TRUST,
        max_weight: float = INTL_MAX_WEIGHT,
        max_age_days: int = 730,  # 2 years
    ):
        self.decay_factor = decay_factor
        self.games_full_trust = games_full_trust
        self.max_weight = max_weight
        self.max_age_days = max_age_days

        # Career international Elo per team
        self.ratings: dict = {}  # {team: elo}
        # Timestamped game records: {team: [(date, game_count_in_event)]}
        self._game_history: dict = {}  # {team: [(pd.Timestamp, int)]}
        # Each team's regional prior (cached for decay target)
        self._regional_priors: dict = {}  # {team: prior_elo}
        # Current reference date (updated as tournaments are absorbed)
        self._current_date: pd.Timestamp = None

    def _recent_game_count(self, team: str) -> int:
        """Count international games within the max_age_days window."""
        history = self._game_history.get(team, [])
        if not history or self._current_date is None:
            return 0
        cutoff = self._current_date - timedelta(days=self.max_age_days)
        return sum(count for date, count in history if date >= cutoff)

    def get_intl_weight(self, team: str) -> float:
        """
        How much to trust a team's international Elo vs regional prior.

        Only counts games from the last max_age_days (default 2 years).
        Returns 0.0 for teams with no recent international games,
        ramps linearly to max_weight at games_full_trust games.
        """
        games = self._recent_game_count(team)
        if games == 0:
            return 0.0
        raw = min(games / self.games_full_trust, 1.0)
        return raw * self.max_weight

    def get_blended_prior(
        self, team: str, regional_prior: float, domestic_deviation: float
    ) -> float:
        """
        Compute a team's prior Elo blending international experience
        with regional prior.

        For a team with many international games (e.g. BLG):
            prior ≈ international_elo + domestic_deviation (regional prior fades)
        For a team with zero international games (e.g. FEARX):
            prior = regional_prior + domestic_deviation (unchanged from before)
        """
        w = self.get_intl_weight(team)

        if w == 0.0 or team not in self.ratings:
            return regional_prior + domestic_deviation

        intl_base = self.ratings[team]
        blended_base = (1 - w) * regional_prior + w * intl_base
        return blended_base + domestic_deviation

    def absorb_tournament(self, predictor: "TournamentPredictor", tournament_date: pd.Timestamp = None) -> None:
        """
        After a tournament ends, absorb each team's final tournament Elo
        into the persistent store. Games older than max_age_days will be
        excluded from trust weight calculations.
        """
        if tournament_date is not None:
            self._current_date = tournament_date

        for team, tourn_elo in predictor.tournament_engine.ratings.items():
            games_this_event = predictor.games_played.get(team, 0)
            if games_this_event == 0:
                continue

            # Record timestamped game history
            if team not in self._game_history:
                self._game_history[team] = []
            if self._current_date is not None:
                self._game_history[team].append((self._current_date, games_this_event))

            # Use only recent games for weighting the Elo merge
            recent_prev = self._recent_game_count(team) - games_this_event
            recent_total = recent_prev + games_this_event

            if team in self.ratings and recent_prev > 0:
                prev_weight = recent_prev / recent_total
                new_weight = games_this_event / recent_total
                self.ratings[team] = (
                    prev_weight * self.ratings[team] + new_weight * tourn_elo
                )
            else:
                # No recent history or first event — use tournament Elo directly
                self.ratings[team] = tourn_elo

            # Cache regional prior for decay
            league = predictor.team_leagues.get(team)
            if league:
                shrunk = 1200 + predictor.prior_shrinkage * (
                    REGIONAL_ELO_PRIORS.get(league, REGIONAL_ELO_DEFAULT) - 1200
                )
                self._regional_priors[team] = shrunk

    def decay(self) -> None:
        """
        Between tournaments, regress international Elo toward the
        regional prior. This prevents stale ratings from dominating.
        """
        for team in self.ratings:
            target = self._regional_priors.get(team, 1200)
            self.ratings[team] = (
                self.ratings[team]
                + self.decay_factor * (target - self.ratings[team])
            )

    def save(self, path: str) -> None:
        """Save persistent international Elo state to a JSON file."""
        # Convert Timestamps to ISO strings for JSON serialization
        serializable_history = {}
        for team, entries in self._game_history.items():
            serializable_history[team] = [
                [dt.isoformat() if hasattr(dt, "isoformat") else str(dt), count]
                for dt, count in entries
            ]

        state = {
            "decay_factor": self.decay_factor,
            "games_full_trust": self.games_full_trust,
            "max_weight": self.max_weight,
            "max_age_days": self.max_age_days,
            "ratings": self.ratings,
            "game_history": serializable_history,
            "regional_priors": self._regional_priors,
            "current_date": self._current_date.isoformat() if self._current_date else None,
        }
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(state, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "PersistentInternationalElo":
        """Load persistent international Elo state from a JSON file."""
        with open(path) as f:
            state = json.load(f)
        obj = cls(
            decay_factor=state["decay_factor"],
            games_full_trust=state["games_full_trust"],
            max_weight=state["max_weight"],
            max_age_days=state["max_age_days"],
        )
        obj.ratings = state["ratings"]
        obj._regional_priors = state["regional_priors"]
        obj._current_date = (
            pd.Timestamp(state["current_date"]) if state["current_date"] else None
        )
        # Restore game history with Timestamps
        for team, entries in state["game_history"].items():
            obj._game_history[team] = [
                (pd.Timestamp(dt_str), count) for dt_str, count in entries
            ]
        return obj

    def get_team_info(self, team: str) -> dict:
        """Debug info for a team's persistent international state."""
        total_games = sum(c for _, c in self._game_history.get(team, []))
        recent_games = self._recent_game_count(team)
        return {
            "team": team,
            "intl_elo": round(self.ratings.get(team, 0), 1),
            "intl_games": recent_games,
            "intl_games_total": total_games,
            "intl_weight": round(self.get_intl_weight(team), 2),
            "regional_prior": round(self._regional_priors.get(team, 0), 1),
        }


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
        prior_shrinkage: float = 0.6,
        persistent_intl: PersistentInternationalElo = None,
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
            prior_shrinkage: Compress regional gaps toward the global mean.
                             1.0 = use full Lolesports gaps (overconfident).
                             0.0 = ignore regional gaps entirely.
                             0.6 = use 60% of the gap (calibrated via backtest on
                             1,036 international games to minimize log loss while
                             keeping calibration error under 0.06).
            persistent_intl: Persistent international Elo store. When provided,
                             teams with international experience will use their
                             career international Elo instead of the regional
                             prior, with weight proportional to games played.
        """
        self.domestic_engine = domestic_engine
        self.team_leagues = team_leagues
        self.k_factor = k_factor
        self.scale_factor = scale_factor
        self.blend_games = blend_games
        self.max_tournament_weight = max_tournament_weight
        self.prior_shrinkage = prior_shrinkage
        self.persistent_intl = persistent_intl

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

        Method:
            1. Get team's deviation from their league average (within-league rank)
            2. Get regional strength from Lolesports scores
            3. Shrink the regional gap toward the global mean
               (because raw Lolesports gaps produce overconfident predictions)
            4. If a persistent international Elo exists for this team, blend
               between the regional prior and their career international Elo
               based on how many international games they've played.
               Teams with 20+ intl games mostly use their proven intl Elo.
               Teams with 0 intl games use pure regional prior (unchanged).
            5. Prior = blended base + within-league deviation

        The shrinkage reflects that international play has more variance
        than regional strength scores imply. Mid-tier regions upset
        more than the raw gaps suggest (~31% for Tier2 vs Tier3).
        """
        if team in self.prior_elo:
            return self.prior_elo[team]

        league = self.team_leagues.get(team)
        regional_strength = REGIONAL_ELO_PRIORS.get(league, REGIONAL_ELO_DEFAULT)

        # Shrink regional strength toward global mean
        global_mean = 1200  # Approximate midpoint of all regional priors
        shrunk_regional = global_mean + self.prior_shrinkage * (regional_strength - global_mean)

        if team in self.domestic_engine.ratings:
            domestic_elo = self.domestic_engine.ratings[team]
            league_avg = self._league_avg(league)
            deviation = domestic_elo - league_avg
        else:
            deviation = 0.0

        # Blend with persistent international Elo if available
        if self.persistent_intl is not None:
            prior = self.persistent_intl.get_blended_prior(
                team, shrunk_regional, deviation
            )
        else:
            prior = shrunk_regional + deviation

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
    prior_shrinkage: float = 0.55,
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

    # Persistent international Elo — carries across tournaments
    persistent_intl = PersistentInternationalElo()

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

        # Initialize tournament predictor with persistent international Elo
        predictor = TournamentPredictor(
            domestic_engine=domestic_engine,
            team_leagues=team_leagues,
            k_factor=k_factor_tournament,
            blend_games=blend_games,
            max_tournament_weight=max_tournament_weight,
            prior_shrinkage=prior_shrinkage,
            persistent_intl=persistent_intl,
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

        # Absorb this tournament's results into persistent international Elo
        tournament_date = t_df["date"].max()
        persistent_intl.absorb_tournament(predictor, tournament_date=tournament_date)
        persistent_intl.decay()

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
