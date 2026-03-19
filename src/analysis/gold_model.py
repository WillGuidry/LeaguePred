"""
Gold Prediction Model + ELO Blend
----------------------------------
Trains gradient boosting models on pre-game features to predict:
    1. Gold differential at 10 minutes
    2. Gold differential at 20 minutes
    3. Win probability

Then blends the gold-model win probability with pure ELO to find the
optimal combination. The gold model captures signal that ELO misses
(player matchups, rolling form, champion comfort, scaling tendencies)
but ELO already captures most of the variance. The blend finds the
marginal gain from gold features on top of ELO.

Blend method: logistic stacking
    logit(p_blend) = β0 + β1·logit(p_elo) + β2·logit(p_gold)
    Trained on the temporal test set via LogisticRegression.

Usage:
    python -m src.analysis.gold_model <csv_path> --predict "Team A" "Team B" [--league LEAGUE]
"""

import math
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score, log_loss, accuracy_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

# Use gradient boosting from sklearn (no xgboost dependency needed)
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from src.analysis.gold_pca import (
    load_player_level,
    extract_game_player_data,
    SimpleTeamElo,
    PlayerEloEngine,
    TeamFeatureTracker,
    PlayerStatsTracker,
    ChampionTracker,
    build_feature_names,
    TEAM_FEATURE_KEYS,
    PLAYER_FEATURE_KEYS,
    CHAMPION_FEATURE_KEYS,
    POSITIONS,
)
from src.config import REGIONAL_ELO_PRIORS, REGIONAL_ELO_DEFAULT


class GoldPredictionModel:
    """
    Predicts gold@10, gold@20, and win probability from pre-game features.

    Processes all historical data chronologically, building rolling features
    with no leakage, then trains gradient boosting models.
    """

    def __init__(self):
        # Trackers (built up during chronological processing)
        self.team_elo = SimpleTeamElo()
        self.player_elo = PlayerEloEngine()
        self.team_tracker = TeamFeatureTracker(window=20)
        self.player_tracker = PlayerStatsTracker(window=20)
        self.champ_tracker = ChampionTracker(min_games=3)

        # Models (trained after processing)
        self.model_gd10 = None
        self.model_gd20 = None
        self.model_win = None
        self.scaler = None
        self.feature_names = build_feature_names()

        # Blend model (ELO + Gold → combined probability)
        self.blend_model = None  # LogisticRegression on logits
        self.blend_metrics = {}  # Backtest results

        # Track team->league mapping
        self.team_leagues = {}

        # Track latest roster per team
        self.team_rosters = {}  # {team: {position: player}}

    def process_and_train(self, csv_path: str, leagues: list = None,
                          min_date: str = None, min_team_games: int = 5):
        """
        Process all data chronologically and train models.

        Returns evaluation metrics on the held-out temporal test set.
        """
        df = load_player_level(csv_path, leagues=leagues, min_date=min_date)
        games, player_games = extract_game_player_data(df)

        # Index player_games by gameid
        pg_index = defaultdict(list)
        for pg in player_games:
            pg_index[pg["gameid"]].append(pg)

        n_features = len(self.feature_names)
        features_list = []
        targets_gd10 = []
        targets_gd20 = []
        targets_win = []
        elo_probs = []  # Pure ELO probability for team_a (captured BEFORE update)
        valid_mask = []

        print(f"Processing {len(games)} games chronologically...")

        for i, game in enumerate(games):
            team_a = game["team_a"]
            team_b = game["team_b"]
            winner = game["winner"]
            league = game.get("league")
            gd10 = game.get("golddiffat10")
            gd15 = game.get("golddiffat15")
            gd20 = game.get("golddiffat20")

            # Track league
            if league:
                self.team_leagues[team_a] = league
                self.team_leagues[team_b] = league

            # Init Elo
            self.team_elo.get_rating(team_a, league)
            self.team_elo.get_rating(team_b, league)

            # Capture pure ELO probability BEFORE update (no leakage)
            elo_prob_a = self.team_elo.expected(team_a, team_b)
            elo_probs.append(elo_prob_a)

            # === PREDICT PHASE ===
            game_player_data = pg_index.get(game["gameid"], [])
            row = self._extract_features(team_a, team_b, league, game_player_data, game)

            features_list.append(row)
            targets_gd10.append(gd10 if gd10 is not None else np.nan)
            targets_gd20.append(gd20 if gd20 is not None else np.nan)
            targets_win.append(1.0 if winner == team_a else 0.0)

            feats_a = self.team_tracker.get_features(team_a)
            feats_b = self.team_tracker.get_features(team_b)
            has_enough = (feats_a["team_n_games"] >= min_team_games and
                          feats_b["team_n_games"] >= min_team_games)
            has_target = gd10 is not None and not (isinstance(gd10, float) and np.isnan(gd10)) and gd10 != 0.0
            valid_mask.append(has_enough and has_target)

            # === UPDATE PHASE ===
            self._update_trackers(game, game_player_data, team_a, team_b, winner)

            # Track rosters
            for pg in game_player_data:
                team = pg["team"]
                if team not in self.team_rosters:
                    self.team_rosters[team] = {}
                self.team_rosters[team][pg["position"]] = pg["player"]

            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(games)} games processed")

        X = np.array(features_list)
        y_gd10 = np.array(targets_gd10)
        y_gd20 = np.array(targets_gd20)
        y_win = np.array(targets_win)
        elo_p = np.array(elo_probs)
        valid = np.array(valid_mask)

        print(f"Total: {len(X)} games, {valid.sum()} valid for training")

        # Filter to valid
        X_v = X[valid]
        y10_v = y_gd10[valid]
        y20_v = y_gd20[valid]
        ywin_v = y_win[valid]
        elo_v = elo_p[valid]

        # Handle NaN in gd20
        gd20_valid = ~np.isnan(y20_v) & (y20_v != 0.0)

        # Temporal split: last 20% for eval
        split_idx = int(len(X_v) * 0.8)
        X_train, X_test = X_v[:split_idx], X_v[split_idx:]
        y10_train, y10_test = y10_v[:split_idx], y10_v[split_idx:]
        y20_train, y20_test = y20_v[:split_idx], y20_v[split_idx:]
        ywin_train, ywin_test = ywin_v[:split_idx], ywin_v[split_idx:]
        elo_train, elo_test = elo_v[:split_idx], elo_v[split_idx:]

        # Standardize
        self.scaler = StandardScaler()
        X_train_s = self.scaler.fit_transform(np.nan_to_num(X_train))
        X_test_s = self.scaler.transform(np.nan_to_num(X_test))

        # Train gold@10 model
        print("\nTraining Gold@10 model...")
        self.model_gd10 = GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=10, random_state=42
        )
        self.model_gd10.fit(X_train_s, y10_train)
        pred_10 = self.model_gd10.predict(X_test_s)
        mae_10 = mean_absolute_error(y10_test, pred_10)
        r2_10 = r2_score(y10_test, pred_10)
        print(f"  Gold@10 — MAE: {mae_10:.0f}g, R²: {r2_10:.4f}")

        # Train gold@20 model
        print("Training Gold@20 model...")
        gd20_train_mask = ~np.isnan(y20_train) & (y20_train != 0.0)
        gd20_test_mask = ~np.isnan(y20_test) & (y20_test != 0.0)
        self.model_gd20 = GradientBoostingRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=10, random_state=42
        )
        self.model_gd20.fit(X_train_s[gd20_train_mask], y20_train[gd20_train_mask])
        if gd20_test_mask.sum() > 0:
            pred_20 = self.model_gd20.predict(X_test_s[gd20_test_mask])
            mae_20 = mean_absolute_error(y20_test[gd20_test_mask], pred_20)
            r2_20 = r2_score(y20_test[gd20_test_mask], pred_20)
            print(f"  Gold@20 — MAE: {mae_20:.0f}g, R²: {r2_20:.4f}")
        else:
            mae_20, r2_20 = 0, 0

        # Train win model
        print("Training Win Probability model...")
        self.model_win = GradientBoostingClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, min_samples_leaf=10, random_state=42
        )
        self.model_win.fit(X_train_s, ywin_train)
        gold_probs_test = self.model_win.predict_proba(X_test_s)[:, 1]
        ll_gold = log_loss(ywin_test, gold_probs_test)
        acc_gold = accuracy_score(ywin_test, (gold_probs_test > 0.5).astype(int))
        print(f"  Win Prob — Log Loss: {ll_gold:.4f}, Accuracy: {acc_gold:.1%}")

        # Feature importance (top 20 for gold@10)
        importances = self.model_gd10.feature_importances_
        top_idx = np.argsort(importances)[::-1][:20]
        print(f"\nTop 20 features for Gold@10 prediction:")
        for j in top_idx:
            print(f"  {self.feature_names[j]:<55} {importances[j]:.4f}")

        # =================================================================
        # ELO + GOLD BLEND ANALYSIS
        # =================================================================
        print(f"\n{'='*70}")
        print("ELO + GOLD BLEND ANALYSIS")
        print(f"{'='*70}")

        # --- ELO-only metrics on test set ---
        elo_clipped = np.clip(elo_test, 0.01, 0.99)
        ll_elo = log_loss(ywin_test, elo_clipped)
        acc_elo = accuracy_score(ywin_test, (elo_test > 0.5).astype(int))
        brier_elo = brier_score_loss(ywin_test, elo_clipped)

        print(f"\n  ELO-only (test set, {len(ywin_test)} games):")
        print(f"    Log Loss:  {ll_elo:.4f}")
        print(f"    Accuracy:  {acc_elo:.1%}")
        print(f"    Brier:     {brier_elo:.4f}")

        # --- Gold-only metrics ---
        gold_clipped = np.clip(gold_probs_test, 0.01, 0.99)
        brier_gold = brier_score_loss(ywin_test, gold_clipped)

        print(f"\n  Gold-model-only (test set):")
        print(f"    Log Loss:  {ll_gold:.4f}")
        print(f"    Accuracy:  {acc_gold:.1%}")
        print(f"    Brier:     {brier_gold:.4f}")

        # --- Logistic blend on logits ---
        # Convert to logits (clip to avoid inf)
        def safe_logit(p):
            p = np.clip(p, 0.001, 0.999)
            return np.log(p / (1 - p))

        logit_elo = safe_logit(elo_test)
        logit_gold = safe_logit(gold_probs_test)

        # Stack: [logit_elo, logit_gold] → LogisticRegression → blended prob
        X_blend = np.column_stack([logit_elo, logit_gold])

        # Use first 50% of test for fitting blend, last 50% for final eval
        blend_split = len(X_blend) // 2
        X_blend_fit, X_blend_eval = X_blend[:blend_split], X_blend[blend_split:]
        y_blend_fit, y_blend_eval = ywin_test[:blend_split], ywin_test[blend_split:]
        elo_eval = elo_test[blend_split:]
        gold_eval = gold_probs_test[blend_split:]

        self.blend_model = LogisticRegression(C=1.0, max_iter=1000)
        self.blend_model.fit(X_blend_fit, y_blend_fit)

        # Blend coefficients
        b0 = self.blend_model.intercept_[0]
        b_elo = self.blend_model.coef_[0][0]
        b_gold = self.blend_model.coef_[0][1]

        print(f"\n  Blend coefficients (logistic stacking):")
        print(f"    intercept:    {b0:+.4f}")
        print(f"    β_elo:        {b_elo:+.4f}  (weight on ELO logit)")
        print(f"    β_gold:       {b_gold:+.4f}  (weight on Gold logit)")
        elo_share = abs(b_elo) / (abs(b_elo) + abs(b_gold)) * 100
        gold_share = abs(b_gold) / (abs(b_elo) + abs(b_gold)) * 100
        print(f"    → ELO share: {elo_share:.0f}%, Gold share: {gold_share:.0f}%")

        # Blended predictions on eval set
        blend_probs_eval = self.blend_model.predict_proba(X_blend_eval)[:, 1]
        blend_clipped = np.clip(blend_probs_eval, 0.01, 0.99)

        elo_eval_clipped = np.clip(elo_eval, 0.01, 0.99)
        gold_eval_clipped = np.clip(gold_eval, 0.01, 0.99)

        ll_elo_eval = log_loss(y_blend_eval, elo_eval_clipped)
        ll_gold_eval = log_loss(y_blend_eval, gold_eval_clipped)
        ll_blend_eval = log_loss(y_blend_eval, blend_clipped)

        acc_elo_eval = accuracy_score(y_blend_eval, (elo_eval > 0.5).astype(int))
        acc_gold_eval = accuracy_score(y_blend_eval, (gold_eval > 0.5).astype(int))
        acc_blend_eval = accuracy_score(y_blend_eval, (blend_probs_eval > 0.5).astype(int))

        brier_elo_eval = brier_score_loss(y_blend_eval, elo_eval_clipped)
        brier_gold_eval = brier_score_loss(y_blend_eval, gold_eval_clipped)
        brier_blend_eval = brier_score_loss(y_blend_eval, blend_clipped)

        print(f"\n  {'Model':<20} {'Log Loss':>10} {'Accuracy':>10} {'Brier':>10}")
        print(f"  {'-'*50}")
        print(f"  {'ELO only':<20} {ll_elo_eval:>10.4f} {acc_elo_eval:>9.1%} {brier_elo_eval:>10.4f}")
        print(f"  {'Gold only':<20} {ll_gold_eval:>10.4f} {acc_gold_eval:>9.1%} {brier_gold_eval:>10.4f}")
        print(f"  {'ELO+Gold blend':<20} {ll_blend_eval:>10.4f} {acc_blend_eval:>9.1%} {brier_blend_eval:>10.4f}")

        # Marginal gain
        ll_gain = ll_elo_eval - ll_blend_eval
        acc_gain = acc_blend_eval - acc_elo_eval
        brier_gain = brier_elo_eval - brier_blend_eval
        print(f"\n  Marginal gain (blend vs ELO):")
        print(f"    Log Loss:  {ll_gain:+.4f} ({'better' if ll_gain > 0 else 'worse'})")
        print(f"    Accuracy:  {acc_gain:+.1%}")
        print(f"    Brier:     {brier_gain:+.4f} ({'better' if brier_gain > 0 else 'worse'})")

        # --- Breakdown by ELO confidence bucket ---
        print(f"\n  Marginal gain by ELO confidence bucket:")
        print(f"  {'ELO Range':<20} {'N':>5} {'ELO LL':>8} {'Blend LL':>9} {'Δ LL':>8} {'ELO Acc':>8} {'Blend Acc':>9} {'Δ Acc':>7}")
        print(f"  {'-'*75}")

        buckets = [
            ("50-55% (toss-up)", 0.45, 0.55),
            ("55-60% (lean)", 0.40, 0.45),
            ("60-70% (clear)", 0.30, 0.40),
            ("70-80% (strong)", 0.20, 0.30),
            ("80%+ (dominant)", 0.00, 0.20),
        ]

        for label, lo_dist, hi_dist in buckets:
            # Distance from 0.5 for each ELO prob
            dist = np.abs(elo_eval - 0.5)
            mask = (dist >= lo_dist) & (dist < hi_dist)
            n = mask.sum()
            if n < 10:
                print(f"  {label:<20} {n:>5}   (too few games)")
                continue

            elo_b = np.clip(elo_eval[mask], 0.01, 0.99)
            blend_b = np.clip(blend_probs_eval[mask], 0.01, 0.99)
            y_b = y_blend_eval[mask]

            ll_e = log_loss(y_b, elo_b)
            ll_bl = log_loss(y_b, blend_b)
            acc_e = accuracy_score(y_b, (elo_eval[mask] > 0.5).astype(int))
            acc_bl = accuracy_score(y_b, (blend_probs_eval[mask] > 0.5).astype(int))

            print(f"  {label:<20} {n:>5} {ll_e:>8.4f} {ll_bl:>9.4f} {ll_e-ll_bl:>+8.4f} {acc_e:>7.1%} {acc_bl:>9.1%} {acc_bl-acc_e:>+6.1%}")

        # --- How much does blend shift the odds? ---
        print(f"\n  Average odds shift (blend vs ELO):")
        shifts = blend_probs_eval - elo_eval
        abs_shifts = np.abs(shifts)
        print(f"    Mean absolute shift: {np.mean(abs_shifts):.1%}")
        print(f"    Median shift:        {np.median(abs_shifts):.1%}")
        print(f"    Max shift:           {np.max(abs_shifts):.1%}")
        print(f"    Std of shifts:       {np.std(shifts):.1%}")

        # Distribution of shift sizes
        print(f"\n  Shift distribution:")
        for thresh in [0.01, 0.02, 0.05, 0.10]:
            pct = (abs_shifts >= thresh).mean()
            print(f"    ≥{thresh:.0%} shift:  {pct:.1%} of games")

        self.blend_metrics = {
            "ll_elo": ll_elo_eval, "ll_gold": ll_gold_eval, "ll_blend": ll_blend_eval,
            "acc_elo": acc_elo_eval, "acc_gold": acc_gold_eval, "acc_blend": acc_blend_eval,
            "brier_elo": brier_elo_eval, "brier_gold": brier_gold_eval, "brier_blend": brier_blend_eval,
            "blend_intercept": b0, "blend_b_elo": b_elo, "blend_b_gold": b_gold,
        }

        # Slow-comp residual analysis
        print(f"\n{'='*70}")
        print("SLOW-COMP RESIDUAL ANALYSIS")
        print(f"{'='*70}")
        if gd20_test_mask.sum() > 0:
            pred_20_all = self.model_gd20.predict(X_test_s[gd20_test_mask])
            pred_10_for20 = self.model_gd10.predict(X_test_s[gd20_test_mask])
            actual_20 = y20_test[gd20_test_mask]
            actual_10 = y10_test[gd20_test_mask]

            scaling_factor = actual_20 - actual_10
            predicted_scaling = pred_20_all - pred_10_for20
            residual = scaling_factor - predicted_scaling
            print(f"  Avg gold growth (10→20): {np.mean(scaling_factor):+.0f}g")
            print(f"  Avg predicted growth: {np.mean(predicted_scaling):+.0f}g")
            print(f"  Avg residual: {np.mean(residual):+.0f}g")
            print(f"  Residual std: {np.std(residual):.0f}g")

        return {
            "gd10_mae": mae_10, "gd10_r2": r2_10,
            "gd20_mae": mae_20, "gd20_r2": r2_20,
            "win_logloss": ll_gold, "win_accuracy": acc_gold,
            **self.blend_metrics,
        }

    def _extract_features(self, team_a, team_b, league, game_player_data, game=None):
        """Extract the full feature vector for a matchup."""
        n_features = len(self.feature_names)
        row = np.zeros(n_features)
        idx = 0

        feats_a = self.team_tracker.get_features(team_a)
        feats_b = self.team_tracker.get_features(team_b)
        elo_a = self.team_elo.get_rating(team_a, league)
        elo_b = self.team_elo.get_rating(team_b, league)
        feats_a["team_elo"] = elo_a
        feats_b["team_elo"] = elo_b

        # Group players by team/position/champion
        players_a, players_b = {}, {}
        champs_a, champs_b = {}, {}
        for pg in game_player_data:
            if pg["team"] == team_a:
                players_a[pg["position"]] = pg["player"]
                champs_a[pg["position"]] = pg.get("champion", "unknown")
            elif pg["team"] == team_b:
                players_b[pg["position"]] = pg["player"]
                champs_b[pg["position"]] = pg.get("champion", "unknown")

        # If no player data, use last known roster
        if not players_a and team_a in self.team_rosters:
            players_a = self.team_rosters[team_a]
        if not players_b and team_b in self.team_rosters:
            players_b = self.team_rosters[team_b]

        # Differential team features
        for key in TEAM_FEATURE_KEYS:
            row[idx] = feats_a.get(key, 0) - feats_b.get(key, 0)
            idx += 1

        # Raw team features (side a)
        for key in TEAM_FEATURE_KEYS:
            row[idx] = feats_a.get(key, 0)
            idx += 1

        # Raw team features (side b)
        for key in TEAM_FEATURE_KEYS:
            row[idx] = feats_b.get(key, 0)
            idx += 1

        # Player features by position
        player_elos_a, player_elos_b = [], []
        for pos in POSITIONS:
            player_a = players_a.get(pos)
            player_b = players_b.get(pos)

            if player_a:
                pf_a = self.player_tracker.get_features(player_a, pos)
                pf_a["player_elo"] = self.player_elo.get_rating(player_a)
                player_elos_a.append(pf_a["player_elo"])
            else:
                pf_a = self.player_tracker._default_features()
                pf_a["player_elo"] = 1500.0
                player_elos_a.append(1500.0)

            if player_b:
                pf_b = self.player_tracker.get_features(player_b, pos)
                pf_b["player_elo"] = self.player_elo.get_rating(player_b)
                player_elos_b.append(pf_b["player_elo"])
            else:
                pf_b = self.player_tracker._default_features()
                pf_b["player_elo"] = 1500.0
                player_elos_b.append(1500.0)

            for key in PLAYER_FEATURE_KEYS:
                row[idx] = pf_a.get(key, 0) - pf_b.get(key, 0)
                idx += 1

        # Raw player features (side a)
        for pos in POSITIONS:
            player_a = players_a.get(pos)
            if player_a:
                pf_a = self.player_tracker.get_features(player_a, pos)
                pf_a["player_elo"] = self.player_elo.get_rating(player_a)
            else:
                pf_a = self.player_tracker._default_features()
                pf_a["player_elo"] = 1500.0
            for key in PLAYER_FEATURE_KEYS:
                row[idx] = pf_a.get(key, 0)
                idx += 1

        # Raw player features (side b)
        for pos in POSITIONS:
            player_b = players_b.get(pos)
            if player_b:
                pf_b = self.player_tracker.get_features(player_b, pos)
                pf_b["player_elo"] = self.player_elo.get_rating(player_b)
            else:
                pf_b = self.player_tracker._default_features()
                pf_b["player_elo"] = 1500.0
            for key in PLAYER_FEATURE_KEYS:
                row[idx] = pf_b.get(key, 0)
                idx += 1

        # Champion features
        champ_wr_diffs, champ_gd10_diffs, champ_comfort_diffs = [], [], []
        for pos in POSITIONS:
            player_a = players_a.get(pos)
            player_b = players_b.get(pos)
            champ_a = champs_a.get(pos, "unknown")
            champ_b = champs_b.get(pos, "unknown")

            cf_a = self.champ_tracker.get_champion_features(champ_a, player_a or "", pos)
            cf_b = self.champ_tracker.get_champion_features(champ_b, player_b or "", pos)

            for key in CHAMPION_FEATURE_KEYS:
                row[idx] = cf_a.get(key, 0) - cf_b.get(key, 0)
                idx += 1

            champ_wr_diffs.append(cf_a.get("champ_win_rate", 0.5) - cf_b.get("champ_win_rate", 0.5))
            champ_gd10_diffs.append(cf_a.get("champ_gd10", 0) - cf_b.get("champ_gd10", 0))
            champ_comfort_diffs.append(cf_a.get("player_champ_games", 0) - cf_b.get("player_champ_games", 0))

        # Raw champion features (side a)
        for pos in POSITIONS:
            player_a = players_a.get(pos)
            champ_a = champs_a.get(pos, "unknown")
            cf_a = self.champ_tracker.get_champion_features(champ_a, player_a or "", pos)
            for key in CHAMPION_FEATURE_KEYS:
                row[idx] = cf_a.get(key, 0)
                idx += 1

        # Raw champion features (side b)
        for pos in POSITIONS:
            player_b = players_b.get(pos)
            champ_b = champs_b.get(pos, "unknown")
            cf_b = self.champ_tracker.get_champion_features(champ_b, player_b or "", pos)
            for key in CHAMPION_FEATURE_KEYS:
                row[idx] = cf_b.get(key, 0)
                idx += 1

        # Aggregate draft features
        row[idx] = np.mean(champ_wr_diffs) if champ_wr_diffs else 0; idx += 1
        row[idx] = np.mean(champ_gd10_diffs) if champ_gd10_diffs else 0; idx += 1
        row[idx] = np.mean(champ_comfort_diffs) if champ_comfort_diffs else 0; idx += 1

        # Context features
        row[idx] = elo_a - elo_b; idx += 1
        row[idx] = np.mean(player_elos_a) - np.mean(player_elos_b); idx += 1
        side = game.get("side_a", "blue") if game else "blue"
        row[idx] = 1.0 if side == "blue" else 0.0; idx += 1
        row[idx] = float(game.get("playoffs", 0)) if game else 0.0; idx += 1

        return row

    def _update_trackers(self, game, game_player_data, team_a, team_b, winner):
        """Update all trackers after a game result."""
        self.team_elo.update(team_a, team_b, winner)

        def _safe(g, col, default=0.0):
            val = g.get(col, default)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            return float(val)

        for team, prefix in [(team_a, "a"), (team_b, "b")]:
            sign = 1.0 if prefix == "a" else -1.0
            team_stats = {
                "result": 1 if winner == team else 0,
                "golddiffat10": sign * _safe(game, "a_golddiffat10"),
                "golddiffat15": sign * _safe(game, "a_golddiffat15"),
                "golddiffat20": sign * _safe(game, "a_golddiffat20"),
                "xpdiffat10": sign * _safe(game, "a_xpdiffat10"),
                "gamelength": _safe(game, "gamelength", 1800),
                "kills": _safe(game, f"{prefix}_kills"),
                "dragons": _safe(game, f"{prefix}_dragons"),
                "barons": _safe(game, f"{prefix}_barons"),
                "towers": _safe(game, f"{prefix}_towers"),
                "firstblood": _safe(game, f"{prefix}_firstblood"),
                "firstdragon": _safe(game, f"{prefix}_firstdragon"),
                "firstherald": _safe(game, f"{prefix}_firstherald"),
                "firsttower": _safe(game, f"{prefix}_firsttower"),
                "totalgold": _safe(game, f"{prefix}_totalgold"),
                "turretplates": _safe(game, f"{prefix}_turretplates"),
                "opp_turretplates": _safe(game, f"{prefix}_opp_turretplates"),
                "csdiffat10": 0.0,
            }
            self.team_tracker.record_game(team, team_stats)

        expected_a = self.team_elo.expected(team_a, team_b)
        for pg in game_player_data:
            won = pg["team"] == winner
            team_exp = expected_a if pg["team"] == team_a else (1 - expected_a)
            avg_kda = np.mean([p["kda"] for p in game_player_data]) or 2.0
            contribution = min(1.5, max(0.5, pg["kda"] / avg_kda))
            self.player_elo.update(pg["player"], won, team_exp, contribution)
            self.player_tracker.record_game(pg["player"], pg["position"], pg)
            self.champ_tracker.record_game(
                champion=pg.get("champion", "unknown"),
                player=pg["player"],
                position=pg["position"],
                stats=pg,
            )

    def predict_match(self, team_a: str, team_b: str, league: str = None):
        """
        Predict gold@10, gold@20, and win probability for a matchup.

        Uses last-known rosters and current rolling stats.
        No draft info (pre-draft prediction).
        """
        if self.model_gd10 is None:
            raise RuntimeError("Model not trained. Call process_and_train() first.")

        # Build feature vector using last-known rosters (no champion info)
        game_player_data = []
        roster_a = self.team_rosters.get(team_a, {})
        roster_b = self.team_rosters.get(team_b, {})

        # Create pseudo player_game entries so _extract_features can find them
        for pos, player in roster_a.items():
            game_player_data.append({"team": team_a, "player": player,
                                     "position": pos, "champion": "unknown"})
        for pos, player in roster_b.items():
            game_player_data.append({"team": team_b, "player": player,
                                     "position": pos, "champion": "unknown"})

        if not league:
            league = self.team_leagues.get(team_a, self.team_leagues.get(team_b))

        game_ctx = {"side_a": "blue", "playoffs": 0}
        row = self._extract_features(team_a, team_b, league, game_player_data, game_ctx)
        row_scaled = self.scaler.transform(np.nan_to_num(row.reshape(1, -1)))

        gd10 = self.model_gd10.predict(row_scaled)[0]
        gd20 = self.model_gd20.predict(row_scaled)[0]
        gold_win_prob = self.model_win.predict_proba(row_scaled)[0][1]

        # Slow-comp residual: difference between gd20 and what gd10 growth would predict
        scaling = gd20 - gd10

        # Compute blended probability (ELO + Gold)
        elo_prob = self.team_elo.expected(team_a, team_b)
        if self.blend_model is not None:
            def safe_logit(p):
                p = np.clip(p, 0.001, 0.999)
                return np.log(p / (1 - p))
            logit_e = safe_logit(elo_prob)
            logit_g = safe_logit(gold_win_prob)
            blend_input = np.array([[logit_e, logit_g]])
            win_prob = self.blend_model.predict_proba(blend_input)[0][1]
        else:
            win_prob = gold_win_prob

        # Team context
        elo_a = self.team_elo.get_rating(team_a, league)
        elo_b = self.team_elo.get_rating(team_b, league)
        feats_a = self.team_tracker.get_features(team_a)
        feats_b = self.team_tracker.get_features(team_b)

        # Player Elos
        player_elos_a = {}
        player_elos_b = {}
        for pos in POSITIONS:
            pa = roster_a.get(pos)
            pb = roster_b.get(pos)
            if pa:
                player_elos_a[pos] = (pa, self.player_elo.get_rating(pa))
            if pb:
                player_elos_b[pos] = (pb, self.player_elo.get_rating(pb))

        return {
            "team_a": team_a,
            "team_b": team_b,
            "pred_gd10": gd10,
            "pred_gd20": gd20,
            "gold_scaling_10_to_20": scaling,
            "win_prob_a": win_prob,
            "win_prob_b": 1 - win_prob,
            "elo_win_prob_a": elo_prob,
            "gold_win_prob_a": gold_win_prob,
            "blend_win_prob_a": win_prob,
            "elo_a": elo_a,
            "elo_b": elo_b,
            "elo_diff": elo_a - elo_b,
            "team_a_stats": {
                "win_rate": feats_a["team_win_rate"],
                "avg_gd10": feats_a["team_avg_gd10"],
                "avg_gd15": feats_a["team_avg_gd15"],
                "avg_gd20": feats_a["team_avg_gd20"],
                "momentum_5": feats_a["team_momentum_5"],
                "avg_gamelength": feats_a["team_avg_gamelength"],
                "n_games": feats_a["team_n_games"],
            },
            "team_b_stats": {
                "win_rate": feats_b["team_win_rate"],
                "avg_gd10": feats_b["team_avg_gd10"],
                "avg_gd15": feats_b["team_avg_gd15"],
                "avg_gd20": feats_b["team_avg_gd20"],
                "momentum_5": feats_b["team_momentum_5"],
                "avg_gamelength": feats_b["team_avg_gamelength"],
                "n_games": feats_b["team_n_games"],
            },
            "player_elos_a": player_elos_a,
            "player_elos_b": player_elos_b,
            "roster_a": roster_a,
            "roster_b": roster_b,
        }


def print_prediction(pred: dict):
    """Pretty-print a match prediction with ELO+Gold blend breakdown."""
    ta = pred["team_a"]
    tb = pred["team_b"]
    gd10 = pred["pred_gd10"]
    gd20 = pred["pred_gd20"]
    scaling = pred["gold_scaling_10_to_20"]
    wp_a = pred["win_prob_a"]
    elo_wp = pred.get("elo_win_prob_a", wp_a)
    gold_wp = pred.get("gold_win_prob_a", wp_a)
    blend_wp = pred.get("blend_win_prob_a", wp_a)

    fav = ta if wp_a > 0.5 else tb
    fav_pct = max(wp_a, 1 - wp_a)

    print(f"\n{'='*70}")
    print(f"  {ta}  vs  {tb}")
    print(f"{'='*70}")

    # Blended win probability (main number)
    print(f"\n  Win Probability (ELO+Gold Blend):")
    bar_a = "█" * int(blend_wp * 40)
    bar_b = "█" * int((1 - blend_wp) * 40)
    print(f"    {ta:<20} {blend_wp:>5.1%}  {bar_a}")
    print(f"    {tb:<20} {1-blend_wp:>5.1%}  {bar_b}")
    print(f"    → Favored: {fav} ({fav_pct:.1%})")

    # Component breakdown
    elo_shift = blend_wp - elo_wp
    print(f"\n  Probability Breakdown:")
    print(f"    {'ELO only:':<20} {ta} {elo_wp:>5.1%}  /  {tb} {1-elo_wp:>5.1%}")
    print(f"    {'Gold model:':<20} {ta} {gold_wp:>5.1%}  /  {tb} {1-gold_wp:>5.1%}")
    print(f"    {'ELO+Gold blend:':<20} {ta} {blend_wp:>5.1%}  /  {tb} {1-blend_wp:>5.1%}")
    direction = ta if elo_shift > 0 else tb
    print(f"    Gold tilt:         {abs(elo_shift):>+.1%} toward {direction}")

    print(f"\n  Gold Predictions:")
    gd10_fav = ta if gd10 > 0 else tb
    gd20_fav = ta if gd20 > 0 else tb
    print(f"    Gold Diff @ 10 min:  {gd10:>+.0f}g  ({gd10_fav} ahead)")
    print(f"    Gold Diff @ 20 min:  {gd20:>+.0f}g  ({gd20_fav} ahead)")
    print(f"    Gold Growth 10→20:   {scaling:>+.0f}g  ", end="")
    if abs(scaling) > abs(gd10) * 0.5 and gd10 != 0:
        if (scaling > 0 and gd10 > 0) or (scaling < 0 and gd10 < 0):
            print("(lead extends — dominant)")
        else:
            print("(comeback potential — slow scaling)")
    elif abs(scaling) < 200:
        print("(stable — game state holds)")
    else:
        print("")

    # Elo comparison
    print(f"\n  Team Elo:")
    print(f"    {ta:<20} {pred['elo_a']:>7.0f}")
    print(f"    {tb:<20} {pred['elo_b']:>7.0f}")
    print(f"    Difference:          {pred['elo_diff']:>+.0f}")

    # Rolling stats
    for side, label in [("team_a_stats", ta), ("team_b_stats", tb)]:
        s = pred[side]
        print(f"\n  {label} Rolling Stats ({int(s['n_games'])} games):")
        print(f"    Win Rate:     {s['win_rate']:.1%}")
        print(f"    Avg GD@10:    {s['avg_gd10']:>+.0f}g")
        print(f"    Avg GD@15:    {s['avg_gd15']:>+.0f}g")
        print(f"    Avg GD@20:    {s['avg_gd20']:>+.0f}g")
        gd_growth = s['avg_gd20'] - s['avg_gd10']
        print(f"    Gold Growth:  {gd_growth:>+.0f}g (10→20)")
        print(f"    Momentum (5): {s['momentum_5']:.1%}")
        print(f"    Avg Length:   {s['avg_gamelength']/60:.1f} min")

    # Player Elos
    for side, label in [("player_elos_a", ta), ("player_elos_b", tb)]:
        elos = pred[side]
        if elos:
            print(f"\n  {label} Player Elos:")
            for pos in POSITIONS:
                if pos in elos:
                    name, elo = elos[pos]
                    print(f"    {pos:<5} {name:<18} {elo:>7.0f}")
            avg = np.mean([e[1] for e in elos.values()])
            print(f"    {'avg':<5} {'':18} {avg:>7.0f}")

    # Slow-comp analysis
    sa = pred["team_a_stats"]
    sb = pred["team_b_stats"]
    growth_a = sa["avg_gd20"] - sa["avg_gd10"]
    growth_b = sb["avg_gd20"] - sb["avg_gd10"]
    print(f"\n  Slow-Comp Analysis:")
    print(f"    {ta} gold growth (10→20): {growth_a:>+.0f}g/game")
    print(f"    {tb} gold growth (10→20): {growth_b:>+.0f}g/game")
    if growth_a > growth_b + 200:
        print(f"    → {ta} scales harder mid-game")
    elif growth_b > growth_a + 200:
        print(f"    → {tb} scales harder mid-game")
    else:
        print(f"    → Similar scaling profiles")

    print()


WORLDS_LEAGUES = [
    "LCK", "LPL", "LEC", "LCS", "LCP", "PCS",
    "CBLOL", "VCS", "LTA N", "LTA S", "LLA", "LJL", "TCL",
    "MSI", "WLDs",
]


def main():
    parser = argparse.ArgumentParser(description="Gold prediction model + ELO blend")
    parser.add_argument("csv_path", help="Path to Oracle's Elixir CSV")
    parser.add_argument("--predict", nargs=2, action="append",
                        metavar=("TEAM_A", "TEAM_B"),
                        help="Predict a matchup (can specify multiple)")
    parser.add_argument("--leagues", nargs="+", default=None,
                        help="Leagues to include (default: worlds-qualifying leagues)")
    parser.add_argument("--all-leagues", action="store_true",
                        help="Use all leagues instead of worlds-only")
    parser.add_argument("--min-date", default=None)
    args = parser.parse_args()

    # Default to worlds-qualifying leagues unless --all-leagues
    leagues = args.leagues
    if leagues is None and not args.all_leagues:
        leagues = WORLDS_LEAGUES
        print(f"Filtering to worlds-qualifying leagues: {', '.join(leagues)}")

    model = GoldPredictionModel()
    metrics = model.process_and_train(args.csv_path, leagues=leagues,
                                       min_date=args.min_date)

    if args.predict:
        for team_a, team_b in args.predict:
            pred = model.predict_match(team_a, team_b)
            print_prediction(pred)

    return model


if __name__ == "__main__":
    main()
