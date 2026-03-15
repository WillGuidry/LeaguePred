"""
Elo Pipeline
-------------
Connects the Elo engine to data and evaluation.

This is the backbone pipeline:
    1. Load match data
    2. Process matches chronologically through Elo engine
    3. Generate predictions for each match BEFORE updating ratings
    4. Evaluate prediction quality

Key design principles:
    - Predict FIRST, then update (prevents look-ahead leakage)
    - Teams start at their regional Elo prior, not a flat default
    - Roster changes are detected automatically and trigger rating regression
"""

import numpy as np
import pandas as pd
from typing import Optional

from src.elo.engine import EloEngine
from src.evaluation.metrics import full_report, compare_models
from src.evaluation.splits import temporal_split, walk_forward_splits
from src.config import (
    REGIONAL_ELO_PRIORS,
    REGIONAL_ELO_DEFAULT,
    ROSTER_CHANGE_MINOR_THRESHOLD,
    ROSTER_CHANGE_MAJOR_THRESHOLD,
    ROSTER_MINOR_REGRESSION,
    ROSTER_MAJOR_REGRESSION,
    ACTIVE_MODULES,
    LEAD_STATE_WINDOW,
    LEAD_STATE_MIN_GAMES,
    LEAD_STATE_SHRINKAGE_WEIGHT,
)
from src.modules.margin_of_victory import extract_mov_from_row
from src.modules.team_state_tracker import TeamStateTracker
from src.modules.lead_state import LeadStateModule


def _get_regional_elo(league: str) -> float:
    """Get the starting Elo for a team based on its league."""
    return REGIONAL_ELO_PRIORS.get(league, REGIONAL_ELO_DEFAULT)


def _detect_roster_change(
    team: str,
    current_roster: list,
    roster_history: dict,
) -> int:
    """
    Detect how many players changed since this team's last game.

    Args:
        team: Team name.
        current_roster: List of player names in current game.
        roster_history: {team: last_known_roster} dict.

    Returns:
        Number of players changed (0-5). Returns 0 if first game.
    """
    if team not in roster_history or not current_roster:
        return 0

    last_roster = roster_history[team]
    if not last_roster:
        return 0

    # Count players NOT in the previous roster
    last_set = set(last_roster)
    current_set = set(current_roster)
    new_players = current_set - last_set

    return len(new_players)


def run_elo_pipeline(
    df: pd.DataFrame,
    k_factor: float = 32.0,
    default_elo: float = 1500.0,
    scale_factor: float = 400.0,
    season_regression: float = 0.3,
    regression_col: Optional[str] = "split",
    use_regional_priors: bool = True,
    detect_roster_changes: bool = True,
    use_mov: Optional[bool] = None,
    use_lead_state: Optional[bool] = None,
) -> pd.DataFrame:
    """
    Run the Elo engine over a chronological sequence of matches.

    For each match:
        1. Initialize new teams at their regional Elo (not flat 1500)
        2. Check for roster changes and regress if needed
        3. Record the pre-match Elo prediction
        4. Process the match result and update ratings

    Args:
        df: Match data sorted by date, with columns:
            team_a, team_b, winner (required)
            date, league, split, patch (optional)
            roster_a, roster_b (optional, for roster change detection)
        k_factor: Elo K-factor.
        default_elo: Fallback Elo (only used if regional priors disabled).
        scale_factor: Elo scale factor.
        season_regression: How much to regress between splits.
        regression_col: Column to detect split boundaries (None to disable).
        use_regional_priors: Use league-specific starting Elo.
        detect_roster_changes: Auto-detect and handle roster changes.
        use_mov: Enable MOV multiplier. None = use ACTIVE_MODULES config.
        use_lead_state: Enable Lead State features. None = use ACTIVE_MODULES config.

    Returns:
        DataFrame with original data plus prediction columns.
    """
    df = df.sort_values("date").reset_index(drop=True) if "date" in df.columns else df.copy()

    engine = EloEngine(
        default_elo=default_elo,
        k_factor=k_factor,
        scale_factor=scale_factor,
    )

    # Determine which modules are active
    mov_active = use_mov if use_mov is not None else ACTIVE_MODULES.get("margin_of_victory", False)
    lead_state_active = use_lead_state if use_lead_state is not None else ACTIVE_MODULES.get("lead_state", False)

    # Initialize Lead State tracker and module
    tracker = None
    lead_state_module = None
    if lead_state_active:
        tracker = TeamStateTracker(
            window=LEAD_STATE_WINDOW,
            min_games=LEAD_STATE_MIN_GAMES,
            shrinkage_weight=LEAD_STATE_SHRINKAGE_WEIGHT,
        )
        lead_state_module = LeadStateModule(tracker)

    predictions = []
    last_split = None
    roster_history = {}  # {team_name: [player1, player2, ...]}
    team_leagues = {}    # {team_name: league} for regional prior lookup

    has_rosters = "roster_a" in df.columns and "roster_b" in df.columns

    for idx, row in df.iterrows():
        team_a = row["team_a"]
        team_b = row["team_b"]
        winner = row["winner"]
        league = row.get("league", None)

        # --- Initialize new teams at regional Elo ---
        for team in [team_a, team_b]:
            if team not in engine.ratings:
                if use_regional_priors and league:
                    regional_elo = _get_regional_elo(league)
                    engine.add_team(team, elo=regional_elo)
                    team_leagues[team] = league
                else:
                    engine.add_team(team)

        # --- Handle season regression at split boundaries ---
        if regression_col and regression_col in df.columns:
            current_split = row[regression_col]
            if last_split is not None and current_split != last_split:
                # Regress toward each team's regional mean, not global mean
                if use_regional_priors:
                    _regress_regional(engine, team_leagues, season_regression)
                else:
                    engine.regress_to_mean(season_regression)
            last_split = current_split

        # --- Detect roster changes ---
        roster_changes_a = 0
        roster_changes_b = 0
        if detect_roster_changes and has_rosters:
            roster_a = row.get("roster_a", [])
            roster_b = row.get("roster_b", [])

            # Ensure rosters are lists
            if isinstance(roster_a, str):
                roster_a = []
            if isinstance(roster_b, str):
                roster_b = []

            roster_changes_a = _detect_roster_change(team_a, roster_a, roster_history)
            roster_changes_b = _detect_roster_change(team_b, roster_b, roster_history)

            # Apply regression for roster changes
            for team, changes in [(team_a, roster_changes_a), (team_b, roster_changes_b)]:
                if changes >= ROSTER_CHANGE_MAJOR_THRESHOLD:
                    _regress_team_to_regional(engine, team, team_leagues, ROSTER_MAJOR_REGRESSION)
                elif changes >= ROSTER_CHANGE_MINOR_THRESHOLD:
                    _regress_team_to_regional(engine, team, team_leagues, ROSTER_MINOR_REGRESSION)

            # Update roster history
            if roster_a:
                roster_history[team_a] = roster_a
            if roster_b:
                roster_history[team_b] = roster_b

        # --- PREDICT FIRST (before seeing the result) ---
        pred = engine.predict(team_a, team_b)

        # Lead State features (pre-match, from prior games only)
        lead_edge = 0.0
        lead_confidence = 0.0
        lead_feats_a = {}
        lead_feats_b = {}
        if lead_state_active and lead_state_module:
            lead_edge = lead_state_module.compute(team_a, team_b)
            lead_confidence = lead_state_module.confidence()
            lead_feats_a = lead_state_module._last_features_a
            lead_feats_b = lead_state_module._last_features_b

        # Actual outcome (1 if team_a won, 0 if team_b won)
        if winner == team_a:
            actual_a = 1
        elif winner == team_b:
            actual_a = 0
        else:
            actual_a = np.nan

        # Compute MOV multiplier (post-match, from this game's stats)
        mov_multiplier = 1.0
        if mov_active and pd.notna(actual_a):
            mov_multiplier = extract_mov_from_row(row, winner)

        pred_row = {
            "elo_a": pred["elo_a"],
            "elo_b": pred["elo_b"],
            "elo_diff": pred["elo_diff"],
            "pred_prob_a": pred["win_prob_a"],
            "pred_prob_b": pred["win_prob_b"],
            "predicted_winner": pred["predicted_winner"],
            "actual_outcome_a": actual_a,
            "roster_changes_a": roster_changes_a,
            "roster_changes_b": roster_changes_b,
            "mov_multiplier": mov_multiplier,
        }

        # Add lead state features to output
        if lead_state_active:
            pred_row["lead_edge"] = lead_edge
            pred_row["lead_confidence"] = lead_confidence
            for feat in ["avg_gd15", "avg_gd20", "first_dragon_rate",
                         "first_herald_rate", "first_tower_rate", "first_blood_rate",
                         "win_rate_when_ahead_2k_15", "win_rate_when_ahead_3k_20",
                         "win_rate_when_behind_2k_15", "avg_game_length_when_ahead"]:
                pred_row[f"ls_{feat}_a"] = lead_feats_a.get(feat, np.nan)
                pred_row[f"ls_{feat}_b"] = lead_feats_b.get(feat, np.nan)

        predictions.append(pred_row)

        # --- NOW update ratings with the result ---
        if pd.notna(actual_a):
            engine.process_match(team_a, team_b, winner, margin_multiplier=mov_multiplier)

            # Update Lead State tracker with this game's post-match stats
            if lead_state_active and tracker:
                _record_team_stats(tracker, row, team_a, team_b, winner)

    pred_df = pd.DataFrame(predictions)
    result = pd.concat([df.reset_index(drop=True), pred_df], axis=1)

    return result


def _regress_regional(engine: EloEngine, team_leagues: dict, fraction: float) -> None:
    """Regress each team toward its own regional mean, not a global mean."""
    for team in list(engine.ratings.keys()):
        league = team_leagues.get(team)
        if league:
            regional_mean = _get_regional_elo(league)
        else:
            regional_mean = engine.default_elo

        current = engine.ratings[team]
        engine.ratings[team] = current + fraction * (regional_mean - current)


def _regress_team_to_regional(
    engine: EloEngine,
    team: str,
    team_leagues: dict,
    fraction: float,
) -> None:
    """Regress a single team toward its regional mean after a roster change."""
    if team not in engine.ratings:
        return

    league = team_leagues.get(team)
    if league:
        regional_mean = _get_regional_elo(league)
    else:
        regional_mean = engine.default_elo

    current = engine.ratings[team]
    engine.ratings[team] = current + fraction * (regional_mean - current)


def _record_team_stats(
    tracker: TeamStateTracker,
    row,
    team_a: str,
    team_b: str,
    winner: str,
) -> None:
    """
    Extract post-match stats from a game row and record them in the tracker.

    Called in the UPDATE phase after the match result is known.
    Records stats for both teams with appropriate perspective (gold diff
    is positive for the team that's ahead, negative for the one behind).
    """
    def _safe(key, default=0.0):
        val = row.get(key) if hasattr(row, 'get') else getattr(row, key, None) if hasattr(row, key) else None
        if val is None:
            return default
        try:
            if isinstance(val, float) and np.isnan(val):
                return default
        except (TypeError, ValueError):
            pass
        return float(val)

    # Team A stats (from team_a's perspective)
    stats_a = {
        "result": 1 if winner == team_a else 0,
        "golddiffat15": _safe("a_golddiffat15"),
        "golddiffat20": _safe("a_golddiffat20"),
        "firstdragon": _safe("a_firstdragon"),
        "firstherald": _safe("a_firstherald"),
        "firsttower": _safe("a_firsttower"),
        "firstblood": _safe("a_firstblood"),
        "dragons": _safe("a_dragons"),
        "barons": _safe("a_barons"),
        "towers": _safe("a_towers"),
        "gamelength": _safe("gamelength"),
        "totalgold": _safe("a_totalgold"),
        "opp_totalgold": _safe("b_totalgold"),
    }

    # Team B stats (from team_b's perspective -- gold diffs inverted)
    stats_b = {
        "result": 1 if winner == team_b else 0,
        "golddiffat15": -_safe("a_golddiffat15"),
        "golddiffat20": -_safe("a_golddiffat20"),
        "firstdragon": _safe("b_firstdragon"),
        "firstherald": _safe("b_firstherald"),
        "firsttower": _safe("b_firsttower"),
        "firstblood": _safe("b_firstblood"),
        "dragons": _safe("b_dragons"),
        "barons": _safe("b_barons"),
        "towers": _safe("b_towers"),
        "gamelength": _safe("gamelength"),
        "totalgold": _safe("b_totalgold"),
        "opp_totalgold": _safe("a_totalgold"),
    }

    tracker.record_game(team_a, stats_a)
    tracker.record_game(team_b, stats_b)


def evaluate_elo(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    k_factor: float = 32.0,
    default_elo: float = 1500.0,
    scale_factor: float = 400.0,
    season_regression: float = 0.3,
    use_regional_priors: bool = True,
    detect_roster_changes: bool = True,
    model_name: str = "Elo Baseline",
    use_mov: Optional[bool] = None,
    use_lead_state: Optional[bool] = None,
) -> dict:
    """
    Run Elo on the full dataset and evaluate on the test portion.

    Important: Elo ratings are built from ALL data chronologically,
    but evaluation metrics are computed ONLY on the test set.
    """
    results = run_elo_pipeline(
        df,
        k_factor=k_factor,
        default_elo=default_elo,
        scale_factor=scale_factor,
        season_regression=season_regression,
        use_regional_priors=use_regional_priors,
        detect_roster_changes=detect_roster_changes,
        use_mov=use_mov,
        use_lead_state=use_lead_state,
    )

    train, test = temporal_split(results, test_fraction=test_fraction)

    test_clean = test.dropna(subset=["actual_outcome_a", "pred_prob_a"])

    if len(test_clean) == 0:
        print("WARNING: No valid predictions in test set.")
        return {}

    y_true = test_clean["actual_outcome_a"].values
    y_prob = test_clean["pred_prob_a"].values

    report = full_report(y_true, y_prob, model_name=model_name)
    report["elo_params"] = {
        "k_factor": k_factor,
        "default_elo": default_elo,
        "scale_factor": scale_factor,
        "season_regression": season_regression,
        "use_regional_priors": use_regional_priors,
        "detect_roster_changes": detect_roster_changes,
    }
    report["results_df"] = results

    return report


def compare_regional_vs_flat(
    df: pd.DataFrame,
    k_factor: float = 32.0,
    test_fraction: float = 0.2,
) -> pd.DataFrame:
    """
    A/B test: regional priors vs flat 1500 starting Elo.
    This is an ablation test to prove regional priors add value.
    """
    # Run both variants
    results_flat = run_elo_pipeline(
        df, k_factor=k_factor, use_regional_priors=False, detect_roster_changes=False,
    )
    results_regional = run_elo_pipeline(
        df, k_factor=k_factor, use_regional_priors=True, detect_roster_changes=False,
    )
    results_regional_roster = run_elo_pipeline(
        df, k_factor=k_factor, use_regional_priors=True, detect_roster_changes=True,
    )

    # Evaluate test portion
    split_idx = int(len(df) * (1 - test_fraction))

    predictions = {}
    for name, result in [
        ("Flat 1500", results_flat),
        ("Regional priors", results_regional),
        ("Regional + roster", results_regional_roster),
    ]:
        test = result.iloc[split_idx:]
        test_clean = test.dropna(subset=["actual_outcome_a", "pred_prob_a"])
        predictions[name] = test_clean["pred_prob_a"].values

    # Get y_true from any result (they're the same)
    test_clean = results_flat.iloc[split_idx:].dropna(subset=["actual_outcome_a"])
    y_true = test_clean["actual_outcome_a"].values

    return compare_models(y_true, predictions)


def walk_forward_evaluate_elo(
    df: pd.DataFrame,
    n_splits: int = 5,
    min_train_size: int = 200,
    k_factor: float = 32.0,
    default_elo: float = 1500.0,
    scale_factor: float = 400.0,
    season_regression: float = 0.3,
    use_regional_priors: bool = True,
    detect_roster_changes: bool = True,
    model_name: str = "Elo Baseline",
) -> dict:
    """Walk-forward evaluation of Elo predictions."""
    results = run_elo_pipeline(
        df,
        k_factor=k_factor,
        default_elo=default_elo,
        scale_factor=scale_factor,
        season_regression=season_regression,
        use_regional_priors=use_regional_priors,
        detect_roster_changes=detect_roster_changes,
    )

    splits = walk_forward_splits(
        results,
        n_splits=n_splits,
        min_train_size=min_train_size,
    )

    from src.evaluation.metrics import log_loss, brier_score, accuracy, calibration_error

    split_results = []
    all_y_true = []
    all_y_prob = []

    for i, (train, test) in enumerate(splits):
        test_clean = test.dropna(subset=["actual_outcome_a", "pred_prob_a"])
        if len(test_clean) == 0:
            continue

        y_true = test_clean["actual_outcome_a"].values
        y_prob = test_clean["pred_prob_a"].values

        split_results.append({
            "split": i + 1,
            "train_size": len(train),
            "test_size": len(test_clean),
            "log_loss": round(log_loss(y_true, y_prob), 4),
            "brier_score": round(brier_score(y_true, y_prob), 4),
            "accuracy": round(accuracy(y_true, y_prob), 4),
            "calibration_error": round(calibration_error(y_true, y_prob), 4),
        })

        all_y_true.extend(y_true)
        all_y_prob.extend(y_prob)

    splits_df = pd.DataFrame(split_results)

    print(f"\n{'='*70}")
    print(f"WALK-FORWARD EVALUATION: {model_name} ({n_splits} splits)")
    print(f"{'='*70}")
    print(splits_df.to_string(index=False))
    print(f"\nMean log loss:     {splits_df['log_loss'].mean():.4f} (+/- {splits_df['log_loss'].std():.4f})")
    print(f"Mean Brier score:  {splits_df['brier_score'].mean():.4f} (+/- {splits_df['brier_score'].std():.4f})")
    print(f"Mean accuracy:     {splits_df['accuracy'].mean():.1%}")
    print()

    return {
        "model": model_name,
        "splits": splits_df,
        "aggregate_log_loss": round(log_loss(np.array(all_y_true), np.array(all_y_prob)), 4),
        "aggregate_brier": round(brier_score(np.array(all_y_true), np.array(all_y_prob)), 4),
        "aggregate_accuracy": round(accuracy(np.array(all_y_true), np.array(all_y_prob)), 4),
    }


def grid_search_elo(
    df: pd.DataFrame,
    k_factors: list = None,
    scale_factors: list = None,
    regressions: list = None,
    test_fraction: float = 0.2,
    use_regional_priors: bool = True,
) -> pd.DataFrame:
    """
    Grid search over Elo hyperparameters.

    IMPORTANT: This uses a single train/test split.
    After finding good params, validate with walk_forward_evaluate_elo.
    """
    if k_factors is None:
        k_factors = [16, 24, 32, 40, 48]
    if scale_factors is None:
        scale_factors = [400]
    if regressions is None:
        regressions = [0.0, 0.2, 0.3, 0.4, 0.5]

    from src.evaluation.metrics import log_loss as ll_fn, brier_score as bs_fn, accuracy as acc_fn

    results = []
    total = len(k_factors) * len(scale_factors) * len(regressions)
    print(f"Grid search: {total} combinations...")

    for k in k_factors:
        for s in scale_factors:
            for r in regressions:
                pipeline_result = run_elo_pipeline(
                    df, k_factor=k, scale_factor=s, season_regression=r,
                    use_regional_priors=use_regional_priors,
                )

                split_idx = int(len(pipeline_result) * (1 - test_fraction))
                test = pipeline_result.iloc[split_idx:]
                test_clean = test.dropna(subset=["actual_outcome_a", "pred_prob_a"])

                if len(test_clean) == 0:
                    continue

                y_true = test_clean["actual_outcome_a"].values
                y_prob = test_clean["pred_prob_a"].values

                results.append({
                    "k_factor": k,
                    "scale_factor": s,
                    "regression": r,
                    "log_loss": round(ll_fn(y_true, y_prob), 4),
                    "brier_score": round(bs_fn(y_true, y_prob), 4),
                    "accuracy": round(acc_fn(y_true, y_prob), 4),
                    "test_size": len(test_clean),
                })

    results_df = pd.DataFrame(results).sort_values("log_loss")
    print(f"\nTop 5 configurations:")
    print(results_df.head().to_string(index=False))
    print()

    return results_df
