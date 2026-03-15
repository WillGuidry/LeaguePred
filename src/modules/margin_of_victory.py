"""
Margin of Victory Module
-------------------------
Computes a dominance multiplier from post-match stats to scale the Elo K-factor.

A dominant win (large gold lead, fast game, objective control) updates Elo more
aggressively. A narrow/scrappy win updates it less. This makes the Elo system
respond faster to true skill gaps and slower to coin-flip results.

The multiplier is applied AFTER the match result is known (it's a post-match
adjustment to the Elo update, not a pre-match feature).

Leakage note:
    MOV modifies HOW MUCH Elo changes from a known result. It does NOT use
    future information for prediction. The multiplier comes from the same game
    whose result is being processed.
"""

import math
import numpy as np
from typing import Optional

from src.config import MARGIN_WEIGHTS, MARGIN_MULTIPLIER_MIN, MARGIN_MULTIPLIER_MAX


def _sigmoid(x: float) -> float:
    """Standard sigmoid, mapping any real number to (0, 1)."""
    return 1.0 / (1.0 + math.exp(-x))


def _normalize_gold_diff_15(gold_diff: float) -> float:
    """Normalize gold diff at 15 minutes. 1500 gold gap → ~0.73."""
    return _sigmoid(gold_diff / 1500.0)


def _normalize_game_duration(gamelength_seconds: float) -> float:
    """
    Shorter game = more dominant.
    25 min (1500s) baseline → 1.0, 50 min (3000s) → 0.0.
    """
    return 1.0 - max(0.0, min(1.0, (gamelength_seconds - 1500.0) / 1500.0))


def _normalize_kill_diff(kill_diff: float) -> float:
    """Normalize kill differential. 10-kill gap → ~0.73."""
    return _sigmoid(kill_diff / 10.0)


def _normalize_objective_diff(
    dragon_diff: float, baron_diff: float, tower_diff: float
) -> float:
    """
    Combined objective differential.
    A sum of 5 (e.g., +2 dragons, +1 baron, +2 towers) → ~0.73.
    """
    total = dragon_diff + baron_diff + tower_diff
    return _sigmoid(total / 5.0)


def _normalize_gold_diff_end(gold_diff: float) -> float:
    """Normalize final gold differential. 5000 gold gap → ~0.73."""
    return _sigmoid(gold_diff / 5000.0)


def compute_dominance_score(
    gold_diff_15: float,
    gamelength: float,
    kills_a: float,
    kills_b: float,
    dragons_a: float,
    dragons_b: float,
    barons_a: float,
    barons_b: float,
    towers_a: float,
    towers_b: float,
    totalgold_a: float,
    totalgold_b: float,
    weights: Optional[dict] = None,
) -> float:
    """
    Compute a dominance score from post-match stats.

    Returns:
        Float in [0, 1] where 0.5 = average/neutral, 1.0 = maximally dominant.
    """
    w = weights or MARGIN_WEIGHTS

    components = {
        "gold_diff_15": _normalize_gold_diff_15(gold_diff_15),
        "game_duration": _normalize_game_duration(gamelength),
        "kill_diff": _normalize_kill_diff(kills_a - kills_b),
        "objective_diff": _normalize_objective_diff(
            dragons_a - dragons_b, barons_a - barons_b, towers_a - towers_b
        ),
        "gold_diff_end": _normalize_gold_diff_end(totalgold_a - totalgold_b),
    }

    score = sum(w.get(k, 0.0) * v for k, v in components.items())
    return score


def compute_margin_multiplier(
    gold_diff_15: float,
    gamelength: float,
    kills_a: float,
    kills_b: float,
    dragons_a: float,
    dragons_b: float,
    barons_a: float,
    barons_b: float,
    towers_a: float,
    towers_b: float,
    totalgold_a: float,
    totalgold_b: float,
    winner_is_a: bool = True,
) -> float:
    """
    Compute the margin multiplier for an Elo update.

    The multiplier scales the K-factor:
        - Dominant win → multiplier > 1.0 → bigger Elo swing
        - Close win → multiplier < 1.0 → smaller Elo swing

    Args:
        winner_is_a: If True, team_a won. Stats are oriented from team_a's perspective.

    Returns:
        Float in [MARGIN_MULTIPLIER_MIN, MARGIN_MULTIPLIER_MAX].
    """
    # Orient stats from the winner's perspective
    if winner_is_a:
        gd15 = gold_diff_15
        k_a, k_b = kills_a, kills_b
        d_a, d_b = dragons_a, dragons_b
        b_a, b_b = barons_a, barons_b
        t_a, t_b = towers_a, towers_b
        g_a, g_b = totalgold_a, totalgold_b
    else:
        gd15 = -gold_diff_15
        k_a, k_b = kills_b, kills_a
        d_a, d_b = dragons_b, dragons_a
        b_a, b_b = barons_b, barons_a
        t_a, t_b = towers_b, towers_a
        g_a, g_b = totalgold_b, totalgold_a

    dominance = compute_dominance_score(
        gold_diff_15=gd15,
        gamelength=gamelength,
        kills_a=k_a,
        kills_b=k_b,
        dragons_a=d_a,
        dragons_b=d_b,
        barons_a=b_a,
        barons_b=b_b,
        towers_a=t_a,
        towers_b=t_b,
        totalgold_a=g_a,
        totalgold_b=g_b,
    )

    # Map dominance score [0, 1] to multiplier range
    # dominance=0.5 (neutral) → multiplier=1.0
    # dominance=1.0 (max dominant) → multiplier=MAX
    # dominance=0.0 (very narrow/losing-side stats) → multiplier=MIN
    multiplier = MARGIN_MULTIPLIER_MIN + (
        (MARGIN_MULTIPLIER_MAX - MARGIN_MULTIPLIER_MIN) * dominance
    )

    return max(MARGIN_MULTIPLIER_MIN, min(MARGIN_MULTIPLIER_MAX, multiplier))


def extract_mov_from_row(row: dict, winner: str) -> float:
    """
    Extract MOV multiplier from a game row (as produced by the data loader).

    This is the integration point called from the pipeline. Handles missing
    data gracefully by returning 1.0 (no adjustment) when stats are unavailable.

    Args:
        row: Dict-like game row with a_/b_ prefixed stats.
        winner: Name of the winning team.

    Returns:
        Margin multiplier float.
    """
    team_a = row.get("team_a")
    winner_is_a = (winner == team_a)

    # Extract stats with safe defaults
    def _safe(key, default=0.0):
        val = row.get(key)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)

    gold_diff_15 = _safe("a_golddiffat15", 0.0)
    gamelength = _safe("gamelength", 1800.0)  # Default 30 min
    kills_a = _safe("a_kills", 0.0)
    kills_b = _safe("b_kills", 0.0)
    dragons_a = _safe("a_dragons", 0.0)
    dragons_b = _safe("b_dragons", 0.0)
    barons_a = _safe("a_barons", 0.0)
    barons_b = _safe("b_barons", 0.0)
    towers_a = _safe("a_towers", 0.0)
    towers_b = _safe("b_towers", 0.0)
    totalgold_a = _safe("a_totalgold", 0.0)
    totalgold_b = _safe("b_totalgold", 0.0)

    # If we don't have enough data, return neutral multiplier
    if totalgold_a == 0.0 and totalgold_b == 0.0:
        return 1.0

    return compute_margin_multiplier(
        gold_diff_15=gold_diff_15,
        gamelength=gamelength,
        kills_a=kills_a,
        kills_b=kills_b,
        dragons_a=dragons_a,
        dragons_b=dragons_b,
        barons_a=barons_a,
        barons_b=barons_b,
        towers_a=towers_a,
        towers_b=towers_b,
        totalgold_a=totalgold_a,
        totalgold_b=totalgold_b,
        winner_is_a=winner_is_a,
    )
