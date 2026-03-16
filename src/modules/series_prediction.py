"""
Series Score Prediction
-----------------------
Given a per-game win probability, compute the probability of each possible
series score for Best-of-3 (BO3) and Best-of-5 (BO5) formats.

Math:
    For team A with per-game win prob p in a BO-N series (first to W wins):
    P(A wins W-k) = C(W-1+k, k) * p^W * (1-p)^k
    because A must win the last game, and in the preceding (W-1+k) games,
    A wins (W-1) and B wins k.

Usage:
    from src.modules.series_prediction import predict_series

    # BO3 with team A having 60% per-game win probability
    result = predict_series(0.60, best_of=3)
    print(result)

    # From moneyline odds
    result = predict_series_from_odds(-150, +130, best_of=3)
"""

from math import comb


def predict_series(win_prob_a: float, best_of: int = 3) -> dict:
    """
    Compute series score probabilities from a per-game win probability.

    Args:
        win_prob_a: Team A's probability of winning any single game (0-1).
        best_of: Series format — 1, 3, or 5.

    Returns:
        Dict with:
            - team_a_wins: probability team A wins the series
            - team_b_wins: probability team B wins the series
            - scores: dict mapping "W-L" to probability for each possible score
            - expected_games: expected number of games in the series
            - most_likely_score: the single most probable series score
    """
    if not 0 <= win_prob_a <= 1:
        raise ValueError(f"win_prob_a must be between 0 and 1, got {win_prob_a}")

    if best_of not in (1, 3, 5):
        raise ValueError(f"best_of must be 1, 3, or 5, got {best_of}")

    p = win_prob_a
    q = 1 - p
    wins_needed = (best_of + 1) // 2  # 2 for BO3, 3 for BO5

    if best_of == 1:
        return {
            "team_a_wins": p,
            "team_b_wins": q,
            "scores": {"1-0": p, "0-1": q},
            "expected_games": 1.0,
            "most_likely_score": "1-0" if p >= q else "0-1",
        }

    scores = {}
    team_a_wins = 0.0
    team_b_wins = 0.0
    expected_games = 0.0

    for losses in range(wins_needed):
        # Team A wins (wins_needed)-(losses): A wins the last game,
        # and in the prior (wins_needed - 1 + losses) games, A won (wins_needed - 1)
        prior_games = wins_needed - 1 + losses
        prob_a = comb(prior_games, losses) * (p ** wins_needed) * (q ** losses)
        total_games = wins_needed + losses
        score_label = f"{wins_needed}-{losses}"
        scores[score_label] = prob_a
        team_a_wins += prob_a
        expected_games += prob_a * total_games

        # Team B wins (wins_needed)-(losses): mirror
        prob_b = comb(prior_games, losses) * (q ** wins_needed) * (p ** losses)
        score_label_b = f"{losses}-{wins_needed}"
        scores[score_label_b] = prob_b
        team_b_wins += prob_b
        expected_games += prob_b * total_games

    # Sort scores by probability descending
    scores = dict(sorted(scores.items(), key=lambda x: -x[1]))
    most_likely = max(scores, key=scores.get)

    return {
        "team_a_wins": team_a_wins,
        "team_b_wins": team_b_wins,
        "scores": scores,
        "expected_games": expected_games,
        "most_likely_score": most_likely,
    }


def odds_to_prob(odds: float) -> float:
    """
    Convert American moneyline odds to implied probability.

    Examples:
        -150 -> 0.6000 (risk 150 to win 100)
        +130 -> 0.4348 (risk 100 to win 130)
        -200 -> 0.6667
        +200 -> 0.3333
    """
    if odds < 0:
        return abs(odds) / (abs(odds) + 100)
    else:
        return 100 / (odds + 100)


def decimal_to_prob(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability (e.g., 1.67 -> 0.5988)."""
    return 1 / decimal_odds


def remove_vig(prob_a: float, prob_b: float) -> tuple:
    """
    Remove the bookmaker's vig (overround) to get fair probabilities.
    The raw implied probabilities from odds sum to > 1.0 due to the vig.
    We normalize them back to sum to 1.0.
    """
    total = prob_a + prob_b
    return prob_a / total, prob_b / total


def predict_series_from_odds(
    odds_a: float,
    odds_b: float,
    best_of: int = 3,
    odds_format: str = "american",
) -> dict:
    """
    Predict series score probabilities from betting odds.

    IMPORTANT: Betting odds for a series (e.g., BO3 match winner) already
    price in the series format. To get per-game win probability, we need to
    back it out. This function handles that conversion.

    Args:
        odds_a: Odds for team A (series winner odds from the book).
        odds_b: Odds for team B (series winner odds from the book).
        best_of: Series format (3 or 5).
        odds_format: "american" (-150/+130) or "decimal" (1.67/2.30).

    Returns:
        Same dict as predict_series(), plus:
            - implied_series_prob_a/b: raw implied probs from odds
            - fair_series_prob_a/b: after removing vig
            - per_game_prob_a: the backed-out per-game win probability
    """
    # Convert odds to implied probability
    if odds_format == "american":
        raw_a = odds_to_prob(odds_a)
        raw_b = odds_to_prob(odds_b)
    elif odds_format == "decimal":
        raw_a = decimal_to_prob(odds_a)
        raw_b = decimal_to_prob(odds_b)
    else:
        raise ValueError(f"Unknown odds_format: {odds_format}")

    fair_a, fair_b = remove_vig(raw_a, raw_b)

    # Back out per-game win probability from series win probability
    per_game_p = _series_prob_to_game_prob(fair_a, best_of)

    result = predict_series(per_game_p, best_of)
    result["implied_series_prob_a"] = raw_a
    result["implied_series_prob_b"] = raw_b
    result["fair_series_prob_a"] = fair_a
    result["fair_series_prob_b"] = fair_b
    result["per_game_prob_a"] = per_game_p
    return result


def _series_win_prob(p: float, best_of: int) -> float:
    """Given per-game win prob p, compute probability of winning the series."""
    wins_needed = (best_of + 1) // 2
    total = 0.0
    for losses in range(wins_needed):
        prior_games = wins_needed - 1 + losses
        total += comb(prior_games, losses) * (p ** wins_needed) * ((1 - p) ** losses)
    return total


def _series_prob_to_game_prob(series_prob: float, best_of: int) -> float:
    """
    Back out the per-game win probability from a series win probability.
    Uses binary search since the mapping is monotonic.

    Example: If odds imply team A has 65% to win a BO3, what's their
    per-game win probability? (Answer: ~57.5%)
    """
    if best_of == 1:
        return series_prob

    lo, hi = 0.0, 1.0
    for _ in range(100):  # binary search converges quickly
        mid = (lo + hi) / 2
        if _series_win_prob(mid, best_of) < series_prob:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def print_series_prediction(result: dict, team_a: str = "Team A", team_b: str = "Team B") -> None:
    """Pretty-print series score prediction."""
    print(f"\n{'='*60}")
    print(f"  Series Score Prediction: {team_a} vs {team_b}")
    print(f"{'='*60}")

    if "per_game_prob_a" in result:
        print(f"\n  Odds-derived probabilities:")
        print(f"    Series win (fair):  {team_a} {result['fair_series_prob_a']:.1%}  vs  {team_b} {result['fair_series_prob_b']:.1%}")
        print(f"    Per-game win prob:  {team_a} {result['per_game_prob_a']:.1%}  vs  {team_b} {1-result['per_game_prob_a']:.1%}")

    print(f"\n  Series winner probability:")
    print(f"    {team_a}: {result['team_a_wins']:.1%}")
    print(f"    {team_b}: {result['team_b_wins']:.1%}")

    print(f"\n  Score probabilities:")
    for score, prob in result["scores"].items():
        w, l = score.split("-")
        if int(w) > int(l):
            label = f"    {team_a} {score}"
        else:
            label = f"    {team_b} {l}-{w}"
        bar_len = int(prob * 40)
        bar = "\u2588" * bar_len
        print(f"{label:<30} {prob:>6.1%}  {bar}")

    print(f"\n  Expected games: {result['expected_games']:.2f}")
    ml = result["most_likely_score"]
    w, l = ml.split("-")
    if int(w) > int(l):
        print(f"  Most likely score: {team_a} {ml}")
    else:
        print(f"  Most likely score: {team_b} {l}-{w}")
    print()
