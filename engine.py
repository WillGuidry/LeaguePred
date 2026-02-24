"""
ELO Engine
----------
The core rating system for LeaguePred.

This is Module 0 — the foundation everything else builds on.
Start here. Get this working. Validate it. Then layer edges on top.

The math is simple:
    Expected score:  E = 1 / (1 + 10^((R_opponent - R_team) / 400))
    Rating update:   R_new = R_old + K * (actual - expected)

What makes our version special:
    - Dynamic K-factor (optional)
    - Season regression (ratings decay between splits)
    - Roster change handling
    - Margin-of-victory modifier (when Module 1 is active)
"""

import math
from typing import Dict, Optional, Tuple


class EloEngine:
    """
    Core ELO rating engine for professional League of Legends teams.
    
    Basic usage:
        engine = EloEngine()
        engine.add_team("T1")
        engine.add_team("Gen.G")
        engine.process_match("T1", "Gen.G", winner="T1")
        print(engine.get_rating("T1"))
    """

    def __init__(
        self,
        default_elo: float = 1500.0,
        k_factor: float = 32.0,
        scale_factor: float = 400.0,
    ):
        """
        Initialize the ELO engine.
        
        Args:
            default_elo: Starting rating for new teams
            k_factor: How much ratings change per game (higher = more volatile)
            scale_factor: ELO difference where expected win rate is ~76%
        """
        self.default_elo = default_elo
        self.k_factor = k_factor
        self.scale_factor = scale_factor

        # Team ratings: {"team_name": current_elo}
        self.ratings: Dict[str, float] = {}

        # History: list of dicts recording every rating change
        self.history: list = []

    # =========================================================================
    # CORE METHODS
    # =========================================================================

    def add_team(self, team: str, elo: Optional[float] = None) -> None:
        """Register a team with a starting ELO."""
        if team not in self.ratings:
            self.ratings[team] = elo if elo is not None else self.default_elo

    def get_rating(self, team: str) -> float:
        """Get a team's current ELO rating."""
        if team not in self.ratings:
            self.add_team(team)
        return self.ratings[team]

    def expected_score(self, team_a: str, team_b: str) -> float:
        """
        Calculate the expected win probability for team_a against team_b.
        
        Returns:
            Float between 0 and 1 representing team_a's win probability.
        
        Example:
            >>> engine.expected_score("T1", "Gen.G")
            0.64  # T1 has a 64% expected win rate
        """
        r_a = self.get_rating(team_a)
        r_b = self.get_rating(team_b)
        return 1.0 / (1.0 + math.pow(10, (r_b - r_a) / self.scale_factor))

    def process_match(
        self,
        team_a: str,
        team_b: str,
        winner: str,
        margin_multiplier: float = 1.0,
        match_context: Optional[dict] = None,
    ) -> Tuple[float, float]:
        """
        Process a single match result and update ratings.
        
        Args:
            team_a: First team name
            team_b: Second team name
            winner: Which team won (must be team_a or team_b)
            margin_multiplier: Dominance modifier from Module 1 (default 1.0 = standard)
            match_context: Optional dict with metadata (date, patch, league, etc.)
        
        Returns:
            Tuple of (new_rating_a, new_rating_b)
        """
        # Make sure both teams exist
        self.add_team(team_a)
        self.add_team(team_b)

        # Get current ratings
        r_a = self.ratings[team_a]
        r_b = self.ratings[team_b]

        # Calculate expected scores
        e_a = self.expected_score(team_a, team_b)
        e_b = 1.0 - e_a

        # Actual scores (1 for win, 0 for loss)
        if winner == team_a:
            s_a, s_b = 1.0, 0.0
        elif winner == team_b:
            s_a, s_b = 0.0, 1.0
        else:
            raise ValueError(f"Winner '{winner}' must be either '{team_a}' or '{team_b}'")

        # Update ratings
        # R_new = R_old + K * M * (actual - expected)
        k = self.k_factor
        m = margin_multiplier

        new_r_a = r_a + k * m * (s_a - e_a)
        new_r_b = r_b + k * m * (s_b - e_b)

        # Store updated ratings
        self.ratings[team_a] = new_r_a
        self.ratings[team_b] = new_r_b

        # Log the change
        self.history.append({
            "team_a": team_a,
            "team_b": team_b,
            "winner": winner,
            "elo_a_before": r_a,
            "elo_b_before": r_b,
            "elo_a_after": new_r_a,
            "elo_b_after": new_r_b,
            "expected_a": e_a,
            "margin_multiplier": m,
            "context": match_context or {},
        })

        return new_r_a, new_r_b

    # =========================================================================
    # RATING MANAGEMENT
    # =========================================================================

    def regress_to_mean(self, fraction: float = 0.3) -> None:
        """
        Regress all ratings toward the default ELO.
        Call this between splits/seasons to prevent ratings from drifting too far.
        
        Args:
            fraction: How much to regress. 0.0 = no change, 1.0 = full reset.
                      0.3 means ratings move 30% closer to the default.
        
        Example:
            A team at 1700 with fraction=0.3:
            New rating = 1700 + 0.3 * (1500 - 1700) = 1700 - 60 = 1640
        """
        for team in self.ratings:
            current = self.ratings[team]
            self.ratings[team] = current + fraction * (self.default_elo - current)

    def handle_roster_change(self, team: str, severity: float = 0.5) -> None:
        """
        Partially reset a team's rating after a roster change.
        
        Args:
            team: Team that changed roster
            severity: How significant the change is.
                      0.0 = sub swap (minor)
                      0.5 = 2-3 players changed
                      1.0 = full rebuild
        """
        if team in self.ratings:
            current = self.ratings[team]
            self.ratings[team] = current + severity * (self.default_elo - current)

    # =========================================================================
    # REPORTING
    # =========================================================================

    def get_rankings(self) -> list:
        """
        Get all teams sorted by ELO rating (highest first).
        
        Returns:
            List of (team_name, elo_rating) tuples, sorted descending.
        """
        return sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)

    def print_rankings(self, top_n: Optional[int] = None) -> None:
        """Print a formatted leaderboard."""
        rankings = self.get_rankings()
        if top_n:
            rankings = rankings[:top_n]

        print(f"\n{'Rank':<6} {'Team':<25} {'ELO':>8}")
        print("-" * 42)
        for i, (team, elo) in enumerate(rankings, 1):
            print(f"{i:<6} {team:<25} {elo:>8.1f}")
        print()

    def predict(self, team_a: str, team_b: str) -> dict:
        """
        Generate a prediction for an upcoming match.
        
        Returns:
            Dict with prediction details:
            {
                "team_a": "T1",
                "team_b": "Gen.G",
                "elo_a": 1623.4,
                "elo_b": 1587.2,
                "elo_diff": 36.2,
                "win_prob_a": 0.552,
                "win_prob_b": 0.448,
                "predicted_winner": "T1",
            }
        """
        e_a = self.expected_score(team_a, team_b)
        r_a = self.get_rating(team_a)
        r_b = self.get_rating(team_b)

        return {
            "team_a": team_a,
            "team_b": team_b,
            "elo_a": round(r_a, 1),
            "elo_b": round(r_b, 1),
            "elo_diff": round(r_a - r_b, 1),
            "win_prob_a": round(e_a, 3),
            "win_prob_b": round(1 - e_a, 3),
            "predicted_winner": team_a if e_a >= 0.5 else team_b,
        }


# =============================================================================
# QUICK TEST
# =============================================================================
# Run this file directly to see a basic demo:
#   python -m src.elo.engine
# Or:
#   python src/elo/engine.py

if __name__ == "__main__":
    print("=" * 50)
    print("LeaguePred ELO Engine — Quick Demo")
    print("=" * 50)

    engine = EloEngine()

    # Simulate a few LCK matches
    matches = [
        ("T1", "Gen.G", "T1"),
        ("Hanwha Life", "KT Rolster", "Hanwha Life"),
        ("Gen.G", "Hanwha Life", "Gen.G"),
        ("T1", "KT Rolster", "T1"),
        ("T1", "Hanwha Life", "Hanwha Life"),
        ("Gen.G", "KT Rolster", "Gen.G"),
        ("T1", "Gen.G", "Gen.G"),
        ("Hanwha Life", "KT Rolster", "Hanwha Life"),
    ]

    print("\nProcessing matches...\n")
    for team_a, team_b, winner in matches:
        engine.process_match(team_a, team_b, winner)
        print(f"  {winner} beat {team_a if winner == team_b else team_b}")

    # Show rankings
    engine.print_rankings()

    # Make a prediction
    pred = engine.predict("T1", "Hanwha Life")
    print(f"Prediction: {pred['team_a']} ({pred['win_prob_a']:.1%}) vs {pred['team_b']} ({pred['win_prob_b']:.1%})")
    print(f"  → Predicted winner: {pred['predicted_winner']}")
    print()
