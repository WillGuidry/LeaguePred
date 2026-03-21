"""
Simulate match predictions using regional Elo priors.

Usage:
    python simulate_matches.py
"""

import sys
sys.path.insert(0, ".")

from src.elo.engine import EloEngine
from src.config import REGIONAL_ELO_PRIORS
from predict import print_prediction, log_prediction


# Team-to-league mapping for teams we want to simulate
TEAM_LEAGUES = {
    "JDG": "LPL",
    "BLG": "LPL",
}

# Matches to simulate
MATCHES = [
    ("JDG", "BLG"),
]


def main():
    engine = EloEngine()

    # Initialize each team at their regional Elo prior
    for team, league in TEAM_LEAGUES.items():
        prior = REGIONAL_ELO_PRIORS.get(league, 1500)
        engine.add_team(team, elo=prior)

    print("\n" + "=" * 60)
    print("  ESPORTS MATCH SIMULATIONS (Regional Elo Priors)")
    print("=" * 60)
    print("\n  Teams initialized at regional Elo priors:")
    for team, league in TEAM_LEAGUES.items():
        elo = engine.get_rating(team)
        print(f"    {team:<25} ({league})  Elo: {elo:.0f}")

    for team_a, team_b in MATCHES:
        pred = engine.predict(team_a, team_b)
        league_a = TEAM_LEAGUES[team_a]
        league_b = TEAM_LEAGUES[team_b]
        pred["league_a"] = league_a
        pred["league_b"] = league_b
        pred["cross_regional"] = False
        print_prediction(pred)
        log_prediction(pred)


if __name__ == "__main__":
    main()
