"""
Lead State Efficiency Module
------------------------------
Captures how teams generate, maintain, convert, and sometimes throw leads.

Five feature families:
    A. Lead Creation — where/how teams get ahead
    B. Advantage Quality — how robust/stable leads are
    C. Lead Conversion — translating leads into wins efficiently
    D. Throw Tendency — failing from winning states
    E. Comeback / Resistance — performing from losing states

The module outputs a signed edge score (positive = team_a advantage) and
per-team feature vectors for downstream use in ensemble models.

Leakage prevention:
    - All features come from the TeamStateTracker, which only contains
      games recorded BEFORE the current prediction
    - No future game data is ever accessed
"""

from typing import Dict, Optional, Tuple

from src.modules.base_module import BaseModule
from src.modules.team_state_tracker import TeamStateTracker
from src.config import (
    LEAD_STATE_WEIGHTS,
    LEAD_STATE_MIN_GAMES,
)


# Normalization constants for computing edge scores.
# These map raw feature differentials to approximately [-1, 1].
#
# For rate features (already [0, 1]): norm = 1.0
# For gold values: chosen so a typical meaningful gap → ~±1.0
# For time values: chosen so a ~10-min difference → ~±1.0
#
# Direction convention:
#   - Most features: higher = better for team, so (val_a - val_b) / norm
#   - Inverted features (marked below): lower = better, so (val_b - val_a) / norm

_NORMALIZATION = {
    # Family A: Lead Creation
    "avg_gd10": 1000.0,
    "avg_gd15": 1500.0,
    "first_dragon_rate": 1.0,
    "first_herald_rate": 1.0,
    "first_tower_rate": 1.0,
    "first_blood_rate": 1.0,
    "plate_diff": 3.0,               # ~3 plate diff = ±1.0

    # Family B: Advantage Quality
    "lead_stability": 1.0,
    "compound_lead_rate": 1.0,
    "gold_volatility_when_ahead": 1500.0,  # INVERTED: lower = better

    # Family C: Lead Conversion
    "win_rate_when_ahead_2k_15": 1.0,
    "win_rate_when_ahead_3k_20": 1.0,
    "close_time_ahead_15": 600.0,          # INVERTED: lower = better (faster close)
    "close_time_ahead_20": 600.0,          # INVERTED: lower = better
    "gold_snowball_rate": 5000.0,          # 5k gold growth = ±1.0
    "dragon_soul_rate": 1.0,
    "herald_tower_conv": 1.0,
    "baron_win_rate": 1.0,

    # Family D: Throw Tendency
    "throw_rate_2k_15": 1.0,              # INVERTED: lower = better
    "lead_evaporation_rate": 1.0,         # INVERTED: lower = better
    "baron_throw_rate": 1.0,              # INVERTED: lower = better

    # Family E: Comeback / Resistance
    "win_rate_when_behind_2k_15": 1.0,
    "gold_recovery_rate": 1.0,
    "extend_time_behind": 600.0,          # Higher = better (stalls longer)
}

# Features where LOWER is better for the team
# Edge computed as (val_b - val_a) / norm instead of (val_a - val_b) / norm
_INVERTED_FEATURES = {
    "gold_volatility_when_ahead",  # Less volatility = more stable leads
    "close_time_ahead_15",         # Faster close = better
    "close_time_ahead_20",         # Faster close = better
    "throw_rate_2k_15",            # Lower throw rate = better
    "lead_evaporation_rate",       # Lower evaporation = better
    "baron_throw_rate",            # Lower throw rate = better
}


class LeadStateModule(BaseModule):
    """
    Lead State Efficiency module for pre-match prediction.

    Computes a composite edge score from rolling team-level features
    across 5 families. The edge represents the style/efficiency advantage
    of team_a over team_b, independent of raw Elo strength.

    Usage:
        tracker = TeamStateTracker(window=20)
        module = LeadStateModule(tracker)

        edge = module.compute(team_a, team_b, match_context)
        confidence = module.confidence()
    """

    name = "lead_state_efficiency"

    def __init__(
        self,
        tracker: TeamStateTracker,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.tracker = tracker
        self.weights = weights or LEAD_STATE_WEIGHTS
        self._last_confidence = 0.0
        self._last_features_a = {}
        self._last_features_b = {}

    def compute(self, team_a: str, team_b: str, match_context: dict = None) -> float:
        """
        Compute the lead state edge for team_a vs team_b.

        Returns:
            Float where positive favors team_a, negative favors team_b.
            Magnitude indicates strength of style edge.
            Zero = no style edge detected (or insufficient data).
        """
        feats_a = self.tracker.get_features(team_a)
        feats_b = self.tracker.get_features(team_b)

        self._last_features_a = feats_a
        self._last_features_b = feats_b

        conf_a = self.tracker.get_confidence(team_a)
        conf_b = self.tracker.get_confidence(team_b)
        self._last_confidence = min(conf_a, conf_b)

        if self._last_confidence == 0.0:
            return 0.0

        edge = 0.0
        for feature, weight in self.weights.items():
            if feature not in feats_a or feature not in feats_b:
                continue

            val_a = feats_a[feature]
            val_b = feats_b[feature]
            norm = _NORMALIZATION.get(feature, 1.0)

            if feature in _INVERTED_FEATURES:
                diff = (val_b - val_a) / norm
            else:
                diff = (val_a - val_b) / norm

            edge += weight * diff

        edge *= self._last_confidence
        return edge

    def confidence(self) -> float:
        return self._last_confidence

    def get_team_features(self, team: str) -> dict:
        """Get the raw feature dict for a single team."""
        return self.tracker.get_features(team)

    def get_matchup_features(
        self, team_a: str, team_b: str
    ) -> Tuple[dict, dict, float]:
        """Get full feature breakdown for a matchup."""
        edge = self.compute(team_a, team_b)
        return self._last_features_a, self._last_features_b, edge

    def get_feature_contributions(
        self, team_a: str, team_b: str
    ) -> Dict[str, float]:
        """
        Break down the edge score into per-feature contributions.

        Returns:
            Dict of feature_name -> contribution to edge.
        """
        feats_a = self.tracker.get_features(team_a)
        feats_b = self.tracker.get_features(team_b)

        contributions = {}
        for feature, weight in self.weights.items():
            if feature not in feats_a or feature not in feats_b:
                continue

            val_a = feats_a[feature]
            val_b = feats_b[feature]
            norm = _NORMALIZATION.get(feature, 1.0)

            if feature in _INVERTED_FEATURES:
                diff = (val_b - val_a) / norm
            else:
                diff = (val_a - val_b) / norm

            contributions[feature] = weight * diff

        return contributions

    def get_family_contributions(
        self, team_a: str, team_b: str
    ) -> Dict[str, float]:
        """
        Aggregate edge contributions by feature family.

        Returns:
            Dict with keys: "lead_creation", "advantage_quality",
            "lead_conversion", "throw_tendency", "comeback_resistance"
        """
        contribs = self.get_feature_contributions(team_a, team_b)

        families = {
            "lead_creation": [
                "avg_gd10", "avg_gd15", "first_dragon_rate",
                "first_herald_rate", "first_tower_rate",
                "first_blood_rate", "plate_diff",
            ],
            "advantage_quality": [
                "lead_stability", "compound_lead_rate",
                "gold_volatility_when_ahead",
            ],
            "lead_conversion": [
                "win_rate_when_ahead_2k_15", "win_rate_when_ahead_3k_20",
                "close_time_ahead_15", "close_time_ahead_20",
                "gold_snowball_rate", "dragon_soul_rate",
                "herald_tower_conv", "baron_win_rate",
            ],
            "throw_tendency": [
                "throw_rate_2k_15", "lead_evaporation_rate",
                "baron_throw_rate",
            ],
            "comeback_resistance": [
                "win_rate_when_behind_2k_15", "gold_recovery_rate",
                "extend_time_behind",
            ],
        }

        family_sums = {}
        for family, features in families.items():
            family_sums[family] = sum(
                contribs.get(f, 0.0) for f in features
            )

        return family_sums
