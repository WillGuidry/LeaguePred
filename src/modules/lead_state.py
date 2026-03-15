"""
Lead State Efficiency Module
------------------------------
Captures how teams generate, express, and convert advantages.

This module computes team-level style factors from historical game data:
    - Family A (Lead Creation): Where/how teams get ahead
    - Family C (Lead Conversion): How efficiently teams close from winning positions

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


# Normalization constants for computing edge scores
# These map raw feature values to approximately [-1, 1] for weighting
_NORMALIZATION = {
    "avg_gd15": 1500.0,        # 1500 gold diff → ±1.0
    "avg_gd20": 2500.0,        # 2500 gold diff → ±1.0
    "first_dragon_rate": 1.0,  # Already in [0, 1]
    "first_herald_rate": 1.0,
    "first_tower_rate": 1.0,
    "first_blood_rate": 1.0,
    "win_rate_when_ahead_2k_15": 1.0,
    "win_rate_when_ahead_3k_20": 1.0,
    "win_rate_when_behind_2k_15": 1.0,
    "avg_game_length_when_ahead": 600.0,  # 10-min range → ±1.0
}


class LeadStateModule(BaseModule):
    """
    Lead State Efficiency module for pre-match prediction.

    Computes a composite edge score from rolling team-level features.
    The edge score represents the style/efficiency advantage of team_a
    over team_b, independent of raw Elo strength.

    Usage:
        tracker = TeamStateTracker(window=20)
        module = LeadStateModule(tracker)

        # In prediction loop:
        edge = module.compute(team_a, team_b, match_context)
        confidence = module.confidence()
    """

    name = "lead_state_efficiency"

    def __init__(
        self,
        tracker: TeamStateTracker,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            tracker: TeamStateTracker instance with game history.
            weights: Feature weights for edge computation.
                     If None, uses LEAD_STATE_WEIGHTS from config.
        """
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

        # Compute confidence from both teams' data availability
        conf_a = self.tracker.get_confidence(team_a)
        conf_b = self.tracker.get_confidence(team_b)
        self._last_confidence = min(conf_a, conf_b)

        if self._last_confidence == 0.0:
            return 0.0

        # Compute weighted edge
        edge = 0.0
        for feature, weight in self.weights.items():
            if feature not in feats_a or feature not in feats_b:
                continue

            val_a = feats_a[feature]
            val_b = feats_b[feature]
            norm = _NORMALIZATION.get(feature, 1.0)

            if feature == "avg_game_length_when_ahead":
                # Lower game length when ahead = better (faster closer)
                # Invert so positive = team_a closes faster
                diff = (val_b - val_a) / norm
            else:
                diff = (val_a - val_b) / norm

            edge += weight * diff

        # Scale by confidence so low-data predictions are dampened
        edge *= self._last_confidence

        return edge

    def confidence(self) -> float:
        """Return confidence from the last compute() call."""
        return self._last_confidence

    def get_team_features(self, team: str) -> dict:
        """Get the raw feature dict for a single team (for analysis/export)."""
        return self.tracker.get_features(team)

    def get_matchup_features(
        self, team_a: str, team_b: str
    ) -> Tuple[dict, dict, float]:
        """
        Get full feature breakdown for a matchup.

        Returns:
            (features_a, features_b, edge_score)
        """
        edge = self.compute(team_a, team_b)
        return self._last_features_a, self._last_features_b, edge

    def get_feature_contributions(
        self, team_a: str, team_b: str
    ) -> Dict[str, float]:
        """
        Break down the edge score into per-feature contributions.

        Useful for understanding which features drive the prediction.

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

            if feature == "avg_game_length_when_ahead":
                diff = (val_b - val_a) / norm
            else:
                diff = (val_a - val_b) / norm

            contributions[feature] = weight * diff

        return contributions
