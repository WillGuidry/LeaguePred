"""
Base Module
-----------
All feature modules inherit from this class.
This ensures every module has a consistent interface,
making it easy to plug modules in and out.
"""

from abc import ABC, abstractmethod


class BaseModule(ABC):
    """
    Abstract base class for all LeaguePred feature modules.
    
    Every module must implement:
        - name: a human-readable module name
        - compute(): takes match/team data, returns a feature value
        - confidence(): returns how confident the module is in its output
    
    Usage:
        class DraftModule(BaseModule):
            name = "draft_analysis"
            
            def compute(self, team_a, team_b, match_context):
                # ... your logic here ...
                return draft_edge_score
            
            def confidence(self):
                return self._confidence
    """

    name = "base_module"

    @abstractmethod
    def compute(self, team_a: str, team_b: str, match_context: dict) -> float:
        """
        Compute the feature value for a given matchup.
        
        Args:
            team_a: Name/ID of team A
            team_b: Name/ID of team B
            match_context: Dict containing relevant context like:
                - patch: current game patch
                - date: match date
                - league: which league/tournament
                - side: blue/red side assignments
                - draft: champion picks/bans (if available)
        
        Returns:
            Float representing the module's edge estimate.
            Positive = favors team_a, Negative = favors team_b.
            Zero = no edge detected.
        """
        pass

    @abstractmethod
    def confidence(self) -> float:
        """
        Return the module's confidence in its last computation.
        
        Returns:
            Float between 0.0 and 1.0.
            0.0 = no confidence (insufficient data)
            1.0 = full confidence
        """
        pass

    def is_active(self) -> bool:
        """Check if this module has enough data to produce useful output."""
        return self.confidence() > 0.1

    def __repr__(self):
        return f"<Module: {self.name}>"
