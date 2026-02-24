"""
LeaguePred Global Settings
--------------------------
Central configuration for the entire prediction system.
Tune these values as you learn what works best.
"""

# =============================================================================
# ELO SETTINGS
# =============================================================================

# Starting ELO for new teams
DEFAULT_ELO = 1500

# K-factor: how much ELO changes per game
# Higher = more reactive, Lower = more stable
# Chess uses 10-40 depending on player level
# We start at 32 and can tune from there
K_FACTOR = 32

# Between splits, regress ratings toward the mean by this fraction
# 0.0 = no regression, 1.0 = full reset to DEFAULT_ELO
SEASON_REGRESSION = 0.3

# How much to reset ELO when a team makes a major roster change (2+ players)
ROSTER_CHANGE_REGRESSION = 0.5

# =============================================================================
# MARGIN OF VICTORY SETTINGS
# =============================================================================

# Dominance score weights (should sum to 1.0)
MARGIN_WEIGHTS = {
    "gold_diff_15": 0.25,      # Gold lead at 15 minutes
    "game_duration": 0.20,     # Shorter + bigger lead = more dominant
    "kill_diff": 0.15,         # Kill differential
    "objective_diff": 0.20,    # Dragon + Baron + Tower differential
    "gold_diff_end": 0.20,     # Final gold differential
}

# Cap the dominance multiplier so one game can't swing ELO too wildly
MARGIN_MULTIPLIER_MIN = 0.5
MARGIN_MULTIPLIER_MAX = 2.0

# =============================================================================
# ROLLING PERFORMANCE SETTINGS
# =============================================================================

# Number of recent games to consider for momentum
MOMENTUM_WINDOW_SHORT = 5
MOMENTUM_WINDOW_MEDIUM = 10
MOMENTUM_WINDOW_LONG = 15

# =============================================================================
# DRAFT SETTINGS
# =============================================================================

# Minimum games on a champion before we trust the win rate
CHAMPION_MIN_GAMES = 5

# How much to weight team-specific champion stats vs global stats
# 1.0 = only team stats, 0.0 = only global stats
CHAMPION_TEAM_WEIGHT = 0.6

# =============================================================================
# PREDICTION SETTINGS
# =============================================================================

# ELO difference that corresponds to ~76% win probability
# In standard ELO, this is 400 (a 400-point gap = ~90% expected)
# We may want to adjust this for LoL's specific dynamics
ELO_SCALE_FACTOR = 400

# =============================================================================
# DATA SETTINGS
# =============================================================================

# Supported regions/leagues
REGIONS = [
    "LCK",    # Korea
    "LPL",    # China
    "LEC",    # Europe
    "LCS",    # North America
    "PCS",    # Pacific
    "VCS",    # Vietnam
    "LLA",    # Latin America
    "CBLOL",  # Brazil
    "LJL",    # Japan
]

INTERNATIONAL_EVENTS = ["MSI", "Worlds"]

# =============================================================================
# MODULE ACTIVATION
# =============================================================================
# Toggle feature modules on/off
# Start with just ELO, then enable modules as you build them

ACTIVE_MODULES = {
    "margin_of_victory": False,
    "draft_analysis": False,
    "rolling_performance": False,
    "patch_dominance": False,
    "lane_dominance": False,
    "series_dynamics": False,
    "game_state_maps": False,
}
