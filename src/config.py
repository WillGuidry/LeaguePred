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
# REGIONAL STRENGTH PRIORS
# =============================================================================
# Source: Lolesports regional strength model
# Teams in each league start at this Elo instead of the flat default.
# This prevents the system from treating a CBLOL team as equal to an LCK team.
#
# For leagues not listed here, we estimate based on tier:
#   Tier 1 (major regions): LCK, LPL, LEC, LCS — use Lolesports scores
#   Tier 2 (secondary): regional leagues with international exposure
#   Tier 3 (minor): development leagues, academy
#
# These priors get washed out over time as the Elo system sees results,
# but they matter a lot early on when we have few games per team.

REGIONAL_ELO_PRIORS = {
    # Major regions (Lolesports scores)
    "LCK": 1586,
    "LPL": 1353,
    "LEC": 1169,
    "LCP": 1156,
    "LCS": 1094,
    "CBLOL": 842,

    # Secondary regions (estimated relative to majors)
    "VCS": 1000,
    "LJL": 950,
    "TCL": 950,
    "PCS": 1050,       # Absorbed into LCP but may appear in older data
    "LLA": 850,

    # European regional leagues (below LEC, above minor)
    "LFL": 1000,       # French league, strongest ERL
    "PRM": 950,        # German league
    "NLC": 930,        # Nordic
    "HLL": 920,        # Hitpoint league
    "EBL": 910,        # Balkans
    "LIT": 900,        # Italy
    "ROL": 900,        # Romania  -- guessing here but we can look later
    "AL": 900,         # Austria
    "RL": 900,         # Iberian
    "LES": 890,        # Spain
    "EM": 1050,        # EU Masters (mix of ERL top teams)

    # Americas secondary
    "Americas Cup": 950,
    "CD": 850,

    # Academy / development
    "LCKC": 1200,      # LCK Challengers — feeder to LCK
    "LPLOL": 1050,     # LPL development
    "LRN": 850,        # LCS amateur north
    "LRS": 850,        # LCS amateur south
    "HW": 900,         # Hellenic league -- guessing, we can research

    # Catch-all events
    "CCWS": 900,       # Community/wildcard events -- guessing
}

# Fallback for leagues not listed above
REGIONAL_ELO_DEFAULT = 950

INTERNATIONAL_EVENTS = ["MSI", "Worlds"]

# =============================================================================
# ROSTER CHANGE SETTINGS
# =============================================================================
# How many players must change for it to count as a roster change
# LoL teams have 5 players. Swapping 1 sub is minor, 3+ is major.
ROSTER_CHANGE_MINOR_THRESHOLD = 1   # 1 player changed
ROSTER_CHANGE_MAJOR_THRESHOLD = 3   # 3+ players changed
ROSTER_MINOR_REGRESSION = 0.15      # Regress 15% toward regional mean
ROSTER_MAJOR_REGRESSION = 0.40      # Regress 40% toward regional mean

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
