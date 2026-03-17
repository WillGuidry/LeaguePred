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
# LEAD STATE EFFICIENCY SETTINGS
# =============================================================================

# Rolling window: how many recent games to consider per team
LEAD_STATE_WINDOW = 20

# Minimum games before producing non-prior features
LEAD_STATE_MIN_GAMES = 5

# Bayesian prior weight for state-conditional features
# A team with this many qualifying games gets 50% shrinkage toward the prior
LEAD_STATE_SHRINKAGE_WEIGHT = 5.0

# Gold diff thresholds for defining "ahead" and "behind" states
AHEAD_THRESHOLD_15 = 2000   # Gold diff at 15 min to count as "ahead"
AHEAD_THRESHOLD_20 = 3000   # Gold diff at 20 min to count as "ahead"

# Feature weights for edge score computation
# These control how much each feature contributes to the style edge
LEAD_STATE_WEIGHTS = {
    "avg_gd15": 0.25,                     # Early gold lead tendency
    "avg_gd20": 0.15,                     # Mid-game gold tendency
    "first_dragon_rate": 0.10,            # Objective priority (bot-side)
    "first_herald_rate": 0.05,            # Objective priority (top-side)
    "first_tower_rate": 0.10,             # Map pressure conversion
    "first_blood_rate": 0.05,             # Early aggression
    "win_rate_when_ahead_2k_15": 0.15,    # Closeout efficiency
    "win_rate_when_ahead_3k_20": 0.05,    # Conversion from comfort
    "win_rate_when_behind_2k_15": 0.05,   # Comeback ability
    "avg_game_length_when_ahead": 0.05,   # Closing speed
}

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
    # === TIER 1: Major regions (Lolesports strength scores) ===
    "LCK": 1586,
    "LPL": 1353,
    "LEC": 1169,
    "LCP": 1156,       # Pacific (replaced PCS in 2025)
    "LCS": 1094,
    "PCS": 1100,       # Pacific Championship (pre-2025)

    # === International events ===
    "MSI": 1300,
    "WLDs": 1300,      # Worlds
    "EWC": 1200,       # Esports World Cup
    "FST": 1300,       # First Stand Tournament (annual, since 2025)
    "Asia Master": 1300,
    "ASI": 1200,
    "IC": 1200,
    "DCup": 1100,      # Demacia Cup (China)
    "KeSPA": 1100,     # Korean esports cup

    # === TIER 2: Strong secondary regions ===
    "LDL": 1200,       # LPL Development — very strong
    "LCKC": 1200,      # LCK Challengers
    "VCS": 1000,       # Vietnam
    "CBLOL": 842,      # Brazil
    "TCL": 950,        # Turkey
    "LJL": 950,        # Japan
    "LLA": 850,        # Latin America (pre-LTA)
    "LTA": 900,        # League of the Americas (2025+)
    "LTA N": 950,      # LTA North
    "LTA S": 850,      # LTA South
    "LCO": 900,        # Oceania
    "LCL": 900,        # CIS/Russia

    # === TIER 2.5: European Regional Leagues ===
    "LFL": 1000,       # France — strongest ERL
    "LVP SL": 970,     # Spain SuperLiga
    "PRM": 950,        # Germany
    "NLC": 930,        # Nordic
    "UKLC": 920,       # UK
    "UL": 920,         # Ultraliga (Poland)
    "HLL": 920,        # Hitpoint (Czech/Slovak)
    "EBL": 910,        # Balkans
    "LIT": 900,        # Italy
    "ROL": 900,        # Romania
    "AL": 900,         # Austria/Swiss
    "RL": 900,         # Iberian
    "LES": 890,        # Spain secondary
    "HW": 900,         # Hellenic (Greece)
    "GL": 880,         # Greek League
    "GLL": 880,        # GLL
    "BL": 880,         # Baltic
    "EL": 880,         # Elite League
    "EM": 1050,        # EU Masters
    "EUM": 1050,       # EU Masters (alt code)
    "NLC Aurora Open": 900,

    # === TIER 3: Development / Academy ===
    "LPLOL": 1050,     # LPL development
    "NACL": 900,       # NA Challengers
    "LCSA": 850,       # LCS Academy (pre-NACL)
    "CBLOLA": 750,     # CBLOL Academy
    "LFL2": 880,       # LFL Division 2
    "LJLA": 850,       # LJL Academy
    "PRMP": 850,       # Prime League Promotion
    "LRN": 850,        # LCS amateur north
    "LRS": 850,        # LCS amateur south
    "EBLPA": 810,      # EBL Promotion
    "GLLPA": 810,      # GLL Promotion
    "Americas Cup": 950,

    # === TIER 3.5: Smaller / amateur leagues ===
    "CD": 850, "DDH": 850, "LMF": 850, "PGN": 850, "ESLOL": 850,
    "LHE": 850, "HC": 850, "HM": 850, "UPL": 850, "NEXO": 850,
    "LAS": 850, "PCL": 850, "PGC": 850, "RCL": 850, "DL": 850,
    "CU": 850, "EPL": 850, "BIG": 850, "CDF": 850, "ASCI": 850,
    "AOL": 850, "VL": 850, "BM": 850, "GSG": 850, "EGL": 850,
    "TAL": 850, "USP": 850, "HS": 850, "OTBLX": 850, "UGP": 850,
    "NERD": 800, "NASG": 800, "SL (LATAM)": 850, "SL": 850,
    "CCWS": 900, "FST": 900, "CT": 850,
}

# Fallback for leagues not listed above
REGIONAL_ELO_DEFAULT = 900

INTERNATIONAL_EVENTS = ["MSI", "WLDs", "EWC", "Asia Master", "FST"]

# =============================================================================
# PERSISTENT INTERNATIONAL ELO SETTINGS
# =============================================================================
# Instead of discarding international Elo between tournaments, we keep a
# running "career international Elo" that decays between events.
# Teams with many international games rely less on regional priors.

# Between tournaments, regress international Elo toward the regional prior
# 0.0 = no decay (full memory), 1.0 = full reset (current behavior)
INTL_DECAY_BETWEEN_EVENTS = 0.3

# How many career international games before a team's intl Elo is fully trusted
# At this threshold, the team's prior is ~100% international Elo, ~0% regional
INTL_GAMES_FULL_TRUST = 20

# Maximum weight for international Elo when computing a team's prior
# Even at full trust, keep some regional prior influence as a safety floor
INTL_MAX_WEIGHT = 0.85

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
    "lead_state": False,
    "draft_analysis": False,
    "rolling_performance": False,
    "patch_dominance": False,
    "lane_dominance": False,
    "series_dynamics": False,
    "game_state_maps": False,
}
