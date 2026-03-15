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
# These control how much each feature contributes to the style edge.
# Organized into 5 families — weights within each family can be tuned,
# and full-family ablation reveals which families carry signal.

LEAD_STATE_WEIGHTS = {
    # --- Family A: Lead Creation (total ~0.22) ---
    "avg_gd10": 0.03,                     # Early lane phase gold tendency
    "avg_gd15": 0.08,                     # Mid-early gold lead tendency
    "first_dragon_rate": 0.03,            # Bot-side objective priority
    "first_herald_rate": 0.02,            # Top-side objective priority
    "first_tower_rate": 0.03,             # Map pressure conversion
    "first_blood_rate": 0.01,             # Early aggression signal
    "plate_diff": 0.02,                   # Laning phase tower pressure

    # --- Family B: Advantage Quality (total ~0.10) ---
    "lead_stability": 0.05,              # Lead at 15 still held at 20
    "compound_lead_rate": 0.02,          # Multi-dimensional leads
    "gold_volatility_when_ahead": 0.03,  # Lead chaos (lower = better)

    # --- Family C: Lead Conversion (total ~0.38) ---
    "win_rate_when_ahead_2k_15": 0.10,   # Closeout efficiency from 15
    "win_rate_when_ahead_3k_20": 0.04,   # Closeout from comfortable lead
    "close_time_ahead_15": 0.06,         # Closing speed from +2k@15
    "close_time_ahead_20": 0.03,         # Closing speed from +3k@20
    "gold_snowball_rate": 0.05,          # Lead growth from 15 to end
    "dragon_soul_rate": 0.03,            # Dragon-to-soul conversion
    "herald_tower_conv": 0.04,           # Herald-to-tower efficiency
    "baron_win_rate": 0.03,              # Baron-to-win conversion

    # --- Family D: Throw Tendency (total ~0.15) ---
    "throw_rate_2k_15": 0.06,            # Lose% from +2k@15 (anti-signal)
    "lead_evaporation_rate": 0.05,       # Lead held at 15 lost by 20
    "baron_throw_rate": 0.04,            # Had baron + lead, still lost

    # --- Family E: Comeback / Resistance (total ~0.15) ---
    "win_rate_when_behind_2k_15": 0.06,  # Win from -2k@15 deficit
    "gold_recovery_rate": 0.05,          # Behind@15 → ahead@20
    "extend_time_behind": 0.04,          # Stalling ability when losing
}

# =============================================================================
# ROLLING MOMENTUM SETTINGS
# =============================================================================

# Window sizes for recent form tracking
MOMENTUM_WINDOW_SHORT = 5    # Hot streak / slump detection
MOMENTUM_WINDOW_MEDIUM = 10  # Short-term form
MOMENTUM_WINDOW_LONG = 15    # Sustained form baseline

# Exponential decay half-life (in games) for weighting recent results
# A game from half_life games ago has 50% the weight of the most recent game
MOMENTUM_DECAY_HALF_LIFE = 4.0

# How much weight each signal gets in the composite momentum score
MOMENTUM_SIGNAL_WEIGHTS = {
    "win_rate_short": 0.30,       # Recent win rate (short window)
    "win_rate_medium": 0.15,      # Medium-term win rate
    "streak": 0.20,               # Current streak bonus/penalty
    "elo_trend": 0.20,            # Elo trajectory (rising/falling)
    "dominance_trend": 0.15,      # Quality of recent wins
}

# Streak scaling: diminishing returns on long streaks
# 3-game streak = full value, 6+ = capped
MOMENTUM_STREAK_CAP = 6

# Minimum games before momentum module activates
MOMENTUM_MIN_GAMES = 5

# =============================================================================
# SERIES DYNAMICS SETTINGS
# =============================================================================

# Track how teams adapt within a Bo3/Bo5 and under pressure

# Weight for game-1 performance vs later games in adaptation score
SERIES_GAME1_WEIGHT = 0.40

# Minimum Bo3/Bo5 series before trusting adaptation stats
SERIES_MIN_SERIES = 5

# Rolling window of recent series to consider
SERIES_WINDOW = 20

# Feature weights for series edge computation
SERIES_SIGNAL_WEIGHTS = {
    "game1_win_rate": 0.25,          # How often team wins game 1
    "adaptation_rate": 0.25,         # Win rate in games 2+ relative to game 1
    "elimination_win_rate": 0.20,    # Performance in must-win games
    "reverse_sweep_rate": 0.15,      # Ability to come back from 0-1 / 0-2
    "closeout_rate": 0.15,           # Converting match point (2-1 in Bo3, 3-2 in Bo5)
}

# Bayesian prior weight for series features (same concept as lead state)
SERIES_SHRINKAGE_WEIGHT = 5.0

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
    # === TIER 1: Major regions ===
    # Data-driven from 2025 international event win rates (Elo formula):
    #   LCK 59.8% (266g), LPL 51.2% (285g), LEC 39.1% (69g), LCP 34.3% (67g)
    # Anchored at LCK=1500, gaps = -400*log10(1/WR - 1) relative to LCK
    "LCK": 1500,        # 59.8% intl WR, 266 games
    "LPL": 1440,        # 51.2% intl WR, 285 games
    "LEC": 1354,        # 39.1% intl WR, 69 games
    "LCP": 1319,        # 34.3% intl WR, 67 games (Pacific, 2025+)
    "LCS": 1320,        # ~LCP level (merged into LTA 2025)
    "PCS": 1311,        # 33.3% intl WR, 39 games (Pacific, pre-2025)

    # === International events ===
    # Field is pre-selected; use mid-point between top and bottom regions
    "MSI": 1450,
    "WLDs": 1450,       # Worlds
    "EWC": 1400,        # Esports World Cup
    "Asia Master": 1450,
    "ASI": 1400,
    "IC": 1400,
    "DCup": 1400,       # Demacia Cup (China domestic)
    "KeSPA": 1400,      # Korean esports cup

    # === TIER 2: Secondary regions (data-driven where available) ===
    "LCKC": 1507,       # 60.7% intl WR, 173 games (!)
    "LDL": 1280,        # LPL Development (est. LPL - 160)
    "VCS": 1240,        # 25.0% intl WR, 16 games
    "TCL": 1200,        # Turkey (est. between VCS and LTA N)
    "CBLOL": 1150,      # Brazil (est. ~LJL level)
    "LJL": 1222,        # 23.1% intl WR, 26 games
    "LLA": 1150,        # Latin America (pre-LTA)
    "LTA": 1200,        # League of the Americas (2025+)
    "LTA N": 1384,      # 43.2% intl WR, 44 games
    "LTA S": 1324,      # 35.0% intl WR, 20 games
    "LCO": 1150,        # Oceania (est. ~CBLOL level)
    "LCL": 1150,        # CIS/Russia

    # === TIER 2.5: European Regional Leagues ===
    # LVP SL measured at 58.1% intl WR (43 games) = 1488
    # Other ERLs estimated relative to LVP SL using EU Masters performance
    "LFL": 1490,        # France — traditionally strongest ERL, ~LVP SL
    "LVP SL": 1488,     # 58.1% intl WR, 43 games (measured)
    "PRM": 1420,        # Germany — strong ERL (~LVP SL - 70)
    "NLC": 1380,        # Nordic (~LVP SL - 110)
    "UKLC": 1340,       # UK
    "UL": 1340,         # Ultraliga (Poland)
    "HLL": 1320,        # Hitpoint (Czech/Slovak)
    "EBL": 1300,        # Balkans
    "LIT": 1300,        # Italy
    "ROL": 1300,        # Romania
    "AL": 1300,         # Austria/Swiss
    "RL": 1300,         # Iberian
    "LES": 1280,        # Spain secondary
    "HW": 1300,         # Hellenic (Greece)
    "GL": 1280,         # Greek League
    "GLL": 1280,        # GLL
    "BL": 1280,         # Baltic
    "EL": 1280,         # Elite League
    "EM": 1450,         # EU Masters (top ERL teams, measured-adjacent)
    "EUM": 1450,        # EU Masters (alt code)
    "NLC Aurora Open": 1300,

    # === TIER 3: Development / Academy ===
    # Estimated as parent league minus ~160-200 points
    "LPLOL": 1280,      # LPL development (~LPL - 160)
    "NACL": 1150,       # NA Challengers (~LCS - 170)
    "LCSA": 1100,       # LCS Academy (pre-NACL)
    "CBLOLA": 1000,     # CBLOL Academy
    "LFL2": 1320,       # LFL Division 2 (~LFL - 170)
    "LJLA": 1070,       # LJL Academy
    "PRMP": 1250,       # Prime League Promotion (~PRM - 170)
    "LRN": 1100,        # LCS amateur north
    "LRS": 1100,        # LCS amateur south
    "EBLPA": 1130,      # EBL Promotion
    "GLLPA": 1110,      # GLL Promotion
    "Americas Cup": 1200,

    # === TIER 3.5: Smaller / amateur leagues ===
    # No international data; estimated at ~1100 (between academy and minor region)
    "NEXO": 1120,       # 14.3% intl WR, 7 games (measured but tiny sample)
    "CD": 1100, "DDH": 1100, "LMF": 1100, "PGN": 1100, "ESLOL": 1100,
    "LHE": 1100, "HC": 1100, "HM": 1100, "UPL": 1100,
    "LAS": 1100, "PCL": 1100, "PGC": 1100, "RCL": 1100, "DL": 1100,
    "CU": 1100, "EPL": 1100, "BIG": 1100, "CDF": 1100, "ASCI": 1100,
    "AOL": 1100, "VL": 1100, "BM": 1100, "GSG": 1100, "EGL": 1100,
    "TAL": 1100, "USP": 1100, "HS": 1100, "OTBLX": 1100, "UGP": 1100,
    "NERD": 1050, "NASG": 1050, "SL (LATAM)": 1100, "SL": 1100,
    "CCWS": 1150, "FST": 1150, "CT": 1100,
}

# Fallback for leagues not listed above
REGIONAL_ELO_DEFAULT = 900

INTERNATIONAL_EVENTS = ["MSI", "WLDs", "EWC", "Asia Master"]

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
