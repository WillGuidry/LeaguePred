# LeaguePred 🎮📊

## A Ground-Up ELO Prediction System for Professional League of Legends

LeaguePred is a modular, data-driven prediction engine for professional League of Legends esports. It starts from a foundational ELO rating system and layers analytical edges on top — champion draft intelligence, rolling form, patch meta awareness, lane pressure differentials, and more — to produce sharper win probability estimates than raw ELO alone.

The philosophy is simple: **ELO tells you who is better. The edges tell you why and by how much.**

This project is built for internal analytical use — a tool for people who watch pro LoL seriously and want to understand matchups at a deeper level than "Team A is ranked higher."

---

## Table of Contents

1. [Project Philosophy](#project-philosophy)
2. [System Architecture](#system-architecture)
3. [Module Breakdown](#module-breakdown)
4. [Data Sources & Requirements](#data-sources--requirements)
5. [Getting Started](#getting-started)
6. [Project Structure](#project-structure)
7. [Development Roadmap](#development-roadmap)
8. [Analytical Framework](#analytical-framework)
9. [Technical Notes](#technical-notes)
10. [Glossary](#glossary)

---

## Project Philosophy

### Why ELO as the Foundation?

ELO is elegant because it's self-correcting. A team that keeps winning sees their rating climb, which means future wins against weaker opponents yield smaller gains. It naturally handles the problem of "how good is this team relative to the field?" without needing a human to decide.

But raw ELO has blind spots in League of Legends:

- **It doesn't know about draft.** A team running a comfort composition on a favorable patch is a different animal than the same team forced onto off-meta picks.
- **It doesn't understand game state.** A team that wins a 45-minute nailbiter and a team that demolishes in 22 minutes get the same ELO reward — but one of those results tells you a lot more about relative strength.
- **It can't see momentum.** A team on a 7-game tear with a new jungler is not the same team their ELO from three weeks ago suggests.
- **It ignores the meta.** Patch shifts can completely restructure the competitive landscape overnight. A team dominant on 14.1 may be mediocre on 14.3 if their champion pool gets nerfed.

LeaguePred's job is to keep ELO's strengths (simplicity, self-correction, interpretability) while systematically addressing its weaknesses through modular feature layers.

### The Analyst Mindset

This project is designed to think like the best analysts in the scene:

- **Like LS:** Respect draft as a game-deciding factor. Champion-level data matters. Composition synergy matters. Understanding which team has inevitability (scaling advantage) vs. which team needs to force the pace — that's information ELO doesn't capture but draft analysis can.
- **Like Azael:** Game flow is everything. Gold leads at 15 minutes, objective sequencing, the difference between a team that wins through superior macro and one that wins through individual outplays. How a team wins tells you how likely they are to keep winning.
- **Like Thorin:** Historical context, form, momentum, head-to-head dynamics. The narrative around a team matters because it often reflects real structural advantages — coaching quality, roster synergy, mental resilience in series play.

The goal is not to replace human analysis but to give it a quantitative backbone.

---

## System Architecture

LeaguePred is built as a **pipeline of modular components**, each feeding into a central prediction engine.

```
┌─────────────────────────────────────────────────────┐
│                   DATA LAYER                         │
│  Match History · Draft Data · Player Stats · Patch   │
│  Notes · Objective Stats · Gold/XP Timelines         │
└──────────────────────┬──────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────┐
│               PROCESSING LAYER                       │
│                                                      │
│  ┌──────────┐  ┌───────────┐  ┌──────────────────┐  │
│  │  ELO     │  │  Feature  │  │  Game State      │  │
│  │  Engine  │  │  Modules  │  │  Analyzer         │  │
│  └────┬─────┘  └─────┬─────┘  └────────┬─────────┘  │
│       │              │                  │            │
│       ▼              ▼                  ▼            │
│  ┌─────────────────────────────────────────────┐     │
│  │          FEATURE AGGREGATOR                  │     │
│  │   Combines ELO + all active feature modules  │     │
│  └──────────────────┬──────────────────────────┘     │
│                     │                                │
└─────────────────────┼────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────┐
│              PREDICTION LAYER                        │
│                                                      │
│  ┌──────────────┐  ┌─────────────────────────────┐   │
│  │  Win Prob     │  │  XGBoost Feature Optimizer  │   │
│  │  Calculator   │  │  (selects best edges)       │   │
│  └──────┬───────┘  └──────────────┬──────────────┘   │
│         │                         │                  │
│         ▼                         ▼                  │
│  ┌─────────────────────────────────────────────┐     │
│  │           OUTPUT & REPORTING                 │     │
│  │  Match Predictions · Confidence Levels ·     │     │
│  │  Feature Attribution · Model Performance     │     │
│  └─────────────────────────────────────────────┘     │
│                                                      │
└─────────────────────────────────────────────────────┘
```

### Why Modular?

Professional LoL data is messy. Some leagues have detailed timeline data. Others barely have post-game stats. Patches change every two weeks. Rosters shuffle mid-split. A rigid model breaks under these conditions.

Modular design means every feature layer is a plug-in. If champion draft data is unavailable for a particular league, you disable that module and the system still works — it just falls back to the features it does have. If you discover that a new feature (say, scuttle crab control rate) has predictive power, you can add it without rewriting the core.

Each module has a standard interface: it takes in match/team data, outputs a numerical adjustment or feature vector, and reports its own confidence level.

---

## Module Breakdown

### Module 0: Core ELO Engine
**Status: Build First**
**Priority: Foundation — everything depends on this**

The base rating system. Every team starts at a default rating (typically 1500). After each match, ratings adjust based on the result and the expected outcome.

Key design decisions:
- **K-factor tuning:** How much ratings move per game. Too high and ratings are noisy. Too low and they don't react to real changes (roster swaps, meta shifts). We'll likely want a dynamic K-factor that's higher early in a split and lower as more games are played.
- **Regional scaling:** Is an LCK 1500 the same as an LCS 1500? Probably not. We need a way to calibrate across regions, likely using international events (MSI, Worlds) as anchor points.
- **Decay:** Teams that haven't played in a while (off-season) should have their rating regress toward the mean. How aggressively is a tuning parameter.
- **New team/roster handling:** When a team makes a significant roster change, their ELO should partially reset. How much depends on how many players changed and which roles.

### Module 1: Margin of Victory (xG-Style Adjustments)
**Status: Build Second**
**Priority: High — addresses ELO's biggest blind spot**

Standard ELO treats all wins the same. A 1-kill game that went 50 minutes is the same as a 20-kill stomp at 25 minutes. This module fixes that.

We create a "dominance score" for each game based on:
- **Gold differential at 15 minutes** — early game execution
- **Game duration** — shorter games with larger leads indicate more decisive victories
- **Kill differential** — raw aggression gap
- **Objective differential** — dragons, barons, towers taken vs. given
- **Gold differential at end** — total economic dominance

The dominance score modifies how much ELO changes after a game. A dominant win earns more ELO. A narrow, lucky win earns less. A competitive loss costs less than a blowout loss.

Think of it like expected goals (xG) in football — the scoreline says 1-0, but if one team had 25 shots and the other had 2, the xG tells a different story. We want that same insight for LoL.

### Module 2: Champion Draft Analysis
**Status: Build Third**
**Priority: High — the LS module**

Draft is arguably the most undervalued predictor in competitive LoL. This module tracks:
- **Champion win rates** on the current patch (global and per-team)
- **Comfort picks:** Does this team have a significantly higher win rate on certain champions? Are they getting those champions?
- **Composition archetypes:** Early game / scaling / teamfight / split-push / poke. Does one team have a structural advantage in how the game should play out?
- **Draft adaptation:** Did a team get counter-picked? Are they on a known composition or improvising?
- **Blue/Red side performance:** Some teams have dramatically different win rates based on side, often driven by draft preferences.

The output is a draft edge score: a number (positive or negative) reflecting whether draft favors Team A or Team B for this specific game.

### Module 3: Rolling Performance (Form & Momentum)
**Status: Medium Priority**

ELO reflects all-time accumulated performance. But a team's current form — their last 5-10 games — often matters more than their season-long track record.

This module computes:
- **Rolling win rate** (last 5, 10, 15 games)
- **Rolling dominance score** (are they winning harder or softer recently?)
- **Performance trend** (improving, stable, declining)
- **Win streak / loss streak effects**

The output is a momentum modifier that nudges the prediction toward or away from the team's base ELO.

### Module 4: Patch Dominance
**Status: Medium Priority**

League of Legends patches change the game every two weeks. A team that was dominant on patch 14.1 might struggle on 14.3 if their champion pool or playstyle was nerfed.

This module tracks:
- **Team performance segmented by patch**
- **Champion pool overlap with patch meta** (are this team's best champions strong right now?)
- **Historical patch transition performance** (does this team adapt quickly or slowly to meta shifts?)

### Module 5: Lane Dominance Profiles
**Status: Medium Priority**

Every professional team has lane matchup tendencies. Some teams play through top lane. Others funnel resources bot. This module creates per-lane performance profiles:
- **Average gold differential at 15 by lane**
- **Average CS differential at 15 by lane**
- **Lane kill participation**
- **Jungle proximity by lane** (where does the jungler spend time?)

When two teams meet, we can compare their lane profiles to estimate where pressure advantages will emerge.

### Module 6: Series Dynamics
**Status: Lower Priority (but important for playoffs)**

Best-of-3 and best-of-5 series play differently than best-of-1. Some teams are notorious for losing game 1 but adapting. Others crumble after a loss.

This module tracks:
- **Win rate after winning game 1 vs. losing game 1**
- **Adaptation rate** (do they tend to improve or decline across a series?)
- **Reverse sweep history**
- **Draft adaptation in series** (do they adjust bans and picks effectively between games?)

### Module 7: Gold/XP Advantage Maps
**Status: Lower Priority (data-dependent)**

For leagues that provide timeline data, this module builds predictive curves:
- **Gold lead at time T → win probability** (how predictive is a 2k gold lead at 15 minutes vs. 25 minutes?)
- **XP differential curves**
- **Objective control timelines** (first dragon timing, first herald, baron attempts)

These maps help contextualize the dominance scores from Module 1 and provide richer game-state features.

### Module 8: XGBoost Feature Optimizer
**Status: Build After Modules 0-3 Are Working**
**Priority: This is the meta-module**

Once we have multiple feature modules producing outputs, we need to know which ones actually matter and how to weight them. This is where XGBoost comes in.

Using historical match data with known outcomes, we train an XGBoost model where:
- **Features** = ELO difference + outputs from all active modules
- **Target** = match result (win/loss)

XGBoost gives us:
- **Feature importance rankings** — which modules are actually predictive?
- **Optimal feature weights** — how much should each module influence the final prediction?
- **Interaction effects** — does draft edge matter more when ELO is close?
- **Model performance metrics** — accuracy, log loss, calibration curves

This module doesn't replace the ELO system — it sits on top and tells us how to combine everything optimally.

---

## Data Sources & Requirements

### Primary Data Needs

| Data Type | What We Need | Possible Sources |
|---|---|---|
| Match results | Win/loss, teams, date, patch, league | Oracle's Elixir, Leaguepedia, Riot API |
| Game stats | Kills, deaths, assists, gold, objectives, duration | Oracle's Elixir, Riot API |
| Draft data | Champion picks and bans per game, pick order | Oracle's Elixir, Leaguepedia |
| Timeline data | Gold/XP at intervals, event timestamps | Riot API (limited availability) |
| Patch notes | Champion changes, item changes, meta shifts | Riot patch notes, wiki |
| Roster data | Player-team mappings with dates | Leaguepedia, manual tracking |

### Oracle's Elixir

Tim Sevenhuysen's Oracle's Elixir (oracleselixir.com) is likely our primary data source. It provides downloadable CSVs of professional match data across all major and minor leagues. The data includes per-game and per-player statistics, draft information, and more.

Key considerations:
- Data format may change between seasons
- Some columns may have missing values for certain leagues
- Historical data availability varies by region
- Always check for data freshness — there can be delays

### Data Pipeline Philosophy

We build the data pipeline to be resilient:
1. **Raw data** goes into `/data/raw/` — never modified
2. **Cleaned data** goes into `/data/processed/` — standardized column names, handled missing values, consistent team naming
3. **Feature data** goes into `/data/features/` — each module's computed features
4. **Model data** goes into `/data/models/` — trained model artifacts, predictions, performance logs

---

## Getting Started

### Prerequisites

- **Python 3.10+** — our primary language
- **Git** — version control (you're on GitHub, so you're set)
- **pip** — Python package manager (comes with Python)
- **A code editor** — VS Code recommended

### Installation

```bash
# Clone the repository
git clone https://github.com/WillGuidry/LeaguePred.git
cd LeaguePred

# Create a virtual environment (keeps dependencies isolated)
python3 -m venv venv
source venv/bin/activate   # On Mac/Linux

# Install dependencies (once we have them)
pip install -r requirements.txt
```

### Initial Setup Checklist

- [ ] Clone the repo
- [ ] Create virtual environment
- [ ] Install dependencies
- [ ] Download initial dataset (Oracle's Elixir match data)
- [ ] Place raw data in `/data/raw/`
- [ ] Run data cleaning script
- [ ] Run base ELO engine on historical data
- [ ] Verify ELO ratings look reasonable

---

## Project Structure

```
LeaguePred/
│
├── README.md                  # You're reading it
├── requirements.txt           # Python dependencies
├── .gitignore                 # Files Git should ignore
│
├── config/
│   ├── settings.py            # Global settings (K-factor, default ELO, etc.)
│   └── modules.py             # Which feature modules are active
│
├── data/
│   ├── raw/                   # Untouched source data (CSVs, JSONs)
│   ├── processed/             # Cleaned, standardized data
│   ├── features/              # Computed features from each module
│   └── models/                # Saved model artifacts
│
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py          # Data loading utilities
│   │   ├── cleaner.py         # Data cleaning and standardization
│   │   └── validators.py      # Data integrity checks
│   │
│   ├── elo/
│   │   ├── __init__.py
│   │   ├── engine.py          # Core ELO calculation
│   │   ├── ratings.py         # Team rating storage and history
│   │   └── config.py          # ELO-specific parameters
│   │
│   ├── modules/
│   │   ├── __init__.py
│   │   ├── base_module.py     # Abstract base class for all modules
│   │   ├── margin.py          # Module 1: Margin of Victory / xG
│   │   ├── draft.py           # Module 2: Champion Draft Analysis
│   │   ├── momentum.py        # Module 3: Rolling Performance
│   │   ├── patch.py           # Module 4: Patch Dominance
│   │   ├── lanes.py           # Module 5: Lane Dominance
│   │   ├── series.py          # Module 6: Series Dynamics
│   │   └── game_state.py      # Module 7: Gold/XP Maps
│   │
│   ├── optimizer/
│   │   ├── __init__.py
│   │   ├── xgboost_model.py   # Module 8: XGBoost Feature Optimizer
│   │   ├── feature_select.py  # Feature importance and selection
│   │   └── calibration.py     # Probability calibration
│   │
│   ├── prediction/
│   │   ├── __init__.py
│   │   ├── predictor.py       # Main prediction pipeline
│   │   └── evaluator.py       # Backtesting and accuracy tracking
│   │
│   └── utils/
│       ├── __init__.py
│       ├── constants.py       # Team names, regions, etc.
│       └── helpers.py         # Shared utility functions
│
├── notebooks/                 # Jupyter notebooks for exploration
│   ├── 01_data_exploration.ipynb
│   ├── 02_elo_tuning.ipynb
│   └── 03_feature_analysis.ipynb
│
├── tests/                     # Unit tests
│   ├── test_elo.py
│   ├── test_modules.py
│   └── test_prediction.py
│
└── scripts/
    ├── run_elo.py             # Compute ELO ratings from historical data
    ├── run_predictions.py     # Generate predictions for upcoming matches
    └── run_backtest.py        # Evaluate model against historical results
```

---

## Development Roadmap

### Phase 1: Foundation (Start Here)
**Goal: Get base ELO working on real data**

1. Set up project structure and virtual environment
2. Download and explore Oracle's Elixir data
3. Write data loader and cleaner
4. Implement core ELO engine
5. Run ELO on 2-3 splits of historical data
6. Validate: Do the ratings make sense? Are top teams rated highest?
7. Baseline accuracy: What % of games does raw ELO predict correctly?

**Expected baseline: ~60-65% accuracy with raw ELO**

### Phase 2: First Edge — Margin of Victory
**Goal: Make ELO smarter about HOW teams win/lose**

1. Define dominance score formula
2. Implement margin-adjusted ELO updates
3. Compare accuracy: Does margin-adjusted ELO beat raw ELO?
4. Tune the dominance score weights

### Phase 3: Draft Intelligence
**Goal: Account for the single biggest variable ELO ignores**

1. Parse champion pick/ban data
2. Build champion win rate tables (global + per-team + per-patch)
3. Implement draft edge scoring
4. Integrate draft edge into predictions
5. Validate: Does adding draft improve accuracy?

### Phase 4: Form & Context
**Goal: Layer in momentum, patch awareness, and lane profiles**

1. Implement rolling performance module
2. Implement patch dominance module
3. Implement lane dominance profiles
4. Test each module individually and in combination

### Phase 5: Optimization
**Goal: Let the data tell us what matters**

1. Collect all module outputs as features
2. Train XGBoost model on historical data
3. Analyze feature importance
4. Build optimized prediction pipeline
5. Implement backtesting framework
6. Measure: accuracy, log loss, calibration, ROI

### Phase 6: Series & Game State (Stretch)
**Goal: Handle playoff contexts and live game analysis**

1. Implement series dynamics module
2. Build gold/XP advantage maps (data permitting)
3. Explore live prediction updates during games

---

## Analytical Framework

### How We Think About Prediction

A match prediction is not a single number. It's a probability distribution shaped by layers of evidence:

```
Base ELO Difference          →  "Team A is generally better"
  + Margin History           →  "...and they win convincingly"
  + Draft Edge               →  "...and they got a favorable draft"
  + Form Modifier            →  "...and they're on a hot streak"
  + Patch Context            →  "...on a patch that suits their style"
  + Lane Matchups            →  "...with lane advantages in 2 of 3 solo lanes"
  = Final Win Probability
```

Each layer should be independently justifiable. If someone asks "why do you have Team A at 68%?", you should be able to point to the specific factors and their contributions.

### Key Metrics We Track

- **Accuracy:** % of games predicted correctly (above 50% = better than coin flip)
- **Log Loss:** Measures how well-calibrated our probabilities are (lower = better). A confident wrong prediction is punished more than a cautious wrong prediction.
- **Calibration:** When we say 70%, does that team win ~70% of the time?
- **Feature Attribution:** For each prediction, which modules contributed most?
- **Brier Score:** Another calibration metric — average squared difference between predicted probability and actual outcome.

### What "Good" Looks Like

For professional LoL prediction:
- **60-62% accuracy** with raw ELO = decent baseline
- **65-68% accuracy** with optimized features = very competitive
- **70%+ accuracy** = exceptional (and worth double-checking for overfitting)

For context, most public prediction models and betting markets sit in the 62-67% range for professional LoL. If we can consistently hit 67%+ with good calibration, we're doing something right.

---

## Technical Notes

### Python Libraries We'll Use

```
# Core
pandas           # Data manipulation
numpy            # Numerical computing

# Machine Learning
scikit-learn     # Model evaluation, preprocessing
xgboost          # Gradient boosting for feature optimization

# Visualization
matplotlib       # Plotting
seaborn          # Statistical visualization

# Utilities
requests         # API calls (if using Riot API)
tqdm             # Progress bars for long computations
```

### ELO Math Refresher

The expected score for Team A against Team B:

```
E_A = 1 / (1 + 10^((R_B - R_A) / 400))
```

After a match, Team A's new rating:

```
R_A_new = R_A + K * (S_A - E_A)
```

Where:
- `R_A`, `R_B` = current ratings
- `E_A` = expected score (probability of winning)
- `S_A` = actual result (1 for win, 0 for loss)
- `K` = K-factor (controls how much ratings change per game)

With our margin-of-victory modifier, this becomes:

```
R_A_new = R_A + K * M * (S_A - E_A)
```

Where `M` is the dominance multiplier (> 1 for dominant results, < 1 for narrow results).

### Git Workflow for This Project

We keep it simple:

```bash
# Before starting work
git pull                        # Get latest changes

# After making changes
git add .                       # Stage changes
git commit -m "Clear message"   # Commit with description
git push                        # Push to GitHub
```

Commit message conventions:
- `"Add base ELO engine"` — new feature
- `"Fix K-factor calculation"` — bug fix
- `"Update README with Phase 2 notes"` — documentation
- `"Refactor data loader for Oracle's Elixir v2"` — code improvement

---

## Glossary

| Term | Definition |
|---|---|
| **ELO** | Rating system originally designed for chess. Measures relative skill based on game results. |
| **K-factor** | How sensitive ELO is to a single game result. Higher K = more volatile ratings. |
| **xG (Expected Goals)** | Concept from football analytics. Measures the quality of chances created, not just the scoreline. Our "dominance score" is the LoL equivalent. |
| **Feature** | Any measurable input to our prediction model (ELO difference, draft edge, momentum score, etc.) |
| **Module** | A self-contained component that computes one type of feature. Can be enabled or disabled independently. |
| **Calibration** | How well our predicted probabilities match reality. If we say 70%, it should happen ~70% of the time. |
| **Log Loss** | A scoring metric that punishes confident wrong predictions heavily. Rewards well-calibrated probabilities. |
| **Backtest** | Running the model on historical data to see how it would have performed. Our primary validation method. |
| **Oracle's Elixir** | The primary public data source for professional LoL statistics, maintained by Tim Sevenhuysen. |
| **Meta** | The current state of the game — which champions, strategies, and playstyles are strongest on the current patch. |
| **Draft** | The champion selection phase before a game. Each team bans 5 champions and picks 5 champions in an alternating format. |
| **Composition** | The set of 5 champions a team selects and the strategic identity they form together (e.g., teamfight comp, split-push comp). |

---

## Contributing

This is a personal/team project. If you're reading this and want to contribute ideas or data sources, open an issue or reach out.

---

*"The draft is the game before the game." — Every analyst, probably*

*Built with curiosity, Python, and an unreasonable number of VOD reviews.*
