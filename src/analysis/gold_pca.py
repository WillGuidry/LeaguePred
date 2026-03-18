"""
Gold@10 Pre-Game Feature PCA Analysis
--------------------------------------
Identifies which pre-game factors explain the most variance in gold@10 outcomes.

Approach:
    1. Load Oracle's Elixir CSV at PLAYER level (not collapsed)
    2. Process games chronologically, building rolling features from prior games only
    3. Feature families:
        - Team Elo + Elo differential
        - Rolling team momentum (win% over last 5/10/15 games)
        - Lead State features (avg gold diffs, first objective rates)
        - Player Elo (individual rating system per player)
        - Player lane metrics (CS/min, gold/min, KDA, lane winrate per position)
        - Player dominance (lane diff metrics — how much a player outperforms opponent)
        - Champion metrics (CURRENT GAME draft — win rates, player comfort, matchups)
        - Side (blue/red) and meta context
    4. Target: gold differential at 10 minutes (golddiffat10)
    5. Run PCA on standardized pre-game features, report variance explained

Leakage prevention:
    - ALL team/player rolling features use only data from games BEFORE the current game
    - Champion features use the current draft (known pre-game) but rolling stats from prior games
    - gold@10 is the TARGET, never a feature
    - No post-match stats leak into features

Usage:
    python -m src.analysis.gold_pca <path_to_oracle_csv> [--leagues LCK LPL LEC] [--min-date 2024-01-01]
"""

import math
import argparse
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from typing import Optional

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from src.config import REGIONAL_ELO_PRIORS, REGIONAL_ELO_DEFAULT


# =============================================================================
# PLAYER ELO ENGINE
# =============================================================================

class PlayerEloEngine:
    """
    Individual player rating system.
    Tracks each player's Elo, updated after each game based on team outcome
    weighted by the player's individual contribution.
    """

    def __init__(self, default_elo: float = 1500.0, k_factor: float = 24.0):
        self.default_elo = default_elo
        self.k_factor = k_factor
        self.ratings = {}  # {player_name: elo}
        self.game_counts = {}  # {player_name: n_games}

    def get_rating(self, player: str) -> float:
        return self.ratings.get(player, self.default_elo)

    def get_games(self, player: str) -> int:
        return self.game_counts.get(player, 0)

    def update(self, player: str, won: bool, team_expected: float,
               contribution_weight: float = 1.0):
        """
        Update player Elo after a game.

        Args:
            player: Player name.
            won: Whether the player's team won.
            team_expected: Team's expected win probability (from team Elo).
            contribution_weight: Multiplier based on individual performance (0.5-1.5).
        """
        if player not in self.ratings:
            self.ratings[player] = self.default_elo
            self.game_counts[player] = 0

        actual = 1.0 if won else 0.0
        # Player-level expected uses team expected as baseline
        # but the update is scaled by individual contribution
        delta = self.k_factor * contribution_weight * (actual - team_expected)
        self.ratings[player] += delta
        self.game_counts[player] += 1


# =============================================================================
# ROLLING PLAYER STATS TRACKER
# =============================================================================

class PlayerStatsTracker:
    """
    Tracks per-player rolling statistics by position.

    Stores recent games per player and computes lane-specific metrics:
    - Lane win rate (did team win when this player played this lane)
    - CS per minute
    - Gold per minute
    - KDA
    - Gold diff at 10 (lane dominance)
    - XP diff at 10 (lane pressure)
    """

    def __init__(self, window: int = 20):
        self.window = window
        # {player_name: deque of game stat dicts}
        self.history = defaultdict(lambda: deque(maxlen=window))
        # {player_name: {position: deque of game stat dicts}}
        self.position_history = defaultdict(lambda: defaultdict(lambda: deque(maxlen=window)))

    def record_game(self, player: str, position: str, stats: dict):
        """Record a player's game stats (called in UPDATE phase)."""
        self.history[player].append(stats)
        self.position_history[player][position].append(stats)

    def get_features(self, player: str, position: str) -> dict:
        """
        Get rolling features for a player at a specific position.
        Called in PREDICT phase (before game result known).
        """
        all_games = list(self.history.get(player, []))
        pos_games = list(self.position_history.get(player, {}).get(position, []))

        n_all = len(all_games)
        n_pos = len(pos_games)

        if n_all == 0:
            return self._default_features()

        # Overall stats
        win_rate = np.mean([g["result"] for g in all_games])
        avg_kda = np.mean([g.get("kda", 2.0) for g in all_games])

        # Position-specific stats
        if n_pos > 0:
            lane_wr = np.mean([g["result"] for g in pos_games])
            lane_cspm = np.mean([g.get("cspm", 0) for g in pos_games])
            lane_gpm = np.mean([g.get("gpm", 0) for g in pos_games])
            lane_kda = np.mean([g.get("kda", 2.0) for g in pos_games])
            lane_gd10 = np.mean([g.get("golddiffat10", 0) for g in pos_games])
            lane_xpd10 = np.mean([g.get("xpdiffat10", 0) for g in pos_games])
            lane_csd10 = np.mean([g.get("csdiffat10", 0) for g in pos_games])
        else:
            lane_wr = win_rate
            lane_cspm = np.mean([g.get("cspm", 0) for g in all_games])
            lane_gpm = np.mean([g.get("gpm", 0) for g in all_games])
            lane_kda = avg_kda
            lane_gd10 = 0.0
            lane_xpd10 = 0.0
            lane_csd10 = 0.0

        # Recent form (last 5 games)
        recent = all_games[-5:] if n_all >= 5 else all_games
        recent_wr = np.mean([g["result"] for g in recent])

        return {
            "player_win_rate": win_rate,
            "player_kda": avg_kda,
            "player_n_games": n_all,
            "lane_win_rate": lane_wr,
            "lane_cspm": lane_cspm,
            "lane_gpm": lane_gpm,
            "lane_kda": lane_kda,
            "lane_gd10": lane_gd10,
            "lane_xpd10": lane_xpd10,
            "lane_csd10": lane_csd10,
            "lane_n_games": n_pos,
            "player_recent_wr": recent_wr,
        }

    def _default_features(self) -> dict:
        return {
            "player_win_rate": 0.5,
            "player_kda": 2.0,
            "player_n_games": 0,
            "lane_win_rate": 0.5,
            "lane_cspm": 7.0,
            "lane_gpm": 350.0,
            "lane_kda": 2.0,
            "lane_gd10": 0.0,
            "lane_xpd10": 0.0,
            "lane_csd10": 0.0,
            "lane_n_games": 0,
            "player_recent_wr": 0.5,
        }


# =============================================================================
# CHAMPION / DRAFT TRACKER
# =============================================================================

class ChampionTracker:
    """
    Tracks rolling champion statistics for draft-based features.

    Uses the CURRENT GAME's draft (known pre-game) combined with
    historical performance data from PRIOR games.

    Tracks:
    - Global champion win rate (across all games)
    - Player-champion comfort (how often + how well a player plays this champ)
    - Champion gold@10 tendency (does this champ tend to be up/down gold early)
    - Position-specific champion stats
    """

    def __init__(self, min_games: int = 3):
        self.min_games = min_games
        # {champion: [game_records]}
        self.champ_history = defaultdict(list)
        # {(player, champion): [game_records]}
        self.player_champ_history = defaultdict(list)
        # {(champion, position): [game_records]}
        self.champ_position_history = defaultdict(list)
        # Global stats for shrinkage
        self._total_games = 0
        self._total_wins = 0
        self._total_gd10 = 0.0

    def record_game(self, champion: str, player: str, position: str, stats: dict):
        """Record a champion performance (UPDATE phase)."""
        self.champ_history[champion].append(stats)
        self.player_champ_history[(player, champion)].append(stats)
        self.champ_position_history[(champion, position)].append(stats)
        self._total_games += 1
        self._total_wins += stats.get("result", 0)
        self._total_gd10 += stats.get("golddiffat10", 0)

    def get_champion_features(self, champion: str, player: str,
                               position: str) -> dict:
        """
        Get pre-game features for a champion pick (PREDICT phase).
        Uses rolling data from prior games only.
        """
        if not champion or champion == "unknown":
            return self._default_features()

        champ_games = self.champ_history.get(champion, [])
        pc_games = self.player_champ_history.get((player, champion), [])
        cp_games = self.champ_position_history.get((champion, position), [])

        n_champ = len(champ_games)
        n_pc = len(pc_games)
        n_cp = len(cp_games)

        global_wr = self._total_wins / max(self._total_games, 1)

        # Champion global win rate (with shrinkage)
        if n_champ >= self.min_games:
            champ_wr = np.mean([g["result"] for g in champ_games])
            # Shrink toward global
            shrink = n_champ / (n_champ + 10)
            champ_wr = shrink * champ_wr + (1 - shrink) * global_wr
        else:
            champ_wr = global_wr

        # Champion average gold diff at 10
        if n_champ >= self.min_games:
            champ_gd10 = np.mean([g.get("golddiffat10", 0) for g in champ_games])
        else:
            champ_gd10 = 0.0

        # Player-champion comfort (games played + win rate)
        if n_pc >= 2:
            pc_wr = np.mean([g["result"] for g in pc_games])
            pc_kda = np.mean([g.get("kda", 2.0) for g in pc_games])
            pc_gd10 = np.mean([g.get("golddiffat10", 0) for g in pc_games])
        else:
            pc_wr = champ_wr
            pc_kda = 2.0
            pc_gd10 = champ_gd10

        # Champion in this position
        if n_cp >= self.min_games:
            cp_wr = np.mean([g["result"] for g in cp_games])
            cp_gd10 = np.mean([g.get("golddiffat10", 0) for g in cp_games])
        else:
            cp_wr = champ_wr
            cp_gd10 = champ_gd10

        return {
            "champ_win_rate": champ_wr,
            "champ_gd10": champ_gd10,
            "champ_n_games": n_champ,
            "player_champ_wr": pc_wr,
            "player_champ_kda": pc_kda,
            "player_champ_gd10": pc_gd10,
            "player_champ_games": n_pc,
            "champ_position_wr": cp_wr,
            "champ_position_gd10": cp_gd10,
        }

    def _default_features(self) -> dict:
        global_wr = self._total_wins / max(self._total_games, 1) if self._total_games > 0 else 0.5
        return {
            "champ_win_rate": global_wr,
            "champ_gd10": 0.0,
            "champ_n_games": 0,
            "player_champ_wr": global_wr,
            "player_champ_kda": 2.0,
            "player_champ_gd10": 0.0,
            "player_champ_games": 0,
            "champ_position_wr": global_wr,
            "champ_position_gd10": 0.0,
        }


# =============================================================================
# ROLLING TEAM STATS TRACKER (EXTENDED)
# =============================================================================

class TeamFeatureTracker:
    """
    Extended team tracker that computes momentum and rolling performance.
    Builds on the existing TeamStateTracker concept but adds momentum windows.
    """

    def __init__(self, window: int = 20):
        self.window = window
        self.history = defaultdict(lambda: deque(maxlen=window))

    def record_game(self, team: str, stats: dict):
        self.history[team].append(stats)

    def get_features(self, team: str) -> dict:
        """Get rolling team features from prior games only."""
        games = list(self.history.get(team, []))
        n = len(games)

        if n == 0:
            return self._default_features()

        # Basic rolling stats
        win_rate = np.mean([g["result"] for g in games])
        avg_gd10 = np.mean([g.get("golddiffat10", 0) for g in games])
        avg_gd15 = np.mean([g.get("golddiffat15", 0) for g in games])
        avg_gd20 = np.mean([g.get("golddiffat20", 0) for g in games])
        avg_gamelength = np.mean([g.get("gamelength", 1800) for g in games])
        avg_kills = np.mean([g.get("kills", 0) for g in games])
        avg_dragons = np.mean([g.get("dragons", 0) for g in games])
        avg_towers = np.mean([g.get("towers", 0) for g in games])
        avg_barons = np.mean([g.get("barons", 0) for g in games])

        # First objective rates
        fb_rate = np.mean([g.get("firstblood", 0) for g in games])
        fd_rate = np.mean([g.get("firstdragon", 0) for g in games])
        fh_rate = np.mean([g.get("firstherald", 0) for g in games])
        ft_rate = np.mean([g.get("firsttower", 0) for g in games])

        # Momentum: win rate over different windows
        def window_wr(w):
            recent = games[-w:] if n >= w else games
            return np.mean([g["result"] for g in recent])

        momentum_5 = window_wr(5)
        momentum_10 = window_wr(10)
        momentum_15 = window_wr(15)

        # Streak: current win/loss streak
        streak = 0
        if n > 0:
            last_result = games[-1]["result"]
            for g in reversed(games):
                if g["result"] == last_result:
                    streak += 1
                else:
                    break
            if last_result == 0:
                streak = -streak  # Negative for loss streaks

        # Gold efficiency: gold diff per minute trend
        gd10_values = [g.get("golddiffat10", 0) for g in games]
        gd10_trend = 0.0
        if n >= 3:
            # Simple linear trend: positive = improving, negative = declining
            x = np.arange(len(gd10_values))
            if np.std(gd10_values) > 0:
                gd10_trend = np.corrcoef(x, gd10_values)[0, 1]
                if np.isnan(gd10_trend):
                    gd10_trend = 0.0

        # Variance in gold@10 (consistency measure)
        gd10_std = np.std(gd10_values) if n >= 2 else 1000.0

        # Average XP diff at 10
        avg_xpd10 = np.mean([g.get("xpdiffat10", 0) for g in games])

        # CS diff at 10
        avg_csd10 = np.mean([g.get("csdiffat10", 0) for g in games])

        # Turret plate differential
        avg_plates = np.mean([g.get("turretplates", 0) for g in games])
        avg_opp_plates = np.mean([g.get("opp_turretplates", 0) for g in games])

        return {
            "team_win_rate": win_rate,
            "team_avg_gd10": avg_gd10,
            "team_avg_gd15": avg_gd15,
            "team_avg_gd20": avg_gd20,
            "team_avg_gamelength": avg_gamelength,
            "team_avg_kills": avg_kills,
            "team_avg_dragons": avg_dragons,
            "team_avg_towers": avg_towers,
            "team_avg_barons": avg_barons,
            "team_fb_rate": fb_rate,
            "team_fd_rate": fd_rate,
            "team_fh_rate": fh_rate,
            "team_ft_rate": ft_rate,
            "team_momentum_5": momentum_5,
            "team_momentum_10": momentum_10,
            "team_momentum_15": momentum_15,
            "team_streak": streak,
            "team_gd10_trend": gd10_trend,
            "team_gd10_std": gd10_std,
            "team_avg_xpd10": avg_xpd10,
            "team_avg_csd10": avg_csd10,
            "team_avg_plates": avg_plates,
            "team_avg_opp_plates": avg_opp_plates,
            "team_n_games": n,
        }

    def _default_features(self) -> dict:
        return {
            "team_win_rate": 0.5,
            "team_avg_gd10": 0.0,
            "team_avg_gd15": 0.0,
            "team_avg_gd20": 0.0,
            "team_avg_gamelength": 1800.0,
            "team_avg_kills": 10.0,
            "team_avg_dragons": 2.0,
            "team_avg_towers": 5.0,
            "team_avg_barons": 0.5,
            "team_fb_rate": 0.5,
            "team_fd_rate": 0.5,
            "team_fh_rate": 0.5,
            "team_ft_rate": 0.5,
            "team_momentum_5": 0.5,
            "team_momentum_10": 0.5,
            "team_momentum_15": 0.5,
            "team_streak": 0,
            "team_gd10_trend": 0.0,
            "team_gd10_std": 1000.0,
            "team_avg_xpd10": 0.0,
            "team_avg_csd10": 0.0,
            "team_avg_plates": 2.0,
            "team_avg_opp_plates": 2.0,
            "team_n_games": 0,
        }


# =============================================================================
# TEAM ELO (LIGHTWEIGHT, JUST FOR FEATURE EXTRACTION)
# =============================================================================

class SimpleTeamElo:
    """Lightweight team Elo for feature extraction only."""

    def __init__(self, k_factor: float = 32.0, scale: float = 400.0):
        self.k_factor = k_factor
        self.scale = scale
        self.ratings = {}

    def get_rating(self, team: str, league: str = None) -> float:
        if team not in self.ratings:
            default = REGIONAL_ELO_PRIORS.get(league, REGIONAL_ELO_DEFAULT) if league else 1500.0
            self.ratings[team] = default
        return self.ratings[team]

    def expected(self, team_a: str, team_b: str) -> float:
        ra = self.ratings.get(team_a, 1500)
        rb = self.ratings.get(team_b, 1500)
        return 1.0 / (1.0 + math.pow(10, (rb - ra) / self.scale))

    def update(self, team_a: str, team_b: str, winner: str):
        ra = self.get_rating(team_a)
        rb = self.get_rating(team_b)
        ea = 1.0 / (1.0 + math.pow(10, (rb - ra) / self.scale))
        sa = 1.0 if winner == team_a else 0.0
        self.ratings[team_a] = ra + self.k_factor * (sa - ea)
        self.ratings[team_b] = rb + self.k_factor * ((1 - sa) - (1 - ea))


# =============================================================================
# DATA LOADING — PLAYER-LEVEL
# =============================================================================

def load_player_level(filepath: str, leagues: list = None,
                      min_date: str = None) -> pd.DataFrame:
    """
    Load Oracle's Elixir CSV and return player-level rows.
    Does NOT collapse to team level — we need individual player stats.
    """
    print(f"Loading player-level data from {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)
    df.columns = df.columns.str.strip().str.lower()

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if min_date:
            df = df[df["date"] >= pd.to_datetime(min_date)]

    if leagues and "league" in df.columns:
        df = df[df["league"].isin(leagues)]

    print(f"  {len(df)} rows, {df['gameid'].nunique()} games")
    return df


def extract_game_player_data(df: pd.DataFrame) -> tuple:
    """
    From the raw Oracle's Elixir dataframe, extract:
    1. games: list of game dicts with team-level info + gold@10 target
    2. player_games: list of player-game dicts with individual stats

    Returns (games, player_games) sorted chronologically.
    """
    team_col = "teamname" if "teamname" in df.columns else "team"
    player_col = "playername" if "playername" in df.columns else "playerid"

    # Separate team rows and player rows
    player_rows = df[df["position"].isin(["top", "jng", "mid", "bot", "sup"])].copy()
    team_rows = df[df["position"] == "team"].copy()

    if team_rows.empty:
        raise ValueError("No team-level rows found (position == 'team')")

    games = []
    player_games = []

    for game_id, group in team_rows.groupby("gameid"):
        if len(group) != 2:
            continue

        row_a = group.iloc[0]
        row_b = group.iloc[1]

        team_a = row_a.get(team_col, "Unknown")
        team_b = row_b.get(team_col, "Unknown")

        # Determine winner
        winner = None
        if "result" in row_a.index:
            winner = team_a if row_a["result"] == 1 else team_b

        if winner is None:
            continue

        def _safe(row, col, default=0.0):
            val = row.get(col, None)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            return float(val)

        # Build game record
        game = {
            "gameid": game_id,
            "date": row_a.get("date", None),
            "league": row_a.get("league", None),
            "patch": row_a.get("patch", None),
            "team_a": team_a,
            "team_b": team_b,
            "winner": winner,
            "side_a": str(row_a.get("side", "blue")).lower(),
            "side_b": str(row_b.get("side", "red")).lower(),
            "playoffs": int(_safe(row_a, "playoffs", 0)),
            "gamelength": _safe(row_a, "gamelength", 1800),
            # TARGET: gold diff at 10 (from team_a perspective)
            "golddiffat10": _safe(row_a, "golddiffat10"),
            # Additional targets for later analysis
            "golddiffat15": _safe(row_a, "golddiffat15"),
            "golddiffat20": _safe(row_a, "golddiffat20"),
            # Post-match stats needed for tracker updates
            "a_golddiffat10": _safe(row_a, "golddiffat10"),
            "a_golddiffat15": _safe(row_a, "golddiffat15"),
            "a_golddiffat20": _safe(row_a, "golddiffat20"),
            "a_xpdiffat10": _safe(row_a, "xpdiffat10"),
            "a_kills": _safe(row_a, "kills"),
            "a_deaths": _safe(row_a, "deaths"),
            "a_dragons": _safe(row_a, "dragons"),
            "a_barons": _safe(row_a, "barons"),
            "a_towers": _safe(row_a, "towers"),
            "a_firstblood": _safe(row_a, "firstblood"),
            "a_firstdragon": _safe(row_a, "firstdragon"),
            "a_firstherald": _safe(row_a, "firstherald"),
            "a_firsttower": _safe(row_a, "firsttower"),
            "a_totalgold": _safe(row_a, "totalgold"),
            "a_turretplates": _safe(row_a, "turretplates"),
            "a_opp_turretplates": _safe(row_a, "opp_turretplates"),
            "b_golddiffat10": _safe(row_b, "golddiffat10"),
            "b_golddiffat15": _safe(row_b, "golddiffat15"),
            "b_golddiffat20": _safe(row_b, "golddiffat20"),
            "b_xpdiffat10": _safe(row_b, "xpdiffat10"),
            "b_kills": _safe(row_b, "kills"),
            "b_deaths": _safe(row_b, "deaths"),
            "b_dragons": _safe(row_b, "dragons"),
            "b_barons": _safe(row_b, "barons"),
            "b_towers": _safe(row_b, "towers"),
            "b_firstblood": _safe(row_b, "firstblood"),
            "b_firstdragon": _safe(row_b, "firstdragon"),
            "b_firstherald": _safe(row_b, "firstherald"),
            "b_firsttower": _safe(row_b, "firsttower"),
            "b_totalgold": _safe(row_b, "totalgold"),
            "b_turretplates": _safe(row_b, "turretplates"),
            "b_opp_turretplates": _safe(row_b, "opp_turretplates"),
        }
        games.append(game)

        # Extract player-level data for this game
        game_players = player_rows[player_rows["gameid"] == game_id]
        for _, prow in game_players.iterrows():
            player_name = prow.get(player_col, None)
            if pd.isna(player_name):
                continue

            position = prow.get("position", "unknown")
            team = prow.get(team_col, "Unknown")
            gamelength_sec = _safe(row_a, "gamelength", 1800)
            gamelength_min = max(gamelength_sec / 60.0, 1.0)

            kills = _safe(prow, "kills", 0)
            deaths = _safe(prow, "deaths", 0)
            assists = _safe(prow, "assists", 0)
            kda = (kills + assists) / max(deaths, 1.0)

            total_cs = _safe(prow, "minionkills", 0) + _safe(prow, "monsterkills", 0)
            total_gold = _safe(prow, "totalgold", 0)

            # Champion pick (known pre-game from draft)
            champion = prow.get("champion", "unknown")
            if pd.isna(champion):
                champion = "unknown"

            player_game = {
                "gameid": game_id,
                "player": player_name,
                "team": team,
                "position": position,
                "champion": str(champion),
                "result": 1 if winner == team else 0,
                "kills": kills,
                "deaths": deaths,
                "assists": assists,
                "kda": kda,
                "cspm": total_cs / gamelength_min,
                "gpm": total_gold / gamelength_min,
                "golddiffat10": _safe(prow, "golddiffat10"),
                "xpdiffat10": _safe(prow, "xpdiffat10"),
                "csdiffat10": _safe(prow, "csat10", 0) - _safe(prow, "opp_csat10", 0)
                    if "opp_csat10" in prow.index else 0.0,
            }
            player_games.append(player_game)

    # Sort by date
    games.sort(key=lambda g: g["date"] if pd.notna(g.get("date")) else pd.Timestamp.min)

    print(f"  Extracted {len(games)} games, {len(player_games)} player-game records")
    return games, player_games


# =============================================================================
# FEATURE MATRIX BUILDER
# =============================================================================

POSITIONS = ["top", "jng", "mid", "bot", "sup"]

# Feature names for player-level features (per position)
PLAYER_FEATURE_KEYS = [
    "player_elo", "player_win_rate", "player_kda", "player_n_games",
    "lane_win_rate", "lane_cspm", "lane_gpm", "lane_kda",
    "lane_gd10", "lane_xpd10", "lane_csd10", "lane_n_games",
    "player_recent_wr",
]

# Champion/draft features (per position, from CURRENT game's draft)
CHAMPION_FEATURE_KEYS = [
    "champ_win_rate", "champ_gd10", "champ_n_games",
    "player_champ_wr", "player_champ_kda", "player_champ_gd10",
    "player_champ_games", "champ_position_wr", "champ_position_gd10",
]

# Feature names for team-level features
TEAM_FEATURE_KEYS = [
    "team_elo", "team_win_rate",
    "team_avg_gd10", "team_avg_gd15", "team_avg_gd20",
    "team_avg_gamelength", "team_avg_kills",
    "team_avg_dragons", "team_avg_towers", "team_avg_barons",
    "team_fb_rate", "team_fd_rate", "team_fh_rate", "team_ft_rate",
    "team_momentum_5", "team_momentum_10", "team_momentum_15",
    "team_streak", "team_gd10_trend", "team_gd10_std",
    "team_avg_xpd10", "team_avg_csd10",
    "team_avg_plates", "team_avg_opp_plates",
    "team_n_games",
]


def build_feature_names() -> list:
    """Build the full list of feature names in the feature matrix."""
    names = []

    # Differential team features (team_a - team_b)
    for key in TEAM_FEATURE_KEYS:
        names.append(f"diff_{key}")

    # Raw team features for both sides
    for side in ["a", "b"]:
        for key in TEAM_FEATURE_KEYS:
            names.append(f"{side}_{key}")

    # Differential player features by position
    for pos in POSITIONS:
        for key in PLAYER_FEATURE_KEYS:
            names.append(f"diff_{pos}_{key}")

    # Raw player features for both sides by position
    for side in ["a", "b"]:
        for pos in POSITIONS:
            for key in PLAYER_FEATURE_KEYS:
                names.append(f"{side}_{pos}_{key}")

    # Champion/draft differential features by position (CURRENT game draft)
    for pos in POSITIONS:
        for key in CHAMPION_FEATURE_KEYS:
            names.append(f"diff_{pos}_{key}")

    # Raw champion features for both sides by position
    for side in ["a", "b"]:
        for pos in POSITIONS:
            for key in CHAMPION_FEATURE_KEYS:
                names.append(f"{side}_{pos}_{key}")

    # Aggregate draft features
    names.extend([
        "avg_champ_wr_diff",       # avg champion win rate diff across all positions
        "avg_champ_gd10_diff",     # avg champion gd10 tendency diff
        "avg_player_champ_comfort_diff",  # avg player-champion comfort diff
    ])

    # Context features
    names.extend([
        "elo_diff",           # team_a Elo - team_b Elo
        "avg_player_elo_diff",  # avg player Elo diff
        "side_a_is_blue",     # blue side advantage
        "is_playoffs",
    ])

    return names


def build_feature_matrix(
    games: list,
    player_games: list,
    min_team_games: int = 5,
    min_player_games: int = 3,
) -> tuple:
    """
    Process games chronologically and build the pre-game feature matrix.

    For each game, all features are computed from PRIOR games only.
    After prediction features are extracted, trackers are updated with results.

    Returns:
        (feature_matrix: np.ndarray, targets: np.ndarray, feature_names: list,
         game_ids: list, valid_mask: list)
    """
    # Initialize trackers
    team_elo = SimpleTeamElo()
    player_elo = PlayerEloEngine()
    team_tracker = TeamFeatureTracker(window=20)
    player_tracker = PlayerStatsTracker(window=20)
    champ_tracker = ChampionTracker(min_games=3)

    # Index player_games by gameid for fast lookup
    player_game_index = defaultdict(list)
    for pg in player_games:
        player_game_index[pg["gameid"]].append(pg)

    # Track which players are on which team for each game
    # (needed to get player features at prediction time)
    feature_names = build_feature_names()
    n_features = len(feature_names)

    features_list = []
    targets = []
    game_ids = []
    valid_mask = []

    for i, game in enumerate(games):
        team_a = game["team_a"]
        team_b = game["team_b"]
        winner = game["winner"]
        league = game.get("league")
        gd10 = game.get("golddiffat10", None)

        # Initialize team Elo if new
        team_elo.get_rating(team_a, league)
        team_elo.get_rating(team_b, league)

        # =================================================================
        # PREDICT PHASE: Extract pre-game features from prior data
        # =================================================================

        # Team features
        feats_a = team_tracker.get_features(team_a)
        feats_b = team_tracker.get_features(team_b)

        # Team Elo
        elo_a = team_elo.get_rating(team_a, league)
        elo_b = team_elo.get_rating(team_b, league)
        feats_a["team_elo"] = elo_a
        feats_b["team_elo"] = elo_b

        # Player features — need to find players for this game
        game_player_data = player_game_index.get(game["gameid"], [])

        # Group players by team and position (+ champion from current draft)
        players_a = {}  # {position: player_name}
        players_b = {}
        champs_a = {}   # {position: champion_name}
        champs_b = {}
        for pg in game_player_data:
            if pg["team"] == team_a:
                players_a[pg["position"]] = pg["player"]
                champs_a[pg["position"]] = pg.get("champion", "unknown")
            elif pg["team"] == team_b:
                players_b[pg["position"]] = pg["player"]
                champs_b[pg["position"]] = pg.get("champion", "unknown")

        # Build feature vector
        row = np.zeros(n_features)
        idx = 0

        # Differential team features
        for key in TEAM_FEATURE_KEYS:
            row[idx] = feats_a.get(key, 0) - feats_b.get(key, 0)
            idx += 1

        # Raw team features (side a)
        for key in TEAM_FEATURE_KEYS:
            row[idx] = feats_a.get(key, 0)
            idx += 1

        # Raw team features (side b)
        for key in TEAM_FEATURE_KEYS:
            row[idx] = feats_b.get(key, 0)
            idx += 1

        # Player features by position
        player_elos_a = []
        player_elos_b = []

        for pos in POSITIONS:
            player_a = players_a.get(pos)
            player_b = players_b.get(pos)

            if player_a:
                pf_a = player_tracker.get_features(player_a, pos)
                pf_a["player_elo"] = player_elo.get_rating(player_a)
                player_elos_a.append(pf_a["player_elo"])
            else:
                pf_a = player_tracker._default_features()
                pf_a["player_elo"] = 1500.0
                player_elos_a.append(1500.0)

            if player_b:
                pf_b = player_tracker.get_features(player_b, pos)
                pf_b["player_elo"] = player_elo.get_rating(player_b)
                player_elos_b.append(pf_b["player_elo"])
            else:
                pf_b = player_tracker._default_features()
                pf_b["player_elo"] = 1500.0
                player_elos_b.append(1500.0)

            # Differential player features for this position
            for key in PLAYER_FEATURE_KEYS:
                row[idx] = pf_a.get(key, 0) - pf_b.get(key, 0)
                idx += 1

        # Raw player features (side a, all positions)
        for pos in POSITIONS:
            player_a = players_a.get(pos)
            if player_a:
                pf_a = player_tracker.get_features(player_a, pos)
                pf_a["player_elo"] = player_elo.get_rating(player_a)
            else:
                pf_a = player_tracker._default_features()
                pf_a["player_elo"] = 1500.0

            for key in PLAYER_FEATURE_KEYS:
                row[idx] = pf_a.get(key, 0)
                idx += 1

        # Raw player features (side b, all positions)
        for pos in POSITIONS:
            player_b = players_b.get(pos)
            if player_b:
                pf_b = player_tracker.get_features(player_b, pos)
                pf_b["player_elo"] = player_elo.get_rating(player_b)
            else:
                pf_b = player_tracker._default_features()
                pf_b["player_elo"] = 1500.0

            for key in PLAYER_FEATURE_KEYS:
                row[idx] = pf_b.get(key, 0)
                idx += 1

        # Champion/draft features (CURRENT GAME draft — known pre-game)
        champ_wr_diffs = []
        champ_gd10_diffs = []
        champ_comfort_diffs = []

        for pos in POSITIONS:
            player_a = players_a.get(pos)
            player_b = players_b.get(pos)
            champ_a = champs_a.get(pos, "unknown")
            champ_b = champs_b.get(pos, "unknown")

            cf_a = champ_tracker.get_champion_features(champ_a, player_a or "", pos)
            cf_b = champ_tracker.get_champion_features(champ_b, player_b or "", pos)

            # Differential champion features for this position
            for key in CHAMPION_FEATURE_KEYS:
                row[idx] = cf_a.get(key, 0) - cf_b.get(key, 0)
                idx += 1

            champ_wr_diffs.append(cf_a.get("champ_win_rate", 0.5) - cf_b.get("champ_win_rate", 0.5))
            champ_gd10_diffs.append(cf_a.get("champ_gd10", 0) - cf_b.get("champ_gd10", 0))
            champ_comfort_diffs.append(cf_a.get("player_champ_games", 0) - cf_b.get("player_champ_games", 0))

        # Raw champion features (side a, all positions)
        for pos in POSITIONS:
            player_a = players_a.get(pos)
            champ_a = champs_a.get(pos, "unknown")
            cf_a = champ_tracker.get_champion_features(champ_a, player_a or "", pos)
            for key in CHAMPION_FEATURE_KEYS:
                row[idx] = cf_a.get(key, 0)
                idx += 1

        # Raw champion features (side b, all positions)
        for pos in POSITIONS:
            player_b = players_b.get(pos)
            champ_b = champs_b.get(pos, "unknown")
            cf_b = champ_tracker.get_champion_features(champ_b, player_b or "", pos)
            for key in CHAMPION_FEATURE_KEYS:
                row[idx] = cf_b.get(key, 0)
                idx += 1

        # Aggregate draft features
        row[idx] = np.mean(champ_wr_diffs); idx += 1
        row[idx] = np.mean(champ_gd10_diffs); idx += 1
        row[idx] = np.mean(champ_comfort_diffs); idx += 1

        # Context features
        row[idx] = elo_a - elo_b; idx += 1
        row[idx] = np.mean(player_elos_a) - np.mean(player_elos_b); idx += 1
        row[idx] = 1.0 if game.get("side_a") == "blue" else 0.0; idx += 1
        row[idx] = float(game.get("playoffs", 0)); idx += 1

        features_list.append(row)
        targets.append(gd10 if gd10 is not None else np.nan)
        game_ids.append(game["gameid"])

        # Valid = both teams have enough games AND target is not NaN
        has_enough = (feats_a["team_n_games"] >= min_team_games and
                      feats_b["team_n_games"] >= min_team_games)
        has_target = gd10 is not None and not np.isnan(gd10) and gd10 != 0.0
        valid_mask.append(has_enough and has_target)

        # =================================================================
        # UPDATE PHASE: Record results for future games
        # =================================================================

        # Update team Elo
        team_elo.update(team_a, team_b, winner)

        # Update team tracker
        def _safe_game(row, col, default=0.0):
            val = row.get(col, default)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                return default
            return float(val)

        for team, prefix in [(team_a, "a"), (team_b, "b")]:
            sign = 1.0 if prefix == "a" else -1.0
            team_stats = {
                "result": 1 if winner == team else 0,
                "golddiffat10": sign * _safe_game(game, "a_golddiffat10"),
                "golddiffat15": sign * _safe_game(game, "a_golddiffat15"),
                "golddiffat20": sign * _safe_game(game, "a_golddiffat20"),
                "xpdiffat10": sign * _safe_game(game, "a_xpdiffat10"),
                "gamelength": _safe_game(game, "gamelength", 1800),
                "kills": _safe_game(game, f"{prefix}_kills"),
                "dragons": _safe_game(game, f"{prefix}_dragons"),
                "barons": _safe_game(game, f"{prefix}_barons"),
                "towers": _safe_game(game, f"{prefix}_towers"),
                "firstblood": _safe_game(game, f"{prefix}_firstblood"),
                "firstdragon": _safe_game(game, f"{prefix}_firstdragon"),
                "firstherald": _safe_game(game, f"{prefix}_firstherald"),
                "firsttower": _safe_game(game, f"{prefix}_firsttower"),
                "totalgold": _safe_game(game, f"{prefix}_totalgold"),
                "turretplates": _safe_game(game, f"{prefix}_turretplates"),
                "opp_turretplates": _safe_game(game, f"{prefix}_opp_turretplates"),
            }

            # CS diff at 10 — approximate from team-level data if available
            team_stats["csdiffat10"] = 0.0  # Not available at team level

            team_tracker.record_game(team, team_stats)

        # Update player Elo and player tracker
        expected_a = team_elo.expected(team_a, team_b)
        for pg in game_player_data:
            won = pg["team"] == winner
            team_exp = expected_a if pg["team"] == team_a else (1 - expected_a)

            # Contribution weight: scale by KDA relative to game average
            avg_kda = np.mean([p["kda"] for p in game_player_data]) or 2.0
            contribution = min(1.5, max(0.5, pg["kda"] / avg_kda))

            player_elo.update(pg["player"], won, team_exp, contribution)
            player_tracker.record_game(pg["player"], pg["position"], pg)

            # Update champion tracker with this game's result
            champ_tracker.record_game(
                champion=pg.get("champion", "unknown"),
                player=pg["player"],
                position=pg["position"],
                stats=pg,
            )

        if (i + 1) % 2000 == 0:
            n_valid = sum(valid_mask)
            print(f"  Processed {i + 1}/{len(games)} games ({n_valid} valid)")

    X = np.array(features_list)
    y = np.array(targets)
    valid = np.array(valid_mask)

    print(f"\nFeature matrix: {X.shape[0]} games x {X.shape[1]} features")
    print(f"Valid games (enough history + target): {valid.sum()}")

    return X, y, feature_names, game_ids, valid


# =============================================================================
# PCA ANALYSIS
# =============================================================================

def run_pca_analysis(X: np.ndarray, y: np.ndarray, feature_names: list,
                     valid: np.ndarray, n_components: int = 30):
    """
    Run PCA on the pre-game feature matrix and analyze variance structure.

    Args:
        X: Feature matrix (n_games x n_features)
        y: Target values (golddiffat10)
        feature_names: List of feature names
        valid: Boolean mask for valid games
        n_components: Number of PCA components to analyze
    """
    # Filter to valid games
    X_valid = X[valid]
    y_valid = y[valid]

    print(f"\n{'='*80}")
    print("PCA ANALYSIS: Pre-Game Features → Gold Diff @ 10 Minutes")
    print(f"{'='*80}")
    print(f"Valid games: {len(X_valid)}")
    print(f"Features: {len(feature_names)}")

    # Remove constant / near-constant features
    stds = np.std(X_valid, axis=0)
    nonconstant = stds > 1e-8
    X_filtered = X_valid[:, nonconstant]
    filtered_names = [n for n, keep in zip(feature_names, nonconstant) if keep]
    n_removed = len(feature_names) - len(filtered_names)
    if n_removed > 0:
        print(f"Removed {n_removed} constant features")

    # Handle NaN/Inf
    X_filtered = np.nan_to_num(X_filtered, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)

    # Run PCA
    n_components = min(n_components, X_scaled.shape[1], X_scaled.shape[0])
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    print(f"\n{'─'*80}")
    print("VARIANCE EXPLAINED BY PRINCIPAL COMPONENTS")
    print(f"{'─'*80}")
    print(f"{'PC':<6} {'Variance %':>12} {'Cumulative %':>14}")
    print(f"{'─'*6} {'─'*12} {'─'*14}")

    cumulative = 0.0
    for i, var in enumerate(pca.explained_variance_ratio_):
        cumulative += var
        bar = "█" * int(var * 100)
        print(f"PC{i+1:<4} {var*100:>11.2f}% {cumulative*100:>13.2f}%  {bar}")
        if cumulative > 0.95:
            print(f"  ... ({n_components - i - 1} more components omitted, <5% remaining)")
            break

    # Feature importance: which features load most heavily on top PCs
    print(f"\n{'─'*80}")
    print("TOP FEATURE LOADINGS PER PRINCIPAL COMPONENT")
    print(f"{'─'*80}")

    for pc_idx in range(min(10, n_components)):
        loadings = pca.components_[pc_idx]
        # Top positive and negative loadings
        top_idx = np.argsort(np.abs(loadings))[::-1][:10]

        var_pct = pca.explained_variance_ratio_[pc_idx] * 100
        print(f"\nPC{pc_idx+1} ({var_pct:.1f}% variance):")
        for j in top_idx:
            direction = "+" if loadings[j] > 0 else "-"
            print(f"  {direction} {filtered_names[j]:<55} {loadings[j]:>+.4f}")

    # Correlation of each PC with the target (gold diff @ 10)
    print(f"\n{'─'*80}")
    print("PC CORRELATION WITH GOLD DIFF @ 10")
    print(f"{'─'*80}")
    print(f"{'PC':<6} {'Correlation':>12} {'Var Explained':>14} {'Target Info':>14}")
    print(f"{'─'*6} {'─'*12} {'─'*14} {'─'*14}")

    for i in range(min(15, n_components)):
        corr = np.corrcoef(X_pca[:, i], y_valid)[0, 1]
        var_pct = pca.explained_variance_ratio_[i] * 100
        # "Target info" = variance * correlation^2 — how much this PC matters for gold@10
        info = var_pct * corr**2
        bar = "█" * int(abs(corr) * 40)
        print(f"PC{i+1:<4} {corr:>+11.4f} {var_pct:>13.2f}% {info:>13.4f}%  {bar}")

    # Individual feature correlations with target
    print(f"\n{'─'*80}")
    print("TOP INDIVIDUAL FEATURE CORRELATIONS WITH GOLD DIFF @ 10")
    print(f"{'─'*80}")

    correlations = []
    for j in range(X_filtered.shape[1]):
        corr = np.corrcoef(X_filtered[:, j], y_valid)[0, 1]
        if not np.isnan(corr):
            correlations.append((filtered_names[j], corr, abs(corr)))

    correlations.sort(key=lambda x: x[2], reverse=True)

    print(f"\n{'Feature':<60} {'Correlation':>12}")
    print(f"{'─'*60} {'─'*12}")
    for name, corr, _ in correlations[:40]:
        bar = "█" * int(abs(corr) * 40)
        print(f"{name:<60} {corr:>+11.4f}  {bar}")

    # Feature family analysis
    print(f"\n{'─'*80}")
    print("VARIANCE BY FEATURE FAMILY")
    print(f"{'─'*80}")

    families = {
        "Team Elo": [n for n in filtered_names if "team_elo" in n or n == "elo_diff"],
        "Team Momentum (5/10/15 win%)": [n for n in filtered_names if "momentum" in n or "streak" in n],
        "Team Gold Diff History": [n for n in filtered_names if "gd10" in n or "gd15" in n or "gd20" in n],
        "Team Objectives (FB/FD/FH/FT)": [n for n in filtered_names if any(x in n for x in ["fb_rate", "fd_rate", "fh_rate", "ft_rate"])],
        "Team Game Style": [n for n in filtered_names if any(x in n for x in ["gamelength", "kills", "dragons", "towers", "barons", "plates"])],
        "Player Elo": [n for n in filtered_names if "player_elo" in n],
        "Player Win Rates": [n for n in filtered_names if "player_win_rate" in n or "player_recent_wr" in n],
        "Lane Win Rates": [n for n in filtered_names if "lane_win_rate" in n or "lane_wr" in n],
        "Lane Dominance (GD10/XPD10/CSD10)": [n for n in filtered_names if any(x in n for x in ["lane_gd10", "lane_xpd10", "lane_csd10"])],
        "Lane Economy (CS/min, Gold/min)": [n for n in filtered_names if any(x in n for x in ["lane_cspm", "lane_gpm"])],
        "Player KDA": [n for n in filtered_names if "kda" in n],
        "Champion Win Rates": [n for n in filtered_names if "champ_win_rate" in n or "champ_wr" in n or "champ_position_wr" in n],
        "Champion GD10 Tendency": [n for n in filtered_names if "champ_gd10" in n or "champ_position_gd10" in n],
        "Player-Champion Comfort": [n for n in filtered_names if "player_champ" in n],
        "Context (Side, Playoffs)": [n for n in filtered_names if any(x in n for x in ["side", "playoff"])],
    }

    name_to_idx = {n: i for i, n in enumerate(filtered_names)}

    for family_name, family_features in families.items():
        if not family_features:
            continue

        # Get indices of features in this family
        fidx = [name_to_idx[f] for f in family_features if f in name_to_idx]
        if not fidx:
            continue

        # Average absolute correlation with target
        corrs = [abs(np.corrcoef(X_filtered[:, j], y_valid)[0, 1])
                 for j in fidx if not np.isnan(np.corrcoef(X_filtered[:, j], y_valid)[0, 1])]
        avg_corr = np.mean(corrs) if corrs else 0.0
        max_corr = max(corrs) if corrs else 0.0

        # Variance explained by this family (using sub-PCA)
        if len(fidx) >= 2:
            X_family = X_scaled[:, fidx]
            family_pca = PCA(n_components=min(3, len(fidx)))
            family_pca.fit(X_family)
            family_var = sum(family_pca.explained_variance_ratio_[:3]) * 100
        else:
            family_var = 100.0

        bar = "█" * int(avg_corr * 40)
        print(f"\n{family_name} ({len(fidx)} features):")
        print(f"  Avg |corr| with GD@10: {avg_corr:.4f}  Max: {max_corr:.4f}  {bar}")
        print(f"  Top-3 PC variance within family: {family_var:.1f}%")

        # Top features in this family
        family_corrs = [(filtered_names[j], np.corrcoef(X_filtered[:, j], y_valid)[0, 1])
                        for j in fidx]
        family_corrs.sort(key=lambda x: abs(x[1]), reverse=True)
        for name, corr in family_corrs[:3]:
            print(f"    {name:<55} r={corr:+.4f}")

    # Summary recommendations
    print(f"\n{'='*80}")
    print("SUMMARY: WHERE THE VARIANCE LIVES")
    print(f"{'='*80}")

    # Rank families by average correlation
    family_ranks = []
    for family_name, family_features in families.items():
        fidx = [name_to_idx[f] for f in family_features if f in name_to_idx]
        if not fidx:
            continue
        corrs = [abs(np.corrcoef(X_filtered[:, j], y_valid)[0, 1])
                 for j in fidx if not np.isnan(np.corrcoef(X_filtered[:, j], y_valid)[0, 1])]
        if corrs:
            family_ranks.append((family_name, np.mean(corrs), max(corrs), len(fidx)))

    family_ranks.sort(key=lambda x: x[1], reverse=True)

    print(f"\n{'Rank':<6} {'Feature Family':<45} {'Avg |r|':>8} {'Max |r|':>8} {'#Feats':>7}")
    print(f"{'─'*6} {'─'*45} {'─'*8} {'─'*8} {'─'*7}")
    for rank, (name, avg, mx, count) in enumerate(family_ranks, 1):
        print(f"{rank:<6} {name:<45} {avg:>7.4f} {mx:>7.4f} {count:>7}")

    total_var_90 = 0
    cumulative = 0.0
    for i, var in enumerate(pca.explained_variance_ratio_):
        cumulative += var
        if cumulative >= 0.90:
            total_var_90 = i + 1
            break

    print(f"\nPCs needed for 90% variance: {total_var_90}")
    print(f"Total features: {len(filtered_names)}")
    print(f"Effective dimensionality: ~{total_var_90} (compression ratio: {len(filtered_names)/max(total_var_90,1):.1f}x)")

    return {
        "pca": pca,
        "scaler": scaler,
        "feature_names": filtered_names,
        "X_valid": X_valid,
        "y_valid": y_valid,
        "X_pca": X_pca,
        "correlations": correlations,
        "family_ranks": family_ranks,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="PCA analysis of pre-game features for gold@10 prediction"
    )
    parser.add_argument("csv_path", help="Path to Oracle's Elixir CSV file")
    parser.add_argument("--leagues", nargs="+", default=None,
                        help="Filter to specific leagues (e.g., LCK LPL LEC)")
    parser.add_argument("--min-date", default=None,
                        help="Minimum date filter (YYYY-MM-DD)")
    parser.add_argument("--min-team-games", type=int, default=5,
                        help="Minimum team games before including in analysis")
    parser.add_argument("--min-player-games", type=int, default=3,
                        help="Minimum player games before including in analysis")
    parser.add_argument("--n-components", type=int, default=30,
                        help="Number of PCA components to analyze")
    args = parser.parse_args()

    # Load data at player level
    df = load_player_level(args.csv_path, leagues=args.leagues,
                           min_date=args.min_date)

    # Extract structured game + player data
    games, player_games = extract_game_player_data(df)

    # Build feature matrix (chronological, no leakage)
    X, y, feature_names, game_ids, valid = build_feature_matrix(
        games, player_games,
        min_team_games=args.min_team_games,
        min_player_games=args.min_player_games,
    )

    # Run PCA analysis
    results = run_pca_analysis(X, y, feature_names, valid,
                               n_components=args.n_components)

    print("\nDone. Use results dict for further analysis.")
    return results


if __name__ == "__main__":
    main()
