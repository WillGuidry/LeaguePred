"""
Data Pipeline
-------------
Ingest Oracle's Elixir CSV exports, clean, normalize per-patch,
parse champion picks/bans, and produce analysis-ready DataFrames.
"""

import os
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def ingest_oracle_data(csv_path: str) -> pd.DataFrame:
    """
    Read an Oracle's Elixir CSV export and return a raw DataFrame.

    Oracle's Elixir exports contain one row per player per game (10 rows per game)
    plus team-level summary rows (2 per game, 12 total rows per game).

    Args:
        csv_path: Path to the CSV file.

    Returns:
        Raw DataFrame with all columns preserved.
    """
    df = pd.read_csv(csv_path, low_memory=False)
    # Standardize column names to lowercase with underscores
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df


def ingest_multiple_csvs(csv_dir: str) -> pd.DataFrame:
    """
    Read all CSV files in a directory and concatenate them.

    Args:
        csv_dir: Directory containing Oracle's Elixir CSV exports.

    Returns:
        Combined DataFrame from all CSVs.
    """
    csv_files = sorted(Path(csv_dir).glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {csv_dir}")

    frames = []
    for f in csv_files:
        df = ingest_oracle_data(str(f))
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def parse_patch_version(patch_str: str) -> tuple:
    """
    Parse a patch string like '14.3' into (major, minor) tuple.

    Returns:
        Tuple of (major, minor) integers, or (0, 0) for unparseable patches.
    """
    if pd.isna(patch_str):
        return (0, 0)
    match = re.match(r"(\d+)\.(\d+)", str(patch_str))
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (0, 0)


def add_patch_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a patch_window column that groups games by patch version.

    The patch_window is a string like '14.03' (zero-padded minor version
    for correct sorting).
    """
    df = df.copy()

    def _make_window(patch):
        major, minor = parse_patch_version(patch)
        if major == 0 and minor == 0:
            return "unknown"
        return f"{major}.{minor:02d}"

    df["patch_window"] = df["patch"].apply(_make_window)
    return df


def clean_and_normalize(df: pd.DataFrame, config: Optional[dict] = None) -> pd.DataFrame:
    """
    Clean and normalize Oracle's Elixir data.

    Steps:
    - Parse date column to datetime
    - Filter to configured leagues
    - Separate team-level rows from player-level rows
    - Normalize numeric columns per patch window
    - Add derived columns
    """
    df = df.copy()

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Filter leagues
    if config:
        leagues = config.get("data", {}).get("leagues", [])
        if leagues and "league" in df.columns:
            df = df[df["league"].isin(leagues)].copy()

    # Add patch window
    if "patch" in df.columns:
        df = add_patch_window(df)

    # Ensure numeric types for key columns
    numeric_cols = [
        "golddiffat15", "kills", "deaths", "assists",
        "dragons", "barons", "towers", "gamelength",
        "firstdragon", "firstherald", "result",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Sort by date for chronological processing
    if "date" in df.columns:
        df = df.sort_values("date").reset_index(drop=True)

    return df


def split_team_player_rows(df: pd.DataFrame) -> tuple:
    """
    Split the DataFrame into team-level and player-level rows.

    Oracle's Elixir uses 'position' == 'team' for team-level aggregates.

    Returns:
        Tuple of (team_df, player_df).
    """
    if "position" not in df.columns:
        return df, pd.DataFrame()

    team_df = df[df["position"] == "team"].copy()
    player_df = df[df["position"] != "team"].copy()
    return team_df, player_df


def extract_draft_data(player_df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse champion picks per game with player and position assignments.

    From the player-level rows, extract the draft information for each game.

    Returns:
        DataFrame with columns:
        [gameid, side, teamname, position, playername, champion]
    """
    cols = ["gameid", "side", "teamname", "position", "playername", "champion"]
    available = [c for c in cols if c in player_df.columns]

    if not available:
        return pd.DataFrame(columns=cols)

    draft_df = player_df[available].copy()
    draft_df = draft_df.dropna(subset=["champion"])
    return draft_df


def extract_bans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract ban data from team-level rows.

    Oracle's Elixir stores bans in columns like ban1, ban2, ..., ban5.

    Returns:
        DataFrame with columns: [gameid, side, teamname, ban_order, champion_banned]
    """
    ban_cols = [c for c in df.columns if re.match(r"ban\d+$", c)]
    if not ban_cols:
        return pd.DataFrame(columns=["gameid", "side", "teamname", "ban_order", "champion_banned"])

    id_cols = ["gameid", "side", "teamname"]
    available_ids = [c for c in id_cols if c in df.columns]

    # Only from team rows
    team_df = df[df.get("position", "") == "team"] if "position" in df.columns else df

    records = []
    for _, row in team_df.iterrows():
        for i, ban_col in enumerate(sorted(ban_cols), 1):
            champ = row.get(ban_col)
            if pd.notna(champ):
                record = {c: row.get(c) for c in available_ids}
                record["ban_order"] = i
                record["champion_banned"] = champ
                records.append(record)

    return pd.DataFrame(records)


def build_game_level_dataset(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot team-level rows into one row per game with team_a / team_b columns.

    Each game has exactly 2 team-level rows (Blue side and Red side).
    We pivot so each game becomes a single row with features for both teams.

    Returns:
        DataFrame with one row per game, columns prefixed with team_a_ / team_b_.
    """
    if team_df.empty:
        return pd.DataFrame()

    # Separate by side
    blue = team_df[team_df["side"] == "Blue"].copy()
    red = team_df[team_df["side"] == "Red"].copy()

    if blue.empty or red.empty:
        return pd.DataFrame()

    # Rename columns with prefixes
    blue_renamed = blue.rename(
        columns={c: f"team_a_{c}" for c in blue.columns if c != "gameid"}
    )
    red_renamed = red.rename(
        columns={c: f"team_b_{c}" for c in red.columns if c != "gameid"}
    )

    # Merge on gameid
    merged = pd.merge(blue_renamed, red_renamed, on="gameid", how="inner")

    # Add convenience columns
    if "team_a_teamname" in merged.columns:
        merged["team_a"] = merged["team_a_teamname"]
        merged["team_b"] = merged["team_b_teamname"]

    if "team_a_result" in merged.columns:
        merged["winner"] = np.where(
            merged["team_a_result"] == 1,
            merged.get("team_a", "Blue"),
            merged.get("team_b", "Red"),
        )

    # Use team_a's metadata for shared columns
    for col in ["date", "league", "patch", "patch_window", "gamelength"]:
        a_col = f"team_a_{col}"
        if a_col in merged.columns:
            merged[col] = merged[a_col]

    return merged


def normalize_features_per_patch(
    df: pd.DataFrame,
    feature_cols: list,
    patch_col: str = "patch_window",
) -> pd.DataFrame:
    """
    Z-score normalize feature columns within each patch window.

    This accounts for meta shifts where e.g. average gold diff @15
    may be systematically higher or lower on certain patches.
    """
    df = df.copy()
    for col in feature_cols:
        if col not in df.columns:
            continue
        col_norm = f"{col}_norm"
        # Group by patch and z-score normalize
        grouped = df.groupby(patch_col)[col]
        df[col_norm] = grouped.transform(
            lambda x: (x - x.mean()) / x.std() if x.std() > 0 else 0.0
        )
    return df


def run_pipeline(csv_path_or_dir: str, config_path: str = "config.yaml") -> dict:
    """
    Run the full data pipeline.

    Args:
        csv_path_or_dir: Path to a single CSV or directory of CSVs.
        config_path: Path to config.yaml.

    Returns:
        Dict with keys:
        - 'raw': raw ingested DataFrame
        - 'team': cleaned team-level rows
        - 'player': cleaned player-level rows
        - 'games': one-row-per-game pivoted dataset
        - 'draft': champion pick assignments
        - 'bans': champion ban data
    """
    config = load_config(config_path)

    # Ingest
    path = Path(csv_path_or_dir)
    if path.is_dir():
        raw_df = ingest_multiple_csvs(str(path))
    else:
        raw_df = ingest_oracle_data(str(path))

    # Clean
    clean_df = clean_and_normalize(raw_df, config)

    # Split
    team_df, player_df = split_team_player_rows(clean_df)

    # Draft
    draft_df = extract_draft_data(player_df)
    bans_df = extract_bans(clean_df)

    # Game-level dataset
    games_df = build_game_level_dataset(team_df)

    # Normalize key features per patch
    feature_cols_to_norm = ["team_a_golddiffat15", "team_b_golddiffat15"]
    if not games_df.empty:
        games_df = normalize_features_per_patch(games_df, feature_cols_to_norm)

    return {
        "raw": raw_df,
        "team": team_df,
        "player": player_df,
        "games": games_df,
        "draft": draft_df,
        "bans": bans_df,
    }


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_pipeline.py <csv_path_or_dir> [config.yaml]")
        print("  csv_path_or_dir: Path to Oracle's Elixir CSV or directory of CSVs")
        sys.exit(1)

    csv_input = sys.argv[1]
    cfg_path = sys.argv[2] if len(sys.argv) > 2 else "config.yaml"

    result = run_pipeline(csv_input, cfg_path)
    print(f"Raw rows:     {len(result['raw']):,}")
    print(f"Team rows:    {len(result['team']):,}")
    print(f"Player rows:  {len(result['player']):,}")
    print(f"Games:        {len(result['games']):,}")
    print(f"Draft picks:  {len(result['draft']):,}")
    print(f"Bans:         {len(result['bans']):,}")

    if not result["games"].empty and "patch_window" in result["games"].columns:
        print(f"\nPatch windows: {sorted(result['games']['patch_window'].unique())}")
