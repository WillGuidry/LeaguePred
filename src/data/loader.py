"""
Data Loader
------------
Ingests match-level data from CSV files (Oracle's Elixir format or similar).

Oracle's Elixir data has one row per player per game (10 rows per game).
We collapse that into one row per game with team-level aggregates.

Supports:
    - Oracle's Elixir CSV (auto-detected)
    - Generic match-level CSV (team_a, team_b, winner, date)
    - Custom CSV with column mapping

Leakage note:
    All columns are checked for pre-match availability.
    Post-game stats (kills, gold, etc.) are kept in a separate frame
    and NEVER mixed into pre-match features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional


# Columns that are available BEFORE match start (safe for prediction)
PRE_MATCH_COLS = [
    "date", "league", "year", "split", "patch",
    "team_a", "team_b", "side_a", "side_b",
    "playoffs", "game_number",
]

# Columns that are only available AFTER match ends (potential leakage)
POST_MATCH_COLS = [
    "winner", "gamelength", "result",
    "kills", "deaths", "assists",
    "firstblood", "firstdragon", "firstherald", "firstbaron", "firsttower",
    "dragons", "barons", "towers", "inhibitors",
    "goldat10", "goldat15", "xpat10", "xpat15",
    "csat10", "csat15", "golddiffat10", "golddiffat15",
    "totalgold", "earnedgold", "minionkills", "monsterkills",
]


def load_oracle_csv(filepath: str, min_date: Optional[str] = None) -> pd.DataFrame:
    """
    Load an Oracle's Elixir CSV and collapse to game-level rows.

    Oracle's format: 12 rows per game (10 players + 2 team summary rows).
    We keep only the team summary rows (where 'position' == 'team')
    and pivot to get one row per game.

    Args:
        filepath: Path to the CSV file.
        min_date: Optional minimum date filter (YYYY-MM-DD).

    Returns:
        DataFrame with one row per game, columns for both teams.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, low_memory=False)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Auto-detect format
    if "position" in df.columns:
        return _parse_oracle_format(df, min_date)
    elif "team_a" in df.columns and "team_b" in df.columns:
        return _parse_generic_format(df, min_date)
    else:
        # Try to be flexible — look for common column patterns
        return _parse_flexible_format(df, min_date)


def _parse_oracle_format(df: pd.DataFrame, min_date: Optional[str]) -> pd.DataFrame:
    """Parse Oracle's Elixir format (12 rows per game)."""

    # Keep only team summary rows
    team_rows = df[df["position"] == "team"].copy()

    if team_rows.empty:
        raise ValueError(
            "No team-level rows found. Check that 'position' column contains 'team' entries."
        )

    # Parse date
    if "date" in team_rows.columns:
        team_rows["date"] = pd.to_datetime(team_rows["date"], errors="coerce")
        if min_date:
            team_rows = team_rows[team_rows["date"] >= pd.to_datetime(min_date)]

    # Each game has a unique gameid with two team rows
    # Split into blue side (first row) and red side (second row)
    if "gameid" not in team_rows.columns:
        raise ValueError("No 'gameid' column found. Cannot group rows into games.")

    games = []
    for game_id, group in team_rows.groupby("gameid"):
        if len(group) != 2:
            continue  # Skip malformed games

        row_a = group.iloc[0]
        row_b = group.iloc[1]

        game = _build_game_row(row_a, row_b, game_id)
        games.append(game)

    result = pd.DataFrame(games)

    # Sort by date
    if "date" in result.columns:
        result = result.sort_values("date").reset_index(drop=True)

    print(f"Loaded {len(result)} games from {filepath_summary(result)}")
    return result


def _build_game_row(row_a: pd.Series, row_b: pd.Series, game_id: str) -> dict:
    """Build a single game row from two team rows."""
    game = {"gameid": game_id}

    # Team names
    team_col = "teamname" if "teamname" in row_a.index else "team"
    game["team_a"] = row_a.get(team_col, "Unknown")
    game["team_b"] = row_b.get(team_col, "Unknown")

    # Side assignments
    if "side" in row_a.index:
        game["side_a"] = str(row_a["side"]).lower() if pd.notna(row_a["side"]) else "blue"
        game["side_b"] = str(row_b["side"]).lower() if pd.notna(row_b["side"]) else "red"

    # Result — determine winner
    if "result" in row_a.index:
        if row_a["result"] == 1:
            game["winner"] = game["team_a"]
        else:
            game["winner"] = game["team_b"]

    # Context columns
    for col in ["date", "league", "year", "split", "patch", "playoffs", "gamelength"]:
        if col in row_a.index:
            game[col] = row_a[col]

    # Game number within a series (bo3/bo5)
    for col in ["game", "game_number"]:
        if col in row_a.index:
            game["game_number"] = row_a[col]
            break

    # Post-match stats for team A (prefixed)
    # These are kept for validation/analysis but flagged as post-match
    stat_cols = [
        "kills", "deaths", "assists", "dragons", "barons",
        "towers", "inhibitors", "totalgold", "earnedgold",
        "goldat10", "goldat15", "xpat10", "xpat15",
        "golddiffat10", "golddiffat15",
        "firstblood", "firstdragon", "firstherald", "firstbaron", "firsttower",
        "minionkills", "monsterkills",
    ]

    for col in stat_cols:
        if col in row_a.index:
            game[f"a_{col}"] = row_a[col]
            game[f"b_{col}"] = row_b[col]

    return game


def _parse_generic_format(df: pd.DataFrame, min_date: Optional[str]) -> pd.DataFrame:
    """Parse a simple CSV with team_a, team_b, winner columns."""
    required = ["team_a", "team_b", "winner"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if min_date:
            df = df[df["date"] >= pd.to_datetime(min_date)]
        df = df.sort_values("date").reset_index(drop=True)

    print(f"Loaded {len(df)} games (generic format)")
    return df


def _parse_flexible_format(df: pd.DataFrame, min_date: Optional[str]) -> pd.DataFrame:
    """
    Attempt to parse an unknown CSV format by looking for common patterns.
    Raises informative errors if we can't figure it out.
    """
    # Look for team columns with various names
    team_patterns = {
        "team_a": ["team_a", "team1", "teamname_a", "blue_team", "blueteam"],
        "team_b": ["team_b", "team2", "teamname_b", "red_team", "redteam"],
        "winner": ["winner", "winning_team", "result"],
    }

    col_map = {}
    for target, candidates in team_patterns.items():
        for candidate in candidates:
            if candidate in df.columns:
                col_map[candidate] = target
                break

    if len(col_map) < 2:
        available = ", ".join(df.columns[:20].tolist())
        raise ValueError(
            f"Could not auto-detect format. Available columns: {available}\n"
            f"Expected columns like: team_a, team_b, winner (or Oracle's Elixir format with 'position' column)"
        )

    df = df.rename(columns=col_map)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        if min_date:
            df = df[df["date"] >= pd.to_datetime(min_date)]
        df = df.sort_values("date").reset_index(drop=True)

    print(f"Loaded {len(df)} games (auto-detected format)")
    return df


def filepath_summary(df: pd.DataFrame) -> str:
    """Generate a human-readable summary of the loaded data."""
    parts = []
    if "date" in df.columns:
        dates = df["date"].dropna()
        if len(dates) > 0:
            parts.append(f"{dates.min().strftime('%Y-%m-%d')} to {dates.max().strftime('%Y-%m-%d')}")
    if "league" in df.columns:
        leagues = df["league"].dropna().unique()
        parts.append(f"{len(leagues)} leagues")
    return ", ".join(parts) if parts else "unknown date range"


def get_pre_match_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only pre-match columns from the dataframe.
    Use this to prevent leakage when building features.
    """
    available = [c for c in PRE_MATCH_COLS if c in df.columns]
    return df[available].copy()


def get_post_match_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract post-match stats (for validation and module training only).
    NEVER use these as prediction features.
    """
    stat_cols = [c for c in df.columns if c.startswith(("a_", "b_"))]
    meta_cols = [c for c in ["gameid", "date", "team_a", "team_b", "winner"] if c in df.columns]
    return df[meta_cols + stat_cols].copy()


def validate_data(df: pd.DataFrame) -> dict:
    """
    Run basic quality checks on loaded data.

    Returns:
        Dict with validation results and warnings.
    """
    report = {
        "total_games": len(df),
        "warnings": [],
        "errors": [],
    }

    if len(df) == 0:
        report["errors"].append("No games loaded.")
        return report

    # Check required columns
    if "winner" not in df.columns:
        report["errors"].append("No 'winner' column — cannot evaluate predictions.")

    if "team_a" not in df.columns or "team_b" not in df.columns:
        report["errors"].append("Missing team columns (team_a / team_b).")

    # Check for missing winners
    if "winner" in df.columns:
        missing_winners = df["winner"].isna().sum()
        if missing_winners > 0:
            report["warnings"].append(f"{missing_winners} games have no winner recorded.")

    # Check date coverage
    if "date" in df.columns:
        dates = df["date"].dropna()
        if len(dates) > 0:
            report["date_range"] = (
                dates.min().strftime("%Y-%m-%d"),
                dates.max().strftime("%Y-%m-%d"),
            )
            # Check for large gaps
            sorted_dates = dates.sort_values()
            gaps = sorted_dates.diff().dt.days
            max_gap = gaps.max()
            if max_gap and max_gap > 60:
                report["warnings"].append(
                    f"Largest gap between games: {int(max_gap)} days. "
                    "Check for missing data or off-season periods."
                )

    # Check team counts
    if "team_a" in df.columns and "team_b" in df.columns:
        all_teams = pd.concat([df["team_a"], df["team_b"]]).dropna().unique()
        report["unique_teams"] = len(all_teams)

        # Check for duplicate team names (common data issue)
        team_lower = pd.Series([t.lower().strip() for t in all_teams])
        dupes = team_lower[team_lower.duplicated()].unique()
        if len(dupes) > 0:
            report["warnings"].append(
                f"Possible duplicate team names (case/whitespace): {list(dupes)[:5]}"
            )

    # Check league coverage
    if "league" in df.columns:
        report["leagues"] = df["league"].dropna().unique().tolist()

    return report


def print_validation_report(report: dict) -> None:
    """Pretty-print a validation report."""
    print(f"\n{'='*50}")
    print("DATA VALIDATION REPORT")
    print(f"{'='*50}")
    print(f"Total games: {report['total_games']}")

    if "date_range" in report:
        print(f"Date range: {report['date_range'][0]} to {report['date_range'][1]}")
    if "unique_teams" in report:
        print(f"Unique teams: {report['unique_teams']}")
    if "leagues" in report:
        print(f"Leagues: {', '.join(report['leagues'][:10])}")

    if report.get("warnings"):
        print(f"\nWarnings ({len(report['warnings'])}):")
        for w in report["warnings"]:
            print(f"  ⚠ {w}")

    if report.get("errors"):
        print(f"\nErrors ({len(report['errors'])}):")
        for e in report["errors"]:
            print(f"  ✗ {e}")

    if not report.get("warnings") and not report.get("errors"):
        print("\nNo issues found.")
    print()
