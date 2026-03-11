"""
ingest.py
---------
End-to-end data ingestion: download Oracle's Elixir CSVs from Google Drive,
run the data pipeline, validate output, and save processed DataFrames to disk.

Usage:
    # Download and process (uses Google Drive folder by default):
    python ingest.py

    # Process local CSVs:
    python ingest.py --local data/raw

    # Download specific years:
    python ingest.py --years 2024 2025 2026
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

from data_pipeline import run_pipeline

# Google Drive file IDs for Oracle's Elixir match data
GDRIVE_FILE_IDS = {
    2020: "1dlSIczXShnv1vIfGNvBjgk-thMKA5j7d",
    2021: "1fzwTTz77hcnYjOnO9ONeoPrkWCoOSecA",
    2022: "1EHmptHyzY8owv0BAcNKtkQpMwfkURwRy",
    2023: "1XXk2LO0CsNADBB1LRGOV5rUpyZdEZ8s2",
    2024: "1IjIEhLc9n8eLKeY-yh_YigKVWbhgGBsN",
    2025: "1v6LRphp2kYciU4SXp0PCjEMuev1bDejc",
    2026: "1hnpbrUpBMS1TZI7IovfpKeZfWJH1Aptm",
}

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")


def download_from_gdrive(file_id: str, dest_path: Path) -> bool:
    """Download a file from Google Drive by file ID."""
    import requests

    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    session = requests.Session()

    # First request — may return a confirmation page for large files
    resp = session.get(url, stream=True)

    # Check for virus scan confirmation redirect
    confirm_token = None
    for key, value in resp.cookies.items():
        if key.startswith("download_warning"):
            confirm_token = value
            break

    if confirm_token:
        url = f"{url}&confirm={confirm_token}"
        resp = session.get(url, stream=True)

    # Verify we got CSV, not HTML
    content_start = b""
    for chunk in resp.iter_content(chunk_size=1024):
        content_start = chunk
        break

    if content_start and b"<html" in content_start.lower():
        print(f"  ERROR: Got HTML instead of CSV for {dest_path.name}")
        return False

    # Write file
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_path, "wb") as f:
        f.write(content_start)
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    return True


def download_years(years: list[int], force: bool = False) -> list[Path]:
    """Download CSVs for specified years. Skip if already present."""
    downloaded = []

    for year in years:
        if year not in GDRIVE_FILE_IDS:
            print(f"  SKIP: No file ID for {year}")
            continue

        filename = f"{year}_LoL_esports_match_data_from_OraclesElixir.csv"
        dest = RAW_DIR / filename

        if dest.exists() and not force:
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"  EXISTS: {filename} ({size_mb:.1f} MB)")
            downloaded.append(dest)
            continue

        print(f"  Downloading {filename}...", end=" ", flush=True)
        if download_from_gdrive(GDRIVE_FILE_IDS[year], dest):
            size_mb = dest.stat().st_size / (1024 * 1024)
            print(f"OK ({size_mb:.1f} MB)")
            downloaded.append(dest)
        else:
            print("FAILED")

    return downloaded


def validate_raw(csv_path: Path) -> dict:
    """Quick validation of a raw Oracle's Elixir CSV."""
    df = pd.read_csv(csv_path, nrows=100, low_memory=False)
    cols = [c.strip().lower().replace(" ", "_") for c in df.columns]

    required = {"gameid", "date", "league", "side", "teamname", "position", "result"}
    missing = required - set(cols)

    return {
        "file": csv_path.name,
        "columns": len(cols),
        "missing_required": list(missing),
        "valid": len(missing) == 0,
    }


def validate_pipeline_output(result: dict) -> dict:
    """Validate the pipeline output."""
    issues = []

    if result["games"].empty:
        issues.append("games DataFrame is empty")

    if not result["games"].empty:
        games = result["games"]

        if "team_a" not in games.columns or "team_b" not in games.columns:
            issues.append("missing team_a/team_b columns")

        if "winner" not in games.columns:
            issues.append("missing winner column")

        if "date" in games.columns:
            null_dates = games["date"].isna().sum()
            if null_dates > 0:
                issues.append(f"{null_dates} games with null dates")

        if "team_a_result" in games.columns:
            results = games["team_a_result"].dropna()
            if not results.isin([0, 1, 0.0, 1.0]).all():
                issues.append("unexpected values in result column")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "raw_rows": len(result["raw"]),
        "team_rows": len(result["team"]),
        "player_rows": len(result["player"]),
        "games": len(result["games"]),
        "draft_picks": len(result["draft"]),
        "bans": len(result["bans"]),
    }


def save_processed(result: dict, output_dir: Path):
    """Save processed DataFrames to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for key in ["team", "player", "games", "draft", "bans"]:
        df = result[key]
        if not df.empty:
            path = output_dir / f"{key}.csv"
            df.to_csv(path, index=False)
            print(f"  Saved {key}.csv ({len(df):,} rows)")


def print_summary(result: dict):
    """Print a human-readable summary of the processed data."""
    games = result["games"]
    if games.empty:
        print("\n  No games produced!")
        return

    print(f"\n  Games:        {len(games):,}")
    print(f"  Team rows:    {len(result['team']):,}")
    print(f"  Player rows:  {len(result['player']):,}")
    print(f"  Draft picks:  {len(result['draft']):,}")
    print(f"  Bans:         {len(result['bans']):,}")

    if "date" in games.columns:
        dates = games["date"].dropna()
        if len(dates) > 0:
            print(f"  Date range:   {dates.min()} → {dates.max()}")

    if "league" in games.columns:
        leagues = games["league"].dropna().unique()
        print(f"  Leagues ({len(leagues)}): {', '.join(sorted(leagues))}")

    if "patch_window" in games.columns:
        patches = sorted(games["patch_window"].dropna().unique())
        print(f"  Patches ({len(patches)}): {', '.join(patches)}")

    if "team_a" in games.columns:
        teams = set(games["team_a"].unique()) | set(games["team_b"].unique())
        print(f"  Unique teams: {len(teams)}")

    # Win rate sanity check
    if "team_a_result" in games.columns:
        blue_wr = games["team_a_result"].mean()
        print(f"  Blue side WR: {blue_wr:.1%}")


def main():
    parser = argparse.ArgumentParser(description="LeaguePred data ingestion")
    parser.add_argument(
        "--local", type=str, default=None,
        help="Path to local CSV file or directory (skip download)"
    )
    parser.add_argument(
        "--years", type=int, nargs="+", default=[2024, 2025, 2026],
        help="Years to download (default: 2024 2025 2026)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if files exist"
    )
    parser.add_argument(
        "--config", type=str, default="config.yaml",
        help="Path to config.yaml"
    )
    args = parser.parse_args()

    print("=" * 60)
    print("LeaguePred Data Ingestion")
    print("=" * 60)

    # Step 1: Get raw data
    if args.local:
        csv_source = args.local
        print(f"\n[1/4] Using local data: {csv_source}")
    else:
        print(f"\n[1/4] Downloading Oracle's Elixir data ({args.years})...")
        files = download_years(args.years, force=args.force)
        if not files:
            print("  No files downloaded. Exiting.")
            sys.exit(1)
        csv_source = str(RAW_DIR)

    # Step 2: Validate raw data
    print("\n[2/4] Validating raw CSVs...")
    source_path = Path(csv_source)
    if source_path.is_dir():
        csvs = sorted(source_path.glob("*.csv"))
    else:
        csvs = [source_path]

    for csv in csvs:
        v = validate_raw(csv)
        status = "OK" if v["valid"] else f"MISSING: {v['missing_required']}"
        print(f"  {v['file']}: {v['columns']} cols — {status}")

    # Step 3: Run pipeline
    print(f"\n[3/4] Running data pipeline on {csv_source}...")
    result = run_pipeline(csv_source, args.config)

    validation = validate_pipeline_output(result)
    if validation["valid"]:
        print("  Pipeline validation: PASSED")
    else:
        print("  Pipeline validation: ISSUES FOUND")
        for issue in validation["issues"]:
            print(f"    - {issue}")

    print_summary(result)

    # Step 4: Save processed output
    print(f"\n[4/4] Saving processed data to {PROCESSED_DIR}/...")
    save_processed(result, PROCESSED_DIR)

    print("\n" + "=" * 60)
    print("Ingestion complete.")
    print("=" * 60)

    return result


if __name__ == "__main__":
    main()
