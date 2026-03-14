"""
Time-Based Validation Splits
-----------------------------
NEVER use random train/test splits for time-series prediction.
Always train on past, test on future.

This module provides:
    - Simple temporal train/test split
    - Walk-forward validation (expanding window)
    - Split-based validation (train on spring, test on summer, etc.)

Leakage note:
    Any feature engineering must be fit ONLY on training data.
    Even aggregates like "team average gold at 15" must be computed
    using only games before the prediction date.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional


def temporal_split(
    df: pd.DataFrame,
    test_fraction: float = 0.2,
    date_col: str = "date",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple temporal train/test split.
    The earliest (1-test_fraction) of games are training, the rest are test.

    Args:
        df: DataFrame with a date column, sorted by date.
        test_fraction: Fraction of data to use for testing.
        date_col: Name of the date column.

    Returns:
        (train_df, test_df)
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_fraction))

    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()

    print(f"Temporal split: {len(train)} train, {len(test)} test")
    if date_col in df.columns:
        print(f"  Train: {train[date_col].min()} to {train[date_col].max()}")
        print(f"  Test:  {test[date_col].min()} to {test[date_col].max()}")

    return train, test


def walk_forward_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    min_train_size: int = 200,
    date_col: str = "date",
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Walk-forward (expanding window) cross-validation.

    Each split trains on all data up to a cutoff point,
    then tests on the next chunk. The training set grows with each split.

    This is the gold standard for time-series model evaluation.

    Args:
        df: DataFrame sorted by date.
        n_splits: Number of test windows.
        min_train_size: Minimum number of training games.
        date_col: Name of the date column.

    Returns:
        List of (train_df, test_df) tuples.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    n = len(df)

    if n < min_train_size + n_splits:
        raise ValueError(
            f"Not enough data for {n_splits} splits with min_train_size={min_train_size}. "
            f"Have {n} games, need at least {min_train_size + n_splits}."
        )

    # Calculate test window size
    available = n - min_train_size
    test_size = available // n_splits

    splits = []
    for i in range(n_splits):
        train_end = min_train_size + i * test_size
        test_end = train_end + test_size

        # Last split gets any remaining data
        if i == n_splits - 1:
            test_end = n

        train = df.iloc[:train_end].copy()
        test = df.iloc[train_end:test_end].copy()
        splits.append((train, test))

    print(f"Walk-forward: {n_splits} splits, {test_size} games per test window")
    for i, (train, test) in enumerate(splits):
        print(f"  Split {i+1}: train={len(train)}, test={len(test)}")

    return splits


def split_by_period(
    df: pd.DataFrame,
    train_filter: dict,
    test_filter: dict,
    date_col: str = "date",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split by specific time periods or metadata.

    Example: train on 2023 Spring, test on 2023 Summer.

    Args:
        df: DataFrame with date and metadata columns.
        train_filter: Dict of {column: value} pairs for training data.
        test_filter: Dict of {column: value} pairs for test data.

    Returns:
        (train_df, test_df)
    """
    train_mask = pd.Series(True, index=df.index)
    for col, val in train_filter.items():
        if isinstance(val, list):
            train_mask &= df[col].isin(val)
        else:
            train_mask &= df[col] == val

    test_mask = pd.Series(True, index=df.index)
    for col, val in test_filter.items():
        if isinstance(val, list):
            test_mask &= df[col].isin(val)
        else:
            test_mask &= df[col] == val

    train = df[train_mask].sort_values(date_col).reset_index(drop=True)
    test = df[test_mask].sort_values(date_col).reset_index(drop=True)

    # Leakage check: ensure test data is strictly after training data
    if date_col in df.columns:
        train_max = train[date_col].max()
        test_min = test[date_col].min()
        if train_max >= test_min:
            print(
                f"WARNING: Training data ({train_max}) overlaps with "
                f"test data ({test_min}). This may cause leakage!"
            )

    print(f"Period split: {len(train)} train, {len(test)} test")
    return train, test
