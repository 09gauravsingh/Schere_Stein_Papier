from __future__ import annotations

from typing import Iterator

import pandas as pd


def time_split(df: pd.DataFrame, time_col: str, train_end: pd.Timestamp, valid_end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = df[df[time_col] <= train_end]
    valid = df[(df[time_col] > train_end) & (df[time_col] <= valid_end)]
    test = df[df[time_col] > valid_end]
    return train, valid, test


def rolling_backtests(df: pd.DataFrame, time_col: str, window_size: int, horizon: int) -> Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Yield rolling train/validation splits by index window, assuming df sorted by time_col.
    window_size: number of rows in training window
    horizon: number of rows in validation window
    """
    df = df.sort_values(time_col).reset_index(drop=True)
    for start in range(0, len(df) - window_size - horizon + 1, horizon):
        train = df.iloc[start : start + window_size]
        valid = df.iloc[start + window_size : start + window_size + horizon]
        yield train, valid
