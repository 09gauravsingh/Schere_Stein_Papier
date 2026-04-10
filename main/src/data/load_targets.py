from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_targets(csv_path: Path) -> pd.DataFrame:
    """
    Load target_timestamps.csv and return a dataframe with timestamp_utc.
    """
    df = pd.read_csv(csv_path, parse_dates=["timestamp_utc"])
    if "timestamp_utc" not in df.columns:
        raise ValueError("target_timestamps.csv must contain 'timestamp_utc'")
    return df.sort_values("timestamp_utc")
