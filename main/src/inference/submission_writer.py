from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_submission(df: pd.DataFrame, out_path: Path) -> None:
    required = ["timestamp_utc", "pred_power_kw", "pred_p90_kw"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    if (df["pred_power_kw"] < 0).any() or (df["pred_p90_kw"] < 0).any():
        raise ValueError("Predictions must be non-negative")

    if (df["pred_p90_kw"] < df["pred_power_kw"]).any():
        raise ValueError("pred_p90_kw must be >= pred_power_kw")

    df[required].to_csv(out_path, index=False)
