from __future__ import annotations

from pathlib import Path

import pandas as pd


def _read_weather_file(path: Path) -> pd.DataFrame | None:
    # German exports often use semicolon; try that first for robustness.
    try:
        df = pd.read_csv(path, sep=";", low_memory=False)
    except Exception:
        df = pd.read_csv(path, sep=None, engine="python")
    time_cols = [c for c in df.columns if "time" in c.lower() or "datum" in c.lower() or "date" in c.lower()]
    value_cols = [c for c in df.columns if c not in time_cols]
    if not time_cols or not value_cols:
        return None
    tcol = time_cols[0]
    vcol = value_cols[-1]
    out = df[[tcol, vcol]].copy()
    out.columns = ["timestamp_utc", path.stem]
    out["timestamp_utc"] = pd.to_datetime(out["timestamp_utc"], errors="coerce", utc=True).dt.tz_convert(None)
    out[path.stem] = pd.to_numeric(out[path.stem].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    return out.dropna(subset=["timestamp_utc"])


def load_weather_folder(folder: Path) -> pd.DataFrame:
    """
    Load and merge weather CSVs from a folder into a single time-indexed dataframe.
    """
    frames = []
    for path in sorted(folder.glob("*.csv")):
        df = _read_weather_file(path)
        if df is not None:
            frames.append(df)

    if not frames:
        raise ValueError(f"No weather files found in {folder}")

    merged = frames[0]
    for df in frames[1:]:
        merged = merged.merge(df, on="timestamp_utc", how="outer")

    return merged.sort_values("timestamp_utc")
