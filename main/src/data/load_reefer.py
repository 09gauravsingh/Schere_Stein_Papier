from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_NUMERIC_COLS = [
    "AvPowerCons",
    "TtlEnergyConsHour",
    "TtlEnergyCons",
    "TemperatureSetPoint",
    "TemperatureAmbient",
    "TemperatureReturn",
    "RemperatureSupply",
]


def _coerce_decimal(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def load_reefer(
    csv_path: Path,
    usecols: Iterable[str] | None = None,
    numeric_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """
    Load reefer_release.csv with semicolon delimiter and decimal-comma normalization.
    """
    df = pd.read_csv(
        csv_path,
        sep=";",
        usecols=list(usecols) if usecols else None,
        parse_dates=["EventTime"] if (usecols is None or "EventTime" in usecols) else None,
        low_memory=False,
    )

    numeric_cols = list(numeric_cols) if numeric_cols else DEFAULT_NUMERIC_COLS
    for col in numeric_cols:
        if col in df.columns:
            df[col] = _coerce_decimal(df[col])

    return df
