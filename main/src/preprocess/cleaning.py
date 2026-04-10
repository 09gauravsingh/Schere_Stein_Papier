from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


@dataclass
class ImputationConfig:
    numeric_cols: list[str]
    categorical_cols: list[str]
    numeric_fill: dict[str, float]


def build_imputation_config(df: pd.DataFrame, numeric_cols: Iterable[str], categorical_cols: Iterable[str]) -> ImputationConfig:
    numeric_cols = [c for c in numeric_cols if c in df.columns]
    categorical_cols = [c for c in categorical_cols if c in df.columns]
    numeric_fill = {c: float(df[c].median()) for c in numeric_cols}
    return ImputationConfig(numeric_cols=numeric_cols, categorical_cols=categorical_cols, numeric_fill=numeric_fill)


def apply_imputation(df: pd.DataFrame, config: ImputationConfig) -> pd.DataFrame:
    out = df.copy()
    for c in config.numeric_cols:
        out[f"{c}_missing"] = out[c].isna().astype(int)
        out[c] = out[c].fillna(config.numeric_fill[c])
    for c in config.categorical_cols:
        out[c] = out[c].fillna("UNKNOWN")
    return out


def drop_missing_target(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    if target_col not in df.columns:
        return df
    return df.dropna(subset=[target_col])
