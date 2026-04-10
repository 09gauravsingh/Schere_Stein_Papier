from __future__ import annotations

import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame, time_col: str = "EventTime") -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out[time_col].dt.hour
    out["dow"] = out[time_col].dt.dayofweek
    out["month"] = out[time_col].dt.month
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    return out


def add_lag_features(df: pd.DataFrame, target_col: str, lags: list[int]) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        out[f"{target_col}_lag{lag}"] = out[target_col].shift(lag)
    return out


def add_rolling_features(df: pd.DataFrame, target_col: str, windows: list[int]) -> pd.DataFrame:
    out = df.copy()
    for w in windows:
        out[f"{target_col}_roll{w}_mean"] = out[target_col].shift(1).rolling(w).mean()
        out[f"{target_col}_roll{w}_std"] = out[target_col].shift(1).rolling(w).std()
        out[f"{target_col}_roll{w}_max"] = out[target_col].shift(1).rolling(w).max()
    return out


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "effective_temp" in out.columns and "tier3_share" in out.columns:
        out["ambient_x_tier3_share"] = out["effective_temp"] * out["tier3_share"]
    if "effective_temp" in out.columns and "hour" in out.columns:
        out["ambient_x_hour"] = out["effective_temp"] * out["hour"]
    if "effective_temp" in out.columns and "hardware_highload_share" in out.columns:
        out["ambient_x_hw_high"] = out["effective_temp"] * out["hardware_highload_share"]
    return out
