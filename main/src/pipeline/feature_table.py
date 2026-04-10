from __future__ import annotations

import pandas as pd

from main.src.preprocess.aggregation import add_mix_features, aggregate_hourly
from main.src.preprocess.feature_builder import add_lag_features, add_rolling_features, add_time_features


def _normalize_hourly_ts(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.floor("h").dt.tz_localize(None)


def _weather_to_hourly_mean(weather: pd.DataFrame) -> pd.DataFrame:
    w = weather.copy()
    w["timestamp_utc"] = _normalize_hourly_ts(w["timestamp_utc"])
    numeric = [c for c in w.columns if c != "timestamp_utc" and pd.api.types.is_numeric_dtype(w[c])]
    return w.groupby("timestamp_utc", as_index=False)[numeric].mean()


def _add_lagged_columns(df: pd.DataFrame, cols: list[str], lags: list[int]) -> pd.DataFrame:
    for col in cols:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


def build_hourly_feature_table(df: pd.DataFrame, weather: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Build a model-ready hourly table from container-level reefer rows.
    Returns a dataframe with `timestamp_utc`, `total_power_kw`, and feature columns.
    """
    base = df.copy()
    base["EventTime"] = _normalize_hourly_ts(base["EventTime"])

    hourly = aggregate_hourly(base).copy()
    mix = add_mix_features(base, top_n=5)
    hourly = hourly.merge(mix, on="EventTime", how="left")
    hourly = hourly.rename(columns={"EventTime": "timestamp_utc"})
    hourly["timestamp_utc"] = _normalize_hourly_ts(hourly["timestamp_utc"])
    hourly = hourly.sort_values("timestamp_utc").drop_duplicates(subset=["timestamp_utc"], keep="last")

    if weather is not None:
        w = _weather_to_hourly_mean(weather)
        w = w.rename(columns={c: f"wx_{c}" for c in w.columns if c != "timestamp_utc"})
        hourly = hourly.merge(w, on="timestamp_utc", how="left")
        wx_cols = [c for c in hourly.columns if c.startswith("wx_") and not c.endswith("_missing")]
        for col in wx_cols:
            hourly[col] = pd.to_numeric(
                hourly[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
            missing_col = f"{col}_missing"
            hourly[missing_col] = hourly[col].isna().astype(float)
            fill_val = float(hourly[col].median(skipna=True)) if hourly[col].notna().any() else 0.0
            hourly[col] = hourly[col].fillna(fill_val)

    hourly = add_time_features(hourly, time_col="timestamp_utc")
    hourly = add_lag_features(hourly, target_col="total_power_kw", lags=[1, 2, 3, 6, 12, 24, 48, 168])
    hourly = add_rolling_features(hourly, target_col="total_power_kw", windows=[6, 12, 24, 72])

    # Terminal-level temperature / load context (literature: ambient temp drives reefer kW).
    # Keep raw ambient_avg / setpoint_avg: same-hour values come from the reefer mix at that hour
    # (known when predicting that hour's aggregate load — not the hidden target label).
    # Also add lags for inertia and short-horizon history.
    for col in ("ambient_avg", "setpoint_avg", "wear_avg", "reefer_count"):
        if col in hourly.columns:
            hourly = add_lag_features(hourly, col, lags=[1, 6, 24])

    mix_cols = [c for c in hourly.columns if c.startswith("tier") or c.startswith("size_") or c.startswith("hw_")]
    if mix_cols:
        hourly[mix_cols] = hourly[mix_cols].fillna(0.0)
        hourly = _add_lagged_columns(hourly, mix_cols, lags=[1, 24])
        hourly = hourly.drop(columns=mix_cols)

    wx_cols = [c for c in hourly.columns if c.startswith("wx_") and not c.endswith("_missing")]
    if wx_cols:
        hourly = _add_lagged_columns(hourly, wx_cols, lags=[1, 6, 24])
        hourly = hourly.drop(columns=wx_cols)

    hourly = hourly.sort_values("timestamp_utc").reset_index(drop=True)
    return hourly


def select_feature_columns(feature_table: pd.DataFrame) -> list[str]:
    excluded = {"timestamp_utc", "total_power_kw"}
    return [
        c
        for c in feature_table.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(feature_table[c])
    ]
