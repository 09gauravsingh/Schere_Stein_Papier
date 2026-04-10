from __future__ import annotations

import pandas as pd


def hardware_panel(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby("HardwareType")["AvPowerCons"].agg(["count", "mean"]).reset_index()


def overall_panel(hourly_df: pd.DataFrame) -> pd.DataFrame:
    return hourly_df[["EventTime", "total_power_kw"]].rename(columns={"EventTime": "timestamp_utc"})


def weather_panel(df: pd.DataFrame, ambient_col: str, wind_col: str) -> pd.DataFrame:
    return df[[ambient_col, wind_col, "AvPowerCons"]].copy()


def tier_size_panel(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["stack_tier", "ContainerSize"])["AvPowerCons"].agg(["count", "mean"]).reset_index()
