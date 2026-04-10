from __future__ import annotations

import numpy as np
import pandas as pd


def age_wear_block(df: pd.DataFrame, p90: float = 0.9) -> pd.DataFrame:
    """
    Build numeric output for wear/age brackets based on TtlEnergyCons quintiles.
    """
    if "TtlEnergyCons" not in df.columns or "AvPowerCons" not in df.columns:
        raise ValueError("TtlEnergyCons and AvPowerCons are required for wear analysis")
    out = df.copy().dropna(subset=["TtlEnergyCons", "AvPowerCons"])
    out["wear_quintile"] = pd.qcut(out["TtlEnergyCons"], 5, labels=["Q1", "Q2", "Q3", "Q4", "Q5"])
    grouped = (
        out.groupby("wear_quintile")["AvPowerCons"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            p90=lambda s: s.quantile(p90),
        )
        .reset_index()
    )
    q1_mean = grouped.loc[grouped["wear_quintile"] == "Q1", "mean"].iloc[0]
    grouped["delta_vs_q1"] = grouped["mean"] - q1_mean
    grouped["multiplier_vs_q1"] = grouped["mean"] / q1_mean
    return grouped


def hardware_block(df: pd.DataFrame, p90: float = 0.9) -> pd.DataFrame:
    if "HardwareType" not in df.columns or "AvPowerCons" not in df.columns:
        raise ValueError("HardwareType and AvPowerCons are required")
    grouped = (
        df.dropna(subset=["HardwareType", "AvPowerCons"])
        .groupby("HardwareType")["AvPowerCons"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            p90=lambda s: s.quantile(p90),
        )
        .reset_index()
    )
    grouped["share"] = grouped["count"] / grouped["count"].sum()
    baseline_mean = grouped["mean"].min()
    grouped["multiplier_vs_baseline"] = grouped["mean"] / baseline_mean
    return grouped.sort_values("mean", ascending=False)


def ambient_block(df: pd.DataFrame, bins: list[float], labels: list[str], p90: float = 0.9) -> pd.DataFrame:
    if "TemperatureAmbient" not in df.columns or "AvPowerCons" not in df.columns:
        raise ValueError("TemperatureAmbient and AvPowerCons are required")
    out = df.copy().dropna(subset=["TemperatureAmbient", "AvPowerCons"])
    out["ambient_band"] = pd.cut(out["TemperatureAmbient"], bins=bins, labels=labels, include_lowest=True)
    grouped = (
        out.groupby("ambient_band")["AvPowerCons"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            p90=lambda s: s.quantile(p90),
        )
        .reset_index()
    )
    # slope estimate per band (simple linear fit on raw points)
    slopes = []
    for band in labels:
        band_df = out[out["ambient_band"] == band]
        if len(band_df) < 2:
            slopes.append(np.nan)
            continue
        x = band_df["TemperatureAmbient"].values
        y = band_df["AvPowerCons"].values
        coef = np.polyfit(x, y, 1)[0]
        slopes.append(coef)
    grouped["slope_w_per_c"] = slopes
    return grouped


def weather_block(
    df: pd.DataFrame,
    temp_col: str,
    wind_col: str,
    bins: list[float],
    labels: list[str],
    p90: float = 0.9,
    neutral_label: str | None = None,
) -> pd.DataFrame:
    if temp_col not in df.columns or wind_col not in df.columns or "AvPowerCons" not in df.columns:
        raise ValueError("Weather columns and AvPowerCons are required")
    out = df.copy().dropna(subset=[temp_col, wind_col, "AvPowerCons"])
    out["weather_band"] = pd.cut(out[temp_col], bins=bins, labels=labels, include_lowest=True)
    grouped = (
        out.groupby("weather_band", observed=True)["AvPowerCons"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            p90=lambda s: s.quantile(p90),
        )
        .reset_index()
    )
    wind_by_band = out.groupby("weather_band", observed=True)[wind_col].mean().reset_index()
    wind_by_band = wind_by_band.rename(columns={wind_col: "wind_mean"})
    grouped = grouped.merge(wind_by_band, on="weather_band", how="left")

    # Drop bands with no rows (avoids empty CSV lines)
    grouped = grouped.dropna(subset=["count"])
    grouped = grouped[grouped["count"] > 0]

    neutral_label = neutral_label or labels[len(labels) // 2]
    neutral_rows = grouped[grouped["weather_band"] == neutral_label]
    if not neutral_rows.empty:
        neutral_mean = float(neutral_rows["mean"].iloc[0])
    else:
        neutral_mean = float(grouped["mean"].median()) if len(grouped) else float("nan")
    grouped["delta_vs_neutral"] = grouped["mean"] - neutral_mean
    return grouped


def tier_size_block(df: pd.DataFrame, p90: float = 0.9) -> pd.DataFrame:
    if "stack_tier" not in df.columns or "ContainerSize" not in df.columns or "AvPowerCons" not in df.columns:
        raise ValueError("stack_tier, ContainerSize, and AvPowerCons are required")
    grouped = (
        df.dropna(subset=["stack_tier", "ContainerSize", "AvPowerCons"])
        .groupby(["stack_tier", "ContainerSize"])["AvPowerCons"]
        .agg(
            count="count",
            mean="mean",
            median="median",
            p90=lambda s: s.quantile(p90),
        )
        .reset_index()
    )
    return grouped
