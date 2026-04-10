from __future__ import annotations

from pathlib import Path

import pandas as pd

from main.src.data.load_reefer import load_reefer
from main.src.data.load_targets import load_targets
from main.src.data.load_weather import load_weather_folder
from main.src.reporting.analysis_blocks import (
    age_wear_block,
    hardware_block,
    ambient_block,
    weather_block,
    tier_size_block,
)
from main.src.inference.submission_writer import write_submission
from main.src.pipeline.prediction_breakdown import enrich_prediction_breakdown


def _normalize_hourly_ts(series: pd.Series) -> pd.Series:
    """Match reefer hourly index to targets: UTC wall time, naive, floored to hour."""
    return pd.to_datetime(series, utc=True).dt.floor("h").dt.tz_localize(None)


def _weather_to_hourly_mean(weather: pd.DataFrame) -> pd.DataFrame:
    """Weather files have irregular timestamps; aggregate to hourly means for merge."""
    w = weather.copy()
    w["timestamp_utc"] = _normalize_hourly_ts(w["timestamp_utc"])
    numeric = [c for c in w.columns if c != "timestamp_utc"]
    return w.groupby("timestamp_utc", as_index=False)[numeric].mean()


def _select_weather_columns(weather_df: pd.DataFrame) -> tuple[str, str]:
    temp_candidates = [c for c in weather_df.columns if "temperatur" in c.lower() or "temp" in c.lower()]
    wind_candidates = [c for c in weather_df.columns if "wind" in c.lower() and "richtung" not in c.lower()]
    if not temp_candidates or not wind_candidates:
        raise ValueError("Could not detect temperature and wind columns in weather data")
    return temp_candidates[0], wind_candidates[0]


def _baseline_predict(hourly: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    time_col = "timestamp_utc" if "timestamp_utc" in hourly.columns else "EventTime"
    h = hourly.copy()
    h[time_col] = _normalize_hourly_ts(h[time_col])
    h = h.groupby(time_col, as_index=False)["total_power_kw"].mean()
    h = h.sort_values(time_col)
    hourly_index = h.set_index(time_col)["total_power_kw"]

    tgt_ts = _normalize_hourly_ts(targets["timestamp_utc"])

    preds = []
    for ts in tgt_ts:
        prev = ts - pd.Timedelta(hours=24)
        if prev in hourly_index.index:
            val = hourly_index.loc[prev]
            if isinstance(val, pd.Series):
                val = float(val.iloc[-1])
            else:
                val = float(val)
        else:
            # nearest past hour with data (still 24h-ahead safe: use history only)
            past = hourly_index[hourly_index.index < ts]
            val = float(past.iloc[-1]) if len(past) else float(hourly_index.iloc[-1])
        preds.append(val)

    pred_power = pd.Series(preds)
    pred_p90 = pred_power * 1.10
    out = pd.DataFrame(
        {
            "timestamp_utc": tgt_ts.values,
            "pred_power_kw": pred_power.values,
            "pred_p90_kw": pred_p90.values,
            "hardware_contribution": 0.0,
            "ambient_weather_adj": 0.0,
            "tier_size_adj": 0.0,
        }
    )
    return out


def run_all(
    reefer_path: Path,
    weather_folder: Path,
    targets_path: Path,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_reefer(reefer_path)

    # Analysis blocks
    wear_tbl = age_wear_block(df)
    hw_tbl = hardware_block(df)
    ambient_bins = [-50, 0, 10, 20, 30, 50]
    ambient_labels = ["<0", "0-10", "10-20", "20-30", "30+"]
    ambient_tbl = ambient_block(df, bins=ambient_bins, labels=ambient_labels)
    tier_size_tbl = tier_size_block(df)

    wear_tbl.to_csv(output_dir / "analysis_age_wear.csv", index=False)
    hw_tbl.to_csv(output_dir / "analysis_hardware.csv", index=False)
    ambient_tbl.to_csv(output_dir / "analysis_ambient.csv", index=False)
    tier_size_tbl.to_csv(output_dir / "analysis_tier_size.csv", index=False)

    # Weather block (3-month window) — must align to hourly buckets
    weather = _weather_to_hourly_mean(load_weather_folder(weather_folder))
    temp_col, wind_col = _select_weather_columns(weather)
    hourly = (
        df.groupby("EventTime", as_index=False)
        .agg(total_power_w=("AvPowerCons", "sum"))
        .rename(columns={"EventTime": "timestamp_utc"})
    )
    hourly["timestamp_utc"] = _normalize_hourly_ts(hourly["timestamp_utc"])
    hourly = hourly.groupby("timestamp_utc", as_index=False)["total_power_w"].sum()
    weather_merged = hourly.merge(weather, on="timestamp_utc", how="inner")
    # Terminal total load in kW (same units as pred_power_kw); was W and blew up means.
    weather_merged["AvPowerCons"] = weather_merged["total_power_w"] / 1000.0
    weather_tbl = weather_block(
        weather_merged,
        temp_col=temp_col,
        wind_col=wind_col,
        bins=ambient_bins,
        labels=ambient_labels,
    )
    weather_tbl.to_csv(output_dir / "analysis_weather_3mo.csv", index=False)

    # Overall 24h prediction and submission
    targets = load_targets(targets_path)
    hourly_full = (
        df.assign(_et=_normalize_hourly_ts(df["EventTime"]))
        .groupby("_et", as_index=False)
        .agg(total_power_kw=("AvPowerCons", lambda s: s.sum() / 1000.0))
        .rename(columns={"_et": "timestamp_utc"})
    )
    pred_24h = _baseline_predict(hourly_full, targets)
    global_amb = float(df["TemperatureAmbient"].median())
    pred_24h = enrich_prediction_breakdown(df, pred_24h, hw_tbl, global_amb)
    pred_24h.to_csv(output_dir / "prediction_24h.csv", index=False)
    write_submission(pred_24h, output_dir / "predictions.csv")


if __name__ == "__main__":
    repo_root = Path("/Users/omkarsomeshwarkondhalkar/Movies/project/eurogate")
    run_all(
        reefer_path=repo_root / "participant_package/reefer_release.csv",
        weather_folder=repo_root / "participant_package/Wetterdaten Okt 25 - 23 Feb 26",
        targets_path=repo_root / "participant_package/target_timestamps.csv",
        output_dir=repo_root / "main/outputs",
    )
