"""
Reefer Peak Load Challenge — prediction script
==============================================

Inputs (place in the same folder as this script):
  reefer_release.csv      — full reefer data (semicolon-separated, decimal comma)
  target_timestamps.csv   — target hours to predict (one per row, column: timestamp_utc)

Bundled (already in this folder):
  segment_list.csv                — hardware/size/tier segment definitions
  segment_model_summaries.csv    — pre-trained mixed-effects coefficients (2025)

Output:
  predictions.csv  — timestamp_utc, pred_power_kw, pred_p90_kw

Strategy
--------
pred_power_kw  = fleet total AvPowerCons at (t − 1 h) / 1000
                 rolling carry-forward of the last known hour

pred_p90_kw    = max(pred_power_kw, model fleet total at t)
                 where the model is a mixed-effects regression per segment
                 (intercept + ambient_temp + sin/cos 24-h cycle)
                 trained on 2025 reefer data, applied with n_containers and
                 mean TemperatureAmbient from reefer_release at t − 1 h
"""

from __future__ import annotations
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent

REEFER_CSV    = HERE / "input"  / "reefer_release.csv"
TARGET_CSV    = HERE / "input"  / "target_timestamps.csv"
SEGMENT_CSV   = HERE / "segment_list.csv"
SUMMARIES_CSV = HERE / "segment_model_summaries.csv"
OUTPUT_CSV    = HERE / "output" / "predictions.csv"


# ── data loading ──────────────────────────────────────────────────────────

def load_reefer() -> pd.DataFrame:
    print("Loading reefer data …")
    df = pd.read_csv(
        REEFER_CSV, sep=";", encoding="utf-8-sig", decimal=",",
        usecols=["container_uuid", "HardwareType", "EventTime",
                 "AvPowerCons", "TemperatureAmbient", "ContainerSize", "stack_tier"],
        low_memory=False,
    )
    df["EventTime"]          = pd.to_datetime(df["EventTime"], errors="coerce")
    df["AvPowerCons"]        = pd.to_numeric(df["AvPowerCons"],        errors="coerce")
    df["TemperatureAmbient"] = pd.to_numeric(df["TemperatureAmbient"], errors="coerce")
    df["ContainerSize"]      = pd.to_numeric(df["ContainerSize"],      errors="coerce").astype("Int64")
    df["stack_tier"]         = pd.to_numeric(df["stack_tier"],         errors="coerce").astype("Int64")
    df = df.dropna(subset=["EventTime", "AvPowerCons"])
    df = df[df["AvPowerCons"] >= 0]
    df["hour_ts"] = df["EventTime"].dt.floor("h")
    print(f"  {len(df):,} rows  {df['EventTime'].min()} → {df['EventTime'].max()}")
    return df


# ── segment filtering ─────────────────────────────────────────────────────

def filter_segment(df: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    mask = (df["HardwareType"] == row["hw"]) & (df["ContainerSize"] == int(row["sz"]))
    if row["split"]:
        mask = mask & (df["stack_tier"] == int(row["tier"]))
    return df[mask]


# ── model loading ─────────────────────────────────────────────────────────

def load_models() -> dict:
    """Load pre-trained fixed-effect coefficients from segment_model_summaries.csv."""
    summaries = pd.read_csv(SUMMARIES_CSV)
    models = {}
    for _, row in summaries.iterrows():
        label = row["label"]
        if row["status"] != "ok":
            models[label] = None
            continue
        models[label] = {
            "intercept":    row["intercept"],
            "beta_weather": row["beta_weather"],
            "beta_sin24":   row["beta_sin24"],
            "beta_cos24":   row["beta_cos24"],
        }
    trained = sum(1 for v in models.values() if v is not None)
    print(f"  Loaded {trained}/{len(models)} segment models.")
    return models


# ── per-container model prediction ───────────────────────────────────────

def model_predict_W(fe: dict, ambient_temp: float, hour: int) -> float:
    sin24 = np.sin(2 * np.pi * hour / 24)
    cos24 = np.cos(2 * np.pi * hour / 24)
    val   = (fe["intercept"]
             + fe["beta_weather"] * ambient_temp
             + fe["beta_sin24"]   * sin24
             + fe["beta_cos24"]   * cos24)
    return max(val, 0.0)


# ── main ──────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    reefer   = load_reefer()
    segments = pd.read_csv(SEGMENT_CSV)
    targets  = pd.read_csv(TARGET_CSV)
    targets["hour_ts"] = (pd.to_datetime(targets["timestamp_utc"], utc=True)
                          .dt.tz_localize(None))

    print("\nLoading pre-trained segment models …")
    models = load_models()

    # fleet hourly totals (W) and mean ambient temp — both from reefer, per hour
    fleet_hourly  = reefer.groupby("hour_ts")["AvPowerCons"].sum()
    ambient_hourly = reefer.groupby("hour_ts")["TemperatureAmbient"].mean()

    # global fallback temp in case t-1h has no readings
    global_mean_temp = reefer["TemperatureAmbient"].mean()

    print(f"\nPredicting {len(targets)} target hours …")
    rows = []
    for _, trow in targets.iterrows():
        ts      = trow["hour_ts"]
        prev_ts = ts - pd.Timedelta(hours=1)

        # pred_power_kw: fleet sum at t-1h carried forward
        fleet_prev_W  = fleet_hourly.get(prev_ts, np.nan)
        pred_power_kw = fleet_prev_W / 1000 if not np.isnan(fleet_prev_W) else 0.0

        # ambient temp at t-1h averaged across all containers
        temp      = ambient_hourly.get(prev_ts, global_mean_temp)
        prev_snap = reefer[reefer["hour_ts"] == prev_ts]

        # model fleet total: per-segment model × n_containers(t-1h)
        model_total_W = 0.0
        for _, seg_row in segments.iterrows():
            label    = seg_row["label"]
            seg_prev = filter_segment(prev_snap, seg_row)
            n_c      = seg_prev["container_uuid"].nunique()
            if n_c == 0:
                continue

            fe = models.get(label)
            if fe is not None:
                pred_W = model_predict_W(fe, temp, ts.hour)
            else:
                pred_W = seg_prev["AvPowerCons"].mean()

            model_total_W += pred_W * n_c

        pred_p90_kw = max(pred_power_kw, model_total_W / 1000)

        rows.append({
            "timestamp_utc": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "pred_power_kw": round(pred_power_kw, 4),
            "pred_p90_kw":   round(pred_p90_kw,   4),
        })

    # write output
    out = pd.DataFrame(rows)
    OUTPUT_CSV.parent.mkdir(exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(out)} rows → {OUTPUT_CSV}")

    # sanity checks
    assert len(out) == len(targets),                          "Row count mismatch"
    assert out.duplicated("timestamp_utc").sum() == 0,        "Duplicate timestamps"
    assert (out["pred_power_kw"] >= 0).all(),                 "Negative pred_power_kw"
    assert (out["pred_p90_kw"] >= out["pred_power_kw"]).all(), "p90 < mean pred"
    print("Checks passed.")
    print(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
