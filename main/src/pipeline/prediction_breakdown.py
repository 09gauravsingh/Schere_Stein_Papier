from __future__ import annotations

import numpy as np
import pandas as pd


def _norm_ts(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).dt.floor("h").dt.tz_localize(None)


def enrich_prediction_breakdown(
    df: pd.DataFrame,
    pred_df: pd.DataFrame,
    hw_tbl: pd.DataFrame,
    global_ambient_median: float,
) -> pd.DataFrame:
    """
    Replace placeholder zeros with lag-hour (t-24h) interpretable splits.

    - hardware_contribution_kw: share of lag-hour terminal load (kW) from top-5 hardware by historical mean.
    - ambient_weather_adj_kw: small adjustment from lag-hour mean ambient vs global median (scaled to kW).
    - tier_size_adj_kw: lag-hour excess kW vs tier-1-only baseline using within-slice tier means.
    """
    out = pred_df.copy()
    df = df.copy()
    df["_et"] = _norm_ts(df["EventTime"])

    top_hw = set(hw_tbl.nlargest(5, "mean")["HardwareType"].astype(str))

    hw_kw: list[float] = []
    amb_kw: list[float] = []
    tier_kw: list[float] = []

    tgt_ts = _norm_ts(pred_df["timestamp_utc"])

    for i, ts in enumerate(tgt_ts):
        prev = ts - pd.Timedelta(hours=24)
        sl = df[df["_et"] == prev]
        pred_row = float(pred_df["pred_power_kw"].iloc[i])

        if sl.empty:
            hw_kw.append(0.0)
            amb_kw.append(0.0)
            tier_kw.append(0.0)
            continue

        total_w = sl["AvPowerCons"].sum()
        total_kw = total_w / 1000.0
        if total_w <= 0:
            hw_kw.append(0.0)
            amb_kw.append(0.0)
            tier_kw.append(0.0)
            continue

        hw_w = sl[sl["HardwareType"].astype(str).isin(top_hw)]["AvPowerCons"].sum()
        hw_kw.append(float(hw_w / 1000.0))

        amb = sl["TemperatureAmbient"].mean()
        if pd.isna(amb):
            amb = global_ambient_median
        amb_frac = float(np.tanh((amb - global_ambient_median) / 15.0) * 0.05)
        amb_kw.append(pred_row * amb_frac)

        tmeans = sl.dropna(subset=["stack_tier"]).groupby("stack_tier")["AvPowerCons"].mean()
        tier1_key = next((k for k in tmeans.index if float(k) == 1.0), None)
        if tier1_key is not None and len(tmeans) > 1:
            base = float(tmeans.loc[tier1_key])
            mix = 0.0
            for tier, mean_t in tmeans.items():
                share = (sl["stack_tier"] == tier).mean()
                mix += share * (float(mean_t) - base)
            tier_kw.append(float(mix / 1000.0 * (pred_row / max(total_kw, 1e-6))))
        else:
            tier_kw.append(0.0)

    out["hardware_contribution"] = hw_kw
    out["ambient_weather_adj"] = amb_kw
    out["tier_size_adj"] = tier_kw
    return out
