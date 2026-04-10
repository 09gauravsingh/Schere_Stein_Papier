from __future__ import annotations

import pandas as pd


def aggregate_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate container-level rows to hourly terminal-level features.
    Assumes EventTime is already parsed to datetime and AvPowerCons is numeric.
    """
    required = ["EventTime", "AvPowerCons"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    df = df.copy()
    df["power_kw"] = df["AvPowerCons"] / 1000.0

    hourly = (
        df.groupby("EventTime", as_index=False)
        .agg(
            total_power_kw=("power_kw", "sum"),
            reefer_count=("container_visit_uuid", "nunique") if "container_visit_uuid" in df.columns else ("power_kw", "size"),
            ambient_avg=("TemperatureAmbient", "mean") if "TemperatureAmbient" in df.columns else ("power_kw", "mean"),
            setpoint_avg=("TemperatureSetPoint", "mean") if "TemperatureSetPoint" in df.columns else ("power_kw", "mean"),
            wear_avg=("TtlEnergyCons", "mean") if "TtlEnergyCons" in df.columns else ("power_kw", "mean"),
        )
        .sort_values("EventTime")
    )

    return hourly


def add_mix_features(
    df: pd.DataFrame,
    top_n: int = 5,
    top_hw_types: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute hourly mix features (tier/size/hardware shares) if source columns exist.
    """
    base = df.copy()
    base["EventTime"] = pd.to_datetime(base["EventTime"], utc=True).dt.floor("h").dt.tz_localize(None)

    def _share(group: pd.DataFrame, col: str, label: str) -> pd.Series:
        return group[col].value_counts(normalize=True).rename_axis(label).reset_index(name="share")

    def _slug(text: str) -> str:
        cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(text))
        while "__" in cleaned:
            cleaned = cleaned.replace("__", "_")
        return cleaned.strip("_")

    out = base[["EventTime"]].drop_duplicates().sort_values("EventTime")

    if "stack_tier" in base.columns:
        tier = base.groupby("EventTime").apply(lambda g: _share(g, "stack_tier", "stack_tier")).reset_index(level=0)
        tier_pivot = tier.pivot_table(index="EventTime", columns="stack_tier", values="share", fill_value=0)
        tier_pivot = tier_pivot.add_prefix("tier").add_suffix("_share").reset_index()
        out = out.merge(tier_pivot, on="EventTime", how="left")

    if "ContainerSize" in base.columns:
        size = base.groupby("EventTime").apply(lambda g: _share(g, "ContainerSize", "ContainerSize")).reset_index(level=0)
        size_pivot = size.pivot_table(index="EventTime", columns="ContainerSize", values="share", fill_value=0)
        size_pivot = size_pivot.add_prefix("size_").add_suffix("_share").reset_index()
        out = out.merge(size_pivot, on="EventTime", how="left")

    if "HardwareType" in base.columns:
        if top_hw_types is None:
            top_hw_types = (
                base["HardwareType"]
                .astype(str)
                .value_counts(dropna=True)
                .head(top_n)
                .index.tolist()
            )
        hw = (
            base.groupby("EventTime")
            .apply(lambda g: _share(g, "HardwareType", "HardwareType"))
            .reset_index(level=0)
        )
        top_cols: list[str] = []
        for hw_type in top_hw_types:
            col = f"hw_{_slug(hw_type)}_share"
            top_cols.append(col)
            series = hw[hw["HardwareType"].astype(str) == str(hw_type)][["EventTime", "share"]]
            series = series.rename(columns={"share": col})
            out = out.merge(series, on="EventTime", how="left")
        if top_cols:
            out["hw_other_share"] = 1.0 - out[top_cols].sum(axis=1, skipna=True)
        else:
            out["hw_other_share"] = 1.0

    return out
