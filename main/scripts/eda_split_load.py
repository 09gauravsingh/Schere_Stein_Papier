"""
Compare mean aggregate load and reefer count across calendar regions (structural shift EDA).

Uses the same hourly aggregation as modeling (no sequence-length filter).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from main.dl.config import (
    CAL_TEST_DAY_FIRST,
    CAL_TEST_DAY_LAST,
    CAL_TEST_MONTH,
    CAL_TEST_YEAR,
    CAL_TRAIN_LAST_MONTH,
    CAL_TRAIN_YEAR,
    CAL_VAL_DAY_FIRST,
    CAL_VAL_DAY_LAST,
    CAL_VAL_MONTH,
    CAL_VAL_YEAR,
    MULTI_VAL_LATE_DEC_TRAIN_FROM_DAY,
    VAL_WINDOWS,
)
from main.src.data.load_reefer import load_reefer
from main.src.preprocess.aggregation import aggregate_hourly


def _floor_hour_utc_naive(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.floor("h").dt.tz_localize(None)


def _print_block(name: str, sub: pd.DataFrame) -> None:
    if sub.empty:
        print(f"{name}: (empty)")
        return
    rc = "reefer_count" if "reefer_count" in sub.columns else None
    mean_kw = float(sub["total_power_kw"].mean())
    mean_r = float(sub[rc].mean()) if rc else float("nan")
    print(f"{name}: hours={len(sub):,}  mean_total_power_kw={mean_kw:.3f}  mean_reefer_count={mean_r:.3f}")


def main() -> None:
    default_root = Path(__file__).resolve().parents[2]
    ap = argparse.ArgumentParser(description="EDA: mean kW and reefer count by train/val/test regions.")
    ap.add_argument("--repo-root", type=Path, default=default_root)
    ap.add_argument("--reefer", type=Path, default=None)
    ns = ap.parse_args()
    root = ns.repo_root
    path = ns.reefer or (root / "participant_package/reefer_release.csv")

    df = load_reefer(path)
    base = df.copy()
    base["EventTime"] = _floor_hour_utc_naive(base["EventTime"])
    hourly = aggregate_hourly(base).rename(columns={"EventTime": "timestamp_utc"})
    ts = pd.to_datetime(hourly["timestamp_utc"])

    train_m = (ts.dt.year == CAL_TRAIN_YEAR) & (ts.dt.month <= CAL_TRAIN_LAST_MONTH)
    val_m = (
        (ts.dt.year == CAL_VAL_YEAR)
        & (ts.dt.month == CAL_VAL_MONTH)
        & (ts.dt.day >= int(CAL_VAL_DAY_FIRST))
        & (ts.dt.day <= int(CAL_VAL_DAY_LAST))
    )
    test_m = (
        (ts.dt.year == CAL_TEST_YEAR)
        & (ts.dt.month == CAL_TEST_MONTH)
        & (ts.dt.day >= int(CAL_TEST_DAY_FIRST))
        & (ts.dt.day <= int(CAL_TEST_DAY_LAST))
    )

    print("--- DL regions (config CAL_*; matches strategy=december layout) ---")
    _print_block(f"Train {CAL_TRAIN_YEAR}-01..{CAL_TRAIN_YEAR}-{CAL_TRAIN_LAST_MONTH:02d}", hourly.loc[train_m])
    _print_block(
        f"Val   {CAL_VAL_YEAR}-{CAL_VAL_MONTH:02d}-{CAL_VAL_DAY_FIRST:02d}..{CAL_VAL_DAY_LAST:02d}",
        hourly.loc[val_m],
    )
    _print_block(
        f"Test  {CAL_TEST_YEAR}-{CAL_TEST_MONTH:02d}-{CAL_TEST_DAY_FIRST:02d}..{CAL_TEST_DAY_LAST:02d}",
        hourly.loc[test_m],
    )

    print("\n--- Multi-window val slices (config VAL_WINDOWS) ---")
    for i, w in enumerate(VAL_WINDOWS):
        wm = (
            (ts.dt.year == int(w["year"]))
            & (ts.dt.month == int(w["month"]))
            & (ts.dt.day >= int(w["day_first"]))
            & (ts.dt.day <= int(w["day_last"]))
        )
        _print_block(f"Val window {i + 1} {w}", hourly.loc[wm])

    late = MULTI_VAL_LATE_DEC_TRAIN_FROM_DAY
    if late is not None:
        union_m = pd.Series(False, index=hourly.index)
        for w in VAL_WINDOWS:
            union_m |= (
                (ts.dt.year == int(w["year"]))
                & (ts.dt.month == int(w["month"]))
                & (ts.dt.day >= int(w["day_first"]))
                & (ts.dt.day <= int(w["day_last"]))
            )
        train_multi_m = train_m & (~union_m)
        train_multi_m |= (ts.dt.year == CAL_TRAIN_YEAR) & (ts.dt.month == 12) & (ts.dt.day >= int(late))
        print(f"\n--- multi_window train targets (excl. val windows, Dec>={late}) ---")
        _print_block("Train (multi_window layout)", hourly.loc[train_multi_m])


if __name__ == "__main__":
    main()
