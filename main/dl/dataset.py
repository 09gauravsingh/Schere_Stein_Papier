from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def pick_sequence_columns(df: pd.DataFrame) -> list[str]:
    """All numeric columns except time and target (aligns with tree feature richness)."""
    exclude = {"timestamp_utc", "total_power_kw"}
    cols = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    if not cols:
        raise ValueError("No usable numeric columns found for DL sequence model.")
    return cols


def compute_split_bounds(n_rows: int, train_frac: float, val_frac: float) -> tuple[int, int]:
    train_end = int(n_rows * train_frac)
    val_end = int(n_rows * (train_frac + val_frac))
    return train_end, val_end


def _seq_ok(idxs: list[int], seq_len: int) -> list[int]:
    return [i for i in idxs if i >= seq_len]


def assert_holdout_months_excluded_from_training(
    timestamps: pd.Series,
    train_idx: list[int],
    seq_len: int,
    fit_row_mask: np.ndarray,
    *,
    holdout_year_months: set[tuple[int, int]],
) -> None:
    """
    Fail fast if training uses any row in the given (year, month) calendar months
    as a label, inside a training input window, or in scaler/y-stats fit.

    For the default split, pass {(2026, 1)} so all of January 2026 (val + test days)
    stays out of training — nothing from that month may enter the model during fit.
    """
    if not holdout_year_months:
        return
    ts = pd.to_datetime(timestamps, utc=False).reset_index(drop=True)
    in_holdout = pd.Series(False, index=ts.index, dtype=bool)
    for y, m in holdout_year_months:
        in_holdout |= (ts.dt.year == int(y)) & (ts.dt.month == int(m))
    in_holdout_np = in_holdout.to_numpy(dtype=bool)

    fit_row_mask = np.asarray(fit_row_mask, dtype=bool)
    if fit_row_mask.shape[0] != len(ts):
        raise ValueError("fit_row_mask length must match timestamps.")
    if np.any(in_holdout_np & fit_row_mask):
        raise ValueError(
            f"Scaler/y-stats fit includes holdout month(s) {sorted(holdout_year_months)} — "
            "those months must be val/test only."
        )

    for t in train_idx:
        if t < seq_len:
            continue
        if bool(in_holdout_np[t]):
            raise ValueError(
                f"Train target index {t} ({ts.iloc[t]}) falls in holdout month(s) "
                f"{sorted(holdout_year_months)}."
            )
        if np.any(in_holdout_np[t - seq_len : t]):
            raise ValueError(
                f"Train target index {t} ({ts.iloc[t]}) sequence includes a row in "
                f"holdout month(s) {sorted(holdout_year_months)}."
            )


def _mask_val_windows(ts: pd.Series, windows: list[dict]) -> pd.Series:
    m = pd.Series(False, index=ts.index)
    for w in windows:
        y = int(w["year"])
        mo = int(w["month"])
        d0 = int(w["day_first"])
        d1 = int(w["day_last"])
        m |= (ts.dt.year == y) & (ts.dt.month == mo) & (ts.dt.day >= d0) & (ts.dt.day <= d1)
    return m


def calendar_split_dl_indices(
    timestamps: pd.Series,
    *,
    strategy: Literal["december", "multi_window"],
    train_year: int,
    train_last_month: int,
    val_year: int,
    val_month: int,
    val_day_first: int,
    val_day_last: int,
    val_windows: list[dict] | None,
    late_dec_train_from_day: int | None,
    test_year: int,
    test_month: int,
    test_day_first: int,
    test_day_last: int,
    seq_len: int,
) -> tuple[list[int], list[list[int]], list[int], pd.Series]:
    """
    Row indices for DL targets. Returns:
      train_idx, val_idx_windows (one list per val window), test_idx, timestamps.

    december: train = Jan..train_last_month; val = val_year/month/days; test = test_year/month/days.
    multi_window: train = Jan..train_last_month minus val slice hours, plus Dec late_dec_train_from_day..31;
                  val = each window in val_windows; test = test_year/month/days (same as december branch).
    """
    ts = pd.to_datetime(timestamps)
    test_m = (
        (ts.dt.year == test_year)
        & (ts.dt.month == test_month)
        & (ts.dt.day >= int(test_day_first))
        & (ts.dt.day <= int(test_day_last))
    )
    test_idx = _seq_ok(np.flatnonzero(test_m.to_numpy()).tolist(), seq_len)

    if strategy == "december":
        train_m = (ts.dt.year == train_year) & (ts.dt.month <= train_last_month)
        val_m = (
            (ts.dt.year == val_year)
            & (ts.dt.month == val_month)
            & (ts.dt.day >= int(val_day_first))
            & (ts.dt.day <= int(val_day_last))
        )
        train_idx = _seq_ok(np.flatnonzero(train_m.to_numpy()).tolist(), seq_len)
        val_one = _seq_ok(np.flatnonzero(val_m.to_numpy()).tolist(), seq_len)
        return train_idx, [val_one], test_idx, ts

    if strategy != "multi_window":
        raise ValueError(f"Unknown val strategy: {strategy!r}")

    windows = list(val_windows or [])
    if not windows:
        raise ValueError("multi_window requires non-empty val_windows.")

    val_union_m = _mask_val_windows(ts, windows)
    train_m = (ts.dt.year == train_year) & (ts.dt.month <= train_last_month) & (~val_union_m)
    if late_dec_train_from_day is not None:
        train_m |= (
            (ts.dt.year == train_year)
            & (ts.dt.month == 12)
            & (ts.dt.day >= int(late_dec_train_from_day))
        )
    train_idx = _seq_ok(np.flatnonzero(train_m.to_numpy()).tolist(), seq_len)

    val_idx_windows: list[list[int]] = []
    for w in windows:
        wm = (
            (ts.dt.year == int(w["year"]))
            & (ts.dt.month == int(w["month"]))
            & (ts.dt.day >= int(w["day_first"]))
            & (ts.dt.day <= int(w["day_last"]))
        )
        val_idx_windows.append(_seq_ok(np.flatnonzero(wm.to_numpy()).tolist(), seq_len))

    return train_idx, val_idx_windows, test_idx, ts


def calendar_split_target_indices(
    timestamps: pd.Series,
    *,
    train_year: int,
    train_last_month: int,
    val_year: int,
    val_month: int,
    val_day_first: int = 1,
    val_day_last: int = 31,
    test_year: int,
    test_month: int,
    test_day_first: int = 1,
    test_day_last: int = 31,
    seq_len: int,
) -> tuple[list[int], list[int], list[int], pd.Series]:
    """
    Return row indices (into the sorted hourly frame) for prediction targets.
    Train: train_year, months 1..train_last_month
    Val: val_year, val_month, val_day_first..val_day_last (inclusive)
    Test: test_year, test_month, test_day_first..test_day_last (inclusive)
    Only indices i >= seq_len are kept (enough history in the continuous table).
    """
    train_idx, val_w, test_idx, ts = calendar_split_dl_indices(
        timestamps,
        strategy="december",
        train_year=train_year,
        train_last_month=train_last_month,
        val_year=val_year,
        val_month=val_month,
        val_day_first=val_day_first,
        val_day_last=val_day_last,
        val_windows=None,
        late_dec_train_from_day=None,
        test_year=test_year,
        test_month=test_month,
        test_day_first=test_day_first,
        test_day_last=test_day_last,
        seq_len=seq_len,
    )
    return train_idx, val_w[0], test_idx, ts


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, x_2d: np.ndarray) -> "Standardizer":
        mean = x_2d.mean(axis=0)
        std = x_2d.std(axis=0)
        std = np.where(std < 1e-8, 1.0, std)
        return cls(mean=mean, std=std)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std


class SequenceDataset(Dataset):
    def __init__(
        self,
        x_2d: np.ndarray,
        y: np.ndarray,
        seq_len: int,
        target_indices: list[int] | None = None,
        start_idx: int = 0,
        end_idx: int = 0,
    ) -> None:
        self.x = x_2d
        self.y = y
        self.seq_len = seq_len
        if target_indices is not None:
            self.indices = [i for i in target_indices if i >= seq_len]
        else:
            self.indices = [i for i in range(max(start_idx, seq_len), end_idx)]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        t = self.indices[idx]
        x_seq = self.x[t - self.seq_len : t, :]
        y_t = self.y[t]
        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_t, dtype=torch.float32)
