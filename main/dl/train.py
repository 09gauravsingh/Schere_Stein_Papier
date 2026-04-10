from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None  # type: ignore[misc, assignment]

import main.dl.config as dl_cfg
from main.dl.config import (
    BATCH_SIZE,
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
    DEEP_HEAD,
    DROPOUT,
    EARLY_STOP_EMA_ALPHA,
    EARLY_STOP_MIN_DELTA,
    EARLY_STOP_PATIENCE,
    GRAD_CLIP_NORM,
    HEAD_DROPOUT,
    HIDDEN_SIZE,
    HUBER_BETA_PEAK,
    LR,
    LOSS_W_ALL,
    LOSS_W_PEAK,
    LOSS_W_PIN,
    LR_PLATEAU_FACTOR,
    LR_PLATEAU_MIN,
    LR_PLATEAU_PATIENCE,
    LSTM_NUM_LAYERS,
    MAX_EPOCHS,
    MULTI_VAL_LATE_DEC_TRAIN_FROM_DAY,
    P90_HEAD_LR_MULT,
    SEQ_LEN,
    SEED,
    TRAIN_EXCLUDED_YEAR_MONTHS,
    USE_ATTENTION,
    USE_SMOOTH_L1_PEAK,
    VAL_STRATEGY,
    VAL_WINDOWS,
    WEIGHT_DECAY,
)
from main.dl.dataset import (
    SequenceDataset,
    Standardizer,
    assert_holdout_months_excluded_from_training,
    calendar_split_dl_indices,
    pick_sequence_columns,
)
from main.dl.model import LSTMForecaster
from main.src.data.load_reefer import load_reefer
from main.src.data.load_weather import load_weather_folder
from main.src.eval.metrics import composite_score, mae, mae_peak, pinball_loss
from main.src.pipeline.feature_table import build_hourly_feature_table


def _pinball_loss_torch(y_true: torch.Tensor, y_pred: torch.Tensor, q: float = 0.9) -> torch.Tensor:
    diff = y_true - y_pred
    return torch.mean(torch.maximum(q * diff, (q - 1.0) * diff))


def collect_preds_kw(
    model_point: LSTMForecaster,
    model_p90: LSTMForecaster,
    loader: DataLoader,
    device: torch.device,
    y_mean: float,
    y_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All predictions in kW (denormalized)."""
    model_point.eval()
    model_p90.eval()
    ys: list[np.ndarray] = []
    pp: list[np.ndarray] = []
    qq: list[np.ndarray] = []
    with torch.no_grad():
        for x_seq, y_norm in loader:
            x_seq = x_seq.to(device)
            y_norm = y_norm.to(device)
            p_n = model_point(x_seq)
            q_n = model_p90(x_seq)
            q_n = torch.maximum(q_n, p_n)
            y_kw = (y_norm * y_std + y_mean).cpu().numpy()
            p_kw = (p_n * y_std + y_mean).cpu().numpy()
            q_kw = (q_n * y_std + y_mean).cpu().numpy()
            ys.append(y_kw)
            pp.append(p_kw)
            qq.append(q_kw)
    return (
        np.concatenate(ys, axis=0),
        np.concatenate(pp, axis=0),
        np.concatenate(qq, axis=0),
    )


def _bin_means(preds: np.ndarray, feature: pd.Series) -> list[float] | None:
    values = feature.to_numpy(dtype=float)
    if np.all(np.isnan(values)):
        return None
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return None
    q1, q2 = np.nanquantile(values, [0.33, 0.66])
    m1 = values <= q1
    m2 = (values > q1) & (values <= q2)
    m3 = values > q2
    means = [
        float(np.nanmean(preds[m1])) if m1.any() else float("nan"),
        float(np.nanmean(preds[m2])) if m2.any() else float("nan"),
        float(np.nanmean(preds[m3])) if m3.any() else float("nan"),
    ]
    return means


def _pick_wx_lag1_col(
    columns,
    *keywords: str,
    exclude: tuple[str, ...] = (),
) -> str | None:
    for c in columns:
        name = str(c)
        if not name.endswith("_lag1") or "missing" in name.lower():
            continue
        cl = name.lower()
        if any(ex in cl for ex in exclude):
            continue
        if any(k in cl for k in keywords):
            return name
    return None


def _sanitize_history_json(history: list[dict]) -> list[dict]:
    """Make training history JSON-serializable (no NaN/inf, no tuple values)."""
    import math

    def fix(v):  # noqa: ANN001
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        if isinstance(v, list):
            return [fix(x) for x in v]
        if isinstance(v, tuple):
            return [fix(x) for x in v]
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    return [fix(row) for row in history]


def _low_high_means(preds: np.ndarray, feature: pd.Series) -> tuple[float, float] | None:
    values = feature.to_numpy(dtype=float)
    if np.all(np.isnan(values)):
        return None
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return None
    q1, q3 = np.nanquantile(values, [0.25, 0.75])
    low = values <= q1
    high = values >= q3
    if not low.any() or not high.any():
        return None
    return float(np.nanmean(preds[low])), float(np.nanmean(preds[high]))


def _log_line(msg: str, use_tqdm_write: bool) -> None:
    if use_tqdm_write and tqdm is not None:
        tqdm.write(msg, file=sys.stderr)
    else:
        print(msg, file=sys.stderr)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cpu_state_dict(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in module.state_dict().items()}


def _build_checkpoint_bundle(
    *,
    model_point: LSTMForecaster,
    model_p90: LSTMForecaster,
    best_epoch_1based: int,
    best_val_composite_kw: float,
    seq_cols: list[str],
    n_features: int,
    scaler: Standardizer,
    y_mean: float,
    y_std: float,
    seed: int,
    val_strategy: str,
) -> dict:
    return {
        "point_state": _cpu_state_dict(model_point),
        "p90_state": _cpu_state_dict(model_p90),
        "best_epoch": int(best_epoch_1based),
        "best_val_composite_kw": float(best_val_composite_kw),
        "seq_cols": list(seq_cols),
        "n_features": int(n_features),
        "scaler_mean": np.asarray(scaler.mean, dtype=np.float64),
        "scaler_scale": np.asarray(scaler.std, dtype=np.float64),
        "y_mean": float(y_mean),
        "y_std": float(y_std),
        "seed": int(seed),
        "val_strategy": str(val_strategy),
        "config_snapshot": {
            "SEQ_LEN": dl_cfg.SEQ_LEN,
            "HIDDEN_SIZE": dl_cfg.HIDDEN_SIZE,
            "NUM_LAYERS": dl_cfg.LSTM_NUM_LAYERS,
            "DROPOUT": dl_cfg.DROPOUT,
            "USE_ATTENTION": dl_cfg.USE_ATTENTION,
            "DEEP_HEAD": dl_cfg.DEEP_HEAD,
            "BATCH_SIZE": dl_cfg.BATCH_SIZE,
            "LR": dl_cfg.LR,
            "LOSS_W_ALL": dl_cfg.LOSS_W_ALL,
            "LOSS_W_PEAK": dl_cfg.LOSS_W_PEAK,
            "LOSS_W_PIN": dl_cfg.LOSS_W_PIN,
            "USE_SMOOTH_L1_PEAK": dl_cfg.USE_SMOOTH_L1_PEAK,
        },
        "use_attention": bool(dl_cfg.USE_ATTENTION),
        "use_deep_head": bool(dl_cfg.DEEP_HEAD),
        "head_dropout": float(dl_cfg.HEAD_DROPOUT),
        "hidden_size": int(dl_cfg.HIDDEN_SIZE),
        "lstm_num_layers": int(dl_cfg.LSTM_NUM_LAYERS),
        "dropout": float(dl_cfg.DROPOUT),
    }


def run_train_dl(
    data_path: Path,
    out_dir: Path,
    *,
    max_epochs: int | None = None,
    early_stop_patience: int | None = None,
    early_stop_min_delta: float | None = None,
    show_progress: bool = True,
    seed: int | None = None,
    weather_folder: Path | None = None,
    val_strategy: str | None = None,
    val_windows: list[dict] | None = None,
    late_dec_train_from_day: int | None = None,
) -> dict[str, float | int | str | list[int] | bool | None]:
    n_epochs = max_epochs if max_epochs is not None else MAX_EPOCHS
    stop_patience = early_stop_patience if early_stop_patience is not None else EARLY_STOP_PATIENCE
    stop_min_delta = early_stop_min_delta if early_stop_min_delta is not None else EARLY_STOP_MIN_DELTA
    seed_v = int(seed if seed is not None else SEED)
    _set_seed(seed_v)

    strategy = (val_strategy or VAL_STRATEGY).strip().lower()
    if strategy not in ("december", "multi_window"):
        raise ValueError(f"val_strategy must be 'december' or 'multi_window', got {strategy!r}")
    windows_cfg = list(val_windows) if val_windows is not None else list(VAL_WINDOWS)
    late_dec = (
        late_dec_train_from_day
        if late_dec_train_from_day is not None
        else (MULTI_VAL_LATE_DEC_TRAIN_FROM_DAY if strategy == "multi_window" else None)
    )

    df = load_reefer(data_path)
    weather = load_weather_folder(weather_folder) if weather_folder else None
    feat = build_hourly_feature_table(df, weather=weather).dropna().reset_index(drop=True)
    seq_cols = pick_sequence_columns(feat)

    x_all = feat[seq_cols].to_numpy(dtype=float)
    y_all = feat["total_power_kw"].to_numpy(dtype=float)

    train_idx, val_idx_windows, test_idx, ts_series = calendar_split_dl_indices(
        feat["timestamp_utc"],
        strategy=strategy,  # type: ignore[arg-type]
        train_year=CAL_TRAIN_YEAR,
        train_last_month=CAL_TRAIN_LAST_MONTH,
        val_year=CAL_VAL_YEAR,
        val_month=CAL_VAL_MONTH,
        val_day_first=CAL_VAL_DAY_FIRST,
        val_day_last=CAL_VAL_DAY_LAST,
        val_windows=windows_cfg,
        late_dec_train_from_day=late_dec,
        test_year=CAL_TEST_YEAR,
        test_month=CAL_TEST_MONTH,
        test_day_first=CAL_TEST_DAY_FIRST,
        test_day_last=CAL_TEST_DAY_LAST,
        seq_len=SEQ_LEN,
    )
    val_idx_union = sorted({i for w in val_idx_windows for i in w})
    fit_mask = (ts_series.dt.year == CAL_TRAIN_YEAR) & (ts_series.dt.month <= CAL_TRAIN_LAST_MONTH)
    fit_rows = fit_mask.to_numpy()
    if fit_rows.sum() < 100:
        raise ValueError(
            f"Not enough {CAL_TRAIN_YEAR} Jan..{CAL_TRAIN_LAST_MONTH} rows for scaler ({fit_rows.sum()}). "
            "Check reefer date range."
        )
    if TRAIN_EXCLUDED_YEAR_MONTHS:
        assert_holdout_months_excluded_from_training(
            ts_series,
            train_idx,
            SEQ_LEN,
            fit_rows,
            holdout_year_months=set(TRAIN_EXCLUDED_YEAR_MONTHS),
        )
        print(
            "Verified: training excludes calendar month(s) "
            f"{sorted(TRAIN_EXCLUDED_YEAR_MONTHS)} — not used as train labels, "
            "LSTM inputs, or scaler/y-stats fit (val/test only).",
            file=sys.stderr,
        )
    if not train_idx:
        raise ValueError(f"Empty train target indices (train={len(train_idx)}). Check calendar config vs hourly table.")
    for wi, widx in enumerate(val_idx_windows):
        if not widx:
            raise ValueError(
                f"Empty val window {wi} ({windows_cfg[wi] if wi < len(windows_cfg) else '?'}). "
                "Check VAL_WINDOWS vs data coverage."
            )
    skip_test_eval = not test_idx
    if skip_test_eval:
        print(
            f"WARNING: no test target hours for {CAL_TEST_YEAR}-{CAL_TEST_MONTH:02d} "
            f"days {CAL_TEST_DAY_FIRST}..{CAL_TEST_DAY_LAST} (data may end before this range). "
            "Training continues; metrics_test will omit composite (checkpoint still from val). "
            "Adjust CAL_TEST_* if your CSV includes late January.",
            file=sys.stderr,
        )

    scaler = Standardizer.fit(x_all[fit_rows])
    x_scaled = scaler.transform(x_all)

    y_fit = y_all[fit_rows]
    y_mean = float(np.mean(y_fit))
    y_std = float(max(np.std(y_fit), 1e-3))
    y_norm = (y_all - y_mean) / y_std

    ds_train = SequenceDataset(x_scaled, y_norm, seq_len=SEQ_LEN, target_indices=train_idx)
    ds_val_list = [
        SequenceDataset(x_scaled, y_norm, seq_len=SEQ_LEN, target_indices=widx) for widx in val_idx_windows
    ]
    ds_test = (
        None
        if skip_test_eval
        else SequenceDataset(x_scaled, y_norm, seq_len=SEQ_LEN, target_indices=test_idx)
    )

    val_feat_concat = pd.concat(
        [feat.iloc[widx].reset_index(drop=True) for widx in val_idx_windows],
        ignore_index=True,
    )
    wind_col = _pick_wx_lag1_col(val_feat_concat.columns, "wind")
    wx_temp_col = _pick_wx_lag1_col(
        val_feat_concat.columns, "temp", "temperatur", "airtmp", exclude=("wind", "feucht", "humid")
    )
    wx_humid_col = _pick_wx_lag1_col(val_feat_concat.columns, "feucht", "humid", "rfeuchte", "nass")
    amb_lag_col = "ambient_avg_lag1" if "ambient_avg_lag1" in val_feat_concat.columns else None
    tier3_col = next(
        (c for c in val_feat_concat.columns if "tier3" in c.lower() and c.endswith("_lag1")),
        None,
    )
    hw_cols = [
        c
        for c in val_feat_concat.columns
        if c.startswith("hw_") and c.endswith("_share_lag1") and "other" not in c
    ]
    if hw_cols:
        hw_cols = sorted(hw_cols, key=lambda c: float(val_feat_concat[c].mean(skipna=True)), reverse=True)[:2]

    wx_lag1_all = sorted(
        c
        for c in feat.columns
        if str(c).startswith("wx_") and str(c).endswith("_lag1") and "missing" not in str(c).lower()
    )
    preview = ", ".join(wx_lag1_all[:14]) + (" ..." if len(wx_lag1_all) > 14 else "")
    print(
        f"DL features: {len(seq_cols)} numeric cols | wx_*_lag1 count={len(wx_lag1_all)} | {preview}",
        file=sys.stderr,
    )

    g = torch.Generator()
    g.manual_seed(seed_v)
    dl_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    dl_train_eval = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=False)
    dl_val_list = [DataLoader(ds_v, batch_size=BATCH_SIZE, shuffle=False) for ds_v in ds_val_list]
    dl_test = (
        None
        if ds_test is None
        else DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_feat = x_scaled.shape[1]
    model_point = LSTMForecaster(
        n_features=n_feat,
        hidden_size=HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS,
        dropout=DROPOUT,
        use_attention=USE_ATTENTION,
        use_deep_head=DEEP_HEAD,
        head_dropout=HEAD_DROPOUT,
    ).to(device)
    model_p90 = LSTMForecaster(
        n_features=n_feat,
        hidden_size=HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS,
        dropout=DROPOUT,
        use_attention=USE_ATTENTION,
        use_deep_head=DEEP_HEAD,
        head_dropout=HEAD_DROPOUT,
    ).to(device)

    n_param_pt = sum(p.numel() for p in model_point.parameters())
    n_param_p9 = sum(p.numel() for p in model_p90.parameters())
    print(
        f"Model parameters: point={n_param_pt} p90={n_param_p9} total={n_param_pt + n_param_p9}",
        file=sys.stderr,
    )
    _peak_desc = (
        f"SmoothL1(beta={HUBER_BETA_PEAK}) on batch top-decile"
        if USE_SMOOTH_L1_PEAK
        else "L1 on batch top-decile"
    )
    print(
        f"Train loss: {LOSS_W_ALL}*L1_all + {LOSS_W_PEAK}*peak({_peak_desc}) + {LOSS_W_PIN}*pinball_p90",
        file=sys.stderr,
    )

    all_params = list(model_point.parameters()) + list(model_p90.parameters())
    opt_point = torch.optim.AdamW(model_point.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    opt_p90 = torch.optim.AdamW(
        model_p90.parameters(), lr=LR * P90_HEAD_LR_MULT, weight_decay=WEIGHT_DECAY
    )
    scheduler_point = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_point,
        mode="min",
        factor=LR_PLATEAU_FACTOR,
        patience=LR_PLATEAU_PATIENCE,
        min_lr=LR_PLATEAU_MIN,
    )
    scheduler_p90 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt_p90,
        mode="min",
        factor=LR_PLATEAU_FACTOR,
        patience=LR_PLATEAU_PATIENCE,
        min_lr=LR_PLATEAU_MIN,
    )

    best_val_comp = float("inf")
    best_epoch = 0
    best_val_window_comps: tuple[float, ...] | None = None
    patience_left = stop_patience
    val_ema: float | None = None
    best_val_ema = float("inf")
    ema_alpha = float(EARLY_STOP_EMA_ALPHA)
    history: list[dict[str, float | int]] = []

    use_pb = bool(show_progress and tqdm is not None)
    use_tw = use_pb
    if show_progress and tqdm is None:
        print("Tip: install tqdm for progress bars: pip install tqdm", file=sys.stderr)

    if strategy == "multi_window":
        wh = [len(w) for w in val_idx_windows]
        wdesc = " | ".join(f"w{i + 1}={windows_cfg[i]} n={wh[i]}" for i in range(len(val_idx_windows)))
        split_msg = (
            f"DL calendar split | strategy=multi_window | fit scaler+y on {CAL_TRAIN_YEAR} Jan..{CAL_TRAIN_LAST_MONTH} | "
            f"train target hours={len(train_idx)} (excl. val windows; Dec>={late_dec} train if set) | "
            f"val windows: {wdesc} | union val hours={len(val_idx_union)} | "
            f"test {CAL_TEST_YEAR}-{CAL_TEST_MONTH:02d} d{CAL_TEST_DAY_FIRST:02d}..d{CAL_TEST_DAY_LAST:02d} "
            f"hours={len(test_idx)}"
        )
    else:
        split_msg = (
            f"DL calendar split | strategy=december | fit scaler+y on {CAL_TRAIN_YEAR} Jan..{CAL_TRAIN_LAST_MONTH} | "
            f"train target hours={len(train_idx)} | "
            f"val {CAL_VAL_YEAR}-{CAL_VAL_MONTH:02d} d{CAL_VAL_DAY_FIRST:02d}..d{CAL_VAL_DAY_LAST:02d} "
            f"hours={len(val_idx_windows[0])} | "
            f"test {CAL_TEST_YEAR}-{CAL_TEST_MONTH:02d} d{CAL_TEST_DAY_FIRST:02d}..d{CAL_TEST_DAY_LAST:02d} "
            f"hours={len(test_idx)}"
        )
    _log_line(split_msg, use_tw)
    _log_line(
        "Checkpoint selection: mean val_composite across val windows when strategy=multi_window; "
        "single val window when strategy=december. EMA early-stop uses the same val_comp.",
        use_tw,
    )

    _log_line(
        "epoch | train_loss_n | tr_mae tr_peak tr_pin tr_comp | val_mae val_peak val_pin val_comp | "
        "val_w* | val_ema | lr | best_raw | gap_c | stop  "
        "(val_comp = mean composite over windows; val_w* = per-window composite kW)",
        use_tw,
    )
    _log_line("-" * 128, use_tw)

    epoch_iter = range(n_epochs)
    if use_pb:
        epoch_pbar = tqdm(
            epoch_iter,
            desc="Epochs",
            unit="epoch",
            total=n_epochs,
            file=sys.stderr,
            dynamic_ncols=True,
        )
    else:
        epoch_pbar = epoch_iter

    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in epoch_pbar:
        model_point.train()
        model_p90.train()
        train_loss_sum = 0.0
        n_batches = 0

        batch_iter = dl_train
        if use_pb:
            batch_iter = tqdm(
                dl_train,
                desc=f"Train batches {epoch + 1}/{n_epochs}",
                leave=False,
                unit="batch",
                file=sys.stderr,
                dynamic_ncols=True,
            )

        for x_seq, y in batch_iter:
            x_seq = x_seq.to(device)
            y = y.to(device)

            p_hat = model_point(x_seq)
            q_hat = model_p90(x_seq)
            q_hat = torch.maximum(q_hat, p_hat)

            mae_all = F.l1_loss(p_hat, y)
            thr = torch.quantile(y, 0.9)
            peak = y >= thr
            if peak.any():
                if USE_SMOOTH_L1_PEAK:
                    mae_peak_b = F.smooth_l1_loss(
                        p_hat[peak], y[peak], beta=HUBER_BETA_PEAK, reduction="mean"
                    )
                else:
                    mae_peak_b = F.l1_loss(p_hat[peak], y[peak])
            else:
                mae_peak_b = mae_all
            pin = _pinball_loss_torch(y, q_hat, q=0.9)
            loss = LOSS_W_ALL * mae_all + LOSS_W_PEAK * mae_peak_b + LOSS_W_PIN * pin

            opt_point.zero_grad()
            opt_p90.zero_grad()
            loss.backward()
            if GRAD_CLIP_NORM and GRAD_CLIP_NORM > 0:
                torch.nn.utils.clip_grad_norm_(all_params, GRAD_CLIP_NORM)
            opt_point.step()
            opt_p90.step()
            train_loss_sum += float(loss.item())
            n_batches += 1
            if use_pb and hasattr(batch_iter, "set_postfix"):
                batch_iter.set_postfix(loss=f"{float(loss.item()):.4f}")

        train_loss_n = train_loss_sum / max(n_batches, 1)

        y_tr, p_tr, q_tr = collect_preds_kw(model_point, model_p90, dl_train_eval, device, y_mean, y_std)
        p_tr = np.maximum(p_tr, 0.0)
        q_tr = np.maximum(q_tr, p_tr)
        tr_mae = mae(y_tr, p_tr)
        tr_peak = mae_peak(y_tr, p_tr, 0.9)
        tr_pin = pinball_loss(y_tr, q_tr, 0.9)
        tr_comp = composite_score(tr_mae, tr_peak, tr_pin)

        y_va_parts: list[np.ndarray] = []
        p_va_parts: list[np.ndarray] = []
        q_va_parts: list[np.ndarray] = []
        window_comps: list[float] = []
        for dl_v in dl_val_list:
            y_w, p_w, q_w = collect_preds_kw(model_point, model_p90, dl_v, device, y_mean, y_std)
            p_w = np.maximum(p_w, 0.0)
            q_w = np.maximum(q_w, p_w)
            window_comps.append(
                float(
                    composite_score(
                        float(mae(y_w, p_w)),
                        float(mae_peak(y_w, p_w, 0.9)),
                        float(pinball_loss(y_w, q_w, 0.9)),
                    )
                )
            )
            y_va_parts.append(y_w)
            p_va_parts.append(p_w)
            q_va_parts.append(q_w)
        y_va = np.concatenate(y_va_parts, axis=0)
        p_va = np.concatenate(p_va_parts, axis=0)
        q_va = np.concatenate(q_va_parts, axis=0)
        va_mae = mae(y_va, p_va)
        va_peak = mae_peak(y_va, p_va, 0.9)
        va_pin = pinball_loss(y_va, q_va, 0.9)
        va_comp = float(np.mean(window_comps)) if window_comps else composite_score(va_mae, va_peak, va_pin)
        val_w_str = "/".join(f"{c:.1f}" for c in window_comps)

        wind_means = _bin_means(p_va, val_feat_concat[wind_col]) if wind_col else None
        wx_temp_means = _bin_means(p_va, val_feat_concat[wx_temp_col]) if wx_temp_col else None
        wx_humid_means = _bin_means(p_va, val_feat_concat[wx_humid_col]) if wx_humid_col else None
        amb_lag_means = _bin_means(p_va, val_feat_concat[amb_lag_col]) if amb_lag_col else None
        tier_means = _bin_means(p_va, val_feat_concat[tier3_col]) if tier3_col else None
        hw_summary: dict[str, tuple[float, float]] = {}
        for col in hw_cols:
            low_high = _low_high_means(p_va, val_feat_concat[col])
            if low_high is not None:
                hw_summary[col] = low_high

        lr_now = float(opt_point.param_groups[0]["lr"])
        scheduler_point.step(va_comp)
        scheduler_p90.step(va_comp)

        # Best weights = best raw val composite (mean across windows when multi_window).
        if va_comp < best_val_comp - stop_min_delta:
            best_val_comp = float(va_comp)
            best_epoch = int(epoch + 1)
            best_val_window_comps = tuple(float(x) for x in window_comps)
            torch.save(
                _build_checkpoint_bundle(
                    model_point=model_point,
                    model_p90=model_p90,
                    best_epoch_1based=best_epoch,
                    best_val_composite_kw=best_val_comp,
                    seq_cols=seq_cols,
                    n_features=n_feat,
                    scaler=scaler,
                    y_mean=y_mean,
                    y_std=y_std,
                    seed=seed_v,
                    val_strategy=strategy,
                ),
                out_dir / "checkpoint_best.pt",
            )

        if ema_alpha <= 0.0 or ema_alpha >= 1.0:
            val_ema = float(va_comp)
        elif val_ema is None:
            val_ema = float(va_comp)
        else:
            val_ema = (1.0 - ema_alpha) * val_ema + ema_alpha * float(va_comp)

        if val_ema < best_val_ema - stop_min_delta:
            best_val_ema = float(val_ema)
            patience_left = stop_patience
        else:
            patience_left -= 1

        row = {
            "epoch": epoch + 1,
            "train_loss_norm": train_loss_n,
            "train_mae_kw": tr_mae,
            "train_mae_peak_kw": tr_peak,
            "train_pinball_p90": tr_pin,
            "train_composite_kw": tr_comp,
            "val_mae_kw": va_mae,
            "val_mae_peak_kw": va_peak,
            "val_pinball_p90": va_pin,
            "val_composite_kw": va_comp,
            "val_window_composites_kw": window_comps,
            "val_composite_ema_kw": val_ema,
            "wind_bin_means_kw": wind_means,
            "wx_temperature_bin_means_kw": wx_temp_means,
            "wx_humidity_bin_means_kw": wx_humid_means,
            "ambient_avg_lag1_bin_means_kw": amb_lag_means,
            "tier3_bin_means_kw": tier_means,
            "hardware_share_low_high_kw": (
                {k: [float(a), float(b)] for k, (a, b) in hw_summary.items()} if hw_summary else None
            ),
            "lr": lr_now,
            "best_val_composite_kw": best_val_comp,
            "best_val_composite_ema_kw": best_val_ema,
            "patience_left": patience_left,
        }
        history.append(row)

        gap_c = float(va_comp - tr_comp)
        line = (
            f"{epoch + 1:5d} | {train_loss_n:12.5f} | "
            f"{tr_mae:7.2f} {tr_peak:8.2f} {tr_pin:6.2f} {tr_comp:8.2f} | "
            f"{va_mae:7.2f} {va_peak:8.2f} {va_pin:6.2f} {va_comp:8.2f} | "
            f"{val_w_str:16s} | {val_ema:7.2f} | {lr_now:.2e} | {best_val_comp:8.3f} | {gap_c:+7.2f} | {patience_left:3d}"
        )
        _log_line(line, use_tw)

        if wind_means is not None:
            _log_line(
                f"    wind({wind_col}) low/mid/high: {wind_means[0]:.1f}/{wind_means[1]:.1f}/{wind_means[2]:.1f} kW",
                use_tw,
            )
        if wx_temp_means is not None:
            _log_line(
                f"    wx_temp({wx_temp_col}) low/mid/high: {wx_temp_means[0]:.1f}/{wx_temp_means[1]:.1f}/{wx_temp_means[2]:.1f} kW",
                use_tw,
            )
        if wx_humid_means is not None:
            _log_line(
                f"    wx_humid({wx_humid_col}) low/mid/high: {wx_humid_means[0]:.1f}/{wx_humid_means[1]:.1f}/{wx_humid_means[2]:.1f} kW",
                use_tw,
            )
        if amb_lag_means is not None:
            _log_line(
                f"    reefer_amb_lag({amb_lag_col}) low/mid/high: {amb_lag_means[0]:.1f}/{amb_lag_means[1]:.1f}/{amb_lag_means[2]:.1f} kW",
                use_tw,
            )
        if tier_means is not None:
            _log_line(
                f"    tier3({tier3_col}) low/mid/high: {tier_means[0]:.1f}/{tier_means[1]:.1f}/{tier_means[2]:.1f} kW",
                use_tw,
            )
        if hw_summary:
            for col, (low, high) in hw_summary.items():
                _log_line(
                    f"    hw({col}) low/high: {low:.1f}/{high:.1f} kW",
                    use_tw,
                )

        if use_pb and hasattr(epoch_pbar, "set_postfix"):
            epoch_pbar.set_postfix(
                tr_c=f"{tr_comp:.1f}",
                val_c=f"{va_comp:.1f}",
                gap_c=f"{gap_c:+.1f}",
                v_ema=f"{val_ema:.1f}",
                best=f"{best_val_comp:.1f}",
                es=patience_left,
            )

        torch.save(
            _build_checkpoint_bundle(
                model_point=model_point,
                model_p90=model_p90,
                best_epoch_1based=int(epoch + 1),
                best_val_composite_kw=float(va_comp),
                seq_cols=seq_cols,
                n_features=n_feat,
                scaler=scaler,
                y_mean=y_mean,
                y_std=y_std,
                seed=seed_v,
                val_strategy=strategy,
            ),
            out_dir / "checkpoint_latest.pt",
        )

        if patience_left <= 0:
            _log_line(
                f"Early stop: val composite EMA (mean over windows) did not improve by > {stop_min_delta} kW "
                f"for {stop_patience} epochs (checkpoint = best raw mean val_comp, epoch {best_epoch}).",
                use_tw,
            )
            break

    best_path = out_dir / "checkpoint_best.pt"
    if not best_path.is_file():
        raise RuntimeError("DL training failed to produce checkpoint_best.pt.")

    try:
        ck_best = torch.load(best_path, map_location=device, weights_only=False)
    except TypeError:
        ck_best = torch.load(best_path, map_location=device)
    if "point_state" in ck_best:
        model_point.load_state_dict(ck_best["point_state"])
        model_p90.load_state_dict(ck_best["p90_state"])
    elif "model_point_state_dict" in ck_best:
        model_point.load_state_dict(ck_best["model_point_state_dict"])
        model_p90.load_state_dict(ck_best["model_p90_state_dict"])
    else:
        raise KeyError("checkpoint_best.pt missing point_state/p90_state (or legacy model_*_state_dict).")
    _be = int(ck_best.get("best_epoch", best_epoch))
    _bvc = float(
        ck_best.get("best_val_composite_kw", ck_best.get("best_selection_val_composite_kw", best_val_comp))
    )
    _log_line(f"Best checkpoint: epoch {_be}, val_composite={_bvc:.3f} kW", use_tw)

    if skip_test_eval or dl_test is None:
        results = {
            "mae_all": None,
            "mae_peak": None,
            "pinball_p90": None,
            "composite_score": None,
            "test_eval_skipped": True,
            "test_eval_skip_reason": (
                f"No hourly rows for test window {CAL_TEST_YEAR}-{CAL_TEST_MONTH:02d} "
                f"days {CAL_TEST_DAY_FIRST}-{CAL_TEST_DAY_LAST}."
            ),
        }
    else:
        y_te, p_te, q_te = collect_preds_kw(model_point, model_p90, dl_test, device, y_mean, y_std)
        p_te = np.maximum(p_te, 0.0)
        q_te = np.maximum(q_te, p_te)

        results = {
            "mae_all": float(mae(y_te, p_te)),
            "mae_peak": float(mae_peak(y_te, p_te, 0.9)),
            "pinball_p90": float(pinball_loss(y_te, q_te, 0.9)),
        }
        results["composite_score"] = float(
            composite_score(results["mae_all"], results["mae_peak"], results["pinball_p90"])
        )
        results["test_eval_skipped"] = False
    results["train_target_hours"] = int(len(train_idx))
    results["val_target_hours"] = int(len(val_idx_union))
    results["val_window_target_hours"] = [int(len(w)) for w in val_idx_windows]
    results["val_strategy"] = strategy
    results["best_epoch"] = int(best_epoch)
    results["test_target_hours"] = int(len(test_idx))
    if strategy == "multi_window":
        ld_part = f"dec{int(late_dec)}plus_" if late_dec is not None else "no_late_dec_"
        results["split_description"] = (
            f"train_{CAL_TRAIN_YEAR}_m01_m{CAL_TRAIN_LAST_MONTH:02d}_excl_valwindows_{ld_part}"
            f"val_multi_{len(val_idx_windows)}win_test_{CAL_TEST_YEAR}m{CAL_TEST_MONTH:02d}_"
            f"d{CAL_TEST_DAY_FIRST:02d}_d{CAL_TEST_DAY_LAST:02d}"
        )
    else:
        results["split_description"] = (
            f"train_{CAL_TRAIN_YEAR}_m01_m{CAL_TRAIN_LAST_MONTH:02d}_"
            f"val_{CAL_VAL_YEAR}m{CAL_VAL_MONTH:02d}_d{CAL_VAL_DAY_FIRST:02d}_d{CAL_VAL_DAY_LAST:02d}_"
            f"test_{CAL_TEST_YEAR}m{CAL_TEST_MONTH:02d}_d{CAL_TEST_DAY_FIRST:02d}_d{CAL_TEST_DAY_LAST:02d}"
        )
    results["best_val_composite_kw"] = float(best_val_comp)
    results["epochs_ran"] = int(history[-1]["epoch"]) if history else 0

    meta = {
        "hidden_size": HIDDEN_SIZE,
        "lstm_num_layers": LSTM_NUM_LAYERS,
        "seq_len": SEQ_LEN,
        "y_mean": y_mean,
        "y_std": y_std,
        "dropout": DROPOUT,
        "use_attention": USE_ATTENTION,
        "use_deep_head": DEEP_HEAD,
        "head_dropout": HEAD_DROPOUT,
    }

    ckpt_common = {
        "n_features": n_feat,
        "seq_cols": seq_cols,
        "hidden_size": HIDDEN_SIZE,
        "lstm_num_layers": LSTM_NUM_LAYERS,
        "dropout": DROPOUT,
        "use_attention": USE_ATTENTION,
        "use_deep_head": DEEP_HEAD,
        "head_dropout": HEAD_DROPOUT,
        "meta": meta,
    }
    torch.save({**ckpt_common, "state_dict": model_point.state_dict()}, out_dir / "model_point.pt")
    torch.save({**ckpt_common, "state_dict": model_p90.state_dict()}, out_dir / "model_p90.pt")
    np.savez(
        out_dir / "scaler.npz",
        mean=scaler.mean,
        std=scaler.std,
        y_mean=np.array(y_mean),
        y_std=np.array(y_std),
    )
    (out_dir / "metrics_test.json").write_text(json.dumps(results, indent=2))
    (out_dir / "training_history.json").write_text(json.dumps(_sanitize_history_json(history), indent=2))
    (out_dir / "training_config.json").write_text(
        json.dumps(
            {
                "max_epochs": n_epochs,
                "early_stop_patience": stop_patience,
                "early_stop_min_delta_kw": stop_min_delta,
                "early_stop_ema_alpha": ema_alpha,
                "p90_head_lr_mult": P90_HEAD_LR_MULT,
                "seed": seed_v,
                "seq_len": SEQ_LEN,
                "hidden_size": HIDDEN_SIZE,
                "lstm_num_layers": LSTM_NUM_LAYERS,
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "loss_w_all": LOSS_W_ALL,
                "loss_w_peak": LOSS_W_PEAK,
                "loss_w_pin": LOSS_W_PIN,
                "use_smooth_l1_peak": USE_SMOOTH_L1_PEAK,
                "huber_beta_peak": HUBER_BETA_PEAK,
                "lr_plateau_patience": LR_PLATEAU_PATIENCE,
                "use_attention": USE_ATTENTION,
                "use_deep_head": DEEP_HEAD,
                "head_dropout": HEAD_DROPOUT,
                "calendar_split": {
                    "val_strategy": strategy,
                    "val_windows": windows_cfg if strategy == "multi_window" else None,
                    "late_dec_train_from_day": late_dec,
                    "scaler_and_y_stats": f"{CAL_TRAIN_YEAR}-01 .. {CAL_TRAIN_YEAR}-{CAL_TRAIN_LAST_MONTH:02d}",
                    "train_targets": (
                        f"{CAL_TRAIN_YEAR}-01 .. {CAL_TRAIN_YEAR}-{CAL_TRAIN_LAST_MONTH:02d} "
                        f"excluding val windows; Dec {late_dec}+ train targets when multi_window"
                        if strategy == "multi_window"
                        else f"{CAL_TRAIN_YEAR}-01 .. {CAL_TRAIN_YEAR}-{CAL_TRAIN_LAST_MONTH:02d}"
                    ),
                    "val_targets": (
                        [dict(w) for w in windows_cfg]
                        if strategy == "multi_window"
                        else (
                            f"{CAL_VAL_YEAR}-{CAL_VAL_MONTH:02d}-"
                            f"{CAL_VAL_DAY_FIRST:02d} .. {CAL_VAL_DAY_LAST:02d}"
                        )
                    ),
                    "test_targets": (
                        f"{CAL_TEST_YEAR}-{CAL_TEST_MONTH:02d}-"
                        f"{CAL_TEST_DAY_FIRST:02d} .. {CAL_TEST_DAY_LAST:02d}"
                    ),
                    "train_target_hours": len(train_idx),
                    "val_target_hours_union": len(val_idx_union),
                    "val_window_target_hours": [len(w) for w in val_idx_windows],
                    "test_target_hours": len(test_idx),
                    "best_epoch": int(best_epoch),
                },
            },
            indent=2,
        )
    )
    return results


if __name__ == "__main__":
    default_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="Train DL forecaster (train/val/test time split).")
    p.add_argument("--repo-root", type=Path, default=default_root, help="Repository root (contains participant_package/, main/).")
    p.add_argument("--reefer", type=Path, default=None, help="Override path to reefer_release.csv")
    p.add_argument("--out-dir", type=Path, default=None, help="Directory for checkpoints (default: <repo>/main/models/dl)")
    p.add_argument("--weather-folder", type=Path, default=None, help="Folder with hourly weather CSVs")
    p.add_argument("--max-epochs", type=int, default=None)
    p.add_argument("--early-stop-patience", type=int, default=None)
    p.add_argument("--early-stop-min-delta", type=float, default=None)
    p.add_argument("--seed", type=int, default=None, help="RNG seed (default from config).")
    p.add_argument(
        "--val-strategy",
        type=str,
        default=None,
        choices=("december", "multi_window"),
        help="Validation layout (default: main.dl.config.VAL_STRATEGY). "
        "multi_window = late Oct + mid Nov + early Dec; checkpoint uses mean val composite.",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm bars (plain logs only).",
    )
    args = p.parse_args()
    root = args.repo_root
    data_path = args.reefer or (root / "participant_package/reefer_release.csv")
    out_dir = args.out_dir or (root / "main/models/dl")
    weather_folder = args.weather_folder or (root / "participant_package/Wetterdaten Okt 25 - 23 Feb 26")
    metrics = run_train_dl(
        data_path=data_path,
        out_dir=out_dir,
        weather_folder=weather_folder,
        max_epochs=args.max_epochs,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        show_progress=not args.no_progress,
        seed=args.seed,
        val_strategy=args.val_strategy,
    )
    print("\nDL test window metrics (kW); None if test window has no rows in CSV")
    for k, v in metrics.items():
        print(f"  {k}: {v}")