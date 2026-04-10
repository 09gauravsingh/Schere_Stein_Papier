from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from main.dl.config import DROPOUT, HIDDEN_SIZE, LSTM_NUM_LAYERS, SEQ_LEN
from main.dl.model import LSTMForecaster
from main.src.data.load_reefer import load_reefer
from main.src.data.load_weather import load_weather_folder
from main.src.data.load_targets import load_targets
from main.src.inference.submission_writer import write_submission
from main.src.pipeline.feature_table import build_hourly_feature_table


def _torch_load(path: Path, map_location: str):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _norm_ts_key(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    if t.tzinfo is not None:
        t = t.tz_convert("UTC").tz_localize(None)
    return t.floor("h")


def _scaler_arrays_from_checkpoint(ck: dict) -> tuple[np.ndarray, np.ndarray]:
    if "scaler_mean" in ck and "scaler_scale" in ck:
        return np.asarray(ck["scaler_mean"]), np.asarray(ck["scaler_scale"])
    raise KeyError("Unified checkpoint missing scaler_mean/scaler_scale")


def _build_models_from_meta(
    ck: dict,
    *,
    n_features: int,
    map_location: str,
) -> tuple[LSTMForecaster, LSTMForecaster, dict]:
    hidden = int(ck.get("hidden_size", HIDDEN_SIZE))
    n_layers = int(ck.get("lstm_num_layers", LSTM_NUM_LAYERS))
    dropout = float(ck.get("dropout", 0.3))
    use_attention = bool(ck.get("use_attention", False))
    use_deep_head = bool(ck.get("use_deep_head", False))
    head_dropout = float(ck.get("head_dropout", 0.1))

    model_point = LSTMForecaster(
        n_features=n_features,
        hidden_size=hidden,
        num_layers=n_layers,
        dropout=dropout,
        use_attention=use_attention,
        use_deep_head=use_deep_head,
        head_dropout=head_dropout,
    )
    model_p90 = LSTMForecaster(
        n_features=n_features,
        hidden_size=hidden,
        num_layers=n_layers,
        dropout=dropout,
        use_attention=use_attention,
        use_deep_head=use_deep_head,
        head_dropout=head_dropout,
    )

    if "point_state" in ck and "p90_state" in ck:
        model_point.load_state_dict(ck["point_state"])
        model_p90.load_state_dict(ck["p90_state"])
    elif "model_point_state_dict" in ck and "model_p90_state_dict" in ck:
        model_point.load_state_dict(ck["model_point_state_dict"])
        model_p90.load_state_dict(ck["model_p90_state_dict"])
    else:
        raise KeyError(
            "Checkpoint must contain point_state/p90_state or legacy model_*_state_dict keys."
        )

    model_point.to(map_location)
    model_p90.to(map_location)
    model_point.eval()
    model_p90.eval()
    return model_point, model_p90, ck


def run_infer_dl(
    data_path: Path,
    target_path: Path,
    model_dir: Path,
    out_path: Path,
    weather_folder: Path | None = None,
    *,
    checkpoint_path: Path | None = None,
) -> None:
    map_location = "cpu"

    if checkpoint_path is not None:
        ck_unified = _torch_load(checkpoint_path, map_location)
        seq_cols = ck_unified["seq_cols"]
        n_features = int(ck_unified["n_features"])
        x_mean, x_scale = _scaler_arrays_from_checkpoint(ck_unified)
        y_mean = float(ck_unified["y_mean"])
        y_std = float(ck_unified["y_std"])
        model_point, model_p90, _meta = _build_models_from_meta(
            ck_unified, n_features=n_features, map_location=map_location
        )
    else:
        ckpt_point = _torch_load(model_dir / "model_point.pt", map_location)
        ckpt_p90 = _torch_load(model_dir / "model_p90.pt", map_location)
        scaler = np.load(model_dir / "scaler.npz")
        seq_cols = ckpt_point["seq_cols"]
        n_features = int(ckpt_point["n_features"])
        hidden = int(ckpt_point.get("hidden_size", HIDDEN_SIZE))
        n_layers = int(ckpt_point.get("lstm_num_layers", LSTM_NUM_LAYERS))
        meta_pt = ckpt_point.get("meta") or {}
        dropout = float(ckpt_point.get("dropout", meta_pt.get("dropout", DROPOUT)))
        use_attention = bool(ckpt_point.get("use_attention", False))
        use_deep_head = bool(ckpt_point.get("use_deep_head", False))
        head_dropout = float(ckpt_point.get("head_dropout", 0.1))

        y_mean = float(scaler["y_mean"]) if "y_mean" in scaler.files else 0.0
        y_std = float(scaler["y_std"]) if "y_std" in scaler.files else 1.0
        x_mean = np.asarray(scaler["mean"])
        x_scale = np.asarray(scaler["std"])

        model_point = LSTMForecaster(
            n_features=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            dropout=dropout,
            use_attention=use_attention,
            use_deep_head=use_deep_head,
            head_dropout=head_dropout,
        )
        model_p90 = LSTMForecaster(
            n_features=n_features,
            hidden_size=hidden,
            num_layers=n_layers,
            dropout=dropout,
            use_attention=use_attention,
            use_deep_head=use_deep_head,
            head_dropout=head_dropout,
        )
        model_point.load_state_dict(ckpt_point["state_dict"])
        model_p90.load_state_dict(ckpt_p90["state_dict"])
        model_point.eval()
        model_p90.eval()

    df = load_reefer(data_path)
    weather = load_weather_folder(weather_folder) if weather_folder else None
    feat = build_hourly_feature_table(df, weather=weather).sort_values("timestamp_utc").reset_index(drop=True)
    targets = load_targets(target_path).sort_values("timestamp_utc").reset_index(drop=True)

    x = feat[seq_cols].to_numpy(dtype=float)
    x = (x - x_mean) / x_scale
    ts_to_idx = {_norm_ts_key(t): i for i, t in enumerate(feat["timestamp_utc"])}

    point_preds: list[float] = []
    p90_preds: list[float] = []
    kept_ts: list[pd.Timestamp] = []

    with torch.no_grad():
        for ts in targets["timestamp_utc"]:
            key = _norm_ts_key(ts)
            idx = ts_to_idx.get(key)
            if idx is None or idx < SEQ_LEN:
                continue
            x_seq = torch.tensor(x[idx - SEQ_LEN : idx, :], dtype=torch.float32).unsqueeze(0)
            p = float(model_point(x_seq).item()) * y_std + y_mean
            q = float(model_p90(x_seq).item()) * y_std + y_mean
            p = max(p, 0.0)
            q = max(q, p)
            kept_ts.append(key)
            point_preds.append(p)
            p90_preds.append(q)

    out = pd.DataFrame(
        {"timestamp_utc": kept_ts, "pred_power_kw": point_preds, "pred_p90_kw": p90_preds}
    )
    if len(out) != len(targets):
        raise ValueError(
            f"DL inference produced {len(out)} rows for {len(targets)} targets. "
            "Need at least SEQ_LEN historical hours before each target."
        )
    write_submission(out, out_path)


if __name__ == "__main__":
    default_root = Path(__file__).resolve().parents[2]
    p = argparse.ArgumentParser(description="DL inference: write predictions CSV for target timestamps.")
    p.add_argument("--repo-root", type=Path, default=default_root)
    p.add_argument("--reefer", type=Path, default=None)
    p.add_argument("--targets", type=Path, default=None, help="Target timestamps CSV")
    p.add_argument("--target-csv", type=Path, default=None, help="Alias for --targets")
    p.add_argument("--model-dir", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None, help="Unified checkpoint_best.pt (self-contained).")
    p.add_argument("--weather-folder", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--output", type=Path, default=None, help="Alias for --out")
    args = p.parse_args()
    root = args.repo_root
    target_path = args.target_csv or args.targets or (root / "participant_package/target_timestamps.csv")
    out_path = args.output or args.out or (root / "main/outputs/predictions_dl.csv")
    run_infer_dl(
        data_path=args.reefer or (root / "participant_package/reefer_release.csv"),
        target_path=target_path,
        model_dir=args.model_dir or (root / "main/models/dl"),
        out_path=out_path,
        weather_folder=args.weather_folder or (root / "participant_package/Wetterdaten Okt 25 - 23 Feb 26"),
        checkpoint_path=args.checkpoint,
    )
    print(f"Wrote {out_path}")
