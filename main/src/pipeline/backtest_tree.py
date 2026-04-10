from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from main.src.data.load_reefer import load_reefer
from main.src.data.load_weather import load_weather_folder
from main.src.eval.metrics import composite_score, mae, mae_peak, pinball_loss
from main.src.modeling.persistence import load_pickle
from main.src.pipeline.feature_table import build_hourly_feature_table


def run_backtest(
    model_dir: Path,
    data_path: Path,
    weather_folder: Path | None = None,
    holdout_hours: int = 24 * 28,
) -> dict[str, float]:
    model_point = load_pickle(model_dir / "model_point.pkl")
    model_p90 = load_pickle(model_dir / "model_p90.pkl")
    feature_cols = json.loads((model_dir / "feature_columns.json").read_text())

    df = load_reefer(data_path)
    weather = load_weather_folder(weather_folder) if weather_folder else None
    feat = build_hourly_feature_table(df, weather=weather).dropna().reset_index(drop=True)
    if len(feat) <= holdout_hours:
        raise ValueError(f"Not enough rows ({len(feat)}) for holdout_hours={holdout_hours}")

    test = feat.tail(holdout_hours).copy()
    x_test = test[feature_cols]
    y_true = test["total_power_kw"].to_numpy(dtype=float)
    y_pred = np.maximum(model_point.predict(x_test), 0.0)
    y_p90 = np.maximum(model_p90.predict(x_test), y_pred)

    m_all = mae(y_true, y_pred)
    m_peak = mae_peak(y_true, y_pred, quantile=0.9)
    p90 = pinball_loss(y_true, y_p90, quantile=0.9)
    score = composite_score(m_all, m_peak, p90)
    return {
        "mae_all": float(m_all),
        "mae_peak": float(m_peak),
        "pinball_p90": float(p90),
        "composite_score": float(score),
        "holdout_hours": float(holdout_hours),
    }


if __name__ == "__main__":
    default_root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description="Backtest saved tree models on a time holdout.")
    ap.add_argument("--repo-root", type=Path, default=default_root)
    ap.add_argument("--reefer", type=Path, default=None)
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--weather-folder", type=Path, default=None)
    ap.add_argument("--holdout-hours", type=int, default=24 * 28)
    ns = ap.parse_args()
    root = ns.repo_root
    results = run_backtest(
        model_dir=ns.model_dir or (root / "main/models/tree"),
        data_path=ns.reefer or (root / "participant_package/reefer_release.csv"),
        weather_folder=ns.weather_folder or (root / "participant_package/Wetterdaten Okt 25 - 23 Feb 26"),
        holdout_hours=ns.holdout_hours,
    )
    print(f"Tree backtest (last {int(results['holdout_hours'])} hours)")
    for k, v in results.items():
        print(f"{k}: {v:.6f}" if "hours" not in k else f"{k}: {int(v)}")
