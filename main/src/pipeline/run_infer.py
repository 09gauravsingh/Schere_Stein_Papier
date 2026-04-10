from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from main.src.data.load_reefer import load_reefer
from main.src.data.load_weather import load_weather_folder
from main.src.data.load_targets import load_targets
from main.src.inference.predict_24h import predict_next_24h
from main.src.inference.submission_writer import write_submission
from main.src.modeling.persistence import load_pickle
from main.src.pipeline.feature_table import build_hourly_feature_table


def run_infer(
    model_point,
    model_p90,
    feature_table,
    feature_cols: list[str],
    target_path: Path,
    out_path: Path,
) -> None:
    targets = load_targets(target_path)
    targets["timestamp_utc"] = pd.to_datetime(targets["timestamp_utc"], utc=True).dt.floor("h").dt.tz_localize(None)
    features = targets.merge(feature_table, on="timestamp_utc", how="left").sort_values("timestamp_utc")
    features = features.ffill()
    if features[feature_cols].isna().any().any():
        raise ValueError("Missing feature rows for one or more target timestamps.")
    model_input = features[feature_cols].copy()
    model_input["timestamp_utc"] = features["timestamp_utc"].values
    preds = predict_next_24h(model_point, model_p90, model_input, timestamp_col="timestamp_utc")
    write_submission(preds, out_path)


if __name__ == "__main__":
    default_root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description="Tree inference: predictions CSV from saved models.")
    ap.add_argument("--repo-root", type=Path, default=default_root)
    ap.add_argument("--reefer", type=Path, default=None)
    ap.add_argument("--targets", type=Path, default=None)
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--weather-folder", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ns = ap.parse_args()
    root = ns.repo_root
    model_dir = ns.model_dir or (root / "main/models/tree")
    model_point = load_pickle(model_dir / "model_point.pkl")
    model_p90 = load_pickle(model_dir / "model_p90.pkl")
    feature_cols = json.loads((model_dir / "feature_columns.json").read_text())

    df = load_reefer(ns.reefer or (root / "participant_package/reefer_release.csv"))
    weather = load_weather_folder(ns.weather_folder or (root / "participant_package/Wetterdaten Okt 25 - 23 Feb 26"))
    feature_table = build_hourly_feature_table(df, weather=weather)

    out_path = ns.out or (root / "main/outputs/predictions_tree.csv")
    run_infer(
        model_point=model_point,
        model_p90=model_p90,
        feature_table=feature_table,
        feature_cols=feature_cols,
        target_path=ns.targets or (root / "participant_package/target_timestamps.csv"),
        out_path=out_path,
    )
    print(f"Wrote {out_path}")
