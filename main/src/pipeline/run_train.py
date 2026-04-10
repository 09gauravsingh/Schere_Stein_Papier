from __future__ import annotations

import argparse
import json
from pathlib import Path

from main.src.data.load_reefer import load_reefer
from main.src.preprocess.cleaning import build_imputation_config, apply_imputation, drop_missing_target
from main.src.data.load_weather import load_weather_folder
from main.src.pipeline.feature_table import build_hourly_feature_table, select_feature_columns
from main.src.modeling.train_point import train_point_model
from main.src.modeling.train_quantile import train_quantile_model
from main.src.modeling.persistence import save_pickle


def run_train(data_path: Path, model_dir: Path, weather_folder: Path | None = None) -> tuple[object, object]:
    df = load_reefer(data_path)
    df = drop_missing_target(df, "AvPowerCons")

    config = build_imputation_config(
        df,
        numeric_cols=["AvPowerCons", "TemperatureAmbient", "TemperatureSetPoint"],
        categorical_cols=["HardwareType", "ContainerSize", "stack_tier"],
    )
    df = apply_imputation(df, config)

    weather = load_weather_folder(weather_folder) if weather_folder else None
    hourly = build_hourly_feature_table(df, weather=weather).dropna().reset_index(drop=True)
    feature_cols = select_feature_columns(hourly)
    X = hourly[feature_cols]
    y = hourly["total_power_kw"]

    model_point = None
    model_p90 = None
    try:
        point_params = {"iterations": 300, "learning_rate": 0.05, "depth": 8}
        quant_params = {"iterations": 300, "learning_rate": 0.05, "depth": 8}
        model_point = train_point_model(X, y, model_type="catboost", params=point_params)
        model_p90 = train_quantile_model(X, y, model_type="catboost", params=quant_params)
        model_backend = "catboost"
    except ImportError:
        try:
            point_params = {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 8}
            quant_params = {"n_estimators": 400, "learning_rate": 0.05, "max_depth": 8}
            model_point = train_point_model(X, y, model_type="lightgbm", params=point_params)
            model_p90 = train_quantile_model(X, y, model_type="lightgbm", params=quant_params)
            model_backend = "lightgbm"
        except ImportError:
            point_params = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3, "random_state": 42}
            quant_params = {"n_estimators": 300, "learning_rate": 0.05, "max_depth": 3, "random_state": 42}
            model_point = train_point_model(X, y, model_type="sklearn_gbr", params=point_params)
            model_p90 = train_quantile_model(X, y, model_type="sklearn_gbr", params=quant_params)
            model_backend = "sklearn_gbr"

    model_dir.mkdir(parents=True, exist_ok=True)
    save_pickle(model_point, model_dir / "model_point.pkl")
    save_pickle(model_p90, model_dir / "model_p90.pkl")
    (model_dir / "feature_columns.json").write_text(json.dumps(feature_cols, indent=2))
    (model_dir / "model_backend.txt").write_text(model_backend)
    return model_point, model_p90


if __name__ == "__main__":
    default_root = Path(__file__).resolve().parents[3]
    ap = argparse.ArgumentParser(description="Train tree models (CatBoost > LightGBM > sklearn GBR fallback).")
    ap.add_argument("--repo-root", type=Path, default=default_root)
    ap.add_argument("--reefer", type=Path, default=None)
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--weather-folder", type=Path, default=None)
    ns = ap.parse_args()
    root = ns.repo_root
    run_train(
        ns.reefer or (root / "participant_package/reefer_release.csv"),
        ns.model_dir or (root / "main/models/tree"),
        ns.weather_folder or (root / "participant_package/Wetterdaten Okt 25 - 23 Feb 26"),
    )
    print(f"Saved tree models under {ns.model_dir or (root / 'main/models/tree')}")
