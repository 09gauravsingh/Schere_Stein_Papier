from __future__ import annotations

from typing import Any

import pandas as pd


def train_point_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "catboost",
    params: dict[str, Any] | None = None,
) -> Any:
    """
    Train point forecast model for pred_power_kw.
    model_type: 'catboost' or 'lightgbm'
    """
    params = params or {}
    if model_type == "catboost":
        try:
            from catboost import CatBoostRegressor
        except ImportError as exc:
            raise ImportError("CatBoost not installed. Please install catboost.") from exc
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, verbose=False)
        return model
    if model_type == "lightgbm":
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise ImportError("LightGBM not installed. Please install lightgbm.") from exc
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train)
        return model
    if model_type == "sklearn_gbr":
        from sklearn.ensemble import GradientBoostingRegressor

        model = GradientBoostingRegressor(**params)
        model.fit(X_train, y_train)
        return model
    raise ValueError(f"Unsupported model_type: {model_type}")
