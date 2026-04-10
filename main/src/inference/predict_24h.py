from __future__ import annotations

import numpy as np
import pandas as pd


def predict_next_24h(
    model_point,
    model_p90,
    features_df: pd.DataFrame,
    timestamp_col: str = "timestamp_utc",
) -> pd.DataFrame:
    """
    Generate next-24h predictions and enforce constraints.
    """
    x = features_df.drop(columns=[timestamp_col], errors="ignore")
    preds_point = model_point.predict(x)
    preds_p90 = model_p90.predict(x)

    preds_point = np.maximum(preds_point, 0)
    preds_p90 = np.maximum(preds_p90, preds_point)

    out = pd.DataFrame(
        {
            "timestamp_utc": features_df[timestamp_col].values,
            "pred_power_kw": preds_point,
            "pred_p90_kw": preds_p90,
        }
    )
    return out
