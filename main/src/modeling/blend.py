from __future__ import annotations

import numpy as np


def blend_predictions(preds: list[np.ndarray], weights: list[float]) -> np.ndarray:
    if len(preds) != len(weights):
        raise ValueError("preds and weights must be the same length")
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()
    stacked = np.vstack(preds)
    return np.dot(weights, stacked)
