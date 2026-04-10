from __future__ import annotations

import numpy as np


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mae_peak(y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.9) -> float:
    threshold = np.quantile(y_true, quantile)
    mask = y_true >= threshold
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, quantile: float = 0.9) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.maximum(quantile * diff, (quantile - 1) * diff)))


def composite_score(mae_all: float, mae_peak: float, pinball_p90: float) -> float:
    return 0.5 * mae_all + 0.3 * mae_peak + 0.2 * pinball_p90
