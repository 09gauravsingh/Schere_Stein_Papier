# Modeling Module

Model training helpers for point and quantile forecasts.

## Files

- `train_point.py`: point model training (`pred_power_kw`).
- `train_quantile.py`: quantile training (`pred_p90_kw`, q=0.9).
- `blend.py`: optional ensemble blending.

## Notes

CatBoost and LightGBM are optional dependencies. If not installed, use baseline methods or install packages before training.

Default training config (in `pipeline/run_train.py`) uses **300 boosting iterations**. This is the closest equivalent to “epochs” for tree models.
