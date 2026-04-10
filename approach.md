---
# Approach — Reefer Load Forecasting

## Problem framing
Hourly point forecast (pred_power_kw) and upper-bound estimate
(pred_p90_kw) of aggregate reefer electricity demand, predicting
the next hour given the last 168 hours of features.

## Feature engineering (89 features, all lagged to avoid leakage)

**Container mix** (from reefer release data):
- Tier share by cargo type: frozen / chilled / ambient — lag 1, 6, 24
- Hardware brand share: top 5 brands + other — lag 1, 6, 24  
- Container size share: 20ft vs 40ft — lag 1, 6, 24
- Connected reefer count — lag 1, 6, 24, rolling means
- Average ambient and setpoint temperature of connected 
  reefers — lag 1, 6, 24 (also kept as raw current-hour feature)

**Weather** (two station sensors):
- Air temperature — lag 1, 6, 24
- Wind speed — lag 1, 6, 24
- Wind direction — lag 1, 6, 24

**Time**: hour of day, day of week, month, is_weekend

## Model architecture
Implemented in `main/dl/model.py` (`LSTMForecaster`), controlled by `USE_ATTENTION` and
`DEEP_HEAD` in `main/dl/config.py` (both **True** for current training runs).

- Sequence input: last 168 timesteps × ~89 numeric features
- LSTM: 2 layers, `hidden_size` from config (default **96**), dropout between layers
- **When `USE_ATTENTION=True`:** scaled dot-product attention over all LSTM timesteps
  (query = last-layer hidden state), context concatenated with that hidden state,
  projected to `hidden_size` + LayerNorm, then the prediction head(s)
- **When `USE_ATTENTION=False`:** last timestep output only (plain LSTM baseline)
- Two heads: point forecast and P90 (pinball); **when `DEEP_HEAD=True`**, each head is a
  small MLP (Linear → LayerNorm → ReLU → Dropout → Linear). **When `DEEP_HEAD=False`**,
  each head is the original shallow stack (Linear → ReLU → Dropout → Linear)

Earlier experiments used `USE_ATTENTION=False` and `DEEP_HEAD=False` (plain LSTM only);
retrain after changing flags — old `checkpoint_best.pt` weights are not compatible across
architecture changes.

## Training
- Loss (configurable in `main/dl/config.py`): default **0.45×L1_all + 0.35×peak + 0.2×pinball_p90**,
  where **peak** is the mean error on the **top decile of y in each batch**; optionally **Smooth L1**
  on that peak subset (more robust than pure L1 when peaks are noisy). Leaderboard metric remains
  **0.5×MAE + 0.3×MAE_peak + 0.2×pinball** on val — training weights are chosen to stress peaks slightly more.
- Optimizer: **AdamW** with **separate LRs** for point vs P90 head; **weight decay** and **dropout**
  tuned for generalization (smaller batch size, lower base LR than 1e-3 is common in LSTM forecasting notes).
- Gradient clipping, **ReduceLROnPlateau** on val composite (patience from config), **EMA early stopping**
  on val composite while the saved checkpoint tracks **best raw** val composite.

**Expectations:** Moving val composite from ~38 kW toward **25–30 kW** may need more than hyperparameters
(features, ensembles, or different inductive bias). The profile above targets **stabler peaks** and a
**smaller train/val gap**, not a guaranteed leaderboard tier.

## Validation strategy
Train: full 2025 (Jan–Dec), 7844 target hours
Val:   Jan 1–10 2026, 223 hours
This directly optimizes the checkpoint for January conditions.

## Key findings
- December val produced inflated scores (63.6 kW composite)
  because December load is structurally harder than January
- January val immediately gave 37.7 kW composite — 40% better
- Temperature is the strongest weather signal
- Frozen cargo share (tier3) drives the largest load variation
