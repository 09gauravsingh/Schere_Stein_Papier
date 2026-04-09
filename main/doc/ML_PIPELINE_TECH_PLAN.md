# End-to-End ML Pipeline Technical Plan (Based on EDA Results)

This plan converts analysis insights into an implementation-ready forecasting pipeline for:

- `pred_power_kw` (point)
- `pred_p90_kw` (q=0.9)

aligned to challenge scoring and future operational usage.

## 1) Pipeline Objectives

Primary:

- accurate next-24h hourly load forecasts
- robust peak-hour behavior
- calibrated upper estimate (P90)

Secondary:

- provide explainable outputs for operations dashboard
- keep architecture extensible for future external signals

## 2) Data Pipeline Design

## 2.1 Inputs

- Reefer telemetry (`reefer_release.csv`)
- Provided weather files
- Target hours (`target_timestamps.csv`)

## 2.2 Parsing and Standardization (Critical)

- delimiter: semicolon (`;`)
- decimal conversion: comma to dot for numeric columns
- datetime normalization:
  - parse all times to UTC
  - aggregate to hourly buckets

## 2.3 Feature Store Schema (hourly level)

Target table at hour `t`:

- target: total power at hour `t` (for training only)
- temporal: hour, weekday, month, cyclic encodings
- lags: t-1, t-2, t-3, t-6, t-12, t-24, t-48, t-168
- rolling windows: mean/std/max for 6h/12h/24h/72h
- categorical distributions:
  - HardwareType proportions
  - ContainerSize proportions (20ft/40ft/45ft)
  - stack_tier proportions (Tier1/Tier2/Tier3)
- weather:
  - ambient temp, wind speed, wind direction
  - lagged weather and rolling weather deltas
- interaction features:
  - ambient x tier3_share
  - ambient x hour
  - hardware_highload_share x hour

## 2.4 Data Quality Gates

Fail pipeline if:

- more than threshold missing in core numeric columns
- unexpected timestamp gaps
- abnormal zero-rate jumps vs baseline period

## 3) Train / Validation / Test Strategy

Use strict time-based splits (no random split).

Recommended structure:

1. **Train window:** earliest period through `T_train_end`
2. **Validation window:** next contiguous block
3. **Test window (offline holdout):** latest contiguous block before final inference

For robustness, use rolling backtests:

- Fold 1: train [A..B], validate [B+1..C]
- Fold 2: train [A..C], validate [C+1..D]
- Fold 3: train [A..D], validate [D+1..E]

Each fold must emulate 24h-ahead prediction constraints.

## 4) Modeling Stack

## 4.1 Point Forecast Model (`pred_power_kw`)

Primary:

- LightGBM/CatBoost regressor on engineered hourly features

Secondary:

- TCN or CNN-BiLSTM sequence model (optional ensemble member)

## 4.2 P90 Model (`pred_p90_kw`)

Preferred:

- direct quantile model at q=0.9 (LightGBM/CatBoost quantile objective)

Fallback:

- residual quantile calibration layer:
  - `pred_p90 = pred_point + q90(residual | context_bucket)`

Always enforce:

- `pred_p90 = max(pred_p90, pred_power)`

## 4.3 Loss and Optimization

Point model:

- optimize MAE (or Huber) with peak-hour sample weighting

Quantile model:

- optimize pinball loss at q=0.9

Composite model selection criterion (offline):

- `0.5 * mae_all + 0.3 * mae_peak + 0.2 * pinball_p90`

## 4.4 Backpropagation Context

For deep sequence models (TCN/CNN-BiLSTM):

- use standard minibatch gradient descent + backpropagation through network layers
- early stopping on validation composite score
- gradient clipping to stabilize spikes
- learning-rate schedule (cosine/step decay)

Note:

- GBDT models do not use neural backpropagation; they optimize tree splits via boosting iterations.

## 5) Peak-Focused Design (from Findings)

Given EDA:

- top stack tier increases load
- midday is higher load window
- certain hardware families dominate peaks

Implement:

- explicit peak-hour indicator features
- tier and hardware interaction terms
- weighted penalty for errors in top decile of true load
- monitor P90 coverage by tier/hardware bucket

## 6) Evaluation Framework

For each fold and final holdout:

- MAE all hours
- MAE peak hours (>= 90th percentile true load)
- pinball loss q=0.9
- calibration:
  - coverage of p90 (should be near 90% under calibration target)

Slice metrics:

- by stack tier share regime
- by container size mix (20ft-heavy vs 40ft-heavy)
- by hardware mix
- by ambient temperature band

## 7) Inference and Submission Flow

1. Build features for all target timestamps using only historical data available before each target hour.
2. Generate point and quantile predictions.
3. Apply constraints:
   - non-negative
   - `pred_p90 >= pred_power`
4. Export exact submission schema.
5. Run validation checks (row count, duplicates, numeric checks).

## 8) Dashboard/Serving Output Contract

For each hour in next 24h:

- timestamp
- predicted load (`pred_power_kw`)
- risk load (`pred_p90_kw`)
- risk gap (`pred_p90_kw - pred_power_kw`)
- peak flag
- decomposition panels:
  - tier contribution (T1/T2/T3)
  - size contribution (20ft/40ft)
  - hardware high-load contribution

## 9) MLOps and Reproducibility

- versioned feature code
- saved model artifacts + feature list
- deterministic seeds
- run metadata log:
  - data window
  - metrics
  - hyperparameters
- single command/notebook to regenerate predictions end to end

## 10) Immediate Execution Checklist

1. Finalize cleaned feature pipeline with decimal-comma handling.
2. Implement rolling split backtesting.
3. Train point + q0.9 models.
4. Add peak-weighting and interaction features.
5. Validate by operational slices (tier, size, hardware, temperature).
6. Freeze best model config and export reproducible inference notebook/script.

This plan is optimized for hackathon delivery while preserving a direct path to production-grade forecasting for terminal planning.
