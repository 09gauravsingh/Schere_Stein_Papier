# Reefer Peak Load Hackathon Plan (24 Hours)

## Objective

Build a reproducible 24-hour-ahead hourly forecasting system for:

- `pred_power_kw` (point forecast)
- `pred_p90_kw` (upper risk estimate)

and optimize directly for the official score:

- `0.5 * mae_all + 0.3 * mae_peak + 0.2 * pinball_p90`

## Challenge Constraints (Must Respect)

From the participant package docs:

- Use only supplied files for the official submission.
- No future leakage relative to each target hour.
- Predict every timestamp in `target_timestamps.csv`.
- Ensure `pred_p90_kw >= pred_power_kw`.
- Deliver rerunnable code for organizer hidden rerun.

## 24-Hour Execution Timeline

### 0-2h: Data + Baseline

- Parse `reefer_release.csv`, weather CSVs, `target_timestamps.csv`.
- Build aggregate hourly target: sum of `Power` (kW conversion if required by notebook/data checks).
- Train simple baseline:
  - Yesterday same hour (`lag_24`)
  - Optional weekly same hour (`lag_168`)
- Set initial `pred_p90_kw = pred_power_kw * 1.10`.
- Generate first valid submission pipeline end to end.

### 2-6h: Feature Engineering

- Time features: hour, weekday, weekend, month, holiday proxy (if allowed from supplied data only).
- Autoregressive features: lags (`1, 2, 3, 6, 12, 24, 48, 168`), rolling stats (mean/max/std).
- Reefer state features (aggregated per hour):
  - connected container count
  - hardware mix ratios (`HardwareType`)
  - setpoint and ambient summary stats
  - rack occupancy proxy from `LocationRack`
- Weather features from supplied weather files:
  - temperature, wind speed, wind direction
  - lagged weather and rolling deltas

### 6-12h: Model Stack + Validation

- Validation: walk-forward backtesting (multiple folds), with peak-hour slice evaluation.
- Train model ensemble:
  1. Gradient boosting tree model (LightGBM/CatBoost/XGBoost)
  2. Sequence model (CNN-BiLSTM or TCN) for temporal patterns
  3. Optional foundation-model forecaster adapter (if time permits)
- Blend predictions using weighted average tuned on validation score.

### 12-18h: P90 / Uncertainty

- Use quantile model (q=0.9) or residual-quantile calibration:
  - `pred_p90 = pred_power + q90(residual | context)`
- Enforce monotonic rule:
  - `pred_p90 = max(pred_p90, pred_power)`
- Improve peak behavior with peak-weighted loss or sample weighting.

### 18-22h: Hardening + Reproducibility

- Single command notebook/script to regenerate predictions.
- Guardrails:
  - missing timestamps
  - negative predictions clipped to zero
  - duplicate handling
- Save artifacts (model, feature config, validation outputs).

### 22-24h: Final Submission Pack

- `predictions.csv`
- concise `approach.md`
- rerunnable notebook/code
- LLM usage notes (feature ideation, model comparison, documentation support)

## Recommended Models (Practical + Competitive)

### Tier 1 (must do)

- **LightGBM/CatBoost quantile + point models**: fast, robust under 24-hour constraints.
- **CNN-BiLSTM or TCN**: captures local temporal dynamics and nonlinear transitions.

### Tier 2 (if time and compute allow)

- **PatchTST** fine-tuning for multivariate sequence forecasting.
- **Chronos-Bolt / TimesFM style foundation model** as a zero-shot or light-tuning benchmark.

## GCP Implementation (Fastest Path)

- Data + feature pipeline in notebooks (Vertex AI Workbench) or local Python.
- Optional experiment tracking in Vertex AI Experiments.
- Store artifacts in Cloud Storage bucket.
- Optional batch inference as a simple Python job on Vertex AI custom job.

## External Data Decision

For this hackathon submission, **do not use internet datasets** because official rules specify only supplied files.

For post-hackathon productization, plan a parallel R&D branch that adds:

- high-resolution weather reanalysis/forecast
- port congestion and vessel schedule signals
- electricity market indicators (day-ahead price, peak tariffs)

## Deliverable Mapping

- Point forecast quality -> `pred_power_kw`
- Risk-aware operations -> `pred_p90_kw`
- Future optimization readiness -> retained exogenous feature interfaces and modular pipeline

## Next Document

See `main/architecture_planning.md` for full system architecture, model selection matrix, and flowcharts.