# Master Project Plan (Consolidated)

This is the single source of truth for the hackathon project plan. It consolidates participant-package rules, evaluation criteria, and all internal analysis findings (EDA + blueprint + weather analysis).

## 1) Challenge Objective (Canonical)

Predict **hourly aggregate reefer power demand** 24 hours ahead for each timestamp in `target_timestamps.csv` with:

- `pred_power_kw` (best point forecast)
- `pred_p90_kw` (upper-risk estimate)

Final result package for the next 24 hours (hourly) must provide 4 views:

1. **Brand/Hardware-based predictions** (different container control hardware families)
2. **Overall EUROGATE requirement view** (terminal total hourly forecast + risk)
3. **Ambient + weather conditioned view** (temperature/wind context)
4. **Tier + container-size view** (Tier1/Tier2/Tier3 and 20ft/40ft/45ft breakdown)

## 2) Evaluation Criteria (Must Appear in Final Output)

From `EVALUATION_AND_WINNER_SELECTION.md` (lines 5–9):

> Your live leaderboard submission is evaluated on three things over the released public target hours:
> 1. overall forecast accuracy
> 2. accuracy during high-load hours
> 3. quality of your upper-risk estimate `pred_p90_kw`

Scoring formula:

`0.5 * mae_all + 0.3 * mae_peak + 0.2 * pinball_p90`

## 3) Rules and Constraints (Must-Pass Checklist)

- Use **only the supplied files** for official submission.
- Strict 24-hour-ahead forecasting: no future leakage.
- Submit one prediction per timestamp in `target_timestamps.csv`.
- Predictions must be **numeric, non-negative**.
- Enforce **`pred_p90_kw >= pred_power_kw`**.
- Code must be **re-runnable** by organizers on hidden timestamps.

## 4) Data Sources and Parsing

### Primary Data Inputs

- `participant_package/reefer_release.csv`
- `participant_package/Wetterdaten Okt 25 - 23 Feb 26/`
- `participant_package/target_timestamps.csv`

### Parsing Standard

- CSV delimiter is `;`
- Numeric fields use decimal commas (e.g., `6,25`) → convert to floats
- Normalize all timestamps to UTC and aggregate hourly

### Canonical Column Mapping (Actual Schema)

| Concept | Actual Column |
|---|---|
| Container visit | `container_visit_uuid` |
| Container | `container_uuid` |
| Hardware type | `HardwareType` |
| Event hour | `EventTime` |
| Average power (W) | `AvPowerCons` |
| Total energy per hour | `TtlEnergyConsHour` |
| Lifetime energy | `TtlEnergyCons` |
| Temp setpoint | `TemperatureSetPoint` |
| Temp ambient | `TemperatureAmbient` |
| Temp return | `TemperatureReturn` |
| Temp supply | `RemperatureSupply` |
| Container size | `ContainerSize` |
| Stack tier | `stack_tier` |

## 5) Key Findings Integrated Into Modeling

### Dominant Drivers

- **Hardware type**: strongest predictor (very large variance between families).
- **Stack tier**: top-tier containers consume ~5–6% more than ground-tier.
- **Hourly cycle**: midday peak, night low.

### Weather and Ambient

- **Weather vs ambient**: highly correlated at aggregate level, weak at individual level.
- **Effective temperature**: use `ambient` first, fallback to `weather` when needed.
- **Wind**: observed negative correlation likely confounded → use as secondary lag feature only.

### Container Size

- 40ft is dominant and slightly higher power than 20ft.
- Include size mix as features and dashboard output.

### Lifecycle/Wear (From Blueprint)

- First ~12 hours of visit: higher power (~9.6% stabilization effect).
- `TtlEnergyCons` acts as wear proxy (r≈0.30, up to ~172% spread).

## 6) Feature Engineering Plan

### Core Features

- Time: hour, weekday, month, cyclic encodings
- Lags: 1,2,3,6,12,24,48,168
- Rolling windows: mean/std/max for 6h/12h/24h/72h

### Aggregated Mix Features (hourly)

- `tier1_share`, `tier2_share`, `tier3_share`
- `size_20ft_share`, `size_40ft_share`, `size_45ft_share`
- `hardware_highload_share`

### Weather/Ambient Features

- `effective_temp` (ambient if available, else weather)
- `ambient_roll6h/12h/24h`
- `ambient_delta_3h/6h/12h`
- `wind_speed_lag6h` (optional, low weight)

### Interaction Features

- `ambient_x_tier3_share`
- `ambient_x_hour`
- `ambient_x_hw_high`

### Wear/Lifecycle Features

- `avg_ttl_energy_cons` per hour
- `fresh_visit_share` (visits within first 12h)

## 6A) Missing Data Handling (Before Model Training)

Yes, missing-data handling is required before training. Use this sequence:

1. **Schema and parsing checks**
   - enforce `sep=';'`
   - convert decimal commas to numeric
   - parse timestamps and sort by hour
2. **Mandatory target quality checks**
   - drop rows with missing target (`AvPowerCons`) before aggregation
   - validate hourly aggregation completeness
3. **Feature-level strategy**
   - numeric weather/temperature features: median imputation + missing flags
   - lag/rolling features: keep NaN for warmup horizon, then drop warmup rows in train
   - categorical (`HardwareType`, `ContainerSize`, `stack_tier`): fill with `"UNKNOWN"`
4. **Coverage safeguards**
   - monitor missing rate by time window
   - alert if drift in missingness exceeds threshold
5. **Inference consistency**
   - apply the same imputation logic and saved preprocessing steps at prediction time

## 7) Train/Validation/Test Protocol

Use **time-based splits only**:

- Train: earliest period
- Validation: next contiguous block
- Test: latest contiguous block before inference

Add **rolling backtests** to simulate the hidden rerun.

## 8) Model Stack

### Primary

- LightGBM/CatBoost point regressor
- LightGBM/CatBoost quantile model (q=0.9)

### Secondary

- Sequence model (TCN or CNN-BiLSTM) as optional ensemble

### Quantile Guard

`pred_p90_kw = max(pred_p90_kw, pred_power_kw)`

## 9) Evaluation and Validation

Optimize for:

- MAE (all)
- MAE (peak hours)
- Pinball loss (q=0.9)

Slice diagnostics:

- by tier3 share regime
- by temperature band
- by hardware high-load share
- by 20ft vs 40ft mix

## 10) Dashboard Output Contract

Required operational outputs:

- 24h forecast table: `pred_power_kw`, `pred_p90_kw`, risk gap
- Peak risk flags by hour
- Hardware/brand panel: per-hardware-family hourly contribution and risk
- Overall EUROGATE panel: terminal total hourly forecast and peak alerts
- Weather-conditioned panel: ambient, weather, wind context vs forecast
- Tier panel: Tier1/Tier2/Tier3 contribution
- Container-size panel: 20ft vs 40ft vs 45ft contribution and risk profile

## 10A) Mandatory Final Model Output Block (Task-1 Sequence)

In addition to `pred_power_kw` and `pred_p90_kw`, the final model output package must include a structured numeric analysis in this exact sequence:

1. **Age Analysis (Wear-based)**
   - Use wear/age brackets (e.g., wear quintiles from `TtlEnergyCons`).
   - Report per bracket: count, mean power, median power, P90 power, delta vs youngest bracket.
2. **Brand / Hardware Analysis**
   - Report per `HardwareType`: count, mean power, P90 power, share of total load, relative multiplier vs baseline hardware.
3. **Ambient Analysis (before weather)**
   - Bin ambient/effective temperature into ranges.
   - Report per range: mean power, P90 power, and slope estimate (W per °C where applicable).
4. **Weather Analysis (3-month window only)**
   - Restrict to the weather-covered period only.
   - Report per weather bucket (temp and wind): count, mean power, P90 power, and deltas vs neutral bucket.
5. **Tier + Container Size Analysis**
   - Tier results: Tier1/Tier2/Tier3 count, mean power, P90, uplift vs Tier1.
   - Container size results: 20ft/40ft/45ft count, mean power, P90, uplift vs 20ft.
6. **Overall Prediction**
   - Final next-24h hourly table combining all above factors:
     - timestamp
     - `pred_power_kw`
     - `pred_p90_kw`
     - hardware/brand contribution
     - ambient/weather-conditioned adjustment
     - tier + size adjustment

### Numeric-Only Requirement

All six blocks must contain concrete numeric outputs (counts, means, percent changes, uplifts, multipliers, or slopes). Avoid qualitative-only summaries.

## 11) Submission Package

Must include:

- `predictions.csv` (matching `templates/submission_template.csv`)
- `approach.md` (short method summary)
- rerunnable notebook or script (organizer rerun ready)
- analysis output artifact (markdown/csv bundle) containing the 6 mandatory numeric blocks above

## 12) Final Notes

This plan aligns directly with the evaluation criteria and the data evidence. It prioritizes strong peak performance and calibrated P90 estimates while keeping the pipeline reproducible and ready for organizer reruns.
