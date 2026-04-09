# Reefer Data Analysis - Results Explanation and Model Usage Guide

This document summarizes what we learned from the 4 EDA notebooks and the `main/study` outputs, and explains how to use these findings in forecasting models and dashboards.

## 1) What Was Analyzed

Sources reviewed:

- `main/data_visualization/01_data_overview.ipynb`
- `main/data_visualization/02_weather_and_load_relationships.ipynb`
- `main/data_visualization/03_container_position_and_hardware_analysis.ipynb`
- `main/data_visualization/04_model_selection_and_future_study.ipynb`
- `main/study/EXPLORATION_FINDINGS.md`
- `main/study/hourly_pattern.csv`

## 2) Core Data Conditions

- Scale is large enough for robust ML: ~3.77M hourly records.
- Coverage spans ~1 year (strong for seasonality and temporal backtesting).
- Data quality is generally good, with low missingness for critical fields.
- Important parsing note: numeric fields use decimal commas in raw CSV (`6,25`), so conversion to float must be enforced before aggregation/modeling.

## 3) Key Patterns Found

## A. Time Pattern (Strong)

- Clear hourly cycle:
  - low load around 02:00-05:00 UTC
  - peak load around 12:00-15:00 UTC
- From `hourly_pattern.csv`, mean hourly load rises from about 2232 W (04:00) to about 2538 W (14:00), a meaningful intraday swing.

Why it matters:

- `hour_of_day` is a high-value feature.
- P90 should be more conservative during midday peak window.

## B. Stack Tier / Position Effect (Important)

- Containers in higher stack tier consume more power on average.
- In findings: Tier 3 (top) is higher than Tier 1 (ground) by around 5-6%.

Your operational insight is correct and important:

- Top-most containers are more exposed (sun/wind), so cooling power rises.
- This should be reflected both in model features and dashboard outputs.

## C. HardwareType Effect (Very Strong)

- Hardware family drives large differences in average power draw.
- Effect size is larger than most other categorical features.

Why it matters:

- HardwareType should be one of the first features in every model.
- Segment-level monitoring by hardware is required in production.

## D. Container Size (20ft vs 40ft)

- 40ft containers are the dominant volume and show slightly higher average power than 20ft in the study summary.
- This is directionally useful even if effect size is smaller than hardware or tier.

How to use:

- Keep `ContainerSize` as a feature.
- Always report 20ft vs 40ft behavior in dashboard and model diagnostics.

## E. Weather Link (Moderate but Real)

- Ambient temperature has positive correlation with power (moderate).
- Other temperature channels are weaker standalone predictors.

How to use:

- Include ambient temp and lagged weather features.
- Use interactions like `ambient_temp x stack_tier`, `ambient_temp x hour`.

## F. Distribution and Peaks

- Load is right-skewed (tail behavior is significant).
- Peak management is critical for challenge scoring and business use.

How to use:

- Use quantile model (q=0.9) rather than fixed percentage uplift for `pred_p90_kw`.

## 4) Why Some Earlier Outputs Were Zero

Earlier zero-heavy/incorrect outputs were mostly caused by parsing mismatch:

- CSV delimiter is `;`, not `,`
- numeric values use decimal commas

After enforcing correct parsing, zero levels should be interpreted as actual operating states (e.g., unplugged/idle) rather than parsing artifacts.

## 5) How to Translate Findings Into Models

## Must-have Features

- temporal: hour, weekday, cyclic hour encoding
- autoregressive: lag 1/2/3/6/12/24/48/168, rolling mean/max/std
- categorical: HardwareType, ContainerSize, stack_tier
- weather: ambient temp, wind, lagged weather
- interactions: `stack_tier x hour`, `ambient x tier`, `hardware x ambient`

## Must-have Targets

- Point target: `pred_power_kw`
- Quantile target: `pred_p90_kw` (q=0.9)

## Recommended Model Set

1. LightGBM/CatBoost point model
2. LightGBM/CatBoost quantile model (q=0.9)
3. Optional sequence model (TCN or CNN-BiLSTM) for ensemble boost

## 6) Dashboard Outputs to Build

To support both hackathon and future operations:

- hourly forecast (next 24h): `pred_power_kw`, `pred_p90_kw`
- risk band: `p90 - point`
- peak-risk flags by hour
- tier-aware panel: Tier1 vs Tier2 vs Tier3 expected load contribution
- size panel: 20ft vs 40ft contribution and trend
- hardware panel: top high-load hardware families
- weather sensitivity panel: temperature/wind impact on next 24h

## 7) Recommended Validation Checks Before Final Submission

- confirm no leakage in 24h-ahead setup
- confirm no missing target timestamps
- enforce non-negative predictions
- enforce `pred_p90_kw >= pred_power_kw`
- evaluate:
  - MAE all hours
  - MAE peak hours
  - pinball loss q=0.9

## 8) Practical Conclusion

Best strategy is not a single global rule. Use a feature-rich, tier-aware, hardware-aware, weather-informed forecasting pipeline with explicit quantile modeling for P90. This matches both the leaderboard metric and the longer-term EUROGATE planning use case.
