REEFER TERMINAL POWER FORECASTING

Model Training Blueprint

EDA Insights, Feature Engineering & Architecture Guide

Data Period

Jan 2025 - Feb 2026

Records

3.77M Reefer Telemetry

Targets

pred_power_kw + pred_p90_kw

Prepared: April 9, 2026 | 13-Hour Hackathon Sprint | Daytona Sandbox Pipeline

1. Executive Summary
The Single Most Important Finding

Hardware type dominates power consumption. Temperature, tier, and wear are strong secondary signals. Time-of-day and container lifecycle complete the picture. Any model that ignores hardware will fail.

This document consolidates all EDA findings into a complete, implementation-ready guide for the 13-hour training sprint. Three independent analyses feed into the final model design:

Weather & Ambient Analysis: external temperature r=0.80 with power (aggregate), but individual response r=0.17 (weak). Reefer ambient sensors are a perfect proxy for weather (r=0.9973) and available for 100% of training data.
Container Lifecycle Analysis: fresh containers consume ~9.6% more power in first 12 hours (stabilisation effect). Heavily-worn containers draw 172% more power than new ones (TtlEnergyCons as wear proxy, r=0.30).
Hardware & Tier Dominance: hardware family and stack tier are the strongest individual-level predictors. All other signals (weather, ambient, hour) amplify or modulate hardware baseline.
Outputs to Produce
Output

Loss Function

Model

Constraint

pred_power_kw

MAE / Huber (peak-weighted)

LightGBM + XGBoost ensemble

Non-negative

pred_p90_kw

Pinball loss q=0.9

CatBoost quantile

pred_p90 >= pred_power

2. Key Data Patterns & Insights
2.1 Hardware Type — Primary Driver
Hardware family is the strongest individual-level predictor of power consumption. The differences between hardware types are large, stable, and persistent across all temperature bands and time periods.

Action

Always encode hardware type as a primary feature. Use target-encoding or leave-one-out encoding since CatBoost handles categoricals natively. LightGBM/XGBoost require ordinal or frequency encoding.

Create binary flags for high-load hardware families (top 20th percentile mean power)
Interaction term: hardware_highload_share x hour_of_day
Interaction term: hardware_highload_share x tier3_share
2.2 Stack Tier — Secondary Driver
Stack tier (Tier1/Tier2/Tier3) has a moderate but consistent effect. Top-tier containers run hotter due to reduced airflow and heat accumulation from lower tiers.

Tier

Relative Power

Note

Tier 1 (ground)

Baseline

Best airflow, lowest load

Tier 2 (middle)

+5-8% vs T1

Moderate heat accumulation

Tier 3 (top)

+12-18% vs T1

Highest load; no shade from above

Feature: tier3_share = fraction of active containers in tier 3 per hour
Interaction: ambient_temp x tier3_share (heat amplification at top tier)
2.3 Time of Day — Diurnal Cycle
A strong hourly cycle exists in aggregate power. Midday hours show higher load; early morning hours show minimum load. This is driven partly by temperature (ambient tracks diurnal cycle) and partly by terminal operations patterns.

Hour Band

Load Level

Interpretation

00:00 - 06:00

Low (trough)

Coldest ambient + minimum arrivals

07:00 - 11:00

Rising

Morning warm-up + new arrivals

11:00 - 16:00

Peak window

Max ambient temp + peak operations

17:00 - 23:00

Declining

Cooling + stabilising containers

Feature: cyclic hour encoding — sin(2*pi*hour/24) and cos(2*pi*hour/24)
Binary flag: is_peak_hour (hours 10-16 inclusive)
Sample weight multiplier: 2.0 for peak hours during training
2.4 Container Size Distribution
Container size mix (20ft / 40ft / 45ft) affects aggregate power. 40ft containers draw more total power but this is largely captured by count. The proportion matters when the mix shifts across the day.

Feature: size_40ft_share, size_20ft_share, size_45ft_share per hour
These are flow-level features — compute at hourly aggregation before training
3. Ambient Sensors as Weather Proxy
Key Finding: r=0.9973

The reefer ambient sensors and external weather station measure the same thermal environment at near-perfect agreement. Ambient is available for 100% of training data; external weather only covers 35%.

3.1 Why Use Ambient Instead of External Weather
Signal

Coverage

Correlation with Power

Use Case

External Weather Temp

35% of data (Oct-Jan)

r=0.80 (aggregate)

24h-ahead inference only

Reefer Ambient Sensors

100% of data

r=0.17 (individual), r=0.80 agg

Training + same-day inference

effective_temp (unified)

100% of data

Same as above

Always use this column

3.2 The Aggregate vs Individual Paradox
The apparent contradiction between r=0.80 (weather vs power at hourly level) and r=0.17 (ambient vs individual container power) is explained by aggregation:

At the hourly aggregate level: all containers' power is summed AND weather shifts for everyone simultaneously — correlation amplifies
At the individual container level: hardware type dominates; a worn high-load unit at 5°C draws more power than a new low-load unit at 25°C
The r=0.80 is real but it is a macro trend, not a container-level causal mechanism
Practical Implication

Do not expect ambient temperature to be a top feature in SHAP analysis. It will rank below hardware, tier, and hour. Its value is in the rolling/lag features and interaction terms, not raw temperature.

3.3 Recommended Ambient Feature Engineering
# Step 1: Build unified effective_temp column

df['effective_temp'] = df['reefer_ambient_avg'].fillna(df['weather_temp'])

# Step 2: Rolling thermal context (cumulative heat load)

df['ambient_roll6h_mean'] = df['effective_temp'].shift(1).rolling(6).mean()

df['ambient_roll12h_mean'] = df['effective_temp'].shift(1).rolling(12).mean()

df['ambient_roll24h_mean'] = df['effective_temp'].shift(1).rolling(24).mean()

# Step 3: Delta features (warming vs cooling trend)

df['ambient_delta_3h'] = df['effective_temp'] - df['effective_temp'].shift(3)

df['ambient_delta_6h'] = df['effective_temp'] - df['effective_temp'].shift(6)

df['ambient_delta_12h'] = df['effective_temp'] - df['effective_temp'].shift(12)

# Step 4: Interaction terms

df['ambient_x_tier3'] = df['effective_temp'] * df['tier3_share']

df['ambient_x_hour'] = df['effective_temp'] * df['hour']

df['ambient_x_hw_high'] = df['effective_temp'] * df['hardware_highload_share']

Note on wind speed: the observed negative correlation with power (-0.21) is a confounding artifact — high wind accompanies low-pressure systems with colder air. Wind speed has no direct causal effect on reefer power and should be used cautiously (only as a secondary lag feature if at all).

4. Container Lifecycle & Wear Effects
4.1 Three Independent Effects
Effect

Correlation

Magnitude

Actionable?

Container wear (TtlEnergyCons)

r=+0.2987

+172% range from new to worn

Yes — for P90

Visit lifecycle (time into visit)

N/A (decay)

-9.6% from hour 0 to stable

Yes — first 12h correction

Visit duration (total stay length)

r=-0.0015

None

No — ignore

4.2 Wear Effect: TtlEnergyCons as Degradation Proxy
Cumulative energy consumption across a container's lifetime is a proxy for mechanical wear. Heavily used containers (compressor degradation, insulation failure) draw dramatically more power:

Wear Quintile

Avg Power

Multiplier vs Quintile 1

Q1 (least used / newest)

1,478 W

1.0x (baseline)

Q2

2,048 W

1.39x

Q3

~2,400 W

~1.62x

Q4

~3,100 W

~2.10x

Q5 (most worn)

4,028 W

2.72x

P90 Implication

Worn containers (Q4-Q5) are disproportionately responsible for peak power events. Incorporating TtlEnergyCons quantile as a feature directly improves P90 calibration and coverage.

4.3 Visit Lifecycle: First-12-Hour Effect
When a container first connects to shore power, it is still cooling down (or warming up) to its set point. This creates a predictable decay in power over the first 12 hours of each visit:

Time into Visit

Avg Power

vs Stable State

0-12h (arrival phase)

2,540 W

+7.0% above stable

12-24h (stabilising)

2,433 W

+2.5% above stable

24-36h

2,396 W

+1.0% above stable

72h+ (plateau)

2,374 W

Stable baseline

Container behavior split:

52% of containers show decreasing power over their visit (normal stabilisation)
31% stay stable (already at equilibrium on arrival)
17% show increasing power (warming from cold storage, or degradation events)
4.4 Wear & Lifecycle Feature Engineering
# WEAR FEATURES (computed at container level, then aggregated hourly)

df['wear_quintile'] = pd.qcut(df['TtlEnergyCons'], q=5, labels=[1,2,3,4,5]).astype(int)

df['is_high_wear'] = (df['wear_quintile'] >= 4).astype(int) # Q4+Q5

df['wear_norm'] = df['TtlEnergyCons'] / df['TtlEnergyCons'].max()

# HOURLY AGGREGATION: wear mix at terminal level

hourly['high_wear_share'] = high_wear_containers / total_active_containers

hourly['mean_wear_quintile']= mean of wear_quintile across active containers

# LIFECYCLE FEATURES (per container, then aggregated)

df['hours_into_visit'] = (df['timestamp'] - df['visit_start']).dt.total_seconds() / 3600

df['is_fresh_arrival'] = (df['hours_into_visit'] < 12).astype(int)

df['lifecycle_decay'] = np.exp(-df['hours_into_visit'] / 12) # exponential decay proxy

# HOURLY AGGREGATION: arrival load at terminal level

hourly['fresh_arrival_share'] = fresh_arrival_containers / total_active_containers

5. Complete Feature Store Schema
This is the definitive feature list for the hourly-level training table. Features are ordered by expected importance (SHAP-estimated).

Group A — Lag & Autoregressive Features (Highest Importance)
Feature

Formula

Why

power_lag_1h

total_power[t-1]

Strong autocorrelation; best single predictor

power_lag_2h

total_power[t-2]

Captures short-cycle inertia

power_lag_3h

total_power[t-3]

power_lag_6h

total_power[t-6]

Half-day pattern

power_lag_12h

total_power[t-12]

Half-day mirror

power_lag_24h

total_power[t-24]

Same-hour yesterday

power_lag_48h

total_power[t-48]

Two-day seasonality

power_lag_168h

total_power[t-168]

Same-hour last week

power_roll6h_mean

mean(t-6..t-1)

Short-window trend

power_roll12h_mean

mean(t-12..t-1)

power_roll24h_mean

mean(t-24..t-1)

Daily average context

power_roll6h_std

std(t-6..t-1)

Volatility signal for P90

power_roll24h_max

max(t-24..t-1)

Peak context for P90

Group B — Hardware & Fleet Composition
Feature

Description

hw_scc6_share

Fraction of active containers with SCC6 hardware

hw_ml3_share

Fraction with ML3 hardware

hw_decosva_share

Fraction with DecosVa hardware

hw_decosvb_share

Fraction with DecosVb hardware

hw_mp4000_share

Fraction with MP4000 hardware

hardware_highload_share

Fraction in top-20th-percentile hardware families

size_40ft_share

Fraction of 40ft containers

size_20ft_share

Fraction of 20ft containers

size_45ft_share

Fraction of 45ft containers

active_container_count

Total plugged-in containers at hour t

Group C — Tier Distribution
Feature

Description

tier1_share

Fraction of containers in stack tier 1

tier2_share

Fraction in tier 2

tier3_share

Fraction in tier 3 (highest heat accumulation)

tier3_count

Absolute count of tier-3 containers

Group D — Temporal & Calendar Features
Feature

Description

hour_sin

sin(2*pi*hour/24) — cyclic hour encoding

hour_cos

cos(2*pi*hour/24)

weekday_sin

sin(2*pi*weekday/7)

weekday_cos

cos(2*pi*weekday/7)

month_sin

sin(2*pi*month/12)

month_cos

cos(2*pi*month/12)

is_peak_hour

Binary: hours 10-16 inclusive

is_weekend

Binary: Saturday or Sunday

hour_of_day

Raw integer 0-23 (redundant with cyclic but useful for trees)

Group E — Ambient / Thermal Features
Feature

Description

effective_temp

Reefer ambient avg, fallback to external weather

ambient_roll6h_mean

6h trailing mean of effective_temp

ambient_roll12h_mean

12h trailing mean

ambient_roll24h_mean

24h trailing mean

ambient_delta_3h

Temperature change over last 3 hours (trend)

ambient_delta_6h

Temperature change over last 6 hours

ambient_x_tier3

Interaction: effective_temp * tier3_share

ambient_x_hour

Interaction: effective_temp * hour_of_day

ambient_x_hw_high

Interaction: effective_temp * hardware_highload_share

wind_speed_lag6h

6h lagged wind speed (confounded; use with caution)

Group F — Wear & Lifecycle Features
Feature

Description

high_wear_share

Fraction of active containers in wear quintiles Q4-Q5

mean_wear_quintile

Mean wear quintile across active containers (1-5)

fresh_arrival_share

Fraction of containers within first 12h of visit

lifecycle_decay_mean

Mean exp(-hours_into_visit/12) across containers

6. Model Architecture & Training Strategy
6.1 Model Stack (Priority Order for 13-Hour Sprint)
Priority

Model

Target

Library

Time Budget

1 (Must)

LightGBM Regressor (MAE)

pred_power_kw

lightgbm

Hours 3-5

1 (Must)

XGBoost Regressor (MAE)

pred_power_kw

xgboost

Hours 5-6

1 (Must)

CatBoost Quantile q=0.9

pred_p90_kw

catboost

Hours 6-7

2 (If time)

Prophet + XGBoost ensemble

pred_power_kw

prophet + xgboost

Hours 9-10

2 (If time)

TFT (Temporal Fusion Transformer)

pred_power_kw

pytorch-forecasting

Hours 9-11

3 (Final)

Weighted ensemble blend

Both targets

numpy / sklearn

Hours 11-12

6.2 Rolling Backtest Validation Structure
Use strict time-based splits. No random splits. Each fold must respect 24h-ahead prediction constraint.

Fold 1: Train [Jan 2025 → Aug 2025] Validate [Sep 2025]

Fold 2: Train [Jan 2025 → Sep 2025] Validate [Oct 2025]

Fold 3: Train [Jan 2025 → Oct 2025] Validate [Nov 2025]

Holdout: Train [Jan 2025 → Nov 2025] Test [Dec 2025 → Jan 2026]

Composite Score = 0.5 * MAE_all + 0.3 * MAE_peak + 0.2 * Pinball_P90

6.3 LightGBM Configuration (Point Forecast)
params_lgbm = {

'objective': 'mae',

'n_estimators': 3000,

'learning_rate': 0.05,

'max_depth': 7,

'num_leaves': 63,

'min_child_samples': 50,

'subsample': 0.8,

'colsample_bytree': 0.8,

'reg_alpha': 0.1,

'reg_lambda': 0.1,

'random_state': 42,

}

# Peak-hour sample weights

weights = np.where(df['is_peak_hour'], 2.0, 1.0)

model.fit(X_train, y_train, sample_weight=weights,

eval_set=[(X_val, y_val)], callbacks=[early_stopping(100)])

6.4 CatBoost Quantile Configuration (P90)
params_catboost = {

'loss_function': 'Quantile:alpha=0.9',

'iterations': 2000,

'learning_rate': 0.05,

'depth': 7,

'l2_leaf_reg': 3.0,

'random_seed': 42,

'cat_features': ['hardware_type', 'container_size'], # native categorical

'early_stopping_rounds': 100,

}

# Always enforce constraint after prediction:

pred_p90 = np.maximum(pred_p90_raw, pred_power_kw)

6.5 Ensemble Blend
# Inverse-MAE weighting from validation folds

w_lgbm = 1 / mae_lgbm_val

w_xgb = 1 / mae_xgb_val

w_total = w_lgbm + w_xgb

pred_power_kw = (w_lgbm * pred_lgbm + w_xgb * pred_xgb) / w_total

# P90 blend (quantile models only)

pred_p90_kw = np.maximum(pred_catboost_q90, pred_power_kw)

7. Daytona Sandbox Setup
7.1 Installation & Workspace Creation
# 1. Install Daytona CLI

curl -sf -L https://download.daytona.io/daytona/install.sh | sudo bash

daytona server -y

# 2. Create workspace from your repo

daytona create https://github.com/YOUR_USER/reefer-forecast

# 3. Or use Python SDK programmatically

pip install daytona

from daytona import Daytona, DaytonaConfig

daytona = Daytona(DaytonaConfig(api_key='YOUR_API_KEY'))

sandbox = daytona.create()

7.2 Project Structure
reefer-forecast/

├── .devcontainer/devcontainer.json

├── data/

│ ├── raw/ ← Drop 2GB CSVs here

│ └── processed/ ← Parquet feature store

├── src/

│ ├── 01_parse_and_features.py

│ ├── 02_train_lgbm_xgb.py

│ ├── 03_train_catboost_quantile.py

│ ├── 04_train_tft.py ← Optional

│ ├── 05_ensemble_blend.py

│ └── 06_submit.py

├── models/ ← Saved .pkl / .cbm artifacts

├── outputs/submission.csv

├── requirements.txt

└── run_all.sh ← Single-command pipeline

7.3 13-Hour Sprint Schedule
Hour

Task

Script

Success Criteria

0-1

Daytona setup + data upload + convert CSVs to Parquet

Manual

data/processed/*.parquet ready

1-3

Feature engineering pipeline

01_parse_and_features.py

~60 features, zero nulls

3-5

LightGBM + XGBoost training + Optuna tuning

02_train_lgbm_xgb.py

Val MAE < baseline

5-7

CatBoost quantile training + P90 calibration

03_train_catboost_quantile.py

P90 coverage ~90%

7-9

SHAP analysis + feature debugging + slice metrics

Notebook

Hardware SHAP > weather SHAP

9-11

TFT (optional) or Prophet+XGB if time allows

04_train_tft.py

Beat LightGBM on val MAE

11-12

Ensemble blend + constraint enforcement

05_ensemble_blend.py

p90 >= power for all rows

12-13

Submission generation + QA + export

06_submit.py

Row count match, no nulls

7.4 Performance Tips for 2GB Data
Convert semicolon-delimited CSVs to Parquet in step 1 — reduces I/O by ~5x for all subsequent runs
Use float32 instead of float64 for all numeric features — halves memory footprint
Train LightGBM with n_jobs=-1 — parallelise across all available cores
Use CatBoost's built-in early stopping to avoid manual epoch tuning
Cache feature engineering output; never recompute from raw CSV after step 1
8. Data Parsing & Quality Gates
8.1 Critical Parsing Rules
Do Not Skip This

The raw CSVs use semicolons as delimiters and commas as decimal separators. Failing to handle this will silently corrupt all numeric features.

df = pd.read_csv('reefer_release.csv',

sep=';',

decimal=',',

parse_dates=['Timestamp'],

low_memory=False)

# Normalise to UTC

df['timestamp'] = pd.to_datetime(df['Timestamp'], utc=True)

# Aggregate to hourly buckets

df['hour_bucket'] = df['timestamp'].dt.floor('H')

8.2 Pipeline Quality Gates
Gate

Check

Action on Failure

Missing data

< 5% nulls in any core numeric column

Abort + report column names

Timestamp gaps

No gaps > 2h in hourly series

Log warning; interpolate if < 4h

Zero-rate jump

Zero power rate < 2x baseline period

Investigate before training

Feature count

Exactly N features expected in feature store

Abort if mismatch

P90 constraint

pred_p90 >= pred_power for all rows

Enforce with np.maximum post-hoc

9. Evaluation Framework
9.1 Primary Metrics
Metric

Formula

Target

MAE all hours

mean(|pred - actual|)

Minimise; baseline = naive lag-24h

MAE peak hours

mean(|pred - actual|) for top-10% actual load hours

0.3x weight in composite

Pinball loss q=0.9

mean(max(0.9*(y-yhat), 0.1*(yhat-y)))

Minimise; coverage should be ~90%

P90 coverage

fraction of hours where actual <= pred_p90

Target 90% ± 2%

Composite score

0.5*MAE_all + 0.3*MAE_peak + 0.2*Pinball_P90

Primary selection criterion

9.2 Slice Metrics (Must Monitor)
By stack tier share regime: low tier3_share vs high tier3_share hours
By container size mix: 20ft-dominant vs 40ft-dominant hours
By hardware mix: high_wear_share > 0.4 vs < 0.2
By ambient temperature band: cold (<5°C), mild (5-15°C), warm (>15°C)
By wear regime: high fresh_arrival_share hours (terminal influx events)
Watch For: Systematic P90 Under-coverage on Worn-Fleet Hours

The wear effect (TtlEnergyCons) is the most likely source of P90 miscalibration. If high_wear_share > 0.4, check that pred_p90 coverage stays near 90%. Add a post-hoc calibration layer if coverage drops below 85% in this slice.

10. Pre-Training Checklist
Data & Features
CSVs parsed with sep=';', decimal=','
All timestamps normalised to UTC and aggregated to hourly buckets
effective_temp column built (ambient first, weather fallback)
All lag features use .shift(1) — no look-ahead leakage
Wear quintile computed per container, aggregated to hourly share
fresh_arrival_share computed from visit start timestamps
All interaction terms created (ambient x tier3, ambient x hw_high, etc.)
Feature store saved as Parquet before any model training
Model Training
Rolling folds defined: no random splits, strict time ordering
Peak-hour sample weights set to 2.0 for point forecast training
LightGBM: MAE objective, early stopping on validation composite
CatBoost: Quantile:alpha=0.9 objective, hardware/size as cat_features
Deterministic seeds set (random_state=42 everywhere)
Model artifacts saved to models/ after each fold
Submission & QA
pred_power_kw >= 0 for all rows (non-negative constraint)
pred_p90_kw >= pred_power_kw for all rows (P90 dominance constraint)
Row count matches target_timestamps.csv exactly
No nulls in submission columns
Slice metrics checked for worn-fleet P90 coverage
End of Document | Reefer Terminal Power Forecasting Blueprint | April 2026