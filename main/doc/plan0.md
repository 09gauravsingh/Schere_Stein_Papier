Here's a clean, cursor-ready plan:

---Here is the full cursor-ready plan text to paste directly into your Cursor plan stage:

---

## Task 1 — Reefer Power Demand Forecasting: Model Build Plan

### Stage 0 — Data ingestion & parsing
- Load `reefer_release.csv` with `sep=';'`, convert all decimal commas to floats
- Parse `EventTime` → UTC, sort, aggregate to hourly
- Load weather folder (`Wetterdaten Okt 25 – 23 Feb 26`) using same parsing standard
- Load `target_timestamps.csv` — these are the only allowed inference targets

### Stage 1 — Missing data handling (before any analysis)
1. Drop rows with null `AvPowerCons` before aggregation
2. Numeric weather/temp fields → median imputation + binary missing flag
3. `HardwareType`, `ContainerSize`, `stack_tier` → fill with `"UNKNOWN"`
4. Lag/rolling features → leave NaN for warmup horizon, then drop warmup rows from train
5. Save all imputation values — apply identical logic at inference time

---

### Analysis Block 1 — Age / Wear
- Compute wear quintiles from `TtlEnergyCons` (proxy for container age/lifecycle)
- Flag visits within first 12 hours (`fresh_visit_share`) — expect ~9.6% higher power
- **Required numeric output per quintile:** count, mean W, median W, P90 W, delta vs youngest bracket

### Analysis Block 2 — Brand / Hardware
- Group by `HardwareType` — this is the strongest predictor (highest variance across families)
- **Required numeric output per hardware type:** count, mean W, P90 W, share of total load %, multiplier vs baseline hardware

### Analysis Block 3 — Ambient analysis *(complete before weather)*
- Build `effective_temp` = `TemperatureAmbient` when available, else fall back to weather temp
- Bin into temperature ranges (e.g. <0°C, 0–10°C, 10–20°C, 20–30°C, >30°C)
- **Required numeric output per bin:** mean W, P90 W, slope estimate W/°C

### Analysis Block 4 — Weather analysis *(3-month window only)*
- Restrict strictly to Oct 2025 – Feb 2026 weather data
- Create temp buckets and wind speed lag 6h feature (use as secondary only — negative correlation is confounded)
- **Required numeric output per bucket:** count, mean W, P90 W, delta vs neutral bucket

### Analysis Block 5 — Tier + Container size
- Tier: `stack_tier` → Tier1 / Tier2 / Tier3 (top tier expected ~5–6% uplift vs ground)
- Size: `ContainerSize` → 20ft / 40ft / 45ft (40ft dominant, slightly higher than 20ft)
- **Required numeric output:** count, mean W, P90 W, uplift vs Tier1 / uplift vs 20ft

---

### Stage 2 — Feature engineering
- **Time:** hour, weekday, month, cyclic sin/cos encodings
- **Lags:** 1, 2, 3, 6, 12, 24, 48, 168h
- **Rolling windows:** mean / std / max for 6h / 12h / 24h / 72h
- **Mix features (hourly aggregate):** `tier1_share`, `tier2_share`, `tier3_share`, `size_20ft_share`, `size_40ft_share`, `size_45ft_share`, `hardware_highload_share`
- **Ambient/weather:** `effective_temp`, `ambient_roll6h/12h/24h`, `ambient_delta_3h/6h/12h`, `wind_speed_lag6h`
- **Interactions:** `ambient × tier3_share`, `ambient × hour`, `ambient × hw_high`
- **Wear/lifecycle:** `avg_ttl_energy_cons` per hour, `fresh_visit_share`

### Stage 3 — Model training
- **Point model:** LightGBM or CatBoost regressor
- **Risk model:** LightGBM or CatBoost quantile regressor (q=0.9)
- **Split:** time-based only — train → validation → test (no shuffle)
- **Loss target:** `0.5 × MAE_all + 0.3 × MAE_peak + 0.2 × pinball_p90`
- **Guard:** enforce `pred_p90_kw >= pred_power_kw` at all times
- **Slice diagnostics:** by tier3 share regime, temp band, hardware high-load share, 20ft vs 40ft mix

---

### Block 6 — Final output package
**Per-hour table (24h ahead):**
`timestamp | pred_power_kw | pred_p90_kw | hardware_contribution | ambient_weather_adj | tier_size_adj`

**4 required dashboard views:**
1. Brand/hardware panel — per-family hourly contribution + risk
2. Overall EUROGATE panel — terminal total forecast + peak alerts
3. Ambient + weather panel — effective temp, wind context vs forecast
4. Tier + size panel — Tier1/2/3 and 20ft/40ft/45ft breakdown

**Submission files:**
- `predictions.csv` (matches `templates/submission_template.csv`)
- `approach.md`
- Rerunnable notebook (organizer rerun ready on hidden timestamps)
- Analysis artifact (markdown/CSV) with all 6 numeric blocks

---

> **Hard constraints:** no future leakage, non-negative predictions, one row per target timestamp, code must rerun cleanly on hidden data.