# Approach — Reefer Peak Load Forecasting

## How We Used Claude

This solution was built end-to-end with Claude (Anthropic, claude-sonnet-4-6) as an interactive data science and coding assistant. The workflow was conversational: we described goals in natural language, Claude read the data and existing code, proposed and implemented solutions, and we refined iteratively based on results and metrics. All Python scripts were written by Claude.

---

## Phase 1: Exploratory Data Analysis (Claude as research assistant)

**Goal:** Understand what drives aggregate reefer power consumption.

We asked Claude to analyse `reefer_release.csv` (3.77M hourly rows, 375 days). Claude autonomously chose appropriate statistical methods — ANOVA eta² for categorical factors, Pearson r for continuous — without being told which to use.

**Key findings Claude surfaced:**

- **HardwareType** explains 23% of power variance — the dominant structural driver
- **Ambient temperature** is weak globally (0.38%) but strong within specific hardware segments (r = 0.73+ for high-power types like MP4000 and ML3)
- **Stack tier** is near-zero globally but meaningful for certain type-size combinations
- No wear signal: longer plug-in duration shows no positive correlation with consumption

Claude produced 11 visualisations and 4 documentation files across 25+ code executions. When a request was ambiguous ("show variation over 24h"), Claude provided both interpretations — power variation within segments and segment-share stability — without needing clarification.

---

## Phase 2: Segmentation (61 segments)

**Goal:** Define modelling segments that capture the dominant structure.

We instructed Claude to split by tier only where |r(tier, power)| > 0.1 at the daily aggregation level. Claude:

1. Computed daily-level tier correlations per hardware × size combination (avoiding sample-size bias from hourly data)
2. Built conditional segment logic (split/unsplit based on threshold)
3. Enumerated and exported all 61 resulting segments with metadata
4. Flagged 2 segments with <2 containers as edge cases

**Result:** 24 base segments → 61 after tier splits, with a machine-readable `segment_list.csv`.

---

## Phase 3: Mixed-Effects Model (Claude as ML engineer)

**Goal:** Train a per-segment model capturing temperature and time-of-day effects.

Claude implemented one **Random Intercept Mixed Effects Model** per segment using statsmodels (REML, LBFGS):

```
AvPowerCons ~ β₀ + β₁·ambient_temp + β₂·sin(2π·h/24) + β₃·cos(2π·h/24) + u_container
u_container ~ N(0, σ²_u)
```

59/61 models trained successfully on 2025 data. The 2 skipped segments were constant-filled from their last observed value.

Claude then evaluated the models on January 2026 data, computed MAE/RMSE/MAPE per day and per segment, and generated time series comparison plots.

---

## Phase 4: Baseline Discovery (Claude identified the better model)

**The turning point.** We asked Claude to compute a simple constant baseline — carry each container's last known consumption forward — and compare it to the segment model.

Claude wrote `baseline_comparison.py` and reported:

| Hour | Constant baseline MAE | Segment model MAE |
|------|-----------------------|-------------------|
| Jan 1 | 11.2 kW | 95.9 kW |

The constant baseline was ~8× better. Claude explained why: fleet composition and individual consumption levels are highly autocorrelated hour-to-hour; the model's extrapolation noise exceeds the signal it captures over 24 hours.

**This finding reoriented the entire strategy.** We switched to the constant as the point forecast.

---

## Phase 5: P90 Estimation — Iterative Refinement

**Goal:** A `pred_p90_kw` that is meaningfully better than a fixed uplift, minimising pinball loss.

Claude went through four distinct iterations:

**Iteration 1 — Per-container residual P90:**
Claude computed P90 from individual container residuals and multiplied by fleet count. Result: P90 = pred × 1.286, 100% coverage, grossly over-conservative. Claude diagnosed the root cause: per-container noise (std ~171 W) multiplied across 600+ containers inflates variance 600×, violating the central limit theorem that applies at the aggregate level.

**Iteration 2 — Aggregate residual P90:**
Claude shifted to aggregate (total kW) residuals. Better, but still 100% coverage at +211 kW margin.

**Iteration 3 — Ratio-based P90:**
Claude reasoned that since the model over-predicts 68% of hours, an additive offset compounded the bias. Instead: `P90 = pred × ratio_p90` where the ratio is calibrated from actual/pred on 2025 training data. Claude tested six variants (flat ratio, by-hour ratio, recent 3-month, additive, pct-error). Best: hour-stratified ratio at 55.6 kW combined score.

**Iteration 4 — Pinball loss analysis:**
When we challenged whether the P90 was still too high, Claude derived the asymmetric pinball formula directly:

- Overshooting (p90 > actual): penalised at 0.1× weight
- Undershooting (p90 < actual): penalised at 0.9× weight

Claude swept a range of multipliers and found `pred × 1.05` optimal at **53.56 kW combined** with 86% coverage. At this point, the mixed-effects model total became the natural P90 signal instead of a ratio multiplier.

**Final P90 structure:**

```
pred_p90_kw(t) = max(pred_power_kw(t), model_fleet_total(t))
```

The model provides the upper bound only when it has an upward signal (temperature or time-of-day effect). This avoids blanket padding while retaining the structural constraint `pred_p90_kw >= pred_power_kw`.

---

## Phase 6: Production Pipeline

**Goal:** Self-contained, reproducible submission folder.

Claude:

- Replaced the external weather file with `TemperatureAmbient` from `reefer_release` itself (averaged across all containers at `t−1h`), eliminating the external file dependency
- Switched from a frozen Dec-31 snapshot to rolling `t−1h` lookups for both container counts and fleet totals — dropping MAE from 55.9 kW to 13.9 kW
- Pre-serialised model coefficients to `segment_model_summaries.csv`, cutting runtime from ~105s (retrain) to ~22s
- Verified all four submission format checks programmatically

---

## Final Model

**Point forecast:**

```
pred_power_kw(t) = sum(AvPowerCons for all containers at t−1h) / 1000
```

**P90 upper estimate:**

```
pred_p90_kw(t) = max(pred_power_kw(t), model_fleet_total(t))
model_fleet_total(t) = Σ_segments [ (β₀ + β₁·temp(t−1h) + β₂·sin24 + β₃·cos24) × n_containers(t−1h) ] / 1000
```

**Evaluation on public 223-hour window (Jan 1–10 2026):**

| Metric | Value | Weight | Contribution |
|--------|-------|--------|-------------|
| `mae_all` | 13.9 kW | 0.5 | 6.95 |
| `mae_peak` | — | 0.3 | — |
| `pinball_p90` | — | 0.2 | — |

---

## What Claude Did vs. What We Decided

| Claude | Us |
|--------|-----|
| Wrote all Python scripts | Chose to optimise point forecast over model complexity |
| Chose statistical methods autonomously | Decided the `max(constant, model)` structure for p90 |
| Identified the 8× better constant baseline | Approved switching strategy mid-way |
| Diagnosed per-container noise inflation | Decided acceptable P90 coverage level |
| Swept multiplier space, found optimal P90 | Confirmed final submission parameters |
| Built self-contained submission folder | Defined folder and input/output structure |

---

## Reproducibility

```bash
pip install -r requirements.txt
python predict.py
```

Runtime: ~22 seconds. Organizers can substitute the hidden full `target_timestamps.csv` and complete `reefer_release.csv` into `input/` — the script reads both and writes `output/predictions.csv`.

**Submission checks (verified):**

| Check | Result |
|-------|--------|
| All target timestamps present | PASS (223/223) |
| No duplicate timestamps | PASS |
| `pred_power_kw` non-negative | PASS |
| `pred_p90_kw >= pred_power_kw` | PASS |
