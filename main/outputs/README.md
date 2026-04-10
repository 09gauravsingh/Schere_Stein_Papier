# Pipeline outputs

## What each file means

| File | Purpose |
|------|---------|
| `analysis_age_wear.csv` | Mean/median/P90 power by wear quintile (`TtlEnergyCons`). |
| `analysis_hardware.csv` | Per `HardwareType` stats + share of rows + multiplier vs lowest-mean hardware. |
| `analysis_ambient.csv` | Power by ambient temperature band + slope (W/°C) within band. |
| `analysis_weather_3mo.csv` | Same bands, merged hourly weather + terminal total power (weather window only). |
| `analysis_tier_size.csv` | Power by `stack_tier` × `ContainerSize`. |
| `prediction_24h.csv` | Extended forecast table (baseline + placeholder breakdown columns). |
| `predictions.csv` | Leaderboard submission columns only. |

## Dashboard ideas

1. **Streamlit** — `st.line_chart` on `prediction_24h.csv` (`timestamp_utc` vs `pred_power_kw` / `pred_p90_kw`); `st.dataframe` for each analysis CSV; bar charts for hardware and tier×size.
2. **Plotly Dash** — multi-tab app: Forecast | Hardware | Ambient | Weather | Tier/Size.
3. **Notebook** — pandas + matplotlib/seaborn; quick for the hackathon.
4. **Grafana / BI** — load CSVs into a DB or Google Sheets and build panels.

## Breakdown columns in `prediction_24h.csv`

After `run_all`, these are filled from the **lag hour (t−24h)** slice (interpretable, not necessarily additive to `pred_power_kw`):

- `hardware_contribution` — kW from top-5 hardware types in that lag hour.
- `ambient_weather_adj` — small kW-scale adjustment from lag-hour ambient vs global median.
- `tier_size_adj` — tier-mix effect vs tier-1 baseline in that lag hour, scaled to forecast.

See `VALIDATION.md` for how to judge correctness without hidden labels.

## Figures (evaluation-aligned)

From repo root:

```bash
python3 -m main.src.reporting.visualize_outputs
```

PNG files are written to `main/outputs/figures/`, including charts for overall forecast, peak proxy, and P90 risk gap vs `EVALUATION_AND_WINNER_SELECTION.md` themes.
