# How to check if predictions are “true”

You **cannot** prove accuracy on the public target hours without the hidden labels. Use these checks instead.

## Structural checks (must pass)

1. **Row count** matches `target_timestamps.csv` (no missing/extra hours).
2. **No duplicates** in `timestamp_utc`.
3. **Non-negative** `pred_power_kw` and `pred_p90_kw`.
4. **`pred_p90_kw >= pred_power_kw`** for every row (leaderboard rule).

`write_submission()` enforces 3–4 already.

## Plausibility checks

1. **Variation**: point forecast should not be a single flat value (unless data truly constant); plot `prediction_24h.csv`.
2. **Risk gap**: `pred_p90_kw - pred_power_kw` should be positive; typical baselines use ~10% uplift; learned quantiles often differ by hour.
3. **Peak proxy**: many challenges weight **high-load hours**; mark hours above the 90th percentile of your own `pred_power_kw` and sanity-check they align with historically high hours (midday, warm periods).

## Backtest (strongest check you can do locally)

1. Hold out a **final time window** of historical data.
2. Build features using only past data, predict the holdout, compare to actual hourly totals.
3. Report **MAE (all)**, **MAE (peak)**, **pinball P90** — same structure as `EVALUATION_AND_WINNER_SELECTION.md`.

That is the closest analogue to “is it true?” before private scoring.
