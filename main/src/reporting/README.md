# Reporting Outputs

This module produces all non-submission artifacts required by the plan.

## Required Numeric Analysis Blocks (CSV/Markdown)

1. **Age/Wear** – wear quintiles from `TtlEnergyCons` with count, mean, median, P90, delta vs Q1.
2. **Brand/Hardware** – per `HardwareType`: count, mean, P90, share, multiplier vs baseline.
3. **Ambient** – binned ambient/effective temperature ranges with count, mean, P90, slope.
4. **Weather (3-month only)** – weather temp and wind buckets with count, mean, P90, delta vs neutral.
5. **Tier + Size** – Tier1/2/3 and 20ft/40ft/45ft splits with count, mean, P90, uplift.
6. **Overall 24h Prediction** – hourly table with `pred_power_kw`, `pred_p90_kw`, and adjustment columns.

## Dashboard Tables

- Hardware/brand contribution panel
- Overall EUROGATE hourly forecast panel
- Ambient + weather context panel
- Tier + container size panel

## Submission Outputs (outside this module)

`inference/submission_writer.py` writes `predictions.csv` with the required columns.
