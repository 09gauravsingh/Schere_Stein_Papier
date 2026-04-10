# Reefer Peak Load — Submission

## How to run

1. Place these two files in the `input/` folder:
   - `input/reefer_release.csv`
   - `input/target_timestamps.csv`

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run:
   ```
   python predict.py
   ```

4. Output: `output/predictions.csv` — ready to submit.

## Bundled files (do not remove)

- `segment_list.csv` — hardware/size/tier segment definitions
- `segment_model_summaries.csv` — pre-trained mixed-effects coefficients (trained on 2025 reefer data)

## Approach

- **pred_power_kw**: fleet total AvPowerCons at `t − 1h` from reefer_release, carried forward as-is.
- **pred_p90_kw**: `max(pred_power_kw, model_fleet_total)` where the model is a mixed-effects regression per segment (intercept + ambient_temp + 24h sin/cos cycle), applied using n_containers and mean `TemperatureAmbient` from reefer_release at `t − 1h`. No external weather file needed.
