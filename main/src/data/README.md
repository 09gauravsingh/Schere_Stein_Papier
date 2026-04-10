# Data Module

Loaders for all raw inputs.

## Files

- `load_reefer.py`: loads `reefer_release.csv` with semicolon delimiter and decimal-comma normalization.
- `load_weather.py`: loads and merges weather CSVs into a single hourly table.
- `load_targets.py`: loads `target_timestamps.csv`.

## Usage

Use these functions in preprocessing or pipeline scripts to ensure consistent parsing rules.
