# ML Pipeline Modules

This folder contains the code structure for the complete ML pipeline.

## Data Ingestion

- `data/load_reefer.py`: parses `reefer_release.csv` with `sep=';'` and decimal-comma normalization.
- `data/load_weather.py`: loads weather CSVs and merges on `timestamp_utc`.
- `data/load_targets.py`: loads `target_timestamps.csv`.

## Preprocessing

- `preprocess/cleaning.py`:
  - drops missing targets,
  - builds numeric median imputations,
  - fills categorical missing values with `UNKNOWN`,
  - adds `_missing` flags for numeric columns.
- `preprocess/aggregation.py`:
  - aggregates container rows to hourly totals,
  - computes hourly mix features (tier/size/hardware shares).
- `preprocess/feature_builder.py`:
  - time encodings, lag features, rolling windows, interactions.

## Modeling and Evaluation

- `modeling/train_point.py`: point model (CatBoost/LightGBM).
- `modeling/train_quantile.py`: q=0.9 quantile model.
- `eval/metrics.py`: MAE, pinball loss, composite score.
- `eval/splitter.py`: time-based splits and rolling backtests.

## Inference + Output

- `inference/predict_24h.py`: predicts next 24h and enforces constraints.
- `inference/submission_writer.py`: validates and writes `predictions.csv`.
- `reporting/analysis_blocks.py`: required numeric analysis blocks (age, hardware, ambient, weather, tier/size).
- `reporting/dashboard_tables.py`: dashboard output tables.

## Orchestration

- `pipeline/run_train.py`: end-to-end training pipeline (baseline implementation).
- `pipeline/run_infer.py`: inference entry point (requires trained models and features).
- `pipeline/run_all.py`: one-command runner that outputs 6 analysis tables + submission file.

## Quick Run (All Outputs)

From repo root:

```
python3 -m main.src.pipeline.run_all
```

Outputs will be written to `main/outputs/`.
