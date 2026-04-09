# Data Visualization Workspace

This folder contains exploratory notebooks to understand reefer load behavior, weather impact, and modeling options.

## Notebooks

1. `01_data_overview.ipynb`
   - Data shape and date coverage
   - Missing value checks
   - Aggregate load trend and hour/day seasonality

2. `02_weather_and_load_relationships.ipynb`
   - Merge weather files with aggregate load
   - Correlation heatmaps and scatter plots
   - Lagged weather sensitivity checks

3. `03_container_position_and_hardware_analysis.ipynb`
   - Position (`LocationRack`) contribution analysis
   - Hardware and container-size load behavior
   - Peak-hour cohort summary

4. `04_model_selection_and_future_study.ipynb`
   - Quick benchmark baselines (lag + tree)
   - p90 quantile proxy and peak-hour error checks
   - Future model queue for deeper experiments

## How to Use

- Open notebooks in order from 1 to 4.
- Run cells top to bottom.
- If weather CSV column names differ, adjust the parsing logic in notebook 2.

## Suggested Next Steps

- Add walk-forward validation visual diagnostics.
- Add SHAP feature attribution charts for tree models.
- Add interactive Plotly dashboards once core EDA findings are stable.
