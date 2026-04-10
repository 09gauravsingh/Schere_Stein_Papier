# Project Flow (End-to-End)

This is the top-level execution guide for the ML project.

## Step-by-Step Run

1. **Prepare environment**
   - Python 3.10+ recommended
   - Install dependencies: `pandas`, `numpy`
   - Optional for model training: `catboost`, `lightgbm`, `scikit-learn`

2. **Run the end-to-end baseline**
   - Generates all 6 analysis tables + `predictions.csv`
   - Uses a simple lag-24 baseline for predictions (replace with trained models for best accuracy)
   - Command:

```
python3 -m main.src.pipeline.run_all
```

3. **Outputs**
   - `main/outputs/analysis_age_wear.csv`
   - `main/outputs/analysis_hardware.csv`
   - `main/outputs/analysis_ambient.csv`
   - `main/outputs/analysis_weather_3mo.csv`
   - `main/outputs/analysis_tier_size.csv`
   - `main/outputs/prediction_24h.csv`
   - `main/outputs/predictions.csv` (submission-ready)

4. **Model training (optional)**
   - Use `main/src/pipeline/run_train.py` to train point and quantile models.

## What the Outputs Explain

- **Age/Wear**: power shifts across wear quintiles from `TtlEnergyCons`.
- **Hardware/Brand**: consumption per `HardwareType`.
- **Ambient**: power vs ambient temperature bands.
- **Weather (3-month)**: weather-driven power only in the weather-covered period.
- **Tier + Size**: tier-level and container-size splits.
- **Overall 24h**: final hourly predictions with risk estimates.
