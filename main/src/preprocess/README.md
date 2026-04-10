# Preprocess Module

Cleaning, aggregation, and feature engineering utilities.

## Files

- `cleaning.py`: missing-data policy, imputation config, target dropping.
- `aggregation.py`: container-level to hourly aggregation and mix features.
- `feature_builder.py`: time encodings, lag/rolling features, interactions.

## Notes

Apply the same imputation policy at training and inference time.
