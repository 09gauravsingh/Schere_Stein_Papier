# Pipeline Module

Orchestration entrypoints.

## Files

- `run_train.py`: baseline training pipeline.
- `run_infer.py`: inference entry point (requires trained models + features).
- `run_all.py`: end-to-end runner generating analysis tables + submission output.

## Usage

From repo root:

```
python3 -m main.src.pipeline.run_all
python3 -m main.src.pipeline.run_train
python3 -m main.src.pipeline.run_infer
python3 -m main.src.pipeline.backtest_tree
```

## Supervisor handoff (tree baseline)

1. Train + save models under `main/models/tree/`.
2. Run inference to write `main/outputs/predictions_tree.csv`.
3. Optionally run backtest to print `mae_all`, `mae_peak`, `pinball_p90`, and composite score.
