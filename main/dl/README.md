## DL Pipeline (Train / Val / Test)

This package provides a simple sequence DL baseline for comparison with tree models.

### Split strategy (calendar)

- **Scaler + target normalization (`y_mean` / `y_std`):** all hours in **2025 January through `CAL_TRAIN_LAST_MONTH`** (default **December** = full 2025).
- **Train targets:** same span (**2025 Jan–Dec** by default); each sample uses the prior `SEQ_LEN` hours from the continuous hourly table.
- **Validation targets:** **`CAL_VAL_YEAR`–`CAL_VAL_MONTH`, days `CAL_VAL_DAY_FIRST`…`CAL_VAL_DAY_LAST`** (default **2026-01-01 … 01-10**).
- **Test targets (local labeled holdout):** **`CAL_TEST_YEAR`–`CAL_TEST_MONTH`, days `CAL_TEST_DAY_FIRST`…`CAL_TEST_DAY_LAST`** (default **2026-01-11 … 01-31**), as a proxy for the supervisor’s later-January evaluation.

Constants live in `main/dl/config.py` (`CAL_*`). **`TRAIN_EXCLUDED_YEAR_MONTHS`** (default **`{(2026, 1)}`**) is enforced at train time: no row in those calendar months may appear as a train label, inside any training LSTM window, or in scaler / `y_mean` / `y_std` fit — so **all of January 2026 stays out of training** and is only used for val/test.

Optional **`VAL_STRATEGY=multi_window`** uses three 2025 windows for val and a different train mask (see config). The tree pipeline (`run_train.py`) still trains on all hourly rows unless you change it separately.

**Note:** With the default split, local **test** is **Jan 11–31**, not the same as **val** (Jan 1–10). If you repeatedly tune against `metrics_test.json`, you still leak information about Jan 11–31; only the supervisor’s private slice is truly unseen.

### Commands

From repo root:

```bash
python3 -m main.dl.train
python3 -m main.dl.infer
```

Training prints **every epoch**: normalized training loss, then **train vs validation** in **kW** (`mae_all`, `mae_peak`, `pinball_p90`, composite), learning rate, best val composite, and early-stop patience counter. **Early stopping** triggers when validation composite does not improve by more than `EARLY_STOP_MIN_DELTA` (kW) for `EARLY_STOP_PATIENCE` epochs. History is saved to `main/models/dl/training_history.json`.

Optional flags: `--max-epochs`, `--early-stop-patience`, `--early-stop-min-delta`, `--no-progress`.

Progress: each epoch shows a **batch-level** tqdm bar during training, plus an **epoch-level** bar with postfix `tr_c` / `val_c` (composite kW), `gap_c` (val−train composite), `v_ema` (smoothed val composite), `best`, `es` (early-stop patience on EMA). Full numeric rows print under the bars. Requires `tqdm` (`pip install tqdm`).

**Early stopping:** patience tracks an **EMA of `val_comp`** so single-epoch spikes in `val_peak_kw` do not reset training forever; the **saved checkpoint** still follows the best **raw** `val_comp`. See `notes.md` in the repo root.

Artifacts saved to `main/models/dl/`:

- `model_point.pt`
- `model_p90.pt`
- `scaler.npz`
- `metrics_test.json`

Inference output:

- `main/outputs/predictions_dl.csv`

### Notes

- This is intended for benchmarking against tree models using the same evaluation metrics.
- Use the lower composite score on your local holdout/backtest when choosing final submission.
- After changing `pick_sequence_columns` or `SEQ_LEN`, retrain before inference; checkpoints embed `seq_cols` and `n_features`.
