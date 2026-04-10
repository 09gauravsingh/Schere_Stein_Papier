Here is the full DL plan — architecture first, then the cursor-ready build plan below it.---

## DL model build plan — cursor-ready

### Environment setup

```
# requirements additions
torch>=2.2
numpy pandas scikit-learn
pyarrow fastparquet   # fast hourly feature reads
```

File structure to create:

```
main/
  dl/
    dataset.py        # ReeferSequenceDataset
    model.py          # CNNBiLSTMForecaster
    train.py          # training loop + backtest
    infer.py          # load saved model → predictions.csv
    config.py         # all hyperparams in one place
  outputs/dl/
    model_point.pt
    model_p90.pt
    scaler.pkl
    predictions_dl.csv
```

---

### Step 1 — `config.py`

Single source of truth for all hyperparams. Cursor should fill this first.

```python
SEQ_LEN        = 168     # 7 days of hourly history as input
PRED_HORIZON   = 24      # forecast next 24h
BATCH_SIZE     = 64
EPOCHS         = 60
LR             = 1e-3
HIDDEN         = 128
CNN_KERNELS    = [3, 6, 12]
DROPOUT        = 0.2
QUANTILE       = 0.9
LOSS_WEIGHTS   = dict(mae_all=0.5, mae_peak=0.3, pinball=0.2)
PEAK_HOURS     = list(range(9, 19))   # hours 9–18 defined as peak
TRAIN_END      = "2025-09-30"
VAL_END        = "2025-11-30"
# test = everything after VAL_END up to inference window
```

---

### Step 2 — `dataset.py` — `ReeferSequenceDataset`

What Cursor must build:

- Load the hourly aggregated feature table (output of the existing EDA pipeline — same columns as the master plan's feature list)
- For each target hour `t`, build an input tensor of shape `[SEQ_LEN, num_features]` using only rows `t-SEQ_LEN` to `t-1` (strict no-leakage)
- Target tensor: next 24 hourly `AvPowerCons` in kW (aggregated terminal total)
- Apply `StandardScaler` to numeric features; fit only on train split, transform val/test/infer with saved scaler
- Return `(x_seq, x_static, y_24h)` where `x_static` is the hourly mix vector (hardware share, tier share, size share) for hour `t`

```python
class ReeferSequenceDataset(Dataset):
    def __init__(self, df, seq_len, horizon, scaler=None, fit_scaler=False): ...
    def __getitem__(self, idx): ...   # returns x_seq, x_static, y
```

---

### Step 3 — `model.py` — `CNNBiLSTMForecaster`

Architecture exactly as diagrammed above:

```python
class CNNBiLSTMForecaster(nn.Module):
    def __init__(self, n_features, n_static, hidden=128, kernels=[3,6,12], dropout=0.2):
        # 1) parallel 1D Conv1d for each kernel size → concat → project
        # 2) BiLSTM(2 layers, hidden=128, dropout=0.2)
        # 3) concat last hidden state with x_static embedding
        # 4) shared dense: 256 → 128 → ReLU
        # 5) point_head: 128 → 64 → 24  (pred_power_kw for each of 24h)
        # 6) q90_head:   128 → 64 → 24  (pred_p90_kw for each of 24h)
    
    def forward(self, x_seq, x_static):
        ...
        return pred_point, pred_q90   # both shape [batch, 24]
```

---

### Step 4 — `train.py` — training loop

Key implementation requirements for Cursor:

```python
def combined_loss(pred_point, pred_q90, y, peak_hours, weights):
    mae_all   = F.l1_loss(pred_point, y)
    peak_mask = torch.zeros_like(y); peak_mask[:, peak_hours] = 1
    mae_peak  = (F.l1_loss(pred_point*peak_mask, y*peak_mask))
    pinball   = pinball_loss(pred_q90, y, q=0.9)
    # guard: enforce p90 >= point in loss
    p90_guard = F.relu(pred_point - pred_q90).mean()
    return weights['mae_all']*mae_all + weights['mae_peak']*mae_peak + weights['pinball']*pinball + 0.1*p90_guard

def pinball_loss(pred, target, q):
    err = target - pred
    return torch.mean(torch.max(q*err, (q-1)*err))
```

Training loop must:
- Use `AdamW` + `CosineAnnealingLR`
- Log train/val loss every epoch
- Save best checkpoint by val pinball loss (`outputs/dl/model_best.pt`)
- Run a built-in **backtest** at end of training: hold out last 4 weeks, compute `mae_all`, `mae_peak`, `pinball_p90` and print them — this is the local analogue to the leaderboard score

---

### Step 5 — `infer.py` — inference

```python
# load saved model_best.pt + scaler.pkl
# build feature table for target_timestamps.csv (same pipeline, no leakage)
# run model.forward() for each target hour
# enforce pred_p90 >= pred_power, clip negatives
# write outputs/dl/predictions_dl.csv
#   columns: timestamp, pred_power_kw, pred_p90_kw
```

Entrypoint command: `python -m main.dl.infer --target target_timestamps.csv`

---

### Step 6 — backtest validation (must-pass before submit)

Cursor must implement `eval_backtest()` inside `train.py`:

```python
def eval_backtest(model, test_loader, peak_hours):
    # returns dict: mae_all, mae_peak, pinball_p90
    # print as table — this is your local leaderboard score
```

Target thresholds to aim for before submitting DL predictions:
- `mae_all` should beat the lag-24h baseline MAE
- `mae_peak` should be lower than flat-average baseline
- `pinball_p90` should be lower than the naive 10% uplift rule currently in `run_all`

---

### Step 7 — handoff & submission wiring

- Copy `outputs/dl/predictions_dl.csv` → `predictions.csv` (rename to match template)
- Save `model_best.pt` + `scaler.pkl` alongside the notebook — organizers need both to rerun
- Add to `approach.md`: model type, SEQ_LEN, training window, backtest scores, Python + torch versions
- Rerun check: `python -m main.dl.infer` must produce identical `predictions.csv` from scratch

---

### Honest tradeoff vs tree model

The DL path is worth doing here because the data has genuine temporal structure — 168h sequences, daily cycles, and multi-factor interactions — that a sequence model can capture without manual lag engineering. The risk is training time and the need for a clean backtest before trusting it over the existing CatBoost path. Run both, compare backtest scores on the same holdout period, submit whichever wins.