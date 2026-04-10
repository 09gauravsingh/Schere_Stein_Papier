from __future__ import annotations

SEQ_LEN = 168
# Smaller batches often generalize better on noisy series (common LSTM practice); slower but stabler.
BATCH_SIZE = 64
MAX_EPOCHS = 150
# Lower LR + higher weight decay reduces train/val gap; pair with ReduceLROnPlateau patience below.
LR = 4e-4
# P90 head: slower LR reduces pinball instability vs point head.
P90_HEAD_LR_MULT = 0.4
WEIGHT_DECAY = 6e-4
# Slightly more capacity than 96; regularize with dropout + WD so val_peak does not explode.
HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 2
DROPOUT = 0.35
GRAD_CLIP_NORM = 0.8

# Architecture toggles — attention + deep heads regressed val (~49 vs ~37.7) in our runs; keep off unless retuning.
USE_ATTENTION = False
DEEP_HEAD = False
HEAD_DROPOUT = 0.15

# Training loss weights (sum should be 1.0). Slightly up-weight peak vs leaderboard composite (0.3 peak).
LOSS_W_ALL = 0.45
LOSS_W_PEAK = 0.35
LOSS_W_PIN = 0.2
# Smooth L1 on top-decile hours only: less brittle than pure L1 when batch peaks are noisy (MDPI / robust regression practice).
USE_SMOOTH_L1_PEAK = True
HUBER_BETA_PEAK = 0.08

# ReduceLROnPlateau: wait longer before dropping LR (avoids shrinking on single noisy val epochs).
LR_PLATEAU_PATIENCE = 7
LR_PLATEAU_FACTOR = 0.5
LR_PLATEAU_MIN = 1e-5

# Reproducibility (DataLoader shuffle)
SEED = 42

# Early stopping: patience uses EMA of val composite to ignore single-epoch spikes in val_peak.
# Best checkpoint is still saved on raw val_composite_kw improvement.
EARLY_STOP_PATIENCE = 15
EARLY_STOP_MIN_DELTA = 0.08
EARLY_STOP_EMA_ALPHA = 0.2

# Calendar split (DL): align val with January conditions; hold out late Jan for leaderboard-style local test.
# Train targets: all of CAL_TRAIN_YEAR through CAL_TRAIN_LAST_MONTH (default full 2025).
# Val targets: CAL_VAL_YEAR / CAL_VAL_MONTH / inclusive day range (default 2026-01-01 .. 01-10).
# Test targets: CAL_TEST_* / inclusive day range (default 2026-01-11 .. 01-31) — proxy for supervisor holdout.
# Scaler + y_mean/y_std still fit on all rows in train_year with month <= CAL_TRAIN_LAST_MONTH (default full 2025).
CAL_TRAIN_YEAR = 2025
CAL_TRAIN_LAST_MONTH = 12
CAL_VAL_YEAR = 2026
CAL_VAL_MONTH = 1
CAL_VAL_DAY_FIRST = 1
CAL_VAL_DAY_LAST = 10
CAL_TEST_YEAR = 2026
CAL_TEST_MONTH = 1
CAL_TEST_DAY_FIRST = 11
CAL_TEST_DAY_LAST = 31

# Entire calendar months that must not appear in DL training (targets, sequence inputs, scaler fit).
# Default: all of January 2026 (val Jan 1–10 + test Jan 11–31) stays unknown during training.
TRAIN_EXCLUDED_YEAR_MONTHS: set[tuple[int, int]] = {(2026, 1)}

# If the public CSV ends around Jan 10, the test window is empty locally; training still runs and
# metrics_test.json records test_eval_skipped=True (checkpoint is selected from val only).

# Legacy split (repro older runs): CAL_TRAIN_LAST_MONTH=11; CAL_VAL_YEAR/MONTH=2025,12;
# CAL_VAL_DAY_FIRST/LAST=1,31; CAL_TEST_DAY_FIRST/LAST=1,10; keep CAL_VAL in 2025 and test in 2026 Jan early.

# Validation strategy for checkpoint selection (DL).
# "december" — single holdout window (name is legacy): val = CAL_VAL_* + day range; test = CAL_TEST_* + day range.
# "multi_window" — late Oct, mid Nov, early Dec; checkpoint uses mean composite across windows.
# Training excludes those hours from Jan–Nov targets and adds Dec 15–31 as extra train targets.
VAL_STRATEGY = "december"
VAL_WINDOWS = [
    {"year": 2025, "month": 10, "day_first": 18, "day_last": 31},
    {"year": 2025, "month": 11, "day_first": 8, "day_last": 21},
    {"year": 2025, "month": 12, "day_first": 1, "day_last": 14},
]
# When VAL_STRATEGY == "multi_window", include these December days as train targets (not val).
MULTI_VAL_LATE_DEC_TRAIN_FROM_DAY = 15

# Backwards compatibility
EPOCHS = MAX_EPOCHS
