"""
Single source of truth for paths, hyperparameters, and toggles.

Split / label configuration is IDENTICAL to the Phase 1 CNN-LSTM baseline so
that Phase 3 can compare the two models on the exact same held-out test rows.
Only the model architecture (src/model.py) and the training knobs flagged
below (USE_CLASS_WEIGHT, EPOCHS) are intentionally different from Phase 1.
"""

import os

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
DATA_PATH = "CIC-darknet2020-dataset/cicdarknet2020.parquet"

RESULTS_DIR = "results"
FIGURES_DIR = "figures"

MODEL_PATH = os.path.join(RESULTS_DIR, "transformer_lstm_model.keras")
SCALER_PATH = os.path.join(RESULTS_DIR, "scaler.joblib")
HISTORY_PATH = os.path.join(RESULTS_DIR, "history.json")
TEST_DATA_PATH = os.path.join(RESULTS_DIR, "test_data.npz")
METRICS_TXT = os.path.join(RESULTS_DIR, "metrics.txt")
METRICS_JSON = os.path.join(RESULTS_DIR, "metrics.json")
CONFUSION_PNG = os.path.join(RESULTS_DIR, "confusion_matrix.png")
CLF_REPORT_TXT = os.path.join(RESULTS_DIR, "classification_report.txt")
ACC_CURVE_PNG = os.path.join(FIGURES_DIR, "accuracy_curve.png")
LOSS_CURVE_PNG = os.path.join(FIGURES_DIR, "loss_curve.png")
ROC_CURVE_PNG = os.path.join(FIGURES_DIR, "roc_curve.png")

# --------------------------------------------------------------------------
# Model identity (used in metrics.txt / figure titles for clean comparison)
# --------------------------------------------------------------------------
MODEL_NAME = "Transformer-Enhanced LSTM"

# --------------------------------------------------------------------------
# Label configuration — IDENTICAL to Phase 1
# --------------------------------------------------------------------------
LABEL_COL = "Label"
APP_LABEL_COL = "Label.1"  # 8 application types, casing inconsistent; not used this phase

# "Non-Tor" is hyphenated; "NonVPN" is NOT — copy exactly, do not "fix" casing.
DARKNET_CLASSES = ["Tor", "VPN"]
BENIGN_CLASSES = ["Non-Tor", "NonVPN"]

CLASS_NAMES = ["Benign", "Darknet"]  # index 0, 1

# --------------------------------------------------------------------------
# Split / reproducibility — IDENTICAL to Phase 1 (do not change)
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20

# --------------------------------------------------------------------------
# Preprocessing toggle — same as Phase 1
# --------------------------------------------------------------------------
DROP_CONSTANT_COLS = True

# --------------------------------------------------------------------------
# Training — defaults tuned for better results this phase
# --------------------------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 50  # up from baseline's 30; attention models often need longer to converge

# Biggest lever on minority (Darknet) recall given the 82/18 imbalance.
# CHANGED from baseline (was False). See Fair-Comparison Note in src/train.py.
USE_CLASS_WEIGHT = True

# "bce" = binary_crossentropy (default). "focal" = tf.keras.losses.BinaryFocalCrossentropy,
# an alternative imbalance lever, documented here but off by default.
LOSS_FN = "bce"

EARLY_STOPPING_PATIENCE = 7
REDUCE_LR_PATIENCE = 4

# --------------------------------------------------------------------------
# CNN front-end (kept from baseline for lineage)
# --------------------------------------------------------------------------
CONV1_FILTERS = 128
CONV1_KERNEL = 3
CONV2_FILTERS = 64
CONV2_KERNEL = 3
POOL_SIZE = 2
CONV_PADDING = "same"

# --------------------------------------------------------------------------
# LSTM (kept from baseline)
# --------------------------------------------------------------------------
LSTM_UNITS = 100  # sequence-returning LSTM that feeds the transformer block
DROPOUT_RATE = 0.3

# --------------------------------------------------------------------------
# Transformer encoder block (the enhancement) — NEW
# --------------------------------------------------------------------------
NUM_TRANSFORMER_BLOCKS = 1  # stack more (e.g. 2) to experiment
NUM_HEADS = 4
KEY_DIM = 32  # per-head dimension
FF_DIM = 128  # feed-forward hidden size inside the block
TRANSFORMER_DROPOUT = 0.1
USE_POSITIONAL_ENCODING = False  # LSTM already encodes order; toggle on to experiment

# --------------------------------------------------------------------------
# Classification head — NEW (modest)
# --------------------------------------------------------------------------
DENSE_HEAD_UNITS = 64
