"""
Single source of truth for paths, hyperparameters, and toggles.

Split / label configuration for the binary task is IDENTICAL to the Phase 1
CNN-LSTM baseline so that Phase 3 can compare the two models on the exact
same held-out test rows. This model also supports application-type
classification (8 classes from Label.1) and a hierarchical 4-class task
(Tor/VPN/Non-Tor/NonVPN from Label, reported at both the 4-class and the
grouped-binary level) -- selected via TASK or the --task CLI flag in
train.py / evaluate.py.

Output paths are resolved per task via get_paths(task), not as flat module
constants, because the task can be overridden at the CLI after this module
is imported -- a flat MODEL_PATH etc. would go stale the moment --task
differs from TASK below.
"""

import os
from types import SimpleNamespace

# --------------------------------------------------------------------------
# Task switch
# --------------------------------------------------------------------------
TASK = "binary"  # "binary" | "application" | "fourclass". CLI --task overrides this.

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
DATA_PATH = "CIC-darknet2020-dataset/cicdarknet2020.parquet"

RESULTS_DIR = "results"
FIGURES_DIR = "figures"


def get_paths(task: str) -> SimpleNamespace:
    """All output paths for `task`, isolated under results/<task>/ and figures/<task>/."""
    results_dir = os.path.join(RESULTS_DIR, task)
    figures_dir = os.path.join(FIGURES_DIR, task)
    return SimpleNamespace(
        RESULTS_DIR=results_dir,
        FIGURES_DIR=figures_dir,
        MODEL_PATH=os.path.join(results_dir, "transformer_lstm_model.keras"),
        SCALER_PATH=os.path.join(results_dir, "scaler.joblib"),
        HISTORY_PATH=os.path.join(results_dir, "history.json"),
        DATA_SPLITS_PATH=os.path.join(results_dir, "data_splits.npz"),
        CLASS_NAMES_JSON=os.path.join(results_dir, "class_names.json"),
        METRICS_TXT=os.path.join(results_dir, "metrics.txt"),
        METRICS_JSON=os.path.join(results_dir, "metrics.json"),
        CONFUSION_PNG=os.path.join(results_dir, "confusion_matrix.png"),
        CLF_REPORT_TXT=os.path.join(results_dir, "classification_report.txt"),
        ACC_CURVE_PNG=os.path.join(figures_dir, "accuracy_curve.png"),
        LOSS_CURVE_PNG=os.path.join(figures_dir, "loss_curve.png"),
        ROC_CURVE_PNG=os.path.join(figures_dir, "roc_curve.png"),
    )


# --------------------------------------------------------------------------
# Model identity (used in metrics.txt / figure titles for clean comparison)
# --------------------------------------------------------------------------
MODEL_NAME = "Transformer-Enhanced LSTM"

# --------------------------------------------------------------------------
# Label configuration
# --------------------------------------------------------------------------
LABEL_COL = "Label"
APP_LABEL_COL = "Label.1"

# Binary task -- IDENTICAL to Phase 1.
# "Non-Tor" is hyphenated; "NonVPN" is NOT — copy exactly, do not "fix" casing.
DARKNET_CLASSES = ["Tor", "VPN"]
BENIGN_CLASSES = ["Non-Tor", "NonVPN"]
CLASS_NAMES = ["Benign", "Darknet"]  # index 0, 1

# Application task -- Label.1 has casing variants that collapse to 8 canonical
# classes. Variants not listed here map to themselves (already canonical).
APP_LABEL_CANON = {
    "AUDIO-STREAMING": "Audio-Streaming",
    "Audio-Streaming": "Audio-Streaming",
    "Video-streaming": "Video-Streaming",
    "Video-Streaming": "Video-Streaming",
    "File-transfer": "File-Transfer",
    "File-Transfer": "File-Transfer",
    # Browsing, P2P, Chat, Email, VOIP map to themselves
}
# APP_CLASS_NAMES is intentionally not hardcoded here -- preprocessing.py
# derives it (sorted unique, post-normalization) and persists it to
# class_names.json so evaluate.py uses a consistent class order.

# Fourclass task -- the 4 underlying classes behind the binary grouping
# (Label has no casing issues, unlike Label.1, so no normalization map is
# needed). Same column as LABEL_COL; named separately for clarity in the
# fourclass code path.
FOURCLASS_LABEL_COL = "Label"
# FOURCLASS_NAMES is intentionally not hardcoded here, same reasoning as
# APP_CLASS_NAMES -- preprocessing.py LabelEncodes Label (sorted unique:
# Non-Tor, NonVPN, Tor, VPN) and persists the order to class_names.json.

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
# Training
# --------------------------------------------------------------------------
BATCH_SIZE = 64
EPOCHS = 50  # EarlyStopping cuts it short if it plateaus; used for both tasks

# Binary: default is now a MATCHED comparison with the Phase 1 baseline
# (class weights off). The operating point is instead improved via
# validation-set threshold tuning (see TUNE_THRESHOLD) rather than by
# skewing the loss with class weights. Flip True to experiment.
USE_CLASS_WEIGHT = False

# Application: 8 classes with ~10x imbalance (Browsing vs VOIP) -- class
# weights are a reasonable default here. Computed for all 8 classes via
# sklearn compute_class_weight("balanced", ...) in train.py.
APP_USE_CLASS_WEIGHT = True

# Fourclass: Tor is ~1.1% of rows (~55x imbalance vs Non-Tor) -- class
# weights are enabled by default here too. Computed for all 4 classes via
# sklearn compute_class_weight("balanced", ...) in train.py.
FOURCLASS_USE_CLASS_WEIGHT = True

# Single lookup train.py reads to resolve the active task's class-weight
# setting, instead of an if/elif chain. Binary and application still come
# from their own flags above (unchanged); only the fourclass entry is new.
CLASS_WEIGHT_BY_TASK = {
    "binary": USE_CLASS_WEIGHT,
    "application": APP_USE_CLASS_WEIGHT,
    "fourclass": FOURCLASS_USE_CLASS_WEIGHT,
}

# Binary only: sweep thresholds on the validation set and pick the one that
# maximizes F1, instead of blindly using 0.5. evaluate.py reports both.
# Fourclass uses argmax (no threshold), same as application.
TUNE_THRESHOLD = True

# "bce" = binary_crossentropy (default, binary task only).
# "focal" = tf.keras.losses.BinaryFocalCrossentropy, an alternative
# imbalance lever, documented here but off by default.
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
# LSTM
# --------------------------------------------------------------------------
LSTM_UNITS = 100  # sequence-returning LSTM that feeds the transformer block
DROPOUT_RATE = 0.3
USE_BILSTM = False  # wrap the LSTM in Bidirectional when True (doubles its output dim)

# --------------------------------------------------------------------------
# Transformer encoder block (the enhancement)
# --------------------------------------------------------------------------
NUM_TRANSFORMER_BLOCKS = 2  # bumped from 1 for a bit more capacity
NUM_HEADS = 4
KEY_DIM = 32  # per-head dimension
FF_DIM = 128  # feed-forward hidden size inside the block
TRANSFORMER_DROPOUT = 0.1
USE_POSITIONAL_ENCODING = False  # LSTM already encodes order; toggle on to experiment

# --------------------------------------------------------------------------
# Classification head
# --------------------------------------------------------------------------
DENSE_HEAD_UNITS = 64
