"""
Single source of truth for paths, hyperparameters, and toggles.

Split / label configuration for the binary task is IDENTICAL to the Phase 1
CNN-LSTM baseline so that Phase 3 can compare the two models on the exact
same held-out test rows. This model also supports application-type
classification (8 classes from Label.1) and a hierarchical 4-class task
(Tor/VPN/Non-Tor/NonVPN from Label, reported at both the 4-class and the
grouped-binary level) -- selected via TASK or the --task CLI flag in
train.py / evaluate.py.

Phase 3 re-targets the dataset-facing layer to CIC-IDS2017 (dhoogla cleaned
parquet, split across 8 day/attack-type files under DATA_DIR) with three
new tasks -- "ids_binary" (Benign vs any attack), "ids_family" (coarse
attack families), "ids_multi" (fine attack types, rare classes folded or
dropped) -- without touching the model architecture below. The old
CIC-Darknet2020 constants and branches are left in place (inert).

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
# "binary" | "application" | "fourclass" -> CIC-Darknet2020 (Phase 2, inert
#   locally now that the dataset file has been removed from the repo).
# "ids_binary" | "ids_family" | "ids_multi" -> CIC-IDS2017 (Phase 3, current).
# CLI --task overrides this.
TASK = "ids_binary"

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
# CIC-IDS2017 (Phase 3) -- ids_binary / ids_family / ids_multi tasks
# --------------------------------------------------------------------------
# Dataset: dhoogla/cicids2017 (Kaggle), "no-metadata" cleaned parquet, split
# across 8 day/attack-type files. DATA_DIR is globbed for "*.parquet" and
# every file is concatenated -- see preprocessing._load_ids2017(). No Flow
# ID / IP / port / timestamp columns are present in this build (verified by
# Step 1 discovery), unlike the raw CICIDS2017 CSVs.
DATA_DIR = "CIC-IDS2017"

# Exact column name confirmed via Step 1 discovery (no surrounding
# whitespace). Named separately from LABEL_COL above (which is the darknet
# dataset's `Label` column) purely for clarity -- the two never coexist in
# the same load_and_prepare() call, since `task` selects which file(s) get
# loaded in the first place.
IDS_LABEL_COL = "Label"

# No identifier/leakage columns were found in Step 1 (this "no-metadata"
# build already excludes Flow ID / IPs / ports / timestamp) -- left empty,
# kept for parity with the DROP_COLS hook called out in the task brief.
IDS_DROP_COLS = []

# Exact benign string discovered in Step 1 -- note it is "Benign", NOT
# "BENIGN" (the all-caps form below is only the ids_family bucket name).
IDS_BENIGN_LABEL = "Benign"
IDS_BINARY_CLASS_NAMES = ["Benign", "Attack"]  # index 0, 1

# ids_multi rare-class threshold: a fine-grained class with fewer rows than
# this (after merging the three Web Attack variants) is folded into its
# ids_family bucket if a same-family sibling survives at >= this count in
# the fine label set, otherwise the rows are dropped outright (folding a
# singleton-family class into itself doesn't fix the small-N problem -- see
# preprocessing._build_ids_multi_target()).
MIN_CLASS_COUNT = 100

# ids_family coarse-bucket rules, matched against the exact Label strings
# found in Step 1 (preprocessing._map_family() applies these):
#   - IDS_BENIGN_LABEL ("Benign", exact)        -> "BENIGN"
#   - label containing "DoS" (this also catches "DDoS", since the substring
#     "DoS" occurs inside "DDoS")                -> "DoS/DDoS"
#   - "FTP-Patator", "SSH-Patator" (exact)       -> "Brute-Force"
#   - label starting with "Web Attack"           -> "Web-Attack" (the
#     discovered strings have a corrupted dash byte, U+FFFD, between
#     "Web Attack" and the variant name -- matching the clean ASCII prefix
#     sidesteps that entirely)
#   - "Bot" (exact)                              -> "Botnet"
#   - "PortScan" (exact)                         -> "PortScan"
#   - "Infiltration" (exact)                     -> "Infiltration"
#   - "Heartbleed" (exact)                       -> "Other" (only 11 rows in
#     the combined dataset -- too rare to stratify as its own family)
#   - anything else (unexpected / unmapped)      -> "Other"
FAMILY_DOS_SUBSTR = "DoS"
FAMILY_WEBATTACK_PREFIX = "Web Attack"
FAMILY_EXACT = {
    "Benign": "BENIGN",
    "FTP-Patator": "Brute-Force",
    "SSH-Patator": "Brute-Force",
    "Bot": "Botnet",
    "PortScan": "PortScan",
    "Infiltration": "Infiltration",
    "Heartbleed": "Other",
}
FAMILY_DEFAULT = "Other"

# Single clean class name the three Web Attack variants merge into for the
# ids_multi (fine) task -- ASCII-only, sidesteps the corrupted-dash strings.
WEBATTACK_MERGED_NAME = "Web Attack"

# Optional stratified majority-only downsample for faster Colab runs on this
# ~2.2M-row (post-cleaning) dataset: every minority-class row is kept, only
# the majority (benign) rows are downsampled to hit this total. None = use
# all rows. If Colab training is slow, try 400_000-600_000.
SUBSAMPLE_N = None

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

# ids_binary: ~5.6x imbalance (Benign vs any-attack), milder than the
# darknet binary task -- kept matched to the same "no class weight, tune
# the threshold instead" philosophy as the darknet binary task above, so
# the two binary tasks are handled identically.
IDS_BINARY_USE_CLASS_WEIGHT = False

# ids_family / ids_multi: heavy imbalance (Benign dominates; Infiltration/
# Heartbleed-derived buckets are tiny) -- class weights are a reasonable
# default here, same reasoning as application/fourclass above.
IDS_FAMILY_USE_CLASS_WEIGHT = True
IDS_MULTI_USE_CLASS_WEIGHT = True

# Single lookup train.py reads to resolve the active task's class-weight
# setting, instead of an if/elif chain. Each task still comes from its own
# flag above (unchanged); this dict just centralizes the lookup.
CLASS_WEIGHT_BY_TASK = {
    "binary": USE_CLASS_WEIGHT,
    "application": APP_USE_CLASS_WEIGHT,
    "fourclass": FOURCLASS_USE_CLASS_WEIGHT,
    "ids_binary": IDS_BINARY_USE_CLASS_WEIGHT,
    "ids_family": IDS_FAMILY_USE_CLASS_WEIGHT,
    "ids_multi": IDS_MULTI_USE_CLASS_WEIGHT,
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
