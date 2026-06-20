"""
Preprocessing pipeline for CIC-Darknet2020, supporting two tasks:

- "binary": Darknet vs Benign from `Label`. Logically IDENTICAL to the
  original Phase 1 baseline pipeline -- same label mapping, same
  constant-column drop, same stratified split with the same RANDOM_STATE,
  same StandardScaler-fit-on-train-only policy. This guarantees the
  held-out test set is the exact same rows as Phase 1, the precondition
  for a fair Phase 3 comparison.
- "application": 8-class application type from `Label.1`, after collapsing
  casing variants to canonical names (config.APP_LABEL_CANON).

Both tasks share identical cleaning / constant-column-drop / scaling /
reshape / split mechanics -- they only differ in how the target `y` (and
which label columns get dropped from `X`) is constructed.

Import-safe: nothing in this module trains a model. The __main__ demo at
the bottom only loads data and prints summary stats (pandas / sklearn only).
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib


def _build_binary_target(df: pd.DataFrame, config):
    """Darknet (1) vs Benign (0) from `Label`. Same mapping as Phase 1."""
    darknet_mask = df[config.LABEL_COL].isin(config.DARKNET_CLASSES)
    benign_mask = df[config.LABEL_COL].isin(config.BENIGN_CLASSES)
    unmapped = ~(darknet_mask | benign_mask)
    if unmapped.any():
        bad_values = df.loc[unmapped, config.LABEL_COL].unique().tolist()
        raise AssertionError(
            f"Found {int(unmapped.sum())} rows with unmapped Label values: {bad_values}"
        )
    y = darknet_mask.astype(int)

    # Label.1 (8 application types) needs casing normalization before it could ever be used.
    drop_cols = [config.LABEL_COL]
    if config.APP_LABEL_COL in df.columns:
        drop_cols.append(config.APP_LABEL_COL)
    X = df.drop(columns=drop_cols)
    return y, X, config.CLASS_NAMES


def _build_application_target(df: pd.DataFrame, config):
    """8-class application type from `Label.1`, after canonicalizing casing variants."""
    normalized = (
        df[config.APP_LABEL_COL].astype(str).str.strip().map(lambda v: config.APP_LABEL_CANON.get(v, v))
    )
    n_unique = normalized.nunique()
    if n_unique != 8:
        raise AssertionError(
            f"Expected exactly 8 application classes after normalization, "
            f"got {n_unique}: {sorted(normalized.unique())}"
        )

    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(normalized), index=normalized.index)
    class_names = encoder.classes_.tolist()

    drop_cols = [config.LABEL_COL, config.APP_LABEL_COL]
    X = df.drop(columns=drop_cols)
    return y, X, class_names


def load_and_prepare(config, task: str = None) -> dict:
    """Load the parquet, build the target for `task`, clean, split, and scale.

    `task` defaults to config.TASK ("binary" or "application").

    Returns a dict with X_train, y_train, X_val, y_val, X_test, y_test
    (each X reshaped to (n_samples, n_features, 1)), plus n_features,
    n_classes, feature_names, dropped_cols, and (application only)
    class_names. The application branch also persists class_names.json
    under the task's results dir so evaluate.py uses a consistent order.
    """
    if task is None:
        task = config.TASK
    if task not in ("binary", "application"):
        raise ValueError(f"Unknown task: {task!r}")

    paths = config.get_paths(task)
    os.makedirs(paths.RESULTS_DIR, exist_ok=True)

    # 1. Load
    df = pd.read_parquet(config.DATA_PATH)

    # 2. Build target + drop label columns from X (task-specific)
    if task == "binary":
        y, X, class_names = _build_binary_target(df, config)
    else:
        y, X, class_names = _build_application_target(df, config)
    n_classes = len(class_names)

    # 3. Defensive cleaning (expect ~0 changes on this pre-cleaned dataset)
    n_before = len(X)
    X = X.replace([np.inf, -np.inf], np.nan)
    inf_replaced = int(X.isna().sum().sum())
    print(f"Replaced inf/-inf with NaN in {inf_replaced} cells")

    nan_mask = X.isna().any(axis=1)
    n_nan_rows = int(nan_mask.sum())
    X = X.loc[~nan_mask]
    y = y.loc[X.index]
    print(f"Dropped {n_nan_rows} rows containing NaN")

    dup_mask = X.duplicated()
    n_dup_rows = int(dup_mask.sum())
    X = X.loc[~dup_mask]
    y = y.loc[X.index]
    print(f"Dropped {n_dup_rows} duplicate rows")
    print(f"Rows: {n_before} -> {len(X)}")

    # 4. Drop constant (zero-variance) columns
    dropped_cols = []
    if config.DROP_CONSTANT_COLS:
        dropped_cols = [c for c in X.columns if X[c].nunique() <= 1]
        X = X.drop(columns=dropped_cols)
        print(f"Dropped {len(dropped_cols)} constant columns: {dropped_cols}")

    feature_names = X.columns.tolist()
    n_features = len(feature_names)
    print(f"Usable feature columns: {n_features}")

    # 5. Stratified split: 64/16/20 train/val/test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=config.VAL_SIZE,
        stratify=y_trainval,
        random_state=config.RANDOM_STATE,
    )

    print("Class distribution (train):", y_train.value_counts().to_dict())
    print("Class distribution (val):  ", y_val.value_counts().to_dict())
    print("Class distribution (test): ", y_test.value_counts().to_dict())

    # 6. Scale -- fit on TRAIN ONLY
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, paths.SCALER_PATH)

    # 7. Reshape to 3D: (n_samples, n_features, 1)
    X_train_3d = X_train_scaled.reshape(-1, n_features, 1)
    X_val_3d = X_val_scaled.reshape(-1, n_features, 1)
    X_test_3d = X_test_scaled.reshape(-1, n_features, 1)

    result = {
        "X_train": X_train_3d,
        "y_train": y_train.to_numpy(),
        "X_val": X_val_3d,
        "y_val": y_val.to_numpy(),
        "X_test": X_test_3d,
        "y_test": y_test.to_numpy(),
        "n_features": n_features,
        "n_classes": n_classes,
        "feature_names": feature_names,
        "dropped_cols": dropped_cols,
    }

    if task == "application":
        result["class_names"] = class_names
        with open(paths.CLASS_NAMES_JSON, "w") as f:
            json.dump(class_names, f)
        print(f"Application class names ({len(class_names)}): {class_names}")

    return result


if __name__ == "__main__":
    # Demo: pandas / sklearn only. Never builds or fits a model.
    import sys

    # repo root is the parent of src/, needed since this is run as
    # `python src/preprocessing.py` (only src/ is auto-added to sys.path)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config as cfg

    data = load_and_prepare(cfg)
    print("task:", cfg.TASK)
    print("n_features:", data["n_features"])
    print("n_classes:", data["n_classes"])
    print("X_train shape:", data["X_train"].shape)
    print("X_val shape:  ", data["X_val"].shape)
    print("X_test shape: ", data["X_test"].shape)
