"""
Preprocessing pipeline for CIC-Darknet2020 binary classification.

This module's spec is IDENTICAL to the Phase 1 CNN-LSTM baseline: same
label mapping, same constant-column drop, same stratified split with the
same RANDOM_STATE, and the same StandardScaler-fit-on-train-only policy.
Keeping this identical guarantees the held-out test set is the exact same
rows as Phase 1 -- the precondition for a fair Phase 3 comparison.

Import-safe: nothing in this module trains a model. The __main__ demo at
the bottom only loads data and prints summary stats (pandas / sklearn only).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def load_and_prepare(config) -> dict:
    """Load the parquet, build the binary target, clean, split, and scale.

    Returns a dict with X_train, y_train, X_val, y_val, X_test, y_test
    (each X reshaped to (n_samples, n_features, 1)), plus n_features,
    feature_names, and dropped_cols.
    """
    # 1. Load
    df = pd.read_parquet(config.DATA_PATH)

    # 2. Build binary y from the multi-class Label column
    darknet_mask = df[config.LABEL_COL].isin(config.DARKNET_CLASSES)
    benign_mask = df[config.LABEL_COL].isin(config.BENIGN_CLASSES)
    unmapped = ~(darknet_mask | benign_mask)
    if unmapped.any():
        bad_values = df.loc[unmapped, config.LABEL_COL].unique().tolist()
        raise AssertionError(
            f"Found {int(unmapped.sum())} rows with unmapped Label values: {bad_values}"
        )
    y = darknet_mask.astype(int)

    # 3. Drop label columns from X
    drop_cols = [config.LABEL_COL]
    if config.APP_LABEL_COL in df.columns:
        drop_cols.append(config.APP_LABEL_COL)
    X = df.drop(columns=drop_cols)

    # 4. Defensive cleaning (expect ~0 changes on this pre-cleaned dataset)
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

    # 5. Drop constant (zero-variance) columns
    dropped_cols = []
    if config.DROP_CONSTANT_COLS:
        dropped_cols = [c for c in X.columns if X[c].nunique() <= 1]
        X = X.drop(columns=dropped_cols)
        print(f"Dropped {len(dropped_cols)} constant columns: {dropped_cols}")

    feature_names = X.columns.tolist()
    n_features = len(feature_names)
    print(f"Usable feature columns: {n_features}")

    # 6. Stratified split: 64/16/20 train/val/test
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

    # 7. Scale -- fit on TRAIN ONLY
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    joblib.dump(scaler, config.SCALER_PATH)

    # 8. Reshape to 3D: (n_samples, n_features, 1)
    X_train_3d = X_train_scaled.reshape(-1, n_features, 1)
    X_val_3d = X_val_scaled.reshape(-1, n_features, 1)
    X_test_3d = X_test_scaled.reshape(-1, n_features, 1)

    return {
        "X_train": X_train_3d,
        "y_train": y_train.to_numpy(),
        "X_val": X_val_3d,
        "y_val": y_val.to_numpy(),
        "X_test": X_test_3d,
        "y_test": y_test.to_numpy(),
        "n_features": n_features,
        "feature_names": feature_names,
        "dropped_cols": dropped_cols,
    }


if __name__ == "__main__":
    # Demo: pandas / sklearn only. Never builds or fits a model.
    import os
    import sys

    # repo root is the parent of src/, needed since this is run as
    # `python src/preprocessing.py` (only src/ is auto-added to sys.path)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config as cfg

    os.makedirs(cfg.RESULTS_DIR, exist_ok=True)
    data = load_and_prepare(cfg)
    print("n_features:", data["n_features"])
    print("X_train shape:", data["X_train"].shape)
    print("X_val shape:  ", data["X_val"].shape)
    print("X_test shape: ", data["X_test"].shape)
