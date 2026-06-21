"""
Preprocessing pipeline for CIC-Darknet2020 (Phase 2, inert) and CIC-IDS2017
(Phase 3, current), supporting six tasks:

CIC-Darknet2020 (single parquet, config.DATA_PATH):
- "binary": Darknet vs Benign from `Label`. Logically IDENTICAL to the
  original Phase 1 baseline pipeline -- same label mapping, same
  constant-column drop, same stratified split with the same RANDOM_STATE,
  same StandardScaler-fit-on-train-only policy. This guarantees the
  held-out test set is the exact same rows as Phase 1, the precondition
  for a fair Phase 3 comparison.
- "application": 8-class application type from `Label.1`, after collapsing
  casing variants to canonical names (config.APP_LABEL_CANON).
- "fourclass": the 4 classes underlying the binary grouping (Tor, VPN,
  Non-Tor, NonVPN) from `Label`. evaluate.py reports this hierarchically --
  the 4-class result, plus a binary view obtained by grouping it.

CIC-IDS2017 (folder of parquets, config.DATA_DIR, all concatenated -- see
_load_ids2017()):
- "ids_binary": Benign (0) vs any attack (1) from `Label`. The one task
  that is genuinely class-comparable to the darknet "binary" task.
- "ids_family": coarse attack families (DoS/DDoS, Brute-Force, Web-Attack,
  Botnet, PortScan, Infiltration, Other) from `Label`, via _map_family().
- "ids_multi": fine attack types from `Label`, with the three Web Attack
  variants merged into one class and ultra-rare classes folded/dropped
  per config.MIN_CLASS_COUNT -- see _build_ids_multi_target().

All six tasks share identical cleaning / constant-column-drop / scaling /
reshape / split mechanics -- they only differ in how the target `y` (and
which label columns get dropped from `X`) is constructed. For the
CIC-IDS2017 tasks, an optional stratified subsample (config.SUBSAMPLE_FRAC)
is applied right after cleaning and before the per-task target is built --
stratified on the raw fine `Label` column so ids_binary/ids_family/
ids_multi all draw the exact same rows, and so ids_multi's web-variant
merge + MIN_CLASS_COUNT rare-class fold (see _build_ids_multi_target())
runs on the subsampled counts rather than the full pre-subsample counts.

Import-safe: nothing in this module trains a model. The __main__ demo at
the bottom only loads data and prints summary stats (pandas / sklearn only).
"""

import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

IDS_TASKS = ("ids_binary", "ids_family", "ids_multi")
DARKNET_TASKS = ("binary", "application", "fourclass")


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


def _build_fourclass_target(df: pd.DataFrame, config):
    """4-class Tor/VPN/Non-Tor/NonVPN from `Label`. No casing issues, unlike Label.1."""
    encoder = LabelEncoder()
    y = pd.Series(
        encoder.fit_transform(df[config.FOURCLASS_LABEL_COL]), index=df.index
    )
    class_names = encoder.classes_.tolist()

    drop_cols = [config.LABEL_COL, config.APP_LABEL_COL]
    X = df.drop(columns=drop_cols)
    return y, X, class_names


def _load_ids2017(config) -> pd.DataFrame:
    """Glob config.DATA_DIR for *.parquet and concatenate all of them.

    CIC-IDS2017 is shipped here as 8 day/attack-type files that together
    form the full dataset -- using only some of them would silently drop
    whole attack classes (e.g. Heartbleed only exists in one file).
    """
    files = sorted(glob.glob(os.path.join(config.DATA_DIR, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {config.DATA_DIR}")

    dfs = [pd.read_parquet(f) for f in files]
    for f, d in zip(files, dfs):
        print(f"  {f}: shape={d.shape}")

    common_cols = set(dfs[0].columns)
    for d in dfs[1:]:
        common_cols &= set(d.columns)
    common_cols = [c for c in dfs[0].columns if c in common_cols]
    if any(len(d.columns) != len(common_cols) for d in dfs):
        print(f"WARNING: parquet files have differing columns; aligned on {len(common_cols)} common columns")

    df = pd.concat([d[common_cols] for d in dfs], ignore_index=True)
    print(f"Loaded {len(files)} parquet files from {config.DATA_DIR}, combined shape {df.shape}")
    return df


def _map_family(label: str, config) -> str:
    """Map a fine CIC-IDS2017 label to its coarse ids_family bucket. See
    config.py's "CIC-IDS2017 (Phase 3)" block for the documented rule set.
    """
    if label in config.FAMILY_EXACT:
        return config.FAMILY_EXACT[label]
    if config.FAMILY_DOS_SUBSTR in label:  # also catches "DDoS" (contains "DoS")
        return "DoS/DDoS"
    if label.startswith(config.FAMILY_WEBATTACK_PREFIX):
        return "Web-Attack"
    return config.FAMILY_DEFAULT


def _build_ids_binary_target(df: pd.DataFrame, config):
    """Benign (0) vs any attack (1) from `Label`."""
    label = df[config.IDS_LABEL_COL].astype(str).str.strip()
    y = (label != config.IDS_BENIGN_LABEL).astype(int)
    X = df.drop(columns=[c for c in [config.IDS_LABEL_COL] + config.IDS_DROP_COLS if c in df.columns])
    return y, X, config.IDS_BINARY_CLASS_NAMES


def _build_ids_family_target(df: pd.DataFrame, config):
    """Coarse attack-family classification from `Label` via _map_family()."""
    label = df[config.IDS_LABEL_COL].astype(str).str.strip()
    families = label.map(lambda v: _map_family(v, config))

    # Anything that landed in "Other" without being the explicitly-documented
    # Heartbleed catch-all is an unexpected/unmapped label -- surface it
    # loudly instead of silently lumping it in.
    unexpected = families.eq(config.FAMILY_DEFAULT) & ~label.isin(
        [k for k, v in config.FAMILY_EXACT.items() if v == config.FAMILY_DEFAULT]
    )
    if unexpected.any():
        bad = sorted(label[unexpected].unique().tolist())
        print(f"WARNING: {int(unexpected.sum())} rows fell into '{config.FAMILY_DEFAULT}' via unmapped labels: {bad}")

    print("ids_family bucket counts:", families.value_counts().to_dict())

    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(families), index=families.index)
    class_names = encoder.classes_.tolist()

    X = df.drop(columns=[c for c in [config.IDS_LABEL_COL] + config.IDS_DROP_COLS if c in df.columns])
    return y, X, class_names


def _build_ids_multi_target(df: pd.DataFrame, config):
    """Fine-grained attack-type classification from `Label`.

    Two cleanups before LabelEncoding:
    1. Merge the three "Web Attack ..." variants into one
       config.WEBATTACK_MERGED_NAME class.
    2. Any resulting class with fewer than config.MIN_CLASS_COUNT rows is
       folded into its ids_family bucket IF a same-family sibling survives
       at >= MIN_CLASS_COUNT in the fine label set (a real merge), otherwise
       its rows are dropped (folding a singleton-family class into itself
       doesn't fix the small-N problem and would still break stratification).
    """
    label = df[config.IDS_LABEL_COL].astype(str).str.strip()
    merged = label.where(~label.str.startswith(config.FAMILY_WEBATTACK_PREFIX), config.WEBATTACK_MERGED_NAME)

    counts = merged.value_counts()
    rare_classes = counts[counts < config.MIN_CLASS_COUNT].index.tolist()

    keep_mask = pd.Series(True, index=merged.index)
    folded, dropped = {}, {}
    for cls in rare_classes:
        family = _map_family(cls, config)
        siblings = [
            c for c in counts.index
            if c != cls and _map_family(c, config) == family and counts[c] >= config.MIN_CLASS_COUNT
        ]
        if siblings:
            merged = merged.where(merged != cls, family)
            folded[cls] = {"family": family, "rows": int(counts[cls])}
        else:
            keep_mask &= merged != cls
            dropped[cls] = int(counts[cls])

    print(f"ids_multi: rare classes (< {config.MIN_CLASS_COUNT} rows) folded into a family bucket: {folded}")
    print(f"ids_multi: rare classes (< {config.MIN_CLASS_COUNT} rows) dropped (no viable fold target): {dropped}")

    df = df.loc[keep_mask]
    merged = merged.loc[keep_mask]
    print("ids_multi final class counts:", merged.value_counts().to_dict())

    encoder = LabelEncoder()
    y = pd.Series(encoder.fit_transform(merged), index=merged.index)
    class_names = encoder.classes_.tolist()

    X = df.drop(columns=[c for c in [config.IDS_LABEL_COL] + config.IDS_DROP_COLS if c in df.columns])
    return y, X, class_names


def _clean_features(X: pd.DataFrame, config):
    """Defensive cleaning shared by every task: inf -> NaN, drop NaN rows,
    drop duplicate rows, drop constant columns. Returns the cleaned X and
    the dropped constant-column names; the caller re-aligns any parallel
    Series (y, or a raw label column) to X.index afterwards.
    """
    n_before = len(X)
    X = X.replace([np.inf, -np.inf], np.nan)
    inf_replaced = int(X.isna().sum().sum())
    print(f"Replaced inf/-inf with NaN in {inf_replaced} cells")

    nan_mask = X.isna().any(axis=1)
    n_nan_rows = int(nan_mask.sum())
    X = X.loc[~nan_mask]
    print(f"Dropped {n_nan_rows} rows containing NaN")

    dup_mask = X.duplicated()
    n_dup_rows = int(dup_mask.sum())
    X = X.loc[~dup_mask]
    print(f"Dropped {n_dup_rows} duplicate rows")
    print(f"Rows: {n_before} -> {len(X)}")

    dropped_cols = []
    if config.DROP_CONSTANT_COLS:
        dropped_cols = [c for c in X.columns if X[c].nunique() <= 1]
        X = X.drop(columns=dropped_cols)
        print(f"Dropped {len(dropped_cols)} constant columns: {dropped_cols}")

    return X, dropped_cols


def _safe_counts(s: pd.Series) -> dict:
    """value_counts() as a dict with console-safe keys. The raw CIC-IDS2017
    `Label` column carries a corrupted U+FFFD byte in the three "Web Attack"
    variants (a pre-existing upstream artifact, see config.py) which crashes
    printing on non-UTF-8 consoles -- escape non-ASCII bytes for display only.
    """
    return {
        str(k).encode("ascii", "backslashreplace").decode("ascii"): int(v)
        for k, v in s.value_counts().items()
    }


def _stratified_subsample(X: pd.DataFrame, label: pd.Series, config):
    """Stratified subsample of (X, label) down to config.SUBSAMPLE_FRAC of
    the rows, stratified on the raw fine CIC-IDS2017 label so ids_binary /
    ids_family / ids_multi all draw the exact same rows (this runs before
    any task-specific target is built, on the common cleaned feature set).

    A class so rare that round(count * frac) < 2 is kept in full rather
    than sampled, since a stratified split needs >= 2 members per class.
    """
    frac = config.SUBSAMPLE_FRAC
    if frac is None or frac >= 1.0:
        return X, label

    n_before = len(X)
    counts = label.value_counts()
    guard_classes = [c for c, n in counts.items() if round(n * frac) < 2]

    if guard_classes:
        guard_counts = {
            str(c).encode("ascii", "backslashreplace").decode("ascii"): int(counts[c])
            for c in guard_classes
        }
        print(f"SUBSAMPLE_FRAC={frac}: classes too rare to subsample, kept in full: {guard_counts}")
        guard_mask = label.isin(guard_classes)
        X_guard, label_guard = X.loc[guard_mask], label.loc[guard_mask]
        X_rest, label_rest = X.loc[~guard_mask], label.loc[~guard_mask]
    else:
        X_guard, label_guard = None, None
        X_rest, label_rest = X, label

    X_sample, _, label_sample, _ = train_test_split(
        X_rest, label_rest,
        train_size=frac,
        stratify=label_rest,
        random_state=config.RANDOM_STATE,
    )

    if X_guard is not None:
        X_sample = pd.concat([X_sample, X_guard])
        label_sample = pd.concat([label_sample, label_guard])

    print(f"Subsampled {n_before} -> {len(X_sample)} rows (stratified, frac={frac})")
    print("Class distribution after subsampling:", _safe_counts(label_sample))

    return X_sample, label_sample


def load_and_prepare(config, task: str = None) -> dict:
    """Load the data, build the target for `task`, clean, split, and scale.

    `task` defaults to config.TASK. One of:
      "binary", "application", "fourclass"   -- CIC-Darknet2020 (Phase 2)
      "ids_binary", "ids_family", "ids_multi" -- CIC-IDS2017 (Phase 3)

    Returns a dict with X_train, y_train, X_val, y_val, X_test, y_test
    (each X reshaped to (n_samples, n_features, 1)), plus n_features,
    n_classes, feature_names, dropped_cols, and class_names for every task
    except the two binary tasks (whose class names are fixed in config and
    don't need persisting). application/fourclass/ids_family/ids_multi
    persist class_names.json under the task's results dir so evaluate.py
    uses a consistent order.
    """
    if task is None:
        task = config.TASK
    if task not in DARKNET_TASKS + IDS_TASKS:
        raise ValueError(f"Unknown task: {task!r}")

    paths = config.get_paths(task)
    os.makedirs(paths.RESULTS_DIR, exist_ok=True)

    # 1. Load
    if task in IDS_TASKS:
        df = _load_ids2017(config)
    else:
        df = pd.read_parquet(config.DATA_PATH)

    if task in IDS_TASKS:
        # 2. Clean + (optionally) subsample the common feature matrix BEFORE
        # building the per-task target. All three IDS tasks share the same
        # raw `Label` column and the same cleaned X at this point, so the
        # stratified subsample draws identical rows for ids_binary/
        # ids_family/ids_multi; ids_multi's web-variant merge + rare-class
        # fold (below) then runs on the already-subsampled counts.
        raw_label = df[config.IDS_LABEL_COL].astype(str).str.strip()
        X = df.drop(columns=[c for c in [config.IDS_LABEL_COL] + config.IDS_DROP_COLS if c in df.columns])

        X, dropped_cols = _clean_features(X, config)
        raw_label = raw_label.loc[X.index]

        X, raw_label = _stratified_subsample(X, raw_label, config)

        feature_names = X.columns.tolist()
        n_features = len(feature_names)
        print(f"Usable feature columns: {n_features}")

        # 3. Build the per-task target on the cleaned + subsampled rows.
        df = X.copy()
        df[config.IDS_LABEL_COL] = raw_label
        if task == "ids_binary":
            y, X, class_names = _build_ids_binary_target(df, config)
        elif task == "ids_family":
            y, X, class_names = _build_ids_family_target(df, config)
        else:  # ids_multi
            y, X, class_names = _build_ids_multi_target(df, config)
        n_classes = len(class_names)
    else:
        # 2. Build target + drop label columns from X (task-specific)
        if task == "binary":
            y, X, class_names = _build_binary_target(df, config)
        elif task == "application":
            y, X, class_names = _build_application_target(df, config)
        else:  # fourclass
            y, X, class_names = _build_fourclass_target(df, config)
        n_classes = len(class_names)

        # 3. Defensive cleaning (expect ~0 changes on this pre-cleaned dataset)
        X, dropped_cols = _clean_features(X, config)
        y = y.loc[X.index]

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

    if task in ("application", "fourclass", "ids_family", "ids_multi"):
        result["class_names"] = class_names
        with open(paths.CLASS_NAMES_JSON, "w") as f:
            json.dump(class_names, f)
        print(f"{task} class names ({len(class_names)}): {class_names}")

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
