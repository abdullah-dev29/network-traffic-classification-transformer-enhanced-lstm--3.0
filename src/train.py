"""
Training orchestration for the Transformer-Enhanced LSTM.

This script is meant to run on Google Colab (GPU), not on a local machine.
It loads and prepares the data for the chosen --task, builds the model,
fits it, and persists everything evaluate.py needs. It contains no
evaluation logic.

--------------------------------------------------------------------------
Fair-Comparison Note (binary task)
--------------------------------------------------------------------------
Earlier defaults trained the binary model with USE_CLASS_WEIGHT=True,
which won on F1/AUC/recall but lost on precision/accuracy against the
Phase 1 baseline (which trained without class weights) -- not an
apples-to-apples comparison. The default is now MATCHED to the baseline:
USE_CLASS_WEIGHT=False. The model's higher AUC is instead cashed in via
validation-set threshold tuning in evaluate.py (config.TUNE_THRESHOLD):
the decision threshold is chosen to maximize F1 on the validation set,
then applied to the test set. evaluate.py reports metrics at both the
default 0.5 threshold and the tuned threshold, so neither number is
hidden.

To experiment with class-weighted training instead (a separate, valid
choice, not a substitute for threshold tuning), set USE_CLASS_WEIGHT=True
in config.py and describe the run accordingly in the report. Do not
silently mix settings.
--------------------------------------------------------------------------
"""

import argparse
import json
import os
import random
import sys

import numpy as np

# repo root is the parent of src/, needed since this is run as
# `python src/train.py` (only src/ is auto-added to sys.path)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from model import build_transformer_lstm
from preprocessing import load_and_prepare


def set_seeds(seed: int) -> None:
    """Seed random, numpy, and tensorflow. Full GPU determinism is not guaranteed."""
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    if hasattr(tf.keras.utils, "set_random_seed"):
        tf.keras.utils.set_random_seed(seed)
    else:
        tf.random.set_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Transformer-Enhanced LSTM.")
    parser.add_argument(
        "--task", choices=["binary", "application"], default=config.TASK,
        help="Which classification task to train (default: config.TASK).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    task = args.task
    paths = config.get_paths(task)

    import tensorflow as tf
    from sklearn.utils.class_weight import compute_class_weight

    set_seeds(config.RANDOM_STATE)

    os.makedirs(paths.RESULTS_DIR, exist_ok=True)
    os.makedirs(paths.FIGURES_DIR, exist_ok=True)

    data = load_and_prepare(config, task=task)
    model = build_transformer_lstm(
        n_features=data["n_features"], n_classes=data["n_classes"], task=task
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=config.REDUCE_LR_PATIENCE,
            factor=0.5,
        ),
    ]

    use_class_weight = config.USE_CLASS_WEIGHT if task == "binary" else config.APP_USE_CLASS_WEIGHT
    class_weight = None
    if use_class_weight:
        weights = compute_class_weight(
            "balanced", classes=np.unique(data["y_train"]), y=data["y_train"]
        )
        class_weight = dict(enumerate(weights))
        print("Using class weights:", class_weight)

    history = model.fit(
        data["X_train"],
        data["y_train"],
        validation_data=(data["X_val"], data["y_val"]),
        epochs=config.EPOCHS,
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    model.save(paths.MODEL_PATH)

    with open(paths.HISTORY_PATH, "w") as f:
        json.dump(history.history, f)

    # Both validation and test arrays are persisted: validation is needed by
    # evaluate.py for threshold tuning (binary task), test is the held-out
    # set the metrics are ultimately reported on.
    np.savez_compressed(
        paths.DATA_SPLITS_PATH,
        X_val=data["X_val"],
        y_val=data["y_val"],
        X_test=data["X_test"],
        y_test=data["y_test"],
    )

    final_epoch = len(history.history["loss"]) - 1
    print(f"Task: {task}")
    print(f"Final epoch: {final_epoch + 1}")
    for key, values in history.history.items():
        print(f"  {key}: {values[final_epoch]:.4f}")


if __name__ == "__main__":
    main()
