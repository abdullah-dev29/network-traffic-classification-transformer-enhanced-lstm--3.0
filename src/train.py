"""
Training orchestration for the Transformer-Enhanced LSTM.

This script is meant to run on Google Colab (GPU), not on a local machine.
It loads and prepares the data, builds the model, fits it, and persists
everything evaluate.py needs. It contains no evaluation logic.

--------------------------------------------------------------------------
Fair-Comparison Note
--------------------------------------------------------------------------
This phase ships with imbalance-aware training defaults
(USE_CLASS_WEIGHT=True, EPOCHS=50) to get the strongest result from the
proposed model. Two valid ways to report the comparison vs. the Phase 1
baseline:

1. Best-model comparison (default): report the proposed model at these
   settings. Describe it honestly as "Transformer-Enhanced LSTM with
   class-weighted training." This is what most papers report.
2. Architecture-only ablation: to isolate the contribution of the
   architecture alone, set USE_CLASS_WEIGHT=False and EPOCHS=30 to match
   Phase 1 exactly -- then the only difference between the two models is
   model.py.

Whichever you choose, state it clearly in the report. Do not silently mix
settings.
--------------------------------------------------------------------------
"""

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


def main():
    import tensorflow as tf
    from sklearn.utils.class_weight import compute_class_weight

    set_seeds(config.RANDOM_STATE)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.FIGURES_DIR, exist_ok=True)

    data = load_and_prepare(config)
    model = build_transformer_lstm(n_features=data["n_features"])
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

    class_weight = None
    if config.USE_CLASS_WEIGHT:
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

    model.save(config.MODEL_PATH)

    with open(config.HISTORY_PATH, "w") as f:
        json.dump(history.history, f)

    np.savez_compressed(
        config.TEST_DATA_PATH,
        X_test=data["X_test"],
        y_test=data["y_test"],
    )

    final_epoch = len(history.history["loss"]) - 1
    print(f"Final epoch: {final_epoch + 1}")
    for key, values in history.history.items():
        print(f"  {key}: {values[final_epoch]:.4f}")


if __name__ == "__main__":
    main()
