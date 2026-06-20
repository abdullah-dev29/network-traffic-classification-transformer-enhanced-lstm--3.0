"""
Evaluation for the Transformer-Enhanced LSTM.

Runs second on Colab, after train.py. Loads the saved model, test data, and
training history -- it never re-trains. Produces the same six metrics,
the same confusion-matrix / ROC / accuracy-loss-curve figures, and the
same metrics.json schema as Phase 1, so Phase 3 can load both phases'
metrics.json and tabulate them directly.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless: no display available when run via `!python`

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# repo root is the parent of src/, needed since this is run as
# `python src/evaluate.py` (only src/ is auto-added to sys.path)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def main():
    import tensorflow as tf

    model = tf.keras.models.load_model(config.MODEL_PATH)

    test_data = np.load(config.TEST_DATA_PATH)
    X_test, y_test = test_data["X_test"], test_data["y_test"]

    with open(config.HISTORY_PATH, "r") as f:
        history = json.load(f)

    y_prob = model.predict(X_test).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Precision:   {precision:.4f}")
    print(f"Recall:      {recall:.4f}")
    print(f"F1-score:    {f1:.4f}")
    print(f"ROC-AUC:     {auc:.4f}")
    print(f"Specificity: {specificity:.4f}")

    # --- metrics.json (schema identical to Phase 1) ---
    metrics = {
        "model_name": config.MODEL_NAME,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "specificity": specificity,
    }
    with open(config.METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    # --- metrics.txt (table mirroring the paper's Table 2) ---
    header = f"{'Model':<30}{'Accuracy':>10}{'F1':>10}{'Recall':>10}{'Precision':>10}{'AUC':>10}{'Specificity':>12}"
    row = (
        f"{config.MODEL_NAME:<30}{accuracy:>10.4f}{f1:>10.4f}{recall:>10.4f}"
        f"{precision:>10.4f}{auc:>10.4f}{specificity:>12.4f}"
    )
    with open(config.METRICS_TXT, "w") as f:
        f.write(header + "\n")
        f.write(row + "\n")

    # --- classification report ---
    report = classification_report(y_test, y_pred, target_names=config.CLASS_NAMES)
    with open(config.CLF_REPORT_TXT, "w") as f:
        f.write(report)

    # --- confusion matrix ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix -- {config.MODEL_NAME}")
    plt.tight_layout()
    plt.savefig(config.CONFUSION_PNG, dpi=150)
    plt.close()

    # --- ROC curve ---
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve -- {config.MODEL_NAME}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(config.ROC_CURVE_PNG, dpi=150)
    plt.close()

    # --- accuracy curve ---
    plt.figure(figsize=(6, 5))
    plt.plot(history["accuracy"], label="train")
    plt.plot(history["val_accuracy"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Curve -- {config.MODEL_NAME}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.ACC_CURVE_PNG, dpi=150)
    plt.close()

    # --- loss curve ---
    plt.figure(figsize=(6, 5))
    plt.plot(history["loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve -- {config.MODEL_NAME}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(config.LOSS_CURVE_PNG, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
