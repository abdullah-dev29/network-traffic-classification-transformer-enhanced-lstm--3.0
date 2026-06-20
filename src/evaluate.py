"""
Evaluation for the Transformer-Enhanced LSTM.

Runs second on Colab, after train.py (for the same --task). Loads the
saved model, validation/test data, and training history -- it never
re-trains.

Binary task: sweeps validation-set thresholds to find the F1-maximizing
operating point (config.TUNE_THRESHOLD), then reports test-set metrics at
both the default 0.5 threshold and the tuned threshold. This is how the
model's higher AUC gets cashed in as a better, balanced operating point
instead of being hidden behind a blind 0.5 cutoff.

Application task: reports accuracy, macro/weighted F1, macro precision/
recall, and macro one-vs-rest AUC, plus a per-class report and an 8x8
confusion matrix. Macro-F1 and the confusion matrix are the headline
metrics here given the ~10x class imbalance, not raw accuracy.

Fourclass task: trains one 4-class (Tor/VPN/Non-Tor/NonVPN) softmax
classifier, then reports TWO levels -- Level 2 is the 4-class result
(same kind of macro metrics as application), Level 1 is the binary
Darknet-vs-Benign view obtained by grouping the 4-class prediction (and
summing the Tor+VPN probability mass for AUC). Tor is ~1.1% of rows
(~55x imbalance), so it is expected to be the weak class -- macro-F1 and
the confusion matrix are the headline Level-2 metrics, not raw accuracy.
"""

import argparse
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


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the Transformer-Enhanced LSTM.")
    parser.add_argument(
        "--task", choices=["binary", "application", "fourclass"], default=config.TASK,
        help="Which classification task to evaluate (default: config.TASK).",
    )
    return parser.parse_args()


def _binary_metrics(y_true, y_pred, y_prob):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "specificity": tn / (tn + fp),
    }


def _plot_curves(history, paths, model_name):
    plt.figure(figsize=(6, 5))
    plt.plot(history["accuracy"], label="train")
    plt.plot(history["val_accuracy"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"Accuracy Curve -- {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths.ACC_CURVE_PNG, dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(history["loss"], label="train")
    plt.plot(history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Loss Curve -- {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(paths.LOSS_CURVE_PNG, dpi=150)
    plt.close()


def evaluate_binary(model, data, history, paths):
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]

    y_prob_val = model.predict(X_val).ravel()
    y_prob_test = model.predict(X_test).ravel()

    best_threshold = 0.5
    if config.TUNE_THRESHOLD:
        best_f1 = -1.0
        for t in np.arange(1, 100) / 100.0:  # 0.01 .. 0.99, avoids float-arange drift
            preds = (y_prob_val >= t).astype(int)
            f1 = f1_score(y_val, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = round(float(t), 2)
        print(f"Best threshold by validation F1: {best_threshold:.2f} (val F1={best_f1:.4f})")

    y_pred_05 = (y_prob_test >= 0.5).astype(int)
    y_pred_best = (y_prob_test >= best_threshold).astype(int)

    metrics_05 = _binary_metrics(y_test, y_pred_05, y_prob_test)
    metrics_best = _binary_metrics(y_test, y_pred_best, y_prob_test)

    print("--- Threshold 0.50 ---")
    for k, v in metrics_05.items():
        print(f"  {k}: {v:.4f}")
    print(f"--- Threshold {best_threshold:.2f} (tuned) ---")
    for k, v in metrics_best.items():
        print(f"  {k}: {v:.4f}")

    # --- metrics.json ---
    metrics = {
        "model_name": config.MODEL_NAME,
        "task": "binary",
        "threshold_0_5": metrics_05,
        "best_threshold": best_threshold,
        "threshold_tuned": metrics_best,
    }
    with open(paths.METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    # --- metrics.txt (two Table-2-style rows) ---
    header = (
        f"{'Model':<42}{'Accuracy':>10}{'F1':>10}{'Recall':>10}"
        f"{'Precision':>10}{'AUC':>10}{'Specificity':>12}"
    )

    def _row(name, m):
        return (
            f"{name:<42}{m['accuracy']:>10.4f}{m['f1']:>10.4f}{m['recall']:>10.4f}"
            f"{m['precision']:>10.4f}{m['auc']:>10.4f}{m['specificity']:>12.4f}"
        )

    with open(paths.METRICS_TXT, "w") as f:
        f.write(header + "\n")
        f.write(_row(f"{config.MODEL_NAME} (thr=0.50)", metrics_05) + "\n")
        f.write(_row(f"{config.MODEL_NAME} (thr=best={best_threshold:.2f})", metrics_best) + "\n")

    # --- classification report (at the tuned threshold, the recommended operating point) ---
    report = classification_report(y_test, y_pred_best, target_names=config.CLASS_NAMES)
    with open(paths.CLF_REPORT_TXT, "w") as f:
        f.write(f"Classification report at tuned threshold ({best_threshold:.2f})\n\n")
        f.write(report)

    # --- confusion matrix at the tuned threshold ---
    cm = confusion_matrix(y_test, y_pred_best)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix -- {config.MODEL_NAME} (thr={best_threshold:.2f})")
    plt.tight_layout()
    plt.savefig(paths.CONFUSION_PNG, dpi=150)
    plt.close()

    # --- ROC curve (threshold-independent) ---
    fpr, tpr, _ = roc_curve(y_test, y_prob_test)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {metrics_best['auc']:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve -- {config.MODEL_NAME}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(paths.ROC_CURVE_PNG, dpi=150)
    plt.close()

    _plot_curves(history, paths, config.MODEL_NAME)


def evaluate_application(model, data, history, paths):
    X_test, y_test = data["X_test"], data["y_test"]
    class_names = data["class_names"]

    y_prob = model.predict(X_test)
    y_pred = np.argmax(y_prob, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")

    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro F1:        {macro_f1:.4f}")
    print(f"Weighted F1:     {weighted_f1:.4f}")
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall:    {macro_recall:.4f}")
    print(f"Macro AUC (OvR): {macro_auc:.4f}")

    # --- metrics.json ---
    metrics = {
        "model_name": config.MODEL_NAME,
        "task": "application",
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_auc": macro_auc,
    }
    with open(paths.METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    # --- per-class report ---
    report = classification_report(y_test, y_pred, target_names=class_names)
    with open(paths.CLF_REPORT_TXT, "w") as f:
        f.write(report)

    # --- metrics.txt (readable summary + per-class report) ---
    with open(paths.METRICS_TXT, "w") as f:
        f.write(f"{config.MODEL_NAME} -- application-type classification (8 classes)\n\n")
        f.write(f"Accuracy:        {accuracy:.4f}\n")
        f.write(f"Macro F1:        {macro_f1:.4f}\n")
        f.write(f"Weighted F1:     {weighted_f1:.4f}\n")
        f.write(f"Macro Precision: {macro_precision:.4f}\n")
        f.write(f"Macro Recall:    {macro_recall:.4f}\n")
        f.write(f"Macro AUC (OvR): {macro_auc:.4f}\n\n")
        f.write(report)

    # --- 8x8 confusion matrix (raw counts) ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix -- {config.MODEL_NAME} (application)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(paths.CONFUSION_PNG, dpi=150)
    plt.close()

    # --- row-normalized version so small classes (VOIP, Email) are readable ---
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    norm_png = paths.CONFUSION_PNG.replace(".png", "_normalized.png")
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (row-normalized) -- {config.MODEL_NAME} (application)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(norm_png, dpi=150)
    plt.close()

    _plot_curves(history, paths, config.MODEL_NAME)


def evaluate_fourclass(model, data, history, paths):
    """Two-level reporting: Level 2 (4-class, fine) plus Level 1 (binary, grouped from it)."""
    X_test, y_test = data["X_test"], data["y_test"]
    class_names = data["class_names"]

    y_prob = model.predict(X_test)
    y_pred = np.argmax(y_prob, axis=1)

    # --- Level 2: 4-class (fine) ---
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")

    level2 = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_auc": macro_auc,
    }

    # --- Level 1: binary (coarse), derived by grouping the 4-class prediction ---
    idx_to_binary = np.array([1 if name in config.DARKNET_CLASSES else 0 for name in class_names])
    y_test_binary = idx_to_binary[y_test]
    y_pred_binary = idx_to_binary[y_pred]
    darknet_idx = [i for i, name in enumerate(class_names) if name in config.DARKNET_CLASSES]
    p_darknet = y_prob[:, darknet_idx].sum(axis=1)  # P(Tor) + P(VPN), for binary AUC

    level1 = _binary_metrics(y_test_binary, y_pred_binary, p_darknet)

    print("--- Level 1: binary (grouped from 4-class prediction) ---")
    for k, v in level1.items():
        print(f"  {k}: {v:.4f}")
    print("--- Level 2: 4-class ---")
    for k, v in level2.items():
        print(f"  {k}: {v:.4f}")

    # --- metrics.json ---
    metrics = {
        "model_name": config.MODEL_NAME,
        "task": "fourclass",
        "level1_binary": level1,
        "level2_fourclass": level2,
    }
    with open(paths.METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    # --- per-class report (Level 2) ---
    report = classification_report(y_test, y_pred, target_names=class_names)
    with open(paths.CLF_REPORT_TXT, "w") as f:
        f.write(report)

    # --- metrics.txt: Level-1 table row, then Level-2 summary + per-class report ---
    header = (
        f"{'Model':<42}{'Accuracy':>10}{'F1':>10}{'Recall':>10}"
        f"{'Precision':>10}{'AUC':>10}{'Specificity':>12}"
    )
    row = (
        f"{config.MODEL_NAME + ' (Level 1: binary, grouped)':<42}{level1['accuracy']:>10.4f}"
        f"{level1['f1']:>10.4f}{level1['recall']:>10.4f}{level1['precision']:>10.4f}"
        f"{level1['auc']:>10.4f}{level1['specificity']:>12.4f}"
    )
    with open(paths.METRICS_TXT, "w") as f:
        f.write("Level 1 -- Binary (Darknet vs Benign), grouped from the 4-class prediction\n")
        f.write(header + "\n")
        f.write(row + "\n\n")
        f.write("Level 2 -- 4-class (Tor, VPN, Non-Tor, NonVPN)\n\n")
        f.write(f"Accuracy:        {accuracy:.4f}\n")
        f.write(f"Macro F1:        {macro_f1:.4f}\n")
        f.write(f"Weighted F1:     {weighted_f1:.4f}\n")
        f.write(f"Macro Precision: {macro_precision:.4f}\n")
        f.write(f"Macro Recall:    {macro_recall:.4f}\n")
        f.write(f"Macro AUC (OvR): {macro_auc:.4f}\n\n")
        f.write(report)

    # --- 2x2 confusion matrix (Level 1, binary) ---
    cm_binary = confusion_matrix(y_test_binary, y_pred_binary)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_binary, annot=True, fmt="d", cmap="Blues",
        xticklabels=config.CLASS_NAMES, yticklabels=config.CLASS_NAMES,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix -- {config.MODEL_NAME} (Level 1: binary, grouped)")
    plt.tight_layout()
    plt.savefig(os.path.join(paths.FIGURES_DIR, "confusion_matrix_binary.png"), dpi=150)
    plt.close()

    # --- 4x4 confusion matrix (Level 2, raw counts) ---
    cm4 = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm4, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix -- {config.MODEL_NAME} (Level 2: 4-class)")
    plt.tight_layout()
    plt.savefig(os.path.join(paths.FIGURES_DIR, "confusion_matrix_4class.png"), dpi=150)
    plt.close()

    # --- 4x4 confusion matrix, row-normalized so rare Tor is readable ---
    cm4_norm = cm4.astype(float) / cm4.sum(axis=1, keepdims=True)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm4_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (row-normalized) -- {config.MODEL_NAME} (Level 2: 4-class)")
    plt.tight_layout()
    plt.savefig(os.path.join(paths.FIGURES_DIR, "confusion_matrix_4class_normalized.png"), dpi=150)
    plt.close()

    _plot_curves(history, paths, config.MODEL_NAME)


def main():
    args = parse_args()
    task = args.task
    paths = config.get_paths(task)

    import tensorflow as tf

    model = tf.keras.models.load_model(paths.MODEL_PATH)

    splits = np.load(paths.DATA_SPLITS_PATH)
    data = {
        "X_val": splits["X_val"],
        "y_val": splits["y_val"],
        "X_test": splits["X_test"],
        "y_test": splits["y_test"],
    }

    with open(paths.HISTORY_PATH, "r") as f:
        history = json.load(f)

    if task == "binary":
        evaluate_binary(model, data, history, paths)
    elif task == "application":
        with open(paths.CLASS_NAMES_JSON, "r") as f:
            data["class_names"] = json.load(f)
        evaluate_application(model, data, history, paths)
    else:  # fourclass
        with open(paths.CLASS_NAMES_JSON, "r") as f:
            data["class_names"] = json.load(f)
        evaluate_fourclass(model, data, history, paths)


if __name__ == "__main__":
    main()
