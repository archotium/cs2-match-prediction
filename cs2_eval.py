"""
cs2_eval.py
===========

Evaluation and plotting helpers for the CS2 match-prediction models.

Shared metric printing and plotting (confusion matrix, ROC curve,
precision-recall curve) so each model cell in the notebook stays short:

    from cs2_eval import evaluate
    evaluate("XGBoost", y_test, y_pred, y_proba)

and comparing several models is one call:

    from cs2_eval import compare_models
    compare_models(y_test, {
        "Basic":  (y_pred_basic,  proba_basic),
        "Tuned":  (y_pred_tuned,  proba_tuned),
        "Pruned": (y_pred_pruned, proba_pruned),
    })
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, log_loss, brier_score_loss,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, precision_recall_curve, average_precision_score,
)

# A consistent colour per comparison role, reused across plots.
_DEFAULT_CMAPS = ["Blues", "Greens", "Oranges", "Purples", "Reds"]
_DEFAULT_COLORS = ["blue", "green", "orange", "purple", "red"]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def print_metrics(name: str, y_test, y_pred, y_proba) -> dict:
    """Print the standard metric block and return the scores as a dict.

    y_pred  : predicted 0/1 labels.
    y_proba : predicted probability of class 1 (used for AUC / log loss / Brier).
    """
    scores = {
        "accuracy": accuracy_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "log_loss": log_loss(y_test, y_proba),
        "brier": brier_score_loss(y_test, y_proba),
    }
    print(f"=== {name} ===")
    print("Accuracy:    ", round(scores["accuracy"], 4))
    print("ROC AUC:     ", round(scores["roc_auc"], 4))
    print("Log Loss:    ", round(scores["log_loss"], 4))
    print("Brier Score: ", round(scores["brier"], 4))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    return scores


# ---------------------------------------------------------------------------
# Single-model plots
# ---------------------------------------------------------------------------

def plot_confusion(y_test, y_pred, title: str | None = None,
                   cmap: str = "Blues", ax=None):
    """Draw a single confusion matrix."""
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=cmap, ax=ax, colorbar=ax is None)
    target_ax = disp.ax_ if ax is None else ax
    if title:
        target_ax.set_title(title)
    target_ax.grid(False)
    return disp


def plot_roc(y_test, y_proba, label: str = "Model", ax=None, color="blue"):
    """Draw a single ROC curve (with the random-classifier diagonal)."""
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, lw=2, color=color, label=f"{label} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--",
            label="Random Classifier")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.legend(loc="lower right")
    return ax


def plot_pr(y_test, y_proba, label: str = "Model", ax=None, color="blue"):
    """Draw a single precision-recall curve."""
    precision, recall, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, marker=".", color=color,
            label=f"{label} (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend()
    return ax


def evaluate(name: str, y_test, y_pred, y_proba, cmap: str = "Blues") -> dict:
    """Print metrics and draw confusion matrix + ROC + PR for one model.

    Returns the metric dict.
    """
    scores = print_metrics(name, y_test, y_pred, y_proba)

    plot_confusion(y_test, y_pred, title=name, cmap=cmap)
    plt.tight_layout()
    plt.show()

    plot_roc(y_test, y_proba, label=name)
    plt.tight_layout()
    plt.show()

    plot_pr(y_test, y_proba, label=name)
    plt.tight_layout()
    plt.show()

    return scores


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------

def compare_models(y_test, results: dict) -> None:
    """Overlay several models' ROC and PR curves, plus side-by-side confusions.

    results : ordered dict of  name -> (y_pred, y_proba).
              e.g. {"Basic": (y_pred1, proba1), "Tuned": (y_pred2, proba2)}
    """
    names = list(results.keys())

    # --- ROC overlay ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(names):
        _, y_proba = results[name]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax.plot(fpr, tpr, color=_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)],
                label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    # --- PR overlay ---
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(names):
        _, y_proba = results[name]
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        ax.plot(recall, precision, color=_DEFAULT_COLORS[i % len(_DEFAULT_COLORS)],
                label=f"{name} (AP = {ap:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

    # --- confusion matrices side by side ---
    fig, axes = plt.subplots(1, len(names), figsize=(6 * len(names), 5))
    if len(names) == 1:
        axes = [axes]
    for i, name in enumerate(names):
        y_pred, _ = results[name]
        plot_confusion(y_test, y_pred, title=name,
                       cmap=_DEFAULT_CMAPS[i % len(_DEFAULT_CMAPS)], ax=axes[i])
    plt.tight_layout()
    plt.show()
