"""Classification metrics and plotting for HARUSPEX model evaluation."""

from __future__ import annotations

from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from haruspex.utils.logging import get_logger
from haruspex.utils.storage import save_plot

logger = get_logger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """Compute standard binary classification metrics.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities for the positive class.

    Returns:
        Dictionary of metric name → value.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_proba is not None:
        try:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics["auc_roc"] = 0.0
            logger.warning("AUC-ROC could not be computed (single class in y_true)")

    return metrics


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_name: str = "confusion_matrix",
    title: str = "Confusion Matrix",
    labels: Optional[list[str]] = None,
) -> plt.Figure:
    """Generate and save a confusion matrix plot.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        save_name: Base filename for saving.
        title: Plot title.
        labels: Class labels. Defaults to ["Benign", "Malware"].

    Returns:
        Matplotlib figure.
    """
    labels = labels or ["Benign", "Malware"]
    # Explicitly pin labels=[0, 1] so the matrix is always 2x2, even when
    # only one class is present in y_true/y_pred (e.g. a held-out split
    # where the model gets every sample right).
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        title=title,
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    # Add text annotations
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14, fontweight="bold",
            )

    fig.tight_layout()
    save_plot(fig, save_name)
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    save_name: str = "roc_curve",
    title: str = "ROC Curve",
) -> plt.Figure:
    """Generate and save an ROC curve plot.

    Args:
        y_true: Ground truth labels.
        y_proba: Predicted probabilities.
        save_name: Base filename for saving.
        title: Plot title.

    Returns:
        Matplotlib figure.
    """
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="#00d4ff", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random")
    ax.fill_between(fpr, tpr, alpha=0.1, color="#00d4ff")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    save_plot(fig, save_name)
    return fig


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    save_name: str = "feature_importance",
    title: str = "Feature Importance",
    top_n: int = 20,
) -> plt.Figure:
    """Generate and save a feature importance bar chart.

    Args:
        importances: Feature importance scores.
        feature_names: Feature name labels.
        save_name: Base filename for saving.
        title: Plot title.
        top_n: Number of top features to display.

    Returns:
        Matplotlib figure.
    """
    top_idx = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in top_idx]
    top_values = importances[top_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        range(len(top_names)),
        top_values[::-1],
        color="#00d4ff",
        edgecolor="#003d4d",
    )
    ax.set_yticks(range(len(top_names)))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Importance Score", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    save_plot(fig, save_name)
    return fig
