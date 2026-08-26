"""Plotting utilities for HARUSPEX Deep Learning models.

Generates:
  - Confusion Matrices & ROC Curves
  - Training History Plots (Loss & Accuracy)
  - Feature Importance Plots
  - Group Gate Weight Distributions (Boxplot/Violin)
  - Group Multi-Head Attention Heatmaps
  - Per-Sample/Per-Family Gate Weight Profiles
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

from deepLearning.utils.config import FEATURE_GROUPS, get_feature_names

logger = logging.getLogger("haruspex.deepLearning.plots")

GROUP_NAMES = [g[0] for g in FEATURE_GROUPS]


def save_figure(fig: plt.Figure, save_path: Union[str, Path]) -> Path:
    """Save matplotlib figure to disk."""
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.suffix:
        path = path.with_suffix(".png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved plot to %s", path)
    return path


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Confusion Matrix",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Benign", "Malware"],
        yticklabels=["Benign", "Malware"],
        ax=ax,
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=10)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    title: str = "ROC Curve",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#00d4ff", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="#8b95a8", linestyle="--", lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_feature_importance(
    importance_scores: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_k: int = 20,
    title: str = "Top Feature Importances",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot top-k feature importances."""
    names = feature_names or get_feature_names()
    top_idx = np.argsort(importance_scores)[::-1][:top_k]

    top_names = [names[i] for i in top_idx][::-1]
    top_scores = [importance_scores[i] for i in top_idx][::-1]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top_names, top_scores, color="#aa66ff")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Importance Score", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_group_gate_distributions(
    gate_weights: np.ndarray,  # (N, 9)
    group_names: Sequence[str] = GROUP_NAMES,
    title: str = "Group Gate Weight Distribution",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot violin/boxplot distribution of 9 group gate weights."""
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.violinplot(data=gate_weights, ax=ax, palette="mako", inner="quartile")
    ax.set_xticks(range(len(group_names)))
    ax.set_xticklabels(group_names, rotation=30, ha="right")
    ax.set_ylabel("Gate Weight (Softmax)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_group_attention_heatmap(
    attn_matrix: np.ndarray,  # (9, 9) or (N, 9, 9)
    group_names: Sequence[str] = GROUP_NAMES,
    title: str = "Group Multi-Head Attention Heatmap",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot 9x9 attention heatmap showing inter-group attention weights."""
    if attn_matrix.ndim == 3:
        attn_matrix = np.mean(attn_matrix, axis=0)  # Average over samples

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        attn_matrix,
        annot=True,
        fmt=".3f",
        cmap="viridis",
        xticklabels=group_names,
        yticklabels=group_names,
        ax=ax,
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Key / Value Group", fontsize=10)
    ax.set_ylabel("Query Group", fontsize=10)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig


def plot_family_gate_profiles(
    family_gate_dict: Dict[str, np.ndarray],  # family_name -> mean_gate_weights (9,)
    group_names: Sequence[str] = GROUP_NAMES,
    title: str = "Gate Weight Profile Across Malware Families",
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    """Plot heatmap of gate weights across different malware families."""
    families = list(family_gate_dict.keys())
    data = np.vstack([family_gate_dict[f] for f in families])  # (num_families, 9)

    fig, ax = plt.subplots(figsize=(10, max(4, len(families) * 0.4)))
    sns.heatmap(
        data,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu",
        xticklabels=group_names,
        yticklabels=families,
        ax=ax,
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel("Feature Group", fontsize=10)
    ax.set_ylabel("Malware Family", fontsize=10)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()

    if save_path:
        save_figure(fig, save_path)
    return fig
