"""Unified training loop for HARUSPEX malware detection models."""

from __future__ import annotations

from typing import Optional

import numpy as np

from haruspex.config import RANDOM_SEED, ensure_output_dirs
from haruspex.data.splits import SplitResult, random_split
from haruspex.models.base import MalwareDetector
from haruspex.models.lightgbm_model import LightGBMDetector
from haruspex.models.xgboost_model import XGBoostDetector
from haruspex.utils.logging import get_logger
from haruspex.utils.storage import save_model

logger = get_logger(__name__)


def train_model(
    model: MalwareDetector,
    split: SplitResult,
    save_name: Optional[str] = None,
) -> dict[str, float]:
    """Train a single model on a given split.

    Uses a 90/10 internal train/val partition from the training set for
    early stopping / monitoring, then evaluates on the held-out test set.

    Args:
        model: An untrained MalwareDetector instance.
        split: A SplitResult containing train/test data.
        save_name: If provided, save the trained model under this name.

    Returns:
        Dictionary of training and test metrics.
    """
    # Create internal validation set from training data
    n_train = split.X_train.shape[0]
    n_val = max(int(n_train * 0.1), 100)
    rng = np.random.RandomState(RANDOM_SEED)
    val_idx = rng.choice(n_train, size=n_val, replace=False)
    train_idx = np.setdiff1d(np.arange(n_train), val_idx)

    X_train = split.X_train[train_idx]
    y_train = split.y_train[train_idx]
    X_val = split.X_train[val_idx]
    y_val = split.y_train[val_idx]

    # Train
    train_metrics = model.fit(X_train, y_train, X_val, y_val)

    # Evaluate on test set
    test_pred = model.predict(split.X_test)
    test_proba = model.predict_proba(split.X_test)
    test_acc = float(np.mean(test_pred == split.y_test))

    from sklearn.metrics import f1_score, roc_auc_score

    test_f1 = float(f1_score(split.y_test, test_pred))
    try:
        test_auc = float(roc_auc_score(split.y_test, test_proba))
    except ValueError:
        test_auc = 0.0  # Single class in test set

    metrics = {
        **train_metrics,
        "test_accuracy": test_acc,
        "test_f1": test_f1,
        "test_auc": test_auc,
    }

    logger.info(
        "%s on %s split: test_acc=%.4f, test_f1=%.4f, test_auc=%.4f",
        model.model_name,
        split.split_name,
        test_acc,
        test_f1,
        test_auc,
    )

    # Save if requested
    if save_name:
        ensure_output_dirs()
        save_model(model, save_name)

    return metrics


def train_all_models(
    X: np.ndarray,
    y: np.ndarray,
    metadata: "pd.DataFrame",
    save: bool = True,
) -> dict[str, dict]:
    """Train both XGBoost and LightGBM on a random split.

    Args:
        X: Full feature matrix.
        y: Full label array.
        metadata: Full metadata DataFrame.
        save: Whether to save trained models to disk.

    Returns:
        Dict mapping model_name → {"model": MalwareDetector, "metrics": dict}.
    """
    import pandas as pd

    logger.info("Training all models on random split")

    split = random_split(X, y, metadata)

    results: dict[str, dict] = {}

    for ModelClass, name in [
        (XGBoostDetector, "xgboost"),
        (LightGBMDetector, "lightgbm"),
    ]:
        model = ModelClass()
        metrics = train_model(model, split, save_name=name if save else None)
        results[name] = {"model": model, "metrics": metrics, "split": split}

    # Log comparison
    logger.info("=== Model Comparison (Random Split) ===")
    for name, result in results.items():
        m = result["metrics"]
        logger.info(
            "  %s: acc=%.4f, f1=%.4f, auc=%.4f",
            name,
            m["test_accuracy"],
            m["test_f1"],
            m["test_auc"],
        )

    return results
