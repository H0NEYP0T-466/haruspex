"""Evaluation protocols for HARUSPEX malware detection.

Four evaluation strategies to assess model generalisation:
  1. Random split — baseline sanity check
  2. Family-held-out — zero-day (family-based) evaluation
  3. Temporal split — zero-day (time-based) evaluation
  4. Cross-dataset — train on BODMAS, test on EMBER (placeholder)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from haruspex.config import HELD_OUT_FAMILIES, TEMPORAL_CUTOFF, get_feature_names
from haruspex.data.splits import (
    SplitResult,
    family_held_out_split,
    random_split,
    temporal_split,
)
from haruspex.evaluation.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
)
from haruspex.models.base import MalwareDetector
from haruspex.utils.logging import get_logger
from haruspex.utils.storage import save_report

logger = get_logger(__name__)


def evaluate_on_split(
    model: MalwareDetector,
    split: SplitResult,
    plot_prefix: str = "",
) -> dict:
    """Evaluate a trained model on a given split.

    Args:
        model: Trained MalwareDetector.
        split: SplitResult with test data.
        plot_prefix: Prefix for saved plot filenames.

    Returns:
        Dictionary of metrics.
    """
    y_pred = model.predict(split.X_test)
    y_proba = model.predict_proba(split.X_test)

    metrics = compute_metrics(split.y_test, y_pred, y_proba)

    # Generate plots
    prefix = f"{plot_prefix}_{split.split_name}" if plot_prefix else split.split_name
    plot_confusion_matrix(
        split.y_test, y_pred,
        save_name=f"{prefix}_confusion_matrix",
        title=f"Confusion Matrix — {model.model_name} ({split.split_name})",
    )
    try:
        plot_roc_curve(
            split.y_test, y_proba,
            save_name=f"{prefix}_roc_curve",
            title=f"ROC Curve — {model.model_name} ({split.split_name})",
        )
    except ValueError:
        logger.warning("ROC curve skipped for %s (single class)", split.split_name)

    logger.info(
        "Evaluation [%s/%s]: acc=%.4f, f1=%.4f, auc=%.4f",
        model.model_name,
        split.split_name,
        metrics["accuracy"],
        metrics["f1"],
        metrics.get("auc_roc", 0.0),
    )

    return {
        "split": split.split_name,
        "model": model.model_name,
        "metrics": metrics,
        "train_size": split.train_size,
        "test_size": split.test_size,
    }


def evaluate_random_split(
    model: MalwareDetector,
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
) -> dict:
    """Evaluate on standard random 80/20 split."""
    split = random_split(X, y, metadata)
    return evaluate_on_split(model, split, plot_prefix=model.model_name)


def evaluate_family_held_out(
    model: MalwareDetector,
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    held_out_families: Optional[list[str]] = None,
) -> dict:
    """Evaluate on family-held-out split (zero-day by family)."""
    families = held_out_families or HELD_OUT_FAMILIES
    split = family_held_out_split(X, y, metadata, held_out_families=families)
    result = evaluate_on_split(model, split, plot_prefix=model.model_name)
    result["held_out_families"] = families
    return result


def evaluate_temporal(
    model: MalwareDetector,
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    cutoff_date: Optional[str] = None,
) -> dict:
    """Evaluate on temporal split (zero-day by time)."""
    cutoff = cutoff_date or TEMPORAL_CUTOFF
    split = temporal_split(X, y, metadata, cutoff_date=cutoff)
    result = evaluate_on_split(model, split, plot_prefix=model.model_name)
    result["cutoff_date"] = cutoff
    return result


def evaluate_cross_dataset(
    model: MalwareDetector,
    max_samples: Optional[int] = 10000,
) -> dict:
    """Evaluate cross-dataset: train on BODMAS → test on EMBER 2018.

    Loads the EMBER benchmark test set and evaluates the trained BODMAS model
    to measure cross-domain generalization.
    """
    from haruspex.data.ember import is_ember_available, load_ember

    if not is_ember_available():
        logger.info("EMBER not available — skipping cross-dataset evaluation")
        return {
            "split": "cross_dataset",
            "model": model.model_name,
            "metrics": {},
            "status": "skipped",
            "reason": "EMBER dataset not available",
        }

    try:
        X_ember, y_ember, meta_ember = load_ember(max_samples=max_samples)

        split = SplitResult(
            X_train=np.empty((0, 2381), dtype=np.float32),
            y_train=np.empty((0,), dtype=np.int32),
            X_test=X_ember,
            y_test=y_ember,
            train_indices=np.array([]),
            test_indices=np.arange(len(y_ember)),
            metadata_train=pd.DataFrame(),
            metadata_test=meta_ember,
            split_name="cross_dataset",
        )

        result = evaluate_on_split(model, split, plot_prefix=model.model_name)
        result["status"] = "success"
        return result

    except Exception as e:
        logger.warning("Cross-dataset evaluation error: %s", e)
        return {
            "split": "cross_dataset",
            "model": model.model_name,
            "metrics": {},
            "status": "failed",
            "error": str(e),
        }


def run_all_evaluations(
    model: MalwareDetector,
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    held_out_families: Optional[list[str]] = None,
    cutoff_date: Optional[str] = None,
) -> dict:
    """Run all four evaluation protocols and save results.

    Args:
        model: Trained MalwareDetector.
        X: Full feature matrix.
        y: Full label array.
        metadata: Full metadata DataFrame.
        held_out_families: Families to hold out.
        cutoff_date: Temporal split cutoff date.

    Returns:
        Combined results dict with all evaluation outcomes.
    """
    logger.info("Running all evaluations for %s", model.model_name)

    results = {
        "model": model.model_name,
        "evaluations": [],
    }

    # 1. Random split
    logger.info("--- Evaluation 1/4: Random split ---")
    results["evaluations"].append(evaluate_random_split(model, X, y, metadata))

    # 2. Family-held-out
    logger.info("--- Evaluation 2/4: Family-held-out ---")
    results["evaluations"].append(
        evaluate_family_held_out(model, X, y, metadata, held_out_families)
    )

    # 3. Temporal split
    logger.info("--- Evaluation 3/4: Temporal split ---")
    results["evaluations"].append(
        evaluate_temporal(model, X, y, metadata, cutoff_date)
    )

    # 4. Cross-dataset (placeholder)
    logger.info("--- Evaluation 4/4: Cross-dataset ---")
    results["evaluations"].append(evaluate_cross_dataset(model))

    # Generate feature importance plot
    try:
        importances = model.get_feature_importance()
        feature_names = get_feature_names()
        plot_feature_importance(
            importances,
            feature_names,
            save_name=f"{model.model_name}_feature_importance",
            title=f"Feature Importance — {model.model_name}",
        )
    except Exception as e:
        logger.warning("Feature importance plot failed: %s", e)

    # Save combined results
    save_report(results, f"evaluation_results_{model.model_name}")

    return results
