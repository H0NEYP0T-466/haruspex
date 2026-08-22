#!/usr/bin/env python3
"""HARUSPEX Evaluation Script.

Evaluates pre-trained models (outputs/models/) on BODMAS and/or EMBER 2018.

Usage:
    cd backend
    source .venv/bin/activate
    python scripts/evaluate.py                          # both models, all protocols
    python scripts/evaluate.py --model xgboost          # single model
    python scripts/evaluate.py --dataset ember          # cross-dataset (EMBER) only
    python scripts/evaluate.py --max-ember-samples 20000
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Make `haruspex` importable when running from backend/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from haruspex.config import ensure_output_dirs
from haruspex.data.bodmas import load_bodmas
from haruspex.evaluation.metrics import compute_metrics
from haruspex.evaluation.protocols import (
    evaluate_cross_dataset,
    evaluate_family_held_out,
    evaluate_random_split,
    evaluate_temporal,
)
from haruspex.models.lightgbm_model import LightGBMDetector
from haruspex.models.xgboost_model import XGBoostDetector
from haruspex.utils.logging import get_logger
from haruspex.utils.storage import load_model, save_report

logger = get_logger(__name__)

MODEL_REGISTRY = {
    "xgboost": XGBoostDetector,
    "lightgbm": LightGBMDetector,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pre-trained HARUSPEX models on BODMAS / EMBER 2018.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        choices=["xgboost", "lightgbm", "all"],
        default="all",
        help="Which trained model(s) to evaluate.",
    )
    parser.add_argument(
        "--dataset",
        choices=["bodmas", "ember", "both"],
        default="both",
        help="BODMAS runs all 3 split protocols; EMBER runs cross-dataset eval.",
    )
    parser.add_argument(
        "--max-ember-samples",
        type=int,
        default=10_000,
        help="Subsample size for the EMBER test partition (memory/time control).",
    )
    return parser.parse_args()


def load_models(requested: str) -> list[tuple[str, object]]:
    """Load every requested model that exists on disk."""
    names = list(MODEL_REGISTRY) if requested == "all" else [requested]
    loaded = []
    for name in names:
        try:
            loaded.append((name, load_model(name, MODEL_REGISTRY[name])))
        except FileNotFoundError:
            logger.warning("Model '%s' not found in outputs/models/ — run train.py first. Skipping.", name)
    return loaded


def print_metrics_row(split_name: str, metrics: dict | None, status: str = "") -> None:
    """Print one aligned result line."""
    if metrics:
        logger.info(
            "%-22s acc=%.4f  precision=%.4f  recall=%.4f  f1=%.4f  auc=%.4f",
            split_name,
            metrics.get("accuracy", 0.0),
            metrics.get("precision", 0.0),
            metrics.get("recall", 0.0),
            metrics.get("f1", 0.0),
            metrics.get("auc_roc", 0.0),
        )
    else:
        logger.info("%-22s %s", split_name, status or "skipped")


def run_bodmas_evaluations(model, X, y, metadata) -> dict:
    """Random / family-held-out / temporal splits on BODMAS."""
    results = []
    protocols = [
        ("random", lambda: evaluate_random_split(model, X, y, metadata)),
        ("family-held-out", lambda: evaluate_family_held_out(model, X, y, metadata)),
        ("temporal", lambda: evaluate_temporal(model, X, y, metadata)),
    ]
    for name, fn in protocols:
        logger.info("--- %s [%s] ---", name, model.model_name)
        try:
            results.append(fn())
        except Exception as exc:
            logger.error("%s evaluation failed for %s: %s", name, model.model_name, exc)
            results.append({"split": name, "model": model.model_name, "status": "failed", "error": str(exc)})
    return {"model": model.model_name, "evaluations": results}


def run_ember_evaluation(model, max_samples: int) -> dict:
    """Cross-dataset evaluation: BODMAS-trained model vs EMBER 2018 test partition."""
    logger.info("--- cross-dataset (EMBER 2018) [%s] ---", model.model_name)
    try:
        return evaluate_cross_dataset(model, max_samples=max_samples)
    except Exception as exc:
        logger.error("EMBER evaluation failed for %s: %s", model.model_name, exc)
        return {"split": "cross_dataset", "model": model.model_name, "status": "failed", "error": str(exc)}


def summarize(label: str, evaluations: list[dict]) -> None:
    logger.info("")
    logger.info("=== %s ===", label)
    for ev in evaluations:
        print_metrics_row(ev.get("split", "?"), ev.get("metrics"), ev.get("status", ""))
    logger.info("")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    ensure_output_dirs()

    models = load_models(args.model)
    if not models:
        logger.error("No trained models found. Run scripts/train.py first.")
        return 1

    # Load BODMAS once, shared across models
    dataset = None
    if args.dataset in ("bodmas", "both"):
        logger.info("Loading BODMAS dataset...")
        dataset = load_bodmas()
        logger.info(
            "BODMAS ready: %d samples (%d benign / %d malware), %d features",
            dataset.n_samples, dataset.n_benign, dataset.n_malware, dataset.n_features,
        )

    all_results: dict[str, dict] = {}
    for name, model in models:
        logger.info("=" * 60)
        logger.info("Evaluating model: %s", name.upper())
        logger.info("=" * 60)

        entry: dict = {"model": name, "evaluations": []}

        if dataset is not None:
            entry["evaluations"] += run_bodmas_evaluations(
                model, dataset.X, dataset.y, dataset.metadata
            )["evaluations"]

        if args.dataset in ("ember", "both"):
            entry["evaluations"].append(run_ember_evaluation(model, args.max_ember_samples))

        summarize(f"{name.upper()} Results", entry["evaluations"])
        all_results[name] = entry

    report_path = save_report(all_results, "evaluation_results")
    logger.info("Full report saved to %s", report_path)

    # Non-zero exit if every protocol failed for every model
    any_success = any(
        ev.get("metrics") for entry in all_results.values() for ev in entry["evaluations"]
    )
    return 0 if any_success else 1


if __name__ == "__main__":
    sys.exit(main())
