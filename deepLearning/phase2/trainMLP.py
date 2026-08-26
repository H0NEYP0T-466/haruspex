#!/usr/bin/env python3
"""HARUSPEX Phase 2 — Train Group-wise Residual MLP on EMBER 2018 and evaluate cross-dataset on BODMAS."""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# Ensure root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from deepLearning.utils.config import PHASE2_DIR
from deepLearning.utils.data import load_ember_data, load_bodmas_data
from deepLearning.utils.splits import random_split
from deepLearning.utils.plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_group_gate_distributions,
    plot_group_attention_heatmap,
)
from deepLearning.models.group_mlp import GroupMLPDetector

# Set up logging
models_dir = PHASE2_DIR / "models"
plots_dir = PHASE2_DIR / "plots"
reports_dir = PHASE2_DIR / "reports"

for d in (models_dir, plots_dir, reports_dir):
    d.mkdir(parents=True, exist_ok=True)

log_file = reports_dir / "trainMLP.log"
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file), mode="w"),
    ],
)
logger = logging.getLogger("phase2_trainMLP")


def evaluate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict[str, float]:
    """Calculate standard evaluation metrics."""
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    try:
        auc_val = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        auc_val = float("nan")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc_val,
    }


def main():
    start_time = time.time()
    logger.info("========================================================================")
    logger.info("  HARUSPEX Phase 2 — Train Group-wise Residual MLP on EMBER 2018")
    logger.info("========================================================================")

    # 1. Load EMBER Train Partition
    logger.info("[1/5] Loading EMBER 2018 train partition (600K labeled samples)...")
    ember_train = load_ember_data(partition="train")
    X, y, meta = ember_train.X, ember_train.y, ember_train.metadata
    family_ids, cat_ids = ember_train.family_ids, ember_train.category_ids

    # 2. Random Split Training
    logger.info("[2/5] Performing Random Split (80/20)...")
    r_split = random_split(X, y, meta, family_ids, cat_ids)

    n_val = int(len(r_split.X_train) * 0.1)
    X_tr, y_tr = r_split.X_train[:-n_val], r_split.y_train[:-n_val]
    fam_tr, cat_tr = r_split.family_train[:-n_val], r_split.cat_train[:-n_val]

    X_val, y_val = r_split.X_train[-n_val:], r_split.y_train[-n_val:]
    fam_val, cat_val = r_split.family_train[-n_val:], r_split.cat_train[-n_val:]

    logger.info("[3/5] Fitting Group-wise Residual MLP Model on EMBER...")
    mlp = GroupMLPDetector(
        model_name_suffix="ember",
        num_families=1,
        num_categories=2,
        use_gating=True,
        use_attention=True,
        use_aux_heads=False,  # EMBER doesn't have family labels
        max_epochs=100,
        patience=15,
        batch_size=512,
    )

    mlp.fit(
        X_train=X_tr,
        y_train=y_tr,
        X_val=X_val,
        y_val=y_val,
        family_train=fam_tr,
        family_val=fam_val,
        cat_train=cat_tr,
        cat_val=cat_val,
    )

    # Save checkpoint
    model_save_path = str(models_dir / "ember_mlp.pth")
    mlp.save(model_save_path)

    # 3. Evaluate Protocols
    logger.info("[4/5] Running Evaluation Protocols...")
    results = {}

    # Internal Test Split
    preds = mlp.predict(r_split.X_test)
    proba = mlp.predict_proba(r_split.X_test)
    results["internal_test"] = evaluate_metrics(r_split.y_test, preds, proba)
    logger.info("  EMBER Internal Test: acc=%.4f, f1=%.4f, auc=%.4f",
                results["internal_test"]["accuracy"], results["internal_test"]["f1"], results["internal_test"]["auc"])

    # Official EMBER Test Partition
    logger.info("  Evaluating on Official EMBER Test Partition (200K samples)...")
    ember_test = load_ember_data(partition="test")
    off_preds = mlp.predict(ember_test.X)
    off_proba = mlp.predict_proba(ember_test.X)
    results["official_ember_test"] = evaluate_metrics(ember_test.y, off_preds, off_proba)
    logger.info("  Official EMBER Test: acc=%.4f, f1=%.4f, auc=%.4f",
                results["official_ember_test"]["accuracy"], results["official_ember_test"]["f1"], results["official_ember_test"]["auc"])

    plot_confusion_matrix(ember_test.y, off_preds, title="Group-MLP — EMBER Test Confusion Matrix", save_path=plots_dir / "mlp_ember_test_cm.png")
    plot_roc_curve(ember_test.y, off_proba, title="Group-MLP — EMBER Test ROC Curve", save_path=plots_dir / "mlp_ember_test_roc.png")

    # Cross-dataset Evaluation on BODMAS
    logger.info("  Evaluating Cross-Dataset Generalization on BODMAS (134K samples)...")
    try:
        bodmas = load_bodmas_data()
        c_preds = mlp.predict(bodmas.X)
        c_proba = mlp.predict_proba(bodmas.X)
        results["cross_dataset_bodmas"] = evaluate_metrics(bodmas.y, c_preds, c_proba)
        logger.info("  Cross-Dataset (BODMAS): acc=%.4f, f1=%.4f, auc=%.4f",
                    results["cross_dataset_bodmas"]["accuracy"], results["cross_dataset_bodmas"]["f1"], results["cross_dataset_bodmas"]["auc"])

        plot_confusion_matrix(bodmas.y, c_preds, title="Group-MLP → BODMAS Cross-Dataset Confusion Matrix", save_path=plots_dir / "mlp_bodmas_cross_cm.png")
        plot_roc_curve(bodmas.y, c_proba, title="Group-MLP → BODMAS Cross-Dataset ROC Curve", save_path=plots_dir / "mlp_bodmas_cross_roc.png")
    except Exception as e:
        logger.warning("  Cross-dataset BODMAS evaluation failed: %s", e)
        results["cross_dataset_bodmas"] = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.0}

    # 4. Generate Visualizations Unique to Group-MLP
    logger.info("[5/5] Generating Group-MLP Visualizations (Gating Distributions & Attention Heatmaps)...")
    gate_weights = mlp.get_gate_weights(ember_test.X[:2000])
    attn_weights = mlp.get_attention_weights(ember_test.X[:2000])

    plot_group_gate_distributions(gate_weights, title="Group Gate Weight Distribution (EMBER Test)", save_path=plots_dir / "mlp_gate_distribution.png")
    plot_group_attention_heatmap(attn_weights, title="Group Multi-Head Self-Attention Heatmap (EMBER)", save_path=plots_dir / "mlp_attention_heatmap.png")

    report_dict = {
        "model_name": mlp.model_name,
        "training_dataset": "EMBER_2018",
        "eval_protocols": results,
        "elapsed_seconds": time.time() - start_time,
    }

    json_report_path = reports_dir / "mlp_results.json"
    txt_report_path = reports_dir / "mlp_results.txt"

    with open(json_report_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    with open(txt_report_path, "w") as f:
        f.write("========================================================================\n")
        f.write("  HARUSPEX Phase 2 — Group-wise Residual MLP Results (EMBER Trained)\n")
        f.write("========================================================================\n\n")
        for proto, m in results.items():
            f.write(f"Protocol: {proto}\n")
            f.write(f"  Accuracy:  {m['accuracy']:.4f}\n")
            f.write(f"  Precision: {m['precision']:.4f}\n")
            f.write(f"  Recall:    {m['recall']:.4f}\n")
            f.write(f"  F1-Score:  {m['f1']:.4f}\n")
            f.write(f"  AUC-ROC:   {m['auc']:.4f}\n\n")

    logger.info("Phase 2 Group-MLP Training & Evaluation Complete! Results saved to %s", reports_dir)


if __name__ == "__main__":
    main()
