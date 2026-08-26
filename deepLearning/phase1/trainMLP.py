#!/usr/bin/env python3
"""HARUSPEX Phase 1 — Train Group-wise Residual MLP on BODMAS and evaluate on 4 protocols + EMBER cross-dataset."""

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

from deepLearning.utils.config import PHASE1_DIR, HELD_OUT_FAMILIES, TEMPORAL_CUTOFF
from deepLearning.utils.data import load_bodmas_data, load_ember_data
from deepLearning.utils.splits import random_split, family_held_out_split, temporal_split
from deepLearning.utils.plots import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_group_gate_distributions,
    plot_group_attention_heatmap,
    plot_family_gate_profiles,
)
from deepLearning.models.group_mlp import GroupMLPDetector

# Set up logging
models_dir = PHASE1_DIR / "models"
plots_dir = PHASE1_DIR / "plots"
reports_dir = PHASE1_DIR / "reports"

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
logger = logging.getLogger("phase1_trainMLP")


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
    logger.info("  HARUSPEX Phase 1 — Train Group-wise Residual MLP on BODMAS")
    logger.info("========================================================================")

    # 1. Load Dataset
    logger.info("[1/5] Loading BODMAS dataset...")
    bodmas = load_bodmas_data()
    X, y, meta = bodmas.X, bodmas.y, bodmas.metadata
    family_ids, cat_ids = bodmas.family_ids, bodmas.category_ids

    num_families = len(bodmas.family_encoder)
    num_categories = len(bodmas.category_encoder)

    # 2. Random Split Training
    logger.info("[2/5] Performing Random Split (80/20)...")
    r_split = random_split(X, y, meta, family_ids, cat_ids)

    # Internal Validation split (10% of train)
    n_val = int(len(r_split.X_train) * 0.1)
    X_tr, y_tr = r_split.X_train[:-n_val], r_split.y_train[:-n_val]
    fam_tr, cat_tr = r_split.family_train[:-n_val], r_split.cat_train[:-n_val]

    X_val, y_val = r_split.X_train[-n_val:], r_split.y_train[-n_val:]
    fam_val, cat_val = r_split.family_train[-n_val:], r_split.cat_train[-n_val:]

    logger.info("[3/5] Fitting Group-wise Residual MLP Model on BODMAS...")
    mlp = GroupMLPDetector(
        model_name_suffix="bodmas",
        num_families=num_families,
        num_categories=num_categories,
        use_gating=True,
        use_attention=True,
        use_aux_heads=True,
        max_epochs=100,
        patience=15,
        batch_size=1024,
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
    model_save_path = str(models_dir / "mlp.pth")
    mlp.save(model_save_path)

    # 3. Evaluate Protocols
    logger.info("[4/5] Running Evaluation Protocols...")
    results = {}

    # Protocol 1: Random Split
    preds = mlp.predict(r_split.X_test)
    proba = mlp.predict_proba(r_split.X_test)
    results["random_split"] = evaluate_metrics(r_split.y_test, preds, proba)
    logger.info("  Protocol 1 (Random Split): acc=%.4f, f1=%.4f, auc=%.4f",
                results["random_split"]["accuracy"], results["random_split"]["f1"], results["random_split"]["auc"])

    plot_confusion_matrix(r_split.y_test, preds, title="Group-MLP — Random Split Confusion Matrix", save_path=plots_dir / "mlp_random_cm.png")
    plot_roc_curve(r_split.y_test, proba, title="Group-MLP — Random Split ROC Curve", save_path=plots_dir / "mlp_random_roc.png")

    # Protocol 2: Family-Held-Out Split
    fh_split = family_held_out_split(X, y, meta, family_ids, cat_ids, HELD_OUT_FAMILIES)
    fh_preds = mlp.predict(fh_split.X_test)
    fh_proba = mlp.predict_proba(fh_split.X_test)
    results["family_held_out"] = evaluate_metrics(fh_split.y_test, fh_preds, fh_proba)
    logger.info("  Protocol 2 (Family-Held-Out): acc=%.4f, f1=%.4f, auc=%.4f",
                results["family_held_out"]["accuracy"], results["family_held_out"]["f1"], results["family_held_out"]["auc"])

    plot_confusion_matrix(fh_split.y_test, fh_preds, title="Group-MLP — Family-Held-Out Confusion Matrix", save_path=plots_dir / "mlp_family_cm.png")

    # Protocol 3: Temporal Split
    t_split = temporal_split(X, y, meta, family_ids, cat_ids, TEMPORAL_CUTOFF)
    t_preds = mlp.predict(t_split.X_test)
    t_proba = mlp.predict_proba(t_split.X_test)
    results["temporal_split"] = evaluate_metrics(t_split.y_test, t_preds, t_proba)
    logger.info("  Protocol 3 (Temporal Split): acc=%.4f, f1=%.4f, auc=%.4f",
                results["temporal_split"]["accuracy"], results["temporal_split"]["f1"], results["temporal_split"]["auc"])

    plot_confusion_matrix(t_split.y_test, t_preds, title="Group-MLP — Temporal Split Confusion Matrix", save_path=plots_dir / "mlp_temporal_cm.png")
    plot_roc_curve(t_split.y_test, t_proba, title="Group-MLP — Temporal Split ROC Curve", save_path=plots_dir / "mlp_temporal_roc.png")

    # Protocol 4: Cross-dataset Evaluation on EMBER
    logger.info("  Protocol 4: Loading EMBER test partition for Cross-Dataset Generalization...")
    try:
        ember_test = load_ember_data(partition="test")
        c_preds = mlp.predict(ember_test.X)
        c_proba = mlp.predict_proba(ember_test.X)
        results["cross_dataset"] = evaluate_metrics(ember_test.y, c_preds, c_proba)
        logger.info("  Protocol 4 (Cross-Dataset EMBER): acc=%.4f, f1=%.4f, auc=%.4f",
                    results["cross_dataset"]["accuracy"], results["cross_dataset"]["f1"], results["cross_dataset"]["auc"])

        plot_confusion_matrix(ember_test.y, c_preds, title="Group-MLP — EMBER Cross-Dataset Confusion Matrix", save_path=plots_dir / "mlp_cross_cm.png")
        plot_roc_curve(ember_test.y, c_proba, title="Group-MLP — EMBER Cross-Dataset ROC Curve", save_path=plots_dir / "mlp_cross_roc.png")
    except Exception as e:
        logger.warning("  Protocol 4 (Cross-Dataset EMBER) failed: %s", e)
        results["cross_dataset"] = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.0}

    # 4. Generate Visualizations Unique to Group-MLP
    logger.info("[5/5] Generating Group-MLP Visualizations (Gating Distributions & Attention Heatmaps)...")
    gate_weights = mlp.get_gate_weights(r_split.X_test[:2000])
    attn_weights = mlp.get_attention_weights(r_split.X_test[:2000])

    plot_group_gate_distributions(gate_weights, title="Group Gate Weight Distribution (BODMAS Test)", save_path=plots_dir / "mlp_gate_distribution.png")
    plot_group_attention_heatmap(attn_weights, title="Group Multi-Head Self-Attention Heatmap", save_path=plots_dir / "mlp_attention_heatmap.png")

    # Per-Family Gate Weight Profile
    top_families = meta[meta["family"] != ""]["family"].value_counts().head(8).index.tolist()
    family_gate_profile = {}
    for fam in top_families:
        fam_idx = np.where(meta["family"] == fam)[0][:100]
        if len(fam_idx) > 0:
            fam_gates = mlp.get_gate_weights(X[fam_idx])
            family_gate_profile[fam] = np.mean(fam_gates, axis=0)

    if family_gate_profile:
        plot_family_gate_profiles(family_gate_profile, title="Gate Weight Profiles per Malware Family", save_path=plots_dir / "mlp_family_gate_profiles.png")

    # Save Reports
    report_dict = {
        "model_name": mlp.model_name,
        "training_dataset": "BODMAS",
        "eval_protocols": results,
        "elapsed_seconds": time.time() - start_time,
    }

    json_report_path = reports_dir / "mlp_results.json"
    txt_report_path = reports_dir / "mlp_results.txt"

    with open(json_report_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    with open(txt_report_path, "w") as f:
        f.write("========================================================================\n")
        f.write("  HARUSPEX Phase 1 — Group-wise Residual MLP Results (BODMAS Trained)\n")
        f.write("========================================================================\n\n")
        for proto, m in results.items():
            f.write(f"Protocol: {proto}\n")
            f.write(f"  Accuracy:  {m['accuracy']:.4f}\n")
            f.write(f"  Precision: {m['precision']:.4f}\n")
            f.write(f"  Recall:    {m['recall']:.4f}\n")
            f.write(f"  F1-Score:  {m['f1']:.4f}\n")
            f.write(f"  AUC-ROC:   {m['auc']:.4f}\n\n")

    logger.info("Phase 1 Group-MLP Training & Evaluation Complete! Results saved to %s", reports_dir)


if __name__ == "__main__":
    main()
