#!/usr/bin/env python3
"""HARUSPEX Phase 1 — Kaggle TabNet Training & Evaluation Script for GPU T4.

Instructions for Kaggle:
  1. Create a Kaggle Notebook with GPU T4 enabled.
  2. Git clone your repo: git clone https://github.com/yourusername/HARUSPEX.git
  3. cd HARUSPEX && python deepLearning/phase1/kaggletabnet.py
"""

from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path

# --- Auto-install pytorch-tabnet on Kaggle ---
try:
    import pytorch_tabnet
except ImportError:
    print("[Kaggle Setup] Installing pytorch-tabnet...")
    os.system("pip install -q pytorch-tabnet")
    import pytorch_tabnet

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

# --- Resolve Root and Backend Paths ---
_SCRIPT_DIR = Path(__file__).resolve().parent
_DL_DIR = _SCRIPT_DIR.parent
_PROJECT_ROOT = _DL_DIR.parent
_BACKEND_DIR = _PROJECT_ROOT / "backend"

for p in [_PROJECT_ROOT, _BACKEND_DIR, _DL_DIR]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from deepLearning.utils.config import HELD_OUT_FAMILIES, TEMPORAL_CUTOFF, get_device
from deepLearning.utils.data import load_bodmas_data, load_ember_data
from deepLearning.utils.splits import random_split, family_held_out_split, temporal_split
from deepLearning.utils.plots import plot_confusion_matrix, plot_roc_curve, plot_feature_importance
from deepLearning.models.tabnet_model import TabNetDetector

# Kaggle-aware output directory (saves directly to Kaggle /kaggle/working/ output)
if Path("/kaggle/working").exists():
    out_base = Path("/kaggle/working") / "deepLearning" / "phase1"
else:
    out_base = _DL_DIR / "phase1"

models_dir = out_base / "models"
plots_dir = out_base / "plots"
reports_dir = out_base / "reports"

for d in (models_dir, plots_dir, reports_dir):
    d.mkdir(parents=True, exist_ok=True)

import logging
log_file = reports_dir / "kaggletabnet.log"
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file), mode="w"),
    ],
)
logger = logging.getLogger("kaggle_tabnet")


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


def find_bodmas_dir() -> Path:
    """Find BODMAS dataset path on local system or Kaggle dataset mounts."""
    possible_paths = [
        Path("/kaggle/input/datasets/h0neyp0t/bodmas"),
        Path("/kaggle/input/h0neyp0t/bodmas"),
        Path("/kaggle/input/bodmas/BODMAS"),
        Path("/kaggle/input/bodmas"),
        Path("/kaggle/input/haruspex-dataset/BODMAS"),
        _BACKEND_DIR / "dataset" / "BODMAS",
    ]
    for p in possible_paths:
        if p.exists() and ((p / "bodmas_X.dat").exists() or (p / "bodmas.npz").exists()):
            logger.info("Found BODMAS dataset at: %s", p)
            return p

    # Dynamic search under /kaggle/input if available
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for root, _, files in os.walk(kaggle_input):
            if "bodmas.npz" in files or "bodmas_X.dat" in files:
                found_path = Path(root)
                logger.info("Dynamically discovered BODMAS dataset at: %s", found_path)
                return found_path

    logger.info("Using default BODMAS dataset directory: %s", _BACKEND_DIR / "dataset" / "BODMAS")
    return _BACKEND_DIR / "dataset" / "BODMAS"


def find_ember_dir() -> Path:
    """Find EMBER dataset path on local system or Kaggle dataset mounts."""
    possible_paths = [
        Path("/kaggle/input/datasets/h0neyp0t/ember-2018"),
        Path("/kaggle/input/datasets/h0neyp0t/Ember_2018"),
        Path("/kaggle/input/ember-2018"),
        Path("/kaggle/input/ember_2018"),
        Path("/kaggle/input/Ember_2018"),
        _BACKEND_DIR / "dataset" / "ember2018",
    ]
    for p in possible_paths:
        if p.exists() and ((p / "X_test.dat").exists() or (p / "X_train.dat").exists() or (p / "test_features.jsonl").exists()):
            logger.info("Found EMBER dataset at: %s", p)
            return p

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for root, _, files in os.walk(kaggle_input):
            if "X_test.dat" in files or "test_features.jsonl" in files:
                found_path = Path(root)
                logger.info("Dynamically discovered EMBER dataset at: %s", found_path)
                return found_path

    return _BACKEND_DIR / "dataset" / "ember2018"


def main():
    start_time = time.time()
    logger.info("========================================================================")
    logger.info("  HARUSPEX Kaggle Pipeline — TabNet Training on GPU T4 (BODMAS)")
    logger.info("========================================================================")
    
    device = get_device()
    logger.info("PyTorch Execution Device: %s (CUDA Available: %s)", device, torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU Model: %s", torch.cuda.get_device_name(0))

    # 1. Load Dataset
    logger.info("[1/5] Loading BODMAS dataset...")
    bodmas_path = find_bodmas_dir()
    bodmas = load_bodmas_data(bodmas_dir=bodmas_path)
    X, y, meta = bodmas.X, bodmas.y, bodmas.metadata
    family_ids, cat_ids = bodmas.family_ids, bodmas.category_ids

    # 2. Random Split Training
    logger.info("[2/5] Performing Random Split (80/20)...")
    r_split = random_split(X, y, meta, family_ids, cat_ids)

    n_val = int(len(r_split.X_train) * 0.1)
    X_tr, y_tr = r_split.X_train[:-n_val], r_split.y_train[:-n_val]
    X_val, y_val = r_split.X_train[-n_val:], r_split.y_train[-n_val:]

    # Batch size and steps for TabNet
    batch_size = 1024 if torch.cuda.is_available() else 2048

    logger.info("[3/5] Fitting TabNet Model on GPU T4...")
    tabnet = TabNetDetector(
        model_name_suffix="bodmas",
        max_epochs=100,
        patience=15,
        batch_size=batch_size,
    )
    tabnet.fit(X_tr, y_tr, X_val, y_val)

    # Save trained model zip
    model_save_path = str(models_dir / "tabnet")
    tabnet.save(model_save_path)
    logger.info("TabNet model saved to %s", model_save_path)

    # 3. Evaluate Protocols
    logger.info("[4/5] Running Evaluation Protocols...")
    results = {}

    # Protocol 1: Random Split Test
    preds = tabnet.predict(r_split.X_test)
    proba = tabnet.predict_proba(r_split.X_test)
    results["random_split"] = evaluate_metrics(r_split.y_test, preds, proba)
    logger.info("  Protocol 1 (Random Split): acc=%.4f, f1=%.4f, auc=%.4f",
                results["random_split"]["accuracy"], results["random_split"]["f1"], results["random_split"]["auc"])

    plot_confusion_matrix(r_split.y_test, preds, title="TabNet — Random Split Confusion Matrix", save_path=plots_dir / "tabnet_random_cm.png")
    plot_roc_curve(r_split.y_test, proba, title="TabNet — Random Split ROC Curve", save_path=plots_dir / "tabnet_random_roc.png")

    # Protocol 2: Family-Held-Out Split (trained on fh_split.X_train to avoid data leakage)
    logger.info("  Protocol 2: Fitting dedicated Family-Held-Out model on train split...")
    fh_split = family_held_out_split(X, y, meta, family_ids, cat_ids, HELD_OUT_FAMILIES)
    n_val_fh = int(len(fh_split.X_train) * 0.1)
    tabnet_fh = TabNetDetector(model_name_suffix="bodmas_fh", max_epochs=50, patience=10, batch_size=batch_size)
    tabnet_fh.fit(fh_split.X_train[:-n_val_fh], fh_split.y_train[:-n_val_fh], fh_split.X_train[-n_val_fh:], fh_split.y_train[-n_val_fh:])
    fh_preds = tabnet_fh.predict(fh_split.X_test)
    fh_proba = tabnet_fh.predict_proba(fh_split.X_test)
    results["family_held_out"] = evaluate_metrics(fh_split.y_test, fh_preds, fh_proba)
    logger.info("  Protocol 2 (Family-Held-Out): acc=%.4f, f1=%.4f, auc=%.4f",
                results["family_held_out"]["accuracy"], results["family_held_out"]["f1"], results["family_held_out"]["auc"])

    plot_confusion_matrix(fh_split.y_test, fh_preds, title="TabNet — Family-Held-Out Confusion Matrix", save_path=plots_dir / "tabnet_family_cm.png")

    # Protocol 3: Temporal Split (trained on t_split.X_train to avoid data leakage)
    logger.info("  Protocol 3: Fitting dedicated Temporal model on train split...")
    t_split = temporal_split(X, y, meta, family_ids, cat_ids, TEMPORAL_CUTOFF)
    n_val_t = int(len(t_split.X_train) * 0.1)
    tabnet_t = TabNetDetector(model_name_suffix="bodmas_t", max_epochs=50, patience=10, batch_size=batch_size)
    tabnet_t.fit(t_split.X_train[:-n_val_t], t_split.y_train[:-n_val_t], t_split.X_train[-n_val_t:], t_split.y_train[-n_val_t:])
    t_preds = tabnet_t.predict(t_split.X_test)
    t_proba = tabnet_t.predict_proba(t_split.X_test)
    results["temporal_split"] = evaluate_metrics(t_split.y_test, t_preds, t_proba)
    logger.info("  Protocol 3 (Temporal Split): acc=%.4f, f1=%.4f, auc=%.4f",
                results["temporal_split"]["accuracy"], results["temporal_split"]["f1"], results["temporal_split"]["auc"])

    plot_confusion_matrix(t_split.y_test, t_preds, title="TabNet — Temporal Split Confusion Matrix", save_path=plots_dir / "tabnet_temporal_cm.png")
    plot_roc_curve(t_split.y_test, t_proba, title="TabNet — Temporal Split ROC Curve", save_path=plots_dir / "tabnet_temporal_roc.png")

    # Protocol 4: Cross-dataset Evaluation on EMBER
    logger.info("  Protocol 4: Checking EMBER test partition for Cross-Dataset Generalization...")
    try:
        ember_path = find_ember_dir()
        ember_test = load_ember_data(ember_dir=ember_path, partition="test")
        c_preds = tabnet.predict(ember_test.X)
        c_proba = tabnet.predict_proba(ember_test.X)
        results["cross_dataset"] = evaluate_metrics(ember_test.y, c_preds, c_proba)
        logger.info("  Protocol 4 (Cross-Dataset EMBER): acc=%.4f, f1=%.4f, auc=%.4f",
                    results["cross_dataset"]["accuracy"], results["cross_dataset"]["f1"], results["cross_dataset"]["auc"])

        plot_confusion_matrix(ember_test.y, c_preds, title="TabNet — EMBER Cross-Dataset Confusion Matrix", save_path=plots_dir / "tabnet_cross_cm.png")
        plot_roc_curve(ember_test.y, c_proba, title="TabNet — EMBER Cross-Dataset ROC Curve", save_path=plots_dir / "tabnet_cross_roc.png")
    except Exception as e:
        logger.warning("  Protocol 4 (Cross-Dataset EMBER) skipped: %s", e)
        results["cross_dataset"] = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.0}

    # 4. Feature Importance & Reports
    logger.info("[5/5] Generating TabNet Feature Importances & Reports...")
    importance = tabnet.get_feature_importance(bodmas.X)
    plot_feature_importance(importance, bodmas.feature_names, top_k=20, title="TabNet Feature Importance (BODMAS)", save_path=plots_dir / "tabnet_feature_importance.png")

    report_dict = {
        "model_name": tabnet.model_name,
        "training_dataset": "BODMAS",
        "device_used": str(device),
        "eval_protocols": results,
        "elapsed_seconds": time.time() - start_time,
    }

    json_report_path = reports_dir / "tabnet_results.json"
    txt_report_path = reports_dir / "tabnet_results.txt"

    with open(json_report_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    with open(txt_report_path, "w") as f:
        f.write("========================================================================\n")
        f.write("  HARUSPEX Kaggle Phase 1 — TabNet Results (GPU Accelerated)\n")
        f.write("========================================================================\n\n")
        for proto, m in results.items():
            f.write(f"Protocol: {proto}\n")
            f.write(f"  Accuracy:  {m['accuracy']:.4f}\n")
            f.write(f"  Precision: {m['precision']:.4f}\n")
            f.write(f"  Recall:    {m['recall']:.4f}\n")
            f.write(f"  F1-Score:  {m['f1']:.4f}\n")
            f.write(f"  AUC-ROC:   {m['auc']:.4f}\n\n")

    logger.info("Kaggle TabNet Training Finished Successfully! Artifacts saved to %s", out_base)


if __name__ == "__main__":
    main()
