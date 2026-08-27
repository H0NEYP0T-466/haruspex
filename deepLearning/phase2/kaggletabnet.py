#!/usr/bin/env python3
"""HARUSPEX Phase 2 — Kaggle TabNet Training on EMBER 2018 + Cross-Dataset Evaluation on BODMAS.

Instructions for Kaggle:
  1. Create a Kaggle Notebook with GPU T4 enabled.
  2. Attach datasets: 'Ember_2018' and 'bodmas'.
  3. Git clone your repo: git clone https://<TOKEN>@github.com/H0NEYP0T-466/haruspex.git
  4. cd haruspex && python deepLearning/phase2/kaggletabnet.py
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

from deepLearning.utils.config import get_device
from deepLearning.utils.data import load_ember_data, load_bodmas_data
from deepLearning.utils.plots import plot_confusion_matrix, plot_roc_curve, plot_feature_importance
from deepLearning.models.tabnet_model import TabNetDetector

# Kaggle-aware output directory
if Path("/kaggle/working").exists():
    out_base = Path("/kaggle/working") / "deepLearning" / "phase2"
else:
    out_base = _DL_DIR / "phase2"

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
logger = logging.getLogger("kaggle_tabnet_phase2")


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
        if p.exists() and ((p / "X_train.dat").exists() or (p / "X_test.dat").exists() or (p / "train_features_0.jsonl").exists()):
            logger.info("Found EMBER dataset at: %s", p)
            return p

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for root, _, files in os.walk(kaggle_input):
            if "X_train.dat" in files or "train_features_0.jsonl" in files:
                found_path = Path(root)
                logger.info("Dynamically discovered EMBER dataset at: %s", found_path)
                return found_path

    logger.info("Using default EMBER dataset directory: %s", _BACKEND_DIR / "dataset" / "ember2018")
    return _BACKEND_DIR / "dataset" / "ember2018"


def find_bodmas_dir() -> Path:
    """Find BODMAS dataset path on local system or Kaggle dataset mounts."""
    possible_paths = [
        Path("/kaggle/input/datasets/h0neyp0t/bodmas"),
        Path("/kaggle/input/h0neyp0t/bodmas"),
        Path("/kaggle/input/bodmas/BODMAS"),
        Path("/kaggle/input/bodmas"),
        _BACKEND_DIR / "dataset" / "BODMAS",
    ]
    for p in possible_paths:
        if p.exists() and ((p / "bodmas_X.dat").exists() or (p / "bodmas.npz").exists()):
            logger.info("Found BODMAS dataset at: %s", p)
            return p

    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        for root, _, files in os.walk(kaggle_input):
            if "bodmas.npz" in files or "bodmas_X.dat" in files:
                found_path = Path(root)
                logger.info("Dynamically discovered BODMAS dataset at: %s", found_path)
                return found_path

    return _BACKEND_DIR / "dataset" / "BODMAS"


def main():
    start_time = time.time()
    logger.info("========================================================================")
    logger.info("  HARUSPEX Kaggle Phase 2 — TabNet Training on EMBER 2018 (GPU T4)")
    logger.info("========================================================================")

    device = get_device()
    logger.info("PyTorch Execution Device: %s (CUDA Available: %s)", device, torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("GPU Model: %s", torch.cuda.get_device_name(0))

    # 1. Load EMBER Dataset
    logger.info("[1/4] Loading EMBER dataset (Train & Test partitions)...")
    ember_path = find_ember_dir()
    ember_train = load_ember_data(ember_dir=ember_path, partition="train")
    ember_test = load_ember_data(ember_dir=ember_path, partition="test")

    # Internal Validation split (10% of train)
    n_val = int(len(ember_train.X) * 0.1)
    X_tr, y_tr = ember_train.X[:-n_val], ember_train.y[:-n_val]
    X_val, y_val = ember_train.X[-n_val:], ember_train.y[-n_val:]

    batch_size = 2048 if torch.cuda.is_available() else 1024

    # 2. Fit TabNet Classifier
    logger.info("[2/4] Fitting TabNet Model on EMBER...")
    tabnet = TabNetDetector(
        model_name_suffix="ember",
        max_epochs=100,
        patience=15,
        batch_size=batch_size,
    )
    tabnet.fit(X_tr, y_tr, X_val, y_val)

    # Free training split memory before test evaluation
    del X_tr, y_tr, X_val, y_val
    gc.collect()

    # 3. Evaluate EMBER Test Partition
    logger.info("[3/4] Evaluating on EMBER 2018 Test Partition...")
    test_preds = tabnet.predict(ember_test.X, batch_size=4096)
    test_proba = tabnet.predict_proba(ember_test.X, batch_size=4096)
    ember_results = evaluate_metrics(ember_test.y, test_preds, test_proba)
    logger.info("  EMBER 2018 Test Performance: acc=%.4f, f1=%.4f, auc=%.4f",
                ember_results["accuracy"], ember_results["f1"], ember_results["auc"])

    plot_confusion_matrix(ember_test.y, test_preds, title="TabNet — EMBER 2018 Test Confusion Matrix", save_path=plots_dir / "ember_tabnet_test_cm.png")
    plot_roc_curve(ember_test.y, test_proba, title="TabNet — EMBER 2018 Test ROC Curve", save_path=plots_dir / "ember_tabnet_test_roc.png")

    # 4. Evaluate Cross-Dataset Generalization on BODMAS
    logger.info("[4/4] Evaluating Cross-Dataset Generalization on BODMAS...")
    cross_results = {}
    try:
        bodmas_path = find_bodmas_dir()
        bodmas = load_bodmas_data(bodmas_dir=bodmas_path)
        bodmas_preds = tabnet.predict(bodmas.X)
        bodmas_proba = tabnet.predict_proba(bodmas.X)
        cross_results = evaluate_metrics(bodmas.y, bodmas_preds, bodmas_proba)
        logger.info("  Cross-Dataset BODMAS Generalization: acc=%.4f, f1=%.4f, auc=%.4f",
                    cross_results["accuracy"], cross_results["f1"], cross_results["auc"])

        plot_confusion_matrix(bodmas.y, bodmas_preds, title="TabNet — Cross-Dataset BODMAS Confusion Matrix", save_path=plots_dir / "ember_tabnet_cross_bodmas_cm.png")
        plot_roc_curve(bodmas.y, bodmas_proba, title="TabNet — Cross-Dataset BODMAS ROC Curve", save_path=plots_dir / "ember_tabnet_cross_bodmas_roc.png")
    except Exception as e:
        logger.warning("  Cross-Dataset BODMAS Evaluation skipped: %s", e)
        cross_results = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "auc": 0.0}

    # Generate Feature Importance
    logger.info("Generating TabNet Feature Importances...")
    importance = tabnet.get_feature_importance()
    plot_feature_importance(importance, ember_train.feature_names, top_k=20, title="TabNet Feature Importance (EMBER 2018)", save_path=plots_dir / "ember_tabnet_feature_importance.png")

    report_dict = {
        "model_name": tabnet.model_name,
        "training_dataset": "EMBER 2018",
        "device_used": str(device),
        "ember_test": ember_results,
        "cross_dataset_bodmas": cross_results,
        "elapsed_seconds": time.time() - start_time,
    }

    json_report_path = reports_dir / "ember_tabnet_results.json"
    txt_report_path = reports_dir / "ember_tabnet_results.txt"

    with open(json_report_path, "w") as f:
        json.dump(report_dict, f, indent=2)

    with open(txt_report_path, "w") as f:
        f.write("========================================================================\n")
        f.write("  HARUSPEX Kaggle Phase 2 — TabNet (EMBER 2018 Primary)\n")
        f.write("========================================================================\n\n")
        f.write(f"EMBER 2018 Test Accuracy:  {ember_results['accuracy']:.4f}\n")
        f.write(f"EMBER 2018 Test F1-Score:  {ember_results['f1']:.4f}\n")
        f.write(f"EMBER 2018 Test AUC-ROC:   {ember_results['auc']:.4f}\n\n")
        f.write(f"BODMAS Cross-Dataset Acc: {cross_results['accuracy']:.4f}\n")
        f.write(f"BODMAS Cross-Dataset F1:  {cross_results['f1']:.4f}\n")
        f.write(f"BODMAS Cross-Dataset AUC: {cross_results['auc']:.4f}\n")

    logger.info("Kaggle TabNet Phase 2 Training Finished Successfully! Artifacts saved to %s", out_base)


if __name__ == "__main__":
    main()
