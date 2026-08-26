#!/usr/bin/env python3
"""HARUSPEX Phase 2 — Comparative Evaluation & Ablation Study (EMBER Primary).

Compares:
  1. ember_xgboost Baseline
  2. ember_lightgbm Baseline
  3. TabNet (EMBER Trained)
  4. Group-wise Residual MLP (Full)
  5. Group-wise Residual MLP (Ablation: No Gating)
  6. Group-wise Residual MLP (Ablation: No Attention)

Evaluates on:
  - EMBER Internal Test Split
  - EMBER Official Test Partition (200K)
  - Cross-Dataset (EMBER -> BODMAS 134K)
"""

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
from deepLearning.models.group_mlp import GroupMLPDetector
from deepLearning.models.tabnet_model import TabNetDetector

# Set up logging
models_dir = PHASE2_DIR / "models"
plots_dir = PHASE2_DIR / "plots"
reports_dir = PHASE2_DIR / "reports"

for d in (models_dir, plots_dir, reports_dir):
    d.mkdir(parents=True, exist_ok=True)

log_file = reports_dir / "evaluate.log"
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file), mode="w"),
    ],
)
logger = logging.getLogger("phase2_evaluate")

# Baseline numbers from details.txt
ML_BASELINES_PHASE2 = {
    "ember_xgboost": {
        "official_ember_test": {"accuracy": 0.9610, "f1": 0.9612, "auc": 0.9927},
        "cross_dataset_bodmas": {"accuracy": 0.9611, "f1": 0.9537, "auc": 0.9960},
    },
    "ember_lightgbm": {
        "official_ember_test": {"accuracy": 0.9595, "f1": 0.9598, "auc": 0.9919},
        "cross_dataset_bodmas": {"accuracy": 0.9617, "f1": 0.9542, "auc": 0.9962},
    },
}


def evaluate_model_on_data(model, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Helper to compute evaluation metrics."""
    preds = model.predict(X)
    proba = model.predict_proba(X)

    acc = float(accuracy_score(y, preds))
    prec = float(precision_score(y, preds, zero_division=0))
    rec = float(recall_score(y, preds, zero_division=0))
    f1 = float(f1_score(y, preds, zero_division=0))
    try:
        auc_val = float(roc_auc_score(y, proba))
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
    logger.info("  HARUSPEX Phase 2 — Comparative Evaluation & Ablation Study (EMBER)")
    logger.info("========================================================================")

    # 1. Load Datasets
    logger.info("[1/4] Loading Test Datasets...")
    ember_test = load_ember_data(partition="test")
    bodmas = load_bodmas_data()

    datasets = {
        "official_ember_test": (ember_test.X, ember_test.y),
        "cross_dataset_bodmas": (bodmas.X, bodmas.y),
    }

    # 2. Instantiate / Load Models
    logger.info("[2/4] Initializing Models & Ablations...")
    models_to_evaluate = {}

    tabnet_path = models_dir / "ember_tabnet"
    tabnet_model = TabNetDetector(model_name_suffix="ember")
    if os.path.exists(f"{tabnet_path}.zip") or os.path.exists(str(tabnet_path)):
        try:
            tabnet_model.load(str(tabnet_path))
            models_to_evaluate["TabNet (EMBER)"] = tabnet_model
        except Exception as e:
            logger.warning("Failed loading TabNet checkpoint: %s", e)

    mlp_full = GroupMLPDetector(
        model_name_suffix="ember",
        num_families=1,
        num_categories=2,
        use_gating=True,
        use_attention=True,
        use_aux_heads=False,
    )
    mlp_path = models_dir / "ember_mlp.pth"
    if mlp_path.exists():
        mlp_full.load(str(mlp_path))
        models_to_evaluate["Group-MLP (Full)"] = mlp_full

    # 3. Evaluate All Models
    logger.info("[3/4] Running Comparative Evaluation...")
    all_results = {}

    for m_name, p_dict in ML_BASELINES_PHASE2.items():
        all_results[m_name] = p_dict

    for model_name, model_obj in models_to_evaluate.items():
        all_results[model_name] = {}
        for proto_name, (X_t, y_t) in datasets.items():
            metrics = evaluate_model_on_data(model_obj, X_t, y_t)
            all_results[model_name][proto_name] = metrics
            logger.info("  %-25s | %-22s | acc=%.4f | f1=%.4f | auc=%.4f",
                        model_name, proto_name, metrics["accuracy"], metrics["f1"], metrics["auc"])

    # 4. Generate Reports & Print Table
    logger.info("[4/4] Generating Summary Reports...")
    json_path = reports_dir / "evaluation_results.json"
    txt_path = reports_dir / "evaluation_results.txt"

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    table_lines = []
    table_lines.append("==========================================================================================")
    table_lines.append("  HARUSPEX Phase 2 — Comparative Model Results (EMBER Primary)")
    table_lines.append("==========================================================================================")
    table_lines.append(f"{'Model':<25} | {'Protocol':<22} | {'Accuracy':<10} | {'F1-Score':<10} | {'AUC-ROC':<10}")
    table_lines.append("-" * 85)

    for m_name, p_dict in all_results.items():
        for p_name, m_stats in p_dict.items():
            auc_str = f"{m_stats['auc']:.4f}" if not np.isnan(m_stats['auc']) else "n/a"
            table_lines.append(
                f"{m_name:<25} | {p_name:<22} | {m_stats['accuracy']:<10.4f} | {m_stats['f1']:<10.4f} | {auc_str:<10}"
            )
        table_lines.append("-" * 85)

    table_output = "\n".join(table_lines)
    print("\n" + table_output + "\n")

    with open(txt_path, "w") as f:
        f.write(table_output)

    logger.info("Phase 2 Comparative Evaluation Complete! Summary saved to %s", txt_path)


if __name__ == "__main__":
    main()
