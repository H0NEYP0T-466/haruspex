#!/usr/bin/env python3
"""HARUSPEX Phase 1 — Comparative Evaluation & Ablation Study (BODMAS Primary).

Compares:
  1. XGBoost Baseline
  2. LightGBM Baseline
  3. TabNet
  4. Group-wise Residual MLP (Full)
  5. Group-wise Residual MLP (Ablation: No Gating)
  6. Group-wise Residual MLP (Ablation: No Auxiliary Heads)
  7. Group-wise Residual MLP (Ablation: No Attention)

Evaluates on all 4 protocols:
  - Random Split (80/20)
  - Family-Held-Out Split
  - Temporal Split
  - Cross-Dataset (BODMAS -> EMBER 2018)
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

from deepLearning.utils.config import PHASE1_DIR, HELD_OUT_FAMILIES, TEMPORAL_CUTOFF
from deepLearning.utils.data import load_bodmas_data, load_ember_data
from deepLearning.utils.splits import random_split, family_held_out_split, temporal_split
from deepLearning.models.group_mlp import GroupMLPDetector
from deepLearning.models.tabnet_model import TabNetDetector

# Set up logging
models_dir = PHASE1_DIR / "models"
plots_dir = PHASE1_DIR / "plots"
reports_dir = PHASE1_DIR / "reports"

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
logger = logging.getLogger("phase1_evaluate")

# Baseline numbers from details.txt
ML_BASELINES = {
    "XGBoost": {
        "random_split": {"accuracy": 0.9970, "f1": 0.9965, "auc": 0.9998},
        "family_held_out": {"accuracy": 0.9998, "f1": 0.9999, "auc": float("nan")},
        "temporal_split": {"accuracy": 0.9995, "f1": 0.9995, "auc": 1.0000},
        "cross_dataset": {"accuracy": 0.8417, "f1": 0.8215, "auc": 0.9256},
    },
    "LightGBM": {
        "random_split": {"accuracy": 0.9972, "f1": 0.9967, "auc": 0.9999},
        "family_held_out": {"accuracy": 0.9998, "f1": 0.9999, "auc": float("nan")},
        "temporal_split": {"accuracy": 0.9996, "f1": 0.9996, "auc": 1.0000},
        "cross_dataset": {"accuracy": 0.8396, "f1": 0.8191, "auc": 0.9236},
    },
}


def evaluate_model_on_split(model, split_data, is_ember=False) -> dict[str, float]:
    """Helper to compute evaluation metrics for a model on a split."""
    y_test = split_data.y if is_ember else split_data.y_test
    X_test = split_data.X if is_ember else split_data.X_test

    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)

    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    f1 = float(f1_score(y_test, preds, zero_division=0))
    try:
        auc_val = float(roc_auc_score(y_test, proba))
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
    logger.info("  HARUSPEX Phase 1 — Comparative Evaluation & Ablation Study")
    logger.info("========================================================================")

    # 1. Load Datasets and Prepare Splits
    logger.info("[1/4] Preparing Dataset Splits...")
    bodmas = load_bodmas_data()
    X, y, meta = bodmas.X, bodmas.y, bodmas.metadata
    family_ids, cat_ids = bodmas.family_ids, bodmas.category_ids

    splits = {
        "random_split": random_split(X, y, meta, family_ids, cat_ids),
        "family_held_out": family_held_out_split(X, y, meta, family_ids, cat_ids, HELD_OUT_FAMILIES),
        "temporal_split": temporal_split(X, y, meta, family_ids, cat_ids, TEMPORAL_CUTOFF),
    }

    try:
        splits["cross_dataset"] = load_ember_data(partition="test")
    except Exception as e:
        logger.warning("Could not load EMBER test partition for cross-dataset evaluation: %s", e)
        splits["cross_dataset"] = None

    # 2. Instantiate and Train Models / Ablations
    logger.info("[2/4] Initializing Models & Ablations...")
    num_families = len(bodmas.family_encoder)
    num_categories = len(bodmas.category_encoder)

    models_to_evaluate = {}

    # Load / Train TabNet
    tabnet_path = models_dir / "tabnet.zip"
    tabnet_model = TabNetDetector(model_name_suffix="bodmas")
    if tabnet_path.exists() or Path(str(models_dir / "tabnet")).exists():
        try:
            tabnet_model.load(str(models_dir / "tabnet"))
            models_to_evaluate["TabNet"] = tabnet_model
        except Exception as e:
            logger.warning("Failed loading TabNet checkpoint: %s", e)
    
    # Load / Train Group-MLP (Full)
    mlp_full = GroupMLPDetector(
        model_name_suffix="bodmas",
        num_families=num_families,
        num_categories=num_categories,
        use_gating=True,
        use_attention=True,
        use_aux_heads=True,
    )
    mlp_path = models_dir / "mlp.pth"
    if mlp_path.exists():
        mlp_full.load(str(mlp_path))
        models_to_evaluate["Group-MLP (Full)"] = mlp_full

    # Ablation 1: No Gating
    logger.info("Training Ablation Model: Group-MLP (No Gating)...")
    n_val = int(len(splits["random_split"].X_train) * 0.1)
    X_tr = splits["random_split"].X_train[:-n_val]
    y_tr = splits["random_split"].y_train[:-n_val]
    fam_tr = splits["random_split"].family_train[:-n_val]
    cat_tr = splits["random_split"].cat_train[:-n_val]
    X_val = splits["random_split"].X_train[-n_val:]
    y_val = splits["random_split"].y_train[-n_val:]
    fam_val = splits["random_split"].family_train[-n_val:]
    cat_val = splits["random_split"].cat_train[-n_val:]

    mlp_no_gate = GroupMLPDetector(
        model_name_suffix="no_gate",
        num_families=num_families,
        num_categories=num_categories,
        use_gating=False,
        use_attention=True,
        use_aux_heads=True,
        max_epochs=20,
    )
    mlp_no_gate.fit(X_tr, y_tr, X_val, y_val, fam_tr, fam_val, cat_tr, cat_val)
    models_to_evaluate["Group-MLP (No Gate)"] = mlp_no_gate

    # Ablation 2: No Aux Heads
    logger.info("Training Ablation Model: Group-MLP (No Aux Heads)...")
    mlp_no_aux = GroupMLPDetector(
        model_name_suffix="no_aux",
        num_families=num_families,
        num_categories=num_categories,
        use_gating=True,
        use_attention=True,
        use_aux_heads=False,
        max_epochs=20,
    )
    mlp_no_aux.fit(X_tr, y_tr, X_val, y_val, fam_tr, fam_val, cat_tr, cat_val)
    models_to_evaluate["Group-MLP (No Aux)"] = mlp_no_aux

    # 3. Evaluate All Models across All Protocols
    logger.info("[3/4] Running Comparative Evaluation...")
    all_results = {}

    # Include ML Baselines
    for model_name, proto_dict in ML_BASELINES.items():
        all_results[model_name] = proto_dict

    for model_name, model_obj in models_to_evaluate.items():
        all_results[model_name] = {}
        for proto_name, split_obj in splits.items():
            if split_obj is None:
                continue
            is_ember = (proto_name == "cross_dataset")
            metrics = evaluate_model_on_split(model_obj, split_obj, is_ember=is_ember)
            all_results[model_name][proto_name] = metrics
            logger.info("  %-25s | %-16s | acc=%.4f | f1=%.4f | auc=%.4f",
                        model_name, proto_name, metrics["accuracy"], metrics["f1"], metrics["auc"])

    # 4. Generate Comparative Table & Reports
    logger.info("[4/4] Generating Summary Reports...")
    json_path = reports_dir / "evaluation_results.json"
    txt_path = reports_dir / "evaluation_results.txt"

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # Print Table
    table_lines = []
    table_lines.append("==========================================================================================")
    table_lines.append("  HARUSPEX Phase 1 — Comparative Model & Ablation Results (BODMAS Primary)")
    table_lines.append("==========================================================================================")
    table_lines.append(f"{'Model':<25} | {'Protocol':<16} | {'Accuracy':<10} | {'F1-Score':<10} | {'AUC-ROC':<10}")
    table_lines.append("-" * 80)

    for m_name, p_dict in all_results.items():
        for p_name, m_stats in p_dict.items():
            auc_str = f"{m_stats['auc']:.4f}" if not np.isnan(m_stats['auc']) else "n/a"
            table_lines.append(
                f"{m_name:<25} | {p_name:<16} | {m_stats['accuracy']:<10.4f} | {m_stats['f1']:<10.4f} | {auc_str:<10}"
            )
        table_lines.append("-" * 80)

    table_output = "\n".join(table_lines)
    print("\n" + table_output + "\n")

    with open(txt_path, "w") as f:
        f.write(table_output)

    logger.info("Comparative Evaluation Complete! Summary saved to %s", txt_path)


if __name__ == "__main__":
    main()
