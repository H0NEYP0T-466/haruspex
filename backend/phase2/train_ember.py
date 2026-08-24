#!/usr/bin/env python3
"""HARUSPEX Phase 2 — Train on EMBER 2018, Generalize on BODMAS.

This is the reverse of Phase 1:
  Phase 1: Train on BODMAS → Generalize on EMBER
  Phase 2: Train on EMBER  → Generalize on BODMAS

Pipeline:
  1. Load EMBER train partition (from pre-vectorized .dat memmaps)
  2. Train/test split (80/20) within EMBER
  3. Train XGBoost + LightGBM
  4. Evaluate on EMBER test split
  5. Cross-dataset generalization: evaluate on BODMAS
  6. Save models as ember_xgboost.json, ember_lightgbm.json
  7. Generate SHAP explanations + correlation reports
  8. Print summary

Prerequisites:
  - Run `python phase2/prepare_ember.py` first to vectorize EMBER JSONL → .dat
  - OR use the --from-jsonl flag to vectorize on-the-fly (slower)

Usage:
    cd backend
    source venv/bin/activate
    python phase2/train_ember.py                    # Default: 200K subsample
    python phase2/train_ember.py --max-samples 0    # Full dataset (needs ~10GB RAM)
    python phase2/train_ember.py --max-samples 100000  # 100K subsample
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Add backend/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from haruspex.config import (
    BACKEND_DIR,
    NUM_FEATURES,
    REPORTS_DIR,
    ensure_output_dirs,
    get_feature_names,
)
from haruspex.correlation.correlator import Correlator
from haruspex.data.splits import SplitResult, random_split
from haruspex.evaluation.metrics import (
    compute_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
)
from haruspex.forensics.ioc_extractor import IOCExtractor
from haruspex.forensics.static_analysis import StaticAnalyzer
from haruspex.models.lightgbm_model import LightGBMDetector
from haruspex.models.trainer import train_model
from haruspex.models.xgboost_model import XGBoostDetector
from haruspex.utils.logging import get_logger
from haruspex.utils.storage import save_model, save_report
from haruspex.xai.shap_explainer import SHAPExplainer

logger = get_logger(__name__)

EMBER_DIR = BACKEND_DIR / "dataset" / "ember2018"
DEFAULT_MAX_SAMPLES = 200_000  # Safe for 16GB RAM


def load_ember_train(
    max_samples: int = DEFAULT_MAX_SAMPLES,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Load EMBER train partition from .dat memmaps.

    Args:
        max_samples: Maximum number of labeled samples to load. 0 = all.
        seed: RNG seed for subsampling.

    Returns:
        (X, y) arrays with labeled samples only.
    """
    x_dat = EMBER_DIR / "X_train.dat"
    y_dat = EMBER_DIR / "y_train.dat"

    if not (x_dat.exists() and y_dat.exists()):
        raise FileNotFoundError(
            f"X_train.dat / y_train.dat not found in {EMBER_DIR}.\n"
            "Run `python phase2/prepare_ember.py` first to vectorize EMBER train data."
        )

    logger.info("Loading EMBER train partition from %s", x_dat)
    y_raw = np.memmap(str(y_dat), dtype=np.float32, mode="r")

    # Filter out unlabeled samples (label == -1)
    valid_idx = np.where((y_raw != -1) & (~np.isnan(y_raw)))[0]
    logger.info("EMBER train: %d total rows, %d labeled", y_raw.size, valid_idx.size)

    if valid_idx.size == 0:
        raise ValueError("No labeled samples found in EMBER train partition.")

    # Subsample if requested
    if max_samples > 0 and valid_idx.size > max_samples:
        rng = np.random.RandomState(seed)
        valid_idx = rng.choice(valid_idx, size=max_samples, replace=False)
        valid_idx.sort()
        logger.info("Subsampled to %d samples", max_samples)

    X_raw = np.memmap(str(x_dat), dtype=np.float32, mode="r").reshape(-1, NUM_FEATURES)
    X = np.asarray(X_raw[valid_idx], dtype=np.float32)
    y = np.asarray(y_raw[valid_idx], dtype=np.int32)

    n_benign = int((y == 0).sum())
    n_malware = int((y == 1).sum())
    logger.info("Loaded: %d samples (%d benign, %d malware)", X.shape[0], n_benign, n_malware)

    return X, y


def load_ember_test() -> tuple[np.ndarray, np.ndarray]:
    """Load EMBER test partition from .dat memmaps."""
    x_dat = EMBER_DIR / "X_test.dat"
    y_dat = EMBER_DIR / "y_test.dat"

    if not (x_dat.exists() and y_dat.exists()):
        raise FileNotFoundError(f"X_test.dat / y_test.dat not found in {EMBER_DIR}")

    y_raw = np.memmap(str(y_dat), dtype=np.float32, mode="r")
    valid_idx = np.where((y_raw != -1) & (~np.isnan(y_raw)))[0]

    X_raw = np.memmap(str(x_dat), dtype=np.float32, mode="r").reshape(-1, NUM_FEATURES)
    X = np.asarray(X_raw[valid_idx], dtype=np.float32)
    y = np.asarray(y_raw[valid_idx], dtype=np.int32)

    logger.info("EMBER test: %d labeled samples (%d benign, %d malware)",
                X.shape[0], int((y == 0).sum()), int((y == 1).sum()))
    return X, y


def main() -> None:
    parser = argparse.ArgumentParser(description="HARUSPEX Phase 2: Train on EMBER")
    parser.add_argument(
        "--max-samples", type=int, default=DEFAULT_MAX_SAMPLES,
        help=f"Max training samples (0 = all). Default: {DEFAULT_MAX_SAMPLES}",
    )
    parser.add_argument(
        "--skip-generalization", action="store_true",
        help="Skip cross-dataset BODMAS evaluation",
    )
    args = parser.parse_args()

    start_time = time.time()
    ensure_output_dirs()

    print("=" * 72)
    print("  HARUSPEX Phase 2 — Train on EMBER 2018")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Step 1: Load EMBER train data
    # ------------------------------------------------------------------
    print(f"\n[1/8] Loading EMBER train partition (max_samples={args.max_samples or 'ALL'})...")
    X_train_full, y_train_full = load_ember_train(max_samples=args.max_samples)
    print(f"  Loaded: {X_train_full.shape[0]} samples, {X_train_full.shape[1]} features")
    print(f"  Benign: {int((y_train_full == 0).sum())} | Malware: {int((y_train_full == 1).sum())}")

    # Create metadata DataFrame for splits
    import pandas as pd

    metadata = pd.DataFrame({
        "sha": [f"ember_train_{i}" for i in range(len(y_train_full))],
        "family": "",
        "category": np.where(y_train_full == 1, "malware", "benign"),
        "timestamp": pd.Timestamp("2018-01-01", tz="UTC"),  # EMBER doesn't have timestamps
    })

    feature_names = get_feature_names()

    # ------------------------------------------------------------------
    # Step 2: Train models
    # ------------------------------------------------------------------
    print("\n[2/8] Training models on EMBER data...")
    split = random_split(X_train_full, y_train_full, metadata, test_size=0.2)

    # Train XGBoost
    print("  Training XGBoost...")
    xgb_model = XGBoostDetector()
    xgb_metrics = train_model(xgb_model, split, save_name="ember_xgboost")
    print(f"  XGBoost: test_acc={xgb_metrics['test_accuracy']:.4f}, "
          f"test_f1={xgb_metrics['test_f1']:.4f}, "
          f"test_auc={xgb_metrics['test_auc']:.4f}")

    # Train LightGBM
    print("  Training LightGBM...")
    lgb_model = LightGBMDetector()
    lgb_metrics = train_model(lgb_model, split, save_name="ember_lightgbm")
    print(f"  LightGBM: test_acc={lgb_metrics['test_accuracy']:.4f}, "
          f"test_f1={lgb_metrics['test_f1']:.4f}, "
          f"test_auc={lgb_metrics['test_auc']:.4f}")

    # Select best model by test AUC
    if xgb_metrics["test_auc"] >= lgb_metrics["test_auc"]:
        best_model = xgb_model
        best_name = "ember_xgboost"
        best_metrics = xgb_metrics
    else:
        best_model = lgb_model
        best_name = "ember_lightgbm"
        best_metrics = lgb_metrics

    print(f"\n  Best model: {best_name} (AUC={best_metrics['test_auc']:.4f})")

    # ------------------------------------------------------------------
    # Step 3: Evaluate on EMBER test partition
    # ------------------------------------------------------------------
    print("\n[3/8] Evaluating on EMBER official test partition...")
    try:
        X_test_ember, y_test_ember = load_ember_test()

        for model, name in [(xgb_model, "ember_xgboost"), (lgb_model, "ember_lightgbm")]:
            preds = model.predict(X_test_ember)
            proba = model.predict_proba(X_test_ember)

            from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

            acc = accuracy_score(y_test_ember, preds)
            f1 = f1_score(y_test_ember, preds)
            try:
                auc = roc_auc_score(y_test_ember, proba)
            except ValueError:
                auc = float("nan")

            print(f"  {name} on EMBER test: acc={acc:.4f}, f1={f1:.4f}, auc={auc:.4f}")

            # Save plots
            fig = plot_confusion_matrix(y_test_ember, preds, title=f"{name} — EMBER Test")
            from haruspex.utils.storage import save_plot
            save_plot(fig, f"{name}_ember_test_confusion_matrix")

            fig = plot_roc_curve(y_test_ember, proba, title=f"{name} — EMBER Test ROC")
            save_plot(fig, f"{name}_ember_test_roc_curve")

    except FileNotFoundError as e:
        print(f"  WARNING: Could not load EMBER test partition: {e}")

    # ------------------------------------------------------------------
    # Step 4: Cross-dataset generalization on BODMAS
    # ------------------------------------------------------------------
    if not args.skip_generalization:
        print("\n[4/8] Cross-dataset generalization: evaluating on BODMAS...")
        try:
            from haruspex.data.bodmas import load_bodmas

            bodmas = load_bodmas()
            print(f"  BODMAS: {bodmas.n_samples} samples ({bodmas.n_benign} benign, {bodmas.n_malware} malware)")

            for model, name in [(xgb_model, "ember_xgboost"), (lgb_model, "ember_lightgbm")]:
                preds = model.predict(bodmas.X)
                proba = model.predict_proba(bodmas.X)

                acc = accuracy_score(bodmas.y, preds)
                f1 = f1_score(bodmas.y, preds)
                try:
                    auc = roc_auc_score(bodmas.y, proba)
                except ValueError:
                    auc = float("nan")

                print(f"  {name} on BODMAS: acc={acc:.4f}, f1={f1:.4f}, auc={auc:.4f}")

                fig = plot_confusion_matrix(bodmas.y, preds, title=f"{name} → BODMAS Generalization")
                save_plot(fig, f"{name}_bodmas_generalization_confusion_matrix")

                fig = plot_roc_curve(bodmas.y, proba, title=f"{name} → BODMAS Generalization ROC")
                save_plot(fig, f"{name}_bodmas_generalization_roc_curve")

        except Exception as e:
            print(f"  WARNING: BODMAS evaluation failed: {e}")
    else:
        print("\n[4/8] Skipping cross-dataset generalization (--skip-generalization)")

    # ------------------------------------------------------------------
    # Step 5: SHAP explanations for best model
    # ------------------------------------------------------------------
    print(f"\n[5/8] Generating SHAP explanations for {best_name}...")
    shap_explainer = SHAPExplainer(best_model, feature_names=feature_names)

    shap_values = shap_explainer.explain_dataset(split.X_test, max_samples=100)
    shap_explainer.plot_summary(shap_values, save_path="phase2_shap_summary")

    global_importance = shap_explainer.get_global_importance(shap_values, top_k=10)
    print("  Top-10 SHAP features:")
    for feat in global_importance:
        print(f"    {feat['feature_name']}: {feat['mean_abs_shap']:.4f}")

    # ------------------------------------------------------------------
    # Step 6: Correlation engine on 5 test samples
    # ------------------------------------------------------------------
    print("\n[6/8] Running correlation engine on 5 EMBER test samples...")
    correlator = Correlator()
    static_analyzer = StaticAnalyzer()
    ioc_extractor = IOCExtractor()

    rng = np.random.RandomState(42)
    n_test = split.X_test.shape[0]
    sample_indices = rng.choice(n_test, min(5, n_test), replace=False)

    investigation_reports = []

    for i, idx in enumerate(sample_indices):
        x_sample = split.X_test[idx]

        # SHAP explanation
        shap_exp = shap_explainer.explain_sample(x_sample)

        # Forensic analysis from feature vector
        forensic_report = static_analyzer.analyze_feature_vector(x_sample, feature_names)
        iocs = ioc_extractor.extract_from_analysis(forensic_report)

        # Correlate
        evidence = correlator.correlate(shap_exp, forensic_report, iocs, top_k=20)

        # Generate report
        prediction = "malware" if best_model.predict(x_sample.reshape(1, -1))[0] == 1 else "benign"
        confidence = float(best_model.predict_proba(x_sample.reshape(1, -1))[0])

        report = correlator.generate_report(
            sample_id=f"ember_test_sample_{i}",
            prediction=prediction,
            confidence=confidence,
            model_name=best_name,
            evidence_items=evidence,
            shap_explanation=shap_exp,
            forensic_report=forensic_report,
            iocs=iocs,
        )

        investigation_reports.append(report)
        save_report(report.to_dict(), f"phase2_investigation_sample_{i}")

        n_correlated = sum(1 for e in evidence if e.source == "correlated")
        label = "malware" if split.y_test[idx] == 1 else "benign"
        print(f"  Sample {i+1}: {label} → pred={prediction} ({confidence:.1%}), "
              f"{len(evidence)} evidence ({n_correlated} correlated)")

    # ------------------------------------------------------------------
    # Step 7: Feature importance plots
    # ------------------------------------------------------------------
    print("\n[7/8] Saving feature importance plots...")
    for model, name in [(xgb_model, "ember_xgboost"), (lgb_model, "ember_lightgbm")]:
        fig = plot_feature_importance(model, feature_names, top_k=20, title=f"{name} Feature Importance")
        save_plot(fig, f"{name}_feature_importance")

    # ------------------------------------------------------------------
    # Step 8: Save summary
    # ------------------------------------------------------------------
    print("\n[8/8] Saving training summary...")
    summary = {
        "phase": "phase2_ember",
        "training_dataset": "EMBER_2018",
        "generalization_dataset": "BODMAS",
        "max_samples": args.max_samples,
        "actual_samples": int(X_train_full.shape[0]),
        "models": {
            "ember_xgboost": xgb_metrics,
            "ember_lightgbm": lgb_metrics,
            "best_model": best_name,
        },
        "shap_global_importance": global_importance,
        "investigation_reports": len(investigation_reports),
        "total_correlated_evidence": sum(
            r.correlated_evidence for r in investigation_reports
        ),
    }
    save_report(summary, "phase2_training_summary")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    elapsed = time.time() - start_time
    print("\n" + "=" * 72)
    print("  HARUSPEX Phase 2 — Training Complete!")
    print("=" * 72)
    print(f"\n  Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"\n  Models saved:")
    print(f"    outputs/models/ember_xgboost.json")
    print(f"    outputs/models/ember_lightgbm.json")
    print(f"\n  EMBER train/test performance:")
    print(f"    XGBoost:  acc={xgb_metrics['test_accuracy']:.4f}, f1={xgb_metrics['test_f1']:.4f}, auc={xgb_metrics['test_auc']:.4f}")
    print(f"    LightGBM: acc={lgb_metrics['test_accuracy']:.4f}, f1={lgb_metrics['test_f1']:.4f}, auc={lgb_metrics['test_auc']:.4f}")
    print(f"\n  Top-10 SHAP features ({best_name}):")
    for feat in global_importance:
        print(f"    {feat['feature_name']:30s}: {feat['mean_abs_shap']:.4f}")
    print(f"\n  Investigation reports: {len(investigation_reports)}")
    total_correlated = sum(r.correlated_evidence for r in investigation_reports)
    total_evidence = sum(r.total_evidence for r in investigation_reports)
    print(f"  Total evidence items: {total_evidence}")
    print(f"  Correlated evidence:  {total_correlated}")
    print(f"\n  Outputs saved to: outputs/")
    print("=" * 72)


if __name__ == "__main__":
    main()
