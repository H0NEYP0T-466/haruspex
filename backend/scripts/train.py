#!/usr/bin/env python3
"""HARUSPEX Training Pipeline — Single Entry Point.

Runs the complete training pipeline:
  1. Load BODMAS dataset
  2. Train XGBoost + LightGBM models
  3. Evaluate on all splits (random, family-held-out, temporal)
  4. Generate SHAP explanations for the best model
  5. Generate LIME explanations for 5 test samples
  6. Run correlation engine on those samples
  7. Save all outputs (models, plots, reports)
  8. Print summary

Usage:
    cd backend
    source .venv/bin/activate
    python scripts/train.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

# Add backend/ to path so haruspex package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from haruspex.config import (
    HELD_OUT_FAMILIES,
    REPORTS_DIR,
    TEMPORAL_CUTOFF,
    ensure_output_dirs,
    get_feature_names,
)
from haruspex.correlation.correlator import Correlator
from haruspex.data.bodmas import load_bodmas
from haruspex.data.splits import random_split
from haruspex.evaluation.metrics import (
    plot_confusion_matrix,
    plot_feature_importance,
    plot_roc_curve,
)
from haruspex.evaluation.protocols import run_all_evaluations
from haruspex.forensics.ioc_extractor import IOCExtractor
from haruspex.forensics.static_analysis import StaticAnalyzer
from haruspex.models.lightgbm_model import LightGBMDetector
from haruspex.models.trainer import train_model
from haruspex.models.xgboost_model import XGBoostDetector
from haruspex.utils.storage import save_model, save_report
from haruspex.xai.lime_explainer import LIMEExplainer
from haruspex.xai.shap_explainer import SHAPExplainer


def main() -> None:
    """Run the full HARUSPEX training pipeline."""
    start_time = time.time()
    ensure_output_dirs()

    print("=" * 72)
    print("  HARUSPEX Training Pipeline")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    print("\n[1/8] Loading BODMAS dataset...")
    dataset = load_bodmas()
    print(f"  Loaded: {dataset.n_samples} samples, {dataset.n_features} features")
    print(f"  Benign: {dataset.n_benign} | Malware: {dataset.n_malware}")

    X, y, metadata = dataset.X, dataset.y, dataset.metadata
    feature_names = dataset.feature_names

    # ------------------------------------------------------------------
    # Step 2: Train models
    # ------------------------------------------------------------------
    print("\n[2/8] Training models...")
    split = random_split(X, y, metadata, test_size=0.2)

    # Train XGBoost
    print("  Training XGBoost...")
    xgb_model = XGBoostDetector()
    xgb_metrics = train_model(xgb_model, split, save_name="xgboost")
    print(f"  XGBoost: test_acc={xgb_metrics['test_accuracy']:.4f}, "
          f"test_f1={xgb_metrics['test_f1']:.4f}, "
          f"test_auc={xgb_metrics['test_auc']:.4f}")

    # Train LightGBM
    print("  Training LightGBM...")
    lgb_model = LightGBMDetector()
    lgb_metrics = train_model(lgb_model, split, save_name="lightgbm")
    print(f"  LightGBM: test_acc={lgb_metrics['test_accuracy']:.4f}, "
          f"test_f1={lgb_metrics['test_f1']:.4f}, "
          f"test_auc={lgb_metrics['test_auc']:.4f}")

    # Select best model by validation AUC (preventing test set selection bias)
    if xgb_metrics.get("val_auc", 0.0) >= lgb_metrics.get("val_auc", 0.0):
        best_model = xgb_model
        best_name = "xgboost"
        best_metrics = xgb_metrics
    else:
        best_model = lgb_model
        best_name = "lightgbm"
        best_metrics = lgb_metrics

    print(f"\n  Best model (selected by val_auc={best_metrics.get('val_auc', 0.0):.4f}): {best_name} (test_auc={best_metrics['test_auc']:.4f})")

    # ------------------------------------------------------------------
    # Step 3: Evaluate on all splits
    # ------------------------------------------------------------------
    print("\n[3/8] Running evaluations on all splits...")
    all_results = {}
    for model, name in [(xgb_model, "xgboost"), (lgb_model, "lightgbm")]:
        print(f"  Evaluating {name}...")
        results = run_all_evaluations(model, X, y, metadata)
        all_results[name] = results

    # Save combined results
    save_report(all_results, "evaluation_results")

    # ------------------------------------------------------------------
    # Step 4: SHAP explanations
    # ------------------------------------------------------------------
    print("\n[4/8] Generating SHAP explanations for best model...")
    shap_explainer = SHAPExplainer(best_model, feature_names=feature_names)

    # Dataset-level SHAP values
    shap_values = shap_explainer.explain_dataset(split.X_test, max_samples=100)
    shap_explainer.plot_summary(shap_values, save_path="shap_summary")

    global_importance = shap_explainer.get_global_importance(shap_values, top_k=10)
    print("  Top-10 SHAP features:")
    for feat in global_importance:
        print(f"    {feat['feature_name']}: {feat['mean_abs_shap']:.4f}")

    # ------------------------------------------------------------------
    # Step 5: LIME explanations for 5 samples
    # ------------------------------------------------------------------
    print("\n[5/8] Generating LIME explanations for 5 test samples...")
    # Use a subset of training data for LIME background
    lime_bg_size = min(1000, split.X_train.shape[0])
    rng = np.random.RandomState(42)
    lime_bg_idx = rng.choice(split.X_train.shape[0], lime_bg_size, replace=False)
    lime_explainer = LIMEExplainer(
        split.X_train[lime_bg_idx],
        feature_names=feature_names,
    )

    # Select 5 random test samples (mix of malware and benign)
    n_test = split.X_test.shape[0]
    sample_indices = rng.choice(n_test, min(5, n_test), replace=False)

    shap_explanations = []
    lime_explanations = []

    for i, idx in enumerate(sample_indices):
        x_sample = split.X_test[idx]
        sample_meta = split.metadata_test.iloc[idx]

        # SHAP
        shap_exp = shap_explainer.explain_sample(x_sample)
        shap_explanations.append(shap_exp)

        # LIME
        lime_exp = lime_explainer.explain_sample(best_model, x_sample, num_features=10)
        lime_explainer.plot_explanation(lime_exp, save_path=f"lime_sample_{i}")
        lime_explanations.append(lime_exp)

        label = "malware" if split.y_test[idx] == 1 else "benign"
        print(f"  Sample {i+1}: {label} (family={sample_meta.get('family', 'N/A')}, "
              f"pred={shap_exp.prediction:.3f})")

    # ------------------------------------------------------------------
    # Step 6: Correlation engine
    # ------------------------------------------------------------------
    print("\n[6/8] Running correlation engine on 5 samples...")
    correlator = Correlator()
    static_analyzer = StaticAnalyzer()
    ioc_extractor = IOCExtractor()

    investigation_reports = []

    for i, idx in enumerate(sample_indices):
        x_sample = split.X_test[idx]
        sample_meta = split.metadata_test.iloc[idx]

        # Forensic analysis (from feature vector)
        forensic_report = static_analyzer.analyze_feature_vector(x_sample, feature_names)

        # IOC extraction
        iocs = ioc_extractor.extract_from_analysis(forensic_report)

        # Correlate
        evidence = correlator.correlate(
            shap_explanations[i], forensic_report, iocs, top_k=20
        )

        # Generate report
        prediction = "malware" if best_model.predict(x_sample.reshape(1, -1))[0] == 1 else "benign"
        confidence = float(best_model.predict_proba(x_sample.reshape(1, -1))[0])

        report = correlator.generate_report(
            sample_id=f"test_sample_{i}",
            prediction=prediction,
            confidence=confidence,
            model_name=best_name,
            evidence_items=evidence,
            shap_explanation=shap_explanations[i],
            lime_features=lime_explanations[i].top_features(k=10),
            forensic_report=forensic_report,
            iocs=iocs,
            sample_sha256=str(sample_meta.get("sha", "")),
            sample_family=str(sample_meta.get("family", "")),
            sample_category=str(sample_meta.get("category", "")),
        )

        investigation_reports.append(report)

        # Save individual report
        save_report(report.to_dict(), f"investigation_sample_{i}")

        n_correlated = sum(1 for e in evidence if e.source == "correlated")
        print(f"  Sample {i+1}: {prediction} ({confidence:.1%}), "
              f"{len(evidence)} evidence items ({n_correlated} correlated)")

        # Save text report
        txt_path = REPORTS_DIR / f"investigation_sample_{i}.txt"
        txt_path.write_text(report.to_text())

    # ------------------------------------------------------------------
    # Step 7: Save everything
    # ------------------------------------------------------------------
    print("\n[7/8] Saving all outputs...")
    # Models already saved in Step 2
    # Plots already saved in Steps 3-5
    # Reports already saved in Step 6

    # Save training summary
    summary = {
        "models": {
            "xgboost": xgb_metrics,
            "lightgbm": lgb_metrics,
            "best_model": best_name,
        },
        "dataset": {
            "total_samples": dataset.n_samples,
            "n_features": dataset.n_features,
            "n_benign": dataset.n_benign,
            "n_malware": dataset.n_malware,
        },
        "shap_global_importance": global_importance,
        "investigation_reports": len(investigation_reports),
        "total_correlated_evidence": sum(
            r.correlated_evidence for r in investigation_reports
        ),
    }
    save_report(summary, "training_summary")

    # ------------------------------------------------------------------
    # Step 8: Print summary
    # ------------------------------------------------------------------
    elapsed = time.time() - start_time
    print("\n" + "=" * 72)
    print("  HARUSPEX Training Complete!")
    print("=" * 72)
    print(f"\n  Time elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"\n  Model performance across all splits:")

    for model_name, model_results in all_results.items():
        print(f"\n  {model_name.upper()}:")
        for eval_result in model_results["evaluations"]:
            split_name = eval_result["split"]
            m = eval_result.get("metrics", {})
            if m:
                print(f"    {split_name:20s}: acc={m.get('accuracy', 0):.4f}, "
                      f"f1={m.get('f1', 0):.4f}, "
                      f"auc={m.get('auc_roc', 0):.4f}")
            else:
                print(f"    {split_name:20s}: {eval_result.get('status', 'N/A')}")

    print(f"\n  Top-10 SHAP features ({best_name}):")
    for feat in global_importance:
        print(f"    {feat['feature_name']:30s}: {feat['mean_abs_shap']:.4f}")

    total_correlated = sum(r.correlated_evidence for r in investigation_reports)
    total_evidence = sum(r.total_evidence for r in investigation_reports)
    print(f"\n  Investigation reports: {len(investigation_reports)}")
    print(f"  Total evidence items: {total_evidence}")
    print(f"  Correlated evidence:  {total_correlated}")

    print(f"\n  Outputs saved to: outputs/")
    print("=" * 72)


if __name__ == "__main__":
    main()
