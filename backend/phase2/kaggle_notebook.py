#!/usr/bin/env python3
"""
HARUSPEX Phase 2 — Self-Contained Kaggle Training Notebook
==========================================================

Upload this single file to Kaggle, attach the EMBER 2018 dataset, and run.
Produces: ember_xgboost.json + ember_lightgbm.json as output artifacts.

KAGGLE SETUP:
  1. Create a new Kaggle Notebook (Python)
  2. Add the EMBER 2018 dataset to the notebook
     - The dataset should be at /kaggle/input/ember2018/ or similar
     - Should contain: train_features_0.jsonl .. train_features_5.jsonl, test_features.jsonl
  3. Paste this entire file into a single code cell (or upload as .py)
  4. Hit "Run All"
  5. Download outputs from /kaggle/working/

Estimated runtime: 30-45 min on Kaggle GPU/CPU instance (full 800K samples).

After training:
  - Download ember_xgboost.json and ember_lightgbm.json from Kaggle output
  - Copy them to your local: HARUSPEX/outputs/models/
"""

# =============================================================================
# 0. Install dependencies
# =============================================================================
import subprocess
import sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

# These should be pre-installed on Kaggle, but just in case:
for pkg in ["xgboost", "lightgbm", "scikit-learn", "shap", "lief"]:
    try:
        __import__(pkg)
    except ImportError:
        install(pkg)

# ember package (for feature extraction)
try:
    import ember
except ImportError:
    install("ember")
    # If ember pip install fails, install from source:
    try:
        import ember
    except ImportError:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q",
            "git+https://github.com/elastic/ember.git"
        ])

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

print("=" * 72)
print("  HARUSPEX Phase 2 — Kaggle Training")
print("=" * 72)

# =============================================================================
# 1. Locate EMBER dataset
# =============================================================================
# Try common Kaggle paths
CANDIDATE_DIRS = [
    Path("/kaggle/input/ember2018"),
    Path("/kaggle/input/ember-2018"),
    Path("/kaggle/input/ember"),
    Path("/kaggle/input"),  # fallback: check if JSONL files are directly here
]

EMBER_DIR = None
for d in CANDIDATE_DIRS:
    if d.exists() and list(d.glob("train_features_*.jsonl")):
        EMBER_DIR = d
        break
    # Check subdirectories one level deep
    if d.exists():
        for sub in d.iterdir():
            if sub.is_dir() and list(sub.glob("train_features_*.jsonl")):
                EMBER_DIR = sub
                break
    if EMBER_DIR:
        break

if EMBER_DIR is None:
    # Local fallback for testing
    local_dir = Path("dataset/ember2018")
    if local_dir.exists() and list(local_dir.glob("train_features_*.jsonl")):
        EMBER_DIR = local_dir
    else:
        raise FileNotFoundError(
            "EMBER dataset not found! Add the EMBER 2018 dataset to this Kaggle notebook.\n"
            "Expected train_features_*.jsonl files in the input directory."
        )

OUTPUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./phase2_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\n  EMBER dataset: {EMBER_DIR}")
print(f"  Output dir:    {OUTPUT_DIR}")

train_jsonls = sorted(EMBER_DIR.glob("train_features_*.jsonl"))
test_jsonl = EMBER_DIR / "test_features.jsonl"
print(f"  Train files:   {len(train_jsonls)}")
print(f"  Test file:     {'yes' if test_jsonl.exists() else 'no'}")

# =============================================================================
# 2. Feature extraction constants (EMBER 2381 features)
# =============================================================================
NUM_FEATURES = 2381

FEATURE_GROUPS = [
    ("byte_hist",           0,   256),
    ("byte_entropy_hist", 256,   256),
    ("string_info",       512,   104),
    ("general_info",      616,    10),
    ("header_info",       626,    62),
    ("section_info",      688,   255),
    ("import_info",       943,  1280),
    ("export_info",      2223,   128),
    ("data_dir_info",    2351,    30),
]

def get_feature_names():
    names = []
    for group_name, start, count in FEATURE_GROUPS:
        for i in range(count):
            names.append(f"{group_name}_{i}")
    return names


# =============================================================================
# 3. Patch ember SectionInfo bug (scikit-learn >=1.2 compatibility)
# =============================================================================
def patch_ember():
    """Fix ember's SectionInfo.process_raw_features for modern scikit-learn."""
    try:
        import ember.features as ef
    except ImportError:
        return None

    if getattr(ef.SectionInfo, "_patched", False):
        return ef.PEFeatureExtractor(feature_version=2, print_feature_warning=False)

    def process_raw_features(self, raw_obj):
        sections = raw_obj["sections"]
        general = [
            len(sections),
            sum(1 for s in sections if s["size"] == 0),
            sum(1 for s in sections if s["name"] == ""),
            sum(1 for s in sections if "MEM_READ" in s["props"] and "MEM_EXECUTE" in s["props"]),
            sum(1 for s in sections if "MEM_WRITE" in s["props"]),
        ]
        section_sizes = [(s["name"], s["size"]) for s in sections]
        section_sizes_hashed = FeatureHasher(50, input_type="pair").transform([section_sizes]).toarray()[0]
        section_entropy = [(s["name"], s["entropy"]) for s in sections]
        section_entropy_hashed = FeatureHasher(50, input_type="pair").transform([section_entropy]).toarray()[0]
        section_vsize = [(s["name"], s["vsize"]) for s in sections]
        section_vsize_hashed = FeatureHasher(50, input_type="pair").transform([section_vsize]).toarray()[0]
        entry_name_hashed = FeatureHasher(50, input_type="string").transform([[raw_obj["entry"]]]).toarray()[0]
        characteristics = [p for s in sections for p in s["props"] if s["name"] == raw_obj["entry"]]
        characteristics_hashed = FeatureHasher(50, input_type="string").transform([characteristics]).toarray()[0]
        return np.hstack([
            general, section_sizes_hashed, section_entropy_hashed,
            section_vsize_hashed, entry_name_hashed, characteristics_hashed,
        ]).astype(np.float32)

    ef.SectionInfo.process_raw_features = process_raw_features
    ef.SectionInfo._patched = True
    print("  Applied ember/scikit-learn compatibility patch")
    return ef.PEFeatureExtractor(feature_version=2, print_feature_warning=False)

extractor = patch_ember()

# =============================================================================
# 4. Vectorize EMBER train + test data
# =============================================================================
print("\n[1/6] Vectorizing EMBER dataset...")

def vectorize_jsonl_files(jsonl_paths, max_samples=0, label="train"):
    """Vectorize EMBER JSONL files into numpy arrays."""
    X_list = []
    y_list = []
    n_errors = 0
    total_read = 0

    for jsonl_path in jsonl_paths:
        print(f"  Reading {jsonl_path.name}...")
        with open(jsonl_path) as f:
            for line in f:
                if max_samples > 0 and len(y_list) >= max_samples:
                    break
                total_read += 1
                entry = json.loads(line)
                label_val = entry.get("label", -1)
                if label_val == -1:
                    continue  # Skip unlabeled
                try:
                    vec = extractor.process_raw_features(entry)
                    X_list.append(vec)
                    y_list.append(label_val)
                except Exception as e:
                    n_errors += 1
                    if n_errors <= 3:
                        print(f"    WARNING: Vectorization error: {e}")

                if len(y_list) % 50000 == 0 and len(y_list) > 0:
                    print(f"    ... {len(y_list)} samples vectorized")

        if max_samples > 0 and len(y_list) >= max_samples:
            break

    print(f"  {label}: {len(y_list)} labeled samples from {total_read} total rows ({n_errors} errors)")
    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)


t0 = time.time()

# Vectorize train
print("\n  Vectorizing TRAIN partition...")
X_train, y_train = vectorize_jsonl_files(train_jsonls, max_samples=0, label="train")

# Vectorize test
print("\n  Vectorizing TEST partition...")
X_test, y_test = vectorize_jsonl_files([test_jsonl], max_samples=0, label="test")

vec_time = time.time() - t0
print(f"\n  Vectorization done in {vec_time:.1f}s ({vec_time/60:.1f} min)")
print(f"  Train: {X_train.shape} ({int((y_train==0).sum())} benign, {int((y_train==1).sum())} malware)")
print(f"  Test:  {X_test.shape} ({int((y_test==0).sum())} benign, {int((y_test==1).sum())} malware)")

# =============================================================================
# 5. Train models
# =============================================================================
print("\n[2/6] Training XGBoost...")

import xgboost as xgb

t0 = time.time()
xgb_clf = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    tree_method="hist",
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    random_state=42,
    verbosity=0,
    n_jobs=-1,
)

# Internal validation split
from sklearn.model_selection import train_test_split

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
)

xgb_clf.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    verbose=False,
)
xgb_time = time.time() - t0
print(f"  XGBoost trained in {xgb_time:.1f}s")

# XGBoost evaluation
xgb_preds = xgb_clf.predict(X_test)
xgb_proba = xgb_clf.predict_proba(X_test)[:, 1]
xgb_acc = accuracy_score(y_test, xgb_preds)
xgb_f1 = f1_score(y_test, xgb_preds)
xgb_auc = roc_auc_score(y_test, xgb_proba)
print(f"  XGBoost on EMBER test: acc={xgb_acc:.4f}, f1={xgb_f1:.4f}, auc={xgb_auc:.4f}")

print("\n[3/6] Training LightGBM...")

import lightgbm as lgbm

t0 = time.time()
lgb_clf = lgbm.LGBMClassifier(
    objective="binary",
    metric="binary_logloss",
    n_estimators=200,
    max_depth=8,
    learning_rate=0.1,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    random_state=42,
    verbose=-1,
    n_jobs=-1,
)

lgb_clf.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
)
lgb_time = time.time() - t0
print(f"  LightGBM trained in {lgb_time:.1f}s")

# LightGBM evaluation
lgb_preds = lgb_clf.predict(X_test)
lgb_proba = lgb_clf.predict_proba(X_test)[:, 1]
lgb_acc = accuracy_score(y_test, lgb_preds)
lgb_f1 = f1_score(y_test, lgb_preds)
lgb_auc = roc_auc_score(y_test, lgb_proba)
print(f"  LightGBM on EMBER test: acc={lgb_acc:.4f}, f1={lgb_f1:.4f}, auc={lgb_auc:.4f}")

# =============================================================================
# 6. Save models in HARUSPEX-compatible format
# =============================================================================
print("\n[4/6] Saving models...")

# XGBoost — save as JSON (HARUSPEX uses xgb.Booster.load_model)
xgb_path = OUTPUT_DIR / "ember_xgboost.json"
xgb_clf.save_model(str(xgb_path))
print(f"  Saved: {xgb_path} ({xgb_path.stat().st_size / 1e6:.1f} MB)")

# LightGBM — save as JSON (HARUSPEX uses lgb.Booster)
lgb_path = OUTPUT_DIR / "ember_lightgbm.json"
lgb_clf.booster_.save_model(str(lgb_path))
print(f"  Saved: {lgb_path} ({lgb_path.stat().st_size / 1e6:.1f} MB)")

# =============================================================================
# 7. SHAP explanations
# =============================================================================
print("\n[5/6] Generating SHAP explanations...")

import shap

feature_names = get_feature_names()

# Use best model for SHAP
best_name = "ember_xgboost" if xgb_auc >= lgb_auc else "ember_lightgbm"
best_clf = xgb_clf if xgb_auc >= lgb_auc else lgb_clf
print(f"  Best model: {best_name}")

# TreeSHAP on 100 test samples
shap_sample = X_test[:100]
tree_explainer = shap.TreeExplainer(best_clf)
shap_values = tree_explainer.shap_values(shap_sample)

# Handle different SHAP output formats
if isinstance(shap_values, list):
    sv = shap_values[1]  # Class 1 (malware) SHAP values
else:
    sv = shap_values

mean_abs_shap = np.abs(sv).mean(axis=0)
top_indices = np.argsort(mean_abs_shap)[::-1][:10]

print("  Top-10 SHAP features:")
shap_importance = []
for idx in top_indices:
    feat_name = feature_names[idx] if idx < len(feature_names) else f"feature_{idx}"
    importance = float(mean_abs_shap[idx])
    shap_importance.append({"feature_name": feat_name, "mean_abs_shap": importance})
    print(f"    {feat_name:30s}: {importance:.4f}")

# =============================================================================
# 8. Save training report
# =============================================================================
print("\n[6/6] Saving training report...")

report = {
    "phase": "phase2_ember_kaggle",
    "training_dataset": "EMBER_2018",
    "train_samples": int(X_train.shape[0]),
    "test_samples": int(X_test.shape[0]),
    "results": {
        "ember_xgboost": {
            "accuracy": float(xgb_acc),
            "f1": float(xgb_f1),
            "auc": float(xgb_auc),
            "train_time_seconds": float(xgb_time),
        },
        "ember_lightgbm": {
            "accuracy": float(lgb_acc),
            "f1": float(lgb_f1),
            "auc": float(lgb_auc),
            "train_time_seconds": float(lgb_time),
        },
        "best_model": best_name,
    },
    "shap_top_features": shap_importance,
    "classification_report_xgboost": classification_report(y_test, xgb_preds, output_dict=True),
    "classification_report_lightgbm": classification_report(y_test, lgb_preds, output_dict=True),
}

report_path = OUTPUT_DIR / "phase2_training_report.json"
with open(report_path, "w") as f:
    json.dump(report, f, indent=2, default=str)
print(f"  Saved: {report_path}")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 72)
print("  HARUSPEX Phase 2 — Kaggle Training Complete!")
print("=" * 72)
print(f"""
  Models saved to: {OUTPUT_DIR}
    - ember_xgboost.json  ({xgb_path.stat().st_size / 1e6:.1f} MB)
    - ember_lightgbm.json ({lgb_path.stat().st_size / 1e6:.1f} MB)

  EMBER Test Results:
    XGBoost:  acc={xgb_acc:.4f}, f1={xgb_f1:.4f}, auc={xgb_auc:.4f}
    LightGBM: acc={lgb_acc:.4f}, f1={lgb_f1:.4f}, auc={lgb_auc:.4f}
    Best:     {best_name}

  NEXT STEPS:
    1. Download ember_xgboost.json and ember_lightgbm.json from Kaggle Output
    2. Copy them to: HARUSPEX/outputs/models/
    3. Restart the HARUSPEX backend: python -m haruspex.api.main
    4. The frontend model selector will now show both BODMAS and EMBER models
""")
print("=" * 72)
