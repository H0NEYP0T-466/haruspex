"""Explanation endpoint — SHAP/LIME for individual dataset samples.

POST /api/explain       — explain a specific sample by dataset index
POST /api/explain/random — explain a random sample from the active dataset
GET  /api/dataset/info  — dataset summary (accepts ?dataset=bodmas|ember)
GET  /api/dataset/samples — list samples with filtering (accepts ?dataset=bodmas|ember)
"""

from __future__ import annotations

import random as stdlib_random
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from haruspex.config import get_feature_names
from haruspex.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Static dataset metadata — these are fixed properties of the datasets,
# not values that need to be recomputed on every request.
# ---------------------------------------------------------------------------

# BODMAS: loaded on demand (has rich metadata: families, categories, etc.).
_bodmas_dataset = None

# EMBER 2018 — full dataset (train + test). Numbers are static; loaded
# lazily only when a sample actually needs to be explained or listed.
EMBER_TOTAL_SAMPLES = 1_000_000    # 800K train + 200K test
EMBER_N_FEATURES = 2381
EMBER_N_BENIGN = 500_000           # approximate 50/50 split
EMBER_N_MALWARE = 500_000


def _load_bodmas():
    """Lazy-load BODMAS (cached for the process lifetime)."""
    global _bodmas_dataset
    if _bodmas_dataset is None:
        from haruspex.data.bodmas import load_bodmas
        _bodmas_dataset = load_bodmas()
    return _bodmas_dataset


def _load_ember():
    """Lazy-load EMBER full dataset (train + test). Cached."""
    from haruspex.data.ember import load_ember
    # Load both partitions and concatenate for the full dataset size.
    X_test, y_test, meta_test = load_ember(test_only=True)
    # Train partition
    try:
        X_train, y_train, meta_train = load_ember(test_only=False)
        X = np.vstack([X_train, X_test])
        y = np.concatenate([y_train, y_test])
        meta = pd.concat([meta_train, meta_test], ignore_index=True)
    except Exception:
        # If train partition is unavailable, fall back to test only.
        X, y, meta = X_test, y_test, meta_test
    return X, y, meta


# Cache for EMBER after first load
_ember_data: tuple | None = None


def _get_ember_data():
    global _ember_data
    if _ember_data is None:
        _ember_data = _load_ember()
    return _ember_data


# ---------------------------------------------------------------------------
# Helpers shared by explain and random explain
# ---------------------------------------------------------------------------

def _run_explanation(model, used_model: str, used_group: str, x_sample, feature_names, num_features: int):
    """Run the full explanation pipeline on a single sample vector."""
    from haruspex.xai.shap_explainer import SHAPExplainer
    from haruspex.forensics.static_analysis import StaticAnalyzer
    from haruspex.forensics.ioc_extractor import IOCExtractor
    from haruspex.correlation.correlator import Correlator

    # SHAP explanation
    shap_explainer = SHAPExplainer(model, feature_names=feature_names)
    shap_exp = shap_explainer.explain_sample(x_sample)

    # Forensic analysis
    analyzer = StaticAnalyzer()
    forensic_report = analyzer.analyze_feature_vector(x_sample, feature_names)

    ioc_extractor = IOCExtractor()
    iocs = ioc_extractor.extract_from_analysis(forensic_report)

    # Correlation
    correlator = Correlator()
    evidence = correlator.correlate(shap_exp, forensic_report, iocs, top_k=num_features)

    # Prediction
    pred_label = int(model.predict(x_sample.reshape(1, -1))[0])
    pred_proba = float(model.predict_proba(x_sample.reshape(1, -1))[0])
    prediction = "malware" if pred_label == 1 else "benign"
    confidence = pred_proba if pred_label == 1 else (1.0 - pred_proba)

    report = correlator.generate_report(
        sample_id="dataset_sample",
        prediction=prediction,
        confidence=confidence,
        model_name=f"{used_group}/{used_model}",
        evidence_items=evidence,
        shap_explanation=shap_exp,
        forensic_report=forensic_report,
        iocs=iocs,
    )

    return report, prediction, confidence


# ---------------------------------------------------------------------------
# Explain a specific sample
# ---------------------------------------------------------------------------

class ExplainRequest(BaseModel):
    """Request body for sample explanation."""

    sample_index: int
    model_name: Optional[str] = None
    model_group: Optional[str] = None
    num_features: int = 20


@router.post("/explain")
async def explain_sample(request: ExplainRequest):
    """Explain a specific dataset sample by index.

    The dataset is selected from request.model_group:
      - "bodmas" -> BODMAS dataset
      - "ember"  -> EMBER dataset
    """
    from haruspex.api.main import get_model

    try:
        model, used_model, used_group = get_model(request.model_name, request.model_group)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Load the dataset corresponding to the model group
    if used_group == "ember":
        X, y, meta = _get_ember_data()
        n_samples = X.shape[0]
        if request.sample_index < 0 or request.sample_index >= n_samples:
            raise HTTPException(
                status_code=400,
                detail=f"Sample index must be 0-{n_samples - 1}",
            )
        idx = request.sample_index
        x_sample = X[idx]
        true_label = int(y[idx])
        sample_meta = meta.iloc[idx]
    else:
        dataset = _load_bodmas()
        if request.sample_index < 0 or request.sample_index >= dataset.n_samples:
            raise HTTPException(
                status_code=400,
                detail=f"Sample index must be 0-{dataset.n_samples - 1}",
            )
        idx = request.sample_index
        x_sample = dataset.X[idx]
        true_label = int(dataset.y[idx])
        sample_meta = dataset.metadata.iloc[idx]

    feature_names = get_feature_names()
    report, prediction, confidence = _run_explanation(
        model, used_model, used_group, x_sample, feature_names, request.num_features,
    )

    # Enrich report with sample metadata
    report_dict = report.to_dict()
    report_dict["sample"] = {
        **report_dict.get("sample", {}),
        "sha256": str(sample_meta.get("sha", "")),
        "family": str(sample_meta.get("family", "")),
        "category": str(sample_meta.get("category", "")),
    }

    return {
        **report_dict,
        "true_label": "malware" if true_label == 1 else "benign",
        "sample_index": idx,
    }


# ---------------------------------------------------------------------------
# Explain a random sample
# ---------------------------------------------------------------------------

class RandomExplainRequest(BaseModel):
    """Request body for random sample explanation."""

    model_name: Optional[str] = None
    model_group: Optional[str] = None
    label_filter: Optional[str] = None  # "malware", "benign", or None for any
    num_features: int = 20


@router.post("/explain/random")
async def explain_random_sample(request: RandomExplainRequest):
    """Explain a random sample from the dataset.

    The dataset is selected from request.model_group:
      - "bodmas" -> BODMAS dataset
      - "ember"  -> EMBER dataset

    Optionally filter by label ("malware" or "benign").
    """
    from haruspex.api.main import get_model

    try:
        model, used_model, used_group = get_model(request.model_name, request.model_group)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if used_group == "ember":
        X, y, meta = _get_ember_data()
        if request.label_filter == "malware":
            valid_indices = np.where(y == 1)[0]
        elif request.label_filter == "benign":
            valid_indices = np.where(y == 0)[0]
        else:
            valid_indices = np.arange(X.shape[0])
    else:
        dataset = _load_bodmas()
        if request.label_filter == "malware":
            valid_indices = np.where(dataset.y == 1)[0]
        elif request.label_filter == "benign":
            valid_indices = np.where(dataset.y == 0)[0]
        else:
            valid_indices = np.arange(dataset.n_samples)

    if len(valid_indices) == 0:
        raise HTTPException(status_code=404, detail="No samples match the filter.")

    idx = int(stdlib_random.choice(valid_indices))

    if used_group == "ember":
        X, y, meta = _get_ember_data()
        x_sample = X[idx]
        true_label = int(y[idx])
        sample_meta = meta.iloc[idx]
    else:
        dataset = _load_bodmas()
        x_sample = dataset.X[idx]
        true_label = int(dataset.y[idx])
        sample_meta = dataset.metadata.iloc[idx]

    feature_names = get_feature_names()
    report, prediction, confidence = _run_explanation(
        model, used_model, used_group, x_sample, feature_names, request.num_features,
    )

    report_dict = report.to_dict()
    report_dict["sample"] = {
        **report_dict.get("sample", {}),
        "sha256": str(sample_meta.get("sha", "")),
        "family": str(sample_meta.get("family", "")),
        "category": str(sample_meta.get("category", "")),
    }

    return {
        **report_dict,
        "true_label": "malware" if true_label == 1 else "benign",
        "sample_index": idx,
    }


# ---------------------------------------------------------------------------
# Dataset info
# ---------------------------------------------------------------------------


@router.get("/dataset/info")
async def dataset_info(dataset: str = "bodmas"):
    """Return dataset summary information.

    Query params:
      - dataset: "bodmas" (default) or "ember"

    BODMAS returns live stats from the loaded dataset (families, categories).
    EMBER returns static, pre-computed values for the full EMBER 2018
    dataset (train + test combined) — no loading or computation needed.
    """
    from haruspex.data.bodmas import get_family_sample_counts

    if dataset == "ember":
        # Static metadata for the full EMBER 2018 dataset.
        # EMBER has no family or category columns, so those are empty.
        return {
            "total_samples": EMBER_TOTAL_SAMPLES,
            "n_features": EMBER_N_FEATURES,
            "n_benign": EMBER_N_BENIGN,
            "n_malware": EMBER_N_MALWARE,
            "n_families": 0,
            "families": [],
            "top_families": [],
            "categories": {},
        }

    # BODMAS — load and compute live stats
    try:
        dataset_obj = _load_bodmas()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {e}")

    families = dataset_obj.metadata["family"].replace("", None).dropna().unique()
    family_counts = get_family_sample_counts(dataset_obj.metadata)
    categories = dataset_obj.metadata["category"].value_counts().to_dict()

    return {
        "total_samples": dataset_obj.n_samples,
        "n_features": dataset_obj.n_features,
        "n_benign": dataset_obj.n_benign,
        "n_malware": dataset_obj.n_malware,
        "n_families": len(families),
        "families": sorted(families.tolist())[:50],
        "top_families": [
            {"name": name, "count": int(count)}
            for name, count in family_counts.head(10).items()
        ],
        "categories": categories,
    }


# ---------------------------------------------------------------------------
# List samples with pagination/filtering
# ---------------------------------------------------------------------------

@router.get("/dataset/samples")
async def list_samples(
    family: Optional[str] = None,
    category: Optional[str] = None,
    label: Optional[str] = None,
    offset: int = 0,
    limit: int = 50,
    dataset: str = "bodmas",
):
    """List samples with optional filtering.

    Query params:
      - dataset: "bodmas" (default) or "ember"
      - family, category: ignored for EMBER (no family/category metadata)
      - label: "benign" or "malware"
      - offset, limit: pagination
    """
    if dataset == "ember":
        X, y, meta = _get_ember_data()
        # Build a DataFrame that matches the BODMAS shape for the response
        df = meta.copy()
        df["index"] = range(len(df))
        df["label"] = ["malware" if v == 1 else "benign" for v in y]
        # EMBER has no family/category — provide placeholder columns
        if "family" not in df.columns:
            df["family"] = ""
        if "category" not in df.columns:
            df["category"] = "ember"
        if "timestamp" not in df.columns:
            df["timestamp"] = "1970-01-01"
        # Apply label filter only (family/category filters don't apply to EMBER)
        if label:
            df = df[df["label"] == label]
        total = len(df)
        df = df.iloc[offset:offset + limit]
        samples = []
        for _, row in df.iterrows():
            samples.append({
                "index": int(row["index"]),
                "sha256": str(row.get("sha", "")),
                "family": str(row.get("family", "")),
                "category": str(row.get("category", "")),
                "label": row["label"],
                "timestamp": str(row.get("timestamp", "")),
            })
        return {"total": total, "offset": offset, "limit": limit, "samples": samples}

    # BODMAS
    dataset_obj = _load_bodmas()
    meta = dataset_obj.metadata.copy()
    meta["index"] = range(len(meta))
    if "label" not in meta.columns:
        meta["label"] = ["malware" if y == 1 else "benign" for y in dataset_obj.y]

    # Apply family/category filters (BODMAS only)
    if family:
        meta = meta[meta["family"] == family]
    if category:
        meta = meta[meta["category"] == category]
    if label:
        meta = meta[meta["label"] == label]

    total = len(meta)
    meta = meta.iloc[offset:offset + limit]

    samples = []
    for _, row in meta.iterrows():
        samples.append({
            "index": int(row["index"]),
            "sha256": str(row.get("sha", "")),
            "family": str(row.get("family", "")),
            "category": str(row.get("category", "")),
            "label": row["label"],
            "timestamp": str(row.get("timestamp", "")),
        })

    return {"total": total, "offset": offset, "limit": limit, "samples": samples}