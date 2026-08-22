"""Explanation endpoint — SHAP/LIME for individual BODMAS samples.

POST /api/explain — explain a specific sample by BODMAS dataset index.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from haruspex.config import get_feature_names
from haruspex.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Cache the loaded dataset
_dataset_cache: dict = {}


def _get_dataset():
    """Lazy-load the BODMAS dataset."""
    if "dataset" not in _dataset_cache:
        from haruspex.data.bodmas import load_bodmas

        _dataset_cache["dataset"] = load_bodmas()
    return _dataset_cache["dataset"]


class ExplainRequest(BaseModel):
    """Request body for sample explanation."""

    sample_index: int
    model_name: Optional[str] = None
    num_features: int = 20


class DatasetInfoResponse(BaseModel):
    """Dataset information."""

    total_samples: int
    n_features: int
    n_benign: int
    n_malware: int
    n_families: int


@router.post("/explain")
async def explain_sample(request: ExplainRequest):
    """Explain a specific BODMAS dataset sample.

    Generates SHAP and forensic analysis for the given sample index.
    """
    from haruspex.api.main import get_model_store

    model_store = get_model_store()
    if not model_store:
        raise HTTPException(status_code=503, detail="No models loaded.")

    # Load dataset
    try:
        dataset = _get_dataset()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {e}")

    if request.sample_index < 0 or request.sample_index >= dataset.n_samples:
        raise HTTPException(
            status_code=400,
            detail=f"Sample index must be 0-{dataset.n_samples - 1}",
        )

    # Select model
    model_name = request.model_name
    if model_name and model_name in model_store:
        model = model_store[model_name]
        used_model = model_name
    elif "_default" in model_store:
        model = model_store["_default"]
        used_model = next(k for k in ["xgboost", "lightgbm"] if k in model_store)
    else:
        raise HTTPException(status_code=503, detail="No default model.")

    idx = request.sample_index
    x_sample = dataset.X[idx]
    true_label = int(dataset.y[idx])
    sample_meta = dataset.metadata.iloc[idx]

    feature_names = get_feature_names()

    # SHAP explanation
    from haruspex.xai.shap_explainer import SHAPExplainer

    shap_explainer = SHAPExplainer(model, feature_names=feature_names)
    shap_exp = shap_explainer.explain_sample(x_sample)

    # Forensic analysis
    from haruspex.forensics.static_analysis import StaticAnalyzer
    from haruspex.forensics.ioc_extractor import IOCExtractor
    from haruspex.correlation.correlator import Correlator

    analyzer = StaticAnalyzer()
    forensic_report = analyzer.analyze_feature_vector(x_sample, feature_names)

    ioc_extractor = IOCExtractor()
    iocs = ioc_extractor.extract_from_analysis(forensic_report)

    # Correlation
    correlator = Correlator()
    evidence = correlator.correlate(shap_exp, forensic_report, iocs, top_k=request.num_features)

    # Prediction
    pred_label = int(model.predict(x_sample.reshape(1, -1))[0])
    pred_proba = float(model.predict_proba(x_sample.reshape(1, -1))[0])
    prediction = "malware" if pred_label == 1 else "benign"
    confidence = pred_proba if pred_label == 1 else (1.0 - pred_proba)

    report = correlator.generate_report(
        sample_id=f"bodmas_{idx}",
        prediction=prediction,
        confidence=confidence,
        model_name=used_model,
        evidence_items=evidence,
        shap_explanation=shap_exp,
        forensic_report=forensic_report,
        iocs=iocs,
        sample_sha256=str(sample_meta.get("sha", "")),
        sample_family=str(sample_meta.get("family", "")),
        sample_category=str(sample_meta.get("category", "")),
    )

    return {
        **report.to_dict(),
        "true_label": "malware" if true_label == 1 else "benign",
        "sample_index": idx,
    }


@router.get("/dataset/info")
async def dataset_info():
    """Return BODMAS dataset summary information."""
    try:
        dataset = _get_dataset()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {e}")

    families = dataset.metadata["family"].replace("", None).dropna().unique()

    return {
        "total_samples": dataset.n_samples,
        "n_features": dataset.n_features,
        "n_benign": dataset.n_benign,
        "n_malware": dataset.n_malware,
        "n_families": len(families),
        "families": sorted(families.tolist())[:50],  # First 50
        "categories": dataset.metadata["category"].value_counts().to_dict(),
    }


@router.get("/dataset/samples")
async def list_samples(
    family: Optional[str] = None,
    category: Optional[str] = None,
    label: Optional[str] = None,
    offset: int = 0,
    limit: int = 50,
):
    """List BODMAS samples with optional filtering."""
    try:
        dataset = _get_dataset()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {e}")

    meta = dataset.metadata.copy()
    meta["index"] = range(len(meta))
    meta["label"] = ["malware" if y == 1 else "benign" for y in dataset.y]

    # Apply filters
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

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "samples": samples,
    }
