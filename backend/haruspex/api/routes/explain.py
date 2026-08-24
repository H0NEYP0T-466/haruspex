"""Explanation endpoint — SHAP/LIME for individual dataset samples.

POST /api/explain       — explain a specific sample by dataset index
POST /api/explain/random — explain a random sample from the active dataset
GET  /api/dataset/info  — dataset summary
GET  /api/dataset/samples — list samples with filtering
"""

from __future__ import annotations

import random as stdlib_random
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from haruspex.config import get_feature_names
from haruspex.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# Cache loaded datasets
_dataset_cache: dict = {}


def _get_dataset(dataset_name: str = "bodmas"):
    """Lazy-load a dataset by name."""
    if dataset_name not in _dataset_cache:
        if dataset_name == "bodmas":
            from haruspex.data.bodmas import load_bodmas
            _dataset_cache["bodmas"] = load_bodmas()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    return _dataset_cache[dataset_name]


class ExplainRequest(BaseModel):
    """Request body for sample explanation."""

    sample_index: int
    model_name: Optional[str] = None
    model_group: Optional[str] = None
    num_features: int = 20


class RandomExplainRequest(BaseModel):
    """Request body for random sample explanation."""

    model_name: Optional[str] = None
    model_group: Optional[str] = None
    num_features: int = 20
    label_filter: Optional[str] = None  # "malware", "benign", or None for any


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


@router.post("/explain")
async def explain_sample(request: ExplainRequest):
    """Explain a specific BODMAS dataset sample.

    Generates SHAP and forensic analysis for the given sample index.
    """
    from haruspex.api.main import get_model

    try:
        model, used_model, used_group = get_model(request.model_name, request.model_group)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Load dataset
    try:
        dataset = _get_dataset("bodmas")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {e}")

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


@router.post("/explain/random")
async def explain_random_sample(request: RandomExplainRequest):
    """Explain a random sample from the BODMAS dataset.

    Optionally filter by label ("malware" or "benign").
    """
    from haruspex.api.main import get_model

    try:
        model, used_model, used_group = get_model(request.model_name, request.model_group)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        dataset = _get_dataset("bodmas")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {e}")

    # Pick a random sample, optionally filtered by label
    if request.label_filter == "malware":
        valid_indices = np.where(dataset.y == 1)[0]
    elif request.label_filter == "benign":
        valid_indices = np.where(dataset.y == 0)[0]
    else:
        valid_indices = np.arange(dataset.n_samples)

    if len(valid_indices) == 0:
        raise HTTPException(status_code=404, detail="No samples match the filter.")

    idx = int(stdlib_random.choice(valid_indices))
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


@router.get("/dataset/info")
async def dataset_info():
    """Return BODMAS dataset summary information."""
    from haruspex.data.bodmas import get_family_sample_counts

    try:
        dataset = _get_dataset("bodmas")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {e}")

    families = dataset.metadata["family"].replace("", None).dropna().unique()
    family_counts = get_family_sample_counts(dataset.metadata)

    return {
        "total_samples": dataset.n_samples,
        "n_features": dataset.n_features,
        "n_benign": dataset.n_benign,
        "n_malware": dataset.n_malware,
        "n_families": len(families),
        "families": sorted(families.tolist())[:50],  # First 50 (alphabetical)
        "top_families": [
            {"name": name, "count": int(count)}
            for name, count in family_counts.head(10).items()
        ],
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
        dataset = _get_dataset("bodmas")
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
