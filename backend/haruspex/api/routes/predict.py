"""Prediction endpoint — full HARUSPEX investigation pipeline.

POST /api/predict
  Accepts a PE file (multipart upload) OR a JSON feature vector.
  Returns a complete investigation report with:
    - Model prediction + confidence
    - SHAP + LIME explanations
    - Forensic analysis
    - Correlated evidence
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from haruspex.config import get_feature_names, NUM_FEATURES
from haruspex.correlation.correlator import Correlator
from haruspex.forensics.ioc_extractor import IOCExtractor
from haruspex.forensics.static_analysis import StaticAnalyzer
from haruspex.utils.logging import get_logger
from haruspex.xai.shap_explainer import SHAPExplainer

logger = get_logger(__name__)
router = APIRouter()


class FeatureVectorRequest(BaseModel):
    """Request body for feature vector prediction."""

    features: list[float]
    model_name: Optional[str] = None


class PredictionResponse(BaseModel):
    """Prediction response."""

    prediction: str
    confidence: float
    risk_score: float
    model_name: str
    evidence_summary: dict
    evidence_items: list[dict]
    top_shap_features: list[dict]
    forensic_summary: dict
    ioc_count: int


@router.post("/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    features_json: Optional[str] = Form(None),
    model_name: Optional[str] = Form(None),
):
    """Run the full HARUSPEX investigation pipeline.

    Accepts EITHER:
      - A PE file upload (multipart form)
      - A JSON string of feature values (form field)

    Returns a complete investigation report.
    """
    from haruspex.api.main import get_model_store

    model_store = get_model_store()
    if not model_store:
        raise HTTPException(status_code=503, detail="No models loaded. Run train.py first.")

    # Select model
    if model_name and model_name in model_store:
        model = model_store[model_name]
        used_model = model_name
    elif "_default" in model_store:
        model = model_store["_default"]
        used_model = next(k for k in ["xgboost", "lightgbm"] if k in model_store)
    else:
        raise HTTPException(status_code=503, detail="No default model available.")

    static_analyzer = StaticAnalyzer()
    ioc_extractor = IOCExtractor()
    correlator = Correlator()
    feature_names = get_feature_names()

    pe_path: Optional[Path] = None
    features: Optional[np.ndarray] = None
    forensic_report: Optional[ForensicReport] = None
    iocs: Optional[list] = None
    sample_id: str = "api_upload"

    try:
        # --- Mode 1: PE file upload ---
        if file and file.filename:
            logger.info("Processing PE file upload: %s", file.filename)
            content = await file.read()

            # Save to temp file for LIEF/YARA analysis
            with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as tmp:
                tmp.write(content)
                pe_path = Path(tmp.name)

            # Forensic analysis on real PE file
            forensic_report = static_analyzer.analyze_pe(pe_path)
            iocs = ioc_extractor.extract_all(forensic_report, pe_path)

            # Extract 2381 EMBER-compatible features from PE file bytes
            from haruspex.forensics.pe_feature_extractor import PEFeatureExtractor
            extractor = PEFeatureExtractor()
            features = extractor.extract(content)
            sample_id = file.filename

        # --- Mode 2: Feature vector ---
        elif features_json:
            import json

            try:
                feat_list = json.loads(features_json)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON in features_json")

            if len(feat_list) != NUM_FEATURES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Expected {NUM_FEATURES} features, got {len(feat_list)}",
                )

            features = np.array(feat_list, dtype=np.float32)

        else:
            raise HTTPException(
                status_code=400,
                detail="Provide either a PE file upload or a features_json form field.",
            )

        # --- Run pipeline on feature vector ---
        x = features.reshape(1, -1)

        # Prediction
        pred_label = int(model.predict(x)[0])
        pred_proba = float(model.predict_proba(x)[0])
        prediction = "malware" if pred_label == 1 else "benign"
        confidence = pred_proba if pred_label == 1 else (1.0 - pred_proba)

        # SHAP explanation
        shap_explainer = SHAPExplainer(model, feature_names=feature_names)
        shap_exp = shap_explainer.explain_sample(features)

        # Forensic analysis (if not already performed on real PE file)
        if forensic_report is None:
            forensic_report = static_analyzer.analyze_feature_vector(features, feature_names)
        if iocs is None:
            iocs = ioc_extractor.extract_from_analysis(forensic_report)

        # Correlation
        evidence = correlator.correlate(shap_exp, forensic_report, iocs, top_k=20)

        # Build report
        report = correlator.generate_report(
            sample_id=sample_id,
            prediction=prediction,
            confidence=confidence,
            model_name=used_model,
            evidence_items=evidence,
            shap_explanation=shap_exp,
            forensic_report=forensic_report,
            iocs=iocs,
        )

        return report.to_dict()

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Prediction pipeline error")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
    finally:
        # Cleanup temp PE file
        if pe_path and pe_path.exists():
            pe_path.unlink(missing_ok=True)


@router.post("/predict/vector")
async def predict_from_vector(request: FeatureVectorRequest):
    """Predict from a feature vector (JSON body).

    Alternative to the form-based /predict endpoint.
    """
    from haruspex.api.main import get_model_store

    model_store = get_model_store()
    if not model_store:
        raise HTTPException(status_code=503, detail="No models loaded.")

    if len(request.features) != NUM_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {NUM_FEATURES} features, got {len(request.features)}",
        )

    model_name = request.model_name
    if model_name and model_name in model_store:
        model = model_store[model_name]
        used_model = model_name
    elif "_default" in model_store:
        model = model_store["_default"]
        used_model = next(k for k in ["xgboost", "lightgbm"] if k in model_store)
    else:
        raise HTTPException(status_code=503, detail="No default model available.")

    features = np.array(request.features, dtype=np.float32)
    x = features.reshape(1, -1)

    # Quick prediction (no full pipeline)
    pred_label = int(model.predict(x)[0])
    pred_proba = float(model.predict_proba(x)[0])
    prediction = "malware" if pred_label == 1 else "benign"
    confidence = pred_proba if pred_label == 1 else (1.0 - pred_proba)

    # SHAP explanation
    feature_names = get_feature_names()
    shap_explainer = SHAPExplainer(model, feature_names=feature_names)
    shap_exp = shap_explainer.explain_sample(features)

    # Forensic + correlation
    static_analyzer = StaticAnalyzer()
    ioc_extractor = IOCExtractor()
    correlator = Correlator()

    forensic_report = static_analyzer.analyze_feature_vector(features, feature_names)
    iocs = ioc_extractor.extract_from_analysis(forensic_report)
    evidence = correlator.correlate(shap_exp, forensic_report, iocs, top_k=20)

    report = correlator.generate_report(
        sample_id="api_vector",
        prediction=prediction,
        confidence=confidence,
        model_name=used_model,
        evidence_items=evidence,
        shap_explanation=shap_exp,
        forensic_report=forensic_report,
        iocs=iocs,
    )

    return report.to_dict()
