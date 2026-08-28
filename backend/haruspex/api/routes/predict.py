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

import tempfile
from pathlib import Path
from typing import Any, Optional

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from haruspex.config import NUM_FEATURES, get_feature_names
from haruspex.correlation.correlator import Correlator
from haruspex.forensics.ioc_extractor import IOCExtractor
from haruspex.forensics.static_analysis import StaticAnalyzer
from haruspex.utils.logging import get_logger
from haruspex.xai.shap_explainer import SHAPExplainer

logger = get_logger(__name__)
router = APIRouter()


def _uploaded_file_forensics(content: bytes, filename: str) -> tuple[dict, int]:
    """Analyze a raw PE without turning an unvalidated vector into an ML verdict."""
    pe_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix or ".bin", delete=False) as tmp:
            tmp.write(content)
            pe_path = Path(tmp.name)
        report = StaticAnalyzer().analyze_pe(pe_path)
        iocs = IOCExtractor().extract_all(report, pe_path)
        return report.to_dict(), len(iocs)
    finally:
        if pe_path and pe_path.exists():
            pe_path.unlink(missing_ok=True)


def _require_training_compatible_upload_vectorizer() -> None:
    """Prevent scientifically invalid raw-upload model predictions."""
    from haruspex.forensics.pe_feature_extractor import PEFeatureExtractor
    if not PEFeatureExtractor.training_compatible:
        raise HTTPException(
            status_code=422,
            detail=(
                "ML verdicts for uploaded PE files are disabled: the installed raw-PE "
                "extractor has not been validated as the exact vectorizer used to train "
                "these 2,381-feature models. Use a stored dataset vector/sample for ML "
                "comparison; uploaded files still support forensic/YARA analysis via "
                "/api/predict/upload-forensics."
            ),
        )


class FeatureVectorRequest(BaseModel):
    """Request body for feature vector prediction."""

    features: list[float]
    model_name: Optional[str] = None
    model_group: Optional[str] = None


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


def _get_loaded_models() -> list[tuple[str, str, Any]]:
    """Return all loaded (group, model_name, model) tuples."""
    from haruspex.api.main import get_model_store

    store = get_model_store()
    loaded: list[tuple[str, str, Any]] = []
    for group_name, group_store in store.items():
        if group_name.startswith("_") or not isinstance(group_store, dict):
            continue
        for model_name, model_obj in group_store.items():
            if model_name.startswith("_"):
                continue
            loaded.append((group_name, model_name, model_obj))
    return loaded


@router.post("/predict")
async def predict(
    file: Optional[UploadFile] = File(None),
    features_json: Optional[str] = Form(None),
    model_name: Optional[str] = Form(None),
    model_group: Optional[str] = Form(None),
):
    """Run the full HARUSPEX investigation pipeline.

    Accepts EITHER:
      - A PE file upload (multipart form)
      - A JSON string of feature values (form field)

    Optional:
      - model_name: "xgboost" or "lightgbm"
      - model_group: "bodmas" or "ember"

    Returns a complete investigation report.
    """
    from haruspex.api.main import get_model

    try:
        model, used_model, used_group = get_model(model_name, model_group)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    static_analyzer = StaticAnalyzer()
    ioc_extractor = IOCExtractor()
    correlator = Correlator()
    feature_names = get_feature_names()

    pe_path: Optional[Path] = None
    features: Optional[np.ndarray] = None
    forensic_report = None
    iocs: Optional[list] = None
    sample_id: str = "api_upload"

    try:
        # --- Mode 1: PE file upload ---
        if file and file.filename:
            _require_training_compatible_upload_vectorizer()
            logger.info("Processing PE file upload: %s (group=%s, model=%s)",
                        file.filename, used_group, used_model)
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
            model_name=f"{used_group}/{used_model}",
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


@router.post("/predict/all")
async def predict_all(
    file: Optional[UploadFile] = File(None),
    features_json: Optional[str] = Form(None),
):
    """Run all loaded models on the same uploaded PE file or feature vector."""
    loaded_models = _get_loaded_models()
    if not loaded_models:
        raise HTTPException(status_code=503, detail="No models loaded.")

    static_analyzer = StaticAnalyzer()
    ioc_extractor = IOCExtractor()
    feature_names = get_feature_names()

    pe_path: Optional[Path] = None
    features: Optional[np.ndarray] = None
    forensic_report = None
    iocs: Optional[list] = None
    sample_id: str = "api_upload"

    try:
        if file and file.filename:
            _require_training_compatible_upload_vectorizer()
            logger.info("Processing PE file upload for all-model testing: %s", file.filename)
            content = await file.read()

            with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as tmp:
                tmp.write(content)
                pe_path = Path(tmp.name)

            forensic_report = static_analyzer.analyze_pe(pe_path)
            iocs = ioc_extractor.extract_all(forensic_report, pe_path)

            from haruspex.forensics.pe_feature_extractor import PEFeatureExtractor

            extractor = PEFeatureExtractor()
            features = extractor.extract(content)
            sample_id = file.filename

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
            forensic_report = static_analyzer.analyze_feature_vector(features, feature_names)
            iocs = ioc_extractor.extract_from_analysis(forensic_report)

        else:
            raise HTTPException(
                status_code=400,
                detail="Provide either a PE file upload or a features_json form field.",
            )

        x = features.reshape(1, -1)
        per_model_results = []
        for group_name, model_name, model in loaded_models:
            pred_label = int(model.predict(x)[0])
            pred_proba = float(model.predict_proba(x)[0])
            prediction = "malware" if pred_label == 1 else "benign"
            confidence = pred_proba if pred_label == 1 else (1.0 - pred_proba)

            per_model_results.append(
                {
                    "model_group": group_name,
                    "model_name": model_name,
                    "prediction": prediction,
                    "confidence": confidence,
                    "malware_probability": pred_proba,
                }
            )

        per_model_results.sort(key=lambda r: (r["model_group"], r["model_name"]))
        malware_votes = sum(1 for r in per_model_results if r["prediction"] == "malware")
        benign_votes = len(per_model_results) - malware_votes
        consensus = "tie"
        if malware_votes > benign_votes:
            consensus = "malware"
        elif benign_votes > malware_votes:
            consensus = "benign"

        return {
            "sample_id": sample_id,
            "models_tested": len(per_model_results),
            "summary": {
                "consensus": consensus,
                "malware_votes": malware_votes,
                "benign_votes": benign_votes,
            },
            "results": per_model_results,
            "forensic_summary": forensic_report.to_dict() if forensic_report else {},
            "ioc_count": len(iocs) if iocs else 0,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("All-model prediction pipeline error")
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")
    finally:
        if pe_path and pe_path.exists():
            pe_path.unlink(missing_ok=True)


@router.post("/predict/upload-forensics")
async def analyze_uploaded_file_forensics(file: UploadFile = File(...)):
    """Run safe static/YARA analysis for an uploaded PE; no ML verdict is claimed."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Provide a PE file.")
    try:
        content = await file.read()
        forensic_summary, ioc_count = _uploaded_file_forensics(content, file.filename)
        return {
            "sample_id": file.filename,
            "ml_verdict_available": False,
            "notice": (
                "Static and YARA analysis completed. ML verdicts are intentionally disabled "
                "until the exact training-time PE vectorizer is installed and verified."
            ),
            "forensic_summary": forensic_summary,
            "ioc_count": ioc_count,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Uploaded-file forensic analysis error")
        raise HTTPException(status_code=400, detail=f"Forensic analysis failed: {str(e)}")


@router.post("/predict/vector")
async def predict_from_vector(request: FeatureVectorRequest):
    """Predict from a feature vector (JSON body).

    Alternative to the form-based /predict endpoint.
    """
    from haruspex.api.main import get_model

    if len(request.features) != NUM_FEATURES:
        raise HTTPException(
            status_code=400,
            detail=f"Expected {NUM_FEATURES} features, got {len(request.features)}",
        )

    try:
        model, used_model, used_group = get_model(request.model_name, request.model_group)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

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
        model_name=f"{used_group}/{used_model}",
        evidence_items=evidence,
        shap_explanation=shap_exp,
        forensic_report=forensic_report,
        iocs=iocs,
    )

    return report.to_dict()
