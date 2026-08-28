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


def _vectorize_uploaded_pe(content: bytes) -> np.ndarray:
    """Create the official EMBER v2 vector shared by EMBER/BODMAS-compatible models."""
    try:
        from haruspex.forensics.ember_vectorizer import EmberV2PEVectorizer
        return EmberV2PEVectorizer().extract(content)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not vectorize uploaded file with the EMBER v2 PE extractor: {exc}",
        ) from exc


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

            # Exact official EMBER v2 feature layout; BODMAS was trained on
            # the compatible 2,381-column EMBER schema.
            features = _vectorize_uploaded_pe(content)
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
        shap_exp = None
        try:
            shap_explainer = SHAPExplainer(model, feature_names=feature_names)
            shap_exp = shap_explainer.explain_sample(features)
        except Exception as e:
            logger.warning("SHAP explanation failed: %s", e)

        # LIME explanation (best-effort; uses pre-cached explainer)
        lime_exp = None
        try:
            from haruspex.api.main import get_lime_explainer
            lime_explainer = get_lime_explainer()
            if lime_explainer is not None:
                lime_exp = lime_explainer.explain_sample(model, features)
        except Exception as e:
            logger.warning("LIME explanation failed: %s", e)

        # Forensic analysis (if not already performed on real PE file)
        if forensic_report is None:
            forensic_report = static_analyzer.analyze_feature_vector(features, feature_names)
        if iocs is None:
            iocs = ioc_extractor.extract_from_analysis(forensic_report)

        # Correlation
        evidence = correlator.correlate(shap_exp, forensic_report, iocs, top_k=20)

        # Build report
        lime_features = lime_exp.top_features(k=10) if lime_exp else None
        report = correlator.generate_report(
            sample_id=sample_id,
            prediction=prediction,
            confidence=confidence,
            model_name=f"{used_group}/{used_model}",
            evidence_items=evidence,
            shap_explanation=shap_exp,
            forensic_report=forensic_report,
            iocs=iocs,
            lime_features=lime_features,
        )

        report_dict = report.to_dict()
        if file and file.filename:
            report_dict["metadata"]["feature_vectorizer"] = "official_ember_v2"
            report_dict["metadata"]["vector_schema"] = "EMBER v2 / BODMAS-compatible (2381 features)"
        return report_dict

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
            logger.info("Processing PE file upload for all-model testing: %s", file.filename)
            content = await file.read()

            with tempfile.NamedTemporaryFile(suffix=".exe", delete=False) as tmp:
                tmp.write(content)
                pe_path = Path(tmp.name)

            forensic_report = static_analyzer.analyze_pe(pe_path)
            iocs = ioc_extractor.extract_all(forensic_report, pe_path)

            features = _vectorize_uploaded_pe(content)
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
            "feature_vectorizer": "official_ember_v2" if file and file.filename else "caller_provided",
            "vector_schema": "EMBER v2 / BODMAS-compatible (2381 features)",
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
    """Run full analysis for an uploaded PE: ML verdict + static/YARA + SHAP."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Provide a PE file.")
    try:
        content = await file.read()
        forensic_summary, ioc_count = _uploaded_file_forensics(content, file.filename)

        from haruspex.api.main import get_model, get_model_store

        # Attempt ML prediction + SHAP + LIME with the verified EMBER v2 vectorizer
        ml_result = None
        shap_result = None
        lime_result = None
        try:
            features = _vectorize_uploaded_pe(content)
            x = features.reshape(1, -1)
            feature_names = get_feature_names()
            loaded_models = _get_loaded_models()
            if loaded_models:
                per_model = []
                for group_name, model_name, model in loaded_models:
                    pred_label = int(model.predict(x)[0])
                    pred_proba = float(model.predict_proba(x)[0])
                    prediction = "malware" if pred_label == 1 else "benign"
                    confidence = pred_proba if pred_label == 1 else (1.0 - pred_proba)
                    per_model.append({
                        "model_group": group_name,
                        "model_name": model_name,
                        "prediction": prediction,
                        "confidence": confidence,
                    })
                per_model.sort(key=lambda r: (r["model_group"], r["model_name"]))
                malware_votes = sum(1 for r in per_model if r["prediction"] == "malware")
                benign_votes = len(per_model) - malware_votes
                consensus = "tie"
                if malware_votes > benign_votes:
                    consensus = "malware"
                elif benign_votes > malware_votes:
                    consensus = "benign"
                ml_result = {
                    "available": True,
                    "consensus": consensus,
                    "malware_votes": malware_votes,
                    "benign_votes": benign_votes,
                    "per_model": per_model,
                }

                # SHAP for the first (default) model
                try:
                    default_model = loaded_models[0][2]
                    shap_explainer = SHAPExplainer(default_model, feature_names=feature_names)
                    shap_exp = shap_explainer.explain_sample(features)
                    shap_result = shap_exp.top_features(k=15)
                except Exception as e:
                    logger.warning("SHAP explanation failed for upload: %s", e)

                # LIME for the first (default) model
                try:
                    from haruspex.api.main import get_lime_explainer
                    lime_explainer = get_lime_explainer()
                    if lime_explainer is not None:
                        lime_exp = lime_explainer.explain_sample(default_model, features)
                        lime_result = lime_exp.top_features(k=10)
                except Exception as e:
                    logger.warning("LIME explanation failed for upload: %s", e)

        except Exception as e:
            ml_result = {"available": False, "error": str(e)}

        return {
            "sample_id": file.filename,
            "ml_verdict_available": ml_result.get("available", False) if ml_result else False,
            "ml_verdict": ml_result,
            "shap_features": shap_result or [],
            "lime_features": lime_result or [],
            "feature_vectorizer": "official_ember_v2",
            "vector_schema": "EMBER v2 / BODMAS-compatible (2381 features)",
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
