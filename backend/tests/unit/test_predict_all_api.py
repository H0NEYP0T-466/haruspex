"""Unit tests for the /api/predict/all endpoint."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from haruspex.api.routes.predict import router
from haruspex.config import NUM_FEATURES


class _DummyModel:
    def __init__(self, pred: int, proba: float):
        self._pred = pred
        self._proba = proba

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.array([self._pred], dtype=np.int32)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.array([self._proba], dtype=np.float32)


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


def test_predict_all_vector_success(monkeypatch):
    client = _client()

    monkeypatch.setattr(
        "haruspex.api.routes.predict._get_loaded_models",
        lambda: [
            ("bodmas", "xgboost", _DummyModel(pred=0, proba=0.2)),
            ("ember", "tabnet", _DummyModel(pred=1, proba=0.9)),
        ],
    )

    features = [0.0] * NUM_FEATURES
    response = client.post("/api/predict/all", data={"features_json": json.dumps(features)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["models_tested"] == 2
    assert payload["summary"]["consensus"] == "tie"
    assert len(payload["results"]) == 2
    assert payload["results"][0]["model_group"] == "bodmas"
    assert payload["results"][1]["model_group"] == "ember"


def test_predict_all_requires_input(monkeypatch):
    client = _client()
    monkeypatch.setattr(
        "haruspex.api.routes.predict._get_loaded_models",
        lambda: [("bodmas", "xgboost", _DummyModel(pred=0, proba=0.2))],
    )

    response = client.post("/api/predict/all")
    assert response.status_code == 400
    assert "Provide either a PE file upload or a features_json form field." in response.text
