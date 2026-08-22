"""HARUSPEX FastAPI application — main entry point."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from haruspex.api.routes.explain import router as explain_router
from haruspex.api.routes.predict import router as predict_router
from haruspex.config import API_HOST, API_PORT, ensure_output_dirs
from haruspex.models.xgboost_model import XGBoostDetector
from haruspex.models.lightgbm_model import LightGBMDetector
from haruspex.utils.logging import get_logger
from haruspex.utils.storage import load_model, list_saved_models

logger = get_logger(__name__)

# Global model store — loaded on startup
_model_store: dict[str, Any] = {}


def get_model_store() -> dict[str, Any]:
    """Return the global model store."""
    return _model_store


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load trained models on application startup."""
    ensure_output_dirs()
    logger.info("HARUSPEX API starting up — loading models...")

    # Try loading each model type
    model_classes = {
        "xgboost": XGBoostDetector,
        "lightgbm": LightGBMDetector,
    }

    for name, cls in model_classes.items():
        try:
            model = load_model(name, cls)
            _model_store[name] = model
            logger.info("Loaded model: %s", name)
        except FileNotFoundError:
            logger.warning("Model '%s' not found — train first via scripts/train.py", name)

    if _model_store:
        # Set default model (prefer xgboost)
        default = "xgboost" if "xgboost" in _model_store else next(iter(_model_store))
        _model_store["_default"] = _model_store[default]
        logger.info("Default model: %s", default)
    else:
        logger.warning("No models loaded — /predict will return errors")

    logger.info("HARUSPEX API ready (%d models loaded)", len(_model_store) - (1 if "_default" in _model_store else 0))

    yield

    logger.info("HARUSPEX API shutting down")


# Create FastAPI app
app = FastAPI(
    title="HARUSPEX",
    description=(
        "Explainable AI framework for zero-day malware detection "
        "and digital forensic investigation."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware — allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",   # Vite dev server
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(predict_router, prefix="/api", tags=["prediction"])
app.include_router(explain_router, prefix="/api", tags=["explanation"])


@app.get("/api/health")
async def health_check():
    """Liveness check endpoint."""
    models = [k for k in _model_store if not k.startswith("_")]
    return {
        "status": "healthy",
        "models_loaded": models,
        "version": "0.1.0",
    }


@app.get("/api/models")
async def list_models():
    """List available trained models."""
    loaded = [k for k in _model_store if not k.startswith("_")]
    saved = list_saved_models()
    return {
        "loaded": loaded,
        "saved": saved,
        "default": next(
            (k for k in ["xgboost", "lightgbm"] if k in _model_store),
            None,
        ),
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "haruspex.api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )
