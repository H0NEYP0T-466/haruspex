"""HARUSPEX FastAPI application — main entry point."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure backend/ and project root (for deepLearning) are on sys.path
_backend_dir = Path(__file__).resolve().parent.parent.parent
_project_root = _backend_dir.parent

for p in [_backend_dir, _project_root]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from haruspex.api.routes.explain import router as explain_router
from haruspex.api.routes.predict import router as predict_router
from haruspex.config import API_HOST, API_PORT, ensure_output_dirs
from haruspex.models.xgboost_model import XGBoostDetector
from haruspex.models.lightgbm_model import LightGBMDetector
from haruspex.utils.logging import get_logger
from haruspex.utils.storage import load_model, list_saved_models

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Model store — organized by group (bodmas / ember)
# ---------------------------------------------------------------------------
# Structure:
#   _model_store = {
#       "bodmas": {"xgboost": model, "lightgbm": model, "_default": model},
#       "ember":  {"xgboost": model, "lightgbm": model, "_default": model},
#       "_active_group": "bodmas",
#   }
_model_store: dict[str, Any] = {}


def get_model_store() -> dict[str, Any]:
    """Return the global model store."""
    return _model_store


def get_model(model_name: str | None = None, model_group: str | None = None) -> tuple[Any, str, str]:
    """Resolve a model from the store.

    Args:
        model_name: Optional model name ("xgboost", "lightgbm", "mlp", "tabnet").
        model_group: Optional group name ("bodmas" or "ember").

    Returns:
        Tuple of (model, used_model_name, used_group_name).

    Raises:
        ValueError: If requested model/group is not loaded.
    """
    # Determine group
    group = model_group or _model_store.get("_active_group", "bodmas")
    if group not in _model_store or not isinstance(_model_store.get(group), dict):
        available = [k for k in _model_store if not k.startswith("_") and isinstance(_model_store[k], dict)]
        if not available:
            raise ValueError("No models loaded. Please train models first.")
        group = available[0]

    group_store = _model_store[group]

    # Determine model within group
    if model_name:
        clean_name = model_name.lower().strip()
        if clean_name in group_store:
            return group_store[clean_name], clean_name, group
        else:
            avail = [k for k in group_store if not k.startswith("_")]
            raise ValueError(
                f"Model '{model_name}' is not trained or loaded in group '{group}'. "
                f"Available models in '{group}': {', '.join(avail) if avail else 'none'}"
            )
    elif "_default" in group_store:
        name = next((k for k in ["xgboost", "lightgbm", "mlp", "tabnet"] if k in group_store), "unknown")
        return group_store["_default"], name, group
    else:
        raise ValueError(f"No models available in group '{group}'.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load trained models on application startup."""
    ensure_output_dirs()
    logger.info("HARUSPEX API starting up — loading models...")

    # Model groups: (group_name, file_prefix)
    model_groups = [
        ("bodmas", ""),          # xgboost.json, lightgbm.json, mlp.pth, tabnet.zip
        ("ember", "ember_"),     # ember_xgboost.json, ember_lightgbm.json, ember_mlp.pth, ember_tabnet.zip
    ]

    from deepLearning.models.group_mlp import GroupMLPDetector
    from deepLearning.models.tabnet_model import TabNetDetector

    model_classes = {
        "xgboost": XGBoostDetector,
        "lightgbm": LightGBMDetector,
        "mlp": GroupMLPDetector,
        "tabnet": TabNetDetector,
    }

    total_loaded = 0
    for group_name, prefix in model_groups:
        group_store: dict[str, Any] = {}

        for algo_name, cls in model_classes.items():
            file_name = f"{prefix}{algo_name}"
            try:
                model = load_model(file_name, cls)
                group_store[algo_name] = model
                logger.info("Loaded model: %s/%s (file: %s)", group_name, algo_name, file_name)
                total_loaded += 1
            except FileNotFoundError:
                logger.warning(
                    "Model '%s/%s' not found (file: %s) — train first",
                    group_name, algo_name, file_name,
                )

        if group_store:
            # Set default model within group (prefer xgboost)
            default = "xgboost" if "xgboost" in group_store else next(iter(group_store))
            group_store["_default"] = group_store[default]
            _model_store[group_name] = group_store
            logger.info("Group '%s': %d models loaded (default: %s)",
                        group_name, len(group_store) - 1, default)

    # Set active group (prefer bodmas, fallback to first available)
    if _model_store:
        active = "bodmas" if "bodmas" in _model_store else next(
            k for k in _model_store if not k.startswith("_")
        )
        _model_store["_active_group"] = active

    logger.info("HARUSPEX API ready (%d models loaded across %d groups)",
                total_loaded,
                sum(1 for k in _model_store if not k.startswith("_")))

    yield

    logger.info("HARUSPEX API shutting down")


# Create FastAPI app
app = FastAPI(
    title="HARUSPEX",
    description=(
        "Explainable AI framework for zero-day malware detection "
        "and digital forensic investigation."
    ),
    version="0.2.0",
    lifespan=lifespan,
)

# CORS middleware — allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    groups = {}
    for group_name in _model_store:
        if group_name.startswith("_"):
            continue
        group = _model_store[group_name]
        if isinstance(group, dict):
            groups[group_name] = [k for k in group if not k.startswith("_")]

    return {
        "status": "healthy",
        "model_groups": groups,
        "active_group": _model_store.get("_active_group", None),
        "version": "0.2.0",
    }


@app.get("/api/models")
async def list_models():
    """List available trained models, organized by group."""
    groups = {}
    for group_name in _model_store:
        if group_name.startswith("_"):
            continue
        group = _model_store[group_name]
        if isinstance(group, dict):
            groups[group_name] = {
                "models": [k for k in group if not k.startswith("_")],
                "default": next(
                    (k for k in ["xgboost", "lightgbm"] if k in group), None
                ),
            }

    saved = list_saved_models()
    return {
        "groups": groups,
        "active_group": _model_store.get("_active_group", None),
        "saved": saved,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "haruspex.api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
    )
