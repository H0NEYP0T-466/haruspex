"""Storage helpers — save/load models, plots, and reports to the outputs/ directory."""

from __future__ import annotations

import json
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI
import matplotlib.pyplot as plt

from haruspex.config import MODEL_DIR, PLOTS_DIR, REPORTS_DIR, ensure_output_dirs
from haruspex.utils.logging import get_logger

logger = get_logger(__name__)


def _timestamp_str() -> str:
    """Return a compact UTC timestamp for file naming."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


# ---------------------------------------------------------------------------
# Model persistence
# ---------------------------------------------------------------------------

def save_model(model: Any, name: str, directory: Optional[Path] = None) -> Path:
    """Save a trained model to disk.

    The model's own ``save()`` method is called if available (XGBoost/LightGBM
    native format), otherwise falls back to pickle.

    Args:
        model: Trained model instance (must have a ``save()`` method).
        name: Base filename (without extension).
        directory: Target directory. Defaults to ``config.MODEL_DIR``.

    Returns:
        Path to the saved file.
    """
    ensure_output_dirs()
    target_dir = directory or MODEL_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(model, "save"):
        path = target_dir / f"{name}.json"
        model.save(str(path))
    else:
        path = target_dir / f"{name}.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)

    logger.info("Model saved: %s", path)
    return path


def load_model(name: str, model_class: Any, directory: Optional[Path] = None) -> Any:
    """Load a trained model from disk.

    Args:
        name: Base filename (without extension).
        model_class: The model class to instantiate and load into.
        directory: Source directory. Defaults to ``config.MODEL_DIR``.

    Returns:
        Loaded model instance.
    """
    source_dir = directory or MODEL_DIR

    # Try JSON first (native format), then pickle
    json_path = source_dir / f"{name}.json"
    pkl_path = source_dir / f"{name}.pkl"

    if json_path.exists():
        model = model_class()
        model.load(str(json_path))
        logger.info("Model loaded from %s", json_path)
        return model
    elif pkl_path.exists():
        with open(pkl_path, "rb") as f:
            model = pickle.load(f)  # noqa: S301
        logger.info("Model loaded from %s", pkl_path)
        return model
    else:
        raise FileNotFoundError(
            f"No model file found for '{name}' in {source_dir}. "
            f"Looked for {json_path} and {pkl_path}."
        )


def list_saved_models(directory: Optional[Path] = None) -> list[dict[str, str]]:
    """List all saved models in the model directory.

    Returns:
        List of dicts with 'name', 'path', and 'format' keys.
    """
    source_dir = directory or MODEL_DIR
    if not source_dir.exists():
        return []

    models = []
    for path in sorted(source_dir.iterdir()):
        if path.suffix in (".json", ".pkl"):
            models.append({
                "name": path.stem,
                "path": str(path),
                "format": path.suffix.lstrip("."),
            })
    return models


# ---------------------------------------------------------------------------
# Plot persistence
# ---------------------------------------------------------------------------

def save_plot(
    fig: plt.Figure,
    name: str,
    subdir: str = "",
    directory: Optional[Path] = None,
) -> Path:
    """Save a matplotlib figure to disk.

    Args:
        fig: Matplotlib figure.
        name: Base filename (without extension).
        subdir: Optional subdirectory inside plots/.
        directory: Target directory. Defaults to ``config.PLOTS_DIR``.

    Returns:
        Path to the saved PNG file.
    """
    ensure_output_dirs()
    target_dir = (directory or PLOTS_DIR)
    if subdir:
        target_dir = target_dir / subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    path = target_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Plot saved: %s", path)
    return path


# ---------------------------------------------------------------------------
# Report persistence
# ---------------------------------------------------------------------------

def save_report(
    report: dict | list,
    name: str,
    directory: Optional[Path] = None,
    also_text: bool = True,
) -> Path:
    """Save an investigation or evaluation report as JSON (and optionally .txt).

    Args:
        report: Report data (must be JSON-serializable).
        name: Base filename (without extension).
        directory: Target directory. Defaults to ``config.REPORTS_DIR``.
        also_text: If True, also save a human-readable .txt version.

    Returns:
        Path to the saved JSON file.
    """
    ensure_output_dirs()
    target_dir = directory or REPORTS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    json_path = target_dir / f"{name}.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Report saved: %s", json_path)

    if also_text:
        txt_path = target_dir / f"{name}.txt"
        with open(txt_path, "w") as f:
            f.write(_report_to_text(report))
        logger.info("Text report saved: %s", txt_path)

    return json_path


def load_report(name: str, directory: Optional[Path] = None) -> dict:
    """Load a JSON report from disk."""
    source_dir = directory or REPORTS_DIR
    path = source_dir / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Report not found: {path}")
    with open(path) as f:
        return json.load(f)


def _report_to_text(report: dict | list, indent: int = 0) -> str:
    """Convert a report dict/list to human-readable text."""
    lines: list[str] = []
    prefix = "  " * indent

    if isinstance(report, dict):
        for key, value in report.items():
            if isinstance(value, (dict, list)):
                lines.append(f"{prefix}{key}:")
                lines.append(_report_to_text(value, indent + 1))
            else:
                lines.append(f"{prefix}{key}: {value}")
    elif isinstance(report, list):
        for i, item in enumerate(report):
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}[{i}]:")
                lines.append(_report_to_text(item, indent + 1))
            else:
                lines.append(f"{prefix}- {item}")
    else:
        lines.append(f"{prefix}{report}")

    return "\n".join(lines)
