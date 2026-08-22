"""HARUSPEX configuration — single source of truth for paths, hyperparameters, and constants."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Load .env from backend/ directory
# ---------------------------------------------------------------------------
_BACKEND_DIR = Path(__file__).resolve().parent.parent
load_dotenv(_BACKEND_DIR / ".env")

# ---------------------------------------------------------------------------
# Path configuration
# ---------------------------------------------------------------------------
BACKEND_DIR: Path = _BACKEND_DIR
PROJECT_ROOT: Path = _BACKEND_DIR.parent

DATASET_DIR: Path = _BACKEND_DIR / os.getenv("DATASET_DIR", "dataset/BODMAS")
OUTPUT_DIR: Path = (
    Path(os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "outputs")))
    if os.path.isabs(os.getenv("OUTPUT_DIR", ""))
    else _BACKEND_DIR / os.getenv("OUTPUT_DIR", "../outputs")
)
MODEL_DIR: Path = OUTPUT_DIR / "models"
PLOTS_DIR: Path = OUTPUT_DIR / "plots"
REPORTS_DIR: Path = OUTPUT_DIR / "reports"

# ---------------------------------------------------------------------------
# Dataset files
# ---------------------------------------------------------------------------
BODMAS_NPZ: Path = DATASET_DIR / "bodmas.npz"
BODMAS_METADATA_CSV: Path = DATASET_DIR / "bodmas_metadata.csv"
BODMAS_CATEGORY_CSV: Path = DATASET_DIR / "bodmas_malware_category.csv"

# ---------------------------------------------------------------------------
# Random seed
# ---------------------------------------------------------------------------
RANDOM_SEED: int = int(os.getenv("RANDOM_SEED", "42"))

# ---------------------------------------------------------------------------
# Temporal split cutoff
# ---------------------------------------------------------------------------
TEMPORAL_CUTOFF: str = os.getenv("TEMPORAL_CUTOFF", "2020-06-01")

# ---------------------------------------------------------------------------
# Held-out families for zero-day (family-based) evaluation
# Chosen: 4 mid-size families to keep test set manageable but meaningful
# ---------------------------------------------------------------------------
HELD_OUT_FAMILIES: list[str] = [
    "gepys",       # 1124 samples — trojan
    "benjamin",    # 1071 samples — worm
    "musecador",   # 1054 samples — trojan
    "autoit",      #  969 samples — trojan
]

# ---------------------------------------------------------------------------
# API configuration
# ---------------------------------------------------------------------------
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8014"))


# ---------------------------------------------------------------------------
# Model hyperparameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class XGBoostParams:
    """Default hyperparameters for XGBoost malware detector."""

    objective: str = "binary:logistic"
    eval_metric: str = "logloss"
    tree_method: str = "hist"
    n_estimators: int = 200
    max_depth: int = 8
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5
    random_state: int = RANDOM_SEED
    verbosity: int = 0

    def to_dict(self) -> dict:
        """Return as dict suitable for XGBClassifier kwargs."""
        return {
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "tree_method": self.tree_method,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "random_state": self.random_state,
            "verbosity": self.verbosity,
        }


@dataclass(frozen=True)
class LightGBMParams:
    """Default hyperparameters for LightGBM malware detector."""

    objective: str = "binary"
    metric: str = "binary_logloss"
    n_estimators: int = 200
    max_depth: int = 8
    learning_rate: float = 0.1
    num_leaves: int = 63
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_samples: int = 20
    random_state: int = RANDOM_SEED
    verbose: int = -1

    def to_dict(self) -> dict:
        """Return as dict suitable for LGBMClassifier kwargs."""
        return {
            "objective": self.objective,
            "metric": self.metric,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_samples": self.min_child_samples,
            "random_state": self.random_state,
            "verbose": self.verbose,
        }


XGBOOST_PARAMS = XGBoostParams()
LIGHTGBM_PARAMS = LightGBMParams()


# ---------------------------------------------------------------------------
# EMBER-compatible feature names (2,381 features)
# ---------------------------------------------------------------------------
# BODMAS uses the EMBER feature extraction schema.  The .npz file does NOT
# store feature names, so we generate them from the known EMBER layout:
#
#   Group                  | Count | Indices
#   -----------------------|-------|----------
#   Byte histogram         |   256 |    0 –  255
#   Byte-entropy histogram |   256 |  256 –  511
#   String info            |   104 |  512 –  615
#   General file info      |    10 |  616 –  625
#   Header info            |    62 |  626 –  687
#   Section info           |   255 |  688 –  942
#   Import info            |  1280 |  943 – 2222
#   Export info            |   128 | 2223 – 2350
#   Data directories       |    30 | 2351 – 2380
#   -----------------------|-------|----------
#   Total                  |  2381 |
# ---------------------------------------------------------------------------

# Feature group boundaries (start_index, count, prefix)
FEATURE_GROUPS: list[tuple[str, int, int]] = [
    ("byte_hist",           0,   256),
    ("byte_entropy_hist", 256,   256),
    ("string_info",       512,   104),
    ("general_info",      616,    10),
    ("header_info",       626,    62),
    ("section_info",      688,   255),
    ("import_info",       943,  1280),
    ("export_info",      2223,   128),
    ("data_dir_info",    2351,    30),
]

NUM_FEATURES: int = 2381


def get_feature_names() -> list[str]:
    """Generate EMBER-compatible feature names for the 2,381 BODMAS features."""
    names: list[str] = []
    for group_name, start, count in FEATURE_GROUPS:
        for i in range(count):
            names.append(f"{group_name}_{i}")
    assert len(names) == NUM_FEATURES, f"Expected {NUM_FEATURES} features, got {len(names)}"
    return names


def get_feature_group(index: int) -> Optional[str]:
    """Return the feature group name for a given feature index."""
    for group_name, start, count in FEATURE_GROUPS:
        if start <= index < start + count:
            return group_name
    return None


def get_feature_group_range(group_name: str) -> Optional[tuple[int, int]]:
    """Return (start_index, end_index) for a feature group."""
    for name, start, count in FEATURE_GROUPS:
        if name == group_name:
            return (start, start + count)
    return None


# ---------------------------------------------------------------------------
# Ensure output directories exist
# ---------------------------------------------------------------------------
def ensure_output_dirs() -> None:
    """Create all output directories if they don't exist."""
    for d in (OUTPUT_DIR, MODEL_DIR, PLOTS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
