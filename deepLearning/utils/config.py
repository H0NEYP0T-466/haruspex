"""HARUSPEX Deep Learning Configuration — Single source of truth for DL models.

Tries to import shared constants from `haruspex.config` if available,
with standalone fallbacks so the `deepLearning/` module can run independently.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

# Ensure project root & backend are on sys.path
_CURRENT_DIR = Path(__file__).resolve().parent
_DL_DIR = _CURRENT_DIR.parent
_PROJECT_ROOT = _DL_DIR.parent
_BACKEND_DIR = _PROJECT_ROOT / "backend"

for path in [_PROJECT_ROOT, _BACKEND_DIR, _DL_DIR]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# ---------------------------------------------------------------------------
# Attempt import from haruspex.config
# ---------------------------------------------------------------------------
try:
    from haruspex.config import (
        FEATURE_GROUPS as _HARUSPEX_FEATURE_GROUPS,
        NUM_FEATURES as _HARUSPEX_NUM_FEATURES,
        RANDOM_SEED as _HARUSPEX_RANDOM_SEED,
        TEMPORAL_CUTOFF as _HARUSPEX_TEMPORAL_CUTOFF,
        HELD_OUT_FAMILIES as _HARUSPEX_HELD_OUT_FAMILIES,
        PROJECT_ROOT as _HARUSPEX_PROJECT_ROOT,
        BACKEND_DIR as _HARUSPEX_BACKEND_DIR,
        OUTPUT_DIR as _HARUSPEX_OUTPUT_DIR,
        get_feature_names as _haruspex_get_feature_names,
    )
    _HAS_HARUSPEX = True
except ImportError:
    _HAS_HARUSPEX = False

# ---------------------------------------------------------------------------
# Project Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT: Path = _HARUSPEX_PROJECT_ROOT if _HAS_HARUSPEX else _PROJECT_ROOT
BACKEND_DIR: Path = _HARUSPEX_BACKEND_DIR if _HAS_HARUSPEX else _BACKEND_DIR
DL_DIR: Path = _DL_DIR

OUTPUT_DIR: Path = (
    _HARUSPEX_OUTPUT_DIR if _HAS_HARUSPEX else (PROJECT_ROOT / "outputs")
)

DL_MODELS_DIR: Path = DL_DIR / "models"
PHASE1_DIR: Path = DL_DIR / "phase1"
PHASE2_DIR: Path = DL_DIR / "phase2"

# ---------------------------------------------------------------------------
# Constants & Dataset Configurations
# ---------------------------------------------------------------------------
NUM_FEATURES: int = 2381
RANDOM_SEED: int = 42
TEMPORAL_CUTOFF: str = "2020-06-01"
HELD_OUT_FAMILIES: list[str] = ["gepys", "benjamin", "musecador", "autoit"]

# Feature Group Definitions (9 groups, exact index ranges)
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


def get_feature_names() -> list[str]:
    """Generate EMBER-compatible feature names for the 2,381 features."""
    if _HAS_HARUSPEX:
        return _haruspex_get_feature_names()
    names: list[str] = []
    for group_name, start, count in FEATURE_GROUPS:
        for i in range(count):
            names.append(f"{group_name}_{i}")
    return names


# ---------------------------------------------------------------------------
# Deep Learning Hyperparameters
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GroupMLPHyperparameters:
    """Hyperparameters for Group-wise Residual MLP."""

    d_model: int = 128
    n_heads: int = 2
    d_k: int = 64
    gate_hidden_dim: int = 64
    n_res_blocks: int = 6
    dropout: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size_bodmas: int = 1024
    batch_size_ember: int = 2048
    max_epochs: int = 100
    patience: int = 15
    aux_family_weight: float = 0.3
    aux_category_weight: float = 0.2


@dataclass(frozen=True)
class TabNetHyperparameters:
    """Hyperparameters for TabNet Classifier."""

    n_d: int = 64
    n_a: int = 64
    n_steps: int = 5
    gamma: float = 1.5
    lambda_sparse: float = 1e-3
    batch_size_bodmas: int = 1024
    batch_size_ember: int = 2048
    max_epochs: int = 100
    patience: int = 15
    lr: float = 2e-2


GROUP_MLP_PARAMS = GroupMLPHyperparameters()
TABNET_PARAMS = TabNetHyperparameters()

# ---------------------------------------------------------------------------
# PyTorch Device & Threading Helper
# ---------------------------------------------------------------------------
def get_device():
    """Return PyTorch torch.device ('cuda' if available else 'cpu') & tune CPU threads."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")

    # Tune CPU multi-threading to utilize all physical CPU cores
    num_cores = os.cpu_count() or 4
    try:
        torch.set_num_threads(num_cores)
        torch.set_num_interop_threads(max(1, num_cores // 2))
    except Exception:
        pass

    return torch.device("cpu")
