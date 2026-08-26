"""Dataset loaders and PyTorch Dataset classes for HARUSPEX Deep Learning models.

Supports:
  - BODMAS malware dataset (.npz or .dat memmap) with family/category encodings
  - EMBER 2018 dataset (.dat memmaps via np.fromfile / np.memmap)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from deepLearning.utils.config import (
    BACKEND_DIR,
    NUM_FEATURES,
    get_feature_names,
)

logger = logging.getLogger("haruspex.deepLearning.data")


@dataclass
class DatasetContainer:
    """Container for loaded dataset matrices and metadata."""

    X: np.ndarray                 # (N, 2381) float32
    y: np.ndarray                 # (N,) int32 (0=benign, 1=malware)
    metadata: pd.DataFrame        # sha, timestamp, family, category
    family_ids: np.ndarray        # (N,) int64 family class indices
    category_ids: np.ndarray      # (N,) int64 category class indices
    family_encoder: Dict[str, int] # Mapping family_name -> int
    category_encoder: Dict[str, int] # Mapping category_name -> int
    feature_names: list[str]


def load_bodmas_data(
    bodmas_dir: Optional[Path] = None,
) -> DatasetContainer:
    """Load BODMAS dataset with encoded family and category targets.

    Args:
        bodmas_dir: Directory containing BODMAS files. Defaults to backend/dataset/BODMAS.

    Returns:
        DatasetContainer with feature matrix, labels, family/category IDs, and encoders.
    """
    _dir = bodmas_dir or (BACKEND_DIR / "dataset" / "BODMAS")
    x_dat = _dir / "bodmas_X.dat"
    y_dat = _dir / "bodmas_y.dat"
    npz_path = _dir / "bodmas.npz"
    meta_path = _dir / "bodmas_metadata.csv"
    cat_path = _dir / "bodmas_malware_category.csv"

    # --- Load features X and labels y ---
    if x_dat.exists() and y_dat.exists():
        logger.info("Loading BODMAS features X and y from .dat memmaps: %s", x_dat)
        y = np.fromfile(str(y_dat), dtype=np.int32)
        X = np.memmap(str(x_dat), dtype=np.float32, mode="r").reshape(len(y), NUM_FEATURES)
    elif npz_path.exists():
        logger.info("Loading BODMAS from npz: %s", npz_path)
        data = np.load(str(npz_path), allow_pickle=True)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int32)
    else:
        raise FileNotFoundError(f"BODMAS dataset files not found in {_dir}")

    # --- Load metadata & categories ---
    if not (meta_path.exists() and cat_path.exists()):
        raise FileNotFoundError(f"BODMAS metadata/category CSVs missing in {_dir}")

    metadata = pd.read_csv(meta_path)
    categories = pd.read_csv(cat_path)

    metadata = metadata.merge(
        categories, left_on="sha", right_on="sha256", how="left"
    ).drop(columns=["sha256"], errors="ignore")

    metadata["family"] = metadata["family"].fillna("")
    metadata["category"] = metadata["category"].fillna("benign")

    # Build Label Encoders
    unique_families = sorted([f for f in metadata["family"].unique() if f != ""])
    family_encoder = {fam: i + 1 for i, fam in enumerate(unique_families)}
    family_encoder[""] = 0  # Benign or unknown

    unique_categories = sorted(metadata["category"].unique())
    category_encoder = {cat: i for i, cat in enumerate(unique_categories)}

    family_ids = np.array([family_encoder.get(f, 0) for f in metadata["family"]], dtype=np.int64)
    category_ids = np.array([category_encoder.get(c, 0) for c in metadata["category"]], dtype=np.int64)

    feature_names = get_feature_names()

    logger.info(
        "BODMAS dataset loaded: %d samples, %d features, %d unique families, %d categories",
        X.shape[0], X.shape[1], len(unique_families), len(unique_categories)
    )

    return DatasetContainer(
        X=X,
        y=y,
        metadata=metadata,
        family_ids=family_ids,
        category_ids=category_ids,
        family_encoder=family_encoder,
        category_encoder=category_encoder,
        feature_names=feature_names,
    )


def load_ember_data(
    ember_dir: Optional[Path] = None,
    partition: str = "test",  # "train" or "test"
    max_samples: int = 0,
) -> DatasetContainer:
    """Load EMBER 2018 dataset partition directly from .dat memmap files.

    Args:
        ember_dir: Directory containing EMBER files. Defaults to backend/dataset/ember2018.
        partition: "train" or "test".
        max_samples: Subsampling limit (0 = load all).

    Returns:
        DatasetContainer with feature matrix, labels, dummy family/category IDs.
    """
    _dir = ember_dir or (BACKEND_DIR / "dataset" / "ember2018")
    if partition == "train":
        x_dat = _dir / "X_train.dat"
        y_dat = _dir / "y_train.dat"
    else:
        x_dat = _dir / "X_test.dat"
        y_dat = _dir / "y_test.dat"

    if not (x_dat.exists() and y_dat.exists()):
        raise FileNotFoundError(f"EMBER {partition} files not found: {x_dat}, {y_dat}")

    logger.info("Loading EMBER (%s partition) from %s", partition, x_dat)

    y_raw = np.memmap(str(y_dat), dtype=np.float32, mode="r")
    valid_idx = np.where((y_raw != -1) & (~np.isnan(y_raw)))[0]

    if max_samples > 0 and len(valid_idx) > max_samples:
        rng = np.random.RandomState(42)
        valid_idx = rng.choice(valid_idx, size=max_samples, replace=False)
        valid_idx.sort()

    X_raw = np.memmap(str(x_dat), dtype=np.float32, mode="r").reshape(-1, NUM_FEATURES)

    X = np.asarray(X_raw[valid_idx], dtype=np.float32)
    y = np.asarray(y_raw[valid_idx], dtype=np.int32)

    # Dummy family & category IDs for EMBER (since EMBER lacks family/category labels)
    family_ids = np.zeros(len(y), dtype=np.int64)
    category_ids = np.zeros(len(y), dtype=np.int64)

    metadata = pd.DataFrame({
        "sha": [f"ember_{partition}_{i}" for i in valid_idx],
        "family": "",
        "category": np.where(y == 1, "malware", "benign"),
        "timestamp": "2018-01-01",
    })

    feature_names = get_feature_names()

    logger.info(
        "EMBER (%s) dataset loaded: %d labeled samples (%d benign, %d malware)",
        partition, X.shape[0], int((y == 0).sum()), int((y == 1).sum())
    )

    return DatasetContainer(
        X=X,
        y=y,
        metadata=metadata,
        family_ids=family_ids,
        category_ids=category_ids,
        family_encoder={"": 0},
        category_encoder={"benign": 0, "malware": 1},
        feature_names=feature_names,
    )


# ---------------------------------------------------------------------------
# PyTorch Dataset Definition
# ---------------------------------------------------------------------------

class MalwarePyTorchDataset(Dataset):
    """PyTorch Dataset for Malware Detection with optional Multi-task Targets."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        family_ids: Optional[np.ndarray] = None,
        category_ids: Optional[np.ndarray] = None,
    ):
        """Initialize dataset.

        Args:
            X: Feature matrix (N, 2381)
            y: Binary target array (N,)
            family_ids: Optional family class targets (N,)
            category_ids: Optional category class targets (N,)
        """
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

        if family_ids is not None:
            self.family_y = torch.as_tensor(family_ids, dtype=torch.long)
        else:
            self.family_y = torch.zeros(len(y), dtype=torch.long)

        if category_ids is not None:
            self.category_y = torch.as_tensor(category_ids, dtype=torch.long)
        else:
            self.category_y = torch.zeros(len(y), dtype=torch.long)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.X[idx],
            self.y[idx],
            self.family_y[idx],
            self.category_y[idx],
        )


def create_dataloader(
    X: np.ndarray,
    y: np.ndarray,
    family_ids: Optional[np.ndarray] = None,
    category_ids: Optional[np.ndarray] = None,
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """Create PyTorch DataLoader for training/eval."""
    ds = MalwarePyTorchDataset(X, y, family_ids, category_ids)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
