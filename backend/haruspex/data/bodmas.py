"""BODMAS dataset loader.

Loads the BODMAS malware analysis dataset from pre-downloaded files:
  - bodmas.npz        → feature matrix X (134435 × 2381) and labels y
  - bodmas_metadata.csv → SHA-256 hash, timestamp, malware family
  - bodmas_malware_category.csv → SHA-256 hash, malware category (14 types)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from haruspex.config import (
    BODMAS_CATEGORY_CSV,
    BODMAS_METADATA_CSV,
    BODMAS_NPZ,
    get_feature_names,
)
from haruspex.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BodmasDataset:
    """Container for the loaded BODMAS dataset."""

    X: np.ndarray                # (n_samples, 2381) float32 feature matrix
    y: np.ndarray                # (n_samples,) int32 labels (0=benign, 1=malware)
    metadata: pd.DataFrame       # sha, timestamp, family, category columns
    feature_names: list[str]     # 2381 EMBER-compatible feature names

    @property
    def n_samples(self) -> int:
        """Total number of samples."""
        return self.X.shape[0]

    @property
    def n_features(self) -> int:
        """Number of features per sample."""
        return self.X.shape[1]

    @property
    def n_malware(self) -> int:
        """Number of malware samples."""
        return int(self.y.sum())

    @property
    def n_benign(self) -> int:
        """Number of benign samples."""
        return int((self.y == 0).sum())


def load_bodmas(
    npz_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    category_path: Optional[str] = None,
) -> BodmasDataset:
    """Load the BODMAS dataset from disk.

    Args:
        npz_path: Path to bodmas.npz. Defaults to config.BODMAS_NPZ.
        metadata_path: Path to bodmas_metadata.csv. Defaults to config.BODMAS_METADATA_CSV.
        category_path: Path to bodmas_malware_category.csv. Defaults to config.BODMAS_CATEGORY_CSV.

    Returns:
        BodmasDataset with features, labels, metadata, and feature names.

    Raises:
        FileNotFoundError: If any dataset file is missing.
        ValueError: If row counts between npz and metadata CSV don't align.
    """
    _npz = npz_path or str(BODMAS_NPZ)
    _meta = metadata_path or str(BODMAS_METADATA_CSV)
    _cat = category_path or str(BODMAS_CATEGORY_CSV)

    logger.info("Loading BODMAS dataset from %s", _npz)

    # --- Load feature matrix and labels ---
    data = np.load(_npz, allow_pickle=True)
    X = data["X"]
    y = data["y"]

    logger.info("Features X: shape=%s, dtype=%s", X.shape, X.dtype)
    logger.info("Labels  y: shape=%s, dtype=%s | benign=%d, malware=%d",
                y.shape, y.dtype, int((y == 0).sum()), int(y.sum()))

    # --- Load metadata ---
    logger.info("Loading metadata from %s", _meta)
    metadata = pd.read_csv(_meta)

    # Validate alignment
    if len(metadata) != X.shape[0]:
        raise ValueError(
            f"Row count mismatch: npz has {X.shape[0]} samples, "
            f"metadata CSV has {len(metadata)} rows."
        )

    # Parse timestamps (mixed timezone-aware/naive in BODMAS)
    metadata["timestamp"] = pd.to_datetime(
        metadata["timestamp"], format="mixed", utc=True
    )

    # --- Load malware categories and merge ---
    logger.info("Loading malware categories from %s", _cat)
    categories = pd.read_csv(_cat)
    metadata = metadata.merge(
        categories, left_on="sha", right_on="sha256", how="left"
    ).drop(columns=["sha256"], errors="ignore")

    # Fill NaN family/category for benign samples
    metadata["family"] = metadata["family"].fillna("")
    metadata["category"] = metadata["category"].fillna("benign")

    # Generate feature names
    feature_names = get_feature_names()

    logger.info(
        "BODMAS loaded: %d samples (%d benign, %d malware), %d features, %d families",
        X.shape[0],
        int((y == 0).sum()),
        int(y.sum()),
        X.shape[1],
        metadata["family"].replace("", pd.NA).dropna().nunique(),
    )

    return BodmasDataset(X=X, y=y, metadata=metadata, feature_names=feature_names)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_families(metadata: pd.DataFrame) -> list[str]:
    """Return sorted list of unique malware family names (excluding benign)."""
    families = metadata["family"].replace("", pd.NA).dropna().unique().tolist()
    return sorted(families)


def get_timestamps(metadata: pd.DataFrame) -> pd.Series:
    """Return the timestamp column as a timezone-aware datetime Series."""
    ts = metadata["timestamp"]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        ts = pd.to_datetime(ts, format="mixed", utc=True)
    return ts


def is_malicious(y: np.ndarray, index: int) -> bool:
    """Check if a sample at the given index is malware."""
    return bool(y[index] == 1)


def get_family_sample_counts(metadata: pd.DataFrame) -> pd.Series:
    """Return a Series of family → sample count, sorted descending."""
    families = metadata.loc[metadata["family"] != "", "family"]
    return families.value_counts().sort_values(ascending=False)
