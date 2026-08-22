"""Dataset splitting strategies for malware detection evaluation.

Three split strategies designed to test different aspects of generalisation:
  1. Random split — standard stratified 80/20 (sanity-check baseline)
  2. Family-held-out — zero-day evaluation: entire malware families removed from training
  3. Temporal split — zero-day evaluation: train on older samples, test on newer ones
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from haruspex.config import HELD_OUT_FAMILIES, RANDOM_SEED, TEMPORAL_CUTOFF
from haruspex.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SplitResult:
    """Container for a train/test split with full traceability."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    train_indices: np.ndarray
    test_indices: np.ndarray
    metadata_train: pd.DataFrame
    metadata_test: pd.DataFrame
    split_name: str

    @property
    def train_size(self) -> int:
        return len(self.y_train)

    @property
    def test_size(self) -> int:
        return len(self.y_test)

    def summary(self) -> dict:
        """Return a summary dict of the split."""
        return {
            "split_name": self.split_name,
            "train_size": self.train_size,
            "test_size": self.test_size,
            "train_malware_ratio": float(self.y_train.mean()),
            "test_malware_ratio": float(self.y_test.mean()),
        }


def random_split(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
) -> SplitResult:
    """Standard stratified random split.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Label array (n_samples,).
        metadata: Metadata DataFrame aligned with X/y.
        test_size: Fraction of data for testing.
        seed: Random seed for reproducibility.

    Returns:
        SplitResult with stratified train/test partition.
    """
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=seed, stratify=y
    )

    result = SplitResult(
        X_train=X[train_idx],
        y_train=y[train_idx],
        X_test=X[test_idx],
        y_test=y[test_idx],
        train_indices=train_idx,
        test_indices=test_idx,
        metadata_train=metadata.iloc[train_idx].reset_index(drop=True),
        metadata_test=metadata.iloc[test_idx].reset_index(drop=True),
        split_name="random",
    )

    logger.info(
        "Random split: train=%d (%.1f%% malware), test=%d (%.1f%% malware)",
        result.train_size,
        result.y_train.mean() * 100,
        result.test_size,
        result.y_test.mean() * 100,
    )
    return result


def family_held_out_split(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    held_out_families: list[str] | None = None,
    val_size: float = 0.1,
    seed: int = RANDOM_SEED,
) -> SplitResult:
    """Zero-day evaluation: hold out entire malware families.

    ALL samples from the specified families go to the test set — none in training.
    Remaining families (plus all benign samples) form the train set.

    Args:
        X: Feature matrix.
        y: Label array.
        metadata: Metadata DataFrame with 'family' column.
        held_out_families: Families to exclude from training. Defaults to config.
        val_size: Unused (kept for API consistency). Validation is done internally.
        seed: Random seed.

    Returns:
        SplitResult where test set contains only samples from held-out families.
    """
    families = held_out_families or HELD_OUT_FAMILIES
    logger.info("Family-held-out split: holding out families %s", families)

    # Identify test indices (samples belonging to held-out families)
    family_col = metadata["family"]
    test_mask = family_col.isin(families)
    test_idx = np.where(test_mask.values)[0]
    train_idx = np.where(~test_mask.values)[0]

    if len(test_idx) == 0:
        raise ValueError(
            f"No samples found for held-out families {families}. "
            f"Available families: {sorted(family_col.unique()[:20])}"
        )

    result = SplitResult(
        X_train=X[train_idx],
        y_train=y[train_idx],
        X_test=X[test_idx],
        y_test=y[test_idx],
        train_indices=train_idx,
        test_indices=test_idx,
        metadata_train=metadata.iloc[train_idx].reset_index(drop=True),
        metadata_test=metadata.iloc[test_idx].reset_index(drop=True),
        split_name="family_held_out",
    )

    logger.info(
        "Family-held-out split: train=%d (%.1f%% malware), test=%d (%.1f%% malware), "
        "held-out families=%s",
        result.train_size,
        result.y_train.mean() * 100,
        result.test_size,
        result.y_test.mean() * 100,
        families,
    )
    return result


def temporal_split(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    cutoff_date: str | None = None,
) -> SplitResult:
    """Zero-day evaluation: temporal split by first-seen timestamp.

    Training set = samples seen BEFORE the cutoff date.
    Test set = samples seen ON or AFTER the cutoff date.

    Args:
        X: Feature matrix.
        y: Label array.
        metadata: Metadata DataFrame with 'timestamp' column.
        cutoff_date: ISO-format date string (e.g. '2020-06-01'). Defaults to config.

    Returns:
        SplitResult with temporal train/test partition.
    """
    cutoff = cutoff_date or TEMPORAL_CUTOFF
    logger.info("Temporal split: cutoff date = %s", cutoff)

    # Parse timestamps if needed
    timestamps = metadata["timestamp"]
    if not pd.api.types.is_datetime64_any_dtype(timestamps):
        timestamps = pd.to_datetime(timestamps, format="mixed", utc=True)

    cutoff_ts = pd.Timestamp(cutoff, tz="UTC")
    train_mask = timestamps < cutoff_ts
    train_idx = np.where(train_mask.values)[0]
    test_idx = np.where(~train_mask.values)[0]

    if len(test_idx) == 0:
        raise ValueError(
            f"No samples found after cutoff date {cutoff}. "
            f"Timestamp range: {timestamps.min()} to {timestamps.max()}"
        )

    result = SplitResult(
        X_train=X[train_idx],
        y_train=y[train_idx],
        X_test=X[test_idx],
        y_test=y[test_idx],
        train_indices=train_idx,
        test_indices=test_idx,
        metadata_train=metadata.iloc[train_idx].reset_index(drop=True),
        metadata_test=metadata.iloc[test_idx].reset_index(drop=True),
        split_name="temporal",
    )

    logger.info(
        "Temporal split (cutoff=%s): train=%d (%.1f%% malware), test=%d (%.1f%% malware)",
        cutoff,
        result.train_size,
        result.y_train.mean() * 100,
        result.test_size,
        result.y_test.mean() * 100,
    )
    return result
