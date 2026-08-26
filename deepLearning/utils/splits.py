"""Evaluation split functions for HARUSPEX Deep Learning pipelines.

Implements the 4 standard evaluation protocols:
  1. Random Split: Stratified 80/20 train/test split.
  2. Family-Held-Out Split: Holds out target malware families (gepys, benjamin, musecador, autoit).
  3. Temporal Split: Cutoff before/after cutoff date (2020-06-01).
  4. Cross-Dataset Split: Primary dataset (e.g. BODMAS) vs Generalization dataset (e.g. EMBER).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from deepLearning.utils.config import (
    HELD_OUT_FAMILIES,
    RANDOM_SEED,
    TEMPORAL_CUTOFF,
)

logger = logging.getLogger("haruspex.deepLearning.splits")


@dataclass
class SplitData:
    """Container for split results."""

    split_name: str
    X_train: np.ndarray
    y_train: np.ndarray
    family_train: np.ndarray
    cat_train: np.ndarray

    X_test: np.ndarray
    y_test: np.ndarray
    family_test: np.ndarray
    cat_test: np.ndarray

    metadata_train: pd.DataFrame
    metadata_test: pd.DataFrame


def random_split(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    family_ids: np.ndarray,
    category_ids: np.ndarray,
    test_size: float = 0.2,
    seed: int = RANDOM_SEED,
) -> SplitData:
    """Perform a stratified random train/test split."""
    indices = np.arange(len(y))
    idx_train, idx_test = train_test_split(
        indices, test_size=test_size, stratify=y, random_state=seed
    )

    logger.info(
        "Random Split (seed=%d): %d train, %d test",
        seed, len(idx_train), len(idx_test)
    )

    return SplitData(
        split_name="random_split",
        X_train=X[idx_train],
        y_train=y[idx_train],
        family_train=family_ids[idx_train],
        cat_train=category_ids[idx_train],
        X_test=X[idx_test],
        y_test=y[idx_test],
        family_test=family_ids[idx_test],
        cat_test=category_ids[idx_test],
        metadata_train=metadata.iloc[idx_train].reset_index(drop=True),
        metadata_test=metadata.iloc[idx_test].reset_index(drop=True),
    )


def family_held_out_split(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    family_ids: np.ndarray,
    category_ids: np.ndarray,
    held_out_families: Optional[Sequence[str]] = None,
) -> SplitData:
    """Perform family-held-out split where target families are held out for testing."""
    families_to_hold = list(held_out_families or HELD_OUT_FAMILIES)

    test_mask = metadata["family"].isin(families_to_hold).values
    train_mask = ~test_mask

    idx_train = np.where(train_mask)[0]
    idx_test = np.where(test_mask)[0]

    logger.info(
        "Family-Held-Out Split (%s): %d train, %d test (all malware)",
        ", ".join(families_to_hold), len(idx_train), len(idx_test)
    )

    return SplitData(
        split_name="family_held_out",
        X_train=X[idx_train],
        y_train=y[idx_train],
        family_train=family_ids[idx_train],
        cat_train=category_ids[idx_train],
        X_test=X[idx_test],
        y_test=y[idx_test],
        family_test=family_ids[idx_test],
        cat_test=category_ids[idx_test],
        metadata_train=metadata.iloc[idx_train].reset_index(drop=True),
        metadata_test=metadata.iloc[idx_test].reset_index(drop=True),
    )


def temporal_split(
    X: np.ndarray,
    y: np.ndarray,
    metadata: pd.DataFrame,
    family_ids: np.ndarray,
    category_ids: np.ndarray,
    cutoff_date: str = TEMPORAL_CUTOFF,
) -> SplitData:
    """Perform temporal split (train < cutoff_date, test >= cutoff_date)."""
    ts = pd.to_datetime(metadata["timestamp"], format="mixed", utc=True)
    cutoff = pd.Timestamp(cutoff_date, tz="UTC")

    train_mask = (ts < cutoff).values
    test_mask = (ts >= cutoff).values

    idx_train = np.where(train_mask)[0]
    idx_test = np.where(test_mask)[0]

    logger.info(
        "Temporal Split (cutoff=%s): %d train, %d test",
        cutoff_date, len(idx_train), len(idx_test)
    )

    return SplitData(
        split_name="temporal_split",
        X_train=X[idx_train],
        y_train=y[idx_train],
        family_train=family_ids[idx_train],
        cat_train=category_ids[idx_train],
        X_test=X[idx_test],
        y_test=y[idx_test],
        family_test=family_ids[idx_test],
        cat_test=category_ids[idx_test],
        metadata_train=metadata.iloc[idx_train].reset_index(drop=True),
        metadata_test=metadata.iloc[idx_test].reset_index(drop=True),
    )
