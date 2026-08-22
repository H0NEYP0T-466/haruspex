"""EMBER dataset loader and vectorizer.

Loads the EMBER 2018 benchmark dataset from backend/dataset/ember2018/.
Supports both pre-vectorized memory-mapped files (X_test.dat, y_test.dat)
and on-the-fly vectorization from test_features.jsonl.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from haruspex.config import DATASET_DIR, get_feature_names
from haruspex.utils.logging import get_logger

logger = get_logger(__name__)

# Default EMBER directory
EMBER_DIR = DATASET_DIR.parent / "ember2018"


def is_ember_available(data_dir: Optional[str | Path] = None) -> bool:
    """Check whether the EMBER dataset files exist on disk."""
    target_dir = Path(data_dir) if data_dir else EMBER_DIR
    if not target_dir.exists():
        return False

    # Check for vectorized .dat files (either partition) or jsonl feature files
    has_dat_test = (target_dir / "X_test.dat").exists() and (target_dir / "y_test.dat").exists()
    has_dat_train = (target_dir / "X_train.dat").exists() and (target_dir / "y_train.dat").exists()
    has_jsonl = (target_dir / "test_features.jsonl").exists()

    return has_dat_test or has_dat_train or has_jsonl


def load_ember(
    data_dir: Optional[str | Path] = None,
    test_only: bool = True,
    max_samples: Optional[int] = None,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load the EMBER dataset test partition.

    Args:
        data_dir: Path to EMBER directory. Defaults to backend/dataset/ember2018/.
        test_only: If True, load test set only (recommended for cross-dataset evaluation).
        max_samples: If set, subsample to max_samples for fast evaluation.

    Returns:
        Tuple of (X, y, metadata DataFrame).

    Raises:
        FileNotFoundError: If EMBER dataset is not found.
    """
    target_dir = Path(data_dir) if data_dir else EMBER_DIR
    if not is_ember_available(target_dir):
        raise FileNotFoundError(
            f"EMBER dataset not found in {target_dir}. "
            "Ensure test_features.jsonl or X_test.dat exists in backend/dataset/ember2018/."
        )

    logger.info("Loading EMBER dataset from %s", target_dir)

    # Prefer pre-vectorized .dat files (no ember/lief dependency); fall back to jsonl.
    x_dat = target_dir / "X_test.dat"
    y_dat = target_dir / "y_test.dat"
    if not (x_dat.exists() and y_dat.exists()):
        x_dat = target_dir / "X_train.dat"
        y_dat = target_dir / "y_train.dat"

    if x_dat.exists() and y_dat.exists():
        partition = x_dat.stem.split("_")[1]
        logger.info("Reading vectorized EMBER %s set via numpy memmap...", partition)
        X_test = np.memmap(str(x_dat), dtype=np.float32, mode="r").reshape(-1, 2381)
        y_test = np.memmap(str(y_dat), dtype=np.float32, mode="r")
    else:
        logger.info("No .dat files found — parsing test_features.jsonl directly...")
        X_test, y_test = _load_from_jsonl(target_dir / "test_features.jsonl", max_samples=max_samples)

    # Remove unlabeled samples (label == -1 in EMBER)
    valid_mask = (y_test != -1) & (~np.isnan(y_test))
    X_test = np.array(X_test[valid_mask], dtype=np.float32)
    y_test = np.array(y_test[valid_mask], dtype=np.int32)

    # Subsample if requested
    if max_samples and X_test.shape[0] > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(X_test.shape[0], max_samples, replace=False)
        X_test = X_test[idx]
        y_test = y_test[idx]

    # Construct metadata DataFrame
    metadata = pd.DataFrame({
        "sha": [f"ember_{i}" for i in range(len(y_test))],
        "label": ["malware" if y == 1 else "benign" for y in y_test],
        "dataset": "EMBER_2018",
    })

    logger.info(
        "EMBER loaded: %d samples (%d benign, %d malware)",
        X_test.shape[0],
        int((y_test == 0).sum()),
        int((y_test == 1).sum()),
    )

    return X_test, y_test, metadata


def _load_from_jsonl(jsonl_path: Path, max_samples: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """Fallback loader directly from jsonl using EMBER PEFeatureExtractor vectorizer."""
    import json

    logger.info("Parsing features from %s", jsonl_path)
    X_list = []
    y_list = []

    try:
        import ember.features
        extractor = ember.features.PEFeatureExtractor(feature_version=2)
    except Exception as e:
        logger.warning("Could not import ember.features: %s", e)
        extractor = None

    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if max_samples and len(y_list) >= max_samples:
                break
            entry = json.loads(line)
            lbl = entry.get("label", -1)
            if lbl == -1:
                continue

            if extractor is not None:
                vec = extractor.process_raw_features(entry)
            else:
                # Reconstruct vector from JSON keys
                vec = np.zeros(2381, dtype=np.float32)
                hist = entry.get("histogram", [])
                if hist:
                    vec[0:len(hist)] = hist
                ent = entry.get("byteentropy", [])
                if ent:
                    vec[256:256+len(ent)] = ent

            X_list.append(vec)
            y_list.append(lbl)

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)
