"""EMBER dataset loader and vectorizer.

Loads the EMBER 2018 benchmark dataset from backend/dataset/ember2018/ for
cross-dataset evaluation of HARUSPEX models (which are trained only on
BODMAS, so any EMBER partition is a legitimate, unseen holdout).

Prefers pre-vectorized memmap files (``X_test.dat``/``y_test.dat`` or
``X_train.dat``/``y_train.dat``) since reading them has no dependency on the
version-fragile ``ember``/``lief`` feature extractor. Use
``scripts/prepare_ember.py`` to (re)generate those files from the raw
``*.jsonl`` feature files.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from haruspex.config import DATASET_DIR, NUM_FEATURES
from haruspex.utils.logging import get_logger

logger = get_logger(__name__)

# Default EMBER directory
EMBER_DIR = DATASET_DIR.parent / "ember2018"

# Partitions checked in priority order (test preferred over train).
_DAT_PARTITIONS = ("test", "train")


def is_ember_available(data_dir: Optional[str | Path] = None) -> bool:
    """Check whether any usable EMBER dataset files exist on disk."""
    target_dir = Path(data_dir) if data_dir else EMBER_DIR
    if not target_dir.exists():
        return False

    has_dat = any(
        (target_dir / f"X_{p}.dat").exists() and (target_dir / f"y_{p}.dat").exists()
        for p in _DAT_PARTITIONS
    )
    has_jsonl = (target_dir / "test_features.jsonl").exists()
    return has_dat or has_jsonl


def _resolve_dat_partition(target_dir: Path) -> tuple[Optional[Path], Optional[Path], str]:
    """Return (X path, y path, partition name) for the best available `.dat` pair.

    A `.dat` pair is only considered usable if it isn't an empty/zeroed
    placeholder — the upstream `ember` package pre-allocates these files
    with `mode="w+"` *before* vectorizing, so a crash partway through
    vectorization leaves a file that exists but contains no real data.
    """
    for partition in _DAT_PARTITIONS:
        x_dat = target_dir / f"X_{partition}.dat"
        y_dat = target_dir / f"y_{partition}.dat"
        if not (x_dat.exists() and y_dat.exists()):
            continue
        y = np.memmap(str(y_dat), dtype=np.float32, mode="r")
        if y.size == 0 or not np.any(y != 0):
            logger.warning(
                "%s looks like an empty/zeroed placeholder (incomplete vectorization) — ignoring.",
                y_dat.name,
            )
            continue
        return x_dat, y_dat, partition
    return None, None, ""


def _load_memmap(x_dat: Path, y_dat: Path) -> tuple[np.memmap, np.memmap]:
    """Memory-map pre-vectorized EMBER feature/label files (zero-copy, lazy read)."""
    logger.info("Reading vectorized EMBER features via numpy memmap: %s", x_dat.name)
    X = np.memmap(str(x_dat), dtype=np.float32, mode="r").reshape(-1, NUM_FEATURES)
    y = np.memmap(str(y_dat), dtype=np.float32, mode="r")
    return X, y


def load_ember(
    data_dir: Optional[str | Path] = None,
    test_only: bool = True,
    max_samples: Optional[int] = None,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load an EMBER 2018 partition for cross-dataset evaluation.

    Args:
        data_dir: Path to EMBER directory. Defaults to backend/dataset/ember2018/.
        test_only: If True (default), prefer the official test partition
            (X_test.dat/y_test.dat). If unavailable, transparently falls back
            to the train partition (X_train.dat/y_train.dat) — this remains a
            fair holdout since HARUSPEX models are only ever trained on BODMAS.
        max_samples: If set, subsample to at most this many labeled rows.
            Subsampling happens on the label array *before* the (potentially
            multi-GB) feature memmap is materialized, so this stays cheap.
        seed: RNG seed used for subsampling.

    Returns:
        Tuple of (X, y, metadata DataFrame).

    Raises:
        FileNotFoundError: If no usable EMBER files are found.
    """
    target_dir = Path(data_dir) if data_dir else EMBER_DIR
    if not is_ember_available(target_dir):
        raise FileNotFoundError(
            f"EMBER dataset not found in {target_dir}. Expected one of: "
            "X_test.dat+y_test.dat, X_train.dat+y_train.dat, or test_features.jsonl."
        )

    logger.info("Loading EMBER dataset from %s", target_dir)

    x_dat, y_dat, partition = _resolve_dat_partition(target_dir)
    if x_dat is not None and y_dat is not None:
        X, y = _load_memmap(x_dat, y_dat)
        if partition != "test":
            logger.warning(
                "X_test.dat not found — using the EMBER '%s' partition as the "
                "cross-dataset holdout instead. This is still unseen data for "
                "HARUSPEX models, which are trained only on BODMAS.",
                partition,
            )
    else:
        partition = "test"
        logger.info(
            "No usable .dat files found — parsing test_features.jsonl directly "
            "(slow; run scripts/prepare_ember.py once to cache vectorized features)..."
        )
        X, y = _load_from_jsonl(target_dir / "test_features.jsonl", max_samples=max_samples)

    # Drop unlabeled rows (label == -1 in EMBER) using the cheap 1-D label
    # array first, so we never touch the multi-GB feature matrix unnecessarily.
    valid_idx = np.where((y != -1) & (~np.isnan(y)))[0]
    if valid_idx.size == 0:
        raise ValueError(f"No labeled samples found in EMBER '{partition}' partition.")

    # Subsample the *indices* before materializing any feature rows.
    if max_samples and valid_idx.size > max_samples:
        rng = np.random.RandomState(seed)
        valid_idx = rng.choice(valid_idx, size=max_samples, replace=False)
        valid_idx.sort()

    X_out = np.asarray(X[valid_idx], dtype=np.float32)
    y_out = np.asarray(y[valid_idx], dtype=np.int32)

    metadata = pd.DataFrame({
        "sha": [f"ember_{partition}_{i}" for i in valid_idx],
        "label": np.where(y_out == 1, "malware", "benign"),
        "dataset": "EMBER_2018",
        "partition": partition,
    })

    logger.info(
        "EMBER loaded (%s partition): %d samples (%d benign, %d malware)",
        partition,
        X_out.shape[0],
        int((y_out == 0).sum()),
        int((y_out == 1).sum()),
    )

    return X_out, y_out, metadata


# ---------------------------------------------------------------------------
# ember/scikit-learn compatibility shim
# ---------------------------------------------------------------------------
#
# ember.features.SectionInfo.process_raw_features() passes the entry-point
# section name to FeatureHasher.transform() as a bare string:
#
#       FeatureHasher(50, input_type="string").transform([raw_obj['entry']])
#
# `input_type="string"` expects each *sample* to be an iterable of tokens, so
# the argument should be `[[raw_obj['entry']]]` (one sample, one token) —
# not `[raw_obj['entry']]` (which IS a single string). Old scikit-learn
# silently iterated over the string's characters as bogus tokens; ember on
# scikit-learn >= 1.2 raises:
#
#       ValueError: Samples can not be a single string. The input must be
#       an iterable over iterables of strings.
#
# This has nothing to do with the `lief` version warning ember also prints;
# it reproduces regardless of lief version. We patch it here (in our own
# code, not in site-packages) so it survives a fresh `pip install`.
# ---------------------------------------------------------------------------

def _patch_ember_known_bugs() -> None:
    """Monkey-patch ember's SectionInfo hashing bug. Idempotent."""
    import ember.features as ember_features

    if getattr(ember_features.SectionInfo, "_haruspex_patched", False):
        return

    from sklearn.feature_extraction import FeatureHasher

    def process_raw_features(self, raw_obj):
        sections = raw_obj["sections"]
        general = [
            len(sections),
            sum(1 for s in sections if s["size"] == 0),
            sum(1 for s in sections if s["name"] == ""),
            sum(1 for s in sections if "MEM_READ" in s["props"] and "MEM_EXECUTE" in s["props"]),
            sum(1 for s in sections if "MEM_WRITE" in s["props"]),
        ]
        section_sizes = [(s["name"], s["size"]) for s in sections]
        section_sizes_hashed = FeatureHasher(50, input_type="pair").transform([section_sizes]).toarray()[0]
        section_entropy = [(s["name"], s["entropy"]) for s in sections]
        section_entropy_hashed = FeatureHasher(50, input_type="pair").transform([section_entropy]).toarray()[0]
        section_vsize = [(s["name"], s["vsize"]) for s in sections]
        section_vsize_hashed = FeatureHasher(50, input_type="pair").transform([section_vsize]).toarray()[0]
        # Fixed: [[str]] (one sample, one token) instead of [str] (a bare string).
        entry_name_hashed = FeatureHasher(50, input_type="string").transform([[raw_obj["entry"]]]).toarray()[0]
        characteristics = [p for s in sections for p in s["props"] if s["name"] == raw_obj["entry"]]
        characteristics_hashed = FeatureHasher(50, input_type="string").transform([characteristics]).toarray()[0]

        return np.hstack([
            general,
            section_sizes_hashed,
            section_entropy_hashed,
            section_vsize_hashed,
            entry_name_hashed,
            characteristics_hashed,
        ]).astype(np.float32)

    ember_features.SectionInfo.process_raw_features = process_raw_features
    ember_features.SectionInfo._haruspex_patched = True
    logger.info("Applied ember/scikit-learn compatibility patch to SectionInfo.")


def _get_extractor(feature_version: int = 2):
    """Import ember, apply compatibility patches, and return a ready extractor."""
    try:
        import ember.features as ember_features
    except Exception as e:
        raise RuntimeError(
            "Vectorizing EMBER raw features requires the 'ember' package "
            "(pip install ember) plus a compatible 'lief' install."
        ) from e

    _patch_ember_known_bugs()
    return ember_features.PEFeatureExtractor(feature_version=feature_version, print_feature_warning=False)


def vectorize_partition(
    jsonl_paths: list[Path],
    x_dat_path: Path,
    y_dat_path: Path,
    feature_version: int = 2,
    log_every: int = 20_000,
) -> int:
    """Vectorize raw EMBER JSONL feature files into memmapped `.dat` files.

    Writes directly into pre-allocated float32 memmaps in the same on-disk
    format the upstream `ember` package produces, so the result can later be
    read back with a plain `np.memmap` — no `ember`/`lief` dependency needed
    at read time. Rows that fail to vectorize are zero-filled with label -1
    (dropped automatically by `load_ember`).

    Args:
        jsonl_paths: Raw feature JSONL files to concatenate, in order.
        x_dat_path: Output path for the feature matrix (float32, N x NUM_FEATURES).
        y_dat_path: Output path for the label vector (float32, N).
        feature_version: EMBER feature schema version.
        log_every: How often (in rows) to log progress.

    Returns:
        Number of rows written.
    """
    extractor = _get_extractor(feature_version)

    nrows = sum(1 for p in jsonl_paths for _ in open(p))
    logger.info("Vectorizing %d raw samples from %d file(s) -> %s", nrows, len(jsonl_paths), x_dat_path.name)

    X = np.memmap(str(x_dat_path), dtype=np.float32, mode="w+", shape=(nrows, extractor.dim))
    y = np.memmap(str(y_dat_path), dtype=np.float32, mode="w+", shape=(nrows,))

    n_errors = 0
    row = 0
    for jsonl_path in jsonl_paths:
        with open(jsonl_path) as f:
            for line in f:
                raw = json.loads(line)
                try:
                    X[row] = extractor.process_raw_features(raw)
                    y[row] = raw.get("label", -1)
                except Exception as e:
                    n_errors += 1
                    if n_errors <= 5:
                        logger.warning("Row %d failed vectorization (%s) — marking unlabeled.", row, e)
                    X[row] = 0.0
                    y[row] = -1
                row += 1
                if row % log_every == 0:
                    logger.info("  ... %d/%d vectorized", row, nrows)

    X.flush()
    y.flush()
    if n_errors:
        logger.warning("Vectorization finished with %d/%d row errors (marked label=-1).", n_errors, nrows)
    else:
        logger.info("Vectorization complete: %d rows -> %s, %s", nrows, x_dat_path.name, y_dat_path.name)
    return nrows


def _load_from_jsonl(
    jsonl_path: Path, max_samples: Optional[int] = None
) -> tuple[np.ndarray, np.ndarray]:
    """In-memory fallback: vectorizes raw EMBER JSON features without caching to disk.

    Only used when no usable pre-vectorized `.dat` files exist. Prefer
    `scripts/prepare_ember.py` (which uses `vectorize_partition`) for
    anything beyond a quick one-off/subsampled check, since this re-parses
    and re-vectorizes from scratch on every call.
    """
    extractor = _get_extractor()

    logger.info("Parsing and vectorizing features from %s", jsonl_path)
    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    n_errors = 0

    with open(jsonl_path) as f:
        for line in f:
            if max_samples and len(y_list) >= max_samples:
                break
            entry = json.loads(line)
            label = entry.get("label", -1)
            if label == -1:
                continue
            try:
                vec = extractor.process_raw_features(entry)
            except Exception as e:
                n_errors += 1
                if n_errors <= 3:
                    logger.warning("Skipping sample that failed vectorization: %s", e)
                continue
            X_list.append(vec)
            y_list.append(label)

    if n_errors:
        logger.warning("Skipped %d samples that failed vectorization.", n_errors)
    if not y_list:
        raise RuntimeError(
            f"No samples could be vectorized from {jsonl_path}. The 'ember' "
            "feature extractor may be incompatible with the installed "
            "lief/scikit-learn versions."
        )

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int32)
