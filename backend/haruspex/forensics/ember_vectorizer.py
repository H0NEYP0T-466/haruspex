"""Official EMBER v2 raw-PE vectorization with explicit verification support.

This adapter uses the same ``ember.features.PEFeatureExtractor`` v2 API that
created this repository's EMBER memmaps.  It is deliberately separate from the
legacy custom extractor: callers must opt into a dataset/vectorizer contract.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Iterator

import numpy as np

from haruspex.config import NUM_FEATURES
from haruspex.data.ember import _get_extractor


class EmberV2PEVectorizer:
    """Produce an EMBER v2, 2,381-feature vector from PE bytes."""

    feature_version = 2
    feature_count = NUM_FEATURES

    def __init__(self) -> None:
        self._extractor = _get_extractor(feature_version=self.feature_version)

    def extract(self, pe_bytes: bytes) -> np.ndarray:
        """Vectorize raw PE bytes through the official EMBER v2 extractor."""
        # ember==0.1.0 predates LIEF 1.x and references removed exception names
        # in ``feature_vector``.  Keep its feature definitions/hashing intact,
        # but use modern LIEF's parser for the raw-record creation step.
        # EMBER 0.1.0 also uses the NumPy alias removed in NumPy 2.
        if not hasattr(np, "int"):
            np.int = int  # type: ignore[attr-defined]
        try:
            import lief
            binary = lief.PE.parse(list(bytes(pe_bytes)))
        except Exception:
            binary = None
        raw_record = {"sha256": hashlib.sha256(pe_bytes).hexdigest()}
        raw_record.update({
            feature.name: feature.raw_features(pe_bytes, binary)
            for feature in self._extractor.features
        })
        vector = self._extractor.process_raw_features(raw_record)
        vector = np.asarray(vector, dtype=np.float32)
        if vector.shape != (NUM_FEATURES,):
            raise ValueError(f"Expected {NUM_FEATURES} EMBER v2 features, got {vector.shape}")
        if not np.isfinite(vector).all():
            raise ValueError("EMBER v2 extractor returned non-finite features")
        return vector

    def process_raw_record(self, raw_record: dict) -> np.ndarray:
        """Vectorize an EMBER JSONL raw-feature record for regression tests."""
        return np.asarray(self._extractor.process_raw_features(raw_record), dtype=np.float32)


def iter_jsonl_records(path: str | Path) -> Iterator[dict]:
    """Yield EMBER raw records without loading a full JSONL shard into memory."""
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)
