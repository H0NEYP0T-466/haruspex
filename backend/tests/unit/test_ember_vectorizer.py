"""Regression coverage for the official EMBER v2 vectorizer adapter."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from haruspex.config import NUM_FEATURES
from haruspex.forensics.ember_vectorizer import EmberV2PEVectorizer


def test_ember_v2_raw_record_matches_stored_test_vector():
    root = Path(__file__).resolve().parent.parent.parent / "dataset" / "ember2018"
    with (root / "test_features.jsonl").open(encoding="utf-8") as handle:
        raw_record = json.loads(next(handle))
    expected = np.memmap(root / "X_test.dat", dtype=np.float32, mode="r").reshape(-1, NUM_FEATURES)[0]
    actual = EmberV2PEVectorizer().process_raw_record(raw_record)
    assert np.array_equal(actual, expected)
