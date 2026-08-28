#!/usr/bin/env python3
"""Verify EMBER v2 vectorization against this project's stored EMBER memmap."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from haruspex.config import NUM_FEATURES
from haruspex.forensics.ember_vectorizer import EmberV2PEVectorizer, iter_jsonl_records


def main() -> None:
    root = Path(__file__).resolve().parent.parent / "dataset" / "ember2018"
    vectorizer = EmberV2PEVectorizer()
    expected = np.memmap(root / "X_test.dat", dtype=np.float32, mode="r").reshape(-1, NUM_FEATURES)
    checked = 0
    for index, record in enumerate(iter_jsonl_records(root / "test_features.jsonl")):
        actual = vectorizer.process_raw_record(record)
        if not np.array_equal(actual, expected[index]):
            raise SystemExit(f"Mismatch at row {index}: max diff={np.abs(actual - expected[index]).max()}")
        checked += 1
        if checked == 25:
            break
    print(f"Verified {checked} EMBER v2 raw records exactly against X_test.dat")


if __name__ == "__main__":
    main()
