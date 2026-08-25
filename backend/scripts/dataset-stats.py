#!/usr/bin/env python3
"""Compute exact dataset statistics and print hardcoded constant assignments.

Run from backend/:
    .venv/bin/python scripts/dataset-stats.py

Paste the output into haruspex/api/routes/explain.py to update the constants.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure backend/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np


def ember_stats() -> dict:
    """Compute EMBER stats directly from .dat memmap files."""
    ember_dir = Path(__file__).resolve().parent.parent / "dataset" / "ember2018"

    x_train = np.memmap(str(ember_dir / "X_train.dat"), dtype=np.float32, mode="r").reshape(-1, 2381)
    y_train = np.memmap(str(ember_dir / "y_train.dat"), dtype=np.float32, mode="r")
    x_test = np.memmap(str(ember_dir / "X_test.dat"), dtype=np.float32, mode="r").reshape(-1, 2381)
    y_test = np.memmap(str(ember_dir / "y_test.dat"), dtype=np.float32, mode="r")

    total = x_train.shape[0] + x_test.shape[0]
    benign = int((y_train == 0).sum()) + int((y_test == 0).sum())
    malware = int((y_train == 1).sum()) + int((y_test == 1).sum())
    unlabeled = int((y_train == -1).sum())

    return {
        "total": total,
        "features": x_test.shape[1],
        "benign": benign,
        "malware": malware,
        "unlabeled_train": unlabeled,
        "train_rows": x_train.shape[0],
        "test_rows": x_test.shape[0],
    }


def bodmas_stats() -> dict:
    """Compute BODMAS stats directly from .npz file."""
    data = np.load(
        Path(__file__).resolve().parent.parent / "dataset" / "BODMAS" / "bodmas.npz",
        allow_pickle=True,
    )
    X = data["X"]
    y = data["y"]
    return {
        "total": X.shape[0],
        "features": X.shape[1],
        "benign": int((y == 0).sum()),
        "malware": int((y == 1).sum()),
    }


if __name__ == "__main__":
    e = ember_stats()
    b = bodmas_stats()

    print("# EMBER 2018 — full dataset (train + test)")
    print(f"# Train: {e['train_rows']:,} rows ({e['unlabeled_train']:,} unlabeled)")
    print(f"# Test:  {e['test_rows']:,} rows")
    print(f"EMBER_TOTAL_SAMPLES = {e['total']:,}")
    print(f"EMBER_N_FEATURES = {e['features']}")
    print(f"EMBER_N_BENIGN = {e['benign']:,}")
    print(f"EMBER_N_MALWARE = {e['malware']:,}")
    print()
    print("# BODMAS")
    print(f"BODMAS_TOTAL_SAMPLES = {b['total']:,}")
    print(f"BODMAS_N_FEATURES = {b['features']}")
    print(f"BODMAS_N_BENIGN = {b['benign']:,}")
    print(f"BODMAS_N_MALWARE = {b['malware']:,}")