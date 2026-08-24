#!/usr/bin/env python3
"""Prepare EMBER 2018 — vectorize raw JSONL feature files into .dat memmaps.

Run this ONCE before train_ember.py. It converts the raw EMBER JSONL files
(train_features_*.jsonl) into fast numpy memmap files (X_train.dat, y_train.dat)
that can be loaded instantly for training.

Usage:
    cd backend
    source venv/bin/activate       # or .venv
    python phase2/prepare_ember.py

The X_test.dat / y_test.dat files should already exist (shipped with EMBER 2018).
This script only generates the train partition memmaps.

Expected time:
    ~30-60 min on i7-6500U (800K samples)
    ~10-15 min on Kaggle / modern CPU
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Add backend/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from haruspex.config import BACKEND_DIR
from haruspex.data.ember import vectorize_partition, is_ember_available
from haruspex.utils.logging import get_logger

logger = get_logger(__name__)

EMBER_DIR = BACKEND_DIR / "dataset" / "ember2018"


def main() -> None:
    """Vectorize EMBER train partition JSONL → .dat memmaps."""
    print("=" * 72)
    print("  EMBER 2018 — Vectorize Train Partition")
    print("=" * 72)

    if not EMBER_DIR.exists():
        print(f"\n  ERROR: EMBER directory not found: {EMBER_DIR}")
        print("  Download EMBER 2018 and place files in backend/dataset/ember2018/")
        sys.exit(1)

    # Collect train JSONL files (train_features_0.jsonl .. train_features_5.jsonl)
    train_jsonls = sorted(EMBER_DIR.glob("train_features_*.jsonl"))
    if not train_jsonls:
        print(f"\n  ERROR: No train_features_*.jsonl files found in {EMBER_DIR}")
        sys.exit(1)

    print(f"\n  Found {len(train_jsonls)} train JSONL files:")
    for p in train_jsonls:
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"    {p.name} ({size_mb:.0f} MB)")

    x_train_dat = EMBER_DIR / "X_train.dat"
    y_train_dat = EMBER_DIR / "y_train.dat"

    # Check if already vectorized
    if x_train_dat.exists() and y_train_dat.exists():
        import numpy as np

        y = np.memmap(str(y_train_dat), dtype=np.float32, mode="r")
        n_labeled = int(np.sum((y != -1) & (~np.isnan(y))))
        if n_labeled > 0:
            print(f"\n  X_train.dat and y_train.dat already exist ({n_labeled} labeled samples).")
            print("  Delete them manually to re-vectorize.")
            return

    # Check X_test.dat exists (should come with EMBER download)
    x_test_dat = EMBER_DIR / "X_test.dat"
    y_test_dat = EMBER_DIR / "y_test.dat"
    if x_test_dat.exists() and y_test_dat.exists():
        print(f"\n  X_test.dat already exists (good — shipped with EMBER 2018)")
    else:
        # Also vectorize test if missing
        test_jsonl = EMBER_DIR / "test_features.jsonl"
        if test_jsonl.exists():
            print(f"\n  X_test.dat not found — will also vectorize test partition")
            t0 = time.time()
            vectorize_partition([test_jsonl], x_test_dat, y_test_dat)
            print(f"  Test vectorization done in {time.time() - t0:.1f}s")

    # Vectorize train partition
    print(f"\n  Starting train vectorization ({len(train_jsonls)} files)...")
    print("  This will take 30-60 minutes on a laptop CPU.")
    print()

    t0 = time.time()
    nrows = vectorize_partition(train_jsonls, x_train_dat, y_train_dat, log_every=50_000)
    elapsed = time.time() - t0

    print(f"\n  Done! Vectorized {nrows} samples in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Output: {x_train_dat} ({x_train_dat.stat().st_size / 1e9:.2f} GB)")
    print(f"          {y_train_dat} ({y_train_dat.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
