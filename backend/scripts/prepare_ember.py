#!/usr/bin/env python3
"""Vectorize raw EMBER 2018 JSONL feature files into fast-loading `.dat` files.

The upstream `ember` package's own vectorizer crashes on modern scikit-learn
(a bare string gets passed to `FeatureHasher.transform()` in
`SectionInfo.process_raw_features`, which newer scikit-learn rejects). This
script applies a compatibility patch (see `haruspex.data.ember`) and
vectorizes directly, writing standard EMBER-format memmap files:

    dataset/ember2018/X_test.dat  + y_test.dat   (200,000 samples, ~9 min)
    dataset/ember2018/X_train.dat + y_train.dat  (800,000 samples, ~35 min, --include-train)

Run this once; scripts/evaluate.py will then load the cached `.dat` files
directly (fast memmap read, no `ember`/`lief` dependency at evaluation time).

Usage:
    cd backend
    source .venv/bin/activate
    python scripts/prepare_ember.py                  # test split only
    python scripts/prepare_ember.py --include-train   # + train split
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from haruspex.data.ember import EMBER_DIR, vectorize_partition
from haruspex.utils.logging import get_logger

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Vectorize raw EMBER 2018 JSONL features into .dat files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default=None, help="Override the EMBER dataset directory.")
    parser.add_argument(
        "--include-train",
        action="store_true",
        help="Also vectorize the 800k-sample train split (slow, not needed for cross-dataset eval).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    target_dir = Path(args.data_dir) if args.data_dir else EMBER_DIR

    test_jsonl = target_dir / "test_features.jsonl"
    if not test_jsonl.exists():
        logger.error("Missing %s", test_jsonl)
        return 1

    vectorize_partition(
        [test_jsonl],
        target_dir / "X_test.dat",
        target_dir / "y_test.dat",
    )

    if args.include_train:
        train_shards = [target_dir / f"train_features_{i}.jsonl" for i in range(6)]
        missing = [p for p in train_shards if not p.exists()]
        if missing:
            logger.error("Missing train shard(s): %s", missing)
            return 1
        vectorize_partition(
            train_shards,
            target_dir / "X_train.dat",
            target_dir / "y_train.dat",
        )

    logger.info("Done. scripts/evaluate.py will now use the cached .dat files.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
