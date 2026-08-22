"""Unit tests for dataset splitting strategies."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from haruspex.data.bodmas import load_bodmas
from haruspex.data.splits import (
    SplitResult,
    family_held_out_split,
    random_split,
    temporal_split,
)


@pytest.fixture(scope="module")
def dataset():
    """Load dataset once for all tests."""
    ds = load_bodmas()
    return ds.X, ds.y, ds.metadata


class TestRandomSplit:
    """Tests for stratified random splitting."""

    def test_returns_split_result(self, dataset):
        X, y, meta = dataset
        result = random_split(X, y, meta)
        assert isinstance(result, SplitResult)
        assert result.split_name == "random"

    def test_correct_proportions(self, dataset):
        X, y, meta = dataset
        result = random_split(X, y, meta, test_size=0.2)
        expected_test = int(len(y) * 0.2)
        assert abs(result.test_size - expected_test) < 100  # Allow small variance

    def test_no_overlap(self, dataset):
        X, y, meta = dataset
        result = random_split(X, y, meta)
        train_set = set(result.train_indices)
        test_set = set(result.test_indices)
        assert len(train_set & test_set) == 0

    def test_covers_all_samples(self, dataset):
        X, y, meta = dataset
        result = random_split(X, y, meta)
        assert result.train_size + result.test_size == len(y)

    def test_stratified(self, dataset):
        """Train and test should have similar malware ratios."""
        X, y, meta = dataset
        result = random_split(X, y, meta)
        overall_ratio = y.mean()
        train_ratio = result.y_train.mean()
        test_ratio = result.y_test.mean()
        # Should be within 2% of overall ratio
        assert abs(train_ratio - overall_ratio) < 0.02
        assert abs(test_ratio - overall_ratio) < 0.02


class TestFamilyHeldOutSplit:
    """Tests for family-held-out splitting."""

    def test_held_out_families_in_test_only(self, dataset):
        """All samples from held-out families must be in test set only."""
        X, y, meta = dataset
        held_out = ["gepys", "benjamin"]
        result = family_held_out_split(X, y, meta, held_out_families=held_out)

        # No held-out family samples in training
        train_families = result.metadata_train["family"].unique()
        for fam in held_out:
            assert fam not in train_families, f"Family {fam} found in training set!"

        # Held-out samples exist in test set
        test_families = result.metadata_test["family"].unique()
        for fam in held_out:
            assert fam in test_families, f"Family {fam} not found in test set!"

    def test_no_leakage(self, dataset):
        """No sample appears in both train and test."""
        X, y, meta = dataset
        result = family_held_out_split(X, y, meta, held_out_families=["autoit"])
        train_set = set(result.train_indices)
        test_set = set(result.test_indices)
        assert len(train_set & test_set) == 0

    def test_invalid_family_raises(self, dataset):
        """Holding out nonexistent families raises ValueError."""
        X, y, meta = dataset
        with pytest.raises(ValueError):
            family_held_out_split(X, y, meta, held_out_families=["nonexistent_family_xyz"])


class TestTemporalSplit:
    """Tests for temporal splitting."""

    def test_cutoff_respected(self, dataset):
        """All training samples before cutoff, all test samples after."""
        X, y, meta = dataset
        cutoff = "2020-06-01"
        result = temporal_split(X, y, meta, cutoff_date=cutoff)

        cutoff_ts = pd.Timestamp(cutoff, tz="UTC")

        train_ts = pd.to_datetime(result.metadata_train["timestamp"], utc=True)
        test_ts = pd.to_datetime(result.metadata_test["timestamp"], utc=True)

        assert (train_ts < cutoff_ts).all(), "Training samples found after cutoff!"
        assert (test_ts >= cutoff_ts).all(), "Test samples found before cutoff!"

    def test_covers_all(self, dataset):
        X, y, meta = dataset
        result = temporal_split(X, y, meta)
        assert result.train_size + result.test_size == len(y)

    def test_no_overlap(self, dataset):
        X, y, meta = dataset
        result = temporal_split(X, y, meta)
        train_set = set(result.train_indices)
        test_set = set(result.test_indices)
        assert len(train_set & test_set) == 0
