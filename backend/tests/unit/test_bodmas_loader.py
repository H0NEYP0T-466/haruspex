"""Unit tests for the BODMAS dataset loader."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from haruspex.data.bodmas import (
    BodmasDataset,
    get_families,
    get_family_sample_counts,
    get_timestamps,
    is_malicious,
    load_bodmas,
)


@pytest.fixture(scope="module")
def dataset() -> BodmasDataset:
    """Load the BODMAS dataset once for all tests in this module."""
    return load_bodmas()


class TestBodmasLoader:
    """Tests for the BODMAS dataset loader."""

    def test_load_returns_dataset(self, dataset: BodmasDataset):
        """Loader returns a BodmasDataset instance."""
        assert isinstance(dataset, BodmasDataset)

    def test_feature_matrix_shape(self, dataset: BodmasDataset):
        """Feature matrix has expected dimensions (134435 × 2381)."""
        assert dataset.X.shape == (134435, 2381)
        assert dataset.X.dtype == np.float32

    def test_label_shape(self, dataset: BodmasDataset):
        """Label array has correct length."""
        assert dataset.y.shape == (134435,)

    def test_label_values(self, dataset: BodmasDataset):
        """Labels are binary (0 or 1)."""
        unique = np.unique(dataset.y)
        assert set(unique) == {0, 1}

    def test_label_distribution(self, dataset: BodmasDataset):
        """Label distribution matches expected counts."""
        assert dataset.n_benign == 77142
        assert dataset.n_malware == 57293

    def test_metadata_alignment(self, dataset: BodmasDataset):
        """Metadata row count matches feature matrix."""
        assert len(dataset.metadata) == dataset.X.shape[0]

    def test_metadata_columns(self, dataset: BodmasDataset):
        """Metadata has required columns."""
        required = {"sha", "timestamp", "family", "category"}
        assert required.issubset(set(dataset.metadata.columns))

    def test_feature_names_count(self, dataset: BodmasDataset):
        """Feature names list matches feature count."""
        assert len(dataset.feature_names) == 2381

    def test_no_nan_in_features(self, dataset: BodmasDataset):
        """No NaN values in the feature matrix."""
        assert not np.isnan(dataset.X).any()

    def test_benign_family_is_empty(self, dataset: BodmasDataset):
        """Benign samples have empty string as family."""
        benign_mask = dataset.y == 0
        benign_families = dataset.metadata.loc[benign_mask, "family"]
        assert (benign_families == "").all()


class TestHelperFunctions:
    """Tests for BODMAS helper functions."""

    def test_get_families_returns_sorted_list(self, dataset: BodmasDataset):
        """get_families returns a sorted list of unique family names."""
        families = get_families(dataset.metadata)
        assert isinstance(families, list)
        assert len(families) > 0
        assert families == sorted(families)

    def test_get_timestamps_returns_series(self, dataset: BodmasDataset):
        """get_timestamps returns a datetime Series."""
        ts = get_timestamps(dataset.metadata)
        assert isinstance(ts, pd.Series)
        assert pd.api.types.is_datetime64_any_dtype(ts)

    def test_is_malicious(self, dataset: BodmasDataset):
        """is_malicious correctly identifies malware/benign samples."""
        # Find a known malware sample
        malware_idx = np.where(dataset.y == 1)[0][0]
        benign_idx = np.where(dataset.y == 0)[0][0]

        assert is_malicious(dataset.y, malware_idx) is True
        assert is_malicious(dataset.y, benign_idx) is False

    def test_family_sample_counts(self, dataset: BodmasDataset):
        """get_family_sample_counts returns sorted counts."""
        counts = get_family_sample_counts(dataset.metadata)
        assert isinstance(counts, pd.Series)
        assert len(counts) > 0
        # Should be sorted descending
        assert counts.iloc[0] >= counts.iloc[-1]
