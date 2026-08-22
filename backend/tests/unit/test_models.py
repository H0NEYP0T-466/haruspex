"""Unit tests for model fit/predict/save/load round-trip."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from haruspex.models.base import MalwareDetector
from haruspex.models.xgboost_model import XGBoostDetector
from haruspex.models.lightgbm_model import LightGBMDetector


@pytest.fixture
def small_dataset():
    """Create a small synthetic dataset for model tests."""
    rng = np.random.RandomState(42)
    n_samples = 500
    n_features = 50

    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)

    # 80/20 split
    split = int(n_samples * 0.8)
    return {
        "X_train": X[:split],
        "y_train": y[:split],
        "X_test": X[split:],
        "y_test": y[split:],
    }


@pytest.mark.parametrize("ModelClass", [XGBoostDetector, LightGBMDetector])
class TestModelInterface:
    """Test that both models implement the MalwareDetector interface."""

    def test_is_malware_detector(self, ModelClass):
        """Model inherits from MalwareDetector."""
        model = ModelClass()
        assert isinstance(model, MalwareDetector)

    def test_not_fitted_initially(self, ModelClass):
        """Model is not fitted before training."""
        model = ModelClass()
        assert not model.is_fitted

    def test_fit_predict(self, ModelClass, small_dataset):
        """Model can fit and predict."""
        model = ModelClass()
        model.fit(
            small_dataset["X_train"],
            small_dataset["y_train"],
            small_dataset["X_test"],
            small_dataset["y_test"],
        )
        assert model.is_fitted

        preds = model.predict(small_dataset["X_test"])
        assert preds.shape == small_dataset["y_test"].shape
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba(self, ModelClass, small_dataset):
        """predict_proba returns values in [0, 1]."""
        model = ModelClass()
        model.fit(small_dataset["X_train"], small_dataset["y_train"])

        proba = model.predict_proba(small_dataset["X_test"])
        assert proba.shape == (small_dataset["X_test"].shape[0],)
        assert np.all(proba >= 0.0)
        assert np.all(proba <= 1.0)

    def test_feature_importance(self, ModelClass, small_dataset):
        """get_feature_importance returns correct shape."""
        model = ModelClass()
        model.fit(small_dataset["X_train"], small_dataset["y_train"])

        imp = model.get_feature_importance()
        assert imp.shape == (small_dataset["X_train"].shape[1],)

    def test_save_load_roundtrip(self, ModelClass, small_dataset):
        """Model can be saved and loaded with consistent predictions."""
        model = ModelClass()
        model.fit(small_dataset["X_train"], small_dataset["y_train"])

        # Predict before save
        pred_before = model.predict_proba(small_dataset["X_test"])

        # Save
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            save_path = f.name
        model.save(save_path)

        # Load into new instance
        model2 = ModelClass()
        model2.load(save_path)
        assert model2.is_fitted

        # Predict after load
        pred_after = model2.predict_proba(small_dataset["X_test"])

        # Predictions should be very close (may not be identical due to float precision)
        np.testing.assert_allclose(pred_before, pred_after, atol=1e-4)

        # Cleanup
        Path(save_path).unlink(missing_ok=True)

    def test_model_name(self, ModelClass):
        """model_name returns expected string."""
        model = ModelClass()
        assert model.model_name in ("xgboost", "lightgbm")

    def test_repr(self, ModelClass):
        """repr includes model name and fitted status."""
        model = ModelClass()
        r = repr(model)
        assert model.model_name in r
        assert "fitted" in r.lower() or "False" in r
