"""Unit tests for SHAP explainer."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from haruspex.models.xgboost_model import XGBoostDetector
from haruspex.xai.shap_explainer import SHAPExplainer, SHAPExplanation


@pytest.fixture(scope="module")
def trained_model_and_data():
    """Train a small XGBoost model for SHAP tests."""
    rng = np.random.RandomState(42)
    n_samples = 300
    n_features = 50

    X = rng.randn(n_samples, n_features).astype(np.float32)
    y = (X[:, 0] + X[:, 1] > 0).astype(np.int32)

    model = XGBoostDetector(n_estimators=20, max_depth=3)
    model.fit(X[:250], y[:250])

    feature_names = [f"feat_{i}" for i in range(n_features)]

    return model, X, y, feature_names


class TestSHAPExplainer:
    """Tests for the SHAP explainer."""

    def test_explain_sample_returns_explanation(self, trained_model_and_data):
        """explain_sample returns a SHAPExplanation."""
        model, X, _, feature_names = trained_model_and_data
        explainer = SHAPExplainer(model, feature_names=feature_names)

        exp = explainer.explain_sample(X[0])
        assert isinstance(exp, SHAPExplanation)

    def test_shap_values_shape(self, trained_model_and_data):
        """SHAP values have correct shape."""
        model, X, _, feature_names = trained_model_and_data
        explainer = SHAPExplainer(model, feature_names=feature_names)

        exp = explainer.explain_sample(X[0])
        assert exp.shap_values.shape == (X.shape[1],)

    def test_shap_values_nonzero(self, trained_model_and_data):
        """SHAP values contain non-zero entries."""
        model, X, _, feature_names = trained_model_and_data
        explainer = SHAPExplainer(model, feature_names=feature_names)

        exp = explainer.explain_sample(X[0])
        assert np.any(exp.shap_values != 0)

    def test_top_features(self, trained_model_and_data):
        """top_features returns correct number of features."""
        model, X, _, feature_names = trained_model_and_data
        explainer = SHAPExplainer(model, feature_names=feature_names)

        exp = explainer.explain_sample(X[0])
        top = exp.top_features(k=5)
        assert len(top) == 5
        assert all("feature_name" in f for f in top)
        assert all("shap_value" in f for f in top)

    def test_explain_dataset(self, trained_model_and_data):
        """explain_dataset returns correct shape."""
        model, X, _, feature_names = trained_model_and_data
        explainer = SHAPExplainer(model, feature_names=feature_names)

        shap_values = explainer.explain_dataset(X[:20], max_samples=10)
        assert shap_values.shape[0] <= 10
        assert shap_values.shape[1] == X.shape[1]

    def test_global_importance(self, trained_model_and_data):
        """get_global_importance returns sorted features."""
        model, X, _, feature_names = trained_model_and_data
        explainer = SHAPExplainer(model, feature_names=feature_names)

        shap_values = explainer.explain_dataset(X[:20], max_samples=10)
        importance = explainer.get_global_importance(shap_values, top_k=5)

        assert len(importance) == 5
        # Should be sorted by mean_abs_shap descending
        vals = [f["mean_abs_shap"] for f in importance]
        assert vals == sorted(vals, reverse=True)
