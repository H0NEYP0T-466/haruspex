"""SHAP explainability for HARUSPEX malware detection models.

Provides TreeSHAP (exact, fast) for tree-based models and KernelSHAP
as a fallback for any model type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from haruspex.config import get_feature_names
from haruspex.models.base import MalwareDetector
from haruspex.utils.logging import get_logger
from haruspex.utils.storage import save_plot

logger = get_logger(__name__)


@dataclass
class SHAPExplanation:
    """Container for SHAP explanation of a single sample."""

    shap_values: np.ndarray       # (n_features,) SHAP values
    base_value: float             # expected value (base rate)
    feature_values: np.ndarray    # (n_features,) actual feature values
    feature_names: list[str]      # feature name labels
    prediction: float             # model prediction probability

    def top_features(self, k: int = 20) -> list[dict]:
        """Return top-K features by absolute SHAP value."""
        abs_shap = np.abs(self.shap_values)
        top_idx = np.argsort(abs_shap)[::-1][:k]
        return [
            {
                "index": int(idx),
                "feature_name": self.feature_names[idx],
                "shap_value": float(self.shap_values[idx]),
                "feature_value": float(self.feature_values[idx]),
            }
            for idx in top_idx
        ]


class SHAPExplainer:
    """SHAP explanation generator for HARUSPEX models.

    Uses TreeExplainer for tree-based models (XGBoost, LightGBM) and
    KernelExplainer as a fallback for arbitrary models.
    """

    def __init__(
        self,
        model: MalwareDetector,
        background_data: Optional[np.ndarray] = None,
        feature_names: Optional[list[str]] = None,
    ) -> None:
        """Initialise the SHAP explainer.

        Args:
            model: Trained HARUSPEX model.
            background_data: Background dataset for KernelSHAP (sampled if large).
            feature_names: Feature name labels. Defaults to EMBER-compatible names.
        """
        self._model = model
        self._feature_names = feature_names or get_feature_names()
        self._background_data = background_data
        self._explainer: Any = None
        self._explainer_type: str = ""

        self._init_explainer()

    def _init_explainer(self) -> None:
        """Initialise the appropriate SHAP explainer."""
        underlying = self._model.get_underlying_model()
        model_name = self._model.model_name

        if "xgboost" in model_name or "lightgbm" in model_name:
            logger.info("Using TreeExplainer for %s", model_name)
            self._explainer = shap.TreeExplainer(underlying)
            self._explainer_type = "tree"
        else:
            logger.info("Using KernelExplainer (fallback) for %s", model_name)
            bg = self._background_data
            if bg is None:
                bg = np.zeros((20, len(self._feature_names)), dtype=np.float32)
            elif bg.shape[0] > 50:
                idx = np.random.RandomState(42).choice(bg.shape[0], 50, replace=False)
                bg = bg[idx]
            self._explainer = shap.KernelExplainer(
                self._model.predict_proba, bg
            )
            self._explainer_type = "kernel"

    def explain_sample(self, x_sample: np.ndarray) -> SHAPExplanation:
        """Generate SHAP / feature attribution explanation for a single sample.

        Args:
            x_sample: Feature vector (n_features,) or (1, n_features).

        Returns:
            SHAPExplanation with per-feature contributions.
        """
        x = x_sample.reshape(1, -1) if x_sample.ndim == 1 else x_sample

        # Fast-path for PyTorch Deep Learning models (Gradient x Input in 0.003s)
        if hasattr(self._model, "get_gradient_saliency"):
            sv = self._model.get_gradient_saliency(x[0])
            pred = float(self._model.predict_proba(x)[0])
            return SHAPExplanation(
                shap_values=sv,
                base_value=0.5,
                feature_values=x[0],
                feature_names=self._feature_names,
                prediction=pred,
            )

        shap_values = self._explainer.shap_values(x)

        # Handle multi-output (binary classification): take positive class
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # positive class, first sample
        elif shap_values.ndim == 3:
            sv = shap_values[0, :, 1]  # (samples, features, classes)
        else:
            sv = shap_values[0]

        # Base value
        base = getattr(self._explainer, "expected_value", 0.5)
        if isinstance(base, (list, np.ndarray)):
            base_val = float(base[1]) if len(base) > 1 else float(base[0])
        else:
            base_val = float(base)

        # Model prediction
        pred = float(self._model.predict_proba(x)[0])

        return SHAPExplanation(
            shap_values=sv,
            base_value=base_val,
            feature_values=x[0],
            feature_names=self._feature_names,
            prediction=pred,
        )

    def explain_dataset(
        self, X: np.ndarray, max_samples: int = 100
    ) -> np.ndarray:
        """Generate SHAP values for multiple samples.

        Args:
            X: Feature matrix (n_samples, n_features).
            max_samples: Maximum number of samples to explain.

        Returns:
            SHAP values matrix (n_samples, n_features).
        """
        if X.shape[0] > max_samples:
            idx = np.random.RandomState(42).choice(X.shape[0], max_samples, replace=False)
            X_sub = X[idx]
        else:
            X_sub = X

        logger.info("Computing SHAP values for %d samples...", X_sub.shape[0])
        shap_values = self._explainer.shap_values(X_sub)

        # Handle multi-output
        if isinstance(shap_values, list):
            return shap_values[1]  # positive class
        elif shap_values.ndim == 3:
            return shap_values[:, :, 1]
        return shap_values

    def plot_summary(
        self,
        shap_values: np.ndarray,
        X: Optional[np.ndarray] = None,
        save_path: Optional[str] = None,
        max_display: int = 20,
    ) -> Optional[plt.Figure]:
        """Generate and save a SHAP summary plot.

        Args:
            shap_values: SHAP values matrix (n_samples, n_features).
            X: Original feature matrix for colour encoding.
            save_path: File path to save the plot. If None, returns figure.
            max_display: Maximum number of features to display.

        Returns:
            Matplotlib figure if save_path is None.
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        # Bar summary (more robust than beeswarm for large feature sets)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs_shap)[::-1][:max_display]
        top_names = [self._feature_names[i] for i in top_idx]
        top_values = mean_abs_shap[top_idx]

        ax.barh(range(len(top_names)), top_values[::-1], color="#00d4ff", edgecolor="#003d4d")
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names[::-1], fontsize=9)
        ax.set_xlabel("Mean |SHAP value|", fontsize=11)
        ax.set_title("SHAP Feature Importance (Top 20)", fontsize=13, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        if save_path:
            save_plot(fig, save_path.replace(".png", "").split("/")[-1])
            return None

        return fig

    def get_global_importance(
        self, shap_values: np.ndarray, top_k: int = 20
    ) -> list[dict]:
        """Return global feature importance from SHAP values.

        Args:
            shap_values: SHAP values matrix (n_samples, n_features).
            top_k: Number of top features to return.

        Returns:
            List of dicts with feature_name and mean_abs_shap.
        """
        mean_abs = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs)[::-1][:top_k]
        return [
            {
                "index": int(idx),
                "feature_name": self._feature_names[idx],
                "mean_abs_shap": float(mean_abs[idx]),
            }
            for idx in top_idx
        ]
