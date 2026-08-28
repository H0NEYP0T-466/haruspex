"""LIME explainability for HARUSPEX malware detection models.

Generates local interpretable explanations for individual predictions
using LIME (Local Interpretable Model-agnostic Explanations).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

from haruspex.config import get_feature_names
from haruspex.models.base import MalwareDetector
from haruspex.utils.logging import get_logger
from haruspex.utils.storage import save_plot

logger = get_logger(__name__)


@dataclass
class LIMEExplanation:
    """Container for a LIME explanation of a single sample."""

    feature_contributions: list[dict]  # [{feature_name, weight, value}]
    prediction: float                  # predicted probability
    intercept: float                   # LIME model intercept
    score: float                       # LIME local model R² score

    def top_features(self, k: int = 10) -> list[dict]:
        """Return top-K features by absolute weight."""
        sorted_feats = sorted(
            self.feature_contributions,
            key=lambda x: abs(x["weight"]),
            reverse=True,
        )
        return sorted_feats[:k]


class LIMEExplainer:
    """LIME explanation generator for HARUSPEX models.

    Uses :class:`lime.lime_tabular.LimeTabularExplainer` to generate
    local linear approximations of the model's decision boundary.
    """

    def __init__(
        self,
        training_data: np.ndarray,
        feature_names: Optional[list[str]] = None,
        class_names: Optional[list[str]] = None,
    ) -> None:
        """Initialise the LIME explainer.

        Args:
            training_data: Training feature matrix (for computing statistics).
            feature_names: Feature name labels. Defaults to EMBER-compatible names.
            class_names: Class labels. Defaults to ["benign", "malware"].
        """
        self._feature_names = feature_names or get_feature_names()
        self._class_names = class_names or ["benign", "malware"]

        self._explainer = LimeTabularExplainer(
            training_data=training_data,
            feature_names=self._feature_names,
            class_names=self._class_names,
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )
        logger.info("LIME explainer initialized with %d training samples", training_data.shape[0])

    def explain_sample(
        self,
        model: MalwareDetector,
        x_sample: np.ndarray,
        num_features: int = 10,
        num_samples: int = 1000,
    ) -> LIMEExplanation:
        """Generate LIME explanation for a single sample.

        Args:
            model: Trained HARUSPEX model.
            x_sample: Feature vector (n_features,).
            num_features: Number of features to include in explanation.
            num_samples: Number of perturbed samples for LIME.

        Returns:
            LIMEExplanation with feature contributions.
        """
        x = x_sample.ravel()

        # LIME expects a predict_proba function
        def predict_fn(X: np.ndarray) -> np.ndarray:
            proba = model.predict_proba(X)
            return np.column_stack([1 - proba, proba])

        explanation = self._explainer.explain_instance(
            x,
            predict_fn,
            num_features=num_features,
            num_samples=num_samples,
            labels=(1,),
        )

        # Extract contributions for positive class (malware = class 1)
        avail = explanation.available_labels()
        label = 1 if 1 in avail else (avail[0] if avail else 0)
        feature_list = explanation.as_list(label=label)
        local_pred = explanation.local_pred[0] if hasattr(explanation, "local_pred") else 0.0
        score = explanation.score if hasattr(explanation, "score") else 0.0
        intercept = explanation.intercept.get(label, 0.0) if hasattr(explanation, "intercept") and isinstance(explanation.intercept, dict) else 0.0

        contributions = []
        for feat_desc, weight in feature_list:
            # LIME returns feature descriptions like "feature_name <= 0.5"
            # We extract just the feature name for cleaner output
            contributions.append({
                "feature_description": feat_desc,
                "feature_name": self._extract_feature_name(feat_desc),
                "weight": float(weight),
            })

        pred = float(model.predict_proba(x.reshape(1, -1))[0])

        return LIMEExplanation(
            feature_contributions=contributions,
            prediction=pred,
            intercept=float(intercept),
            score=float(score),
        )

    def plot_explanation(
        self,
        explanation: LIMEExplanation,
        save_path: Optional[str] = None,
        title: str = "LIME Explanation",
    ) -> Optional[plt.Figure]:
        """Generate a bar chart of LIME feature contributions.

        Args:
            explanation: A LIMEExplanation object.
            save_path: Base filename for saving. If None, returns figure.
            title: Plot title.

        Returns:
            Matplotlib figure if save_path is None.
        """
        top = explanation.top_features(k=10)
        names = [f["feature_description"] for f in top]
        weights = [f["weight"] for f in top]
        colours = ["#ff4444" if w > 0 else "#44ff44" for w in weights]

        fig, ax = plt.subplots(figsize=(10, 6))
        y_pos = range(len(names))
        ax.barh(y_pos, weights[::-1], color=colours[::-1], edgecolor="#333")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names[::-1], fontsize=8)
        ax.set_xlabel("Feature Weight", fontsize=11)
        ax.set_title(
            f"{title} (pred={explanation.prediction:.3f})",
            fontsize=12,
            fontweight="bold",
        )
        ax.axvline(x=0, color="gray", linestyle="--", linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()

        if save_path:
            save_plot(fig, save_path)
            return None
        return fig

    @staticmethod
    def _extract_feature_name(description: str) -> str:
        """Extract the base feature name from a LIME feature description.

        LIME produces descriptions like 'byte_hist_42 <= 0.05' — we want
        just 'byte_hist_42'.
        """
        for sep in (" <= ", " < ", " > ", " >= ", " = "):
            if sep in description:
                return description.split(sep)[0].strip()
        return description.strip()
