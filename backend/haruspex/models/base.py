"""Abstract base class for HARUSPEX malware detection models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np


class MalwareDetector(ABC):
    """Abstract interface for malware detection models.

    All concrete models (XGBoost, LightGBM, etc.) must implement this interface
    to ensure interchangeability throughout the pipeline.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the human-readable name of the model."""

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Return True if the model has been trained."""

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """Train the model.

        Args:
            X_train: Training feature matrix.
            y_train: Training labels.
            X_val: Optional validation features (for early stopping).
            y_val: Optional validation labels.

        Returns:
            Dictionary of training metrics.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels (0 = benign, 1 = malware).

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Array of predicted labels.
        """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for the positive class (malware).

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Array of malware probabilities (n_samples,).
        """

    @abstractmethod
    def get_feature_importance(self) -> np.ndarray:
        """Return feature importance scores.

        Returns:
            Array of importance scores (n_features,).
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the trained model to disk.

        Args:
            path: File path to save to.
        """

    @abstractmethod
    def load(self, path: str) -> None:
        """Load a trained model from disk.

        Args:
            path: File path to load from.
        """

    @abstractmethod
    def get_underlying_model(self) -> Any:
        """Return the underlying ML model object (for SHAP/LIME compatibility)."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name={self.model_name!r}, fitted={self.is_fitted})"
