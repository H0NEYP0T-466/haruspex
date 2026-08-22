"""XGBoost malware detector wrapper."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from xgboost import XGBClassifier

from haruspex.config import XGBOOST_PARAMS
from haruspex.models.base import MalwareDetector
from haruspex.utils.logging import get_logger

logger = get_logger(__name__)


class XGBoostDetector(MalwareDetector):
    """XGBoost-based binary malware classifier.

    Wraps :class:`xgboost.XGBClassifier` with the HARUSPEX model interface,
    providing consistent fit/predict/save/load across the pipeline.
    """

    def __init__(self, **override_params: Any) -> None:
        """Initialise with default hyperparameters (overridable).

        Args:
            **override_params: Any XGBClassifier parameter to override defaults.
        """
        params = XGBOOST_PARAMS.to_dict()
        params.update(override_params)
        self._model = XGBClassifier(**params)
        self._fitted = False

    @property
    def model_name(self) -> str:
        return "xgboost"

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> dict[str, float]:
        """Train the XGBoost model.

        Uses early stopping if validation data is provided.
        """
        logger.info(
            "Training XGBoost: %d samples, %d features",
            X_train.shape[0],
            X_train.shape[1],
        )

        fit_kwargs: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = [(X_val, y_val)]

        self._model.fit(X_train, y_train, **fit_kwargs)
        self._fitted = True

        # Compute training metrics
        train_pred = self._model.predict(X_train)
        train_acc = float(np.mean(train_pred == y_train))
        metrics = {"train_accuracy": train_acc}

        if X_val is not None and y_val is not None:
            val_pred = self._model.predict(X_val)
            val_acc = float(np.mean(val_pred == y_val))
            metrics["val_accuracy"] = val_acc
            logger.info("XGBoost trained: train_acc=%.4f, val_acc=%.4f", train_acc, val_acc)
        else:
            logger.info("XGBoost trained: train_acc=%.4f", train_acc)

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels."""
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return malware probability (positive class)."""
        proba = self._model.predict_proba(X)
        return proba[:, 1]

    def get_feature_importance(self) -> np.ndarray:
        """Return feature importance (gain-based)."""
        return self._model.feature_importances_

    def save(self, path: str) -> None:
        """Save model in XGBoost's native JSON format."""
        self._model.save_model(path)
        logger.info("XGBoost model saved to %s", path)

    def load(self, path: str) -> None:
        """Load model from XGBoost JSON file."""
        self._model.load_model(path)
        self._fitted = True
        logger.info("XGBoost model loaded from %s", path)

    def get_underlying_model(self) -> XGBClassifier:
        """Return the underlying XGBClassifier for SHAP/LIME."""
        return self._model
