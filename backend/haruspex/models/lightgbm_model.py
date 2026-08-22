"""LightGBM malware detector wrapper."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from lightgbm import LGBMClassifier

from haruspex.config import LIGHTGBM_PARAMS
from haruspex.models.base import MalwareDetector
from haruspex.utils.logging import get_logger

logger = get_logger(__name__)


class LightGBMDetector(MalwareDetector):
    """LightGBM-based binary malware classifier.

    Wraps :class:`lightgbm.LGBMClassifier` with the HARUSPEX model interface.
    """

    def __init__(self, **override_params: Any) -> None:
        """Initialise with default hyperparameters (overridable).

        Args:
            **override_params: Any LGBMClassifier parameter to override defaults.
        """
        params = LIGHTGBM_PARAMS.to_dict()
        params.update(override_params)
        self._model = LGBMClassifier(**params)
        self._fitted = False

    @property
    def model_name(self) -> str:
        return "lightgbm"

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
        """Train the LightGBM model.

        Uses early stopping if validation data is provided.
        """
        logger.info(
            "Training LightGBM: %d samples, %d features",
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
            logger.info("LightGBM trained: train_acc=%.4f, val_acc=%.4f", train_acc, val_acc)
        else:
            logger.info("LightGBM trained: train_acc=%.4f", train_acc)

        return metrics

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels."""
        if hasattr(self._model, "_Booster") and not hasattr(self._model, "fitted_"):
            # Model loaded directly from booster file
            proba = self._model._Booster.predict(X)
            return (proba >= 0.5).astype(int)
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return malware probability (positive class)."""
        if hasattr(self._model, "_Booster") and not hasattr(self._model, "fitted_"):
            # Model loaded directly from booster file
            return self._model._Booster.predict(X)
        proba = self._model.predict_proba(X)
        return proba[:, 1]

    def get_feature_importance(self) -> np.ndarray:
        """Return feature importance (gain-based)."""
        if hasattr(self._model, "_Booster") and not hasattr(self._model, "fitted_"):
            return self._model._Booster.feature_importance(importance_type="gain")
        return self._model.feature_importances_

    def save(self, path: str) -> None:
        """Save model in LightGBM's native text format."""
        if hasattr(self._model, "_Booster"):
            self._model._Booster.save_model(path)
        else:
            self._model.booster_.save_model(path)
        logger.info("LightGBM model saved to %s", path)

    def load(self, path: str) -> None:
        """Load model from LightGBM text file."""
        import lightgbm as lgb

        booster = lgb.Booster(model_file=path)
        self._model = LGBMClassifier()
        self._model._Booster = booster
        self._model._classes = np.array([0, 1])
        self._model._n_classes = 2
        self._fitted = True
        logger.info("LightGBM model loaded from %s", path)

    def get_underlying_model(self) -> LGBMClassifier:
        """Return the underlying LGBMClassifier for SHAP/LIME."""
        return self._model
