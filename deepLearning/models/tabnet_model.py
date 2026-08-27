"""TabNet Classifier Architecture & Wrapper for HARUSPEX Malware Detection.

Wraps `pytorch-tabnet`'s `TabNetClassifier` to conform to the `MalwareDetector` interface.
Provides feature importances via TabNet's sparse attention masks, training history tracking,
and KernelSHAP integration.
"""

from __future__ import annotations

import os
import logging
from typing import Dict, Any, Optional

import numpy as np
import torch

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_BACKEND = _ROOT / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from haruspex.models.base import MalwareDetector
from deepLearning.utils.config import (
    NUM_FEATURES,
    TABNET_PARAMS,
    get_device,
)

logger = logging.getLogger("haruspex.deepLearning.models.tabnet")

# Attempt importing TabNetClassifier
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    _HAS_TABNET_PKG = True
except ImportError:
    _HAS_TABNET_PKG = False


class TabNetDetector(MalwareDetector):
    """TabNet Malware Detector conforming to the MalwareDetector interface."""

    def __init__(
        self,
        model_name_suffix: str = "",
        n_d: int = TABNET_PARAMS.n_d,
        n_a: int = TABNET_PARAMS.n_a,
        n_steps: int = TABNET_PARAMS.n_steps,
        gamma: float = TABNET_PARAMS.gamma,
        lambda_sparse: float = TABNET_PARAMS.lambda_sparse,
        batch_size: int = TABNET_PARAMS.batch_size_bodmas,
        max_epochs: int = TABNET_PARAMS.max_epochs,
        patience: int = TABNET_PARAMS.patience,
        lr: float = TABNET_PARAMS.lr,
    ):
        self._name_suffix = model_name_suffix
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.lr = lr

        self.device = get_device()
        self.model = None
        self._fitted = False
        self.training_history: Dict[str, list] = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

    @property
    def model_name(self) -> str:
        base = "tabnet"
        if self._name_suffix:
            base = f"{base}_{self._name_suffix}"
        return base

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _init_model(self):
        if self.model is None:
            if not _HAS_TABNET_PKG:
                raise ImportError(
                    "pytorch-tabnet package is not installed. "
                    "Install it via `pip install pytorch-tabnet`."
                )

            device_str = "cuda" if self.device.type == "cuda" else "cpu"

            self.model = TabNetClassifier(
                n_d=self.n_d,
                n_a=self.n_a,
                n_steps=self.n_steps,
                gamma=self.gamma,
                lambda_sparse=self.lambda_sparse,
                optimizer_fn=torch.optim.Adam,
                optimizer_params=dict(lr=self.lr),
                mask_type="sparsemax",
                device_name=device_str,
                verbose=1,
            )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Train the TabNetClassifier model."""
        self._init_model()
        assert self.model is not None

        eval_set = []
        eval_name = []
        if X_val is not None and y_val is not None:
            eval_set = [(X_val, y_val.astype(np.int64))]
            eval_name = ["val"]

        logger.info("Training %s on device: %s (%d samples)", self.model_name, self.device, len(X_train))

        self.model.fit(
            X_train=X_train,
            y_train=y_train.astype(np.int64),
            eval_set=eval_set,
            eval_name=eval_name,
            eval_metric=["accuracy", "logloss"],
            max_epochs=self.max_epochs,
            patience=self.patience,
            batch_size=self.batch_size,
            virtual_batch_size=min(256, self.batch_size),
            num_workers=0,
            drop_last=False,
        )

        self._fitted = True

        # Extract training metrics
        test_pred = self.predict(X_val if X_val is not None else X_train)
        target = y_val if y_val is not None else y_train
        acc = float(np.mean(test_pred == target))

        return {
            "train_accuracy": acc,
            "best_epoch": self.model.best_cost if hasattr(self.model, "best_cost") else self.max_epochs,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels (0 = benign, 1 = malware)."""
        if not self._fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predicting.")
        preds = self.model.predict(X)
        return preds.astype(np.int32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict positive class probabilities (malware)."""
        if not self._fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predicting.")
        probas = self.model.predict_proba(X)
        # Handle 2D output (n_samples, 2)
        if probas.ndim == 2 and probas.shape[1] == 2:
            return probas[:, 1].astype(np.float32)
        return probas.astype(np.float32)

    def get_feature_importance(self) -> np.ndarray:
        """Return feature importances from TabNet's attention masks."""
        if not self._fitted or self.model is None:
            return np.ones(NUM_FEATURES) / NUM_FEATURES
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return np.ones(NUM_FEATURES) / NUM_FEATURES

    def get_gradient_saliency(self, X: np.ndarray) -> np.ndarray:
        """Compute TabNet per-sample feature attributions matching log-odds scale."""
        if not self._fitted or self.model is None:
            return np.ones(NUM_FEATURES) / NUM_FEATURES
        x_flat = np.array(X, copy=True).reshape(1, -1)
        prob = float(self.predict_proba(x_flat)[0])
        try:
            masks, _ = self.model.explain(x_flat)
            mask_weight = masks[0].flatten()
            total_mask = np.sum(np.abs(mask_weight)) + 1e-9
            logit_shift = float(np.log(max(prob, 1e-5) / max(1.0 - prob, 1e-5)))
            return (mask_weight / total_mask) * logit_shift
        except Exception:
            global_imp = self.get_feature_importance()
            return global_imp * (prob - 0.5)

    def explain_masks(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[int, np.ndarray]]:
        """Return TabNet masks and aggregate feature attributions."""
        if not self._fitted or self.model is None:
            raise RuntimeError("Model must be fitted before explaining masks.")
        masks, masks_dict = self.model.explain(X)
        return masks, masks_dict

    def save(self, path: str) -> None:
        """Save trained TabNet model to disk."""
        if not self._fitted or self.model is None:
            raise RuntimeError("Model must be fitted before saving.")

        # Save zip or file
        if path.endswith(".zip"):
            save_path = path[:-4]
        else:
            save_path = path

        self.model.save_model(save_path)
        logger.info("Saved %s model to %s", self.model_name, save_path)

    def load(self, path: str) -> None:
        """Load trained TabNet model from disk."""
        self._init_model()
        assert self.model is not None

        if path.endswith(".zip"):
            load_path = path
        elif os.path.exists(f"{path}.zip"):
            load_path = f"{path}.zip"
        else:
            load_path = path

        self.model.load_model(load_path)
        self._fitted = True
        logger.info("Loaded %s model from %s", self.model_name, load_path)

    def get_underlying_model(self) -> Any:
        """Return underlying TabNetClassifier instance."""
        return self.model
