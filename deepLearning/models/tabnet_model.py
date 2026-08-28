"""TabNet Classifier Architecture & Wrapper for HARUSPEX Malware Detection.

Wraps `pytorch-tabnet`'s `TabNetClassifier` to conform to the `MalwareDetector` interface.
Provides feature importances via TabNet's sparse attention masks, training history tracking,
and KernelSHAP integration.
"""

from __future__ import annotations

import os
import logging
from typing import Dict, Any, Optional, Tuple

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
            # IMPORTANT: pytorch-tabnet's internal compute_importance=True (default)
            # runs explain() over the FULL X_train after training finishes, using an
            # O(n_batches^2) np.vstack accumulation per decision step (n_steps=5 on
            # GPU). On large datasets (e.g. 500K+ rows) this causes a massive RAM
            # spike right after early stopping and can OOM-kill the process/notebook.
            # We disable it here and compute importances ourselves afterwards on a
            # small, safe subsample via get_feature_importance().
            compute_importance=False,
        )

        self._fitted = True

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        best_cost = self.model.best_cost if hasattr(self.model, "best_cost") else self.max_epochs
        best_acc = 0.0
        # NOTE: pytorch-tabnet's `History` callback object only implements
        # `__getitem__` (for direct key access), not `__contains__`. Using the
        # `in` operator on it falls back to Python's legacy sequence protocol
        # (probing history[0], history[1], ...), which raises an uncaught
        # KeyError instead of the expected membership check. Check against the
        # underlying `.history` dict instead, which behaves like a normal dict.
        history_dict = getattr(self.model, "history", None)
        history_data = getattr(history_dict, "history", None) if history_dict is not None else None
        if isinstance(history_data, dict) and "val_accuracy" in history_data and history_data["val_accuracy"]:
            best_acc = float(history_data["val_accuracy"][-1])

        return {
            "train_accuracy": best_acc,
            "best_epoch": best_cost,
        }

    def predict(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Predict binary labels (0 = benign, 1 = malware) in memory-safe batches."""
        if not self._fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predicting.")
        if len(X) <= batch_size:
            preds = self.model.predict(X)
            return preds.astype(np.int32)
        preds_list = []
        for i in range(0, len(X), batch_size):
            bx = X[i : i + batch_size]
            p = self.model.predict(bx)
            preds_list.append(p.astype(np.int32))
        return np.concatenate(preds_list, axis=0)

    def predict_proba(self, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        """Predict positive class probabilities (malware) in memory-safe batches."""
        if not self._fitted or self.model is None:
            raise RuntimeError("Model must be fitted before predicting.")
        if len(X) <= batch_size:
            probas = self.model.predict_proba(X)
            if probas.ndim == 2 and probas.shape[1] == 2:
                return probas[:, 1].astype(np.float32)
            return probas.astype(np.float32)
        probas_list = []
        for i in range(0, len(X), batch_size):
            bx = X[i : i + batch_size]
            p = self.model.predict_proba(bx)
            p_pos = p[:, 1].astype(np.float32) if p.ndim == 2 and p.shape[1] == 2 else p.astype(np.float32)
            probas_list.append(p_pos)
        return np.concatenate(probas_list, axis=0)

    def get_feature_importance(self, X_sample: Optional[np.ndarray] = None) -> np.ndarray:
        """Return feature importances safely without RAM spikes."""
        if not self._fitted or self.model is None:
            return np.ones(NUM_FEATURES) / NUM_FEATURES
        if X_sample is not None and hasattr(self.model, "explain"):
            try:
                # Sample up to 2000 items to prevent RAM explosion
                subset = X_sample[:2000] if len(X_sample) > 2000 else X_sample
                masks, _ = self.model.explain(subset)
                imp = np.mean(masks, axis=0)
                norm = np.sum(imp) + 1e-9
                return imp / norm
            except Exception as e:
                logger.warning("Subset feature importance calculation failed: %s", e)
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        return np.ones(NUM_FEATURES) / NUM_FEATURES

    def get_gradient_saliency(self, X: np.ndarray) -> np.ndarray:
        """Return unsigned TabNet selection weights (heuristic, not SHAP)."""
        if not self._fitted or self.model is None:
            return np.ones(NUM_FEATURES) / NUM_FEATURES
        x_flat = np.array(X, copy=True).reshape(1, -1)
        try:
            masks, _ = self.model.explain(x_flat)
            mask_weight = masks[0].flatten()
            total_mask = np.sum(np.abs(mask_weight)) + 1e-9
            return mask_weight / total_mask
        except Exception:
            global_imp = self.get_feature_importance()
            return global_imp

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
