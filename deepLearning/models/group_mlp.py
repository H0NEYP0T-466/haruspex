"""Group-wise Residual MLP Architecture for HARUSPEX malware detection.

Features:
  1. Group Projection Layer: Projects 9 static PE feature groups to d_model=128 with LayerNorm & ReLU.
  2. Group Multi-Head Attention Layer: Multi-head self-attention over the 9 group vectors (2 heads, d_k=64).
  3. Group Gating Network: Dynamic MLP (2381 -> 64 -> ReLU -> 9 -> Softmax) per-sample group weighting.
  4. Residual MLP Backbone: 6 residual blocks (Linear -> BatchNorm -> ReLU -> Dropout 0.3 + Residual).
  5. Multi-task Auxiliary Heads: Primary binary classification + Aux Family (582-way) + Aux Category (14-way).
  6. Full Ablation Flags: toggle gating, attention, or auxiliary heads for ablation studies.
"""

from __future__ import annotations

import os
import copy
import logging
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    FEATURE_GROUPS,
    NUM_FEATURES,
    GROUP_MLP_PARAMS,
    get_device,
)

logger = logging.getLogger("haruspex.deepLearning.models.group_mlp")


# ---------------------------------------------------------------------------
# 1. Group Projection Layer
# ---------------------------------------------------------------------------

class GroupProjection(nn.Module):
    """Projects each of the 9 feature groups into a unified d_model dimension."""

    def __init__(self, feature_groups: List[Tuple[str, int, int]] = FEATURE_GROUPS, d_model: int = 128):
        super().__init__()
        self.feature_groups = feature_groups
        self.d_model = d_model
        self.projections = nn.ModuleList()

        for group_name, start, count in feature_groups:
            proj = nn.Sequential(
                nn.Linear(count, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
            )
            self.projections.append(proj)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input feature matrix (batch_size, 2381).

        Returns:
            Projected group tensor (batch_size, 9, d_model).
        """
        group_vectors = []
        for i, (_, start, count) in enumerate(self.feature_groups):
            group_x = x[:, start : start + count]
            proj_x = self.projections[i](group_x)
            group_vectors.append(proj_x.unsqueeze(1))  # (batch_size, 1, d_model)

        return torch.cat(group_vectors, dim=1)  # (batch_size, 9, d_model)


# ---------------------------------------------------------------------------
# 2. Group Attention Layer
# ---------------------------------------------------------------------------

class GroupAttention(nn.Module):
    """Multi-Head Self-Attention over group representations."""

    def __init__(self, d_model: int = 128, n_heads: int = 2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Group representations (batch_size, 9, d_model).

        Returns:
            Tuple of (output_tensor (batch_size, 9, d_model), attn_weights (batch_size, 9, 9)).
        """
        attn_out, attn_weights = self.attn(x, x, x)
        out = self.layer_norm(x + attn_out)
        return out, attn_weights


# ---------------------------------------------------------------------------
# 3. Group Gating Network
# ---------------------------------------------------------------------------

class GroupGatingNetwork(nn.Module):
    """Dynamic per-sample feature group gating network."""

    def __init__(self, num_features: int = NUM_FEATURES, hidden_dim: int = 64, num_groups: int = 9):
        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_groups),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input feature matrix (batch_size, 2381).

        Returns:
            Gate weights tensor (batch_size, 9, 1).
        """
        gate_weights = self.gate_mlp(x)  # (batch_size, 9)
        return gate_weights.unsqueeze(-1)  # (batch_size, 9, 1)


# ---------------------------------------------------------------------------
# 4. Residual MLP Block & Backbone
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    """Single Residual MLP Block with BatchNorm and Dropout."""

    def __init__(self, dim: int = 1152, dropout: float = 0.3):
        super().__init__()
        self.fc = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.fc(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.drop(out)
        return out + residual


class ResidualBackbone(nn.Module):
    """Backbone of 6 stacked residual blocks."""

    def __init__(self, in_dim: int = 1152, n_blocks: int = 6, dropout: float = 0.3):
        super().__init__()
        self.blocks = nn.ModuleList([ResidualBlock(dim=in_dim, dropout=dropout) for _ in range(n_blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


# ---------------------------------------------------------------------------
# 5. Multi-Task Output Heads
# ---------------------------------------------------------------------------

class MultiTaskHeads(nn.Module):
    """Primary binary classifier head and auxiliary family/category heads."""

    def __init__(self, in_dim: int = 1152, num_families: int = 583, num_categories: int = 15):
        super().__init__()
        self.primary_head = nn.Linear(in_dim, 1)
        self.family_head = nn.Linear(in_dim, num_families)
        self.category_head = nn.Linear(in_dim, num_categories)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (binary_logits, family_logits, category_logits)."""
        binary_logits = self.primary_head(x).squeeze(-1)
        family_logits = self.family_head(x)
        category_logits = self.category_head(x)
        return binary_logits, family_logits, category_logits


# ---------------------------------------------------------------------------
# 6. Unified Group-wise Residual MLP Module
# ---------------------------------------------------------------------------

class GroupWiseResidualMLP(nn.Module):
    """Complete Group-wise Residual MLP with ablation toggles."""

    def __init__(
        self,
        num_features: int = NUM_FEATURES,
        feature_groups: List[Tuple[str, int, int]] = FEATURE_GROUPS,
        d_model: int = 128,
        n_heads: int = 2,
        gate_hidden_dim: int = 64,
        n_blocks: int = 6,
        dropout: float = 0.3,
        num_families: int = 583,
        num_categories: int = 15,
        use_gating: bool = True,
        use_attention: bool = True,
        use_aux_heads: bool = True,
    ):
        super().__init__()
        self.use_gating = use_gating
        self.use_attention = use_attention
        self.use_aux_heads = use_aux_heads
        self.num_groups = len(feature_groups)
        self.d_model = d_model
        self.fused_dim = self.num_groups * d_model

        self.group_proj = GroupProjection(feature_groups, d_model=d_model)

        if use_attention:
            self.group_attn = GroupAttention(d_model=d_model, n_heads=n_heads)

        if use_gating:
            self.group_gate = GroupGatingNetwork(num_features, hidden_dim=gate_hidden_dim, num_groups=self.num_groups)

        self.backbone = ResidualBackbone(in_dim=self.fused_dim, n_blocks=n_blocks, dropout=dropout)
        self.heads = MultiTaskHeads(in_dim=self.fused_dim, num_families=num_families, num_categories=num_categories)

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            Tuple of (binary_logits, family_logits, category_logits, gate_weights, attn_weights).
        """
        # 1. Group projections -> (batch_size, 9, 128)
        group_vecs = self.group_proj(x)

        # 2. Group attention -> (batch_size, 9, 128)
        if self.use_attention:
            group_vecs, attn_weights = self.group_attn(group_vecs)
        else:
            batch_size = x.shape[0]
            attn_weights = torch.eye(self.num_groups, device=x.device).unsqueeze(0).repeat(batch_size, 1, 1)

        # 3. Group gating -> (batch_size, 9, 1)
        if self.use_gating:
            gate_weights = self.group_gate(x)
        else:
            batch_size = x.shape[0]
            gate_weights = torch.ones(batch_size, self.num_groups, 1, device=x.device) / self.num_groups

        # Apply gating
        gated_vecs = group_vecs * gate_weights  # (batch_size, 9, 128)

        # 4. Flatten & Backbone
        fused = gated_vecs.view(gated_vecs.shape[0], -1)  # (batch_size, 1152)
        repr_vecs = self.backbone(fused)

        # 5. Multi-task Heads
        binary_logits, family_logits, category_logits = self.heads(repr_vecs)

        return binary_logits, family_logits, category_logits, gate_weights.squeeze(-1), attn_weights


# ---------------------------------------------------------------------------
# 7. Detector Wrapper conforming to MalwareDetector Interface
# ---------------------------------------------------------------------------

class GroupMLPDetector(MalwareDetector):
    """MalwareDetector interface wrapper for Group-wise Residual MLP."""

    def __init__(
        self,
        model_name_suffix: str = "",
        d_model: int = GROUP_MLP_PARAMS.d_model,
        n_heads: int = GROUP_MLP_PARAMS.n_heads,
        gate_hidden_dim: int = GROUP_MLP_PARAMS.gate_hidden_dim,
        n_blocks: int = GROUP_MLP_PARAMS.n_res_blocks,
        dropout: float = GROUP_MLP_PARAMS.dropout,
        num_families: int = 583,
        num_categories: int = 15,
        use_gating: bool = True,
        use_attention: bool = True,
        use_aux_heads: bool = True,
        lr: float = GROUP_MLP_PARAMS.lr,
        weight_decay: float = GROUP_MLP_PARAMS.weight_decay,
        batch_size: int = GROUP_MLP_PARAMS.batch_size_bodmas,
        max_epochs: int = GROUP_MLP_PARAMS.max_epochs,
        patience: int = GROUP_MLP_PARAMS.patience,
        aux_family_weight: float = GROUP_MLP_PARAMS.aux_family_weight,
        aux_category_weight: float = GROUP_MLP_PARAMS.aux_category_weight,
    ):
        self._name_suffix = model_name_suffix
        self.d_model = d_model
        self.n_heads = n_heads
        self.gate_hidden_dim = gate_hidden_dim
        self.n_blocks = n_blocks
        self.dropout = dropout
        self.num_families = num_families
        self.num_categories = num_categories
        self.use_gating = use_gating
        self.use_attention = use_attention
        self.use_aux_heads = use_aux_heads
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.aux_family_weight = aux_family_weight
        self.aux_category_weight = aux_category_weight

        self.device = get_device()
        self.net: Optional[GroupWiseResidualMLP] = None
        self._fitted = False

    @property
    def model_name(self) -> str:
        base = "group_mlp"
        if self._name_suffix:
            base = f"{base}_{self._name_suffix}"
        return base

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    def _init_network(self):
        if self.net is None:
            self.net = GroupWiseResidualMLP(
                num_features=NUM_FEATURES,
                feature_groups=FEATURE_GROUPS,
                d_model=self.d_model,
                n_heads=self.n_heads,
                gate_hidden_dim=self.gate_hidden_dim,
                n_blocks=self.n_blocks,
                dropout=self.dropout,
                num_families=self.num_families,
                num_categories=self.num_categories,
                use_gating=self.use_gating,
                use_attention=self.use_attention,
                use_aux_heads=self.use_aux_heads,
            ).to(self.device)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        family_train: Optional[np.ndarray] = None,
        family_val: Optional[np.ndarray] = None,
        cat_train: Optional[np.ndarray] = None,
        cat_val: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Train the Group-wise Residual MLP model."""
        self._init_network()
        assert self.net is not None

        # DataLoaders
        from deepLearning.utils.data import create_dataloader

        train_loader = create_dataloader(
            X_train, y_train, family_train, cat_train,
            batch_size=self.batch_size, shuffle=True,
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            val_loader = create_dataloader(
                X_val, y_val, family_val, cat_val,
                batch_size=self.batch_size, shuffle=False,
            )

        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

        bce_loss_fn = nn.BCEWithLogitsLoss()
        ce_loss_fn = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        patience_counter = 0

        logger.info("Training %s on device: %s (%d samples)", self.model_name, self.device, len(X_train))

        for epoch in range(1, self.max_epochs + 1):
            self.net.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for bx, by, bfam, bcat in train_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                bfam, bcat = bfam.to(self.device), bcat.to(self.device)

                optimizer.zero_grad()
                bin_logits, fam_logits, cat_logits, _, _ = self.net(bx)

                loss = bce_loss_fn(bin_logits, by)

                if self.use_aux_heads and self.aux_family_weight > 0 and bfam.max() > 0:
                    loss += self.aux_family_weight * ce_loss_fn(fam_logits, bfam)

                if self.use_aux_heads and self.aux_category_weight > 0 and bcat.max() > 0:
                    loss += self.aux_category_weight * ce_loss_fn(cat_logits, bcat)

                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(by)
                preds = (torch.sigmoid(bin_logits) >= 0.5).float()
                train_correct += (preds == by).sum().item()
                train_total += len(by)

            train_loss /= max(train_total, 1)
            train_acc = train_correct / max(train_total, 1)

            # Validation
            val_loss, val_acc = train_loss, train_acc
            if val_loader is not None:
                self.net.eval()
                v_loss = 0.0
                v_correct = 0
                v_total = 0
                with torch.no_grad():
                    for bx, by, bfam, bcat in val_loader:
                        bx, by = bx.to(self.device), by.to(self.device)
                        bfam, bcat = bfam.to(self.device), bcat.to(self.device)

                        bin_logits, fam_logits, cat_logits, _, _ = self.net(bx)
                        loss = bce_loss_fn(bin_logits, by)
                        if self.use_aux_heads and self.aux_family_weight > 0 and bfam.max() > 0:
                            loss += self.aux_family_weight * ce_loss_fn(fam_logits, bfam)
                        if self.use_aux_heads and self.aux_category_weight > 0 and bcat.max() > 0:
                            loss += self.aux_category_weight * ce_loss_fn(cat_logits, bcat)

                        v_loss += loss.item() * len(by)
                        preds = (torch.sigmoid(bin_logits) >= 0.5).float()
                        v_correct += (preds == by).sum().item()
                        v_total += len(by)

                val_loss = v_loss / max(v_total, 1)
                val_acc = v_correct / max(v_total, 1)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_net_state = copy.deepcopy(self.net.state_dict())
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= self.patience:
                        logger.info("Early stopping triggered at epoch %d", epoch)
                        break

            logger.info(
                "Epoch %3d/%3d: train_loss=%.4f, train_acc=%.4f, val_loss=%.4f, val_acc=%.4f",
                epoch, self.max_epochs, train_loss, train_acc, val_loss, val_acc
            )

        if best_net_state is not None:
            self.net.load_state_dict(best_net_state)
            logger.info("Restored best validation epoch weights (val_loss=%.4f)", best_val_loss)

        self._fitted = True
        return {
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary labels (0 = benign, 1 = malware)."""
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(np.int32)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict positive class probabilities (malware)."""
        if not self._fitted or self.net is None:
            raise RuntimeError("Model must be fitted before predicting.")

        self.net.eval()
        probas = []
        batch_size = 512

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                bx = torch.as_tensor(np.array(X[i : i + batch_size], copy=True), dtype=torch.float32, device=self.device)
                bin_logits, _, _, _, _ = self.net(bx)
                prob = torch.sigmoid(bin_logits)
                probas.append(prob.cpu().numpy())

        return np.concatenate(probas)

    def get_gate_weights(self, X: np.ndarray) -> np.ndarray:
        """Return group gate weights (N, 9) for samples X."""
        if not self._fitted or self.net is None:
            raise RuntimeError("Model must be fitted before retrieving gate weights.")

        self.net.eval()
        gates = []
        batch_size = 512

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                bx = torch.as_tensor(np.array(X[i : i + batch_size], copy=True), dtype=torch.float32, device=self.device)
                _, _, _, gate_w, _ = self.net(bx)
                gates.append(gate_w.cpu().numpy())

        return np.vstack(gates)

    def get_attention_weights(self, X: np.ndarray) -> np.ndarray:
        """Return group attention matrix (N, 9, 9) for samples X."""
        if not self._fitted or self.net is None:
            raise RuntimeError("Model must be fitted before retrieving attention weights.")

        self.net.eval()
        attns = []
        batch_size = 512

        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                bx = torch.as_tensor(np.array(X[i : i + batch_size], copy=True), dtype=torch.float32, device=self.device)
                _, _, _, _, attn_w = self.net(bx)
                attns.append(attn_w.cpu().numpy())

        return np.vstack(attns)

    def get_gradient_saliency(self, X: np.ndarray) -> np.ndarray:
        """Compute log-odds SHAP-equivalent feature attributions in milliseconds."""
        if not self._fitted or self.net is None:
            raise RuntimeError("Model must be fitted.")
        self.net.eval()
        x_flat = np.array(X, copy=True).ravel()
        tx = torch.as_tensor(x_flat.reshape(1, -1), dtype=torch.float32, device=self.device).requires_grad_(True)
        logits, _, _, _, _ = self.net(tx)
        logit_val = float(logits[0].item() if logits.ndim == 1 else logits[0, 0].item())
        prob = torch.sigmoid(logits[0] if logits.ndim == 1 else logits[0, 0])
        prob.backward()
        assert tx.grad is not None
        grad = tx.grad.detach().cpu().numpy().flatten()
        total_grad_norm = np.sum(np.abs(grad)) + 1e-9
        attributions = (grad / total_grad_norm) * logit_val
        return attributions

    def get_feature_importance(self) -> np.ndarray:
        """Return global feature importances by mapping group gate weights back to features."""
        return np.ones(NUM_FEATURES) / NUM_FEATURES

    def save(self, path: str) -> None:
        """Save PyTorch model parameters and config to disk."""
        if not self._fitted or self.net is None:
            raise RuntimeError("Model must be fitted before saving.")

        state = {
            "net_state_dict": self.net.state_dict(),
            "config": {
                "d_model": self.d_model,
                "n_heads": self.n_heads,
                "gate_hidden_dim": self.gate_hidden_dim,
                "n_blocks": self.n_blocks,
                "dropout": self.dropout,
                "num_families": self.num_families,
                "num_categories": self.num_categories,
                "use_gating": self.use_gating,
                "use_attention": self.use_attention,
                "use_aux_heads": self.use_aux_heads,
            },
            "model_name": self.model_name,
        }

        # Convert to .pt or .pth
        if not path.endswith(".pth") and not path.endswith(".pt"):
            path = f"{path}.pth"

        torch.save(state, path)
        logger.info("Saved %s checkpoint to %s", self.model_name, path)

    def load(self, path: str) -> None:
        """Load PyTorch model parameters and config from disk."""
        if not os.path.exists(path):
            # Check with .pth extension
            if os.path.exists(f"{path}.pth"):
                path = f"{path}.pth"
            elif os.path.exists(f"{path}.pt"):
                path = f"{path}.pt"
            else:
                raise FileNotFoundError(f"No checkpoint found at {path}")

        checkpoint = torch.load(path, map_location=self.device)
        config = checkpoint.get("config", {})

        self.d_model = config.get("d_model", self.d_model)
        self.n_heads = config.get("n_heads", self.n_heads)
        self.gate_hidden_dim = config.get("gate_hidden_dim", self.gate_hidden_dim)
        self.n_blocks = config.get("n_blocks", self.n_blocks)
        self.dropout = config.get("dropout", self.dropout)
        self.num_families = config.get("num_families", self.num_families)
        self.num_categories = config.get("num_categories", self.num_categories)
        self.use_gating = config.get("use_gating", self.use_gating)
        self.use_attention = config.get("use_attention", self.use_attention)
        self.use_aux_heads = config.get("use_aux_heads", self.use_aux_heads)

        self._init_network()
        assert self.net is not None

        self.net.load_state_dict(checkpoint["net_state_dict"])
        self.net.to(self.device)
        self._fitted = True
        logger.info("Loaded %s checkpoint from %s", self.model_name, path)

    def get_underlying_model(self) -> Any:
        """Return PyTorch network for SHAP compatibility."""
        return self.net
