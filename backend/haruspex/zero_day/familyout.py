"""Family-held-out zero-day evaluation.

Trains XGBoost and LightGBM on BODMAS, then tests on:
  - Malware from families NEVER seen during training (zero-day families)
  - Benign samples (to verify the model doesn't overfit to malware)

Workflow:
  1. Load BODMAS
  2. Training set  = all benign + malware from known families
  3. Test set      = all benign + malware from held-out families
  4. Train XGBoost -> evaluate -> train LightGBM -> evaluate
  5. Save models + plots + metrics to zero-day/data/
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

_BACKEND = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_BACKEND))

from haruspex.config import HELD_OUT_FAMILIES, RANDOM_SEED
from haruspex.utils.logging import get_logger

logger = get_logger(__name__)

HELD_OUT = HELD_OUT_FAMILIES
DATA_DIR = Path(__file__).resolve().parent / "data"


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def load_bodmas_raw():
    from haruspex.data.bodmas import load_bodmas
    dataset = load_bodmas()
    return dataset.X, dataset.y, dataset.metadata


def family_held_out_split(X, y, meta, held_out_families=None):
    families = meta["family"].values
    held_out_families = held_out_families or HELD_OUT
    benign_mask = y == 0
    held_out_mask = np.isin(families, held_out_families) & (y == 1)
    train_mask = ~held_out_mask
    test_mask = held_out_mask | benign_mask
    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    logger.info(
        "Split: train=%d (mal=%d, ben=%d) | test=%d (mal=%d, ben=%d)",
        len(X_train), int(y_train.sum()), int((y_train == 0).sum()),
        len(X_test), int(y_test.sum()), int((y_test == 0).sum()),
    )
    return X_train, y_train, X_test, y_test


def plot_roc(y_true, y_proba, name, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"{name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve - {name} (Family-Held-Out)", fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_confusion(y_true, y_pred, name, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=["benign", "malware"], yticklabels=["benign", "malware"],
           ylabel="True Label", xlabel="Predicted Label",
           title=f"Confusion Matrix - {name}")
    thresh = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def evaluate(name, y_true, y_pred, y_proba, data_dir):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_proba)

    print(f"\n{'=' * 50}")
    print(f"  {name} - Family-Held-Out Results")
    print(f"{'=' * 50}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['benign', 'malware'])}")

    plot_roc(y_true, y_proba, name, data_dir / f"{name.lower()}_roc.png")
    plot_confusion(y_true, y_pred, name, data_dir / f"{name.lower()}_confusion.png")

    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc_roc": auc}


class EpochLogger:
    """Logs per-round train/test accuracy for boosting models."""

    def __init__(self):
        self.round_num = 0

    def xgboost_callback(self):
        from xgboost.callback import TrainingCallback
        class _CB(TrainingCallback):
            def __init__(self, logger_obj):
                super().__init__()
                self._log = logger_obj
            def after_iteration(self, model, epoch, evals_log):
                self._log.log_xgb(epoch, evals_log)
                return False
        return _CB(self)

    def lightgbm_callback(self):
        import lightgbm as lgb
        def _cb(env):
            epoch = env.evaluation_result_list
            self.round_num += 1
            parts = []
            for name, metric, val, _ in epoch:
                parts.append(f"{name}:{metric}={val:.6f}")
            print(f"  [{self.round_num:>4d}] {' | '.join(parts)}")
        return _cb

    def log_xgb(self, epoch, evals_log):
        parts = []
        for name, vals in evals_log.items():
            metric_name = list(vals.keys())[0]
            val = vals[metric_name][epoch]
            parts.append(f"{name}:{metric_name}={val:.6f}")
        print(f"  [{epoch + 1:>4d}] {' | '.join(parts)}")


def train_xgboost(X_train, y_train, X_test, y_test, data_dir):
    import xgboost as xgb
    from haruspex.models.xgboost_model import XGBoostDetector

    print("\n--- XGBoost Training (200 rounds) ---")
    raw = xgb.XGBClassifier(
        objective="binary:logistic", eval_metric="logloss",
        tree_method="hist", n_estimators=200, max_depth=8,
        learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, random_state=RANDOM_SEED, verbosity=0,
    )

    logger_cb = EpochLogger()
    raw.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False,
        callbacks=[logger_cb.xgboost_callback()],
    )

    detector = XGBoostDetector()
    detector._model = raw
    detector._fitted = True
    model_path = data_dir / "xgboost_family_held_out.json"
    detector.save(str(model_path))
    print(f"  Saved model: {model_path}")

    y_pred = raw.predict(X_test)
    y_proba = raw.predict_proba(X_test)[:, 1]
    return evaluate("XGBoost", y_test, y_pred, y_proba, data_dir)


def train_lightgbm(X_train, y_train, X_test, y_test, data_dir):
    import lightgbm as lgb
    from haruspex.models.lightgbm_model import LightGBMDetector

    print("\n--- LightGBM Training (200 rounds) ---")
    raw = lgb.LGBMClassifier(
        objective="binary", metric="binary_logloss",
        n_estimators=200, max_depth=8, learning_rate=0.1,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=20, random_state=RANDOM_SEED, verbose=-1,
    )

    logger_cb = EpochLogger()
    raw.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        callbacks=[logger_cb.lightgbm_callback()],
    )

    detector = LightGBMDetector()
    detector._model = raw
    detector._fitted = True
    model_path = data_dir / "lightgbm_family_held_out.json"
    detector.save(str(model_path))
    print(f"  Saved model: {model_path}")

    y_pred = raw.predict(X_test)
    y_proba = raw.predict_proba(X_test)[:, 1]
    return evaluate("LightGBM", y_test, y_pred, y_proba, data_dir)


def main():
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(name)s: %(message)s",
        stream=sys.stdout,
    )

    data_dir = ensure_data_dir()
    print("Loading BODMAS dataset...")
    X, y, meta = load_bodmas_raw()
    print(f"\nDataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Benign: {int((y == 0).sum())}")
    print(f"  Malware: {int(y.sum())}")
    print(f"  Families: {meta['family'].nunique()}")
    print(f"  Held-out families: {HELD_OUT}")
    print(f"  Saving to: {data_dir}\n")

    X_train, y_train, X_test, y_test = family_held_out_split(X, y, meta)

    xgb_m = train_xgboost(X_train, y_train, X_test, y_test, data_dir)
    lgbm_m = train_lightgbm(X_train, y_train, X_test, y_test, data_dir)

    df = pd.DataFrame([xgb_m, lgbm_m])
    csv_path = data_dir / "metrics.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved metrics: {csv_path}")

    print(f"\n{'=' * 50}")
    print("  Summary")
    print(f"{'=' * 50}")
    print(f"  {'Model':<12} {'Accuracy':>8} {'F1':>8} {'AUC-ROC':>8}")
    print(f"  {'-'*40}")
    for m in [xgb_m, lgbm_m]:
        print(f"  {m['model']:<12} {m['accuracy']:>8.4f} {m['f1']:>8.4f} {m['auc_roc']:>8.4f}")
    print(f"\n  All outputs saved to: {data_dir}/")


if __name__ == "__main__":
    main()
