"""HARUSPEX ML model wrappers."""

from haruspex.models.base import MalwareDetector
from haruspex.models.xgboost_model import XGBoostDetector
from haruspex.models.lightgbm_model import LightGBMDetector
from haruspex.models.trainer import train_model, train_all_models

__all__ = [
    "MalwareDetector",
    "XGBoostDetector",
    "LightGBMDetector",
    "train_model",
    "train_all_models",
]
