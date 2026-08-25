"""Explanation endpoint — SHAP/LIME for individual dataset samples.

POST /api/explain       — explain a specific sample by dataset index
POST /api/explain/random — explain a random sample from the active dataset
GET  /api/dataset/info  — dataset summary (accepts ?dataset=bodmas|ember)
GET  /api/dataset/samples — list samples with filtering (accepts ?dataset=bodmas|ember)
"""

from __future__ import annotations

import random as stdlib_random
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from haruspex.config import get_feature_names
from haruspex.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Static dataset metadata — computed once (see scripts/dataset-stats.py)
# and hardcoded here. Never recomputed on every request.
# ---------------------------------------------------------------------------

# EMBER 2018 — full dataset (train + test combined).
# Source: backend/dataset/ember2018/{X,y}_{train,test}.dat
# Train: 800K rows (300K benign, 300K malware, 200K unlabeled=-1)
# Test:  200K rows (100K benign, 100K malware)
EMBER_TOTAL_SAMPLES = 1_000_000
EMBER_N_FEATURES = 2381
EMBER_N_BENIGN = 400_000
EMBER_N_MALWARE = 400_000

# BODMAS — malware dataset.
# Source: backend/dataset/BODMAS/bodmas.npz
BODMAS_TOTAL_SAMPLES = 134_435
BODMAS_N_FEATURES = 2381
BODMAS_N_BENIGN = 77_142
BODMAS_N_MALWARE = 57_293

# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# BODMAS lazy loader — memmap .dat files, never loads 1.2GB into RAM.
# Reads individual rows on demand (~10KB per sample).
# ---------------------------------------------------------------------------

class _BodmasLazy:
    """Lazy BODMAS dataset accessor backed by numpy memmaps.

    - Labels (y) are tiny (~0.5 MB) and loaded once.
    - Metadata comes from CSVs (~3 MB), loaded once.
    - Features (X) stay as a memmap; individual rows read on demand.
    """

    def __init__(self):
        self._X_memmap = None
        self._y = None
        self._metadata = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        from haruspex.config import (
            BACKEND_DIR, NUM_FEATURES,
            BODMAS_METADATA_CSV, BODMAS_CATEGORY_CSV,
        )

        bodmas_dir = BACKEND_DIR / "dataset" / "BODMAS"
        x_dat = bodmas_dir / "bodmas_X.dat"
        y_dat = bodmas_dir / "bodmas_y.dat"

        if not (x_dat.exists() and y_dat.exists()):
            raise FileNotFoundError(
                f"BODMAS .dat files not found in {bodmas_dir}. "
                "Run the conversion: python -c \"import numpy as np; "
                "d=np.load('dataset/BODMAS/bodmas.npz',allow_pickle=True); "
                "d['X'].tofile('dataset/BODMAS/bodmas_X.dat'); "
                "d['y'].tofile('dataset/BODMAS/bodmas_y.dat')\""
            )

        # Labels — tiny, load fully
        self._y = np.fromfile(str(y_dat), dtype=np.int32)

        # Features — memmap, never materialized
        self._X_memmap = np.memmap(
            str(x_dat), dtype=np.float32, mode="r"
        ).reshape(len(self._y), NUM_FEATURES)

        # Metadata — from CSVs (~3 MB)
        metadata = pd.read_csv(str(BODMAS_METADATA_CSV))
        categories = pd.read_csv(str(BODMAS_CATEGORY_CSV))
        metadata = metadata.merge(
            categories, left_on="sha", right_on="sha256", how="left"
        ).drop(columns=["sha256"], errors="ignore")
        metadata["family"] = metadata["family"].fillna("")
        metadata["category"] = metadata["category"].fillna("benign")
        self._metadata = metadata

        self._loaded = True
        logger.info(
            "BODMAS lazy loader ready: %d samples (memmap, no full materialization)",
            len(self._y),
        )

    @property
    def n_samples(self) -> int:
        self._ensure_loaded()
        return len(self._y)

    @property
    def y(self) -> np.ndarray:
        self._ensure_loaded()
        return self._y

    @property
    def metadata(self) -> pd.DataFrame:
        self._ensure_loaded()
        return self._metadata

    def get_sample(self, idx: int) -> np.ndarray:
        """Read a single sample's feature vector from the memmap (~10KB)."""
        self._ensure_loaded()
        return np.asarray(self._X_memmap[idx], dtype=np.float32)

    def get_meta(self, idx: int) -> dict:
        """Get metadata dict for sample at idx."""
        self._ensure_loaded()
        row = self._metadata.iloc[idx]
        return {
            "sha": str(row.get("sha", "")),
            "family": str(row.get("family", "")),
            "category": str(row.get("category", "")),
        }


_bodmas_lazy = _BodmasLazy()


# ---------------------------------------------------------------------------
# EMBER lazy loader — same pattern, memmap .dat files.
# ---------------------------------------------------------------------------

class _EmberLazy:
    """Lazy EMBER dataset accessor backed by numpy memmaps."""

    def __init__(self):
        self._X_memmap = None
        self._y = None
        self._valid_idx = None
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        from haruspex.config import BACKEND_DIR, NUM_FEATURES

        ember_dir = BACKEND_DIR / "dataset" / "ember2018"
        x_dat = ember_dir / "X_test.dat"
        y_dat = ember_dir / "y_test.dat"

        if not (x_dat.exists() and y_dat.exists()):
            raise FileNotFoundError(f"EMBER test .dat files not found in {ember_dir}")

        y_raw = np.memmap(str(y_dat), dtype=np.float32, mode="r")
        self._valid_idx = np.where((y_raw != -1) & (~np.isnan(y_raw)))[0]
        self._y = np.asarray(y_raw[self._valid_idx], dtype=np.int32)

        self._X_memmap = np.memmap(
            str(x_dat), dtype=np.float32, mode="r"
        ).reshape(-1, NUM_FEATURES)

        self._loaded = True
        logger.info(
            "EMBER lazy loader ready: %d labeled samples (memmap, no full materialization)",
            len(self._y),
        )

    @property
    def n_samples(self) -> int:
        self._ensure_loaded()
        return len(self._y)

    @property
    def y(self) -> np.ndarray:
        self._ensure_loaded()
        return self._y

    def get_sample(self, idx: int) -> np.ndarray:
        """Read a single sample's feature vector from the memmap."""
        self._ensure_loaded()
        raw_idx = self._valid_idx[idx]
        return np.asarray(self._X_memmap[raw_idx], dtype=np.float32)

    def get_meta(self, idx: int) -> dict:
        self._ensure_loaded()
        raw_idx = self._valid_idx[idx]
        label = "malware" if self._y[idx] == 1 else "benign"
        return {
            "sha": f"ember_test_{raw_idx}",
            "family": "",
            "category": label,
            "label": label,
            "dataset": "EMBER_2018",
        }

    def get_metadata_df(self) -> pd.DataFrame:
        self._ensure_loaded()
        return pd.DataFrame({
            "sha": [f"ember_test_{i}" for i in self._valid_idx],
            "family": "",
            "category": np.where(self._y == 1, "malware", "benign"),
            "label": np.where(self._y == 1, "malware", "benign"),
            "timestamp": "2018-01-01",
            "dataset": "EMBER_2018",
        })


_ember_lazy = _EmberLazy()


# ---------------------------------------------------------------------------
# Shared explanation pipeline
# ---------------------------------------------------------------------------

def _run_explanation(model, used_model: str, used_group: str, x_sample, feature_names, num_features: int):
    """Run the full explanation pipeline on a single sample vector."""
    from haruspex.xai.shap_explainer import SHAPExplainer
    from haruspex.forensics.static_analysis import StaticAnalyzer
    from haruspex.forensics.ioc_extractor import IOCExtractor
    from haruspex.correlation.correlator import Correlator

    shap_explainer = SHAPExplainer(model, feature_names=feature_names)
    shap_exp = shap_explainer.explain_sample(x_sample)

    analyzer = StaticAnalyzer()
    forensic_report = analyzer.analyze_feature_vector(x_sample, feature_names)

    ioc_extractor = IOCExtractor()
    iocs = ioc_extractor.extract_from_analysis(forensic_report)

    correlator = Correlator()
    evidence = correlator.correlate(shap_exp, forensic_report, iocs, top_k=num_features)

    pred_label = int(model.predict(x_sample.reshape(1, -1))[0])
    pred_proba = float(model.predict_proba(x_sample.reshape(1, -1))[0])
    prediction = "malware" if pred_label == 1 else "benign"
    confidence = pred_proba if pred_label == 1 else (1.0 - pred_proba)

    report = correlator.generate_report(
        sample_id="dataset_sample",
        prediction=prediction,
        confidence=confidence,
        model_name=f"{used_group}/{used_model}",
        evidence_items=evidence,
        shap_explanation=shap_exp,
        forensic_report=forensic_report,
        iocs=iocs,
    )

    return report, prediction, confidence


# ---------------------------------------------------------------------------
# Explain a specific sample
# ---------------------------------------------------------------------------

class ExplainRequest(BaseModel):
    sample_index: int
    model_name: Optional[str] = None
    model_group: Optional[str] = None
    num_features: int = 20


@router.post("/explain")
async def explain_sample(request: ExplainRequest):
    """Explain a specific dataset sample by index."""
    from haruspex.api.main import get_model

    try:
        model, used_model, used_group = get_model(request.model_name, request.model_group)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if used_group == "ember":
        if request.sample_index < 0 or request.sample_index >= _ember_lazy.n_samples:
            raise HTTPException(status_code=400, detail=f"Sample index must be 0-{_ember_lazy.n_samples - 1}")
        idx = request.sample_index
        x_sample = _ember_lazy.get_sample(idx)
        true_label = int(_ember_lazy.y[idx])
        sample_meta = _ember_lazy.get_meta(idx)
    else:
        if request.sample_index < 0 or request.sample_index >= _bodmas_lazy.n_samples:
            raise HTTPException(status_code=400, detail=f"Sample index must be 0-{_bodmas_lazy.n_samples - 1}")
        idx = request.sample_index
        x_sample = _bodmas_lazy.get_sample(idx)
        true_label = int(_bodmas_lazy.y[idx])
        sample_meta = _bodmas_lazy.get_meta(idx)

    feature_names = get_feature_names()
    report, prediction, confidence = _run_explanation(
        model, used_model, used_group, x_sample, feature_names, request.num_features,
    )

    report_dict = report.to_dict()
    report_dict["sample"] = {
        **report_dict.get("sample", {}),
        "sha256": str(sample_meta.get("sha", "")),
        "family": str(sample_meta.get("family", "")),
        "category": str(sample_meta.get("category", "")),
    }

    return {
        **report_dict,
        "true_label": "malware" if true_label == 1 else "benign",
        "sample_index": idx,
    }


# ---------------------------------------------------------------------------
# Explain a random sample
# ---------------------------------------------------------------------------

class RandomExplainRequest(BaseModel):
    model_name: Optional[str] = None
    model_group: Optional[str] = None
    label_filter: Optional[str] = None
    num_features: int = 20


@router.post("/explain/random")
async def explain_random_sample(request: RandomExplainRequest):
    """Explain a random sample from the dataset."""
    from haruspex.api.main import get_model

    try:
        model, used_model, used_group = get_model(request.model_name, request.model_group)
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

    if used_group == "ember":
        y = _ember_lazy.y  # tiny array, already in RAM
        if request.label_filter == "malware":
            valid_indices = np.where(y == 1)[0]
        elif request.label_filter == "benign":
            valid_indices = np.where(y == 0)[0]
        else:
            valid_indices = np.arange(_ember_lazy.n_samples)
    else:
        y = _bodmas_lazy.y
        if request.label_filter == "malware":
            valid_indices = np.where(y == 1)[0]
        elif request.label_filter == "benign":
            valid_indices = np.where(y == 0)[0]
        else:
            valid_indices = np.arange(_bodmas_lazy.n_samples)

    if len(valid_indices) == 0:
        raise HTTPException(status_code=404, detail="No samples match the filter.")

    idx = int(stdlib_random.choice(valid_indices))

    if used_group == "ember":
        x_sample = _ember_lazy.get_sample(idx)  # reads ONE row from memmap (~10KB)
        true_label = int(_ember_lazy.y[idx])
        sample_meta = _ember_lazy.get_meta(idx)
    else:
        x_sample = _bodmas_lazy.get_sample(idx)
        true_label = int(_bodmas_lazy.y[idx])
        sample_meta = _bodmas_lazy.get_meta(idx)

    feature_names = get_feature_names()
    report, prediction, confidence = _run_explanation(
        model, used_model, used_group, x_sample, feature_names, request.num_features,
    )

    report_dict = report.to_dict()
    report_dict["sample"] = {
        **report_dict.get("sample", {}),
        "sha256": str(sample_meta.get("sha", "")),
        "family": str(sample_meta.get("family", "")),
        "category": str(sample_meta.get("category", "")),
    }

    return {
        **report_dict,
        "true_label": "malware" if true_label == 1 else "benign",
        "sample_index": idx,
    }


# ---------------------------------------------------------------------------
# Dataset info — serves hardcoded stats; BODMAS families/categories
# are read from metadata (structural, doesn't change).
# ---------------------------------------------------------------------------

@router.get("/dataset/info")
async def dataset_info(dataset: str = "bodmas"):
    """Return dataset summary information."""
    from haruspex.data.bodmas import get_family_sample_counts

    if dataset == "ember":
        # Static metadata for the full EMBER 2018 dataset.
        # EMBER has no family or category columns.
        return {
            "total_samples": EMBER_TOTAL_SAMPLES,
            "n_features": EMBER_N_FEATURES,
            "n_benign": EMBER_N_BENIGN,
            "n_malware": EMBER_N_MALWARE,
            "n_families": 0,
            "families": [],
            "top_families": [],
            "categories": {},
        }

    # BODMAS — metadata from lazy loader (CSVs only, ~3 MB)
    try:
        metadata = _bodmas_lazy.metadata
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load metadata: {e}")

    families = metadata["family"].replace("", None).dropna().unique()
    family_counts = get_family_sample_counts(metadata)
    categories = metadata["category"].value_counts().to_dict()

    return {
        "total_samples": BODMAS_TOTAL_SAMPLES,
        "n_features": BODMAS_N_FEATURES,
        "n_benign": BODMAS_N_BENIGN,
        "n_malware": BODMAS_N_MALWARE,
        "n_families": len(families),
        "families": sorted(families.tolist())[:50],
        "top_families": [
            {"name": name, "count": int(count)}
            for name, count in family_counts.head(10).items()
        ],
        "categories": categories,
    }


# ---------------------------------------------------------------------------
# List samples with pagination/filtering
# ---------------------------------------------------------------------------

@router.get("/dataset/samples")
async def list_samples(
    family: Optional[str] = None,
    category: Optional[str] = None,
    label: Optional[str] = None,
    offset: int = 0,
    limit: int = 50,
    dataset: str = "bodmas",
):
    """List samples with optional filtering."""
    if dataset == "ember":
        df = _ember_lazy.get_metadata_df()  # built from tiny label array, no feature data
        df["index"] = range(len(df))
        if label:
            df = df[df["label"] == label]
        total = len(df)
        df = df.iloc[offset:offset + limit]
        samples = []
        for _, row in df.iterrows():
            samples.append({
                "index": int(row["index"]),
                "sha256": str(row.get("sha", "")),
                "family": str(row.get("family", "")),
                "category": str(row.get("category", "")),
                "label": row["label"],
                "timestamp": str(row.get("timestamp", "")),
            })
        return {"total": total, "offset": offset, "limit": limit, "samples": samples}

    # BODMAS — use lazy loader metadata (no 1.2GB feature matrix)
    meta = _bodmas_lazy.metadata.copy()
    meta["index"] = range(len(meta))
    if "label" not in meta.columns:
        meta["label"] = meta["category"].apply(lambda c: "benign" if c == "benign" else "malware")

    if family:
        meta = meta[meta["family"] == family]
    if category:
        meta = meta[meta["category"] == category]
    if label:
        meta = meta[meta["label"] == label]

    total = len(meta)
    meta = meta.iloc[offset:offset + limit]

    samples = []
    for _, row in meta.iterrows():
        samples.append({
            "index": int(row["index"]),
            "sha256": str(row.get("sha", "")),
            "family": str(row.get("family", "")),
            "category": str(row.get("category", "")),
            "label": row["label"],
            "timestamp": str(row.get("timestamp", "")),
        })

    return {"total": total, "offset": offset, "limit": limit, "samples": samples}