# HARUSPEX Phase 2 — EMBER-Trained Models

Phase 2 trains XGBoost and LightGBM on the EMBER 2018 dataset (the reverse of
Phase 1, which trains on BODMAS). BODMAS is used as the cross-dataset
generalization benchmark.

## Quick Start

### Option A: Local Training (16 GB RAM, subsampled to 200K samples)

```bash
cd backend
source venv/bin/activate   # or .venv

# Step 1: Vectorize EMBER train JSONL → .dat memmaps (run ONCE, ~30-60 min)
python phase2/prepare_ember.py

# Step 2: Train on EMBER, evaluate on BODMAS
python phase2/train_ember.py                      # Default: 200K subsample
python phase2/train_ember.py --max-samples 100000  # Smaller subsample
python phase2/train_ember.py --max-samples 0       # Full dataset (needs ~10GB RAM)
```

### Option B: Kaggle Training (full 800K samples, recommended)

1. Go to [kaggle.com/notebooks](https://www.kaggle.com/notebooks) → "New Notebook"
2. Add the EMBER 2018 dataset to your notebook
3. Copy the contents of `kaggle_notebook.py` into a single code cell
4. Hit **Run All**
5. Download `ember_xgboost.json` and `ember_lightgbm.json` from Kaggle Output
6. Copy them to: `HARUSPEX/outputs/models/`

## Output Files

After training (either local or Kaggle), you'll have:

```
outputs/models/
├── xgboost.json          # Phase 1 (trained on BODMAS)
├── lightgbm.json         # Phase 1 (trained on BODMAS)
├── ember_xgboost.json    # Phase 2 (trained on EMBER)
└── ember_lightgbm.json   # Phase 2 (trained on EMBER)
```

The HARUSPEX API and frontend will automatically detect and load all 4 models,
grouped as "BODMAS Models" and "EMBER Models" in the UI.

## File Overview

| File | Purpose |
|------|---------|
| `prepare_ember.py` | One-time vectorization of EMBER JSONL → .dat memmaps |
| `train_ember.py` | Main training script (local, supports subsampling) |
| `kaggle_notebook.py` | Self-contained Kaggle training script |
| `README.md` | This file |
