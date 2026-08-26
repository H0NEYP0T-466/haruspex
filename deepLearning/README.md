# HARUSPEX — Deep Learning Extensions

This directory contains Deep Learning extensions for **HARUSPEX** — an explainable AI framework for zero-day malware detection and digital forensic investigation.

The extensions introduce two tabular deep learning models:
1. **TabNet Classifier** (`pytorch-tabnet`): Sequential attention mechanism for tabular feature selection.
2. **Group-wise Residual MLP** (Novel Architecture): Normalizes and fuses 9 static PE feature groups via group projections, multi-head group self-attention, per-sample dynamic group gating, 6 residual blocks, and multi-task auxiliary heads (family & category classification).

---

## Directory Structure

```
deepLearning/
├── phase1/
│   ├── traintabnet.py          # Train TabNet on BODMAS, evaluate on 4 protocols + EMBER
│   ├── trainMLP.py             # Train Group-wise Residual MLP on BODMAS, evaluate on 4 protocols + EMBER
│   └── evaluate.py             # Comparative evaluation + Ablation Study (BODMAS primary)
├── phase2/
│   ├── traintabnet.py          # Train TabNet on EMBER, evaluate on 4 protocols + BODMAS
│   ├── trainMLP.py             # Train Group-wise Residual MLP on EMBER, evaluate on 4 protocols + BODMAS
│   └── evaluate.py             # Comparative evaluation + Ablation Study (EMBER primary)
├── models/
│   ├── tabnet_model.py         # TabNet detector wrapper conforming to MalwareDetector interface
│   └── group_mlp.py            # Group-wise Residual MLP with Projections, Attention, Gating, & Multi-task Heads
├── utils/
│   ├── data.py                 # PyTorch Datasets & loaders for BODMAS (.npz/.dat) & EMBER (.dat)
│   ├── splits.py               # 4 Protocol splits (Random, Family-Held-Out, Temporal, Cross-Dataset)
│   ├── plots.py                # Visualizations (ROC, Confusion Matrix, Gate Weight distributions, Attention Heatmaps)
│   └── config.py               # Single source of truth for hyperparameters, feature groups, device management
├── requirements.txt            # PyTorch, pytorch-tabnet, scikit-learn, pandas, numpy, matplotlib, shap
└── README.md                   # Instructions and architectural guide
```

---

## Model Architecture Details — Group-wise Residual MLP

```
Input Features (2381 dims)
   │
   ├── Group Projection Layer (9 separate projections → 9 × 128 dims = 1152)
   │     • byte_hist (256) → 128
   │     • byte_entropy_hist (256) → 128
   │     • string_info (104) → 128
   │     • general_info (10) → 128
   │     • header_info (62) → 128
   │     • section_info (255) → 128
   │     • import_info (1280) → 128  (Normalizes scale, prevents import_info dominance)
   │     • export_info (128) → 128
   │     • data_dir_info (30) → 128
   │
   ├── Group Attention Layer (Multi-head self-attention over 9 group vectors: 2 heads, d_k=64)
   │     • Learns cross-group relations (e.g. byte entropy + section header anomalies)
   │
   ├── Group Gating Network (MLP: 2381 → 64 → ReLU → 9 → Softmax)
   │     • Dynamic per-sample group weights that down-weight brittle family features on unseen zero-day samples
   │
   ├── Concatenation & Residual Backbone (9 × 128 = 1152 dims)
   │     • 6 × Residual Blocks: Linear(1152 → 1152) → BatchNorm1d → ReLU → Dropout(0.3) + Residual
   │
   └── Multi-Task Output Heads
         • Primary Head (Binary Malware/Benign) — Always active
         • Aux Family Head (582-way) — Training only (forces invariant representation)
         • Aux Category Head (14-way) — Training only
```

---

## Installation & Setup

Ensure PyTorch and dependencies are installed in your virtual environment:

```bash
cd /home/honeypot/Projects/FAST_API/HARUSPEX/backend
source venv/bin/activate
pip install -r ../deepLearning/requirements.txt
```

---

## Execution Guide

### Phase 1: BODMAS Primary

Train and evaluate models trained on the BODMAS dataset (134,435 samples):

```bash
# 1. Train TabNet on BODMAS
python deepLearning/phase1/traintabnet.py

# 2. Train Group-wise Residual MLP on BODMAS
python deepLearning/phase1/trainMLP.py

# 3. Run Comparative Evaluation & Ablation Study
python deepLearning/phase1/evaluate.py
```

Outputs will be generated in:
- `deepLearning/phase1/models/` (`mlp.pth`, `tabnet.zip`)
- `deepLearning/phase1/plots/` (ROC, confusion matrices, gate distributions, attention heatmaps)
- `deepLearning/phase1/reports/` (`evaluation_results.json`, `evaluation_results.txt`)

---

### Phase 2: EMBER Primary

Train and evaluate models trained on the EMBER 2018 dataset (600,000 labeled samples):

```bash
# 1. Train TabNet on EMBER
python deepLearning/phase2/traintabnet.py

# 2. Train Group-wise Residual MLP on EMBER
python deepLearning/phase2/trainMLP.py

# 3. Run Comparative Evaluation
python deepLearning/phase2/evaluate.py
```

Outputs will be generated in `deepLearning/phase2/models/`, `plots/`, and `reports/`.

---

## Evaluation Protocols

Each phase evaluates models on all four standard protocols:
1. **Random Split** (80/20 stratified random split)
2. **Family-Held-Out Split** (Zero-day test on held-out families: `gepys`, `benjamin`, `musecador`, `autoit`)
3. **Temporal Split** (Train on samples before `2020-06-01`, test on samples after)
4. **Cross-Dataset** (Phase 1: BODMAS → EMBER 200K test set | Phase 2: EMBER → BODMAS 134K dataset)
