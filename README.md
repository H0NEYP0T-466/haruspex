<p align="center">

  <!-- Core -->
  ![GitHub License](https://img.shields.io/github/license/H0NEYP0T-466/haruspex?style=for-the-badge&color=brightgreen)
  ![GitHub Stars](https://img.shields.io/github/stars/H0NEYP0T-466/haruspex?style=for-the-badge&color=yellow)
  ![GitHub Forks](https://img.shields.io/github/forks/H0NEYP0T-466/haruspex?style=for-the-badge&color=blue)
  ![GitHub Issues](https://img.shields.io/github/issues/H0NEYP0T-466/haruspex?style=for-the-badge&color=red)
  ![GitHub Pull Requests](https://img.shields.io/github/issues-pr/H0NEYP0T-466/haruspex?style=for-the-badge&color=orange)
  ![Contributions Welcome](https://img.shields.io/badge/Contributions-Welcome-brightgreen?style=for-the-badge)

  <!-- Activity -->
  ![Last Commit](https://img.shields.io/github/last-commit/H0NEYP0T-466/haruspex?style=for-the-badge&color=purple)
  ![Commit Activity](https://img.shields.io/github/commit-activity/m/H0NEYP0T-466/haruspex?style=for-the-badge&color=teal)
  ![Repo Size](https://img.shields.io/github/repo-size/H0NEYP0T-466/haruspex?style=for-the-badge&color=blueviolet)
  ![Code Size](https://img.shields.io/github/languages/code-size/H0NEYP0T-466/haruspex?style=for-the-badge&color=indigo)

  <!-- Languages -->
  ![Top Language](https://img.shields.io/github/languages/top/H0NEYP0T-466/haruspex?style=for-the-badge&color=critical)
  ![Languages Count](https://img.shields.io/github/languages/count/H0NEYP0T-466/haruspex?style=for-the-badge&color=success)

  <!-- Community -->
  ![Discussions](https://img.shields.io/github/discussions/H0NEYP0T-466/haruspex?style=for-the-badge&color=blue)
  ![Documentation](https://img.shields.io/badge/Docs-Available-green?style=for-the-badge&logo=readthedocs&logoColor=white)
  ![Open Source Love](https://img.shields.io/badge/Open%20Source-%E2%9D%A4-red?style=for-the-badge)

</p>

<h1 align="center">HARUSPEX</h1>

<p align="center">
  <strong>Explainable AI Framework for Zero-Day Malware Detection & Digital Forensic Investigation</strong>
</p>

<p align="center">
  HARUSPEX bridges the gap between AI model explanations and forensic evidence by correlating SHAP/LIME feature attributions with PE analysis findings into unified investigation reports.
</p>

---

## Table of Contents

- [Architecture](#architecture)
- [Features](#features)
- [Demo / Links](#-demo--links)
- [Installation](#-installation)
- [Usage](#-usage)
- [Evaluation Protocols](#evaluation-protocols)
- [API Endpoints](#api-endpoints)
- [Results — All 8 Models](#-results--all-8-models)
  - [Summary Table](#summary-table)
  - [BODMAS-Trained Models](#bodmas-trained-models)
  - [EMBER 2018-Trained Models](#ember-2018-trained-models)
  - [Zero-Day Family-Held-Out Results](#-zero-day-family-held-out-results)
- [Plots & Visualizations](#-plots--visualizations)
- [Folder Structure](#-folder-structure)
- [Tech Stack](#-tech-stack)
- [Dependencies & Packages](#-dependencies--packages)
- [Testing](#testing)
- [Contributing](#-contributing)
- [License](#-license)
- [Security](#-security)
- [Code of Conduct](#-code-of-conduct)

---

## Architecture

```
Input (PE file / Feature vector)
         │
    ┌────┴────┐
    │  Models  │  XGBoost / LightGBM / Group-MLP / TabNet (2381 EMBER features)
    └────┬────┘
         │
   ┌─────┴─────┐
   │  XAI Layer │  TreeSHAP + LIME explanations
   └─────┬─────┘
         │              ┌──────────────────┐
         ├──────────────│  Forensics Layer  │  LIEF PE parsing + YARA + IOCs
         │              └────────┬─────────┘
   ┌─────┴─────┐                │
   │ Correlator │ ◄─────────────┘   ★ Core contribution
   └─────┬─────┘
         │
   ┌─────┴──────┐
   │   Report    │  Unified investigation output
   └────────────┘
```

---

## Features

- **Multi-Model Ensemble** — XGBoost, LightGBM, Group-MLP, and TabNet classifiers
- **Explainable AI** — TreeSHAP global/local explanations + LIME local interpretable explanations
- **Forensic Correlation** — Core innovation: correlates AI feature attributions with PE static analysis (LIEF), YARA rules, and IOC extraction into unified severity-scored reports
- **4 Evaluation Protocols** — Random, Family-Held-Out (zero-day), Temporal, and Cross-Dataset generalization
- **Zero-Day Detection** — Holds out entire malware families to simulate truly unseen threats
- **REST API** — FastAPI backend with prediction, explanation, and dataset browsing endpoints
- **React Dashboard** — Modern TypeScript frontend with interactive visualizations
- **PE File Upload** — Accepts raw PE binaries for on-demand analysis
- **YARA Rule Engine** — 5 rule categories: anti-debug, crypto indicators, dropper indicators, packed binary, suspicious imports
- **EMBER-Compatible** — Uses 2381 EMBER feature vectors for standard PE representation

---

## Demo / Links

| Resource | Link |
|----------|------|
| Repository | [github.com/H0NEYP0T-466/haruspex](https://github.com/H0NEYP0T-466/haruspex) |
| Issues | [Report a Bug](https://github.com/H0NEYP0T-466/haruspex/issues) |
| Contributing | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Security Policy | [SECURITY.md](SECURITY.md) |

---

## Installation

### Prerequisites

- **Python** 3.10+
- **Node.js** 18+ (for frontend)
- **BODMAS Dataset** (134K samples, 2381 EMBER-compatible features)
- Optional: **EMBER 2018** dataset for cross-dataset evaluation

### Backend Setup

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env

# Train models (requires BODMAS dataset in backend/dataset/BODMAS/)
python scripts/train.py

# (Optional) Vectorize EMBER 2018 for cross-dataset evaluation
python scripts/prepare_ember.py
python scripts/evaluate.py

# (Optional) Run zero-day family-held-out evaluation
cd haruspex/zero_day
python familyout.py

# Start API server
python -m haruspex.api.main
```

### Deep Learning Setup (Optional)

```bash
cd deepLearning
pip install -r requirements.txt

# Phase 1: Train on BODMAS
python phase1/trainMLP.py
python phase1/traintabnet.py

# Phase 2: Train on EMBER 2018
python phase2/trainMLP.py
python phase2/traintabnet.py
```

### Frontend Setup

```bash
# From project root
npm install
npm run dev
```

The frontend runs on `http://localhost:5173` and proxies API requests to `http://localhost:8014`.

---

## Usage

### CLI — Training

```bash
cd backend
source .venv/bin/activate
python scripts/train.py          # Train XGBoost + LightGBM on BODMAS
python scripts/evaluate.py       # Evaluate on all 4 protocols
```

### CLI — Zero-Day Evaluation

```bash
cd backend/haruspex/zero_day
python familyout.py              # Family-held-out zero-day evaluation
```

### API — Prediction

```bash
# Upload PE file for full investigation pipeline
curl -X POST http://localhost:8014/api/predict \
  -F "file=@sample.exe"

# Predict from feature vector
curl -X POST http://localhost:8014/api/predict/vector \
  -H "Content-Type: application/json" \
  -d '{"features": [0.1, 0.2, ...]}'

# Run all loaded models on same sample
curl -X POST http://localhost:8014/api/predict/all \
  -F "file=@sample.exe"

# Explain a BODMAS sample by index
curl -X POST http://localhost:8014/api/explain \
  -H "Content-Type: application/json" \
  -d '{"sample_index": 42}'

# Health check
curl http://localhost:8014/api/health
```

### Frontend

```bash
npm run dev    # Development server at http://localhost:5173
npm run build  # Production build
```

---

## Evaluation Protocols

| Protocol | Description | Split Strategy |
|----------|-------------|----------------|
| **Random** | Standard 80/20 stratified split | Sanity-check baseline |
| **Family-Held-Out** | Hold out entire malware families (gepys, benjamin, musecador, autoit) | Zero-day (family-based) |
| **Temporal** | Train before cutoff (2020-06-01), test after | Zero-day (time-based) |
| **Cross-Dataset** | Train BODMAS → test EMBER (200K samples) | Cross-domain generalization |

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/predict` | Full investigation pipeline (PE file or feature vector) |
| `POST` | `/api/predict/vector` | Predict from JSON feature vector |
| `POST` | `/api/predict/all` | Run all loaded models on the same uploaded PE/vector sample |
| `POST` | `/api/explain` | Explain a BODMAS sample by index |
| `GET` | `/api/models` | List available models |
| `GET` | `/api/health` | API health check |
| `GET` | `/api/dataset/info` | Dataset statistics |
| `GET` | `/api/dataset/samples` | Browse samples with filtering |

---

## Results — All 8 Models

HARUSPEX trains and evaluates **8 models** across **two training datasets** and **four evaluation protocols**.

### Summary Table

#### BODMAS-Trained Models (4)

| Model | Random Acc | Family-Held-Out Acc | Temporal Acc | Cross-Dataset (EMBER) Acc |
|-------|------------|---------------------|--------------|---------------------------|
| **XGBoost** | 99.70% | 99.98% | 99.95% | 84.17% |
| **LightGBM** | 99.72% | 99.98% | 99.96% | 84.00% |
| **Group-MLP** | 99.55% | 100.00% | 99.95% | 81.35% |
| **TabNet** | 99.46% | 99.91% | 99.81% | 80.72% |

#### EMBER 2018-Trained Models (4)

| Model | EMBER Test Acc | Cross-Dataset (BODMAS) Acc | EMBER AUC | BODMAS AUC |
|-------|----------------|---------------------------|-----------|------------|
| **XGBoost** | 96.10% | 96.11% | 99.27% | 99.60% |
| **LightGBM** | 95.95% | 96.17% | 99.19% | 99.62% |
| **Group-MLP** | 93.71% | 97.53% | 97.53% | 99.40% |
| **TabNet** | 94.84% | 66.08% | 98.49% | 81.60% |

---

### BODMAS-Trained Models

#### 1. XGBoost (BODMAS)

| Protocol | Accuracy | Precision | Recall | F1 | AUC-ROC |
|----------|----------|-----------|--------|-----|---------|
| Random | 99.70% | 99.79% | 99.51% | 99.65% | 99.98% |
| Family-Held-Out | 99.98% | 100.00% | 99.98% | 99.99% | N/A |
| Temporal | 99.95% | 99.98% | 99.91% | 99.95% | 100.00% |
| Cross-Dataset (EMBER) | 84.17% | 94.13% | 72.88% | 82.15% | 92.56% |

#### 2. LightGBM (BODMAS)

| Protocol | Accuracy | Precision | Recall | F1 | AUC-ROC |
|----------|----------|-----------|--------|-----|---------|
| Random | 99.72% | 99.83% | 99.50% | 99.67% | 99.99% |
| Family-Held-Out | 99.98% | 100.00% | 99.98% | 99.99% | N/A |
| Temporal | 99.96% | 99.98% | 99.94% | 99.96% | 100.00% |
| Cross-Dataset (EMBER) | 84.00% | 94.64% | 72.08% | 81.83% | 92.57% |

#### 3. Group-MLP (BODMAS)

| Protocol | Accuracy | Precision | Recall | F1 | AUC |
|----------|----------|-----------|--------|-----|-----|
| Random | 99.55% | 99.51% | 99.43% | 99.47% | 99.93% |
| Family-Held-Out | 100.00% | 100.00% | 100.00% | 100.00% | N/A |
| Temporal | 99.95% | 99.97% | 99.92% | 99.94% | 99.99% |
| Cross-Dataset (EMBER) | 81.35% | 94.58% | 66.52% | 78.11% | 87.15% |

#### 4. TabNet (BODMAS)

| Protocol | Accuracy | Precision | Recall | F1 | AUC |
|----------|----------|-----------|--------|-----|-----|
| Random | 99.46% | 99.48% | 99.26% | 99.37% | 99.85% |
| Family-Held-Out | 99.91% | 100.00% | 99.91% | 99.95% | N/A |
| Temporal | 99.81% | 99.82% | 99.76% | 99.79% | 99.96% |
| Cross-Dataset (EMBER) | 80.72% | 91.15% | 68.05% | 77.92% | 86.20% |

---

### EMBER 2018-Trained Models

#### 5. XGBoost (EMBER 2018)

| Protocol | Accuracy | F1 | AUC-ROC |
|----------|----------|-----|---------|
| EMBER Test (200K) | 96.10% | 96.12% | 99.27% |
| EMBER Internal (120K) | 97.21% | 97.21% | 99.64% |
| Cross-Dataset (BODMAS) | 96.11% | 95.37% | 99.60% |

#### 6. LightGBM (EMBER 2018)

| Protocol | Accuracy | F1 | AUC-ROC |
|----------|----------|-----|---------|
| EMBER Test (200K) | 95.95% | 95.98% | 99.19% |
| EMBER Internal (120K) | 96.90% | 96.89% | 99.57% |
| Cross-Dataset (BODMAS) | 96.17% | 95.42% | 99.62% |

#### 7. Group-MLP (EMBER 2018)

| Protocol | Accuracy | Precision | Recall | F1 | AUC |
|----------|----------|-----------|--------|-----|-----|
| EMBER Test (200K) | 93.71% | 95.60% | 91.64% | 93.58% | 97.53% |
| Cross-Dataset (BODMAS) | 97.53% | 96.92% | 97.30% | 97.11% | 99.40% |

#### 8. TabNet (EMBER 2018)

| Protocol | Accuracy | Precision | Recall | F1 | AUC |
|----------|----------|-----------|--------|-----|-----|
| EMBER Test (200K) | 94.84% | 94.78% | 94.90% | 94.84% | 98.49% |
| Cross-Dataset (BODMAS) | 66.08% | 56.64% | 87.03% | 68.62% | 81.60% |

---

### Zero-Day Family-Held-Out Results

The most critical evaluation: models trained on known families and tested on **4 held-out families** never seen during training — simulating true zero-day malware detection.

#### Dataset Details

| Metric | Value |
|--------|-------|
| Total samples | 134,435 |
| Features | 2,381 (EMBER-compatible) |
| Benign | 77,142 |
| Malware | 57,293 |
| Families | 583 |
| Held-out families | `gepys`, `benjamin`, `musecador`, `autoit` |
| Train set | 130,217 (53,075 malware + 77,142 benign) |
| Test set | 81,360 (77,142 benign + 4,218 malware) |

#### XGBoost vs LightGBM — Family-Held-Out

| Model | Accuracy | Precision | Recall | F1 Score | AUC-ROC | Malware Missed |
|-------|----------|-----------|--------|----------|---------|----------------|
| XGBoost | 99.87% | 100.00% | 97.51% | 98.74% | 100.00% | 106 / 4,218 |
| **LightGBM** | **100.00%** | **100.00%** | **99.91%** | **99.95%** | **100.00%** | **4 / 4,218** |

#### Confusion Matrices — Family-Held-Out

**XGBoost Confusion Matrix:**
![XGBoost Confusion Matrix — Family-Held-Out](backend/haruspex/zero_day/data/xgboost_confusion.png)

**LightGBM Confusion Matrix:**
![LightGBM Confusion Matrix — Family-Held-Out](backend/haruspex/zero_day/data/lightgbm_confusion.png)

#### ROC Curves — Family-Held-Out

**XGBoost ROC:**
![XGBoost ROC — Family-Held-Out](backend/haruspex/zero_day/data/xgboost_roc.png)

**LightGBM ROC:**
![LightGBM ROC — Family-Held-Out](backend/haruspex/zero_day/data/lightgbm_roc.png)

---

## Plots & Visualizations

### BODMAS-Trained XGBoost Plots

| Plot | Link |
|------|------|
| Random Confusion Matrix | ![](outputs/plots/xgboost_random_confusion_matrix.png) |
| Random ROC | ![](outputs/plots/xgboost_random_roc_curve.png) |
| Family-Held-Out Confusion Matrix | ![](outputs/plots/xgboost_family_held_out_confusion_matrix.png) |
| Family-Held-Out ROC | ![](outputs/plots/xgboost_family_held_out_roc_curve.png) |
| Temporal Confusion Matrix | ![](outputs/plots/xgboost_temporal_confusion_matrix.png) |
| Temporal ROC | ![](outputs/plots/xgboost_temporal_roc_curve.png) |
| Cross-Dataset Confusion Matrix | ![](outputs/plots/xgboost_cross_dataset_confusion_matrix.png) |
| Cross-Dataset ROC | ![](outputs/plots/xgboost_cross_dataset_roc_curve.png) |
| Feature Importance | ![](outputs/plots/xgboost_feature_importance.png) |

### BODMAS-Trained LightGBM Plots

| Plot | Link |
|------|------|
| Random Confusion Matrix | ![](outputs/plots/lightgbm_random_confusion_matrix.png) |
| Random ROC | ![](outputs/plots/lightgbm_random_roc_curve.png) |
| Family-Held-Out Confusion Matrix | ![](outputs/plots/lightgbm_family_held_out_confusion_matrix.png) |
| Family-Held-Out ROC | ![](outputs/plots/lightgbm_family_held_out_roc_curve.png) |
| Temporal Confusion Matrix | ![](outputs/plots/lightgbm_temporal_confusion_matrix.png) |
| Temporal ROC | ![](outputs/plots/lightgbm_temporal_roc_curve.png) |
| Cross-Dataset Confusion Matrix | ![](outputs/plots/lightgbm_cross_dataset_confusion_matrix.png) |
| Cross-Dataset ROC | ![](outputs/plots/lightgbm_cross_dataset_roc_curve.png) |
| Feature Importance | ![](outputs/plots/lightgbm_feature_importance.png) |

### BODMAS-Trained Group-MLP Plots

| Plot | Link |
|------|------|
| Random Confusion Matrix | ![](deepLearning/phase1/plots/mlp_random_cm.png) |
| Random ROC | ![](deepLearning/phase1/plots/mlp_random_roc.png) |
| Temporal Confusion Matrix | ![](deepLearning/phase1/plots/mlp_temporal_cm.png) |
| Temporal ROC | ![](deepLearning/phase1/plots/mlp_temporal_roc.png) |
| Family-Held-Out Confusion Matrix | ![](deepLearning/phase1/plots/mlp_family_cm.png) |
| Cross-Dataset Confusion Matrix | ![](deepLearning/phase1/plots/mlp_cross_cm.png) |
| Cross-Dataset ROC | ![](deepLearning/phase1/plots/mlp_cross_roc.png) |
| Attention Heatmap | ![](deepLearning/phase1/plots/mlp_attention_heatmap.png) |
| Gate Distribution | ![](deepLearning/phase1/plots/mlp_gate_distribution.png) |
| Gate Profiles (Family) | ![](deepLearning/phase1/plots/mlp_family_gate_profiles.png) |

### BODMAS-Trained TabNet Plots

| Plot | Link |
|------|------|
| Random Confusion Matrix | ![](deepLearning/phase1/plots/tabnet_random_cm.png) |
| Random ROC | ![](deepLearning/phase1/plots/tabnet_random_roc.png) |
| Temporal Confusion Matrix | ![](deepLearning/phase1/plots/tabnet_temporal_cm.png) |
| Temporal ROC | ![](deepLearning/phase1/plots/tabnet_temporal_roc.png) |
| Family-Held-Out Confusion Matrix | ![](deepLearning/phase1/plots/tabnet_family_cm.png) |
| Cross-Dataset Confusion Matrix | ![](deepLearning/phase1/plots/tabnet_cross_cm.png) |
| Cross-Dataset ROC | ![](deepLearning/phase1/plots/tabnet_cross_roc.png) |
| Feature Importance | ![](deepLearning/phase1/plots/tabnet_feature_importance.png) |

### EMBER 2018-Trained XGBoost Plots

| Plot | Link |
|------|------|
| EMBER Test Confusion Matrix | ![](outputs/plots/ember_xgboost_ember_test_confusion_matrix.png) |
| EMBER Test ROC | ![](outputs/plots/ember_xgboost_ember_test_roc_curve.png) |
| BODMAS Generalization Confusion Matrix | ![](outputs/plots/ember_xgboost_bodmas_generalization_confusion_matrix.png) |
| BODMAS Generalization ROC | ![](outputs/plots/ember_xgboost_bodmas_generalization_roc_curve.png) |
| Feature Importance | ![](outputs/plots/ember_xgboost_feature_importance.png) |

### EMBER 2018-Trained LightGBM Plots

| Plot | Link |
|------|------|
| EMBER Test Confusion Matrix | ![](outputs/plots/ember_lightgbm_ember_test_confusion_matrix.png) |
| EMBER Test ROC | ![](outputs/plots/ember_lightgbm_ember_test_roc_curve.png) |
| BODMAS Generalization Confusion Matrix | ![](outputs/plots/ember_lightgbm_bodmas_generalization_confusion_matrix.png) |
| BODMAS Generalization ROC | ![](outputs/plots/ember_lightgbm_bodmas_generalization_roc_curve.png) |
| Feature Importance | ![](outputs/plots/ember_lightgbm_feature_importance.png) |

### EMBER 2018-Trained Group-MLP Plots

| Plot | Link |
|------|------|
| EMBER Test Confusion Matrix | ![](deepLearning/phase2/plots/ember_mlp_test_cm.png) |
| EMBER Test ROC | ![](deepLearning/phase2/plots/ember_mlp_test_roc.png) |
| Cross-BODMAS Confusion Matrix | ![](deepLearning/phase2/plots/ember_mlp_cross_bodmas_cm.png) |
| Cross-BODMAS ROC | ![](deepLearning/phase2/plots/ember_mlp_cross_bodmas_roc.png) |
| Attention Heatmap | ![](deepLearning/phase2/plots/ember_mlp_attention_heatmap.png) |
| Gate Distribution | ![](deepLearning/phase2/plots/ember_mlp_gate_distribution.png) |

### EMBER 2018-Trained TabNet Plots

| Plot | Link |
|------|------|
| EMBER Test Confusion Matrix | ![](deepLearning/phase2/plots/ember_tabnet_test_cm.png) |
| EMBER Test ROC | ![](deepLearning/phase2/plots/ember_tabnet_test_roc.png) |
| Cross-BODMAS Confusion Matrix | ![](deepLearning/phase2/plots/ember_tabnet_cross_bodmas_cm.png) |
| Cross-BODMAS ROC | ![](deepLearning/phase2/plots/ember_tabnet_cross_bodmas_roc.png) |
| Feature Importance | ![](deepLearning/phase2/plots/ember_tabnet_feature_importance.png) |

### XAI Plots

| Plot | Link |
|------|------|
| SHAP Summary | ![](outputs/plots/shap_summary.png) |
| Phase 2 SHAP Summary | ![](outputs/plots/phase2_shap_summary.png) |
| LIME Sample 0 | ![](outputs/plots/lime_sample_0.png) |
| LIME Sample 1 | ![](outputs/plots/lime_sample_1.png) |
| LIME Sample 2 | ![](outputs/plots/lime_sample_2.png) |
| LIME Sample 3 | ![](outputs/plots/lime_sample_3.png) |
| LIME Sample 4 | ![](outputs/plots/lime_sample_4.png) |

---

## Folder Structure

```
HARUSPEX/
├── src/                              # React+TypeScript frontend
│   ├── pages/                        # Dashboard, Explorer, Upload pages
│   ├── services/api.ts               # API client
│   └── assets/hero.png               # Hero image
├── public/                           # Static assets (favicon, icons)
├── backend/
│   ├── haruspex/                     # Python package
│   │   ├── config.py                 # Paths, hyperparameters, feature names
│   │   ├── api/                      # FastAPI endpoints (predict, explain, models)
│   │   ├── data/                     # BODMAS loader + split strategies
│   │   ├── models/                   # XGBoost + LightGBM wrappers
│   │   ├── xai/                      # SHAP + LIME explainers
│   │   ├── forensics/                # Static PE analysis + YARA rules + IOCs
│   │   ├── correlation/              # ★ Evidence correlation engine (core contribution)
│   │   ├── evaluation/               # Metrics + 4 evaluation protocols
│   │   ├── zero_day/                 # Family-held-out zero-day evaluation
│   │   │   ├── familyout.py          # Zero-day evaluation script
│   │   │   ├── logs.txt              # Evaluation logs
│   │   │   └── data/                 # Models, plots, metrics for zero-day
│   │   └── utils/                    # Logging + storage
│   ├── scripts/                      # Training, evaluation, dataset prep
│   ├── tests/                        # Unit + integration tests
│   ├── phase2/                       # EMBER 2018 training scripts
│   ├── dataset/                      # BODMAS + EMBER raw data
│   ├── requirements.txt
│   └── pyproject.toml
├── deepLearning/                     # Deep learning models
│   ├── phase1/                       # BODMAS-trained MLP + TabNet
│   │   ├── plots/                    # 18 evaluation plots
│   │   └── reports/                  # Result JSONs + logs
│   ├── phase2/                       # EMBER-trained MLP + TabNet
│   │   ├── plots/                    # 11 evaluation plots
│   │   └── reports/                  # Result JSONs + logs
│   ├── models/                       # Shared model architectures
│   └── utils/                        # Config, data loaders, plotting
├── outputs/                          # Generated by train.py
│   ├── models/                       # Saved model files
│   ├── plots/                        # 35 evaluation + XAI plots
│   └── reports/                      # Investigation reports (JSON + TXT)
├── .github/                          # GitHub templates & workflows
├── README.md
├── package.json
├── vite.config.ts
└── tsconfig.json
```

---

## Tech Stack

### Languages

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=for-the-badge&logo=typescript&logoColor=white)

### Frameworks & Libraries

![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![React](https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB)
![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-FF6F00?style=for-the-badge&logo=xgboost&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-FF9800?style=for-the-badge&logo=lightgbm&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-F05033?style=for-the-badge&logo=shap&logoColor=white)
![LIME](https://img.shields.io/badge/LIME-FF5722?style=for-the-badge)
![Recharts](https://img.shields.io/badge/Recharts-333333?style=for-the-badge&logo=react&logoColor=white)

### Forensics & Security

![LIEF](https://img.shields.io/badge/LIEF-4CAF50?style=for-the-badge)
![YARA](https://img.shields.io/badge/YARA-F44336?style=for-the-badge&logo=yara&logoColor=white)

### Databases & Data

![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-FFDD00?style=for-the-badge&logo=matplotlib&logoColor=black)

### DevOps / CI / Tools

![Vite](https://img.shields.io/badge/Vite-646CFF?style=for-the-badge&logo=vite&logoColor=white)
![Ruff](https://img.shields.io/badge/Ruff-F7D44C?style=for-the-badge&logo=ruff&logoColor=black)
![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)
![Oxlint](https://img.shields.io/badge/Oxlint-4D4D4D?style=for-the-badge)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)

---

## Dependencies & Packages

### Backend — Runtime Dependencies

<details>
<summary><strong>Web API</strong></summary>

[![fastapi](https://img.shields.io/pypi/v/fastapi?style=for-the-badge&label=fastapi)](https://pypi.org/project/fastapi/) — High-performance async web framework

[![uvicorn](https://img.shields.io/pypi/v/uvicorn?style=for-the-badge&label=uvicorn)](https://pypi.org/project/uvicorn/) — ASGI server

[![pydantic](https://img.shields.io/pypi/v/pydantic?style=for-the-badge&label=pydantic)](https://pypi.org/project/pydantic/) — Data validation

[![python-multipart](https://img.shields.io/pypi/v/python-multipart?style=for-the-badge&label=python-multipart)](https://pypi.org/project/python-multipart/) — File upload support

[![python-dotenv](https://img.shields.io/pypi/v/python-dotenv?style=for-the-badge&label=python-dotenv)](https://pypi.org/project/python-dotenv/) — Environment variable management

</details>

<details>
<summary><strong>Machine Learning</strong></summary>

[![xgboost](https://img.shields.io/pypi/v/xgboost?style=for-the-badge&label=xgboost)](https://pypi.org/project/xgboost/) — Gradient boosting framework

[![lightgbm](https://img.shields.io/pypi/v/lightgbm?style=for-the-badge&label=lightgbm)](https://pypi.org/project/lightgbm/) — Gradient boosting framework

[![scikit-learn](https://img.shields.io/pypi/v/scikit-learn?style=for-the-badge&label=scikit-learn)](https://pypi.org/project/scikit-learn/) — ML utilities

</details>

<details>
<summary><strong>Explainable AI (XAI)</strong></summary>

[![shap](https://img.shields.io/pypi/v/shap?style=for-the-badge&label=shap)](https://pypi.org/project/shap/) — SHAP (SHapley Additive exPlanations)

[![lime](https://img.shields.io/pypi/v/lime?style=for-the-badge&label=lime)](https://pypi.org/project/lime/) — Local Interpretable Model-agnostic Explanations

</details>

<details>
<summary><strong>Data & Visualization</strong></summary>

[![pandas](https://img.shields.io/pypi/v/pandas?style=for-the-badge&label=pandas)](https://pypi.org/project/pandas/) — Data manipulation

[![numpy](https://img.shields.io/pypi/v/numpy?style=for-the-badge&label=numpy)](https://pypi.org/project/numpy/) — Numerical computing

[![matplotlib](https://img.shields.io/pypi/v/matplotlib?style=for-the-badge&label=matplotlib)](https://pypi.org/project/matplotlib/) — Plotting

</details>

<details>
<summary><strong>Forensics</strong></summary>

[![lief](https://img.shields.io/pypi/v/lief?style=for-the-badge&label=lief)](https://pypi.org/project/lief/) — PE/ELF/MachO parser

[![yara-python](https://img.shields.io/pypi/v/yara-python?style=for-the-badge&label=yara-python)](https://pypi.org/project/yara-python/) — YARA rule engine

[![ember](https://img.shields.io/pypi/v/ember?style=for-the-badge&label=ember)](https://pypi.org/project/ember/) — EMBER v2 feature vectorizer

</details>

### Backend — Dev Dependencies

<details>
<summary><strong>Testing & Tooling</strong></summary>

[![pytest](https://img.shields.io/pypi/v/pytest?style=for-the-badge&label=pytest)](https://pypi.org/project/pytest/) — Test framework

[![httpx](https://img.shields.io/pypi/v/httpx?style=for-the-badge&label=httpx)](https://pypi.org/project/httpx/) — HTTP client (FastAPI TestClient)

[![ruff](https://img.shields.io/pypi/v/ruff?style=for-the-badge&label=ruff)](https://pypi.org/project/ruff/) — Python linter + formatter

</details>

### Deep Learning — Runtime Dependencies

<details>
<summary><strong>Deep Learning</strong></summary>

[![torch](https://img.shields.io/pypi/v/torch?style=for-the-badge&label=torch)](https://pypi.org/project/torch/) — PyTorch deep learning framework

[![pytorch-tabnet](https://img.shields.io/pypi/v/pytorch-tabnet?style=for-the-badge&label=pytorch-tabnet)](https://pypi.org/project/pytorch-tabnet/) — TabNet implementation for PyTorch

</details>

### Frontend — Runtime Dependencies

<details>
<summary><strong>Frontend Packages</strong></summary>

[![react](https://img.shields.io/npm/v/react?style=for-the-badge&label=react)](https://www.npmjs.com/package/react) — UI library

[![react-dom](https://img.shields.io/npm/v/react-dom?style=for-the-badge&label=react-dom)](https://www.npmjs.com/package/react-dom) — React DOM renderer

[![react-router-dom](https://img.shields.io/npm/v/react-router-dom?style=for-the-badge&label=react-router-dom)](https://www.npmjs.com/package/react-router-dom) — Client-side routing

[![recharts](https://img.shields.io/npm/v/recharts?style=for-the-badge&label=recharts)](https://www.npmjs.com/package/recharts) — Charting library

</details>

### Frontend — Dev Dependencies

<details>
<summary><strong>Build & Dev Tools</strong></summary>

[![vite](https://img.shields.io/npm/v/vite?style=for-the-badge&label=vite)](https://www.npmjs.com/package/vite) — Build tool & dev server

[![typescript](https://img.shields.io/npm/v/typescript?style=for-the-badge&label=typescript)](https://www.npmjs.com/package/typescript) — TypeScript compiler

[![oxlint](https://img.shields.io/npm/v/oxlint?style=for-the-badge&label=oxlint)](https://www.npmjs.com/package/oxlint) — Fast linter

[![@vitejs/plugin-react](https://img.shields.io/npm/v/@vitejs/plugin-react?style=for-the-badge&label=@vitejs/plugin-react)](https://www.npmjs.com/package/@vitejs/plugin-react) — Vite React plugin

</details>

---

## Testing

```bash
cd backend
source .venv/bin/activate

# Run all tests
pytest tests/ -v

# Run specific test modules
pytest tests/unit/test_models.py -v
pytest tests/unit/test_forensics.py -v
pytest tests/unit/test_correlator.py -v
pytest tests/integration/test_full_pipeline.py -v
```

### Linting

```bash
# Python (backend)
cd backend
ruff check .
ruff format .

# TypeScript (frontend)
npm run lint
```

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Security

To report security vulnerabilities, please see [SECURITY.md](SECURITY.md).

---

## Code of Conduct

This project follows the Contributor Covenant Code of Conduct — see [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

---

<p align="center">Made with ❤ by <a href="https://github.com/H0NEYP0T-466">H0NEYP0T-466</a></p>
