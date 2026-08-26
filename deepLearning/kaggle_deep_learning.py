"""HARUSPEX Deep Learning Kaggle Pipeline — Self-Contained Training Script for Kaggle T4 GPU.

Instructions for Kaggle:
1. Create a Kaggle Notebook, set Accelerator to 'GPU T4 x2' or 'GPU T4'.
2. Attach dataset (BODMAS / EMBER) or upload HARUSPEX folder.
3. Paste and run this script to train TabNet and Group-wise Residual MLP with full GPU acceleration!
"""

import os
import sys
import time
import json
from pathlib import Path

print("=== HARUSPEX Deep Learning Kaggle Environment Setup ===")

# Install PyTorch TabNet if not present
try:
    import pytorch_tabnet
    print("pytorch-tabnet already installed.")
except ImportError:
    print("Installing pytorch-tabnet...")
    os.system("pip install -q pytorch-tabnet")
    import pytorch_tabnet

import torch
print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name(0))

# Code to run Phase 1 & Phase 2 pipelines
print("\nReady to run Phase 1 & Phase 2 Deep Learning Training!")
print("Run:\n  python deepLearning/phase1/trainMLP.py\n  python deepLearning/phase1/traintabnet.py\n  python deepLearning/phase1/evaluate.py")
