"""
Configuration file for Automated Tablet Defect Detection System
"""

import os
from pathlib import Path

# ===================== PATH CONFIGURATION =====================
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "capsule"
TRAIN_DIR = DATA_DIR / "train" / "good"
TEST_DIR = DATA_DIR / "test"
GROUND_TRUTH_DIR = DATA_DIR / "ground_truth"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
MODEL_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# ===================== MODEL CONFIGURATION =====================
# Backbone architecture (ResNet18 for balance between speed and accuracy)
BACKBONE = "resnet18"
FEATURE_LAYERS = ["layer1", "layer2", "layer3"]  # Multi-scale features

# Image preprocessing
IMAGE_SIZE = (224, 224)  # Standard ImageNet size
MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
STD = [0.229, 0.224, 0.225]

# PaDiM parameters
REDUCE_DIM = 100  # Dimensionality reduction via random projection
EPSILON = 1e-5    # Numerical stability for covariance matrix

# ===================== INFERENCE CONFIGURATION =====================
ANOMALY_THRESHOLD = 15.0  # Decision threshold for Mahalanobis distance (tunable)
HEATMAP_COLORMAP = "jet"  # Colormap for visualization
HEATMAP_ALPHA = 0.4      # Overlay transparency

# ===================== TRAINING CONFIGURATION =====================
BATCH_SIZE = 32
NUM_WORKERS = 4  # Dataloader workers (set to 0 for Windows compatibility)

# ===================== EVALUATION CONFIGURATION =====================
DEFECT_TYPES = ["crack", "faulty_imprint", "poke", "scratch", "squeeze"]
