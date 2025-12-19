"""
Training script for PaDiM anomaly detection model
"""

import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import config
from src.data_loader import get_dataloader
from src.feature_extractor import FeatureExtractor, extract_embeddings
from src.padim import PaDiM


def train_padim():
    """Train PaDiM model on normal training data"""
    
    print("=" * 60)
    print("AUTOMATED TABLET DEFECT DETECTION - TRAINING")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize feature extractor
    print("\nInitializing feature extractor...")
    extractor = FeatureExtractor(
        backbone=config.BACKBONE,
        layers=config.FEATURE_LAYERS
    ).to(device)
    
    # Display feature dimensions
    dims = extractor.get_feature_dimensions()
    print("\nFeature dimensions:")
    for layer, dim_info in dims.items():
        print(f"  {layer}: {dim_info}")
    
    # Load training data (only good samples)
    print(f"\nLoading training data from {config.TRAIN_DIR}...")
    train_loader = get_dataloader(
        config.TRAIN_DIR,
        batch_size=config.BATCH_SIZE,
        shuffle=False
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    
    # Extract embeddings from all training samples
    print("\nExtracting features from training data...")
    all_embeddings = []
    
    with torch.no_grad():
        for batch_idx, (images, paths, _) in enumerate(tqdm(train_loader)):
            images = images.to(device)
            
            # Extract multi-scale embeddings
            embeddings = extract_embeddings(extractor, images)
            all_embeddings.append(embeddings.cpu().numpy())
    
    # Concatenate all embeddings
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Embeddings shape: {all_embeddings.shape}")
    
    # Train PaDiM model
    print("\nTraining PaDiM model...")
    padim_model = PaDiM(
        reduce_dim=config.REDUCE_DIM,
        epsilon=config.EPSILON
    )
    padim_model.fit(all_embeddings)
    
    # Save model
    model_path = config.MODEL_DIR / "padim_model.pkl"
    padim_model.save(model_path)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Model saved to: {model_path}")
    
    return padim_model, extractor


if __name__ == "__main__":
    train_padim()
