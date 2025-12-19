"""
Standalone inference script for single image prediction
"""

import torch
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

import config
from src.feature_extractor import FeatureExtractor, extract_embeddings
from src.padim import PaDiM
from src.visualize import save_prediction


def predict_single_image(image_path: str, 
                         model_path: str = None,
                         threshold: float = 0.5,
                         save_result: bool = True) -> dict:
    """
    Run inference on a single image
    
    Args:
        image_path: Path to input image
        model_path: Path to trained PaDiM model (default: models/padim_model.pkl)
        threshold: Anomaly threshold
        save_result: Whether to save visualization
        
    Returns:
        Dictionary with prediction results
    """
    if model_path is None:
        model_path = config.MODEL_DIR / "padim_model.pkl"
    
    # Check files exist
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}. Run train.py first.")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    padim_model = PaDiM()
    padim_model.load(model_path)
    
    # Load feature extractor
    print("Loading feature extractor...")
    extractor = FeatureExtractor(
        backbone=config.BACKBONE,
        layers=config.FEATURE_LAYERS
    ).to(device)
    
    # Load and preprocess image
    print(f"Processing image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    
    from src.data_loader import load_single_image
    img_tensor, original = load_single_image(image_path)
    img_tensor = img_tensor.to(device)
    
    # Extract features
    print("Extracting features...")
    with torch.no_grad():
        embeddings = extract_embeddings(extractor, img_tensor)
    
    # Predict
    print("Computing anomaly score...")
    embeddings_np = embeddings.cpu().numpy()
    anomaly_score, anomaly_map = padim_model.predict(embeddings_np)
    
    # Make decision
    is_defective = anomaly_score > threshold
    prediction = "DEFECTIVE" if is_defective else "NORMAL"
    
    # Print results
    print("\n" + "=" * 60)
    print(f"PREDICTION: {prediction}")
    print(f"Anomaly Score: {anomaly_score:.4f}")
    print(f"Threshold: {threshold:.4f}")
    print("=" * 60)
    
    # Save visualization
    if save_result:
        output_path = config.RESULTS_DIR / f"prediction_{Path(image_path).stem}.png"
        save_prediction(image, anomaly_score, anomaly_map, str(output_path), threshold)
        print(f"\nResult saved to: {output_path}")
    
    return {
        'image_path': str(image_path),
        'prediction': prediction,
        'anomaly_score': float(anomaly_score),
        'threshold': threshold,
        'is_defective': is_defective
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on a single tablet image"
    )
    parser.add_argument(
        'image_path',
        type=str,
        help='Path to input image'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Path to trained model (default: models/padim_model.pkl)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Anomaly threshold (default: 0.5)'
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save result visualization'
    )
    
    args = parser.parse_args()
    
    predict_single_image(
        image_path=args.image_path,
        model_path=args.model,
        threshold=args.threshold,
        save_result=not args.no_save
    )


if __name__ == "__main__":
    main()
