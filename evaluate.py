"""
Evaluation script for PaDiM anomaly detection model
"""

import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import sys
import json

sys.path.append(str(Path(__file__).parent))

import config
from src.data_loader import get_dataloader
from src.feature_extractor import FeatureExtractor, extract_embeddings
from src.padim import PaDiM
from src.visualize import plot_roc_curve, save_prediction
from PIL import Image


def evaluate_padim():
    """Evaluate PaDiM model on test data"""
    
    print("=" * 60)
    print("AUTOMATED TABLET DEFECT DETECTION - EVALUATION")
    print("=" * 60)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("\nLoading trained model...")
    model_path = config.MODEL_DIR / "padim_model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run train.py first.")
    
    padim_model = PaDiM()
    padim_model.load(model_path)
    
    # Initialize feature extractor
    print("Initializing feature extractor...")
    extractor = FeatureExtractor(
        backbone=config.BACKBONE,
        layers=config.FEATURE_LAYERS
    ).to(device)
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    
    all_scores = []
    all_labels = []
    all_predictions = []
    
    defect_types = ["good"] + config.DEFECT_TYPES
    
    for defect_type in defect_types:
        test_dir = config.TEST_DIR / defect_type
        
        if not test_dir.exists():
            print(f"Skipping {defect_type} (directory not found)")
            continue
        
        print(f"\nProcessing {defect_type}...")
        
        # Ground truth: 0 for good, 1 for defect
        is_defect = 1 if defect_type != "good" else 0
        
        # Get dataloader
        test_loader = get_dataloader(test_dir, batch_size=1, shuffle=False)
        
        for images, paths, _ in tqdm(test_loader):
            images = images.to(device)
            
            # Extract embeddings
            with torch.no_grad():
                embeddings = extract_embeddings(extractor, images)
            
            # Predict anomaly
            embeddings_np = embeddings.cpu().numpy()
            anomaly_score, anomaly_map = padim_model.predict(embeddings_np)
            
            all_scores.append(anomaly_score)
            all_labels.append(is_defect)
            
            # Save some example predictions
            if len(all_predictions) < 20:  # Save first 20 examples
                img_path = paths[0]
                img = Image.open(img_path)
                
                save_path = config.RESULTS_DIR / f"{defect_type}_{Path(img_path).name}"
                save_prediction(img, anomaly_score, anomaly_map, str(save_path))
                all_predictions.append({
                    'image': img_path,
                    'score': float(anomaly_score),
                    'label': is_defect
                })
    
    # Compute metrics
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    # ROC-AUC
    roc_auc = roc_auc_score(all_labels, all_scores)
    print(f"\n{'=' * 60}")
    print(f"IMAGE-LEVEL ROC-AUC: {roc_auc:.4f}")
    print(f"{'=' * 60}")
    
    # Find optimal threshold using Youden's J statistic
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    print(f"\nOptimal threshold: {optimal_threshold:.4f}")
    
    # Compute precision and recall at optimal threshold
    predictions = (all_scores >= optimal_threshold).astype(int)
    
    tp = np.sum((predictions == 1) & (all_labels == 1))
    fp = np.sum((predictions == 1) & (all_labels == 0))
    fn = np.sum((predictions == 0) & (all_labels == 1))
    tn = np.sum((predictions == 0) & (all_labels == 0))
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / len(all_labels)
    
    print(f"\nMetrics at optimal threshold:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(f"  TP: {tp}, FP: {fp}")
    print(f"  FN: {fn}, TN: {tn}")
    
    # Plot ROC curve
    roc_path = config.RESULTS_DIR / "roc_curve.png"
    plot_roc_curve(fpr, tpr, roc_auc, str(roc_path))
    
    # Save results
    results = {
        'roc_auc': float(roc_auc),
        'optimal_threshold': float(optimal_threshold),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'accuracy': float(accuracy),
        'confusion_matrix': {
            'tp': int(tp), 'fp': int(fp),
            'fn': int(fn), 'tn': int(tn)
        }
    }
    
    results_path = config.RESULTS_DIR / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print(f"Example predictions saved to {config.RESULTS_DIR}")
    
    return results


if __name__ == "__main__":
    evaluate_padim()
