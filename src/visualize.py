"""
Visualization utilities for anomaly detection results
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image
from typing import Tuple, Optional
import config


def apply_heatmap(image: np.ndarray, 
                  anomaly_map: np.ndarray, 
                  alpha: float = 0.4,
                  colormap: str = "jet") -> np.ndarray:
    """
    Overlay anomaly heatmap on original image
    
    Args:
        image: Original image [H, W, 3] in RGB, range [0, 255]
        anomaly_map: Anomaly map [H', W'] normalized to [0, 1]
        alpha: Overlay transparency
        colormap: Matplotlib colormap name
        
    Returns:
        Overlayed image [H, W, 3] in RGB
    """
    H, W = image.shape[:2]
    
    # Resize anomaly map to match image size
    anomaly_map_resized = cv2.resize(anomaly_map, (W, H))
    
    # Normalize to [0, 1]
    anomaly_map_norm = (anomaly_map_resized - anomaly_map_resized.min()) / \
                       (anomaly_map_resized.max() - anomaly_map_resized.min() + 1e-8)
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    heatmap = cmap(anomaly_map_norm)[:, :, :3]  # Remove alpha channel
    heatmap = (heatmap * 255).astype(np.uint8)
    
    # Ensure image is uint8
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    
    # Blend image and heatmap
    overlayed = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    
    return overlayed


def create_result_visualization(original_image: Image.Image,
                                anomaly_score: float,
                                anomaly_map: np.ndarray,
                                threshold: float = 0.5,
                                ground_truth: Optional[np.ndarray] = None) -> Image.Image:
    """
    Create comprehensive result visualization
    
    Args:
        original_image: PIL Image
        anomaly_score: Image-level anomaly score
        anomaly_map: Pixel-level anomaly map
        threshold: Decision threshold
        ground_truth: Optional ground truth mask
        
    Returns:
        Combined visualization as PIL Image
    """
    # Convert to numpy array
    img_np = np.array(original_image)
    
    # Create figure with subplots
    n_cols = 4 if ground_truth is not None else 3
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 4, 4))
    
    # Original image
    axes[0].imshow(img_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    # Anomaly heatmap
    heatmap_overlay = apply_heatmap(img_np, anomaly_map, alpha=config.HEATMAP_ALPHA)
    axes[1].imshow(heatmap_overlay)
    
    prediction = "DEFECTIVE" if anomaly_score > threshold else "NORMAL"
    color = "red" if anomaly_score > threshold else "green"
    axes[1].set_title(f"Prediction: {prediction}\nScore: {anomaly_score:.3f}", 
                     color=color, fontweight='bold')
    axes[1].axis('off')
    
    # Raw anomaly map
    im = axes[2].imshow(anomaly_map, cmap=config.HEATMAP_COLORMAP)
    axes[2].set_title("Anomaly Map")
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2], fraction=0.046)
    
    # Ground truth (if available)
    if ground_truth is not None:
        axes[3].imshow(ground_truth, cmap='gray')
        axes[3].set_title("Ground Truth")
        axes[3].axis('off')
    
    plt.tight_layout()
    
    # Convert to PIL Image
    fig.canvas.draw()
    vis_np = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    vis_np = vis_np.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    vis_pil = Image.fromarray(vis_np)
    
    plt.close(fig)
    
    return vis_pil


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float, 
                   save_path: Optional[str] = None):
    """Plot ROC curve"""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Image-Level Anomaly Detection', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.close()


def save_prediction(image: Image.Image, 
                   anomaly_score: float,
                   anomaly_map: np.ndarray,
                   save_path: str,
                   threshold: float = 0.5):
    """Save prediction result with overlay"""
    img_np = np.array(image)
    overlay = apply_heatmap(img_np, anomaly_map, alpha=config.HEATMAP_ALPHA)
    
    # Add text annotation
    prediction = "DEFECTIVE" if anomaly_score > threshold else "NORMAL"
    color = (255, 0, 0) if anomaly_score > threshold else (0, 255, 0)
    
    cv2.putText(overlay, f"{prediction} ({anomaly_score:.3f})", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                1, color, 2, cv2.LINE_AA)
    
    # Save
    Image.fromarray(overlay).save(save_path)
