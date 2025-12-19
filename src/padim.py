"""
PaDiM: Patch Distribution Modeling for Anomaly Detection
Implementation based on: https://arxiv.org/abs/2011.08785
"""

import torch
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.random_projection import SparseRandomProjection
from typing import Tuple, Optional
import pickle
from pathlib import Path
import config


class PaDiM:
    """
    PaDiM anomaly detection model
    
    Models the distribution of patch-level features using multivariate Gaussian
    and detects anomalies via Mahalanobis distance.
    """
    
    def __init__(self, reduce_dim: int = 100, epsilon: float = 1e-5):
        """
        Args:
            reduce_dim: Target dimensionality after random projection
            epsilon: Small constant for numerical stability
        """
        self.reduce_dim = reduce_dim
        self.epsilon = epsilon
        
        # Model parameters (learned from training data)
        self.mean = None           # [H, W, D]
        self.cov_inv = None        # [H, W, D, D]
        self.projection = None     # Random projection matrix
        
        self.feature_shape = None  # Spatial dimensions of features
        
    def fit(self, embeddings: np.ndarray):
        """
        Fit PaDiM model on normal training embeddings
        
        Args:
            embeddings: Training embeddings [N, D, H, W]
        """
        N, D, H, W = embeddings.shape
        self.feature_shape = (H, W)
        
        print(f"Fitting PaDiM on {N} training samples...")
        print(f"Original embedding dimension: {D}")
        
        # Apply random dimensionality reduction
        if D > self.reduce_dim:
            embeddings_reshaped = embeddings.transpose(0, 2, 3, 1).reshape(-1, D)
            
            self.projection = SparseRandomProjection(
                n_components=self.reduce_dim,
                random_state=42
            )
            embeddings_reduced = self.projection.fit_transform(embeddings_reshaped)
            embeddings = embeddings_reduced.reshape(N, H, W, self.reduce_dim)
            D = self.reduce_dim
            print(f"Reduced embedding dimension: {D}")
        else:
            embeddings = embeddings.transpose(0, 2, 3, 1)  # [N, H, W, D]
        
        # Compute mean and covariance for each spatial location
        self.mean = np.mean(embeddings, axis=0)  # [H, W, D]
        
        # Compute covariance inverse for each pixel
        self.cov_inv = np.zeros((H, W, D, D))
        
        for h in range(H):
            for w in range(W):
                # Get features at this spatial location across all samples
                features = embeddings[:, h, w, :]  # [N, D]
                
                # Compute covariance matrix
                cov = np.cov(features.T) + self.epsilon * np.eye(D)
                
                # Compute precision matrix (inverse covariance)
                try:
                    self.cov_inv[h, w] = np.linalg.inv(cov)
                except np.linalg.LinAlgError:
                    # Fallback to pseudo-inverse if singular
                    self.cov_inv[h, w] = np.linalg.pinv(cov)
        
        print("PaDiM model fitted successfully!")
    
    def predict(self, embeddings: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute anomaly score and heatmap for test embeddings
        
        Args:
            embeddings: Test embeddings [1, D, H, W] (single image)
            
        Returns:
            anomaly_score: Image-level anomaly score (scalar)
            anomaly_map: Pixel-level anomaly heatmap [H, W]
        """
        if self.mean is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Apply same dimensionality reduction as training
        _, D, H, W = embeddings.shape
        
        if self.projection is not None:
            embeddings_reshaped = embeddings.transpose(0, 2, 3, 1).reshape(-1, D)
            embeddings_reduced = self.projection.transform(embeddings_reshaped)
            embeddings = embeddings_reduced.reshape(1, H, W, self.reduce_dim)
            D = self.reduce_dim
        else:
            embeddings = embeddings.transpose(0, 2, 3, 1)  # [1, H, W, D]
        
        embeddings = embeddings[0]  # Remove batch dimension [H, W, D]
        
        # Compute Mahalanobis distance for each pixel
        anomaly_map = np.zeros((H, W))
        
        for h in range(H):
            for w in range(W):
                delta = embeddings[h, w] - self.mean[h, w]  # [D]
                
                # Mahalanobis distance: sqrt(delta^T * Sigma^-1 * delta)
                distance = np.sqrt(
                    delta @ self.cov_inv[h, w] @ delta.T
                )
                anomaly_map[h, w] = distance
        
        # Apply Gaussian smoothing to reduce noise
        anomaly_map = gaussian_filter(anomaly_map, sigma=4)
        
        # Image-level score is max of anomaly map
        anomaly_score = np.max(anomaly_map)
        
        return anomaly_score, anomaly_map
    
    def save(self, path: Path):
        """Save model parameters to disk"""
        model_dict = {
            'mean': self.mean,
            'cov_inv': self.cov_inv,
            'projection': self.projection,
            'feature_shape': self.feature_shape,
            'reduce_dim': self.reduce_dim,
            'epsilon': self.epsilon
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_dict, f)
        
        print(f"Model saved to {path}")
    
    def load(self, path: Path):
        """Load model parameters from disk"""
        with open(path, 'rb') as f:
            model_dict = pickle.load(f)
        
        self.mean = model_dict['mean']
        self.cov_inv = model_dict['cov_inv']
        self.projection = model_dict['projection']
        self.feature_shape = model_dict['feature_shape']
        self.reduce_dim = model_dict['reduce_dim']
        self.epsilon = model_dict['epsilon']
        
        print(f"Model loaded from {path}")
