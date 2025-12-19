"""
Feature extraction using pretrained ResNet backbone
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from typing import List, Dict
import config


class FeatureExtractor(nn.Module):
    """Extract multi-scale features from ResNet backbone"""
    
    def __init__(self, backbone: str = "resnet18", layers: List[str] = None):
        """
        Args:
            backbone: Backbone architecture name
            layers: List of layer names to extract features from
        """
        super().__init__()
        
        if layers is None:
            layers = config.FEATURE_LAYERS
        
        self.layers = layers
        
        # Load pretrained ResNet
        if backbone == "resnet18":
            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Register hooks to extract intermediate features
        self.feature_maps = {}
        self.hooks = []
        
        for name, module in model.named_children():
            if name in self.layers:
                hook = module.register_forward_hook(self._get_hook(name))
                self.hooks.append(hook)
        
        # Keep only the feature extraction part (remove classifier)
        self.model = model
        self.model.eval()
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
    
    def _get_hook(self, name: str):
        """Create forward hook to capture intermediate features"""
        def hook(module, input, output):
            self.feature_maps[name] = output
        return hook
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract features from multiple layers
        
        Args:
            x: Input tensor [B, 3, H, W]
            
        Returns:
            Dictionary mapping layer names to feature tensors
        """
        self.feature_maps.clear()
        
        with torch.no_grad():
            _ = self.model(x)
        
        return self.feature_maps
    
    def get_feature_dimensions(self) -> Dict[str, int]:
        """Get output dimensions for each layer"""
        dummy_input = torch.randn(1, 3, *config.IMAGE_SIZE)
        features = self.forward(dummy_input)
        
        dimensions = {}
        for name, feat in features.items():
            dimensions[name] = {
                'channels': feat.shape[1],
                'height': feat.shape[2],
                'width': feat.shape[3]
            }
        
        return dimensions
    
    def __del__(self):
        """Remove hooks when object is destroyed"""
        for hook in self.hooks:
            hook.remove()


def extract_embeddings(extractor: FeatureExtractor, 
                       x: torch.Tensor) -> torch.Tensor:
    """
    Extract and concatenate multi-scale embeddings
    
    Args:
        extractor: Feature extractor model
        x: Input tensor [B, 3, H, W]
        
    Returns:
        Concatenated embeddings [B, D, H', W'] where D is total channels
    """
    feature_maps = extractor(x)
    
    # Get target spatial dimensions from the largest feature map
    target_size = None
    for name in extractor.layers:
        feat = feature_maps[name]
        if target_size is None or (feat.shape[2] > target_size[0]):
            target_size = (feat.shape[2], feat.shape[3])
    
    # Resize all features to same spatial dimensions and concatenate
    aligned_features = []
    for name in extractor.layers:
        feat = feature_maps[name]
        if feat.shape[2:] != target_size:
            feat = torch.nn.functional.interpolate(
                feat, size=target_size, mode='bilinear', align_corners=False
            )
        aligned_features.append(feat)
    
    # Concatenate along channel dimension
    embeddings = torch.cat(aligned_features, dim=1)
    
    return embeddings
