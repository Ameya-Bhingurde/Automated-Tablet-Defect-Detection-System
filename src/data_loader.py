"""
Data loading and preprocessing utilities
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
import config


class TabletDataset(Dataset):
    """Dataset for loading tablet images"""
    
    def __init__(self, root_dir: Path, transform=None, mask_dir: Optional[Path] = None):
        """
        Args:
            root_dir: Directory containing images
            transform: Optional transform to apply to images
            mask_dir: Optional directory containing ground truth masks
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mask_dir = mask_dir
        
        # Get all PNG images
        self.image_paths = sorted(list(root_dir.glob("*.png")))
        
        if not self.image_paths:
            raise ValueError(f"No images found in {root_dir}")
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str, Optional[torch.Tensor]]:
        """
        Returns:
            image: Preprocessed image tensor
            image_path: Path to the image
            mask: Ground truth mask (if available)
        """
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Load mask if available
        mask = None
        if self.mask_dir is not None:
            mask_path = self.mask_dir / img_path.name
            if mask_path.exists():
                mask = Image.open(mask_path).convert("L")
                mask = transforms.Resize(config.IMAGE_SIZE)(mask)
                mask = torch.tensor(np.array(mask), dtype=torch.float32)
                mask = (mask > 0).float()  # Binarize
        
        if self.transform:
            image = self.transform(image)
        
        return image, str(img_path), mask


def get_transforms(is_train: bool = False):
    """Get image preprocessing transforms"""
    transform_list = [
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ]
    
    # No augmentation needed for unsupervised anomaly detection
    return transforms.Compose(transform_list)


def custom_collate(batch):
    """Custom collate function to handle None masks"""
    images = torch.stack([item[0] for item in batch])
    paths = [item[1] for item in batch]
    masks = [item[2] for item in batch]
    
    # Convert None masks to empty list if all are None
    if all(m is None for m in masks):
        masks = None
    else:
        # Stack non-None masks, pad None with zeros
        masks = torch.stack([m if m is not None else torch.zeros_like(masks[0]) for m in masks])
    
    return images, paths, masks


def get_dataloader(data_dir: Path, batch_size: int = 32, 
                   shuffle: bool = False, mask_dir: Optional[Path] = None) -> DataLoader:
    """Create DataLoader for tablet images"""
    
    transform = get_transforms()
    dataset = TabletDataset(data_dir, transform=transform, mask_dir=mask_dir)
    
    # Set num_workers to 0 for Windows compatibility
    num_workers = 0
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,  # Disable for CPU
        collate_fn=custom_collate
    )
    
    return dataloader



def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize image tensor for visualization"""
    mean = torch.tensor(config.MEAN).view(3, 1, 1)
    std = torch.tensor(config.STD).view(3, 1, 1)
    return tensor * std + mean


import numpy as np  # Need this import

def load_single_image(image_path: str) -> Tuple[torch.Tensor, Image.Image]:
    """
    Load and preprocess a single image for inference
    
    Args:
        image_path: Path to the image
        
    Returns:
        preprocessed: Preprocessed tensor [1, 3, H, W]
        original: Original PIL image
    """
    original = Image.open(image_path).convert("RGB")
    transform = get_transforms()
    preprocessed = transform(original).unsqueeze(0)  # Add batch dimension
    
    return preprocessed, original
