"""Dataset classes for loading and processing face recognition data."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class FacecapDataset(Dataset):
    """Dataset class for loading facecap face recognition data."""
    
    def __init__(
        self,
        data_root: Union[str, Path],
        split: str = "train",
        transform: Optional[transforms.Compose] = None,
        target_size: Tuple[int, int] = (112, 112),
    ):
        """
        Initialize the facecap dataset.
        
        Args:
            data_root: Root directory containing facecap data
            split: Dataset split ('train', 'val', or 'test')
            transform: Optional image transformations
            target_size: Target image size (height, width)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.target_size = target_size
        
        # Load labels
        self.labels = self._load_labels()
        
        # Load image paths and labels
        self.samples = self._load_samples()
        
        # Set up transforms
        if transform is None:
            self.transform = self._get_default_transform()
        else:
            self.transform = transform
    
    def _load_labels(self) -> Dict[int, str]:
        """Load class labels from labels.txt file."""
        labels_file = self.data_root / "labels.txt"
        labels = {}
        
        if labels_file.exists():
            with open(labels_file, 'r') as f:
                for idx, line in enumerate(f):
                    label = line.strip()
                    labels[idx] = label
        else:
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        
        return labels
    
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load image paths and corresponding labels."""
        split_file = self.data_root / f"{self.split}_list.txt"
        samples = []
        
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        
        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    # Parse line: "path/to/image.jpg label"
                    parts = line.split()
                    if len(parts) >= 2:
                        image_path = parts[0]
                        label = int(parts[1])
                        
                        # Construct full image path
                        full_path = self.data_root / image_path
                        if full_path.exists():
                            samples.append((full_path, label))
        
        if not samples:
            raise ValueError(f"No valid samples found for split: {self.split}")
        
        return samples
    
    def _get_default_transform(self) -> transforms.Compose:
        """Get default image transformations."""
        if self.split == "train":
            # Training transforms with augmentation
            return transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                ),
                transforms.RandomRotation(degrees=10),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            # Validation/test transforms without augmentation
            return transforms.Compose([
                transforms.Resize(self.target_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (image_tensor, label)
        """
        image_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_counts(self) -> Dict[int, int]:
        """Get the number of samples per class."""
        class_counts = {}
        for _, label in self.samples:
            class_counts[label] = class_counts.get(label, 0) + 1
        return class_counts
    
    def get_sample_weights(self) -> torch.Tensor:
        """Get sample weights for balanced sampling."""
        class_counts = self.get_class_counts()
        total_samples = len(self.samples)
        
        # Calculate weights inversely proportional to class frequency
        weights = []
        for _, label in self.samples:
            weight = total_samples / (len(class_counts) * class_counts[label])
            weights.append(weight)
        
        return torch.tensor(weights, dtype=torch.float32)


class FaceImageDataset(Dataset):
    """Dataset for loading individual face images with preprocessing."""
    
    def __init__(
        self,
        image_paths: List[Union[str, Path]],
        labels: Optional[List[int]] = None,
        transform: Optional[transforms.Compose] = None,
        target_size: Tuple[int, int] = (112, 112),
    ):
        """
        Initialize face image dataset.
        
        Args:
            image_paths: List of image file paths
            labels: Optional list of labels (for supervised learning)
            transform: Optional image transformations
            target_size: Target image size (height, width)
        """
        self.image_paths = [Path(p) for p in image_paths]
        self.labels = labels
        self.target_size = target_size
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
    
    def __len__(self) -> int:
        """Return the number of images."""
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
        """
        Get an image from the dataset.
        
        Args:
            idx: Image index
            
        Returns:
            Image tensor or tuple of (image_tensor, label)
        """
        image_path = self.image_paths[idx]
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            raise RuntimeError(f"Error loading image {image_path}: {e}")
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Return with or without label
        if self.labels is not None:
            return image, self.labels[idx]
        else:
            return image


def create_data_loaders(
    data_root: Union[str, Path],
    batch_size: int = 32,
    num_workers: int = 4,
    target_size: Tuple[int, int] = (112, 112),
) -> Tuple[torch.utils.data.DataLoader, ...]:
    """
    Create data loaders for training, validation, and testing.
    
    Args:
        data_root: Root directory containing facecap data
        batch_size: Batch size for data loaders
        num_workers: Number of worker processes for data loading
        target_size: Target image size (height, width)
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Create datasets
    train_dataset = FacecapDataset(
        data_root=data_root,
        split="train",
        target_size=target_size
    )
    
    val_dataset = FacecapDataset(
        data_root=data_root,
        split="val",
        target_size=target_size
    )
    
    test_dataset = FacecapDataset(
        data_root=data_root,
        split="test",
        target_size=target_size
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def analyze_dataset(data_root: Union[str, Path]) -> Dict:
    """
    Analyze the facecap dataset and return statistics.
    
    Args:
        data_root: Root directory containing facecap data
        
    Returns:
        Dictionary containing dataset statistics
    """
    data_root = Path(data_root)
    stats = {}
    
    # Load datasets
    for split in ["train", "val", "test"]:
        try:
            dataset = FacecapDataset(data_root, split=split)
            class_counts = dataset.get_class_counts()
            
            stats[split] = {
                "num_samples": len(dataset),
                "num_classes": len(class_counts),
                "class_counts": class_counts,
                "min_samples_per_class": min(class_counts.values()) if class_counts else 0,
                "max_samples_per_class": max(class_counts.values()) if class_counts else 0,
                "avg_samples_per_class": np.mean(list(class_counts.values())) if class_counts else 0,
            }
        except Exception as e:
            stats[split] = {"error": str(e)}
    
    return stats