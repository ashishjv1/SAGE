"""
Data Utilities for SAGE
========================

Handles dataset loading, transformations, and data corruption experiments.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.datasets import CIFAR10, CIFAR100
import PIL.Image as Image
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TinyImageNet(Dataset):
    """TinyImageNet dataset implementation"""
    
    def __init__(self, root: str, split: str = 'train', transform=None):
        self.root_dir = root
        self.split = split
        self.transform = transform
        
        # Load class names
        with open(os.path.join(self.root_dir, 'wnids.txt'), 'r') as f:
            self.classes = f.read().strip().split()
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # Load images
        self.images = self._load_images()

    def _load_images(self):
        images = []
        if self.split == 'train':
            for cls in self.classes:
                cls_dir = os.path.join(self.root_dir, self.split, cls, 'images')
                if os.path.exists(cls_dir):
                    for image_file in os.listdir(cls_dir):
                        image_path = os.path.join(cls_dir, image_file)
                        images.append((image_path, self.class_to_idx[cls]))
        elif self.split == 'val':
            val_dir = os.path.join(self.root_dir, self.split, 'images')
            image_to_cls = {}
            val_annotations = os.path.join(self.root_dir, self.split, 'val_annotations.txt')
            if os.path.exists(val_annotations):
                with open(val_annotations, 'r') as f:
                    for line in f.read().strip().split('\n'):
                        parts = line.split()
                        if len(parts) >= 2:
                            image_to_cls[parts[0].strip()] = parts[1].strip()
                
                if os.path.exists(val_dir):
                    for image_file in os.listdir(val_dir):
                        if image_file in image_to_cls:
                            image_path = os.path.join(val_dir, image_file)
                            images.append((image_path, self.class_to_idx[image_to_cls[image_file]]))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, label = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


class ImageNetDataset(Dataset):
    """ImageNet dataset wrapper"""
    
    def __init__(self, root: str, split: str = 'train', transform=None):
        from torchvision import datasets
        
        split_dir = 'train' if split == 'train' else 'val'
        self.dataset = datasets.ImageFolder(
            os.path.join(root, split_dir),
            transform=transform
        )
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class CorruptedLabelDataset(Dataset):
    """Dataset wrapper that applies label corruption"""
    
    def __init__(self, dataset, corruption_rate: float, num_classes: int, seed: int = 42):
        self.dataset = dataset
        self.corruption_rate = corruption_rate
        self.num_classes = num_classes
        
        # Generate corrupted labels
        np.random.seed(seed)
        self.corrupted_labels = self._generate_corrupted_labels()
    
    def _generate_corrupted_labels(self):
        """Generate corrupted labels for a fraction of the dataset"""
        n_samples = len(self.dataset)
        n_corrupt = int(n_samples * self.corruption_rate)
        
        # Select indices to corrupt
        corrupt_indices = set(np.random.choice(n_samples, n_corrupt, replace=False))
        
        corrupted_labels = {}
        for idx in corrupt_indices:
            _, original_label = self.dataset[idx]
            # Generate random wrong label
            wrong_labels = list(range(self.num_classes))
            wrong_labels.remove(original_label)
            corrupted_labels[idx] = np.random.choice(wrong_labels)
        
        return corrupted_labels
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if idx in self.corrupted_labels:
            label = self.corrupted_labels[idx]
        return image, label


class MinorityDownsampledDataset(Dataset):
    """Dataset that downsamples minority classes"""
    
    def __init__(self, dataset, downsample_rate: float, num_classes: int, seed: int = 42):
        self.dataset = dataset
        self.downsample_rate = downsample_rate
        self.num_classes = num_classes
        
        # Compute class distribution
        self.class_counts = self._compute_class_distribution()
        
        # Determine minority classes (bottom 50%)
        sorted_classes = sorted(self.class_counts.items(), key=lambda x: x[1])
        self.minority_classes = set([cls for cls, _ in sorted_classes[:num_classes//2]])
        
        # Generate valid indices after downsampling
        np.random.seed(seed)
        self.valid_indices = self._generate_valid_indices()
    
    def _compute_class_distribution(self):
        """Compute number of samples per class"""
        class_counts = {i: 0 for i in range(self.num_classes)}
        for _, label in self.dataset:
            class_counts[label] += 1
        return class_counts
    
    def _generate_valid_indices(self):
        """Generate indices after downsampling minority classes"""
        valid_indices = []
        
        for idx in range(len(self.dataset)):
            _, label = self.dataset[idx]
            
            if label in self.minority_classes:
                # Keep only (1 - downsample_rate) of minority samples
                if np.random.random() > self.downsample_rate:
                    valid_indices.append(idx)
            else:
                # Keep all majority samples
                valid_indices.append(idx)
        
        return valid_indices
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        return self.dataset[actual_idx]


def get_cifar_transforms(dataset_name: str) -> Tuple[T.Compose, T.Compose]:
    """Get standard CIFAR transforms with strong augmentation"""
    
    if dataset_name == 'cifar10':
        # CIFAR-10 normalization
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    else:  # CIFAR-100
        # CIFAR-100 normalization 
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    
    # Strong augmentation pipeline
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4, padding_mode='reflect'),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        T.Normalize(mean, std, inplace=True),
        T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=mean),
    ])
    
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean, std)
    ])
    
    return train_transform, test_transform


def get_tiny_imagenet_transforms() -> Tuple[T.Compose, T.Compose]:
    """Get TinyImageNet transforms"""
    mean = [0.480, 0.448, 0.397]
    std = [0.276, 0.269, 0.282]
    
    train_transform = T.Compose([
        T.RandomResizedCrop(64, scale=(0.8, 1.0), 
                          interpolation=T.InterpolationMode.BICUBIC),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=mean),
        T.Normalize(mean, std, inplace=True),
    ])
    
    test_transform = T.Compose([
        T.Resize(64, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(64),
        T.ToTensor(),
        T.Normalize(mean, std, inplace=True),
    ])
    
    return train_transform, test_transform


def get_imagenet_transforms() -> Tuple[T.Compose, T.Compose]:
    """Get ImageNet transforms"""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_transform = T.Compose([
        T.RandomResizedCrop(224),
        T.RandomHorizontalFlip(),
        T.RandAugment(num_ops=2, magnitude=9),
        T.ToTensor(),
        T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3), value=mean),
        T.Normalize(mean, std, inplace=True),
    ])
    
    test_transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean, std, inplace=True),
    ])
    
    return train_transform, test_transform


def get_dataset(dataset_name: str, data_path: str) -> Tuple[Dataset, Dataset]:
    """Load dataset with appropriate transforms"""
    
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'cifar10':
        train_transform, test_transform = get_cifar_transforms('cifar10')
        
        train_dataset = CIFAR10(
            root=data_path, 
            train=True, 
            transform=train_transform,
            download=True
        )
        
        test_dataset = CIFAR10(
            root=data_path,
            train=False,
            transform=test_transform,
            download=True
        )
        
    elif dataset_name == 'cifar100':
        train_transform, test_transform = get_cifar_transforms('cifar100')
        
        train_dataset = CIFAR100(
            root=data_path,
            train=True,
            transform=train_transform,
            download=True
        )
        
        test_dataset = CIFAR100(
            root=data_path,
            train=False,
            transform=test_transform,
            download=True
        )
        
    elif dataset_name == 'tiny_imagenet':
        train_transform, test_transform = get_tiny_imagenet_transforms()
        
        train_dataset = TinyImageNet(
            root=os.path.join(data_path, 'tiny-imagenet-200'),
            split='train',
            transform=train_transform
        )
        
        test_dataset = TinyImageNet(
            root=os.path.join(data_path, 'tiny-imagenet-200'),
            split='val',
            transform=test_transform
        )
        
    elif dataset_name == 'imagenet':
        train_transform, test_transform = get_imagenet_transforms()
        
        train_dataset = ImageNetDataset(
            root=os.path.join(data_path, 'imagenet'),
            split='train',
            transform=train_transform
        )
        
        test_dataset = ImageNetDataset(
            root=os.path.join(data_path, 'imagenet'),
            split='val',
            transform=test_transform
        )
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    logger.info(f"Loaded {dataset_name}: {len(train_dataset)} train, {len(test_dataset)} test samples")
    
    return train_dataset, test_dataset


def apply_label_corruption(dataset: Dataset, corruption_rate: float) -> Dataset:
    """Apply label corruption to dataset"""
    if corruption_rate <= 0:
        return dataset
    
    # Determine number of classes
    if hasattr(dataset, 'classes'):
        num_classes = len(dataset.classes)
    elif hasattr(dataset, 'class_to_idx'):
        num_classes = len(dataset.class_to_idx)
    else:
        # Infer from dataset
        labels = [dataset[i][1] for i in range(min(1000, len(dataset)))]
        num_classes = len(set(labels))
    
    return CorruptedLabelDataset(dataset, corruption_rate, num_classes)


def apply_minority_downsampling(dataset: Dataset, downsample_rate: float) -> Dataset:
    """Apply minority class downsampling to dataset"""
    if downsample_rate <= 0:
        return dataset
    
    # Determine number of classes
    if hasattr(dataset, 'classes'):
        num_classes = len(dataset.classes)
    elif hasattr(dataset, 'class_to_idx'):
        num_classes = len(dataset.class_to_idx)
    else:
        # Infer from dataset
        labels = [dataset[i][1] for i in range(min(1000, len(dataset)))]
        num_classes = len(set(labels))
    
    return MinorityDownsampledDataset(dataset, downsample_rate, num_classes)


def compute_dataset_statistics(dataset: Dataset, num_classes: int) -> dict:
    """Compute dataset statistics for analysis"""
    
    # Count samples per class
    class_counts = {i: 0 for i in range(num_classes)}
    
    for _, label in dataset:
        class_counts[label] += 1
    
    total_samples = sum(class_counts.values())
    
    statistics = {
        'total_samples': total_samples,
        'num_classes': num_classes,
        'class_counts': class_counts,
        'class_frequencies': {k: v/total_samples for k, v in class_counts.items()},
        'min_class_size': min(class_counts.values()),
        'max_class_size': max(class_counts.values()),
        'mean_class_size': total_samples / num_classes,
        'class_imbalance_ratio': max(class_counts.values()) / min(class_counts.values())
    }
    
    return statistics


# Example usage and testing functions
def test_data_loading():
    """Test data loading functionality"""
    print("Testing data loading...")
    
    # Test CIFAR-100
    train_dataset, test_dataset = get_dataset('cifar100', './data')
    print(f"CIFAR-100: {len(train_dataset)} train, {len(test_dataset)} test")
    
    # Test corruption
    corrupted_dataset = apply_label_corruption(train_dataset, 0.2)
    print(f"Corrupted dataset: {len(corrupted_dataset)} samples")
    
    # Test downsampling
    downsampled_dataset = apply_minority_downsampling(train_dataset, 0.3)
    print(f"Downsampled dataset: {len(downsampled_dataset)} samples")
    
    # Test statistics
    stats = compute_dataset_statistics(train_dataset, 100)
    print(f"Dataset statistics: {stats['total_samples']} samples, "
          f"imbalance ratio: {stats['class_imbalance_ratio']:.2f}")


if __name__ == '__main__':
    test_data_loading()
