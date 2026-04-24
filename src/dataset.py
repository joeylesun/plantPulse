"""
Dataset loading, augmentation, and splitting for PlantVillage.

Rubric items addressed here:
  - Image augmentation (>=4 techniques)
  - Train / validation / test split
  - PyTorch DataLoader with batching + shuffling

AI attribution: Scaffolding of this module was drafted with Claude (Anthropic)
and then reviewed/edited by the author. See ATTRIBUTION.md.
"""

from pathlib import Path
from typing import Tuple, List

import torch
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms


# Standard ImageNet normalization stats (ResNet-50 was pretrained with these)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224  # ResNet-50 expects 224x224


def build_train_transform(use_augmentation: bool = True) -> transforms.Compose:
    """Build the training-time transform pipeline.

    If use_augmentation is False, we only do resize + normalize. This toggle is
    used for the ablation study (with vs. without augmentation).

    Augmentation techniques (5 total, satisfies the ">=4 techniques" rubric item):
        1. RandomResizedCrop  - random zoom + crop
        2. RandomRotation - rotate up to +-15 degrees
        3. RandomHorizontalFlip - mirror the leaf
        4. ColorJitter - vary brightness/contrast/saturation (lighting robustness)
        5. RandomErasing - randomly black out patches (occlusion robustness)
    """
    if not use_augmentation:
        return transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    return transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.1)),
    ])


def build_eval_transform() -> transforms.Compose:
    """Deterministic transform for validation and test sets.

    No augmentation here - we want reproducible metrics.
    """
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class _TransformSubset(torch.utils.data.Dataset):
    """Wraps a Subset and applies a different transform to it.

    We need this because random_split gives us Subsets that share the parent
    dataset's transform. For val/test we want no augmentation, so we override.
    """

    def __init__(self, subset: Subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # Access the underlying dataset via the subset's indices, but skip the
        # parent transform by reading the raw image directly.
        parent = self.subset.dataset  # ImageFolder
        real_idx = self.subset.indices[idx]
        path, label = parent.samples[real_idx]
        img = parent.loader(path)
        if self.transform:
            img = self.transform(img)
        return img, label


def build_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 2,
    train_split: float = 0.70,
    val_split: float = 0.15,
    use_augmentation: bool = True,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Build train/val/test DataLoaders from a PlantVillage-style image folder.

    Expected directory layout:
        data_dir/
            Apple___Apple_scab/
                img1.jpg
                img2.jpg
                ...
            Apple___healthy/
                ...

    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Data directory not found: {data_dir}. "
            f"See data/README.md for download instructions."
        )

    # Base dataset with no transform - we'll wrap subsets individually
    base_dataset = datasets.ImageFolder(root=str(data_dir), transform=None)
    class_names = base_dataset.classes
    n_total = len(base_dataset)

    # Compute split sizes. Test set gets whatever's left over.
    n_train = int(train_split * n_total)
    n_val = int(val_split * n_total)
    n_test = n_total - n_train - n_val

    # Deterministic split for reproducibility across ablation runs
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(
        base_dataset, [n_train, n_val, n_test], generator=generator
    )

    train_ds = _TransformSubset(train_subset, build_train_transform(use_augmentation))
    val_ds = _TransformSubset(val_subset, build_eval_transform())
    test_ds = _TransformSubset(test_subset, build_eval_transform())

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Dataset loaded: {n_total} images across {len(class_names)} classes")
    print(f"  Train: {n_train} | Val: {n_val} | Test: {n_test}")
    print(f"  Augmentation: {'ON' if use_augmentation else 'OFF'}")

    return train_loader, val_loader, test_loader, class_names
