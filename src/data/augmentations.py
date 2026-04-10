"""
augmentations.py — Albumentations training transforms for chest X-ray segmentation.
"""

import albumentations as A


def get_train_transforms() -> A.Compose:
    """Return augmentation pipeline for training.

    Designed for chest X-rays with small pneumothorax masks:
    - Geometric: conservative rotations, horizontal flip only (no vertical — anatomy matters)
    - Intensity: brightness/contrast + CLAHE for X-ray contrast enhancement
    - Elastic: reduced alpha (20) to avoid destroying thin mask regions
    - Noise: mild Gaussian noise
    """
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.ElasticTransform(alpha=20, sigma=5, p=0.15),
        A.GaussNoise(std_range=(0.02, 0.05), p=0.2),
    ])
