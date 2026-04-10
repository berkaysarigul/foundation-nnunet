"""
dataset.py — PyTorch Dataset for SIIM-ACR pneumothorax segmentation.

Loads preprocessed PNG images and masks from data/processed/pneumothorax/.
Applies optional albumentations transforms (train only).
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.mask_variants import resolve_mask_dir, resolve_mask_variant


class PneumothoraxDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        img_size: int = 256,
        transform=None,
        mask_variant: str | None = None,
    ):
        """
        Args:
            data_dir:      Path to processed data directory (data/processed/pneumothorax).
            split:         "train", "val", or "test".
            img_size:      Runtime resize dimension (256 or 512).
            transform:     albumentations Compose pipeline, or None.
            mask_variant:  Processed mask directory to read from. If omitted,
                           the dataset uses dilated masks for train and
                           original masks for validation/test.
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.transform = transform

        splits_path = self.data_dir / "splits.json"
        with open(splits_path) as f:
            splits = json.load(f)

        if split not in splits:
            raise ValueError(f"Unknown split '{split}'. Expected one of {list(splits.keys())}.")

        self.split = split
        purpose = "train" if split == "train" else "eval"
        self.mask_variant = resolve_mask_variant(mask_variant, purpose=purpose)
        self.mask_dir = resolve_mask_dir(self.data_dir, self.mask_variant)
        if not self.mask_dir.exists():
            raise FileNotFoundError(
                f"Mask variant directory not found: {self.mask_dir}. "
                f"Expected processed dataset to contain separate mask variants."
            )

        self.image_ids: list[str] = splits[split]

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_id = self.image_ids[idx]

        # 1. Read grayscale image and mask
        image = cv2.imread(
            str(self.data_dir / "images" / f"{image_id}.png"),
            cv2.IMREAD_GRAYSCALE,
        )
        mask = cv2.imread(
            str(self.mask_dir / f"{image_id}.png"),
            cv2.IMREAD_GRAYSCALE,
        )

        if image is None:
            raise FileNotFoundError(f"Image not found: {image_id}.png")
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {image_id}.png")

        # 2. Resize — INTER_LINEAR for image, INTER_NEAREST for mask (critical)
        if image.shape[0] != self.img_size or image.shape[1] != self.img_size:
            image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        # 3. Augmentation (albumentations expects HWC for image, HW for mask)
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # 4. Normalise image to 0-1 float
        image = image.astype(np.float32) / 255.0

        # 5. Binarise mask: threshold at 127 to handle any residual edge values
        mask = (mask > 127).astype(np.float32)

        # 6. Add channel dim and convert to tensors → (1, H, W)
        image = torch.from_numpy(image).unsqueeze(0)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask
