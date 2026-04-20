"""
PyTorch dataset for SIIM-ACR pneumothorax segmentation.

Loads preprocessed PNG images and masks from a processed dataset root and
optionally applies the fixed D-031 train-only ROI crop policy before the
existing augmentation stack.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.mask_variants import resolve_mask_dir, resolve_mask_variant


TRAIN_CROP_MODE_ALIASES = {
    "none": "none",
    "off": "none",
    "disabled": "none",
    "roi_train_only": "roi_train_only",
    "roitrainonly": "roi_train_only",
}


def _normalize_component_name(value: str) -> str:
    return "".join(ch for ch in str(value).lower() if ch.isalnum() or ch == "_")


def resolve_train_crop_config(
    train_crop: dict[str, Any] | None,
    *,
    split: str,
    img_size: int,
) -> dict[str, int | str] | None:
    if not train_crop:
        return None

    normalized_mode = _normalize_component_name(train_crop.get("mode", "none"))
    mode = TRAIN_CROP_MODE_ALIASES.get(normalized_mode)
    if mode is None:
        raise ValueError(
            "Unsupported data.train_crop.mode. Supported values: 'none', 'roi_train_only'."
        )
    if mode == "none":
        return None
    if split != "train":
        raise ValueError(
            "data.train_crop is only supported for split='train'; "
            f"received split={split!r}."
        )
    if img_size != 512:
        raise ValueError(
            "data.train_crop requires data.input_size=512 so crops can be "
            "resized back to the fixed authoritative tensor size."
        )

    crop_size = int(train_crop.get("crop_size", 384))
    if crop_size != 384:
        raise ValueError(
            "data.train_crop.crop_size must stay fixed at 384 for the immediate "
            "D-031 crop comparison."
        )

    return {"mode": mode, "crop_size": crop_size}


def _resolve_positive_crop_start(
    bbox_min: int,
    bbox_max: int,
    *,
    image_size: int,
    crop_size: int,
) -> int:
    max_start = image_size - crop_size
    bbox_size = bbox_max - bbox_min + 1
    if bbox_size >= crop_size:
        center = (bbox_min + bbox_max) // 2
        return max(0, min(center - crop_size // 2, max_start))

    start_min = max(0, bbox_max - crop_size + 1)
    start_max = min(bbox_min, max_start)
    if start_min > start_max:
        center = (bbox_min + bbox_max) // 2
        return max(0, min(center - crop_size // 2, max_start))
    return random.randint(start_min, start_max)


def apply_train_roi_crop(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    crop_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if image.shape != mask.shape:
        raise ValueError(
            "ROI crop requires image and mask to share the same spatial shape; "
            f"got image={image.shape}, mask={mask.shape}."
        )
    if image.shape != (512, 512):
        raise ValueError(
            "ROI crop requires 512 x 512 processed inputs for the immediate D-031 policy; "
            f"got {image.shape}."
        )
    if crop_size <= 0 or crop_size > image.shape[0] or crop_size > image.shape[1]:
        raise ValueError(
            f"ROI crop size must fit inside the processed image. Got crop_size={crop_size}."
        )

    binary_mask = mask > 127
    height, width = image.shape

    if binary_mask.any():
        coords = np.argwhere(binary_mask)
        top = _resolve_positive_crop_start(
            int(coords[:, 0].min()),
            int(coords[:, 0].max()),
            image_size=height,
            crop_size=crop_size,
        )
        left = _resolve_positive_crop_start(
            int(coords[:, 1].min()),
            int(coords[:, 1].max()),
            image_size=width,
            crop_size=crop_size,
        )
    else:
        top = random.randint(0, height - crop_size)
        left = random.randint(0, width - crop_size)

    image_crop = image[top:top + crop_size, left:left + crop_size]
    mask_crop = mask[top:top + crop_size, left:left + crop_size]
    return (
        cv2.resize(image_crop, (width, height), interpolation=cv2.INTER_LINEAR),
        cv2.resize(mask_crop, (width, height), interpolation=cv2.INTER_NEAREST),
    )


class PneumothoraxDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        img_size: int = 256,
        transform=None,
        mask_variant: str | None = None,
        train_crop: dict[str, Any] | None = None,
        splits_path: str | Path | None = None,
    ):
        """
        Args:
            data_dir:      Path to the processed dataset directory.
            split:         "train", "val", or "test".
            img_size:      Runtime resize dimension (256 or 512).
            transform:     albumentations Compose pipeline, or None.
            mask_variant:  Processed mask directory to read from. If omitted,
                           the dataset uses dilated masks for train and
                           original masks for validation/test.
            train_crop:    Optional train-only crop policy config.
            splits_path:   Optional override path for the split manifest.
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.transform = transform

        resolved_splits_path = Path(splits_path) if splits_path is not None else self.data_dir / "splits.json"
        with resolved_splits_path.open(encoding="utf-8") as handle:
            splits = json.load(handle)

        if split not in splits:
            raise ValueError(f"Unknown split '{split}'. Expected one of {list(splits.keys())}.")

        self.split = split
        self.splits_path = resolved_splits_path
        purpose = "train" if split == "train" else "eval"
        self.mask_variant = resolve_mask_variant(mask_variant, purpose=purpose)
        self.mask_dir = resolve_mask_dir(self.data_dir, self.mask_variant)
        if not self.mask_dir.exists():
            raise FileNotFoundError(
                f"Mask variant directory not found: {self.mask_dir}. "
                f"Expected processed dataset to contain separate mask variants."
            )

        self.train_crop = resolve_train_crop_config(
            train_crop,
            split=split,
            img_size=img_size,
        )
        self.image_ids: list[str] = splits[split]

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_id = self.image_ids[idx]

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

        if self.train_crop is not None:
            image, mask = apply_train_roi_crop(
                image,
                mask,
                crop_size=int(self.train_crop["crop_size"]),
            )

        if image.shape[0] != self.img_size or image.shape[1] != self.img_size:
            image = cv2.resize(image, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = image.astype(np.float32) / 255.0
        mask = (mask > 127).astype(np.float32)

        image_tensor = torch.from_numpy(image).unsqueeze(0)
        mask_tensor = torch.from_numpy(mask).unsqueeze(0)

        return image_tensor, mask_tensor
