"""
Foundation_X official preprocessing variants for PR-9A-FIX diagnostics.

The official segmentation datasets use OpenCV-style 3-channel image loading,
Albumentations ImageNet normalization, `ToTensorV2()`, and then divide the
tensor by 255 in `__getitem__`. The `official_siim_*` variants intentionally
preserve that behavior, including `Normalize(..., max_pixel_value=1)`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


FOUNDATION_X_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
FOUNDATION_X_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)

PREPROCESS_VARIANTS = ("official_siim_224", "official_siim_512", "current_pr9a_512")


@dataclass
class PreprocessResult:
    """Preprocessed tensor plus detailed stats for diagnostics."""

    tensor: torch.Tensor
    stats: dict[str, Any]


def available_preprocess_variants() -> tuple[str, ...]:
    return PREPROCESS_VARIANTS


def preprocess_image_for_foundation_x(image_path: Path, variant: str) -> PreprocessResult:
    """Preprocess one image using a named PR-9A-FIX variant."""
    variant = str(variant)
    if variant == "official_siim_224":
        return _preprocess_official_siim(Path(image_path), output_size=224, variant=variant)
    if variant == "official_siim_512":
        return _preprocess_official_siim(Path(image_path), output_size=512, variant=variant)
    if variant == "current_pr9a_512":
        return _preprocess_current_pr9a(Path(image_path), output_size=512, variant=variant)
    raise ValueError(
        f"Unknown preprocessing variant {variant!r}. "
        f"Expected one of {', '.join(PREPROCESS_VARIANTS)}."
    )


def _preprocess_official_siim(image_path: Path, output_size: int, variant: str) -> PreprocessResult:
    """
    Reproduce official `SIIM_PXSDataset.__getitem__` image preprocessing.

    Upstream behavior:
    - `cv2.imread(imagePath, cv2.IMREAD_COLOR)` returns HWC BGR uint8.
    - `cv2.resize(..., interpolation=cv2.INTER_AREA)`.
    - Albumentations Normalize with max_pixel_value=1.
    - ToTensorV2 produces CHW.
    - The dataset then does `img = img / 255`.
    """
    image_bgr, read_backend = _read_opencv_color(image_path)
    original_shape = list(image_bgr.shape)
    resized = _resize_cv2(image_bgr, output_size=output_size, interpolation="area")
    before_stats = _array_stats(resized)

    arr = resized.astype(np.float32)
    normalized = (arr - FOUNDATION_X_MEAN.reshape(1, 1, 3)) / FOUNDATION_X_STD.reshape(1, 1, 3)
    chw = np.transpose(normalized, (2, 0, 1)) / 255.0
    tensor = torch.from_numpy(chw.astype(np.float32)).unsqueeze(0)

    stats = {
        "variant": variant,
        "input_file_path": str(image_path),
        "read_backend": read_backend,
        "channel_order": "opencv_bgr",
        "original_image_shape": original_shape,
        "resized_image_shape": list(resized.shape),
        "output_tensor_shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "resize_interpolation": "cv2.INTER_AREA",
        "normalization": {
            "mean": FOUNDATION_X_MEAN.tolist(),
            "std": FOUNDATION_X_STD.tolist(),
            "max_pixel_value": 1,
            "post_tensor_divide_by_255": True,
        },
        "stats_before_normalization": before_stats,
        "stats_after_normalization": _tensor_stats(tensor),
    }
    return PreprocessResult(tensor=tensor, stats=stats)


def _preprocess_current_pr9a(image_path: Path, output_size: int, variant: str) -> PreprocessResult:
    """
    Recreate the reverse-engineered PR-9A tensor view for the official model.

    The old PR-9A script read a grayscale PNG, scaled to [0, 1], then the
    reverse-engineered model repeated to RGB and applied ImageNet normalization.
    This variant emits the resulting 3-channel tensor directly.
    """
    image = Image.open(image_path).convert("L")
    original_shape = [image.size[1], image.size[0], 1]
    if image.size != (output_size, output_size):
        image = image.resize((output_size, output_size), Image.BILINEAR)
    gray01 = np.asarray(image, dtype=np.float32) / 255.0
    rgb01 = np.repeat(gray01[:, :, None], 3, axis=2)
    before_stats = _array_stats(rgb01)
    normalized = (rgb01 - FOUNDATION_X_MEAN.reshape(1, 1, 3)) / FOUNDATION_X_STD.reshape(1, 1, 3)
    chw = np.transpose(normalized, (2, 0, 1))
    tensor = torch.from_numpy(chw.astype(np.float32)).unsqueeze(0)

    stats = {
        "variant": variant,
        "input_file_path": str(image_path),
        "read_backend": "PIL.convert('L')",
        "channel_order": "rgb_repeated_from_grayscale",
        "original_image_shape": original_shape,
        "resized_image_shape": [output_size, output_size, 3],
        "output_tensor_shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "resize_interpolation": "PIL.Image.BILINEAR",
        "normalization": {
            "mean": FOUNDATION_X_MEAN.tolist(),
            "std": FOUNDATION_X_STD.tolist(),
            "input_range": "[0, 1]",
            "post_tensor_divide_by_255": False,
        },
        "stats_before_normalization": before_stats,
        "stats_after_normalization": _tensor_stats(tensor),
    }
    return PreprocessResult(tensor=tensor, stats=stats)


def _read_opencv_color(image_path: Path) -> tuple[np.ndarray, str]:
    try:
        import cv2  # type: ignore

        arr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if arr is not None:
            return arr, "cv2.imread(..., IMREAD_COLOR)"
    except Exception:
        pass

    pil = Image.open(image_path).convert("RGB")
    rgb = np.asarray(pil, dtype=np.uint8)
    bgr = rgb[:, :, ::-1].copy()
    return bgr, "PIL RGB fallback converted to BGR"


def _resize_cv2(arr: np.ndarray, output_size: int, interpolation: str) -> np.ndarray:
    try:
        import cv2  # type: ignore

        interp = cv2.INTER_AREA if interpolation == "area" else cv2.INTER_LINEAR
        return cv2.resize(arr, (output_size, output_size), interpolation=interp)
    except Exception:
        pil_mode = "RGB" if arr.ndim == 3 and arr.shape[2] == 3 else "L"
        if pil_mode == "RGB":
            rgb = arr[:, :, ::-1]
            pil = Image.fromarray(rgb.astype(np.uint8), mode="RGB")
            resized = pil.resize((output_size, output_size), Image.Resampling.BOX)
            return np.asarray(resized, dtype=np.uint8)[:, :, ::-1].copy()
        pil = Image.fromarray(arr.astype(np.uint8), mode="L")
        resized = pil.resize((output_size, output_size), Image.Resampling.BOX)
        return np.asarray(resized, dtype=np.uint8)


def _array_stats(arr: np.ndarray) -> dict[str, Any]:
    arr_float = arr.astype(np.float32, copy=False)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(np.min(arr_float)),
        "max": float(np.max(arr_float)),
        "mean": float(np.mean(arr_float)),
        "std": float(np.std(arr_float)),
    }


def _tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach().float().cpu()
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "min": float(detached.min().item()),
        "max": float(detached.max().item()),
        "mean": float(detached.mean().item()),
        "std": float(detached.std(unbiased=False).item()),
    }


__all__ = [
    "FOUNDATION_X_MEAN",
    "FOUNDATION_X_STD",
    "PREPROCESS_VARIANTS",
    "PreprocessResult",
    "available_preprocess_variants",
    "preprocess_image_for_foundation_x",
]
