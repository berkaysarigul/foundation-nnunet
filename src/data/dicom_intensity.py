"""DICOM intensity handling helpers for SIIM preprocessing."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pydicom
from pydicom.pixels import apply_modality_lut, apply_voi_lut


def resolve_dicom_read_path(dicom_path: str | Path) -> str:
    """Return a path string safe for pydicom on the current platform."""
    resolved = str(Path(dicom_path).resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return "\\\\?\\" + resolved
    return resolved


def read_dicom_dataset(dicom_path: str | Path, *, stop_before_pixels: bool = False):
    """Read a DICOM file with Windows long-path support when needed."""
    return pydicom.dcmread(
        resolve_dicom_read_path(dicom_path),
        stop_before_pixels=stop_before_pixels,
    )


def has_modality_transform(ds) -> bool:
    """Return True when modality LUT or rescale parameters are present."""
    return any(
        getattr(ds, key, None) is not None
        for key in ("ModalityLUTSequence", "RescaleSlope", "RescaleIntercept")
    )


def has_voi_transform(ds) -> bool:
    """Return True when VOI LUT or windowing metadata are present."""
    return any(
        getattr(ds, key, None) is not None
        for key in ("VOILUTSequence", "WindowCenter", "WindowWidth")
    )


def invert_monochrome1(pixels: np.ndarray) -> np.ndarray:
    """Invert a MONOCHROME1 image into a MONOCHROME2-style intensity ordering."""
    pixels = np.asarray(pixels)
    return pixels.max() + pixels.min() - pixels


def convert_pixels_to_uint8(pixels: np.ndarray) -> np.ndarray:
    """Convert a 2D pixel array into uint8 for PNG export.

    If the input is already non-negative and within [0, 255], preserve those
    values exactly. Otherwise fall back to linear min-max scaling into uint8.
    """
    array = np.asarray(pixels)
    array = np.squeeze(array)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D image array, got shape {array.shape}")

    if np.min(array) >= 0 and np.max(array) <= 255:
        return array.astype(np.uint8, copy=False)

    array = array.astype(np.float32)
    array -= float(array.min())
    max_value = float(array.max())
    if max_value > 0:
        array = (array / max_value) * 255.0
    return np.clip(array, 0, 255).astype(np.uint8)


def prepare_dicom_pixels_for_png(ds) -> tuple[np.ndarray, list[str]]:
    """Convert a DICOM dataset into a PNG-ready uint8 image and applied transforms."""
    pixels = ds.pixel_array
    applied_transforms: list[str] = []

    if has_modality_transform(ds):
        pixels = apply_modality_lut(pixels, ds)
        applied_transforms.append("modality_lut")

    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        pixels = invert_monochrome1(np.asarray(pixels))
        applied_transforms.append("monochrome1_inversion")

    if has_voi_transform(ds):
        pixels = apply_voi_lut(pixels, ds)
        applied_transforms.append("voi_lut")

    return convert_pixels_to_uint8(np.asarray(pixels)), applied_transforms

