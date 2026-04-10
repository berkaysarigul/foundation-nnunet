"""Unit tests for the accepted DICOM intensity policy."""

from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.data.dicom_intensity import (
    convert_pixels_to_uint8,
    invert_monochrome1,
    prepare_dicom_pixels_for_png,
    resolve_dicom_read_path,
)


class _FakeDataset:
    def __init__(self, pixel_array, photometric="MONOCHROME2", **attrs):
        self.pixel_array = pixel_array
        self.PhotometricInterpretation = photometric
        for key, value in attrs.items():
            setattr(self, key, value)


class TestDicomIntensityPolicy(unittest.TestCase):
    def test_uint8_pixels_are_preserved_without_rescaling(self) -> None:
        pixels = np.array([[0, 64], [128, 200]], dtype=np.uint8)
        converted = convert_pixels_to_uint8(pixels)
        self.assertTrue(np.array_equal(converted, pixels))

    def test_float_pixels_within_display_range_are_preserved(self) -> None:
        pixels = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        converted = convert_pixels_to_uint8(pixels)
        expected = np.array([[10, 20], [30, 40]], dtype=np.uint8)
        self.assertTrue(np.array_equal(converted, expected))

    def test_out_of_range_pixels_are_min_max_scaled_into_uint8(self) -> None:
        pixels = np.array([[10.0, 20.0], [30.0, 410.0]], dtype=np.float32)
        converted = convert_pixels_to_uint8(pixels)
        expected = np.array([[0, 6], [12, 255]], dtype=np.uint8)
        self.assertTrue(np.array_equal(converted, expected))

    def test_monochrome1_pixels_are_inverted_before_png_export(self) -> None:
        pixels = np.array([[0, 64], [128, 255]], dtype=np.uint8)
        inverted = invert_monochrome1(pixels)
        expected = np.array([[255, 191], [127, 0]], dtype=np.uint8)
        self.assertTrue(np.array_equal(inverted, expected))

    def test_prepare_dicom_pixels_tracks_applied_transforms(self) -> None:
        ds = _FakeDataset(
            pixel_array=np.array([[0, 32], [128, 255]], dtype=np.uint8),
            photometric="MONOCHROME1",
        )
        png_pixels, transforms = prepare_dicom_pixels_for_png(ds)
        self.assertEqual(transforms, ["monochrome1_inversion"])
        self.assertEqual(int(png_pixels.min()), 0)
        self.assertEqual(int(png_pixels.max()), 255)

    def test_windows_long_paths_receive_prefix(self) -> None:
        resolved = resolve_dicom_read_path(Path("data/raw/SIIM-ACR"))
        self.assertTrue(resolved)
        if "\\" in resolved:
            self.assertTrue(resolved.startswith("\\\\?\\") or not resolved[1:3] == ":\\")


if __name__ == "__main__":
    unittest.main()
