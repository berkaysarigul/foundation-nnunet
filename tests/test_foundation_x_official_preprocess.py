"""Tests for Foundation_X official preprocessing variants."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.data.foundation_x_official_preprocess import (
    PREPROCESS_VARIANTS,
    preprocess_image_for_foundation_x,
)


class TestFoundationXOfficialPreprocess(unittest.TestCase):
    def test_preprocess_variants_emit_expected_shapes_and_stats(self):
        with tempfile.TemporaryDirectory() as td:
            image_path = Path(td) / "case_0000.png"
            arr = np.linspace(0, 255, 32 * 48, dtype=np.uint8).reshape(32, 48)
            Image.fromarray(arr, mode="L").save(image_path)

            expected_shapes = {
                "official_siim_224": [1, 3, 224, 224],
                "official_siim_512": [1, 3, 512, 512],
                "current_pr9a_512": [1, 3, 512, 512],
            }
            for variant in PREPROCESS_VARIANTS:
                result = preprocess_image_for_foundation_x(image_path, variant)
                self.assertEqual(list(result.tensor.shape), expected_shapes[variant])
                self.assertEqual(str(result.tensor.dtype), "torch.float32")
                self.assertIn("stats_before_normalization", result.stats)
                self.assertIn("stats_after_normalization", result.stats)
                self.assertEqual(result.stats["input_file_path"], str(image_path))
                self.assertLess(
                    result.stats["stats_after_normalization"]["min"],
                    result.stats["stats_after_normalization"]["max"],
                )


if __name__ == "__main__":
    unittest.main()
