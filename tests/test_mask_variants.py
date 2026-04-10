"""Smoke tests for processed mask-variant policy and helpers."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.data.mask_variants import (
    DEFAULT_EVAL_MASK_VARIANT,
    DEFAULT_TRAIN_MASK_VARIANT,
    build_mask_variant_manifest,
    resolve_mask_dir,
    resolve_mask_variant,
    validate_mask_variant,
)


class TestMaskVariantPolicy(unittest.TestCase):
    def test_default_variants_follow_training_and_eval_policy(self) -> None:
        self.assertEqual(resolve_mask_variant(None, purpose="train"), DEFAULT_TRAIN_MASK_VARIANT)
        self.assertEqual(resolve_mask_variant(None, purpose="val"), DEFAULT_EVAL_MASK_VARIANT)
        self.assertEqual(resolve_mask_variant(None, purpose="test"), DEFAULT_EVAL_MASK_VARIANT)
        self.assertEqual(resolve_mask_variant(None, purpose="eval"), DEFAULT_EVAL_MASK_VARIANT)

    def test_explicit_variant_validation(self) -> None:
        self.assertEqual(validate_mask_variant("original_masks"), "original_masks")
        self.assertEqual(validate_mask_variant("dilated_masks"), "dilated_masks")
        with self.assertRaisesRegex(ValueError, "Unknown mask variant"):
            validate_mask_variant("masks")

    def test_mask_directory_resolution(self) -> None:
        data_dir = Path("data/processed/pneumothorax")
        self.assertEqual(
            resolve_mask_dir(data_dir, "original_masks"),
            data_dir / "original_masks",
        )
        self.assertEqual(
            resolve_mask_dir(data_dir, purpose="train"),
            data_dir / DEFAULT_TRAIN_MASK_VARIANT,
        )
        self.assertEqual(
            resolve_mask_dir(data_dir, purpose="eval"),
            data_dir / DEFAULT_EVAL_MASK_VARIANT,
        )

    def test_manifest_records_policy_and_scientific_implications(self) -> None:
        manifest = build_mask_variant_manifest()
        self.assertEqual(manifest["available_variants"], ["original_masks", "dilated_masks"])
        self.assertEqual(manifest["default_train_mask_variant"], "dilated_masks")
        self.assertEqual(manifest["default_eval_mask_variant"], "original_masks")
        self.assertEqual(manifest["final_reporting_mask_variant"], "original_masks")
        self.assertEqual(manifest["dilation_policy"]["type"], "separate_mask_variant")
        self.assertIn("official SIIM-style validation", manifest["scientific_implications"]["original_masks"])
        self.assertIn("training target", manifest["scientific_implications"]["dilated_masks"])


if __name__ == "__main__":
    unittest.main()
