"""Regression tests for the first authoritative pretrained baseline config."""

from __future__ import annotations

import unittest
from pathlib import Path

import yaml

from src.training.trainer import resolve_training_component_config


class TestAuthoritativePretrainedConfig(unittest.TestCase):
    def test_config_locks_the_fixed_pretrained_baseline_protocol(self) -> None:
        config_path = Path("configs/pretrained_resnet34_authoritative.yaml")
        with config_path.open(encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)

        self.assertEqual(cfg["model"]["type"], "pretrained_resnet34_unet")
        self.assertEqual(cfg["data"]["processed_dir"], "data/processed/pneumothorax_trusted_v1")
        self.assertEqual(cfg["data"]["input_size"], 512)
        self.assertEqual(cfg["data"]["train_mask_variant"], "dilated_masks")
        self.assertEqual(cfg["data"]["eval_mask_variant"], "original_masks")
        self.assertEqual(cfg["training"]["batch_size"], 8)
        self.assertEqual(cfg["training"]["epochs"], 150)
        self.assertEqual(cfg["training"]["early_stopping_patience"], 30)
        self.assertEqual(cfg["training"]["learning_rate"], 0.0001)
        self.assertEqual(cfg["training"]["weight_decay"], 0.01)
        self.assertEqual(cfg["seed"], 42)
        self.assertEqual(cfg["device"], "auto")
        self.assertEqual(
            resolve_training_component_config(cfg),
            {
                "loss": "dice_focal",
                "optimizer": "AdamW",
                "scheduler": "ReduceLROnPlateau",
            },
        )
        self.assertEqual(cfg["selection"]["metric"], "val_dice_pos_mean")
        self.assertEqual(cfg["selection"]["postprocess"], "none")
        self.assertEqual(cfg["selection"]["threshold_candidates"][9], 0.5)


if __name__ == "__main__":
    unittest.main()
