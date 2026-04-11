"""Regression tests for the accepted immediate trainer config surface."""

from __future__ import annotations

import unittest

import torch

from src.training.losses import DiceFocalLoss
from src.training.trainer import (
    build_loss,
    build_optimizer,
    build_scheduler,
    resolve_training_component_config,
    validate_resume_training_components,
)


class TestTrainerConfigSurface(unittest.TestCase):
    def make_cfg(
        self,
        *,
        loss_type: str = "dice_focal",
        optimizer: str = "AdamW",
        scheduler: str = "ReduceLROnPlateau",
    ) -> dict:
        return {
            "loss": {"type": loss_type},
            "training": {
                "learning_rate": 1e-4,
                "weight_decay": 1e-2,
                "optimizer": optimizer,
                "scheduler": scheduler,
            },
        }

    def test_resolve_training_component_config_normalizes_supported_aliases(self) -> None:
        cfg = self.make_cfg(
            loss_type="DiceFocalLoss",
            optimizer="adam",
            scheduler="off",
        )

        self.assertEqual(
            resolve_training_component_config(cfg),
            {
                "loss": "dice_focal",
                "optimizer": "Adam",
                "scheduler": "none",
            },
        )

    def test_build_loss_optimizer_and_scheduler_follow_config(self) -> None:
        cfg = self.make_cfg(optimizer="Adam", scheduler="none")
        training_components = resolve_training_component_config(cfg)
        model = torch.nn.Conv2d(1, 1, kernel_size=1)

        criterion = build_loss(training_components)
        optimizer = build_optimizer(cfg, model, training_components)
        scheduler = build_scheduler(optimizer, training_components)

        self.assertIsInstance(criterion, DiceFocalLoss)
        self.assertIsInstance(optimizer, torch.optim.Adam)
        self.assertIsNone(scheduler)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], cfg["training"]["learning_rate"])
        self.assertAlmostEqual(
            optimizer.param_groups[0]["weight_decay"],
            cfg["training"]["weight_decay"],
        )

    def test_unsupported_config_values_fail_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "loss.type"):
            resolve_training_component_config(self.make_cfg(loss_type="bce"))

        with self.assertRaisesRegex(ValueError, "training.optimizer"):
            resolve_training_component_config(self.make_cfg(optimizer="SGD"))

        with self.assertRaisesRegex(ValueError, "training.scheduler"):
            resolve_training_component_config(self.make_cfg(scheduler="CosineAnnealingLR"))

    def test_resume_training_components_require_metadata(self) -> None:
        expected = {
            "loss": "dice_focal",
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau",
        }

        with self.assertRaisesRegex(ValueError, "missing training_components metadata"):
            validate_resume_training_components({}, expected)

    def test_resume_training_components_reject_mismatched_surface(self) -> None:
        expected = {
            "loss": "dice_focal",
            "optimizer": "AdamW",
            "scheduler": "ReduceLROnPlateau",
        }
        resume_state = {
            "training_components": {
                "loss": "dice_focal",
                "optimizer": "Adam",
                "scheduler": "none",
            }
        }

        with self.assertRaisesRegex(ValueError, "do not match"):
            validate_resume_training_components(resume_state, expected)


if __name__ == "__main__":
    unittest.main()
