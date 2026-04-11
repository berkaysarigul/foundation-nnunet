"""Regression tests for the pretrained ResNet34 encoder U-Net path."""

from __future__ import annotations

import importlib.util
import unittest
from unittest.mock import patch

import torch

from src.evaluation.evaluate import build_model as build_eval_model
from src.models.resnet34_unet import PretrainedResNet34UNet
from src.training.trainer import build_model as build_train_model

HAS_TORCHVISION = importlib.util.find_spec("torchvision") is not None


class DummyPretrainedModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.args = args
        self.kwargs = kwargs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class TestPretrainedResNet34UNet(unittest.TestCase):
    def make_cfg(self, model_type: str = "pretrained_resnet34_unet") -> dict:
        return {
            "model": {
                "type": model_type,
                "in_channels": 1,
                "num_classes": 1,
                "base_filters": 64,
            },
            "data": {
                "input_size": 512,
            },
            "foundation_x": {
                "checkpoint_path": "checkpoints/foundation_x.pth",
                "frozen": True,
            },
        }

    @unittest.skipUnless(HAS_TORCHVISION, "torchvision is not installed in this environment")
    def test_forward_preserves_resolution_for_grayscale_input(self) -> None:
        model = PretrainedResNet34UNet(pretrained=False)
        model.eval()
        x = torch.randn(1, 1, 256, 256)

        with torch.no_grad():
            y = model(x)

        self.assertEqual(y.shape, (1, 1, 256, 256))
        self.assertTrue(torch.all(y >= 0.0).item())
        self.assertTrue(torch.all(y <= 1.0).item())

    @unittest.skipUnless(HAS_TORCHVISION, "torchvision is not installed in this environment")
    def test_grayscale_adaptation_stays_inside_model_path(self) -> None:
        model = PretrainedResNet34UNet(pretrained=False)

        self.assertEqual(model.stem[0].in_channels, 1)
        self.assertEqual(model.stem[0].out_channels, 64)

    def test_constructor_reports_missing_torchvision_dependency(self) -> None:
        if HAS_TORCHVISION:
            self.skipTest("torchvision is installed; missing-dependency path is not active.")

        with self.assertRaisesRegex(ModuleNotFoundError, "requires torchvision"):
            PretrainedResNet34UNet(pretrained=False)

    def test_trainer_and_evaluator_factories_accept_pretrained_model_type(self) -> None:
        cfg = self.make_cfg()

        with patch("src.models.resnet34_unet.PretrainedResNet34UNet", DummyPretrainedModel):
            train_model = build_train_model(cfg)
            eval_model = build_eval_model(cfg, "pretrained_resnet34_unet")

        self.assertIsInstance(train_model, DummyPretrainedModel)
        self.assertIsInstance(eval_model, DummyPretrainedModel)
        self.assertEqual(train_model.kwargs["in_channels"], 1)
        self.assertEqual(eval_model.kwargs["num_classes"], 1)


if __name__ == "__main__":
    unittest.main()
