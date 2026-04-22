import unittest
from unittest.mock import patch
import sys
import types

import torch
import torch.nn.functional as F

if "timm" not in sys.modules:
    sys.modules["timm"] = types.SimpleNamespace(create_model=lambda *args, **kwargs: None)

from src.models.backbone import (
    FOUNDATION_X_RGB_MEAN,
    FOUNDATION_X_RGB_STD,
    FoundationXBackbone,
    normalize_foundation_x_input,
)
from src.models.hybrid import HybridFoundationUNet


class DummyFeaturesOnlyBackbone(torch.nn.Module):
    def forward(self, x: torch.Tensor):
        return [x.permute(0, 2, 3, 1).contiguous()]


class DummyFoundationXBackbone(torch.nn.Module):
    def __init__(self, checkpoint_path: str, frozen: bool = True, img_size: int = 32):
        super().__init__()
        self.frozen = frozen
        self.backbone = torch.nn.Identity()
        self.proj1 = torch.nn.Conv2d(3, 128, kernel_size=1)
        self.proj2 = torch.nn.Conv2d(3, 256, kernel_size=1)
        self.proj3 = torch.nn.Conv2d(3, 512, kernel_size=1)
        self.proj4 = torch.nn.Conv2d(3, 1024, kernel_size=1)

        if frozen:
            for param in self.parameters():
                param.requires_grad = False
            self.backbone.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor):
        x = normalize_foundation_x_input(x)
        with torch.set_grad_enabled(not self.frozen):
            return [
                self.proj1(F.avg_pool2d(x, 4)),
                self.proj2(F.avg_pool2d(x, 8)),
                self.proj3(F.avg_pool2d(x, 16)),
                self.proj4(F.avg_pool2d(x, 32)),
            ]


class FoundationXBackboneNormalizationTests(unittest.TestCase):
    def test_normalize_foundation_x_input_repeats_and_normalizes_per_channel(self):
        x = torch.tensor([[[[0.0, 1.0]]]], dtype=torch.float32)

        normalized = normalize_foundation_x_input(x)

        self.assertEqual(normalized.shape, (1, 3, 1, 2))
        for channel_index, (mean, std) in enumerate(
            zip(FOUNDATION_X_RGB_MEAN, FOUNDATION_X_RGB_STD)
        ):
            expected = (x[:, :1] - mean) / std
            torch.testing.assert_close(
                normalized[:, channel_index : channel_index + 1],
                expected,
            )

    def test_normalize_foundation_x_input_rejects_non_grayscale_input(self):
        x = torch.randn(2, 3, 8, 8)

        with self.assertRaises(AssertionError):
            normalize_foundation_x_input(x)


class FoundationXBackboneGradientPolicyTests(unittest.TestCase):
    def _make_backbone_without_checkpoint(self, frozen: bool) -> FoundationXBackbone:
        backbone = FoundationXBackbone.__new__(FoundationXBackbone)
        torch.nn.Module.__init__(backbone)
        backbone.backbone = DummyFeaturesOnlyBackbone()
        backbone.frozen = frozen
        return backbone

    def test_foundation_x_backbone_forward_applies_branch_normalization(self):
        backbone = self._make_backbone_without_checkpoint(frozen=False)
        x = torch.tensor([[[[0.0, 1.0], [0.5, 0.25]]]], dtype=torch.float32)

        features = backbone(x)

        torch.testing.assert_close(features[0], normalize_foundation_x_input(x))

    def test_foundation_x_backbone_forward_disables_grad_when_frozen(self):
        backbone = self._make_backbone_without_checkpoint(frozen=True)
        x = torch.randn(2, 1, 8, 8, requires_grad=True)

        features = backbone(x)

        self.assertFalse(features[0].requires_grad)

    def test_foundation_x_backbone_forward_preserves_grad_when_unfrozen(self):
        backbone = self._make_backbone_without_checkpoint(frozen=False)
        x = torch.randn(2, 1, 8, 8, requires_grad=True)

        features = backbone(x)

        self.assertTrue(features[0].requires_grad)


class HybridGradientFlowTests(unittest.TestCase):
    def _make_model(self, *, frozen_backbone: bool) -> HybridFoundationUNet:
        with patch("src.models.hybrid.FoundationXBackbone", DummyFoundationXBackbone):
            model = HybridFoundationUNet(
                backbone_checkpoint="unused.pth",
                in_channels=1,
                num_classes=1,
                base_filters=4,
                frozen_backbone=frozen_backbone,
                img_size=32,
            )
        model.train()
        return model

    def test_frozen_hybrid_backbone_receives_no_gradients(self):
        model = self._make_model(frozen_backbone=True)
        x = torch.randn(2, 1, 32, 32)

        output = model(x)
        output.mean().backward()

        backbone_grads = [
            param.grad
            for param in model.foundation_x.parameters()
            if param.requires_grad
        ]
        self.assertEqual(backbone_grads, [])

    def test_unfrozen_hybrid_backbone_receives_gradients(self):
        model = self._make_model(frozen_backbone=False)
        x = torch.randn(2, 1, 32, 32)

        output = model(x)
        output.mean().backward()

        backbone_grads = [
            param.grad
            for param in model.foundation_x.parameters()
            if param.requires_grad
        ]
        self.assertTrue(backbone_grads)
        self.assertTrue(
            any(grad is not None and float(grad.abs().sum().item()) > 0.0 for grad in backbone_grads)
        )


if __name__ == "__main__":
    unittest.main()
