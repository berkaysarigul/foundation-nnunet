"""
backbone.py - Foundation X Swin-B feature extractor.

Loads pretrained Swin-B weights from the Foundation X checkpoint and exposes
4 multi-scale feature maps for use in the hybrid decoder.

Checkpoint key structure (discovered):
    ckpt['model']['backbone.0.*'] -> Swin-B weights
    embed_dim=128, patch_size=4, window_size=7
    Stage output channels: [128, 256, 512, 1024]

Key remapping required:
    Checkpoint: layers.N.blocks.* -> timm: layers_N.blocks.*
"""

from __future__ import annotations

import re

import timm
import torch
import torch.nn as nn


FOUNDATION_X_RGB_MEAN = (0.485, 0.456, 0.406)
FOUNDATION_X_RGB_STD = (0.229, 0.224, 0.225)


def _remap_key(key: str) -> str:
    """Remap checkpoint key format to timm's key format."""

    def shift_downsample(match: re.Match) -> str:
        return f"layers_{int(match.group(1)) + 1}.downsample."

    key = re.sub(r"layers\.(\d+)\.downsample\.", shift_downsample, key)
    key = re.sub(r"layers\.(\d+)\.", r"layers_\1.", key)
    return key


def repeat_grayscale_to_rgb(x: torch.Tensor) -> torch.Tensor:
    """Repeat a grayscale BCHW tensor across 3 RGB channels."""
    if x.ndim != 4:
        raise AssertionError(f"Expected BCHW input, got shape={tuple(x.shape)}")
    if x.shape[1] != 1:
        raise AssertionError(
            "Foundation X branch expects grayscale BCHW input before RGB adaptation; "
            f"got shape={tuple(x.shape)}"
        )
    return x.repeat(1, 3, 1, 1)


def normalize_foundation_x_input(x: torch.Tensor) -> torch.Tensor:
    """Build the Foundation X branch-specific RGB-normalized input view."""
    rgb = repeat_grayscale_to_rgb(x)
    mean = torch.as_tensor(
        FOUNDATION_X_RGB_MEAN,
        dtype=rgb.dtype,
        device=rgb.device,
    ).view(1, 3, 1, 1)
    std = torch.as_tensor(
        FOUNDATION_X_RGB_STD,
        dtype=rgb.dtype,
        device=rgb.device,
    ).view(1, 3, 1, 1)
    return (rgb - mean) / std


class FoundationXBackbone(nn.Module):
    def __init__(self, checkpoint_path: str, frozen: bool = True, img_size: int = 256):
        """
        Args:
            checkpoint_path: Path to Foundation X .pth checkpoint.
            frozen: If True, backbone weights are not updated during training.
            img_size: Input image size (256 or 512). Must match training config.
        """
        super().__init__()

        self.backbone = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=img_size,
        )

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_weights = ckpt["model"]

        prefix = "backbone.0."
        backbone_weights = {
            _remap_key(k[len(prefix):]): v
            for k, v in model_weights.items()
            if k.startswith(prefix)
        }

        missing, unexpected = self.backbone.load_state_dict(backbone_weights, strict=False)
        print(
            f"[FoundationXBackbone] Loaded {len(backbone_weights)} keys. "
            f"Missing: {len(missing)}, Unexpected: {len(unexpected)}"
        )

        if frozen:
            for param in self.backbone.parameters():
                param.requires_grad = False
            self.backbone.eval()

        self.frozen = frozen

    def train(self, mode: bool = True):
        """Keep backbone in eval() only when the Foundation X branch is frozen."""
        super().train(mode)
        if self.frozen:
            self.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            x: (batch, 1, H, W) grayscale image

        Returns:
            List of 4 feature maps:
                f1: (batch, 128, H/4, W/4)
                f2: (batch, 256, H/8, W/8)
                f3: (batch, 512, H/16, W/16)
                f4: (batch, 1024, H/32, W/32)
        """
        x = normalize_foundation_x_input(x)

        with torch.set_grad_enabled(not self.frozen):
            features = [f.permute(0, 3, 1, 2).contiguous() for f in self.backbone(x)]

        return features
