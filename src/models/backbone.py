"""
backbone.py — Foundation X Swin-B feature extractor.

Loads pretrained Swin-B weights from the Foundation X checkpoint and exposes
4 multi-scale feature maps for use in the hybrid decoder.

Checkpoint key structure (discovered):
    ckpt['model']['backbone.0.*']  → Swin-B weights
    embed_dim=128, patch_size=4, window_size=7
    Stage output channels: [128, 256, 512, 1024]

Key remapping required:
    Checkpoint: layers.N.blocks.*  →  timm: layers_N.blocks.*
"""

import re

import timm
import torch
import torch.nn as nn


def _remap_key(key: str) -> str:
    """Remap checkpoint key format to timm's key format.

    Two differences:
    1. Checkpoint 'layers.N.' → timm 'layers_N.'  (dot vs underscore index)
    2. Checkpoint places downsample at end of layer N;
       timm places it at start of layer N+1.
       So 'layers.N.downsample.*' → 'layers_(N+1).downsample.*'
    """
    # Shift downsample from layer N to layer N+1
    def shift_downsample(m: re.Match) -> str:
        return f"layers_{int(m.group(1)) + 1}.downsample."

    key = re.sub(r"layers\.(\d+)\.downsample\.", shift_downsample, key)
    # Remaining layers.N. → layers_N.
    key = re.sub(r"layers\.(\d+)\.", r"layers_\1.", key)
    return key


class FoundationXBackbone(nn.Module):
    def __init__(self, checkpoint_path: str, frozen: bool = True, img_size: int = 256):
        """
        Args:
            checkpoint_path: Path to Foundation X .pth checkpoint.
            frozen:          If True, backbone weights are not updated during training.
            img_size:        Input image size (256 or 512). Must match training config.
        """
        super().__init__()

        # 1. Create Swin-B feature extractor via timm
        #    img_size must match the actual input size — Swin precomputes window attention
        #    masks at init time for a fixed spatial resolution.
        self.backbone = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=img_size,
        )

        # 2. Load checkpoint
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model_weights = ckpt["model"]

        # 3. Strip "backbone.0." prefix and remap "layers.N." → "layers_N."
        prefix = "backbone.0."
        backbone_weights = {
            _remap_key(k[len(prefix):]): v
            for k, v in model_weights.items()
            if k.startswith(prefix)
        }

        # strict=False: timm features_only adds extra norm layers not in the checkpoint
        missing, unexpected = self.backbone.load_state_dict(backbone_weights, strict=False)
        print(f"[FoundationXBackbone] Loaded {len(backbone_weights)} keys. "
              f"Missing: {len(missing)}, Unexpected: {len(unexpected)}")

        # 4. Freeze
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
                f1: (batch, 128,  H/4,  W/4)   Stage 1
                f2: (batch, 256,  H/8,  W/8)   Stage 2
                f3: (batch, 512,  H/16, W/16)  Stage 3
                f4: (batch, 1024, H/32, W/32)  Stage 4
        """
        # Grayscale → RGB: Swin-B expects 3-channel input
        x = x.repeat(1, 3, 1, 1)  # (B, 1, H, W) → (B, 3, H, W)

        with torch.set_grad_enabled(not self.frozen):
            # timm Swin outputs (B, H, W, C) — permute to (B, C, H, W) for conv decoders
            features = [f.permute(0, 3, 1, 2).contiguous() for f in self.backbone(x)]

        return features
