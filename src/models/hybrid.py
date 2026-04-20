"""
hybrid.py — Hybrid Foundation-nnU-Net.

Injects Foundation X Swin-B multi-scale features into a U-Net decoder via
FusionBlocks at each encoder level. Foundation X defaults to a frozen branch,
but gradient behavior follows the active `frozen_backbone` configuration.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone import FoundationXBackbone
from src.models.unet import ConvBlock


class FusionBlock(nn.Module):
    """Fuses Foundation X and U-Net encoder features via concat + 1×1 Conv."""

    def __init__(self, fx_channels: int, unet_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(fx_channels + unet_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, fx_feat: torch.Tensor, unet_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fx_feat:   Foundation X feature (B, fx_ch, H_fx, W_fx)
            unet_feat: U-Net encoder feature (B, unet_ch, H, W)
        Returns:
            Fused feature (B, out_ch, H, W)
        """
        if fx_feat.shape[2:] != unet_feat.shape[2:]:
            fx_feat = F.interpolate(
                fx_feat, size=unet_feat.shape[2:], mode="bilinear", align_corners=False
            )
        return self.conv(torch.cat([fx_feat, unet_feat], dim=1))


def _shape4(name: str, tensor_or_shape) -> tuple[int, int, int, int]:
    """Normalize a tensor or shape-like object to a 4D shape tuple."""
    shape = tuple(tensor_or_shape.shape if hasattr(tensor_or_shape, "shape") else tensor_or_shape)
    if len(shape) != 4:
        raise AssertionError(f"{name} must be 4D, got shape={shape}")
    return shape  # type: ignore[return-value]


def assert_corrected_hybrid_scale_contract(
    *,
    fx0,
    fx1,
    fx2,
    fx3,
    e3,
    e4,
    h16_context,
    h32_context,
) -> None:
    """
    Validate the corrected P1.10 scale contract fixed by D-055/D-056/D-057.

    Expected spatial alignment:
      - fx[0] <-> e3 at H/4
      - fx[1] <-> e4 at H/8
      - fx[2] <-> H/16 bottleneck/context
      - fx[3] <-> dedicated H/32 context head
      - H/32 context reconnects to H/16 through exactly one 2x transition
    """

    shapes = {
        "fx[0]": _shape4("fx[0]", fx0),
        "fx[1]": _shape4("fx[1]", fx1),
        "fx[2]": _shape4("fx[2]", fx2),
        "fx[3]": _shape4("fx[3]", fx3),
        "e3": _shape4("e3", e3),
        "e4": _shape4("e4", e4),
        "h16_context": _shape4("h16_context", h16_context),
        "h32_context": _shape4("h32_context", h32_context),
    }

    batch_size = shapes["fx[0]"][0]
    for name, shape in shapes.items():
        if shape[0] != batch_size:
            raise AssertionError(
                f"Batch mismatch for {name}: expected batch={batch_size}, got shape={shape}"
            )

    def require_same_spatial(left_name: str, right_name: str) -> None:
        left = shapes[left_name]
        right = shapes[right_name]
        if left[2:] != right[2:]:
            raise AssertionError(
                f"Spatial mismatch: {left_name} shape={left} must match {right_name} shape={right}"
            )

    require_same_spatial("fx[0]", "e3")
    require_same_spatial("fx[1]", "e4")
    require_same_spatial("fx[2]", "h16_context")
    require_same_spatial("fx[3]", "h32_context")

    h16 = shapes["h16_context"]
    h32 = shapes["h32_context"]
    if h16[2] != h32[2] * 2 or h16[3] != h32[3] * 2:
        raise AssertionError(
            "Corrected deepest-context reconnect must be exactly one H/32 -> H/16 2x transition: "
            f"h32_context shape={h32}, h16_context shape={h16}"
        )


class HybridFoundationUNet(nn.Module):
    def __init__(
        self,
        backbone_checkpoint: str,
        in_channels: int = 1,
        num_classes: int = 1,
        base_filters: int = 64,
        frozen_backbone: bool = True,
        img_size: int = 256,
    ):
        super().__init__()
        f = base_filters          # 64
        self.frozen_backbone = frozen_backbone

        # Foundation X backbone (frozen by default)
        self.foundation_x = FoundationXBackbone(
            backbone_checkpoint, frozen=frozen_backbone, img_size=img_size
        )

        # U-Net encoder (trainable)
        self.enc1 = ConvBlock(in_channels, f)        # → 64ch
        self.enc2 = ConvBlock(f,     f * 2)          # → 128ch
        self.enc3 = ConvBlock(f * 2, f * 4)          # → 256ch
        self.enc4 = ConvBlock(f * 4, f * 8)          # → 512ch
        self.pool = nn.MaxPool2d(2)

        # Fusion blocks — Foundation X channels confirmed: [128, 256, 512, 1024]
        self.fusion1 = FusionBlock(128,  f,      f)       # 128+64  → 64
        self.fusion2 = FusionBlock(256,  f * 2,  f * 2)   # 256+128 → 128
        self.fusion3 = FusionBlock(512,  f * 4,  f * 4)   # 512+256 → 256
        self.fusion4 = FusionBlock(1024, f * 8,  f * 8)   # 1024+512 → 512

        # Bottleneck
        self.bottleneck = ConvBlock(f * 8, f * 16)        # 512 → 1024

        # Decoder — skip connections use fused features
        self.up4  = nn.ConvTranspose2d(f * 16, f * 8,  kernel_size=2, stride=2)
        self.dec4 = ConvBlock(f * 16, f * 8)   # 512(up) + 512(fused4) → 512

        self.up3  = nn.ConvTranspose2d(f * 8,  f * 4,  kernel_size=2, stride=2)
        self.dec3 = ConvBlock(f * 8,  f * 4)   # 256(up) + 256(fused3) → 256

        self.up2  = nn.ConvTranspose2d(f * 4,  f * 2,  kernel_size=2, stride=2)
        self.dec2 = ConvBlock(f * 4,  f * 2)   # 128(up) + 128(fused2) → 128

        self.up1  = nn.ConvTranspose2d(f * 2,  f,      kernel_size=2, stride=2)
        self.dec1 = ConvBlock(f * 2,  f)        # 64(up)  + 64(fused1)  → 64

        # Output
        self.final = nn.Conv2d(f, num_classes, kernel_size=1)

    def train(self, mode: bool = True):
        """Keep Foundation X backbone in eval() only when that branch is frozen."""
        super().train(mode)
        if self.frozen_backbone:
            self.foundation_x.backbone.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, H, W) grayscale X-ray
        Returns:
            (batch, 1, H, W) sigmoid mask, values in [0, 1]
        """
        # Foundation X feature extraction — frozen, no gradients
        fx = self.foundation_x(x)   # [f1, f2, f3, f4] all (B, C, H, W)

        # U-Net encoder (trainable)
        e1 = self.enc1(x)               # (B, 64,  H,   W)
        e2 = self.enc2(self.pool(e1))   # (B, 128, H/2, W/2)
        e3 = self.enc3(self.pool(e2))   # (B, 256, H/4, W/4)
        e4 = self.enc4(self.pool(e3))   # (B, 512, H/8, W/8)

        # Fusion: Foundation X + U-Net encoder features
        fused1 = self.fusion1(fx[0], e1)   # (B, 64,  H,   W)
        fused2 = self.fusion2(fx[1], e2)   # (B, 128, H/2, W/2)
        fused3 = self.fusion3(fx[2], e3)   # (B, 256, H/4, W/4)
        fused4 = self.fusion4(fx[3], e4)   # (B, 512, H/8, W/8)

        # Bottleneck
        b = self.bottleneck(self.pool(fused4))   # (B, 1024, H/16, W/16)

        # Decoder + fused skip connections
        d4 = self.dec4(torch.cat([self.up4(b),  fused4], dim=1))  # (B, 512, H/8,  W/8)
        d3 = self.dec3(torch.cat([self.up3(d4), fused3], dim=1))  # (B, 256, H/4,  W/4)
        d2 = self.dec2(torch.cat([self.up2(d3), fused2], dim=1))  # (B, 128, H/2,  W/2)
        d1 = self.dec1(torch.cat([self.up1(d2), fused1], dim=1))  # (B, 64,  H,    W)

        return torch.sigmoid(self.final(d1))
