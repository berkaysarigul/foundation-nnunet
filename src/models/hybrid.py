"""
hybrid.py — Hybrid Foundation-nnU-Net.

Injects Foundation X Swin-B multi-scale features into a U-Net decoder via
FusionBlocks at each encoder level. Foundation X backbone is frozen; only
the U-Net encoder, fusion blocks, and decoder are trained.
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
        """Keep Foundation X backbone in eval() at all times to prevent
        BatchNorm stats corruption, even when the rest of the model is training."""
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
        with torch.no_grad():
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
