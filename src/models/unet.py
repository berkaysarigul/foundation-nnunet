"""
unet.py — Baseline U-Net for binary segmentation.

Architecture:
    Encoder:     4 levels (64 → 128 → 256 → 512), MaxPool2d downsampling
    Bottleneck:  512 → 1024
    Decoder:     4 levels, ConvTranspose2d upsampling + skip connections (concat)
    Output:      1x1 Conv → Sigmoid
"""

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    """Two consecutive Conv3x3 → BatchNorm → ReLU blocks."""

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 1, base_filters: int = 64):
        super().__init__()
        f = base_filters  # 64

        # Encoder
        self.enc1 = ConvBlock(in_channels, f)        # → 64ch,  H×W
        self.enc2 = ConvBlock(f,     f * 2)          # → 128ch, H/2×W/2
        self.enc3 = ConvBlock(f * 2, f * 4)          # → 256ch, H/4×W/4
        self.enc4 = ConvBlock(f * 4, f * 8)          # → 512ch, H/8×W/8
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(f * 8, f * 16)   # → 1024ch, H/16×W/16

        # Decoder
        self.up4  = nn.ConvTranspose2d(f * 16, f * 8,  kernel_size=2, stride=2)
        self.dec4 = ConvBlock(f * 16, f * 8)         # 512+512 → 512

        self.up3  = nn.ConvTranspose2d(f * 8,  f * 4,  kernel_size=2, stride=2)
        self.dec3 = ConvBlock(f * 8,  f * 4)         # 256+256 → 256

        self.up2  = nn.ConvTranspose2d(f * 4,  f * 2,  kernel_size=2, stride=2)
        self.dec2 = ConvBlock(f * 4,  f * 2)         # 128+128 → 128

        self.up1  = nn.ConvTranspose2d(f * 2,  f,      kernel_size=2, stride=2)
        self.dec1 = ConvBlock(f * 2,  f)             # 64+64   → 64

        # Output
        self.final = nn.Conv2d(f, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 1, H, W)
        Returns:
            (batch, 1, H, W) — sigmoid output, values in [0, 1]
        """
        # Encoder
        e1 = self.enc1(x)                            # (B, 64,   H,    W)
        e2 = self.enc2(self.pool(e1))                # (B, 128,  H/2,  W/2)
        e3 = self.enc3(self.pool(e2))                # (B, 256,  H/4,  W/4)
        e4 = self.enc4(self.pool(e3))                # (B, 512,  H/8,  W/8)

        # Bottleneck
        b = self.bottleneck(self.pool(e4))           # (B, 1024, H/16, W/16)

        # Decoder + skip connections
        d4 = self.dec4(torch.cat([self.up4(b),  e4], dim=1))  # (B, 512,  H/8,  W/8)
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))  # (B, 256,  H/4,  W/4)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # (B, 128,  H/2,  W/2)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # (B, 64,   H,    W)

        return torch.sigmoid(self.final(d1))
