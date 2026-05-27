"""Lightweight logits-output U-Net for the Foundation X prior refiner."""

from __future__ import annotations

import torch
import torch.nn as nn


def _make_norm(channels: int, norm: str) -> nn.Module:
    if norm == "batch":
        return nn.BatchNorm2d(channels)
    if norm == "group":
        groups = min(8, channels)
        while channels % groups != 0:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    raise ValueError("norm must be 'batch' or 'group'.")


class RefinerConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, norm: str) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(out_channels, norm),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(out_channels, norm),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FoundationXPriorRefinerUNet(nn.Module):
    """Simple 2D U-Net that returns raw logits for binary segmentation."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 32,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        if in_channels <= 0:
            raise ValueError("in_channels must be positive.")
        if out_channels <= 0:
            raise ValueError("out_channels must be positive.")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive.")

        f = int(base_channels)
        self.enc1 = RefinerConvBlock(in_channels, f, norm=norm)
        self.enc2 = RefinerConvBlock(f, f * 2, norm=norm)
        self.enc3 = RefinerConvBlock(f * 2, f * 4, norm=norm)
        self.enc4 = RefinerConvBlock(f * 4, f * 8, norm=norm)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = RefinerConvBlock(f * 8, f * 16, norm=norm)

        self.up4 = nn.ConvTranspose2d(f * 16, f * 8, kernel_size=2, stride=2)
        self.dec4 = RefinerConvBlock(f * 16, f * 8, norm=norm)
        self.up3 = nn.ConvTranspose2d(f * 8, f * 4, kernel_size=2, stride=2)
        self.dec3 = RefinerConvBlock(f * 8, f * 4, norm=norm)
        self.up2 = nn.ConvTranspose2d(f * 4, f * 2, kernel_size=2, stride=2)
        self.dec2 = RefinerConvBlock(f * 4, f * 2, norm=norm)
        self.up1 = nn.ConvTranspose2d(f * 2, f, kernel_size=2, stride=2)
        self.dec1 = RefinerConvBlock(f * 2, f, norm=norm)

        self.final = nn.Conv2d(f, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.final(d1)


RefinerUNet = FoundationXPriorRefinerUNet
