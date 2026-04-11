"""
resnet34_unet.py - ImageNet-pretrained ResNet34 encoder U-Net.

This model keeps grayscale adaptation inside the model path by replacing the
first ResNet convolution with a 1-channel equivalent initialized from the
pretrained RGB filters. No separate RGB dataset pipeline is required.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.models.unet import ConvBlock


def _adapt_conv1_to_grayscale(conv1: nn.Conv2d) -> nn.Conv2d:
    grayscale_conv = nn.Conv2d(
        in_channels=1,
        out_channels=conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=conv1.bias is not None,
    )
    with torch.no_grad():
        grayscale_conv.weight.copy_(conv1.weight.mean(dim=1, keepdim=True))
        if conv1.bias is not None and grayscale_conv.bias is not None:
            grayscale_conv.bias.copy_(conv1.bias)
    return grayscale_conv


def _build_resnet34_encoder(pretrained: bool) -> nn.Module:
    ResNet34_Weights, resnet34 = _require_torchvision_resnet34()
    weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    encoder = resnet34(weights=weights)
    encoder.conv1 = _adapt_conv1_to_grayscale(encoder.conv1)
    return encoder


def _require_torchvision_resnet34():
    try:
        from torchvision.models import ResNet34_Weights, resnet34
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "PretrainedResNet34UNet requires torchvision. Install the repository "
            "training dependencies before using model.type='pretrained_resnet34_unet'."
        ) from exc
    return ResNet34_Weights, resnet34


class DecoderBlock(nn.Module):
    """Upsample, optionally concatenate a skip feature, then refine with ConvBlock."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=2,
            stride=2,
        )
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class PretrainedResNet34UNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        base_filters: int = 64,
        pretrained: bool = True,
    ):
        super().__init__()
        if in_channels != 1:
            raise ValueError(
                "PretrainedResNet34UNet currently supports grayscale input only "
                "because the trusted dataset pipeline is single-channel."
            )

        encoder = _build_resnet34_encoder(pretrained=pretrained)
        f = base_filters
        final_filters = max(f // 2, 32)

        self.stem = nn.Sequential(
            encoder.conv1,
            encoder.bn1,
            encoder.relu,
        )
        self.maxpool = encoder.maxpool
        self.encoder1 = encoder.layer1
        self.encoder2 = encoder.layer2
        self.encoder3 = encoder.layer3
        self.encoder4 = encoder.layer4

        self.dec4 = DecoderBlock(in_channels=512, skip_channels=256, out_channels=f * 4)
        self.dec3 = DecoderBlock(in_channels=f * 4, skip_channels=128, out_channels=f * 2)
        self.dec2 = DecoderBlock(in_channels=f * 2, skip_channels=64, out_channels=f)
        self.dec1 = DecoderBlock(in_channels=f, skip_channels=64, out_channels=f)
        self.dec0 = DecoderBlock(in_channels=f, skip_channels=0, out_channels=final_filters)
        self.final = nn.Conv2d(final_filters, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stem = self.stem(x)                       # (B, 64, H/2,  W/2)
        e1 = self.encoder1(self.maxpool(stem))   # (B, 64, H/4,  W/4)
        e2 = self.encoder2(e1)                   # (B, 128, H/8,  W/8)
        e3 = self.encoder3(e2)                   # (B, 256, H/16, W/16)
        e4 = self.encoder4(e3)                   # (B, 512, H/32, W/32)

        d4 = self.dec4(e4, e3)                   # (B, 256, H/16, W/16)
        d3 = self.dec3(d4, e2)                   # (B, 128, H/8,  W/8)
        d2 = self.dec2(d3, e1)                   # (B, 64,  H/4,  W/4)
        d1 = self.dec1(d2, stem)                 # (B, 64,  H/2,  W/2)
        d0 = self.dec0(d1)                       # (B, 32,  H,    W)

        return torch.sigmoid(self.final(d0))
