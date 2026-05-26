"""
foundation_x_full.py — Full Foundation X model (Swin-B + PPN + FPN + segmentation heads).

This module is a best-effort reverse-engineered reconstruction from the
checkpoint key inventory of `checkpoints/foundation_x.pth`. It is NOT
copy-pasted from the official JLiangLab/Foundation_X repository code. The
module-level attribute names match the checkpoint's `backbone.0.*` key
structure so that the released weights can be loaded with `strict=False`
and a deterministic key-remapping function.

Documented assumptions (informed by the checkpoint key shapes):

  - The deepest Swin-B stage (1024 ch, H/32) is replaced by a PSPNet-style
    Pyramid Pooling Module (`segmentation_PPN`) which produces a
    context-enhanced 1024-ch feature at H/32. The PPN has 4 pooling stages
    that each produce 256 channels; the bottleneck conv consumes
    `1024 + 4*256 = 2048` ch and outputs 1024 ch. (Matches
    `segmentation_PPN.bottleneck.0.weight` shape `(1024, 2048, 3, 3)`.)
  - The FPN has 3 lateral `conv1x1` adapters of output dim 128:
        conv1x1[0]: 256  -> 128   (Swin stage 1, H/8)
        conv1x1[1]: 512  -> 128   (Swin stage 2, H/16)
        conv1x1[2]: 1024 -> 128   (PPN output,   H/32)
    Swin stage 0 (128 ch, H/4) is already at 128 channels and is used as the
    base level without a `conv1x1` projection. Three `smooth_conv` layers
    smooth the three upsampled features; `conv_fusion` concatenates the four
    H/4 features along channel (4 * 128 = 512) and projects back to 128 ch.
  - Each `segmentation_heads[i]` is a `Conv2d(128, C_i, 3, 3)`. From the
    inventory, heads 0/1/2/3/5 are binary (C=1) and head 4 is 13-class
    (likely ChestX-Det). Heads operate on the FPN output at H/4 and the
    script that consumes this module is responsible for upsampling logits
    to the full input resolution.
  - `Segnorm0..3` are `LayerNorm` on the channel dim applied to the four
    Swin stage outputs before they enter PPN/FPN. They are applied in BHWC
    form (the native output format of timm Swin features) and the tensors
    are permuted to BCHW afterwards.
  - `Locnorm1/2/3` (3 LayerNorms for stages 1/2/3) belong to the
    localization branch. They are instantiated for state_dict completeness
    but are not used by the segmentation forward path.

If the JLiangLab/Foundation_X reference code becomes available later, this
file should be replaced (or extended) so the forward graph matches the
official implementation byte-for-byte. Until then, `forward_segmentation`
is a best-effort head-sweep inference path and not a verbatim reproduction
of the paper's segmentation graph.
"""

from __future__ import annotations

import re
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.backbone import (
    FOUNDATION_X_RGB_MEAN,
    FOUNDATION_X_RGB_STD,
    normalize_foundation_x_input,
    repeat_grayscale_to_rgb,
)


# Default head channel inventory derived from foundation_x.pth checkpoint:
#   segmentation_heads.0/1/2/3/5  -> (1, 128, 3, 3)
#   segmentation_heads.4          -> (13, 128, 3, 3)   # multiclass, likely ChestX-Det
DEFAULT_HEAD_OUT_CHANNELS = {0: 1, 1: 1, 2: 1, 3: 1, 4: 13, 5: 1}

DEFAULT_BINARY_HEAD_INDICES = (0, 1, 2, 3, 5)


def _remap_swin_key(inner_key: str) -> str:
    """Remap Swin keys from `layers.N.*` checkpoint form to timm `layers_N.*` form."""

    def shift_downsample(match: re.Match) -> str:
        return f"layers_{int(match.group(1)) + 1}.downsample."

    key = re.sub(r"layers\.(\d+)\.downsample\.", shift_downsample, inner_key)
    key = re.sub(r"layers\.(\d+)\.", r"layers_\1.", key)
    return key


class SegmentationPPN(nn.Module):
    """
    Reverse-engineered Pyramid Pooling Module on the deepest Swin stage.

    Forward (assumed):
        feats = [x]
        for stage in stages: feats.append(upsample(stage(x), to=x.shape))
        x = bottleneck(cat(feats, dim=1))

    Each `stage` = AdaptiveAvgPool2d -> Conv2d(1024->256, 1x1, bias=False) ->
                   BatchNorm2d(256) -> ReLU. (Weight shape 256x1024x1x1
                   confirms the 1x1 conv from 1024 -> 256.)

    Bottleneck = Conv2d(2048->1024, 3x3, padding=1, bias=False) ->
                 BatchNorm2d(1024) -> ReLU. (Shape 1024x2048x3x3 confirms
                 input 1024 + 4*256 = 2048.)
    """

    def __init__(
        self,
        in_channels: int = 1024,
        reduction_channels: int = 256,
        pool_sizes: Iterable[int] = (1, 2, 3, 6),
        out_channels: int = 1024,
    ):
        super().__init__()
        self.stages = nn.ModuleList()
        for ps in pool_sizes:
            stage = nn.Sequential(
                nn.AdaptiveAvgPool2d(ps),
                nn.Conv2d(in_channels, reduction_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_channels),
                nn.ReLU(inplace=True),
            )
            self.stages.append(stage)
        fused_in = in_channels + len(tuple(pool_sizes)) * reduction_channels
        self.bottleneck = nn.Sequential(
            nn.Conv2d(fused_in, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [x]
        for stage in self.stages:
            pooled = stage(x)
            up = F.interpolate(pooled, size=x.shape[2:], mode="bilinear", align_corners=False)
            feats.append(up)
        x = torch.cat(feats, dim=1)
        return self.bottleneck(x)


class SegmentationFPN(nn.Module):
    """
    Reverse-engineered FPN that fuses 4 multi-scale features to a single
    H/4 feature map at 128 channels.

    Forward (assumed UPerNet-style top-down):
        f0(128,H/4), f1(256,H/8), f2(512,H/16), f3_ppn(1024,H/32)
        p3 = conv1x1[2](f3_ppn)                                   # (128, H/32)
        p2 = conv1x1[1](f2) + upsample(p3, f2.shape)               # (128, H/16)
        p1 = conv1x1[0](f1) + upsample(p2, f1.shape)               # (128, H/8)
        p3_s = smooth_conv[2](p3)
        p2_s = smooth_conv[1](p2)
        p1_s = smooth_conv[0](p1)
        # f0 stays at 128 ch
        cat at H/4 -> conv_fusion -> (128, H/4)
    """

    def __init__(self):
        super().__init__()
        self.conv1x1 = nn.ModuleList([
            nn.Conv2d(256, 128, kernel_size=1),
            nn.Conv2d(512, 128, kernel_size=1),
            nn.Conv2d(1024, 128, kernel_size=1),
        ])
        self.smooth_conv = nn.ModuleList([
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
        ])
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

    def forward(self, feats: list[torch.Tensor]) -> torch.Tensor:
        if len(feats) != 4:
            raise AssertionError(f"Expected 4 feature maps, got {len(feats)}")
        f0, f1, f2, f3_ppn = feats

        # Lateral 1x1
        p3 = self.conv1x1[2](f3_ppn)
        p2 = self.conv1x1[1](f2) + F.interpolate(
            p3, size=f2.shape[2:], mode="bilinear", align_corners=False
        )
        p1 = self.conv1x1[0](f1) + F.interpolate(
            p2, size=f1.shape[2:], mode="bilinear", align_corners=False
        )

        # Smooth
        p3_s = self.smooth_conv[2](p3)
        p2_s = self.smooth_conv[1](p2)
        p1_s = self.smooth_conv[0](p1)

        # Upsample all to H/4 (f0 scale) and concat
        target = f0.shape[2:]
        p3_up = F.interpolate(p3_s, size=target, mode="bilinear", align_corners=False)
        p2_up = F.interpolate(p2_s, size=target, mode="bilinear", align_corners=False)
        p1_up = F.interpolate(p1_s, size=target, mode="bilinear", align_corners=False)

        x = torch.cat([f0, p1_up, p2_up, p3_up], dim=1)
        return self.conv_fusion(x)


class FoundationXFullNet(nn.Module):
    """
    Reverse-engineered full Foundation X model:
      Swin-B encoder + Segnorm{0..3} + segmentation_PPN + segmentation_FPN
      + segmentation_heads (per-task binary or multiclass).

    The Locnorm{1..3} layers are instantiated for state_dict completeness
    but are not used by the segmentation forward path. Classification
    heads, the Swin ImageNet head, and DINO-style keys (if present) are
    intentionally not instantiated and will appear as `unexpected` keys
    when loading the released checkpoint with `strict=False`.

    Args:
        head_out_channels: optional dict mapping head index -> num output
            channels. Defaults to the inventory of the released checkpoint:
            heads 0/1/2/3/5 are binary, head 4 is 13-class.
        img_size: input spatial size in pixels (H == W). Defaults to 512.

    Forward contract:
        Input  : (B, 1, H, W) grayscale, float32, values in [0, 1].
        Output : raw logits via `forward_segmentation(x, head_idx)`,
                 shape (B, C_head, H, W). The caller is responsible for
                 applying sigmoid (binary heads) or softmax (multiclass).
    """

    def __init__(
        self,
        head_out_channels: dict[int, int] | None = None,
        img_size: int = 512,
    ):
        super().__init__()
        import timm  # imported here to keep optional for static analysis

        self.img_size = int(img_size)
        self.head_out_channels = dict(head_out_channels or DEFAULT_HEAD_OUT_CHANNELS)

        # Swin-B encoder (features_only -> 4 BHWC outputs at H/4, H/8, H/16, H/32)
        self.swin = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=self.img_size,
        )

        # Per-stage Segnorms / Locnorms (LayerNorms on channel dim, applied to BHWC)
        self.Segnorm0 = nn.LayerNorm(128)
        self.Segnorm1 = nn.LayerNorm(256)
        self.Segnorm2 = nn.LayerNorm(512)
        self.Segnorm3 = nn.LayerNorm(1024)
        # Locnorm{1..3}: not used by segmentation, but kept for state_dict completeness
        self.Locnorm1 = nn.LayerNorm(256)
        self.Locnorm2 = nn.LayerNorm(512)
        self.Locnorm3 = nn.LayerNorm(1024)

        # PPN on deepest stage, FPN over 4 stages, per-task heads
        self.segmentation_PPN = SegmentationPPN()
        self.segmentation_FPN = SegmentationFPN()

        max_idx = max(self.head_out_channels)
        heads: list[nn.Module] = []
        for i in range(max_idx + 1):
            out_ch = int(self.head_out_channels.get(i, 1))
            heads.append(nn.Conv2d(128, out_ch, kernel_size=3, padding=1))
        self.segmentation_heads = nn.ModuleList(heads)

    # ─── Loading ────────────────────────────────────────────────────────────

    def remap_foundation_x_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
        """
        Build a remapped state_dict whose keys match this module's attribute
        structure. Keys not under `backbone.0.*` are dropped.

        Returns the remapped dict and a small stats dict useful for reports.
        """
        prefix = "backbone.0."
        remapped: dict[str, torch.Tensor] = {}
        stats = {
            "raw_total": 0,
            "raw_with_prefix": 0,
            "routed_swin": 0,
            "routed_segnorm": 0,
            "routed_locnorm": 0,
            "routed_ppn": 0,
            "routed_fpn": 0,
            "routed_heads": 0,
            "skipped_non_backbone": 0,
            "skipped_classification_heads": 0,
            "skipped_swin_head": 0,
            "skipped_other": 0,
        }
        for raw_key, value in state_dict.items():
            stats["raw_total"] += 1
            if not raw_key.startswith(prefix):
                stats["skipped_non_backbone"] += 1
                continue
            stats["raw_with_prefix"] += 1
            inner = raw_key[len(prefix):]

            # Swin sub-keys: patch_embed, layers.N, norm  (head is ImageNet, skipped below)
            if (
                inner.startswith("patch_embed")
                or inner.startswith("layers.")
                or inner.startswith("layers_")
                or inner.startswith("norm.")
            ):
                new_key = f"swin.{_remap_swin_key(inner)}"
                remapped[new_key] = value
                stats["routed_swin"] += 1
                continue

            if inner.startswith("head."):
                # Swin ImageNet classification head — not used in segmentation
                stats["skipped_swin_head"] += 1
                continue

            if inner.startswith("classification_heads"):
                # Task-specific classification heads — not used in segmentation
                stats["skipped_classification_heads"] += 1
                continue

            if inner.startswith("Segnorm") or inner.startswith("Locnorm"):
                # Direct module names match
                remapped[inner] = value
                if inner.startswith("Segnorm"):
                    stats["routed_segnorm"] += 1
                else:
                    stats["routed_locnorm"] += 1
                continue

            if inner.startswith("segmentation_PPN"):
                remapped[inner] = value
                stats["routed_ppn"] += 1
                continue

            if inner.startswith("segmentation_FPN"):
                remapped[inner] = value
                stats["routed_fpn"] += 1
                continue

            if inner.startswith("segmentation_heads"):
                remapped[inner] = value
                stats["routed_heads"] += 1
                continue

            stats["skipped_other"] += 1

        return remapped, stats

    def load_foundation_x_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, list[str] | dict[str, int]]:
        """
        Remap the Foundation X checkpoint state_dict and load with strict=False.

        Returns a diagnostic dict with `missing`, `unexpected`, and `route_stats`.
        """
        remapped, route_stats = self.remap_foundation_x_state_dict(state_dict)
        missing, unexpected = self.load_state_dict(remapped, strict=False)
        return {
            "missing": [str(k) for k in missing],
            "unexpected": [str(k) for k in unexpected],
            "route_stats": route_stats,
            "remapped_total": len(remapped),
        }

    # ─── Forward ────────────────────────────────────────────────────────────

    def encode(self, x_grayscale_01: torch.Tensor) -> list[torch.Tensor]:
        """
        Run Swin-B on the Foundation X branch normalization view and return
        the 4 per-stage features in BCHW form after Segnorm.

        Input: (B, 1, H, W) grayscale, float32, in [0, 1].
        Returns 4 tensors with channels (128, 256, 512, 1024).
        """
        if x_grayscale_01.ndim != 4 or x_grayscale_01.shape[1] != 1:
            raise AssertionError(
                f"FoundationXFullNet expects (B, 1, H, W) grayscale input; "
                f"got {tuple(x_grayscale_01.shape)}"
            )
        rgb_norm = normalize_foundation_x_input(x_grayscale_01)
        feats_bhwc = self.swin(rgb_norm)  # list of 4 BHWC tensors
        if len(feats_bhwc) != 4:
            raise AssertionError(
                f"Swin features_only must emit 4 maps; got {len(feats_bhwc)}"
            )
        # Apply Segnorms on channel dim (last dim of BHWC)
        f0 = self.Segnorm0(feats_bhwc[0])
        f1 = self.Segnorm1(feats_bhwc[1])
        f2 = self.Segnorm2(feats_bhwc[2])
        f3 = self.Segnorm3(feats_bhwc[3])
        # Permute to BCHW
        feats = [
            f0.permute(0, 3, 1, 2).contiguous(),
            f1.permute(0, 3, 1, 2).contiguous(),
            f2.permute(0, 3, 1, 2).contiguous(),
            f3.permute(0, 3, 1, 2).contiguous(),
        ]
        return feats

    def segmentation_features(self, x_grayscale_01: torch.Tensor) -> torch.Tensor:
        """
        Returns the FPN output (B, 128, H/4, W/4) ready to be consumed by any
        segmentation head.
        """
        feats = self.encode(x_grayscale_01)
        ppn_out = self.segmentation_PPN(feats[3])
        fpn_inputs = [feats[0], feats[1], feats[2], ppn_out]
        fpn_out = self.segmentation_FPN(fpn_inputs)
        return fpn_out

    def forward_segmentation(
        self,
        x_grayscale_01: torch.Tensor,
        head_idx: int,
        upsample_to_input: bool = True,
    ) -> torch.Tensor:
        """
        Run direct segmentation through head `head_idx` and return raw logits.

        - For binary heads (out_channels=1) the caller should apply
          `torch.sigmoid` to obtain probabilities.
        - For multiclass heads (e.g. head 4 with 13 channels) the caller
          should apply softmax/argmax. This script's head sweep skips
          multiclass heads by default.

        Returns:
            (B, C_head, H, W)  if upsample_to_input=True (default),
            (B, C_head, H/4, W/4) otherwise.
        """
        if head_idx < 0 or head_idx >= len(self.segmentation_heads):
            raise IndexError(
                f"Head index {head_idx} out of range [0, {len(self.segmentation_heads)})"
            )
        H, W = x_grayscale_01.shape[-2], x_grayscale_01.shape[-1]
        fpn_out = self.segmentation_features(x_grayscale_01)
        logits = self.segmentation_heads[head_idx](fpn_out)
        if upsample_to_input:
            logits = F.interpolate(
                logits, size=(H, W), mode="bilinear", align_corners=False
            )
        return logits

    def forward(self, x: torch.Tensor, head_idx: int = 0) -> torch.Tensor:
        """Default forward: route through `forward_segmentation` with `head_idx`."""
        return self.forward_segmentation(x, head_idx=head_idx, upsample_to_input=True)


__all__ = [
    "DEFAULT_BINARY_HEAD_INDICES",
    "DEFAULT_HEAD_OUT_CHANNELS",
    "FOUNDATION_X_RGB_MEAN",
    "FOUNDATION_X_RGB_STD",
    "FoundationXFullNet",
    "SegmentationFPN",
    "SegmentationPPN",
    "normalize_foundation_x_input",
    "repeat_grayscale_to_rgb",
]
