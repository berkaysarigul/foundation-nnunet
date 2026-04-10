"""
losses.py — Per-image Dice + Focal loss for binary segmentation.

DiceFocalLoss combines:
  - PerImageDiceLoss: Dice computed per image then averaged (not batch-flattened)
  - FocalLoss: down-weights easy negatives, focuses on hard positive pixels
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerImageDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        """
        Args:
            smooth: Additive smoothing. Use 1.0 for training stability
                    (prevents division instability on empty masks).
        """
        super().__init__()
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (batch, 1, H, W) — sigmoid output (0-1)
            target: (batch, 1, H, W) — binary mask (0 or 1)
        """
        B = pred.shape[0]
        pred_flat   = pred.view(B, -1)
        target_flat = target.view(B, -1)
        intersection = (pred_flat * target_flat).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (
            pred_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth
        )
        return (1.0 - dice).mean()


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.8, gamma: float = 2.0):
        """
        Args:
            alpha: Balancing factor for positive class (0-1).
                   0.8 weights positives 4× more than negatives.
            gamma: Focusing parameter. 2.0 is standard.
                   Higher gamma → more focus on hard examples.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (batch, 1, H, W) — sigmoid output (0-1)
            target: (batch, 1, H, W) — binary mask (0 or 1)
        """
        bce = F.binary_cross_entropy(pred, target, reduction="none")
        # p_t: probability of the true class
        p_t = pred * target + (1.0 - pred) * (1.0 - target)
        # alpha_t: class-balancing weight
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        focal_weight = alpha_t * (1.0 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class DiceFocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.8, gamma: float = 2.0, smooth: float = 1.0):
        """Combined per-image Dice loss + Focal loss.

        Args:
            alpha:  Focal loss class-balancing factor (positive class weight).
            gamma:  Focal loss focusing parameter.
            smooth: Dice loss smoothing factor.
        """
        super().__init__()
        self.dice  = PerImageDiceLoss(smooth)
        self.focal = FocalLoss(alpha, gamma)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (batch, 1, H, W) — sigmoid output (0-1)
            target: (batch, 1, H, W) — binary mask (0 or 1)
        """
        return self.dice(pred, target) + self.focal(pred, target)
