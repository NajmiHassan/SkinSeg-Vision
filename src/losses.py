"""
Loss functions and evaluation metrics for binary segmentation.
"""

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────
#  Loss Functions
# ──────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Soft Dice loss — measures region overlap. Robust to class imbalance."""
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs   = torch.sigmoid(logits).view(-1)
        targets = targets.view(-1)
        inter   = (probs * targets).sum()
        return 1.0 - (2.0 * inter + self.smooth) / (probs.sum() + targets.sum() + self.smooth)


class BCEDiceLoss(nn.Module):
    """
    Combined BCE + Dice loss — industry standard for medical segmentation.
      BCE  → per-pixel accuracy
      Dice → region-level overlap
    """
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.w_bce  = bce_weight
        self.w_dice = dice_weight
        self.bce    = nn.BCEWithLogitsLoss()
        self.dice   = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.w_bce * self.bce(logits, targets) + self.w_dice * self.dice(logits, targets)


# ──────────────────────────────────────────────────────────────
#  Metrics
# ──────────────────────────────────────────────────────────────

def dice_score(logits: torch.Tensor, targets: torch.Tensor,
               threshold: float = 0.5, smooth: float = 1.0) -> float:
    """Dice / F1 coefficient. Range [0, 1] — higher is better."""
    probs   = (torch.sigmoid(logits) > threshold).float().view(-1)
    targets = targets.view(-1)
    inter   = (probs * targets).sum()
    return ((2.0 * inter + smooth) / (probs.sum() + targets.sum() + smooth)).item()


def iou_score(logits: torch.Tensor, targets: torch.Tensor,
              threshold: float = 0.5, smooth: float = 1.0) -> float:
    """Intersection over Union (Jaccard index). Range [0, 1] — stricter than Dice."""
    probs   = (torch.sigmoid(logits) > threshold).float().view(-1)
    targets = targets.view(-1)
    inter   = (probs * targets).sum()
    union   = probs.sum() + targets.sum() - inter
    return ((inter + smooth) / (union + smooth)).item()


def pixel_accuracy(logits: torch.Tensor, targets: torch.Tensor,
                   threshold: float = 0.5) -> float:
    """Fraction of correctly classified pixels."""
    preds = (torch.sigmoid(logits) > threshold).float()
    return (preds == targets).float().mean().item()
