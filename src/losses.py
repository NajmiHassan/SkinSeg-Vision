"""
Loss functions and evaluation metrics for binary segmentation.

A note on how Dice is averaged, because it changes the reported number by
several points:

    Batch-pooled  — flatten every pixel in the batch into one vector and
                    compute a single overlap. Large lesions dominate the
                    numerator and denominator, so errors on small lesions
                    barely register. This reads high.

    Per-image     — compute Dice for each image, then average. Every image
                    counts equally regardless of lesion size. This is what
                    the ISIC leaderboard and the segmentation literature
                    report.

On this project the same checkpoint scores 0.772 batch-pooled and 0.736
per-image. Use `dice_score` (per-image) for anything you publish; the pooled
variant is kept only so old training logs remain reproducible.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ──────────────────────────────────────────────────────────────
#  Losses
# ──────────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """
    Soft Dice loss, averaged per image.

    Per-image averaging matters during training too: it stops a batch that
    happens to contain one huge lesion from drowning out the gradient signal
    of the small ones.
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits).flatten(1)      # (B, H*W)
        targets = targets.flatten(1)
        inter = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * inter + self.smooth) / (denom + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    Combined BCE + soft Dice.

      BCE  -> per-pixel calibration
      Dice -> region-level overlap, robust to foreground/background imbalance
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.w_bce = bce_weight
        self.w_dice = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (self.w_bce * self.bce(logits, targets)
                + self.w_dice * self.dice(logits, targets))


# ──────────────────────────────────────────────────────────────
#  Metrics — per image, then averaged over the batch
# ──────────────────────────────────────────────────────────────

def _binarise(logits: torch.Tensor, threshold: float) -> torch.Tensor:
    return (torch.sigmoid(logits) > threshold).float().flatten(1)


@torch.no_grad()
def dice_score(logits: torch.Tensor, targets: torch.Tensor,
               threshold: float = 0.5, smooth: float = 1.0) -> float:
    """Mean per-image Dice / F1 over the batch. Range [0, 1], higher is better."""
    preds = _binarise(logits, threshold)
    targets = targets.flatten(1)
    inter = (preds * targets).sum(dim=1)
    denom = preds.sum(dim=1) + targets.sum(dim=1)
    return ((2.0 * inter + smooth) / (denom + smooth)).mean().item()


@torch.no_grad()
def iou_score(logits: torch.Tensor, targets: torch.Tensor,
              threshold: float = 0.5, smooth: float = 1.0) -> float:
    """Mean per-image IoU (Jaccard). Stricter than Dice."""
    preds = _binarise(logits, threshold)
    targets = targets.flatten(1)
    inter = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1) - inter
    return ((inter + smooth) / (union + smooth)).mean().item()


@torch.no_grad()
def pixel_accuracy(logits: torch.Tensor, targets: torch.Tensor,
                   threshold: float = 0.5) -> float:
    """
    Fraction of correctly classified pixels.

    Reported for completeness, but weak as a segmentation metric: most
    dermoscopy images are mostly background, so predicting all-background
    already scores around 0.80.
    """
    preds = (torch.sigmoid(logits) > threshold).float()
    return (preds == targets).float().mean().item()


@torch.no_grad()
def dice_score_pooled(logits: torch.Tensor, targets: torch.Tensor,
                      threshold: float = 0.5, smooth: float = 1.0) -> float:
    """
    Batch-pooled Dice — kept only for backwards comparison with earlier runs.
    Reads systematically higher than `dice_score`. Do not report this.
    """
    preds = (torch.sigmoid(logits) > threshold).float().view(-1)
    targets = targets.view(-1)
    inter = (preds * targets).sum()
    return ((2.0 * inter + smooth) / (preds.sum() + targets.sum() + smooth)).item()
