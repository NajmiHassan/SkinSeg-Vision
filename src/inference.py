"""
Inference utilities shared by the Gradio app and the evaluation script.

Three things happen here that do not happen during training, and each one
buys accuracy at zero training cost:

1. Test-time augmentation (TTA) — average the sigmoid maps over the identity,
   horizontal flip, vertical flip and both. The model was trained with flip
   augmentation, so it is already flip-equivariant; averaging cancels a
   good chunk of the boundary jitter.

2. Full-resolution thresholding — upsample the *probability* map bilinearly
   to the original image size and threshold there, rather than thresholding
   at 256x256 and upsampling a binary mask with NEAREST. Nearest-neighbour
   upsampling of a binary mask quantises the boundary to 256-grid steps,
   which on a 1022x767 dermoscopy image is a ~4px staircase.

3. Largest-connected-component + hole filling — ISIC Task 1 ground truth is
   a single contiguous lesion, always. Any extra blob the model produces is
   a guaranteed false positive: ruler marks, vignette corners, ink dots.
   Dropping everything but the biggest component removes them outright.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

try:
    from scipy import ndimage as _ndi
    _HAS_SCIPY = True
except ImportError:                                  # pragma: no cover
    _ndi = None
    _HAS_SCIPY = False


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ──────────────────────────────────────────────────────────────
#  Forward passes
# ──────────────────────────────────────────────────────────────

@torch.no_grad()
def predict_probs(model, x: torch.Tensor, tta: bool = True) -> torch.Tensor:
    """
    Run the model and return sigmoid probabilities, shape (B, 1, H, W).

    Args:
        model : UNet in eval() mode.
        x     : normalised input batch, (B, 3, H, W).
        tta   : average over the 4-way flip group when True.
    """
    def _fwd(t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(model(t))

    probs = _fwd(x)
    if not tta:
        return probs

    # dims=[3] is horizontal (width), dims=[2] is vertical (height)
    probs = probs + _fwd(x.flip(dims=[3])).flip(dims=[3])
    probs = probs + _fwd(x.flip(dims=[2])).flip(dims=[2])
    probs = probs + _fwd(x.flip(dims=[2, 3])).flip(dims=[2, 3])
    return probs / 4.0


def upsample_probs(probs: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Bilinearly resize a probability map to (height, width)."""
    return F.interpolate(probs, size=size, mode='bilinear', align_corners=False)


# ──────────────────────────────────────────────────────────────
#  Mask cleanup
# ──────────────────────────────────────────────────────────────

def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Retain only the largest 8-connected foreground blob.

    Falls back to a no-op when SciPy is unavailable so the app still runs.
    """
    if not _HAS_SCIPY or mask.sum() == 0:
        return mask

    structure = np.ones((3, 3), dtype=np.uint8)          # 8-connectivity
    labelled, n = _ndi.label(mask, structure=structure)
    if n <= 1:
        return mask

    # Index 0 is the background label; ignore it when picking the winner.
    sizes = np.bincount(labelled.ravel())
    sizes[0] = 0
    return (labelled == sizes.argmax()).astype(mask.dtype)


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Close interior holes — a lesion is solid, not a ring."""
    if not _HAS_SCIPY or mask.sum() == 0:
        return mask
    return _ndi.binary_fill_holes(mask).astype(mask.dtype)


def postprocess(mask: np.ndarray, min_area_frac: float = 0.0005) -> np.ndarray:
    """
    Clean a binary mask: drop spurious blobs, then fill interior holes.

    Args:
        mask          : uint8/bool array of 0s and 1s.
        min_area_frac : if the surviving component covers less than this
                        fraction of the image, treat the prediction as empty.
                        Guards against a stray 20-pixel speck being promoted
                        to "the lesion" when the model finds nothing.
    """
    mask = keep_largest_component(mask)
    mask = fill_holes(mask)
    if mask.mean() < min_area_frac:
        return np.zeros_like(mask)
    return mask


# ──────────────────────────────────────────────────────────────
#  Metrics (per image — the honest ones)
# ──────────────────────────────────────────────────────────────

def dice_np(pred: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> float:
    """Per-image Dice on binary numpy arrays."""
    pred = pred.astype(np.float64).ravel()
    target = target.astype(np.float64).ravel()
    inter = (pred * target).sum()
    return float((2.0 * inter + smooth) / (pred.sum() + target.sum() + smooth))


def iou_np(pred: np.ndarray, target: np.ndarray, smooth: float = 1.0) -> float:
    """Per-image IoU on binary numpy arrays."""
    pred = pred.astype(np.float64).ravel()
    target = target.astype(np.float64).ravel()
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter
    return float((inter + smooth) / (union + smooth))
