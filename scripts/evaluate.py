"""
Evaluate a checkpoint on the held-out split, scoring one image at a time.

Also ablates the two free inference-time wins, so the README numbers are
reproducible rather than asserted:

    python scripts/evaluate.py \
        --image_dir ... --mask_dir ... \
        --weights checkpoints/best_weights.pth --ablate
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.checkpoint import load_into                                # noqa: E402
from src.dataset import ISICDataset, get_val_transforms, split_filenames  # noqa: E402
from src.inference import dice_np, iou_np, postprocess, predict_probs     # noqa: E402
from src.model import UNet                                          # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--image_dir', required=True)
    p.add_argument('--mask_dir', required=True)
    p.add_argument('--weights', default='checkpoints/best_weights.pth')
    p.add_argument('--split_file', default=None,
                   help="val_split.json from training. Falls back to re-deriving "
                        "the split from --seed and --val_split.")
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--val_split', type=float, default=0.15)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--ablate', action='store_true',
                   help="Score all four TTA x cleanup combinations.")
    return p.parse_args()


@torch.no_grad()
def score(model, dataset, device, threshold, tta, clean, batch_size=16):
    """Mean per-image Dice, IoU and pixel accuracy at 256x256."""
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True)

    dices, ious, accs = [], [], []
    label = f"TTA={'on ' if tta else 'off'} clean={'on ' if clean else 'off'}"

    for images, masks in tqdm(loader, desc=label, leave=False):
        images = images.to(device, non_blocking=True)
        probs = predict_probs(model, images, tta=tta).cpu().numpy()
        gts = masks.numpy()

        for prob, gt in zip(probs, gts):
            pred = (prob[0] > threshold).astype(np.uint8)
            if clean:
                pred = postprocess(pred)
            truth = gt[0].astype(np.uint8)

            dices.append(dice_np(pred, truth))
            ious.append(iou_np(pred, truth))
            accs.append(float((pred == truth).mean()))

    return {
        'dice': float(np.mean(dices)),
        'iou': float(np.mean(ious)),
        'pixel_acc': float(np.mean(accs)),
        'dice_std': float(np.std(dices)),
        'dice_p10': float(np.percentile(dices, 10)),
        'n': len(dices),
    }


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.split_file and os.path.exists(args.split_file):
        with open(args.split_file) as fh:
            val_imgs = json.load(fh)['val_images']
        print(f"Loaded {len(val_imgs)} validation filenames from {args.split_file}")
    else:
        _, val_imgs = split_filenames(args.image_dir, args.val_split, args.seed)
        print(f"Re-derived {len(val_imgs)} validation filenames "
              f"(seed={args.seed}, val_split={args.val_split})")

    dataset = ISICDataset(args.image_dir, args.mask_dir,
                          get_val_transforms(args.img_size), images=val_imgs)

    model = UNet(pretrained=False).to(device)
    meta = load_into(model, args.weights, map_location=device)
    model.eval()
    print(f"Loaded {args.weights}"
          + (f" (epoch {meta['epoch']})" if meta.get('epoch') else ""))

    combos = ([(False, False), (True, False), (False, True), (True, True)]
              if args.ablate else [(True, True)])

    print(f"\n{'TTA':<6}{'cleanup':<10}{'Dice':>8}{'IoU':>8}{'PixAcc':>9}{'p10 Dice':>10}")
    print("-" * 51)
    results = {}
    for tta, clean in combos:
        m = score(model, dataset, device, args.threshold, tta, clean, args.batch_size)
        results[f"tta={tta},clean={clean}"] = m
        print(f"{str(tta):<6}{str(clean):<10}"
              f"{m['dice']:>8.4f}{m['iou']:>8.4f}"
              f"{m['pixel_acc']:>9.4f}{m['dice_p10']:>10.4f}")

    print(f"\nScored {results[list(results)[0]]['n']} images individually.")
    print("p10 Dice is the 10th percentile — the tail matters more than the mean "
          "when a single bad mask is the failure mode a clinician would notice.")

    with open('eval_results.json', 'w') as fh:
        json.dump(results, fh, indent=2)
    print("Wrote eval_results.json")


if __name__ == '__main__':
    main()
