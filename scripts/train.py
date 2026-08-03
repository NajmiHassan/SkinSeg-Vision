"""
Training script for ISIC 2018 skin lesion segmentation.

Mirrors the recipe used in notebooks/skinseg-vision.ipynb:
differential learning rates, cosine annealing, AMP, early stopping.

Usage:
    python scripts/train.py \
        --image_dir /path/to/ISIC2018_Task1-2_Training_Input \
        --mask_dir  /path/to/ISIC2018_Task1_Training_GroundTruth \
        --epochs 25 --batch_size 16
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.checkpoint import export_weights                      # noqa: E402
from src.dataset import get_loaders                            # noqa: E402
from src.losses import BCEDiceLoss, dice_score, iou_score      # noqa: E402
from src.model import UNet                                     # noqa: E402

# Parameter-name prefixes belonging to the pre-trained ResNet-34 encoder.
ENCODER_PREFIXES = ('enc0', 'enc1', 'enc2', 'enc3', 'enc4', 'pool')


def parse_args():
    p = argparse.ArgumentParser(description="Train ResNet34-UNet on ISIC 2018")
    p.add_argument('--image_dir', required=True)
    p.add_argument('--mask_dir', required=True)
    p.add_argument('--checkpoint_dir', default='checkpoints')
    p.add_argument('--img_size', type=int, default=256)
    p.add_argument('--epochs', type=int, default=25)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--encoder_lr', type=float, default=1e-4,
                   help="Low: the encoder is pre-trained and only needs nudging.")
    p.add_argument('--decoder_lr', type=float, default=1e-3,
                   help="High: the decoder starts from random init.")
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--val_split', type=float, default=0.15)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--patience', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def build_optimizer(model, args):
    """
    Two parameter groups with different learning rates.

    A single LR is the main reason a pre-trained-encoder U-Net stalls: an LR
    high enough to train a random decoder will wash out ImageNet features in
    the encoder, and an LR low enough to preserve them leaves the decoder
    barely moving. Ten-to-one is the usual split.
    """
    encoder, decoder = [], []
    for name, param in model.named_parameters():
        (encoder if name.startswith(ENCODER_PREFIXES) else decoder).append(param)

    print(f"Encoder tensors: {len(encoder)} | Decoder tensors: {len(decoder)}")
    return optim.AdamW(
        [{'params': encoder, 'lr': args.encoder_lr},
         {'params': decoder, 'lr': args.decoder_lr}],
        weight_decay=args.weight_decay,
    )


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss, total_dice = 0.0, 0.0
    for images, masks in tqdm(loader, desc='  Train', leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with autocast('cuda', enabled=device.type == 'cuda'):
            logits = model(images)
            loss = criterion(logits, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        total_dice += dice_score(logits.detach().float(), masks)

    n = len(loader)
    return total_loss / n, total_dice / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0
    for images, masks in tqdm(loader, desc='  Val  ', leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        with autocast('cuda', enabled=device.type == 'cuda'):
            logits = model(images)
            loss = criterion(logits, masks)

        logits = logits.float()
        total_loss += loss.item()
        total_dice += dice_score(logits, masks)
        total_iou += iou_score(logits, masks)

    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_loader, val_loader, val_imgs = get_loaders(
        args.image_dir, args.mask_dir,
        img_size=args.img_size, batch_size=args.batch_size,
        val_split=args.val_split, num_workers=args.num_workers, seed=args.seed,
    )

    # Persist the split so evaluation can never drift out of sync with training.
    split_path = os.path.join(args.checkpoint_dir, 'val_split.json')
    with open(split_path, 'w') as fh:
        json.dump({'seed': args.seed, 'val_split': args.val_split,
                   'val_images': val_imgs}, fh, indent=2)

    model = UNet(pretrained=True).to(device)
    criterion = BCEDiceLoss()
    optimizer = build_optimizer(model, args)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = GradScaler('cuda', enabled=device.type == 'cuda')

    best_dice, bad_epochs = 0.0, 0
    history = {k: [] for k in
               ('train_loss', 'val_loss', 'train_dice', 'val_dice', 'val_iou')}
    ckpt_path = os.path.join(args.checkpoint_dir, 'best_model.pth')

    print(f"\nTraining for {args.epochs} epochs\n")
    for epoch in range(1, args.epochs + 1):
        print(f"Epoch [{epoch:02d}/{args.epochs}]")

        train_loss, train_dice = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device)
        val_loss, val_dice, val_iou = validate(
            model, val_loader, criterion, device)
        scheduler.step()

        for key, value in zip(history,
                              (train_loss, val_loss, train_dice, val_dice, val_iou)):
            history[key].append(value)

        print(f"  Train -> Loss: {train_loss:.4f} | Dice: {train_dice:.4f}")
        print(f"  Val   -> Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")

        if val_dice > best_dice:
            best_dice, bad_epochs = val_dice, 0
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_dice': best_dice,
                'config': vars(args),
            }, ckpt_path)
            print(f"  saved checkpoint (Dice={best_dice:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement in {args.patience} epochs)")
                break
        print()

    with open(os.path.join(args.checkpoint_dir, 'history.json'), 'w') as fh:
        json.dump(history, fh, indent=2)

    # Deployment artifact: model tensors only, roughly a third of the size.
    export_weights(ckpt_path, os.path.join(args.checkpoint_dir, 'best_weights.pth'))

    print(f"\nDone. Best val Dice (batch mean of per-image): {best_dice:.4f}")
    print("Run scripts/evaluate.py for the final per-image figure.")


if __name__ == '__main__':
    main()
