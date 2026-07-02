"""
Training script for ISIC 2018 skin lesion segmentation.

Usage:
    python scripts/train.py \
        --image_dir /path/to/ISIC2018_Task1-2_Training_Input \
        --mask_dir  /path/to/ISIC2018_Task1_Training_GroundTruth \
        --epochs 40 \
        --batch_size 16
"""

import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# Allow running from repo root
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.model   import UNet
from src.dataset import get_loaders
from src.losses  import BCEDiceLoss, dice_score, iou_score


# ──────────────────────────────────────────────────────────────
#  CLI Args
# ──────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train UNet on ISIC 2018")
    p.add_argument('--image_dir',      required=True)
    p.add_argument('--mask_dir',       required=True)
    p.add_argument('--checkpoint_dir', default='checkpoints')
    p.add_argument('--img_size',       type=int,   default=256)
    p.add_argument('--epochs',         type=int,   default=40)
    p.add_argument('--batch_size',     type=int,   default=16)
    p.add_argument('--lr',             type=float, default=3e-4)
    p.add_argument('--val_split',      type=float, default=0.15)
    p.add_argument('--num_workers',    type=int,   default=2)
    p.add_argument('--seed',           type=int,   default=42)
    return p.parse_args()


# ──────────────────────────────────────────────────────────────
#  Train / Validate
# ──────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss, total_dice = 0.0, 0.0
    for images, masks in tqdm(loader, desc='  Train', leave=False):
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        with autocast():
            logits = model(images)
            loss   = criterion(logits, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        total_dice += dice_score(logits.detach(), masks)
    n = len(loader)
    return total_loss / n, total_dice / n


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, total_dice, total_iou = 0.0, 0.0, 0.0
    for images, masks in tqdm(loader, desc='  Val  ', leave=False):
        images, masks = images.to(device), masks.to(device)
        with autocast():
            logits = model(images)
            loss   = criterion(logits, masks)
        total_loss += loss.item()
        total_dice += dice_score(logits, masks)
        total_iou  += iou_score(logits, masks)
    n = len(loader)
    return total_loss / n, total_dice / n, total_iou / n


# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device : {device}")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    train_loader, val_loader = get_loaders(
        args.image_dir, args.mask_dir,
        img_size=args.img_size, batch_size=args.batch_size,
        val_split=args.val_split, num_workers=args.num_workers, seed=args.seed,
    )

    model     = UNet(pretrained=True).to(device)
    criterion = BCEDiceLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler    = GradScaler()

    best_dice = 0.0
    print(f"\n🚀 Training for {args.epochs} epochs\n")

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch [{epoch:02d}/{args.epochs}]")

        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
        val_loss,   val_dice,   val_iou = validate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"  Train → Loss: {train_loss:.4f} | Dice: {train_dice:.4f}")
        print(f"  Val   → Loss: {val_loss:.4f}   | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")

        if val_dice > best_dice:
            best_dice = val_dice
            ckpt_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            torch.save({
                'epoch'      : epoch,
                'model_state': model.state_dict(),
                'optimizer'  : optimizer.state_dict(),
                'best_dice'  : best_dice,
                'config'     : vars(args),
            }, ckpt_path)
            print(f"  💾 Saved checkpoint (Dice={best_dice:.4f})")
        print()

    print(f"🎉 Done! Best Val Dice: {best_dice:.4f}")


if __name__ == '__main__':
    main()
