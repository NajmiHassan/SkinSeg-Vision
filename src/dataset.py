"""
ISIC 2018 Task 1 dataset loader with augmentation pipelines.
"""

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ──────────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────────

class ISICDataset(Dataset):
    """
    ISIC 2018 Task 1 binary segmentation dataset.

    Expects:
      image_dir/ → *.jpg or *.png dermoscopy images
      mask_dir/  → *_segmentation.png binary masks

    Masks are binarized: 0 (background) or 1 (lesion).
    """
    def __init__(self, image_dir: str, mask_dir: str, transform=None):
        self.image_dir = image_dir
        self.mask_dir  = mask_dir
        self.transform = transform
        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_name  = self.images[idx]
        stem      = os.path.splitext(img_name)[0]
        img_path  = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, stem + '_segmentation.png')

        image = np.array(Image.open(img_path).convert('RGB'))
        mask  = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        mask  = (mask > 127).astype(np.float32)   # binarize → 0.0 or 1.0

        if self.transform:
            aug   = self.transform(image=image, mask=mask)
            image = aug['image']
            mask  = aug['mask'].unsqueeze(0)       # (1, H, W)

        return image, mask


# ──────────────────────────────────────────────────────────────
#  Augmentation Pipelines
# ──────────────────────────────────────────────────────────────

# ImageNet stats (encoder is pre-trained on ImageNet)
_MEAN = (0.485, 0.456, 0.406)
_STD  = (0.229, 0.224, 0.225)


def get_train_transforms(img_size: int = 256) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1,
                           rotate_limit=15, p=0.4),
        A.ColorJitter(brightness=0.2, contrast=0.2,
                      saturation=0.2, hue=0.1, p=0.4),
        A.GaussNoise(p=0.2),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


def get_val_transforms(img_size: int = 256) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=_MEAN, std=_STD),
        ToTensorV2(),
    ])


# ──────────────────────────────────────────────────────────────
#  DataLoader Factory
# ──────────────────────────────────────────────────────────────

def get_loaders(image_dir: str, mask_dir: str, img_size: int = 256,
                batch_size: int = 16, val_split: float = 0.15,
                num_workers: int = 2, seed: int = 42):
    """Return (train_loader, val_loader) with reproducible split."""
    all_images = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    train_imgs, val_imgs = train_test_split(
        all_images, test_size=val_split, random_state=seed
    )

    train_ds         = ISICDataset(image_dir, mask_dir, get_train_transforms(img_size))
    val_ds           = ISICDataset(image_dir, mask_dir, get_val_transforms(img_size))
    train_ds.images  = train_imgs
    val_ds.images    = val_imgs

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Train samples : {len(train_ds)}")
    print(f"Val   samples : {len(val_ds)}")
    return train_loader, val_loader
