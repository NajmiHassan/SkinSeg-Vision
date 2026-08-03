"""
ISIC 2018 Task 1 dataset loader with augmentation pipelines.
"""

from __future__ import annotations

import os

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

_IMG_EXT = ('.jpg', '.jpeg', '.png')

# ImageNet statistics — the encoder was pre-trained under them.
_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def list_images(image_dir: str) -> list[str]:
    """
    Image filenames, sorted, excluding non-image files.

    The ISIC download ships an ATTRIBUTION.txt and a LICENSE.txt inside the
    image directory. An unfiltered os.listdir() picks them up and the loader
    then dies mid-epoch on a PIL decode error, so filter at the source.
    """
    return sorted(
        f for f in os.listdir(image_dir)
        if f.lower().endswith(_IMG_EXT)
    )


# ──────────────────────────────────────────────────────────────
#  Dataset
# ──────────────────────────────────────────────────────────────

class ISICDataset(Dataset):
    """
    ISIC 2018 Task 1 binary segmentation dataset.

    Expects:
      image_dir/ -> *.jpg dermoscopy images
      mask_dir/  -> <stem>_segmentation.png binary masks

    Masks are binarised to 0.0 (background) or 1.0 (lesion).
    """

    def __init__(self, image_dir: str, mask_dir: str, transform=None,
                 images: list[str] | None = None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        # Accepting the file list at construction time avoids the
        # build-then-overwrite-.images pattern, which silently breaks if a
        # subclass ever caches anything derived from the list.
        self.images = images if images is not None else list_images(image_dir)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        img_name = self.images[idx]
        stem = os.path.splitext(img_name)[0]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, stem + '_segmentation.png')

        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        mask = (mask > 127).astype(np.float32)

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask'].unsqueeze(0)          # (1, H, W)

        return image, mask


# ──────────────────────────────────────────────────────────────
#  Augmentation
# ──────────────────────────────────────────────────────────────

def get_train_transforms(img_size: int = 256) -> A.Compose:
    return A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        # Affine replaces ShiftScaleRotate, which Albumentations now warns on.
        A.Affine(
            translate_percent=(-0.05, 0.05),
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            p=0.4,
        ),
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
#  DataLoader factory
# ──────────────────────────────────────────────────────────────

def split_filenames(image_dir: str, val_split: float = 0.15, seed: int = 42):
    """Deterministic train/val filename split. Same seed -> same split."""
    return train_test_split(
        list_images(image_dir), test_size=val_split, random_state=seed
    )


def get_loaders(image_dir: str, mask_dir: str, img_size: int = 256,
                batch_size: int = 16, val_split: float = 0.15,
                num_workers: int = 4, seed: int = 42):
    """
    Return (train_loader, val_loader, val_filenames).

    val_filenames is returned so downstream evaluation can rebuild exactly
    the held-out split without re-deriving it and risking a mismatch.
    """
    train_imgs, val_imgs = split_filenames(image_dir, val_split, seed)

    train_ds = ISICDataset(image_dir, mask_dir,
                           get_train_transforms(img_size), images=train_imgs)
    val_ds = ISICDataset(image_dir, mask_dir,
                         get_val_transforms(img_size), images=val_imgs)

    common = dict(num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        common.update(persistent_workers=True, prefetch_factor=4)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, drop_last=True, **common)
    val_loader = DataLoader(val_ds, batch_size=batch_size,
                            shuffle=False, **common)

    print(f"Train samples : {len(train_ds)}")
    print(f"Val samples   : {len(val_ds)}")
    return train_loader, val_loader, val_imgs
