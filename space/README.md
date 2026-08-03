---
title: Skin Lesion Segmentation
emoji: 🔬
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
short_description: ResNet-34 U-Net segmenting skin lesions from dermoscopy
tags:
  - medical-imaging
  - segmentation
  - pytorch
  - unet
  - isic-2018
---

# Skin Lesion Segmentation — ISIC 2018

A ResNet-34 U-Net that delineates lesion boundaries in dermoscopy images.
Upload an image, get a binary mask and an overlay.

**Held-out performance** (390 images, scored per image, threshold 0.5, with
TTA and mask cleanup): Dice **0.736** · IoU **0.620** · pixel accuracy **0.906**.

Trained for 25 epochs on 2,204 ISIC 2018 Task 1 images at 256×256, on a single
Tesla T4 (~92 minutes). BCE + soft Dice loss, AdamW with differential learning
rates (encoder 1e-4, decoder 1e-3), cosine annealing.

## Controls

- **Threshold** — sigmoid probability cutoff. Lower it for faint lesions.
- **Test-time augmentation** — averages four flipped predictions. Smooths
  boundary noise at ~4× compute.
- **Mask cleanup** — keeps the largest connected region and fills holes.
  ISIC ground truth is always one contiguous lesion, so this reliably removes
  ruler marks, ink dots and vignette corners.

## Limitations

Dice of 0.74 is mid-range for this benchmark, not state of the art. The model
is weakest on low-contrast and amelanotic lesions, images containing rulers or
coloured stickers, and heavily vignetted dermoscopy captures. Reliability drops
near image borders.

## Disclaimer

Research and educational use only. This is **not a medical device**, has not
been clinically validated, and must not be used for diagnosis or treatment
decisions. Always consult a qualified dermatologist.

---

Source: [github.com/NajmiHassan/SkinSeg-Vision](https://github.com/NajmiHassan/SkinSeg-Vision)
