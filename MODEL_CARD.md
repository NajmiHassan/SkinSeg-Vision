---
language: en
license: mit
tags:
  - medical-imaging
  - image-segmentation
  - skin-lesion
  - dermoscopy
  - unet
  - resnet34
  - pytorch
  - isic-2018
datasets:
  - isic-2018
metrics:
  - dice
  - iou
pipeline_tag: image-segmentation
---

# Skin Lesion Segmentation — ResNet-34 U-Net

Binary segmentation of skin lesions in dermoscopy images, trained on ISIC 2018 Task 1.

**Live demo →** [Hugging Face Spaces](https://huggingface.co/spaces/NajmiHassan1/SkinSeg-Vision)
**Code →** [GitHub](https://github.com/NajmiHassan/SkinSeg-Vision)

---

## Model Description

A U-Net with a ResNet-34 encoder pre-trained on ImageNet, fine-tuned end-to-end.
Given a dermoscopy image the model produces a pixel-wise binary mask separating
lesion from surrounding skin.

- **Encoder:** ResNet-34, ImageNet pre-trained
- **Decoder:** transposed-convolution upsampling with skip connections
- **Loss:** 0.5 × BCEWithLogits + 0.5 × soft Dice
- **Parameters:** 47.9M
- **Precision:** mixed (`torch.amp`)

---

## Evaluation Results

Held-out split of 390 images, **scored one image at a time** at threshold 0.5.

| Metric | Score |
|---|---|
| Dice / F1 | **0.736** |
| IoU (Jaccard) | **0.620** |
| Pixel accuracy | **0.906** |

This is mid-range for the benchmark. Published ISIC 2018 baselines reach
0.85–0.90 Dice, and this model does not match them.

### A note on how Dice was averaged

The training log for this run reports a best validation Dice of 0.772. That
number pools every pixel in a batch of 16 into one overlap calculation, which
lets large lesions dominate and dilutes errors on small ones. Averaging Dice
per image, then taking the mean — the convention in the segmentation
literature and on the ISIC leaderboard — gives **0.736** for the same
checkpoint. The lower number is the one reported above and the one to compare
against other work.

Pixel accuracy is included for completeness but is weak here: most dermoscopy
images are mostly background, so predicting all-background already scores
around 0.80.

### Qualitative predictions

![Qualitative segmentation results](assets/predictions.png)

*Each row: input image · ground truth · prediction with per-image Dice.*

Per-image Dice across the six sampled cases: **0.930, 0.833, 0.797, 0.674,
0.593, 0.373**. That spread is the honest picture. The best cases are large,
well-demarcated pigmented lesions. The 0.373 case is a faint low-contrast
lesion beside a red calibration sticker, which the model segments instead of
the lesion.

Raw predictions also show speckle noise and spurious blobs on ruler markings
and vignette borders. The demo app suppresses these at inference time by
keeping only the largest connected component — ISIC ground truth is always a
single contiguous lesion, so any additional region is a certain false positive.

---

## Training Data

[ISIC 2018 Challenge — Task 1: Lesion Segmentation](https://challenge.isic-archive.com/landing/2018/)

2,594 dermoscopy images with expert-annotated binary masks, covering diverse
lesion types, sizes and acquisition conditions. Split 2,204 train / 390
validation with seed 42.

![Sample images and masks](assets/sample_data.png)

---

## Training Details

| Parameter | Value |
|---|---|
| Input size | 256 × 256 |
| Epochs | 25 (early stopping patience 8, never triggered) |
| Batch size | 16 |
| Optimizer | AdamW, weight decay 1e-4 |
| Learning rate | encoder 1e-4, decoder 1e-3 |
| LR scheduler | Cosine annealing to 1e-6 |
| Loss | BCE + soft Dice (0.5 / 0.5) |
| Precision | Mixed (`torch.amp`) |
| Train / val split | 85% / 15%, seed 42 |
| Hardware | Single Tesla T4, ~92 minutes |

**Augmentations:** HorizontalFlip, VerticalFlip, RandomRotate90, Affine
(translate ±5%, scale 0.9–1.1, rotate ±15°), ColorJitter, GaussNoise.

### Differential learning rates

The encoder and decoder train at rates an order of magnitude apart. A single
shared rate is the common reason a pre-trained-encoder U-Net stalls: a rate
high enough to train a randomly initialised decoder washes ImageNet features
out of the encoder, and a rate low enough to preserve them leaves the decoder
barely moving.

### Training curves

![Training history](assets/training_history.png)

Train and validation loss track each other to the final epoch with no
divergence, and validation Dice was still climbing at epoch 25. The model is
**underfitting**, not overfitting. Added regularisation would not help; more
capacity, higher input resolution, or a longer schedule would.

---

## How to Use

```bash
pip install torch torchvision albumentations scipy huggingface_hub Pillow numpy
```

```python
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from huggingface_hub import hf_hub_download

from src.model import UNet          # copy model.py from the GitHub repo

ckpt_path = hf_hub_download(
    repo_id="NajmiHassan1/skinseg-vision",
    filename="best_weights.pth",
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet(pretrained=False).to(device)
ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
model.load_state_dict(ckpt["model_state"])
model.eval()

transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

image = np.array(Image.open("your_image.jpg").convert("RGB"))
h, w = image.shape[:2]
inp = transform(image=image)["image"].unsqueeze(0).to(device)

with torch.no_grad():
    probs = torch.sigmoid(model(inp))

# Threshold at full resolution, not at 256x256. Upsampling a binary mask
# with nearest-neighbour quantises the boundary to the 256-grid, which is a
# visible staircase on a 1022x767 dermoscopy image.
probs = F.interpolate(probs, size=(h, w), mode="bilinear", align_corners=False)
mask = (probs.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
```

For test-time augmentation and mask cleanup — both worth using, both free —
see `src/inference.py` in the GitHub repo.

---

## Limitations

- Dice 0.736 is below published baselines for this benchmark.
- Weakest on low-contrast and amelanotic lesions, and on images containing
  rulers, ink markings or coloured stickers. In the sampled cases above, the
  worst failure segments a calibration sticker instead of the lesion.
- Heavy vignetting degrades results; reliability drops near image borders.
- Trained and evaluated at 256 × 256. Fine boundary detail is discarded at
  that resolution, which caps achievable Dice regardless of architecture.
- Single train/validation split, no cross-validation, so the reported figure
  carries a meaningful error bar.
- Dermoscopy only. Will not transfer to histology, clinical photography or
  other modalities.
- Evaluated on ISIC 2018 alone; performance on other dermoscopy datasets is
  unmeasured.
- **Not a medical device.** Not clinically validated.

---

## Intended Use

| Appropriate | Not appropriate |
|---|---|
| Research and experimentation | Clinical diagnosis |
| Benchmarking segmentation methods | Medical decision-making |
| Educational demonstration | Deployment without clinical validation |
| Pre-processing in research pipelines | Any safety-critical application |

---

## Citation

```bibtex
@misc{hassan2026skinsegvision,
  author    = {Hassan, Najmi},
  title     = {Skin Lesion Segmentation with ResNet-34 U-Net on ISIC 2018},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/NajmiHassan1/skinseg-vision}
}
```

**Dataset:**

```bibtex
@article{codella2019skin,
  title   = {Skin lesion analysis toward melanoma detection 2018: A challenge
             hosted by the International Skin Imaging Collaboration (ISIC)},
  author  = {Codella, Noel and Rotemberg, Veronica and Tschandl, Philipp and
             others},
  journal = {arXiv preprint arXiv:1902.03368},
  year    = {2019}
}

@article{tschandl2018ham10000,
  title   = {The HAM10000 dataset, a large collection of multi-source
             dermatoscopic images of common pigmented skin lesions},
  author  = {Tschandl, Philipp and Rosendahl, Cliff and Kittler, Harald},
  journal = {Scientific Data},
  volume  = {5},
  pages   = {180161},
  year    = {2018}
}
```

---

## Acknowledgements

- [ISIC Archive](https://www.isic-archive.com/) for the dataset
- [torchvision](https://pytorch.org/vision/) for the ResNet-34 backbone
- [Albumentations](https://albumentations.ai/) for augmentation
- [Gradio](https://gradio.app/) for the demo interface