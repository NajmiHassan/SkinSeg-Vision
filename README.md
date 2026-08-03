# SkinSeg-Vision — Skin Lesion Segmentation on ISIC 2018

A ResNet-34 U-Net for binary lesion segmentation in dermoscopy images, trained
end-to-end on the ISIC 2018 Task 1 dataset. Includes the training pipeline, a
reproducible evaluation harness, and a Gradio demo.

[**Live demo**](https://huggingface.co/spaces/NajmiHassan1/SkinSeg-Vision) ·
[**Weights**](https://huggingface.co/NajmiHassan1/skinseg-vision)

![Predictions](assets/predictions.png)

---

## Results

Held-out split of 390 images, scored **one image at a time** at threshold 0.5.

| Configuration | Dice | IoU | Pixel acc. |
|---|---|---|---|
| Raw model output | 0.736 | 0.620 | 0.906 |
| + test-time augmentation | *run `evaluate.py --ablate`* | | |
| + mask cleanup | *run `evaluate.py --ablate`* | | |

The 0.736 figure is the one to compare against published ISIC numbers. It is
mid-range for this benchmark — competitive baselines land around 0.85–0.90.

### Why the training log says 0.772

The metric printed each epoch during training pools every pixel in a batch of
16 into a single overlap calculation. Large lesions then dominate both the
numerator and the denominator, and errors on small lesions barely move the
number. Per-image averaging — one Dice per image, then a mean — is what the
segmentation literature reports, and it gives 0.736 for the same checkpoint.

`src/losses.py` now averages per image by default. `dice_score_pooled` is kept
only so the older logs remain interpretable; it should not be reported.

---

## Training

| | |
|---|---|
| Dataset | ISIC 2018 Task 1 — 2,594 image/mask pairs |
| Split | 2,204 train / 390 validation, seed 42 |
| Input | 256×256, ImageNet normalisation |
| Architecture | ResNet-34 encoder (ImageNet) + U-Net decoder, 47.9M params |
| Loss | 0.5 × BCEWithLogits + 0.5 × soft Dice |
| Optimiser | AdamW, weight decay 1e-4 |
| Learning rate | encoder 1e-4, decoder 1e-3, cosine annealing to 1e-6 |
| Schedule | 25 epochs, early stopping patience 8 |
| Precision | AMP (fp16) |
| Hardware | Tesla T4, ~92 minutes |

![Training curves](assets/training_history.png)

**On the differential learning rates.** A single LR across the whole network
is the usual reason a pre-trained-encoder U-Net plateaus early. An LR high
enough to train a randomly initialised decoder will wash the ImageNet features
out of the encoder; an LR low enough to preserve them leaves the decoder
barely moving. Splitting the two roughly ten-to-one resolves the conflict.

Train and validation curves track each other closely to the end, with no
divergence — the model is **underfitting**, not overfitting. More capacity,
higher input resolution, or longer training are the levers that will move the
number, not more regularisation.

---

## Repository layout

```
├── src/
│   ├── model.py        ResNet-34 U-Net
│   ├── dataset.py      Dataset, augmentations, loaders
│   ├── losses.py       BCE+Dice loss, per-image metrics
│   ├── inference.py    TTA, full-res thresholding, mask cleanup
│   └── checkpoint.py   Checkpoint loading and weight export
├── scripts/
│   ├── train.py        Training loop
│   ├── evaluate.py     Per-image evaluation with ablations
│   └── export_weights.py
├── app/app.py          Gradio demo
├── space/              Flat bundle ready to push to a HF Space
├── notebooks/          Kaggle training notebook
└── assets/             Figures
```

---

## Quick start

```bash
git clone https://github.com/NajmiHassan/SkinSeg-Vision
cd SkinSeg-Vision
pip install -r requirements.txt
```

### Run the demo

Weights download from the Hub automatically on first launch:

```bash
python app/app.py
```

Or point at a local checkpoint:

```bash
MODEL_PATH=checkpoints/best_weights.pth python app/app.py
```

### Train

```bash
python scripts/train.py \
  --image_dir data/ISIC2018_Task1-2_Training_Input \
  --mask_dir  data/ISIC2018_Task1_Training_GroundTruth \
  --epochs 25 --batch_size 16
```

Writes `checkpoints/best_model.pth` (full checkpoint), `best_weights.pth`
(model tensors only), `history.json`, and `val_split.json`.

### Evaluate

```bash
python scripts/evaluate.py \
  --image_dir data/ISIC2018_Task1-2_Training_Input \
  --mask_dir  data/ISIC2018_Task1_Training_GroundTruth \
  --weights checkpoints/best_weights.pth \
  --split_file checkpoints/val_split.json \
  --ablate
```

---

## Inference-time improvements

Three things the demo does that training did not, each free:

**Test-time augmentation.** Averages sigmoid maps over the four-element flip
group. The model was trained with flip augmentation and is already roughly
flip-equivariant, so averaging cancels boundary jitter.

**Full-resolution thresholding.** The probability map is upsampled bilinearly
to the original image size *before* thresholding. Thresholding at 256×256 and
then upsampling the binary mask with nearest-neighbour quantises the boundary
to the 256-grid — a visible staircase on a 1022×767 dermoscopy image.

**Largest-component + hole filling.** ISIC Task 1 ground truth is always a
single contiguous lesion. Any additional blob is therefore a guaranteed false
positive, and the model produces plenty of them: ruler markings, ink dots,
vignette corners. Keeping only the largest connected component removes them
outright, and filling interior holes closes the speckle gaps visible in the
raw predictions above.

---

## Known limitations

- Dice 0.736 is not state of the art. Published ISIC 2018 baselines reach
  0.85–0.90.
- Weakest on low-contrast and amelanotic lesions, images with rulers or
  coloured stickers, and heavily vignetted captures.
- Trained and evaluated at 256×256; fine boundary detail is lost at that
  resolution, which caps achievable Dice regardless of architecture.
- Single train/val split, no cross-validation, so the reported figure carries
  a meaningful error bar.
- `UNet.up_to_half` is a dead layer, never called in `forward()`. It stays
  declared so the released checkpoint loads strictly; remove it at the next
  retrain.

## Next steps

Ranked by expected gain per unit of effort:

1. **Train at 384×384 or 512×512.** The single biggest constraint. The curves
   show underfitting, and boundary precision is resolution-bound.
2. **Longer schedule.** Validation Dice was still climbing at epoch 25 and
   early stopping never triggered.
3. **Boundary-aware loss.** Add a Tversky or boundary term to penalise
   contour error, which is what Dice under-weights.
4. **Five-fold cross-validation** for an honest error bar.
5. **Hair removal preprocessing** (DullRazor-style inpainting) — hair occlusion
   is a visible failure mode in the sample predictions.

---

## Dataset

ISIC 2018 Challenge Task 1, via the
[Kaggle mirror](https://www.kaggle.com/datasets/tschandl/isic2018-challenge-task1-data-segmentation).
Please observe the ISIC Archive terms of use.

## Disclaimer

Research and educational use only. Not a medical device, not clinically
validated, not for diagnostic use.

## License

MIT
