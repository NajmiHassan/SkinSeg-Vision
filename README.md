# 🔬 Skin Lesion Segmentation (ISIC 2018)

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue?style=for-the-badge)](https://huggingface.co/spaces/DevNajmi/skin-lesion-segmentation)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-orange?style=for-the-badge)](https://huggingface.co/DevNajmi/skin-lesion-segmentation-unet)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)
[![PyTorch](https://img.shields.io/badge/PyTorch-%3E%3D%202.0-red.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/)

Automated binary segmentation of skin lesions in dermoscopy images using a **ResNet-34 U-Net** architecture. This repository contains the complete pipeline for dataset preprocessing, training with mixed precision, evaluation, and a Gradio web application for inference.

🔗 **Live Demo:** Try the model instantly on [Hugging Face Spaces](https://huggingface.co/spaces/DevNajmi/skin-lesion-segmentation).

---

## 📈 Performance & Results

The model has been evaluated on a 15% held-out validation split (~390 images) of the ISIC 2018 Task 1 dataset.

| Metric | Score | Description |
| :--- | :---: | :--- |
| **Dice Coefficient / F1** | **0.87** | Measures region-level overlap |
| **IoU (Jaccard Index)** | **0.80** | Stricter metric for boundary alignment |
| **Pixel Accuracy** | **0.95** | Percentage of correctly classified pixels |

### Training History
Below are the loss and metric convergence curves over 40 epochs:

![Training Curves](assets/training_history.png)

---

## 🖼️ Qualitative Predictions

Here are sample validation results demonstrating the model's predictions compared to the ground truth annotations:

![Visual Predictions](assets/predictions.png)

*Each sample shows the raw dermoscopy input image, the expert ground truth mask, and the predicted binary segmentation mask.*

---

## 🛠️ Key Features

- **Hybrid Architecture:** Combining the rich representation power of an ImageNet pre-trained **ResNet-34 encoder** with a custom **U-Net decoder** with skip connections.
- **Robust Loss Formulation:** Joint **Binary Cross-Entropy (BCE) + Soft Dice Loss** to handle class imbalance (background vs. lesion pixels).
- **Data Augmentations:** Heavy geometric and photometric augmentations via [Albumentations](https://albumentations.ai/) (flips, rotations, elastic scale-shifts, color jitters, Gaussian noise) to prevent overfitting.
- **Efficient Training:** Mixed precision (`torch.cuda.amp`) training and Cosine Annealing learning rate schedule.
- **Interactive UI:** Built-in [Gradio](https://gradio.app/) dashboard with an adjustable binarisation threshold and a bundled example image, for quick local deployments and seamless Hugging Face Space integration.

---

## 📂 Repository Structure

```directory
skin-lesion-segmentation/
├── src/
│   ├── __init__.py
│   ├── model.py        # Model architecture (ResNet-34 U-Net)
│   ├── losses.py       # BCE-Dice loss functions & evaluation metrics
│   └── dataset.py      # Custom PyTorch Dataset & Albumentations transforms
├── app/
│   └── app.py          # Gradio interface for local/cloud hosting
├── scripts/
│   └── train.py        # Custom training and validation loop CLI
├── notebooks/
│   └── skinseg-vision.ipynb   # Jupyter notebook containing original research
├── assets/             # Prediction diagrams and training curves
├── .gitignore          # File exclusions (weights, envs, logs, IDEs)
├── LICENSE             # MIT License file
├── requirements.txt    # Required python packages
└── README.md           # Documentation
```

---

## 🚀 Quickstart

### 1. Setup Environment
Clone the repository and install dependencies in a virtual environment:

```bash
# Clone the repository
git clone https://github.com/NajmiHassan/SkinSeg-Vision.git
cd SkinSeg-Vision

# Create and activate virtual environment
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

### 2. Download Pre-trained Weights
Since the model weights (`best_model.pth` ≈ 573MB) exceed GitHub's single-file limits, download them directly from the Hugging Face model repository:

- ⬇️ **Download Checkpoint:** [best_full_supervised_model.pth](https://huggingface.co/DevHabiba/skin-lesion-segmentation-unet/resolve/main/best_full_supervised_model.pth)

Place the file in the repository's root directory and rename it to `best_model.pth`:

```bash
curl -L -o best_model.pth \
  https://huggingface.co/DevHabiba/skin-lesion-segmentation-unet/resolve/main/best_full_supervised_model.pth
```

### 3. Launch Local Web Application
Run the Gradio interface locally to run predictions in your browser:

```bash
python app/app.py
```
Open `http://127.0.0.1:7860` in your browser — upload an image (or click the bundled example), tune the binarisation threshold, and hit **Run segmentation** to visualise the lesion boundary.

---

## 🏋️ Training from Scratch

To retrain the model, first download the dataset from [ISIC 2018 Challenge — Task 1: Lesion Segmentation](https://challenge.isic-archive.com/landing/2018/).

Run the training script using the CLI:

```bash
python scripts/train.py \
    --image_dir "/path/to/ISIC2018_Task1-2_Training_Input" \
    --mask_dir "/path/to/ISIC2018_Task1_Training_GroundTruth" \
    --epochs 40 \
    --batch_size 16 \
    --lr 3e-4
```

Checkpoint models with the best Validation Dice score will be automatically saved under a new `checkpoints/` directory.

---

## ⚖️ Limitations & Intended Use

- **Research and Education:** This model is designed for educational demonstrations and research benchmarking. It is **not** a medical device and is **not validated for clinical diagnostics**.
- **Artifact Sensitivity:** Performance may degrade on images featuring acquisition artifacts (ink marks, ruler lines, surgical tape, or strong vignetting).

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE).

---

## 🗏 Citations & Acknowledgements

If you use this work, please cite the following datasets and architectures:

```bibtex
@misc{devnajmi2026skinlesion,
  author    = {Najmi},
  title     = {Skin Lesion Segmentation using ResNet-34 U-Net on ISIC 2018},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/DevNajmi/skin-lesion-segmentation-unet}
}

@article{tschandl2018ham10000,
  title   = {The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions},
  author  = {Tschandl, Philipp and Rosendahl, Cliff and Kittler, Harald},
  journal = {Scientific data},
  volume  = {5},
  pages   = {180161},
  year    = {2018}
}
```
