"""
Gradio web app for ISIC 2018 skin lesion segmentation.

Run locally:
    python app/app.py

Deploy to Hugging Face Spaces (SDK: gradio):
    Upload this file + best_model.pth + requirements.txt to your Space.
"""

import os
import sys

import albumentations as A
import gradio as gr
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import UNet

# ──────────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────────

ROOT       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT = os.path.join(ROOT, 'best_model.pth')
IMG_SIZE   = 256
THRESHOLD  = 0.5
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_MEAN = np.array([0.485, 0.456, 0.406])
_STD  = np.array([0.229, 0.224, 0.225])

# ──────────────────────────────────────────────────────────────
#  Load Model (once at startup)
# ──────────────────────────────────────────────────────────────

if not os.path.exists(CHECKPOINT):
    raise SystemExit(
        f"Checkpoint not found at {CHECKPOINT}\n\n"
        "Download the weights and place them in the repository root:\n"
        "  curl -L -o best_model.pth \\\n"
        "    https://huggingface.co/DevHabiba/skin-lesion-segmentation-unet/"
        "resolve/main/best_full_supervised_model.pth"
    )

print(f"Loading model on {DEVICE}...")
model = UNet(pretrained=False).to(DEVICE)
# weights_only=False: the checkpoint also carries optimizer state and config.
ckpt  = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state'])
model.eval()

BEST_DICE = ckpt.get('best_dice')
# Plain ASCII: the Windows console defaults to cp1252 and cannot encode emoji.
if BEST_DICE is not None:
    print(f"Model loaded (best Val Dice: {BEST_DICE:.4f})")
else:
    print("Model loaded")

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=_MEAN.tolist(), std=_STD.tolist()),
    ToTensorV2(),
])

# ──────────────────────────────────────────────────────────────
#  Inference
# ──────────────────────────────────────────────────────────────


def predict(pil_image: Image.Image, threshold: float = THRESHOLD):
    """Run segmentation and return the mask, the overlay, and a summary."""
    if pil_image is None:
        return None, None, "Upload a dermoscopy image to run segmentation."

    img_np = np.array(pil_image.convert('RGB'))
    orig_h, orig_w = img_np.shape[:2]

    aug = transform(image=img_np)
    inp = aug['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(inp)
        prob   = torch.sigmoid(logits).squeeze().cpu().numpy()   # (256, 256)
        mask   = (prob > threshold).astype(np.uint8) * 255

    # Resize output back to original resolution
    mask_pil = Image.fromarray(mask, mode='L').resize(
        (orig_w, orig_h), Image.NEAREST
    )

    # Colour overlay: green channel on original
    overlay      = img_np.copy()
    mask_np_bool = np.array(mask_pil) > 127
    overlay[mask_np_bool, 0] = (overlay[mask_np_bool, 0] * 0.5).astype(np.uint8)
    overlay[mask_np_bool, 1] = np.clip(
        overlay[mask_np_bool, 1] * 0.5 + 128, 0, 255
    ).astype(np.uint8)
    overlay[mask_np_bool, 2] = (overlay[mask_np_bool, 2] * 0.5).astype(np.uint8)
    overlay_pil = Image.fromarray(overlay)

    lesion_pct = mask_np_bool.mean() * 100
    dice_txt   = f"{BEST_DICE:.3f}" if BEST_DICE is not None else "n/a"
    info = (
        f"**Lesion coverage:** {lesion_pct:.1f}% of image  \n"
        f"**Threshold:** {threshold:.2f}\n\n"
        f"*Model: ResNet-34 U-Net | Dataset: ISIC 2018 | "
        f"Best Val Dice ≈ {dice_txt}*"
    )
    return mask_pil, overlay_pil, info


# ──────────────────────────────────────────────────────────────
#  Gradio Interface
# ──────────────────────────────────────────────────────────────

DESCRIPTION = """
## Skin Lesion Segmentation — ISIC 2018

Upload a dermoscopy image and the model will automatically delineate the lesion boundary.

**Architecture:** ResNet-34 encoder + U-Net decoder  
**Loss:** BCE + Dice  
**Training set:** ISIC 2018 Task 1 (~2,594 images)  
**Metrics (validation):** Dice ≈ 0.87 · IoU ≈ 0.80
"""

EXAMPLE_IMAGE = os.path.join(ROOT, 'assets', 'example_lesion.png')

with gr.Blocks(title="Skin Lesion Segmentation") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        inp_img = gr.Image(type='pil', label='Input dermoscopy image')

    threshold = gr.Slider(
        minimum=0.05,
        maximum=0.95,
        value=THRESHOLD,
        step=0.05,
        label='Binarisation threshold',
        info='Sigmoid probability above which a pixel is labelled as lesion.',
    )

    run_btn = gr.Button("Run segmentation", variant="primary")

    with gr.Row():
        out_mask    = gr.Image(type='pil', label='Predicted mask')
        out_overlay = gr.Image(type='pil', label='Overlay (green = lesion)')

    info_box = gr.Markdown()

    run_btn.click(
        fn=predict,
        inputs=[inp_img, threshold],
        outputs=[out_mask, out_overlay, info_box],
    )

    if os.path.exists(EXAMPLE_IMAGE):
        gr.Examples(
            examples=[[EXAMPLE_IMAGE, THRESHOLD]],
            inputs=[inp_img, threshold],
            cache_examples=False,
        )

    gr.Markdown(
        "Research and educational use only — this is not a medical device and "
        "is not validated for clinical diagnostics."
    )

if __name__ == '__main__':
    demo.launch(share=False)
