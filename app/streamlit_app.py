"""
Streamlit web app for ISIC 2018 skin lesion segmentation.

Run locally:
    streamlit run app/streamlit_app.py

Deploy to Hugging Face Spaces (SDK: streamlit):
    Upload this file + best_model.pth + requirements.txt to your Space.
"""

import io
import os
import sys

import albumentations as A
import numpy as np
import streamlit as st
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.model import UNet

# ──────────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────────

CHECKPOINT = os.path.join(os.path.dirname(__file__), '..', 'best_model.pth')
IMG_SIZE   = 256
DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_MEAN = np.array([0.485, 0.456, 0.406])
_STD  = np.array([0.229, 0.224, 0.225])

st.set_page_config(
    page_title="Skin Lesion Segmentation",
    page_icon="🔬",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────
#  Load Model (cached across reruns and sessions)
# ──────────────────────────────────────────────────────────────


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    """Load the trained U-Net once; reused for every rerun."""
    model = UNet(pretrained=False).to(DEVICE)
    # weights_only=False: the checkpoint also carries optimizer state and config.
    ckpt  = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    return model, ckpt


@st.cache_resource
def get_transform():
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=_MEAN.tolist(), std=_STD.tolist()),
        ToTensorV2(),
    ])


# ──────────────────────────────────────────────────────────────
#  Inference
# ──────────────────────────────────────────────────────────────


def predict(model, pil_image: Image.Image, threshold: float):
    """Run segmentation and return (mask, overlay, lesion coverage %)."""
    img_np = np.array(pil_image.convert('RGB'))
    orig_h, orig_w = img_np.shape[:2]

    aug = get_transform()(image=img_np)
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

    return mask_pil, overlay_pil, mask_np_bool.mean() * 100


def to_png_bytes(pil_image: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil_image.save(buf, format='PNG')
    return buf.getvalue()


# ──────────────────────────────────────────────────────────────
#  Streamlit Interface
# ──────────────────────────────────────────────────────────────

st.title("🔬 Skin Lesion Segmentation — ISIC 2018")
st.markdown(
    "Upload a dermoscopy image and the model will automatically delineate the "
    "lesion boundary."
)

with st.sidebar:
    st.header("Model")
    st.markdown(
        "**Architecture:** ResNet-34 encoder + U-Net decoder  \n"
        "**Loss:** BCE + Dice  \n"
        "**Training set:** ISIC 2018 Task 1 (~2,594 images)  \n"
        "**Metrics (validation):** Dice ≈ 0.87 · IoU ≈ 0.80"
    )
    st.divider()
    st.header("Settings")
    threshold = st.slider(
        "Binarisation threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.5,
        step=0.05,
        help="Sigmoid probability above which a pixel is labelled as lesion.",
    )
    st.caption(f"Running on **{DEVICE.type.upper()}**")

if not os.path.exists(CHECKPOINT):
    st.error(
        f"Checkpoint not found at `{os.path.abspath(CHECKPOINT)}`.\n\n"
        "Download `best_model.pth` from the Hugging Face model repository and "
        "place it in the repository root."
    )
    st.stop()

model, ckpt = load_model()

uploaded = st.file_uploader(
    "Input dermoscopy image",
    type=['png', 'jpg', 'jpeg', 'bmp', 'tif', 'tiff'],
)

if uploaded is None:
    st.info("Upload an image to run segmentation.")
    st.stop()

image = Image.open(uploaded)

with st.spinner("Running segmentation..."):
    mask_pil, overlay_pil, lesion_pct = predict(model, image, threshold)

col1, col2, col3 = st.columns(3)
with col1:
    st.image(image, caption="Input image", width='stretch')
with col2:
    st.image(mask_pil, caption="Predicted mask", width='stretch')
with col3:
    st.image(overlay_pil, caption="Overlay (green = lesion)", width='stretch')

m1, m2, m3 = st.columns(3)
m1.metric("Lesion coverage", f"{lesion_pct:.1f}%")
m2.metric("Threshold", f"{threshold:.2f}")
m3.metric("Best val Dice", f"{ckpt.get('best_dice', 0):.3f}")

d1, d2 = st.columns(2)
d1.download_button(
    "⬇️ Download mask (PNG)",
    data=to_png_bytes(mask_pil),
    file_name=f"{os.path.splitext(uploaded.name)[0]}_mask.png",
    mime="image/png",
    width='stretch',
)
d2.download_button(
    "⬇️ Download overlay (PNG)",
    data=to_png_bytes(overlay_pil),
    file_name=f"{os.path.splitext(uploaded.name)[0]}_overlay.png",
    mime="image/png",
    width='stretch',
)

st.caption(
    "Research and educational use only — this is not a medical device and is "
    "not validated for clinical diagnostics."
)
