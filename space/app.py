"""
Gradio demo for ISIC 2018 skin lesion segmentation — ZeroGPU compatible.

ZeroGPU allocates a GPU only for the duration of a function decorated with
@spaces.GPU, and releases it immediately afterwards. Two consequences shape
this file:

  1. torch.cuda.is_available() returns False at import time, even though a
     GPU will be available inside the decorated function. So the usual
     availability check picks the wrong device. Detect ZeroGPU from the
     environment instead.

  2. Weights must be loaded with map_location='cpu' and the model moved to
     the device afterwards. Loading straight to 'cuda' at startup happens
     before any GPU exists.

The @spaces.GPU decorator is a no-op outside ZeroGPU, so this file still
runs unchanged on a local machine or on CPU hardware.
"""

from __future__ import annotations

import os
import sys

import albumentations as A
import gradio as gr
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image

# `spaces` only exists on Hugging Face infrastructure. Locally, fall back to
# a decorator that does nothing, so the same file runs in both places.
try:
    import spaces
    _HAS_SPACES = True
except ImportError:
    _HAS_SPACES = False

    class _SpacesShim:
        @staticmethod
        def GPU(*args, **kwargs):
            # Support both @spaces.GPU and @spaces.GPU(duration=...)
            if len(args) == 1 and callable(args[0]) and not kwargs:
                return args[0]
            return lambda fn: fn

    spaces = _SpacesShim()

# ── Import shim: works flat (Spaces) or packaged (GitHub repo) ────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from src.model import UNet
    from src.inference import (IMAGENET_MEAN, IMAGENET_STD, postprocess,
                               predict_probs, upsample_probs)
    from src.checkpoint import load_into
except ModuleNotFoundError:
    from model import UNet
    from inference import (IMAGENET_MEAN, IMAGENET_STD, postprocess,
                           predict_probs, upsample_probs)
    from checkpoint import load_into


# ──────────────────────────────────────────────────────────────
#  Config
# ──────────────────────────────────────────────────────────────

IMG_SIZE = 256
THRESHOLD = 0.5

# On ZeroGPU, cuda is not visible at import time but will be inside the
# decorated function, so trust the environment flag over is_available().
ON_ZEROGPU = os.environ.get('SPACES_ZERO_GPU') is not None
if ON_ZEROGPU:
    DEVICE = torch.device('cuda')
else:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

HF_MODEL_REPO = os.environ.get('HF_MODEL_REPO', 'NajmiHassan1/skinseg-vision')
WEIGHTS_FILE = os.environ.get('HF_MODEL_FILE', 'best_weights.pth')

# Reported on the 390-image held-out split, one image at a time.
METRICS = {'dice': 0.7360, 'iou': 0.6201, 'pixel_acc': 0.9057}


def resolve_weights() -> str:
    """Find the checkpoint locally, or pull it from the Hub."""
    explicit = os.environ.get('MODEL_PATH')
    candidates = [explicit] if explicit else []
    candidates += [
        os.path.join(_HERE, WEIGHTS_FILE),
        os.path.join(_ROOT, WEIGHTS_FILE),
        os.path.join(_ROOT, 'checkpoints', 'best_model.pth'),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path

    print(f"No local weights found; fetching {HF_MODEL_REPO}/{WEIGHTS_FILE}")
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo_id=HF_MODEL_REPO, filename=WEIGHTS_FILE)


# ──────────────────────────────────────────────────────────────
#  Load model once at startup
# ──────────────────────────────────────────────────────────────

print(f"ZeroGPU: {ON_ZEROGPU} | target device: {DEVICE}")
_ckpt_path = resolve_weights()

model = UNet(pretrained=False)
# Always load to CPU first: on ZeroGPU no GPU exists yet at this point.
_meta = load_into(model, _ckpt_path, map_location='cpu')
model.eval()
model.to(DEVICE)

print(f"Loaded weights from {_ckpt_path}")
if _meta.get('best_dice') is not None:
    print(f"  training-time best val Dice: {_meta['best_dice']:.4f}")

transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
    ToTensorV2(),
])


# ──────────────────────────────────────────────────────────────
#  Inference
# ──────────────────────────────────────────────────────────────

@spaces.GPU(duration=30)
def predict(pil_image: Image.Image,
            threshold: float = THRESHOLD,
            use_tta: bool = True,
            clean_mask: bool = True):
    """Segment a dermoscopy image; return (mask, overlay, summary markdown)."""
    if pil_image is None:
        return None, None, "Upload a dermoscopy image to run segmentation."

    img_np = np.array(pil_image.convert('RGB'))
    orig_h, orig_w = img_np.shape[:2]

    inp = transform(image=img_np)['image'].unsqueeze(0).to(DEVICE)

    probs = predict_probs(model, inp, tta=use_tta)              # (1,1,256,256)
    probs = upsample_probs(probs, (orig_h, orig_w))             # (1,1,H,W)
    prob_np = probs.squeeze().cpu().numpy()

    mask = (prob_np > threshold).astype(np.uint8)
    if clean_mask:
        mask = postprocess(mask)

    mask_bool = mask.astype(bool)
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8), mode='L')

    # Green tint inside the lesion, so the underlying texture stays visible.
    overlay = img_np.copy()
    overlay[mask_bool, 0] = (overlay[mask_bool, 0] * 0.5).astype(np.uint8)
    overlay[mask_bool, 1] = np.clip(
        overlay[mask_bool, 1] * 0.5 + 128, 0, 255).astype(np.uint8)
    overlay[mask_bool, 2] = (overlay[mask_bool, 2] * 0.5).astype(np.uint8)
    overlay_pil = Image.fromarray(overlay)

    coverage = mask_bool.mean() * 100
    confidence = float(prob_np[mask_bool].mean()) if mask_bool.any() else 0.0

    if not mask_bool.any():
        info = (
            "**No lesion detected** at this threshold.\n\n"
            "Try lowering the threshold, or disabling mask cleanup if the "
            "lesion is genuinely fragmented."
        )
    else:
        info = (
            f"**Lesion coverage:** {coverage:.1f}% of image  \n"
            f"**Mean confidence inside mask:** {confidence:.2f}  \n"
            f"**Threshold:** {threshold:.2f} · "
            f"**TTA:** {'on' if use_tta else 'off'} · "
            f"**Cleanup:** {'on' if clean_mask else 'off'}"
        )
    return mask_pil, overlay_pil, info


# ──────────────────────────────────────────────────────────────
#  Interface
# ──────────────────────────────────────────────────────────────

DESCRIPTION = f"""
# Skin Lesion Segmentation — ISIC 2018

Upload a dermoscopy image and the model will delineate the lesion boundary.

**Architecture:** ResNet-34 encoder + U-Net decoder (47.9M parameters)
**Loss:** BCE + soft Dice · **Trained on:** ISIC 2018 Task 1, 2,204 images
**Held-out performance** (390 images, scored per image):
Dice **{METRICS['dice']:.3f}** · IoU **{METRICS['iou']:.3f}** ·
Pixel accuracy **{METRICS['pixel_acc']:.3f}**
"""

NOTES = """
### About the controls

**Threshold** — the sigmoid probability above which a pixel counts as lesion.
Lower it for faint or amelanotic lesions; raise it when the prediction bleeds
into surrounding skin.

**Test-time augmentation** — averages predictions over four flips. Costs ~4x
the compute and smooths boundary noise. Worth leaving on.

**Mask cleanup** — keeps only the largest connected region and fills interior
holes. ISIC ground truth is always a single contiguous lesion, so this removes
ruler marks, ink dots and vignette corners that the model sometimes picks up.
Note that on an image the model does not understand, cleanup will still return
one smooth confident-looking blob — it tidies the output without making it
correct.

### Known limitations

This model expects **dermoscopy** images: contact macro shots where a single
lesion fills much of the frame under flat, even lighting. Clinical photographs,
phone snapshots and whole-body or facial images are outside its training
distribution and the output on them is not meaningful.

Trained at 256x256 on 2.2k images for 25 epochs; per-image Dice of 0.74 is
mid-range for this benchmark, not state of the art. It struggles most with
low-contrast lesions, images containing rulers or coloured stickers, and
strong vignetting. Boundaries tend to be over-segmented by a small margin.
"""

DISCLAIMER = (
    "Research and educational use only. This is not a medical device, has not "
    "been clinically validated, and must not be used for diagnosis."
)

with gr.Blocks(title="Skin Lesion Segmentation") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            inp_img = gr.Image(type='pil', label='Input dermoscopy image')
            threshold = gr.Slider(
                minimum=0.05, maximum=0.95, value=THRESHOLD, step=0.05,
                label='Binarisation threshold',
            )
            with gr.Row():
                tta_box = gr.Checkbox(value=True, label='Test-time augmentation')
                clean_box = gr.Checkbox(value=True, label='Mask cleanup')
            run_btn = gr.Button("Run segmentation", variant="primary")

        with gr.Column(scale=2):
            with gr.Row():
                out_mask = gr.Image(type='pil', label='Predicted mask')
                out_overlay = gr.Image(type='pil', label='Overlay (green = lesion)')
            info_box = gr.Markdown()

    run_btn.click(
        fn=predict,
        inputs=[inp_img, threshold, tta_box, clean_box],
        outputs=[out_mask, out_overlay, info_box],
    )

    for _dir in (os.path.join(_HERE, 'examples'),
                 os.path.join(_ROOT, 'assets', 'examples')):
        if os.path.isdir(_dir):
            _files = sorted(
                os.path.join(_dir, f) for f in os.listdir(_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            )
            if _files:
                gr.Examples(
                    examples=[[f, THRESHOLD, True, True] for f in _files],
                    inputs=[inp_img, threshold, tta_box, clean_box],
                    cache_examples=False,
                )
                break

    gr.Markdown(NOTES)
    gr.Markdown(f"---\n{DISCLAIMER}")


if __name__ == '__main__':
    # gradio_client's schema walker raises "TypeError: argument of type 'bool'
    # is not iterable" on a component schema whose additionalProperties is a
    # bare bool, killing the Space on /info at startup. show_api=False skips
    # schema generation; ssr_mode=False turns off SSR, which this demo does not need.
    demo.launch(share=False, show_api=False, ssr_mode=False)
