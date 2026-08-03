# Deploying to Hugging Face

Two repos, because a 182 MB file does not belong in a Space's git history:
a **model repo** holding the weights, and a **Space** holding the app that
downloads them at startup.

The Hugging Face username is `NajmiHassan1`, the Kaggle username is
`najmihassan101`, and the GitHub handle is `NajmiHassan`.

---

## 0. Prepare the weights

Download the Kaggle output first:

```bash
kaggle kernels output najmihassan101/skinseg-vision -p ./kaggle_output
```

You want `best_weights.pth` (182 MB), not `checkpoints/best_model.pth`
(547 MB) — the larger file is two thirds AdamW optimizer moments, which are
useless for inference. If you only have the big one:

```bash
python scripts/export_weights.py \
  --src kaggle_output/checkpoints/best_model.pth \
  --dst best_weights.pth
```

Sanity-check that it loads before uploading anything:

```bash
python -c "
from src.model import UNet
from src.checkpoint import load_into
m = UNet(pretrained=False)
meta = load_into(m, 'best_weights.pth')
print('loaded OK', meta.get('best_dice'))
"
```

If that prints without raising, the architecture in `src/model.py` matches the
checkpoint exactly and the Space will work.

---

## 1. Model repo

```bash
pip install huggingface_hub
hf auth login                  # paste a WRITE token from hf.co/settings/tokens

hf repo create skinseg-vision --repo-type model
hf upload NajmiHassan1/skinseg-vision \
  best_weights.pth best_weights.pth
```

The CLI handles LFS for you. Verify the file appears at
`huggingface.co/NajmiHassan1/skinseg-vision/tree/main`.

---

## 2. Space

```bash
hf repo create SkinSeg-Vision --repo-type space --space-sdk gradio --flavor zero-a10g
git clone https://huggingface.co/spaces/NajmiHassan1/SkinSeg-Vision hf-space
```

ZeroGPU hands the Space a GPU only for the duration of a call to a function
decorated with `@spaces.GPU`, and takes it back as soon as that function
returns. That has three consequences for the code. `torch.cuda.is_available()`
runs at import time, before any decorated function has been entered, so it
returns `False` even though a GPU will be there inside the decorated call —
`app.py` therefore reads the `SPACES_ZERO_GPU` environment variable to decide
what device to target. Weights are loaded with `map_location='cpu'` and the
model is moved to the device afterwards, because at startup there is no GPU to
load them onto. And `requirements.txt` must list the `spaces` package, which
provides the decorator.

Copy in the flat bundle — the Space root has no `src/` package, which is why
`app.py` carries an import shim that falls back to flat imports:

```bash
cp space/app.py space/model.py space/inference.py space/checkpoint.py \
   space/requirements.txt space/README.md hf-space/

cd hf-space
git add -A
git commit -m "ResNet-34 U-Net lesion segmentation demo"
git push
```

The Space README's YAML frontmatter is what configures the Space — `sdk`,
`app_file`, `sdk_version`. Without it the build will not start.

Build takes about 4–6 minutes. The first request is slow: the Space downloads
the weights from your model repo, then caches them.

---

## 3. Optional — ship weights inside the Space instead

Skips the Hub download and makes cold starts faster, at the cost of a fat
Space repo:

```bash
cd hf-space
git lfs install
git lfs track "*.pth"
cp ../best_weights.pth .
git add .gitattributes best_weights.pth
git commit -m "Add weights"
git push
```

`resolve_weights()` checks local paths before reaching for the Hub, so no code
change is needed.

---

## 4. Example images

The demo shows an example gallery if `assets/examples/` exists. Add three or
four ISIC images:

```bash
mkdir -p hf-space/assets/examples
cp data/ISIC2018_Task1-2_Training_Input/ISIC_0000000.jpg hf-space/assets/examples/
```

Prefer images from the **validation** split — showing off on training images
is a bad look on a public demo.

---

## Configuration

Set these under Space **Settings → Variables and secrets** if you need to
override the defaults:

| Variable | Default | Purpose |
|---|---|---|
| `HF_MODEL_REPO` | `NajmiHassan1/skinseg-vision` | Hub repo holding the weights |
| `HF_MODEL_FILE` | `best_weights.pth` | Filename inside that repo |
| `MODEL_PATH` | *unset* | Absolute path, overrides everything |

None of these are secrets — plain variables are fine.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'src'`** — you copied `app/app.py` but
not `model.py`, `inference.py` and `checkpoint.py` alongside it. The Space
needs all four at the root.

**`Error(s) in loading state_dict ... Missing key(s): up_to_half.weight`** —
`model.py` on the Space is out of sync with the one used at training time. Copy
`src/model.py` over again; the dead `up_to_half` layer must stay declared.

**`RuntimeError: CUDA error`** on a free Space — free Spaces are CPU-only.
`DEVICE` already falls back to CPU, so this means something explicitly asked
for CUDA. Check `MODEL_PATH` is not pointing at a CUDA-serialised tensor;
`load_into` passes `map_location`, which handles it.

**Space builds but the page is blank** — check the Logs tab. Almost always a
`sdk_version` in the README that does not exist. Bump it to a real Gradio
release.

**Predictions look worse than in the notebook** — the notebook thresholded at
256×256; the app thresholds at full resolution after bilinear upsampling. That
is the better behaviour, but it will not be pixel-identical.
