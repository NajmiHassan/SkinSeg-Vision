"""
Checkpoint loading and export.

The training loop writes a full checkpoint (model + optimizer + config,
~547 MB). Deployment only needs the model tensors (~182 MB), so the two
formats are both supported here and `export_weights` converts one to
the other.
"""

from __future__ import annotations

import os
from typing import Any

import torch


# Keys a checkpoint dict might use for the model tensors, in priority order.
_STATE_KEYS = ('model_state', 'state_dict', 'model_state_dict', 'model')


def _unwrap_state_dict(obj: Any) -> tuple[dict, dict]:
    """
    Pull (state_dict, metadata) out of whatever torch.load returned.

    Handles a bare state_dict, a training checkpoint, and a DataParallel
    checkpoint whose keys carry a 'module.' prefix.
    """
    meta: dict = {}

    if isinstance(obj, dict) and any(k in obj for k in _STATE_KEYS):
        for key in _STATE_KEYS:
            if key in obj:
                state = obj[key]
                break
        meta = {k: v for k, v in obj.items() if k not in _STATE_KEYS}
    else:
        state = obj

    while isinstance(state, dict) and any(k in state for k in _STATE_KEYS):
        for key in _STATE_KEYS:
            if key in state:
                state = state[key]
                break    

    if not isinstance(state, dict):
        raise TypeError(
            f"Expected a state_dict, got {type(state).__name__}. "
            "Is this file actually a model checkpoint?"
        )

    if any(k.startswith('module.') for k in state):
        state = {k.removeprefix('module.'): v for k, v in state.items()}

    return state, meta


def load_into(model: torch.nn.Module, path: str,
              map_location: Any = 'cpu', strict: bool = True) -> dict:
    """
    Load weights from `path` into `model`. Returns the checkpoint metadata
    (epoch, best_dice, config, ...) — empty when the file is a bare
    state_dict.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint at {path}")

    # weights_only=True is the safe default. Training checkpoints store a
    # plain-dict config, so they load fine under it; fall back only if the
    # file carries something exotic that the safe unpickler rejects.
    try:
        obj = torch.load(path, map_location=map_location, weights_only=True)
    except Exception:
        obj = torch.load(path, map_location=map_location, weights_only=False)

    state, meta = _unwrap_state_dict(obj)
    missing, unexpected = model.load_state_dict(state, strict=strict)

    if missing or unexpected:
        print(f"  missing keys    : {len(missing)}")
        print(f"  unexpected keys : {len(unexpected)}")

    return meta


def export_weights(src: str, dst: str) -> None:
    """
    Strip optimizer state from a training checkpoint.

    Cuts the file roughly to a third of its size, which matters when the
    artifact has to travel through Git LFS or the Hugging Face Hub.
    """
    obj = torch.load(src, map_location='cpu', weights_only=False)
    state, meta = _unwrap_state_dict(obj)

    payload = {
        'model_state': state,
        'best_dice': meta.get('best_dice'),
        'epoch': meta.get('epoch'),
        'arch': 'resnet34_unet',
        'img_size': 256,
    }
    torch.save(payload, dst)

    before = os.path.getsize(src) / 1024 ** 2
    after = os.path.getsize(dst) / 1024 ** 2
    print(f"{src}  {before:.1f} MB  ->  {dst}  {after:.1f} MB")
