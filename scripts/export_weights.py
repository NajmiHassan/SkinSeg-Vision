"""
Strip optimizer state from a training checkpoint.

    python scripts/export_weights.py \
        --src checkpoints/best_model.pth \
        --dst best_weights.pth

547 MB -> ~182 MB. Run this before pushing weights to the Hub or Git LFS;
the optimizer moments are dead weight for inference.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from src.checkpoint import export_weights  # noqa: E402

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--src', default='checkpoints/best_model.pth')
    p.add_argument('--dst', default='best_weights.pth')
    a = p.parse_args()
    export_weights(a.src, a.dst)
