#!/usr/bin/env python3
"""Deterministic random-init checkpoint for the C runtime's parity test in CI.

  fixture_ckpt.py --out PATH [--seed 0] [--width 1.0]

TinyVNOCR (train/model.py) built from a fixed seed and saved in train.py's checkpoint
format ({"model", "vocab", "width", "step", "best_cer"}, plus a "fixture" note).
Freshly initialised BatchNorm folds to nothing (mean 0, var 1, scale 1, shift 0), so the BN
scales and shifts are drawn at random and the running statistics are measured on random
inputs. The exporter's BN folding then really is exercised, and activations stay in a
trained-like range instead of shrinking towards zero layer by layer. No data and no
downloads are involved.
"""
import argparse
import sys
from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "synth"))
from common import load_vocab  # noqa: E402
from model import TinyVNOCR  # noqa: E402

CALIB_WIDTHS = (64, 128, 256, 512)     # random grey inputs, 4 lines each, values in [-1, 1]


@torch.no_grad()
def make(seed=0, width=1.0):
    """Returns (model in eval mode, vocab). Same seed, same weights, bit for bit."""
    vocab = load_vocab()
    torch.manual_seed(seed)
    model = TinyVNOCR(len(vocab) + 1, width=width)
    g = torch.Generator().manual_seed(seed + 1)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.weight.copy_(torch.rand(m.weight.shape, generator=g) + 0.5)          # 0.5 .. 1.5
            m.bias.copy_((torch.rand(m.bias.shape, generator=g) - 0.5) * 0.4)      # -0.2 .. 0.2
            m.reset_running_stats()
            m.momentum = None          # running stats = plain average over the calibration batches
    model.train()
    for w in CALIB_WIDTHS:
        model(torch.rand(4, 1, 32, w, generator=g) * 2 - 1)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.momentum = 0.1           # back to the PyTorch default, as train.py leaves it
    return model.eval(), vocab


def save(path, seed=0, width=1.0):
    model, vocab = make(seed, width)
    ck = {"model": model.state_dict(), "vocab": vocab, "width": width, "step": 0, "best_cer": 1.0,
          "fixture": {"seed": seed, "note": "random init + random BN, CI parity fixture, never trained"}}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ck, path)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=float, default=1.0)
    args = ap.parse_args()
    print(save(args.out, args.seed, args.width))


if __name__ == "__main__":
    main()
