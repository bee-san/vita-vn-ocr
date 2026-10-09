#!/usr/bin/env python3
"""Size and cost of a TinyVNOCR config: parameters, export weight bytes, MACs and an
estimated peak activation arena, for the Vita/PSP memory budget.

  cost.py [--width 1.0 --bottleneck 96 --classes 5355 --height 32] [--json]
  cost.py --ckpt runs/NAME/ckpt_last.pt

Everything is measured on the real torch model (forward hooks on a 1 x H x W input),
not re-derived by hand, so it tracks model.py.

Weights, after BatchNorm folding (every conv + BN becomes a conv with a bias):
  fp32 = 4 B per weight and per bias;
  int8 = 1 B per weight + 4 B scale and 4 B bias per output channel (symmetric
         per-output-channel int8 with fp32 scales and biases, as quantised by
         model.fake_quant_weight). Vocab text is counted separately (UTF-8 + NUL).

Arena (ESTIMATE; replaced by runtime/'s exact formula once it merges): fp32
activations, executed layer by layer at full line width W, with
  - a plain conv / non-residual block holding its input and output;
  - a residual block holding its input x and the depthwise output d, the pointwise
    result being accumulated into x in place (x + pw(d));
  - the head evaluated one frame at a time with an immediate argmax, so only one
    frame of logits (n_classes floats) is live, plus T int32 class ids.
The input buffer (height x W floats after the resize) is part of the arena; the
caller's RGBA crop is not. "head_full" adds the cost of materialising all T x C
logits instead, for a runtime that does not stream the head.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import DS, Ctx, TinyVNOCR, from_checkpoint  # noqa: E402

WIDTHS = (256, 640, 1024)
F32 = 4


def convs(model):
    return [m for m in model.modules() if isinstance(m, nn.Conv2d)]


def weight_bytes(model):
    """(folded fp32 bytes, int8 bytes, folded parameter count)."""
    fp32 = int8 = n = 0
    for m in convs(model):
        w, cout = m.weight.numel(), m.out_channels
        has_bias = m.bias is not None or m is not model.bottleneck   # BN folds into a bias
        b = cout if has_bias else 0
        n += w + b
        fp32 += F32 * (w + b)
        int8 += w + F32 * cout + F32 * b
    return fp32, int8, n


def macs(model, height, width):
    total = 0

    def hook(m, inp, out):
        nonlocal total
        k = m.kernel_size[0] * m.kernel_size[1] * m.in_channels // m.groups
        total += out.numel() * k

    hs = [m.register_forward_hook(hook) for m in convs(model)]
    with torch.no_grad():
        model.eval()(torch.zeros(1, 1, height, width))
    for h in hs:
        h.remove()
    return total


def arena(model, height, width, n_classes):
    """Estimated peak live activation bytes (see module docstring)."""
    x = torch.zeros(1, 1, height, width)
    size = lambda t: t.numel() * F32  # noqa: E731
    peak = size(x)
    with torch.no_grad():
        for blk in model.features:
            if isinstance(blk, (DS, Ctx)):
                d = blk.dw(x)
                peak = max(peak, size(x) + size(d))
                p = blk.pw(d)
                if isinstance(blk, Ctx) or blk.res:
                    x = x + p                     # accumulated into x in place
                else:
                    peak = max(peak, size(d) + size(p))
                    x = p
            else:
                y = blk(x)
                peak = max(peak, size(x) + size(y))
                x = y
        b = model.bottleneck(x)
        peak = max(peak, size(x) + size(b))
        frames = b.shape[-1]
        streamed = max(peak, size(b) + F32 * n_classes + 4 * frames)
        full = max(peak, size(b) + F32 * n_classes * frames)
    return streamed, full


def vocab_bytes(vocab):
    return sum(len(c.encode("utf-8")) + 1 for c in vocab)


def report(model, n_classes, height, vocab=None):
    fp32, int8, folded = weight_bytes(model)
    r = {"params": sum(p.numel() for p in model.parameters()), "params_folded": folded,
         "weights_fp32": fp32, "weights_int8": int8,
         "head_params": model.head.weight.numel() + model.head.bias.numel()
         + model.bottleneck.weight.numel(),
         "vocab_bytes": vocab_bytes(vocab) if vocab else None, "height": height}
    for w in WIDTHS:
        streamed, full = arena(model, height, w, n_classes)
        r[f"macs_{w}"] = macs(model, height, w)
        r[f"arena_{w}"] = streamed
        r[f"arena_head_full_{w}"] = full
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--bottleneck", type=int, default=96)
    ap.add_argument("--classes", type=int, default=5355)
    ap.add_argument("--height", type=int, default=32)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    torch.set_num_threads(1)
    vocab = None
    if args.ckpt:
        model, meta = from_checkpoint(args.ckpt)
        vocab, height, classes = meta["vocab"], meta["height"], len(meta["vocab"]) + 1
    else:
        model = TinyVNOCR(args.classes, width=args.width, bottleneck=args.bottleneck)
        height, classes = args.height, args.classes
    r = report(model, classes, height, vocab)
    if args.json:
        print(json.dumps(r))
        return
    print(f"params {r['params']:,} (folded {r['params_folded']:,}, head+bottleneck "
          f"{r['head_params']:,}); weights fp32 {r['weights_fp32']:,} B, int8 "
          f"{r['weights_int8']:,} B")
    for w in WIDTHS:
        print(f"  W={w:4d}: {r[f'macs_{w}'] / 1e6:7.1f} MMAC, est. arena {r[f'arena_{w}']:,} B "
              f"({r[f'arena_head_full_{w}']:,} B with all logits materialised)")


if __name__ == "__main__":
    main()
