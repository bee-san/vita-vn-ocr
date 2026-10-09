#!/usr/bin/env python3
"""Size and cost of a TinyVNOCR config: parameters, export weight bytes, MACs and an
estimated peak activation arena, for the Vita/PSP memory budget.

  cost.py [--width 1.0 --bottleneck 96 --classes 5355 --height 32] [--json]
  cost.py --ckpt runs/NAME/ckpt_last.pt

Everything is measured on the real torch model (forward hooks on a 1 x H x W input),
not re-derived by hand, so it tracks model.py.

Weights, after BatchNorm folding (every conv + BN becomes a conv with a bias):
  fp32 = 4 B per weight and per bias;
  int8 = 1 B per weight + 4 B scale per output channel for the pointwise, row
         collapse, bottleneck and head convs (symmetric per-output-channel int8,
         model.int8_conv / runtime's export.py --int8); the stem and depthwise
         weights and every bias stay 4 B. Vocab text is counted separately (UTF-8 +
         NUL). Blob headers and channel padding are not counted.

Exact sizes from runtime/ (when its train/export.py is importable: after it merges,
or via --export / $VOCR_EXPORT pointing at a checkout's export.py): the blob bytes
build_blob() writes (fp32 and int8; header, channel padding and vocab included) and
plan_arena(), the exact bytes vocr_arena_bytes() asks for, per arithmetic mode (f32,
w8f, w8i) at its default tiling (4-frame tiles, 16-frame head batches). The runtime
streams the line in tiles, so its arena barely grows with the line width.

Non-streaming arena ESTIMATE (always reported, for a runtime that does not tile the
width): fp32 activations, executed layer by layer at full line width W, with
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
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import DS, Ctx, TinyVNOCR, from_checkpoint, int8_conv  # noqa: E402

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
        int8 += (w + F32 * cout if int8_conv(m) else F32 * w) + F32 * b
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


def runtime_exporter(path=None):
    """runtime's export module (exact blob and arena sizes), or None if unavailable."""
    cands = [Path(p) for p in (path, os.environ.get("VOCR_EXPORT")) if p] + [HERE / "export.py"]
    for p in cands:
        if not p.exists():
            continue
        spec = importlib.util.spec_from_file_location("vocr_export", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if all(hasattr(mod, n) for n in ("build_blob", "model_ops", "read_blob", "plan_arena",
                                         "MODE_F32", "MODE_W8F", "MODE_W8I")):
            return mod
    return None


def runtime_sizes(E, model, vocab, height, widths=WIDTHS):
    """Blob bytes and exact arena bytes per mode and width from runtime's exporter."""
    out = {}
    for int8 in (False, True):
        blob, _ = E.build_blob(E.model_ops(model.eval(), height), vocab, height, int8, 1024, b"")
        parsed = E.read_blob(blob)
        out["blob_int8" if int8 else "blob_f32"] = len(blob)
        modes = (("w8f", E.MODE_W8F), ("w8i", E.MODE_W8I)) if int8 else (("f32", E.MODE_F32),)
        for name, mode in modes:
            for w in widths:
                out[f"rt_arena_{name}_{w}"] = E.plan_arena(parsed, w, 0, 0, mode)
    return out


def default_vocab(n_classes):
    """The repo vocab with n_classes - 1 entries, or placeholders of 3 UTF-8 bytes."""
    data = HERE.parent / "data"
    for p in [data / "vocab.txt", *sorted(data.glob("vocab_*.txt"))]:
        v = [c for c in p.read_text(encoding="utf-8").split("\n") if c] if p.exists() else []
        if len(v) == n_classes - 1:
            return v
    return [chr(0x4E00 + i) for i in range(n_classes - 1)]


def report(model, n_classes, height, vocab=None, exporter=None):
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
    if exporter is not None:
        r.update(runtime_sizes(exporter, model, vocab or default_vocab(n_classes), height))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--bottleneck", type=int, default=96)
    ap.add_argument("--classes", type=int, default=5355)
    ap.add_argument("--height", type=int, default=32)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--export", help="runtime's train/export.py (default: next to this file)")
    args = ap.parse_args()
    torch.set_num_threads(1)
    exporter = runtime_exporter(args.export)
    vocab = None
    if args.ckpt:
        model, meta = from_checkpoint(args.ckpt)
        vocab, height, classes = meta["vocab"], meta["height"], len(meta["vocab"]) + 1
    else:
        model = TinyVNOCR(args.classes, width=args.width, bottleneck=args.bottleneck)
        height, classes = args.height, args.classes
    r = report(model, classes, height, vocab, exporter)
    if args.json:
        print(json.dumps(r))
        return
    print(f"params {r['params']:,} (folded {r['params_folded']:,}, head+bottleneck "
          f"{r['head_params']:,}); weights fp32 {r['weights_fp32']:,} B, int8 "
          f"{r['weights_int8']:,} B")
    for w in WIDTHS:
        print(f"  W={w:4d}: {r[f'macs_{w}'] / 1e6:7.1f} MMAC, non-streaming est. arena "
              f"{r[f'arena_{w}']:,} B ({r[f'arena_head_full_{w}']:,} B with all logits materialised)")
    if exporter is None:
        print("  runtime exporter not found: no exact blob / arena sizes")
        return
    print(f"  runtime blob: fp32 {r['blob_f32']:,} B, int8 {r['blob_int8']:,} B")
    for mode in ("f32", "w8f", "w8i"):
        print(f"  runtime arena {mode}: " + ", ".join(f"W={w} {r[f'rt_arena_{mode}_{w}']:,} B"
                                                   for w in WIDTHS))


if __name__ == "__main__":
    main()
