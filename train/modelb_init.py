#!/usr/bin/env python3
"""Model B (vita-vn-ocr #126): a warm start for a shallower TinyVNOCR from a trained checkpoint.

  modelb_init.py --ckpt SRC.pt --out INIT.pt [--depth 0] [--ctx 1] [--height 24]

The target keeps the source's channels, bottleneck, vocab, temporal kernel and stride and has fewer stride-1
DS blocks per stage (--depth, at most the source's) and fewer temporal Ctx blocks (--ctx, at most the source's).
Every module the target keeps takes the source's weights and BatchNorm statistics: the stem, the three
downsampling DS blocks, the first `depth` stride-1 DS blocks of each stage, the row collapse, the first `ctx` Ctx
blocks, the bottleneck and the head. The blocks it drops are residual (x + f(x)), so the target is the source
with those blocks' residual branches removed: a start for a fine-tune, not the same function.

Writes a train.py-format checkpoint (step 0, the target's arch keys) that train2.py --init reads with the
target's --depth / --ctx (and the same --vocab). --height is only recorded: the weights fit any height 17..32.
"""
import argparse
import hashlib
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import Ctx, TinyVNOCR  # noqa: E402


def roles(depth, ctx):
    """The role of each module of TinyVNOCR.features, in order (train/model.py's layout)."""
    out = ["stem", "down1"] + [f"s1.{i}" for i in range(depth)] + ["down2"] + [f"s2.{i}" for i in range(depth)]
    out += ["down3"] + [f"s3.{i}" for i in range(depth)] + ["collapse"] + [f"ctx.{i}" for i in range(ctx)]
    return out


def shallower(src, depth, ctx):
    """The target TinyVNOCR (eval mode) with src's weights in every module it keeps."""
    a = dict(src.arch)
    if a.get("rep"):
        raise ValueError("fuse a rep model first (train/model.py fuse_rep)")
    if not 0 <= depth <= a["depth"] or not 0 <= ctx <= a["ctx"]:
        raise ValueError(f"target depth {depth} / ctx {ctx} must be at most the source's {a['depth']} / {a['ctx']}")
    a.update(depth=depth, ctx=ctx)
    a.pop("rep", None)
    dst = TinyVNOCR(src.head.out_channels, **a)
    s_mods = dict(zip(roles(src.arch["depth"], src.arch["ctx"]), src.features))
    d_roles = roles(depth, ctx)
    assert len(d_roles) == len(dst.features) and len(s_mods) == len(src.features)
    with torch.no_grad():
        for r, mod in zip(d_roles, dst.features):
            assert isinstance(mod, Ctx) == r.startswith("ctx.")
            mod.load_state_dict(s_mods[r].state_dict())
        dst.bottleneck.load_state_dict(src.bottleneck.state_dict())
        dst.head.load_state_dict(src.head.state_dict())
    return dst.eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--ctx", type=int, default=None, help="Ctx blocks to keep (default: the source's)")
    ap.add_argument("--height", type=int, default=None, help="recorded height (default: the source's)")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    from model import from_checkpoint
    src, meta = from_checkpoint(ck)
    ctx = meta["ctx"] if a.ctx is None else a.ctx
    dst = shallower(src, a.depth, ctx)
    out = {"model": dst.state_dict(), "step": 0, "vocab": list(ck["vocab"]), "width": meta["width"],
           "bottleneck": meta["bottleneck"], "depth": a.depth, "ctx": ctx, "ctx_k": meta["ctx_k"],
           "height": int(a.height or meta["height"]), "unk": ck.get("unk"), "chans": ck.get("chans"), "rep": 0,
           "stride": meta["stride"], "freeze_bn_stats": False,
           "init_from": {"path": str(a.ckpt), "sha256": hashlib.sha256(Path(a.ckpt).read_bytes()).hexdigest(),
                         "step": ck.get("step"), "depth": meta["depth"], "ctx": meta["ctx"]}}
    torch.save(out, a.out)
    n_src = sum(p.numel() for p in src.parameters())
    n_dst = sum(p.numel() for p in dst.parameters())
    print(f"{a.out}: depth {meta['depth']} -> {a.depth}, ctx {meta['ctx']} -> {ctx}, params {n_src:,} -> {n_dst:,}")


if __name__ == "__main__":
    main()
