#!/usr/bin/env python3
"""Parity of the C runtime against PyTorch.

  parity.py --fixture                         # random-init model, random crops, no data
  parity.py --ckpt CKPT [--shard SHARD.npz]   # default shard: the last synth shard (val)
  parity.py --ckpt CKPT --set real            # data/real crops, through the RGBA path
  parity.py ... --cli "qemu-arm -cpu cortex-a9 build/arm/vocr_cli_neon"

The checkpoint is exported with train/export.py (fp32 and int8). For every line and mode:
  * preprocessing: the runtime's grey input == cv2.resize(INTER_LINEAR) to the model's
    height plus replicate padding, bit for bit (RGBA input: after synth/common.py's integer
    luma);
  * PyTorch: max |C - reference| <= tol * max(1, max |reference|). f32 is compared with the
    original model (eval mode, batch 1, the crop padded to a multiple of 8 like the runtime
    does); w8f and w8i with a PyTorch forward of the int8 blob (dequantised weights; int16
    activations and integer sums for w8i), so the runtime is tested, not the quantisation;
  * strings: the C runtime's greedy CTC string equals PyTorch's, except on a near tie. A
    string may differ only if, at every frame whose argmax differs, PyTorch's own top-1 /
    top-2 margin is at most twice the measured logit difference (counted as
    text_diff_tie_only; it does occur in w8i, whose int16 activations PyTorch computes in a
    different summation order);
  * exact: the logits are bit-identical to tests/vocr_ref.py, a numpy model of the C
    arithmetic, so the C strings are exactly that model's;
  * tiles: the logits are bit-identical for several tile / head-batch sizes;
  * invert: on the same lines, VOCR_INVERT gives exactly the input and logits of 255 - grey
    inverted before the resize, as synth/common.line_crops does for dark text;
  * arena: vocr_arena_bytes() == train/export.py plan_arena().
Quantisation cost is reported as CER against the labels and agreement with f32.

The C side is build/libvocr.so through ctypes, or any vocr_cli build with --cli
(sanitizer builds, ARM builds under qemu). --report writes the JSON object that CI's
ci/parity_report.py checks.
"""
import argparse
import json
import math
import os
import shlex
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
RT = HERE.parent
ROOT = RT.parent
sys.path[:0] = [str(HERE), str(RT), str(ROOT / "train"), str(ROOT / "synth")]
import vocr_ref  # noqa: E402

import export as E  # noqa: E402

H, MAX_W, STRIDE = 32, 1024, 8
TOL = {"f32": 1e-3, "w8f": 1e-3, "w8i": 5e-3}
TILES = {"lib": [(1, 1), (2, 5), (3, 7), (7, 3), (16, 16), (128, 128)],   # (tile, head) frames
         "cli": [(1, 1), (3, 7), (128, 128)]}
MODE_ID = {"f32": E.MODE_F32, "w8f": E.MODE_W8F, "w8i": E.MODE_W8I}
ARENA_WIDTHS = (8, 100, 256, 640, 1000, 1024)


# ---------------------------------------------------------------- data

def to_gray(rgb):
    r, g, b = (rgb[..., i].astype(np.uint16) for i in range(3))
    return ((77 * r + 150 * g + 29 * b) >> 8).astype(np.uint8)


def load_synth(data, n, shard=None):
    path = Path(shard) if shard else sorted((data / "synth").glob("synth_*.npz"))[-1]
    z = np.load(path)
    labels = json.loads(str(z["labels"]))
    d, offs, sh = z["data"], z["offsets"], z["shapes"]
    out = []
    for i in range(min(n, len(labels))):
        h, w = (int(v) for v in sh[i])
        out.append((f"{path.stem}:{i}", d[offs[i]:offs[i + 1]].reshape(h, w), labels[i]))
    return out


def load_real(data, n):
    from PIL import Image
    recs = [json.loads(x) for x in (data / "real" / "lines.jsonl").read_text().splitlines()]
    out = []
    for r in recs[:n]:
        rgb = np.asarray(Image.open(data / "real" / "crops" / f"{r['id']:05d}.png").convert("RGB"))
        rgba = np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, np.uint8)], axis=2)
        out.append((f"real:{r['id']}:{r['split']}", rgba, r["label"] if r["agree"] else None))
    return out


def fixture_lines(n, seed=0):
    """Random crops with text-like strokes; a third are RGBA, some are tiny."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        h = int(rng.integers(12, 64))
        w = int(rng.integers(4, 1400)) if i % 7 else int(rng.integers(1, 12))
        bg, fg = rng.integers(0, 256, 2)
        g = np.full((h, w), bg, np.float32)
        for _ in range(int(rng.integers(0, 3 * max(1, w // h)))):
            x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
            g[y0:y0 + int(rng.integers(1, h // 3 + 2)), x0:x0 + int(rng.integers(1, h // 2 + 2))] = fg
        g = np.clip(g + rng.normal(0, 8, g.shape), 0, 255).astype(np.uint8)
        if i % 3 == 1:
            rgba = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
            rgba[..., 1] = g
            out.append((f"fixture:{i}", rgba, None))
        else:
            out.append((f"fixture:{i}", g, None))
    return out


def prep(img, height=H, invert=False):
    """train.py's resize + make_batch for one line at the model's height: (padded uint8,
    unpadded width, float input). invert: 255 - grey first, as common.line_crops does."""
    g = to_gray(img) if img.ndim == 3 else img
    if invert:
        g = 255 - g
    h, w = g.shape
    nw = max(8, min(MAX_W, int(round(w * height / h))))
    r = cv2.resize(g, (nw, height), interpolation=cv2.INTER_LINEAR)
    W = (nw + STRIDE - 1) // STRIDE * STRIDE
    padded = np.concatenate([r, np.repeat(r[:, -1:], W - nw, 1)], 1)
    return padded, nw, padded.astype(np.float32) / 127.5 - 1.0


# ---------------------------------------------------------------- PyTorch references

def greedy(best, vocab):
    """vocab excludes the blank: class t > 0 is vocab[t - 1]."""
    out, last = [], 0
    for t in best:
        t = int(t)
        if t and t != last:
            out.append(vocab[t - 1])
        last = t
    return "".join(out)


def _quant16(X):
    import torch
    mx = X.abs().amax(1)
    ok = mx > 0
    inv = torch.where(ok, 32767.0 / torch.where(ok, mx, torch.ones_like(mx)), torch.zeros_like(mx))
    t = X * inv[:, None]
    q = torch.trunc(t + torch.where(torch.signbit(t), -0.5, 0.5)).clamp(-32767, 32767)
    return q, torch.where(ok, mx / 32767.0, torch.zeros_like(mx))


def _matmul(X, op, mode):
    import torch
    if mode == "w8i":
        q, sx = _quant16(X)
        w = torch.from_numpy(op["w"].astype(np.float64))
        acc = torch.zeros(X.shape[0], w.shape[0])
        for k0 in range(0, X.shape[1], vocr_ref.CHUNK):       # int32-safe chunks, as in vocr.c
            acc = acc + (q[:, k0:k0 + vocr_ref.CHUNK].double() @ w[:, k0:k0 + vocr_ref.CHUNK].T).float()
        Y = acc * (torch.from_numpy(op["scale"])[None, :] * sx[:, None])
    else:
        Y = X @ torch.from_numpy(E.dequant(op)).T
    if op["bias"] is not None:
        Y = Y + torch.from_numpy(op["bias"])[None, :]
    return Y


def blob_forward(blob, x, mode):
    """PyTorch float32 forward of a parsed blob; x: H x W float32. Returns T x C logits."""
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(x)[None, None]
    dw_in = None
    for op in blob["ops"]:
        if op["type"] in (E.T_CONV, E.T_DW):
            w = torch.from_numpy(op["w"]).view(op["cout"], 1, op["kh"], op["kw"])
            b = torch.from_numpy(op["bias"]) if op["bias"] is not None else None
            dw_in = t
            t = F.conv2d(t, w, b, (op["sh"], op["sw"]), (op["ph"], op["pw"]),
                         groups=1 if op["type"] == E.T_CONV else op["cout"])
            if op["relu"]:
                t = F.relu(t)
            continue
        _, c, h, w = t.shape
        X = t[0].permute(2, 1, 0).reshape(w, h * c) if op["collapse"] else \
            t[0].permute(1, 2, 0).reshape(h * w, c)
        Y = _matmul(X, op, mode)
        if op["relu"]:
            Y = F.relu(Y)
        ho = 1 if op["collapse"] else h
        if op["res"]:
            Y = Y + dw_in[0].permute(1, 2, 0).reshape(ho * w, -1)
        t = Y.T.reshape(1, -1, 1, w) if op["collapse"] else Y.reshape(ho, w, -1).permute(2, 0, 1)[None]
    return t[0, :, 0, :].T.contiguous().numpy()


def fixture_model(seed=0, width=0.5):
    """Random-init TinyVNOCR whose BatchNorm statistics are measured on fixture crops (so
    activations vary along the line like a trained net's) with random BN scales and shifts,
    so the exporter's folding is exercised."""
    import torch
    import torch.nn as nn

    from common import load_vocab
    from model import TinyVNOCR
    torch.manual_seed(seed)
    vocab = load_vocab(ROOT / "data" / "vocab.txt")
    m = TinyVNOCR(len(vocab) + 1, width=width)
    bns = [mod for mod in m.modules() if isinstance(mod, nn.BatchNorm2d)]
    xs = [prep(img)[2] for _, img, _ in fixture_lines(16, seed + 1)]
    W = max(x.shape[1] for x in xs)
    batch = torch.from_numpy(np.stack([np.pad(x, ((0, 0), (0, W - x.shape[1])), mode="edge")
                                       for x in xs]))[:, None]
    with torch.no_grad():
        for bn in bns:
            bn.reset_running_stats()
            bn.momentum = None
        m.train()
        m.features(batch)
        g = torch.Generator().manual_seed(seed)
        for bn in bns:
            bn.weight.copy_(torch.empty_like(bn.weight).uniform_(0.7, 1.3, generator=g))
            bn.bias.copy_(torch.empty_like(bn.bias).uniform_(-0.3, 0.2, generator=g))
        m.head.weight.mul_(4.0)
    m.eval()
    return m, vocab


# ---------------------------------------------------------------- C backends
# run(items) -> [(text, logits frames x classes, grey input H x padded W, width)]

class LibBackend:
    def __init__(self, blob_path, mode, tile=0, head=0):
        from pyvocr import Vocr
        self.blob_path, self.mode = blob_path, mode
        self.v = Vocr(blob_path, mode=mode, tile=tile, head=head)

    def run(self, items, invert=False):
        out = []
        for img in items:
            text, logits = self.v.run(img, logits=True, invert=invert)
            grey, w = self.v.preprocess(img, invert=invert)
            out.append((text, logits, grey, w))
        return out

    def arena(self, widths, tile, head):
        from pyvocr import Vocr
        return {w: Vocr(self.blob_path, mode=self.mode, tile=tile, head=head, max_width=w).arena_bytes
                for w in widths}


class CliBackend:
    """A vocr_cli binary (possibly under qemu) over a pack file."""

    def __init__(self, cmd, blob_path, mode, tile=0, head=0):
        self.base = shlex.split(cmd) + [str(blob_path), "--mode", mode]
        self.cmd = self.base + ["--crc"] + (["--tile", str(tile)] if tile else []) + \
            (["--head", str(head)] if head else [])

    def run(self, items, invert=False):
        cmd = self.cmd + (["--invert"] if invert else [])
        with tempfile.TemporaryDirectory() as td:
            pack, lg, inp = Path(td) / "in.bin", Path(td) / "logits.bin", Path(td) / "input.bin"
            with open(pack, "wb") as f:
                f.write(struct.pack("<I", len(items)))
                for img in items:
                    f.write(struct.pack("<III", img.shape[0], img.shape[1], 4 if img.ndim == 3 else 1))
                    f.write(np.ascontiguousarray(img).tobytes())
            p = subprocess.run(cmd + ["--pack", str(pack), "--logits", str(lg), "--input-out", str(inp)],
                               capture_output=True, stdin=subprocess.DEVNULL, timeout=7200)
            if p.returncode:
                raise RuntimeError(f"{' '.join(cmd)} failed ({p.returncode}): "
                                   f"{p.stderr.decode(errors='replace')[-3000:]}")
            texts = p.stdout.decode("utf-8").split("\n")[:len(items)]
            raw, rin = lg.read_bytes(), inp.read_bytes()
        out, off, oi = [], 0, 0
        for text in texts:
            frames, classes = struct.unpack_from("<II", raw, off)
            a = np.frombuffer(raw, np.float32, frames * classes, off + 8).reshape(frames, classes)
            off += 8 + 4 * frames * classes
            h, pw, w = struct.unpack_from("<III", rin, oi)
            g = np.frombuffer(rin, np.uint8, h * pw, oi + 12).reshape(h, pw)
            oi += 12 + h * pw
            out.append((text, a.copy(), g.copy(), w))
        return out

    def arena(self, widths, tile, head):
        cmd = self.base + ["--tile", str(tile), "--head", str(head), "--arena", ",".join(map(str, widths))]
        p = subprocess.run(cmd, capture_output=True, stdin=subprocess.DEVNULL, timeout=600, check=True)
        got = {}
        for line in p.stdout.decode().splitlines():
            if line.startswith("max_width"):
                _, w, _, n = line.split()
                got[int(w)] = int(n)
        return got


# ---------------------------------------------------------------- checks

def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


_W = {}


def make_backend(cfg, mode, tile=0, head=0):
    path = cfg["blobs"]["f32" if mode == "f32" else "w8"]
    if cfg["cli"]:
        return CliBackend(cfg["cli"], path, mode, tile, head)
    return LibBackend(path, mode, tile, head)


def worker_init(cfg):
    import torch
    torch.set_num_threads(int(cfg.get("threads", 1)))
    torch.set_flush_denormal(False)      # keep IEEE denormals, as the C code and numpy do
    if cfg["fixture"]:
        model, vocab = fixture_model(cfg["seed"], cfg["width"])
    else:
        model, vocab, _ = E.load_checkpoint(cfg["ckpt"])
    model.eval()
    blobs = {k: E.read_blob(p) for k, p in cfg["blobs"].items()}
    kind = "cli" if cfg["cli"] else "lib"
    _W.update(cfg=cfg, vocab=vocab, model=model, blobs=blobs, height=blobs["f32"]["height"],
              main={m: make_backend(cfg, m) for m in cfg["modes"]},
              tiles={(m, t, h): make_backend(cfg, m, t, h) for m in cfg["modes"] for t, h in TILES[kind]})


def reference(mode, x):
    import torch
    blob = _W["blobs"]["f32" if mode == "f32" else "w8"]
    with torch.no_grad():
        if mode == "f32":
            ref = _W["model"](torch.from_numpy(x)[None, None])[:, 0].numpy()
        else:
            ref = blob_forward(blob, x, mode)
    return ref, vocr_ref.forward(blob, x, mode)


def check_batch(batch):
    cfg, vocab = _W["cfg"], _W["vocab"]
    imgs = [b[2] for b in batch]
    height = _W["height"]
    preps = [prep(img, height) for img in imgs]
    recs = [{"i": i, "id": ident, "label": label, "w": nw, "frames": int(x.shape[1] // STRIDE)}
            for (i, ident, _, label), (_, nw, x) in zip(batch, preps)]
    tile_idx = [k for k, b in enumerate(batch) if b[0] < cfg["tile_lines"]]
    for mode in cfg["modes"]:
        outs = _W["main"][mode].run(imgs)
        tiled = {key: be.run([imgs[k] for k in tile_idx]) if tile_idx else []
                 for key, be in _W["tiles"].items() if key[0] == mode}
        inverted = _W["main"][mode].run([imgs[k] for k in tile_idx], invert=True) if tile_idx else []
        tol = TOL[mode] * cfg["tol_scale"]
        for k, (rec, (padded, nw, x), (c_text, c_log, c_in, c_w)) in enumerate(zip(recs, preps, outs)):
            m = {"text": c_text, "prep_ok": bool(c_w == nw and np.array_equal(c_in, padded))}
            ref, exact = reference(mode, x)
            if c_log.shape != ref.shape:
                m.update(ok=False, exact=False, shape=list(c_log.shape))
                rec[mode] = m
                continue
            scale = max(1.0, float(np.abs(ref).max()))
            diff = float(np.abs(c_log - ref).max())
            rb, cb = ref.argmax(1), c_log.argmax(1)
            top2 = np.sort(ref, 1)[:, -2:]
            margin = top2[:, 1] - top2[:, 0]
            mism = np.flatnonzero(rb != cb)
            m.update(diff=diff, scale=scale, ok=diff <= tol * scale, exact=bool(np.array_equal(c_log, exact)),
                     ref_text=greedy(rb, vocab), self_ok=greedy(cb, vocab) == c_text,
                     argmax_mismatch=int(len(mism)),
                     tie_only=bool(np.all(margin[mism] <= 2 * diff + 1e-12)),
                     min_margin=float(margin.min()) if len(margin) else 0.0)
            m["same_text"] = m["ref_text"] == c_text
            if k in tile_idx:
                j = tile_idx.index(k)
                m["tiles_ok"] = all(t[j][0] == c_text and np.array_equal(t[j][1], c_log)
                                    for t in tiled.values())
                # VOCR_INVERT == the C arithmetic on 255 - grey, inverted before the resize
                i_padded, i_nw, i_x = prep(imgs[k], height, invert=True)
                i_text, i_log, i_in, i_w = inverted[j]
                i_ref = vocr_ref.forward(_W["blobs"]["f32" if mode == "f32" else "w8"], i_x, mode)
                m["invert_ok"] = bool(i_w == i_nw and np.array_equal(i_in, i_padded) and
                                      i_log.shape == i_ref.shape and np.array_equal(i_log, i_ref) and
                                      greedy(i_log.argmax(1), vocab) == i_text)
            rec[mode] = m
    return recs


def arena_checks(cfg):
    """vocr_arena_bytes() against plan_arena() over widths, tiles and modes."""
    ok, table = True, []
    kind = "cli" if cfg["cli"] else "lib"
    for mode in cfg["modes"]:
        key = "f32" if mode == "f32" else "w8"
        blob = E.read_blob(cfg["blobs"][key])
        for t, h in [(0, 0)] + TILES[kind]:
            got = make_backend(cfg, mode).arena(ARENA_WIDTHS, t, h)
            for w in ARENA_WIDTHS:
                want = E.plan_arena(blob, w, t, h, MODE_ID[mode])
                ok &= got.get(w) == want
                if (t, h) == (0, 0) and w in (256, 640, 1024):
                    table.append({"mode": mode, "max_width": w, "arena": got.get(w), "plan": want})
    return bool(ok), table


def summarize(recs, modes):
    s = {"lines": len(recs)}
    labelled = [r for r in recs if r["label"]]
    s["labelled"] = len(labelled)
    for mode in modes:
        ms = [r[mode] for r in recs]
        tl = [m["tiles_ok"] for m in ms if "tiles_ok" in m]
        il = [m["invert_ok"] for m in ms if "invert_ok" in m]
        d = {"prep_ok": sum(m["prep_ok"] for m in ms),
             "exact": sum(m.get("exact", False) for m in ms),
             "logits_ok": sum(m.get("ok", False) for m in ms),
             "max_norm_diff": max((m.get("diff", math.inf) / m.get("scale", 1.0)) for m in ms),
             "max_abs_diff": max(m.get("diff", math.inf) for m in ms),
             "same_text": sum(m.get("same_text", False) for m in ms),
             "text_diff_tie_only": sum((not m.get("same_text", False)) and m.get("tie_only", False)
                                       for m in ms),
             "self_ok": sum(m.get("self_ok", False) for m in ms),
             "argmax_mismatch_frames": sum(m.get("argmax_mismatch", 0) for m in ms),
             "tiles_checked": len(tl), "tiles_ok": sum(tl),
             "invert_checked": len(il), "invert_ok": sum(il)}
        if labelled:
            chars = sum(len(r["label"]) for r in labelled)
            d["cer"] = sum(edit_distance(r[mode]["text"], r["label"]) for r in labelled) / chars
            d["cer_pytorch"] = sum(edit_distance(r[mode].get("ref_text", ""), r["label"])
                                   for r in labelled) / chars
            d["exact_lines"] = sum(r[mode]["text"] == r["label"] for r in labelled) / len(labelled)
        if mode != "f32" and "f32" in modes:
            d["same_text_as_f32"] = sum(r[mode]["text"] == r["f32"]["text"] for r in recs)
            if labelled:
                d["cer_minus_f32"] = d["cer"] - s["f32"]["cer"]
        s[mode] = d
    return s


def failures(s, modes):
    bad, n = [], s["lines"]
    for mode in modes:
        d = s[mode]
        for key, what in (("prep_ok", "preprocessing"), ("exact", "bit-exact logits"),
                          ("logits_ok", "logits tolerance"), ("self_ok", "CTC decode")):
            if d[key] != n:
                bad.append(f"{mode} {what} ({d[key]}/{n})")
        if d["same_text"] + d["text_diff_tie_only"] != n:
            bad.append(f"{mode} text ({d['same_text']}/{n} identical, {d['text_diff_tie_only']} near ties)")
        if d["tiles_ok"] != d["tiles_checked"]:
            bad.append(f"{mode} tile invariance ({d['tiles_ok']}/{d['tiles_checked']})")
        if d["invert_ok"] != d["invert_checked"]:
            bad.append(f"{mode} VOCR_INVERT ({d['invert_ok']}/{d['invert_checked']})")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", help="checkpoint (train.py format)")
    ap.add_argument("--fixture", action="store_true", help="random-init model and crops, no data")
    ap.add_argument("--set", default="synth-val", choices=["synth-val", "real"])
    ap.add_argument("--shard", help="a render.py shard (default: the last synth shard, the val shard)")
    ap.add_argument("--data", type=Path, default=Path(os.environ.get("VOCR_DATA", ROOT / "data")))
    ap.add_argument("--n", type=int, default=10**9, help="at most this many lines")
    ap.add_argument("--modes", default="f32,w8f,w8i")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cli", default="", help="vocr_cli command line instead of libvocr.so")
    ap.add_argument("--tile-lines", type=int, default=40, help="lines checked at every tile size")
    ap.add_argument("--tol-scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=float, default=0.5, help="fixture model width multiplier")
    ap.add_argument("--report", type=Path, help="summary JSON (CI's VOCR_REPORT contract)")
    ap.add_argument("--out", type=Path, help="summary and per-line results as JSON")
    args = ap.parse_args()
    if not args.fixture and not args.ckpt:
        ap.error("--ckpt or --fixture is required")
    modes = args.modes.split(",")
    t0 = time.time()
    tmp = tempfile.TemporaryDirectory()
    if args.fixture:
        model, vocab = fixture_model(args.seed, args.width)
        height, sha = H, b"fixture"
    else:
        import hashlib
        model, vocab, height = E.load_checkpoint(args.ckpt)
        sha = hashlib.sha256(Path(args.ckpt).read_bytes()).digest()
    ops = E.model_ops(model, height)
    blobs = {}
    for key, int8 in (("f32", False), ("w8", True)):
        data, _ = E.build_blob(ops, vocab, height, int8, MAX_W, sha)
        blobs[key] = str(Path(tmp.name) / f"model_{key}.vocr")
        Path(blobs[key]).write_bytes(data)
    if args.shard:
        lines = load_synth(args.data, args.n, args.shard)
    elif args.fixture:
        lines = fixture_lines(min(args.n, 40), args.seed)
    elif args.set == "real":
        lines = load_real(args.data, args.n)
    else:
        lines = load_synth(args.data, args.n)
    cfg = {"fixture": args.fixture, "seed": args.seed, "width": args.width, "ckpt": args.ckpt,
           "blobs": blobs, "modes": modes, "cli": args.cli, "tile_lines": args.tile_lines,
           "tol_scale": args.tol_scale}
    arena_ok, arena = arena_checks(cfg)
    items = [(i, ident, img, label) for i, (ident, img, label) in enumerate(lines)]
    batches = [items[i:i + 32] for i in range(0, len(items), 32)]
    recs = []
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(args.workers, worker_init, (cfg,)) as pool:
            for k, out in enumerate(pool.imap(check_batch, batches)):
                recs += out
                if k % 25 == 0:
                    print(f"{len(recs)}/{len(items)} lines, {time.time() - t0:.0f}s", flush=True)
    else:
        worker_init({**cfg, "threads": int(os.environ.get("OMP_NUM_THREADS", "1"))})
        for b in batches:
            recs += check_batch(b)
    s = summarize(recs, modes)
    bad = failures(s, modes) + ([] if arena_ok else ["arena plan"])
    s.update(ckpt=args.ckpt or "fixture", shard=args.shard, set=None if args.shard else
             ("fixture" if args.fixture else args.set), cli=args.cli, arena_ok=arena_ok, arena=arena,
             blob_bytes={k: Path(p).stat().st_size for k, p in blobs.items()}, tol=TOL,
             seconds=round(time.time() - t0, 1), failed=bad)
    s["pass"] = not bad
    print(json.dumps(s, indent=1, ensure_ascii=False))
    if args.report:
        args.report.write_text(json.dumps(s, ensure_ascii=False))
    if args.out:
        args.out.write_text(json.dumps({"summary": s, "lines": recs}, ensure_ascii=False))
    print("PARITY_OK" if not bad else f"PARITY_FAIL {bad}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
