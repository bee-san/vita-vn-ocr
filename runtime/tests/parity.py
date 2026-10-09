#!/usr/bin/env python3
"""Parity of the C runtime against PyTorch.

  parity.py --fixture                                   # CI: random-init model, no data
  parity.py --ckpt models/v1/ckpt_best.pt --set synth-val [--n 10000] [--workers 5]
  parity.py --ckpt ... --set real                       # data/real/crops (RGBA path)
  parity.py --ckpt ... --set synth-val --n 200 --cli "qemu-arm -cpu cortex-a9 build/arm/vocr_cli_neon"

For every line and mode it checks:
  * preprocessing: vocr_preprocess() == cv2.resize(INTER_LINEAR) + replicate padding,
    bit for bit (and for RGBA input, after the integer luma of synth/common.py);
  * logits: max |C - reference| <= tol * max(1, max |reference|) per line;
  * text: the C runtime's greedy CTC string == the reference's.
References: f32 is the original PyTorch model (eval mode, batch 1, the crop padded to
its own multiple of 8, as the runtime sees it). w8f and w8i are PyTorch re-implementations
of the int8 blob (dequantised weights; plus per-position int16 activations with int32
sums for w8i), so they test the runtime, not the quantisation. Quantisation cost is
reported separately as CER against the labels and agreement with the f32 model.
It also checks that the logits are bit-identical for several tile / head-batch sizes
and that vocr_arena_bytes() == train/export.py plan_arena().

--cli runs the C side through a vocr_cli binary (sanitizer builds, ARM under qemu)
instead of build/libvocr.so; --save-logits keeps its logits to compare backends
bit for bit with --compare-logits.
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
sys.path[:0] = [str(RT), str(ROOT / "train"), str(ROOT / "synth")]
import export as E  # noqa: E402
from pyvocr import Vocr  # noqa: E402

H, MAX_W, STRIDE = 32, 1024, 8
TOL = {"f32": 1e-3, "w8f": 1e-3, "w8i": 5e-3}
TILES = [(1, 1), (2, 5), (3, 7), (7, 3), (16, 16), (128, 128)]   # (tile, head) frames


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
    """Random crops with text-like strokes; about a third are RGBA."""
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


def grey_of(img):
    return to_gray(img) if img.ndim == 3 else img


def prep(img):
    """train.py's resize32 + make_batch for one line: (uint8 resized, padded float input)."""
    g = grey_of(img)
    h, w = g.shape
    nw = max(8, min(MAX_W, int(round(w * H / h))))
    r = cv2.resize(g, (nw, H), interpolation=cv2.INTER_LINEAR)
    W = (nw + STRIDE - 1) // STRIDE * STRIDE
    x = np.empty((H, W), np.float32)
    x[:, :nw] = r
    x[:, nw:] = r[:, -1:]
    return r, x / 127.5 - 1.0


# ---------------------------------------------------------------- references

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
    """Per-row int16 quantisation exactly as vocr.c quant16()."""
    import torch
    mx = X.abs().amax(1)
    inv = torch.where(mx > 0, 32767.0 / torch.where(mx > 0, mx, torch.ones_like(mx)),
                      torch.zeros_like(mx))
    t = X * inv[:, None]
    q = torch.trunc(t + torch.where(torch.signbit(t), -0.5, 0.5)).clamp(-32767, 32767)
    return q, torch.where(mx > 0, mx / 32767.0, torch.zeros_like(mx))


def _matmul(X, op, mode):
    import torch
    if mode == "w8i":
        q, sx = _quant16(X)
        acc = (q.double() @ torch.from_numpy(op["w"].astype(np.float64)).T).float()
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
        if op["collapse"]:
            X = t[0].permute(2, 1, 0).reshape(w, h * c)
        else:
            X = t[0].permute(1, 2, 0).reshape(h * w, c)
        Y = _matmul(X, op, mode)
        if op["relu"]:
            Y = F.relu(Y)
        ho = 1 if op["collapse"] else h
        if op["res"]:
            Y = Y + dw_in[0].permute(1, 2, 0).reshape(ho * w, -1)
        if op["collapse"]:
            t = Y.T.reshape(1, -1, 1, w)
        else:
            t = Y.reshape(ho, w, -1).permute(2, 0, 1)[None]
    return t[0, :, 0, :].T.contiguous().numpy()


class Reference:
    def __init__(self, model, blobs):
        self.model, self.blobs = model, blobs

    def logits(self, x, mode):
        import torch
        with torch.no_grad():
            if mode == "f32":
                return self.model(torch.from_numpy(x)[None, None])[:, 0].numpy()
            return blob_forward(self.blobs["w8"], x, mode)


# ---------------------------------------------------------------- C backends

class LibBackend:
    def __init__(self, blob_path, mode, tile=0, head=0):
        self.v = Vocr(blob_path, mode=mode, tile=tile, head=head)

    def run(self, items):
        return [self.v.run(img, logits=True) for img in items]


class CliBackend:
    """Runs a vocr_cli binary (possibly under qemu) over a pack file."""

    def __init__(self, cmd, blob_path, mode, tile=0, head=0, save=None):
        self.cmd = shlex.split(cmd) + [str(blob_path), "--mode", mode, "--crc"]
        if tile:
            self.cmd += ["--tile", str(tile)]
        if head:
            self.cmd += ["--head", str(head)]
        self.save = save

    def run(self, items):
        with tempfile.TemporaryDirectory() as td:
            pack, lg = Path(td) / "in.bin", Path(td) / "logits.bin"
            with open(pack, "wb") as f:
                f.write(struct.pack("<I", len(items)))
                for img in items:
                    ch = 4 if img.ndim == 3 else 1
                    f.write(struct.pack("<III", img.shape[0], img.shape[1], ch))
                    f.write(np.ascontiguousarray(img).tobytes())
            p = subprocess.run(self.cmd + ["--pack", str(pack), "--logits", str(lg)],
                               capture_output=True, stdin=subprocess.DEVNULL, timeout=3600)
            if p.returncode:
                raise RuntimeError(f"{self.cmd[0]} failed ({p.returncode}): {p.stderr.decode()[-2000:]}")
            texts = p.stdout.decode("utf-8").split("\n")[:len(items)]
            raw = lg.read_bytes()
        out, off = [], 0
        for text in texts:
            frames, classes = struct.unpack_from("<II", raw, off)
            off += 8
            a = np.frombuffer(raw, np.float32, frames * classes, off).reshape(frames, classes)
            off += 4 * frames * classes
            out.append((text, a.copy()))
        if self.save:
            np.savez_compressed(self.save, *[a for _, a in out])
        return out


# ---------------------------------------------------------------- checks

def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def fixture_model(seed=0, width=0.5):
    """Random-init TinyVNOCR whose BatchNorm statistics are calibrated on fixture crops
    (so activations vary along the line like a trained net's) and then perturbed, so
    folding is exercised with non-trivial scales and shifts."""
    import torch
    import torch.nn as nn
    from common import load_vocab
    from model import TinyVNOCR
    torch.manual_seed(seed)
    vocab = load_vocab(ROOT / "data" / "vocab.txt")
    m = TinyVNOCR(len(vocab) + 1, width=width)
    bns = [mod for mod in m.modules() if isinstance(mod, nn.BatchNorm2d)]
    xs = [prep(img)[1] for _, img, _ in fixture_lines(16, seed + 1)]
    W = max(x.shape[1] for x in xs)
    batch = torch.from_numpy(np.stack([np.pad(x, ((0, 0), (0, W - x.shape[1])), mode="edge")
                                       for x in xs]))[:, None]
    with torch.no_grad():
        for bn in bns:
            bn.reset_running_stats()
            bn.momentum = None                     # cumulative average over the batch
        m.train()
        m.features(batch)
        g = torch.Generator().manual_seed(seed)
        for bn in bns:
            bn.weight.copy_(torch.empty_like(bn.weight).uniform_(0.7, 1.3, generator=g))
            bn.bias.copy_(torch.empty_like(bn.bias).uniform_(-0.3, 0.2, generator=g))
        m.head.weight.mul_(4.0)
    m.eval()
    return m, vocab


_W = {}


def worker_init(cfg):
    import torch
    torch.set_num_threads(int(cfg.get("threads", 1)))
    torch.set_flush_denormal(False)
    if cfg["fixture"]:
        model, vocab = fixture_model(cfg["seed"], cfg["width"])
    else:
        model, vocab, _ = E.load_checkpoint(cfg["ckpt"])
    blobs = {k: E.read_blob(p) for k, p in cfg["blobs"].items()}
    ref = Reference(model, blobs)
    backends, tiles = {}, {}
    for mode in cfg["modes"]:
        path = cfg["blobs"]["f32" if mode == "f32" else "w8"]
        if cfg["cli"]:
            backends[mode] = CliBackend(cfg["cli"], path, mode)
        else:
            backends[mode] = LibBackend(path, mode)
            for t, h in TILES:
                tiles[mode, t, h] = LibBackend(path, mode, t, h)
    _W.update(cfg=cfg, vocab=vocab, ref=ref, backends=backends, tiles=tiles,
              prep=LibBackend(cfg["blobs"]["f32" if "f32" in cfg["modes"] else "w8"],
                              cfg["modes"][0]).v)


def check_batch(batch):
    cfg, vocab, ref = _W["cfg"], _W["vocab"], _W["ref"]
    recs = []
    xs, preps = [], []
    for idx, ident, img, label in batch:
        r, x = prep(img)
        c_in, c_w = _W["prep"].preprocess(img)
        padded = np.concatenate([r, np.repeat(r[:, -1:], x.shape[1] - r.shape[1], 1)], 1)
        preps.append(bool(c_w == r.shape[1] and np.array_equal(c_in, padded)))
        xs.append(x)
        recs.append({"i": idx, "id": ident, "label": label, "w": int(r.shape[1]),
                     "frames": int(x.shape[1] // STRIDE), "prep_ok": preps[-1]})
    imgs = [b[2] for b in batch]
    for mode in cfg["modes"]:
        outs = _W["backends"][mode].run(imgs)
        tol = TOL[mode] * cfg["tol_scale"]
        for rec, x, img, (c_text, c_log) in zip(recs, xs, imgs, outs):
            rl = ref.logits(x, mode)
            m = {"text": c_text}
            if c_log.shape != rl.shape:
                m.update(ok=False, shape=list(c_log.shape))
                rec[mode] = m
                continue
            scale = max(1.0, float(np.abs(rl).max()))
            diff = float(np.abs(c_log - rl).max())
            rb, cb = rl.argmax(1), c_log.argmax(1)
            top2 = np.sort(rl, 1)[:, -2:]
            margin = top2[:, 1] - top2[:, 0]
            mism = np.flatnonzero(rb != cb)
            m.update(diff=diff, scale=scale, ok=diff <= tol * scale,
                     ref_text=greedy(rb, vocab), self_ok=greedy(cb, vocab) == c_text,
                     argmax_mismatch=int(len(mism)),
                     tie_only=bool(np.all(margin[mism] <= 2 * diff + 1e-12)),
                     min_margin=float(margin.min()) if len(margin) else 0.0)
            m["same_text"] = m["ref_text"] == c_text
            if rec["i"] < cfg["tile_lines"] and not cfg["cli"]:
                same = True
                for (md, t, h), be in _W["tiles"].items():
                    if md == mode:
                        tt, tl = be.v.run(img, logits=True)
                        same &= tt == c_text and np.array_equal(tl, c_log)
                m["tiles_ok"] = bool(same)
            rec[mode] = m
    return recs


def arena_checks(blobs, modes):
    """vocr_arena_bytes() against plan_arena() over widths, tiles and modes."""
    out, ok = [], True
    for mode in modes:
        key = "f32" if mode == "f32" else "w8"
        b = E.read_blob(blobs[key])
        for w in (8, 100, 256, 640, 1000, 1024):
            for t, h in [(0, 0)] + TILES:
                c = Vocr(blobs[key], mode=mode, tile=t, head=h, max_width=w).arena_bytes
                p = E.plan_arena(b, w, t, h, E.MODE_W8F if mode == "w8f" else
                                 E.MODE_W8I if mode == "w8i" else E.MODE_F32)
                ok &= c == p
                if (t, h) == (0, 0) and w in (256, 640, 1024):
                    out.append({"mode": mode, "max_width": w, "arena": c, "plan": p})
    return ok, out


def summarize(recs, modes, has_cli):
    s = {"lines": len(recs), "prep_ok": sum(r["prep_ok"] for r in recs)}
    labelled = [r for r in recs if r["label"]]
    s["labelled"] = len(labelled)
    for mode in modes:
        ms = [r[mode] for r in recs]
        d = {"logits_ok": sum(m.get("ok", False) for m in ms),
             "max_norm_diff": max((m.get("diff", math.inf) / m.get("scale", 1)) for m in ms),
             "max_abs_diff": max(m.get("diff", math.inf) for m in ms),
             "same_text": sum(m.get("same_text", False) for m in ms),
             "self_ok": sum(m.get("self_ok", False) for m in ms),
             "text_diff_tie_only": sum((not m.get("same_text", False)) and m.get("tie_only", False)
                                       for m in ms),
             "argmax_mismatch_frames": sum(m.get("argmax_mismatch", 0) for m in ms)}
        if not has_cli:
            tl = [m["tiles_ok"] for m in ms if "tiles_ok" in m]
            d["tiles_checked"], d["tiles_ok"] = len(tl), sum(tl)
        if labelled:
            chars = sum(len(r["label"]) for r in labelled)
            d["cer_c"] = sum(edit_distance(r[mode]["text"], r["label"]) for r in labelled) / chars
            d["cer_ref"] = sum(edit_distance(r[mode].get("ref_text", ""), r["label"])
                               for r in labelled) / chars
            d["exact_c"] = sum(r[mode]["text"] == r["label"] for r in labelled) / len(labelled)
        if mode != "f32" and "f32" in modes:
            d["same_as_f32_c"] = sum(r[mode]["text"] == r["f32"]["text"] for r in recs)
        s[mode] = d
    return s


def failed(s, modes, has_cli, tie_ok):
    bad = []
    if s["prep_ok"] != s["lines"]:
        bad.append("preprocessing")
    for mode in modes:
        d = s[mode]
        if d["logits_ok"] != s["lines"]:
            bad.append(f"{mode} logits")
        if d["self_ok"] != s["lines"]:
            bad.append(f"{mode} CTC")
        if d["same_text"] + (d["text_diff_tie_only"] if tie_ok else 0) != s["lines"]:
            bad.append(f"{mode} text")
        if not has_cli and d["tiles_ok"] != d["tiles_checked"]:
            bad.append(f"{mode} tile invariance")
    return bad


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt")
    ap.add_argument("--fixture", action="store_true", help="random-init model and crops (CI)")
    ap.add_argument("--set", default="synth-val", choices=["synth-val", "real", "fixture"])
    ap.add_argument("--shard", help="synth shard (default: the last, train.py's val shard)")
    ap.add_argument("--data", type=Path, default=Path(os.environ.get("VOCR_DATA", ROOT / "data")))
    ap.add_argument("--n", type=int, default=10**9)
    ap.add_argument("--modes", default="f32,w8f,w8i")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--cli", default="", help="vocr_cli command instead of libvocr.so")
    ap.add_argument("--tile-lines", type=int, default=50, help="lines checked at every tile size")
    ap.add_argument("--tol-scale", type=float, default=1.0)
    ap.add_argument("--allow-ties", action="store_true",
                    help="accept text differences caused only by near-tied logits")
    ap.add_argument("--blob-dir", type=Path, help="keep the exported blobs here")
    ap.add_argument("--save-logits", type=Path, help="npz of the C logits (--cli, one mode)")
    ap.add_argument("--compare-logits", nargs=2, type=Path, metavar=("A", "B"),
                    help="exit 0 iff two --save-logits files are bit-identical")
    ap.add_argument("--out", type=Path, help="write the summary and per-line results as JSON")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--width", type=float, default=0.5, help="fixture width multiplier")
    args = ap.parse_args()

    if args.compare_logits:
        a, b = (np.load(p) for p in args.compare_logits)
        same = a.files == b.files and all(np.array_equal(a[k], b[k]) for k in a.files)
        print(json.dumps({"bit_identical": bool(same), "lines": len(a.files)}))
        sys.exit(0 if same else 1)
    if not args.fixture and not args.ckpt:
        ap.error("--ckpt or --fixture is required")
    modes = args.modes.split(",")
    t0 = time.time()
    tmp = tempfile.TemporaryDirectory()
    bdir = args.blob_dir or Path(tmp.name)
    bdir.mkdir(parents=True, exist_ok=True)
    if args.fixture:
        model, vocab = fixture_model(args.seed, args.width)
        height, sha = H, b"fixture"
    else:
        model, vocab, height = E.load_checkpoint(args.ckpt)
        import hashlib
        sha = hashlib.sha256(Path(args.ckpt).read_bytes()).digest()
    ops = E.model_ops(model, height)
    blobs = {}
    for key, int8 in (("f32", False), ("w8", True)):
        data, meta = E.build_blob(ops, vocab, height, int8, MAX_W, sha)
        blobs[key] = str(bdir / f"model_{key}.vocr")
        Path(blobs[key]).write_bytes(data)
    if args.save_logits and len(modes) != 1:
        ap.error("--save-logits needs exactly one mode")
    if not args.cli:
        arena_ok, arena = arena_checks(blobs, modes)
    else:
        arena_ok, arena = True, []
    if args.fixture or args.set == "fixture":
        lines = fixture_lines(min(args.n, 40), args.seed)
    elif args.set == "real":
        lines = load_real(args.data, args.n)
    else:
        lines = load_synth(args.data, args.n, args.shard)
    cfg = {"fixture": args.fixture, "seed": args.seed, "width": args.width, "ckpt": args.ckpt,
           "blobs": blobs, "modes": modes, "cli": args.cli, "tile_lines": args.tile_lines,
           "tol_scale": args.tol_scale}
    items = [(i, ident, img, label) for i, (ident, img, label) in enumerate(lines)]
    bs = 32
    batches = [items[i:i + bs] for i in range(0, len(items), bs)]
    recs = []
    if args.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(args.workers, worker_init, (cfg,)) as pool:
            for k, out in enumerate(pool.imap(check_batch, batches)):
                recs += out
                if k % 20 == 0:
                    print(f"{len(recs)}/{len(items)} lines, {time.time() - t0:.0f}s", flush=True)
    else:
        if args.save_logits:
            cfg["save"] = str(args.save_logits)
        worker_init({**cfg, "threads": int(os.environ.get("OMP_NUM_THREADS", "1"))})
        if args.save_logits:
            be = _W["backends"][modes[0]]
            be.save = None
            outs = []
            for b in batches:
                outs += be.run([x[2] for x in b])
            np.savez_compressed(args.save_logits, *[a for _, a in outs])
            print(json.dumps({"saved": str(args.save_logits), "lines": len(outs)}))
            return
        for b in batches:
            recs += check_batch(b)
    s = summarize(recs, modes, bool(args.cli))
    s.update(set="fixture" if args.fixture else args.set, ckpt=args.ckpt, cli=args.cli,
             arena_ok=arena_ok, arena=arena, seconds=round(time.time() - t0, 1),
             blob_bytes={k: Path(p).stat().st_size for k, p in blobs.items()}, tol=TOL)
    bad = failed(s, modes, bool(args.cli), args.allow_ties or args.fixture)
    if not arena_ok:
        bad.append("arena plan")
    s["failed"] = bad
    print(json.dumps(s, indent=1, ensure_ascii=False))
    if args.out:
        args.out.write_text(json.dumps({"summary": s, "lines": recs}, ensure_ascii=False))
    print("PARITY_OK" if not bad else f"PARITY_FAIL {bad}", flush=True)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
