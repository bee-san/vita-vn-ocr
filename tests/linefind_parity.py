#!/usr/bin/env python3
"""Parity check: runtime/vocr_lines.c against its Python twin in synth/common.py.

  linefind_parity.py --bin build/vocr_lines_test [--cases 400] [--seed 1]
                     [--regions eval/linefind/manifests/research_b.jsonl ...]

Renders procedural test regions (text-like strokes of both polarities, outlines,
noise, gradients, frames, tiny and full-screen sizes, grey and RGBA, padded
strides), optionally adds real regions from line-finder manifests (local
screenshots only), runs both finders through the C driver and the twin, and exits
non-zero on the first box, polarity flag or return code that differs.
"""
import argparse
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "synth"))
from common import find_lines_ex  # noqa: E402


def text_region(rng, w, h):
    """Rows of block glyphs with random size, spacing, polarity, outline and clutter."""
    bg = rng.integers(0, 256, 3)
    img = np.empty((h, w, 3), np.int32)
    img[:] = bg
    if rng.random() < 0.5:                                    # gradient background
        t = np.linspace(0, 1, w)[None, :, None]
        img = (img * (1 - t) + rng.integers(0, 256, 3) * t).astype(np.int32)
    fg = rng.integers(0, 256, 3)
    oc = rng.integers(0, 256, 3)
    fs = int(rng.integers(6, 40))
    pitch = fs + int(rng.integers(2, fs + 4))
    y = int(rng.integers(0, max(1, pitch)))
    outline = rng.random() < 0.3
    while y + fs < h:
        x = int(rng.integers(0, max(1, w // 6)))
        if x + fs >= w:
            break
        end = int(rng.integers(x + fs, w + 1))
        while x + fs <= end:
            gw = int(rng.integers(max(2, fs // 3), fs + 1))
            sw = max(1, fs // int(rng.integers(5, 12)))
            gl = np.zeros((fs, gw), bool)
            for _ in range(int(rng.integers(1, 4))):
                if rng.random() < 0.5:
                    r0 = int(rng.integers(0, fs))
                    gl[r0:r0 + sw, :] = True
                else:
                    c0 = int(rng.integers(0, gw))
                    gl[:, c0:c0 + sw] = True
            if outline:
                big = np.zeros((fs + 2, gw + 2), bool)
                for dy in range(3):
                    for dx in range(3):
                        big[dy:dy + fs, dx:dx + gw] |= gl
                ys, xs = np.nonzero(big)
                ok = (y - 1 + ys >= 0) & (y - 1 + ys < h) & (x - 1 + xs < w) & (x - 1 + xs >= 0)
                img[y - 1 + ys[ok], x - 1 + xs[ok]] = oc
            ys, xs = np.nonzero(gl)
            ok = (x + xs) < w
            img[y + ys[ok], x + xs[ok]] = fg
            x += gw + int(rng.integers(0, fs))
        y += pitch
    if rng.random() < 0.3:                                    # frame border
        img[int(rng.integers(0, h)), :] = rng.integers(0, 256, 3)
        img[:, int(rng.integers(0, w))] = rng.integers(0, 256, 3)
    if rng.random() < 0.5:
        img = img + rng.normal(0, rng.uniform(0, 25), img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_cases(seed, n):
    rng = np.random.default_rng(seed)
    cases = []
    for i in range(n):
        kind = i % 8
        if kind == 0:
            w, h = int(rng.integers(1, 12)), int(rng.integers(1, 12))
        elif kind == 1:
            w, h = 960, 544
        elif kind == 2:
            w, h = 480, 272
        else:
            w, h = int(rng.integers(16, 961)), int(rng.integers(8, 545))
        if kind == 3:
            img = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)            # pure noise
        elif kind == 4:
            img = np.full((h, w, 3), rng.integers(0, 256), np.uint8)          # flat
        else:
            img = text_region(rng, w, h)
        if rng.random() < 0.5:
            img = np.concatenate([img, rng.integers(0, 256, (h, w, 1), dtype=np.uint8)], axis=2)
        elif rng.random() < 0.5:
            img = (img[..., 0].astype(np.uint16) * 77 + img[..., 1] * 150 + img[..., 2] * 29 >> 8).astype(np.uint8)
        cases.append(img)
    return cases


def write_batch(path, cases, rng):
    with open(path, "wb") as f:
        for img, finder in cases:
            h, w = img.shape[:2]
            if img.ndim == 2:
                bpp, px = 1, img
            elif img.shape[2] == 4:
                bpp, px = 4, img
            else:
                bpp, px = 4, np.concatenate([img, np.full((h, w, 1), 255, np.uint8)], axis=2)
            row = np.ascontiguousarray(px).reshape(h, w * bpp)
            extra = int(rng.integers(0, 9)) if rng.random() < 0.3 else 0
            stride = w * bpp + extra
            buf = np.zeros((h, stride), np.uint8)
            buf[:, :w * bpp] = row
            if extra:
                buf[:, w * bpp:] = 0xA5                                      # junk past the row
            f.write(struct.pack("<5I", w, h, bpp, 0 if finder == "grad" else 1, stride))
            f.write(buf.tobytes())


def expected(img, finder):
    r = find_lines_ex(img, finder)
    if r is None:
        return "-3 0 0"
    boxes, dark, trunc = r
    parts = [f"0 {len(boxes)} {int(trunc)}"] + [f"{x} {y} {w} {h} {d}" for (x, y, w, h), d in zip(boxes, dark)]
    return " ".join(parts)


def run(binary, cases, seed=0):
    """cases: [(img, finder)]. Returns the list of mismatches (index, c, py)."""
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory() as td:
        batch = Path(td) / "batch.bin"
        write_batch(batch, cases, rng)
        out = subprocess.run([str(binary), str(batch)], check=True, capture_output=True, text=True).stdout
    got = out.strip("\n").split("\n") if out.strip() else []
    if len(got) != len(cases):
        return [(-1, f"{len(got)} results", f"{len(cases)} cases")]
    bad = []
    for i, ((img, finder), line) in enumerate(zip(cases, got)):
        exp = expected(img, finder)
        if line.strip() != exp:
            bad.append((i, line.strip(), exp))
    return bad


def real_regions(manifests):
    """Region crops of every manifest row whose screenshot is available locally."""
    sys.path.insert(0, str(ROOT / "eval" / "linefind"))
    from build_gt import SHOT_DIRS, read, region_rgb
    out, missing = [], 0
    for m in manifests:
        for r in read(m):
            try:
                out.append(region_rgb(r, SHOT_DIRS))
            except FileNotFoundError:
                missing += 1
    if missing:
        print(f"{missing} regions skipped: screenshot not local (eval/linefind/build_gt.py fetch)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--cases", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--regions", action="append", default=[])
    args = ap.parse_args()
    imgs = make_cases(args.seed, args.cases) + real_regions(args.regions)
    cases = [(img, f) for img in imgs for f in ("grad", "legacy")]
    bad = run(args.bin, cases, args.seed)
    n_lines = sum(len((find_lines_ex(img, "grad") or [[]])[0]) for img, f in cases if f == "grad")
    if bad:
        for i, c, p in bad[:10]:
            img, f = cases[i] if i >= 0 else (None, None)
            print(f"MISMATCH case {i} ({f}, shape {None if img is None else img.shape}):\n  C  {c}\n  py {p}")
        print(f"LINEFIND_PARITY_FAIL {len(bad)}/{len(cases)}")
        return 1
    print(f"LINEFIND_PARITY_OK {len(cases)} runs ({len(imgs)} regions, {n_lines} grad lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
