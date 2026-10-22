#!/usr/bin/env python3
"""Parity of runtime/vocr_lines_widen.c with its Python twin synth/lines_widen.py (issue #109).

    lines_widen_parity.py --bin runtime/build/widen/vocr_lines_widen_test [--cases 3000] [--seed 0]

Random line strips (grey and RGBA, light or dark glyph runs over noisy or textured backgrounds, gaps of every
width) and regions whose edges cut a glyph, fall in a gap, or touch the strip's own edge, with finder-like boxes
that start at the region's first column or not: the C driver and the twin must give the same box and sides.
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
import lines_widen as LW  # noqa: E402


def case(rng):
    H = int(rng.integers(6, 48))
    W = int(rng.integers(8, 160))
    bpp = int(rng.choice([1, 4]))
    bg = int(rng.integers(0, 256))
    img = np.clip(rng.normal(bg, rng.choice([0, 3, 12, 30]), (H, W)), 0, 255).astype(np.int32)
    x = 0
    while x < W:                                     # glyph runs and gaps
        run = int(rng.integers(1, 14))
        if rng.random() < 0.6:
            v = int(rng.integers(0, 256))
            y0 = int(rng.integers(0, max(1, H // 3)))
            y1 = int(rng.integers(max(y0 + 1, H - H // 3), H + 1))
            img[y0:y1, x:x + run] = v
        x += run + int(rng.integers(0, 6))
    if bpp == 4:
        rgb = np.stack([img, np.clip(img + rng.integers(-20, 21), 0, 255), np.clip(img - rng.integers(-20, 21), 0, 255),
                        np.full_like(img, 255)], axis=-1)
        px = rgb.astype(np.uint8)
    else:
        px = img.astype(np.uint8)
    rx = int(rng.integers(0, W))
    rw = int(rng.integers(1, W - rx + 1))
    bx = 0 if rng.random() < 0.6 else int(rng.integers(0, rw))
    if rng.random() < 0.5:
        bw = rw - bx
    else:
        bw = int(rng.integers(1, rw - bx + 1))
    return px, bpp, rx, rw, bx, bw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--cases", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    blobs, expect = [], []
    for _ in range(a.cases):
        px, bpp, rx, rw, bx, bw = case(rng)
        H, W = px.shape[:2]
        blobs.append(struct.pack("<7I", W, H, bpp, rx, rw, bx, bw) + np.ascontiguousarray(px).tobytes())
        x, w, sides = LW.widen(px, rx, rw, bx, bw)
        expect.append(f"{sides} {x} {w}")
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "cases.bin"
        p.write_bytes(struct.pack("<I", len(blobs)) + b"".join(blobs))
        res = subprocess.run(a.bin.split() + ["--parity", str(p)], capture_output=True, text=True)
    if res.returncode:
        sys.exit(f"driver failed rc={res.returncode}: {res.stderr[-2000:]}")
    got = res.stdout.splitlines()
    if len(got) != len(expect):
        sys.exit(f"{len(got)} results for {len(expect)} cases")
    bad = [i for i, (g, e) in enumerate(zip(got, expect)) if g != e]
    if bad:
        for i in bad[:5]:
            print("case", i, "C:", got[i], "twin:", expect[i], file=sys.stderr)
        sys.exit(f"widen parity: {len(bad)} of {len(expect)} cases differ")
    widened = sum(1 for e in expect if e.split()[0] != "0")
    print(f"widen parity: {len(expect)} cases ({widened} widened): C == twin on every case")


if __name__ == "__main__":
    main()
