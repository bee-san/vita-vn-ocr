#!/usr/bin/env python3
"""Parity check: runtime/vocr_lines_split.c against its Python twin synth/lines_split.py.

  lines_split_parity.py --bin build/lines/vocr_lines_split_test [--cases 400] [--seed 1] [--rows]
                        [--cohort DIR/NAME.pkl ...] [--limit N]

Renders tests/linefind_parity.py's procedural regions (text-like strokes of both polarities, outlines, noise,
gradients, frames, tiny and full-screen sizes, grey and RGBA, padded strides) and busy ones (rows of strokes over
line art and blobs, some set close together), optionally adds the regions of whole-dialogue cohorts (private,
local only), runs the GRAD finder, the split (flags 0 or VOCR_SPLIT_DEFAULT) and the refinement (flags 0 or
VOCR_REFINE_DEFAULT) through the C driver and the twins, and exits non-zero on the first box, polarity flag, count
or return code that differs. --rows uses the C row-callback form of the split.
"""
import argparse
import pickle
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "synth"), str(ROOT / "tests")]
import lines_refine as R  # noqa: E402
import lines_split as S  # noqa: E402
from common import find_lines_ex  # noqa: E402
from linefind_parity import make_cases  # noqa: E402

FLAG_SETS = ((S.DEFAULT, 0), (S.DEFAULT, R.DEFAULT), (0, R.DEFAULT), (0, 0))


def busy_region(rng):
    """Rows of outlined or plain strokes over a scene of line art and blobs, as through a translucent box."""
    h = int(rng.integers(30, 140))
    w = int(rng.integers(60, 700))
    paper = int(rng.integers(0, 256))
    g = np.full((h, w), paper, np.int32)
    for _ in range(int(rng.integers(5, 60))):                      # line art
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        x1, y1 = int(rng.integers(0, w)), int(rng.integers(0, h))
        n = max(abs(x1 - x0), abs(y1 - y0)) + 1
        v = int(rng.integers(0, 256))
        for t in range(n):
            x = x0 + (x1 - x0) * t // n
            y = y0 + (y1 - y0) * t // n
            g[y, x] = v
    for _ in range(int(rng.integers(0, 5))):                       # blobs
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        g[y0:y0 + int(rng.integers(3, 40)), x0:x0 + int(rng.integers(3, 60))] = int(rng.integers(0, 256))
    alpha = float(rng.uniform(0.2, 0.9))                             # the translucent box
    box = int(rng.integers(0, 256))
    g = (alpha * box + (1 - alpha) * g).astype(np.int32)
    lh = int(rng.integers(9, 30))
    pitch = lh + int(rng.integers(0, lh))
    ink = int(rng.integers(0, 256))
    outline = rng.random() < 0.6
    y = int(rng.integers(0, max(1, h // 3)))
    rows = int(rng.integers(1, 5))
    for i in range(rows):
        if y + lh > h:
            break
        x1 = w - int(rng.integers(0, w // 3 + 1))
        if i == rows - 1 and rng.random() < 0.5:
            x1 = max(8, x1 // int(rng.integers(2, 6)))
        cw = int(rng.integers(4, lh + 1))
        yy, xx = np.mgrid[y:y + lh, 4:x1]
        cx, cy = (xx - 4) % (cw + 2), yy - y
        on = (cx < cw) & ((cx % 4 < 2) | (cy % 6 < 2) | ((cx + cy) % 7 == 0))
        if outline:
            ring = (cx < cw + 1) & ((cx % 4 == 2) | (cy % 6 == 2)) & ~on
            g[yy[ring], xx[ring]] = 255 - ink
        g[yy[on], xx[on]] = ink
        y += pitch
    g = np.clip(g, 0, 255).astype(np.uint8)
    if rng.random() < 0.5:
        rgb = np.stack([g, np.clip(g.astype(int) + int(rng.integers(-20, 21)), 0, 255).astype(np.uint8), g], -1)
        return rgb
    return g


def write_batch(path, cases, rng):
    with open(path, "wb") as f:
        for img, (sf, rf) in cases:
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
            f.write(struct.pack("<7I", w, h, bpp, 0, stride, sf, rf))
            f.write(buf.tobytes())


def expected(img, flags):
    sf, rf = flags
    boxes, dark, _ = find_lines_ex(img, "grad")
    sb, sd, _ = S.split_lines(img, boxes, dark, sf)
    rb, rd, _ = R.refine_lines(img, sb, sd, rf)
    return [len(boxes), len(sb), len(rb), rb, rd]


def parse(line):
    v = [int(t) for t in line.split()]
    rc, found, nsplit, n = v[0], v[1], v[2], v[3]
    boxes = [tuple(v[6 + 5 * i:10 + 5 * i]) for i in range(n)]
    dark = [v[10 + 5 * i] for i in range(n)]
    return rc, [found, nsplit, n, boxes, dark], v[5]


def run(binary, cases, seed=0, rows=False):
    """cases: [(img, (split flags, refine flags))]. Returns (mismatches [(index, C, twin)], splits)."""
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory() as td:
        batch = Path(td) / "batch.bin"
        write_batch(batch, cases, rng)
        cmd = [str(binary)] + (["--rows"] if rows else []) + [str(batch)]
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    got = out.strip("\n").split("\n") if out.strip() else []
    if len(got) != len(cases):
        return [(-1, f"{len(got)} results", f"{len(cases)} cases")], 0
    bad, splits = [], 0
    for i, ((img, flags), line) in enumerate(zip(cases, got)):
        rc, c, _ = parse(line)
        exp = expected(img, flags)
        splits += int(exp[1] > exp[0])
        if rc != 0 or c != exp:
            bad.append((i, f"rc={rc} {c}", str(exp)))
    return bad, splits


def cohort_regions(paths, limit):
    out = []
    for p in paths:
        with open(p, "rb") as f:
            cases = pickle.load(f)
        seen = set()
        for c in cases:
            key = (c.get("unit") or c["id"], c.get("variant"))
            if key in seen:
                continue
            seen.add(key)
            out.append(c["image"])
            if limit and len(out) >= limit:
                break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--cases", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--rows", action="store_true")
    ap.add_argument("--cohort", action="append", default=[])
    ap.add_argument("--limit", type=int, default=0, help="regions per cohort (0: all)")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed + 11)
    imgs = make_cases(args.seed, args.cases) + [busy_region(rng) for _ in range(args.cases)]
    imgs += cohort_regions(args.cohort, args.limit)
    cases = [(img, FLAG_SETS[int(rng.integers(0, len(FLAG_SETS)))] if k else FLAG_SETS[1])
             for img in imgs for k in range(2)]
    bad, splits = run(args.bin, cases, args.seed, args.rows)
    if bad:
        for i, c, p in bad[:10]:
            img, f = cases[i] if i >= 0 else (None, None)
            print(f"MISMATCH case {i} (flags {f}, shape {None if img is None else img.shape}):\n  C  {c}\n  py {p}")
        print(f"SPLIT_PARITY_FAIL {len(bad)}/{len(cases)}")
        return 1
    print(f"SPLIT_PARITY_OK {len(cases)} runs ({len(imgs)} regions, {splits} runs with a box split)"
          f"{' rows' if args.rows else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
