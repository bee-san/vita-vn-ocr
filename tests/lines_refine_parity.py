#!/usr/bin/env python3
"""Parity check: runtime/vocr_lines_refine.c against its Python twin synth/lines_refine.py.

  lines_refine_parity.py --bin build/lines/vocr_lines_refine_test [--cases 400] [--seed 1] [--rows]
                         [--cohort DIR/NAME.pkl ...]

Renders tests/linefind_parity.py's procedural regions (text-like strokes of both polarities, outlines,
noise, gradients, frames, tiny and full-screen sizes, grey and RGBA, padded strides), optionally adds
the regions of finder-tighten's development cohorts (eval/finder_tighten.py cohorts; private, local
only), runs the GRAD finder and then the refinement with several flag sets through the C driver and the
twin, and exits non-zero on the first box, polarity flag or return code that differs. --rows uses the
C row-callback form.
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
from common import find_lines_ex  # noqa: E402
from linefind_parity import make_cases  # noqa: E402

FLAG_SETS = (R.DEFAULT, R.ALL, R.TIGHTEN, R.SPLIT, R.TIGHTEN | R.FRAMES, R.CURSOR | R.PORTRAIT,
             R.TIGHTEN | R.CURSOR, R.TIGHTEN | R.SPLIT | R.PORTRAIT, 0)


def write_batch(path, cases, rng):
    with open(path, "wb") as f:
        for img, flags in cases:
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
            f.write(struct.pack("<6I", w, h, bpp, 0, stride, flags))
            f.write(buf.tobytes())


def expected(img, flags):
    boxes, dark, _ = find_lines_ex(img, "grad")
    rb, rd, _ = R.refine_lines(img, boxes, dark, flags)
    return [len(boxes), len(rb), rb, rd]


def parse(line):
    v = [int(t) for t in line.split()]
    rc, found, n = v[0], v[1], v[2]
    boxes = [tuple(v[5 + 5 * i:9 + 5 * i]) for i in range(n)]
    dark = [v[9 + 5 * i] for i in range(n)]
    return rc, [found, n, boxes, dark]


def run(binary, cases, seed=0, rows=False):
    """cases: [(img, flags)]. Returns the list of mismatches (index, C, twin)."""
    rng = np.random.default_rng(seed)
    with tempfile.TemporaryDirectory() as td:
        batch = Path(td) / "batch.bin"
        write_batch(batch, cases, rng)
        cmd = [str(binary)] + (["--rows"] if rows else []) + [str(batch)]
        out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    got = out.strip("\n").split("\n") if out.strip() else []
    if len(got) != len(cases):
        return [(-1, f"{len(got)} results", f"{len(cases)} cases")]
    bad = []
    for i, ((img, flags), line) in enumerate(zip(cases, got)):
        rc, c = parse(line)
        exp = expected(img, flags)
        if rc != 0 or c != exp:
            bad.append((i, f"rc={rc} {c}", str(exp)))
    return bad


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
    rng = np.random.default_rng(args.seed + 7)
    imgs = make_cases(args.seed, args.cases) + cohort_regions(args.cohort, args.limit)
    cases = [(img, int(FLAG_SETS[int(rng.integers(0, len(FLAG_SETS)))]) if k else R.DEFAULT)
             for img in imgs for k in range(2)]
    bad = run(args.bin, cases, args.seed, args.rows)
    if bad:
        for i, c, p in bad[:10]:
            img, f = cases[i] if i >= 0 else (None, None)
            print(f"MISMATCH case {i} (flags {f}, shape {None if img is None else img.shape}):\n  C  {c}\n  py {p}")
        print(f"REFINE_PARITY_FAIL {len(bad)}/{len(cases)}")
        return 1
    print(f"REFINE_PARITY_OK {len(cases)} runs ({len(imgs)} regions){' rows' if args.rows else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
