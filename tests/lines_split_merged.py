#!/usr/bin/env python3
"""Parity on the splitting branch: runtime/vocr_lines_split.c against its twin on merged-line regions.

  lines_split_merged.py --bin build/split/vocr_lines_split_test [--cases 200] [--seed 3] [--rows]

tests/lines_split_parity.py's regions rarely make the split act (about 1 run in 45). These regions are rows of
outlined glyphs set close together over short scene strokes, at a contrast the GRAD finder takes for text: the case
the split is for, where GRAD often returns one box for two or three rows and the split separates them (about 3
regions in 8). Each region runs through the finder, the split and the refinement (lines_split_parity.FLAG_SETS) in
the C driver and the twins, with lines_split_parity.run, which is unchanged (its sha256 is pinned by
eval/translucent_box_protocol.json). Added after the audit of #102 (n8).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "synth"), str(ROOT / "tests")]
import lines_split_parity as LP  # noqa: E402


def merged_region(rng):
    """Two or three rows of outlined glyphs set close together over short scene strokes."""
    lh = int(rng.integers(12, 25))
    gap = int(rng.integers(3, 9))
    rows = int(rng.integers(2, 4))
    top = int(rng.integers(4, 12))
    h = top + rows * (lh + gap) + int(rng.integers(4, 12))
    w = int(rng.integers(160, 520))
    base = int(rng.integers(150, 221))
    ink = base - int(rng.integers(60, 121))
    g = np.full((h, w), base, np.int32)
    for _ in range(int(rng.integers(60, 200))):                     # short scene strokes, every direction
        x0, y0 = int(rng.integers(0, w)), int(rng.integers(0, h))
        length, ang = int(rng.integers(6, 40)), float(rng.random()) * np.pi
        for t in range(length):
            x, y = int(x0 + t * np.cos(ang)), int(y0 + t * np.sin(ang))
            if 0 <= x < w and 0 <= y < h:
                g[y, x] = ink
    fill, outline = (255, int(rng.integers(0, 50))) if rng.random() < 0.8 else (int(rng.integers(0, 40)), 235)
    y = top
    for i in range(rows):
        x1 = w - int(rng.integers(4, 40)) if i + 1 < rows else int(rng.integers(w // 3, w - 4))
        for x in range(6, x1 - lh + 1, lh + 4):                     # block glyphs: strokes 2 px wide, bars
            gl = np.zeros((lh, lh), bool)
            gl[:, 1::4] = True
            gl[:, 2::4] = True
            gl[0, :] = gl[-1, :] = True
            ring = np.zeros_like(gl)
            ring[:, :-1] |= gl[:, 1:]
            ring[:, 1:] |= gl[:, :-1]
            cell = g[y:y + lh, x:x + lh]
            cell[ring & ~gl] = outline
            cell[gl] = fill
        y += lh + gap
    img = np.clip(g, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:                                          # some RGB, as the console's frames are
        img = np.stack([img, img, np.clip(img.astype(int) + int(rng.integers(-15, 16)), 0, 255).astype(np.uint8)], -1)
    return img


def cases(seed, n):
    rng = np.random.default_rng(seed)
    imgs = [merged_region(rng) for _ in range(n)]
    return [(img, LP.FLAG_SETS[int(rng.integers(0, len(LP.FLAG_SETS)))] if k else LP.FLAG_SETS[1])
            for img in imgs for k in range(2)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--cases", type=int, default=200)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--rows", action="store_true")
    args = ap.parse_args()
    cs = cases(args.seed, args.cases)
    bad, splits = LP.run(args.bin, cs, args.seed, args.rows)
    if bad:
        for i, c, p in bad[:10]:
            print(f"MISMATCH case {i}:\n  C  {c}\n  py {p}")
        print(f"SPLIT_MERGED_PARITY_FAIL {len(bad)}/{len(cs)}")
        return 1
    if not splits:
        print("SPLIT_MERGED_PARITY_FAIL no run split a box: the generator no longer exercises the splitting branch")
        return 1
    print(f"SPLIT_MERGED_PARITY_OK {len(cs)} runs ({args.cases} regions, {splits} runs with a box split)"
          f"{' rows' if args.rows else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
