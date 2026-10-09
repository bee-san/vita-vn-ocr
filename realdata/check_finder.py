#!/usr/bin/env python3
"""Compare realdata's training crops with the console line finder's crops.

  check_finder.py [--sample N --seed S] [--margin F]

For a random sample of accepted lines (manifests/lines_*.jsonl): rescale the screenshot
to the line's first target as build.py render does, cut a region around the line (its
train box grown by --margin line heights up and down and 2 line heights sideways, the
kind of box a player draws), and run synth/common.find_lines_ex(region, "grad"), the
twin of runtime/vocr_lines.c. The finder box that overlaps the train box most is its
match. Reports how often the finder finds the line (IoU >= 0.5), the per-side edge
differences (finder minus train box), and how often common.line_dark() on the train
box agrees with the finder's own dark flag.
"""
import argparse
import random
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build  # noqa: E402
import lines as L  # noqa: E402
from vndb import load_shots, shot_key, shot_path  # noqa: E402

sys.path.insert(0, str(HERE.parent / "synth"))
from common import find_lines_ex, line_dark, to_gray  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--margin", type=float, default=0.6)
    args = ap.parse_args()
    shots = {shot_key(s): s for s in load_shots()}
    rows = random.Random(args.seed).sample(build.load_lines(), args.sample)
    found, deltas, dark_agree, n_dark = 0, {k: [] for k in ("x0", "y0", "x1", "y1")}, 0, Counter()
    per_target = {"vita": {k: [] for k in ("x0", "y0", "x1", "y1")}, "psp": {k: [] for k in ("x0", "y0", "x1", "y1")}}
    for r in rows:
        s = shots[r["id"].rsplit("_", 1)[0]]
        rgb = np.ascontiguousarray(cv2.imread(str(shot_path(s)), cv2.IMREAD_COLOR)[:, :, ::-1])
        t = r["targets"][0]
        sc = L.target_scale(rgb.shape[1], rgb.shape[0], t) if s["kind"] == "hires" else 1.0
        img = L.scale_image(rgb, sc)
        H, W = img.shape[:2]
        bx0, by0, bx1, by1 = L.train_box(r["ink"], sc, W, H)
        lh = by1 - by0
        rx0, ry0 = max(0, bx0 - 2 * lh), max(0, by0 - int(args.margin * lh))
        rx1, ry1 = min(W, bx1 + 2 * lh), min(H, by1 + int(args.margin * lh))
        region = img[ry0:ry1, rx0:rx1]
        got = find_lines_ex(region, "grad")
        boxes, darks = (got[0], got[1]) if got else ([], [])
        mine = (bx0 - rx0, by0 - ry0, bx1 - rx0, by1 - ry0)
        cands = [((x, y, x + w, y + h), d) for (x, y, w, h), d in zip(boxes, darks)]
        best = max(cands, key=lambda c: build.iou(c[0], mine), default=None)
        d_mine = line_dark(to_gray(img), (bx0, by0, bx1 - bx0, by1 - by0))
        n_dark[d_mine] += 1
        if best is None or build.iou(best[0], mine) < 0.5:
            continue
        found += 1
        for k, a, b in zip(("x0", "y0", "x1", "y1"), best[0], mine):
            deltas[k].append(a - b)
            per_target[t][k].append(a - b)
        dark_agree += int(best[1] == d_mine)
    print(f"{len(rows)} lines: the grad finder finds {found} ({found / len(rows):.1%}) at IoU >= 0.5")
    for k, v in deltas.items():
        v = np.array(v)
        print(f"  {k}: finder - train box median {np.median(v):+.0f}, |delta| <= 1 px on "
              f"{np.mean(np.abs(v) <= 1):.1%}, p10 {np.percentile(v, 10):+.0f} p90 {np.percentile(v, 90):+.0f}")
    for t, d in per_target.items():
        med = " ".join(f"{k} {np.median(v):+.0f}" for k, v in d.items())
        print(f"  {t} ({len(d['x0'])} lines): median deltas {med}")
    print(f"  line_dark on the train box agrees with the finder's flag on {dark_agree}/{found}; "
          f"dark train boxes {n_dark[1]}/{len(rows)}")


if __name__ == "__main__":
    main()
