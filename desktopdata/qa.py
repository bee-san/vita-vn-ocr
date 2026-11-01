#!/usr/bin/env python3
"""QA sheets of desktop-data shards: a few frames with their boxes drawn (positives by class, ignore areas in red),
each scaled to a tile, at most 1280 px per side (for one look per sheet; never committed).

  qa.py sheet --shard PATH (no suffix) [--n 6] [--seed 0] [--cols 2] --out PNG
  qa.py crops --shard PATH [--n 24] [--seed 0] --out PNG      line crops at their own size (a 1:1 check of the boxes)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from desktopdata import shards as SH  # noqa: E402

COLOUR = {"dialogue": (255, 140, 0), "name": (255, 0, 255), "notdlg": (0, 200, 255), "text": (0, 220, 0),
          "choice": (0, 160, 255), "ui": (255, 255, 0), "scene": (160, 80, 255)}


def cmd_sheet(args):
    import cv2
    data, samples = SH.read_shard(args.shard)
    rng = np.random.default_rng(args.seed)
    pick = sorted(rng.choice(len(samples), min(args.n, len(samples)), replace=False).tolist())
    cols = args.cols
    tw = (1280 - 4 * (cols - 1)) // cols
    tiles = []
    for i in pick:
        s = samples[i]
        g = SH.canvas(data, s)
        im = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        t = max(1, int(round(max(s["w"], s["h"]) / 640)))
        for b in s["ignore"]:
            cv2.rectangle(im, (b[0], b[1]), (b[2] - 1, b[3] - 1), (0, 0, 255), t)
        for ln in s["lines"]:
            c = COLOUR.get(ln[5], (0, 255, 0))
            cv2.rectangle(im, (ln[0], ln[1]), (ln[2] - 1, ln[3] - 1), c[::-1], t)
        f = tw / s["w"]
        im = cv2.resize(im, (tw, max(1, int(round(s["h"] * f)))), interpolation=cv2.INTER_AREA)
        tag = f"{s['w']}x{s['h']} {s.get('shot') or s['meta'].get('scene', '')}"[:60]
        cv2.putText(im, tag, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        tiles.append(im)
    rows = []
    for k in range(0, len(tiles), cols):
        row = tiles[k:k + cols]
        h = max(t.shape[0] for t in row)
        row = [np.pad(t, ((0, h - t.shape[0]), (0, 0), (0, 0))) for t in row]
        while len(row) < cols:
            row.append(np.zeros((h, tw, 3), np.uint8))
        rows.append(np.hstack([np.pad(t, ((0, 0), (0, 4), (0, 0))) for t in row])[:, :1280])
    sheet = np.vstack([np.pad(r, ((0, 4), (0, 0), (0, 0))) for r in rows])
    if sheet.shape[0] > 1280:
        f = 1280 / sheet.shape[0]
        sheet = cv2.resize(sheet, (int(sheet.shape[1] * f), 1280), interpolation=cv2.INTER_AREA)
    cv2.imwrite(args.out, sheet)
    print(args.out, sheet.shape[1], sheet.shape[0], [samples[i].get("shot") for i in pick])


def cmd_crops(args):
    import cv2
    data, samples = SH.read_shard(args.shard)
    rng = np.random.default_rng(args.seed)
    cand = [(i, j) for i, s in enumerate(samples) for j in range(len(s["lines"]))]
    pick = [cand[k] for k in sorted(rng.choice(len(cand), min(args.n, len(cand)), replace=False).tolist())]
    rows, cur, cur_w = [], [], 0
    for i, j in pick:
        s = samples[i]
        g = SH.canvas(data, s)
        x0, y0, x1, y1 = s["lines"][j][:4]
        m = max(4, (y1 - y0) // 3)
        c = cv2.cvtColor(np.ascontiguousarray(g[max(0, y0 - m):y1 + m, max(0, x0 - m):x1 + m]), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(c, (min(m, x0), min(m, y0)), (min(m, x0) + x1 - x0 - 1, min(m, y0) + y1 - y0 - 1), (0, 0, 255), 1)
        if c.shape[1] > 1270:
            f = 1270 / c.shape[1]
            c = cv2.resize(c, (1270, max(1, int(c.shape[0] * f))), interpolation=cv2.INTER_AREA)
        if cur_w + c.shape[1] > 1280 and cur:
            rows.append(cur)
            cur, cur_w = [], 0
        cur.append(c)
        cur_w += c.shape[1] + 4
    if cur:
        rows.append(cur)
    out_rows = []
    for r in rows:
        h = max(t.shape[0] for t in r)
        out_rows.append(np.hstack([np.pad(t, ((0, h - t.shape[0]), (0, 4), (0, 0)), constant_values=255) for t in r]))
    w = max(r.shape[1] for r in out_rows)
    sheet = np.vstack([np.pad(r, ((0, 4), (0, w - r.shape[1]), (0, 0)), constant_values=255) for r in out_rows])
    if sheet.shape[0] > 1280:
        sheet = sheet[:1280]
    cv2.imwrite(args.out, sheet)
    print(args.out, sheet.shape[1], sheet.shape[0])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("sheet", "crops"):
        p = sub.add_parser(name)
        p.add_argument("--shard", required=True)
        p.add_argument("--n", type=int, default=6 if name == "sheet" else 24)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--cols", type=int, default=2)
        p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"sheet": cmd_sheet, "crops": cmd_crops}[args.cmd](args)


if __name__ == "__main__":
    main()
