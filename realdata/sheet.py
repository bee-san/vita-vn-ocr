#!/usr/bin/env python3
"""Contact sheets for checking line labels by eye.

  sheet.py --lines GLOB [--ids a,b,...|--idfile F] [--sample N --seed S] [--skip-how hand]
           --out DIR [--height 44] [--per-sheet 24] [--crop view|train|vita|psp]

Each sheet is at most 1280x1280 px (safe to view). Row i shows the line's voter view, or
its training crop (--crop train: the crop build.py renders for the line's first target,
the exact pixels its label goes with), scaled to a fixed height, numbered on the left.
--lines takes label.py output or the accepted-line manifests (manifests/lines_*.jsonl).
The matching table (DIR/sheet_NN.txt, also printed) lists every row's id, kind and the
voters' readings plus the label if the record has one, so the image can be compared
with the text.
"""
import argparse
import glob
import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import lines as L  # noqa: E402
from vndb import load_shots, realdata_dir, shot_key  # noqa: E402

W_MAX, NUM_W = 1280, 52


def with_shot(recs):
    """Manifest rows (lines_<kind>.jsonl) carry no shot fields: add them from the shots manifests."""
    shots = {shot_key(s): s for s in load_shots()}
    out = []
    for r in recs:
        if "url" not in r:
            s = shots[r["id"].rsplit("_", 1)[0]]
            r = {**r, "url": s["url"], "vn": s["vn"], "kind": s["kind"], "dims": s["dims"]}
        out.append(r)
    return out


def shot_file(rec):
    return realdata_dir() / "shots" / rec["kind"] / f"{rec['vn']}_{rec['url'].rsplit('/', 1)[-1]}"


def crop_of(rec, which, cache):
    path = shot_file(rec)
    if path not in cache:
        if len(cache) > 64:
            cache.clear()
        cache[path] = cv2.imread(str(path), cv2.IMREAD_COLOR)
    bgr = cache[path]
    if which == "view":
        h, w = bgr.shape[:2]
        x0, y0, x1, y1 = rec.get("view") or L.voter_box(rec["ink"], w, h)
        return bgr[y0:y1, x0:x1]
    h, w = bgr.shape[:2]
    s = L.target_scale(w, h, which)
    img = L.scale_image(bgr, s)
    x0, y0, x1, y1 = L.train_box(rec["ink"], s, img.shape[1], img.shape[0])
    return img[y0:y1, x0:x1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lines", required=True, help="file or glob")
    ap.add_argument("--skip-how", default="", help="drop lines whose 'how' is this (e.g. hand)")
    ap.add_argument("--ids", default="")
    ap.add_argument("--idfile", type=Path)
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--height", type=int, default=44)
    ap.add_argument("--per-sheet", type=int, default=24)
    ap.add_argument("--crop", default="view", choices=["view", "train", "vita", "psp"])
    args = ap.parse_args()
    recs = with_shot([json.loads(x) for f in sorted(glob.glob(args.lines))
                      for x in Path(f).read_text(encoding="utf-8").splitlines() if x.strip()])
    if args.skip_how:
        recs = [r for r in recs if r.get("how") != args.skip_how]
    ids = [i for i in args.ids.split(",") if i]
    if args.idfile:
        ids += [x.strip() for x in args.idfile.read_text().split() if x.strip()]
    if ids:
        by = {r["id"]: r for r in recs}
        recs = [by[i] for i in ids]
    elif args.sample:
        recs = random.Random(args.seed).sample(recs, min(args.sample, len(recs)))
    args.out.mkdir(parents=True, exist_ok=True)
    cache, row_h = {}, args.height + 6
    for s0 in range(0, len(recs), args.per_sheet):
        chunk = recs[s0:s0 + args.per_sheet]
        sheet = np.full((row_h * len(chunk), W_MAX, 3), 40, np.uint8)
        table = []
        for j, rec in enumerate(chunk):
            n = s0 + j
            c = crop_of(rec, rec["targets"][0] if args.crop == "train" else args.crop, cache)
            nw = max(1, int(round(c.shape[1] * args.height / c.shape[0])))
            c = cv2.resize(c, (nw, args.height), interpolation=cv2.INTER_AREA if c.shape[0] > args.height
                           else cv2.INTER_CUBIC)[:, :W_MAX - NUM_W]
            y = j * row_h + 3
            sheet[y:y + args.height, NUM_W:NUM_W + c.shape[1]] = c
            cv2.putText(sheet, str(n), (2, y + args.height - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 0), 1, cv2.LINE_AA)
            reads = " | ".join(f"{k}:{v[0]}" for k, v in rec.get("r", {}).items())
            lab = f"  LABEL:{rec['label']}" if rec.get("label") is not None else ""
            tgt = f"->{rec['targets'][0]}" if args.crop == "train" else ""
            table.append(f"{n:4d} {rec['id']} [{rec['kind']}{tgt}]{lab}  {reads}")
        cv2.imwrite(str(args.out / f"sheet_{s0 // args.per_sheet:02d}.png"), sheet)
        (args.out / f"sheet_{s0 // args.per_sheet:02d}.txt").write_text("\n".join(table) + "\n", encoding="utf-8")
        print("\n".join(table), flush=True)
    print(f"{len(recs)} lines on {(len(recs) + args.per_sheet - 1) // args.per_sheet} sheets in {args.out}",
          flush=True)


if __name__ == "__main__":
    main()
