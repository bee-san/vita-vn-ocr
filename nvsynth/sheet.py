#!/usr/bin/env python3
"""Contact sheets for the NVIDIA line crops, alone or next to real console crops.

  sheet.py shard  SHARD.npz [--n 36] [--seed 0] --out sheet.png
      crops as train.py sees them (grey, height 32) with their labels
  sheet.py compare 'SHARDS*.npz' --real-lines LINES.jsonl --real-shots DIR
                   [--kind vita|psp] [--pad 4] [--n 30] --out sheet.png
      NVIDIA crops (left) next to real crops (right), both grey at height 32:
      real lines are cut from screenshots with the same ink box + pad the
      eval harness and the line finder use (4 px on Vita, 2 on PSP)

Real lines are realdata rows ({"url", "kind", "ink": [x0, y0, x1, y1]}); their
screenshots are DIR/<vn>_<vndb image id>.jpg. Screenshots are copyrighted game
images: keep the sheets local and never commit them. Every sheet is at most
1280 px on each side.
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "synth"))
from common import DATA, to_gray  # noqa: E402

SIDE = 1280
ROW = 36


def label_font(size=16):
    fonts = sorted((DATA / "fonts" / "notosansjp").glob("*.[to]tf"))
    return ImageFont.truetype(str(fonts[0]), size) if fonts else ImageFont.load_default()


def load_shard(path):
    z = np.load(path)
    data, offs, shapes = z["data"], z["offsets"], z["shapes"]
    labels = json.loads(str(z["labels"]))
    return [(data[offs[i]:offs[i + 1]].reshape(shapes[i]), labels[i]) for i in range(len(labels))]


def to32(g):
    h, w = g.shape
    return cv2.resize(g, (max(8, int(round(w * 32 / h))), 32), interpolation=cv2.INTER_LINEAR)


def real_crops(lines_path, shots, kind, pad, n, rng):
    rows = [json.loads(x) for x in Path(lines_path).read_text().splitlines() if x.strip()]
    rows = [r for r in rows if r.get("kind") == kind and r.get("ink")]
    out = []
    for i in rng.permutation(len(rows)):
        if len(out) >= n:
            break
        r = rows[i]
        shot = Path(shots) / f"{r['vn']}_{r['url'].rsplit('/', 1)[-1]}"
        if not shot.exists():
            continue
        rgb = np.asarray(Image.open(shot).convert("RGB"))
        x0, y0, x1, y1 = r["ink"]
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(rgb.shape[1], x1 + pad), min(rgb.shape[0], y1 + pad)
        if x1 - x0 < 8 or y1 - y0 < 6:
            continue
        out.append(to32(to_gray(rgb[y0:y1, x0:x1])))
    return out


def save(sheet, out):
    sheet.thumbnail((SIDE, SIDE))
    sheet.save(out)
    print(f"{out}: {sheet.size[0]}x{sheet.size[1]}", flush=True)


def cmd_shard(args):
    rng = np.random.default_rng(args.seed)
    items = load_shard(args.shard)
    pick = [items[i] for i in rng.choice(len(items), min(args.n, len(items)), replace=False)]
    font = label_font()
    sheet = Image.new("L", (SIDE, ROW * len(pick)), 0)
    d = ImageDraw.Draw(sheet)
    for k, (g, lab) in enumerate(pick):
        im = Image.fromarray(g)
        sheet.paste(im.crop((0, 0, min(im.width, 760), 32)), (0, ROW * k + 2))
        d.text((770, ROW * k + 7), lab[:28], font=font, fill=255)
    save(sheet, args.out)


def cmd_compare(args):
    rng = np.random.default_rng(args.seed)
    nv = []
    for p in sorted(glob.glob(args.shards)):
        nv += [g for g, _ in load_shard(p)]
    nv = [nv[i] for i in rng.choice(len(nv), min(args.n, len(nv)), replace=False)]
    real = real_crops(args.real_lines, args.real_shots, args.kind, args.pad, args.n, rng)
    half = SIDE // 2
    rows = max(len(nv), len(real))
    sheet = Image.new("L", (SIDE, ROW * rows + 24), 40)
    d = ImageDraw.Draw(sheet)
    font = label_font(15)
    d.text((6, 2), "NVIDIA OCR-Synthetic-Multilingual-v1 (ja) line crops", font=font, fill=255)
    d.text((half + 6, 2), f"real {args.kind} crops (ink box + {args.pad} px)", font=font, fill=255)
    for col, crops in ((0, nv), (half, real)):
        for k, g in enumerate(crops):
            im = Image.fromarray(g)
            sheet.paste(im.crop((0, 0, min(im.width, half - 8), 32)), (col + 2, 24 + ROW * k + 2))
    save(sheet, args.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("shard")
    a.add_argument("shard")
    a.add_argument("--n", type=int, default=36)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", required=True)
    b = sub.add_parser("compare")
    b.add_argument("shards")
    b.add_argument("--real-lines", required=True)
    b.add_argument("--real-shots", required=True)
    b.add_argument("--kind", default="vita", choices=("vita", "psp"))
    b.add_argument("--pad", type=int, default=None, help="default 4 on Vita, 2 on PSP")
    b.add_argument("--n", type=int, default=30)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", required=True)
    args = ap.parse_args()
    if args.cmd == "shard":
        cmd_shard(args)
    else:
        if args.pad is None:
            args.pad = 4 if args.kind == "vita" else 2
        cmd_compare(args)


if __name__ == "__main__":
    main()
