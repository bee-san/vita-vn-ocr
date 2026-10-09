#!/usr/bin/env python3
"""Spot-check accepted teacher labels by eye.

  spot.py sheet  [--n 100 --seed 1] [--per-sheet 10]   sample + contact sheets in $TEACHER_OUT/spot/
  spot.py report                                       error rate from teacher/manifests/spot.jsonl

`sheet` draws N accepted lines uniformly at random (seeded) from teacher/manifests/lines_*.jsonl
and renders each as it goes into the npz (the training crop at its target resolution),
scaled to 48 px high, with its label typed underneath in BIZ UDGothic (OFL, from
$HERD/shared/data/fonts; not committed) and its number. Sheets stay under 1280 px. Each line
is then compared by eye, glyph by glyph (small kana, dakuten, ー/一, punctuation and ellipsis
cell counts included), and recorded in teacher/manifests/spot.jsonl as
  {"id", "label", "ok": true|false, "truth": the correct text if wrong, "note"}.
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from teacher import build  # noqa: E402
from teacher import geometry as G  # noqa: E402
from teacher.label import HERD, OUT  # noqa: E402

FONT = HERD / "shared" / "data" / "fonts" / "bizudgothic" / "BIZUDGothic-Regular.ttf"
SPOT = build.MAN / "spot.jsonl"


def accepted():
    rows = []
    for kind in ("vita", "psp", "hires"):
        p = build.MAN / f"lines_{kind}.jsonl"
        if p.exists():
            rows += [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    return rows


def crop_image(r, shots):
    import cv2
    s = shots[r["id"].rsplit("_", 1)[0]]
    bgr = cv2.imread(str(build.shot_path(s)), cv2.IMREAD_COLOR)
    t = r["targets"][0]
    sc = G.target_scale(bgr.shape[1], bgr.shape[0], t) if s["kind"] == "hires" else 1.0
    img = G.scale_image(np.ascontiguousarray(bgr[:, :, ::-1]), sc)
    x0, y0, x1, y1 = G.crop_box(r["crop"], sc, G.PAD[t], img.shape[1], img.shape[0])
    return img[y0:y1, x0:x1], t


def cmd_sheet(args):
    from PIL import Image, ImageDraw, ImageFont
    rows = accepted()
    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.n, len(rows)))
    shots = {json.loads(x)["shot"]: json.loads(x) for x in (build.MAN / "shots.jsonl").read_text(
        encoding="utf-8").splitlines() if x.strip()}
    font = ImageFont.truetype(str(FONT), 22)
    small = ImageFont.truetype(str(FONT), 14)
    out = OUT / "spot"
    out.mkdir(parents=True, exist_ok=True)
    (out / "sample.jsonl").write_text("".join(json.dumps({"n": i, "id": r["id"], "label": r["label"]},
                                                         ensure_ascii=False) + "\n" for i, r in enumerate(sample)),
                                      encoding="utf-8")
    H_CROP = 48
    W, row_h = 1260, H_CROP + 34 + 16
    for s0 in range(0, len(sample), args.per_sheet):
        chunk = sample[s0:s0 + args.per_sheet]
        sheet = Image.new("RGB", (W, row_h * len(chunk) + 6), (255, 255, 255))
        d = ImageDraw.Draw(sheet)
        for j, r in enumerate(chunk):
            crop, t = crop_image(r, shots)
            h, w = crop.shape[:2]
            f = H_CROP / max(1, h)
            if w * f > W - 60:
                f = (W - 60) / w
            im = Image.fromarray(crop).resize((max(1, int(w * f)), max(1, int(h * f))), Image.LANCZOS)
            y = 4 + j * row_h
            d.text((4, y + 10), f"{s0 + j:02d}", fill=(200, 0, 0), font=small)
            sheet.paste(im, (40, y))
            d.text((40, y + H_CROP + 4), r["label"], fill=(0, 0, 0), font=font)
            d.text((W - 300, y + H_CROP + 12), f"{r['id']} {t}", fill=(120, 120, 120), font=small)
        p = out / f"sheet_{s0 // args.per_sheet:02d}.png"
        sheet.save(p)
        print(p, sheet.size, flush=True)


def cmd_report(args):
    rows = [json.loads(x) for x in SPOT.read_text(encoding="utf-8").splitlines() if x.strip()]
    bad = [r for r in rows if not r["ok"]]
    print(f"{len(bad)} wrong of {len(rows)} ({100 * len(bad) / max(1, len(rows)):.1f}%)")
    for r in bad:
        print(f"  {r['id']}: label {r['label']!r} truth {r.get('truth')!r} {r.get('note') or ''}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("sheet")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--per-sheet", type=int, default=8)
    sub.add_parser("report")
    args = ap.parse_args()
    {"sheet": cmd_sheet, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
