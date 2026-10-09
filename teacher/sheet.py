#!/usr/bin/env python3
"""Contact sheets of teacher lines on eval's test screenshots, from measure.py's per-line dump,
for checking the rule's mistakes by eye (sheets stay under 1280 px).

  sheet.py DUMP.lines.json OUTDIR [--status wrong,partial,unmatched] [--limit 60]

Each row shows the teacher crop (scaled to 36 px high, at most 1100 px wide) with the label,
the truth and the decision underneath, plus a context strip of the screenshot around it.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from teacher.label import EVAL_DATA, EVAL_SETS  # noqa: E402
from teacher.spot import FONT  # noqa: E402


def main():
    from PIL import Image, ImageDraw, ImageFont
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("out")
    ap.add_argument("--status", default="wrong,partial")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--per-sheet", type=int, default=8)
    args = ap.parse_args()
    d = json.loads(Path(args.dump).read_text(encoding="utf-8"))
    want = set(args.status.split(","))
    rows = []
    for r in d["lines"]:
        st = "wrong" if r["ok"] is False else r["status"]
        if st in want and (r["label"] or st == "wrong"):
            rows.append((st, r))
    rows = rows[:args.limit]
    font = ImageFont.truetype(str(FONT), 18)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    W, rh = 1260, 36 + 3 * 22 + 70
    for s0 in range(0, len(rows), args.per_sheet):
        chunk = rows[s0:s0 + args.per_sheet]
        sheet = Image.new("RGB", (W, rh * len(chunk)), (255, 255, 255))
        dr = ImageDraw.Draw(sheet)
        for j, (st, r) in enumerate(chunk):
            gid = r["gt"][0] if r["gt"] else None
            vn_shot = gid.split("-")[1] if gid else None
            plat = EVAL_SETS[r["set"]]
            path = EVAL_DATA / plat / "shots" / (vn_shot + ".jpg") if vn_shot else None
            y = j * rh
            if path and path.exists():
                img = np.asarray(Image.open(path).convert("RGB"))
                x0, y0, x1, y1 = r["crop"]
                crop = img[max(0, y0 - 2):y1 + 2, max(0, x0 - 2):x1 + 2]
                f = 36 / max(1, crop.shape[0])
                if crop.shape[1] * f > 1100:
                    f = 1100 / crop.shape[1]
                im = Image.fromarray(crop).resize((max(1, int(crop.shape[1] * f)), max(1, int(crop.shape[0] * f))))
                sheet.paste(im, (4, y))
                cy0, cy1 = max(0, y0 - 30), min(img.shape[0], y1 + 30)
                ctx = Image.fromarray(img[cy0:cy1, :]).copy()
                cd = ImageDraw.Draw(ctx)
                cd.rectangle([x0, y0 - cy0, x1, y1 - cy0], outline=(255, 0, 0))
                f2 = 70 / max(1, ctx.height)
                ctx = ctx.resize((max(1, int(ctx.width * f2)), 70))
                sheet.paste(ctx, (W - ctx.width - 4, y + 36 + 66 - 70 + 70 - 70))
            dr.text((4, y + 38), f"[{s0 + j}] {st} {r['set']} {gid}", fill=(200, 0, 0), font=font)
            dr.text((4, y + 60), f"label {r['label']}", fill=(0, 0, 160), font=font)
            dr.text((4, y + 82), f"truth {r['truth']}", fill=(0, 110, 0), font=font)
            dr.text((4, y + 104), (f"{r['how']} " + " | ".join(f"{k}:{v}" for k, v in r["reads"].items()))[:110],
                    fill=(90, 90, 90), font=ImageFont.truetype(str(FONT), 13))
        p = out / f"sheet_{s0 // args.per_sheet:02d}.png"
        sheet.save(p)
        print(p, sheet.size)


if __name__ == "__main__":
    main()
