#!/usr/bin/env python3
"""The hand check of MeikiOCR's reference boxes (docs/char-boxes.md "Hand-checked set").

  sheets   a seeded sample of development dialogue lines with reference boxes (charbox_sets units + meiki),
           drawn on contact sheets (<= 1280 px a side): each line's view enlarged, MeikiOCR's box of every
           reference character as a coloured frame (alternating colours) with its index, the reference text
           below. Only MeikiOCR's boxes are drawn, never an estimate of the method being scored.
  record   verdicts (one JSON line per sheet line: {"uid", "verdict": "ok" | "drop", "bad": [indices]}) into
           the hand-check file, from a compact text form: one line per sheet row "k ok", "k ok 3,7" (boxes 3
           and 7 wrong) or "k drop" (the line cannot be judged).

  .venv-ci/bin/python eval/charbox_hand.py sheets --units U --meiki M --out DIR [--per-part 48 --rows 12]
  .venv-ci/bin/python eval/charbox_hand.py record --sheet DIR/S00.json --verdicts "0 ok;1 ok 4;2 drop" --out H
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "synth"))
import charbox_sets as CS  # noqa: E402

SEED = 20261023
COLOURS = [(255, 64, 64), (64, 220, 255), (255, 210, 0), (120, 255, 120)]


def font(size):
    from PIL import ImageFont
    for p in ("/local/home/skerraut/herd-vita-ocr/shared/data/fonts/notosansjp",):
        fs = sorted(Path(p).glob("*.ttf")) + sorted(Path(p).glob("*.otf"))
        if fs:
            return ImageFont.truetype(str(fs[0]), size)
    return ImageFont.load_default()


def cmd_sheets(args):
    from PIL import Image, ImageDraw
    units = {u["uid"]: u for u in CS.jsonl(args.units)}
    gts = {}
    for r in CS.jsonl(args.meiki):
        u = units[r["uid"]]
        b, _ = CS.reference_boxes(u, r["meiki_text"], r["meiki_chars"])
        if any(x is not None for x in b):
            gts[r["uid"]] = b
    rng = np.random.default_rng(SEED)
    picks = []
    for part in args.parts:
        pool = sorted(uid for uid, u in units.items() if uid in gts and u["part"] == part and u["cls"] == "dialogue"
                      and not u["vertical"])
        picks += [pool[i] for i in rng.permutation(len(pool))[:args.per_part]]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    f_lab = font(16)
    for s in range(0, len(picks), args.rows):
        rows, meta = [], []
        for k, uid in enumerate(picks[s:s + args.rows]):
            u = units[uid]
            rgb = CS.frame_rgb(u["frame"])
            H, W = rgb.shape[:2]
            vx, vy, vw, vh = CS.view_box(u, (W, H))
            crop = Image.fromarray(rgb[vy:vy + vh, vx:vx + vw])
            z = min(max(1.0, 40.0 / max(1, u["box"][3])), 1240.0 / max(1, vw))
            im = crop.resize((max(1, round(vw * z)), max(1, round(vh * z))), Image.NEAREST)
            canvas = Image.new("RGB", (1280, im.height + 46), (24, 24, 24))
            canvas.paste(im, (30, 18))
            d = ImageDraw.Draw(canvas)
            d.text((2, 18), str(k), fill=(255, 255, 255), font=f_lab)
            for i, b in enumerate(gts[uid]):
                if b is None:
                    continue
                x0, y0, x1, y1 = [(b[0] - vx) * z + 30, (b[1] - vy) * z + 18, (b[2] - vx) * z + 30,
                                  (b[3] - vy) * z + 18]
                c = COLOURS[i % len(COLOURS)]
                d.rectangle((x0, y0, x1 - 1, y1 - 1), outline=c)
                d.text((x0 + 1, 1 if i % 2 == 0 else y1 + 1), str(i), fill=c, font=f_lab)
            d.text((30, im.height + 26), f"{u['part']} {u['text']}", fill=(255, 255, 160), font=f_lab)
            rows.append(canvas)
            meta.append({"k": k, "uid": uid, "boxes": gts[uid]})
        height = min(1280, sum(r.height for r in rows))
        sheet = Image.new("RGB", (1280, height), (0, 0, 0))
        y = 0
        for r in rows:
            if y + r.height > 1280:
                break
            sheet.paste(r, (0, y))
            y += r.height
        name = f"S{s // args.rows:02d}"
        sheet.save(out / f"{name}.png")
        (out / f"{name}.json").write_text(json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8")
        print(name, len(meta), "lines", f"{sheet.width}x{sheet.height}")


def cmd_record(args):
    meta = json.loads(Path(args.sheet).read_text(encoding="utf-8"))
    by_k = {m["k"]: m for m in meta}
    out = []
    for part in [p.strip() for p in args.verdicts.split(";") if p.strip()]:
        tok = part.split()
        k, verdict = int(tok[0]), tok[1]
        bad = [int(v) for v in tok[2].split(",")] if len(tok) > 2 else []
        if verdict not in ("ok", "drop") or k not in by_k:
            raise SystemExit(f"bad verdict {part!r}")
        out.append({"uid": by_k[k]["uid"], "sheet": Path(args.sheet).stem, "verdict": verdict, "bad": bad,
                    "reviewer_kind": "assistant"})
    if len(out) != len(meta):
        raise SystemExit(f"{len(out)} verdicts for {len(meta)} lines")
    with open(args.out, "a", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"{len(out)} verdicts -> {args.out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("sheets")
    a.add_argument("--units", required=True)
    a.add_argument("--meiki", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--parts", nargs="+", default=["vita", "psp"])
    a.add_argument("--per-part", type=int, default=48)
    a.add_argument("--rows", type=int, default=12)
    a = sub.add_parser("record")
    a.add_argument("--sheet", required=True)
    a.add_argument("--verdicts", required=True)
    a.add_argument("--out", required=True)
    args = ap.parse_args()
    {"sheets": cmd_sheets, "record": cmd_record}[args.cmd](args)


if __name__ == "__main__":
    main()
