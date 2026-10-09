#!/usr/bin/env python3
"""Contact sheets for hand-checking (build tool). Every sheet is at most 1280 px.

  sheets.py lines vita START COUNT [--scale S]   crops of lines_cand.jsonl rows
                                                  START..START+COUNT-1, numbered, plus
                                                  their readings printed to stdout
  sheets.py shots vita START COUNT [--file F]    screenshots at half size (Vita) or
                                                  full size (PSP) with the line boxes
                                                  (green) and regions (yellow) of F
                                                  (default review.jsonl)
Output: data/eval/sheets/<platform>_<kind>_<START>.png
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from fetch import data_dir  # noqa: E402

MAX = 1280
FONT = sorted((HERE.parents[1] / "data" / "fonts" / "notosansjp").glob("*.ttf"))


def font(size):
    return ImageFont.truetype(str(FONT[0]), size) if FONT else ImageFont.load_default()


def parse_ids(spec):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def lines_sheet(which, start, count, scale, ids=None, tag=None):
    data = data_dir()
    rows = [json.loads(x) for x in (data / which / "lines_cand.jsonl").read_text(encoding="utf-8").splitlines()]
    sel = [(i, rows[i]) for i in ids] if ids else list(enumerate(rows))[start:start + count]
    f = font(18)
    tiles, cache = [], {}
    for idx, r in sel:
        if r["shot"] not in cache:
            cache = {r["shot"]: np.asarray(Image.open(data / which / "shots" / r["shot"]).convert("RGB"))}
        img = cache[r["shot"]]
        x, y, w, h = r["box"]
        H, W = img.shape[:2]
        crop = Image.fromarray(img[max(0, y - 4):min(H, y + h + 4), max(0, x - 4):min(W, x + w + 4)])
        s = scale
        if crop.width * s > MAX - 70:
            s = (MAX - 70) / crop.width
        crop = crop.resize((max(1, int(crop.width * s)), max(1, int(crop.height * s))), Image.LANCZOS)
        tiles.append((idx, crop))
    height = sum(t.height + 8 for _, t in tiles) + 4
    sheet = Image.new("RGB", (MAX, min(height, MAX)), (128, 128, 128))
    d = ImageDraw.Draw(sheet)
    yy = 4
    for idx, t in tiles:
        if yy + t.height > MAX:
            print(f"sheet full at {idx}; stopping", file=sys.stderr)
            break
        d.text((4, yy + max(0, t.height // 2 - 10)), str(idx), font=f, fill=(255, 255, 0))
        sheet.paste(t, (66, yy))
        yy += t.height + 8
        d.line([(0, yy - 4), (MAX, yy - 4)], fill=(255, 0, 255))
    sheet = sheet.crop((0, 0, MAX, min(yy, MAX)))
    out = data / "sheets" / f"{which}_lines_{tag or f'{start:04d}'}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    for idx, r in sel:
        fl = f" furi={r['furigana_text']}" if r.get("furigana") else ""
        ps = " | ".join(f"p:{p['decided'] or p['label']}" for p in r.get("p") or [])
        print(f"{idx} {r['key']} box={r['box']} hint={r['hint']}{fl}\n  a: {r['a']}\n  b: {r['b']}\n  c: {r['c']}"
              + (f"\n  {ps}" if ps else ""))
    print(out)


def shots_sheet(which, start, count, file, vns=None):
    data = data_dir()
    rows = [json.loads(x) for x in (data / which / file).read_text(encoding="utf-8").splitlines()]
    by = {}
    for r in rows:
        by.setdefault(r["shot"], []).append(r)
    if vns:
        allshots = [json.loads(x)["shot"] for x in (data / which / "shots.jsonl").read_text(
            encoding="utf-8").splitlines()]
        shots = sorted(s for s in allshots if s.split("_")[0] in vns)[start:start + count]
    else:
        shots = sorted(by)[start:start + count]
    scale = 0.5 if which == "vita" else 1.0
    tw, th = int((960 if which == "vita" else 480) * scale), int((544 if which == "vita" else 272) * scale)
    cols = 2
    rws = (len(shots) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * (tw + 6), rws * (th + 24)), (40, 40, 40))
    d = ImageDraw.Draw(sheet)
    f = font(14)
    for k, s in enumerate(shots):
        im = Image.open(data / which / "shots" / s).convert("RGB")
        im = im.resize((tw, th), Image.LANCZOS)
        dd = ImageDraw.Draw(im)
        for r in by.get(s, []):
            if r.get("drop"):
                continue
            x, y, w, h = r["box"]
            col = (255, 64, 64) if r.get("uncertain") else (0, 255, 0)
            dd.rectangle((x * scale, y * scale, (x + w) * scale, (y + h) * scale), outline=col)
            if r.get("region"):
                rx, ry, rw, rh = r["region"]
                dd.rectangle((rx * scale, ry * scale, (rx + rw) * scale, (ry + rh) * scale), outline=(255, 255, 0))
        cx, cy = (k % cols) * (tw + 6), (k // cols) * (th + 24)
        sheet.paste(im, (cx, cy + 20))
        d.text((cx + 2, cy + 2), f"{start + k}: {s}", font=f, fill=(255, 255, 255))
    out = data / "sheets" / f"{which}_shots_{start:04d}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    print(out, sheet.size)


def final_sheet(which, ids, tag, scale):
    """Decided lines as the eval crops them (final box + 4 px), with their GT printed."""
    data = data_dir()
    rows = [json.loads(x) for x in (data / which / "review.jsonl").read_text(encoding="utf-8").splitlines()]
    sel = [rows[i] for i in ids]
    f = font(18)
    tiles, cache = [], {}
    for r in sel:
        if r["shot"] not in cache:
            cache = {r["shot"]: np.asarray(Image.open(data / which / "shots" / r["shot"]).convert("RGB"))}
        img = cache[r["shot"]]
        x, y, w, h = r.get("box_override") or r["box"]
        H, W = img.shape[:2]
        crop = Image.fromarray(img[max(0, y - 4):min(H, y + h + 4), max(0, x - 4):min(W, x + w + 4)])
        s = min(scale, (MAX - 70) / crop.width)
        tiles.append((r["idx"], crop.resize((max(1, int(crop.width * s)), max(1, int(crop.height * s))),
                                            Image.LANCZOS)))
    height = min(MAX, sum(t.height + 8 for _, t in tiles) + 4)
    sheet = Image.new("RGB", (MAX, height), (128, 128, 128))
    d = ImageDraw.Draw(sheet)
    yy, shown = 4, 0
    for idx, t in tiles:
        if yy + t.height > MAX:
            break
        d.text((4, yy + max(0, t.height // 2 - 10)), str(idx), font=f, fill=(255, 255, 0))
        sheet.paste(t, (66, yy))
        yy += t.height + 8
        shown += 1
    out = data / "sheets" / f"{which}_final_{tag}.png"
    sheet.crop((0, 0, MAX, min(yy, MAX))).save(out)
    for r in sel[:shown]:
        print(f"{r['idx']} [{r['status']}] {r['text']}  {sorted(set(r.get('tags_add') or []))}")
    if shown < len(sel):
        print(f"(sheet full: {len(sel) - shown} not shown, first {sel[shown]['idx']})")
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["lines", "shots", "final"])
    ap.add_argument("which", choices=["vita", "psp"])
    ap.add_argument("start", type=int)
    ap.add_argument("count", type=int)
    ap.add_argument("--scale", type=float, default=None)
    ap.add_argument("--file", default="review.jsonl")
    ap.add_argument("--ids", default=None, help="explicit row ids, e.g. 3,5,10-14 (lines only)")
    ap.add_argument("--vn", default=None, help="shots mode: only these VN ids (comma list)")
    args = ap.parse_args()
    if args.kind == "final":
        final_sheet(args.which, parse_ids(args.ids), f"s{args.start:04d}", args.scale or
                    (1.0 if args.which == "vita" else 2.4))
    elif args.kind == "lines":
        ids = parse_ids(args.ids) if args.ids else None
        lines_sheet(args.which, args.start, args.count, args.scale or (1.3 if args.which == "vita" else 2.6),
                    ids, tag=f"s{args.start:04d}" if ids else None)
    else:
        shots_sheet(args.which, args.start, args.count, args.file,
                    set(args.vn.split(",")) if args.vn else None)


if __name__ == "__main__":
    main()
