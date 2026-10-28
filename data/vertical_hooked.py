"""Console columns: the column finder's own crops of the owner's hooked vertical pages, labelled exactly (issue #159).

  vertical_hooked.py cil --pagepool DIR/PCSG00653.jsonl --pages DIR --splits train --out OUT.npz [--procs 5]
  vertical_hooked.py sheet --shard OUT.npz --out SHEET.png [--n 40] [--seed 0]

The vertical counterpart of data/console_crops.py (docs/console-crops.md, VOCR6's adopted crop-in-the-loop factor
S2cil). For every page of the hooked page pool (vocr6-ingest pagepool, schema vocr6-hooked/1) whose lines are all
vertical columns, the region is the "tight" unit Rubi's tools/vertical_eval.py scores (the union of the page's
dialogue columns plus 12 px). The column finder P1 runs on it as Rubi runs it (synth/vertical_twin.find_columns,
VOCR_COLUMNS_VALLEYS: the bit-exact twin of runtime/vocr_vertical.c). Each found column is matched to the page's
hooked columns by their glyph cells (the record's cell_boxes): a found box is *whole* when it holds the centre of
every cell of exactly one column and no cell centre of another column and no centre of another renderer text box
(a speaker's name column, an unverified line); only whole boxes are kept, labelled with that column's hooked text.
Partial and merged boxes have no exact label and are left out (counted).

A kept crop is what the console's job hands the recognizer: the box's grey pixels (the finder's own 3-px pad),
turned 90 degrees anticlockwise (vertical_twin.rotate_ccw) and inverted when the twin calls its text dark
(vocr_line_dark on the box, as the job computes it with VOCR_COLUMNS_NO_DARK). Shards are train/train2.py's format
(data / offsets / shapes / labels / meta) at native size. Everything read and written here is private
(copyrighted game frames and their text): the shards stay outside git.
"""
import argparse
import json
import multiprocessing as mp
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "synth"))

PAD = 12              # Rubi tools/vertical_eval.py: a tight region is its columns' union plus 12 px (Vita)
VALLEYS = 1           # runtime/vocr_vertical.h VOCR_COLUMNS_VALLEYS (P1, the selected finder; Rubi's)


def clip_box(x0, y0, x1, y1, w, h):
    """Rubi tools/vertical_eval.py clip_box: [x, y, w, h] inside the frame, at least 1 px."""
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(round(x1))), min(h, int(round(y1)))
    return [x0, y0, max(1, x1 - x0), max(1, y1 - y0)]


def tight_region(cols, w, h):
    return clip_box(min(c["box"][0] for c in cols) - PAD, min(c["box"][1] for c in cols) - PAD,
                    max(c["box"][0] + c["box"][2] for c in cols) + PAD,
                    max(c["box"][1] + c["box"][3] for c in cols) + PAD, w, h)


def inside(pt, box):
    x, y = pt
    return box[0] <= x < box[0] + box[2] and box[1] <= y < box[1] + box[3]


def centre(xyxy):
    return ((xyxy[0] + xyxy[2]) / 2.0, (xyxy[1] + xyxy[3]) / 2.0)


def classify(box, cols, others):
    """('whole', column index) | ('partial' | 'merged' | 'empty' | 'other', None) for a found box in page pixels.
    cols: [{"cells": [(cx, cy)]}], others: centres of the page's other renderer text boxes."""
    held = [sum(inside(p, box) for p in c["cells"]) for c in cols]
    hit = [i for i, n in enumerate(held) if n]
    if any(inside(p, box) for p in others):
        return ("other", None)
    if not hit:
        return ("empty", None)
    if len(hit) > 1:
        return ("merged", None)
    i = hit[0]
    return ("whole", i) if held[i] == len(cols[i]["cells"]) else ("partial", None)


_PAGES = None


def _init(pages):
    global _PAGES
    _PAGES = Path(pages)


def page_crops(rec):
    """The kept console columns of one page record: (crops, labels, metas, counts)."""
    from PIL import Image

    import vertical_twin as vt
    lines = [ln for ln in rec["lines"] if not ln.get("drop")]
    cnt = Counter()
    if not lines or not all(ln.get("vertical") for ln in lines) or rec.get("untrusted"):
        cnt["page_skipped"] += 1
        return [], [], [], cnt
    w, h = rec["dims"]
    covered = {i for ln in lines for i in ln.get("box_index") or []}
    others = [centre(b["xyxy"]) for b in rec["boxes"] if b["index"] not in covered]
    cols = []
    for ln in lines:
        cells = ln.get("cell_boxes") or []
        if not cells:   # no glyph cells: the column's box stands for its one cell set
            x, y, bw, bh = ln["box"]
            cells = [[x, y, x + bw, y + bh]]
        cols.append({"text": ln["text"], "id": ln["id"], "box": ln["box"], "cells": [centre(c) for c in cells]})
    rx, ry, rw, rh = tight_region(cols, w, h)
    rgb = np.asarray(Image.open(_PAGES / rec["image"]).convert("RGB"))
    region = np.ascontiguousarray(rgb[ry:ry + rh, rx:rx + rw])
    g = vt.grey_of(region)
    boxes, dark, _ = vt.find_columns(region, VALLEYS)
    crops, labels, metas, used = [], [], [], set()
    for b, d in zip(boxes, dark):
        page_box = (b[0] + rx, b[1] + ry, b[2], b[3])
        kind, i = classify(page_box, cols, others)
        cnt[kind] += 1
        if kind != "whole" or i in used:
            if kind == "whole":
                cnt["whole_dup"] += 1
            continue
        used.add(i)
        x, y, bw, bh = b
        crop = vt.rotate_ccw(g[y:y + bh, x:x + bw])
        if d:
            crop = 255 - crop
        crops.append(np.ascontiguousarray(crop, np.uint8))
        labels.append(cols[i]["text"])
        metas.append({"real": True, "pool": "hooked-v1-console-columns", "id": cols[i]["id"],
                      "title_id": rec["title_id"], "page": rec["page"], "split": rec["split"], "platform": "vita",
                      "region": [rx, ry, rw, rh], "box": list(page_box), "dark": int(d), "inverted": bool(d),
                      "image_sha256": rec["image_sha256"], "source": "find_columns_p1_whole"})
    cnt["columns"] += len(cols)
    cnt["columns_whole"] += len(used)
    cnt["pages"] += 1
    return crops, labels, metas, cnt


def cmd_cil(a):
    splits = set(a.splits.split(","))
    recs = [json.loads(line) for line in open(a.pagepool, encoding="utf-8") if line.strip()]
    recs = [r for r in recs if r["split"] in splits]
    crops, labels, metas, cnt = [], [], [], Counter()
    with mp.get_context("spawn").Pool(a.procs, initializer=_init, initargs=(a.pages,)) as pool:
        for c, lab, m, k in pool.imap(page_crops, recs, chunksize=4):
            crops += c
            labels += lab
            metas += m
            cnt.update(k)
    shapes = np.array([c.shape for c in crops], np.int32).reshape(-1, 2)
    offs = np.concatenate([[0], np.cumsum([c.size for c in crops])]).astype(np.int64)
    data = np.concatenate([c.ravel() for c in crops]) if crops else np.zeros(0, np.uint8)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, data=data, offsets=offs, shapes=shapes, labels=json.dumps(labels, ensure_ascii=False),
             meta=json.dumps(metas, ensure_ascii=False))
    tmp.replace(out)
    print(json.dumps({"out": str(out), "crops": len(crops), "splits": sorted(splits), "counts": dict(cnt)}))


def cmd_sheet(a):
    """A contact sheet of random crops, each turned back upright, numbered; its labels in SHEET.json (private)."""
    from PIL import Image, ImageDraw
    z = np.load(a.shard)
    labels = json.loads(str(z["labels"]))
    rng = np.random.default_rng(a.seed)
    idx = sorted(rng.choice(len(labels), size=min(a.n, len(labels)), replace=False).tolist())
    canvas = Image.new("L", (1280, 1280), 255)
    x, y, rowh, meta = 0, 0, 0, []
    for n, i in enumerate(idx):
        o0, o1 = int(z["offsets"][i]), int(z["offsets"][i + 1])
        h, w = (int(v) for v in z["shapes"][i])
        up = np.rot90(z["data"][o0:o1].reshape(h, w), -1)
        im = Image.fromarray(np.ascontiguousarray(up))
        if im.height > 560:
            im = im.resize((max(1, im.width * 560 // im.height), 560))
        if x + im.width + 8 > 1280:
            x, y, rowh = 0, y + rowh + 16, 0
        if y + im.height + 16 > 1280:
            break
        canvas.paste(im, (x + 2, y + 14))
        ImageDraw.Draw(canvas).text((x + 2, y), str(n + 1), fill=0)
        meta.append({"n": n + 1, "i": i, "text": labels[i]})
        x += im.width + 8
        rowh = max(rowh, im.height)
    canvas.save(a.out)
    Path(a.out).with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=0), encoding="utf-8")
    print(len(meta), "crops ->", a.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cil")
    p.add_argument("--pagepool", required=True)
    p.add_argument("--pages", required=True)
    p.add_argument("--splits", default="train")
    p.add_argument("--out", required=True)
    p.add_argument("--procs", type=int, default=5)
    p = sub.add_parser("sheet")
    p.add_argument("--shard", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    {"cil": cmd_cil, "sheet": cmd_sheet}[a.cmd](a)


if __name__ == "__main__":
    main()
