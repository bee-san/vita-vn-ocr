#!/usr/bin/env python3
"""Turn detector candidates into reviewable line candidates (build tool, heavy.sh).

  assemble.py vita|psp [--reread]

Reads data/eval/<platform>/cands.jsonl and writes lines_cand.jsonl, one row per
candidate line: fragments of one text line are merged, each box is refined to the
line's ink extent (polarity-agnostic contrast against the box border), small boxes
sitting just above a line are attached to it as furigana, and style hints are
computed. Merged lines, lines with furigana and label_real.py lines the detector
missed are re-read on their final eval crop (box + 4 px) by PP-OCRv5 server (a),
manga-ocr (b) and PP-OCRv6 small (c); other lines keep candidates.py's readings of
the detector box (--reread re-reads every line). The hand-check then decides
every line's text.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
sys.path.insert(0, str(EVAL))
from backends import MangaOcrRec, PaddleOnnx, find_model, to_gray  # noqa: E402
from fetch import data_dir  # noqa: E402
from metrics import norm  # noqa: E402

PAD = 4


def vov(a, b):
    """Vertical overlap of two x0,y0,x1,y1 boxes as a share of the smaller height."""
    inter = min(a[3], b[3]) - max(a[1], b[1])
    return max(0, inter) / max(1, min(a[3] - a[1], b[3] - b[1]))


def jp_ratio(s):
    s = s or ""
    jp = sum(0x3040 <= ord(c) <= 0x30FF or 0x4E00 <= ord(c) <= 0x9FFF or c in "々〆ー" for c in s)
    return jp / max(1, len(s))


def merge_fragments(boxes):
    """Merge det boxes that are pieces of one horizontal text line."""
    boxes = [dict(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i]["bbox"], boxes[j]["bbox"]
                h = max(a[3] - a[1], b[3] - b[1])
                gap = max(a[0], b[0]) - min(a[2], b[2])
                hratio = min(a[3] - a[1], b[3] - b[1]) / max(1, h)
                if vov(a, b) >= 0.6 and gap <= 1.2 * h and hratio >= 0.6:
                    u = [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]
                    boxes[i] = {"bbox": u, "parts": boxes[i].get("parts", [a]) + boxes[j].get("parts", [b]),
                                "p": (boxes[i].get("p") or []) + (boxes[j].get("p") or [])}
                    del boxes[j]
                    changed = True
                    break
            if changed:
                break
    return boxes


def _ink_box(m, bbox, win):
    """Box of the longest run of ink rows of mask m that touches the det box rows."""
    x0, y0, x1, y1 = bbox
    wx0, wy0 = win
    rows = np.flatnonzero(m.sum(1) >= max(2, 0.01 * m.shape[1]))
    if not len(rows):
        return None
    runs, s, last = [], rows[0], rows[0]
    for r in rows[1:]:
        if r - last > 3:
            runs.append((s, last))
            s = r
        last = r
    runs.append((s, last))
    by0, by1 = y0 - wy0, y1 - wy0
    runs = [r for r in runs if r[1] >= by0 and r[0] < by1]
    if not runs:
        return None
    r0, r1 = max(runs, key=lambda r: r[1] - r[0])
    cols = np.flatnonzero(m[r0:r1 + 1].sum(0) >= 1)
    if not len(cols):
        return None
    ib = [wx0 + int(cols[0]), wy0 + int(r0), wx0 + int(cols[-1]) + 1, wy0 + int(r1) + 1]
    if ib[3] - ib[1] < 0.45 * (y1 - y0) or ib[2] - ib[0] < 0.75 * (x1 - x0):
        return None
    sub = m[r0:r1 + 1, cols[0]:cols[-1] + 1]
    if sub.mean() < 0.03:
        return None
    return ib


def refine(rgb, bbox):
    """Ink extent of a det box. Three masks are tried: pixels that differ from the
    median of the window's border ring by more than a contrast threshold, near-white
    pixels and near-black pixels (each dilated by 1 px so outlines stay inside); the
    smallest valid box wins, so text over busy scenes still gets a tight box. Falls
    back to the det box."""
    import cv2
    H, W = rgb.shape[:2]
    x0, y0, x1, y1 = bbox
    wx0, wy0, wx1, wy1 = max(0, x0 - 3), max(0, y0 - 3), min(W, x1 + 3), min(H, y1 + 3)
    win = rgb[wy0:wy1, wx0:wx1]
    g = to_gray(win).astype(np.float32)
    if g.shape[0] < 6 or g.shape[1] < 6:
        return bbox, {}
    ring = np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]])
    bg = float(np.median(ring))
    d = np.abs(g - bg)
    t = max(28.0, 0.4 * float(np.percentile(d, 99)))
    k = np.ones((3, 3), np.uint8)
    masks = [d > t,
             cv2.dilate((win.min(2) >= 200).astype(np.uint8), k).astype(bool),
             cv2.dilate((win.max(2) <= 70).astype(np.uint8), k).astype(bool)]
    boxes = [b for b in (_ink_box(m, bbox, (wx0, wy0)) for m in masks) if b is not None]
    if not boxes:
        return bbox, {}
    ib = min(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
    m = masks[0]
    sub = rgb[ib[1]:ib[3], ib[0]:ib[2]].astype(np.float32)
    gm = g[ib[1] - wy0:ib[3] - wy0, ib[0] - wx0:ib[2] - wx0]
    mm = m[ib[1] - wy0:ib[3] - wy0, ib[0] - wx0:ib[2] - wx0]
    hint = {}
    if mm.sum() >= 10:
        brighter = float(((gm > bg) & mm).sum() / mm.sum())
        hint["bright_frac"] = round(brighter, 3)
        strong = mm & (np.abs(gm - bg) >= np.percentile(np.abs(gm - bg)[mm], 60))
        core = sub[strong] if strong.sum() >= 5 else sub[mm]
        hint["chroma"] = round(float((core.max(1) - core.min(1)).mean()), 1)
        hint["bg"] = round(bg, 1)
    return ib, hint


def attach_furigana(lines):
    """Small boxes directly above a bigger line become that line's furigana."""
    if not lines:
        return lines
    hs = sorted(b["bbox"][3] - b["bbox"][1] for b in lines)
    med = hs[len(hs) // 2]
    keep = []
    for b in lines:
        x0, y0, x1, y1 = b["bbox"]
        h = y1 - y0
        host = None
        if h <= 0.65 * med:
            for o in lines:
                if o is b:
                    continue
                ox0, oy0, ox1, oy1 = o["bbox"]
                oh = oy1 - oy0
                if h <= 0.65 * oh and 0 <= oy0 - y1 <= 0.5 * oh and x0 >= ox0 - 4 and x1 <= ox1 + 4:
                    host = o
                    break
        if host is None:
            keep.append(b)
        else:
            host.setdefault("furigana", []).append(b["bbox"])
            host.setdefault("furigana_text", []).append(b.get("c") or b.get("a") or "")
    for b in keep:
        for f in b.get("furigana", []):
            b["bbox"] = [min(b["bbox"][0], f[0]), min(b["bbox"][1], f[1]),
                         max(b["bbox"][2], f[2]), max(b["bbox"][3], f[3])]
    return keep


def main(which, reread=False):
    data = data_dir()
    rows = [json.loads(x) for x in (data / which / "cands.jsonl").read_text(encoding="utf-8").splitlines()]
    rec_a = PaddleOnnx(find_model("ch_PP-OCRv5_rec_server.onnx"), max_w=4096)
    rec_c = PaddleOnnx(find_model("PP-OCRv6_rec_small.onnx"), max_w=4096)
    rec_b = MangaOcrRec()
    out_path = data / which / "lines_cand.jsonl"
    n_out, t0 = 0, time.time()
    with open(out_path, "w", encoding="utf-8") as out:
        for k, r in enumerate(rows):
            rgb = np.asarray(Image.open(data / which / "shots" / r["shot"]).convert("RGB"))
            H, W = rgb.shape[:2]
            boxes = []
            for b in r["boxes"]:
                txt = (b.get("a") or "") + (b.get("c") or "") + "".join(p.get("a") or "" for p in b.get("p", []))
                if b.get("a") is None and not b.get("p"):
                    continue                       # vertical / unread box
                if jp_ratio(txt) < 0.3:
                    continue                       # English, digits, copyright strings
                boxes.append(b)
            boxes = merge_fragments(boxes)
            for b in boxes:
                b["det"] = list(b["bbox"])
                b["bbox"], b["hint"] = refine(rgb, b["bbox"])
            boxes = attach_furigana(boxes)
            boxes.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
            for i, b in enumerate(boxes):
                x0, y0, x1, y1 = b["bbox"]
                if not reread and "parts" not in b and b.get("a") is not None and not b.get("furigana"):
                    a, bb, c = b["a"], b.get("b") or "", b["c"]  # det-box readings, same text
                else:
                    crop = np.ascontiguousarray(rgb[max(0, y0 - PAD):min(H, y1 + PAD),
                                                    max(0, x0 - PAD):min(W, x1 + PAD)])
                    a, c = rec_a.read([crop])[0], rec_c.read([crop])[0]
                    bb = rec_b.read([crop])[0]
                votes = [norm(a), norm(bb), norm(c)]
                prop = c if votes[2] in (votes[0], votes[1]) or votes[0] != votes[1] else a
                row = {"key": f"{r['shot']}#{i}", "shot": r["shot"], "vn": r["vn"], "dims": [W, H],
                       "box": [x0, y0, x1 - x0, y1 - y0], "det": b["det"], "a": a, "b": bb, "c": c,
                       "prop": prop, "agree": len(set(votes)), "hint": b.get("hint", {}),
                       "furigana": b.get("furigana", []), "furigana_text": b.get("furigana_text", []),
                       "p": [{k2: p[k2] for k2 in ("id", "label", "decided", "split", "a", "b")}
                             for p in b.get("p") or []]}
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                n_out += 1
            out.flush()
            if k % 20 == 0:
                print(f"{k}/{len(rows)} shots, {n_out} lines, {time.time() - t0:.0f}s", flush=True)
    print(f"assemble: done, {n_out} lines", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], reread="--reread" in sys.argv)
