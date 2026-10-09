#!/usr/bin/env python3
"""Pseudo-label real PS Vita VN screenshots with two strong OCRs.

For each data/vndb/shots/*.jpg, PP-OCRv5 (mobile det) locates text and the
dialogue region is the union of text lines in the lower part of the screen, i.e.
the box a user would draw with Square. common.find_lines then crops its lines
exactly as the overlay does. When the finder rejects the union (light frame,
UI), each detected line is tried as its own region instead.

Each crop is read by PP-OCRv5 server rec and by manga-ocr. manga-ocr squashes its
input to 224x224, which wrecks long horizontal lines, so a long crop is cut at
inter-character gaps and re-flowed into a square-ish multi-row block first. A line
becomes a training label only when both readings agree after normalisation.
Lines that are mostly ASCII (English releases, copyright strings) are skipped.

Outputs in data/real/: lines.jsonl (every crop and both readings), crops/<id>.png
(RGB crop) and real_train.npz (agreed train-VN lines, render.py shard format).
Resumable: screenshots already in lines.jsonl or skipped.jsonl are skipped.
"""
import json
import math
import re
import sys
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "synth"))
from build_corpus import norm  # noqa: E402
from common import DATA, find_lines, is_test_vn, to_gray  # noqa: E402

OUT = DATA / "real"
THREADS = 4


def canon(s):
    """Label form matching the corpus conventions (full-width !? and ～, … for ...)."""
    s = norm(s)
    for dots in ("...", "．．．", "・・・"):
        s = s.replace(dots, "…")
    return s.translate(str.maketrans({"!": "！", "?": "？", "~": "～"}))


def agree_key(s):
    return unicodedata.normalize("NFKC", canon(s))


SMALL = str.maketrans("ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ", "あいうえおつやゆよわアイウエオツヤユヨワカケ")
ELLIPSIS = re.compile(r"[…‥．・.。]*[…‥．・.][…‥．・.。]*")


def loose_key(s):
    """Agreement key that ignores ellipsis spelling and small-kana size."""
    return ELLIPSIS.sub("…", agree_key(s)).translate(SMALL)


def decide(a, b):
    """Label for one crop from the PP-OCRv5 (a) and manga-ocr (b) readings, or None.
    Exact agreement uses a's spelling; agreement up to ellipsis/small-kana takes
    b's kana (PP-OCRv5 often reads っ as つ) with a's ellipses."""
    if not a or mostly_ascii(a):
        return None
    if agree_key(a) == agree_key(b):
        return canon(a)
    if loose_key(a) != loose_key(b):
        return None
    ca, cb = canon(a), canon(b)
    if ELLIPSIS.sub("…", ca) == ELLIPSIS.sub("…", cb):   # only the ellipsis spelling differs
        return ca
    if len(ca) != len(cb):
        return None
    out = []                                              # only kana size differs
    for x, y in zip(ca, cb):
        if x != y and x.translate(SMALL) != y.translate(SMALL):
            return None
        out.append(y)
    return "".join(out)


def mostly_ascii(s):
    s = s.strip()
    return not s or sum(ord(c) < 128 for c in s) > 0.5 * len(s)


def candidate_boxes(boxes, h_img):
    cands = []
    for b in boxes:
        (x0, y0), (x1, y1) = b.min(0), b.max(0)
        bw, bh = x1 - x0, y1 - y0
        if not 12 <= bh <= 70 or bw < 1.6 * bh:          # horizontal text lines only
            continue
        if (y0 + y1) / 2 < 0.55 * h_img:                 # dialogue sits low on screen
            continue
        if y1 > h_img - 4 and bh < 30:                   # bottom-edge UI strip
            continue
        cands.append((int(x0), int(y0), int(x1), int(y1)))
    return cands


def expand(box, w_img, h_img, m=6):
    x0, y0, x1, y1 = box
    x0, y0, x1, y1 = max(0, x0 - m), max(0, y0 - m), min(w_img, x1 + m), min(h_img, y1 + m)
    return x0, y0, x1 - x0, y1 - y0


def regions_and_lines(bgr, cands):
    """[(region, [line boxes])]: the union region if the finder accepts it,
    otherwise every candidate line on its own."""
    h_img, w_img = bgr.shape[:2]
    union = (min(c[0] for c in cands), min(c[1] for c in cands),
             max(c[2] for c in cands), max(c[3] for c in cands))
    x, y, w, h = expand(union, w_img, h_img)
    lines = find_lines(bgr[y:y + h, x:x + w, ::-1])
    if lines:
        return [((x, y, w, h), lines)], "union"
    out = []
    for c in cands:
        x, y, w, h = expand(c, w_img, h_img)
        lines = find_lines(bgr[y:y + h, x:x + w, ::-1])
        if lines:
            out.append(((x, y, w, h), lines))
    return out, "per-line"


def blockify(rgb):
    """Re-flow a long one-line crop into k stacked rows so manga-ocr's 224x224
    resize keeps characters legible. Cuts land on the darkest column near each
    k-th of the width, i.e. a gap between glyphs."""
    h, w = rgb.shape[:2]
    k = int(round(math.sqrt(w / max(h, 1))))
    if k < 2:
        return rgb
    ink = (rgb.min(axis=2) >= 160).sum(axis=0).astype(np.float32)
    cuts = [0]
    for i in range(1, k):
        t = int(i * w / k)
        lo, hi = max(cuts[-1] + h // 2, t - int(0.6 * h)), min(w - h // 2, t + int(0.6 * h))
        if hi <= lo:
            continue
        cuts.append(lo + int(np.argmin(ink[lo:hi])))
    cuts.append(w)
    pieces = [rgb[:, a:b] for a, b in zip(cuts, cuts[1:]) if b > a]
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    bg = np.median(border, axis=0).astype(np.uint8)
    width, gap = max(p.shape[1] for p in pieces), max(4, h // 4)
    block = np.empty((len(pieces) * (h + gap) + gap, width + 2 * gap, 3), np.uint8)
    block[:] = bg
    for i, p in enumerate(pieces):
        y = gap + i * (h + gap)
        block[y:y + h, gap:gap + p.shape[1]] = p
    return block


def make_engine():
    from rapidocr import ModelType, OCRVersion, RapidOCR
    params = {"Det.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType.MOBILE,
              "Rec.ocr_version": OCRVersion.PPOCRV5, "Rec.model_type": ModelType.SERVER,
              "Global.log_level": "error"}
    try:
        return RapidOCR(params={**params, "EngineConfig.onnxruntime.intra_op_num_threads": THREADS})
    except Exception as e:  # older/newer config schema: fall back to defaults
        print(f"thread option rejected ({e}); using default threads", flush=True)
        return RapidOCR(params=params)


def write_train_npz(records):
    crops, labels = [], []
    for r in records:
        label = decide(r["a"], r["b"]) if r["split"] == "train" else None
        if label:
            rgb = np.asarray(Image.open(OUT / "crops" / f"{r['id']:05d}.png").convert("RGB"))
            crops.append(to_gray(rgb))
            labels.append(label)
    if not crops:
        return 0
    shapes = np.array([c.shape for c in crops], np.int32)
    offsets = np.concatenate([[0], np.cumsum(shapes[:, 0] * shapes[:, 1])]).astype(np.int64)
    np.savez_compressed(OUT / "real_train.npz", data=np.concatenate([c.ravel() for c in crops]),
                        offsets=offsets, shapes=shapes,
                        labels=np.array(json.dumps(labels, ensure_ascii=False)),
                        meta=np.array(json.dumps([{"real": True}] * len(labels))))
    return len(crops)


def main():
    torch.set_num_threads(THREADS)
    (OUT / "crops").mkdir(parents=True, exist_ok=True)
    manifest, skipped_path = OUT / "lines.jsonl", OUT / "skipped.jsonl"
    records = [json.loads(x) for x in manifest.read_text().splitlines()] if manifest.exists() else []
    skipped = {json.loads(x)["shot"] for x in skipped_path.read_text().splitlines()} \
        if skipped_path.exists() else set()
    done = {r["shot"] for r in records} | skipped
    next_id = max((r["id"] for r in records), default=-1) + 1

    eng = make_engine()
    from manga_ocr import MangaOcr
    mocr = MangaOcr()
    shots = sorted((DATA / "vndb" / "shots").glob("*.jpg"))
    t0 = time.time()
    with open(manifest, "a") as man, open(skipped_path, "a") as skp:
        for n, path in enumerate(shots):
            if path.name in done:
                continue
            vn = path.name.split("_")[0]
            bgr = cv2.imread(str(path))
            # rapidocr remembers use_det/use_rec between calls: always pass all three.
            det = eng(bgr, use_det=True, use_cls=False, use_rec=False)
            boxes = getattr(det, "boxes", None)
            cands = candidate_boxes(boxes, bgr.shape[0]) if boxes is not None else []
            found, mode = regions_and_lines(bgr, cands) if cands else ([], "none")
            if not found:
                skp.write(json.dumps({"shot": path.name, "why": "no-text" if not cands
                                      else "finder-rejected"}) + "\n")
                skp.flush()
                continue
            for (x, y, w, h), lines in found:
                for (lx, ly, lw, lh) in lines:
                    crop = bgr[y + ly:y + ly + lh, x + lx:x + lx + lw]
                    ra = eng(crop, use_det=False, use_cls=False, use_rec=True)
                    txts, scores = getattr(ra, "txts", None), getattr(ra, "scores", None)
                    a = txts[0] if txts else ""
                    if mostly_ascii(a):
                        continue
                    b = mocr(Image.fromarray(blockify(crop[:, :, ::-1])))
                    ok = agree_key(a) == agree_key(b)
                    rec = {"id": next_id, "shot": path.name, "vn": vn,
                           "split": "test" if is_test_vn(vn) else "train", "mode": mode,
                           "region": [x, y, w, h], "box": [lx, ly, lw, lh], "a": a,
                           "a_score": round(float(scores[0]) if scores else 0.0, 4),
                           "b": b, "agree": ok, "label": canon(a) if ok else None}
                    cv2.imwrite(str(OUT / "crops" / f"{next_id:05d}.png"), crop)
                    man.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    records.append(rec)
                    next_id += 1
            man.flush()
            if n % 25 == 0:
                agreed = sum(r["agree"] for r in records)
                print(f"{n}/{len(shots)} shots, {len(records)} lines, {agreed} agreed, "
                      f"{time.time() - t0:.0f}s", flush=True)
    n_train = write_train_npz(records)
    test = [r for r in records if r["split"] == "test"]
    print(f"done: {len(records)} lines ({sum(r['agree'] for r in records)} agreed), "
          f"real_train.npz {n_train} lines, test lines {len(test)}", flush=True)
    print("LABEL_DONE", flush=True)


if __name__ == "__main__":
    if "--rescore" in sys.argv:
        recs = [json.loads(x) for x in (OUT / "lines.jsonl").read_text().splitlines()]
        n = write_train_npz(recs)
        print(f"rescored {len(recs)} lines: real_train.npz {n} lines, exact agree "
              f"{sum(r['agree'] for r in recs)}", flush=True)
    else:
        main()
