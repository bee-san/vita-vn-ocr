#!/usr/bin/env python3
"""Candidate text lines for hand-checking the test sets (build tool, run via heavy.sh).

  candidates.py vita|psp

For every screenshot in data/eval/<platform>/shots.jsonl whose release includes
Japanese: PP-OCRv5 server det finds text boxes, and each axis-aligned box crop
(2 px margin) is read by PP-OCRv5 server rec (a), manga-ocr on the re-flowed
block (b) and PP-OCRv6 small rec (c). For Vita shots that data/label_real.py
already processed, its crop, readings and pseudo-label are attached to the box
they overlap (p). Output: data/eval/<platform>/cands.jsonl, one row per shot.
Resumable: shots already in the output are skipped.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
EVAL = HERE.parent
ROOT = EVAL.parent
sys.path.insert(0, str(EVAL))
from backends import MangaOcrRec, PaddleOnnx, find_model, threads  # noqa: E402
from fetch import data_dir  # noqa: E402


def load_label_real():
    spec = importlib.util.spec_from_file_location("label_real", ROOT / "data" / "label_real.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    iw, ih = max(0, min(ax1, bx1) - max(ax0, bx0)), max(0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    return inter / max(1, (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter)


def has_jp(s):
    return any(0x3040 <= ord(c) <= 0x30FF or 0x4E00 <= ord(c) <= 0x9FFF for c in s)


def main(which):
    from rapidocr import ModelType, OCRVersion, RapidOCR
    data = data_dir()
    det = RapidOCR(params={"Det.ocr_version": OCRVersion.PPOCRV5, "Det.model_type": ModelType.SERVER,
                           "Global.log_level": "error", "Global.use_cls": False, "Global.use_rec": False,
                           "EngineConfig.onnxruntime.intra_op_num_threads": threads()})
    rec_a = PaddleOnnx(find_model("ch_PP-OCRv5_rec_server.onnx"), max_w=4096)
    rec_c = PaddleOnnx(find_model("PP-OCRv6_rec_small.onnx"), max_w=4096)
    rec_b = MangaOcrRec()
    lr = load_label_real()
    old = {}
    real = ROOT / "data" / "real" / "lines.jsonl"
    if which == "vita" and real.exists():
        for line in real.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            old.setdefault(r["shot"], []).append(r)
    shots = [json.loads(x) for x in (data / which / "shots.jsonl").read_text(encoding="utf-8").splitlines()]
    out_path = data / which / "cands.jsonl"
    done = set()
    if out_path.exists():
        done = {json.loads(x)["shot"] for x in out_path.read_text(encoding="utf-8").splitlines()}
    t0 = time.time()
    with open(out_path, "a", encoding="utf-8") as out:
        for n, s in enumerate(shots):
            if s["shot"] in done or "ja" not in s["langs"]:
                continue
            bgr = cv2.imread(str(data / which / "shots" / s["shot"]))
            H, W = bgr.shape[:2]
            res = det(bgr, use_det=True, use_cls=False, use_rec=False)
            quads = getattr(res, "boxes", None)
            quads = [] if quads is None else quads
            boxes = []
            for q in quads:
                q = np.asarray(q, np.float32)
                x0, y0 = np.floor(q.min(0)).astype(int)
                x1, y1 = np.ceil(q.max(0)).astype(int)
                x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
                if x1 - x0 < 4 or y1 - y0 < 4:
                    continue
                b = {"bbox": [int(x0), int(y0), int(x1), int(y1)], "quad": q.round(1).tolist()}
                if (x1 - x0) >= 0.9 * (y1 - y0):                  # horizontal text only
                    m = 2
                    crop = bgr[max(0, y0 - m):min(H, y1 + m), max(0, x0 - m):min(W, x1 + m), ::-1]
                    crop = np.ascontiguousarray(crop)
                    b["a"] = rec_a.read([crop])[0]
                    b["c"] = rec_c.read([crop])[0]
                    b["b"] = rec_b.read([crop])[0] if has_jp(b["a"] + b["c"]) else ""
                boxes.append(b)
            for r in old.get(s["shot"], []):
                rx, ry = r["region"][:2]
                lx, ly, lw, lh = r["box"]
                ob = (rx + lx, ry + ly, rx + lx + lw, ry + ly + lh)
                best = max(boxes, key=lambda b: iou(ob, b["bbox"]), default=None)
                rec = {"id": r["id"], "box": list(ob), "a": r["a"], "b": r["b"], "agree": r["agree"],
                       "label": r["label"], "decided": lr.decide(r["a"], r["b"]), "split": r["split"]}
                if best is not None and iou(ob, best["bbox"]) > 0.3:
                    best.setdefault("p", []).append(rec)
                else:
                    boxes.append({"bbox": list(ob), "p": [rec], "orphan": True})
            boxes.sort(key=lambda b: (b["bbox"][1], b["bbox"][0]))
            out.write(json.dumps({"shot": s["shot"], "vn": s["vn"], "dims": [W, H], "boxes": boxes},
                                 ensure_ascii=False) + "\n")
            out.flush()
            if n % 10 == 0:
                print(f"{n}/{len(shots)} {time.time() - t0:.0f}s", flush=True)
    print("candidates: done", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
