"""The OCR readers ("voters") that pseudo-label real line crops.

  PaddleRec("v5s")  PP-OCRv5 server rec (rapidocr's ONNX), CTC, dict from the ONNX metadata
  PaddleRec("v6s")  PP-OCRv6 small rec (rapidocr's ONNX), same interface
  MangaOCR()        manga-ocr (kha-white/manga-ocr-base); long lines are re-flowed into a
                    square-ish block first, because it squashes its input to 224x224
  Meiki()           meiki.text.recognition.v0 (rtr46, LGPL-3.0): a D-FINE character detector
                    trained on Japanese video games; input padded to 960x32

Every reader takes a BGR uint8 line crop and returns (text, score, char_scores): the raw
reading, a line confidence in [0, 1] and one confidence per output character (manga-ocr
has no per-character scores and returns an empty list). Readers only label training
data: none of them ships on the console.

Threads follow OMP_NUM_THREADS (heavy.sh sets 5).
"""
import math
import os
from pathlib import Path

import cv2
import numpy as np

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
VENV_MODELS = HERD / "shared" / "venv" / "lib" / "python3.11" / "site-packages" / "rapidocr" / "models"
MODELS = {
    "v5s": VENV_MODELS / "ch_PP-OCRv5_rec_server.onnx",
    "v6s": VENV_MODELS / "PP-OCRv6_rec_small.onnx",
    "meiki": HERD / "shared" / "research" / "scratch" / "eval" / "models" / "meiki" / "meiki.text.rec.v0.960x32.onnx",
}


def threads():
    return max(1, int(os.environ.get("OMP_NUM_THREADS", "4")))


def session(path):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads()
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


class PaddleRec:
    """PP-OCR CTC recognizer, preprocessed like rapidocr: BGR, height 48, (x/255 - 0.5)/0.5,
    right-padded with zeros to at least 320 px."""

    def __init__(self, name):
        self.name = name
        self.s = session(MODELS[name])
        chars = self.s.get_modelmeta().custom_metadata_map["character"].split("\n")
        self.chars = [""] + chars + [" "]               # 0 = CTC blank, last = space
        n_out = self.s.get_outputs()[0].shape[-1]
        if n_out != len(self.chars):
            raise RuntimeError(f"{name}: {n_out} classes but {len(self.chars)} dict entries")

    def __call__(self, bgr):
        h, w = bgr.shape[:2]
        nw = max(8, int(math.ceil(48 * w / max(h, 1))))
        x = cv2.resize(bgr, (nw, 48), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        x = (x / 255.0 - 0.5) / 0.5
        pad = np.zeros((48, max(nw, 320), 3), np.float32)
        pad[:, :nw] = x
        y = self.s.run(None, {"x": pad.transpose(2, 0, 1)[None]})[0][0]   # T x C (softmaxed)
        idx, prob = y.argmax(1), y.max(1)
        text, scores, last = [], [], 0
        for k, p in zip(idx, prob):
            if k and k != last:
                text.append(self.chars[k])
                scores.append([float(p)])
            elif k and k == last:
                scores[-1].append(float(p))
            last = k
        cs = [float(np.mean(s)) for s in scores]
        return "".join(text).strip(), float(np.mean(cs)) if cs else 0.0, cs


def blockify(rgb):
    """Re-flow a long one-line crop into k stacked rows so manga-ocr's 224x224 resize keeps
    characters legible (as data/label_real.py). Cuts land on the column with the least
    contrast near each k-th of the width, i.e. a gap between glyphs, for either polarity."""
    h, w = rgb.shape[:2]
    k = int(round(math.sqrt(w / max(h, 1))))
    if k < 2:
        return rgb
    g = rgb.astype(np.int16).max(axis=2)
    bg = np.median(np.concatenate([g[0], g[-1], g[:, 0], g[:, -1]]))
    ink = (np.abs(g - bg) >= 48).sum(axis=0).astype(np.float32)
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
    bgc = np.median(border, axis=0).astype(np.uint8)
    width, gap = max(p.shape[1] for p in pieces), max(4, h // 4)
    block = np.empty((len(pieces) * (h + gap) + gap, width + 2 * gap, 3), np.uint8)
    block[:] = bgc
    for i, p in enumerate(pieces):
        y = gap + i * (h + gap)
        block[y:y + h, gap:gap + p.shape[1]] = p
    return block


class MangaOCR:
    name = "manga"

    def __init__(self):
        import torch
        torch.set_num_threads(threads())
        from manga_ocr import MangaOcr
        self.m = MangaOcr()

    def __call__(self, bgr):
        from PIL import Image
        return self.m(Image.fromarray(blockify(bgr[:, :, ::-1]))).strip(), 1.0, []


class Meiki:
    """meiki text recognizer: per-character boxes and scores, de-duplicated by x overlap and
    read left to right (the model card's inference.py, threshold 0.1)."""
    name = "meiki"
    W, H, THR, OVERLAP = 960, 32, 0.1, 0.3

    def __init__(self):
        self.s = session(MODELS["meiki"])

    def __call__(self, bgr):
        h, w = bgr.shape[:2]
        nw, nh = int(round(w * self.H / max(h, 1))), self.H
        if nw > self.W:
            nh, nw = max(1, int(round(nh * self.W / nw))), self.W
        x = np.zeros((self.H, self.W, 3), np.uint8)
        x[:nh, :nw] = cv2.resize(bgr[:, :, ::-1], (nw, nh), interpolation=cv2.INTER_LINEAR)
        inp = x.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        labels, boxes, scores = self.s.run(None, {"images": inp, "orig_target_sizes":
                                                  np.array([[self.W, self.H]], np.int64)})
        cand = []
        for lab, box, sc in zip(labels[0], boxes[0], scores[0]):
            if sc < self.THR:
                continue
            x1, x2 = min(float(box[0]), nw), min(float(box[2]), nw)
            cand.append((float(sc), x1, x2, chr(int(lab))))
        cand.sort(reverse=True)
        kept = []
        for sc, x1, x2, ch in cand:
            wc = x2 - x1 + 1e-6
            if all(max(0.0, min(x2, b) - max(x1, a)) / wc <= self.OVERLAP for _, a, b, _ in kept):
                kept.append((sc, x1, x2, ch))
        kept.sort(key=lambda c: c[1])
        cs = [c[0] for c in kept]
        return "".join(c[3] for c in kept), float(np.mean(cs)) if cs else 0.0, cs


def load(names):
    makers = {"v5s": lambda: PaddleRec("v5s"), "v6s": lambda: PaddleRec("v6s"),
              "manga": MangaOCR, "meiki": Meiki}
    return {n: makers[n]() for n in names}
