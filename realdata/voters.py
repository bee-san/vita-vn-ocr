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

The models are pinned (MODELS, MANGA): the ONNX files by sha256, with their public source.
model_path() looks for a file in $REALDATA_MODELS if set, then in rapidocr's own model dir
(the installed package's rapidocr/models, where rapidocr 3.10.0 keeps them) or, for meiki,
the research campaign's copy, and checks the sha256 before a reader loads it. manga-ocr is
loaded from the Hugging Face snapshot at the pinned revision.

Threads follow OMP_NUM_THREADS (heavy.sh sets 5).
"""
import hashlib
import importlib.util
import math
import os
from pathlib import Path

import cv2
import numpy as np

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
RAPIDOCR_URL = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx/"
MEIKI_URL = ("https://huggingface.co/rtr46/meiki.txt.recognition.v0/resolve/"
             "a28cf5874dc2438ebb1c86336be26bcec51e3375/meiki.text.rec.v0.960x32.onnx")
# reader: (file name, sha256, public source). The PP-OCR hashes are the ones in rapidocr
# 3.10.0's default_models.yaml; meiki's is the Hugging Face LFS sha256 at that revision.
MODELS = {
    "v5s": ("ch_PP-OCRv5_rec_server.onnx", "e09385400eaaaef34ceff54aeb7c4f0f1fe014c27fa8b9905d4709b65746562a",
            RAPIDOCR_URL + "PP-OCRv5/rec/ch_PP-OCRv5_rec_server.onnx"),
    "v6s": ("PP-OCRv6_rec_small.onnx", "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884",
            RAPIDOCR_URL + "PP-OCRv6/rec/PP-OCRv6_rec_small.onnx"),
    "meiki": ("meiki.text.rec.v0.960x32.onnx", "3e96bc772fbee9717e536a6353032bb944c3382dd2f6960ef4890decda43b000",
              MEIKI_URL),
}
MANGA = ("kha-white/manga-ocr-base", "aa6573bd10b0d446cbf622e29c3e084914df9741")   # repo, revision


def model_dirs(name):
    dirs = [Path(os.environ["REALDATA_MODELS"])] if os.environ.get("REALDATA_MODELS") else []
    if name == "meiki":
        dirs.append(HERD / "shared" / "research" / "scratch" / "eval" / "models" / "meiki")
    else:
        spec = importlib.util.find_spec("rapidocr")
        if spec is not None and spec.origin:
            dirs.append(Path(spec.origin).parent / "models")
    return dirs


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_path(name):
    """The pinned model file of reader `name`, once its sha256 has been checked."""
    fname, sha, url = MODELS[name]
    dirs = model_dirs(name)
    for d in dirs:
        path = d / fname
        if path.is_file():
            got = file_sha256(path)
            if got != sha:
                raise RuntimeError(f"{name}: {path} has sha256 {got}, expected {sha} ({url})")
            return path
    raise FileNotFoundError(f"{name}: {fname} is not in {[str(d) for d in dirs]}; download {url} "
                            f"(sha256 {sha}) and point REALDATA_MODELS at its directory")


def threads():
    return max(1, int(os.environ.get("OMP_NUM_THREADS", "4")))


cv2.setNumThreads(threads())


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
        self.s = session(model_path(name))
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


def manga_snapshot():
    """Local directory of kha-white/manga-ocr-base at the pinned revision (the cache first)."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    try:
        return snapshot_download(MANGA[0], revision=MANGA[1], local_files_only=True)
    except LocalEntryNotFoundError:
        return snapshot_download(MANGA[0], revision=MANGA[1])


class MangaOCR:
    name = "manga"

    def __init__(self):
        import torch
        torch.set_num_threads(threads())
        from manga_ocr import MangaOcr
        self.m = MangaOcr(manga_snapshot())

    def __call__(self, bgr):
        from PIL import Image
        return self.m(Image.fromarray(blockify(bgr[:, :, ::-1]))).strip(), 1.0, []


class Meiki:
    """meiki text recognizer: per-character boxes and scores, de-duplicated by x overlap and
    read left to right (the model card's inference.py, threshold 0.1)."""
    name = "meiki"
    W, H, THR, OVERLAP = 960, 32, 0.1, 0.3

    def __init__(self):
        self.s = session(model_path("meiki"))

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
