"""The OCR engines of the teacher labeller. All run locally on the CPU; none calls a cloud OCR.

  Meiki     MeikiOCR (rtr46/meikiocr 0.3.5, the engine owocr's `meikiocr` key wraps):
            meiki.text.detect.v0.1 finds text lines on the whole screenshot (scaled to fit
            960x544), meiki.text.recognition.v0 reads each one at 960x32 and returns one box
            and confidence per character. Called through meikiocr's own methods, but with
            our sessions (thread count) and the detector's box scores kept.
  PaddleRec PP-OCRv5 server rec ("v5s") and PP-OCRv6 small rec ("v6s"), rapidocr 3.10.0's
            ONNX files, CTC greedy with a confidence per character.
  MangaOCR  manga-ocr base (kha-white), long lines re-flowed into a square-ish block first
            (it squashes its input to 224x224), as data/label_real.py and eval do.
  ScreenAI  Chrome Screen AI through owocr's ChromeScreenAI (local library from Chromium's
            CIPD; "entirely on device and do not send any data to network"). label.py runs
            every engine inside `unshare -rn` (no network) anyway.

Model files are pinned by repository revision or sha256 (PINS) and live under
$TEACHER_MODELS (default $HERD/shared/artifacts/teacher/models); `label.py fetch` puts them
there. Threads follow OMP_NUM_THREADS (heavy.sh sets 5).

Every reader returns plain Python data so label.py can write it as JSON:
  Meiki.ocr(bgr)  -> [{"det": [x0, y0, x1, y1], "det_conf": p, "vertical": bool,
                       "text": str, "chars": [[c, x0, y0, x1, y1, conf], ...]}, ...]
  reader(bgr)     -> [text, mean char conf, min char conf]   (manga-ocr: [text])
  ScreenAI.ocr(bgr) -> [{"box": [x0, y0, x1, y1], "angle": a, "text": str, "conf": p,
                         "vertical": bool, "chars": [[c, x0, y0, x1, y1, conf], ...]}, ...]
"""
import hashlib
import math
import os
import unicodedata
from pathlib import Path

import numpy as np

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
MODELS = Path(os.environ.get("TEACHER_MODELS") or HERD / "shared" / "artifacts" / "teacher" / "models")

RAPID = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.10.0/onnx"
PINS = {
    # name: (source, file, sha256); hf = (repo, revision)
    "meiki_det": (("hf", "rtr46/meiki.text.detect.v0", "a9cffa4f60cbf72ddb87edf19c6f98a01cd042e6"),
                  "meiki.text.detect.v0.1.960x544.onnx",
                  "40b6a016667745cae7d3055929ae3b8b1e7716aac795f5904cd3c2c7c3b8404b"),
    "meiki_rec": (("hf", "rtr46/meiki.txt.recognition.v0", "a28cf5874dc2438ebb1c86336be26bcec51e3375"),
                  "meiki.text.rec.v0.960x32.onnx",
                  "3e96bc772fbee9717e536a6353032bb944c3382dd2f6960ef4890decda43b000"),
    "meiki_vrec": (("hf", "rtr46/meiki.txt.recognition.v0", "a28cf5874dc2438ebb1c86336be26bcec51e3375"),
                   "meiki.text.rec.v0.vertical.32x480.onnx",
                   "2c2a83a23bc3b7e6c63962175f507ecc6c5e85cc174f17bdec37d9bbd0bf895a"),
    "v5s": (("url", f"{RAPID}/PP-OCRv5/rec/ch_PP-OCRv5_rec_server.onnx"), "ch_PP-OCRv5_rec_server.onnx",
            "e09385400eaaaef34ceff54aeb7c4f0f1fe014c27fa8b9905d4709b65746562a"),
    "v6s": (("url", f"{RAPID}/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx"), "PP-OCRv6_rec_small.onnx",
            "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884"),
    "manga": (("hf", "kha-white/manga-ocr-base", "aa6573bd10b0d446cbf622e29c3e084914df9741"), None, None),
    "screenai": (("cipd", "chromium/third_party/screen-ai/linux"), "screen-ai-linux.zip",
                 "ac82d1f5c3beb856650fa8935ee22724451bd2f650b88d719ec4361cb1045646"),
}
UA = "vita-vn-ocr-teacher/0.1 (pseudo-labelling, low volume; github.com/bee-san/vita-vn-ocr)"


def threads():
    return max(1, int(os.environ.get("OMP_NUM_THREADS", "1")))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def model_path(name):
    """Local path of a pinned model file (fetch() must have run)."""
    src, fname, _ = PINS[name]
    if name == "manga":                     # the Hugging Face cache snapshot of the pinned revision
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(src[1], revision=src[2], local_files_only=True))
    if name == "screenai":
        return MODELS / "screen_ai" / PINS[name][2][:8]
    return MODELS / fname


def fetch(names=None, mirrors=()):
    """Download (or copy from a mirror directory) every pinned model and check its sha256."""
    import shutil
    import urllib.request
    MODELS.mkdir(parents=True, exist_ok=True)
    for name in names or PINS:
        src, fname, want = PINS[name]
        if name == "manga":
            from huggingface_hub import snapshot_download
            snapshot_download(src[1], revision=src[2])
            continue
        dest = model_path(name)
        if name == "screenai":
            if (dest / "resources" / "libchromescreenai.so").exists():
                continue
            zpath = MODELS / "screen_ai" / fname
            zpath.parent.mkdir(parents=True, exist_ok=True)
            if not zpath.exists() or sha256(zpath) != want:
                url = f"https://chrome-infra-packages.appspot.com/dl/{src[1]}/+/latest"
                req = urllib.request.Request(url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=600) as r, open(zpath, "wb") as f:
                    shutil.copyfileobj(r, f)
            got = sha256(zpath)
            if got != want:
                raise RuntimeError(f"Screen AI package sha256 {got} != pinned {want}: CIPD 'latest' moved; "
                                   "update PINS after checking the new version")
            import zipfile
            zipfile.ZipFile(zpath).extractall(dest)
            continue
        if dest.exists() and sha256(dest) == want:
            continue
        mirror = next((Path(m) / fname for m in mirrors if (Path(m) / fname).exists()), None)
        if mirror is not None and sha256(mirror) == want:
            shutil.copyfile(mirror, dest)
        elif src[0] == "hf":
            from huggingface_hub import hf_hub_download
            shutil.copyfile(hf_hub_download(src[1], fname, revision=src[2]), dest)
        else:
            req = urllib.request.Request(src[1], headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=600) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f)
        if sha256(dest) != want:
            raise RuntimeError(f"{dest}: sha256 mismatch")
        print(f"fetched {name} -> {dest}", flush=True)


def session(path, n_threads=None):
    import onnxruntime as ort
    ort.set_default_logger_severity(3)
    so = ort.SessionOptions()
    so.intra_op_num_threads = n_threads or threads()
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


# ----------------------------------------------------------------------------- MeikiOCR

class Meiki:
    """MeikiOCR with the detector's scores kept and raw character confidences.

    meikiocr multiplies the confidence of punctuation candidates by punct_conf_factor
    before its overlap suppression, so that punctuation gives way to an overlapping glyph
    (owocr uses 0.2). The suppression is kept, but the confidence reported for an accepted
    punctuation character is divided back, so thresholds compare like with like.
    `rgb` feeds the image as RGB, as owocr does (meikiocr's own docs say BGR)."""

    name = "meiki"

    def __init__(self, punct=0.2, det_thr=0.5, rec_thr=0.1, rgb=False, n_threads=None):
        from meikiocr import MeikiOCR
        m = MeikiOCR.__new__(MeikiOCR)        # skip its __init__: our sessions, pinned files
        m.det_session = session(model_path("meiki_det"), n_threads)
        m.rec_session = session(model_path("meiki_rec"), n_threads)
        m.vrec_session = session(model_path("meiki_vrec"), n_threads)
        m.active_provider = "CPUExecutionProvider"
        m.max_batch_size = 8
        self.m, self.punct, self.det_thr, self.rec_thr, self.rgb = m, punct, det_thr, rec_thr, rgb

    def _raw(self, chars):
        out = []
        for c in chars:
            conf = c["conf"]
            if self.punct != 1.0 and unicodedata.category(c["char"]).startswith("P"):
                conf /= self.punct
            out.append([c["char"], *[int(v) for v in c["bbox"]], round(float(conf), 4)])
        return out

    def ocr(self, bgr):
        img = np.ascontiguousarray(bgr[:, :, ::-1]) if self.rgb else bgr
        m = self.m
        tensor, scale = m._preprocess_for_detection(img)
        _, boxes, scores = m._run_detection_inference(tensor, scale)
        boxes, scores = boxes[0], scores[0]
        keep = scores > self.det_thr
        h, w = img.shape[:2]
        boxes = np.clip(boxes[keep], 0, np.array([w, h, w, h])).astype(np.int32)
        scores = scores[keep]
        order = np.argsort(boxes[:, 1], kind="stable")
        text_boxes = [{"bbox": boxes[i].tolist()} for i in order]
        confs = [float(scores[i]) for i in order]
        if not text_boxes:
            return []
        results = [{"text": "", "chars": [], "is_vertical": False} for _ in text_boxes]
        hi, vi = [], []
        for i, tb in enumerate(text_boxes):
            x1, y1, x2, y2 = tb["bbox"]
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            (vi if y2 - y1 > x2 - x1 else hi).append(i)
        for idx, vertical in ((hi, False), (vi, True)):
            if idx:
                m._process_recognition_pipeline(img, text_boxes, idx, results, self.rec_thr, self.punct, vertical)
        out = []
        for tb, conf, r in zip(text_boxes, confs, results):
            out.append({"det": tb["bbox"], "det_conf": round(conf, 4), "vertical": bool(r["is_vertical"]),
                        "text": r["text"].replace("\ufffd", ""), "chars": self._raw(r["chars"])})
        return out

    def __call__(self, bgr):
        """Recognizer only, on a line crop: [text, mean conf, min conf, chars]."""
        img = np.ascontiguousarray(bgr[:, :, ::-1]) if self.rgb else bgr
        r = self.m.run_recognition([img], conf_threshold=self.rec_thr, punct_conf_factor=self.punct)[0]
        chars = self._raw(r["chars"])
        cs = [c[5] for c in chars]
        return [r["text"].replace("\ufffd", ""), round(float(np.mean(cs)), 4) if cs else 0.0,
                round(min(cs), 4) if cs else 0.0, chars]


# ----------------------------------------------------------------------------- PP-OCR rec

class PaddleRec:
    """PP-OCR CTC recognizer, preprocessed like rapidocr: BGR, height 48, (x/255 - 0.5)/0.5,
    right-padded with zeros to at least 320 px. Width is capped at 4096 px (eval's cap)."""

    def __init__(self, name, n_threads=None):
        self.name = name
        self.s = session(model_path(name), n_threads)
        chars = self.s.get_modelmeta().custom_metadata_map["character"].split("\n")
        self.chars = [""] + chars + [" "]               # 0 = CTC blank, last = space
        self.inp = self.s.get_inputs()[0].name

    def __call__(self, bgr):
        import cv2
        h, w = bgr.shape[:2]
        nw = min(4096, max(8, int(math.ceil(48 * w / max(h, 1)))))
        x = cv2.resize(bgr, (nw, 48), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        x = (x / 255.0 - 0.5) / 0.5
        pad = np.zeros((48, max(nw, 320), 3), np.float32)
        pad[:, :nw] = x
        y = self.s.run(None, {self.inp: pad.transpose(2, 0, 1)[None]})[0][0]   # T x C, softmaxed
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
        return ["".join(text).strip(), round(float(np.mean(cs)), 4) if cs else 0.0,
                round(min(cs), 4) if cs else 0.0]


# ----------------------------------------------------------------------------- manga-ocr

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

    def __init__(self, n_threads=None):
        import logging

        import torch
        logging.getLogger("transformers").setLevel(logging.ERROR)
        from transformers.utils import logging as tlog
        tlog.set_verbosity_error()
        torch.set_num_threads(n_threads or threads())
        from loguru import logger
        logger.disable("manga_ocr")
        from manga_ocr import MangaOcr
        self.m = MangaOcr(pretrained_model_name_or_path=str(model_path("manga")), force_cpu=True)

    def __call__(self, bgr):
        from PIL import Image
        return [self.m(Image.fromarray(np.ascontiguousarray(blockify(bgr[:, :, ::-1])))).strip()]


# ----------------------------------------------------------------------------- Screen AI

class ScreenAI:
    """Chrome Screen AI through owocr.ocr.ChromeScreenAI. owocr looks for the library in
    ~/.config/screen_ai/resources; it is pointed at the pinned copy instead, and the raw
    result (with confidences) is kept rather than owocr's normalised OcrResult."""

    name = "screenai"

    def __init__(self):
        import owocr.ocr as oo
        from loguru import logger
        logger.disable("owocr")
        base = model_path("screenai")
        if not (base / "resources" / "libchromescreenai.so").exists():
            raise FileNotFoundError(f"{base}: run `label.py fetch` first")
        home = MODELS / "screen_ai" / "home"
        link = home / ".config" / "screen_ai"
        link.parent.mkdir(parents=True, exist_ok=True)
        if not link.exists():
            link.symlink_to(base)

        class _Home(type(Path())):
            @classmethod
            def home(cls):
                return home

        class _Keep(oo.ChromeScreenAI):
            def _to_generic_result(self, response, *a):
                self.last = response
                return super()._to_generic_result(response, *a)

        saved, oo.Path = oo.Path, _Home        # only while owocr resolves model_dir
        try:
            self.e = _Keep()
        finally:
            oo.Path = saved
        if not self.e.available:
            raise RuntimeError("Chrome Screen AI did not initialise")

    @staticmethod
    def _box(b, sx, sy):
        x, y, w, h = (float(b.get(k, 0)) for k in ("x", "y", "width", "height"))
        a = math.radians(float(b.get("angle", 0)))
        pts = [(x, y), (x + w * math.cos(a), y + w * math.sin(a)),
               (x - h * math.sin(a), y + h * math.cos(a)),
               (x + w * math.cos(a) - h * math.sin(a), y + w * math.sin(a) + h * math.cos(a))]
        xs, ys = [p[0] * sx for p in pts], [p[1] * sy for p in pts]
        return [int(math.floor(min(xs))), int(math.floor(min(ys))), int(math.ceil(max(xs))), int(math.ceil(max(ys)))]

    def ocr(self, bgr):
        from PIL import Image
        img = Image.fromarray(np.ascontiguousarray(bgr[:, :, ::-1]))
        self.e.last = None
        ok, _ = self.e(img)
        if not ok or self.e.last is None:
            return []
        pw, ph = img.width, img.height
        if any(v > self.e.max_pixel_size for v in img.size):      # owocr shrank it
            f = min(self.e.max_pixel_size / pw, self.e.max_pixel_size / ph)
            sx, sy = pw / int(pw * f), ph / int(ph * f)
        else:
            sx = sy = 1.0
        out = []
        for ln in self.e.last.get("lines", []):
            chars = []
            for w in ln.get("words", []):
                for s in w.get("symbols", []):
                    chars.append([s.get("utf8_string", ""), *self._box(s.get("bounding_box", {}), sx, sy),
                                  round(float(s.get("confidence", 0.0)), 4)])
            d = (ln.get("direction") or "").replace("DIRECTION_", "")
            out.append({"box": self._box(ln.get("bounding_box", {}), sx, sy),
                        "angle": round(float(ln.get("bounding_box", {}).get("angle", 0.0)), 2),
                        "text": ln.get("utf8_string", ""), "conf": round(float(ln.get("confidence", 0.0)), 4),
                        "vertical": d == "TOP_TO_BOTTOM", "chars": chars})
        return out

    def __call__(self, bgr):
        """Recognizer use on a one-line crop: the segments on the crop's main row (the tallest
        segment's row; furigana and stray UI above or below it are left out), left to right."""
        lines = [ln for ln in self.ocr(bgr) if not ln["vertical"] and ln["text"]]
        if not lines:
            return ["", 0.0, 0.0]
        main = max(lines, key=lambda ln: ln["box"][3] - ln["box"][1])

        def vov(a, b):
            inter = min(a[3], b[3]) - max(a[1], b[1])
            return max(0, inter) / max(1, min(a[3] - a[1], b[3] - b[1]))
        row = sorted((ln for ln in lines if vov(ln["box"], main["box"]) >= 0.5
                      and (ln["box"][3] - ln["box"][1]) >= 0.65 * (main["box"][3] - main["box"][1])),
                     key=lambda ln: ln["box"][0])
        cs = [ln["conf"] for ln in row]
        return ["".join(ln["text"] for ln in row), round(float(np.mean(cs)), 4), round(min(cs), 4)]


def load(names, n_threads=None):
    makers = {"meiki": lambda: Meiki(n_threads=n_threads), "meiki-rgb": lambda: Meiki(rgb=True, n_threads=n_threads),
              "v5s": lambda: PaddleRec("v5s", n_threads), "v6s": lambda: PaddleRec("v6s", n_threads),
              "manga": lambda: MangaOCR(n_threads), "screenai": ScreenAI}
    return {n: makers[n]() for n in names}
