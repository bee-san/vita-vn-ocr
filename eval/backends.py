"""Recognizer backends for eval/run.py.

Every backend takes RGB uint8 line crops (H x W x 3 numpy arrays) and returns
one string per crop. make_recognizer() turns a --model spec into a backend:

  path/to/ckpt.pt          tiny CTC model from train/model.py (PyTorch checkpoint)
  path/to/model.onnx       an ONNX export: 1-channel input = tiny model (height 32,
                           train/train.py preprocessing); 3-channel = PaddleOCR rec
                           (height 48, overlay preprocessing, dict from metadata)
  ppocr-v5-mobile          PP-OCRv5 mobile rec, the model vita-jp-overlay ships
  ppocr-v5-mobile-jis      the same with its CTC head masked to JIS X 0208 + ASCII +
                           full-width forms (identical to pruning the head's rows)
  ppocr-v5-server          PP-OCRv5 server rec (pseudo-labelling engine)
  ppocr-v6-small[-jis]     PP-OCRv6 small rec
  ncnn:DIR                 an ncnn param/bin pair (e.g. the JIS int8 model) run by the
                           pinned ncnn build in eval/ncnn (see eval/ncnn/build.sh)
  manga-ocr[-raw]          manga-ocr reference; long lines are re-flowed into a
                           square-ish block first unless -raw (same as data/label_real.py)
  cmd:COMMAND              any external recognizer, e.g. the C runtime's host CLI.
                           Crops are written as binary PPM (P6, RGB) or PGM/PNG
                           (--cmd-format). If COMMAND contains {} it runs once per crop
                           with {} replaced by the crop path and its stdout is the text;
                           otherwise it runs once with one crop path per stdin line and
                           must print exactly one UTF-8 line of text per input line.
"""
import os
import shlex
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT / "train", ROOT / "synth"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Where the PaddleOCR ONNX files may live: $VOCR_MODELS, rapidocr's package
# directory (server rec, v6 small) and the research scratch copies (v5 mobile).
_HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
MODEL_DIRS = [Path(p) for p in os.environ.get("VOCR_MODELS", "").split(os.pathsep) if p] + [
    ROOT / "data" / "eval" / "models",
    _HERD / "shared" / "research" / "scratch" / "eval" / "models" / "rapidocr",
    _HERD / "shared" / "research" / "scratch" / "mem" / "models",
]
PPOCR = {
    "ppocr-v5-mobile": "ch_PP-OCRv5_rec_mobile.onnx",
    "ppocr-v5-server": "ch_PP-OCRv5_rec_server.onnx",
    "ppocr-v6-small": "PP-OCRv6_rec_small.onnx",
}


def threads():
    return int(os.environ.get("OMP_NUM_THREADS") or os.environ.get("TORCH_NUM_THREADS") or 1)


def find_model(name):
    dirs = list(MODEL_DIRS)
    try:
        import rapidocr
        dirs.append(Path(rapidocr.__file__).parent / "models")
    except ImportError:
        pass
    for d in dirs:
        if (d / name).exists():
            return d / name
    raise FileNotFoundError(f"{name} not found in {[str(d) for d in dirs]} (set VOCR_MODELS)")


def to_gray(rgb):
    """Integer luma, identical to synth/common.py and the C side."""
    r, g, b = (rgb[..., i].astype(np.uint16) for i in range(3))
    return ((77 * r + 150 * g + 29 * b) >> 8).astype(np.uint8)


def ctc_greedy(ids, vocab, blank=0):
    out, last = [], blank
    for t in ids:
        t = int(t)
        if t != blank and t != last:
            out.append(vocab[t - 1])
        last = t
    return "".join(out)


def jis_x0208():
    """JIS X 0208 + printable ASCII + full-width forms + a few VN symbols: the class
    set the research pruned PP-OCRv5's head to (scratch/acc/eval_rec.py)."""
    out = set()
    for a in range(0xA1, 0xFF):
        for b in range(0xA1, 0xFF):
            try:
                out.add(bytes([a, b]).decode("euc_jp"))
            except UnicodeDecodeError:
                pass
    out |= {chr(c) for c in range(0x21, 0x7F)}
    out |= {chr(c) for c in range(0xFF01, 0xFF5F)}
    out |= set("♪♡♥☆★〜～")
    return out


class Recognizer:
    name = "?"
    info = {}

    def read(self, crops):
        raise NotImplementedError

    def close(self):
        pass


class TinyTorch(Recognizer):
    """train/model.py checkpoint; preprocessing = train/train.py (grey, height 32
    bilinear, width <= 1024, padded to a multiple of 8 by repeating the last column),
    one line per forward pass, greedy CTC."""

    H, MAX_W, STRIDE = 32, 1024, 8

    def __init__(self, path):
        import torch

        from model import TinyVNOCR
        torch.set_num_threads(threads())
        ck = torch.load(path, map_location="cpu", weights_only=False)
        self.vocab = ck["vocab"]
        self.model = TinyVNOCR(len(self.vocab) + 1, width=ck.get("width", 1.0))
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.torch = torch
        n = sum(p.numel() for p in self.model.parameters())
        self.name = f"tiny:{Path(path).name}"
        self.info = {"params": n, "fp32_weight_bytes": 4 * n, "classes": len(self.vocab) + 1,
                     "step": ck.get("step"), "vocab_set": set(self.vocab)}

    @classmethod
    def prep(cls, rgb):
        g = to_gray(rgb) if rgb.ndim == 3 else rgb
        h, w = g.shape
        nw = max(8, min(cls.MAX_W, int(round(w * cls.H / h))))
        g = cv2.resize(g, (nw, cls.H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        W = (nw + cls.STRIDE - 1) // cls.STRIDE * cls.STRIDE
        x = np.empty((1, 1, cls.H, W), np.float32)
        x[0, 0, :, :nw] = g
        x[0, 0, :, nw:] = g[:, -1:]
        return x / 127.5 - 1.0, (nw + cls.STRIDE - 1) // cls.STRIDE

    def read(self, crops):
        out = []
        with self.torch.no_grad():
            for c in crops:
                x, t = self.prep(c)
                logits = self.model(self.torch.from_numpy(x))[:t, 0]
                out.append(ctc_greedy(logits.argmax(1).numpy(), self.vocab))
        return out


class TinyOnnx(Recognizer):
    def __init__(self, path, vocab_path=ROOT / "data" / "vocab.txt"):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads()
        self.s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.inp = self.s.get_inputs()[0].name
        self.vocab = [c for c in Path(vocab_path).read_text(encoding="utf-8").split("\n") if c]
        self.name = f"tiny-onnx:{Path(path).name}"
        self.info = {"vocab_set": set(self.vocab)}

    def read(self, crops):
        out = []
        for c in crops:
            x, t = TinyTorch.prep(c)
            y = self.s.run(None, {self.inp: x})[0]
            y = y[:t, 0] if y.ndim == 3 and y.shape[1] == 1 else y.reshape(-1, y.shape[-1])[:t]
            out.append(ctc_greedy(y.argmax(1), self.vocab))
        return out


class PaddleOnnx(Recognizer):
    """PaddleOCR CTC recogniser. Preprocessing follows vita-jp-overlay's
    core/local_ocr.cpp: BGR, height 48, width (w*48 + h/2)/h clamped to [8, 1024],
    bilinear, (p - 127.5) / 127.5, no padding; greedy CTC with class 0 = blank,
    class i = dict[i-1] and the last class a space."""

    def __init__(self, path, keep=None, max_w=1024, name=None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads()
        so.inter_op_num_threads = 1
        self.s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.inp = self.s.get_inputs()[0].name
        chars = self.s.get_modelmeta().custom_metadata_map.get("character")
        if not chars:
            raise ValueError(f"{path}: no 'character' metadata; not a PaddleOCR rec export")
        self.vocab = chars.splitlines() + [" "]
        self.max_w = max_w
        self.mask = None
        if keep is not None:
            self.mask = np.array([0] + [i + 1 for i, ch in enumerate(self.vocab) if ch in keep or ch == " "])
        self.name = name or f"ppocr:{Path(path).name}"
        kept = self.vocab if self.mask is None else [self.vocab[i - 1] for i in self.mask[1:]]
        self.info = {"classes": len(kept) + 1, "vocab_set": set(kept)}

    def prep(self, rgb):
        h, w = rgb.shape[:2]
        nw = min(self.max_w, max(8, (w * 48 + h // 2) // h))
        x = cv2.resize(rgb, (nw, 48), interpolation=cv2.INTER_LINEAR)[:, :, ::-1].astype(np.float32)
        return np.ascontiguousarray(((x - 127.5) / 127.5).transpose(2, 0, 1)[None])

    def read(self, crops):
        out = []
        for c in crops:
            if c.ndim == 2:
                c = np.repeat(c[:, :, None], 3, axis=2)
            y = self.s.run(None, {self.inp: self.prep(c)})[0][0]
            if y.shape[1] != len(self.vocab) + 1:
                raise RuntimeError(f"class count {y.shape[1]} != dict + 2 ({len(self.vocab) + 1})")
            ids = self.mask[np.argmax(y[:, self.mask], axis=1)] if self.mask is not None else y.argmax(1)
            out.append(ctc_greedy(ids, self.vocab))
        return out


def blockify(rgb):
    """Re-flow a long one-line crop into k stacked rows so manga-ocr's 224x224 resize
    keeps characters legible (copy of data/label_real.py's blockify)."""
    h, w = rgb.shape[:2]
    k = int(round(np.sqrt(w / max(h, 1))))
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


class MangaOcrRec(Recognizer):
    def __init__(self, raw=False):
        import torch
        from PIL import Image
        torch.set_num_threads(threads())
        from manga_ocr import MangaOcr
        self.m = MangaOcr(force_cpu=True)
        self.Image = Image
        self.raw = raw
        self.name = "manga-ocr" + ("-raw" if raw else "")
        n = sum(p.numel() for p in self.m.model.parameters())
        self.info = {"params": n, "fp32_weight_bytes": 4 * n}

    def read(self, crops):
        out = []
        for c in crops:
            if c.ndim == 2:
                c = np.repeat(c[:, :, None], 3, axis=2)
            img = c if self.raw else blockify(c)
            out.append(self.m(self.Image.fromarray(np.ascontiguousarray(img))))
        return out


def write_crop(path, rgb, fmt):
    if fmt == "png":
        cv2.imwrite(str(path), rgb[:, :, ::-1])
        return
    if fmt == "pgm":
        g = to_gray(rgb)
        Path(path).write_bytes(b"P5\n%d %d\n255\n" % (g.shape[1], g.shape[0]) + g.tobytes())
        return
    rgb = np.ascontiguousarray(rgb)
    Path(path).write_bytes(b"P6\n%d %d\n255\n" % (rgb.shape[1], rgb.shape[0]) + rgb.tobytes())


class Cmd(Recognizer):
    def __init__(self, command, fmt="ppm", timeout=600):
        self.command, self.fmt, self.timeout = command, fmt, timeout
        self.name = f"cmd:{command}"
        self.info = {}

    def read(self, crops):
        with tempfile.TemporaryDirectory(prefix="vocr_eval_") as tmp:
            paths = []
            for i, c in enumerate(crops):
                p = Path(tmp) / f"{i:06d}.{self.fmt}"
                write_crop(p, c if c.ndim == 3 else np.repeat(c[:, :, None], 3, axis=2), self.fmt)
                paths.append(str(p))
            if "{}" in self.command:
                out = []
                for p in paths:
                    cmd = self.command.replace("{}", shlex.quote(p))
                    r = subprocess.run(cmd, shell=True, capture_output=True, timeout=self.timeout,
                                       stdin=subprocess.DEVNULL)
                    if r.returncode:
                        raise RuntimeError(f"{cmd} failed ({r.returncode}): {r.stderr.decode()[-500:]}")
                    out.append(r.stdout.decode("utf-8", "replace").strip("\r\n"))
                return out
            r = subprocess.run(self.command, shell=True, input="\n".join(paths).encode() + b"\n",
                               capture_output=True, timeout=self.timeout * max(1, len(paths) // 100))
            if r.returncode:
                raise RuntimeError(f"{self.command} failed ({r.returncode}): {r.stderr.decode()[-500:]}")
            lines = r.stdout.decode("utf-8", "replace").split("\n")
            if lines and lines[-1] == "":
                lines.pop()
            if len(lines) != len(paths):
                raise RuntimeError(f"{self.command} printed {len(lines)} lines for {len(paths)} crops")
            return [x.rstrip("\r") for x in lines]


class Ncnn(Recognizer):
    """PaddleOCR rec in ncnn, run by eval/ncnn/vocr_ncnn_rec (built against the
    overlay's pinned ncnn). Preprocessing is the overlay's: ncnn from_pixels_resize
    RGBA->BGR to height 48, (p - 127.5) / 127.5, greedy CTC over the dict."""

    def __init__(self, model_dir):
        d = Path(model_dir)
        params = sorted(d.glob("*.param"))
        if len(params) != 1:
            raise FileNotFoundError(f"{d}: expected exactly one .param file, found {params}")
        self.param, self.bin = params[0], params[0].with_suffix(".bin")
        dicts = sorted(d.glob("*dict*.txt")) + sorted(d.glob("vocab*.txt"))
        if not dicts:
            raise FileNotFoundError(f"{d}: no dict file (one character per line, without blank)")
        self.vocab = Path(dicts[0]).read_text(encoding="utf-8").split("\n")
        if self.vocab and self.vocab[-1] == "":
            self.vocab.pop()
        self.exe = Path(os.environ.get("VOCR_NCNN_REC", HERE / "ncnn" / "build" / "vocr_ncnn_rec"))
        if not self.exe.exists():
            raise FileNotFoundError(f"{self.exe} missing: run eval/ncnn/build.sh first")
        self.name = f"ncnn:{d.name}"
        self.info = {"weights_file_bytes": self.bin.stat().st_size, "classes": len(self.vocab) + 1,
                     "vocab_set": set(self.vocab)}

    def read(self, crops):
        with tempfile.TemporaryDirectory(prefix="vocr_ncnn_") as tmp:
            paths = []
            for i, c in enumerate(crops):
                p = Path(tmp) / f"{i:06d}.ppm"
                write_crop(p, c if c.ndim == 3 else np.repeat(c[:, :, None], 3, axis=2), "ppm")
                paths.append(str(p))
            r = subprocess.run([str(self.exe), str(self.param), str(self.bin)],
                               input=("\n".join(paths) + "\n").encode(), capture_output=True,
                               timeout=3600)
            if r.returncode:
                raise RuntimeError(f"vocr_ncnn_rec failed ({r.returncode}): {r.stderr.decode()[-800:]}")
            out = []
            for row in r.stdout.decode().splitlines()[:len(paths)]:
                ids = [int(t) for t in row.split()] if row.strip() else []
                out.append("".join(self.vocab[t - 1] if 0 < t <= len(self.vocab) else " " for t in ids))
            if len(out) != len(paths):
                raise RuntimeError(f"vocr_ncnn_rec returned {len(out)} rows for {len(paths)} crops")
            self.last_stderr = r.stderr.decode()
            return out


def make_recognizer(spec, cmd_format="ppm"):
    if spec.startswith("cmd:"):
        return Cmd(spec[4:], cmd_format)
    if spec.startswith("ncnn:"):
        return Ncnn(spec[5:])
    if spec in ("manga-ocr", "manga-ocr-raw"):
        return MangaOcrRec(raw=spec.endswith("-raw"))
    base = spec[:-4] if spec.endswith("-jis") else spec
    if base in PPOCR:
        return PaddleOnnx(find_model(PPOCR[base]), keep=jis_x0208() if spec.endswith("-jis") else None,
                          name=spec)
    p = Path(spec)
    if p.suffix == ".pt":
        return TinyTorch(p)
    if p.suffix == ".onnx":
        import onnxruntime as ort
        shape = ort.InferenceSession(str(p), providers=["CPUExecutionProvider"]).get_inputs()[0].shape
        return TinyOnnx(p) if shape[1] == 1 else PaddleOnnx(p)
    raise ValueError(f"unknown --model {spec!r}")


def timed_read(rec, crops, chunk=64):
    """Read crops in chunks; returns texts and host seconds (wall) per crop."""
    texts, t0 = [], time.perf_counter()
    for i in range(0, len(crops), chunk):
        texts += rec.read(crops[i:i + chunk])
    return texts, (time.perf_counter() - t0) / max(1, len(crops))


def nfkc(s):
    return unicodedata.normalize("NFKC", s)
