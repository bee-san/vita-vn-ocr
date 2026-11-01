#!/usr/bin/env python3
"""VOCR's desktop pipeline (issue #169, docs/desktop.md): the desktop line detector and a VOCR6 recognizer on ONNX
Runtime's CPU provider, for apps that read a game window on a PC (GameSentenceMiner and others).

    from vocr_desktop import Detector, read_frame
    from vocr_onnx import Recognizer
    det = Detector("vocr_desktop_det.onnx", threads=4)          # any frame size; `size` picks the input policy
    rec = Recognizer("VOCR6.2_L_fast.onnx")
    lines = read_frame(frame, det, rec, workers=4)               # frame: H x W grey or H x W x 3/4 RGB(A) uint8
    for ln in lines:
        ln.text, ln.box, ln.tag, ln.chars                        # box (x, y, w, h) in frame px; chars as vocr_onnx

    python3 vocr_desktop.py --det vocr_desktop_det.onnx --rec VOCR6.2_L_fast.onnx [--size fit:960] IMAGE...

The detector is a fully convolutional network trained at many text sizes (train/desktop_det.py), so it reads a frame
at any size: `size` scales the frame before the network, keeping its aspect ratio ("fit:N": the longer side down
to N px when it is longer, never up; "long:N": the longer side to N px; "native": as it is), and the boxes come back
in the frame's pixels. The network's input is the grey frame (integer luma, (77 R + 150 G + 29 B) >> 8, as the
recognizer's) padded on the right and bottom to a multiple of 32 by repeating its last column and row. Its output is
four maps at a quarter of the input's resolution plus the tag maps: a line's kernel (K, a logit), its orientation (O,
a logit: vertical when > 0), the distances from a kernel cell's centre to the line's two edges across it (A1, A2, in
input px) and the tags (dialogue, name, choice, UI, scene; textspot6's classes). The decode is textspot6's
(runtime/vocr_textspot6.c) on float maps: the 4-connected components of K > 0 with at least `min_cells` cells, the
line's extent across it from the mean of its cells' A1 / A2, along it the component's cells, grown by `pad_pct` of
its size across (at least `pad_min` px), its tag the class with the largest summed logit (class 0's logit is 0).

Each horizontal box is cut from the frame and read with its dark flag (vocr_onnx.line_dark: dark text is inverted
first, as VOCR was trained); read_frame's `workers` read the lines on that many threads (ONNX Runtime releases the
GIL while a network runs). Vertical boxes are returned without text: vertical text needs VOCR6-V and the column finder
(docs/vocr6v.md), not exported here.
"""
import json
from dataclasses import dataclass, field

import numpy as np

STRIDE = 4                 # the maps' stride in input px
ALIGN = 32                 # the network's input sides are multiples of this
CLASSES = ("dialogue", "name", "choice", "ui", "scene")


def _session(path, threads=1, providers=None, options=None):
    import onnxruntime as ort
    so = options or ort.SessionOptions()
    if options is None:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=providers or ["CPUExecutionProvider"])


def grey_of(img):
    """uint8 grey of an H x W grey or H x W x 3/4 RGB(A) image: (77 R + 150 G + 29 B) >> 8 (runtime/vocr.c's)."""
    a = np.asarray(img)
    if a.dtype != np.uint8:
        raise TypeError(f"images are uint8, not {a.dtype}")
    if a.ndim == 2:
        return a
    if a.ndim == 3 and a.shape[2] in (3, 4):
        r, g, b = (a[..., i].astype(np.uint16) for i in range(3))
        return ((77 * r + 150 * g + 29 * b) >> 8).astype(np.uint8)
    raise ValueError("an H x W grey or H x W x 3 / 4 RGB(A) image")


def scale_for(policy, w, h):
    """The factor a w x h frame is resized by before the network, for a size policy (module docstring)."""
    if policy in (None, "", "native"):
        return 1.0
    kind, _, n = str(policy).partition(":")
    if not n:
        kind, n = "fit", kind
    n = float(n)
    long = max(w, h)
    if kind == "fit":
        return min(1.0, n / long)
    if kind == "long":
        return n / long
    raise ValueError(f"size policy {policy!r}: fit:N, long:N or native")


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float
    vertical: int
    conf: float
    tag: int
    tconf: float


@dataclass
class Line:
    text: str
    box: tuple
    tag: int
    conf: float
    dark: int
    vertical: int = 0
    chars: list = field(default_factory=list)
    text_chars: list = field(default_factory=list)
    greedy: str = ""
    conf_min: float = 1.0


class Detector:
    """The desktop line detector (an ONNX file train/desktop_det.py exports)."""

    def __init__(self, path, threads=1, providers=None, session_options=None, size=None):
        self.sess = _session(path, threads, providers, session_options)
        meta = {k: json.loads(v) for k, v in self.sess.get_modelmeta().custom_metadata_map.items()
                if k.startswith("vocr.")}
        if meta.get("vocr.kind") != "desktop-detector":
            raise ValueError(f"{path}: not a VOCR desktop detector")
        self.meta = meta
        self.size = size or meta.get("vocr.size", "fit:960")
        self.min_cells = int(meta.get("vocr.min_cells", 2))
        self.conf_min = float(meta.get("vocr.conf_min", 0.5))
        self.pad_pct = float(meta.get("vocr.pad_pct", 0.18))
        self.pad_min = float(meta.get("vocr.pad_min", 2.0))
        self.min_across = float(meta.get("vocr.min_across", 4.0))
        self.tags = int(meta.get("vocr.tags", 4))
        self.input_name = self.sess.get_inputs()[0].name

    def preprocess(self, g):
        """(network input [1, 1, H', W'] uint8, factor) of a grey frame: resized by the size policy (INTER_AREA down,
        INTER_LINEAR up), padded right and bottom to a multiple of 32 by edge replication."""
        import cv2
        h, w = g.shape
        f = scale_for(self.size, w, h)
        if f != 1.0:
            nw, nh = max(1, int(round(w * f))), max(1, int(round(h * f)))
            g = cv2.resize(g, (nw, nh), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
        hh, ww = g.shape
        ph, pw = -hh % ALIGN, -ww % ALIGN
        if ph or pw:
            g = np.pad(g, ((0, ph), (0, pw)), mode="edge")
        fx, fy = ww / w, hh / h
        return np.ascontiguousarray(g, np.uint8)[None, None], (fx, fy, ww, hh)

    def maps(self, x):
        return self.sess.run(None, {self.input_name: x})[0][0]

    def detect(self, img):
        """[Box] in the frame's px, in the decode's order (by confidence)."""
        g = grey_of(img)
        x, (fx, fy, ww, hh) = self.preprocess(g)
        boxes = decode(self.maps(x), ww, hh, self.min_cells, self.conf_min, self.pad_pct, self.pad_min,
                       self.min_across, self.tags)
        H, W = g.shape
        out = []
        for b in boxes:
            x0, y0 = max(0.0, b.x / fx), max(0.0, b.y / fy)
            x1, y1 = min(float(W), (b.x + b.w) / fx), min(float(H), (b.y + b.h) / fy)
            if x1 > x0 and y1 > y0:
                out.append(Box(x0, y0, x1 - x0, y1 - y0, b.vertical, b.conf, b.tag, b.tconf))
        return out

    def boxes(self, rgb):
        """eval/desktop.py's detector interface: [[x0, y0, x1, y1, vertical, conf, tag]] in frame px."""
        return [[b.x, b.y, b.x + b.w, b.y + b.h, b.vertical, round(b.conf, 4), b.tag] for b in self.detect(rgb)]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode(maps, w, h, min_cells=2, conf_min=0.5, pad_pct=0.18, pad_min=2.0, min_across=4.0, tags=4):
    """[Box] in network-input px (the image's w x h part) from float maps [4 + T, h / 4, w / 4] (module docstring).
    Every component at once: per-label sums by np.bincount, the along extent from cv2's component boxes."""
    import cv2
    hm, wm = -(-h // STRIDE), -(-w // STRIDE)
    k = np.ascontiguousarray(maps[0, :hm, :wm], np.float64)
    mask = (k > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
    if n <= 1:
        return []
    flat = lab.ravel()
    on = np.flatnonzero(flat)
    L = flat[on]
    rows, cols = np.divmod(on, wm)
    cnt = np.bincount(L, minlength=n).astype(np.float64)
    cnt[0] = 1.0
    conf = np.bincount(L, 1.0 / (1.0 + np.exp(-k.ravel()[on])), minlength=n) / cnt
    vert = np.bincount(L, maps[1, :hm, :wm].ravel()[on].astype(np.float64), minlength=n) / cnt > 0
    vl = vert[L]
    across = np.where(vl, cols, rows)
    cc = STRIDE * across + (STRIDE - 1) / 2.0
    near = np.bincount(L, cc - maps[2, :hm, :wm].ravel()[on], minlength=n) / cnt
    far = np.bincount(L, cc + maps[3, :hm, :wm].ravel()[on], minlength=n) / cnt
    tsum = np.stack([np.zeros(n)] + [np.bincount(L, maps[4 + i, :hm, :wm].ravel()[on].astype(np.float64), minlength=n)
                                     for i in range(tags)], 1)
    out = []
    for c in range(1, n):
        if stats[c, cv2.CC_STAT_AREA] < min_cells or conf[c] < conf_min:
            continue
        size = far[c] - near[c]
        if size < min_across:
            continue
        t0, n0 = (cv2.CC_STAT_TOP, cv2.CC_STAT_HEIGHT) if vert[c] else (cv2.CC_STAT_LEFT, cv2.CC_STAT_WIDTH)
        lo, hi = STRIDE * int(stats[c, t0]), STRIDE * int(stats[c, t0] + stats[c, n0])
        pad = max(pad_min, size * pad_pct)
        a0, a1_, b0, b1 = near[c] - pad, far[c] + pad, lo - pad, hi + pad
        x0, x1, y0, y1 = (a0, a1_, b0, b1) if vert[c] else (b0, b1, a0, a1_)
        x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)
        if x1 <= x0 or y1 <= y0:
            continue
        sc = tsum[c]
        tag = int(np.argmax(sc))
        second = float(np.max(np.delete(sc, tag))) if tags else 0.0
        out.append(Box(float(x0), float(y0), float(x1 - x0), float(y1 - y0), int(vert[c]), float(conf[c]), tag,
                       (float(sc[tag]) - second) / float(cnt[c])))
    out.sort(key=lambda b: (-b.conf, b.y, b.x))
    return out


def reading_order(boxes):
    """Indices of boxes (x, y, w, h first) in reading order: rows top to bottom (a box joins a row when it overlaps
    the row's first box vertically by at least half the smaller height), left to right within a row."""
    idx = sorted(range(len(boxes)), key=lambda i: (boxes[i][1] + boxes[i][3] / 2.0, boxes[i][0]))
    rows = []
    for i in idx:
        x, y, w, h = boxes[i][:4]
        if rows:
            fx, fy, fw, fh = boxes[rows[-1][0]][:4]
            if min(fy + fh, y + h) - max(fy, y) >= 0.5 * min(fh, h):
                rows[-1].append(i)
                continue
        rows.append([i])
    return [i for r in rows for i in sorted(r, key=lambda j: boxes[j][0])]


def crop_of(b, W, H):
    """A float box (x, y, w, h) as integer crop bounds (x, y, w, h), rounded outward and clipped, or None."""
    import math
    x0, y0 = max(0, int(math.floor(b[0]))), max(0, int(math.floor(b[1])))
    x1, y1 = min(W, int(math.ceil(b[0] + b[2]))), min(H, int(math.ceil(b[1] + b[3])))
    return (x0, y0, x1 - x0, y1 - y0) if x1 > x0 and y1 > y0 else None


def read_frame(frame, det, rec, lm=None, sections=True, tags=None, region=None, workers=1, executor=None):
    """Every line of a frame (or of a region (x, y, w, h) of it), top to bottom: [Line], boxes and character cells
    in frame px. det: a Detector; rec: a vocr_onnx.Recognizer; lm: a vocr_onnx.CharLM; tags: the tag classes to read
    (None: all); workers: lines read in parallel on that many threads (or pass a concurrent.futures executor)."""
    import vocr_onnx as V
    g = grey_of(frame)
    H, W = g.shape
    x0 = y0 = 0
    sub = g
    if region is not None:
        rx, ry, rw, rh = (int(v) for v in region)
        x0, y0 = min(max(0, rx), W), min(max(0, ry), H)
        x1, y1 = min(max(0, rx + rw), W), min(max(0, ry + rh), H)
        sub = np.ascontiguousarray(g[y0:y1, x0:x1])
        if not sub.size:
            return []
    boxes = det.detect(sub)
    if tags is not None:
        boxes = [b for b in boxes if b.tag in tags]
    order = reading_order([(b.x, b.y, b.w, b.h) for b in boxes])
    sh, sw = sub.shape
    jobs = []
    for i in order:
        b = boxes[i]
        cb = crop_of((b.x, b.y, b.w, b.h), sw, sh)
        if cb is not None:
            jobs.append((b, cb))

    def one(job):
        b, (cx, cy, cw, ch) = job
        if b.vertical:
            return Line("", (x0 + cx, y0 + cy, cw, ch), b.tag, b.conf, 0, 1)
        dark = V.line_dark(sub, (cx, cy, cw, ch))
        r = rec.read(sub[cy:cy + ch, cx:cx + cw], invert=bool(dark), sections=sections, lm=lm)
        for c in r.chars + r.text_chars:
            c.x, c.x0, c.x1 = c.x + x0 + cx, c.x0 + x0 + cx, c.x1 + x0 + cx
            c.y0, c.y1 = y0 + cy, y0 + cy + ch
        return Line(r.text, (x0 + cx, y0 + cy, cw, ch), b.tag, b.conf, int(dark), 0, r.chars, r.text_chars,
                    r.greedy, r.conf_min)

    if executor is not None and len(jobs) > 1:
        return list(executor.map(one, jobs))
    if workers > 1 and len(jobs) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(one, jobs))
    return [one(j) for j in jobs]


def _load_image(path):
    from PIL import Image
    return np.asarray(Image.open(path).convert("RGB"))


def main(argv=None):
    import argparse
    import sys
    import time
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import vocr_onnx as V
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--det", required=True, help="the desktop detector's ONNX file")
    ap.add_argument("--rec", required=True, help="a VOCR6 recognizer's ONNX file (VOCR6.2_L_fast.onnx)")
    ap.add_argument("--lm", help="a VOCR character LM (.vlm): beam 8 + the LM at Rubi's settings")
    ap.add_argument("--size", help="the detector's input policy: fit:N, long:N or native (default: the file's)")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--region", help="x,y,w,h: read only this region of each image")
    ap.add_argument("images", nargs="+")
    a = ap.parse_args(argv)
    det = Detector(a.det, threads=a.threads, size=a.size)
    rec = V.Recognizer(a.rec, threads=1)
    lm = V.CharLM(a.lm, tier=rec.name[-1:] if rec.name else None) if a.lm else None
    region = tuple(int(v) for v in a.region.split(",")) if a.region else None
    for p in a.images:
        img = _load_image(p)
        t0 = time.perf_counter()
        lines = read_frame(img, det, rec, lm=lm, region=region, workers=a.threads)
        ms = (time.perf_counter() - t0) * 1e3
        print(json.dumps({"image": p, "ms": round(ms, 1), "lines": [
            {"text": ln.text, "box": [int(v) for v in ln.box], "tag": CLASSES[ln.tag], "vertical": ln.vertical,
             "chars": [[c.text, round(c.x0, 1), round(c.x1, 1)] for c in ln.text_chars]} for ln in lines]},
            ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
