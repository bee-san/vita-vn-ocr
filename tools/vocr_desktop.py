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
import math
from dataclasses import dataclass, field

import numpy as np

STRIDE = 4                 # the maps' stride in input px
ALIGN = 32                 # the network's input sides are multiples of this
MAX_SIDE = 8192            # a policy's N, as desktop/vocr_desk.h's VOCR_DESK_MAX_SIDE
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
    """The factor a w x h frame is resized by before the network, for a size policy (module docstring). N is 1 to
    8192 px, as the C path accepts (ValueError otherwise)."""
    if policy in (None, "", "native"):
        return 1.0
    kind, _, n = str(policy).partition(":")
    if not n:
        kind, n = "fit", kind
    n = float(n)
    if not 1 <= n <= MAX_SIDE:
        raise ValueError(f"size policy {policy!r}: N is 1 to {MAX_SIDE} px")
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
    logit: float            # the mean K logit over the box's kernel cells (the decode's confidence)
    tag: int
    tconf: float            # (best tag score - second) / cells
    cells: int = 0


@dataclass
class Line:
    text: str
    box: tuple
    tag: int
    logit: float            # the detector's mean K logit of the box
    dark: int
    vertical: int = 0
    chars: list = field(default_factory=list)
    text_chars: list = field(default_factory=list)
    greedy: str = ""
    conf_min: float = 1.0


class Detector:
    """The desktop line detector (an ONNX file train/desktop_det.py exports)."""

    def __init__(self, path, threads=1, providers=None, session_options=None, size=None, pad=None):
        self.sess = _session(path, threads, providers, session_options)
        meta = {k: json.loads(v) for k, v in self.sess.get_modelmeta().custom_metadata_map.items()
                if k.startswith("vocr.")}
        if meta.get("vocr.kind") != "desktop-detector":
            raise ValueError(f"{path}: not a VOCR desktop detector")
        self.meta = meta
        self.size = size or meta.get("vocr.size", "fit:960")
        self.min_cells = int(meta.get("vocr.min_cells", 2))
        if "vocr.conf_min_logit" in meta:
            self.conf_min_logit = float(meta["vocr.conf_min_logit"])
        else:                                   # files of the first runs: a probability (0.5: logit 0)
            p = float(meta.get("vocr.conf_min", 0.5))
            self.conf_min_logit = math.log(p / (1.0 - p))
        self.pad_pct = float(meta.get("vocr.pad_pct", 0.18))
        self.pad_min = float(meta.get("vocr.pad_min", 2.0))
        self.pad_frame = int(meta.get("vocr.pad_frame", 0))
        self.pad_frame_min = int(meta.get("vocr.pad_frame_min", 2))
        if pad == "frame":          # MeikiOCR's reference pipeline's pad: the ink box + max(2, round(4 H / 544)) px
            self.pad_pct, self.pad_min, self.pad_frame = 0.0, 0.0, 4
        elif pad == "pct":
            self.pad_pct, self.pad_min, self.pad_frame = 0.18, 2.0, 0
        elif pad is not None:
            raise ValueError(f"pad rule {pad!r}: pct or frame")
        self.min_across = float(meta.get("vocr.min_across", 4.0))
        self.tags = int(meta.get("vocr.tags", 4))
        self.input_name = self.sess.get_inputs()[0].name

    def preprocess(self, g, f=None):
        """(network input [1, 1, H', W'] uint8, factor) of a grey frame: resized by the size policy (INTER_AREA down,
        INTER_LINEAR up) or by the factor f when given, padded right and bottom to a multiple of 32 by edge
        replication."""
        import cv2
        h, w = g.shape
        if f is None:
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

    def detect(self, img, frame=None):
        """[Box] in the image's px, in the decode's order (by confidence). frame=(W, H): the image is a region of a
        W x H frame, read at the scale the size policy gives the whole frame and with the whole frame's pad, as a
        read of the whole frame would read it (read_frame's region)."""
        g = grey_of(img)
        H, W = g.shape
        f = None
        ph = H
        if frame is not None:
            fw, fh = (int(v) for v in frame)
            f = scale_for(self.size, fw, fh)
            ph = fh
        x, (fx, fy, ww, hh) = self.preprocess(g, f)
        boxes = decode(self.maps(x), ww, hh, self.min_cells, self.conf_min_logit, self.pad_pct, self.pad_min,
                       self.min_across, self.tags)
        fp = 0.0
        if self.pad_frame:
            fp = float(max(self.pad_frame_min, round(self.pad_frame * ph / 544.0)))
        out = []
        for b in boxes:
            x0, y0 = max(0.0, b.x / fx - fp), max(0.0, b.y / fy - fp)
            x1, y1 = min(float(W), (b.x + b.w) / fx + fp), min(float(H), (b.y + b.h) / fy + fp)
            if x1 > x0 and y1 > y0:
                out.append(Box(x0, y0, x1 - x0, y1 - y0, b.vertical, b.logit, b.tag, b.tconf, b.cells))
        return out

    def boxes(self, rgb):
        """eval/desktop.py's detector interface: [[x0, y0, x1, y1, vertical, logit, tag]] in frame px."""
        return [[b.x, b.y, b.x + b.w, b.y + b.h, b.vertical, round(b.logit, 4), b.tag] for b in self.detect(rgb)]


MAX_BOXES = 256           # desktop/vocr_desk.h VOCR_DESK_MAX_BOXES: the boxes kept per frame


def decode(maps, w, h, min_cells=2, conf_min_logit=0.0, pad_pct=0.18, pad_min=2.0, min_across=4.0, tags=4,
           max_boxes=MAX_BOXES):
    """[Box] in network-input px (the image's w x h part) from float maps [4 + T, H / 4, W / 4] (module docstring).
    desktop/vocr_desk.c's vocr_desk_decode is the same rule with the same sums in the same order (in double, over the
    cells in raster order), so given the same maps the two give the same boxes bit for bit: the 4-connected components
    of K > 0 (labels in the order of each component's first cell), per component the mean K logit (its confidence),
    the orientation (vertical when the mean O logit > 0), the extent across it from the mean of its cells' centre -
    A1 and centre + A2, along it its cells' extent, grown by max(pad_min, pad_pct x the size across); the tag the
    first class of the largest summed tag logit (class 0 scores 0). Sorted by logit (descending), then y, then x,
    stable; the first max_boxes are kept."""
    import cv2
    hm, wm = -(-h // STRIDE), -(-w // STRIDE)
    k = np.ascontiguousarray(maps[0, :hm, :wm])
    n, lab, stats, _ = cv2.connectedComponentsWithStats((k > 0).astype(np.uint8), connectivity=4)
    if n <= 1:
        return []
    flat = lab.ravel()
    on = np.flatnonzero(flat)
    L = flat[on]
    rows, cols = np.divmod(on, wm)

    def sums(m):
        return np.bincount(L, np.ascontiguousarray(m).ravel()[on].astype(np.float64), minlength=n)
    cnt = np.bincount(L, minlength=n).astype(np.float64)
    cnt[0] = 1.0
    logit = sums(k) / cnt
    vert = sums(maps[1, :hm, :wm]) / cnt > 0
    cc = STRIDE * np.where(vert[L], cols, rows) + (STRIDE - 1) / 2.0
    near = np.bincount(L, cc - np.ascontiguousarray(maps[2, :hm, :wm]).ravel()[on].astype(np.float64),
                       minlength=n) / cnt
    far = np.bincount(L, cc + np.ascontiguousarray(maps[3, :hm, :wm]).ravel()[on].astype(np.float64),
                      minlength=n) / cnt
    tsum = [sums(maps[4 + i, :hm, :wm]) for i in range(tags)]
    out = []
    for c in range(1, n):
        if stats[c, cv2.CC_STAT_AREA] < min_cells or logit[c] < conf_min_logit:
            continue
        size = float(far[c] - near[c])
        if size < min_across:
            continue
        t0, n0 = (cv2.CC_STAT_TOP, cv2.CC_STAT_HEIGHT) if vert[c] else (cv2.CC_STAT_LEFT, cv2.CC_STAT_WIDTH)
        lo, hi = STRIDE * int(stats[c, t0]), STRIDE * int(stats[c, t0] + stats[c, n0])
        pad = size * pad_pct
        if pad_min > pad:
            pad = pad_min
        a0, a1_, b0, b1 = float(near[c]) - pad, float(far[c]) + pad, lo - pad, hi + pad
        x0, x1, y0, y1 = (a0, a1_, b0, b1) if vert[c] else (b0, b1, a0, a1_)
        x0, y0 = (0.0 if x0 < 0.0 else x0), (0.0 if y0 < 0.0 else y0)
        x1, y1 = (float(w) if x1 > w else x1), (float(h) if y1 > h else y1)
        if x1 <= x0 or y1 <= y0:
            continue
        sc = [0.0] + [float(t[c]) for t in tsum]
        tag = 0
        for i in range(1, len(sc)):
            if sc[i] > sc[tag]:
                tag = i
        second = max((v for i, v in enumerate(sc) if i != tag), default=0.0) if tags else 0.0
        out.append(Box(x0, y0, x1 - x0, y1 - y0, int(vert[c]), float(logit[c]), tag, (sc[tag] - second) / cnt[c],
                       int(cnt[c])))
    out.sort(key=lambda b: (-b.logit, b.y, b.x))
    return out[:max_boxes]


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
    (None: all); workers: lines read in parallel on that many threads (or pass a concurrent.futures executor). A region
    is read at the scale and with the box pad the detector's policy gives the whole frame (Detector.detect's frame);
    no benchmark set read a region (docs/desktop.md)."""
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
    boxes = det.detect(sub, frame=(W, H)) if region is not None else det.detect(sub)
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
            return Line("", (x0 + cx, y0 + cy, cw, ch), b.tag, b.logit, 0, 1)
        dark = V.line_dark(sub, (cx, cy, cw, ch))
        r = rec.read(sub[cy:cy + ch, cx:cx + cw], invert=bool(dark), sections=sections, lm=lm)
        for c in r.chars + r.text_chars:
            c.x, c.x0, c.x1 = c.x + x0 + cx, c.x0 + x0 + cx, c.x1 + x0 + cx
            c.y0, c.y1 = y0 + cy, y0 + cy + ch
        return Line(r.text, (x0 + cx, y0 + cy, cw, ch), b.tag, b.logit, int(dark), 0, r.chars, r.text_chars,
                    r.greedy, r.conf_min)

    if executor is not None and len(jobs) > 1:
        return list(executor.map(one, jobs))
    if workers > 1 and len(jobs) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(one, jobs))
    return [one(j) for j in jobs]


# ----------------------------------------------------------------------------- the C path (desktop/libvocrdesk.so)

_CTYPES = []


def _c_types():
    """(CBox, CSet, CLine): the ctypes mirrors of vocr_desk_box, vocr_desk_set and vocr_desk_line (made once)."""
    if _CTYPES:
        return _CTYPES[0]
    import ctypes as C

    class CBox(C.Structure):
        _fields_ = [("x", C.c_double), ("y", C.c_double), ("w", C.c_double), ("h", C.c_double),
                    ("logit", C.c_double), ("tconf", C.c_double), ("vertical", C.c_uint), ("tag", C.c_uint),
                    ("cells", C.c_uint)]

    class CSet(C.Structure):
        _fields_ = [("count", C.c_uint), ("truncated", C.c_uint), ("in_w", C.c_uint), ("in_h", C.c_uint),
                    ("boxes", CBox * MAX_BOXES)]

    class CLine(C.Structure):
        _fields_ = [("x", C.c_int), ("y", C.c_int), ("w", C.c_int), ("h", C.c_int), ("tag", C.c_uint),
                    ("dark", C.c_uint), ("vertical", C.c_uint), ("conf_min", C.c_float), ("text", C.c_char * 1024)]
    _CTYPES.append((CBox, CSet, CLine))
    return _CTYPES[0]


_LIBS = {}


def c_library(path=None):
    """desktop/build/libvocrdesk.so (or `path`, or $VOCR_DESK_LIB) through ctypes, its prototypes set."""
    import ctypes as C
    import os
    from pathlib import Path
    path = str(path or os.environ.get("VOCR_DESK_LIB") or
               Path(__file__).resolve().parents[1] / "desktop" / "build" / "libvocrdesk.so")
    if path in _LIBS:
        return _LIBS[path]
    L = C.CDLL(path)
    CBox, CSet, CLine = _c_types()
    vp, sz, u, i = C.c_void_p, C.c_size_t, C.c_uint, C.c_int
    L.vocr_desk_model_bytes.restype = sz
    L.vocr_desk_set_bytes.restype = sz
    L.vocr_desk_load.argtypes = [vp, vp, sz]
    L.vocr_desk_maps.argtypes = [vp, vp, u, u, vp, sz, C.POINTER(C.POINTER(C.c_float))]
    L.vocr_desk_decode_bytes.argtypes = [u, u, u]
    L.vocr_desk_decode_bytes.restype = sz
    L.vocr_desk_decode.argtypes = [vp, vp, u, u, u, u, vp, sz, C.POINTER(CSet)]
    L.vocr_desk_arena_bytes.argtypes = [vp, u, u, u, u]
    L.vocr_desk_arena_bytes.restype = sz
    L.vocr_desk_detect.argtypes = [vp, vp, u, u, u, u, u, vp, sz, C.POINTER(CSet)]
    L.vocr_desk_resize.argtypes = [vp, u, u, u, vp, u, u, vp]
    L.vocr_desk_resize.restype = i
    L.vocr_desk_policy_ok.argtypes = [u, u]
    L.vocr_desk_policy_ok.restype = i
    L.vocr_desk_resize_scratch.argtypes = [u, u, u, u]
    L.vocr_desk_resize_scratch.restype = sz
    L.vocr_desk_input_size.argtypes = [u, u, u, u] + [C.POINTER(u)] * 4
    L.vocr_desk_page_new.argtypes = [vp, sz, vp, sz, u, i, u, C.POINTER(i)]
    L.vocr_desk_page_new.restype = vp
    L.vocr_desk_page_run.argtypes = [vp, vp, u, u, u]
    L.vocr_desk_page_run_rgb.argtypes = [vp, vp, u, u, u, u]
    L.vocr_desk_page_detect.argtypes = [vp, vp, u, u, u]
    L.vocr_desk_page_count.argtypes = [vp]
    L.vocr_desk_page_count.restype = u
    L.vocr_desk_page_line.argtypes = [vp, u]
    L.vocr_desk_page_line.restype = C.POINTER(CLine)
    L.vocr_desk_page_boxes.argtypes = [vp]
    L.vocr_desk_page_boxes.restype = C.POINTER(CSet)
    L.vocr_desk_page_detect_us.argtypes = [vp]
    L.vocr_desk_page_detect_us.restype = C.c_double
    L.vocr_desk_page_total_us.argtypes = [vp]
    L.vocr_desk_page_total_us.restype = C.c_double
    L.vocr_desk_page_free.argtypes = [vp]
    if L.vocr_desk_set_bytes() != C.sizeof(CSet):
        raise RuntimeError(f"{path}: vocr_desk_set is {L.vocr_desk_set_bytes()} B, ctypes says {C.sizeof(CSet)}")
    _LIBS[path] = L
    return L


def _aligned(data):
    """bytes -> a 64-byte aligned uint8 array holding them (the blobs need 4-byte alignment)."""
    raw = np.zeros(len(data) + 64, np.uint8)
    off = (-raw.ctypes.data) % 64
    buf = raw[off:off + len(data)]
    buf[:] = np.frombuffer(data, np.uint8)
    return buf


def policy_of(size):
    """(kind, n) of a size policy string for the C API: native 0, fit:N 1, long:N 2 (N 1-8192); None: the blob's
    (-1). ValueError for anything else, before the C library sees it (it refuses those too)."""
    if size is None:
        return -1, 0
    if size in ("", "native"):
        return 0, 0
    kind, _, n = str(size).partition(":")
    if not n:
        kind, n = "fit", kind
    if kind not in ("fit", "long") or not n.isdigit() or not 1 <= int(n) <= MAX_SIDE:
        raise ValueError(f"size policy {size!r}: fit:N, long:N (N 1 to {MAX_SIDE}) or native")
    return {"fit": 1, "long": 2}[kind], int(n)


class CModel:
    """A VDSK blob loaded by the C library, for the network, the decode and the resize alone (tests, parity)."""

    def __init__(self, vdsk, lib=None):
        import ctypes as C
        self.L = c_library(lib)
        self.blob = _aligned(open(vdsk, "rb").read() if not isinstance(vdsk, (bytes, bytearray)) else vdsk)
        self.m = np.zeros(self.L.vocr_desk_model_bytes() + 64, np.uint8)
        rc = self.L.vocr_desk_load(self.m.ctypes.data, self.blob.ctypes.data, len(self.blob))
        if rc:
            raise ValueError(f"vocr_desk_load: {rc}")
        self.C = C
        self._arena = np.zeros(0, np.uint8)

    def arena(self, n):
        """A scratch buffer of at least n bytes, kept between calls (a fresh one would page-fault inside a timing)."""
        if self._arena.size < n:
            self._arena = np.zeros(n, np.uint8)
        return self._arena

    def maps(self, x):
        """The network on a padded input (H x W uint8, multiples of 32): [4 + T, H / 4, W / 4] float32."""
        C = self.C
        x = np.ascontiguousarray(x, np.uint8)
        ph, pw = x.shape
        arena = self.arena(self.L.vocr_desk_arena_bytes(self.m.ctypes.data, 0, 0, pw, ph) + 64)
        out = C.POINTER(C.c_float)()
        rc = self.L.vocr_desk_maps(self.m.ctypes.data, x.ctypes.data, pw, ph, arena.ctypes.data, arena.size,
                                   C.byref(out))
        if rc:
            raise ValueError(f"vocr_desk_maps: {rc}")
        n = ph // 4 * (pw // 4) * self.channels()
        hwc = np.ctypeslib.as_array(out, shape=(n,)).reshape(ph // 4, pw // 4, -1)
        return np.ascontiguousarray(hwc.transpose(2, 0, 1)).copy()

    def channels(self):
        import struct
        return 4 + struct.unpack_from("<I", bytes(self.blob[56:60]))[0]

    def decode(self, maps, iw, ih):
        """vocr_desk_decode on [4 + T, mh, mw] float32 maps: [Box] in input px."""
        _, CSet, _ = _c_types()
        hwc = np.ascontiguousarray(np.asarray(maps, np.float32).transpose(1, 2, 0))
        mh, mw, _ = hwc.shape
        scratch = np.zeros(self.L.vocr_desk_decode_bytes(mw, mh, self.channels() - 4) + 64, np.uint8)
        st = CSet()
        rc = self.L.vocr_desk_decode(self.m.ctypes.data, hwc.ctypes.data, mw, mh, iw, ih, scratch.ctypes.data,
                                     scratch.size, self.C.byref(st))
        if rc:
            raise ValueError(f"vocr_desk_decode: {rc}")
        return [Box(b.x, b.y, b.w, b.h, int(b.vertical), b.logit, int(b.tag), b.tconf, int(b.cells))
                for b in st.boxes[:st.count]]

    def detect(self, grey, size=None):
        """vocr_desk_detect on a grey frame: [Box] in frame px."""
        _, CSet, _ = _c_types()
        g = np.ascontiguousarray(grey_of(grey))
        h, w = g.shape
        kind, n = policy_of(size)
        if kind < 0:
            import struct
            kind, n = struct.unpack_from("<II", bytes(self.blob[64:72]))
        arena = self.arena(self.L.vocr_desk_arena_bytes(self.m.ctypes.data, kind, n, w, h) + 64)
        st = CSet()
        rc = self.L.vocr_desk_detect(self.m.ctypes.data, g.ctypes.data, w, h, w, kind, n, arena.ctypes.data,
                                     arena.size, self.C.byref(st))
        if rc:
            raise ValueError(f"vocr_desk_detect: {rc}")
        return [Box(b.x, b.y, b.w, b.h, int(b.vertical), b.logit, int(b.tag), b.tconf, int(b.cells))
                for b in st.boxes[:st.count]]

    def resize(self, grey, dw, dh):
        g = np.ascontiguousarray(grey, np.uint8)
        h, w = g.shape
        out = np.zeros((dh, dw), np.uint8)
        scratch = np.zeros(self.L.vocr_desk_resize_scratch(w, h, dw, dh) + 64, np.uint8)
        rc = self.L.vocr_desk_resize(g.ctypes.data, w, h, w, out.ctypes.data, dw, dh, scratch.ctypes.data)
        if rc:
            raise ValueError(f"vocr_desk_resize: {rc}")
        return out


class CDesk:
    """The C page pipeline (desktop/libvocrdesk.so): the desktop detector's VDSK blob and a VOCR6 recognizer blob
    (.vocr, read by runtime/vocr.c in W8I with its head sections), lines read on `threads` threads in C."""

    def __init__(self, det, rec, threads=1, size=None, lib=None):
        import ctypes as C
        self.C = C
        self.L = c_library(lib)
        self.det = _aligned(open(det, "rb").read())
        self.rec = _aligned(open(rec, "rb").read())
        kind, n = policy_of(size)
        err = C.c_int(0)
        self.p = self.L.vocr_desk_page_new(self.det.ctypes.data, len(self.det), self.rec.ctypes.data, len(self.rec),
                                           int(threads), kind, n, C.byref(err))
        if not self.p:
            raise ValueError(f"vocr_desk_page_new: {err.value}")

    def __del__(self):
        if getattr(self, "p", None):
            self.L.vocr_desk_page_free(self.p)
            self.p = None

    def _set(self):
        st = self.L.vocr_desk_page_boxes(self.p).contents
        return [Box(b.x, b.y, b.w, b.h, int(b.vertical), b.logit, int(b.tag), b.tconf, int(b.cells))
                for b in st.boxes[:st.count]]

    def detect(self, img):
        g = np.ascontiguousarray(grey_of(img))
        rc = self.L.vocr_desk_page_detect(self.p, g.ctypes.data, g.shape[1], g.shape[0], g.shape[1])
        if rc:
            raise ValueError(f"vocr_desk_page_detect: {rc}")
        return self._set()

    def boxes(self, rgb):
        """eval/desktop.py's detector interface: [[x0, y0, x1, y1, vertical, logit, tag]] in frame px."""
        return [[b.x, b.y, b.x + b.w, b.y + b.h, b.vertical, round(b.logit, 4), b.tag] for b in self.detect(rgb)]

    def page(self, img):
        """Every line of a frame in reading order: [(text, (x, y, w, h), tag, dark, vertical)]."""
        a = np.ascontiguousarray(img, np.uint8)
        if a.ndim == 3:
            rc = self.L.vocr_desk_page_run_rgb(self.p, a.ctypes.data, a.shape[1], a.shape[0], a.strides[0],
                                               a.shape[2])
        else:
            rc = self.L.vocr_desk_page_run(self.p, a.ctypes.data, a.shape[1], a.shape[0], a.strides[0])
        if rc:
            raise ValueError(f"vocr_desk_page_run: {rc}")
        out = []
        for i in range(self.L.vocr_desk_page_count(self.p)):
            ln = self.L.vocr_desk_page_line(self.p, i).contents
            out.append((ln.text.decode("utf-8", "replace"), (ln.x, ln.y, ln.w, ln.h), int(ln.tag), int(ln.dark),
                        int(ln.vertical)))
        return out

    def times_us(self):
        return self.L.vocr_desk_page_detect_us(self.p), self.L.vocr_desk_page_total_us(self.p)


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
