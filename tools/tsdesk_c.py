#!/usr/bin/env python3
"""textspot-desktop's C path through ctypes (issue #173, docs/textspot-desktop.md): desktop/libvocrdesk.so's
vocr_tsd_* functions (desktop/vocr_tsd.h), the detector of a VTSD blob (train/tsdesk_blob.py) and its page pipeline
with VOCR's recognizer in C (runtime/vocr.c, W8I, the blob's head sections on).

    from tsdesk_c import CModel, CPage
    det = CModel("textspot_desktop.vtsd", threads=1)          # the C detector (AVX2 + FMA, or plain C)
    boxes = det.detect(frame)                                   # [tsdesk.Box], frame px (tsdesk.Detector.detect's)
    page = CPage("textspot_desktop.vtsd", "VOCR6.2_L_w8_2l.vocr", threads=4)
    for ln in page.page(frame):                                 # [tsdesk.Line] in reading order
        ln.text, ln.box, ln.crop, ln.tag

frame: H x W grey or H x W x 3 / 4 RGB(A) uint8 (rows may be strided). The library is desktop/build/libvocrdesk.so,
$VOCR_DESK_LIB or `lib` (make -C desktop). CModel.maps / .decode run the network and the decode alone (parity
tests); given the same maps, the C decode gives tools/tsdesk.decode's boxes bit for bit.
"""
import ctypes as C
import importlib.util
import os
import re
import struct
import sys
from pathlib import Path

import numpy as np


def _tools_tsdesk():
    """tools/tsdesk.py (the reference pipeline: Box, Line, grey_of), loaded by its path: as `tsdesk` when no module
    holds the name, else (eval/tsdesk.py shares it) as `tsdesk_tools`."""
    here = Path(__file__).resolve().with_name("tsdesk.py")
    mod = sys.modules.get("tsdesk")
    if mod is not None and Path(getattr(mod, "__file__", None) or "").resolve() == here:
        return mod
    name = "tsdesk" if mod is None else "tsdesk_tools"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, here)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


T = _tools_tsdesk()

MAX_BOXES = 256           # vocr_tsd.h VOCR_TSD_MAX_BOXES
TEXT = 1024               # VOCR_TSD_TEXT
HEADER = 160


class CBox(C.Structure):
    _fields_ = [("x", C.c_double), ("y", C.c_double), ("w", C.c_double), ("h", C.c_double),
                ("logit", C.c_double), ("tconf", C.c_double), ("vertical", C.c_uint), ("tag", C.c_uint),
                ("cells", C.c_uint)]


class CSet(C.Structure):
    _fields_ = [("count", C.c_uint), ("truncated", C.c_uint), ("in_w", C.c_uint), ("in_h", C.c_uint),
                ("boxes", CBox * MAX_BOXES)]


class CLine(C.Structure):
    _fields_ = [("x", C.c_int), ("y", C.c_int), ("w", C.c_int), ("h", C.c_int), ("bx", C.c_double),
                ("by", C.c_double), ("bw", C.c_double), ("bh", C.c_double), ("logit", C.c_double),
                ("tag", C.c_uint), ("dark", C.c_uint), ("vertical", C.c_uint), ("conf_min", C.c_float),
                ("text", C.c_char * TEXT)]


class COpts(C.Structure):
    _fields_ = [("stride", C.c_uint), ("tags", C.c_uint), ("min_cells", C.c_uint), ("max_boxes", C.c_uint),
                ("conf_min_logit", C.c_double), ("min_across", C.c_double), ("along_reach", C.c_double)]


_LIBS = {}


def default_lib():
    return os.environ.get("VOCR_DESK_LIB") or str(Path(__file__).resolve().parents[1] / "desktop" / "build" /
                                                  "libvocrdesk.so")


def c_library(path=None):
    """The library (path, $VOCR_DESK_LIB or desktop/build/libvocrdesk.so) through ctypes, its prototypes set."""
    path = str(path or default_lib())
    if path in _LIBS:
        return _LIBS[path]
    L = C.CDLL(path)
    vp, sz, u, i, d = C.c_void_p, C.c_size_t, C.c_uint, C.c_int, C.c_double
    for name in ("vocr_tsd_model_bytes", "vocr_tsd_set_bytes", "vocr_tsd_box_bytes", "vocr_tsd_line_bytes"):
        getattr(L, name).restype = sz
        getattr(L, name).argtypes = []
    L.vocr_tsd_simd.restype = i
    L.vocr_tsd_load.argtypes = [vp, vp, sz]
    L.vocr_tsd_input_size.argtypes = [u, u, u, u] + [C.POINTER(u)] * 4
    L.vocr_tsd_maps_bytes.argtypes = [vp, u, u, u]
    L.vocr_tsd_maps_bytes.restype = sz
    L.vocr_tsd_maps_mt.argtypes = [vp, vp, u, u, vp, sz, C.POINTER(C.POINTER(C.c_float)), vp, vp, u]
    L.vocr_tsd_decode_bytes.argtypes = [u, u, u]
    L.vocr_tsd_decode_bytes.restype = sz
    L.vocr_tsd_decode_ex.argtypes = [C.POINTER(COpts), vp, u, u, u, u, vp, sz, C.POINTER(CSet)]
    L.vocr_tsd_arena_bytes.argtypes = [vp, u, u, u, u, u, u]
    L.vocr_tsd_arena_bytes.restype = sz
    L.vocr_tsd_detect_mt.argtypes = [vp, vp, u, u, u, u, u, u, vp, sz, C.POINTER(CSet), vp, vp, u]
    L.vocr_tsd_detect_ex.argtypes = [vp, vp, u, u, u, u, u, u, vp, vp, sz, C.POINTER(CSet), vp, vp, u]
    L.vocr_tsd_arena_bytes_ex.argtypes = [vp, u, u, u, u, u, u, i]
    L.vocr_tsd_arena_bytes_ex.restype = sz
    L.vocr_tsd_page_set_refine.argtypes = [vp, vp]
    L.vocr_tsdr_nopts.restype = i
    L.vocr_tsd_pool_new.argtypes = [u]
    L.vocr_tsd_pool_new.restype = vp
    L.vocr_tsd_pool_free.argtypes = [vp]
    L.vocr_desk_resize.argtypes = [vp, u, u, u, vp, u, u, vp]
    L.vocr_desk_resize_scratch.argtypes = [u, u, u, u]
    L.vocr_desk_resize_scratch.restype = sz
    L.vocr_desk_grey.argtypes = [vp, u, u, u, u, vp]
    L.vocr_tsd_page_new.argtypes = [vp, sz, vp, sz, u, i, u, u, u, u, C.POINTER(i)]
    L.vocr_tsd_page_new.restype = vp
    L.vocr_tsd_page_set_tags.argtypes = [vp, u]
    L.vocr_tsd_page_run.argtypes = [vp, vp, u, u, u, u]
    L.vocr_tsd_page_detect.argtypes = [vp, vp, u, u, u, u]
    L.vocr_tsd_page_count.argtypes = [vp]
    L.vocr_tsd_page_count.restype = u
    L.vocr_tsd_page_line.argtypes = [vp, u]
    L.vocr_tsd_page_line.restype = C.POINTER(CLine)
    L.vocr_tsd_page_boxes.argtypes = [vp]
    L.vocr_tsd_page_boxes.restype = C.POINTER(CSet)
    L.vocr_tsd_page_pad.argtypes = [vp, C.POINTER(i), C.POINTER(i)]
    L.vocr_tsd_page_detect_us.argtypes = [vp]
    L.vocr_tsd_page_detect_us.restype = d
    L.vocr_tsd_page_total_us.argtypes = [vp]
    L.vocr_tsd_page_total_us.restype = d
    L.vocr_tsd_page_free.argtypes = [vp]
    for name, cls in (("vocr_tsd_set_bytes", CSet), ("vocr_tsd_box_bytes", CBox), ("vocr_tsd_line_bytes", CLine)):
        if getattr(L, name)() != C.sizeof(cls):
            raise RuntimeError(f"{path}: {name} {getattr(L, name)()} B, ctypes says {C.sizeof(cls)}")
    L.runner = C.cast(L.vocr_tsd_pool_run, vp)
    _LIBS[path] = L
    return L


def aligned(data):
    """bytes -> a 64-byte aligned uint8 array holding them (the blobs need 4-byte alignment)."""
    raw = np.zeros(len(data) + 64, np.uint8)
    off = (-raw.ctypes.data) % 64
    buf = raw[off:off + len(data)]
    buf[:] = np.frombuffer(bytes(data), np.uint8)
    return buf


def _blob_bytes(blob):
    if isinstance(blob, (bytes, bytearray, memoryview)):
        return bytes(blob)
    return Path(blob).read_bytes()


def policy_of(size, default=None):
    """(kind, n) of an input policy for the C API: native 0, fit:N 1, long:N 2; None: `default` (the blob's)."""
    if size is None:
        return default
    if size in ("", "native"):
        return 0, 0
    kind, _, n = str(size).partition(":")
    if not n:
        kind, n = "fit", kind
    return {"fit": 1, "long": 2}[kind], int(n)


def pad_rule(pad):
    """(h, v, px) of a reading pad: "auto" / None -> the k rule (4, 4) (max(2, round(k H / 544)) frame px, eval's
    pad "auto"); "kN" / "kNvM" (eval/tsdesk.PAD_RULES' names) or a (kh, kv) tuple -> the k rule; an int -> fixed px."""
    if pad in (None, "auto"):
        return 4, 4, 0
    if isinstance(pad, (tuple, list)):
        return int(pad[0]), int(pad[1]), 0
    if isinstance(pad, (int, np.integer)):
        return int(pad), int(pad), 1
    mt = re.fullmatch(r"k(\d+)(?:v(\d+))?", str(pad))
    if mt:
        kh = int(mt.group(1))
        return kh, int(mt.group(2) or kh), 0
    if str(pad).isdigit():
        return int(pad), int(pad), 1
    raise ValueError(f"reading pad {pad!r}: auto, kN, kNvM, (kh, kv) or N px")


def refine_opts(refine, lib):
    """The C refinement's options (a vocr_tsdr_opts as ints in tools/tsdesk_refine.KEYS order) for refine: None / False
    (off: None), True (tools/tsdesk_refine.DEFAULTS) or a dict of its options."""
    if refine in (None, False):
        return None
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import tsdesk_refine as TR
    o = TR.options({} if refine is True else refine)
    if lib.vocr_tsdr_nopts() != len(TR.KEYS):
        raise RuntimeError("the C refinement's options differ from tools/tsdesk_refine.KEYS")
    return (C.c_int * len(TR.KEYS))(*[o[k] for k in TR.KEYS])


def _image(img):
    """(array, w, h, row stride, channels) of a grey or RGB(A) uint8 image for the C API (rows may be strided)."""
    a = np.asarray(img)
    if a.dtype != np.uint8:
        raise TypeError(f"images are uint8, not {a.dtype}")
    if a.ndim == 3 and a.shape[2] in (3, 4):
        if a.strides[2] != 1 or a.strides[1] != a.shape[2]:
            a = np.ascontiguousarray(a)
        ch = a.shape[2]
    elif a.ndim == 2:
        if a.strides[1] != 1:
            a = np.ascontiguousarray(a)
        ch = 1
    else:
        raise ValueError("an H x W grey or H x W x 3 / 4 RGB(A) image")
    if a.strides[0] < a.shape[1] * ch:
        a = np.ascontiguousarray(a)
    return a, a.shape[1], a.shape[0], a.strides[0], ch


def _boxes(st):
    return [T.Box(b.x, b.y, b.w, b.h, int(b.vertical), b.logit, int(b.tag), b.tconf, int(b.cells))
            for b in st.boxes[:st.count]]


class CModel:
    """A VTSD blob loaded by the C library: the detector (detect), and its network, decode and resize alone."""

    def __init__(self, blob, threads=1, lib=None):
        self.L = c_library(lib)
        self.blob = aligned(_blob_bytes(blob))
        self.m = np.zeros(self.L.vocr_tsd_model_bytes() + 64, np.uint8)
        rc = self.L.vocr_tsd_load(self.m.ctypes.data, self.blob.ctypes.data, len(self.blob))
        if rc:
            raise ValueError(f"vocr_tsd_load: {rc}")
        hdr = bytes(self.blob[:HEADER])
        self.hs = struct.unpack_from("<I", hdr, 20)[0]
        self.tags, self.out = struct.unpack_from("<II", hdr, 68)
        mc, mb, kind, n = struct.unpack_from("<4I", hdr, 80)
        conf, across, reach = struct.unpack_from("<3d", hdr, 128)
        self.default_policy = (kind, n)
        self.size = "native" if kind == 0 else f"{('fit', 'long')[kind - 1]}:{n}"
        self.decode_opts = {"min_cells": mc, "conf_min_logit": conf, "min_across": across, "along_reach": reach,
                            "max_boxes": mb}
        self.threads = int(threads)
        self.pool = None
        if self.threads > 1:
            self.pool = self.L.vocr_tsd_pool_new(self.threads)
            if not self.pool:
                raise ValueError(f"vocr_tsd_pool_new({threads})")
        self._arena = np.zeros(0, np.uint8)
        self._set = CSet()

    def __del__(self):
        if getattr(self, "pool", None):
            self.L.vocr_tsd_pool_free(self.pool)
            self.pool = None

    def _run(self):
        """(runner, pool, tasks): one row band per thread (vocr_tsd_page.c's TASKS_OF)."""
        return (self.L.runner, self.pool, self.threads) if self.pool else (None, None, 1)

    def arena(self, n):
        """A scratch buffer of at least n bytes, kept between calls (a fresh one would page-fault inside a timing)."""
        if self._arena.size < n:
            self._arena = np.zeros(n, np.uint8)
        return self._arena

    def maps(self, x):
        """The network on a padded grey input (H x W uint8, multiples of 32, or [1, 1, H, W]):
        [6 + T, H / hs, W / hs] float32 (the ONNX graph's output layout)."""
        x = np.ascontiguousarray(np.asarray(x, np.uint8).reshape(np.asarray(x).shape[-2:]))
        ph, pw = x.shape
        arena = self.arena(self.L.vocr_tsd_maps_bytes(self.m.ctypes.data, pw, ph, self._run()[2]) + 64)
        out = C.POINTER(C.c_float)()
        run, pool, nt = self._run()
        rc = self.L.vocr_tsd_maps_mt(self.m.ctypes.data, x.ctypes.data, pw, ph, arena.ctypes.data, arena.size,
                                     C.byref(out), run, pool, nt)
        if rc:
            raise ValueError(f"vocr_tsd_maps: {rc}")
        mh, mw = ph // self.hs, pw // self.hs
        hwc = np.ctypeslib.as_array(out, shape=(mh * mw * self.out,)).reshape(mh, mw, self.out)
        return np.ascontiguousarray(hwc.transpose(2, 0, 1))

    def decode(self, maps, iw, ih, **opts):
        """vocr_tsd_decode on [6 + T, mh, mw] float32 maps: [Box] in input px (tsdesk.decode's); opts override the
        blob's decode constants (min_cells, conf_min_logit, min_across, along_reach, max_boxes)."""
        o = dict(self.decode_opts)
        o.update(opts)
        hwc = np.ascontiguousarray(np.asarray(maps, np.float32).transpose(1, 2, 0))
        mh, mw, ch = hwc.shape
        tags = ch - 6
        co = COpts(self.hs, tags, int(o["min_cells"]), int(o["max_boxes"]), float(o["conf_min_logit"]),
                   float(o["min_across"]), float(o["along_reach"]))
        scratch = np.zeros(self.L.vocr_tsd_decode_bytes(mw, mh, tags) + 64, np.uint8)
        st = CSet()
        rc = self.L.vocr_tsd_decode_ex(C.byref(co), hwc.ctypes.data, mw, mh, int(iw), int(ih), scratch.ctypes.data,
                                       scratch.size, C.byref(st))
        if rc:
            raise ValueError(f"vocr_tsd_decode: {rc}")
        self.truncated = int(st.truncated)
        return _boxes(st)

    def detect(self, img, size=None, refine=None):
        """vocr_tsd_detect_ex on a grey or RGB(A) frame in memory: [Box] in frame px (tsdesk.Detector.detect's), at
        the input policy `size` (None: the blob's); refine: None (off), True or a dict of tools/tsdesk_refine's options
        (tsdesk.Detector's refine: every horizontal box snapped to its line's ink on the full-resolution luma)."""
        a, w, h, stride, ch = _image(img)
        kind, n = policy_of(size, self.default_policy)
        run, pool, nt = self._run()
        ro = self._refine_opts(refine)
        need = self.L.vocr_tsd_arena_bytes_ex(self.m.ctypes.data, kind, n, w, h, ch, nt, int(ro is not None))
        if not need:
            raise ValueError(f"a {w} x {h} frame")
        arena = self.arena(need)
        st = self._set
        rc = self.L.vocr_tsd_detect_ex(self.m.ctypes.data, a.ctypes.data, w, h, stride, ch, kind, n,
                                       C.cast(ro, C.c_void_p) if ro is not None else None, arena.ctypes.data,
                                       arena.size, C.byref(st), run, pool, nt)
        if rc:
            raise ValueError(f"vocr_tsd_detect: {rc}")
        self.truncated = int(st.truncated)
        return _boxes(st)

    def _refine_opts(self, refine):
        """refine_opts(refine), kept per setting (the options are parsed once)."""
        if refine in (None, False):
            return None
        key = repr(sorted(refine.items())) if isinstance(refine, dict) else "defaults"
        cache = self.__dict__.setdefault("_ropts", {})
        if key not in cache:
            cache[key] = refine_opts(refine, self.L)
        return cache[key]

    def boxes(self, rgb, size=None, refine=None):
        """eval/desktop.py's detector interface: [[x0, y0, x1, y1, vertical, logit, tag]] in frame px."""
        return [[b.x, b.y, b.x + b.w, b.y + b.h, b.vertical, round(b.logit, 4), b.tag]
                for b in self.detect(rgb, size, refine)]

    def input_size(self, w, h, size=None):
        """(iw, ih, pw, ph): the network input's image part and its padded size for a w x h frame."""
        kind, n = policy_of(size, self.default_policy)
        v = [C.c_uint() for _ in range(4)]
        self.L.vocr_tsd_input_size(kind, n, w, h, *[C.byref(x) for x in v])
        return tuple(int(x.value) for x in v)

    def resize(self, grey, dw, dh):
        """vocr_desk_resize (vocr-desktop's integer resize, the C detector's): a grey image to dw x dh."""
        g = np.ascontiguousarray(grey, np.uint8)
        h, w = g.shape
        out = np.zeros((dh, dw), np.uint8)
        scratch = np.zeros(self.L.vocr_desk_resize_scratch(w, h, dw, dh) + 64, np.uint8)
        self.L.vocr_desk_resize(g.ctypes.data, w, h, w, out.ctypes.data, dw, dh, scratch.ctypes.data)
        return out

    def preprocess(self, img, size=None):
        """(network input [1, 1, ph, pw] uint8, (fx, fy, iw, ih)) as the C detector builds it: the integer luma, the
        policy's integer resize (vocr_desk_resize), edge padding to multiples of 32 (tsdesk.Detector.preprocess with
        the C resize in place of OpenCV's)."""
        g = T.grey_of(img)
        h, w = g.shape
        iw, ih, pw, ph = self.input_size(w, h, size)
        if (iw, ih) != (w, h):
            g = self.resize(g, iw, ih)
        x = np.pad(g, ((0, ph - ih), (0, pw - iw)), mode="edge")
        return np.ascontiguousarray(x, np.uint8)[None, None], (iw / w, ih / h, iw, ih)


class CPage:
    """The C page pipeline: the VTSD detector and a VOCR6 recognizer blob (.vocr, runtime/vocr.c W8I with its head
    sections), every horizontal box (refined when refine is set, as CModel.detect refines) grown by the reading pad
    (pad_rule) and read on `threads` threads in C."""

    def __init__(self, det, rec, threads=1, size=None, pad="auto", tags=None, refine=None, lib=None):
        self.L = c_library(lib)
        self.det = aligned(_blob_bytes(det))
        self.rec = aligned(_blob_bytes(rec))
        kind, n = policy_of(size, (-1, 0))
        ph, pv, px = pad_rule(pad)
        err = C.c_int(0)
        self.p = self.L.vocr_tsd_page_new(self.det.ctypes.data, len(self.det), self.rec.ctypes.data, len(self.rec),
                                          int(threads), kind, n, ph, pv, px, C.byref(err))
        if not self.p:
            raise ValueError(f"vocr_tsd_page_new: {err.value}")
        if tags is not None:
            self.L.vocr_tsd_page_set_tags(self.p, sum(1 << int(t) for t in tags))
        self._ropts = refine_opts(refine, self.L)       # the page copies the options
        if self._ropts is not None:
            self.L.vocr_tsd_page_set_refine(self.p, C.cast(self._ropts, C.c_void_p))

    def __del__(self):
        if getattr(self, "p", None):
            self.L.vocr_tsd_page_free(self.p)
            self.p = None

    def detect(self, img):
        a, w, h, stride, ch = _image(img)
        rc = self.L.vocr_tsd_page_detect(self.p, a.ctypes.data, w, h, stride, ch)
        if rc:
            raise ValueError(f"vocr_tsd_page_detect: {rc}")
        return _boxes(self.L.vocr_tsd_page_boxes(self.p).contents)

    def page(self, img):
        """Every line of a frame in reading order: [tsdesk.Line] (text, tight box, crop read, tag, logit, dark,
        vertical); vertical boxes come back without text."""
        a, w, h, stride, ch = _image(img)
        rc = self.L.vocr_tsd_page_run(self.p, a.ctypes.data, w, h, stride, ch)
        if rc:
            raise ValueError(f"vocr_tsd_page_run: {rc}")
        out = []
        for i in range(self.L.vocr_tsd_page_count(self.p)):
            ln = self.L.vocr_tsd_page_line(self.p, i).contents
            out.append(T.Line(ln.text.decode("utf-8", "replace"), (ln.bx, ln.by, ln.bw, ln.bh),
                              (ln.x, ln.y, ln.w, ln.h), int(ln.tag), ln.logit, int(ln.dark), int(ln.vertical)))
        return out

    def pad(self):
        """The reading pad of the last run, frame px (horizontal, vertical)."""
        a, b = C.c_int(), C.c_int()
        self.L.vocr_tsd_page_pad(self.p, C.byref(a), C.byref(b))
        return a.value, b.value

    def times_us(self):
        """(the detector, the whole page) of the last run, us."""
        return self.L.vocr_tsd_page_detect_us(self.p), self.L.vocr_tsd_page_total_us(self.p)
