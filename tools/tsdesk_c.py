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

The round-2 options (protocol v2 amendment 4's C6) are tools/tsdesk's, as dicts in its option names, None = off =
the plain boxes bit for bit: decode (DEFAULTS and DECODE_EXTRA: vote_pct in C; saddle_pct, rescue_milli and
clamp_ends have no C twin and are refused), post (POST_DEFAULTS; gap_pct refused) and, for the page, drop
(DROP_DEFAULTS):
    det.detect(frame, refine=True, decode={"vote_pct": 70},
               post={"merge_pct": 70, "rows_pct": 65, "valley_pct": 0, "ext_pct": 1200})
    CPage(det_blob, rec_blob, pad="k6v4", refine=True, decode={...}, post={...},
          drop={"nojp": 2, "nojp_gap_pct": 200, "nojp_anchor_pct": 50, "rep_pct": 50})

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
                ("conf_mean", C.c_float), ("text", C.c_char * TEXT)]


class COpts(C.Structure):
    _fields_ = [("stride", C.c_uint), ("tags", C.c_uint), ("min_cells", C.c_uint), ("max_boxes", C.c_uint),
                ("conf_min_logit", C.c_double), ("min_across", C.c_double), ("along_reach", C.c_double),
                ("vote_pct", C.c_int)]


NO_C_POST = ("ext_band_pct", "touch_pct", "ksplit_pct", "ksplit_k", "ksplit_min_pct")   # post options, no C twin
POST_KEYS = tuple(k for k in T.POST_DEFAULTS if k not in NO_C_POST)          # vocr_tsdp_opts' fields
DROP_KEYS = tuple(T.DROP_DEFAULTS)          # vocr_tsdd_opts' (the cut's three: the C page runs it)


class CPost(C.Structure):
    _fields_ = [(k, C.c_int) for k in POST_KEYS]


class CDrop(C.Structure):
    _fields_ = [(k, C.c_int) for k in DROP_KEYS]


class CDetOpts(C.Structure):                # vocr_tsd_opts
    _fields_ = [("decode", C.POINTER(COpts)), ("refine", C.c_void_p), ("post", C.POINTER(CPost))]


class CPBox(C.Structure):                   # vocr_tsdp_box
    _fields_ = [("x", C.c_double), ("y", C.c_double), ("w", C.c_double), ("h", C.c_double),
                ("logit", C.c_double), ("tconf", C.c_double), ("vertical", C.c_uint), ("tag", C.c_uint),
                ("cells", C.c_uint)]


NO_C_TWIN = ("saddle_pct", "rescue_milli", "clamp_ends", "along_pct")   # DECODE_EXTRA options the C decode lacks


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
    L.vocr_tsd_policy_ok.argtypes = [u, u]
    L.vocr_tsd_policy_ok.restype = i
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
    L.vocr_tsd_resize.argtypes = [vp, u, u, u, vp, u, u, vp]
    L.vocr_tsd_resize_scratch.argtypes = [u, u, u, u]
    L.vocr_tsd_resize_scratch.restype = sz
    L.vocr_tsd_grey.argtypes = [vp, u, u, u, u, vp]
    L.vocr_tsd_page_new.argtypes = [vp, sz, vp, sz, u, i, u, u, u, u, C.POINTER(i)]
    L.vocr_tsd_page_new.restype = vp
    L.vocr_tsd_page_set_tags.argtypes = [vp, u]
    L.vocr_tsd_page_set_rule.argtypes = [vp, u, d, d, d]
    L.vocr_tsd_page_set_rule.restype = i
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
    L.vocr_tsd_decode_bytes_ex.argtypes = [u, u, u, C.POINTER(COpts)]
    L.vocr_tsd_decode_bytes_ex.restype = sz
    L.vocr_tsd_arena_bytes_opt.argtypes = [vp, u, u, u, u, u, u, C.POINTER(CDetOpts)]
    L.vocr_tsd_arena_bytes_opt.restype = sz
    L.vocr_tsd_detect_opt.argtypes = [vp, vp, u, u, u, u, u, u, C.POINTER(CDetOpts), vp, sz, C.POINTER(CSet), vp,
                                      vp, u]
    L.vocr_tsd_page_set_decode.argtypes = [vp, C.POINTER(COpts)]
    L.vocr_tsd_page_set_post.argtypes = [vp, C.POINTER(CPost)]
    L.vocr_tsd_page_set_drop.argtypes = [vp, C.POINTER(CDrop)]
    L.vocr_tsdp_nopts.restype = i
    L.vocr_tsdd_nopts.restype = i
    L.vocr_tsdp_defaults.argtypes = [C.POINTER(CPost)]
    L.vocr_tsdd_defaults.argtypes = [C.POINTER(CDrop)]
    L.vocr_tsdp_scratch_bytes.argtypes = [u, u, u]
    L.vocr_tsdp_scratch_bytes.restype = sz
    L.vocr_tsdp_pre.argtypes = [vp, u, u, u, C.POINTER(CPBox), C.POINTER(u), u, C.POINTER(CPost), vp, sz]
    L.vocr_tsdp_pre.restype = i
    L.vocr_tsdp_post.argtypes = [vp, u, u, u, C.POINTER(CPBox), C.POINTER(u), C.POINTER(CPost), vp, sz]
    L.vocr_tsdp_post.restype = i
    L.vocr_tsdd_drop.argtypes = [C.POINTER(d), C.POINTER(C.c_char_p), C.POINTER(C.c_float), u, C.POINTER(CDrop),
                                 C.POINTER(C.c_ubyte)]
    L.vocr_tsdd_drop.restype = i
    L.vocr_tsdc_cut.argtypes = [vp, u, u, u, C.POINTER(d), C.POINTER(d), C.POINTER(C.c_ubyte), u, i, i, vp, sz,
                                C.POINTER(d), C.POINTER(d), C.POINTER(u), C.POINTER(u), C.POINTER(u)]
    L.vocr_tsdc_cut.restype = i
    L.vocr_tsdc_is_mark.argtypes = [C.c_char_p, u]
    L.vocr_tsdc_is_mark.restype = i
    L.vocr_tsd_page_set_charbox_cal.argtypes = [vp, C.POINTER(i)]
    L.vocr_tsd_page_set_charbox_cal.restype = i
    L.vocr_tsd_page_line_chars.argtypes = [vp, u, C.POINTER(C.c_int32), C.POINTER(u), u]
    L.vocr_tsd_page_line_chars.restype = u
    L.vocr_tsdd_has_japanese.argtypes = [C.c_char_p]
    L.vocr_tsdd_has_japanese.restype = i
    L.vocr_tsdd_japanese_share.argtypes = [C.c_char_p]
    L.vocr_tsdd_japanese_share.restype = d
    for name, cls in (("vocr_tsd_set_bytes", CSet), ("vocr_tsd_box_bytes", CBox), ("vocr_tsd_line_bytes", CLine)):
        if getattr(L, name)() != C.sizeof(cls):
            raise RuntimeError(f"{path}: {name} {getattr(L, name)()} B, ctypes says {C.sizeof(cls)}")
    if L.vocr_tsdp_nopts() != len(POST_KEYS) or L.vocr_tsdd_nopts() != len(DROP_KEYS):
        raise RuntimeError(f"{path}: the post-pass / reading filter options differ from tools/tsdesk's")
    cp, cd = CPost(), CDrop()
    L.vocr_tsdp_defaults(C.byref(cp))
    L.vocr_tsdd_defaults(C.byref(cd))
    if [getattr(cp, k) for k in POST_KEYS] != [T.POST_DEFAULTS[k] for k in POST_KEYS] or \
            [getattr(cd, k) for k in DROP_KEYS] != [T.DROP_DEFAULTS[k] for k in DROP_KEYS]:
        raise RuntimeError(f"{path}: the C post-pass / reading-filter defaults differ from tools/tsdesk's")
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


RULE_FRAME, RULE_LINE, RULE_PX, RULE_CLIP = 0, 1, 2, 3      # vocr_tsd.h VOCR_TSD_RULE_*
_NUM = r"(\d+(?:\.\d*)?|\.\d+)"


def rule_of(pad):
    """(kind, a, b, c) of a reading rule (eval/tsdesk_eval.margin_hv's rules; vocr_tsd_page_set_rule's arguments):
      "auto" / None        the frame rule k4 (eval's pad "auto")
      "kN", "kAvB"         (RULE_FRAME, A, B, 2): max(2, round(A H / 544)) px left and right, max(2, round(B H / 544))
                           top and bottom for an H px tall frame; a (kh, kv) tuple likewise
      "kAvBfC"             (RULE_FRAME, A, B, C): the horizontal pad at least C px (protocol v2 amendment 2)
      "hXvY"               (RULE_LINE, X, Y, 0): X times the line's height along it and Y times it across (amendment 1)
      "cXvY"               (RULE_CLIP, X, Y, 0): hXvY's margins, each side cut to half the gap to the nearest other box
                           of the frame on that side (amendment 3; rule_sides)
      N (int, or digits)   (RULE_PX, N, N, 0): N px on every side"""
    if pad in (None, "auto"):
        return RULE_FRAME, 4.0, 4.0, 2.0
    if isinstance(pad, (tuple, list)):
        return RULE_FRAME, float(pad[0]), float(pad[1]), 2.0
    if isinstance(pad, (int, np.integer)):
        return RULE_PX, float(pad), float(pad), 0.0
    s = str(pad)
    if s.isdigit():
        return RULE_PX, float(s), float(s), 0.0
    mt = re.fullmatch(rf"k{_NUM}(?:v{_NUM})?", s)
    if mt:
        a = float(mt.group(1))
        return RULE_FRAME, a, float(mt.group(2) or a), 2.0
    mt = re.fullmatch(rf"k{_NUM}v{_NUM}f{_NUM}", s)
    if mt:
        return RULE_FRAME, float(mt.group(1)), float(mt.group(2)), float(mt.group(3))
    mt = re.fullmatch(rf"([hc]){_NUM}v{_NUM}", s)
    if mt:
        return RULE_LINE if mt.group(1) == "h" else RULE_CLIP, float(mt.group(2)), float(mt.group(3)), 0.0
    raise ValueError(f"reading rule {pad!r}: auto, kN, kAvB, kAvBfC, hXvY, cXvY, (kh, kv) or N px")


def rule_hv(rule, b, H):
    """(horizontal, vertical) pad in frame px of a reading rule (rule_of's, or its spec) for the box b = [x0, y0, x1,
    y1] in an H px tall frame: the C page's arithmetic (vocr_tsd_page_set_rule), eval/tsdesk_eval.margin_hv's values
    (a clip rule's margins before its clip: rule_sides)."""
    kind, a, bb, c = rule if isinstance(rule, tuple) and len(rule) == 4 else rule_of(rule)
    if kind == RULE_PX:
        return a, bb
    if kind == RULE_FRAME:
        return max(c, float(round(a * H / 544))), max(2.0, float(round(bb * H / 544)))
    w, h = max(1.0, float(b[2]) - float(b[0])), max(1.0, float(b[3]) - float(b[1]))
    if w >= h:
        return max(1.0, a * h), max(1.0, bb * h)
    return max(1.0, bb * w), max(1.0, a * w)


def rule_sides(rule, b, H, others=()):
    """(left, top, right, bottom) margins in frame px of a reading rule for the box b = [x0, y0, x1, y1] in an H px
    tall frame, others: every box of the frame as [x0, y0, x1, y1, ...] (b itself among them; a clip rule's
    neighbours): rule_hv's pads on both sides, a clip rule's cut to half the gap to the nearest other box on each side
    (eval/tsdesk_eval.clip_sides; the C page's clip_sides)."""
    r = rule if isinstance(rule, tuple) and len(rule) == 4 else rule_of(rule)
    ph, pv = rule_hv(r, b, H)
    if r[0] != RULE_CLIP:
        return ph, pv, ph, pv
    x0, y0, x1, y1 = (float(v) for v in b[:4])
    left, right, top, bottom = ph, ph, pv, pv
    for o in others:
        ox0, oy0, ox1, oy1 = (float(v) for v in o[:4])
        if (ox0, oy0, ox1, oy1) == (x0, y0, x1, y1):
            continue
        across = min(y1, oy1) - max(y0, oy0)
        along = min(x1, ox1) - max(x0, ox0)
        if across >= 0.5 * min(y1 - y0, oy1 - oy0) and along <= 0:
            if ox0 >= x1:
                right = min(right, (ox0 - x1) / 2.0)
            elif ox1 <= x0:
                left = min(left, (x0 - ox1) / 2.0)
        elif along > 0 and across <= 0:
            if oy0 >= y1:
                bottom = min(bottom, (oy0 - y1) / 2.0)
            elif oy1 <= y0:
                top = min(top, (y0 - oy1) / 2.0)
    return left, top, right, bottom


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


def post_struct(post):
    """A CPost of tools/tsdesk.POST_DEFAULTS options (a dict), or None for None; gap_pct (the along split) has no C
    twin and is refused."""
    if post in (None, False):
        return None
    o = T.post_options(post)
    no = [k for k in ("gap_pct", "touch_pct", "ext_band_pct", "ksplit_pct") if o[k]]
    if no:
        raise ValueError(f"post {no}: no C twin")
    o = {k: o[k] for k in POST_KEYS}
    return CPost(*[o[k] for k in POST_KEYS])


def drop_struct(drop):
    """A CDrop of tools/tsdesk.DROP_DEFAULTS options (a dict), or None."""
    if drop in (None, False):
        return None
    o = T.drop_options(drop)
    return CDrop(*[o[k] for k in DROP_KEYS])


def charbox_cal(tier="L"):
    """The recognizer's charbox calibration for the C page's cut ([shift, group 0..7], 1/16 resized px): eval/
    charboxes_cal.json's for a tier (S, N, L; tools/export_onnx.charbox_cals, the ONNX files' vocr.charbox_cal), or
    a vocr_onnx recognizer file's own (a path ending in .onnx)."""
    if str(tier).endswith(".onnx"):
        import vocr_onnx as V
        cal = V.Recognizer(str(tier)).cal
        if cal is None:
            raise ValueError(f"{tier}: no vocr.charbox_cal")
        return [int(v) for v in cal]
    import export_onnx
    cals = export_onnx.charbox_cals()
    if tier not in cals:
        raise KeyError(f"no charbox calibration for tier {tier!r}: {sorted(cals)}")
    return [int(v) for v in cals[tier]]


def decode_struct(decode, base, hs, tags):
    """A COpts of the blob's decode options `base` (min_cells, conf_min_logit, ...) updated with `decode` (a dict of
    tools/tsdesk.DEFAULTS / DECODE_EXTRA options), or None for None. The extras without a C twin are refused."""
    if decode in (None, False):
        return None
    bad = set(decode) - set(T.DEFAULTS) - set(T.DECODE_EXTRA)
    if bad:
        raise KeyError(f"unknown decode options {sorted(bad)}")
    no = [k for k in NO_C_TWIN if decode.get(k)] + (["vote_depth"] if int(decode.get("vote_depth", 2)) != 2 else [])
    if no:
        raise ValueError(f"decode {no}: no C twin")
    o = dict(base)
    o.update({k: v for k, v in decode.items() if k in T.DEFAULTS})
    return COpts(hs, tags, int(o["min_cells"]), int(o["max_boxes"]), float(o["conf_min_logit"]),
                 float(o["min_across"]), float(o["along_reach"]), int(decode.get("vote_pct", 0)))


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
        blob's decode constants (min_cells, conf_min_logit, min_across, along_reach, max_boxes) and set vote_pct."""
        o = dict(self.decode_opts)
        o.update(opts)
        hwc = np.ascontiguousarray(np.asarray(maps, np.float32).transpose(1, 2, 0))
        mh, mw, ch = hwc.shape
        tags = ch - 6
        co = COpts(self.hs, tags, int(o["min_cells"]), int(o["max_boxes"]), float(o["conf_min_logit"]),
                   float(o["min_across"]), float(o["along_reach"]), int(o.get("vote_pct", 0)))
        scratch = np.zeros(self.L.vocr_tsd_decode_bytes_ex(mw, mh, tags, C.byref(co)) + 64, np.uint8)
        st = CSet()
        rc = self.L.vocr_tsd_decode_ex(C.byref(co), hwc.ctypes.data, mw, mh, int(iw), int(ih), scratch.ctypes.data,
                                       scratch.size, C.byref(st))
        if rc:
            raise ValueError(f"vocr_tsd_decode: {rc}")
        self.truncated = int(st.truncated)
        return _boxes(st)

    def detect(self, img, size=None, refine=None, decode=None, post=None):
        """vocr_tsd_detect_ex on a grey or RGB(A) frame in memory: [Box] in frame px (tsdesk.Detector.detect's), at
        the input policy `size` (None: the blob's); refine: None (off), True or a dict of tools/tsdesk_refine's options
        (tsdesk.Detector's refine: every horizontal box snapped to its line's ink on the full-resolution luma);
        decode, post: tools/tsdesk's decode options (vote_pct) and post-pass options (dicts; None: off), through
        vocr_tsd_detect_opt (tsdesk.Detector(decode_opts=..., post=...)'s boxes)."""
        if decode or post:
            return self._detect_opt(img, size, refine, decode, post)
        a, w, h, stride, ch = _image(img)
        kind, n = policy_of(size, self.default_policy)
        run, pool, nt = self._run()
        ro = self._refine_opts(refine)
        need = self.L.vocr_tsd_arena_bytes_ex(self.m.ctypes.data, kind, n, w, h, ch, nt, int(ro is not None))
        if not need:
            raise ValueError(f"a {w} x {h} frame under the policy {size or 'default'}: refused (a side of 0 or past "
                             "8192 px, or fit:N / long:N outside 1 <= N <= 8192)")
        arena = self.arena(need)
        st = self._set
        rc = self.L.vocr_tsd_detect_ex(self.m.ctypes.data, a.ctypes.data, w, h, stride, ch, kind, n,
                                       C.cast(ro, C.c_void_p) if ro is not None else None, arena.ctypes.data,
                                       arena.size, C.byref(st), run, pool, nt)
        if rc:
            raise ValueError(f"vocr_tsd_detect: {rc}")
        self.truncated = int(st.truncated)
        return _boxes(st)

    def det_opts(self, refine=None, decode=None, post=None):
        """(CDetOpts, the structs it points to) of the round-2 options (kept alive with it)."""
        co = decode_struct(decode, self.decode_opts, self.hs, self.tags)
        cp = post_struct(post)
        ro = self._refine_opts(refine)
        opt = CDetOpts(C.pointer(co) if co is not None else None, C.cast(ro, C.c_void_p) if ro is not None else None,
                       C.pointer(cp) if cp is not None else None)
        return opt, (co, cp, ro)

    def _detect_opt(self, img, size, refine, decode, post):
        a, w, h, stride, ch = _image(img)
        kind, n = policy_of(size, self.default_policy)
        run, pool, nt = self._run()
        opt, keep = self.det_opts(refine, decode, post)
        need = self.L.vocr_tsd_arena_bytes_opt(self.m.ctypes.data, kind, n, w, h, ch, nt, C.byref(opt))
        if not need:
            raise ValueError(f"a {w} x {h} frame under the policy {size or 'default'}: refused")
        arena = self.arena(need)
        st = self._set
        rc = self.L.vocr_tsd_detect_opt(self.m.ctypes.data, a.ctypes.data, w, h, stride, ch, kind, n, C.byref(opt),
                                        arena.ctypes.data, arena.size, C.byref(st), run, pool, nt)
        del keep
        if rc:
            raise ValueError(f"vocr_tsd_detect_opt: {rc}")
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

    def boxes(self, rgb, size=None, refine=None, decode=None, post=None):
        """eval/desktop.py's detector interface: [[x0, y0, x1, y1, vertical, logit, tag]] in frame px."""
        return [[b.x, b.y, b.x + b.w, b.y + b.h, b.vertical, round(b.logit, 4), b.tag]
                for b in self.detect(rgb, size, refine, decode, post)]

    def post_pre(self, grey, boxes, post):
        """vocr_tsdp_pre on [tsdesk.Box] in frame px of a grey frame: tools/tsdesk.post_pre's boxes (tests)."""
        return self._post(grey, boxes, post, True)

    def post_post(self, grey, boxes, post):
        """vocr_tsdp_post: tools/tsdesk.post_post's boxes (tests)."""
        return self._post(grey, boxes, post, False)

    def _post(self, grey, boxes, post, pre):
        g = np.ascontiguousarray(grey, np.uint8)
        H, W = g.shape
        cap = MAX_BOXES
        arr = (CPBox * cap)()
        if len(boxes) > cap:
            raise ValueError(f"{len(boxes)} boxes: at most {cap}")
        for i, b in enumerate(boxes):
            arr[i] = CPBox(b.x, b.y, b.w, b.h, b.logit, b.tconf, int(b.vertical), int(b.tag), int(b.cells))
        n = C.c_uint(len(boxes))
        cp = post_struct(post)
        scratch = np.zeros(self.L.vocr_tsdp_scratch_bytes(W, H, cap) + 64, np.uint8)
        if pre:
            rc = self.L.vocr_tsdp_pre(g.ctypes.data, W, H, W, arr, C.byref(n), cap, C.byref(cp), scratch.ctypes.data,
                                      scratch.size)
        else:
            rc = self.L.vocr_tsdp_post(g.ctypes.data, W, H, W, arr, C.byref(n), C.byref(cp), scratch.ctypes.data,
                                       scratch.size)
        if rc:
            raise ValueError(f"vocr_tsdp_{'pre' if pre else 'post'}: {rc}")
        return [T.Box(b.x, b.y, b.w, b.h, int(b.vertical), b.logit, int(b.tag), b.tconf, int(b.cells))
                for b in arr[:n.value]]

    def cut_reading(self, grey, b, chars, pct, rel_pct=200):
        """vocr_tsdc_cut on a grey frame, a tight box [x0, y0, x1, y1] and [(text, x)] characters: tools/tsdesk.
        cut_reading's None or [(x0, x1, text)] (tests)."""
        g = np.ascontiguousarray(grey, np.uint8)
        H, W = g.shape
        n = len(chars)
        box = (C.c_double * 4)(*[float(v) for v in b[:4]])
        cx = (C.c_double * max(1, n))(*[float(x) for _, x in chars])
        mk = (C.c_ubyte * max(1, n))(*[self.L.vocr_tsdc_is_mark(t.encode("utf-8"), len(t.encode("utf-8")))
                                       for t, _ in chars])
        x0, x1 = (C.c_double * max(1, n))(), (C.c_double * max(1, n))()
        first, end, np_ = (C.c_uint * max(1, n))(), (C.c_uint * max(1, n))(), C.c_uint()
        nb = self.L.vocr_tsdp_scratch_bytes(W, H, 1)
        scr = np.zeros(nb + 64, np.uint8)
        rc = self.L.vocr_tsdc_cut(g.ctypes.data, W, H, W, box, cx, mk, n, int(pct), int(rel_pct), scr.ctypes.data,
                                  nb + 64, x0, x1, first, end, C.byref(np_))
        if rc < 0:
            raise ValueError(f"vocr_tsdc_cut: {rc}")
        if rc == 0:
            return None
        return [(x0[k], x1[k], "".join(t for t, _ in chars[first[k]:end[k]])) for k in range(np_.value)]

    def drop_set(self, items, drop):
        """vocr_tsdd_drop on [(box (x, y, w, h), text, conf_mean)]: tools/tsdesk.drop_set's indices (tests)."""
        n = len(items)
        xy = (C.c_double * max(1, 4 * n))(*[float(v) for b, _, _ in items for v in b])
        txt = (C.c_char_p * max(1, n))(*[t.encode("utf-8") for _, t, _ in items])
        cf = (C.c_float * max(1, n))(*[float(c) for _, _, c in items])
        out = (C.c_ubyte * max(1, n))()
        cd = drop_struct(drop)
        rc = self.L.vocr_tsdd_drop(xy, txt, cf, n, C.byref(cd), out)
        if rc < 0:
            raise ValueError(f"vocr_tsdd_drop: {rc}")
        return {i for i in range(n) if out[i]}

    def input_size(self, w, h, size=None):
        """(iw, ih, pw, ph): the network input's image part and its padded size for a w x h frame."""
        kind, n = policy_of(size, self.default_policy)
        v = [C.c_uint() for _ in range(4)]
        self.L.vocr_tsd_input_size(kind, n, w, h, *[C.byref(x) for x in v])
        return tuple(int(x.value) for x in v)

    def resize(self, grey, dw, dh):
        """vocr_tsd_resize (the C detector's integer resize: vocr-desktop's vocr_desk_resize rule): a grey image to dw x
        dh."""
        g = np.ascontiguousarray(grey, np.uint8)
        h, w = g.shape
        out = np.zeros((dh, dw), np.uint8)
        scratch = np.zeros(self.L.vocr_tsd_resize_scratch(w, h, dw, dh) + 64, np.uint8)
        self.L.vocr_tsd_resize(g.ctypes.data, w, h, w, out.ctypes.data, dw, dh, scratch.ctypes.data)
        return out

    def preprocess(self, img, size=None):
        """(network input [1, 1, ph, pw] uint8, (fx, fy, iw, ih)) as the C detector builds it: the integer luma, the
        policy's integer resize (vocr_tsd_resize), edge padding to multiples of 32 (tsdesk.Detector.preprocess with
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
    sections), every horizontal box (refined when refine is set, as CModel.detect refines) grown by the reading rule
    `pad` (rule_of: auto, kN, kAvB, kAvBfC, hXvY, cXvY or N px) and read on `threads` threads in C. The defaults (pad
    "auto", no refinement) are the page's plain setting; round 1's measured one is pad="k6v4", refine=True
    (docs/textspot-desktop.md)."""

    def __init__(self, det, rec, threads=1, size=None, pad="auto", tags=None, refine=None, decode=None, post=None,
                 drop=None, lib=None, charbox_cal=None):
        self.L = c_library(lib)
        self.det = aligned(_blob_bytes(det))
        self.rec = aligned(_blob_bytes(rec))
        kind, n = policy_of(size, (-1, 0))
        self.rule = rule_of(pad)
        err = C.c_int(0)
        self.p = self.L.vocr_tsd_page_new(self.det.ctypes.data, len(self.det), self.rec.ctypes.data, len(self.rec),
                                          int(threads), kind, n, 4, 4, 0, C.byref(err))
        if not self.p:
            raise ValueError(f"vocr_tsd_page_new: {err.value}")
        rc = self.L.vocr_tsd_page_set_rule(self.p, *self.rule)
        if rc:
            raise ValueError(f"vocr_tsd_page_set_rule{self.rule}: {rc}")
        if tags is not None:
            self.L.vocr_tsd_page_set_tags(self.p, sum(1 << int(t) for t in tags))
        self._ropts = refine_opts(refine, self.L)       # the page copies the options
        if self._ropts is not None:
            self.L.vocr_tsd_page_set_refine(self.p, C.cast(self._ropts, C.c_void_p))
        if decode:      # the decode's options on the blob's (read from its header, as CModel does)
            hdr = bytes(self.det[:HEADER])
            hs = struct.unpack_from("<I", hdr, 20)[0]
            tags_n = struct.unpack_from("<II", hdr, 68)[0]
            mc, mb = struct.unpack_from("<2I", hdr, 80)
            conf, across, reach = struct.unpack_from("<3d", hdr, 128)
            base = {"min_cells": mc, "conf_min_logit": conf, "min_across": across, "along_reach": reach,
                    "max_boxes": mb}
            self._dopts = decode_struct(decode, base, hs, tags_n)
            self.L.vocr_tsd_page_set_decode(self.p, C.byref(self._dopts))
        if post:
            self._popts = post_struct(post)
            self.L.vocr_tsd_page_set_post(self.p, C.byref(self._popts))
        if charbox_cal is not None:      # the cut's character positions (charbox_cal(); also keeps them: page())
            cal = (C.c_int * 9)(*[int(v) for v in charbox_cal])
            rc = self.L.vocr_tsd_page_set_charbox_cal(self.p, cal)
            if rc:
                raise ValueError(f"vocr_tsd_page_set_charbox_cal: {rc}")
        elif drop and T.drop_options(drop)["cut_pct"]:
            raise ValueError("the cut (cut_pct) needs the recognizer's charbox calibration: charbox_cal=")
        if drop:
            self._xopts = drop_struct(drop)
            self.L.vocr_tsd_page_set_drop(self.p, C.byref(self._xopts))

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

    def page(self, img, chars=False):
        """Every line of a frame in reading order: [tsdesk.Line] (text, tight box, crop read, tag, logit, dark,
        vertical); vertical boxes come back without text. chars (with charbox_cal set): also each line's kept
        characters [(text, x frame px)] (tools/tsdesk.cut_read_lines' input): (lines, chars)."""
        a, w, h, stride, ch = _image(img)
        rc = self.L.vocr_tsd_page_run(self.p, a.ctypes.data, w, h, stride, ch)
        if rc:
            raise ValueError(f"vocr_tsd_page_run: {rc}")
        out, cs = [], []
        c16, off = (C.c_int32 * 160)(), (C.c_uint * 160)()
        for i in range(self.L.vocr_tsd_page_count(self.p)):
            ln = self.L.vocr_tsd_page_line(self.p, i).contents
            out.append(T.Line(ln.text.decode("utf-8", "replace"), (ln.bx, ln.by, ln.bw, ln.bh),
                              (ln.x, ln.y, ln.w, ln.h), int(ln.tag), ln.logit, int(ln.dark), int(ln.vertical)))
            if chars:
                n = self.L.vocr_tsd_page_line_chars(self.p, i, c16, off, 160)
                raw = bytes(ln.text)
                ends = [off[k] for k in range(1, n)] + [len(raw.split(b"\0", 1)[0])]
                cs.append([(raw[off[k]:ends[k]].decode("utf-8", "replace"), c16[k] / 16.0) for k in range(n)])
        return (out, cs) if chars else out

    def pad(self):
        """The reading pad of the last run, frame px (horizontal, vertical); (-1, -1) under a line rule (hXvY: each
        line's own pad, its crop)."""
        a, b = C.c_int(), C.c_int()
        self.L.vocr_tsd_page_pad(self.p, C.byref(a), C.byref(b))
        return a.value, b.value

    def times_us(self):
        """(the detector, the whole page) of the last run, us."""
        return self.L.vocr_tsd_page_detect_us(self.p), self.L.vocr_tsd_page_total_us(self.p)
