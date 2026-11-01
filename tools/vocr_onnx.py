#!/usr/bin/env python3
"""VOCR through ONNX Runtime: the reference pipeline for desktop apps (docs/onnx.md, issue #163).

One file, NumPy and onnxruntime only (Pillow for the command line's image files), so an app can copy it as it is:

    from vocr_onnx import Spotter, Recognizer, CharLM, read_region
    spot = Spotter("textspot8_exact.onnx")
    rec = Recognizer("VOCR6.2_N_exact.onnx")
    lm = CharLM("vocr_lm6_o3_1000k.vlm")                       # optional: beam 8 + the character LM, Rubi's settings
    lines = read_region(frame, spot, rec, region=(x, y, w, h), lm=lm)   # frame: H x W grey or H x W x 3/4 RGB(A)
    for ln in lines:
        ln.text, ln.box, ln.chars                               # chars: [Char(text, frame, x0, x1, ...)] in frame px

    python3 vocr_onnx.py --models DIR [--tier S|N|L] [--mode exact|fast] [--region X,Y,W,H] [--lm FILE] IMAGE...

Everything outside the networks is a port of the C runtime's own code, integer for integer, so with the exact models
the lines, texts and character positions are the C runtime's (eval/onnx_parity.py checks it on development data):

- preprocess(): runtime/vocr.c's line preprocessing: integer luma (77 R + 150 G + 29 B) >> 8, 255 - grey for dark
  text, OpenCV INTER_LINEAR's fixed-point resize to the model's height (width round(w * 32 / h), half to even, clamped
  to [8, 1024]), right pad to a multiple of 8 by repeating the last column;
- the recognizer's outputs: greedy CTC (blank 0; a frame's class is emitted when it is not the blank and differs from
  the frame before), each character's run of frames, and the confidence runtime/vocr.c reports (conf_min, conf_mean);
- CharLM.decode(): runtime/vocr_lm.c's CTC prefix beam search with the character n-gram LM (lm/vlm.py's model);
- Recognizer char positions: runtime/vocr_charbox.c's cells (eval/charbox.py: calibrated frame centres, cut halfway
  between neighbours), in crop pixels, then in frame pixels;
- Spotter: runtime/vocr_textspot6.c's decode of a VTSP version 4 head (synth/textspot6_twin.py): kernel components,
  boxes, tags, the 64-box cap, the dark flag (vocr_line_dark), blocks, reading order, the dialogue pick.

Rubi's own line path adds what this file leaves out: the counter re-decode (runtime/vocr_post.c), the cascade to the
next tier for a doubtful line, its text normalisation and its filter for lines with no Japanese.
"""
import json
import struct
import zlib
from dataclasses import dataclass, field

import numpy as np

F32 = np.float32
FORMAT = 1


def _session(path, threads=1, providers=None, options=None):
    import onnxruntime as ort
    so = options or ort.SessionOptions()
    if options is None:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=providers or ["CPUExecutionProvider"])


def _meta(sess):
    raw = sess.get_modelmeta().custom_metadata_map
    out = {}
    for k, v in raw.items():
        try:
            out[k] = json.loads(v)
        except ValueError:
            out[k] = v
    if out.get("vocr.format") != FORMAT:
        raise ValueError(f"not a vocr ONNX file of format {FORMAT} (vocr.format {out.get('vocr.format')!r})")
    return out


# ----------------------------------------------------------------------------- images

def grey_of(img):
    """uint8 grey of an H x W grey or H x W x 3/4 RGB(A) image: (77 R + 150 G + 29 B) >> 8, as the C side."""
    a = np.asarray(img)
    if a.dtype != np.uint8:
        raise TypeError(f"images are uint8, not {a.dtype}")
    if a.ndim == 2:
        return a
    if a.ndim == 3 and a.shape[2] in (3, 4):
        r, g, b = (a[..., i].astype(np.uint16) for i in range(3))
        return ((77 * r + 150 * g + 29 * b) >> 8).astype(np.uint8)
    raise ValueError("an H x W grey or H x W x 3 / 4 RGB(A) image")


def resized_width(w, h, height=32, max_width=1024, min_width=8):
    """vocr_resized_width(): round(w * height / h), half to even, clamped to [min_width, max_width]."""
    num = w * height
    q = (2 * num + h) // (2 * h)
    if (2 * num + h) % (2 * h) == 0 and q & 1:
        q -= 1
    return max(min_width, min(max_width, q))


def _coef(n_out, n_in):
    """runtime/vocr.c row_table / col_coef for an n_in -> n_out resize: per output index (i0, i1, a0, a1), the two
    source indices and their 11-bit weights, from the same double / float arithmetic as OpenCV's INTER_LINEAR."""
    scale = 1.0 / (float(n_out) / float(n_in))
    t = (np.arange(n_out, dtype=np.float64) + 0.5) * scale
    f = (t - 0.5).astype(F32)
    s = f.astype(np.int64)                                     # (int) truncates; floor for negatives below
    s = np.where(s.astype(F32) > f, s - 1, s)
    f = (f - s.astype(F32)).astype(F32)
    return s, f


def _round_even(v):
    return np.rint(v).astype(np.int64)                         # v >= 0, float32: cvRound's half to even


def preprocess(img, invert=False, height=32, stride=8, max_width=1024):
    """(the network's input, H x padded width uint8; the resized width nw) for a line crop, as vocr_run() sees it."""
    g = grey_of(img)
    h, w = g.shape
    if not h or not w or h > 65535 or w > 65535:
        raise ValueError("an empty or oversized crop")
    if invert:
        g = 255 - g
    nw = resized_width(w, h, height, max_width)
    padded = (nw + stride - 1) // stride * stride
    sy, fy = _coef(height, h)
    r0 = np.clip(sy, 0, h - 1)
    r1 = np.clip(sy + 1, 0, h - 1)
    b0 = _round_even(((F32(1) - fy) * F32(2048)).astype(F32))
    b1 = _round_even((fy * F32(2048)).astype(F32))
    sx, fx = _coef(nw, w)
    low = sx < 0
    fx = np.where(low, F32(0), fx).astype(F32)
    sx = np.where(low, 0, sx)
    high = sx >= w - 1
    fx = np.where(high, F32(0), fx).astype(F32)
    sx = np.where(high, w - 1, sx)
    a0 = _round_even(((F32(1) - fx) * F32(2048)).astype(F32))
    a1 = _round_even((fx * F32(2048)).astype(F32))
    s1 = np.where(sx + 1 < w, sx + 1, sx)
    cols = np.minimum(np.arange(padded), nw - 1)               # the pad repeats the last resized column
    sx, s1, a0, a1 = sx[cols], s1[cols], a0[cols], a1[cols]
    gi = g.astype(np.int64)
    p0, p1 = gi[r0], gi[r1]                                    # [height, w]
    h0 = p0[:, sx] * a0[None, :] + p0[:, s1] * a1[None, :]
    h1 = p1[:, sx] * a0[None, :] + p1[:, s1] * a1[None, :]
    out = (((b0[:, None] * (h0 >> 4)) >> 16) + ((b1[:, None] * (h1 >> 4)) >> 16) + 2) >> 2
    return out.astype(np.uint8), nw


# ----------------------------------------------------------------------------- recognizer

_LOG2E = np.frombuffer(np.uint32(0x3FB8AA3B).tobytes(), F32)[0]
_POLY = [np.frombuffer(np.uint32(u).tobytes(), F32)[0]
         for u in (0x39218489, 0x3AAEC3FF, 0x3C1D955B, 0x3D635847, 0x3E75FDF0, 0x3F317218, 0x3F800000)]


def exp_neg(x):
    """runtime/vocr.c exp_neg in float32 (x clamped to [-30, 0])."""
    x = np.asarray(x, F32)
    x = np.where(x > F32(-30), x, F32(-30)).astype(F32)
    x = np.where(x > 0, F32(0), x).astype(F32)
    t = (x * _LOG2E).astype(F32)
    e = t.astype(np.int32)
    e = np.where(e.astype(F32) > t, e - 1, e).astype(np.int32)
    f = (t - e.astype(F32)).astype(F32)
    p = np.full(x.shape, _POLY[0], F32)
    for c in _POLY[1:]:
        p = ((p * f).astype(F32) + c).astype(F32)
    p2 = ((e + 127).astype(np.uint32) << np.uint32(23)).view(F32)
    return (p * p2).astype(F32)


def topk_conf(topv):
    """vocr.c topk_conf per frame: 1 / (1 + sum_j exp_neg(v_j - v_0)), float32, in order."""
    v = np.asarray(topv, F32)
    s = np.ones(len(v), F32)
    for j in range(1, v.shape[1]):
        s = (s + exp_neg((v[:, j] - v[:, 0]).astype(F32))).astype(F32)
    return (F32(1) / s).astype(F32)


def spans(best):
    """[(class, f0, f1)]: the greedy characters, each with its run of frames."""
    out, last = [], 0
    for t, c in enumerate(np.asarray(best, np.int64).tolist()):
        if c and c != last:
            out.append([c, t, t])
        elif c and c == last:
            out[-1][2] = t
        last = c
    return [tuple(s) for s in out]


# eval/charbox.py's calibration groups
_CLOSE = "、。，．」』）】〉》〕｝］〙〗”’"
_OPEN = "「『（【〈《〔｛［〘〖“‘"
_DOTS = "…‥・"
_SMALL = "ぁぃぅぇぉっゃゅょゎゕゖァィゥェォッャュョヮヵヶㇰㇱㇲㇳㇴㇵㇶㇷㇸㇹㇺㇻㇼㇽㇾㇿ"
_LONG = "ー―─～〜－"
_MARK = "！？♪☆★♥♡※"
Q = 16                                                         # charbox fixed point: 1/16 px


def char_group(ch):
    """eval/charbox.py group(): 0 other, 1 close, 2 open, 3 dots, 4 small, 5 long, 6 mark, 7 half."""
    if len(ch) != 1:
        return 0
    for g, s in ((1, _CLOSE), (2, _OPEN), (3, _DOTS), (4, _SMALL), (5, _LONG), (6, _MARK)):
        if ch in s:
            return g
    o = ord(ch)
    return 7 if o < 0x80 or 0xFF61 <= o <= 0xFF9F else 0


def _div_round(a, b):
    return (a + b // 2) // b if a >= 0 else -((-a + b // 2) // b)


def cells(sp, w, h, nw, cal, groups, stride=8):
    """runtime/vocr_charbox.c's boxes without INK or PITCH (eval/charbox.py positions(flags=0)): per greedy
    character (centre, x0, x1) in 1/16 crop px, along the line."""
    n = len(sp)
    if not n:
        return []
    cal = list(cal) if cal is not None else [0] * 9
    c = []
    for cls, f0, _ in sp:
        r = Q * (stride * f0 + stride // 2) + cal[0] + cal[1 + groups[cls]]
        c.append(min(max(_div_round(r * w, nw), 0), w * Q))
    d = sorted(c[i + 1] - c[i] for i in range(n - 1))
    pitch = d[(len(d) - 1) // 2] if d else h * Q
    if pitch <= 0:
        pitch = h * Q
    cuts = [_div_round(c[i] + c[i + 1], 2) for i in range(n - 1)]
    lo = [max(0, c[0] - pitch // 2)] + cuts
    hi = cuts + [min(w * Q, c[-1] + pitch // 2)]
    return [(c[i], lo[i], max(lo[i], hi[i])) for i in range(n)]


@dataclass
class Char:
    text: str
    cls: int
    f0: int                     # its run of CTC frames, f0 .. f1 (-1, -1: a character the LM inserted)
    f1: int
    x: float                    # the centre and the cell [x0, x1) along the line, in the frame's (or crop's) px
    x0: float
    x1: float
    y0: int = 0                 # the line's rows
    y1: int = 0


SYNC = 8                        # vocr_charbox.h VOCR_CHARBOX_SYNC


def align_map(src, dst, sync=SYNC):
    """eval/charbox.py align_map (runtime/vocr_charbox.c vocr_charbox_align): dst[j] -> the index of the src character
    it is aligned to, or -1 (inserted). Equal characters align; at a difference the shortest skip (di on src, dj on
    dst, each at most sync) after which two characters agree (or a text ends right after) is taken, the most balanced
    first; its first min(di, dj) characters are substitutions. With no such skip the two are a substitution."""
    src = [ord(c) if isinstance(c, str) else int(c) for c in src]
    dst = [ord(c) if isinstance(c, str) else int(c) for c in dst]
    n, m = len(src), len(dst)
    out = [-1] * m
    i = j = 0
    while i < n and j < m:
        if src[i] == dst[j]:
            out[j] = i
            i += 1
            j += 1
            continue
        found = None
        for s in range(1, 2 * sync + 1):
            best = None
            for di in range(max(0, s - sync), min(s, sync) + 1):
                dj = s - di
                if i + di >= n or j + dj >= m or src[i + di] != dst[j + dj]:
                    continue
                if not (i + di + 1 >= n or j + dj + 1 >= m or src[i + di + 1] == dst[j + dj + 1]):
                    continue
                key = abs(di - dj) * 64 + dj
                if best is None or key < best[0]:
                    best = (key, di, dj)
            if best is not None:
                found = best[1:]
                break
        if found is None:
            out[j] = i
            i += 1
            j += 1
            continue
        di, dj = found
        for t in range(min(di, dj)):
            out[j + t] = i + t
        i += di
        j += dj
    return out


def place(text, chars):
    """Cells for the characters of `text` (e.g. the beam + LM text) from the greedy characters `chars`, through
    align_map: an aligned character takes its greedy character's cell; an inserted one (f0 = f1 = cls = -1) shares its
    neighbour's cell, the one before it (after it at the line's start), as docs/char-boxes.md says. Always new Char
    objects (the greedy ones are not shared)."""
    if "".join(c.text for c in chars) == text:
        return [Char(c.text, c.cls, c.f0, c.f1, c.x, c.x0, c.x1, c.y0, c.y1) for c in chars]
    m = align_map([c.text for c in chars], list(text))
    out = []
    for j, (ch, i) in enumerate(zip(text, m)):
        if i >= 0:
            c = chars[i]
            out.append(Char(ch, c.cls, c.f0, c.f1, c.x, c.x0, c.x1, c.y0, c.y1))
            continue
        before = [k for k in m[:j] if k >= 0]
        after = [k for k in m[j + 1:] if k >= 0]
        nb = chars[before[-1]] if before else chars[after[0]] if after else None
        if nb is None:
            out.append(Char(ch, -1, -1, -1, 0.0, 0.0, 0.0))
        else:
            out.append(Char(ch, -1, -1, -1, nb.x, nb.x0, nb.x1, nb.y0, nb.y1))
    return out


@dataclass
class Line:
    text: str                   # the greedy text, or the beam + LM text with lm=
    greedy: str
    box: tuple = (0, 0, 0, 0)   # x, y, w, h in the frame (or the crop for Recognizer.read)
    chars: list = field(default_factory=list)   # the greedy characters with their cells
    text_chars: list = field(default_factory=list)   # the characters of `text` with cells (place(): LM insertions
                                                     # share a neighbour's cell)
    conf_min: float = 1.0
    conf_mean: float = 1.0
    frames: int = 0
    width: int = 0              # the resized width before padding
    dark: int = 0
    tag: int = 0                # the spotter's class: 0 dialogue, 1 name, 2 choice, 3 UI, 4 scene
    conf: int = 0
    block: int = -1
    outputs: dict = field(default_factory=dict, repr=False)


class Recognizer:
    """A recognizer exported by tools/export_onnx.py (its "exact" or "fast" file)."""

    def __init__(self, path, threads=1, providers=None, session_options=None):
        self.sess = _session(path, threads, providers, session_options)
        self.meta = m = _meta(self.sess)
        if m["vocr.kind"] != "recognizer":
            raise ValueError(f"{path}: not a recognizer")
        self.vocab = m["vocr.vocab"]
        self.height, self.stride, self.max_width = m["vocr.height"], m["vocr.stride"], m["vocr.max_width"]
        self.mode, self.name = m["vocr.mode"], m.get("vocr.name", "")
        self.cal = m.get("vocr.charbox_cal")
        self.groups = [char_group(t) for t in self.vocab]
        self.has_sections = "head_sections" in [i.name for i in self.sess.get_inputs()]
        self.outputs = [o.name for o in self.sess.get_outputs()]

    def preprocess(self, img, invert=False):
        return preprocess(img, invert, self.height, self.stride, self.max_width)

    def forward(self, grey, sections=True, names=None):
        """The network on a preprocessed line (H x W uint8, W a multiple of the stride): {output: array}. sections:
        the blob's head gate / two-level head as Rubi runs them (ignored by a model without them)."""
        feeds = {"grey": np.ascontiguousarray(grey, np.uint8)[None, None]}
        if self.has_sections:
            feeds["head_sections"] = np.array(bool(sections))
        names = list(names or ("best", "topk_logits", "topk_classes", "blank_logit"))
        return dict(zip(names, self.sess.run(names, feeds)))

    def read(self, img, invert=False, sections=True, lm=None, outputs=False):
        """One line crop (H x W grey or RGB(A) uint8) -> Line, in the crop's px. invert: dark text (the spotter's
        dark flag). lm: a CharLM for the beam + LM text."""
        g = grey_of(img)
        x, nw = self.preprocess(g, invert)
        out = self.forward(x, sections)
        best = out["best"]
        sp = spans(best)
        greedy = "".join(self.vocab[c] for c, _, _ in sp)
        cmin = cmean = 1.0
        if sp:
            conf = topk_conf(out["topk_logits"])
            p = conf[[f0 for _, f0, _ in sp]]
            cmin = float(p.min())
            s = F32(0)
            for v in p:
                s = F32(s + v)
            cmean = float(F32(s / F32(len(p))))
        h, w = g.shape
        chars = [Char(self.vocab[cls], cls, f0, f1, c / Q, x0 / Q, x1 / Q, 0, h)
                 for (cls, f0, f1), (c, x0, x1) in zip(sp, cells(sp, w, h, nw, self.cal, self.groups, self.stride))]
        text = greedy
        if lm is not None and greedy:
            text = lm.decode(self, out["topk_logits"], out["topk_classes"], out["blank_logit"])
        return Line(text, greedy, (0, 0, w, h), chars, place(text, chars), cmin, cmean, len(best), nw,
                    int(bool(invert)), outputs=out if outputs else {})


# ----------------------------------------------------------------------------- the character LM (VOLM)

UNK, BOS, EOS, FIRST = 0, 1, 2, 3
NONE = -1
NEG = -(1 << 30)
LP_FLOOR = -65536
MARGIN_CAP = 64.0
MAX_FRAMES = 1024
FLAG_BOS, FLAG_EOS = 1, 2
QL = 1024
BRACKETS = ("「」", "『』", "（）", "【】", "〈〉", "《》", "〔〕", "［］")
# Rubi's decoder settings per LM file (Rubi_for_vita core/vocr_ocr.c lms[] and lm_retuned[]; LM-settings.json of
# v0.3.7-lm6.2): beam 8, top-8, gate 8 nats, bracket 8 nats; alpha, beta in 1/1000 nat
RUBI_LM = {"vocr_lm_o3_500k.vlm": dict(alpha=500, beta=3000, flags=FLAG_EOS),
           "vocr_lm6_o3_1000k.vlm": dict(alpha=600, beta=2500, flags=FLAG_BOS),
           "vocr_lm_o3_2000k.vlm": dict(alpha=500, beta=3000, flags=FLAG_EOS),
           "vocr_lm6_o3_4000k.vlm": dict(alpha=600, beta=2500, flags=FLAG_BOS)}
RUBI_LM_RETUNED = {("N", "vocr_lm_o3_2000k.vlm"): dict(alpha=400, beta=1500, flags=FLAG_BOS | FLAG_EOS)}


def _tdiv(a, b):
    q = abs(a) // abs(b)
    return q if (a >= 0) == (b > 0) else -q


def _milli_q10(x):
    a = abs(int(x))
    q = (a * QL + 500) // 1000
    return q if x >= 0 else -q


def _margin_q10(top, v):
    d = F32(F32(top) - F32(v))
    if not d >= F32(0):
        d = F32(0)
    elif d > F32(MARGIN_CAP):
        d = F32(MARGIN_CAP)
    return -max(int(F32(d * F32(QL)) + F32(0.5)), 1)


def _bracket_role(text):
    if text == "(":
        return 3
    if text == ")":
        return 0x80 | 3
    for k, (o, c) in enumerate(BRACKETS, 1):
        if text == o:
            return k
        if text == c:
            return 0x80 | k
    return 0


def _bracket_next(st, r):
    s0, s1 = st & 0xFF, st >> 8
    if not r:
        return st, False
    if not r & 0x80:
        return r | s0 << 8, False
    r &= 0x7F
    if s0 == r:
        return s1, False
    if not s0:
        return st, False
    return (0 if s1 == r else st), True


class CharLM:
    """A VOLM character n-gram LM (runtime/vocr_lm.h; lm/vlm.py) and runtime/vocr_lm.c's beam decoder."""

    def __init__(self, path_or_bytes, beam=8, topk=8, gate=8000, bracket=8000, alpha=None, beta=None, flags=None,
                 tier=None):
        b = bytes(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else open(path_or_bytes,
                                                                                          "rb").read()
        name = "" if isinstance(path_or_bytes, (bytes, bytearray)) else str(path_or_bytes).replace("\\", "/") \
            .rsplit("/", 1)[-1]
        if len(b) < 256 or b[:4] != b"VOLM":
            raise ValueError("not a VOLM blob")
        version, total, order, V, _flags, crc, voc_off = struct.unpack_from("<7I", b, 4)
        if version != 1 or total != len(b) or not 2 <= order <= 6 or V < FIRST:
            raise ValueError("unsupported VOLM header")
        if crc != zlib.crc32(b[64:total]) & 0xFFFFFFFF:
            raise ValueError("VOLM CRC mismatch")
        self.order, self.V = order, V
        self.codepoints = np.frombuffer(b, "<u4", V - FIRST, voc_off).astype(np.int64)
        self.tok, self.prob, self.bo, self.child, self.cbp, self.cbb = [], [], [], [], [], []
        for k in range(1, order + 1):
            n, ot, op, ob, oc, ocp, ocb, _ = struct.unpack_from("<8I", b, 64 + (k - 1) * 32)
            self.tok.append(np.frombuffer(b, "<u2", n, ot).astype(np.int64).tolist() if k >= 2 else None)
            self.prob.append(np.frombuffer(b, np.uint8, n, op).astype(np.int64).tolist())
            self.bo.append(np.frombuffer(b, np.uint8, n, ob).astype(np.int64).tolist() if k < order else None)
            self.child.append(np.frombuffer(b, "<u4", n + 1, oc).astype(np.int64).tolist() if k < order else None)
            self.cbp.append(np.frombuffer(b, "<i2", 256, ocp).astype(np.int64).tolist())
            self.cbb.append(np.frombuffer(b, "<i2", 256, ocb).astype(np.int64).tolist() if k < order else None)
        s = dict(RUBI_LM.get(name, {}))
        s.update(RUBI_LM_RETUNED.get((tier, name), {}))
        if None in (alpha, beta, flags) and not s:
            raise ValueError(f"{name}: not one of Rubi's LMs, so give alpha, beta and flags")
        self.alpha = s["alpha"] if alpha is None else alpha
        self.beta = s["beta"] if beta is None else beta
        self.flags = s["flags"] if flags is None else flags
        self.beam, self.topk, self.gate, self.bracket = beam, topk, gate, bracket
        self._maps = {}

    def token_of(self, text):
        if len(text) != 1:
            return UNK
        cp = ord(text)
        i = int(np.searchsorted(self.codepoints, cp))
        return FIRST + i if i < len(self.codepoints) and self.codepoints[i] == cp else UNK

    def _find(self, k, parent, t):
        lo, hi = self.child[k - 1][parent], self.child[k - 1][parent + 1]
        toks = self.tok[k]
        while lo < hi:
            mid = (lo + hi) >> 1
            v = toks[mid]
            if v < t:
                lo = mid + 1
            elif v > t:
                hi = mid
            else:
                return mid
        return NONE

    def initial(self):
        s = [NONE] * (self.order - 1)
        if self.flags & FLAG_BOS:
            s[0] = BOS
        return tuple(s)

    def score(self, state, t):
        """(log P(t | state) in Q10, the next state): vocr_lm.c lm_score."""
        N = self.order
        e = [NONE] * N
        for k in range(1, N):
            p = state[k - 1]
            e[k] = self._find(k, p, t) if p != NONE else NONE
        lp = 0
        for k in range(N - 1, 0, -1):
            if e[k] != NONE:
                lp += self.cbp[k][self.prob[k][e[k]]]
                break
            if state[k - 1] != NONE:
                lp += self.cbb[k - 1][self.bo[k - 1][state[k - 1]]]
        else:
            lp += self.cbp[0][self.prob[0][t]]
        return lp, (t, *e[1:N - 1])

    def _class_maps(self, vocab):
        key = id(vocab)
        if key not in self._maps:
            def fold(t):
                r = _bracket_role(t)
                return t if not r else "」" if r & 0x80 else "「"
            cmap = [UNK] + [self.token_of(fold(t)) for t in vocab[1:]]
            roles = [_bracket_role(t) for t in vocab]
            self._maps[key] = (cmap, roles)
        return self._maps[key]

    def decode(self, rec, topv, topc, blank=None):
        """The beam + LM text of one line from its per-frame top-K and blank logits (rec: its Recognizer)."""
        return "".join(rec.vocab[c] for c in self.decode_classes(rec.vocab, topv, topc, blank))

    def decode_classes(self, vocab, topv, topc, blank=None):
        """vocr_lm_decode (lm/vlm.py decode): the classes of the best prefix."""
        cmap, roles = self._class_maps(vocab)
        topv = np.asarray(topv, F32)
        topc = np.asarray(topc, np.int64)
        T = len(topv)
        if T > MAX_FRAMES:
            raise ValueError(f"{T} frames: the decoder takes at most {MAX_FRAMES}")
        k = min(self.topk, topc.shape[1]) if T else 0
        a, b, g, br = (_milli_q10(v) for v in (self.alpha, self.beta, self.gate, self.bracket))
        use_lm = a != 0
        par, cls, ln, hs, bk = [NONE], [0], [0], [0], [0]
        st = [self.initial()]
        hyps = [(0, 0, NEG)]
        tv, tc = topv.tolist(), topc.tolist()
        bl = None if blank is None else np.asarray(blank, F32).tolist()
        for t in range(T):
            cand = [(tc[t][0], 0)]
            for j in range(1, k):
                s = _margin_q10(tv[t][0], tv[t][j])
                if -s >= g:
                    break
                cand.append((tc[t][j], s))
            if bl is not None and all(c != 0 for c, _ in cand):
                s = _margin_q10(tv[t][0], bl[t])
                if -s < g:
                    cand.append((0, s))
            stay = [[NEG, NEG] for _ in hyps]
            new = []
            for i, (n, sb, snb) in enumerate(hyps):
                s = max(sb, snb)
                last = cls[n] if n else 0
                for c, q in cand:
                    if c == 0:
                        stay[i][0] = max(stay[i][0], s + q)
                        continue
                    if c == last:
                        if snb > NEG:
                            stay[i][1] = max(stay[i][1], snb + q)
                        src = sb
                    else:
                        src = s
                    if src <= NEG:
                        continue
                    h = (hs[n] * 0x9E3779B1 + c + 1) & 0xFFFFFFFF
                    j = next((j for j, (m, _, _) in enumerate(hyps) if m and cls[m] == c and ln[m] == ln[n] + 1
                              and hs[m] == h and _same_prefix(par, cls, par[m], n)), None)
                    if use_lm:
                        lp, nst = self.score(st[n], cmap[c])
                        lmq = _tdiv(a * min(max(lp, LP_FLOOR), 0), QL)
                    else:
                        lmq, nst = 0, st[n]
                    val = src + q + lmq + b
                    bst = 0
                    if br:
                        bst, mis = _bracket_next(bk[n], roles[c])
                        if mis:
                            val -= br
                    if j is not None:
                        stay[j][1] = max(stay[j][1], val)
                    else:
                        new.append((n, c, val, nst, bst))
            ranked = [(max(x), i) for i, x in enumerate(stay) if max(x) > NEG]
            ranked += [(e[2], len(hyps) + j) for j, e in enumerate(new) if e[2] > NEG]
            ranked.sort(key=lambda r: (-r[0], r[1]))
            nxt = []
            for _, idx in ranked[:self.beam]:
                if idx < len(hyps):
                    nxt.append((hyps[idx][0], stay[idx][0], stay[idx][1]))
                else:
                    n, c, val, nst, bst = new[idx - len(hyps)]
                    par.append(n)
                    cls.append(c)
                    ln.append(ln[n] + 1)
                    hs.append((hs[n] * 0x9E3779B1 + c + 1) & 0xFFFFFFFF)
                    bk.append(bst)
                    st.append(nst)
                    nxt.append((len(par) - 1, NEG, val))
            hyps = nxt
        best, best_s = 0, None
        for n, sb, snb in hyps:
            s = max(sb, snb)
            if use_lm and self.flags & FLAG_EOS:
                lp, _ = self.score(st[n], EOS)
                s += _tdiv(a * min(max(lp, LP_FLOOR), 0), QL)
            if best_s is None or s > best_s:
                best, best_s = n, s
        out = []
        n = best
        while n:
            out.append(cls[n])
            n = par[n]
        return out[::-1]


def _same_prefix(par, cls, x, y):
    while x != y:
        if cls[x] != cls[y]:
            return False
        x, y = par[x], par[y]
    return True


# ----------------------------------------------------------------------------- spotter

@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int
    vertical: int
    conf: int
    dark: int
    block: int
    tag: int
    tconf: int


@dataclass(frozen=True)
class Block:
    x: int
    y: int
    w: int
    h: int
    vertical: int
    first: int
    count: int
    tag: int


@dataclass(frozen=True)
class Spots:
    boxes: tuple
    blocks: tuple
    pick: int                  # the dialogue block (tag 0) Rubi would pick on a whole frame, -1 when none
    truncated: int
    presence: int


BOX_MAX, PAD_PCT, PAD_MIN, MIN_ACROSS, TCONF_MAX = 64, 18, 2, 4, 127
MAX_SIDE = 2048                 # runtime/vocr_textspot.h VOCR_TEXTSPOT_MAX_SIDE: the C spotter refuses a larger side


def halve(g):
    """The 2 x 2 mean (rounded) of the spotter's half scale: ceil(H / 2) x ceil(W / 2)."""
    g = np.asarray(g, np.int32)
    if g.shape[0] % 2:
        g = np.vstack([g, g[-1:]])
    if g.shape[1] % 2:
        g = np.hstack([g, g[:, -1:]])
    return ((g[0::2, 0::2] + g[0::2, 1::2] + g[1::2, 0::2] + g[1::2, 1::2] + 2) >> 2).astype(np.uint8)


def _first_cum(cum, k):
    """The smallest v with cum[v] >= k (the C loops' percentile rule)."""
    return int(np.searchsorted(cum, k, side="left"))


def line_dark(g, box):
    """vocr_line_dark: 1 if the text in box (x, y, w, h) of grey g is darker than its background."""
    x, y, w, h = box
    a = np.asarray(g)[y:y + h, x:x + w].astype(np.int32)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    bg = int(np.sort(border)[len(border) // 2])
    cum = np.cumsum(np.bincount(a.ravel(), minlength=256)).astype(np.int64)
    n = w * h
    lo = _first_cum(cum, n * 2 // 100)
    hi = _first_cum(cum, n * 98 // 100)
    d = max(16, (hi - lo) // 4)
    nd = int((a < bg - d).sum())
    nb = int((a > bg + d).sum())
    return 1 if nd > 8 * nb else 0


def _components(mask):
    """4-connected components of a boolean map as lists of runs (row, c0, c1)."""
    h, w = mask.shape
    parent = {}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    runs_prev, runs_all = [], []
    for i in range(h):
        row = mask[i]
        runs = []
        j = 0
        while j < w:
            if row[j]:
                j0 = j
                while j < w and row[j]:
                    j += 1
                rid = len(runs_all)
                runs_all.append((i, j0, j))
                parent[rid] = rid
                runs.append(rid)
            else:
                j += 1
        for r in runs:
            _, a0, a1 = runs_all[r]
            for q in runs_prev:
                _, b0, b1 = runs_all[q]
                if a0 < b1 and b0 < a1:
                    ra, rb = find(r), find(q)
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)
        runs_prev = runs
    label = {}
    for r in range(len(runs_all)):
        label.setdefault(find(r), []).append(runs_all[r])
    return list(label.values())


def _joins(a, b):
    """textspot6_twin._joins on boxes (x, y, w, h, vertical, conf, dark, tag, tconf): one tag, one orientation,
    an overlap along the line, a gap across it at most the larger size across, sizes across within 3 : 2."""
    if a[7] != b[7] or a[4] != b[4]:
        return False
    if a[4]:
        ov = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
        gap = max(a[0], b[0]) - min(a[0] + a[2], b[0] + b[2])
        sa, sb = a[2], b[2]
    else:
        ov = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
        gap = max(a[1], b[1]) - min(a[1] + a[3], b[1] + b[3])
        sa, sb = a[3], b[3]
    return ov > 0 and gap <= max(sa, sb) and 2 * max(sa, sb) <= 3 * min(sa, sb)


def _rows(items, key_sort, lo, size, key_row):
    out, run = [], []
    for it in sorted(items, key=key_sort):
        if run:
            f = run[0]
            ov = min(lo(it) + size(it), lo(f) + size(f)) - max(lo(it), lo(f))
            if 2 * ov >= min(size(it), size(f)):
                run.append(it)
                continue
            out += sorted(run, key=key_row)
        run = [it]
    return out + sorted(run, key=key_row)


def _order_block(members):
    if members[0][4]:
        return _rows(members, key_sort=lambda b: (-(b[0] + b[2]), b[1], b[0], b[2], b[3]),
                     lo=lambda b: b[0], size=lambda b: b[2], key_row=lambda b: (b[1], -(b[0] + b[2]), b[3], b[2]))
    return _rows(members, key_sort=lambda b: (b[1], b[0], b[3], b[2]),
                 lo=lambda b: b[1], size=lambda b: b[3], key_row=lambda b: (b[0], b[1], b[3], b[2]))


def _group_and_order(boxes, height):
    """textspot6_twin.group_and_order: boxes [(x, y, w, h, vertical, conf, dark, tag, tconf)] -> (ordered boxes
    [(..., block)], blocks [(x, y, w, h, vertical, first, count, tag)], pick)."""
    boxes = sorted(tuple(b) for b in boxes)
    n = len(boxes)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(n):
        for j in range(i + 1, n):
            if _joins(boxes[i], boxes[j]):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(boxes[i])
    blocks = []
    for members in groups.values():
        x0 = min(b[0] for b in members)
        y0 = min(b[1] for b in members)
        x1 = max(b[0] + b[2] for b in members)
        y1 = max(b[1] + b[3] for b in members)
        blocks.append(((x0, y0, x1 - x0, y1 - y0, members[0][4], members[0][7]), _order_block(members)))
    rows, run = [], []
    for bk in sorted(blocks, key=lambda t: (t[0][1], t[0][0], t[0][3], t[0][2], t[0][5])):
        if run:
            f = run[0][0]
            b = bk[0]
            ov = min(b[1] + b[3], f[1] + f[3]) - max(b[1], f[1])
            if 2 * ov >= min(b[3], f[3]):
                run.append(bk)
                continue
            rows.append(run)
        run = [bk]
    if run:
        rows.append(run)
    ordered = []
    for run in rows:
        if all(t[0][4] for t in run):
            ordered += sorted(run, key=lambda t: (-(t[0][0] + t[0][2]), t[0][1], t[0][3], t[0][2], t[0][5]))
        else:
            ordered += sorted(run, key=lambda t: (t[0][0], t[0][1], t[0][3], t[0][2], t[0][5]))
    out_boxes, out_blocks = [], []
    for k, (bb, members) in enumerate(ordered):
        out_blocks.append((*bb[:5], len(out_boxes), len(members), bb[5]))
        out_boxes += [(*m, k) for m in members]
    pick, best = -1, None
    for k, (bb, members) in enumerate(ordered):
        if bb[5] != 0:
            continue
        score = sum(m[2] * m[3] for m in members) * (2 if 2 * bb[1] + bb[3] >= height else 1)
        key = (score, bb[1])
        if best is None or key > best:
            pick, best = k, key
    return out_boxes, out_blocks, pick


class Spotter:
    """textspot8 exported by tools/export_onnx.py (its "exact" or "int8" file; both give the C runtime's codes)."""

    def __init__(self, path, threads=1, providers=None, session_options=None):
        self.sess = _session(path, threads, providers, session_options)
        self.meta = m = _meta(self.sess)
        if m["vocr.kind"] != "spotter":
            raise ValueError(f"{path}: not a spotter")
        self.tags = m["vocr.tags"]
        self.half_above, self.half_above_h = m["vocr.half_above"], m["vocr.half_above_h"]
        self.conf_min, self.min_cells = m["vocr.conf_min"], m["vocr.min_cells"]
        self.stride = m["vocr.map_stride"]
        self.classes = m["vocr.classes"]

    def scale_of(self, width, height):
        """2 when a frame of this size is halved before the network (a region is read at its frame's scale)."""
        wide = self.half_above and width > self.half_above
        tall = self.half_above_h and height > self.half_above_h
        return 2 if wide or tall else 1

    def head(self, x):
        return self.sess.run(["head"], {"grey": np.ascontiguousarray(x, np.uint8)[None, None]})[0][0]

    def maps(self, x):
        """The decode's maps (4 + T, ceil(h / 4), ceil(w / 4)) of a network input x (h x w uint8)."""
        head = self.head(x).astype(np.int64)
        h, w = x.shape
        T = self.tags
        hn, wn = head.shape[1:]
        out = np.empty((4 + T, 2 * hn, 2 * wn), np.int64)
        for dy in (0, 1):
            for dx in (0, 1):
                c = 4 * (2 * dy + dx)
                out[:4, dy::2, dx::2] = head[c:c + 4]
                c = 16 + T * (2 * dy + dx)
                out[4:, dy::2, dx::2] = head[c:c + T]
        return out[:, :-(-h // 4), :-(-w // 4)]

    def spot(self, img, scale=None, frame_size=None):
        """Boxes on a frame or region (grey or RGB(A) uint8, 1 to 2048 px a side, as the C spotter takes). scale: 1 or
        2, or None for the blob's rule on frame_size (width, height; default the image's own): read a region cut from
        a frame at the frame's scale, as Rubi does."""
        g = grey_of(img)
        H_, W_ = g.shape
        if not H_ or not W_ or H_ > MAX_SIDE or W_ > MAX_SIDE:
            raise ValueError(f"a {W_} x {H_} image: the spotter reads 1 to {MAX_SIDE} px a side, as the C runtime "
                             "does (scale the frame to 960 x 544 first)")
        if scale is None:
            fw, fh = frame_size if frame_size is not None else (W_, H_)
            scale = self.scale_of(fw, fh)
        if scale not in (1, 2):
            raise ValueError("scale: 1 or 2")
        x = halve(g) if scale == 2 else g
        mp = self.maps(x)
        return self.decode(mp, scale, g)

    def decode(self, maps, f, g):
        s = self.stride
        tags = maps.shape[0] - 4
        kmap, omap, a1map, a2map = (maps[c] for c in range(4))
        H_, W_ = g.shape
        cands = []
        for runs in _components(kmap > 0):
            n = sum(b - a for _, a, b in runs)
            if n < self.min_cells:
                continue
            rows = np.concatenate([np.full(b - a, i, np.int64) for i, a, b in runs])
            cols = np.concatenate([np.arange(a, b, dtype=np.int64) for _, a, b in runs])
            conf = int(kmap[rows, cols].sum()) // n
            if conf < self.conf_min:
                continue
            vertical = int(omap[rows, cols].sum()) > 0
            a1, a2 = a1map[rows, cols], a2map[rows, cols]
            across_idx, along_idx = (cols, rows) if vertical else (rows, cols)
            c2 = 2 * s * across_idx + s - 1
            near = (f * int((c2 - a1).sum())) // (2 * n)
            far = -((-(f * int((c2 + a2).sum()))) // (2 * n))
            lo, hi = f * s * int(along_idx.min()), f * s * (int(along_idx.max()) + 1)
            across = far - near
            if across < MIN_ACROSS * f:
                continue
            pad = max(PAD_MIN, (across * PAD_PCT + 50) // 100)
            a0, a1_, b0, b1 = near - pad, far + pad, lo - pad, hi + pad
            if vertical:
                x0, x1, y0, y1 = a0, a1_, b0, b1
            else:
                x0, x1, y0, y1 = b0, b1, a0, a1_
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W_, x1), min(H_, y1)
            if x1 <= x0 or y1 <= y0:
                continue
            scores = [0] + [int(maps[4 + t][rows, cols].sum()) for t in range(tags)]
            tag = 0
            for c in range(1, len(scores)):
                if scores[c] > scores[tag]:
                    tag = c
            tconf = 0
            if len(scores) > 1:
                second = max(scores[c] for c in range(len(scores)) if c != tag)
                tconf = min(TCONF_MAX, (scores[tag] - second) // n)
            key = (-conf, -n, int(vertical), y0, x0, y1 - y0, x1 - x0, tag)
            cands.append((key, (x0, y0, x1 - x0, y1 - y0, int(vertical), conf, tag, tconf)))
        truncated = int(len(cands) > BOX_MAX)
        kept = [c for _, c in sorted(cands)[:BOX_MAX]]
        boxes9 = [(bx, by, bw, bh, v, conf, int(line_dark(g, (bx, by, bw, bh))), tag, tconf)
                  for bx, by, bw, bh, v, conf, tag, tconf in kept]
        boxes, blocks, pick = _group_and_order(boxes9, H_)
        presence = max([b[5] for b in boxes if b[7] == 0], default=0)
        return Spots(tuple(Box(x, y, w, h, v, c, d, blk, t, tc) for x, y, w, h, v, c, d, t, tc, blk in boxes),
                     tuple(Block(*b) for b in blocks), pick, truncated, presence)


# ----------------------------------------------------------------------------- pages

def read_region(frame, spotter, rec, region=None, lm=None, sections=True, tags=None, scale=None):
    """Read a frame or a region of it as Rubi reads the region a player draws: the spotter on the region alone at
    the frame's scale, then every horizontal box (or the boxes of `tags` only) cut from the region's grey and read
    with its dark flag. region: (x, y, w, h) in frame px, or None for the whole frame. Returns [Line] in reading
    order, boxes and character cells in frame px."""
    g = grey_of(frame)
    H, W = g.shape
    rx, ry, rw, rh = (int(v) for v in (region if region is not None else (0, 0, W, H)))
    x0, y0 = min(max(0, rx), W), min(max(0, ry), H)             # the region, clipped to the frame
    x1, y1 = min(max(0, rx + rw), W), min(max(0, ry + rh), H)
    sub = np.ascontiguousarray(g[y0:y1, x0:x1])
    if not sub.size:
        return []
    sp = spotter.spot(sub, scale=scale, frame_size=(W, H))
    out = []
    for b in sp.boxes:
        if b.vertical or (tags is not None and b.tag not in tags):
            continue
        crop = sub[b.y:b.y + b.h, b.x:b.x + b.w]
        ln = rec.read(crop, invert=bool(b.dark), sections=sections, lm=lm)
        ln.box = (x0 + b.x, y0 + b.y, b.w, b.h)
        ln.tag, ln.conf, ln.block, ln.dark = b.tag, b.conf, b.block, b.dark
        for c in ln.chars + ln.text_chars:          # place() copies: no Char is in both lists
            c.x, c.x0, c.x1 = c.x + x0 + b.x, c.x0 + x0 + b.x, c.x1 + x0 + b.x
            c.y0, c.y1 = y0 + b.y, y0 + b.y + b.h
        out.append(ln)
    return out


# ----------------------------------------------------------------------------- command line

def _load_image(path):
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def main(argv=None):
    import argparse
    from pathlib import Path
    ap = argparse.ArgumentParser(description="VOCR through ONNX Runtime (docs/onnx.md)")
    ap.add_argument("images", nargs="+")
    ap.add_argument("--models", required=True, help="the directory holding the ONNX files")
    ap.add_argument("--tier", default="N", choices=("S", "N", "L"))
    ap.add_argument("--mode", default="exact", choices=("exact", "fast"))
    ap.add_argument("--spotter", default="exact", choices=("exact", "int8"))
    ap.add_argument("--region", help="X,Y,W,H in frame px (default: the whole frame)")
    ap.add_argument("--lm", help="a .vlm character LM (beam 8 + LM at Rubi's settings)")
    ap.add_argument("--plain", action="store_true", help="without the blob's head gate / two-level head")
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args(argv)
    d = Path(a.models)
    spot = Spotter(d / f"textspot8_{a.spotter}.onnx", a.threads)
    rec = Recognizer(d / f"VOCR6.2_{a.tier}_{a.mode}.onnx", a.threads)
    lm = CharLM(a.lm, tier=a.tier) if a.lm else None
    region = tuple(int(v) for v in a.region.split(",")) if a.region else None
    for p in a.images:
        lines = read_region(_load_image(p), spot, rec, region, lm=lm, sections=not a.plain)
        print(json.dumps({"image": p, "lines": [
            {"text": ln.text, "greedy": ln.greedy, "box": list(ln.box), "tag": spot.classes[ln.tag],
             "dark": ln.dark, "conf_min": round(ln.conf_min, 4),
             # one [character, first CTC frame (-1: inserted by the LM), cell start, cell end] per character of "text"
             "chars": [[c.text, c.f0, round(c.x0, 2), round(c.x1, 2)] for c in ln.text_chars]} for ln in lines]},
            ensure_ascii=False))


if __name__ == "__main__":
    main()
