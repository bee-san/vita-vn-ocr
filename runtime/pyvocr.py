"""ctypes binding of the vocr C runtime (runtime/vocr.h) for host tests and evaluation.

    from pyvocr import Vocr
    r = Vocr("v1_w8.vocr", mode="w8i")          # (re)builds runtime/build/libvocr.so if needed
    text = r.text(crop)                          # crop: HxW uint8 grey or HxWx4 uint8 RGBA
    text = r.text(crop, invert=True)             # dark text: read as 255 - grey (VOCR_INVERT)
    text, logits = r.run(crop, logits=True)      # logits: frames x classes float32
    r.arena_bytes                                # exact arena bytes of this context
    out = r.run_topk(crop, 8)                    # greedy text + per-frame top-8 + confidence
    lm = VocrLM(r, "lm.vlm", beam=8, topk=8, alpha=300, beta=0, gate=3000)   # runtime/vocr_lm.h
    text = lm.decode(out["topv"], out["topc"], out["blank"])                  # beam + LM text
    out2, st = r.redecode(out, tau=2000, open=2000)  # runtime/vocr_post.h: counter re-decode (edited copy)
    text, n = post_counters(text, r.lib)             # the counter rule: ッ月 -> ヶ月 ...
    g = Vocr("VOCR5_N_gated.vocr", mode="w8i", head_gate="on")   # docs/headgate.md: skip the head on blank frames
    out = g.run_topk(crop, 8, gate_score=True)       # ... out["gated"] frames skipped, out["gate_score"] per frame

Not part of the console build: the runtime itself is dependency-free C.
"""
import ctypes as C
import os
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LIB = HERE / "build" / "libvocr.so"
MODES = {"auto": 0, "f32": 1, "w8f": 2, "w8i": 3}
HEAD_GATE = {"off": 0, "on": 1, "score": 2}
GREY8, RGBA8, INVERT = 1, 4, 0x100


class _Opts(C.Structure):
    _fields_ = [("tile_frames", C.c_uint), ("head_frames", C.c_uint), ("max_width", C.c_uint),
                ("mode", C.c_int), ("head_gate", C.c_uint)]


class _Image(C.Structure):
    _fields_ = [("pixels", C.c_void_p), ("width", C.c_uint), ("height", C.c_uint),
                ("stride", C.c_size_t), ("format", C.c_int)]


class _Extra(C.Structure):
    _fields_ = [("logits", C.c_void_p), ("best", C.c_void_p), ("cap_frames", C.c_uint),
                ("cancel", C.c_void_p), ("user", C.c_void_p), ("width", C.c_uint),
                ("frames", C.c_uint), ("chars", C.c_uint), ("topk", C.c_uint),
                ("topk_logit", C.c_void_p), ("topk_class", C.c_void_p), ("blank_logit", C.c_void_p),
                ("conf_min", C.c_float), ("conf_mean", C.c_float),
                ("head_in", C.c_void_p), ("gate_score", C.c_void_p), ("gated", C.c_uint), ("ungated", C.c_uint)]


class _PostOpts(C.Structure):
    _fields_ = [("tau", C.c_int), ("open", C.c_int)]


class _PostStats(C.Structure):
    _fields_ = [("contexts", C.c_uint), ("numerals", C.c_uint), ("brackets", C.c_uint)]


class _LMOpts(C.Structure):
    _fields_ = [("beam", C.c_uint), ("topk", C.c_uint), ("max_frames", C.c_uint), ("alpha", C.c_int),
                ("beta", C.c_int), ("gate", C.c_int), ("flags", C.c_uint), ("bracket", C.c_int)]


def _aligned(nbytes, align=16):
    raw = np.empty(nbytes + align, np.uint8)
    off = (-raw.ctypes.data) % align
    return raw[off:off + nbytes]


def _stale(lib):
    """True if lib is missing or older than the sources it is built from."""
    if not lib.exists():
        return True
    built = lib.stat().st_mtime
    return any((HERE / src).stat().st_mtime > built
               for src in ("vocr.c", "vocr.h", "vocr_lm.c", "vocr_lm.h", "vocr_post.c", "vocr_post.h"))


def load_lib(path=None):
    """The ctypes library: `path`, else $VOCR_LIB (a frozen build, used as is), else build/libvocr.so
    (rebuilt whenever a runtime source is newer)."""
    path = Path(path) if path else Path(os.environ["VOCR_LIB"]) if os.environ.get("VOCR_LIB") else LIB
    if path == LIB and _stale(LIB):        # never test an old build of vocr.c
        subprocess.run(["make", "-s", "-C", str(HERE), str(LIB.relative_to(HERE))], check=True)
    lib = C.CDLL(str(path))
    vp, sz, u = C.c_void_p, C.c_size_t, C.c_uint
    lib.vocr_load.argtypes = [vp, vp, sz]
    lib.vocr_check_crc.argtypes = [vp, sz]
    lib.vocr_arena_bytes.argtypes = [vp, C.POINTER(_Opts)]
    lib.vocr_arena_bytes.restype = sz
    lib.vocr_init.argtypes = [vp, vp, C.POINTER(_Opts), vp, sz]
    lib.vocr_run.argtypes = [vp, C.POINTER(_Image), C.c_char_p, sz, C.POINTER(_Extra)]
    lib.vocr_resized_width.argtypes = [vp, u, u]
    lib.vocr_resized_width.restype = u
    lib.vocr_preprocess.argtypes = [vp, C.POINTER(_Image), vp, sz, C.POINTER(u), C.POINTER(u)]
    lib.vocr_strerror.argtypes = [C.c_int]
    lib.vocr_strerror.restype = C.c_char_p
    lib.vocr_sizes.argtypes = [C.POINTER(sz), C.POINTER(sz)]
    lib.vocr_class_text.argtypes = [vp, u, C.POINTER(u)]
    lib.vocr_class_text.restype = vp
    if hasattr(lib, "vocr_has_gate"):        # the head gate (docs/headgate.md, issue #117)
        lib.vocr_has_gate.argtypes = [vp]
    if hasattr(lib, "vocr_post_counters"):   # runtime/vocr_post.c (issue #109)
        lib.vocr_post_counters.argtypes = [C.c_char_p]
        lib.vocr_post_counters.restype = u
        lib.vocr_post_redecode.argtypes = [vp, C.POINTER(_Extra), C.POINTER(_PostOpts), C.POINTER(_PostStats)]
        lib.vocr_post_greedy.argtypes = [vp, C.POINTER(_Extra), C.c_char_p, sz, C.POINTER(u)]
    if not hasattr(lib, "vocr_lm_load"):     # a library built from vocr.c alone (eval/vocr3_report.py): no decoder
        return lib
    lib.vocr_lm_check_crc.argtypes = [vp, sz]
    lib.vocr_lm_load.argtypes = [vp, vp, sz]
    lib.vocr_lm_workspace_bytes.argtypes = [vp, vp, C.POINTER(_LMOpts)]
    lib.vocr_lm_workspace_bytes.restype = sz
    lib.vocr_lm_init.argtypes = [vp, vp, vp, C.POINTER(_LMOpts), vp, sz]
    lib.vocr_lm_decode.argtypes = [vp, C.POINTER(_Extra), C.c_char_p, sz, C.POINTER(u)]
    lib.vocr_lm_token.argtypes = [vp, C.c_uint32]
    lib.vocr_lm_token.restype = u
    lib.vocr_lm_score.argtypes = [vp, C.POINTER(C.c_int32), u, C.POINTER(C.c_int32)]
    lib.vocr_lm_score.restype = C.c_int32
    lib.vocr_lm_initial.argtypes = [vp, u, C.POINTER(C.c_int32)]
    lib.vocr_lm_sizes.argtypes = [C.POINTER(sz), C.POINTER(sz)]
    return lib


class VocrError(RuntimeError):
    def __init__(self, lib, rc, what):
        super().__init__(f"{what}: {lib.vocr_strerror(rc).decode()} ({rc})")
        self.rc = rc


class Vocr:
    @classmethod
    def from_checkpoint(cls, ckpt, int8=True, **kw):
        """Export a train.py checkpoint in memory (train/export.py) and load it.
        int8=True gives the 0.86 MB int8 blob (default mode w8f), False the fp32 one."""
        import hashlib
        import sys
        sys.path.insert(0, str(HERE.parent / "train"))
        import export
        model, vocab, height = export.load_checkpoint(ckpt)
        blob, _ = export.build_blob(export.model_ops(model, height), vocab, height, int8,
                                    source_sha=hashlib.sha256(Path(ckpt).read_bytes()).digest())
        return cls(blob, **kw)

    def __init__(self, blob, mode="auto", tile=0, head=0, max_width=0, lib=None, check_crc=True, head_gate="off"):
        """head_gate: "off", "on" (skip the head on the frames the blob's gate calls blank) or "score" (compute
        the gate's scores only); "on" and "score" need mode w8i and a blob with a gate (docs/headgate.md)."""
        self.lib = lib if lib is not None else load_lib()
        data = Path(blob).read_bytes() if isinstance(blob, (str, Path)) else bytes(blob)
        self._blob = _aligned(len(data))
        self._blob[:] = np.frombuffer(data, np.uint8)
        msz, csz = C.c_size_t(), C.c_size_t()
        self.lib.vocr_sizes(C.byref(msz), C.byref(csz))
        self._model = C.create_string_buffer(msz.value)
        self._ctx = C.create_string_buffer(csz.value)
        if check_crc:
            self._check(self.lib.vocr_check_crc(self._blob.ctypes.data, len(data)), "vocr_check_crc")
        self._check(self.lib.vocr_load(self._model, self._blob.ctypes.data, len(data)), "vocr_load")
        self.opts = _Opts(tile, head, max_width, MODES[mode] if isinstance(mode, str) else mode,
                          HEAD_GATE[head_gate] if isinstance(head_gate, str) else head_gate)
        self.arena_bytes = self.lib.vocr_arena_bytes(self._model, C.byref(self.opts))
        if not self.arena_bytes:
            raise ValueError(f"invalid options {mode=} {tile=} {head=} {max_width=} for this blob")
        self._arena = _aligned(self.arena_bytes)
        self._check(self.lib.vocr_init(self._ctx, self._model, C.byref(self.opts),
                                       self._arena.ctypes.data, self.arena_bytes), "vocr_init")
        self.blob_bytes = len(data)
        self.n_classes = int.from_bytes(data[24:28], "little")
        self.stride = int.from_bytes(data[20:22], "little")
        self.has_gate = bool(self.lib.vocr_has_gate(self._model)) if hasattr(self.lib, "vocr_has_gate") else False
        self.head_k = self._head_k(data)
        self._text = C.create_string_buffer(1 << 16)

    @staticmethod
    def _head_k(data):
        """The head's input width K (the last op record's field 32)."""
        n_ops = int.from_bytes(data[22:24], "little")
        ops = int.from_bytes(data[28:32], "little")
        rec = ops + 64 * (n_ops - 1)
        return int.from_bytes(data[rec + 32:rec + 34], "little")

    def _check(self, rc, what):
        if rc:
            raise VocrError(self.lib, rc, what)

    def _image(self, img, invert=False):
        img = np.asarray(img)
        if img.dtype != np.uint8:      # no silent casts: a [0, 1] float crop would become 0/1
            raise TypeError(f"image must be uint8, not {img.dtype}")
        img = np.ascontiguousarray(img)
        inv = INVERT if invert else 0
        if img.ndim == 2:
            return img, _Image(img.ctypes.data, img.shape[1], img.shape[0], img.strides[0], GREY8 | inv)
        if img.ndim == 3 and img.shape[2] == 4:
            return img, _Image(img.ctypes.data, img.shape[1], img.shape[0], img.strides[0], RGBA8 | inv)
        raise ValueError("image must be HxW grey or HxWx4 RGBA uint8")

    def resized_width(self, w, h):
        return self.lib.vocr_resized_width(self._model, w, h)

    def class_text(self, c):
        """UTF-8 text of class c ("" for the blank, class 0)."""
        n = C.c_uint()
        p = self.lib.vocr_class_text(self._model, c, C.byref(n))
        return C.string_at(p, n.value).decode("utf-8") if n.value else ""

    def preprocess(self, img, invert=False):
        """The grey network input (height x padded width, uint8) and the unpadded width."""
        keep, im = self._image(img, invert)
        w, pw = C.c_uint(), C.c_uint()
        self._check(self.lib.vocr_preprocess(self._model, C.byref(im), None, 0, C.byref(w),
                                              C.byref(pw)), "vocr_preprocess")
        h = int.from_bytes(bytes(self._blob[16:18]), "little")
        out = np.empty((h, pw.value), np.uint8)
        self._check(self.lib.vocr_preprocess(self._model, C.byref(im), out.ctypes.data, out.size,
                                              C.byref(w), C.byref(pw)), "vocr_preprocess")
        return out, w.value

    def _gate_buffers(self, ex, cap, head_in, gate_score):
        hi = gs = None
        if head_in:
            hi = np.zeros((cap, self.head_k), np.float32)
            ex.head_in = hi.ctypes.data
        if gate_score:
            gs = np.full(cap, np.nan, np.float32)
            ex.gate_score = gs.ctypes.data
        return hi, gs

    def run(self, img, logits=False, invert=False, head_in=False, gate_score=False):
        """invert=True reads the crop as 255 - grey (VOCR_INVERT), for dark text. head_in / gate_score: keep each
        frame's head input (frames x K) / gate score in self.last_head_in / self.last_gate_score (docs/headgate.md);
        self.last_gated counts the frames the head gate skipped (none when the logits are asked for)."""
        keep, im = self._image(img, invert)
        ex = _Extra()
        out = best = None
        cap = self.resized_width(im.width, im.height) // self.stride + 1
        best = np.zeros(cap, np.uint16)
        ex.best, ex.cap_frames = best.ctypes.data, cap
        if logits:
            out = np.zeros((cap, self.n_classes), np.float32)
            ex.logits = out.ctypes.data
        hi, gs = self._gate_buffers(ex, cap, head_in, gate_score)
        rc = self.lib.vocr_run(self._ctx, C.byref(im), self._text, len(self._text), C.byref(ex))
        self._check(rc, "vocr_run")
        del keep
        text = self._text.value.decode("utf-8")
        self.last_best = best[:ex.frames]
        self.last_gated = ex.gated
        self.last_head_in = hi[:ex.frames] if hi is not None else None
        self.last_gate_score = gs[:ex.frames] if gs is not None else None
        return (text, out[:ex.frames]) if logits else text

    def text(self, img, invert=False):
        return self.run(img, invert=invert)

    def run_topk(self, img, k, invert=False, blank=True, head_in=False, gate_score=False, ungated=False):
        """vocr_run with vocr_extra.topk = k: the greedy text, per frame the k largest logits (topv,
        decreasing) and their classes (topc), the blank's logit, and conf_min / conf_mean; "best" (each
        frame's argmax), "gated" (frames the head gate skipped) and, if asked, "head_in" / "gate_score".
        ungated=True: this line without skipping (vocr_extra.ungated), as head_gate off reads it."""
        keep, im = self._image(img, invert)
        cap = self.resized_width(im.width, im.height) // self.stride + 1
        topv = np.zeros((cap, k), np.float32)
        topc = np.zeros((cap, k), np.uint16)
        bl = np.zeros(cap, np.float32)
        best = np.zeros(cap, np.uint16)
        ex = _Extra()
        ex.best, ex.cap_frames, ex.topk = best.ctypes.data, cap, k
        ex.topk_logit, ex.topk_class = topv.ctypes.data, topc.ctypes.data
        ex.blank_logit = bl.ctypes.data if blank else None
        ex.ungated = 1 if ungated else 0
        hi, gs = self._gate_buffers(ex, cap, head_in, gate_score)
        self._check(self.lib.vocr_run(self._ctx, C.byref(im), self._text, len(self._text), C.byref(ex)), "vocr_run")
        del keep
        n = ex.frames
        self.last_best = best[:n]
        out = {"text": self._text.value.decode("utf-8"), "topv": topv[:n], "topc": topc[:n],
               "blank": bl[:n] if blank else None, "conf_min": ex.conf_min, "conf_mean": ex.conf_mean,
               "frames": n, "chars": ex.chars, "best": best[:n], "gated": ex.gated}
        if hi is not None:
            out["head_in"] = hi[:n]
        if gs is not None:
            out["gate_score"] = gs[:n]
        return out

    def redecode(self, out, tau=0, open=0):
        """runtime/vocr_post.h's counter-context re-decode on run_topk()'s output (tau, open in 1/1000 nat):
        (a copy of out with the edited topc and its greedy "text", {"edits", "contexts", "numerals",
        "brackets"}). out itself is not changed."""
        ex, topv, topc, blank = _redecode_extra(out)
        st = _PostStats()
        rc = self.lib.vocr_post_redecode(self._model, C.byref(ex), C.byref(_PostOpts(tau, open)), C.byref(st))
        if rc < 0:
            raise VocrError(self.lib, rc, "vocr_post_redecode")
        res = dict(out, topc=topc)
        if rc:
            n = C.c_uint()
            self._check(self.lib.vocr_post_greedy(self._model, C.byref(ex), self._text, len(self._text), C.byref(n)),
                        "vocr_post_greedy")
            res["text"] = self._text.value.decode("utf-8")
        return res, {"edits": rc, "contexts": st.contexts, "numerals": st.numerals, "brackets": st.brackets}


def post_counters(text, lib=None):
    """runtime/vocr_post.h's counter rule through the C library: (text, changes)."""
    lib = lib if lib is not None else load_lib()
    buf = C.create_string_buffer(text.encode("utf-8"))
    n = lib.vocr_post_counters(buf)
    return buf.value.decode("utf-8"), n


def _redecode_extra(out):
    topv = np.ascontiguousarray(out["topv"], np.float32)
    topc = np.array(out["topc"], np.uint16, copy=True)
    ex = _Extra()
    ex.frames, ex.cap_frames, ex.topk = topv.shape[0], topv.shape[0], topv.shape[1]
    ex.topk_logit, ex.topk_class = topv.ctypes.data, topc.ctypes.data
    blank = None
    if out.get("blank") is not None:
        blank = np.ascontiguousarray(out["blank"], np.float32)
        ex.blank_logit = blank.ctypes.data
    return ex, topv, topc, blank


class VocrLM:
    """runtime/vocr_lm.h: the character n-gram LM and CTC prefix beam search, for one Vocr model.
    lm: a VOLM blob (path or bytes), or None with alpha = 0. alpha, beta, gate, bracket in 1/1000 (nat)."""

    def __init__(self, vocr, lm, beam=8, topk=8, alpha=0, beta=0, gate=0, flags=0, max_frames=0, bracket=0,
                 check_crc=True):
        self.lib, self.vocr = vocr.lib, vocr
        lsz, csz = C.c_size_t(), C.c_size_t()
        self.lib.vocr_lm_sizes(C.byref(lsz), C.byref(csz))
        self._lm = C.create_string_buffer(lsz.value)
        self._ctx = C.create_string_buffer(csz.value)
        self._blob = None
        lm_ptr = None
        if lm is not None:
            data = Path(lm).read_bytes() if isinstance(lm, (str, Path)) else bytes(lm)
            self._blob = _aligned(len(data))
            self._blob[:] = np.frombuffer(data, np.uint8)
            if check_crc:
                self._check(self.lib.vocr_lm_check_crc(self._blob.ctypes.data, len(data)), "vocr_lm_check_crc")
            self._check(self.lib.vocr_lm_load(self._lm, self._blob.ctypes.data, len(data)), "vocr_lm_load")
            lm_ptr = self._lm
            self.lm_bytes = len(data)
        self.opts = _LMOpts(beam, topk, max_frames, alpha, beta, gate, flags, bracket)
        self.workspace_bytes = self.lib.vocr_lm_workspace_bytes(vocr._model, lm_ptr, C.byref(self.opts))
        if not self.workspace_bytes:
            raise ValueError(f"invalid vocr_lm options {beam=} {topk=} {alpha=} {beta=} {gate=} {flags=} {bracket=}")
        self._ws = _aligned(self.workspace_bytes)
        self._check(self.lib.vocr_lm_init(self._ctx, vocr._model, lm_ptr, C.byref(self.opts), self._ws.ctypes.data,
                                          self.workspace_bytes), "vocr_lm_init")
        self._text = C.create_string_buffer(1 << 16)

    def _check(self, rc, what):
        if rc:
            raise VocrError(self.lib, rc, what)

    def decode(self, topv, topc, blank=None):
        """The beam + LM text of one line from its per-frame top-K (frames x K) and blank logits."""
        topv = np.ascontiguousarray(topv, np.float32)
        topc = np.ascontiguousarray(topc, np.uint16)
        ex = _Extra()
        ex.frames, ex.cap_frames, ex.topk = topv.shape[0], topv.shape[0], topv.shape[1]
        ex.topk_logit, ex.topk_class = topv.ctypes.data, topc.ctypes.data
        if blank is not None:
            blank = np.ascontiguousarray(blank, np.float32)
            ex.blank_logit = blank.ctypes.data
        n = C.c_uint()
        self._check(self.lib.vocr_lm_decode(self._ctx, C.byref(ex), self._text, len(self._text), C.byref(n)),
                    "vocr_lm_decode")
        return self._text.value.decode("utf-8")

    def score(self, state, token):
        """(log P(token | state) in Q10, next state) through the C scorer (tests)."""
        st = (C.c_int32 * len(state))(*state)
        nxt = (C.c_int32 * len(state))()
        lp = self.lib.vocr_lm_score(self._lm, st, token, nxt)
        return lp, tuple(nxt)
