"""ctypes binding of the vocr C runtime (runtime/vocr.h) for host tests and evaluation.

    from pyvocr import Vocr
    r = Vocr("v1_w8.vocr", mode="w8i")          # (re)builds runtime/build/libvocr.so if needed
    text = r.text(crop)                          # crop: HxW uint8 grey or HxWx4 uint8 RGBA
    text = r.text(crop, invert=True)             # dark text: read as 255 - grey (VOCR_INVERT)
    text, logits = r.run(crop, logits=True)      # logits: frames x classes float32
    r.arena_bytes                                # exact arena bytes of this context

Not part of the console build: the runtime itself is dependency-free C.
"""
import ctypes as C
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LIB = HERE / "build" / "libvocr.so"
MODES = {"auto": 0, "f32": 1, "w8f": 2, "w8i": 3}
GREY8, RGBA8, INVERT = 1, 4, 0x100


class _Opts(C.Structure):
    _fields_ = [("tile_frames", C.c_uint), ("head_frames", C.c_uint), ("max_width", C.c_uint),
                ("mode", C.c_int), ("tail_conf", C.c_uint)]


class _Image(C.Structure):
    _fields_ = [("pixels", C.c_void_p), ("width", C.c_uint), ("height", C.c_uint),
                ("stride", C.c_size_t), ("format", C.c_int)]


class _Extra(C.Structure):
    _fields_ = [("logits", C.c_void_p), ("best", C.c_void_p), ("cap_frames", C.c_uint),
                ("cancel", C.c_void_p), ("user", C.c_void_p), ("width", C.c_uint),
                ("frames", C.c_uint), ("chars", C.c_uint)]


def _aligned(nbytes, align=16):
    raw = np.empty(nbytes + align, np.uint8)
    off = (-raw.ctypes.data) % align
    return raw[off:off + nbytes]


def _stale(lib):
    """True if lib is missing or older than the sources it is built from."""
    if not lib.exists():
        return True
    built = lib.stat().st_mtime
    return any((HERE / src).stat().st_mtime > built for src in ("vocr.c", "vocr.h"))


def load_lib(path=None):
    path = Path(path) if path else LIB
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

    def __init__(self, blob, mode="auto", tile=0, head=0, max_width=0, lib=None, check_crc=True, tail_conf=0):
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
        self.opts = _Opts(tile, head, max_width, MODES[mode] if isinstance(mode, str) else mode, tail_conf)
        self.arena_bytes = self.lib.vocr_arena_bytes(self._model, C.byref(self.opts))
        if not self.arena_bytes:
            raise ValueError(f"invalid options {mode=} {tile=} {head=} {max_width=} for this blob")
        self._arena = _aligned(self.arena_bytes)
        self._check(self.lib.vocr_init(self._ctx, self._model, C.byref(self.opts),
                                       self._arena.ctypes.data, self.arena_bytes), "vocr_init")
        self.blob_bytes = len(data)
        self.n_classes = int.from_bytes(data[24:28], "little")
        self.stride = int.from_bytes(data[20:22], "little")
        self._text = C.create_string_buffer(1 << 16)

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

    def run(self, img, logits=False, invert=False):
        """invert=True reads the crop as 255 - grey (VOCR_INVERT), for dark text."""
        keep, im = self._image(img, invert)
        ex = _Extra()
        out = best = None
        cap = self.resized_width(im.width, im.height) // self.stride + 1
        best = np.zeros(cap, np.uint16)
        ex.best, ex.cap_frames = best.ctypes.data, cap
        if logits:
            out = np.zeros((cap, self.n_classes), np.float32)
            ex.logits = out.ctypes.data
        rc = self.lib.vocr_run(self._ctx, C.byref(im), self._text, len(self._text), C.byref(ex))
        self._check(rc, "vocr_run")
        del keep
        text = self._text.value.decode("utf-8")
        self.last_best = best[:ex.frames]
        return (text, out[:ex.frames]) if logits else text

    def text(self, img, invert=False):
        return self.run(img, invert=invert)
