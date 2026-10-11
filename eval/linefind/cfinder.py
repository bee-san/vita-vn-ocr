"""Run runtime/vocr_lines.c from Python through its host driver (vocr_lines_test).

The driver is built on first use with the system C compiler unless VOCR_LINES_BIN
points at an existing build.
"""
import os
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def build(out=ROOT / "build" / "vocr_lines_test", cc=None):
    out = Path(out)
    src = [ROOT / "runtime" / "vocr_lines.c", ROOT / "runtime" / "vocr_lines_test.c"]
    if not out.exists() or any(s.stat().st_mtime > out.stat().st_mtime for s in src):
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([cc or os.environ.get("CC", "cc"), "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                        *map(str, src), "-o", str(out)], check=True)
    return out


class Finder:
    """callable(img) -> [(x, y, w, h)] or None, like common.find_lines."""

    def __init__(self, finder="grad", binary=None):
        self.code = {"grad": 0, "legacy": 1}[finder]
        self.bin = Path(binary or os.environ.get("VOCR_LINES_BIN") or build())
        self.last_dark = []
        self.last_truncated = False

    def __call__(self, img):
        self.last_dark = []
        self.last_truncated = False
        img = np.asarray(img)
        h, w = img.shape[:2]
        if img.ndim == 2:
            bpp, px = 1, img
        else:
            bpp = 4
            px = img if img.shape[2] == 4 else np.concatenate([img, np.full((h, w, 1), 255, np.uint8)], axis=2)
        with tempfile.NamedTemporaryFile(suffix=".bin") as f:
            f.write(struct.pack("<5I", w, h, bpp, self.code, w * bpp))
            f.write(np.ascontiguousarray(px, dtype=np.uint8).tobytes())
            f.flush()
            out = subprocess.run([str(self.bin), f.name], check=True, capture_output=True, text=True).stdout
        v = [int(t) for t in out.split()]
        if v[0] != 0:
            return None
        n = v[1]
        self.last_truncated = bool(v[2])
        boxes = [tuple(v[3 + 5 * i:3 + 5 * i + 4]) for i in range(n)]
        self.last_dark = [v[3 + 5 * i + 4] for i in range(n)]
        return boxes
