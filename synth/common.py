"""Shared pieces for the Vita VN OCR toolchain.

find_lines() is an exact port of vjo_ocr_find_lines (core/ocr_lines.c), so the
training crops are the same crops the overlay hands its recognizer on the Vita.
"""
import hashlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]          # vita-ocr/
DATA = ROOT / "data"
MAX_W, MAX_H, MAX_LINES, MAX_LINE_H = 960, 544, 8, 96


def find_lines(rgb):
    """rgb: HxWx3 or HxWx4 uint8. Returns [(x, y, w, h), ...], or None where the
    C code returns VJO_E_OCR_REGION (oversized region, light background, a line
    taller than 88 px, or more than 8 lines)."""
    h_img, w_img = rgb.shape[:2]
    if not w_img or not h_img or w_img > MAX_W or h_img > MAX_H:
        return None
    bright = (rgb[..., 0] >= 180) & (rgb[..., 1] >= 180) & (rgb[..., 2] >= 180)
    counts = bright.sum(axis=1)
    if (counts > w_img * 3 // 4).any():
        return None
    has = counts > 0
    left = np.where(has, bright.argmax(axis=1), w_img)
    right = np.where(has, w_img - bright[:, ::-1].argmax(axis=1), 0)
    lines, active, start, last = [], False, 0, 0
    for y in range(h_img + 1):
        count = counts[y] if y < h_img else 0
        if count >= 2:
            if not active:
                start, active = y, True
            last = y
        if active and (y == h_img or y - last >= 3):
            h = last - start + 1
            if h >= 8:
                if h > MAX_LINE_H - 8 or len(lines) == MAX_LINES:
                    return None
                x1, x2 = int(left[start:last + 1].min()), int(right[start:last + 1].max())
                if x2 > x1:
                    lx = x1 - 4 if x1 > 4 else 0
                    ly = start - 4 if start > 4 else 0
                    ex = x2 + 4 if x2 + 4 < w_img else w_img
                    ey = last + 5 if last + 5 < h_img else h_img
                    lines.append((lx, ly, ex - lx, ey - ly))
            active = False
    return lines


def to_gray(rgb):
    """Integer luma, identical to the C side: (77 R + 150 G + 29 B) >> 8."""
    r, g, b = (rgb[..., i].astype(np.uint16) for i in range(3))
    return ((77 * r + 150 * g + 29 * b) >> 8).astype(np.uint8)


def load_vocab(path=DATA / "vocab.txt"):
    """Class i + 1 is vocab[i]; class 0 is the CTC blank."""
    return [c for c in Path(path).read_text(encoding="utf-8").split("\n") if c]


def is_test_vn(vn_id):
    """Deterministic VN-level split: about 15% of VNs are held out for evaluation,
    so no screenshot of a test VN is ever used for training or as a background."""
    return int(hashlib.md5(vn_id.encode()).hexdigest(), 16) % 100 < 15
