"""Line finders for eval/run.py --finder.

A finder takes an RGB region (H x W x 3 uint8, the box a player draws) and returns
line boxes [(x, y, w, h), ...] in region coordinates, or None when it rejects the
region (the overlay's VJO_E_OCR_REGION).

  overlay        synth/common.py find_lines(rgb, "legacy"): exact port of vita-jp-overlay's
                 bright-pixel row projection (core/ocr_lines.c), the shipped finder
  gradient       the research's polarity-agnostic gradient-energy projection
                 (scratch/real/grad_finder.py): rows above 25% of the max |dx|+|dy|
                 profile form runs, gaps under 3 px merge, runs under 8 px or under
                 0.55x the median height are dropped; full-width bands, 4/6 px pad
  py:MOD:FUNC    any Python function with the same contract (MOD may also be a .py path)
  cmd:COMMAND    an external finder: the region is written as a binary PPM, {} in
                 COMMAND is replaced by its path (else the path is appended); it prints
                 one "x y w h" line per text line; a non-zero exit means "rejected"
"""
import importlib
import importlib.util
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (ROOT, ROOT / "synth"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def overlay(rgb):
    from common import find_lines
    return find_lines(rgb, "legacy")


def gradient(rgb):
    g = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).astype(np.float32)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return []
    e = np.abs(np.diff(g, axis=1))[:-1, :] + np.abs(np.diff(g, axis=0))[:, :-1]
    prof = np.convolve(e.sum(1), np.ones(3) / 3, mode="same")
    if prof.max() <= 0:
        return []
    on = prof > 0.25 * prof.max()
    runs, start = [], None
    for y, v in enumerate(on):
        if v and start is None:
            start = y
        if not v and start is not None:
            runs.append([start, y])
            start = None
    if start is not None:
        runs.append([start, len(on)])
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < 3:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    runs = [r for r in merged if r[1] - r[0] >= 8]
    if runs:
        med = float(np.median([r[1] - r[0] for r in runs]))
        runs = [r for r in runs if r[1] - r[0] >= 0.55 * med]
    H, W = rgb.shape[:2]
    out = []
    for a, b in runs:
        y0, y1 = max(0, a - 4), min(H, b + 6)
        out.append((0, y0, W, y1 - y0))
    return out


def _py(spec):
    mod, func = spec.rsplit(":", 1)
    if mod.endswith(".py"):
        s = importlib.util.spec_from_file_location(Path(mod).stem, mod)
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
    else:
        m = importlib.import_module(mod)
    return getattr(m, func)


def _cmd(command):
    from backends import write_crop

    def run(rgb):
        with tempfile.TemporaryDirectory(prefix="vocr_find_") as tmp:
            p = Path(tmp) / "region.ppm"
            write_crop(p, rgb, "ppm")
            cmd = command.replace("{}", shlex.quote(str(p))) if "{}" in command \
                else f"{command} {shlex.quote(str(p))}"
            r = subprocess.run(cmd, shell=True, capture_output=True, timeout=120, stdin=subprocess.DEVNULL)
            if r.returncode:
                return None
            boxes = []
            for line in r.stdout.decode().splitlines():
                if line.strip():
                    x, y, w, h = (int(v) for v in line.split()[:4])
                    boxes.append((x, y, w, h))
            return boxes
    return run


def make_finder(spec):
    if spec == "overlay":
        return overlay
    if spec == "gradient":
        return gradient
    if spec.startswith("py:"):
        return _py(spec[3:])
    if spec.startswith("cmd:"):
        return _cmd(spec[4:])
    raise ValueError(f"unknown --finder {spec!r}")


def overlap(det, gt):
    """Vertical overlap of two (x, y, w, h) boxes as a share of the smaller height,
    zero unless they also overlap horizontally."""
    dx0, dy0, dw, dh = det
    gx0, gy0, gw, gh = gt
    if min(dx0 + dw, gx0 + gw) <= max(dx0, gx0):
        return 0.0
    inter = min(dy0 + dh, gy0 + gh) - max(dy0, gy0)
    return max(0.0, inter / max(1, min(dh, gh)))


def match(dets, gts, thresh=0.5):
    """One-to-one greedy matching by overlap(); a detection that also covers a
    second ground-truth line by more than half of that line's height is a merge,
    and only its best line counts as found. Returns (pairs, unmatched dets)."""
    cand = sorted(((overlap(d, g["box_r"]), i, j) for i, d in enumerate(dets)
                   for j, g in enumerate(gts)), reverse=True)
    used_d, used_g, pairs = set(), set(), []
    for ov, i, j in cand:
        if ov < thresh or i in used_d or j in used_g:
            continue
        used_d.add(i)
        used_g.add(j)
        pairs.append((i, j))
    return pairs, [i for i in range(len(dets)) if i not in used_d]
