#!/usr/bin/env python3
"""Train textspot-desktop's line detector (issue #173, eval/textspot_desktop_protocol.json, docs/textspot-desktop.md):
the speed tier beside vocr-desktop's detector (#169). The network is train/tsdesk_net.py's family; the windows, data
sources, augmentation and schedule are vocr-desktop's (train/desktop_det.py), so the two differ in the network, the
head's stride, the along-line edge maps and the recipe options below.

  tsdesk_det.py --shape p8a --seed N --out RUN --hook-manifest M --hook-shards DIR [--hook-splits train]
                --manifest M ... --shards DIR ... [--extra NAME=MANIFEST:ROOT[:SPLITS] ...] [--mix ...]
                [--recipe base|junk] [--skip-vns FILE] [--steps 40000] [--threads 4] [--workers 1] [--probe N]
  tsdesk_det.py export --run RUN --out DIR [--name NAME] [--size fit:960]

Targets (targets() below; the head's stride s = the shape's hs, 8 or 4): textspot6_train.targets' kernel, orientation
and across distances at stride s, textspot8's edge-kernel weights (the K weight of a line's first and last kernel
column x 4, only for an end inside the window), and the along distances E1 = (centre - start) and E2 = (end - centre)
on the kernel cells whose centre lies within max(2 s, 1.5 x the line's size across) of that end (an end inside the
window only). Loss: textspot2_train.loss_of (K BCE + dice, O BCE, A smooth L1 of (A - target) / 8), + the E terms
(smooth L1 of (E - target) / 8 over each end's cells, times the line weight, normalised by their count), + textspot6's
tag loss.

Recipes: base = the sources as built; junk = on the hooked sources (hook, desk_hook) the weak MeikiOCR boxes (score
< 0.5, meeting no renderer box) that synth/textspot6_pages.py made ignore areas are background (a page's last
counts.meiki_ignore ignore entries; a desk_hook page only when its ignore count equals meta.src_ignore).

--skip-vns FILE: VNs (one per line) whose frames leave the real half (textspot2's FramePool skip) and every --extra
pool whose samples carry a vn (the held-out dense sets, eval/tsdesk_dense.py).

Outputs (RUN must not exist or be empty): run.json, log.jsonl, final.pt (state dict, shape), final.onnx, done.json.
"""
import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "train", ROOT / "synth", ROOT / "tools", ROOT):
    if str(_p) not in sys.path:
        sys.path.append(str(_p)) if _p == ROOT else sys.path.insert(0, str(_p))

import desktop_det as DD  # noqa: E402
import tsdesk_net as N  # noqa: E402

EDGE_CELLS = 1
EDGE_WEIGHT = 4.0
E_WIN_CELLS = 2.0
E_WIN_ACROSS = 1.5
E_WEIGHT = 1.0
DECODE = {"min_cells": 2, "conf_min_logit": 0.0, "min_across": 4.0, "along_reach": 3.0, "max_boxes": 256}
DEFAULT_SIZE = "fit:960"
ATTRIBUTION = ("textspot-desktop line detector, bee-san/vita-vn-ocr (GPL-3.0-or-later). Trained on textspot's "
               "synthetic frames, MeikiOCR's detector boxes on VNDB and owner screenshots, desktop-data's pages and "
               "the owner's private hooked PS Vita / PSP captures (training input only): for non-commercial use only. "
               "See TEXTSPOT-DESKTOP-ATTRIBUTION.md.")


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- targets

def targets(lines, ignore, w, h, s, clip=False):
    """(K, O, A1, A2, E1, E2, valid, kernel, e1_mask, e2_mask, W, M) at stride s on the WIN_H x WIN_W window (lines
    [[x0, y0, x1, y1, vertical, weight, mask]] in window px, the valid region w x h at the top left); the module
    docstring has the rule."""
    B, _, _, _ = DD._setup()
    ho, wo = B.WIN_H // s, B.WIN_W // s
    K = np.zeros((ho, wo), np.float32)
    O = np.zeros((ho, wo), np.float32)  # noqa: E741
    A1 = np.zeros((ho, wo), np.float32)
    A2 = np.zeros((ho, wo), np.float32)
    E1 = np.zeros((ho, wo), np.float32)
    E2 = np.zeros((ho, wo), np.float32)
    m1 = np.zeros((ho, wo), bool)
    m2 = np.zeros((ho, wo), bool)
    Wt = np.ones((ho, wo), np.float32)
    Mk = np.zeros((ho, wo), np.int64)
    owner = np.full((ho, wo), -1, np.int32)
    conflict = np.zeros((ho, wo), bool)
    valid = np.zeros((ho, wo), bool)
    vh, vw = (h + s - 1) // s, (w + s - 1) // s
    valid[:vh, :vw] = True
    ign = np.zeros((ho, wo), bool)

    def block_mark(m, b):
        x0, y0, x1, y1 = b
        if clip:        # --ignore-clip: a box is cut to the window first; one outside it marks nothing
            x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
        if x1 <= x0 or y1 <= y0:
            return
        # without the clip, a box ending s px or more above or left of the window gives a negative slice end, which
        # numpy reads from the far side: most of the window's rows or columns become ignore (--ignore-clip fixes it)
        m[max(0, y0 // s):min(ho, (y1 - 1) // s + 1), max(0, x0 // s):min(wo, (x1 - 1) // s + 1)] = True

    for b in ignore:
        block_mark(ign, b)
    centres = np.arange(max(ho, wo)) * s + (s - 1) / 2
    edge_w = []
    for n, ln in enumerate(lines):
        x0, y0, x1, y1 = ln[:4]
        vert = bool(ln[4])
        a0, a1, b0, b1 = (x0, x1, y0, y1) if vert else (y0, y1, x0, x1)
        lim_a = w if vert else h
        lim_b = h if vert else w
        if a1 <= 0 or a0 >= lim_a or b1 <= 0 or b0 >= lim_b:
            continue
        if a0 < 0 or a1 > lim_a:
            block_mark(ign, (max(0, x0), max(0, y0), min(w, x1), min(h, y1)))
            continue
        real_start, real_end = b0 >= 0, b1 <= lim_b
        bs, be = b0, b1
        b0, b1 = max(0, b0), min(lim_b, b1)
        across = a1 - a0
        c = (a0 + a1) / 2
        r = max(across / 4, s / 2)
        n_across = ho if not vert else wo
        idx = np.flatnonzero(np.abs(centres[:n_across] - c) <= r)
        if not len(idx):
            idx = np.array([int(np.clip(round((c - (s - 1) / 2) / s), 0, n_across - 1))])
        along = np.arange(b0 // s, (b1 - 1) // s + 1)
        along = along[along < (wo if not vert else ho)]
        if not len(along):
            continue
        if vert:
            ii, jj = np.meshgrid(along, idx, indexing="ij")
        else:
            ii, jj = np.meshgrid(idx, along, indexing="ij")
        ii, jj = ii.ravel(), jj.ravel()
        cc = centres[jj] if vert else centres[ii]
        ca = centres[ii] if vert else centres[jj]
        taken = owner[ii, jj] >= 0
        conflict[ii[taken], jj[taken]] = True
        owner[ii, jj] = n
        K[ii, jj] = 1
        O[ii, jj] = float(vert)
        A1[ii, jj] = cc - a0
        A2[ii, jj] = a1 - cc
        Wt[ii, jj] = float(ln[5])
        Mk[ii, jj] = int(ln[6])
        reach = max(E_WIN_CELLS * s, E_WIN_ACROSS * across)
        E1[ii, jj] = ca - bs
        E2[ii, jj] = be - ca
        if real_start:
            sel = (ca - bs) <= reach
            m1[ii[sel], jj[sel]] = True
        if real_end:
            sel = (be - ca) <= reach
            m2[ii[sel], jj[sel]] = True
        ends = set()
        if real_start:
            ends |= set(along[:EDGE_CELLS].tolist())
        if real_end:
            ends |= set(along[-EDGE_CELLS:].tolist())
        if ends:
            sel = np.isin(ii if vert else jj, list(ends))
            edge_w.append((ii[sel], jj[sel], n))
    kern = (K > 0) & ~conflict & valid
    valid_k = valid & ~conflict & (~ign | kern)
    for ei, ej, n in edge_w:                 # the end cells a line still owns
        own = (owner[ei, ej] == n) & kern[ei, ej]
        Wt[ei[own], ej[own]] *= EDGE_WEIGHT
    K[conflict] = 0
    Wt[~kern] = 1.0
    Mk[~kern] = 0
    Mk[Mk == B.ALL] = 0
    m1 &= kern
    m2 &= kern
    return K, O, A1, A2, E1, E2, valid_k, kern, m1, m2, Wt, Mk


# ----------------------------------------------------------------------------- recipes and pools

EXACT_SOURCES_DEFAULT = ("hook", "desk_hook", "synth", "desk_synth", "desk_ui")   # ink boxes from renderer geometry
#                                                                                  or rendering (desk_ui: ui_synth)

def hooked_as(src):
    """The hooked source a source name reads as: "hook", "desk_hook" (also an extra source named desk_hook_*: a
    subset of 1a's pages, such as train/tsdesk_hard_source.py's desk_hook_hard, which gets 1a's target patch, label
    edits, junk recipe and exact ends), or None."""
    if src in ("hook", "desk_hook"):
        return src
    return "desk_hook" if str(src).startswith("desk_hook_") else None


def exact_source(src, sources=EXACT_SOURCES_DEFAULT):
    """True when a source's boxes are exact ink boxes (a name in sources, a desk_hook_* source when desk_hook is, or a
    desk_ui_* source when desk_ui is: a later desk_ui version, desktopdata/ui_synth.py --version N, such as
    desk_ui_v2)."""
    return src in sources or hooked_as(src) in sources or (str(src).startswith("desk_ui_") and "desk_ui" in sources)


def strip_weak(smp, src):
    """The junk recipe: a hooked page's weak MeikiOCR ignore areas dropped (they become background)."""
    src = hooked_as(src)
    if src is None:
        return smp
    meta = smp.get("meta") or {}
    c = meta.get("counts") or {}
    n = int(c.get("meiki_ignore") or 0)
    if n <= 0:
        return smp
    ign = smp["ignore"]
    if src == "hook":
        total = sum(int(c.get(k) or 0) for k in ("ruby_ignore", "drop_ignore", "uncovered_ignore", "meiki_ignore"))
        if total != len(ign):
            return smp
    elif len(ign) != int(meta.get("src_ignore", -1)):
        return smp
    return dict(smp, ignore=ign[:len(ign) - n])


class SkipPool(DD.ExtraPool):
    """An --extra pool without the samples of the skipped VNs."""

    def __init__(self, manifest, root, splits=None, skip_vns=frozenset()):
        super().__init__(manifest, root, splits)
        before = len(self.items)
        self.items = [(d, s) for d, s in self.items if s.get("vn") not in skip_vns]
        self.skipped = before - len(self.items)
        if not self.items:
            raise SystemExit(f"{manifest}: every sample skipped")


def add_extra_sources(sources, mix, extra, skip_vns):
    rec = {}
    for name, (man, root, splits) in extra.items():
        if name not in mix:
            raise SystemExit(f"--extra {name}: no weight for it in --mix")
        pool = SkipPool(man, root, splits, skip_vns)
        if not hasattr(sources, "extra_pools"):
            sources.extra_pools = {}
        sources.extra_pools[name] = pool
        sources.names.append(name)
        sources.p = np.append(np.asarray(sources.p, np.float64) * 1.0, float(mix[name]))
        sources.draw[name] = pool.draw
        rec[name] = {"samples": len(pool.items), "skipped_by_vn": pool.skipped}
    p = np.asarray(sources.p, np.float64)
    sources.p = p / p.sum()
    return rec


def make_sources(c):
    B, _, _, ST = DD._setup()
    hook = B.HookPool(c["hook_manifest"], c["hook_shards"], c["hook_splits"]) if c["mix"].get("hook") else None
    frames = ST.FramePool(c["manifests"], c["shards"], skip=c["skip"]) if c["manifests"] else None
    base_mix = {k: v for k, v in c["mix"].items() if k in ("hook", "synth", "real")}
    sources = B.Sources(base_mix, hook, frames, c["title_temp"])
    sources.p = np.asarray(sources.p) * sum(base_mix.values())
    extra = add_extra_sources(sources, c["mix"], c.get("extra", {}), frozenset(c.get("skip_vns", ())))
    if c.get("targets"):
        extra["targets"] = patch_pools(sources, hook, c["targets"])
    if c.get("label_edits"):
        extra["label_edits"] = edit_pools(sources, hook, c["label_edits"])
    return sources, hook, frames, extra


EXACT_SOURCES = EXACT_SOURCES_DEFAULT


ORN = {"near_p": 0.7, "above_p": 0.6, "size": [0.6, 1.1], "gap": [0.15, 1.0], "pitch": [1.15, 2.4],
       "length": [0.5, 1.3], "alt_p": 0.3, "kinds": ("star4", "diamond", "ring", "spinner_dots", "hourglass",
                                                      "cursor_down_outline", "flower", "cross")}


def _draw_motif(img, kind, cx, cy, size, value, rng):
    import cv2

    import textspot8_train as T8M
    r = max(2.0, size / 2.0)
    if kind == "flower":
        for k in range(5):
            a = 2 * math.pi * k / 5
            cv2.circle(img, (int(round(cx + 0.55 * r * math.cos(a))), int(round(cy + 0.55 * r * math.sin(a)))),
                       max(1, int(round(0.35 * r))), value, -1, lineType=cv2.LINE_AA)
    elif kind == "cross":
        t = max(1, int(round(size / 8)))
        cv2.line(img, (int(cx - 0.7 * r), int(cy)), (int(cx + 0.7 * r), int(cy)), value, t, lineType=cv2.LINE_AA)
        cv2.line(img, (int(cx), int(cy - 0.7 * r)), (int(cx), int(cy + 0.7 * r)), value, t, lineType=cv2.LINE_AA)
    else:
        T8M.draw_icon(img, kind, cx, cy, size, value, rng)


def add_ornaments(img, pos, ign, rng):
    """One row of repeated identical (or two alternating) small ornaments drawn as background on the window (in
    place): above or below a horizontal positive, about as long as it (p near_p), or on free background; never over a
    positive or an ignore area. Text boxes framed by strips of repeated ornaments (PCSG00854's) are not text lines.
    Returns the row's box or None."""
    h, w = img.shape
    taken = [b[:4] for b in pos] + [b[:4] for b in ign]
    hz = [b for b in pos if not b[4]]
    if hz and rng.random() < ORN["near_p"]:
        b = hz[int(rng.integers(len(hz)))]
        lh = max(6, b[3] - b[1])
        size = lh * rng.uniform(*ORN["size"])
        gap = lh * rng.uniform(*ORN["gap"])
        cy = b[1] - gap - size / 2 if rng.random() < ORN["above_p"] else b[3] + gap + size / 2
        x0 = b[0] + lh * rng.uniform(-0.6, 0.6)
        length = (b[2] - b[0]) * rng.uniform(*ORN["length"])
    else:
        size = rng.uniform(8, 28)
        cy = rng.uniform(size, max(size + 1, h - size))
        x0 = rng.uniform(0, w / 2)
        length = size * rng.uniform(6, 40)
    pitch = size * rng.uniform(*ORN["pitch"])
    n = int(length // pitch)
    if n < 4:
        return None
    box = [int(x0 - size / 2) - 1, int(cy - size / 2) - 1, int(x0 + (n - 1) * pitch + size / 2) + 2,
           int(cy + size / 2) + 2]
    box = [max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])]
    if box[2] - box[0] < 2 * size or box[3] - box[1] < 4:
        return None
    import textspot8_train as T8M
    if not T8M._free(box, taken):
        return None
    kinds = ORN["kinds"]
    k1 = kinds[int(rng.integers(len(kinds)))]
    k2 = kinds[int(rng.integers(len(kinds)))] if rng.random() < ORN["alt_p"] else k1
    patch = img[box[1]:box[3], box[0]:box[2]]
    bg = float(np.median(patch)) if patch.size else 128.0
    value = (235 if bg < 128 else 20) if rng.random() < 0.7 else int(rng.integers(0, 256))
    for i in range(n):
        cx = x0 + i * pitch
        if cx + size / 2 > w:
            break
        _draw_motif(img, k1 if i % 2 == 0 else k2, cx, cy, size, value, rng)
    return box


MARKS = {"max": 2, "before_p": 0.7, "size": [0.45, 0.9], "gap": [0.15, 0.9], "between_gap": 1.5, "jitter": 0.08,
         "icon_p": 0.25, "blur_p": 0.5, "blur": [0.3, 0.8], "min_line": 6, "flat": 0.35, "flat_floor": 24.0,
         "glyphs": ("bullet", "diamond", "square", "triangle")}
# --marks-glyphs: v2 (the default) draws no ・ or ※: ・ is text the dense labels keep (a list line starts with it), so
# marks-trained models learnt to drop it; v1 is the first set, for the runs made with it (bit for bit)
MARK_GLYPHS = {"v1": ("nakaguro", "bullet", "diamond", "square", "triangle", "kome"),
               "v2": ("bullet", "diamond", "square", "triangle")}


def _mark_extent(kind, s):
    """(left, up, right, down): a mark's ink around its centre, px, for a mark of size s (_draw_mark)."""
    e = {"nakaguro": max(1.0, 0.11 * s), "bullet": 0.36 * s, "diamond": 0.42 * s, "square": 0.34 * s,
         "kome": 0.45 * s}.get(kind)
    if kind == "triangle":
        return 0.3 * s, 0.36 * s, 0.36 * s, 0.36 * s
    if e is None:                                   # a T8 icon (textspot8_train.draw_icon): within half its size
        e = 0.5 * s
    return e, e, e, e


def _draw_mark(img, kind, cx, cy, s, value, rng):
    """One single mark on the uint8 window img (in place): a glyph mark (・ ● ◆ ■ ▶ ※) or a T8 icon."""
    import cv2

    import textspot8_train as T8M
    v = int(value)
    c = (int(round(cx)), int(round(cy)))

    def poly(pts):
        cv2.fillPoly(img, [np.round(np.asarray(pts)).astype(np.int32)], v, lineType=cv2.LINE_AA)
    if kind in ("nakaguro", "bullet"):
        cv2.circle(img, c, max(1, int(round(_mark_extent(kind, s)[0]))), v, -1, lineType=cv2.LINE_AA)
    elif kind == "diamond":
        e = 0.42 * s
        poly([[cx, cy - e], [cx + e, cy], [cx, cy + e], [cx - e, cy]])
    elif kind == "square":
        e = 0.34 * s
        poly([[cx - e, cy - e], [cx + e, cy - e], [cx + e, cy + e], [cx - e, cy + e]])
    elif kind == "triangle":
        poly([[cx - 0.3 * s, cy - 0.36 * s], [cx + 0.36 * s, cy], [cx - 0.3 * s, cy + 0.36 * s]])
    elif kind == "kome":
        e, t = 0.3 * s, max(1, int(round(s / 10)))
        cv2.line(img, (int(cx - e), int(cy - e)), (int(cx + e), int(cy + e)), v, t, lineType=cv2.LINE_AA)
        cv2.line(img, (int(cx - e), int(cy + e)), (int(cx + e), int(cy - e)), v, t, lineType=cv2.LINE_AA)
        for dx, dy in ((0.38, 0), (-0.38, 0), (0, 0.38), (0, -0.38)):
            cv2.circle(img, (int(round(cx + dx * s)), int(round(cy + dy * s))), max(1, int(round(0.07 * s))), v, -1,
                       lineType=cv2.LINE_AA)
    else:
        T8M.draw_icon(img, kind, cx, cy, s, value, rng)


def add_marks(img, pos, ign, rng, taken=(), glyphs=None):
    """1..MARKS["max"] single marks drawn as background on the window (in place): a glyph mark (・ ● ◆ ■ ▶ ※) or
    (p icon_p) a T8 icon, either before a horizontal positive's start (its right edge gap x the line's height before
    the start, on the line's row) or (p 1 - before_p, when the window has two positives side by side on one row at
    least between_gap line heights apart) in the middle of the gap between them; never over a positive, an ignore area
    or a box of `taken`. A bullet before a choice or a list line, or a mark between two buttons, is not text: round
    1's dense-page boxes took them in. A mark goes only on flat background: the 5-95% grey range under its box at most
    max(flat_floor, flat x the line's own range), so it never covers ink no label holds (a glyph a source's box missed,
    a texture). Returns the boxes drawn [x0, y0, x1, y1]."""
    import cv2

    import textspot8_train as T8M
    h, w = img.shape
    M = MARKS
    hz = [b for b in pos if not b[4] and b[3] - b[1] >= M["min_line"]]
    if not hz:
        return []
    pairs = []
    for i, a in enumerate(hz):
        for j, b in enumerate(hz):
            if i == j or b[0] <= a[2]:
                continue
            lh = min(a[3] - a[1], b[3] - b[1])
            if min(a[3], b[3]) - max(a[1], b[1]) >= 0.5 * lh and b[0] - a[2] >= M["between_gap"] * lh:
                pairs.append((a, b))
    occupied = [b[:4] for b in pos] + [b[:4] for b in ign] + [list(t[:4]) for t in taken]
    drawn = []
    for _ in range(int(rng.integers(1, int(M["max"]) + 1))):
        icon = rng.random() < M["icon_p"]
        kinds = T8M.ICON_KINDS if icon else (glyphs or M["glyphs"])
        kind = kinds[int(rng.integers(len(kinds)))]
        if pairs and rng.random() >= M["before_p"]:
            a, b = pairs[int(rng.integers(len(pairs)))]
            p_row = a
            lh = min(a[3] - a[1], b[3] - b[1])
            s = min(lh * rng.uniform(*M["size"]), 0.5 * (b[0] - a[2]))
            cx = (a[2] + b[0]) / 2.0
            cy = (max(a[1], b[1]) + min(a[3], b[3])) / 2.0 + lh * rng.uniform(-M["jitter"], M["jitter"])
        else:
            p = p_row = hz[int(rng.integers(len(hz)))]
            lh = p[3] - p[1]
            s = lh * rng.uniform(*M["size"])
            cx = p[0] - lh * rng.uniform(*M["gap"]) - _mark_extent(kind, s)[2]
            cy = (p[1] + p[3]) / 2.0 + lh * rng.uniform(-M["jitter"], M["jitter"])
        el, eu, er, ed = _mark_extent(kind, s)
        box = [int(math.floor(cx - el)) - 1, int(math.floor(cy - eu)) - 1, int(math.ceil(cx + er)) + 2,
               int(math.ceil(cy + ed)) + 2]
        if box[0] < 0 or box[1] < 0 or box[2] > w or box[3] > h or not T8M._free(box, occupied + drawn):
            continue
        patch = img[box[1]:box[3], box[0]:box[2]]
        ref = img[max(0, int(p_row[1])):int(p_row[3]), max(0, int(p_row[0])):int(p_row[2])]
        contrast = float(np.percentile(ref, 95) - np.percentile(ref, 5)) if ref.size else 0.0
        if patch.size == 0 or float(np.percentile(patch, 95) - np.percentile(patch, 5)) > max(
                M["flat_floor"], M["flat"] * contrast):
            continue                                # never over ink, a glyph the source's box missed or a texture
        bg = float(np.median(patch))
        value = (235 if bg < 128 else 20) if rng.random() < 0.7 else int(rng.integers(0, 256))
        _draw_mark(img, kind, cx, cy, s, value, rng)
        if rng.random() < M["blur_p"]:
            q = np.ascontiguousarray(img[box[1]:box[3], box[0]:box[2]])
            img[box[1]:box[3], box[0]:box[2]] = cv2.GaussianBlur(q, (0, 0), rng.uniform(*M["blur"]))
        drawn.append(box)
    return drawn


def trim_bands(bands, marks):
    """The end bands (end_bands) without the columns of the marks drawn in them: a band a mark lies in keeps the parts
    beside the mark (between the mark and the line's end too: a thin end glyph the source's box missed stays an
    ignore area), so the mark's own cells are background."""
    out = []
    for g in bands:
        parts = [list(g)]
        for m in marks:
            nxt = []
            for p in parts:
                if m[0] < p[2] and p[0] < m[2] and m[1] < p[3] and p[1] < m[3]:
                    nxt += [q for q in ([p[0], p[1], m[0], p[3]], [m[2], p[1], p[2], p[3]]) if q[2] > q[0]]
                else:
                    nxt.append(p)
            parts = nxt
        out += parts
    return out


def end_bands(pos, f, gap=0.0):
    """Ignore areas just beyond both ends of every positive (window px): f x the line's size across past each end,
    the line's own extent across (a source whose ends come from teacher/geometry.refine may stop before a thin end
    glyph: there, the cells past its end are neither kernel nor background). Kernel cells keep their targets.
    gap (--end-ignore-gap G; 0 = off, the bands as they were): a band never covers more than G of the gap to the
    nearest other positive past that end on the same row (vertical overlap >= half the smaller height; for a vertical
    line, the same column), so the rest of the gap between two items on a row is background; a neighbour that touches
    or overlaps that end leaves no band there."""
    if not gap or gap <= 0:
        out = []
        for x0, y0, x1, y1, vertical, *_ in pos:
            if vertical:
                d = f * (x1 - x0)
                out += [[x0, int(y0 - d), x1, y0], [x0, y1, x1, int(y1 + d + 0.999)]]
            else:
                d = f * (y1 - y0)
                out += [[int(x0 - d), y0, x0, y1], [x1, y0, int(x1 + d + 0.999), y1]]
        return out
    out = []
    for i, p in enumerate(pos):
        x0, y0, x1, y1, vertical = p[:5]
        # along (a0, a1) and across (c0, c1) extents of p and of the others, in p's own orientation
        a0, a1, c0, c1 = (y0, y1, x0, x1) if vertical else (x0, x1, y0, y1)
        d = f * (c1 - c0)
        before, after = d, d
        for j, q in enumerate(pos):
            if j == i:
                continue
            b0, b1, e0, e1 = (q[1], q[3], q[0], q[2]) if vertical else (q[0], q[2], q[1], q[3])
            if min(c1, e1) - max(c0, e0) < 0.5 * min(c1 - c0, e1 - e0):
                continue                            # not on p's row (column)
            if b0 > a0 and b1 > a1:                 # past p's end
                after = min(after, max(0.0, gap * (b0 - a1)))
            elif b1 < a1 and b0 < a0:               # before p's start
                before = min(before, max(0.0, gap * (a0 - b1)))
        # a band the cap does not shorten is rounded as before; a capped one inward, so it stays within G of the gap
        s0 = int(a0 - d) if before >= d else int(math.ceil(a0 - before))
        s1 = int(a1 + d + 0.999) if after >= d else int(math.floor(a1 + after))
        if vertical:
            bands = ([x0, s0, x1, y0], [x0, y1, x1, s1])
        else:
            bands = ([s0, y0, x0, y1], [x1, y0, s1, y1])
        out += [b for b in bands if b[2] > b[0] and b[3] > b[1]]
    return out


def patch_pools(sources, hook, targets_dir):
    """Amendment 1: the hooked sources' verified positives get the target cache's ink boxes (train/tsdesk_targets.py;
    a sample whose stored boxes do not match the cache's refined_line boxes keeps its own)."""
    import tsdesk_targets as TT
    cache, doc = TT.load_cache(targets_dir)
    rec = {"targets": str(targets_dir), "targets_sha256": TT._sha(Path(targets_dir) / "targets.json")}
    if hook is not None:
        n_ok = n_bad = 0
        for t, lst in hook.by_title.items():
            for i, (d, smp) in enumerate(lst):
                ps = TT.patch_sample(smp, cache, "hook")
                if ps is None:
                    n_bad += 1
                else:
                    lst[i] = (d, ps)
                    n_ok += 1
        rec["hook"] = {"patched": n_ok, "kept": n_bad}
    for name, pool in sorted(getattr(sources, "extra_pools", {}).items()):
        if hooked_as(name) != "desk_hook":
            continue
        n_ok = n_bad = 0
        for i, (d, smp) in enumerate(pool.items):
            ps = TT.patch_sample(smp, cache, "desk_hook")
            if ps is None:
                n_bad += 1
            else:
                pool.items[i] = (d, ps)
                n_ok += 1
        rec[name] = {"patched": n_ok, "kept": n_bad}
    return rec


PC_EDIT_SOURCES = ("desk_realplus", "desk_dense", "desk_hard")    # the extra sources made of 1b's frames


def pc_source(name):
    """Whether an extra source is made of 1b's frames (a PC label-edit cache applies to it): desk_realplus,
    desk_dense, desk_hard or one of them with a suffix (desk_hard_v2)."""
    return any(name == p or name.startswith(p + "_") for p in PC_EDIT_SOURCES)


def edit_pools(sources, hook, edits_dirs):
    """--label-edits: label-edit caches (train/tsdesk_coverage.py), applied after the target patch; the samples' other
    lines and ignore areas are untouched. A hooked cache (edits) appends its notdlg positives (MeikiOCR's 'apart' text
    boxes, native px) to the hook and desk_hook (desk_hook_*) samples, desk_hook's mapped through each capture; a PC
    cache (pcedits, scope "pc") appends its positives to the samples of the extra sources made of 1b's frames
    (pc_source: desk_realplus, desk_dense, desk_hard) by screenshot, when the sample's size is the cache's."""
    import tsdesk_coverage as TC
    if isinstance(edits_dirs, (str, Path)):
        edits_dirs = [edits_dirs]
    recs = []
    for edits_dir in edits_dirs:
        doc = json.loads((Path(edits_dir) / "edits.json").read_text(encoding="utf-8"))
        rec = {"label_edits": str(edits_dir), "edits_sha256": sha256_file(Path(edits_dir) / "edits.json")}
        extra = sorted(getattr(sources, "extra_pools", {}).items())
        if doc.get("scope") == "pc":
            rec["scope"] = "pc"
            edits, _ = TC.load_pc_edits(edits_dir)
            for name, pool in extra:
                if not pc_source(name):
                    continue
                n_s = n_l = 0
                for i, (d, smp) in enumerate(pool.items):
                    es = TC.apply_pc_edits(smp, edits)
                    if es is not None:
                        n_s += 1
                        n_l += len(es["lines"]) - len(smp["lines"])
                        pool.items[i] = (d, es)
                rec[name] = {"samples_edited": n_s, "lines_added": n_l}
            recs.append(rec)
            continue
        edits, _ = TC.load_edits(edits_dir)
        pools = [("hook", "hook", [lst for lst in hook.by_title.values()] if hook is not None else [])]
        for name, pool in extra:
            if hooked_as(name) == "desk_hook":
                pools.append((name, "desk_hook", [pool.items]))
        for name, src, lists in pools:
            n_s = n_l = 0
            for lst in lists:
                for i, (d, smp) in enumerate(lst):
                    es = TC.apply_edits(smp, edits, src)
                    if es is not None:
                        n_s += 1
                        n_l += len(es["lines"]) - len(smp["lines"])
                        lst[i] = (d, es)
            rec[name] = {"samples_edited": n_s, "lines_added": n_l}
        recs.append(rec)
    return recs[0] if len(recs) == 1 else recs


def make_batch(sources, rng, cfg):
    B, _, _, _ = DD._setup()
    s = int(cfg["stride"])
    WH, WW = B.WIN_H, B.WIN_W
    bsz = cfg["batch"]
    x = np.zeros((bsz, 1, WH, WW), np.uint8)
    ho, wo = WH // s, WW // s
    tg = [np.zeros((bsz, ho, wo), np.float32) for _ in range(7)]
    bm = [np.zeros((bsz, ho, wo), bool) for _ in range(4)]
    mk = np.zeros((bsz, ho, wo), np.int64)
    e_src = cfg.get("e_sources")
    end_ig = float(cfg.get("end_ignore") or 0.0)
    end_gap = float(cfg.get("end_ignore_gap") or 0.0)
    ign_clip = bool(cfg.get("ignore_clip"))
    orn_p = float(cfg.get("ornaments") or 0.0)
    marks_p = float(cfg.get("marks") or 0.0)
    mark_glyphs = MARK_GLYPHS[cfg.get("marks_glyphs") or "v2"]
    for b in range(bsz):
        src, canvas, smp = sources.pick(rng)
        if cfg.get("recipe") == "junk":
            smp = strip_weak(smp, src)
        img, pos, ign = DD.make_window_ms(canvas, smp, src, rng, sources, cfg["scale"], cfg.get("aug", True))
        h, w = img.shape
        x[b, 0] = np.pad(img, ((0, WH - h), (0, WW - w)), mode="edge")
        ign0, bands = ign, []
        if end_ig > 0 and not exact_source(src):
            bands = end_bands(pos, end_ig, end_gap)
            ign = list(ign) + bands
        orn = None
        if orn_p > 0 and cfg.get("aug", True) and rng.random() < orn_p:
            img = np.array(img, copy=True)
            orn = add_ornaments(img, pos, ign, rng)
            if orn is not None:
                x[b, 0] = np.pad(img, ((0, WH - h), (0, WW - w)), mode="edge")
        if marks_p > 0 and cfg.get("aug", True) and rng.random() < marks_p:
            img = np.array(img, copy=True)
            drawn = add_marks(img, pos, ign0, rng, [orn] if orn is not None else [], mark_glyphs)
            if drawn:
                x[b, 0] = np.pad(img, ((0, WH - h), (0, WW - w)), mode="edge")
                if bands:
                    ign = list(ign0) + trim_bands(bands, drawn)
        K, O, A1, A2, E1, E2, v, k, m1, m2, Wt, M = targets(pos, ign, w, h, s, ign_clip)  # noqa: E741
        if e_src is not None and not exact_source(src, e_src):
            m1[:] = False
            m2[:] = False
        for t, a in zip(tg, (K, O, A1, A2, E1, E2, Wt)):
            t[b] = a
        for t, a in zip(bm, (v, k, m1, m2)):
            t[b] = a
        mk[b] = M
    return (torch.from_numpy(x), *(torch.from_numpy(t) for t in tg), *(torch.from_numpy(t) for t in bm),
            torch.from_numpy(mk))


_REMAP_MODULES = ("tsdesk", "desktop_det", "textspot", "desktopdata", "shards")


def remap_memmaps(obj, _seen=None):
    """Re-open every np.memmap that obj holds (in lists, tuples, dicts or the attributes of this project's objects), so
    the pages a data worker has touched are unmapped and its resident set stays bounded (protocol v2 amendment 20: the
    host's memory guard stops a process at about 48 GB; random draws over the memory-mapped shards would otherwise
    make every page resident). The arrays' contents, and so every batch, are unchanged. Returns obj (or its new
    memmap)."""
    if _seen is None:
        _seen = set()
    if isinstance(obj, np.memmap):
        return np.load(obj.filename, mmap_mode="r") if obj.filename else obj
    if id(obj) in _seen:
        return obj
    _seen.add(id(obj))
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            obj[i] = remap_memmaps(v, _seen)
    elif isinstance(obj, dict):
        for k in list(obj):
            obj[k] = remap_memmaps(obj[k], _seen)
    elif isinstance(obj, tuple):
        return tuple(remap_memmaps(v, _seen) for v in obj)
    elif hasattr(obj, "__dict__") and type(obj).__module__.startswith(_REMAP_MODULES):
        for k, v in list(vars(obj).items()):
            nv = remap_memmaps(v, _seen)
            if nv is not v:
                setattr(obj, k, nv)
    return obj


class Batches(torch.utils.data.IterableDataset):
    def __init__(self, cfg, seed):
        super().__init__()
        self.cfg, self.seed = cfg, seed

    def __iter__(self):
        info = torch.utils.data.get_worker_info()
        wid = info.id if info is not None else 0
        sources, _, _, _ = make_sources(self.cfg)
        rng = np.random.default_rng([self.seed, wid])
        every, i = int(self.cfg.get("remap_every") or 0), 0
        while True:
            if every and i and i % every == 0:
                sources = remap_memmaps(sources)
            yield make_batch(sources, rng, self.cfg)
            i += 1


# ----------------------------------------------------------------------------- loss

def e_loss(y, E1, E2, m1, m2, Wt):
    out, parts = 0.0, {}
    for i, (t, m) in enumerate(((E1, m1), (E2, m2))):
        mf = m.float()
        n = mf.sum().clamp(min=1)
        li = (F.smooth_l1_loss((y[:, 4 + i] - t) / 8, torch.zeros_like(t), reduction="none") * mf * Wt).sum() / n
        out = out + li
        parts[f"e{i + 1}"] = float(li)
    return out, parts


def loss_all(y, batch):
    B, T2, _, _ = DD._setup()
    _, K, O, A1, A2, E1, E2, Wt, vm, km, m1, m2, Mk = batch  # noqa: E741
    loss, parts = T2.loss_of(y[:, :4], K, O, A1, A2, vm, km, Wt)
    le, eparts = e_loss(y, E1, E2, m1, m2, Wt)
    lt, tparts = B.tag_loss(y[:, 6:], Mk, Wt)
    total = loss + E_WEIGHT * le + B.TAG_WEIGHT * lt
    parts.update(eparts)
    parts.update(tparts)
    parts["loss"] = float(total)
    return total, parts


# ----------------------------------------------------------------------------- export

def init_from(model, run_dir, shape):
    """Load a finished run's final.pt into model (the same shape): its record (path, sha256, steps)."""
    import hashlib
    p = Path(run_dir) / "final.pt"
    ck = torch.load(p, map_location="cpu", weights_only=False)
    if ck.get("kind") != N.KIND or ck.get("shape") != shape:
        raise SystemExit(f"{p}: not a textspot-desktop checkpoint of this shape")
    model.load_state_dict(ck["state_dict"])
    return {"run": str(run_dir), "pt_sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "steps": ck.get("steps"),
            "seed": ck.get("seed")}


def load_run(run_dir):
    ck = torch.load(Path(run_dir) / "final.pt", map_location="cpu", weights_only=False)
    if ck.get("kind") != N.KIND:
        raise SystemExit(f"{run_dir}/final.pt: not a textspot-desktop checkpoint")
    m = N.TsdNet(ck["shape"], int(ck.get("tags", N.TAGS)))
    m.load_state_dict(ck["state_dict"])
    return m.eval(), ck


def cmd_export(argv):
    ap = argparse.ArgumentParser(prog="tsdesk_det.py export")
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="textspot_desktop")
    ap.add_argument("--size", default=DEFAULT_SIZE)
    a = ap.parse_args(argv)
    m, ck = load_run(a.run)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    onnx_path = out / f"{a.name}.onnx"
    worst = N.export_onnx(m, onnx_path, N.onnx_meta(ck["shape"], a.name, a.size, DECODE), doc=ATTRIBUTION)
    rec = {"run": str(a.run), "pt_sha256": sha256_file(Path(a.run) / "final.pt"), "size": a.size,
           "onnx": onnx_path.name, "onnx_sha256": sha256_file(onnx_path), "onnx_bytes": onnx_path.stat().st_size,
           "onnx_vs_torch_max_abs": worst, "params": N.n_params(m), "shape": ck["shape"]}
    try:
        import tsdesk_blob as TB  # the C path's blob (desktop/vocr_tsd.c), when present
        blob = TB.blob_bytes(m, a.size, DECODE)
        (out / f"{a.name}.vtsd").write_bytes(blob)
        rec.update(vtsd=f"{a.name}.vtsd", vtsd_sha256=hashlib.sha256(blob).hexdigest(), vtsd_bytes=len(blob))
    except ImportError:
        pass
    print(json.dumps(rec), flush=True)
    return rec


# ----------------------------------------------------------------------------- the run

def parse(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shape", required=True, choices=sorted(N.SHAPES))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hook-manifest")
    ap.add_argument("--hook-shards")
    ap.add_argument("--hook-splits", default="train")
    ap.add_argument("--manifest", nargs="*", default=[])
    ap.add_argument("--shards", nargs="*", default=[])
    ap.add_argument("--extra", nargs="*", default=[], help="NAME=MANIFEST:ROOT[:SPLITS]: more sources for --mix")
    ap.add_argument("--recipe", default="base", choices=("base", "junk"))
    ap.add_argument("--skip-vns", help="a file of VNs (one per line) left out of the real half and the extra pools")
    ap.add_argument("--targets", help="amendment 1: the ink-box target cache (train/tsdesk_targets.py build) for the "
                                      "hooked sources' verified positives")
    ap.add_argument("--label-edits", nargs="+", help="label-edit caches (train/tsdesk_coverage.py edits: notdlg "
                                                       "positives for the hooked sources; pcedits: for the PC "
                                                       "sources), applied after --targets")
    ap.add_argument("--e-exact", action="store_true", help="amendment 1: the along-end (E) loss only on the sources "
                                                            "with exact ink boxes (EXACT_SOURCES)")
    ap.add_argument("--ornaments", type=float, default=0.0,
                    help="the share of windows that get one row of repeated small ornaments as background "
                         "(add_ornaments)")
    ap.add_argument("--marks", type=float, default=0.0,
                    help="the share of windows that get 1-2 single marks (a bullet glyph or a T8 icon) drawn as "
                         "background before a positive's start or between two side-by-side positives (add_marks); "
                         "the end bands of --end-ignore lose the marks' columns (trim_bands)")
    ap.add_argument("--marks-glyphs", default="v2", choices=sorted(MARK_GLYPHS),
                    help="the glyph marks --marks draws: v2 (the default: ● ◆ ■ ▶, no ・ or ※, which the dense labels "
                         "keep as text) or v1 (・ ● ◆ ■ ▶ ※: the runs made before the option)")
    ap.add_argument("--ignore-clip", action="store_true",
                    help="cut every ignore area to the window before marking its cells (targets: without it, an area "
                         "s px or more above or left of the window marks most of the window as ignore)")
    ap.add_argument("--end-ignore-gap", type=float, default=0.0,
                    help="with --end-ignore: an end band never covers more than G of the gap to the next positive on "
                         "the same row, so the rest of the gap is background (end_bands; 0 = off)")
    ap.add_argument("--end-ignore", type=float, default=0.0,
                    help="the cells within F x a line's size across past each end of a positive of a source not in "
                         "EXACT_SOURCES are ignore areas (end_bands)")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--remap-every", type=int, default=0,
                    help="amendment 20: re-open the data worker's memory-mapped shards every N batches (0: never), "
                         "which keeps its resident set bounded; the batches are unchanged")
    ap.add_argument("--init", help="protocol v2 amendment 7: start from a finished run's final.pt (same shape); the "
                                   "schedule, optimiser and EMA start afresh")
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--mix", default="hook=0.3375,synth=0.15,real=0.2625,desk_hook=0.1,desk_realplus=0.1,"
                                     "desk_synth=0.05")
    ap.add_argument("--scale", default="0.42,1.9")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--probe", type=int, default=0)
    ap.add_argument("--no-aug", action="store_true")
    ap.add_argument("--bn-batches", type=int, default=32)
    ap.add_argument("--ema-start", type=int, default=1000)
    a = ap.parse_args(argv)
    a.mix_d = {k: float(v) for k, v in (kv.split("=") for kv in a.mix.split(","))}
    a.scale_t = tuple(float(v) for v in a.scale.split(","))
    a.name = f"{a.shape}_{a.recipe}_seed{a.seed}"
    return a


def code_sha256():
    files = ["train/tsdesk_det.py", "train/tsdesk_net.py", "train/desktop_det.py", "train/textspot6_train.py",
             "train/textspot8_train.py", "train/textspot2_train.py", "train/textspot_train.py",
             "train/textdet_train.py"]
    return {f: sha256_file(ROOT / f) for f in files if (ROOT / f).exists()}


def main(argv=None):
    a = parse(argv)
    B, T2, T8M, ST = DD._setup()
    import textdet_train as TT
    t_proc = time.time()
    out = Path(a.out)
    if out.exists() and any(out.iterdir()) and not a.probe:
        raise SystemExit(f"{out} is not empty: a run starts from scratch in a fresh directory")
    out.mkdir(parents=True, exist_ok=True)
    shape = N.SHAPES[a.shape]
    skip_vns = frozenset()
    if a.skip_vns:
        skip_vns = frozenset(v.strip() for v in Path(a.skip_vns).read_text(encoding="utf-8").split() if v.strip())
    skip_shots, base_skip_vns, lists = T2.skip_lists(a.manifest) if a.manifest else (frozenset(), frozenset(), [])
    cfg = {"mix": a.mix_d, "remap_every": a.remap_every, "hook_manifest": a.hook_manifest, "hook_shards": a.hook_shards,
           "hook_splits": tuple(a.hook_splits.split(",")), "manifests": list(a.manifest), "shards": list(a.shards),
           "skip": (skip_shots, frozenset(base_skip_vns) | skip_vns), "skip_vns": sorted(skip_vns),
           "title_temp": B.TITLE_TEMP, "scale": a.scale_t, "batch": a.batch, "aug": not a.no_aug,
           "extra": DD.parse_extra(a.extra), "recipe": a.recipe, "stride": int(shape["hs"]),
           "targets": a.targets, "e_sources": list(EXACT_SOURCES) if a.e_exact else None,
           "end_ignore": a.end_ignore, "end_ignore_gap": a.end_ignore_gap, "ignore_clip": a.ignore_clip,
           "ornaments": a.ornaments, "marks": a.marks, "marks_glyphs": a.marks_glyphs, "label_edits": a.label_edits}
    sources, hook, frames, extra_rec = make_sources(cfg)
    pool_rec = {"sources": dict(zip(sources.names, (round(float(q), 6) for q in sources.p))),
                "extra": {k: {"manifest": v[0], "root": v[1], "splits": v[2], "manifest_sha256": sha256_file(v[0]),
                              **extra_rec.get(k, {})} for k, v in cfg["extra"].items()},
                "hook": hook.summary(B.TITLE_TEMP) if hook else None,
                "frames": frames.summary() if frames else None, "text_free_synthetic": len(sources.empty),
                "skip_vns": {"file": a.skip_vns, "sha256": sha256_file(a.skip_vns) if a.skip_vns else None,
                             "vns": len(skip_vns)},
                "real_half_dropped": frames.dropped if frames is not None else None,
                "targets": extra_rec.get("targets"), "label_edits": extra_rec.get("label_edits")}
    skip_rec = ST.skip_record(lists, frames.dropped) if lists and frames is not None else None
    del hook, frames, sources
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed)
    model = N.TsdNet(shape)
    init_rec = init_from(model, a.init, shape) if a.init else None
    opt = torch.optim.AdamW(TT.param_groups(list(model.parameters())), lr=a.lr)
    loader = torch.utils.data.DataLoader(Batches(cfg, a.seed), batch_size=None, num_workers=a.workers,
                                         persistent_workers=a.workers > 0,
                                         prefetch_factor=4 if a.workers > 0 else None)
    it = iter(loader)
    run = {"kind": "textspot-desktop-run", "name": a.name, "started": ST.utc(), "shape": shape,
           "params": N.n_params(model), "macs_960x544": N.macs(shape, 960, 544), "init": init_rec,
           "args": {k: v for k, v in vars(a).items() if k not in ("mix_d", "scale_t")}, "mix": a.mix_d,
           "scale": list(a.scale_t), "window": [B.WIN_H, B.WIN_W], "batch": a.batch, "recipe": a.recipe,
           "targets": {"stride": cfg["stride"], "edge_cells": EDGE_CELLS, "edge_weight": EDGE_WEIGHT,
                       "e_win_cells": E_WIN_CELLS, "e_win_across": E_WIN_ACROSS, "e_weight": E_WEIGHT,
                       "ink_cache": a.targets, "e_sources": cfg["e_sources"], "end_ignore": a.end_ignore,
                       "end_ignore_gap": a.end_ignore_gap or None, "ignore_clip": a.ignore_clip,
                       "ornaments": {"p": a.ornaments, **ORN} if a.ornaments else None,
                       "marks": {"p": a.marks, **MARKS, "glyph_set": a.marks_glyphs,
                                 "glyphs": list(MARK_GLYPHS[a.marks_glyphs]), "icons": list(T8M.ICON_KINDS)}
                       if a.marks else None},
           "aug": B.AUG if not a.no_aug else None, "t8": {k: T8M.T8[k] for k in ("icon_p", "tight_p", "tight_margin")},
           "class_mask": B.CLASS_MASK, "pool": pool_rec, "skip": skip_rec,
           "manifests_sha256": {m: sha256_file(m) for m in [*a.manifest, *([a.hook_manifest] if a.hook_manifest
                                                                           else [])]},
           "code_sha256": code_sha256(), "git": ST.git_state(), "torch": torch.__version__, "threads": a.threads,
           "workers": a.workers}
    if not a.probe:
        (out / "run.json").write_text(json.dumps(run, indent=1) + "\n", encoding="utf-8")
    log = open(out / ("probe.jsonl" if a.probe else "log.jsonl"), "a", encoding="utf-8")
    ema = T2.EMA(0.999)
    n_steps = a.probe if a.probe else a.steps
    t0 = time.time()
    agg, n_agg, times = {}, 0, []
    model.train()
    for step in range(n_steps):
        ts = time.perf_counter()
        lr = T2.lr_at(step, a.steps, a.lr, a.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        batch = next(it)
        td = time.perf_counter()
        y = model(batch[0])
        loss, parts = loss_all(y, batch)
        if not math.isfinite(float(loss)):
            raise SystemExit(f"step {step + 1}: the loss is {float(loss)}")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), B.CLIP)
        opt.step()
        if step + 1 >= a.ema_start:
            ema.step(model.parameters())
        times.append((td - ts, time.perf_counter() - ts))
        for k, v in parts.items():
            agg[k] = agg.get(k, 0.0) + v
        n_agg += 1
        if (step + 1) % a.log_every == 0 or step + 1 == n_steps:
            rec = {"step": step + 1, "lr": round(lr, 6), "sec": round(time.time() - t0, 1),
                   **{k: round(v / n_agg, 5) for k, v in agg.items()}}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(json.dumps(rec), flush=True)
            agg, n_agg = {}, 0
    if a.probe:
        ms = np.array([t for _, t in times[2:]]) * 1e3
        wait = np.array([t for t, _ in times[2:]]) * 1e3
        probe = {"name": a.name, "params": N.n_params(model), "ms_per_step": round(float(ms.mean()), 1),
                 "ms_waiting_for_data": round(float(wait.mean()), 1), "threads": a.threads, "workers": a.workers,
                 "startup_s": round(t0 - t_proc, 1), "pool": pool_rec["extra"], "sources": pool_rec["sources"],
                 "targets": pool_rec["targets"]}
        (out / "probe.json").write_text(json.dumps(probe, indent=1) + "\n")
        print("tsdesk_det probe:", json.dumps(probe), flush=True)
        return probe
    ema.copy_to(model.parameters())
    n_bn = DD.reestimate_bn(model, it, a.bn_batches)
    log.write(json.dumps({"step": n_steps, "ema": "the run's model", "ema_steps": ema.steps, "bn_layers": n_bn}) + "\n")
    torch.save({"kind": N.KIND, "shape": shape, "tags": N.TAGS, "seed": a.seed, "steps": a.steps,
                "recipe": a.recipe, "state_dict": model.state_dict(), "run": run}, out / "final.pt")
    worst = N.export_onnx(model, out / "final.onnx", N.onnx_meta(shape, a.name, DEFAULT_SIZE, DECODE), doc=ATTRIBUTION)
    done = {"finished": ST.utc(), "name": a.name, "steps": a.steps, "seconds": round(time.time() - t0, 1),
            "wall_seconds": round(time.time() - t_proc, 1), "params": N.n_params(model),
            "onnx_bytes": (out / "final.onnx").stat().st_size, "onnx_sha256": sha256_file(out / "final.onnx"),
            "pt_sha256": sha256_file(out / "final.pt"), "onnx_vs_torch_max_abs": worst}
    (out / "done.json").write_text(json.dumps(done, indent=1) + "\n", encoding="utf-8")
    log.close()
    print("tsdesk_det finished:", json.dumps(done), flush=True)
    return done


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        cmd_export(sys.argv[2:])
    else:
        main()
