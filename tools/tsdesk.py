#!/usr/bin/env python3
"""textspot-desktop: VOCR's fast desktop line detector with tight boxes (issue #173, docs/textspot-desktop.md), and
its page pipeline with a VOCR6 recognizer, on ONNX Runtime's CPU provider. The C path (desktop/, AVX2 + FMA with a
scalar fallback) runs the same network and gives the same boxes from the same maps.

    from tsdesk import Detector, read_frame
    from vocr_onnx import Recognizer
    det = Detector("textspot_desktop.onnx")                 # any frame size; `size` picks the input policy
    rec = Recognizer("VOCR6.2_L_fast.onnx")
    lines = read_frame(frame, det, rec, workers=4)          # frame: H x W grey or H x W x 3/4 RGB(A) uint8
    for ln in lines:
        ln.text, ln.box, ln.crop, ln.tag                    # box: the line's tight ink box (x, y, w, h), frame px

    python3 tsdesk.py --det textspot_desktop.onnx --rec VOCR6.2_L_fast.onnx [--size fit:960] IMAGE...

The detector (train/tsdesk_net.py) is a small fully convolutional network trained at many text sizes, so it reads a
frame at any size. `size` scales the frame first, keeping its aspect ratio: "fit:N" brings the longer side down to N px
(never up), "long:N" scales it to N px up or down, "native" keeps it. The network reads the grey frame (integer luma,
(77 R + 150 G + 29 B) >> 8, as VOCR's recognizer) padded right and bottom to a multiple of 32 by repeating its last
column and row. It returns maps at the head's stride s (8): K (a line's kernel, a logit), O (orientation: vertical when
> 0), A1 and A2 (the distances from a cell's centre to the line's two edges across it), E1 and E2 (the distances from
a cell's centre to the line's start and end along it), all in input px, and 4 tag logits.

The decode gives each line's ink box with no margin: the 4-connected components of K > 0 with at least `min_cells`
cells and a mean K logit of at least `conf_min_logit`; across, the mean over the component's cells of centre - A1 and
centre + A2; along, the mean of centre - E1 over the cells of its first column (row, for a vertical line) and of
centre + E2 over its last, each kept within the component's cells' reach (`along_reach` x its size across beyond
them, and never inside the first or last cell's far edge). The recognizer reads a box grown by `pad` (default
READ_PAD = "k6v4", the protocol's: max(2, round(6 H / 544)) frame px left and right and max(2, round(4 H / 544)) top
and bottom for an H px tall frame; "auto" is k4, the pad VOCR's eval crops and MeikiOCR's reference pipeline use).

Options (#173 round 2, helper textspot-desktop-boxes), every one off by default, so that with none set the boxes are
the decode's and the refinement's bit for bit; none has a C twin yet:
  - the decode's DECODE_EXTRA: vote_pct (split a component across where its cells' edge votes are bimodal: two
    tightly spaced rows), saddle_pct (split along at K saddles), rescue_milli (decode weak isolated components too);
  - the post-pass on frame-px boxes, POST_DEFAULTS: merge_pct, rows_pct and gap_pct before the refinement, ext_pct
    after it (the section "the post-pass" below);
  - the reading filter, DROP_DEFAULTS, in read_frame(drop=...): nojp and rep_pct (readings that are not Japanese or
    are an ornament), conf_milli (drop_set()).
    det = Detector(path, decode_opts={"vote_pct": 70}, refine=True,
                   post={"merge_pct": 70, "rows_pct": 65, "valley_pct": 0, "ext_pct": 1200})
    lines = read_frame(frame, det, rec, drop={"nojp": 2, "nojp_gap_pct": 200, "nojp_anchor_pct": 50, "rep_pct": 50})
"""
import json
import math
from dataclasses import dataclass

import numpy as np

ALIGN = 32
CLASSES = ("dialogue", "name", "choice", "ui", "scene")
MAX_BOXES = 256
DEFAULTS = {"min_cells": 2, "conf_min_logit": 0.0, "min_across": 4.0, "along_reach": 3.0, "max_boxes": MAX_BOXES}
DECODE_EXTRA = {"saddle_pct": 0, "vote_pct": 0, "rescue_milli": 0, "clamp_ends": 0,
                "vote_depth": 2, "along_pct": 0, "along_depth": 3}               # decode options (vote_pct: C too)
#                                                                       (helper textspot-desktop-boxes), 0 = off


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
    """The factor a w x h frame is resized by before the network, for a size policy (module docstring)."""
    if policy in (None, "", "native"):
        return 1.0
    kind, _, n = str(policy).partition(":")
    if not n:
        kind, n = "fit", kind
    n = float(n)
    long = max(w, h)
    if kind == "fit":
        return min(1.0, n / long)
    if kind == "long":
        return n / long
    raise ValueError(f"size policy {policy!r}: fit:N, long:N or native")


def input_size(policy, w, h):
    """(iw, ih) of the network input's image part for a w x h frame (each side round(side x f), at least 1)."""
    f = scale_for(policy, w, h)
    if f == 1.0:
        return w, h
    return max(1, int(round(w * f))), max(1, int(round(h * f)))


Q14 = 16384


def _axis(n, m):
    """(idx [m, K], weights [m, K]) of one axis of desktop/vocr_desk.c's vocr_desk_resize: Q14 area weights when the
    axis shrinks (m < n), bilinear on pixel centres when it grows; the weights of an output sum to 16384."""
    if m == n:
        return np.arange(m, dtype=np.int64)[:, None], np.full((m, 1), Q14, np.int64)
    K = n // m + 2 if m < n else 2
    idx = np.zeros((m, K), np.int64)
    wt = np.zeros((m, K), np.int64)
    for i in range(m):
        if m < n:
            lo, hi = i * n, i * n + n
            j, k, cum, prev = lo // m, 0, 0, 0
            while j * m < hi and k < K:
                a, b = max(lo, j * m), min(hi, (j + 1) * m)
                cum += b - a
                q = (cum * 2 * Q14 + n) // (2 * n)
                idx[i, k], wt[i, k] = j, q - prev
                prev = q
                j += 1
                k += 1
            for kk in range(k, K):
                idx[i, kk] = idx[i, kk - 1]
        else:
            num = (2 * i + 1) * n - m
            if num <= 0:
                idx[i, :2], wt[i, 0] = 0, Q14
            else:
                x0, r = num // (2 * m), num - (num // (2 * m)) * 2 * m
                if x0 >= n - 1:
                    idx[i, :2], wt[i, 0] = n - 1, Q14
                else:
                    w1 = (r * Q14 + m) // (2 * m)
                    idx[i, 0], idx[i, 1], wt[i, 0], wt[i, 1] = x0, x0 + 1, Q14 - w1, w1
    return idx, wt


def resize(g, dw, dh):
    """desktop/vocr_desk.c's vocr_desk_resize on a uint8 grey image, bit for bit: separable integer weights (the rows
    first, then the columns), (acc + 2^27) >> 28, clamped. The detector's C path resizes this way, so the ONNX path
    and the C path see the same network input."""
    g = np.asarray(g, np.uint8)
    h, w = g.shape
    if (w, h) == (dw, dh):
        return g.copy()
    ix, wx = _axis_cached(w, dw)
    iy, wy = _axis_cached(h, dh)
    tmp = np.zeros((h, dw), np.int32)
    for k in range(ix.shape[1]):
        tmp += g[:, ix[:, k]].astype(np.int32) * wx[:, k].astype(np.int32)[None, :]
    acc = np.zeros((dh, dw), np.int64)
    for k in range(iy.shape[1]):
        acc += tmp[iy[:, k]].astype(np.int64) * wy[:, k][:, None]
    return np.clip((acc + (1 << 27)) >> 28, 0, 255).astype(np.uint8)


_AXES = {}


def _axis_cached(n, m):
    key = (int(n), int(m))
    if key not in _AXES:
        if len(_AXES) > 64:
            _AXES.clear()
        _AXES[key] = _axis(n, m)
    return _AXES[key]


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
    box: tuple              # the detector's tight box (x, y, w, h), frame px
    crop: tuple             # the crop read: the box grown by the pad, integer frame px
    tag: int
    logit: float
    dark: int
    vertical: int = 0


def saddle_split(K, Om, lab, n, stats, pct):
    """The decode's saddle option: each component of >= 3 cells along its line (the orientation of its mean O) split
    at every interior along index whose K profile (the component's largest K across, at that index) is a local
    minimum at most pct % of the profile's median: the cells up to that index stay, the cells after it get a new
    label. (n, lab, stats, cut) with stats in cv2's layout and cut[label] = (its start is a split, its end is one)."""
    lab = lab.copy()
    st = [list(stats[c]) for c in range(n)]
    cut = [(False, False)] * n
    out_n = n
    for c in range(1, n):
        x, y, cw, ch, area = (int(v) for v in stats[c][:5])
        if area < 3:
            continue
        sub = lab[y:y + ch, x:x + cw] == c
        vert = float(Om[y:y + ch, x:x + cw][sub].mean()) > 0
        ks = np.where(sub, K[y:y + ch, x:x + cw], -np.inf)
        prof = ks.max(axis=1) if vert else ks.max(axis=0)
        if len(prof) < 3:
            continue
        med = float(np.median(prof))
        if med <= 0:
            continue
        thr = pct / 100.0 * med
        cuts = [i for i in range(1, len(prof) - 1)
                if prof[i] <= thr and prof[i] <= prof[i - 1] and prof[i] < prof[i + 1]]
        if not cuts:
            continue
        bounds = [0] + [i + 1 for i in cuts] + [len(prof)]
        labels = [c] + list(range(out_n, out_n + len(cuts)))
        out_n += len(cuts)
        for k_, (a0, a1) in enumerate(zip(bounds[:-1], bounds[1:])):
            part = np.zeros_like(sub)
            if vert:
                part[a0:a1, :] = sub[a0:a1, :]
            else:
                part[:, a0:a1] = sub[:, a0:a1]
            lab[y:y + ch, x:x + cw][part] = labels[k_]
            rr, qq = np.nonzero(part)
            entry = [x + int(qq.min()), y + int(rr.min()), int(qq.max() - qq.min()) + 1, int(rr.max() - rr.min()) + 1,
                     int(part.sum())]
            if labels[k_] == c:
                st[c] = entry
            else:
                st.append(entry)
            flags = (k_ > 0, k_ < len(cuts))
            if labels[k_] == c:
                cut[c] = flags
            else:
                cut.append(flags)
    if out_n == n:
        return n, lab, stats, cut
    return out_n, lab, np.array(st, dtype=stats.dtype).reshape(out_n, stats.shape[1]), cut


def _seq(a):
    """The sum of a float64 array added in order, one term at a time (the C twin's loop; np.sum adds pairwise)."""
    return float(np.add.accumulate(a)[-1]) if len(a) else 0.0


def vote_split(maps, lab, n, stats, pct, s, depth=2, cap=None):
    """The decode's vote option: a component whose cells span two or more indices across its line (the orientation
    of its summed O) is split across where its cells' edge votes are bimodal, two tightly spaced lines whose kernels
    touched: over the split points between consecutive across indices, the one that best separates the mean near
    votes (centre - A1) of the cells before it from those after it, kept when both that separation and the far votes'
    (centre + A2) are at least pct % of the component's mean size across (A1 + A2); recursively, `depth` levels, while
    the labels stay under `cap` (the C decode's label capacity). Every sum is the C twin's (desktop/vocr_tsd.c): the
    cells in raster order, one term at a time; a mean over cells before or after a split point adds the per-index sums
    in index order. (n, lab, stats) with stats in cv2's layout."""
    hm, wm = lab.shape
    Om, A1, A2 = (np.ascontiguousarray(maps[i, :hm, :wm]).astype(np.float64) for i in (1, 2, 3))
    lab = lab.copy()
    st = [list(stats[c]) for c in range(n)]
    half = (s - 1) / 2.0
    todo = [(c, depth) for c in range(1, n)]
    while todo:
        c, dep = todo.pop()
        if cap is not None and n >= cap:
            break
        x, y, cw, ch = (int(v) for v in st[c][:4])
        sub = lab[y:y + ch, x:x + cw] == c
        rr, qq = np.nonzero(sub)
        m = len(rr)
        if m < 4:
            continue
        vert = _seq(Om[rr + y, qq + x]) / m > 0.0
        ai = (qq + x) if vert else (rr + y)
        cc = s * ai.astype(np.float64) + half
        near = cc - A1[rr + y, qq + x]
        far = cc + A2[rr + y, qq + x]
        size = _seq(far - near) / m
        idx = np.unique(ai)
        if len(idx) < 2 or size <= 0:
            continue
        pos = np.searchsorted(idx, ai)
        k_n = np.bincount(pos, minlength=len(idx)).astype(np.float64)
        s_n = np.bincount(pos, near, minlength=len(idx))
        s_f = np.bincount(pos, far, minlength=len(idx))
        best = None
        for k in range(len(idx) - 1):
            ln = lf = lc = hn = hf = hc = 0.0
            for q in range(len(idx)):
                if q <= k:
                    ln, lf, lc = ln + s_n[q], lf + s_f[q], lc + k_n[q]
                else:
                    hn, hf, hc = hn + s_n[q], hf + s_f[q], hc + k_n[q]
            dn = hn / hc - ln / lc
            df = hf / hc - lf / lc
            sep = dn if dn <= df else df
            if best is None or sep > best[0]:
                best = (sep, k)
        if best is None or best[0] * 100.0 < pct * size:
            continue
        lo = ai <= idx[best[1]]
        part = np.zeros_like(sub)
        part[rr[~lo], qq[~lo]] = True
        lab[y:y + ch, x:x + cw][part] = n
        keep = sub & ~part
        for msk, lbl in ((keep, c), (part, n)):
            r2, q2 = np.nonzero(msk)
            entry = [x + int(q2.min()), y + int(r2.min()), int(q2.max() - q2.min()) + 1, int(r2.max() - r2.min()) + 1,
                     int(msk.sum())]
            if lbl == c:
                st[c] = entry
            else:
                st.append(entry)
        n += 1
        if dep > 1:
            todo += [(c, dep - 1), (n - 1, dep - 1)]
    return n, lab, np.array(st, dtype=stats.dtype).reshape(n, stats.shape[1])


def along_split(maps, lab, n, stats, pct, s, depth=3, cap=None, w=2):
    """The decode's along option: a component is split along its line (the orientation of its summed O) where its
    cells' along votes part, two items of a row whose kernels touched (a label and its value, menu entries, a bullet
    and its text): the network's end votes (centre + E2) of the cells before a split point stop where the start votes
    (centre - E1) of the cells after it begin, while along one line every cell votes the line's own two ends. Over the
    split points between consecutive along indices (columns; rows for a vertical line), the separation is the mean
    start vote of the cells of the w indices after the point minus the mean end vote of the cells of the w indices
    before it; the best point is kept when its separation is at least -pct % of the component's mean size across
    (A1 + A2) and each part holds at least w indices; recursively, `depth` levels, while the labels stay under `cap`.
    Sums as vote_split's: the cells in raster order, one term at a time; a window's mean adds its per-index sums in
    index order. (n, lab, stats) with stats in cv2's layout."""
    hm, wm = lab.shape
    Om, A1, A2, E1, E2 = (np.ascontiguousarray(maps[i, :hm, :wm]).astype(np.float64) for i in (1, 2, 3, 4, 5))
    lab = lab.copy()
    st = [list(stats[c]) for c in range(n)]
    half = (s - 1) / 2.0
    todo = [(c, depth) for c in range(1, n)]
    while todo:
        c, dep = todo.pop()
        if cap is not None and n >= cap:
            break
        x, y, cw, ch = (int(v) for v in st[c][:4])
        sub = lab[y:y + ch, x:x + cw] == c
        rr, qq = np.nonzero(sub)
        m = len(rr)
        if m < 2 * w:
            continue
        R, Q = rr + y, qq + x
        vert = _seq(Om[R, Q]) / m > 0.0
        size = _seq(A1[R, Q] + A2[R, Q]) / m
        ai = R if vert else Q
        cc = s * ai.astype(np.float64) + half
        start = cc - E1[R, Q]
        end = cc + E2[R, Q]
        idx = np.unique(ai)
        if len(idx) < 2 * w or size <= 0:
            continue
        pos = np.searchsorted(idx, ai)
        k_n = np.bincount(pos, minlength=len(idx)).astype(np.float64)
        s_s = np.bincount(pos, start, minlength=len(idx))
        s_e = np.bincount(pos, end, minlength=len(idx))
        best = None
        for k in range(w - 1, len(idx) - w):
            le = lc = hs = hc = 0.0
            for q in range(k - w + 1, k + 1):
                le, lc = le + s_e[q], lc + k_n[q]
            for q in range(k + 1, k + w + 1):
                hs, hc = hs + s_s[q], hc + k_n[q]
            sep = hs / hc - le / lc
            if best is None or sep > best[0]:
                best = (sep, k)
        if best is None or best[0] * 100.0 < -pct * size:
            continue
        hi = ai > idx[best[1]]
        part = np.zeros_like(sub)
        part[rr[hi], qq[hi]] = True
        lab[y:y + ch, x:x + cw][part] = n
        keep = sub & ~part
        for msk, lbl in ((keep, c), (part, n)):
            r2, q2 = np.nonzero(msk)
            entry = [x + int(q2.min()), y + int(r2.min()), int(q2.max() - q2.min()) + 1, int(r2.max() - r2.min()) + 1,
                     int(msk.sum())]
            if lbl == c:
                st[c] = entry
            else:
                st.append(entry)
        n += 1
        if dep > 1:
            todo += [(c, dep - 1), (n - 1, dep - 1)]
    return n, lab, np.array(st, dtype=stats.dtype).reshape(n, stats.shape[1])


def rescue_components(k, lab, n, stats, t_low):
    """The decode's rescue option: the 4-connected components of K > -t_low that hold no cell of K > 0 (a line the
    network saw too weakly: no kernel cell, nothing found there) added with new labels. (n, lab, stats, first new
    label)."""
    import cv2
    m, lab2, st2, _ = cv2.connectedComponentsWithStats((k > -t_low).astype(np.uint8), connectivity=4)
    if m <= 1:
        return n, lab, stats, n
    has_pos = np.bincount(lab2.ravel(), weights=(k > 0).ravel().astype(np.float64), minlength=m) > 0
    new = [c for c in range(1, m) if not has_pos[c]]
    if not new:
        return n, lab, stats, n
    lab = lab.copy()
    rows = [stats]
    for i, c in enumerate(new):
        lab[lab2 == c] = n + i
        rows.append(st2[c:c + 1].astype(stats.dtype))
    return n + len(new), lab, np.concatenate(rows, 0), n


def decode(maps, w, h, stride, min_cells=2, conf_min_logit=0.0, min_across=4.0, along_reach=3.0, tags=4,
           max_boxes=MAX_BOXES, saddle_pct=0, vote_pct=0, rescue_milli=0, clamp_ends=0, vote_depth=2,
           along_pct=0, along_depth=3):
    """[Box] in network-input px (the image's w x h part) from float maps [6 + T, H / s, W / s] (module docstring).
    The C decode (desktop/vocr_tsd.c) is the same rule with the same sums in the same order (in double, over the
    cells in raster order), so given the same maps the two give the same boxes bit for bit. Components are labelled
    in the order of each one's first cell; boxes are sorted by mean logit (descending), then y, then x, stable, and
    the first max_boxes kept. Off by default: vote_pct > 0, each component split across where its cells' edge votes
    are bimodal (vote_split(), vote_depth levels, 2 by default; C twins of both). No C twin yet:
    along_pct > 0, each component (after the vote split) split along where its cells' end and start votes part
    (along_split(), along_depth levels: two items of a row whose kernels touched);
    saddle_pct > 0, each component split along at the K saddles saddle_split() finds, the parts' ends at a split at
    their outermost cells' edges (no E1 / E2 there); rescue_milli > 0, the components of K > -rescue_milli / 1000
    with no K > 0 cell decoded too (rescue_components(); their mean logit is negative, so conf_min_logit does not
    apply to them); clamp_ends, an end regressed past the kernel never reaches into another component of the same
    orientation in its row (across overlap at least half the smaller size across): it stops at that component's
    outermost cell edge (a label's end run on over the number beside it)."""
    import cv2
    s = int(stride)
    hm, wm = -(-h // s), -(-w // s)
    k = np.ascontiguousarray(maps[0, :hm, :wm])
    n, lab, stats, _ = cv2.connectedComponentsWithStats((k > 0).astype(np.uint8), connectivity=4)
    if n <= 1:
        return []
    cut = None
    if rescue_milli:
        n, lab, stats, _ = rescue_components(k, lab, n, stats, rescue_milli / 1000.0)
    if vote_pct:
        n, lab, stats = vote_split(maps, lab, n, stats, vote_pct, s, depth=int(vote_depth),
                                   cap=(maps.shape[1] * maps.shape[2]) // 2 + 2)
    if along_pct:
        n, lab, stats = along_split(maps, lab, n, stats, along_pct, s, depth=int(along_depth),
                                    cap=(maps.shape[1] * maps.shape[2]) // 2 + 2)
    if saddle_pct:
        n, lab, stats, cut = saddle_split(k, np.ascontiguousarray(maps[1, :hm, :wm]), lab, n, stats, saddle_pct)
    flat = lab.ravel()
    on = np.flatnonzero(flat)
    L = flat[on]
    rows, cols = np.divmod(on, wm)

    def m_of(c):
        return np.ascontiguousarray(maps[c, :hm, :wm]).ravel()[on].astype(np.float64)

    cnt = np.bincount(L, minlength=n).astype(np.float64)
    cnt[0] = 1.0
    pos = np.bincount(L, (m_of(0) > 0).astype(np.float64), minlength=n) if rescue_milli else cnt
    logit = np.bincount(L, m_of(0), minlength=n) / cnt
    vert = np.bincount(L, m_of(1), minlength=n) / cnt > 0
    vL = vert[L]
    half = (s - 1) / 2.0
    cc = s * np.where(vL, cols, rows).astype(np.float64) + half
    near = np.bincount(L, cc - m_of(2), minlength=n) / cnt
    far = np.bincount(L, cc + m_of(3), minlength=n) / cnt
    ai = np.where(vL, rows, cols)
    first = np.where(vert, stats[:, cv2.CC_STAT_TOP], stats[:, cv2.CC_STAT_LEFT])
    last = first + np.where(vert, stats[:, cv2.CC_STAT_HEIGHT], stats[:, cv2.CC_STAT_WIDTH]) - 1
    ca = s * ai.astype(np.float64) + half
    s1 = ai == first[L]
    s2 = ai == last[L]
    n1 = np.bincount(L[s1], minlength=n).astype(np.float64)
    n2 = np.bincount(L[s2], minlength=n).astype(np.float64)
    n1[0] = n2[0] = 1.0
    start = np.bincount(L[s1], (ca - m_of(4))[s1], minlength=n) / n1
    end = np.bincount(L[s2], (ca + m_of(5))[s2], minlength=n) / n2
    tsum = [np.bincount(L, m_of(6 + i), minlength=n) for i in range(tags)]
    out, kern = [], []
    for c in range(1, n):
        if stats[c, cv2.CC_STAT_AREA] < min_cells or (logit[c] < conf_min_logit and pos[c] > 0):
            continue
        size = float(far[c] - near[c])
        if size < min_across:
            continue
        lo, hi = float(s * int(first[c])), float(s * (int(last[c]) + 1))
        b0, b1 = float(start[c]), float(end[c])
        if cut is not None and cut[c][0]:
            b0 = lo
        if cut is not None and cut[c][1]:
            b1 = hi
        if b0 > lo + s:
            b0 = lo + s
        if b0 < lo - along_reach * size:
            b0 = lo - along_reach * size
        if b1 < hi - s:
            b1 = hi - s
        if b1 > hi + along_reach * size:
            b1 = hi + along_reach * size
        a0, a1 = float(near[c]), float(far[c])
        x0, x1, y0, y1 = (a0, a1, b0, b1) if vert[c] else (b0, b1, a0, a1)
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
        kern.append((lo, hi))
    if clamp_ends:
        out = _clamp_ends(out, kern, w, h)
    out.sort(key=lambda b: (-b.logit, b.y, b.x))
    return out[:max_boxes]


def _clamp_ends(boxes, kern, w, h):
    """decode()'s clamp_ends: each box's along ends kept off the other same-orientation components of its row (their
    kernel cells' extent along, kern[i] = (lo, hi))."""
    import dataclasses
    out = []
    for i, b in enumerate(boxes):
        v = b.vertical
        a0, a1 = (b.x, b.x + b.w) if v else (b.y, b.y + b.h)           # across
        e0, e1 = (b.y, b.y + b.h) if v else (b.x, b.x + b.w)           # along
        lo, hi = kern[i]
        for j, o in enumerate(boxes):
            if j == i or o.vertical != v:
                continue
            o0, o1 = (o.x, o.x + o.w) if v else (o.y, o.y + o.h)
            if min(a1, o1) - max(a0, o0) < 0.5 * min(a1 - a0, o1 - o0):
                continue
            olo, ohi = kern[j]
            if olo >= hi and e1 > olo:
                e1 = olo
            if ohi <= lo and e0 < ohi:
                e0 = ohi
        if e1 <= e0:
            out.append(b)
        elif v:
            out.append(dataclasses.replace(b, y=e0, h=e1 - e0))
        else:
            out.append(dataclasses.replace(b, x=e0, w=e1 - e0))
    return out


# ----------------------------------------------------------------------------- the post-pass (#173 round 2)
#
# Options on the decode's frame-px boxes (helper textspot-desktop-boxes). Every option is an integer and 0 turns it
# off, so with POST_DEFAULTS the boxes are the decode's and the refinement's bit for bit. The pre-refinement options
# run in this order on the decode's boxes, before tools/tsdesk_refine.py (vflip_pct first, then merge_pct, touch_pct
# (merge_touching), join_pct (join_row), rows_pct and gap_pct; touch_pct, gap_pct, ext_band_pct and ksplit_pct have no
# C twin):
#   vflip_pct  a vertical box whose ink is a stack of glyph bands with wide gaps becomes horizontal boxes (vflip);
#   join_pct   two dialogue-tagged boxes on one row with a small gap along and similar heights join (join_row);
#   ksplit_pct the last step, after the post-refinement options: a horizontal box is cut at an ink gap the detector's
#              own K map does not bridge (split_kgap; Detector only, it needs the maps);
#   merge_pct  two boxes of one orientation that overlap across by at least half the smaller one's size across and
#              whose intersection is at least merge_pct % of the smaller one's area become their union (the larger
#              box's fields): a fragment the decode found inside or beside its line (a 「, a ―。 at the end);
#   rows_pct   a horizontal box split across at an ink valley (rows with at most valley_pct % of the median ink row's
#              count) between two ink bands whose heights are within rows_pct % of each other, each at least
#              max(4, 20 % of the box's height) rows tall and at least rows_len % of the box's height long in ink:
#              two tightly spaced rows boxed as one (a band of ruby, about half a line's height, is not split off,
#              nor a short label whose glyphs have a gap across); at most two levels;
#   gap_pct    a horizontal box split along at every run of at least max(2, r(h, gap_pct)) empty ink columns between
#              its first and last ink column (h its height) that holds fewer than gap_weak pixels of weak ink (both
#              thresholds halved: a faint 、 or 。): neighbouring buttons, labels and numbers in one box.
# The post-refinement option runs on the refined boxes:
#   ext_pct    a horizontal box's ends extended over the dash or ellipsis run beyond them (extend_ends).
# The ink is the refinement's test (tools/tsdesk_refine.py steps 3-6, without its margins' noise term): on the box
# grown by max(2, r(h, 25)) px, bg the lower median of that window's border ring, a pixel is ink when |g - bg| and
# its 3 x 3 contrast both exceed max(28, 40 % of their 99th percentile over the box) and a 4-neighbour is ink too.

POST_DEFAULTS = {"merge_pct": 0, "rows_pct": 0, "valley_pct": 10, "rows_len": 300, "gap_pct": 0, "gap_weak": 3,
                 "ext_pct": 0, "ext_gap_pct": 70, "ext_run_pct": 15, "ext_dash_pct": 80, "ext_join": 1,
                 "ext_band_pct": 0, "touch_pct": 0, "vflip_pct": 0, "vflip_join_pct": 100,
                 "vflip_min": 3, "vflip_h_pct": 50, "ksplit_pct": 0, "ksplit_k": -2, "ksplit_min_pct": 50,
                 "join_pct": 0, "join_h_pct": 70, "join_ov_pct": 80}
POST_PRE = ("vflip_pct", "merge_pct", "touch_pct", "join_pct", "rows_pct", "gap_pct")   # before the refinement
POST_POST = ("ext_pct",)                              # the options that run after it


def post_options(opts=None):
    o = dict(POST_DEFAULTS)
    for k, v in (opts or {}).items():
        if k not in POST_DEFAULTS:
            raise KeyError(f"unknown post-pass option {k!r}")
        o[k] = int(v)
    return o


def _r100(v, pct):
    return (int(v) * int(pct) + 50) // 100


def _pixel_box(b, W, H):
    """(ix0, iy0, ix1, iy1) of a Box: floor / ceil of its edges, clipped to the frame."""
    ix0 = min(max(int(math.floor(b.x)), 0), W)
    iy0 = min(max(int(math.floor(b.y)), 0), H)
    ix1 = min(max(int(math.ceil(b.x + b.w)), 0), W)
    iy1 = min(max(int(math.ceil(b.y + b.h)), 0), H)
    return ix0, iy0, ix1, iy1


def ink_mask(g, ix0, iy0, ix1, iy1, floor=28, k_pct=40, weak=False):
    """The ink of the pixel box [ix0, ix1) x [iy0, iy1) of grey g (the post-pass's test, above): a bool array of the
    box's shape. weak: also the weak ink, (mask, weak mask), where both thresholds are halved (a faint 、 or 。 that
    the strong test misses), without the 4-neighbour rule."""
    import tsdesk_refine as TR
    H, W = g.shape
    h, w = iy1 - iy0, ix1 - ix0
    A = max(2, _r100(h, 25))
    wx0, wy0, wx1, wy1 = max(0, ix0 - A), max(0, iy0 - A), min(W, ix1 + A), min(H, iy1 + A)
    win = g[wy0:wy1, wx0:wx1].astype(np.int16)
    ring = np.concatenate([win[0], win[-1], win[1:-1, 0], win[1:-1, -1]])
    bg = int(np.partition(ring, (ring.size - 1) // 2)[(ring.size - 1) // 2])
    d = np.abs(win - bg)
    c3 = TR._c3(g, wx0, wy0, wx1, wy1)
    by0, bx0 = iy0 - wy0, ix0 - wx0
    t = max(floor, _r100(TR.quantile(d[by0:by0 + h, bx0:bx0 + w], 99), k_pct))
    tc = max(floor, _r100(TR.quantile(c3[by0:by0 + h, bx0:bx0 + w], 99), k_pct))
    m0 = (d > t) & (c3 > tc)
    nb = np.zeros_like(m0)
    nb[1:] |= m0[:-1]
    nb[:-1] |= m0[1:]
    nb[:, 1:] |= m0[:, :-1]
    nb[:, :-1] |= m0[:, 1:]
    m = (m0 & nb)[by0:by0 + h, bx0:bx0 + w]
    if not weak:
        return m
    wk = ((d > t // 2) & (c3 > tc // 2))[by0:by0 + h, bx0:bx0 + w]
    return m, wk


def merge_overlaps(boxes, pct):
    """merge_pct: the pair with the largest intersection (as a share of the smaller box's area) merged first, until no
    pair qualifies. The order of the boxes is kept (a merged box takes the place of the earlier of the two)."""
    import dataclasses
    out = list(boxes)
    while True:
        best = None
        for i in range(len(out)):
            a = out[i]
            for j in range(i + 1, len(out)):
                b = out[j]
                if a.vertical != b.vertical:
                    continue
                dx = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
                dy = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
                if dx <= 0 or dy <= 0:
                    continue
                across, size = (dx, min(a.w, b.w)) if a.vertical else (dy, min(a.h, b.h))
                if across < 0.5 * size:
                    continue
                small = min(a.w * a.h, b.w * b.h)
                if small <= 0 or dx * dy * 100 < pct * small:
                    continue
                share = dx * dy / small
                if best is None or share > best[0]:
                    best = (share, i, j)
        if best is None:
            return out
        _, i, j = best
        a, b = out[i], out[j]
        big = a if a.w * a.h >= b.w * b.h else b
        x0, y0 = min(a.x, b.x), min(a.y, b.y)
        x1, y1 = max(a.x + a.w, b.x + b.w), max(a.y + a.h, b.y + b.h)
        out[i] = dataclasses.replace(big, x=x0, y=y0, w=x1 - x0, h=y1 - y0, cells=a.cells + b.cells)
        del out[j]


def merge_touching(boxes, pct):
    """touch_pct: two boxes of one orientation that overlap across by at least pct % of the smaller size across and
    touch or overlap along (no gap between them) become their union (the larger box's fields, the place of the
    earlier), pairs in order until none qualifies: a line the decode cut in two where its kernel dips (at a ――)."""
    import dataclasses
    out = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(out)):
            a = out[i]
            for j in range(i + 1, len(out)):
                b = out[j]
                if a.vertical != b.vertical:
                    continue
                if a.vertical:
                    across = min(a.x + a.w, b.x + b.w) - max(a.x, b.x)
                    size = min(a.w, b.w)
                    gap = max(b.y - (a.y + a.h), a.y - (b.y + b.h))
                else:
                    across = min(a.y + a.h, b.y + b.h) - max(a.y, b.y)
                    size = min(a.h, b.h)
                    gap = max(b.x - (a.x + a.w), a.x - (b.x + b.w))
                if gap > 0 or across * 100 < pct * size:
                    continue
                big = a if a.w * a.h >= b.w * b.h else b
                x0, y0 = min(a.x, b.x), min(a.y, b.y)
                x1, y1 = max(a.x + a.w, b.x + b.w), max(a.y + a.h, b.y + b.h)
                out[i] = dataclasses.replace(big, x=x0, y=y0, w=x1 - x0, h=y1 - y0, cells=a.cells + b.cells)
                del out[j]
                changed = True
                break
            if changed:
                break
    return out


def join_row(boxes, pct, h_pct=70, ov_pct=80):
    """join_pct: two horizontal boxes the detector tags as dialogue (tag 0) on one row (vertical overlap at least
    ov_pct % of the smaller height: pieces of one line, not a piece between two lines), the smaller height at least
    h_pct % of the larger, whose gap along is at most pct % of
    the larger height (an overlap is a gap below 0) become their union (the larger box's fields, cells summed, the
    place of the earlier), pairs in order until none qualifies: a dialogue line the decode broke into pieces along
    its row (a stride-4 head's kernel gaps). Boxes with another tag stay apart (the reading stage's cut's items)."""
    import dataclasses
    out = list(boxes)
    changed = True
    while changed:
        changed = False
        for i in range(len(out)):
            a = out[i]
            if a.vertical or a.tag != 0:
                continue
            for j in range(i + 1, len(out)):
                b = out[j]
                if b.vertical or b.tag != 0:
                    continue
                lo, hi = min(a.h, b.h), max(a.h, b.h)
                if (min(a.y + a.h, b.y + b.h) - max(a.y, b.y)) * 100 < ov_pct * lo or lo * 100 < h_pct * hi:
                    continue
                if max(b.x - (a.x + a.w), a.x - (b.x + b.w)) * 100 > pct * hi:
                    continue
                big = a if a.w * a.h >= b.w * b.h else b
                x0, y0 = min(a.x, b.x), min(a.y, b.y)
                x1, y1 = max(a.x + a.w, b.x + b.w), max(a.y + a.h, b.y + b.h)
                out[i] = dataclasses.replace(big, x=x0, y=y0, w=x1 - x0, h=y1 - y0, cells=a.cells + b.cells)
                del out[j]
                changed = True
                break
            if changed:
                break
    return out


def vflip(g, boxes, gap_pct, join_pct=100, min_bands=3, h_pct=50):
    """vflip_pct: a vertical box whose ink is a stack of at least min_bands glyph bands (runs of ink rows, each at
    least max(2, 15 % of the box's width) tall), their median height at least h_pct % of the box's width, with a
    median gap between them of at least gap_pct % of their median height, is a stack of horizontal items, not a
    column of vertical text (whose glyphs nearly touch): a kana grid, a menu, a list of short labels. It becomes one
    horizontal box per band (the band's rows, its ink columns; at most MAX_BOXES with the boxes kept, the C twin's
    room: desktop/vocr_tsd_post.c); then the pieces of the frame's
    converted boxes that sit in one row (vertical overlap at least half the smaller height) within a gap of
    join_pct % of the larger height join, pairs in order until none does (the larger piece's fields)."""
    import dataclasses
    H, W = g.shape
    out, pieces = [], []
    for b in boxes:
        if not b.vertical:
            out.append(b)
            continue
        ix0, iy0, ix1, iy1 = _pixel_box(b, W, H)
        w, h = ix1 - ix0, iy1 - iy0
        if w < 4 or h < 8:
            out.append(b)
            continue
        m = ink_mask(g, ix0, iy0, ix1, iy1)
        rc = m.sum(1)
        bands, y = [], 0
        while y < h:
            if not rc[y]:
                y += 1
                continue
            y0 = y
            while y < h and rc[y]:
                y += 1
            if y - y0 >= max(2, _r100(w, 15)):
                bands.append((y0, y))
        if len(bands) < max(2, min_bands):
            out.append(b)
            continue
        hs = sorted(e - a for a, e in bands)
        gaps = sorted(bands[i + 1][0] - bands[i][1] for i in range(len(bands) - 1))
        hm = hs[(len(hs) - 1) // 2]
        if gaps[(len(gaps) - 1) // 2] * 100 < gap_pct * hm or hm * 100 < h_pct * w:
            out.append(b)
            continue
        for a, e in bands:
            cols = np.flatnonzero(m[a:e].any(0))
            pieces.append(dataclasses.replace(b, x=float(ix0 + int(cols[0])), y=float(iy0 + a),
                                              w=float(int(cols[-1]) - int(cols[0]) + 1), h=float(e - a), vertical=0,
                                              cells=1))
    pieces = pieces[:max(0, MAX_BOXES - len(out))]                      # the C twin's room (cap = MAX_BOXES)
    changed = True
    while changed:
        changed = False
        for i in range(len(pieces)):
            a = pieces[i]
            for j in range(i + 1, len(pieces)):
                c = pieces[j]
                if min(a.y + a.h, c.y + c.h) - max(a.y, c.y) < 0.5 * min(a.h, c.h):
                    continue
                if max(c.x - (a.x + a.w), a.x - (c.x + c.w)) * 100 > join_pct * max(a.h, c.h):
                    continue
                big = a if a.w * a.h >= c.w * c.h else c
                x0, y0 = min(a.x, c.x), min(a.y, c.y)
                x1, y1 = max(a.x + a.w, c.x + c.w), max(a.y + a.h, c.y + c.h)
                pieces[i] = dataclasses.replace(big, x=x0, y=y0, w=x1 - x0, h=y1 - y0, cells=a.cells + c.cells)
                del pieces[j]
                changed = True
                break
            if changed:
                break
    return out + pieces


def split_kgap(g, boxes, K, fx, fy, stride, gap_pct, k_max=-2, min_pct=50):
    """ksplit_pct: [Box] with each horizontal box cut at its runs of empty ink columns (ink_mask) at least
    max(2, gap_pct % of its height) long where the detector's K map does not bridge the gap: the lowest, over the K
    columns the run covers, of the column's maximum over the box's K rows is below k_max (a logit). Such a box was
    joined after the decode (the refinement's or the extension's reach, or the merge): a line and an icon beside it,
    two items of a row. A cut is kept only when the ink on either side, to the next kept cut, is at least min_pct % of
    the height wide (an end mark like the 」 after a ？ stays). The first part keeps the box's start and the last its
    end; the edges at a cut lie on the ink either side of the gap."""
    import dataclasses
    H, W = g.shape
    KH, KW = K.shape
    out = []
    for b in boxes:
        ix0, iy0, ix1, iy1 = _pixel_box(b, W, H)
        h, w = iy1 - iy0, ix1 - ix0
        if b.vertical or h < 4 or w < 4:
            out.append(b)
            continue
        cc = ink_mask(g, ix0, iy0, ix1, iy1).any(0)
        ink = np.flatnonzero(cc)
        if len(ink) < 2:
            out.append(b)
            continue
        G = max(2, _r100(h, gap_pct))
        r0 = min(max(int(iy0 * fy) // stride, 0), KH - 1)
        r1 = min(max(int(math.ceil(iy1 * fy / stride)), r0 + 1), KH)
        cuts = []                                                        # (last ink col + 1, next ink col), box px
        for i in np.flatnonzero(np.diff(ink) - 1 >= G):
            a, e = int(ink[i]) + 1, int(ink[i + 1])
            c0 = min(max(int((ix0 + a) * fx) // stride, 0), KW - 1)
            c1 = min(max(int(math.ceil((ix0 + e) * fx / stride)), c0 + 1), KW)
            if float(K[r0:r1, c0:c1].max(0).min()) < k_max:
                cuts.append((a, e))
        if not cuts:
            out.append(b)
            continue
        mw = _r100(h, min_pct)
        keep, start = [], int(ink[0])
        for k, (a, e) in enumerate(cuts):
            nxt = cuts[k + 1][0] if k + 1 < len(cuts) else int(ink[-1]) + 1
            if a - start >= mw and nxt - e >= mw:
                keep.append((a, e))
                start = e
        if not keep:
            out.append(b)
            continue
        xs = [b.x] + [float(ix0 + e) for _, e in keep]
        xe = [float(ix0 + a) for a, _ in keep] + [b.x + b.w]
        out.extend(dataclasses.replace(b, x=s0, w=e0 - s0) for s0, e0 in zip(xs, xe) if e0 > s0)
    return out[:MAX_BOXES]


def split_rows(g, b, rows_pct, valley_pct=10, rows_len=300, depth=0):
    """rows_pct: [Box] of a horizontal box split across at an ink valley (above), recursively to two levels; each
    band's ink must run at least rows_len % of the box's height along (two rows of a page, not a short label whose
    glyphs have a gap across)."""
    import dataclasses
    H, W = g.shape
    ix0, iy0, ix1, iy1 = _pixel_box(b, W, H)
    h, w = iy1 - iy0, ix1 - ix0
    if b.vertical or h < 8 or w < 8 or depth >= 2:
        return [b]
    m = ink_mask(g, ix0, iy0, ix1, iy1)
    rc = m.sum(1)
    ink = np.flatnonzero(rc)
    if len(ink) < 2:
        return [b]
    vals = rc[ink]
    med = int(np.partition(vals, (vals.size - 1) // 2)[(vals.size - 1) // 2])
    low = _r100(med, valley_pct)
    dense = np.flatnonzero(rc > low)
    if len(dense) < 2:
        return [b]
    lo, hi = int(dense[0]), int(dense[-1]) + 1
    min_band = max(4, _r100(h, 20))
    min_len = _r100(h, rows_len)
    best = None
    y = lo
    while y < hi:
        if rc[y] > low:
            y += 1
            continue
        v0 = y
        while y < hi and rc[y] <= low:
            y += 1
        v1 = y                                   # the valley's rows [v0, v1), dense rows on both sides
        ha, hb = v0 - lo, hi - v1
        if min(ha, hb) < min_band or min(ha, hb) * 100 < rows_pct * max(ha, hb):
            continue
        ca = np.flatnonzero(m[lo:v0].any(0))
        cb = np.flatnonzero(m[v1:hi].any(0))
        if not len(ca) or not len(cb) or ca[-1] - ca[0] + 1 < min_len or cb[-1] - cb[0] + 1 < min_len:
            continue
        key = (min(ha, hb) / max(ha, hb), v1 - v0)
        if best is None or key > best[0]:
            best = (key, v0, v1)
    if best is None:
        return [b]
    _, v0, v1 = best
    top = dataclasses.replace(b, h=float(iy0 + v0) - b.y)
    bot = dataclasses.replace(b, y=float(iy0 + v1), h=b.y + b.h - float(iy0 + v1))
    return (split_rows(g, top, rows_pct, valley_pct, rows_len, depth + 1)
            + split_rows(g, bot, rows_pct, valley_pct, rows_len, depth + 1))


def split_along(g, b, gap_pct, gap_weak=3):
    """gap_pct: [Box] of a horizontal box split along at its long runs of empty ink columns (above); a run that holds
    at least gap_weak pixels of weak ink (ink_mask(weak=True): a faint 、 or 。) is not a gap. The first part keeps
    the box's start and the last its end; the edges at a split lie on the ink either side of the gap."""
    import dataclasses
    H, W = g.shape
    ix0, iy0, ix1, iy1 = _pixel_box(b, W, H)
    h, w = iy1 - iy0, ix1 - ix0
    if b.vertical or h < 4 or w < 4:
        return [b]
    m, wk = ink_mask(g, ix0, iy0, ix1, iy1, weak=True)
    cc = m.sum(0)
    ink = np.flatnonzero(cc)
    if len(ink) < 2:
        return [b]
    G = max(2, _r100(h, gap_pct))
    wc = wk.sum(0)
    cut = [int(c) for c in np.flatnonzero(np.diff(ink) - 1 >= G)
           if gap_weak <= 0 or int(wc[ink[c] + 1:ink[c + 1]].sum()) < gap_weak]
    if not cut:
        return [b]
    starts = [b.x] + [float(ix0 + int(ink[c + 1])) for c in cut]
    ends = [float(ix0 + int(ink[c]) + 1) for c in cut] + [b.x + b.w]
    return [dataclasses.replace(b, x=s, w=e - s) for s, e in zip(starts, ends) if e > s]


def _ink_beyond(g, iy0, iy1, bx0, bx1, wx0, wx1, floor=28, k_pct=40):
    """The ink of columns [wx0, wx1) over rows [iy0 - A, iy1 + A) (A = max(2, r(h, 25))), with the background and
    the thresholds of the line's own pixel box [bx0, bx1) x [iy0, iy1): bg the lower median of the border ring of the
    union of the line's box and the window, t and tc from the line's box (ink_mask's test). (mask, A)."""
    import tsdesk_refine as TR
    H, W = g.shape
    h = iy1 - iy0
    A = max(2, _r100(h, 25))
    ux0, ux1 = max(0, min(bx0, wx0) - 1), min(W, max(bx1, wx1) + 1)
    uy0, uy1 = max(0, iy0 - A), min(H, iy1 + A)
    win = g[uy0:uy1, ux0:ux1].astype(np.int16)
    ring = np.concatenate([win[0], win[-1], win[1:-1, 0], win[1:-1, -1]])
    bg = int(np.partition(ring, (ring.size - 1) // 2)[(ring.size - 1) // 2])
    d = np.abs(win - bg)
    c3 = TR._c3(g, ux0, uy0, ux1, uy1)
    ry0, rx0, rx1 = iy0 - uy0, bx0 - ux0, bx1 - ux0
    t = max(floor, _r100(TR.quantile(d[ry0:ry0 + h, rx0:rx1], 99), k_pct))
    tc = max(floor, _r100(TR.quantile(c3[ry0:ry0 + h, rx0:rx1], 99), k_pct))
    m0 = (d > t) & (c3 > tc)
    nb = np.zeros_like(m0)
    nb[1:] |= m0[:-1]
    nb[:-1] |= m0[1:]
    nb[:, 1:] |= m0[:, :-1]
    nb[:, :-1] |= m0[:, 1:]
    m = m0 & nb
    return m[:, wx0 - ux0:wx1 - ux0], ry0, uy1 - uy0


def _walk_dash(m, r0, r1, cy0, cy1, cols, G, Gr, small, low_small, D):
    """The end extension's walk over window columns `cols` (outward order) of mask m (rows [r0, r1) the line's).
    Dash columns hold ink inside the middle rows [cy0, cy1) only. From the box's edge the walk crosses at most G empty
    columns to the run and, before it, one small glyph (at most low_small columns, its ink in the lower half: 、 。);
    inside the run it crosses at most Gr empty columns (a solid ―― run, not the dots of …… or an ornament row); after
    at least D dash columns it takes one glyph of at most `small` columns (「 」 。) and ends. (the index into cols of
    the new edge or -1, ran out: the walk reached the window's end on the run)."""
    line = m[r0:r1]
    mid = (r1 - r0) // 2
    n = len(cols)

    def is_dash(col):
        return col[cy0:cy1].any() and not col[:cy0].any() and not col[cy1:].any()
    i = 0
    while i < min(n, small) and line[:, cols[i]].any() and not is_dash(line[:, cols[i]]):
        i += 1                                  # the edge glyph's own ink just outside the box (rounding)
    gap, dash, last = 0, 0, -1
    while i < n:
        col = line[:, cols[i]]
        if not col.any():
            gap += 1
            if gap > (Gr if dash else G):
                break
            i += 1
            continue
        gap = 0
        if is_dash(col):
            dash += 1
            last = i
            i += 1
            continue
        j = i                                   # a glyph: its run of ink columns
        while j < n and line[:, cols[j]].any():
            j += 1
        if dash >= D:
            if j - i <= small:
                last = j - 1
            return last, False
        rows = np.flatnonzero(line[:, cols[i:j]].any(1))
        if not dash and j - i <= low_small and len(rows) and rows[0] >= mid:
            i = j
            continue
        break
    else:
        return (last if dash >= D else -1), dash >= D and gap == 0
    return (last if dash >= D else -1), False


def extend_ends(g, boxes, o):
    """ext_pct: each horizontal box's ends extended over the solid dash run (――) beyond them on the frame's ink, up to
    r(h, ext_pct) px: from the end at most max(2, r(h, ext_gap_pct)) empty columns to the run (past a 、 or 。 before
    it), inside the run at most max(2, r(h, ext_run_pct)), and the 「 」 or 。 after it; a run shorter than
    r(h, ext_dash_pct) px, one that reaches the search window's end on ink, or one with ink in the rows just above or
    below it on more than a tenth of its columns (a frame's border, a texture: the window's max(2, r(h, 25)) rows, or
    max(1, r(h, ext_band_pct)) of them) is not taken. An end never grows into
    another box beside it (tools/tsdesk_refine's neighbour_limits); with ext_join an end that reaches that box (within
    2 px) joins the two (a line the decode found in two pieces either side of its dashes). The network gives long
    dashes background logits, so its boxes stop where the dashes start."""
    import dataclasses

    import tsdesk_refine as TR
    H, W = g.shape
    rows = [[b.x, b.y, b.x + b.w, b.y + b.h, b.vertical] for b in boxes]
    lims = TR.neighbour_limits(rows, W, H)
    out, reach = [], []
    for b, lim in zip(boxes, lims):
        ix0, iy0, ix1, iy1 = _pixel_box(b, W, H)
        h = iy1 - iy0
        reach.append((False, False))
        if b.vertical or h < 6 or ix1 - ix0 < 4:
            out.append(b)
            continue
        L = _r100(h, o["ext_pct"])
        G = max(2, _r100(h, o["ext_gap_pct"]))
        Gr = max(2, _r100(h, o["ext_run_pct"]))
        D = max(2, _r100(h, o["ext_dash_pct"]))
        q = _r100(h, 25)
        x0, x1 = b.x, b.x + b.w
        hit = [False, False]
        for side in (-1, 1):
            if side < 0:
                wx0, wx1 = max(0, int(lim[0]), ix0 - L), ix0
            else:
                wx0, wx1 = ix1, min(W, int(lim[2]), ix1 + L)
            if wx1 - wx0 < D:
                continue
            m, r0, nr = _ink_beyond(g, iy0, iy1, ix0, ix1, wx0, wx1)
            cols = list(range(wx1 - wx0 - 1, -1, -1)) if side < 0 else list(range(wx1 - wx0))
            k, ran_out = _walk_dash(m, r0, r0 + h, q, h - q, cols, G, Gr, _r100(h, 80), _r100(h, 60), D)
            limit = (wx0 > 0 and wx0 == ix0 - L) if side < 0 else (wx1 < W and wx1 == ix1 + L)
            if k < 0 or (ran_out and limit):
                continue
            c = cols[k]
            span = range(min(c, cols[0]), max(c, cols[0]) + 1)
            band = max(1, _r100(h, o["ext_band_pct"])) if o["ext_band_pct"] else nr     # the guard's rows
            bt, bb = max(0, r0 - band), min(nr, r0 + h + band)
            outer = np.concatenate([m[bt:r0, span], m[r0 + h:bb, span]]) if bb > r0 + h or r0 > bt else None
            if outer is not None and outer.size and int(outer.any(0).sum()) * 10 > len(span):
                continue
            if side < 0:
                x0 = float(wx0 + c)
                hit[0] = wx0 > 0 and wx0 == int(lim[0]) and c <= 2
            else:
                x1 = float(wx0 + c + 1)
                hit[1] = wx1 < W and wx1 == int(lim[2]) and c >= wx1 - wx0 - 3
        reach[-1] = tuple(hit)
        out.append(b if (x0, x1) == (b.x, b.x + b.w) else dataclasses.replace(b, x=x0, w=x1 - x0))
    if not o["ext_join"] or not any(a or z for a, z in reach):
        return out
    joined = list(out)
    alive = [True] * len(out)
    for i, (a, z) in enumerate(reach):
        if not (a or z) or not alive[i]:
            continue
        bi = joined[i]
        for j in range(len(joined)):
            if j == i or not alive[j] or joined[j].vertical:
                continue
            bj = joined[j]
            dy = min(bi.y + bi.h, bj.y + bj.h) - max(bi.y, bj.y)
            if dy < 0.5 * min(bi.h, bj.h):
                continue
            touch_r = z and bj.x >= bi.x and bj.x - (bi.x + bi.w) <= 2         # touching or overlapping
            touch_l = a and bj.x + bj.w <= bi.x + bi.w and bi.x - (bj.x + bj.w) <= 2
            if touch_r or touch_l:
                x0, y0 = min(bi.x, bj.x), min(bi.y, bj.y)
                x1, y1 = max(bi.x + bi.w, bj.x + bj.w), max(bi.y + bi.h, bj.y + bj.h)
                big = bi if bi.w >= bj.w else bj
                bi = dataclasses.replace(big, x=x0, y=y0, w=x1 - x0, h=y1 - y0, cells=bi.cells + bj.cells)
                joined[i] = bi
                alive[j] = False
    return [b for b, keep in zip(joined, alive) if keep]


def post_post(g, boxes, opts):
    """The post-refinement options (above) on frame-px Box objects of grey frame g."""
    o = post_options(opts)
    out = list(boxes)
    if o["ext_pct"]:
        out = extend_ends(g, out, o)
    return out


def post_pre(g, boxes, opts):
    """The pre-refinement options (above) on frame-px Box objects of grey frame g."""
    o = post_options(opts)
    out = list(boxes)
    if o["vflip_pct"]:
        out = vflip(g, out, o["vflip_pct"], o["vflip_join_pct"], o["vflip_min"], o["vflip_h_pct"])
    if o["merge_pct"]:
        out = merge_overlaps(out, o["merge_pct"])
    if o["touch_pct"]:
        out = merge_touching(out, o["touch_pct"])
    if o["join_pct"]:
        out = join_row(out, o["join_pct"], o["join_h_pct"], o["join_ov_pct"])
    if o["rows_pct"]:
        out = [p for b in out for p in split_rows(g, b, o["rows_pct"], o["valley_pct"], o["rows_len"])]
    if o["gap_pct"]:
        out = [p for b in out for p in split_along(g, b, o["gap_pct"], o["gap_weak"])]
    return out[:MAX_BOXES]


# ----------------------------------------------------------------------------- the recognizer as a junk filter

# a Japanese character: kana, kanji, CJK symbols and punctuation, full-width forms, …, ― and —
JAPANESE = "\u3000-\u303f\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\uff01-\uff60\u2025\u2026\u2015\u2014"
DROP_DEFAULTS = {"nojp": 0, "nojp_gap_pct": 100, "nojp_anchor_pct": 0, "rep_pct": 0, "conf_milli": 0, "rep_len": 4,
                 "cut_pct": 0, "cut_rel_pct": 200, "cut_dlg": 1, "nojp_num_pct": 0,
                 "trim_milli": 0, "trim_len": 2, "trim_gap_pct": 25, "lead_drop": 0,
                 "garb_drop": 0, "garb_len": 5, "garb_pct": 60}
# cut_reading's marks: no cut on either side of a punctuation mark (a phrase's end, a bracket, a dash, a dot: the
# gap beside it is the sentence's own, and the mark belongs to the text it closes or opens)
CUT_MARKS = set("、。，．,.！？!?…‥」』）)】〉》〕］]｝}ー―～〜：:；;・「『（(【〈《〔［[｛{")
_JP_RE = None


def cut_reading(g, b, chars, pct, rel_pct=200):
    """cut_pct (a reading-stage option): a read horizontal line cut into items at its wide ink gaps, by its characters'
    positions: a label and its value, menu entries, an icon beside a label. b: the box [x0, y0, x1, y1, ...] in frame
    px; chars: [(text, x)] the reading's characters in order with their centres in frame px. A gap is the longest run
    of empty ink columns (ink_mask over the box) lying between two consecutive characters' centres; it cuts the line
    when it is at least max(2, pct % of the height) and at least rel_pct % of the line's median such gap (a wide-set
    font keeps its lines), unless either character is a punctuation mark (CUT_MARKS). A run as long between the box's
    first (last) ink and the first (last) character's centre trims ink that read as no character (an icon). None (no
    cut, no trim) or [(x0, x1, text)] in frame px: every character in the piece between its neighbouring cuts (the
    first and last pieces take every character before and after them); the first piece keeps the box's start and the
    last its end unless trimmed; the edges at a cut lie on the ink either side of the gap."""
    H, W = g.shape
    ix0 = min(max(int(math.floor(b[0])), 0), W)
    iy0 = min(max(int(math.floor(b[1])), 0), H)
    ix1 = min(max(int(math.ceil(b[2])), 0), W)
    iy1 = min(max(int(math.ceil(b[3])), 0), H)
    h = iy1 - iy0
    if h < 4 or ix1 - ix0 < 4 or not chars:
        return None
    ink = np.flatnonzero(ink_mask(g, ix0, iy0, ix1, iy1).any(0))
    if len(ink) < 2:
        return None
    runs = [(int(ink[j]) + 1, int(ink[j + 1])) for j in np.flatnonzero(np.diff(ink) > 1)]    # empty, box px

    def widest(a, e):
        best = None
        for r in runs:
            if r[0] >= a and r[1] <= e and (best is None or r[1] - r[0] > best[1] - best[0]):
                best = r
        return best

    xs = [x - ix0 for _, x in chars]
    gaps = [widest(xs[i], xs[i + 1]) for i in range(len(chars) - 1)]
    lens = sorted((r[1] - r[0]) if r else 0 for r in gaps)
    med = lens[(len(lens) - 1) // 2] if lens else 0
    thr = max(2, _r100(h, pct), -(-int(med) * int(rel_pct) // 100))
    cuts = [(i, r) for i, r in enumerate(gaps) if r and r[1] - r[0] >= thr
            and chars[i][0] not in CUT_MARKS and chars[i + 1][0] not in CUT_MARKS]
    lead = widest(int(ink[0]), xs[0])
    tail = widest(xs[-1], int(ink[-1]) + 1)
    lead = lead if lead and lead[1] - lead[0] >= thr else None
    tail = tail if tail and tail[1] - tail[0] >= thr else None
    if not cuts and not lead and not tail:
        return None
    starts = [float(ix0 + lead[1]) if lead else float(b[0])] + [float(ix0 + r[1]) for _, r in cuts]
    ends = [float(ix0 + r[0]) for _, r in cuts] + [float(ix0 + tail[0]) if tail else float(b[2])]
    bounds = [0] + [i + 1 for i, _ in cuts] + [len(chars)]                # the characters of each piece
    out = []
    for k, (s0, e0) in enumerate(zip(starts, ends)):
        t = "".join(c for c, _ in chars[bounds[k]:bounds[k + 1]])
        if t and e0 > s0:
            out.append((s0, e0, t))
    return out


def trim_ends(g, b, chars, confs, o):
    """trim_milli: a line's first (last) at most trim_len characters dropped when an ink gap of at least
    max(2, trim_gap_pct % of the box's pixel height) parts them from the rest, their mean confidence is under
    trim_milli / 1000 and the rest's is not (an icon read as a character: ゲ、 before グレナダ). b: the tight box
    [x0, y0, x1, ...]; chars: [(text, x)] with confs their probabilities. None (no trim), or (x0, x1, kept chars,
    kept confs): the box's new ends (on the ink beside the gap; the other end as it was)."""
    H, W = g.shape
    n = len(chars)
    if n < 2 or not o["trim_milli"]:
        return None
    ix0 = min(max(int(math.floor(b[0])), 0), W)
    iy0 = min(max(int(math.floor(b[1])), 0), H)
    ix1 = min(max(int(math.ceil(b[2])), 0), W)
    iy1 = min(max(int(math.ceil(b[3])), 0), H)
    h = iy1 - iy0
    if h < 4 or ix1 - ix0 < 4:
        return None
    ink = np.flatnonzero(ink_mask(g, ix0, iy0, ix1, iy1).any(0))
    runs = [(int(ink[j]) + 1, int(ink[j + 1])) for j in np.flatnonzero(np.diff(ink) > 1)]
    thr = max(2, _r100(h, o["trim_gap_pct"]))
    lim = o["trim_milli"] / 1000.0

    def gap(i):                      # the widest run between characters i and i + 1 at least thr long, or None
        a, e = chars[i][1] - ix0, chars[i + 1][1] - ix0
        best = None
        for r in runs:
            if r[0] >= a and r[1] <= e and r[1] - r[0] >= thr and (best is None or r[1] - r[0] > best[1] - best[0]):
                best = r
        return best

    def mean(v):
        return sum(v) / len(v) if v else 1.0

    lo, hi, x0, x1 = 0, n, float(b[0]), float(b[2])
    for k in range(1, min(o["trim_len"], n - 1) + 1):            # the leading k characters
        r = gap(k - 1)
        if r and mean(confs[:k]) < lim <= mean(confs[k:]):
            lo, x0 = k, float(ix0 + r[1])
            break
    for k in range(1, min(o["trim_len"], n - 1 - lo) + 1):       # the trailing k characters
        r = gap(n - k - 1)
        if r and mean(confs[n - k:]) < lim <= mean(confs[lo:n - k]):
            hi, x1 = n - k, float(ix0 + r[0])
            break
    if lo == 0 and hi == n:
        return None
    return x0, x1, chars[lo:hi], confs[lo:hi]


def cut_pieces(g, b, tag, chars, o, confs=None):
    """The reading stage's cut of one read horizontal line under reading-filter options o (drop_options(): cut_pct,
    cut_rel_pct, cut_dlg; trim_milli with confs): cut_reading's pieces [(x0, x1, text)], or None (the line stays
    whole: the cut off, no cut found, or a line the detector tags as dialogue (tag 0) with cut_dlg 0, or with
    cut_dlg N >= 2 unless every piece holds at most N characters). With trim_milli, trim_ends first drops an end
    read with low confidence across an ink gap (any tag), and the cut runs on the rest. b: the tight box [x0, y0,
    x1, y1] in frame px; chars: [(text, x)] the reading's characters with their centres in frame px; confs their
    probabilities."""
    trimmed = None
    if o["trim_milli"] and confs is not None:
        trimmed = trim_ends(g, b, chars, confs, o)
        if trimmed is not None:
            x0, x1, chars, _ = trimmed
            b = [x0, b[1], x1, b[3]] + list(b[4:])
    cut = None
    if o["cut_pct"]:
        cut = cut_reading(g, b, chars, o["cut_pct"], o["cut_rel_pct"])
        if cut is not None and tag == 0 and o["cut_dlg"] != 1:
            if o["cut_dlg"] == 0 or max(len(t) for _, _, t in cut) > o["cut_dlg"]:
                cut = None
    if cut is None and trimmed is not None:
        cut = [(b[0], b[2], "".join(t for t, _ in chars))]
    return cut


def drop_options(opts=None):
    o = dict(DROP_DEFAULTS)
    for k, v in (opts or {}).items():
        if k not in DROP_DEFAULTS:
            raise KeyError(f"unknown reading-filter option {k!r}")
        o[k] = int(v)
    return o


def _jp_re():
    global _JP_RE
    if _JP_RE is None:
        import re
        _JP_RE = re.compile(f"[{JAPANESE}]")
    return _JP_RE


def has_japanese(text):
    return bool(_jp_re().search(text))


def japanese_share(text):
    """The share of a reading's characters that are Japanese (JAPANESE), 0 for the empty reading."""
    return len(_jp_re().findall(text)) / len(text) if text else 0.0


# lead_drop's leader marks: dots, rules and dashes a table draws between a label and its value
LEADER_MARKS = set("…‥・･ーｰ―─━－-‐‑–—_＿.．·•~〜～=＝")


def is_leader(text):
    """A reading made only of leader marks (LEADER_MARKS), spaces aside."""
    t = "".join(text.split())
    return bool(t) and all(c in LEADER_MARKS for c in t)


def _char_kind(c):
    if "0" <= c <= "9" or "０" <= c <= "９" or ("a" <= c.lower() <= "z" and c.isascii()) or "Ａ" <= c <= "Ｚ" or \
            "ａ" <= c <= "ｚ":
        return "al"
    if "\u3041" <= c <= "\u3096" or "\u30a1" <= c <= "\u30fa":
        return "kana"
    if "\u4e00" <= c <= "\u9fff" or "\u3400" <= c <= "\u4dbf" or c in "々〆ヶ":
        return "kanji"
    return "other"


def is_garbage(text, min_len=5, pct=60):
    """garb_drop's reading: at least min_len characters (spaces aside), at least pct % of them digits or Latin
    letters, at least one kana, no kanji, and no two kana side by side (an ornament strip or a logo read as
    エ100い9, しL1O00, いnいnmnw: a real value carries a kanji unit, 100円, and real words keep their kana together)."""
    t = "".join(text.split())
    if len(t) < max(1, min_len):
        return False
    ks = [_char_kind(c) for c in t]
    if ks.count("al") * 100 < pct * len(t) or "kanji" in ks or "kana" not in ks:
        return False
    return all(not (a == "kana" and b == "kana") for a, b in zip(ks, ks[1:]))


def is_numeric(text):
    """A value: at least one digit (ASCII or full-width), and digits at least half of the characters."""
    t = "".join(text.split())
    n = sum(1 for c in t if "0" <= c <= "9" or "０" <= c <= "９")
    return n > 0 and 2 * n >= len(t)


def drop_set(items, o):
    """The indices of the readings the filter drops, for options o (drop_options(); every rule off by default).
    items: [(box (x, y, w, h), text, conf_mean[, tag])] of a frame's horizontal boxes, read (the detector's tag: 0
    dialogue; a 3-tuple counts as dialogue).
      garb_drop 1: a garbage reading (is_garbage with garb_len, garb_pct), any tag (the ornament strips it targets
              sit in dialogue boxes);
      lead_drop 1: a reading made only of leader marks (is_leader: ……, ・・・, ――) whose box is not tagged as dialogue
              (a table's leader dots or rule read alone; hval's …… lines are dialogue); it still anchors for nojp
              2 (the value after a leader stays);
      nojp 1: a reading with no Japanese character (JAPANESE; the empty reading too);
      nojp 2: the same, unless the reading is not empty and a reading with one sits in the same row (vertical
              overlap >= half the smaller height)
              within a gap of nojp_gap_pct % of the larger height along it (a line's own Latin or digit part: "A"
              before スポーツ), and at least nojp_anchor_pct % of that reading's characters are Japanese (an
              ornament misread as 「000OO0 does not anchor its neighbours);
              with nojp_num_pct, a numeric reading (digits at least half its characters: a value such as 33 or
              100/100) is kept by a reading with one in its row within nojp_num_pct % instead (a stats table's
              right-aligned values);
      rep_pct: a reading of at least rep_len characters (4) whose most frequent character is not Japanese and makes
              up at least rep_pct % of it (an ornament strip read as イf0u000u0000000c; no reference line of hval, the
              dense sets or dev-v2 is one, at rep_len 3 either; at 2, 6億 and 7章 are);
      conf_milli: a reading whose mean character confidence (runtime/vocr.c's conf_mean) is under conf_milli / 1000."""
    from collections import Counter
    out = set()
    tags = [it[3] if len(it) > 3 else 0 for it in items]
    items = [it[:3] for it in items]
    lead = [bool(o["lead_drop"]) and tg != 0 and is_leader(t) for (_, t, _), tg in zip(items, tags)]
    jp = [has_japanese(t) for _, t, _ in items]
    anchor = [j and japanese_share(t) * 100.0 >= o["nojp_anchor_pct"] for j, (_, t, _) in zip(jp, items)]
    for i, (b, t, c) in enumerate(items):
        if lead[i] or (o["garb_drop"] and is_garbage(t, o["garb_len"], o["garb_pct"])):
            out.add(i)
            continue
        if o["conf_milli"] and c * 1000.0 < o["conf_milli"]:
            out.add(i)
            continue
        if o["rep_pct"] and len(t) >= max(1, o["rep_len"]):
            ch, k = Counter(t).most_common(1)[0]
            if not has_japanese(ch) and k * 100 >= o["rep_pct"] * len(t):
                out.add(i)
                continue
        if not o["nojp"] or jp[i]:
            continue
        if o["nojp"] >= 2 and t:
            x, y, w, h = b
            keep = False
            allow = o["nojp_gap_pct"]
            if o["nojp_num_pct"] and is_numeric(t):
                allow = max(allow, o["nojp_num_pct"])
            for j, (b2, _, _) in enumerate(items):
                if j == i or not anchor[j]:
                    continue
                x2, y2, w2, h2 = b2
                if min(y + h, y2 + h2) - max(y, y2) < 0.5 * min(h, h2):
                    continue
                gap = max(x2 - (x + w), x - (x2 + w2))
                if gap * 100 <= allow * max(h, h2):
                    keep = True
                    break
            if keep:
                continue
        out.add(i)
    return out


class Detector:
    """textspot-desktop's detector (an ONNX file train/tsdesk_det.py exports)."""

    def __init__(self, path, threads=1, providers=None, session_options=None, size=None, decode_opts=None,
                 resize_mode="exact", refine=None, post=None):
        if resize_mode not in ("exact", "cv2"):
            raise ValueError(f"resize_mode {resize_mode!r}: exact (the C path's integer resize) or cv2")
        self.resize_mode = resize_mode
        # refine: None (off), True (tools/tsdesk_refine.DEFAULTS) or a dict of its options: every horizontal box's
        # edges snapped to its line's ink on the full-resolution grey frame after the decode
        self.refine = None if refine in (None, False) else ({} if refine is True else dict(refine))
        # post: None (off) or a dict of POST_DEFAULTS options (the post-pass; every option 0 = off)
        self.post = post_options(post) if post else None
        self.sess = _session(path, threads, providers, session_options)
        meta = {k: json.loads(v) for k, v in self.sess.get_modelmeta().custom_metadata_map.items()
                if k.startswith("vocr.")}
        if meta.get("vocr.kind") != "textspot-desktop-detector":
            raise ValueError(f"{path}: not a textspot-desktop detector")
        self.meta = meta
        self.size = size or meta.get("vocr.size", "fit:960")
        self.stride = int(meta["vocr.stride"])
        self.tags = int(meta.get("vocr.tags", 4))
        d = dict(DEFAULTS, **DECODE_EXTRA)
        d.update({k: v for k, v in (meta.get("vocr.decode") or {}).items() if k in DEFAULTS})
        bad = set(decode_opts or {}) - set(d)
        if bad:
            raise KeyError(f"unknown decode options {sorted(bad)}")
        d.update(decode_opts or {})
        self.opts = d
        self.input_name = self.sess.get_inputs()[0].name

    def preprocess(self, g):
        """(network input [1, 1, H', W'] uint8, (fx, fy, iw, ih)) of a grey frame: resized by the size policy with
        the C path's integer resize (resize(): area weights down, bilinear up), padded right and bottom to a multiple
        of 32 by edge replication."""
        h, w = g.shape
        iw, ih = input_size(self.size, w, h)
        if (iw, ih) != (w, h):
            if self.resize_mode == "cv2":      # an app's resize: within a grey level of the integer one, faster here
                import cv2
                g = cv2.resize(g, (iw, ih), interpolation=cv2.INTER_AREA if iw < w else cv2.INTER_LINEAR)
            else:
                g = resize(g, iw, ih)
        ph, pw = -ih % ALIGN, -iw % ALIGN
        if ph or pw:
            g = np.pad(g, ((0, ph), (0, pw)), mode="edge")
        return np.ascontiguousarray(g, np.uint8)[None, None], (iw / w, ih / h, iw, ih)

    def maps(self, x):
        return self.sess.run(None, {self.input_name: x})[0][0]

    def detect(self, img):
        """[Box] in the frame's px (tight ink boxes), in the decode's order (by confidence)."""
        g = grey_of(img)
        x, (fx, fy, iw, ih) = self.preprocess(g)
        o = self.opts
        maps = self.maps(x)
        boxes = decode(maps, iw, ih, self.stride, o["min_cells"], o["conf_min_logit"], o["min_across"],
                       o["along_reach"], self.tags, o["max_boxes"], int(o["saddle_pct"]), int(o["vote_pct"]),
                       int(o["rescue_milli"]), int(o["clamp_ends"]), int(o["vote_depth"]), int(o["along_pct"]),
                       int(o["along_depth"]))
        H, W = g.shape
        out = []
        for b in boxes:
            x0, y0 = max(0.0, b.x / fx), max(0.0, b.y / fy)
            x1, y1 = min(float(W), (b.x + b.w) / fx), min(float(H), (b.y + b.h) / fy)
            if x1 > x0 and y1 > y0:
                out.append(Box(x0, y0, x1 - x0, y1 - y0, b.vertical, b.logit, b.tag, b.tconf, b.cells))
        if self.post is not None and any(self.post[k] for k in POST_PRE):
            out = post_pre(g, out, self.post)
        if self.refine is not None:
            import tsdesk_refine
            out = tsdesk_refine.refine_det(g, out, self.refine)
        if self.post is not None and any(self.post[k] for k in POST_POST):
            out = post_post(g, out, self.post)
        if self.post is not None and self.post["ksplit_pct"]:
            p = self.post
            out = split_kgap(g, out, maps[0], fx, fy, self.stride, p["ksplit_pct"], p["ksplit_k"], p["ksplit_min_pct"])
        return out

    def boxes(self, rgb):
        """eval/desktop.py's detector interface: [[x0, y0, x1, y1, vertical, logit, tag]] in frame px."""
        return [[b.x, b.y, b.x + b.w, b.y + b.h, b.vertical, round(b.logit, 4), b.tag] for b in self.detect(rgb)]


READ_PAD = "k6v4"          # the reading pad the protocol chose on exact boxes (#173 amendment 1)


def pad_of(pad, H):
    """The reading pad in frame px for an H px tall frame: "kN" = max(2, round(N H / 544)) on every side, "kAvB" = A
    horizontally and B vertically (a (horizontal, vertical) tuple; "k6v4", the default, is textspot-desktop's), "auto"
    = k4 (eval/desktop.py's), or an integer."""
    if pad is None:
        pad = READ_PAD
    if pad == "auto":
        pad = "k4"
    s = str(pad)
    if s.startswith("k"):
        a, _, b = s[1:].partition("v")
        ph, pv = max(2, int(round(float(a) * H / 544))), max(2, int(round(float(b or a) * H / 544)))
        return ph if ph == pv else (ph, pv)
    return int(pad)


def crop_of(b, p, W, H):
    """A float box (x, y, w, h) grown by p px (or a (horizontal, vertical) pair), rounded outward and clipped: integer
    (x, y, w, h), or None."""
    ph, pv = p if isinstance(p, tuple) else (p, p)
    x0, y0 = max(0, int(math.floor(b[0] - ph))), max(0, int(math.floor(b[1] - pv)))
    x1, y1 = min(W, int(math.ceil(b[0] + b[2] + ph))), min(H, int(math.ceil(b[1] + b[3] + pv)))
    return (x0, y0, x1 - x0, y1 - y0) if x1 > x0 and y1 > y0 else None


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


def read_frame(frame, det, rec, pad=READ_PAD, lm=None, tags=None, workers=1, executor=None, drop=None):
    """Every line of a frame, top to bottom: [Line] in frame px. det: a Detector; rec: a vocr_onnx.Recognizer;
    lm: a vocr_onnx.CharLM; tags: the tag classes to read (None: all); workers: lines read on that many threads;
    drop: None (off) or DROP_DEFAULTS options: with cut_pct, each read horizontal line is first cut into items
    (cut_pieces: by its characters' positions, the greedy ones, or the LM text's placed on them), each piece a line
    in its place with its own text, tight box (x0, y, x1 - x0, h) and the pad's crop of it, while the frame keeps
    at most MAX_BOXES lines (a line whose pieces would pass that stays whole: the C page's room); then the
    horizontal lines whose readings drop_set() rejects are left out."""
    import vocr_onnx as V
    g = grey_of(frame)
    H, W = g.shape
    dropo = drop_options(drop) if drop else None
    boxes = det.detect(g)
    if tags is not None:
        boxes = [b for b in boxes if b.tag in tags]
    p = pad_of(pad, H)
    order = reading_order([(b.x, b.y, b.w, b.h) for b in boxes])
    jobs = []
    for i in order:
        b = boxes[i]
        cb = crop_of((b.x, b.y, b.w, b.h), p, W, H)
        if cb is not None:
            jobs.append((b, cb))

    def one(job):
        b, (cx, cy, cw, ch) = job
        tight = (b.x, b.y, b.w, b.h)
        if b.vertical:
            return Line("", tight, (cx, cy, cw, ch), b.tag, b.logit, 0, 1)
        dark = V.line_dark(g, (cx, cy, cw, ch))
        r = rec.read(g[cy:cy + ch, cx:cx + cw], invert=bool(dark), lm=lm)
        chars = [(c.text, cx + c.x) for c in (r.text_chars if lm is not None else r.chars)]
        return Line(r.text, tight, (cx, cy, cw, ch), b.tag, b.logit, int(dark), 0), r.conf_mean, chars

    if executor is not None and len(jobs) > 1:
        res = list(executor.map(one, jobs))
    elif workers > 1 and len(jobs) > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=workers) as ex:
            res = list(ex.map(one, jobs))
    else:
        res = [one(j) for j in jobs]
    if dropo is None:
        return [r if isinstance(r, Line) else r[0] for r in res]
    return cut_and_filter(g, res, dropo, lambda tb: crop_of(tb, p, W, H))


def cut_read_lines(g, res, drop, crop_fn):
    """The reading stage's cut over a frame's read lines (read_frame's step; the C page's twin: desktop/
    vocr_tsd_page.c cut_lines). res: in reading order, a vertical box as its Line, a read horizontal line as
    (Line, conf_mean, chars) with chars [(text, x)] its reading's characters and their centres in frame px (from a
    vocr_onnx read result r of the crop at crop x cx: [(c.text, cx + c.x) for c in r.chars], or r.text_chars with an
    LM); chars None keeps the line whole. drop: DROP_DEFAULTS options (cut_pct, cut_rel_pct, cut_dlg: cut_pieces);
    crop_fn(tight box (x, y, w, h)) -> a piece's crop under the reading rule. Each cut line is replaced in its place
    by its pieces (Line(text, (x0, y, x1 - x0, h), crop_fn(that box), tag, logit, dark, 0), conf_mean, None), while
    the frame keeps at most MAX_BOXES lines (a line whose pieces would pass that stays whole)."""
    o = drop_options(drop)
    if not o["cut_pct"]:
        return list(res)
    out, left = [], len(res)
    for r in res:
        left -= 1
        if isinstance(r, Line) or len(r) < 3 or r[2] is None:
            out.append(r)
            continue
        ln, conf, chars = r[0], r[1], r[2]
        bx, by, bw, bh = ln.box
        cut = cut_pieces(g, [bx, by, bx + bw, by + bh], ln.tag, chars, o)
        if cut is None or len(out) + len(cut) + left > MAX_BOXES:
            out.append(r)
            continue
        for x0, x1, t in cut:
            tb = (x0, by, x1 - x0, bh)
            out.append((Line(t, tb, crop_fn(tb), ln.tag, ln.logit, ln.dark, 0), conf, None))
    return out


def cut_and_filter(g, res, drop, crop_fn):
    """read_frame's last steps on a frame's read lines (res and crop_fn as cut_read_lines): the cut (with cut_pct),
    then the reading filter (drop_set over the horizontal lines' (box, text, conf_mean)). [Line] kept, in order."""
    o = drop_options(drop)
    res = cut_read_lines(g, res, o, crop_fn)
    hz = [k for k, r in enumerate(res) if not isinstance(r, Line)]
    gone = {hz[i] for i in drop_set([(res[k][0].box, res[k][0].text, res[k][1], res[k][0].tag) for k in hz], o)}
    return [r if isinstance(r, Line) else r[0] for k, r in enumerate(res) if k not in gone]


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
    ap.add_argument("--det", required=True, help="textspot-desktop's ONNX file")
    ap.add_argument("--rec", required=True, help="a VOCR6 recognizer's ONNX file (VOCR6.2_L_fast.onnx)")
    ap.add_argument("--lm", help="a VOCR character LM (.vlm): beam 8 + the LM at Rubi's settings")
    ap.add_argument("--size", help="the detector's input policy: fit:N, long:N or native (default: the file's)")
    ap.add_argument("--pad", default="k6v4", help="the reading pad: k6v4 (the default), kN, auto or N frame px")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--refine", action="store_true", help="tools/tsdesk_refine.py on the boxes")
    ap.add_argument("--decode", help="JSON: decode options (DEFAULTS, DECODE_EXTRA), e.g. '{\"vote_pct\": 70}'")
    ap.add_argument("--post", help="JSON: post-pass options (POST_DEFAULTS)")
    ap.add_argument("--drop", help="JSON: reading-filter options (DROP_DEFAULTS)")
    ap.add_argument("images", nargs="+")
    a = ap.parse_args(argv)
    det = Detector(a.det, threads=1, size=a.size, refine=True if a.refine else None,
                   decode_opts=json.loads(a.decode) if a.decode else None, post=json.loads(a.post) if a.post else None)
    drop = json.loads(a.drop) if a.drop else None
    rec = V.Recognizer(a.rec, threads=1)
    lm = V.CharLM(a.lm, tier=rec.name[-1:] if rec.name else None) if a.lm else None
    for p in a.images:
        img = _load_image(p)
        t0 = time.perf_counter()
        lines = read_frame(img, det, rec, pad=a.pad, lm=lm, workers=a.threads, drop=drop)
        ms = (time.perf_counter() - t0) * 1e3
        print(json.dumps({"image": p, "ms": round(ms, 1), "lines": [
            {"text": ln.text, "box": [round(v, 1) for v in ln.box], "crop": list(ln.crop), "tag": CLASSES[ln.tag],
             "vertical": ln.vertical} for ln in lines]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
