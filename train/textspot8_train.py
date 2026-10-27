"""Train one textspot8 run (vocr6-spotter2, docs/textspot8.md, eval/textspot8_protocol.json): textspot6's tagged
full-frame spotter (train/textspot6_train.py: the same networks, VTSP version-4 blobs, data sources, schedule, QAT and
export) with the changes textspot8 registers, each a setting in T8 (recorded in run.json):

  edge kernel   the K weight of the kernel cells at either end of a line along it (the first and last `edge_cells`
                cells, every cell across) is multiplied by `edge_weight`: the line ends that textspot6's spotters cut
                (a leading 「 or ―, a trailing 」 。 … ― 、 ！: on hval 231 of base_t64n8c16_seed1's 306 cut lines
                lack one end, mostly one character)
  across        the A1 / A2 targets (the line's extent across it) are moved inward by `across_shrink` network px at
                each side (never below a quarter of the size across), so the decoded box (A1 / A2 plus the decode's
                18% pad) lands nearer GRAD's crop margin
  icons         with p `icon_p` per window, 1-2 synthetic animated-icon-like marks (spinner dot rings and arcs, page
                cursors (filled and outlined triangles), a four-point star, a diamond, a ring, an hourglass) are
                drawn just after the end of a positive line or on free background; they are background (K = 0),
                never over a positive or an ignore area
  tight         with p `tight_p` per window, the window is cut to a block of positives plus a margin of 0-0.4 of the
                line height (the region a player draws tight to the text, as the crop decode reads it)
  zoom / blend  textspot6's zoom and translucent-box blend, wider: zoom p and range, blend p and alpha

  textspot8_train.py --recipe t8 --shape t64n8c16 --seed N --out RUN (textspot6_train's other arguments)

Everything else is textspot6_train's (its docstring), and its outputs are the same files (final.pt, final.vtsp, ...).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "train", ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textdet_train as TT  # noqa: E402
import textspot6_train as B  # noqa: E402

T8 = {
    "edge_cells": 1,
    "edge_weight": 4.0,
    "across_shrink": 0.0,
    "icon_p": 0.35,
    "icon_max": 2,
    "icon_size": [0.45, 1.1],
    "tight_p": 0.15,
    "tight_margin": [0.0, 0.4],
    "zoom_p": 0.45,
    "zoom": [0.6, 1.6],
    "blend_p": 0.2,
    "blend_alpha": [0.45, 0.9],
}
RECIPES = {"t8": {"steps": 30000, "lr": 4e-3, "warmup": 500, "mix": "base"}}
ICON_KINDS = ("spinner_dots", "spinner_arc", "cursor_down", "cursor_down_outline", "cursor_right", "star4",
              "diamond", "ring", "hourglass")


# ----------------------------------------------------------------------------- targets

_base_targets = B.targets


def targets(lines, ignore, w, h, s=B.MAP_STRIDE):
    """textspot6_train.targets with the edge-kernel weights and the across shrink (module docstring). A line end
    cut by the window's border is not a line end: only an end inside the valid area is weighted."""
    ho, wo = B.WIN_H // s, B.WIN_W // s
    shrink = float(T8["across_shrink"])
    K, O, A1, A2, valid_k, kern, Wt, Mk = _base_targets(lines, ignore, w, h, s)  # noqa: E741
    if shrink <= 0 and T8["edge_weight"] == 1.0:
        return K, O, A1, A2, valid_k, kern, Wt, Mk
    centres = np.arange(max(ho, wo)) * s + (s - 1) / 2
    e = int(T8["edge_cells"])
    for ln in lines:
        x0, y0, x1, y1 = ln[:4]
        vert = bool(ln[4])
        a0, a1, b0, b1 = (x0, x1, y0, y1) if vert else (y0, y1, x0, x1)
        lim_a, lim_b = (w, h) if vert else (h, w)
        if a1 <= 0 or a0 >= lim_a or b1 <= 0 or b0 >= lim_b or a0 < 0 or a1 > lim_a:
            continue
        real_start, real_end = b0 >= 0, b1 <= lim_b
        b0, b1 = max(0, b0), min(lim_b, b1)
        c = (a0 + a1) / 2
        r = max((a1 - a0) / 4, s / 2)
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
        own = kern[ii, jj]
        ii, jj = ii[own], jj[own]
        if not len(ii):
            continue
        if shrink > 0:
            d = min(shrink, (a1 - a0) / 4.0)
            cc = centres[jj] if vert else centres[ii]
            A1[ii, jj] = cc - (a0 + d)
            A2[ii, jj] = (a1 - d) - cc
        if T8["edge_weight"] != 1.0:
            ends = set()
            if real_start:
                ends |= set(along[:e].tolist())
            if real_end:
                ends |= set(along[-e:].tolist())
            if ends:
                sel = np.isin(ii if vert else jj, list(ends))
                Wt[ii[sel], jj[sel]] *= float(T8["edge_weight"])
    return K, O, A1, A2, valid_k, kern, Wt, Mk


# ----------------------------------------------------------------------------- icons

def draw_icon(img, kind, cx, cy, size, value, rng):
    """One synthetic icon mark on the uint8 window img (in place), centred at (cx, cy), size px across."""
    import cv2
    s = max(4.0, float(size))
    r = s / 2.0
    v = int(value)
    th = max(1, int(round(s / 8)))
    c = (int(round(cx)), int(round(cy)))
    if kind == "spinner_dots":
        n = int(rng.integers(6, 13))
        phase = rng.uniform(0, 2 * math.pi)
        for k in range(n):
            a = phase + 2 * math.pi * k / n
            fade = 0.35 + 0.65 * ((k + 1) / n)
            val = int(round(128 + (v - 128) * fade))
            p = (int(round(cx + 0.7 * r * math.cos(a))), int(round(cy + 0.7 * r * math.sin(a))))
            cv2.circle(img, p, max(1, int(round(r * 0.18))), val, -1, lineType=cv2.LINE_AA)
    elif kind == "spinner_arc":
        a0 = rng.uniform(0, 360)
        cv2.ellipse(img, c, (max(2, int(r * 0.8)), max(2, int(r * 0.8))), 0, a0, a0 + rng.uniform(90, 300), v,
                    th + 1, lineType=cv2.LINE_AA)
    elif kind in ("cursor_down", "cursor_down_outline", "cursor_right"):
        if kind == "cursor_right":
            pts = np.array([[cx - 0.5 * r, cy - 0.7 * r], [cx + 0.7 * r, cy], [cx - 0.5 * r, cy + 0.7 * r]])
        else:
            pts = np.array([[cx - 0.7 * r, cy - 0.45 * r], [cx + 0.7 * r, cy - 0.45 * r], [cx, cy + 0.65 * r]])
        pts = np.round(pts).astype(np.int32)
        if kind == "cursor_down_outline":
            cv2.polylines(img, [pts], True, v, th, lineType=cv2.LINE_AA)
        else:
            cv2.fillPoly(img, [pts], v, lineType=cv2.LINE_AA)
    elif kind == "star4":
        pts = []
        for k in range(8):
            a = math.pi / 4 * k - math.pi / 2
            rr = r * (0.95 if k % 2 == 0 else 0.3)
            pts.append([cx + rr * math.cos(a), cy + rr * math.sin(a)])
        cv2.fillPoly(img, [np.round(np.array(pts)).astype(np.int32)], v, lineType=cv2.LINE_AA)
    elif kind == "diamond":
        pts = np.round(np.array([[cx, cy - 0.8 * r], [cx + 0.6 * r, cy], [cx, cy + 0.8 * r], [cx - 0.6 * r, cy]]))
        cv2.fillPoly(img, [pts.astype(np.int32)], v, lineType=cv2.LINE_AA)
    elif kind == "ring":
        cv2.circle(img, c, max(2, int(r * 0.75)), v, th + 1, lineType=cv2.LINE_AA)
    elif kind == "hourglass":
        pts = np.round(np.array([[cx - 0.6 * r, cy - 0.8 * r], [cx + 0.6 * r, cy - 0.8 * r], [cx - 0.6 * r,
                                 cy + 0.8 * r], [cx + 0.6 * r, cy + 0.8 * r]])).astype(np.int32)
        cv2.fillPoly(img, [pts], v, lineType=cv2.LINE_AA)
    else:
        raise ValueError(kind)


def _free(box, others):
    x0, y0, x1, y1 = box
    for b in others:
        if x0 < b[2] and b[0] < x1 and y0 < b[3] and b[1] < y1:
            return False
    return True


def add_icons(img, pos, ign, rng):
    """Draw 1..icon_max icons on the window (in place): after a horizontal positive's end (p 0.6) or on free
    background; never over a positive or an ignore area. Returns the boxes drawn [x0, y0, x1, y1]."""
    h, w = img.shape
    taken = [b[:4] for b in pos] + [b[:4] for b in ign]
    hz = [b for b in pos if not b[4]]
    drawn = []
    for _ in range(int(rng.integers(1, int(T8["icon_max"]) + 1))):
        kind = ICON_KINDS[int(rng.integers(len(ICON_KINDS)))]
        if hz and rng.random() < 0.6:
            b = hz[int(rng.integers(len(hz)))]
            lh = max(4, b[3] - b[1])
            size = lh * rng.uniform(*T8["icon_size"])
            cx = b[2] + size * rng.uniform(0.35, 1.2)
            cy = (b[1] + b[3]) / 2 + lh * rng.uniform(-0.1, 0.25)
        else:
            size = rng.uniform(6, 28)
            cx, cy = rng.uniform(0, w), rng.uniform(0, h)
        box = [int(cx - size / 2) - 1, int(cy - size / 2) - 1, int(cx + size / 2) + 2, int(cy + size / 2) + 2]
        if box[0] < 0 or box[1] < 0 or box[2] > w or box[3] > h or not _free(box, taken + drawn):
            continue
        # the icon's grey: the text's side of the window (bright or dark), or a random grey
        patch = img[box[1]:box[3], box[0]:box[2]]
        bg = float(np.median(patch)) if patch.size else 128.0
        if rng.random() < 0.7:
            value = 235 if bg < 128 else 20
        else:
            value = int(rng.integers(0, 256))
        draw_icon(img, kind, cx, cy, size, value, rng)
        drawn.append(box)
    return drawn


# ----------------------------------------------------------------------------- tight regions

def tight_crop(img, pos, ign, rng):
    """Cut the window to one block of positives plus a margin (module docstring): (img, pos, ign) in the cut's px,
    or the inputs unchanged when no block fits."""
    if not pos:
        return img, pos, ign
    seed = pos[int(rng.integers(len(pos)))]
    x0, y0, x1, y1 = seed[:4]
    lh = max(4, (x1 - x0) if seed[4] else (y1 - y0))
    for _ in range(3):
        grown = [x0 - 1.5 * lh, y0 - 1.5 * lh, x1 + 1.5 * lh, y1 + 1.5 * lh]
        for b in pos:
            if b[4] == seed[4] and b[0] < grown[2] and grown[0] < b[2] and b[1] < grown[3] and grown[1] < b[3]:
                x0, y0, x1, y1 = min(x0, b[0]), min(y0, b[1]), max(x1, b[2]), max(y1, b[3])
    m = [int(round(lh * rng.uniform(*T8["tight_margin"]))) for _ in range(4)]
    h, w = img.shape
    cx0, cy0 = max(0, x0 - m[0]), max(0, y0 - m[1])
    cx1, cy1 = min(w, x1 + m[2]), min(h, y1 + m[3])
    if cx1 - cx0 < 16 or cy1 - cy0 < 16:
        return img, pos, ign

    def sh(b):
        return [b[0] - cx0, b[1] - cy0, b[2] - cx0, b[3] - cy0, *b[4:]]
    return np.ascontiguousarray(img[cy0:cy1, cx0:cx1]), [sh(b) for b in pos], [sh(b) for b in ign]


# ----------------------------------------------------------------------------- the window

_base_make_window = B.make_window


def make_window(canvas, smp, src, rng, f_of, sources=None, aug=True):
    """textspot6_train.make_window, then (aug only) the icons and the tight cut."""
    img, pos, ign = _base_make_window(canvas, smp, src, rng, f_of, sources, aug)
    if not aug:
        return img, pos, ign
    img = np.array(img, copy=True)
    if rng.random() < T8["icon_p"]:
        import cv2
        for x0, y0, x1, y1 in add_icons(img, pos, ign, rng):
            if rng.random() < 0.5:      # the window was already blurred / noised: soften the icon like it
                patch = np.ascontiguousarray(img[y0:y1, x0:x1])
                img[y0:y1, x0:x1] = cv2.GaussianBlur(patch, (0, 0), rng.uniform(0.4, 1.0))
    if rng.random() < T8["tight_p"]:
        img, pos, ign = tight_crop(img, pos, ign, rng)
    return img, pos, ign


def apply_settings():
    """textspot8's settings into textspot6_train's module (its make_batch and run read these names)."""
    B.AUG.update(zoom_p=T8["zoom_p"], zoom=list(T8["zoom"]), blend_p=T8["blend_p"], blend_alpha=list(T8["blend_alpha"]))
    B.targets = targets
    B.make_window = make_window
    B.RECIPES.update(RECIPES)
    base_code = B.code_sha256

    def code_sha256():
        d = base_code()
        d["train/textspot8_train.py"] = TT.sha256(ROOT / "train" / "textspot8_train.py")
        return d
    B.code_sha256 = code_sha256


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--t8-json" in argv:
        i = argv.index("--t8-json")
        T8.update(json.loads(argv[i + 1]))
        del argv[i:i + 2]
    apply_settings()
    out = None
    if "--out" in argv:
        out = Path(argv[argv.index("--out") + 1])
    res = B.main(argv)
    if out is not None and (out / "run.json").exists():
        run = json.loads((out / "run.json").read_text(encoding="utf-8"))
        run["textspot8"] = {"settings": T8, "icon_kinds": list(ICON_KINDS)}
        (out / "run.json").write_text(json.dumps(run, indent=1) + "\n", encoding="utf-8")
    return res


if __name__ == "__main__":
    main()
