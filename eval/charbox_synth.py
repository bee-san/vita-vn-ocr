#!/usr/bin/env python3
"""charbox_synth: exact per-glyph boxes from synth4's development-split renders (issue #114, overlay-boxes).

The overlay maps every recognized character to a screen box from the CTC frame where it fires
(eval/charbox.py) and scores those boxes against MeikiOCR's per-character boxes on real frames. This
builds an independent ground truth, synthetic dialogue regions whose every labelled glyph has its EXACT
advance cell and ink box, and measures how accurate MeikiOCR's character boxes are on them.

Renderer: synth/synth4.py (VOCR5's generator; it patches synth/synth3.py), unchanged, initialised as
VOCR5's s4v5631 render was (vocr5-train/render/render.sh: --vocab data/vocab_vn_5631.txt --profile
personal, curriculum_v5631.json, targets_v5631.json, synth3-v3's scene hashes, the font-coverage cache of
that render; CFG = synth3 v3 + synth4's CFG_PATCH, frames None, checked equal to the render's recipe) except
split "dev" (held-out text pools, contexts, names and scenes; synth3's exclusions and eval/manifests/
reserved.json keep every reserved VN out). synth3.render_region(rng, platform) runs for vita, psp and psp2x.
The renderer is only observed, from outside:
  Layer3.__init__ / glyph / arrays   every glyph call (layer, pen origin, character, font object, fill index,
                                     small, ruby) and the arrays() arguments, plus a copy of the arrays before
                                     render_region's typewriter ramp multiplies them in place
  render.draw_cursor, compose_text   the page cursor (for the replay check) and the style whose shadow
                                     decides the ink rule
  synth3.sample_name                 the speaker name (which glyphs of an inline name line are the name)

With frames None (VOCR5's CFG) the geometric steps between a glyph call and the returned region are
Layer3.arrays()'s resample of the canvas to the region (ss2 and area atlases BOX, bilinear atlases BILINEAR,
pixel fonts at 2x NEAREST; none at scale 1) and, for psp2x, the region's upscale to the Vita's screen
(cv2: 2x nearest, 2x bilinear, 1.995x bilinear; scanlines and JPEG are photometric). There is no frame
placement or crop: synth3 v3 draws the dialogue region itself (version 4's console frames are not supported).

Per labelled glyph (furigana, page cursors and the typewriter ghost are never in a glyph list):
  ink   the glyph drawn ALONE into a Layer3 built with the region layer's own arguments, through the same
        arrays() (resample, 4-bit quantisation, gamma), the typewriter ramp's per-pixel factor,
        compose_text's ink rule (fill, outline, hard shadow when its alpha >= 0.5 and blur <= 1) and the
        psp2x upscale with the image's own interpolation, thresholded at 0.5: [x0, y0, x1, y1], the pixel
        edges of the returned region (pixel i spans [i, i + 1)); None when nothing reaches 0.5. ink_lo is
        the same at coverage >= 0.1 and peak the glyph's highest coverage: thin faces at PSP sizes never
        reach 0.5 (their ink is a few pixels or None, and so is the renderer's own line box, so such a
        line's gt crop cuts its glyphs: gt_cuts counts the glyphs whose ink, gt_cuts_lo whose ink_lo, the
        crop does not hold).
  cell  [P_k, P_k+1) between successive pen positions of the displayed string (full-width spaces included,
        so the cell after a space starts right; the last ends at the line's x1), tracking included, P as
        drawn (rounded where the raster mode draws at integer origins), canvas -> region through the very
        resize arrays() applies (resample_positions), then the psp2x scale (the image's W2 / w). clipped:
        the cell leaves the region (a name tab wider than a bare region).
Per line: label (no full-width spaces), displayed string, the spaces' cells, kind (and per glyph its role:
"name" for the name part of an inline name line), the line's ink box from the renderer's own mask (no
furigana) and with furigana, band, the eval-convention gt crop (synth3.gt_box with no random margin: the
labelled ink + furigana + 4 px Vita / psp2x, 2 px PSP), its dark flag (common.line_dark on that crop) and
the renderer's acceptance tests (vocab, CTC feasibility, bracket-only). Per region: RGB, render meta, the
renderer's ink mask (packed bits) and a replay check: every recorded draw call replayed through the same
steps reproduces the renderer's own arrays and ink mask.

  charbox_synth.py render [--platform vita psp psp2x] --regions 3000 --chunks 20 --workers 5 [--seed S] [--out DIR]
  charbox_synth.py check  --data DIR/vita.pkl [--out DIR/check_vita.json]
  charbox_synth.py sheet  --data DIR/vita.pkl --out S.png [--pick outlined|pixel|any] [--raster atlas] [--n 12]
  charbox_synth.py meiki  --data DIR/vita.pkl --out DIR/meiki_vita.jsonl          (teacher/.venv)
  charbox_synth.py score  --dir DIR [--platforms vita psp psp2x] --out DIR/meiki_boxes.json

A region (pickle DIR/<platform>.pkl: {"recipe", "regions": [...]}) is {"id", "rgb", "meta", "geom", "lines",
"ink_mask" (np.packbits of the renderer's ink mask along x), "replay", "speaker"}; a line is {"li", "kind",
"label", "display", "spaces", "glyphs": [{"ch", "cell", "ink", "ink_lo", "peak", "clipped", "px", "c", "role"}],
"faded", "ink", "ink_furi", "ink_glyphs", "band", "top", "fs", "baseline", "gt_box" [x, y, w, h], "gt_ink",
"gt_cuts", "gt_cuts_lo", "pad", "dark", "in_vocab", "ctc_ok", "brackets_only", ...}.

Renders and readings stay under $HERD/shared/artifacts/overlay-boxes/synth/; never commit them.
"""
import argparse
import copy
import hashlib
import inspect
import json
import math
import os
import pickle
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
RENDER = HERD / "shared" / "artifacts" / "vocr5-train" / "render"
OUT = HERD / "shared" / "artifacts" / "overlay-boxes" / "synth"
# VOCR5's s4v5631 render (vocr5-train/render/render.sh, its recipe s4v5631/shards/recipe.json)
RECIPE = {"vocab": ROOT / "data" / "vocab_vn_5631.txt", "profile": "personal",
          "curriculum": RENDER / "curriculum_v5631.json", "targets": RENDER / "targets_v5631.json",
          "scene_hashes": HERD / "shared" / "artifacts" / "synth3-v3" / "exclude" / "scene_hashes.json",
          "font_cache": RENDER / "fonts" / "coverage_cache_v5631.json",
          "recipe": RENDER / "s4v5631" / "shards" / "recipe.json"}
PLATFORMS = ("vita", "psp", "psp2x")
SEED = 20261023
SPACE = "\u3000"
SMALL_KANA = set("ぁぃぅぇぉっゃゅょゎァィゥェォッャュョヮヵヶ")
PUNCT = ("、", "。", "」", "「")
BRACKETS_ONLY = set("「」『』（）()、。・")
INK_LO = 0.1        # ink_lo: every pixel the glyph covers by at least a tenth (thin faces never reach 0.5)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- observing the renderer

class Rec:
    """What one render_region call drew: its Layer3 instances (each carrying `_cb`: constructor arguments,
    draw events, arrays() arguments with the returned arrays and their copies), compose_text calls and
    sample_name results."""
    active = False

    def reset(self):
        self.layers, self.compose, self.names = [], [], []


REC = Rec()
_O = {}             # the renderer's own functions, before observation


def observe():
    """Wrap the renderer's drawing entry points (idempotent; after synth4.patch()). Every wrapper calls the
    original with the same arguments and returns its result: rendering and its random stream are unchanged."""
    import render as S2
    import synth3 as S3
    if _O:
        return
    _O.update(init=S3.Layer3.__init__, glyph=S3.Layer3.glyph, arrays=S3.Layer3.arrays, cursor=S2.draw_cursor,
              compose=S2.compose_text, sample_name=S3.sample_name)
    sig = inspect.signature(_O["init"])

    def init(self, *a, **k):
        _O["init"](self, *a, **k)
        self._cb = None
        if REC.active:
            b = sig.bind(self, *a, **k)
            b.apply_defaults()
            args = dict(b.arguments)
            args.pop("self")
            self._cb = {"args": args, "events": [], "arrays": None}
            REC.layers.append(self)

    def glyph(self, x, base, ch, font, k=0, small=False, ruby=False):
        if REC.active and getattr(self, "_cb", None) is not None:
            self._cb["events"].append(("glyph", (x, base, ch, font, k, small, ruby)))
        return _O["glyph"](self, x, base, ch, font, k, small, ruby)

    def arrays(self, q4_canvas=False, q4=False, gamma=1.0):
        out = _O["arrays"](self, q4_canvas, q4, gamma)
        if REC.active and getattr(self, "_cb", None) is not None:
            fills, s1, s2, _ = out
            self._cb["arrays"] = {"args": (q4_canvas, q4, gamma), "refs": (fills, s1, s2),
                                  "copies": ([f.copy() for f in fills], None if s1 is None else s1.copy(),
                                             None if s2 is None else s2.copy())}
        return out

    def draw_cursor(layer, kind, cx, cy, s):
        if REC.active and getattr(layer, "_cb", None) is not None:
            layer._cb["events"].append(("cursor", (kind, cx, cy, s, layer.cursor_k)))
        return _O["cursor"](layer, kind, cx, cy, s)

    def compose_text(img, fills, s1, s2, cols, st, grad_rows=None):
        if REC.active:
            REC.compose.append({"fills": fills, "st": {k: copy.deepcopy(st.get(k))
                                                       for k in ("shadow", "glow", "stroke", "stroke2")}})
        return _O["compose"](img, fills, s1, s2, cols, st, grad_rows)

    def sample_name(rng):
        r = _O["sample_name"](rng)
        if REC.active:
            REC.names.append(r)
        return r

    S3.Layer3.__init__, S3.Layer3.glyph, S3.Layer3.arrays = init, glyph, arrays
    S2.draw_cursor, S2.compose_text, S3.sample_name = draw_cursor, compose_text, sample_name


# ----------------------------------------------------------------------------- geometry

def resample_positions(cs, n_out, n_canvas, resample):
    """Canvas x coordinates (pixel edges) -> the region's, through the PIL resize Layer3.arrays() applies (none
    when the widths are equal): where a vertical edge at c lands after the very same resize, measured as the
    row sum of its step image (canvas pixels left of c full, the one it cuts in proportion), resized by PIL with
    the same filter and sizes. Exact by construction for all three filters: PIL samples its kernels at source
    pixel centres (a BOX downscale by a non-integer factor is not an area average: up to 0.35 px from
    c * n_out / n_canvas) and steps NEAREST's sampling position by accumulation (a closed form misses boundary
    pixels)."""
    cs = np.asarray(cs, np.float64)
    if n_canvas == n_out or not len(cs):
        return cs
    from PIL import Image
    inside = np.clip(cs, 0.0, float(n_canvas))
    steps = np.clip(inside[:, None] - np.arange(n_canvas, dtype=np.float64)[None, :], 0.0, 1.0).astype(np.float32)
    out = np.asarray(Image.fromarray(steps).resize((n_out, len(cs)), resample), np.float64).sum(axis=1)
    return out + (cs - inside) * (n_out / n_canvas)      # beyond the canvas (a clipped name tab): linear


def bbox(mask):
    rows = np.flatnonzero(mask.any(axis=1))
    if not len(rows):
        return None
    cols = np.flatnonzero(mask.any(axis=0))
    return [int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1]


def ramp_factors(arr):
    """The per-pixel factor render_region's typewriter ramp multiplied each array by (None: untouched)."""
    def fac(ref, cp):
        if ref is None or np.array_equal(ref, cp):
            return None
        f = np.ones_like(cp)
        nz = cp > 0
        f[nz] = ref[nz] / cp[nz]
        return f
    refs, cps = arr["refs"], arr["copies"]
    out = {"fill": [fac(r, c) for r, c in zip(refs[0], cps[0])], "s1": fac(refs[1], cps[1]),
           "s2": fac(refs[2], cps[2])}
    return out if any(v is not None for v in out["fill"] + [out["s1"], out["s2"]]) else None


def _apply(fills, s1, s2, fac):
    if fac is None:
        return fills, s1, s2
    fills = [f * fac["fill"][k] if fac["fill"][k] is not None else f for k, f in enumerate(fills)]
    s1 = s1 * fac["s1"] if s1 is not None and fac["s1"] is not None else s1
    s2 = s2 * fac["s2"] if s2 is not None and fac["s2"] is not None else s2
    return fills, s1, s2


def _ink(fills, s1, s2, st):
    """compose_text's ink (fill union, outer outline, hard shadow) on a dummy image; glow only tints the
    image, never the ink, so it is left out."""
    dummy = np.zeros(fills[0].shape + (3,), np.float32)
    return _O["compose"](dummy, fills, s1, s2, [np.zeros(3, np.float32)] * len(fills), dict(st, glow=None))[1]


def _upscale(a, up, interp=None):
    if up is None:
        return a
    import cv2
    i = interp or up["interp"]
    return cv2.resize(a, (up["W"], up["H"]), interpolation=cv2.INTER_NEAREST if i == "nearest" else cv2.INTER_LINEAR)


def glyph_mask(layer, call, st, fac, up):
    """One glyph drawn alone through its layer's steps: its float ink at the returned region's size."""
    import synth3 as S3
    a = dict(layer._cb["args"])
    x, base, ch, font, k, small, _ = call
    a["n_fill"] = k + 1
    g = S3.Layer3(**a)
    _O["glyph"](g, x, base, ch, font, k, small, False)
    g.ruby = None                    # nothing is drawn there; skip its resample
    fills, s1, s2, _ = _O["arrays"](g, *layer._cb["arrays"]["args"])
    fills, s1, s2 = _apply(fills, s1, s2, fac)
    return _upscale(_ink(fills, s1, s2, st), up)


def replay(layer, st, fac):
    """Every recorded draw call of a layer replayed into a fresh Layer3: (arrays identical to the renderer's
    before the ramp, the layer's ink before any upscale)."""
    import synth3 as S3
    g = S3.Layer3(**layer._cb["args"])
    for kind, args in layer._cb["events"]:
        if kind == "glyph":
            _O["glyph"](g, *args)
        else:
            kd, cx, cy, s, ck = args
            g.cursor_k = ck
            _O["cursor"](g, kd, cx, cy, s)
    fills, s1, s2, _ = _O["arrays"](g, *layer._cb["arrays"]["args"])
    cf, c1, c2 = layer._cb["arrays"]["copies"]
    same = all((p is None and q is None) or (p is not None and q is not None and np.array_equal(p, q))
               for p, q in zip(fills + [s1, s2], cf + [c1, c2]))
    fills, s1, s2 = _apply(fills, s1, s2, fac)
    return same, _ink(fills, s1, s2, st)


# ----------------------------------------------------------------------------- one region

def name_pens(calls, ss, x1):
    """(displayed string, pen positions) of a name drawn without tracking (name line, name tab): a gap
    between two glyph calls is full-width spaces of the name's font."""
    disp, pens, pos = "", [], None
    sp = calls[0][3].getlength(SPACE) / ss
    for c in calls:
        if pos is not None:
            n = 0
            while c[0] > pos + 1e-6 * max(1.0, abs(pos)) and n < 8 and sp > 0:
                disp += SPACE
                pens.append(pos)
                pos += sp
                n += 1
            if abs(c[0] - pos) > 1e-6 * max(1.0, abs(pos)):
                raise ValueError(f"name pen {c[0]} does not follow {pos}")
        disp += c[2]
        pens.append(c[0])
        pos = c[0] + c[3].getlength(c[2]) / ss
    if abs(pos - x1) > 1e-6 * max(1.0, abs(x1)):
        raise ValueError(f"name ends at {pos}, its x1 is {x1}")
    return disp, pens


def build(img, layout, ink_b, ruby_b, fill_m, meta, rid):
    """The region record of one render_region result and the recorder's state."""
    from PIL import Image

    import render as S2
    import synth3 as S3
    comp = REC.compose

    def layer_of(fills):
        return next(L for L in REC.layers if L._cb["arrays"] is not None and L._cb["arrays"]["refs"][0] is fills)
    main = layer_of(comp[0]["fills"])
    tab = layer_of(comp[1]["fills"]) if len(comp) > 1 else None
    a = main._cb["args"]
    w, h, ss, integer = a["w"], a["h"], a["scale"], a["integer"]
    Wc, Hc = main.fill[0].size
    H, W = img.shape[:2]
    tags = meta["capture"].split("+")
    up, k_final = None, 1.0
    if (H, W) != (h, w):
        up = {"W": W, "H": H, "interp": "nearest" if "nearest" in tags else "linear"}
        k_final = 1.995 if "x1995" in tags else 2.0
    sx = W / w
    fac = ramp_factors(main._cb["arrays"])

    def X(ps):          # pen positions (region px before any upscale) -> the returned region's x
        e = [round(p * ss) if integer else p * ss for p in ps]     # where Layer3 draws them (its _xy)
        return [float(v) * sx for v in resample_positions(e, w, Wc, a["resample"])]

    def Y(p):           # a baseline (region px before any upscale) -> the returned region's y
        e = round(p * ss) if integer else p * ss
        return float(resample_positions([e], h, Hc, a["resample"])[0]) * (H / h)

    # replay check: the recorded calls rebuild the renderer's own arrays and ink mask
    same, ink_r = replay(main, comp[0]["st"], fac)
    if tab is not None:
        same_t, ink_t = replay(tab, comp[1]["st"], None)
        same = same and same_t
        np.maximum(ink_r, ink_t, out=ink_r)
    ink_r = _upscale(ink_r.astype(np.float32), up, "linear") >= 0.5        # the renderer's mask interpolation
    replay_diff = int((ink_r != ink_b).sum())

    main_calls = [ev[1] for ev in main._cb["events"] if ev[0] == "glyph" and not ev[1][6]]
    tab_calls = [ev[1] for ev in tab._cb["events"] if ev[0] == "glyph"] if tab is not None else []
    speaker = REC.names[0] if meta.get("name") else None
    pad = S3.CFG["pad"]["psp" if meta["target"] == "psp" else "vita"]
    grey = S3.to_gray(img)
    lines, it = [], 0
    for li, L in enumerate(layout):
        if tab is not None and li == 0:            # the name tab, inserted first, drawn on its own layer
            calls, layer, st, lfac = tab_calls, tab, comp[1]["st"], None
        else:
            n = len(L["text"]) if L.get("xs") is None else sum(ch != SPACE for ch in L["line"])
            calls, layer, st, lfac = main_calls[it:it + n], main, comp[0]["st"], fac
            it += n
        if not L.get("text"):
            continue
        x1 = L["x1"] / k_final
        if L.get("xs") is None:
            disp, pens = name_pens(calls, ss, x1)
        else:
            disp, pens, ci = L["line"], [], 0
            for k, ch in enumerate(disp):
                if ch == SPACE:
                    pens.append(L["xs"][k] / k_final)
                    continue
                c = calls[ci]
                ci += 1
                if c[2] != ch or c[0] * k_final != L["xs"][k]:
                    raise ValueError(f"glyph call {c[:3]} is not {ch!r} at {L['xs'][k]}")
                pens.append(c[0])
        edges = X(pens + [x1])
        gidx = [k for k, ch in enumerate(disp) if ch != SPACE]
        label = L["text"]
        if "".join(disp[k] for k in gidx[:len(label)]) != label or len(calls) != len(gidx):
            raise ValueError(f"line {li}: label {label!r} is not the drawn {disp!r}")
        n_name = 0
        if L["kind"] == "name":
            n_name = len(label)
        elif L["kind"] == "name+dialogue" and speaker and label.startswith(speaker[1]):
            n_name = len(speaker[1])
        glyphs, faded = [], []
        for j, k in enumerate(gidx):
            fm = glyph_mask(layer, calls[j], st, lfac, up)
            m = fm >= 0.5
            ib = bbox(m)
            g = {"ch": disp[k], "cell": [edges[k], edges[k + 1]], "ink": ib, "ink_lo": bbox(fm >= INK_LO),
                 "peak": round(float(fm.max()), 4) if fm.size else 0.0,
                 "clipped": bool(edges[k] < -1e-6 or edges[k + 1] > W + 1e-6)}
            if ib is not None:
                ys, xs = np.nonzero(m[ib[1]:ib[3], ib[0]:ib[2]])
                g["px"] = int(len(xs))
                g["c"] = [round(float(xs.mean()) + ib[0] + 0.5, 3), round(float(ys.mean()) + ib[1] + 0.5, 3)]
            if j < len(label):
                g["role"] = "name" if j < n_name else ("dialogue" if L["kind"] == "name+dialogue" else L["kind"])
                glyphs.append(g)
            else:
                faded.append(g)
        saved = L.get("ruby_top")
        L["ruby_top"] = None
        ib_main = S3.line_ink(L, layout, ink_b, ruby_b, ink_b.shape)
        L["ruby_top"] = saved
        ib_furi = S3.line_ink(L, layout, ink_b, ruby_b, ink_b.shape)
        box, gib = S3.gt_box(L, layout, ink_b, ruby_b, pad, np.random.default_rng(0), img.shape,
                             extra={"p": 0.0}, fill=fill_m)
        inks = [g["ink"] for g in glyphs if g["ink"] is not None]
        cut = cut_lo = None
        if box is not None:                # glyphs whose ink (>= 0.5; ink_lo: >= INK_LO) the gt crop does not hold
            bx, by, bw, bh = box

            def outside(b):
                return b is not None and (b[0] < bx or b[1] < by or b[2] > bx + bw or b[3] > by + bh)
            cut = sum(1 for g in glyphs if outside(g["ink"]))
            cut_lo = sum(1 for g in glyphs if outside(g["ink_lo"]))
        lines.append({
            "li": li, "kind": L["kind"], "label": label, "display": disp,
            "spaces": [[edges[k], edges[k + 1]] for k, ch in enumerate(disp) if ch == SPACE],
            "x01": [edges[0], edges[-1]], "glyphs": glyphs, "faded": faded,
            "ink": None if ib_main is None else [int(v) for v in ib_main],
            "ink_furi": None if ib_furi is None else [int(v) for v in ib_furi],
            "ink_glyphs": ([min(b[0] for b in inks), min(b[1] for b in inks), max(b[2] for b in inks),
                            max(b[3] for b in inks)] if inks else None),
            "furigana": saved is not None, "furi_top": None if saved is None else int(saved),
            "band": [float(L["y0"]), float(L["y1"])], "top": float(L["top"]), "fs": float(L["fs"]),
            "baseline": Y(calls[0][1]) if calls else None,
            "gt_box": None if box is None else [int(v) for v in box], "gt_ink": None if gib is None else
            [int(v) for v in gib], "gt_cuts": cut, "gt_cuts_lo": cut_lo, "pad": pad, "fillbox": bool(L.get("fillbox")),
            "dark": None if box is None else int(S3.line_dark(grey, box)),
            "in_vocab": all(c in S3.R.vocab_set for c in label),
            "ctc_ok": None if box is None else bool(S2.crop_ctc_feasible((box[3], box[2]), label)),
            "brackets_only": set(label) <= BRACKETS_ONLY,
            "coloured": bool(L.get("coloured")), "fade": int(L.get("fade", 0))})
    if it != len(main_calls):
        raise ValueError(f"{len(main_calls) - it} glyph calls left over")
    geom = {"canvas": [Wc, Hc], "ss": ss, "integer": bool(integer), "resample": {
        Image.NEAREST: "nearest", Image.BOX: "box", Image.BILINEAR: "bilinear"}.get(a["resample"], str(a["resample"])),
        "pre": [w, h], "size": [W, H], "up": up, "k_final": k_final, "ramp": fac is not None}
    meta = {k: v for k, v in meta.items() if not k.startswith("_")}
    return {"id": rid, "rgb": img, "meta": meta, "geom": geom, "lines": lines,
            "ink_mask": np.packbits(ink_b, axis=1), "replay": {"arrays_equal": bool(same), "ink_diff": replay_diff},
            "speaker": None if speaker is None else list(speaker)}


def region_record(rng, platform, rid):
    """("ok", record) | (None, None) for an unusable draw | ("error", message) for a renderer exception."""
    import synth3 as S3
    REC.reset()
    REC.active = True
    try:
        r = S3.render_region(rng, platform)
    except Exception as e:  # noqa: BLE001 - as run_shard: a rare bad draw is skipped
        return "error", f"{type(e).__name__}: {e}"
    finally:
        REC.active = False
    if r is None:
        return None, None
    img, layout, ink_b, meta = r
    ruby_b, fill_m = meta.pop("_ruby_mask"), meta.pop("_fill_mask")
    return "ok", build(img, layout, ink_b, ruby_b, fill_m, meta, rid)


# ----------------------------------------------------------------------------- render

def init_renderer(out):
    """synth4 initialised as VOCR5's s4v5631 render, split dev; the font-coverage cache is copied first so a
    shared file is never written."""
    sys.path.insert(0, str(ROOT / "synth"))
    import cv2
    cv2.setNumThreads(1)
    import synth3 as S3
    import synth4 as S4
    cache = Path(out) / "font_cache_v5631.json"
    if not cache.exists():
        Path(out).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(RECIPE["font_cache"], cache)
    S4.init4(RECIPE["curriculum"], assets=S3.ASSETS, split="dev", profile=RECIPE["profile"], font_cache=cache,
             scene_hashes=RECIPE["scene_hashes"], targets=RECIPE["targets"], vocab=RECIPE["vocab"])
    if S3.CFG.get("frames") is not None:
        raise SystemExit("version 4 console frames are not supported (VOCR5's CFG has frames None)")
    observe()
    return S3, S4


def recipe_record(S3, S4, args):
    import cv2
    from PIL import Image, features
    vr = json.loads(Path(RECIPE["recipe"]).read_text(encoding="utf-8"))
    cfg = json.loads(json.dumps(S3.CFG, ensure_ascii=False))
    code = {p: sha256(ROOT / "synth" / p) for p in ("synth3.py", "render.py", "common.py", "synth3_text.py",
                                                    "synth4.py")}
    R = S3.R
    return {"generator": "synth4", "split": "dev", "seed": args.seed, "regions": args.regions, "chunks": args.chunks,
            "profile": RECIPE["profile"], "vocab": str(RECIPE["vocab"].relative_to(ROOT)),
            "vocab_sha256": sha256(RECIPE["vocab"]), "curriculum_sha256": sha256(RECIPE["curriculum"]),
            "targets_sha256": sha256(RECIPE["targets"]), "scene_hashes_sha256": sha256(RECIPE["scene_hashes"]),
            "font_cache": str(RECIPE["font_cache"]), "code_sha256": code, "charbox_synth_sha256": sha256(__file__),
            "vocr5_render": {"recipe": str(RECIPE["recipe"]), "code_equal": code == vr["code_sha256"],
                             "cfg_equal": cfg == vr["cfg"], "cfg4_equal": json.loads(json.dumps(S4.CFG4)) == vr["cfg4"],
                             "vocab_equal": sha256(RECIPE["vocab"]) == vr["vocab_sha256"]},
            "cfg": cfg, "cfg4": S4.CFG4, "fonts": len(R.fonts), "text": R.text_identity,
            "background_shots": {k: len(v) for k, v in R.bg.items()}, "background_vns": len(R.bg_vns),
            "exclusions": R.exclusions["files"], "scenes_dropped": R.scene_dropped,
            "versions": {"numpy": np.__version__, "opencv": cv2.__version__, "pillow": Image.__version__,
                         "freetype": features.version("freetype2")}}


def run_chunk(job):
    platform, chunk, n, seed, parts = job
    import synth3 as S3
    import synth4 as S4
    path = Path(parts) / f"{platform}_c{chunk:02d}.pkl"
    if path.exists():
        return platform, chunk, -1, 0, 0, 0, 0.0
    pi = PLATFORMS.index(platform)
    rng = np.random.default_rng([seed, pi, chunk])
    R = S3.R
    R.counts[:] = 0
    R._focus_w = None
    R.quota_t = S4.shard_targets(pi * 100 + chunk, seed, S4.R4["deficit"], S4.CFG4["curriculum"]["shards"])
    regions, errors, nones, bad, t0 = [], [], 0, [], time.time()
    while len(regions) < n:
        rid = f"{platform}_c{chunk:02d}_{len(regions):04d}"
        try:
            status, rec = region_record(rng, platform, rid)
        except Exception as e:  # noqa: BLE001 - this script's own failure on a region: recorded, never hidden
            bad.append(f"{rid}: {type(e).__name__}: {e}")
            if len(bad) > 20:
                raise
            continue
        if status == "error":
            errors.append(rec)
            if len(errors) > 1000:
                raise RuntimeError("too many render errors")
            continue
        if status is None:
            nones += 1
            continue
        regions.append(rec)
        for ln in rec["lines"]:                    # the curriculum's accounting, as run_shard's crops do
            for c in ln["label"]:
                if c in R.cid:
                    R.counts[R.cid[c]] += 1
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"regions": regions, "render_errors": errors, "unusable": nones, "build_errors": bad}, f,
                    protocol=5)
    tmp.rename(path)
    return platform, chunk, len(regions), len(errors), nones, len(bad), time.time() - t0


def cmd_render(args):
    import multiprocessing as mp
    out = Path(args.out)
    parts = out / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    S3, S4 = init_renderer(out)
    rec = recipe_record(S3, S4, args)
    print(json.dumps(rec["vocr5_render"]), f"fonts {rec['fonts']}", f"scenes {rec['background_shots']}", flush=True)
    if not all(v for k, v in rec["vocr5_render"].items() if k != "recipe"):
        raise SystemExit("the renderer differs from VOCR5's s4v5631 render")
    per = args.regions // args.chunks
    jobs = [(p, c, per, args.seed, str(parts)) for p in args.platform for c in range(args.chunks)]
    t0 = time.time()
    with mp.get_context("fork").Pool(args.workers) as pool:
        for p, c, n, e, u, b, dt in pool.imap_unordered(run_chunk, jobs):
            print(f"{p} chunk {c}: {n} regions, {e} render errors, {u} unusable draws, {b} build errors in "
                  f"{dt:.0f}s ({time.time() - t0:.0f}s)", flush=True)
    for p in args.platform:
        regions, errors, nones, bad = [], [], 0, []
        for c in range(args.chunks):
            with open(parts / f"{p}_c{c:02d}.pkl", "rb") as f:
                d = pickle.load(f)
            regions += d["regions"]
            errors += d["render_errors"]
            nones += d["unusable"]
            bad += d["build_errors"]
        doc = {"kind": "charbox-synth", "schema": 1, "platform": p, "recipe": rec, "render_errors": errors,
               "unusable_draws": nones, "build_errors": bad, "regions": regions}
        path = out / f"{p}.pkl"
        with open(path.with_name(path.name + ".tmp"), "wb") as f:
            pickle.dump(doc, f, protocol=5)
        path.with_name(path.name + ".tmp").rename(path)
        print(f"{p}: {len(regions)} regions, {sum(len(r['lines']) for r in regions)} lines, "
              f"{sum(len(ln['glyphs']) for r in regions for ln in r['lines'])} glyphs -> {path}", flush=True)
    print("RENDER_DONE", flush=True)


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ----------------------------------------------------------------------------- checks

def cmd_check(args):
    d = load(args.data)
    regs = d["regions"]
    c = defaultdict(int)
    kinds = defaultdict(int)
    hist = defaultdict(lambda: defaultdict(int))
    worst = []
    for r in regs:
        W, H = r["geom"]["size"]
        c["regions"] += 1
        m = r["meta"]
        hist["raster"][m["raster"]] += 1
        hist["layout"][m["layout"]] += 1
        hist["category"][m["cat"]] += 1
        hist["upscale"][next((t for t in m["capture"].split("+") if t in ("bilinear", "nearest", "x1995",
                                                                       "scanlines")), "none")] += 1
        hist["outlined"]["outline" if m["sw"] > 0 else "shadow" if m["shadow"] else "plain"] += 1
        c["replay_arrays_differ"] += not r["replay"]["arrays_equal"]
        c["replay_ink_differs"] += r["replay"]["ink_diff"] > 0
        c["ramp_regions"] += r["geom"]["ramp"]
        for ln in r["lines"]:
            c["lines"] += 1
            kinds[ln["kind"]] += 1
            c["lines_no_gt"] += ln["gt_box"] is None
            c["lines_no_ink"] += ln["ink"] is None
            c["lines_furigana"] += ln["furigana"]
            c["spaces"] += len(ln["spaces"])
            c["faded"] += len(ln["faded"])
            c["lines_train_ok"] += bool(ln["gt_box"] and ln["in_vocab"] and ln["ctc_ok"] and not ln["brackets_only"])
            c["lines_gt_cuts_glyphs"] += bool(ln["gt_cuts"])
            c["lines_gt_cuts_glyphs_lo"] += bool(ln["gt_cuts_lo"])
            c["lines_usable"] += bool(ln["gt_box"] and ln["ctc_ok"] and not ln["gt_cuts"])
            gl = ln["glyphs"]
            if len(gl) != len(ln["label"]) or "".join(g["ch"] for g in gl) != ln["label"]:
                c["label_mismatch"] += 1
            prev = None
            for g in gl:
                c["glyphs"] += 1
                x0, x1 = g["cell"]
                c["cell_nonpositive"] += x1 <= x0
                c["cell_outside"] += x0 < -1e-6 or x1 > W + 1e-6
                c["clipped"] += g["clipped"]
                c["faint_peak_below_half"] += g["peak"] < 0.5
                c["glyph_no_ink_lo"] += g["ink_lo"] is None
                if prev is not None and x0 < prev - 1e-6:
                    c["cell_not_monotone"] += 1
                prev = x1
                b = g["ink"]
                if b is None:
                    c["glyph_no_ink"] += 1
                    continue
                c["ink_outside_region"] += b[0] < 0 or b[1] < 0 or b[2] > W or b[3] > H
                cx = (b[0] + b[2]) / 2
                c["ink_centre_in_cell"] += x0 <= cx < x1
                cross = b[0] < x0 - 1e-6 or b[2] > x1 + 1e-6
                c["ink_crosses_cell"] += cross
                c["ink_crosses_cell_by_2px"] += b[0] < x0 - 2 or b[2] > x1 + 2
                if ln["ink"] is not None:
                    L = ln["ink"]
                    out = max(L[0] - b[0], L[1] - b[1], b[2] - L[2], b[3] - L[3])
                    c["glyph_ink_outside_line_ink"] += out > 0
                    if out > 1:
                        worst.append((out, r["id"], ln["li"], g["ch"]))
                if ln["gt_box"] is not None:
                    x, y, w, h = ln["gt_box"]
                    c["glyph_ink_outside_gt_crop"] += b[0] < x or b[1] < y or b[2] > x + w or b[3] > y + h
            for i in range(len(gl) - 1):     # contiguous cells unless a full-width space sits between
                gap = gl[i + 1]["cell"][0] - gl[i]["cell"][1]
                c["cells_gap"] += abs(gap) > 1e-6
            c["cells_gap_is_space"] += sum(1 for s in ln["spaces"]
                                           if any(abs(s[0] - g["cell"][1]) < 1e-6 for g in gl))
    out = {"platform": d["platform"], "counts": dict(c), "kinds": dict(kinds),
           "hist": {k: dict(sorted(v.items())) for k, v in hist.items()}, "render_errors": len(d["render_errors"]),
           "unusable_draws": d["unusable_draws"], "build_errors": d["build_errors"][:20],
           "worst_outside_line_ink": sorted(worst, reverse=True)[:10]}
    print(json.dumps(out, ensure_ascii=False, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- contact sheets

def label_font(size):
    from PIL import ImageFont
    fdir = HERD / "shared" / "data" / "fonts" / "notosansjp"
    return ImageFont.truetype(str(sorted(fdir.glob("*.ttf"))[0]), size)


def pick_lines(regs, pick, n, rng, raster=None):
    def ok(r, ln):
        m = r["meta"]
        if not ln["glyphs"] or ln["ink"] is None or (raster and not m["raster"].startswith(raster)):
            return False
        if pick == "outlined":
            return m["sw"] > 0 or m["shadow"]
        if pick == "pixel":
            return m["raster"].startswith("pixel")
        return True
    cands = [(i, j) for i, r in enumerate(regs) for j, ln in enumerate(r["lines"]) if ok(r, ln)]
    if pick == "any" and cands:          # psp2x: every upscale mode on the sheet
        by = defaultdict(list)
        for i, j in cands:
            by[next((t for t in regs[i]["meta"]["capture"].split("+") if t in ("bilinear", "nearest", "x1995",
                                                                             "scanlines")), "?")].append((i, j))
        out = []
        while len(out) < n and any(by.values()):
            for k in sorted(by):
                if by[k] and len(out) < n:
                    out.append(by[k].pop(int(rng.integers(len(by[k])))))
        return out
    idx = rng.choice(len(cands), size=min(n, len(cands)), replace=False)
    return [cands[k] for k in sorted(idx)]


def cmd_sheet(args):
    """Lines at `zoom` x (nearest) with each glyph's ink box (green / yellow, drawn just outside its pixels),
    its ink_lo box where that differs by more than a pixel (grey), its cell as a red / blue bar under the
    line with a tick at the cell's start, and full-width spaces' cells as hollow bars. A line wider than the
    sheet shows its start, or (every other row) its end, where the last cell must end at the line's x1."""
    from PIL import Image, ImageDraw
    d = load(args.data)
    regs = d["regions"]
    rng = np.random.default_rng(args.seed)
    picks = pick_lines(regs, args.pick, args.n, rng, args.raster)
    z, maxw = args.zoom, 1280
    font = label_font(13)
    rows, shown = [], []
    for n_row, (i, j) in enumerate(picks):
        r, ln = regs[i], regs[i]["lines"][j]
        H, W = r["rgb"].shape[:2]
        span = (maxw - 8) // z
        x0 = max(0, int(math.floor(ln["x01"][0])) - 6)
        if (args.end or n_row % 2) and ln["x01"][1] + 6 - x0 > span:
            x0 = max(0, int(math.ceil(ln["x01"][1])) + 6 - span)
        x1 = min(W, x0 + span)
        ref = ln["gt_box"] or [ln["ink"][0], ln["ink"][1], ln["ink"][2] - ln["ink"][0], ln["ink"][3] - ln["ink"][1]]
        y0, y1 = max(0, ref[1] - 3), min(H, ref[1] + ref[3] + 3)
        crop = Image.fromarray(r["rgb"][y0:y1, x0:x1]).resize(((x1 - x0) * z, (y1 - y0) * z), Image.NEAREST)
        strip, tag_h = 10, 17
        im = Image.new("RGB", (maxw, crop.height + strip + tag_h + 2), (24, 24, 24))
        im.paste(crop, (0, tag_h))
        dr = ImageDraw.Draw(im)
        yb = tag_h + crop.height + 2

        def rect(b, colour):
            dr.rectangle(((b[0] - x0) * z - 1, (b[1] - y0) * z - 1 + tag_h, (b[2] - x0) * z, (b[3] - y0) * z + tag_h),
                         outline=colour)
        for k, g in enumerate(ln["glyphs"]):
            cx0, cx1 = round((g["cell"][0] - x0) * z), round((g["cell"][1] - x0) * z)
            colour = ((255, 60, 60), (60, 140, 255))[k % 2]
            dr.rectangle((cx0, yb, cx1 - 1, yb + strip - 3), fill=colour)
            dr.line((cx0, tag_h, cx0, tag_h + 3 * z), fill=colour)
            lo, hi = g["ink_lo"], g["ink"]
            if lo is not None and (hi is None or max(abs(u - v) for u, v in zip(hi, lo)) > 1):
                rect(lo, (150, 150, 150))
            if g["ink"] is not None:
                rect(g["ink"], ((0, 255, 0), (255, 255, 0))[k % 2])
        for sp in ln["spaces"]:
            dr.rectangle((round((sp[0] - x0) * z), yb, round((sp[1] - x0) * z) - 1, yb + strip - 3),
                         outline=(200, 200, 200))
        m = r["meta"]
        tag = (f"{n_row + 1}. {r['id']} L{ln['li']} {ln['kind']} | {m['fam']} {m['raster']} sw{m['sw']} "
               f"sh{int(m['shadow'])} {m['capture']} | x{x0}-{x1} | {ln['label'][:36]}")
        dr.text((2, 1), tag, font=font, fill=(255, 255, 160))
        rows.append(im)
        shown.append({"row": n_row + 1, "id": r["id"], "li": ln["li"], "label": ln["label"], "x": [x0, x1],
                      "raster": m["raster"], "capture": m["capture"], "fam": m["fam"]})
    sheet = Image.new("RGB", (maxw, min(1280, sum(im.height for im in rows))), (0, 0, 0))
    y, kept = 0, 0
    for im in rows:
        if y + im.height > sheet.height:
            break
        sheet.paste(im, (0, y))
        y += im.height
        kept += 1
    sheet.save(args.out)
    Path(args.out).with_suffix(".json").write_text(json.dumps(shown[:kept], ensure_ascii=False, indent=0) + "\n",
                                                   encoding="utf-8")
    print(f"{kept} lines -> {args.out} ({sheet.width}x{sheet.height})")


# ----------------------------------------------------------------------------- MeikiOCR

def cmd_meiki(args):
    """MeikiOCR's recognizer on every gt crop, as teacher/label.py reads a line's view: the BGR crop of the
    region, engines.load(["meiki"]) (punct factor 0.2, rec threshold 0.1). Boxes go to region pixels."""
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from teacher import engines
    eng = engines.load(["meiki"])["meiki"]
    d = load(args.data)
    n, t0 = 0, time.time()
    with open(args.out, "w", encoding="utf-8") as f:
        for r in d["regions"]:
            for ln in r["lines"]:
                if ln["gt_box"] is None:
                    continue
                x, y, w, h = ln["gt_box"]
                view = np.ascontiguousarray(r["rgb"][y:y + h, x:x + w, ::-1])
                text, mean, low, chars = eng(view)
                f.write(json.dumps({"id": r["id"], "li": ln["li"], "text": text, "mean": mean, "min": low,
                                    "chars": [[c[0], c[1] + x, c[2] + y, c[3] + x, c[4] + y, c[5]] for c in chars]},
                                   ensure_ascii=False) + "\n")
                n += 1
    print(f"{n} crops read in {time.time() - t0:.0f}s -> {args.out}")


def iou1(a0, a1, b0, b1):
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    return inter / union if union > 0 else 0.0


def char_class(c):
    if c in PUNCT:
        return c
    if c in SMALL_KANA:
        return "small kana"
    return "other"


def cmd_score(args):
    """MeikiOCR's character boxes against the exact ones, on the lines it reads exactly (per character, after
    eval/metrics.norm), per glyph: its box centre inside the glyph's ink box (2-D, along x, and in ink_lo) and
    inside its cell (x); the 1-D IoU along the line with ink and cell; the centre's offset from the ink centre
    in units of the line's ink height (dx_floor adds back the half pixel MeikiOCR's int() of both edges takes
    off its centre); its width over the ink's. Rows: all, 、。」「, small kana, other, by glyph role, style,
    psp2x upscale."""
    sys.path.insert(0, str(HERE))
    from metrics import norm
    out = {"kind": "charbox-synth-meiki", "schema": 1, "platforms": {}}
    for p in args.platforms:
        d = load(Path(args.dir) / f"{p}.pkl")
        reads = {}
        with open(Path(args.dir) / f"meiki_{p}.jsonl", encoding="utf-8") as f:
            for line in f:
                o = json.loads(line)
                reads[(o["id"], o["li"])] = o
        acc = defaultdict(lambda: defaultdict(list))
        n = defaultdict(int)
        for r in d["regions"]:
            for ln in r["lines"]:
                o = reads.get((r["id"], ln["li"]))
                if o is None:
                    continue
                n["lines_read"] += 1
                n["lines_read_" + ln["kind"]] += 1
                lab = [norm(g["ch"]) for g in ln["glyphs"]]
                got = [norm(c[0]) for c in o["chars"]]
                n["lines_text_equal"] += norm(o["text"]) == norm(ln["label"])
                if got != lab:
                    continue
                n["lines_matched"] += 1
                n["lines_matched_" + ln["kind"]] += 1
                m = r["meta"]
                style = "style:" + ("pixel" if m["raster"].startswith("pixel") else "outline" if m["sw"] > 0 else
                                    "shadow" if m["shadow"] else "plain")
                upm = "up:" + next((t for t in m["capture"].split("+") if t in ("bilinear", "nearest", "x1995",
                                                                              "scanlines")), "none")
                ih = ln["ink"][3] - ln["ink"][1] if ln["ink"] else None
                for g, c in zip(ln["glyphs"], o["chars"]):
                    n["glyphs"] += 1
                    if g["ink"] is None or not ih:
                        n["glyphs_no_ink"] += 1
                        continue
                    bx0, by0, bx1, by1 = c[1:5]
                    mx, my = (bx0 + bx1) / 2, (by0 + by1) / 2
                    ix0, iy0, ix1, iy1 = g["ink"]
                    cx0, cx1 = g["cell"]
                    lo = g["ink_lo"]
                    v = {"in_ink": float(ix0 <= mx < ix1 and iy0 <= my < iy1), "in_ink_x": float(ix0 <= mx < ix1),
                         "in_ink_lo": float(lo[0] <= mx < lo[2] and lo[1] <= my < lo[3]),
                         "in_cell": float(cx0 <= mx < cx1), "iou_ink": iou1(bx0, bx1, ix0, ix1),
                         "iou_cell": iou1(bx0, bx1, cx0, cx1), "dx": (mx - (ix0 + ix1) / 2) / ih,
                         "dx_floor": (mx + 0.5 - (ix0 + ix1) / 2) / ih,
                         "dy": (my - (iy0 + iy1) / 2) / ih,
                         "dist": math.hypot(mx - (ix0 + ix1) / 2, my - (iy0 + iy1) / 2) / ih,
                         "w_ratio": (bx1 - bx0) / max(1, ix1 - ix0)}
                    for key in ("all", "class:" + char_class(g["ch"]), "role:" + g["role"], style, upm):
                        for k2, val in v.items():
                            acc[key][k2].append(val)
        rows = {}
        for key, vals in sorted(acc.items()):
            rows[key] = {"glyphs": len(vals["in_ink"])}
            for k2, arr in vals.items():
                a = np.asarray(arr, np.float64)
                rows[key][k2] = round(float(a.mean()), 4)
                if k2 in ("dx", "dx_floor", "dy", "dist"):
                    rows[key][k2 + "_median"] = round(float(np.median(a)), 4)
        out["platforms"][p] = {"counts": dict(n), "rows": rows}
        print(p, json.dumps(dict(n)), flush=True)
        for key in ["all"] + [f"class:{c}" for c in PUNCT + ("small kana", "other")] + \
                ["role:dialogue", "role:name", "role:narration", "role:choice", "role:ui"] + \
                [f"style:{c}" for c in ("plain", "outline", "shadow", "pixel")] + \
                [f"up:{c}" for c in ("bilinear", "nearest", "x1995", "scanlines")]:
            if key in rows:
                q = rows[key]
                print(f"  {key:18s} n={q['glyphs']:6d} in_ink {q['in_ink']:.4f} in_cell {q['in_cell']:.4f} "
                      f"iou_ink {q['iou_ink']:.3f} iou_cell {q['iou_cell']:.3f} dx {q['dx']:+.3f} dy {q['dy']:+.3f} "
                      f"dist {q['dist']:.3f}", flush=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["render", "check", "sheet", "meiki", "score"])
    ap.add_argument("--platform", nargs="+", choices=PLATFORMS, default=list(PLATFORMS))
    ap.add_argument("--platforms", nargs="+", choices=PLATFORMS, default=list(PLATFORMS))
    ap.add_argument("--regions", type=int, default=1500)
    ap.add_argument("--chunks", type=int, default=10)
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=None)
    ap.add_argument("--data")
    ap.add_argument("--dir", default=str(OUT))
    ap.add_argument("--pick", choices=["outlined", "pixel", "any"], default="any")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--zoom", type=int, default=2)
    ap.add_argument("--end", action="store_true", help="sheet: show each line's end when it does not fit")
    ap.add_argument("--raster", help="sheet: only regions whose raster mode starts with this (atlas, pixel2, ss2)")
    args = ap.parse_args()
    if args.mode == "render":
        args.out = args.out or str(OUT)
        return cmd_render(args)
    return {"check": cmd_check, "sheet": cmd_sheet, "meiki": cmd_meiki, "score": cmd_score}[args.mode](args)


if __name__ == "__main__":
    main()
