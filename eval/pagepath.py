#!/usr/bin/env python3
"""The page-path benchmark (issue #129): whole dialogue read the way Rubi reads it, from the line finder to the text
the player sees, on the owner's hooked pages (#125) and on the development whole-dialogue units.

  pagepath.py units --pagepool DIR --pages DIR --set hval|tune|test [--per-title N] --out UNITS.jsonl
      region units of the hooked pages: each page's dialogue groups as the regions a player draws (the page pool's
      base / expanded-4 / expanded-8 regions: eval/finder_tighten.py's rule), with their verified lines
  pagepath.py run --units UNITS.jsonl|DEV.pkl --pages DIR --finder SPEC|--boxes FILE --models DIR --lm FILE
                  --rubi RUBI --cache DB --out RESULT.json [--rows ROWS.jsonl] [--tier N] [--workers 5]
      find (or take) every unit's line boxes, read every crop the way Rubi reads it, score
  pagepath.py compare --base RESULT.json --arm RESULT.json [--rows-base R --rows-arm R] [--out JSON]
      paired difference in dialogue CER with title / VN cluster bootstrap intervals

Units. A hooked unit is one dialogue group of a page cut as a region: the page pool's (data/import_hooked.py pages)
regions, which group a page's verified horizontal dialogue lines with eval's rule, take their union plus 12 px
(Vita) / 6 px (PSP) and grow it by 0 / 4 / 8 px, and keep every other text box (speaker names, unverified rows,
vertical columns) out of it. Its reference is those lines' labels, top to bottom; its cells are the lines'
character cells (renderer geometry). A development unit is one of #119's whole-dialogue units
($A/vocr5-analysis/work/units.pkl: teacher_val and devnative, other games than the corpus).

Views. "console" (the default): what Rubi gets. Vita frames as they are; a PSP frame upscaled 2x bilinear before it
is cut (Adrenaline shows a PSP game 2x on the Vita's screen; cv2.INTER_LINEAR on the colour frame, then
synth/common.to_gray). "native": every frame at its own size.

The path (Rubi main bd130a8, core/vocr_ocr.c, game_ocr/main.c): the finder (Rubi: VOCR_LINES_TEXTBOX_RECOVERY), its
short-line probe (#194), the refinement on PSP titles (ocr_refine_lines = psp), all in eval/pagepath_find.c; then
per crop, read with its dark flag as VOCR_INVERT: W8I greedy with the top 8 kept, the counter re-decode (6 / 4
nats), beam 8 + the 1 MB LM (alpha 500, beta 3500, gate 8000, EOS, bracket 8000) when the greedy text is not empty;
the cascade (a line whose greedy confidence is under 0.7 is read again by the next tier up: S -> N, N -> L; its
re-decoded beam text shown, empty when its greedy text is); Rubi's core/textnorm.c (built from --rubi); the Shell's
non_japanese_filter = lines (a line with no hiragana, katakana or CJK goes). The cascade's time ceiling is not
modelled.

Scores. Dialogue CER of a unit: every crop's text in box order, each normalised with eval/metrics.norm and joined,
against the reference lines joined the same way, by Levenshtein distance (train/general_vn_textbox_guard
transcript_score's count for a region without names). Line statistics against the reference cells: a line is whole
when one box holds every cell centre, cut when the best box holds only some, missed when none holds any; a box is
merged when it holds over half the cells of two or more lines, junk when it holds no cell.

--finder SPEC: a preset (rubi, rubi-noprobe, grad, grad+refine, textbox, textbox_recovery) with optional overrides
"preset,key=value,...": mode=grad|textbox|textbox_recovery, probe=0|1, refine=psp|all|off|<flags>, fix=<int>
(runtime/vocr_lines_fix.h, #129's finder fixes; built with -DPAGEPATH_FIX), lines=<path to a vocr_lines.c
variant>. Presets: rubi = Rubi today (TEXTBOX_RECOVERY, its short-line probe, the refinement on PSP titles).
--finder oracle: each reference line's own box plus its pad (the recognizer's floor on the page path).
--boxes FILE: precomputed boxes, JSONL {"unit": id, "boxes": [[x, y, w, h(, dark)], ...]} in region pixels (a learned
spotter's, vocr6-spotter); a box without its dark flag gets synth/common.line_dark's. --frame-boxes FILE: JSONL
{"title", "page", "view": "vita"|"psp2x"|"psp", "boxes": [...]} in frame pixels: a unit keeps the boxes whose centre
lies in its region, clipped to it.

Everything read here is private (copyrighted frames and the owner's text): units, rows and the cache stay under
$HERD/shared/artifacts; RESULT.json holds counts only.
"""
import argparse
import hashlib
import json
import os
import pickle
import sqlite3
import struct
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "runtime", ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from metrics import norm  # noqa: E402

SCHEMA = "vocr6-pagepath/1"
FINDER_MODES = {"grad": 0, "legacy": 1, "textbox": 2, "textbox_recovery": 3}
REFINE_ALL = 0x1F                       # VOCR_REFINE_DEFAULT
PRESETS = {
    "rubi": dict(mode="textbox_recovery", probe=1, refine="psp", fix=0),
    "rubi-noprobe": dict(mode="textbox_recovery", probe=0, refine="psp", fix=0),
    "textbox_recovery": dict(mode="textbox_recovery", probe=0, refine="off", fix=0),
    "textbox": dict(mode="textbox", probe=0, refine="off", fix=0),
    "grad": dict(mode="grad", probe=0, refine="off", fix=0),
    "grad+refine": dict(mode="grad", probe=0, refine="all", fix=0),
}
TOPK = 8
LM_OPTS = dict(beam=8, topk=8, alpha=500, beta=3500, gate=8000, flags=2, bracket=8000)   # core/vocr_ocr.c lm_opts
REDECODE = (6000, 4000)                 # VJO_VOCR_COUNTER_TAU / _OPEN
CASCADE_CONF = 700                      # VJO_VOCR_CASCADE_CONF
NEXT_TIER = {"S": "N", "N": "L", "L": None}
BLOBS = {"S": "VOCR5_S_w8.vocr", "N": "VOCR5_N_w8.vocr", "L": "VOCR5_L_w8.vocr"}
MAX_WIDTH = 1024                        # the model's clamp (Rubi: 1,024 px / 8 frames)
BOOT_REPS, BOOT_SEED = 10000, 0


# ----------------------------------------------------------------------------- small helpers

def sha256_file(path, bufsize=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, rows):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def lev(a, b):
    """Levenshtein distance of two strings: the bit-parallel algorithm of Myers (1999) in Hyyro's form for the
    global distance, on Python integers (any length). Equals eval/metrics.lev (tests/test_pagepath.py)."""
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    m = len(b)
    if not m:
        return len(a)
    peq = {}
    for i, c in enumerate(b):
        peq[c] = peq.get(c, 0) | (1 << i)
    full = (1 << m) - 1
    top = 1 << (m - 1)
    pv, mv, score = full, 0, m
    for c in a:
        eq = peq.get(c, 0)
        xv = eq | mv
        xh = ((((eq & pv) + pv) & full) ^ pv) | eq
        ph = mv | (~(xh | pv) & full)
        mh = pv & xh
        if ph & top:
            score += 1
        elif mh & top:
            score -= 1
        ph = ((ph << 1) | 1) & full
        mh = (mh << 1) & full
        pv = mh | (~(xv | ph) & full)
        mv = ph & xv
    return score


def has_japanese(t):
    """Rubi core/utf.h vjo_is_japanese_char over a line (core/textfilter.c, non_japanese_filter = lines)."""
    return any(0x3040 <= ord(c) <= 0x309F or 0x30A0 <= ord(c) <= 0x30FF or 0x4E00 <= ord(c) <= 0x9FAF for c in t)


def line_conf(g, cmin):
    """core/vocr_ocr.c line_conf: conf_min in 1/1000, 1000 for a line read as empty."""
    if not g:
        return 1000
    v = cmin * 1000.0 + 0.5
    return 0 if v <= 0 else 1000 if v >= 1000 else int(v)


def compose(n_read, up_read, cascade_conf=CASCADE_CONF):
    """Rubi's text for one crop before textnorm (core/vocr_ocr.c finish_line): n_read / up_read = (greedy, beam,
    conf_min) of the job's tier and of the next tier up (None: no cascade)."""
    g, b, cmin = n_read
    if not g:
        return ""                      # a line read as empty stays empty
    if up_read is not None and line_conf(g, cmin) < cascade_conf:
        g2, b2, _ = up_read
        return b2 if g2 else ""
    return b


def crop_key(img, invert):
    h = hashlib.blake2b(digest_size=16)
    h.update(struct.pack("<3I", img.shape[0], img.shape[1], int(bool(invert))))
    h.update(np.ascontiguousarray(img).tobytes())
    return h.hexdigest()


# ----------------------------------------------------------------------------- frames and regions

class Frames:
    """Grey frames of the hooked pages in a view, cached for the pages of one unit batch."""

    def __init__(self, pages_dir, view="console"):
        self.dir = Path(pages_dir)
        self.view = view
        self.cache = {}

    def scale(self, platform):
        return 2 if (self.view == "console" and platform == "psp") else 1

    def grey(self, tid, page, platform):
        k = (tid, page)
        if k not in self.cache:
            import cv2

            from common import to_gray
            bgr = cv2.imread(str(self.dir / tid / "pages" / f"{page}.png"), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(self.dir / tid / "pages" / f"{page}.png")
            rgb = np.ascontiguousarray(bgr[..., ::-1])
            s = self.scale(platform)
            if s != 1:
                rgb = cv2.resize(rgb, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            if len(self.cache) > 64:
                self.cache.clear()
            self.cache[k] = to_gray(rgb)
        return self.cache[k]


def cells_of(line):
    """Character cells of a page-pool line, [[x0, y0, x1, y1]] native pixels (the renderer's cell boxes: a furigana
    line's hold its main text only); a line without cells gets its box cut into one even cell per character."""
    cb = line.get("cell_boxes") or []
    if cb:
        return [[float(v) for v in b[:4]] for b in cb]
    x, y, w, h = line.get("box_main") or line["box"]
    n = max(1, len(norm(line["text"])))
    return [[x + i * w / n, y, x + (i + 1) * w / n, y + h] for i in range(n)]


def contrast3(g):
    """max - min of each 3x3 window, edges replicated (synth/common.contrast3, GRAD's step 2)."""
    from common import contrast3 as c3
    return c3(g)


def ink_points(img, cells, boxes):
    """One point per inked reference cell: the centroid of the cell's high-contrast pixels (GRAD's own test, its
    threshold from the line's box: max(40, 2/5 of the 99th percentile of the 3x3 contrast)), or None for a cell
    with no ink (a space). Cell boxes are advance cells, so a punctuation mark's ink can sit far from the cell's
    centre; its centroid is where the finder's box has to reach."""
    H, W = img.shape
    c = contrast3(img)
    out = []
    for cl, b in zip(cells, boxes):
        bx0, by0 = max(0, int(b[0])), max(0, int(b[1]))
        bx1, by1 = min(W, int(np.ceil(b[0] + b[2]))), min(H, int(np.ceil(b[1] + b[3])))
        sub = c[by0:by1, bx0:bx1]
        t = max(40, 2 * int(np.percentile(sub, 99)) // 5) if sub.size else 255
        pts = []
        for x0, y0, x1, y1 in cl:
            xa, ya = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
            xb, yb = min(W, int(np.ceil(x1))), min(H, int(np.ceil(y1)))
            m = c[ya:yb, xa:xb] >= t if xb > xa and yb > ya else np.zeros((0, 0), bool)
            if m.sum() < 2:
                pts.append(None)
                continue
            ys, xs = np.nonzero(m)
            pts.append((xa + float(xs.mean()), ya + float(ys.mean())))
        out.append(pts)
    return out


def hooked_units(pagepool, sets, per_title=0, seed=20261026, titles=None):
    """Region units of the hooked pages whose split is in `sets` ("hval", "test", "tune": a fixed sample of
    `per_title` train pages per title, for tuning constants). Geometry in native pixels; no pixels."""
    pagepool = Path(pagepool)
    man = json.loads((pagepool / "manifest.json").read_text(encoding="utf-8"))
    units = []
    for tid in sorted(man["titles"]):
        if titles and tid not in titles:
            continue
        rows = read_jsonl(pagepool / f"{tid}.jsonl")
        want = []
        for r in rows:
            if not r.get("regions"):
                continue
            if r["split"] in sets or (r["split"] == "train" and "tune" in sets):
                want.append(r)
        if "tune" in sets and per_title:
            tr = [r for r in want if r["split"] == "train"]
            rng = np.random.default_rng(int(hashlib.sha256(f"{seed}:{tid}".encode()).hexdigest()[:8], 16))
            keep = set(rng.choice(len(tr), size=min(per_title, len(tr)), replace=False).tolist()) if tr else set()
            tr_ids = {id(r) for i, r in enumerate(tr) if i in keep}
            want = [r for r in want if r["split"] != "train" or id(r) in tr_ids]
        for r in want:
            by_id = {ln["id"]: ln for ln in r["lines"]}
            groups = []
            for g in r["regions"]:
                key = tuple(g["lines"])
                if key not in groups:
                    groups.append(key)
                gi = groups.index(key)
                lines = [by_id[i] for i in g["lines"]]
                lines.sort(key=lambda ln: (ln["box"][1], ln["box"][0]))
                units.append({
                    "uid": f"{tid}:{r['page']}:{gi}:{g['variant']}", "set": "tune" if r["split"] == "train" else
                    r["split"], "source": "hooked", "cluster": tid, "title": tid, "page": r["page"],
                    "platform": r["platform"], "variant": g["variant"], "region": [int(v) for v in g["region"]],
                    "image_sha256": r["image_sha256"],
                    "lines": [{"id": ln["id"], "text": ln["text"], "box": ln.get("box_main") or ln["box"],
                               "pad": ln.get("pad"), "cells": cells_of(ln)} for ln in lines]})
    return units, {"pagepool_manifest_sha256": sha256_file(pagepool / "manifest.json"),
                   "importer_sha256": man.get("importer_sha256"), "splits_sha256": man.get("splits_sha256")}


def dev_units(path):
    """#119's whole-dialogue development units (vocr5-analysis units.pkl): region pixels included."""
    d = pickle.load(open(path, "rb"))
    out = []
    for i, u in enumerate(d["units"]):
        lines = []
        for ln in u["lines"]:
            x0, y0, x1, y1 = ln["box"]
            n = max(1, len(norm(ln["text"])))
            cells = [[x0 + k * (x1 - x0) / n, y0, x0 + (k + 1) * (x1 - x0) / n, y1] for k in range(n)]
            lines.append({"id": ln["id"], "text": ln["text"], "box": [x0, y0, x1 - x0, y1 - y0], "cells": cells})
        out.append({"uid": f"dev:{i}:{u['unit']}:{u['variant']}", "set": "dev", "source": u["cohort"],
                    "cluster": u["vn"], "title": u["vn"], "platform": u["kind"], "src_kind": u.get("src_kind"),
                    "variant": u["variant"], "region": None, "image": u["image"], "lines": lines,
                    "region_origin": [0, 0]})
    return out, {"dev_units_sha256": sha256_file(path)}


def unit_images(units, frames):
    """(grey region image, scale) of every unit in its view; hooked units cut from their (upscaled) frame."""
    import cv2
    for u in units:
        if u.get("image") is not None:
            s = 2 if (frames.view == "console" and u["platform"] == "psp") else 1
            img = u["image"] if s == 1 else cv2.resize(u["image"], None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
            yield u, np.ascontiguousarray(img), s, (0, 0)
            continue
        s = frames.scale(u["platform"])
        g = frames.grey(u["title"], u["page"], u["platform"])
        x0, y0, x1, y1 = (v * s for v in u["region"])
        yield u, np.ascontiguousarray(g[y0:y1, x0:x1]), s, (x0, y0)


def ref_geometry(u, s, origin):
    """Reference cell boxes [[x0, y0, x1, y1]] and line boxes [x, y, w, h] of a unit in its region's pixels (view
    scale)."""
    ox, oy = origin
    cells = [[[c[0] * s - ox, c[1] * s - oy, c[2] * s - ox, c[3] * s - oy] for c in ln["cells"]] for ln in u["lines"]]
    boxes = [[ln["box"][0] * s - ox, ln["box"][1] * s - oy, ln["box"][2] * s, ln["box"][3] * s] for ln in u["lines"]]
    return cells, boxes


# ----------------------------------------------------------------------------- the finder

def parse_finder(spec):
    name, *rest = spec.split(",")
    if name not in PRESETS:
        raise SystemExit(f"unknown finder preset {name!r} ({', '.join(PRESETS)})")
    cfg = dict(PRESETS[name], preset=name, lines=None)
    for kv in rest:
        k, _, v = kv.partition("=")
        if k == "mode":
            if v not in FINDER_MODES:
                raise SystemExit(f"unknown finder mode {v!r}")
            cfg[k] = v
        elif k in ("probe", "fix"):
            cfg[k] = int(v, 0)
        elif k == "refine":
            cfg[k] = v
        elif k == "lines":
            cfg[k] = str(Path(v).resolve())
        else:
            raise SystemExit(f"unknown finder key {k!r}")
    return cfg


def refine_flags(cfg, platform):
    r = cfg["refine"]
    if r == "off":
        return 0
    if r == "all":
        return REFINE_ALL
    if r == "psp":
        return REFINE_ALL if platform == "psp" else 0
    return int(r, 0)


def build_driver(cfg, out_dir, cc="gcc"):
    """eval/pagepath_find.c with the frozen finder (or cfg["lines"], a variant of it; or runtime/vocr_lines_fix.c
    when cfg["fix"]), cached by the sha256 of its sources."""
    rt = ROOT / "runtime"
    srcs = [ROOT / "eval" / "pagepath_find.c", rt / "vocr_lines_refine.c"]
    defs = []
    if cfg["fix"]:
        srcs += [rt / "vocr_lines_fix.c"]
        defs.append("-DPAGEPATH_FIX")
    srcs.append(Path(cfg["lines"]) if cfg["lines"] else rt / "vocr_lines.c")
    h = hashlib.sha256(" ".join([cc] + defs).encode())
    for s in srcs + [rt / "vocr_lines.h", rt / "vocr_lines_refine.h"]:
        h.update(s.read_bytes())
    exe = Path(out_dir) / f"pagepath_find_{h.hexdigest()[:16]}"
    if not exe.exists():
        exe.parent.mkdir(parents=True, exist_ok=True)
        tmp = exe.with_suffix(".tmp")
        subprocess.run([cc, "-std=c99", "-O2", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror", *defs,
                        f"-I{rt}", "-o", str(tmp), *map(str, srcs)], check=True)
        os.replace(tmp, exe)
    return exe, {"driver_sha256": h.hexdigest(), "sources": {str(s.relative_to(ROOT)) if s.is_relative_to(ROOT)
                                                            else str(s): sha256_file(s) for s in srcs}}


def run_driver(exe, jobs):
    """jobs: [(grey, mode, fix, probe, refine)] -> [(rc, found, probed, [(x, y, w, h, dark)])]."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        for img, mode, fix, probe, rf in jobs:
            h, w = img.shape
            f.write(struct.pack("<8I", w, h, mode, fix, probe, rf, 0, 0))
            f.write(np.ascontiguousarray(img, np.uint8).tobytes())
        path = f.name
    try:
        out = subprocess.run([str(exe), path], check=True, capture_output=True, text=True).stdout.splitlines()
    finally:
        os.unlink(path)
    if len(out) != len(jobs):
        raise RuntimeError(f"driver returned {len(out)} lines for {len(jobs)} regions")
    res = []
    for line in out:
        v = [int(t) for t in line.split()]
        rc, found, probed, n = v[:4]
        res.append((rc, found, probed, [tuple(v[4 + 5 * i:9 + 5 * i]) for i in range(n)]))
    return res


# ----------------------------------------------------------------------------- reading crops

_W = {}


def _init(lib, blobs, lm):
    from pyvocr import Vocr, VocrLM, load_lib
    L = load_lib(lib)
    for tag, path in blobs.items():
        r = Vocr(path, mode="w8i", max_width=MAX_WIDTH, lib=L)
        _W[tag] = (r, VocrLM(r, lm, **LM_OPTS))


def _read(job):
    key, tag, img, inv = job
    r, lm = _W[tag]
    o = r.run_topk(img, TOPK, invert=bool(inv), blank=True)
    b = ""
    if o["text"]:
        red, _ = r.redecode(o, *REDECODE)
        b = lm.decode(red["topv"], red["topc"], red["blank"])
    return key, tag, o["text"], b, float(o["conf_min"]), int(o["frames"])


class Cache:
    """Crop readings by (model tag, crop key) in SQLite: a model tag names the blob, the LM and the decode."""

    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path), timeout=600)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS reads (tag TEXT, key TEXT, g TEXT, b TEXT, cmin REAL, "
                        "frames INTEGER, PRIMARY KEY (tag, key))")

    def get(self, tag, keys):
        out = {}
        keys = list(keys)
        for i in range(0, len(keys), 900):
            part = keys[i:i + 900]
            q = f"SELECT key, g, b, cmin, frames FROM reads WHERE tag = ? AND key IN ({','.join('?' * len(part))})"
            for k, g, b, c, fr in self.db.execute(q, [tag, *part]):
                out[k] = (g, b, c, fr)
        return out

    def put(self, rows):
        self.db.executemany("INSERT OR REPLACE INTO reads VALUES (?, ?, ?, ?, ?, ?)", rows)
        self.db.commit()


def model_tag(blob, lm):
    return (f"{sha256_file(blob)[:20]}|{sha256_file(lm)[:20]}|"
            f"{json.dumps(LM_OPTS, sort_keys=True)}|rd{REDECODE[0]}/{REDECODE[1]}|k{TOPK}|w{MAX_WIDTH}")


def read_all(crops, tags, lm, cache, workers):
    """crops: {key: (img, invert)}; tags: [(tag, blob)] to read every crop with. Fills the cache."""
    lib = str(ROOT / "runtime" / "build" / "libvocr.so")
    from pyvocr import load_lib
    load_lib()                          # (re)builds runtime/build/libvocr.so if a source is newer
    for tag, blob in tags:
        have = cache.get(tag, crops)
        todo = [(k, tag, crops[k][0], crops[k][1]) for k in crops if k not in have]
        if not todo:
            continue
        t0 = time.time()
        rows = []
        with Pool(workers, initializer=_init, initargs=(lib, {tag: blob}, lm)) as pool:
            for i, (k, t, g, b, c, fr) in enumerate(pool.imap_unordered(_read, todo, chunksize=8)):
                rows.append((t, k, g, b, c, fr))
                if len(rows) >= 2000:
                    cache.put(rows)
                    rows = []
                if i % 5000 == 0:
                    print(f"read {tag[:12]} {i}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
        cache.put(rows)


def rubi_norm_tool(rubi, out_dir, cc="gcc"):
    """Rubi's core/textnorm.c as a filter (stdin lines -> normalised lines; a dropped line prints empty)."""
    src = Path(rubi) / "core" / "textnorm.c"
    h = hashlib.sha256(src.read_bytes() + (Path(rubi) / "core" / "textnorm.h").read_bytes()).hexdigest()
    exe = Path(out_dir) / f"rubi_norm_{h[:16]}"
    if not exe.exists():
        exe.parent.mkdir(parents=True, exist_ok=True)
        main = Path(out_dir) / f"rubi_norm_{h[:16]}.c"
        main.write_text('#include <stdio.h>\n#include <string.h>\n#include "textnorm.h"\n'
                        'int main(void) { static char b[1 << 16]; while (fgets(b, sizeof b, stdin)) {\n'
                        '  size_t n = strlen(b); if (n && b[n - 1] == \'\\n\') b[--n] = 0;\n'
                        '  vjo_text_normalize(b, NULL); fputs(b, stdout); fputc(\'\\n\', stdout); } return 0; }\n')
        tmp = exe.with_suffix(".tmp")
        subprocess.run([cc, "-std=c99", "-O2", f"-I{Path(rubi) / 'core'}", "-o", str(tmp), str(main), str(src)],
                       check=True)
        os.replace(tmp, exe)
    return exe, {"textnorm_sha256": sha256_file(src)}


def textnorm(exe, texts):
    if exe is None:
        return list(texts)
    inp = "".join(t.replace("\n", " ") + "\n" for t in texts)
    out = subprocess.run([str(exe)], input=inp.encode("utf-8"), capture_output=True, check=True).stdout
    lines = out.decode("utf-8").split("\n")
    if len(lines) != len(texts) + 1 or lines[-1] != "":
        raise RuntimeError("textnorm returned a different number of lines")
    return lines[:-1]


# ----------------------------------------------------------------------------- scoring

def inside(box, pt):
    x, y, w, h = box[:4]
    return x <= pt[0] < x + w and y <= pt[1] < y + h


def line_stats(boxes, points):
    """Per reference line its status (whole / cut / missed) and per box its kind (line / merged / junk), from the
    lines' inked cell points (ink_points): a box holds a cell when it holds its point."""
    pts = [[p for p in pl if p is not None] for pl in points]
    cov = [[sum(1 for p in pl if inside(b, p)) for pl in pts] for b in boxes]
    status = []
    for li, pl in enumerate(pts):
        best = max((cov[bi][li] for bi in range(len(boxes))), default=0)
        status.append("whole" if best == len(pl) else "cut" if best else "missed")
    kinds = []
    for bi in range(len(boxes)):
        held = [li for li, pl in enumerate(pts) if pl and 2 * cov[bi][li] > len(pl)]
        kinds.append("junk" if not any(cov[bi]) else "merged" if len(held) > 1 else "line")
    return status, kinds, cov


def score_unit(u, boxes, texts, points):
    ref = "".join(norm(ln["text"]) for ln in u["lines"])
    pred = "".join(norm(t) for t in texts)
    status, kinds, _ = line_stats(boxes, points)
    st = Counter(status)
    kd = Counter(kinds)
    return {"errors": lev(pred, ref), "chars": len(ref), "lines": len(u["lines"]), "boxes": len(boxes),
            "whole": st["whole"], "cut": st["cut"], "missed": st["missed"], "merged": kd["merged"],
            "junk": kd["junk"], "cells": sum(1 for pl in points for p in pl if p is not None), "status": status,
            "kinds": kinds}


def summarize(rows, key=lambda r: "all"):
    out = {}
    for r in rows:
        k = key(r)
        s = out.setdefault(k, Counter())
        s["units"] += 1
        for f in ("errors", "chars", "lines", "boxes", "whole", "cut", "missed", "merged", "junk", "frames"):
            s[f] += r.get(f, 0)
    res = {}
    for k, s in sorted(out.items()):
        res[k] = dict(s, cer=round(100.0 * s["errors"] / max(1, s["chars"]), 4),
                      line_recall=round(s["whole"] / max(1, s["lines"]), 4),
                      box_precision=round((s["boxes"] - s["junk"]) / max(1, s["boxes"]), 4))
    return res


# ----------------------------------------------------------------------------- run

def load_units(path):
    p = Path(path)
    if p.suffix == ".pkl":
        return dev_units(p)
    doc = read_jsonl(p)
    return doc[1:], doc[0]


def external_boxes(u, s, origin, region_boxes, frame_boxes):
    if region_boxes is not None:
        return [tuple(b) for b in region_boxes.get(u["uid"], [])]
    view = ("psp2x" if s == 2 else u["platform"])
    fb = frame_boxes.get((u["title"], u.get("page"), view), [])
    ox, oy = origin
    out = []
    for b in fb:
        cx, cy = b[0] + b[2] / 2.0, b[1] + b[3] / 2.0
        x0, y0 = b[0] - ox, b[1] - oy
        if not inside((0, 0, u["_w"], u["_h"]), (cx - ox, cy - oy)):
            continue
        x1, y1 = min(u["_w"], x0 + b[2]), min(u["_h"], y0 + b[3])
        x0, y0 = max(0, x0), max(0, y0)
        if x1 > x0 and y1 > y0:
            out.append((int(x0), int(y0), int(x1 - x0), int(y1 - y0), *b[4:5]))
    return out


def oracle_boxes(u, s, origin):
    """The reference lines' own crops: box (box_main) plus pad, in the unit's region pixels."""
    _, boxes = ref_geometry(u, s, origin)
    out = []
    for ln, (x, y, w, h) in zip(u["lines"], boxes):
        pad = ln.get("pad")
        pad = (4 if u["platform"] == "vita" else 2) if pad is None else pad
        pad *= s
        x0, y0 = max(0, int(np.floor(x - pad))), max(0, int(np.floor(y - pad)))
        x1, y1 = min(u["_w"], int(np.ceil(x + w + pad))), min(u["_h"], int(np.ceil(y + h + pad)))
        if x1 > x0 and y1 > y0:
            out.append((x0, y0, x1 - x0, y1 - y0))
    return out


def cmd_run(a):
    from common import line_dark
    units, umeta = load_units(a.units)
    if a.set:
        units = [u for u in units if u["set"] in a.set.split(",")]
    if a.limit:
        units = units[:a.limit]
    frames = Frames(a.pages, a.view) if a.pages else Frames(".", a.view)
    work = Path(a.work)
    tier = a.tier
    blobs = {t: Path(a.models) / BLOBS[t] for t in ("S", "N", "L")}
    tags = {t: model_tag(blobs[t], a.lm) for t in blobs}
    meta = {"schema": SCHEMA, "kind": "vocr6-pagepath-result", "units_file": str(a.units),
            "units_sha256": sha256_file(a.units), **umeta, "view": a.view, "tier": tier,
            "cascade": NEXT_TIER[tier] if not a.no_cascade else None,
            "blobs": {t: {"name": blobs[t].name, "sha256": sha256_file(blobs[t])} for t in blobs},
            "lm": {"name": Path(a.lm).name, "sha256": sha256_file(a.lm)}, "lm_opts": LM_OPTS,
            "redecode": REDECODE, "cascade_conf": CASCADE_CONF, "repo_head": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
            "pagepath_sha256": sha256_file(__file__)}
    region_boxes = frame_boxes = None
    oracle = a.finder == "oracle"
    if oracle:
        meta["finder"] = {"oracle": "each reference line's box (box_main for a furigana line) plus its pad (the line"
                                    " record's, else 4 px Vita / 2 px PSP native), clipped to the region; dark from"
                                    " synth/common.line_dark"}
        cfg = None
    elif a.boxes:
        region_boxes = {r["unit"]: r["boxes"] for r in read_jsonl(a.boxes)}
        meta["finder"] = {"boxes": str(a.boxes), "boxes_sha256": sha256_file(a.boxes)}
        cfg = None
    elif a.frame_boxes:
        frame_boxes = {(r["title"], r.get("page"), r["view"]): r["boxes"] for r in read_jsonl(a.frame_boxes)}
        meta["finder"] = {"frame_boxes": str(a.frame_boxes), "frame_boxes_sha256": sha256_file(a.frame_boxes)}
        cfg = None
    else:
        cfg = parse_finder(a.finder)
        exe, dmeta = build_driver(cfg, work / "bin")
        meta["finder"] = dict(cfg, **dmeta)
    norm_exe = None
    if a.rubi:
        norm_exe, nmeta = rubi_norm_tool(a.rubi, work / "bin")
        meta["textnorm"] = dict(nmeta, rubi_head=subprocess.run(["git", "rev-parse", "HEAD"], cwd=a.rubi,
                                                                 capture_output=True, text=True).stdout.strip())
    else:
        meta["textnorm"] = None
    t0 = time.time()
    # 1. boxes and crops
    plan = []                                   # per unit: (u, boxes, dark, crop keys, cells, probe info)
    crops = {}
    batch = []

    def flush():
        res = run_driver(exe, [(img, FINDER_MODES[cfg["mode"]], cfg["fix"], cfg["probe"],
                                refine_flags(cfg, u["platform"])) for u, img, s, o in batch])
        for (u, img, s, o), (rc, found, probed, bx) in zip(batch, res):
            add(u, img, s, o, [b[:4] for b in bx], [b[4] for b in bx], rc, found, probed)
        batch.clear()

    def add(u, img, s, o, boxes, dark, rc, found, probed):
        keys = []
        for (x, y, w, h), dk in zip(boxes, dark):
            c = np.ascontiguousarray(img[y:y + h, x:x + w])
            if c.size == 0:
                keys.append(None)
                continue
            k = crop_key(c, dk)
            crops.setdefault(k, (c, int(dk)))
            keys.append(k)
        cells, rboxes = ref_geometry(u, s, o)
        points = ink_points(img, cells, rboxes)
        plan.append((u, [list(b) for b in boxes], list(dark), keys, points, {"rc": rc, "found": found,
                                                                                "probed": probed}))

    for u, img, s, o in unit_images(units, frames):
        u["_h"], u["_w"] = img.shape
        if cfg is None:
            bx = oracle_boxes(u, s, o) if oracle else external_boxes(u, s, o, region_boxes, frame_boxes)
            boxes = [tuple(int(v) for v in b[:4]) for b in bx]
            dark = [int(b[4]) if len(b) > 4 else line_dark(img, b[:4]) for b in bx]
            add(u, img, s, o, boxes, dark, 0, len(boxes), 0)
            continue
        batch.append((u, img, s, o))
        if len(batch) >= 256:
            flush()
    if batch:
        flush()
    print(f"found boxes for {len(plan)} units, {len(crops)} distinct crops, {time.time() - t0:.0f}s", flush=True)
    # 2. readings
    cache = Cache(a.cache)
    up = NEXT_TIER[tier] if not a.no_cascade else None
    read_all(crops, [(tags[tier], blobs[tier])], a.lm, cache, a.workers)
    first = cache.get(tags[tier], crops)
    if up:
        low = {k: crops[k] for k, (g, b, c, fr) in first.items() if g and line_conf(g, c) < CASCADE_CONF}
        read_all(low, [(tags[up], blobs[up])], a.lm, cache, a.workers)
        second = cache.get(tags[up], low)
    else:
        second = {}
    # 3. text and scores
    raw = []
    for u, boxes, dark, keys, cells, info in plan:
        for k in keys:
            if k is None:
                raw.append("")
                continue
            g, b, c, fr = first[k]
            up_read = second.get(k)
            raw.append(compose((g, b, c), (up_read[0], up_read[1], up_read[2]) if up_read else None))
    normed = textnorm(norm_exe, raw)
    shown = [t if has_japanese(t) else "" for t in normed]
    rows, i = [], 0
    for u, boxes, dark, keys, points, info in plan:
        texts = shown[i:i + len(keys)]
        i += len(keys)
        r = score_unit(u, boxes, texts, points)
        r.update(uid=u["uid"], set=u["set"], cluster=u["cluster"], platform=u["platform"], variant=u["variant"],
                 source=u["source"], frames=sum(first[k][3] for k in keys if k), cascaded=sum(
                     1 for k in keys if k and k in second), **info)
        r["boxes_xywh"] = boxes
        r["dark"] = dark
        r["texts"] = texts
        rows.append(r)
    if a.rows:
        write_jsonl(a.rows, rows)
    pub = [{k: v for k, v in r.items() if k not in ("texts", "status", "kinds")} for r in rows]
    meta["seconds"] = round(time.time() - t0, 1)
    meta["summary"] = {"all": summarize(pub), "by_platform": summarize(pub, lambda r: f"{r['set']}:{r['platform']}"),
                       "by_variant": summarize(pub, lambda r: f"{r['set']}:{r['variant']}"),
                       "by_title": summarize(pub, lambda r: f"{r['set']}:{r['cluster']}")}
    meta["units"] = [{k: r[k] for k in ("uid", "set", "cluster", "platform", "variant", "errors", "chars", "lines",
                                        "boxes", "whole", "cut", "missed", "merged", "junk", "frames")} for r in pub]
    Path(a.out).write_text(json.dumps(meta, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
    s = meta["summary"]["by_platform"]
    for k, v in s.items():
        print(f"{k}: units {v['units']} CER {v['cer']:.3f}% ({v['errors']}/{v['chars']}) recall {v['line_recall']:.3f}"
              f" precision {v['box_precision']:.3f} cut {v['cut']} missed {v['missed']} merged {v['merged']}"
              f" junk {v['junk']}", flush=True)
    print("PAGEPATH_RUN_DONE", a.out, flush=True)


# ----------------------------------------------------------------------------- compare

def cluster_bootstrap(rows_a, rows_b, reps=BOOT_REPS, seed=BOOT_SEED):
    """(CER(a) - CER(b)) in points with whole clusters resampled (sorted cluster ids, cluster sums,
    default_rng(seed).integers, numpy's linear 2.5 / 97.5 percentiles: eval/vocr3_report.cluster_bootstrap)."""
    a = {r["uid"]: r for r in rows_a}
    b = {r["uid"]: r for r in rows_b}
    if set(a) != set(b) or any(a[k]["chars"] != b[k]["chars"] for k in a):
        raise ValueError("compare needs the same units with the same references")
    sums = defaultdict(lambda: [0, 0, 0])
    for k in a:
        s = sums[a[k]["cluster"]]
        s[0] += a[k]["errors"]
        s[1] += b[k]["errors"]
        s[2] += a[k]["chars"]
    keys = sorted(sums)
    ea = np.array([sums[k][0] for k in keys], np.float64)
    eb = np.array([sums[k][1] for k in keys], np.float64)
    n = np.array([sums[k][2] for k in keys], np.float64)
    tot_n = max(1.0, float(n.sum()))
    out = {"clusters": len(keys), "chars": int(n.sum()), "arm_errors": int(ea.sum()), "base_errors": int(eb.sum()),
           "arm_cer": round(100 * ea.sum() / tot_n, 4), "base_cer": round(100 * eb.sum() / tot_n, 4),
           "delta": round(100 * (ea.sum() - eb.sum()) / tot_n, 4), "reps": reps, "seed": seed}
    if len(keys) < 2 or n.sum() == 0:
        return dict(out, ci95=None)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), (reps, len(keys)))
    tot = n[idx].sum(1)
    ok = tot > 0
    ds = (ea[idx].sum(1)[ok] - eb[idx].sum(1)[ok]) / tot[ok]
    lo, hi = np.percentile(ds, [2.5, 97.5])
    worse = [k for k in keys if sums[k][0] > sums[k][1]]
    better = [k for k in keys if sums[k][0] < sums[k][1]]
    return dict(out, ci95=[round(100 * float(lo), 4), round(100 * float(hi), 4)],
                clusters_better=len(better), clusters_worse=len(worse))


def cmd_compare(a):
    A = json.loads(Path(a.arm).read_text(encoding="utf-8"))
    B = json.loads(Path(a.base).read_text(encoding="utf-8"))
    res = {}
    groups = defaultdict(lambda: ([], []))
    ua = {u["uid"]: u for u in A["units"]}
    for ub in B["units"]:
        u = ua[ub["uid"]]
        for g in (f"{u['set']}:all", f"{u['set']}:{u['platform']}"):
            groups[g][0].append(u)
            groups[g][1].append(ub)
    for g, (ra, rb) in sorted(groups.items()):
        res[g] = cluster_bootstrap(ra, rb)
        c = res[g]
        print(f"{g}: {c['base_cer']:.3f}% -> {c['arm_cer']:.3f}%  delta {c['delta']:+.3f} "
              f"CI {c['ci95']}  clusters {c['clusters']} (better {c.get('clusters_better')}, worse "
              f"{c.get('clusters_worse')})", flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps({"base": a.base, "arm": a.arm, "base_sha256": sha256_file(a.base),
                                           "arm_sha256": sha256_file(a.arm), "compare": res}, indent=1) + "\n")


# ----------------------------------------------------------------------------- units

def cmd_units(a):
    sets = set(a.set.split(","))
    units, meta = hooked_units(a.pagepool, sets, a.per_title, a.seed, set(a.titles.split(",")) if a.titles else None)
    head = {"schema": SCHEMA, "kind": "vocr6-pagepath-units", "sets": sorted(sets), "per_title": a.per_title,
            "seed": a.seed, **meta, "units": len(units),
            "counts": dict(Counter(f"{u['set']}:{u['platform']}:{u['variant']}" for u in units))}
    write_jsonl(a.out, [head] + units)
    print(json.dumps(head["counts"], indent=0))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("units")
    p.add_argument("--pagepool", required=True)
    p.add_argument("--set", required=True, help="hval, test, tune (comma separated)")
    p.add_argument("--per-title", type=int, default=40, help="tune: train pages per title")
    p.add_argument("--seed", type=int, default=20261026)
    p.add_argument("--titles")
    p.add_argument("--out", required=True)
    p = sub.add_parser("run")
    p.add_argument("--units", required=True, help="UNITS.jsonl (pagepath.py units) or #119's units.pkl")
    p.add_argument("--set", help="keep these sets only (comma separated)")
    p.add_argument("--pages", help="the extracted pages: PAGES/<title>/pages/NNNNNN.png")
    p.add_argument("--view", default="console", choices=("console", "native"))
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--finder")
    g.add_argument("--boxes")
    g.add_argument("--frame-boxes")
    p.add_argument("--models", required=True, help="directory with VOCR5_{S,N,L}_w8.vocr")
    p.add_argument("--lm", required=True)
    p.add_argument("--rubi", help="a Rubi_for_vita checkout (core/textnorm.c); without it no textnorm")
    p.add_argument("--tier", default="N", choices=("S", "N", "L"))
    p.add_argument("--no-cascade", action="store_true")
    p.add_argument("--cache", required=True)
    p.add_argument("--work", default=str(Path(tempfile.gettempdir()) / "pagepath"))
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--rows")
    p.add_argument("--out", required=True)
    p = sub.add_parser("compare")
    p.add_argument("--base", required=True)
    p.add_argument("--arm", required=True)
    p.add_argument("--out")
    a = ap.parse_args(argv)
    {"units": cmd_units, "run": cmd_run, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()
