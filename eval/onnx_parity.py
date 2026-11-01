#!/usr/bin/env python3
"""The ONNX exports (tools/export_onnx.py) against the C runtime on development data: issue #163's pre-registered
rules R1-R4 (docs/onnx.md, "Parity").

  onnx_parity.py --onnx DIR --blobs DIR --out RESULT.json [--sets fixed,regions,frames,tsdev,stress]
                 [--workers 5] [--limit N] [--cbin DIR] [--variants ortconv,...]

DIR (--onnx) holds tools/export_onnx.py release's files; --blobs the v0.5.0 blobs they came from and the LMs. Sets
(development data only; no protected set is read):
- fixed: the 1,724 fixed development lines (#119's units file: teacher_val dev + devnative), each read in both
  polarities;
- regions: the 1,161 development whole-dialogue regions (eval/textspot6.py crop_jobs: a PSP region upscaled 2x as
  Adrenaline shows it), each read as Rubi reads a region: the spotter on the region alone at its frame's scale;
- frames: the 354 rebuilt console-view frames of those units, and tsdev: textspot-dev's 188 frames, each read whole;
- stress: random crops (1x1 to 2049x19) and random frames (1x1 to 1921x1081).

Checks, per tier (S, N, L) for every line crop (the fixed lines, every horizontal box the C spotter returns, the
random crops), each against runtime/vocr.c through runtime/pyvocr.py:
- R1 exact: vs W8I with default options (head sections off): logits (every frame and class), best, top-8 values and
  classes, blank logit, greedy text, conf_min and conf_mean, bit for bit; vs W8I with the blob's sections (N: head
  gate on + two-level, L: two-level): the same but the logits, plus the gate and escape scores where the C runtime
  computes them;
- R3 fast (reported): text, best and character runs vs W8F and vs W8I (sections off and on), the largest logit
  difference;
- the LM: the beam + LM text (runtime/vocr_lm.c at Rubi's settings for the tier, through pyvocr) on the C runtime's
  top-8 vs tools/vocr_onnx.py CharLM on the exact (and the fast) model's top-8, for the polarity Rubi reads (the
  spotter's dark flag; a fixed line: vocr_line_dark of the crop).
Per frame or region (R2): the spotter's decode maps (runtime/vocr_textspot6_test --maps) and its result (--batch:
every box, block, pick, truncated, presence) vs the exact and the int8 ONNX spotter decoded by tools/vocr_onnx.py.
R4 (pipeline): tools/vocr_onnx.read_region with the exact files and the 1 MB LM vs the C path on the same region
(the C spotter's horizontal boxes, each crop read by the C runtime with the blob's sections and decoded by the C LM):
the boxes, dark flags, greedy and LM texts and every character's frames and cell.

--variants: other recognizer files in DIR (VOCR6.2_<tier>_<v>.onnx), checked like the fast files and reported, never
gated: e.g. ortconv, astra's native-int8 form (tools/export_onnx.py recognizer --convs ort: the exact matrices, ONNX
Runtime's float32 convolutions).

RESULT.json holds counts and line ids only (no image, text or reading).
"""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import struct
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "tools", ROOT / "runtime", ROOT / "eval", ROOT / "synth", ROOT / "train"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import charbox as CB  # noqa: E402

ART = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr")) / "shared" / "artifacts"
UNITS = ART / "vocr5-analysis" / "work" / "units.pkl"
TIERS = (("S", "VOCR6.2_S_w8.vocr"), ("N", "VOCR6.2_N_w8g2.vocr"), ("L", "VOCR6.2_L_w8_2l.vocr"))
SPOT_BLOB = "textspot8_t64n8c16_w8.vtsp"
LM_FILE = "vocr_lm6_o3_1000k.vlm"
SCALE_FLAG = {1: 32, 2: 64}
SCHEMA = "vocr-onnx-parity/1"
F32 = np.float32


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def bits_eq(a, b):
    a, b = np.asarray(a, F32), np.asarray(b, F32)
    return a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32))


# ----------------------------------------------------------------------------- workers

_W = {}


def _init(onnx_dir, blobs_dir, threads, variants=()):
    import pyvocr
    import vocr_onnx as V

    import head2level
    lib = pyvocr.load_lib()
    _W["V"] = V
    _W["tiers"] = {}
    for t, blob in TIERS:
        data = (Path(blobs_dir) / blob).read_bytes()
        plain = pyvocr.Vocr(data, mode="w8i", lib=lib)
        w8f = pyvocr.Vocr(data, mode="w8f", lib=lib)
        two = head2level.read_trailer(data) is not None
        sec = pyvocr.Vocr(data, mode="w8i", lib=lib, head_gate="on" if plain.has_gate else "off",
                          two_level=1 if two else 0) if (plain.has_gate or two) else None
        cl = V.CharLM(Path(blobs_dir) / LM_FILE, tier=t)
        clm = pyvocr.VocrLM(sec or plain, str(Path(blobs_dir) / LM_FILE), beam=cl.beam, topk=cl.topk, alpha=cl.alpha,
                            beta=cl.beta, gate=cl.gate, flags=cl.flags, bracket=cl.bracket)
        _W["tiers"][t] = dict(plain=plain, w8f=w8f, sec=sec, clm=clm, lm=cl,
                              exact=V.Recognizer(Path(onnx_dir) / f"VOCR6.2_{t}_exact.onnx", threads),
                              fast=V.Recognizer(Path(onnx_dir) / f"VOCR6.2_{t}_fast.onnx", threads),
                              variants={v: V.Recognizer(Path(onnx_dir) / f"VOCR6.2_{t}_{v}.onnx", threads)
                                        for v in variants})
    _W["spot"] = {m: V.Spotter(Path(onnx_dir) / f"textspot8_{m}.onnx", threads) for m in ("exact", "int8")}


ALL = ("logits", "best", "topk_logits", "topk_classes", "blank_logit")


def _onnx(rec, x, sections):
    names = list(ALL)
    for n in ("gate_score", "escape_score"):
        if n in rec.outputs:
            names.append(n)
    return rec.forward(x, sections, names)


def _runs(best):
    V = _W["V"]
    return [(c, f0, f1) for c, f0, f1 in V.spans(best)]


def _conf(V, topv, best):
    sp = V.spans(best)
    if not sp:
        return 1.0, 1.0
    conf = V.topk_conf(topv)
    p = conf[[f0 for _, f0, _ in sp]]
    s = F32(0)
    for v in p:
        s = F32(s + v)
    return float(p.min()), float(F32(s / F32(len(p))))


def read_crop(img, invert, rubi, tiers=None):
    """Every check of one line crop; returns {tier: {check: bool or number}}."""
    V = _W["V"]
    out = {}
    for t, T in _W["tiers"].items():
        if tiers and t not in tiers:
            continue
        r = {}
        x, nw = T["exact"].preprocess(img, invert)
        c_text, c_logits = T["plain"].run(img, logits=True, invert=invert)
        c = T["plain"].run_topk(img, 8, invert=invert)
        f = T["w8f"].run_topk(img, 8, invert=invert)
        _, f_logits = T["w8f"].run(img, logits=True, invert=invert)
        e = _onnx(T["exact"], x, False)
        r["frames"] = int(len(c["best"]))
        r["x_eq"] = bool(np.array_equal(x, T["plain"].preprocess(img, invert)[0]))
        ecm = _conf(V, e["topk_logits"], e["best"])
        e_text = "".join(T["exact"].vocab[k] for k, _, _ in V.spans(e["best"]))
        r["exact_logits"] = bits_eq(e["logits"], c_logits)
        r["exact_best"] = bool(np.array_equal(e["best"], c["best"]))
        r["exact_topk"] = bits_eq(e["topk_logits"], c["topv"]) and bool(np.array_equal(e["topk_classes"], c["topc"]))
        r["exact_blank"] = bits_eq(e["blank_logit"], c["blank"])
        r["exact_text"] = e_text == c["text"] == c_text
        r["exact_conf"] = bool(F32(ecm[0]) == F32(c["conf_min"]) and F32(ecm[1]) == F32(c["conf_mean"]))
        fa = _onnx(T["fast"], x, False)
        f_text = "".join(T["fast"].vocab[k] for k, _, _ in V.spans(fa["best"]))
        r["fast_text_w8f"] = f_text == f["text"]
        r["fast_text_w8i"] = f_text == c["text"]
        r["fast_best_w8f"] = bool(np.array_equal(fa["best"], f["best"]))
        r["fast_best_w8i"] = bool(np.array_equal(fa["best"], c["best"]))
        r["fast_runs_w8i"] = _runs(fa["best"]) == _runs(c["best"])
        r["fast_frames_w8i"] = int((fa["best"] != c["best"]).sum())
        r["fast_maxdiff_w8f"] = float(np.abs(fa["logits"] - f_logits).max()) if len(f_logits) else 0.0
        r["fast_maxdiff_w8i"] = float(np.abs(fa["logits"] - c_logits).max()) if len(c_logits) else 0.0
        for v, rv in T["variants"].items():            # --variants: reported against W8I, never gated
            ov = _onnx(rv, x, False)
            r[f"{v}_logits"] = bits_eq(ov["logits"], c_logits)
            r[f"{v}_best_w8i"] = bool(np.array_equal(ov["best"], c["best"]))
            r[f"{v}_runs_w8i"] = _runs(ov["best"]) == _runs(c["best"])
            r[f"{v}_text_w8i"] = "".join(rv.vocab[k] for k, _, _ in V.spans(ov["best"])) == c["text"]
            r[f"{v}_frames_w8i"] = int((ov["best"] != c["best"]).sum())
        if T["sec"] is not None:
            s = T["sec"].run_topk(img, 8, invert=invert, gate_score=T["plain"].has_gate, esc=True)
            es = _onnx(T["exact"], x, True)
            esm = _conf(V, es["topk_logits"], es["best"])
            r["sec_best"] = bool(np.array_equal(es["best"], s["best"]))
            r["sec_topk"] = bits_eq(es["topk_logits"], s["topv"]) and bool(np.array_equal(es["topk_classes"],
                                                                                          s["topc"]))
            r["sec_blank"] = bits_eq(es["blank_logit"], s["blank"])
            r["sec_text"] = "".join(T["exact"].vocab[k] for k, _, _ in V.spans(es["best"])) == s["text"]
            r["sec_conf"] = bool(F32(esm[0]) == F32(s["conf_min"]) and F32(esm[1]) == F32(s["conf_mean"]))
            if "gate_score" in s:
                r["sec_gate"] = bits_eq(es["gate_score"], s["gate_score"])
                r["gated"] = int(s["gated"])
            if "escape_score" in es:
                ok = ~np.isnan(s["esc"])
                r["sec_escape"] = bits_eq(es["escape_score"][ok], s["esc"][ok])
                r["escaped"] = int(s["escaped"])
            fs = _onnx(T["fast"], x, True)
            r["fast_sec_text"] = "".join(T["fast"].vocab[k] for k, _, _ in V.spans(fs["best"])) == s["text"]
            r["fast_sec_best"] = bool(np.array_equal(fs["best"], s["best"]))
            for v, rv in T["variants"].items():
                vs = _onnx(rv, x, True)
                r[f"{v}_sec_text"] = "".join(rv.vocab[k] for k, _, _ in V.spans(vs["best"])) == s["text"]
                r[f"{v}_sec_best"] = bool(np.array_equal(vs["best"], s["best"]))
                r[f"{v}_sec_frames"] = int((vs["best"] != s["best"]).sum())
            lm_c_in, lm_e, lm_f = s, es, fs
        else:
            lm_c_in, lm_e, lm_f = c, e, fa
        if rubi:
            if lm_c_in["text"]:
                c_lm = T["clm"].decode(lm_c_in["topv"], lm_c_in["topc"], lm_c_in["blank"])
                r["lm_exact"] = T["lm"].decode(T["exact"], lm_e["topk_logits"], lm_e["topk_classes"],
                                               lm_e["blank_logit"]) == c_lm
                f_greedy = "".join(T["fast"].vocab[k] for k, _, _ in V.spans(lm_f["best"]))
                r["lm_fast"] = (T["lm"].decode(T["fast"], lm_f["topk_logits"], lm_f["topk_classes"],
                                               lm_f["blank_logit"]) if f_greedy else "") == c_lm
            r["c_text"] = c["text"]
            r["fast_text"] = f_text
        out[t] = r
    return out


def _crop_job(job):
    key, img, invert, rubi = job
    return key, read_crop(img, invert, rubi)


def _parse_set(line):
    v = [int(x) for x in line.split()]
    rc, n, nb, pick, trunc, pres = v[:6]
    boxes = [tuple(v[6 + 10 * i: 16 + 10 * i]) for i in range(n)]
    o = 6 + 10 * n
    blocks = [tuple(v[o + 8 * i: o + 8 + 8 * i]) for i in range(nb)]
    return rc, boxes, blocks, pick, trunc, pres


def _spots_line(r):
    parts = [0, len(r.boxes), len(r.blocks), r.pick, r.truncated, r.presence]
    for b in r.boxes:
        parts += [b.x, b.y, b.w, b.h, b.vertical, b.conf, b.dark, b.block, b.tag, b.tconf]
    for k in r.blocks:
        parts += [k.x, k.y, k.w, k.h, k.vertical, k.first, k.count, k.tag]
    return " ".join(str(int(p)) for p in parts)


def _unit_job(job):
    """One frame or region: R2 against the C driver's lines, then every C box's crop (R1/R3/LM), then R4."""
    key, g, scale, c_line, c_maps = job
    V = _W["V"]
    res = {"key": key}
    rc, boxes, blocks, pick, trunc, pres = _parse_set(c_line)
    res["c_rc"] = rc
    for m, sp in _W["spot"].items():
        f = scale if scale else sp.scale_of(g.shape[1], g.shape[0])
        x = V.halve(g) if f == 2 else g
        mp_ = sp.maps(x)
        res[f"spot_{m}_maps"] = c_maps is not None and mp_.shape == c_maps.shape and bool(np.array_equal(mp_, c_maps))
        r = sp.spot(g, scale=f)
        res[f"spot_{m}_result"] = _spots_line(r) == c_line.strip()
    crops = []
    for b in boxes:
        x0, y0, w, h, vert, conf, dark = b[:7]
        if vert:
            continue
        crops.append((x0, y0, w, h, dark))
    res["crops"] = [read_crop(np.ascontiguousarray(g[y:y + h, x:x + w]), bool(d), True) for x, y, w, h, d in crops]
    # R4: the ONNX pipeline on the region against the C path (the C boxes, read by C with the sections + the C LM)
    f = scale if scale else _W["spot"]["exact"].scale_of(g.shape[1], g.shape[0])
    pipe = {}
    for t, T in _W["tiers"].items():
        lines = V.read_region(g, _W["spot"]["exact"], T["exact"], lm=T["lm"], sections=True, scale=f)
        cr = T["sec"] or T["plain"]
        ok = len(lines) == len(crops)
        for ln, (x, y, w, h, d) in zip(lines, crops):
            img = np.ascontiguousarray(g[y:y + h, x:x + w])
            o = cr.run_topk(img, 8, invert=bool(d))
            c_lm = T["clm"].decode(o["topv"], o["topc"], o["blank"]) if o["text"] else ""
            nw = cr.resized_width(w, h)
            # the C side's cells: eval/charbox.py, runtime/vocr_charbox.c's twin (make check-charbox), on C's frames
            cb = CB.positions(o["best"], w, h, nw, flags=0, cal=T["exact"].cal, groups=T["exact"].groups)
            c_chars = [(T["exact"].vocab[p["cls"]], p["f0"], p["f1"], (x + p["c"] / 16, x + p["x0"] / 16,
                                                                     x + p["x1"] / 16)) for p in cb]
            o_chars = [(ch.text, ch.f0, ch.f1, (ch.x, ch.x0, ch.x1)) for ch in ln.chars]
            ok = ok and ln.box == (x, y, w, h) and ln.dark == d and ln.greedy == o["text"] and ln.text == c_lm \
                and o_chars == c_chars
        pipe[t] = ok
    res["pipeline"] = pipe
    return res


# ----------------------------------------------------------------------------- the C spotter driver

def build_driver(cdir):
    import textspot6_parity as TP6
    return TP6.build(cdir)[0]


def c_spot(exe, blob, jobs, mode):
    """The driver's output per job ((key, grey, scale)): --batch lines, or --maps arrays [planes, rows, cols]."""
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        for _, g, scale in jobs:
            g = np.ascontiguousarray(g, np.uint8)
            f.write(struct.pack("<4I", g.shape[1], g.shape[0], 1, SCALE_FLAG[scale] if scale else 0) + g.tobytes())
        path = f.name
    try:
        out = subprocess.run([str(exe), "--batch" if mode == "batch" else "--maps", str(blob), path],
                             capture_output=True, text=True, check=True).stdout.splitlines()
    finally:
        os.unlink(path)
    if mode == "batch":
        if len(out) != len(jobs):
            raise SystemExit(f"the driver printed {len(out)} lines for {len(jobs)} records")
        return out
    maps = []
    for i in range(len(jobs)):
        vals, tail = out[2 * i], out[2 * i + 1].split()
        rc, rows, wm = int(tail[0]), int(tail[1]), int(tail[2])
        v = np.array([int(x) for x in vals[1:].split()], np.int64)
        planes = len(v) // max(1, rows * wm) if rows * wm else 0
        maps.append(v.reshape(rows, wm, planes).transpose(2, 0, 1) if rc == 0 and rows * wm else None)
    return maps


# ----------------------------------------------------------------------------- sets

def fixed_jobs(limit=0):
    import vocr_onnx as V
    d = pickle.load(open(UNITS, "rb"))
    rows = d["fixed"][:limit] if limit else d["fixed"]
    jobs, refs = [], {}
    for i, r in enumerate(rows):                # one line id can be in two files (devnative's crop variants)
        img = np.ascontiguousarray(r["img"], np.uint8)
        h, w = img.shape
        dark = V.line_dark(img, (0, 0, w, h))
        name = "name" in (r["meta"].get("styles") or [])
        refs[i] = (r["ref_raw"], name, r["vn"])
        for inv in (False, True):
            jobs.append(((i, r["id"], inv), img, inv, inv == bool(dark)))
    return jobs, refs


def unit_jobs(set_name, limit=0):
    import textspot6 as E6
    if set_name == "regions":
        import textspot6_twin as T6
        m = T6.Model((Path(_ARGS.blobs) / SPOT_BLOB).read_bytes())
        it = ((uid, img, T6.scale_of(m, fw, fh)) for uid, img, fw, fh in E6.crop_jobs("dev"))
    elif set_name == "frames":
        it = ((k["frame"], g, None) for k, g in E6.dev_frames())
    elif set_name == "tsdev":
        it = ((k["frame"], g, None) for k, g in E6.tsdev_frames())
    elif set_name == "stress":
        rng = np.random.default_rng(20261031)
        it = ((f"random:{w}x{h}", rng.integers(0, 256, (h, w), dtype=np.uint8), None)
              for w, h in ((1, 1), (17, 9), (321, 119), (960, 544), (1921, 1081), (480, 272), (900, 150)))
    else:
        raise SystemExit(set_name)
    out = [(k, np.ascontiguousarray(g, np.uint8), s) for k, g, s in it]
    return out[:limit] if limit else out


def stress_crops():
    rng = np.random.default_rng(20261101)
    sizes = ((1, 1), (7, 33), (641, 31), (1027, 32), (2049, 19), (8, 32), (9, 33), (500, 64), (3000, 40))
    return [((f"random:{w}x{h}", inv), rng.integers(0, 256, (h, w), dtype=np.uint8), inv, True)
            for w, h in sizes for inv in (False, True)]


# ----------------------------------------------------------------------------- scoring

def lev(a, b):
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def tally(records):
    """{tier: {check: {"n": lines checked, "fail": lines failing, "ids": first failing ids}}} plus the fast model's
    largest logit differences and frame counts."""
    out = defaultdict(lambda: defaultdict(lambda: {"n": 0, "fail": 0, "ids": []}))
    extra = defaultdict(lambda: Counter())
    for key, rec in records:
        for t, r in rec.items():
            for k, v in r.items():
                if isinstance(v, (bool, np.bool_)):        # a check (numpy comparisons give np.bool_)
                    c = out[t][k]
                    c["n"] += 1
                    if not v:
                        c["fail"] += 1
                        if len(c["ids"]) < 10:
                            c["ids"].append(str(key))
                elif k.startswith("fast_maxdiff"):
                    extra[t][k] = max(extra[t][k], v)
                elif k in ("frames", "fast_frames_w8i", "gated", "escaped") or k.endswith(("_frames_w8i",
                                                                                          "_sec_frames")):
                    extra[t][k] += v
    res = {}
    for t in out:
        res[t] = {k: dict(v) for k, v in sorted(out[t].items())}
        res[t]["counts"] = dict(extra[t])
    return res


_ARGS = None


def main(argv=None):
    global _ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--blobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets", default="fixed,regions,frames,tsdev,stress")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cbin", default=str(ART / "vocr-onnx" / "cbuild6"))
    ap.add_argument("--variants", default="", help="other recognizer files in --onnx, VOCR6.2_<tier>_<v>.onnx, checked "
                    "and reported against W8I (not gated), e.g. ortconv: tools/export_onnx.py recognizer --convs ort")
    a = _ARGS = ap.parse_args(argv)
    import onnxruntime
    import pyvocr
    pyvocr.load_lib()
    t0 = time.time()
    res = {"kind": SCHEMA, "onnx": {p.name: sha256(p) for p in sorted(Path(a.onnx).glob("*.onnx"))},
           "blobs": {b: sha256(Path(a.blobs) / b) for _, b in TIERS} | {SPOT_BLOB: sha256(Path(a.blobs) / SPOT_BLOB),
                                                                        LM_FILE: sha256(Path(a.blobs) / LM_FILE)},
           "runtime_lib_sha256": sha256(pyvocr.LIB), "onnxruntime": onnxruntime.__version__, "sets": {}}
    exe = build_driver(a.cbin)
    spot_blob = Path(a.blobs) / SPOT_BLOB
    ctx = mp.get_context("fork")
    variants = tuple(v for v in a.variants.split(",") if v)
    res["variants"] = list(variants)
    with ctx.Pool(a.workers, initializer=_init, initargs=(a.onnx, a.blobs, 1, variants)) as pool:
        for s in a.sets.split(","):
            t1 = time.time()
            if s == "fixed":
                jobs, refs = fixed_jobs(a.limit)
                recs = pool.map(_crop_job, jobs, chunksize=8)
                entry = {"reads": len(recs), "lines": len(refs), "line_ids": len({k[1] for k, _ in recs}),
                         "checks": tally(recs)}
                cer = defaultdict(lambda: Counter())
                for (key, rec), job in zip(recs, jobs):
                    if not job[3]:
                        continue
                    ref, name, _ = refs[key[0]]
                    from metrics import norm
                    rn = norm(ref)
                    part = "names" if name else "dialogue"
                    for t, r in rec.items():
                        cer[t][part + "_chars"] += len(rn)
                        cer[t][part + "_c_w8i"] += lev(rn, norm(r["c_text"]))
                        cer[t][part + "_onnx_fast"] += lev(rn, norm(r["fast_text"]))
                entry["cer"] = {t: dict(v) for t, v in cer.items()}
                for t, r in entry["checks"].items():
                    for k in list(r):
                        if k in ("c_text", "fast_text"):
                            del r[k]
                res["sets"][s] = entry
            else:
                units = unit_jobs(s, a.limit)
                lines, maps = [], []
                for k in range(0, len(units), 200):
                    part = units[k:k + 200]
                    lines += c_spot(exe, spot_blob, part, "batch")
                    maps += c_spot(exe, spot_blob, part, "maps")
                jobs = [(k, g, sc, ln, m) for (k, g, sc), ln, m in zip(units, lines, maps)]
                out = pool.map(_unit_job, jobs, chunksize=2)
                spot = Counter()
                pipe = defaultdict(Counter)
                recs = []
                first = {}
                for r in out:
                    for m in ("exact", "int8"):
                        for what in ("maps", "result"):
                            k = f"spot_{m}_{what}"
                            spot[k + "_n"] += 1
                            if not r[k]:
                                spot[k + "_fail"] += 1
                                first.setdefault(k, str(r["key"]))
                    for t, ok in r["pipeline"].items():
                        pipe[t]["n"] += 1
                        pipe[t]["fail"] += int(not ok)
                        if not ok:
                            first.setdefault(f"pipeline_{t}", str(r["key"]))
                    recs += [((r["key"], i), c) for i, c in enumerate(r["crops"])]
                checks = tally(recs)
                for t, r in checks.items():
                    for k in ("c_text", "fast_text"):
                        r.pop(k, None)
                res["sets"][s] = {"units": len(out), "crops": len(recs), "spotter": dict(spot),
                                  "pipeline": {t: dict(v) for t, v in pipe.items()}, "first_failure": first,
                                  "checks": checks}
                if s == "stress":
                    sc = pool.map(_crop_job, stress_crops(), chunksize=1)
                    res["sets"]["stress_crops"] = {"reads": len(sc), "checks": tally(sc)}
                    for t, r in res["sets"]["stress_crops"]["checks"].items():
                        for k in ("c_text", "fast_text"):
                            r.pop(k, None)
            res["sets"][s]["seconds"] = round(time.time() - t1, 1)
            print(f"{s}: {json.dumps(summary(res['sets'][s]))}", flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    res["verdict"] = verdict(res)
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("verdict", json.dumps(res["verdict"]), flush=True)


R1 = ("exact_logits", "exact_best", "exact_topk", "exact_blank", "exact_text", "exact_conf", "x_eq",
      "sec_best", "sec_topk", "sec_blank", "sec_text", "sec_conf", "sec_gate", "sec_escape", "lm_exact")
# R1's checks every tier must have counted somewhere, and those a tier with head sections must have counted (sec_gate
# and sec_escape depend on which section a blob has): a check that never reaches the tally fails the verdict
R1_ALWAYS = ("exact_logits", "exact_best", "exact_topk", "exact_blank", "exact_text", "exact_conf", "x_eq", "lm_exact")
R1_SECTIONS = ("sec_best", "sec_topk", "sec_blank", "sec_text", "sec_conf")


def summary(entry):
    """Failures per tier of the gated checks, the spotter's and the pipeline's."""
    out = {}
    for t, r in entry.get("checks", {}).items():
        out[t] = {k: r[k]["fail"] for k in R1 if k in r and r[k]["fail"]}
        out[t]["fast_text_w8i_fail"] = r.get("fast_text_w8i", {}).get("fail", 0)
        out[t]["lines"] = r.get("exact_text", {}).get("n", 0)
    if "spotter" in entry:
        out["spotter_fail"] = {k: v for k, v in entry["spotter"].items() if k.endswith("_fail")}
        out["pipeline_fail"] = {t: v.get("fail", 0) for t, v in entry["pipeline"].items()}
    return out


def missing_checks(res):
    """{tier: [R1 checks that no set counted]} (R1_ALWAYS for every tier, R1_SECTIONS for a tier with a sec_ check)."""
    seen = {}
    for e in res["sets"].values():
        for t, r in e.get("checks", {}).items():
            seen.setdefault(t, set()).update(k for k, v in r.items() if isinstance(v, dict) and v.get("n"))
    out = {}
    for t, keys in seen.items():
        need = list(R1_ALWAYS) + (list(R1_SECTIONS) if any(k.startswith("sec_") for k in keys) else [])
        miss = [k for k in need if k not in keys]
        if miss:
            out[t] = miss
    return out


def verdict(res):
    """R1, R2 and R4 pass when no gated check failed anywhere, and (R1) every check of R1_ALWAYS and, for a tier with
    head sections, R1_SECTIONS was counted."""
    r1 = r2 = r4 = True
    for s, e in res["sets"].items():
        for t, r in e.get("checks", {}).items():
            for k in R1:
                if k in r and r[k]["fail"]:
                    r1 = False
        for k, v in e.get("spotter", {}).items():
            if k.endswith("_fail") and v:
                r2 = False
        for t, v in e.get("pipeline", {}).items():
            if v.get("fail"):
                r4 = False
    miss = missing_checks(res)
    return {"R1_exact_recognizer": r1 and not miss, "R2_spotter": r2, "R4_pipeline": r4, "R1_missing_checks": miss}


if __name__ == "__main__":
    main()
