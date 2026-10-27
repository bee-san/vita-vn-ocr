#!/usr/bin/env python3
"""T4 (vocr6-lm, #127): Rubi's character LM decoded the way Rubi decodes it, on the hooked hval set and on VOCR5's
development cohorts (eval/vocr6_lm_protocol.json; docs/vocr6-lm.md).

  vocr6_lm.py hval-cache --tier T --blob B --hval HVAL.pkl --out DIR        (heavy; private)
  vocr6_lm.py screen --lm NAME=PATH ... --hval-text HVAL.txt --out SCREEN.json
  vocr6_lm.py tune --lm NAME=PATH --grid JSON --tiers S,N,L --out DIR/NAME.npz [--hval-cache DIR] [--procs 5]
  vocr6_lm.py select --protocol eval/vocr6_lm_protocol.json --tune DIR --out eval/vocr6_lm_dev_<date>.json

Reads. Every crop the console would hand the recognizer, read once per tier (VOCR5-S/N/L, the release blobs, W8I,
head gate off) through the C runtime with the top-8 extras, the blank's logit and conf_min:
- dev: VOCR5's development cohorts, the reads of the head gate's development cache ("today": the release blobs;
  eval/headgate_eval.py cache): tv (teacher_val's development part) and dn (devnative's four development files),
  both as stored (rv: the console's polarity) and with the inversion undone (rvn), as VOCR5's score reads them; wd
  (tv_dlg + dn_dlg, every region through the C GRAD finder with flags 0 and with the refinement);
- hval: the hooked corpus' screening pages (data/import_hooked.py's split), hc: their verified lines as clean crops,
  hp: their whole-dialogue regions through the page path (hval-cache; module of the same name below).

Rubi's line path (Rubi_for_vita core/vocr_ocr.c finish_line, as eval/headgate_eval.Decoder.rubi models it): a line
read as empty stays empty; else vita-vn-ocr #109's counter re-decode (tau 6, open 4 nats) of the top-K, the beam +
LM decode of the (edited) top-K with the LM's settings, and the counter text rule; then the confidence cascade: a
line whose greedy confidence (conf_min x 1000 + 0.5, truncated) is under the threshold shows the next tier up's
reading of the same crop, decoded by the same steps with the same LM (S -> N, N -> L; L has none).

Scores (dialogue CER; names a secondary column). tv/dn: per line, the edit distance of each view's text against the
reference (eval/metrics.norm on both), over both views' reference characters; wd and hp: per region, the edit
distance of the found lines' texts joined in reading order against the region's reference lines joined (= the
error count of train/general_vn_textbox_guard.transcript_score's "all"/"full", which VOCR5's wd uses); hc: per line.
dev score = 0.25 tv + 0.25 dn + 0.5 wd (VOCR5's); the protocol's objective = 0.5 hp + 0.5 dev score. Intervals:
cluster bootstrap (10,000 resamples; VNs for tv/dn/wd, titles for hc/hp; each component resampled on its own).

Every file this script writes outside eval/ holds development crops' readings or the owner's game text: private.
The JSON it writes into eval/ holds counts, rates and hashes only.
"""
import argparse
import datetime
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "train", ROOT / "synth", ROOT / "runtime", ROOT / "runtime" / "tests", ROOT / "lm"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from metrics import norm  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ART = HERD / "shared" / "artifacts"
PRIV = HERD / "shared" / "private" / "vocr6-lm"
DEV_CACHE = HERD / "shared" / "private" / "head-gate" / "dev"
BLOBS = {"S": ART / "vocr5-train" / "blobs" / "v5f_S_seed2_w8.vocr",
         "N": ART / "vocr5-train" / "blobs" / "v5f_N_seed3_w8.vocr",
         "L": ART / "vocr5-train" / "blobs" / "v5f_L_seed2_w8.vocr"}
OLD_LMS = {t: ART / "ctc-lm-decode" / "release_pub2" / f"vocr_lm_o3_{t}.vlm" for t in ("500k", "1000k", "2000k",
                                                                                     "4000k")}
RUBI_BETA = {"500k": 3000, "1000k": 3500, "2000k": 3000, "4000k": 3500}     # Rubi core/vocr_ocr.c lms[]
RUBI_OPTS = {"beam": 8, "topk": 8, "alpha": 500, "gate": 8000, "flags": 2, "bracket": 8000}
COUNTER = (6000, 4000)            # Rubi's VJO_VOCR_COUNTER_TAU / _OPEN
CASCADE_CONF = 700                # Rubi's VJO_VOCR_CASCADE_CONF
UP = {"S": "N", "N": "L", "L": None}
K = 8
EXCLUDE_DEV_VNS = {"v12188": "a hooked train title (PCSG00576): its text is in the new LMs",
                   "v18432": "a hooked test title (PCSG00807, Dance with Devils)"}
# hooked train titles whose text is in the new LMs and which relate to a protected set (vocr6-common: flag as seen)
SEEN_TITLES = {
    "PCSG00866": "Collar x Malice (v18160): shares a release with vita-test's Collar x Malice -Unlimited- (v21789)",
    "ULJM05383": "planetarian (v34): shares a release with CLANNAD",
    "PCSG00007": "Little Busters! Converted Edition (v5, unmapped in its metadata): shares a release with CLANNAD"}
WD = ("tv_dlg", "dn_dlg")
N_BOOT = 10000
SEEDS = {"tv": 0, "dn": 1, "wd": 2, "hp": 3, "hc": 4}
DEV_W = {"tv": 0.25, "dn": 0.25, "wd": 0.5}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def lev(a, b):
    """Levenshtein distance (bit-parallel, Hyyrö 2001); equal to eval/metrics.lev (tests/test_vocr6_lm.py)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    m = len(a)
    peq = {}
    for i, c in enumerate(a):
        peq[c] = peq.get(c, 0) | (1 << i)
    mask, hi = (1 << m) - 1, 1 << (m - 1)
    pv, mv, score = mask, 0, m
    for c in b:
        eq = peq.get(c, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | (~(xh | pv) & mask)
        mh = pv & xh
        if ph & hi:
            score += 1
        elif mh & hi:
            score -= 1
        ph = ((ph << 1) | 1) & mask
        mh = (mh << 1) & mask
        pv = mh | (~(xv | ph) & mask)
        mv = ph & xv
    return score


def fold_of(cluster):
    """The protocol's 2-fold split of clusters (VNs, hooked titles): sha256 of the id, low bit."""
    return int(hashlib.sha256(cluster.encode()).hexdigest(), 16) % 2


def line_conf(o):
    """Rubi's core/vocr_ocr.c line_conf: conf_min x 1000 + 0.5 in float, truncated; 1000 for an empty line."""
    if not o["text"]:
        return 1000
    c = np.float32(np.float32(o["conf_min"]) * np.float32(1000.0) + np.float32(0.5))
    return 0 if c <= 0 else 1000 if c >= 1000 else int(c)


# ---------------------------------------------------------------- units

class Unit:
    """One scored row: a part (tv, dn, wd, hc, hp), a cluster (VN or hooked title), a name flag, and segments
    [(reference, read keys)]: its errors are the sum over segments of the edit distance between the reference and
    the segment's reads' texts joined (each normalised)."""
    __slots__ = ("part", "uid", "cluster", "name", "segs", "chars", "memo", "platform")

    def __init__(self, part, uid, cluster, name, segs, platform=None):
        self.part, self.uid, self.cluster, self.name, self.platform = part, uid, cluster, bool(name), platform
        self.segs = [(ref, tuple(keys)) for ref, keys in segs]
        self.chars = sum(len(r) for r, _ in self.segs)
        self.memo = {}

    def state(self):
        """Unit(*state()) rebuilds it (plain data: the private caches never pickle this class)."""
        return (self.part, self.uid, self.cluster, self.name, [(r, list(k)) for r, k in self.segs], self.platform)

    def errors(self, texts, show=None):
        """The unit's errors under texts {key: raw text}; show (RubiShow) applies Rubi's display rules to each
        segment's lines first (protocol amendment 1)."""
        e = 0
        for i, (ref, keys) in enumerate(self.segs):
            if show is None:
                pred = "".join(norm(texts[k]) for k in keys)
            else:
                raw = tuple(t for t in (texts[k] for k in keys) if t)
                pred = show.memo.get(raw)
                if pred is None:
                    pred = show.memo[raw] = "".join(norm(x) for x in show(raw))
            m = self.memo.get((i, pred))
            if m is None:
                m = self.memo[(i, pred)] = lev(ref, pred)
            e += m
        return e


RUBI_TEXTNORM_SHA256 = "3fa190177bb23244692cabe743c239a7721bf5ead4f082212a06e7d0077bb6fe"   # Rubi main bd130a8
RUBI_TEXTNORM_H_SHA256 = "e950ae0a3c3bea7fe8911909c7958e5f24e25eaea0fdca5032e233e65a6d5f21"


def is_japanese_char(c):
    """Rubi_for_vita core/utf.h vjo_is_japanese_char."""
    cp = ord(c)
    return 0x3040 <= cp <= 0x309F or 0x30A0 <= cp <= 0x30FF or 0x4E00 <= cp <= 0x9FAF


class RubiShow:
    """What Rubi shows of a job's lines (protocol amendment 1): the lines read (an empty one adds nothing) joined
    by '\\n' as finish_line joins them; the game worker's core/textnorm.c (vjo_text_normalize_as, flags 0: every
    rule, the border rule included, as non_japanese_filter = lines, the default, runs it) built from a Rubi
    checkout at RUBI_TEXTNORM_SHA256; then the Shell's core/textfilter.c vjo_filter_lines (non_japanese_filter =
    lines): a line without a kana or kanji goes. Returns the shown lines."""

    def __init__(self, rubi, build_dir):
        import ctypes as C
        import subprocess
        src, hdr = Path(rubi) / "core" / "textnorm.c", Path(rubi) / "core" / "textnorm.h"
        if sha256(src) != RUBI_TEXTNORM_SHA256 or sha256(hdr) != RUBI_TEXTNORM_H_SHA256:
            raise SystemExit(f"{src}: not Rubi main bd130a8's textnorm.c/.h")
        out = Path(build_dir) / "librubi_textnorm.so"
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            subprocess.run(["gcc", "-std=c99", "-O2", "-fPIC", "-shared", "-I", str(src.parent), "-o", str(out),
                            str(src)], check=True)
        self.lib = C.CDLL(str(out))
        self.lib.vjo_text_normalize_as.argtypes = [C.c_char_p, C.c_uint, C.c_void_p]
        self.lib.vjo_text_normalize_as.restype = C.c_uint
        self.C, self.memo = C, {}

    def textnorm(self, s):
        buf = self.C.create_string_buffer(s.encode("utf-8"))
        self.lib.vjo_text_normalize_as(buf, 0, None)
        return buf.value.decode("utf-8")

    def __call__(self, lines):
        text = self.textnorm("\n".join(x for x in lines if x))
        return [x for x in text.split("\n") if any(is_japanese_char(c) for c in x)]


def dev_units():
    """The development units (headgate_eval's rows, VOCR5's pooling), without EXCLUDE_DEV_VNS: (units, read keys in
    the dev cache's order as row_keys gives them)."""
    import finder_tighten as FT
    import headgate_eval as HE
    fixed = HE.fixed_rows()
    units = []
    for i, (coh, f, r) in enumerate(fixed):
        if r["vn"] in EXCLUDE_DEV_VNS:
            continue
        units.append(Unit(coh, f"{coh}|{f}|{i}", r["vn"], r["name"],
                          [(r["ref"], [(coh, f, i, "rv")]), (r["ref"], [(coh, f, i, "rvn")])]))
    wdc = {n: FT.load_cohort(HE.COHORTS, n) for n in WD}
    tv_shots = {c["shot"] for c in wdc["tv_dlg"]}
    c = HE.load_cache(DEV_CACHE, "N")
    keys = HE.row_keys(c["keys"], fixed)
    lines = {}
    for k in keys:
        if k[0] in WD:
            lines.setdefault(k[:3], []).append(k)
    for coh in WD:
        for mode in ("grad", "refine"):
            for ci, case in enumerate(wdc[coh]):
                if coh == "dn_dlg" and case["shot"] in tv_shots:
                    continue
                if case["vn"] in EXCLUDE_DEV_VNS:
                    continue
                ref = "".join(norm(x["text"]) for x in case["lines"])
                ks = sorted(lines.get((coh, mode, ci), []), key=lambda k: k[3])
                units.append(Unit("wd", f"{coh}|{mode}|{case['unit']}|{case['variant']}", case["vn"], False,
                                  [(ref, ks)]))
    return units, keys


# ---------------------------------------------------------------- reads and Rubi's line path

class Tier:
    """One tier's reads: per key the counter-edited top-K (what the LM decodes), the greedy text and the line
    confidence; decode(lm, opts) gives Rubi's text of every read."""

    def __init__(self, tier, reads, blob=None):
        from pyvocr import Vocr
        self.tier = tier
        self.rt = Vocr(str(blob or BLOBS[tier]), mode="w8i", max_width=1024)
        self.keys, self.src, self.greedy, self.conf = [], [], {}, {}
        n_edit = 0
        for k, o in reads:
            self.keys.append(k)
            self.greedy[k] = o["text"]
            self.conf[k] = line_conf(o)
            if not o["text"]:
                self.src.append(None)
                continue
            res, st = self.rt.redecode(o, *COUNTER)
            s = res if st["edits"] else o
            n_edit += bool(st["edits"])
            self.src.append((np.ascontiguousarray(s["topv"], np.float32), np.ascontiguousarray(s["topc"], np.uint16),
                             np.ascontiguousarray(s["blank"], np.float32)))
        self.counter_edits = n_edit

    def decode(self, lm, opts):
        """{key: Rubi's text} with the LM blob (path or bytes; None: greedy) at opts (beam, topk, alpha, beta, gate,
        flags, bracket)."""
        from pyvocr import VocrLM, post_counters
        if lm is None:                                   # the recognizer's own reading (eval/headgate_eval's greedy)
            return dict(self.greedy)
        dec = VocrLM(self.rt, lm, check_crc=False, **opts)
        out = {}
        lib = self.rt.lib
        for k, s in zip(self.keys, self.src):
            out[k] = "" if s is None else post_counters(dec.decode(*s), lib)[0]
        return out


def cascade(texts, conf, up_texts, thr):
    """The cascade's shown text of every read: the next tier's when the line's confidence is under thr."""
    if up_texts is None:
        return texts
    return {k: (up_texts[k] if conf[k] < thr else v) for k, v in texts.items()}


def normed(texts):
    return {k: norm(v) for k, v in texts.items()}


def dev_reads(tier):
    """[(row key, read)] of the head gate's development cache, arm today (the release blob's own reads)."""
    import headgate_eval as HE
    c = HE.load_cache(DEV_CACHE, tier)
    fixed = HE.fixed_rows()
    keys = HE.row_keys(c["keys"], fixed)
    if sha256(BLOBS[tier]) != c["meta"]["base_sha256"]:
        raise SystemExit(f"{tier}: the dev cache read another blob")
    return [(k, r["today"]) for k, r in zip(keys, c["res"])], c["meta"]


# ---------------------------------------------------------------- scoring

def unit_errors(units, texts, show=None):
    return np.array([u.errors(texts, show) for u in units], np.int64)


def rate(units, err, part, name=False, mask=None):
    m = np.array([u.part == part and u.name == name for u in units])
    if mask is not None:
        m &= mask
    c = sum(u.chars for u, x in zip(units, m) if x)
    return float(err[m].sum() / c) if c else float("nan")


def score(units, err, mask=None):
    """{tv, dn, wd, hc, hp: dialogue CER %, dev: 0.25 tv + 0.25 dn + 0.5 wd, obj: 0.5 hp + 0.5 dev}."""
    out = {p: 100 * rate(units, err, p, mask=mask) for p in ("tv", "dn", "wd", "hc", "hp")}
    out["dev"] = sum(w * out[p] for p, w in DEV_W.items())
    out["obj"] = 0.5 * out["hp"] + 0.5 * out["dev"] if not np.isnan(out["hp"]) else out["dev"]
    return out


def boot_part(units, ex, ey, part, n_boot=N_BOOT):
    """(point, replicates) of the dialogue CER difference y - x on one part, points, its clusters resampled."""
    idx = [i for i, u in enumerate(units) if u.part == part and not u.name]
    if not idx:
        return None, None, 0
    cl = np.array([units[i].cluster for i in idx])
    ch = np.array([units[i].chars for i in idx], float)
    x, y = ex[idx].astype(float), ey[idx].astype(float)
    ug, inv = np.unique(cl, return_inverse=True)
    gx, gy, gc = (np.bincount(inv, weights=v, minlength=len(ug)) for v in (x, y, ch))
    draw = np.random.default_rng(SEEDS[part]).integers(0, len(ug), size=(n_boot, len(ug)))
    point = (y.sum() - x.sum()) / ch.sum() * 100
    return float(point), (gy[draw].sum(1) - gx[draw].sum(1)) / gc[draw].sum(1) * 100, len(ug)


def ci(point, reps):
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"delta_points": round(float(point), 4), "ci95": [round(float(lo), 4), round(float(hi), 4)]}


def compare(units, ex, ey, n_boot=N_BOOT):
    """Paired y - x: per part, the dev score and the objective, each with its 95% cluster-bootstrap interval."""
    out, parts = {}, {}
    for p in ("tv", "dn", "wd", "hc", "hp"):
        pt, reps, n = boot_part(units, ex, ey, p, n_boot)
        if pt is None:
            continue
        parts[p] = (pt, reps)
        out[p] = {**ci(pt, reps), "clusters": n}
    if all(p in parts for p in DEV_W):
        pt = sum(w * parts[p][0] for p, w in DEV_W.items())
        reps = sum(w * parts[p][1] for p, w in DEV_W.items())
        out["dev"] = ci(pt, reps)
        if "hp" in parts:
            out["obj"] = ci(0.5 * parts["hp"][0] + 0.5 * pt, 0.5 * parts["hp"][1] + 0.5 * reps)
    return out


# ---------------------------------------------------------------- the grid

def grid_configs(grid):
    """Every combination of the grid's lists (dict of lists), in a fixed order (keys sorted, values as listed)."""
    keys = sorted(grid)
    out = [{}]
    for k in keys:
        out = [dict(c, **{k: v}) for c in out for v in grid[k]]
    return out


def lm_opts(cfg):
    return {"beam": cfg.get("beam", 8), "topk": cfg.get("topk", 8), "alpha": cfg["alpha"], "beta": cfg["beta"],
            "gate": cfg.get("gate", 8000), "flags": cfg.get("flags", 2), "bracket": cfg.get("bracket", 8000)}


_W = {}


def _init(tiers, hval_dir, rubi=None, both=False):
    _W["tiers"], _W["units"] = load_all(tiers, hval_dir)
    _W["show"] = RubiShow(rubi, PRIV / "build") if rubi else None
    _W["both"] = both


def path_errors(units, tiers, tx, thrs):
    """{(tier, thr | "none"): unit errors} on the amended path (Rubi's display rules) when a RubiShow is loaded, and
    with --both also on the original path under keys "raw:<tier>"."""
    out = {}
    paths = [("", _W["show"])] + ([("raw:", None)] if _W["show"] is not None and _W["both"] else [])
    for pre, show in paths:
        for t in tiers:
            out[(pre + t, "none")] = unit_errors(units, tx[t], show)
            if UP[t] in tiers:
                for thr in thrs:
                    out[(pre + t, thr)] = unit_errors(units, cascade(tx[t], tiers[t].conf, tx[UP[t]], thr), show)
    return out


def load_all(tiers, hval_dir):
    """({tier: Tier}, units) over dev and (when hval_dir is given) hval."""
    units, _ = dev_units()
    reads = {t: dev_reads(t)[0] for t in tiers}
    if hval_dir:
        hu, hr = hval_load(hval_dir, tiers)
        units += hu
        for t in tiers:
            reads[t] += hr[t]
    return {t: Tier(t, reads[t]) for t in tiers}, units


def _tune_one(job):
    lm, cfg, thrs = job
    tiers, units = _W["tiers"], _W["units"]
    opts = lm_opts(cfg)
    tx = {t: tiers[t].decode(lm, opts) for t in tiers}
    return cfg, path_errors(units, tiers, tx, thrs)


def cmd_tune(a):
    tiers = a.tiers.split(",")
    grid = json.loads(a.grid)
    thrs = [int(x) for x in a.cascade.split(",")]
    name, _, lm = a.lm.partition("=")
    lm_path = None if lm in ("", "none") else lm
    cfgs = grid_configs(grid) if lm_path else [{"alpha": 0, "beta": 0}]
    t0 = time.time()
    with mp.get_context("fork").Pool(a.procs, initializer=_init,
                                     initargs=(tiers, a.hval_cache, a.rubi, a.both)) as pool:
        res = pool.map(_tune_one, [(lm_path, c, thrs) for c in cfgs], chunksize=1)
    _init(tiers, a.hval_cache)
    units = _W["units"]
    errs = {}
    for cfg, out in res:
        for (t, thr), e in out.items():
            errs.setdefault(f"{t}|{thr}", []).append(e)
    meta = {"lm": name, "lm_path": lm_path, "lm_sha256": sha256(lm_path) if lm_path else None, "grid": grid,
            "configs": cfgs, "cascade": thrs, "tiers": tiers, "blobs": {t: sha256(BLOBS[t]) for t in tiers},
            "units": [[u.part, u.uid, u.cluster, u.name, u.chars, u.platform] for u in units],
            "hval_cache": str(a.hval_cache) if a.hval_cache else None, "seconds": round(time.time() - t0, 1),
            "display_rules": bool(a.rubi), "created": utc_now()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, meta=json.dumps(meta), **{k: np.stack(v).astype(np.int16) for k, v in errs.items()})
    best = {}
    for k, v in errs.items():
        sc = [score(units, e)["obj"] for e in v]
        i = int(np.argmin(sc))
        best[k] = (round(sc[i], 4), cfgs[i])
    print(json.dumps({"lm": name, "configs": len(cfgs), "seconds": meta["seconds"], "best": best}), flush=True)


# ---------------------------------------------------------------- held-out cross-entropy

def line_logprob(lm, s, bos):
    """Natural-log probability of the line s under the VOLM blob lm (vlm.LM), with </s>: per character and in all."""
    import vlm
    st = lm.initial(vlm.FLAG_BOS if bos else 0)
    tot = 0
    for ch in s:
        lp, st = lm.score(st, lm.token_of(ch))
        tot += lp
    lp, _ = lm.score(st, vlm.EOS)
    return (tot + lp) / vlm.Q


def cmd_screen(a):
    """Held-out nats per character (with </s>) of every LM on the hval lines and the dev references, BOS on/off."""
    import vlm
    base = {c for c in (ROOT / "data" / "vocab.txt").read_text(encoding="utf-8").split("\n") if c}
    hv = [s for s in Path(a.hval_text).read_text(encoding="utf-8").split("\n") if s and all(c in base for c in s)]
    units, _ = dev_units()
    dv = sorted({u.segs[0][0] for u in units if u.part in ("tv", "dn") and not u.name
                 and all(c in base for c in u.segs[0][0])})
    out = {"kind": "vocr6-lm-screen", "created": utc_now(), "hval_lines": len(hv), "dev_lines": len(dv),
           "hval_text_sha256": hashlib.sha256("\n".join(hv).encode()).hexdigest(),
           "dev_text_sha256": hashlib.sha256("\n".join(dv).encode()).hexdigest(), "lms": {}}
    for spec in a.lm:
        name, _, path = spec.partition("=")
        lm = vlm.LM(Path(path).read_bytes())
        row = {"sha256": sha256(path), "bytes": Path(path).stat().st_size}
        for tag, lines in (("hval", hv), ("dev", dv)):
            n = sum(len(s) + 1 for s in lines)
            for bos in (True, False):
                row[f"{tag}_{'bos' if bos else 'nobos'}"] = round(-sum(line_logprob(lm, s, bos) for s in lines) / n,
                                                                  4)
        out["lms"][name] = row
        print(name, json.dumps(row), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- selection

PARTS = ("tv", "dn", "wd", "hc", "hp")


class Scorer:
    """Vectorised scores of error matrices (configs x units) over the units of a tune file."""

    def __init__(self, units):
        self.part = np.array([u[0] for u in units])
        self.cluster = np.array([u[2] for u in units])
        self.name = np.array([bool(u[3]) for u in units])
        self.chars = np.array([u[4] for u in units], float)
        self.plat = np.array([u[5] or "" for u in units])
        self.fold = np.array([fold_of(c) for c in self.cluster])

    def mask(self, part, fold=None, name=False, plat=None):
        m = (self.part == part) & (self.name == name)
        if fold is not None:
            m &= self.fold == fold
        if plat is not None:
            m &= self.plat == plat
        return m

    def cer(self, E, part, **kw):
        m = self.mask(part, **kw)
        c = self.chars[m].sum()
        return np.asarray(E)[..., m].sum(-1) / c * 100 if c else np.full(np.shape(E)[:-1], np.nan)

    def scores(self, E, fold=None):
        out = {p: self.cer(E, p, fold=fold) for p in PARTS}
        out["dev"] = sum(w * out[p] for p, w in DEV_W.items())
        out["obj"] = 0.5 * out["hp"] + 0.5 * out["dev"] if not np.all(np.isnan(out["hp"])) else out["dev"]
        return out


def load_tune(path, keys=None):
    """(meta, {key: errors configs x units}) of a tune file; keys: only those (the selection's)."""
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    return meta, {k: z[k].astype(np.int64) for k in z.files if k != "meta" and (keys is None or k in keys)}


def rounded(d, n=4):
    return {k: (None if v is None or np.isnan(v) else round(float(v), n)) for k, v in d.items()}


def cmd_select(a):
    """The protocol's cross-validated selection and rule for one LM tier (module docstring)."""
    proto = json.loads(Path(a.protocol).read_text(encoding="utf-8"))
    pre = "raw:" if a.path == "raw" else ""
    inc_meta, inc = load_tune(a.incumbent)
    greedy_meta, greedy = load_tune(a.greedy)
    inc = {k[len(pre):]: v for k, v in inc.items() if k.startswith(pre) and (pre or not k.startswith("raw:"))}
    greedy = {k[len(pre):]: v for k, v in greedy.items() if k.startswith(pre) and (pre or not k.startswith("raw:"))}
    units = inc_meta["units"]
    sc = Scorer(units)
    U = [Unit(u[0], u[1], u[2], u[3], [], u[5]) for u in units]
    for u, x in zip(U, units):
        u.chars = x[4]
    thrs = [int(t) for t in proto["grid"]["cascade_conf_selectable"]]
    need = {f"{pre}{m}|{t}" for m in ("N", "S") for t in thrs} | {f"{pre}L|none"}
    lms = []
    for p in a.tune:
        meta, errs = load_tune(p, need)
        errs = {k[len(pre):]: v for k, v in errs.items()}
        if meta["units"] != units:
            raise SystemExit(f"{p}: other units than the incumbent's")
        lms.append((meta, errs))
    # options in (recipe order, grid order, threshold order)
    opts = [(i, j, t) for i, (meta, _) in enumerate(lms) for j in range(len(meta["configs"])) for t in thrs]

    def err(model, o):
        meta, errs = lms[o[0]]
        key = f"{model}|{o[2]}" if model != "L" else "L|none"
        return errs[key][o[1]]

    choice = {}
    for f in (0, 1, None):
        vals = {}
        for i, (meta, errs) in enumerate(lms):
            for t in thrs:
                for j, x in enumerate(sc.scores(errs[f"N|{t}"], fold=f)["obj"]):
                    vals[(i, j, t)] = x
        best = None
        for o in opts:                                   # the first minimum in (recipe, grid, threshold) order
            if best is None or vals[o] < best[0] - 1e-12:
                best = (vals[o], o)
        choice[f] = best[1]

    def describe(o):
        meta = lms[o[0]][0]
        return {"lm": meta["lm"], "lm_sha256": meta["lm_sha256"], "config": meta["configs"][o[1]], "cascade_conf": o[2]}

    inc_key = {"N": f"N|{proto['incumbent']['cascade_conf']}", "S": f"S|{proto['incumbent']['cascade_conf']}",
               "L": "L|none"}
    out = {"kind": "vocr6-lm-select", "schema": 1, "created": utc_now(), "tier": a.tier, "path": a.path,
           "protocol_sha256": sha256(a.protocol), "incumbent": {"lm": inc_meta["lm"], "sha256": inc_meta["lm_sha256"],
                                                                "config": inc_meta["configs"][0]},
           "units": {p: int(sc.mask(p).sum()) for p in PARTS}, "clusters": {
               p: int(len(set(sc.cluster[sc.mask(p)]))) for p in PARTS},
           "recipes": [m["lm"] for m, _ in lms], "options": len(opts),
           "chosen": {"fold0": describe(choice[0]), "fold1": describe(choice[1]), "in_sample": describe(choice[None])},
           "models": {}}
    cv_units = {}
    for model in ("N", "S", "L"):
        e_inc = inc[inc_key[model]][0]
        e_cv = np.where(sc.fold == 0, err(model, choice[1]), err(model, choice[0]))   # fold 0 scored by fold 1's
        e_in = err(model, choice[None])
        cv_units[model] = e_cv
        g = greedy[f"{model}|none"][0]
        rep = {"greedy": rounded(sc.scores(g)), "incumbent": rounded(sc.scores(e_inc)),
               "cv": rounded(sc.scores(e_cv)), "in_sample": rounded(sc.scores(e_in)),
               "cv_minus_incumbent": compare(U, e_inc, e_cv), "in_sample_minus_incumbent": compare(U, e_inc, e_in),
               "incumbent_minus_greedy": compare(U, g, e_inc)}
        rep["by_platform"] = {pl: {p: {"incumbent": round(float(sc.cer(e_inc, p, plat=pl)), 4),
                                       "cv": round(float(sc.cer(e_cv, p, plat=pl)), 4),
                                       "in_sample": round(float(sc.cer(e_in, p, plat=pl)), 4)}
                                   for p in ("hc", "hp")} for pl in ("vita", "psp")}
        rep["names"] = {p: {"incumbent": round(float(sc.cer(e_inc, p, name=True)), 4),
                            "in_sample": round(float(sc.cer(e_in, p, name=True)), 4)} for p in ("tv", "dn")}
        out["models"][model] = rep
    hp_titles = sorted(set(sc.cluster[sc.mask("hp")]))
    out["hp_by_title"] = {}
    for tid in hp_titles:
        m = sc.mask("hp") & (sc.cluster == tid)
        c = sc.chars[m].sum()
        e_inc = inc[inc_key["N"]][0]
        out["hp_by_title"][tid] = {"platform": str(sc.plat[m][0]), "chars": int(c),
                                   "incumbent": round(float(e_inc[m].sum() / c * 100), 3),
                                   "cv": round(float(cv_units["N"][m].sum() / c * 100), 3)}
    # report only: hp without the titles whose incumbent page-path CER is above 50% (a finder that reads a box's
    # decoration as lines of junk: their insertions follow the decoder's per-character bonus, not the text)
    e_inc_n = inc[inc_key["N"]][0]
    junk = [t for t, v in out["hp_by_title"].items() if v["incumbent"] > 50]
    keep = ~np.isin(sc.cluster, junk)
    Uk = [u for u, k in zip(U, keep) if k]
    out["hp_without_junk_titles"] = {"titles": junk, "cv_minus_incumbent": compare(
        Uk, e_inc_n[keep], cv_units["N"][keep]).get("hp")}
    # the cascade thresholds beyond the selectable ones, at the in-sample LM and setting (report only)
    i, j, _ = choice[None]
    _, all_thr = load_tune(a.tune[i], {f"{pre}N|{t}" for t in lms[i][0]["cascade"]})
    out["cascade_report"] = {str(t): rounded(sc.scores(all_thr[f"{pre}N|{t}"][j])) for t in lms[i][0]["cascade"]}
    r = {m: out["models"][m]["cv_minus_incumbent"] for m in ("N", "S", "L")}

    def worse(c):
        return c["ci95"][0] > 0

    rule = {"a_hp_better": bool(r["N"]["hp"]["ci95"][1] < 0),
            "b_dev_not_worse": bool(not worse(r["N"]["dev"])),
            "c_S_L_not_worse": bool(all(not worse(r[m][p]) for m in ("S", "L") for p in ("hp", "dev"))),
            "d_bytes": None, "d_workspace": None}
    meta_in = lms[i][0]
    if meta_in["lm_path"]:
        nb = Path(meta_in["lm_path"]).stat().st_size
        rule["d_bytes"] = bool(nb <= proto["incumbent"]["lms"][a.tier]["bytes"])
        rule["lm_bytes"] = nb
        ws = workspace_bytes(meta_in["lm_path"], lm_opts(meta_in["configs"][j]))
        rule["workspace_bytes"] = ws
        rule["d_workspace"] = bool(max(ws.values()) <= 49152)
    rule["adopted"] = bool(all(v for k, v in rule.items() if k[0] in "abcd" and k[1] == "_"))
    out["rule"] = rule
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    np.savez_compressed(a.out + ".cv.npz", units=json.dumps(units), **{f"cv_{m}": cv_units[m] for m in cv_units},
                        **{f"inc_{m}": inc[inc_key[m]][0] for m in inc_key},
                        **{f"in_{m}": err(m, choice[None]) for m in ("N", "S", "L")})
    print(json.dumps({"tier": a.tier, "chosen": out["chosen"]["in_sample"], "rule": rule,
                      "N": {k: out["models"]["N"][k] for k in ("incumbent", "cv", "in_sample")}}, ensure_ascii=False))


def cmd_families(a):
    """Protocol amendment 2: family B (game text) against family A (none), both cross-validated (select's .cv.npz)."""
    za, zb = np.load(a.a + ".cv.npz"), np.load(a.b + ".cv.npz")
    units = json.loads(str(za["units"]))
    if json.loads(str(zb["units"])) != units:
        raise SystemExit("the two selections scored other units")
    U = [Unit(u[0], u[1], u[2], u[3], [], u[5]) for u in units]
    for u, x in zip(U, units):
        u.chars = x[4]
    cmp = {m: compare(U, za[f"cv_{m}"], zb[f"cv_{m}"]) for m in ("N", "S", "L")}
    cin = {m: compare(U, za[f"in_{m}"], zb[f"in_{m}"]) for m in ("N", "S", "L")}
    ra, rb = (json.loads(Path(x).read_text(encoding="utf-8")) for x in (a.a, a.b))

    def worse(c):
        return c["ci95"][0] > 0
    rule = {"B_passes_rule": bool(rb["rule"]["adopted"]), "A_passes_rule": bool(ra["rule"]["adopted"]),
            "hp_better": bool(cmp["N"]["hp"]["ci95"][1] < 0), "objective_better": bool(cmp["N"]["obj"]["ci95"][1] < 0),
            "dev_not_worse": bool(not worse(cmp["N"]["dev"])),
            "S_L_not_worse": bool(all(not worse(cmp[m][p]) for m in ("S", "L") for p in ("hp", "dev")))}
    rule["B_instead_of_A"] = bool(rule["B_passes_rule"] and rule["hp_better"] and rule["objective_better"]
                                  and rule["dev_not_worse"] and rule["S_L_not_worse"])
    rule["ships"] = "B" if rule["B_instead_of_A"] else "A" if rule["A_passes_rule"] else "incumbent"
    out = {"kind": "vocr6-lm-families", "created": utc_now(), "tier": ra["tier"], "path": ra.get("path"),
           "A": {"select": Path(a.a).name, "chosen": ra["chosen"]["in_sample"]},
           "B": {"select": Path(a.b).name, "chosen": rb["chosen"]["in_sample"]},
           "B_minus_A_cv": cmp, "B_minus_A_in_sample": cin, "rule": rule}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({"tier": out["tier"], "rule": rule, "N_obj": cmp["N"]["obj"], "N_hp": cmp["N"]["hp"],
                      "N_dev": cmp["N"]["dev"]}))


def _cv_units(path):
    """(the units as stored, as Unit objects, the .cv.npz arrays) of a select output's cross-validated errors."""
    z = np.load(path)
    units = json.loads(str(z["units"]))
    U = [Unit(u[0], u[1], u[2], u[3], [], u[5]) for u in units]
    for u, x in zip(U, units):
        u.chars = x[4]
    return units, U, z


def _sub_compare(U, ex, ey, drop):
    keep = np.flatnonzero(~drop)
    return compare([U[i] for i in keep], ex[keep], ey[keep])


def cmd_checks(a):
    """The audit's checks (audit-vocr6-lm F1-F3, F5, N1, N6), recomputed from the private outputs; counts, rates and
    ids only. f2: the 4 MB game-text LM against the 1 MB one at the 4 MB tier, and the 1 MB one against the 4 MB
    incumbent (the selections' cross-validated errors, same units). f3: hval units holding a line that is verbatim
    (eval/metrics.norm) a hook_train line, and the shipped choices against the incumbent without them. f5: speaker-name
    CER by line flags at the shipped alpha and beta (tune4: the screening game-text recipe and ctclm.2's rebuild), and
    the share of the shipped blobs' bigrams and trigrams that the same recipe without game text does not keep. n1: the
    #129 dev run's units of the two dropped VNs. n6: the dev score without v18643, the sequel of a hooked title.
    seen: the hooked titles flagged seen, with their exported lines and hval units."""
    from memorization import as_chars, kept_ngrams
    from vlm import LM
    sel, P = Path(a.select_dir), Path(a.priv)
    files = {"1000k": sel / "1000k_B_shown.json.cv.npz", "4000k": sel / "4000k_B3_shown.json.cv.npz"}
    out = {"kind": "vocr6-lm-checks", "schema": 1, "created": utc_now(), "audit": "audit-vocr6-lm (F1-F3, F5, N1, N6)"}
    keys = ("hc", "hp", "dev", "obj")
    u1, U, z1 = _cv_units(files["1000k"])
    u4, _, z4 = _cv_units(files["4000k"])
    if u1 != u4:
        raise SystemExit("the 1 MB and 4 MB selections scored other units")
    out["f2_4mb_tier"] = {m: {"4mb_game_text_minus_1mb_game_text": {k: v for k, v in
                                                                   compare(U, z1[f"cv_{m}"], z4[f"cv_{m}"]).items()
                                                                   if k in keys},
                              "1mb_game_text_minus_4mb_incumbent": {k: v for k, v in
                                                                    compare(U, z4[f"inc_{m}"], z1[f"cv_{m}"]).items()
                                                                    if k in keys}}
                          for m in ("N", "S", "L")}
    # f3: verbatim train lines in hval
    train = {norm(x) for x in (P / "text" / "hook_train.txt").read_text(encoding="utf-8").splitlines()} - {""}
    raw = (P / "text" / "hook_train.txt").read_text(encoding="utf-8").splitlines()
    held = [x for x in (P / "text" / "hval_lines.txt").read_text(encoding="utf-8").splitlines() if x]
    raw_set = set(raw)
    held_raw_seen = sum(x in raw_set for x in held)
    regs, ch_all, ch_seen = [], 0, 0
    for f in sorted(Path(a.pagepool).glob("*.jsonl")):          # cmd_hval_units' region order
        for line in f.read_text(encoding="utf-8").splitlines():
            pg = json.loads(line)
            if pg["split"] != "hval" or not pg["regions"]:
                continue
            text = {ln["id"]: norm(ln["text"]) for ln in pg["lines"]}
            for r in pg["regions"]:
                t = [text[i] for i in r["lines"]]
                hit = [x in train for x in t]
                ch_all += sum(len(x) for x in t)
                ch_seen += sum(len(x) for x, h in zip(t, hit) if h)
                regs.append((pg["title_id"], str(pg["page"]), r["variant"], any(hit)))
    hv = pickle.loads((Path(a.hval_cache) / "units.pkl").read_bytes())
    hc_seen = {x[1] for x in hv if x[0] == "hc" and x[4][0][0] in train}
    hp_idx = [i for i, u in enumerate(u1) if u[0] == "hp"]
    if len(hp_idx) != 2 * len(regs):
        raise SystemExit("hp units are not the page pool's regions x 2 finder modes")
    seen = np.zeros(len(u1), bool)
    for j, i in enumerate(hp_idx):
        tid, page, variant, hit = regs[j % len(regs)]
        if u1[i][2] != tid or u1[i][1].split("|")[1:3] != [page, variant]:
            raise SystemExit("hp units out of the page pool's order")
        seen[i] = hit
    for i, u in enumerate(u1):
        if u[0] == "hc":
            seen[i] = u[1] in hc_seen
    hc = np.array([u[0] == "hc" for u in u1])
    j854 = np.array([u[2] == "PCSG00854" for u in u1])
    f3 = {"hval_lines": len(held), "hval_lines_verbatim_in_hook_train": sum(norm(x) in train for x in held),
          "hval_lines_byte_identical_to_a_hook_train_line": held_raw_seen,
          "hp_regions": len(regs), "hp_regions_seen": int(sum(r[3] for r in regs)),
          "hp_units": len(hp_idx), "hp_units_seen": int(seen[hp_idx].sum()),
          "hp_seen_line_char_share": round(ch_seen / ch_all, 4),
          "hc_units": int(hc.sum()), "hc_units_seen": int(seen[hc].sum()), "by_tier": {}}
    for t, f in files.items():
        _, UU, z = _cv_units(f)
        f3["by_tier"][t] = {
            "without_seen": {k: v for k, v in _sub_compare(UU, z["inc_N"], z["cv_N"], seen).items() if k in keys},
            "without_seen_and_PCSG00854": {k: v for k, v in _sub_compare(UU, z["inc_N"], z["cv_N"], seen | j854).items()
                                           if k in keys}}
    out["f3_seen_hval"] = f3
    # f5: names by flags
    names = {}
    for nm in ("h64_1000k", "repro_1000k", "inc_1000k"):
        meta, e = load_tune(Path(a.tune4) / f"{nm}.npz", {"N|700"})
        sc = Scorer(meta["units"])
        rows = {}
        for j, c in enumerate(meta["configs"]):
            if nm == "inc_1000k" or (c["alpha"] == 600 and c["beta"] == 2500):
                rows[f"a{c['alpha']} b{c['beta']} f{c['flags']}"] = {
                    p: round(float(sc.cer(e["N|700"][j], p, name=True)), 4) for p in ("dn", "tv")}
        names[nm] = rows
    out["f5_names_by_flags"] = {"key": "N|700", "lms": names}
    grams = {}
    for blob, base in ((Path(a.final_dir) / "vocr_lm6_o3_1000k.vlm", Path(a.lm_dir) / "v6b_1000k.vlm"),
                       (Path(a.final_dir) / "vocr_lm6_o3_4000k.vlm", Path(a.lm_dir) / "v6brl_4000k.vlm")):
        g = {}
        for q in (blob, base):
            lm = LM(q.read_bytes())
            k = kept_ngrams(lm)
            g[q] = {o: {as_chars(lm, x) for x in k[o]} for o in (2, 3)}
        n = {o: len(g[blob][o]) for o in (2, 3)}
        d = {o: len(g[blob][o] - g[base][o]) for o in (2, 3)}
        th = {q.name: json.loads(q.with_suffix(".json").read_text(encoding="utf-8")).get("chosen_thresholds")
              for q in (blob, base)}
        grams[blob.name] = {"baseline": base.name, "bigrams": n[2], "trigrams": n[3],
                            "not_kept_by_baseline": d, "share": round((d[2] + d[3]) / (n[2] + n[3]), 4),
                            "trigram_share": round(d[3] / n[3], 4), "count_thresholds": th}
    out["f5_ngrams_vs_no_game_text"] = grams
    # n1: the #129 dev run's units of the dropped VNs
    n1 = {}
    for arm in ("old", "new"):
        rows = [json.loads(x) for x in (Path(a.pagepath) / f"{arm}_dev.rows.jsonl").read_text().splitlines()]
        drop = [r for r in rows if r["cluster"] in EXCLUDE_DEV_VNS]
        n1[arm] = {"units": len(drop), "errors": sum(r["errors"] for r in drop),
                   "chars": sum(r["chars"] for r in drop), "all_chars": sum(r["chars"] for r in rows),
                   "by_vn": {v: sum(r["cluster"] == v for r in drop) for v in sorted(EXCLUDE_DEV_VNS)}}
    out["n1_pagepath_dropped_vns"] = n1
    n6 = {}
    for t, f in files.items():
        uu, UU, z = _cv_units(f)
        m = np.array([u[2] == "v18643" for u in uu])
        n6[t] = {"v18643_units": int(m.sum()), **{k: v for k, v in _sub_compare(UU, z["inc_N"], z["cv_N"], m).items()
                                                  if k in ("dev", "wd", "obj")}}
    out["n6_without_v18643"] = n6
    seen_t = {}
    for name in ("hook_train", "hook_final"):
        j = json.loads((P / "text" / f"{name}.json").read_text(encoding="utf-8"))
        for tid in SEEN_TITLES:
            c = j["titles"][tid]["counts"]
            seen_t.setdefault(tid, {})[name] = {"lines": c["kept"], "chars": c["chars"]}
    for tid in SEEN_TITLES:
        seen_t[tid]["hval_units"] = {p: sum(1 for x in hv if x[0] == p and x[2] == tid) for p in ("hc", "hp")}
        seen_t[tid]["why"] = SEEN_TITLES[tid]
    out["seen_titles"] = seen_t
    import finder_tighten as FT
    FT.assert_public(out)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    ks = ("hp_units_seen", "hc_units_seen", "hval_lines_verbatim_in_hook_train")
    print(json.dumps({"f2_N": out["f2_4mb_tier"]["N"]["4mb_game_text_minus_1mb_game_text"]["obj"],
                      "f3": {k: f3[k] for k in ks}}), flush=True)


def cmd_summary(a):
    """The committed result (eval/vocr6_lm_dev_<date>.json): every selection, the family comparisons, the screens,
    the dictionary variant, the memorisation report and the shipped blobs. Counts, rates, hashes, ids only."""
    sel = Path(a.select_dir)
    out = {"kind": "vocr6-lm-dev", "schema": 1, "created": utc_now(), "issue": 127,
           "protocol_sha256": sha256(a.protocol), "tiers": {}, "original_path": {}}
    for f in sorted(sel.glob("*.json")):
        tier, rest = f.stem.split("_", 1)
        doc = json.loads(f.read_text(encoding="utf-8"))
        part, path = rest.rsplit("_", 1)
        (out["tiers"] if path == "shown" else out["original_path"]).setdefault(tier, {})[part] = doc
    for key, files in (("screen", a.screen), ("memorisation", a.memo), ("lexicon", a.lexicon),
                       ("final_check", a.final_check), ("sensitivity", a.sensitivity),
                       ("pagepath_check", a.pagepath_check)):
        out[key] = [json.loads(Path(x).read_text(encoding="utf-8")) for x in files]
    out["shipped"] = json.loads(Path(a.shipped).read_text(encoding="utf-8")) if a.shipped else None
    if a.checks:
        out["checks"] = json.loads(Path(a.checks).read_text(encoding="utf-8"))
    import finder_tighten as FT
    FT.assert_public(out)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"{a.out}: {Path(a.out).stat().st_size:,} B")


def _pp(x):
    return "–" if x is None else f"{x:.3f}"


def _ci(c):
    return f"{c['delta_points']:+.3f} ({c['ci95'][0]:+.3f} to {c['ci95'][1]:+.3f})"


def cmd_tables(a):
    """docs/vocr6-lm.md's tables from the summary JSON (between <!-- table:NAME --> and <!-- /table:NAME -->)."""
    import re
    j = json.loads(Path(a.json).read_text(encoding="utf-8"))
    T = j["tiers"]
    tabs = {}
    rows = ["| LM tier | ships | recipe (game-text weight, re-cut) | alpha | beta | flags | cascade |",
            "|---|---|---|---:|---:|---|---:|"]
    for t in ("500k", "1000k", "2000k", "4000k"):
        s = (j.get("shipped") or {}).get(t) or {}
        if not s or s.get("family") == "incumbent":
            rows.append(f"| {t} | ctclm.2 (unchanged) | – | 500 | {RUBI_BETA[t]} | EOS | 700 |")
            continue
        c = s["config"]
        fl = " \\| ".join(n for b, n in ((1, "BOS"), (2, "EOS")) if c["flags"] & b) or "0"
        rows.append(f"| {t} | `{s['file']}` | {s['recipe']} | {c['alpha']} | {c['beta']} | {fl} | "
                    f"{s['cascade_conf']} |")
    tabs["shipped"] = rows
    rows = ["| LM tier | part | greedy | incumbent | A in-sample | A cross-validated | B in-sample | "
            "B cross-validated |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for t in ("500k", "1000k", "2000k", "4000k"):
        bkey = "B3" if "B3" in T[t] else "B"
        A, B = T[t]["A"]["models"]["N"], T[t][bkey]["models"]["N"]
        for p, name in (("hp", "hval page path"), ("hc", "hval clean lines"), ("dev", "dev score"),
                        ("wd", "dev whole dialogue"), ("obj", "objective")):
            rows.append(f"| {t} | {name} | {_pp(A['greedy'][p])} | {_pp(A['incumbent'][p])} | "
                        f"{_pp(A['in_sample'][p])} | {_pp(A['cv'][p])} | {_pp(B['in_sample'][p])} | "
                        f"{_pp(B['cv'][p])} |")
    tabs["results"] = rows
    rows = ["| LM tier | family | chosen (in-sample) | hp, CV − incumbent | dev score, CV − incumbent | (a) | (b) | "
            "(c) | (d) | adopted |", "|---|---|---|---|---|---|---|---|---|---|"]
    for t in ("500k", "1000k", "2000k", "4000k"):
        for fam in ("A", "B", "B3"):
            if fam not in T[t]:
                continue
            r = T[t][fam]
            c = r["chosen"]["in_sample"]
            cm = r["models"]["N"]["cv_minus_incumbent"]
            ru = r["rule"]
            yn = {True: "yes", False: "no", None: "–"}
            label = {"A": "A (no game text)", "B": "B (game text)", "B3": "B, order 3"}[fam]
            rows.append(f"| {t} | {label} | `{c['lm']}` a{c['config']['alpha']} b{c['config']['beta']} "
                        f"f{c['config']['flags']} c{c['cascade_conf']} | {_ci(cm['hp'])} | {_ci(cm['dev'])} | "
                        f"{yn[ru['a_hp_better']]} | {yn[ru['b_dev_not_worse']]} | {yn[ru['c_S_L_not_worse']]} | "
                        f"{yn[ru['d_bytes'] and ru['d_workspace']]} | {yn[ru['adopted']]} |")
    tabs["rule"] = rows
    rows = ["| LM tier | comparison | objective | hval page path | hval clean lines | dev score | ships |",
            "|---|---|---|---|---|---|---|"]
    for t in ("500k", "1000k", "2000k", "4000k"):
        for key, name in (("AB", "B − A"), ("B3vsA", "B (order 3) − A"), ("BvsB3", "B (order 4) − B (order 3)")):
            if key not in T[t]:
                continue
            c = T[t][key]["B_minus_A_cv"]["N"]
            ships = {"AB": {"A": "A", "B": "B"}, "B3vsA": {"A": "A", "B": "B, order 3"},
                     "BvsB3": {"A": "B, order 3", "B": "B, order 4"}}[key].get(T[t][key]["rule"]["ships"],
                                                                         T[t][key]["rule"]["ships"])
            rows.append(f"| {t} | {name} | {_ci(c['obj'])} | {_ci(c['hp'])} | {_ci(c['hc'])} | {_ci(c['dev'])} | "
                        f"{ships} |")
    tabs["families"] = rows
    rows = ["| LM tier | model | hval page path | hval clean lines | dev score | objective |",
            "|---|---|---|---|---|---|"]
    for t in ("1000k", "4000k"):
        bkey = "B3" if "B3" in T[t] else "B"
        for m, name in (("S", "VOCR5-S + cascade to N"), ("N", "VOCR5-N + cascade to L"), ("L", "VOCR5-L")):
            c = T[t][bkey]["models"][m]["cv_minus_incumbent"]
            rows.append(f"| {t} | {name} | {_ci(c['hp'])} | {_ci(c['hc'])} | {_ci(c['dev'])} | "
                        f"{_ci(c['obj'])} |")
    tabs["models"] = rows
    rows = ["| LM tier | platform | hc incumbent | hc B (CV) | hp incumbent | hp B (CV) |",
            "|---|---|---:|---:|---:|---:|"]
    for t in ("1000k", "4000k"):
        bkey = "B3" if "B3" in T[t] else "B"
        for pl in ("vita", "psp"):
            x = T[t][bkey]["models"]["N"]["by_platform"][pl]
            rows.append(f"| {t} | {pl} | {x['hc']['incumbent']:.3f} | {x['hc']['cv']:.3f} | "
                        f"{x['hp']['incumbent']:.3f} | {x['hp']['cv']:.3f} |")
    tabs["platform"] = rows
    rows = ["| LM tier | path | A chosen | B chosen | ships |", "|---|---|---|---|---|"]
    for t in ("500k", "1000k", "2000k", "4000k"):
        o = j["original_path"].get(t)
        if not o:
            continue
        ca, cb = o["A"]["chosen"]["in_sample"], o["B"]["chosen"]["in_sample"]
        rows.append(f"| {t} | original | `{ca['lm']}` a{ca['config']['alpha']} b{ca['config']['beta']} "
                    f"f{ca['config']['flags']} | `{cb['lm']}` a{cb['config']['alpha']} b{cb['config']['beta']} "
                    f"f{cb['config']['flags']} | {o['AB']['rule']['ships']} |")
    tabs["original"] = rows
    if j.get("screen"):
        rows = ["| LM | bytes | hval lines, BOS | hval, no BOS | dev references, BOS | dev, no BOS |",
                "|---|---:|---:|---:|---:|---:|"]
        for s in j["screen"]:
            for name, r in s["lms"].items():
                rows.append(f"| `{name}` | {r['bytes']:,} | {r['hval_bos']:.3f} | {r['hval_nobos']:.3f} | "
                            f"{r['dev_bos']:.3f} | {r['dev_nobos']:.3f} |")
        tabs["screen"] = rows
    doc = Path(a.doc).read_text(encoding="utf-8")
    for k, rows in tabs.items():
        pat = re.compile(rf"(<!-- table:{k} -->\n).*?(<!-- /table:{k} -->)", re.S)
        doc, n = pat.subn(lambda m, rows=rows: m.group(1) + "\n".join(rows) + "\n" + m.group(2), doc)
        if not n:
            print(f"no marker for table {k}", file=sys.stderr)
    Path(a.doc).write_text(doc, encoding="utf-8")


def workspace_bytes(lm, opts):
    """vocr_lm_workspace_bytes of each VOCR5 tier with this LM and setting (Rubi: VJO_VOCR_LM_WS_BYTES 49,152 B)."""
    from pyvocr import Vocr, VocrLM
    return {t: VocrLM(Vocr(str(BLOBS[t]), mode="w8i", max_width=1024), lm, **opts).workspace_bytes for t in BLOBS}


# ---------------------------------------------------------------- the dictionary-aware prototype

def decode_py(tier, lm, opts, lexicon, bonus):
    """Rubi's line path with lm/vlm.py's Python twin of the decoder and the lexicon bonus: ({key: text}, stats)."""
    import vlm
    from pyvocr import post_counters
    rt = tier.rt
    texts = [rt.class_text(c) for c in range(rt.n_classes)]
    cmap = vlm.class_map(lm, texts)
    roles = [vlm.bracket_role(t) for t in texts]
    out, st = {}, {"lm_calls": 0}
    lexicon.steps = 0
    for k, s in zip(tier.keys, tier.src):
        if s is None:
            out[k] = ""
            continue
        cl = vlm.decode(lm, cmap, s[0], s[1], s[2], roles=roles, stats=st, lexicon=lexicon, lex_bonus=bonus, **opts)
        out[k] = post_counters("".join(texts[c] for c in cl), rt.lib)[0]
    return out, {"lm_calls": st["lm_calls"], "lex_steps": lexicon.steps,
                 "lines": sum(s is not None for s in tier.src)}


def _lex_one(job):
    import lexicon as LX
    import vlm
    lm_path, cfg, lex_path, lx, thrs = job
    tiers, units = _W["tiers"], _W["units"]
    lm = vlm.LM(Path(lm_path).read_bytes())
    words = [tuple(lm.token_of(ch) for ch in w) for w, _ in LX.load(lex_path, lx["max_rank"])]
    lexi = vlm.Lexicon([w for w in words if vlm.UNK not in w], lx.get("max_len", 8))
    opts = lm_opts(cfg)
    tx, stats = {}, {}
    for t in tiers:
        tx[t], stats[t] = decode_py(tiers[t], lm, opts, lexi, lx["bonus"])
    return lx, path_errors(units, tiers, tx, thrs), stats, len(lexi.words)


def cmd_lex_tune(a):
    """The lexicon bonus over a small grid, on top of one LM and setting (the Python twin; slow but exact)."""
    tiers = a.tiers.split(",")
    grid = json.loads(a.grid)
    cfg = json.loads(a.config)
    thrs = [int(x) for x in a.cascade.split(",")]
    name, _, lm = a.lm.partition("=")
    lxs = grid_configs(grid)
    t0 = time.time()
    with mp.get_context("fork").Pool(a.procs, initializer=_init,
                                     initargs=(tiers, a.hval_cache, a.rubi, a.both)) as pool:
        res = pool.map(_lex_one, [(lm, cfg, a.lexicon, lx, thrs) for lx in lxs], chunksize=1)
    _init(tiers, a.hval_cache)
    units = _W["units"]
    errs, stats = {}, []
    for lx, out, st, nw in res:
        for (t, thr), e in out.items():
            errs.setdefault(f"{t}|{thr}", []).append(e)
        stats.append({"lexicon": lx, "words": nw, "stats": st})
    meta = {"lm": f"{name}+lex", "lm_path": lm, "lm_sha256": sha256(lm), "base_config": cfg, "lexicon": a.lexicon,
            "lexicon_sha256": sha256(a.lexicon), "grid": grid, "configs": [dict(cfg, lex=lx) for lx in lxs],
            "cascade": thrs, "tiers": tiers, "blobs": {t: sha256(BLOBS[t]) for t in tiers},
            "units": [[u.part, u.uid, u.cluster, u.name, u.chars, u.platform] for u in units],
            "hval_cache": str(a.hval_cache) if a.hval_cache else None, "stats": stats,
            "display_rules": bool(a.rubi), "seconds": round(time.time() - t0, 1), "created": utc_now()}
    np.savez_compressed(a.out, meta=json.dumps(meta), **{k: np.stack(v).astype(np.int16) for k, v in errs.items()})
    sc = Scorer(meta["units"])
    for k in sorted(errs):
        if k.startswith("N|"):
            print(k, [round(float(x), 4) for x in sc.scores(np.stack(errs[k]))["obj"]], flush=True)


# ---------------------------------------------------------------- hval (the hooked screening pages)

def page_rgb(pages_dir, tid, page, sha_want):
    import cv2
    p = Path(pages_dir) / tid / "pages" / f"{page}.png"
    data = p.read_bytes()
    if hashlib.sha256(data).hexdigest() != sha_want:
        raise SystemExit(f"{tid}:{page}: the page image differs from its sha256")
    return np.ascontiguousarray(cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)[..., ::-1])


def read_pool(path):
    """[(meta, label, crop)] of one of data/import_hooked.py's pool files (train/train2.py's shard format)."""
    z = np.load(path)
    data, offs, shapes = z["data"], z["offsets"], z["shapes"]
    labels, metas = json.loads(str(z["labels"])), json.loads(str(z["meta"]))
    return [(m, lab, data[offs[i]:offs[i + 1]].reshape(int(shapes[i][0]), int(shapes[i][1])))
            for i, (lab, m) in enumerate(zip(labels, metas))]


def cmd_hval_units(a):
    """The hval units and the crops they read (tier-independent): hc, the screening pool's verified dialogue lines
    (POOLS/hval/line, PSP from line_x2 in the Rubi view), as stored (the console's polarity); hp, every hval page's
    whole-dialogue regions (PAGEPOOL, eval/finder_tighten.py's rule) cut from the page (PSP upscaled 2x bilinear in
    the Rubi view, as Adrenaline shows it), each through the C GRAD finder (flags 0) and with the refinement
    (VOCR_REFINE_DEFAULT), each found line read with its dark flag as VOCR_INVERT."""
    import cv2
    import finder_tighten as FT

    import lines_refine as LR
    from common import to_gray
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    units, crops = [], {}
    pools = Path(a.pools) / "hval"
    for f in sorted((pools / "line").glob("*.npz")):
        tid = f.stem
        rows = read_pool(f)
        plat = rows[0][0]["platform"] if rows else None
        if a.view == "rubi" and plat == "psp":
            rows = read_pool(pools / "line_x2" / f.name)
        for m, lab, g in rows:
            if m.get("vertical") or m.get("name"):
                continue
            k = ("hc", tid, m["id"])
            crops[k] = (np.ascontiguousarray(g), False)
            units.append(Unit("hc", f"{tid}|{m['id']}", tid, False, [(norm(lab), [k])], m["platform"]))
    ident = FT.build(out / "build", "gcc")
    flags = {"grad": 0, "refine": LR.parse_flags("default")}
    imgs, meta = [], []
    for f in sorted(Path(a.pagepool).glob("*.jsonl")):
        for line in f.read_text(encoding="utf-8").splitlines():
            p = json.loads(line)
            if p["split"] != "hval" or not p["regions"]:
                continue
            text = {ln["id"]: ln["text"] for ln in p["lines"]}
            rgb = page_rgb(a.pages, p["title_id"], p["page"], p["image_sha256"])
            s = 1
            if a.view == "rubi" and p["platform"] == "psp":
                h, w = rgb.shape[:2]
                rgb, s = cv2.resize(rgb, (2 * w, 2 * h), interpolation=cv2.INTER_LINEAR), 2
            grey = to_gray(rgb)
            for r in p["regions"]:
                x0, y0, x1, y1 = (s * v for v in r["region"])
                imgs.append(np.ascontiguousarray(grey[y0:y1, x0:x1]))
                meta.append((p["title_id"], p["page"], r["variant"], p["platform"],
                             "".join(norm(text[i]) for i in r["lines"])))
    for mode, fl in flags.items():
        found = FT.lines_of(ident["driver"], imgs, fl)
        for img, (tid, page, variant, plat, ref), (boxes, dark, _, _, _) in zip(imgs, meta, found):
            keys = []
            for li, ((x, y, w, h), d) in enumerate(zip(boxes, dark)):
                k = ("hp", tid, page, variant, mode, li)
                crops[k] = (np.ascontiguousarray(img[y:y + h, x:x + w]), bool(d))
                keys.append(k)
            units.append(Unit("hp", f"{tid}|{page}|{variant}|{mode}", tid, False, [(ref, keys)], plat))
    (out / "units.pkl").write_bytes(pickle.dumps([u.state() for u in units], protocol=4))
    (out / "crops.pkl").write_bytes(pickle.dumps(crops, protocol=4))
    info = {"view": a.view, "pools": str(a.pools), "pagepool": str(a.pagepool), "units": len(units),
            "crops": len(crops), "by_part": {p: sum(u.part == p for u in units) for p in ("hc", "hp")},
            "titles": len({u.cluster for u in units}), "finder": {k: v for k, v in ident.items() if k != "driver"},
            "flags": flags, "created": utc_now()}
    (out / "units.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info), flush=True)


_R = {}


def _rinit(blob):
    from pyvocr import Vocr
    _R["rt"] = Vocr(str(blob), mode="w8i", max_width=1024)


def _rwork(job):
    k, img, inv = job
    o = _R["rt"].run_topk(img, K, invert=inv)
    return k, {x: o[x] for x in ("text", "topv", "topc", "blank", "conf_min", "frames")}


def cmd_hval_cache(a):
    """One tier's reads of every hval crop (the release blob, W8I, head gate off, top-8 extras)."""
    d = Path(a.dir)
    crops = pickle.loads((d / "crops.pkl").read_bytes())
    jobs = [(k, g, inv) for k, (g, inv) in crops.items()]
    t0 = time.time()
    with mp.get_context("fork").Pool(a.procs, initializer=_rinit, initargs=(BLOBS[a.tier],)) as pool:
        res = dict(pool.imap_unordered(_rwork, jobs, chunksize=16))
    reads = [(k, res[k]) for k, _, _ in jobs]
    doc = {"tier": a.tier, "blob": BLOBS[a.tier].name, "blob_sha256": sha256(BLOBS[a.tier]), "reads": reads,
           "seconds": round(time.time() - t0, 1), "created": utc_now()}
    (d / f"reads_{a.tier}.pkl").write_bytes(pickle.dumps(doc, protocol=4))
    print(json.dumps({k: doc[k] for k in ("tier", "blob_sha256", "seconds")} | {"reads": len(reads)}), flush=True)


def hval_load(d, tiers):
    """(units, {tier: [(key, read)]}) of the hval cache directory (hval-units, hval-cache)."""
    d = Path(d)
    units = [Unit(*st) for st in pickle.loads((d / "units.pkl").read_bytes())]
    reads = {}
    for t in tiers:
        c = pickle.loads((d / f"reads_{t}.pkl").read_bytes())
        if c["blob_sha256"] != sha256(BLOBS[t]):
            raise SystemExit(f"{d}: tier {t} read another blob")
        reads[t] = c["reads"]
    return units, reads


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("hval-units")
    p.add_argument("--pools", required=True)
    p.add_argument("--pagepool", required=True)
    p.add_argument("--pages", required=True)
    p.add_argument("--view", choices=("rubi", "native"), default="rubi")
    p.add_argument("--out", required=True)
    p = sub.add_parser("hval-cache")
    p.add_argument("--tier", required=True)
    p.add_argument("--dir", required=True)
    p.add_argument("--procs", type=int, default=5)
    p = sub.add_parser("tune")
    p.add_argument("--lm", required=True, help="NAME=PATH (NAME=none: greedy)")
    p.add_argument("--grid", default=json.dumps({"alpha": [500], "beta": [3500]}))
    p.add_argument("--cascade", default=str(CASCADE_CONF), help="cascade thresholds, comma-separated")
    p.add_argument("--tiers", default="S,N,L")
    p.add_argument("--hval-cache")
    p.add_argument("--rubi", help="a Rubi_for_vita checkout: its display rules (protocol amendment 1)")
    p.add_argument("--both", action="store_true", help="with --rubi: the original path's errors too (raw:<tier>)")
    p.add_argument("--procs", type=int, default=5)
    p.add_argument("--out", required=True)
    p = sub.add_parser("screen")
    p.add_argument("--lm", action="append", required=True)
    p.add_argument("--hval-text", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("summary")
    p.add_argument("--protocol", required=True)
    p.add_argument("--select-dir", required=True)
    p.add_argument("--screen", action="append", default=[])
    p.add_argument("--memo", action="append", default=[])
    p.add_argument("--lexicon", action="append", default=[])
    p.add_argument("--final-check", action="append", default=[])
    p.add_argument("--sensitivity", action="append", default=[])
    p.add_argument("--pagepath-check", action="append", default=[], help="the #129 benchmark check (report)")
    p.add_argument("--shipped")
    p.add_argument("--checks", help="the checks command's output (the audit's re-derived numbers)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("checks")
    p.add_argument("--select-dir", required=True)
    p.add_argument("--priv", required=True, help="the private dir: text/hook_train.txt, hval_lines.txt, *.json")
    p.add_argument("--hval-cache", required=True)
    p.add_argument("--pagepool", required=True)
    p.add_argument("--tune4", required=True)
    p.add_argument("--lm-dir", required=True)
    p.add_argument("--final-dir", required=True)
    p.add_argument("--pagepath", required=True, help="the #129 benchmark runs (old/new_dev.rows.jsonl)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("tables")
    p.add_argument("--json", required=True)
    p.add_argument("--doc", required=True)
    p = sub.add_parser("families")
    p.add_argument("--a", required=True, help="select output of family A (no game text)")
    p.add_argument("--b", required=True, help="select output of family B (game text)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("lex-tune")
    p.add_argument("--lm", required=True, help="NAME=PATH")
    p.add_argument("--config", required=True, help="the n-gram setting (JSON: alpha, beta, flags, ...)")
    p.add_argument("--lexicon", required=True, help="lm/lexicon.py's LEX.tsv")
    p.add_argument("--grid", required=True, help='JSON {"bonus": [...], "max_rank": [...]}')
    p.add_argument("--cascade", default=str(CASCADE_CONF))
    p.add_argument("--tiers", default="S,N,L")
    p.add_argument("--hval-cache")
    p.add_argument("--rubi", help="a Rubi_for_vita checkout: its display rules (protocol amendment 1)")
    p.add_argument("--both", action="store_true", help="with --rubi: the original path's errors too (raw:<tier>)")
    p.add_argument("--procs", type=int, default=5)
    p.add_argument("--out", required=True)
    p = sub.add_parser("select")
    p.add_argument("--protocol", required=True)
    p.add_argument("--tier", required=True, help="the LM tier (500k, 1000k, 2000k, 4000k)")
    p.add_argument("--incumbent", required=True, help="tune file of the incumbent LM at Rubi's setting")
    p.add_argument("--greedy", required=True, help="tune file of --lm greedy=none")
    p.add_argument("--tune", action="append", required=True, help="tune files of the tier's recipes")
    p.add_argument("--path", choices=("shown", "raw"), default="shown",
                   help="shown: Rubi's display rules (amendment 1; the decision); raw: the original line path")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    cmds = {"tune": cmd_tune, "screen": cmd_screen, "hval-units": cmd_hval_units, "hval-cache": cmd_hval_cache,
            "select": cmd_select, "lex-tune": cmd_lex_tune, "families": cmd_families, "summary": cmd_summary,
            "checks": cmd_checks, "tables": cmd_tables}
    cmds[a.cmd](a)


if __name__ == "__main__":
    main()
