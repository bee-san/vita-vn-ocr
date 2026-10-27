#!/usr/bin/env python3
"""VOCR6 stage 4 (vocr6-final-lm; #125, #127): Rubi's decoder settings on VOCR6's final S/N/L blobs, with #109's
synthetic counter set as a guard, and the counter re-decode thresholds on VOCR6's logits. The protocol is
eval/vocr6_final_lm_protocol.json (registered before any read of a final blob); this file implements it.

  vocr6_final_lm.py counter-cache --blob B --out FILE          one blob's reads of #109's counter set (private)
  vocr6_final_lm.py tune --protocol P --lm-tier 1000k --rubi RUBI --hval-dir DIR \\
      --blob S=FINAL_S --blob N=FINAL_N --blob L=FINAL_L --dev-cache S=FILE ... --hval-blob S=READER_S ... \\
      --counter-cache S=FILE ... --out DIR/1000k.npz [--configs FILE] [--arms b,c,g] [--procs 5]
  vocr6_final_lm.py select --protocol P --tune 1000k=FILE ... [--own 1000k=FILE ...] --out RESULT.json
  vocr6_final_lm.py continuity --cache FILE --blob B --out FILE.json   vocr6-rubi-ship's VOCR5-N counter numbers

Reads (eval/vocr6_lm.py's, unchanged): the dev units are VOCR5's development cohorts (tv, dn in both views, wd
through GRAD and GRAD + refine) without v12188 / v18432, read by each final blob (vocr6_lm.py dev-cache); the hval
units are vocr6-lm's hval_rubi set (hc 3,170 clean dialogue lines, hp 12,654 whole-dialogue region reads, 46
titles), read by each tier's hval reader (vocr6_lm.py hval-cache): the final blob when its training held no hval
page, else its stage-4a sibling of the same seed (hval out). The counter set is #109's 3,969 synthetic crops
(eval/kanum_dev.py render: 189 public development-pool lines x 3 forms x 7 left-edge cuts), read by each final blob.

Rubi's line path (Rubi_for_vita core/vocr_ocr.c finish_line, as eval/vocr6_lm.py models it): a line read as empty
stays empty; else the counter re-decode (tau, open) of its top-K, the beam + LM decode of the edited top-K at the
setting, the counter text rule; then the cascade (a line whose greedy confidence is under the threshold shows the
next tier's reading of the same crop, decoded by the same steps at the same setting: the job's setting); then what
Rubi shows (vocr6_lm.RubiShow: core/textnorm.c and the Shell's non-Japanese line filter). The dev units and the
counter set are decoded with the LM tier's shipped blob, the hval units with its screening build (the shipped
game-text blobs hold hval's text).

Every file this script writes outside eval/ holds development crops' readings or the owner's game text: private.
The JSON it writes into eval/ (select) holds counts, rates, settings and hashes only.
"""
import argparse
import json
import multiprocessing as mp
import pickle
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "train", ROOT / "synth", ROOT / "runtime", ROOT / "runtime" / "tests", ROOT / "lm"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import vocr6_lm as V  # noqa: E402
from metrics import lev, norm  # noqa: E402

HERD = V.HERD
PRIV = HERD / "shared" / "private" / "vocr6-final-lm"
SYNTH = HERD / "shared" / "artifacts" / "kanum-fix" / "synth" / "synth.npz"
UP = {"S": "N", "N": "L", "L": None}
TIERS = ("S", "N", "L")
K = 8
# vocr6-rubi-ship's kept test (artifacts/vocr6-rubi-ship/kanum_ke.py): the output holds a numeral, a small ke and a
# counter kanji; on eval/metrics.norm of the shown text
SITE = re.compile("[0-9０-９一二三四五六七八九十百千何数幾][ヶヵケ箇][月所年国条]")
CS_SEED = 5
REAL_PARTS = ("tv", "dn", "wd", "hc", "hp")


def sha256(p):
    return V.sha256(p)


def utc_now():
    return V.utc_now()


# ---------------------------------------------------------------- the counter set

def counter_units(synth=SYNTH):
    """One unit per crop of #109's counter set: part "cs", cluster = its source pool line (3 forms x 7 cuts each:
    eval/kanum_dev.py sy_texts gives three texts per pool line, in order), its key ("cs", crop id)."""
    import kanum_dev as KD
    rows = KD.sy_rows(synth)
    units = [V.Unit("cs", r["id"], f"cs{r['line'] // 3:03d}", False, [(r["ref"], [("cs", r["id"])])]) for r in rows]
    return units, rows


_C = {}


def _cinit(blob):
    from pyvocr import Vocr
    _C["rt"] = Vocr(str(blob), mode="w8i", max_width=1024)


def _cwork(job):
    k, img = job
    o = _C["rt"].run_topk(np.ascontiguousarray(img), K, invert=False)
    return k, {x: o[x] for x in ("text", "topv", "topc", "blank", "conf_min", "frames")}


def cmd_counter_cache(a):
    """One blob's reads of the counter set: W8I, head gate off, top-8, the blank's logit, conf_min (as kanum_dev.py
    cache reads it: the crops as drawn, bright text, no inversion)."""
    _, rows = counter_units(a.synth)
    jobs = [(("cs", r["id"]), r["img"]) for r in rows]
    t0 = time.time()
    with mp.get_context("fork").Pool(a.procs, initializer=_cinit, initargs=(a.blob,)) as pool:
        res = dict(pool.imap_unordered(_cwork, jobs, chunksize=16))
    doc = {"blob": str(a.blob), "blob_sha256": sha256(a.blob), "synth": str(a.synth), "synth_sha256": sha256(a.synth),
           "reads": [(k, res[k]) for k, _ in jobs], "seconds": round(time.time() - t0, 1), "created": utc_now()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_bytes(pickle.dumps(doc, protocol=4))
    print(json.dumps({"blob_sha256": doc["blob_sha256"], "reads": len(jobs), "seconds": doc["seconds"]}), flush=True)


def cs_kept(units, texts, show):
    """Per counter unit: whether the text Rubi shows keeps the counter (SITE), and its errors."""
    kept, err = [], []
    for u in units:
        ref, keys = u.segs[0]
        if show is None:
            pred = "".join(norm(texts[k]) for k in keys)
        else:
            raw = tuple(t for t in (texts[k] for k in keys) if t)
            pred = show.memo.get(raw)
            if pred is None:
                pred = show.memo[raw] = "".join(norm(x) for x in show(raw))
        kept.append(SITE.search(pred) is not None)
        err.append(lev(ref, pred))
    return np.array(kept, bool), np.array(err, np.int64)


# ---------------------------------------------------------------- reads

def load_reads(path, want_sha, what):
    d = pickle.loads(Path(path).read_bytes())
    if d["blob_sha256"] != want_sha:
        raise SystemExit(f"{path}: {what} read another blob ({d['blob_sha256'][:12]}) than {want_sha[:12]}")
    return d


class TierReads:
    """One tier's reads of every unit: the dev and counter-set reads of its final blob, the hval reads of its hval
    reader (the same vocabulary: checked). decode() gives Rubi's text of every read at a setting and counter
    thresholds; the counter re-decode's output is computed once per thresholds and kept for the shipped ones."""

    def __init__(self, tier, blob, groups, reader=None):
        from pyvocr import Vocr
        self.tier = tier
        self.rt = Vocr(str(blob), mode="w8i", max_width=1024)
        if reader is not None and Path(reader) != Path(blob):
            other = Vocr(str(reader), mode="w8i", max_width=1024)
            same = other.n_classes == self.rt.n_classes and all(
                other.class_text(c) == self.rt.class_text(c) for c in range(self.rt.n_classes))
            if not same:
                raise SystemExit(f"{tier}: the hval reader {reader} has another vocabulary than {blob}")
        self.keys, self.group, self.reads, self.greedy, self.conf = [], [], [], {}, {}
        for g, rs in groups.items():
            for k, o in rs:
                self.keys.append(k)
                self.group.append(g)
                self.reads.append(o)
                self.greedy[k] = o["text"]
                self.conf[k] = V.line_conf(o)
        self._src = {}

    def src(self, counter, keep=False):
        """[(topv, topc, blank) or None] after the counter re-decode at counter = (tau, open), and its edit count."""
        if counter in self._src:
            return self._src[counter]
        out, n = [], 0
        for o in self.reads:
            if not o["text"]:
                out.append(None)
                continue
            s = o
            if counter[0] or counter[1]:
                res, st = self.rt.redecode(o, *counter)
                if st["edits"]:
                    s, n = res, n + 1
            out.append((np.ascontiguousarray(s["topv"], np.float32), np.ascontiguousarray(s["topc"], np.uint16),
                        np.ascontiguousarray(s["blank"], np.float32)))
        if keep:
            self._src[counter] = (out, n)
        return out, n

    def decode(self, lms, opts, counter):
        """{key: Rubi's text}; lms: {group: LM bytes} (None for every group: the recognizer's greedy text, without
        the re-decode or the rule, as eval/vocr6_lm.py's greedy arm)."""
        from pyvocr import VocrLM, post_counters
        if lms is None:
            return dict(self.greedy)
        decs = {g: VocrLM(self.rt, lms[g], check_crc=False, **opts) for g in sorted(set(self.group))}
        src, _ = self.src(counter)
        lib = self.rt.lib
        return {k: "" if s is None else post_counters(decs[g].decode(*s), lib)[0]
                for k, g, s in zip(self.keys, self.group, src)}


def path_keys(thrs):
    """The job paths scored: S and N alone ("none") and with their cascade at each threshold; L alone."""
    return [("S", "none"), *[("S", t) for t in thrs], ("N", "none"), *[("N", t) for t in thrs], ("L", "none")]


def job_texts(tx, tiers, tier, thr):
    if thr == "none":
        return tx[tier]
    return V.cascade(tx[tier], tiers[tier].conf, tx[UP[tier]], thr)


_W = {}


def _init(state):
    _W.update(state)


def _job(job):
    arm, idx, cfg, counter = job
    tiers, units, cs_units, show = _W["tiers"], _W["units"], _W["cs_units"], _W["show"]
    if arm == "g":
        tx = {t: tiers[t].decode(None, None, None) for t in tiers}
    else:
        tx = {t: tiers[t].decode(_W["lms"], V.lm_opts(cfg), counter) for t in tiers}
    errs, kept = {}, {}
    for t, thr in _W["paths"]:
        texts = job_texts(tx, tiers, t, thr)
        errs[f"{t}|{thr}"] = V.unit_errors(units, texts, show).astype(np.int16)
        kept[f"{t}|{thr}"] = cs_kept(cs_units, texts, show)[0]
    return arm, idx, errs, kept


def protocol_grid(proto, lm_tier):
    """(configs of arm b in grid order, the shipped config, arm c's (tau, open) cells, cascade thresholds)."""
    g = proto["arms"]["b"]["grid"]
    cfgs = V.grid_configs(g)
    ship = dict(proto["lm_tiers"][lm_tier]["shipped"])
    if ship not in cfgs:
        raise SystemExit(f"{lm_tier}: the shipped setting {ship} is not in arm b's grid")
    c = proto["arms"]["c"]
    cells = [(int(t), int(o)) for t in c["tau"] for o in c["open"]]
    return cfgs, ship, cells, [int(t) for t in proto["arms"]["b"]["cascade_conf"]]


def cmd_tune(a):
    """Arm b's grid, arm c's counter cells and greedy, decoded on every unit of every tier for one LM tier: errors
    per job path and the counter set's kept flags (private npz)."""
    proto = json.loads(Path(a.protocol).read_text(encoding="utf-8"))
    lt = proto["lm_tiers"][a.lm_tier]
    cfgs, ship, cells, thrs = protocol_grid(proto, a.lm_tier)
    casc_ship = int(lt["cascade_conf"])
    if casc_ship not in thrs:
        raise SystemExit(f"{a.lm_tier}: the shipped cascade threshold {casc_ship} is not selectable")
    if a.configs:
        cfgs = json.loads(Path(a.configs).read_text(encoding="utf-8"))
    arms = a.arms.split(",")
    blobs = dict(s.split("=", 1) for s in a.blob)
    readers = dict(s.split("=", 1) for s in a.hval_blob) if a.hval_blob else {}
    devc = dict(s.split("=", 1) for s in a.dev_cache)
    csc = dict(s.split("=", 1) for s in a.counter_cache)
    for t in TIERS:
        readers.setdefault(t, blobs[t])
    dev_units, _ = V.dev_units()
    hval = Path(a.hval_dir)
    hval_units = [V.Unit(*st) for st in pickle.loads((hval / "units.pkl").read_bytes())]
    cs_units, _ = counter_units(a.synth)
    units = dev_units + hval_units + cs_units
    tiers, prov = {}, {}
    for t in TIERS:
        bs, rs = sha256(blobs[t]), sha256(readers[t])
        if devc[t] == "headgate":                       # VOCR5's release blobs: the head gate cache's reads
            if bs != sha256(V.BLOBS[t]):
                raise SystemExit(f"{t}: the head gate cache holds VOCR5-{t}'s reads, not {blobs[t]}'s")
            dv = {"reads": V.dev_reads(t)[0]}
        else:
            dv = load_reads(devc[t], bs, f"{t} dev")
        hv = load_reads(hval / f"reads_{t}.{Path(readers[t]).stem}.pkl", rs, f"{t} hval")
        cs = load_reads(csc[t], bs, f"{t} counter set")
        tiers[t] = TierReads(t, blobs[t], {"dev": dv["reads"], "hval": hv["reads"], "cs": cs["reads"]}, readers[t])
        prov[t] = {"blob": str(blobs[t]), "blob_sha256": bs, "hval_reader": str(readers[t]), "hval_reader_sha256": rs,
                   "reads": {"dev": len(dv["reads"]), "hval": len(hv["reads"]), "cs": len(cs["reads"])}}
    need = {k for u in units for _, ks in u.segs for k in ks}
    for t in TIERS:
        missing = need - set(tiers[t].keys)
        if missing:
            raise SystemExit(f"{t}: {len(missing)} unit keys have no read (e.g. {sorted(missing)[:2]})")
    lms = {g: Path(lt[src]["path"]).read_bytes() for g, src in (("dev", "dev"), ("cs", "dev"), ("hval", "hval"))}
    for src in ("dev", "hval"):
        if sha256(lt[src]["path"]) != lt[src]["sha256"]:
            raise SystemExit(f"{a.lm_tier}: {lt[src]['path']} is not the pinned LM")
    shipped_counter = (int(proto["counter_shipped"]["tau"]), int(proto["counter_shipped"]["open"]))
    for t in TIERS:                                     # the shipped thresholds' re-decode, shared by every b job
        tiers[t].src(shipped_counter, keep=True)
    show = V.RubiShow(a.rubi, PRIV / "build")
    paths = path_keys(thrs)
    jobs = []
    if "b" in arms:
        jobs += [("b", j, c, shipped_counter) for j, c in enumerate(cfgs)]
    if "c" in arms:
        jobs += [("c", i, ship, cell) for i, cell in enumerate(cells)]
    if "g" in arms:
        jobs.append(("g", 0, None, None))
    state = {"tiers": tiers, "units": units, "cs_units": cs_units, "show": show, "lms": lms, "paths": paths}
    t0 = time.time()
    _init(state)
    with mp.get_context("fork").Pool(a.procs) as pool:
        res = pool.map(_job, jobs, chunksize=1)
    out = {}
    for arm in arms:
        rows = sorted((r for r in res if r[0] == arm), key=lambda r: r[1])
        for key in rows[0][2]:
            out[f"E_{arm}_{key}"] = np.stack([r[2][key] for r in rows])
            out[f"K_{arm}_{key}"] = np.stack([r[3][key] for r in rows])
    edits = {t: {"shipped": tiers[t].src(shipped_counter)[1]} for t in TIERS}
    meta = {"kind": "vocr6-final-lm-tune", "lm_tier": a.lm_tier, "protocol_sha256": sha256(a.protocol),
            "lms": {k: {"path": lt[k]["path"], "sha256": lt[k]["sha256"]} for k in ("dev", "hval")},
            "configs": cfgs, "shipped": ship, "cells": cells, "counter_shipped": shipped_counter, "arms": arms,
            "cascade_shipped": casc_ship,
            "paths": [f"{t}|{thr}" for t, thr in paths], "tiers": prov, "counter_edits": edits,
            "units": [[u.part, u.uid, u.cluster, u.name, u.chars, u.platform] for u in units],
            "hval_dir": str(hval), "rubi_textnorm_sha256": V.RUBI_TEXTNORM_SHA256,
            "seconds": round(time.time() - t0, 1), "created": utc_now()}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, meta=json.dumps(meta), **out)
    print(json.dumps({"lm_tier": a.lm_tier, "jobs": len(jobs), "seconds": meta["seconds"]}), flush=True)


# ---------------------------------------------------------------- selection and the rule (pure)

def first_min(values, order):
    """The first option in `order` whose value is the minimum (ties go to the earlier option)."""
    best = None
    for i in order:
        if best is None or values[i] < values[best] - 1e-12:
            best = i
    return best


def cv_choose(obj0, obj1, obj_all, order):
    """2-fold cross-validated choice: (fold 0's choice, fold 1's, in-sample), each the first minimum in order."""
    return first_min(obj0, order), first_min(obj1, order), first_min(obj_all, order)


def cv_errors(fold, e_by_option, c0, c1):
    """Each fold's units scored by the other fold's choice: fold 0 by c1 (chosen on fold 1), fold 1 by c0."""
    return np.where(fold == 0, e_by_option[c1], e_by_option[c0])


def cs_compare(clusters, chars, ex, ey, kx, ky, n_boot=V.N_BOOT, seed=CS_SEED):
    """Paired y - x on the counter set: CER (points) and the share of crops keeping the counter (points), each with
    its 95% interval over the source pool lines (cluster bootstrap)."""
    ug, inv = np.unique(np.asarray(clusters), return_inverse=True)
    ch = np.asarray(chars, float)
    gx, gy, gc = (np.bincount(inv, weights=np.asarray(v, float), minlength=len(ug)) for v in (ex, ey, ch))
    kxg, kyg, n = (np.bincount(inv, weights=np.asarray(v, float), minlength=len(ug)) for v in
                   (kx, ky, np.ones(len(inv))))
    draw = np.random.default_rng(seed).integers(0, len(ug), size=(n_boot, len(ug)))
    cer = (gy[draw].sum(1) - gx[draw].sum(1)) / gc[draw].sum(1) * 100
    kept = (kyg[draw].sum(1) - kxg[draw].sum(1)) / n[draw].sum(1) * 100
    pc = (float(np.sum(ey)) - float(np.sum(ex))) / float(ch.sum()) * 100
    pk = (float(np.sum(ky)) - float(np.sum(kx))) / len(inv) * 100
    return {"cer": V.ci(pc, cer), "kept_share": V.ci(pk, kept), "clusters": int(len(ug)),
            "kept": [int(np.sum(kx)), int(np.sum(ky))], "errors": [int(np.sum(ex)), int(np.sum(ey))],
            "crops": int(len(inv)), "chars": int(ch.sum())}


def worse(c):
    return c["ci95"][0] > 0


def apply_rule(cv_cmp, cs_cmp, admissible):
    """The protocol's rule for one arm at one tier: the cross-validated objective resolved better than the shipped
    setting's, none of hp, hc and the dev score resolved worse (cross-validated), the counter set's CER not resolved
    worse and its kept share not resolved lower (the setting that would ship), and the setting admissible."""
    out = {"objective_better": bool(cv_cmp["obj"]["ci95"][1] < 0),
           "hp_not_worse": not worse(cv_cmp["hp"]), "hc_not_worse": not worse(cv_cmp["hc"]),
           "dev_not_worse": not worse(cv_cmp["dev"]),
           "counter_cer_not_worse": not worse(cs_cmp["cer"]),
           "counter_kept_not_worse": not (cs_cmp["kept_share"]["ci95"][1] < 0),
           "admissible": bool(admissible)}
    out["passes"] = bool(all(out.values()))
    return out


def decide(rule_b, rule_c, d_b, d_c):
    """keep, b or c: an arm that passes; both: the lower cross-validated objective difference (point), b on a tie."""
    if rule_b["passes"] and rule_c["passes"]:
        return "b" if d_b <= d_c else "c"
    return "b" if rule_b["passes"] else "c" if rule_c["passes"] else "keep"


def admissible(cfg, counter, ws, ws_limit):
    """The runtime's documented preconditions (docs/counters.md: the counter re-decode's tau and open at most the
    LM's gate; Rubi: the decoder workspace within VJO_VOCR_LM_WS_BYTES)."""
    return bool(counter[0] <= cfg["gate"] and counter[1] <= cfg["gate"] and ws <= ws_limit)


# ---------------------------------------------------------------- select

def option_order(n_cfg, ship_j, thrs, ship_thr):
    """Arm b's options (config, threshold) in tie-break order: the shipped option first, then grid order."""
    opts = [(j, t) for j in range(n_cfg) for t in thrs]
    first = (ship_j, ship_thr)
    return [first] + [o for o in opts if o != first]


def workspace(blob_paths, lm_path, cfg):
    from pyvocr import Vocr, VocrLM
    return {t: VocrLM(Vocr(str(p), mode="w8i", max_width=1024), lm_path, check_crc=False,
                      **V.lm_opts(cfg)).workspace_bytes for t, p in blob_paths.items()}


def tier_select(sc, U, E, Kk, cs_idx, cs_units, tier, cfgs, ship_j, thrs, cells, ship_cell, paths_present,
                casc_ship):
    """Arms b and c at one job tier: choices, cross-validated and in-sample scores, comparisons, counter set."""
    tthrs = thrs if tier != "L" else ["none"]
    ship_thr = casc_ship if tier != "L" else "none"

    def key(t):
        return f"{tier}|{t}"
    for t in tthrs:
        if key(t) not in paths_present:
            raise SystemExit(f"tune file lacks {key(t)}")
    out = {}
    e_ship = E["b"][key(ship_thr)][ship_j]
    k_ship = Kk["b"][key(ship_thr)][ship_j]
    cl = [u.cluster for u in cs_units]
    ch = [u.chars for u in cs_units]
    # arm b: options (config, threshold)
    order = option_order(len(cfgs), ship_j, tthrs, ship_thr)
    flat = {o: i for i, o in enumerate(order)}
    eb = np.stack([E["b"][key(t)][j] for j, t in order])
    scores = {f: sc.scores(eb, fold=f)["obj"] for f in (0, 1, None)}
    idx = list(range(len(order)))
    c0, c1, ca = cv_choose(scores[0], scores[1], scores[None], idx)
    e_cv = cv_errors(sc.fold, eb, c0, c1)
    e_in = eb[ca]
    k_in = Kk["b"][key(order[ca][1])][order[ca][0]]

    def describe(i):
        j, t = order[i]
        return {"config": cfgs[j], "cascade_conf": t, "is_shipped": bool(i == 0)}
    out["b"] = {"options": len(order), "chosen": {"fold0": describe(c0), "fold1": describe(c1),
                                                  "in_sample": describe(ca)},
                "shipped": V.rounded(sc.scores(e_ship)), "cv": V.rounded(sc.scores(e_cv)),
                "in_sample": V.rounded(sc.scores(e_in)),
                "cv_minus_shipped": V.compare(U, e_ship, e_cv), "in_sample_minus_shipped": V.compare(U, e_ship, e_in),
                "counter": cs_compare(cl, ch, e_ship[cs_idx], e_in[cs_idx], k_ship, k_in),
                "counter_folds": {f: cs_compare(cl, ch, e_ship[cs_idx], eb[c][cs_idx], k_ship,
                                                Kk["b"][key(order[c][1])][order[c][0]])
                                  for f, c in (("fold0", c0), ("fold1", c1))},
                "_flat": flat, "_choice": (order[c0], order[c1], order[ca])}
    # arm c: counter cells at the shipped setting and threshold (the shipped cell first)
    corder = [ship_cell] + [i for i in range(len(cells)) if i != ship_cell]
    ec = E["c"][key(ship_thr)]
    kc = Kk["c"][key(ship_thr)]
    s = {f: sc.scores(ec, fold=f)["obj"] for f in (0, 1, None)}
    d0, d1, da = cv_choose(s[0], s[1], s[None], corder)
    e_cvc = cv_errors(sc.fold, ec, d0, d1)
    out["c"] = {"options": len(cells), "chosen": {"fold0": cells[d0], "fold1": cells[d1], "in_sample": cells[da]},
                "cv": V.rounded(sc.scores(e_cvc)), "in_sample": V.rounded(sc.scores(ec[da])),
                "cv_minus_shipped": V.compare(U, e_ship, e_cvc),
                "in_sample_minus_shipped": V.compare(U, e_ship, ec[da]),
                "counter": cs_compare(cl, ch, e_ship[cs_idx], ec[da][cs_idx], k_ship, kc[da]),
                "cells": {f"{t}/{o}": {"objective": round(float(s[None][i]), 4),
                                       "counter_kept": int(kc[i].sum()),
                                       "counter_cer": round(float(ec[i][cs_idx].sum() / np.sum(ch) * 100), 4),
                                       "counter_vs_shipped": cs_compare(cl, ch, e_ship[cs_idx], ec[i][cs_idx], k_ship,
                                                                        kc[i])}
                          for i, (t, o) in enumerate(cells)},
                "_choice": (d0, d1, da)}
    # the shipped equals arm c's shipped cell (the same decode twice): a harness check
    out["check_shipped_twice_equal"] = bool(np.array_equal(ec[ship_cell], e_ship) and
                                            np.array_equal(kc[ship_cell], k_ship))
    return out


def by_platform(sc, e_x, e_y):
    return {pl: {p: {"x": round(float(sc.cer(e_x, p, plat=pl)), 4), "y": round(float(sc.cer(e_y, p, plat=pl)), 4)}
                 for p in ("hc", "hp")} for pl in ("vita", "psp")}


def names(sc, e_x, e_y):
    return {p: {"x": round(float(sc.cer(e_x, p, name=True)), 4), "y": round(float(sc.cer(e_y, p, name=True)), 4)}
            for p in ("tv", "dn")}


def load_tune(path):
    z = np.load(path)
    meta = json.loads(str(z["meta"]))
    E, Kk = {}, {}
    for f in z.files:
        if f == "meta":
            continue
        kind, arm, key = f.split("_", 2)
        (E if kind == "E" else Kk).setdefault(arm, {})[key] = z[f].astype(np.int64) if kind == "E" else z[f]
    return meta, E, Kk


def cmd_select(a):
    """The protocol's rule per LM tier and job tier (S + cascade, N + cascade, L); the result record (public)."""
    import finder_tighten as FT
    proto = json.loads(Path(a.protocol).read_text(encoding="utf-8"))
    out = {"kind": "vocr6-final-lm-result", "schema": 1, "created": utc_now(), "label": "vocr6-final-lm",
           "issues": [125, 127], "protocol_sha256": sha256(a.protocol), "lm_tiers": {}, "decisions": {}}
    own = dict(s.split("=", 1) for s in a.own) if a.own else {}
    ws_limit = int(proto["rule"]["workspace_limit_bytes"])
    decided = list(TIERS)
    for am in proto.get("amendments", []):              # amendment 2: only these job tiers are decided
        decided = am.get("decide_tiers", decided)
    out["decide_tiers"] = decided
    for spec in a.tune:
        lm_tier, path = spec.split("=", 1)
        meta, E, Kk = load_tune(path)
        if meta["protocol_sha256"] != sha256(a.protocol):
            raise SystemExit(f"{path}: tuned under another protocol")
        units = meta["units"]
        sc = V.Scorer(units)
        U = [V.Unit(u[0], u[1], u[2], u[3], [], u[5]) for u in units]
        for u, x in zip(U, units):
            u.chars = x[4]
        cs_idx = np.array([u[0] == "cs" for u in units])
        cs_units = [u for u in U if u.part == "cs"]
        cfgs, ship = meta["configs"], meta["shipped"]
        ship_j = cfgs.index(ship)
        cells = [tuple(c) for c in meta["cells"]]
        ship_cell = cells.index(tuple(meta["counter_shipped"]))
        thrs = [int(p.split("|")[1]) for p in meta["paths"] if p.startswith("S|") and p != "S|none"]
        rec = {"tune": Path(path).name, "lms": meta["lms"], "tiers": meta["tiers"], "counter_edits":
               meta["counter_edits"], "shipped": ship, "grid_configs": len(cfgs), "counter_cells": len(cells),
               "units": {p: int(sc.mask(p).sum()) for p in (*REAL_PARTS, "cs")},
               "clusters": {p: int(len(set(sc.cluster[sc.mask(p)]))) for p in (*REAL_PARTS, "cs")}, "jobs": {}}
        blobs = {t: meta["tiers"][t]["blob"] for t in TIERS}
        casc_ship = int(meta["cascade_shipped"])
        for tier in TIERS:
            r = tier_select(sc, U, E, Kk, cs_idx, cs_units, tier, cfgs, ship_j, thrs, cells, ship_cell,
                            set(E["b"]), casc_ship)
            ship_thr = casc_ship if tier != "L" else "none"
            if "g" in E:                                # the recognizer alone (#127's greedy), and with its cascade
                g = E["g"][f"{tier}|none"][0]
                r["greedy"] = V.rounded(sc.scores(g))
                r["greedy_cascade"] = V.rounded(sc.scores(E["g"][f"{tier}|{ship_thr}"][0]))
                r["shipped_minus_greedy"] = V.compare(U, g, E["b"][f"{tier}|{ship_thr}"][ship_j])
            jb, tb = r["b"]["_choice"][2]
            e_in = E["b"][f"{tier}|{tb}"][jb]
            e_ship = E["b"][f"{tier}|{ship_thr}"][ship_j]
            r["b"]["by_platform"] = by_platform(sc, e_ship, e_in)
            r["b"]["names"] = names(sc, e_ship, e_in)
            r["no_cascade"] = {"shipped": V.rounded(sc.scores(E["b"][f"{tier}|none"][ship_j]))}
            job_blobs = {t: blobs[t] for t in (tier, UP[tier]) if t}
            lm_dev = meta["lms"]["dev"]["path"]
            ws_b = workspace(job_blobs, lm_dev, cfgs[jb])
            ws_c = workspace(job_blobs, lm_dev, ship)
            cb = tuple(meta["counter_shipped"])
            cc = cells[r["c"]["_choice"][2]]
            r["b"]["workspace_bytes"] = ws_b
            r["c"]["workspace_bytes"] = ws_c
            rb = apply_rule(r["b"]["cv_minus_shipped"], r["b"]["counter"],
                            admissible(cfgs[jb], cb, max(ws_b.values()), ws_limit))
            rc = apply_rule(r["c"]["cv_minus_shipped"], r["c"]["counter"],
                            admissible(ship, cc, max(ws_c.values()), ws_limit))
            d = decide(rb, rc, r["b"]["cv_minus_shipped"]["obj"]["delta_points"],
                       r["c"]["cv_minus_shipped"]["obj"]["delta_points"])
            r["b"]["rule"], r["c"]["rule"] = rb, rc
            r["decision"] = d
            if lm_tier in own:
                r["own_hval"] = own_report(own[lm_tier], tier, ship, cfgs[jb], tb, ship_thr)
            for arm in ("b", "c"):
                r[arm].pop("_flat", None)
                r[arm].pop("_choice", None)
            rec["jobs"][tier] = r
            setting = {"keep": None, "b": {"config": cfgs[jb], "cascade_conf": tb, "counter": list(cb)},
                       "c": {"config": ship, "cascade_conf": ship_thr, "counter": list(cc)}}[d]
            if tier in decided:
                out["decisions"].setdefault(tier, {})[lm_tier] = {"decision": "keep" if d == "keep" else "change",
                                                                 "arm": d, "setting": setting}
            else:                                       # amendment 2: reported, not decided
                out["decisions"].setdefault(tier, {})[lm_tier] = {"decision": "report only", "arm": "keep",
                                                                 "setting": None, "rule_outcome": d,
                                                                 "rule_setting": setting}
        out["lm_tiers"][lm_tier] = rec
    FT.assert_public(out)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out["decisions"], ensure_ascii=False), flush=True)


def own_report(path, tier, ship, cfg_b, thr_b, ship_thr):
    """Report only: the final blob's own hval reads (a blob that trained on hval) at the shipped setting and at arm
    b's in-sample choice, when the own tune file decoded both."""
    meta, E, _ = load_tune(path)
    units = meta["units"]
    sc = V.Scorer(units)
    U = [V.Unit(u[0], u[1], u[2], u[3], [], u[5]) for u in units]
    for u, x in zip(U, units):
        u.chars = x[4]
    cfgs = meta["configs"]
    if ship not in cfgs or cfg_b not in cfgs:
        return {"missing": True}
    ex = E["b"][f"{tier}|{ship_thr}"][cfgs.index(ship)]
    ey = E["b"][f"{tier}|{thr_b}"][cfgs.index(cfg_b)]
    c = V.compare(U, ex, ey)
    return {"tune": Path(path).name, "shipped": V.rounded(sc.scores(ex)), "in_sample_b": V.rounded(sc.scores(ey)),
            "in_sample_b_minus_shipped": {k: c[k] for k in ("hc", "hp") if k in c}}


# ---------------------------------------------------------------- continuity with vocr6-rubi-ship (#306)

def cmd_continuity(a):
    """vocr6-rubi-ship's counter-set numbers (VOCR5-N alone, no cascade, no display rules, Rubi's counter re-decode
    6 / 4 nats + beam + LM + counter rule): ctclm.2 1 MB at alpha 500 / beta 3500 / EOS kept 3,104 crops (1,847
    errors), the game-text 1 MB LM at 600 / 2500 / BOS 3,015 (1,936). Counts only."""
    from pyvocr import Vocr, VocrLM, post_counters
    d = pickle.loads(Path(a.cache).read_bytes())
    if d["blob_sha256"] != sha256(a.blob):
        raise SystemExit("the cache read another blob")
    cs_units, _ = counter_units(a.synth)
    rt = Vocr(str(a.blob), mode="w8i", max_width=1024)
    reads = dict((k, o) for k, o in d["reads"])
    out = {"kind": "vocr6-final-lm-continuity", "created": utc_now(), "blob_sha256": d["blob_sha256"], "arms": {}}
    for spec in a.lm:
        name, path, al, be, fl = spec.split(":")
        dec = VocrLM(rt, path, alpha=int(al), beta=int(be), flags=int(fl), beam=8, topk=8, gate=8000, bracket=8000)
        texts = {}
        for k, o in reads.items():
            if not o["text"]:
                texts[k] = ""
                continue
            res, st = rt.redecode(o, 6000, 4000)
            s = res if st["edits"] else o
            texts[k] = post_counters(dec.decode(s["topv"], s["topc"], s["blank"]), rt.lib)[0]
        kept, err = cs_kept(cs_units, texts, None)
        out["arms"][name] = {"lm_sha256": sha256(path), "alpha": int(al), "beta": int(be), "flags": int(fl),
                             "kept": int(kept.sum()), "errors": int(err.sum()),
                             "chars": int(sum(u.chars for u in cs_units)), "crops": len(cs_units)}
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out["arms"]), flush=True)


# ---------------------------------------------------------------- gate-on reads (report only)

KANA_COUNTER = re.compile("[カケ][月所年国条]")      # Rubi core/vocr_ocr.c kana_counter: full-size カ / ケ before one
READ_KEYS = ("text", "topv", "topc", "blank", "conf_min", "frames")
_G = {}


def _ginit(blob, two):
    from pyvocr import Vocr
    kw = {"two_level": 1} if two else {}
    _G["rt"] = Vocr(str(blob), mode="w8i", max_width=1024, head_gate="on", **kw)
    _G["two"] = bool(two)


def rubi_read(rt, two, img, inv, counter=(6000, 4000)):
    """One crop read as Rubi reads it with ocr_head_gate = on (core/vocr_ocr.c finish_line, read_again): the gated
    (and, with two, two-level) run; when its run skipped something (frames the gate skipped; with two, frames that
    ran the frequent rows only) and either its counter re-decode finds a counter context or (two) its greedy text
    holds a full-size カ / ケ before a counter character, the line is read again with the full head (ungated) and that
    reading stands. Returns (read, re_read)."""
    o = rt.run_topk(img, K, invert=inv)
    if o["text"]:
        _, st = rt.redecode(o, *counter)
        skipped = o["gated"] > 0 or (two and o["escaped"] + o["gated"] < o["frames"])
        if skipped and (st["contexts"] or (two and KANA_COUNTER.search(o["text"]))):
            o2 = rt.run_topk(img, K, invert=inv, ungated=True)
            return {x: o2[x] for x in READ_KEYS}, True
    return {x: o[x] for x in READ_KEYS}, False


def _gwork(job):
    k, img, inv = job
    r, again = rubi_read(_G["rt"], _G["two"], np.ascontiguousarray(img), bool(inv))
    return k, r, again


def cmd_gated_cache(a):
    """Report only: one gated blob's reads of the dev cohorts, the hval crops and the counter set as Rubi reads them
    with ocr_head_gate = on (rubi_read), in dev-cache / hval-cache / counter-cache's formats (private)."""
    import headgate_eval as HE
    from pyvocr import Vocr

    def has_two_level(blob):
        try:
            Vocr(str(blob), mode="w8i", max_width=1024, two_level=1)
            return True
        except ValueError:
            return False
    two = has_two_level(a.blob) if a.two_level == "auto" else a.two_level == "on"
    if not Vocr(str(a.blob), mode="w8i", max_width=1024).has_gate:
        raise SystemExit(f"{a.blob}: no head gate in this blob")
    bs, stem = sha256(a.blob), Path(a.blob).stem
    out = {}
    t0 = time.time()
    fixed = HE.fixed_rows()
    wd, ident, flags = HE.wd_lines(Path(a.dev_out).parent / "build")
    rd = HE.reads(fixed, wd)
    keys = HE.row_keys([k for k, _, _ in rd], fixed)
    crops = pickle.loads((Path(a.hval_dir) / "crops.pkl").read_bytes())
    _, rows = counter_units(a.synth)
    groups = {"dev": [(keys[i], img, inv) for i, (_, img, inv) in enumerate(rd)],
              "hval": [(k, g, inv) for k, (g, inv) in crops.items()],
              "cs": [(("cs", r["id"]), r["img"], False) for r in rows]}
    with mp.get_context("fork").Pool(a.procs, initializer=_ginit, initargs=(a.blob, two)) as pool:
        for g, jobs in groups.items():
            res = {k: (r, again) for k, r, again in pool.imap_unordered(_gwork, jobs, chunksize=16)}
            out[g] = ([(k, res[k][0]) for k, _, _ in jobs], sum(v[1] for v in res.values()))
    base = {"tier": a.tier, "blob": str(a.blob), "blob_sha256": bs, "head_gate": "on", "two_level": two,
            "counter_reread": "rubi core/vocr_ocr.c read_again at 6000 / 4000", "created": utc_now()}
    Path(a.dev_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.dev_out).write_bytes(pickle.dumps(dict(base, reads=out["dev"][0], rereads=out["dev"][1],
                                                  meta={"reads": len(rd), "flags": flags}), protocol=4))
    (Path(a.hval_dir) / f"reads_{a.tier}.{stem}.pkl").write_bytes(
        pickle.dumps(dict(base, reads=out["hval"][0], rereads=out["hval"][1]), protocol=4))
    Path(a.counter_out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.counter_out).write_bytes(pickle.dumps(dict(base, reads=out["cs"][0], rereads=out["cs"][1],
                                                      synth_sha256=sha256(a.synth)), protocol=4))
    print(json.dumps({"tier": a.tier, "blob_sha256": bs, "two_level": two, "seconds": round(time.time() - t0, 1),
                      "reads": {g: len(v[0]) for g, v in out.items()}, "rereads": {g: v[1] for g, v in out.items()}}),
          flush=True)


def cmd_compare_runs(a):
    """Report only: two tune files over the same units (e.g. gate-on reads against gate-off reads), at one setting
    each (its index in the file's configs), per job path: scores, the paired differences y - x and the counter set."""
    import finder_tighten as FT
    mx, ex_, kx_ = load_tune(a.x)
    my, ey_, ky_ = load_tune(a.y)
    if mx["units"] != my["units"]:
        raise SystemExit("the two tune files scored other units")
    units = mx["units"]
    sc = V.Scorer(units)
    U = [V.Unit(u[0], u[1], u[2], u[3], [], u[5]) for u in units]
    for u, x in zip(U, units):
        u.chars = x[4]
    cs_idx = np.array([u[0] == "cs" for u in units])
    cs = [u for u in U if u.part == "cs"]
    cfg = json.loads(a.config)
    jx, jy = mx["configs"].index(cfg), my["configs"].index(cfg)
    out = {"kind": "vocr6-final-lm-compare-runs", "created": utc_now(), "what": a.what, "config": cfg,
           "x": {"tune": Path(a.x).name, "tiers": mx["tiers"]}, "y": {"tune": Path(a.y).name, "tiers": my["tiers"]},
           "paths": {}}
    for key in sorted(set(ex_["b"]) & set(ey_["b"])):
        e0, e1 = ex_["b"][key][jx], ey_["b"][key][jy]
        k0, k1 = kx_["b"][key][jx], ky_["b"][key][jy]
        out["paths"][key] = {"x": V.rounded(sc.scores(e0)), "y": V.rounded(sc.scores(e1)),
                             "y_minus_x": V.compare(U, e0, e1),
                             "units_changed": int((e0 != e1)[~cs_idx].sum()),
                             "counter": cs_compare([u.cluster for u in cs], [u.chars for u in cs], e0[cs_idx],
                                                   e1[cs_idx], k0, k1)}
    FT.assert_public(out)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for key, r in out["paths"].items():
        d = r["y_minus_x"]
        print(key, json.dumps({k: [d[k]["delta_points"], d[k]["ci95"]] for k in ("hc", "hp", "dev", "obj") if k in d}),
              "counter kept", r["counter"]["kept"], "errors", r["counter"]["errors"], flush=True)


# ---------------------------------------------------------------- summary (result.json, tables)

def _pt(c):
    return f"{c['delta_points']:+.3f} ({c['ci95'][0]:+.3f} to {c['ci95'][1]:+.3f})"


def summarise(sel):
    """The compact record (result.json) and markdown tables of a select output."""
    res = {"kind": "vocr6-final-lm-summary", "schema": 1, "created": utc_now(), "protocol_sha256":
           sel["protocol_sha256"], "decisions": sel["decisions"], "blobs": {}, "jobs": {}}
    md = ["| tier | LM | decision | shipped: obj / hp / hc / dev | b (CV) - shipped, objective | b chosen (in-sample) "
          "| b guards: hp / hc / dev | c (CV) - shipped, objective | c chosen |",
          "|---|---|---|---|---|---|---|---|---|"]
    cs = ["| tier | LM | arm | counters kept (of 3,969) | counter-set CER | kept share - shipped | CER - shipped |",
          "|---|---|---|---:|---:|---|---|"]
    for lt, rec in sel["lm_tiers"].items():
        for t, b in rec["tiers"].items():
            res["blobs"][t] = {k: b[k] for k in ("blob", "blob_sha256", "hval_reader", "hval_reader_sha256")}
        for tier, r in rec["jobs"].items():
            sh, rb, rc = r["b"]["shipped"], r["b"], r["c"]
            cb, cc = rb["cv_minus_shipped"], rc["cv_minus_shipped"]
            d = sel["decisions"][tier][lt]
            res["jobs"].setdefault(tier, {})[lt] = {
                "decision": d, "greedy": r.get("greedy"), "shipped": sh,
                "b": {"chosen": rb["chosen"], "cv": rb["cv"], "cv_minus_shipped": cb, "rule": rb["rule"],
                      "counter": rb["counter"], "workspace_bytes": rb["workspace_bytes"]},
                "c": {"chosen": rc["chosen"], "cv": rc["cv"], "cv_minus_shipped": cc, "rule": rc["rule"],
                      "counter": rc["counter"],
                      "cells": {k: {"objective": v["objective"], "counter_kept": v["counter_kept"],
                                    "counter_cer": v["counter_cer"]} for k, v in rc["cells"].items()}},
                "counter_edits_shipped": rec["counter_edits"][tier]["shipped"],
                "check_shipped_twice_equal": r["check_shipped_twice_equal"]}
            ch = rb["chosen"]["in_sample"]
            c0 = ch["config"]
            fl = "|".join(n for bit, n in ((1, "BOS"), (2, "EOS")) if c0["flags"] & bit) or "0"
            arm = "" if d["arm"] == "keep" else f" ({d['arm']})"
            mark = " (shipped)" if ch["is_shipped"] else ""
            casc = "" if ch["cascade_conf"] == "none" else f" c{ch['cascade_conf']}"
            ci_, co_ = rc["chosen"]["in_sample"]
            md.append(f"| {tier} | {lt} | {d['decision']}{arm} | "
                      f"{sh['obj']:.3f} / {sh['hp']:.3f} / {sh['hc']:.3f} / {sh['dev']:.3f} | {_pt(cb['obj'])} | "
                      f"a{c0['alpha']} b{c0['beta']} {fl}{casc}{mark}"
                      f" | {_pt(cb['hp'])} / {_pt(cb['hc'])} / {_pt(cb['dev'])} | {_pt(cc['obj'])} | {ci_}/{co_} |")
            k = rb["counter"]
            e0, e1 = (100 * k["errors"][i] / k["chars"] for i in (0, 1))
            cs.append(f"| {tier} | {lt} | shipped | {k['kept'][0]:,} | {e0:.3f}% | – | – |")
            cs.append(f"| {tier} | {lt} | b (in-sample) | {k['kept'][1]:,} | {e1:.3f}% | "
                      f"{_pt(k['kept_share'])} | {_pt(k['cer'])} |")
            k = rc["counter"]
            e1 = 100 * k["errors"][1] / k["chars"]
            cs.append(f"| {tier} | {lt} | c (in-sample {ci_}/{co_}) | {k['kept'][1]:,} | {e1:.3f}% | "
                      f"{_pt(k['kept_share'])} | {_pt(k['cer'])} |")
    return res, "\n".join(md) + "\n\n" + "\n".join(cs) + "\n"


def cmd_summary(a):
    import finder_tighten as FT
    sel = json.loads(Path(a.select).read_text(encoding="utf-8"))
    res, md = summarise(sel)
    res["select_sha256"] = sha256(a.select)
    FT.assert_public(res)
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    if a.md:
        Path(a.md).write_text(md, encoding="utf-8")
    print(json.dumps(res["decisions"], ensure_ascii=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("counter-cache")
    p.add_argument("--blob", required=True)
    p.add_argument("--synth", default=str(SYNTH))
    p.add_argument("--procs", type=int, default=5)
    p.add_argument("--out", required=True)
    p = sub.add_parser("tune")
    p.add_argument("--protocol", required=True)
    p.add_argument("--lm-tier", required=True, choices=("500k", "1000k", "2000k", "4000k"))
    p.add_argument("--rubi", required=True, help="a Rubi_for_vita checkout at the pinned core/textnorm.c")
    p.add_argument("--hval-dir", required=True, help="units.pkl and reads_<T>.<reader stem>.pkl (vocr6_lm hval-cache)")
    p.add_argument("--blob", action="append", required=True, help="T=PATH: tier T's final blob")
    p.add_argument("--hval-blob", action="append", default=[], help="T=PATH: tier T's hval reader (default: --blob)")
    p.add_argument("--dev-cache", action="append", required=True, help="T=FILE: vocr6_lm dev-cache of the final blob")
    p.add_argument("--counter-cache", action="append", required=True, help="T=FILE: counter-cache of the final blob")
    p.add_argument("--synth", default=str(SYNTH))
    p.add_argument("--configs", help="a JSON list of settings instead of arm b's grid (the own-hval report)")
    p.add_argument("--arms", default="b,c,g")
    p.add_argument("--procs", type=int, default=5)
    p.add_argument("--out", required=True)
    p = sub.add_parser("select")
    p.add_argument("--protocol", required=True)
    p.add_argument("--tune", action="append", required=True, help="LMTIER=FILE")
    p.add_argument("--own", action="append", default=[], help="LMTIER=FILE: the final blobs' own hval (report)")
    p.add_argument("--out", required=True)
    p = sub.add_parser("continuity")
    p.add_argument("--cache", required=True)
    p.add_argument("--blob", required=True)
    p.add_argument("--synth", default=str(SYNTH))
    p.add_argument("--lm", action="append", required=True, help="NAME:PATH:ALPHA:BETA:FLAGS")
    p.add_argument("--out", required=True)
    p = sub.add_parser("summary")
    p.add_argument("--select", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--md")
    p = sub.add_parser("gated-cache")
    p.add_argument("--tier", required=True)
    p.add_argument("--blob", required=True, help="a gated blob (w8g / w8g2)")
    p.add_argument("--two-level", choices=("auto", "on", "off"), default="auto")
    p.add_argument("--hval-dir", required=True)
    p.add_argument("--dev-out", required=True)
    p.add_argument("--counter-out", required=True)
    p.add_argument("--synth", default=str(SYNTH))
    p.add_argument("--procs", type=int, default=5)
    p = sub.add_parser("compare-runs")
    p.add_argument("--x", required=True)
    p.add_argument("--y", required=True)
    p.add_argument("--config", required=True, help="the setting (JSON), present in both files' configs")
    p.add_argument("--what", default="")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"counter-cache": cmd_counter_cache, "tune": cmd_tune, "select": cmd_select, "continuity": cmd_continuity,
     "summary": cmd_summary, "gated-cache": cmd_gated_cache, "compare-runs": cmd_compare_runs}[a.cmd](a)


if __name__ == "__main__":
    main()
