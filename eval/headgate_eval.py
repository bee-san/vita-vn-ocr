#!/usr/bin/env python3
"""The head gate's development measurement (eval/headgate_protocol.json "development"; docs/headgate.md).

  headgate_eval.py cache --tier T --base VOCR5_T_w8.vocr --gated VOCR5_T_w8g.vocr --out DIR    (heavy; private)
  headgate_eval.py score --cache DIR --lm vocr_lm_o3_1000k.vlm --out eval/headgate_dev_<date>.json [--synth SY.npz]
  headgate_eval.py a9 --dir A9_DIR --blobs DIR --cache DIR --out eval/headgate_a9_<date>.json --savings SAVINGS.json

cache: every read of VOCR5's development cohorts (eval/vocr5_protocol.json "cohorts") through the C runtime in W8I,
in both arms: "today" (the release blob, head_gate off) and "gated" (the gated blob, head_gate ON), with top-8
extras, the blank logits, each frame's argmax and the gate's scores. tv: teacher_val_dev.npz, dn: devnative's four
*_dev.npz, both views (rv: as stored, the console's polarity; rvn: the inversion undone); wd: tv_dlg and dn_dlg,
every region through the C GRAD finder (flags 0) and with the C refinement (VOCR_REFINE_DEFAULT), each line read
with its dark flag as VOCR_INVERT (eval/finder_tighten.py's path, C == twin checked on every case). On every tenth
read the gated blob with head_gate off must equal today bit for bit, and the gated read must equal
train/headgate.simulate(today, score > t). The cache holds per-frame top-K and texts of development crops: private.

score: per tier and arm, three decodes of every read (the protocol's "decodes"): greedy; rubi (Rubi's line path:
the counter re-decode {6000, 4000}, the beam + 1 MB LM at Rubi's settings, a line read as empty staying empty, then
the counter rule); rubi_cascade (rubi, and a line whose confidence is under 700 read by the next tier up with the
same steps, in the same arm). Components tv / dn / wd and the development score as eval/vocr5_select.py builds
them (dialogue only; names a secondary column), gated - today with its VN-cluster bootstrap, and the protocol's
rule. Beside it: the skip rates, the non-blank frames skipped, lines changed, the LM decoder's candidates per frame
and, with --synth, #109's synthetic counter-context set (report only). The output holds counts, rates and hashes.
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
for _p in (ROOT / "eval", ROOT / "train", ROOT / "synth", ROOT / "runtime", ROOT / "runtime" / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from metrics import lev, norm  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ART = HERD / "shared" / "artifacts"
POOLS = ART / "owner-drive-ingest-2" / "reclaim" / "pools"
COHORTS = ART / "vocr5-train" / "cohorts"
TV = "teacher_val_dev.npz"
DN = ("devnative_tv_native_ink_dev.npz", "devnative_tv_native_grad_dev.npz", "devnative_strict_ink_dev.npz",
      "devnative_strict_grad_dev.npz")
WD = ("tv_dlg", "dn_dlg")
K = 8
LM_OPTS = {"beam": 8, "topk": 8, "alpha": 500, "beta": 3500, "gate": 8000, "flags": 2, "bracket": 8000}
COUNTER = (6000, 4000)            # Rubi's VJO_VOCR_COUNTER_TAU / _OPEN
CASCADE_CONF = 700                # Rubi's VJO_VOCR_CASCADE_CONF
CASCADE = {"S": "N", "N": "L", "L": None}
ARMS = ("today", "gated")
DECODES = ("greedy", "rubi", "rubi_cascade")          # the protocol's rule reads these
REREAD = ("rubi_reread", "rubi_cascade_reread")       # with the counter-context re-read (report: what Rubi ships)
MARGIN = 0.05                     # the rule's non-inferiority margin, points
MIN_SAVING = 0.05                 # the rule's minimum estimated saving, share of the nominal box


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------- the units

def fixed_rows():
    """tv and dn rows: [(cohort, file, row)], row = {id, vn, ref, img, inverted, name, kind, grad}."""
    import synth3_ablation as S
    out = []
    for f in (TV, *DN):
        z = np.load(POOLS / f)
        meta = {m["id"]: m for m in json.loads(str(z["meta"]))}
        for r in S.load_dev(POOLS / f):
            m = meta[r["id"]]
            st = m.get("styles") or []
            if isinstance(st, str):
                import ast
                st = ast.literal_eval(st)
            r["name"] = ("name" in st) if f != TV else ("namelike" in r["tags"])
            r["kind"] = m.get("kind")
            r["grad"] = "_grad_" in f
            out.append(("tv" if f == TV else "dn", f, r))
    return out


def wd_lines(build_dir):
    """{cohort: (cases, {mode: [(boxes, dark, org, rows, found)]})} through the C finder, C == twin checked."""
    import finder_tighten as FT

    import lines_refine as LR
    ident = FT.build(build_dir, "gcc")
    flags = LR.parse_flags("default")
    res = {}
    for name in WD:
        cases = FT.load_cohort(COHORTS, name)
        imgs = [c["image"] for c in cases]
        res[name] = (cases, {"grad": FT.lines_of(ident["driver"], imgs, 0), "refine": FT.lines_of(ident["driver"],
                                                                                                   imgs, flags)})
    return res, ident, flags


def reads(fixed, wd):
    """Every read: (key, crop, invert)."""
    out = []
    for coh, f, r in fixed:
        g = r["img"]
        out.append(((coh, f, r["id"], "rv"), g, False))
        out.append(((coh, f, r["id"], "rvn"), 255 - g if r["inverted"] else g, False))
    for coh, (cases, modes) in wd.items():
        for mode, per in modes.items():
            for ci, (c, (boxes, dark, _, _, _)) in enumerate(zip(cases, per)):
                for li, ((x, y, w, h), d) in enumerate(zip(boxes, dark)):
                    out.append(((coh, mode, ci, li), np.ascontiguousarray(c["image"][y:y + h, x:x + w]), bool(d)))
    return out


# ---------------------------------------------------------------- cache

_W = {}


def _init(base, gated):
    from pyvocr import Vocr

    import headgate as HG
    _W["today"] = Vocr(base, mode="w8i", max_width=1024)
    _W["gated"] = Vocr(gated, mode="w8i", max_width=1024, head_gate="on")
    _W["off"] = Vocr(gated, mode="w8i", max_width=1024, head_gate="off")
    _W["vocab"] = [_W["today"].class_text(c) for c in range(_W["today"].n_classes)]
    _W["t"] = HG.read_gate(Path(gated).read_bytes())["t"]


def _slim(o):
    return {k: o[k] for k in ("text", "topv", "topc", "blank", "best", "conf_min", "conf_mean", "frames", "chars",
                              "gated")} | ({"gate_score": o["gate_score"]} if "gate_score" in o else {})


def _same(a, b):
    return (a["text"] == b["text"] and np.array_equal(a["topv"].view(np.uint32), b["topv"].view(np.uint32))
            and np.array_equal(a["topc"], b["topc"]) and np.array_equal(a["best"], b["best"])
            and np.array_equal(a["blank"].view(np.uint32), b["blank"].view(np.uint32))
            and np.float32(a["conf_min"]) == np.float32(b["conf_min"])
            and np.float32(a["conf_mean"]) == np.float32(b["conf_mean"]) and a["chars"] == b["chars"])


def _work(job):
    import headgate as HG
    j, img, inv, check = job
    a = _W["today"].run_topk(img, K, invert=inv)
    g = _W["gated"].run_topk(img, K, invert=inv, gate_score=True)
    chk = None
    if check:
        o = _W["off"].run_topk(img, K, invert=inv)
        skip = g["gate_score"] > _W["t"]
        sim = HG.simulate(a, skip, _W["vocab"])
        chk = (_same(a, o), _same(sim, g) and sim["gated"] == g["gated"])
    return j, _slim(a), _slim(g), chk


def cmd_cache(a):
    t0 = time.time()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    fixed = fixed_rows()
    wd, ident, flags = wd_lines(out / "build")
    rd = reads(fixed, wd)
    jobs = [(j, img, inv, j % 10 == 0) for j, (_, img, inv) in enumerate(rd)]
    res = [None] * len(rd)
    checks = {"off_equals_today": 0, "gated_equals_simulation": 0, "checked": 0}
    with mp.get_context("fork").Pool(a.procs, initializer=_init, initargs=(a.base, a.gated)) as pool:
        for n, (j, x, g, chk) in enumerate(pool.imap_unordered(_work, jobs, chunksize=8)):
            res[j] = {"today": x, "gated": g}
            if chk is not None:
                checks["checked"] += 1
                checks["off_equals_today"] += bool(chk[0])
                checks["gated_equals_simulation"] += bool(chk[1])
            if n % 2000 == 0:
                print(f"  {a.tier}: {n}/{len(jobs)}", flush=True)
    if checks["off_equals_today"] != checks["checked"] or checks["gated_equals_simulation"] != checks["checked"]:
        raise SystemExit(f"parity failed: {checks}")
    meta = {"tier": a.tier, "base": Path(a.base).name, "base_sha256": sha256(a.base), "gated": Path(a.gated).name,
            "gated_sha256": sha256(a.gated), "reads": len(rd), "checks": checks, "native": ident, "flags": flags,
            "pools": {f: sha256(POOLS / f) for f in (TV, *DN)},
            "cohorts": {c: sha256(COHORTS / f"{c}.pkl") for c in WD}, "seconds": round(time.time() - t0, 1),
            "created": utc_now()}
    with open(out / f"cache_{a.tier}.pkl", "wb") as fh:
        pickle.dump({"meta": meta, "keys": [k for k, _, _ in rd], "res": res}, fh, protocol=4)
    (out / f"cache_{a.tier}.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(json.dumps({k: meta[k] for k in ("tier", "reads", "checks", "seconds")}), flush=True)


# ---------------------------------------------------------------- decodes

def line_conf(o):
    """Rubi's core/vocr_ocr.c line_conf: conf_min x 1000 + 0.5 in float, truncated; 1000 for an empty line."""
    if not o["text"]:
        return 1000
    c = np.float32(np.float32(o["conf_min"]) * np.float32(1000.0) + np.float32(0.5))
    return 0 if c <= 0 else 1000 if c >= 1000 else int(c)


class Decoder:
    """One tier's C decoders (the release blob's model: the class texts are the gated blob's too)."""

    def __init__(self, blob, lm):
        from pyvocr import Vocr, VocrLM
        self.rt = Vocr(blob, mode="w8i", max_width=1024)
        self.lm = VocrLM(self.rt, lm, **LM_OPTS)

    def rubi(self, o, ungated=None):
        """Rubi's finish_line on one read's extras, then its text rule (core/textnorm.c's mirror of it). ungated:
        the same line read without skipping, which replaces a gated read whose re-decode finds a counter context
        (the re-read of the protocol's "counter_check"); returns (text, the read used)."""
        from pyvocr import post_counters
        if not o["text"]:
            return "", o
        res, st = self.rt.redecode(o, *COUNTER)
        if ungated is not None and st["contexts"]:
            o = ungated
            if not o["text"]:
                return "", o
            res, st = self.rt.redecode(o, *COUNTER)
        src = res if st["edits"] else o
        t = self.lm.decode(src["topv"], src["topc"], src["blank"])
        return post_counters(t, self.rt.lib)[0], o


def lm_candidates(o, gate_q10=8192):
    """Per read: (frames, blank-only frames, non-blank candidates) of vocr_lm_decode's candidate rule at Rubi's
    gate (a class within the gate of the frame's top-1; the blank also from blank_logit)."""
    tv, tc, bl = o["topv"].astype(np.float32), o["topc"], o["blank"].astype(np.float32)
    d = np.clip(tv[:, :1] - tv, 0, 64).astype(np.float32)
    q = np.floor((d * np.float32(1024) + np.float32(0.5)).astype(np.float32)).astype(np.int64)
    q[:, 0] = 0
    cand = q < gate_q10
    cand &= np.cumprod(cand, axis=1).astype(bool)        # the loop stops at the first one outside the gate
    has_blank = (cand & (tc == 0)).any(1)
    db = np.clip(tv[:, 0] - bl, 0, 64).astype(np.float32)
    qb = np.floor((db * np.float32(1024) + np.float32(0.5)).astype(np.float32)).astype(np.int64)
    blank_extra = ~has_blank & (qb < gate_q10)
    nonblank = (cand & (tc != 0)).sum(1)
    blank_only = (nonblank == 0) & (has_blank | blank_extra)
    return len(tv), int(blank_only.sum()), int(nonblank.sum())


# ---------------------------------------------------------------- score

def load_cache(d, tier):
    with open(Path(d) / f"cache_{tier}.pkl", "rb") as fh:
        c = pickle.load(fh)
    return c


def comps_for(fixed, wdc, texts):
    """vocr5_select Components (tv, dn, wd) from {model: {key: normalised text}}."""
    import vocr5_select as V

    from general_vn_textbox_guard import transcript_score
    out = {n: V.Component(n) for n in ("tv", "dn", "wd")}
    tv_shots = {c["shot"] for c in wdc["tv_dlg"]}
    for model, tx in texts.items():
        for coh in ("tv", "dn"):
            rows = [(f, r) for c, f, r in fixed if c == coh]
            rows.sort(key=lambda fr: fr[0])        # vocr5_select.load_fixed pools the files in sorted order
            e = [sum(lev(tx[(coh, f, r["id"], v)], r["ref"]) for v in ("rv", "rvn")) for f, r in rows]
            out[coh].add(model, e, [2 * len(r["ref"]) for _, r in rows], [r["vn"] for _, r in rows],
                         [r["name"] for _, r in rows], {})
        rows = []
        for coh in WD:
            for mode in ("grad", "refine"):
                for ci, case in enumerate(wdc[coh]):
                    if coh == "dn_dlg" and case["shot"] in tv_shots:
                        continue
                    preds = tx.get(("wdcase", coh, mode, ci), [])
                    sc = transcript_score(case["lines"], preds)["all"]["full"]
                    rows.append((f"{coh}|{mode}|{case['unit']}|{case['variant']}", sc["errors"], sc["chars"],
                                 case["vn"]))
        rows.sort()
        out["wd"].add(model, [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows],
                      np.zeros(len(rows), bool), {})
    return out


def rule(cmp, saving_share):
    """The protocol's per-decode conditions: (a) not resolved worse, (b) margin, (c) saving."""
    a = not cmp["score"]["ci95"][0] > 0 and all(not cmp[c]["ci95"][0] > 0 for c in ("tv", "dn", "wd"))
    b = cmp["score"]["ci95"][1] <= MARGIN
    c = saving_share is not None and saving_share >= MIN_SAVING
    return {"not_resolved_worse": bool(a), "upper_within_margin": bool(b), "saving_at_least_5pct": bool(c)}


def cmd_counter(a):
    """Only the synthetic counter check, merged into an existing development report (--out)."""
    import finder_tighten as FT
    blobs = dict(s.partition("=")[::2] for s in a.blob)
    report = json.loads(Path(a.out).read_text(encoding="utf-8"))
    report["counter_check"] = counter_check(a, blobs)
    FT.assert_public(report)
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


def cmd_score(a):
    import finder_tighten as FT
    import vocr5_select as V
    if a.counter_only:
        return cmd_counter(a)
    t0 = time.time()
    fixed = fixed_rows()
    wdc = {n: FT.load_cohort(COHORTS, n) for n in WD}
    caches = {t: load_cache(a.cache, t) for t in ("S", "N", "L")}
    keys = caches["S"]["keys"]
    for t, c in caches.items():
        if c["keys"] != keys:
            raise SystemExit(f"{t}: the cache holds other reads")
    blobs = dict(s.partition("=")[::2] for s in a.blob)
    savings = json.loads(Path(a.savings).read_text()) if a.savings else {}
    texts = {}            # (tier, arm, decode) -> {key: text}
    confs = {}            # (tier, arm) -> {key: conf}
    rereads = {}
    for tier in ("S", "N", "L"):
        dec = Decoder(blobs[tier], a.lm)
        c = caches[tier]
        for arm in ARMS:
            g, r, cf, rr, cfr = {}, {}, {}, {}, {}
            n_re = 0
            for k, res in zip(keys, c["res"]):
                o = res[arm]
                g[k] = o["text"]
                r[k], _ = dec.rubi(o)
                cf[k] = line_conf(o)
                if arm == "gated":           # the re-read: today's read is the ungated read, bit for bit
                    rr[k], used = dec.rubi(o, ungated=res["today"])
                    n_re += used is not o
                    cfr[k] = line_conf(used)
                else:
                    rr[k], cfr[k] = r[k], cf[k]
            texts[(tier, arm, "greedy")], texts[(tier, arm, "rubi")], confs[(tier, arm)] = g, r, cf
            texts[(tier, arm, "rubi_reread")], confs[(tier, arm, "reread")] = rr, cfr
            if arm == "gated":
                rereads[tier] = n_re
        print(f"{tier}: decoded ({rereads[tier]} gated reads re-read for a counter context)", flush=True)
    for tier in ("S", "N", "L"):
        for arm in ARMS:
            nxt = CASCADE[tier]
            for d, cd, ck in (("rubi", "rubi_cascade", (tier, arm)), ("rubi_reread", "rubi_cascade_reread",
                                                                      (tier, arm, "reread"))):
                base = texts[(tier, arm, d)]
                if nxt is None:
                    texts[(tier, arm, cd)] = dict(base)
                else:
                    up = texts[(nxt, arm, d)]
                    cf = confs[ck]
                    texts[(tier, arm, cd)] = {k: (up[k] if cf[k] < CASCADE_CONF else v) for k, v in base.items()}
    report = {"kind": "headgate-dev", "schema": 1, "created": utc_now(),
              "protocol_sha256": sha256(ROOT / "eval" / "headgate_protocol.json"),
              "lm": {"file": Path(a.lm).name, "sha256": sha256(a.lm), **LM_OPTS, "counter": COUNTER,
                     "cascade_conf": CASCADE_CONF, "cascade": CASCADE},
              "caches": {t: c["meta"] for t, c in caches.items()}, "tiers": {}}
    for meta in report["caches"].values():
        meta.pop("native", None)
    for tier in ("S", "N", "L"):
        res = caches[tier]["res"]
        tier_rep = {"decodes": {}, "skip": skip_stats(keys, res, fixed, wdc), "lm_candidates": {}}
        for arm in ARMS:
            n_f = n_bo = n_nb = 0
            for k, rr in zip(keys, res):
                f_, bo, nb = lm_candidates(rr[arm])
                n_f, n_bo, n_nb = n_f + f_, n_bo + bo, n_nb + nb
            tier_rep["lm_candidates"][arm] = {"frames": n_f, "blank_only_share": round(n_bo / n_f, 5),
                                              "nonblank_candidates_per_frame": round(n_nb / n_f, 4)}
        tier_rep["reread_reads"] = rereads[tier]
        for d in DECODES + REREAD:
            tx = {}
            for arm in ARMS:
                t = texts[(tier, arm, d)]
                m = {k: norm(v) for k, v in t.items() if k[0] in ("tv", "dn")}
                for k, v in t.items():
                    if k[0] in WD:
                        m.setdefault(("wdcase", k[0], k[1], k[2]), []).append(norm(v))
                tx[arm] = m
            comps = comps_for(fixed, wdc, tx)
            cmp = V.compare(comps, ["today"], ["gated"])
            changed = {"fixed_lines": sum(texts[(tier, "today", d)][k] != texts[(tier, "gated", d)][k]
                                          for k in keys if k[0] in ("tv", "dn")),
                       "wd_lines": sum(texts[(tier, "today", d)][k] != texts[(tier, "gated", d)][k]
                                       for k in keys if k[0] in WD)}
            names = {arm: {c: round(100 * comps[c].rate([arm], "name"), 4) for c in ("tv", "dn")} for arm in ARMS}
            sv = savings.get(tier, {}).get("nominal_share")
            entry = {"compare": cmp, "score_today": round(V.score_of(comps, "today"), 4),
                     "score_gated": round(V.score_of(comps, "gated"), 4), "changed": changed, "names_cer_pct": names,
                     "rule": rule(cmp, sv)}
            entry["verdict"] = verdict(entry["rule"])
            tier_rep["decodes"][d] = entry
            print(tier, d, json.dumps({"score": cmp["score"], "tv": cmp["tv"]["delta_points"],
                                       "dn": cmp["dn"]["delta_points"], "wd": cmp["wd"]["delta_points"],
                                       "changed": changed}), flush=True)
        vs = [tier_rep["decodes"][d]["verdict"] for d in DECODES]       # the pre-registered rule: these three
        tier_rep["verdict"] = ("adopted" if all(v == "adopted" for v in vs) else
                               "rejected" if any(v == "rejected" for v in vs) else "behind_the_switch")
        report["tiers"][tier] = tier_rep
    if a.synth:
        report["counter_check"] = counter_check(a, blobs)
    report["seconds"] = round(time.time() - t0, 1)
    FT.assert_public(report)
    Path(a.out).write_text(json.dumps(report, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({t: report["tiers"][t]["verdict"] for t in report["tiers"]}))


def verdict(r):
    if not r["not_resolved_worse"]:
        return "rejected"
    return "adopted" if r["upper_within_margin"] and r["saving_at_least_5pct"] else "behind_the_switch"


def skip_stats(keys, res, fixed, wdc):
    """Shares of frames the gate skipped, over every read and by part; non-blank argmax frames skipped."""
    info = {(c, f, r["id"]): r for c, f, r in fixed}
    groups = {}

    def add(g, fr, sk, nb_sk, nb):
        e = groups.setdefault(g, [0, 0, 0, 0])
        e[0] += fr
        e[1] += sk
        e[2] += nb_sk
        e[3] += nb

    for k, rr in zip(keys, res):
        t, g = rr["today"], rr["gated"]
        skip = g["topv"][:, 1] == -np.inf if g["topv"].shape[1] > 1 else np.zeros(len(g["best"]), bool)
        if int(skip.sum()) != g["gated"]:
            raise SystemExit(f"{k}: skipped frames {int(skip.sum())} != gated {g['gated']}")
        nb = t["best"] != 0
        vals = (len(skip), int(skip.sum()), int((skip & nb).sum()), int(nb.sum()))
        parts = ["all"]
        if k[0] in ("tv", "dn"):
            r = info[k[:3]]
            plat = {"vita": "vita", "psp": "psp", "hires": "pc"}.get(r["kind"], str(r["kind"]))
            parts += [f"cohort:{k[0]}", f"view:{k[3]}", "names" if r["name"] else "dialogue"]
            if not r["name"]:
                parts += [f"dialogue|platform:{plat}"]
                if r["grad"]:
                    parts += ["dialogue|dn_grad_crops"]
        else:
            case = wdc[k[0]][k[2]]
            plat = {"vita": "vita", "psp": "psp", "hires": "pc"}.get(case.get("kind"), str(case.get("kind")))
            parts += ["dialogue", "wd_finder_lines", f"wd_finder_lines|{k[1]}", f"wd_finder_lines|platform:{plat}"]
        for p in parts:
            add(p, *vals)
    return {g: {"frames": v[0], "skipped": v[1], "skip_share": round(v[1] / v[0], 5) if v[0] else None,
                "nonblank_frames": v[3], "nonblank_skipped": v[2],
                "nonblank_skipped_share": round(v[2] / v[3], 6) if v[3] else None} for g, v in sorted(groups.items())}


def counter_check(a, blobs):
    """#109's synthetic counter-context set (eval/kanum_dev.py render; private renders of development pool texts),
    as stored: CER and lines changed, today vs gated, greedy and rubi. Report only."""
    import kanum_dev as KD
    from pyvocr import Vocr
    rows = KD.sy_rows(a.synth)
    out = {"file_sha256": sha256(a.synth), "lines": len(rows), "tiers": {}}
    for tier in ("S", "N", "L"):
        dec = Decoder(blobs[tier], a.lm)
        gated = Vocr(a.gated_blob[tier], mode="w8i", max_width=1024, head_gate="on")
        rep = {}
        tx = {arm: {"greedy": [], "rubi": [], "rubi_reread": []} for arm in ARMS}
        n_re = 0
        for r in rows:
            o0 = dec.rt.run_topk(r["img"], K)
            og = gated.run_topk(r["img"], K)
            for arm, o in (("today", o0), ("gated", og)):
                tx[arm]["greedy"].append(norm(o["text"]))
                tx[arm]["rubi"].append(norm(dec.rubi(o)[0]))
            tx["today"]["rubi_reread"].append(tx["today"]["rubi"][-1])
            t_re, used = dec.rubi(og, ungated=gated.run_topk(r["img"], K, ungated=True))
            tx["gated"]["rubi_reread"].append(norm(t_re))
            n_re += used is not og
        rep["reread_lines"] = n_re
        for d in ("greedy", "rubi", "rubi_reread"):
            e = {arm: [lev(p, r["ref"]) for p, r in zip(tx[arm][d], rows)] for arm in ARMS}

            def tally(keep, d=d, e=e):
                idx = [i for i, r in enumerate(rows) if keep(r)]
                n = sum(len(rows[i]["ref"]) for i in idx)
                return {"lines": len(idx), "cer_today_pct": round(100 * sum(e["today"][i] for i in idx) / n, 4),
                        "cer_gated_pct": round(100 * sum(e["gated"][i] for i in idx) / n, 4),
                        "lines_changed": sum(tx["today"][d][i] != tx["gated"][d][i] for i in idx),
                        "lines_worse": sum(e["gated"][i] > e["today"][i] for i in idx),
                        "lines_better": sum(e["gated"][i] < e["today"][i] for i in idx)}
            rep[d] = tally(lambda r: True)
            rep[d]["by_cut"] = {str(c): tally(lambda r, c=c: r["cut"] == c) for c in KD.SY_CUTS}
        out["tiers"][tier] = rep
        print("counter", tier, json.dumps(rep), flush=True)
    return out


def cmd_a9(a):
    """The protocol's "estimates": vita/a9_layers.py measurements (eval/reproducibility/headgate-20261023/a9.sh) of
    each tier's gated blob on ocr-speed's 24 lines: main's runtime (the baseline, the same blob: main ignores the
    section), this runtime with the gate off, on and score-only; with the PLD model (pld1) and without (pld0, the
    safe basis). Per phase: the high bound x 1.000676 (VOCR3-H15's console rate) at 333 MHz. The saving at a skip
    share s: head(off) - head(on at s) - gate(s), with head(on at s) = head(off) x (1 - s) x kappa (kappa: the
    measured batch efficiency on the pack) and gate(s) = gate(score) + c x s (c: the measured cost of a skipped
    frame's blank logit); s = the pack's own share (measured) and the development share on console-cut lines (wd's
    finder lines, from the dev cache). Every millisecond is an estimate."""
    sys.path.insert(0, str(ROOT / "vita"))
    import a9_layers as AL
    from pyvocr import Vocr

    import export
    scale, mhz = 1.000676, 333
    A9 = Path(a.dir)
    pack = A9 / "pack24.bin"
    src = HERD / "shared" / "artifacts" / "ocr-speed" / "packs" / "dev_res.bin"
    if not pack.exists():
        AL.subset_pack(src, pack, 60, 100, 24)
    lines = read_pack(pack)
    out = {"kind": "headgate-a9", "schema": 1, "created": utc_now(), "scale": scale, "mhz": mhz,
           "pack": {"source": src.name, "sha256": sha256(pack), "lines": len(lines)},
           "box": {"nominal_frames": 240, "finder_ms": 100.0, "real_frames": {"vita": 89, "psp": 93, "adrenaline": 95}},
           "tiers": {}}
    savings = {}
    for tier in ("S", "N", "L"):
        blob = Path(a.blobs) / f"VOCR5_{tier}_w8g.vocr"
        head_id = len(export.read_blob(blob.read_bytes())["ops"]) - 1
        rt = Vocr(blob, mode="w8i", max_width=1024, head_gate="on")
        fr = sk = 0
        for g in lines:
            o = rt.run_topk(g, K)
            fr += o["frames"]
            sk += o["gated"]
        s_pack = sk / fr
        c = load_cache(a.cache, tier)
        wf = ws = 0
        for k, rr in zip(c["keys"], c["res"]):
            if k[0] in WD:
                wf += rr["gated"]["frames"]
                ws += rr["gated"]["gated"]
        s_dev = ws / wf
        tr = {"blob": blob.name, "blob_sha256": sha256(blob), "pack_frames": fr, "pack_skip_share": round(s_pack, 5),
              "dev_wd_skip_share": round(s_dev, 5), "runs": {}}
        for pld in (0, 1):
            ph = {}
            texts = {}
            for cfg in ("main", "off", "score", "on"):
                d = A9 / f"{tier}_{cfg}_pld{pld}"
                meta, frames, res = AL.analyse(d, mhz)
                if frames != fr:
                    raise SystemExit(f"{d}: {frames} frames, the pack has {fr}")
                ms = {i: r["high"] * scale / frames / (mhz * 1000) for i, r in res.items()}
                ph[cfg] = {"ms_per_frame": sum(ms.values()), "head_ms": ms.get(head_id, 0.0),
                           "gate_ms": ms.get(AL.PROF_GATE, 0.0),
                           "insns_per_frame": sum(r["insns"] for r in res.values()) / frames,
                           "head_insns_per_frame": res[head_id]["insns"] / frames,
                           "gate_insns_per_frame": res.get(AL.PROF_GATE, {}).get("insns", 0) / frames,
                           "other_ms": sum(v for i, v in ms.items() if i not in (head_id, AL.PROF_GATE))}
                texts[cfg] = meta["runs"]["prof"]["texts_sha256"]
            if not (texts["main"] == texts["off"] == texts["score"]):
                raise SystemExit(f"{tier} pld{pld}: main / off / score texts differ")
            h0, h1 = ph["off"]["head_ms"], ph["on"]["head_ms"]
            g0, g1 = ph["score"]["gate_ms"], ph["on"]["gate_ms"]
            kappa = h1 / (h0 * (1 - s_pack))
            cb = (g1 - g0) / s_pack
            save = {}
            for name, s in (("pack", s_pack), ("dev_wd", s_dev)):
                d_ms = h0 - h0 * (1 - s) * kappa - (g0 + cb * s)
                box0 = 240 * ph["off"]["ms_per_frame"] + 100.0
                save[name] = {"skip_share": round(s, 5), "ms_per_frame": round(d_ms, 4),
                              "nominal_box_ms": round(240 * d_ms, 1),
                              "nominal_box_ms_444": round(240 * d_ms * 333 / 444, 1),
                              "nominal_share": round(240 * d_ms / box0, 5),
                              "real_box_ms": {p: round(n * d_ms, 1) for p, n in out["box"]["real_frames"].items()}}
            measured = ph["off"]["ms_per_frame"] - ph["on"]["ms_per_frame"]
            tr["runs"][f"pld{pld}"] = {
                "phases": {cfg: {k: round(v, 4) for k, v in p.items()} for cfg, p in ph.items()},
                "texts_main_off_score_equal": True, "kappa": round(kappa, 4),
                "blank_logit_ms_per_skipped": round(cb, 5),
                "measured_saving_ms_per_frame": round(measured, 4),
                "off_vs_main_ms_per_frame": round(ph["off"]["ms_per_frame"] - ph["main"]["ms_per_frame"], 4),
                "box_ms_off": round(240 * ph["off"]["ms_per_frame"] + 100.0, 1),
                "box_ms_on_pack": round(240 * ph["on"]["ms_per_frame"] + 100.0, 1), "saving": save}
        savings[tier] = {"nominal_share": tr["runs"]["pld0"]["saving"]["dev_wd"]["nominal_share"],
                         "basis": "pld0 (no prefetch credit), dev wd skip share"}
        out["tiers"][tier] = tr
        print(tier, json.dumps({p: {"off": r["phases"]["off"]["ms_per_frame"], "on": r["phases"]["on"]["ms_per_frame"],
                                    "dev": r["saving"]["dev_wd"]} for p, r in tr["runs"].items()}), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    Path(a.savings).write_text(json.dumps(savings, indent=1) + "\n")


def read_pack(path):
    """A vocr_cli pack's crops (u32 count; per crop u32 h, w, channels, then the bytes): grey only."""
    import struct
    data = Path(path).read_bytes()
    n, off, out = struct.unpack_from("<I", data, 0)[0], 4, []
    for _ in range(n):
        h, w, ch = struct.unpack_from("<III", data, off)
        off += 12
        px = np.frombuffer(data, np.uint8, h * w * ch, off).reshape(h, w, ch)
        off += h * w * ch
        out.append(np.ascontiguousarray(px[:, :, 0] if ch == 1 else px))       # grey, or RGBA as vocr_cli reads it
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("a9")
    p.add_argument("--dir", required=True, help="the measurement directories (TIER_CONFIG_pldP)")
    p.add_argument("--blobs", required=True, help="the gated blobs' directory")
    p.add_argument("--cache", required=True, help="the dev caches (the development skip share)")
    p.add_argument("--out", required=True)
    p.add_argument("--savings", required=True)
    p = sub.add_parser("cache")
    p.add_argument("--tier", required=True)
    p.add_argument("--base", required=True)
    p.add_argument("--gated", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--procs", type=int, default=5)
    p = sub.add_parser("score")
    p.add_argument("--cache", required=True)
    p.add_argument("--blob", action="append", required=True, help="TIER=release blob (decoders)")
    p.add_argument("--gated-blob", action="append", default=[], help="TIER=gated blob (counter check)")
    p.add_argument("--lm", required=True)
    p.add_argument("--savings", help="JSON {tier: {nominal_share}} from the A9 estimates (rule (c))")
    p.add_argument("--synth", help="eval/kanum_dev.py's synthetic counter set (report only)")
    p.add_argument("--counter-only", action="store_true", help="only the counter check, merged into --out")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "score":
        a.gated_blob = dict(s.partition("=")[::2] for s in a.gated_blob)
    {"cache": cmd_cache, "score": cmd_score, "a9": cmd_a9}[a.cmd](a)


if __name__ == "__main__":
    main()
