#!/usr/bin/env python3
"""VOCR6 stage 4 (#125): the head gate (#120) and the two-level head (#122) fit on a VOCR6 blob, their rules, the VOCR6
decision path and the console estimates (eval/vocr6_gate_protocol.json; docs/headgate.md, docs/head2level.md).

  vocr6_gate.py init --out OUT [options]                             the run's configuration and the development lines
                                                                      (before tiers are prepared in parallel)
  vocr6_gate.py run --tier T --blob B [--next-blob NB] --out OUT     the tier (its next tier first when needed), decided
  vocr6_gate.py prepare --tier T --blob B --out OUT                   fits, blobs, development reads, VOCR6-path
                                                                      dumps, A9
  vocr6_gate.py decide --tier T --out OUT                             the rules, the VOCR6 path, the fallbacks
  vocr6_gate.py result --out OUT --result RESULT.json                 every tier's decision, numbers and estimates
  vocr6_gate.py check-vocr5 --out OUT --record REC.json               step 2: a run on the v0.3.0 blobs against
                                                                      #120's / #122's records
  vocr6_gate.py timing --out OUT                                      each step's wall clock and slot-hours

One tier, in the protocol's order (Large, Normal, Small: a tier's VOCR6-path arm cascades into the next tier's decided
configuration; its frozen cascades into the next tier's arm of the same kind):
  gate      train/headgate.py features + fit on the tier's blob: VOCG sections at eps 1e-3 (and the 1e-4 fallback)
  2L        (N, L) train/head2level.py permute (#122's ranking), eval/head2level_extract.py train, head2level_fit.py
            escapes (K_alt), build (r96, recall 0.999 / 0.9999), compose (with the tier's gate), retarget (t = +-inf)
  reads     every development crop of VOCR5's cohorts (head2level_dev.all_crops) through each arm in the C runtime (W8I,
            top-8): greedy, Rubi's line confidence, the counter re-decode + beam + the 1 MB LM, the counter rule; #120's
            parity checks on every tenth gated read, #122's exact twin on every eighth crop of a two-level blob
  rules     #120's and #122's development rules as frozen (eval/vocr5_select.py's components and bootstrap), without
            the VNs of the protocol's development.exclude_vns
  path      vocr6-lead's kit (eval/vocr6_score.py read / score, eval/vocr6_select.py compare at feat/vocr6-t1-stages
            8e4c261): hval plan v2 and #119's dev plan, the shipped LM, Rubi's cascade; Y (the arm) against X (the
            ungated blob) not resolved worse on hval_page and dev
  a9        vita/a9_layers.py measure + report on ocr-speed's 24 development lines (sub24.bin), with and without the
            PLD model; every millisecond is an estimate (x 1.000676 to VOCR3-H15's measured console rate, 333 MHz)

Steps are idempotent: each writes OUT/<T>/steps/<step>.json with its wall clock and slot-hours (the driver runs in
one heavy.sh slot). Features, development reads and VOCR6-path dumps hold owner-derived frames or texts and stay under
--priv; OUT holds blobs, fit records, scores (numbers only) and A9 measurements.
"""
import argparse
import datetime
import hashlib
import json
import os
import pickle
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for _p in (HERE, ROOT / "train", ROOT / "runtime", ROOT / "runtime" / "tests", ROOT / "synth", ROOT / "vita"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ART = HERD / "shared" / "artifacts"
PRIVATE = HERD / "shared" / "private"
PROTOCOL = ROOT / "eval" / "vocr6_gate_protocol.json"
PY = sys.executable

TIERS = ("S", "N", "L")
NEXT = {"S": "N", "N": "L", "L": None}
ORDER = ("L", "N", "S")
TWO_LEVEL_TIERS = ("N", "L")
EXCLUDE_VNS = ("v12188", "v18432")
TEST_TITLE = "PCSG00807"
DECODES = ("greedy", "rubi", "rubi_cascade")
REREAD = ("rubi_reread", "rubi_cascade_reread")          # report only: with Rubi's re-read without skipping

RANK = ART / "head-2level" / "rank.json"
RANK_SHA = "3bc6732d9e75d756b85228177476ce88ae950ec561c1110a667c899ecc699d10"
DEV_LM = ART / "vocr5-train" / "release" / "v0.3.0" / "vocr_lm_o3_1000k.vlm"
DEV_LM_SHA = "993000611555171daf6361e29a503b0871a5f3dd240a19555d55f16aaa3ab3d9"
RUBI_LM = dict(beam=8, topk=8, alpha=500, beta=3500, gate=8000, flags=2, bracket=8000)
COUNTER = (6000, 4000)
CASCADE_CONF = 700
TOPK = 8
REREAD_RE = re.compile("[カケ][月所年国条]")        # Rubi core/vocr_ocr.c kana_counter

PACK = ART / "ocr-speed-resume" / "packs" / "sub24.bin"
PACK_SHA = "3162151aaf3013f434c5ad7996d04ca2308fe2d3896ce06de14909878906c8b7"
SPEED_ENV = ART / "ocr-speed" / "env.sh"
SCALE, MHZ, MHZ_HI = 1.000676, 333, 444
NOMINAL = {"lines": 3, "frames_per_line": 80, "finder_ms": 100.0}
NOMINAL_FRAMES = NOMINAL["lines"] * NOMINAL["frames_per_line"]
REAL_FRAMES = 89
PROF_GATE, PROF_ATTN = 102, 103
GATE_ESC_MACS = 96 * 16 + 2 * 16 + 16                 # head2level_fit.ESC_MACS_ROWS x 96

KIT = Path(os.environ.get("VOCR6_GATE_KIT", "/local/home/skerraut/herd-synth3/clones/vocr6-final-gate-kit"))
KIT_PINS = {"eval/vocr6_score.py": "808b49eedb97446ea76867a7beb2d2f14609680b0f19aa7eabdd7ad4afd7c3bb",
            "eval/vocr6_select.py": "db06fafa167332c76cc368b3ed7cf5d0145ee90ffd46b523d787d3feea74ec54"}
PATH_PLANS = {
    "hval": (ART / "vocr6-lead" / "plans" / "hval_plan_v2.pkl",
             "3a6f80e1dfe4e4f5cf61d2b56aef4253e2039939a76231d344e17a4d351d97ba"),
    "dev": (ART / "vocr5-analysis" / "work" / "plan.pkl",
            "12938dfa38cd8f9b22a5607e469d061bd4b87bfab4deca1ab55e78116e485ef3")}
DEV_UNITS = (ART / "vocr5-analysis" / "work" / "units.pkl",
             "73384ae1020608ebbdf45440d8855f8dc793f5068ef05981b71aef75b031cc7b")
RUBI_NORM = (ART / "vocr6-lead" / "tools" / "rubi_norm",
             "12c2c60ec17d343b8a8ba5be822ca252459018cc5024b91a6cdd332d1e618ae0")
# the LM per plan (protocol amendment 1 = vocr6-lead's stages.4 amendment 3): hval with the screening build (the
# shipped LM holds hval's text), dev with the shipped 1 MB LM; Rubi's settings of v0.3.7-lm6.1
PATH_LM_HVAL = PRIVATE / "vocr6-lm" / "lm" / "h64_1000k.vlm"
PATH_LM_DEV = ART / "vocr6-lm" / "release" / "v0.3.7-lm6.1" / "vocr_lm6_o3_1000k.vlm"
PATH_LM_SHA = {"hval": "b104bd1904d8f80e7e191e01ecb7decaf9fd81dacccc4d79e4040182b82727ca",
               "dev": "41591accf8499f1c34557e585cca382de2c4b5ca15f270c985612c32487e251f"}
PATH_LM_OPTS = "alpha=600,beta=2500,flags=1"
# vocr6-lead's stage-4 plans (feat/vocr6-t1-stages 2dc491e): the same comparison, report only
PLANS4 = {
    "hval4": (ART / "vocr6-lead" / "plans" / "hval_plan_v3.pkl",
              "17b054496a612df9b152a16309e7e21237ba86d27f566e7d1980bdde945d063a"),
    "dev4": (ART / "vocr6-lead" / "plans" / "dev_plan_v2.pkl",
             "abb37391c4086f690e11b0d0021dbc498cfeb004795379ffabd7d0c58e173e6e")}

# the released v0.3.0 blobs: a tier whose final blob is one keeps #120's / #122's verdict (protocol models.vocr5_final)
VOCR5_RELEASED = {"f175bfcb8cec65d69e85dd5f2cf707c9a711ada7203c0d1099cd2d0142978b8d": ("S", "gate", None),
                  "20be179def8033171435ba62388657b3d3c932b365093737948c636e025e3bdc": ("N", "gate", "2Lg"),
                  "50413b37dbd85b2acb0908967bdeaf7ca67803850e0627a7b7cc3d274edc0ba3": ("L", "gate", "2Lg")}
# the committed records of #120 and #122 the VOCR5 run is checked against (step 2)
REF = {"gate_release": ART / "head-gate" / "release", "gate_blobs": ART / "head-gate" / "blobs",
       "two_blobs": ART / "head-2level" / "blobs", "two_release": ART / "head-2level" / "release",
       "gate_fit": ROOT / "eval" / "headgate_fit_20261023.json",
       "gate_dev": ROOT / "eval" / "headgate_dev_20261023.json",
       "gate_a9": ROOT / "eval" / "headgate_a9_20261023.json",
       "two_fit": ROOT / "eval" / "head2level_fit_20261023.json",
       "two_dev": ROOT / "eval" / "head2level_dev_20261023.json"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def rnd(x, n=4):
    return None if x is None else round(float(x), n)


# ---------------------------------------------------------------- arms (pure)

def arm_parts(arm):
    """An arm's configuration: today (the tier's blob), gate / gate_e4 (the head gate at eps 1e-3 / 1e-4), 2L and
    2Lg[_e4] (the two-level head at K_alt with the head's own rare rows, alone / with the gate) [_rc9999: the escape
    at recall 0.9999]. -> {"two", "gate" (None, "e3", "e4"), "rc" (None, "r0.999", "r0.9999")}."""
    if arm == "today":
        return {"two": False, "gate": None, "rc": None}
    m = re.fullmatch(r"(gate|2Lg|2L)(_e4)?(_rc9999)?", arm)
    if not m or (m.group(1) == "gate" and m.group(3)) or (m.group(1) == "2L" and m.group(2)):
        raise ValueError(f"not an arm: {arm!r}")
    kind = m.group(1)
    return {"two": kind != "gate", "gate": None if kind == "2L" else ("e4" if m.group(2) else "e3"),
            "rc": None if kind == "gate" else ("r0.9999" if m.group(3) else "r0.999")}


def arm_name(two, gate, rc=None):
    if not two:
        return "today" if gate is None else ("gate" if gate == "e3" else "gate_e4")
    s = "2Lg" if gate else "2L"
    s += "_e4" if gate == "e4" else ""
    return s + ("_rc9999" if rc == "r0.9999" else "")


def gate_arm_of(arm):
    """The gate-only arm an arm's gate is (None without one): the reference of #122's 2L+gate rule."""
    g = arm_parts(arm)["gate"]
    return None if g is None else arm_name(False, g)


def cascade_arm(arm):
    """The frozen development cascade (eval/vocr6_gate_protocol.json development.cascade): the next tier's arm of the
    same kind, at its primary settings (the 1e-3 gate, recall 0.999)."""
    p = arm_parts(arm)
    if not p["two"]:
        return "today" if p["gate"] is None else "gate"
    return "2Lg" if p["gate"] else "2L"


def two_candidate(tier, gate_final, rc=None):
    """#122's candidate on top of the tier's adopted gate (None: no gate): 2Lg (or 2Lg_e4) if a gate was adopted, else
    2L; None for Small."""
    if tier not in TWO_LEVEL_TIERS:
        return None
    return arm_name(True, None if gate_final is None else arm_parts(gate_final)["gate"], rc)


def a9_cfgs(arm):
    """[(config, pld bases)] of the A9 measurements an arm's rules read."""
    p = arm_parts(arm)
    if arm == "today":
        return [("today", (0, 1))]
    if not p["two"]:
        return [(f"{arm}_{m}", (0, 1)) for m in ("off", "score", "on")]
    out = [(arm, (0, 1)), ("today", (0, 1)), ("2L_tinf", (0,)), ("2L_t-inf", (0,))]
    g = gate_arm_of(arm)
    if g:
        out.append((f"{g}_on", (0, 1)))
    return out


def a9_cfg_of_arm(arm):
    """The A9 measurement of an arm as it runs in Rubi."""
    p = arm_parts(arm)
    return "today" if arm == "today" else (f"{arm}_on" if not p["two"] else arm)


# ---------------------------------------------------------------- rules (pure)

GATE_MARGIN, GATE_MIN_SAVING = 0.05, 0.05
TWO_MARGIN, TWO_MIN_SAVING_PCT, RARE_LOSS_MAX, RARE_UPPER_MAX = 0.05, 3.0, 1.0, 2.0


def not_resolved_worse(cmp, comps=("tv", "dn", "wd")):
    """Neither the score interval nor any component's lies entirely above 0 (eval/headgate_protocol.json rule (a))."""
    return not cmp["score"]["ci95"][0] > 0 and all(not cmp[c]["ci95"][0] > 0 for c in comps)


def gate_rule(cmp, saving_share):
    """eval/headgate_protocol.json development.rule.per_tier for one decode (as eval/headgate_eval.py rule)."""
    return {"not_resolved_worse": bool(not_resolved_worse(cmp)),
            "upper_within_margin": bool(cmp["score"]["ci95"][1] <= GATE_MARGIN),
            "saving_at_least_5pct": bool(saving_share is not None and saving_share >= GATE_MIN_SAVING)}


def gate_verdict(rules):
    """{decode: gate_rule} -> adopted / rejected / behind_the_switch (#120's per tier verdict over its 3 decodes)."""
    vs = []
    for d in DECODES:
        r = rules[d]
        vs.append("rejected" if not r["not_resolved_worse"] else
                  "adopted" if r["upper_within_margin"] and r["saving_at_least_5pct"] else "behind_the_switch")
    return "adopted" if all(v == "adopted" for v in vs) else "rejected" if "rejected" in vs else "behind_the_switch"


def two_rule(cmp, rare):
    """eval/head2level_protocol.json rule (a)-(c) for one decode and reference (as eval/head2level_dev.py score)."""
    return {"a_not_resolved_worse": bool(not_resolved_worse(cmp)),
            "b_noninferior_0.05": bool(cmp["score"]["ci95"][1] <= TWO_MARGIN),
            "c_rare_recall": bool(rare["combined"]["loss_points"] <= RARE_LOSS_MAX and
                                  rare["combined"]["ci95"][1] <= RARE_UPPER_MAX)}


def two_verdict(checks, d_ok, needs_gate_ref, has_gate_ref):
    """#122's per-arm verdict (eval/head2level_dev.py decide): checks = {"ref/decode": two_rule}; adoptable / not
    adopted / rejected, and the failed conditions."""
    fails = [f"{k}: {c}" for k, r in checks.items() for c, ok in r.items() if not ok]
    if needs_gate_ref and not has_gate_ref:
        fails.append("gate reference missing")
    if not d_ok:
        fails.append("d_saving_at_least_3pct")
    a_ok = all(r["a_not_resolved_worse"] for r in checks.values())
    return ("adoptable" if not fails else "rejected" if not a_ok else "not adopted"), fails


def path_check(cmp_doc):
    """The VOCR6 path (protocol vocr6_path.check): Y - X not resolved worse on hval_page and on dev (rubi path)."""
    t = cmp_doc["total"]
    hv, dv = t["hval_page"]["ci95"], t["dev"]["ci95"]
    return {"hval_page_not_resolved_worse": bool(not hv[0] > 0), "dev_not_resolved_worse": bool(not dv[0] > 0),
            "not_worse": bool(not hv[0] > 0 and not dv[0] > 0)}


def read_again(gated, escaped, frames, contexts, two, greedy):
    """Rubi core/vocr_ocr.c read_again: a line read again without skipping (report only here)."""
    skipped = gated > 0 or (two and escaped + gated < frames)
    return bool(skipped and (contexts > 0 or (two and REREAD_RE.search(greedy or "") is not None)))


def line_conf(conf_min, greedy):
    """Rubi core/vocr_ocr.c line_conf: conf_min x 1000 + 0.5 truncated, clamped; 1000 for an empty greedy line."""
    if not greedy:
        return 1000
    c = np.float32(np.float32(conf_min) * np.float32(1000.0) + np.float32(0.5))
    return 0 if c <= 0 else 1000 if c >= 1000 else int(c)


def cascade(texts, confs, up):
    """A line under CASCADE_CONF shows the next tier's text (up), else its own."""
    if up is None:
        return list(texts)
    if len(up) != len(texts):
        raise ValueError("the cascade's texts hold another number of crops")
    return [u if c < CASCADE_CONF else t for t, c, u in zip(texts, confs, up)]


def box_ms(r, sum_t2=None, frames_per_line=None, lines=None, finder_ms=None):
    """Est. ms of the nominal box (3 lines of 80 frames + 100 ms) from an A9 reading r (a9_read): per frame, and with an
    attention op its scores-and-sums phase projected from the pack's sum of T^2 to the box's lines (protocol
    estimates.nominal_box)."""
    fpl = frames_per_line or NOMINAL["frames_per_line"]
    nl = lines or NOMINAL["lines"]
    fin = NOMINAL["finder_ms"] if finder_ms is None else finder_ms
    core = r.get("attn_core_ms", 0.0) or 0.0
    if core and sum_t2:
        per_t2 = core * r["frames"] / sum_t2
        return nl * fpl * (r["ms_per_frame"] - core) + nl * fpl * fpl * per_t2 + fin
    return nl * fpl * r["ms_per_frame"] + fin


def gate_saving(off, score, on, s_pack, s_dev, box0):
    """eval/headgate_eval.py a9's projection: the saving per frame at skip share s is head(off) - head(off) (1 - s)
    kappa - (gate(score) + c s), kappa the measured batch efficiency, c the measured blank-logit cost per skipped
    frame; and its share of the nominal box box0 (ms). Every value is an estimate."""
    h0, h1 = off["head_ms"], on["head_ms"]
    g0, g1 = score["gate_ms"], on["gate_ms"]
    kappa = h1 / (h0 * (1 - s_pack))
    cb = (g1 - g0) / s_pack
    out = {"kappa": rnd(kappa), "blank_logit_ms_per_skipped": rnd(cb, 5), "saving": {}}
    for name, s in (("pack", s_pack), ("dev_wd", s_dev)):
        d = h0 - h0 * (1 - s) * kappa - (g0 + cb * s)
        out["saving"][name] = {"skip_share": rnd(s, 5), "ms_per_frame": rnd(d),
                               "nominal_box_ms": rnd(NOMINAL_FRAMES * d, 1),
                               "nominal_box_ms_444": rnd(NOMINAL_FRAMES * d * MHZ / MHZ_HI, 1),
                               "nominal_share": rnd(NOMINAL_FRAMES * d / box0, 5),
                               "real_box_ms": rnd(REAL_FRAMES * d, 1)}
    return out


def saving_pct(base_ms, arm_ms, box):
    sv = base_ms - arm_ms
    return {"saved_ms_per_frame": rnd(sv), "saved_ms_nominal_box": rnd(NOMINAL_FRAMES * sv, 1),
            "saved_ms_real_box": rnd(REAL_FRAMES * sv, 1),
            "saved_pct_nominal_box": rnd(100 * NOMINAL_FRAMES * sv / box, 2),
            "saved_ms_nominal_box_444": rnd(NOMINAL_FRAMES * sv * MHZ / MHZ_HI, 1)}


def two_saving(today, ref, arm, never, always, wd, gated, box_today):
    """eval/head2level_time.py: the measured saving against the reference (gate for 2Lg, today for 2L) and the
    projection to the development rates on wd's finder-cut lines; (d) needs both >= 3% of the nominal box (head2level
    amendment 2).
    today / ref / arm: A9 readings without prefetch credit; never / always: the 2L blob at t = +inf / -inf."""
    meas = saving_pct(ref["ms_per_frame"], arm["ms_per_frame"], box_today)
    th = today["head_ms"]
    if gated:
        s, f = wd["gated_share"], wd["fire_share_all_frames"]
        sv = (1 - s) * (th - never["head_ms"]) - f * (always["head_ms"] - never["head_ms"])
        proj = {"wd_gated_share": s, "wd_fire_share_all_frames": f}
    else:
        f = wd["fire_share"]
        sv = th - (never["head_ms"] + f * (always["head_ms"] - never["head_ms"]))
        proj = {"wd_fire_share": f}
    proj.update(head_ms_never=rnd(never["head_ms"]), head_ms_always=rnd(always["head_ms"]), today_head_ms=rnd(th),
                **saving_pct(sv, 0.0, box_today))
    lo = min(meas["saved_pct_nominal_box"], proj["saved_pct_nominal_box"])
    return {"measured": meas, "projected": proj, "saved_pct_nominal_box_safe_min": lo,
            "d_saving_at_least_3pct": bool(lo >= TWO_MIN_SAVING_PCT)}


def final_arm(gate_final, two_final):
    return two_final or gate_final or "today"


def shipping_name(prefix, tier, arm):
    """The shipped file name of a tier's decided configuration."""
    p = arm_parts(arm)
    if arm == "today":
        return f"{prefix}_{tier}_w8.vocr"
    if not p["two"]:
        return f"{prefix}_{tier}_w8g.vocr"
    return f"{prefix}_{tier}_w8g2.vocr" if p["gate"] else f"{prefix}_{tier}_w8_2l.vocr"


def parse_final_blobs(text):
    """vocr6-lead's message 'FINAL BLOBS S=<path> N=<path> L=<path>' (the last one in text) -> {tier: path}."""
    found = None
    for m in re.finditer(r"FINAL BLOBS((?:\s+[SNL]=\S+){3})", text):
        found = dict(kv.split("=", 1) for kv in m.group(1).split())
    if not found or set(found) != set(TIERS):
        return None
    return {t: found[t].rstrip(".,;") for t in TIERS}


# ---------------------------------------------------------------- the run's context

class Ctx:
    """OUT's configuration (OUT/config.json, written once): prefix, private dir, the VOCR6 path's LM, flags."""

    def __init__(self, out, args=None):
        self.out = Path(out)
        cfg_path = self.out / "config.json"
        if cfg_path.exists():
            cfg = json.loads(cfg_path.read_text())
        else:
            if args is None:
                raise SystemExit(f"{cfg_path}: no run here")
            cfg = {"prefix": args.prefix, "all_vns": bool(args.all_vns), "frozen_only": bool(args.frozen_only),
                   "priv": str(args.priv or PRIVATE / "vocr6-final-gate" / self.out.name), "kit": str(KIT),
                   "path_lm": {"hval": str(args.path_lm_hval), "dev": str(args.path_lm_dev)},
                   "path_lm_opts": args.path_lm_opts, "plans4": not args.no_plans4, "procs": args.procs,
                   "created": now(), "repo_head": git_head(), "protocol_sha256": sha256(PROTOCOL)}
            self.out.mkdir(parents=True, exist_ok=True)
            cfg_path.write_text(json.dumps(cfg, indent=1) + "\n")
        self.cfg = cfg
        self.prefix, self.all_vns, self.frozen_only = cfg["prefix"], cfg["all_vns"], cfg["frozen_only"]
        self.priv, self.kit = Path(cfg["priv"]), Path(cfg["kit"])
        self.procs = cfg["procs"]
        self.exclude = frozenset() if self.all_vns else frozenset(EXCLUDE_VNS)
        self._dev = None

    def tdir(self, tier, *parts):
        d = self.out.joinpath(tier, *parts)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def pdir(self, tier, *parts):
        d = self.priv.joinpath(tier, *parts)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def input(self, tier):
        p = self.out / tier / "input.json"
        if not p.exists():
            raise SystemExit(f"{tier}: not prepared in {self.out}")
        return json.loads(p.read_text())

    def blob(self, tier, arm):
        if arm == "today":
            return Path(self.input(tier)["blob"])
        p, b, P = arm_parts(arm), self.out / tier / "blobs", self.prefix
        if not p["two"]:
            return b / (f"{P}_{tier}_w8g.vocr" if p["gate"] == "e3" else f"{P}_{tier}_w8g_eps0.0001.vocr")
        K = self.kalt(tier)
        rc = "" if p["rc"] == "r0.999" else "_rc9999"
        if p["gate"] is None:
            return b / f"{P}_{tier}_2L_K{K}_r96{rc}.vocr"
        return b / f"{P}_{tier}_2Lg{'' if p['gate'] == 'e3' else '_e4'}_K{K}_r96{rc}.vocr"

    def perm(self, tier):
        return self.out / tier / "blobs" / f"{self.prefix}_{tier}_perm.vocr"

    def kalt(self, tier):
        return json.loads((self.out / tier / "fit_2l.json").read_text())["K_alt"]

    def dev(self):
        if self._dev is None:
            self._dev = Dev(self.priv / "dev")
        return self._dev


def git_head():
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def step(ctx, tier, name, fn):
    """Run fn once: OUT/<tier>/steps/<name>.json records it (wall clock, slot-hours at one slot) and what it returns."""
    rec = ctx.tdir(tier, "steps") / f"{name}.json"
    if rec.exists():
        return json.loads(rec.read_text())
    t0, start = time.time(), now()
    print(f"step {tier} {name}: started {start}", flush=True)
    info = fn() or {}
    wall = time.time() - t0
    doc = {"tier": tier, "step": name, "start": start, "end": now(), "wall_s": round(wall, 1), "slots": 1,
           "slot_hours": round(wall / 3600, 4), **info}
    tmp = rec.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n")
    tmp.replace(rec)
    print(f"step {tier} {name}: {wall:.0f} s", flush=True)
    return doc


def sh(ctx, tier, name, cmd, cwd=ROOT, env=None):
    log = ctx.tdir(tier, "logs") / f"{name}.log"
    with open(log, "ab") as f:
        f.write(f"\n# {now()} {' '.join(str(c) for c in cmd)}\n".encode())
        f.flush()
        p = subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL, env=env)
    if p.returncode:
        raise SystemExit(f"{tier} {name} failed (rc {p.returncode}); see {log}")


# ---------------------------------------------------------------- inputs and blobs

def blob_info(path):
    import export as E
    b = E.read_blob(Path(path).read_bytes())
    ops = b["ops"]
    attn = [op for op in ops if op["type"] == 5]
    return {"n_classes": b["n_classes"], "ops": len(ops), "head_k": ops[-1]["k"], "stride": b["stride"],
            "height": b["height"], "max_width": b["max_width"], "params": b["params"],
            "attention": ({"dim": attn[0]["attn_dim"], "heads": attn[0]["heads"]} if attn else None),
            "has_gate": b.get("gate") is not None}


def cmd_input(ctx, tier, blob):
    p = ctx.out / tier / "input.json"
    blob = Path(blob).resolve()
    rec = {"tier": tier, "blob": str(blob), "sha256": sha256(blob), "bytes": blob.stat().st_size, **blob_info(blob),
           "recorded": now()}
    if p.exists():
        old = json.loads(p.read_text())
        if old["sha256"] != rec["sha256"]:
            raise SystemExit(f"{tier}: {ctx.out} was prepared with another blob ({old['blob']})")
        return old
    if rec["has_gate"]:
        raise SystemExit(f"{blob}: a gated blob; give the tier's int8 blob (*_w8.vocr)")
    ctx.tdir(tier)
    p.write_text(json.dumps(rec, indent=1) + "\n")
    return rec


def ensure_gate_fit(ctx, tier):
    blob = ctx.input(tier)["blob"]
    feat = ctx.pdir(tier, "features")
    step(ctx, tier, "gate_features", lambda: sh(ctx, tier, "gate_features", [
        PY, "train/headgate.py", "features", "--tier", tier, "--blob", blob, "--out", feat, "--procs", ctx.procs]))

    def fit():
        sh(ctx, tier, "gate_fit", [PY, "train/headgate.py", "fit", "--tier", tier, "--blob", blob, "--features", feat,
                                   "--out", ctx.tdir(tier, "blobs"), "--record", ctx.out / tier / "fit_gate.json",
                                   "--prefix", ctx.prefix])
        r = json.loads((ctx.out / tier / "fit_gate.json").read_text())
        return {"gates": {e: {k: g[k] for k in ("file", "sha256", "bytes", "t", "cal_skip_share")}
                          for e, g in r["gates"].items()}, "c_equals_twin": r["c_equals_twin"]}
    step(ctx, tier, "gate_fit", fit)


def ensure_two_fit(ctx, tier):
    if sha256(RANK) != RANK_SHA:
        raise SystemExit(f"{RANK}: not #122's ranking")
    blob = ctx.input(tier)["blob"]
    perm = ctx.perm(tier)
    step(ctx, tier, "two_permute", lambda: sh(ctx, tier, "two_permute", [
        PY, "train/head2level.py", "permute", "--blob", blob, "--rank", RANK, "--out", perm]))
    train = ctx.pdir(tier, "two") / f"train_{tier}.npz"
    step(ctx, tier, "two_extract", lambda: sh(ctx, tier, "two_extract", [
        PY, "eval/head2level_extract.py", "train", "--tier", tier, "--blob", perm, "--out", train,
        "--procs", ctx.procs]))

    def escapes():
        sh(ctx, tier, "two_escapes", [PY, "eval/head2level_fit.py", "escapes", "--tier", tier, "--train", train,
                                      "--blob", perm, "--out", ctx.out / tier / "fit_2l.json", "--threads", ctx.procs])
        f = json.loads((ctx.out / tier / "fit_2l.json").read_text())
        return {"K_alt": f["K_alt"], "W": {k: v["W"] for k, v in f["fits"].items()}}
    step(ctx, tier, "two_escapes", escapes)

    def timing_blobs():
        two = ensure_blob(ctx, tier, "2L")
        b = ctx.tdir(tier, "blobs")
        for tag, t in (("tinf", "inf"), ("t-inf", "-inf")):
            sh(ctx, tier, "two_retarget", [PY, "eval/head2level_fit.py", "retarget", "--blob", two, f"--t={t}",
                                           "--out", b / f"time_{tier}_2L_{tag}.vocr"])
    step(ctx, tier, "two_timing_blobs", timing_blobs)


def ensure_blob(ctx, tier, arm):
    """The arm's blob (gate blobs come from the gate fit; two-level blobs are built and composed here)."""
    path = ctx.blob(tier, arm)
    if path.exists():
        return path
    p = arm_parts(arm)
    if not p["two"]:
        raise SystemExit(f"{path}: missing (the gate fit writes it)")
    base = ctx.blob(tier, arm_name(True, None, p["rc"]))
    if not base.exists():
        K = ctx.kalt(tier)
        sh(ctx, tier, "two_build", [PY, "eval/head2level_fit.py", "build", "--tier", tier, "--blob", ctx.perm(tier),
                                    "--fit", ctx.out / tier / "fit_2l.json", "--K", K, "--recall", p["rc"],
                                    "--out", base])
    if p["gate"]:
        sh(ctx, tier, "two_compose", [PY, "eval/head2level_fit.py", "compose", "--two", base,
                                      "--gated", ctx.blob(tier, gate_arm_of(arm)), "--out", path])
    return path


# ---------------------------------------------------------------- the development cohorts

WD = ("tv_dlg", "dn_dlg")
TV = "teacher_val_dev"
DN = ("devnative_tv_native_ink_dev", "devnative_tv_native_grad_dev", "devnative_strict_ink_dev",
      "devnative_strict_grad_dev")


class Dev:
    """VOCR5's development sample as eval/headgate_protocol.json pins it, in eval/head2level_dev.py's read order."""

    def __init__(self, d):
        import head2level_dev as H2D
        H2D.check_cohorts()
        self.dir = Path(d)
        self.pools = H2D.POOLS
        self.fixed = H2D.fixed_rows_cached()
        self.cases = {n: H2D.load_cohort_cached(n) for n in WD}
        with open(self.dir / "wd_lines.pkl", "rb") as f:
            self.wl = pickle.load(f)
        self.n_lines = {n: {m: [len(x[0]) for x in self.wl["cohorts"][n][m]] for m in ("grad", "refine")} for n in WD}
        self.tv_shots = {c["shot"] for c in self.cases["tv_dlg"]}
        names = {}
        for stem in DN:
            z = np.load(self.pools / f"{stem}.npz")
            names.update({m["id"]: "name" in (m.get("styles") or []) for m in json.loads(str(z["meta"]))})
        self.dn_names = names
        keys = []
        for stem, rows in self.fixed.items():
            for i in range(len(rows)):
                keys += [f"{stem}|{i}|rv", f"{stem}|{i}|rvn"]
        for name in WD:
            for mode in ("grad", "refine"):
                for ci in range(len(self.cases[name])):
                    keys += [f"{name}|{mode}|{ci}|{li}" for li in range(self.n_lines[name][mode][ci])]
        self.keys = keys
        self.index = {k: i for i, k in enumerate(keys)}
        self.keys_sha256 = hashlib.sha256("\n".join(keys).encode()).hexdigest()

    def vn_of(self, key):
        p = key.split("|")
        if p[0] in WD:
            return self.cases[p[0]][int(p[2])]["vn"]
        return self.fixed[p[0]][int(p[1])]["vn"]

    def is_wd(self, key):
        return key.split("|", 1)[0] in WD

    def is_name(self, stem, r):
        return ("namelike" in r["tags"]) if stem == TV else self.dn_names[r["id"]]

    def crops(self):
        import head2level_dev as H2D
        cr = H2D.all_crops(self.dir)
        if [c[0] for c in cr] != self.keys:
            raise SystemExit("head2level_dev.all_crops holds other crops than the cohorts")
        return cr

    def rows(self, tx, exclude):
        """One model's rows of the components tv / dn / wd from [raw text per key], as VOCR5's selection scores them
        (eval/headgate_eval.py comps_for, rows keyed by row; eval/vocr5_select.py load_fixed / load_wd's order): per
        component (errors, reference characters, VNs, name flags); rows of the VNs in exclude left out."""
        from metrics import lev, norm

        from general_vn_textbox_guard import transcript_score
        ix = self.index
        out = {}
        for coh, stems in (("tv", (TV,)), ("dn", DN)):
            e, c, v, nm = [], [], [], []
            for stem in sorted(stems, key=lambda s: str(self.pools / f"{s}.npz")):
                for i, r in enumerate(self.fixed[stem]):
                    if r["vn"] in exclude:
                        continue
                    e.append(sum(lev(norm(tx[ix[f"{stem}|{i}|{w}"]]), r["ref"]) for w in ("rv", "rvn")))
                    c.append(2 * len(r["ref"]))
                    v.append(r["vn"])
                    nm.append(self.is_name(stem, r))
            out[coh] = (e, c, v, nm)
        rows = []
        for coh in WD:
            for mode in ("grad", "refine"):
                for ci, case in enumerate(self.cases[coh]):
                    if (coh == "dn_dlg" and case["shot"] in self.tv_shots) or case["vn"] in exclude:
                        continue
                    preds = [norm(tx[ix[f"{coh}|{mode}|{ci}|{li}"]]) for li in range(self.n_lines[coh][mode][ci])]
                    sc = transcript_score(case["lines"], preds)["all"]["full"]
                    rows.append((f"{coh}|{mode}|{case['unit']}|{case['variant']}", sc["errors"], sc["chars"],
                                 case["vn"]))
        rows.sort(key=lambda r: r[0])
        if len({r[0] for r in rows}) != len(rows):
            raise SystemExit("wd: two units share a key")
        out["wd"] = ([r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows], [False] * len(rows))
        return out

    def components(self, models):
        """eval/vocr5_select.py Components tv / dn / wd of {model: rows()}."""
        import vocr5_select as V
        out = {n: V.Component(n) for n in ("tv", "dn", "wd")}
        for model, rows in models.items():
            for n in out:
                e, c, v, nm = rows[n]
                out[n].add(model, e, c, v, nm, {})
        return out

    def rare_rows(self, tx, rare, exclude):
        """eval/head2level_dev.py rare_rows over [raw text per key], rows of the VNs in exclude left out."""
        import head2level_dev as H2D
        from metrics import norm
        ix = self.index
        out = {}
        for comp, stems in (("tv", (TV,)), ("dn", DN)):
            tot, hit, vns = [], [], []
            for stem in sorted(stems, key=lambda s: str(self.pools / f"{s}.npz")):
                for i, r in enumerate(self.fixed[stem]):
                    if self.is_name(stem, r) or r["vn"] in exclude:
                        continue
                    t = h = 0
                    for v in ("rv", "rvn"):
                        a, b = H2D.align_recall(norm(tx[ix[f"{stem}|{i}|{v}"]]), r["ref"], rare)
                        t, h = t + a, h + b
                    tot.append(t)
                    hit.append(h)
                    vns.append(r["vn"])
            out[comp] = (np.array(tot, float), np.array(hit, float), np.array(vns))
        tot, hit, vns = [], [], []
        for name in WD:
            for mode in ("grad", "refine"):
                for ci, c in enumerate(self.cases[name]):
                    if (name == "dn_dlg" and c["shot"] in self.tv_shots) or c["vn"] in exclude:
                        continue
                    pred = "".join(norm(tx[ix[f"{name}|{mode}|{ci}|{li}"]])
                                   for li in range(self.n_lines[name][mode][ci]))
                    ref = "".join(norm(ln["text"]) for ln in c["lines"])
                    a, b = H2D.align_recall(pred, ref, rare)
                    tot.append(a)
                    hit.append(b)
                    vns.append(c["vn"])
        out["wd"] = (np.array(tot, float), np.array(hit, float), np.array(vns))
        return out


def ensure_dev_lines(ctx):
    """The C GRAD finder's and the refinement's lines on wd's regions, once per run (eval/head2level_dev.py lines: C ==
    twin box for box), in the private development directory."""
    d = ctx.priv / "dev"
    d.mkdir(parents=True, exist_ok=True)

    def run():
        sh(ctx, "dev", "dev_lines", [PY, "eval/head2level_dev.py", "lines", "--cache", d])
        return {"wd_lines_sha256": sha256(d / "wd_lines.pkl")}
    return step(ctx, "dev", "dev_lines", run)


# ---------------------------------------------------------------- development reads

_R = {}


def _rinit(blob, two, gate, base, check):
    from pyvocr import Vocr, VocrLM
    kw = {}
    if two:
        kw["two_level"] = 1
    if gate:
        kw["head_gate"] = "on"
    rt = Vocr(str(blob), mode="w8i", max_width=1024, **kw)
    _R.update(rt=rt, lm=VocrLM(rt, str(DEV_LM), **RUBI_LM), two=two, gate=gate, check=check)
    if check:
        import headgate as HG
        _R["base"] = Vocr(str(base), mode="w8i", max_width=1024)
        _R["off"] = Vocr(str(blob), mode="w8i", max_width=1024, head_gate="off")
        _R["vocab"] = [_R["base"].class_text(c) for c in range(_R["base"].n_classes)]
        _R["t"] = HG.read_gate(Path(blob).read_bytes())["t"]


def _rread(job):
    import headgate_eval as HE
    from pyvocr import post_counters

    import headgate as HG
    i, chk = job
    _, img, inv = _R["crops"][i]
    rt, lm, two, gate = _R["rt"], _R["lm"], _R["two"], _R["gate"]
    o = rt.run_topk(img, TOPK, invert=inv, gate_score=bool(gate and not two), esc=bool(two))
    g = o["text"]
    lmt, ctx = "", 0
    if g:
        o2, st = rt.redecode(o, *COUNTER)
        lmt = lm.decode(o2["topv"], o2["topc"], o2["blank"])
        ctx = int(st["contexts"])
    skipped = (o["topv"][:, 1] == -np.inf) if o["topv"].shape[1] > 1 else np.zeros(o["frames"], bool)
    check = None
    if chk and _R["check"]:
        a = _R["base"].run_topk(img, TOPK, invert=inv)
        off = _R["off"].run_topk(img, TOPK, invert=inv)
        sim = HG.simulate(a, o["gate_score"] > _R["t"], _R["vocab"])
        check = (bool(HE._same(a, off)), bool(HE._same(sim, o) and sim["gated"] == o["gated"]))
    return (i, g, lmt, post_counters(lmt, rt.lib)[0] if lmt else "", line_conf(o["conf_min"], g), int(o["frames"]),
            int(o["escaped"]), int(o["gated"]), ctx, o["best"].astype(np.uint16), skipped,
            HE.lm_candidates(o), check)


def ensure_reads(ctx, tier, arm):
    """Every development crop through one arm (excluded VNs' crops are not read unless all_vns)."""
    def run():
        import multiprocessing as mp

        from pyvocr import Vocr
        dev = ctx.dev()
        blob = ensure_blob(ctx, tier, arm)
        p = arm_parts(arm)
        gate_only = p["gate"] is not None and not p["two"]
        crops = dev.crops()
        read = [dev.vn_of(k) not in ctx.exclude for k, _, _ in crops]
        idx = [i for i, r in enumerate(read) if r]
        jobs = [(i, gate_only and n % 10 == 0) for n, i in enumerate(idx)]
        _R["crops"] = crops
        res = [None] * len(crops)
        t0 = time.time()
        with mp.get_context("fork").Pool(ctx.procs, initializer=_rinit,
                                         initargs=(blob, p["two"], p["gate"] is not None, ctx.input(tier)["blob"],
                                                   gate_only)) as pool:
            for n, r in enumerate(pool.imap_unordered(_rread, jobs, chunksize=8)):
                res[r[0]] = r
                if n % 4000 == 0:
                    print(f"  {tier} {arm}: {n}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
        empty = (None, "", "", "", 1000, 0, 0, 0, 0, np.zeros(0, np.uint16), np.zeros(0, bool), (0, 0, 0), None)
        res = [r if r is not None else empty for r in res]
        checks = [r[12] for r in res if r[12] is not None]
        ck = {"checked": len(checks), "off_equals_today": sum(c[0] for c in checks),
              "gated_equals_simulation": sum(c[1] for c in checks)}
        if ck["off_equals_today"] != ck["checked"] or ck["gated_equals_simulation"] != ck["checked"]:
            raise SystemExit(f"{tier} {arm}: parity failed {ck}")
        d = ctx.pdir(tier, "reads")
        off = np.zeros(len(res) + 1, np.int64)
        off[1:] = np.cumsum([len(r[9]) for r in res])
        np.savez_compressed(d / f"{arm}.npz", off=off, best=np.concatenate([r[9] for r in res]),
                            skipped=np.concatenate([r[10] for r in res]), frames=np.array([r[5] for r in res]),
                            escaped=np.array([r[6] for r in res]), gated=np.array([r[7] for r in res]),
                            conf=np.array([r[4] for r in res]), contexts=np.array([r[8] for r in res]),
                            lmc=np.array([r[11] for r in res], np.int64), read=np.array(read))
        kw = {"two_level": 1} if p["two"] else {}
        if p["gate"]:
            kw["head_gate"] = "on"
        rt = Vocr(str(blob), mode="w8i", max_width=1024, **kw)
        meta = {"tier": tier, "arm": arm, "blob": str(blob), "blob_sha256": sha256(blob), "blob_bytes": rt.blob_bytes,
                "arena_w8i": rt.arena_bytes, "keys_sha256": dev.keys_sha256, "crops": len(crops), "read": len(idx),
                "exclude": sorted(ctx.exclude), "checks": ck, "greedy": [r[1] for r in res], "lm": [r[2] for r in res],
                "rubi": [r[3] for r in res], "seconds": round(time.time() - t0, 1)}
        (d / f"{arm}.json").write_text(json.dumps(meta, ensure_ascii=False) + "\n", encoding="utf-8")
        return {"crops": len(crops), "read": len(idx), "checks": ck, "arena_w8i": rt.arena_bytes}
    return step(ctx, tier, f"read_{arm}", run)


def load_reads(ctx, tier, arm):
    d = ctx.priv / tier / "reads"
    meta = json.loads((d / f"{arm}.json").read_text(encoding="utf-8"))
    if meta["keys_sha256"] != ctx.dev().keys_sha256:
        raise SystemExit(f"{tier} {arm}: read on other crops")
    z = np.load(d / f"{arm}.npz")
    meta.update({k: z[k] for k in z.files})
    return meta


def ensure_twin(ctx, tier, arm="2L"):
    """#122's exact twin (train/head2level.py twolevel_exact) against the C runtime on every 8th development crop (of
    the read VNs): escape scores, fired frames, argmax, top-8, blank logits and the greedy text."""
    def run():
        import multiprocessing as mp
        blob = ensure_blob(ctx, tier, arm)
        dev = ctx.dev()
        crops = [c for c in dev.crops()[::8] if dev.vn_of(c[0]) not in ctx.exclude]
        _R["crops"] = crops
        with mp.get_context("fork").Pool(ctx.procs, initializer=_tinit, initargs=(blob,)) as pool:
            parts = pool.map(_twin, range(len(crops)), chunksize=16)
        diff = {k: sum(p[k] for p in parts) for k in parts[0] if k not in ("frames", "fired")}
        out = {"blob_sha256": sha256(blob), "crops": len(crops), "frames": sum(p["frames"] for p in parts),
               "fired": sum(p["fired"] for p in parts), "differences": diff}
        if any(diff.values()):
            raise SystemExit(f"{tier} {arm}: the twin differs from the C runtime {diff}")
        return out
    return step(ctx, tier, f"twin_{arm}", run)


def _tinit(blob):
    from pyvocr import Vocr

    import head2level as H2
    data = Path(blob).read_bytes()
    _R.update(head=H2.head_arrays(data), two=H2.read_trailer(data), full=Vocr(str(blob), mode="w8i", max_width=1024),
              rt=Vocr(str(blob), mode="w8i", max_width=1024, two_level=1))


def _twin(i):
    import head2level as H2
    _, img, inv = _R["crops"][i]
    _R["full"].run(img, head_in=True, invert=inv)
    x = _R["full"].last_head_in
    tw = H2.twolevel_exact(x, _R["head"], _R["two"], topk=TOPK)
    o = _R["rt"].run_topk(img, TOPK, invert=inv, esc=True)
    rt = _R["rt"]
    best = tw["best"]
    greedy = "".join(rt.class_text(int(c)) for j, c in enumerate(best) if c and (j == 0 or c != best[j - 1]))
    return {"esc": int((o["esc"] != tw["score"]).sum()), "fired_count": int(o["escaped"] != int(tw["fire"].sum())),
            "best": int((o["best"] != tw["best"]).sum()), "topc": int((o["topc"] != tw["topc"]).any(axis=1).sum()),
            "topv": int((o["topv"] != tw["topv"]).any(axis=1).sum()), "blank": int((o["blank"] != tw["blank"]).sum()),
            "text": int(o["text"] != greedy), "frames": len(x), "fired": int(tw["fire"].sum())}


# ---------------------------------------------------------------- the frozen rules

def texts_of(ctx, tier, arm, reread_today=None):
    """{decode: [raw text per key]} and the confidences of one arm's reads (cascades are added by the caller)."""
    r = load_reads(ctx, tier, arm)
    out = {"greedy": list(r["greedy"]), "rubi": list(r["rubi"])}
    conf = [int(c) for c in r["conf"]]
    p = arm_parts(arm)
    if reread_today is None:
        out["rubi_reread"], conf_rr = list(out["rubi"]), list(conf)
    else:
        rr, conf_rr = [], []
        for i in range(len(conf)):
            if read_again(int(r["gated"][i]), int(r["escaped"][i]), int(r["frames"][i]), int(r["contexts"][i]),
                          p["two"], r["greedy"][i]):
                rr.append(reread_today["rubi"][i])
                conf_rr.append(reread_today["conf"][i])
            else:
                rr.append(out["rubi"][i])
                conf_rr.append(conf[i])
        out["rubi_reread"] = rr
    return out, conf, conf_rr, r


_TEXTS, _ROWS = {}, {}


def arm_texts(ctx, tier, arm, with_cascade=True):
    """Every decode of an arm (the protocol's three and the two re-read reports), the cascade into the next tier's arm
    of the same kind (development.cascade); and its reads. Cached per run."""
    key = (str(ctx.out), tier, arm, with_cascade)
    if key in _TEXTS:
        return _TEXTS[key]
    today = None
    if arm != "today":
        t, c, _, _ = texts_of(ctx, tier, "today")
        today = {"rubi": t["rubi"], "conf": c}
    tx, conf, conf_rr, reads = texts_of(ctx, tier, arm, today)
    nt = NEXT[tier]
    up = up_rr = None
    if nt and with_cascade:
        ntx, _, _ = arm_texts(ctx, nt, cascade_arm(arm), with_cascade=False)
        up, up_rr = ntx["rubi"], ntx["rubi_reread"]
    tx["rubi_cascade"] = cascade(tx["rubi"], conf, up)
    tx["rubi_cascade_reread"] = cascade(tx["rubi_reread"], conf_rr, up_rr)
    _TEXTS[key] = (tx, conf, reads)
    return _TEXTS[key]


def model_rows(ctx, tier, arm, decode):
    key = (str(ctx.out), tier, arm, decode)
    if key not in _ROWS:
        tx, _, _ = arm_texts(ctx, tier, arm)
        _ROWS[key] = ctx.dev().rows(tx[decode], ctx.exclude)
    return _ROWS[key]


def skip_stats(ctx, reads, today_reads):
    """Shares of frames an arm's gate skipped / its escape fired on, over the read development crops: all, wd's
    finder-cut lines, fixed dialogue, names; non-blank (today's argmax) frames skipped."""
    dev = ctx.dev()
    groups = {}
    for i, k in enumerate(dev.keys):
        if not reads["read"][i] or dev.vn_of(k) in ctx.exclude:
            continue
        a, b = reads["off"][i], reads["off"][i + 1]
        fr = int(reads["frames"][i])
        sk = reads["skipped"][a:b]
        nb = today_reads["best"][today_reads["off"][i]:today_reads["off"][i + 1]] != 0
        vals = (fr, int(reads["gated"][i]), int(reads["escaped"][i]), int((sk & nb).sum()), int(nb.sum()))
        parts = ["all", "wd" if dev.is_wd(k) else "fixed"]
        if not dev.is_wd(k):
            stem, j = k.split("|")[:2]
            parts.append("fixed_names" if dev.is_name(stem, dev.fixed[stem][int(j)]) else "fixed_dialogue")
        for g in parts:
            e = groups.setdefault(g, [0, 0, 0, 0, 0])
            for n, v in enumerate(vals):
                e[n] += v
    return {g: {"frames": v[0], "gated": v[1], "escaped": v[2], "gated_share": rnd(v[1] / max(1, v[0]), 5),
                "fire_share_all_frames": rnd(v[2] / max(1, v[0]), 5), "fire_share": rnd(v[2] / max(1, v[0] - v[1]), 5),
                "nonblank_frames": v[4], "nonblank_skipped": v[3],
                "nonblank_skipped_share": rnd(v[3] / max(1, v[4]), 6)} for g, v in sorted(groups.items())}


def compare_arms(ctx, tier, x, y):
    """Per decode: eval/vocr5_select.py compare of y against x (VN clusters, seeds 0 / 1 / 2), the scores, lines
    changed and the names column."""
    import vocr5_select as V
    dev = ctx.dev()
    tx, _, rx = arm_texts(ctx, tier, x)
    ty, _, ry = arm_texts(ctx, tier, y)
    keep = [i for i, k in enumerate(dev.keys) if dev.vn_of(k) not in ctx.exclude]
    out = {}
    for d in DECODES + REREAD:
        comps = dev.components({x: model_rows(ctx, tier, x, d), y: model_rows(ctx, tier, y, d)})
        cmp = V.compare(comps, [x], [y])
        diff = [i for i in keep if tx[d][i] != ty[d][i]]
        out[d] = {"compare": cmp, "score_x": rnd(V.score_of(comps, x)), "score_y": rnd(V.score_of(comps, y)),
                  "changed": {"fixed_reads": sum(not dev.is_wd(dev.keys[i]) for i in diff),
                              "fixed_rows": len({dev.keys[i].rsplit("|", 1)[0] for i in diff
                                                 if not dev.is_wd(dev.keys[i])}),
                              "wd_lines": sum(dev.is_wd(dev.keys[i]) for i in diff)},
                  "names_cer_pct": {m: {c: rnd(100 * comps[c].rate([m], "name")) for c in ("tv", "dn")}
                                    for m in (x, y)}}
    return out, (tx, ty, rx, ry)


def frozen_gate(ctx, tier, arm):
    """#120's rule (eval/headgate_protocol.json development.rule) for a gate arm against the tier's ungated blob."""
    ensure_reads(ctx, tier, "today")
    ensure_reads(ctx, tier, arm)
    nt = NEXT[tier]
    res, (_, _, rx, ry) = compare_arms(ctx, tier, "today", arm)
    est = gate_estimate(ctx, tier, arm, ry)
    share = est["pld0"]["saving"]["dev_wd"]["nominal_share"]
    rules = {d: gate_rule(res[d]["compare"], share) for d in DECODES}
    for d in res:
        res[d]["rule"] = gate_rule(res[d]["compare"], share)
    return {"arm": arm, "blob": str(ctx.blob(tier, arm)), "blob_sha256": sha256(ctx.blob(tier, arm)),
            "cascade": None if nt is None else {"tier": nt, "arm": cascade_arm(arm)}, "decodes": res,
            "skip": skip_stats(ctx, ry, rx), "estimate": est, "verdict": gate_verdict(rules)}


def frozen_two(ctx, tier, arm):
    """#122's rule (eval/head2level_protocol.json development.rule) for a two-level arm: against today and, with a
    gate, against that gate (rare-character recall, (a)-(d))."""
    import head2level_dev as H2D

    import head2level as H2
    dev = ctx.dev()
    blob = ensure_blob(ctx, tier, arm)
    for a in ("today", arm) + ((gate_arm_of(arm),) if gate_arm_of(arm) else ()):
        ensure_reads(ctx, tier, a)
    two = H2.read_trailer(Path(blob).read_bytes())
    L1 = 8 * two["l1_blocks"]
    rare = H2D.rare_set(blob, L1)
    refs = ["today"] + ([gate_arm_of(arm)] if gate_arm_of(arm) else [])
    out = {"arm": arm, "blob": str(blob), "blob_sha256": sha256(blob), "K": ctx.kalt(tier), "L1": L1,
           "rare_classes": len(rare), "vs": {}}
    tarm, _, rarm = arm_texts(ctx, tier, arm)
    rr_arm = {d: dev.rare_rows(tarm[d], rare, ctx.exclude) for d in DECODES}
    checks = {}
    for ref in refs:
        res, (tref, _, _, _) = compare_arms(ctx, tier, ref, arm)
        for d in DECODES:
            rc = H2D.recall_compare(dev.rare_rows(tref[d], rare, ctx.exclude), rr_arm[d])
            res[d]["rare_recall"] = rc
            res[d]["rule"] = two_rule(res[d]["compare"], rc)
            checks[f"{ref}/{d}"] = res[d]["rule"]
        out["vs"][ref] = res
    today_reads = load_reads(ctx, tier, "today")
    out["escape"] = skip_stats(ctx, rarm, today_reads)
    out["oracle_share_wd"] = oracle_share(ctx, tier, today_reads, L1)
    tm = two_estimate(ctx, tier, arm, out["escape"]["wd"])
    out["estimate"] = tm
    verdict, fails = two_verdict(checks, tm["d_saving_at_least_3pct"], arm_parts(arm)["gate"] is not None,
                                 len(refs) > 1)
    out["verdict"], out["fails"] = verdict, fails
    nt = NEXT[tier]
    out["cascade"] = None if nt is None else {"tier": nt, "arm": cascade_arm(arm)}
    return out


def shipped_cascade(ctx, tier, arm):
    """Report only (#122's 'mixed'): the tier's decided configuration with Rubi's cascade into the next tier's decided
    configuration, the mix Rubi ships, against today (today -> today) and the arm's frozen cascade."""
    import vocr5_select as V
    nt = NEXT[tier]
    up = decided_arm(ctx, nt)
    dev = ctx.dev()
    tx, conf, _ = arm_texts(ctx, tier, arm)
    ntx, _, _ = arm_texts(ctx, nt, up, with_cascade=False)
    shipped = cascade(tx["rubi"], conf, ntx["rubi"])
    rows = {"today": model_rows(ctx, tier, "today", "rubi_cascade"),
            "frozen": model_rows(ctx, tier, arm, "rubi_cascade"), "shipped": dev.rows(shipped, ctx.exclude)}
    comps = dev.components(rows)
    keep = [i for i, k in enumerate(dev.keys) if dev.vn_of(k) not in ctx.exclude]
    return {"arm": arm, "next": {"tier": nt, "arm": up}, "report_only": True,
            "scores": {m: rnd(V.score_of(comps, m), 5) for m in rows},
            "cascaded_lines": sum(1 for i in keep if conf[i] < CASCADE_CONF),
            "shipped_vs": {ref: V.compare(comps, [ref], ["shipped"]) for ref in ("today", "frozen")}}


def oracle_share(ctx, tier, today_reads, L1):
    """The share of wd frames whose full-head argmax lies outside level 1 (#122's oracle; today's argmax mapped to
    the permuted classes)."""
    import head2level_dev as H2D
    ranks = H2D.perm_ranks(ctx.input(tier)["blob"], ctx.perm(tier))
    dev = ctx.dev()
    n = m = 0
    for i, k in enumerate(dev.keys):
        if not dev.is_wd(k) or not today_reads["read"][i] or dev.vn_of(k) in ctx.exclude:
            continue
        b = ranks[today_reads["best"][today_reads["off"][i]:today_reads["off"][i + 1]].astype(np.int64)]
        n += len(b)
        m += int((b >= L1).sum())
    return rnd(m / max(1, n), 5)


# ---------------------------------------------------------------- A9 estimates

def a9_env():
    env = dict(os.environ)
    for line in SPEED_ENV.read_text().splitlines():
        m = re.match(r"\s*export\s+(\w+)=(\S+)", line)
        if m:
            env[m.group(1)] = m.group(2)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def a9_blob_cli(ctx, tier, cfg):
    """(blob, vocr_cli options) of an A9 configuration."""
    if cfg == "today":
        return ctx.blob(tier, "today"), ""
    if cfg in ("2L_tinf", "2L_t-inf"):
        return ctx.out / tier / "blobs" / f"time_{tier}_{cfg}.vocr", "--two-level"
    m = re.fullmatch(r"(gate(?:_e4)?)_(off|score|on)", cfg)
    if m:
        return ensure_blob(ctx, tier, m.group(1)), f"--head-gate {m.group(2)}"
    p = arm_parts(cfg)
    return ensure_blob(ctx, tier, cfg), "--two-level" + (" --head-gate on" if p["gate"] else "")


def ensure_a9(ctx, tier, arm):
    jobs = []
    for cfg, plds in a9_cfgs(arm):
        for pld in plds:
            d = ctx.out / tier / "a9" / f"pld{pld}_{cfg}"
            if not (d / "report.json").exists():
                jobs.append((cfg, pld, d))
    if not jobs:
        return
    env = a9_env()

    def one(j):
        cfg, pld, d = j
        blob, cli = a9_blob_cli(ctx, tier, cfg)
        d.mkdir(parents=True, exist_ok=True)
        cmd = [PY, "vita/a9_layers.py", "measure", "--blob", str(blob), "--pack", str(PACK), "--out", str(d), "--mode",
               "w8i", "--min-frames", "60", "--max-frames", "100", "--limit", "24", "--cli-opts",
               f"--topk {TOPK} {cli}".strip()] + (["--no-pld"] if pld == 0 else [])
        with open(d / "measure.log", "wb") as f:
            p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
        if p.returncode:
            return f"{cfg} pld{pld}: measure rc {p.returncode}"
        with open(d / "layers.md", "wb") as f:
            p = subprocess.run([PY, "vita/a9_layers.py", "report", str(d), "--json", str(d / "report.json"), "--scale",
                                str(SCALE)], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                               env=env)
        return None if not p.returncode else f"{cfg} pld{pld}: report rc {p.returncode}"

    def run():
        with ThreadPoolExecutor(4) as ex:
            errs = [e for e in ex.map(one, jobs) if e]
        if errs:
            raise SystemExit(f"{tier} A9: {errs}")
        return {"runs": [f"pld{p}_{c}" for c, p, _ in jobs]}
    return step(ctx, tier, f"a9_{arm}", run)


_A9 = {}


def a9_read(ctx, tier, cfg, pld, mhz=MHZ):
    """One A9 measurement's calibrated ms per frame (high bound x SCALE at mhz): all phases, the head, the gate step,
    the attention's scores-and-sums phase; texts and runtime hashes."""
    key = (str(ctx.out), tier, cfg, pld, mhz)
    if key in _A9:
        return _A9[key]
    import a9_layers as AL
    d = ctx.out / tier / "a9" / f"pld{pld}_{cfg}"
    meta, frames, res = AL.analyse(d, mhz)
    ms = {i: r["high"] * SCALE / frames / (mhz * 1000) for i, r in res.items()}
    head = max(i for i in res if i < 100)
    out = {"frames": frames, "ms_per_frame": sum(ms.values()), "head_ms": ms.get(head, 0.0),
           "gate_ms": ms.get(PROF_GATE, 0.0), "attn_core_ms": ms.get(PROF_ATTN, 0.0),
           "insns_per_frame": sum(r["insns"] for r in res.values()) / frames,
           "head_insns_per_frame": res[head]["insns"] / frames, "texts_sha256": meta["runs"]["prof"]["texts_sha256"],
           "vocr_c_sha256": meta["vocr_c_sha256"], "blob_sha256": meta["blob_sha256"], "cli_opts": meta["cli_opts"]}
    _A9[key] = out
    return out


_PACKT = {}


def pack_t():
    """The CTC frame count of every line a9_layers.py measures of the pack (60-100 frames, the first 24)."""
    if "t" not in _PACKT:
        import vocr6_attn_cost as AC
        if sha256(PACK) != PACK_SHA:
            raise SystemExit(f"{PACK}: not ocr-speed's sub24 pack")
        _PACKT["t"] = AC.pack_frames(PACK)
    return _PACKT["t"]


def pack_skip_share(ctx, tier, arm):
    """The share of the pack's frames the arm's gate skips (host runtime, the a9 lines)."""
    import a9_layers as AL
    import headgate_eval as HE
    from pyvocr import Vocr
    sub = ctx.tdir(tier, "a9") / "pack24.bin"
    if not sub.exists():
        AL.subset_pack(PACK, sub, 60, 100, 24)
    rt = Vocr(str(ensure_blob(ctx, tier, arm)), mode="w8i", max_width=1024, head_gate="on")
    fr = sk = 0
    for g in HE.read_pack(sub):
        o = rt.run_topk(g, TOPK)
        fr += o["frames"]
        sk += o["gated"]
    return sk / fr, fr


def gate_estimate(ctx, tier, arm, reads):
    """#120's estimates (eval/headgate_eval.py a9) of a gate arm, with and without prefetch credit; rule (c) reads
    pld0's dev_wd share."""
    ensure_a9(ctx, tier, arm)
    t = pack_t()
    st2 = sum(x * x for x in t)
    s_pack, fr = pack_skip_share(ctx, tier, arm)
    dev = ctx.dev()
    wf = ws = 0
    for i, k in enumerate(dev.keys):
        if dev.is_wd(k) and reads["read"][i] and dev.vn_of(k) not in ctx.exclude:
            wf += int(reads["frames"][i])
            ws += int(reads["gated"][i])
    out = {"pack_frames": fr, "pack_skip_share": rnd(s_pack, 5), "dev_wd_skip_share": rnd(ws / wf, 5)}
    for pld in (0, 1):
        r = {m: a9_read(ctx, tier, f"{arm}_{m}", pld) for m in ("off", "score", "on")}
        if r["off"]["frames"] != fr:
            raise SystemExit(f"{tier} {arm}: A9 measured {r['off']['frames']} frames, the pack has {fr}")
        if not r["off"]["texts_sha256"] == r["score"]["texts_sha256"]:
            raise SystemExit(f"{tier} {arm} pld{pld}: off / score texts differ")
        box0 = box_ms(r["off"], st2)
        g = gate_saving(r["off"], r["score"], r["on"], s_pack, ws / wf, box0)
        out[f"pld{pld}"] = {"phases": {m: {k: rnd(v) for k, v in x.items() if isinstance(v, float)}
                                       for m, x in r.items()},
                            "box_ms_off": rnd(box0, 1), "box_ms_on_pack": rnd(box_ms(r["on"], st2), 1),
                            "measured_saving_ms_per_frame": rnd(r["off"]["ms_per_frame"] - r["on"]["ms_per_frame"]),
                            **g}
    return out


def two_estimate(ctx, tier, arm, wd):
    """#122's estimates (eval/head2level_time.py) of a two-level arm against its reference."""
    ensure_a9(ctx, tier, arm)
    t = pack_t()
    st2 = sum(x * x for x in t)
    g = gate_arm_of(arm)
    ref_cfg = f"{g}_on" if g else "today"
    today = a9_read(ctx, tier, "today", 0)
    box_today = box_ms(today, st2)
    out = two_saving(today, a9_read(ctx, tier, ref_cfg, 0), a9_read(ctx, tier, arm, 0),
                     a9_read(ctx, tier, "2L_tinf", 0), a9_read(ctx, tier, "2L_t-inf", 0), wd, g is not None, box_today)
    out["reference"] = ref_cfg
    out["nominal_box_ms_today"] = rnd(box_today, 1)
    pl = {"today": a9_read(ctx, tier, "today", 1), "ref": a9_read(ctx, tier, ref_cfg, 1),
          "arm": a9_read(ctx, tier, arm, 1)}
    out["measured_pld"] = saving_pct(pl["ref"]["ms_per_frame"], pl["arm"]["ms_per_frame"], box_ms(pl["today"], st2))
    out["vs_today"] = {"pld0": saving_pct(today["ms_per_frame"], a9_read(ctx, tier, arm, 0)["ms_per_frame"], box_today),
                       "pld1": saving_pct(pl["today"]["ms_per_frame"], pl["arm"]["ms_per_frame"],
                                          box_ms(pl["today"], st2))}
    if a9_read(ctx, tier, arm, 0)["texts_sha256"] != today["texts_sha256"]:
        out["pack_texts_differ_from_today"] = True
    return out


def config_estimate(ctx, tier, arm):
    """Time and memory of a configuration as Rubi runs it (estimates): ms per frame, the nominal box at 333 and 444
    MHz (the cycle model at 444, and the clock ratio), with and without prefetch credit, the real box, MACs, bytes and
    the W8I arena."""
    from pyvocr import Vocr

    import export as E
    import head2level as H2
    import headgate as HG
    ensure_a9(ctx, tier, arm)
    cfg = a9_cfg_of_arm(arm)
    t = pack_t()
    st2 = sum(x * x for x in t)
    blob = ensure_blob(ctx, tier, arm)
    data = Path(blob).read_bytes()
    p = arm_parts(arm)
    out = {"arm": arm, "blob": str(blob), "blob_sha256": sha256(blob), "blob_bytes": len(data)}
    g, two = HG.read_gate(data), H2.read_trailer(data)
    out["vocg_bytes"] = g["bytes"] if g else 0
    out["voc2_bytes"] = two["bytes"] if two else 0
    b = E.read_blob(data)
    tot, head = E.macs_per_frame(b)
    out["macs_per_frame_full_head"] = int(tot)
    out["macs_per_frame_full_head_T80"] = int(E.macs_per_frame(b, NOMINAL["frames_per_line"])[0])
    out["head_macs_per_frame"] = int(head)
    kw = {"head_gate": "on"} if p["gate"] else {}
    if p["two"]:
        kw["two_level"] = 1
    out["arena_w8i_1024"] = int(Vocr(str(blob), mode="w8i", max_width=1024, **kw).arena_bytes)
    out["arena_w8i_1024_off"] = int(Vocr(str(blob), mode="w8i", max_width=1024).arena_bytes)
    for pld in (0, 1):
        r = a9_read(ctx, tier, cfg, pld)
        r4 = a9_read(ctx, tier, cfg, pld, MHZ_HI)
        bx = box_ms(r, st2)
        out[f"pld{pld}"] = {"ms_per_frame_333": rnd(r["ms_per_frame"]), "head_ms_333": rnd(r["head_ms"]),
                            "nominal_box_ms_333": rnd(bx, 1), "nominal_box_ms_444": rnd(box_ms(r4, st2), 1),
                            "nominal_box_ms_444_clock_ratio": rnd((bx - NOMINAL["finder_ms"]) * MHZ / MHZ_HI +
                                                                  NOMINAL["finder_ms"], 1),
                            "real_box_ms_333": rnd(REAL_FRAMES * r["ms_per_frame"] + NOMINAL["finder_ms"], 1),
                            "insns_per_frame": rnd(r["insns_per_frame"], 0), "vocr_c_sha256": r["vocr_c_sha256"]}
    return out


def effective_macs(est, skip):
    """MACs per frame at the development shares (estimate): the full head scaled by the frames it runs on."""
    tot, head = est["macs_per_frame_full_head"], est["head_macs_per_frame"]
    p = arm_parts(est["arm"])
    if est["arm"] == "today":
        return tot
    s = skip["gated_share"] if p["gate"] else 0.0
    k = 96
    gate = 96 * 16 + 16 if p["gate"] else 0
    if not p["two"]:
        return int(round(tot - head + gate + (1 - s) * head))
    n_cls = head // k
    L1 = est["L1"]
    f_all = skip["fire_share_all_frames"]
    return int(round(tot - head + gate + (1 - s) * (L1 * k + GATE_ESC_MACS) + f_all * (n_cls - L1) * k))


# ---------------------------------------------------------------- the VOCR6 path (vocr6-lead's kit)

def check_kit(ctx):
    bad = {f: h for f, h in KIT_PINS.items() if sha256(ctx.kit / f) != h}
    if bad:
        raise SystemExit(f"the kit at {ctx.kit} is not the protocol's: {sorted(bad)}")
    if not (ctx.kit / "runtime" / "build" / "libvocr.so").exists():
        raise SystemExit(f"build the kit's runtime: make -C {ctx.kit}/runtime build/libvocr.so")
    for name, (p, h) in list(PATH_PLANS.items()) + [("units", DEV_UNITS), ("rubi_norm", RUBI_NORM)]:
        if sha256(p) != h:
            raise SystemExit(f"{p} differs from the protocol's sha256")


def ensure_path_hygiene(ctx, tier):
    def run():
        check_kit(ctx)
        with open(PATH_PLANS["hval"][0], "rb") as f:
            plan = pickle.load(f)
        titles = {u["title"] for u in plan["units"]} | {x["title"] for x in plan["fixed"]}
        if TEST_TITLE in titles:
            raise SystemExit(f"the hval plan holds {TEST_TITLE}")
        lms = {k: {"path": v, "sha256": sha256(v)} for k, v in ctx.cfg["path_lm"].items()}
        for k, v in lms.items():
            if v["sha256"] != PATH_LM_SHA[k] and not os.environ.get("VOCR6_GATE_ANY_LM"):
                raise SystemExit(f"the {k} LM {v['path']} is not the protocol's (amendment 1)")
        if ctx.cfg.get("plans4"):
            for name, (pp, h) in PLANS4.items():
                if sha256(pp) != h:
                    raise SystemExit(f"{pp} differs from the protocol's sha256 (amendment 1)")
                if name == "hval4":
                    with open(pp, "rb") as f:
                        p4 = pickle.load(f)
                    t4 = {u["title"] for u in p4["units"]} | {x["title"] for x in p4["fixed"]}
                    if TEST_TITLE in t4:
                        raise SystemExit(f"{pp} holds {TEST_TITLE}")
        return {"hval_titles": len(titles), "test_title_absent": True, "lm": lms, "lm_opts": ctx.cfg["path_lm_opts"],
                "plans4": ctx.cfg.get("plans4")}
    return step(ctx, tier, "path_hygiene", run)


def ensure_dumps(ctx, tier, arm):
    if ctx.frozen_only:
        return None

    def run():
        blob = ensure_blob(ctx, tier, arm)
        p = arm_parts(arm)
        d = ctx.pdir(tier, "path")
        out = {}
        plans = dict(PATH_PLANS, **(PLANS4 if ctx.cfg.get("plans4") else {}))
        for plan, (pp, _) in plans.items():
            dump = d / f"dump_{plan}_{arm}.json"
            if dump.exists():
                continue
            lm = ctx.cfg["path_lm"]["hval" if plan.startswith("hval") else "dev"]
            cmd = [PY, "eval/vocr6_score.py", "read", "--plan", pp, "--blob", blob, "--lm", lm,
                   "--lm-opts", ctx.cfg["path_lm_opts"], "--out", dump, "--workers", ctx.procs]
            if p["gate"]:
                cmd += ["--head-gate", "on"]
            if p["two"]:
                cmd += ["--two-level"]
            sh(ctx, tier, f"path_{arm}", cmd, cwd=ctx.kit)
            out[plan] = sha256(dump)
        return {"dumps": out}
    ensure_path_hygiene(ctx, tier)
    return step(ctx, tier, f"path_{arm}", run)


def path_score(ctx, tier, arm, name, up_tier, up_arm, four=False):
    """vocr6_score.py score of the arm's dumps (cascade into up_tier's up_arm dumps) -> OUT/<tier>/path/<dir>/ (path4:
    on vocr6-lead's stage-4 plans, report only)."""
    hv, dv = ("hval4", "dev4") if four else ("hval", "dev")
    plans = dict(PATH_PLANS, **PLANS4)
    sub = ctx.tdir(tier, "path4" if four else "path", "X" if name == "X" else arm)
    out = sub / f"score_{name}.json"
    casc = sub / f"cascade_{name}.json"
    want = {"tier": up_tier, "arm": up_arm}
    if out.exists() and casc.exists() and json.loads(casc.read_text()) == want:
        return sub
    for f in [out, casc] + list(sub.glob("compare_*.json")):
        f.unlink(missing_ok=True)
    d = ctx.priv / tier / "path"
    cmd = [PY, "eval/vocr6_score.py", "score", "--name", name, "--blob", ensure_blob(ctx, tier, arm),
           "--rubi-norm", RUBI_NORM[0], "--hval-plan", plans[hv][0],
           "--hval-dump", d / f"dump_{hv}_{arm}.json",
           "--dev-plan", plans[dv][0], "--dev-units", DEV_UNITS[0], "--dev-dump", d / f"dump_{dv}_{arm}.json",
           "--exclude-vns", *EXCLUDE_VNS, "--out", out]
    if up_tier:
        u = ctx.priv / up_tier / "path"
        cmd += ["--hval-cascade-dump", u / f"dump_{hv}_{up_arm}.json",
                "--dev-cascade-dump", u / f"dump_{dv}_{up_arm}.json"]
    sh(ctx, tier, f"path_score_{name}_{arm}{'_4' if four else ''}", cmd, cwd=ctx.kit)
    casc.write_text(json.dumps(want) + "\n")
    return sub


def vocr6_path(ctx, tier, arm):
    """The protocol's vocr6_path: Y (the arm, cascade into the next tier's decided configuration) against X (the
    ungated blob, cascade into the next tier's ungated blob), vocr6_select.py compare on the rubi and greedy paths."""
    if ctx.frozen_only:
        return None
    nt = NEXT[tier]
    up = decided_arm(ctx, nt) if nt else None
    ensure_dumps(ctx, tier, "today")
    ensure_dumps(ctx, tier, arm)
    if nt:
        ensure_dumps(ctx, nt, "today")
        ensure_dumps(ctx, nt, up)
    res = {"x": {"arm": "today", "cascade": None if nt is None else {"tier": nt, "arm": "today"}},
           "y": {"arm": arm, "cascade": None if nt is None else {"tier": nt, "arm": up}}}
    for four in ((False, True) if ctx.cfg.get("plans4") else (False,)):
        xs = path_score(ctx, tier, "today", "X", nt, "today", four)
        ys = path_score(ctx, tier, arm, "Y", nt, up, four)
        env = dict(os.environ, VOCR6_ANY_PLAN="1") if four else None     # amendment 1: v3 holds v2's unit ids
        part = {}
        for path in ("rubi", "greedy"):
            f = ys / f"compare_{path}.json"
            if not f.exists():
                log = ctx.tdir(tier, "logs") / f"path_compare_{arm}{'_4' if four else ''}.log"
                with open(log, "ab") as fh:
                    p = subprocess.run([PY, "eval/vocr6_select.py", "compare", "--scores", str(xs), str(ys), "--x",
                                        "X", "--y", "Y", "--path", path, "--out", str(f)], cwd=ctx.kit, stdout=fh,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
                if p.returncode:
                    raise SystemExit(f"{tier} {arm}: vocr6_select compare failed; see {log}")
            part[path] = {"total": json.loads(f.read_text())["total"]}
        for name, sub in (("X", xs), ("Y", ys)):
            sc = json.loads((sub / f"score_{name}.json").read_text())
            part[f"{name.lower()}_summary"] = sc["summary"]
            part[f"{name.lower()}_blob_sha256"] = sc["blob_sha256"]
            part[f"{name.lower()}_hval_plan_sha256"] = sc["hval_plan_sha256"]
        if four:
            part["check"] = path_check({"total": part["rubi"]["total"]})
            res["stage4_plans_report"] = part
        else:
            res.update(rubi=part["rubi"], greedy=part["greedy"])
            res["x"].update(summary=part["x_summary"], blob_sha256=part["x_blob_sha256"])
            res["y"].update(summary=part["y_summary"], blob_sha256=part["y_blob_sha256"])
    res["check"] = path_check({"total": res["rubi"]["total"]})
    res["not_worse"] = res["check"]["not_worse"]
    return res


def decided_arm(ctx, tier):
    p = ctx.out / tier / "decision.json"
    if not p.exists():
        raise SystemExit(f"decide {tier} first (its configuration is the cascade's)")
    return json.loads(p.read_text())["final_arm"]


# ---------------------------------------------------------------- prepare / decide / run

def prepare(ctx, tier, blob):
    cmd_input(ctx, tier, blob)
    ensure_gate_fit(ctx, tier)
    if tier in TWO_LEVEL_TIERS:
        ensure_two_fit(ctx, tier)
        ensure_blob(ctx, tier, "2Lg")
    ensure_dev_lines(ctx)
    arms = ["today", "gate"] + (["2L", "2Lg"] if tier in TWO_LEVEL_TIERS else [])
    for a in arms:
        ensure_reads(ctx, tier, a)
    if tier in TWO_LEVEL_TIERS:
        ensure_twin(ctx, tier, "2L")
    for a in arms:
        ensure_a9(ctx, tier, a)
    for a in ["today", "gate"] + (["2Lg"] if tier in TWO_LEVEL_TIERS else []):
        ensure_dumps(ctx, tier, a)
    (ctx.out / tier / "prepared.json").write_text(json.dumps({"tier": tier, "at": now()}) + "\n")


def decide(ctx, tier):
    """The protocol's decision for one tier (decision.order: the next tier decided first)."""
    nt = NEXT[tier]
    if nt and not ctx.frozen_only:
        decided_arm(ctx, nt)
    dec = {"kind": "vocr6-gate-decision", "tier": tier, "at": now(), "frozen_only": ctx.frozen_only,
           "exclude_vns": sorted(ctx.exclude), "gate": {}, "two_level": {}}
    gate_final = None
    for g in ("gate", "gate_e4"):
        fr = frozen_gate(ctx, tier, g)
        pth = vocr6_path(ctx, tier, g)
        ok = fr["verdict"] == "adopted" and (pth is None or pth["not_worse"])
        dec["gate"][g] = {"frozen": fr, "path": pth, "adopted": bool(ok)}
        print(f"{tier} {g}: frozen {fr['verdict']}, path " +
              ("-" if pth is None else ("not worse" if pth["not_worse"] else "resolved worse")) +
              f" -> {'adopted' if ok else 'not adopted'}", flush=True)
        if ok:
            gate_final = g
            break
    two_final = None
    if tier in TWO_LEVEL_TIERS:
        for rc in ("r0.999", "r0.9999"):
            arm = two_candidate(tier, gate_final, rc)
            fr = frozen_two(ctx, tier, arm)
            pth = vocr6_path(ctx, tier, arm)
            ok = fr["verdict"] == "adoptable" and (pth is None or pth["not_worse"])
            dec["two_level"][arm] = {"frozen": fr, "path": pth, "adopted": bool(ok)}
            print(f"{tier} {arm}: frozen {fr['verdict']} {fr['fails'][:3]}, path " +
                  ("-" if pth is None else ("not worse" if pth["not_worse"] else "resolved worse")) +
                  f" -> {'adopted' if ok else 'not adopted'}", flush=True)
            if ok:
                two_final = arm
                break
    released = VOCR5_RELEASED.get(ctx.input(tier)["sha256"])
    if released and released[0] == tier and not ctx.all_vns:
        # models.vocr5_final: the released VOCR5 blob keeps #120's / #122's verdict and blobs (step 2 reproduces them
        # byte for byte); the rules above are reported beside it
        dec["kept_vocr5_verdict"] = {"by_the_rules": {"gate": gate_final, "two_level": two_final},
                                     "kept": {"gate": released[1], "two_level": released[2]}}
        gate_final, two_final = released[1], released[2]
    fa = final_arm(gate_final, two_final)
    dec.update(gate_final=gate_final, two_final=two_final, final_arm=fa)
    dec["estimate"] = {"today": config_estimate(ctx, tier, "today"), "final": config_estimate(ctx, tier, fa)}
    if fa != "today":
        sk = (dec["two_level"][two_final]["frozen"]["escape"]["wd"] if two_final else
              dec["gate"][gate_final]["frozen"]["skip"]["wd"])
        if two_final:
            dec["estimate"]["final"]["L1"] = dec["two_level"][two_final]["frozen"]["L1"]
        dec["estimate"]["final"]["effective_macs_per_frame_dev_wd"] = effective_macs(dec["estimate"]["final"], sk)
    if nt and fa != "today":
        dec["shipped_cascade"] = shipped_cascade(ctx, tier, fa)
    fb = ensure_blob(ctx, tier, fa)
    fin = ctx.tdir("final")
    name = shipping_name(ctx.prefix, tier, fa)
    if fa != "today":
        shutil.copyfile(fb, fin / name)
    dec["final_blob"] = {"path": str(fin / name) if fa != "today" else str(fb), "name": name, "sha256": sha256(fb),
                         "bytes": fb.stat().st_size, "source": str(fb)}
    dec["decided_at"] = now()
    (ctx.out / tier / "decision.json").write_text(json.dumps(dec, indent=1) + "\n")
    print(f"{tier}: decided {fa} ({name}, {fb.stat().st_size} B)", flush=True)
    return dec


def run_tier(ctx, tier, blob, next_blob):
    nt = NEXT[tier]
    if nt and not (ctx.out / nt / "decision.json").exists():
        if NEXT[nt] is None or ctx.frozen_only:
            if not next_blob:
                raise SystemExit(f"{tier} needs {nt}: give --next-blob")
            prepare(ctx, nt, next_blob)
            if not ctx.frozen_only:
                decide(ctx, nt)
        else:
            raise SystemExit(f"{tier} needs {nt} decided first: run {nt} (with its next tier's blob) before {tier}")
    elif nt and next_blob:
        cmd_input(ctx, nt, next_blob)
    prepare(ctx, tier, blob)
    return decide(ctx, tier)


# ---------------------------------------------------------------- result, timing, the VOCR5 check

def compact_compare(c):
    return {k: {"delta_points": v["delta_points"], "ci95": v["ci95"]} for k, v in c.items()
            if isinstance(v, dict) and "ci95" in v}


def summarise_tier(ctx, tier):
    dec = json.loads((ctx.out / tier / "decision.json").read_text())
    inp = ctx.input(tier)
    s = {"input": {k: inp[k] for k in ("blob", "sha256", "bytes", "n_classes", "attention")},
         "gate_adopted": dec["gate_final"], "two_level_adopted": dec["two_final"], "final_arm": dec["final_arm"],
         "final_blob": dec["final_blob"], "arms": {}}
    for part in ("gate", "two_level"):
        for arm, v in dec[part].items():
            fr = v["frozen"]
            a = {"adopted": v["adopted"], "frozen_verdict": fr["verdict"], "blob_sha256": fr["blob_sha256"],
                 "fails": fr.get("fails")}
            if part == "gate":
                a["decodes"] = {d: {"score_ungated": fr["decodes"][d]["score_x"],
                                    "score_gated": fr["decodes"][d]["score_y"],
                                    **compact_compare(fr["decodes"][d]["compare"]),
                                    "changed": fr["decodes"][d]["changed"], "rule": fr["decodes"][d]["rule"]}
                                for d in DECODES + REREAD}
                a["skip"] = {g: fr["skip"][g] for g in ("all", "wd") if g in fr["skip"]}
                a["saving_estimate_pld0"] = fr["estimate"]["pld0"]["saving"]
            else:
                a["K"] = fr["K"]
                a["vs"] = {ref: {d: {"score_ref": x[d]["score_x"], "score_arm": x[d]["score_y"],
                                     **compact_compare(x[d]["compare"]), "rare_recall": x[d]["rare_recall"]["combined"],
                                     "rule": x[d]["rule"]} for d in DECODES} for ref, x in fr["vs"].items()}
                a["escape_wd"] = fr["escape"]["wd"]
                a["saving_estimate"] = {k: fr["estimate"][k] for k in ("measured", "projected",
                                                                       "saved_pct_nominal_box_safe_min")}
            if v["path"]:
                a["vocr6_path"] = {"check": v["path"]["check"],
                                   "rubi": compact_compare(v["path"]["rubi"]["total"]),
                                   "rubi_platforms": {k: {p: v["path"]["rubi"]["total"][k].get(p)
                                                          for p in ("vita", "psp")}
                                                      for k in ("hval_page", "hval_clean")},
                                   "greedy": compact_compare(v["path"]["greedy"]["total"]),
                                   "x_summary": v["path"]["x"]["summary"]["rubi"],
                                   "y_summary": v["path"]["y"]["summary"]["rubi"],
                                   "cascade": v["path"]["y"]["cascade"]}
                if "stage4_plans_report" in v["path"]:
                    r4 = v["path"]["stage4_plans_report"]
                    a["vocr6_path"]["stage4_plans_report"] = {"check": r4["check"],
                                                              "rubi": compact_compare(r4["rubi"]["total"]),
                                                              "x_summary": r4["x_summary"]["rubi"],
                                                              "y_summary": r4["y_summary"]["rubi"]}
            s["arms"][arm] = a
    s["estimate"] = dec["estimate"]
    return s


def cmd_result(a):
    ctx = Ctx(a.out)
    res = {"kind": "vocr6-gate-result", "issue": 125, "protocol": "eval/vocr6_gate_protocol.json",
           "protocol_sha256": sha256(PROTOCOL), "config": ctx.cfg, "created": now(), "repo_head": git_head(),
           "estimates_note": "every millisecond is an estimate (vita/a9_layers.py under qemu x 1.000676, VOCR3-H15's "
                             "measured console rate, 333 MHz; nothing ran on a console)", "tiers": {}}
    for t in ORDER:
        if (ctx.out / t / "decision.json").exists():
            res["tiers"][t] = summarise_tier(ctx, t)
    res["steps"] = timing(ctx)
    txt = json.dumps(res, indent=1, ensure_ascii=False) + "\n"
    if re.search(r"[A-Z0-9_]+_(DONE|BLOCKED)\b", txt):
        raise SystemExit("the result holds a completion-style token")
    Path(a.result).parent.mkdir(parents=True, exist_ok=True)
    Path(a.result).write_text(txt, encoding="utf-8")
    print(one_line(res), flush=True)


def one_line(res):
    parts = []
    for t in TIERS:
        if t not in res["tiers"]:
            continue
        s = res["tiers"][t]
        e = s["estimate"]
        parts.append(f"{t}: gate {'y' if s['gate_adopted'] else 'n'}" +
                     (f", 2L {'y' if s['two_level_adopted'] else 'n'}" if t in TWO_LEVEL_TIERS else "") +
                     f", est. box {e['today']['pld0']['nominal_box_ms_333']:.0f} -> "
                     f"{e['final']['pld0']['nominal_box_ms_333']:.0f} ms @333")
    return "; ".join(parts)


def timing(ctx):
    rows = []
    for t in ("dev",) + TIERS:
        d = ctx.out / t / "steps"
        if d.exists():
            rows += [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]
        dp = ctx.out / t / "decision.json"
        if dp.exists():               # the decision phase: the rules' scoring, the VOCR6 path's scores, the estimates
            dec = json.loads(dp.read_text())
            t0 = datetime.datetime.strptime(dec["at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
            end = dec.get("decided_at") or datetime.datetime.fromtimestamp(
                dp.stat().st_mtime, datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            t1 = datetime.datetime.strptime(end, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=datetime.timezone.utc)
            wall = (t1 - t0).total_seconds()
            # the steps a decision ran itself (a fallback's reads, A9 and dumps) are rows of their own: not twice
            inner = sum(r["wall_s"] for r in rows if r["tier"] == t and r["start"] >= dec["at"] and r["end"] <= end)
            own = max(0.0, wall - inner)
            rows.append({"tier": t, "step": "decide", "start": dec["at"], "end": end, "wall_s": round(own, 1),
                         "slot_hours": round(own / 3600, 4), "wall_s_with_inner_steps": wall})
    tot = {t: round(sum(r["slot_hours"] for r in rows if r["tier"] == t), 3) for t in ("dev",) + TIERS}
    return {"rows": [{k: r[k] for k in ("tier", "step", "start", "end", "wall_s", "slot_hours")} for r in rows],
            "slot_hours": tot, "slot_hours_total": round(sum(tot.values()), 3)}


def cmd_timing(a):
    ctx = Ctx(a.out)
    t = timing(ctx)
    for r in t["rows"]:
        print(f"{r['tier']} {r['step']:<28} {r['wall_s']:>8.1f} s  {r['slot_hours']:.3f} slot-h")
    print(json.dumps({"slot_hours": t["slot_hours"], "total": t["slot_hours_total"]}))


def _eq(a, b, tol=0.0):
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= tol
    return a == b


def cmd_check_vocr5(a):
    """Step 2: the run on the released v0.3.0 blobs (--prefix VOCR5 --all-vns --frozen-only) against #120's and
    #122's artifacts and committed records: blobs byte for byte, fit records, development scores, A9 savings."""
    ctx = Ctx(a.out)
    rep = {"kind": "vocr6-gate-vocr5-check", "out": str(ctx.out), "created": now(), "blobs": {}, "fits": {}, "dev": {},
           "a9": {}}
    for t in TIERS:
        if not (ctx.out / t / "decision.json").exists():
            continue
        rel = {f"{ctx.prefix}_{t}_w8g.vocr": REF["gate_release"] / f"VOCR5_{t}_w8g.vocr",
               f"{ctx.prefix}_{t}_w8g_eps0.0001.vocr": REF["gate_blobs"] / f"VOCR5_{t}_w8g_eps0.0001.vocr"}
        if t in TWO_LEVEL_TIERS:
            rel.update({f"{ctx.prefix}_{t}_perm.vocr": REF["two_blobs"] / f"VOCR5_{t}_perm.vocr",
                        f"{ctx.prefix}_{t}_2L_K500_r96.vocr": REF["two_blobs"] / f"VOCR5_{t}_2L_K500_r96.vocr",
                        f"{ctx.prefix}_{t}_2Lg_K500_r96.vocr": REF["two_release"] / f"VOCR5_{t}_w8g2.vocr",
                        f"time_{t}_2L_tinf.vocr": REF["two_blobs"] / f"time_{t}_2L_K500_r96_tinf.vocr",
                        f"time_{t}_2L_t-inf.vocr": REF["two_blobs"] / f"time_{t}_2L_K500_r96_t-inf.vocr"})
        for mine, ref in rel.items():
            p = ctx.out / t / "blobs" / mine
            rep["blobs"][f"{t}/{mine}"] = {"ref": str(ref), "exists": p.exists(),
                                           "identical": (p.exists() and ref.exists() and
                                                         p.read_bytes() == ref.read_bytes()),
                                           "sha256": sha256(p) if p.exists() else None,
                                           "ref_sha256": sha256(ref) if ref.exists() else None}
        gf = json.loads(REF["gate_fit"].read_text())["tiers"][t]
        mf = json.loads((ctx.out / t / "fit_gate.json").read_text())
        fit = {"calibration": {k: [mf["calibration"][k], gf["calibration"][k]] for k in gf["calibration"]},
               "gates": {e: {k: [mf["gates"][e][k], g[k]] for k in g if k in mf["gates"][e]}
                         for e, g in gf["gates"].items()}}
        fit["equal"] = all(_eq(*v) for v in fit["calibration"].values()) and \
            all(_eq(*v) for g in fit["gates"].values() for v in g.values())
        rep["fits"][f"{t}/gate"] = fit
        if t in TWO_LEVEL_TIERS:
            tf = json.loads(REF["two_fit"].read_text())["tiers"][t]
            m2 = json.loads((ctx.out / t / "fit_2l.json").read_text())
            esc = {K: {k: [m2["fits"][K].get(k), v] for k, v in o.items() if k not in ("seconds",)}
                   for K, o in tf["escapes_r96"].items()}
            rep["fits"][f"{t}/escapes"] = {"K_alt": [m2["K_alt"], tf["K_alt"]], "escapes": esc,
                                           "equal": m2["K_alt"] == tf["K_alt"] and
                                           all(_eq(*v) for o in esc.values() for v in o.values())}
        dec = json.loads((ctx.out / t / "decision.json").read_text())
        gd = json.loads(REF["gate_dev"].read_text())["tiers"][t]
        mine = dec["gate"]["gate"]["frozen"]
        dv = {}
        for d in DECODES + REREAD:
            r, m = gd["decodes"][d], mine["decodes"][d]
            dv[d] = {"score_today": [m["score_x"], r["score_today"]], "score_gated": [m["score_y"], r["score_gated"]],
                     "score_delta": [m["compare"]["score"], r["compare"]["score"]],
                     **{c: [{k: m["compare"][c][k] for k in ("delta_points", "ci95")},
                            {k: r["compare"][c][k] for k in ("delta_points", "ci95")}] for c in ("tv", "dn", "wd")},
                     "changed": [m["changed"], r["changed"]], "rule": [m["rule"], r["rule"]]}
            dv[d]["equal"] = (_eq(m["score_x"], r["score_today"], 5e-5) and _eq(m["score_y"], r["score_gated"], 5e-5)
                              and all(v[0] == v[1] for k, v in dv[d].items() if k in ("score_delta", "tv", "dn", "wd",
                                                                                       "changed")))
        rep["dev"][f"{t}/gate"] = {"verdict": [mine["verdict"], gd["verdict"]], "decodes": dv,
                                   "equal": mine["verdict"] == gd["verdict"] and all(x["equal"] for x in dv.values())}
        ga = json.loads(REF["gate_a9"].read_text())["tiers"][t]["runs"]
        rep["a9"][f"{t}/gate"] = {f"pld{p}": {"dev_wd": [mine["estimate"][f"pld{p}"]["saving"]["dev_wd"],
                                                         ga[f"pld{p}"]["saving"]["dev_wd"]],
                                              "kappa": [mine["estimate"][f"pld{p}"]["kappa"], ga[f"pld{p}"]["kappa"]]}
                                  for p in (0, 1)}
        if t in TWO_LEVEL_TIERS and dec["two_level"]:
            td = json.loads(REF["two_dev"].read_text())
            ref_arm = td["development"]["tiers"][t]["arms"]["2Lg_Kalt_r96"]
            arm = next(iter(dec["two_level"]))
            mine = dec["two_level"][arm]["frozen"]
            names = {"today": "today", "gate": "gate"}
            vs = {}
            for ref, rname in names.items():
                for d in DECODES:
                    m, r = mine["vs"][ref][d], ref_arm["vs"][rname][d]
                    vs[f"{ref}/{d}"] = {"score": [m["compare"]["score"], r["dev"]["score"]],
                                        "rare": [{k: m["rare_recall"]["combined"][k] for k in ("loss_points", "ci95")},
                                                 {k: r["rare_recall"]["combined"][k] for k in ("loss_points", "ci95")}],
                                        "rule": [m["rule"], r["rule"]]}
                    vs[f"{ref}/{d}"]["equal"] = all(v[0] == v[1] for v in vs[f"{ref}/{d}"].values())
            rd = td["decision"]["tiers"][t]
            rep["dev"][f"{t}/two_level"] = {"arm": arm,
                                            "verdict": [mine["verdict"], rd["arms"]["2Lg_Kalt_r96"]["verdict"]],
                                            "vs": vs, "equal": all(v["equal"] for v in vs.values()) and
                                            mine["verdict"] == rd["arms"]["2Lg_Kalt_r96"]["verdict"]}
            rt = td["timing"]["tiers"][t]["arms"]["2Lg_K500_r96"]
            rep["a9"][f"{t}/two_level"] = {
                "measured_pct": [mine["estimate"]["measured"]["saved_pct_nominal_box"],
                                 rt["measured"]["nopld"]["saved_pct_nominal_box"]],
                "projected_pct": [mine["estimate"]["projected"]["saved_pct_nominal_box"],
                                  rt["projected"]["saved_pct_nominal_box"]]}
    sd = ctx.out / "S" / "decision.json"
    if sd.exists() and "shipped_cascade" in json.loads(sd.read_text()):
        mine = json.loads(sd.read_text())["shipped_cascade"]
        ref = json.loads(REF["two_dev"].read_text())["mixed_cascade"]
        rep["dev"]["S/shipped_cascade"] = {
            "scores": [mine["scores"], ref["scores"]],
            "vs_frozen_gate": [mine["shipped_vs"]["frozen"]["score"], ref["shipped_vs"]["gate"]["score"]],
            "vs_today": [mine["shipped_vs"]["today"]["score"], ref["shipped_vs"]["today"]["score"]],
            "equal": (abs(mine["scores"]["shipped"] - ref["scores"]["shipped"]) < 5e-6 and
                      mine["shipped_vs"]["today"]["score"] == ref["shipped_vs"]["today"]["score"])}
    rep["all_blobs_identical"] = all(v["identical"] for v in rep["blobs"].values())
    rep["all_fits_equal"] = all(v["equal"] for v in rep["fits"].values())
    rep["all_dev_equal"] = all(v["equal"] for v in rep["dev"].values())
    Path(a.record).write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: rep[k] for k in ("all_blobs_identical", "all_fits_equal", "all_dev_equal")}))
    for k, v in rep["blobs"].items():
        if not v["identical"]:
            print("blob differs:", k, v["sha256"], v["ref_sha256"])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "prepare", "decide", "init"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        if name != "init":
            p.add_argument("--tier", required=True, choices=TIERS)
        if name in ("run", "prepare"):
            p.add_argument("--blob", required=True)
        if name == "run":
            p.add_argument("--next-blob", default="")
        p.add_argument("--priv", default="", help="private outputs (features, reads, dumps); default "
                                                  "$HERD/shared/private/vocr6-final-gate/<OUT's name>")
        p.add_argument("--prefix", default="VOCR6", help="the blobs' names (VOCR5 for the reproduction)")
        p.add_argument("--all-vns", action="store_true", help="score every VN (the reproduction of #120 / #122)")
        p.add_argument("--frozen-only", action="store_true", help="#120's / #122's rules only, no VOCR6 path")
        p.add_argument("--path-lm-hval", default=str(PATH_LM_HVAL), help="the VOCR6 path's hval LM (amendment 1)")
        p.add_argument("--path-lm-dev", default=str(PATH_LM_DEV), help="the VOCR6 path's dev LM (amendment 1)")
        p.add_argument("--path-lm-opts", default=PATH_LM_OPTS)
        p.add_argument("--no-plans4", action="store_true", help="skip the report on vocr6-lead's stage-4 plans")
        p.add_argument("--procs", type=int, default=5)
    p = sub.add_parser("result")
    p.add_argument("--out", required=True)
    p.add_argument("--result", required=True)
    p = sub.add_parser("timing")
    p.add_argument("--out", required=True)
    p = sub.add_parser("check-vocr5")
    p.add_argument("--out", required=True)
    p.add_argument("--record", required=True)
    a = ap.parse_args()
    if a.cmd in ("run", "prepare", "decide", "init"):
        ctx = Ctx(a.out, a)
        if a.cmd == "init":
            ensure_dev_lines(ctx)
            print(json.dumps(ctx.cfg))
        elif a.cmd == "run":
            run_tier(ctx, a.tier, a.blob, a.next_blob)
        elif a.cmd == "prepare":
            prepare(ctx, a.tier, a.blob)
        else:
            decide(ctx, a.tier)
    else:
        {"result": cmd_result, "timing": cmd_timing, "check-vocr5": cmd_check_vocr5}[a.cmd](a)


if __name__ == "__main__":
    main()
