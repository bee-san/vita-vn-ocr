#!/usr/bin/env python3
"""textspot2's pre-registered selection rule (issue #106, docs/textspot2.md, eval/textspot2_protocol.json "rule").

Frozen with the protocol, before any textspot2 run or score. Pure functions over the arm summaries of textspot's
frozen metrics (eval/textspot_metrics.py: arm_summary, compare, beats, vn_diff, f1_diff); nothing here reads a
file. eval/textspot2.py (the scoring wrapper) builds the candidate records and calls apply_rule2 once.

  median_seed(runs)       an arm's representative run: the median of its finished seeds by development all-text CER
                          (sorted by all-text CER, dialogue CER, -F1, seed; index n // 2, so the worse of the two
                          middle seeds when n is even); never the best seed
  better(cmp)             resolved better on all-text CER or F1, and resolved worse on none of all-text CER, F1 and
                          dialogue CER (VN-cluster 95% intervals of textspot_metrics.compare)
  eligible(c, rule)       VTSP bytes, arena at 960 x 544, est. ms per 960 x 544 frame at 333 MHz, C == twin
  cost_key(c)             est. ms, then VTSP bytes, then arena, then det before det+refine, then the name
  chain(qual)             the cheapest qualifier, replaced by the cheapest costlier qualifier resolved better than
                          it (better()), repeated until none is
  apply_rule2(...)        the main pick (budget est. <= 150 ms) and the fast pick (est. <= 80 ms) with every
                          comparison
  dnc_units(frames)       the secondary column's units: dialogue + name + choice references only, no junk

A candidate record (eval/textspot2.py): {"name", "arm_id", "seed", "decode" ("det" | "det+refine"), "blob_bytes",
"arena_960", "est_ms_333_960", "c_equal", "arm": textspot_metrics.arm_summary(...)}; a reference ("ref", the
textspot winner; "fast_ref", textspot's fastest qualifier) is {"name", "arm"}.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "eval") not in sys.path:
    sys.path.insert(0, str(ROOT / "eval"))
import textspot_metrics as M  # noqa: E402
from metrics import norm  # noqa: E402

DNC = ("dialogue", "name", "choice")      # the secondary column's classes (report only)


# ----------------------------------------------------------------------------- seeds

def median_index(n):
    """Index of an arm's representative among its n finished seeds sorted best first: n // 2 (n = 3: the middle
    one; n = 2: the worse one; n = 1: that one)."""
    if n < 1:
        raise ValueError("no finished seed")
    return n // 2


def seed_key(r):
    """Sort key of a finished seed, best first: all-text CER, dialogue CER, -F1, seed."""
    return (r["cer_all"], r["cer_dialogue"], -r["f1"], r["seed"])


def median_seed(runs):
    """runs: [{"seed", "cer_all", "cer_dialogue", "f1", ...}] of one arm and decode (development, VOCR5-N) -> the
    representative run (median_index of the sorted list) and the order {"order": [seeds best first], "index"}."""
    rs = sorted(runs, key=seed_key)
    i = median_index(len(rs))
    return rs[i], {"order": [r["seed"] for r in rs], "index": i}


# ----------------------------------------------------------------------------- comparisons

def resolved_above(ci):
    return bool(ci) and ci[0] > 0


def resolved_below(ci):
    return bool(ci) and ci[1] < 0


def better(cmp):
    """cmp: textspot_metrics.compare(candidate, reference). Resolved better on all-text CER (interval below 0) or
    on F1 (interval above 0), and resolved worse on none: F1 not below 0, all-text CER not above 0, dialogue CER
    not above 0. An interval that could not be formed (too few clusters) resolves nothing."""
    f, e, d = cmp["f1"]["ci95"], cmp["cer_all"]["ci95"], cmp["cer_dialogue"]["ci95"]
    gain = resolved_above(f) or resolved_below(e)
    loss = resolved_below(f) or resolved_above(e) or resolved_above(d)
    return bool(gain and not loss)


def eligible(c, rule, ms_max=None):
    """Within the budget: VTSP bytes <= vtsp_max, arena <= arena_max, est. ms <= ms_max (default the main
    budget, ms_max_333_960) and the C runtime equal to the twin on every development frame."""
    ms_max = rule["ms_max_333_960"] if ms_max is None else ms_max
    return bool(c["blob_bytes"] <= rule["vtsp_max"] and c["arena_960"] is not None and
                c["arena_960"] <= rule["arena_max"] and c["est_ms_333_960"] is not None and
                c["est_ms_333_960"] <= ms_max and c["c_equal"] is True)


def cost_key(c):
    return (c["est_ms_333_960"], c["blob_bytes"], c["arena_960"], c["decode"] != "det", c["name"])


def chain(qual):
    """The cheapest qualifier W; the cheapest qualifier costlier than W that is better() than W replaces it;
    repeated until none is. -> (winner record or None, steps)."""
    q = sorted(qual, key=cost_key)
    if not q:
        return None, []
    w, steps = q[0], []
    while True:
        nxt = None
        for c in q:
            if cost_key(c) <= cost_key(w):
                continue
            cmp = M.compare(c["arm"], w["arm"])
            if better(cmp):
                nxt = c
                steps.append({"replaces": w["name"], "by": c["name"], "comparison": cmp})
                break
        if nxt is None:
            return w, steps
        w = nxt


# ----------------------------------------------------------------------------- the rule

def apply_rule2(cands, baselines, ref, fast_ref, rule):
    """cands: the median-seed candidate records; baselines: {name: arm_summary} (textspot's frozen baseline, read
    by VOCR5-N); ref: the textspot winner; fast_ref: textspot's fastest qualifier (the fast pick's tier-2
    comparator); rule: eval/textspot2_protocol.json "rule" (vtsp_max, arena_max, ms_max_333_960,
    fast_ms_max_333_960).

    Every eligible candidate is compared with every baseline (textspot's bar: textspot_metrics.beats against all)
    and with ref (better()). Main pick: the chain over the eligible candidates that clear the bar and are better
    than ref. Fast pick: the same over those with est. ms <= fast_ms_max_333_960 (tier 1); when none of them is
    better than ref, the same with fast_ref as the comparator (tier 2); else none."""
    elig = sorted((c for c in cands if eligible(c, rule)), key=cost_key)
    comps = {}
    for c in elig:
        bars = {b: M.compare(c["arm"], a) for b, a in sorted(baselines.items())}
        vs_ref = M.compare(c["arm"], ref["arm"])
        comps[c["name"]] = {"baselines": bars, "bar": all(M.beats(x) for x in bars.values()), "vs_ref": vs_ref,
                            "better_than_ref": better(vs_ref)}
    main_q = [c for c in elig if comps[c["name"]]["bar"] and comps[c["name"]]["better_than_ref"]]
    main, main_steps = chain(main_q)
    fast_pool = [c for c in elig if c["est_ms_333_960"] <= rule["fast_ms_max_333_960"]]
    fast_q = [c for c in fast_pool if comps[c["name"]]["bar"] and comps[c["name"]]["better_than_ref"]]
    tier = 1 if fast_q else None
    if not fast_q:
        for c in fast_pool:
            v = M.compare(c["arm"], fast_ref["arm"])
            comps[c["name"]]["vs_fast_ref"] = v
            comps[c["name"]]["better_than_fast_ref"] = better(v)
        fast_q = [c for c in fast_pool if comps[c["name"]]["bar"] and comps[c["name"]]["better_than_fast_ref"]]
        tier = 2 if fast_q else None
    fast, fast_steps = chain(fast_q)
    return {"eligible": [c["name"] for c in elig],
            "bar": [c["name"] for c in elig if comps[c["name"]]["bar"]],
            "main": {"qualifying": [c["name"] for c in main_q], "winner": main["name"] if main else None,
                     "steps": main_steps, "comparator": ref["name"]},
            "fast": {"pool": [c["name"] for c in fast_pool], "qualifying": [c["name"] for c in fast_q],
                     "tier": tier, "winner": fast["name"] if fast else None, "steps": fast_steps,
                     "comparator": (ref["name"] if tier == 1 else fast_ref["name"] if tier == 2 else None)},
            "comparisons": comps}


# ----------------------------------------------------------------------------- the secondary column

def dnc_units(frames):
    """frames: [{"unit", "vn", "refs": [{"cls", "ignore", "text"}], "per_ref": e2e_frame's per_ref}] -> units
    {"unit", "vn", "errors", "chars"} over the dialogue, name and choice references only (junk not counted):
    the secondary column (report only), comparable with textspot_metrics.vn_diff."""
    out = []
    for f in frames:
        e = n = 0
        for r, err in zip(f["refs"], f["per_ref"]):
            if r.get("ignore") or r["cls"] not in DNC:
                continue
            e += err
            n += len(norm(r["text"]))
        out.append({"unit": f["unit"], "vn": f["vn"], "errors": e, "chars": n})
    return out
