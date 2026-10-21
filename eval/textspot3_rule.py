#!/usr/bin/env python3
"""textspot3's pre-registered selection rule (issue: "textspot3: a 500 KB text spotter", docs/textspot3.md,
eval/textspot3_protocol.json "rule").

Frozen with the protocol, before any textspot3 run or score. Pure functions over the arm summaries of textspot's
frozen metrics (eval/textspot_metrics.py: arm_summary, compare, beats, vn_diff, f1_diff); nothing here reads a file.
The scoring wrapper builds the candidate records and calls apply_rule3 once.

  median_seed(runs)       an arm's representative run per decode: the median of its finished seeds by development
                          dialogue CER (sorted by dialogue CER, all-text CER, -F1, seed; index n // 2, so the worse of
                          the two middle seeds when n is even); never the best seed
  qualifies(cmp)          against ref (the textspot winner): dialogue CER resolved better (interval below 0), and
                          neither all-text CER resolved worse (interval above 0) nor F1 resolved worse (below 0)
  replaces(cmp)           a costlier qualifier against the current pick: all-text CER resolved better, and neither
                          dialogue CER resolved worse nor F1 resolved worse
  eligible(c, rule)       VTSP bytes, arena at 960 x 544, est. ms per 960 x 544 frame at 333 MHz, C == twin
  cost_key(c)             VTSP bytes, then est. ms, then arena, then det before det+refine, then the name
  chain(qual)             the cheapest qualifier; the cheapest costlier qualifier that replaces() it takes its place,
                          repeated until none does
  apply_rule3(...)        the pick with every comparison (and textspot's bar against its eight baselines, report
                          only)
  recall_by_class(...)    report: per-class recall pooled over frames (textspot_metrics.match_frame's counts)

A candidate record: {"name", "arm_id", "seed", "decode" ("det" | "det+refine"), "blob_bytes", "arena_960",
"est_ms_333_960", "c_equal", "arm": textspot_metrics.arm_summary(...)}; ref is {"name", "arm"}.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "eval") not in sys.path:
    sys.path.insert(0, str(ROOT / "eval"))
import textspot_metrics as M  # noqa: E402

RULE = {"vtsp_max": 512000, "arena_max": 262144, "ms_max_333_960": 300.0}


# ----------------------------------------------------------------------------- seeds

def median_index(n):
    """Index of an arm's representative among its n finished seeds sorted best first: n // 2 (n = 3: the middle
    one; n = 2: the worse one; n = 1: that one)."""
    if n < 1:
        raise ValueError("no finished seed")
    return n // 2


def seed_key(r):
    """Sort key of a finished seed, best first: dialogue CER, all-text CER, -F1, seed."""
    return (r["cer_dialogue"], r["cer_all"], -r["f1"], r["seed"])


def median_seed(runs):
    """runs: [{"seed", "cer_dialogue", "cer_all", "f1", ...}] of one arm and decode (development, VOCR5-N) -> the
    representative run (median_index of the sorted list) and the order {"order": [seeds best first], "index"}."""
    rs = sorted(runs, key=seed_key)
    i = median_index(len(rs))
    return rs[i], {"order": [r["seed"] for r in rs], "index": i}


# ----------------------------------------------------------------------------- comparisons

def resolved_above(ci):
    return bool(ci) and ci[0] > 0


def resolved_below(ci):
    return bool(ci) and ci[1] < 0


def qualifies(cmp):
    """cmp: textspot_metrics.compare(candidate, ref). Dialogue CER resolved below 0, all-text CER not resolved above
    0, F1 not resolved below 0. An interval that could not be formed resolves nothing (so a missing dialogue
    interval never qualifies)."""
    f, e, d = cmp["f1"]["ci95"], cmp["cer_all"]["ci95"], cmp["cer_dialogue"]["ci95"]
    return bool(resolved_below(d) and not resolved_above(e) and not resolved_below(f))


def replaces(cmp):
    """cmp: textspot_metrics.compare(costlier qualifier, current pick). All-text CER resolved below 0, dialogue CER
    not resolved above 0, F1 not resolved below 0."""
    f, e, d = cmp["f1"]["ci95"], cmp["cer_all"]["ci95"], cmp["cer_dialogue"]["ci95"]
    return bool(resolved_below(e) and not resolved_above(d) and not resolved_below(f))


def eligible(c, rule=RULE):
    """Within the budget: VTSP bytes <= vtsp_max, arena <= arena_max, est. ms <= ms_max_333_960 and the C runtime
    equal to the twin on every development frame."""
    return bool(c["blob_bytes"] is not None and c["blob_bytes"] <= rule["vtsp_max"] and
                c["arena_960"] is not None and c["arena_960"] <= rule["arena_max"] and
                c["est_ms_333_960"] is not None and c["est_ms_333_960"] <= rule["ms_max_333_960"] and
                c["c_equal"] is True)


def cost_key(c):
    return (c["blob_bytes"], c["est_ms_333_960"], c["arena_960"], c["decode"] != "det", c["name"])


def chain(qual):
    """The cheapest qualifier W; the cheapest qualifier costlier than W that replaces() W takes its place; repeated
    until none does. -> (winner record or None, steps)."""
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
            if replaces(cmp):
                nxt = c
                steps.append({"replaces": w["name"], "by": c["name"], "comparison": cmp})
                break
        if nxt is None:
            return w, steps
        w = nxt


# ----------------------------------------------------------------------------- the rule

def apply_rule3(cands, ref, baselines=None, rule=RULE):
    """cands: the median-seed candidate records; ref: the textspot winner {"name", "arm"}; baselines: {name:
    arm_summary} (textspot's frozen baseline, VOCR5-N; report only: textspot's bar is recorded, not required);
    rule: eval/textspot3_protocol.json "rule" (vtsp_max, arena_max, ms_max_333_960).

    Every eligible candidate is compared with ref (qualifies()). The pick: chain() over the qualifiers. No
    qualifier: no pick (the verdict is negative and textspot's winner stands)."""
    elig = sorted((c for c in cands if eligible(c, rule)), key=cost_key)
    comps = {}
    for c in elig:
        vs_ref = M.compare(c["arm"], ref["arm"])
        rec = {"vs_ref": vs_ref, "qualifies": qualifies(vs_ref)}
        if baselines:
            bars = {b: M.compare(c["arm"], a) for b, a in sorted(baselines.items())}
            rec["baselines"] = bars
            rec["bar"] = all(M.beats(x) for x in bars.values())
        comps[c["name"]] = rec
    qual = [c for c in elig if comps[c["name"]]["qualifies"]]
    pick, steps = chain(qual)
    return {"eligible": [c["name"] for c in elig],
            "ineligible": sorted(c["name"] for c in cands if not eligible(c, rule)),
            "qualifying": [c["name"] for c in qual],
            "bar": [c["name"] for c in elig if comps[c["name"]].get("bar")],
            "winner": pick["name"] if pick else None, "steps": steps, "comparator": ref["name"],
            "comparisons": comps}


# ----------------------------------------------------------------------------- report

def recall_by_class(matches):
    """matches: [textspot_metrics.match_frame result] -> {class: {"tp", "n", "recall"}} pooled over the frames."""
    out = {}
    for cls in M.CLASSES:
        tp = sum(m["tp_cls"][cls] for m in matches)
        n = tp + sum(m["fn_cls"][cls] for m in matches)
        out[cls] = {"tp": tp, "n": n, "recall": round(tp / n, 4) if n else None}
    return out
