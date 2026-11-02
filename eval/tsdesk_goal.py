#!/usr/bin/env python3
"""textspot-desktop round 2's goal as code (issue #173, eval/textspot_desktop_protocol_v2.json "rule").

  tsdesk_goal.py check --spec SPEC.json [--out GOAL.json]

SPEC names the stored results (eval/desktop.py read results; a timing record) of one candidate:
  {"candidate": "NAME", "policy": "long:960",
   "reads": {"dev":            {"cand": DEV.json,  "R": R.json,  "T8": T8.json},
             "dev_l544":       {"cand": ..., "R": ..., "T8": ...},
             "dense_dev":      {"cand": ..., "R": ...},
             "dense_dev_l544": {"cand": ..., "R": ...},
             "hval":           {"cand": ..., "R": ...}},
   "timing": TIMING.json}
TIMING: {"det_1t": {"cand": {"vita": p50_ms}, "meiki": {"vita": p50_ms}},
         "page_4t": {"cand": {"vita": ms, "psp": ms, "pc": ms}, "R": {"vita": ms, "psp": ms, "pc": ms}},
         "interleaved": true, ...}  (eval/tsdesk_time.py's goal timing writes it; any extra keys are reported)
A missing read or timing leaves its clause "unread" (not passed). The gap is max over the four G1 sets of
(candidate CER - R CER), in points: the goal's accuracy part is met when it is <= 0 and every interval clause holds.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "tools", ROOT / "synth", ROOT / "train", ROOT / "teacher", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

G1_SETS = ("dev", "dense_dev", "dev_l544", "dense_dev_l544")
VIEW_SETS = ("dev", "dev_l544")
SPEED_X = 5.0          # G2: the detector at least 5 times faster than MeikiOCR's at 1 thread, 960x544
PAGE_FRAC = 1.0 / 3.0  # G3: the page at 4 threads at most a third of R's, on every timing view
PAGE_VIEWS = ("vita", "psp", "pc")
GUARD_CLUSTER = "PCSG00854"


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def cer(doc):
    e, c = sum(u["errors"] for u in doc["units"]), sum(u["chars"] for u in doc["units"])
    return 100.0 * e / max(1, c)


def not_above_zero(ci):
    """A paired interval not entirely above 0 (None: too few clusters, counts as not resolved)."""
    return ci is None or ci[0] <= 0.0


def _num(x):
    return None if x is None else round(float(x), 4)


def views_of(cmp):
    """{"vita": entry, "psp": entry} of compare_docs' groups: the set's vita and psp views, own size or fit (the
    view names of resampled frames carry '@fitWxH')."""
    out = {}
    for k, v in cmp.items():
        view = k.split(":", 1)[1]
        base = view.split("@", 1)[0]
        if base in ("vita", "psp") and (view == base or view.startswith(base + "@fit")):
            out[base] = v
    return out


def all_of(cmp):
    k = [k for k in cmp if k.endswith(":all")]
    return cmp[k[0]] if len(k) == 1 else None


def g1_set(name, r):
    import desktop as D
    if not r or not r.get("cand") or not r.get("R"):
        return {"state": "unread", "pass": False}
    a, b = load(r["cand"]), load(r["R"])
    cmp = D.compare_docs(a, b)
    whole = all_of(cmp)
    out = {"cand_cer": _num(cer(a)), "R_cer": _num(cer(b)), "delta": _num(whole["delta"]),
           "ci95": [_num(x) for x in whole["ci95"]] if whole.get("ci95") else None}
    out["point_ok"] = cer(a) <= cer(b) + 1e-12
    out["interval_ok"] = not_above_zero(out["ci95"])
    ok = out["point_ok"] and out["interval_ok"]
    if name in VIEW_SETS:
        vs = views_of(cmp)
        out["views"] = {v: {"delta": _num(e["delta"]), "ci95": [_num(x) for x in e["ci95"]] if e.get("ci95") else None,
                            "ok": not_above_zero(e.get("ci95"))} for v, e in sorted(vs.items())}
        ok = ok and len(out["views"]) == 2 and all(v["ok"] for v in out["views"].values())
    out["pass"] = bool(ok)
    return out


def g4(reads):
    import desktop as D
    out = {"junk": {"state": "unread", "pass": False}, "views_vs_T8": {}}
    h = reads.get("hval") or {}
    if h.get("cand") and h.get("R"):
        a, b = load(h["cand"]), load(h["R"])
        ja = sum(u["junk"] for u in a["units"] if u["cluster"] == GUARD_CLUSTER)
        jb = sum(u["junk"] for u in b["units"] if u["cluster"] == GUARD_CLUSTER)
        out["junk"] = {"cand": ja, "R": jb, "pass": ja <= jb}
    ok = out["junk"]["pass"]
    for s in VIEW_SETS:
        r = reads.get(s) or {}
        if not (r.get("cand") and r.get("T8")):
            out["views_vs_T8"][s] = {"state": "unread", "pass": False}
            ok = False
            continue
        vs = views_of(D.compare_docs(load(r["cand"]), load(r["T8"])))
        e = {v: {"delta": _num(x["delta"]), "ci95": [_num(y) for y in x["ci95"]] if x.get("ci95") else None,
                 "ok": not_above_zero(x.get("ci95"))} for v, x in sorted(vs.items())}
        out["views_vs_T8"][s] = dict(e, **{"pass": len(e) == 2 and all(x["ok"] for x in e.values())})
        ok = ok and out["views_vs_T8"][s]["pass"]
    out["pass"] = bool(ok)
    return out


def speed(timing):
    if not timing:
        return {"state": "unread", "pass": False}, {"state": "unread", "pass": False}
    t = load(timing)
    d = t.get("det_1t") or {}
    try:
        c, m = float(d["cand"]["vita"]), float(d["meiki"]["vita"])
        g2 = {"cand_ms": c, "meiki_ms": m, "ratio": round(m / c, 3), "bar_ms": round(m / SPEED_X, 3),
              "pass": c <= m / SPEED_X}
    except (KeyError, TypeError, ZeroDivisionError):
        g2 = {"state": "unread", "pass": False}
    p = t.get("page_4t") or {}
    g3 = {"views": {}}
    try:
        for v in PAGE_VIEWS:
            c, r = float(p["cand"][v]), float(p["R"][v])
            g3["views"][v] = {"cand_ms": c, "R_ms": r, "frac": round(c / r, 3), "pass": c <= r * PAGE_FRAC}
        g3["pass"] = all(x["pass"] for x in g3["views"].values())
    except (KeyError, TypeError, ZeroDivisionError):
        g3 = {"state": "unread", "pass": False}
    for g in (g2, g3):
        g["interleaved"] = bool(t.get("interleaved"))
        g["pass"] = g["pass"] and g["interleaved"]
    return g2, g3


def check(spec):
    reads = spec.get("reads") or {}
    out = {"kind": "textspot-desktop-goal-check", "candidate": spec.get("candidate"), "policy": spec.get("policy"),
           "protocol": "eval/textspot_desktop_protocol_v2.json", "G1": {}}
    for s in G1_SETS:
        out["G1"][s] = g1_set(s, reads.get(s))
    deltas = [v["delta"] for v in out["G1"].values() if v.get("delta") is not None]
    out["gap"] = _num(max(deltas)) if len(deltas) == len(G1_SETS) else None
    out["G1"]["pass"] = all(out["G1"][s]["pass"] for s in G1_SETS)
    out["G2"], out["G3"] = speed(spec.get("timing"))
    out["G4"] = g4(reads)
    out["met_on_dev"] = all(out[g]["pass"] for g in ("G1", "G2", "G3", "G4"))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("check")
    p.add_argument("--spec", required=True)
    p.add_argument("--out")
    a = ap.parse_args(argv)
    out = check(load(a.spec))
    text = json.dumps(out, indent=1, default=float)
    if a.out:
        Path(a.out).write_text(text + "\n", encoding="utf-8")
    print(json.dumps({"candidate": out["candidate"], "gap": out["gap"], "met_on_dev": out["met_on_dev"],
                      **{g: out[g]["pass"] for g in ("G1", "G2", "G3", "G4")}}))


if __name__ == "__main__":
    main()
