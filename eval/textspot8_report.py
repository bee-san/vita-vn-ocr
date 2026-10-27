#!/usr/bin/env python3
"""textspot8's report tables (docs/textspot8.md): summaries, paired intervals and per-VN spreads of eval/pagepath.py
results, the way textspot6 reports them (eval/textspot6_select.py's intervals: eval/vocr3_report.cluster_bootstrap,
10,000 resamples, seed 0, VN clusters on dev, title clusters on hval).

  textspot8_report.py table --spec SPEC.json --out OUT.json [--md OUT.md]
      SPEC: {"baselines": {NAME: {"dev": RESULT, "hval": RESULT}}, "rows": {NAME: {"dev": RESULT, "hval": RESULT}},
             "against": [BASELINE NAMES], "seen_dev_clusters": ["v12188"]}
      per row: dev CER (the seen clusters left out: the headline) and the full set, Vita / PSP, hval CER, line recall
      (whole lines / lines), box precision (boxes that are not junk / boxes: pagepath's), their F1, and against every
      baseline in "against": the paired difference with its interval, VNs better / worse / level, the largest gains
      and losses, and the difference without the VN with the largest gain (report only)

RESULT files hold counts only; OUT holds counts, CERs and VN ids only.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import vocr3_report as VR  # noqa: E402

SEEN_DEV = ("v12188",)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def units_of(res, exclude=()):
    return [u for u in res["units"] if u["cluster"] not in exclude]


def summary(units):
    t = defaultdict(int)
    for u in units:
        for k in ("errors", "chars", "lines", "boxes", "whole", "cut", "missed", "merged", "junk"):
            t[k] += u[k]
    rec = t["whole"] / t["lines"] if t["lines"] else 0.0
    prec = (t["boxes"] - t["junk"]) / t["boxes"] if t["boxes"] else 0.0
    f1 = 2 * rec * prec / (rec + prec) if rec + prec else 0.0
    return {"units": len(units), "chars": t["chars"], "errors": t["errors"],
            "cer": round(100 * t["errors"] / max(1, t["chars"]), 4), "lines": t["lines"], "whole": t["whole"],
            "cut": t["cut"], "missed": t["missed"], "merged": t["merged"], "junk": t["junk"], "boxes": t["boxes"],
            "line_recall": round(rec, 4), "box_precision": round(prec, 4), "f1": round(f1, 4)}


def by_platform(units):
    out = {}
    for p in sorted({u["platform"] for u in units}):
        out[p] = summary([u for u in units if u["platform"] == p])
    return out


def pair(arm_units, base_units):
    a = {u["uid"]: (u["errors"], u["chars"]) for u in arm_units}
    b = {u["uid"]: (u["errors"], u["chars"]) for u in base_units}
    if set(a) != set(b) or any(a[k][1] != b[k][1] for k in a):
        raise SystemExit("pair: the two results hold different units or references")
    cl = {u["uid"]: u["cluster"] for u in arm_units}
    ea, eb, n = sum(v[0] for v in a.values()), sum(v[0] for v in b.values()), sum(v[1] for v in a.values())
    out = VR.cluster_bootstrap(a, b, cl)
    out["delta"] = round(100 * (ea - eb) / max(1, n), 4)
    return out


def spread(arm_units, base_units, top=5):
    per = defaultdict(lambda: [0, 0, 0])
    bu = {u["uid"]: u for u in base_units}
    for u in arm_units:
        s = per[u["cluster"]]
        s[0] += bu[u["uid"]]["errors"]
        s[1] += u["errors"]
        s[2] += u["chars"]
    diffs = sorted(((v[1] - v[0], k, v[0], v[1]) for k, v in per.items()))
    gains = [[k, b, a] for d, k, b, a in diffs if d < 0][:top]
    losses = [[k, b, a] for d, k, b, a in reversed(diffs) if d > 0][:top]
    better = sum(1 for d, *_ in diffs if d < 0)
    worse = sum(1 for d, *_ in diffs if d > 0)
    out = {"vns": len(per), "better": better, "worse": worse, "level": len(per) - better - worse,
           "gains": gains, "losses": losses}
    if gains:
        top_vn = gains[0][0]
        out["without_top"] = {"vn": top_vn, "diff": pair([u for u in arm_units if u["cluster"] != top_vn],
                                                          [u for u in base_units if u["cluster"] != top_vn])}
    return out


def per_vn(res, exclude=()):
    per = defaultdict(lambda: [0, 0, 0])
    for u in units_of(res, exclude):
        s = per[u["cluster"]]
        s[0] += u["errors"]
        s[1] += u["chars"]
        s[2] += 1
    return {k: {"errors": v[0], "chars": v[1], "units": v[2]} for k, v in sorted(per.items())}


def row(res_dev, res_hval, seen=SEEN_DEV):
    out = {}
    if res_dev is not None:
        du = units_of(res_dev, seen)
        out["dev"] = summary(du)
        out["dev_by_platform"] = by_platform(du)
        out["dev_all"] = summary(units_of(res_dev))
    if res_hval is not None:
        hu = units_of(res_hval)
        out["hval"] = summary(hu)
        out["hval_by_platform"] = by_platform(hu)
    return out


def cmd_table(args):
    spec = load(args.spec)
    seen = tuple(spec.get("seen_dev_clusters", SEEN_DEV))
    res = {}
    loaded = {}

    def get(p):
        if p not in loaded:
            loaded[p] = load(p) if p and Path(p).exists() else None
        return loaded[p]
    allrows = dict(spec.get("baselines", {}))
    allrows.update(spec.get("rows", {}))
    for name, paths in allrows.items():
        d, h = get(paths.get("dev")), get(paths.get("hval"))
        r = row(d, h, seen)
        for bname in spec.get("against", []):
            if bname == name or bname not in spec.get("baselines", {}):
                continue
            bd, bh = get(spec["baselines"][bname].get("dev")), get(spec["baselines"][bname].get("hval"))
            vs = {}
            if d is not None and bd is not None:
                vs["dev"] = pair(units_of(d, seen), units_of(bd, seen))
                vs["dev_spread"] = spread(units_of(d, seen), units_of(bd, seen))
            if h is not None and bh is not None:
                vs["hval"] = pair(units_of(h), units_of(bh))
            r[f"vs_{bname}"] = vs
        res[name] = r
    out = {"kind": "textspot8-report", "seen_dev_clusters": list(seen), "spec": spec, "rows": res}
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.md:
        Path(args.md).write_text(markdown(out), encoding="utf-8")
    print(f"textspot8 report written: {args.out}", flush=True)


def _ci(v):
    if not v or v.get("ci95") is None:
        return ""
    return f"{v['delta']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]"


def markdown(out):
    against = out["spec"].get("against", [])
    head = ["row", "dev CER", "dev Vita / PSP", "hval CER", "dev line R / box P / F1", "hval line R / box P / F1"]
    head += [f"dev - {b}" for b in against] + [f"hval - {b}" for b in against]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, r in out["rows"].items():
        d, h = r.get("dev"), r.get("hval")
        dp = r.get("dev_by_platform", {})
        cells = [name, f"{d['cer']:.2f}" if d else "",
                 " / ".join(f"{dp[p]['cer']:.2f}" for p in ("vita", "psp") if p in dp),
                 f"{h['cer']:.2f}" if h else "",
                 f"{d['line_recall']:.3f} / {d['box_precision']:.3f} / {d['f1']:.3f}" if d else "",
                 f"{h['line_recall']:.3f} / {h['box_precision']:.3f} / {h['f1']:.3f}" if h else ""]
        cells += [_ci(r.get(f"vs_{b}", {}).get("dev")) for b in against]
        cells += [_ci(r.get(f"vs_{b}", {}).get("hval")) for b in against]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("table")
    t.add_argument("--spec", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--md")
    args = ap.parse_args(argv)
    {"table": cmd_table}[args.cmd](args)


if __name__ == "__main__":
    main()
