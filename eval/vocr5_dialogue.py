#!/usr/bin/env python3
"""VOCR5's dialogue-primary fixed-line tables (eval/vocr5_protocol.json "locked_report": dialogue = lines without the
'name' style), from the locked report's stored parts, with Google Lens as a report-only arm (amendment 2).

  vocr5_dialogue.py tables --report REPORT.json [--lens SET=RUN_OUT.json ...] --out DIALOGUE.json [--md TABLES.md]

REPORT.json is eval/vocr3_report.py's rendered report (its "shared.fixed_line[polarity][arm].lines[set]": integer
per-line records with styles). --lens takes eval/run.py --out files of the Lens arm (eval/lens_lines.py serve, native
polarity), one per set. Per arm, set and polarity: dialogue (no 'name' style), names, all text: errors, chars, lines,
exact lines, CER; pooled over vita-test and psp-test. Pairs (the protocol's, plus VOCR5_S/N/L - Lens): CAND - BASE in
points with eval/vocr3_report.py's paired line and VN-cluster bootstraps (10,000 resamples, seed 0, at least 5 VNs),
over the lines both arms read. Lens reads the screen, so its lines pair with the other arms' native-polarity lines and
also, as the protocol's primary view, with their console-polarity lines (the same crops' ground truth). Counts only.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import metrics  # noqa: E402
import vocr3_report as R  # noqa: E402

SETS = ("vita-test", "psp-test")
PARTS = {"dialogue": lambda x: "name" not in x["styles"], "names": lambda x: "name" in x["styles"],
         "all": lambda x: True}
LENS = "Lens"


def tally(lines):
    return {"lines": len(lines), "chars": sum(x["n"] for x in lines), "errors": sum(x["errors"] for x in lines),
            "exact": sum(x["errors"] == 0 for x in lines),
            "cer": round(100 * sum(x["errors"] for x in lines) / max(1, sum(x["n"] for x in lines)), 3)}


def lens_lines(path, set_name):
    """eval/run.py --out of the Lens arm -> line records shaped as vocr3_report's (id, vn, styles, n, errors)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = d["sets"][set_name]["lines"] if "sets" in d else d["lines"]
    man = {json.loads(x)["id"]: json.loads(x) for x in
           (HERE / "manifests" / f"{set_name}.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()}
    out = []
    for r in rows:
        m = man[r["id"]]
        ref = metrics.norm(m["text"])
        pred = metrics.norm(r.get("pred", ""))
        out.append({"id": r["id"], "vn": m["vn"], "styles": sorted(m.get("styles") or []), "n": len(ref),
                    "errors": metrics.lev(pred, ref)})
        if out[-1]["errors"] != r.get("errors", out[-1]["errors"]):
            raise SystemExit(f"{set_name} {r['id']}: run.py's errors differ from the recount")
    return out


def pair(cand, base, keep, boot):
    """cand/base: {set: [records]} -> per set and pooled CAND - BASE over shared lines satisfying keep."""
    out = {}
    for s in [x for x in SETS if x in cand and x in base] + ["pooled"]:
        if s == "pooled":
            c = [dict(x, id=f"{t}:{x['id']}") for t in SETS if t in cand and t in base for x in cand[t]]
            b = [dict(x, id=f"{t}:{x['id']}") for t in SETS if t in cand and t in base for x in base[t]]
        else:
            c, b = cand[s], base[s]
        bm = {x["id"]: x for x in b}
        sel = [x for x in c if x["id"] in bm and keep(x)]
        a = {x["id"]: (x["errors"], x["n"]) for x in sel}
        bb = {x["id"]: (bm[x["id"]]["errors"], bm[x["id"]]["n"]) for x in sel}
        if any(a[i][1] != bb[i][1] for i in a):
            raise SystemExit(f"{s}: reference lengths differ between arms")
        lines = R.paired_bootstrap(a, bb, boot["reps"], boot["seed"])
        vn = R.cluster_bootstrap(a, bb, {x["id"]: x["vn"] for x in sel}, boot["reps"], boot["seed"],
                                 boot["min_clusters"])
        out[s] = {"delta": lines["delta"], "lines_ci95": lines["ci95"], "vn_ci95": vn.get("ci95"),
                  "units": lines["units"], "chars": lines["chars"], "candidate_errors": lines["candidate_errors"],
                  "baseline_errors": lines["baseline_errors"], "vn_clusters": vn["clusters"]}
    return out


def tables(report, lens, pairs):
    fl = report["shared"]["fixed_line"]
    boot = report["shared"]["spec"]["bootstrap"]
    arms = {pol: {a: e["lines"] for a, e in fl[pol].items()} for pol in fl}
    if lens:
        arms.setdefault("native", {})[LENS] = lens
    out = {"kind": "vocr5-dialogue-tables", "schema": 1, "parts": list(PARTS), "bootstrap": boot,
           "lens_lines": {s: len(v) for s, v in (lens or {}).items()}, "arms": {}, "pairs": {}}
    for pol, by_arm in arms.items():
        for a, sets in by_arm.items():
            e = out["arms"].setdefault(pol, {}).setdefault(a, {})
            for part, keep in PARTS.items():
                e[part] = {s: tally([x for x in sets[s] if keep(x)]) for s in SETS if s in sets}
                e[part]["pooled"] = tally([x for s in SETS if s in sets for x in sets[s] if keep(x)])
    for cand, base in pairs:
        for pol in arms:
            if base == LENS and pol != "native":
                continue
            if cand not in arms[pol] or base not in arms[pol]:
                continue
            out["pairs"].setdefault(f"{cand}-{base}", {})[pol] = {
                part: pair(arms[pol][cand], arms[pol][base], keep, boot) for part, keep in PARTS.items()}
        if base == LENS and "console" in arms and cand in arms["console"]:
            out["pairs"][f"{cand}-{base}"]["console_vs_lens_native"] = {
                part: pair(arms["console"][cand], arms["native"][LENS], keep, boot) for part, keep in PARTS.items()}
    return out


def markdown(doc, order):
    md = ["| arm (polarity) | vita-test dialogue | psp-test dialogue | pooled dialogue | pooled exact | pooled names |",
          "|---|---|---|---|---|---|"]
    for pol in ("console", "native"):
        for a in order:
            e = doc["arms"].get(pol, {}).get(a)
            if not e:
                continue
            d, n = e["dialogue"], e["names"]
            cells = [f"{d[s]['cer']:.2f}% ({d[s]['errors']}/{d[s]['chars']})" if s in d else "-" for s in SETS]
            p = d["pooled"]
            md.append(f"| {a} ({pol}) | " + " | ".join(cells)
                      + f" | **{p['cer']:.2f}%** ({p['errors']}/{p['chars']}) | "
                      f"{100 * p['exact'] / max(1, p['lines']):.1f}% | {n['pooled']['cer']:.2f}% |")
    md += ["", "| pair (dialogue, points; line / VN-cluster 95% CI) | view | vita-test | psp-test | pooled |",
           "|---|---|---|---|---|"]

    def cell(c):
        vn = c["vn_ci95"]
        return (f"{c['delta']:+.2f} ({c['lines_ci95'][0]:+.2f} to {c['lines_ci95'][1]:+.2f}; VN "
                + (f"{vn[0]:+.2f} to {vn[1]:+.2f})" if vn else "n/a)"))
    for name, views in doc["pairs"].items():
        for view, parts in views.items():
            d = parts["dialogue"]
            md.append(f"| {name} | {view} | " + " | ".join(cell(d[s]) if s in d else "-" for s in SETS) +
                      f" | {cell(d['pooled'])} |")
    return "\n".join(md)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tables")
    t.add_argument("--report", required=True)
    t.add_argument("--protocol", default=str(HERE / "vocr5_protocol.json"))
    t.add_argument("--lens", action="append", default=[], help="SET=eval/run.py --out JSON of the Lens arm")
    t.add_argument("--out", required=True)
    t.add_argument("--md")
    args = ap.parse_args()
    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    proto = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    lens = {}
    for spec in args.lens:
        s, _, p = spec.partition("=")
        lens[s] = lens_lines(p, s)
    pairs = [tuple(p) for p in proto["locked_report"]["pairs"]]
    if lens:
        pairs += [("VOCR5_S", LENS), ("VOCR5_N", LENS), ("VOCR5_L", LENS)]
    doc = tables(report, lens, pairs)
    Path(args.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    order = list(proto["locked_report"]["arms"]) + ([LENS] if lens else [])
    md = markdown(doc, order)
    if args.md:
        Path(args.md).write_text(md + "\n", encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
