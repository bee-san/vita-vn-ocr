#!/usr/bin/env python3
"""Markdown tables of the teacher's measurements, straight from the recorded JSON files, so the
numbers in docs/teacher.md and the PR are the measured ones.

  report.py [--results $TEACHER_OUT/results]  -> stdout

Reads engine_<engine>.json (eval/run.py --out, recognizer-only on eval's crops), rules.json
(measure.py rules --out), hand.json (measure.py hand), teacher/manifests/stats.json and
teacher/manifests/spot.jsonl, whichever exist.
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from teacher.label import OUT  # noqa: E402

NAMES = {"meiki": "MeikiOCR rec (BGR, as its docs)", "meiki-rgb": "MeikiOCR rec (RGB, as owocr feeds it)",
         "v6s": "PP-OCRv6 small rec", "v5s": "PP-OCRv5 server rec", "manga": "manga-ocr (re-flowed)",
         "screenai": "Chrome Screen AI (owocr)"}
STYLES = ("bright-on-dark", "dark-on-light", "outlined", "name", "coloured", "mincho", "small")


def pct(v, d=2):
    return "-" if v is None else f"{100 * v:.{d}f}%"


def engines(res):
    out = []
    files = {p.stem[len("engine_"):]: p for p in sorted(res.glob("engine_*.json"))}
    if not files:
        return out
    sets = sorted({s for p in files.values() for s in json.loads(p.read_text())["sets"]})
    out.append("| engine | " + " | ".join(f"{s} CER / exact" for s in sets) + " | host ms/line (1 thread) |")
    out.append("|---|" + "---|" * len(sets) + "---|")
    for e in NAMES:
        if e not in files:
            continue
        d = json.loads(files[e].read_text())
        cells = [f"{pct(d['sets'][s]['all']['cer'])} / {pct(d['sets'][s]['all']['exact'], 1)}"
                 if s in d["sets"] else "-" for s in sets]
        ms = "/".join(str(round(d["sets"][s]["host_ms_per_line"])) for s in sets if s in d["sets"])
        out.append(f"| {NAMES[e]} | " + " | ".join(cells) + f" | {ms} |")
    out.append("")
    for s in sets:
        out.append(f"Per style, {s} (CER):\n")
        out.append("| style | " + " | ".join(e for e in NAMES if e in files) + " |")
        out.append("|---|" + "---|" * sum(e in files for e in NAMES))
        for st in STYLES:
            row = []
            for e in NAMES:
                if e not in files:
                    continue
                d = json.loads(files[e].read_text())["sets"].get(s, {}).get("styles", {}).get(st)
                row.append(pct(d["cer"]) if d else "-")
            if any(c != "-" for c in row):
                any_e = next(iter(files.values()))
                n = json.loads(any_e.read_text())["sets"].get(s, {}).get("styles", {}).get(st, {}).get("lines", "?")
                out.append(f"| {st} ({n}) | " + " | ".join(row) + " |")
        out.append("")
    return out


def fmt_rule(r):
    v = "+".join(r.get("voters") or []) or "none"
    s = f"primary {r.get('primary', 'meikiv')}, voters {v}, votes >= {r.get('votes')}"
    if r.get("min_conf"):
        s += f", MeikiOCR min char conf >= {r['min_conf']}"
    if r.get("det_conf"):
        s += f", det score >= {r['det_conf']}"
    if r.get("veto"):
        s += ", confusable veto"
    if r.get("trunc"):
        t = r["trunc"]
        s += ", truncation check (" + ("+".join(t) if isinstance(t, (list, tuple)) else "any dissent") + ")"
    return s


def rules(res):
    p = res / "rules.json"
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    out = ["Per-engine accuracy on the teacher's own lines (matched to the truth; each engine reads the "
           "teacher's view of the line):", "", "| set | engine | lines | CER | exact |", "|---|---|---|---|---|"]
    for s, eng in d["engines"].items():
        for e, v in eng.items():
            out.append(f"| {s} | {e} | {v['lines']} | {pct(v['cer'])} | {pct(v['exact'], 1)} |")
    c = d["chosen"]
    out += ["", f"Chosen rule: **{fmt_rule(c['rule'])}**", "",
            "| | accepted (scored) | precision | 95% CI | recall of truth lines | label CER |",
            "|---|---|---|---|---|---|"]

    def row(name, s, ci=None):
        return (f"| {name} | {s['accepted']} ({s['accepted_scored']}) | {pct(s['precision'])} | "
                f"{ci or '-'} | {pct(s['recall'], 1)} of {s['gt_lines']} | {pct(s['label_cer'], 3)} |")
    ci = c.get("precision_ci95")
    out.append(row("pooled", c["pooled"], f"{pct(ci[0])}-{pct(ci[1])}" if ci and ci[0] is not None else None))
    for s, v in c["per_set"].items():
        out.append(row(s, v))
    for f in d.get("two_fold") or []:
        if f:
            out.append(row(f"held-out half {1 - f['chosen_on']} (rule chosen on half {f['chosen_on']})",
                           f["held_out"]))
    out += ["", "Reference rules (pooled):", "", "| rule | accepted (scored) | precision | recall | label CER |",
            "|---|---|---|---|---|"]
    for name, s in d["reference"].items():
        out.append(f"| {name} | {s['accepted']} ({s['accepted_scored']}) | {pct(s['precision'])} | "
                   f"{pct(s['recall'], 1)} | {pct(s['label_cer'], 3)} |")
    out += ["", "Precision/recall frontier of the rule grid (pooled, best 12):", "",
            "| recall | precision | rule |", "|---|---|---|"]
    for f in d["frontier"][:12]:
        out.append(f"| {pct(f['recall'], 1)} | {pct(f['precision'])} | {fmt_rule(f['rule'])} |")
    return out + [""]


def build_stats():
    p = HERE / "manifests" / "stats.json"
    if not p.exists():
        return []
    s = json.loads(p.read_text())
    out = [f"Relabel: {s['screenshots_read']} screenshots read, {s['lines_read']} lines, **{s['accepted']} accepted** "
           f"({s['unique_labels']} distinct labels, {s['chars']} characters) from {s['screenshots']} screenshots, "
           f"{s['sessions']} capture sessions, VNs {s['vns']}.", "",
           "| | lines |", "|---|---|"]
    for k, v in s["by_kind"].items():
        out.append(f"| kind {k} | {v} |")
    for k, v in s["crops_by_split_target"].items():
        out.append(f"| crops {k} | {v} |")
    out += ["", "Rejected: " + ", ".join(f"{k} {v}" for k, v in s["rejected"].items()), ""]
    return out


def spot():
    p = HERE / "manifests" / "spot.jsonl"
    if not p.exists():
        return []
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    bad = [r for r in rows if not r["ok"]]
    out = [f"Spot check: **{len(bad)} wrong of {len(rows)}** random accepted lines ({pct(len(bad) / len(rows), 1)})."]
    for r in bad:
        out.append(f"- `{r['id']}`: label 「{r['label']}」, truth 「{r.get('truth')}」 {r.get('note') or ''}")
    return out + [""]


def styles(res):
    """Per-style precision and recall of the chosen rule on eval's sets, from rules.lines.json
    (measure.py's per-line dump) and the manifests' style tags."""
    p = res / "rules.lines.json"
    if not p.exists():
        return []
    from teacher import measure
    d = json.loads(p.read_text(encoding="utf-8"))
    tags = {}
    for name in ("vita-test", "psp-test"):
        try:
            rows = measure.gt_rows(name)
        except SystemExit:
            continue
        for rs in rows.values():
            for g in rs:
                if not g.get("uncertain"):
                    tags[g["id"]] = g.get("styles") or []
    recalled, acc, ok = set(), {}, {}
    for r in d["lines"]:
        if r["label"] is None or r["truth"] is None:
            continue
        for gid in r["gt"]:
            for t in tags.get(gid, []):
                acc[t] = acc.get(t, 0) + 1
                ok[t] = ok.get(t, 0) + bool(r["ok"])
            if r["ok"]:
                recalled.add(gid)
    out = ["Chosen rule by style (vita-test + psp-test; a joined row counts once per truth line):", "",
           "| style | truth lines | recall | accepted (scored) | precision |", "|---|---|---|---|---|"]
    for t in sorted({t for ts in tags.values() for t in ts}):
        ids = [i for i, ts in tags.items() if t in ts]
        rec = sum(i in recalled for i in ids) / len(ids)
        prec = ok.get(t, 0) / acc[t] if acc.get(t) else None
        out.append(f"| {t} | {len(ids)} | {pct(rec, 1)} | {acc.get(t, 0)} | {pct(prec)} |")
    return out + [""]


def timing():
    """Host seconds per engine in the relabel (label.py's per-screenshot timers)."""
    from teacher import build
    shots = build.load_reads(OUT / "reads")
    if not shots:
        return []
    tot, n_lines = {}, 0
    for r in shots.values():
        for k, v in r.get("secs", {}).items():
            tot[k] = tot.get(k, 0.0) + v
        n_lines += len(r["lines"])
    out = [f"Relabel cost (host, one thread per worker): {len(shots)} screenshots, {n_lines} lines.", "",
           "| engine | s per screenshot | s total |", "|---|---|---|"]
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1]):
        out.append(f"| {k} | {v / len(shots):.3f} | {v:.0f} |")
    return out + [""]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default=str(OUT / "results"))
    args = ap.parse_args()
    res = Path(args.results)
    hand = res / "hand.json"
    lines = engines(res) + rules(res) + styles(res)
    if hand.exists():
        h = json.loads(hand.read_text())
        lines += [f"realdata's hand-checked training-side lines: {h['correct']}/{h['scored']} accepted labels "
                  f"match exactly ({pct(h['correct'] / max(1, h['scored']))}, 95% CI {h['ci95']}).", ""]
    print("\n".join(lines + build_stats() + spot() + timing()))


if __name__ == "__main__":
    main()
