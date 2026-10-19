#!/usr/bin/env python3
"""Markdown tables for docs/textspot.md (issue #104) from the committed records: the frozen baseline
(eval/textspot_baseline_<date>.json, eval/textspot.py baseline) and the selection record (eval/textspot_dev_<date>.json,
eval/textspot.py select), plus the report recognizers' units (VOCR5-S, VOCR5-L) from the candidates' unit directories.
It only formats: every number comes from those records and units files (eval/textspot.py, pinned by amendment 2,
keeps its own shorter `tables`).

  textspot_tables.py --baseline JSON [--dev JSON --units DIR ...] [--eligible-only] [--family A|B]
                     [--part-arm NAME] [--part-baselines A,B] [--out MD]

Tables:
  baselines   every baseline: P / R / F1, mean IoU, CER all and dialogue (VOCR5-N; VOCR5-S / L report), the share of
              text-free frames with a box and boxes per text-free frame, the pick's hit share, C = twin
  candidates  every candidate in the rule's size order (VTSP bytes, then MACs per 960x544 frame, det before
              det+refine): VTSP bytes, MACs, the arena for every frame up to 960x544 (det+refine: the larger of the
              spotter's and the refinement's), the estimated ms per 960x544 frame at 333 and 444 MHz (the A9 cycle
              model's high bound x 1.0007, plus 1.4 ms per refined line at 333 MHz for det+refine; x 333/444 at 444
              MHz: estimates), C = twin, P / R / F1, CER all and dialogue (N; S / L report), the text-free
              false-alarm share, the pick's hit share, eligible, qualifies (winner / report candidate marked)
  pareto      amendment 4's speed / size Pareto front: the eligible candidates no other eligible candidate
              dominates on VTSP bytes, est. ms at 333 MHz (444 MHz is the same order), dev F1 and both CERs (at least
              as good on every key and better on one; --pareto-keys picks other keys), in the rule's size order
  rule        the winner's (or the report candidate's) comparisons with every baseline: F1, CER all and dialogue
              differences with their VN-cluster 95% intervals, and whether it beats that baseline
  parts       its per-part rows (vita, psp, adrenaline, pc) next to the baselines with the highest F1 and the
              lowest all-text CER (or --part-baselines), and its text-free frames
  classes     recall per class of it and of every baseline
Without --dev only the baselines' tables are written.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "synth", ROOT / "train"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot as TS  # noqa: E402
import textspot_metrics as M  # noqa: E402

REPORT = ("VOCR5-S", "VOCR5-L")
MHZ = 333 / 444


def num(v, nd=3):
    return "–" if v is None else f"{v:.{nd}f}"


def pct(v):
    return "–" if v is None else f"{100 * v:.0f}%"


def cer(s, key):
    c = (s or {}).get(key)
    return "–" if not c or c.get("cer") is None else f"{c['cer']:.2f}"


def both(sums, key):
    """'S / L' of a key over the report recognizers' summaries ({model: summary})."""
    return " / ".join(cer(sums.get(m), key) for m in REPORT)


def size_key(c):
    return (c["blob_bytes"], c["macs_960"], c["decode"] != "det", c["name"])


def ci(x, nd):
    if x is None or x.get("delta") is None:
        return "–"
    c = x.get("ci95")
    iv = f" ({c[0]:+.{nd}f} to {c[1]:+.{nd}f})" if c else " (no interval)"
    return f"{x['delta']:+.{nd}f}{iv}"


def report_units(dirs, sel=None):
    """{arm: {model: summary}} of every arm's units in the directories (VOCR5-N and the report recognizers), each
    report file checked to score the same boxes as the arm's VOCR5-N units, and the VOCR5-N summary checked against
    the selection record's when one is given."""
    out = {}
    for d in dirs or []:
        for b in sorted(Path(d).glob("*/boxes.json")):
            n = b.parent / f"units_{TS.RULE_MODEL}.json"
            if not n.exists():
                continue
            nd = json.loads(n.read_text(encoding="utf-8"))
            arm = b.parent.name
            e = {TS.RULE_MODEL: nd["summary"]}
            for m in REPORT:
                f = b.parent / f"units_{m}.json"
                if f.exists():
                    doc = json.loads(f.read_text(encoding="utf-8"))
                    if doc["meta"]["boxes_sha256"] != nd["meta"]["boxes_sha256"]:
                        raise SystemExit(f"{f}: the units of other boxes than {n}")
                    e[m] = doc["summary"]
            c = (sel or {}).get("candidates", {}).get(arm)
            if c is not None and c.get("summary") is not None and c["summary"] != nd["summary"]:
                raise SystemExit(f"{n}: not the units the selection record scored")
            out[arm] = e
    return out


# ----------------------------------------------------------------------------- baselines

def baseline_table(base):
    md = ["| baseline | P | R | F1 | mean IoU | CER all (N) | dialogue (N) | all S / L | dialogue S / L | "
          "text-free frames with a box | boxes per text-free frame | pick hit | C = twin |",
          "|---|" + "---:|" * 11 + "---|"]
    for arm in TS.BASELINES:
        e = base["arms"][arm]
        s = e[TS.RULE_MODEL]["summary"]
        d, tf = s["detection"], s["text_free"]
        sl = {m: e[m]["summary"] for m in REPORT if m in e}
        md.append(f"| `{arm}` | {num(d['precision'])} | {num(d['recall'])} | {num(d['f1'])} | "
                  f"{num(s['tightness']['mean_iou'])} | {cer(s, 'cer_all')} | {cer(s, 'cer_dialogue')} | "
                  f"{both(sl, 'cer_all')} | {both(sl, 'cer_dialogue')} | {pct(tf['share_with_any_box'])} | "
                  f"{num(tf['boxes_per_frame'], 2)} | {pct(s['pick']['hit_share'])} | {s['c_equal']} |")
    return "\n".join(md)


# ----------------------------------------------------------------------------- candidates

def mark(name, c, res):
    if name == res.get("winner"):
        return "winner"
    if name in res.get("qualifying", []):
        return "yes"
    return "no (report candidate)" if name == res.get("report_candidate") else "no"


def candidate_table(sel, rep, eligible_only=False, family=None):
    res = sel["result"]
    rows = sorted(((n, c) for n, c in sel["candidates"].items()
                   if (c["eligible"] or not eligible_only) and (family is None or c.get("family") == family)),
                  key=lambda kv: size_key(kv[1]))
    md = ["| candidate | decode | VTSP B | MACs / 960×544 | arena B (≤ 960×544) | est. ms @333 / @444 | C = twin | "
          "P | R | F1 | CER all (N) | dialogue (N) | all S / L | dialogue S / L | text-free frames with a box | "
          "pick hit | eligible | qualifies |",
          "|---|---|" + "---:|" * 4 + "---|" + "---:|" * 9 + "---|---|"]
    for name, c in rows:
        s = c.get("summary") or c.get("detection_summary") or {}
        d = s.get("detection") or {}
        ms = c.get("est_ms_333_960")
        est = "–" if ms is None else f"{ms:.1f} / {ms * MHZ:.1f}"
        sl = {m: v for m, v in rep.get(name, {}).items() if m in REPORT}
        md.append(f"| `{name}` | {c['decode']} | {c['blob_bytes']:,} | {c['macs_960'] / 1e6:.1f} M | "
                  f"{'–' if c.get('arena_960') is None else format(c['arena_960'], ',')} | {est} | {c['c_equal']} | "
                  f"{num(d.get('precision'))} | {num(d.get('recall'))} | {num(d.get('f1'))} | {cer(s, 'cer_all')} | "
                  f"{cer(s, 'cer_dialogue')} | {both(sl, 'cer_all')} | {both(sl, 'cer_dialogue')} | "
                  f"{pct((s.get('text_free') or {}).get('share_with_any_box'))} | "
                  f"{pct((s.get('pick') or {}).get('hit_share'))} | {'yes' if c['eligible'] else 'no'} | "
                  f"{mark(name, c, res)} |")
    n_el = sum(1 for _, c in rows if c["eligible"])
    md += ["", f"{len(rows)} candidates, {n_el} eligible, {len(res.get('qualifying', []))} qualifying; winner: "
               f"{res.get('winner') or 'none'}; report candidate: {res.get('report_candidate') or 'none'}."]
    ra = sel.get("refine_arena") or {}
    if ra.get("bytes_960x544") is None:
        md.append("WARNING: the selection record has no refinement arena (select ran without --cbin): every "
                  "det+refine candidate was ineligible for an unknown arena.")
    return "\n".join(md)


PARETO_KEYS = {"bytes": lambda c, s: c["blob_bytes"], "ms": lambda c, s: c["est_ms_333_960"],
               "f1": lambda c, s: -s["detection"]["f1"], "cer_all": lambda c, s: s["cer_all"]["cer"],
               "cer_dialogue": lambda c, s: s["cer_dialogue"]["cer"]}
PARETO_DEFAULT = ("bytes", "ms", "f1", "cer_all", "cer_dialogue")


def pareto(sel, keys=PARETO_DEFAULT):
    """[(name, candidate)] of the eligible candidates (read by VOCR5-N) that no other dominates on keys (lower is
    better for each; F1 is negated), in the rule's size order."""
    pts = [(n, c, tuple(PARETO_KEYS[k](c, c["summary"]) for k in keys)) for n, c in sel["candidates"].items()
           if c["eligible"] and c.get("summary")]
    front = [(n, c) for n, c, v in pts
             if not any(w != v and all(a <= b for a, b in zip(w, v)) for _, _, w in pts)]
    return sorted(front, key=lambda kv: size_key(kv[1]))


def pareto_table(sel, keys=PARETO_DEFAULT):
    res = sel["result"]
    front = pareto(sel, keys)
    md = ["| candidate | decode | VTSP B | est. ms @333 / @444 | F1 | CER all (N) | dialogue (N) | qualifies |",
          "|---|---|---:|---:|---:|---:|---:|---|"]
    for name, c in front:
        s, ms = c["summary"], c["est_ms_333_960"]
        md.append(f"| `{name}` | {c['decode']} | {c['blob_bytes']:,} | {ms:.1f} / {ms * MHZ:.1f} | "
                  f"{num(s['detection']['f1'])} | {cer(s, 'cer_all')} | {cer(s, 'cer_dialogue')} | "
                  f"{mark(name, c, res)} |")
    n = sum(1 for c in sel["candidates"].values() if c["eligible"] and c.get("summary"))
    md += ["", f"{len(front)} of {n} eligible candidates are on the front of {', '.join(keys)} (lower bytes, ms and "
               "CER, higher F1); est. ms are A9 cycle-model estimates."]
    return "\n".join(md)


def focus(sel, name=None):
    res = sel["result"]
    return name or res.get("winner") or res.get("report_candidate")


def rule_table(sel, name=None):
    name = focus(sel, name)
    if name is None:
        return "No eligible candidate: no comparison."
    comps = sel["result"]["comparisons"].get(name) or sel.get("report_comparisons", {}).get(name)
    if not comps:
        return f"`{name}` has no comparisons in the record."
    md = [f"| `{name}` minus | F1 Δ (VN-cluster 95% CI) | CER all Δ, points (CI) | dialogue CER Δ, points (CI) | "
          "beats |", "|---|---:|---:|---:|---|"]
    for b, cmp in sorted(comps.items()):
        md.append(f"| `{b}` | {ci(cmp['f1'], 4)} | {ci(cmp['cer_all'], 2)} | {ci(cmp['cer_dialogue'], 2)} | "
                  f"{'yes' if M.beats(cmp) else 'no'} |")
    return "\n".join(md)


def default_part_baselines(base):
    bs = {a: base["arms"][a][TS.RULE_MODEL]["summary"] for a in TS.BASELINES}
    f1 = max(TS.BASELINES, key=lambda a: (bs[a]["detection"]["f1"] or 0.0, a))
    ce = min(TS.BASELINES, key=lambda a: (bs[a]["cer_all"]["cer"], a))
    return list(dict.fromkeys([f1, ce]))


def part_table(sel, base, name=None, baselines=None):
    name = focus(sel, name)
    if name is None:
        return "No eligible candidate: no per-part table."
    s = sel["candidates"][name]["summary"]
    baselines = baselines or default_part_baselines(base)
    arms = [(f"`{name}`", s)] + [(f"`{b}`", base["arms"][b][TS.RULE_MODEL]["summary"]) for b in baselines]
    md = ["| part | frames | references | " + " | ".join(f"{a} F1 | CER all | dialogue" for a, _ in arms) + " |",
          "|---|---:|---:|" + "---:|" * (3 * len(arms))]
    tf_line = []
    for p, ps in s["per_part"].items():
        if p in ("text-free", "text_free"):
            for a, x in arms:
                t = x["per_part"].get(p, {}).get("text_free", {})
                tf_line.append(f"{a} {pct(t.get('share_with_any_box'))} with a box "
                               f"({num(t.get('boxes_per_frame'), 2)} per frame)")
            continue
        d = ps["detection"]
        cells = []
        for _, x in arms:
            q = x["per_part"].get(p, {})
            cells.append(f"{num((q.get('detection') or {}).get('f1'))} | {cer(q, 'cer_all')} | "
                         f"{cer(q, 'cer_dialogue')}")
        md.append(f"| {p} | {ps['frames']} | {d['tp'] + d['fn']} | " + " | ".join(cells) + " |")
    if tf_line:
        n = s["per_part"].get("text-free", s["per_part"].get("text_free", {})).get("frames")
        md += ["", f"Text-free frames ({n}): " + "; ".join(tf_line) + "."]
    return "\n".join(md)


def class_table(sel, base, name=None):
    name = focus(sel, name) if sel else None
    arms = ([(f"`{name}`", sel["candidates"][name]["summary"])] if name else []) + \
        [(f"`{b}`", base["arms"][b][TS.RULE_MODEL]["summary"]) for b in TS.BASELINES]
    refs = arms[0][1]["recall_by_class"]
    md = ["| recall by class | " + " | ".join(f"{c} ({refs[c]['refs']})" for c in M.CLASSES) + " |",
          "|---|" + "---:|" * len(M.CLASSES)]
    for a, s in arms:
        r = s["recall_by_class"]
        md.append(f"| {a} | " + " | ".join(num(r[c]["recall"]) for c in M.CLASSES) + " |")
    return "\n".join(md)


def render(base, sel=None, rep=None, eligible_only=False, family=None, part_arm=None, part_baselines=None,
           pareto_keys=PARETO_DEFAULT):
    out = ["### Baselines (development)", "", baseline_table(base), ""]
    if sel is None:
        return "\n".join(out + ["### Recall by class", "", class_table(None, base), ""])
    if sel["manifest"]["sha256"] != base["manifest"]["sha256"]:
        raise SystemExit("the selection record and the baseline are of different manifests")
    out += ["### Candidates (rule's size order)", "", candidate_table(sel, rep or {}, eligible_only, family), "",
            "### Speed / size Pareto front (eligible candidates)", "", pareto_table(sel, pareto_keys), "",
            "### The rule's comparisons", "", rule_table(sel, part_arm), "",
            "### Per part", "", part_table(sel, base, part_arm, part_baselines), "",
            "### Recall by class", "", class_table(sel, base, part_arm), ""]
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--dev", default="", help="the selection record (eval/textspot.py select)")
    ap.add_argument("--units", action="append", default=[], help="the candidates' unit directories (S / L units)")
    ap.add_argument("--eligible-only", action="store_true")
    ap.add_argument("--family", default=None, choices=("A", "B"))
    ap.add_argument("--part-arm", default=None, help="the candidate of the per-part table (default: the winner, "
                                                     "else the report candidate)")
    ap.add_argument("--part-baselines", default="", help="comma-separated baselines beside it")
    ap.add_argument("--pareto-keys", default=",".join(PARETO_DEFAULT), help=f"of {sorted(PARETO_KEYS)}")
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)
    base = json.loads(Path(a.baseline).read_text(encoding="utf-8"))
    sel = json.loads(Path(a.dev).read_text(encoding="utf-8")) if a.dev else None
    rep = report_units(a.units, sel) if sel else None
    keys = tuple(k for k in a.pareto_keys.split(",") if k)
    if not keys or any(k not in PARETO_KEYS for k in keys):
        raise SystemExit(f"--pareto-keys: of {sorted(PARETO_KEYS)}")
    text = render(base, sel, rep, a.eligible_only, a.family, a.part_arm,
                  [b for b in a.part_baselines.split(",") if b] or None, keys)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
