#!/usr/bin/env python3
"""textspot6's result tables (docs/textspot6.md "Results"), from the scored runs' files; no scoring here.

  textspot6_tables.py [--eval DIR] [--runs NAME,...] [--out TABLES.md] [--json TABLES.json]

Tables: the page-path dialogue CER of every scored run and decode (unseen dev with the seen VN left out, and hval;
Vita / PSP); the baselines (GRAD + refine, rubi,fix=1 = Rubi's next finder, rubi,fix=5, textspot3's pick, the oracle
when scored); the paired differences against GRAD + refine and rubi,fix=1 (cluster bootstrap; amendment 12);
textspot's whole-frame metrics (F1, all-text CER, dialogue CER, the tagged dialogue CER, text-free frames with a
dialogue box); the hval box metrics (P / R / F1 and tag accuracy per view); the C == twin checks; with --selection,
the per-VN spread of every representative's unseen-dev difference against GRAD + refine (VNs better / worse /
level, the largest gains and losses, and, report only, the difference without its largest-gain VN); the
tight-region rows (OUT/tight/, score_tight.sh) against GRAD + refine and rubi,fix=1; with --per-title RUN:DECODE,
that run's errors per hval title and per development VN beside GRAD + refine's and rubi,fix=1's (Vita and PSP apart).
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
import textspot6_select as SEL  # noqa: E402

DEFAULT_EVAL = Path("/local/home/skerraut/herd-vita-ocr/shared/artifacts/vocr6-spotter/eval")
DECODES = ("crop", "crop+refine", "det", "det+refine")
SEEN = ("v12188",)


def fmt(v, nd=2):
    return "" if v is None else f"{v:.{nd}f}"


def ci(c):
    if not c or c.get("ci95") is None:
        return ""
    return f"{c['delta']:+.2f} [{c['ci95'][0]:+.2f}, {c['ci95'][1]:+.2f}]"


def spread(arm_path, base_path, seen=SEEN, top=5):
    """Per-VN errors of an arm against GRAD + refine on unseen dev: counts of VNs better / worse / level, the largest
    gains and losses, and the paired difference without the largest-gain VN (report only)."""
    a, b = SEL.load_rows(arm_path, seen), SEL.load_rows(base_path, seen)
    per = defaultdict(lambda: [0, 0])
    for k in b:
        per[b[k]["cluster"]][0] += b[k]["errors"]
        per[b[k]["cluster"]][1] += a[k]["errors"]
    d = sorted(per.items(), key=lambda kv: (kv[1][1] - kv[1][0], kv[0]))
    gains = [(v, e[0], e[1]) for v, e in d if e[1] < e[0]][:top]
    losses = [(v, e[0], e[1]) for v, e in reversed(d) if e[1] > e[0]][:top]
    vn = gains[0][0] if gains else None
    keep = {k: r for k, r in a.items() if r["cluster"] != vn}
    return {"all": SEL.diff(a, b), "vns": len(per), "better": sum(1 for e in per.values() if e[1] < e[0]),
            "worse": sum(1 for e in per.values() if e[1] > e[0]), "level": sum(1 for e in per.values() if e[1] == e[0]),
            "gains": gains, "losses": losses,
            "without_top": {"vn": vn, "diff": SEL.diff(keep, {k: b[k] for k in keep})}}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", default=str(DEFAULT_EVAL))
    ap.add_argument("--runs", help="comma list (default: every scored run directory)")
    ap.add_argument("--out")
    ap.add_argument("--json")
    ap.add_argument("--selection", help="the selection record (eval/textspot6_dev_<date>.json): per-VN spread")
    ap.add_argument("--per-title", help="RUN:DECODE: per hval title and per dev VN, against the two baselines")
    a = ap.parse_args(argv)
    E = Path(a.eval)
    runs = a.runs.split(",") if a.runs else sorted(p.parent.name for p in E.glob("*/SCORED")
                                                   if p.parent.name.startswith(("base_", "nohook_", "kd_", "final_")))
    base = {n: (SEL.load_rows(E / n / "dev.json", SEEN), SEL.load_rows(E / n / "hval.json"))
            for n in ("grad_refine", "grad_refine_fix1", "grad_refine_fix5", "oracle") if (E / n / "dev.json").exists()}
    out, rec = [], {"baselines": {}, "runs": {}}
    out.append("| finder | dev | dev Vita / PSP | hval | hval Vita / PSP |")
    out.append("|---|---:|---|---:|---|")
    for n, (d, h) in base.items():
        pd, ph = SEL.per_platform(d), SEL.per_platform(h)
        out.append(f"| {n} | {fmt(SEL.cer(d))} | {fmt(pd.get('vita'))} / {fmt(pd.get('psp'))} | {fmt(SEL.cer(h))} | "
                   f"{fmt(ph.get('vita'))} / {fmt(ph.get('psp'))} |")
        rec["baselines"][n] = {"dev": SEL.cer(d), "hval": SEL.cer(h), "dev_platform": pd, "hval_platform": ph}
    out.append("")
    gd, gh = base["grad_refine"]
    fd, fh = base.get("grad_refine_fix1", (None, None))
    out.append("| run | decode | dev | dev Vita / PSP | hval | hval Vita / PSP | dev − GRAD+refine | "
               "hval − GRAD+refine | dev − fix=1 | hval − fix=1 |")
    out.append("|---|---|---:|---|---:|---|---|---|---|---|")
    for run in ["textspot3_pick", *runs]:
        for dec in DECODES:
            pdv, phv = E / run / f"dev_{dec}.json", E / run / f"hval_{dec}.json"
            if not pdv.exists() or not phv.exists():
                continue
            d, h = SEL.load_rows(pdv, SEEN), SEL.load_rows(phv)
            cg, chg = SEL.diff(d, gd), SEL.diff(h, gh)
            cf = SEL.diff(d, fd) if fd else None
            chf = SEL.diff(h, fh) if fh else None
            pd, ph = SEL.per_platform(d), SEL.per_platform(h)
            out.append(f"| {run} | {dec} | {fmt(SEL.cer(d))} | {fmt(pd.get('vita'))} / {fmt(pd.get('psp'))} | "
                       f"{fmt(SEL.cer(h))} | {fmt(ph.get('vita'))} / {fmt(ph.get('psp'))} | {ci(cg)} | {ci(chg)} | "
                       f"{ci(cf)} | {ci(chf)} |")
            rec["runs"].setdefault(run, {})[dec] = {"dev": SEL.cer(d), "hval": SEL.cer(h), "dev_platform": pd,
                                                    "hval_platform": ph, "vs_grad_refine": {"dev": cg, "hval": chg},
                                                    "vs_fix1": {"dev": cf, "hval": chf}}
    out.append("")
    out.append("| run | decode | F1 | CER all | dialogue | tagged dialogue | text-free with a dialogue box | "
               "UI recall |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for run in ["textspot3_pick", *runs]:
        for k in ("det", "detref"):
            p = E / run / f"tsdev_{k}.json"
            if not p.exists():
                continue
            s = json.loads(p.read_text())["models"]["VOCR5-N"]["summary"]
            tf = s.get("presence_text_free", {})
            out.append(f"| {run} | {k} | {fmt(s['detection']['f1'], 3)} | {fmt(s['cer_all']['cer'])} | "
                       f"{fmt(s['cer_dialogue']['cer'])} | {fmt(s['tagged_dialogue']['cer'])} | "
                       f"{tf.get('with_dialogue_box')}/{tf.get('frames')} | "
                       f"{fmt(s['recall_by_class']['ui']['recall'], 3)} |")
            rec["runs"].setdefault(run, {})[f"tsdev_{k}"] = {
                "f1": s["detection"]["f1"], "cer_all": s["cer_all"]["cer"], "dialogue": s["cer_dialogue"]["cer"],
                "tagged_dialogue": s["tagged_dialogue"]["cer"],
                "text_free_with_dialogue_box": tf.get("with_dialogue_box"), "recall_by_class": s["recall_by_class"]}
    out.append("")
    out.append("| run | view | P | R | F1 | dialogue recall | name recall | dialogue tag acc | name tag acc |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for run in ["textspot3_pick", *runs]:
        p = E / run / "hval_boxes.json"
        if not p.exists():
            continue
        v = json.loads(p.read_text())["views"]
        for view, s in v.items():
            out.append(f"| {run} | {view} | {fmt(s['precision'], 3)} | {fmt(s['recall'], 3)} | {fmt(s['f1'], 3)} | "
                       f"{fmt(s['recall_dialogue'], 3)} | {fmt(s['recall_name'], 3)} | "
                       f"{fmt(s['tag_acc_dialogue'], 3)} | {fmt(s['tag_acc_name'], 3)} |")
        rec["runs"].setdefault(run, {})["hval_boxes"] = v
    out.append("")
    out.append("| run | C == twin (records equal / all, per set) |")
    out.append("|---|---|")
    for run in runs:
        p = E / run / "parity.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        out.append(f"| {run} | " + "; ".join(f"{k} {v['equal']}/{v['records']}" for k, v in r["sets"].items()) + " |")
        rec["runs"].setdefault(run, {})["parity"] = {k: [v["equal"], v["records"]] for k, v in r["sets"].items()}
    if a.selection:
        sel = json.loads(Path(a.selection).read_text(encoding="utf-8"))
        out.append("")
        out.append("| representative | dev − GRAD+refine | VNs better / worse / level | largest gains (VN: errors "
                   "GRAD+refine → spotter) | largest losses | without the largest-gain VN |")
        out.append("|---|---|---|---|---|---|")
        rec["spread"] = {}
        for c in sorted(sel["candidates"], key=lambda c: c["dev_cer"]):
            name = f"{c['arm']}_seed{c['representative_seed']}"
            sp = spread(E / name / f"dev_{c['decode']}.json", E / "grad_refine" / "dev.json")
            rec["spread"][f"{c['arm']}|{c['decode']}"] = dict(sp, run=name)
            g = ", ".join(f"{v} {b} → {s_}" for v, b, s_ in sp["gains"][:3])
            lo = ", ".join(f"{v} {b} → {s_}" for v, b, s_ in sp["losses"][:3])
            out.append(f"| {c['arm']} {c['decode']} s{c['representative_seed']} | {ci(sp['all'])} | "
                       f"{sp['better']} / {sp['worse']} / {sp['level']} | {g} | {lo} | {sp['without_top']['vn']}: "
                       f"{ci(sp['without_top']['diff'])} |")
    if a.per_title:
        run, dec = a.per_title.rsplit(":", 1)
        rec["per_title"] = {"run": run, "decode": dec, "hval": {}, "dev": {}}
        out.append("")
        out.append(f"| set | title / VN | platform | units | chars | GRAD+refine | rubi,fix=1 | {run} {dec} |")
        out.append("|---|---|---|---:|---:|---:|---:|---:|")
        for st, arm, seen in (("hval", E / run / f"hval_{dec}.json", ()), ("dev", E / run / f"dev_{dec}.json", ())):
            g = SEL.load_rows(E / "grad_refine" / f"{st}.json", seen)
            f1 = SEL.load_rows(E / "grad_refine_fix1" / f"{st}.json", seen)
            r = SEL.load_rows(arm, seen)
            per = {}
            for k, u in g.items():
                c = per.setdefault(u["cluster"], {"platform": set(), "units": 0, "chars": 0, "grad_refine": 0,
                                                  "rubi_fix1": 0, "spotter": 0})
                c["platform"].add(u["platform"])
                c["units"] += 1
                c["chars"] += u["chars"]
                c["grad_refine"] += u["errors"]
                c["rubi_fix1"] += f1[k]["errors"]
                c["spotter"] += r[k]["errors"]
            for cl in sorted(per, key=lambda x: (sorted(per[x]["platform"]), x)):
                c = dict(per[cl], platform="/".join(sorted(per[cl]["platform"])))
                rec["per_title"][st][cl] = c
                pc = (lambda e: f"{100 * e / c['chars']:.2f}") if c["chars"] else (lambda e: "")
                out.append(f"| {st} | {cl}{' (seen)' if cl in SEEN else ''} | {c['platform']} | {c['units']} | "
                           f"{c['chars']} | {pc(c['grad_refine'])} | {pc(c['rubi_fix1'])} | {pc(c['spotter'])} |")
    tight = E / "tight"
    if tight.exists():
        out.append("")
        out.append("| tight set | run | decode | CER | Vita / PSP | − GRAD+refine | − fix=1 |")
        out.append("|---|---|---|---:|---|---|---|")
        rec["tight"] = {}
        for st in ("dev_tight", "hval_tight", "hval_tight2"):
            seen = SEEN if st == "dev_tight" else ()
            g = tight / "grad_refine" / f"{st}.json"
            f1 = tight / "grad_refine_fix1" / f"{st}.json"
            if not g.exists() or not f1.exists():
                continue
            gr, fr = SEL.load_rows(g, seen), SEL.load_rows(f1, seen)
            rows = [("grad_refine", "", gr), ("grad_refine_fix1", "", fr)]
            for d in sorted(p for p in tight.iterdir() if p.is_dir() and not p.name.startswith("grad_refine")):
                for dec in DECODES:
                    q = d / f"{st}_{dec}.json"
                    if q.exists():
                        rows.append((d.name, dec, SEL.load_rows(q, seen)))
            for name, dec, r in rows:
                pr = SEL.per_platform(r)
                cg = SEL.diff(r, gr) if name != "grad_refine" else None
                cf = SEL.diff(r, fr) if name != "grad_refine_fix1" else None
                out.append(f"| {st} | {name} | {dec} | {fmt(SEL.cer(r))} | {fmt(pr.get('vita'))} / "
                           f"{fmt(pr.get('psp'))} | {ci(cg)} | {ci(cf)} |")
                rec["tight"].setdefault(st, {})[f"{name}|{dec}" if dec else name] = {
                    "units": len(r), "cer": SEL.cer(r), "platform": pr, "vs_grad_refine": cg, "vs_fix1": cf}
    text = "\n".join(out) + "\n"
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    if a.json:
        Path(a.json).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
