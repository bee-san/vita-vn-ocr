#!/usr/bin/env python3
"""textspot6's result tables (docs/textspot6.md "Results"), from the scored runs' files; no scoring here.

  textspot6_tables.py [--eval DIR] [--runs NAME,...] [--out TABLES.md] [--json TABLES.json]

Tables: the page-path dialogue CER of every scored run and decode (unseen dev with the seen VN left out, and hval;
Vita / PSP); the baselines (GRAD + refine, rubi,fix=1 = Rubi's next finder, rubi,fix=5, textspot3's pick, the oracle
when scored); the paired differences against GRAD + refine and rubi,fix=1 (cluster bootstrap; amendment 12);
textspot's whole-frame metrics (F1, all-text CER, dialogue CER, the tagged dialogue CER, text-free frames with a
dialogue box); the hval box metrics (P / R / F1 and tag accuracy per view); the C == twin checks.
"""
import argparse
import json
import sys
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", default=str(DEFAULT_EVAL))
    ap.add_argument("--runs", help="comma list (default: every scored run directory)")
    ap.add_argument("--out")
    ap.add_argument("--json")
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
    text = "\n".join(out) + "\n"
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
    if a.json:
        Path(a.json).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
