#!/usr/bin/env python3
"""Markdown tables of docs/textdet.md from the committed result files (so the document's numbers are the JSONs').

  textdet_tables.py round eval/textdet_round<N>_dev_<date>.json [--a9 eval/textdet_a9_20261018.json ...]
  textdet_tables.py protected eval/textdet_protected_<date>.json eval/textdet_clannad_<date>.json
"""
import argparse
import json
import sys
from pathlib import Path

MODELS = ("VOCR5-S", "VOCR5-N", "VOCR5-L")


def ci(c):
    return "—" if c is None else f"{c[0]:+.2f} to {c[1]:+.2f}"


def a9_index(paths):
    out = {}
    for p in paths:
        d = json.loads(Path(p).read_text(encoding="utf-8"))
        for row in d["table"]:
            out[(row["shape"], tuple(row["size"]))] = row
    return out


def round_tables(path, a9_paths):
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    a9 = a9_index(a9_paths)
    reps = d["candidates"]
    order = d["rule"]["order"]
    finder = d.get("finder", "det")
    print(f"\n#### Round {d.get('round', '?')}: {finder}\n")
    print("| run | int8 B | arena B | MACs box / screen (Vita) | est. ms box @333 (A9 model) | wd S / N / L | "
          f"{finder} − GRAD, N (VN CI) | {finder} − refine, N (VN CI) | native Vita N | empty | qualifies |")
    print("|---|---:|---:|---|---:|---|---|---|---:|---:|---|")
    for c in order:
        r = reps[c]
        mm = r["models"]
        a = a9.get((r["shape"], (900, 150)))
        ms = f"{a['ms_per_region']['333']['calibrated']:.0f}" if a else "—"
        cs = r["costs"]
        print(f"| {c} | {r['blob_bytes']:,} | {r['arena_960x544']:,} | {cs['vita_box']['macs'] / 1e6:.1f} M / "
              f"{cs['vita_screen']['macs'] / 1e6:.0f} M | {ms} | "
              + " / ".join(f"{mm[m]['wd']['det']['cer']:.2f}" for m in MODELS)
              + f" | {mm['VOCR5-N']['det_minus_grad']['delta']:+.2f} ({ci(mm['VOCR5-N']['det_minus_grad']['ci95'])})"
              + f" | {mm['VOCR5-N']['det_minus_refine']['delta']:+.2f} "
                f"({ci(mm['VOCR5-N']['det_minus_refine']['ci95'])})"
              + f" | {mm['VOCR5-N']['native_vita']['det']['cer']:.2f} | {r['empty_rate']:.2f} | "
              + ("yes" if r["qualifies"] else "no") + " |")
    b = d["baseline"]
    print("\nBaseline (the same units): GRAD " + " / ".join(f"{b[m]['grad']['cer']:.2f}" for m in MODELS) +
          ", GRAD + refine " + " / ".join(f"{b[m]['refine']['cer']:.2f}" for m in MODELS) + " (S / N / L).")
    rule = d["rule"]
    print(f"\nRule: winner {rule['winner']}; report candidate {rule.get('report_candidate')}; "
          f"qualifying {rule['qualifying']}; Pareto front (bytes vs mean wd CER) {d['pareto_mean_wd_cer']}.")
    best = rule["winner"] or rule.get("report_candidate")
    if best:
        r = reps[best]
        print(f"\n{best}, all three models:\n")
        print("| model | GRAD | GRAD + refine | " + finder + " | − GRAD (VN CI) | − refine (VN CI) | native Vita "
              "GRAD / refine / " + finder + " | native − refine (VN CI) |")
        print("|---|---:|---:|---:|---|---|---|---|")
        for m in MODELS:
            x = r["models"][m]
            nv = x["native_vita"]
            print(f"| {m} | {x['wd']['grad']['cer']:.2f} | {x['wd']['refine']['cer']:.2f} | {x['wd']['det']['cer']:.2f}"
                  f" | {x['det_minus_grad']['delta']:+.2f} ({ci(x['det_minus_grad']['ci95'])})"
                  f" | {x['det_minus_refine']['delta']:+.2f} ({ci(x['det_minus_refine']['ci95'])})"
                  f" | {nv['grad']['cer']:.2f} / {nv['refine']['cer']:.2f} / {nv['det']['cer']:.2f}"
                  f" | {x['native_det_minus_refine']['delta']:+.2f} ({ci(x['native_det_minus_refine']['ci95'])}) |")
        lf = r["linefind_dev"]
        print(f"\nlinefind development (vita_dev + psp_dev): line F1 {lf['line_f1']:.3f}, region exact "
              f"{lf['region_exact']:.3f}, merged / split / cut {lf['merged']} / {lf['split']} / {lf['cut']}; "
              f"empty rate {r['empty_rate']:.3f}.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("round")
    p.add_argument("json")
    p.add_argument("--a9", action="append", default=[])
    args = ap.parse_args()
    if args.cmd == "round":
        round_tables(args.json, args.a9)


if __name__ == "__main__":
    sys.exit(main())
