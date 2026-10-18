#!/usr/bin/env python3
"""textgate: the public records and the tables of docs/textgate.md.

  textgate_tables.py record --calibration JSON --selection JSON --parity JSON --sizes JSON --out-cal JSON --out-dev JSON
      The committed records: the calibration (every candidate's operating points on the training-side cohorts) and
      the development selection (eval/textgate.py select), with local paths written relative to $HERD or the clone.
  textgate_tables.py tables --cal JSON --dev JSON [--report JSON]
      Markdown tables (calibration, development, the winner), printed.
"""
import argparse
import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERD = os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr")
MODELS = ("VOCR5-S", "VOCR5-N", "VOCR5-L")


def scrub(obj):
    """Local paths as $HERD/... or ./... (the records name no machine or home directory)."""
    if isinstance(obj, dict):
        return {k: scrub(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [scrub(v) for v in obj]
    if isinstance(obj, str):
        s = obj.replace(HERD, "$HERD").replace(str(ROOT), ".")
        return re.sub(r"/(?:local/)?home/[^/\s\"]+/[^\s\"]*", "<local path>", s)
    return obj


def cmd_record(args):
    cal = json.loads(Path(args.calibration).read_text())
    sel = json.loads(Path(args.selection).read_text())
    parity = json.loads(Path(args.parity).read_text()) if args.parity else {}
    sizes = json.loads(Path(args.sizes).read_text()) if args.sizes else {}
    cal_doc = {"kind": "textgate-calibration-record", "computed": cal["computed"], "cohorts": cal["cohorts"],
               "targets": cal["targets"], "k_set": cal["k_set"], "screen_empty": cal["screen_empty"],
               "cal_empty_finder_none": cal["cal_empty_finder_none"],
               "cal_dlg_finder_none": cal.get("cal_dlg_finder_none"),
               "candidates": {n: {k: v for k, v in c.items() if k != "blob"} for n, c in cal["candidates"].items()}}
    dev_doc = dict(sel, kind="textgate-development-record", parity=parity, sizes=sizes)
    for path, doc in ((args.out_cal, cal_doc), (args.out_dev, dev_doc)):
        Path(path).write_text(json.dumps(scrub(doc), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        print(path)


def fmt_ci(d):
    ci = d.get("ci95")
    return f"{d['delta']:+.3f} ({ci[0]:+.2f} to {ci[1]:+.2f})" if ci else f"{d['delta']:+.3f}"


def cmd_tables(args):
    cal = json.loads(Path(args.cal).read_text())
    dev = json.loads(Path(args.dev).read_text())
    print("| candidate | int8 B | region, τ 0.998 | boxes, τ 0.998 | region, τ 0.999 | boxes, τ 0.999 |")
    print("|---|---:|---:|---:|---:|---:|")

    def cell(o):
        v = f"{o['empty_rate']:.3f}"
        return f"**{v}**" if o["screen_pass"] else v
    for name, c in sorted(cal["candidates"].items(), key=lambda kv: (kv[1]["bytes"], kv[0])):
        m = c["modes"]
        cells = [cell(m[mode][tau]) if mode in m else "" for tau in ("0.998", "0.999") for mode in ("region", "boxes")]
        print(f"| {name} | {c['bytes']:,} | " + " | ".join(cells) + " |")
    print()
    print("| operating point | total B (weights + code) | wd cases stopped | gate − refine, S / N / L (VN CI) | "
          "native Vita, N | empty: no line (of 66) | out of the scene pool (of 12) | qualifies |")
    print("|---|---:|---:|---|---|---:|---:|---|")
    for key in dev["order"]:
        r = dev["candidates"][key]
        mm = r["models"]
        e = r["empty_development"]
        print(f"| {key} | {r['total_bytes']:,} ({r['weight_bytes']:,} + {r['code_bytes']:,}) | "
              f"{r['wd_cases_dropped']} | " + " / ".join(fmt_ci(mm[m]["gate_minus_refine"]) for m in MODELS) +
              f" | {fmt_ci(mm['VOCR5-N']['native_gate_minus_refine'])} | {e['none_returned']} ({e['rate']:.3f}) | "
              f"{e['not_in_scene_pool'][0]} | {'**yes**' if r['qualifies'] else 'no'} |")
    w = dev.get("winner")
    if w:
        r = dev["candidates"][w]
        print()
        print(f"Winner {w}: {r['total_bytes']:,} B ({r['weight_bytes']:,} B int8 + {r['code_bytes']:,} B code), "
              f"k {r['k']}, t {r['t']}")
        for m in MODELS:
            mm = r["models"][m]
            print(f"  {m}: wd {mm['wd']['refine']['cer']} -> {mm['wd']['gate']['cer']} "
                  f"({fmt_ci(mm['gate_minus_refine'])}); native Vita {mm['native_vita']['refine']['cer']} -> "
                  f"{mm['native_vita']['gate']['cer']}; per platform " +
                  ", ".join(f"{p} {mm['per_platform'][p]['refine']['cer']} -> {mm['per_platform'][p]['gate']['cer']}"
                            for p in ("vita", "psp")))
        print(f"  empties: development {r['empty_development']}; report {r['empty_report']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("record")
    for a in ("--calibration", "--selection", "--out-cal", "--out-dev"):
        p.add_argument(a, required=True)
    p.add_argument("--parity", default="")
    p.add_argument("--sizes", default="")
    p = sub.add_parser("tables")
    p.add_argument("--cal", required=True)
    p.add_argument("--dev", required=True)
    args = ap.parse_args()
    {"record": cmd_record, "tables": cmd_tables}[args.cmd](args)


if __name__ == "__main__":
    main()
