#!/usr/bin/env python3
"""Collect sweep runs into a Pareto table (Markdown + JSON).

  sweep_table.py [--runs runs] [--eval eval_ckpt_last.json] [--json docs/arch-sweep.json]

For every runs/<name>/ with an eval JSON (sweep_eval.py) this reads the config
from the checkpoint, measures its cost with cost.report and prints one row:
parameters, int8 weight bytes, estimated arena at the longest line (1024 px at
height 32, i.e. 768 px at height 24 for the same physical text), MACs at that
width, synth-val CER and real CER (fp32 and int8 weights). A row is on the
Pareto front when no other row has both a smaller (int8 weights + arena) and a
lower pooled real CER.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cost  # noqa: E402
from model import from_checkpoint  # noqa: E402

REAL = ("real-hand", "real-pseudo")


def pooled(res, variant, sets=REAL):
    e = sum(res[f"{variant}/{s}"]["errors"] for s in sets if f"{variant}/{s}" in res)
    n = sum(res[f"{variant}/{s}"]["chars"] for s in sets if f"{variant}/{s}" in res)
    return e / n if n else None


def train_hours(log):
    """Wall hours of the training loop from the step-rate log lines (approximate)."""
    if not log.exists():
        return None
    rates = [float(m) for m in re.findall(r"(\d+) lines/s", log.read_text())]
    steps = re.findall(r"step (\d+)/(\d+)", log.read_text())
    if not rates or not steps:
        return None
    batch = 96
    return int(steps[-1][0]) * batch / (sum(rates) / len(rates)) / 3600


def rows(runs_dir, eval_name):
    out = []
    for ev in sorted(runs_dir.glob(f"*/{eval_name}")):
        res = json.loads(ev.read_text(encoding="utf-8"))
        ck = torch.load(ev.parent / res["ckpt"], map_location="cpu", weights_only=False)
        model, meta = from_checkpoint(ck)
        n_cls = len(meta["vocab"]) + 1
        w_max = 1024 * meta["height"] // 32
        c = cost.report(model, n_cls, meta["height"], meta["vocab"])
        streamed, _ = cost.arena(model, meta["height"], w_max, n_cls)
        r = res["results"]
        out.append({
            "run": ev.parent.name, "width": meta["width"], "classes": n_cls,
            "bottleneck": meta["bottleneck"], "height": meta["height"],
            "qat": bool(res.get("qat")), "step": res["step"],
            "params": c["params"], "weights_int8": c["weights_int8"],
            "weights_fp32": c["weights_fp32"], "vocab_bytes": c["vocab_bytes"],
            "arena_max": streamed, "w_max": w_max, "macs_max": cost.macs(model, meta["height"], w_max),
            **{f"arena_{w}": c[f"arena_{w}"] for w in cost.WIDTHS},
            "synth_fp32": r["fp32/synth-val"]["cer"], "synth_int8": r["int8/synth-val"]["cer"],
            "synth_line_fp32": r["fp32/synth-val"]["line_acc"],
            **{f"{s}_{v}": r[f"{v}/{s}"]["cer"] for s in REAL for v in ("fp32", "int8")
               if f"{v}/{s}" in r},
            "real_fp32": pooled(r, "fp32"), "real_int8": pooled(r, "int8"),
            "real_chars": sum(r[f"fp32/{s}"]["chars"] for s in REAL if f"fp32/{s}" in r),
            "train_hours": train_hours(ev.parent / "train.log"),
        })
    for a in out:
        size = a["weights_int8"] + a["arena_max"]
        a["pareto"] = a["real_int8"] is not None and not any(
            b is not a and b["real_int8"] is not None
            and b["weights_int8"] + b["arena_max"] <= size and b["real_int8"] <= a["real_int8"]
            and (b["weights_int8"] + b["arena_max"] < size or b["real_int8"] < a["real_int8"])
            for b in out)
    return out


def pct(x):
    return "–" if x is None else f"{100 * x:.2f}%"


def kb(x):
    return f"{x / 1024:,.0f}"


def markdown(rs):
    head = ("| run | width | classes | bneck | H | QAT | params | int8 weights KiB | "
            "est. arena KiB (max line) | weights+arena KiB | MMAC (max line) | "
            "synth-val CER fp32 / int8 | real-hand CER int8 | real-pseudo CER int8 | "
            "real pooled CER fp32 / int8 | Pareto |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for a in sorted(rs, key=lambda a: a["weights_int8"] + a["arena_max"]):
        lines.append(
            f"| {a['run']} | {a['width']} | {a['classes']} | {a['bottleneck']} | {a['height']} | "
            f"{'yes' if a['qat'] else ''} | {a['params']:,} | {kb(a['weights_int8'])} | "
            f"{kb(a['arena_max'])} | {kb(a['weights_int8'] + a['arena_max'])} | "
            f"{a['macs_max'] / 1e6:.0f} | {pct(a['synth_fp32'])} / {pct(a['synth_int8'])} | "
            f"{pct(a.get('real-hand_int8'))} | {pct(a.get('real-pseudo_int8'))} | "
            f"{pct(a['real_fp32'])} / {pct(a['real_int8'])} | {'**yes**' if a['pareto'] else ''} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=HERE.parent / "runs")
    ap.add_argument("--eval", default="eval_ckpt_last.json")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    torch.set_num_threads(1)
    rs = rows(args.runs, args.eval)
    print(markdown(rs))
    if args.json:
        args.json.write_text(json.dumps(rs, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
