#!/usr/bin/env python3
"""Collect sweep runs into a Pareto table (Markdown + JSON).

  sweep_table.py [--runs runs] [--eval eval_ckpt_last.json] [--ref base_s1]
                 [--export path/to/runtime/train/export.py] [--json docs/arch-sweep.json]

For every runs/<name>/ with an eval JSON (sweep_eval.py) this reads the config from
the checkpoint, measures its cost with cost.report and prints one row: parameters,
int8 blob bytes and the runtime's exact arena (cost.runtime_sizes, when runtime's
exporter is importable; otherwise cost.py's estimates, marked "est."), MACs at the
longest line (1024 px at height 32, i.e. 768 px at height 24 for the same physical
text), synth-val CER and real CER (fp32 and int8 weights).

The decision metric ("real CER") pools eval's vita-test with dark-on-light lines
inverted by their style tag (vita-test-inv) and the research's hand lines from 17
other games (real-hand-x): errors / reference characters, int8 (W8F) weights. The v1
recipe renders only light text, so the as-is vita-test mostly measures that data gap
(which synth2 and the real fine-tuning close), not capacity; it is shown alongside.
Without eval/ results the provisional real-hand + real-pseudo pool is used.
"Δ vs ref" is the pooled real CER minus the reference run's, with a 95% paired
bootstrap interval over lines (10,000 resamples, seed 0). A row is on the Pareto
front when no other row has both a smaller on-console footprint (int8 blob + W8F
arena) and a lower real CER.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cost  # noqa: E402
from model import from_checkpoint  # noqa: E402
from sweep_eval import score_norm  # noqa: E402
from train import edit_distance  # noqa: E402

REAL = ("vita-test-inv", "real-hand-x")
PROVISIONAL = ("real-hand", "real-pseudo")


def pooled(res, variant, sets=REAL):
    e = sum(res[f"{variant}/{s}"]["errors"] for s in sets if f"{variant}/{s}" in res)
    n = sum(res[f"{variant}/{s}"]["chars"] for s in sets if f"{variant}/{s}" in res)
    return e / n if n else None


def real_sets(res):
    r = res["results"] if "results" in res else res
    return REAL if all(f"int8/{s}" in r for s in REAL) else PROVISIONAL


def line_errors(res, variant="int8", sets=REAL):
    """{(set, line id): (errors, reference chars)} of one eval JSON."""
    out = {}
    for s in sets:
        for d in res["results"].get(f"{variant}/{s}", {}).get("lines_detail", []):
            p, r = score_norm(d["pred"]), score_norm(d["ref"])
            out[(s, d["id"])] = (edit_distance(list(p), list(r)), len(r))
    return out


def paired_delta(a, b, n_boot=10_000, seed=0):
    """Pooled CER(a) - CER(b) over their common lines, with a 95% bootstrap interval."""
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    ea = np.array([a[k][0] for k in keys], np.float64)
    eb = np.array([b[k][0] for k in keys], np.float64)
    n = np.array([a[k][1] for k in keys], np.float64)
    delta = (ea.sum() - eb.sum()) / n.sum()
    idx = np.random.default_rng(seed).integers(0, len(keys), (n_boot, len(keys)))
    boot = ((ea[idx] - eb[idx]).sum(1)) / n[idx].sum(1)
    return delta, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def train_minutes(log, batch=96):
    """Approximate wall minutes of the training loop: steps x batch / mean logged lines/s
    (the shared box's load varies, so this is only a rough cost indication)."""
    if not log.exists():
        return None
    text = log.read_text()
    rates = [float(m) for m in re.findall(r"(\d+) lines/s", text)]
    steps = re.findall(r"step (\d+)/(\d+)", text)
    if not rates or not steps:
        return None
    return int(steps[-1][0]) * batch / (sum(rates) / len(rates)) / 60


def rows(runs_dir, eval_name, exporter, ref):
    out, errs = [], {}
    for ev in sorted(runs_dir.glob(f"*/{eval_name}")):
        res = json.loads(ev.read_text(encoding="utf-8"))
        ck = torch.load(ev.parent / res["ckpt"], map_location="cpu", weights_only=False)
        model, meta = from_checkpoint(ck)
        n_cls = len(meta["vocab"]) + 1
        w_max = 1024 * meta["height"] // 32
        c = cost.report(model, n_cls, meta["height"], meta["vocab"], exporter)
        est_arena, _ = cost.arena(model, meta["height"], w_max, n_cls)
        r = res["results"]
        rs = real_sets(r)
        row = {
            "run": ev.parent.name, "width": meta["width"], "classes": n_cls,
            "bottleneck": meta["bottleneck"], "height": meta["height"],
            "qat": bool(res.get("qat")), "step": res["step"], "params": c["params"],
            "weights_int8_est": c["weights_int8"], "weights_fp32_est": c["weights_fp32"],
            "arena_est_max": est_arena, "w_max": w_max,
            "macs_max": cost.macs(model, meta["height"], w_max),
            "synth_fp32": r["fp32/synth-val"]["cer"], "synth_int8": r["int8/synth-val"]["cer"],
            "synth_line_fp32": r["fp32/synth-val"]["line_acc"],
            "synth_by_cat_fp32": r["fp32/synth-val"].get("by_cat"),
            "real_sets": list(rs),
            "real_fp32": pooled(r, "fp32", rs), "real_int8": pooled(r, "int8", rs),
            "real_w8i": pooled(r, "w8i", rs),
            "real_chars": sum(r[f"fp32/{s}"]["chars"] for s in rs if f"fp32/{s}" in r),
            "train_minutes": train_minutes(ev.parent / "train.log",
                                           int((ck.get("args") or {}).get("batch", 96))),
        }
        for s in (*REAL, *PROVISIONAL, "vita-test", "psp-test", "psp-test-inv"):
            for v in ("fp32", "int8"):
                if f"{v}/{s}" in r:
                    row[f"{s}_{v}"] = r[f"{v}/{s}"]["cer"]
                    row[f"{s}_line_{v}"] = r[f"{v}/{s}"]["line_acc"]
                    if r[f"{v}/{s}"].get("cer_unk_ok") is not None:
                        row[f"{s}_unk_ok_{v}"] = r[f"{v}/{s}"]["cer_unk_ok"]
        if exporter is not None:
            row.update({k: v for k, v in c.items() if k.startswith(("blob_", "rt_arena_"))})
            row["footprint"] = row["blob_int8"] + row[f"rt_arena_w8f_{cost.WIDTHS[-1]}"]
        else:
            row["footprint"] = row["weights_int8_est"] + row["arena_est_max"]
        errs[row["run"]] = line_errors(res, "int8", rs)
        out.append(row)
    if ref in errs:
        for row in out:
            if row["run"] != ref:
                row["delta_vs_ref"] = paired_delta(errs[row["run"]], errs[ref])
    for a in out:
        a["pareto"] = a["real_int8"] is not None and not any(
            b is not a and b["real_int8"] is not None
            and b["footprint"] <= a["footprint"] and b["real_int8"] <= a["real_int8"]
            and (b["footprint"] < a["footprint"] or b["real_int8"] < a["real_int8"])
            for b in out)
    return out


def pct(x):
    return "–" if x is None else f"{100 * x:.2f}%"


def kib(x):
    return f"{x / 1024:,.0f}"


def markdown(rs, exact, ref):
    size_cols = ("int8 blob KiB | W8F arena KiB | footprint KiB" if exact else
                 "int8 weights KiB (est.) | non-streaming arena KiB (est.) | footprint KiB (est.)")
    head = (f"| run | width | classes | bneck | H | QAT | params | {size_cols} | MMAC (longest line) | "
            f"synth-val CER fp32 / int8 | vita-test CER (as is) | vita-test-inv CER | real-hand-x CER | "
            f"real CER fp32 / int8 | Δ vs {ref} (95% CI) | Pareto |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for a in sorted(rs, key=lambda a: (a["footprint"], a["run"])):
        if exact:
            sizes = f"{kib(a['blob_int8'])} | {kib(a[f'rt_arena_w8f_{cost.WIDTHS[-1]}'])}"
        else:
            sizes = f"{kib(a['weights_int8_est'])} | {kib(a['arena_est_max'])}"
        d = a.get("delta_vs_ref")
        delta = "ref" if a["run"] == ref else ("–" if d is None else
                                               f"{100 * d[0]:+.2f} ({100 * d[1]:+.2f}, {100 * d[2]:+.2f})")
        lines.append(
            f"| {a['run']} | {a['width']} | {a['classes']} | {a['bottleneck']} | {a['height']} | "
            f"{'yes' if a['qat'] else ''} | {a['params']:,} | {sizes} | {kib(a['footprint'])} | "
            f"{a['macs_max'] / 1e6:.0f} | {pct(a['synth_fp32'])} / {pct(a['synth_int8'])} | "
            f"{pct(a.get('vita-test_int8'))} | {pct(a.get('vita-test-inv_int8'))} | "
            f"{pct(a.get('real-hand-x_int8'))} | "
            f"{pct(a['real_fp32'])} / {pct(a['real_int8'])} | {delta} | "
            f"{'**yes**' if a['pareto'] else ''} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=HERE.parent / "runs")
    ap.add_argument("--eval", default="eval_ckpt_last.json")
    ap.add_argument("--ref", default="base_s1")
    ap.add_argument("--export", help="runtime's train/export.py (default: next to this file)")
    ap.add_argument("--json", type=Path)
    args = ap.parse_args()
    torch.set_num_threads(1)
    exporter = cost.runtime_exporter(args.export)
    rs = rows(args.runs, args.eval, exporter, args.ref)
    print(markdown(rs, exporter is not None, args.ref))
    if args.json:
        args.json.write_text(json.dumps(rs, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
