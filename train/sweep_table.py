#!/usr/bin/env python3
"""Collect sweep runs into a Pareto table (Markdown + JSON).

  sweep_table.py [--runs runs] [--eval eval_ckpt_last.json] [--ref base_s1]
                 [--export path/to/runtime/train/export.py] [--json docs/arch-sweep.json]
                 [--doc docs/arch-sweep.md]

--doc rewrites the blocks between <!-- BEGIN name --> and <!-- END name --> markers in
the Markdown file: "results" (this table), "styles" (per-style CER) and "cost"
(cost.table(), the size and speed of the sweep grid).

For every runs/<name>/ with an eval JSON (sweep_eval.py) this reads the config from
the checkpoint, measures its cost with cost.report and prints one row: parameters,
int8 blob bytes and the runtime's exact arena (cost.runtime_sizes, when runtime's
exporter is importable; otherwise cost.py's estimates, marked "est."), MACs at the
longest line (1024 px at height 32, i.e. 768 px at height 24 for the same physical
text), synth-val CER and real CER (fp32 and int8 weights).

The decision metric ("real CER") pools eval's vita-test and psp-test with their
dark-on-light lines inverted by style tag (vita-test-inv, psp-test-inv) and the
research's hand lines from 17 other games (real-hand-x): errors / reference
characters, int8 (W8F) weights. Only sets that every run was scored on enter the
pool, and a set scored on different manifest versions (sha256 in the eval JSON)
across runs is reported and left out. The v1 recipe renders only light text, so the
as-is sets mostly measure that data gap (which synth2 and the real fine-tuning
close), not capacity; they are shown alongside. Without eval/ results the
provisional real-hand + real-pseudo pool is used.
"Δ vs ref" is the pooled real CER minus the reference run's, with a 95% paired
bootstrap interval over lines (10,000 resamples, seed 0). It is computed from the same
per-line errors as the pooled CER columns (eval/metrics.py on eval's sets), so the point
estimate equals the difference of the table's own pooled CERs.

Runs are grouped by training data (the checkpoint's --real args): by default the table
shows the synthetic-only runs (plus v1); --data real shows the runs that mixed in
realdata's real_train.npz. real-hand-x shares games and screenshots with realdata's
training split, so for real-data runs it is left out of the pool, which is then
vita-test-inv + psp-test-inv (both verified free of realdata's games and sessions).

Two Pareto fronts are
taken over the runs with the reference run's step budget only (other budgets, e.g.
v1's 32k steps, are shown for scale): memory (on-console footprint = int8 blob + W8F
arena) and speed (estimated Cortex-A9 ms for a 640-px line, about the p90 real line
width; cost.a9_ms). A row is on a front when no other such row is at least as good on
both axes and better on one.
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

REAL = ("vita-test-inv", "psp-test-inv", "real-hand-x")
PROVISIONAL = ("real-hand", "real-pseudo")


def pooled(res, variant, sets=REAL):
    e = sum(res[f"{variant}/{s}"]["errors"] for s in sets if f"{variant}/{s}" in res)
    n = sum(res[f"{variant}/{s}"]["chars"] for s in sets if f"{variant}/{s}" in res)
    return e / n if n else None


def train_data(ck):
    """'synth' or 'synth+real' from a checkpoint's recorded args (v1 has none: synth)."""
    real = ((ck.get("args") or {}).get("real") or "").strip()
    return "synth+real" if real else "synth"


def common_sets(evals, pool=REAL):
    """The pool sets every eval JSON has, on one manifest version each (else PROVISIONAL)."""
    keep = []
    for s in pool:
        if not all(f"int8/{s}" in e["results"] for e in evals):
            continue
        shas = {e.get("manifests", {}).get(s) for e in evals}
        if len(shas) > 1:
            print(f"warning: {s} was scored on {len(shas)} manifest versions; left out of the pool "
                  "(rerun sweep_eval on the stale runs)", file=sys.stderr)
            continue
        keep.append(s)
    return tuple(keep) or PROVISIONAL


def line_errors(res, variant="int8", sets=REAL):
    """{(set, line id): (errors, reference chars)} of one eval JSON, under the scorer that
    gave the set's pooled errors: the per-line counts sweep_eval stores, or for an older
    JSON a re-score with eval/metrics.py on eval's sets and score_norm on the others.
    Warns if the lines do not add up to the set's errors."""
    metrics = None
    out = {}
    for s in sets:
        r = res["results"].get(f"{variant}/{s}", {})
        tot = [0, 0]
        for d in r.get("lines_detail", []):
            if "errors" in d:
                e, n = d["errors"], d["n"]
            elif s.startswith(("vita-test", "psp-test")):
                if metrics is None:
                    from sweep_eval import eval_modules
                    metrics = eval_modules()[1]
                p, g = metrics.norm(d["pred"]), metrics.norm(d["ref"])
                e, n = metrics.lev(p, g), len(g)
            else:
                p, g = score_norm(d["pred"]), score_norm(d["ref"])
                e, n = edit_distance(list(p), list(g)), len(g)
            out[(s, d["id"])] = (e, n)
            tot[0] += e
            tot[1] += n
        if r.get("lines_detail") and tot != [r["errors"], r["chars"]]:
            print(f"warning: {res.get('run')} {variant}/{s}: lines add up to {tot[0]}/{tot[1]}, "
                  f"the set reports {r['errors']}/{r['chars']}", file=sys.stderr)
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


def rows(runs_dir, eval_name, exporter, ref, data="synth"):
    out, errs = [], {}
    paths, evals, cks = [], [], []
    for p in sorted(runs_dir.glob(f"*/{eval_name}")):
        res = json.loads(p.read_text(encoding="utf-8"))
        ck = torch.load(p.parent / res["ckpt"], map_location="cpu", weights_only=False)
        if train_data(ck) == data or (data == "synth+real" and p.parent.name == "v1"):
            paths.append(p)
            evals.append(res)
            cks.append(ck)
    pool = REAL if data == "synth" else tuple(s for s in REAL if s != "real-hand-x")
    rs = common_sets(evals, pool)
    for ev, res, ck in zip(paths, evals, cks):
        model, meta = from_checkpoint(ck)
        arch = {k: meta[k] for k in ("depth", "ctx", "ctx_k")}
        n_cls = len(meta["vocab"]) + 1
        w_max = 1024 * meta["height"] // 32
        c = cost.report(model, n_cls, meta["height"], meta["vocab"], exporter)
        est_arena, _ = cost.arena(model, meta["height"], w_max, n_cls)
        r = res["results"]
        row = {
            "run": ev.parent.name, "data": train_data(ck), "width": meta["width"], "classes": n_cls,
            "bottleneck": meta["bottleneck"], "height": meta["height"], **arch,
            "a9_ms_640": c["a9_ms_640"], "a9_ms_max": c["a9_ms_1024"],
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
        styles = r.get("int8/vita-test", {}).get("styles") or {}
        row["vita_bright_on_dark_int8"] = styles.get("bright-on-dark")
        row["vita_styles_int8"] = styles
        row["psp_styles_int8"] = r.get("int8/psp-test", {}).get("styles") or {}
        row["manifests"] = res.get("manifests", {})
        con = [s for s in ("vita-test-con", "psp-test-con") if f"int8/{s}" in r]
        if len(con) == 2:
            row["console_int8"] = pooled(r, "int8", tuple(con))
        for s in (*REAL, *PROVISIONAL, "vita-test", "psp-test", "psp-test-inv", "vita-test-con",
                  "psp-test-con"):
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
        if row["data"] == "synth+real":       # real-hand-x shares games with realdata's train split
            for k in [k for k in row if k.startswith("real-hand-x_")]:
                row[k] = None
        errs[row["run"]] = line_errors(res, "int8", rs)
        out.append(row)
    if ref in errs:
        for row in out:
            if row["run"] != ref:
                row["delta_vs_ref"] = paired_delta(errs[row["run"]], errs[ref])
    budget = next((a["step"] for a in out if a["run"] == ref), None)
    matched = [a for a in out if budget is None or a["step"] == budget]
    def front(a, key):
        return a in matched and a["real_int8"] is not None and not any(
            b is not a and b["real_int8"] is not None
            and key(b) <= key(a) and b["real_int8"] <= a["real_int8"]
            and (key(b) < key(a) or b["real_int8"] < a["real_int8"])
            for b in matched)

    for a in out:
        a["matched_budget"] = a in matched
        a["pareto"] = front(a, lambda r: r["footprint"])
        a["pareto_speed"] = front(a, lambda r: r["a9_ms_640"][0])
    return out


def pct(x):
    return "–" if x is None else f"{100 * x:.2f}%"


def kib(x):
    return f"{x / 1024:,.0f}"


def pareto_cell(a):
    if not a["matched_budget"]:
        return f"({a['step']:,} steps)"
    return " / ".join(n for n, k in (("memory", "pareto"), ("speed", "pareto_speed")) if a[k])


def knobs(a):
    """The non-v1 architecture knobs of a row, e.g. 'w1.5 d2'."""
    parts = [f"w{a['width']:g}" if a["width"] != 1.0 else "",
             f"b{a['bottleneck']}" if a["bottleneck"] != 96 else "",
             f"h{a['height']}" if a["height"] != 32 else "",
             f"{a['classes'] - 1} chars" if a["classes"] != 5355 else "",
             f"d{a['depth']}" if a["depth"] != 1 else "",
             f"ctx{a['ctx']}" if a["ctx"] != 2 else "", f"k{a['ctx_k']}" if a["ctx_k"] != 5 else "",
             "QAT" if a["qat"] else ""]
    return " ".join(p for p in parts if p) or "v1 layout"


def markdown(rs, exact, ref):
    size_cols = ("int8 blob KiB | W8F arena KiB | footprint KiB" if exact else
                 "int8 weights KiB (est.) | non-streaming arena KiB (est.) | footprint KiB (est.)")
    head = (f"| run | knobs | params | {size_cols} | MMAC (longest line) | est. A9 ms, 640-px line | "
            f"synth-val CER fp32 / int8 | vita-test CER (as is) | of which bright-on-dark | "
            f"vita-test-inv CER | psp-test CER (as is) | psp-test-inv CER | real-hand-x CER | "
            f"vita+psp CER, console polarity | "
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
            f"| {a['run']} | {knobs(a)} | {a['params']:,} | {sizes} | {kib(a['footprint'])} | "
            f"{a['macs_max'] / 1e6:.0f} | {a['a9_ms_640'][1]:.0f}-{a['a9_ms_640'][0]:.0f} | "
            f"{pct(a['synth_fp32'])} / {pct(a['synth_int8'])} | "
            f"{pct(a.get('vita-test_int8'))} | {pct(a.get('vita_bright_on_dark_int8'))} | "
            f"{pct(a.get('vita-test-inv_int8'))} | {pct(a.get('psp-test_int8'))} | "
            f"{pct(a.get('psp-test-inv_int8'))} | {pct(a.get('real-hand-x_int8'))} | "
            f"{pct(a.get('console_int8'))} | "
            f"{pct(a['real_fp32'])} / {pct(a['real_int8'])} | {delta} | "
            f"{pareto_cell(a)} |")
    return "\n".join(lines)


STYLE_COLS = ("bright-on-dark", "dark-on-light", "outlined", "coloured", "name", "coloured-name",
              "furigana", "mincho", "small", "no-box")


def style_table(rs):
    """Per-style int8 CER on vita-test and psp-test as is (eval's tags), matched-budget runs
    plus any reference rows."""
    head = ("| run | set | " + " | ".join(STYLE_COLS) + " |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for a in sorted(rs, key=lambda a: (a["footprint"], a["run"])):
        for name, key in (("vita-test", "vita_styles_int8"), ("psp-test", "psp_styles_int8")):
            st = a.get(key) or {}
            if st:
                lines.append(f"| {a['run']} | {name} | " + " | ".join(pct(st.get(c)) for c in STYLE_COLS) + " |")
    return "\n".join(lines)


def quant_table(rs):
    """fp32 vs int8 weights (W8F) vs W8I, pooled real CER and synth-val CER, per run."""
    head = ("| run | QAT | real CER fp32 | real CER int8 W8F | real CER W8I (sim.) | "
            "synth-val CER fp32 | synth-val CER int8 |")
    lines = [head, "|" + "---|" * (head.count("|") - 1)]
    for a in sorted(rs, key=lambda a: (not a["qat"], a["run"])):
        lines.append(f"| {a['run']} | {'yes' if a['qat'] else ''} | {pct(a['real_fp32'])} | "
                     f"{pct(a['real_int8'])} | {pct(a['real_w8i'])} | {pct(a['synth_fp32'])} | "
                     f"{pct(a['synth_int8'])} |")
    return "\n".join(lines)


def replace_block(text, name, body):
    begin, end = f"<!-- BEGIN {name} -->", f"<!-- END {name} -->"
    i, j = text.find(begin), text.find(end)
    if i < 0 or j < i:
        raise SystemExit(f"markers for {name!r} not found")
    return text[:i + len(begin)] + "\n" + body + "\n" + text[j:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=Path, default=HERE.parent / "runs")
    ap.add_argument("--eval", default="eval_ckpt_last.json")
    ap.add_argument("--ref", default="base_s1")
    ap.add_argument("--export", help="runtime's train/export.py (default: next to this file)")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--doc", type=Path, help="Markdown file with BEGIN/END table markers")
    ap.add_argument("--data", choices=("synth", "real"), default="synth",
                    help="synthetic-only runs (default) or the runs that mixed in real data")
    args = ap.parse_args()
    torch.set_num_threads(1)
    exporter = cost.runtime_exporter(args.export)
    data = "synth+real" if args.data == "real" else "synth"
    rs = rows(args.runs, args.eval, exporter, args.ref, data)
    table = markdown(rs, exporter is not None, args.ref)
    print(table)
    if args.json:
        args.json.write_text(json.dumps(rs, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.doc:
        text = args.doc.read_text(encoding="utf-8")
        blocks = ((("results", table), ("styles", style_table(rs)), ("quant", quant_table(rs)),
                   ("cost", cost.table(exporter))) if data == "synth" else
                  (("results-real", table), ("styles-real", style_table(rs))))
        for name, body in blocks:
            if f"<!-- BEGIN {name} -->" in text:
                text = replace_block(text, name, body)
        args.doc.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
