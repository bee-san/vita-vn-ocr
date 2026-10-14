#!/usr/bin/env python3
"""Small PNG curves for docs/vocab-expand.md (<100 KB each). Needs matplotlib (a throwaway venv; not a
repo dependency): see docs/vocab-expand.md's Reproduce. Only the generated PNGs are committed.

  vocab_expand_plots.py coverage --curves curves.json --out docs/benchmarks/vocab-coverage.png
  vocab_expand_plots.py cost     --tiers tier_rows.json --out docs/benchmarks/vocab-cost.png
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TIERS = (3501, 4987, 5630, 6500, 7500)


def save(fig, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80, bbox_inches="tight")
    kb = Path(out).stat().st_size / 1024
    print(f"{out}: {kb:.1f} KB")
    if kb > 100:
        raise SystemExit(f"{out} is {kb:.0f} KB (> 100 KB)")


def cmd_coverage(args):
    d = json.loads(Path(args.curves).read_text(encoding="utf-8"))
    cur = d["curve"]
    x = [r["classes"] for r in cur]
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for g, style in (("VN design", "-"), ("VN held out", "--")):
        ax.plot(x, [r[g]["corpus"] for r in cur], style, label=f"{g} (corpus)")
        ax.plot(x, [r[g]["p95"] for r in cur], style, alpha=0.45, label=f"{g} (p95 title)")
    for name, mk in (("FL10", "o"), ("H15", "s")):
        r = d["fixed"][name]
        ax.scatter([r["classes"]], [r["VN design"]["corpus"]], marker=mk, zorder=5, label=f"{name} {r['classes']}")
    for t in TIERS:
        ax.axvline(t, color="0.85", lw=0.8, zorder=0)
    ax.set_xlabel("output classes (CTC blank included)")
    ax.set_ylabel("VN chars missed per 1,000")
    ax.set_yscale("symlog", linthresh=0.05)
    ax.set_title("Visual-novel coverage vs vocabulary size (Jiten, occurrence-weighted)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    save(fig, args.out)


def cmd_cost(args):
    rows = json.loads(Path(args.tiers).read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    for w, label, mk in ((100, "width 1.0 (FL10 shape)", "o"), (150, "width 1.5 (H15 shape)", "s")):
        pts = sorted((r["config"]["classes"], r["blob_bytes"] / 1e6, r)
                     for n, r in rows.items() if r.get("config") and r["config"]["width"] == w / 100
                     and r["config"]["bottleneck"] == 96)
        ax.plot([c for c, _, _ in pts], [b for _, b, _ in pts], f"{mk}-", label=label)
    ax.axhspan(0, 1.0, color="tab:green", alpha=0.08)
    ax.axhspan(1.0, 4.0, color="tab:orange", alpha=0.07)
    ax.text(3600, 0.5, "Small < 1 MB", fontsize=7, color="tab:green")
    ax.text(3600, 2.3, "Normal \u2264 4 MB", fontsize=7, color="tab:orange")
    for t in TIERS:
        ax.axvline(t, color="0.9", lw=0.8, zorder=0)
    ax.set_xlabel("output classes")
    ax.set_ylabel("int8 blob (MB)")
    ax.set_title("Head cost vs vocabulary size (exact export)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    save(fig, args.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("coverage")
    c.add_argument("--curves", required=True)
    c.add_argument("--out", required=True)
    k = sub.add_parser("cost")
    k.add_argument("--tiers", required=True)
    k.add_argument("--out", required=True)
    args = ap.parse_args()
    {"coverage": cmd_coverage, "cost": cmd_cost}[args.cmd](args)


if __name__ == "__main__":
    main()
