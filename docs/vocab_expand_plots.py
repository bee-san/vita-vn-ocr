#!/usr/bin/env python3
"""Small PNG curves for docs/vocab-expand.md (<100 KB each). Needs matplotlib (a throwaway venv; not a
repo dependency): see docs/vocab-expand.md's Reproduce. Only the generated PNGs are committed.

  vocab_expand_plots.py coverage --agg agg.json --tail tail.json --out docs/benchmarks/vocab-coverage.png
  vocab_expand_plots.py cost     --tiers tier_rows.json --out docs/benchmarks/vocab-cost.png
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TIERS = {"Small": 4987, "Normal": 5631, "Large": 6459}
LISTS = ("VN", "Novel", "Anime", "Drama", "Video game", "Global")


def save(fig, out):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=80, bbox_inches="tight")
    kb = Path(out).stat().st_size / 1024
    print(f"{out}: {kb:.1f} KB")
    if kb > 100:
        raise SystemExit(f"{out} is {kb:.0f} KB (> 100 KB)")


def cmd_coverage(args):
    d = json.loads(Path(args.agg).read_text(encoding="utf-8"))
    cur = [r for r in d["curve"] if 3000 <= r["classes"] <= 8000]
    x = [r["classes"] for r in cur]
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for name in LISTS:
        ax.plot(x, [max(r[name]["chars"], 1e-4) for r in cur], "-", lw=2.2 if name == "VN" else 1.0,
                label=f"{name} list (aggregate, s = {d['zipf']['s']})")
    if args.tail:
        t = json.loads(Path(args.tail).read_text(encoding="utf-8"))
        pts = [(int(k.split()[1]), v["design"]["p95"]) for k, v in t["vocabs"].items() if k.startswith("VN ")]
        pts = sorted((n, max(p, 1e-4)) for n, p in pts if 3000 <= n <= 8000)
        ax.plot([n for n, _ in pts], [p for _, p in pts], "k:", marker="v", ms=4,
                label="95th-percentile VN title (per-deck sample)")
    for name, mk in (("FL10", "o"), ("H15", "s")):
        r = d["fixed"][name]
        ax.scatter([r["classes"]], [r["VN"]["chars"]], marker=mk, color="k", zorder=5, label=f"{name} (VN list)")
    for t, n in TIERS.items():
        ax.axvline(n, color="0.85", lw=0.8, zorder=0)
        ax.text(n, 4.5, t, fontsize=7, rotation=90, va="top", ha="right", color="0.4")
    ax.set_xlabel("output classes (CTC blank included)")
    ax.set_ylabel("characters missed per 1,000")
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 6)
    ax.set_title("Characters a VN-ordered vocab cannot emit, by Jiten list", fontsize=10)
    ax.legend(fontsize=6.5, ncol=2, loc="lower left")
    ax.grid(True, alpha=0.3)
    save(fig, args.out)


def cmd_cost(args):
    rows = json.loads(Path(args.tiers).read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    for w, label, mk in ((1.0, "width 1.0, bottleneck 96 (FL10's shape)", "o"),
                         (1.0, "width 1.0, bottleneck 64", "^"), (1.5, "width 1.5, bottleneck 96 (H15's shape)", "s")):
        b = 64 if "64" in label else 96
        pts = sorted((r["config"]["classes"], r["blob_bytes"] / 1e6) for n, r in rows.items()
                     if r.get("config") and r["config"]["width"] == w and r["config"]["bottleneck"] == b
                     and not n.startswith("t_"))
        ax.plot([c for c, _ in pts], [v for _, v in pts], f"{mk}-", ms=4, label=label)
    for n, r in rows.items():
        if n.startswith("t_") and r.get("config") and r["config"]["width"] >= 2.5:
            ax.scatter([r["classes"]], [r["blob_bytes"] / 1e6], marker="*", s=60, color="k", zorder=5)
            ax.annotate(n[2:], (r["classes"], r["blob_bytes"] / 1e6), fontsize=6.5, xytext=(4, -3),
                        textcoords="offset points")
    ax.axhline(1.0, color="tab:green", lw=0.8, ls="--")
    ax.axhline(4.0, color="tab:orange", lw=0.8, ls="--")
    ax.text(3050, 1.05, "Small < 1 MB", fontsize=7, color="tab:green")
    ax.text(3050, 3.75, "Normal \u2264 4 MB (Large \u2264 9 MB)", fontsize=7, color="tab:orange")
    ax.set_xlabel("output classes")
    ax.set_ylabel("int8 blob (MB)")
    ax.set_title("Blob size against vocabulary size (exact random-init exports)", fontsize=10)
    ax.legend(fontsize=7, loc="center right")
    ax.grid(True, alpha=0.3)
    save(fig, args.out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("coverage")
    c.add_argument("--agg", required=True)
    c.add_argument("--tail")
    c.add_argument("--out", required=True)
    k = sub.add_parser("cost")
    k.add_argument("--tiers", required=True)
    k.add_argument("--out", required=True)
    args = ap.parse_args()
    {"coverage": cmd_coverage, "cost": cmd_cost}[args.cmd](args)


if __name__ == "__main__":
    main()
