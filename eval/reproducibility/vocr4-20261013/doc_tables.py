#!/usr/bin/env python3
"""Markdown tables for docs/vocr4.md from the committed decision and selection files (reported numbers only).

  doc_tables.py [--date 20261014]
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
E = ROOT / "eval"


def pct(x):
    return f"{100 * x:.2f}%"


def fmt(d):
    return f"{d['delta_points']:+.3f} ({d['ci95'][0]:+.3f} to {d['ci95'][1]:+.3f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="20261014")
    a = ap.parse_args()
    print("| size | decision | development score | teacher_val | devnative | pairs (seed 1, seed 2) | adopted |")
    print("|---|---|---|---|---|---|---|")
    for size in ("FL10", "H15", "W20"):
        for kind in ("real", "synth"):
            p = E / f"vocr4_{kind}_{size}_{a.date}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text(encoding="utf-8"))
            t = d["total"]
            pairs = ", ".join(f"{v['development_score']['delta_points']:+.3f}" for v in d["pairs"].values())
            print(f"| {size} | {d['x']} -> {d['y']} | {fmt(t['development_score'])} | {fmt(t['teacher_val'])} | "
                  f"{fmt(t['devnative'])} | {pairs} | {'yes' if d['adopted'] else 'no'} |")
    for size in ("FL10", "H15", "W20"):
        p = E / f"vocr4_selection_{size}_{a.date}.json"
        if not p.exists():
            continue
        s = json.loads(p.read_text(encoding="utf-8"))
        print(f"\n{size}: candidates {', '.join(s['candidates'])} (arm {s['final_arm']}); selected **{s['selected']}** "
              f"(`{s['selected_blob_sha256'][:8]}…`)\n")
        print("| run | arm | teacher_val CER | devnative CER | development score "
              "| blob B / arena B / est. ms high, 333 MHz |")
        print("|---|---|---|---|---|---|")
        for r in sorted(s["runs"], key=lambda r: r["score"]):
            b = r["budget"]
            tag = " **(selected)**" if r["run"] == s["selected"] else (" (candidate)" if r["candidate"] else "")
            print(f"| {r['run']}{tag} | {r['arm']} | {pct(r['teacher_val_cer'])} | {pct(r['devnative_cer'])} | "
                  f"{pct(r['score'])} | {b['blob_bytes']:,} / {b['arena_w8i']:,} / {b['ms_333_high']:.0f} |")


if __name__ == "__main__":
    main()
