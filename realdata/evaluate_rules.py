#!/usr/bin/env python3
"""Score the voters and the voting rules against hand-checked lines.

  evaluate_rules.py --lines FILE [--hand realdata/manifests/handcheck.jsonl] [--set dev]

For every hand-checked line with a confident truth: each voter's exact-line accuracy
(on key()), and for each rule in vote.RULES its yield (accepted / checked lines) and
precision (accepted labels whose key() equals the truth's). Lines whose truth was left
uncertain are counted but not scored. Broken down by screenshot kind.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vote  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lines", type=Path, required=True)
    ap.add_argument("--hand", type=Path, default=HERE / "manifests" / "handcheck.jsonl")
    ap.add_argument("--set", default="dev")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    hand = {h["id"]: h for h in map(json.loads, args.hand.read_text(encoding="utf-8").splitlines())
            if h and h.get("set") == args.set}
    recs = {r["id"]: r for r in map(json.loads, args.lines.read_text(encoding="utf-8").splitlines())}
    rows = [(recs[i], h) for i, h in hand.items() if i in recs]
    unsure = sum(h["truth"] is None for _, h in rows)
    rows = [(r, h) for r, h in rows if h["truth"] is not None]
    out = {"checked": len(rows) + unsure, "uncertain": unsure, "voters": {}, "rules": {}}
    kinds = sorted({r["kind"] for r, _ in rows}) + ["all"]
    for name in ("v5s", "manga", "v6s", "meiki"):
        acc = defaultdict(lambda: [0, 0])
        for r, h in rows:
            if name in r["r"]:
                for k in (r["kind"], "all"):
                    acc[k][0] += vote.key(r["r"][name][0]) == vote.key(h["truth"])
                    acc[k][1] += 1
        out["voters"][name] = {k: f"{a}/{n} = {a / n:.1%}" for k, (a, n) in acc.items() if n}
    for name, rule in vote.RULES.items():
        st = defaultdict(lambda: [0, 0, 0])            # accepted, correct, total
        errs = []
        for r, h in rows:
            lab, how = rule({k: v[0] for k, v in r["r"].items()})
            for k in (r["kind"], "all"):
                st[k][2] += 1
                if lab is not None:
                    st[k][0] += 1
                    st[k][1] += vote.key(lab) == vote.key(h["truth"])
            if lab is not None and vote.key(lab) != vote.key(h["truth"]):
                errs.append(f"{r['id']}: {lab} != {h['truth']}")
        out["rules"][name] = {k: {"yield": f"{a}/{n} = {a / n:.1%}",
                                  "precision": f"{c}/{a} = {c / a:.1%}" if a else "-"}
                              for k, (a, c, n) in st.items()}
        out["rules"][name]["errors"] = errs
    print(f"checked {out['checked']} lines ({unsure} uncertain, not scored)")
    print("voter exact-line accuracy:")
    for name, d in out["voters"].items():
        print(f"  {name:6s} " + "  ".join(f"{k}: {d.get(k, '-')}" for k in kinds))
    print("rules (yield / precision):")
    for name, d in out["rules"].items():
        print(f"  {name:14s} " + "  ".join(f"{k}: {d[k]['yield']} / {d[k]['precision']}" for k in kinds if k in d))
    if args.json:
        args.json.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
