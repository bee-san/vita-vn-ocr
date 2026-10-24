#!/usr/bin/env python3
"""stride16_clannad_dialogue.py: the locked report's CLANNAD dialogue pairs (lines that are not a speaker name, the
references' own flag), computed from eval/clannad_refs.py score's stored per-line errors of the one attempt, with the
same page bootstrap (eval/vocr3_report.py paired_bootstrap; 10,000 resamples, seed 0). Nothing is read again:
the input is the attempt's score.<suffix>.private.jsonl and the frozen refs.jsonl. Output: counts, rates and
intervals only.

  stride16_clannad_dialogue.py --work WORK --private SCORE.private.jsonl --pair CAND:BASE ... --out OUT.json
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import metrics  # noqa: E402
import vocr3_report as R  # noqa: E402


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding="utf-8").splitlines() if x.strip()]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--private", required=True)
    ap.add_argument("--pair", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    refs = {x["id"]: x for x in read_jsonl(Path(a.work) / "refs.jsonl") if x["status"] == "ref"}
    rows = read_jsonl(a.private)
    if {r["id"] for r in rows} != set(refs):
        raise SystemExit("the private scores hold other lines than the frozen references")
    dlg = [r for r in rows if not refs[r["id"]]["name"]]
    out = {"kind": "stride16-clannad-dialogue", "lines": len(dlg), "pages": len({r["page"] for r in dlg}),
           "bootstrap": {"unit": "page", "reps": 10000, "seed": 0}, "pairs": {}}
    for p in a.pair:
        cand, base = p.split(":")
        sums = {}
        for r in dlg:
            n = len(metrics.norm(r["ref"]))
            s = sums.setdefault(r["page"], [0, 0, 0])
            s[0] += r["errors"][cand]
            s[1] += r["errors"][base]
            s[2] += n
        ca = {pg: (v[0], v[2]) for pg, v in sums.items()}
        cb = {pg: (v[1], v[2]) for pg, v in sums.items()}
        res = R.paired_bootstrap(ca, cb, 10000, 0)
        out["pairs"][p] = {"cer_delta": res["delta"], "cer_ci95": res["ci95"],
                           "candidate_errors": res["candidate_errors"], "baseline_errors": res["baseline_errors"],
                           "chars": res["chars"], "pages": res["units"]}
        print(p, json.dumps(out["pairs"][p]), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
