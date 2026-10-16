#!/usr/bin/env python3
"""CLANNAD with Google Lens's reading as the reference (report only; issue #90's "Lens = 100%" view).

  vocr5_clannad_lens.py --private SCORE.private.jsonl --refs WORK/refs.jsonl --out VIEW.json

SCORE.private.jsonl is eval/clannad_refs.py score's private per-line file of an attempt (every scored line: its page,
the frozen reference and each system's reading, Lens's included); refs.jsonl the frozen references (the name flag).
For every system and two subsets of the scored lines (all; dialogue = the lines the references do not flag as a
speaker name): the edit distance between the system's reading and Lens's reading over Lens's characters (both through
eval/metrics.norm, the scoring rule), and the share of lines it reads exactly as Lens does, with 95% intervals from
whole-page bootstraps (eval/clannad_refs.py's: 10,000 resamples, seed 0). This measures agreement with Lens, not
accuracy: where Lens is wrong a system that is right counts as an error. Public output: counts and rates only.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import clannad_refs as CR  # noqa: E402
import metrics  # noqa: E402

LENS = "Lens"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def view(rows, names, systems):
    out = {}
    for sub, keep in (("all", lambda r: True), ("dialogue", lambda r: not names[r["id"]])):
        sel = [r for r in rows if keep(r)]
        pages = sorted({r["page"] for r in sel})
        idx = CR.boot_idx(len(pages), CR.REPS, CR.SEED)
        pos = {p: i for i, p in enumerate(pages)}
        res = {}
        for s in systems:
            e = [0] * len(pages)
            n = [0] * len(pages)
            x = [0] * len(pages)
            m = [0] * len(pages)
            for r in sel:
                i = pos[r["page"]]
                t = metrics.norm(r["preds"][LENS] or "")
                p = metrics.norm(r["preds"][s] or "")
                e[i] += metrics.lev(p, t)
                n[i] += len(t)
                x[i] += p == t
                m[i] += 1
            res[s] = {"lines": sum(m), "lens_chars": sum(n), "edits": sum(e),
                      "cer_vs_lens": round(100 * sum(e) / sum(n), 4) if sum(n) else None,
                      "cer_vs_lens_ci95": CR.rate_ci(e, n, idx),
                      "same_as_lens": round(100 * sum(x) / sum(m), 4), "same_as_lens_ci95": CR.rate_ci(x, m, idx)}
        out[sub] = {"pages": len(pages), "lines": len(sel), "systems": res}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--private", required=True)
    ap.add_argument("--refs", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(x) for x in Path(a.private).read_text(encoding="utf-8").splitlines() if x.strip()]
    refs = [json.loads(x) for x in Path(a.refs).read_text(encoding="utf-8").splitlines() if x.strip()]
    names = {r["id"]: bool(r.get("name")) for r in refs if r.get("status") == "ref"}
    if {r["id"] for r in rows} != set(names):
        raise SystemExit("the private rows are not the frozen references' scored lines")
    systems = [s for s in rows[0]["preds"] if s not in (LENS, "cached")]
    doc = {"kind": "vocr5-clannad-lens-view", "schema": 1, "report_only": True,
           "what": "each system's reading scored against Google Lens's reading of the same CLANNAD line crops (the "
                   "'Lens = 100%' view): agreement with Lens, not accuracy",
           "inputs": {"private_rows_sha256": sha256(a.private), "refs_sha256": sha256(a.refs)},
           "bootstrap": {"unit": "page", "reps": CR.REPS, "seed": CR.SEED}, "view": view(rows, names, systems)}
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    for sub, v in doc["view"].items():
        print(f"{sub}: {v['lines']} lines, {v['pages']} pages")
        for s, r in v["systems"].items():
            print(f"  {s}: {r['cer_vs_lens']:.2f}% {r['cer_vs_lens_ci95']} of {r['lens_chars']} Lens chars; "
                  f"same as Lens {r['same_as_lens']:.1f}%")


if __name__ == "__main__":
    main()
