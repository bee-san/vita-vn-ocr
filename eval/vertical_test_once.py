#!/usr/bin/env python3
"""The vertical test set, read once (eval/vertical_protocol.json "test", amendment 4).

  vertical_test_once.py --sets DIR --private DIR --blob-v V.vocr --blob-v3500 V3500.vocr --blob-n N.vocr \\
      --cli runtime/build/vocr_cli --attempts DIR --out eval/vertical_test_20261018.json

Before anything is read it writes DIR/vertical-test.json (the attempt record: time, git head, the inputs' sha256);
an existing record refuses the run. Then: the selection (finder P1 + VOCR5-V, fixed columns and end to end) and,
report only on the same read, R* (a1 + VOCR5-N, fixed and end to end with P1) and VOCR5-V3500 (fixed columns).
Public cohorts (VNDB and manga pages) and the private one (the owner's frames) are scored apart and pooled for VN;
the output holds aggregate scores and intervals only for the private part (no text of the owner's frames).
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", required=True, help="dir with columns_test.jsonl and regions_test.jsonl (public)")
    ap.add_argument("--private", required=True, help="dir with the private columns_test.jsonl / regions_test.jsonl")
    ap.add_argument("--blob-v", required=True)
    ap.add_argument("--blob-v3500", required=True)
    ap.add_argument("--blob-n", required=True)
    ap.add_argument("--cli", required=True)
    ap.add_argument("--attempts", required=True)
    ap.add_argument("--work", required=True, help="private dir for the per-row outputs")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rec = Path(args.attempts) / "vertical-test.json"
    if rec.exists():
        raise SystemExit(f"{rec} exists: the test set has been read already")
    inputs = {str(p): sha(p) for p in (Path(args.sets) / "columns_test.jsonl", Path(args.sets) / "regions_test.jsonl",
                                        Path(args.private) / "columns_test.jsonl",
                                        Path(args.private) / "regions_test.jsonl", args.blob_v, args.blob_v3500,
                                        args.blob_n)}
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).decode().strip()
    Path(args.attempts).mkdir(parents=True, exist_ok=True)
    rec.write_text(json.dumps({"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git": head,
                               "inputs": inputs}, indent=1) + "\n")
    import vertical as EV
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    cli = os.path.abspath(args.cli)

    def model(blob):
        return f"cmd:{cli} {os.path.abspath(blob)} --mode w8i --stdin"

    runs = {
        "V_fixed": ("columns", "rot", model(args.blob_v), None),
        "V_e2e_P1": ("e2e", "rot", model(args.blob_v), "py1"),
        "Rstar_fixed": ("columns", "a1", model(args.blob_n), None),
        "Rstar_e2e_P1": ("e2e", "a1", model(args.blob_n), "py1"),
        "V3500_fixed": ("columns", "rot", model(args.blob_v3500), None),
    }
    out = {"attempt": json.loads(rec.read_text()), "runs": {}}
    for name, (kind, arm, mdl, finder) in runs.items():
        for part, d in (("public", args.sets), ("private", args.private)):
            manifest = Path(d) / ("columns_test.jsonl" if kind == "columns" else "regions_test.jsonl")
            dest = work / f"{name}_{part}.json"
            ns = argparse.Namespace(manifest=str(manifest), arm=arm, model=mdl, out=str(dest), limit=0,
                                    finder=finder)
            (EV.cmd_columns if kind == "columns" else EV.cmd_e2e)(ns)
        pub = json.load(open(work / f"{name}_public.json"))
        prv = json.load(open(work / f"{name}_private.json"))
        rows = pub["rows"] + prv["rows"]
        res = {"scores_public": {k: {kk: vv for kk, vv in v.items() if kk != "clusters"}
                                 for k, v in pub["scores"].items()},
               "scores_private": {k: {kk: vv for kk, vv in v.items() if kk != "clusters"}
                                  for k, v in prv["scores"].items()},
               "scores_pooled": {k: {kk: vv for kk, vv in v.items() if kk != "clusters"}
                                 for k, v in EV.score_rows(rows).items()}}
        json.dump({"rows": rows}, open(work / f"{name}_rows.json", "w"), ensure_ascii=False)
        out["runs"][name] = res
    # the selection against R*, pooled VN (cluster bootstrap) and manga (page bootstrap)
    for a, b in (("Rstar_fixed", "V_fixed"), ("Rstar_e2e_P1", "V_e2e_P1"), ("V_fixed", "V3500_fixed")):
        ra = json.load(open(work / f"{a}_rows.json"))["rows"]
        rb = json.load(open(work / f"{b}_rows.json"))["rows"]
        out.setdefault("diffs", {})[f"{b} - {a}"] = {c: EV.cluster_diff(ra, rb, c) for c in ("vn", "manga")}
    rec_doc = json.loads(rec.read_text())
    rec_doc["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rec.write_text(json.dumps(rec_doc, indent=1) + "\n")
    out["attempt"] = rec_doc
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    for name, r in out["runs"].items():
        p = r["scores_pooled"]
        print(name, {k: f"{100 * v['cer']:.2f}% ({v['errs']}/{v['chars']})" for k, v in p.items() if k != "all"})
    for k, v in out["diffs"].items():
        print(k, {c: (round(x["diff"], 2), round(x["lo"], 2), round(x["hi"], 2)) for c, x in v.items()})


if __name__ == "__main__":
    main()
