#!/usr/bin/env python3
"""A public record of desktop results (issue #169): counts and rates only, from eval/desktop.py result files.

  desktop_record.py --res DIR --set SET --ref NAME --names NAME,... --out RECORD.json [--note TEXT]

For every NAME (a result file SET_NAME.json under --res): its summary (CER, errors, characters, line recall, box
precision, cut / missed / merged / junk counts) overall, per view and per group, and its paired difference against the
reference NAME (eval/pagepath.py's cluster bootstrap: VN clusters on dev-v2, confirm-v2 and the PC test set, title
clusters on hval) overall and per view. No text, no box and no frame is written.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import desktop as D  # noqa: E402

KEEP = ("units", "errors", "chars", "lines", "boxes", "whole", "cut", "missed", "merged", "junk", "cer", "line_recall",
        "box_precision")


def slim(s):
    return {k: s[k] for k in KEEP if k in s}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--res", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--names", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    res = Path(a.res)

    def load(n):
        return json.loads((res / f"{a.set}_{n}.json").read_text(encoding="utf-8"))
    ref = load(a.ref)
    rec = {"kind": "vocr-desktop-record", "set": a.set, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "protocol_sha256": D.sha256_file(ROOT / "eval" / "desktop_protocol.json"), "repo_head": D.git_head(),
           "frames_sha256": ref["frames_sha256"], "reference": a.ref, "note": a.note, "rows": {}}
    for n in [a.ref] + [x for x in a.names.split(",") if x and x != a.ref]:
        d = load(n)
        if d["frames_sha256"] != ref["frames_sha256"]:
            raise SystemExit(f"{n}: read on other frames than the reference")
        row = {"file_sha256": D.sha256_file(res / f"{a.set}_{n}.json"),
               "det": d.get("det"), "pad": d.get("pad"), "rec": d.get("rec"),
               "all": slim(d["summary"]["all"]),
               "by_view": {k: slim(v) for k, v in d["summary"]["by_view"].items()},
               "by_group": {k: slim(v) for k, v in d["summary"]["by_group"].items()}}
        if n != a.ref:
            cmp = D.compare_docs(d, ref)
            row["against_reference"] = {k: {x: v[x] for x in ("delta", "ci95", "arm_cer", "base_cer", "clusters",
                                                                 "clusters_better", "clusters_worse") if x in v}
                                        for k, v in cmp.items()}
        rec["rows"][n] = row
    Path(a.out).write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for n, row in rec["rows"].items():
        ag = row.get("against_reference", {}).get(f"{a.set}:all", {})
        print(f"{n:28s} {row['all']['cer']:.3f}%  {ag.get('delta', '')} {ag.get('ci95', '')}", flush=True)


if __name__ == "__main__":
    main()
