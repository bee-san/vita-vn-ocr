#!/usr/bin/env python3
"""The VOCR4 locked report's spec (eval/vocr4_protocol.json "locked_report") from the committed selections.

  vocr4_make_spec.py --protocol eval/vocr4_protocol.json --selection FL10=SEL.json --selection H15=SEL.json \
                     --selection W20=SEL.json --vocr3-spec eval/vocr3_report_spec_20261013.json \
                     --vocr3-blobs DIR --release-dir DIR --blob-dir OUT_BLOBS --out spec.json

The released baselines (VOCR1, VOCR2, VOCR_H15, F20) come from --release-dir and VOCR3's three blobs from
--vocr3-blobs, each checked against the sha256 VOCR3's committed report spec gives it; the VOCR4 arms are the
selections' blobs (eval/vocr4_select.py select), checked against the selections' sha256, copied into --blob-dir
as VOCR4_<SIZE>_w8.vocr. Arms, pairs and gates are the protocol's. The run behind each VOCR4 arm goes to
<out>.runs.json, because eval/vocr3_report.py rejects spec fields it does not know.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

SIZES = ("FL10", "H15", "W20")
RELEASED = ("VOCR1", "VOCR2", "VOCR_H15", "F20")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def build(proto, selections, vocr3_spec, vocr3_blobs, release_dir, blob_dir):
    blob_dir = Path(blob_dir)
    blob_dir.mkdir(parents=True, exist_ok=True)
    old = {a["name"]: a for a in vocr3_spec["arms"]}
    arms, runs = [], {}
    for name in RELEASED + tuple(f"VOCR3_{s}" for s in SIZES):
        a = old[name]
        src = (Path(release_dir) if name in RELEASED else Path(vocr3_blobs)) / a["file"]
        if sha256(src) != a["sha256"]:
            raise SystemExit(f"{src}: sha256 differs from VOCR3's report spec ({a['sha256'][:12]})")
        shutil.copy2(src, blob_dir / a["file"])
        arms.append({"name": name, "file": a["file"], "finder": a["finder"], "sha256": a["sha256"],
                     "role": a["role"] if name in RELEASED else f"{a['role']} (VOCR3's selection)"})
    for size in SIZES:
        sel = selections[size]
        if sel.get("kind") != "vocr4-selection" or sel["size"] != size:
            raise SystemExit(f"{size}: not a VOCR4 selection of that size")
        blob = Path(sel["selected_blob"])
        if sha256(blob) != sel["selected_blob_sha256"]:
            raise SystemExit(f"{blob}: sha256 differs from the {size} selection")
        file = f"VOCR4_{size}_w8.vocr"
        shutil.copy2(blob, blob_dir / file)
        runs[f"VOCR4_{size}"] = sel["selected"]
        arms.append({"name": f"VOCR4_{size}", "file": file, "finder": "grad", "sha256": sel["selected_blob_sha256"],
                     "role": f"VOCR4-{size}: {sel['selected']} (arm {sel['final_arm']})"})
    known = {a["name"] for a in arms}
    if known != set(proto["locked_report"]["arms"]):
        raise SystemExit(f"arms differ from the protocol's: {sorted(known ^ set(proto['locked_report']['arms']))}")
    pairs = [{"candidate": a, "baseline": b} for a, b in proto["locked_report"]["pairs"]]
    gates = [{"candidate": a, "baseline": b} for a, b in proto["qualification"]["gates"]]
    for p in pairs + gates:
        if p["candidate"] not in known or p["baseline"] not in known:
            raise SystemExit(f"unknown arm in {p}")
    spec = {"schema": 1, "kind": "vocr3-report-spec", "arms": arms, "pairs": pairs, "gates": gates,
            "sets": ["vita-test", "psp-test"], "polarities": ["console", "native"], "primary_polarity": "console",
            "bootstrap": {"reps": 10000, "seed": 0, "min_clusters": 5}, "dialogue": True}
    return spec, runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--selection", action="append", required=True, help="SIZE=selection.json")
    ap.add_argument("--vocr3-spec", required=True)
    ap.add_argument("--vocr3-blobs", required=True)
    ap.add_argument("--release-dir", required=True)
    ap.add_argument("--blob-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    proto = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    sels = {}
    for s in args.selection:
        size, _, path = s.partition("=")
        sels[size] = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(sels) != set(SIZES):
        raise SystemExit("one selection per size is needed")
    old = json.loads(Path(args.vocr3_spec).read_text(encoding="utf-8"))
    spec, runs = build(proto, sels, old, args.vocr3_blobs, args.release_dir, args.blob_dir)
    Path(args.out).write_text(json.dumps(spec, indent=1) + "\n", encoding="utf-8")
    Path(args.out).with_suffix(".runs.json").write_text(
        json.dumps({"kind": "vocr4-report-spec-runs", "selected_runs": runs}, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({a["name"]: a["sha256"][:12] for a in spec["arms"]}))


if __name__ == "__main__":
    main()
