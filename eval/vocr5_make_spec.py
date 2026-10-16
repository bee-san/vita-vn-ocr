#!/usr/bin/env python3
"""The VOCR5 locked report's spec (eval/vocr5_protocol.json "locked_report") from the committed selections.

  vocr5_make_spec.py --protocol eval/vocr5_protocol.json --selection Small=SEL.json --selection Normal=SEL.json \
                     --selection Large=SEL.json --vocr4-spec eval/vocr4_report_spec_20261014.json \
                     --blobs DIR [DIR ...] --blob-dir OUT --out spec.json

A copy of eval/vocr4_make_spec.py. The released baselines (VOCR1, VOCR2, VOCR_H15, F20), VOCR3's three and VOCR4's
three take their files, finders and sha256 from VOCR4's committed report spec; each file is looked up in --blobs
(the first directory holding it) and checked against that sha256. The VOCR5 arms are the selections' blobs
(eval/vocr5_select.py select), checked against the selections' sha256 and copied into --blob-dir as
VOCR5_<S|N|L>_w8.vocr. Arms, pairs and gates are the protocol's. The run behind each VOCR5 arm goes to
<out>.runs.json, because eval/vocr3_report.py rejects spec fields it does not know.
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

TIERS = {"Small": "S", "Normal": "N", "Large": "L"}
RELEASED = ("VOCR1", "VOCR2", "VOCR_H15", "F20")
OLD3 = ("VOCR3_FL10", "VOCR3_H15", "VOCR3_W20")
OLD4 = ("VOCR4_FL10", "VOCR4_H15", "VOCR4_W20")
GATES = (("VOCR5_S", "VOCR1"), ("VOCR5_N", "VOCR_H15"), ("VOCR5_L", "VOCR_H15"))   # protocol locked_report.gates


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find(dirs, file):
    for d in dirs:
        if (Path(d) / file).exists():
            return Path(d) / file
    raise SystemExit(f"{file}: in none of {dirs}")


def build(proto, selections, vocr4_spec, blob_dirs, blob_dir):
    blob_dir = Path(blob_dir)
    blob_dir.mkdir(parents=True, exist_ok=True)
    old = {a["name"]: a for a in vocr4_spec["arms"]}
    arms, runs = [], {}
    for name in RELEASED + OLD3 + OLD4:
        a = old[name]
        src = find(blob_dirs, a["file"])
        if sha256(src) != a["sha256"]:
            raise SystemExit(f"{src}: sha256 differs from the committed report spec ({a['sha256'][:12]})")
        shutil.copy2(src, blob_dir / a["file"])
        arms.append({"name": name, "file": a["file"], "finder": a["finder"], "sha256": a["sha256"], "role": a["role"]})
    for tier, t in TIERS.items():
        sel = selections[tier]
        if sel.get("kind") != "vocr5-selection" or sel["tier"] != tier:
            raise SystemExit(f"{tier}: not a VOCR5 selection of that tier")
        blob = Path(os.path.expandvars(sel["selected_blob"])).expanduser()   # a kept incumbent: $HERD/...
        if sha256(blob) != sel["selected_blob_sha256"]:
            raise SystemExit(f"{blob}: sha256 differs from the {tier} selection")
        file = f"VOCR5_{t}_w8.vocr"
        shutil.copy2(blob, blob_dir / file)
        runs[f"VOCR5_{t}"] = sel["selected"]
        how = f"arm {sel['adopted_arm']}" if sel["replaces"] else "the incumbent stays"
        arms.append({"name": f"VOCR5_{t}", "file": file, "finder": "grad", "sha256": sel["selected_blob_sha256"],
                     "role": f"VOCR5 {tier}: {sel['selected']} ({how})"})
    known = {a["name"] for a in arms}
    if known != set(proto["locked_report"]["arms"]):
        raise SystemExit(f"arms differ from the protocol's: {sorted(known ^ set(proto['locked_report']['arms']))}")
    pairs = [{"candidate": a, "baseline": b} for a, b in proto["locked_report"]["pairs"]]
    gates = [{"candidate": a, "baseline": b} for a, b in GATES]
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
    ap.add_argument("--selection", action="append", required=True, help="TIER=selection.json")
    ap.add_argument("--vocr4-spec", required=True)
    ap.add_argument("--blobs", nargs="+", required=True, help="directories holding the earlier arms' files")
    ap.add_argument("--blob-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    proto = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    sels = {}
    for s in args.selection:
        tier, _, path = s.partition("=")
        sels[tier] = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(sels) != set(TIERS):
        raise SystemExit("one selection per tier is needed")
    v4 = json.loads(Path(args.vocr4_spec).read_text(encoding="utf-8"))
    spec, runs = build(proto, sels, v4, args.blobs, args.blob_dir)
    Path(args.out).write_text(json.dumps(spec, indent=1) + "\n", encoding="utf-8")
    Path(args.out).with_suffix(".runs.json").write_text(
        json.dumps({"kind": "vocr5-report-spec-runs", "selected_runs": runs}, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({a["name"]: a["sha256"][:12] for a in spec["arms"]}))


if __name__ == "__main__":
    main()
