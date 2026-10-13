#!/usr/bin/env python3
"""The VOCR3 locked report's spec (eval/vocr3_protocol.json "locked_report") from the committed selections.

  vocr3_make_spec.py --protocol eval/vocr3_protocol.json --selection FL10=SEL.json --selection H15=SEL.json \
                     --selection W20=SEL.json --blob-dir DIR --release-dir DIR --out spec.json

Each selection is an eval/vocr3_select.py output (with its variant decision applied: the selected run and its blob
sha256). The selected blobs are copied into --blob-dir as VOCR3_<SIZE>_w8.vocr and checked against the
selection's sha256; the released baselines are copied from --release-dir and checked against the protocol's
hashes (F20's against the hash given with --f20-sha256). The arms, pairs and gates are the protocol's
(amendment 1 included), with the protocol's names written with underscores (VOCR3-FL10 -> VOCR3_FL10) for
eval/vocr3_report.py; the selected run behind each VOCR3 arm goes to <out>.runs.json (the harness rejects
spec fields it does not know).
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

F20_SHA256 = "f8c1abe4743a71a742e718523f4206f992d3bd4fb69de321997b2bd6f16779c0"   # v0.2.0 F20_w8.vocr


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def arm_name(name):
    return name.replace("-", "_")


def build(proto, selections, blob_dir, release_dir, f20_sha=F20_SHA256):
    blob_dir = Path(blob_dir)
    blob_dir.mkdir(parents=True, exist_ok=True)
    released = {"VOCR1": ("FL10_w8.vocr", proto["sizes"]["FL10"]["baselines"]["VOCR1"]["sha256"], "grad"),
                "VOCR2": ("VOCR2_FL10_w8.vocr", proto["sizes"]["FL10"]["baselines"]["VOCR2"]["sha256"], "textbox"),
                "VOCR_H15": ("H15_w8.vocr", proto["sizes"]["H15"]["baselines"]["VOCR_H15"]["sha256"], "grad"),
                "F20": ("F20_w8.vocr", f20_sha, "grad")}
    arms, roles = [], {}
    for name, (file, sha, finder) in released.items():
        src = Path(release_dir) / file
        if sha256(src) != sha:
            raise SystemExit(f"{src}: sha256 differs from {sha}")
        shutil.copy2(src, blob_dir / file)
        arms.append({"name": name, "file": file, "finder": finder, "sha256": sha, "role": "released baseline"})
    for size, sel in selections.items():
        blob = Path(sel["selected_blob"])
        if sha256(blob) != sel["selected_blob_sha256"]:
            raise SystemExit(f"{blob}: sha256 differs from the {size} selection")
        file = f"VOCR3_{size}_w8.vocr"
        shutil.copy2(blob, blob_dir / file)
        roles[size] = sel["selected"]
        arms.append({"name": f"VOCR3_{size}", "file": file, "finder": "grad", "sha256": sel["selected_blob_sha256"],
                     "role": f"VOCR3-{size}: {sel['selected']}"})
        if size == "FL10":
            arms.append({"name": "VOCR3_FL10_C3", "file": file, "finder": "textbox",
                         "sha256": sel["selected_blob_sha256"],
                         "role": "VOCR3-FL10 with VOCR2's C3 finder, reported only"})
    lr, q = proto["locked_report"], proto["qualification"]
    known = {a["name"] for a in arms}
    pairs = [{"candidate": arm_name(a), "baseline": arm_name(b)} for a, b in lr["pairs"]]
    gates = [{"candidate": arm_name(a), "baseline": arm_name(b)} for a, b in q["gates"]]
    for p in pairs + gates:
        if p["candidate"] not in known or p["baseline"] not in known:
            raise SystemExit(f"unknown arm in {p}")
    return {"schema": 1, "kind": "vocr3-report-spec", "arms": arms, "pairs": pairs, "gates": gates,
            "sets": ["vita-test", "psp-test"], "polarities": ["console", "native"], "primary_polarity": "console",
            "bootstrap": {"reps": 10000, "seed": 0, "min_clusters": 5}, "dialogue": True}, roles


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--selection", action="append", required=True, help="SIZE=selection.json")
    ap.add_argument("--blob-dir", required=True)
    ap.add_argument("--release-dir", required=True)
    ap.add_argument("--f20-sha256", default=F20_SHA256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    proto = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    sels = {}
    for s in args.selection:
        size, _, path = s.partition("=")
        sels[size] = json.loads(Path(path).read_text(encoding="utf-8"))
    spec, roles = build(proto, sels, args.blob_dir, args.release_dir, args.f20_sha256)
    Path(args.out).write_text(json.dumps(spec, indent=1) + "\n", encoding="utf-8")
    # which run each VOCR3 arm is: a sidecar, because eval/vocr3_report.py rejects fields it does not know
    runs = {"kind": "vocr3-report-spec-runs", "selected_runs": roles}
    Path(args.out).with_suffix(".runs.json").write_text(json.dumps(runs, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({a["name"]: a["sha256"][:12] for a in spec["arms"]}))


if __name__ == "__main__":
    main()
