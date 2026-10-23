#!/usr/bin/env python3
"""The stride16 locked report's spec (eval/stride16_protocol.json "locked_report") from the committed selections.

  stride16_make_spec.py --protocol eval/stride16_protocol.json --selection Small=SEL.json --selection Normal=SEL.json \\
                        --selection Large=SEL.json --vocr5-spec eval/vocr5_report_spec_20261016.json \\
                        --blobs DIR [DIR ...] --blob-dir OUT --out spec.json

A copy of eval/vocr5_make_spec.py's design. The incumbents VOCR5_S / N / L take their files and sha256 from VOCR5's
committed report spec (each looked up in --blobs and checked). Each stride16 arm (S16, N16, N12, L16) is its best run
by development score, adopted or not (the selection's runs list, eval/stride16_select.py: the lowest score, ties at 5
decimals to the lower seed), copied into --blob-dir as <ARM>_w8.vocr after a sha256 check. Pairs: each arm against its
tier's incumbent, and N16 - N12. No gates (the protocol reports none). The run behind each arm and whether its arm was
adopted go to <out>.runs.json, because eval/vocr3_report.py rejects spec fields it does not know.
"""
import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

INCUMBENTS = {"Small": "VOCR5_S", "Normal": "VOCR5_N", "Large": "VOCR5_L"}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def find(dirs, file):
    for d in dirs:
        if (Path(d) / file).exists():
            return Path(d) / file
    raise SystemExit(f"{file}: in none of {dirs}")


def best_run(sel, arm):
    rows = [r for r in sel["runs"] if r["arm"] == arm]
    if not rows:
        raise SystemExit(f"{arm}: no scored run in the {sel['tier']} selection")
    seed = lambda r: int(re.search(r"_seed(\d+)$", r["run"]).group(1))  # noqa: E731
    return min(rows, key=lambda r: (round(r["score"], 5), seed(r)))


def build(proto, selections, vocr5_spec, blob_dirs, blob_dir):
    blob_dir = Path(blob_dir)
    blob_dir.mkdir(parents=True, exist_ok=True)
    old = {a["name"]: a for a in vocr5_spec["arms"]}
    arms, runs = [], {}
    for tier, name in INCUMBENTS.items():
        a = old[name]
        src = find(blob_dirs, a["file"])
        if sha256(src) != a["sha256"] or a["sha256"] != proto["tiers"][tier]["incumbent"]["sha256"]:
            raise SystemExit(f"{src}: sha256 differs from VOCR5's report spec or the protocol's incumbent")
        shutil.copy2(src, blob_dir / a["file"])
        arms.append({"name": name, "file": a["file"], "finder": "grad", "sha256": a["sha256"],
                     "role": f"incumbent {tier}: VOCR5's selection"})
    pairs = []
    for arm, a in proto["arms"].items():
        sel = selections[a["tier"]]
        if sel.get("kind") != "stride16-selection" or sel["tier"] != a["tier"]:
            raise SystemExit(f"{a['tier']}: not a stride16 selection of that tier")
        r = best_run(sel, arm)
        blob = Path(r["blob"])
        if sha256(blob) != r["blob_sha256"]:
            raise SystemExit(f"{blob}: sha256 differs from the {a['tier']} selection")
        file = f"{arm}_w8.vocr"
        shutil.copy2(blob, blob_dir / file)
        adopted = sel["adopted_arm"] == arm
        runs[arm] = {"run": r["run"], "adopted": adopted, "dev_score": r["score"]}
        arms.append({"name": arm, "file": file, "finder": "grad", "sha256": r["blob_sha256"],
                     "role": f"{a['tier']} stride {a['stride']}: {r['run']} "
                             f"({'adopted' if adopted else 'not adopted: report only'})"})
        pairs.append({"candidate": arm, "baseline": INCUMBENTS[a["tier"]]})
    pairs.append({"candidate": "N16", "baseline": "N12"})
    spec = {"schema": 1, "kind": "vocr3-report-spec", "arms": arms, "pairs": pairs, "gates": [],
            "sets": ["vita-test", "psp-test"], "polarities": ["console", "native"], "primary_polarity": "console",
            "bootstrap": {"reps": 10000, "seed": 0, "min_clusters": 5}, "dialogue": True}
    return spec, runs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", required=True)
    ap.add_argument("--selection", action="append", required=True, help="TIER=selection.json")
    ap.add_argument("--vocr5-spec", required=True)
    ap.add_argument("--blobs", nargs="+", required=True, help="directories holding the incumbents' files")
    ap.add_argument("--blob-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    proto = json.loads(Path(args.protocol).read_text(encoding="utf-8"))
    sels = {}
    for s in args.selection:
        tier, _, path = s.partition("=")
        sels[tier] = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(sels) != set(INCUMBENTS):
        raise SystemExit("one selection per tier is needed")
    v5 = json.loads(Path(args.vocr5_spec).read_text(encoding="utf-8"))
    spec, runs = build(proto, sels, v5, args.blobs, args.blob_dir)
    Path(args.out).write_text(json.dumps(spec, indent=1) + "\n", encoding="utf-8")
    Path(args.out).with_suffix(".runs.json").write_text(
        json.dumps({"kind": "stride16-report-spec-runs", "arms": runs}, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({a["name"]: a["sha256"][:12] for a in spec["arms"]}))


if __name__ == "__main__":
    main()
