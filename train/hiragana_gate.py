#!/usr/bin/env python3
"""Fail closed unless a held-out W8I candidate beats targeted training within FL10 memory."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime"))
from pyvocr import Vocr  # noqa: E402


def check(baseline, targeted, candidate, blob):
    failures = []
    names = {"vita-test", "psp-test", "clannad-test"}
    for result in (baseline, targeted, candidate):
        if result["model_info"].get("backend") != "vocr" or result["model_info"].get("mode") != "w8i":
            raise ValueError("acceptance requires measurements from the C W8I backend (vocr:)")
        if result.get("finder") is not None or result.get("polarity") != "console":
            raise ValueError("acceptance requires recognizer-only reports with console polarity")
        if result.get("pad") != baseline.get("pad"):
            raise ValueError("acceptance crop padding differs")
        if not names <= result["sets"].keys():
            raise ValueError("all three held-out real screenshot sets are required")
        if any(result["datasets"][n]["split"] != "test" for n in names):
            raise ValueError("acceptance requires held-out test sessions")
    for n in names:
        if any(r["datasets"][n] != baseline["datasets"][n] for r in (targeted, candidate)):
            raise ValueError(f"{n}: dataset identities differ")
        a, b, c = (r["sets"][n] for r in (baseline, targeted, candidate))
        identities = [[(row["id"], row["text"]) for row in scored["lines"]] for scored in (a, b, c)]
        if not identities[0] or any(rows != identities[0] for rows in identities[1:]):
            raise ValueError(f"{n}: scored line identities differ")
        if not a["all"]["chars"] == b["all"]["chars"] == c["all"]["chars"]:
            raise ValueError(f"{n}: reference character counts differ")
        if c["all"]["errors"] > min(a["all"]["errors"], b["all"]["errors"]):
            failures.append(f"{n}: total CER regression")
    h = candidate["sets"]["clannad-test"]["hiragana"]
    controls = [r["sets"]["clannad-test"]["hiragana"] for r in (baseline, targeted)]
    if not h["chars"] or any(h["chars"] != c["chars"] for c in controls):
        raise ValueError("held-out CLANNAD needs identical hiragana references")
    if h["errors"] >= min(c["errors"] for c in controls):
        failures.append("CLANNAD hiragana does not improve over baseline and targeted control")
    runtime = Vocr(blob, mode="w8i", max_width=1024)
    if candidate["model_info"]["sha256"] != hashlib.sha256(Path(blob).read_bytes()).hexdigest():
        raise ValueError("candidate results belong to a different model blob")
    if runtime.blob_bytes != 657664 or runtime.arena_bytes != 124832:
        failures.append("FL10 model/arena memory budget changed")
    return {"accepted": not failures, "failures": failures,
            "blob_bytes": runtime.blob_bytes, "arena_bytes": runtime.arena_bytes,
            "physical_console_validation": "pending"}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "targeted", "candidate", "blob", "out"):
        ap.add_argument(f"--{name}", type=Path, required=True)
    args = ap.parse_args()
    result = check(*(json.loads(p.read_text()) for p in (args.baseline, args.targeted, args.candidate)), args.blob)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["accepted"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
