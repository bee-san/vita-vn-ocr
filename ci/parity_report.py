#!/usr/bin/env python3
"""Check the parity report that `make -C runtime check` writes to $VOCR_REPORT.

  parity_report.py REPORT --ckpt VOCR_CKPT --shard VOCR_SHARD

The report is one JSON object. Required keys: "ckpt" and "shard" (the VOCR_CKPT and
VOCR_SHARD the target was given, which shows the test ran on CI's fixture), "lines" (crops
compared, at least 1) and "pass" (true). Any other keys, such as per-mode logit differences
or string matches, are free. The whole report is printed and added to the GitHub job summary.
The contract is at the top of ci/runtime.sh.
"""
import argparse
import json
import os
import sys
from pathlib import Path


def problems(report, ckpt, shard):
    """Contract violations in a loaded report; an empty list means it passes."""
    if not isinstance(report, dict):
        return ["the report is not a JSON object"]
    out = []
    for key, want in (("ckpt", ckpt), ("shard", shard)):
        got = report.get(key)
        if not isinstance(got, str) or os.path.realpath(got) != os.path.realpath(want):
            out.append(f'"{key}" is {got!r}, expected the VOCR_{key.upper()} that CI passed ({want})')
    lines = report.get("lines")
    if not isinstance(lines, int) or isinstance(lines, bool) or lines < 1:
        out.append(f'"lines" is {lines!r}, expected the number of crops compared (at least 1)')
    if report.get("pass") is not True:
        out.append(f'"pass" is {report.get("pass")!r}, expected true')
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("report", type=Path)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--shard", required=True)
    args = ap.parse_args()
    try:
        report = json.loads(args.report.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        sys.exit(f"error: no readable parity report at VOCR_REPORT={args.report} ({e}); "
                 "the check target must write one (contract at the top of ci/runtime.sh)")
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a", encoding="utf-8") as f:
            f.write(f"### Runtime parity report\n\n```json\n{text}\n```\n\n")
    bad = problems(report, args.ckpt, args.shard)
    if bad:
        sys.exit("error: parity report does not meet the contract:\n  " + "\n  ".join(bad))
    print("PARITY_REPORT_OK")


if __name__ == "__main__":
    main()
