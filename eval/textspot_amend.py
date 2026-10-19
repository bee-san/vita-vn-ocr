#!/usr/bin/env python3
"""Append a dated amendment to eval/textspot_protocol.json (issue #104, docs/textspot.md). Nothing above the
"amendments" list changes; an amendment can re-pin sources (their sha256 as of the amendment), pin files by sha256
(manifests, records) and state what it changes, before the scores it could affect.

  textspot_amend.py --protocol eval/textspot_protocol.json --id 1 --before "every family B run" \
      --what "..." [--what "..."] [--pin-source PATH ...] [--pin NAME=PATH ...] [--set KEY=JSON ...]

--pin-source PATH records PATH's sha256 in the amendment's sources_sha256 (tests/test_textspot.py then checks it);
--pin NAME=PATH records a file's sha256 and size under "pins"; --set KEY=JSON stores a value under "settings".
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", default=str(ROOT / "eval" / "textspot_protocol.json"))
    ap.add_argument("--id", type=int, required=True)
    ap.add_argument("--before", required=True, help="the scores or runs this amendment precedes")
    ap.add_argument("--what", action="append", required=True)
    ap.add_argument("--pin-source", action="append", default=[])
    ap.add_argument("--pin", action="append", default=[])
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    path = Path(a.protocol)
    p = json.loads(path.read_text(encoding="utf-8"))
    ids = [x["id"] for x in p["amendments"]]
    if a.id in ids or (ids and a.id != max(ids) + 1) or (not ids and a.id != 1):
        raise SystemExit(f"amendment ids so far {ids}: the next is {max(ids, default=0) + 1}")
    entry = {"id": a.id,
             "utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
             "before": a.before, "what": a.what}
    if a.pin_source:
        entry["sources_sha256"] = {s: sha256(ROOT / s) for s in a.pin_source}
    if a.pin:
        pins = {}
        for spec in a.pin:
            name, _, f = spec.partition("=")
            pins[name] = {"path": f, "sha256": sha256(f), "bytes": Path(f).stat().st_size}
        entry["pins"] = pins
    if a.set:
        entry["settings"] = {k: json.loads(v) for k, _, v in (s.partition("=") for s in a.set)}
    p["amendments"].append(entry)
    path.write_text(json.dumps(p, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(entry, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
