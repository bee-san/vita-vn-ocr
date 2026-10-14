#!/usr/bin/env python3
"""Write data/manifests/synth2_bg_exclude.txt: screenshots render.py and calibrate.py must not use.

  - every VN of the reservation (eval/manifests/reserved.json, docs/reservations.md): the test
    sets' VNs (vita-test, psp-test and their release partners: test_games.json "all"; the CLANNAD
    family) and the development sample, tagged eval-test and dev-sample
  - every local screenshot whose VNDB release id is in "sessions" (test capture sessions)

The reserved VNs are excluded in code as well (synth/common.is_reserved_vn, load_exclusions reads
the committed test_games.json and reserved.json), so a stale manifest cannot let a reserved game
through; tests/test_synth2.py fails when the manifest no longer covers that list. Until 2026-10-14
the list also held every is_test_vn() game with screenshots, the set games' VNDB relations and the
research's 28 hand-transcribed screenshots of synth2's provisional A/B; those are training-side now.

  bg_exclude.py [--eval-games PATH ...]
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DATA  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
EVAL_GAMES_COMMITTED = HERE.parent / "eval" / "manifests" / "test_games.json"
# eval's working copy ($HERD/shared/artifacts/eval/test_games.json) still holds the pre-2026-10-14 list; only the
# committed file counts now.
EVAL_GAMES = [EVAL_GAMES_COMMITTED]
RESERVED = HERE.parent / "eval" / "manifests" / "reserved.json"
OUT = HERE.parent / "data" / "manifests" / "synth2_bg_exclude.txt"


def load_exclusions(path=OUT):
    """VN ids and screenshot names that must never be a background or a calibration crop:
    the exclusion manifest's entries plus every VN in the committed eval test_games.json."""
    out = set()
    if path.exists():
        out = {x.split()[0] for x in path.read_text().splitlines() if x.strip() and x[0] != "#"}
    if EVAL_GAMES_COMMITTED.exists():
        out |= set(json.loads(EVAL_GAMES_COMMITTED.read_text(encoding="utf-8"))["all"])
    if RESERVED.exists():
        r = json.loads(RESERVED.read_text(encoding="utf-8"))
        out |= set(r["test"]) | set(r["dev_sample"])
    return out


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def session_shots(sessions):
    """Local screenshot names (Vita and PSP) whose VNDB release id is a test capture session."""
    out = []
    for s in (DATA / "vndb" / "screenshots.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(s)
        if r["release"] in sessions:
            name = f"{r['vn']}_{r['url'].rsplit('/', 1)[-1]}"
            if (DATA / "vndb" / "shots" / name).exists() or (DATA / "vndb_psp" / "shots" / name).exists():
                out.append(name)
    return sorted(set(out))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-games", type=Path, nargs="+", default=EVAL_GAMES,
                    help="test_games.json files to merge (missing ones are skipped)")
    args = ap.parse_args()
    vns, sessions, used = set(), set(), []
    for path in args.eval_games:
        if path.exists():
            games = json.loads(path.read_text(encoding="utf-8"))
            vns |= set(games["all"])
            sessions |= set(games.get("sessions", []))
            used.append(str(path))
    if not used:
        raise SystemExit("no eval test_games.json found")
    res = json.loads(RESERVED.read_text(encoding="utf-8"))
    vns |= set(res["test"])
    dev = sorted(set(res["dev_sample"]) - vns, key=lambda v: int(v[1:]))
    vns = sorted(vns, key=lambda v: int(v[1:]))
    lines = ["# synth2 background exclusions (written by synth/bg_exclude.py): a VNDB VN id (every",
             "# screenshot of that VN) or a screenshot file name. Source: eval/manifests/reserved.json",
             f"# ({len(vns)} test VN ids, {len(dev)} development-sample VN ids; docs/reservations.md)",
             f"# and test_games.json ({len(sessions)} test capture sessions)."]
    lines += [f"{v}  eval-test" for v in vns]
    lines += [f"{v}  dev-sample" for v in dev]
    lines += [f"{n}  eval-session" for n in session_shots(sessions)]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}: {len(lines) - 4} entries from {', '.join(used)}", flush=True)


if __name__ == "__main__":
    main()
