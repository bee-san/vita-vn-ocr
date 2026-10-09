#!/usr/bin/env python3
"""Write data/manifests/synth2_bg_exclude.txt: screenshots render.py and calibrate.py must not use.

  - every VN id in eval's test_games.json "all": the reserved test games, every is_test_vn()
    game with screenshots, VNs sharing a release with a set screenshot, and the set games'
    VNDB relations
  - every local screenshot whose VNDB release id is in "sessions" (test capture sessions)
  - the screenshots of the research's hand-transcribed real lines (the provisional A/B set),
    matched to data/vndb/shots by sha256

is_test_vn() games are excluded in code as well. Eval's list only grows: re-run this before
every render.

  bg_exclude.py [--eval-games PATH]
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
EVAL_GAMES = HERD / "shared" / "artifacts" / "eval" / "test_games.json"
RESEARCH = HERD / "shared" / "research" / "scratch" / "real"
OUT = HERE.parent / "data" / "manifests" / "synth2_bg_exclude.txt"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def research_shots():
    """data/vndb/shots names of the screenshots behind the research's hand-transcribed lines."""
    by_sha = {sha(p): p.name for p in (DATA / "vndb" / "shots").glob("*.jpg")}
    files = [next((RESEARCH / "shots").glob(f"{it['shot']}.*"))
             for it in json.loads((RESEARCH / "gt_real.json").read_text(encoding="utf-8"))]
    files += [RESEARCH / "q5_run_a" / "raw" / Path(it["file"]).name
              for it in json.loads((RESEARCH / "q5_run_a" / "gt.json").read_text(encoding="utf-8"))]
    names = {by_sha.get(sha(f)) for f in files}
    if None in names:
        raise SystemExit("a research screenshot is not in data/vndb/shots")
    return sorted(names)


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
    ap.add_argument("--eval-games", type=Path, default=EVAL_GAMES)
    args = ap.parse_args()
    games = json.loads(args.eval_games.read_text())
    vns = sorted(set(games["all"]), key=lambda v: int(v[1:]))
    sessions = set(games.get("sessions", []))
    lines = ["# synth2 background exclusions (written by synth/bg_exclude.py): a VNDB VN id (every",
             "# screenshot of that VN) or a screenshot file name. Source: eval's test_games.json",
             f"# ({len(vns)} reserved VN ids, {len(sessions)} test capture sessions) and the research's",
             "# hand-transcribed real sets."]
    lines += [f"{v}  eval-test" for v in vns]
    lines += [f"{n}  eval-session" for n in session_shots(sessions)]
    lines += [f"{n}  research-real" for n in research_shots()]
    OUT.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT}: {len(lines) - 4} entries", flush=True)


if __name__ == "__main__":
    main()
