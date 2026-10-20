#!/usr/bin/env python3
"""Check that two checkouts of train/textspot2_train.py train the same run on the same data: the logged losses and
learning rates of every step, the QAT checkpoint (folded weights, biases, activation scales) and the int8 network
(final.vtd, all but its CRC and checkpoint-hash fields). Used to show that adding the d2s path changed no 4-map
recipe (the conductor, 2026-10-20T10:00Z).

  identity_check.py --old OLD_REPO --new NEW_REPO --out DIR --name NAME -- TRAINER_ARGS...

TRAINER_ARGS are train/textspot2_train.py's arguments without --out; each checkout runs them with its own
interpreter (REPO/.venv/bin/python, else REPO/.venv-ci/bin/python) into DIR/old/NAME and DIR/new/NAME. Prints
IDENTICAL NAME ... or exits non-zero with what differs.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch


def python_of(repo):
    for v in (".venv", ".venv-ci"):
        p = Path(repo) / v / "bin" / "python"
        if p.exists():
            return str(p)
    return sys.executable


def run(repo, args, out):
    p = subprocess.run([python_of(repo), "train/textspot2_train.py", *args, "--out", str(out)], cwd=repo,
                       capture_output=True, text=True, timeout=7200, stdin=subprocess.DEVNULL)
    if p.returncode:
        raise SystemExit(f"{repo}: rc {p.returncode}\n{p.stdout[-1500:]}\n{p.stderr[-3000:]}")


def log_of(run_dir):
    out = []
    for line in (Path(run_dir) / "log.jsonl").read_text().splitlines():
        r = json.loads(line)
        r.pop("sec", None)
        out.append(r)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--old", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("args", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    targs = a.args[1:] if a.args[:1] == ["--"] else a.args
    dirs = {k: Path(a.out) / k / a.name for k in ("old", "new")}
    for k, repo in (("old", a.old), ("new", a.new)):
        run(repo, targs, dirs[k])
    bad = []
    lo, ln = log_of(dirs["old"]), log_of(dirs["new"])
    if lo != ln:
        diff = next(i for i, (x, y) in enumerate(zip(lo + [None], ln + [None])) if x != y)
        bad.append(f"log.jsonl differs from record {diff}: {lo[diff] if diff < len(lo) else None} / "
                   f"{ln[diff] if diff < len(ln) else None}")
    co = torch.load(dirs["old"] / "final.pt", map_location="cpu", weights_only=False)
    cn = torch.load(dirs["new"] / "final.pt", map_location="cpu", weights_only=False)
    if co["act_scale"] != cn["act_scale"]:
        bad.append("act_scale differs")
    if len(co["w"]) != len(cn["w"]) or not all(torch.equal(x, y) for x, y in zip(co["w"] + co["b"], cn["w"] + cn["b"])):
        bad.append("the checkpoint's weights or biases differ")
    vo, vn = (dirs["old"] / "final.vtd").read_bytes(), (dirs["new"] / "final.vtd").read_bytes()
    if (vo[:12], vo[16:24], vo[32:]) != (vn[:12], vn[16:24], vn[32:]):
        bad.append("final.vtd differs (outside its CRC and checkpoint-hash fields)")
    if bad:
        for b in bad:
            print(f"DIFFERS {a.name}: {b}")
        raise SystemExit(1)
    steps = sum(1 for r in ln if "lr" in r)
    print(f"IDENTICAL {a.name}: {len(ln)} log records ({steps} step records), {len(cn['w'])} layers, "
          f"final.vtd {len(vn)} B")


if __name__ == "__main__":
    main()
