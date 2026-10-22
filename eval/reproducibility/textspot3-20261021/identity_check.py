#!/usr/bin/env python3
"""Check that two checkouts of train/textspot3_train.py train the same run on the same data: the logged losses and
learning rates of every step (every record but its time), the QAT checkpoint (folded weights, biases, activation
scales, shape, kind) and the int8 network (T: final.vtd, all but its CRC and checkpoint-hash fields; M: final.vtm,
likewise) and final.vtsp's network part. Used to show that a change of the trainer changed no started path (the
textspot3-train prompt: "If the trainer file must change after any run has started, prove with an identity check").

  identity_check.py --old OLD_REPO --new NEW_REPO --out DIR --name NAME -- TRAINER_ARGS...

TRAINER_ARGS are train/textspot3_train.py's arguments without --out; each checkout runs them with its own
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
    p = subprocess.run([python_of(repo), "train/textspot3_train.py", *args, "--out", str(out)], cwd=repo,
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


def net_file(run_dir):
    for nf in ("vtd", "vtm"):
        if (Path(run_dir) / f"final.{nf}").exists():
            return nf
    raise SystemExit(f"{run_dir}: neither final.vtd nor final.vtm")


def compare(old, new):
    bad = []
    lo, ln = log_of(old), log_of(new)
    if lo != ln:
        diff = next(i for i, (x, y) in enumerate(zip(lo + [None], ln + [None])) if x != y)
        bad.append(f"log.jsonl differs from record {diff}: {lo[diff] if diff < len(lo) else None} / "
                   f"{ln[diff] if diff < len(ln) else None}")
    co = torch.load(old / "final.pt", map_location="cpu", weights_only=False)
    cn = torch.load(new / "final.pt", map_location="cpu", weights_only=False)
    for k in ("kind", "shape", "seed", "steps", "act_scale"):
        if co.get(k) != cn.get(k):
            bad.append(f"the checkpoint's {k} differs")
    if len(co["w"]) != len(cn["w"]) or not all(torch.equal(x, y) for x, y in zip(co["w"] + co["b"], cn["w"] + cn["b"])):
        bad.append("the checkpoint's weights or biases differ")
    nf = net_file(old)
    if net_file(new) != nf:
        bad.append("the network files differ in kind")
    vo, vn = (old / f"final.{nf}").read_bytes(), (new / f"final.{nf}").read_bytes()
    if (vo[:12], vo[16:24], vo[32:]) != (vn[:12], vn[16:24], vn[32:]):
        bad.append(f"final.{nf} differs (outside its CRC and checkpoint-hash fields)")
    so, sn = (old / "final.vtsp").read_bytes(), (new / "final.vtsp").read_bytes()
    if (so[:4], so[16:32]) != (sn[:4], sn[16:32]) or so[32:] != vo or sn[32:] != vn:
        bad.append("final.vtsp's header or its network differ")
    steps = sum(1 for r in ln if "lr" in r)
    return bad, f"{len(ln)} log records ({steps} step records), {len(cn['w'])} layers, final.{nf} {len(vn)} B"


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
    bad, what = compare(dirs["old"], dirs["new"])
    if bad:
        for b in bad:
            print(f"DIFFERS {a.name}: {b}")
        raise SystemExit(1)
    print(f"IDENTICAL {a.name}: {what}")


if __name__ == "__main__":
    main()
