#!/usr/bin/env python3
"""Check one finished textspot2 run (train/textspot2_train.py, or train/textspot_train.py for r0) and record it.

  check_run.py RUN [--finished FILE]

Checks: run.json names the directory (<recipe>_h2_<shape>_seed<n>; an r0 run is textspot's trainer, whose run.json
says h2_<shape>_seed<n>); done.json's sha256 and bytes are the blobs'; final.vtd loads in synth/textdet_twin.Model
and final.vtsp in synth/textspot_twin.Model with half_above 600 and conf_min 1, its network is the VTDT blob and
named the run's shape; the twin (textspot_twin.spot) runs on two pinned procedural frames (960 x 544 and 480 x 272)
and returns boxes inside them; done.json's steps are run.json's; a run that read a textspot or textspot2 data pin
recorded its skip lists; run.json's code sha256 equal the files in this checkout and its manifests' sha256 the
manifests here (a difference is printed, not fatal: the blob checks are what make a run usable). With --finished it
appends "name vtsp_sha256 vtsp_bytes steps wall_seconds" to FILE unless that line is already there. Prints CHECK_OK
or exits non-zero.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "synth"))
import textdet_twin as TD  # noqa: E402
import textspot_twin as TS  # noqa: E402

HALF_ABOVE = 600


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def procedural_frame(seed, H, W):
    """Grey blocks and stripes (text-like rows of strokes) with noise; pinned by the seed (textspot's check)."""
    rng = np.random.default_rng(seed)
    g = np.full((H, W), int(rng.integers(0, 256)), np.int32)
    for _ in range(int(rng.integers(10, 30))):
        y0, x0 = int(rng.integers(0, H)), int(rng.integers(0, W))
        sub = g[y0:y0 + int(rng.integers(4, 40)), x0:x0 + int(rng.integers(8, 300))]
        sub[:] = int(rng.integers(0, 256))
        sub[:, ::int(rng.integers(2, 6))] = int(rng.integers(0, 256))
    return np.clip(g + rng.integers(-12, 13, g.shape), 0, 255).astype(np.uint8)


def expected_name(info):
    if info.get("kind") == "textspot-run":                 # r0: textspot's trainer names runs <scale>_<shape>_seed<n>
        return f"r0_{info['name']}"
    return info["name"]


def check(run_dir):
    run_dir = Path(run_dir)
    info = json.loads((run_dir / "run.json").read_text())
    done = json.loads((run_dir / "done.json").read_text())
    vtd, vtsp = (run_dir / "final.vtd").read_bytes(), (run_dir / "final.vtsp").read_bytes()
    bad = []
    if info.get("kind") not in ("textspot2-run", "textspot-run"):
        bad.append(f"run.json kind {info.get('kind')!r}")
    if expected_name(info) != run_dir.name.split(".failed")[0]:
        bad.append(f"run.json names {expected_name(info)}, the directory is {run_dir.name}")
    if info.get("half_above") != HALF_ABOVE or info.get("scale") != "h2":
        bad.append(f"run.json: scale {info.get('scale')}, half_above {info.get('half_above')} (h2, 600)")
    for key, data in (("vtd", vtd), ("vtsp", vtsp)):
        if done[f"{key}_sha256"] != hashlib.sha256(data).hexdigest() or done[f"{key}_bytes"] != len(data):
            bad.append(f"done.json's {key} sha256 / bytes differ from final.{key}")
    if done["pt_sha256"] != sha(run_dir / "final.pt"):
        bad.append("done.json's pt sha256 differs from final.pt")
    net = TD.Model(vtd)
    m = TS.Model(vtsp)
    shape = info["shape"]["name"]
    if (m.half_above, m.conf_min, m.name, net.name) != (HALF_ABOVE, 1, shape, shape):
        bad.append(f"final.vtsp: half_above {m.half_above}, conf_min {m.conf_min}, name {m.name}")
    if vtsp[32:] != vtd or TS.pack(vtd, HALF_ABOVE, 1) != vtsp:
        bad.append("final.vtsp is not pack(final.vtd, 600, 1)")
    if done["steps"] != info["steps"]:
        bad.append(f"done.json steps {done['steps']} != run.json steps {info['steps']}")
    sk = info.get("skip") or {}
    pins = [m_ for m_ in info["manifests_sha256"] if Path(m_).name.startswith(("textspot_shards", "textspot2_shards"))]
    if pins and not sk.get("lists"):
        bad.append("the run read a textspot data pin but recorded no skip lists")
    boxes = {}
    for seed, (H, W) in ((1, (544, 960)), (2, (272, 480))):
        r = TS.spot(m, procedural_frame(seed, H, W))
        if not all(0 <= b.x and 0 <= b.y and b.x + b.w <= W and b.y + b.h <= H for b in r.boxes):
            bad.append(f"a box outside the {W}x{H} frame")
        boxes[f"{W}x{H}"] = len(r.boxes)
    notes = []
    for f, want_sha in info["code_sha256"].items():
        if not (ROOT / f).exists() or sha(ROOT / f) != want_sha:
            notes.append(f"code {f}: the run's sha256 is not this checkout's")
    for f, want_sha in info["manifests_sha256"].items():
        p = Path(f) if Path(f).is_absolute() else ROOT / f
        if not p.exists() or sha(p) != want_sha:
            notes.append(f"manifest {f}: the run's sha256 is not this checkout's")
    return info, done, bad, notes, boxes


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--finished")
    args = ap.parse_args()
    info, done, bad, notes, boxes = check(args.run)
    name = Path(args.run).name
    for n in notes:
        print(f"{name}: NOTE {n}")
    if bad:
        for b in bad:
            print(f"{name}: BAD {b}")
        raise SystemExit(1)
    line = f"{name} {done['vtsp_sha256']} {done['vtsp_bytes']} {done['steps']} {done['wall_seconds']}"
    if args.finished:
        f = Path(args.finished)
        old = f.read_text().splitlines() if f.exists() else []
        if line not in old:
            with open(f, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")
    sk = info.get("skip") or {}
    print(f"CHECK_OK {line} boxes {boxes} skip-dropped {sk.get('dropped_total')} "
          f"({info['started']} -> {done['finished']})")


if __name__ == "__main__":
    main()
