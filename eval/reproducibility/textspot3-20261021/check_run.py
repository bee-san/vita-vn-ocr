#!/usr/bin/env python3
"""Check one finished textspot3 run (train/textspot3_train.py) and record it.

  check_run.py RUN [--finished FILE] [--cbin2 runtime/vocr_textspot2_test] [--cbin3 runtime/vocr_textspot3_test]
               [--plan PLAN.jsonl]

Checks: run.json is a textspot3 run that names the directory (<recipe>_<shape>_seed<n>, a moved failed attempt's
.failed<k> suffix aside) with its family's scale (T: half_above 480; M: wider than 480 or taller than 272);
done.json's sha256 and bytes are the blobs'; done.json's steps are run.json's; a run that read a textspot or textspot2
data pin recorded its skip lists.
- Family T: final.vtd loads in synth/textspot2_twin.Net (stride 8, 16 outputs) and final.vtsp in
  synth/textspot2_twin.Model as VTSP version 2 with half_above 480 and conf_min 1, named the run's shape, equal to
  textspot2_twin.pack(final.vtd, 480, 1, version 2); the twin (textspot2_twin.spot) runs on two pinned procedural
  frames (960 x 544 and 480 x 272) and returns boxes inside them; with --cbin2, the C driver
  runtime/vocr_textspot2_test gives the twin's --batch line for each frame.
- Family M: final.vtsp loads in synth/textspot3_twin.Model as VTSP version 3 with conf_min 1, named the run's shape;
  textspot3_twin.spot runs on the same frames, boxes inside them; with --cbin3, runtime/vocr_textspot3_test gives the
  twin's line for each frame.
With --plan, the run is the plan's record of its name: the same recipe, family, shape, seed, steps and extras, and
for kd the plan's teacher (run.json's teacher final.pt sha256 is the plan's). run.json's code sha256 equal the files in
this checkout and its manifests' sha256 the manifests here (a difference is printed, not fatal: the blob checks are
what make a run usable). With --finished it appends
"name vtsp_sha256 vtsp_bytes steps wall_seconds" to FILE unless that line is already there. Prints CHECK_OK or exits
non-zero.
"""
import argparse
import hashlib
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT / "synth", ROOT / "train"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot2_twin as TS2  # noqa: E402

HALF_T = 480
SCALES = {"T": {"rule": "half_above", "half_above": 480,
                "what": "a canvas wider than 480 px is halved (textspot_twin.halve)"},
          "M": {"rule": "wider or taller", "half_above": 480, "half_above_h": 272,
                "what": "a canvas wider than 480 px or taller than 272 px is halved (textspot_twin.halve)"}}
FRAMES = ((1, (544, 960)), (2, (272, 480)))


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


def twin_line(r):
    """The C driver's --batch line of a Result (runtime/vocr_textspot2_test.c: rc count block_count pick truncated,
    then x y w h vertical conf dark block per box, then x y w h vertical first count per block)."""
    out = [0, len(r.boxes), len(r.blocks), r.pick, r.truncated]
    for b in r.boxes:
        out += [b.x, b.y, b.w, b.h, b.vertical, b.conf, b.dark, b.block]
    for b in r.blocks:
        out += [b.x, b.y, b.w, b.h, b.vertical, b.first, b.count]
    return " ".join(map(str, out))


def c_lines(cbin, vtsp_path, frames):
    """The C driver's --batch lines for grey frames (records: <4I w, h, bpp 1, flags 0, then the pixels)."""
    with tempfile.TemporaryDirectory() as d:
        f = Path(d) / "frames.bin"
        with open(f, "wb") as fh:
            for g in frames:
                fh.write(struct.pack("<4I", g.shape[1], g.shape[0], 1, 0))
                fh.write(np.ascontiguousarray(g).tobytes())
        p = subprocess.run([str(cbin), "--batch", str(vtsp_path), str(f)], capture_output=True, text=True,
                           timeout=600, stdin=subprocess.DEVNULL)
    if p.returncode:
        raise RuntimeError(f"{cbin} --batch: rc {p.returncode}: {p.stderr[-500:]}")
    return p.stdout.splitlines()


def plan_differences(info, rec):
    """Where run.json differs from its plan record (textspot3_plan.py's): [] when it is the plan's run."""
    want_ex = [x for x in ("size", "clean") if x in (rec.get("extras") or "").split(",")] \
        if rec.get("extras") is not None else {"size": ["size"], "clean": ["clean"]}.get(rec["recipe"], [])
    bad = []
    for key, have, want in (("recipe", info.get("recipe"), rec["recipe"]),
                            ("family", info.get("family"), rec["family"]),
                            ("shape", info["shape"]["name"], rec["shape"]), ("seed", info.get("seed"), rec["seed"]),
                            ("steps", info.get("steps"), rec["steps"]), ("extras", info.get("extras"), want_ex)):
        if have != want:
            bad.append(f"run.json {key} {have!r}, the plan's {want!r}")
    teacher = ((info.get("settings") or {}).get("kd") or {}).get("teacher")
    if rec.get("teacher_pt_sha256") and (teacher or {}).get("pt_sha256") != rec["teacher_pt_sha256"]:
        bad.append("run.json's teacher final.pt sha256 is not the plan's")
    if rec["recipe"] != "kd" and teacher is not None:
        bad.append("a teacher in a run the plan has without one")
    return bad


def check(run_dir, cbin2=None, cbin3=None, plan=None):
    run_dir = Path(run_dir)
    info = json.loads((run_dir / "run.json").read_text())
    done = json.loads((run_dir / "done.json").read_text())
    vtsp = (run_dir / "final.vtsp").read_bytes()
    vtd = (run_dir / "final.vtd").read_bytes() if (run_dir / "final.vtd").exists() else None
    bad = []
    fam = info.get("family")
    if info.get("kind") != "textspot3-run" or fam not in SCALES:
        bad.append(f"run.json kind {info.get('kind')!r}, family {fam!r}")
        return info, done, bad, [], {}
    if info["name"] != run_dir.name.split(".failed")[0]:
        bad.append(f"run.json names {info['name']}, the directory is {run_dir.name}")
    if info.get("scale") != SCALES[fam]:
        bad.append(f"run.json: scale {info.get('scale')} is not family {fam}'s")
    for key, data in (("vtd", vtd), ("vtsp", vtsp)):
        want = (done.get(f"{key}_sha256"), done.get(f"{key}_bytes"))
        have = (hashlib.sha256(data).hexdigest(), len(data)) if data is not None else (None, None)
        if want != have:
            bad.append(f"done.json's {key} sha256 / bytes differ from final.{key}")
    if done["pt_sha256"] != sha(run_dir / "final.pt"):
        bad.append("done.json's pt sha256 differs from final.pt")
    shape = info["shape"]["name"]
    if fam == "T":
        twin, cbin = TS2, cbin2
        if vtd is None:
            bad.append("a family T run without final.vtd")
            return info, done, bad, [], {}
        net, m = TS2.Net(vtd), TS2.Model(vtsp)
        if m.version != TS2.VERSION_D2S or (net.stride, net.outputs, m.stride) != (8, 16, 4):
            bad.append(f"final.vtsp: version {m.version}, network stride {net.stride} / outputs {net.outputs}")
        if vtsp[32:] != vtd or TS2.pack(vtd, HALF_T, 1, version=TS2.VERSION_D2S) != vtsp:
            bad.append("final.vtsp is not textspot2_twin.pack(final.vtd, 480, 1, version 2)")
        if (m.half_above, m.conf_min, m.name, net.name) != (HALF_T, 1, shape, shape):
            bad.append(f"final.vtsp: half_above {m.half_above}, conf_min {m.conf_min}, name {m.name}")
        if done.get("vtsp_version") != TS2.VERSION_D2S:
            bad.append(f"done.json: vtsp_version {done.get('vtsp_version')}")
    else:
        import textspot3_twin as TW3
        twin, cbin = TW3, cbin3
        m = TW3.Model(vtsp)
        if (m.version, m.conf_min, m.name) != (done.get("vtsp_version"), 1, shape) or m.version < 3:
            bad.append(f"final.vtsp: version {m.version}, conf_min {m.conf_min}, name {m.name}")
    if done["steps"] != info["steps"]:
        bad.append(f"done.json steps {done['steps']} != run.json steps {info['steps']}")
    if plan is not None:
        recs = {json.loads(x)["run"]: json.loads(x) for x in Path(plan).read_text().splitlines() if x.strip()}
        rec = recs.get(info["name"])
        if rec is None:
            bad.append(f"{info['name']} is not a run of {Path(plan).name}")
        else:
            bad += plan_differences(info, rec)
    sk = info.get("skip") or {}
    pins = [m_ for m_ in info["manifests_sha256"] if Path(m_).name.startswith(("textspot_shards", "textspot2_shards"))]
    if pins and not sk.get("lists"):
        bad.append("the run read a textspot data pin but recorded no skip lists")
    boxes, frames, lines = {}, [], []
    for seed, (H, W) in FRAMES:
        g = procedural_frame(seed, H, W)
        r = twin.spot(m, g)
        if not all(0 <= b.x and 0 <= b.y and b.x + b.w <= W and b.y + b.h <= H for b in r.boxes):
            bad.append(f"a box outside the {W}x{H} frame")
        boxes[f"{W}x{H}"] = len(r.boxes)
        frames.append(g)
        lines.append(twin_line(r))
    if cbin:
        got = c_lines(cbin, run_dir / "final.vtsp", frames)
        if got != lines:
            bad.append(f"the C driver {Path(cbin).name} differs from the twin on the procedural frames")
        boxes["c_runtime"] = "equal to the twin"
    notes = []
    for f, want_sha in info["code_sha256"].items():
        if not (ROOT / f).exists() or sha(ROOT / f) != want_sha:
            notes.append(f"code {f}: the run's sha256 is not this checkout's")
    for f, want_sha in info["manifests_sha256"].items():
        p = Path(f) if Path(f).is_absolute() else ROOT / f
        if not p.exists() or sha(p) != want_sha:
            notes.append(f"manifest {f}: the run's sha256 is not this checkout's")
    return info, done, bad, notes, boxes


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--finished")
    ap.add_argument("--cbin2", help="runtime/vocr_textspot2_test: family T's C driver, compared with the twin")
    ap.add_argument("--cbin3", help="runtime/vocr_textspot3_test: family M's C driver, compared with the twin")
    ap.add_argument("--plan", help="the sweep's plan: the run must be its record (recipe, family, shape, seed, steps, "
                                   "extras, teacher)")
    args = ap.parse_args(argv)
    info, done, bad, notes, boxes = check(args.run, args.cbin2, args.cbin3, args.plan)
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
    print(f"CHECK_OK {line} family {info['family']} boxes {boxes} skip-dropped {sk.get('dropped_total')} "
          f"({info['started']} -> {done['finished']})")


if __name__ == "__main__":
    main()
