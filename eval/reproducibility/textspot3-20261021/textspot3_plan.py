#!/usr/bin/env python3
"""The runs of textspot3's training sweep (docs/textspot3.md "Arms and stages", eval/textspot3_protocol.json
"stages") and each run's trainer command line, for sweep.sh.

  textspot3_plan.py stage1 --family T|M [--protocol P] [--est PROBE_JSON] --out PLAN.jsonl
          stage 1 of one family, from the protocol's registered shapes (its "shapes", added by an amendment):
          T: base on every registered T shape; M: base on S, M and L, and size, clean and long on M; seeds 1-3 each
  textspot3_plan.py stage2 --choices CHOICES.json [--runs-dir DIR] [--protocol P] [--est PROBE_JSON] --out PLAN.jsonl
          stage 2 from the conductor's choices (below): kd on each of the two students with base + the kept extras
          from the teacher run's final.pt, and base + the kept extras on L when an extra is kept; seeds 1-3 each
  textspot3_plan.py list PLAN.jsonl            the run names, in the plan's order (longest first)
  textspot3_plan.py args PLAN.jsonl RUN --out-dir D
                                               the run's argv after the interpreter, NUL-separated (sweep.sh reads it
                                               with mapfile -d ''); a kd record's teacher final.pt must still have the
                                               plan's sha256
  textspot3_plan.py timeout PLAN.jsonl RUN     the run's timeout in seconds
  textspot3_plan.py family PLAN.jsonl RUN      the run's family (T or M: which C driver checks it)

A plan line: {"run", "trainer", "recipe", "family", "shape", "seed", "steps", "extras", "teacher", "est_s",
"timeout_s"} (kd: also "teacher_run", "teacher_pt_sha256"). Every run reads the protocol's data:
data/manifests/textspot_shards_v1.json + data/manifests/textspot2_shards_v1.json, with synth_frames and the pin's
lab_<source> and more_<source> directories (a pin directory's basename is looked up under $T2/shards/{lab,more} and
$P2/shards/{lab,more}: exactly one match). The order is longest first by the probe record's estimate (est_s), else by
steps x MACs per 256 x 512 window; ties by name. The timeout is a hang guard: 4 x the estimate rounded up to the hour
and at least 12 h for 30,000 steps (24 h for 60,000), or those floors without an estimate.

Registered shapes (eval/textspot3_protocol.json "shapes", each with its "family", "T" or "M"; an M shape's size point
in "point", "S", "M" or "L"; or --m-points S=name,M=name,L=name). Every plan records the protocol's sha256.

CHOICES.json (the conductor's, from stage 1's scores; every key required, no other key):
  {"size_kept": bool, "clean_kept": bool, "kd_teacher_run": RUN, "kd_students": [SHAPE, SHAPE]}
Checked against the protocol: kd_teacher_run is a stage-1 run of an M shape that has finished in --runs-dir
(default $HERD/shared/artifacts/textspot3/runs): its final.pt and done.json exist and agree, and the plan records its
sha256; kd_students are two distinct registered shapes other than the teacher's. A kd record's teacher is written as
"$HERD/shared/artifacts/textspot3/runs/<run>" and expanded at the command line. Base + the kept extras on L is the
recipe size (size kept), clean (clean kept) or base --extras size,clean (both: named size+clean_<L>_seed<n>).
"""
import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT / "train", ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
PROTOCOL = ROOT / "eval" / "textspot3_protocol.json"
SEEDS = (1, 2, 3)
POINTS = ("S", "M", "L")
STEPS = {"long": 60000}
FLOOR = {30000: 43200, 60000: 86400}        # 12 h and 24 h
TRAINER = "train/textspot3_train.py"
SYNTH_MANIFEST = "data/manifests/textspot_shards_v1.json"
T2_MANIFEST = "data/manifests/textspot2_shards_v1.json"
CHOICE_KEYS = {"size_kept": bool, "clean_kept": bool, "kd_teacher_run": str, "kd_students": list}
TEACHER_RUNS = "$HERD/shared/artifacts/textspot3/runs"
NAME_RECIPE = {(): "base", ("size",): "size", ("clean",): "clean", ("size", "clean"): "size+clean"}


def paths(herd=HERD):
    a, p = herd / "shared" / "artifacts", herd / "shared" / "private"
    return {"TS": a / "textspot", "T2": a / "textspot2", "P2": p / "textspot2", "T3": a / "textspot3"}


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- the protocol's shapes

def shapes(protocol=PROTOCOL):
    """The registered shapes: {name: dict with "family"} from the protocol's "shapes" (non-dict entries, such as a
    grammar note, are skipped)."""
    doc = json.loads(Path(protocol).read_text(encoding="utf-8"))
    out = {}
    for name, s in (doc.get("shapes") or {}).items():
        if isinstance(s, dict):
            if s.get("family") not in ("T", "M"):
                raise SystemExit(f"{protocol}: shape {name} has no family T or M")
            out[name] = {"name": name, **s}
    return out


def m_points(reg, given=None):
    """{"S": name, "M": name, "L": name}: --m-points when given (every name a registered M shape), else the M shapes'
    "point" fields (exactly one shape per point)."""
    ms = {k: v for k, v in reg.items() if v["family"] == "M"}
    if given:
        pts = dict(given)
    else:
        pts = {}
        for name, s in sorted(ms.items()):
            p = s.get("point")
            if p in POINTS:
                if p in pts:
                    raise SystemExit(f"two M shapes at point {p}: {pts[p]}, {name}")
                pts[p] = name
    if sorted(pts) != sorted(POINTS) or any(v not in ms for v in pts.values()) or len(set(pts.values())) != 3:
        raise SystemExit(f"the M points {pts} are not three registered M shapes at S, M and L "
                         f"(registered M shapes: {sorted(ms)})")
    return pts


def parse_points(text):
    if not text:
        return None
    out = {}
    for item in text.split(","):
        k, _, v = item.partition("=")
        if k not in POINTS or not v or k in out:
            raise SystemExit(f"--m-points {text!r}: S=name,M=name,L=name")
        out[k] = v
    return out


# ----------------------------------------------------------------------------- runs

def run_name(recipe, shape, seed, extras=()):
    head = recipe if recipe in ("kd", "long") else NAME_RECIPE[tuple(extras)]
    return f"{head}_{shape}_seed{seed}"


def spec(recipe, family, shape, seed, extras=None, teacher=None, **more):
    """One plan record. extras: None (the recipe's own) or a comma list (base and kd: the kept extras)."""
    steps = STEPS.get(recipe, 30000)
    own = {"size": ("size",), "clean": ("clean",)}.get(recipe, ())
    ex = tuple(x for x in ("size", "clean") if x in (extras or "").split(",")) if extras is not None else own
    return {"run": run_name(recipe, shape, seed, ex), "trainer": TRAINER, "recipe": recipe, "family": family,
            "shape": shape, "seed": seed, "steps": steps, "extras": extras, "teacher": teacher, **more}


def stage1(family, reg, points=None):
    if family == "T":
        ts = sorted(k for k, v in reg.items() if v["family"] == "T")
        if not ts:
            raise SystemExit("no registered T shape")
        return [spec("base", "T", sh, n) for sh in ts for n in SEEDS]
    pts = m_points(reg, points)
    runs = [spec("base", "M", pts[p], n) for p in POINTS for n in SEEDS]
    runs += [spec(r, "M", pts["M"], n) for r in ("size", "clean", "long") for n in SEEDS]
    return runs


def parse_run(name):
    """'<recipe>_<shape>_seed<n>' -> (recipe part, shape, n)."""
    head, _, n = name.rpartition("_seed")
    recipe, _, shape = head.partition("_")
    if not (recipe and shape and n.isdigit()):
        raise ValueError(f"{name!r} is not <recipe>_<shape>_seed<n>")
    return recipe, shape, int(n)


def read_choices(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise SystemExit(f"{path}: a JSON object of {sorted(CHOICE_KEYS)}")
    return doc


def teacher_record(name, runs_dir):
    """The teacher run's final.pt sha256, after checking that the run finished: final.pt and done.json exist and
    done.json's pt_sha256 is final.pt's."""
    d = Path(runs_dir) / name
    pt, done = d / "final.pt", d / "done.json"
    if not pt.exists() or not done.exists():
        raise SystemExit(f"{d}: the kd teacher has not finished (final.pt and done.json)")
    sha = file_sha256(pt)
    if json.loads(done.read_text(encoding="utf-8")).get("pt_sha256") != sha:
        raise SystemExit(f"{d}: done.json's pt_sha256 is not final.pt's")
    return sha


def check_choices(c, reg, points=None):
    missing, extra = sorted(set(CHOICE_KEYS) - set(c)), sorted(set(c) - set(CHOICE_KEYS))
    if missing or extra:
        raise SystemExit(f"choices: missing {missing}, unknown {extra}")
    for k, typ in CHOICE_KEYS.items():
        if type(c[k]) is not typ:
            raise SystemExit(f"choices: {k} must be a {typ.__name__}, not {c[k]!r}")
    stage1_runs = {r["run"]: r for fam in ("T", "M") for r in (stage1(fam, reg, points) if
                                                               any(v["family"] == fam for v in reg.values()) else [])}
    t = stage1_runs.get(c["kd_teacher_run"])
    if t is None or t["family"] != "M":
        raise SystemExit(f"choices: kd_teacher_run {c['kd_teacher_run']!r} is not a stage-1 run of an M shape")
    st = c["kd_students"]
    if len(st) != 2 or len(set(st)) != 2 or not all(isinstance(s, str) and s in reg for s in st):
        raise SystemExit(f"choices: kd_students {st!r} must be two distinct registered shapes")
    if t["shape"] in st:
        raise SystemExit(f"choices: the teacher's shape {t['shape']} is a kd student")
    return c


def stage2_from_choices(c, reg, runs_dir=None, points=None):
    """Stage 2's runs from the conductor's checked choices; kd records carry teacher_run and teacher_pt_sha256."""
    check_choices(c, reg, points)
    runs_dir = Path(runs_dir) if runs_dir else paths()["T3"] / "runs"
    kept = [x for x, k in (("size", "size_kept"), ("clean", "clean_kept")) if c[k]]
    sha = teacher_record(c["kd_teacher_run"], runs_dir)
    teacher = f"{TEACHER_RUNS}/{c['kd_teacher_run']}"
    runs = [spec("kd", reg[sh]["family"], sh, n, ",".join(kept) or "none", teacher,
                 teacher_run=c["kd_teacher_run"], teacher_pt_sha256=sha) for sh in c["kd_students"] for n in SEEDS]
    if kept:
        L = m_points(reg, points)["L"]
        if kept == ["size"]:
            runs += [spec("size", "M", L, n) for n in SEEDS]
        elif kept == ["clean"]:
            runs += [spec("clean", "M", L, n) for n in SEEDS]
        else:
            runs += [spec("base", "M", L, n, "size,clean") for n in SEEDS]
    names = [r["run"] for r in runs]
    s1 = {r["run"] for fam in ("T", "M") if any(v["family"] == fam for v in reg.values())
          for r in stage1(fam, reg, points)}
    if len(set(names)) != len(names) or set(names) & s1:
        raise SystemExit(f"stage 2 names a run twice or a stage-1 run: {sorted(set(names) & s1)}")
    return runs


# ----------------------------------------------------------------------------- order, timeouts, command lines

def macs(reg, shape):
    """MACs of one 256 x 512 window: textspot2_export.cost (T) or textspot3_model.cost (M)."""
    s = reg[shape]
    if s["family"] == "T":
        import textspot2_export as X
        d = {k: (list(s[k]) if isinstance(s[k], list) else s[k])
             for k in ("name", "stride", "c0", "c1", "c2", "chans", "dil", "block", "head")}
        return X.cost(d, 512, 256)["macs"]
    import textspot3_model as M3
    return M3.cost(s, 512, 256)["macs"]


def timeout_of(r):
    floor = FLOOR[r["steps"]]
    if r.get("est_s"):
        return max(floor, 3600 * math.ceil(4 * r["est_s"] / 3600))
    return floor


def order(runs, reg, est=None):
    """Longest first: the probe record's estimate per run when it has one for every run, else steps x MACs."""
    est = est or {}
    if all(r["run"] in est for r in runs):
        for r in runs:
            r["est_s"] = round(float(est[r["run"]]))
        out = sorted(runs, key=lambda r: (-r["est_s"], r["run"]))
    else:
        for r in runs:
            r["est_s"] = None
        out = sorted(runs, key=lambda r: (-(r["steps"] * macs(reg, r["shape"])), r["run"]))
    for r in out:
        r["timeout_s"] = timeout_of(r)
    return out


def read_est(path):
    """{run: estimated seconds} from a probe record's "estimate"."est_run_s" (probe_summary.py), {} without one."""
    if not path:
        return {}
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(((doc.get("estimate") or {}).get("est_run_s") or {}))


def expand(path, herd=HERD):
    """A plan path with a leading $HERD (the repo's placeholder) -> the path under herd."""
    path = str(path)
    return str(Path(herd) / path[len("$HERD/"):]) if path.startswith("$HERD/") else path


def pin_dirs(manifest, part, roots):
    """The directories of a textspot2 pin's shards whose basename starts with part + '_' (lab or more), each found
    under exactly one of roots."""
    doc = json.loads(Path(manifest).read_text(encoding="utf-8"))
    names = sorted({s["dir"] for s in doc["shards"] if s["dir"].startswith(part + "_")})
    if not names:
        raise SystemExit(f"{manifest}: no {part}_<source> shards")
    out = []
    for n in names:
        hits = [Path(r) / n for r in roots if (Path(r) / n).is_dir()]
        if len(hits) != 1:
            raise SystemExit(f"{n}: found {len(hits)} times under {[str(r) for r in roots]}")
        out.append(str(hits[0]))
    return out


def data_args(repo=ROOT, herd=HERD):
    p = paths(herd)
    roots = [p["T2"] / "shards" / "lab", p["T2"] / "shards" / "more", p["P2"] / "shards" / "lab",
             p["P2"] / "shards" / "more"]
    man = Path(repo) / T2_MANIFEST
    dirs = [str(p["TS"] / "shards" / "synth_frames")] + pin_dirs(man, "lab", roots) + pin_dirs(man, "more", roots)
    return [SYNTH_MANIFEST, T2_MANIFEST], dirs


def argv(r, out_dir, repo=ROOT, herd=HERD):
    mans, dirs = data_args(repo, herd)
    a = [r["trainer"], "--recipe", r["recipe"], "--family", r["family"], "--shape", r["shape"], "--seed",
         str(r["seed"]), "--manifest", *mans, "--shards", *dirs, "--out", str(Path(out_dir) / r["run"]),
         "--threads", "5"]
    if r.get("extras") is not None:
        a += ["--extras", r["extras"]]
    if r.get("teacher"):
        teacher = expand(r["teacher"], herd)
        if r.get("teacher_pt_sha256"):
            pt = Path(teacher) / "final.pt"
            if not pt.exists() or file_sha256(pt) != r["teacher_pt_sha256"]:
                raise SystemExit(f"{pt}: not the plan's teacher (sha256 {r['teacher_pt_sha256'][:16]}...)")
        a += ["--teacher", teacher]
    return a


def load(plan):
    return [json.loads(x) for x in Path(plan).read_text(encoding="utf-8").splitlines() if x.strip()]


def write(runs, out):
    Path(out).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in runs), encoding="utf-8")


def main(argv_=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("stage1")
    a1.add_argument("--family", required=True, choices=("T", "M"))
    a1.add_argument("--protocol", default=str(PROTOCOL))
    a1.add_argument("--m-points", help="S=name,M=name,L=name (else the M shapes' \"point\" fields)")
    a1.add_argument("--est")
    a1.add_argument("--out", required=True)
    a2 = sub.add_parser("stage2")
    a2.add_argument("--choices", required=True, help="the conductor's choices (the module docstring)")
    a2.add_argument("--runs-dir", help="where the kd teacher run is (default $HERD/shared/artifacts/textspot3/runs)")
    a2.add_argument("--protocol", default=str(PROTOCOL))
    a2.add_argument("--m-points")
    a2.add_argument("--est")
    a2.add_argument("--out", required=True)
    for name in ("list", "args", "timeout", "family"):
        s = sub.add_parser(name)
        s.add_argument("plan")
        if name != "list":
            s.add_argument("run")
        if name == "args":
            s.add_argument("--out-dir", required=True)
    args = ap.parse_args(argv_)
    if args.cmd in ("stage1", "stage2"):
        reg = shapes(args.protocol)
        pts = parse_points(args.m_points)
        if args.cmd == "stage1":
            runs = stage1(args.family, reg, pts)
        else:
            runs = stage2_from_choices(read_choices(args.choices), reg, args.runs_dir, pts)
        runs = order(runs, reg, read_est(args.est))
        sha = file_sha256(args.protocol)
        for r in runs:
            r["protocol_sha256"] = sha
        write(runs, args.out)
        print(f"{len(runs)} runs -> {args.out}")
        return
    runs = {r["run"]: r for r in load(args.plan)}
    if args.cmd == "list":
        print(" ".join(runs))
        return
    if args.run not in runs:
        raise SystemExit(f"{args.run}: not in {args.plan}")
    if args.cmd == "timeout":
        print(runs[args.run]["timeout_s"])
    elif args.cmd == "family":
        print(runs[args.run]["family"])
    else:
        sys.stdout.write("\0".join(argv(runs[args.run], args.out_dir)) + "\0")


if __name__ == "__main__":
    main()
