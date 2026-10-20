#!/usr/bin/env python3
"""The runs of textspot2's training sweep (docs/textspot2.md "Arms and stages", eval/textspot2_protocol.json
"stages") and each run's trainer command line, for sweep.sh.

  textspot2_plan.py stage1 [--est PROBE_JSON] --out PLAN.jsonl
          stage 1's 38 runs: the ladder on s4_c24n4 (lab, data, ema, cls; seeds 1-3) and r0 seed 3; long on
          s4_c24n4 and s4p_c8 (seeds 1-3); data on s4p_c8, s4p_c6n4, s4p_c8n4, s4p_c10n4, s4p_c12, s4_c32n3
          (seeds 1-3) and r0 seed 3 of s4p_c8
  textspot2_plan.py stage2 --choices CHOICES.json [--runs-dir DIR] [--est PROBE_JSON] [--datapath JSON] \
          --out PLAN.jsonl
          stage 2's runs from the conductor's choices (docs/textspot2.md "Arms and stages", stage 2; below): kd on
          the fast shape with data + the kept extras from the teacher run's final.pt, long (ema, + cls when kept) on
          every shape of long_on, the d2s shapes under data + the kept extras; seeds 1-3 each; longest first by the
          committed probe records' measured step times (estimates)
  textspot2_plan.py stage2 --extras X [--kd-shape S --kd-teacher DIR] [--long-shapes A,B] [--d2s-shapes C,D] \
          [--est PROBE_JSON] --out PLAN.jsonl
          the same from explicit arguments (tests, ad hoc)
  textspot2_plan.py list PLAN.jsonl            the run names, in the plan's order (longest first)
  textspot2_plan.py args PLAN.jsonl RUN --out-dir D
                                               the run's argv after the interpreter, NUL-separated (sweep.sh
                                               reads it with mapfile -d ''); a kd record's teacher final.pt must
                                               still have the plan's sha256
  textspot2_plan.py timeout PLAN.jsonl RUN     the run's timeout in seconds

A plan line: {"run", "trainer", "recipe", "shape", "seed", "steps", "extras", "teacher", "data", "est_s",
"timeout_s"}. Data per run (the conductor, 2026-10-20T09:00Z): r0 reads what textspot's sweep read
(data/manifests/textspot_shards_v1.json + data/manifests/textdet_shards_v1.json; synth_frames and #97's real
directories); lab reads textspot_shards_v1.json + textspot2_shards_v1.json with synth_frames and the pin's lab_<source>
directories; every other recipe also its more_<source> directories. A pin's directory basename is looked up under
$T2/shards/{lab,more} and $P2/shards/{lab,more} (exactly one match). The order is longest first by the probe record's
estimate (est_s), else by steps x MACs per 256 x 512 window; ties by name.

CHOICES.json (the conductor's, from the scores of stage 1; every key required, no other key):
  {"ema_kept": bool, "cls_kept": bool, "fast_shape": S, "main_shape": S, "kd_teacher_run": RUN, "long_on": [S...],
   "d2s": bool}
Checked against the protocol: the fast and main shapes are shapes of stage 1's data arms; kd_teacher_run is a
stage-1 run other than r0 (a control, never a candidate) that has finished in --runs-dir (default
$HERD/shared/artifacts/textspot2/runs): its final.pt and done.json exist and agree, and the plan records its sha256;
long_on is exactly the main and fast shapes that are not s4_c24n4 or s4p_c8 (stage 1 ran long on those). A kd
record's teacher is written as "$HERD/shared/artifacts/textspot2/runs/<run>" and expanded at the command line.
Stage 2's estimates: the probe's measured ms per step of the student's shape under data (d2s shapes: the data-path
probe's), plus for kd the teacher's forward, (kd - data) on s4_c24n4 scaled by the teacher shape's data step time;
QAT at the probe's median ratio (kd: its own smoke's); the overhead of the data (kd: kd) smoke run.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "train"))
import textdet_model as TM  # noqa: E402
import textspot2_shapes as S2  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
SEEDS = (1, 2, 3)
LADDER = ("lab", "data", "ema", "cls")
LONG1 = ("s4_c24n4", "s4p_c8")
DATA1 = ("s4p_c8", "s4p_c6n4", "s4p_c8n4", "s4p_c10n4", "s4p_c12", "s4_c32n3")
R0 = ("s4_c24n4", "s4p_c8")
STEPS = {"long": 60000}
TIMEOUT = {30000: 21600, 60000: 43200}       # 6 h and 12 h: about 4x the probe's estimates; a hang, not a slow run
SYNTH_MANIFEST = "data/manifests/textspot_shards_v1.json"
TEXTDET_MANIFEST = "data/manifests/textdet_shards_v1.json"
T2_MANIFEST = "data/manifests/textspot2_shards_v1.json"
SOURCES = ("realplus", "teacher", "devnative", "owner")
PROBE_JSON = ROOT / "eval" / "textspot2_train_probe_20261020.json"
DATAPATH_JSON = ROOT / "eval" / "textspot2_train_datapath_20261020.json"
CHOICE_KEYS = {"ema_kept": bool, "cls_kept": bool, "fast_shape": str, "main_shape": str, "kd_teacher_run": str,
               "long_on": list, "d2s": bool}
TEACHER_RUNS = "$HERD/shared/artifacts/textspot2/runs"


def paths(herd=HERD):
    a, p = herd / "shared" / "artifacts", herd / "shared" / "private"
    return {"TS": a / "textspot", "TD": a / "textdet", "PD": p / "textdet", "T2": a / "textspot2",
            "P2": p / "textspot2"}


def spec(recipe, shape, seed, extras=None, teacher=None, **more):
    """One plan record. more: extra keys of a stage-2 record (a kd run's teacher_run and teacher_pt_sha256);
    stage 1's records have none, so its plan is the same as before they existed."""
    steps = STEPS.get(recipe, 30000)
    trainer = "train/textspot_train.py" if recipe == "r0" else "train/textspot2_train.py"
    data = "r0" if recipe == "r0" else ("lab" if recipe == "lab" else "data")
    return {"run": f"{recipe}_h2_{shape}_seed{seed}", "trainer": trainer, "recipe": recipe, "shape": shape,
            "seed": seed, "steps": steps, "extras": extras, "teacher": teacher, "data": data,
            "timeout_s": TIMEOUT[steps], **more}


def stage1():
    runs = [spec(r, "s4_c24n4", n) for r in LADDER for n in SEEDS]
    runs.append(spec("r0", "s4_c24n4", 3))
    runs += [spec("long", sh, n) for sh in LONG1 for n in SEEDS]
    runs += [spec("data", sh, n) for sh in DATA1 for n in SEEDS]
    runs.append(spec("r0", "s4p_c8", 3))
    return runs


def stage2(extras, kd_shape=None, kd_teacher=None, long_shapes=(), d2s_shapes=()):
    """extras: the kept extras ('none', 'ema', 'cls' or 'ema,cls'). long always has ema (data + ema), plus cls
    when it is kept."""
    ex = [x for x in ("ema", "cls") if x in (extras or "").split(",")]
    runs = []
    if kd_shape:
        if not kd_teacher:
            raise SystemExit("--kd-shape needs --kd-teacher")
        runs += [spec("kd", kd_shape, n, ",".join(ex) or "none", str(kd_teacher)) for n in SEEDS]
    long_ex = ",".join(x for x in ("ema", "cls") if x == "ema" or x in ex)
    runs += [spec("long", sh, n, long_ex) for sh in long_shapes for n in SEEDS]
    runs += [spec("data", sh, n, ",".join(ex) or "none") for sh in d2s_shapes for n in SEEDS]
    names = [r["run"] for r in runs]
    if len(set(names)) != len(names):
        raise SystemExit("stage 2 names a run twice")
    return runs


def parse_run(name):
    """'<recipe>_h2_<shape>_seed<n>' -> (recipe, shape, n)."""
    head, _, n = name.rpartition("_seed")
    recipe, _, shape = head.partition("_h2_")
    if not (recipe and shape and n.isdigit()):
        raise ValueError(f"{name!r} is not <recipe>_h2_<shape>_seed<n>")
    return recipe, shape, int(n)


def stage1_data_shapes():
    """The shapes of stage 1's data arms (the fast and main shape are chosen among them)."""
    return sorted({r["shape"] for r in stage1() if r["recipe"] == "data"})


def read_choices(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise SystemExit(f"{path}: a JSON object of {sorted(CHOICE_KEYS)}")
    return doc


def check_choices(c):
    """The protocol's checks of the conductor's stage-2 choices (the module docstring); SystemExit on any."""
    missing, extra = sorted(set(CHOICE_KEYS) - set(c)), sorted(set(c) - set(CHOICE_KEYS))
    if missing or extra:
        raise SystemExit(f"choices: missing {missing}, unknown {extra}")
    for k, typ in CHOICE_KEYS.items():
        if type(c[k]) is not typ:
            raise SystemExit(f"choices: {k} must be a {typ.__name__}, not {c[k]!r}")
    data_shapes = stage1_data_shapes()
    for k in ("fast_shape", "main_shape"):
        if c[k] not in data_shapes:
            raise SystemExit(f"choices: {k} {c[k]!r} is not a shape of stage 1's data arms {data_shapes}")
    if not all(isinstance(s, str) for s in c["long_on"]) or len(set(c["long_on"])) != len(c["long_on"]):
        raise SystemExit(f"choices: long_on {c['long_on']!r} must be distinct shape names")
    rule = sorted({s for s in (c["main_shape"], c["fast_shape"]) if s not in LONG1})
    if sorted(c["long_on"]) != rule:
        raise SystemExit(f"choices: long_on {sorted(c['long_on'])} is not the protocol's {rule} (the main and fast "
                         f"shapes that are not {', '.join(LONG1)})")
    names = {r["run"]: r for r in stage1()}
    t = names.get(c["kd_teacher_run"])
    if t is None or t["recipe"] == "r0":
        raise SystemExit(f"choices: kd_teacher_run {c['kd_teacher_run']!r} is not a stage-1 candidate run "
                         "(a stage-1 run other than r0)")
    return c


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


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


def stage2_from_choices(c, runs_dir=None):
    """Stage 2's runs from the conductor's checked choices: kd on the fast shape (data + the kept extras, the
    teacher run's final.pt), long on long_on (ema, + cls when kept), the d2s shapes under data + the kept extras
    when d2s; seeds 1-3 each. kd records carry teacher_run and teacher_pt_sha256."""
    check_choices(c)
    runs_dir = Path(runs_dir) if runs_dir else paths()["T2"] / "runs"
    kept = ",".join(x for x, k in (("ema", "ema_kept"), ("cls", "cls_kept")) if c[k]) or "none"
    sha = teacher_record(c["kd_teacher_run"], runs_dir)
    runs = stage2(kept, c["fast_shape"], f"{TEACHER_RUNS}/{c['kd_teacher_run']}", c["long_on"],
                  sorted(S2.D2S) if c["d2s"] else [])
    for r in runs:
        if r["recipe"] == "kd":
            r.update(teacher_run=c["kd_teacher_run"], teacher_pt_sha256=sha)
    clash = sorted({r["run"] for r in runs} & {r["run"] for r in stage1()})
    if clash:
        raise SystemExit(f"stage 2 names stage-1 runs: {clash}")
    return runs


def stage2_estimates(runs, probe_json=PROBE_JSON, datapath_json=DATAPATH_JSON):
    """Estimated seconds per stage-2 run (the module docstring), or {} when a figure is missing (order() then
    falls back to steps x MACs)."""
    try:
        pr = json.loads(Path(probe_json).read_text(encoding="utf-8"))
        dp = json.loads(Path(datapath_json).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    ms = {(p["recipe"], p["shape"]): p["ms_per_step"] for p in pr["probes"]}
    data = {s: v for (rec, s), v in ms.items() if rec == "data"}
    for row in dp["rows"]:
        if row["recipe"] == "data" and row.get("ms_per_step") and S2.SHAPES[row["shape"]]["head"] == "kroa_x2":
            data.setdefault(row["shape"], row["ms_per_step"])
    ref = data.get("s4_c24n4")
    if ref is None or ("kd", "s4_c24n4") not in ms:
        return {}
    teacher_ref = ms[("kd", "s4_c24n4")] - ref
    extra = {x: max(0.0, ms.get((x, "s4_c24n4"), ref) - ref) for x in ("ema", "cls")}
    smokes, ratio = pr["smokes"], pr["stage1_estimate"]["qat_ratio_median"]
    out = {}
    for r in runs:
        if r["shape"] not in data:
            return {}
        m = data[r["shape"]] + sum(extra[x] for x in ("ema", "cls") if x in (r.get("extras") or "").split(","))
        q, over = ratio, smokes["data"]["overhead_s"]
        if r["recipe"] == "kd":
            t_shape = parse_run(r["teacher_run"] if r.get("teacher_run") else Path(r["teacher"]).name)[1]
            if t_shape not in data:
                return {}
            m += teacher_ref * data[t_shape] / ref
            q, over = smokes["kd"]["qat_ratio"] or ratio, smokes["kd"]["overhead_s"]
        out[r["run"]] = over + r["steps"] * m / 1000 * (0.8 + 0.2 * q)
    return out


def expand(path, herd=HERD):
    """A plan path with a leading $HERD (the repo's placeholder) -> the path under herd."""
    path = str(path)
    return str(Path(herd) / path[len("$HERD/"):]) if path.startswith("$HERD/") else path


def macs(shape):
    """MACs of one 256 x 512 window: textdet_model.cost on the protocol's shape dict, or textspot2_export.cost
    for a d2s shape (head kroa_x2)."""
    s = S2.SHAPES[shape]
    d = {k: (list(s[k]) if isinstance(s[k], list) else s[k])
         for k in ("name", "stride", "c0", "c1", "c2", "chans", "dil", "block", "head")}
    if s["head"] == "kroa_x2":
        import textspot2_export as X
        return X.cost(d, 512, 256)["macs"]
    return TM.cost(d, 512, 256)["macs"]


def order(runs, est=None):
    """Longest first: the probe record's estimate per run when it has one for every run, else steps x MACs."""
    est = est or {}
    if all(r["run"] in est for r in runs):
        for r in runs:
            r["est_s"] = round(float(est[r["run"]]))
        return sorted(runs, key=lambda r: (-r["est_s"], r["run"]))
    for r in runs:
        r["est_s"] = None
    return sorted(runs, key=lambda r: (-(r["steps"] * (macs(r["shape"]) or 0)), r["run"]))


def read_est(path):
    if not path:
        return {}
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return {k: v for k, v in ((doc.get("stage1_estimate") or {}).get("est_run_s") or {}).items()}


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


def data_args(r, repo=ROOT, herd=HERD):
    p = paths(herd)
    if r["data"] == "r0":
        return ([SYNTH_MANIFEST, TEXTDET_MANIFEST],
                [str(p["TS"] / "shards" / "synth_frames"), str(p["TD"] / "shards" / "realplus"),
                 str(p["TD"] / "shards" / "teacher"), str(p["TD"] / "shards" / "devnative"),
                 str(p["PD"] / "shards" / "owner")])
    roots = [p["T2"] / "shards" / "lab", p["T2"] / "shards" / "more", p["P2"] / "shards" / "lab",
             p["P2"] / "shards" / "more"]
    man = Path(repo) / T2_MANIFEST
    dirs = [str(p["TS"] / "shards" / "synth_frames")] + pin_dirs(man, "lab", roots)
    if r["data"] == "data":
        dirs += pin_dirs(man, "more", roots)
    return [SYNTH_MANIFEST, T2_MANIFEST], dirs


def argv(r, out_dir, repo=ROOT, herd=HERD):
    mans, dirs = data_args(r, repo, herd)
    out = str(Path(out_dir) / r["run"])
    if r["recipe"] == "r0":
        return [r["trainer"], "--shape", r["shape"], "--scale", "h2", "--seed", str(r["seed"]), "--manifest", *mans,
                "--shards", *dirs, "--out", out, "--threads", "5"]
    a = [r["trainer"], "--recipe", r["recipe"], "--shape", r["shape"], "--seed", str(r["seed"]),
         "--manifest", *mans, "--shards", *dirs, "--out", out, "--threads", "5"]
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


def names(text):
    return [x for x in text.split(",") if x]


def load(plan):
    return [json.loads(x) for x in Path(plan).read_text(encoding="utf-8").splitlines() if x.strip()]


def write(runs, out):
    Path(out).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in runs), encoding="utf-8")


def main(argv_=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("stage1")
    a1.add_argument("--est")
    a1.add_argument("--out", required=True)
    a2 = sub.add_parser("stage2")
    a2.add_argument("--choices", help="the conductor's choices (the module docstring)")
    a2.add_argument("--runs-dir", help="where the kd teacher run is (default $HERD/shared/artifacts/textspot2/runs)")
    a2.add_argument("--datapath", default=str(DATAPATH_JSON), help="the data-path probe record (d2s step times)")
    a2.add_argument("--extras")
    a2.add_argument("--kd-shape")
    a2.add_argument("--kd-teacher")
    a2.add_argument("--long-shapes", default="")
    a2.add_argument("--d2s-shapes", default="")
    a2.add_argument("--est")
    a2.add_argument("--out", required=True)
    a3 = sub.add_parser("list")
    a3.add_argument("plan")
    a4 = sub.add_parser("args")
    a4.add_argument("plan")
    a4.add_argument("run")
    a4.add_argument("--out-dir", required=True)
    a5 = sub.add_parser("timeout")
    a5.add_argument("plan")
    a5.add_argument("run")
    args = ap.parse_args(argv_)
    if args.cmd in ("stage1", "stage2"):
        if args.cmd == "stage1":
            runs = order(stage1(), read_est(args.est))
        elif args.choices:
            if args.extras or args.kd_shape or args.kd_teacher or args.long_shapes or args.d2s_shapes:
                raise SystemExit("--choices or --extras ...: not both")
            runs = stage2_from_choices(read_choices(args.choices), args.runs_dir)
            runs = order(runs, stage2_estimates(runs, args.est or PROBE_JSON, args.datapath))
        else:
            if args.extras is None:
                raise SystemExit("stage2: --choices, or --extras with the runs' arguments")
            runs = stage2(args.extras, args.kd_shape, args.kd_teacher, names(args.long_shapes),
                          names(args.d2s_shapes))
            runs = order(runs, read_est(args.est))
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
        return
    sys.stdout.write("\0".join(argv(runs[args.run], args.out_dir)) + "\0")


if __name__ == "__main__":
    main()
