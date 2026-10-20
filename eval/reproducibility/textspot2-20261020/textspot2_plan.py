#!/usr/bin/env python3
"""The runs of textspot2's training sweep (docs/textspot2.md "Arms and stages", eval/textspot2_protocol.json
"stages") and each run's trainer command line, for sweep.sh.

  textspot2_plan.py stage1 [--est PROBE_JSON] --out PLAN.jsonl
          stage 1's 38 runs: the ladder on s4_c24n4 (lab, data, ema, cls; seeds 1-3) and r0 seed 3; long on
          s4_c24n4 and s4p_c8 (seeds 1-3); data on s4p_c8, s4p_c6n4, s4p_c8n4, s4p_c10n4, s4p_c12, s4_c32n3
          (seeds 1-3) and r0 seed 3 of s4p_c8
  textspot2_plan.py stage2 --extras X [--kd-shape S --kd-teacher DIR] [--long-shapes A,B] [--d2s-shapes C,D] \
          [--est PROBE_JSON] --out PLAN.jsonl
          stage 2's runs as the conductor's amendment names them: kd on the fast shape from the teacher run with
          the kept extras, long on the listed shapes with the kept extras (ema always: long is data + ema), d2s
          shapes under data + the kept extras; seeds 1-3 each
  textspot2_plan.py list PLAN.jsonl            the run names, in the plan's order (longest first)
  textspot2_plan.py args PLAN.jsonl RUN --out-dir D
                                               the run's argv after the interpreter, NUL-separated (sweep.sh
                                               reads it with mapfile -d '')
  textspot2_plan.py timeout PLAN.jsonl RUN     the run's timeout in seconds

A plan line: {"run", "trainer", "recipe", "shape", "seed", "steps", "extras", "teacher", "data", "est_s",
"timeout_s"}. Data per run (the conductor, 2026-10-20T09:00Z): r0 reads what textspot's sweep read
(data/manifests/textspot_shards_v1.json + data/manifests/textdet_shards_v1.json; synth_frames and #97's real
directories); lab reads textspot_shards_v1.json + textspot2_shards_v1.json with synth_frames and the pin's lab_<source>
directories; every other recipe also its more_<source> directories. A pin's directory basename is looked up under
$T2/shards/{lab,more} and $P2/shards/{lab,more} (exactly one match). The order is longest first by the probe record's
estimate (est_s), else by steps x MACs per 256 x 512 window; ties by name.
"""
import argparse
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


def paths(herd=HERD):
    a, p = herd / "shared" / "artifacts", herd / "shared" / "private"
    return {"TS": a / "textspot", "TD": a / "textdet", "PD": p / "textdet", "T2": a / "textspot2",
            "P2": p / "textspot2"}


def spec(recipe, shape, seed, extras=None, teacher=None):
    steps = STEPS.get(recipe, 30000)
    trainer = "train/textspot_train.py" if recipe == "r0" else "train/textspot2_train.py"
    data = "r0" if recipe == "r0" else ("lab" if recipe == "lab" else "data")
    return {"run": f"{recipe}_h2_{shape}_seed{seed}", "trainer": trainer, "recipe": recipe, "shape": shape,
            "seed": seed, "steps": steps, "extras": extras, "teacher": teacher, "data": data,
            "timeout_s": TIMEOUT[steps]}


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
        a += ["--teacher", r["teacher"]]
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
    a2.add_argument("--extras", required=True)
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
            runs = stage1()
        else:
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
