#!/usr/bin/env python3
"""textspot6's training sweep (issue #128, docs/textspot6.md "Model, recipe, seeds"): every run of a plan through
heavy.sh (5 threads each), at most MAX_PAR at once, in the plan's order.

  sweep.py OUT PLAN [--max-par 2 | --cap-file FILE] [--runs NAME,...] [--extra "--steps 300"]
  sweep.py plan --stage 1 --out PLAN.jsonl            the stage-1 plan of eval/textspot6_protocol.json
  sweep.py plan --stage 1b --out PLAN.jsonl           amendment 4: stage 1's remaining runs with seed 3 added

--cap-file: the number of runs at once is read from FILE (one integer) before each launch, so the label's slot share
(vocr6/slots.txt) can be kept as it changes; a run never stops early when the cap drops. HEAVY_SLOTS (env, default
15: heavy.sh's slots since 2026-10-26T10:20Z) is passed to heavy.sh.

Each run trains to its last step (nothing is selected or stopped early on a score). A run that exits without
done.json (a crash or its timeout) is moved to OUT/<run>.failed<k> and restarted from scratch with the same seed, at
most 3 attempts. Every start and end is appended to the shared experiment log (/local/home/skerraut/herd-synth3/vocr6/
experiments.jsonl, under flock). The checkout must be clean: every run records the commit it ran from.
"""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
A = HERD / "shared" / "artifacts"
PV = HERD / "shared" / "private"
EXPLOG = Path("/local/home/skerraut/herd-synth3/vocr6/experiments.jsonl")
HOOK_MANIFEST = REPO / "data" / "manifests" / "textspot6_pages_v1.json"
HOOK_SHARDS = PV / "vocr6-spotter" / "shards_v1"
MANIFESTS = [REPO / "data" / "manifests" / "textspot_shards_v1.json",
             REPO / "data" / "manifests" / "textspot2_shards_v1.json"]
SHARD_DIRS = [A / "textspot" / "shards" / "synth_frames",
              A / "textspot2" / "shards" / "lab" / "lab_realplus", A / "textspot2" / "shards" / "lab" / "lab_teacher",
              A / "textspot2" / "shards" / "lab" / "lab_devnative", PV / "textspot2" / "shards" / "lab" / "lab_owner",
              A / "textspot2" / "shards" / "more" / "more_realplus",
              A / "textspot2" / "shards" / "more" / "more_teacher",
              PV / "textspot2" / "shards" / "more" / "more_owner"]
TIMEOUT_S = 43200
HEAVY_SLOTS = os.environ.get("HEAVY_SLOTS", "15")
TMP = "/local/home/skerraut/kt/t6s"


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def git_head():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                          text=True).stdout.strip()


def explog(rec):
    rec = {"label": "vocr6-spotter", "time": utc(), **rec}
    with open(EXPLOG, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(json.dumps(rec) + "\n")
        fcntl.flock(f, fcntl.LOCK_UN)


def command(r, out):
    cmd = [str(REPO / ".venv" / "bin" / "python"), str(REPO / "train" / "textspot6_train.py"), "--recipe", r["recipe"],
           "--shape", r["shape"], "--seed", str(r["seed"]), "--out", str(out), "--hook-manifest", str(HOOK_MANIFEST),
           "--hook-shards", str(HOOK_SHARDS), "--hook-splits", r.get("splits", "train"),
           "--manifest", *map(str, MANIFESTS), "--shards", *map(str, SHARD_DIRS), "--threads", "5", "--workers", "1"]
    if r.get("teacher"):
        cmd += ["--teacher", r["teacher"]]
    if r.get("steps"):
        cmd += ["--steps", str(r["steps"])]
    return cmd


def plan_stage1():
    out = []
    for arm, recipe, shape in (("base_t64n8c16", "base", "t64n8c16"), ("base_qS4", "base", "qS4"),
                               ("base_qL5", "base", "qL5"), ("nohook_t64n8c16", "nohook", "t64n8c16")):
        for seed in (1, 2):
            out.append({"run": f"{arm}_seed{seed}", "arm": arm, "recipe": recipe, "shape": shape, "seed": seed,
                        "stage": 1})
    # interleave so both slots get one of each arm early: seed 1 of every arm, then seed 2
    return sorted(out, key=lambda r: (r["seed"], ["base_t64n8c16", "base_qL5", "base_qS4",
                                                  "nohook_t64n8c16"].index(r["arm"])))


def plan_stage1b(done=("base_t64n8c16_seed1",)):
    """Amendment 4: stage 1 with seed 3 for every arm; the runs not yet started, seed by seed."""
    out = []
    arms = (("base_t64n8c16", "base", "t64n8c16"), ("base_qL5", "base", "qL5"), ("base_qS4", "base", "qS4"),
            ("nohook_t64n8c16", "nohook", "t64n8c16"))
    for seed in (1, 2, 3):
        for arm, recipe, shape in arms:
            name = f"{arm}_seed{seed}"
            if name not in done:
                out.append({"run": name, "arm": arm, "recipe": recipe, "shape": shape, "seed": seed, "stage": 1})
    return out


def read_cap(path, default):
    try:
        return max(0, int(Path(path).read_text().split()[0]))
    except (OSError, ValueError, IndexError):
        return default


def run_one(r, out_root, extra):
    out = out_root / r["run"]
    for attempt in range(1, 4):
        if (out / "done.json").exists():
            return True
        if out.exists() and any(out.iterdir()):
            k = 1
            while (out_root / f"{r['run']}.failed{k}").exists():
                k += 1
            shutil.move(str(out), str(out_root / f"{r['run']}.failed{k}"))
        log = out_root / "logs" / f"{r['run']}.attempt{attempt}.log"
        cmd = ["env", f"HEAVY_SLOTS={HEAVY_SLOTS}", str(HERD / "heavy.sh"), "timeout", str(TIMEOUT_S), *command(r, out),
               *extra]
        explog({"kind": "run_start", "arm": r["arm"], "run": r["run"], "seed": r["seed"], "stage": r["stage"],
                "attempt": attempt, "commit": git_head(), "cmd": " ".join(cmd[3:])})
        t0 = time.time()
        with open(log, "w") as f:
            p = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=f, stderr=subprocess.STDOUT,
                               env=dict(os.environ, TMPDIR=TMP))
        ok = (out / "done.json").exists()
        rec = {"kind": "run_end", "arm": r["arm"], "run": r["run"], "seed": r["seed"], "stage": r["stage"],
               "attempt": attempt, "rc": p.returncode, "ok": ok, "wall_clock_s": round(time.time() - t0),
               "slot_hours": round((time.time() - t0) / 3600, 3), "checkpoint": str(out / "final.pt")}
        if ok:
            d = json.loads((out / "done.json").read_text())
            rec.update(steps=d["steps"], vtsp_bytes=d["vtsp_bytes"], vtsp_sha256=d["vtsp_sha256"])
        explog(rec)
        if ok:
            return True
    return False


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] == "plan":
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--stage", required=True, choices=("1", "1b"))
        ap.add_argument("--out", required=True)
        a = ap.parse_args(argv)
        rows = plan_stage1() if a.stage == "1" else plan_stage1b() if a.stage == "1b" else []
        Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{a.out}: {len(rows)} runs")
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("plan")
    ap.add_argument("--max-par", type=int, default=2)
    ap.add_argument("--cap-file", help="read the number of runs at once from this file before each launch")
    ap.add_argument("--runs")
    ap.add_argument("--extra", default="")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args(argv)
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=REPO, capture_output=True,
                           text=True).stdout.strip()
    if dirty and not a.allow_dirty:
        raise SystemExit(f"the checkout has modified tracked files: commit first\n{dirty}")
    out_root = Path(a.out)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    Path(TMP).mkdir(parents=True, exist_ok=True)
    rows = [json.loads(x) for x in Path(a.plan).read_text().splitlines() if x.strip()]
    if a.runs:
        want = set(a.runs.split(","))
        rows = [r for r in rows if r["run"] in want]
    extra = a.extra.split() if a.extra else []
    import threading
    results, threads, lock = [], [], threading.Lock()

    def work(r):
        ok = run_one(r, out_root, extra)
        with lock:
            results.append((r["run"], ok))

    for r in rows:
        while True:
            cap = read_cap(a.cap_file, a.max_par) if a.cap_file else a.max_par
            threads = [t for t in threads if t.is_alive()]
            if len(threads) < cap:
                break
            time.sleep(30)
        t = threading.Thread(target=work, args=(r,), daemon=False)
        t.start()
        threads.append(t)
        time.sleep(5)
    for t in threads:
        t.join()
    for name, ok in results:
        print(name, "OK" if ok else "FAILED")
    print("TEXTSPOT6_SWEEP_DONE", sum(ok for _, ok in results), "of", len(results))


if __name__ == "__main__":
    main()
