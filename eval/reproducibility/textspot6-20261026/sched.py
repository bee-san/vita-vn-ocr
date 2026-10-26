#!/usr/bin/env python3
"""textspot6's scheduler (docs/textspot6.md, amendment 8): ONE process runs the label's heavy jobs within its slot
share (vocr6/slots.txt), training and scoring together, so that the share is never exceeded and never idle.

  sched.py --runs RUNS --plan PLAN.jsonl --cap-file CAP [--adopt NAME:PID ...] [--pre CMD ...]

- Jobs: every pending scoring job first (a finished run, i.e. one with done.json, that has no OUT/<run>/SCORED or
  FAILED marker; the --pre commands before them), then the training runs of PLAN in order (sweep.py's command, through
  heavy.sh, restarted from scratch with the same seed after a crash, at most 3 attempts).
- At most CAP jobs at once (an integer read from CAP before each launch). An adopted run (--adopt NAME:PID: a trainer
  started by an earlier sweep) counts as a job until its process exits; its run_end is then logged.
- The runtime helper's window: while RUNS/helper.reserve exists, the scheduler keeps one slot of the share free for
  vocr6-spotter-runtime and writes RUNS/helper_go (removed otherwise). The helper runs one heavy job at a time, and
  only while RUNS/helper_go exists. helper.reserve is removed when vocr6-spotter-runtime.status starts with
  RUNTIME_READY.
- Every start and end goes to the shared experiment log (vocr6/experiments.jsonl, flock).
Extra jobs: a line appended to RUNS/extra_jobs.txt (a bash command) is run once, before the scoring and training
jobs, as a --pre command is (lines already taken are counted in RUNS/extra_jobs.taken).
Stop: touch RUNS/sched.stop (running jobs finish; nothing new starts).
"""
import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sweep as SW  # noqa: E402

HERD = SW.HERD
A = SW.A
EXPLOG = SW.EXPLOG
HELPER_STATUS = Path("/local/home/skerraut/herd-synth3/vocr6-spotter-runtime.status")
SCORE = HERE / "score_run.sh"


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def explog(rec):
    SW.explog(rec)


def alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:
        st = Path(f"/proc/{pid}/stat").read_text().split()
        return st[2] != "Z"
    except OSError:
        return False


def read_cap(path, default=0):
    return SW.read_cap(path, default)


class Job:
    def __init__(self, kind, name, cmd=None, run=None, pid=None):
        self.kind, self.name, self.cmd, self.run, self.pid = kind, name, cmd, run, pid
        self.proc, self.t0, self.attempt, self.log = None, time.time(), 1, None

    def done(self):
        if self.proc is not None:
            return self.proc.poll() is not None
        return not alive(self.pid)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--cap-file", required=True)
    ap.add_argument("--out", default=str(A / "vocr6-spotter" / "eval"))
    ap.add_argument("--adopt", nargs="*", default=[], help="NAME:PID of a running trainer of an earlier sweep")
    ap.add_argument("--pre", nargs="*", default=[], help="scoring commands to run first (strings, run by bash)")
    ap.add_argument("--pagepath", required=True)
    a = ap.parse_args(argv)
    runs, out = Path(a.runs), Path(a.out)
    (runs / "logs").mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(parents=True, exist_ok=True)
    plan = [json.loads(x) for x in Path(a.plan).read_text().splitlines() if x.strip()]
    plan_names = {r["run"] for r in plan}
    meta = {r["run"]: r for r in plan}
    active = []
    for spec in a.adopt:
        name, pid = spec.rsplit(":", 1)
        active.append(Job("adopted", name, pid=int(pid)))
        explog({"kind": "note", "what": f"sched.py adopts the running trainer {name} (pid {pid}) of an earlier sweep"})
    pre = list(a.pre)
    started = {j.name for j in active}
    failed_score = set()
    attempts = {}
    heavy = ["env", f"HEAVY_SLOTS={SW.HEAVY_SLOTS}", str(HERD / "heavy.sh")]
    print(f"{utc()} sched start: {len(plan)} planned runs, {len(active)} adopted, {len(pre)} pre jobs", flush=True)
    while True:
        # reap
        for j in list(active):
            if not j.done():
                continue
            active.remove(j)
            if j.kind in ("train", "adopted"):
                d = runs / j.name
                ok = (d / "done.json").exists()
                r = meta.get(j.name, {"arm": j.name.rsplit("_seed", 1)[0], "seed": int(j.name.rsplit("seed", 1)[1]),
                                      "stage": 1})
                rec = {"kind": "run_end", "arm": r["arm"], "run": j.name, "seed": r["seed"], "stage": r.get("stage", 1),
                       "attempt": j.attempt, "ok": ok, "checkpoint": str(d / "final.pt"),
                       "via": "sched.py" + (" (adopted)" if j.kind == "adopted" else "")}
                if ok:
                    dj = json.loads((d / "done.json").read_text())
                    rec.update(steps=dj["steps"], vtsp_bytes=dj["vtsp_bytes"], vtsp_sha256=dj["vtsp_sha256"],
                               wall_clock_s=dj["wall_seconds"], slot_hours=round(dj["wall_seconds"] / 3600, 3))
                explog(rec)
                print(f"{utc()} {j.kind} {j.name} ended ok={ok}", flush=True)
                if not ok and j.kind == "train":
                    attempts[j.name] = attempts.get(j.name, 1) + 1
                    if attempts[j.name] <= 3:
                        started.discard(j.name)
            elif j.kind == "score":
                ok = j.proc.returncode == 0
                if ok:
                    (out / j.name).mkdir(parents=True, exist_ok=True)
                    (out / j.name / "SCORED").write_text(utc() + "\n")
                else:
                    failed_score.add(j.name)
                    (out / j.name).mkdir(parents=True, exist_ok=True)
                    (out / j.name / "FAILED").write_text(utc() + "\n")
                print(f"{utc()} score {j.name} ended ok={ok}", flush=True)
            elif j.kind == "pre":
                print(f"{utc()} pre job ended rc={j.proc.returncode}: {j.name}", flush=True)
        # the helper's window
        reserve = runs / "helper.reserve"
        if reserve.exists() and HELPER_STATUS.exists() and \
                HELPER_STATUS.read_text().strip().startswith("RUNTIME_READY"):
            reserve.unlink()
            explog({"kind": "note", "what": "sched.py: the runtime helper reported RUNTIME_READY; its slot returns"})
        cap = read_cap(a.cap_file, 0)
        mine = cap - (1 if reserve.exists() else 0)
        go = runs / "helper_go"
        if reserve.exists() and len(active) <= cap - 1:
            go.write_text(utc() + "\n")
        elif go.exists():
            go.unlink()
        stop = (runs / "sched.stop").exists()
        xf, xt = runs / "extra_jobs.txt", runs / "extra_jobs.taken"
        if xf.exists():
            lines = [x for x in xf.read_text().splitlines() if x.strip()]
            taken = int(xt.read_text().strip() or 0) if xt.exists() else 0
            if len(lines) > taken:
                pre.extend(lines[taken:])
                xt.write_text(f"{len(lines)}\n")
        # launch
        while not stop and len(active) < mine:
            job = None
            if pre:
                cmd = pre.pop(0)
                job = Job("pre", cmd[:80], cmd=["bash", "-c", cmd])
            if job is None:
                finished = sorted((p.parent for p in runs.glob("*/done.json")),
                                  key=lambda p: (p / "done.json").stat().st_mtime)
                for d in finished:
                    name = d.name
                    if (out / name / "SCORED").exists() or (out / name / "FAILED").exists() or name in failed_score:
                        continue
                    if any(j.kind == "score" and j.name == name for j in active):
                        continue
                    log = out / "logs" / f"score_{name}.log"
                    cmd = [*heavy, "timeout", "10800", "env", f"PAGEPATH={a.pagepath}", f"OUT={out}", str(SCORE),
                           str(d), name]
                    job = Job("score", name, cmd=cmd)
                    job.log = log
                    break
            if job is None:
                for r in plan:
                    if r["run"] in started:
                        continue
                    d = runs / r["run"]
                    if (d / "done.json").exists():
                        started.add(r["run"])
                        continue
                    if d.exists() and any(d.iterdir()):
                        k = 1
                        while (runs / f"{r['run']}.failed{k}").exists():
                            k += 1
                        os.rename(d, runs / f"{r['run']}.failed{k}")
                    cmd = [*heavy, "timeout", str(SW.TIMEOUT_S), *SW.command(r, d)]
                    job = Job("train", r["run"], cmd=cmd, run=r)
                    job.attempt = attempts.get(r["run"], 1)
                    job.log = runs / "logs" / f"{r['run']}.attempt{job.attempt}.log"
                    started.add(r["run"])
                    explog({"kind": "run_start", "arm": r["arm"], "run": r["run"], "seed": r["seed"],
                            "stage": r["stage"], "attempt": job.attempt, "commit": SW.git_head(),
                            "cmd": " ".join(shlex.quote(c) for c in cmd[3:]), "via": "sched.py"})
                    break
            if job is None:
                break
            fh = open(job.log, "w") if job.log else open(out / "logs" / "pre.log", "a")
            job.proc = subprocess.Popen(job.cmd, stdin=subprocess.DEVNULL, stdout=fh, stderr=subprocess.STDOUT,
                                        env=dict(os.environ, TMPDIR=SW.TMP))
            active.append(job)
            print(f"{utc()} start {job.kind} {job.name}", flush=True)
            time.sleep(5)
        all_trained = all((runs / r["run"] / "done.json").exists() or attempts.get(r["run"], 1) > 3 for r in plan)
        pending_scores = [d for d in runs.glob("*/done.json") if not (out / d.parent.name / "SCORED").exists()
                          and d.parent.name not in failed_score and (d.parent.name in plan_names or
                                                                     any(d.parent.name == j for j in started))]
        if (stop or (all_trained and not pending_scores and not pre)) and not active:
            break
        time.sleep(30)
    if (runs / "helper_go").exists():
        (runs / "helper_go").unlink()
    print(f"{utc()} sched finished", flush=True)


if __name__ == "__main__":
    main()
