#!/usr/bin/env python3
"""Summarise a textspot3 training sweep (sweep.sh's OUT directory) for the report.

  runs_summary.py OUT PLAN.jsonl [--json FILE] [--md FILE]
  runs_summary.py OUT PLAN.jsonl --record FILE --stage NAME [--md FILE]
                  add (or replace) one stage in the committed runs record FILE (eval/textspot3_train_runs_*.json):
                  the stage's sweep (OUT as "$HERD/...", the commit, MAX_PAR, the plan and its sha256, started and
                  SWEEP_DONE, all read from OUT/sweep.log) and this summary of its runs

Per run of the plan (textspot3_plan.py stage1 or stage2): started and finished (UTC), wall time, attempts (sweep.sh's
moved failed attempts and crashes.jsonl), the loss curve (the trainer's logged "loss", the total it optimised: at
steps 200, 2,000, 10,000, 20,000, the last float record, the first record after QAT starts and the last; the parts
averaged over the last five records; the EMA swap's step), clean's counts (clean runs) and the blobs' bytes and
sha256. Across the finished runs it checks that every run recorded the same commit, and per family the same code
(the trainer's files), manifests, skip record and protocols. Reads only the runs' own files; scores nothing.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _plan():
    """textspot3_plan.py of this directory, without putting the directory on sys.path (textspot2's sweep scripts
    share this file's name)."""
    if "textspot3_plan" in sys.modules:
        return sys.modules["textspot3_plan"]
    spec = importlib.util.spec_from_file_location("textspot3_plan", HERE / "textspot3_plan.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["textspot3_plan"] = mod
    spec.loader.exec_module(mod)
    return mod


PL = _plan()

HERD = os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr")
START = re.compile(r"^(\S+) sweep start: (.*), MAX_PAR (\d+), head ([0-9a-f]{40}), plan (\S+) \(([0-9a-f]{64})\)")
DONE = re.compile(r"^(\S+) SWEEP_DONE ")
PARTS = ("k_bce", "dice", "o", "a")
KD_PARTS = ("kd_k", "kd_o", "kd_a")
AT = (200, 2000, 10000, 20000)
REPRO = "eval/reproducibility/textspot3-20261021"


def jsonl(path):
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def curve(log, qat_start):
    recs = [r for r in log if "lr" in r and "loss" in r]
    by = {r["step"]: r for r in recs}
    out = {f"total_{s}": round(by[s]["loss"], 4) for s in AT if s in by}
    pre = [r for r in recs if r["step"] <= qat_start]
    post = [r for r in recs if r["step"] > qat_start]
    if pre:
        out["total_last_float"] = round(pre[-1]["loss"], 4)
        out["step_last_float"] = pre[-1]["step"]
    if post:
        out["total_first_qat"] = round(post[0]["loss"], 4)
        out["step_first_qat"] = post[0]["step"]
    if recs:
        out["total_final"] = round(recs[-1]["loss"], 4)
        out["step_final"] = recs[-1]["step"]
        tail = recs[-5:]
        keys = PARTS + tuple(k for k in KD_PARTS if all(k in r for r in tail))
        out["parts_last5"] = {k: round(sum(r[k] for r in tail) / len(tail), 5) for k in keys}
        out["ms_per_step"] = round(1000 * recs[-1]["sec"] / recs[-1]["step"], 1)
    swap = [r for r in log if r.get("ema_swap")]
    if swap:
        out["ema_swap_step"], out["ema_swap_updates"] = swap[0]["step"], swap[0]["ema_steps"]
    avg = [r for r in log if r.get("qat_average") == "the run's model"]
    if avg:
        out["qat_average_updates"] = avg[0]["qat_average_steps"]
    return out


def summarise(out_dir, plan):
    out_dir = Path(out_dir)
    crashes = jsonl(out_dir / "crashes.jsonl")
    restarts = jsonl(out_dir / "restarts.jsonl")
    rows = []
    for spec in PL.load(plan):
        name = spec["run"]
        d = out_dir / name
        failed = sorted(p.name for p in out_dir.glob(f"{name}.failed*") if p.is_dir())
        row = {"run": name, "recipe": spec["recipe"], "family": spec["family"], "shape": spec["shape"],
               "seed": spec["seed"], "state": "missing", "failed_attempts": failed,
               "crashes": [c for c in crashes if c.get("run") == name],
               "restarts": [r for r in restarts if r.get("run") == name]}
        if (d / "run.json").exists():
            info = json.loads((d / "run.json").read_text())
            sk = info.get("skip") or {}
            code = {k: v for k, v in info["code_sha256"].items()
                    if k.startswith(("train/textspot", "train/textdet", "synth/textspot", "synth/textdet"))}
            clean = (info.get("settings") or {}).get("clean")
            row.update(state="running", started=info["started"], steps=info["steps"], qat_start=info["qat_start"],
                       extras=info.get("extras"), git=info.get("git"), code_sha256=code,
                       manifests_sha256=info["manifests_sha256"],
                       protocols_sha256=[info.get(k) for k in ("protocol_sha256", "textspot2_protocol_sha256",
                                                               "textspot_protocol_sha256")],
                       skip={"dropped_total": sk.get("dropped_total"),
                             "lists": [{k: v for k, v in r.items() if k.endswith("sha256") or k.startswith("skip")}
                                       for r in sk.get("lists") or []]},
                       teacher=((info.get("settings") or {}).get("kd") or {}).get("teacher"),
                       curve=curve(jsonl(d / "log.jsonl"), info["qat_start"]))
            if clean:
                row["clean"] = clean["total"]
        if (d / "done.json").exists():
            done = json.loads((d / "done.json").read_text())
            row.update(state="finished", finished=done["finished"], wall_seconds=done["wall_seconds"],
                       train_seconds=done["seconds"], vtd_bytes=done.get("vtd_bytes"),
                       vtd_sha256=done.get("vtd_sha256"), vtsp_bytes=done["vtsp_bytes"],
                       vtsp_sha256=done["vtsp_sha256"], vtsp_version=done.get("vtsp_version"),
                       pt_sha256=done["pt_sha256"])
        rows.append(row)
    fin = [r for r in rows if r["state"] == "finished"]
    same = {"git": len({json.dumps(r.get("git"), sort_keys=True) for r in fin}) <= 1}
    for fam in sorted({r["family"] for r in fin}):
        part = [r for r in fin if r["family"] == fam]
        for key in ("code_sha256", "manifests_sha256", "skip", "protocols_sha256"):
            same[f"{key}[{fam}]"] = len({json.dumps(r.get(key), sort_keys=True) for r in part}) <= 1
    return {"kind": "textspot3-train-runs", "plan": str(plan), "runs": rows, "finished": len(fin), "of": len(rows),
            "consistent": same, "crashes": len(crashes), "restarts": len(restarts)}


def placeholder(path, herd=HERD):
    """A path under $HERD written as "$HERD/..." (no host paths in a committed record)."""
    path, herd = str(path), str(herd).rstrip("/")
    return "$HERD" + path[len(herd):] if path == herd or path.startswith(herd + "/") else path


def sweep_record(out_dir, plan, herd=HERD):
    """The sweeps of this plan in OUT/sweep.log: every start line naming the plan's file (its UTC, stage label,
    MAX_PAR, head and plan sha256) and the SWEEP_DONE lines that followed each, before the next start."""
    log = Path(out_dir) / "sweep.log"
    starts, current = [], None
    for line in (log.read_text().splitlines() if log.exists() else []):
        m = START.match(line)
        if m:
            current = None
            if Path(m.group(5)).name == Path(plan).name:
                current = {"started": m.group(1), "stage": m.group(2), "max_par": int(m.group(3)),
                           "commit": m.group(4), "plan_sha256": m.group(6), "sweep_done": None}
                starts.append(current)
            continue
        d = DONE.match(line)
        if d and current is not None:
            current["sweep_done"] = d.group(1)
    if not starts:
        return {"out": placeholder(Path(out_dir).resolve(), herd), "launches": []}
    first, last = starts[0], starts[-1]
    return {"out": placeholder(Path(out_dir).resolve(), herd), "stage": first["stage"], "commit": first["commit"],
            "commits": sorted({s["commit"] for s in starts}), "max_par": first["max_par"],
            "plan": f"{REPRO}/{Path(plan).name}", "plan_sha256": hashlib.sha256(Path(plan).read_bytes()).hexdigest(),
            "plan_sha256_at_start": first["plan_sha256"], "started": first["started"],
            "sweep_done": last["sweep_done"], "launches": starts}


def record(path, stage, out_dir, plan, herd=HERD):
    """Add or replace one stage in the runs record at path (created when missing)."""
    p = Path(path)
    doc = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {
        "kind": "textspot3-train-runs", "schema": 1, "issue": "https://github.com/bee-san/vita-vn-ocr/issues/108",
        "what": "the training runs of docs/textspot3.md (train/textspot3_train.py) through "
                f"{REPRO}/sweep.sh, summarised by runs_summary.py from each run's own run.json, log.jsonl and "
                "done.json: started, finished, wall time, attempts, the training-loss curve and the blobs' bytes and "
                "sha256. Training losses only: nothing here is a score, and no run was selected or stopped on any "
                "figure", "sweeps": {}, "stages": {}}
    s = summarise(out_dir, plan)
    s["plan"] = f"{REPRO}/{Path(plan).name}"
    doc["sweeps"][stage] = sweep_record(out_dir, plan, herd)
    doc["stages"][stage] = s
    p.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    return doc


def markdown(doc):
    lines = ["| run | started | finished | wall h | attempts | loss 200 / last float / first QAT / final | final "
             "k_bce / dice / o / a (last 5) | VTSP B | final.vtsp sha256 |", "|---|---|---|---:|---:|---|---|---:|---|"]
    for r in doc["runs"]:
        c = r.get("curve", {})
        p = c.get("parts_last5", {})
        loss = " / ".join(f"{c[k]:.3f}" if k in c else "-" for k in ("total_200", "total_last_float",
                                                                      "total_first_qat", "total_final"))
        parts = " / ".join(f"{p[k]:.3f}" for k in PARTS) if p else "-"
        if p and "kd_k" in p:
            parts += " (kd " + " / ".join(f"{p[k]:.3f}" for k in KD_PARTS) + ")"
        wall = f"{r['wall_seconds'] / 3600:.2f}" if "wall_seconds" in r else "-"
        lines.append(f"| {r['run']} | {r.get('started', '-')} | {r.get('finished', r['state'])} | {wall} | "
                     f"{1 + len(r['failed_attempts'])} | {loss} | {parts} | {r.get('vtsp_bytes', '-')} | "
                     f"`{r.get('vtsp_sha256', '-')}` |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("plan")
    ap.add_argument("--json")
    ap.add_argument("--md")
    ap.add_argument("--record", help="the committed runs record to add this stage to")
    ap.add_argument("--stage", help="--record: the stage's key (stage1_T, stage1_M, stage2)")
    args = ap.parse_args(argv)
    if args.record and not args.stage:
        ap.error("--record needs --stage")
    if args.record:
        record(args.record, args.stage, args.out, args.plan)
    doc = summarise(args.out, args.plan)
    if args.json:
        Path(args.json).write_text(json.dumps(doc, indent=1) + "\n")
    md = markdown(doc)
    if args.md:
        Path(args.md).write_text(md)
    print(md)
    print(f"finished {doc['finished']}/{doc['of']}; crashes {doc['crashes']}, restarts {doc['restarts']}; "
          f"consistent {doc['consistent']}")


if __name__ == "__main__":
    main()
