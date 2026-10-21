#!/usr/bin/env python3
"""probe.sh's plan, and the summary of its speed probes and smoke runs with the estimate of a stage's wall time.

  probe_summary.py plan --out PLAN.jsonl [--teacher DIR] RECIPE:SHAPE ...
          the probe and smoke runs as textspot3_plan.py records (seed 1; kd with --extras none and the teacher DIR),
          so probe.sh gets the sweep's own command lines from textspot3_plan.py args
  probe_summary.py summary OUT [--out JSON] [--par 8] [--family T] [--protocol P]
          the record of OUT's probes and smokes and the estimate of the family's stage-1 runs (textspot3_plan.py
          stage1) at --par runs at once

Per probe (seed 1): ms per step over 200 steps after one warm-up step (mean, median, p90; the time spent waiting for
the loader) and the startup, for every pass probe.sh ran (OUT/probe, then OUT/probe2, ...); the estimate uses the
mean over the passes of each pass's mean ms per step. Per smoke run (300 steps, QAT from step 240): its wall time, ms
per step before and after QAT (its log, every 20 steps), its final losses, clean's counts, the blob's bytes and
check_run.py's verdict (the C driver equal to the twin).

A stage-1 run's estimate: overhead + steps x ms, with ms = the probe's mean ms per step for its shape under base
(size, clean and long on the same shape: base's, plus size's or clean's measured difference on the probed shape;
long's step is base's), the last 20 % of the steps (QAT) at the median QAT / float step ratio of the smoke runs other
than kd's, and overhead = the startup (the shard sha256 checks and the pool) plus the rest of the base smoke's
overhead (calibration, BatchNorm re-estimation, export: forward passes), scaled by the shape's step time against the
smoke's shape. The stage's wall time is the list schedule sweep.sh runs (textspot3_plan.py's order: longest first;
the next run starts when one of --par slots frees). Estimates, from probes run at most two at a time on a shared box.
"""
import argparse
import heapq
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
QAT_FRAC = 0.8


def _plan():
    """textspot3_plan.py of this directory (not by sys.path: textspot2's sweep scripts share file names)."""
    if "textspot3_plan" in sys.modules:
        return sys.modules["textspot3_plan"]
    spec = importlib.util.spec_from_file_location("textspot3_plan", HERE / "textspot3_plan.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["textspot3_plan"] = mod
    spec.loader.exec_module(mod)
    return mod


PL = _plan()


def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def log_ms(log, a, b):
    sec = {r["step"]: r["sec"] for r in log if "sec" in r and "lr" in r}
    return round(1000 * (sec[b] - sec[a]) / (b - a), 1) if a in sec and b in sec else None


def schedule(order, est, par):
    """List scheduling: runs in order, each on the first slot to free. -> (makespan s, [(run, start, end)])."""
    slots = [(0.0, k) for k in range(par)]
    heapq.heapify(slots)
    out = []
    for r in order:
        t, k = heapq.heappop(slots)
        out.append((r, round(t), round(t + est[r])))
        heapq.heappush(slots, (t + est[r], k))
    return (max(e for _, _, e in out) if out else 0), out


def probe_plan(items, teacher, reg):
    runs, seen = [], set()
    for item in items:
        recipe, _, shape = item.partition(":")
        if recipe not in ("base", "size", "clean", "long", "kd") or shape not in reg:
            raise SystemExit(f"{item!r}: recipe:shape with a textspot3 recipe and a registered shape")
        r = PL.spec(recipe, reg[shape]["family"], shape, 1, "none" if recipe == "kd" else None,
                    teacher if recipe == "kd" else None)
        if r["run"] not in seen:
            seen.add(r["run"])
            r["timeout_s"] = 5400
            runs.append(r)
    return runs


def summary(out, par=(8,), family="T", protocol=PL.PROTOCOL):
    out = Path(out)
    reg = PL.shapes(protocol)
    pass_dirs = ["probe"] + sorted((d.name for d in out.glob("probe[0-9]*") if d.is_dir()), key=lambda n: int(n[5:]))
    keys = ("ms_per_step", "ms_per_step_median", "ms_per_step_p90", "ms_waiting_for_data", "warmup_ms", "startup_s")
    probes = {}
    for d in sorted((out / "probe").iterdir()):
        if not (d / "probe.json").exists():
            continue
        name = d.name
        passes = []
        for pd in pass_dirs:
            f = out / pd / name / "probe.json"
            if f.exists():
                p = json.loads(f.read_text())
                passes.append({"pass": pd, "finished": p["finished"], **{k: p[k] for k in keys}})
        p0 = json.loads((d / "probe.json").read_text())
        mean = sum(q["ms_per_step"] for q in passes) / len(passes)
        probes[(p0["recipe"], p0["shape"])] = {
            "run": name, "recipe": p0["recipe"], "family": p0["family"], "shape": p0["shape"],
            "extras": p0["extras"], "macs_window_256x512": PL.macs(reg, p0["shape"]), "ms_per_step": round(mean, 2),
            "passes": passes, "n_passes": len(passes),
            "ms_per_step_range": [min(q["ms_per_step"] for q in passes), max(q["ms_per_step"] for q in passes)]}
    log_text = (out / "probe.log").read_text() if (out / "probe.log").exists() else ""
    smokes = {}
    for d in sorted((out / "smoke").iterdir()):
        if not (d / "done.json").exists():
            continue
        done = json.loads((d / "done.json").read_text())
        info = json.loads((d / "run.json").read_text())
        log = jsonl(d / "log.jsonl")
        steps = [r for r in log if "lr" in r]
        fl, qt = log_ms(log, 20, 240), log_ms(log, 260, 300)
        last = steps[-1]
        ratio = round(qt / fl, 3) if fl and qt else None
        overhead = done["wall_seconds"] - ((240 * fl + 60 * qt) / 1000 if fl and qt else 0.0)
        clean = (info.get("settings") or {}).get("clean")
        teacher = ((info.get("settings") or {}).get("kd") or {}).get("teacher")
        check = [ln for ln in log_text.splitlines() if ln.startswith(f"CHECK_OK {d.name} ")]
        smokes[info["recipe"]] = {
            "run": d.name, "shape": info["shape"]["name"], "family": info["family"], "extras": info["extras"],
            "wall_s": done["wall_seconds"], "train_s": done["seconds"],
            "startup_s": round(done["wall_seconds"] - done["seconds"], 1), "ms_float_20_240": fl,
            "ms_qat_260_300": qt, "qat_ratio": ratio, "overhead_s": round(max(0.0, overhead), 1),
            "final": {k: last[k] for k in ("k_bce", "dice", "o", "a", "kd_k", "kd_o", "kd_a", "loss") if k in last},
            "ema_swap": [{k: r[k] for k in ("step", "ema_steps", "bn_layers")} for r in log if r.get("ema_swap")],
            "qat": any(r.get("qat") for r in log),
            "qat_average_updates": next((r["qat_average_steps"] for r in log
                                         if r.get("qat_average") == "the run's model"), None),
            "teacher": {k: teacher[k] for k in ("name", "family", "shape", "pt_sha256")} if teacher else None,
            "clean": clean["total"] if clean else None, "vtsp_bytes": done["vtsp_bytes"],
            "vtsp_sha256": done["vtsp_sha256"], "vtsp_version": done.get("vtsp_version"),
            "check_ok": bool(check), "c_runtime_equal": bool(check) and "'c_runtime': 'equal to the twin'" in check[0]}
    ratios = sorted(s["qat_ratio"] for k, s in smokes.items() if k != "kd" and s["qat_ratio"])
    median_ratio = (ratios[len(ratios) // 2] if len(ratios) % 2 else
                    (ratios[len(ratios) // 2 - 1] + ratios[len(ratios) // 2]) / 2) if ratios else 1.0
    base_smoke = smokes.get("base")
    runs = PL.stage1(family, reg) if any(v["family"] == family for v in reg.values()) else []
    est, rows = {}, []
    if base_smoke and runs:
        ref_shape = base_smoke["shape"]
        ref_ms = probes[("base", ref_shape)]["ms_per_step"]
        extra = {x: probes[(x, ref_shape)]["ms_per_step"] - ref_ms for x in ("size", "clean")
                 if (x, ref_shape) in probes}
        startup = base_smoke["startup_s"]
        rest = max(0.0, base_smoke["overhead_s"] - startup)
        for r in runs:
            ms = probes[("base", r["shape"])]["ms_per_step"] + max(0.0, extra.get(r["recipe"], 0.0))
            over = startup + rest * ms / ref_ms
            t = over + r["steps"] * ms / 1000 * (QAT_FRAC + (1 - QAT_FRAC) * median_ratio)
            est[r["run"]] = t
            rows.append({"run": r["run"], "steps": r["steps"], "ms_per_step": round(ms, 2),
                         "qat_ratio": round(median_ratio, 3), "overhead_s": round(over, 1), "est_s": round(t),
                         "est_h": round(t / 3600, 2)})
    order = [r["run"] for r in PL.order(runs, reg, est)] if est else []
    sched = {}
    for p in par:
        if order:
            span, plan = schedule(order, est, p)
            sched[str(p)] = {"makespan_s": round(span), "makespan_h": round(span / 3600, 2), "plan": plan}
    starts = [ln for ln in log_text.splitlines() if " probe start: " in ln]
    return {"kind": "textspot3-train-probe", "family": family,
            "what": "speed probes (200 steps after one warm-up step) and 300-step smoke runs (QAT from step 240, "
                    "export, VTSP, check_run.py with the C driver) on the pinned shards (data/manifests/"
                    "textspot_shards_v1.json + textspot2_shards_v1.json) with the sweep's own command lines "
                    "(textspot3_plan.py args), seed 1; not runs of the sweep, never used",
            "head": starts[0].split(" head ")[1].split(",")[0] if starts else None, "threads": 5,
            "concurrency": "at most 2 of these jobs at once (probe.sh MAX_PAR 2), on a shared 64-core box",
            "passes": pass_dirs, "probes": list(probes.values()), "smokes": smokes,
            "estimate": {"family": family, "runs": len(runs), "order": order, "rows": rows,
                         "qat_ratio_median": round(median_ratio, 3), "est_run_s": {k: round(v) for k, v in est.items()},
                         "slot_hours": round(sum(est.values()) / 3600, 2), "schedule": sched},
            "label": "estimates: measured ms per step x steps; QAT at the smoke runs' median QAT / float step ratio; "
                     "8 runs at once may run slower per step than probes run two at a time (textspot2's stage 1 at 6 "
                     "at once: 1.10x its estimate on average, 0.95-1.30x per run)"}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("plan")
    a1.add_argument("--out", required=True)
    a1.add_argument("--teacher")
    a1.add_argument("--protocol", default=str(PL.PROTOCOL))
    a1.add_argument("items", nargs="+")
    a2 = sub.add_parser("summary")
    a2.add_argument("out")
    a2.add_argument("--out", dest="json_out")
    a2.add_argument("--par", type=int, nargs="+", default=[8])
    a2.add_argument("--family", default="T", choices=("T", "M"))
    a2.add_argument("--protocol", default=str(PL.PROTOCOL))
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        runs = probe_plan(args.items, args.teacher, PL.shapes(args.protocol))
        PL.write(runs, args.out)
        print(f"{len(runs)} probe runs -> {args.out}")
        return
    doc = summary(args.out, args.par, args.family, args.protocol)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(doc, indent=1) + "\n")
    for p in doc["probes"]:
        q = p["passes"][0]
        print(f"probe {p['run']:28s} {p['ms_per_step']:7.1f} ms/step over {p['n_passes']} pass(es) (pass 1: median "
              f"{q['ms_per_step_median']:7.1f}, p90 {q['ms_per_step_p90']:7.1f}, data wait "
              f"{q['ms_waiting_for_data']:5.1f}; startup {q['startup_s']:5.1f} s)")
    for k, s in doc["smokes"].items():
        print(f"smoke {s['run']:28s} wall {s['wall_s']:6.1f} s (startup {s['startup_s']:5.1f}; float "
              f"{s['ms_float_20_240']}, qat {s['ms_qat_260_300']} ms/step) {s['vtsp_bytes']} B check "
              f"{'OK' if s['check_ok'] else 'MISSING'} C {'=' if s['c_runtime_equal'] else '?'} final {s['final']}")
    e = doc["estimate"]
    for par, s in e["schedule"].items():
        print(f"stage 1 {e['family']}: {e['runs']} runs, {par} at once: {s['makespan_h']:.2f} h "
              f"({e['slot_hours']:.1f} slot-hours)")


if __name__ == "__main__":
    main()
