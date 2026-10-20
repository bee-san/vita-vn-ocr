#!/usr/bin/env python3
"""Summarise probe.sh's speed probes and smoke runs, and estimate stage 1's wall time at 5 and 6 runs at once.

  probe_summary.py OUT [--out JSON] [--par 5 6]

Per probe (seed 1): ms per step over 200 steps after one warm-up step (mean, median, p90; the time spent waiting for
the loader) and the startup. Per smoke run (300 steps on s4_c24n4, QAT from step 240): its wall time, ms per step
before and after QAT (its log, every 20 steps), its final losses, the blob's bytes and check_run.py's verdict.

A stage-1 run's estimate: overhead + steps x ms, with ms = the probe's mean ms per step for its shape and recipe
(lab and r0: data's probe of the shape; long on s4p_c8: data's s4p_c8 plus the ema - data difference on
s4_c24n4), the last 20 % of the steps (QAT) at the smoke run's QAT / float ratio of its recipe, and overhead = the
recipe's smoke wall time minus its own 300 steps (240 float and 60 QAT steps at its logged ms per step): startup,
the shard sha256 checks, calibration, BatchNorm re-estimation, export. The stage's wall time is the list schedule
sweep.sh runs (textspot2_plan.py's order: longest first; the next run starts when one of --par slots frees).
Estimates, from probes run one at a time on a shared box.
"""
import argparse
import heapq
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(HERE))
import textspot2_plan as PL  # noqa: E402

PROBES = [("data", s) for s in ("s4_c24n4", "s4p_c8", "s4p_c6n4", "s4p_c8n4", "s4p_c10n4", "s4p_c12", "s4_c32n3")] + \
    [(r, "s4_c24n4") for r in ("ema", "cls", "long", "kd")]
SMOKES = [(r, "s4_c24n4") for r in ("lab", "data", "ema", "cls", "long", "kd", "r0")]
QAT_FRAC = 0.8


def jsonl(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def log_ms(log, a, b):
    sec = {r["step"]: r["sec"] for r in log if "sec" in r}
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
    return max(e for _, _, e in out), out


def run_name(recipe, shape):
    return f"{recipe}_h2_{shape}_seed{3 if recipe == 'r0' else 1}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--out", dest="json_out")
    ap.add_argument("--par", type=int, nargs="+", default=[5, 6])
    args = ap.parse_args()
    out = Path(args.out)
    probes, smokes = {}, {}
    for recipe, shape in PROBES:
        p = json.loads((out / "probe" / run_name(recipe, shape) / "probe.json").read_text())
        probes[(recipe, shape)] = {"run": run_name(recipe, shape), "recipe": recipe, "shape": shape,
                                   "macs_window_256x512": PL.macs(shape), **{k: p[k] for k in (
                                       "ms_per_step", "ms_per_step_median", "ms_per_step_p90",
                                       "ms_waiting_for_data", "warmup_ms", "startup_s", "extras")}}
    log_text = (out / "probe.log").read_text() if (out / "probe.log").exists() else ""
    for recipe, shape in SMOKES:
        name = run_name(recipe, shape)
        d = out / "smoke" / name
        done = json.loads((d / "done.json").read_text())
        log = jsonl(d / "log.jsonl")
        steps = [r for r in log if "lr" in r]
        fl, qt = log_ms(log, 20, 240), log_ms(log, 260, 300)
        last = steps[-1]
        ratio = round(qt / fl, 3) if fl and qt else None
        overhead = done["wall_seconds"] - ((240 * fl + 60 * qt) / 1000 if fl and qt else 0.0)
        smokes[recipe] = {"run": name, "wall_s": done["wall_seconds"], "train_s": done["seconds"],
                          "startup_s": round(done["wall_seconds"] - done["seconds"], 1),
                          "ms_float_20_240": fl, "ms_qat_260_300": qt, "qat_ratio": ratio,
                          "overhead_s": round(max(0.0, overhead), 1),
                          "final": {k: last[k] for k in ("k_bce", "dice", "o", "a", "kd_k", "kd_o", "kd_a", "loss")
                                    if k in last},
                          "ema_swap": any(r.get("ema_swap") for r in log), "qat": any(r.get("qat") for r in log),
                          "vtsp_bytes": done["vtsp_bytes"], "vtsp_sha256": done["vtsp_sha256"],
                          "check_ok": f"CHECK_OK {name} " in log_text}
    ema_extra = probes[("ema", "s4_c24n4")]["ms_per_step"] - probes[("data", "s4_c24n4")]["ms_per_step"]

    def ms_of(r):
        rec, shape = r["recipe"], r["shape"]
        if (rec, shape) in probes:
            return probes[(rec, shape)]["ms_per_step"]
        if rec == "long":
            return probes[("data", shape)]["ms_per_step"] + max(0.0, ema_extra)
        return probes[("data", shape)]["ms_per_step"]       # lab, r0: data's network and batch

    est = {}
    runs = PL.stage1()
    rows = []
    for r in runs:
        sm = smokes[r["recipe"] if r["recipe"] in smokes else "data"]
        ms = ms_of(r)
        ratio = sm["qat_ratio"] or 1.0
        overhead = sm["overhead_s"]
        t = overhead + r["steps"] * ms / 1000 * (QAT_FRAC + (1 - QAT_FRAC) * ratio)
        est[r["run"]] = t
        rows.append({"run": r["run"], "steps": r["steps"], "ms_per_step": round(ms, 2), "qat_ratio": ratio,
                     "overhead_s": round(overhead, 1), "est_s": round(t), "est_h": round(t / 3600, 2)})
    order = [r["run"] for r in PL.order(runs, est)]
    sched = {}
    for par in args.par:
        span, plan = schedule(order, est, par)
        sched[str(par)] = {"makespan_s": round(span), "makespan_h": round(span / 3600, 2), "plan": plan}
    doc = {"kind": "textspot2-train-probe",
           "what": "speed probes (200 steps after one warm-up step) of every stage-1 shape under data and of s4_c24n4 "
                   "under ema, cls, long and kd, and 300-step smoke runs of every recipe on s4_c24n4 (r0 included), "
                   "seed 1 (r0: 3), on textspot's shards (data/manifests/textspot_shards_v1.json + "
                   "textdet_shards_v1.json; the textspot2 shards were not pinned yet); not runs of the sweep",
           "threads": 5, "concurrency": 1, "probes": list(probes.values()), "smokes": smokes,
           "stage1_estimate": {"runs": len(runs), "order": order, "rows": rows,
                               "est_run_s": {k: round(v) for k, v in est.items()},
                               "slot_hours": round(sum(est.values()) / 3600, 2), "schedule": sched},
           "label": "estimates: measured ms per step x steps on a shared 64-core box, one probe at a time; QAT at the "
                    "smoke runs' QAT / float ratio; 5-6 runs at once may run slower per step (textspot's sweep: "
                    "4.79 h measured against 4.52 h estimated, x 1.06)"}
    text = json.dumps(doc, indent=1) + "\n"
    if args.json_out:
        Path(args.json_out).write_text(text)
    for p in probes.values():
        print(f"probe {p['run']:24s} {p['ms_per_step']:7.1f} ms/step (median {p['ms_per_step_median']:7.1f}, "
              f"p90 {p['ms_per_step_p90']:7.1f}, data wait {p['ms_waiting_for_data']:5.1f}; startup "
              f"{p['startup_s']:5.1f} s)")
    for k, s in smokes.items():
        print(f"smoke {s['run']:24s} wall {s['wall_s']:6.1f} s (startup {s['startup_s']:5.1f}; float "
              f"{s['ms_float_20_240']}, qat {s['ms_qat_260_300']} ms/step) {s['vtsp_bytes']} B check "
              f"{'OK' if s['check_ok'] else 'MISSING'} final {s['final']}")
    for par, s in sched.items():
        print(f"stage 1: {len(runs)} runs, {par} at once: {s['makespan_h']:.2f} h "
              f"({sum(est.values()) / 3600:.1f} slot-hours)")


if __name__ == "__main__":
    main()
