#!/usr/bin/env python3
"""Summarise probe.sh's speed probes and smoke runs, and estimate the sweep's wall time.

  probe_summary.py OUT [--out JSON] [--par 5]

Per shape (at its scale, seed 1): the probe's ms per step (mean, median, p90 over 200 steps; the time spent waiting
for the loader), the smoke run's wall time (startup, 300 steps, QAT calibration at step 240, export) and its ms per
step before and after QAT (from its log, every 20 steps). A run's estimated wall time is the smoke run's plus 29,700
steps at the probe's mean ms per step; the sweep's is the list schedule sweep.sh runs (its default order, the next
run starting when one of --par slots frees). Estimates, from probes run --par at a time on a shared box.
"""
import argparse
import heapq
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "train"))
import textdet_model as TM  # noqa: E402

STEPS = 30000


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--out", dest="json_out")
    ap.add_argument("--par", type=int, default=5)
    args = ap.parse_args()
    out = Path(args.out)
    fb = json.loads((ROOT / "eval" / "textspot_protocol.json").read_text())["candidates"]["family_b"]
    rows, est = [], {}
    for scale, shapes in fb["shapes"].items():
        for shape in shapes:
            name = f"{scale}_{shape}_seed1"
            pr = json.loads((out / "probe" / name / "probe.json").read_text())
            sm_dir = out / "smoke" / name
            done = json.loads((sm_dir / "done.json").read_text())
            log = [json.loads(x) for x in (sm_dir / "log.jsonl").read_text().splitlines()]
            t_run = done["wall_seconds"] + (STEPS - 300) * pr["ms_per_step"] / 1000
            for n in fb["seeds"]:
                est[f"{scale}_{shape}_seed{n}"] = t_run
            rows.append({"run": name, "scale": scale, "shape": shape,
                         "macs_window_256x512": TM.cost(TM.SHAPES[shape], 512, 256)["macs"],
                         "probe_ms_per_step": pr["ms_per_step"], "probe_ms_median": pr["ms_per_step_median"],
                         "probe_ms_p90": pr["ms_per_step_p90"], "probe_ms_waiting_for_data": pr["ms_waiting_for_data"],
                         "smoke_wall_s": done["wall_seconds"], "smoke_startup_s": round(done["wall_seconds"] -
                                                                                         done["seconds"], 1),
                         "smoke_ms_float_20_240": log_ms(log, 20, 240), "smoke_ms_qat_260_300": log_ms(log, 260, 300),
                         "smoke_final_loss": {k: log[-1][k] for k in ("k_bce", "dice", "o", "a") if k in log[-1]},
                         "smoke_vtsp_bytes": done["vtsp_bytes"], "est_run_s": round(t_run),
                         "est_run_h": round(t_run / 3600, 2)})
    order = sorted(est, key=lambda r: (-TM.cost(TM.SHAPES[r.split("_", 1)[1].rsplit("_seed", 1)[0]], 512, 256)["macs"],
                                       r.split("_", 1)[0], r.split("_", 1)[1].rsplit("_seed", 1)[0],
                                       int(r.rsplit("_seed", 1)[1])))
    span, plan = schedule(order, est, args.par)
    lpt_span, _ = schedule(sorted(est, key=lambda r: -est[r]), est, args.par)
    doc = {"kind": "textspot-train-probe", "what": "speed probes (200 steps after one warm-up step) and 300-step "
           "smoke runs of every family B shape at its scale, seed 1, on #97's shards (data/manifests/"
           "textdet_shards_v1.json: its synthetic regions stood in for the full-frame shards); not runs of the sweep",
           "threads": 5, "concurrency": args.par, "rows": rows,
           "sweep_estimate": {"runs": len(est), "par": args.par, "order": order, "makespan_s": round(span),
                              "makespan_h": round(span / 3600, 2), "lpt_makespan_h": round(lpt_span / 3600, 2),
                              "slot_hours": round(sum(est.values()) / 3600, 2), "plan": plan},
           "label": "estimates: measured ms per step x 30,000 on a shared 64-core box; the QAT phase assumed as fast "
                    "as the float phase except as the smoke runs measured"}
    text = json.dumps(doc, indent=1) + "\n"
    if args.json_out:
        Path(args.json_out).write_text(text)
    for r in rows:
        print(f"{r['run']:18s} probe {r['probe_ms_per_step']:6.1f} ms/step (median {r['probe_ms_median']:6.1f}, "
              f"data wait {r['probe_ms_waiting_for_data']:5.1f})  smoke {r['smoke_wall_s']:6.1f} s "
              f"(float {r['smoke_ms_float_20_240']}, qat {r['smoke_ms_qat_260_300']} ms/step)  est run "
              f"{r['est_run_h']:.2f} h")
    print(f"sweep: {len(est)} runs, {args.par} at once: {span / 3600:.2f} h in sweep.sh's order "
          f"(LPT {lpt_span / 3600:.2f} h; {sum(est.values()) / 3600:.1f} slot-hours)")


if __name__ == "__main__":
    main()
