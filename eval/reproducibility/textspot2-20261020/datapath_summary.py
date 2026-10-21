#!/usr/bin/env python3
"""Summarise datapath_probe.sh: per probe, the speed (ms per step over 200 steps, the loader's wait, the startup with
every shard's sha256 check), the shards read and not read, the data rule, the skip record (expected against dropped
per manifest), the pool per source (canvases, lines, weighted lines, class lines); the d2s smoke run's wall time,
losses, blob and check. Reads only the probes' own files; scores nothing.

  datapath_summary.py OUT [--out JSON]
"""
import argparse
import json
from pathlib import Path


def summary_of(run_dir):
    info = json.loads((run_dir / "run.json").read_text())
    sk = info["skip"]
    pool = info["pool"]
    rec = {"run": info["name"], "recipe": info["recipe"], "shape": info["shape"]["name"],
           "head": info["shape"].get("head"), "extras": info["extras"], "git": info["git"],
           "manifests_sha256": info["manifests_sha256"], "shards_dirs": [Path(d).name for d in info["shards_dirs"]],
           "shards_verified": info["shards_verified"], "data_rule": info["data_rule"],
           "shards_read": len(pool["shards_read"]), "shards_not_read": pool["shards_not_read"],
           "skip": {"dropped_total": sk["dropped_total"], "per_manifest": sk["per_manifest"],
                    "matches_pin": sk["matches_pin"], "pin_differs": sk["pin_differs"],
                    "lists": [{k: v for k, v in r.items() if k != "pin_counts"} for r in sk["lists"]]},
           "pool": {src: {"canvases": v["canvases"], "lines": v["lines"], "vertical": v["vertical"],
                          "ignore": v["ignore"], "weighted": pool["weights"][src]["weighted"],
                          "weight_sum": pool["weights"][src]["weight_sum"],
                          "weight_min": pool["weights"][src]["weight_min"],
                          "cls_lines": pool["weights"][src]["cls_lines"]} for src, v in pool["by_source"].items()},
           "mix_effective": pool["mix_effective"]}
    p = run_dir / "probe.json"
    if p.exists():
        pr = json.loads(p.read_text())
        rec.update({k: pr[k] for k in ("ms_per_step", "ms_per_step_median", "ms_per_step_p90", "ms_waiting_for_data",
                                       "warmup_ms", "startup_s", "steps_timed")})
    d = run_dir / "done.json"
    if d.exists():
        done = json.loads(d.read_text())
        log = [json.loads(x) for x in (run_dir / "log.jsonl").read_text().splitlines() if x.strip()]
        steps = [r for r in log if "lr" in r]
        rec["smoke"] = {"wall_s": done["wall_seconds"], "steps": done["steps"], "vtsp_bytes": done["vtsp_bytes"],
                        "vtsp_sha256": done["vtsp_sha256"], "vtsp_version": done.get("vtsp_version", 1),
                        "final": {k: steps[-1][k] for k in ("k_bce", "dice", "o", "a", "loss")},
                        "ema_swap": any(r.get("ema_swap") for r in log), "qat": any(r.get("qat") for r in log)}
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--out", dest="json_out")
    args = ap.parse_args()
    out = Path(args.out)
    log = (out / "probe.log").read_text()
    rows = [summary_of(d) for kind in ("probe", "smoke") for d in sorted((out / kind).iterdir())
            if d.is_dir() and ".failed" not in d.name and (d / "run.json").exists()]
    for r in rows:
        if "smoke" in r:
            r["smoke"]["check_ok"] = f"CHECK_OK {r['run']} " in log
            r["smoke"]["check_line"] = next((x for x in log.splitlines() if x.startswith(f"CHECK_OK {r['run']} ")),
                                            None)
    doc = {"kind": "textspot2-train-datapath-probe",
           "what": "the sweep's own command lines (textspot2_plan.py args) on the pinned textspot2 shards, each with "
                   "--probe 200 (lab and data on s4_c24n4; data on both d2s shapes), and a 300-step d2s smoke run "
                   "(data + ema on s8_c24n4x2) checked with the C driver; one at a time on the shared box; not runs "
                   "of the sweep, nothing used",
           "rows": rows,
           "label": "measured on a shared 64-core box, 5 threads, one job at a time; times are host times"}
    text = json.dumps(doc, indent=1) + "\n"
    if args.json_out:
        Path(args.json_out).write_text(text)
    for r in rows:
        pm = {Path(m["manifest"]).name: (m["dropped"], m["expected_dropped"]) for m in r["skip"]["per_manifest"]}
        print(f"{r['run']:28s} ms/step {r.get('ms_per_step', '-')} (wait {r.get('ms_waiting_for_data', '-')}, "
              f"startup {r.get('startup_s', '-')} s) dirs {r['shards_dirs']} read {r['shards_read']} unread "
              f"{r['shards_not_read']} dropped(per manifest, expected) {pm} canvases "
              f"{ {k: v['canvases'] for k, v in r['pool'].items()} } weighted "
              f"{ {k: v['weighted'] for k, v in r['pool'].items()} }")
        if "smoke" in r:
            print(f"  smoke {r['smoke']}")


if __name__ == "__main__":
    main()
