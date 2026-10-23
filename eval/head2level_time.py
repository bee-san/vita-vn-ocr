#!/usr/bin/env python3
"""head2level: est. ms per box of the two-level arms from the A9 measurements (eval/head2level_protocol.json
"estimates"; every time here is an estimate, nothing ran on a console).

  head2level_time.py --a9 DIR --dev DEV_RESULTS.json --out TIMING.json

DIR holds vita/a9_layers.py measurements (head2level a9_run.sh: {pld,nopld}_<tier>_<arm>[_tinf|_t-inf]/report.json)
on ocr-speed's 24 development lines (1,609 frames for the VOCR5 shapes), --topk 8, calibrated x1.000676 to VOCR3-H15's
measured 4,942 us per CTC frame at an assumed 333 MHz (444 MHz by the clock ratio). Per arm:
  (i)  measured: the pack's own per-frame time with the arm's blob and switches;
  (ii) projected: the head's per-frame time a + f (b - a), a = the escape never firing (t = +inf: level 1 and the
       escape), b = always firing (t = -inf: and level 2 on every frame), f = the development fire share on wd's
       finder-cut lines (the console's crops), against today's head.
Boxes: nominal = 240 frames + 100 ms (the owner's 3-line box), real = 89 frames + 100 ms (text-stats-boxes).
"""
import argparse
import json
from pathlib import Path

MHZ = 333.0
NOMINAL, REAL, FINDER_MS = 240, 89, 100.0


def load(d):
    r = json.loads((Path(d) / "report.json").read_text())
    F, s = r["frames"], r["scale"]
    ph = {int(k): v for k, v in r["phases"].items()}
    head = max(k for k in ph if k < 100)

    def ms(cyc):
        return cyc * s / F / (MHZ * 1e3)
    return {"frames": F, "ms_per_frame": ms(sum(v["high"] for v in ph.values())), "head_ms": ms(ph[head]["high"]),
            "head_phase": head, "blob_sha256": r["meta"]["blob_sha256"], "cli_opts": r["meta"]["cli_opts"],
            "texts_sha256": r["meta"]["runs"]["prof"]["texts_sha256"]}


def box(ms_frame, frames):
    return frames * ms_frame + FINDER_MS


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a9", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.a9)
    dev = json.loads(Path(a.dev).read_text())
    out = {"kind": "head2level-timing", "mhz": MHZ, "boxes": {"nominal_frames": NOMINAL, "real_frames": REAL,
                                                              "finder_ms": FINDER_MS}, "tiers": {}}
    for t in ("S", "N", "L"):
        if not (d / f"nopld_{t}_today").exists():
            continue
        tier = {"today": {b: load(d / f"{b}_{t}_today") for b in ("nopld", "pld") if (d / f"{b}_{t}_today").exists()},
                "arms": {}}
        for p in sorted(d.glob(f"nopld_{t}_2L*")):
            arm = p.name[len(f"nopld_{t}_"):]
            if arm.endswith("_tinf") or arm.endswith("_t-inf"):
                continue
            e = {b: load(d / f"{b}_{t}_{arm}") for b in ("nopld", "pld") if (d / f"{b}_{t}_{arm}").exists()}
            res = {"measured": {}, "projected": None}
            for b, m in e.items():
                base = tier["today"][b]
                if m["texts_sha256"] != base["texts_sha256"]:
                    res.setdefault("pack_texts_differ_from_today", []).append(b)
                sv = base["ms_per_frame"] - m["ms_per_frame"]
                res["measured"][b] = {"ms_per_frame": round(m["ms_per_frame"], 4), "head_ms": round(m["head_ms"], 4),
                                      "today_ms_per_frame": round(base["ms_per_frame"], 4),
                                      "today_head_ms": round(base["head_ms"], 4),
                                      "saved_ms_per_frame": round(sv, 4),
                                      "saved_ms_nominal_box": round(NOMINAL * sv, 1),
                                      "saved_ms_real_box": round(REAL * sv, 1),
                                      "nominal_box_ms_today": round(box(base["ms_per_frame"], NOMINAL), 1),
                                      "saved_pct_nominal_box": round(100 * NOMINAL * sv /
                                                                     box(base["ms_per_frame"], NOMINAL), 2)}
            lo, hi = d / f"nopld_{t}_{arm}_tinf", d / f"nopld_{t}_{arm}_t-inf"
            # the dev arm name for this timing arm: 2L_K500_* is the development run's 2L_Kalt_* (K_alt = 500)
            dev_arm = arm
            if arm not in dev["tiers"].get(t, {}).get("arms", {}):
                dev_arm = arm.replace(f"_K{dev_arm.split('_K')[1].split('_')[0]}_", "_Kalt_")
            dinfo = dev["tiers"].get(t, {}).get("arms", {}).get(dev_arm)
            if lo.exists() and hi.exists() and dinfo is not None:
                A, B = load(lo), load(hi)
                f = dinfo["escape"]["wd"]["fire_share"]
                base = tier["today"]["nopld"]
                head = A["head_ms"] + f * (B["head_ms"] - A["head_ms"])
                sv = base["head_ms"] - head
                res["projected"] = {"dev_arm": dev_arm, "wd_fire_share": f, "head_ms_never": round(A["head_ms"], 4),
                                    "head_ms_always": round(B["head_ms"], 4), "head_ms": round(head, 4),
                                    "today_head_ms": round(base["head_ms"], 4), "saved_ms_per_frame": round(sv, 4),
                                    "saved_ms_nominal_box": round(NOMINAL * sv, 1),
                                    "saved_ms_real_box": round(REAL * sv, 1),
                                    "saved_pct_nominal_box": round(100 * NOMINAL * sv /
                                                                   box(base["ms_per_frame"], NOMINAL), 2)}
            safe = [v for v in (res["measured"].get("nopld", {}).get("saved_pct_nominal_box"),
                                (res["projected"] or {}).get("saved_pct_nominal_box")) if v is not None]
            res["saved_pct_nominal_box_safe_min"] = min(safe) if safe else None
            res["d_saving_at_least_3pct"] = bool(safe) and min(safe) >= 3.0
            tier["arms"][arm] = res
        out["tiers"][t] = tier
        for arm, r in tier["arms"].items():
            m = r["measured"].get("nopld", {})
            print(f"{t} {arm}: measured {m.get('saved_ms_nominal_box')} ms ({m.get('saved_pct_nominal_box')}%), "
                  f"projected {(r['projected'] or {}).get('saved_ms_nominal_box')} ms "
                  f"({(r['projected'] or {}).get('saved_pct_nominal_box')}%) d={r['d_saving_at_least_3pct']}")
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
