#!/usr/bin/env python3
"""head2level: est. ms per box of the two-level arms from the A9 measurements (eval/head2level_protocol.json
"estimates"; every time here is an estimate, nothing ran on a console).

  head2level_time.py --a9 DIR --dev DEV_RESULTS.json --out TIMING.json

DIR holds vita/a9_layers.py measurements (a9_run_int.sh: {pld,nopld}_<tier>_<arm>[_tinf|_t-inf]/report.json) on
ocr-speed's 24 development lines (1,609 frames for the VOCR5 shapes), --topk 8, one runtime build for every arm,
calibrated x1.000676 to VOCR3-H15's measured 4,942 us per CTC frame at an assumed 333 MHz (444 MHz by the clock
ratio). Arms: today (the released blob), gate (head-gate's gated blob, --head-gate on), 2L_* (--two-level) and
2Lg_* (--two-level --head-gate on). Per arm, against its reference (2L: today; 2Lg: gate; both also against today):
  (i)  measured: the pack's own per-frame times;
  (ii) projected to the development rates on wd's finder-cut lines (the console's crops): the head's per-frame time
       a + f (b - a) for 2L, with a = the escape never firing (t = +inf: level 1 and the escape), b = always firing
       (t = -inf: and level 2 on every frame), f = the escape's share of the frames; for 2Lg against the gate,
       saving = (1 - s) (head_today - a) - f (b - a), s = the gate's skip share, f = the escaped share of all frames
       (the gate's own cost is the same in both arms).
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
            "gate_phase_ms": ms(ph[102]["high"]) if 102 in ph else 0.0, "blob_sha256": r["meta"]["blob_sha256"],
            "cli_opts": r["meta"]["cli_opts"], "texts_sha256": r["meta"]["runs"]["prof"]["texts_sha256"]}


def box(ms_frame, frames):
    return frames * ms_frame + FINDER_MS


def saving(base_ms, arm_ms, nominal_box_ms):
    sv = base_ms - arm_ms
    return {"saved_ms_per_frame": round(sv, 4), "saved_ms_nominal_box": round(NOMINAL * sv, 1),
            "saved_ms_real_box": round(REAL * sv, 1), "saved_pct_nominal_box": round(100 * NOMINAL * sv /
                                                                                     nominal_box_ms, 2),
            "saved_ms_nominal_box_444": round(NOMINAL * sv * MHZ / 444.0, 1)}


def dev_arm(dev, t, arm):
    arms = dev["tiers"].get(t, {}).get("arms", {})
    if arm in arms:
        return arm, arms[arm]
    k = arm.split("_K")[1].split("_")[0]
    alt = arm.replace(f"_K{k}_", "_Kalt_")
    return alt, arms.get(alt)


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
        if not (d / f"nopld_{t}_today" / "report.json").exists():
            continue
        ref = {name: {b: load(d / f"{b}_{t}_{name}") for b in ("nopld", "pld")
                      if (d / f"{b}_{t}_{name}" / "report.json").exists()} for name in ("today", "gate")}
        tier = {"today": ref["today"], "gate": ref["gate"], "arms": {}}
        nominal = {b: box(m["ms_per_frame"], NOMINAL) for b, m in ref["today"].items()}
        if ref["gate"].get("nopld"):
            g = ref["gate"]["nopld"]
            tier["gate_vs_today"] = {"measured": saving(ref["today"]["nopld"]["ms_per_frame"], g["ms_per_frame"],
                                                        nominal["nopld"])}
        for p in sorted(d.glob(f"nopld_{t}_2L*")):
            arm = p.name[len(f"nopld_{t}_"):]
            if arm.endswith("_tinf") or arm.endswith("_t-inf"):
                continue
            gate = arm.startswith("2Lg")
            rname = "gate" if gate else "today"
            e = {b: load(d / f"{b}_{t}_{arm}") for b in ("nopld", "pld")
                 if (d / f"{b}_{t}_{arm}" / "report.json").exists()}
            if "nopld" not in e:
                continue
            res = {"reference": rname, "measured": {}, "measured_vs_today": {}, "projected": None}
            for b, m in e.items():
                if b not in ref[rname]:
                    continue
                r = ref[rname][b]
                res["measured"][b] = {"ms_per_frame": round(m["ms_per_frame"], 4), "head_ms": round(m["head_ms"], 4),
                                      "ref_ms_per_frame": round(r["ms_per_frame"], 4),
                                      "ref_head_ms": round(r["head_ms"], 4),
                                      "nominal_box_ms_today": round(nominal[b], 1),
                                      **saving(r["ms_per_frame"], m["ms_per_frame"], nominal[b])}
                res["measured_vs_today"][b] = saving(ref["today"][b]["ms_per_frame"], m["ms_per_frame"], nominal[b])
                if m["texts_sha256"] != ref["today"][b]["texts_sha256"]:
                    res.setdefault("pack_texts_differ_from_today", []).append(b)
            base2l = arm.replace("2Lg_", "2L_")
            lo, hi = d / f"nopld_{t}_{base2l}_tinf", d / f"nopld_{t}_{base2l}_t-inf"
            dname, dinfo = dev_arm(dev, t, arm)
            if (lo / "report.json").exists() and (hi / "report.json").exists() and dinfo is not None:
                A, B = load(lo), load(hi)
                wd = dinfo["escape"]["wd"]
                th = ref["today"]["nopld"]["head_ms"]
                if gate:
                    s, f = wd["gated_share"], wd["fire_share_all_frames"]
                    sv = (1 - s) * (th - A["head_ms"]) - f * (B["head_ms"] - A["head_ms"])
                    res["projected"] = {"dev_arm": dname, "wd_gated_share": s, "wd_fire_share_all_frames": f,
                                        "head_ms_never": round(A["head_ms"], 4),
                                        "head_ms_always": round(B["head_ms"], 4), "today_head_ms": round(th, 4),
                                        **saving(sv, 0.0, nominal["nopld"])}
                else:
                    f = wd["fire_share"]
                    head = A["head_ms"] + f * (B["head_ms"] - A["head_ms"])
                    res["projected"] = {"dev_arm": dname, "wd_fire_share": f, "head_ms_never": round(A["head_ms"], 4),
                                        "head_ms_always": round(B["head_ms"], 4), "head_ms": round(head, 4),
                                        "today_head_ms": round(th, 4), **saving(th, head, nominal["nopld"])}
            safe = [v for v in (res["measured"].get("nopld", {}).get("saved_pct_nominal_box"),
                                (res["projected"] or {}).get("saved_pct_nominal_box")) if v is not None]
            res["saved_pct_nominal_box_safe_min"] = min(safe) if safe else None
            res["d_saving_at_least_3pct"] = bool(safe) and min(safe) >= 3.0
            tier["arms"][arm] = res
            m = res["measured"].get("nopld", {})
            pj = res["projected"] or {}
            vt = res["measured_vs_today"].get("nopld", {})
            print(f"{t} {arm} vs {rname}: measured {m.get('saved_ms_nominal_box')} ms "
                  f"({m.get('saved_pct_nominal_box')}%), projected {pj.get('saved_ms_nominal_box')} ms "
                  f"({pj.get('saved_pct_nominal_box')}%) d={res['d_saving_at_least_3pct']}; "
                  f"vs today {vt.get('saved_ms_nominal_box')} ms", flush=True)
        out["tiers"][t] = tier
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
