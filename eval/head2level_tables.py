#!/usr/bin/env python3
"""head2level: the markdown tables of docs/head2level.md from the committed records (numbers only).

  head2level_tables.py --fit S=FIT_S.json ... --r48 S:1500=R48.json ... --dev DEV.json --timing T.json
                       --decision D.json [--counter S=C_S.json ...]
"""
import argparse
import json
from pathlib import Path

DECODES = ("greedy", "rubi", "rubi_cascade")


def ci(x):
    return f"{x['delta_points']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}]"


def rci(x):
    return f"{x['loss_points']:+.2f} [{x['ci95'][0]:+.2f}, {x['ci95'][1]:+.2f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", nargs="+", required=True)
    ap.add_argument("--r48", nargs="*", default=[])
    ap.add_argument("--dev", required=True)
    ap.add_argument("--timing", required=True)
    ap.add_argument("--decision", required=True)
    ap.add_argument("--counter", nargs="*", default=[])
    a = ap.parse_args()
    fits = {t: json.loads(Path(p).read_text()) for t, p in (x.split("=", 1) for x in a.fit)}
    r48 = {}
    for x in a.r48:
        k, p = x.split("=", 1)
        r48[tuple(k.split(":"))] = json.loads(Path(p).read_text())
    dev = json.loads(Path(a.dev).read_text())
    tim = json.loads(Path(a.timing).read_text())
    dec = json.loads(Path(a.decision).read_text())
    print("### Escapes (training-side calibration split)\n")
    print("| tier | K | level-1 rows | calibration positives | fire share (0.999 recall) | head rows per frame W |")
    print("|---|---:|---:|---:|---:|---:|")
    for t, f in fits.items():
        for K, o in sorted(f["fits"].items(), key=lambda kv: int(kv[0])):
            th = o["thresholds"]["r0.999"]
            mark = " (K_alt)" if int(K) == f["K_alt"] else ""
            print(f"| {t} | {K}{mark} | {o['L1']:,} | {o['cal_positives']:,} | {100 * th['cal_fire_share']:.2f}% | "
                  f"{o['W']:,.0f} |")
    if r48:
        print("\n### r48 (the 48-wide rare block, calibration split)\n")
        print("| tier | K | mean KL (float) | argmax = today's (exact) | non-blank frames | "
              "rare-block bytes: r96 -> r48 | escape fire share |")
        print("|---|---:|---:|---:|---:|---:|---:|")
        for (t, K), r in sorted(r48.items()):
            print(f"| {t} | {K} | {r['cal_mean_kl_float']:.5f} | {100 * r['cal_argmax_agree_exact']:.3f}% | "
                  f"{100 * r['cal_argmax_agree_exact_nonblank']:.3f}% | {r['bytes_rare96']:,} -> {r['bytes_r48']:,} | "
                  f"{100 * r['escape']['thresholds']['r0.999']['cal_fire_share']:.2f}% |")
    print("\n### Development (dev score in CER points, 95% VN-cluster intervals; rare recall loss in points)\n")
    print("| tier | arm | verdict | vs | greedy | rubi | rubi_cascade | rare recall loss (rubi) | escape fires (wd) | "
          "head rows / frame (wd) |")
    print("|---|---|---|---|---|---|---|---|---:|---:|")
    for t, tier in dev["tiers"].items():
        for arm, info in tier["arms"].items():
            if arm in ("today", "gate"):
                continue
            v = dec["tiers"][t]["arms"][arm]["verdict"]
            for ref, cm in info["vs"].items():
                cells = [ci(cm[d]["dev"]["score"]) if d in cm else "-" for d in DECODES]
                rr = rci(cm["rubi"]["rare_recall"]["combined"]) if "rubi" in cm else "-"
                e = info["escape"]["wd"]
                print(f"| {t} | {arm} | {v} | {ref} | {' | '.join(cells)} | {rr} | {100 * e['fire_share']:.2f}% | "
                      f"{e.get('head_rows_per_frame', '-'):,} |")
    print("\n### Time (est., one Cortex-A9 core at 333 MHz, no prefetch credit; nominal box 240 frames + 0.1 s)\n")
    print("| tier | arm | ms / frame | saved per nominal box vs its reference: measured / projected | % of box | "
          "real box (89 frames) | vs today | 444 MHz | (d) |")
    print("|---|---|---:|---|---:|---:|---:|---:|---|")
    for t, tier in tim["tiers"].items():
        td = tier["today"]["nopld"]
        print(f"| {t} | today | {td['ms_per_frame']:.3f} | | | | | | |")
        if tier.get("gate"):
            g = tier["gate"]["nopld"]
            gv = tier["gate_vs_today"]["measured"]
            print(f"| {t} | gate | {g['ms_per_frame']:.3f} | {gv['saved_ms_nominal_box']:.0f} (vs today) | "
                  f"{gv['saved_pct_nominal_box']:.1f}% | {gv['saved_ms_real_box']:.0f} | | "
                  f"{gv['saved_ms_nominal_box_444']:.0f} | |")
        for arm, r in tier["arms"].items():
            m = r["measured"]["nopld"]
            pj = r["projected"] or {}
            vt = r["measured_vs_today"]["nopld"]
            print(f"| {t} | {arm} | {m['ms_per_frame']:.3f} | {m['saved_ms_nominal_box']:.0f} / "
                  f"{pj.get('saved_ms_nominal_box', float('nan')):.0f} (vs {r['reference']}) | "
                  f"{m['saved_pct_nominal_box']:.1f}% / {pj.get('saved_pct_nominal_box', float('nan')):.1f}% | "
                  f"{m['saved_ms_real_box']:.0f} / {pj.get('saved_ms_real_box', float('nan')):.0f} | "
                  f"{vt['saved_ms_nominal_box']:.0f} | {m['saved_ms_nominal_box_444']:.0f} | "
                  f"{'yes' if r['d_saving_at_least_3pct'] else 'no'} |")
    print("\n### Selection\n")
    for t, d in dec["tiers"].items():
        print(f"- {t}: candidates {', '.join(d['candidates'])}; adoptable {', '.join(d['adoptable']) or 'none'}; "
              f"selected **{d['selected'] or 'none (today)'}**")
    if a.counter:
        print("\n### Counter-context set (#109, synthetic, report only)\n")
        print("| tier | arm | greedy CER (lines worse) | rubi CER (lines worse) | with the re-read (lines worse) | "
              "re-read lines |")
        print("|---|---|---:|---:|---:|---:|")
        for x in a.counter:
            t, p = x.split("=", 1)
            c = json.loads(Path(p).read_text())
            for arm, r in c["arms"].items():
                print(f"| {t} | {arm} | {r['greedy']['cer_pct']:.2f}% ({r['greedy']['lines_worse']}) | "
                      f"{r['rubi']['cer_pct']:.2f}% ({r['rubi']['lines_worse']}) | "
                      f"{r['rubi_reread']['cer_pct']:.2f}% ({r['rubi_reread']['lines_worse']}) | {r['reread_lines']} |")


if __name__ == "__main__":
    main()
