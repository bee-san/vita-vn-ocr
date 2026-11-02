#!/usr/bin/env python3
"""textspot-desktop's result tables (issue #173, docs/textspot-desktop.md): markdown from the stored counts.

  tsdesk_tables.py screen --res DIR --arms A,B,... [--bench BENCH.json]
      the screening table: per arm, its parameters and MACs, the network's ONNX Runtime time, hval and dense_screen CER
      at fit:960 and long:960, the screening score, hval without PCSG00854, PCSG00854's junk boxes, line recall, the
      per-edge median / p90 |error| and the tightness cost (hval, fit:960)
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "train", ROOT / "eval", ROOT / "tools", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

EDGES = ("left", "top", "right", "bottom")


def load(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def cer(doc, drop=None):
    U = [u for u in doc["units"] if drop is None or u["cluster"] != drop]
    e, c = sum(u["errors"] for u in U), sum(u["chars"] for u in U)
    return 100.0 * e / max(1, c)


def f2(v, n=2):
    return "" if v is None else f"{v:.{n}f}"


def cmd_screen(a):
    import tsdesk_net as N
    res = Path(a.res)
    rows = []
    head = ("| arm | shape | params | MACs at 960×544 | ORT 1 thread (ms) | hval fit:960 / long:960 | dense_screen "
            "fit:960 / long:960 | score | hval without PCSG00854 (fit:960) | PCSG00854 junk | edges median |e| L/T/R/B "
            "| p90 | tightness cost |")
    rows.append(head)
    rows.append("|" + "---|" * 13)
    for arm in a.arms.split(","):
        run = load(Path(a.runs) / arm / "run.json") or {}
        shape = run.get("shape") or {}
        name = arm.split("_")[1] if "_" in arm else arm
        sh = N.SHAPES.get(name, shape)
        macs = N.macs(sh, 960, 544)["total"] if sh else None
        nt = load(res / f"net_time_{arm}.json") or {}
        c = {}
        for st in ("hval", "dense_screen"):
            for pol in ("fit960", "long960"):
                d = load(res / f"{st}_{arm}_{pol}_L_fast.json")
                c[(st, pol)] = d
        vals = [cer(d) for d in c.values() if d is not None]
        score = sum(vals) / 4 if len(vals) == 4 else None
        hv = c[("hval", "fit960")]
        ed = load(res / f"edges_hval_{arm}_fit960.json")
        if ed:
            e = ed["edges"]["all"]
            med = "/".join(f2(e[k]["median_abs"], 1) for k in EDGES)
            p90 = "/".join(f2(e[k]["p90_abs"], 1) for k in EDGES)
            cost = ed["tightness_cost"]["all"]
            tc = f"{cost['delta']:+.3f} [{cost['ci95'][0]:+.3f}, {cost['ci95'][1]:+.3f}]" if cost["ci95"] else \
                f"{cost['delta']:+.3f}"
        else:
            med = p90 = tc = ""
        junk = sum(u["junk"] for u in hv["units"] if u["cluster"] == "PCSG00854") if hv else None
        cc = {k: (cer(d) if d is not None else None) for k, d in c.items()}
        rows.append(f"| {arm} | {name} | {run.get('params', '')} | {macs / 1e6:.1f} M | "
                    f"{f2((nt.get('960x544') or {}).get('median_ms'))} | "
                    f"{f2(cc[('hval', 'fit960')])} / {f2(cc[('hval', 'long960')])} | "
                    f"{f2(cc[('dense_screen', 'fit960')])} / {f2(cc[('dense_screen', 'long960')])} | "
                    f"{f2(score, 3)} | {f2(cer(hv, 'PCSG00854') if hv else None)} | {'' if junk is None else junk} | "
                    f"{med} | {p90} | {tc} |")
    print("\n".join(rows))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("screen")
    p.add_argument("--res", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--arms", required=True)
    a = ap.parse_args(argv)
    {"screen": cmd_screen}[a.cmd](a)


if __name__ == "__main__":
    main()
