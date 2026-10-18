#!/usr/bin/env python3
"""Cortex-A9 estimates of the text-presence gate's own cost (runtime/vocr_textgate.c), per region: estimates only.

  textgate_a9.py measure --out DIR [--shapes p8_c4,...] [--sizes 900x150,450x80,960x544,480x272] [--regions 2]
                         [--modes region,boxes] [--level -O2] [--seed 1] [--blob SHAPE=FILE ...] [--jobs 5]
  textgate_a9.py report DIR [--json eval/textgate_a9_YYYYMMDD.json]

eval/textdet_a9.py's method, for the gate: vocr_textgate.c (and vocr_lines.c / vocr_lines_rows.c, which the driver
links for the finder mode) compiled by the Vita compiler (arm-vita-eabi-gcc, Thumb-2, the Cortex-A9 flags of
vita/a9_cycles.py, --level, -O2 as Rubi builds its runtime) and linked into the host driver
runtime/vocr_textgate_test.c by an ARM Linux toolchain, so qemu-arm runs the Vita's own machine code for the gate.
Each batch (one shape, one region size, one mode; --regions grey regions of that size tiled with
textdet_export.procedural_region, eval/textdet_a9.region) runs twice under the plugin-enabled qemu-arm with the
Vita's caches: once with --dry (the driver reads the records, runs nothing) and once with --batch in the image form
(boxes mode: three boxes); the gate's cost is the difference, priced by vita/a9_cycles.model_mode (low / high cycle
bounds at 444 and 333 MHz); ms = high bound x 1.0007 (ocr-speed's console calibration, docs/ocr-speed.md).

Also the MAC-rate estimate the protocol names (eval/textgate_protocol.json "estimates"): VOCR3-H15 runs 1,564,992
MACs per CTC frame in 5.0-5.2 ms on one A9 core at 333 MHz (NEON); the gate's scalar code is slower than that rate,
so the instruction-count estimate is the one to read. Nothing of the gate has run on a console.
"""
import argparse
import concurrent.futures as cf
import datetime
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "vita", ROOT / "train", ROOT / "synth", ROOT / "eval", ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import a9_cycles as A  # noqa: E402
import textdet_a9 as TA  # noqa: E402

SIZES = ((900, 150), (450, 80), (960, 544), (480, 272))
CAL = TA.CAL
H15_MACS, H15_MS = 1_564_992, (5.0, 5.2)
SOURCES = ("vocr_textgate.c", "vocr_lines.c", "vocr_lines_rows.c")
MODES = ("region", "boxes")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lookup_shapes():
    import textdet_model as TM
    import textgate_train as G
    return {**TM.SHAPES, **G.SHAPES}


def macs(shape, w, h):
    import textdet_model as TM
    return TM.cost(shape, w, h)["macs"]


def build(out, vitasdk, arm_cc, level):
    vcc = Path(vitasdk) / "bin/arm-vita-eabi-gcc"
    objs = []
    for src in SOURCES:
        o = out / src.replace(".c", "_vita.o")
        subprocess.run([str(vcc), "-std=c99", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror", *A.ARCH, level,
                        "-c", "-o", str(o), str(ROOT / "runtime" / src)], check=True)
        objs.append(o)
    drv = out / "textgate_driver_vitacc"
    subprocess.run([arm_cc, "-std=c99", "-Wall", "-Wextra", "-Werror", *A.ARCH, "-O2", "-static",
                    "-Wl,--no-enum-size-warning", "-o", str(drv), *map(str, objs),
                    str(ROOT / "runtime/vocr_textgate_test.c")], check=True, capture_output=True)
    objdump = arm_cc[:-3] + "objdump" if arm_cc.endswith("gcc") else "objdump"
    dis = subprocess.run([objdump, "-d", "--no-show-raw-insn", str(drv)], check=True, capture_output=True,
                         text=True).stdout
    (out / "driver.dis").write_text(dis)
    rows = A.parse_dis(dis)
    with open(out / "classes.txt", "w") as f:
        for a, (_, mn, ops) in sorted(rows.items()):
            unit = A.classify(mn, ops)[0]
            if unit != "I":
                f.write(f"{a:x} {unit}\n")
    funcs = set()
    nm = Path(vitasdk) / "bin/arm-vita-eabi-nm"
    txt = subprocess.run([str(nm), "--defined-only", str(objs[0])], check=True, capture_output=True, text=True).stdout
    funcs |= {ln.split()[-1] for ln in txt.splitlines() if ln.split()[1:2] and ln.split()[1] in "tT"}
    (out / "functions.txt").write_text("\n".join(sorted(funcs)) + "\n")
    return drv


def blob_for(shape, seed, given):
    if shape["name"] in given:
        p = Path(given[shape["name"]])
        return p.read_bytes(), "file:" + p.parent.name + "/" + p.name
    import textgate_parity as P
    return P.random_blob(shape, seed), f"random_int_layers seed {seed}"


def measure_one(job):
    out, q, drv, name, (w, h), mode, n, seed, blob, blob_src = job
    tag = f"{name}_{w}x{h}_{mode}"
    rng = np.random.default_rng([seed, w, h])
    stage = Path(tempfile.mkdtemp(prefix="vtg-a9-", dir="/tmp"))   # a fixed-length real path, as a9_cycles.py
    try:
        shutil.copy2(drv, stage / drv.name)
        b = stage / "det.vtd"
        b.write_bytes(blob)
        batch = stage / "regions.bin"
        boxes = [(w // 10, h // 8, w * 7 // 10, h // 4), (w // 10, h * 3 // 8, w * 6 // 10, h // 4),
                 (w // 10, h * 5 // 8, w * 5 // 10, h // 4)]
        with open(batch, "wb") as f:
            for _ in range(n):
                g = TA.region(rng, w, h)
                if mode == "region":
                    f.write(struct.pack("<4I", w, h, 1, 2 << 4))
                    f.write(g.tobytes())
                else:
                    f.write(struct.pack("<4I", w, h, 1, 4 | (2 << 4)))
                    f.write(g.tobytes())
                    f.write(struct.pack("<I", len(boxes)) + b"".join(struct.pack("<4I", *bx) for bx in boxes))
        res = {}
        for flag, k in (("--dry", "dry"), ("--batch", "batch")):
            res[k] = TA.run_qemu(q, drv, stage, flag, b, batch, out / f"counts_{tag}_{k}.txt",
                                 out / f"cache_{tag}_{k}.txt", out / "classes.txt")
        lines = [ln.split() for ln in res["batch"].splitlines()]
        return {"shape": name, "size": [w, h], "mode": mode, "regions": n,
                "blob_sha256": hashlib.sha256(blob).hexdigest(), "blob_bytes": len(blob), "blob": blob_src,
                "batch_sha256": sha256(batch),
                "rc": sorted({int(v[0]) for v in lines})}
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def cmd_measure(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    vitasdk = args.vitasdk or os.environ.get("VITASDK") or str(Path(os.environ.get(
        "HERD", "/local/home/skerraut/herd-vita-ocr")) / "shared/artifacts/vita/vitasdk")
    arm_cc = args.arm_cc or os.environ.get("ARM_CC") or A.default_arm_cc()
    q = Path(args.qemu) if args.qemu else A.default_qemu()
    shapes = lookup_shapes()
    names = [s for s in args.shapes.split(",") if s] or list(shapes)
    sizes = [tuple(int(v) for v in s.split("x")) for s in args.sizes.split(",")]
    modes = [m for m in args.modes.split(",") if m]
    given = dict(b.split("=", 1) for b in args.blob)
    drv = build(out, vitasdk, arm_cc, args.level)
    jobs = []
    for name in names:
        blob, src = blob_for(shapes[name], args.seed, given)
        for wh in sizes:
            for mode in modes:
                jobs.append((out, q, drv, name, wh, mode, args.regions, args.seed, blob, src))
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        runs = list(ex.map(measure_one, jobs))
    meta = {"kind": "textgate-a9", "computed": datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"), "method": __doc__.split("\n\n")[1].strip().replace("\n", " "),
        "sources_sha256": {f"runtime/{s}": sha256(ROOT / "runtime" / s) for s in
                           (*SOURCES, "vocr_textgate_test.c", "vocr_textgate.h")},
        "tools": {"vita_cc": TA.tool_version(str(Path(vitasdk) / "bin/arm-vita-eabi-gcc")),
                  "arm_cc": TA.tool_version(arm_cc), "qemu": TA.tool_version(str(q / "bin/qemu-arm")),
                  "qemu_plugins_sha256": {p.name: sha256(p) for p in sorted((q / "plugins").glob("*.so"))},
                  "a9_cycles_sha256": sha256(ROOT / "vita/a9_cycles.py")},
        "flags": ["-std=c99", "-ffp-contract=off", *A.ARCH, args.level, "(Thumb-2: the VitaSDK default)"],
        "cache": A.CACHE_ARGS, "calibration": CAL, "runs": runs}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{len(runs)} batches measured into {out}")


def table_of(d):
    d = Path(d)
    meta = json.loads((d / "meta.json").read_text())
    rows = A.parse_dis((d / "driver.dis").read_text())
    funcs = set((d / "functions.txt").read_text().split())
    shapes = lookup_shapes()
    table = []
    for run in meta["runs"]:
        w, h = run["size"]
        tag = f"{run['shape']}_{w}x{h}_{run['mode']}"
        r = TA.cost(d, rows, funcs, tag, run["regions"])
        m = macs(shapes[run["shape"]], w, h) if run["shape"] in shapes else None
        table.append({**run, "macs": m,
                      "mac_rate_ms_333": [round(m / H15_MACS * v, 2) for v in H15_MS] if m else None,
                      "insns_per_region": r["insns_per_frame"], "cycles_per_region": r["cycles_per_frame"],
                      "ms_per_region": r["ms_per_region"], "top_functions": [[f, round(c)] for f, c in
                                                                              r["top_functions"]],
                      "negative_count_addresses": r["negative_count_addresses"]})
    return meta, table


def cmd_report(args):
    meta, table = table_of(args.dir)
    res = {k: v for k, v in meta.items() if k != "runs"}
    res["note"] = ("estimates: qemu-arm instruction counts priced by vita/a9_cycles.py's Cortex-A9 model, not a "
                   "console measurement; ms = high bound x 1.0007 at the clock. mac_rate_ms_333: MACs at VOCR3-H15's "
                   "NEON rate (1,564,992 MACs in 5.0-5.2 ms at 333 MHz), a lower bound for scalar code")
    res["table"] = table
    print("| shape | mode | region | MACs | MAC-rate ms @333 | insns/region | insns/MAC | est. ms @333 | "
          "est. ms @444 |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|")
    for t in table:
        lo, hi = t["mac_rate_ms_333"]
        print(f"| {t['shape']} | {t['mode']} | {t['size'][0]}x{t['size'][1]} | {t['macs']:,} | {lo:.1f}-{hi:.1f} | "
              f"{t['insns_per_region']:,.0f} | {t['insns_per_region'] / t['macs']:.1f} | "
              f"{t['ms_per_region'][333]['calibrated']:.1f} | {t['ms_per_region'][444]['calibrated']:.1f} |")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--out", required=True)
    m.add_argument("--shapes", default="")
    m.add_argument("--sizes", default=",".join(f"{w}x{h}" for w, h in SIZES))
    m.add_argument("--modes", default=",".join(MODES))
    m.add_argument("--regions", type=int, default=2)
    m.add_argument("--level", default="-O2")
    m.add_argument("--seed", type=int, default=1)
    m.add_argument("--blob", action="append", default=[], help="SHAPE=FILE: a trained blob instead of a random one")
    m.add_argument("--jobs", type=int, default=5)
    m.add_argument("--vitasdk")
    m.add_argument("--arm-cc")
    m.add_argument("--qemu")
    m.set_defaults(fn=cmd_measure)
    r = sub.add_parser("report")
    r.add_argument("dir")
    r.add_argument("--json")
    r.set_defaults(fn=cmd_report)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
