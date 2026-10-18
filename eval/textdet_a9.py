#!/usr/bin/env python3
"""Cortex-A9 estimate of the learned text detector's own cost (runtime/vocr_textdet.c), per region: estimates only.

  textdet_a9.py measure --out DIR [--shapes s4_c8,...] [--sizes 900x150,450x80,960x544,480x272] [--regions 2]
                        [--seed 1] [--blob SHAPE=FILE ...] [--jobs 5] [--vitasdk DIR] [--arm-cc GCC] [--qemu DIR]
  textdet_a9.py report DIR [--json eval/textdet_a9_YYYYMMDD.json] [--also DIR2 --also-label TEXT]

eval/finder_tighten.py's a9 method (cmd_a9), for the detector: vocr_textdet.c (and vocr_lines.c / vocr_lines_rows.c,
whose vocr_line_dark gives the dark flags) compiled by the Vita compiler (arm-vita-eabi-gcc, Thumb-2, -O2, the
Cortex-A9 flags of vita/a9_cycles.py) and linked into the host driver runtime/vocr_textdet_test.c by an ARM Linux
toolchain, so qemu-arm runs the Vita's own machine code for the runtime. Each batch (one shape, one region size,
`--regions` procedural regions of that size from train/textdet_export.procedural_region tiles, grey) runs twice under
the plugin-enabled qemu-arm with the Vita's caches (vita/qemu/insn_mix.c and qemu's cache plugin,
a9_cycles.CACHE_ARGS): once with --dry (the driver reads the records, runs nothing) and once with --batch; the
detector's cost is the difference, per region, priced by vita/a9_cycles.model_mode (low / high cycle bounds at 444
and 333 MHz). ms = high bound x 1.0007 (ocr-speed's console calibration, docs/ocr-speed.md) at that clock. The
network's cost does not depend on the weights (random blobs, train/textdet_export.random_int_layers, unless --blob
names a trained one); the decode's barely. The guest runs from a fresh /tmp staging copy with relative names, an
empty environment and -seed 0, as a9_cycles.py does, so the cache statistics do not depend on the caller.
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
for _p in (ROOT / "vita", ROOT / "train", ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import a9_cycles as A  # noqa: E402

SIZES = ((900, 150), (450, 80), (960, 544), (480, 272))
CAL = 1.0007                 # docs/ocr-speed.md: the console ran VOCR3-H15 at 1.0007 x the high bound (333 MHz)
SOURCES = ("vocr_textdet.c", "vocr_lines.c", "vocr_lines_rows.c")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tool_version(cmd):
    try:
        return subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=60).stdout.splitlines()[0]
    except (OSError, IndexError, subprocess.SubprocessError):
        return None


def build(out, vitasdk, arm_cc):
    """The Vita compiler's objects linked into the host driver; its disassembly and the runtime's functions."""
    vcc = Path(vitasdk) / "bin/arm-vita-eabi-gcc"
    objs = []
    for src in SOURCES:
        o = out / src.replace(".c", "_vita.o")
        subprocess.run([str(vcc), "-std=c99", "-Wall", "-Wextra", "-Werror", *A.ARCH, "-O2", "-c", "-o", str(o),
                        str(ROOT / "runtime" / src)], check=True)
        objs.append(o)
    drv = out / "textdet_driver_vitacc"
    subprocess.run([arm_cc, "-std=c99", "-Wall", "-Wextra", "-Werror", *A.ARCH, "-O2", "-static",
                    "-Wl,--no-enum-size-warning", "-o", str(drv), *map(str, objs),
                    str(ROOT / "runtime/vocr_textdet_test.c")], check=True, capture_output=True)
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
    for o in objs:
        txt = subprocess.run([str(nm), "--defined-only", str(o)], check=True, capture_output=True, text=True).stdout
        funcs |= {ln.split()[-1] for ln in txt.splitlines() if ln.split()[1:2] and ln.split()[1] in "tT"}
    (out / "functions.txt").write_text("\n".join(sorted(funcs)) + "\n")
    size = subprocess.run([str(Path(vitasdk) / "bin/arm-vita-eabi-size"), str(objs[0])], check=True,
                          capture_output=True, text=True).stdout.split("\n")[1].split()[0]
    return drv, rows, funcs, objs, int(size)


def region(rng, w, h):
    """A w x h grey region tiled with procedural regions (text-like stripes and blocks on a plain ground)."""
    import textdet_export as E
    g = np.full((h, w), int(rng.integers(0, 256)), np.uint8)
    y = 0
    while y < h:
        x, rh = 0, 0
        while x < w:
            p = E.procedural_region(rng)
            ph, pw = min(p.shape[0], h - y), min(p.shape[1], w - x)
            g[y:y + ph, x:x + pw] = p[:ph, :pw]
            x += pw
            rh = max(rh, ph)
        y += rh
    return g


def blob_for(shape, seed, given):
    import textdet_export as E
    if shape in given:
        return Path(given[shape]).read_bytes(), "file:" + Path(given[shape]).name
    sh, q = E.random_int_layers(shape, seed)
    return E.pack(sh, q, hashlib.sha256(f"random:{shape}:{seed}".encode()).digest()), f"random_int_layers seed {seed}"


def run_qemu(q, drv, stage, mode, blob, batch, counts, cache, classes):
    plugins = ["-plugin", f"{q}/plugins/libinsn_mix.so,out={counts},classes={classes}",
               "-plugin", f"{q}/plugins/libcache.so,{A.CACHE_ARGS}", "-d", "plugin", "-D", str(cache)]
    p = subprocess.run([str(q / "bin/qemu-arm"), "-cpu", "cortex-a9", "-seed", "0", *plugins, f"./{drv.name}", mode,
                        blob.name, batch.name], capture_output=True, text=True, stdin=subprocess.DEVNULL,
                       timeout=14400, env={}, cwd=stage)
    if p.returncode:
        raise RuntimeError(f"qemu {mode} failed: {p.stderr[-2000:]}")
    return p.stdout


def measure_one(job):
    """One (shape, size) batch: dry and real runs in its own staging directory."""
    out, q, drv, shape, (w, h), n, seed, blob, blob_src = job
    tag = f"{shape}_{w}x{h}"
    rng = np.random.default_rng([seed, w, h])
    stage = Path(tempfile.mkdtemp(prefix="vtd-a9-", dir="/tmp"))   # fixed-length real path, as a9_cycles.py
    try:
        shutil.copy2(drv, stage / drv.name)
        b = stage / "det.vtd"
        b.write_bytes(blob)
        batch = stage / "regions.bin"
        with open(batch, "wb") as f:
            for _ in range(n):
                g = region(rng, w, h)
                f.write(struct.pack("<4I", w, h, 1, 0))
                f.write(g.tobytes())
        res = {}
        # --empty reads the records and runs nothing; the same length as --batch, so the guest stack sits alike
        for mode, k in (("--empty", "dry"), ("--batch", "batch")):
            res[k] = run_qemu(q, drv, stage, mode, b, batch, out / f"counts_{tag}_{k}.txt",
                              out / f"cache_{tag}_{k}.txt", out / "classes.txt")
        lines = [ln.split() for ln in res["batch"].splitlines()]
        return {"shape": shape, "size": [w, h], "regions": n, "blob_sha256": hashlib.sha256(blob).hexdigest(),
                "blob_bytes": len(blob), "blob": blob_src, "batch_sha256": sha256(batch),
                "rc": sorted({int(v[0]) for v in lines}), "lines_out": sum(int(v[1]) for v in lines)}
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def cost(out, rows, funcs, tag, regions):
    c_on, x_on = A.read_counts(out / f"counts_{tag}_batch.txt")
    c_off, x_off = A.read_counts(out / f"counts_{tag}_dry.txt")
    delta = {a: c_on.get(a, 0) - c_off.get(a, 0) for a in set(c_on) | set(c_off)}
    neg = sum(1 for v in delta.values() if v < 0)
    delta = {a: v for a, v in delta.items() if v > 0}
    extra = {k: x_on.get(k, 0) - x_off.get(k, 0) for k in set(x_on) | set(x_off)}
    con, coff = A.read_cache(out / f"cache_{tag}_batch.txt"), A.read_cache(out / f"cache_{tag}_dry.txt")
    cache = {k: max(0, con[k] - coff[k]) for k in con} if con and coff else None
    r = A.model_mode(rows, delta, extra, funcs, regions, cache)
    r["ms_per_region"] = {mhz: {"low": r["cycles_per_frame"][mhz][0] / (mhz * 1000),
                                "high": r["cycles_per_frame"][mhz][1] / (mhz * 1000),
                                "calibrated": CAL * r["cycles_per_frame"][mhz][1] / (mhz * 1000)} for mhz in A.CLOCKS}
    r["negative_count_addresses"] = neg
    return r


def cmd_measure(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    vitasdk = args.vitasdk or os.environ.get("VITASDK")
    if not vitasdk:
        raise SystemExit("set VITASDK or --vitasdk")
    arm_cc = args.arm_cc or os.environ.get("ARM_CC") or A.default_arm_cc()
    q = Path(args.qemu) if args.qemu else A.default_qemu()
    import textdet_model as TM
    shapes = [s for s in args.shapes.split(",") if s] or list(TM.SHAPES)
    sizes = [tuple(int(v) for v in s.split("x")) for s in args.sizes.split(",")]
    given = dict(b.split("=", 1) for b in args.blob)
    drv, rows, funcs, objs, text = build(out, vitasdk, arm_cc)
    jobs = []
    for shape in shapes:
        blob, src = blob_for(shape, args.seed, given)
        for wh in sizes:
            jobs.append((out, q, drv, shape, wh, args.regions, args.seed, blob, src))
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        runs = list(ex.map(measure_one, jobs))
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = {"kind": "textdet-a9", "computed": now,
            "method": __doc__.split("\n\n")[1].strip().replace("\n", " "),
            "sources_sha256": {f"runtime/{s}": sha256(ROOT / "runtime" / s) for s in
                               (*SOURCES, "vocr_textdet_test.c", "vocr_textdet.h")},
            "tools": {"vita_cc": tool_version(str(Path(vitasdk) / "bin/arm-vita-eabi-gcc")),
                      "arm_cc": tool_version(arm_cc), "qemu": tool_version(str(q / "bin/qemu-arm")),
                      "qemu_plugins_sha256": {p.name: sha256(p) for p in sorted((q / "plugins").glob("*.so"))},
                      "a9_cycles_sha256": sha256(ROOT / "vita/a9_cycles.py")},
            "flags": ["-std=c99", *A.ARCH, "-O2", "(Thumb-2: the VitaSDK default)"], "cache": A.CACHE_ARGS,
            "vocr_textdet_o_text_bytes": text, "calibration": CAL, "runs": runs}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{len(runs)} batches measured into {out}")


def table_of(d):
    """The per-(shape, size) rows of one measured directory."""
    d = Path(d)
    meta = json.loads((d / "meta.json").read_text())
    rows = A.parse_dis((d / "driver.dis").read_text())
    funcs = set((d / "functions.txt").read_text().split())
    import textdet_model as TM
    table = []
    for run in meta["runs"]:
        w, h = run["size"]
        tag = f"{run['shape']}_{w}x{h}"
        r = cost(d, rows, funcs, tag, run["regions"])
        shape = TM.SHAPES.get(run["shape"])
        macs = TM.cost(shape, w, h)["macs"] if shape else None
        table.append({**run, "macs": macs, "insns_per_region": r["insns_per_frame"],
                      "cycles_per_region": r["cycles_per_frame"], "ms_per_region": r["ms_per_region"],
                      "l1d_misses_per_region": r["l1d_misses_per_frame"],
                      "l2_misses_per_region": r["l2_misses_per_frame"],
                      "low_bound_set_by": max(r["low_parts_per_frame"], key=r["low_parts_per_frame"].get),
                      "top_functions": [[f, round(c)] for f, c in r["top_functions"]],
                      "negative_count_addresses": r["negative_count_addresses"]})
    return meta, table


def cmd_report(args):
    meta, table = table_of(args.dir)
    res = {k: v for k, v in meta.items() if k != "runs"}
    res["note"] = ("estimates: qemu-arm instruction counts priced by vita/a9_cycles.py's Cortex-A9 model, not a "
                   "console measurement; ms = high bound x 1.0007 at the clock")
    res["table"] = table
    if args.also:
        ameta, atable = table_of(args.also)
        if ameta["sources_sha256"] != meta["sources_sha256"]:
            raise SystemExit("--also was measured on other sources")
        res["also"] = {"label": args.also_label, "computed": ameta["computed"], "table": atable}
    print("| shape | region | MACs | insns/region | insns/MAC | cycles/region (low-high, 444 MHz) | est. ms @333 "
          "| est. ms @444 |")
    print("|---|---|---:|---:|---:|---|---:|---:|")
    for t in table:
        lo, hi = t["cycles_per_region"][444]
        print(f"| {t['shape']} | {t['size'][0]}x{t['size'][1]} | {t['macs']:,} | {t['insns_per_region']:,.0f} | "
              f"{t['insns_per_region'] / t['macs']:.2f} | {lo:,.0f}-{hi:,.0f} | "
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
    m.add_argument("--regions", type=int, default=2)
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
    r.add_argument("--also", help="a second measured directory (same sources), e.g. trained blobs")
    r.add_argument("--also-label", default="")
    r.set_defaults(fn=cmd_report)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
