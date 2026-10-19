#!/usr/bin/env python3
"""Cortex-A9 estimate of the full-frame text spotter's cost (runtime/vocr_textspot.c) per frame, and its exact
arena: estimates only.

  textspot_a9.py measure --out DIR [--shapes s4_c8,...] [--scales h2,f1] [--sizes 960x544,480x272]
                         [--formats grey] [--nodark] [--cflags "-O2"] [--seed 1] [--jobs 5] [--vitasdk DIR]
                         [--arm-cc GCC] [--qemu DIR]
  textspot_a9.py report DIR [--json eval/textspot_a9_YYYYMMDD.json] [--also DIR2 --also-label TEXT ...]

eval/textdet_a9.py's method (#97), for the spotter: vocr_textspot.c (and vocr_lines.c / vocr_lines_rows.c, whose
vocr_line_dark gives the dark flags) compiled by the Vita compiler (arm-vita-eabi-gcc, Thumb-2, -O2, the
Cortex-A9 flags of vita/a9_cycles.py) and linked into the host driver runtime/vocr_textspot_test.c by an ARM
Linux toolchain, so qemu-arm runs the Vita's own machine code for the runtime. Each batch (one shape, one scale:
h2 = half_above 600 or f1 = 0, one frame size, one pinned procedural frame of that size tiled with
train/textdet_export.procedural_region, grey unless --formats says rgba) runs twice under the plugin-enabled
qemu-arm with the Vita's caches (vita/qemu/insn_mix.c and qemu's cache plugin, a9_cycles.CACHE_ARGS): once with
--empty (the driver reads the record, runs nothing) and once with --batch (vocr_textspot_find: the halving, the
network, the decode, blocks, order, pick and the dark flags); the spotter's cost is the difference, priced by
vita/a9_cycles.model_mode (low / high cycle bounds). Est. ms at 333 MHz = high bound x 1.0007 (ocr-speed's console
calibration, docs/ocr-speed.md) / 333 MHz; at 444 MHz that x 333/444 (eval/textspot_protocol.json "estimates").
The network's cost does not depend on the weights (random blobs, train/textdet_export.random_int_layers, wrapped by
synth/textspot_twin.pack), the decode's barely. The arena columns are the runtime's own exact functions, run by the
same Vita code under qemu-arm: vocr_textspot_frame_arena_bytes (the frame's need) and vocr_textspot_arena_bytes
(every frame up to that size). The guest runs from a fresh /tmp staging copy with relative names, an empty
environment and -seed 0, as a9_cycles.py does, so the cache statistics do not depend on the caller.
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

SIZES = ((960, 544), (480, 272))
SCALES = {"h2": 600, "f1": 0}
CAL = 1.0007                 # docs/ocr-speed.md: the console ran VOCR3-H15 at 1.0007 x the high bound (333 MHz)
SOURCES = ("vocr_textspot.c", "vocr_lines.c", "vocr_lines_rows.c")
# #97's 34 runs (rounds 1 and 3): their 21 shapes; family B's (eval/textspot_protocol.json) are among them
RUN_SHAPES = ("s4_c4", "s4_c6", "s4_c8", "s4_c8n5", "s4_c12", "s4_c16", "s4_c16n4", "s4_c24", "s4p_c4", "s4p_c8",
              "s8_c8", "s8_c16", "s8_c24", "s8_c32", "s8_c32n4", "s8_c48", "s4_c12n4", "s4_c20n4", "s4_c24n4",
              "s8_c16n4", "s8_c24n4")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tool_version(cmd):
    try:
        return subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=60).stdout.splitlines()[0]
    except (OSError, IndexError, subprocess.SubprocessError):
        return None


def family_b():
    p = json.loads((ROOT / "eval" / "textspot_protocol.json").read_text(encoding="utf-8"))
    return p["candidates"]["family_b"]["shapes"]


def build(out, vitasdk, arm_cc, cflags=("-O2",)):
    """The Vita compiler's objects linked into the host driver; its disassembly and the runtime's functions.
    cflags go to the spotter's own object (vocr_lines*.c keep -O2, as on the console)."""
    vcc = Path(vitasdk) / "bin/arm-vita-eabi-gcc"
    objs = []
    for src in SOURCES:
        o = out / src.replace(".c", "_vita.o")
        flags = list(cflags) if src == SOURCES[0] else ["-O2"]
        subprocess.run([str(vcc), "-std=c99", "-Wall", "-Wextra", "-Werror", *A.ARCH, *flags, "-c", "-o", str(o),
                        str(ROOT / "runtime" / src)], check=True)
        objs.append(o)
    drv = out / "textspot_driver_vitacc"
    subprocess.run([arm_cc, "-std=c99", "-Wall", "-Wextra", "-Werror", *A.ARCH, "-O2", "-static",
                    "-Wl,--no-enum-size-warning", "-o", str(drv), *map(str, objs),
                    str(ROOT / "runtime/vocr_textspot_test.c")], check=True, capture_output=True)
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


def frame(seed, w, h):
    """The pinned w x h grey frame: procedural regions (text-like stripes and blocks on plain grounds) tiled over a
    plain ground, from numpy's default_rng([seed, w, h])."""
    import textdet_export as E
    rng = np.random.default_rng([seed, w, h])
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


def rgba_of(g):
    """An RGBA frame of exactly this grey (R = G = B = grey, alpha 255)."""
    out = np.empty((*g.shape, 4), np.uint8)
    out[..., :3] = g[..., None]
    out[..., 3] = 255
    return out


def blob_for(shape, scale, seed):
    import textdet_export as E
    import textspot_twin as S
    sh, q = E.random_int_layers(shape, seed)
    vtdt = E.pack(sh, q, hashlib.sha256(f"random:{shape}:{seed}".encode()).digest())
    src = f"random_int_layers seed {seed}, pack(half_above {SCALES[scale]}, conf_min 1)"
    return S.pack(vtdt, SCALES[scale], 1), src


def run_qemu(q, drv, stage, mode, blob, batch, counts, cache, classes):
    plugins = ["-plugin", f"{q}/plugins/libinsn_mix.so,out={counts},classes={classes}",
               "-plugin", f"{q}/plugins/libcache.so,{A.CACHE_ARGS}", "-d", "plugin", "-D", str(cache)]
    p = subprocess.run([str(q / "bin/qemu-arm"), "-cpu", "cortex-a9", "-seed", "0", *plugins, f"./{drv.name}", mode,
                        blob.name, batch.name], capture_output=True, text=True, stdin=subprocess.DEVNULL,
                       timeout=14400, env={}, cwd=stage)
    if p.returncode:
        raise RuntimeError(f"qemu {mode} failed: {p.stderr[-2000:]}")
    return p.stdout


def arenas(q, drv, blob, w, h):
    """(vocr_textspot_frame_arena_bytes, vocr_textspot_arena_bytes) of the Vita code at w x h (plain qemu-arm)."""
    stage = Path(tempfile.mkdtemp(prefix="vtsp-arena-", dir="/tmp"))
    try:
        shutil.copy2(drv, stage / drv.name)
        (stage / "spot.vtsp").write_bytes(blob)
        res = []
        for mode in ("--frame-arena", "--arena"):
            p = subprocess.run([str(q / "bin/qemu-arm"), "-cpu", "cortex-a9", f"./{drv.name}", mode, "spot.vtsp",
                                str(w), str(h)], capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=600,
                               env={}, cwd=stage, check=True)
            res.append(int(p.stdout.strip()))
        return res
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def measure_one(job):
    """One (shape, scale, size, format) batch: dry and real runs in its own staging directory."""
    out, q, drv, shape, scale, (w, h), fmt, seed, blob, blob_src, nodark = job
    tag = f"{shape}_{scale}_{w}x{h}_{fmt}"
    stage = Path(tempfile.mkdtemp(prefix="vtsp-a9-", dir="/tmp"))   # fixed-length real path, as a9_cycles.py
    try:
        shutil.copy2(drv, stage / drv.name)
        b = stage / "spot.vtsp"
        b.write_bytes(blob)
        batch = stage / "frames.bin"
        g = frame(seed, w, h)
        img = g if fmt == "grey" else rgba_of(g)
        with open(batch, "wb") as f:
            f.write(struct.pack("<4I", w, h, 1 if fmt == "grey" else 4, 2 if nodark else 0))
            f.write(img.tobytes())
        res = {}
        # --empty reads the records and runs nothing; the same length as --batch, so the guest stack sits alike
        for mode, k in (("--empty", "dry"), ("--batch", "batch")):
            res[k] = run_qemu(q, drv, stage, mode, b, batch, out / f"counts_{tag}_{k}.txt",
                              out / f"cache_{tag}_{k}.txt", out / "classes.txt")
        v = res["batch"].split()
        frame_arena, arena = arenas(q, drv, blob, w, h)
        return {"shape": shape, "scale": scale, "half_above": SCALES[scale], "size": [w, h], "format": fmt,
                "flags": "VOCR_TEXTSPOT_NO_DARK" if nodark else "0", "frames": 1,
                "blob_sha256": hashlib.sha256(blob).hexdigest(), "blob_bytes": len(blob),
                "blob": blob_src, "frame_sha256": hashlib.sha256(g.tobytes()).hexdigest(),
                "batch_sha256": sha256(batch), "rc": int(v[0]), "boxes": int(v[1]), "blocks": int(v[2]),
                "pick": int(v[3]), "truncated": int(v[4]), "frame_arena_bytes": frame_arena, "arena_bytes": arena}
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def cost(out, rows, funcs, tag, frames):
    c_on, x_on = A.read_counts(out / f"counts_{tag}_batch.txt")
    c_off, x_off = A.read_counts(out / f"counts_{tag}_dry.txt")
    delta = {a: c_on.get(a, 0) - c_off.get(a, 0) for a in set(c_on) | set(c_off)}
    neg = sum(1 for v in delta.values() if v < 0)
    delta = {a: v for a, v in delta.items() if v > 0}
    extra = {k: x_on.get(k, 0) - x_off.get(k, 0) for k in set(x_on) | set(x_off)}
    con, coff = A.read_cache(out / f"cache_{tag}_batch.txt"), A.read_cache(out / f"cache_{tag}_dry.txt")
    cache = {k: max(0, con[k] - coff[k]) for k in con} if con and coff else None
    r = A.model_mode(rows, delta, extra, funcs, frames, cache)
    ms333 = CAL * r["cycles_per_frame"][333][1] / (333 * 1000)
    r["ms_per_frame"] = {"333": ms333, "444": ms333 * 333 / 444,
                         "model_444_high": CAL * r["cycles_per_frame"][444][1] / (444 * 1000)}
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
    shapes = [s for s in args.shapes.split(",") if s] or list(RUN_SHAPES)
    scales = [s for s in args.scales.split(",") if s]
    sizes = [tuple(int(v) for v in s.split("x")) for s in args.sizes.split(",")]
    formats = [f for f in args.formats.split(",") if f]
    cflags = args.cflags.split()
    drv, rows, funcs, objs, text = build(out, vitasdk, arm_cc, cflags)
    jobs = []
    for shape in shapes:
        for scale in scales:
            blob, src = blob_for(shape, scale, args.seed)
            for wh in sizes:
                for fmt in formats:
                    jobs.append((out, q, drv, shape, scale, wh, fmt, args.seed, blob, src, args.nodark))
    with cf.ThreadPoolExecutor(args.jobs) as ex:
        runs = list(ex.map(measure_one, jobs))
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = {"kind": "textspot-a9", "computed": now,
            "method": __doc__.split("\n\n")[1].strip().replace("\n", " "),
            "sources_sha256": {f"runtime/{s}": sha256(ROOT / "runtime" / s) for s in
                               (*SOURCES, "vocr_textspot_test.c", "vocr_textspot.h")},
            "tools": {"vita_cc": tool_version(str(Path(vitasdk) / "bin/arm-vita-eabi-gcc")),
                      "arm_cc": tool_version(arm_cc), "qemu": tool_version(str(q / "bin/qemu-arm")),
                      "qemu_plugins_sha256": {p.name: sha256(p) for p in sorted((q / "plugins").glob("*.so"))},
                      "a9_cycles_sha256": sha256(ROOT / "vita/a9_cycles.py"),
                      "textspot_twin_sha256": sha256(ROOT / "synth/textspot_twin.py"),
                      "textdet_export_sha256": sha256(ROOT / "train/textdet_export.py")},
            "flags": ["-std=c99", *A.ARCH, *cflags, "(Thumb-2: the VitaSDK default; vocr_lines*.c at -O2)"],
            "cache": A.CACHE_ARGS,
            "vocr_textspot_o_text_bytes": text, "calibration": CAL, "seed": args.seed, "runs": runs}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{len(runs)} batches measured into {out}")


def table_of(d):
    """The per-(shape, scale, size, format) rows of one measured directory."""
    d = Path(d)
    meta = json.loads((d / "meta.json").read_text())
    rows = A.parse_dis((d / "driver.dis").read_text())
    funcs = set((d / "functions.txt").read_text().split())
    import textdet_model as TM
    table = []
    for run in meta["runs"]:
        w, h = run["size"]
        tag = f"{run['shape']}_{run['scale']}_{w}x{h}_{run['format']}"
        r = cost(d, rows, funcs, tag, run["frames"])
        shape = TM.SHAPES[run["shape"]]
        f = 2 if run["half_above"] and w > run["half_above"] else 1
        table.append({**run, "f": f, "net_input": [-(-w // f), -(-h // f)],
                      "macs": TM.cost(shape, -(-w // f), -(-h // f))["macs"],
                      "insns_per_frame": r["insns_per_frame"], "cycles_per_frame": r["cycles_per_frame"],
                      "ms_per_frame": r["ms_per_frame"], "l1d_misses_per_frame": r["l1d_misses_per_frame"],
                      "l2_misses_per_frame": r["l2_misses_per_frame"],
                      "low_bound_set_by": max(r["low_parts_per_frame"], key=r["low_parts_per_frame"].get),
                      "top_functions": [[fn, round(c)] for fn, c in r["top_functions"]],
                      "negative_count_addresses": r["negative_count_addresses"]})
    return meta, table


def print_table(table, fam):
    print("| shape | family B | scale | frame | net input | MACs | insns/frame | Mcycles @333 (low-high) "
          "| est. ms @333 | est. ms @444 | arena frame | arena up to (the rule's) |")
    print("|---|---|---|---|---|---:|---:|---|---:|---:|---:|---:|")
    for t in table:
        lo, hi = t["cycles_per_frame"][333]
        fb = "yes" if t["shape"] in fam.get(t["scale"], ()) else ""
        print(f"| {t['shape']} | {fb} | {t['scale']} | {t['size'][0]}x{t['size'][1]} {t['format']} | "
              f"{t['net_input'][0]}x{t['net_input'][1]} | {t['macs']:,} | {t['insns_per_frame']:,.0f} | "
              f"{lo / 1e6:.1f}-{hi / 1e6:.1f} | {t['ms_per_frame']['333']:.1f} | {t['ms_per_frame']['444']:.1f} | "
              f"{t['frame_arena_bytes']:,} | {t['arena_bytes']:,} |")


def summary_of(table, nodark, fam, rule):
    """Per (shape, scale): the rule's inputs (the conductor, 2026-10-19 09:5xZ: the arena the rule reads is the up-to
    value vocr_textspot_arena_bytes(m, 960, 544), "arena_up_to_960"; the time is the default call's, dark pass
    included), the frame's own arena and the report-only no-dark time."""
    by = {(t["shape"], t["scale"], tuple(t["size"]), t["format"]): t for t in table}
    nd = {(t["shape"], t["scale"], tuple(t["size"]), t["format"]): t for t in nodark}
    out = []
    for (shape, scale, size, fmt), t in by.items():
        if size != (960, 544) or fmt != "grey":
            continue
        small = by.get((shape, scale, (480, 272), fmt))
        n = nd.get((shape, scale, size, fmt))
        row = {"shape": shape, "scale": scale, "half_above": t["half_above"],
               "family_b": shape in fam.get(scale, ()), "vtsp_bytes": t["blob_bytes"],
               "macs_960": t["macs"], "ms_333_960": t["ms_per_frame"]["333"],
               "ms_444_960": t["ms_per_frame"]["444"],
               "ms_333_960_nodark": n["ms_per_frame"]["333"] if n else None,
               "arena_up_to_960": t["arena_bytes"], "arena_frame_960": t["frame_arena_bytes"]}
        if small:
            row.update({"ms_333_480": small["ms_per_frame"]["333"],
                        "ms_444_480": small["ms_per_frame"]["444"],
                        "arena_up_to_480": small["arena_bytes"], "arena_frame_480": small["frame_arena_bytes"]})
        row["within_limits"] = row["ms_333_960"] <= rule["ms_max_333_960"] and \
            row["arena_up_to_960"] <= rule["arena_max"]
        out.append(row)
    return out


def cmd_report(args):
    meta, table = table_of(args.dir)
    res = {k: v for k, v in meta.items() if k != "runs"}
    res["note"] = ("estimates: qemu-arm instruction counts priced by vita/a9_cycles.py's Cortex-A9 model, not a "
                   "console measurement; est. ms @333 = high bound x 1.0007 / 333 MHz, est. ms @444 = that x 333/444; "
                   "arena bytes are exact (the runtime's functions)")
    res["family_b"] = family_b()
    protocol = json.loads((ROOT / "eval" / "textspot_protocol.json").read_text(encoding="utf-8"))
    res["rule_reads"] = {
        "arena": "arena_up_to_960 = vocr_textspot_arena_bytes(m, 960, 544): every frame up to 960x544 (for h2, the "
                 "600-px frame read at full size dominates); arena_frame_960 = vocr_textspot_frame_arena_bytes(m, 960, "
                 "544), the 960x544 frame's own need, is report only (the conductor, 2026-10-19)",
        "time": "ms_333_960: the default call (flags 0, the dark pass included) on the pinned 960x544 grey frame; "
                "ms_333_960_nodark (VOCR_TEXTSPOT_NO_DARK) is report only",
        "limits": {"arena_max": protocol["rule"]["arena_max"], "ms_max_333_960": protocol["rule"]["ms_max_333_960"]},
        "within_limits": "ms_333_960 <= ms_max_333_960 and arena_up_to_960 <= arena_max (for det; det+refine adds "
                         "the refinement's arena and 1.4 ms per refined line, eval/textspot_metrics.apply_rule)"}
    res["table"] = table
    res["also"] = []
    for k, d in enumerate(args.also):
        ameta, atable = table_of(d)
        if ameta["sources_sha256"] != meta["sources_sha256"]:
            raise SystemExit(f"--also {d} was measured on other sources")
        label = args.also_label[k] if k < len(args.also_label) else ""
        res["also"].append({"label": label, "computed": ameta["computed"], "flags": ameta["flags"],
                            "vocr_textspot_o_text_bytes": ameta["vocr_textspot_o_text_bytes"], "table": atable})
    nodark = next((a["table"] for a in res["also"] if all(t["flags"] != "0" for t in a["table"])), [])
    res["summary"] = summary_of(table, nodark, res["family_b"], protocol["rule"])
    print("| scale | shape | family B | VTSP B | MACs / 960x544 frame | est. ms 960x544 @333 / @444 (rule) "
          "| no-dark @333 | est. ms 480x272 @333 | arena_up_to_960 (rule) | arena_frame_960 | within limits |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in sorted(res["summary"], key=lambda r: (not r["family_b"], r["scale"] != "h2", r["scale"], r["shape"])):
        nd = f"{r['ms_333_960_nodark']:.1f}" if r["ms_333_960_nodark"] is not None else "-"
        print(f"| {r['scale']} | {r['shape']} | {'yes' if r['family_b'] else ''} | {r['vtsp_bytes']:,} | "
              f"{r['macs_960']:,} | {r['ms_333_960']:.1f} / {r['ms_444_960']:.1f} | {nd} | "
              f"{r.get('ms_333_480', float('nan')):.1f} | {r['arena_up_to_960']:,} | {r['arena_frame_960']:,} | "
              f"{'yes' if r['within_limits'] else 'no'} |")
    print()
    print_table(table, res["family_b"])
    for a in res["also"]:
        print(f"\nalso: {a['label']}")
        print_table(a["table"], res["family_b"])
    if args.json:
        Path(args.json).write_text(dumps(res) + "\n")


ALSO_KEYS = ("shape", "scale", "half_above", "size", "format", "flags", "f", "net_input", "macs", "blob_sha256",
             "frame_sha256", "rc", "boxes", "blocks", "pick", "truncated", "insns_per_frame", "cycles_per_frame",
             "ms_per_frame", "l1d_misses_per_frame", "l2_misses_per_frame", "low_bound_set_by", "frame_arena_bytes",
             "arena_bytes", "top_functions")


def dumps(res):
    """The JSON with every table row on one line (the file stays small and diffs row by row)."""
    def rows(t):
        return "[\n" + ",\n".join("  " + json.dumps(r, separators=(",", ":")) for r in t) + "\n ]"
    head = {k: v for k, v in res.items() if k not in ("table", "summary", "also")}
    out = json.dumps(head, indent=1)[:-2]
    out += ',\n "summary": ' + rows(res["summary"]) + ',\n "table": ' + rows(res["table"])
    also = []
    for a in res["also"]:
        meta = json.dumps({k: v for k, v in a.items() if k != "table"}, separators=(",", ":"))[:-1]
        trimmed = [{k: r[k] for k in ALSO_KEYS if k in r} for r in a["table"]]
        also.append("  " + meta + ',"table":' + rows(trimmed) + "}")
    return out + ',\n "also": [\n' + ",\n".join(also) + "\n ]\n}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--out", required=True)
    m.add_argument("--shapes", default="", help="default: the 21 shapes of #97's 34 runs (family B's among them)")
    m.add_argument("--scales", default="h2,f1")
    m.add_argument("--sizes", default=",".join(f"{w}x{h}" for w, h in SIZES))
    m.add_argument("--formats", default="grey", help="grey and/or rgba")
    m.add_argument("--nodark", action="store_true", help="VOCR_TEXTSPOT_NO_DARK (report only)")
    m.add_argument("--cflags", default="-O2", help="the spotter's Vita flags, e.g. '-Os -DVOCR_NO_NEON'")
    m.add_argument("--seed", type=int, default=1)
    m.add_argument("--jobs", type=int, default=5)
    m.add_argument("--vitasdk")
    m.add_argument("--arm-cc")
    m.add_argument("--qemu")
    m.set_defaults(fn=cmd_measure)
    r = sub.add_parser("report")
    r.add_argument("dir")
    r.add_argument("--json")
    r.add_argument("--also", action="append", default=[], help="more measured directories (same sources)")
    r.add_argument("--also-label", action="append", default=[])
    r.set_defaults(fn=cmd_report)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
