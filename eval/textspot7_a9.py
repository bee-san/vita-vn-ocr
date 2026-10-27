#!/usr/bin/env python3
"""Cortex-A9 estimates of textspot7's added costs (issue #128, eval/textspot7_protocol.json "rule.cost_of_F",
"cost_of_D"): Rubi's finder as the pinned page path runs it (rubi,fix=1: TEXTBOX_RECOVERY, the short-line probe,
fix 1 = COLGAP, the refinement on PSP titles) on development regions, and runtime/vocr_textspot7.c's post-pass and
fallback. textspot6's method (eval/textspot_a9.py, eval/textspot6_a9.py): the Vita compiler's machine code
(arm-vita-eabi-gcc, Thumb-2, -O2) linked into a host driver by an ARM Linux toolchain, run under the plugin qemu-arm
with the Vita's caches, a dry run (--empty) subtracted, priced by vita/a9_cycles.py: est. ms at 333 MHz = high bound
x 1.0007 / 333 MHz, at 444 MHz that x 333 / 444. Estimates only: nothing here ran on a console.

  textspot7_a9.py grad --out DIR [--pinned REPO] [--vitasdk DIR] [--arm-cc GCC] [--qemu DIR]
      rubi,fix=1 on: the largest of #119's 387 base regions (console view: a PSP region upscaled 2x, the refinement
      on), the five largest one by one, all 387 (the mean), the median one, and a 960 x 544 region of textspot's
      pinned procedural frame (eval/textspot_a9.frame, seed 1) as a Vita region and as a PSP-title region
  textspot7_a9.py t7 --out DIR --batch T7.bin [--name NAME] [--units N] [...]
      vocr_textspot7.c on a batch of vocr_textspot7_test.c records (eval/textspot7.py a9-batch writes them): the
      mean per record; with --one RECORDS.bin files, each alone
"""
import argparse
import hashlib
import json
import os
import pickle
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "vita", ROOT / "synth", ROOT / "train"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import a9_cycles as A  # noqa: E402

HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
ART = HERD / "shared" / "artifacts"
PINNED = ART / "vocr6-spotter" / "pagepath_snap135" / "repo"
DEV_UNITS = ART / "vocr5-analysis" / "work" / "units.pkl"
ARM_CC = ART / "runtime" / "tools" / "arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-linux-gnueabihf" / "bin" / \
    "arm-none-linux-gnueabihf-gcc"
CAL = 1.0007
RUNTIME = ("vocr_lines.c", "vocr_lines_rows.c", "vocr_lines_fix.c", "vocr_lines_refine.c")
NOT_PRICED = {"main", "slurp", "i32at", "grad_empty", "t7_batch", "rd32", "load"}
TEXTBOX_RECOVERY, FIX, PROBE, REFINE_ALL = 3, 1, 1, 0x1F


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tool_version(cmd):
    try:
        return subprocess.run([cmd, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    except (OSError, IndexError):
        return None


def build(out, pinned, vitasdk, arm_cc):
    """Vita objects: the pinned runtime's line finder files, runtime/vocr_textspot7.c, and the driver (which includes
    the pinned eval/pagepath_find.c); linked statically by the ARM Linux toolchain."""
    vcc = Path(vitasdk) / "bin/arm-vita-eabi-gcc"
    prt = Path(pinned) / "runtime"
    inc = [f"-I{prt}", f"-I{ROOT / 'runtime'}", f"-I{Path(pinned) / 'eval'}"]
    objs, srcs = [], []
    for src in [prt / s for s in RUNTIME] + [ROOT / "runtime" / "vocr_textspot7.c",
                                              ROOT / "eval" / "textspot7_a9_driver.c"]:
        o = out / (src.stem + "_vita.o")
        defs = ["-DPAGEPATH_FIX"] if src.name == "textspot7_a9_driver.c" else []
        subprocess.run([str(vcc), "-std=c99", "-Wall", "-Wextra", *A.ARCH, "-O2", *defs, *inc, "-c", "-o", str(o),
                        str(src)], check=True)
        objs.append(o)
        srcs.append(src)
    drv = out / "textspot7_a9_driver"
    subprocess.run([arm_cc, *A.ARCH, "-O2", "-static", "-Wl,--no-enum-size-warning", "-o", str(drv),
                    *map(str, objs), str(ROOT / "eval" / "textspot7_a9_shim.c")], check=True, capture_output=True)
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
    funcs -= NOT_PRICED
    (out / "functions.txt").write_text("\n".join(sorted(funcs)) + "\n")
    tools = {"vita_cc": tool_version(str(vcc)), "arm_cc": tool_version(arm_cc),
             "sources_sha256": {str(s): sha256(s) for s in srcs}}
    return drv, rows, funcs, tools


def run_qemu(q, drv, stage, mode, batch, counts, cache, classes):
    plugins = ["-plugin", f"{q}/plugins/libinsn_mix.so,out={counts},classes={classes}",
               "-plugin", f"{q}/plugins/libcache.so,{A.CACHE_ARGS}", "-d", "plugin", "-D", str(cache)]
    p = subprocess.run([str(q / "bin/qemu-arm"), "-cpu", "cortex-a9", "-seed", "0", *plugins, f"./{drv.name}", mode,
                        batch.name], capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=14400,
                       env={}, cwd=stage)
    if p.returncode:
        raise RuntimeError(f"qemu {mode} failed ({p.returncode}): {p.stderr[-2000:]}")
    return p.stdout


def price(out, rows, funcs, tag, frames):
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
    return {"frames": frames, "ms_333": ms333, "ms_444": ms333 * 333 / 444,
            "insns_per_frame": r["insns_per_frame"], "top_functions": r["top_functions"][:6],
            "negative_count_addresses": neg}


def measure(out, q, drv, rows, funcs, tag, batch_bytes, frames, modes):
    stage = Path(tempfile.mkdtemp(prefix="vt7-a9-", dir="/tmp"))
    try:
        shutil.copy2(drv, stage / drv.name)
        b = stage / "batch.bin"
        b.write_bytes(batch_bytes)
        res = {}
        for mode, k in ((modes[0], "dry"), (modes[1], "batch")):
            res[k] = run_qemu(q, drv, stage, mode, b, out / f"counts_{tag}_{k}.txt", out / f"cache_{tag}_{k}.txt",
                              out / "classes.txt")
        r = price(out, rows, funcs, tag, frames)
        r.update(batch_sha256=hashlib.sha256(batch_bytes).hexdigest(), stdout_lines=len(res["batch"].splitlines()))
        return r
    finally:
        shutil.rmtree(stage, ignore_errors=True)


def grad_record(g, psp):
    h, w = g.shape
    return struct.pack("<8I", w, h, TEXTBOX_RECOVERY, FIX, PROBE, REFINE_ALL if psp else 0, 0, 0) + \
        np.ascontiguousarray(g, np.uint8).tobytes()


def dev_regions():
    import cv2
    units = pickle.load(open(DEV_UNITS, "rb"))["units"]
    out = []
    for u in units:
        if u["variant"] != "base":
            continue
        g = u["image"]
        psp = u["kind"] == "psp"
        if psp:
            g = cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
        out.append({"unit": u["unit"], "vn": u["vn"], "psp": psp, "grey": np.ascontiguousarray(g)})
    return out


def tools_of(a):
    vitasdk = a.vitasdk or os.environ.get("VITASDK") or str(ART / "vita" / "vitasdk")
    arm_cc = a.arm_cc or os.environ.get("ARM_CC") or (str(ARM_CC) if ARM_CC.exists() else A.default_arm_cc())
    q = Path(a.qemu) if a.qemu else (ART / "vita" / "qemu" if (ART / "vita" / "qemu" / "bin" / "qemu-arm").exists()
                                     else A.default_qemu())
    return vitasdk, arm_cc, q


def cmd_grad(a):
    import textspot_a9 as TA
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    vitasdk, arm_cc, q = tools_of(a)
    drv, rows, funcs, tools = build(out, a.pinned, vitasdk, arm_cc)
    regs = dev_regions()
    px = [r["grey"].size for r in regs]
    order = sorted(range(len(regs)), key=lambda i: (-px[i], i))
    med = sorted(range(len(regs)), key=lambda i: (px[i], i))[(len(regs) - 1) // 2]
    jobs = {}
    for k, i in enumerate(order[:5]):
        jobs[f"largest{k + 1}"] = ([i], f"{regs[i]['unit']} {regs[i]['grey'].shape[1]}x{regs[i]['grey'].shape[0]}"
                                        f" {'psp (refine)' if regs[i]['psp'] else 'vita'}")
    jobs["median"] = ([med], f"{regs[med]['unit']} {regs[med]['grey'].shape[1]}x{regs[med]['grey'].shape[0]}")
    jobs["all387"] = (list(range(len(regs))), f"{len(regs)} base regions")
    res = {}
    t0 = time.time()
    for tag, (idx, what) in jobs.items():
        batch = b"".join(grad_record(regs[i]["grey"], regs[i]["psp"]) for i in idx)
        res[tag] = dict(measure(out, q, drv, rows, funcs, tag, batch, len(idx), ("--empty", "--grad")), what=what,
                        pixels=int(sum(px[i] for i in idx) / len(idx)))
        print(f"{tag}: {what}: est. {res[tag]['ms_333']:.2f} ms @333 per region", flush=True)
    g = TA.frame(1, 960, 544)
    for tag, psp in (("frame960_vita", False), ("frame960_psp", True)):
        res[tag] = dict(measure(out, q, drv, rows, funcs, tag, grad_record(g, psp), 1, ("--empty", "--grad")),
                        what=f"960x544 procedural frame (eval/textspot_a9.frame seed 1), {'psp' if psp else 'vita'}",
                        pixels=960 * 544)
        print(f"{tag}: est. {res[tag]['ms_333']:.2f} ms @333", flush=True)
    doc = {"kind": "textspot7-a9-grad", "finder": "rubi,fix=1 (pinned eval/pagepath_find.c -DPAGEPATH_FIX: "
                                                  "TEXTBOX_RECOVERY, probe 1, fix 1, refine ALL on PSP regions)",
           "method": "eval/textspot_a9.py's: Vita -O2 objects, qemu-arm plugins, dry run subtracted, high bound x "
                     f"{CAL} / MHz", "estimates": True, "tools": tools, "seconds": round(time.time() - t0, 1),
           "rows": res, "largest_ms_333": max(res[k]["ms_333"] for k in res if k.startswith("largest"))}
    (out / "a9_grad.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"grad a9 -> {out / 'a9_grad.json'}; largest {doc['largest_ms_333']:.2f} ms @333")


def cmd_t7(a):
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    vitasdk, arm_cc, q = tools_of(a)
    drv, rows, funcs, tools = build(out, a.pinned, vitasdk, arm_cc)
    res = {}
    batch = Path(a.batch).read_bytes()
    res["all"] = measure(out, q, drv, rows, funcs, f"{a.name}_all", batch, a.units, ("--empty-t7", "--t7"))
    for k, p in enumerate(a.one or []):
        res[f"one{k + 1}"] = measure(out, q, drv, rows, funcs, f"{a.name}_one{k + 1}", Path(p).read_bytes(), 1,
                                     ("--empty-t7", "--t7"))
    doc = {"kind": "textspot7-a9-t7", "name": a.name, "estimates": True, "tools": tools, "rows": res,
           "max_one_ms_333": max([v["ms_333"] for k, v in res.items() if k.startswith("one")] or [None] if a.one
                                 else [None])}
    (out / f"a9_t7_{a.name}.json").write_text(json.dumps(doc, indent=1) + "\n")
    print(f"t7 a9 {a.name}: mean est. {res['all']['ms_333']:.4f} ms @333 per record -> {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("grad", cmd_grad), ("t7", cmd_t7)):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True)
        p.add_argument("--pinned", default=str(PINNED))
        p.add_argument("--vitasdk")
        p.add_argument("--arm-cc")
        p.add_argument("--qemu")
        if name == "t7":
            p.add_argument("--batch", required=True)
            p.add_argument("--name", required=True)
            p.add_argument("--units", type=int, required=True)
            p.add_argument("--one", nargs="*")
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
