#!/usr/bin/env python3
"""Cortex-A9 cycles per line for runtime/vocr.c as the Vita compiler builds it, as a range.

  a9_cycles.py measure --out DIR [--lines 40] [--modes w8i,w8f,f32] [--no-cache]
  a9_cycles.py report DIR [--json OUT]

measure:
  1. compiles runtime/vocr.c with VitaSDK's arm-vita-eabi-gcc and the Vita flags (Thumb-2,
     -O2 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard), and links that object into
     runtime/tools/vocr_cli.c with an ARM Linux toolchain ($ARM_CC), so qemu-arm runs exactly
     the Vita's machine code for the runtime;
  2. runs it under the plugin-enabled qemu-arm (vita/scripts/build_qemu_plugins.sh) on line
     crops from synth-val and vita-test, once per mode, with vita/qemu/insn_mix.c (executions
     of every instruction, NEON <-> VFP unit switches) and qemu's cache simulator configured
     like the Vita: 32 KiB 4-way L1 D and I caches with 32-byte lines, and the 2 MiB L2;
  3. keeps the counts, the cache statistics and the disassembly in DIR.
report:
  joins the counts with the disassembly, keeps the runtime's own functions (vocr.c, plus the
  memset/memcpy/libgcc helpers it calls), gives every instruction its Cortex-A9 cost and
  prints cycles per CTC frame and per 640-px line (80 frames) as a range:
    low  = max(NEON + VFP issue cycles, instructions / 2 (dual issue), load/store pipe
           cycles, L2->L1 refill bandwidth) + 3 cycles per NEON <-> VFP switch
    high = max(NEON + VFP issue cycles, instructions (no dual issue)) + 10 cycles per
           switch (VLDR/VSTR counted as VFP) + 11 per NEON -> ARM register transfer
           + every L1 miss at the L2 latency + every L2 miss at the DRAM latency
  Issue cycles are the Cortex-A9 NEON MPE TRM's (ARM DDI 0409I, tables 3-2 to 3-10), for
  unaligned NEON loads and stores because gcc emits no alignment hints. Memory latencies:
  L2 20-30 cycles, DRAM 110 ns plus the L2 lookup; the L2->L1 refill rate is 8.7 cycles per
  line (7-cpu.com's Cortex-A9 measurements). These are estimates, not console measurements.
"""
import argparse
import collections
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
HERD = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr"))
MODES = ("w8i", "w8f", "f32")
CLOCKS = (444, 333)
FRAMES_640 = 80
WARN = ["-std=c99", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror", "-Wpedantic", "-Wshadow",
        "-Wstrict-prototypes", "-Wmissing-prototypes", "-Wcast-align", "-Wpointer-arith"]
ARCH = ["-mcpu=cortex-a9", "-mfpu=neon", "-mfloat-abi=hard"]
CACHE_ARGS = ("dcachesize=32768,dassoc=4,dblksize=32,icachesize=32768,iassoc=4,iblksize=32,"
              "l2=on,l2cachesize=2097152,l2assoc=8,l2blksize=32,limit=10")
HELPERS = re.compile(r"^(memset|memcpy|memmove|__aeabi_\w+|__udivsi3|__divsi3|__udivmodsi4|__modsi3|"
                     r"__umodsi3|__aeabi_idiv\w*|__aeabi_uidiv\w*|__gnu_\w+|__memset\w*|__memcpy\w*)")
# Timing assumptions (cycles at the core clock unless noted).
L2_LAT = (20, 30)          # L1 miss that hits L2
DRAM_NS = 110              # L2 miss: DRAM latency on top of the L2 lookup
REFILL = 8.7               # sequential L2 -> L1 line refill rate
DRAIN = (3, 10)            # NEON <-> VFP unit switch
N2C = 11                   # NEON -> ARM register transfer latency (TRM table 3-10)


# ---------------------------------------------------------------- instruction costs

_QREG = re.compile(r"\bq\d+\b")
_SREG = re.compile(r"\bs\d+\b")
_DLANE = re.compile(r"\bd\d+\[\d*\]")
_CORE = re.compile(r"^(r\d+|ip|lr|sp|fp|sl|sb|pc)\b")
LS_EXT = ("vldr", "vstr", "vldm", "vstm", "vpush", "vpop")
FLOAT_2Q = {"vmla", "vmls", "vmul", "vadd", "vsub", "vmax", "vmin", "vabs", "vneg", "vabd", "vcvt",
            "vrecpe", "vrsqrte", "vrecps", "vrsqrts", "vceq", "vcge", "vcgt", "vacge", "vacgt", "vcle",
            "vclt", "vnmul", "vfma"}
INT_Q2 = {"vbsl", "vbit", "vbif", "vswp", "vext", "vcls", "vclz", "vcnt", "vsli", "vsri", "vqshl",
          "vrshl", "vqrshl", "vaba"}


def dreg_count(ops):
    """D registers in a register list such as {d16-d17} or {d16, d17, d18} or {d16[]}."""
    m = re.search(r"\{([^}]*)\}", ops)
    if not m:
        return 1
    n = 0
    for part in m.group(1).split(","):
        part = part.strip()
        r = re.match(r"([dsq])(\d+)(?:\[\d*\])?(?:-[dsq](\d+))?", part)
        if not r:
            continue
        lo, hi = int(r.group(2)), int(r.group(3) or r.group(2))
        k = hi - lo + 1
        n += 2 * k if r.group(1) == "q" else (k + 1) // 2 if r.group(1) == "s" else k
    return max(1, n)


def classify(mn, ops):
    """(unit, issue cycles in that unit, load/store pipe cycles, flags) of one instruction.
    unit: N (Advanced SIMD), V (VFP), L (extension-register load/store), I (integer)."""
    m = mn.lower().split(".")
    base, dts = m[0], [d for d in m[1:] if d not in ("w", "n")]
    dt = dts[0] if dts else ""
    flags = set()
    if base.startswith("v"):
        if base.startswith(LS_EXT):
            # VLDR/VSTR move one register (1 cycle when 64-bit aligned, as stack slots are);
            # VLDM/VSTM/VPUSH/VPOP take one cycle per D register (TRM table 3-3)
            n = dreg_count(ops) if base.startswith(("vldm", "vstm", "vpush", "vpop")) else 1
            return "L", n, n, flags
        if re.match(r"v(ld|st)[1-4]", base):
            aligned = re.search(r":\d+\]", ops) is not None
            n = dreg_count(ops)
            lane = "[" in ops.split("}")[0]
            dup = "[]" in ops
            load = base.startswith("vld")
            if lane and not dup:
                issue = (2 if aligned else 3) if load else (1 if aligned else 2)
            elif dup:
                issue = 1 if aligned else 2
            else:
                issue = {1: 2, 2: 2, 3: 3, 4: 3}.get(n, n) - (1 if aligned else 0)
            if base[3] in "234" and not lane:
                issue = max(issue, {2: 2, 3: 4, 4: 4}.get(int(base[3]), 2) - (1 if aligned else 0))
            return "N", issue, n, flags
        is_vfp = (base in ("vmrs", "vmsr", "vcmp", "vcmpe", "vdiv", "vsqrt") or "f64" in dts
                  or _SREG.search(ops) is not None
                  or (base == "vmov" and not dts and not _QREG.search(ops) and _CORE.match(ops.strip())
                      and "[" not in ops)
                  or (base == "vmov" and not dts and re.match(r"d\d+,\s*(r\d+|ip|lr)", ops.strip())))
        if is_vfp:
            if base == "vdiv":
                return "V", 20 if "f64" in dts else 10, 0, flags
            if base == "vsqrt":
                return "V", 28 if "f64" in dts else 13, 0, flags
            if "f64" in dts and base in ("vmul", "vmla", "vmls", "vnmul", "vnmla", "vnmls"):
                return "V", 2, 0, flags
            return "V", 1, 0, flags
        if base == "vmov" and _CORE.match(ops.strip()) and _DLANE.search(ops):
            flags.add("n2c")
            return "N", 1, 0, flags
        q = _QREG.search(ops) is not None
        if base in FLOAT_2Q and dt.startswith("f"):
            return "N", 2 if q else 1, 0, flags
        if base in ("vcvt",):
            return "N", 2 if q else 1, 0, flags
        if base in ("vmull", "vmlal", "vmlsl", "vqdmull", "vqdmlal", "vqdmlsl"):
            return "N", 2 if dt.endswith("32") else 1, 0, flags
        if base in ("vmul", "vmla", "vmls", "vqdmulh", "vqrdmulh"):
            if dt.endswith("32"):
                return "N", 4 if q else 2, 0, flags
            return "N", 2 if q else 1, 0, flags
        if base in ("vzip", "vuzp"):
            return "N", 3 if q else 1, 0, flags
        if base in INT_Q2 or (base in ("vshl",) and len(ops.split(",")) == 3 and "#" not in ops):
            return "N", 2 if q else 1, 0, flags
        if base in ("vtbl", "vtbx"):
            return "N", 2 if dreg_count(ops) <= 2 else 3, 0, flags
        return "N", 1, 0, flags
    b = base.rstrip("sw") if base not in ("subs", "adds", "movs") else base
    if re.match(r"^(ldr|str|ldrex|strex)", base) or base in ("ldrd", "strd"):
        return "I", 1, 1, {"ls"}
    if re.match(r"^(ldm|stm|push|pop)", base):
        n = len(re.findall(r"\b(r\d+|ip|lr|sp|fp|sl|sb|pc)\b", ops.split("{", 1)[-1]))
        c = max(1, (n + 1) // 2)
        return "I", c, c, {"ls"}
    if re.match(r"^(b|bl|blx|bx|cbz|cbnz|tbb|tbh)(eq|ne|cs|cc|hs|lo|mi|pl|vs|vc|hi|ls|ge|lt|gt|le|al)?$", base):
        return "I", 1, 0, {"br"}
    if re.match(r"^(mul|mla|mls|smull|umull|smlal|umlal|smul|smla|umaal)", b):
        return "I", 2 if base.startswith(("smull", "umull", "smlal", "umlal")) else 1, 0, {"mul"}
    if re.match(r"^it[te]*$", base):
        return "I", 0, 0, {"it"}
    return "I", 1, 0, {"alu"}


# ---------------------------------------------------------------- measure

def default_arm_cc():
    for p in (ROOT / "runtime/scripts/tools", HERD / "shared/artifacts/runtime/tools"):
        cc = p / "arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-linux-gnueabihf/bin/arm-none-linux-gnueabihf-gcc"
        if cc.exists():
            return str(cc)
    return "arm-none-linux-gnueabihf-gcc"


def default_qemu():
    for p in (HERE / "tools/qemu", HERD / "shared/artifacts/vita/qemu"):
        if (p / "bin/qemu-arm").exists():
            return p
    raise SystemExit("no plugin-enabled qemu: run vita/scripts/build_qemu_plugins.sh")


def build_cli(out, vitasdk, arm_cc):
    vcc = Path(vitasdk) / "bin/arm-vita-eabi-gcc"
    obj, cli = out / "vocr_vita.o", out / "vocr_cli_vitacc"
    subprocess.run([str(vcc), *WARN, *ARCH, "-O2", "-c", "-o", str(obj), str(ROOT / "runtime/vocr.c")], check=True)
    # VitaSDK builds with short enums; vocr.h passes no enum across the API, so the linker's
    # enum-size warning does not apply.
    subprocess.run([arm_cc, *WARN, *ARCH, "-O2", "-static", "-Wl,--no-enum-size-warning", "-o", str(cli), str(obj),
                    str(ROOT / "runtime/tools/vocr_cli.c"), "-lm"], check=True, capture_output=True)
    objdump = arm_cc[:-3] + "objdump" if arm_cc.endswith("gcc") else "objdump"
    dis = subprocess.run([objdump, "-d", "--no-show-raw-insn", str(cli)], check=True, capture_output=True,
                         text=True).stdout
    (out / "cli.dis").write_text(dis)
    nm = subprocess.run([str(Path(vitasdk) / "bin/arm-vita-eabi-nm"), "--defined-only", str(obj)], check=True,
                        capture_output=True, text=True).stdout
    funcs = sorted({ln.split()[-1] for ln in nm.splitlines() if ln.split()[1:2] and ln.split()[1] in "tT"})
    (out / "vocr_functions.txt").write_text("\n".join(funcs) + "\n")
    return cli, dis


def parse_dis(dis):
    rows, fn = {}, None
    for line in dis.splitlines():
        m = re.match(r"^([0-9a-f]+) <(.+)>:", line)
        if m:
            fn = m.group(2)
            continue
        m = re.match(r"^\s+([0-9a-f]+):\s+(\S+)\s*([^@;]*)", line)
        if m and fn:
            rows[int(m.group(1), 16)] = (fn, m.group(2), m.group(3).strip())
    return rows


def write_pack(path, n, data_dir):
    """n line crops: half synth-val (grey), half vita-test (RGBA), like the Vita bundle."""
    sys.path[:0] = [str(ROOT / "eval")]
    import fetch
    import numpy as np
    import sets
    crops = [np.ascontiguousarray(ln.crop_rgb[..., 0]) for ln in sets.load_synth_val(limit=n - n // 2)]
    try:
        real, _ = sets.load("vita-test", fetch.data_dir(data_dir))
        for ln in real[:n // 2]:
            rgb = ln.crop_rgb
            crops.append(np.ascontiguousarray(np.concatenate([rgb, np.full(rgb.shape[:2] + (1,), 255, np.uint8)], 2)))
    except (FileNotFoundError, OSError) as e:
        print(f"vita-test unavailable ({e}); synth-val only", file=sys.stderr)
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(crops)))
        for c in crops:
            f.write(struct.pack("<III", c.shape[0], c.shape[1], 1 if c.ndim == 2 else 4))
            f.write(c.tobytes())
    return len(crops)


def export_blobs(out, ckpt):
    sys.path[:0] = [str(ROOT / "train")]
    import export
    model, vocab, height = export.load_checkpoint(ckpt)
    paths = {}
    for key, int8 in (("f32", False), ("w8", True)):
        data, _ = export.build_blob(export.model_ops(model, height), vocab, height, int8)
        paths[key] = out / f"model_{key}.vocr"
        paths[key].write_bytes(data)
    return paths


def measure(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    vitasdk = args.vitasdk or os.environ.get("VITASDK")
    if not vitasdk:
        raise SystemExit("set VITASDK (vita/scripts/fetch_vitasdk.sh)")
    cli, dis = build_cli(out, vitasdk, args.arm_cc or os.environ.get("ARM_CC") or default_arm_cc())
    rows = parse_dis(dis)
    with open(out / "classes.txt", "w") as f:
        for a, (_, mn, ops) in sorted(rows.items()):
            unit = classify(mn, ops)[0]
            if unit != "I":
                f.write(f"{a:x} {unit}\n")
    pack = out / "lines.bin"
    n_lines = write_pack(pack, args.lines, args.eval_data)
    blobs = export_blobs(out, args.ckpt)
    q = Path(args.qemu) if args.qemu else default_qemu()
    meta = {"lines": n_lines, "ckpt": str(args.ckpt), "modes": {}}
    for mode in args.modes.split(","):
        blob = blobs["f32" if mode == "f32" else "w8"]
        plugins = ["-plugin", f"{q}/plugins/libinsn_mix.so,out={out}/counts_{mode}.txt,classes={out}/classes.txt"]
        if args.cache:
            plugins += ["-plugin", f"{q}/plugins/libcache.so,{CACHE_ARGS}", "-d", "plugin", "-D",
                        str(out / f"cache_{mode}.txt")]
        p = subprocess.run([str(q / "bin/qemu-arm"), "-cpu", "cortex-a9", *plugins, str(cli), str(blob), "--mode", mode,
                            "--pack", str(pack)], capture_output=True, text=True, stdin=subprocess.DEVNULL,
                           timeout=7200)
        if p.returncode:
            raise SystemExit(f"qemu run failed for {mode}: {p.stderr[-2000:]}")
        m = re.search(r"(\d+) lines, (\d+) frames", p.stderr)
        meta["modes"][mode] = {"lines": int(m.group(1)), "frames": int(m.group(2)),
                               "texts": p.stdout.splitlines()}
        print(f"{mode}: {m.group(0)}", file=sys.stderr)
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))


# ---------------------------------------------------------------- report

def read_counts(path):
    counts, extra = {}, {}
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            k, v = line[1:].split()
            extra[k] = int(v)
        elif line.strip():
            a, c = line.split()
            counts[int(a, 16)] = int(c)
    return counts, extra


def read_cache(path):
    if not Path(path).exists():
        return None
    lines = Path(path).read_text().splitlines()
    for i, line in enumerate(lines):
        if line.startswith("core #"):
            v = lines[i + 1].split()
            return {"data_accesses": int(v[1]), "l1d_misses": int(v[2]), "insn_accesses": int(v[4]),
                    "l1i_misses": int(v[5]), "l2_accesses": int(v[7]), "l2_misses": int(v[8])}
    return None


def model_mode(rows, counts, extra, funcs, frames, cache):
    tot = collections.Counter()
    per_fn = collections.Counter()
    mix = collections.Counter()
    for a, c in counts.items():
        fn, mn, ops = rows.get(a, ("?", "?", ""))
        if fn not in funcs and not HELPERS.match(fn):
            continue                                   # the CLI's own code
        unit, issue, ls, flags = classify(mn, ops)
        per_fn[fn] += c
        mix[mn.split(".")[0]] += c
        tot["insns"] += c
        tot[f"issue_{unit}"] += issue * c
        tot[f"n_{unit}"] += c
        tot["ls"] += ls * c
        if "n2c" in flags:
            tot["n2c"] += c
        for f in flags & {"alu", "br", "mul"}:
            tot[f] += c
    fp = tot["issue_N"] + tot["issue_V"] + tot["issue_L"]
    sw_lo, sw_hi = extra.get("switches_l_free", 0), extra.get("switches_l_vfp", 0)
    l1m = cache["l1d_misses"] if cache else 0
    l2m = cache["l2_misses"] if cache else 0
    low_parts = {"neon_vfp_issue": fp, "dual_issue": tot["insns"] / 2, "ls_pipe": tot["ls"],
                 "int_pipes": (tot["alu"] + tot["br"] + tot["mul"]) / 2, "l2_refill": l1m * REFILL}
    res = {"frames": frames, "insns_per_frame": tot["insns"] / frames,
           "neon_insns_per_frame": tot["n_N"] / frames, "vfp_insns_per_frame": tot["n_V"] / frames,
           "ext_ls_insns_per_frame": tot["n_L"] / frames,
           "neon_issue_per_frame": tot["issue_N"] / frames, "vfp_issue_per_frame": tot["issue_V"] / frames,
           "ext_ls_issue_per_frame": tot["issue_L"] / frames,
           "switches_per_frame": [sw_lo / frames, sw_hi / frames], "n2c_per_frame": tot["n2c"] / frames,
           "l1d_misses_per_frame": l1m / frames, "l2_misses_per_frame": l2m / frames,
           "low_parts_per_frame": {k: v / frames for k, v in low_parts.items()},
           "top_functions": [(f, c / frames) for f, c in per_fn.most_common(8)],
           "top_mnemonics": [(m, c / frames) for m, c in mix.most_common(12)]}
    stall_hi = {mhz: l1m * L2_LAT[1] + l2m * (L2_LAT[1] + DRAM_NS * mhz / 1000) for mhz in CLOCKS}
    low = max(low_parts.values()) + sw_lo * DRAIN[0]
    res["cycles_per_frame"] = {}
    for mhz in CLOCKS:
        high = max(fp, tot["insns"]) + sw_hi * DRAIN[1] + tot["n2c"] * N2C + stall_hi[mhz]
        res["cycles_per_frame"][mhz] = [low / frames, high / frames]
    return res


def report(args):
    d = args.dir
    meta = json.loads((d / "meta.json").read_text())
    rows = parse_dis((d / "cli.dis").read_text())
    funcs = set((d / "vocr_functions.txt").read_text().split())
    out = {"lines": meta["lines"], "modes": {}}
    for mode, mm in meta["modes"].items():
        counts, extra = read_counts(d / f"counts_{mode}.txt")
        r = model_mode(rows, counts, extra, funcs, mm["frames"], read_cache(d / f"cache_{mode}.txt"))
        out["modes"][mode] = r
    print(f"{meta['lines']} lines; Vita machine code (arm-vita-eabi-gcc, Thumb-2) under qemu-arm, Cortex-A9 costs")
    print("| mode | insns/frame | NEON issue | VFP issue | VLDR/VSTR | switches | L1D miss | L2 miss "
          "| cycles/frame | 640-px line at 444 MHz | at 333 MHz |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for mode, r in out["modes"].items():
        lo, hi = r["cycles_per_frame"][444]
        lo3, hi3 = r["cycles_per_frame"][333]
        ms = lambda c, mhz: c * FRAMES_640 / (mhz * 1000)  # noqa: E731
        print(f"| {mode} | {r['insns_per_frame']:,.0f} | {r['neon_issue_per_frame']:,.0f} | "
              f"{r['vfp_issue_per_frame']:,.0f} | {r['ext_ls_insns_per_frame']:,.0f} | "
              f"{r['switches_per_frame'][0]:,.0f}-{r['switches_per_frame'][1]:,.0f} | "
              f"{r['l1d_misses_per_frame']:,.0f} | {r['l2_misses_per_frame']:,.0f} | {lo:,.0f}-{hi:,.0f} | "
              f"{ms(lo, 444):.0f}-{ms(hi, 444):.0f} ms | {ms(lo3, 333):.0f}-{ms(hi3, 333):.0f} ms |")
        r["ms_640px"] = {mhz: [ms(c, mhz) for c in r["cycles_per_frame"][mhz]] for mhz in CLOCKS}
    for mode, r in out["modes"].items():
        print(f"\n{mode}: low bound set by {max(r['low_parts_per_frame'], key=r['low_parts_per_frame'].get)}; "
              + ", ".join(f"{k} {v:,.0f}" for k, v in r["low_parts_per_frame"].items()))
        print("  functions: " + ", ".join(f"{f} {c:,.0f}" for f, c in r["top_functions"]))
        print("  mnemonics: " + ", ".join(f"{m} {c:,.0f}" for m, c in r["top_mnemonics"]))
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--lines", type=int, default=40)
    m.add_argument("--modes", default=",".join(MODES))
    m.add_argument("--ckpt", type=Path, default=HERD / "shared/models/v1/ckpt_best.pt")
    m.add_argument("--eval-data", type=Path)
    m.add_argument("--vitasdk")
    m.add_argument("--arm-cc")
    m.add_argument("--qemu")
    m.add_argument("--no-cache", dest="cache", action="store_false")
    r = sub.add_parser("report")
    r.add_argument("dir", type=Path)
    r.add_argument("--json", type=Path)
    args = ap.parse_args()
    measure(args) if args.cmd == "measure" else report(args)


if __name__ == "__main__":
    main()
