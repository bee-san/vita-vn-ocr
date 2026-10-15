#!/usr/bin/env python3
"""Cortex-A9 cycles per layer of runtime/vocr.c as the Vita compiler builds it, calibrated to the console.

  a9_layers.py measure --blob BLOB --pack PACK --out DIR [--mode w8i] [--min-frames F] [--max-frames F]
                       [--limit N] [--cli-opts '--tile 8'] [--runtime DIR] [--plain] [--no-pld]
  a9_layers.py report DIR [--json OUT] [--scale 1.000676 | --console-ms 4.9417 --console-mhz 333]
  a9_layers.py compare BEFORE_DIR AFTER_DIR [--scale 1.000676 | --console-ms 4.9417 --console-mhz 333]

measure: compiles runtime/vocr.c (or --runtime DIR/vocr.c) with VitaSDK's arm-vita-eabi-gcc and the Vita
  flags (vita/a9_cycles.py), plus -DVOCR_PROF=vocr_prof_mark, which makes vocr_run() call the marker of
  vita/qemu/prof_mark.c as each phase starts (an op's step, the head, the input columns, the rest), and runs
  it under qemu-arm with vita/qemu/insn_mix.c's per-phase counts (marker=, layers=) and its data cache model
  (l1d=on: 32 KiB 4-way L1, 2 MiB 8-way L2, 32-byte lines, LRU). --plain also builds and runs the same code
  without the markers, to check that they do not change the runtime's own instruction count; --no-pld leaves
  the PLD fills out of the cache model (the bound if PLD does nothing). The lines are a vocr_cli pack
  (vita/dev_pack.py), optionally only those of [--min-frames, --max-frames] CTC frames.
report: per phase, vita/a9_cycles.py's cost model (the Cortex-A9 NEON MPE TRM's issue cycles, the same low and
  high bounds), per CTC frame. The console calibration (docs/ocr-speed.md): VOCR3-H15's lines of 60-100 CTC
  frames ran at a frame-weighted 4,942 us per frame in vita-jp-overlay #81's v0.8.0-test.1 log (10 lines;
  the 11 jobs' 5,068 us is recognize time over frames, strip reads included); at an assumed 333 MHz (the log
  does not record the clock) that is 1.646 M cycles per frame, 1.0007 x the high bound of the code the console
  ran (vita-vn-ocr a5d5aad) on 60-100-frame development lines. --console-ms/--console-mhz scale every phase's
  high bound by the one factor that makes their sum equal such a figure (for the model and code it belongs
  to); --scale gives the factor directly, e.g. 1.000676 for any model on this runtime. All of it is an
  estimate except the console figure itself.
compare: the per-phase table of two measurements of the same pack, before -> after.
"""
import argparse
import collections
import hashlib
import json
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "train")]
import a9_cycles as A9  # noqa: E402

PROF_RUN, PROF_SRC = 100, 101
NAMES = {1: "CONV", 2: "DW", 3: "PW", 4: "HEAD"}


def plugin_dir(qemu_root):
    """The plugin built from this checkout's vita/qemu/insn_mix.c (marker=/layers=/l1d= need it)."""
    for p in (os.environ.get("VOCR_A9_PLUGINS"), qemu_root / "plugins"):
        if p and (Path(p) / "libinsn_mix.so").exists():
            return Path(p)
    raise SystemExit("no libinsn_mix.so: run vita/scripts/build_qemu_plugins.sh (or set VOCR_A9_PLUGINS)")


def build(out, vitasdk, arm_cc, runtime, prof):
    vcc = Path(vitasdk) / "bin/arm-vita-eabi-gcc"
    tag = "prof" if prof else "plain"
    obj, cli = out / f"vocr_vita_{tag}.o", out / f"vocr_cli_{tag}"
    lm_obj = out / f"vocr_lm_vita_{tag}.o"
    flags = [*A9.WARN, *A9.ARCH, "-O2"] + (["-DVOCR_PROF=vocr_prof_mark"] if prof else [])
    subprocess.run([str(vcc), *flags, "-c", "-o", str(obj), str(runtime / "vocr.c")], check=True)
    # vocr_cli.c links the CTC beam/LM decoder: the version's vocr_lm.c, else this checkout's (Vita compiler
    # too); only vocr.c's functions are profiled, and the decoder runs only with --lm
    lm_c = runtime / "vocr_lm.c" if (runtime / "vocr_lm.c").exists() else ROOT / "runtime/vocr_lm.c"
    subprocess.run([str(vcc), *A9.WARN, *A9.ARCH, "-O2", "-I", str(ROOT / "runtime"), "-c", "-o", str(lm_obj),
                    str(lm_c)], check=True)
    extra = [str(HERE / "qemu/prof_mark.c")] if prof else []
    subprocess.run([arm_cc, *A9.WARN, *A9.ARCH, "-O2", "-static", "-Wl,--no-enum-size-warning", "-o", str(cli),
                    str(obj), str(lm_obj), *extra, str(ROOT / "runtime/tools/vocr_cli.c"), "-lm"], check=True,
                   capture_output=True)
    objdump = arm_cc[:-3] + "objdump" if arm_cc.endswith("gcc") else "objdump"
    dis = subprocess.run([objdump, "-d", "--no-show-raw-insn", str(cli)], check=True, capture_output=True,
                         text=True).stdout
    (out / f"cli_{tag}.dis").write_text(dis)
    nm = subprocess.run([str(Path(vitasdk) / "bin/arm-vita-eabi-nm"), "--defined-only", str(obj)], check=True,
                        capture_output=True, text=True).stdout
    funcs = sorted({ln.split()[-1] for ln in nm.splitlines() if ln.split()[1:2] and ln.split()[1] in "tT"})
    (out / f"functions_{tag}.txt").write_text("\n".join(funcs) + "\n")
    with open(out / f"classes_{tag}.txt", "w") as f:
        for a, (_, mn, ops) in sorted(A9.parse_dis(dis).items()):
            unit = A9.classify(mn, ops)[0]
            if unit != "I":
                f.write(f"{a:x} {unit}\n")
    marker = None
    for line in dis.splitlines():
        if line.endswith("<vocr_prof_mark>:"):
            marker = int(line.split()[0], 16)
    with open(out / f"pld_{tag}.txt", "w") as f:
        for a, reg, off in pld_sites(A9.parse_dis(dis)):
            f.write(f"{a:x} {reg} {off}\n")
    return cli, marker


PLD_REGS = {**{f"r{i}": i for i in range(13)}, "sb": 9, "sl": 10, "fp": 11, "ip": 12, "sp": 13, "lr": 14}


def pld_sites(rows):
    """The PLD sites for the plugin's cache model (qemu runs PLD as a no-op): (address, base register, offset)
    for every `pld [reg]` / `pld [reg, #imm]` of a parsed disassembly ({address: (function, mnemonic, operands)}),
    in address order. A register-offset form ([r1, r2]) is not modelled."""
    out = []
    for a, (_, mn, ops) in sorted(rows.items()):
        m = re.match(r"\[(\w+)(?:,\s*#(-?\d+))?\]$", ops.strip()) if mn.startswith("pld") else None
        if m and m.group(1) in PLD_REGS:
            out.append((a, PLD_REGS[m.group(1)], int(m.group(2) or 0)))
    return out


def subset_pack(src, dst, lo, hi, limit):
    """Lines of src with lo <= CTC frames <= hi (vita/dev_pack.frames), at most limit, into dst."""
    sys.path[:0] = [str(HERE)]
    import dev_pack
    data = Path(src).read_bytes()
    n, off, keep = struct.unpack_from("<I", data, 0)[0], 4, []
    for _ in range(n):
        h, w, ch = struct.unpack_from("<III", data, off)
        size = 12 + h * w * ch
        fr = dev_pack.frames(w, h)
        if lo <= fr <= hi and (not limit or len(keep) < limit):
            keep.append((data[off:off + size], fr))
        off += size
    with open(dst, "wb") as f:
        f.write(struct.pack("<I", len(keep)))
        for blob, _ in keep:
            f.write(blob)
    return len(keep), sum(fr for _, fr in keep)


def measure(args):
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    vitasdk = args.vitasdk or os.environ.get("VITASDK")
    if not vitasdk:
        raise SystemExit("set VITASDK (vita/scripts/fetch_vitasdk.sh)")
    arm_cc = args.arm_cc or os.environ.get("ARM_CC") or A9.default_arm_cc()
    q = Path(args.qemu) if args.qemu else A9.default_qemu()
    plugins = plugin_dir(q)
    runtime = Path(args.runtime).resolve() if args.runtime else ROOT / "runtime"
    stage = Path(tempfile.mkdtemp(prefix="vocr-a9-", dir="/tmp"))   # fixed-length real path (a9_cycles.py)
    try:
        pack = stage / "lines.bin"
        n_lines, n_frames = subset_pack(args.pack, pack, args.min_frames, args.max_frames, args.limit)
        blob = stage / "model.vocr"
        shutil.copy2(args.blob, blob)
        meta = {"blob": Path(args.blob).name, "blob_path": str(Path(args.blob).resolve()),
                "blob_sha256": hashlib.sha256(blob.read_bytes()).hexdigest(),
                "pack_sha256": hashlib.sha256(pack.read_bytes()).hexdigest(), "lines": n_lines,
                "frames_expected": n_frames, "mode": args.mode, "cli_opts": args.cli_opts,
                "runtime": str(runtime), "vocr_c_sha256": hashlib.sha256((runtime / "vocr.c").read_bytes()).hexdigest(),
                "pld_model": not args.no_pld, "runs": {}}
        for prof in ([True, False] if args.plain else [True]):
            tag = "prof" if prof else "plain"
            cli, marker = build(out, vitasdk, arm_cc, runtime, prof)
            shutil.copy2(cli, stage / cli.name)
            opts = [f"out={out}/counts_{tag}.txt", f"classes={out}/classes_{tag}.txt"]
            if prof:
                if marker is None:
                    raise SystemExit("no vocr_prof_mark in the profiling build")
                opts += [f"marker={marker:x}", f"layers={out}/layers.txt", "l1d=on"]
                if not args.no_pld:      # --no-pld: the cache model sees demand accesses only (PLD does nothing)
                    opts.append(f"pld={out}/pld_{tag}.txt")
            p = subprocess.run([str(q / "bin/qemu-arm"), "-cpu", "cortex-a9", "-seed", "0",
                                "-plugin", f"{plugins}/libinsn_mix.so," + ",".join(opts),
                                f"./{cli.name}", blob.name, "--mode", args.mode, "--pack", pack.name,
                                *shlex.split(args.cli_opts)],
                               capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=14400,
                               env={}, cwd=stage)
            if p.returncode:
                raise SystemExit(f"qemu run failed ({tag}): {p.stderr[-2000:]}")
            m = re.search(r"(\d+) lines, (\d+) frames", p.stderr)
            meta["runs"][tag] = {"lines": int(m.group(1)), "frames": int(m.group(2)),
                                 "texts_sha256": hashlib.sha256(p.stdout.encode()).hexdigest()}
            (out / f"texts_{tag}.txt").write_text(p.stdout)
            print(f"{tag}: {m.group(0)}", file=sys.stderr)
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))


def read_layers(path):
    counts, extra = collections.defaultdict(dict), collections.defaultdict(dict)
    for line in Path(path).read_text().splitlines():
        if line.startswith("#"):
            _, i, k, v = line.split()
            extra[int(i)][k] = int(v)
        elif line.strip():
            i, a, c = line.split()
            counts[int(i)][int(a, 16)] = int(c)
    return counts, extra


def phase_cost(rows, counts, ex, funcs, mhz):
    """vita/a9_cycles.model_mode's bounds for one phase (cycles, not per frame)."""
    tot, per_fn = collections.Counter(), collections.Counter()
    for a, c in counts.items():
        fn, mn, ops = rows.get(a, ("?", "?", ""))
        if fn not in funcs and not A9.HELPERS.match(fn):
            continue
        unit, issue, ls, flags = A9.classify(mn, ops)
        per_fn[fn] += c
        tot["insns"] += c
        tot[f"issue_{unit}"] += issue * c
        tot["ls"] += ls * c
        if "n2c" in flags:
            tot["n2c"] += c
        for f in flags & {"alu", "br", "mul"}:
            tot[f] += c
    fp = tot["issue_N"] + tot["issue_V"] + tot["issue_L"]
    l1m, l2m = ex.get("l1d_misses", 0), ex.get("l2_misses", 0)
    sw_lo, sw_hi = ex.get("switches_l_free", 0), ex.get("switches_l_vfp", 0)
    low = max(fp, tot["insns"] / 2, tot["ls"], (tot["alu"] + tot["br"] + tot["mul"]) / 2, l1m * A9.REFILL) \
        + sw_lo * A9.DRAIN[0]
    high = (max(fp, tot["insns"]) + sw_hi * A9.DRAIN[1] + tot["n2c"] * A9.N2C + l1m * A9.L2_LAT[1]
            + l2m * (A9.L2_LAT[1] + A9.DRAM_NS * mhz / 1000))
    return {"insns": tot["insns"], "neon_issue": tot["issue_N"], "vfp_issue": tot["issue_V"],
            "ls_ext": tot["issue_L"], "switches": [sw_lo, sw_hi], "n2c": tot["n2c"], "l1d_misses": l1m,
            "l2_misses": l2m, "data_accesses": ex.get("data_accesses", 0), "prefetches": ex.get("prefetches", 0),
            "prefetch_fills": ex.get("prefetch_fills", 0), "low": low, "high": high,
            "functions": per_fn.most_common(4)}


def describe(blob_path):
    import export
    b = export.read_blob(Path(blob_path).read_bytes())
    S, s, rows = b["stride"], 1, {}
    for i, op in enumerate(b["ops"]):
        s *= op["sw"]
        cols = S // s
        if op["type"] in (1, 2):
            macs = cols * op["h_out"] * op["cout"] * op["kh"] * op["kw"]
        else:
            macs = cols * (1 if op["collapse"] else op["h_in"]) * op["k"] * op["cout"]
        shape = (f"{NAMES[op['type']]} {op['kh']}x{op['kw']}/{op['sh']}x{op['sw']} {op['cin']}->{op['cout']} "
                 f"h{op['h_in']}" + (" +res" if op["res"] else "") + (" collapse" if op["collapse"] else ""))
        rows[i] = {"shape": shape, "macs_per_frame": macs}
    rows[PROF_SRC] = {"shape": "input columns (resize, normalise)", "macs_per_frame": 0}
    rows[PROF_RUN] = {"shape": "vocr_run setup, tile loop, cancel", "macs_per_frame": 0}
    return rows


def analyse(d, mhz):
    d = Path(d)
    meta = json.loads((d / "meta.json").read_text())
    rows = A9.parse_dis((d / "cli_prof.dis").read_text())
    funcs = set((d / "functions_prof.txt").read_text().split())
    counts, extra = read_layers(d / "layers.txt")
    frames = meta["runs"]["prof"]["frames"]
    res = {i: phase_cost(rows, counts[i], extra.get(i, {}), funcs, mhz) for i in sorted(set(counts) | set(extra))}
    res = {i: r for i, r in res.items() if r["insns"]}
    return meta, frames, res


def fmt_table(desc, res, frames, mhz, scale):
    lines = ["| phase | shape | MACs/frame | insns/frame | NEON issue | L1D miss | cycles/frame (low-high) "
             f"| calibrated ms/frame at {mhz} MHz | share |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    tot_hi = sum(r["high"] for r in res.values())
    for i, r in sorted(res.items(), key=lambda kv: -kv[1]["high"]):
        dsc = desc.get(i, {"shape": f"id {i}", "macs_per_frame": 0})
        cal = r["high"] * scale / frames / (mhz * 1000)
        lines.append(f"| {i} | {dsc['shape']} | {dsc['macs_per_frame']:,} | {r['insns'] / frames:,.0f} | "
                     f"{r['neon_issue'] / frames:,.0f} | {r['l1d_misses'] / frames:,.1f} | "
                     f"{r['low'] / frames:,.0f}-{r['high'] / frames:,.0f} | {cal:.3f} | {r['high'] / tot_hi:.1%} |")
    lo = sum(r["low"] for r in res.values()) / frames
    hi = tot_hi / frames
    lines.append(f"| all | | {sum(v['macs_per_frame'] for v in desc.values()):,} | "
                 f"{sum(r['insns'] for r in res.values()) / frames:,.0f} | "
                 f"{sum(r['neon_issue'] for r in res.values()) / frames:,.0f} | "
                 f"{sum(r['l1d_misses'] for r in res.values()) / frames:,.1f} | {lo:,.0f}-{hi:,.0f} | "
                 f"{hi * scale / (mhz * 1000):.3f} | |")
    return lines


def calib_scale(res, frames, args):
    """The factor that maps the summed high bound per frame onto the console figure (1.0 without one);
    --scale gives it directly, e.g. the factor VOCR3-H15's console figure gives, for a model the console
    has not run."""
    if getattr(args, "scale", 0.0):
        return args.scale
    if not args.console_ms:
        return 1.0
    hi = sum(r["high"] for r in res.values()) / frames
    return args.console_ms * args.console_mhz * 1000 / hi


def report(args):
    meta, frames, res = analyse(args.dir, args.console_mhz)
    desc = describe_from_meta(args.dir, meta, args)
    scale = calib_scale(res, frames, args)
    print(f"{meta['blob']} ({meta['blob_sha256'][:12]}), {meta['lines']} lines, {frames} frames, mode {meta['mode']}"
          f" {meta['cli_opts']}; vocr.c {meta['vocr_c_sha256'][:12]}")
    if "plain" in meta["runs"]:
        same = meta["runs"]["plain"]["texts_sha256"] == meta["runs"]["prof"]["texts_sha256"]
        print(f"markers off: texts identical: {same}")
    if args.console_ms:
        print(f"calibration: console {args.console_ms} ms/frame at {args.console_mhz} MHz = "
              f"{args.console_ms * args.console_mhz * 1000:,.0f} cycles/frame = {scale:.3f} x the model's high bound")
    print("\n".join(fmt_table(desc, res, frames, args.console_mhz, scale)))
    for i, r in sorted(res.items(), key=lambda kv: -kv[1]["high"])[:8]:
        print(f"  {i}: " + ", ".join(f"{f} {c / frames:,.0f}" for f, c in r["functions"]))
    if args.json:
        Path(args.json).write_text(json.dumps({"meta": meta, "frames": frames, "scale": scale,
                                               "phases": {str(k): v for k, v in res.items()}}, indent=1))


def describe_from_meta(d, meta, args):
    blob = args.blob_path or meta.get("blob_path")
    if blob and Path(blob).exists():
        return describe(blob)
    return {}


def compare(args):
    m0, f0, r0 = analyse(args.before, args.console_mhz)
    m1, f1, r1 = analyse(args.after, args.console_mhz)
    if m0["pack_sha256"] != m1["pack_sha256"] or m0["blob_sha256"] != m1["blob_sha256"]:
        raise SystemExit("compare needs the same pack and blob")
    desc = describe_from_meta(args.before, m0, args)
    scale = calib_scale(r0, f0, args)
    print(f"{m0['blob']}, {m0['lines']} lines, {f0} frames; texts identical: "
          f"{m0['runs']['prof']['texts_sha256'] == m1['runs']['prof']['texts_sha256']}")
    if args.console_ms or getattr(args, "scale", 0.0):
        print(f"calibration: {scale:.3f} x the high bound (BEFORE's), at {args.console_mhz} MHz")
    print("| phase | shape | before cycles/frame | after cycles/frame | before ms/frame | after ms/frame | speed-up |")
    print("|---|---|---:|---:|---:|---:|---:|")
    tb = ta = 0
    for i in sorted(set(r0) | set(r1), key=lambda k: -(r0.get(k, {}).get("high", 0))):
        b, a = r0.get(i, {"high": 0, "low": 0}), r1.get(i, {"high": 0, "low": 0})
        tb += b["high"]
        ta += a["high"]
        ms = lambda c, f: c * scale / f / (args.console_mhz * 1000)  # noqa: E731
        print(f"| {i} | {desc.get(i, {}).get('shape', '')} | {b['low'] / f0:,.0f}-{b['high'] / f0:,.0f} | "
              f"{a['low'] / f1:,.0f}-{a['high'] / f1:,.0f} | {ms(b['high'], f0):.3f} | {ms(a['high'], f1):.3f} | "
              f"{(b['high'] / f0) / (a['high'] / f1) if a['high'] else float('inf'):.2f}x |")
    lb = sum(r["low"] for r in r0.values()) / f0
    la = sum(r["low"] for r in r1.values()) / f1
    mb, ma = tb * scale / f0 / (args.console_mhz * 1000), ta * scale / f1 / (args.console_mhz * 1000)
    print(f"| all | | {lb:,.0f}-{tb / f0:,.0f} | {la:,.0f}-{ta / f1:,.0f} | {mb:.3f} | {ma:.3f} | "
          f"{(tb / f0) / (ta / f1):.2f}x (low {lb / la:.2f}x) |")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--blob", required=True)
    m.add_argument("--pack", required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--mode", default="w8i")
    m.add_argument("--min-frames", type=int, default=0)
    m.add_argument("--max-frames", type=int, default=1 << 20)
    m.add_argument("--limit", type=int, default=0)
    m.add_argument("--cli-opts", default="")
    m.add_argument("--runtime", help="directory with the vocr.c to measure (default: this checkout's runtime/)")
    m.add_argument("--plain", action="store_true", help="also run the build without markers")
    m.add_argument("--no-pld", action="store_true",
                   help="no PLD model: the cache model sees only demand accesses, the bound if the A9's PLD does "
                        "nothing (by default every PLD fills its line into L1 and L2 at once, the best case)")
    m.add_argument("--vitasdk")
    m.add_argument("--arm-cc")
    m.add_argument("--qemu")
    for name in ("report", "compare"):
        r = sub.add_parser(name)
        if name == "report":
            r.add_argument("dir")
            r.add_argument("--json")
        else:
            r.add_argument("before")
            r.add_argument("after")
        r.add_argument("--blob-path", help="the blob, for the op shapes")
        r.add_argument("--console-ms", type=float, default=0.0)
        r.add_argument("--console-mhz", type=int, default=333)
        r.add_argument("--scale", type=float, default=0.0,
                       help="a fixed calibration factor on the high bound (instead of --console-ms)")
    args = ap.parse_args()
    {"measure": measure, "report": report, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    main()
