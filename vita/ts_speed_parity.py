#!/usr/bin/env python3
"""Bit-identity of two runtime/vocr_textspot6.c versions (the text spotter, textspot8 in Rubi): every build's maps and
boxes, hashed.

  ts_speed_parity.py --base DIR --new DIR --blob A.vtsp [--blob B.vtsp ...] [--regions PACK] [--frames N]
                     [--builds host-gcc,host-clang,arm-neon,arm-scalar,vita-neon] [--forms noms,ms] [--json OUT]

For each of the two runtime directories (each holding vocr_textspot6.c, its headers, vocr_lines*.c and the driver
vocr_textspot6_test.c), the driver is built five ways, as vita/speed_parity.py builds the recognizer's CLI: host gcc
and clang at -O2, Cortex-A9 NEON and scalar (-DVOCR_NO_NEON) with the Arm GNU Linux toolchain under qemu-arm, and the
Vita's own objects (arm-vita-eabi-gcc -O2, the NEON flags) linked into the ARM driver; each form once without the
multi-scale code (-DVOCR_TEXTSPOT6_NO_MS, Rubi's build of a trunk blob) and once with it. Every build reads one
records file (the driver's FILE format): every region of PACK (vita/dev_pack.py's region pack: u32 count, then per
region u32 width, u32 height and RGBA pixels; development regions only) read as Rubi's crop decode reads a region
(VOCR_TEXTSPOT6_SCALE2, the row-callback form, with and without VOCR_TEXTSPOT_NO_DARK) and by the blob's own scale
rule, in RGBA and in grey; and N procedural frames (eval/textspot_a9.frame) at 960x544, 480x272, 900x150 and odd
sizes, with random-noise frames, padded strides and odd arenas. The driver's --maps output (every map row the decode
reads: K, O, A1, A2 and the tag maps) and --batch output (boxes, classes, blocks, order, pick, presence, dark flags)
are hashed; the result is one SHA-256 of each per (version, build, form, blob), and whether NEW equals BASE in every
one and every build equals every other. Nothing is kept on disk but the hashes (the records file holds crops of
copyrighted screenshots: it lives in a temporary directory and is deleted).
"""
import argparse
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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT / "eval"), str(ROOT / "train"), str(ROOT / "synth")]
import a9_cycles as A9  # noqa: E402

ARM = ["-mcpu=cortex-a9", "-mfpu=neon", "-mfloat-abi=hard"]
SOURCES = ("vocr_lines.c", "vocr_lines_rows.c", "vocr_textspot6.c")
DRIVER = "vocr_textspot6_test.c"
WARN = ["-std=c99", "-Wall", "-Wextra", "-Werror"]
# the driver's record flags (vocr_textspot6_test.c): 1 rows callback, 2 NO_DARK, 4 padded stride, 8 odd arena,
# 16 the up-to arena, 32 SCALE1, 64 SCALE2
ROWS, NO_DARK, PADDED, ODD, UPTO, SCALE1, SCALE2 = 1, 2, 4, 8, 16, 32, 64
SIZES = ((960, 544), (480, 272), (900, 150), (900, 300), (641, 31), (17, 9), (1, 1), (321, 119), (961, 545))


def build(rt, out, which, form, arm_cc, vitasdk):
    """The driver of the runtime in rt for one build ('host-gcc', ..., 'vita-neon') and form ('noms', 'ms')."""
    drv = out / f"ts6_{which}_{form}"
    defs = ["-DVOCR_TEXTSPOT6_NO_MS"] if form == "noms" else []
    src = [str(rt / s) for s in SOURCES]
    inc = ["-I", str(rt)]
    if which in ("host-gcc", "host-clang"):
        cc = "gcc" if which == "host-gcc" else "clang"
        cmd = [cc, *WARN, "-O2", *defs, *inc, "-o", str(drv), *src, str(rt / DRIVER), "-lm"]
    elif which in ("arm-neon", "arm-scalar"):
        cmd = [arm_cc, *WARN, *ARM, "-O2", "-static", *defs, *inc,
               *(["-DVOCR_NO_NEON"] if which == "arm-scalar" else []), "-o", str(drv), *src, str(rt / DRIVER), "-lm"]
    elif which == "vita-neon":
        objs = []
        for s in src:
            o = out / (Path(s).stem + f"_vita_{form}.o")
            subprocess.run([str(Path(vitasdk) / "bin/arm-vita-eabi-gcc"), *WARN, *ARM, "-O2", *defs, *inc, "-c", "-o",
                            str(o), s], check=True)
            objs.append(str(o))
        cmd = [arm_cc, *WARN, *ARM, "-O2", "-static", "-Wl,--no-enum-size-warning", *defs, *inc, "-o", str(drv), *objs,
               str(rt / DRIVER), "-lm"]
    else:
        raise SystemExit(f"unknown build {which}")
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise SystemExit(f"building {drv.name} failed:\n{p.stderr[-3000:]}")
    return drv


def grey_of(rgba):
    """The spotter's grey, (77 R + 150 G + 29 B) >> 8."""
    r, g, b = (rgba[..., i].astype(np.uint32) for i in range(3))
    return ((77 * r + 150 * g + 29 * b) >> 8).astype(np.uint8)


def records(regions, n_frames, seed=7):
    """The records (w, h, bpp, flags, pixel bytes) of the run."""
    out = []
    if regions:
        data = Path(regions).read_bytes()
        n, off = struct.unpack_from("<I", data, 0)[0], 4
        for _ in range(n):
            w, h = struct.unpack_from("<II", data, off)
            px = np.frombuffer(data, np.uint8, w * h * 4, off + 8).reshape(h, w, 4)
            off += 8 + w * h * 4
            # Rubi's crop decode (core/vocr_ocr.c: SCALE of the frame, NO_DARK, the row callback), the dark pass, the
            # blob's own rule, and grey input
            out.append((w, h, 4, ROWS | NO_DARK | SCALE2, px.tobytes()))
            out.append((w, h, 4, SCALE2, px.tobytes()))
            out.append((w, h, 1, 0, grey_of(px).tobytes()))
    if n_frames:
        import textspot_a9
        rng = np.random.default_rng(seed)
        for i in range(n_frames):
            w, h = SIZES[i % len(SIZES)]
            g = textspot_a9.frame(seed + i, w, h)
            if i % 3 == 2:                       # noise: every code of the network's input range
                g = rng.integers(0, 256, (h, w), dtype=np.uint8)
            flags = (ROWS if i & 1 else 0) | (PADDED if i & 2 else 0) | (ODD if i & 4 else 0) | (UPTO if i & 8 else 0)
            out.append((w, h, 1, flags, g.tobytes()))
            out.append((w, h, 4, flags ^ ROWS, textspot_a9.rgba_of(g).tobytes()))
            out.append((w, h, 1, SCALE1 | NO_DARK, g.tobytes()))
            if w > 2 and h > 2:
                out.append((w, h, 1, SCALE2, g.tobytes()))
    return out


def write_records(path, recs):
    with open(path, "wb") as f:
        for w, h, bpp, flags, px in recs:
            f.write(struct.pack("<4I", w, h, bpp, flags))
            f.write(px)


def run(drv, which, qemu, mode, blob, recfile):
    pre = [str(qemu), "-cpu", "cortex-a9"] if which.startswith(("arm", "vita")) else []
    p = subprocess.run([*pre, str(drv), mode, str(blob), str(recfile)], capture_output=True, stdin=subprocess.DEVNULL,
                       timeout=14400)
    if p.returncode:
        raise SystemExit(f"{drv.name} {mode} {blob.name}: rc {p.returncode} {p.stderr[-1500:]!r}")
    return hashlib.sha256(p.stdout).hexdigest(), len(p.stdout.splitlines())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--new", required=True, type=Path)
    ap.add_argument("--blob", action="append", required=True, type=Path)
    ap.add_argument("--regions", type=Path)
    ap.add_argument("--frames", type=int, default=18)
    ap.add_argument("--builds", default="host-gcc,host-clang,arm-neon,arm-scalar,vita-neon")
    ap.add_argument("--forms", default="noms,ms")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--qemu", default=os.environ.get("QEMU_ARM", ""))
    args = ap.parse_args()
    arm_cc = os.environ.get("ARM_CC") or A9.default_arm_cc()
    vitasdk = os.environ.get("VITASDK", "")
    qemu = args.qemu or str(A9.default_qemu() / "bin/qemu-arm")
    work = Path(tempfile.mkdtemp(prefix="vtsp-par-"))
    results, ok = [], True
    try:
        recs = records(args.regions, args.frames)
        recfile = work / "records.bin"
        write_records(recfile, recs)
        rec_sha = hashlib.sha256(recfile.read_bytes()).hexdigest()
        print(f"{len(recs)} records ({recfile.stat().st_size:,} B, sha256 {rec_sha[:16]})", flush=True)
        drvs = {}
        for ver, rt in (("base", args.base), ("new", args.new)):
            for b in args.builds.split(","):
                for form in args.forms.split(","):
                    d = work / ver / b
                    d.mkdir(parents=True, exist_ok=True)
                    drvs[ver, b, form] = build(rt.resolve(), d, b, form, arm_cc, vitasdk)
        for blob in args.blob:
            ref = None
            for b in args.builds.split(","):
                for form in args.forms.split(","):
                    row = {"blob": blob.name, "build": b, "form": form}
                    for ver in ("base", "new"):
                        row[ver] = {}
                        for mode in ("--maps", "--batch"):
                            h, n = run(drvs[ver, b, form], b, qemu, mode, blob, recfile)
                            row[ver][mode[2:]] = {"sha256": h, "lines": n}
                    keys = ("maps", "batch")
                    row["new_equals_base"] = all(row["new"][k]["sha256"] == row["base"][k]["sha256"] for k in keys)
                    got = tuple(row["new"][k]["sha256"] for k in keys)
                    ref = ref or got
                    row["equals_other_builds"] = got == ref
                    ok &= row["new_equals_base"] and row["equals_other_builds"]
                    results.append(row)
                    print(f"{blob.name} {b} {form}: maps {got[0][:16]} ({row['new']['maps']['lines']} lines) boxes "
                          f"{got[1][:16]} new==base {row['new_equals_base']} same as the first build "
                          f"{row['equals_other_builds']}", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if args.json:
        args.json.write_text(json.dumps({"ok": ok, "records": len(recs), "records_sha256": rec_sha,
                                         "regions": str(args.regions) if args.regions else None,
                                         "frames": args.frames, "results": results}, indent=1) + "\n")
    print("parity: identical" if ok else "parity: DIFFERENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
