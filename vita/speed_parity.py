#!/usr/bin/env python3
"""Bit-identity of two runtime/vocr.c versions on real weights: every build's logits and text, hashed.

  speed_parity.py --base DIR --new DIR --pack PACK --blob A.vocr [--blob B.vocr ...]
                  [--modes w8i,w8f] [--builds host-gcc,host-clang,arm-neon,arm-scalar,vita-neon]
                  [--cli-opts '--tile 8'] [--json OUT]

For each of the two runtime directories (each holding vocr.c and vocr.h), the same runtime/tools/vocr_cli.c is
built five ways: host gcc and clang at -O2 (the Makefile's release flags), Cortex-A9 NEON and scalar
(-DVOCR_NO_NEON) with the Arm GNU Linux toolchain under qemu-arm, and the Vita's own object (arm-vita-eabi-gcc,
the flags of vita/CMakeLists.txt) linked into the ARM CLI as vita/a9_cycles.py does. Every build runs every
line of PACK (vita/dev_pack.py) with every blob and mode, writing all logits (vocr_cli --logits: frames x
classes floats per line) into a pipe that is hashed as it streams, and the decoded texts. The result: one
SHA-256 of the logits and one of the texts per (version, build, blob, mode), and whether NEW equals BASE in
every one, and every build equals every other. Nothing is kept on disk but the hashes.
"""
import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE)]
import a9_cycles as A9  # noqa: E402

ARM = ["-mcpu=cortex-a9", "-mfpu=neon", "-mfloat-abi=hard"]


def build(rt, out, which, arm_cc, vitasdk):
    cli = out / f"vocr_cli_{which}"
    src = [str(rt / "vocr.c"), str(ROOT / "runtime/tools/vocr_cli.c")]
    inc = ["-I", str(rt)]
    if which == "host-gcc":
        cmd = ["gcc", *A9.WARN, "-O2", *inc, "-o", str(cli), *src, "-lm"]
    elif which == "host-clang":
        cmd = ["clang", *A9.WARN, "-O2", *inc, "-o", str(cli), *src, "-lm"]
    elif which in ("arm-neon", "arm-scalar"):
        cmd = [arm_cc, *A9.WARN, *ARM, "-O2", "-static", *inc, *(["-DVOCR_NO_NEON"] if which == "arm-scalar" else []),
               "-o", str(cli), *src, "-lm"]
    elif which == "vita-neon":
        obj = out / "vocr_vita.o"
        subprocess.run([str(Path(vitasdk) / "bin/arm-vita-eabi-gcc"), *A9.WARN, *ARM, "-O2", "-c", "-o", str(obj),
                        src[0]], check=True)
        cmd = [arm_cc, *A9.WARN, *ARM, "-O2", "-static", "-Wl,--no-enum-size-warning", *inc, "-o", str(cli), str(obj),
               src[1], "-lm"]
    else:
        raise SystemExit(f"unknown build {which}")
    # vocr_cli.c includes "../vocr.h": give it the version's header next to a tools/ directory
    stage = out / "src"
    (stage / "tools").mkdir(parents=True, exist_ok=True)
    shutil.copy2(rt / "vocr.h", stage / "vocr.h")
    shutil.copy2(ROOT / "runtime/tools/vocr_cli.c", stage / "tools/vocr_cli.c")
    cmd = [str(stage / "tools/vocr_cli.c") if a == src[1] else a for a in cmd]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return cli


def run(cli, which, qemu, blob, mode, pack, opts):
    """(sha256 of all logits, sha256 of the texts, frames) for one run."""
    tmp = Path(tempfile.mkdtemp(prefix="vocr-par-"))
    fifo = tmp / "logits"
    os.mkfifo(fifo)
    h = hashlib.sha256()
    n = [0]

    def reader():
        with open(fifo, "rb") as f:
            while True:
                b = f.read(1 << 20)
                if not b:
                    break
                h.update(b)
                n[0] += len(b)

    t = threading.Thread(target=reader)
    t.start()
    pre = [str(qemu), "-cpu", "cortex-a9"] if which.startswith(("arm", "vita")) else []
    try:
        p = subprocess.run([*pre, str(cli), str(blob), "--mode", mode, "--pack", str(pack), "--logits", str(fifo),
                            *shlex.split(opts)], capture_output=True, text=True, stdin=subprocess.DEVNULL,
                           timeout=7200)
    finally:
        t.join(timeout=60)
        shutil.rmtree(tmp, ignore_errors=True)
    if p.returncode:
        raise SystemExit(f"{cli} {blob} {mode}: {p.stderr[-1500:]}")
    return h.hexdigest(), hashlib.sha256(p.stdout.encode()).hexdigest(), n[0], p.stderr.strip().splitlines()[-1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--new", required=True, type=Path)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--blob", action="append", required=True, type=Path)
    ap.add_argument("--modes", default="w8i,w8f")
    ap.add_argument("--builds", default="host-gcc,host-clang,arm-neon,arm-scalar,vita-neon")
    ap.add_argument("--cli-opts", default="")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--qemu", default=os.environ.get("QEMU_ARM", ""))
    args = ap.parse_args()
    arm_cc = os.environ.get("ARM_CC") or A9.default_arm_cc()
    vitasdk = os.environ.get("VITASDK", "")
    qemu = args.qemu or str(A9.default_qemu() / "bin/qemu-arm")
    work = Path(tempfile.mkdtemp(prefix="vocr-par-build-"))
    results, ok = [], True
    try:
        clis = {}
        for ver, rt in (("base", args.base), ("new", args.new)):
            for b in args.builds.split(","):
                d = work / ver / b
                d.mkdir(parents=True)
                clis[ver, b] = build(rt.resolve(), d, b, arm_cc, vitasdk)
        for blob in args.blob:
            for mode in args.modes.split(","):
                ref = None
                for b in args.builds.split(","):
                    row = {"blob": blob.name, "mode": mode, "build": b}
                    for ver in ("base", "new"):
                        lg, tx, nbytes, tail = run(clis[ver, b], b, qemu, blob, mode, args.pack, args.cli_opts)
                        row[ver] = {"logits_sha256": lg, "texts_sha256": tx, "logit_bytes": nbytes, "cli": tail}
                    row["new_equals_base"] = row["new"]["logits_sha256"] == row["base"]["logits_sha256"] and \
                        row["new"]["texts_sha256"] == row["base"]["texts_sha256"]
                    ref = ref or row["base"]["logits_sha256"]
                    row["equals_other_builds"] = row["new"]["logits_sha256"] == ref
                    ok &= row["new_equals_base"] and row["equals_other_builds"]
                    results.append(row)
                    print(f"{blob.name} {mode} {b}: logits {row['new']['logits_sha256'][:16]} "
                          f"({row['new']['logit_bytes']:,} B) new==base {row['new_equals_base']} "
                          f"same as {args.builds.split(',')[0]} {row['equals_other_builds']} | {row['new']['cli']}",
                          flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    if args.json:
        args.json.write_text(json.dumps({"ok": ok, "pack_sha256": hashlib.sha256(args.pack.read_bytes()).hexdigest(),
                                         "cli_opts": args.cli_opts, "results": results}, indent=1))
    print("PARITY OK" if ok else "PARITY FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
