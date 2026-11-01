#!/usr/bin/env python3
"""Bit-identity of runtime/ in Vita3K: the runtime's own host drivers, built as the Vita app vita/par
(vocr_par.vpk: runtime/tools/vocr_cli.c and runtime/vocr_textspot6_test.c on the Vita build of the runtime), run a set
of jobs on development data in Vita3K; every job's output must equal, byte for byte, the same job's output on the host
build of the same runtime, for both runtime versions (--base, --new), and the new version's the base version's.

  vita3k_parity.py --base DIR --new DIR --sub SUB.bin --pack LINES.bin --regions REGIONS.bin --spot SPOT.vtsp
                   --blob NAME=BLOB[:opts] [--blob ...] --work DIR [--json OUT] [--modes w8i,w8f] [--vitasdk DIR]
                   [--vita3k DIR] [--cmake CMAKE]

The jobs, per --blob NAME=BLOB (opts: the options Rubi runs the blob with, e.g. "--head-gate on --two-level"):
  - every logit of every line of SUB (vocr_cli --logits) in each of --modes: the full head, which a run asking for
    the logits always computes (vocr.h);
  - the text and every frame's top 8, blank logit and line confidence (vocr_cli --topk 8 --topk-out) of every line of
    PACK, W8I with the blob's options: the head gate and the two-level head as Rubi runs them;
and with SPOT (a VTSP trunk blob, the spotter built as Rubi builds it, -DVOCR_TEXTSPOT6_NO_MS): every region of REGIONS
(vita/dev_pack.py's region pack: u32 count, then per region u32 width, u32 height and RGBA pixels) read as Rubi's crop
decode reads a region (the row callback, VOCR_TEXTSPOT_NO_DARK, VOCR_TEXTSPOT6_SCALE2) and by the default call (the
blob's scale rule, the dark pass), its boxes, classes, blocks, order, pick and presence (--batch) and the maps the
decode reads (--maps).

For each runtime directory: the VPK (vita/CMakeLists.txt with -DVOCR_RUNTIME_DIR) and the host drivers (gcc -O2, the
runtime's own flags), then the jobs on the host and in Vita3K (vita/scripts/run_vita3k_par.sh). The work directory
holds crops of copyrighted screenshots (development data only, as vita/dev_pack.py writes it): keep it local. Every
time Vita3K reports is an x86 JIT's, not a console's.
"""
import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE)]
import ts_speed_parity as TS  # noqa: E402

UX = "ux0:data/vocrpar/"
WARN = ["-std=c99", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror"]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def build_vpk(rt, out, vitasdk, cmake):
    env = dict(os.environ, VITASDK=vitasdk, PATH=f"{vitasdk}/bin:{Path(cmake).parent}:{os.environ['PATH']}")
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([cmake, "-S", str(HERE), "-B", str(out),
                    f"-DCMAKE_TOOLCHAIN_FILE={vitasdk}/share/vita.toolchain.cmake", f"-DVOCR_RUNTIME_DIR={rt}"],
                   check=True, capture_output=True, env=env)
    p = subprocess.run([cmake, "--build", str(out), "-j8", "--target", "vocr_par.vpk-vpk"], capture_output=True,
                       text=True, env=env)
    if p.returncode:
        raise SystemExit(f"building vocr_par.vpk of {rt} failed:\n{p.stdout[-3000:]}{p.stderr[-2000:]}")
    return out / "vocr_par.vpk"


def build_host(rt, out):
    """The host drivers on runtime directory rt: vocr_cli and vocr_textspot6_test (-DVOCR_TEXTSPOT6_NO_MS). The drivers
    are this checkout's, as the VPK's (vita/CMakeLists.txt), so both versions run the same driver code."""
    out.mkdir(parents=True, exist_ok=True)
    cli, ts6 = out / "vocr_cli", out / "vocr_textspot6_test"
    drv = ROOT / "runtime"
    subprocess.run(["gcc", *WARN, "-O2", "-I", str(rt), "-o", str(cli), str(rt / "vocr.c"), str(rt / "vocr_lm.c"),
                    str(rt / "vocr_post.c"), str(drv / "tools/vocr_cli.c"), "-lm"], check=True)
    subprocess.run(["gcc", *WARN, "-O2", "-DVOCR_TEXTSPOT6_NO_MS", "-I", str(rt), "-o", str(ts6),
                    str(rt / "vocr_lines.c"), str(rt / "vocr_lines_rows.c"), str(rt / "vocr_textspot6.c"),
                    str(drv / "vocr_textspot6_test.c"), "-lm"], check=True)
    return {"cli": cli, "ts6": ts6}


def region_records(regions):
    """Rubi's crop decode and the default call, per region, in vocr_textspot6_test's record format."""
    recs, data = [], Path(regions).read_bytes()
    n, off = int.from_bytes(data[:4], "little"), 4
    for _ in range(n):
        w, h = int.from_bytes(data[off:off + 4], "little"), int.from_bytes(data[off + 4:off + 8], "little")
        px = data[off + 8:off + 8 + w * h * 4]
        off += 8 + w * h * 4
        recs.append((w, h, 4, TS.ROWS | TS.NO_DARK | TS.SCALE2, px))
        recs.append((w, h, 4, 0, px))
    return recs


def jobs_of(args, inputs):
    """[(tool, [args with {D} for the data directory], [output files])]"""
    jobs = []
    for spec in args.blob:
        name, rest = spec.split("=", 1)
        path, _, opts = rest.partition(":")
        for mode in args.modes.split(","):
            jobs.append(("cli", [f"{{D}}{name}.vocr", "--mode", mode, "--pack", "{D}sub.bin", "--logits",
                                 f"{{D}}logits_{name}_{mode}.bin"], [f"logits_{name}_{mode}.bin"]))
        jobs.append(("cli", [f"{{D}}{name}.vocr", "--mode", "w8i", "--pack", "{D}pack.bin", "--topk", "8", "--topk-out",
                             f"{{D}}topk_{name}.bin", *shlex.split(opts)], [f"topk_{name}.bin"]))
    jobs.append(("ts6", ["--batch", "{D}spot.vtsp", "{D}regions.bin"], []))
    jobs.append(("ts6", ["--maps", "{D}spot.vtsp", "{D}regions.bin"], []))
    return jobs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--new", required=True, type=Path)
    ap.add_argument("--sub", required=True, type=Path)
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--regions", required=True, type=Path)
    ap.add_argument("--spot", required=True, type=Path)
    ap.add_argument("--blob", action="append", required=True)
    ap.add_argument("--modes", default="w8i,w8f")
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--json", type=Path)
    ap.add_argument("--vitasdk", default=os.environ.get("VITASDK", ""))
    ap.add_argument("--vita3k", default=os.environ.get("VITA3K", str(HERE / "tools/vita3k/4154")))
    ap.add_argument("--cmake", default=shutil.which("cmake") or "cmake")
    args = ap.parse_args()
    work = args.work.resolve()
    inp = work / "in"
    inp.mkdir(parents=True, exist_ok=True)
    for spec in args.blob:
        name, rest = spec.split("=", 1)
        shutil.copy2(rest.partition(":")[0], inp / f"{name}.vocr")
    shutil.copy2(args.sub, inp / "sub.bin")
    shutil.copy2(args.pack, inp / "pack.bin")
    shutil.copy2(args.spot, inp / "spot.vtsp")
    TS.write_records(inp / "regions.bin", region_records(args.regions))
    jobs = jobs_of(args, inp)
    (inp / "jobs.txt").write_text("".join(f"{t} {' '.join(a).replace('{D}', UX)}\n" for t, a, _ in jobs))
    res = {"jobs": [f"{t} {' '.join(a)}" for t, a, _ in jobs],
           "inputs": {f.name: sha(f) for f in sorted(inp.iterdir())}, "versions": {}}
    outs = {}
    for ver, rt in (("base", args.base.resolve()), ("new", args.new.resolve())):
        host = build_host(rt, work / f"host_bin_{ver}")
        hdir = work / f"host_{ver}"
        hdir.mkdir(exist_ok=True)
        for k, (tool, a, files) in enumerate(jobs):
            with open(hdir / f"out_{k}.txt", "wb") as so, open(hdir / f"err_{k}.txt", "wb") as se:
                cmd = [str(host[tool]), *[x.replace("{D}", f"{inp}/") for x in a]]
                rc = subprocess.run(cmd, stdout=so, stderr=se, stdin=subprocess.DEVNULL).returncode
            if rc:
                raise SystemExit(f"host job {k} ({tool} {' '.join(a)}) of {ver}: rc {rc}")
            for f in files:                      # the job wrote it into the inputs' directory: move it beside out_k
                shutil.move(str(inp / f), hdir / f)
        vpk = build_vpk(rt, work / f"vpk_{ver}", args.vitasdk, args.cmake)
        vdir = work / f"vita3k_{ver}"
        shutil.rmtree(vdir, ignore_errors=True)
        p = subprocess.run([str(HERE / "scripts/run_vita3k_par.sh"), str(inp), str(vpk), str(vdir)],
                           capture_output=True, text=True, stdin=subprocess.DEVNULL,
                           env=dict(os.environ, VITA3K=args.vita3k, SANDBOX=str(work)))
        print(p.stdout.strip(), file=sys.stderr)
        if p.returncode:
            raise SystemExit(f"Vita3K run of {ver} failed: {p.stderr[-2000:]}")
        status = (vdir / "status.txt").read_text().split()
        rcs = [int(x) for x in status[1::2]]
        if len(rcs) != len(jobs) or any(rcs):
            raise SystemExit(f"Vita3K jobs of {ver}: status {status}")
        outs[ver] = {"host": hdir, "vita3k": vdir}
        res["versions"][ver] = {"runtime": str(rt), "vocr_c_sha256": sha(rt / "vocr.c"),
                                "vocr_textspot6_c_sha256": sha(rt / "vocr_textspot6.c"), "vpk_sha256": sha(vpk),
                                "vita3k_seconds": p.stdout.strip()}
    ok = True
    res["outputs"] = []
    for k, (tool, a, files) in enumerate(jobs):
        for f in [f"out_{k}.txt", *files]:
            h = {f"{ver}_{where}": sha(outs[ver][where] / f) for ver in ("base", "new") for where in ("host", "vita3k")}
            same = len(set(h.values())) == 1
            ok &= same
            res["outputs"].append({"job": k, "file": f, "bytes": (outs["new"]["vita3k"] / f).stat().st_size,
                                   "sha256": h["new_vita3k"], "all_equal": same, **({} if same else {"sha256s": h})})
            print(f"job {k} {f}: {(outs['new']['vita3k'] / f).stat().st_size:,} B {h['new_vita3k'][:16]} "
                  f"{'identical in all four' if same else 'DIFFERENT ' + str(h)}", flush=True)
    res["ok"] = ok
    if args.json:
        args.json.write_text(json.dumps(res, indent=1) + "\n")
    print("vita3k parity: identical" if ok else "vita3k parity: DIFFERENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
