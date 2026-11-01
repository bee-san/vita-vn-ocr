#!/usr/bin/env python3
"""c-speed's host times (#165, docs/c-speed.md "Host speed"): the recognizer's ms per line and the spotter's ms per
region, for two runtime versions, each built by gcc at -O2 (the runtime's flags) and at -O3 -march=native.

  c_speed_host_ms.py --base DIR --new DIR --models DIR --sub SUB.bin --regions REGIONS.bin --out OUT.json
                     [--runs 5] [--cpu 63] [--regions-n 60]

DIR is a runtime/ directory (vocr.c, vocr_lm.c, vocr_post.c, vocr_textspot6.c and their drivers tools/vocr_cli.c and
vocr_textspot6_test.c). The recognizer reads SUB (vita/dev_pack.py lines) in W8I with Rubi's options (top 8; Normal with
its head gate and two-level head, Large with its two-level head; vocr_cli --repeat 3); the spotter reads the first
--regions-n regions of REGIONS (u32 count, then per region u32 w, u32 h and RGBA) as Rubi's crop decode reads them
(vocr_textspot6_test --bench, -DVOCR_TEXTSPOT6_NO_MS, without the dark pass, at scale 2). One thread pinned to --cpu;
each figure is the median of --runs runs (a shared box: other jobs run beside these). Development data only; the
output holds times and counts, no text.
"""
import argparse
import json
import shutil
import statistics
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "vita")]
import ts_speed_parity as TS  # noqa: E402

FLAGS = {"O2": ["-O2"], "O3n": ["-O3", "-march=native"]}
WARN = ["-std=c99", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror"]
MODELS = {"S": ("VOCR6.2_S_w8.vocr", []), "N": ("VOCR6.2_N_w8g2.vocr", ["--head-gate", "on", "--two-level"]),
          "L": ("VOCR6.2_L_w8_2l.vocr", ["--two-level"])}
SPOTTER = "textspot8_t64n8c16_w8.vtsp"


def regions(path, n):
    """The first n regions of a vita/vita3k_parity.py region pack as the spotter driver's records (crop decode)."""
    data, recs = Path(path).read_bytes(), []
    count, off = struct.unpack_from("<I", data)[0], 4
    for i in range(min(count, n)):
        w, h = struct.unpack_from("<II", data, off)
        recs.append((w, h, 4, TS.NO_DARK | TS.SCALE2, data[off + 8:off + 8 + w * h * 4]))
        off += 8 + w * h * 4
    return recs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base", required=True, type=Path)
    ap.add_argument("--new", required=True, type=Path)
    ap.add_argument("--models", required=True, type=Path)
    ap.add_argument("--sub", required=True, type=Path)
    ap.add_argument("--regions", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--cpu", default="63")
    ap.add_argument("--regions-n", type=int, default=60)
    a = ap.parse_args()
    work = Path(tempfile.mkdtemp(prefix="c-speed-host-"))   # the drivers and a copy of the regions: removed below
    try:
        res = measure(a, work)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print("wrote", a.out)


def measure(a, work):
    recs = regions(a.regions, a.regions_n)
    TS.write_records(work / "regions.bin", recs)
    res = {"what": __doc__.strip().split("\n\n")[0], "runs": a.runs, "regions": len(recs),
           "cpu": subprocess.run(["sh", "-c", "grep -m1 'model name' /proc/cpuinfo"], capture_output=True,
                                 text=True).stdout.strip(),
           "gcc": subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout.splitlines()[0],
           "recognizer_ms_per_line": {}, "spotter_ms_per_region": {}}
    pin = ["taskset", "-c", a.cpu]
    for ver, rt in (("base", a.base), ("new", a.new)):
        for fl, f in FLAGS.items():
            cli, ts6 = work / f"cli_{ver}_{fl}", work / f"ts6_{ver}_{fl}"
            subprocess.run(["gcc", *WARN, *f, "-I", str(rt), "-o", str(cli), *(str(rt / s) for s in
                            ("vocr.c", "vocr_lm.c", "vocr_post.c", "tools/vocr_cli.c")), "-lm"], check=True)
            subprocess.run(["gcc", *WARN, *f, "-DVOCR_TEXTSPOT6_NO_MS", "-I", str(rt), "-o", str(ts6),
                            *(str(rt / s) for s in ("vocr_lines.c", "vocr_lines_rows.c", "vocr_textspot6.c",
                                                    "vocr_textspot6_test.c")), "-lm"], check=True)
            for t, (blob, opts) in MODELS.items():
                ms = []
                for _ in range(a.runs):
                    p = subprocess.run([*pin, str(cli), str(a.models / blob), "--mode", "w8i", "--pack", str(a.sub),
                                        "--topk", "8", *opts, "--repeat", "3"], capture_output=True, text=True,
                                       check=True)
                    line = [x for x in p.stderr.splitlines() if "ms/line" in x][-1]
                    ms.append(float(line.split("frames,")[1].split("ms/line")[0]))
                res["recognizer_ms_per_line"][f"{t}_{ver}_{fl}"] = round(statistics.median(ms), 3)
                print(t, ver, fl, res["recognizer_ms_per_line"][f"{t}_{ver}_{fl}"], "ms/line", flush=True)
            us = []
            for _ in range(a.runs):
                p = subprocess.run([*pin, str(ts6), "--bench", str(a.models / SPOTTER), str(work / "regions.bin"), "3"],
                                   capture_output=True, text=True, check=True)
                us.append(float(p.stderr.split("us total")[0].split()[-1]))
            res["spotter_ms_per_region"][f"{ver}_{fl}"] = round(statistics.median(us) / 1000 / len(recs), 3)
            print("spotter", ver, fl, res["spotter_ms_per_region"][f"{ver}_{fl}"], "ms/region", flush=True)
    return res


if __name__ == "__main__":
    main()
