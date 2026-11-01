#!/usr/bin/env python3
"""Two checks of the ONNX files beside eval/onnx_parity.py (issue #163, docs/onnx.md "How exact is exact"):

  onnx_checks.py levels --onnx DIR --blobs DIR --out RESULT.json [--threads 1,5,8] [--lines 100] [--frames 24]
  onnx_checks.py memory --onnx DIR --out RESULT.json

levels: the exact recognizers and both spotter files run at every intra-op thread count of --threads and at every
graph optimisation level ONNX Runtime has (onnxruntime.GraphOptimizationLevel), each against the C runtime
(runtime/vocr.c through runtime/pyvocr.py; runtime/vocr_textspot6.c through its host driver, as eval/onnx_parity.py
builds it). Development data only, plus generated inputs:
- line crops: --lines of #119's fixed development lines (evenly spaced), the horizontal boxes the spotter twin finds
  in as many development regions as --frames, and generated crops (1 x 1 to 4000 x 64; uniform, noise, stripes, a ramp,
  a checkerboard: a uniform line is one on which Normal's gate skips every frame), each in both polarities;
- per line, with the head sections off: logits (every frame and class), best, top-8 values and classes, blank logit,
  text, conf_min and conf_mean, against W8I with default options; with them on (N: gate + two-level, L: two-level):
  best, top-8, blank, text, confidence, gate and escape scores, against W8I with the blob's sections;
- the spotter: --frames textspot-dev frames, --frames development regions at their frame's scale and generated frames
  (1 x 1 to 2048 x 2048), with both files: the decode maps and every box, block, pick, truncated and presence.
Each comparison is one check; the record counts checks and failures per tier (or file), thread count and level.

memory: each .onnx file in DIR loaded and run once in a fresh Python process (a recognizer on a 1,024 px line with
its head sections on, the spotter on a 960 x 544 frame, which it halves): the peak resident memory the process gained
(ru_maxrss after the run minus before the session), in MiB. One measurement per file.

The checks run ONNX Runtime with up to max(--threads) threads: run them where that many cores may be used (a merge
gate lane: at most 8). RESULT.json holds counts only.
"""
import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "tools", ROOT / "runtime", ROOT / "eval", ROOT / "synth", ROOT / "train"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

ART = Path(os.environ.get("HERD", "/local/home/skerraut/herd-vita-ocr")) / "shared" / "artifacts"
TIERS = (("S", "VOCR6.2_S_w8.vocr"), ("N", "VOCR6.2_N_w8g2.vocr"), ("L", "VOCR6.2_L_w8_2l.vocr"))
F32 = np.float32
SCHEMA = "vocr-onnx-checks/1"


def bits_eq(a, b):
    a, b = np.asarray(a, F32), np.asarray(b, F32)
    return bool(a.shape == b.shape and np.array_equal(a.view(np.uint32), b.view(np.uint32)))


def generated_crops():
    rng = np.random.default_rng(20261101)
    out = [("noise1x1", rng.integers(0, 256, (1, 1), dtype=np.uint8))]
    for w, h in ((8, 32), (3, 40), (2049, 19), (4000, 64), (500, 64), (33, 7)):
        out.append((f"noise{w}x{h}", rng.integers(0, 256, (h, w), dtype=np.uint8)))
    for v in (0, 128, 255):
        out.append((f"uniform{v}", np.full((32, 400), v, np.uint8)))
    stripes = np.zeros((36, 300), np.uint8)
    stripes[:, ::6] = 255
    out.append(("stripes", stripes))
    out.append(("ramp", np.tile(np.linspace(0, 255, 320).astype(np.uint8), (30, 1))))
    out.append(("checker", ((np.indices((40, 360)).sum(0) // 4 % 2) * 255).astype(np.uint8)))
    return out


def generated_frames():
    rng = np.random.default_rng(20261102)
    out = []
    for w, h in ((1, 1), (17, 9), (321, 119), (960, 544), (2048, 64), (64, 2048), (2048, 2048)):
        g = rng.integers(0, 256, (h, w), dtype=np.uint8)
        if h > 8 and w > 8:
            g[h // 3: h // 3 + max(2, h // 10), w // 8: w - w // 8] = 255
        out.append((f"frame{w}x{h}", g))
    return out


def line_inputs(n_lines, n_regions):
    import textspot6 as E6

    import textspot6_twin as T6
    d = pickle.load(open(ART / "vocr5-analysis" / "work" / "units.pkl", "rb"))
    fixed = d["fixed"]
    step = max(1, len(fixed) // max(1, n_lines))
    crops = [(f"fixed:{i}", np.ascontiguousarray(fixed[i]["img"], np.uint8))
             for i in range(0, len(fixed), step)][:n_lines]
    m = T6.Model((Path(_ARGS.blobs) / "textspot8_t64n8c16_w8.vtsp").read_bytes())
    jobs = list(E6.crop_jobs("dev"))
    rstep = max(1, len(jobs) // max(1, n_regions))
    regions = []
    for uid, img, fw, fh in jobs[::rstep][:n_regions]:
        sc = T6.scale_of(m, fw, fh)
        regions.append((uid, np.ascontiguousarray(img, np.uint8), sc))
        for b in T6.spot(m, img, scale=sc).boxes:
            if not b.vertical:
                crops.append((f"{uid}#{b.x},{b.y}", np.ascontiguousarray(img[b.y:b.y + b.h, b.x:b.x + b.w])))
    crops += generated_crops()
    return crops, regions


def frame_inputs(n_frames, regions):
    import textspot6 as E6
    frames = [(k["frame"], np.ascontiguousarray(g, np.uint8), None) for k, g in E6.tsdev_frames()]
    step = max(1, len(frames) // max(1, n_frames))
    out = frames[::step][:n_frames] + regions
    out += [(k, g, None) for k, g in generated_frames()]
    return out


def levels(a):
    import onnx_parity as P
    import onnxruntime as ort
    import pyvocr
    import vocr_onnx as V

    import head2level
    lib = pyvocr.load_lib()
    threads = [int(t) for t in a.threads.split(",")]
    lv = [(n, getattr(ort.GraphOptimizationLevel, n)) for n in ort.GraphOptimizationLevel.__members__]
    crops, regions = line_inputs(a.lines, a.frames)
    jobs = frame_inputs(a.frames, regions)
    res = {"kind": SCHEMA, "part": "levels", "onnxruntime": ort.__version__, "threads": threads,
           "levels": [n for n, _ in lv], "line_inputs": len(crops), "reads_per_config": 2 * len(crops),
           "spot_inputs": len(jobs), "recognizers": {}, "spotter": {}}
    t0 = time.time()
    for t, blob in TIERS:
        data = (Path(a.blobs) / blob).read_bytes()
        plain = pyvocr.Vocr(data, mode="w8i", lib=lib)
        two = head2level.read_trailer(data) is not None
        sec = pyvocr.Vocr(data, mode="w8i", lib=lib, head_gate="on" if plain.has_gate else "off",
                          two_level=1 if two else 0) if (plain.has_gate or two) else None
        refs = []
        gated = escaped = all_gated = 0
        for _, img in crops:
            for inv in (False, True):
                _, lg = plain.run(img, logits=True, invert=inv)
                c = plain.run_topk(img, 8, invert=inv)
                s = sec.run_topk(img, 8, invert=inv, gate_score=plain.has_gate, esc=True) if sec else None
                if s is not None:
                    gated += s["gated"]
                    escaped += s["escaped"]
                    all_gated += int(s["frames"] > 0 and s["gated"] == s["frames"])
                refs.append((img, inv, lg, c, s))
        per = {}
        for n in threads:
            for name, level in lv:
                so = ort.SessionOptions()
                so.intra_op_num_threads = n
                so.inter_op_num_threads = 1
                so.graph_optimization_level = level
                so.log_severity_level = 3
                rec = V.Recognizer(Path(a.onnx) / f"VOCR6.2_{t}_exact.onnx", session_options=so)
                checks = fails = 0
                names = ["logits", "best", "topk_logits", "topk_classes", "blank_logit"] + \
                    [o for o in ("gate_score", "escape_score") if o in rec.outputs]
                for img, inv, lg, c, s in refs:
                    x, _ = rec.preprocess(img, inv)
                    for sections, want in ((False, c), (True, s)):
                        if sections and want is None:
                            continue
                        o = rec.forward(x, sections, names)
                        cm = P._conf(V, o["topk_logits"], o["best"])
                        text = "".join(rec.vocab[k] for k, _, _ in V.spans(o["best"]))
                        ok = [np.array_equal(o["best"], want["best"]), bits_eq(o["topk_logits"], want["topv"]),
                              np.array_equal(o["topk_classes"], want["topc"]), bits_eq(o["blank_logit"], want["blank"]),
                              text == want["text"],
                              F32(cm[0]) == F32(want["conf_min"]) and F32(cm[1]) == F32(want["conf_mean"])]
                        if not sections:
                            ok.append(bits_eq(o["logits"], lg))
                        else:
                            if "gate_score" in want:
                                ok.append(bits_eq(o["gate_score"], want["gate_score"]))
                            if "escape_score" in o:
                                m = ~np.isnan(want["esc"])
                                ok.append(bool(np.array_equal(np.isnan(o["escape_score"]), ~m)) and
                                          bits_eq(o["escape_score"][m], want["esc"][m]))
                        checks += len(ok)
                        fails += sum(not bool(v) for v in ok)
                per[f"{n}t {name}"] = {"checks": checks, "fails": fails}
        res["recognizers"][t] = {"configs": per, "checks": sum(v["checks"] for v in per.values()),
                                 "fails": sum(v["fails"] for v in per.values()),
                                 "c_gated_frames": gated, "c_escaped_frames": escaped, "c_all_gated_reads": all_gated}
        print(t, "checks", res["recognizers"][t]["checks"], "fails", res["recognizers"][t]["fails"],
              f"({time.time() - t0:.0f} s)", flush=True)
    exe = P.build_driver(Path(a.cbin))
    blob = Path(a.blobs) / "textspot8_t64n8c16_w8.vtsp"
    c_lines = P.c_spot(exe, blob, jobs, "batch")
    c_maps = P.c_spot(exe, blob, jobs, "maps")
    for mode in ("exact", "int8"):
        per = {}
        for n in threads:
            for name, level in lv:
                so = ort.SessionOptions()
                so.intra_op_num_threads = n
                so.inter_op_num_threads = 1
                so.graph_optimization_level = level
                so.log_severity_level = 3
                sp = V.Spotter(Path(a.onnx) / f"textspot8_{mode}.onnx", session_options=so)
                checks = fails = 0
                for (key, g, scale), cl, cm in zip(jobs, c_lines, c_maps):
                    f = scale if scale else sp.scale_of(g.shape[1], g.shape[0])
                    mp_ = sp.maps(V.halve(g) if f == 2 else g)
                    ok = [cm is not None and mp_.shape == cm.shape and bool(np.array_equal(mp_, cm)),
                          P._spots_line(sp.spot(g, scale=f)) == cl.strip()]
                    checks += len(ok)
                    fails += sum(not v for v in ok)
                per[f"{n}t {name}"] = {"checks": checks, "fails": fails}
        res["spotter"][mode] = {"configs": per, "checks": sum(v["checks"] for v in per.values()),
                                "fails": sum(v["fails"] for v in per.values())}
        print("spotter", mode, "checks", res["spotter"][mode]["checks"], "fails", res["spotter"][mode]["fails"],
              flush=True)
    res["seconds"] = round(time.time() - t0, 1)
    res["all_equal"] = not any(v["fails"] for v in list(res["recognizers"].values()) + list(res["spotter"].values()))
    return res


MEMORY_PROBE = r"""
import resource, sys
import numpy as np
import onnxruntime as ort
path = sys.argv[1]
so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1; so.log_severity_level = 3
before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
names = [i.name for i in s.get_inputs()]
shape = (1, 1, 272, 480) if "textspot" in path else (1, 1, 32, 1024)
feeds = {"grey": np.random.default_rng(0).integers(0, 256, shape, dtype=np.uint8)}
if "head_sections" in names:
    feeds["head_sections"] = np.array(True)
s.run(None, feeds)
print((resource.getrusage(resource.RUSAGE_SELF).ru_maxrss - before) / 1024.0)
"""


def memory(a):
    import onnxruntime as ort
    res = {"kind": SCHEMA, "part": "memory", "onnxruntime": ort.__version__,
           "what": "peak resident memory a fresh Python process gained loading the file and running it once (a "
                   "recognizer: a 1,024 px line, head sections on; the spotter: a 960 x 544 frame halved to "
                   "480 x 272), MiB; one measurement per file", "files": {}}
    for p in sorted(Path(a.onnx).glob("*.onnx")):
        out = subprocess.run([sys.executable, "-c", MEMORY_PROBE, str(p)], capture_output=True, text=True, check=True)
        res["files"][p.name] = round(float(out.stdout.strip().splitlines()[-1]), 1)
        print(p.name, res["files"][p.name], "MiB", flush=True)
    return res


_ARGS = None


def main(argv=None):
    global _ARGS
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    lv = sub.add_parser("levels")
    lv.add_argument("--onnx", required=True)
    lv.add_argument("--blobs", required=True)
    lv.add_argument("--out", required=True)
    lv.add_argument("--threads", default="1,5,8")
    lv.add_argument("--lines", type=int, default=100)
    lv.add_argument("--frames", type=int, default=24)
    lv.add_argument("--cbin", default=str(ART / "vocr-onnx" / "cbuild6"))
    me = sub.add_parser("memory")
    me.add_argument("--onnx", required=True)
    me.add_argument("--out", required=True)
    a = _ARGS = ap.parse_args(argv)
    res = levels(a) if a.cmd == "levels" else memory(a)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print("written", a.out, flush=True)


if __name__ == "__main__":
    main()
