#!/usr/bin/env python3
"""Host timings of the desktop pipelines (issue #169, docs/desktop.md): whole pages, the networks alone, ONNX
Runtime's per-op profiles and the models' MACs. Development frames only (eval/desktop.py's dev and pc sets).

  desktop_time.py pages --out PAGES.pkl [--vita 60] [--psp 60] [--pc 30] [--seed 20261101]
      a seeded sample of dev-v2's frames (Vita view: the 960 x 544 frames, PSP view: 480 x 272) and of the PC sample
      (the screenshots' own sizes), pixels included (private: the shared box only)
  desktop_time.py run --pages PAGES.pkl --pipelines P,... --threads N --out T.json [--modes seq,par] [--passes 3]
                      [--warmup 1]
      every pipeline on every page, interleaved per page: each pass visits the pages in a shuffled order (seeded) and
      runs every pipeline on the page in a shuffled order, so a load spike on the shared host meets every pipeline
      alike; a page's time is the median of its recorded passes; reported p50 / p95 / mean over the pages of each
      view, and with --ref the median over pages of each pipeline's time / the reference pipeline's. A page = the
      detector on the frame (its preprocessing, network and decode) + every horizontal box cut, its dark flag
      (vocr_line_dark) and its recognizer read (preprocessing, network, greedy CTC): the frame is already in memory.
      Modes: seq = one thread of control, every session at N intra-op threads; par = the detector at N intra-op
      threads, then the lines on a pool of N Python threads sharing one recognizer session at 1 intra-op thread
      (ONNX Runtime releases the GIL while it runs). Pipelines:
        meikiocr        MeikiOCR 0.3.5's run_ocr (its detector and its batched recognizer; seq only)
        meikiocr-nospin the same with intra-op spinning off, as MeikiOCR's own constructor sets it
        meiki+T         MeikiOCR's detector, boxes grown by eval/desktop.py's auto pad, read by VOCR6.2-T (T = S, N, L;
                        ":exact" for the exact file, else the fast one)
        ts8+T           textspot8_exact.onnx on the frame (fit to 960 x 544 first when larger), read by VOCR6.2-T
        desk:M@SIZE+T   tools/vocr_desktop.py's detector M at input policy SIZE (M@SIZE@frame: the frame pad rule),
                        read by VOCR6.2-T
        cdesk:V@SIZE+c:T the C path (desktop/libvocrdesk.so, $VOCR_DESK_LIB or desktop/build): the VDSK blob V and
                        VOCR6.2-T's int8 blob read by runtime/vocr.c W8I with its head sections, the whole page in one
                        call (the frame's luma, the detector, every line on N C threads)
        cts8+c:T        textspot8's C spotter (runtime/vocr_textspot6.c) on the frame (fit to 960 x 544 first when
                        larger) and VOCR6.2-T in C (W8I, head sections on), the lines on N Python threads, both built
                        -O3 -march=native (eval/onnx_timing.build, its O3n libraries)
      a "#dK" suffix runs the pipeline's detector at K intra-op threads (K <= N; the protocol's amendment 1: a budget
      of N threads, the detector at its fastest K in {1, 2, 4}); without it the detector runs at N
  desktop_time.py mem --frames FRAMES.pkl --pipelines P,... --threads N --out M.json [--limit 60]
      peak resident memory: each pipeline in a fresh Python process that reads the frames one at a time from disk
      (eval/desktop.load_rgb) and runs the pipeline on each; VmHWM after the imports (and one frame), after the
      models load, and at the end
  desktop_time.py nets --pages PAGES.pkl --threads 1,2,4,8 --opts disable,basic,extended,all --out N.json
                       [--models meiki,ts8,L,desk:M@SIZE] [--cbin DIR]
      each network alone (session.run on inputs already preprocessed) at every graph optimisation level and intra-op
      thread count; and the C runtime (runtime/vocr.c W8I and runtime/vocr_textspot6.c) built at -O2 and at
      -O3 -march=native (eval/onnx_timing.build): vectorisation, fusion and layout, and threads, one at a time
  desktop_time.py profile --pages PAGES.pkl --out P.json [--models ...] [--cases all:1,disable:1,all:4]
      ONNX Runtime's profiler: kernel time per op type (and per fused op) for each network and case
  desktop_time.py macs --out M.json [--desk M@SIZE ...]
      multiply-accumulates: the recognizers per CTC frame (train/export.macs_per_frame on their v0.5.0 blobs, the
      full head), textspot8 per frame (its network at the input the spotter reads), MeikiOCR's detector at 960 x 544
      and its recognizer per 960 x 32 line, and tools/vocr_desktop.py's detectors at their inputs: Conv, MatMul,
      Gemm (and their integer forms) counted from every node's output shape in a run of the graph (GridSample's
      bilinear taps listed apart)

Nothing here is a console time.
"""
import argparse
import json
import os
import pickle
import platform
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "tools", ROOT / "eval", ROOT / "synth", ROOT / "train", ROOT / "teacher", ROOT / "runtime", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import desktop as D  # noqa: E402

V05 = D.ART / "vocr-onnx" / "v0.5.0"
BLOBS = {"S": "VOCR6.2_S_w8.vocr", "N": "VOCR6.2_N_w8g2.vocr", "L": "VOCR6.2_L_w8_2l.vocr"}
SPOT_BLOB = "textspot8_t64n8c16_w8.vtsp"


def host():
    import onnxruntime as ort

    import hostinfo
    return {"cpu": hostinfo.host_label(), "machine": platform.machine(), "python": platform.python_version(),
            "onnxruntime": ort.__version__, "numpy": np.__version__, "cores": os.cpu_count(),
            "load_1min_at_start": round(os.getloadavg()[0], 1)}


# ----------------------------------------------------------------------------- pages

def cmd_pages(a):
    rng = random.Random(a.seed)
    pages = []
    for name, views in (("dev", {"vita": a.vita, "psp": a.psp}), ("pc", {"pc": a.pc})):
        frames, _ = D.load_frames(Path(a.frames_dir) / f"{name}.pkl")
        for view, n in views.items():
            fs = sorted((f for f in frames if f["view"] == view), key=lambda f: f["key"])
            for f in sorted(rng.sample(fs, min(n, len(fs))), key=lambda f: f["key"]):
                D.check_sha(f)
                pages.append({"key": f["key"], "set": name, "view": view, "W": f["W"], "H": f["H"],
                              "rgb": D.load_rgb(f)})
    with open(a.out, "wb") as f:
        pickle.dump({"pages": pages, "seed": a.seed}, f)
    print(json.dumps({"pages": len(pages), "views": {v: sum(p["view"] == v for p in pages)
                                                     for v in ("vita", "psp", "pc")}}), flush=True)


def load_pages(path):
    with open(path, "rb") as f:
        return pickle.load(f)["pages"]


# ----------------------------------------------------------------------------- pipelines

C_LIBS = {}


def c_libs():
    """eval/onnx_timing's -O2 and -O3 -march=native builds of runtime/ (the C baselines' libraries), built once."""
    if not C_LIBS:
        import onnx_timing as OT
        C_LIBS.update(OT.build(D.PV / "vocr-desktop" / "cbuild"))
    return C_LIBS


class Pipeline:
    """One desktop pipeline at one intra-op thread count and mode (module docstring)."""

    def __init__(self, spec, threads, mode, onnx_dir=D.ONNX_DIR):
        import vocr_onnx as V
        self.spec, self.threads, self.mode, self.V = spec, threads, mode, V
        self.meiki = self.cpage = self.cspot = None
        if spec.startswith("cdesk:"):
            import vocr_desktop as VD
            det, _, rec = spec[6:].rpartition("+c:")
            vdsk, _, size = det.partition("@")
            self.cpage = VD.CDesk(vdsk, V05 / BLOBS[rec], threads=threads, size=size or None)
            return
        if spec.startswith("cts8+c:"):
            import onnx_timing as OT
            import pyvocr

            import head2level
            libs = c_libs()
            self.cspot = OT.CSpot(libs["O3n"][1], V05 / SPOT_BLOB)
            data = (V05 / BLOBS[spec[7:]]).read_bytes()
            lib = pyvocr.load_lib(libs["O3n"][0])
            probe = pyvocr.Vocr(data, mode="w8i", lib=lib)
            two = head2level.read_trailer(data) is not None
            self.crecs = [pyvocr.Vocr(data, mode="w8i", lib=lib, head_gate="on" if probe.has_gate else "off",
                                      two_level=1 if two else 0) for _ in range(threads)]
            if threads > 1:
                import threading
                from concurrent.futures import ThreadPoolExecutor
                self.pool = ThreadPoolExecutor(max_workers=threads)
                self.tl = threading.local()
                self.free = list(self.crecs)
            return
        spec, _, dt = spec.partition("#d")
        det_threads = int(dt) if dt else threads
        if det_threads > threads:
            raise ValueError(f"{self.spec}: the detector's threads are within the run's budget ({threads})")
        if spec.startswith("meikiocr"):
            if mode != "seq":
                raise ValueError("meikiocr runs its own batching: seq only")
            self.meiki = D.meiki_object(threads, spinning=False if spec.endswith("nospin") else None)
            return
        det, _, rec = spec.rpartition("+")
        tier, _, form = rec.partition(":")
        form = form or "fast"
        self.pad = "auto" if det == "meiki" else None
        self.det = D.detector(det, det_threads)
        path = Path(onnx_dir) / f"VOCR6.2_{tier}_{form}.onnx"
        if mode == "seq":
            self.rec = V.Recognizer(path, session_options=D.session_options(threads))
        else:
            from concurrent.futures import ThreadPoolExecutor
            self.rec = V.Recognizer(path, session_options=D.session_options(1))
            self.pool = ThreadPoolExecutor(max_workers=threads)

    def _line(self, job):
        g, cb = job
        dark = self.V.line_dark(g, cb)
        x, y, w, h = cb
        return self.rec.read(g[y:y + h, x:x + w], invert=bool(dark)).text

    def _cline(self, job):
        g, cb = job
        if not hasattr(self.tl, "r"):
            self.tl.r = self.free.pop()
        x, y, w, h = cb
        return self.tl.r.run_topk(np.ascontiguousarray(g[y:y + h, x:x + w]), 8,
                                  invert=bool(self.V.line_dark(g, cb)))["text"]

    def _cts8(self, rgb):
        import cv2
        g = self.V.grey_of(rgb)
        H, W = g.shape
        f = min(1.0, 960 / W, 544 / H)
        gs = g if f >= 1.0 else cv2.resize(g, (max(1, int(round(W * f))), max(1, int(round(H * f)))),
                                           interpolation=cv2.INTER_AREA)
        self.cspot.find(gs)
        sp = self.cspot
        n = sp.L.ts6_boxes(sp.set, sp.box_buf.ctypes.data)
        jobs = []
        for x, y, w, h, vertical, _dark in sp.box_buf[:6 * n].reshape(n, 6).tolist():
            if vertical:
                continue
            cb = D.crop_box([x / f, y / f, (x + w) / f, (y + h) / f], 0, W, H)
            if cb:
                jobs.append((g, cb))
        if self.threads > 1 and len(jobs) > 1:
            return list(self.pool.map(self._cline, jobs))
        r = self.crecs[0]
        return [r.run_topk(np.ascontiguousarray(gg[y:y + h, x:x + w]), 8,
                           invert=bool(self.V.line_dark(gg, (x, y, w, h))))["text"] for gg, (x, y, w, h) in jobs]

    def __call__(self, rgb):
        if self.cpage is not None:
            return self.cpage.page(rgb)
        if self.cspot is not None:
            return self._cts8(rgb)
        if self.meiki is not None:
            return [t for _, t, _ in D.meiki_ocr(self.meiki, np.ascontiguousarray(rgb[:, :, ::-1]))]
        boxes = self.det.boxes(rgb)
        g = self.V.grey_of(rgb)
        H, W = g.shape
        p = D.pad_of(self.pad, H) if self.pad else 0
        jobs = []
        for b in boxes:
            if b[4]:
                continue
            cb = D.crop_box(b, p, W, H)
            if cb:
                jobs.append((g, cb))
        if self.mode == "seq" or len(jobs) < 2:
            return [self._line(j) for j in jobs]
        return list(self.pool.map(self._line, jobs))


def pstats(v):
    v = np.asarray(v, np.float64)
    return {"n": int(v.size), "p50": round(float(np.percentile(v, 50)), 3),
            "p95": round(float(np.percentile(v, 95)), 3),
            "mean": round(float(v.mean()), 3), "min": round(float(v.min()), 3), "max": round(float(v.max()), 3)}


def cmd_run(a):
    pages = load_pages(a.pages)
    threads = int(a.threads)
    specs = a.pipelines.split(",")
    modes = a.modes.split(",")
    pipes = {}
    for s in specs:
        for m in modes:
            if (s.startswith("meikiocr") or s.startswith("c")) and m != modes[0]:
                continue                                # one mode: their own threading
            pipes[f"{s}|{m}"] = Pipeline(s, threads, m, a.onnx_dir)
    rng = random.Random(a.seed)
    names = list(pipes)
    times = {n: defaultdict(list) for n in names}
    lines = {n: {} for n in names}
    t0 = time.time()
    order = list(range(len(pages)))
    for p in range(a.warmup + a.passes):
        rng.shuffle(order)
        for i in order:                         # interleaved per page: a load spike meets every pipeline alike
            rng.shuffle(names)
            for n in names:
                ts = time.perf_counter()
                out = pipes[n](pages[i]["rgb"])
                ms = (time.perf_counter() - ts) * 1e3
                if p >= a.warmup:
                    times[n][i].append(ms)
                    lines[n][i] = len(out)
        print(f"pass {p + 1}/{a.warmup + a.passes} {time.time() - t0:.0f} s", flush=True)
    res = {"kind": "vocr-desktop-timing/1", "host": host(), "threads": threads, "passes": a.passes, "ref": a.ref,
           "warmup": a.warmup, "seed": a.seed, "pages_file": str(a.pages), "pages_sha256": D.sha256_file(a.pages),
           "pages": {v: sum(p["view"] == v for p in pages) for v in ("vita", "psp", "pc")},
           "repo_head": D.git_head(), "results": {}}
    ref = {i: float(np.median(v)) for i, v in times[a.ref].items()} if a.ref in times else None
    for n in pipes:
        per_page = {i: float(np.median(v)) for i, v in times[n].items()}
        r = {}
        for view in ("vita", "psp", "pc", "all"):
            idx = [i for i, pg in enumerate(pages) if view == "all" or pg["view"] == view]
            if idx:
                r[view] = pstats([per_page[i] for i in idx])
                r[view]["lines_mean"] = round(float(np.mean([lines[n][i] for i in idx])), 2)
                if ref is not None:
                    r[view]["ratio_to_ref_median"] = round(float(np.median([per_page[i] / ref[i] for i in idx])), 4)
        res["results"][n] = r
        print(n, json.dumps({k: (v["p50"], v["p95"]) for k, v in r.items()}), flush=True)
    res["load_1min_at_end"] = round(os.getloadavg()[0], 1)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- peak memory

MEM_CHILD = r"""
import json, sys, time
sys.path[:0] = {paths!r}
def hwm():
    for ln in open("/proc/self/status"):
        if ln.startswith("VmHWM:"):
            return int(ln.split()[1])
import numpy as np
import desktop as D
import desktop_time as DT
frames, _ = D.load_frames({frames!r})
frames = frames[:{limit}]
D.load_rgb(frames[0])
base = hwm()
p = DT.Pipeline({spec!r}, {threads}, {mode!r})
loaded = hwm()
n = 0
for fr in frames:
    out = p(D.load_rgb(fr))
    n += len(out)
print(json.dumps({{"base_kib": base, "loaded_kib": loaded, "peak_kib": hwm(), "frames": len(frames), "lines": n}}))
"""


def cmd_mem(a):
    res = {"kind": "vocr-desktop-mem/1", "host": host(), "frames_file": str(a.frames),
           "frames_sha256": D.sha256_file(a.frames), "threads": int(a.threads), "limit": a.limit,
           "repo_head": D.git_head(), "results": {}}
    paths = [str(ROOT / "tools"), str(ROOT / "eval"), str(ROOT / "synth"), str(ROOT / "train"), str(ROOT / "teacher"),
             str(ROOT / "runtime"), str(ROOT)]
    import subprocess
    for spec in a.pipelines.split(","):
        mode = "par" if not (spec.startswith("meikiocr") or spec.startswith("c")) else "seq"
        code = MEM_CHILD.format(paths=paths, frames=str(a.frames), limit=int(a.limit), spec=spec,
                                threads=int(a.threads), mode=mode)
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=3600)
        if r.returncode:
            raise SystemExit(f"{spec}: {r.stderr[-2000:]}")
        rec = json.loads(r.stdout.strip().splitlines()[-1])
        rec.update(model_and_run_mib=round((rec["peak_kib"] - rec["base_kib"]) / 1024, 1),
                   peak_mib=round(rec["peak_kib"] / 1024, 1))
        res["results"][spec] = rec
        print(spec, json.dumps(rec), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- networks alone

def net_inputs(model, pages, onnx_dir):
    """(onnx path, [feeds dict]) of a model on the timing pages, preprocessed as its pipeline preprocesses."""
    import cv2
    import engines as E
    import vocr_onnx as V
    if model == "meiki":
        m = D.meiki_object(1, rec=False)
        feeds = []
        for pg in pages:
            if pg["view"] == "pc":
                continue
            t, s = m._preprocess_for_detection(np.ascontiguousarray(pg["rgb"][:, :, ::-1]))
            feeds.append({"images": t, "orig_target_sizes": np.array([[960 / s, 544 / s]], np.int64)})
        return E.model_path("meiki_det"), feeds
    if model == "ts8":
        sp = V.Spotter(Path(onnx_dir) / "textspot8_exact.onnx")
        feeds = []
        for pg in pages:
            if pg["view"] == "pc":
                continue
            g = V.grey_of(pg["rgb"])
            x = V.halve(g) if sp.scale_of(g.shape[1], g.shape[0]) == 2 else g
            feeds.append({"grey": np.ascontiguousarray(x, np.uint8)[None, None]})
        return Path(onnx_dir) / "textspot8_exact.onnx", feeds
    if model in ("S", "N", "L", "S:exact", "N:exact", "L:exact"):
        tier, _, form = model.partition(":")
        path = Path(onnx_dir) / f"VOCR6.2_{tier}_{form or 'fast'}.onnx"
        rec = V.Recognizer(path)
        det = D.Ts8Det()
        feeds = []
        for pg in pages:
            if pg["view"] == "pc":
                continue
            g = V.grey_of(pg["rgb"])
            for b in det.boxes(pg["rgb"]):
                if b[4]:
                    continue
                cb = D.crop_box(b, 0, g.shape[1], g.shape[0])
                x, y, w, h = cb
                xin, _ = rec.preprocess(g[y:y + h, x:x + w], bool(V.line_dark(g, cb)))
                f = {"grey": np.ascontiguousarray(xin, np.uint8)[None, None]}
                if rec.has_sections:
                    f["head_sections"] = np.array(True)
                feeds.append(f)
        return path, feeds
    if model.startswith("desk:"):
        import vocr_desktop as VD
        path, _, size = model[5:].partition("@")
        det = VD.Detector(path, size=size or None)
        feeds = []
        for pg in pages:
            if pg["view"] == "pc":
                continue
            x, _ = det.preprocess(V.grey_of(pg["rgb"]))
            feeds.append({det.input_name: x})
        return Path(path), feeds
    del cv2
    raise SystemExit(f"unknown model {model!r}")


def time_feeds(sess, feeds, passes, warmup):
    """Median over passes of the mean ms per input."""
    per = []
    for p in range(warmup + passes):
        t0 = time.perf_counter()
        for f in feeds:
            sess.run(None, f)
        if p >= warmup:
            per.append((time.perf_counter() - t0) * 1e3 / max(1, len(feeds)))
    return {"median": round(float(np.median(per)), 3), "min": round(min(per), 3), "max": round(max(per), 3)}


def c_rows(pages, cbin, passes, warmup):
    """The C runtime at -O2 and -O3 -march=native: VOCR6.2-L W8I per line (Rubi's head options, and the full head)
    on textspot8's lines of the timing pages, textspot8 per frame (eval/onnx_timing's build and shim)."""
    import onnx_timing as OT
    import pyvocr
    import vocr_onnx as V
    libs = OT.build(cbin)
    det = D.Ts8Det()
    crops, frames = [], []
    for pg in pages:
        if pg["view"] == "pc":
            continue
        g = V.grey_of(pg["rgb"])
        frames.append(g)
        for b in det.boxes(pg["rgb"]):
            if not b[4]:
                cb = D.crop_box(b, 0, g.shape[1], g.shape[0])
                x, y, w, h = cb
                crops.append((np.ascontiguousarray(g[y:y + h, x:x + w]), bool(V.line_dark(g, cb))))
    out = {}
    data = (V05 / BLOBS["L"]).read_bytes()
    for k, (lib, ts) in libs.items():
        L = pyvocr.load_lib(lib)
        plain = pyvocr.Vocr(data, mode="w8i", lib=L)
        sec = pyvocr.Vocr(data, mode="w8i", lib=L, two_level=1)
        spot = OT.CSpot(ts, V05 / SPOT_BLOB)
        for name, fn, items in ((f"C {k} L line (two-level head)", lambda it, r=sec: r.run_topk(it[0], 8, invert=it[1]),
                                 crops),
                                (f"C {k} L line (full head)", lambda it, r=plain: r.run_topk(it[0], 8, invert=it[1]),
                                 crops),
                                (f"C {k} textspot8 frame", lambda g, s=spot: s.find(g), frames)):
            per = []
            for p in range(warmup + passes):
                t0 = time.perf_counter()
                for it in items:
                    fn(it)
                if p >= warmup:
                    per.append((time.perf_counter() - t0) * 1e3 / max(1, len(items)))
            out[name] = {"median": round(float(np.median(per)), 3), "min": round(min(per), 3),
                         "max": round(max(per), 3), "inputs": len(items)}
            print(name, out[name], flush=True)
    return out


def cmd_nets(a):
    import onnxruntime as ort
    pages = load_pages(a.pages)
    threads = [int(t) for t in a.threads.split(",")]
    opts = a.opts.split(",")
    res = {"kind": "vocr-desktop-nets/1", "host": host(), "passes": a.passes, "warmup": a.warmup,
           "pages_sha256": D.sha256_file(a.pages), "repo_head": D.git_head(), "models": {}}
    for model in a.models.split(","):
        path, feeds = net_inputs(model, pages, a.onnx_dir)
        r = {"file": Path(path).name, "inputs": len(feeds), "cases": {}}
        for opt in opts:
            for n in threads:
                if n > a.max_threads:
                    continue
                sess = ort.InferenceSession(str(path), D.session_options(n, opt), providers=["CPUExecutionProvider"])
                r["cases"][f"{opt}:{n}"] = time_feeds(sess, feeds, a.passes, a.warmup)
                print(model, opt, n, r["cases"][f"{opt}:{n}"], flush=True)
                del sess
        res["models"][model] = r
    if a.cbin:
        res["c"] = c_rows(pages, a.cbin, a.passes, a.warmup)
    res["load_1min_at_end"] = round(os.getloadavg()[0], 1)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")


def cmd_profile(a):
    import onnxruntime as ort
    pages = load_pages(a.pages)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    res = {"kind": "vocr-desktop-profile/1", "host": host(), "repo_head": D.git_head(), "models": {}}
    for model in a.models.split(","):
        path, feeds = net_inputs(model, pages, a.onnx_dir)
        feeds = feeds[:a.inputs]
        r = {}
        for case in a.cases.split(","):
            opt, _, n = case.partition(":")
            so = D.session_options(int(n), opt, profile=work / f"{model.replace('/', '_')[:40]}_{opt}_{n}")
            sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
            for f in feeds[:3]:
                sess.run(None, f)                      # warm-up, profiled too: dropped below by run index
            for f in feeds:
                sess.run(None, f)
            prof = sess.end_profiling()
            ev = json.loads(Path(prof).read_text())
            runs = [e for e in ev if e.get("cat") == "Session" and e.get("name") == "model_run"]
            keep_from = runs[3]["ts"] if len(runs) > 3 else 0
            by_op, total = defaultdict(float), 0.0
            for e in ev:
                if e.get("cat") != "Node" or not e.get("name", "").endswith("_kernel_time") or e["ts"] < keep_from:
                    continue
                op = e.get("args", {}).get("op_name", "?")
                by_op[op] += e["dur"]
                total += e["dur"]
            n_runs = max(1, len(runs) - 3)
            r[case] = {"kernel_ms_per_run": round(total / 1e3 / n_runs, 3),
                       "model_run_ms": round(float(np.median([e["dur"] for e in runs[3:]])) / 1e3, 3) if runs[3:]
                       else None,
                       "share_by_op": {k: round(v / max(total, 1e-9), 4) for k, v in
                                       sorted(by_op.items(), key=lambda kv: -kv[1])}}
            print(model, case, r[case]["kernel_ms_per_run"], list(r[case]["share_by_op"].items())[:6], flush=True)
        res["models"][model] = r
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- MACs

MAC_OPS = ("Conv", "ConvInteger", "MatMul", "MatMulInteger", "Gemm", "ConvTranspose")


def graph_macs(path, feeds):
    """Multiply-accumulates of one run of an ONNX graph on `feeds`: every node's output shape from a run with every
    node output exposed (graph optimisations off), then Conv (out elements x Cin / group x kernel), MatMul (out
    elements x K), Gemm; GridSample's bilinear taps (4 per output element) apart. Nodes inside If / Loop are not
    counted (the recognizers' heads: train/export.macs_per_frame counts those)."""
    import onnx
    import onnxruntime as ort
    from onnx import shape_inference
    m = onnx.load(str(path))
    try:
        m = shape_inference.infer_shapes(m)
    except Exception:
        pass
    have = {o.name for o in m.graph.output}
    for n in m.graph.node:
        for o in n.output:
            if o and o not in have:
                m.graph.output.append(onnx.helper.make_empty_tensor_value_info(o))
                have.add(o)
    sess = ort.InferenceSession(m.SerializeToString(), D.session_options(1, "disable"),
                                providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    vals = dict(zip(names, sess.run(names, feeds)))
    init = {t.name: tuple(t.dims) for t in m.graph.initializer}
    shape = {k: tuple(np.asarray(v).shape) for k, v in vals.items()}
    shape.update({k: tuple(np.asarray(v).shape) for k, v in feeds.items()})
    shape.update(init)
    macs = defaultdict(int)
    grid = 0
    for n in m.graph.node:
        if n.op_type in ("Conv", "ConvInteger", "ConvTranspose"):
            out = shape.get(n.output[0])
            w = shape.get(n.input[1 if n.op_type != "ConvInteger" else 1])
            if out is None or w is None:
                continue
            group = next((a.i for a in n.attribute if a.name == "group"), 1)
            if n.op_type == "ConvTranspose":
                macs[n.op_type] += int(np.prod(shape[n.input[0]])) * int(np.prod(w[1:]))
            else:
                macs[n.op_type] += int(np.prod(out)) * int(w[1]) * int(np.prod(w[2:]))
            del group
        elif n.op_type in ("MatMul", "MatMulInteger"):
            out, A = shape.get(n.output[0]), shape.get(n.input[0])
            if out is None or A is None:
                continue
            macs[n.op_type] += int(np.prod(out)) * int(A[-1])
        elif n.op_type == "Gemm":
            out, A = shape.get(n.output[0]), shape.get(n.input[0])
            trans_a = next((a.i for a in n.attribute if a.name == "transA"), 0)
            if out is None or A is None:
                continue
            macs["Gemm"] += int(np.prod(out)) * int(A[0] if trans_a else A[-1])
        elif n.op_type == "GridSample":
            out = shape.get(n.output[0])
            if out is not None:
                grid += 4 * int(np.prod(out))
    params = sum(int(np.prod(d)) for d in init.values())
    return {"macs": int(sum(macs.values())), "by_op": dict(macs), "gridsample_taps": grid, "params": params,
            "file_bytes": Path(path).stat().st_size}


def cmd_macs(a):
    import cv2
    import engines as E
    import vocr_onnx as V

    import export
    res = {"kind": "vocr-desktop-macs/1", "recognizers": {}, "detectors": {}}
    for t, b in BLOBS.items():
        blob = export.read_blob((V05 / b).read_bytes())
        tot, head = export.macs_per_frame(blob)
        res["recognizers"][f"VOCR6.2-{t}"] = {"blob": b, "macs_per_frame": tot, "head_macs_per_frame": head,
                                             "stride_px": blob["stride"], "height": blob["height"],
                                             "fast_onnx_bytes": (D.ONNX_DIR / f"VOCR6.2_{t}_fast.onnx").stat().st_size,
                                             "macs_640px_line": tot * 640 // blob["stride"]}
    rng = np.random.default_rng(0)
    g = rng.integers(0, 256, (544, 960), dtype=np.uint8)
    ts = V.Spotter(D.ONNX_DIR / "textspot8_exact.onnx")
    for W, H in ((960, 544), (480, 272)):
        x = V.halve(g[:H, :W]) if ts.scale_of(W, H) == 2 else g[:H, :W]
        r = graph_macs(D.ONNX_DIR / "textspot8_exact.onnx", {"grey": np.ascontiguousarray(x)[None, None]})
        res["detectors"][f"textspot8 {W}x{H}"] = dict(r, network_input=list(x.shape[::-1]))
    rgb = np.ascontiguousarray(np.stack([g] * 3, -1))
    m = D.meiki_object(1, rec=False)
    t, s = m._preprocess_for_detection(rgb)
    res["detectors"]["MeikiOCR det 960x544"] = graph_macs(
        E.model_path("meiki_det"), {"images": t, "orig_target_sizes": np.array([[960, 544]], np.int64)})
    line = np.zeros((1, 3, 32, 960), np.float32)
    res["recognizers"]["MeikiOCR rec (per line)"] = graph_macs(
        E.model_path("meiki_rec"), {"images": line, "orig_target_sizes": np.array([[960, 32]], np.int64)})
    for spec in a.desk or []:
        import vocr_desktop as VD
        path, _, size = spec.partition("@")
        det = VD.Detector(path, size=size or None)
        for W, H in ((960, 544), (480, 272), (1280, 720), (1920, 1080)):
            gg = cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)
            x, _ = det.preprocess(gg)
            res["detectors"][f"desk {Path(path).name}@{size or 'default'} {W}x{H}"] = dict(
                graph_macs(path, {det.input_name: x}), network_input=[int(x.shape[-1]), int(x.shape[-2])])
    for k, v in {**res["recognizers"], **res["detectors"]}.items():
        print(k, v.get("macs", v.get("macs_per_frame")), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pages")
    p.add_argument("--frames-dir", default=str(D.PV / "vocr-desktop" / "frames"))
    p.add_argument("--out", required=True)
    p.add_argument("--vita", type=int, default=60)
    p.add_argument("--psp", type=int, default=60)
    p.add_argument("--pc", type=int, default=30)
    p.add_argument("--seed", type=int, default=20261101)
    p = sub.add_parser("run")
    p.add_argument("--pages", required=True)
    p.add_argument("--pipelines", required=True)
    p.add_argument("--threads", required=True)
    p.add_argument("--modes", default="seq,par")
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=20261101)
    p.add_argument("--ref", default="meiki+L|par", help="the pipeline|mode the ratios are taken to")
    p.add_argument("--onnx-dir", default=str(D.ONNX_DIR))
    p.add_argument("--out", required=True)
    p = sub.add_parser("mem")
    p.add_argument("--frames", required=True)
    p.add_argument("--pipelines", required=True)
    p.add_argument("--threads", default="4")
    p.add_argument("--limit", type=int, default=60)
    p.add_argument("--out", required=True)
    p = sub.add_parser("nets")
    p.add_argument("--pages", required=True)
    p.add_argument("--models", default="meiki,ts8,L")
    p.add_argument("--threads", default="1,2,4,8")
    p.add_argument("--max-threads", type=int, default=8)
    p.add_argument("--opts", default="disable,basic,extended,all")
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--cbin")
    p.add_argument("--onnx-dir", default=str(D.ONNX_DIR))
    p.add_argument("--out", required=True)
    p = sub.add_parser("profile")
    p.add_argument("--pages", required=True)
    p.add_argument("--models", default="meiki,ts8,L")
    p.add_argument("--cases", default="all:1,disable:1,all:4")
    p.add_argument("--inputs", type=int, default=40)
    p.add_argument("--work", default="/tmp/vocr-desktop-prof")
    p.add_argument("--onnx-dir", default=str(D.ONNX_DIR))
    p.add_argument("--out", required=True)
    p = sub.add_parser("macs")
    p.add_argument("--desk", nargs="*")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    {"pages": cmd_pages, "run": cmd_run, "mem": cmd_mem, "nets": cmd_nets, "profile": cmd_profile,
     "macs": cmd_macs}[a.cmd](a)


if __name__ == "__main__":
    main()
