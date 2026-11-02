#!/usr/bin/env python3
"""Host timings of textspot-desktop's detector (issue #173, eval/textspot_desktop_protocol.json, metrics.speed and
rule R5): the C path (desktop/vocr_tsd.c through tools/tsdesk_c.py) and the ONNX Runtime FP32 path (tools/tsdesk.py)
of the same weights, the detector alone and the whole page with VOCR6.2's recognizer, at 1 and 4 threads; per-step
profiles of the C network; peak memory. Timing pages are hval frames only (vocr-desktop's hval.pkl, read through
eval/desktop.load_rgb) or random frames: never dev-v2, the dense sets, confirm-v2 or the PC test set.

  tsdesk_time.py pages --out PAGES.json [--vita 60] [--psp 60] [--hd 60] [--seed 20261101]
      a seeded sample of hval's frames (keys only; the pixels are read from disk by eval/desktop.load_rgb): Vita
      (960 x 544), PSP (480 x 272), and the Vita sample resampled to 1920 x 1080 (cv2.INTER_LINEAR: the protocol's
      resampled timing pages; view "vita1080")
  tsdesk_time.py build --out DIR [--arch -march=x86-64-v3] [--extra -DVOCR_TSD_SCALAR] [--cc gcc]
      desktop/libvocrdesk.so and vocr_tsd_bench at this checkout, with the compiler's version recorded
  tsdesk_time.py run --pages PAGES.json --pipelines P,... --threads 1,4 --out T.json [--passes 3] [--warmup 1]
                     [--lib LIB.so] [--pad auto] [--random N]
      every pipeline on every page, interleaved per page (each pass visits the pages in a shuffled order and runs the
      pipelines on a page in a shuffled order), one thread count after the other; a page's time is the median of its
      passes; p50 / p95 / mean per view. Pipelines (BLOB: a .vtsd, MODEL: its .onnx, SIZE: an input policy, default
      the file's; T: VOCR6.2's tier S, N or L):
        cdet:BLOB[@SIZE]           the C detector alone: the RGB frame in memory to its final boxes (the luma, the
                                   policy's resize, the padding, the network, the decode), one vocr_tsd_detect_ex call
                                   (the binding's Box objects are not built inside the timing)
      A detector or page spec's BLOB / MODEL part may end in +refine: the box refinement on (cdet / cpage: the C
      refinement inside the detector call; odet / opage: tools/tsdesk.Detector's refine, tools/tsdesk_refine.py),
      e.g. cdet:BLOB@fit:960+refine, cpage:BLOB+refine+c:L. An ONNX spec's SIZE may be followed by @MODE, the
      resize of tools/tsdesk.Detector (exact: its numpy model of the C integer resize, the default; cv2: OpenCV's
      INTER_AREA / INTER_LINEAR, an app's resize), e.g. odet:MODEL@fit:960@cv2+refine
        cpage:BLOB[@SIZE]+c:T      the C page: the detector and VOCR6.2-T's int8 blob read by runtime/vocr.c W8I with
                                   its head sections, every horizontal box grown by --pad, the lines on the same pool
        odet:MODEL[@SIZE]          tools/tsdesk.Detector.detect on the RGB frame (ONNX Runtime FP32, intra-op threads N)
        opage:MODEL[@SIZE]+T       that detector, then every horizontal box grown by --pad, its dark flag
                                   (vocr_onnx.line_dark) and VOCR6.2_T_fast.onnx (one session, 1 intra-op thread),
                                   the lines on a pool of N Python threads
        cnet:BLOB[@SIZE]           the C network alone (vocr_tsd_maps_mt) on the frame's padded network input
                                   (built outside the timing)
        onet:MODEL[@SIZE]          ONNX Runtime's session.run alone on the same input
      Items counted: boxes (detectors), horizontal lines read (pages), maps (networks).
      --random N adds N seeded random frames (smoothed noise) per size 960 x 544 and 1920 x 1080 (views rand544,
      rand1080). A C pipeline spec may end in ^NAME to run on the library --libs NAME=LIB.so[,...] names (e.g. a plain-C
      build beside the AVX2 one); else --lib
  tsdesk_time.py mem --pages PAGES.json --pipelines P,... --threads N --out M.json [--limit 60] [--view vita]
      peak resident memory (VmHWM): each pipeline in a fresh process that loads the frames of a view one at a time
      from disk and runs the pipeline on each: after the imports and one frame, after the models load, at the end
  tsdesk_time.py prof --pages PAGES.json --blob BLOB --out P.json [--view vita] [--runs 300] [--threads 1] [--refine]
      the C network's per-step time on the view's frames from a -DVOCR_TSD_PROF build of vocr_tsd_bench (the frames
      written to a private scratch file and deleted)
  tsdesk_time.py randblobs --out DIR [--shapes p8a,p8s,p8t,p8g,p4a] [--seed 1] [--kernel 0.05]
      random-weight networks of train/tsdesk_net.SHAPES as SHAPE.onnx + SHAPE.vtsd (the same folded weights), for
      timing before trained blobs exist: BatchNorm statistics drawn away from the identity, the K map's bias set so
      that about `kernel` of a random frame's cells are kernel cells and the O map's so that 80% read horizontal

Run decisive timings in one heavy slot (HEAVY_SLOTS=12 $HERD/heavy.sh timeout ... < /dev/null); each result records
the 1-minute load at its start and end. Nothing here is a console time.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "synth", ROOT / "train", ROOT / "teacher", ROOT / "runtime", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
if str(ROOT / "tools") in sys.path:
    sys.path.remove(str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "tools"))             # tools/tsdesk.py ahead of eval/tsdesk.py
import desktop as D  # noqa: E402

V05 = D.ART / "vocr-onnx" / "v0.5.0"
BLOBS = {"S": "VOCR6.2_S_w8.vocr", "N": "VOCR6.2_N_w8g2.vocr", "L": "VOCR6.2_L_w8_2l.vocr"}
HVAL = D.PV / "vocr-desktop" / "frames" / "hval.pkl"
SCHEMA = "textspot-desktop-c-timing/1"


def host():
    import platform

    import hostinfo
    try:
        import onnxruntime as ort
        ortv = ort.__version__
    except ImportError:
        ortv = None
    return {"cpu": hostinfo.host_label(), "machine": platform.machine(), "python": platform.python_version(),
            "onnxruntime": ortv, "numpy": np.__version__, "cores": os.cpu_count(),
            "heavy_slot": os.environ.get("HEAVY_SLOT"), "load_1min_at_start": round(os.getloadavg()[0], 1)}


def pstats(v):
    v = np.asarray(v, np.float64)
    return {"n": int(v.size), "p50": round(float(np.percentile(v, 50)), 3),
            "p95": round(float(np.percentile(v, 95)), 3),
            "mean": round(float(v.mean()), 3), "min": round(float(v.min()), 3), "max": round(float(v.max()), 3)}


# ----------------------------------------------------------------------------- pages

def cmd_pages(a):
    frames, meta = D.load_frames(a.frames)
    rng = random.Random(a.seed)
    out = []
    picked = {}
    for view, n in (("vita", a.vita), ("psp", a.psp)):
        fs = sorted((f for f in frames if f["view"] == view), key=lambda f: f["key"])
        picked[view] = sorted(rng.sample(fs, min(n, len(fs))), key=lambda f: f["key"])
        for f in picked[view]:
            out.append({"key": f["key"], "view": view, "W": f["W"], "H": f["H"], "src": f["src"]})
    for f in picked["vita"][:a.hd]:
        out.append({"key": f["key"], "view": "vita1080", "W": 1920, "H": 1080, "src": f["src"],
                    "resample": {"from": [f["W"], f["H"]], "to": [1920, 1080]}})
    doc = {"kind": "textspot-desktop-c-pages", "frames_file": str(a.frames), "frames_sha256": D.sha256_file(a.frames),
           "seed": a.seed, "pages": out, "views": {v: sum(p["view"] == v for p in out) for v in
                                                   ("vita", "psp", "vita1080")}}
    Path(a.out).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(doc["views"]), flush=True)


def load_rgb(p):
    """A timing page's RGB pixels (eval/desktop.load_rgb: the hval PNG, resampled when the page says so)."""
    import cv2
    if p.get("random"):
        return random_frame(p["W"], p["H"], p["random"])
    if p.get("rgb") is not None:            # the protocol's timing pages (eval/tsdesk_eval.py pages): pixels inside
        return p["rgb"]
    fr = {"key": p["key"], "W": p["W"], "H": p["H"], "src": p["src"]}
    if p.get("resample"):
        fr["W"], fr["H"] = p["resample"]["from"]
        rgb = D.load_rgb(fr)
        return np.ascontiguousarray(cv2.resize(rgb, tuple(p["resample"]["to"]), interpolation=cv2.INTER_LINEAR))
    return D.load_rgb(fr)


def random_frame(w, h, seed):
    """Seeded smoothed noise with bright bars (a frame with some structure; no data)."""
    import cv2
    rng = np.random.default_rng(seed)
    g = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    g = cv2.GaussianBlur(g, (0, 0), 2.0)
    for _ in range(12):
        y, x = int(rng.integers(0, h - 20)), int(rng.integers(0, w - 200))
        g[y:y + int(rng.integers(8, 20)), x:x + int(rng.integers(60, 200))] = 235
    return np.ascontiguousarray(g)


def load_pages(path, random_n=0, seed=20261101):
    if str(path).endswith(".pkl"):          # eval/tsdesk_eval.py pages: the protocol's pages with their pixels
        import pickle
        with open(path, "rb") as f:
            raw = pickle.load(f)
        pages = [dict(p) for p in raw["pages"]]
        doc = {"source": raw.get("source"), "seed": raw.get("seed"),
               "pages": [{k: v for k, v in p.items() if k != "rgb"} for p in pages]}
    else:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        pages = [dict(p) for p in doc["pages"]]
    for i in range(random_n):
        for w, h, view in ((960, 544, "rand544"), (1920, 1080, "rand1080")):
            pages.append({"key": f"random:{view}:{i}", "view": view, "W": w, "H": h, "random": seed * 1000 + i})
    return doc, pages


# ----------------------------------------------------------------------------- builds

def cmd_build(a):
    out = build(a.out, a.arch, a.extra, a.cc)
    print(json.dumps(out), flush=True)


def build(out, arch="-march=x86-64-v3", extra="", cc="gcc"):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(["make", "-s", "-C", str(ROOT / "desktop"), f"OUT={out}", f"ARCH={arch}", f"CC={cc}",
                    f"EXTRA={extra}", "-j8"], check=True)
    ver = subprocess.run([cc, "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    rec = {"dir": str(out), "lib": str(out / "libvocrdesk.so"), "bench": str(out / "vocr_tsd_bench"), "cc": ver,
           "arch": arch, "extra": extra, "opt": "-O3", "repo_head": D.git_head(),
           "dirty": bool(subprocess.run(["git", "status", "--porcelain", "desktop"], cwd=ROOT, capture_output=True,
                                        text=True).stdout.strip())}
    (out / "build.json").write_text(json.dumps(rec, indent=1) + "\n")
    return rec


# ----------------------------------------------------------------------------- pipelines

def split_spec(spec):
    """(kind, path, size, rec, refine) of a pipeline spec."""
    kind, _, rest = spec.partition(":")
    rec = None
    if kind in ("cpage", "opage"):
        rest, _, rec = rest.rpartition("+")
        if kind == "cpage":
            if not rec.startswith("c:"):
                raise ValueError(f"{spec}: cpage:BLOB[@SIZE]+c:T")
            rec = rec[2:]
    refine = rest.endswith("+refine")
    if refine:
        rest = rest[:-len("+refine")]
    path, _, size = rest.partition("@")
    if kind not in ("cdet", "cpage", "odet", "opage", "cnet", "onet") or not path:
        raise ValueError(f"unknown pipeline {spec!r}")
    return kind, path, size or None, rec, refine


def size_mode(size):
    """(size, resize mode) of an ONNX spec's SIZE[@MODE] part."""
    if not size:
        return None, "exact"
    sz, _, mode = size.partition("@")
    if mode not in ("", "exact", "cv2"):
        raise ValueError(f"resize mode {mode!r}: exact or cv2")
    return sz or None, mode or "exact"


def crop_hv(b, ph, pv, W, H):
    """(x, y, w, h) of a float box (x, y, w, h) grown by ph / pv px, rounded outward, clipped; None if empty."""
    import math
    x0, y0 = max(0, int(math.floor(b[0] - ph))), max(0, int(math.floor(b[1] - pv)))
    x1, y1 = min(W, int(math.ceil(b[0] + b[2] + ph))), min(H, int(math.ceil(b[1] + b[3] + pv)))
    return (x0, y0, x1 - x0, y1 - y0) if x1 > x0 and y1 > y0 else None


class Pipeline:
    def __init__(self, spec, threads, pad="auto", lib=None, onnx_dir=D.ONNX_DIR, spinning=None, libs=None):
        import tsdesk_c as TC
        self.spec, self.threads, self.TC = spec, int(threads), TC
        base, _, libname = spec.partition("^")
        if libname:
            lib = (libs or {})[libname]
        self.kind, path, self.size, rec, refine = split_spec(base)
        self.refine = True if refine else None
        self.pad = pad
        self.inputs = {}
        if self.kind in ("cdet", "cnet"):
            self.m = TC.CModel(path, threads=self.threads, lib=lib)
            self.policy = TC.policy_of(self.size, self.m.default_policy)
        elif self.kind == "cpage":
            self.page = TC.CPage(path, V05 / BLOBS[rec], threads=self.threads, size=self.size, pad=pad,
                                 refine=self.refine, lib=lib)
        else:
            import tsdesk as T
            import vocr_onnx as V
            self.T, self.V = T, V
            osize, omode = size_mode(self.size)
            self.det = T.Detector(path, session_options=D.session_options(self.threads, spinning=spinning),
                                  size=osize, refine=self.refine, resize_mode=omode)
            if self.kind == "opage":
                from concurrent.futures import ThreadPoolExecutor
                self.rec = V.Recognizer(Path(onnx_dir) / f"VOCR6.2_{rec}_fast.onnx",
                                        session_options=D.session_options(1))
                self.pool = ThreadPoolExecutor(max_workers=self.threads) if self.threads > 1 else None
                self.kh, self.kv, self.kpx = TC.pad_rule(pad)

    def prepare(self, i, rgb):
        """The network input of frame i (cnet / onet), built once outside the timing."""
        if self.kind == "cnet":
            self.inputs[i] = np.ascontiguousarray(self.m.preprocess(rgb, self.size)[0][0, 0])
        elif self.kind == "onet":
            self.inputs[i] = self.det.preprocess(self.T.grey_of(rgb))[0]

    def _cnet(self, x):
        m, L, C = self.m, self.m.L, self.TC.C
        ph, pw = x.shape
        run, pool, nt = m._run()
        arena = m.arena(L.vocr_tsd_maps_bytes(m.m.ctypes.data, pw, ph, nt) + 64)
        out = C.POINTER(C.c_float)()
        rc = L.vocr_tsd_maps_mt(m.m.ctypes.data, x.ctypes.data, pw, ph, arena.ctypes.data, arena.size, C.byref(out),
                                run, pool, nt)
        if rc:
            raise ValueError(f"vocr_tsd_maps: {rc}")
        return 1

    def _cdet(self, rgb):
        """One vocr_tsd_detect_ex call on the frame: the number of boxes."""
        m, L, C = self.m, self.m.L, self.TC.C
        h, w, ch = rgb.shape
        kind, n = self.policy
        run, pool, nt = m._run()
        ro = m._refine_opts(self.refine)
        need = L.vocr_tsd_arena_bytes_ex(m.m.ctypes.data, kind, n, w, h, ch, nt, int(ro is not None))
        arena = m.arena(need)
        rc = L.vocr_tsd_detect_ex(m.m.ctypes.data, rgb.ctypes.data, w, h, rgb.strides[0], ch, kind, n,
                                  C.cast(ro, C.c_void_p) if ro is not None else None, arena.ctypes.data, arena.size,
                                  C.byref(m._set), run, pool, nt)
        if rc:
            raise ValueError(f"vocr_tsd_detect: {rc}")
        return int(m._set.count)

    def _line(self, job):
        g, cb = job
        x, y, w, h = cb
        return self.rec.read(g[y:y + h, x:x + w], invert=bool(self.V.line_dark(g, cb))).text

    def _opage(self, rgb):
        g = self.T.grey_of(rgb)
        H, W = g.shape
        boxes = self.det.detect(g)
        if self.kpx:
            ph, pv = self.kh, self.kv
        else:
            ph, pv = (max(2, int(round(k * H / 544))) for k in (self.kh, self.kv))
        order = self.T.reading_order([(b.x, b.y, b.w, b.h) for b in boxes])
        jobs = []
        for i in order:
            b = boxes[i]
            if b.vertical:
                continue
            cb = crop_hv((b.x, b.y, b.w, b.h), ph, pv, W, H)
            if cb:
                jobs.append((g, cb))
        if self.pool is not None and len(jobs) > 1:
            return list(self.pool.map(self._line, jobs))
        return [self._line(j) for j in jobs]

    def __call__(self, rgb, i=None):
        """Run on one frame (i: its index, for the networks' prepared inputs): the items it gives (module
        docstring)."""
        if self.kind == "cdet":
            return self._cdet(rgb)
        if self.kind == "cpage":
            return sum(1 for ln in self.page.page(rgb) if not ln.vertical)
        if self.kind == "odet":
            return len(self.det.detect(rgb))
        if self.kind == "cnet":
            return self._cnet(self.inputs[i])
        if self.kind == "onet":
            self.det.maps(self.inputs[i])
            return 1
        return len(self._opage(rgb))


def cmd_run(a):
    doc, pages = load_pages(a.pages, a.random)
    if a.views:
        pages = [p for p in pages if p["view"] in a.views.split(",")]
    t_load = time.time()
    rgbs = [load_rgb(p) for p in pages]
    t_load = time.time() - t_load
    specs = a.pipelines.split(",")
    res = {"schema": SCHEMA, "kind": "textspot-desktop-c-timing", "host": host(), "passes": a.passes,
           "warmup": a.warmup, "seed": a.seed, "pages_file": str(a.pages), "pages_sha256": D.sha256_file(a.pages),
           "pages_frames_sha256": doc.get("frames_sha256"),
           "pages": {v: sum(p["view"] == v for p in pages) for v in sorted({p["view"] for p in pages})},
           "pad": a.pad, "lib": a.lib, "build": None, "repo_head": D.git_head(), "results": {},
           "frames_load_s": round(t_load, 1)}
    if a.lib and (Path(a.lib).parent / "build.json").exists():
        res["build"] = json.loads((Path(a.lib).parent / "build.json").read_text())
    libs = dict(kv.split("=", 1) for kv in a.libs.split(",")) if a.libs else {}
    res["libs"] = libs
    res["builds"] = {k: json.loads((Path(v).parent / "build.json").read_text()) for k, v in libs.items()
                     if (Path(v).parent / "build.json").exists()}
    for threads in [int(t) for t in a.threads.split(",")]:
        pipes = {f"{s}|t{threads}": Pipeline(s, threads, a.pad, a.lib, a.onnx_dir, libs=libs) for s in specs}
        for pp in pipes.values():
            for i, rgb in enumerate(rgbs):
                pp.prepare(i, rgb)
        rng = random.Random(a.seed + threads)
        names = list(pipes)
        times = {n: defaultdict(list) for n in names}
        counts = {n: {} for n in names}
        order = list(range(len(pages)))
        t0 = time.time()
        for p in range(a.warmup + a.passes):
            rng.shuffle(order)
            for i in order:                     # interleaved per page: a load spike meets every pipeline alike
                rng.shuffle(names)
                for n in names:
                    ts = time.perf_counter()
                    k = pipes[n](rgbs[i], i)
                    ms = (time.perf_counter() - ts) * 1e3
                    if p >= a.warmup:
                        times[n][i].append(ms)
                        counts[n][i] = k
            print(f"threads {threads} pass {p + 1}/{a.warmup + a.passes} {time.time() - t0:.0f} s "
                  f"load {os.getloadavg()[0]:.1f}", flush=True)
        for n in pipes:
            per_page = {i: float(np.median(v)) for i, v in times[n].items()}
            r = {}
            for view in sorted({pg["view"] for pg in pages}):
                idx = [i for i, pg in enumerate(pages) if pg["view"] == view]
                r[view] = pstats([per_page[i] for i in idx])
                r[view]["items_mean"] = round(float(np.mean([counts[n][i] for i in idx])), 2)
            res["results"][n] = r
            print(n, json.dumps({k: (v["p50"], v["p95"]) for k, v in r.items()}), flush=True)
        del pipes
    res["load_1min_at_end"] = round(os.getloadavg()[0], 1)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- peak memory

MEM_CHILD = r"""
import json, sys
sys.path[:0] = {paths!r}
def hwm():
    for ln in open("/proc/self/status"):
        if ln.startswith("VmHWM:"):
            return int(ln.split()[1])
import numpy as np
import tsdesk_time as TT
frames = {frames!r}
np.load(frames[0])
base = hwm()
p = TT.Pipeline({spec!r}, {threads}, {pad!r}, {lib!r})
loaded = hwm()
n = 0
for f in frames:                      # one frame in memory at a time
    rgb = np.load(f)
    p.prepare(0, rgb)
    n += p(rgb, 0)
    del rgb
print(json.dumps({{"base_kib": base, "loaded_kib": loaded, "peak_kib": hwm(), "frames": len(frames), "items": n}}))
"""


def cmd_mem(a):
    res = {"schema": SCHEMA, "kind": "textspot-desktop-c-mem", "host": host(), "pages_file": str(a.pages),
           "pages_sha256": D.sha256_file(a.pages), "view": a.view, "threads": int(a.threads), "limit": a.limit,
           "pad": a.pad, "lib": a.lib, "repo_head": D.git_head(), "results": {}}
    paths = [str(ROOT / "tools"), str(ROOT / "eval"), str(ROOT / "synth"), str(ROOT / "train"), str(ROOT / "teacher"),
             str(ROOT / "runtime"), str(ROOT)]
    _, pages = load_pages(a.pages)
    sel = [p for p in pages if p["view"] == a.view][:a.limit]
    if not sel:
        raise SystemExit(f"no {a.view} pages in {a.pages}")
    import shutil

    import numpy as np
    scratch = D.PV / "textspot-desktop-c" / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    # the view's frames one per file in a private scratch directory (deleted below): a child reads one at a time
    tmpdir = tempfile.mkdtemp(dir=scratch)
    files = []
    for i, p in enumerate(sel):
        files.append(str(Path(tmpdir) / f"{i:04d}.npy"))
        np.save(files[-1], np.ascontiguousarray(load_rgb(p)))
    try:
        for spec in a.pipelines.split(","):
            code = MEM_CHILD.format(paths=paths, frames=files, spec=spec, threads=int(a.threads), pad=a.pad,
                                    lib=a.lib)
            r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=3600,
                               env=dict(os.environ, OMP_NUM_THREADS=str(a.threads)))
            if r.returncode:
                raise SystemExit(f"{spec}: {r.stderr[-2000:]}")
            rec = json.loads(r.stdout.strip().splitlines()[-1])
            rec.update(model_and_run_mib=round((rec["peak_kib"] - rec["base_kib"]) / 1024, 1),
                       peak_mib=round(rec["peak_kib"] / 1024, 1))
            res["results"][spec] = rec
            print(spec, json.dumps(rec), flush=True)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------- the per-step profile

def cmd_prof(a):
    _, pages = load_pages(a.pages, a.random)
    pages = [p for p in pages if p["view"] == a.view][:a.limit]
    rgbs = [load_rgb(p) for p in pages]
    h, w, _ = rgbs[0].shape
    if any(r.shape != rgbs[0].shape for r in rgbs):
        raise SystemExit("the view's frames differ in size")
    bdir = Path(a.build_dir)
    b = build(bdir, a.arch, "-DVOCR_TSD_PROF " + (a.extra or ""))
    scratch = D.PV / "textspot-desktop-c" / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=scratch, suffix=".rgb") as f:     # private pixels: deleted on close
        for r in rgbs:
            f.write(np.ascontiguousarray(r).tobytes())
        f.flush()
        cmd = [b["bench"], a.blob, "--w", str(w), "--h", str(h), "--ch", "3", "--raw", f.name, "--runs", str(a.runs),
               "--warmup", str(a.warmup), "--threads", str(a.threads)] + (["--size", a.size] if a.size else []) + \
              (["--refine"] if a.refine else [])
        out = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    out.update(view=a.view, frames=len(rgbs), build=b, host=host(), repo_head=D.git_head(),
               load_1min_at_end=round(os.getloadavg()[0], 1))
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("view", "p50_us", "p95_us")}), json.dumps(out.get("prof_us")), flush=True)


def random_net(shape, seed=1, kernel=0.05):
    """A random-weight TsdNet (cmd_randblobs's rule), in eval mode."""
    import torch

    import tsdesk_net as N
    torch.manual_seed(seed)
    m = N.TsdNet(N.SHAPES[shape] if isinstance(shape, str) else shape)
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.2, 0.2)
            mod.running_var.uniform_(0.5, 1.5)
            mod.weight.data.uniform_(0.5, 1.5)
            mod.bias.data.uniform_(-0.1, 0.1)
    if m.gc is not None:
        with torch.no_grad():
            m.gc.fc.weight.normal_(0, 0.1)
    m.eval()
    g = torch.from_numpy(random_frame(960, 544, seed)[None, :, :, 0].copy())[None]
    with torch.no_grad():
        y = m(g)[0]
        m.head.bias[0] -= float(torch.quantile(y[0].flatten(), 1.0 - kernel))
        m.head.bias[1] -= float(torch.quantile(y[1].flatten(), 0.8))
    return m


def cmd_randblobs(a):
    import hashlib

    import tsdesk_blob as TB
    import tsdesk_net as N
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = {}
    for shape in a.shapes.split(","):
        m = random_net(shape, a.seed, a.kernel)
        N.export_onnx(m, out / f"{shape}.onnx", N.onnx_meta(m.shape, f"random-{shape}", "fit:960", {}), check=False,
                      doc="random weights (eval/tsdesk_time.py randblobs): for timing only")
        blob = TB.blob_bytes(m, "fit:960")
        (out / f"{shape}.vtsd").write_bytes(blob)
        rec[shape] = {"vtsd_sha256": hashlib.sha256(blob).hexdigest(), "vtsd_bytes": len(blob),
                      "macs_960x544": N.macs(m.shape, 960, 544)["total"], "params": N.n_params(m)}
        print(shape, json.dumps(rec[shape]), flush=True)
    (out / "randblobs.json").write_text(json.dumps({"seed": a.seed, "kernel": a.kernel, "blobs": rec}, indent=1) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pages")
    p.add_argument("--frames", default=str(HVAL))
    p.add_argument("--out", required=True)
    p.add_argument("--vita", type=int, default=60)
    p.add_argument("--psp", type=int, default=60)
    p.add_argument("--hd", type=int, default=60)
    p.add_argument("--seed", type=int, default=20261101)
    p = sub.add_parser("build")
    p.add_argument("--out", required=True)
    p.add_argument("--arch", default="-march=x86-64-v3")
    p.add_argument("--extra", default="")
    p.add_argument("--cc", default="gcc")
    p = sub.add_parser("run")
    p.add_argument("--pages", required=True)
    p.add_argument("--pipelines", required=True)
    p.add_argument("--threads", default="1,4")
    p.add_argument("--views")
    p.add_argument("--passes", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=20261101)
    p.add_argument("--pad", default="auto")
    p.add_argument("--lib")
    p.add_argument("--libs", help="NAME=LIB.so[,...]: libraries for pipeline specs ending in ^NAME")
    p.add_argument("--random", type=int, default=0)
    p.add_argument("--onnx-dir", default=str(D.ONNX_DIR))
    p.add_argument("--out", required=True)
    p = sub.add_parser("mem")
    p.add_argument("--pages", required=True)
    p.add_argument("--pipelines", required=True)
    p.add_argument("--threads", default="1")
    p.add_argument("--view", default="vita")
    p.add_argument("--limit", type=int, default=60)
    p.add_argument("--pad", default="auto")
    p.add_argument("--lib")
    p.add_argument("--out", required=True)
    p = sub.add_parser("prof")
    p.add_argument("--pages", required=True)
    p.add_argument("--blob", required=True)
    p.add_argument("--view", default="vita")
    p.add_argument("--limit", type=int, default=60)
    p.add_argument("--size")
    p.add_argument("--runs", type=int, default=300)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--random", type=int, default=0)
    p.add_argument("--arch", default="-march=x86-64-v3")
    p.add_argument("--extra", default="")
    p.add_argument("--refine", action="store_true", help="the box refinement on (vocr_tsdr_defaults)")
    p.add_argument("--build-dir", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("randblobs")
    p.add_argument("--out", required=True)
    p.add_argument("--shapes", default="p8a,p8s,p8t,p8g,p4a")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--kernel", type=float, default=0.05)
    a = ap.parse_args(argv)
    {"pages": cmd_pages, "build": cmd_build, "run": cmd_run, "mem": cmd_mem, "prof": cmd_prof,
     "randblobs": cmd_randblobs}[a.cmd](a)


if __name__ == "__main__":
    main()
