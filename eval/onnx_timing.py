#!/usr/bin/env python3
"""Host timings: the C runtime against the ONNX exports (tools/export_onnx.py) on the same inputs (issue #163,
docs/onnx.md "Speed"). Development data only.

  onnx_timing.py --onnx DIR --blobs DIR --out RESULT.json [--passes 9] [--warmup 2] [--threads 1,5]
                 [--lines 300] [--regions 200] [--frames 100] [--cbin DIR] [--parts line,spot,page,lm]

Inputs: --lines horizontal line crops the spotter cuts from the development regions (the base variant of #119's units,
read as Rubi reads a region), --regions of those regions, and --frames whole frames (textspot-dev's development
frames), each a deterministic sample.

Backends, all called from Python and timed with time.perf_counter around each call:
- C: runtime/vocr.c (W8I, as Rubi runs it: the blob's head sections on; and off) through ctypes, and
  runtime/vocr_textspot6.c through a ctypes shim, each built twice: -O2 (runtime/Makefile's release flags) and
  -O3 -march=native, both with -std=c99 -ffp-contract=off; their outputs are checked equal before timing;
- ONNX: the exact and fast recognizers, astra's "native INT8 keeping 16-bit activations" form for comparison
  (tools/export_onnx.py recognizer(convs="ort"): the exact matrices with ONNX Runtime's own float32 convolutions, built
  in --cbin; "exact-ortconv"), and the exact and int8 spotters, ONNX Runtime CPUExecutionProvider with intra-op threads
  as --threads lists, one inter-op thread. "agreement" counts, on the timed lines, the lines whose logits (sections
  off) are bit-equal to the C runtime's and whose best classes and text (sections on) are equal.
What a call covers:
- line: C vocr_run (resize + network + greedy CTC, top-8 kept); ONNX tools/vocr_onnx.py Recognizer.read (the resize in
  NumPy, the network, greedy CTC, confidence, character cells), and the network alone (session.run on the resized line);
- spot: C vocr_textspot6_find (halving, network, decode, dark flags); ONNX Spotter.spot (halving and decode in Python),
  and the network alone;
- page: the spotter on a region or frame, then every horizontal box read (greedy, no LM): C, and
  tools/vocr_onnx.read_region;
- lm (--parts lm): the beam + 1 MB LM at Rubi's settings on each line's top-8: runtime/vocr_lm.c (-O3 -march=native)
  against tools/vocr_onnx.py CharLM (Python), with the count of lines whose texts agree.
Each pass runs every backend over every input once, the backends in a shuffled order (seeded); a backend's pass
time is its mean per input; the result is the median over --passes passes after --warmup unrecorded ones, with the
range. Host times only: nothing here says anything about a console.
"""
import argparse
import ctypes as C
import json
import os
import platform
import random
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
SPOT_BLOB = "textspot8_t64n8c16_w8.vtsp"
WARN = ["-std=c99", "-ffp-contract=off", "-Wall", "-Wextra", "-Werror"]
OPTS = {"O2": ["-O2"], "O3n": ["-O3", "-march=native"]}
SCALE = {1: 0x2, 2: 0x4}                     # VOCR_TEXTSPOT6_SCALE1 / _SCALE2
SHIM = r"""
#include "vocr_textspot6.h"
size_t ts6_model_bytes(void) { return sizeof(vocr_textspot6); }
size_t ts6_set_bytes(void) { return sizeof(VocrTextspot6Set); }
int ts6_find(const vocr_textspot6 *m, const unsigned char *px, unsigned w, unsigned h, unsigned flags, void *arena,
             size_t n, VocrTextspot6Set *out)
{
    VocrLinesImage im;
    im.pixels = px; im.width = w; im.height = h; im.stride = w; im.bpp = VOCR_LINES_GREY;
    return vocr_textspot6_find(m, &im, flags, arena, n, out);
}
unsigned ts6_boxes(const VocrTextspot6Set *s, unsigned *out6)
{
    unsigned i;
    for (i = 0; i < s->count; i++) {
        const VocrTextspot6Box *b = &s->boxes[i];
        out6[6 * i] = b->x; out6[6 * i + 1] = b->y; out6[6 * i + 2] = b->w; out6[6 * i + 3] = b->h;
        out6[6 * i + 4] = b->vertical; out6[6 * i + 5] = b->dark;
    }
    return s->count;
}
"""


def build(cbin, cc="gcc"):
    """{opt: (libvocr.so, libts6.so)} built from runtime/ into cbin."""
    cbin = Path(cbin)
    cbin.mkdir(parents=True, exist_ok=True)
    (cbin / "ts6_shim.c").write_text(SHIM)
    rt = ROOT / "runtime"
    out = {}
    for name, opt in OPTS.items():
        lib = cbin / f"libvocr_{name}.so"
        ts = cbin / f"libts6_{name}.so"
        subprocess.run([cc, *WARN, *opt, "-fPIC", "-shared", "-o", str(lib),
                        *[str(rt / s) for s in ("vocr.c", "vocr_lm.c", "vocr_post.c")], "-lm"], check=True)
        subprocess.run([cc, *WARN, *opt, "-fPIC", "-shared", f"-I{rt}", "-o", str(ts),
                        *[str(rt / s) for s in ("vocr_lines.c", "vocr_lines_rows.c", "vocr_textspot6.c")],
                        str(cbin / "ts6_shim.c"), "-lm"], check=True)
        out[name] = (lib, ts)
    return out


class CSpot:
    """runtime/vocr_textspot6.c through the shim."""

    def __init__(self, lib, blob):
        import pyvocr
        self.L = L = C.CDLL(str(lib))
        L.ts6_model_bytes.restype = C.c_size_t
        L.ts6_set_bytes.restype = C.c_size_t
        L.vocr_textspot6_load.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t]
        L.vocr_textspot6_frame_arena_bytes_flags.argtypes = [C.c_void_p, C.c_uint, C.c_uint, C.c_uint]
        L.vocr_textspot6_frame_arena_bytes_flags.restype = C.c_size_t
        L.ts6_find.argtypes = [C.c_void_p, C.c_void_p, C.c_uint, C.c_uint, C.c_uint, C.c_void_p, C.c_size_t,
                               C.c_void_p]
        L.ts6_boxes.argtypes = [C.c_void_p, C.c_void_p]
        L.ts6_boxes.restype = C.c_uint
        self.box_buf = np.zeros(6 * 64, np.uint32)
        data = Path(blob).read_bytes()
        self.blob = pyvocr._aligned(len(data))
        self.blob[:] = np.frombuffer(data, np.uint8)
        self.model = C.create_string_buffer(L.ts6_model_bytes())
        self.set = C.create_string_buffer(L.ts6_set_bytes())
        if L.vocr_textspot6_load(self.model, self.blob.ctypes.data, len(data)):
            raise SystemExit("vocr_textspot6_load failed")
        self._arenas = {}

    def find(self, g, scale=None):
        g = np.ascontiguousarray(g, np.uint8)
        fl = SCALE[scale] if scale else 0
        key = (g.shape, fl)
        if key not in self._arenas:
            n = self.L.vocr_textspot6_frame_arena_bytes_flags(self.model, g.shape[1], g.shape[0], fl)
            self._arenas[key] = (C.create_string_buffer(n), n)
        a, n = self._arenas[key]
        rc = self.L.ts6_find(self.model, g.ctypes.data, g.shape[1], g.shape[0], fl, a, n, self.set)
        if rc:
            raise SystemExit(f"vocr_textspot6_find rc {rc}")
        return bytes(self.set.raw)


def corpus(blobs, n_lines, n_regions, n_frames, seed=20261031):
    import textspot6 as E6
    import vocr_onnx as V

    import textspot6_twin as T6
    rng = random.Random(seed)
    m = T6.Model((Path(blobs) / SPOT_BLOB).read_bytes())
    regions = [(uid, img, T6.scale_of(m, fw, fh)) for uid, img, fw, fh in E6.crop_jobs("dev") if uid.endswith(":base")]
    lines = []
    for uid, img, sc in regions:
        r = T6.spot(m, img, scale=sc)
        for b in r.boxes:
            if not b.vertical:
                lines.append((f"{uid}#{b.x},{b.y}", np.ascontiguousarray(img[b.y:b.y + b.h, b.x:b.x + b.w]), b.dark))
    frames = [(k["frame"], g, None) for k, g in E6.tsdev_frames()]
    pick = lambda xs, n: sorted(rng.sample(range(len(xs)), min(n, len(xs))))     # noqa: E731
    lines = [lines[i] for i in pick(lines, n_lines)]
    regions = [regions[i] for i in pick(regions, n_regions)]
    frames = [frames[i] for i in pick(frames, n_frames)]
    del V
    return lines, regions, frames


def timed(fn, items):
    t = 0.0
    for it in items:
        t0 = time.perf_counter()
        fn(it)
        t += time.perf_counter() - t0
    return t / max(1, len(items)) * 1e3


def run_passes(backends, items, passes, warmup, seed):
    """{name: [ms per input per recorded pass]}"""
    rng = random.Random(seed)
    names = list(backends)
    out = {n: [] for n in names}
    for p in range(warmup + passes):
        rng.shuffle(names)
        for n in names:
            ms = timed(backends[n], items)
            if p >= warmup:
                out[n].append(ms)
    return out


def stat(v):
    v = sorted(v)
    return {"median": round(float(np.median(v)), 3), "min": round(v[0], 3), "max": round(v[-1], 3)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--blobs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--passes", type=int, default=9)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--threads", default="1,5")
    ap.add_argument("--lines", type=int, default=300)
    ap.add_argument("--regions", type=int, default=200)
    ap.add_argument("--frames", type=int, default=100)
    ap.add_argument("--cbin", default=str(ART / "vocr-onnx" / "cbuild-timing"))
    ap.add_argument("--parts", default="line,spot,page")
    a = ap.parse_args(argv)
    import export_onnx as X
    import onnxruntime as ort
    import pyvocr
    import vocr_onnx as V

    import head2level
    libs = build(a.cbin)
    lines, regions, frames = corpus(a.blobs, a.lines, a.regions, a.frames)
    threads = [int(t) for t in a.threads.split(",")]
    import hostinfo
    res = {"kind": "vocr-onnx-timing/1", "host": {"cpu": hostinfo.host_label(), "machine": platform.machine(),
                                                  "python": platform.python_version(), "onnxruntime": ort.__version__,
                                                  "gcc": _gcc()},
           "inputs": {"lines": len(lines), "regions": len(regions), "frames": len(frames),
                      "mean_line_width_px": round(float(np.mean([c.shape[1] for _, c, _ in lines])), 1),
                      "mean_line_height_px": round(float(np.mean([c.shape[0] for _, c, _ in lines])), 1)},
           "passes": a.passes, "warmup": a.warmup, "results": {}}
    clibs = {k: pyvocr.load_lib(v[0]) for k, v in libs.items()}
    # the -O3 -march=native build gives the -O2 build's outputs (checked on every timed input)
    checks = {}
    for t, blob in TIERS:
        data = (Path(a.blobs) / blob).read_bytes()
        r2, r3 = (pyvocr.Vocr(data, mode="w8i", lib=clibs[k]) for k in ("O2", "O3n"))
        checks[t] = all(np.array_equal(r2.run(c, logits=True, invert=bool(d))[1].view(np.uint32),
                                       r3.run(c, logits=True, invert=bool(d))[1].view(np.uint32))
                        for _, c, d in lines)
    cs = {k: CSpot(v[1], Path(a.blobs) / SPOT_BLOB) for k, v in libs.items()}
    checks["spotter"] = all(cs["O2"].find(g, s) == cs["O3n"].find(g, s) for _, g, s in regions + frames)
    res["o3n_equals_o2"] = checks
    print("O3n == O2:", checks, flush=True)
    seed = 20261031

    def sess_opts(n):
        so = ort.SessionOptions()
        so.intra_op_num_threads = n
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        return so

    parts = a.parts.split(",")
    for t, blob in TIERS:
        data = (Path(a.blobs) / blob).read_bytes()
        two = head2level.read_trailer(data) is not None
        crec = {}
        for k in OPTS:
            g = pyvocr.Vocr(data, mode="w8i", lib=clibs[k])
            sec = pyvocr.Vocr(data, mode="w8i", lib=clibs[k], head_gate="on" if g.has_gate else "off",
                              two_level=1 if two else 0) if (g.has_gate or two) else g
            crec[k] = (g, sec)
        recs = {}
        for n in threads:
            for mode in ("exact", "fast"):
                recs[(mode, n)] = V.Recognizer(Path(a.onnx) / f"VOCR6.2_{t}_{mode}.onnx",
                                               session_options=sess_opts(n))
        # astra's "native INT8 keeping 16-bit activations": the exact matrices with ONNX Runtime's own float32 convs
        # (tools/export_onnx.py recognizer(convs="ort")), for comparison only
        astra = Path(a.cbin) / f"VOCR6.2_{t}_exact_ortconv.onnx"
        astra.write_bytes(X.recognizer(data, "exact", name=f"VOCR6.2_{t}", convs="ort").SerializeToString())
        for n in threads:
            recs[("exact-ortconv", n)] = V.Recognizer(astra, session_options=sess_opts(n))
        agree = {}
        for mode in ("exact", "exact-ortconv", "fast"):
            r = recs[(mode, threads[0])]
            c = {"lines": 0, "logits_bit_equal": 0, "best_equal": 0, "text_equal": 0, "frames": 0, "frames_differ": 0}
            for _, crop, d in lines:
                x, _ = r.preprocess(crop, bool(d))
                _, lg = crec["O2"][0].run(crop, logits=True, invert=bool(d))
                s_ = crec["O2"][1].run_topk(crop, 8, invert=bool(d))
                o = r.forward(x, False, ["logits"])
                os_ = r.forward(x, True, ["best"])
                c["lines"] += 1
                c["logits_bit_equal"] += int(o["logits"].shape == lg.shape and
                                             np.array_equal(o["logits"].view(np.uint32), lg.view(np.uint32)))
                c["best_equal"] += int(np.array_equal(os_["best"], s_["best"]))
                c["text_equal"] += int("".join(r.vocab[k] for k, _, _ in V.spans(os_["best"])) == s_["text"])
                c["frames"] += len(s_["best"])
                c["frames_differ"] += int((os_["best"] != s_["best"]).sum())
            agree[mode] = c
        res.setdefault("agreement", {})[t] = agree
        print(f"agreement {t}:", json.dumps(agree), flush=True)
        if "line" in parts:
            items = [(c, bool(d)) for _, c, d in lines]
            pre = [recs[("exact", 1)].preprocess(c, d)[0] for c, d in items]
            be = {}
            for k in OPTS:
                g, sec = crec[k]
                be[f"C {k}"] = (lambda it, r=sec: r.run_topk(it[0], 8, invert=it[1]))
                be[f"C {k} plain"] = (lambda it, r=g: r.run_topk(it[0], 8, invert=it[1]))
            for (mode, n), r in recs.items():
                be[f"ONNX {mode} {n}t"] = (lambda it, r=r: r.read(it[0], invert=it[1], sections=True))
                if mode != "exact-ortconv":
                    be[f"ONNX {mode} {n}t plain"] = (lambda it, r=r: r.read(it[0], invert=it[1], sections=False))
            out = run_passes(be, items, a.passes, a.warmup, seed)
            for (mode, n), r in recs.items():         # the network alone, on the resized lines
                out[f"ONNX {mode} {n}t network"] = run_passes({0: lambda x, r=r: r.forward(x, True)}, pre, a.passes,
                                                              a.warmup, seed)[0]
            res["results"][f"line {t}"] = {k: stat(v) for k, v in out.items()}
            print(f"line {t}:", json.dumps({k: v["median"] for k, v in res["results"][f"line {t}"].items()}),
                  flush=True)
        if "page" in parts:
            for set_name, items in (("region", regions), ("frame", frames)):
                be = {}
                for k in OPTS:
                    _, sec = crec[k]

                    def cpage(it, sp=cs[k], r=sec):
                        g, s = it[1], it[2]
                        sp.find(g, s)
                        st = (ctypes_set(sp))
                        for x, y, w, h, vert, dark in st:
                            if not vert:
                                r.run_topk(np.ascontiguousarray(g[y:y + h, x:x + w]), 8, invert=bool(dark))
                    be[f"C {k}"] = cpage
                for n in threads:
                    spot = V.Spotter(Path(a.onnx) / "textspot8_exact.onnx", session_options=sess_opts(n))
                    for mode in ("exact", "fast"):
                        r = recs[(mode, n)]
                        be[f"ONNX {mode} {n}t"] = (lambda it, sp=spot, r=r: V.read_region(
                            it[1], sp, r, scale=it[2] or sp.scale_of(it[1].shape[1], it[1].shape[0])))
                out = run_passes(be, items, a.passes, a.warmup, seed)
                res["results"][f"page {set_name} {t}"] = {k: stat(v) for k, v in out.items()}
                print(f"page {set_name} {t}:", json.dumps({k: v["median"] for k, v in
                                                           res["results"][f"page {set_name} {t}"].items()}),
                      flush=True)
    if "lm" in parts:
        # the beam + 1 MB LM at Rubi's settings on each line's top-8 (the C runtime's, Rubi's sections on): the Python
        # CharLM of tools/vocr_onnx.py against runtime/vocr_lm.c through pyvocr, same inputs, same text
        lm_path = Path(a.blobs) / "vocr_lm6_o3_1000k.vlm"
        for t, blob in TIERS:
            data = (Path(a.blobs) / blob).read_bytes()
            two = head2level.read_trailer(data) is not None
            g = pyvocr.Vocr(data, mode="w8i", lib=clibs["O3n"])
            sec = pyvocr.Vocr(data, mode="w8i", lib=clibs["O3n"], head_gate="on" if g.has_gate else "off",
                              two_level=1 if two else 0) if (g.has_gate or two) else g
            py = V.CharLM(lm_path, tier=t)
            cl = pyvocr.VocrLM(sec, str(lm_path), beam=py.beam, topk=py.topk, alpha=py.alpha, beta=py.beta,
                               gate=py.gate, flags=py.flags, bracket=py.bracket)
            vocab = [sec.class_text(k) for k in range(sec.n_classes)]
            tops = [o for o in (sec.run_topk(c, 8, invert=bool(d)) for _, c, d in lines) if o["text"]]
            same = sum("".join(vocab[k] for k in py.decode_classes(vocab, o["topv"], o["topc"], o["blank"])) ==
                       cl.decode(o["topv"], o["topc"], o["blank"]) for o in tops)
            be = {"C vocr_lm O3n": lambda o, cl=cl: cl.decode(o["topv"], o["topc"], o["blank"]),
                  "Python CharLM": lambda o, py=py, v=vocab: py.decode_classes(v, o["topv"], o["topc"], o["blank"])}
            out = run_passes(be, tops, a.passes, a.warmup, seed)
            res["results"][f"lm {t}"] = {k: stat(v) for k, v in out.items()}
            res["results"][f"lm {t}"]["lines"] = len(tops)
            res["results"][f"lm {t}"]["same_text"] = int(same)
            print(f"lm {t}:", json.dumps({k: stat(v)["median"] for k, v in out.items()}), len(tops), "lines,",
                  same, "same text", flush=True)
    if "spot" in parts:
        for set_name, items in (("region", regions), ("frame", frames)):
            be = {f"C {k}": (lambda it, sp=sp: sp.find(it[1], it[2])) for k, sp in cs.items()}
            for n in threads:
                for mode in ("exact", "int8"):
                    sp = V.Spotter(Path(a.onnx) / f"textspot8_{mode}.onnx", session_options=sess_opts(n))
                    be[f"ONNX {mode} {n}t"] = (lambda it, sp=sp: sp.spot(it[1], scale=it[2]))
                    be[f"ONNX {mode} {n}t network"] = (lambda it, sp=sp: sp.head(
                        V.halve(it[1]) if (it[2] or sp.scale_of(it[1].shape[1], it[1].shape[0])) == 2 else it[1]))
            out = run_passes(be, items, a.passes, a.warmup, seed)
            res["results"][f"spot {set_name}"] = {k: stat(v) for k, v in out.items()}
            med = {k: v["median"] for k, v in res["results"][f"spot {set_name}"].items()}
            print(f"spot {set_name}:", json.dumps(med), flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")


def ctypes_set(sp):
    """The (x, y, w, h, vertical, dark) of every box of the shim's last VocrTextspot6Set."""
    n = sp.L.ts6_boxes(sp.set, sp.box_buf.ctypes.data)
    return [tuple(int(v) for v in sp.box_buf[6 * i:6 * i + 6]) for i in range(n)]


def _gcc():
    try:
        return subprocess.run(["gcc", "-dumpfullversion"], capture_output=True, text=True).stdout.strip()
    except OSError:
        return ""


if __name__ == "__main__":
    main()
