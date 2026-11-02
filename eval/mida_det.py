#!/usr/bin/env python3
"""MeikiOCR's author's line detector `mida_ocr_l.onnx` as a detector backend for the desktop benchmark (issue #169,
eval/desktop.py on feat/vocr-desktop): measurement only.

The model is not in this repository and must not be committed, redistributed or trained on (its outputs included)
until its author states a licence that allows it. It was posted publicly on 2026-11-01 for detection benchmarks
(1,751,881 B, sha256 9f72bce5...). `check_model` reads the file once and refuses it, before onnxruntime sees it, when
it is not that file, references external data anywhere, uses an operator domain other than the default ONNX one, holds
an attribute outside FLOAT / INT / STRING / TENSOR and their lists (subgraphs, sparse tensors), model-local functions
or training_info. onnxruntime is then given the checked bytes, not the path.

What the graph is (read from the file with `onnx`; the measurements are in issue #169):
  input  "images"  float32 (N, 3, h, w), any h and w: the image's 0-255 values / 255 (the graph subtracts 0.5)
  output "maps"    float32 (N, 4, ceil(h / 8), ceil(w / 8)): the four maps of a kernel line detector at stride 8,
                   the head of textspot6 / textspot8 (runtime/vocr_textspot6.c) without its tag maps:
                     K   a line's kernel (a logit; > 0 inside the kernel, a band along the line's middle)
                     O   its orientation (a logit; > 0 for a vertical line)
                     A1  the distance from a cell's centre to the line's near edge across it (top, or left for a
                         vertical line), in stride units (x 8 = input px)
                     A2  the same to the far edge (bottom, or right)
The network: a depthwise-separable backbone (strides 2 to 32; 16, 32, 64, 128, 192 channels), one 4-head
self-attention layer over the stride-32 cells, an FPN back to stride 8 (128 channels); 429,988 float parameters.

The decode follows textspot6's on float maps: the 4-connected components of K > `thr` with at least `min_cells` cells;
a component's extent across the line is the mean of its cells' (centre - 8 A1) and (centre + 8 A2), along it the
component's cells; the kernel can stop short of a line's ends, so each end is extended by `along` x the box's size
across (`ext_start` / `ext_end` set the two ends apart: left / right, or top / bottom for a vertical line); then the
crop pad (`pad`): "none", "frame" (eval/desktop.py's auto pad, MeikiOCR's boxes' pad in the reference pipeline R:
max(2, round(4 H / 544)) frame px) or "pct" (textspot8's: 18% of the size across, at least 2 input px). Its author's
own decode and preprocessing are not published, so the channel order, the input size policy and the decode's
settings were chosen on hval: bgr, thr 0, min_cells 2, along 1.0, pad frame, at long:640 or on frames fit to
960 x 544 (the spec "mida@long:640@along=1.0@min_cells=2", or "mida@box:960x544@along=1.0@min_cells=2").

Boxes are eval/desktop.py's detector rows: [x0, y0, x1, y1, vertical, conf (the mean K logit), tag (-1: none)] in
frame px, sorted by top edge.
"""
import hashlib
import math
from pathlib import Path

import numpy as np

SHA256 = "9f72bce58b50b1837bb16e81c5c035d32d9cd45d84dcef56f0b6fb2ef25e8aed"
SIZE = 1751881
STRIDE = 8
INPUT, OUTPUT = "images", "maps"
DEFAULTS = {"policy": "native", "order": "bgr", "thr": 0.0, "min_cells": 1, "along": 0.0, "ext_start": -1.0,
            "ext_end": -1.0, "pad": "frame", "interp": "linear"}


ALLOWED_ATTRIBUTE_TYPES = ("FLOAT", "INT", "STRING", "TENSOR", "FLOATS", "INTS", "STRINGS", "TENSORS")


def check_bytes(data, sha256=SHA256, name="model"):
    """Static checks of an untrusted ONNX file's bytes before any runtime sees them; returns (facts, the model).

    - the expected bytes: sha256 (unless sha256=None);
    - no external-data reference in any tensor: initializers, sparse initializers (values and indices), and every
      tensor or sparse tensor in any attribute field, whatever the attribute's declared type;
    - attributes of the allowlisted types only (FLOAT, INT, STRING, TENSOR and their lists), holding no field of
      another type: no graph (subgraphs), sparse-tensor, type-proto or undefined attribute;
    - no operator domain or opset import but the default ONNX one, no model-local functions, no training_info;
    - exactly one float input (outside the initializers) and one float output.
    Raises ValueError on any failure."""
    import onnx
    from onnx import AttributeProto, TensorProto
    if sha256 is not None and hashlib.sha256(data).hexdigest() != sha256:
        raise ValueError(f"{name}: sha256 is not {sha256[:8]}...")
    m = onnx.load_from_string(bytes(data))           # bytes only: nothing outside them is read
    problems, domains = [], set()
    allowed = {getattr(AttributeProto, t) for t in ALLOWED_ATTRIBUTE_TYPES}

    def ext(t):
        return t.data_location == TensorProto.EXTERNAL or len(t.external_data) > 0

    def tensors_of(a):
        """Every TensorProto an attribute holds, in any field (the declared type is not trusted)."""
        out = list(a.tensors)
        if a.HasField("t"):
            out.append(a.t)
        for st in list(a.sparse_tensors) + ([a.sparse_tensor] if a.HasField("sparse_tensor") else []):
            out += [st.values, st.indices]
        return out

    def walk(g, where):
        for t in g.initializer:
            if ext(t):
                problems.append(f"{where}: initializer {t.name} has external data")
        for st in g.sparse_initializer:
            if ext(st.values) or ext(st.indices):
                problems.append(f"{where}: a sparse initializer has external data")
        for n in g.node:
            domains.add(n.domain)
            for a in n.attribute:
                kind = AttributeProto.AttributeType.Name(a.type)
                if a.type not in allowed:
                    problems.append(f"{where}: {n.name}.{a.name} is a {kind} attribute")
                if a.HasField("g") or len(a.graphs):
                    problems.append(f"{where}: {n.name}.{a.name} holds a subgraph")
                if a.HasField("sparse_tensor") or len(a.sparse_tensors) or a.HasField("tp") or \
                        len(a.type_protos):
                    problems.append(f"{where}: {n.name}.{a.name} holds a sparse tensor or a type")
                if any(ext(t) for t in tensors_of(a)):
                    problems.append(f"{where}: {n.name}.{a.name} has external data")

    walk(m.graph, "graph")
    bad = sorted(d for d in domains if d not in ("", "ai.onnx"))
    if bad:
        problems.append(f"operator domains {bad}")
    bad = sorted(o.domain for o in m.opset_import if o.domain not in ("", "ai.onnx"))
    if bad:
        problems.append(f"opset imports {bad}")
    if len(m.functions):
        problems.append(f"{len(m.functions)} model-local functions")
    if len(m.training_info):
        problems.append("training_info")
    ins = [i for i in m.graph.input if i.name not in {t.name for t in m.graph.initializer}]
    if len(ins) != 1 or len(m.graph.output) != 1:
        problems.append(f"{len(ins)} inputs and {len(m.graph.output)} outputs (expected 1 and 1)")
    elif ins[0].type.tensor_type.elem_type != TensorProto.FLOAT or \
            m.graph.output[0].type.tensor_type.elem_type != TensorProto.FLOAT:
        problems.append("the input or the output is not float32")
    if problems:
        raise ValueError(f"{name}: " + "; ".join(problems))
    from collections import Counter
    facts = {"file": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
             "ir_version": m.ir_version, "producer": f"{m.producer_name} {m.producer_version}".strip(),
             "opsets": {o.domain or "ai.onnx": o.version for o in m.opset_import},
             "ops": dict(Counter(n.op_type for n in m.graph.node)),
             "params": int(sum(int(np.prod(t.dims)) for t in m.graph.initializer)),
             "input": ins[0].name, "output": m.graph.output[0].name}
    return facts, m


def check_model(path, sha256=SHA256):
    """check_bytes on a file read once: returns (facts, the checked bytes). Callers give ONNX Runtime those bytes,
    never the path, so the file checked is the file loaded."""
    p = Path(path)
    data = p.read_bytes()
    facts, _ = check_bytes(data, sha256, p.name)
    return facts, data


def parse_spec(spec):
    """'mida[:PATH][@POLICY][@KEY=VALUE...]' -> (path or None, settings). POLICY: native | fit:N | long:N | meiki |
    box:WxH (fit inside W x H, up or down, no padding) | stretch:WxH; KEY: order (bgr|rgb), thr, min_cells, along,
    ext_start, ext_end (the start / end of a line's reading direction: left / right, top / bottom; -1 = along), pad
    (none|frame|pct), interp (linear|area)."""
    if not spec.startswith("mida"):
        raise ValueError(f"not a mida spec: {spec!r}")
    head, *parts = spec[4:].split("@")
    path = head[1:] if head.startswith(":") and len(head) > 1 else None
    cfg = dict(DEFAULTS)
    for part in parts:
        if not part:
            continue
        if "=" in part:
            k, _, v = part.partition("=")
            if k not in DEFAULTS or k == "policy":
                raise ValueError(f"mida spec {spec!r}: unknown setting {k!r}")
            cfg[k] = type(DEFAULTS[k])(v) if not isinstance(DEFAULTS[k], str) else v
        else:
            cfg["policy"] = part
    if cfg["order"] not in ("bgr", "rgb") or cfg["pad"] not in ("none", "frame", "pct") or \
            cfg["interp"] not in ("linear", "area"):
        raise ValueError(f"mida spec {spec!r}: bad setting")
    return path, cfg


def input_size(policy, w, h):
    """(network input width, height, content width, height) of a w x h frame under a size policy; the content is
    the resized frame, the rest of the input (only for 'meiki') is zero padding."""
    if policy in ("native", "", None):
        return w, h, w, h
    if policy == "meiki":              # meikiocr 0.3.5's detector input: fit inside 960 x 544, int(), zero pad
        s = min(960 / w, 544 / h)
        cw, ch = int(w * s), int(h * s)
        return 960, 544, cw, ch
    kind, _, n = policy.partition(":")
    if kind == "stretch":
        sw, _, sh = n.partition("x")
        return int(sw), int(sh), int(sw), int(sh)
    if kind == "box":                  # fit inside W x H, up or down, no padding (eval/desktop.py resample's rounding)
        bw, _, bh = n.partition("x")
        s = min(int(bw) / w, int(bh) / h)
        ch = max(1, int(round(h * s)))
        cw = max(1, int(round(w * ch / h)))
        return cw, ch, cw, ch
    n = float(n)
    if kind == "fit":
        s = min(1.0, n / max(w, h))
    elif kind == "long":
        s = n / max(w, h)
    else:
        raise ValueError(f"size policy {policy!r}")
    cw, ch = max(1, int(round(w * s))), max(1, int(round(h * s)))
    return cw, ch, cw, ch


def preprocess(rgb, cfg):
    """(input tensor 1 x 3 x h x w float32, x factor, y factor): content px = frame px x factor."""
    import cv2
    a = np.asarray(rgb)
    if a.dtype != np.uint8 or a.ndim != 3 or a.shape[2] not in (3, 4):
        raise ValueError("an H x W x 3 / 4 RGB(A) uint8 frame")
    img = a[:, :, :3] if cfg["order"] == "rgb" else a[:, :, 2::-1]
    h, w = img.shape[:2]
    iw, ih, cw, ch = input_size(cfg["policy"], w, h)
    if (cw, ch) != (w, h):
        interp = cv2.INTER_AREA if cfg["interp"] == "area" and cw * ch < w * h else cv2.INTER_LINEAR
        img = cv2.resize(np.ascontiguousarray(img), (cw, ch), interpolation=interp)
    x = np.zeros((1, 3, ih, iw), np.float32)
    x[0, :, :ch, :cw] = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))
    return x, cw / w, ch / h


def components(mask):
    """4-connected components of a boolean map: [(rows, cols)] in label order (cv2.connectedComponents)."""
    import cv2
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=4)
    if n <= 1:
        return []
    flat = lab.ravel()
    order = np.argsort(flat, kind="stable")
    counts = np.bincount(flat, minlength=n)
    starts = np.concatenate([[0], np.cumsum(counts)])
    W = lab.shape[1]
    out = []
    for c in range(1, n):
        idx = order[starts[c]:starts[c + 1]]
        out.append((idx // W, idx % W))
    return out


def decode(maps, fx, fy, W, H, cfg):
    """eval/desktop.py detector rows in frame px (module docstring) from one image's maps (4 x gh x gw)."""
    k, o, a1, a2 = (np.asarray(maps[c], np.float64) for c in range(4))
    s = STRIDE
    out = []
    frame_pad = max(2, int(round(4 * H / 544)))
    for rows, cols in components(k > cfg["thr"]):
        n = rows.size
        if n < cfg["min_cells"]:
            continue
        vertical = bool(o[rows, cols].sum() > 0)
        across_idx, along_idx = (cols, rows) if vertical else (rows, cols)
        c2 = s * across_idx + (s - 1) / 2.0
        near = float(np.mean(c2 - s * a1[rows, cols]))
        far = float(np.mean(c2 + s * a2[rows, cols]))
        if far <= near:
            continue
        lo, hi = float(s * along_idx.min()), float(s * (along_idx.max() + 1))
        es = cfg["along"] if cfg["ext_start"] < 0 else cfg["ext_start"]
        ee = cfg["along"] if cfg["ext_end"] < 0 else cfg["ext_end"]
        lo, hi = lo - es * (far - near), hi + ee * (far - near)
        if cfg["pad"] == "pct":
            p = max(2.0, 0.18 * (far - near))
            near, far, lo, hi = near - p, far + p, lo - p, hi + p
        if vertical:
            x0, x1, y0, y1 = near / fx, far / fx, lo / fy, hi / fy
        else:
            x0, x1, y0, y1 = lo / fx, hi / fx, near / fy, far / fy
        if cfg["pad"] == "frame":
            x0, y0, x1, y1 = x0 - frame_pad, y0 - frame_pad, x1 + frame_pad, y1 + frame_pad
        x0, y0, x1, y1 = max(0.0, x0), max(0.0, y0), min(float(W), x1), min(float(H), y1)
        if x1 - x0 < 1 or y1 - y0 < 1:
            continue
        out.append([round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2), int(vertical),
                    round(float(k[rows, cols].mean()), 4), -1])
    out.sort(key=lambda b: (b[1], b[0]))
    return out


class MidaDet:
    """The detector on ONNX Runtime's CPU provider (eval/desktop.py's detector interface: .boxes(rgb))."""

    name = "mida"

    def __init__(self, path, threads=1, session_options=None, sha256=SHA256, **cfg):
        import onnxruntime as ort
        self.facts, data = check_model(path, sha256)       # always: there is no unchecked path
        self.cfg = dict(DEFAULTS, **cfg)
        if session_options is None:
            session_options = ort.SessionOptions()
            session_options.intra_op_num_threads = int(threads)
            session_options.inter_op_num_threads = 1
            session_options.log_severity_level = 3
        self.sess = ort.InferenceSession(data, session_options, providers=["CPUExecutionProvider"])

    def maps(self, x):
        return self.sess.run([OUTPUT], {INPUT: x})[0][0]

    def boxes(self, rgb):
        H, W = rgb.shape[:2]
        x, fx, fy = preprocess(rgb, self.cfg)
        return decode(self.maps(x), fx, fy, W, H, self.cfg)


def detector_from_spec(spec, threads=1, session_options=None, default_path=None):
    path, cfg = parse_spec(spec)
    path = path or default_path
    if not path:
        raise ValueError(f"{spec!r}: no model path (mida:PATH or $MIDA_MODEL)")
    return MidaDet(path, threads=threads, session_options=session_options, **cfg)


def macs(path, w=960, h=544, sha256=SHA256):
    """Multiply-accumulates of one run at w x h (Conv and MatMul, from every node's output shape) and the network's
    input / output shapes: the graph run once by onnxruntime with every node output exposed."""
    import onnx
    import onnxruntime as ort
    _, m = check_bytes(Path(path).read_bytes(), sha256, Path(path).name)   # read once, checked, parsed
    have = {o.name for o in m.graph.output}
    for n in m.graph.node:
        for out in n.output:
            if out and out not in have:
                m.graph.output.append(onnx.helper.make_empty_tensor_value_info(out))
                have.add(out)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    so.intra_op_num_threads = 1
    sess = ort.InferenceSession(m.SerializeToString(), so, providers=["CPUExecutionProvider"])
    x = np.zeros((1, 3, h, w), np.float32)
    names = [o.name for o in sess.get_outputs()]
    shape = {k: tuple(np.asarray(v).shape) for k, v in zip(names, sess.run(names, {INPUT: x}))}
    shape[INPUT] = x.shape
    shape.update({t.name: tuple(t.dims) for t in m.graph.initializer})
    conv = mm = 0
    for n in m.graph.node:
        if n.op_type == "Conv":
            wt = shape[n.input[1]]
            conv += int(np.prod(shape[n.output[0]])) * int(wt[1]) * int(np.prod(wt[2:]))
        elif n.op_type == "MatMul":
            mm += int(np.prod(shape[n.output[0]])) * int(shape[n.input[0]][-1])
    return {"input": [w, h], "maps": list(shape[OUTPUT][1:]), "macs": conv + mm, "conv_macs": conv,
            "matmul_macs": mm, "attention_tokens": int(math.ceil(h / 32) * math.ceil(w / 32))}


# ----------------------------------------------------------------------------- the desktop harness

class CRec:
    """A VOCR6.2 int8 blob read by the C runtime (runtime/vocr.c W8I, with the blob's head sections as Rubi runs
    them: the head gate where the blob has one, the two-level head where it has the trailer), greedy CTC; the
    interface eval/desktop.py's reader uses (.read(crop, invert).text)."""

    def __init__(self, blob, lib=None):
        import sys
        here = Path(__file__).resolve().parents[1]
        for d in (here / "runtime", here / "train"):
            if str(d) not in sys.path:
                sys.path.insert(0, str(d))
        import pyvocr

        import head2level
        data = Path(blob).read_bytes()
        L = pyvocr.load_lib(lib)
        probe = pyvocr.Vocr(data, mode="w8i", lib=L)
        two = head2level.read_trailer(data) is not None
        self.v = pyvocr.Vocr(data, mode="w8i", lib=L, head_gate="on" if probe.has_gate else "off",
                             two_level=1 if two else 0)

    def read(self, img, invert=False):
        from types import SimpleNamespace
        g = np.ascontiguousarray(img if img.ndim == 2 else img[..., 0])
        return SimpleNamespace(text=self.v.run_topk(g, 8, invert=bool(invert))["text"])


def install(D, default_path=None):
    """Register 'mida[:PATH][@...]' detector specs in eval/desktop.py's module D (its detector() factory, which
    desktop.py's detect command and desktop_time.py's pipelines both call), and C-runtime reading of a '.vocr'
    blob as desktop.py read's --rec. desktop.py itself is not changed."""
    if getattr(D, "_mida_installed", False):
        return
    base_detector, base_init_read = D.detector, D._init_read

    def detector(spec, threads=1, opt="all", spinning=None):
        if spec.startswith("mida"):
            return detector_from_spec(spec, threads, D.session_options(threads, opt, spinning), default_path)
        return base_detector(spec, threads, opt, spinning)

    def init_read(rec_path, threads):
        if str(rec_path).endswith(".vocr"):
            import vocr_onnx as V
            D._W["rec"], D._W["V"] = CRec(rec_path), V
            return
        base_init_read(rec_path, threads)

    D.detector, D._init_read, D._mida_installed = detector, init_read, True


MEM_CHILD = r"""
import json, sys
sys.path[:0] = {paths!r}
def hwm():
    for ln in open("/proc/self/status"):
        if ln.startswith("VmHWM:"):
            return int(ln.split()[1])
import desktop as D
import desktop_time as DT
import mida_det
mida_det.install(D, {model!r})
frames, _ = D.load_frames({frames!r})
frames = frames[:{limit}]
D.load_rgb(frames[0])
base = hwm()
p = DT.Pipeline({spec!r}, {threads}, {mode!r})
loaded = hwm()
n = 0
for fr in frames:
    n += len(p(D.load_rgb(fr)))
print(json.dumps({{"base_kib": base, "loaded_kib": loaded, "peak_kib": hwm(), "frames": len(frames), "lines": n}}))
"""


def _harness_paths(repo):
    repo = Path(repo).resolve()
    return [str(Path(__file__).resolve().parent)] + [str(repo / d) for d in
                                                      ("tools", "eval", "synth", "train", "teacher", "runtime", "")]


def main(argv=None):
    """  mida_det.py info MODEL [--macs WxH ...]           the static checks and the MACs
      mida_det.py desktop --repo DIR --model MODEL -- ARGS     eval/desktop.py ARGS (DIR's) with the mida backend
      mida_det.py time --repo DIR --model MODEL -- ARGS        eval/desktop_time.py ARGS (DIR's) with it
      mida_det.py mem --repo DIR --model MODEL --frames F --pipelines P,... --threads N --out M.json [--limit 60]
          desktop_time.py mem's measurement (a fresh process per pipeline, VmHWM) with the backend installed"""
    import argparse
    import json
    import os
    import subprocess
    import sys
    argv = list(sys.argv[1:] if argv is None else argv)
    fwd = []
    if "--" in argv:
        i = argv.index("--")
        argv, fwd = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=main.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("info")
    p.add_argument("model")
    p.add_argument("--macs", nargs="*", default=["960x544", "480x272", "1280x720", "1920x1080"])
    for name in ("desktop", "time", "mem"):
        p = sub.add_parser(name)
        p.add_argument("--repo", required=True, help="a checkout holding eval/desktop.py (feat/vocr-desktop)")
        p.add_argument("--model", default=os.environ.get("MIDA_MODEL"))
        if name == "mem":
            p.add_argument("--frames", required=True)
            p.add_argument("--pipelines", required=True)
            p.add_argument("--threads", default="4")
            p.add_argument("--limit", type=int, default=60)
            p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "info":
        facts, _ = check_model(a.model)
        facts["macs"] = [macs(a.model, *(int(v) for v in s.split("x"))) for s in a.macs]
        print(json.dumps(facts, indent=1))
        return
    if not a.model:
        raise SystemExit("--model (or $MIDA_MODEL): the mida_ocr_l.onnx file")
    check_model(a.model)
    paths = _harness_paths(a.repo)
    sys.path[:0] = [q for q in paths if q not in sys.path]
    import desktop as D
    install(D, a.model)
    if a.cmd == "desktop":
        return D.main(fwd)
    import desktop_time as DT
    if a.cmd == "time":
        return DT.main(fwd)
    res = {"kind": "vocr-desktop-mem/1", "host": DT.host(), "frames_file": str(a.frames),
           "frames_sha256": D.sha256_file(a.frames), "threads": int(a.threads), "limit": a.limit,
           "repo_head": D.git_head(), "results": {}}
    for spec in a.pipelines.split(","):
        mode = "par" if not (spec.startswith("meikiocr") or spec.startswith("c")) else "seq"
        code = MEM_CHILD.format(paths=paths, model=str(a.model), frames=str(a.frames), limit=int(a.limit), spec=spec,
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


if __name__ == "__main__":
    main()
