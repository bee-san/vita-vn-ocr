#!/usr/bin/env python3
"""ONNX exports of the VOCR recognizers and the textspot8 line spotter, for desktop apps (docs/onnx.md, issue #163).

  export_onnx.py recognizer --blob VOCR6.2_N_w8g2.vocr --mode exact|fast [--convs c|ort] --out VOCR6.2_N_exact.onnx
  export_onnx.py spotter --blob textspot8_t64n8c16_w8.vtsp --mode exact|int8 --out textspot8_exact.onnx
  export_onnx.py release --models DIR --out DIR [--tag TAG]    every file of the ONNX release, its manifest and
                                                               SHA256SUMS (the blobs are read from DIR, sha256-checked)

The published int8 blobs are read as they are (train/export.py read_blob, train/headgate.py, train/head2level.py,
synth/textspot6_twin.py); nothing is retrained or re-quantised. The graphs use standard ONNX operators only (opset 17,
IR version 8), so any ONNX Runtime from 1.13 on loads them. Every file is deterministic: the same blob gives the same
bytes.

RECOGNIZER. Input "grey": uint8 [1, 1, H, W], the line as runtime/vocr.c preprocesses it (H = 32, W the resized width
padded to a multiple of the stride by repeating the last column; tools/vocr_onnx.py preprocess). Outputs per CTC frame
(T = W / stride): "logits" [T, classes] float32; "best" [T] int64, each frame's class; "topk_logits" [T, K] and
"topk_classes" [T, K], the K largest logits, ties to the lower class (K = 8 by default, what Rubi's beam + LM reads);
"blank_logit" [T]; and, for a blob with a head gate or a two-level head, "gate_score" / "escape_score" [T]. Such a
blob's model has a second input, "head_sections" (bool scalar):
- true: Rubi's reading, runtime/vocr.c with vocr_opts.head_gate = ON and two_level = 1, computing only what the C
  code computes: a frame whose gate score is above the threshold gets its blank logit alone (best 0, top-K
  [blank, -inf, ...] with classes 0); a kept frame gets level 1 (the blank and the most frequent classes) and the
  other classes only where its escape score is above the escape's threshold; "logits" is -inf where a class was not
  computed and "escape_score" NaN on the frames the gate skipped;
- false: the plain blob's reading (head_gate OFF, two_level 0): the full head on every frame.

- exact: runtime/vocr.c's W8I arithmetic, every float32 operation in the C code's order, so every output equals the C
  runtime's bit for bit (tests/test_onnx_export.py; eval/onnx_parity.py on development data):
  - the stem and depthwise convolutions: each tap's product w * x through a grouped convolution whose kernels hold
    one weight each (a product and zeros: exact on every kernel), then bias, tap 0, tap 1, ... added in that order by
    CumSum;
  - every pointwise layer, the row collapse and the head: int8 weights stored as int8 (as uint8 with zero point 128);
    each position's activations quantised to int16 as quant16 does (q = sign(t) floor(|t| + 0.5), t = x (32767 / m),
    m = max |x|; s = m / 32767), q split into two base-256 digits, and per 512-term chunk the two digit products
    through two MatMulInteger (U8U8: exact on every MLAS kernel; U8S8 can saturate on AVX2), combined in int32
    (256 hi + lo, the chunk's exact sum), each chunk's sum converted to float32 and added in order, then
    S (scale s) + bias, ReLU and the residual, as mm_i16;
  - the head gate and the escape of the two-level head: their layer 1 through the same integer path on the head's
    quantised input, and their score sums b + w[0] h[0] + w[1] h[1] + ... in order (CumSum).
- fast: the same layers in plain float32: Conv and MatMul with the int8 weights dequantised (q scale, as W8F reads
  them) and ONNX Runtime's own kernels; only the gate and the escape keep the exact arithmetic (their thresholds can be
  close). Not bit-exact: within float rounding of W8F.

SPOTTER (textspot8, VTSP version 4 with a VTDT trunk). Input "grey": uint8 [1, 1, h, w], the network's input (the frame
or region, halved by its 2 x 2 mean when the blob's rule or the caller's scale says so; tools/vocr_onnx.py). Output
"head": int8 [1, 16 + 4 T, ceil(h / 8), ceil(w / 8)], the head's codes; tools/vocr_onnx.py decodes them into boxes as
runtime/vocr_textspot6.c does. Both modes equal the C runtime's codes bit for bit:
- exact: float32 convolutions on the integer codes (every sum of products is an integer below 2^24, so exact in any
  order: checked per channel at export), each layer's requantisation floor(((acc + b) m + 2^(sh - 1)) / 2^sh) in
  float64 (every value an integer below 2^53: checked at export), clamped to [0, 127] (ReLU) or [-128, 127];
- int8: the same with ConvInteger (uint8 codes with zero point 128) and int32 sums.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "train", ROOT / "synth", ROOT / "eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

OPSET = 17
IR_VERSION = 8
CHUNK = 512                  # runtime/vocr.c W8I_CHUNK: int32 sums of at most 512 int8 x int16 products
TOPK = 8
FORMAT = 1                   # metadata "vocr.format": bump when inputs, outputs or metadata change meaning
T_CONV, T_DW, T_PW, T_HEAD, T_ATTN = 1, 2, 3, 4, 5
F32 = np.float32


def _onnx():
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    return onnx, TensorProto, helper, numpy_helper


# ----------------------------------------------------------------------------- graph building

class Graph:
    """A small builder: nodes in order, initializers (small constants shared by value), deterministic names."""

    def __init__(self):
        _, self.TP, self.h, self.nh = _onnx()
        self.nodes, self.inits, self.inputs, self.outputs = [], [], [], []
        self._n = 0
        self._consts = {}

    def name(self, base):
        self._n += 1
        return f"{base}_{self._n}"

    def const(self, arr, base="c"):
        arr = np.asarray(arr)
        key = None
        if arr.size <= 64:
            key = (arr.dtype.str, arr.shape, arr.tobytes())
            if key in self._consts:
                return self._consts[key]
        nm = self.name(base)
        self.inits.append(self.nh.from_array(np.ascontiguousarray(arr), nm))
        if key is not None:
            self._consts[key] = nm
        return nm

    def i64(self, *v):
        return self.const(np.array(v, np.int64), "i64")

    def op(self, op_type, inputs, base=None, outputs=1, **attrs):
        outs = [self.name(base or op_type.lower()) for _ in range(outputs)]
        self.nodes.append(self.h.make_node(op_type, list(inputs), outs, name=self.name("n" + op_type), **attrs))
        return outs[0] if outputs == 1 else outs

    def input(self, name, elem, shape):
        self.inputs.append(self.h.make_tensor_value_info(name, elem, shape))
        return name

    def output(self, src, name, elem, shape):
        self.nodes.append(self.h.make_node("Identity", [src], [name], name=self.name("nOut")))
        self.outputs.append(self.h.make_tensor_value_info(name, elem, shape))

    def branch(self, build, graph_name):
        """A subgraph for an If: build() adds nodes (they may read the outer graph's values) and returns
        [(value, elem type)]; returns (GraphProto, its output names)."""
        saved, self.nodes = self.nodes, []
        outs = build()
        nodes, self.nodes = self.nodes, saved
        names, infos = [], []
        for v, elem in outs:
            nm = self.name(graph_name + "_out")
            nodes.append(self.h.make_node("Identity", [v], [nm], name=self.name("nOut")))
            names.append(nm)
            infos.append(self.h.make_tensor_value_info(nm, elem, None))
        return self.h.make_graph(nodes, graph_name, [], infos), names

    def model(self, graph_name, metadata, doc):
        onnx = _onnx()[0]
        g = self.h.make_graph(self.nodes, graph_name, self.inputs, self.outputs, self.inits)
        m = self.h.make_model(g, opset_imports=[self.h.make_opsetid("", OPSET)], ir_version=IR_VERSION,
                              producer_name="vita-vn-ocr tools/export_onnx.py", producer_version=str(FORMAT),
                              doc_string=doc)
        for k in sorted(metadata):
            e = m.metadata_props.add()
            e.key, e.value = k, metadata[k] if isinstance(metadata[k], str) else json.dumps(metadata[k],
                                                                                            ensure_ascii=False)
        onnx.checker.check_model(m, full_check=True)
        return m


# ----------------------------------------------------------------------------- recognizer pieces

def _shape_of(g, x, dims):
    """The dims of x's shape (a 1-D int64 tensor)."""
    return g.op("Slice", [g.op("Shape", [x]), g.i64(dims[0]), g.i64(dims[-1] + 1), g.i64(0)])


def conv_exact(g, x, op):
    """The stem or a depthwise op on x [1, C_in, H, W] float32 in runtime/vocr.c's order: acc = bias, then
    acc = acc + w x tap by tap (dy, then dx), each product and sum rounded on its own. One grouped convolution gives,
    per channel c, taps + 1 planes: the bias (a zero kernel with the conv's bias b_c: 0 + b_c = b_c) and each tap's
    product (a kernel holding that tap's weight alone: w x + zeros = w x, exact in any summation order, with or
    without fused multiply-adds); CumSum then adds them in that order. Returns [1, C, H_out, W_out]. Only the sign of
    a zero can come out otherwise than in C (-0 + +0 is +0), and an equal value: every stem and depthwise op of the
    VOCR blobs has a ReLU, which maps both zeros to +0."""
    kh, kw, sh, sw, ph, pw, cout = op["kh"], op["kw"], op["sh"], op["sw"], op["ph"], op["pw"], op["cout"]
    taps = kh * kw
    w = np.asarray(op["w"], F32)                          # [cout][taps]
    # kernel c (taps + 1) + 1 + t holds w[c][t] at tap t: the stored [cout][1 + taps] weights (a 0 for the bias
    # plane) times a constant one-hot mask, which ONNX Runtime folds into the kernel when it loads the model
    wx = np.concatenate([np.zeros((cout, 1), F32), w], axis=1).reshape(cout, taps + 1, 1, 1)
    mask = np.zeros((taps + 1, kh, kw), F32)
    for t in range(taps):
        mask[1 + t, t // kw, t % kw] = 1
    k = g.op("Reshape", [g.op("Mul", [g.const(wx, "w_taps"), g.const(mask, "onehot")]),
                         g.i64(cout * (taps + 1), 1, kh, kw)])
    bias = np.zeros(cout * (taps + 1), F32)
    if op["bias"] is not None:
        bias[::taps + 1] = np.asarray(op["bias"], F32)
    group = 1 if op["type"] == T_CONV else cout
    p = g.op("Conv", [x, k, g.const(bias, "b_onehot")], group=group, kernel_shape=[kh, kw],
             strides=[sh, sw], pads=[ph, pw, ph, pw], dilations=[1, 1])
    hw = _shape_of(g, p, [2, 3])
    p5 = g.op("Reshape", [p, g.op("Concat", [g.i64(1, cout, taps + 1), hw], axis=0)])  # [1, C, 1 + taps, Ho, Wo]
    acc = g.op("CumSum", [p5, g.const(np.array(2, np.int64), "axis")])
    y = g.op("Squeeze", [g.op("Slice", [acc, g.i64(taps), g.i64(taps + 1), g.i64(2)]), g.i64(2)])
    return g.op("Relu", [y]) if op["relu"] else y


def w8f_weight(g, q, scale, scale_shape):
    """W8F's float weights float(q) scale (runtime/vocr.c dequant8) as Cast and Mul on the stored int8 matrix: ONNX
    Runtime folds them into one float constant when it loads the model, and the file keeps the int8 bytes."""
    return g.op("Mul", [g.op("Cast", [g.const(np.ascontiguousarray(q, np.int8), "wq")], to=g.TP.FLOAT),
                        g.const(np.asarray(scale, F32).reshape(scale_shape), "scale")])


def conv_fast(g, x, op):
    kh, kw, cout = op["kh"], op["kw"], op["cout"]
    group = 1 if op["type"] == T_CONV else cout
    w = np.asarray(op["w"], F32).reshape(cout, 1, kh, kw)
    ins = [x, g.const(w, "w")]
    if op["bias"] is not None:
        ins.append(g.const(np.asarray(op["bias"], F32), "b"))
    y = g.op("Conv", ins, group=group, kernel_shape=[kh, kw], strides=[op["sh"], op["sw"]],
             pads=[op["ph"], op["pw"], op["ph"], op["pw"]], dilations=[1, 1])
    return g.op("Relu", [y]) if op["relu"] else y


class Quant:
    """A tensor's per-position int16 quantisation (runtime/vocr.c quant16) as two uint8 digits. x2d [K, P] float32,
    positions along axis 1: hi [K, P] = floor(q / 256) + 128 and lo [K, P] = q - 256 floor(q / 256), q = hi' 256 + lo
    with hi' = hi - 128; sx [1, P] = m / 32767."""

    def __init__(self, g, x2d):
        ax = g.op("Abs", [x2d])
        mx = g.op("ReduceMax", [ax], axes=[0], keepdims=1)
        ok = g.op("Greater", [mx, g.const(np.array(0, F32))])
        safe = g.op("Where", [ok, mx, g.const(np.array(1, F32))])
        inv = g.op("Div", [g.const(np.array(32767, F32)), safe])
        self.sx = g.op("Div", [mx, g.const(np.array(32767, F32))])
        t = g.op("Mul", [x2d, inv])
        q = g.op("Mul", [g.op("Sign", [t]), g.op("Floor", [g.op("Add", [g.op("Abs", [t]),
                                                                         g.const(np.array(0.5, F32))])])])
        q = g.op("Clip", [q, g.const(np.array(-32767, F32)), g.const(np.array(32767, F32))])
        hi = g.op("Floor", [g.op("Div", [q, g.const(np.array(256, F32))])])
        lo = g.op("Sub", [q, g.op("Mul", [hi, g.const(np.array(256, F32))])])
        self.hi = g.op("Cast", [g.op("Add", [hi, g.const(np.array(128, F32))])], to=g.TP.UINT8)
        self.lo = g.op("Cast", [lo], to=g.TP.UINT8)
        self._t = None

    def transposed(self, g):
        """(hi, lo) as [P, K] and sx as [P, 1], for a matrix on the right."""
        if self._t is None:
            self._t = (g.op("Transpose", [self.hi], perm=[1, 0]), g.op("Transpose", [self.lo], perm=[1, 0]),
                       g.op("Transpose", [self.sx], perm=[1, 0]))
        return self._t


class Rows:
    """Quantised positions already laid out [P, K] (hi, lo) and [P, 1] (sx): e.g. a Quant's rows gathered for a
    subset of the frames. Usable where a Quant is, on the right side only."""

    def __init__(self, hi, lo, sx):
        self._t = (hi, lo, sx)

    def transposed(self, g):
        return self._t

    def gather(self, g, idx):
        return Rows(*(g.op("Gather", [v, idx], axis=0) for v in self._t))


def int8_chunks(g, qd, w_q, side, w_name=None):
    """The exact float32 sums S = 0 + float(c_0) + float(c_1) + ... of an int8 matrix w_q [N][K] against the digits
    of qd (Quant), c_i the int32 sum over k in [512 i, 512 i + 512). side "left": the matrix on the left, S [N, P];
    "right": the activations on the left, S [P, N]. w_name: the matrix already stored as uint8 (w + 128), [K, N] on
    the right (w_q is then only its shape), for K <= 512.

    Bounds (int32 throughout, no wrap): the high digit h = floor(q / 256) is in [-128, 127] and the low one in
    [0, 255]; with |w| <= 127 a chunk's high sum is at most 512 128 127 = 8,323,072 in size, 256 times that
    2,130,706,432 < 2^31, and 256 hi + lo is the chunk's exact sum c_i (|c_i| <= 512 32767 127 < 2^31, vocr.c's own
    bound). train/export.py quantize_rows writes weights in [-127, 127]; a -128 is refused here."""
    N, K = w_q.shape
    if int(np.asarray(w_q).min()) < -127:
        raise SystemExit("an int8 weight of -128: train/export.py never writes one, and the exact graph's int32 "
                         "sums are bounded for |w| <= 127 only")
    zp128, zp0 = g.const(np.array(128, np.uint8), "zp"), g.const(np.array(0, np.uint8), "zp")
    hi, lo = (qd.hi, qd.lo) if side == "left" else qd.transposed(g)[:2]
    if w_name is not None:
        if side != "right" or K > CHUNK:
            raise ValueError("a stored matrix: the right side, K <= 512")
        s_hi = g.op("MatMulInteger", [hi, w_name, zp128, zp128])
        s_lo = g.op("MatMulInteger", [lo, w_name, zp0, zp128])
        return g.op("Cast", [g.op("Add", [g.op("Mul", [s_hi, g.const(np.array(256, np.int32))]), s_lo])],
                    to=g.TP.FLOAT)
    acc = None
    for k0 in range(0, K, CHUNK):
        k1 = min(K, k0 + CHUNK)
        wu = (w_q[:, k0:k1].astype(np.int16) + 128).astype(np.uint8)          # [N, Kc]
        if K > CHUNK:
            ax = 0 if side == "left" else 1
            h_c = g.op("Slice", [hi, g.i64(k0), g.i64(k1), g.i64(ax)])
            l_c = g.op("Slice", [lo, g.i64(k0), g.i64(k1), g.i64(ax)])
        else:
            h_c, l_c = hi, lo
        if side == "left":
            wn = g.const(wu, "wq")
            s_hi = g.op("MatMulInteger", [wn, h_c, zp128, zp128])
            s_lo = g.op("MatMulInteger", [wn, l_c, zp128, zp0])
        else:
            wn = g.const(np.ascontiguousarray(wu.T), "wq")
            s_hi = g.op("MatMulInteger", [h_c, wn, zp128, zp128])
            s_lo = g.op("MatMulInteger", [l_c, wn, zp0, zp128])
        s = g.op("Add", [g.op("Mul", [s_hi, g.const(np.array(256, np.int32))]), s_lo])
        f = g.op("Cast", [s], to=g.TP.FLOAT)
        acc = f if acc is None else g.op("Add", [acc, f])
    return acc


def matmul_exact(g, qd, w_q, scale, bias, relu, side, res=None, w_name=None):
    """mm_i16 of runtime/vocr.c: S (scale_j s_p) + bias_j, ReLU, + res. side and w_name as int8_chunks."""
    S = int8_chunks(g, qd, w_q, side, w_name)
    scale = np.asarray(scale, F32)
    if side == "left":
        ss = g.op("Mul", [g.const(scale.reshape(-1, 1), "scale"), qd.sx])
    else:
        ss = g.op("Mul", [qd.transposed(g)[2], g.const(scale.reshape(1, -1), "scale")])
    y = g.op("Mul", [S, ss])
    if bias is not None:
        b = np.asarray(bias, F32)
        y = g.op("Add", [y, g.const(b.reshape(-1, 1) if side == "left" else b.reshape(1, -1), "bias")])
    if relu:
        y = g.op("Relu", [y])
    if res is not None:
        y = g.op("Add", [y, res])
    return y


def ordered_sum(g, b0, terms):
    """b0 + terms[:, 0] + terms[:, 1] + ... in that order (float32): terms [P, n] -> [P]."""
    rows = _shape_of(g, terms, [0])
    first = g.op("Expand", [g.const(np.array([[b0]], F32), "b0"), g.op("Concat", [rows, g.i64(1)], axis=0)])
    acc = g.op("CumSum", [g.op("Concat", [first, terms], axis=1), g.const(np.array(1, np.int64), "axis")])
    return g.op("Squeeze", [g.op("Slice", [acc, g.i64(-1), g.i64(np.iinfo(np.int64).max), g.i64(1)]), g.i64(1)])


def load_recognizer(data):
    import export
    import head2level
    b = export.read_blob(data)
    if b["int8"] is not True:
        raise SystemExit("a float blob: export_onnx.py reads the int8 blobs (W8I) only")
    if any(op["type"] == T_ATTN for op in b["ops"]):
        raise SystemExit("a blob with an attention op: not supported by this exporter")
    two = head2level.read_trailer(data)
    if two is not None and two["rare_k"]:
        raise SystemExit("a two-level trailer with a rare block: not supported (runtime/vocr.c refuses it too)")
    return b, b["gate"], two


def recognizer(data, mode="exact", topk=TOPK, name="", charbox_cal=None, extra_meta=None, convs="c"):
    """The ONNX ModelProto of a VOCR int8 blob (bytes). mode: exact | fast. convs (exact only): "c", the stem and
    depthwise convolutions in runtime/vocr.c's order (the released files); "ort", ONNX Runtime's own float32 Conv
    there, the rest exact: astra's "native INT8 keeping 16-bit activations" form, for comparison (eval/onnx_timing.py;
    not bit-exact)."""
    if mode not in ("exact", "fast") or convs not in ("c", "ort"):
        raise ValueError("mode: exact or fast; convs: c or ort")
    b, gate, two = load_recognizer(data)
    TP = _onnx()[1]
    g = Graph()
    H, N, S = b["height"], b["n_classes"], b["stride"]
    grey = g.input("grey", TP.UINT8, [1, 1, H, "width"])
    x = g.op("Sub", [g.op("Div", [g.op("Cast", [grey], to=TP.FLOAT), g.const(np.array(127.5, F32))]),
                     g.const(np.array(1, F32))])
    ops = b["ops"]
    dw_in = None
    z = None                       # the head's input, [K, T]
    for i, op in enumerate(ops):
        t = op["type"]
        if t in (T_CONV, T_DW):
            dw_in = x
            x = (conv_exact if mode == "exact" and convs == "c" else conv_fast)(g, x, op)
            continue
        if t == T_HEAD:
            break
        # a pointwise layer, maybe a row collapse: x [1, C, h, w]
        if mode == "fast":
            q = np.asarray(op["w"], np.int8)                                    # [cout][K]
            if op["collapse"]:
                qk = q.reshape(op["cout"], op["h_in"], op["cin"]).transpose(0, 2, 1)[:, :, :, None]
                kh = op["h_in"]
            else:
                qk, kh = q[:, :, None, None], 1
            ins = [x, w8f_weight(g, qk, op["scale"], (-1, 1, 1, 1))]
            if op["bias"] is not None:
                ins.append(g.const(np.asarray(op["bias"], F32), "b"))
            y = g.op("Conv", ins, kernel_shape=[kh, 1], strides=[1, 1], pads=[0, 0, 0, 0])
            if op["relu"]:
                y = g.op("Relu", [y])
            if op["res"]:
                y = g.op("Add", [y, dw_in])
            x = y
            continue
        if op["collapse"]:
            xt = g.op("Transpose", [x], perm=[0, 2, 1, 3])                         # [1, h, C, w]
            x2 = g.op("Reshape", [xt, g.i64(op["h_in"] * op["cin"], -1)])          # k = row C + c
            out_shape = g.op("Concat", [g.i64(1, op["cout"], 1), _shape_of(g, x, [3])], axis=0)
        else:
            x2 = g.op("Reshape", [x, g.i64(op["cin"], -1)])
            out_shape = g.op("Concat", [g.i64(1, op["cout"]), _shape_of(g, x, [2, 3])], axis=0)
        res = g.op("Reshape", [dw_in, g.i64(op["cout"], -1)]) if op["res"] else None
        qd = Quant(g, x2)
        y = matmul_exact(g, qd, op["w"], op["scale"], op["bias"], op["relu"], "left", res)
        if i + 1 < len(ops) and ops[i + 1]["type"] == T_HEAD:
            z = y
        x = g.op("Reshape", [y, out_shape])
    head = ops[-1]
    if head["type"] != T_HEAD:
        raise SystemExit("the last op is not the head")
    if z is None:                                       # fast: the head's input from the last layer's output
        z = g.op("Reshape", [x, g.i64(head["k"], -1)])
    qz = Quant(g, z)                                    # the gate and the escape always read this (W8I's) form
    rows = Rows(*qz.transposed(g))                      # every frame's quantised head input, [T, K]
    zt = g.op("Transpose", [z], perm=[1, 0]) if mode == "fast" else None
    hw = np.asarray(head["w"])
    hs, hb = np.asarray(head["scale"], F32), np.asarray(head["bias"], F32)
    if hw.shape[1] > CHUNK:
        raise SystemExit("a head wider than 512 inputs: not supported")
    # the head's matrix stored once, [K, classes] (uint8 w + 128 exact; int8 w with W8F's dequantisation fast); a class
    # range is a Slice of it, which ONNX Runtime folds into a constant when it loads the model
    if mode == "exact":
        w_full = g.const(np.ascontiguousarray(((hw.astype(np.int16) + 128).astype(np.uint8)).T), "w_head")
    else:
        w_full = w8f_weight(g, np.ascontiguousarray(hw.T), hs, (1, -1))
    ninf = g.const(np.array(-np.inf, F32), "ninf")

    def head_rows(src, r0, r1):
        """The head's classes [r0, r1) on a frame source: (Rows, float rows [P, K] or None) -> [P, r1 - r0]."""
        w = w_full if (r0, r1) == (0, N) else g.op("Slice", [w_full, g.i64(r0), g.i64(r1), g.i64(1)])
        if mode == "exact":
            return matmul_exact(g, src[0], hw[r0:r1], hs[r0:r1], hb[r0:r1], False, "right", w_name=w)
        return g.op("Add", [g.op("MatMul", [src[1], w]), g.const(hb[r0:r1].reshape(1, -1), "b")])

    def gather(src, idx):
        return (src[0].gather(g, idx), g.op("Gather", [src[1], idx], axis=0) if src[1] is not None else None)

    def escape(src, l1):
        """The two-level escape's score per frame of src, l1 its level-1 logits (runtime/vocr.c run_head2)."""
        e = two["esc"]
        eh = matmul_exact(g, src[0], e["q"], e["scale"], e["bias"], False, "right")             # [P, hidden]
        nbv = g.op("ReduceMax", [g.op("Slice", [l1, g.i64(1), g.i64(L1), g.i64(1)])], axes=[1], keepdims=1)
        b0v = g.op("Slice", [l1, g.i64(0), g.i64(1), g.i64(1)])
        a = g.op("Add", [eh, g.op("Mul", [nbv, g.const(e["wl1"].reshape(1, -1), "wl1")])])
        a = g.op("Add", [a, g.op("Mul", [b0v, g.const(e["wb0"].reshape(1, -1), "wb0")])])
        ep = g.op("Mul", [g.op("Relu", [a]), g.const(e["w2"].reshape(1, -1), "w2")])
        return ordered_sum(g, e["b2"], ep)

    def top(lg):
        tv, tc = g.op("TopK", [lg, g.i64(topk)], outputs=2, axis=1, largest=1, sorted=1)
        return tv, tc

    def finish(lg, tv, tc, blank, es):
        # runtime/vocr.c's top-K of a frame the gate skipped: [blank, -inf, ...] with classes 0; best is the first
        # maximum (TopK breaks ties to the lower class, as the C code's strict compares do)
        tc = g.op("Where", [g.op("Equal", [tv, ninf]), g.const(np.array(0, np.int64)), tc])
        best = g.op("Squeeze", [g.op("Slice", [tc, g.i64(0), g.i64(1), g.i64(1)]), g.i64(1)])
        out = [(lg, TP.FLOAT), (best, TP.INT64), (tv, TP.FLOAT), (tc, TP.INT64), (blank, TP.FLOAT)]
        return out + ([(es, TP.FLOAT)] if es is not None else [])

    full = (rows, zt)
    L1 = 8 * two["l1_blocks"] if two is not None else N
    meta = {}
    gs = None
    if gate is not None:
        gh = matmul_exact(g, qz, gate["q"], gate["scale"], gate["bias"], True, "right")         # [T, hidden]
        gs = ordered_sum(g, gate["b2"], g.op("Mul", [gh, g.const(np.asarray(gate["w2"], F32).reshape(1, -1), "w2")]))
        g.output(gs, "gate_score", TP.FLOAT, ["frames"])
        meta["vocr.gate_threshold"] = float(gate["t"])
    if two is not None:
        meta["vocr.escape_threshold"] = float(two["esc"]["t"])
        meta["vocr.level1_classes"] = L1

    def plain():
        """The plain blob's reading: the full head on every frame (vocr_opts.head_gate OFF, two_level 0)."""
        lg = head_rows(full, 0, N)
        es = escape(full, lg) if two is not None else None
        tv, tc = top(lg)
        blank = g.op("Squeeze", [g.op("Slice", [lg, g.i64(0), g.i64(1), g.i64(1)]), g.i64(1)])
        return finish(lg, tv, tc, blank, es)

    def sections():
        """Rubi's reading (head_gate ON, two_level 1): the head only where runtime/vocr.c computes it. A frame the gate
        skips gets its blank logit alone; a kept frame level 1 (the blank and the most frequent classes), and the
        other classes only where the escape fires; "logits" holds -inf where a class was not computed."""
        src, idx, kept_n = full, None, None
        if gs is not None:
            keep = g.op("Not", [g.op("Greater", [gs, g.const(np.array(gate["t"], F32))])])
            idx = g.op("Squeeze", [g.op("NonZero", [keep]), g.i64(0)])                      # [K'] kept frames
            src = gather(full, idx)
            kept_n = _shape_of(g, idx, [0])
        es = None
        if two is not None:
            l1 = head_rows(src, 0, L1)                                                       # [K', L1]
            es = escape(src, l1)
            idx2 = g.op("Squeeze", [g.op("NonZero", [g.op("Greater", [es, g.const(np.array(two["esc"]["t"], F32))])]),
                                    g.i64(0)])                                               # [F] escapes, in src
            rare = head_rows(gather(src, idx2), L1, N)                                       # [F, N - L1]
            n_src = kept_n if kept_n is not None else _shape_of(g, rows.transposed(g)[0], [0])
            base = g.op("Expand", [ninf, g.op("Concat", [n_src, g.i64(N - L1)], axis=0)])
            rare_all = g.op("ScatterND", [base, g.op("Unsqueeze", [idx2, g.i64(1)]), rare])
            lk = g.op("Concat", [l1, rare_all], axis=1)                                      # [K', N]
        else:
            lk = head_rows(src, 0, N)
        if idx is None:
            tv, tc = top(lk)
            blank = g.op("Squeeze", [g.op("Slice", [lk, g.i64(0), g.i64(1), g.i64(1)]), g.i64(1)])
            return finish(lk, tv, tc, blank, es)
        # the kept frames' top-K, with one -inf row appended and cut off again: ONNX Runtime's TopK (1.30) crashes on
        # an input without rows, which a line whose every frame the gate skips would give
        pad = g.const(np.full((1, N), -np.inf, F32), "pad")
        tv, tc = top(g.op("Concat", [lk, pad], axis=0))
        tv = g.op("Slice", [tv, g.i64(0), kept_n, g.i64(0)])
        tc = g.op("Slice", [tc, g.i64(0), kept_n, g.i64(0)])
        b_all = head_rows(full, 0, 1)                                                        # [T, 1]: every blank
        n_all = _shape_of(g, b_all, [0])
        ix = g.op("Unsqueeze", [idx, g.i64(1)])
        lg = g.op("ScatterND", [g.op("Concat", [b_all, g.op("Expand", [ninf, g.op("Concat", [n_all, g.i64(N - 1)],
                                                                                  axis=0)])], axis=1), ix, lk])
        tv = g.op("ScatterND", [g.op("Concat", [b_all, g.op("Expand", [ninf, g.op("Concat", [n_all, g.i64(topk - 1)],
                                                                                  axis=0)])], axis=1), ix, tv])
        tc = g.op("ScatterND", [g.op("Expand", [g.const(np.array(0, np.int64)), g.op("Concat", [n_all, g.i64(topk)],
                                                                                         axis=0)]), ix, tc])
        if es is not None:
            es = g.op("ScatterND", [g.op("Expand", [g.const(np.array(np.nan, F32)), n_all]), ix, es])
        return finish(lg, tv, tc, g.op("Squeeze", [b_all, g.i64(1)]), es)

    names = ["logits", "best", "topk_logits", "topk_classes", "blank_logit"] + (["escape_score"] if two is not None
                                                                                  else [])
    shapes = [["frames", N], ["frames"], ["frames", topk], ["frames", topk], ["frames"], ["frames"]]
    elems = [TP.FLOAT, TP.INT64, TP.FLOAT, TP.INT64, TP.FLOAT, TP.FLOAT]
    if gate is not None or two is not None:
        on = g.input("head_sections", TP.BOOL, [])
        then_g, _ = g.branch(sections, "sections")
        else_g, _ = g.branch(plain, "plain")
        outs = g.op("If", [on], outputs=len(names), then_branch=then_g, else_branch=else_g)
    else:
        outs = [v for v, _ in plain()]
    for v, nm, sh, el in zip(outs, names, shapes, elems):
        g.output(v, nm, el, sh)
    meta.update({
        "vocr.format": FORMAT, "vocr.kind": "recognizer", "vocr.mode": mode, "vocr.name": name,
        "vocr.source_sha256": hashlib.sha256(data).hexdigest(), "vocr.source_bytes": len(data),
        "vocr.height": H, "vocr.stride": S, "vocr.max_width": b["max_width"], "vocr.min_width": 8,
        "vocr.classes": N, "vocr.blank": 0, "vocr.topk": topk, "vocr.vocab": b["vocab"],
        "vocr.head_gate": gate is not None, "vocr.two_level": two is not None,
        "vocr.input": "grey uint8 [1, 1, height, width]: tools/vocr_onnx.py preprocess (runtime/vocr.c's resize, "
                      "integer luma, 255 - grey for dark text, right pad to a multiple of the stride)",
    })
    if convs != "c":
        meta["vocr.convs"] = convs                  # the comparison form: not bit-exact
    if charbox_cal is not None:
        meta["vocr.charbox_cal"] = [int(v) for v in charbox_cal]
    meta.update(extra_meta or {})
    doc = (f"VOCR recognizer {name} ({mode}), exported by vita-vn-ocr tools/export_onnx.py from the int8 blob "
           f"sha256 {meta['vocr.source_sha256']}. docs/onnx.md.")
    return g.model("vocr_recognizer", meta, doc)


# ----------------------------------------------------------------------------- spotter

def load_spotter(data):
    import textspot6_twin as T6
    m = T6.Model(data)
    if m.version != T6.VERSION_TAG or m.form != "trunk":
        raise SystemExit("export_onnx.py exports VTSP version 4 blobs with a VTDT trunk (textspot8) only")
    return m


def check_spotter_bounds(m):
    """Every layer's exactness bounds: |conv sum| <= 128 sum|w| < 2^24 (float32 sums exact in any order) and
    (128 sum|w| + |b|) m + 2^(sh - 1) < 2^53 (the float64 requantisation exact). Returns the margins."""
    worst24 = worst53 = 0.0
    for i, L in enumerate(m.net.layers):
        w = np.abs(L.w.astype(np.int64)).reshape(L.cout, -1).sum(1)
        s = 128 * w
        if (s >= 2 ** 24).any():
            raise SystemExit(f"layer {i}: a float32 conv sum could reach 2^24: not exact")
        v = (s + np.abs(L.b)) * L.m + (np.int64(1) << (L.sh - 1))
        if (v >= 2 ** 53).any():
            raise SystemExit(f"layer {i}: the float64 requantisation could reach 2^53: not exact")
        worst24 = max(worst24, float(s.max()) / 2 ** 24)
        worst53 = max(worst53, float(v.max()) / 2 ** 53)
    return {"conv_sum_over_2^24": round(worst24, 6), "requant_over_2^53": round(worst53, 6)}


def spotter(data, mode="exact", name="", extra_meta=None):
    """The ONNX ModelProto of a textspot8-style VTSP version 4 blob (bytes). mode: exact (float32 convs, float64
    requantisation) | int8 (ConvInteger)."""
    if mode not in ("exact", "int8"):
        raise ValueError("mode: exact or int8")
    import textspot6_twin as T6
    m = load_spotter(data)
    bounds = check_spotter_bounds(m)
    TP = _onnx()[1]
    g = Graph()
    grey = g.input("grey", TP.UINT8, [1, 1, "height", "width"])
    if mode == "exact":
        x = g.op("Sub", [g.op("Cast", [grey], to=TP.FLOAT), g.const(np.array(128, F32))])
    else:
        x = grey                                             # codes + 128 (the input's codes are grey - 128)
    layers = m.net.layers
    for i, L in enumerate(layers):
        k, s, d = L.k, L.stride, L.dil
        p = (k - 1) // 2 * d
        group = L.cin if L.kind == 1 else 1
        attrs = dict(kernel_shape=[k, k], strides=[s, s], pads=[p, p, p, p], dilations=[d, d], group=group)
        if mode == "exact":
            wk = g.op("Cast", [g.const(L.w.astype(np.int8), "wq")], to=TP.FLOAT)     # folded into a constant at load
            acc = g.op("Conv", [x, wk], **attrs)
            acc = g.op("Cast", [acc], to=TP.DOUBLE)
        else:
            wu = (L.w.astype(np.int16) + 128).astype(np.uint8)
            zp = g.const(np.array(128, np.uint8), "zp")
            acc = g.op("Cast", [g.op("ConvInteger", [x, g.const(wu, "wq"), zp, zp], **attrs)], to=TP.DOUBLE)
        b = L.b.astype(np.float64).reshape(1, -1, 1, 1)
        mult = L.m.astype(np.float64).reshape(1, -1, 1, 1)
        half = np.ldexp(1.0, (L.sh - 1).astype(np.int64)).reshape(1, -1, 1, 1)
        inv = np.ldexp(1.0, (-L.sh).astype(np.int64)).reshape(1, -1, 1, 1)
        v = g.op("Mul", [g.op("Add", [acc, g.const(b, "b")]), g.const(mult, "m")])
        v = g.op("Floor", [g.op("Mul", [g.op("Add", [v, g.const(half, "half")]), g.const(inv, "inv")])])
        v = g.op("Clip", [v, g.const(np.array(0.0 if L.relu else -128.0)), g.const(np.array(127.0))])
        if i + 1 == len(layers):
            x = g.op("Cast", [v], to=TP.INT8)
        elif mode == "exact":
            x = g.op("Cast", [v], to=TP.FLOAT)
        else:
            x = g.op("Cast", [g.op("Add", [v, g.const(np.array(128.0))])], to=TP.UINT8)
    g.output(x, "head", TP.INT8, [1, T6.outputs_of(m.tags), "rows", "cols"])
    meta = {
        "vocr.format": FORMAT, "vocr.kind": "spotter", "vocr.mode": mode, "vocr.name": name,
        "vocr.source_sha256": hashlib.sha256(data).hexdigest(), "vocr.source_bytes": len(data),
        "vocr.vtsp_version": m.version, "vocr.tags": m.tags, "vocr.classes": list(T6.CLASSES[:m.tags + 1]),
        "vocr.half_above": m.half_above, "vocr.half_above_h": m.half_above_h, "vocr.conf_min": m.conf_min,
        "vocr.min_cells": m.min_cells, "vocr.net_stride": m.net_stride, "vocr.map_stride": m.stride,
        "vocr.exact_bounds": bounds,
        "vocr.input": "grey uint8 [1, 1, h, w]: the frame or region, halved by its 2x2 mean at scale 2 "
                      "(tools/vocr_onnx.py)",
    }
    meta.update(extra_meta or {})
    doc = (f"VOCR line spotter {name} ({mode}), exported by vita-vn-ocr tools/export_onnx.py from the VTSP blob "
           f"sha256 {meta['vocr.source_sha256']}. docs/onnx.md.")
    return g.model("vocr_spotter", meta, doc)


# ----------------------------------------------------------------------------- the release

# (output name, source blob, tier for charbox calibration) of the release's recognizers; the tiers are Rubi's
RECOGNIZERS = (("VOCR6.2_S", "VOCR6.2_S_w8.vocr", "S"), ("VOCR6.2_N", "VOCR6.2_N_w8g2.vocr", "N"),
               ("VOCR6.2_L", "VOCR6.2_L_w8_2l.vocr", "L"))
SPOTTERS = (("textspot8", "textspot8_t64n8c16_w8.vtsp"),)
SOURCE_RELEASE = "v0.5.0"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repo_commit():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def charbox_cals():
    """eval/charboxes_cal.json's calibration per tier ([shift, group 0..7] in 1/16 resized px; fit on VOCR5)."""
    d = json.loads((ROOT / "eval" / "charboxes_cal.json").read_text(encoding="utf-8"))
    return {t: v["cal"] for t, v in d["models"].items()}


def release(models_dir, out_dir, modes=("exact", "fast"), spot_modes=("exact", "int8")):
    """Every ONNX file of the release from the published blobs in models_dir (checked against models_dir's
    SHA256SUMS when it is there), and onnx-manifest.json. Returns the manifest. The files hold no commit id or date, so
    the same exporter code and blobs give the same bytes anywhere; the manifest records the commit they came from."""
    import onnx
    models_dir, out_dir = Path(models_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sums = {}
    if (models_dir / "SHA256SUMS").exists():
        for line in (models_dir / "SHA256SUMS").read_text().splitlines():
            if line.strip():
                h, f = line.split(None, 1)
                sums[f.strip().lstrip("*")] = h
    cals = charbox_cals()
    commit = repo_commit()
    files = []

    def put(model, fname, src, extra):
        data = model.SerializeToString()
        (out_dir / fname).write_bytes(data)
        onnx.load_from_string(data)
        files.append({"file": fname, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                      "source": src, "source_sha256": sha256(models_dir / src), **extra})

    for stem, src, tier in RECOGNIZERS:
        blob = (models_dir / src).read_bytes()
        if src in sums and sums[src] != hashlib.sha256(blob).hexdigest():
            raise SystemExit(f"{src}: sha256 differs from {models_dir / 'SHA256SUMS'}")
        for mode in modes:
            meta = {"vocr.source_file": src, "vocr.source_release": SOURCE_RELEASE, "vocr.tier": tier,
                    "vocr.licence": LICENCE, "vocr.attribution": "VOCR6-ATTRIBUTION.md",
                    "vocr.charbox_cal_source": CAL_SOURCE.format(tier=tier)}
            m = recognizer(blob, mode, name=stem, charbox_cal=cals.get(tier), extra_meta=meta)
            put(m, f"{stem}_{mode}.onnx", src, {"kind": "recognizer", "mode": mode, "tier": tier})
    for stem, src in SPOTTERS:
        blob = (models_dir / src).read_bytes()
        if src in sums and sums[src] != hashlib.sha256(blob).hexdigest():
            raise SystemExit(f"{src}: sha256 differs from {models_dir / 'SHA256SUMS'}")
        for mode in spot_modes:
            meta = {"vocr.source_file": src, "vocr.source_release": SOURCE_RELEASE, "vocr.licence": LICENCE,
                    "vocr.attribution": "SPOTTER-ATTRIBUTION.md"}
            put(spotter(blob, mode, name=stem, extra_meta=meta), f"{stem}_{mode}.onnx", src,
                {"kind": "spotter", "mode": mode})
    manifest = {"kind": "vocr-onnx-release", "format": FORMAT, "source_release": SOURCE_RELEASE,
                "repo_commit": commit, "opset": OPSET, "ir_version": IR_VERSION, "files": files}
    (out_dir / "onnx-manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n",
                                                encoding="utf-8")
    return manifest


LICENCE = ("Weights: GPL-3.0-or-later (bee-san/vita-vn-ocr), for non-commercial use only: the training inputs carry "
           "non-commercial terms (see the attribution file).")
CAL_SOURCE = ("eval/charboxes_cal.json, fit on VOCR5-{tier} (docs/char-boxes.md); VOCR6-{tier} is a fine-tune of the "
              "same shape and was not refit")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("recognizer")
    r.add_argument("--blob", required=True)
    r.add_argument("--mode", default="exact", choices=("exact", "fast"))
    r.add_argument("--topk", type=int, default=TOPK)
    r.add_argument("--convs", default="c", choices=("c", "ort"),
                   help="exact mode: c (the C runtime's order, bit-exact; the default) or ort (ONNX Runtime's float "
                        "convolutions: astra's native-int8 form, for comparison)")
    r.add_argument("--out", required=True)
    s = sub.add_parser("spotter")
    s.add_argument("--blob", required=True)
    s.add_argument("--mode", default="exact", choices=("exact", "int8"))
    s.add_argument("--out", required=True)
    rl = sub.add_parser("release")
    rl.add_argument("--models", required=True, help="the directory holding v0.5.0's blobs (and its SHA256SUMS)")
    rl.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "recognizer":
        data = Path(a.blob).read_bytes()
        m = recognizer(data, a.mode, a.topk, name=Path(a.blob).stem, convs=a.convs)
        Path(a.out).write_bytes(m.SerializeToString())
        print(f"{a.out}: {Path(a.out).stat().st_size} B, {len(m.graph.node)} nodes")
    elif a.cmd == "spotter":
        data = Path(a.blob).read_bytes()
        m = spotter(data, a.mode, name=Path(a.blob).stem)
        Path(a.out).write_bytes(m.SerializeToString())
        print(f"{a.out}: {Path(a.out).stat().st_size} B, {len(m.graph.node)} nodes")
    else:
        man = release(a.models, a.out)
        for f in man["files"]:
            print(f"{f['file']}: {f['bytes']} B sha256 {f['sha256'][:16]}")


if __name__ == "__main__":
    main()
