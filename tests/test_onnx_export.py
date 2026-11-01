"""The ONNX exports (tools/export_onnx.py) and the reference pipeline (tools/vocr_onnx.py), on generated fixtures only:
random-init recognizers (with a head gate and a two-level head appended), random spotter blobs, random LMs, drawn
crops and frames. The exact files must equal the C runtime (runtime/vocr.c through runtime/pyvocr.py) and its bit-exact
twins bit for bit; the fast files must stay within float rounding of W8F. No development, test or private data is read
here (eval/onnx_parity.py runs the same checks on development data, docs/onnx.md).
"""
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "tools", ROOT / "runtime", ROOT / "train", ROOT / "synth", ROOT / "eval", ROOT / "lm",
          ROOT / "runtime" / "tests", ROOT / "tests"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

import export_onnx as X  # noqa: E402
import vocr_onnx as V  # noqa: E402

F32 = np.float32
VOCAB = [chr(0x4E00 + i) for i in range(200)] + list("「」『』（）…、。ー")


def session(model, threads=1):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.log_severity_level = 3
    return ort.InferenceSession(model.SerializeToString(), so, providers=["CPUExecutionProvider"])


def bits(a):
    return np.ascontiguousarray(np.asarray(a, F32)).view(np.uint32)


# ----------------------------------------------------------------------------- the exact pieces against vocr_ref

def _graph(build, inputs, rank=2):
    g = X.Graph()
    names = [g.input(n, X._onnx()[1].FLOAT, list(shape)) for n, shape in inputs]
    out = build(g, *names)
    g.output(out, "y", X._onnx()[1].FLOAT, [f"d{i}" for i in range(rank)])
    return session(g.model("t", {"vocr.format": X.FORMAT}, ""))


def edge_rows(rng, P, K):
    """Rows that stress quant16 and the 512-term chunks: random, all zero, tiny, alternating extremes, one hot."""
    x = rng.normal(0, 1, (P, K)).astype(F32) * rng.uniform(0.01, 100, (P, 1)).astype(F32)
    x[0] = 0
    x[1] = F32(1e-30) * np.sign(rng.normal(size=K)).astype(F32)
    x[2] = np.where(np.arange(K) % 2, F32(3e4), F32(-3e4))
    x[3] = 0
    x[3, K // 2] = F32(-7.5)
    x[4] = np.round(rng.normal(0, 3, K)) / 2                      # many exact halves after scaling
    return x


@pytest.mark.parametrize("K", [16, 511, 512, 513, 1024, 1792])
def test_matmul_exact_is_vocr_ref(K):
    import vocr_ref as R
    rng = np.random.default_rng(K)
    N, P = 24, 37
    w = rng.integers(-127, 128, (N, K)).astype(np.int8)
    w[0] = 127
    w[1] = -127
    w[2, ::2] = -127                                             # train/export.py's floor: |w| <= 127
    scale = rng.uniform(1e-4, 1e-2, N).astype(F32)
    bias = rng.normal(0, 1, N).astype(F32)
    x = edge_rows(rng, P, K)
    op = {"w": w, "scale": scale, "bias": bias, "relu": True, "cout": N, "int8": True}
    want = R.matmul_exact(x, op, "w8i")                          # [P, N]
    left = _graph(lambda g, a: X.matmul_exact(g, X.Quant(g, a), w, scale, bias, True, "left"), [("a", (K, P))])
    right = _graph(lambda g, a: X.matmul_exact(g, X.Quant(g, a), w, scale, bias, True, "right"), [("a", (K, P))])
    assert np.array_equal(bits(left.run(None, {"a": np.ascontiguousarray(x.T)})[0]), bits(want.T))
    assert np.array_equal(bits(right.run(None, {"a": np.ascontiguousarray(x.T)})[0]), bits(want))


def test_minus_128_weight_is_refused():
    """The exact graph's int32 bounds hold for |w| <= 127 (int8_chunks), all train/export.py writes."""
    w = np.full((8, 16), -128, np.int8)
    with pytest.raises(SystemExit, match="-128"):
        _graph(lambda g, a: X.matmul_exact(g, X.Quant(g, a), w, np.ones(8, F32), None, False, "left"),
               [("a", (16, 3))])


@pytest.mark.parametrize("kind,k,s,p", [("conv", (3, 3), (2, 2), (1, 1)), ("dw", (3, 3), (1, 1), (1, 1)),
                                         ("dw", (3, 3), (2, 2), (1, 1)), ("dw", (3, 3), (2, 1), (1, 1)),
                                         ("dw", (1, 5), (1, 1), (0, 2))])
def test_conv_exact_is_vocr_ref(kind, k, s, p):
    import vocr_ref as R
    rng = np.random.default_rng(sum(k) + sum(s))
    C, H, W = 12, 8, 40
    taps = k[0] * k[1]
    op = {"type": X.T_CONV if kind == "conv" else X.T_DW, "kh": k[0], "kw": k[1], "sh": s[0], "sw": s[1],
          "ph": p[0], "pw": p[1], "cout": C, "relu": True, "w": rng.normal(0, 0.5, (C, taps)).astype(F32),
          "bias": rng.normal(0, 0.3, C).astype(F32), "h_out": (H + 2 * p[0] - k[0]) // s[0] + 1}
    if kind == "conv":
        x = (rng.integers(0, 256, (H, W)).astype(F32) / F32(127.5) - F32(1)).astype(F32)
        feed = x[None, None]
    else:
        x = rng.normal(0, 1, (H, W, C)).astype(F32)
        feed = np.ascontiguousarray(x.transpose(2, 0, 1))[None]
    want = R.conv_exact(x, op)                                   # [Ho, Wo, C]
    sess = _graph(lambda g, a: X.conv_exact(g, a, op), [("a", feed.shape)], rank=4)
    got = sess.run(None, {"a": feed})[0][0].transpose(1, 2, 0)
    assert np.array_equal(bits(got), bits(want))


# ----------------------------------------------------------------------------- whole recognizers against C

@pytest.fixture(scope="module")
def blobs():
    """(plain, gated + two-level) random-init int8 blobs with a 210-class vocab, thresholds at the frames' median gate
    score and the 70th percentile escape score of the test crops, so every branch runs."""
    import torch
    from pyvocr import Vocr
    from torch import nn

    import export
    import head2level as H2
    import headgate as HG
    from model import TinyVNOCR
    torch.manual_seed(7)
    m = TinyVNOCR(len(VOCAB) + 1, width=0.25).eval()
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                mod.running_var.uniform_(0.5, 2.0)
                mod.running_mean.uniform_(-0.2, 0.2)
                mod.weight.uniform_(0.5, 1.5)
                mod.bias.uniform_(-0.5, 0.5)
            if isinstance(mod, nn.Conv2d):
                nn.init.normal_(mod.weight, 0, (2.0 / mod.weight[0].numel()) ** 0.5)
        m.head.bias.normal_(0, 1.0)
        m.head.bias[0] += 2.0
    plain, _ = export.build_blob(export.model_ops(m, 32), VOCAB, 32, True, 1024, b"")
    rng = np.random.default_rng(3)
    hd = H2.head_arrays(plain)
    K = hd["K"]
    gp = HG.quantise(rng.normal(0, 0.3, size=(16, K)), rng.normal(0, 0.5, 16), rng.normal(0, 0.5, 16), 0.0)
    l1b = 4
    w1 = rng.normal(0, 0.3, size=(16, K))
    q, sc = export.quantize_rows(w1)
    esc = dict(q=q, scale=sc.astype(F32), bias=rng.normal(0, 0.5, 16).astype(F32),
               wl1=rng.normal(0, 0.05, 16).astype(F32), wb0=rng.normal(0, 0.05, 16).astype(F32),
               w2=rng.normal(0, 0.5, 16).astype(F32), b2=F32(rng.normal(0, 0.5)), t=F32(0))
    x = np.concatenate([(lambda r: (r.run(g, head_in=True), r.last_head_in)[1])(Vocr(plain, mode="w8i"))
                        for g in crops()])
    gt = F32(np.median(HG.gate_scores(gp, x)))
    qx, sx = H2.quant16_exact(x)
    lg1 = H2.mm_exact(qx, sx, hd["q"][:8 * l1b], hd["scale"][:8 * l1b], hd["bias"][:8 * l1b])
    l1, b0 = H2.level1_inputs(lg1, 8 * l1b)
    esc["t"] = F32(np.quantile(H2.escape_score(qx, sx, l1, b0, esc), 0.7))
    both = H2.attach(HG.append_gate(plain, gp, gt), H2.write_trailer(plain, l1b, esc))
    gate_only = HG.append_gate(plain, gp, gt)                   # a head gate without a two-level head (VOCR5's _w8g)
    return plain, both, gate_only


def crops(n=14, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        h, w = int(rng.integers(18, 44)), int(rng.integers(30, 420))
        g = rng.integers(0, 255, size=(h, w)).astype(np.uint8)
        g[:, : w // 3] //= 3
        out.append(g)
    out += [np.full((32, 64), 255, np.uint8), np.zeros((3, 1), np.uint8), rng.integers(0, 256, (19, 2049), np.uint8)]
    return out


def outputs(sess, x, sections=None):
    feeds = {"grey": x[None, None]}
    if sections is not None:
        feeds["head_sections"] = np.array(sections)
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(names, feeds)))


@pytest.mark.parametrize("which", ["plain", "both", "gate"])
def test_exact_recognizer_is_c_w8i(blobs, which):
    from pyvocr import Vocr
    data = {"plain": blobs[0], "both": blobs[1], "gate": blobs[2]}[which]
    sess = session(X.recognizer(data, "exact"))
    plain_c = Vocr(data, mode="w8i")
    sec_c = None
    if which == "both":
        sec_c = Vocr(data, mode="w8i", head_gate="on", two_level=1)
    elif which == "gate":
        sec_c = Vocr(data, mode="w8i", head_gate="on")
    seen_gated = seen_escaped = 0
    for g in crops():
        for inv in (False, True):
            x, nw = V.preprocess(g, inv)
            assert np.array_equal(x, plain_c.preprocess(g, inv)[0])
            text, lg = plain_c.run(g, logits=True, invert=inv)
            c = plain_c.run_topk(g, 8, invert=inv)
            o = outputs(sess, x, None if which == "plain" else False)
            assert np.array_equal(bits(o["logits"]), bits(lg))
            assert np.array_equal(o["best"], c["best"])
            assert np.array_equal(bits(o["topk_logits"]), bits(c["topv"]))
            assert np.array_equal(o["topk_classes"], c["topc"])
            assert np.array_equal(bits(o["blank_logit"]), bits(c["blank"]))
            if sec_c is None:
                continue
            s = sec_c.run_topk(g, 8, invert=inv, gate_score=True, esc=True)
            o = outputs(sess, x, True)
            for k_onnx, k_c in (("best", "best"), ("topk_classes", "topc")):
                assert np.array_equal(o[k_onnx], s[k_c]), k_onnx
            for k_onnx, k_c in (("topk_logits", "topv"), ("blank_logit", "blank"), ("gate_score", "gate_score")):
                assert np.array_equal(bits(o[k_onnx]), bits(s[k_c])), k_onnx
            if which == "both":
                assert np.array_equal(np.isnan(o["escape_score"]), np.isnan(s["esc"]))
                ok = ~np.isnan(s["esc"])
                assert np.array_equal(bits(o["escape_score"][ok]), bits(s["esc"][ok]))
            else:
                assert "escape_score" not in o
            seen_gated += s["gated"]
            seen_escaped += s["escaped"]
    if which == "both":
        assert seen_gated and seen_escaped                        # the fixture exercises both sections
    if which == "gate":
        assert seen_gated


def test_reader_text_conf_and_cells_are_c(blobs, tmp_path):
    """Recognizer.read on the exact file: greedy text, conf_min / conf_mean and the character cells (eval/charbox.py,
    runtime/vocr_charbox.c's twin) from the C runtime's frames."""
    import charbox as CB
    from pyvocr import Vocr
    data = blobs[1]
    p = tmp_path / "rec.onnx"
    cal = [-36, 0, -12, 4, 38, -16, -10, 0, 8]
    p.write_bytes(X.recognizer(data, "exact", charbox_cal=cal).SerializeToString())
    rec = V.Recognizer(p)
    c = Vocr(data, mode="w8i", head_gate="on", two_level=1)
    groups = [CB.group(t) if len(t) == 1 else 0 for t in rec.vocab]
    for g in crops():
        ln = rec.read(g, sections=True)
        o = c.run_topk(g, 8)
        assert ln.greedy == o["text"]
        assert F32(ln.conf_min) == F32(o["conf_min"]) and F32(ln.conf_mean) == F32(o["conf_mean"])
        h, w = g.shape
        want = CB.positions(o["best"], w, h, c.resized_width(w, h), cal=cal, groups=groups)
        got = [(ch.cls, ch.f0, ch.f1, ch.x * 16, ch.x0 * 16, ch.x1 * 16) for ch in ln.chars]
        assert got == [(q["cls"], q["f0"], q["f1"], q["c"], q["x0"], q["x1"]) for q in want]


def test_fast_recognizer_is_close_to_w8f(blobs):
    from pyvocr import Vocr
    data = blobs[1]
    sess = session(X.recognizer(data, "fast"))
    c = Vocr(data, mode="w8f")
    worst = 0.0
    for g in crops():
        x, _ = V.preprocess(g)
        _, lg = c.run(g, logits=True)
        o = outputs(sess, x, False)
        worst = max(worst, float(np.abs(o["logits"] - lg).max()))
    assert worst < 1e-3


def test_exports_are_deterministic_and_described(blobs):
    a = X.recognizer(blobs[1], "exact", name="t").SerializeToString()
    b = X.recognizer(blobs[1], "exact", name="t").SerializeToString()
    assert a == b
    sess = session(X.recognizer(blobs[1], "fast", name="t"))
    meta = V._meta(sess)
    assert meta["vocr.kind"] == "recognizer" and meta["vocr.mode"] == "fast" and meta["vocr.vocab"][1:] == VOCAB
    assert meta["vocr.head_gate"] and meta["vocr.two_level"] and meta["vocr.level1_classes"] == 32
    assert [i.name for i in sess.get_inputs()] == ["grey", "head_sections"]
    assert "vocr.convs" not in meta and "vocr.convs" not in V._meta(session(X.recognizer(blobs[1], "exact")))


def test_ort_convs_form_is_marked_and_close(blobs):
    """recognizer(convs="ort"), astra's native-int8 form kept for comparison (eval/onnx_timing.py): marked in its
    metadata, within float rounding of W8I, and never written by release()."""
    from pyvocr import Vocr
    data = blobs[1]
    sess = session(X.recognizer(data, "exact", convs="ort"))
    assert V._meta(sess)["vocr.convs"] == "ort"
    c = Vocr(data, mode="w8i")
    for g in crops(6):
        x, _ = V.preprocess(g)
        _, lg = c.run(g, logits=True)
        assert np.abs(outputs(sess, x, False)["logits"] - lg).max() < 1e-2
    with pytest.raises(ValueError):
        X.recognizer(data, "exact", convs="numpy")
    import inspect
    assert "convs" not in inspect.signature(X.release).parameters


# ----------------------------------------------------------------------------- the spotter

TINY_T = {"name": "tinyt", "family": "T", "stride": 8, "c0": 4, "c1": 8, "c2": 16, "chans": [16, 16], "dil": [1, 2],
          "block": "sep", "head": "kroa_x2", "half_above": 480}


@pytest.mark.parametrize("mode", ["exact", "int8"])
def test_spotter_is_the_twin(mode, tmp_path):
    import textspot6_model as M6
    import textspot6_twin as T6
    rng = np.random.default_rng(11)
    for seed, tags in ((1, 4), (2, 2)):
        blob, _ = M6.random_blob(TINY_T, seed, tags, n_cal=2)
        m = T6.Model(blob)
        p = tmp_path / f"spot{seed}.onnx"
        p.write_bytes(X.spotter(blob, mode).SerializeToString())
        sp = V.Spotter(p)
        for h, w in ((1, 1), (9, 17), (119, 321), (272, 480), (544, 960), (150, 900)):
            g = rng.integers(0, 256, (h, w), dtype=np.uint8)
            g[h // 3: h // 3 + max(1, h // 6), w // 8: w - w // 8] = 255
            for scale in (None, 1, 2):
                x, f = T6.net_input(m, g, scale)
                assert np.array_equal(sp.head(x).astype(np.int64), T6.run_head(m, x))
                want, got = T6.spot(m, g, scale=scale), sp.spot(g, scale=scale)
                assert [tuple(vars(b).values()) for b in got.boxes] == [tuple(vars(b).values()) for b in want.boxes]
                assert [tuple(vars(b).values()) for b in got.blocks] == [tuple(vars(b).values()) for b in want.blocks]
                assert (got.pick, got.truncated, got.presence) == (want.pick, want.truncated, want.presence)


def test_spotter_decode_is_the_twin_on_drawn_maps():
    """The decode alone (components, tags and tconf, the 64-box cap, dark flags, blocks, order, pick) on hand-made
    maps with many boxes, against synth/textspot6_twin.spot_maps."""
    import textspot6_model as M6
    import textspot6_twin as T6
    rng = np.random.default_rng(5)
    blob, _ = M6.random_blob(TINY_T, 1, 4, n_cal=2)
    m = T6.Model(blob)

    class S(V.Spotter):
        def __init__(self):                                   # the decode without a session
            self.tags, self.conf_min, self.min_cells, self.stride = m.tags, m.conf_min, m.min_cells, m.stride

    sp = S()
    for trial in range(6):
        H, W = 272, 480
        maps = np.full((4 + m.tags, H // 4, W // 4), -50, np.int64)
        for _ in range(int(rng.integers(5, 90))):
            r, c0 = int(rng.integers(0, H // 4)), int(rng.integers(0, W // 4 - 3))
            c1 = min(W // 4, c0 + int(rng.integers(2, 30)))
            maps[0, r, c0:c1] = int(rng.integers(1, 127))
            maps[1, r, c0:c1] = int(rng.integers(-60, 60))
            maps[2:4, r, c0:c1] = rng.integers(0, 30, (2, c1 - c0))
            maps[4:, r, c0:c1] = rng.integers(-20, 40, (m.tags, 1))
        g = rng.integers(0, 256, (H, W), dtype=np.uint8)
        for f in (1, 2):
            want = T6.spot_maps(m, maps, f, g if f == 1 else np.repeat(np.repeat(g, 2, 0), 2, 1))
            got = sp.decode(maps, f, g if f == 1 else np.repeat(np.repeat(g, 2, 0), 2, 1))
            assert [tuple(vars(b).values()) for b in got.boxes] == [tuple(vars(b).values()) for b in want.boxes]
            assert [tuple(vars(b).values()) for b in got.blocks] == [tuple(vars(b).values()) for b in want.blocks]
            assert (got.pick, got.truncated, got.presence) == (want.pick, want.truncated, want.presence)


def test_line_dark_port():
    from common import line_dark
    rng = np.random.default_rng(9)
    for _ in range(200):
        h, w = int(rng.integers(2, 60)), int(rng.integers(2, 300))
        g = rng.integers(0, 256, (h, w), dtype=np.uint8)
        if rng.random() < 0.5:
            g[h // 4: 3 * h // 4, w // 5: 4 * w // 5] = int(rng.integers(0, 256))
        assert V.line_dark(g, (0, 0, w, h)) == line_dark(g, (0, 0, w, h))


# ----------------------------------------------------------------------------- the LM

def random_lm(rng, vocab, order=3):
    import vlm
    cps = sorted({ord(t) for t in vocab if len(t) == 1})
    Vn = len(cps) + vlm.FIRST
    orders, parents = [], list(range(Vn))
    for k in range(1, order + 1):
        d = {"cb_prob": np.sort(-rng.integers(0, 12000, 256)).astype(np.int16)[::-1].copy()}
        if k == 1:
            n = Vn
        else:
            toks, kids = [], [0]
            for _ in parents:
                t = np.sort(rng.choice(Vn - 1, size=int(rng.integers(0, 6)), replace=False) + 1) if Vn > 1 else []
                toks += [int(v) for v in t]
                kids.append(len(toks))
            d["tok"] = np.array(toks, np.uint16)
            orders[-1]["child"] = np.array(kids, np.uint32)
            n = len(toks)
            parents = list(range(n))
        d["prob"] = rng.integers(0, 256, n).astype(np.uint8)
        if k < order:
            d["bo"] = rng.integers(0, 256, n).astype(np.uint8)
            cb = -rng.integers(0, 3000, 256).astype(np.int16)
            cb[0] = 0
            d["cb_bo"] = cb
        orders.append(d)
    orders[-1].pop("child", None)
    if order > 1:
        orders[-2].setdefault("child", np.zeros(len(orders[-2]["prob"]) + 1, np.uint32))
    return vlm.write_blob(order, cps, orders)


def test_charlm_is_the_c_decoder(blobs):
    from pyvocr import Vocr, VocrLM
    rng = np.random.default_rng(13)
    data = blobs[1]
    c = Vocr(data, mode="w8i", head_gate="on", two_level=1)
    vocab = [c.class_text(k) for k in range(c.n_classes)]
    lm_blob = random_lm(rng, vocab)
    for flags in (1, 2, 3):
        py = V.CharLM(lm_blob, alpha=600, beta=2500, flags=flags)
        cl = VocrLM(c, lm_blob, beam=8, topk=8, alpha=600, beta=2500, gate=8000, flags=flags, bracket=8000)

        class R:
            pass
        r = R()
        r.vocab = vocab
        for g in crops(10, seed=flags):
            o = c.run_topk(g, 8)
            assert py.decode(r, o["topv"], o["topc"], o["blank"]) == cl.decode(o["topv"], o["topc"], o["blank"])


def test_rubi_lm_table():
    """RUBI_LM and RUBI_LM_RETUNED against Rubi's own tables (bee-san/Rubi_for_vita core/vocr_ocr.c lms[]: beta,
    tier, alpha, flags per LM; lm_retuned[]: VOCR6-N's row for the 2000k LM), copied here as (alpha, beta, flags); the
    four tiers' values are also v0.3.7-lm6.2's LM-settings.json (alpha, beta, flags 2 = EOS, 1 = BOS)."""
    want = {"vocr_lm_o3_500k.vlm": (500, 3000, 2), "vocr_lm6_o3_1000k.vlm": (600, 2500, 1),
            "vocr_lm_o3_2000k.vlm": (500, 3000, 2), "vocr_lm6_o3_4000k.vlm": (600, 2500, 1)}
    assert {k: (v["alpha"], v["beta"], v["flags"]) for k, v in V.RUBI_LM.items()} == want
    assert {k: (v["alpha"], v["beta"], v["flags"]) for k, v in V.RUBI_LM_RETUNED.items()} == {
        ("N", "vocr_lm_o3_2000k.vlm"): (400, 1500, V.FLAG_BOS | V.FLAG_EOS)}
    assert struct.calcsize("<7I") == 28                       # the VOLM header fields CharLM unpacks after the magic


def test_align_map_is_charbox_align_map():
    """vocr_onnx.align_map against eval/charbox.py align_map (runtime/vocr_charbox.c's twin) on random text pairs:
    substitutions, insertions, deletions and repeats over small alphabets."""
    import random

    import charbox as CB
    rng = random.Random(4)
    for _ in range(3000):
        alpha = "あいうえおかきく"[: rng.randint(2, 8)]
        src = [rng.choice(alpha) for _ in range(rng.randint(0, 30))]
        dst = list(src)
        for _ in range(rng.randint(0, 6)):
            op = rng.randint(0, 2)
            k = rng.randint(0, len(dst))
            if op == 0 and dst and k < len(dst):
                dst[k] = rng.choice(alpha)
            elif op == 1:
                dst.insert(k, rng.choice(alpha))
            elif dst and k < len(dst):
                del dst[k]
        assert V.align_map(src, dst) == CB.align_map(src, dst)


def test_place_gives_every_shown_character_a_cell():
    chars = [V.Char(t, i + 1, 2 * i, 2 * i, 10.0 * i + 5, 10.0 * i, 10.0 * i + 10, 3, 30)
             for i, t in enumerate("あいう")]
    same = V.place("あいう", chars)
    assert [(c.text, c.x0, c.x1) for c in same] == [(c.text, c.x0, c.x1) for c in chars]
    assert all(a is not b for a, b in zip(same, chars))          # copies: read_region shifts each Char once
    ins = V.place("あいXう", chars)
    assert [c.text for c in ins] == list("あいXう")
    assert (ins[2].f0, ins[2].x0, ins[2].x1, ins[2].y1) == (-1, 10.0, 20.0, 30)   # shares い's cell
    first = V.place("Xあいう", chars)
    assert (first[0].f0, first[0].x0, first[0].x1) == (-1, 0.0, 10.0)            # at the start: the next one's
    sub = V.place("あXう", chars)
    assert (sub[1].text, sub[1].f0, sub[1].x0) == ("X", 2, 10.0)                  # a substitution keeps its cell
    assert V.place("", chars) == [] and V.place("X", []) == [V.Char("X", -1, -1, -1, 0.0, 0.0, 0.0)]


def test_command_line_reads_an_image(blobs, tmp_path):
    """vocr_onnx.py's command line on a drawn frame with generated models: one JSON line, one cell per character of
    each line's text."""
    import json
    import subprocess

    from PIL import Image

    import textspot6_model as M6
    d = tmp_path / "models"
    d.mkdir()
    (d / "VOCR6.2_N_exact.onnx").write_bytes(X.recognizer(blobs[1], "exact").SerializeToString())
    blob, _ = M6.random_blob(TINY_T, 1, 4, n_cal=2)
    (d / "textspot8_exact.onnx").write_bytes(X.spotter(blob, "exact").SerializeToString())
    g = np.full((272, 480, 3), 30, np.uint8)
    g[200:224, 40:400] = 230
    Image.fromarray(g).save(tmp_path / "f.png")
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "vocr_onnx.py"), "--models", str(d), "--tier", "N",
                          str(tmp_path / "f.png")], capture_output=True, text=True, check=True).stdout.splitlines()
    assert len(out) == 1
    rec = json.loads(out[0])
    assert rec["image"].endswith("f.png")
    for ln in rec["lines"]:
        assert len(ln["chars"]) == len(ln["text"]) and len(ln["box"]) == 4


def test_parity_tally_counts_numpy_bools_and_the_verdict_needs_every_check():
    """eval/onnx_parity.py: a check given as numpy.bool_ is counted (audit-vocr-onnx F1), and the verdict fails when a
    check of R1 that every tier must have never reached the tally."""
    import onnx_parity as P
    rec = {"S": {"exact_conf": np.float32(1) == np.float32(1), "exact_text": True, "frames": 3}}
    t = P.tally([("k", rec)])
    assert t["S"]["exact_conf"] == {"n": 1, "fail": 0, "ids": []} and t["S"]["counts"]["frames"] == 3
    full = {k: {"n": 1, "fail": 0} for k in P.R1_ALWAYS}
    res = {"sets": {"x": {"checks": {"S": dict(full), "N": dict(full, sec_best={"n": 1, "fail": 0})}}}}
    v = P.verdict(res)
    assert not v["R1_exact_recognizer"] and v["R1_missing_checks"] == {"N": [k for k in P.R1_SECTIONS
                                                                              if k != "sec_best"]}
    res["sets"]["x"]["checks"]["N"].update({k: {"n": 1, "fail": 0} for k in P.R1_SECTIONS})
    assert P.verdict(res)["R1_exact_recognizer"]
    del res["sets"]["x"]["checks"]["S"]["exact_conf"]
    assert P.verdict(res)["R1_missing_checks"] == {"S": ["exact_conf"]}


def test_spotter_refuses_sides_over_2048_and_read_region_clips(blobs, tmp_path):
    """Spotter.spot takes 1 to 2048 px a side, as runtime/vocr_textspot6.c (audit-vocr-onnx n10); read_region reads a
    region that starts outside the frame as its part inside the frame (n13)."""
    import textspot6_model as M6
    blob, _ = M6.random_blob(TINY_T, 1, 4, n_cal=2)
    p = tmp_path / "spot.onnx"
    p.write_bytes(X.spotter(blob, "exact").SerializeToString())
    sp = V.Spotter(p)
    with pytest.raises(ValueError, match="2048"):
        sp.spot(np.zeros((10, 2049), np.uint8))
    with pytest.raises(ValueError):
        sp.spot(np.zeros((0, 10), np.uint8))
    sp.spot(np.zeros((2048, 10), np.uint8))
    r = tmp_path / "rec.onnx"
    r.write_bytes(X.recognizer(blobs[0], "exact").SerializeToString())
    rec = V.Recognizer(r)
    rng = np.random.default_rng(2)
    g = rng.integers(0, 60, (272, 480), dtype=np.uint8)
    g[10:30, 5:80] = 240
    a = V.read_region(g, sp, rec, region=(-10, -10, 100, 50))
    b = V.read_region(g, sp, rec, region=(0, 0, 90, 40))
    assert [(ln.box, ln.text) for ln in a] == [(ln.box, ln.text) for ln in b]
    assert V.read_region(g, sp, rec, region=(500, 0, 10, 10)) == []
