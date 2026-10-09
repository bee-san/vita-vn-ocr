"""Fast tests for the architecture-sweep pieces: QAT fake quantisation, height 24,
<unk> vocabs, checkpoint round trip and the cost model. CPU only, no data files."""
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train"))
sys.path.insert(0, str(ROOT / "synth"))

import cost  # noqa: E402
from model import TinyVNOCR, fake_quant_weight, from_checkpoint, set_weight_quant  # noqa: E402
from train import UNK, encoder  # noqa: E402

torch.set_num_threads(1)


@pytest.mark.parametrize("height", [24, 32])
@pytest.mark.parametrize("width", [0.25, 1.0])
def test_output_frames(height, width):
    m = TinyVNOCR(101, width=width, bottleneck=48).eval()
    with torch.no_grad():
        out = m(torch.zeros(2, 1, height, 80))
    assert out.shape == (10, 2, 101)


def test_v1_shape_unchanged():
    m = TinyVNOCR(5355)
    assert sum(p.numel() for p in m.parameters()) == 761_771     # v1's 0.762M


def test_fake_quant_is_int8_grid_with_straight_through_grad():
    w = torch.randn(16, 8, 3, 3, requires_grad=True)
    q = fake_quant_weight(w)
    scale = w.detach().abs().flatten(1).amax(1).view(-1, 1, 1, 1) / 127
    k = q.detach() / scale
    assert torch.allclose(k, k.round(), atol=1e-4) and k.abs().max() <= 127 + 1e-4
    q.sum().backward()
    assert torch.equal(w.grad, torch.ones_like(w))


def test_fake_quant_commutes_with_bn_folding():
    """Quantising the raw conv weight per output channel and then folding BN equals
    folding first and quantising the folded weight (what an int8 exporter does)."""
    torch.manual_seed(0)
    conv, bn = nn.Conv2d(8, 16, 3, bias=False), nn.BatchNorm2d(16)
    bn.weight.data.uniform_(-2, 2)                      # negative gammas flip signs
    bn.running_var.data.uniform_(0.1, 3)
    k = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).detach().view(-1, 1, 1, 1)
    a = fake_quant_weight(conv.weight.detach()) * k
    b = fake_quant_weight(conv.weight.detach() * k)
    assert torch.allclose(a, b, atol=1e-6)


def test_int8_set_matches_export():
    m = TinyVNOCR(50)
    set_weight_quant(m, 8)
    stem = m.features[0][0]
    dw = m.features[1].dw[0]
    assert stem.wbits == 0 and dw.wbits == 0
    assert m.features[1].pw[0].wbits == 8 and m.head.wbits == 8 and m.bottleneck.wbits == 8
    assert m.features[7][0].wbits == 8                      # 2x1 row collapse


def test_set_weight_quant_changes_and_restores_output():
    torch.manual_seed(0)
    m = TinyVNOCR(50, width=0.25).eval()
    x = torch.rand(1, 1, 32, 64) * 2 - 1
    with torch.no_grad():
        ref = m(x)
        set_weight_quant(m, 8)
        q = m(x)
        set_weight_quant(m, 0)
        back = m(x)
    assert not torch.equal(ref, q) and torch.equal(ref, back)


def test_unk_encoder():
    plain = encoder(["あ", "い"])
    assert plain("あい") == [1, 2] and plain("あう") == [1, -1]
    unk = encoder(["あ", "い", UNK])
    assert unk("あうい") == [1, 3, 2]


@pytest.mark.parametrize("size", [2500, 3500])
def test_reduced_vocabs(size):
    """data/vocab_N.txt: N entries, a code-point-sorted subset of data/vocab.txt plus the
    trailing <unk>, with every kana of the full vocab kept."""
    from common import DATA, load_vocab
    from vocab import is_kana
    full = load_vocab()
    v = load_vocab(DATA / f"vocab_{size}.txt")
    assert len(v) == size and v[-1] == UNK and UNK not in full
    assert v[:-1] == sorted(v[:-1]) and set(v[:-1]) <= set(full)
    assert {c for c in full if is_kana(c)} <= set(v)
    assert set("、。「」『』！？…ー") <= set(v)


def test_vocab_build_sizes_and_extra():
    from vocab import build
    full = list("あいうアイウ一二三四五")
    freq = {"五": 5.0, "三": 3.0, "一": 1.0}
    assert build(9, freq, full) == sorted("あいうアイウ五三") + [UNK]
    assert build(0, freq, full) == sorted(full) + [UNK]
    v = build(9, freq, full, extra=["龍"])                   # extras are always kept
    assert "龍" in v and "五" in v and "三" not in v and len(v) == 9
    with pytest.raises(ValueError):
        build(5, freq, full)                                 # kana alone need 6 entries


def test_vocab_names_leave_out_test_games(tmp_path):
    """The name ranking never sees a character of one of eval's test games: a name is
    dropped if any of its VNs is reserved (test_games.json 'all') or is_test_vn()."""
    import json

    from common import is_test_vn
    from vocab import load_names, reserved_vns
    reserved = reserved_vns()
    test_vn = next(iter(sorted(reserved)))
    hashed = next(f"v{i}" for i in range(1, 10_000) if is_test_vn(f"v{i}"))
    free = next(f"v{i}" for i in range(1, 10_000) if f"v{i}" not in reserved and not is_test_vn(f"v{i}"))
    rows = [{"id": "c1", "original": "葵", "vns": [free]},
            {"id": "c2", "original": "杏", "vns": [free, test_vn]},
            {"id": "c3", "original": "椎", "vns": [hashed]},
            {"id": "c4", "original": None, "vns": [free]}]
    p = tmp_path / "chars.jsonl"
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    assert load_names(p) == (["葵"], 2)


def test_checkpoint_round_trip(tmp_path):
    torch.manual_seed(0)
    m = TinyVNOCR(4, width=0.25, bottleneck=48).to(memory_format=torch.channels_last)
    ck = {"model": {k: v.contiguous() for k, v in m.state_dict().items()},
          "vocab": ["あ", "い", UNK], "width": 0.25, "bottleneck": 48, "height": 24, "unk": UNK}
    torch.save(ck, tmp_path / "ck.pt")
    m2, meta = from_checkpoint(tmp_path / "ck.pt")
    assert meta["height"] == 24 and meta["unk"] == UNK and meta["bottleneck"] == 48
    x = torch.rand(1, 1, 24, 40)
    with torch.no_grad():
        assert torch.allclose(m.eval()(x), m2(x), atol=1e-6)


def test_cost_model_consistent():
    m = TinyVNOCR(5355)
    r = cost.report(m, 5355, 32)
    assert r["params"] == 761_771
    # BN folding: 4 BN tensors per channel become 1 bias, so folded < trained params
    assert r["params_folded"] < r["params"]
    assert r["weights_fp32"] == 4 * r["params_folded"]
    assert r["weights_int8"] < r["weights_fp32"] / 3
    assert r["arena_256"] < r["arena_640"] < r["arena_1024"]
    assert r["macs_1024"] == 4 * r["macs_256"]
    # height 24 uses the same parameters and needs less arena at the same width
    r24 = cost.report(m, 5355, 24)
    assert r24["params"] == r["params"] and r24["arena_1024"] < r["arena_1024"]


def test_exact_batching_equals_one_line_inference():
    """sweep_eval.predict(exact=True) batches only equal padded widths, so its text is
    what a one-line-at-a-time evaluator (eval/run.py) gets."""
    import numpy as np

    import sweep_eval
    torch.manual_seed(0)
    m = TinyVNOCR(30, width=0.25, bottleneck=48).eval()
    for mod in m.modules():                      # non-trivial BN so the output varies
        if isinstance(mod, nn.BatchNorm2d):
            mod.running_var.uniform_(0.5, 2.0)
    vocab = [chr(0x3042 + i) for i in range(29)]
    rng = np.random.default_rng(0)
    crops = [rng.integers(0, 256, (32, int(w)), dtype=np.uint8) for w in (40, 41, 48, 97, 100, 160, 41)]
    one = [sweep_eval.predict(m, [c], 32, vocab)[0] for c in crops]
    assert sweep_eval.predict(m, crops, 32, vocab, exact=True) == one


def test_paired_bootstrap_delta():
    import sweep_table
    a = {("s", str(i)): (2, 10) for i in range(50)}         # CER 20%
    b = {("s", str(i)): (1, 10) for i in range(50)}         # CER 10%
    d, lo, hi = sweep_table.paired_delta(a, b, n_boot=200)
    assert abs(d - 0.10) < 1e-12 and lo <= d <= hi
    assert sweep_table.paired_delta(a, {}) is None


def test_line_errors_use_the_pooled_cers_scorer(capsys):
    """The paired Δ must come from the same per-line errors as the pooled CER: eval's
    metrics.norm folds ・・ runs like an ellipsis, sweep_eval.score_norm does not (a real
    psp-test line). Stored per-line counts win; older JSONs are re-scored with eval's
    scorer on eval's sets; lines that don't add up to the set's errors are reported."""
    import sweep_table
    line = {"id": "x", "ref": "「……なんだよ。", "pred": "「・・～と・なんだよ。"}
    res = {"run": "r", "results": {"int8/psp-test-inv": {"errors": 3, "chars": 8,
                                                         "lines_detail": [dict(line)]}}}
    assert sweep_table.line_errors(res, "int8", ("psp-test-inv",)) == {("psp-test-inv", "x"): (3, 8)}
    assert capsys.readouterr().err == ""
    stored = {"run": "r", "results": {"int8/psp-test-inv": {
        "errors": 4, "chars": 8, "lines_detail": [{**line, "errors": 4, "n": 8}]}}}
    assert sweep_table.line_errors(stored, "int8", ("psp-test-inv",))[("psp-test-inv", "x")] == (4, 8)
    res["results"]["int8/psp-test-inv"]["errors"] = 5                # inconsistent JSON
    sweep_table.line_errors(res, "int8", ("psp-test-inv",))
    assert "warning" in capsys.readouterr().err


def test_w8i_sizes_for_reductions_over_512():
    """Width 2.0's row collapse has K = 640 > 512: the runtime's W8I sums it in int32
    chunks, so cost.py reports a W8I arena for it (no K limit)."""
    E = cost.runtime_exporter()
    if E is None:
        pytest.skip("train/export.py not importable")
    r = cost.runtime_sizes(E, TinyVNOCR(21, width=2.0), [chr(0x4E00 + i) for i in range(20)], 32)
    assert r["rt_max_k"] == 640
    assert all(r[f"rt_arena_w8i_{w}"] > r[f"rt_arena_w8f_{w}"] > 0 for w in cost.WIDTHS)


def test_c_runtime_w8i_with_reduction_over_512():
    """The C runtime runs W8I for a width-2.0 model (K = 640): logits within the parity
    tolerance of PyTorch with the int8 weights (skips if libvocr.so cannot be built)."""
    import shutil

    import numpy as np

    import export
    from model import set_weight_quant
    torch.manual_seed(0)
    vocab = [chr(0x3042 + i) for i in range(20)]
    m = TinyVNOCR(len(vocab) + 1, width=2.0).eval()
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.running_var.uniform_(0.5, 2.0)
    blob, _ = export.build_blob(export.model_ops(m, 32), vocab, 32, True, 1024, b"")
    if not (shutil.which("cc") or shutil.which("gcc")):
        pytest.skip("no C compiler")
    sys.path.insert(0, str(ROOT / "runtime"))
    from pyvocr import Vocr
    try:
        r = Vocr(blob, mode="w8i")
    except (OSError, RuntimeError, subprocess.CalledProcessError) as e:
        pytest.skip(f"libvocr.so unavailable: {e}")
    g = np.random.default_rng(0).integers(0, 256, (32, 96), dtype=np.uint8)
    _, logits = r.run(g, logits=True)
    set_weight_quant(m, 8)
    with torch.no_grad():
        ref = m(torch.from_numpy(g.astype(np.float32) / 127.5 - 1.0)[None, None])[:, 0].numpy()
    assert logits.shape == ref.shape
    assert np.abs(logits - ref).max() <= 5e-3 * max(1.0, np.abs(ref).max())


def test_predict_is_exact_by_default():
    """Every sweep_eval set is predicted one padded width per batch, so no crop is padded
    beyond what a one-line forward pass pads (in a mixed-width batch the replicate
    padding changes the frames near a shorter crop's right edge)."""
    import collections

    import numpy as np

    import sweep_eval
    m = TinyVNOCR(30, width=0.25, bottleneck=48).eval()
    fed = collections.Counter()

    def recording_model(x):
        fed[x.shape[-1]] += x.shape[0]
        return m(x)

    vocab = [chr(0x3042 + i) for i in range(29)]
    widths = np.random.default_rng(3).integers(16, 400, 40)
    crops = [np.zeros((32, int(w)), np.uint8) for w in widths]
    sweep_eval.predict(recording_model, crops, 32, vocab)
    assert fed == collections.Counter(int(w + 7) // 8 * 8 for w in widths)
    fed.clear()
    sweep_eval.predict(recording_model, crops, 32, vocab, exact=False)   # the old packing
    assert fed != collections.Counter(int(w + 7) // 8 * 8 for w in widths)


def test_unk_scoring():
    import sweep_eval
    r = sweep_eval.score(["ab〓d", "abcd"], ["abxd", "abcd"], unk_vocab=set("abcd"))
    assert r["cer"] == 1 / 8 and r["cer_unk_ok"] == 0 and r["line_acc"] == 0.5


def test_depth_and_ctx_knobs():
    v1 = TinyVNOCR(5355).state_dict()                         # the defaults are v1's layout
    assert v1["features.7.0.weight"].shape == (192, 160, 2, 1)   # 2x1 row collapse
    assert v1["features.9.dw.0.weight"].shape == (192, 1, 1, 5)  # second Ctx block
    assert "features.10.dw.0.weight" not in v1
    big = TinyVNOCR(5355, depth=2, ctx=4, ctx_k=7)
    assert big.features[-1].dw[0].weight.shape == (192, 1, 1, 7)
    assert sum(isinstance(m, type(big.features[2])) for m in big.features) == 9   # 3 + 3*2 DS
    with torch.no_grad():
        assert big.eval()(torch.zeros(1, 1, 32, 64)).shape == (8, 1, 5355)
    ck = {"model": big.state_dict(), "vocab": ["x"] * 5354, "depth": 2, "ctx": 4, "ctx_k": 7}
    m, meta = from_checkpoint(ck)
    assert (meta["depth"], meta["ctx"], meta["ctx_k"]) == (2, 4, 7)


def test_a9_estimate_scales_with_macs():
    r = cost.report(TinyVNOCR(5355), 5355, 32)
    slow, fast = r["a9_ms_1024"]
    assert fast < slow and abs(slow - 1000 * r["macs_1024"] / 0.24e9) < 1e-6


def test_export_and_c_runtime_run_a_deeper_student(tmp_path):
    """export.load_checkpoint rebuilds a depth/ctx/ctx_k checkpoint, and the C runtime's
    f32 logits match PyTorch for it (skips if libvocr.so cannot be built here)."""
    import shutil

    import numpy as np

    import export
    torch.manual_seed(0)
    vocab = [chr(0x3042 + i) for i in range(20)]
    m = TinyVNOCR(len(vocab) + 1, width=0.25, bottleneck=48, depth=2, ctx=3, ctx_k=7).eval()
    for mod in m.modules():
        if isinstance(mod, nn.BatchNorm2d):
            mod.running_var.uniform_(0.5, 2.0)
    path = tmp_path / "deep.pt"
    torch.save({"model": m.state_dict(), "vocab": vocab, "width": 0.25, "bottleneck": 48,
                "depth": 2, "ctx": 3, "ctx_k": 7, "height": 24}, path)
    model, v, height = export.load_checkpoint(path)
    assert height == 24 and v == vocab
    blob, _ = export.build_blob(export.model_ops(model, height), vocab, height)
    if not (shutil.which("cc") or shutil.which("gcc")):
        pytest.skip("no C compiler")
    sys.path.insert(0, str(ROOT / "runtime"))
    from pyvocr import Vocr
    try:
        r = Vocr(blob)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as e:
        pytest.skip(f"libvocr.so unavailable: {e}")
    g = np.random.default_rng(0).integers(0, 256, (24, 96), dtype=np.uint8)
    _, logits = r.run(g, logits=True)
    with torch.no_grad():
        ref = m(torch.from_numpy(g.astype(np.float32) / 127.5 - 1.0)[None, None])[:, 0].numpy()
    assert logits.shape == ref.shape
    assert np.abs(logits - ref).max() <= 1e-3 * max(1.0, np.abs(ref).max())
