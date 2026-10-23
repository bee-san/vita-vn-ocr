"""The blank-frame head gate (docs/headgate.md, issue #117 item 1) on generated data only: its blob section
(train/headgate.py) against the C loader, the C runtime against the bit-exact twin (gate scores, skipped frames,
top-K, text, confidence) on a random-init model at several tilings, head_gate off on a gated blob equal to the
blob without it, and the arena plan's mirror (train/export.plan_arena). No development, test or private data.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "train", ROOT / "runtime", ROOT / "runtime" / "tests", ROOT / "eval"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import headgate as HG  # noqa: E402

VOCAB = list("あいうえおかきくけこさしすせそたちつてと「」。、ー")


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    """A random-init TinyVNOCR as a W8I blob (bytes)."""
    import torch
    from torch import nn

    import export
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
        # blank-heavy, as a trained model's frames are: the blank's bias set so that it wins about 60% of frames
        lg = m(torch.rand(8, 1, 32, 256, generator=torch.Generator().manual_seed(8)) * 2 - 1)   # T x N x C
        d = (lg[:, :, 1:].max(2).values - lg[:, :, 0]).flatten()
        m.head.bias[0] += float(torch.quantile(d, 0.6))
    blob, _ = export.build_blob(export.model_ops(m, 32), VOCAB, 32, True, 1024, b"")
    return blob


def crops(n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        w = int(rng.integers(20, 420))
        g = rng.integers(0, 256, (int(rng.integers(12, 48)), w), dtype=np.uint8)
        g[:, rng.integers(0, w, size=w // 3)] //= 3     # structure: dark columns
        out.append(g)
    return out


def head_inputs(blob, imgs):
    from pyvocr import Vocr
    rt = Vocr(blob, mode="w8i", max_width=1024)
    xs = []
    for g in imgs:
        rt.run(g, head_in=True)
        xs.append(rt.last_head_in.copy())
    return xs


@pytest.fixture(scope="module")
def gated(base):
    """base + a gate fit (a few steps) to the model's own blank frames, threshold at the median score."""
    imgs = crops(24, 1)
    from pyvocr import Vocr
    rt = Vocr(base, mode="w8i", max_width=1024)
    xs, ys = [], []
    for g in imgs:
        rt.run(g, head_in=True)
        xs.append(rt.last_head_in.copy())
        ys.append(rt.last_best == 0)
    x, y = np.concatenate(xs), np.concatenate(ys)
    assert 0.2 < y.mean() < 0.95
    old = dict(HG.FIT)
    HG.FIT.update(epochs=2, batch=512)
    try:
        g = HG.fit_float(x, y)
    finally:
        HG.FIT.clear()
        HG.FIT.update(old)
    params = HG.quantise(g["w1"], g["b1"], g["w2"], g["b2"])
    t = np.float32(np.median(HG.gate_scores(params, x)))
    return HG.append_gate(base, params, t), params, t


def test_section_round_trip(base, gated):
    blob, params, t = gated
    g = HG.read_gate(blob)
    assert HG.read_gate(base) is None
    assert HG.base_of(blob) == base and len(blob) == len(base) + g["bytes"] and len(base) % 16 == 0
    assert g["k"] == HG.head_op(base)["k"] == 96 and g["hidden"] == 16 and g["t"] == t
    assert np.array_equal(g["q"], params["q"]) and np.array_equal(g["w2"], params["w2"])
    assert g["bytes"] == 64 + 16 * 96 + 3 * 64            # 1,792 B for a 96 -> 16 -> 1 gate
    import export
    assert export.read_blob(blob)["gate"]["hidden"] == 16 and export.read_blob(base)["gate"] is None
    bad = bytearray(blob)
    bad[len(base) + 200] ^= 1
    with pytest.raises(ValueError):
        HG.read_gate(bytes(bad))
    with pytest.raises(ValueError):                       # the gate's input must be the head's K
        HG.gate_section(base, dict(params, q=params["q"][:, :64]), t)


def test_c_loader(base, gated):
    import ctypes as C

    from pyvocr import Vocr, load_lib
    blob = gated[0]
    lib = load_lib()
    buf = np.frombuffer(blob, np.uint8).copy()
    assert lib.vocr_check_crc(buf.ctypes.data, len(buf)) == 0
    buf[len(base) + 300] ^= 1
    assert lib.vocr_check_crc(buf.ctypes.data, len(buf)) != 0
    assert Vocr(blob, mode="w8i").has_gate and not Vocr(base, mode="w8i").has_gate
    with pytest.raises(ValueError):                       # ON needs a blob with a gate
        Vocr(base, mode="w8i", head_gate="on")
    with pytest.raises(ValueError):                       # and W8I
        Vocr(blob, mode="w8f", head_gate="on")
    assert C.sizeof(C.c_uint) == 4


def test_off_is_the_base(base, gated):
    from pyvocr import Vocr
    a = Vocr(base, mode="w8i", max_width=1024)
    b = Vocr(gated[0], mode="w8i", max_width=1024)
    assert a.arena_bytes == b.arena_bytes
    for g in crops(30, 2):
        x, y = a.run_topk(g, 8), b.run_topk(g, 8)
        assert x["text"] == y["text"] and y["gated"] == 0
        for k in ("topv", "blank"):
            assert np.array_equal(x[k].view(np.uint32), y[k].view(np.uint32))
        assert np.array_equal(x["topc"], y["topc"]) and x["conf_min"] == y["conf_min"]


def test_scores_c_equal_twin(gated):
    from pyvocr import Vocr
    blob, params, _ = gated
    rt = Vocr(blob, mode="w8i", max_width=1024, head_gate="score")
    n = 0
    for g in crops(40, 3):
        o = rt.run_topk(g, 4, head_in=True, gate_score=True)
        tw = HG.gate_scores(params, o["head_in"])
        assert np.array_equal(o["gate_score"].view(np.uint32), tw.view(np.uint32))
        assert o["gated"] == 0
        n += len(tw)
    assert n > 1000


@pytest.mark.parametrize("tile,head", [(0, 0), (1, 1), (2, 3), (3, 16), (7, 5), (128, 128)])
def test_on_equals_simulation(base, gated, tile, head):
    from pyvocr import Vocr
    blob, params, t = gated
    a = Vocr(base, mode="w8i", max_width=1024)
    on = Vocr(blob, mode="w8i", max_width=1024, head_gate="on", tile=tile, head=head)
    vocab = [a.class_text(c) for c in range(a.n_classes)]
    skipped = frames = 0
    for i, g in enumerate(crops(25, 4)):
        k = (1, 3, 8, 16)[i % 4]
        x = a.run_topk(g, k, head_in=True)
        skip = HG.gate_scores(params, x["head_in"]) > t
        sim = HG.simulate(x, skip, vocab)
        y = on.run_topk(g, k, gate_score=True)
        assert y["text"] == sim["text"] and y["gated"] == sim["gated"] == int(skip.sum())
        assert np.array_equal(y["best"], sim["best"]) and np.array_equal(y["topc"], sim["topc"])
        for key in ("topv", "blank"):
            assert np.array_equal(y[key].view(np.uint32), sim[key].view(np.uint32))
        assert np.float32(y["conf_min"]) == np.float32(sim["conf_min"])
        assert np.float32(y["conf_mean"]) == np.float32(sim["conf_mean"]) and y["chars"] == sim["chars"]
        assert on.run(g) == sim["text"] and np.array_equal(on.last_best, sim["best"])     # without top-K
        txt, _ = on.run(g, logits=True)                    # the logits asked for: nothing skipped
        assert txt == x["text"] and on.last_gated == 0
        u = on.run_topk(g, k, ungated=True)                # vocr_extra.ungated: today's read exactly
        assert u["text"] == x["text"] and u["gated"] == 0 and np.array_equal(u["topc"], x["topc"])
        assert np.array_equal(u["topv"].view(np.uint32), x["topv"].view(np.uint32))
        skipped += int(skip.sum())
        frames += len(skip)
    assert 0.2 < skipped / frames < 0.8


def test_arena_mirror(gated):
    from pyvocr import Vocr

    import export
    blob = gated[0]
    b = export.read_blob(blob)
    for tile, head, mw in ((0, 0, 0), (1, 1, 0), (2, 3, 256), (5, 7, 640), (16, 16, 0), (128, 128, 1024)):
        for hg in (0, 1, 2):
            rt = Vocr(blob, mode="w8i", tile=tile, head=head, max_width=mw, head_gate=hg)
            assert rt.arena_bytes == export.plan_arena(b, mw, tile, head, export.MODE_W8I, hg), (tile, head, mw, hg)
    with pytest.raises(ValueError):
        export.plan_arena(b, 0, 0, 0, export.MODE_W8F, 1)


def test_twin_against_float(gated):
    """The twin's int8 layer is vocr.c's: an all-zero frame scores b2 + w2 . relu(b1) exactly, and any frame
    scores what the float gate on the dequantised weights gives, up to the int16 grid's rounding."""
    params = gated[1]
    z = np.zeros((1, 96), np.float32)
    want = np.float32(params["b2"])
    for j in range(16):
        want = np.float32(want + np.float32(params["w2"][j] * np.float32(max(params["bias"][j], 0))))
    assert HG.gate_scores(params, z)[0] == want
    x = np.random.default_rng(5).normal(size=(200, 96)).astype(np.float32)
    w = params["q"].astype(np.float64) * params["scale"].astype(np.float64)[:, None]
    ref = np.maximum(x.astype(np.float64) @ w.T + params["bias"], 0) @ params["w2"].astype(np.float64) + params["b2"]
    assert np.abs(HG.gate_scores(params, x) - ref).max() < 1e-3 * (1 + np.abs(ref).max())


def test_protocol_is_frozen():
    p = json.loads((ROOT / "eval" / "headgate_protocol.json").read_text(encoding="utf-8"))
    assert p["kind"] == "headgate-protocol" and p["issue"] == 117 and p["frozen"].startswith("2026-10-23T")
    assert set(p["models"]) >= {"S", "N", "L"} and all(len(p["models"][t]["sha256"]) == 64 for t in "SNL")
    for f, h in p["training"]["files"].items():
        assert len(h) == 64 and f.endswith(".npz")
    assert p["development"]["rule"]["per_tier"] and isinstance(p["amendments"], list)
    assert struct.calcsize("<4sHHII2HIIffIIII") == 52                # the section header's fields up to its CRC


def _signed(x, nd=3):
    """+0.013 / −0.002 (U+2212), as docs/headgate.md writes a difference."""
    s = f"{x:+.{nd}f}"
    return s.replace("-", "\u2212")


DEV = sorted((ROOT / "eval").glob("headgate_dev_*.json"))
A9 = sorted((ROOT / "eval").glob("headgate_a9_*.json"))
FIT = sorted((ROOT / "eval").glob("headgate_fit_*.json"))


@pytest.mark.skipif(not (DEV and A9 and FIT), reason="the development records are not committed yet")
def test_doc_numbers_are_the_records():
    """docs/headgate.md's development, time and training tables are the committed records' numbers."""
    doc = (ROOT / "docs" / "headgate.md").read_text(encoding="utf-8")
    dev = json.loads(DEV[-1].read_text(encoding="utf-8"))
    names = {"greedy": "greedy", "rubi": "rubi", "rubi_cascade": "rubi + cascade"}
    for t, tr in dev["tiers"].items():
        assert tr["verdict"] == "adopted"
        for d, label in names.items():
            e = tr["decodes"][d]
            c = e["compare"]["score"]
            cell = f"| {t} | {label} | {e['score_today']:.4f} → {e['score_gated']:.4f} | " \
                   f"{_signed(c['delta_points'])} [{_signed(c['ci95'][0])}, {_signed(c['ci95'][1])}] |"
            assert cell in doc, cell
            assert e["rule"] == {"not_resolved_worse": True, "upper_within_margin": True,
                                 "saving_at_least_5pct": True}
        sk = tr["skip"]["wd_finder_lines"]["skip_share"]
        assert f"{100 * sk:.1f}%" in doc
    a9 = json.loads(A9[-1].read_text(encoding="utf-8"))
    for t, tr in a9["tiers"].items():
        s = tr["runs"]["pld0"]["saving"]["dev_wd"]
        assert f"−{s['nominal_box_ms']:.1f} ms, −{100 * s['nominal_share']:.1f}%" in doc, (t, s)
        assert f"−{s['nominal_box_ms_444']:.1f} ms" in doc
        assert " / ".join(f"−{s['real_box_ms'][p]:.1f}" for p in ("vita", "psp", "adrenaline")) in doc
    fit = json.loads(FIT[-1].read_text(encoding="utf-8"))
    for t, rec in fit["tiers"].items():
        g = rec["gates"]["0.001"]
        assert f"| {t} | {rec['calibration']['auc_int8']:.5f} | {g['t']:.4f} | {100 * g['cal_skip_share']:.1f}% |" \
            in doc, t
        assert rec["c_equals_twin"] == {"lines": 3108, "scores_bitwise": True, "score_mode_best_unchanged": True,
                                        "on_mode_skipped_counts": True, "on_mode_skipped_frames_blank": True}
