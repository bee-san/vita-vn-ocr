"""stride16 (#118): the coarser CTC frame step end to end: the model's stage strides, checkpoints, export, the
runtime's default tile and op stride 3, train2's geometry, the frame-budget tool and the selection rule."""
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in ("train", "eval", "runtime", "synth"):
    if str(ROOT / p) not in sys.path:
        sys.path.insert(0, str(ROOT / p))

import export as E  # noqa: E402
from model import TinyVNOCR, from_checkpoint, stage_strides  # noqa: E402

VOCAB = [chr(0x3042 + i) for i in range(30)]


def _parse(blob):
    """train/export.py read_blob of an in-memory blob."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".vocr", delete=False) as f:
        f.write(blob)
    try:
        return E.read_blob(f.name)
    finally:
        Path(f.name).unlink()


def _model(stride, seed=0, width=0.25):
    torch.manual_seed(seed)
    m = TinyVNOCR(len(VOCAB) + 1, width=width, bottleneck=48, stride=stride)
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_var.uniform_(0.5, 2.0)
            mod.running_mean.uniform_(-0.2, 0.2)
    return m.eval()


def test_stage_strides():
    assert stage_strides(8) == (2, (2, 1))
    assert stage_strides(12) == ((2, 3), (2, 1))
    assert stage_strides(16) == (2, (2, 2))
    for bad in (4, 10, 24, 32):
        with pytest.raises(ValueError):
            stage_strides(bad)


def test_frames_per_stride_and_same_parameters():
    """96 columns give 12 / 8 / 6 frames; every stride has the stride-8 model's state_dict, and stride 8 is the
    default shape with the same initialisation."""
    x = torch.zeros(1, 1, 32, 96)
    shapes = {}
    for s in (8, 12, 16):
        m = _model(s)
        with torch.no_grad():
            assert m(x).shape == (96 // s, 1, len(VOCAB) + 1)
        assert m.stride == s and m.arch["stride"] == s
        shapes[s] = {k: tuple(v.shape) for k, v in m.state_dict().items()}
    assert shapes[8] == shapes[12] == shapes[16]
    torch.manual_seed(3)
    a = TinyVNOCR(len(VOCAB) + 1, width=0.25).state_dict()
    torch.manual_seed(3)
    b = TinyVNOCR(len(VOCAB) + 1, width=0.25, stride=8).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_checkpoints_record_the_stride(tmp_path):
    m = _model(16)
    ck = {"model": m.state_dict(), "vocab": VOCAB, "width": 0.25, "bottleneck": 48, "height": 32, "stride": 16}
    m2, meta = from_checkpoint(ck)
    assert meta["stride"] == 16 and m2.stride == 16
    torch.save(ck, tmp_path / "ck.pt")
    m3, vocab, height = E.load_checkpoint(tmp_path / "ck.pt")
    assert m3.stride == 16 and vocab == VOCAB and height == 32
    ck.pop("stride")                                   # every checkpoint before --stride: stride 8
    assert from_checkpoint(ck)[1]["stride"] == 8


def test_export_width_clamp_follows_the_stride():
    for s, mw in ((8, 1024), (12, 1020), (16, 1024)):
        blob, meta = E.build_blob(E.model_ops(_model(s), 32), VOCAB, 32, int8=True)
        b = _parse(blob)
        assert meta["stride"] == s and (b["max_width"], b["stride"]) == (mw, s) and E.default_max_width(s) == mw
    with pytest.raises(ValueError):
        E.build_blob(E.model_ops(_model(12), 32), VOCAB, 32, int8=True, max_width=1024)


def test_default_tile_mirrors_the_runtime(tmp_path):
    assert [E.default_tile(s) for s in (4, 8, 12, 16, 32, 64, 128)] == [4, 4, 3, 2, 1, 1, 1]
    for s in (8, 12, 16):
        p = tmp_path / f"s{s}.vocr"
        p.write_bytes(E.build_blob(E.model_ops(_model(s), 32), VOCAB, 32, int8=True)[0])
        b = E.read_blob(p)
        assert E.plan_arena(b, 1024, 0, 0, E.MODE_W8I) == E.plan_arena(b, 1024, E.default_tile(s), 16, E.MODE_W8I)


def _vocr(blob, **kw):
    if not (shutil.which("cc") or shutil.which("gcc")):
        pytest.skip("no C compiler")
    from pyvocr import Vocr
    try:
        return Vocr(blob, **kw)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as e:
        pytest.skip(f"libvocr.so unavailable: {e}")


@pytest.mark.parametrize("stride", [12, 16])
def test_c_runtime_runs_coarser_strides(stride):
    """The C runtime loads a stride-12 (op stride 3) and a stride-16 blob, emits ceil(q / stride) frames, matches
    PyTorch in f32, gives the same logits at every tile, and its default arena is plan_arena's."""
    m = _model(stride, seed=1, width=0.5)
    blob, _ = E.build_blob(E.model_ops(m, 32), VOCAB, 32, int8=False)
    r = _vocr(blob, mode="f32")
    g = np.random.default_rng(stride).integers(0, 256, (40, 700), dtype=np.uint8)
    q = r.resized_width(700, 40)
    _, logits = r.run(g, logits=True)
    assert logits.shape == (math.ceil(q / stride), len(VOCAB) + 1)
    grey, nw = r.preprocess(g)
    x = grey.astype(np.float32) / 127.5 - 1.0
    with torch.no_grad():
        ref = m(torch.from_numpy(x)[None, None])[:, 0].numpy()
    assert ref.shape == logits.shape
    assert np.abs(logits - ref).max() <= 1e-3 * max(1.0, np.abs(ref).max())
    for tile in (1, 3, 7):
        assert np.array_equal(_vocr(blob, mode="f32", tile=tile).run(g, logits=True)[1], logits)
    assert r.arena_bytes == E.plan_arena(_parse(blob), 0, 0, 0, E.MODE_F32)


def test_stride8_runtime_default_is_unchanged():
    """A stride-8 blob's default tile is still 4 frames: the arena equals the explicit tile-4 plan."""
    blob, _ = E.build_blob(E.model_ops(_model(8), 32), VOCAB, 32, int8=True)
    r = _vocr(blob, mode="w8i")
    assert r.arena_bytes == E.plan_arena(_parse(blob), 0, 4, 16, E.MODE_W8I)


def test_ctc_floor_at_the_stride():
    from train import ctc_ok
    ids = [1, 2, 2, 3]                                  # 4 characters + 1 repeat = 5 frames
    assert ctc_ok(ids, 40) and ctc_ok(ids, 40, 8)       # 5 frames at stride 8
    assert not ctc_ok(ids, 40, 16) and ctc_ok(ids, 80, 16)
    assert not ctc_ok(ids, 48, 12) and ctc_ok(ids, 49, 12)


def test_train2_geometry_and_batches():
    import train2
    try:
        assert train2.set_geometry(16) == (16, 1024)
        assert train2.set_geometry(12) == (12, 1020)
        train2.set_geometry(16)
        rng = np.random.default_rng(0)
        items = [train2.Item(rng.integers(0, 255, (32, w)).astype(np.uint8), np.array([1, 2, 3])) for w in (50, 97)]
        x, in_len, lab, tgt, tgt_len, kd = train2.make_batch(items, None, [{}])
        assert x.shape[3] % 16 == 0 and x.shape[3] >= 97
        assert in_len.tolist() == [math.ceil(50 / 16), math.ceil(97 / 16)]
        with pytest.raises(SystemExit):
            train2.set_geometry(24)
    finally:
        train2.set_geometry(8)
    assert (train2.STRIDE, train2.MAX_W) == (8, 1024)


def test_floor_tool_matches_the_runtime():
    import stride16_floor as F
    assert F.need("ああいい") == 6 and F.need("あいう") == 3 and F.need("……") == 3
    for s in (12, 16):
        blob, _ = E.build_blob(E.model_ops(_model(s), 32), VOCAB, 32, int8=True)
        r = _vocr(blob, mode="w8i")
        rng = np.random.default_rng(s)
        for _ in range(300):
            w, h = int(rng.integers(1, 3000)), int(rng.integers(1, 200))
            assert F.resized_width(w, h, s) == r.resized_width(w, h), (w, h, s)
        for w, h in ((45, 30), (15, 10), (1, 64), (4000, 30)):          # half-to-even ties and the clamps
            assert F.resized_width(w, h, s) == r.resized_width(w, h)


def _component(name, cand, inc, chars, vns, ratio16, names=None):
    import stride16_select as S
    c = S.Component(name)
    keys = [f"k{i}" for i in range(len(chars))]
    is_name = np.zeros(len(chars), bool) if names is None else names
    c.add("INC", inc, chars, vns, is_name, keys, {"blob_sha256": "x", "blob_bytes": 1, "arena_bytes": 1})
    for r, e in cand.items():
        c.add(r, e, chars, vns, is_name, keys, {"blob_sha256": r, "blob_bytes": 1, "arena_bytes": 1})
    rows = {k: {"T": {"8": 2 * int(10 * q), "12": int(10 * q * 4 / 3), "16": int(10 * q)}, "nr": 10,
                "platform": "psp"} for k, q in zip(keys, ratio16)}
    c.attach_floor(rows)
    return c


def test_select_rule_is_non_inferiority_with_tight_strata():
    """Not resolved worse adopts an equal candidate; a resolved loss on whole dialogue, a fixed-line component or
    only the tight stratum blocks it."""
    import stride16_select as S
    rng = np.random.default_rng(0)
    n = 400
    vns = [f"v{i % 40}" for i in range(n)]
    chars = np.full(n, 20.0)
    inc = rng.poisson(1.0, n).astype(float)
    ratio = np.where(np.arange(n) % 4 == 0, 1.1, 2.0)          # a quarter of the rows tight at stride 16

    def comps(wd_extra=0.0, tight_extra=0.0, tv_extra=0.0):
        same = {"s1": inc.copy(), "s2": inc.copy()}
        wd = {k: v + wd_extra + tight_extra * (ratio < 1.25) for k, v in same.items()}
        tv = {k: v + tv_extra for k, v in same.items()}
        return {"tv": _component("tv", tv, inc, chars, vns, ratio),
                "dn": _component("dn", same, inc, chars, vns, ratio),
                "wd": _component("wd", wd, inc, chars, vns, ratio)}

    proto = {"arms": {"N16": {"tier": "Normal", "stride": 16, "runs": {"fine_tunes": ["s1", "s2"]}}},
             "tiers": {"Normal": {"incumbent": {"name": "INC"},
                                  "limits": {"weights_eq": 1, "arena_w8i_le": 1, "box_s_le": 2.0}}},
             "estimates": {"table": {"N16": {"#85 no-PLD": {"333": [2.68, 0.743], "444": [2.01, 0.583]}}}}}
    d = S.decide(proto, "N16", comps(), n_boot=2000)
    assert d["adopted"] and all(d["rule"].values())
    assert d["strata"]["wd_tight"]["rows"] == 100 and d["strata"]["fixed_tight"]["rows"] == 200
    assert not S.decide(proto, "N16", comps(wd_extra=1.0), n_boot=2000)["adopted"]
    worse_tv = S.decide(proto, "N16", comps(tv_extra=1.0), n_boot=2000)
    assert not worse_tv["adopted"] and not worse_tv["rule"]["b_tv_not_resolved_worse"]
    tight_only = S.decide(proto, "N16", comps(tight_extra=1.0), n_boot=2000)
    assert not tight_only["rule"]["c_wd_tight_not_resolved_worse"] and not tight_only["adopted"]
    proto["tiers"]["Normal"]["limits"]["box_s_le"] = 0.5
    assert not S.decide(proto, "N16", comps(), n_boot=2000)["rule"]["d_within_limits"]


def test_protocol_is_consistent():
    p = json.loads((ROOT / "eval/stride16_protocol.json").read_text(encoding="utf-8"))
    for arm, a in p["arms"].items():
        assert a["runs"]["pretrains"] == [f"pre{a['stride']}_{a['tier'][0]}_seed{s}" for s in a["seeds"]]
        assert a["runs"]["fine_tunes"] == [f"s{a['stride']}f_{a['tier'][0]}_seed{s}" for s in a["seeds"]]
        assert arm in p["estimates"]["table"] and len(a["seeds"]) >= 2
        assert f"--stride {a['stride']}" in a["train2_args"]
    assert p["selection"]["order"] == {"Small": ["S16"], "Normal": ["N16", "N12"], "Large": ["L16"]}
    for t in p["tiers"].values():
        assert t["limits"]["weights_eq"] == t["incumbent"]["blob_bytes"]
