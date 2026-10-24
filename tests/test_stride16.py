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


def test_protected_floor_strata(tmp_path, monkeypatch):
    """The report-only strata of the protected fixed lines: a line counts as tight / below the floor at a stride from
    its crop size and its reference, and each arm's errors are summed over those lines (names out of dialogue)."""
    import types

    import sets
    import stride16_floor as F
    lines = [types.SimpleNamespace(id="a", text="あいうえお", crop_rgb=np.zeros((32, 60, 3), np.uint8)),   # 4 frames
             types.SimpleNamespace(id="b", text="あい", crop_rgb=np.zeros((32, 400, 3), np.uint8)),        # 25 frames
             types.SimpleNamespace(id="c", text="なまえ", crop_rgb=np.zeros((32, 40, 3), np.uint8))]       # 3 frames
    monkeypatch.setattr(sets, "load", lambda name, data: (lines if name == "vita-test" else [], None))
    rec = [{"id": "a", "n": 5, "errors": 2, "styles": []}, {"id": "b", "n": 2, "errors": 0, "styles": []},
           {"id": "c", "n": 3, "errors": 1, "styles": ["name"]}]
    report = {"shared": {"fixed_line": {"console": {"X": {"lines": {"vita-test": rec, "psp-test": []}}}}}}
    (tmp_path / "r.json").write_text(json.dumps(report))
    F.main(["protected", "--report", str(tmp_path / "r.json"), "--data", "x", "--out", str(tmp_path / "o.json")])
    d = json.loads((tmp_path / "o.json").read_text())
    assert d["strata"]["s16_floor_vita-test"] == 1 and d["strata"]["s16_tight_vita-test"] == 2
    x = d["arms"]["X"]
    assert x["s16_floor_dialogue"] == {"lines": 1, "chars": 5, "errors": 2, "cer": 40.0}
    assert x["s16_tight_all"]["lines"] == 2 and x["s16_tight_all"]["errors"] == 3


def test_report_spec_uses_each_arms_best_run(tmp_path):
    import stride16_make_spec as M
    blobs = {}
    for name in ("VOCR5_S_w8.vocr", "VOCR5_N_w8.vocr", "VOCR5_L_w8.vocr", "r1.vocr", "r2.vocr", "r3.vocr", "r4.vocr",
                 "r5.vocr", "r6.vocr", "r7.vocr", "r8.vocr"):
        (tmp_path / name).write_bytes(name.encode())
        blobs[name] = M.sha256(tmp_path / name)
    proto = {"tiers": {t: {"incumbent": {"sha256": blobs[f"VOCR5_{t[0]}_w8.vocr"]}}
                       for t in ("Small", "Normal", "Large")},
             "arms": {"S16": {"tier": "Small", "stride": 16}, "N16": {"tier": "Normal", "stride": 16},
                      "L16": {"tier": "Large", "stride": 16}, "N12": {"tier": "Normal", "stride": 12}}}
    v5 = {"arms": [{"name": f"VOCR5_{t}", "file": f"VOCR5_{t}_w8.vocr", "sha256": blobs[f"VOCR5_{t}_w8.vocr"]}
                   for t in "SNL"]}

    def run(r, arm, score, k):
        return {"run": r, "arm": arm, "score": score, "blob": str(tmp_path / f"r{k}.vocr"),
                "blob_sha256": blobs[f"r{k}.vocr"]}
    sels = {"Small": {"kind": "stride16-selection", "tier": "Small", "adopted_arm": "S16",
                      "runs": [run("s16f_S_seed1", "S16", 3.5, 1), run("s16f_S_seed2", "S16", 3.4, 2)]},
            "Normal": {"kind": "stride16-selection", "tier": "Normal", "adopted_arm": None,
                       "runs": [run("s16f_N_seed1", "N16", 3.3, 3), run("s16f_N_seed2", "N16", 3.3, 4),
                                run("s12f_N_seed1", "N12", 3.2, 5), run("s12f_N_seed2", "N12", 3.1, 6)]},
            "Large": {"kind": "stride16-selection", "tier": "Large", "adopted_arm": "L16",
                      "runs": [run("s16f_L_seed1", "L16", 2.9, 7), run("s16f_L_seed2", "L16", 3.0, 8)]}}
    spec, runs = M.build(proto, sels, v5, [tmp_path], tmp_path / "out")
    assert runs["S16"] == {"run": "s16f_S_seed2", "adopted": True, "dev_score": 3.4}
    assert runs["N16"]["run"] == "s16f_N_seed1" and not runs["N16"]["adopted"]          # tie -> the lower seed
    assert runs["N12"]["run"] == "s12f_N_seed2" and runs["L16"]["run"] == "s16f_L_seed1"
    assert {(p["candidate"], p["baseline"]) for p in spec["pairs"]} == {
        ("S16", "VOCR5_S"), ("N16", "VOCR5_N"), ("N12", "VOCR5_N"), ("L16", "VOCR5_L"), ("N16", "N12")}
    assert (tmp_path / "out" / "N12_w8.vocr").read_bytes() == b"r6.vocr" and spec["gates"] == []


def test_fixed_rows_are_keyed_by_position_not_id(tmp_path):
    """teacher_val_dev holds ids that two different crops share (amendment 1): the frame budget of a row is its own,
    found by its position in the score file's line order."""
    import stride16_floor as F
    import stride16_select as S
    crops = [np.zeros((32, 60), np.uint8), np.zeros((32, 400), np.uint8), np.zeros((16, 400), np.uint8)]
    data = np.concatenate([c.ravel() for c in crops])
    offs = np.cumsum([0] + [c.size for c in crops])
    labels = ["あいうえお", "あい", "かきくけこさし"]
    meta = [{"id": "dup", "vn": "v1", "target": "vita"}, {"id": "dup", "vn": "v1", "target": "vita"},
            {"id": "u", "vn": "v2", "target": "psp"}]
    npz = tmp_path / "teacher_val_dev.npz"
    np.savez(npz, data=data, offsets=offs, shapes=np.array([c.shape for c in crops]), labels=json.dumps(labels),
             meta=json.dumps(meta))
    rows = F.fixed_rows(npz)
    assert [r["id"] for r in rows] == ["dup", "dup", "u"]
    assert [r["T"]["16"] for r in rows] == [4, 25, 50] and [r["nr"] for r in rows] == [5, 2, 7]
    score = {"dev": {"file": str(npz)}, "lines": [{"id": m["id"], "vn": m["vn"], "ref": lab, "tags": []}
                                                  for m, lab in zip(meta, labels)],
             "models": {"X": {"errors": {"rv": [1, 0, 0], "rvn": [1, 0, 0]}, "blob": "b", "blob_sha256": "s",
                              "blob_bytes": 1, "arena_bytes": 1}}}
    (tmp_path / "tv.json").write_text(json.dumps(score))
    c = S.Component("tv")
    S.load_fixed([tmp_path / "tv.json"], c, devnative=False)
    c.attach_floor({f"teacher_val_dev.npz|{k}|{v['id']}": v for k, v in enumerate(rows)})
    assert c.frames[16].tolist() == [4, 25, 50]
    assert (c.ratio(16) < 1.25).tolist() == [True, False, False]


def test_clannad_dialogue_pairs_skip_name_lines(tmp_path):
    """The CLANNAD dialogue pairs count only reference lines that are not a speaker name, per page."""
    import stride16_clannad_dialogue as C
    refs = [{"id": "a", "status": "ref", "name": False, "text": "あいう", "page": "p1"},
            {"id": "b", "status": "ref", "name": True, "text": "なまえ", "page": "p1"},
            {"id": "c", "status": "ref", "name": False, "text": "かきくけ", "page": "p2"},
            {"id": "d", "status": "excluded", "name": False, "text": "x", "page": "p2"}]
    (tmp_path / "refs.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in refs) + "\n")
    rows = [{"id": "a", "page": "p1", "ref": "あいう", "errors": {"X": 1, "Y": 0}},
            {"id": "b", "page": "p1", "ref": "なまえ", "errors": {"X": 3, "Y": 3}},
            {"id": "c", "page": "p2", "ref": "かきくけ", "errors": {"X": 0, "Y": 2}}]
    (tmp_path / "p.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    C.main(["--work", str(tmp_path), "--private", str(tmp_path / "p.jsonl"), "--pair", "X:Y",
            "--out", str(tmp_path / "o.json")])
    d = json.loads((tmp_path / "o.json").read_text())
    p = d["pairs"]["X:Y"]
    assert d["lines"] == 2 and p["chars"] == 7 and p["candidate_errors"] == 1 and p["baseline_errors"] == 2
    assert abs(p["cer_delta"] - 100 * (1 - 2) / 7) < 1e-4
