"""textdet (issue #96): the evaluation harness, the training targets, the data shards and the promotion rule.

No data, no network: synthetic inputs only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "eval", ROOT / "synth", ROOT / "train"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

torch = pytest.importorskip("torch")

import textdet as TD  # noqa: E402
import vocr3_report as V  # noqa: E402

import textdet_data as D  # noqa: E402
import textdet_model as TM  # noqa: E402
import textdet_train as TT  # noqa: E402

PROTOCOL = json.loads((ROOT / "eval" / "textdet_protocol.json").read_text(encoding="utf-8"))


def units(errors, chars, vns, mode="det"):
    return [{"unit": f"u{i}", "vn": v, "errors": e, "chars": c, "shot": f"s{i}", "platform": "vita",
             "variant": "base"} for i, (e, c, v) in enumerate(zip(errors, chars, vns))]


def test_vn_diff_equals_cluster_bootstrap_for_one_pair():
    rng = np.random.default_rng(0)
    n = 60
    vns = [f"v{int(k)}" for k in rng.integers(0, 15, n)]
    chars = rng.integers(5, 40, n).tolist()
    ea = [int(rng.integers(0, c // 2 + 1)) for c in chars]
    eb = [int(rng.integers(0, c // 2 + 1)) for c in chars]
    a, b = units(ea, chars, vns), units(eb, chars, vns)
    got = TD.vn_diff([(a, b)])
    ref = V.cluster_bootstrap({u["unit"]: (u["errors"], u["chars"]) for u in a},
                              {u["unit"]: (u["errors"], u["chars"]) for u in b},
                              {u["unit"]: u["vn"] for u in a}, 10000, 0, 5)
    assert got["clusters"] == ref["clusters"]
    assert got["ci95"] == pytest.approx(ref["ci95"], abs=1e-3)
    assert got["delta"] == pytest.approx(100 * (sum(ea) - sum(eb)) / sum(chars), abs=1e-3)


def test_vn_diff_mean_of_pairs():
    vns = ["v1", "v2", "v3", "v4", "v5", "v6"]
    chars = [10] * 6
    a1, b1 = units([1] * 6, chars, vns), units([2] * 6, chars, vns)
    a2, b2 = units([0] * 6, chars, vns), units([4] * 6, chars, vns)
    d = TD.vn_diff([(a1, b1), (a2, b2)])
    assert d["delta"] == pytest.approx(100 * ((6 - 12) + (0 - 24)) / 2 / 60)
    assert d["ci95"][1] < 0


def test_est_ms_is_h15s_console_rate():
    e = TD.est_ms(TD.H15_MACS)
    assert e["333"] == [5.0, 5.2]
    assert e["444"][0] == pytest.approx(5.0 * 333 / 444, abs=0.01)


def test_protocol_runs_and_shapes():
    runs = PROTOCOL["amendments"][1]["runs"]
    assert len(runs) == 22 and len(set(runs)) == 22
    for r in runs:
        shape, _, seed = r.rpartition("_seed")
        assert shape in TM.SHAPES and seed in ("1", "2")
        assert shape in PROTOCOL["sweep"]["shapes"]
    for name, s in PROTOCOL["sweep"]["shapes"].items():
        c = TM.cost(TM.SHAPES[name], 900, 150)
        assert (c["macs"], c["params"]) == (s["macs"], s["params"])


def test_targets_horizontal_line():
    s = 4
    line = [40, 20, 200, 44, 0, "text"]            # x0, y0, x1, y1: 24 px tall, centre 32
    K, Ov, A1, A2, valid, kern = TT.targets([line], [], 0, 0, 512, 192, s)
    rows = sorted(set(np.nonzero(kern)[0].tolist()))
    centres = [s * i + (s - 1) / 2 for i in rows]
    assert all(abs(c - 32) <= 6 for c in centres) and len(rows) == 3          # max(24/4, 2) = 6 px each side
    cols = sorted(set(np.nonzero(kern)[1].tolist()))
    assert cols[0] == 40 // s and cols[-1] == (200 - 1) // s
    i, j = rows[0], cols[0]
    assert A1[i, j] == pytest.approx(s * i + 1.5 - 20) and A2[i, j] == pytest.approx(44 - (s * i + 1.5))
    assert not Ov[kern].any()
    assert valid.all()


def test_targets_vertical_cut_conflict_and_ignore():
    s = 4
    vert = [100, 10, 124, 150, 1, "text"]
    cut = [300, -10, 400, 12, 0, "text"]           # cut across by the window's top edge
    K, Ov, A1, A2, valid, kern = TT.targets([vert, cut], [[0, 160, 60, 192]], 0, 0, 512, 192, s)
    assert Ov[kern].all() and kern.any()
    assert not valid[0:3, 300 // s:400 // s].any()
    assert not valid[160 // s:, 0:60 // s].any()
    a = [10, 20, 200, 30, 0, "text"]
    b = [10, 21, 200, 31, 0, "text"]               # claims the same cells
    K, Ov, A1, A2, valid, kern = TT.targets([a, b], [], 0, 0, 512, 192, s)
    assert not kern.any() and not K.any()


def test_targets_window_offset_and_padding():
    s = 8
    line = [520, 300, 700, 330, 0, "text"]
    K, Ov, A1, A2, valid, kern = TT.targets([line], [], 500, 280, 300, 100, s)
    assert valid[:13, :38].all() and not valid[13:].any() and not valid[:, 38:].any()
    assert kern.any()


def test_layer_masks_match_runtime_sizes():
    shape = TM.SHAPES["s8_c8"]
    masks = TM.layer_masks(shape, [(101, 333)], 192, 512)
    h, w = 101, 333
    for L, m in zip(TM.layers(shape), masks):
        h, w = TM.out_size(h, L["stride"]), TM.out_size(w, L["stride"])
        assert int(m[0, 0].sum()) == h * w


def test_masked_window_equals_region_alone():
    torch.manual_seed(0)
    shape = TM.SHAPES["s4_c4"]
    model = TM.TextDet(shape).eval()
    region = torch.rand(1, 1, 61, 150) * 2 - 1
    win = torch.zeros(1, 1, 192, 512)
    win[:, :, :61, :150] = region
    masks = TM.layer_masks(shape, [(61, 150)], 192, 512)
    with torch.no_grad():
        a = model(region)
        b = model(win, masks=masks)[:, :, :a.shape[2], :a.shape[3]]
    assert torch.allclose(a, b, atol=1e-5)


def test_group_region_joins_stacked_lines():
    lines = [[10, 10, 300, 34, 0, "text"], [10, 40, 280, 64, 0, "text"], [600, 300, 700, 320, 0, "text"]]
    rng = np.random.default_rng(1)
    x0, y0, x1, y1 = TT.group_region(lines, 0, rng, 544, 960, False)
    assert x0 <= 10 and y0 <= 10 and x1 >= 300 and y1 >= 64 and x1 < 600


def test_shard_roundtrip(tmp_path):
    a = (np.arange(30, dtype=np.uint8).reshape(5, 6))
    b = np.full((3, 4), 7, np.uint8)
    D.write_shard(tmp_path / "x_0000", [
        {"canvas": a, "lines": [[0, 0, 3, 2, 0, "text"]], "ignore": [], "src": "synth:t"},
        {"canvas": b, "lines": [], "ignore": [[0, 0, 1, 1]], "src": "realplus"}])
    data, samples = D.load_shard(tmp_path / "x_0000")
    assert (D.canvas(data, samples[0]) == a).all() and (D.canvas(data, samples[1]) == b).all()
    assert samples[0]["lines"] == [[0, 0, 3, 2, 0, "text"]] and samples[1]["ignore"] == [[0, 0, 1, 1]]


def test_main_box_excludes_ruby():
    H, W = 60, 100
    ink = np.zeros((H, W), bool)
    ruby = np.zeros((H, W), bool)
    ink[30:50, 10:90] = True                     # the line
    ink[22:27, 20:40] = True                     # its furigana
    ruby[22:27, 20:40] = True
    L = {"y0": 20, "y1": 50, "fs": 20, "ix0": 10, "ix1": 90, "ruby_top": 22}
    assert D.main_box(L, [L], ink, ruby, (H, W)) == (10, 30, 90, 50)


def _rep(name, blob, qual, mean=5.0):
    return {"blob_bytes": blob, "costs": {"vita_box": {"macs": blob * 100}}, "qualifies": qual, "is_eligible": True,
            "mean_wd_cer": mean, "conditions": {"no_resolved_native_vita_regression": True,
                                                "empty_rate_ge_0_75": True}, "point_below_both": False}


def test_rule_smallest_qualifying_and_resolved_margin():
    vns = [f"v{i}" for i in range(12)]
    chars = [20] * 12

    def det(errors):
        return {m: {"tv_dlg": {"det": units(errors, chars, vns)}, "dn_dlg": {"det": []}} for m in TD.MODELS}

    reps = {"small": _rep("small", 1000, True), "mid": _rep("mid", 2000, True), "big": _rep("big", 3000, True),
            "tiny": _rep("tiny", 500, False)}
    det_units = {"small": det([5] * 12), "mid": det([5] * 11 + [4]), "big": det([1] * 12), "tiny": det([0] * 12)}
    r = TD.apply_rule(reps, det_units)
    assert r["order"][0] == "tiny" and r["qualifying"] == ["small", "mid", "big"]
    assert r["winner"] == "big"                   # mid's margin is not resolved, big's is
    r2 = TD.apply_rule({k: dict(v, qualifies=False) for k, v in reps.items()}, det_units)
    assert r2["winner"] is None and r2["report_candidate"] in reps


def test_pareto_front():
    reps = {"a": {"blob_bytes": 1, "mean_wd_cer": 5}, "b": {"blob_bytes": 2, "mean_wd_cer": 4},
            "c": {"blob_bytes": 3, "mean_wd_cer": 4.5}, "d": {"blob_bytes": 4, "mean_wd_cer": 3}}
    assert TD.pareto(reps) == ["a", "b", "d"]
