"""Tests of textspot3's trainer (train/textspot3_train.py, issue #108): the recipes, schedules and constants against
the protocol; the size weights (1 everywhere when every line owns >= 64 kernel cells, the exact values on hand-made
lines); clean's rules on hand-made lines and its counts; the scale rules; the base recipe equal to textspot2's ema
recipe (data + ema) step for step on tiny shards; the distillation loss on stride-4 maps with a tiny teacher; a
3-step run of every recipe on tiny hand-made shards in #97's format (family T: a VTSP version 2 at half_above 480
that synth/textspot2_twin.py reads and, with a C compiler, runtime/vocr_textspot2_test runs to the twin's result);
the refusals; the sweep's plan, check and summary. No screenshots, fonts or models; no network.
"""
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "train", ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textdet_data as DATA  # noqa: E402
import textdet_model as TM  # noqa: E402
import textdet_train as TT  # noqa: E402
import textspot2_export as X  # noqa: E402
import textspot2_shapes as S2  # noqa: E402
import textspot2_train as T2  # noqa: E402
import textspot2_twin as TW2  # noqa: E402
import textspot3_train as T3  # noqa: E402
import textspot_train as ST  # noqa: E402

PROTOCOL = json.loads((ROOT / "eval" / "textspot3_protocol.json").read_text(encoding="utf-8"))
# a tiny T shape (the grammar of the registered ones; fast enough for CI) and textspot2's d2s shape
T_TINY = {"name": "t_tiny", "stride": 8, "c0": 4, "c1": 8, "c2": 8, "chans": [8, 8], "dil": [1, 2], "block": "sep",
          "head": "kroa_x2"}
T_D2S = {k: (list(v) if isinstance(v, list) else v) for k, v in S2.SHAPES["s8_c24n4x2"].items() if k in T3.T_FIELDS}


@pytest.fixture(autouse=True)
def shapes_unchanged():
    """No test here may leave train/textdet_model.SHAPES changed (other tests compare it with #97's protocol)."""
    before = dict(TM.SHAPES)
    yield
    assert dict(TM.SHAPES) == before


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- the protocol's recipes, schedules and constants

def test_recipes_and_constants_are_the_protocols():
    assert set(T3.RECIPES) == set(PROTOCOL["recipes"]) - {"trainer"} == {"base", "size", "clean", "long", "kd"}
    for name in ("base", "size", "clean", "kd"):
        r = T3.RECIPES[name]
        assert (r["steps"], r["lr"], r["warmup"]) == (30000, 4e-3, 500)
    r = T3.RECIPES["long"]
    assert (r["steps"], r["lr"], r["warmup"], r["extras"]) == (60000, 2e-3, 1000, ())
    assert T3.RECIPES["base"]["extras"] == () and T3.RECIPES["size"]["extras"] == ("size",)
    assert T3.RECIPES["clean"]["extras"] == ("clean",) and T3.RECIPES["kd"]["extras"] is None
    assert int(TT.QAT_FRAC * 30000) == 24000 and int(TT.QAT_FRAC * 60000) == 48000
    assert (T3.EMA_DECAY, T3.EMA_START, T3.BN_BATCHES, TT.CAL_BATCHES) == (0.999, 1000, 32, 32)
    assert (T3.WIN_H, T3.WIN_W, T3.BATCH, T3.MIN_ACROSS, T3.CLIP) == (256, 512, 16, 4, 5.0)
    assert T3.MIX == TT.MIX == {"synth": 0.5, "realplus": 0.175, "teacher": 0.175, "owner": 0.125,
                                "devnative": 0.025}
    assert (TT.LR, TT.LR_MIN, TT.WARMUP, TT.WD) == (4e-3, 1e-4, 500, 1e-4)
    assert (T3.SIZE_N, T3.SIZE_MIN, T3.SIZE_MAX) == (64, 1.0, 4.0)
    assert (T3.CLEAN_WEIGHT, T3.CLEAN_RATIO, T3.CLEAN_MIN_WIDTH) == (1.0, 16, 6)
    assert T3.HALF_ABOVE_T == PROTOCOL["half_above"]["T"] == 480
    assert PROTOCOL["half_above"]["M"] == "w > 480 or h > 272" and T3.HALF_ABOVE_M == (480, 272)
    assert set(PROTOCOL["families"]) == set(T3.FAMILIES) == {"T", "M"}
    assert T3.MAP_STRIDE == 4 and T3.NET_STRIDE == 8 and T3.N_HEAD == 16 and T3.T_MAX_CHANNELS == 64
    rec = PROTOCOL["recipes"]
    assert "30,000 steps of 16 windows of 256 x 512" in rec["base"] and "QAT from step 24,000" in rec["base"]
    assert "decay 0.999 from step 1,000" in rec["base"] and "32 batches" in rec["base"]
    assert "clamp(sqrt(64 / n), 1, 4)" in rec["size"]
    assert "label weight < 1" in rec["clean"] and "16 x its width" in rec["clean"] and "6 px" in rec["clean"]
    assert "60,000 steps, peak LR 2e-3, 1,000 warm-up steps" in rec["long"] and "48,000" in rec["long"]
    assert "stride-4 maps" in rec["kd"] and "sigmoid(K_t) > 0.5" in rec["kd"]


def test_run_names_and_extras():
    assert T3.run_name("base", "t64n6", 2) == "base_t64n6_seed2"
    assert T3.run_name("size", "m_m", 1) == "size_m_m_seed1" and T3.run_name("clean", "x", 3) == "clean_x_seed3"
    assert T3.run_name("long", "m_m", 1) == "long_m_m_seed1"
    assert T3.run_name("kd", "t_s", 1, ("size", "clean")) == "kd_t_s_seed1" == T3.run_name("kd", "t_s", 1, ())
    # base + the kept extras (stage 2): named after them; base + one extra is that extra's recipe
    assert T3.run_name("base", "m_l", 1, ("size", "clean")) == "size+clean_m_l_seed1"
    assert T3.run_name("base", "m_l", 1, ("size",)) == T3.run_name("size", "m_l", 1) == "size_m_l_seed1"
    assert T3.run_name("base", "m_l", 1, ()) == "base_m_l_seed1"
    assert T3.parse_extras("clean,size") == ("size", "clean") and T3.parse_extras("none") == ()
    assert T3.parse_extras("size") == ("size",) and T3.parse_extras(None) is None
    for bad in ("", "size,size", "ema", "none,size", "cls"):
        with pytest.raises(ValueError):
            T3.parse_extras(bad)


def test_the_schedules():
    for step in range(0, 30000, 7):
        assert T2.lr_at(step, 30000, 4e-3, 500) == TT.lr_at(step, 30000)
    lr = [T2.lr_at(s, 60000, 2e-3, 1000) for s in range(60000)]
    assert lr[0] == pytest.approx(2e-6) and lr[999] == pytest.approx(2e-3) and max(lr) == pytest.approx(2e-3)
    assert all(a < b for a, b in zip(lr[:999], lr[1:1000])) and all(a >= b for a, b in zip(lr[1000:], lr[1001:]))
    assert lr[48000] == pytest.approx(1e-4 + 0.5 * 1.9e-3 * (1 + math.cos(math.pi * 47000 / 59000)))
    assert lr[-1] == pytest.approx(1e-4, abs=1e-8)


# ---------------------------------------------------------------- the scale rules and the windows

def test_the_scale_rules():
    assert T3.half_t(481, 100) == 2 and T3.half_t(480, 544) == 1 and T3.half_t(960, 544) == 2
    assert T3.half_m(481, 100) == 2 and T3.half_m(480, 273) == 2 and T3.half_m(480, 272) == 1
    assert T3.half_m(363, 272) == 1 and T3.half_m(725, 544) == 2
    # on the development and test frames' sizes (eval/textspot3_protocol.json "half_above" why), T, M and
    # textspot's half_above 600 halve the same frames
    for w, h in ((960, 544), (728, 540), (725, 538), (480, 272)):
        h600 = 2 if w > 600 else 1
        assert T3.half_t(w, h) == T3.half_m(w, h) == h600
    assert T3.scale_record("T") == {"rule": "half_above", "half_above": 480,
                                    "what": "a canvas wider than 480 px is halved (textspot_twin.halve)"}
    assert T3.scale_record("M")["half_above_h"] == 272


def test_the_window_is_textspots_at_600_and_halves_above_480(monkeypatch):
    rng = np.random.default_rng(3)
    for W, H in ((960, 544), (700, 300), (560, 300), (481, 272), (480, 272), (363, 272), (200, 120), (1001, 541)):
        g = rng.integers(0, 256, (H, W)).astype(np.uint8)
        for seed in range(4):
            got = T3.window(np.random.default_rng(seed), g, T3.half_t)
            if not 480 < W <= 600:
                want = ST.window(np.random.default_rng(seed), g, "h2")
                assert got[1:] == want[1:] and np.array_equal(got[0], want[0])
            assert got[5] == (2 if W > 480 else 1)
            monkeypatch.setattr(T3, "HALF_ABOVE_T", 600)
            at600 = T3.window(np.random.default_rng(seed), g, T3.half_t)
            want = ST.window(np.random.default_rng(seed), g, "h2")
            assert at600[1:] == want[1:] and np.array_equal(at600[0], want[0])
            monkeypatch.setattr(T3, "HALF_ABOVE_T", 480)


# ---------------------------------------------------------------- targets and the size weights

def random_lines(rng, n, H, W, weights=False):
    out = []
    for _ in range(n):
        vertical = int(rng.random() < 0.3)
        if vertical:
            x0, w = int(rng.integers(-20, W)), int(rng.integers(2, 40))
            y0, h = int(rng.integers(-40, H)), int(rng.integers(10, 300))
        else:
            x0, w = int(rng.integers(-40, W)), int(rng.integers(10, 400))
            y0, h = int(rng.integers(-20, H)), int(rng.integers(2, 40))
        ln = [x0, y0, x0 + w, y0 + h, vertical]
        if weights:
            ln.append(float(rng.choice([1.0, 0.5, 0.8, 0.95])))
        out.append(ln)
    return out


@pytest.mark.parametrize("s", [4, 8])
def test_targets_without_size_are_textspot2s(s):
    rng = np.random.default_rng(s)
    for _ in range(120):
        h, w = int(rng.integers(20, 257)), int(rng.integers(20, 513))
        wx, wy = int(rng.integers(0, 300)), int(rng.integers(0, 200))
        lines = [[ln[0] + wx, ln[1] + wy, ln[2] + wx, ln[3] + wy, *ln[4:]]
                 for ln in random_lines(rng, 14, h, w, weights=True)]
        ignore = [[wx + int(a), wy + int(b), wx + int(a) + 40, wy + int(b) + 12]
                  for a, b in zip(rng.integers(-20, w, 3), rng.integers(-10, h, 3))]
        got = T3.targets(lines, ignore, wx, wy, w, h, s)
        want = T2.targets(lines, ignore, wx, wy, w, h, s)
        for a, b in zip(got, want):
            assert a.dtype == b.dtype and np.array_equal(a, b)
        sized = T3.targets(lines, ignore, wx, wy, w, h, s, size=True)
        for a, b in zip(sized[:6], want[:6]):
            assert np.array_equal(a, b)                       # size changes the weight map only
        k = want[5]
        assert np.all(sized[6][~k] == 1) and np.all(sized[6][k] >= want[6][k])
        assert np.all(sized[6][k] <= 4 * want[6][k] * (1 + 1e-6))


def own_counts(lines, w=512, h=256, s=4):
    """Kernel cells per line, counted from single-line targets (for lines that claim no common cell)."""
    return [int(T3.targets([ln], [], 0, 0, w, h, s)[5].sum()) for ln in lines]


def test_size_weight_is_1_everywhere_when_every_line_owns_64_cells():
    rng = np.random.default_rng(5)
    for _ in range(60):
        lines, y = [], int(rng.integers(0, 10))
        while True:                                     # long horizontal lines, apart: no cell claimed twice
            hgt = int(rng.integers(6, 30))
            if y + hgt > 256:
                break
            x0 = int(rng.integers(0, 200))
            x1 = int(rng.integers(x0 + 260, 513))
            lines.append([x0, y, x1, y + hgt, 0, float(rng.choice([1.0, 0.6, 0.85]))])
            y += hgt + int(rng.integers(12, 40))
        if rng.random() < 0.5:                          # and a full-height vertical one, right of them all
            lines = [ln for ln in lines if ln[2] < 470] + [[480, 0, 500, 256, 1, 0.7]]
        n = own_counts(lines)
        assert min(n) >= 64
        plain = T3.targets(lines, [], 0, 0, 512, 256, 4)
        sized = T3.targets(lines, [], 0, 0, 512, 256, 4, size=True)
        assert np.array_equal(plain[6], sized[6]) and int(plain[5].sum()) == sum(n)
    assert T3.size_factor([64, 65, 1000, 100000]).tolist() == [1.0, 1.0, 1.0, 1.0]


def test_size_weights_on_hand_made_lines():
    # stride 4; an 8-px-tall horizontal line owns one row of cells (r = 2 around its centre), a 16-px-tall one two
    lines = [
        [0, 8, 16, 16, 0, 1.0],         # 4 cells: x 4
        [40, 8, 104, 16, 0, 1.0],       # 16 cells: x 2
        [120, 8, 156, 16, 0, 1.0],      # 9 cells: x 8/3
        [200, 8, 456, 16, 0, 1.0],      # 64 cells: x 1
        [0, 40, 64, 48, 0, 0.5],        # 16 cells, weight 0.5: 1.0
        [100, 40, 132, 56, 0, 0.75],    # two rows of 8: 16 cells, x 2 x 0.75 = 1.5
        [300, 60, 308, 68, 0, 1.0],     # 2 cells (8 x 8): x 4 (clamped from sqrt(32))
        [400, 100, 408, 228, 1, 0.8],   # vertical, 8 px wide, 128 px long: one column of 32 cells: x sqrt(2) x 0.8
        [0, 200, 400, 208, 0, 1.0],     # 100 cells: x 1
    ]
    want_n = [4, 16, 9, 64, 16, 16, 2, 32, 100]
    assert own_counts(lines) == want_n
    want_f = [4.0, 2.0, 8 / 3, 1.0, 1.0, 1.5, 4.0, math.sqrt(2) * 0.8, 1.0]
    K, O, A1, A2, vm, km, W = T3.targets(lines, [], 0, 0, 512, 256, 4, size=True)  # noqa: E741
    for ln, f in zip(lines, want_f):
        one = T3.targets([ln], [], 0, 0, 512, 256, 4)[5]
        assert np.all(W[one] == np.float32(f)), (ln, np.unique(W[one]), f)
    assert np.all(W[~km] == 1)
    # a line cut along by the window counts its cells inside it; cells two lines claim are neither's
    cut = [[-200, 8, 16, 16, 0, 1.0]]                             # inside: x 0-16, 4 cells: x 4
    W = T3.targets(cut, [], 0, 0, 512, 256, 4, size=True)[6]
    assert sorted(np.unique(W).tolist()) == [1.0, 4.0] and int((W == 4).sum()) == 4
    pair = [[0, 8, 64, 16, 0, 1.0], [48, 8, 112, 16, 0, 1.0]]    # 16 cells each, 4 shared: 12 own cells each
    K, O, A1, A2, vm, km, W = T3.targets(pair, [], 0, 0, 512, 256, 4, size=True)  # noqa: E741
    assert int(km.sum()) == 24 and np.all(W[km] == np.float32(math.sqrt(64 / 12)))
    # a window at an offset: the same line in window coordinates
    off = T3.targets([[100, 58, 116, 66, 0, 1.0]], [], 100, 50, 300, 200, 4, size=True)
    assert int(off[5].sum()) == 4 and np.all(off[6][off[5]] == 4)
    # stride 8 (no family trains at 8, but the rule is the map's stride)
    W8 = T3.targets([[0, 8, 64, 24, 0, 1.0]], [], 0, 0, 512, 256, 8, size=True)
    assert int(W8[5].sum()) == 8 and np.all(W8[6][W8[5]] == np.float32(math.sqrt(8)))


# ---------------------------------------------------------------- clean

def test_the_clean_rules_on_hand_made_lines():
    keep = [[10, 10, 200, 30, 0, "text", 1.0],          # horizontal, weight 1
            [10, 40, 200, 60, 0, "text"],                # no weight field: 1
            [300, 10, 310, 170, 1, "text", 1.0],         # vertical, 160 = 16 x 10: not longer
            [320, 10, 326, 50, 1, "text", 1.0],          # vertical, 6 px wide
            [400, 10, 405, 30, 0, "text", 1.0]]          # horizontal and 5 px wide: the vertical rules only
    drop = [([10, 70, 200, 90, 0, "text", 0.75], ("weight",)),
            ([10, 100, 200, 120, 0, "text", 0.999], ("weight",)),
            ([340, 10, 350, 171, 1, "text", 1.0], ("vertical_long",)),
            ([360, 10, 365, 40, 1, "text", 1.0], ("vertical_narrow",)),
            ([370, 10, 374, 200, 1, "text", 0.6], ("weight", "vertical_long", "vertical_narrow"))]
    for ln in keep:
        assert T3.clean_rule(ln) == ()
    for ln, hit in drop:
        assert T3.clean_rule(ln) == hit
    smp = {"lines": [*keep, *(ln for ln, _ in drop)], "ignore": [[500, 5, 520, 25]]}
    kept, ignore, removed = T3.clean_sample(smp)
    assert kept == keep and [ln for ln, _ in removed] == [ln for ln, _ in drop]
    assert ignore == [[500, 5, 520, 25], *([int(v) for v in ln[:4]] for ln, _ in drop)]

    class P:
        items = {"synth": [(0, {"lines": [[0, 0, 5, 300, 1, "ui"], [0, 0, 9, 9, 0, "ui", 0.5]], "ignore": []})],
                 "realplus": [(0, json.loads(json.dumps(smp))), (0, {"lines": [keep[0]], "ignore": []})],
                 "teacher": [(0, {"lines": [drop[0][0]], "ignore": []})], "owner": [], "devnative": []}

    pool = P()
    synth_before = json.dumps(pool.items["synth"])
    rec = T3.clean_pool(pool)
    assert json.dumps(pool.items["synth"]) == synth_before                    # never the synthetic half
    rp = rec["by_source"]["realplus"]
    assert (rp["canvases"], rp["canvases_changed"], rp["lines"], rp["removed"], rp["kept"]) == (2, 1, 11, 5, 6)
    assert rp["hits"] == {"weight": 3, "vertical_long": 2, "vertical_narrow": 2}
    assert (rp["vertical"], rp["removed_vertical"], rp["removed_by_vertical_rules_only"]) == (5, 3, 2)
    assert pool.items["realplus"][0][1]["lines"] == keep and len(pool.items["realplus"][0][1]["ignore"]) == 6
    assert rec["by_source"]["teacher"]["removed"] == 1 and pool.items["teacher"][0][1]["lines"] == []
    assert rec["total"]["removed"] == 6 and rec["total"]["hits"]["weight"] == 4
    assert rec["by_source"]["owner"]["canvases"] == 0


# ---------------------------------------------------------------- tiny shards and runs

def canvas(rng, h, w, boxes):
    g = rng.integers(0, 60, (h, w)).astype(np.uint8)
    for x0, y0, x1, y1 in boxes:
        g[y0:y1, x0:x1] = 230
    return g


def tiny_shards(tmp_path, weights=True, mid=False):
    """The protocol's directories in miniature: synth_frames (a synthetic canvas wider than 600 px, halved by every
    rule; with mid, a second one 560 px wide, halved at 480 but not at 600), lab_realplus and more_realplus (real
    canvases whose lines carry a 7th field; a low-weight line and a long thin vertical one for clean), a synthetic
    manifest and a textspot2 pin with skip lists."""
    rng = np.random.default_rng(11)
    dirs = {k: tmp_path / k for k in ("synth_frames", "lab_realplus", "more_realplus")}
    for d in dirs.values():
        d.mkdir()
    lines = [[100, 40, 400, 58, 0, "dialogue"], [50, 200, 650, 226, 0, "name"], [660, 20, 680, 280, 1, "dialogue"],
             [10, 100, 60, 106, 0, "ui"], [420, 120, 600, 140, 0, "choice"]]
    synth = [{"canvas": canvas(rng, 300, 700, [ln[:4] for ln in lines]), "lines": lines,
              "ignore": [[300, 100, 340, 110]], "src": "synth:frame", "kind": "vita", "vn": None, "shot": None}]
    if mid:
        mlines = [[20, 30, 500, 60, 0, "dialogue"], [30, 100, 90, 112, 0, "ui"], [520, 10, 540, 250, 1, "name"]]
        synth.append({"canvas": canvas(rng, 280, 560, [ln[:4] for ln in mlines]), "lines": mlines, "ignore": [],
                      "src": "synth:frame", "kind": "vita", "vn": None, "shot": None})
    w1, w2 = (0.75, 0.6) if weights else (1.0, 1.0)
    lab = {"canvas": canvas(rng, 136, 240, [[20, 50, 180, 66], [20, 90, 200, 110], [210, 5, 216, 130]]),
           "lines": [[20, 50, 180, 66, 0, "text", 1.0], [20, 90, 200, 110, 0, "text", w1],
                     [210, 5, 216, 130, 1, "text", 1.0]],
           "ignore": [[200, 10, 230, 30]], "src": "realplus", "kind": "psp", "vn": "v1", "shot": "v1_1"}
    more = {"canvas": canvas(rng, 120, 200, [[30, 40, 170, 58], [10, 80, 26, 118]]),
            "lines": [[30, 40, 170, 58, 0, "text", w2], [10, 80, 26, 118, 1, "text", 1.0]],
            "ignore": [], "src": "realplus", "kind": "psp", "vn": "v2", "shot": "v2_1"}
    skipped = {"canvas": canvas(rng, 64, 96, [[10, 20, 80, 36]]), "lines": [[10, 20, 80, 36, 0, "text", 1.0]],
               "ignore": [], "src": "realplus", "kind": "psp", "vn": "v9", "shot": "v9_1"}
    DATA.write_shard(dirs["synth_frames"] / "synth_frames_0000", synth)
    DATA.write_shard(dirs["lab_realplus"] / "lab_realplus_0000", [lab, skipped])
    DATA.write_shard(dirs["more_realplus"] / "more_realplus_0000", [more])

    def pin(d, n):
        return {"dir": d.name, "name": n, "npy_sha256": sha(d / f"{n}.npy"), "json_sha256": sha(d / f"{n}.json")}

    man_s = tmp_path / "synth_man.json"
    man_s.write_text(json.dumps({"kind": "test-shards", "shards": [pin(dirs["synth_frames"], "synth_frames_0000")]}))
    man_2 = tmp_path / "textspot2_shards_v1.json"
    man_2.write_text(json.dumps({
        "kind": "textspot2-shards",
        "shards": [pin(dirs["lab_realplus"], "lab_realplus_0000"), pin(dirs["more_realplus"], "more_realplus_0000")],
        "real": {"measured": {"skip": {
            "eval_screenshot": {"shots": 1, "canvases": 1, "by_source": {"realplus": 1}, "skip_shots": ["v9_1"]},
            "linefind_list_vn": {"shots": 0, "canvases": 0, "by_source": {}, "skip_vns": ["v99"]}}}}}))
    return [man_s, man_2], dirs


def args3(mans, dirs, recipe, out, shape=T_TINY, family="T", seed=1, steps=3, more=None):
    d = [dirs["synth_frames"], dirs["lab_realplus"], dirs["more_realplus"]]
    a = ["--recipe", recipe, "--family", family, "--shape", shape["name"], "--shape-json", json.dumps(shape),
         "--seed", str(seed), "--manifest", *map(str, mans), "--shards", *map(str, d), "--out", str(out),
         "--steps", str(steps), "--threads", "1", "--log-every", "1", "--cal-batches", "2", "--bn-batches", "2",
         "--ema-start", "1", "--workers", "0", "--allow-missing-sources"]
    return a + list(more or [])


def args2(mans, dirs, recipe, out, shape="s8_c24n4x2", seed=1, steps=3):
    """textspot2's trainer on the same shards (tests/test_textspot2_train.py's base_args)."""
    d = [dirs["synth_frames"], dirs["lab_realplus"], dirs["more_realplus"]]
    return ["--recipe", recipe, "--shape", shape, "--seed", str(seed), "--manifest", *map(str, mans),
            "--shards", *map(str, d), "--out", str(out), "--steps", str(steps), "--threads", "1",
            "--log-every", "1", "--cal-batches", "2", "--bn-batches", "2", "--ema-start", "1", "--workers", "0",
            "--allow-missing-sources"]


def log_of(run, drop_sec=False):
    out = [json.loads(x) for x in (run / "log.jsonl").read_text().splitlines()]
    if drop_sec:
        for r in out:
            r.pop("sec", None)
    return out


def same_run(a, b):
    """Two runs trained the same: every log record but its time, the QAT checkpoint, the int8 network (outside its
    CRC and checkpoint-hash fields)."""
    assert log_of(a, True) == log_of(b, True)
    ca = torch.load(a / "final.pt", map_location="cpu", weights_only=False)
    cb = torch.load(b / "final.pt", map_location="cpu", weights_only=False)
    assert ca["kind"] == cb["kind"] and ca["act_scale"] == cb["act_scale"] and len(ca["w"]) == len(cb["w"])
    assert all(torch.equal(x, y) for x, y in zip(ca["w"] + ca["b"], cb["w"] + cb["b"]))
    va, vb = (a / "final.vtd").read_bytes(), (b / "final.vtd").read_bytes()
    assert (va[:12], va[16:24], va[32:]) == (vb[:12], vb[16:24], vb[32:])


def test_base_is_textspot2s_data_plus_ema_step_for_step(tmp_path):
    """The same shards, seed and steps through textspot2's ema recipe (data + ema) and base, on textspot2's d2s
    shape: the same losses, LRs and EMA / QAT records at every step, the same QAT checkpoint and the same int8
    network. The shards' canvases are 700, 240 and 200 px wide, which 480 and 600 halve alike."""
    mans, dirs = tiny_shards(tmp_path)
    a, b = tmp_path / "t2", tmp_path / "t3"
    T2.main(args2(mans, dirs, "ema", a, steps=5))
    T3.main(args3(mans, dirs, "base", b, shape=T_D2S, steps=5))
    same_run(a, b)
    assert any(r.get("ema_swap") for r in log_of(b)) and any(r.get("qat") for r in log_of(b))
    i2, i3 = json.loads((a / "run.json").read_text()), json.loads((b / "run.json").read_text())
    assert (i2["half_above"], i3["scale"]["half_above"]) == (600, 480)
    m2, m3 = TW2.Model((a / "final.vtsp").read_bytes()), TW2.Model((b / "final.vtsp").read_bytes())
    assert (m2.version, m2.half_above, m3.version, m3.half_above) == (2, 600, 2, 480)
    assert (a / "final.vtsp").read_bytes()[32:] == (a / "final.vtd").read_bytes()
    assert i3["pool"]["by_source"] == i2["pool"]["by_source"] and i3["skip"] == i2["skip"]


def test_base_at_600_is_textspot2s_on_a_canvas_between_480_and_600(tmp_path, monkeypatch):
    """With a 560-px canvas in the pool: base at half_above 600 is textspot2's run; at 480 it is not (only that canvas
    is drawn at another scale)."""
    mans, dirs = tiny_shards(tmp_path, mid=True)
    a, b, c = tmp_path / "t2", tmp_path / "t3_600", tmp_path / "t3_480"
    T2.main(args2(mans, dirs, "ema", a, steps=4))
    monkeypatch.setattr(T3, "HALF_ABOVE_T", 600)
    T3.main(args3(mans, dirs, "base", b, shape=T_D2S, steps=4))
    same_run(a, b)
    monkeypatch.setattr(T3, "HALF_ABOVE_T", 480)
    T3.main(args3(mans, dirs, "base", c, shape=T_D2S, steps=4))
    assert log_of(c, True) != log_of(a, True)


CC = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
REPRO = ROOT / "eval" / "reproducibility" / "textspot3-20261021"


def script(name, file):
    """One of the sweep's scripts as a module of its own name, without putting its directory on sys.path
    (textspot2's check_run.py and runs_summary.py share the file names, and its tests import them by name)."""
    import importlib.util
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPRO / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PL = script("textspot3_plan", "textspot3_plan.py")
CR = script("textspot3_check_run", "check_run.py")
RS = script("textspot3_runs_summary", "runs_summary.py")


@pytest.fixture(scope="module")
def cbin2(tmp_path_factory):
    """runtime/vocr_textspot2_test built with the system compiler (skipped without one)."""
    if CC is None:
        pytest.skip("no C compiler")
    out = tmp_path_factory.mktemp("cbuild") / "vocr_textspot2_test"
    subprocess.run([CC, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror",
                    *(str(ROOT / "runtime" / f) for f in ("vocr_lines.c", "vocr_lines_rows.c", "vocr_textspot2.c",
                                                          "vocr_textspot2_test.c")), "-o", str(out)], check=True)
    return out


def check_t_blobs(run, shape):
    done = json.loads((run / "done.json").read_text())
    vtd, vtsp = (run / "final.vtd").read_bytes(), (run / "final.vtsp").read_bytes()
    assert done["vtd_sha256"] == sha(run / "final.vtd") and done["vtsp_sha256"] == sha(run / "final.vtsp")
    assert done["vtsp_bytes"] == len(vtsp) == len(vtd) + 32 and done["pt_sha256"] == sha(run / "final.pt")
    assert done["vtsp_version"] == 2 and done["family"] == "T"
    net = TW2.Net(vtd)
    assert (net.stride, net.outputs, net.min_cells, net.name) == (8, 16, 2, shape)
    m = TW2.Model(vtsp)
    assert (m.version, m.head, m.half_above, m.conf_min, m.stride, m.name) == (2, "kroa_x2", 480, 1, 4, shape)
    assert TW2.pack(vtd, 480, 1, version=TW2.VERSION_D2S) == vtsp
    assert (vtd, vtsp) == X.export_vtsp(run / "final.pt", 480, 1)
    return done


def test_a_three_step_run_of_every_recipe(tmp_path, cbin2):
    mans, dirs = tiny_shards(tmp_path)
    runs = tmp_path / "runs"
    made = {}
    for recipe in ("base", "size", "clean", "long"):
        out = runs / T3.run_name(recipe, "t_tiny", 1)
        T3.main(args3(mans, dirs, recipe, out))
        made[recipe] = out
    out = runs / "kd_t_tiny_seed1"
    T3.main(args3(mans, dirs, "kd", out, more=["--teacher", str(made["base"]), "--extras", "size,clean"]))
    made["kd"] = out
    out = runs / "size+clean_t_tiny_seed1"                        # stage 2's "base + the kept extras"
    T3.main(args3(mans, dirs, "base", out, more=["--extras", "size,clean"]))
    made["base+"] = out
    extras = {"base": [], "size": ["size"], "clean": ["clean"], "long": [], "kd": ["size", "clean"],
              "base+": ["size", "clean"]}
    fin = runs / "finished.txt"
    for key, run in made.items():
        recipe = key.rstrip("+")
        info = json.loads((run / "run.json").read_text())
        assert info["name"] == run.name and info["recipe"] == recipe and info["kind"] == "textspot3-run"
        assert (info["family"], info["scale"]["half_above"], info["steps"], info["qat_start"]) == ("T", 480, 3, 2)
        assert info["shape"] == T_TINY and info["shape_source"].startswith("--shape-json")
        assert info["extras"] == extras[key] and info["settings"]["extras"] == extras[key]
        assert (info["settings"]["size"] is not None) == ("size" in extras[key])
        assert info["settings"]["lr"]["peak"] == (2e-3 if recipe == "long" else 4e-3)
        assert info["settings"]["lr"]["warmup"] == (1000 if recipe == "long" else 500)
        assert info["skip"]["dropped_total"] == 1                     # v9_1 is on the pin's skip list
        assert info["data_rule"]["unread"] == {} and info["settings"]["network"]["family"] == "T"
        clean = info["settings"]["clean"]
        if "clean" in extras[key]:
            # lab: one weight-0.75 line, one 6 x 125 vertical line (125 > 96: longer than 16 x 6); more: weight 0.6
            assert clean["total"]["removed"] == 3 and clean["total"]["hits"] == {
                "weight": 2, "vertical_long": 1, "vertical_narrow": 0}
            assert clean["by_source"]["realplus"]["canvases"] == 2 and clean["total"]["kept"] == 2
            assert info["pool"]["by_source"]["realplus"]["lines"] == 2
            assert info["pool"]["by_source"]["synth"]["lines"] == 5             # the synthetic half untouched
        else:
            assert clean is None and info["pool"]["by_source"]["realplus"]["lines"] == 5
        for f in ("train/textspot3_train.py", "train/textspot2_train.py", "train/textspot_train.py",
                  "train/textdet_train.py", "train/textspot2_export.py", "synth/textspot2_twin.py"):
            assert info["code_sha256"][f] == sha(ROOT / f)
        assert info["protocol_sha256"] == sha(ROOT / "eval" / "textspot3_protocol.json")
        assert info["textspot2_protocol_sha256"] == sha(ROOT / "eval" / "textspot2_protocol.json")
        log = log_of(run)
        steps = [r for r in log if "lr" in r]
        assert [r["step"] for r in steps] == [1, 2, 3] and any(r.get("qat") for r in log)
        assert all(np.isfinite(r[k]) for r in steps for k in ("k_bce", "dice", "o", "a", "loss"))
        assert [r["step"] for r in log if r.get("ema") == "start"] == [1]
        swap = [r for r in log if r.get("ema_swap")]
        assert len(swap) == 1 and swap[0]["step"] == 2 and swap[0]["ema_steps"] == 2 and swap[0]["bn_layers"] > 0
        avg = [r for r in log if r.get("qat_average") == "the run's model"]
        assert len(avg) == 1 and avg[0]["qat_average_steps"] == 2
        if recipe == "kd":
            t = info["settings"]["kd"]["teacher"]
            assert t["name"] == "base_t_tiny_seed1" and t["pt_sha256"] == sha(made["base"] / "final.pt")
            assert t["family"] == "T" and t["shape"] == "t_tiny" and t["scale_disagreements"] == {}
            for r in steps:
                assert r["loss"] == pytest.approx(r["k_bce"] + r["dice"] + 0.5 * r["o"] + r["a"] + r["kd_k"] +
                                                  0.5 * r["kd_o"] + r["kd_a"], abs=2e-4)
        else:
            assert info["settings"]["kd"] is None and not any("kd_k" in r for r in steps)
            for r in steps:
                assert r["loss"] == pytest.approx(r["k_bce"] + r["dice"] + 0.5 * r["o"] + r["a"], abs=2e-4)
        check_t_blobs(run, "t_tiny")
        p = subprocess.run([sys.executable, str(REPRO / "check_run.py"), str(run), "--finished", str(fin),
                            "--cbin2", str(cbin2)], capture_output=True, text=True, timeout=600, cwd=ROOT)
        assert p.returncode == 0 and f"CHECK_OK {run.name} " in p.stdout, p.stdout + p.stderr
        assert "'c_runtime': 'equal to the twin'" in p.stdout
    assert len(fin.read_text().splitlines()) == 6
    ck = torch.load(made["base"] / "final.pt", map_location="cpu", weights_only=False)
    assert ck["kind"] == "textdet-qat" and ck["shape"] == T_TINY and len(ck["act_scale"]) == len(ck["w"])
    assert ck["w"][-1].shape[0] == 16
    # the same seed draws the same windows with and without clean, size or kd: their first float steps differ
    # only through the labels and the loss (base and long differ through the LR from step 1)
    first = {k: [r for r in log_of(v) if "lr" in r][0] for k, v in made.items()}
    assert first["base"]["lr"] == first["size"]["lr"] == first["clean"]["lr"] == first["kd"]["lr"] != \
        first["long"]["lr"]
    assert first["size"]["k_bce"] >= first["base"]["k_bce"] and first["base"]["dice"] == first["size"]["dice"]


def test_the_kd_loss_on_stride4_maps_with_a_tiny_teacher(tmp_path):
    """A d2s student and a d2s teacher, both through to_stride4: textspot2's kd loss on their stride-4 maps; the
    teacher's final.pt in fake quantisation, frozen; a student equal to its teacher has no kd gradient."""
    mans, dirs = tiny_shards(tmp_path)
    teacher_run = tmp_path / "runs" / "base_t_tiny_seed2"
    T3.main(args3(mans, dirs, "base", teacher_run, seed=2))
    teacher, fam, rec = T3.load_teacher(teacher_run)
    assert isinstance(fam, T3.FamilyT) and not teacher.training and teacher.quant
    assert all(not p.requires_grad for p in teacher.parameters())
    assert rec["pt_sha256"] == sha(teacher_run / "final.pt") and rec["family"] == "T"
    g = torch.Generator().manual_seed(4)
    x = torch.randint(0, 256, (2, 1, T3.WIN_H, T3.WIN_W), generator=g, dtype=torch.uint8)
    vhw = [(256, 512), (150, 301)]
    xin = TT.to_input(x)
    t_maps = fam.maps(teacher(xin, masks=fam.masks(vhw)))
    assert t_maps.shape == (2, 4, 64, 128) and not t_maps.requires_grad
    torch.manual_seed(0)
    student = X.TextDetD2S(T_TINY)
    s_maps = fam.maps(student(xin, masks=fam.masks(vhw)))
    assert s_maps.shape == t_maps.shape
    vm = torch.zeros(2, 64, 128, dtype=torch.bool)
    vm[0], vm[1, :38, :76] = True, True
    total, parts = T2.kd_loss(s_maps, t_maps, vm)
    F = torch.nn.functional
    pk = torch.sigmoid(t_maps[:, 0])
    tm = pk > 0.5
    want_k = float(F.binary_cross_entropy_with_logits(s_maps[:, 0], pk, reduction="none")[vm].mean())
    assert parts["kd_k"] == pytest.approx(want_k, rel=1e-5) and parts["kd_cells"] == int(tm.sum())
    total.backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in student.parameters())
    # the student's masked area past the window's region gets no maps (the region alone)
    assert float(s_maps[1, :, 38:].abs().max()) == 0 and float(t_maps[1, :, :, 76:].abs().max()) == 0
    # the teacher equal to the student: zero kd gradient
    same = t_maps.clone().requires_grad_(True)
    tot, _ = T2.kd_loss(same, t_maps, vm)
    tot.backward()
    assert float(same.grad.abs().max()) < 1e-6
    # its maps are the twin's head rearrangement of its 16 channels
    head = teacher(xin, masks=fam.masks(vhw))[0].detach().numpy()
    assert np.array_equal(t_maps[0].numpy(), TW2.head_to_maps(head, 256, 512))


def test_the_refusals(tmp_path):
    mans, dirs = tiny_shards(tmp_path)
    out = tmp_path / "x"
    for recipe, extra in (("size", ["--extras", "clean"]), ("clean", ["--extras", "size"]),
                          ("long", ["--extras", "size"]), ("base", ["--teacher", str(tmp_path)]),
                          ("kd", ["--teacher", str(tmp_path)]), ("kd", ["--extras", "none"]),
                          ("base", ["--extras", "ema"]), ("base", ["--ema-start", "3"])):
        with pytest.raises(SystemExit):
            T3.parse(args3(mans, dirs, recipe, out, more=extra))
    a = T3.parse(["--recipe", "long", "--family", "T", "--shape", "x", "--seed", "1", "--manifest", "m",
                  "--shards", "d", "--out", "o"])
    assert (a.steps, a.extras_set, a.ema_start, a.bn_batches, a.cal_batches, a.threads) == (60000, (), 1000, 32, 32, 5)
    assert a.name == "long_x_seed1" and a.shape_spec is None
    assert T3.parse(args3(mans, dirs, "base", out, more=["--extras", "clean,size"])).name == \
        "size+clean_t_tiny_seed1"
    # family, shape and registry
    with pytest.raises(SystemExit, match="is a family T shape"):
        T3.main(args3(mans, dirs, "base", tmp_path / "a", family="M"))
    with pytest.raises(SystemExit, match="not a registered textspot3 shape"):
        T3.main([x for x in args3(mans, dirs, "base", tmp_path / "b") if x != "--shape-json"
                 and not x.startswith("{")])
    for bad in ({**T_TINY, "chans": [8, 72]}, {**T_TINY, "stride": 4}, {**T_TINY, "block": "plain"},
                {**T_TINY, "head": "kroa"}, {**T_TINY, "dil": [1]}):
        with pytest.raises(SystemExit):
            T3.main(args3(mans, dirs, "base", tmp_path / "c", shape=bad))
    a = args3(mans, dirs, "base", tmp_path / "c", shape={**T_TINY, "name": "other"})
    a[a.index("--shape") + 1] = "t_tiny"
    with pytest.raises(SystemExit, match="names 'other'"):
        T3.main(a)
    assert not (tmp_path / "a").exists() and not (tmp_path / "c").exists()
    # a used directory
    used = tmp_path / "used"
    used.mkdir()
    (used / "log.jsonl").write_text("{}\n")
    with pytest.raises(SystemExit, match="not empty"):
        T3.main(args3(mans, dirs, "base", used))
    # the data rule: the protocol's directories, every manifest shard read
    with pytest.raises(SystemExit, match="more_<source>"):
        a = args3(mans, dirs, "base", tmp_path / "d")
        a.remove(str(dirs["more_realplus"]))
        T3.main(a)
    real = tmp_path / "realplus"
    real.mkdir()
    DATA.write_shard(real / "real_0000", [{"canvas": np.zeros((40, 60), np.uint8), "lines": [], "ignore": [],
                                           "src": "realplus", "kind": "psp", "vn": "v3", "shot": "v3_1"}])
    man_97 = tmp_path / "textdet_like.json"
    man_97.write_text(json.dumps({"kind": "textdet-shards", "shards": [
        {"dir": "realplus", "name": "real_0000", "npy_sha256": sha(real / "real_0000.npy"),
         "json_sha256": sha(real / "real_0000.json")}]}))
    a = args3([*mans, man_97], dirs, "base", tmp_path / "e")
    i = a.index("--shards")
    with pytest.raises(SystemExit, match="not the protocol's directories"):
        T3.main(a[:i + 1] + [str(real)] + a[i + 1:])
    with pytest.raises(SystemExit, match="are not read"):
        T3.main(a)
    T3.main(a + ["--any-shards", "--probe", "1"])
    assert json.loads((tmp_path / "e" / "run.json").read_text())["data_rule"].startswith("any shard directories")
    # a teacher that is not a textspot3 run, or of another scale
    t2run = tmp_path / "t2run"
    T2.main(args2(mans, dirs, "data", t2run, steps=3))
    with pytest.raises(SystemExit, match="not a textspot3 run"):
        T3.main(args3(mans, dirs, "kd", tmp_path / "k1", more=["--teacher", str(t2run), "--extras", "none"]))
    good = tmp_path / "good"
    T3.main(args3(mans, dirs, "base", good))
    info = json.loads((good / "run.json").read_text())
    (good / "run.json").write_text(json.dumps({**info, "scale": {**info["scale"], "half_above": 600}}))
    with pytest.raises(SystemExit, match="not a family T run at its scale"):
        T3.main(args3(mans, dirs, "kd", tmp_path / "k2", more=["--teacher", str(good), "--extras", "none"]))
    # a textspot2 pin without its skip lists; a bad line weight
    doc = json.loads(mans[1].read_text())
    del doc["real"]
    mans[1].write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="without real.measured.skip"):
        T3.main(args3(mans, dirs, "base", tmp_path / "k3"))
    with pytest.raises(ValueError):
        T3.clean_rule([0, 0, 50, 20, 0, "text", -0.5])


def test_family_m_waits_for_its_modules(tmp_path):
    """Until textspot3-runtime's train/textspot3_model.py and synth/textspot3_twin.py exist, an M run refuses with
    a message that names them; family T needs neither."""
    try:
        import textspot3_model  # noqa: F401
        import textspot3_twin  # noqa: F401
        pytest.skip("the M modules exist: the M tests run instead")
    except ImportError:
        pass
    mans, dirs = tiny_shards(tmp_path)
    m_shape = {"name": "m_tiny", "family": "M", "head": "ms_x2", "stride": 8}
    with pytest.raises(SystemExit, match="textspot3_model"):
        T3.main(args3(mans, dirs, "base", tmp_path / "m", shape=m_shape, family="M"))
    assert "textspot3_model" not in sys.modules


def test_a_speed_probe_writes_no_model_and_the_cli(tmp_path):
    mans, dirs = tiny_shards(tmp_path)
    run = tmp_path / "probe"
    a = args3(mans, dirs, "size", run, steps=30000, more=["--probe", "2"])
    p = subprocess.run([sys.executable, str(ROOT / "train" / "textspot3_train.py"), *a], capture_output=True,
                       text=True, timeout=900, cwd=ROOT)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-4000:]
    assert "TEXTSPOT3_PROBE_DONE" in p.stdout
    probe = json.loads((run / "probe.json").read_text())
    assert probe["steps_timed"] == 2 and probe["ms_per_step"] > 0 and probe["name"] == "size_t_tiny_seed1"
    assert probe["family"] == "T" and probe["shape"] == "t_tiny" and probe["extras"] == ["size"]
    info = json.loads((run / "run.json").read_text())
    assert info["scale"]["half_above"] == 480 and info["steps"] == 30000 and info["qat_start"] == 24000
    assert "train/textspot3_train.py" in info["code_sha256"]
    assert not any((run / f).exists() for f in ("final.pt", "final.vtd", "final.vtsp", "done.json"))


def test_the_files_the_protocols_pin_are_unchanged():
    pins = dict(PROTOCOL["sources_sha256"])
    for amend in PROTOCOL.get("amendments", []):
        pins.update(amend.get("sources_sha256", {}))
    for f in ("train/textspot2_train.py", "train/textspot2_export.py", "train/textspot2_shapes.py",
              "train/textspot_train.py", "train/textdet_train.py", "train/textdet_model.py",
              "train/textdet_export.py", "synth/textspot_twin.py", "synth/textspot2_twin.py", "synth/textdet_twin.py",
              "data/manifests/textspot_shards_v1.json", "data/manifests/textspot2_shards_v1.json"):
        assert sha(ROOT / f) == pins[f], f


# ---------------------------------------------------------------- the sweep's plan, check and summary

def fake_protocol(tmp_path, t_names=("t_a", "t_b"), points=True):
    shapes = {n: {**{k: v for k, v in T_TINY.items() if k != "name"}, "family": "T"} for n in t_names}
    for p, n in zip(PL.POINTS, ("m_s", "m_m", "m_l")):
        shapes[n] = {"family": "M", "head": "ms_x2", "stride": 8, **({"point": p} if points else {})}
    shapes["grammar"] = "a note"
    doc = {**PROTOCOL, "shapes": shapes}
    p = tmp_path / "protocol.json"
    p.write_text(json.dumps(doc))
    return p


def test_the_stage1_plans(tmp_path):
    reg = PL.shapes(fake_protocol(tmp_path))
    assert sorted(reg) == ["m_l", "m_m", "m_s", "t_a", "t_b"]
    t = PL.stage1("T", reg)
    assert [r["run"] for r in t] == [f"base_{s}_seed{n}" for s in ("t_a", "t_b") for n in (1, 2, 3)]
    assert all(r["family"] == "T" and r["steps"] == 30000 and r["extras"] is None for r in t)
    m = PL.stage1("M", reg)
    by = {}
    for r in m:
        by.setdefault((r["recipe"], r["shape"]), []).append(r["seed"])
    assert by == {("base", "m_s"): [1, 2, 3], ("base", "m_m"): [1, 2, 3], ("base", "m_l"): [1, 2, 3],
                  ("size", "m_m"): [1, 2, 3], ("clean", "m_m"): [1, 2, 3], ("long", "m_m"): [1, 2, 3]}
    assert {r["run"] for r in m if r["recipe"] == "long"} == {f"long_m_m_seed{n}" for n in (1, 2, 3)}
    assert all(r["steps"] == (60000 if r["recipe"] == "long" else 30000) for r in m)
    assert {r["run"] for r in m if r["recipe"] == "size"} == {f"size_m_m_seed{n}" for n in (1, 2, 3)}
    # longest first by the estimates; the hang guard
    est = {r["run"]: 1000.0 * (i + 1) for i, r in enumerate(m)}
    o = PL.order(m, reg, est)
    assert [r["est_s"] for r in o] == sorted((r["est_s"] for r in o), reverse=True)
    assert all(r["timeout_s"] == max({30000: 43200, 60000: 86400}[r["steps"]],
                                     3600 * math.ceil(4 * r["est_s"] / 3600)) for r in o)
    o = PL.order(t, reg)                                         # no estimates: steps x MACs, the floors
    assert all(r["est_s"] is None and r["timeout_s"] == 43200 for r in o)
    # M points from --m-points, refusals
    reg2 = PL.shapes(fake_protocol(tmp_path, points=False))
    with pytest.raises(SystemExit, match="M points"):
        PL.stage1("M", reg2)
    assert len(PL.stage1("M", reg2, PL.parse_points("S=m_s,M=m_m,L=m_l"))) == 18
    with pytest.raises(SystemExit):
        PL.parse_points("S=m_s,X=m_m")
    with pytest.raises(SystemExit, match="M points"):
        PL.stage1("M", reg2, {"S": "m_s", "M": "m_s", "L": "m_l"})


def test_the_plan_command_lines(tmp_path):
    herd, repo = tmp_path / "herd", tmp_path / "repo"
    t2 = herd / "shared" / "artifacts" / "textspot2" / "shards"
    p2 = herd / "shared" / "private" / "textspot2" / "shards"
    for d in (t2 / "lab" / "lab_realplus", t2 / "lab" / "lab_teacher", p2 / "lab" / "lab_owner",
              t2 / "more" / "more_realplus", p2 / "more" / "more_owner"):
        d.mkdir(parents=True)
    (repo / "data" / "manifests").mkdir(parents=True)
    (repo / PL.T2_MANIFEST).write_text(json.dumps({"kind": "textspot2-shards", "shards": [
        {"dir": d, "name": f"{d}_0000"} for d in ("lab_realplus", "lab_teacher", "lab_owner", "more_realplus",
                                                  "more_owner")]}))
    p = PL.paths(herd)
    r = PL.spec("size", "M", "m_m", 2)
    a = PL.argv(r, "/o", repo, herd)
    i, j = a.index("--shards"), a.index("--out")
    assert a[:10] == ["train/textspot3_train.py", "--recipe", "size", "--family", "M", "--shape", "m_m", "--seed",
                      "2", "--manifest"] and a[10:12] == [PL.SYNTH_MANIFEST, PL.T2_MANIFEST]
    assert a[i + 1:j] == [str(p["TS"] / "shards" / "synth_frames"), str(p2 / "lab" / "lab_owner"),
                          str(t2 / "lab" / "lab_realplus"), str(t2 / "lab" / "lab_teacher"),
                          str(p2 / "more" / "more_owner"), str(t2 / "more" / "more_realplus")]
    assert a[j:] == ["--out", "/o/size_m_m_seed2", "--threads", "5"]
    kd = PL.spec("kd", "T", "t_a", 1, "size", "$HERD/shared/artifacts/textspot3/runs/base_m_l_seed2")
    a = PL.argv(kd, "/o", repo, herd)
    assert a[-4:] == ["--extras", "size", "--teacher",
                      str(herd / "shared" / "artifacts" / "textspot3" / "runs" / "base_m_l_seed2")]
    (t2 / "more" / "lab_teacher").mkdir()                                  # a basename found twice
    with pytest.raises(SystemExit, match="found 2 times"):
        PL.argv(r, "/o", repo, herd)


def fake_teacher(runs_dir, name, payload=b"a teacher checkpoint"):
    d = Path(runs_dir) / name
    d.mkdir(parents=True)
    (d / "final.pt").write_bytes(payload)
    (d / "done.json").write_text(json.dumps({"pt_sha256": hashlib.sha256(payload).hexdigest()}))
    return hashlib.sha256(payload).hexdigest()


CHOICES = {"size_kept": True, "clean_kept": False, "kd_teacher_run": "base_m_l_seed2",
           "kd_students": ["t_a", "m_s"]}


def test_stage2_from_choices(tmp_path):
    proto = fake_protocol(tmp_path)
    reg = PL.shapes(proto)
    runs_dir = tmp_path / "runs"
    sha_t = fake_teacher(runs_dir, "base_m_l_seed2")
    runs = PL.stage2_from_choices(CHOICES, reg, runs_dir)
    names = [r["run"] for r in runs]
    assert names == [f"kd_{s}_seed{n}" for s in ("t_a", "m_s") for n in (1, 2, 3)] + \
        [f"size_m_l_seed{n}" for n in (1, 2, 3)]
    kd = [r for r in runs if r["recipe"] == "kd"]
    assert all(r["extras"] == "size" and r["teacher_run"] == "base_m_l_seed2" and r["teacher_pt_sha256"] == sha_t
               and r["teacher"] == "$HERD/shared/artifacts/textspot3/runs/base_m_l_seed2" for r in kd)
    assert [r["family"] for r in kd] == ["T"] * 3 + ["M"] * 3
    both = PL.stage2_from_choices({**CHOICES, "clean_kept": True}, reg, runs_dir)
    assert [r["run"] for r in both][-3:] == [f"size+clean_m_l_seed{n}" for n in (1, 2, 3)]
    assert {r["extras"] for r in both} == {"size,clean"} and both[-1]["recipe"] == "base"
    clean = PL.stage2_from_choices({**CHOICES, "size_kept": False, "clean_kept": True}, reg, runs_dir)
    assert [r["run"] for r in clean][-3:] == [f"clean_m_l_seed{n}" for n in (1, 2, 3)]
    none = PL.stage2_from_choices({**CHOICES, "size_kept": False}, reg, runs_dir)
    assert len(none) == 6 and {r["extras"] for r in none} == {"none"}
    for change, msg in (({"kd_teacher_run": "base_t_a_seed1"}, "not a stage-1 run of an M shape"),
                        ({"kd_teacher_run": "base_m_l_seed4"}, "not a stage-1 run"),
                        ({"kd_students": ["t_a"]}, "two distinct"), ({"kd_students": ["t_a", "t_a"]}, "two distinct"),
                        ({"kd_students": ["t_a", "m_l"]}, "is a kd student"),
                        ({"kd_students": ["t_a", "nope"]}, "two distinct"), ({"size_kept": 1}, "must be a bool"),
                        ({"kd_teacher_run": "base_m_m_seed1"}, "has not finished")):
        with pytest.raises(SystemExit, match=msg):
            PL.stage2_from_choices({**CHOICES, **change}, reg, runs_dir)
    with pytest.raises(SystemExit, match="missing"):
        PL.stage2_from_choices({k: v for k, v in CHOICES.items() if k != "size_kept"}, reg, runs_dir)
    (runs_dir / "base_m_l_seed2" / "done.json").write_text(json.dumps({"pt_sha256": "0" * 64}))
    with pytest.raises(SystemExit, match="pt_sha256 is not final.pt's"):
        PL.stage2_from_choices(CHOICES, reg, runs_dir)
    # the CLI: the plan file, its order, the protocol's sha256 and the command lines
    (runs_dir / "base_m_l_seed2" / "done.json").write_text(json.dumps({"pt_sha256": sha_t}))
    choices = tmp_path / "choices.json"
    choices.write_text(json.dumps(CHOICES))
    plan = tmp_path / "stage2.jsonl"
    est = tmp_path / "est.json"
    est.write_text(json.dumps({"estimate": {"est_run_s": {n: 3600.0 + i for i, n in enumerate(names)}}}))
    PL.main(["stage2", "--choices", str(choices), "--runs-dir", str(runs_dir), "--protocol", str(proto), "--est",
             str(est), "--out", str(plan)])
    got = PL.load(plan)
    assert len(got) == 9 and all(r["protocol_sha256"] == sha(proto) for r in got)
    assert [r["est_s"] for r in got] == sorted((r["est_s"] for r in got), reverse=True)
    assert PL.parse_run("size+clean_m_l_seed3") == ("size+clean", "m_l", 3)
    assert PL.parse_run("kd_t_a_seed1") == ("kd", "t_a", 1)


def test_the_plan_cli_and_check_run_and_runs_summary(tmp_path):
    proto = fake_protocol(tmp_path, t_names=("t_tiny",))
    plan = tmp_path / "plan.jsonl"
    est = tmp_path / "est.json"
    est.write_text(json.dumps({"estimate": {"est_run_s": {f"base_t_tiny_seed{n}": 100.0 * n for n in (1, 2, 3)}}}))
    PL.main(["stage1", "--family", "T", "--protocol", str(proto), "--est", str(est), "--out", str(plan)])
    p = subprocess.run([sys.executable, str(REPRO / "textspot3_plan.py"), "list", str(plan)], capture_output=True,
                       text=True, timeout=120, cwd=ROOT)
    assert p.stdout.split() == ["base_t_tiny_seed3", "base_t_tiny_seed2", "base_t_tiny_seed1"]
    for cmd, want in (("timeout", "43200"), ("family", "T")):
        p = subprocess.run([sys.executable, str(REPRO / "textspot3_plan.py"), cmd, str(plan), "base_t_tiny_seed1"],
                           capture_output=True, text=True, timeout=120, cwd=ROOT)
        assert p.stdout.strip() == want
    mans, dirs = tiny_shards(tmp_path)
    runs = tmp_path / "runs"
    run = runs / "base_t_tiny_seed1"
    T3.main(args3(mans, dirs, "base", run))
    fin = runs / "finished.txt"
    p = subprocess.run([sys.executable, str(REPRO / "check_run.py"), str(run), "--finished", str(fin)],
                       capture_output=True, text=True, timeout=600, cwd=ROOT)
    assert p.returncode == 0 and "CHECK_OK base_t_tiny_seed1 " in p.stdout, p.stdout + p.stderr
    done = json.loads((run / "done.json").read_text())
    assert fin.read_text().split() == ["base_t_tiny_seed1", done["vtsp_sha256"], str(done["vtsp_bytes"]), "3",
                                       str(done["wall_seconds"])]
    subprocess.run([sys.executable, str(REPRO / "check_run.py"), str(run), "--finished", str(fin)], check=True,
                   capture_output=True, timeout=600, cwd=ROOT)
    assert len(fin.read_text().splitlines()) == 1                           # recorded once
    other = runs / "base_t_tiny_seed2"
    run.rename(other)
    p = subprocess.run([sys.executable, str(REPRO / "check_run.py"), str(other)], capture_output=True, text=True,
                       timeout=600, cwd=ROOT)
    assert p.returncode == 1 and "BAD run.json names base_t_tiny_seed1" in p.stdout
    other.rename(run)
    (run / "final.vtsp").write_bytes(TW2.pack((run / "final.vtd").read_bytes(), 600, 1, version=2))
    info, done2, bad, notes, boxes = CR.check(run)
    assert any("done.json's vtsp" in b for b in bad) and any("pack(final.vtd, 480" in b for b in bad)
    (run / "final.vtsp").write_bytes(TW2.pack((run / "final.vtd").read_bytes(), 480, 1, version=2))
    assert CR.check(run)[2] == []
    doc = RS.summarise(runs, plan)
    assert (doc["finished"], doc["of"]) == (1, 3) and {r["state"] for r in doc["runs"]} == {"finished", "missing"}
    row = next(r for r in doc["runs"] if r["state"] == "finished")
    c = row["curve"]
    assert c["step_final"] == 3 and c["step_first_qat"] == 3 and c["step_last_float"] == 2
    assert c["ema_swap_step"] == 2 and c["qat_average_updates"] == 2 and row["vtsp_version"] == 2
    assert "| base_t_tiny_seed1 |" in RS.markdown(doc) and doc["consistent"]["git"]
    a, b = "a" * 40, "b" * 40
    (runs / "sweep.log").write_text(
        f"2026-10-21T10:00:00Z sweep start: stage 1 T, MAX_PAR 8, head {a}, plan /x/{plan.name} ({sha(plan)}), runs r\n"
        "2026-10-21T12:00:00Z SWEEP_DONE stage 1 T: 1/3 done, 0 running, 0 crashes (x)\n"
        f"2026-10-21T13:00:00Z sweep start: stage 2, MAX_PAR 8, head {b}, plan /y/other.jsonl ({'c' * 64}), runs r\n")
    rec = tmp_path / "record.json"
    doc = RS.record(rec, "stage1_T", runs, plan, herd=tmp_path)
    sw = doc["sweeps"]["stage1_T"]
    assert (sw["out"], sw["commit"], sw["max_par"], sw["started"], sw["sweep_done"]) == \
        ("$HERD/runs", a, 8, "2026-10-21T10:00:00Z", "2026-10-21T12:00:00Z")
    assert sw["plan"] == "eval/reproducibility/textspot3-20261021/plan.jsonl" and "/x/" not in rec.read_text()


def test_the_sweep_script_parses():
    p = subprocess.run(["bash", "-n", str(REPRO / "sweep.sh")], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
