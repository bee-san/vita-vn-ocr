"""Tests of the family B trainer of the full-frame text spotter (train/textspot_train.py, issue #104): the halving
of boxes, the window cut, the targets in a window, the pool's checks, and a 3-step run (and a 2-step speed probe) on
two tiny synthetic canvases written in #97's shard format. No screenshots, fonts or models; no network.
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "train", ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textdet_data as DATA  # noqa: E402
import textdet_model as TM  # noqa: E402
import textdet_train as TT  # noqa: E402
import textdet_twin as TD  # noqa: E402
import textspot_train as ST  # noqa: E402
import textspot_twin as TS  # noqa: E402

FAMILY_B = json.loads((ROOT / "eval" / "textspot_protocol.json").read_text())["candidates"]["family_b"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- the protocol's family B

def test_scales_and_shapes_are_the_protocols():
    assert ST.HALF_ABOVE == {k: v["half_above"] for k, v in FAMILY_B["scales"].items()}
    assert sorted(FAMILY_B["shapes"]) == sorted(ST.HALF_ABOVE)
    runs = [f"{sc}_{sh}_seed{n}" for sc, shapes in FAMILY_B["shapes"].items() for sh in shapes
            for n in FAMILY_B["seeds"]]
    assert len(runs) == len(set(runs)) == 22
    for shapes in FAMILY_B["shapes"].values():
        for name in shapes:
            c = FAMILY_B["shape_costs"][name]
            assert TM.cost(TM.SHAPES[name], 960, 544)["macs"] == c["macs"]["960x544"]
            assert TM.cost(TM.SHAPES[name], 480, 272)["macs"] == c["macs"]["480x272"]
            assert TM.cost(TM.SHAPES[name], 960, 544)["weights_int8"] == c["weights_int8"]
            assert TM.receptive_field(TM.SHAPES[name]) == c["receptive_field"]
    assert (ST.WIN_H, ST.WIN_W, ST.BATCH, ST.MIN_ACROSS) == (256, 512, 16, 4)
    real = {k: v for k, v in ST.MIX.items() if k != "synth"}
    assert ST.MIX["synth"] == pytest.approx(0.5)
    assert {k: v / sum(real.values()) for k, v in real.items()} == pytest.approx(
        {"realplus": 0.35, "teacher": 0.35, "owner": 0.25, "devnative": 0.05})


# ---------------------------------------------------------------- the halving of boxes

def test_half_box_is_exactly_where_the_halved_ink_lies():
    rng = np.random.default_rng(0)
    for _ in range(400):
        H, W = int(rng.integers(2, 70)), int(rng.integers(2, 90))
        x0, x1 = sorted(int(v) for v in rng.choice(W + 1, 2, replace=False))
        y0, y1 = sorted(int(v) for v in rng.choice(H + 1, 2, replace=False))
        g = np.zeros((H, W), np.uint8)
        g[y0:y1, x0:x1] = 255
        h = TS.halve(g)
        assert h.shape == ((H + 1) // 2, (W + 1) // 2)
        ys, xs = np.nonzero(h)
        assert ST.half_box([x0, y0, x1, y1]) == [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def test_half_box_floor_starts_and_ceil_ends():
    assert ST.half_box([0, 0, 1, 1]) == [0, 0, 1, 1]
    assert ST.half_box([3, 4, 9, 10]) == [1, 2, 5, 5]
    assert ST.half_box([10, 11, 17, 18]) == [5, 5, 9, 9]
    assert ST.half_box([959, 543, 960, 544]) == [479, 271, 480, 272]


def test_lines_of_every_class_are_positives_and_thin_ones_become_ignore_areas():
    lines = [[100, 11, 300, 18, 0, "dialogue"],       # 7 px tall -> [5, 9): 4 px, kept at h2
             [100, 10, 300, 16, 0, "name"],           # 6 px -> [5, 8): 3 px, ignore at h2, kept at f1
             [100, 40, 300, 43, 0, "ui"],             # 3 px: ignore at both scales
             [500, 20, 507, 200, 1, "scene"],         # vertical, 7 px wide -> [250, 254): kept
             [520, 20, 526, 200, 1, "choice"],        # vertical, 6 px wide -> [260, 263): ignore at h2
             [600, 300, 610, 340, 1, "dialogue"]]     # vertical, 10 px wide
    ignore = [[5, 5, 9, 9]]
    pos, ign = ST.at_scale(lines, ignore, 2)
    assert pos == [[50, 5, 150, 9, 0], [250, 10, 254, 100, 1], [300, 150, 305, 170, 1]]
    assert ign == [[2, 2, 5, 5], [50, 5, 150, 8], [50, 20, 150, 22], [260, 10, 263, 100]]
    pos, ign = ST.at_scale(lines, ignore, 1)
    assert pos == [[100, 11, 300, 18, 0], [100, 10, 300, 16, 0], [500, 20, 507, 200, 1], [520, 20, 526, 200, 1],
                   [600, 300, 610, 340, 1]]
    assert ign == [[5, 5, 9, 9], [100, 40, 300, 43]]


# ---------------------------------------------------------------- the window cut

def test_a_canvas_that_fits_is_used_whole():
    g = np.random.default_rng(1).integers(0, 256, (200, 400)).astype(np.uint8)
    for scale in ("h2", "f1"):
        img, ox, oy, w, h, f = ST.window(np.random.default_rng(2), g, scale)
        assert (ox, oy, w, h, f) == (0, 0, 400, 200, 1)
        assert np.array_equal(img, g)
    g = np.random.default_rng(1).integers(0, 256, (300, 1000)).astype(np.uint8)    # halved: 150 x 500, fits
    img, ox, oy, w, h, f = ST.window(np.random.default_rng(2), g, "h2")
    assert (ox, oy, w, h, f) == (0, 0, 500, 150, 2)
    assert np.array_equal(img, TS.halve(g))


class EdgeRng:
    """integers(lo, hi) always gives lo (first=True) or hi - 1: the first and the last window."""

    def __init__(self, first):
        self.first = first

    def integers(self, lo, hi):
        return lo if self.first else hi - 1


@pytest.mark.parametrize("H, W", [(544, 960), (545, 961), (544, 725), (543, 963), (720, 1280)])
def test_a_halved_window_is_the_window_of_the_halved_canvas(H, W):
    g = np.random.default_rng(H * W).integers(0, 256, (H, W)).astype(np.uint8)
    full = TS.halve(g)
    hs, ws = min(256, full.shape[0]), min(512, full.shape[1])
    for rng in (EdgeRng(True), EdgeRng(False), *[np.random.default_rng(3)] * 200):
        img, ox, oy, w, h, f = ST.window(rng, g, "h2")
        assert f == 2 and (h, w) == (hs, ws)
        assert 0 <= oy <= full.shape[0] - h and 0 <= ox <= full.shape[1] - w
        assert np.array_equal(img, full[oy:oy + h, ox:ox + w])
        if isinstance(rng, EdgeRng):
            assert (ox, oy) == ((0, 0) if rng.first else (full.shape[1] - ws, full.shape[0] - hs))


def test_f1_and_narrow_canvases_are_not_halved():
    g = np.random.default_rng(4).integers(0, 256, (544, 960)).astype(np.uint8)
    img, ox, oy, w, h, f = ST.window(np.random.default_rng(5), g, "f1")
    assert f == 1 and (h, w) == (256, 512) and np.array_equal(img, g[oy:oy + 256, ox:ox + 512])
    p = np.random.default_rng(6).integers(0, 256, (272, 480)).astype(np.uint8)
    for scale in ("h2", "f1"):
        img, ox, oy, w, h, f = ST.window(np.random.default_rng(7), p, scale)
        assert f == 1 and (ox, w, h) == (0, 480, 256) and 0 <= oy <= 16
        assert np.array_equal(img, p[oy:oy + 256])
    assert ST.factor("h2", 600) == 1 and ST.factor("h2", 601) == 2 and ST.factor("f1", 4000) == 1


def test_window_offsets_are_uniform():
    g = np.zeros((544, 960), np.uint8)
    rng = np.random.default_rng(8)
    xs, ys = [], []
    for _ in range(20000):
        _, ox, oy, w, h, _ = ST.window(rng, g, "f1")
        xs.append(ox)
        ys.append(oy)
    assert (min(xs), max(xs), min(ys), max(ys)) == (0, 448, 0, 288)
    for v, n in ((np.array(xs), 449), (np.array(ys), 289)):
        counts = np.histogram(v, bins=4, range=(0, n))[0]
        assert np.all(np.abs(counts / len(v) - 0.25) < 0.02), counts


class StubPool:
    def __init__(self, canvas, smp):
        self.keys, self.p, self.items, self._canvas = ["synth"], np.array([1.0]), {"synth": [(0, smp)]}, canvas

    def canvas(self, d, smp):
        return self._canvas


def test_the_targets_of_a_halved_window():
    rng = np.random.default_rng(9)
    g = rng.integers(0, 60, (545, 961)).astype(np.uint8)
    g[401:431, 100:500] = 240
    smp = {"lines": [[100, 401, 500, 431, 0, "dialogue"], [700, 10, 703, 400, 1, "ui"]],
           "ignore": [[800, 300, 900, 340]], "src": "synth:frame"}
    x, vhw, K, O, A1, A2, vm, km = ST.make_batch(StubPool(g, smp), np.random.default_rng(10), 4, "h2")  # noqa: E741
    assert (TT.WIN_H, TT.WIN_W) == (192, 512)                     # #97's constants are restored
    full = TS.halve(g)                                            # 273 x 481
    assert x.shape == (16, 1, 256, 512) and K.shape == (16, 64, 128)
    offsets = set()
    for b in range(16):
        assert vhw[b].tolist() == [256, 481]
        win = x[b, 0].numpy()
        oy = next(o for o in range(18) if np.array_equal(win[:, :481], full[o:o + 256]))
        offsets.add(oy)
        assert np.all(win[:, 481:] == 128)
        # the line at the halved scale: [50, 200, 250, 216), 16 px across: kernel rows within max(16/4, 4/2) of
        # its centre 208, columns 50 // 4 .. 249 // 4 (#97's round-1 rule)
        c = 208 - oy
        rows = [i for i in range(64) if abs(4 * i + 1.5 - c) <= 4]
        want = np.zeros((64, 128), bool)
        want[np.ix_(rows, range(12, 63))] = True
        assert np.array_equal(K[b].numpy() > 0, want)
        assert np.array_equal(km[b].numpy(), want)
        i0 = rows[0]
        assert A1[b, i0, 20] == pytest.approx(4 * i0 + 1.5 - (200 - oy))
        assert A2[b, i0, 20] == pytest.approx(216 - oy - (4 * i0 + 1.5))
        assert float(O[b].sum()) == 0
        valid = vm[b].numpy()
        assert not valid[:, 121:].any()
        assert not valid[(150 - oy) // 4, 400 // 4]               # the halved ignore area [400, 150, 450, 170)
        assert not valid[(100 - oy) // 4, 350 // 4]               # the 3-px vertical line became one
        assert valid[(50 - oy) // 4, 30]
    assert len(offsets) > 1


# ---------------------------------------------------------------- shards, the pool, a 3-step run

def tiny_shards(tmp_path):
    """Two tiny canvases in #97's shard format: a synthetic one wider than 600 px (halved at h2) and a real one."""
    rng = np.random.default_rng(11)
    d = tmp_path / "frames"
    d.mkdir()
    a = rng.integers(0, 60, (300, 700)).astype(np.uint8)
    a[40:58, 100:400] = 230
    a[200:226, 50:650] = 230
    a[20:280, 660:680] = 230
    synth = {"canvas": a, "lines": [[100, 40, 400, 58, 0, "dialogue"], [50, 200, 650, 226, 0, "name"],
                                    [660, 20, 680, 280, 1, "dialogue"], [10, 100, 60, 106, 0, "ui"]],
             "ignore": [[300, 100, 340, 110]], "src": "synth:frame", "kind": "vita", "vn": None, "shot": None,
             "meta": {}}
    b = rng.integers(0, 60, (120, 200)).astype(np.uint8)
    b[50:66, 20:180] = 230
    real = {"canvas": b, "lines": [[20, 50, 180, 66, 0, "text"]], "ignore": [], "src": "realplus", "kind": "psp",
            "vn": "v1", "shot": "v1_1", "meta": {}}
    DATA.write_shard(d / "frames_0000", [synth, real])
    man = {"kind": "test-shards", "shards": [{"dir": "frames", "name": "frames_0000",
                                              "npy_sha256": sha(d / "frames_0000.npy"),
                                              "json_sha256": sha(d / "frames_0000.json"), "samples": 2}]}
    (tmp_path / "man.json").write_text(json.dumps(man))
    return tmp_path / "man.json", d


def test_the_pool_reads_the_given_directories_and_checks_the_pins(tmp_path):
    man, d = tiny_shards(tmp_path)
    pool = ST.FramePool([man], [d], verify=True)
    assert {k: len(v) for k, v in pool.items.items()} == {"synth": 1, "realplus": 1, "teacher": 0, "owner": 0,
                                                          "devnative": 0}
    assert pool.keys == ["synth", "realplus"] and pool.p == pytest.approx([0.5 / 0.675, 0.175 / 0.675])
    assert pool.canvas(*pool.items["realplus"][0]).shape == (120, 200)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(SystemExit, match="no manifest shard"):
        ST.FramePool([man], [d, other])
    doc = json.loads(man.read_text())
    doc["shards"].append({**doc["shards"][0], "dir": "elsewhere"})
    man2 = tmp_path / "man2.json"
    man2.write_text(json.dumps(doc))
    assert ST.FramePool([man2], [d]).skipped == {f"{man2}:elsewhere": 1}
    doc["shards"][0]["json_sha256"] = "0" * 64
    man2.write_text(json.dumps(doc))
    with pytest.raises(SystemExit, match="sha256"):
        ST.FramePool([man2], [d], verify=True)


def test_a_run_refuses_a_used_directory_and_a_missing_source(tmp_path):
    man, d = tiny_shards(tmp_path)
    used = tmp_path / "used"
    used.mkdir()
    (used / "log.jsonl").write_text("{}\n")
    base = ["--shape", "s8_c8", "--scale", "h2", "--seed", "1", "--manifest", str(man), "--shards", str(d)]
    with pytest.raises(SystemExit, match="not empty"):
        ST.main(base + ["--out", str(used)])
    with pytest.raises(SystemExit, match="no samples of"):
        ST.main(base + ["--out", str(tmp_path / "fresh")])
    assert not (tmp_path / "fresh").exists()


def procedural_frame(rng, H, W):
    g = np.full((H, W), int(rng.integers(0, 256)), np.int32)
    for _ in range(12):
        y0, x0 = int(rng.integers(0, H)), int(rng.integers(0, W))
        sub = g[y0:y0 + int(rng.integers(4, 40)), x0:x0 + int(rng.integers(8, 300))]
        sub[:] = int(rng.integers(0, 256))
        sub[:, ::int(rng.integers(2, 6))] = int(rng.integers(0, 256))
    return np.clip(g + rng.integers(-12, 13, g.shape), 0, 255).astype(np.uint8)


def test_a_three_step_run(tmp_path):
    man, d = tiny_shards(tmp_path)
    run = tmp_path / "run"
    p = subprocess.run([sys.executable, str(ROOT / "train" / "textspot_train.py"), "--shape", "s8_c8",
                        "--scale", "h2", "--seed", "1", "--manifest", str(man), "--shards", str(d), "--out", str(run),
                        "--steps", "3", "--threads", "1", "--log-every", "1", "--cal-batches", "2",
                        "--allow-missing-sources"], capture_output=True, text=True, timeout=900, cwd=ROOT)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-4000:]
    assert "TEXTSPOT_TRAIN_DONE" in p.stdout
    info = json.loads((run / "run.json").read_text())
    assert (info["name"], info["scale"], info["half_above"], info["seed"], info["steps"], info["qat_start"]) == \
        ("h2_s8_c8_seed1", "h2", 600, 1, 3, 2)
    assert info["recipe"]["window"] == [256, 512] and info["shape"] == TM.SHAPES["s8_c8"]
    assert info["manifests_sha256"] == {str(man): sha(man)}
    assert info["pool"]["by_source"]["synth"]["canvases"] == 1 and info["pool"]["by_source"]["realplus"]["lines"] == 1
    for f in ("train/textspot_train.py", "train/textdet_train.py", "train/textdet_model.py",
              "train/textdet_export.py", "synth/textdet_twin.py", "synth/textspot_twin.py"):
        assert info["code_sha256"][f] == sha(ROOT / f)
    assert all(k.startswith(("train/", "synth/")) for k in info["code_sha256"])
    log = [json.loads(x) for x in (run / "log.jsonl").read_text().splitlines()]
    assert [r["step"] for r in log if "lr" in r] == [1, 2, 3] and any(r.get("qat") for r in log)
    assert all(np.isfinite(r[k]) for r in log if "lr" in r for k in ("k_bce", "dice", "o", "a"))
    done = json.loads((run / "done.json").read_text())
    vtd, vtsp = (run / "final.vtd").read_bytes(), (run / "final.vtsp").read_bytes()
    assert done["steps"] == 3 and done["vtd_sha256"] == sha(run / "final.vtd")
    assert done["vtsp_sha256"] == sha(run / "final.vtsp") and done["vtsp_bytes"] == len(vtsp) == len(vtd) + 32
    assert done["pt_sha256"] == sha(run / "final.pt")
    assert TD.Model(vtd).name == "s8_c8"
    m = TS.Model(vtsp)
    assert (m.half_above, m.conf_min, m.stride, m.name) == (600, 1, 8, "s8_c8")
    assert TS.pack(vtd, 600, 1) == vtsp
    rng = np.random.default_rng(12)
    for H, W in ((544, 960), (272, 480)):
        r = TS.spot(m, procedural_frame(rng, H, W))
        assert len(r.boxes) <= TS.BOX_MAX
        assert all(0 <= b.x and 0 <= b.y and b.x + b.w <= W and b.y + b.h <= H for b in r.boxes)


def test_a_speed_probe_writes_no_model(tmp_path):
    man, d = tiny_shards(tmp_path)
    run = tmp_path / "probe"
    p = subprocess.run([sys.executable, str(ROOT / "train" / "textspot_train.py"), "--shape", "s4_c6",
                        "--scale", "f1", "--seed", "2", "--manifest", str(man), "--shards", str(d), "--out", str(run),
                        "--probe", "2", "--threads", "1", "--allow-missing-sources"],
                       capture_output=True, text=True, timeout=900, cwd=ROOT)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-4000:]
    probe = json.loads((run / "probe.json").read_text())
    assert probe["steps_timed"] == 2 and probe["ms_per_step"] > 0
    assert json.loads((run / "run.json").read_text())["half_above"] == 0
    assert not any((run / f).exists() for f in ("final.pt", "final.vtd", "final.vtsp", "done.json"))
