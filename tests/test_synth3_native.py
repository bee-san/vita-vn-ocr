"""synth3 version 4 (docs/synth3-native.md): native console frames. The region is placed in a whole
960x544 / 480x272 frame, the console's scaling and VNDB's JPEG act on the frame, and the window is cut
back out with its masks and layout; version 3 renders are unchanged when CFG["frames"] is None. CI
fixtures only (the pinned OFL font, the fixture corpus, no screenshots)."""
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "synth"))
sys.path.insert(0, str(ROOT / "tests"))

import native_calib as C  # noqa: E402
import synth3  # noqa: E402
from test_synth3 import assets  # noqa: E402,F401  (module fixture)

PATCH = ROOT / "data" / "manifests" / "synth3_v4_native_cfg.json"


@pytest.fixture
def v4():
    """CFG with the version-4 patch applied, restored afterwards."""
    saved = copy.deepcopy(synth3.CFG)
    synth3.apply_cfg("@" + str(PATCH))
    try:
        yield synth3.CFG
    finally:
        synth3.CFG.clear()
        synth3.CFG.update(saved)


def test_version_3_has_no_frames_and_the_patch_is_valid(v4):
    assert v4["version"] == 4 and v4["frames"] is not None and v4["platform"]["scaled"] == 0.0
    F = v4["frames"]
    for plat in ("vita", "psp", "psp2x"):
        assert abs(sum(F["capture"][plat].values()) - 1.0) < 1e-9
        assert abs(sum(F["port"][plat].values()) - 1.0) < 1e-9
        assert set(F["vndb_420"][plat]) == set(F["capture"][plat]) - {"clean"}
    assert abs(sum(v4["length_target"]) - 1.0) < 1e-3 and len(v4["length_target"]) == len(v4["length_edges"]) - 1
    for k in ("vita", "psp", "bare"):
        assert 0 <= F["effects"][k]["stroke"] <= 1 and 0 <= F["effects"][k]["shadow"] <= 1


def test_version_3_cfg_is_untouched():
    assert synth3.CFG["version"] == 3 and synth3.CFG["frames"] is None


def region_fixture(h=40, w=120):
    img = np.full((h, w, 3), 50.0, np.float32)
    ink = np.zeros((h, w), np.float32)
    ink[12:28, 10:110] = 1.0
    img[ink > 0] = 230.0
    layout = [{"text": "あいう", "top": 12.0, "fs": 16.0, "x0": 10.0, "x1": 110.0, "xs": [10.0, 50.0, 90.0],
               "kind": "dialogue"}]
    return img, ink, ink.copy(), np.zeros_like(ink), layout


def only(F, **kw):
    """The patch's frames block with every random choice pinned."""
    F = copy.deepcopy(F)
    for plat in ("vita", "psp", "psp2x"):
        F["port"][plat] = {kw.get("port", "native"): 1.0}
        F["capture"][plat] = {kw.get("chain", "clean"): 1.0}
    return F


def test_native_clean_frame_returns_the_region_unchanged(v4, monkeypatch):
    v4["frames"] = only(v4["frames"])
    monkeypatch.setitem(v4, "blur", 0.0)
    monkeypatch.setitem(v4, "noise", 0.0)
    monkeypatch.setitem(v4, "dither565", 0.0)
    synth3.R = type("R", (), {"bg": {}})()
    img, ink, core, ruby, layout = region_fixture()
    out, ink_f, core_f, ruby_f, cap, meta, sy = synth3.frame_capture(np.random.default_rng(0), img, ink, core, ruby,
                                                                     layout, "vita", "adv")
    assert sy == 1.0 and out.shape == img.shape and np.array_equal(out, np.clip(img + 0.5, 0, 255).astype(np.uint8))
    assert np.array_equal(ink_f, ink) and meta["port"] == "native" and meta["frame"] == [960, 544]
    assert layout[0]["top"] == 12.0 and layout[0]["xs"] == [10.0, 50.0, 90.0]
    x, y = meta["pos"]
    assert 0 <= x <= 960 - 120 and y >= 544 - 40 - int(0.06 * 544) - 1       # an ADV box sits near the bottom


def test_psp_on_vita_doubles_the_window_masks_and_layout(v4, monkeypatch):
    v4["frames"] = only(v4["frames"])
    monkeypatch.setitem(v4, "dither565", 0.0)
    monkeypatch.setitem(v4["capture"], "psp2x", {"bilinear": 1.0, "jpeg": 0.0})
    synth3.R = type("R", (), {"bg": {}})()
    img, ink, core, ruby, layout = region_fixture(20, 60)
    out, ink_f, _, _, cap, meta, sy = synth3.frame_capture(np.random.default_rng(1), img, ink, core, ruby, layout,
                                                           "psp2x", "adv")
    assert sy == 2.0 and meta["frame"] == [960, 544] and out.shape[:2] == (40, 120) and "bilinear" in cap
    ys, xs = np.nonzero(ink_f >= 0.5)
    assert abs(ys.min() - 2 * 12) <= 1 and abs(xs.min() - 2 * 10) <= 1
    assert layout[0]["top"] == pytest.approx(24.0) and layout[0]["fs"] == pytest.approx(32.0)


def test_stretch_widens_the_window_and_x_coordinates_only(v4, monkeypatch):
    v4["frames"] = only(v4["frames"], port="stretch")
    v4["frames"]["stretch"] = [1.25, 1.25]
    for k in ("blur", "noise", "dither565"):
        monkeypatch.setitem(v4, k, 0.0)
    synth3.R = type("R", (), {"bg": {}})()
    img, ink, core, ruby, layout = region_fixture()
    out, ink_f, _, _, cap, meta, sy = synth3.frame_capture(np.random.default_rng(2), img, ink, core, ruby, layout,
                                                           "vita", "choice")
    assert sy == 1.0 and meta["stretch"] == 1.25 and out.shape[0] == 40
    assert abs(out.shape[1] - 120 * 960 / round(960 / 1.25)) <= 2
    assert layout[0]["fs"] == 16.0 and layout[0]["x1"] - layout[0]["x0"] == pytest.approx(100 * 960 / 768, abs=0.01)


def test_vndb_chain_is_q90_with_the_measured_subsampling(v4, monkeypatch):
    v4["frames"] = only(v4["frames"], chain="q90")
    v4["frames"]["vndb_420"]["psp"]["q90"] = 0.0
    synth3.R = type("R", (), {"bg": {}})()
    img, ink, core, ruby, layout = region_fixture(20, 60)
    out, *_, cap, meta, _ = synth3.frame_capture(np.random.default_rng(3), img, ink, core, ruby, layout, "psp", "adv")
    assert meta["chain"] == "q90" and meta["subsampling"] == 0 and cap[-1] == "q90-444"


def test_jpeg_quality_estimator(tmp_path):
    from PIL import Image
    a = (np.random.default_rng(0).random((32, 32, 3)) * 255).astype(np.uint8)
    for q, sub in ((90, 2), (90, 0), (70, 1)):
        Image.fromarray(a).save(tmp_path / "x.jpg", quality=q, subsampling=sub)
        assert C.jpeg_quality(tmp_path / "x.jpg") == (q, True, sub)
    Image.fromarray(a).save(tmp_path / "x.png")
    assert C.jpeg_quality(tmp_path / "x.png") is None


def test_pair_ratio_flags_nearest_upscales():
    g = np.random.default_rng(0).random((40, 60)).astype(np.float32) * 255
    assert C.pair_ratio(np.repeat(g, 2, axis=1)) < 0.05 < 0.7 < C.pair_ratio(g)


def test_frames_mode_renders_shards_with_frame_meta(assets, v4, tmp_path):  # noqa: F811
    synth3.init(assets=assets, font_manifest=assets / "fonts_manifest.json", scenes=False)
    idx, n, regions, errors, _ = synth3.run_shard((0, 40, 5, str(tmp_path), "synth3"))
    assert n == 40 and errors == 0
    z = np.load(tmp_path / "synth3_00000.npz")
    meta = json.loads(str(z["meta"]))
    labels = json.loads(str(z["labels"]))
    assert all(m["platform"] in ("vita", "psp", "psp2x") and "frame" in m for m in meta)
    for m, lab, (h, w) in zip(meta, labels, z["shapes"]):
        fw, fh = m["frame"]["frame"]                  # a PSP frame on a Vita may be 1.995x: 958 x 543
        assert lab and ((fw, fh) == (480, 272) if m["platform"] == "psp" else abs(fw - 960) <= 2 and abs(fh - 544) <= 1)
        x0, y0, x1, y1 = m["ink_box"]
        assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h
    again = tmp_path / "again"
    again.mkdir()
    synth3.run_shard((0, 40, 5, str(again), "synth3"))
    b = np.load(again / "synth3_00000.npz")
    assert np.array_equal(z["data"], b["data"]) and str(z["meta"]) == str(b["meta"])   # deterministic


def test_merge_replaces_native_crops_in_place_and_keeps_scaled_ones():
    import synth3_native_merge as M

    def shard(platforms, base):
        crops = [np.full((4, 6 + i), base + i, np.uint8) for i in range(len(platforms))]
        shapes = np.array([c.shape for c in crops], np.int32)
        return {"data": np.concatenate([c.ravel() for c in crops]),
                "offsets": np.concatenate([[0], np.cumsum(shapes[:, 0] * shapes[:, 1])]), "shapes": shapes,
                "labels": [f"{base}-{i}" for i in range(len(platforms))],
                "meta": [{"platform": p} for p in platforms]}
    v3 = shard(["scaled", "vita", "psp", "scaled", "psp2x"], 10)
    nat = shard(["psp", "vita", "psp2x", "vita"], 100)
    crops, labels, metas, k = M.merge_one(v3, nat)
    assert k == 3 and labels == ["10-0", "100-0", "100-1", "10-3", "100-2"]
    assert [m["platform"] for m in metas] == ["scaled", "psp", "vita", "scaled", "psp2x"]
    assert np.array_equal(crops[0], M.crop(v3, 0)) and np.array_equal(crops[2], M.crop(nat, 1))
    with pytest.raises(ValueError):
        M.merge_one(v3, shard(["psp"], 0))


def test_native_protocol_copies_study1b_and_lets_devnative_v3_decide():
    base = json.loads((ROOT / "eval" / "synth3_study1b_protocol.json").read_text(encoding="utf-8"))
    p = json.loads((ROOT / "eval" / "synth3_native_protocol.json").read_text(encoding="utf-8"))
    import hashlib
    assert p["base_protocol"]["sha256"] == hashlib.sha256((ROOT / "eval" / "synth3_study1b_protocol.json")
                                                          .read_bytes()).hexdigest()
    for key in ("architecture", "checkpoint", "views", "metric", "bootstrap"):
        assert p[key] == base[key], key
    rb, rp = dict(base["recipe"]), dict(p["recipe"])
    for r in (rb, rp):
        r.pop("trainer"), r.pop("note")
    assert rb == rp and p["seeds"] == base["seeds"]
    assert p["arms"]["D_s3v3"] == base["arms"]["D_s3v3"] and p["comparisons"] == {"E_vs_D": ["D_s3v3", "E_s3nat"]}
    assert p["dev"]["devnative_v3"]["role"] == "decides" and "reported only" in p["dev"]["teacher_val"]["role"]
    assert p["sources"]["nv"]["files"] == base["sources"]["nv"]["files"]
    assert p["sources"]["s3v3"]["files"] == base["sources"]["s3v3"]["files"]
    run = (ROOT / "eval" / "synth3_native_run.sh").read_text(encoding="utf-8")
    old = (ROOT / "eval" / "synth3_study1b_run.sh").read_text(encoding="utf-8")
    tail = lambda s: s[s.index("--steps"):]  # noqa: E731   the train2.py arguments after --run/--out
    assert tail(run) == tail(old)                         # the same invocation, sources aside
    assert 's3e:0.5:$S3E:stream=1,pool=3,inv=0.2' in run and "2b132aaba600ab7dbceeeb1f3a71877d3202e853" in run


def test_dev_filter_drops_only_the_owner_series_rows(tmp_path):
    sys.path.insert(0, str(ROOT / "eval"))
    import synth3_native_dev as DEV
    crops = [np.full((4, 5 + i), i, np.uint8) for i in range(4)]
    shapes = np.array([c.shape for c in crops], np.int32)
    meta = [{"id": f"x{i}", "vn": vn} for i, vn in enumerate(["v1", "v1331", "v2", "v1331"])]
    np.savez_compressed(tmp_path / "d.npz", data=np.concatenate([c.ravel() for c in crops]),
                        offsets=np.concatenate([[0], np.cumsum(shapes[:, 0] * shapes[:, 1])]), shapes=shapes,
                        labels=np.array(json.dumps(["あ", "い", "う", "え"], ensure_ascii=False)),
                        meta=np.array(json.dumps(meta)))
    kept, dropped = DEV.drop_vns(tmp_path / "d.npz", {"v1331"}, tmp_path / "o.npz")
    z = np.load(tmp_path / "o.npz")
    assert (kept, dropped) == (2, 2) and json.loads(str(z["labels"])) == ["あ", "う"]
    assert [m["id"] for m in json.loads(str(z["meta"]))] == ["x0", "x2"]
    assert np.array_equal(z["data"][z["offsets"][1]:z["offsets"][2]].reshape(z["shapes"][1]), crops[2])
