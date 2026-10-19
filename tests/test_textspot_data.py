"""Tests of synth/textspot_data.py (issue #104, docs/textspot.md): the synthetic whole frames of the full-frame text
spotter. The scaling and box rules run on synthetic masks with a stub capture; a few frames are rendered end to end
with the pinned CI font and the fixture corpus over synthetic scenes (no screenshots, no shared data); the data pin
(data/manifests/textspot_shards_v1.json) is checked for its schema and its sums."""
import hashlib
import json
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "synth", ROOT / "teacher"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import geometry  # noqa: E402

import textdet_data as TDD  # noqa: E402
import textspot_data as TS  # noqa: E402

FIXTURES = ROOT / "ci" / "fixtures" / "corpus"
SIZES = {(960, 544), (480, 272), (960, 540), (725, 544)}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- the scalings, on synthetic masks

def _blob(rng, h, w):
    """A random binary mask of h x w whose bounding box is the whole of it."""
    m = rng.random((h, w)) < 0.4
    m[0, int(rng.integers(w))] = m[-1, int(rng.integers(w))] = True
    m[int(rng.integers(h)), 0] = m[int(rng.integers(h)), -1] = True
    return m


def test_the_2x_bilinear_box_is_exactly_twice_the_box():
    rng = np.random.default_rng(0)
    for _ in range(300):
        h, w = int(rng.integers(1, 40)), int(rng.integers(1, 120))
        x, y = int(rng.integers(0, 480 - w + 1)), int(rng.integers(0, 272 - h + 1))
        m = _blob(rng, h, w)
        b = TS.scale_box(x, y, m, (480, 272), (960, 544), cv2.INTER_LINEAR)
        assert b == [2 * x, 2 * y, 2 * (x + w), 2 * (y + h)]


def _area_cover(a, b, n_src, n_dst):
    """Output pixels of an area resize n_src -> n_dst whose coverage by the source interval [a, b) is >= 0.5, and the
    smallest distance of any coverage to 0.5 (ties are left to floating point)."""
    r = n_src / n_dst
    keep, tie = [], 1.0
    for j in range(n_dst):
        lo, hi = j * r, (j + 1) * r
        c = max(0.0, min(hi, b) - max(lo, a)) / r
        tie = min(tie, abs(c - 0.5))
        if c >= 0.5:
            keep.append(j)
    return keep, tie


@pytest.mark.parametrize("size", [(1280, 720), (800, 600), (1024, 768)])
def test_the_fit_box_is_the_area_coverage_of_the_mask(size):
    W, H = size
    s, dst = TS.fit_size(W, H)
    rng = np.random.default_rng(W)
    done = 0
    while done < 60:
        h, w = int(rng.integers(8, 60)), int(rng.integers(8, 200))
        x, y = int(rng.integers(0, W - w + 1)), int(rng.integers(0, H - h + 1))
        kx, tx = _area_cover(x, x + w, W, dst[0])
        ky, ty = _area_cover(y, y + h, H, dst[1])
        if min(tx, ty) < 1e-3:
            continue
        m = np.ones((h, w), bool)        # a full block: its interior rows and columns are covered whole
        b = TS.scale_box(x, y, m, (W, H), dst, cv2.INTER_AREA)
        assert b == [kx[0], ky[0], kx[-1] + 1, ky[-1] + 1]
        done += 1


def test_the_fit_sizes_follow_the_teacher_target_scale():
    for (w, h), want in {(1280, 720): (960, 540), (800, 600): (725, 544), (1024, 768): (725, 544)}.items():
        s, dst = TS.fit_size(w, h)
        assert dst == want and s == geometry.target_scale(w, h, "vita")
        assert dst == (int(round(w * s)), int(round(h * s)))


def _stub_capture(monkeypatch):
    stub = SimpleNamespace(jpeg_frame=lambda a, q, sub: a, dither565=lambda a: a)
    monkeypatch.setattr(TS, "S3", stub)


def _frame(kind, W, H, platform, k=1.0):
    fr = TS.Frame(kind, W, H, platform, k)
    fr.img = np.full((H, W, 3), 128, np.float32)
    fr.scene = "stub"
    return fr


@pytest.mark.parametrize("kind", ["vita", "psp", "adrenaline", "pc"])
def test_finalize_scales_masks_and_ignores_text_under_6_px(kind, monkeypatch):
    _stub_capture(monkeypatch)
    rng = np.random.default_rng(1)
    W, H, plat = {"vita": (960, 544, "vita"), "psp": (480, 272, "psp"), "adrenaline": (480, 272, "psp"),
                  "pc": (1280, 720, "vita")}[kind]
    fr = _frame(kind, W, H, plat)
    fr.add(40, 52, np.ones((20, 100), np.float32), 0, "dialogue")      # a line 20 px tall
    fr.elements.append("a")
    fr.add(300, 10, np.ones((60, 22), np.float32), 1, "scene")         # a column 22 px wide
    fr.elements.append("b")
    fr.add(200, 200, np.ones((2, 30), np.float32), 0, "ui")            # 2 px: under 6 px at any scale here
    fr.ignore.append([10, 10, 20, 20])
    s = TS.finalize(fr, rng)
    h, w = s["canvas"].shape
    if kind == "adrenaline":
        assert (w, h) == (960, 544)
        assert s["lines"][0][:4] == [80, 104, 280, 144] and s["lines"][1][:5] == [600, 20, 644, 140, 1]
        assert [20, 20, 40, 40] in s["ignore"]
    elif kind == "pc":
        assert (w, h) == (960, 540)
        assert s["lines"][0][:4] == [30, 39, 105, 54]       # [40, 140) x [52, 72) at 3/4
    else:
        assert s["lines"][0][:4] == [40, 52, 140, 72] and s["lines"][1][:5] == [300, 10, 322, 70, 1]
    assert [ln[5] for ln in s["lines"]] == ["dialogue", "scene"]
    assert len(s["ignore"]) == 2                            # the given area and the 2 px line
    for b in [ln[:4] for ln in s["lines"]] + s["ignore"]:
        assert 0 <= b[0] < b[2] <= w and 0 <= b[1] < b[3] <= h
    assert s["meta"]["line_element"] == [0, 1]


def test_an_instance_that_vanishes_in_the_fit_becomes_an_ignore_area(monkeypatch):
    _stub_capture(monkeypatch)
    fr = _frame("pc", 1280, 720, "vita")
    m = np.zeros((30, 30), np.float32)
    m[1::4, 1::4] = 1.0         # isolated dots on frame pixels 4k + 1: each straddles two output pixels at 3/4
    fr.add(100, 100, m, 0, "ui")                        # (coverage <= 0.5 x 0.5), so none reaches 0.5
    s = TS.finalize(fr, np.random.default_rng(0))
    assert s["lines"] == [] and len(s["ignore"]) == 1


def test_line_band_box_is_textdet_main_box():
    rng = np.random.default_rng(3)
    for _ in range(50):
        H, W = 120, 300
        ink = rng.random((H, W)) < 0.3
        ruby = np.zeros((H, W), bool)
        ruby[10:16, 50:90] = True
        layout = []
        for i, top in enumerate((20, 60)):
            L = {"top": top, "fs": 24, "y0": top - 2, "y1": top + 26, "ix0": int(rng.integers(0, 40)),
                 "ix1": int(rng.integers(200, 300)), "ruby_top": 10 if i == 0 else None}
            layout.append(L)
        for L in layout:
            y0, y1, x0, x1 = TS.line_band(L, layout, H)
            band = ink[y0:y1, x0:x1] & ~ruby[y0:y1, x0:x1]
            ys, xs = np.nonzero(band)
            got = (x0 + int(xs.min()), y0 + int(ys.min()), x0 + int(xs.max()) + 1, y0 + int(ys.max()) + 1)
            assert got == TDD.main_box(L, layout, ink, ruby, (H, W))


def test_mask_boxes_and_frame_bookkeeping():
    m = np.zeros((40, 100), bool)
    m[5:9, 10:14] = m[5:9, 15:20] = True        # one part (1 px apart)
    m[30:35, 60:70] = True
    assert TS.mask_boxes(m) == [[10, 5, 20, 9], [60, 30, 70, 35]]
    fr = _frame("vita", 960, 544, "vita")
    fr.reserve(100, 100, 200, 150)
    assert not fr.free(150, 120, 250, 160) and fr.free(210, 100, 300, 150) and not fr.free(205, 100, 300, 150, 6)
    mm = np.zeros((10, 10), bool)
    mm[2:5, 3:7] = True
    fr.add(10, 20, mm, 0, "ui")
    x, y, trimmed, theta = fr.inst[0][:4]
    assert (x, y, trimmed.shape, theta) == (13, 22, (3, 4), 0.5)


def test_the_ink_threshold_follows_the_peak_coverage():
    assert TS.instance_theta(np.array([0.0, 1.0])) == 0.5
    assert TS.instance_theta(np.array([0.1, 0.46])) == pytest.approx(0.23)
    assert TS.instance_theta(np.array([0.29])) is None
    fr = _frame("vita", 960, 544, "vita")
    thin = np.zeros((14, 40), np.float32)
    thin[2:12, 3:37] = 0.4                  # a thin face: drawn, never 0.5
    assert fr.add(100, 100, thin, 0, "ui") and fr.inst[-1][3] == pytest.approx(0.2)
    faint = np.full((8, 20), 0.2, np.float32)
    assert not fr.add(300, 100, faint, 0, "ui") and fr.ignore[-1] == [300, 100, 320, 108]


def _layout_line(top, fs, x0, x1, kind="dialogue", sw=1, dy=1):
    return {"top": top, "fs": fs, "x0": x0, "x1": x1, "kind": kind, "y0": top - sw - 1, "y1": top + fs + sw + dy + 1}


def test_region_lines_keep_the_tab_ink_apart_and_label_thin_lines():
    H, W = 120, 300
    main = np.zeros((H, W), np.float32)
    tab = np.zeros((H, W), np.float32)
    tab[20:44, 40:120] = 1.0                    # the tab's name, reaching into the first line's band
    main[40:62, 30:250] = 1.0                   # line 1
    main[78:98, 30:150] = 0.45                  # line 2 in a thin face
    lay = [_layout_line(20, 22, 40, 120, "name"), _layout_line(40, 22, 30, 250), _layout_line(77, 20, 30, 150)]
    ruby = np.zeros((H, W), bool)
    lines, faint = TS.region_lines(lay, [main, tab], True, ruby, None, (H, W))
    boxes = {role + str(i): TS._box(c0, r0, crop, th) for i, (c0, r0, crop, th, role) in enumerate(lines)}
    assert boxes == {"name0": [40, 20, 120, 44], "dialogue1": [30, 40, 250, 62], "dialogue2": [30, 78, 150, 98]}
    assert faint == []
    assert TS._overlap(boxes["name0"], boxes["dialogue1"])          # a tab over the first line: drawn again
    assert not TS._overlap([0, 0, 10, 20], [0, 18, 10, 40])           # two rows shared by tight lines: kept
    n, left = TS.unlabelled_ink([main, tab], list(boxes.values()), ruby, None)
    assert n == 0 and not left.any()
    main[100:104, 200:220] = 1.0                # stray ink no box holds
    n, left = TS.unlabelled_ink([main, tab], list(boxes.values()), ruby, None)
    assert n == 80 and TS.mask_boxes(left) == [[200, 100, 220, 104]]


def test_mix_and_capture_are_distributions():
    for name in ("kind", "empty_size", "empty_mode", "empty_box_layout", "pc_size", "main", "layout", "ui",
                 "scene_text_kind"):
        assert abs(sum(TS.MIX[name].values()) - 1.0) < 1e-9, name
    assert abs(sum(TS.MIX["ui_n"]) - 1.0) < 1e-9 and len(TS.MIX["ui_n"]) == 9     # 0-8 UI elements
    for k in ("vita", "psp", "adrenaline", "pc"):
        assert abs(sum(TS.CAPTURE[k]["chain"].values()) - 1.0) < 1e-9
    assert set(TS.MIX["kind"]) == {"vita", "psp", "adrenaline", "pc", "empty"}
    assert TS.MIN_ACROSS == 6 and TS.CLASSES == ("dialogue", "name", "choice", "ui", "scene")
    assert set(TS.ROLE.values()) <= set(TS.CLASSES)


# ----------------------------------------------------------------------------- frames rendered end to end

def _fixture_lines(name):
    return [s for s in (FIXTURES / f"{name}.txt").read_text(encoding="utf-8").split("\n") if s]


@pytest.fixture(scope="module")
def shard(font, tmp_path_factory):
    """Two shards of frames from the CI font, the fixture corpus and synthetic scenes, rendered twice."""
    import PIL.features

    import fonts
    import synth3
    import synth3_text as T
    root = tmp_path_factory.mktemp("textspot") / "assets"
    fam = root / "fonts" / fonts.FAMILY
    fam.mkdir(parents=True)
    for name in fonts.FILES:
        shutil.copy2(Path(font).parent / name, fam / name)
    f = fam / fonts.FONT_FILE
    manifest = {"kind": "synth3-fonts", "schema": 1, "families": [{
        "family": fonts.FAMILY, "category": "gothic", "root": "synth3", "licence_id": "OFL-1.1",
        "licence_class": "open", "licence_file": f"{fonts.FAMILY}/OFL.txt",
        "files": [{"file": f"{fonts.FAMILY}/{fonts.FONT_FILE}", "bytes": f.stat().st_size,
                   "sha256": synth3.sha256(f), "category": "gothic", "weight": 400, "pixel_em": None}]}]}
    (root / "fonts_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    pools = root / "text" / "pools3"
    pools.mkdir(parents=True)
    meta = {}
    src = {"aozora_dialogue": _fixture_lines("dialogue"), "aozora_narration": _fixture_lines("narration"),
           "wiki": _fixture_lines("wiki"), "tatoeba": [s.strip("「」") for s in _fixture_lines("dialogue")]}
    for name, lines in src.items():
        for split in ("train", "dev"):
            path = pools / f"{name}.{split}.txt"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            meta[f"{name}.{split}"] = {"file": path.name, "lines": len(lines), "class": "open"}
    (pools / "pools.json").write_text(json.dumps({"pools": meta}), encoding="utf-8")
    for split in ("train", "dev"):
        (pools / f"names.{split}.tsv").write_text("vndb\tprimary\t佐藤\t花子\nsyosetu\t-\t健太\n", encoding="utf-8")
        T.cmd_index(type("A", (), {"vocab": ROOT / "data" / "vocab.txt", "pools": str(pools), "split": split,
                                   "k": 8})())
    chars = {c: {"real": 1e-3, "jiten_vn": 1e-3, "hard": 0.0} for c in "".join(sum(src.values(), []))}
    (pools / "targets.json").write_text(json.dumps({"chars": chars}, ensure_ascii=False), encoding="utf-8")
    lists = {"interjections": ["あ", "えっ"], "stammers": ["あ、あの……"], "onomatopoeia": ["ドキドキ"],
             "honorifics": ["さん", "先輩"], "elongations": ["ーっ"], "endings": ["だよ"]}
    (root / "text" / "lists.json").write_text(json.dumps(lists, ensure_ascii=False), encoding="utf-8")
    TS.init(scenes=False, assets=root, font_manifest=root / "fonts_manifest.json")
    saved = {k: TS.MIX[k] for k in ("main", "scene_text_vertical", "vertical_extra")}
    if not PIL.features.check("raqm"):          # vertical typesetting needs raqm; without it, horizontal only
        TS.MIX.update(main={"horizontal": 0.87, "vertical": 0.0, "none": 0.13}, scene_text_vertical=0.0,
                      vertical_extra=0.0)
    out = tmp_path_factory.mktemp("frames")
    again = tmp_path_factory.mktemp("frames_again")
    try:
        for d in (out, again):
            for idx in (0, 1):
                i, n, errors, _ = TS.run_shard((idx, 12, 5, str(d)))
                assert n == 12 and errors == 0
    finally:
        TS.MIX.update(saved)
    return out, again


def test_frames_follow_the_shard_format_and_the_rules(shard):
    out, _ = shard
    n_lines = 0
    for idx in (0, 1):
        data, samples = TDD.load_shard(out / f"synth_frames_{idx:04d}")
        assert len(samples) == 12
        off = 0
        for s in samples:
            assert s["off"] == off
            off += s["h"] * s["w"]
            assert (s["w"], s["h"]) in SIZES
            assert s["src"].startswith("synth:") and s["src"].split(":")[1] in {"vita", "psp", "adrenaline", "pc",
                                                                                "empty"}
            assert s["kind"] == ("psp" if (s["w"], s["h"]) == (480, 272) else "vita")
            assert s["vn"] is None and s["shot"] is None
            if s["src"] == "synth:empty":
                assert s["lines"] == [] and s["ignore"] == []
            else:
                assert s["lines"]
            for x0, y0, x1, y1, v, role in s["lines"]:
                assert 0 <= x0 < x1 <= s["w"] and 0 <= y0 < y1 <= s["h"]
                assert v in (0, 1) and role in TS.CLASSES
                assert ((x1 - x0) if v else (y1 - y0)) >= TS.MIN_ACROSS
            for x0, y0, x1, y1 in s["ignore"]:
                assert 0 <= x0 < x1 <= s["w"] and 0 <= y0 < y1 <= s["h"]
            el = s["meta"]["line_element"]
            assert len(el) == len(s["lines"])
            for i, a in enumerate(s["lines"]):          # the elements never overlap each other's ink
                for j, b in enumerate(s["lines"]):
                    if el[i] < el[j]:
                        assert min(a[2], b[2]) <= max(a[0], b[0]) or min(a[3], b[3]) <= max(a[1], b[1])
            n_lines += len(s["lines"])
        assert data.size == off
    assert n_lines > 24


def test_the_lines_lie_on_ink(shard):
    """Every box holds pixels that differ from the frame around it: a box is never off the ink."""
    out, _ = shard
    data, samples = TDD.load_shard(out / "synth_frames_0000")
    for s in samples:
        cv = TDD.canvas(data, s).astype(np.int32)
        for x0, y0, x1, y1, v, role in s["lines"]:
            box = cv[y0:y1, x0:x1]
            assert box.max() - box.min() >= 40, (s["src"], role)


def test_the_render_is_deterministic(shard):
    out, again = shard
    for idx in (0, 1):
        for suf in (".npy", ".json"):
            assert sha256(out / f"synth_frames_{idx:04d}{suf}") == sha256(again / f"synth_frames_{idx:04d}{suf}")


def test_manifest_command_sums_the_shards(shard, tmp_path):
    out, _ = shard
    (out / "scene_pool.json").write_text(json.dumps({"pool": {"vita": 0, "psp": 0, "hires": 0}, "sha256": None}))
    m = tmp_path / "m.json"
    TS.cmd_manifest(SimpleNamespace(dir=str(out), out=str(m), seed=5, per_shard=12, real_manifest=None,
                                    real_stats=None))
    doc = json.loads(m.read_text())
    assert doc["totals"]["frames"] == 24 and len(doc["shards"]) == 2
    assert doc["totals"]["lines"] == sum(doc["totals"]["by_class"].values())
    assert doc["totals"]["lines"] == sum(doc["totals"]["by_orientation"].values())


# ----------------------------------------------------------------------------- the data pin

def _pins():
    return sorted((ROOT / "data" / "manifests").glob("textspot_shards_v*.json"))


@pytest.mark.skipif(not _pins(), reason="no textspot data pin yet")
def test_the_data_pin_schema_and_sums():
    for p in _pins():
        m = json.loads(p.read_text(encoding="utf-8"))
        assert m["kind"] == "textspot-shards" and m["schema"] == 1 and m["generator"] == "synth/textspot_data.py"
        assert m["classes"] == list(TS.CLASSES) and m["min_across"] == 6 and isinstance(m["seed"], int)
        assert len(m["shards"]) >= 40
        names = [s["name"] for s in m["shards"]]
        assert len(set(names)) == len(names)
        tot = {}
        for s in m["shards"]:
            assert s["dir"] == "synth_frames" and s["frames"] == m["per_shard"]
            assert len(s["npy_sha256"]) == 64 and len(s["json_sha256"]) == 64
            assert sum(s["by_kind"].values()) == s["frames"] == sum(s["by_size"].values())
            assert set(s["by_kind"]) <= {"vita", "psp", "adrenaline", "pc", "empty"}
            assert {tuple(int(v) for v in z.split("x")) for z in s["by_size"]} <= SIZES
            assert sum(s["by_class"].values()) == s["lines"] == sum(s["by_orientation"].values())
            TS._add(tot, {k: s[k] for k in ("frames", "lines", "ignore", "by_kind", "by_class")})
        for k in ("frames", "lines", "ignore", "by_kind", "by_class"):
            assert m["totals"][k] == tot[k], k
        pool = m["scene_pool"]
        assert len(pool["sha256"]) == 64 and all(pool["pool"][k] > 0 for k in ("vita", "psp", "hires"))
        assert all(pool["pool"][k] <= pool["textgate"][k] for k in pool["pool"])
        assert m["real"]["sha256"] == sha256(ROOT / m["real"]["manifest"])
        assert {"synth/textspot_data.py", "synth/synth3.py", "synth/textdet_data.py"} <= set(m["code_sha256"])


@pytest.mark.skipif(not _pins(), reason="no textspot data pin yet")
def test_the_newest_pin_is_the_generator_at_head():
    """Shards never change after they are pinned: a changed generator means a new manifest version."""
    m = json.loads(_pins()[-1].read_text(encoding="utf-8"))
    assert m["code_sha256"]["synth/textspot_data.py"] == sha256(ROOT / "synth" / "textspot_data.py")
