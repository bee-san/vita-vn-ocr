"""desk_ui (desktopdata/ui_synth.py): the synthetic dense-UI source. Its marks, items and boxes, the guard's and the
scene exclusion's rules, the shard and the trainer's reading of it. Synthetic data only."""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "synth", ROOT / "train", ROOT / "teacher", ROOT / "eval", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

pytest.importorskip("cv2")
pytest.importorskip("PIL")

from desktopdata import ui_synth as U  # noqa: E402


def ci_font():
    cache = ROOT / ".cache" / "fonts"
    hits = sorted(cache.rglob("MPLUS1p-Regular.ttf")) if cache.exists() else []
    if hits:
        return hits[0]
    try:
        U.init(scenes=False)
    except Exception:  # noqa: BLE001 - no font pool here
        return None
    R = U.S3().R
    for fam in R.by_cat.get("gothic", {}).values():
        for f in fam:
            if set("・セーブロードAB") <= f.chars and f.licence == "open" and "[wght]" not in f.path.name:
                return f.path
    return None


@pytest.fixture(scope="module")
def synth3():
    try:
        U.init(scenes=False)
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"synth3's resources are not here: {type(e).__name__}")
    U.GUARD_ON = False
    return U


def _frame(W=640, H=360):
    import textspot_data as TSD
    U.D1.TS = TSD
    fr = TSD.Frame("pc", W, H, "vita", H / 544, None)
    fr.img = np.zeros((H, W, 3), np.float32)
    fr.bg, fr.item_info, fr.elements = [], {}, ["test"]
    return fr


def test_icon_masks():
    for kind in U.ICONS:
        m = U.icon_mask(kind, 24)
        assert m.shape == (24, 24) and m.dtype == np.float32
        assert 0.02 < float(m.mean()) < 0.9, kind
    assert U.icon_mask("bullet", 2).shape == (4, 4)


def test_item_box_is_its_ink_and_holds_the_nakaguro():
    path = ci_font()
    if path is None:
        pytest.skip("no CI font")
    from PIL import ImageFont
    font = ImageFont.truetype(str(path), 40)
    st = {"fill": np.array([255, 255, 255], np.float32)}
    it = U.make_item("・セーブ", font, st)
    assert it is not None and it.gap <= U.GAP_IN
    fr = _frame()
    box = U.put(fr, it, 100, 200, kind="list")
    lines, ignore, info = U.instance_boxes(fr)
    assert lines == [[*box, 0, "ui"]] and info == [["・セーブ", "list"]] and ignore == []
    lit = np.nonzero(fr.img.max(2) > 127)
    assert lit[1].min() >= box[0] and lit[1].max() < box[2] and lit[0].min() >= box[1] and lit[0].max() < box[3]
    dot = np.nonzero(fr.img[:, box[0]:box[0] + int(0.6 * it.h)].max(2) > 127)
    assert len(dot[0])                                   # the ・ is inside the box, at its left end


def test_make_item_keeps_inside_gaps_small():
    path = ci_font()
    if path is None:
        pytest.skip("no CI font")
    from PIL import ImageFont
    font = ImageFont.truetype(str(path), 40)
    st = {"fill": np.array([255, 255, 255], np.float32)}
    wide = U.Item("AB" + "　" * 3 + "CD", font, st)
    assert wide.ok and wide.gap > U.GAP_IN
    it = U.make_item("AB" + "　" * 3 + "CD", font, st)
    assert it is not None and it.text == "ABCD" and it.gap <= U.GAP_IN
    spaced = U.make_item("SAVE", font, st, spacing=60.0)          # too much letter spacing: dropped
    assert spaced is not None and spaced.gap <= U.GAP_IN


def test_guard_leaves_marks_alone(monkeypatch):
    rgb = np.zeros((100, 200, 3), np.uint8)
    monkeypatch.setattr(U.D1, "guard_ignore", lambda rgb, lines, ignore: [[10, 10, 30, 30], [100, 50, 140, 80]])
    keep, left = U.guard_ui(rgb, [], [], [[12, 12, 28, 28]])
    assert keep == [[100, 50, 140, 80]] and left == 1
    keep, left = U.guard_ui(rgb, [], [], [])
    assert len(keep) == 2 and left == 0


def test_exclude_scenes(tmp_path, monkeypatch):
    import tsdesk_dense_source as DSS
    files = {}
    for name, body in (("v1_100.jpg", b"a"), ("v2_200.jpg", b"b"), ("v3_300.jpg", b"c"), ("v4_400.jpg", b"d")):
        (tmp_path / name).write_bytes(body)
        files[name] = str(tmp_path / name)
    import hashlib
    sha_d = hashlib.sha256(b"d").hexdigest()
    monkeypatch.setattr(DSS, "exclusion_sets", lambda: {
        "a": {"vns": {"v2"}, "sha256": set(), "shots": set()},
        "b": {"vns": set(), "sha256": {sha_d}, "shots": {"v3_300"}}})
    fake = types.SimpleNamespace(SCENES={"hires": list(files.values()), "vita": [], "psp": []},
                                 SCENE_ASPECT={"16:9": [], "4:3": []})
    fake._aspects = lambda: fake.SCENE_ASPECT.update({"16:9": list(fake.SCENES["hires"])})
    monkeypatch.setattr(U.D1, "TS", fake)
    rec = U.exclude_scenes()
    assert fake.SCENES["hires"] == [files["v1_100.jpg"]] and fake.SCENE_ASPECT["16:9"] == [files["v1_100.jpg"]]
    assert rec["dropped"] == {"vn": 1, "shot": 1, "sha256": 1} and rec["after"]["hires"] == 1


def _same_row_gaps(lines):
    out = []
    for i, a in enumerate(lines):
        for b in lines[i + 1:]:
            ha, hb = a[3] - a[1], b[3] - b[1]
            iy = min(a[3], b[3]) - max(a[1], b[1])
            gap = max(a[0], b[0]) - min(a[2], b[2])
            if iy >= 0.6 * min(ha, hb) and gap >= 0:
                out.append(gap / max(ha, hb))
    return out


def test_frames_follow_the_conventions(synth3):
    rng = np.random.default_rng(3)
    n = native = 0
    seen = set()
    while n < 12:
        r = U.sample_frame(rng)
        if r is None:
            continue
        s, g = r
        n += 1
        L, info, marks = s["lines"], s["meta"]["items"], s["meta"]["marks"]
        assert g.shape == (s["meta"]["size"][1], s["meta"]["size"][0]) and len(L) == len(info)
        assert {ln[5] for ln in L} <= {"ui", "dialogue", "name"}
        for i, a in enumerate(L):
            for b in L[i + 1:]:
                assert not (min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1]))
            for m in marks:                                 # background marks never meet a positive
                assert not (min(m[2], a[2]) > max(m[0], a[0]) and min(m[3], a[3]) > max(m[1], a[1]))
        if not s["meta"]["capture"]:
            native += 1
            assert all(r_ >= U.GAP_BETWEEN for r_ in _same_row_gaps(L))
        seen |= {k for _, k in info}
    assert native and len(seen) >= 4


def test_frames_are_reproducible(synth3):
    a = U.sample_frame(np.random.default_rng(7))
    b = U.sample_frame(np.random.default_rng(7))
    assert (a is None) == (b is None)
    if a is not None:
        assert a[0]["lines"] == b[0]["lines"] and np.array_equal(a[1], b[1])


def test_shard_manifest_and_trainer(synth3, tmp_path):
    import argparse
    pytest.importorskip("torch")
    import desktop_det as DD
    import tsdesk_det as TD
    out = tmp_path / "ui" / "desk_ui_v1"
    idx, n, errors, _ = U.run_shard((0, 3, 5, str(out)))
    assert n == 3 and errors == 0
    man = tmp_path / "tsdesk_ui.json"
    U.cmd_manifest(argparse.Namespace(dir=str(out), seed=5, out=str(man)))
    doc = json.loads(man.read_text())
    assert doc["totals"]["all"]["samples"] == 3 and doc["generator"] == "desktopdata/ui_synth.py"
    pool = DD.ExtraPool(man, tmp_path / "ui", verify=True)
    assert len(pool.items) == 3
    assert TD.exact_source("desk_ui") and not TD.exact_source("desk_realplus")

    class Sources:
        def pick(self, rng):
            canvas, smp = pool.draw(rng)
            return "desk_ui", canvas, smp

        def empty_canvas(self, rng):
            return None
    cfg = {"stride": 8, "batch": 4, "scale": (0.42, 1.9), "aug": True, "recipe": "base", "end_ignore": 1.0,
           "e_sources": list(TD.EXACT_SOURCES)}
    x, K, *rest = TD.make_batch(Sources(), np.random.default_rng(1), cfg)
    assert x.shape[0] == 4 and float(K.sum()) > 0
