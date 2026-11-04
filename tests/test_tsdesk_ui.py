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


# ----------------------------------------------------------------------------- desk_ui v2 (--version 2)

@pytest.fixture()
def v2(synth3):
    U.set_version(2)
    yield U
    U.set_version(1)


def _font(size=40):
    path = ci_font()
    if path is None:
        pytest.skip("no CI font")
    from PIL import ImageFont
    return ImageFont.truetype(str(path), size)


def _meets(a, b):
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def test_set_version():
    assert U.VERSION == 1 and U.mix() is U.MIX_UI
    U.set_version(2)
    try:
        assert U.mix() is U.MIX_UI2 and abs(sum(U.MIX_UI2["layout"].values()) - 1.0) < 1e-9
    finally:
        U.set_version(1)
    with pytest.raises(ValueError):
        U.set_version(3)
    assert set(U.LAYOUTS) < set(U.LAYOUTS2) and set(U.MIX_UI2["layout"]) == set(U.LAYOUTS2)


def test_v2_icon_and_badge_masks():
    for kind in U.ICONS2:
        m = U.icon_mask(kind, 24)
        assert m.shape == (24, 24) and 0.02 < float(m.mean()) < 0.9, kind
    for plate in U.BADGE_PLATES:
        pm, im = U.badge_masks(plate, "check", 20)
        assert pm.shape == im.shape and pm.shape[0] == 20 and float(im.max()) > 0.2
        assert pm.shape[1] == (34 if plate == "pill" else 20)
        assert np.all(im <= pm + 1e-6)                      # the pictogram lies on the plate
    mk = U.icon_mark("doc", 30, np.array([255, 255, 255], np.float32))
    ys, xs = np.nonzero(mk.layers[0][0] > U.MARK_INK)
    assert mk.ok and mk.ink == (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)


def test_group_is_one_box_with_its_pair_gap():
    font = _font()
    st = {"fill": np.array([255, 255, 255], np.float32)}
    a, b = U.make_item("攻撃力", font, st), U.make_item("20", font, st)
    g = U.Group([a, b], [37])
    fr = _frame()
    fr.pair_gap = {}
    box = U.put_group(fr, g, 50, 120, kind="statbar")
    lines, ignore, info = U.instance_boxes(fr)
    assert len(fr.inst) == 1 and lines == [[*box, 0, "ui"]] and info == [["攻撃力 20", "statbar"]]
    assert box == U.item_box(g, 50, 120) and box[2] - box[0] == a.w + 37 + b.w
    assert fr.pair_gap == {0: 37}
    lit = (fr.img.max(2) > 127)[box[1]:box[3], box[0]:box[2]].any(axis=0)
    runs, run = [], 0
    for v in lit:
        if v:
            runs.append(run)
            run = 0
        else:
            run += 1
    assert max(runs) >= 37 - 2                          # the label / value gap is inside the one box
    p = U.pair_group(np.random.default_rng(0), a, b)
    assert U.GAP_PAIR[0] * p.h - 1 <= p.pair_gap <= U.GAP_PAIR[1] * p.h + 1


def test_marks_beside_a_label_are_background_at_their_gaps():
    font = _font()
    st = {"fill": np.array([255, 255, 255], np.float32)}
    it = U.make_item("セーブ", font, st)
    before = U.icon_mark("grid", it.h, np.array([200, 200, 200], np.float32))
    after = U.badge_mark(np.random.default_rng(2), it.h * 0.8, np.array([230, 40, 40], np.float32),
                         np.array([255, 255, 255], np.float32))
    fr = _frame()
    fr.attached = []
    e = U.Entry(it, (before, 0), (after, 4))
    box = U.draw_entry(fr, e, 60, 200, "iconlist")
    assert box == U.item_box(it, 60 + e.lead, 200) and len(fr.bg) == 2
    mb, ma = fr.bg
    assert box[0] - mb[2] == 0 and ma[0] - box[2] == 4              # touching, and 4 px after
    assert not _meets(mb, box) and not _meets(ma, box)
    assert fr.attached == [[0, 0, "before", 0], [1, 0, "after", 4]]
    lines, _, _ = U.instance_boxes(fr)
    assert lines == [[*box, 0, "ui"]]                                # the label's box is its own ink only


def test_leader_marks_fit_their_span():
    font = _font(32)
    rng = np.random.default_rng(4)
    st = {"fill": np.array([255, 255, 255], np.float32)}
    for kind in ("dots", "squares", "dash", "rule", "double", "dotted", "glyph"):
        r = U.leader_mark(rng, kind, 200, 30, np.array([255, 255, 255], np.float32), font, st)
        assert r is not None, kind
        mk, where = r
        assert mk.ok and mk.w <= 200 and where in ("mid", "base", "glyph"), kind
        assert (where == "glyph") == (kind == "glyph")
    assert U.leader_mark(rng, "glyph", 2 * font.getlength("…"), 30, st["fill"], font, st) is None   # under 3 glyphs


def test_v2_frames_follow_the_conventions(v2):
    rng = np.random.default_rng(3)
    n = native = pairs = att = 0
    kinds = set()
    while n < 14:
        r = U.sample_frame(rng)
        if r is None:
            continue
        s, g = r
        n += 1
        L, m = s["lines"], s["meta"]
        assert m["generator"] == "ui_synth 2" and len(L) == len(m["items"]) == len(m["forms"])
        assert len(m["mark_kinds"]) == len(m["marks"]) and all(m["forms"][li] == "pair" for li, _ in m["pairs"])
        assert {ln[5] for ln in L} <= {"ui", "dialogue", "name"}
        assert all(ln[5] == "ui" for ln, (t, k) in zip(L, m["items"]) if k in U.LAYOUTS2)
        for i, a in enumerate(L):
            for b in L[i + 1:]:
                assert not _meets(a, b)
            for mk in m["marks"]:
                if m["capture"]:          # the capture model maps both boxes outward: a touching mark may share 1 px
                    assert min(mk[2], a[2]) - max(mk[0], a[0]) <= 1 or min(mk[3], a[3]) - max(mk[1], a[1]) <= 1
                else:
                    assert not _meets(mk, a)
        kinds |= {k for _, k in m["items"]}
        if m["capture"]:
            continue
        native += 1
        assert all(r_ >= U.GAP_BETWEEN for r_ in _same_row_gaps(L))
        for li, gap in m["pairs"]:
            h = L[li][3] - L[li][1]
            assert U.GAP_PAIR[0] * h - 1.5 <= gap <= U.GAP_PAIR[1] * h + 1.5
            pairs += 1
        for mi, li, side, gap in m["attached"]:
            mk, b = m["marks"][mi], L[li]
            assert (b[0] - mk[2] if side == "before" else mk[0] - b[2]) == gap
            att += 1
    assert native and pairs and att and len(kinds & {"statbar", "ledger", "iconlist", "selector", "framed",
                                                     "pairs2"}) >= 3


def test_versions_do_not_leak(synth3):
    a = U.sample_frame(np.random.default_rng(7))
    U.set_version(2)
    try:
        b = U.sample_frame(np.random.default_rng(7))
    finally:
        U.set_version(1)
    c = U.sample_frame(np.random.default_rng(7))
    assert (a is None) == (c is None)
    if a is not None:
        assert a[0] == c[0] and np.array_equal(a[1], c[1]) and "attached" not in a[0]["meta"]
    assert b is None or b[0]["meta"]["generator"] == "ui_synth 2"


def test_exact_source_prefix():
    import tsdesk_det as TD
    assert TD.EXACT_SOURCES_DEFAULT == ("hook", "desk_hook", "synth", "desk_synth", "desk_ui")
    assert TD.exact_source("desk_ui_v2") and TD.exact_source("desk_ui_v2", ["hook", "desk_ui"])
    assert not TD.exact_source("desk_ui_v2", ["hook", "desk_hook"]) and not TD.exact_source("desk_uiv2")
    assert not TD.exact_source("desk_dense") and not TD.exact_source("desk_realplus")


def test_v2_shard_manifest_and_trainer(v2, tmp_path):
    import argparse
    pytest.importorskip("torch")
    import desktop_det as DD
    import tsdesk_det as TD
    out = tmp_path / "ui" / "desk_ui_v2"
    idx, n, errors, _ = U.run_shard((0, 3, 5, str(out)))
    assert n == 3 and errors == 0
    man = tmp_path / "tsdesk_ui_v2.json"
    U.cmd_manifest(argparse.Namespace(dir=str(out), seed=5, out=str(man), version=2))
    doc = json.loads(man.read_text())
    assert doc["generator_version"] == 2 and "desk_ui_v2=MANIFEST:ROOT" in doc["format"]
    assert doc["mix"]["layout"] == U.MIX_UI2["layout"] and doc["gap_pair"] == list(U.GAP_PAIR)
    with pytest.raises(SystemExit):
        U.cmd_manifest(argparse.Namespace(dir=str(out), seed=5, out=str(tmp_path / "x.json"), version=1))
    pool = DD.ExtraPool(man, tmp_path / "ui", verify=True)
    assert len(pool.items) == 3

    class Sources:
        def pick(self, rng):
            canvas, smp = pool.draw(rng)
            return "desk_ui_v2", canvas, smp

        def empty_canvas(self, rng):
            return None
    cfg = {"stride": 8, "batch": 4, "scale": (0.42, 1.9), "aug": True, "recipe": "base", "end_ignore": 1.0,
           "e_sources": list(TD.EXACT_SOURCES)}
    x, K, _o, _a1, _a2, _e1, _e2, _wt, _v, _kern, m1, _m2, _mk = TD.make_batch(Sources(), np.random.default_rng(1), cfg)
    assert x.shape[0] == 4 and float(K.sum()) > 0 and bool(m1.any())           # an exact source: end losses on
