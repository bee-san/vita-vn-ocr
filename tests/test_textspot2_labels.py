"""Tests of the textspot2 labels and the enlarged real pool (synth/textspot2_labels.py, issue #106; docs/textspot2.md
"Data" b and c with amendment 1): every label rule on hand-made boxes, the transposed refinement of a vertical box,
the colour rebuild against #97's real_canvases, the meiki call against teacher/engines.Meiki.ocr, the caps of arm c,
and a lab and a more shard through train/textspot_train.FramePool. No screenshots, models or network.
"""
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "train", ROOT / "synth", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textdet_data as DATA  # noqa: E402
import textspot2_labels as L  # noqa: E402
import textspot_train as ST  # noqa: E402
from common import to_gray  # noqa: E402


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mk(raw, score, ref_h=None, ref_t=None):
    """A cached meiki box (both refinements default to the raw box)."""
    return {"raw": list(raw), "score": score, "ref_h": list(ref_h or raw), "ref_t": list(ref_t or raw)}


def build(existing=(), ignore=(), furi=(), meiki=(), winner=(), w=960, h=544, **rules):
    return L.build_labels(w, h, [list(x) for x in existing], [list(x) for x in ignore], [list(x) for x in furi],
                          list(meiki), [list(x) for x in winner], rules or None)


# ----------------------------------------------------------------------------- the pins and the settings

def test_every_pinned_file_is_unchanged():
    """Nothing of #97's, textspot's or the teacher's code that a protocol pins was changed."""
    for proto in ("eval/textspot_protocol.json", "eval/textspot2_protocol.json"):
        p = json.loads((ROOT / proto).read_text(encoding="utf-8"))
        pins = dict(p["sources_sha256"])
        for amend in p.get("amendments", []):
            pins.update(amend.get("sources_sha256", {}))
        for path, digest in pins.items():
            assert sha(ROOT / path) == digest, f"{path} is not as {proto} pins it"


def test_the_rules_are_the_protocols():
    p = json.loads((ROOT / "eval" / "textspot2_protocol.json").read_text(encoding="utf-8"))
    a1 = p["amendments"][0]["settings"]
    assert a1["coverage"] == "union" and a1["meiki"] == {"det_thr": 0.2, "input": "BGR", "recognition": False}
    R = L.RULES
    assert (R["det_thr"], R["pos_min"], R["weight_span"], R["cover"], R["vertical_ratio"]) == (0.2, 0.5, 0.3, 0.5, 1.5)
    assert R["across"] == [6, 96] and R["replace_iou"] == 0.5
    assert L.CAPS_MORE == {"pc": 8, "native": 12, "owner": 200} and L.POOL_SALT == "textspot2-pool:"
    assert L.WINNER_SHA256 == p["references"]["ref"]["blob_sha256"]
    a2 = next(a for a in p["amendments"] if a["id"] == 2)["settings"]
    assert R["winner_only_conf_min"] == a2["winner_only_ignore_conf_min"] == 40
    assert a2["low_score_covered"] == "dropped" and R["low_covered"] == "drop"
    assert (R["refine"], R["cover_box"], R["amendment"]) == ("P1", "refined", 2)


# ----------------------------------------------------------------------------- coverage by the union

def test_coverage_is_by_the_union_of_the_boxes():
    b = [0, 0, 100, 10]
    left, mid = [0, 0, 40, 10], [40, 0, 80, 10]
    assert L.union_cover(b, [left]) == pytest.approx(0.4) and not L.covered(b, [left])
    assert L.union_cover(b, [left, mid]) == pytest.approx(0.8) and L.covered(b, [left, mid])
    assert L.union_cover(b, [[0, 0, 60, 10], [20, 0, 60, 10]]) == pytest.approx(0.6)     # overlap counted once
    assert L.union_cover(b, [[0, 0, 50, 10]]) == pytest.approx(0.5) and L.covered(b, [[0, 0, 50, 10]])
    assert L.union_cover(b, [[200, 0, 300, 10]]) == 0.0 and L.union_cover([5, 5, 5, 9], [b]) == 0.0
    assert L.union_cover(b, [[-50, -5, 150, 20]]) == pytest.approx(1.0)


def test_a_line_meiki_splits_in_two_is_covered_and_not_added():
    ex = [[100, 400, 700, 430, 0, "text"]]
    lines, ignore, st = build(existing=ex, meiki=[mk([100, 400, 380, 430], 0.9), mk([380, 400, 700, 430], 0.9)])
    assert lines == [[100, 400, 700, 430, 0, "text", 1.0]] and ignore == []
    assert st["counts"]["high_covered"] == 2 and st["counts"]["positives_added"] == 0


def test_an_existing_positive_split_in_two_covers_a_meiki_line():
    ex = [[100, 400, 380, 430, 0, "text"], [380, 400, 700, 430, 0, "text"]]
    lines, _, st = build(existing=ex, meiki=[mk([100, 400, 700, 430], 0.95)])
    assert len(lines) == 2 and st["counts"]["high_covered"] == 1


# ----------------------------------------------------------------------------- score bands and weights

def test_score_bands():
    m = [mk([100, 100, 300, 124], 0.21), mk([100, 200, 300, 224], 0.4999), mk([100, 300, 300, 324], 0.5)]
    lines, ignore, st = build(meiki=m)
    assert lines == [[100, 300, 300, 324, 0, "text", 0.5]]
    assert ignore == [[100, 100, 300, 124], [100, 200, 300, 224]] and st["counts"]["ignore_low"] == 2


@pytest.mark.parametrize("score,want", [(0.5, 0.5), (0.56, 0.6), (0.65, 0.75), (0.8, 1.0), (0.95, 1.0)])
def test_weights_from_the_score(score, want):
    assert L.weight_of(score, False) == pytest.approx(want)
    assert L.weight_of(score, True) == 1.0
    lines, _, _ = build(meiki=[mk([100, 300, 300, 324], score)])
    assert lines[0][6] == pytest.approx(round(want, 4))


def test_a_positive_the_winner_agrees_with_weighs_one():
    m = [mk([100, 300, 300, 324], 0.55), mk([100, 400, 300, 424], 0.55), mk([100, 480, 300, 504], 0.55)]
    winner = [[90, 295, 310, 330],                     # all of the first
              [100, 400, 200, 424], [200, 400, 250, 424],   # 75% of the second, by the union of two boxes
              [100, 480, 180, 504]]                    # 40% of the third
    lines, _, st = build(meiki=m, winner=winner)
    assert [ln[6] for ln in lines] == [1.0, 1.0, pytest.approx(0.5833)]
    assert st["weights"] == {"1.0 by winner": 2, "0.5-0.6": 1}


def test_existing_positives_weigh_one_and_keep_their_orientation():
    lines, _, _ = build(existing=[[10, 10, 30, 200, 1, "text"], [100, 100, 400, 124, 0, "text"]])
    assert lines == [[10, 10, 30, 200, 1, "text", 1.0], [100, 100, 400, 124, 0, "text", 1.0]]


# ----------------------------------------------------------------------------- ignore areas

def test_winner_boxes_nothing_covers_become_ignore_areas_from_confidence_40():
    ex = [[100, 100, 400, 124, 0, "text"]]
    m = [mk([100, 300, 300, 324], 0.3), mk([500, 300, 600, 324], 0.3), mk([600, 300, 700, 324], 0.3)]
    winner = [[95, 96, 405, 128, 90],        # an existing positive
              [98, 298, 302, 326, 90],       # a raw meiki box of a low score
              [500, 298, 700, 326, 90],      # the union of two raw meiki boxes
              [700, 450, 900, 480, 40],      # nothing: text meiki missed, confident -> ignore area
              [700, 500, 900, 530, 39]]      # nothing, not confident: left as background (amendment 2)
    lines, ignore, st = build(existing=ex, meiki=m, winner=winner)
    assert st["counts"]["ignore_winner_only"] == 1 and [700, 450, 900, 480] in ignore
    assert st["counts"]["winner_only_background"] == 1 and [700, 500, 900, 530] not in ignore
    assert [95, 96, 405, 128] not in ignore and [500, 298, 700, 326] not in ignore
    # a box without its confidence: amendment 1's letter (an ignore area)
    assert build(winner=[[700, 500, 900, 530]])[1] == [[700, 500, 900, 530]]
    # the winner boxes' cached form carries the confidence
    assert L.winner_xyxy({"winner": [[700, 500, 200, 30, 0, 39]]}) == [[700, 500, 900, 530, 39]]


def test_a_winner_box_a_new_positive_covers_is_not_ignored():
    m = [mk([100, 300, 300, 324], 0.9, ref_h=[104, 302, 296, 322])]
    lines, ignore, _ = build(meiki=m, winner=[[100, 300, 300, 324]])
    assert lines == [[104, 302, 296, 322, 0, "text", 1.0]] and ignore == []


def test_existing_ignore_areas_a_new_positive_replaces_at_iou_half():
    old = [[100, 200, 300, 224], [600, 200, 800, 224]]      # #97's uncovered detector boxes
    m = [mk([98, 198, 302, 226], 0.9, ref_h=[101, 201, 299, 223]),
         mk([600, 190, 700, 234], 0.9, ref_h=[602, 201, 698, 223])]
    lines, ignore, st = build(ignore=old, meiki=m)
    assert L.iou([98, 198, 302, 226], old[0]) >= 0.5 and L.iou([600, 190, 700, 234], old[1]) < 0.5
    assert ignore == [[600, 200, 800, 224]] and st["counts"]["ignore_removed"] == 1
    assert len(lines) == 2


def test_a_low_box_over_an_existing_positive_is_dropped():
    ex = [[100, 100, 400, 124, 0, "text"], [100, 130, 400, 154, 0, "text"]]
    m = [mk([100, 100, 400, 154], 0.3), mk([100, 300, 400, 324], 0.3)]   # two lines' union; one uncovered
    lines, ignore, st = build(existing=ex, meiki=m)
    assert ignore == [[100, 300, 400, 324]] and st["counts"]["low_dropped_covered"] == 1
    assert build(existing=ex, meiki=m, low_covered="ignore")[1] == [[100, 100, 400, 154], [100, 300, 400, 324]]


@pytest.mark.parametrize("h,kind", [(5, "size_small"), (6, None), (96, None), (97, "size_large")])
def test_size_gates_read_the_refined_box(h, kind):
    raw = [100, 100, 500, 100 + h + 8]
    m = [mk(raw, 0.9, ref_h=[102, 104, 498, 104 + h])]
    lines, ignore, st = build(meiki=m)
    if kind is None:
        assert lines == [[102, 104, 498, 104 + h, 0, "text", 1.0]] and ignore == []
    else:
        assert lines == [] and ignore == [[102, 104, 498, 104 + h]] and st["counts"][f"ignore_{kind}"] == 1


def test_a_vertical_box_is_gated_by_its_width():
    m = [mk([100, 100, 105, 300], 0.9), mk([200, 100, 230, 300], 0.9)]
    lines, ignore, _ = build(meiki=m)
    assert lines == [[200, 100, 230, 300, 1, "text", 1.0]] and ignore == [[100, 100, 105, 300]]


# ----------------------------------------------------------------------------- furigana

def test_furigana_geometry_without_the_kana_test():
    host = [100, 120, 500, 150]                         # a 30 px line
    assert L.is_furigana_geom([200, 104, 240, 118], host, [200, 104, 240, 118])      # ruby above it
    assert L.is_furigana_geom([200, 115, 240, 128], host, [200, 115, 240, 128])      # touching its top
    assert not L.is_furigana_geom([200, 160, 240, 174], host, [200, 160, 240, 174])  # below it
    assert not L.is_furigana_geom([520, 104, 560, 118], host, [520, 104, 560, 118])  # beside it
    assert not L.is_furigana_geom([200, 90, 240, 118], host, [200, 90, 240, 118])    # too tall (28 > 19.5)
    assert not L.is_furigana_geom([200, 60, 240, 74], host, [200, 60, 240, 74])      # too far above


def test_furigana_becomes_an_ignore_area_by_geometry_or_by_the_record():
    host = mk([100, 120, 500, 150], 0.9)
    ruby = mk([200, 104, 240, 118], 0.9)
    lines, ignore, st = build(meiki=[ruby, host])
    assert lines == [[100, 120, 500, 150, 0, "text", 1.0]] and ignore == [[200, 104, 240, 118]]
    assert st["counts"]["ignore_furigana_geometry"] == 1
    # an existing positive is a host too
    lines, ignore, st = build(existing=[[100, 120, 500, 150, 0, "text"]], meiki=[ruby])
    assert len(lines) == 1 and ignore == [[200, 104, 240, 118]]
    # the record's furigana boxes, alone, anywhere
    far = mk([700, 300, 740, 314], 0.9)
    assert build(meiki=[far])[0][0][:4] == [700, 300, 740, 314]
    lines, ignore, st = build(meiki=[far], furi=[[698, 299, 742, 315]])
    assert lines == [] and ignore == [[700, 300, 740, 314]] and st["counts"]["ignore_furigana_record"] == 1


def test_a_vertical_column_is_not_tested_as_furigana():
    host = mk([100, 120, 500, 300], 0.9)              # a tall host: a column beside it would pass the geometry
    col = mk([200, 100, 215, 140], 0.9)
    lines, _, _ = build(meiki=[col, host])
    assert [200, 100, 215, 140, 1, "text", 1.0] in lines


# ----------------------------------------------------------------------------- the refinement

def column_image():
    """A black 200 x 200 image with a white column of five 16 x 16 glyphs, 6 px apart: ink [90, 20, 106, 124]."""
    img = np.zeros((200, 200, 3), np.uint8)
    for k in range(5):
        img[20 + 22 * k:36 + 22 * k, 90:106] = 255
    return img


def test_a_vertical_box_is_refined_on_the_transposed_crop():
    img, raw = column_image(), [84, 14, 112, 130]
    rh, rt = L.refine_h(img, raw), L.refine_t(img, raw)
    assert rt == [90, 20, 106, 124]
    assert rh != rt                                    # the horizontal refinement cannot follow a column
    for proc in ("P1", "P2"):
        assert L.refined_of({"raw": raw, "ref_h": rh, "ref_t": rt}, proc) == ([90, 20, 106, 124], True)
    lines, _, _ = build(meiki=[{"raw": raw, "score": 0.9, "ref_h": rh, "ref_t": rt}], w=200, h=200)
    assert lines == [[90, 20, 106, 124, 1, "text", 1.0]]


def test_the_transposed_refinement_is_the_refinement_of_the_transpose():
    rng = np.random.default_rng(1)
    for _ in range(20):
        img = np.full((120, 160, 3), int(rng.integers(0, 60)), np.uint8)
        x0, y0 = int(rng.integers(5, 60)), int(rng.integers(5, 40))
        x1, y1 = x0 + int(rng.integers(20, 90)), y0 + int(rng.integers(8, 30))
        img[y0:y1, x0:x1] = 230
        raw = [x0 - 3, y0 - 2, x1 + 4, y1 + 3]
        t = np.ascontiguousarray(np.transpose(img, (1, 0, 2)))
        r = L.refine_h(t, [raw[1], raw[0], raw[3], raw[2]])
        assert L.refine_t(img, raw) == [r[1], r[0], r[3], r[2]]
        assert L.refine_h(img, raw) == [x0, y0, x1, y1]


def test_p1_and_p2_differ_only_in_the_order_of_the_decision():
    m = {"raw": [0, 0, 20, 60], "ref_h": [0, 0, 20, 25], "ref_t": [2, 0, 18, 60]}   # raw tall, refined not
    assert L.refined_of(m, "P1") == ([0, 0, 20, 25], False)
    assert L.refined_of(m, "P2") == ([2, 0, 18, 60], True)
    m = {"raw": [0, 0, 40, 30], "ref_h": [0, 0, 10, 30], "ref_t": [0, 0, 12, 30]}   # raw wide, refined tall
    assert L.refined_of(m, "P1") == ([0, 0, 12, 30], True)
    assert L.refined_of(m, "P2") == ([0, 0, 10, 30], True)


# ----------------------------------------------------------------------------- the colour rebuild

def write_png(path, h, w, seed):
    rng = np.random.default_rng(seed)
    img = (rng.random((h, w, 3)) * 255).astype(np.uint8)
    img = cv2.GaussianBlur(img, (0, 0), 1.5)
    cv2.imwrite(str(path), img[:, :, ::-1])
    return path


@pytest.mark.parametrize("source,kind,size,dims", [
    ("realplus", "vita", (600, 800), (725, 544)),       # PC read at the Vita target: down
    ("realplus", "vita", (360, 480), (725, 544)),       # up
    ("owner", "psp", (272, 480), (480, 272)),           # the same size
    ("devnative", "vita", (544, 960), (960, 544)),
    ("teacher", "psp", (272, 480), None),               # native: its own size
    ("teacher", "hires", (720, 1280), None),            # PC: the Vita target, and for some the PSP target
])
def test_the_colour_source_is_made_as_real_canvases_makes_the_canvas(tmp_path, source, kind, size, dims):
    p = write_png(tmp_path / "v1_1.png", *size, seed=size[0])
    r = {"shot": "v1_1", "vn": "v1", "kind": kind, "path": str(p), "lines": [], "dets": []}
    if dims:
        r["dims"] = list(dims)
    seen = set()
    for seed in range(6):
        for grey, _, _, meta in DATA.real_canvases(r, source, np.random.default_rng(seed)):
            rgb, k, plan = L.colour_source(r, source, meta["target"])
            assert rgb.shape[:2] == grey.shape and np.array_equal(to_gray(rgb), grey)
            assert plan[0] == meta["target"] and round(k, 4) == meta["scale"] or source != "teacher"
            seen.add(meta["target"])
    if kind == "hires":
        assert seen == {"vita", "psp"}


def test_record_furigana_is_scaled_as_real_canvases_scales_boxes():
    r = {"lines": [{"ink": [10, 20, 100, 40], "furi": [[20, 10, 50, 19], [0, 0, 0, 5]]}, {"ink": [1, 1, 2, 2]}]}
    assert L.record_furi(r, 0.5, 400, 300) == [DATA._scaled([20, 10, 50, 19], 0.5, 400, 300)]
    assert L.record_furi(None, 1.0, 10, 10) == []


# ----------------------------------------------------------------------------- the meiki call

class _StubSession:
    def __init__(self, boxes, scores):
        self.boxes, self.scores = boxes, scores
        self.seen = None

    def _preprocess_for_detection(self, image):
        self.seen = image
        return np.zeros((1, 3, 544, 960), np.float32), 1.0

    def _run_detection_inference(self, tensor, scale):
        return [None, self.boxes[None], self.scores[None]]

    def _process_recognition_pipeline(self, *a, **k):      # no recognition
        return None


def test_meiki_detect_is_meiki_ocr_without_its_recognition():
    from teacher import engines as E
    boxes = np.array([[10.7, 300.2, 200.9, 320.5], [5, 100, 50, 130], [-4, 20.5, 70, 44], [900, 500, 990, 560],
                      [1, 1, 3, 3], [10, 100, 60, 128]], np.float32)
    scores = np.array([0.9, 0.2, 0.2000001, 0.55, 0.1, 0.35], np.float32)
    meiki = E.Meiki.__new__(E.Meiki)
    meiki.m, meiki.punct, meiki.det_thr, meiki.rec_thr, meiki.rgb = _StubSession(boxes, scores), 0.2, 0.2, 0.1, False
    bgr = np.zeros((544, 960, 3), np.uint8)
    got = L.meiki_detect(meiki, bgr)
    assert meiki.m.seen is bgr                           # fed as given: BGR (Meiki.rgb is False)
    ref = E.Meiki.ocr(meiki, bgr)
    assert [b for b, _ in got] == [o["det"] for o in ref]
    assert [round(s, 4) for _, s in got] == [o["det_conf"] for o in ref]
    assert [b for b, _ in got] == [[-0 + 0, 20, 70, 44], [10, 100, 60, 128], [10, 300, 200, 320], [900, 500, 960, 544]]


# ----------------------------------------------------------------------------- arm c's selection

def test_the_caps_count_existing_screenshots_and_take_the_hash_order():
    cand = {("pc", "v1"): {f"v1_{i}": None for i in range(10)}, ("pc", "v2"): {"v2_1": None},
            ("native", "v3"): {f"v3_{i}": None for i in range(20)}, ("owner", "v4"): {f"v4_d{i}": None for i in
                                                                                    range(250)}}
    existing = {("pc", "v1"): {"v1_a", "v1_b", "v1_c"}, ("pc", "v2"): {f"v2_x{i}" for i in range(9)},
                ("native", "v3"): {"v3_a"}}
    picked, groups = L.cap_select(cand, existing)
    by = {}
    for g, vn, shot in picked:
        by.setdefault((g, vn), []).append(shot)
    assert by[("pc", "v1")] == sorted(cand[("pc", "v1")], key=L.pool_rank)[:5]
    assert ("pc", "v2") not in by and len(by[("native", "v3")]) == 11 and len(by[("owner", "v4")]) == 200
    assert groups["pc"]["capped_out"] == 5 + 1 and groups["owner"]["new"] == 200
    assert L.pool_rank("v1_2") == hashlib.sha256(b"textspot2-pool:v1_2").hexdigest()


def test_groups_folders_and_ids():
    assert L.group_of("realplus", {"kind": "vita"}) == "pc"
    assert L.group_of("teacher", {"kind": "hires"}) == "pc" and L.group_of("teacher", {"kind": "psp"}) == "native"
    assert L.group_of("devnative", {"kind": "vita"}) == "native" and L.group_of("owner", {}) == "owner"
    assert L.owner_test_folder({"path": "/x/owner-games/vita-foo-v1-TEST/a.png"})
    assert not L.owner_test_folder({"path": "/x/owner-games/vita-foo-v1/a.png"})
    assert L.image_id({"url": "https://t.vndb.org/sf/79/35279.jpg", "shot": "v1_9"}) == "35279"
    assert L.image_id({"shot": "v8985_35279"}) == "35279" and L.image_id({"shot": "v1_d2026_1"}) is None
    assert L.more_target("teacher", {"kind": "hires"}) == "vita" and L.more_target("owner", {"kind": "psp"}) == "psp"


# ----------------------------------------------------------------------------- shards through FramePool

def canvas(h, w, seed):
    return (np.random.default_rng(seed).random((h, w)) * 255).astype(np.uint8)


def test_a_lab_and_a_more_shard_load_in_framepool(tmp_path, monkeypatch):
    # #97's shard: two canvases (one on a skip list)
    d97 = tmp_path / "textdet" / "realplus"
    d97.mkdir(parents=True)
    old = [{"canvas": canvas(272, 480, 0), "lines": [[10, 10, 200, 30, 0, "text"]], "ignore": [[300, 10, 400, 30]],
            "meta": {}, "src": "realplus", "kind": "psp", "vn": "v1", "shot": "v1_10"},
           {"canvas": canvas(544, 960, 1), "lines": [[50, 400, 900, 430, 0, "text"]], "ignore": [], "meta": {},
            "src": "realplus", "kind": "vita", "vn": "v9", "shot": "v9_90"}]
    DATA.write_shard(d97 / "realplus_0000", old)
    _, samples = DATA.load_shard(d97 / "realplus_0000")
    # the lab shard: the kept canvas with new labels, offsets into #97's npy, the npy a symlink
    lab = tmp_path / "shards" / "lab" / "lab_realplus"
    lab.mkdir(parents=True)
    s = samples[0]
    lines, ignore, _ = L.build_labels(s["w"], s["h"], s["lines"], s["ignore"], [],
                                      [mk([320, 8, 380, 32], 0.65, ref_h=[300, 10, 400, 30])], [])
    assert lines[1][6] == 0.75 and ignore == []
    L.jdump({"samples": [{**{k: s[k] for k in ("off", "h", "w", "src", "kind", "vn", "shot")}, "lines": lines,
                          "ignore": ignore, "meta": {}}]}, lab / "realplus_0000.json")
    L.link_npy(lab / "realplus_0000.npy", d97 / "realplus_0000.npy")
    L.link_npy(lab / "realplus_0000.npy", d97 / "realplus_0000.npy")          # idempotent
    # a more shard: a new canvas, written
    more = tmp_path / "shards" / "more" / "more_realplus"
    more.mkdir(parents=True)
    DATA.write_shard(more / "more_realplus_0000", [
        {"canvas": canvas(544, 725, 2), "lines": [[40, 300, 600, 330, 0, "text", 0.8333], [650, 50, 680, 400, 1,
                                                                                          "text", 1.0]],
         "ignore": [[10, 10, 50, 20]], "meta": {}, "src": "realplus", "kind": "vita", "vn": "v2", "shot": "v2_20"}])
    entries = [L.shard_entry("lab", "realplus", lab / "realplus_0000.json"),
               L.shard_entry("more", "realplus", more / "more_realplus_0000.json")]
    assert entries[0]["npy_sha256"] == sha(d97 / "realplus_0000.npy") and entries[0]["npy_symlink_to"]
    assert entries[1]["vertical"] == 1 and entries[1]["lines"] == 2
    _, _, skip = L.textspot_skip()
    doc = {"kind": "textspot2-shards", "shards": entries, "real": {"measured": {"skip": {
        "eval_screenshot": {"skip_shots": ["v1_10"], "shots": 0, "canvases": 0, "by_source": {"realplus": 0}},
        "linefind_list_vn": {"skip_vns": ["v7"], "shots": 0, "canvases": 0, "by_source": {"realplus": 0}}}}}}
    assert set(skip) >= {"eval_screenshot", "linefind_list_vn"}
    man = tmp_path / "textspot2_shards_v1.json"
    man.write_text(json.dumps(doc))
    shots, vns, lists = ST.skip_lists([man])
    assert shots == {"v1_10"} and lists[0]["skip_shots_sha256"] == L.list_sha256(["v1_10"])
    pool = ST.FramePool([man], [lab, more], verify=True, skip=(shots, vns))
    assert len(pool.items["realplus"]) == 1                       # the lab canvas is on the skip list
    pool = ST.FramePool([man], [lab, more], verify=True)
    got = sorted((smp["shot"], len(smp["lines"][0])) for _, smp in pool.items["realplus"])
    assert got == [("v1_10", 7), ("v2_20", 7)]
    d, smp = next((d, x) for d, x in pool.items["realplus"] if x["shot"] == "v1_10")
    assert np.array_equal(pool.canvas(d, smp), old[0]["canvas"])
    pos, ign = ST.at_scale(smp["lines"], smp["ignore"], 2)        # the trainer reads lines[:5] and ignore
    assert pos[1] == [150, 5, 200, 15, 0] and ign == []
    batch = ST.make_batch(pool, np.random.default_rng(0), 4, "h2")
    assert batch[0].shape == (ST.BATCH, 1, ST.WIN_H, ST.WIN_W)
    # the lab runs read only the lab_* directories: a more shard is not read then
    assert ST.FramePool([man], [lab]).skipped == {f"{man}:more_realplus": 1}


def test_the_manifest_skip_block_is_read_by_the_trainer(tmp_path, monkeypatch):
    lab = tmp_path / "lab_realplus"
    lab.mkdir()
    DATA.write_shard(lab / "realplus_0000", [{"canvas": canvas(40, 60, 3), "lines": [], "ignore": [], "meta": {},
                                              "src": "realplus", "kind": "vita", "vn": "v5", "shot": "v5_50"}])
    monkeypatch.setattr(L, "shard_dir", lambda arm, source: lab)
    block = L.skip_block([{"arm": "lab", "source": "realplus", "name": "realplus_0000"}])
    man = tmp_path / "m.json"
    man.write_text(json.dumps({"kind": "textspot2-shards", "shards": [], "real": {"measured": {"skip": block}}}))
    shots, vns, lists = ST.skip_lists([man])
    _, _, skip = L.textspot_skip()
    assert shots == set(skip["eval_screenshot"]["skip_shots"]) and vns == set(skip["linefind_list_vn"]["skip_vns"])
    assert lists[0]["skip_shots_sha256"] == block["eval_screenshot"]["skip_shots_sha256"]
    assert block["eval_screenshot"]["canvases"] == 0 and block["linefind_list_vn"]["by_source"]["realplus"] == 0
    assert block["eval_screenshot"]["textspot_counts"]["canvases"] == skip["eval_screenshot"]["canvases"]
