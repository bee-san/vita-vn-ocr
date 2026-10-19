"""Tests of the full-frame text spotter's twin (synth/textspot_twin.py: the VTSP blob, the 2x2 halving, the frame
decode, blocks, reading order and pick) and of its frozen metrics (eval/textspot_metrics.py: padding, matching,
assignment, end-to-end CER, the VN-cluster intervals and the selection rule), plus the protocol's pins
(eval/textspot_protocol.json). No screenshots, fonts or models; no network.
"""
import hashlib
import json
import struct
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "synth", ROOT / "train", ROOT / "eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import textspot_metrics as M  # noqa: E402

import textdet_export as E  # noqa: E402
import textspot_twin as S  # noqa: E402

PROTOCOL = ROOT / "eval" / "textspot_protocol.json"


def random_vtdt(shape="s8_c8", seed=1):
    shape_d, q = E.random_int_layers(shape, seed)
    return E.pack(shape_d, q)


def maps_of(hm, wm):
    m = np.zeros((4, hm, wm), np.int64)
    m[0] = -10
    m[1] = -10
    return m


def fake(stride=4, min_cells=1, conf_min=1):
    return SimpleNamespace(stride=stride, min_cells=min_cells, conf_min=conf_min)


# ---------------------------------------------------------------- the blob

def test_pack_round_trip():
    vtdt = random_vtdt()
    blob = S.pack(vtdt, half_above=600, conf_min=3)
    m = S.Model(blob)
    assert (m.half_above, m.conf_min, m.blob_bytes) == (600, 3, len(vtdt) + 32)
    assert m.stride == 8 and m.net.blob_bytes == len(vtdt)


@pytest.mark.parametrize("edit", ["magic", "version", "size", "crc", "conf0", "offset", "reserved", "inner"])
def test_loader_rejects(edit):
    blob = bytearray(S.pack(random_vtdt(), 600, 1))

    def fix_crc(b):
        struct.pack_into("<I", b, 12, zlib.crc32(bytes(b[16:])) & 0xFFFFFFFF)

    if edit == "magic":
        blob[0] = ord("X")
    elif edit == "version":
        struct.pack_into("<I", blob, 4, 2)
    elif edit == "size":
        blob += b"\0"
    elif edit == "crc":
        blob[-1] ^= 1
    elif edit == "conf0":
        blob[18] = 0
        fix_crc(blob)
    elif edit == "offset":
        struct.pack_into("<I", blob, 20, 36)
        fix_crc(blob)
    elif edit == "reserved":
        blob[19] = 1
        fix_crc(blob)
    elif edit == "inner":
        blob[32] = ord("X")
        fix_crc(blob)
    with pytest.raises(ValueError):
        S.Model(bytes(blob))


def test_scale_of():
    m = S.Model(S.pack(random_vtdt(), 600, 1))
    assert S.scale_of(m, 960) == 2 and S.scale_of(m, 725) == 2 and S.scale_of(m, 480) == 1 and S.scale_of(m, 600) == 1
    m0 = S.Model(S.pack(random_vtdt(), 0, 1))
    assert S.scale_of(m0, 960) == 1


def test_halve_exact():
    g = np.array([[0, 10, 20], [30, 40, 50], [60, 70, 255]], np.uint8)
    h = S.halve(g)
    assert h.shape == (2, 2)
    assert h[0, 0] == (0 + 10 + 30 + 40 + 2) >> 2
    assert h[0, 1] == (20 + 20 + 50 + 50 + 2) >> 2          # the last column repeats
    assert h[1, 0] == (60 + 70 + 60 + 70 + 2) >> 2          # the last row repeats
    assert h[1, 1] == 255
    g2 = np.random.default_rng(0).integers(0, 256, (544, 960)).astype(np.uint8)
    assert S.halve(g2).shape == (272, 480)


# ---------------------------------------------------------------- the decode on hand-made maps

def horizontal_maps():
    m = maps_of(10, 30)
    m[0, 2, 3:9] = 20                      # kernel row 2, cells 3..8
    m[1, 2, 3:9] = -5
    m[2, 2, 3:9] = 12                      # 6 px to either edge
    m[3, 2, 3:9] = 12
    return m


def test_horizontal_line_f1():
    c = S.candidates(fake(4), horizontal_maps(), 1, 40, 60)
    assert len(c) == 1
    key, box = c[0]
    # c2 = 2*4*2 + 3 = 19; near = floor(6*7 / 12) = 3; far = ceil(6*31 / 12) = 16; across 13, pad 2
    assert box == (10, 1, 28, 17, 0, 20)
    assert key == (-20, -6, 0, 1, 10, 17, 28)


def test_horizontal_line_f2():
    _, box = S.candidates(fake(4), horizontal_maps(), 2, 80, 100)[0]
    # near = floor(2*42 / 12) = 7; far = ceil(2*186 / 12) = 31; across 24, pad 4; along [24, 72)
    assert box == (20, 3, 56, 32, 0, 20)


def test_vertical_column():
    m = maps_of(30, 10)
    m[0, 3:12, 4] = 30
    m[1, 3:12, 4] = 9
    m[2, 3:12, 4] = 10
    m[3, 3:12, 4] = 10
    _, box = S.candidates(fake(4), m, 1, 120, 40)[0]
    # c2 = 2*4*4 + 3 = 35 across columns: near = floor(9*25 / 18) = 12, far = ceil(9*45 / 18) = 23; pad 2;
    # along rows [12, 48)
    assert box == (10, 10, 15, 40, 1, 30)


def test_min_across_scales_with_f():
    m = maps_of(10, 30)
    m[0, 2, 3:9] = 20
    m[2, 2, 3:9] = 2                        # 1 px either side: across 3 at f=1 (< 4), 4 at f=2 (< 8): dropped
    m[3, 2, 3:9] = 2
    assert S.candidates(fake(4), m, 1, 40, 60) == []
    assert S.candidates(fake(4), m, 2, 80, 120) == []
    m[2, 2, 3:9] = 6
    m[3, 2, 3:9] = 6
    assert len(S.candidates(fake(4), m, 1, 40, 60)) == 1 and len(S.candidates(fake(4), m, 2, 80, 120)) == 1


def test_conf_and_conf_min():
    m = horizontal_maps()
    m[0, 2, 3:9] = [1, 2, 3, 4, 5, 6]       # sum 21 over 6 cells: conf 3
    assert S.candidates(fake(4, conf_min=3), m, 1, 40, 60)[0][1][5] == 3
    assert S.candidates(fake(4, conf_min=4), m, 1, 40, 60) == []


def test_min_cells():
    m = maps_of(10, 30)
    m[0, 2, 3] = 20
    m[2:4, 2, 3] = 12
    assert S.candidates(fake(4, min_cells=2), m, 1, 40, 60) == []
    assert len(S.candidates(fake(4, min_cells=1), m, 1, 40, 60)) == 1


def test_negative_sums_floor_ceil_and_clip():
    m = maps_of(10, 30)
    m[0, 0, 3:9] = 20                       # row 0: c2 = 3; A1 = 20 -> near = floor(6*(3-20)/12) = floor(-8.5) = -9
    m[2, 0, 3:9] = 20
    m[3, 0, 3:9] = 5                        # far = ceil(6*8/12) = 4
    _, box = S.candidates(fake(4), m, 1, 40, 60)[0]
    assert box[1] == 0 and box[3] == 4 + 2  # near - pad clipped to 0; far + pad = 6


def test_cap_keeps_highest_conf():
    m = maps_of(140, 30)
    for k in range(70):
        m[0, 2 * k, 2:8] = 1 + k            # 70 components, conf 1..70
        m[2:4, 2 * k, 2:8] = 6
    c = S.candidates(fake(4), m, 1, 560, 120)
    assert len(c) == 70
    kept = [b for _, b in sorted(c)[:S.BOX_MAX]]
    assert sorted(b[5] for b in kept) == list(range(7, 71))


# ---------------------------------------------------------------- blocks, order and pick

def test_dialogue_block_and_ui():
    boxes = [(100, 400, 600, 30, 0, 50, 0), (100, 440, 500, 30, 0, 50, 0),     # two dialogue lines
             (880, 10, 60, 16, 0, 40, 0)]                                       # a UI label, top right
    out, blocks, pick = S.group_and_order(boxes, 544)
    assert len(blocks) == 2
    assert blocks[0][:2] == (880, 10) and blocks[1][:5] == (100, 400, 600, 70, 0)
    assert [b[1] for b in out] == [10, 400, 440] and [b[-1] for b in out] == [0, 1, 1]
    assert pick == 1


def test_side_by_side_menu_and_heights():
    boxes = [(10, 500, 50, 20, 0, 9, 0), (80, 500, 50, 20, 0, 9, 0),           # SAVE  LOAD: no overlap along
             (300, 100, 200, 20, 0, 9, 0), (300, 125, 200, 40, 0, 9, 0)]       # heights 20 / 40: never one block
    out, blocks, pick = S.group_and_order(boxes, 544)
    assert len(blocks) == 4
    assert [b[:2] for b in blocks] == [(300, 100), (300, 125), (10, 500), (80, 500)]


def test_vertical_columns_right_to_left():
    cols = [(600, 50, 30, 300, 1, 9, 0), (640, 50, 30, 300, 1, 9, 0), (680, 60, 30, 200, 1, 9, 0)]
    out, blocks, pick = S.group_and_order(cols, 544)
    assert len(blocks) == 1 and [b[0] for b in out] == [680, 640, 600]
    # two vertical blocks side by side in one row: right to left
    out2, blocks2, _ = S.group_and_order([(100, 50, 30, 200, 1, 9, 0), (700, 60, 30, 200, 1, 9, 0)], 544)
    assert [b[0] for b in blocks2] == [700, 100]


def test_horizontal_rows_inside_block():
    # a line broken into two boxes on one row, then a second line
    boxes = [(300, 400, 200, 30, 0, 9, 0), (100, 402, 180, 30, 0, 9, 0), (100, 440, 400, 30, 0, 9, 0)]
    out, blocks, _ = S.group_and_order(boxes, 544)
    assert len(blocks) == 1
    assert [(b[0], b[1]) for b in out] == [(100, 402), (300, 400), (100, 440)]


def test_pick_lower_half_and_ties():
    upper = (100, 50, 400, 30, 0, 9, 0)
    lower = (100, 400, 300, 30, 0, 9, 0)        # smaller, but doubled in the lower half
    _, blocks, pick = S.group_and_order([upper, lower], 544)
    assert blocks[pick][1] == 400
    _, _, pick_empty = S.group_and_order([], 544)
    assert pick_empty == -1


def test_order_independent_of_input_order():
    rng = np.random.default_rng(3)
    boxes = [(int(rng.integers(0, 900)), int(rng.integers(0, 500)), int(rng.integers(8, 200)),
              int(rng.integers(8, 40)), int(rng.integers(0, 2)), 5, 0) for _ in range(40)]
    ref = S.group_and_order(boxes, 544)
    for _ in range(5):
        perm = [boxes[i] for i in rng.permutation(len(boxes))]
        assert S.group_and_order(perm, 544) == ref


def test_spot_end_to_end_random_blob():
    m = S.Model(S.pack(random_vtdt("s4_c8", 2), 600, 1))
    rng = np.random.default_rng(5)
    for w, h in ((960, 544), (480, 272), (725, 544)):
        g = np.full((h, w), 40, np.uint8)
        for _ in range(12):
            y0, x0 = int(rng.integers(0, h - 30)), int(rng.integers(0, w - 200))
            g[y0:y0 + 20, x0:x0 + 180:3] = 230
        r = S.spot(m, g)
        assert len(r.boxes) <= S.BOX_MAX
        for b in r.boxes:
            assert 0 <= b.x < b.x + b.w <= w and 0 <= b.y < b.y + b.h <= h and 1 <= b.conf <= 127
        assert sum(bk.count for bk in r.blocks) == len(r.boxes)
        assert (r.pick == -1) == (len(r.boxes) == 0)
        rgb = np.repeat(g[..., None], 3, 2)
        assert S.spot(m, rgb) == r


# ---------------------------------------------------------------- metrics

def ref(box, cls="dialogue", text="あいう", vertical=False, ignore=False):
    return {"box": list(box), "cls": cls, "text": text, "vertical": vertical, "ignore": ignore}


def det(box, text="", vertical=False):
    return {"box": list(box), "text": text, "vertical": vertical}


def test_pad_and_iou():
    assert M.pad_box([10, 20, 100, 20], False, 960, 544) == [6, 16, 108, 28]
    assert M.pad_box([0, 0, 30, 200], True, 40, 544) == [0, 0, 35, 205]
    assert M.iou([0, 0, 10, 10], [5, 0, 10, 10]) == pytest.approx(50 / 150)


def test_match_frame():
    refs = [ref([100, 100, 200, 20]), ref([100, 200, 100, 20], cls="ui"), ref([500, 300, 60, 10], ignore=True)]
    dets = [det([96, 96, 208, 28]), det([700, 10, 50, 20]), det([498, 298, 64, 14])]
    m = M.match_frame(dets, refs, 960, 544)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 1)
    assert m["tp_cls"]["dialogue"] == 1 and m["fn_cls"]["ui"] == 1 and m["left_out"] == [2]
    assert m["pairs"][0][2] == pytest.approx(1.0) and m["pairs"][0][3] == pytest.approx(1.0)


def test_assign_merge_fragments_and_junk():
    refs = [ref([100, 100, 200, 20], text="あいうえお"), ref([100, 130, 200, 20], text="かきくけこ")]
    merged = det([96, 96, 208, 58], text="あいうえおかきくけこ")         # half in each line: junk
    a = M.assign([merged], refs, 960, 544)
    assert a == [None]
    e = M.e2e_frame([merged], refs, 960, 544)
    assert e["all"] == (10 + 10, 10) and e["junk"] == 10
    frags = [det([200, 96, 104, 28], text="うえお"), det([96, 96, 104, 28], text="あい"), det([96, 126, 208, 28],
             text="かきくけこ")]
    e2 = M.e2e_frame(frags, refs, 960, 544)
    assert e2["all"] == (0, 10) and e2["per_ref"] == [0, 0]
    e3 = M.e2e_frame([], refs, 960, 544)
    assert e3["all"] == (10, 10) and e3["dialogue"] == (10, 10)


def test_e2e_dialogue_and_ignore():
    refs = [ref([100, 100, 200, 20], text="あいう"), ref([10, 10, 60, 16], cls="ui", text="SAVE"),
            ref([100, 80, 100, 8], ignore=True, text=None)]
    dets = [det([96, 96, 208, 28], text="あいう"), det([6, 6, 68, 24], text="SAVF"), det([100, 79, 100, 9],
            text="ふり"), det([600, 400, 50, 20], text="ゴミ")]
    e = M.e2e_frame(dets, refs, 960, 544)
    assert e["all"] == (1 + 2, 7) and e["dialogue"] == (0, 3) and e["junk"] == 2


def units(errs, chars, vns):
    return [{"unit": f"u{i}", "vn": v, "errors": e, "chars": c} for i, (e, c, v) in enumerate(zip(errs, chars, vns))]


def test_vn_diff_and_f1_diff():
    vns = [f"v{i % 10}" for i in range(40)]
    base = units([5] * 40, [50] * 40, vns)
    better = units([2] * 40, [50] * 40, vns)
    d = M.vn_diff(better, base)
    assert d["delta"] == pytest.approx(-6.0) and d["ci95"][1] < 0
    same = M.vn_diff(base, base)
    assert same["delta"] == 0 and same["ci95"] == [0.0, 0.0]
    cb = [{"unit": f"u{i}", "vn": v, "tp": 5, "fp": 3, "fn": 3} for i, v in enumerate(vns)]
    cc = [{"unit": f"u{i}", "vn": v, "tp": 7, "fp": 1, "fn": 1} for i, v in enumerate(vns)]
    f = M.f1_diff(cc, cb)
    assert f["delta"] == pytest.approx(7 / 8 - 5 / 8) and f["ci95"][0] > 0
    with pytest.raises(ValueError):
        M.vn_diff(better[:-1], base)


def arm(tp, fp, fn, err, chars=50, n=40, dlg_err=None):
    vns = [f"v{i % 10}" for i in range(n)]
    return {"counts": [{"unit": f"u{i}", "vn": v, "tp": tp, "fp": fp, "fn": fn} for i, v in enumerate(vns)],
            "all": units([err] * n, [chars] * n, vns),
            "dialogue": units([err if dlg_err is None else dlg_err] * n, [chars] * n, vns)}


def cand(name, bytes_, err, tp=8, ms=100, arena=60000, c_equal=True, decode="det", dlg_err=None):
    return {"name": name, "decode": decode, "blob_bytes": bytes_, "macs_960": bytes_ * 1000, "arena_960": arena,
            "est_ms_333_960": ms, "c_equal": c_equal, "arm": arm(tp, 10 - tp, 10 - tp, err, dlg_err=dlg_err)}


RULE = {"arena_max": 131072, "ms_max_333_960": 150}


def test_rule_smallest_qualifying_and_replacement():
    baselines = {"grad": arm(5, 5, 5, 10), "s8c24": arm(6, 4, 4, 8)}
    cands = [cand("a", 1000, 9),                        # CER not resolved below s8c24 (9 > 8)
             cand("b", 2000, 6),                        # qualifies
             cand("c", 3000, 6),                        # qualifies, but no better than b
             cand("d", 4000, 3),                        # qualifies and is resolved better than b
             cand("e", 500, 1, ms=200),                 # ineligible: time
             cand("f", 600, 1, c_equal=False)]          # ineligible: C != twin
    r = M.apply_rule(cands, baselines, RULE)
    assert r["eligible"] == ["a", "b", "c", "d"]
    assert r["qualifying"] == ["b", "c", "d"]
    assert r["winner"] == "d"
    assert [(s["replaces"], s["by"]) for s in r["steps"]] == [("b", "d")]


def test_rule_dialogue_guard_and_no_winner():
    baselines = {"grad": arm(5, 5, 5, 10, dlg_err=2)}
    worse_dialogue = cand("x", 1000, 5, dlg_err=6)      # better overall, resolved worse on dialogue
    r = M.apply_rule([worse_dialogue], baselines, RULE)
    assert r["qualifying"] == [] and r["winner"] is None and r["report_candidate"] == "x"


# ---------------------------------------------------------------- the protocol's pins

def sha(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def test_protocol_pins_frozen_sources():
    p = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    pins = dict(p["sources_sha256"])
    for amend in p.get("amendments", []):
        pins.update(amend.get("sources_sha256", {}))
    for path, digest in pins.items():
        assert sha(path) == digest, f"{path} changed after the protocol froze it (an amendment must re-pin it)"
    assert p["rule"]["arena_max"] == 131072 and p["rule"]["ms_max_333_960"] == 150
    assert set(p["baselines"]) >= {"grad", "grad_refine", "grad_refine_gate", "s8c24_textdet", "s8c24_textdet_refine",
                                   "s8c24_spot", "columns", "grad_refine_columns"}
