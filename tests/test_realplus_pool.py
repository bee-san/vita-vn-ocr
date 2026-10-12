"""realplus/pool.py: review tiers, the review plan, PSP-scale geometry, the repeat caps and the npz build
for PC / console screenshots read at Vita scale. No network, screenshots or shared artifacts."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for d in ("realplus", "synth", ""):
    sys.path.insert(0, str(ROOT / d) if d else str(ROOT))

import pool as PL  # noqa: E402

from devnative import decide as D  # noqa: E402
from devnative import sheets as S  # noqa: E402


def _line(k, ink, reads, furi=None):
    m, mv, v, vc = reads
    ln = {"k": k, "det": list(ink), "det_conf": 0.9, "parts": [list(ink)], "ink": list(ink), "crop_ink": list(ink),
          "edge": False, "hint": {"bright_frac": 0.9},
          "reads": {"meiki": [m, [[c, ink[0] + 12 * i, ink[1], ink[0] + 12 * i + 10, ink[3], 0.9]
                                  for i, c in enumerate(m)]],
                    "meikiv": [mv, 0.9, 0.8, []], "v6s": [v, 0.9, 0.8], "v6sc": [vc, 0.9, 0.8]}}
    if furi:
        ln["furi"], ln["furi_text"] = [furi], ["ふり"]
        ln["crop_ink"] = [ink[0], furi[1], ink[2], ink[3]]
    return ln


def test_tiers():
    assert PL.tier(S.candidates(_line(0, [0, 0, 9, 9], ("あい", "あい", "あい", "あい")))) == "auto"
    assert PL.tier(S.candidates(_line(0, [0, 0, 9, 9], ("あい", "あい", "あい", "あぃ")))) == "t1"
    assert PL.tier(S.candidates(_line(0, [0, 0, 9, 9], ("あい", "かい", "あい", "かい")))) == "t1"   # both families
    assert PL.tier(S.candidates(_line(0, [0, 0, 9, 9], ("あい", "あい", "かい", "かい")))) == "t2"   # one each
    assert PL.tier(S.candidates(_line(0, [0, 0, 9, 9], ("あい", "あい", "かい", "")))) == "t2"
    assert PL.tier(S.candidates(_line(0, [0, 0, 9, 9], ("あい", "たい", "かい", "さい")))) == "t3"


def _shot(vn, num, lines):
    return {"shot": f"{vn}_{num}", "vn": vn, "kind": "vita", "url": f"https://t.vndb.org/sf/00/{num}.jpg",
            "lines": lines, "dets": [{"det": ln["det"]} for ln in lines]}


def test_plan_orders_tiers_and_skips_capped_repeats():
    name = ("千鶴", "千鶴", "千鶴", "千鶴")
    shots = {}
    for num in (1, 2):                                    # the name twice as an auto line
        shots[f"v10_{num}"] = _shot("v10", num, [_line(0, [20, 400, 80, 424], name)])
    t1 = ("あいうえお", "あいうえお", "あいうえお", "あいうえ")
    t3 = ("かきくけこ", "たきくけこ", "かさくけこ", "からくけこ")
    shots["v10_3"] = _shot("v10", 3, [_line(0, [20, 400, 80, 424], ("千鶴", "千鶴", "千鶴", "")),
                                      _line(1, [20, 440, 400, 464], t1), _line(2, [20, 480, 400, 504], t3)])
    shots["v11_4"] = _shot("v11", 4, [_line(0, [20, 440, 400, 464], ("さしすせそ", "さしすせそ", "ざしすせそ", ""))])
    lines, order = PL.plan(shots)
    assert lines["v10_1_0"]["tier"] == lines["v10_2_0"]["tier"] == "auto"
    assert lines["v10_3_0"].get("skip") == "repeat-cap" and "v10_3_0" not in order   # 千鶴 is twice in v10
    assert order.index("v10_3_1") < order.index("v11_4_0") < order.index("v10_3_2")    # t1, t2, t3
    assert all(lines[i]["tier"] != "auto" for i in order)


def test_scale_line_keeps_ink_inside():
    ln = _line(0, [101, 51, 303, 77], ("あ", "あ", "あ", "あ"), furi=[150, 40, 200, 49])
    p = PL.scale_line(ln, 0.5, 0.5)
    assert p["ink"] == [50, 25, 152, 39]                  # floor of the start, ceiling of the end
    assert p["crop"] == [50, 20, 152, 39] and p["furi"] == [[75, 20, 100, 25]]
    assert p["det"] == [50, 26, 152, 38]                  # detector boxes are rounded


def test_cap_repeats_per_vn_and_overall():
    rows = [{"id": f"v{v}_{i}_0", "vn": f"v{v}", "label": "はい"} for v in range(1, 7) for i in range(3)]
    kept, capped = PL.cap_repeats(rows)
    assert len(kept) == 8 and len(capped) == 10           # 2 per VN, 8 in all
    assert max(sum(r["vn"] == f"v{v}" for r in kept) for v in range(1, 7)) == 2


def test_read_rows_skip_unhashed_and_duplicates(tmp_path):
    rows = [{"image": 1, "vn": "v1", "kind": "hires", "url": "u1", "release": "r1", "session": "v1/r1/0",
             "order": 1, "sha256": "a", "pixel_sha256": "p", "dims": [800, 600]},
            {"image": 2, "vn": "v1", "kind": "hires", "url": "u2", "release": "r1", "session": "v1/r1/0",
             "order": 0, "sha256": "b", "pixel_sha256": "q", "dims": [800, 600]},
            {"image": 3, "vn": "v2", "kind": "hires", "url": "u3", "release": "r2", "session": "v2/r2/0",
             "order": 2, "dims": [800, 600]}]
    out = PL.read_rows(rows, {"1": "realplus:9"})
    assert [r["shot"] for r in out] == ["v1_2"] and out[0]["kind"] == "vita" and out[0]["source_kind"] == "hires"


def _screen(tmp_path):
    """An 800x600 'PC' screenshot: a dark box with two rows of bright glyph-like marks."""
    import cv2
    img = np.full((600, 800, 3), 25, np.uint8)
    img[420:580, 20:780] = 55
    for top in (440, 490):
        for x in range(40, 700, 26):
            img[top:top + 22, x:x + 18] = 235
            img[top + 10:top + 12, x - 3:x + 21] = 235
    p = tmp_path / "v100_7.png"
    cv2.imwrite(str(p), img[:, :, ::-1])
    return p


def test_build_lines_vita_and_psp_crops(tmp_path):
    path = _screen(tmp_path)
    s = 960 / 800 if 960 / 800 < 544 / 600 else 544 / 600           # Vita fit scale: 0.9067
    ink = [[int(40 * s), int(440 * s), int(700 * s) + 1, int(462 * s) + 1],
           [int(40 * s), int(490 * s), int(700 * s) + 1, int(512 * s) + 1]]
    texts = ["あいうえおかきくけこさしすせそたちつてとな", "らりるれろわをんがぎぐげござじずぜぞだぢづ"]
    lines = [_line(k, ink[k], (t, t, t, t)) for k, t in enumerate(texts)]
    shot = {"shot": "v100_7", "vn": "v100", "kind": "vita", "url": "https://t.vndb.org/sf/07/7.jpg",
            "path": str(path), "release": "r5", "lines": lines, "dets": [{"det": ln["det"]} for ln in lines]}
    shots = {"v100_7": shot}
    labels = []
    for ln in lines:
        rec = D.decide_line(f"v100_7_{ln['k']}", ln, None, None, "train", auto_consensus=True)
        rec.update(shot="v100_7", vn="v100", subset="realplus", kind="vita")
        labels.append(rec)
    assert all(r["status"] == "kept" and r["how"] == "auto" for r in labels)
    src = {"v100_7": {"session": "v100/r5/0", "dims": [800, 600]}}
    files, man, drops, kept = PL.build_lines(labels, shots, src=src)
    assert len(man) == 2 and len(kept) == 2
    for target, (W, H) in (("vita", (725, 544)), ("psp", (363, 272))):
        items = files[(target, "ink")]
        assert len(items) == 2
        for img, lab, meta in items:
            assert img.dtype == np.uint8 and img.ndim == 2 and lab in texts
            assert meta["target"] == target and meta["kind"] == "hires" and meta["session"] == "v100/r5/0"
            assert meta["screen"] == [W, H] and meta["src_dims"] == [800, 600] and meta["dark"] == 0
            x0, y0, x1, y1 = meta["box"]
            assert 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H and img.shape == (y1 - y0, x1 - x0)
            assert all(c[1] >= -1 for c in meta["chars"])
    psp = files[("psp", "ink")][0][2]["box"]
    vita = files[("vita", "ink")][0][2]["box"]
    assert abs(psp[0] - (vita[0] + 4) / 2 + 2) <= 2 and abs((psp[3] - psp[1]) - (vita[3] - vita[1] - 8) / 2 - 4) <= 2
    assert {k for m in man for k in m["crops"]} >= {"vita_ink", "psp_ink"}
    assert not any(t in json.dumps(man, ensure_ascii=False) for t in texts)     # manifest rows carry no text


def test_build_drops_out_of_vocab_and_late_exclusions(tmp_path):
    path = _screen(tmp_path)
    ln = _line(0, [36, 399, 636, 420], ("あいう", "あいう", "あいう", "あいう"))
    ln2 = _line(1, [36, 444, 636, 465], ("あい\U0002000B", "あい\U0002000B", "あい\U0002000B", "あい\U0002000B"))
    shot = {"shot": "v100_7", "vn": "v100", "kind": "vita", "url": "u", "path": str(path), "lines": [ln, ln2],
            "dets": [{"det": ln["det"]}, {"det": ln2["det"]}]}
    labels = []
    for x in (ln, ln2):
        rec = D.decide_line(f"v100_7_{x['k']}", x, None, None, "train", auto_consensus=True)
        rec.update(shot="v100_7", vn="v100", subset="realplus", kind="vita")
        labels.append(rec)
    _, man, drops, _ = PL.build_lines(labels, {"v100_7": shot})
    assert len(man) == 1 and drops["out-of-vocab"] == 1
    _, man, drops, _ = PL.build_lines(labels, {"v100_7": shot}, excluded_vns={"v100"})
    assert not man and drops["excluded-late"] == 2


def test_ink_mixed_drops_crops_that_take_in_a_neighbour():
    a = _line(0, [45, 433, 283, 453], ("あ", "あ", "あ", "あ"))
    b = _line(1, [45, 447, 399, 469], ("い", "い", "い", "い"))      # glyph rows overlap: tight spacing
    c = _line(2, [45, 500, 399, 520], ("う", "う", "う", "う"))
    dets = [{"det": x["det"]} for x in (a, b, c)]
    box_a = [41, 429, 287, 457]
    assert PL.ink_mixed(box_a, PL.read_line(a), dets)                  # half of b's glyphs inside
    assert not PL.ink_mixed([41, 496, 403, 524], PL.read_line(c), dets)


def test_late_exclusions_take_dedupe_copies_and_recomputed_reserved_vns():
    rows = [{"vn": "v1", "image": 10}, {"vn": "v2", "image": 20}, {"vn": "v3", "image": 30}]
    vns, shots = PL.late_exclusions(rows, saved_vns={"v9": "test-vn"}, current_vns={"v2"},
                                    dup_images={"30": "realplus:31", "77": "pool:teacher"}, leak_shots=["v1_10"])
    assert vns == {"v9", "v2"}
    assert shots == {"v3_30", "v1_10"}           # the copy found by dedupe and the leak-check drop


def test_grad_box_running_beyond_the_ink_crop_gets_no_crop(tmp_path, monkeypatch):
    path = _screen(tmp_path)
    ln = _line(0, [36, 399, 636, 420], ("あいう", "あいう", "あいう", "あいう"))
    shot = {"shot": "v100_7", "vn": "v100", "kind": "vita", "url": "u", "path": str(path), "lines": [ln],
            "dets": [{"det": ln["det"]}]}
    rec = D.decide_line("v100_7_0", ln, None, None, "train", auto_consensus=True)
    rec.update(shot="v100_7", vn="v100", subset="realplus", kind="vita")

    def fake_grad(extra):
        def grad_assign(rgb, lines, dets, kind):
            out = []
            for x in lines:
                x0, y0, x1, y1 = x["ink"]
                out.append({"region": [x0, y0, x1, y1], "box": [x0 - 2, y0 - 2, x1 + 2 + extra, y1 + 2],
                            "status": "ok", "dark": 0})
            return out
        return grad_assign

    monkeypatch.setattr(PL.L, "grad_assign", fake_grad(0))
    files, _, drops, _ = PL.build_lines([rec], {"v100_7": shot})
    assert len(files[("vita", "grad")]) == 1 and not drops["vita-grad-unchecked"]
    monkeypatch.setattr(PL.L, "grad_assign", fake_grad(90))             # runs 90 px past the line
    files, _, drops, _ = PL.build_lines([rec], {"v100_7": shot})
    assert not files.get(("vita", "grad")) and drops["vita-grad-unchecked"] == 1
    assert len(files[("vita", "ink")]) == 1


def test_edge_flagged_line_needs_an_edge_check(tmp_path):
    path = _screen(tmp_path)
    ln = _line(0, [36, 399, 330, 420], ("あいう", "あいう", "あいう", "あいう"))     # the row runs on to x 627
    ln["view"] = [20, 392, 700, 428]                                     # read.py's wider view of the line
    shot = {"shot": "v100_7", "vn": "v100", "kind": "vita", "url": "u", "path": str(path), "lines": [ln],
            "dets": [{"det": ln["det"]}]}
    rec = D.decide_line("v100_7_0", ln, None, None, "train", auto_consensus=True)
    rec.update(shot="v100_7", vn="v100", subset="realplus", kind="vita")
    files, man, drops, _ = PL.build_lines([rec], {"v100_7": shot})
    assert not man and drops["edge-unchecked"] == 1
    checked = {"v100_7_0": {"id": "v100_7_0", "crop": [32, 395, 334, 424], "edge_ok": True,
                            "edge_why": "checked", "flags": ["ink-right"]}}
    files, man, drops, _ = PL.build_lines([rec], {"v100_7": shot}, edge=checked)
    assert len(man) == 1 and man[0]["edge"] == {"flags": ["ink-right"], "why": "checked"}
    assert files[("vita", "ink")][0][2]["box"] == [32, 395, 334, 424]
    rejected = {"v100_7_0": dict(checked["v100_7_0"], edge_ok=False, edge_why="edge-cut")}
    _, man, drops, _ = PL.build_lines([rec], {"v100_7": shot}, edge=rejected)
    assert not man and drops["edge-cut"] == 1


def test_scale_line_maps_view_and_chars_to_psp():
    ln = {"ink": [100, 200, 301, 223], "det": [100, 200, 300, 222], "view": [88, 190, 320, 232], "parts": []}
    p = PL.scale_line(ln, 0.5, 0.5)
    assert p["view"] == [44, 95, 160, 116]
    assert PL.scale_chars([["あ", 100, 200, 110, 222, 0.9]], 0.5, 0.5) == [["あ", 50.0, 100.0, 55.0, 111.0, 0.9]]
