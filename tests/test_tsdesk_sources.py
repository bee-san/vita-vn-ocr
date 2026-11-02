"""textspot-desktop round 2 (#173): the dense hard-example source (train/tsdesk_dense_source.py) and the label-coverage
measure (train/tsdesk_coverage.py). No data: synthetic shards and frames only."""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "train", ROOT / "synth", ROOT / "teacher", ROOT / "eval", ROOT / "tools", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

pytest.importorskip("cv2")

import tsdesk_coverage as C  # noqa: E402
import tsdesk_dense_source as S  # noqa: E402
from desktopdata import shards as SH  # noqa: E402

# ----------------------------------------------------------------------------- the dense source


def _smp(vn, shot, sha, n_text, n_name=0, w=64, h=48):
    lines = [[2, 2 + 4 * k, 60, 5 + 4 * k, 0, "text", 1.0] for k in range(n_text)]
    lines += [[2, 40, 20, 44, 0, "name", 1.0]] * n_name
    return {"src": "realplus", "kind": "pc", "vn": vn, "shot": shot, "lines": lines, "ignore": [[0, 0, 3, 3]],
            "meta": {"sha256": sha, "dims": [w, h]}}


def _tree(tmp_path, monkeypatch):
    """A 1b-like shard set and the exclusion sets' files: frames of VNs v1-v9, each set holding one of them."""
    rng = np.random.default_rng(0)
    shards_dir = tmp_path / "pages"
    samples = [_smp("v1", "v1_10", "a" * 64, 9),              # kept
               _smp("v2", "v2_20", "b" * 64, 8, n_name=2),    # kept (8 text lines)
               _smp("v3", "v3_30", "c" * 64, 7, n_name=3),    # not dense: 7 text lines (names do not count)
               _smp("v4", "v4_40", "d" * 64, 10),             # a dense set's VN
               _smp("v5", "v5_50", "e" * 64, 10),             # a dense set's screenshot (sha256) under another VN
               _smp("v6", "v6_60", "f" * 64, 10),             # a dev-v2 VN
               _smp("v7", "v7_70", "0" * 64, 10),             # a PC test screenshot id
               _smp("v8", "v8_80", "1" * 64, 10),             # a confirm-v2 sha256
               _smp("v9", "v9_90", "2" * 64, 10)]             # a reserved test VN
    greys = [rng.integers(0, 255, (48, 64), dtype=np.uint8) for _ in samples]
    recs = []
    for k in range(0, len(samples), 5):
        name = f"rp_{k // 5:03d}"
        SH.write_shard(shards_dir / "desk" / name, samples[k:k + 5], greys[k:k + 5])
        recs.append(SH.shard_record(shards_dir / "desk" / name, "desk"))
    man = tmp_path / "realplus_native_v3.json"
    man.write_text(json.dumps({"plan_sha256": "p" * 64, "shards": recs}))
    (tmp_path / "skip_vns.txt").write_text("v4\n")
    (tmp_path / "dense.json").write_text(json.dumps({"sets": {"dense_dev": {"vns": ["v4"], "shots": [
        {"shot": "v4_40", "sha256": "d" * 64}, {"shot": "v100_1", "sha256": "e" * 64}]}}}))
    frames = tmp_path / "frames"
    frames.mkdir()

    def frames_file(name, cluster, sha, path):
        with open(frames / name, "wb") as f:
            pickle.dump({"frames": [{"cluster": cluster, "src": {"sha256": sha, "path": path}}], "meta": {}}, f)
    frames_file("dev.pkl", "v6", "9" * 64, "/x/v200_2.jpg")
    frames_file("pc.pkl", "v6", "8" * 64, "/x/v200_3.jpg")
    frames_file("confirm.pkl", "v300", "1" * 64, "/x/fresh-psp-v300")
    (tmp_path / "pctest.jsonl").write_text(json.dumps({"vn": "v400", "sha256": "7" * 64, "shot": "v7_70"}) + "\n")
    (tmp_path / "reserved.json").write_text(json.dumps({"test": {"v9": "clannad"}}))
    (tmp_path / "vndb.json").write_text(json.dumps({"titles": {"PCSG00807": {"vns": ["v500"]},
                                                               "PCSG00001": {"vns": ["v2"]}}}))
    for k, v in {"MAN": man, "SHARDS": shards_dir, "DENSE_MAN": tmp_path / "dense.json",
                 "SKIP_VNS": tmp_path / "skip_vns.txt", "FRAMES": frames, "PCTEST": tmp_path / "pctest.jsonl",
                 "RESERVED": tmp_path / "reserved.json", "VNDB_MAP": tmp_path / "vndb.json", "ROOT": tmp_path}.items():
        monkeypatch.setattr(S, k, v)
    (tmp_path / "train").mkdir()
    return samples, greys


def test_dense_source_build_excludes_by_vn_sha_and_shot(tmp_path, monkeypatch):
    import argparse
    samples, greys = _tree(tmp_path, monkeypatch)
    out = tmp_path / "out"
    S.cmd_build(argparse.Namespace(out=str(out), min_lines=8, shard_frames=200, manifest_out=None, verify_npy=True))
    doc = json.loads((out / "manifest.json").read_text())
    assert [f["shot"] for f in doc["frames"]] == ["v1_10", "v2_20"]
    assert doc["totals"]["frames"] == 2 and doc["totals"]["vns"] == 2 and doc["totals"]["text_lines"] == 17
    assert doc["totals"]["frames_of_hooked_vns"] == 1                       # v2 is a hooked training VN: kept
    d = doc["candidates"]["dropped"]
    assert d["frames"] == 6
    assert d["dense_sets:vn"] == 1 and d["dense_sets:sha256"] == 2 and d["dense_sets:shot"] == 1
    assert d["dev_v2:vn"] == 1 and d["pc_test:shot"] == 1 and d["confirm_v2:sha256"] == 1
    assert d["reserved_test:vn"] == 1
    assert doc["check"]["passed"] and doc["check"]["frames_checked"] == 2
    assert all(v[w] == 0 for v in doc["check"]["by_set"].values() for w in ("vn", "sha256", "shot"))
    # the written frames are the source's pixels and labels
    data, got = SH.read_shard(out / S.DIR_NAME / doc["shards"][0]["name"])
    for smp, want, g in zip(got, (samples[0], samples[1]), (greys[0], greys[1])):
        assert np.array_equal(SH.canvas(data, smp), g)
        assert smp["lines"] == want["lines"] and smp["ignore"] == want["ignore"] and smp["vn"] == want["vn"]


def test_dense_source_loads_as_an_extra_pool(tmp_path, monkeypatch):
    import argparse
    pytest.importorskip("torch")
    import tsdesk_det as TD
    _tree(tmp_path, monkeypatch)
    out = tmp_path / "out"
    S.cmd_build(argparse.Namespace(out=str(out), min_lines=8, shard_frames=1, manifest_out=None, verify_npy=False))
    spec = f"desk_dense={out / 'manifest.json'}:{out}"
    import desktop_det as DD
    (name, (man, root, splits)), = DD.parse_extra([spec]).items()
    pool = TD.SkipPool(man, root, splits, frozenset({"v4", "v6"}))
    assert name == "desk_dense" and len(pool.items) == 2 and pool.skipped == 0
    g, smp = pool.draw(np.random.default_rng(0))
    assert g.shape == (smp["h"], smp["w"]) and smp["vn"] in ("v1", "v2")


def test_dense_source_refuses_a_failed_check(tmp_path, monkeypatch):
    import argparse
    _tree(tmp_path, monkeypatch)
    monkeypatch.setattr(S, "hits", lambda rec, ex: [])                      # the first filter lets everything in
    with pytest.raises(SystemExit):
        S.cmd_build(argparse.Namespace(out=str(tmp_path / "o"), min_lines=8, shard_frames=200, manifest_out=None,
                                       verify_npy=False))
    assert not (tmp_path / "o" / "manifest.json").exists()


def test_dense_source_checks_the_shards_json(tmp_path, monkeypatch):
    import argparse
    _tree(tmp_path, monkeypatch)
    p = tmp_path / "pages" / "desk" / "rp_000.json"
    p.write_text(p.read_text().replace('"v1_10"', '"v1_11"'))
    with pytest.raises(SystemExit):
        S.cmd_build(argparse.Namespace(out=str(tmp_path / "o"), min_lines=8, shard_frames=200, manifest_out=None,
                                       verify_npy=False))


# ----------------------------------------------------------------------------- the coverage measure

def test_uncovered_spans_need_min_span():
    lab, boxes = C.label_mask([[10, 10, 50, 30, 0, "dialogue", 1.0]], [], 100, 60)
    assert C.uncovered([10, 10, 50, 30], lab, boxes) == []                  # inside the positive
    sp = C.uncovered([10, 10, 80, 30], lab, boxes)                          # 30 px past its end: a glyph or more
    assert sp == [[50, 10, 80, 30]]
    assert C.uncovered([10, 10, 52, 30], lab, boxes) == []                  # 2 px past it: under MIN_SPAN
    assert C.uncovered([10, 8, 50, 30], lab, boxes) == []                   # 2 rows above it: under MIN_SPAN
    lab2, boxes2 = C.label_mask([], [[60, 10, 90, 30]], 100, 60)
    assert C.uncovered([60, 10, 90, 30], lab2, boxes2) == []                # an ignore area covers it


def test_sample_coverage_counts_and_the_junk_recipe():
    pos = [[10, 10, 50, 30, 0, "dialogue", 1.0]]
    ign = [[0, 50, 5, 55], [60, 40, 90, 50]]                                # a ruby area, then one weak MeikiOCR area
    smp = {"w": 100, "h": 60, "title": "T", "page": "000001", "lines": pos, "ignore": ign,
           "meta": {"counts": {"ruby_ignore": 1, "drop_ignore": 0, "uncovered_ignore": 0, "meiki_ignore": 1}}}
    meiki = [{"raw": [10, 10, 50, 30], "score": 0.95, "box": [10, 10, 50, 30]},          # the dialogue line
             {"raw": [60, 10, 90, 30], "score": 0.8, "box": [61, 11, 89, 29]},           # strong, unlabelled
             {"raw": [60, 40, 90, 50], "score": 0.3, "box": [60, 40, 90, 50]}]           # weak: the ignore area
    c, rows = C.sample_coverage(smp, meiki, "hook", measure="box")
    assert c["strong"] == 2 and c["strong_uncovered"] == 1 and c["strong_uncovered_whole"] == 1
    assert c["weak_areas_junk"] == 1 and c["weak_0.3+"] == 1 and c["samples_uncovered"] == 1
    kinds = sorted(r["kind"] for r in rows)
    assert kinds == ["uncovered", "weak"]
    weak = [r for r in rows if r["kind"] == "weak"][0]
    assert weak["box"] == [60, 40, 90, 50] and weak["score"] == 0.3
    # another source: no junk recipe, the ignore areas stay
    c2, _ = C.sample_coverage(dict(smp), meiki, "desk_synth", measure="box")
    assert c2["weak_areas_junk"] == 0 and c2["strong_uncovered"] == 1


def test_element_key_counts_a_repeated_label_once():
    a = {"native": [96, 16, 136, 32]}
    b = {"native": [97, 17, 137, 33]}                                       # 1 px off: the same 8 px cells
    assert C.element_key(a, "T") == C.element_key(b, "T") != C.element_key(a, "U")
    assert C.element_key({"native": None}, "T") is None


def _grey_line():
    """A grey frame: a dialogue line's ink (rows 20-33, x 20-140, strokes every 4 px) and an unlabelled word
    (rows 60-73, x 20-70)."""
    g = np.full((100, 200), 30, np.uint8)
    xs = np.arange(200)
    for y0, x0, x1 in ((20, 20, 140), (60, 20, 70)):
        st = (xs % 4 == 0) & (xs >= x0) & (xs < x1)
        g[y0:y0 + 14][:, st] = 220
        g[y0 + 6, x0:x1] = 220
    return g


def test_ink_measure_ignores_an_offset_box_over_a_labelled_line():
    g = _grey_line()
    pos = [[20, 20, 140, 34, 0, "dialogue", 1.0]]
    smp = {"w": 200, "h": 100, "lines": pos, "ignore": []}
    offset = {"raw": [20, 16, 140, 28], "score": 0.9, "box": [20, 16, 140, 28]}        # 4 px above its line
    c_box, _ = C.sample_coverage(dict(smp), [offset], "desk_synth", grey=g, measure="box")
    c_ink, _ = C.sample_coverage(dict(smp), [offset], "desk_synth", grey=g, measure="ink")
    assert c_box["strong_uncovered"] == 1 and c_ink["strong_uncovered"] == 0
    word = {"raw": [18, 58, 72, 76], "score": 0.8, "box": [20, 60, 70, 74]}             # unlabelled ink
    c, rows = C.sample_coverage(dict(smp), [offset, word], "desk_synth", grey=g, measure="ink")
    assert c["strong_uncovered"] == 1 and c["strong_uncovered_whole"] == 1
    (r,) = rows
    assert r["spans"] == [[20, 60, 70, 74]] and r["frac"] == 1.0
    # an ignore area over the word covers it
    c2, _ = C.sample_coverage(dict(smp, ignore=[[18, 58, 72, 76]]), [word], "desk_synth", grey=g, measure="ink")
    assert c2["strong_uncovered"] == 0


def test_ink_measure_needs_half_a_glyph():
    g = _grey_line()
    pos = [[20, 20, 136, 34, 0, "dialogue", 1.0]]                                      # misses the last 4 px
    smp = {"w": 200, "h": 100, "lines": pos, "ignore": []}
    m = {"raw": [20, 20, 140, 34], "score": 0.9, "box": [20, 20, 140, 34]}
    c, _ = C.sample_coverage(dict(smp), [m], "desk_synth", grey=g, measure="ink")
    assert c["strong_uncovered"] == 0                                    # one stroke past the label: no glyph


# ----------------------------------------------------------------------------- label edits (apart -> notdlg)

def _edit_sample():
    return {"title": "T", "page": "000001", "w": 960, "h": 544,
            "lines": [[100, 200, 400, 224, 0, "dialogue", 1.0], [700, 30, 800, 50, 0, "notdlg", 0.9]],
            "ignore": [[0, 500, 10, 510]], "meta": {"counts": {"meiki_ignore": 1}}}


def _row(raw, score):
    return {"kind": "uncovered", "where": "apart", "source": "hook", "raw": raw, "score": score}


def test_edit_candidates_keep_scene_text_only():
    smp = _edit_sample()
    meiki = [{"raw": [50, 40, 150, 56], "score": 0.8, "box": [52, 42, 148, 55], "vertical": False},       # kept
             {"raw": [51, 41, 149, 57], "score": 0.6, "box": [53, 43, 147, 56], "vertical": False},       # duplicate
             {"raw": [600, 204, 620, 222], "score": 0.9, "box": [601, 205, 619, 221], "vertical": False}, # row
             {"raw": [120, 188, 200, 198], "score": 0.9, "box": [121, 189, 199, 198], "vertical": False}, # ruby
             {"raw": [690, 35, 760, 60], "score": 0.7, "box": [692, 36, 758, 58], "vertical": False},     # overlap
             {"raw": [300, 60, 320, 160], "score": 0.75, "box": [302, 62, 318, 158], "vertical": True}]   # kept
    meiki.append({"raw": [210, 192, 290, 198], "score": 0.6, "box": [211, 193, 289, 199], "vertical": False})
    # ^ a small Latin label just above the line (a quarter of its height): ruby-like by place, too small for ruby
    rows = [_row(m["raw"], m["score"]) for m in meiki] + [_row([1, 1, 5, 5], 0.9)]                     # no box
    adds, why = C.edit_candidates(smp, rows, meiki)
    assert adds == [[52, 42, 148, 55, 0, "notdlg", 1.0], [302, 62, 318, 158, 1, "notdlg", 0.9167],
                    [211, 193, 289, 199, 0, "notdlg", 0.6667]]
    assert dict(why) == {"duplicate": 1, "row": 1, "ruby": 1, "overlap": 1, "no_meiki_box": 1}


def test_apply_edits_appends_and_maps(monkeypatch):
    import tsdesk_targets as TT
    smp = _edit_sample()
    edits = {"T": {"000001": [[52, 42, 148, 55, 0, "notdlg", 1.0]]}}
    out = C.apply_edits(smp, edits, "hook")
    assert out["lines"] == smp["lines"] + [[52, 42, 148, 55, 0, "notdlg", 1.0]] and out["ignore"] is smp["ignore"]
    assert C.apply_edits(dict(smp, page="000002"), edits, "hook") is None

    class Cap:
        def box(self, b):
            return [2 * b[0], 2 * b[1] + 56, 2 * b[2], 2 * b[3] + 56]
    monkeypatch.setattr(TT, "capture_of", lambda s: Cap())
    d = C.apply_edits(smp, edits, "desk_hook")
    assert d["lines"][-1] == [104, 140, 296, 166, 0, "notdlg", 1.0]
    monkeypatch.setattr(TT, "capture_of", lambda s: None)
    assert C.apply_edits(smp, edits, "desk_hook") is None


def _edit_tree(tmp_path):
    """A hook shard, its pool MeikiOCR cache, the page pool and coverage rows for cmd_edits."""
    smp = dict(_edit_sample(), off=0, src="hook", kind="vita", split="train")
    sh = tmp_path / "shards" / "train"
    sh.mkdir(parents=True)
    (sh / "T_000.json").write_text(json.dumps({"samples": [smp]}))
    meiki = [{"raw": [50, 40, 150, 56], "score": 0.8, "box": [52, 42, 148, 55], "vertical": False},
             {"raw": [600, 204, 620, 222], "score": 0.9, "box": [601, 205, 619, 221], "vertical": False}]
    (tmp_path / "meiki").mkdir()
    (tmp_path / "meiki" / "T.jsonl").write_text(json.dumps({"title": "T", "page": "000001", "meiki": meiki}) + "\n")
    (tmp_path / "pool").mkdir()
    (tmp_path / "pool" / "T.jsonl").write_text(json.dumps({"page": "000001", "split": "train"}) + "\n")
    rows = [dict(_row(m["raw"], m["score"]), title="T", page="000001", dir="train", shard="T_000", i=0) for m in meiki]
    (tmp_path / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return smp


def test_cmd_edits_builds_a_checked_cache(tmp_path):
    import argparse
    _edit_tree(tmp_path)
    C.cmd_edits(argparse.Namespace(rows=[str(tmp_path / "rows.jsonl")], shards=str(tmp_path / "shards"),
                                   meiki=str(tmp_path / "meiki"), targets=None, pool=str(tmp_path / "pool"),
                                   out=str(tmp_path / "edits"), workers=1))
    edits, doc = C.load_edits(tmp_path / "edits")
    assert edits == {"T": {"000001": [[52, 42, 148, 55, 0, "notdlg", 1.0]]}}
    assert doc["added"] == 1 and doc["left_out"] == {"row": 1} and doc["by_split"] == {"train": 1}
    (tmp_path / "edits" / "T.json").write_text("{}\n")
    with pytest.raises(SystemExit):                                     # a changed file is refused
        C.load_edits(tmp_path / "edits")


def test_trainer_edit_pools(tmp_path, monkeypatch):
    import argparse
    pytest.importorskip("torch")
    import tsdesk_det as TD
    import tsdesk_targets as TT
    smp = _edit_tree(tmp_path)
    C.cmd_edits(argparse.Namespace(rows=[str(tmp_path / "rows.jsonl")], shards=str(tmp_path / "shards"),
                                   meiki=str(tmp_path / "meiki"), targets=None, pool=str(tmp_path / "pool"),
                                   out=str(tmp_path / "edits"), workers=1))

    class Hook:
        by_title = {"T": [(0, smp), (0, dict(smp, page="000002"))]}

    class Pool:
        items = [(0, dict(smp))]

    class Sources:
        extra_pools = {"desk_hook": Pool()}

    class Cap:
        def box(self, b):
            return [2 * b[0], 2 * b[1], 2 * b[2], 2 * b[3]]
    monkeypatch.setattr(TT, "capture_of", lambda s: Cap())
    hook, src = Hook(), Sources()
    rec = TD.edit_pools(src, hook, tmp_path / "edits")
    assert rec["hook"] == {"samples_edited": 1, "lines_added": 1} and rec["desk_hook"] == rec["hook"]
    assert hook.by_title["T"][0][1]["lines"][-1] == [52, 42, 148, 55, 0, "notdlg", 1.0]
    assert hook.by_title["T"][1][1] is not smp and len(hook.by_title["T"][1][1]["lines"]) == 2
    assert src.extra_pools["desk_hook"].items[0][1]["lines"][-1] == [104, 84, 296, 110, 0, "notdlg", 1.0]
    assert smp["lines"][-1][5] == "notdlg" and len(smp["lines"]) == 2                 # the input sample untouched
