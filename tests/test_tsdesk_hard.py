"""textspot-desktop round 2 (#173): train/tsdesk_hard_source.py (line status, frame counts, selection, the build and
its exclusion check) and train/tsdesk_det.py's desk_hook_* sources. No data: synthetic boxes and shards only."""
import argparse
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

import tsdesk_dense_source as DS  # noqa: E402
import tsdesk_hard_source as HS  # noqa: E402
from desktopdata import shards as SH  # noqa: E402

# ----------------------------------------------------------------------------- matching

def test_line_points_one_per_glyph():
    assert HS.line_points([0, 0, 100, 20], 0) == [(10.0, 10.0), (30.0, 10.0), (50.0, 10.0), (70.0, 10.0),
                                                   (90.0, 10.0)]
    assert HS.line_points([0, 0, 20, 60], 1) == [(10.0, 10.0), (10.0, 30.0), (10.0, 50.0)]
    assert HS.line_points([0, 0, 5, 20], 0) == [(2.5, 10.0)]                  # shorter than its height: one cell


def test_line_status_whole_cut_missed_and_merges():
    lines = [[0, 0, 100, 20, 0, "text"], [0, 30, 100, 50, 0, "text"], [200, 0, 300, 20, 0, "text"],
             [400, 0, 500, 20, 0, "text"]]
    boxes = [[0, 0, 100, 50],                       # both rows of the first two lines: merged across rows
             [200, 0, 250, 20],                     # the third line's first half: cut
             [600, 0, 700, 20]]                     # nothing: the fourth line is missed
    st, merged = HS.line_status(boxes, lines)
    assert st == ["whole", "whole", "cut", "missed"]
    assert merged == [(0, [0, 1])]
    assert HS.same_row([lines[0], lines[2]]) and not HS.same_row([lines[0], lines[1]])


def test_frame_eval_counts_and_score():
    lines = [[0, 0, 100, 20, 0, "dialogue"], [120, 0, 220, 20, 0, "dialogue"], [0, 40, 100, 60, 0, "notdlg"],
             [0, 80, 10, 82, 0, "dialogue"]]                                 # the last: 2 px across, not scored
    boxes = [[0, 0, 220, 20, 0],                    # the two dialogue lines side by side in one box
             [0, 40, 50, 60, 0],                    # the notdlg line cut
             [300, 300, 340, 320, 0],               # junk
             [500, 500, 540, 520, 0]]               # inside an ignore area: not junk
    c, fail = HS.frame_eval(boxes, lines, [[490, 490, 560, 530]], ("dialogue", "name"), 1.0)
    assert c["main"] == 2 and c["other"] == 1 and c["tiny"] == 1
    assert c["main_whole"] == 2 and c["other_cut"] == 1 and c["merged_side"] == 1 and c["junk"] == 1
    assert c["score"] == 1 and fail == ["w", "w"]
    c2, fail2 = HS.frame_eval([], lines, [], ("dialogue", "name"), 1.0)
    assert c2["main_missed"] == 2 and c2["score"] == 2 and fail2 == ["m", "m"]
    # the policy's scale: at 0.25 a 20 px tall line is 5 px across, under MIN_ACROSS
    c3, _ = HS.frame_eval([], lines, [], ("dialogue", "name"), 0.25)
    assert c3["main"] == 0 and c3["tiny"] == 4 and c3["score"] == 0


def test_detect_read_applies_the_reading_filter(monkeypatch):
    import tsdesk as T

    class Det:
        def detect(self, g):
            return [T.Box(10, 10, 60, 12, 0, 1.0, 0, 0.0), T.Box(10, 40, 60, 12, 0, 1.0, 0, 0.0),
                    T.Box(100, 10, 12, 60, 1, 1.0, 0, 0.0)]

    class Rec:
        def read(self, crop, invert=False):
            class R:
                pass
            r = R()
            r.text, r.conf_mean = ("こんにちは", 0.9) if crop.mean() > 100 else ("0000", 0.9)
            return r
    g = np.zeros((120, 200), np.uint8)
    g[5:25, 5:75] = 200                                                       # the first box reads Japanese
    monkeypatch.setitem(HS._W, "det", Det())
    monkeypatch.setitem(HS._W, "rec", Rec())
    monkeypatch.setitem(HS._W, "pad", "k6v4")
    monkeypatch.setitem(HS._W, "drop", T.drop_options({"nojp": 1}))
    boxes, gone = HS.detect_read(g)
    assert boxes[0] == [10.0, 10.0, 70.0, 22.0, 0] and boxes[2][4] == 1
    assert gone == [1]                                                        # "0000": no Japanese; vertical kept
    monkeypatch.setitem(HS._W, "drop", None)
    assert HS.detect_read(g)[1] == []


def test_select_ranks_and_caps():
    def row(shot, vn, score, main):
        return {"shot": shot, "vn": vn, "pre": {"score": score, "main": main}}
    rows = [row("a", "v1", 3, 10), row("b", "v1", 3, 3), row("c", "v1", 2, 2), row("d", "v2", 1, 1),
            row("e", "v2", 0, 5), row("f", "v3", 5, 8)]
    picked, n = HS.select(rows, per_vn=2, cap=10)
    assert n == 5 and [r["shot"] for r in picked] == ["f", "b", "a", "d"]   # c: v1's third
    picked, _ = HS.select(rows, per_vn=2, cap=2)
    assert [r["shot"] for r in picked] == ["f", "b"]
    assert [r["shot"] for r in HS.select(rows, 5, 10, min_score=3)[0]] == ["f", "b", "a"]


# ----------------------------------------------------------------------------- the build

def _smp(vn, shot, sha, n=3):
    lines = [[2, 2 + 6 * k, 60, 7 + 6 * k, 0, "text", 1.0] for k in range(n)]
    return {"src": "realplus", "kind": "pc", "vn": vn, "shot": shot, "lines": lines, "ignore": [[0, 0, 3, 3]],
            "meta": {"sha256": sha, "dims": [64, 48]}}


def _setup(tmp_path, monkeypatch):
    """1b-like shards of VNs v1-v4 (v4 a dev-v2 VN), the exclusion sets' files, and mined rows for them."""
    rng = np.random.default_rng(0)
    samples = [_smp("v1", "v1_10", "a" * 64), _smp("v1", "v1_11", "b" * 64), _smp("v2", "v2_20", "c" * 64),
               _smp("v3", "v3_30", "d" * 64), _smp("v4", "v4_40", "e" * 64)]
    greys = [rng.integers(0, 255, (48, 64), dtype=np.uint8) for _ in samples]
    root = tmp_path / "pages"
    SH.write_shard(root / "desk" / "rp_000", samples, greys)
    man = tmp_path / "rp.json"
    man.write_text(json.dumps({"shards": [SH.shard_record(root / "desk" / "rp_000", "desk")]}))
    monkeypatch.setitem(HS.SOURCES, "1b", (man, root))
    (tmp_path / "skip.txt").write_text("v9\n")
    (tmp_path / "dense.json").write_text(json.dumps({"sets": {"d": {"vns": ["v9"], "shots": []}}}))
    fr = tmp_path / "frames"
    fr.mkdir()
    for n, vn in (("dev.pkl", "v4"), ("pc.pkl", "v4"), ("confirm.pkl", "v8")):
        with open(fr / n, "wb") as f:
            pickle.dump({"frames": [{"cluster": vn, "src": {"sha256": "f" * 64, "path": "/x/v8_1.jpg"}}]}, f)
    (tmp_path / "pctest.jsonl").write_text(json.dumps({"vn": "v7", "sha256": "1" * 64}) + "\n")
    (tmp_path / "reserved.json").write_text(json.dumps({"test": {"v6": "clannad"}}))
    (tmp_path / "vndb.json").write_text(json.dumps({"titles": {}}))
    for k, v in {"DENSE_MAN": tmp_path / "dense.json", "SKIP_VNS": tmp_path / "skip.txt", "FRAMES": fr,
                 "PCTEST": tmp_path / "pctest.jsonl", "RESERVED": tmp_path / "reserved.json",
                 "VNDB_MAP": tmp_path / "vndb.json"}.items():
        monkeypatch.setattr(DS, k, v)
    mined = tmp_path / "mined"
    (mined / "rows").mkdir(parents=True)
    scores = [2, 1, 3, 0, 4]                                  # v4 (a dev-v2 VN) would rank first: the check stops it
    rows = [{"src": "1b", "dir": "desk", "shard": "rp_000", "i": i, **HS._frame_ids(s), "title": None, "page": None,
             "w": 64, "h": 48, "pre": {"score": sc, "main": 3, "main_missed": sc}, "post": {"score": sc}, "fail": "",
             "boxes": [], "gone": []} for i, (s, sc) in enumerate(zip(samples, scores))]
    (mined / "mine.json").write_text(json.dumps({k: None for k in ("model", "model_sha256", "spec", "policy", "pad",
                                                                     "rec_sha256", "targets", "targets_sha256",
                                                                     "label_edits", "edits_sha256", "code_sha256")}
                                                | {"sources": {"1b": {"frames": 5}}}))
    return samples, greys, rows, mined


def _args(tmp_path, mined, **kw):
    a = dict(mined=str(mined), out=str(tmp_path / "out"), per_vn_1b=1, per_vn_1a=25, max_1b=10, max_1a=10,
             min_score=1, manifest_out_1b=None, manifest_out_1a=None)
    a.update(kw)
    return argparse.Namespace(**a)


def test_build_writes_a_checked_loadable_source(tmp_path, monkeypatch):
    samples, greys, rows, mined = _setup(tmp_path, monkeypatch)
    (mined / "rows" / "1b_rp_000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[:4]))
    HS.cmd_build(_args(tmp_path, mined))
    doc = json.loads((tmp_path / "out" / "manifest_1b.json").read_text())
    assert [f["shot"] for f in doc["frames"]] == ["v1_10", "v2_20"]          # v1's worst, v2; v3 scored 0
    assert doc["name"] == "desk_hard" and doc["check"]["passed"] and doc["check"]["frames_checked"] == 2
    assert doc["candidates"]["frames_with_score"] == 3 and doc["totals"]["failure_kinds_pre"]["main_missed"] == 5
    data, got = SH.read_shard(tmp_path / "out" / "desk_hard_v1" / "hard_000")
    for smp, i in zip(got, (0, 2)):
        assert np.array_equal(SH.canvas(data, smp), greys[i]) and smp["lines"] == samples[i]["lines"]
    pytest.importorskip("torch")
    import desktop_det as DD
    import tsdesk_det as TD
    (name, (man, root, splits)), = DD.parse_extra([f"desk_hard={tmp_path / 'out' / 'manifest_1b.json'}:"
                                                   f"{tmp_path / 'out'}"]).items()
    pool = TD.SkipPool(man, root, splits)
    assert len(pool.items) == 2


def test_build_refuses_an_excluded_frame(tmp_path, monkeypatch):
    _, _, rows, mined = _setup(tmp_path, monkeypatch)
    (mined / "rows" / "1b_rp_000.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))   # v4 among them
    with pytest.raises(SystemExit):
        HS.cmd_build(_args(tmp_path, mined))
    assert not (tmp_path / "out" / "manifest_1b.json").exists()


# ----------------------------------------------------------------------------- desk_hook_* in the trainer

def test_desk_hook_named_sources_read_as_desk_hook(monkeypatch):
    pytest.importorskip("torch")
    import tsdesk_det as TD
    import tsdesk_targets as TT
    assert TD.hooked_as("desk_hook_hard") == "desk_hook" and TD.hooked_as("desk_hook") == "desk_hook"
    assert TD.hooked_as("hook") == "hook" and TD.hooked_as("desk_hard") is None and TD.hooked_as("desk_dense") is None
    assert TD.exact_source("desk_hook_hard") and not TD.exact_source("desk_hard")
    assert TD.exact_source("desk_hook_hard", ["hook", "desk_hook"]) and not TD.exact_source("desk_hook_hard", ["hook"])
    ign = [[0, 0, 5, 5], [10, 10, 15, 15]]
    smp = {"ignore": list(ign), "meta": {"counts": {"meiki_ignore": 1}, "src_ignore": 2}}
    assert TD.strip_weak(smp, "desk_hook_hard")["ignore"] == ign[:1] and TD.strip_weak(smp, "desk_hard") is smp

    class Cap:
        def box(self, b):
            return [2 * v for v in b[:4]]
    monkeypatch.setattr(TT, "capture_of", lambda s: Cap())
    s1 = {"title": "T", "page": "000001", "lines": [[20, 20, 100, 40, 0, "dialogue", 1.0]], "ignore": [], "meta": {}}

    class Pool:
        def __init__(self):
            self.items = [(0, dict(s1))]

    class Sources:
        extra_pools = {"desk_hook_hard": Pool(), "desk_hard": Pool()}
    cache = {"T": {"000001": {"old": [[10, 10, 50, 20]], "new": [[11, 10, 49, 20]]}}}
    monkeypatch.setattr(TT, "load_cache", lambda d: (cache, {}))
    monkeypatch.setattr(TT, "_sha", lambda p: "x")
    src = Sources()
    rec = TD.patch_pools(src, None, "unused")
    assert rec["desk_hook_hard"] == {"patched": 1, "kept": 0} and "desk_hard" not in rec
    assert src.extra_pools["desk_hook_hard"].items[0][1]["lines"][0][:4] == [22, 20, 98, 40]
    assert src.extra_pools["desk_hard"].items[0][1]["lines"][0][:4] == [20, 20, 100, 40]
