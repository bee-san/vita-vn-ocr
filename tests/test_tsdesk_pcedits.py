"""The PC label edits (train/tsdesk_coverage.py pcedits, train/tsdesk_det.py --label-edits with a PC cache):
MeikiOCR boxes in 1b's ignore areas that hold ink become positives. Synthetic data only."""
import argparse
import json
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
from desktopdata import shards as SH  # noqa: E402


def _grey():
    """A 320 x 200 frame, dark strokes on light grey: a labelled line (rows 20-36, x 20-200), a UI word in an ignore
    area (rows 100-116, x 40-120) and a blank ignore area (rows 150-170, x 200-300)."""
    g = np.full((200, 320), 200, np.uint8)
    xs = np.arange(320)
    for y0, x0, x1 in ((20, 20, 200), (100, 40, 120)):
        st = (xs % 5 < 2) & (xs >= x0) & (xs < x1)
        g[y0:y0 + 16][:, st] = 30
        g[y0 + 7, x0:x1] = 30
    return g


def _smp():
    return {"src": "realplus", "kind": "pc", "vn": "v1", "shot": "v1_10", "w": 320, "h": 200,
            "lines": [[20, 20, 200, 36, 0, "text", 1.0]],
            "ignore": [[36, 96, 124, 120], [196, 146, 304, 174]], "meta": {"sha256": "a" * 64}}


def test_pc_candidates():
    g, smp = _grey(), _smp()
    word = [38, 98, 122, 118, 0.8]          # MeikiOCR's box of the UI word: kept, refined to its ink
    weak = [38, 98, 122, 118, 0.4]          # below 0.5: never
    blank = [198, 148, 302, 172, 0.9]       # an ignore area with no ink
    line = [18, 18, 202, 38, 0.95]          # the labelled line: not in an ignore area
    adds, why = C.pc_edit_candidates(smp, g, [word, weak, blank, line])
    assert len(adds) == 1
    (b,) = adds
    assert 38 <= b[0] <= 42 and 98 <= b[1] <= 101 and 118 <= b[2] <= 122 and 115 <= b[3] <= 118
    assert b[4:6] == [0, "notdlg"] and b[6] == pytest.approx(__import__("textspot6_pages").meiki_weight(0.8), abs=1e-4)
    assert why["not_ignore"] == 1 and why["no_ink"] + why["degenerate"] == 1
    adds_t, _ = C.pc_edit_candidates(smp, g, [word], cls="text")
    assert [a[:5] for a in adds_t] == [a[:5] for a in adds] and adds_t[0][5] == "text"
    # a duplicate of a kept box (the surer first) is left out
    adds2, why2 = C.pc_edit_candidates(smp, g, [word, [39, 99, 121, 117, 0.6]])
    assert len(adds2) == 1 and why2["duplicate"] == 1


def test_pc_candidates_overlap_and_clip():
    g, smp = _grey(), _smp()
    smp = dict(smp, ignore=[[10, 0, 210, 60]])          # an ignore area over the labelled line
    adds, why = C.pc_edit_candidates(smp, g, [[18, 18, 202, 38, 0.9]])
    assert adds == [] and why["overlap"] == 1           # MeikiOCR's box of a positive is not added again


def _tree(tmp_path):
    g = _grey()
    other = dict(_smp(), vn="v2", shot="v2_20", meta={"sha256": "b" * 64})
    SH.write_shard(tmp_path / "pages" / "desk" / "rp_000", [_smp(), other], [g, g])
    rec = SH.shard_record(tmp_path / "pages" / "desk" / "rp_000", "desk")
    (tmp_path / "rp.json").write_text(json.dumps({"shards": [rec]}))
    (tmp_path / "meiki").mkdir()
    (tmp_path / "meiki" / "meiki_0of1.jsonl").write_text(
        json.dumps({"shot": "v1_10", "boxes": [[38, 98, 122, 118, 0.8]]}) + "\n" +
        json.dumps({"shot": "v2_20", "boxes": [[38, 98, 122, 118, 0.7]]}) + "\n")
    (tmp_path / "dense.json").write_text(json.dumps({"frames": [{"shot": "v1_10"}]}))
    (tmp_path / "hard.json").write_text(json.dumps({"frames": [{"shot": "v2_20"}]}))
    return argparse.Namespace(manifest=str(tmp_path / "rp.json"), root=str(tmp_path / "pages"),
                              meiki=str(tmp_path / "meiki"), dense_manifest=str(tmp_path / "dense.json"),
                              hard_manifest=str(tmp_path / "hard.json"), cls="notdlg",
                              out=str(tmp_path / "edits"), workers=1)


def test_cmd_pcedits_load_apply(tmp_path, monkeypatch):
    import tsdesk_dense_source as DS
    monkeypatch.setattr(DS, "exclusion_sets", lambda: {"vns": {"v2"}, "sha256": set(), "shots": set()})
    monkeypatch.setattr(DS, "hits", lambda r, ex: ["vn"] if r["vn"] in ex["vns"] else [])
    C.cmd_pcedits(_tree(tmp_path))
    edits, doc = C.load_pc_edits(tmp_path / "edits")
    assert set(edits) == {"v1_10"} and doc["scope"] == "pc" and doc["skipped"] == {"frames": 1}   # v2: excluded
    assert doc["by_source"]["desk_dense"] == {"frames": 1, "frames_edited": 1, "lines_added": 1}
    assert doc["by_source"]["desk_hard"]["frames_edited"] == 0
    smp = _smp()
    es = C.apply_pc_edits(smp, edits)
    assert len(es["lines"]) == 2 and es["lines"][-1][5] == "notdlg" and len(smp["lines"]) == 1
    assert C.apply_pc_edits(dict(smp, w=640), edits) is None                     # another size: not applied
    assert C.apply_pc_edits(dict(smp, shot="v9_1"), edits) is None
    name = next(iter(doc["files"]))
    (tmp_path / "edits" / name).write_text("{}\n")
    with pytest.raises(SystemExit):                                              # a changed file is refused
        C.load_pc_edits(tmp_path / "edits")


def test_trainer_pc_edit_pools(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    import tsdesk_dense_source as DS
    import tsdesk_det as TD
    monkeypatch.setattr(DS, "exclusion_sets", lambda: {"vns": set(), "sha256": set(), "shots": set()})
    monkeypatch.setattr(DS, "hits", lambda r, ex: [])
    C.cmd_pcedits(_tree(tmp_path))

    class Pool:
        def __init__(self):
            self.items = [(0, _smp()), (0, dict(_smp(), shot="v7_70"))]

    class Sources:
        extra_pools = {n: Pool() for n in ("desk_realplus", "desk_dense", "desk_hard_v2", "desk_synth", "desk_hook")}
    src = Sources()
    rec = TD.edit_pools(src, None, [tmp_path / "edits"])
    for n in ("desk_realplus", "desk_dense", "desk_hard_v2"):
        assert rec[n] == {"samples_edited": 1, "lines_added": 1}
        assert src.extra_pools[n].items[0][1]["lines"][-1][5] == "notdlg"
        assert len(src.extra_pools[n].items[1][1]["lines"]) == 1
    for n in ("desk_synth", "desk_hook"):                                       # not 1b's frames: untouched
        assert n not in rec and all(len(s["lines"]) == 1 for _, s in src.extra_pools[n].items)
    assert rec["scope"] == "pc" and TD.pc_source("desk_dense") and not TD.pc_source("desk_densest")
