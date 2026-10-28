"""Tests of VOCR6-V's tools (issue #159): eval/vocr6v.py, data/vertical_hooked.py, data/manga109_covers.py,
synth/vertical_render.py's generation 3 and the frozen protocol. No data, model or network needed."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "eval", ROOT / "data", ROOT / "synth"):
    sys.path.insert(0, str(p))

import vocr6v  # noqa: E402

import vertical_hooked as VH  # noqa: E402


def _unit(uid, refs, region=(100, 50, 200, 300), ignore=(), title="T", page="p1"):
    return {"id": uid, "title": title, "page": page, "vertical": True, "region": list(region),
            "refs": [{"box": b, "text": t} for b, t in refs], "ignore": [list(b) for b in ignore]}


def test_unit_errors_order_names_and_failures():
    u = _unit("T:p1:tight", [([250, 60, 20, 200], "あいう"), ([220, 60, 20, 200], "えお")],
              ignore=[(280, 60, 18, 60)])
    # region-relative boxes: right column first, then the left one; a name inside an ignore box is left out
    r = {"rc": 0, "lines": [{"box": [180, 10, 18, 60], "text": "名前"}, {"box": [150, 10, 20, 200], "text": "あいう"},
                            {"box": [120, 10, 20, 200], "text": "えお"}]}
    assert vocr6v.unit_errors(u, r) == (0, 5)
    r_bad = dict(r, lines=r["lines"][1:2])
    assert vocr6v.unit_errors(u, r_bad) == (2, 5)
    assert vocr6v.unit_errors(u, {"rc": -7, "lines": r["lines"]}) == (5, 5)  # a failed job reads nothing


def test_unit_errors_mostly_inside_is_half_the_line():
    """A line is left out when at least half of its box lies inside another text box of the page (a name tag):
    60% inside is dropped, 40% inside is kept (Rubi tools/vertical_eval.py mostly_inside)."""
    region = (100, 50, 300, 300)
    ignore = [(280, 60, 18, 60)]                       # page pixels: x 280-298, y 60-120
    u = _unit("T:p1:tight", [([150, 60, 20, 200], "あいう")], region=region, ignore=ignore)
    name60 = {"box": [180, 10, 18, 100], "text": "名前"}  # page [280, 60, 18, 100]: 1,080 of 1,800 px inside
    col = {"box": [50, 10, 20, 200], "text": "あいう"}
    assert vocr6v.unit_errors(u, {"rc": 0, "lines": [name60, col]}) == (0, 3)
    u2 = _unit("T:p2:tight", [([150, 60, 20, 200], "あいう"), ([280, 60, 18, 150], "えお")], region=region,
               ignore=ignore)
    line40 = {"box": [180, 10, 18, 150], "text": "えお"}  # page [280, 60, 18, 150]: 1,080 of 2,700 px inside
    assert vocr6v.unit_errors(u2, {"rc": 0, "lines": [col, line40]}) == (0, 5)


def test_cluster_keys_and_cohorts():
    regions = {"v1_10_r0": {"cluster": "c1", "cohort": "vn", "page": "v1_10"},
               "bj01_p009_w1_r2": {"cluster": "blackjack", "cohort": "manga", "page": "bj01_p009_w1"}}
    u1 = _unit("v1_10_r0:tight", [])
    u2 = _unit("bj01_p009_w1_r2:tight", [])
    assert vocr6v.unit_key(u1, "cluster", regions) == "c1"
    assert vocr6v.unit_key(u2, "book", regions) == "bj01_p009"
    assert vocr6v.unit_key(u1, "page", regions) == "T:p1"
    assert vocr6v.cohort_of(u2, regions) == "manga"


def test_boot_diff_sign_and_pairing():
    x = [{"a": [10, 100], "b": [10, 100], "c": [10, 100]}]
    y = [{"a": [5, 100], "b": [4, 100], "c": [6, 100]}, {"a": [5, 100], "b": [5, 100], "c": [5, 100]}]
    point, ci = vocr6v.boot_diff(x, y, n=500)
    assert point == pytest.approx(-5.0)
    assert ci[0] <= point <= ci[1] < 0
    same, ci0 = vocr6v.boot_diff(x, x, n=200)
    assert same == 0 and ci0 == [0.0, 0.0]


def _decide(tmp_path, rule, h, v, g):
    proto = {"comparisons": {"c": {"rule": rule}}}
    paths = []
    for name, obj in (("p", proto), ("h", h), ("v", v), ("g", g)):
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(obj))
        paths.append(str(p))
    return paths


def test_decide_rules(tmp_path, capsys):
    good_h = {"diff": -2.0, "ci95": [-3.0, -1.0], "pairs": [-2.1, -1.9, -2.0]}
    level = {"diff": 0.1, "ci95": [-0.5, 0.7]}
    worse = {"diff": 1.0, "ci95": [0.2, 1.8]}
    for rule, h, v, g, want in (("adopt", good_h, level, level, True),
                                ("adopt", dict(good_h, pairs=[-2.0, 0.1, -1.0]), level, level, False),
                                ("adopt", good_h, worse, level, False),
                                ("adopt", good_h, level, worse, False),
                                ("manga", level, level, {"diff": -1.0, "ci95": [-1.5, -0.2]}, True),
                                ("manga", level, level, level, False),
                                ("fold", worse, level, level, True),
                                ("fold", level, worse, level, False),
                                # each guard on its own (audit-vocr-manga109 F3): the hval interval's upper end,
                                # the manga rule's dev VN and hval guards, the fold rule's manga guard
                                ("adopt", {"diff": -0.3, "ci95": [-1.0, 0.5], "pairs": [-0.2, -0.3, -0.4]}, level,
                                 level, False),
                                ("manga", level, worse, {"diff": -1.0, "ci95": [-1.5, -0.2]}, False),
                                ("manga", worse, level, {"diff": -1.0, "ci95": [-1.5, -0.2]}, False),
                                ("fold", level, level, worse, False)):
        p, hp, vp, gp = _decide(tmp_path, rule, h, v, g)
        vocr6v.main(["decide", "--protocol", p, "--comparison", "c", "--hval", hp, "--vn", vp, "--manga", gp])
        assert json.loads(capsys.readouterr().out)["adopted"] is want, (rule, h, v, g)


def test_test_once_records_every_input_under_its_own_key(tmp_path):
    """The attempt record keys the seven inputs apart: the public and private sets share their file names
    (audit-vocr-manga109 F2: the read of 2026-10-28 lost the public sets' hashes to that collision)."""
    import argparse

    import vocr6v_test_once as T
    for part in ("pub", "priv"):
        (tmp_path / part).mkdir()
        for name in ("columns_test.jsonl", "regions_test.jsonl"):
            (tmp_path / part / name).write_text(f"{part} {name}\n", encoding="utf-8")
    for name in ("old.vocr", "new.vocr", "tool"):
        (tmp_path / name).write_bytes(name.encode())
    a = argparse.Namespace(sets=str(tmp_path / "pub"), private=str(tmp_path / "priv"),
                           blob_old=str(tmp_path / "old.vocr"), blob_new=str(tmp_path / "new.vocr"),
                           tool=str(tmp_path / "tool"))
    files = T.input_paths(a)
    assert len(files) == 7 and len({str(p) for p in files.values()}) == 7
    assert len({T.sha(p) for p in files.values()}) == 7
    assert files["public/columns_test.jsonl"].parent.name == "pub"
    assert files["private/columns_test.jsonl"].parent.name == "priv"
    doc = T.attempt_doc(a, "0" * 40, "2026-10-28T00:00:00Z")   # the record main() writes (audit -b n1)
    assert set(doc["inputs"]) == set(doc["paths"]) == set(files) and len(doc["inputs"]) == 7
    assert len(set(doc["inputs"].values())) == 7
    assert doc["inputs"]["public/columns_test.jsonl"] != doc["inputs"]["private/columns_test.jsonl"]


def test_console_column_classify():
    cols = [{"cells": [(10, 10), (10, 30), (10, 50)]}, {"cells": [(40, 10), (40, 30)]}]
    assert VH.classify((0, 0, 20, 60), cols, []) == ("whole", 0)
    assert VH.classify((0, 0, 20, 40), cols, []) == ("partial", None)
    assert VH.classify((0, 0, 50, 60), cols, []) == ("merged", None)
    assert VH.classify((0, 0, 20, 60), cols, [(10, 55)]) == ("other", None)
    assert VH.classify((100, 100, 5, 5), cols, []) == ("empty", None)
    assert VH.tight_region([{"box": [50, 20, 20, 100]}, {"box": [80, 25, 20, 90]}], 960, 544) == [38, 8, 74, 124]
    assert VH.tight_region([{"box": [2, 2, 20, 100]}], 960, 544) == [0, 0, 34, 114]


def test_render_gen3_excludes_the_vertical_sets_vns():
    import vertical_render as VR
    vns = VR.vertical_set_vns()
    assert {"v17343", "v3259", "v11713"} <= vns          # Yunohana (test), Harukanaru (dev), Trick x Logic (owner)
    assert not any(v.startswith("bj") for v in vns)      # the manga volumes are not VNs
    assert VR.GEN_VERSION == 3


def test_cover_detection_matching():
    import manga109_covers as MC
    assert MC.iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert MC.iou([0, 0, 10, 10], [20, 20, 5, 5]) == 0.0
    refs = [[0, 0, 10, 40], [20, 0, 10, 40]]
    dets = [[1, 1, 10, 40], [100, 0, 10, 10]]
    assert MC.matched(refs, dets) == 1
    assert MC.matched(refs, []) == 0


def test_cover_labels_need_an_engine(tmp_path):
    import manga109_covers as MC
    reads = tmp_path / "reads.jsonl"
    rows = [{"page": "m109_A_w0", "split": "m109s", "columns": [
        {"k": 0, "reads": {"meikid": "あいう", "v6s": "あいう"}},
        {"k": 1, "reads": {"meikid": "かき", "v6s": "かさ"}},
        {"k": 2, "reads": {"meikid": "x", "v6s": "y"}}]}]
    reads.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    review = tmp_path / "review"
    review.mkdir()
    sheet = {"sheet": "S001", "entries": [
        {"n": 1, "id": "m109_A_w0_c0", "candidates": [{"c": 1, "text": "あいう"}]},
        {"n": 2, "id": "m109_A_w0_c1", "candidates": [{"c": 1, "text": "かき"}, {"c": 2, "text": "かさ"}]},
        {"n": 3, "id": "m109_A_w0_c2", "candidates": [{"c": 1, "text": "x"}]}]}
    (review / "S001.json").write_text(json.dumps(sheet, ensure_ascii=False), encoding="utf-8")
    dec = [{"sheet": "S001", "n": 1, "pick": 1}, {"sheet": "S001", "n": 2, "text": "かく"},
           {"sheet": "S001", "n": 3, "skip": "ui"}]
    (review / "decisions.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in dec),
                                            encoding="utf-8")
    out = tmp_path / "labels.jsonl"

    class A:
        pass
    a = A()
    a.reads, a.review, a.out = str(reads), str(review), str(out)
    MC.cmd_labels(a)
    got = {json.loads(x)["id"]: json.loads(x) for x in out.read_text(encoding="utf-8").splitlines()}
    assert got["m109_A_w0_c0"]["why"] == "label" and got["m109_A_w0_c0"]["text"] == "あいう"
    assert got["m109_A_w0_c1"]["why"] == "no_engine_agrees"
    assert got["m109_A_w0_c2"]["why"] == "ui"
    assert all(r["reviewer_kind"] == "assistant" for r in got.values())


def test_protocol_frozen_and_consistent():
    p = json.loads((ROOT / "eval" / "vocr6v_protocol.json").read_text(encoding="utf-8"))
    assert p["incumbent"]["sha256"] == "ccd20c63bb2620f054538b456904bdf19b74d483d3da8de865009e282bb04172"
    for arm in ("V6h", "V6hm"):
        w = sum(float(s.split(":")[1]) for s in p["arms"][arm]["src"])
        assert w == pytest.approx(1.0)
    assert p["recipe"]["seeds"] == [1, 2, 3]
    assert set(p["comparisons"]) == {"V6h_vs_VOCR5V", "V6hm_vs_V6h", "fold"}
    assert {c["rule"] for c in p["comparisons"].values()} <= set(p["rules"])
    assert (ROOT / "eval" / "vocr6v_rubi_measure.patch").read_text().count("VJO_MEASURE") >= 2


def test_rotation_matches_the_twin():
    g = np.arange(12, dtype=np.uint8).reshape(4, 3)
    import vertical_twin as VT
    r = VT.rotate_ccw(g)
    assert r.shape == (3, 4) and r[0, 0] == g[0, 2]
