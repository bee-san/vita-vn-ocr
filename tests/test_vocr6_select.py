"""eval/vocr6_select.py: the VOCR6 T1 decision rule on synthetic score documents (no data needed)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))
import pytest  # noqa: E402
import vocr6_select as S  # noqa: E402


@pytest.fixture(autouse=True)
def no_seen_units(monkeypatch):
    """The synthetic documents' unit keys are not the hval plan's: no unit is train-seen unless a test says so."""
    monkeypatch.setitem(S._SEEN, "units", frozenset())


def doc(name, page_err, clean_err, dev_err, titles=8, vns=12, seed=0):
    """A score document with titles x 5 page units, titles x 6 clean lines (4 dialogue, 2 names) and the dev
    cohorts over vns VNs; *_err are per-row error counts added to a base level."""
    rng = np.random.default_rng(seed)
    t = [f"T{i}" for i in range(titles)]
    page = {"unit": [f"{x}:{k}" for x in t for k in range(5)], "title": [x for x in t for _ in range(5)],
            "platform": [("vita" if i % 2 else "psp") for i in range(titles) for _ in range(5)],
            "variant": ["base"] * (5 * titles), "chars": [100] * (5 * titles),
            "junk": [1 if k == 4 else 0 for _ in t for k in range(5)],
            "merged": [1 if k == 3 and i == 0 else 0 for i, _ in enumerate(t) for k in range(5)],
            "errors": {"rubi": list(3 + page_err + rng.integers(0, 2, 5 * titles))}}
    for i in range(4, 5 * titles, 5):          # the junk unit of each title: a lottery of insertions
        page["errors"]["rubi"][i] = int(rng.integers(0, 80))
    clean = {"id": [f"{x}:{k}" for x in t for k in range(6)], "title": [x for x in t for _ in range(6)],
             "platform": [("vita" if i % 2 else "psp") for i in range(titles) for _ in range(6)],
             "kind": ["line", "line", "line", "line", "name", "name"] * titles, "chars": [20] * (6 * titles),
             "errors": {"rubi": list(1 + clean_err + rng.integers(0, 2, 6 * titles))}}
    v = [f"v{i}" for i in range(vns)]
    fixed = {"id": [f"{x}:{k}" for x in v for k in range(4)], "vn": [x for x in v for _ in range(4)],
             "name": [False, False, False, True] * vns, "chars": [15] * (4 * vns),
             "errors": {"rubi": list(dev_err + rng.integers(0, 2, 4 * vns))}}
    wd = {"unit": [f"{x}:{k}" for x in v for k in range(3)], "vn": [x for x in v for _ in range(3)],
          "kind": ["vita"] * (3 * vns), "chars": [60] * (3 * vns),
          "errors": {"rubi": list(2 + dev_err + rng.integers(0, 2, 3 * vns))}}
    return {"name": name, "rows": {"hval_page": page, "hval_clean": clean, "tv": fixed, "dn": dict(fixed), "wd": wd}}


def test_identical_runs_difference_is_zero():
    a = doc("a", 0, 0, 0)
    out = S.compare([a], [a], n_boot=500)
    assert out["hval_page"]["delta_points"] == 0
    assert out["hval_page"]["ci95"] == [0.0, 0.0]
    assert out["dev"]["delta_points"] == 0


def test_better_y_is_adopted_and_worse_dev_blocks_it():
    x1, x2 = doc("x1", 2, 1, 0, seed=1), doc("x2", 2, 1, 0, seed=2)
    y1, y2 = doc("y1", 0, 0, 0, seed=3), doc("y2", 0, 0, 0, seed=4)
    d = S.decide([x1, x2], [y1, y2], n_boot=2000)
    assert d["total"]["hval_page"]["delta_points"] < 0
    assert d["checks"]["hval_page_ci_below_zero"] and d["checks"]["every_pair_hval_page_below_zero"]
    assert d["adopt_y"]
    z1, z2 = doc("z1", 0, 0, 3, seed=5), doc("z2", 0, 0, 3, seed=6)   # better hval, resolved worse dev
    d = S.decide([x1, x2], [z1, z2], n_boot=2000)
    assert d["checks"]["hval_page_ci_below_zero"]
    assert not d["checks"]["dev_not_resolved_worse"]
    assert not d["adopt_y"]


def test_names_and_name_lines_are_left_out_of_dialogue():
    a = doc("a", 0, 0, 0)
    e, c, cl, keys = S.component(a, "hval_clean", "rubi")
    assert len(e) == 4 * 8 and set(cl) == {f"T{i}" for i in range(8)}
    e, c, cl, keys = S.component(a, "tv", "rubi")
    assert len(e) == 3 * 12


def test_platform_split_and_row_mismatch():
    a, b = doc("a", 0, 0, 0), doc("b", 1, 0, 0)
    out = S.compare([a], [b], n_boot=500)
    assert set(out["hval_page"]) >= {"vita", "psp"}
    assert out["hval_page"]["vita"]["delta_points"] > 0
    c = doc("c", 0, 0, 0, titles=7)
    try:
        S.compare([a], [c], n_boot=10)
    except SystemExit as e:
        assert "other rows" in str(e)
    else:
        raise AssertionError("rows of another plan were compared")


def test_junk_units_are_out_of_the_decision_metric():
    a, b = doc("a", 0, 0, 0, seed=7), doc("b", 0, 0, 0, seed=8)
    e, c, cl, keys = S.component(a, "hval_page", "rubi")
    assert len(e) == 4 * 8
    e, c, cl, keys = S.component(a, "hval_page_all", "rubi")
    assert len(e) == 5 * 8
    out = S.compare([a], [b], n_boot=2000)
    width = lambda k: out[k]["ci95"][1] - out[k]["ci95"][0]   # noqa: E731
    assert width("hval_page") < width("hval_page_all")


def test_seen_units_leave_the_decision_metric(monkeypatch):
    a = doc("a", 0, 0, 0, seed=9)
    monkeypatch.setitem(S._SEEN, "units", frozenset({"T0:0", "T1:1"}))
    e, c, cl, keys = S.component(a, "hval_page", "rubi")
    assert len(e) == 4 * 8 - 2 and "T0:0" not in keys
    e, c, cl, keys = S.component(a, "hval_page_seen", "rubi")
    assert len(e) == 4 * 8 and "T0:0" in keys
    out = S.compare([a], [a], n_boot=200)
    assert {"hval_page", "hval_page_seen", "hval_page_all"} <= set(out)


def test_merged_units_report_column():
    a = doc("a", 0, 0, 0, seed=10)
    e, c, cl, keys = S.component(a, "hval_page_nomerge", "rubi")
    assert len(e) == 4 * 8 - 1 and "T0:3" not in keys
    assert "hval_page_nomerge" in S.compare([a], [a], n_boot=100)


def test_excluded_units_leave_every_page_component(monkeypatch):
    a = doc("a", 0, 0, 0, seed=11)
    monkeypatch.setitem(S._EXCLUDE, "units", frozenset({"T1:0", "T2:2"}))
    e, c, cl, keys = S.component(a, "hval_page", "rubi")
    assert len(e) == 4 * 8 - 2 and "T1:0" not in keys
    e, c, cl, keys = S.component(a, "hval_page_all", "rubi")
    assert len(e) == 5 * 8 - 2
    assert len(S.platform_mask(a, "hval_page", "vita")) == 4 * 8 - 2
    e, c, cl, keys = S.component(a, "hval_clean", "rubi")
    assert len(e) == 4 * 8
