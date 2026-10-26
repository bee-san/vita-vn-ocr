"""eval/vocr6_baseline.py: the bootstrap, the summaries, textspot references from page-pool rows and the cascade."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "data"))
import vocr6_baseline as B  # noqa: E402

import hooked_audit as HA  # noqa: E402


def test_boot_ratio_point_and_interval():
    r = B.boot_ratio([1, 0, 3, 0], [10, 10, 10, 10], ["a", "a", "b", "c"], n=2000)
    assert r["cer"] == 10.0 and r["errors"] == 4 and r["chars"] == 40 and r["clusters"] == 3
    lo, hi = r["ci95"]
    assert 0 <= lo <= 10.0 <= hi <= 30.0                        # all-b resamples give 3 / 10 = 30%, the maximum
    one = B.boot_ratio([2], [20], ["a"])
    assert one["cer"] == 10.0 and one["ci95"] is None           # one cluster: no interval
    assert B.boot_ratio([], [], [])["cer"] is None


def test_summarize_splits_platforms_and_falls_back_to_pages():
    rows = [("T1", "vita", "001", 1, 10), ("T1", "vita", "002", 0, 10), ("T2", "psp", "001", 2, 10)]
    s = B.summarize(rows)
    assert s["vita"]["cer"] == 5.0 and s["vita"]["cluster"] == "page"     # one Vita title: page clusters
    assert s["psp"]["cer"] == 20.0
    assert s["all"]["cer"] == 10.0 and s["all"]["cluster"] == "title" and s["all"]["titles"] == 2
    assert set(s["per_title"]) == {"T1", "T2"} and s["per_title"]["T1"]["clusters"] == 2


def test_spot_refs_boxes_ignores_and_scale():
    row = {"lines": [{"id": "T:1:line1", "kind": "line", "vertical": False, "box": [10, 20, 100, 30],
                      "box_main": [10, 30, 100, 20], "text": "あい", "box_index": [1], "tall_band": False},
                     {"id": "T:1:ruby1", "kind": "ruby", "vertical": False, "box": [40, 20, 20, 10],
                      "box_main": None, "text": "か", "box_index": [2]},
                     {"id": "T:1:name3", "kind": "name", "vertical": False, "box": [5, 80, 40, 15],
                      "box_main": None, "text": "名", "box_index": [3]}],
           "boxes": [{"index": i, "xyxy": [0, 0, 1, 1]} for i in (1, 2, 3)] +
                    [{"index": 4, "xyxy": [200.5, 100.0, 250.25, 120.0]}]}
    refs = B.spot_refs(row, 2)
    main = [r for r in refs if not r["ignore"]]
    assert [r["cls"] for r in main] == ["dialogue", "name"]
    assert main[0]["box"] == [20, 60, 200, 40] and main[0]["text"] == "あい"   # the main text alone, x2
    ign = [r for r in refs if r["ignore"]]
    assert {r["why"] for r in ign} == {"furigana", "undecided"}
    und = [r for r in ign if r["why"] == "undecided"][0]          # the renderer box no record covers
    assert und["box"] == [401, 200, 100, 40]


def test_decodes_cascade_uses_the_next_tier_below_700(monkeypatch):
    import pyvocr
    monkeypatch.setattr(pyvocr, "post_counters", lambda t, lib=None: (t, 0))
    reads = {"S": {"keys": ["a", "b"], "greedy": ["g1", "g2"], "lm": ["s1", "s2"], "lm_reread": ["S1", "S2"],
                   "conf": [699, 700]},
             "N": {"keys": ["a", "b"], "greedy": ["n1", "n2"], "lm": ["n1", "n2"], "lm_reread": ["N1", "N2"],
                   "conf": [100, 900]}}
    d = B.decodes(reads, None)
    assert d["S"]["rubi"] == {"a": "S1", "b": "S2"} and d["S"]["rubi_h2"] == {"a": "s1", "b": "s2"}
    assert d["S"]["rubi_cascade"] == {"a": "N1", "b": "S2"} and d["S"]["cascaded"] == 1
    assert "rubi_cascade" not in d["N"]                            # Large not read: Normal's cascade is undefined


def test_wilson_interval():
    lo, hi = HA.wilson(0, 100)
    assert lo == 0 and 0.03 < hi < 0.04
    lo, hi = HA.wilson(5, 320)
    assert lo < 5 / 320 < hi
    assert HA.wilson(0, 0) is None


def test_arms_are_the_released_files():
    assert B.ARMS["N"]["two"] and B.ARMS["L"]["two"] and not B.ARMS["S"]["two"]
    assert all(B.ARMS[t]["gate"] for t in "SNL") and not B.ARMS["V"]["gate"]
    assert all(len(a["sha256"]) == 64 for a in B.ARMS.values()) and len(B.LM_SHA) == 64
    assert np.isclose(sum(1 for _ in B.TIERS), 3)


M = ROOT / "data" / "manifests" / "vocr6_hooked"


def committed_sections():
    import json
    load = lambda p: json.loads(p.read_text(encoding="utf-8"))  # noqa: E731
    return HA.table_sections(load(M / "index.json"), load(M / "stats_v1.json"), load(M / "overlap_v1.json"),
                             load(M / "label_audit_v1.json"), load(ROOT / "eval" / "vocr6_baseline_20261026.json"))


def test_doc_tables_are_the_committed_records():
    """docs/vocr6-data.md's generated blocks equal what data/hooked_audit.py tables makes of the committed JSON."""
    doc = (ROOT / "docs" / "vocr6-data.md").read_text(encoding="utf-8")
    sec = committed_sections()
    names = [m[1] for m in HA.MARK.findall(doc)]
    assert sorted(names) == sorted(sec), names
    assert HA.splice(doc, sec) == doc, "rerun: data/hooked_audit.py tables ... --doc docs/vocr6-data.md"


def test_filters_table_rows_sum_to_the_records():
    for line in committed_sections()["filters"].splitlines()[2:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        nums = [int(c.replace(",", "")) for c in cells[1:3] + cells[3:6]]
        exact, near = (int(c.replace(",", "")) for c in cells[6].split("/"))
        rest = [int(c.replace(",", "")) for c in cells[7:]]
        assert nums[0] == sum(nums[1:]) + exact + near + sum(rest), line


def test_splice_replaces_only_marked_blocks():
    doc = "a\n<!-- generated:x -->\nold\n<!-- /generated:x -->\nb <!-- generated:y -->\nold\n<!-- /generated:y -->\n"
    out = HA.splice(doc, {"x": "| 1 |", "y": "two"})
    assert out == "a\n<!-- generated:x -->\n| 1 |\n<!-- /generated:x -->\nb <!-- generated:y -->\ntwo\n" \
                  "<!-- /generated:y -->\n"
    assert HA.splice(out, {"x": "| 1 |", "y": "two"}) == out


def test_dev_components_leave_the_excluded_vns_out():
    class Comp:
        def __init__(self, errors, chars, vns, names):
            self.models = {"m": np.asarray(errors, float)}
            self.chars, self.vns, self.is_name = np.asarray(chars, float), np.asarray(vns), np.asarray(names, bool)

    class VS:
        SEEDS = {"tv": 0, "dn": 1, "wd": 2}
        WEIGHTS = {"tv": 0.25, "dn": 0.25, "wd": 0.5}
    comps = {"tv": Comp([1, 9, 0], [10, 10, 10], ["v1", "v18432", "v2"], [False, False, False]),
             "dn": Comp([2, 0, 5], [10, 10, 10], ["v1", "v2", "v1"], [False, False, True]),      # a name row
             "wd": Comp([0, 4], [20, 20], ["v12188", "v3"], [False, False])}
    full = B.dev_components(comps, "m", VS)
    assert full["tv"]["cer"] == round(100 * 10 / 30, 4) and full["dn"]["cer"] == 10.0 and full["wd"]["cer"] == 10.0
    ex = B.dev_components(comps, "m", VS, exclude=B.DEV_COUPLED)
    assert ex["tv"]["cer"] == 5.0 and ex["tv"]["vns"] == 2                 # v18432's row is gone
    assert ex["wd"]["cer"] == 20.0 and ex["wd"]["vns"] == 1                # v12188's row is gone
    assert ex["score"] == round(0.25 * 5.0 + 0.25 * 10.0 + 0.5 * 20.0, 5) and ex["excluded_vns"] == ["v12188", "v18432"]
