"""Pools v2.1's committed records (docs/vocr6-data-v2.md): the doc's generated tables, the v1 / v2.1 comparison and
eval/hooked_v2_baseline.py's aggregation."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "data", ROOT / "eval"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
import hooked_v2_baseline as HB  # noqa: E402

import hooked_audit as HA  # noqa: E402

M1 = ROOT / "data" / "manifests" / "vocr6_hooked"
M2 = ROOT / "data" / "manifests" / "vocr6_hooked_v2"
BASELINE = ROOT / "eval" / "hooked_v2_baseline_20261101.json"


def load(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def committed_sections():
    return HA.v2_sections(load(M1 / "index.json"), load(M1 / "stats_v1.json"), load(M2 / "index.json"),
                          load(M2 / "stats_v2.json"), load(M2 / "overlap_v2.json"), load(M2 / "label_audit_v2.json"),
                          load(BASELINE))


def test_v2_doc_tables_are_the_committed_records():
    """docs/vocr6-data-v2.md's generated blocks equal what data/hooked_audit.py tables2 makes of the committed JSON."""
    doc = (ROOT / "docs" / "vocr6-data-v2.md").read_text(encoding="utf-8")
    sec = committed_sections()
    names = [m[1] for m in HA.MARK.findall(doc)]
    assert sorted(names) == sorted(sec), names
    assert HA.splice(doc, sec) == doc, "rerun: data/hooked_audit.py tables2 ... --doc docs/vocr6-data-v2.md"


def test_compare_platforms_add_up_and_v1_is_the_v1_record():
    n1 = HA.corpus_numbers(load(M1 / "index.json"), load(M1 / "stats_v1.json"))
    n2 = HA.corpus_numbers(load(M2 / "index.json"), load(M2 / "stats_v2.json"))
    for n in (n1, n2):
        for k, v in n["all"].items():
            assert n["vita"][k] + n["psp"][k] == v, k
    assert n1["all"]["usable: dialogue lines"] == 81977 and n1["all"]["pages"] == 55493   # docs/vocr6-data.md
    assert n1["all"]["titles"] == 49 and n2["all"]["titles"] == 138 and n2["all"]["pages"] == 82185


def test_v2_index_has_no_hooked_text_fields():
    idx = load(M2 / "index.json")
    assert idx["pool_tag"] == "hooked-v2" and idx["schema"] == "vocr6-hooked/2"
    row = json.loads((M2 / f"{sorted(idx['titles'])[0]}.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert set(row) <= {"page", "sha256", "split", "complete", "lines", "src"}


def unit(title, page, errors, chars, platform="vita"):
    return {"uid": f"{title}:{page}:0:base", "cluster": title, "platform": platform, "errors": errors, "chars": chars,
            "lines": 1, "whole": 1, "cut": 0, "missed": 0, "merged": 0, "junk": 0}


def test_group_summary_intervals():
    us = [unit(f"T{i}", "000001", i, 100) for i in range(6)]
    g = HB.group_summary(us)
    assert g["cer"] == round(100 * 15 / 600, 4) and g["ci_over"] == "titles" and g["titles"] == 6
    lo, hi = g["ci95"]
    assert lo <= g["cer"] <= hi and lo >= 0
    few = HB.group_summary([unit("A", f"{p:06d}", p % 3, 50) for p in range(1, 9)])      # one title: over pages
    assert few["ci_over"] == "pages" and few["ci95"] is not None
    assert HB.pooled_ci([unit("A", "000001", 1, 10)], lambda u: u["cluster"]) is None   # one cluster: no interval


def test_summarize_splits_new_and_v1_titles():
    res = {"units": [unit("N1", "000001", 2, 100), unit("N1", "000002", 0, 100, "psp"),
                     unit("V1", "000001", 1, 50)]}
    g = HB.summarize(res, {"N1"})
    assert g["new"]["chars"] == 200 and g["v1"]["chars"] == 50 and g["all"]["errors"] == 3
    assert g["new:psp"]["chars"] == 100 and g["new:vita"]["errors"] == 2 and "v1:psp" not in g
    assert HB.per_title(res)["N1"] == {"errors": 2, "chars": 200, "units": 2, "cer": 1.0}
