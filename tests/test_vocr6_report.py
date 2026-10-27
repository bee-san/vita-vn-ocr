"""docs/vocr6.md's numbers come from the committed records (eval/vocr6_results_20261027.json): the stage-4 tables are
rendered from them verbatim, and every stage 1-3 decision row shows the record's differences and intervals."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import vocr6_report as VR  # noqa: E402

DOC = (ROOT / "docs/vocr6.md").read_text(encoding="utf-8")
REC = json.loads((ROOT / "eval/vocr6_results_20261027.json").read_text(encoding="utf-8"))


def test_stage4_tables_are_rendered_from_the_records():
    for block in VR.tables(REC).strip().split("\n\n"):
        assert block in DOC, block.splitlines()[0]


def test_every_screening_decision_row_shows_the_records_numbers():
    rows = [ln for ln in DOC.splitlines() if ln.startswith("| ")]
    for name, d in REC["decisions"].items():
        if d["stage"] not in ("1", "2", "3"):
            continue
        hp, dev = VR.iv(d["total"], "hval_page"), VR.iv(d["total"], "dev")
        hit = [ln for ln in rows if ln.startswith(f"| {d['arm']} |") and hp in ln and dev in ln]
        assert hit, (name, hp, dev)


def test_the_adoption_verdicts_follow_the_rule():
    for name, d in REC["decisions"].items():
        if d["stage"] == "4b":
            continue
        t = d["total"]
        rule = (t["hval_page"]["ci95"][1] < 0 and all(p["hval_page"] < 0 for p in d["pairs"].values())
                and not t["dev"]["ci95"][0] > 0)
        assert rule == d["adopt_y"], name


def test_the_decided_tiers():
    D = REC["decisions"]
    assert [D[f"s4_{t}"]["adopt_y"] for t in "SNL"] == [False, True, True]
    assert all(D[f"s4_H{t}"]["total"]["dev"]["ci95"][0] <= 0 for t in "NL")     # 4b ships for N and L
    assert {D["s2_S2cil"]["adopt_y"]} == {True}
    assert not any(D[n]["adopt_y"] for n in ("s2_S2t5", "s2_S2x2", "s2_S2s6", "s2_S2r70", "s2_S2r30", "s2_S3attn",
                                             "s2_S3rep"))
