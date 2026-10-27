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


README = (ROOT / "README.md").read_text(encoding="utf-8")


def _mean(plan, runs, key):
    return sum(REC["runs"][plan][r]["summary"][key] for r in runs) / len(runs)


def test_every_screening_row_shows_the_records_means_and_verdict():
    """Stages 2-3 rows: the arm's mean hval page, clean, dev score (wd) and its verdict; stage 1 rows the same means."""
    lines = [ln for ln in DOC.splitlines() if ln.startswith("| ")]
    for name, d in REC["decisions"].items():
        if d["stage"] not in ("1", "2", "3"):
            continue
        rows = [ln for ln in lines if ln.split("|")[1].strip() == d["arm"]]   # A15 has a stage-1 row and a base row
        assert rows, name
        for row in rows:
            for key in ("hval_page", "hval_clean_dlg", "dev_score"):
                assert f"| {_mean(d['plan'], d['y'], key):.3f}" in row, (name, key, row[:40])
            if d["stage"] in ("2", "3"):
                assert f"({_mean(d['plan'], d['y'], 'wd'):.2f}%)" in row, name
                assert ("**adopted**" in row) == d["adopt_y"], name


def test_the_summary_and_the_readme_show_the_shipped_numbers():
    R, D = REC["runs"]["shipped"], REC["decisions"]
    for t in ("N", "L"):
        x = R[D[f"s4_{t}"]["x"][0]]["summary"]
        y = _mean("shipped", D[f"s4_{t}"]["y"], "hval_page")
        hp = f"{x['hval_page']:.3f} → {y:.3f}%"
        ship = R[D[f"s4_H{t}"]["y"][0]]["summary"]
        dev = f"{x['dev_score']:.3f} → {ship['dev_score']:.3f}"
        assert hp in DOC and hp in README, (t, hp)
        row = [ln for ln in README.splitlines() if ln.startswith(f"| [`VOCR6_{t}_w8_2l.vocr`]")]
        assert len(row) == 1 and hp in row[0] and dev in row[0], (t, dev)
        assert f"{REC['shipped_blobs'][t]['bytes'] + {'N': 1984, 'L': 1984}[t]:,} B" in row[0], t
