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


def test_follow_up_tables_are_rendered_from_the_records():
    """Stages 4c and 4d (the follow-up, v0.4.1-vocr6.2): every table of docs/vocr6.md's follow-up section."""
    for block in VR.follow_up_tables(REC).strip().split("\n\n"):
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
        if d["stage"] in ("4b", "4c", "4d-fold", "4d-report"):      # checks against a sibling, and a report
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
    # the follow-up: S3long adopted, so 4c replaces 4b where it is not resolved worse on dev (amendment 6); 4d-l
    # adopted over VOCR5-S, 4d-r not, and 4d-l's fold-in not resolved worse than its 4a sibling (amendment 7)
    assert D["s2_S3long"]["adopt_y"] is True
    assert all(not D[f"s4c_{t}"]["total"]["dev"]["ci95"][0] > 0 for t in "NL")
    assert (D["s4d_l"]["adopt_y"], D["s4d_r"]["adopt_y"]) == (True, False)
    assert not D["s4d_lh"]["total"]["dev"]["ci95"][0] > 0
    assert D["s4d_lh"]["y"] == ["v6f4dlh_S_seed1"] and D["s4d_lh"]["x"] == ["v6f4dl_S_seed1"]
    assert {t: b["run"] for t, b in REC["shipped_blobs"].items()} == {"S": "v6f4dlh_S_seed1", "N": "v6f4c_N_seed1",
                                                                      "L": "v6f4c_L_seed1"}


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
    """The README's VOCR6 table: one row per tier for the follow-up's shipped blob (v0.4.1-vocr6.2's release name),
    with its bytes and arena, the held-out hval evidence (stage 4a's means for N and L, 4d-l's for S, against the
    released VOCR5 tier) and the shipped set's development score against VOCR5's."""
    R, D, G = REC["runs"]["shipped"], REC["decisions"], REC["follow_up_gate"]["tiers"]
    held = {"S": D["s4d_l"], "N": D["s4_N"], "L": D["s4_L"]}
    ship = {"S": "v6f4dlh_S_seed1_up4cN", "N": "v6f4c_N_seed1", "L": "v6f4c_L_seed1"}
    for t in "SNL":
        d = held[t]
        x = R[d["x"][0]]["summary"]
        hp = f"{x['hval_page']:.3f} → {_mean('shipped', d['y'], 'hval_page'):.3f}%"
        dev = f"{x['dev_score']:.3f} → {R[ship[t]]['summary']['dev_score']:.3f}"
        assert hp in DOC or t == "S", (t, hp)
        name = "VOCR6.2_" + G[t]["blob"]["name"][len("VOCR6_"):]
        row = [ln for ln in README.splitlines() if ln.startswith(f"| [`{name}`]")]
        assert len(row) == 1 and hp in row[0] and dev in row[0], (t, hp, dev)
        assert f"{G[t]['blob']['bytes']:,} B" in row[0] and f"{G[t]['arena_w8i_1024']:,} B" in row[0], t
        if G[t]["final_arm"] == "today":                 # no head section adopted: the plain blob ships
            assert G[t]["blob"]["sha256"] == REC["shipped_blobs"][t]["sha256"], t
