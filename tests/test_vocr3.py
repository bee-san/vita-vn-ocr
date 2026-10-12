"""VOCR3's frozen protocol: its candidates, baselines and gates are consistent with study 2's protocol
and the released blobs' documented sizes. No data, models or network."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
P = json.loads((ROOT / "eval" / "vocr3_protocol.json").read_text(encoding="utf-8"))
S2 = json.loads((ROOT / "eval" / "synth3_study2_protocol.json").read_text(encoding="utf-8"))


def test_candidates_are_study2_runs():
    study2_runs = {r for runs in S2["arms"].values() for r in runs}
    for size in ("FL10", "H15"):
        for r in P["candidates"][size]["study2"]:
            assert r in study2_runs and r.startswith(f"ft_{size}_")
    assert set(P["candidates"]["FL10"]["study2"]) | set(P["candidates"]["H15"]["study2"]) == study2_runs
    for r in P["candidates"]["FL10"]["step_curve"]:
        assert r.startswith("ft_FL10_C_mix_pre") and r.split("_pre")[1].split("k")[0] in ("64", "128")
        assert any(r.split("_seed")[0][3:] in s for s in S2["step_curve"]["runs"])


def test_sizes_match_study2_and_released_blobs():
    for size in ("FL10", "H15"):
        spec = P["sizes"][size]
        assert spec["vocab"]["sha256"] == S2["architectures"][size]["vocab"]["sha256"]
        assert spec["vocab"]["classes"] == S2["architectures"][size]["vocab"]["classes"]
    assert P["sizes"]["FL10"]["expected"] == {"blob_bytes": 657664, "arena_w8i_bytes": 124832}
    assert P["sizes"]["H15"]["expected"] == {"blob_bytes": 1155968, "arena_w8i_bytes": 182944}
    shas = {b["sha256"] for s in P["sizes"].values() for b in s["baselines"].values() if "sha256" in b}
    assert len(shas) == 3 and all(len(s) == 64 for s in shas)


def test_report_pairs_and_gates_name_known_arms():
    arms = set(P["locked_report"]["arms"])
    for a, b in P["locked_report"]["pairs"] + P["qualification"]["gates"]:
        assert a in arms and b in arms
    assert ["VOCR3-FL10", "VOCR1"] in P["qualification"]["gates"]
    assert ["VOCR3-H15", "VOCR_H15"] in P["qualification"]["gates"]


def test_amendment_1_w20_and_budget():
    w = P["sizes"]["W20"]
    assert w["vocab"]["sha256"] == P["sizes"]["H15"]["vocab"]["sha256"] and w["expected"]["blob_bytes"] == 1558016
    b = P["budget"]
    assert (b["weights_max_bytes"], b["arena_max_bytes"], b["ms_max_per_640px_line"]) == (1600000, 262144, 550)
    for spec in P["sizes"].values():
        assert spec["expected"]["blob_bytes"] <= b["weights_max_bytes"]
        assert spec["expected"]["arena_w8i_bytes"] <= b["arena_max_bytes"]
    runs = P["candidates"]["W20"]["runs"]
    assert sorted(runs) == sorted(f"ft_W20_{a}_seed1" for a in ("A_s2", "B_s3", "C_mix"))
    assert ["VOCR3-W20", "VOCR_H15"] in P["qualification"]["gates"]
    script = (ROOT / "eval" / "vocr3_run.sh").read_text(encoding="utf-8")
    assert "W20)  WIDTH=2.0" in script and "synth3_study2_pools.sha256" in script and "kd=1" not in script
