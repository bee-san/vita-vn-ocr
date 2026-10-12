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
    for size, spec in P["sizes"].items():
        assert spec["vocab"]["sha256"] == S2["architectures"][size]["vocab"]["sha256"]
        assert spec["vocab"]["classes"] == S2["architectures"][size]["vocab"]["classes"]
    assert P["sizes"]["FL10"]["expected"] == {"blob_bytes": 657664, "arena_w8i_bytes": 124832}
    assert P["sizes"]["H15"]["expected"] == {"blob_bytes": 1155968, "arena_w8i_bytes": 182944}
    shas = {b["sha256"] for s in P["sizes"].values() for b in s["baselines"].values()}
    assert len(shas) == 3 and all(len(s) == 64 for s in shas)


def test_report_pairs_and_gates_name_known_arms():
    arms = set(P["locked_report"]["arms"])
    for a, b in P["locked_report"]["pairs"] + P["qualification"]["gates"]:
        assert a in arms and b in arms
    assert ["VOCR3-FL10", "VOCR1"] in P["qualification"]["gates"]
    assert ["VOCR3-H15", "VOCR_H15"] in P["qualification"]["gates"]
