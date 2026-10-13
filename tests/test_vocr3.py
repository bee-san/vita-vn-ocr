"""VOCR3's frozen protocol: its candidates, baselines and gates are consistent with study 2's protocol
and the released blobs' documented sizes. No data, models or network."""
import json
import sys
from pathlib import Path

import numpy as np

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


def test_amendment_2_h15_b_s3():
    assert P["candidates"]["H15"]["extra"] == ["ft_H15_B_s3_seed1"]
    assert len(P["amendments"]) >= 2 and "ft_H15_B_s3_seed1" in P["amendments"][1]["change"]


def _scores(path, shifts, n=200, vns=20, seed=0, tag=""):
    rng = np.random.default_rng(seed)
    lines = [{"id": f"{tag}l{i}", "vn": f"v{i % vns}", "ref": "あいうえお", "tags": []} for i in range(n)]
    base = rng.integers(0, 3, n)
    models = {}
    for run, (shift, sha) in shifts.items():
        e = np.clip(base + shift * (rng.random(n) < 0.6), 0, None).tolist()
        models[run] = {"blob": "x", "blob_sha256": sha, "blob_bytes": 1, "arena_bytes": 1,
                       "errors": {"rv": e, "rvn": e}, "pred": {}, "cer": {}}
    Path(path).write_text(json.dumps({"dev": {}, "lines": lines, "models": models}))
    return str(path)


def _budget(d, entries):
    (d / "a9").mkdir(parents=True)
    sizes = {}
    for name, (sha, ms) in entries.items():
        sizes[name] = {"blob_bytes": 1155968, "arena": {"t4h16": {"w8i": 182944}}}
        (d / "a9" / f"{name}_l22048.json").write_text(json.dumps(
            {"name": name, "blob_sha256": sha, "tile": 4, "l2_kib": 2048, "ms_640px": {"333": [100.0, ms]}}))
    (d / "sizes.json").write_text(json.dumps(sizes))
    return str(d)


def test_select_base_rule_budget_and_variant(tmp_path):
    sys.path.insert(0, str(ROOT / "eval"))
    import vocr3_select as SEL
    runs = {"ft_H15_A_s2_seed1": (0, "a1"), "ft_H15_A_s2_seed2": (0, "a2"), "ft_H15_C_mix_seed1": (-1, "c1"),
            "ft_H15_C_mix_seed2": (0, "c2"), "ft_H15_B_s3_seed1": (-1, "b1"), "rp": (-2, "rp")}
    tv = _scores(tmp_path / "tv.json", runs)
    dn = _scores(tmp_path / "dn_strict_ink.json", runs, n=80, seed=1, tag="d")
    bd = _budget(tmp_path / "budget", {f"m{i}": (sha, 600.0 if sha == "c1" else 400.0)
                                         for i, (_, sha) in enumerate(runs.values())})
    out = tmp_path / "sel.json"
    args = type("A", (), {"protocol": ROOT / "eval" / "vocr3_protocol.json", "size": "H15", "tv": [tv], "dn": [dn],
                          "dn_commit": "x", "budget_dir": [bd], "only": "", "out": out, "realplus": ""})()
    SEL.cmd_base(args)
    sel = json.loads(out.read_text())
    over = [r for r in sel["candidates"] if not r["budget_ok"]]
    assert [r["run"] for r in over] == ["ft_H15_C_mix_seed1"]          # 600 ms > 550: not selectable
    assert sel["selected"] == "ft_H15_B_s3_seed1" and len(sel["candidates"]) == 5
    vout = tmp_path / "var.json"
    SEL.cmd_variant(type("A", (), {"selection": out, "name": "rp", "tv": [tv], "dn": [dn], "out": vout})())
    v = json.loads(vout.read_text())
    assert v["base"] == "ft_H15_B_s3_seed1" and v["teacher_val"]["delta_points"] < 0


def test_realplus_decision_and_rp_candidates(tmp_path):
    sys.path.insert(0, str(ROOT / "eval"))
    import vocr3_select as SEL
    pairs = P["candidates"]["realplus"]["W20"]["decision_pairs"]
    runs = {}
    for i, (a, b) in enumerate(pairs):
        runs[a] = (0, f"a{i}")
        runs[b] = (-1, f"b{i}")
    tv = _scores(tmp_path / "tv.json", runs)
    dn = _scores(tmp_path / "dn.json", runs, n=80, seed=3, tag="d")
    out = tmp_path / "rp.json"
    SEL.cmd_realplus(type("A", (), {"protocol": ROOT / "eval" / "vocr3_protocol.json", "size": "W20", "tv": [tv],
                                    "dn": [dn], "out": out})())
    r = json.loads(out.read_text())
    assert r["adopted"] and r["teacher_val"]["ci95"][1] < 0 and len(r["per_pair"]) == 3
    names = SEL.candidate_names(P, "W20", r)
    assert names == [b for _, b in pairs]
    assert SEL.candidate_names(P, "W20", {**r, "adopted": False}) == P["candidates"]["W20"]["runs"]


def test_rp_runs_match_their_pretrains():
    rp = P["candidates"]["realplus"]
    script = (ROOT / "eval" / "vocr3_rp_run.sh").read_text(encoding="utf-8")
    assert 'RUN=rp_${ARCH}_${ARM}${TAG}_seed${SEED}' in script and "vocr3_realplus.sha256" in script
    assert '"tt:0.2:' in script and '"rd:0.1:' in script and '"rp:0.2:' in script and '"ps:' not in script
    for size, spec in rp.items():
        if size == "note":
            continue
        for base, run in spec["decision_pairs"]:
            assert base.startswith("ft_") and run == "rp_" + base[3:]
        cands = P["candidates"][size]
        all_base = list(cands.get("study2", [])) + list(cands.get("runs", [])) + list(cands.get("step_curve", [])) + \
            list(cands.get("extra", []))
        rp_runs = [r for _, r in spec["decision_pairs"]] + spec["extra_runs"]
        assert sorted("rp_" + b[3:] for b in all_base) == sorted(rp_runs)
