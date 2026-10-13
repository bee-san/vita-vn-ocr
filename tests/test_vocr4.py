"""VOCR4 (eval/vocr4_protocol.json): the protocol's structure, the pool exclusion rule and build, the decisions and
the selection, on toy data. No screenshots, models or network."""
import collections
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
P = json.loads((ROOT / "eval" / "vocr4_protocol.json").read_text(encoding="utf-8"))
SIZES = ("FL10", "H15", "W20")

import vocr4_pools as POOLS  # noqa: E402
import vocr4_select as SEL  # noqa: E402


def test_arms_runs_and_names():
    for arm, prefix in (("C", "v4c"), ("A", "v4a"), ("M", "v4m")):
        for size in SIZES:
            assert P["arms"][arm]["runs"][size] == [f"{prefix}_{size}_seed1", f"{prefix}_{size}_seed2"]
    script = (ROOT / "eval" / "vocr4_run.sh").read_text(encoding="utf-8")
    assert "RUN=v4$(echo \"$ARM\" | tr 'CAM' 'cam')_${SIZE}_seed${SEED}" in script
    assert "pre_${SIZE}_M3_seed${SEED}" in script
    assert subprocess.run(["bash", "-n", str(ROOT / "eval" / "vocr4_run.sh")]).returncode == 0


def _weights(block):
    return [float(w) for w in re.findall(r'--src "[a-z0-9]+:([0-9.]+):', block)]


def test_fine_tune_mixtures_sum_to_one():
    script = (ROOT / "eval" / "vocr4_run.sh").read_text(encoding="utf-8")
    nv = 0.25
    synth = sum(_weights(script.split("FSYN=(", 1)[1].split(")", 1)[0]))
    c = sum(_weights(script.split('if [ "$ARM" = C ]; then', 1)[1].split("else", 1)[0]))
    a_block = script.split('if [ "$ARM" = C ]; then', 1)[1].split("else", 1)[1].split("RUN=v4", 1)[0]
    a_base = _weights(a_block.split("if pinned own_ink.npz", 1)[0])
    a_own = _weights(a_block.split("if pinned own_ink.npz", 1)[1].split("else", 1)[0])
    a_fallback = _weights(a_block.split("if pinned own_ink.npz", 1)[1].split("else", 1)[1])
    assert abs(nv + synth - 0.5) < 1e-9
    assert abs(nv + synth + c - 1) < 1e-9
    assert abs(nv + synth + sum(a_base) + sum(a_own) - 1) < 1e-9
    assert abs(nv + synth + sum(a_base) + sum(a_fallback) - 1) < 1e-9


def test_pretrains_pinned_and_steps():
    pins = (ROOT / "eval" / "vocr4_inputs.sha256").read_text(encoding="utf-8").split("\n")
    paths = [ln.split()[1] for ln in pins if ln.strip()]
    for size in SIZES:
        assert any(p in P["sizes"][size]["pretrain_reused"] for p in paths)
        assert f"ckpt_step{P['sizes'][size]['pretrain_steps']}.pt" in P["sizes"][size]["pretrain_reused"]
    assert all(len(ln.split()[0]) == 64 for ln in pins if ln.strip())
    c = (ROOT / "eval" / "vocr4_pools_c.sha256").read_text(encoding="utf-8")
    assert {ln.split()[1] for ln in c.split("\n") if ln.strip()} == {"tt.npz", "rd.npz", "rp2_ink.npz",
                                                                  "rp2_grad.npz", "exclusions.json"}


def test_report_pairs_gates_and_planetarian():
    arms = set(P["locked_report"]["arms"])
    for a, b in P["locked_report"]["pairs"] + P["qualification"]["gates"]:
        assert a in arms and b in arms and a != b
    for size in SIZES:
        assert [f"VOCR4_{size}", f"VOCR3_{size}"] in P["locked_report"]["pairs"]
    chk = json.loads((ROOT / "eval" / "vocr4_planetarian_check_20261013.json").read_text(encoding="utf-8"))
    assert "v34" in chk["vns"] and chk["total_hits"] == 0
    assert {"vita-test.jsonl", "psp-test.jsonl"} <= set(chk["sets"])
    b = P["budget"]
    for spec in P["sizes"].values():
        assert spec["expected"]["blob_bytes"] <= b["weights_max_bytes"]
        assert spec["expected"]["arena_w8i_bytes"] <= b["arena_max_bytes"]


def _dump():
    """A toy VNDB dump in load_dump's shape: v1 protected (test-set), v2 development, v3 CLANNAD-linked, v4 CLANNAD."""
    return {"rel_vns": {"r1": {"v10", "v1"}, "r2": {"v11", "v2"}, "r3": {"v12", "v4"}, "r4": {"v13"}},
            "relations": collections.defaultdict(set, {"v14": {"v2"}, "v2": {"v14"}, "v15": {"v1"}, "v1": {"v15"}})}


def test_exclusion_rule():
    why = {"v1": "test-set", "v2": "development", "v4": "clannad", "v10": "shared-release",
           "v11": "shared-release", "v12": "shared-release", "v14": "family", "v15": "family", "v20": "val-bucket",
           "v21": "test-vn", "v34": "shared-release"}
    d = _dump()
    d["rel_vns"]["r5"] = {"v34", "v4"}
    link = POOLS.link_classes(d, why)
    ex = POOLS.exclusions(why, link, owner_vns=["v30"], dev_vns=["v31"], allow=["v34"])
    assert ex["v1"] == "test-set" and ex["v2"] == "development" and ex["v4"] == "clannad" and ex["v21"] == "test-vn"
    assert ex["v10"] == "shared-release:testset" and ex["v12"] == "shared-release:clannad"
    assert ex["v15"] == "family:testset"
    assert "v11" not in ex and "v14" not in ex and "v20" not in ex       # linked to development only; split bucket
    assert ex["v30"] == "owner-test-set" and ex["v31"] == "development-cohort" and "v34" not in ex
    with pytest.raises(SystemExit):
        POOLS.exclusions(why, link, owner_vns=["v30"], allow=["v4"])
    with pytest.raises(SystemExit):
        POOLS.exclusions(why, link, owner_vns=["v30"], allow=["v30"])


def _pool(path, rows):
    """A teacher-format npz: rows of (vn, image id, line index)."""
    imgs = [np.full((4, 3 + i), i, np.uint8) for i in range(len(rows))]
    offs = np.concatenate([[0], np.cumsum([g.size for g in imgs])]).astype(np.int64)
    meta = [{"id": f"{vn}_{img}_{k}", "vn": vn, "session": f"{vn}/r{img}/0"} for vn, img, k in rows]
    np.savez(path, data=np.concatenate([g.ravel() for g in imgs]), offsets=offs,
             shapes=np.array([g.shape for g in imgs]), labels=np.array(json.dumps(["あ"] * len(rows))),
             meta=np.array(json.dumps(meta)))
    return str(path)


def test_build_excludes_and_dedupes_by_group(tmp_path):
    ex = tmp_path / "ex.json"
    ex.write_text(json.dumps({"vns": {"v9": "test-set"}}))
    tt = _pool(tmp_path / "tt_src.npz", [("v1", 100, 0), ("v9", 101, 0), ("v1", 102, 1)])
    ink = _pool(tmp_path / "nat_ink_src.npz", [("v2", 200, 0), ("v2", 102, 0), ("v3", 201, 0)])
    grad = _pool(tmp_path / "nat_grad_src.npz", [("v2", 200, 0), ("v3", 201, 0)])
    rp = _pool(tmp_path / "rp_src.npz", [("v5", 200, 3), ("v5", 300, 0)])
    out = tmp_path / "out"
    args = type("A", (), {"exclusions": ex, "out": out, "new": ["nat_ink", "nat_grad", "rp"], "ref": [],
                          "tag": "t", "pools": [f"tt={tt}", f"nat_ink={ink}", f"nat_grad={grad}", f"rp={rp}"]})()
    POOLS.cmd_build(args)
    rec = {r["name"]: r for r in json.loads((out / "pools_record_t.json").read_text())["pools"]}
    assert rec["tt"]["out_rows"] == 2 and rec["tt"]["dropped_by_reason"] == {"test-set": 1}
    assert rec["nat_ink"]["out_rows"] == 2 and rec["nat_ink"]["duplicates_dropped"] == 1    # screenshot 102 in tt
    assert rec["nat_grad"]["out_rows"] == 2                     # same group as nat_ink: never deduplicated against it
    assert rec["rp"]["out_rows"] == 1 and rec["rp"]["duplicates_dropped"] == 1              # screenshot 200 in nat
    z, src = np.load(out / "tt.npz"), np.load(tt)
    kept = [0, 2]
    for j, i in enumerate(kept):
        a = z["data"][z["offsets"][j]:z["offsets"][j + 1]]
        assert np.array_equal(a, src["data"][src["offsets"][i]:src["offsets"][i + 1]])
    args2 = type("A", (), {"exclusions": ex, "out": tmp_path / "out2", "new": ["rp"], "ref": [f"nat_ink={ink}"],
                           "tag": "r", "pools": [f"rp={rp}"]})()
    POOLS.cmd_build(args2)
    r2 = json.loads((tmp_path / "out2" / "pools_record_r.json").read_text())["pools"][0]
    assert r2["out_rows"] == 1 and r2["duplicates_dropped"] == 1


def test_check_flags_hits(tmp_path):
    dev = _pool(tmp_path / "dev.npz", [("v7", 700, 0)])
    pool = _pool(tmp_path / "p.npz", [("v1", 1, 0), ("v7", 2, 0), ("v8", 700, 0), ("v9", 3, 0)])
    dev_vns, dev_imgs = POOLS.cohort_vns_and_images([dev])
    n, hits = POOLS.check_pool(pool, {"v9": "test-set"}, dev_vns, dev_imgs, {"vndb:1"}, set())
    assert n == 4 and hits == {"test-screenshot": 1, "dev-vn": 1, "dev-screenshot": 1, "excluded-vn": 1}


def _scores(path, shifts, n=240, vns=24, seed=0, tag=""):
    rng = np.random.default_rng(seed)
    lines = [{"id": f"{tag}l{i}", "vn": f"{tag}v{i % vns}", "ref": "あいうえお", "tags": []} for i in range(n)]
    base = rng.integers(0, 3, n)
    models = {}
    for run, (shift, sha) in shifts.items():
        e = np.clip(base + shift * (rng.random(n) < 0.6), 0, None).tolist()
        models[run] = {"blob": f"{run}.vocr", "blob_sha256": sha, "blob_bytes": 1, "arena_bytes": 1,
                       "errors": {"rv": e, "rvn": e}, "pred": {}, "cer": {}}
    Path(path).write_text(json.dumps({"dev": {}, "lines": lines, "models": models}))
    return str(path)


def _budget(d, shas, slow=()):
    (d / "a9").mkdir(parents=True)
    sizes = {}
    for i, sha in enumerate(shas):
        sizes[f"m{i}"] = {"blob_bytes": 657664, "arena": {"t4h16": {"w8i": 124832}}}
        ms = 600.0 if sha in slow else 239.0
        (d / "a9" / f"m{i}_l22048.json").write_text(json.dumps(
            {"blob_sha256": sha, "tile": 4, "l2_kib": 2048, "ms_640px": {"333": [100.0, ms]}}))
    (d / "sizes.json").write_text(json.dumps(sizes))
    return str(d)


def _runs(size, shifts):
    out = {}
    for arm, s in shifts.items():
        for i, r in enumerate(P["arms"][arm]["runs"][size]):
            out[r] = (s[i], f"{r}-sha")
    return out


def test_decisions_and_selection(tmp_path):
    runs = _runs("FL10", {"C": (0, 0), "A": (-1, -1), "M": (-1, 0)})
    tv = _scores(tmp_path / "tv.json", runs)
    dn = _scores(tmp_path / "dn.json", runs, n=120, vns=12, seed=1, tag="d")
    cohorts = SEL.load_cohorts([tv], [dn])
    real = SEL.decide(P, "FL10", "real", cohorts, n_boot=2000)
    assert real["adopted"] and real["adopted_arm"] == "A" and len(real["pairs"]) == 2
    assert real["total"]["development_score"]["ci95"][1] < 0
    synth = SEL.decide(P, "FL10", "synth", cohorts, real=real, n_boot=2000)
    assert synth["x"] == "A" and not synth["adopted"] and synth["adopted_arm"] == "A"
    bd = _budget(tmp_path / "b", [s for _, s in runs.values()], slow={"v4a_FL10_seed2-sha"})
    sel = SEL.select(P, "FL10", cohorts, [bd], real, synth)
    assert sel["final_arm"] == "A" and set(sel["candidates"]) == {"v4a_FL10_seed1", "v4a_FL10_seed2"}
    assert sel["selected"] == "v4a_FL10_seed1"                       # seed 2 is over the 550 ms ceiling
    assert len(sel["runs"]) == 6 and sum(r["candidate"] for r in sel["runs"]) == 2
    with pytest.raises(SystemExit):
        SEL.decide(P, "FL10", "synth", cohorts, real=None)


def test_decision_needs_every_pair(tmp_path):
    runs = _runs("H15", {"C": (0, 0), "A": (-2, 1), "M": (0, 0)})
    tv = _scores(tmp_path / "tv.json", runs, seed=4)
    dn = _scores(tmp_path / "dn.json", runs, n=120, vns=12, seed=5, tag="d")
    real = SEL.decide(P, "H15", "real", SEL.load_cohorts([tv], [dn]), n_boot=2000)
    assert not real["rule"]["every_pair_below_zero"] and not real["adopted"] and real["adopted_arm"] == "C"
