"""VOCR5 (eval/vocr5_protocol.json): the protocol's structure, the tier shapes' exact sizes, the run script's names and
mixtures. No screenshots, checkpoints or network; the shape check builds random-init blobs (vita/model_tiers.py)."""
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
P = json.loads((ROOT / "eval" / "vocr5_protocol.json").read_text(encoding="utf-8"))
RUN = (ROOT / "eval" / "vocr5_run.sh").read_text(encoding="utf-8")
TIERS = {"Small": "S", "Normal": "N", "Large": "L"}
# the shapes of eval/vocr5_run.sh, as vita/model_tiers.py CANDIDATES entries
SHAPES = {"Small": (1.0, 96, 1, 2, 5, 4987, 32, 8, None),
          "Normal": (1.5, 96, 1, 2, 5, 5631, 32, 8, None),
          "Large": (2.0, 96, 1, 2, 5, 5631, 32, 8, (64, 128, 256, 480, 576))}


def test_tiers_limits_and_vocabs():
    for name, t in P["tiers"].items():
        e, lim = t["expected"], t["limits"]
        if "weights_lt" in lim:
            assert e["blob_bytes"] < lim["weights_lt"]
        else:
            assert e["blob_bytes"] <= lim["weights_le"]
        assert e["arena_w8i_bytes"] <= lim["arena_le"]
        v = ROOT / t["vocab"]["file"]
        assert hashlib.sha256(v.read_bytes()).hexdigest() == t["vocab"]["sha256"]
        lines = v.read_text(encoding="utf-8").splitlines()
        assert len(lines) + 1 == t["vocab"]["classes"]          # + the CTC blank
        assert lines[-1] == "〓"                                # the <unk> class of a reduced vocab
        assert t["incumbent"]["blob_bytes"] <= e["blob_bytes"]
    lim = {n: t["limits"] for n, t in P["tiers"].items()}
    assert lim["Small"]["weights_lt"] == 1_000_000
    assert lim["Normal"]["weights_le"] == 1_300_000           # conductor 10:07Z / 10:21Z: no test.3 before training
    assert lim["Large"]["weights_le"] == 2_500_000            # owner 18:09Z


def test_box_estimates_meet_the_tiers():
    tab = P["estimates"]["table"]
    for name, shape in (("Small", "s_w100v4987"), ("Normal", "n_w150v5631"), ("Large", "l_lw300v5631")):
        row, box = tab[shape], P["tiers"][name]["limits"]["box_s_le"]
        assert row["bytes"] == P["tiers"][name]["expected"]["blob_bytes"]
        assert row["macs_per_frame"] == P["tiers"][name]["expected"]["macs_per_frame"]
        for kind in ("main", "#85", "#85 no-PLD"):
            for mhz in ("333", "444"):
                ms, s = row[kind][mhz]
                assert abs(240 * ms / 1000 + 0.1 - s) < 0.002
                assert s <= box
    assert tab["n_w150v5631"]["#85 no-PLD"]["333"][1] <= 1.0   # Normal's about 1 s box at 333 MHz


def test_exported_shapes_match_the_protocol():
    pytest.importorskip("torch")
    sys.path[:0] = [str(ROOT / "vita"), str(ROOT / "train"), str(ROOT / "eval")]
    import budget_sweep as BS
    import model_tiers as MT
    for name, cfg in SHAPES.items():
        data, _ = MT.export_blob(cfg)
        info = BS.blob_info(data)
        e = P["tiers"][name]["expected"]
        assert len(data) == e["blob_bytes"], name
        assert info["arena"]["t4h16"]["w8i"] == e["arena_w8i_bytes"], name
        assert info["macs_per_frame"] == e["macs_per_frame"], name


def test_run_script_names_shapes_and_mixtures():
    assert subprocess.run(["bash", "-n", str(ROOT / "eval" / "vocr5_run.sh")]).returncode == 0
    assert "PRUN=pre5_${TIER}_seed${SEED}" in RUN
    assert "RUN=v5$(echo \"$STAGE\" | tr 'PF' 'pf')_${TIER}_seed${SEED}" in RUN
    assert "--chans 64,128,256,480,576" in RUN
    runs = P["runs"]
    for name, t in TIERS.items():
        seeds = P["tiers"][name]["seeds"]
        assert runs["pretrains"][name] == [f"pre5_{t}_seed{s}" for s in seeds]
        for arm in ("P", "F"):
            assert runs["fine_tunes"][name][arm] == [f"v5{arm.lower()}_{t}_seed{s}" for s in seeds]

    def weights(block):
        return [float(w) for w in re.findall(r'--src "[a-z0-9]+:([0-9.]+):', block)]
    pre = RUN.split('if [ "$RECIPE" = C ]; then', 1)[1].split("PRUN=", 1)[0]
    c_pre, m_pre = pre.split("else", 1)
    assert abs(sum(weights(c_pre)) - 1) < 1e-9 and abs(sum(weights(m_pre)) - 1) < 1e-9
    ft = RUN.split("# ---------------------------------------------------------------- fine-tunes", 1)[1]
    syn = ft.split('if [ "$RECIPE" = C ]; then', 1)[1].split('RUN=v5', 1)[0]
    c_ft, m_ft = syn.split("else", 1)
    assert abs(sum(weights(c_ft)) - 0.5) < 1e-9 and abs(sum(weights(m_ft)) - 0.5) < 1e-9


def test_decisions_pairs_and_cohorts():
    arms = set(P["locked_report"]["arms"])
    for a, b in P["locked_report"]["pairs"]:
        assert a in arms and b in arms and a != b
    for cand, inc in (("VOCR5_S", "VOCR4_FL10"), ("VOCR5_N", "VOCR4_H15"), ("VOCR5_L", "VOCR4_W20")):
        assert [cand, inc] in P["locked_report"]["pairs"]
    inc = {t["incumbent"]["sha256"][:8] for t in P["tiers"].values()}
    assert inc <= {v[:8] for v in P["locked_report"]["arms"].values()}
    c = P["cohorts"]
    assert len(c["tv"]["sha256"]) == 64 and len(c["dn"]["files"]) == 4 and len(c["wd"]["files"]) == 2
    assert "0.5 x (tv + dn) / 2 + 0.5 x wd" in P["score"]["dev_score"]
    assert P["amendments"] == [] or all("date" in a and "change" in a for a in P["amendments"])


# ---------------------------------------------------------------- eval/vocr5_pools.py on toy pools

def _npz(path, rows):
    """A train2-format shard: rows of (vn, line id, label)."""
    import numpy as np
    data = np.zeros(len(rows) * 4, np.uint8)
    np.savez(path, data=data, offsets=np.arange(0, len(rows) * 4 + 1, 4), shapes=np.array([[2, 2]] * len(rows)),
             labels=np.array(json.dumps([r[2] for r in rows], ensure_ascii=False)),
             meta=np.array(json.dumps([{"vn": r[0], "id": r[1]} for r in rows])))
    return path


def test_pools_exclude_reserved_and_deduplicate_by_crop_style(tmp_path):
    sys.path.insert(0, str(ROOT / "eval"))
    import vocr5_pools as VP
    ex, _ = VP.reserved()
    res = json.loads((ROOT / "eval" / "manifests" / "reserved.json").read_text(encoding="utf-8"))
    assert set(ex) == set(res["test"]) | set(res["dev_sample"])
    test_vn, dev_vn = sorted(res["test"])[0], sorted(res["dev_sample"])[0]
    exf = tmp_path / "ex.json"
    exf.write_text(json.dumps({"vns": ex}))
    tt = _npz(tmp_path / "tt.npz", [("v9001", "v9001_100_0", "あい"), ("v9001", "v9001_100_1", "うえ")])
    rd = _npz(tmp_path / "rd.npz", [("v9001", "v9001_100_0", "あい")])            # control pools: never deduplicated
    ink = _npz(tmp_path / "ink.npz", [("v9001", "v9001_100_7", "あい"), ("v9002", "v9002_5_0", "か"),
                                      (test_vn, f"{test_vn}_1_0", "き"), (dev_vn, f"{dev_vn}_1_0", "く")])
    grad = _npz(tmp_path / "grad.npz", [("v9001", "v9001_100_3", "あい"), ("v9001", "v9001_100_4", "あい")])
    out = tmp_path / "out"
    argv = ["vocr5_pools.py", "build", "--exclusions", str(exf), "--out", str(out), "--tag", "t",
            f"tt={tt}", f"rd={rd}", f"rp3_ink={ink}", f"rp3_grad={grad}"]
    old = sys.argv
    try:
        sys.argv = argv
        VP.main()
    finally:
        sys.argv = old
    rec = {p["name"]: p for p in json.loads((out / "pools_record_t.json").read_text())["pools"]}
    assert rec["rd"]["out_rows"] == 1                       # rd keeps tt's line: the control pools stay whole
    assert rec["rp3_ink"]["out_rows"] == 1                  # same screenshot + label + ink style dropped, reserved out
    assert rec["rp3_ink"]["duplicates_dropped"] == 1
    assert {k.split(":")[0] for k in rec["rp3_ink"]["dropped_by_reason"]} == {"test", "dev_sample"}
    assert rec["rp3_grad"]["out_rows"] == 2                 # a GRAD crop stays when only its ink crop is elsewhere
    assert VP.group_of("tvt") == "tt" and VP.group_of("own_m8") == "own" and VP.style_of("own_m4") == "ink"
    assert VP.style_of("rp3_grad") == "grad" and VP.style_of("rp3_ref") == "ref"


def test_fine_tune_sources_split_by_rows_and_sum_to_the_real_half():
    sys.path.insert(0, str(ROOT / "eval"))
    import vocr5_pools as VP
    names = [n for _, _, ns, _ in VP.SOURCES for n in ns]
    record = {"pools": [{"name": n, "out": f"/x/{n}.npz", "out_rows": 10 + i} for i, n in enumerate(names)]}
    real = VP.ft_sources(record, {"rp3_grad": "/x/rp3_ref.npz"})
    assert abs(sum(e["weight"] for e in real) - 0.5) < 1e-5
    by = {e["name"]: e for e in real}
    assert abs(by["rpi"]["weight"] + by["rpg"]["weight"] - 0.175) < 1e-6
    assert by["rpi"]["fa"] == 0.3 and by["rpg"]["fa"] == 0 and by["rpg"]["refined"] == ["/x/rp3_ref.npz"]
    script = RUN
    assert 'fa={s[\'fa\']}' in script and '"F" else [])' in script


# ---------------------------------------------------------------- eval/vocr5_select.py on toy components

def _comps(errs):
    """Toy components: errs {model: (tv, dn, wd) per-row errors}; 4 VNs, 10 rows per component."""
    import numpy as np
    sys.path.insert(0, str(ROOT / "eval"))
    import vocr5_select as VS
    comps = {n: VS.Component(n) for n in ("tv", "dn", "wd")}
    vns = [f"v{i % 4}" for i in range(10)]
    for model, (a, b, c) in errs.items():
        for comp, e in zip(comps.values(), (a, b, c)):
            comp.add(model, np.asarray(e, float), np.full(10, 20.0), vns, np.zeros(10, bool), {})
    return VS, comps


def test_decisions_follow_the_rule():
    proto = {"tiers": {"Small": {"incumbent": {"name": "INC"}}},
             "runs": {"fine_tunes": {"Small": {"P": ["v5p_S_seed1", "v5p_S_seed2"],
                                               "F": ["v5f_S_seed1", "v5f_S_seed2"]}}}}
    base = [2] * 10
    better = [0] * 10
    VS, comps = _comps({"v5p_S_seed1": (base, base, base), "v5p_S_seed2": (base, base, base),
                        "v5f_S_seed1": (better, better, better), "v5f_S_seed2": (better, base, better),
                        "INC": (base, base, [3] * 10)})
    arm = VS.decide(proto, "Small", "arm", comps, n_boot=500)
    assert arm["adopted"] and arm["adopted_arm"] == "F" and arm["rule"]["every_pair_below_zero"]
    rep = VS.decide(proto, "Small", "replace", comps, arm, n_boot=500)
    assert rep["replaces"] and rep["runs_x"] == ["INC", "INC"]
    # a candidate worse on one component by a resolved margin is not adopted
    VS, comps = _comps({"v5p_S_seed1": (base, base, base), "v5p_S_seed2": (base, base, base),
                        "v5f_S_seed1": ([5] * 10, better, better), "v5f_S_seed2": ([5] * 10, better, better),
                        "INC": (base, base, base)})
    arm = VS.decide(proto, "Small", "arm", comps, n_boot=500)
    assert not arm["rule"]["no_component_entirely_above_zero"] and arm["adopted_arm"] == "P"
    assert abs(VS.score_of(comps, "v5p_S_seed1") - 10.0) < 1e-9      # 2 errors / 20 chars everywhere = 10 points


# ---------------------------------------------------------------- eval/vocr5_dialogue.py on a toy report

def test_dialogue_tables_split_names_and_pair_arms():
    sys.path.insert(0, str(ROOT / "eval"))
    import vocr5_dialogue as VD

    def recs(errs, names=(0,)):
        return [{"id": f"l{i}", "vn": f"v{i % 6}", "styles": ["name"] if i in names else [], "n": 10, "errors": e}
                for i, e in enumerate(errs)]
    a = {"vita-test": recs([5, 0, 1, 0, 0, 0, 0, 0, 0, 0]), "psp-test": recs([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])}
    b = {"vita-test": recs([5, 2, 2, 1, 0, 0, 0, 0, 0, 0]), "psp-test": recs([0, 2, 1, 1, 0, 0, 0, 0, 0, 0])}
    report = {"shared": {"spec": {"bootstrap": {"reps": 500, "seed": 0, "min_clusters": 5}},
                         "fixed_line": {"console": {"A": {"lines": a}, "B": {"lines": b}}}}}
    doc = VD.tables(report, {}, [("A", "B")])
    d = doc["arms"]["console"]["A"]["dialogue"]
    assert d["vita-test"] == {"lines": 9, "chars": 90, "errors": 1, "exact": 8, "cer": round(100 / 90, 3)}
    assert doc["arms"]["console"]["A"]["names"]["pooled"]["errors"] == 5          # the name lines only
    p = doc["pairs"]["A-B"]["console"]["dialogue"]["pooled"]
    assert p["candidate_errors"] == 2 and p["baseline_errors"] == 9 and p["units"] == 18
    assert p["delta"] == round(100 * (2 - 9) / 180, 6)


def test_fine_tune_pin_is_complete():
    pin = (ROOT / "eval" / "vocr5_pools.sha256").read_text(encoding="utf-8").splitlines()
    pinned = {ln.split()[1]: ln.split()[0] for ln in pin if ln.strip()}
    assert all(len(s) == 64 and int(s, 16) >= 0 for s in pinned.values())
    src = json.loads((ROOT / "eval" / "vocr5_ft_sources.json").read_text(encoding="utf-8"))
    assert abs(sum(e["weight"] for e in src["real"]) - 0.5) < 1e-5
    files = [f for e in src["real"] for f in e["files"] + e.get("refined", [])]
    assert set(files) == set(pinned)
    for e in src["real"]:
        assert (e["fa"] == 0.3) == (not e["name"].endswith("g"))     # fa on the ink sources only
        assert bool(e.get("refined")) == e["name"].endswith("g")      # refined copies with the GRAD sources
    rec = json.loads((ROOT / "eval" / "vocr5_pools_20261016.json").read_text(encoding="utf-8"))
    assert rec["checks"]["reserve_check"]["verdict"] == "RESERVE_OK"
    assert "CHECK_OK" in rec["checks"]["vocr5_pools_check"]
    assert rec["realplus_snapshot"]["refined"]["rows_only"] is True


def test_make_spec_arms_are_the_protocols(tmp_path, monkeypatch):
    import hashlib as H
    sys.path.insert(0, str(ROOT / "eval"))
    import vocr5_make_spec as MS
    v4 = json.loads((ROOT / "eval" / "vocr4_report_spec_20261014.json").read_text(encoding="utf-8"))
    old = tmp_path / "old"
    old.mkdir()
    fake = {}
    for a in v4["arms"]:                         # toy files standing in for the committed spec's blobs
        data = f"toy {a['file']}".encode()
        (old / a["file"]).write_bytes(data)
        fake[a["name"]] = dict(a, sha256=H.sha256(data).hexdigest())
    v4 = dict(v4, arms=[fake[a["name"]] for a in v4["arms"]])
    monkeypatch.setenv("HERD", str(tmp_path))
    sels = {}
    for tier, t in MS.TIERS.items():
        blob = tmp_path / f"{t}.vocr"
        blob.write_bytes(f"toy {tier}".encode())
        path = "$HERD/" + blob.name if tier == "Normal" else str(blob)      # a kept incumbent is a $HERD path
        sels[tier] = {"kind": "vocr5-selection", "tier": tier, "selected": f"run_{t}", "selected_blob": path,
                      "selected_blob_sha256": H.sha256(blob.read_bytes()).hexdigest(),
                      "replaces": tier != "Normal", "adopted_arm": "P"}
    spec, runs = MS.build(P, sels, v4, [old], tmp_path / "blobs")
    assert {a["name"] for a in spec["arms"]} == set(P["locked_report"]["arms"])
    assert [(p["candidate"], p["baseline"]) for p in spec["pairs"]] == [tuple(p) for p in P["locked_report"]["pairs"]]
    assert runs == {"VOCR5_S": "run_S", "VOCR5_N": "run_N", "VOCR5_L": "run_L"}
    assert (tmp_path / "blobs" / "VOCR5_N_w8.vocr").read_bytes() == b"toy Normal"
