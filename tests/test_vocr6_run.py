"""eval/vocr6_run.py: an arm whose init names its stage's base (e.g. W15, picked by the previous stage's rule) takes
that base's init per seed once the protocol records it, and refuses to run before."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))

import vocr6_run as R  # noqa: E402


def proto(filled):
    return {"stages": {"2": {"tier": "N", "defaults": {"batch": 96},
                             "base": {"arm": "W15", "filled": filled},
                             "arms": {"S2x2": {"run": "v6s2x2_N", "mix": "m1_x2", "init": "W15", "steps": 15000}}}}}


def test_base_init_is_filled_per_seed():
    filled = {"arm": "A15", "init": "/x/v5f_N_seed{seed}/ckpt_step15000.pt", "init_sha256": {"1": "a", "3": "b"}}
    st, spec = R.arm_spec(proto(filled), "2", "S2x2")
    assert spec["init"].format(seed=3) == "/x/v5f_N_seed3/ckpt_step15000.pt"
    assert spec["init_sha256"] == {"1": "a", "3": "b"}
    assert spec["base"] == "A15" and spec["batch"] == 96 and spec["tier"] == "N"


def test_unfilled_base_refuses():
    with pytest.raises(SystemExit, match="not filled"):
        R.arm_spec(proto(None), "2", "S2x2")


def test_repo_protocol_stage2_arms_name_the_base():
    doc = json.loads((ROOT / "eval/vocr6_protocol.json").read_text(encoding="utf-8"))
    st = doc["stages"]["2"]
    assert st["base"]["arm"] == "W15"
    for arm in st["arms"].values():
        assert arm["init"] == "W15" and arm["mix"] in doc["mixes"]
        tot = sum(s["weight"] for s in doc["mixes"][arm["mix"]])
        assert abs(tot - 1.0) < 1e-6


def test_base_mix_is_filled():
    filled = {"arm": "S2x2", "mix": "m1_x2", "init": "/i{seed}", "init_sha256": {"1": "a"}}
    arms = {"S3rep": {"run": "v6s3rep_N", "mix": "R2", "init": "R2", "arch_extra": ["--rep", "2"]}}
    doc = {"stages": {"3": {"tier": "N", "base": {"arm": "R2", "filled": filled}, "arms": arms}}}
    st, spec = R.arm_spec(doc, "3", "S3rep")
    assert spec["mix"] == "m1_x2" and spec["init"] == "/i{seed}" and spec["base"] == "S2x2"


def test_repo_protocol_stage3_arms_name_the_base():
    doc = json.loads((ROOT / "eval/vocr6_protocol.json").read_text(encoding="utf-8"))
    st = doc["stages"]["3"]
    assert st["base"]["arm"] == "R2"
    for arm in st["arms"].values():
        assert arm["init"] == "R2" and arm["mix"] == "R2"


def test_extra_pool_files_are_checked_against_their_manifest(tmp_path):
    (tmp_path / "train").mkdir()
    f = tmp_path / "train" / "T1.npz"
    f.write_bytes(b"crops")
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps({"files": {"train/T1.npz": {"sha256": R.sha256(f), "crops": 1}}}))
    pins = {"extra_pools": [{"name": "cc", "manifest": str(man), "sha256": R.sha256(man), "sources": ["cc"]}]}
    sources = [("cc", 0.1, [str(f)], "")]
    rec = R.check_extra_pools(pins, sources, {})
    assert rec["cc_files"] == 1
    f.write_bytes(b"other")
    with pytest.raises(SystemExit, match="not in"):
        R.check_extra_pools(pins, sources, {})


def test_seed_only_arms_refuse_other_seeds(monkeypatch, tmp_path):
    proto = {"stages": {"4": {"seeds": [1, 2, 3], "arms": {"F4L_s2": {"run": "v6f4_L", "tier": "L", "seed_only": 2,
                                                                       "mix": "m", "steps": 10, "init": "/x"}}}}}
    pp = tmp_path / "p.json"
    pp.write_text(json.dumps(proto))
    monkeypatch.setattr(sys, "argv", ["vocr6_run.py", "command", "--stage", "4", "--arm", "F4L_s2", "--seed", "1",
                                      "--label", "x", "--protocol", str(pp)])
    with pytest.raises(SystemExit, match="seed 2 only"):
        R.main()


def test_two_pinned_versions_each_check_their_own_files(tmp_path):
    pins = {"extra_pools": []}
    files = {}
    for v in ("v1", "v2"):
        d = tmp_path / v / "train"
        d.mkdir(parents=True)
        f = d / "T1.npz"
        f.write_bytes(v.encode())
        man = tmp_path / v / "manifest.json"
        man.write_text(json.dumps({"files": {"train/T1.npz": {"sha256": R.sha256(f)}}}))
        pins["extra_pools"].append({"name": f"cc_{v}", "manifest": str(man), "sha256": R.sha256(man),
                                    "sources": ["cc"]})
        files[v] = str(f)
    rec = R.check_extra_pools(pins, [("cc", 0.1, [files["v2"]], "")], {})
    assert rec == {"cc_v2_files": 1, "cc_v2_manifest_sha256": pins["extra_pools"][1]["sha256"]}
    rec = R.check_extra_pools(pins, [("cc", 0.1, [files["v1"], files["v2"]], "")], {})
    assert rec["cc_v1_files"] == 1 and rec["cc_v2_files"] == 1
    stray = tmp_path / "other.npz"
    stray.write_bytes(b"x")
    with pytest.raises(SystemExit, match="no pinned manifest"):
        R.check_extra_pools(pins, [("cc", 0.1, [str(stray)], "")], {})


def _pin(tmp_path, v, names):
    d = tmp_path / v / "train"
    d.mkdir(parents=True)
    files = []
    for n in names:
        f = d / n
        f.write_bytes((v + n).encode())
        files.append(f)
    man = tmp_path / v / "manifest.json"
    man.write_text(json.dumps({"files": {f"train/{f.name}": {"sha256": R.sha256(f)} for f in files}}))
    pin = {"name": f"cc_{v}", "manifest": str(man), "sha256": R.sha256(man), "sources": ["cc"]}
    return pin, [str(f) for f in files]


def test_sources_no_pin_names_are_not_checked(tmp_path):
    """A mix source that no extra pool names (e.g. the hooked pools, checked elsewhere) is skipped, not refused."""
    pin, files = _pin(tmp_path, "v1", ["T1.npz"])
    rec = R.check_extra_pools({"extra_pools": [pin]}, [("hk", 0.15, [str(tmp_path / "elsewhere.npz")], ""),
                                                       ("cc", 0.06, files, "")], {})
    assert rec == {"cc_v1_files": 1, "cc_v1_manifest_sha256": pin["sha256"]}


def test_file_counts_sum_over_a_versions_files(tmp_path):
    pin, files = _pin(tmp_path, "v2", ["T1.npz", "T2.npz", "T3.npz"])
    rec = R.check_extra_pools({"extra_pools": [pin]}, [("cc", 0.06, files[:2], ""), ("cc", 0.04, files[2:], "")], {})
    assert rec["cc_v2_files"] == 3


def test_fold_in_mixes_are_marked_hval_in_training():
    assert R.hval_in_training({"mix": "r3all_fold"}) and not R.hval_in_training({"mix": "r3all"})
    assert not R.hval_in_training({"mix": "m1_cil"}) and not R.hval_in_training({})


def _finished_run(tmp_path, monkeypatch, mix):
    """A run whose training has finished, its blob exported and its score file written (subprocess stubbed)."""
    monkeypatch.setattr(R, "HERD", tmp_path)
    log = tmp_path / "explog.jsonl"
    monkeypatch.setattr(R, "EXPLOG", log)
    runs, scores, blobs = R.dirs("lab")
    rd = runs / "v6f4h_S_seed1"
    rd.mkdir(parents=True)
    (rd / "ckpt_step10.pt").write_bytes(b"ckpt")
    (rd / "vocr6_run.json").write_text(json.dumps({"config_sha256": "c", "inputs": {}, "segments": [{"seconds": 36}]}))
    blobs.mkdir(parents=True)
    (blobs / "v6f4h_S_seed1_w8.vocr").write_bytes(b"blob")
    scores.mkdir(parents=True)
    rubi = {"hval_clean_dlg": 0.4, "hval_page": 1.0, "hval_page_all": 1.5, "dev_score": 2.0}
    (scores / "score_v6f4h_S_seed1.json").write_text(json.dumps({"blob_sha256": "b", "blob_bytes": 4, "arena_bytes": 1,
                                                                   "summary": {"rubi": rubi}}))
    monkeypatch.setattr(R.subprocess, "run", lambda *a, **k: None)
    args = type("A", (), {"label": "lab", "stage": "4", "arm": "F4Sh_s1", "seed": 1, "protocol": "p.json"})()
    spec = {"run": "v6f4h_S", "tier": "S", "mix": mix, "steps": 10}
    R.finish(args, {}, {}, spec)
    return json.loads(log.read_text().splitlines()[-1])


def test_a_fold_in_runs_done_line_says_hval_in_training(tmp_path, monkeypatch):
    rec = _finished_run(tmp_path, monkeypatch, "r3all_fold")
    assert rec["hval_held_out"] is False and "hval in training" in rec["notes"]


def test_a_screening_runs_done_line_has_no_marker(tmp_path, monkeypatch):
    rec = _finished_run(tmp_path, monkeypatch, "r3all")
    assert "hval_held_out" not in rec and "hval in training" not in rec["notes"]


def test_s2fa2_seed3_is_refused():
    p = json.loads((ROOT / "eval/vocr6_protocol.json").read_text(encoding="utf-8"))
    assert p["stages"]["2"]["arms"]["S2fa2"].get("seed_only") == 1
