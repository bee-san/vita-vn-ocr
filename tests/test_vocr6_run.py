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
