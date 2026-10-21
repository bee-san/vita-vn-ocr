"""Tests of stage 1's family-M plan (eval/reproducibility/textspot3-20261021/stage1_M_plan.jsonl, issue #108): the
committed plan is textspot3_plan.py stage1 --family M on the registered M shapes (amendment 4: S qS4, M qM4, L qL5)
with the estimates of eval/textspot3_train_probe_stage1m_20261021.json, and its command lines are the trainer's. A
file of its own, so tests/test_textspot3_train.py stays the bytes amendment 3 pins. No data; no network.
"""
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "train", ROOT / "synth"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
REPRO = ROOT / "eval" / "reproducibility" / "textspot3-20261021"
M_PLAN = REPRO / "stage1_M_plan.jsonl"
T_PLAN = REPRO / "stage1_T_plan.jsonl"
PROBE_RECORD = ROOT / "eval" / "textspot3_train_probe_stage1m_20261021.json"


def script(name, file):
    """One of the sweep's scripts as a module of its own name (not by sys.path: textspot2's share file names)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPRO / file)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PL = script("textspot3_plan", "textspot3_plan.py")


def test_the_registered_m_points():
    reg = PL.shapes()
    assert PL.m_points(reg) == {"S": "qS4", "M": "qM4", "L": "qL5"}
    import textspot3_model as M3
    import textspot3_train as T3
    for name, point in (("qS4", "S"), ("qM4", "M"), ("qL5", "L")):
        s, fam, src = T3.resolve_shape(name)
        assert fam == "M" and s["point"] == point and src.startswith("train/textspot3_shapes.SHAPES")
        assert M3.check_shape(s) is s and M3.layers(s)[-1]["head"] and M3.layers(s)[-1]["cout"] == 16
    assert [M3.vtsp_bytes(reg[n]) for n in ("qS4", "qM4", "qL5")] == [117184, 247392, 447392]


@pytest.mark.skipif(not M_PLAN.exists(), reason="the stage-1 M plan is not committed yet")
def test_the_committed_stage1_M_plan_is_reproducible():
    got = PL.load(M_PLAN)
    reg = PL.shapes()
    want = PL.order(PL.stage1("M", reg), reg, PL.read_est(PROBE_RECORD))
    shas = {r.pop("protocol_sha256") for r in got}
    assert len(shas) == 1 and len(next(iter(shas))) == 64
    assert got == [json.loads(json.dumps(r, sort_keys=True)) for r in want] and len(got) == 18
    by = {}
    for r in got:
        by.setdefault((r["recipe"], r["shape"]), []).append(r["seed"])
        assert r["family"] == "M" and r["extras"] is None and r["teacher"] is None
        assert r["steps"] == (60000 if r["recipe"] == "long" else 30000)
        assert r["timeout_s"] == max(PL.FLOOR[r["steps"]], 3600 * math.ceil(4 * r["est_s"] / 3600))
    assert {k: sorted(v) for k, v in by.items()} == {
        ("base", "qS4"): [1, 2, 3], ("base", "qM4"): [1, 2, 3], ("base", "qL5"): [1, 2, 3],
        ("size", "qM4"): [1, 2, 3], ("clean", "qM4"): [1, 2, 3], ("long", "qM4"): [1, 2, 3]}
    assert [r["est_s"] for r in got] == sorted((r["est_s"] for r in got), reverse=True)
    assert not {r["run"] for r in got} & {r["run"] for r in PL.load(T_PLAN)}       # the two plans share OUT
    doc = json.loads(PROBE_RECORD.read_text())
    assert doc["kind"] == "textspot3-train-probe" and doc["estimate"]["family"] == "M"
    assert set(doc["estimate"]["est_run_s"]) == {r["run"] for r in got} and "8" in doc["estimate"]["schedule"]
    for s in doc["smoke_runs"].values():
        assert s["check_ok"] and s["vtsp_version"] in (2, 3)
        assert s["c_runtime_equal"]                     # the version-3 C driver (and v2's) equal to the twin


@pytest.mark.skipif(not M_PLAN.exists(), reason="the stage-1 M plan is not committed yet")
def test_the_m_plan_command_lines(tmp_path):
    herd, repo = tmp_path / "herd", tmp_path / "repo"
    t2 = herd / "shared" / "artifacts" / "textspot2" / "shards"
    p2 = herd / "shared" / "private" / "textspot2" / "shards"
    for d in (t2 / "lab" / "lab_realplus", p2 / "lab" / "lab_owner", t2 / "more" / "more_realplus"):
        d.mkdir(parents=True)
    (repo / "data" / "manifests").mkdir(parents=True)
    (repo / PL.T2_MANIFEST).write_text(json.dumps({"kind": "textspot2-shards", "shards": [
        {"dir": d, "name": f"{d}_0000"} for d in ("lab_realplus", "lab_owner", "more_realplus")]}))
    import textspot3_train as T3
    for r in PL.load(M_PLAN):
        a = PL.argv(r, "/o", repo, herd)
        assert a[0] == "train/textspot3_train.py" and a[a.index("--family") + 1] == "M"
        args = T3.parse(a[1:])
        assert args.name == r["run"] and args.steps == r["steps"] and args.threads == 5
        assert args.extras_set == {"size": ("size",), "clean": ("clean",)}.get(r["recipe"], ())
        assert "--shape-json" not in a and "--any-shards" not in a and "--steps" not in a
