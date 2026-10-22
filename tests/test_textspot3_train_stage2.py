"""Tests of stage 2's plan (eval/reproducibility/textspot3-20261021/stage2_plan.jsonl, issue #108): the committed plan
is textspot3_plan.py stage2 --choices stage2_choices.json (the conductor's: no extra kept, the kd teacher
base_qL5_seed2, the students t64n8c16 and qS4) with the estimates of eval/textspot3_train_probe_stage2_20261021.json:
kd on each student, seeds 1-3, from the teacher's final.pt whose sha256 the plan records; its command lines are the
trainer's and check the teacher's sha256. The teacher's final.pt is not in CI, so its sha256 is taken from the plan
(the planner checked it against the file when the plan was made). No data; no network.
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
PLAN = REPRO / "stage2_plan.jsonl"
CHOICES = REPRO / "stage2_choices.json"
RECORD = ROOT / "eval" / "textspot3_train_probe_stage2_20261021.json"


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
pytestmark = pytest.mark.skipif(not PLAN.exists(), reason="the stage-2 plan is not committed yet")


def test_the_committed_stage2_plan_is_the_planners_output_for_the_committed_choices(monkeypatch):
    got = PL.load(PLAN)
    choices = PL.read_choices(CHOICES)
    shas = {r["teacher_pt_sha256"] for r in got}
    assert len(shas) == 1 and len(next(iter(shas))) == 64
    monkeypatch.setattr(PL, "teacher_record", lambda name, runs_dir: next(iter(shas)))
    reg = PL.shapes()
    want = PL.order(PL.stage2_from_choices(choices, reg, "/nonexistent"), reg, PL.read_est(RECORD))
    prot = {r.pop("protocol_sha256") for r in got}
    assert len(prot) == 1 and len(next(iter(prot))) == 64
    assert got == [json.loads(json.dumps(r, sort_keys=True)) for r in want]
    assert sorted(r["run"] for r in got) == sorted(f"kd_{s}_seed{n}" for s in choices["kd_students"] for n in (1, 2, 3))
    for r in got:
        assert (r["recipe"], r["steps"], r["teacher_run"]) == ("kd", 30000, choices["kd_teacher_run"])
        assert r["extras"] == "none" and not (choices["size_kept"] or choices["clean_kept"])
        assert r["teacher"] == f"$HERD/shared/artifacts/textspot3/runs/{choices['kd_teacher_run']}"
        assert r["family"] == reg[r["shape"]]["family"]
        assert r["timeout_s"] == max(PL.FLOOR[30000], 3600 * math.ceil(4 * r["est_s"] / 3600))
    assert [r["est_s"] for r in got] == sorted((r["est_s"] for r in got), reverse=True)
    stage1 = {r["run"] for fam in ("T", "M") for r in PL.stage1(fam, reg)}
    assert not {r["run"] for r in got} & stage1


def test_the_estimates_and_the_command_lines(tmp_path):
    doc = json.loads(RECORD.read_text())
    got = PL.load(PLAN)
    e = doc["estimate"]
    assert e["plan"] == "stage2_plan.jsonl" and set(e["est_run_s"]) == {r["run"] for r in got}
    assert all(r["est_s"] == e["est_run_s"][r["run"]] for r in got) and "7" in e["schedule"]
    for s in doc["smoke_runs"].values():
        assert s["check_ok"] and s["c_runtime_equal"] and s["teacher"]["name"] == got[0]["teacher_run"]
    # the command lines: the teacher expanded under $HERD and checked against the plan's sha256
    herd, repo = tmp_path / "herd", tmp_path / "repo"
    t2 = herd / "shared" / "artifacts" / "textspot2" / "shards"
    for d in (t2 / "lab" / "lab_realplus", t2 / "more" / "more_realplus"):
        d.mkdir(parents=True)
    (repo / "data" / "manifests").mkdir(parents=True)
    (repo / PL.T2_MANIFEST).write_text(json.dumps({"kind": "textspot2-shards", "shards": [
        {"dir": d, "name": f"{d}_0000"} for d in ("lab_realplus", "more_realplus")]}))
    teacher = herd / "shared" / "artifacts" / "textspot3" / "runs" / got[0]["teacher_run"]
    teacher.mkdir(parents=True)
    (teacher / "final.pt").write_bytes(b"not the teacher")
    with pytest.raises(SystemExit, match="not the plan's teacher"):
        PL.argv(got[0], "/o", repo, herd)
    import textspot3_train as T3
    rec = {**got[0], "teacher_pt_sha256": PL.file_sha256(teacher / "final.pt")}
    a = PL.argv(rec, "/o", repo, herd)
    args = T3.parse(a[1:])
    assert (args.recipe, args.extras_set, args.teacher, args.name) == ("kd", (), str(teacher), rec["run"])
