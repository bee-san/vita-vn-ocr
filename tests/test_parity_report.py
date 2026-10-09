"""ci/parity_report.py: the evidence CI needs that the runtime's parity test ran on its fixture."""
import json
import subprocess
import sys

import numpy as np
import pytest

import parity_report as P
from conftest import ROOT

CKPT, SHARD, N = "/tmp/run/fixture.pt", "/tmp/run/sandbox/repo/data/synth/synth_00000.npz", 48
GOOD = {"ckpt": CKPT, "shard": SHARD, "lines": N, "pass": True, "f32": {"max_abs_logit_diff": 1e-5}}


def test_a_complete_passing_report_is_accepted():
    assert P.problems(GOOD, CKPT, SHARD, N) == []


@pytest.mark.parametrize("change, needle", [
    ({"ckpt": "/elsewhere/v1.pt"}, '"ckpt"'),
    ({"shard": None}, '"shard"'),
    ({"lines": 0}, '"lines"'),
    ({"lines": N - 1}, '"lines"'),          # some crops of the shard were not compared
    ({"lines": True}, '"lines"'),
    ({"lines": "48"}, '"lines"'),
    ({"pass": False}, '"pass"'),
    ({"pass": "true"}, '"pass"'),
])
def test_each_required_key_is_enforced(change, needle):
    bad = P.problems({**GOOD, **change}, CKPT, SHARD, N)
    assert len(bad) == 1 and needle in bad[0], bad


def test_not_an_object():
    assert P.problems([GOOD], CKPT, SHARD, N) == ["the report is not a JSON object"]


def test_cli_exit_status(tmp_path):
    shard = tmp_path / "synth_00000.npz"            # just the labels, as render.py stores them
    np.savez(shard, labels=np.array(json.dumps(["あ"] * N, ensure_ascii=False)))
    assert P.shard_lines(shard) == N
    run = [sys.executable, str(ROOT / "ci" / "parity_report.py")]
    args = ["--ckpt", CKPT, "--shard", str(shard)]
    report = tmp_path / "parity.json"
    report.write_text(json.dumps({**GOOD, "shard": str(shard)}))
    ok = subprocess.run([*run, str(report), *args], capture_output=True, text=True)
    assert ok.returncode == 0 and "PARITY_REPORT_OK" in ok.stdout, ok.stderr
    report.write_text(json.dumps({**GOOD, "shard": str(shard), "pass": False}))
    bad = subprocess.run([*run, str(report), *args], capture_output=True, text=True)
    assert bad.returncode != 0 and '"pass"' in bad.stderr
    report.write_text(json.dumps({**GOOD, "shard": str(shard), "lines": 1}))
    few = subprocess.run([*run, str(report), *args], capture_output=True, text=True)
    assert few.returncode != 0 and '"lines"' in few.stderr
    missing = subprocess.run([*run, str(tmp_path / "none.json"), *args], capture_output=True, text=True)
    assert missing.returncode != 0 and "no readable parity report" in missing.stderr
