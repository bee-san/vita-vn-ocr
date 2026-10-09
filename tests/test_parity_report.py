"""ci/parity_report.py: the evidence CI needs that the runtime's parity test ran on its fixture."""
import json
import subprocess
import sys

import pytest

import parity_report as P
from conftest import ROOT

CKPT, SHARD = "/tmp/run/fixture.pt", "/tmp/run/sandbox/repo/data/synth/synth_00000.npz"
GOOD = {"ckpt": CKPT, "shard": SHARD, "lines": 48, "pass": True, "f32": {"max_abs_logit_diff": 1e-5}}


def test_a_complete_passing_report_is_accepted():
    assert P.problems(GOOD, CKPT, SHARD) == []


@pytest.mark.parametrize("change, needle", [
    ({"ckpt": "/elsewhere/v1.pt"}, '"ckpt"'),
    ({"shard": None}, '"shard"'),
    ({"lines": 0}, '"lines"'),
    ({"lines": True}, '"lines"'),
    ({"lines": "48"}, '"lines"'),
    ({"pass": False}, '"pass"'),
    ({"pass": "true"}, '"pass"'),
])
def test_each_required_key_is_enforced(change, needle):
    bad = P.problems({**GOOD, **change}, CKPT, SHARD)
    assert len(bad) == 1 and needle in bad[0], bad


def test_not_an_object():
    assert P.problems([GOOD], CKPT, SHARD) == ["the report is not a JSON object"]


def test_cli_exit_status(tmp_path):
    run = [sys.executable, str(ROOT / "ci" / "parity_report.py")]
    report = tmp_path / "parity.json"
    report.write_text(json.dumps(GOOD))
    ok = subprocess.run([*run, str(report), "--ckpt", CKPT, "--shard", SHARD], capture_output=True, text=True)
    assert ok.returncode == 0 and "PARITY_REPORT_OK" in ok.stdout
    report.write_text(json.dumps({**GOOD, "pass": False}))
    bad = subprocess.run([*run, str(report), "--ckpt", CKPT, "--shard", SHARD], capture_output=True, text=True)
    assert bad.returncode != 0 and '"pass"' in bad.stderr
    missing = subprocess.run([*run, str(tmp_path / "none.json"), "--ckpt", CKPT, "--shard", SHARD],
                             capture_output=True, text=True)
    assert missing.returncode != 0 and "no readable parity report" in missing.stderr
