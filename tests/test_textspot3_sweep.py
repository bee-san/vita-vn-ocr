"""Tests of the sweep's 5-run notices (eval/reproducibility/textspot3-20261021/sweep.sh's finished(), issue #108): a
notice at every multiple of 5 finished runs of the plan, each once, also when two runs finish within one check (the
count goes 4 -> 6; amendment 3's sweep.sh tested n % 5 at that moment and skipped the 5). The function is taken from
the script's text and run in bash with the sweep's status() and notify() stubbed. No data; no network.
"""
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "eval" / "reproducibility" / "textspot3-20261021" / "sweep.sh"


def finished_function():
    text = SWEEP.read_text()
    m = re.search(r"^finished\(\) \{.*?^\}$", text, re.S | re.M)
    assert m, "sweep.sh has no finished() function"
    return m.group(0)


def run_sequence(tmp_path, done_steps):
    """done_steps: for each finished() call, the runs whose done.json exists at that moment. -> the notices sent."""
    out = tmp_path / "out"
    (out / "locks").mkdir(parents=True)
    (out / "finished.txt").write_text("r0 0123456789abcdef 27456 30000 1.0\n")
    runs = [f"r{i}" for i in range(12)]
    sent = tmp_path / "sent.txt"
    steps = []
    for done in done_steps:
        mk = " ".join(f"mkdir -p '{out}/{r}' && : > '{out}/{r}/done.json';" for r in done)
        steps.append(f"{mk} finished last_run")
    script = "\n".join([
        "set -euo pipefail",
        f"OUT='{out}'", f"ALL='{' '.join(runs)}'", f"NOTIFIED='{out}/locks/notified-plan'",
        "status() { echo \"stage: status\"; }",
        f"notify() {{ echo \"$*\" >> '{sent}'; }}",
        finished_function(),
        *steps])
    subprocess.run(["bash", "-c", script], check=True, timeout=60)
    return sent.read_text().splitlines() if sent.exists() else []


def test_a_notice_at_every_multiple_of_5_once(tmp_path):
    # one run at a time: 5 and 10, once each
    seq = [[f"r{i}" for i in range(n)] for n in range(1, 12)]
    sent = run_sequence(tmp_path / "a", seq)
    assert [re.search(r"\((\d+)-run notice\)", s).group(1) for s in sent] == ["5", "10"]


@pytest.mark.parametrize("counts, want", [
    ((4, 6, 6), ["5"]),                       # two runs within one check: 4 -> 6 (amendment 3's sweep sent nothing)
    ((3, 11), ["5", "10"]),                   # a jump over two multiples sends both
    ((5, 5, 6, 10, 10), ["5", "10"]),         # a repeat at the same count sends nothing more
    ((1, 2, 3, 4), []),
])
def test_jumps_and_repeats(tmp_path, counts, want):
    seq = [[f"r{i}" for i in range(n)] for n in counts]
    sent = run_sequence(tmp_path, seq)
    assert [re.search(r"\((\d+)-run notice\)", s).group(1) for s in sent] == want
    assert all(s.startswith("stage: status (") and "last: last_run (0123456789ab..., 27456 B)" in s for s in sent)


def test_the_script_parses():
    p = subprocess.run(["bash", "-n", str(SWEEP)], capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
