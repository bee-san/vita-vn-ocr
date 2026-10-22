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


def test_runs_summary_finds_each_sweeps_done_line_in_a_shared_log(tmp_path):
    """Stage 1's T and M sweeps ran side by side and wrote one sweep.log: each plan's SWEEP_DONE line is the one of
    its own stage label, wherever the other sweep's lines fall."""
    import importlib.util
    import json
    import sys
    repro = SWEEP.parent
    for name, file in (("textspot3_plan", "textspot3_plan.py"), ("textspot3_runs_summary", "runs_summary.py")):
        if name not in sys.modules:
            spec = importlib.util.spec_from_file_location(name, repro / file)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
    RS = sys.modules["textspot3_runs_summary"]
    out = tmp_path / "runs"
    out.mkdir()
    plans = {}
    for tag in ("T", "M"):
        plans[tag] = tmp_path / f"stage1_{tag}_plan.jsonl"
        plans[tag].write_text(json.dumps({"run": f"base_x{tag}_seed1"}) + "\n")
    a, b = "a" * 40, "b" * 40
    sha = {k: __import__("hashlib").sha256(v.read_bytes()).hexdigest() for k, v in plans.items()}
    (out / "sweep.log").write_text(
        f"2026-10-21T11:02:54Z sweep start: stage 1 T, MAX_PAR 6, head {a}, plan /x/{plans['T'].name} ({sha['T']}), r\n"
        "2026-10-21T11:03:00Z base_xT_seed1: attempt 1 (timeout 43200s, slot 1): train/textspot3_train.py\n"
        f"2026-10-21T12:46:14Z sweep start: stage 1 M, MAX_PAR 7, head {b}, plan /y/{plans['M'].name} ({sha['M']}), r\n"
        "2026-10-21T14:38:02Z SWEEP_DONE stage 1 T: 9/9 done, 0 running, 0 crashes (2026-10-21T14:38:02Z)\n"
        "2026-10-21T21:00:00Z SWEEP_DONE stage 1 M: 18/18 done, 0 running, 0 crashes (2026-10-21T21:00:00Z)\n")
    t = RS.sweep_record(out, plans["T"], herd=tmp_path)
    m = RS.sweep_record(out, plans["M"], herd=tmp_path)
    assert (t["stage"], t["commit"], t["max_par"], t["started"], t["sweep_done"]) == \
        ("stage 1 T", a, 6, "2026-10-21T11:02:54Z", "2026-10-21T14:38:02Z")
    assert (m["stage"], m["commit"], m["max_par"], m["started"], m["sweep_done"]) == \
        ("stage 1 M", b, 7, "2026-10-21T12:46:14Z", "2026-10-21T21:00:00Z")
    assert t["plan_sha256_at_start"] == sha["T"] and m["plan_sha256_at_start"] == sha["M"]
    # a relaunch of the same stage: its own SWEEP_DONE (the first launch's stays its own)
    with open(out / "sweep.log", "a") as f:
        f.write(f"2026-10-21T22:00:00Z sweep start: stage 1 T, MAX_PAR 6, head {a}, plan /x/{plans['T'].name} "
                f"({sha['T']}), r\n2026-10-21T22:05:00Z SWEEP_DONE stage 1 T: 9/9 done, 0 running, 0 crashes (x)\n")
    t = RS.sweep_record(out, plans["T"], herd=tmp_path)
    assert [x["sweep_done"] for x in t["launches"]] == ["2026-10-21T14:38:02Z", "2026-10-21T22:05:00Z"]
    assert t["sweep_done"] == "2026-10-21T22:05:00Z" and t["started"] == "2026-10-21T11:02:54Z"
