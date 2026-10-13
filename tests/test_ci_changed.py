"""ci/changed.py: the path sets that gate the CI jobs (.github/workflows/ci.yml)."""
import subprocess
from pathlib import Path

import pytest

import changed

ROOT = Path(__file__).resolve().parents[1]


def gates(paths, event="pull_request"):
    return changed.classify(paths, event=event)


def test_documentation_only_runs_nothing_but_lint_and_policy():
    g = gates(["docs/synth3.md", "notes/00-Start-Here.md", "README.md", "runtime/README.md", "LICENSE"])
    assert g == {"pytest": False, "smoke": False, "runtime": False, "cc": ["gcc"], "vita": False}


def test_tests_and_eval_run_pytest_only():
    g = gates(["tests/test_eval.py", "eval/vocr3_report.py", "docs/eval.md"])
    assert g["pytest"] and not g["smoke"] and not g["runtime"] and not g["vita"]


def test_training_code_runs_the_smoke_and_the_parity_check_with_gcc_on_a_pull_request():
    g = gates(["train/export.py"])
    assert g["pytest"] and g["smoke"] and g["runtime"] and not g["vita"]
    assert g["cc"] == ["gcc"]


@pytest.mark.parametrize("path", ["synth/render.py", "data/vocab.txt", "ci/fonts.py",
                                  "ci/fixtures/corpus/wiki.txt", "requirements-ci.lock",
                                  ".github/actions/python-env/action.yml"])
def test_everything_the_smoke_and_runtime_jobs_read_gates_them(path):
    g = gates([path])
    assert g["smoke"] and g["runtime"], path


def test_c_sources_run_both_compilers_and_the_vita_build():
    g = gates(["runtime/vocr.c"])
    assert g["runtime"] and g["vita"] and g["cc"] == ["gcc", "clang"]
    assert not g["smoke"]


def test_vita_app_runs_only_the_vita_build_besides_pytest():
    g = gates(["vita/app/main.c", "vita/CMakeLists.txt"])
    assert g["vita"] and g["pytest"] and not g["runtime"] and not g["smoke"]


def test_workflow_changes_run_every_gated_job():
    g = gates([".github/workflows/ci.yml"])
    assert g["smoke"] and g["runtime"] and g["vita"] and g["pytest"]


def test_pushes_to_main_use_both_compilers():
    assert gates(["train/model.py"], event="push")["cc"] == ["gcc", "clang"]
    assert gates(["docs/a.md"], event="push")["runtime"] is False


def test_unknown_changes_run_everything():
    g = gates(None, event="workflow_dispatch")
    assert g == {"pytest": True, "smoke": True, "runtime": True, "cc": ["gcc", "clang"], "vita": True}


def test_a_new_top_level_directory_counts_as_code():
    g = gates(["somewhere/new.py"])
    assert g["pytest"] and not g["smoke"] and not g["runtime"] and not g["vita"]


def test_without_a_runtime_directory_the_c_jobs_are_skipped():
    g = changed.classify(["runtime/vocr.c"], runtime_exists=False)
    assert not g["runtime"] and not g["vita"]


def test_the_script_runs_against_this_checkout():
    """End to end on the real repo: HEAD against HEAD changes nothing, so only the always-on jobs."""
    r = subprocess.run([str(ROOT / ".venv-ci/bin/python") if (ROOT / ".venv-ci/bin/python").exists() else "python3",
                        str(ROOT / "ci/changed.py"), "--base", "HEAD", "--head", "HEAD", "--event", "pull_request"],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0 and "not a git" in (r.stderr + r.stdout):
        pytest.skip("not a git checkout")
    assert r.returncode == 0, r.stderr
    assert "gates: pytest=false smoke=false runtime=false cc=[\"gcc\"] vita=false" in r.stdout
