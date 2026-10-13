#!/usr/bin/env python3
"""Which CI jobs does this change need? (.github/workflows/ci.yml, job "ruff")

Reads the paths changed between BASE and HEAD and decides, per job, whether anything it
checks could have changed. Written as GitHub Actions outputs (GITHUB_OUTPUT) and printed.

    python ci/changed.py [--base SHA] [--head HEAD] [--event pull_request|push|workflow_dispatch]

Without a base (workflow_dispatch, a force-push, a failed fetch) every job runs. ruff and the
repo policy test run on every event regardless; they are not gated here.

    pytest   anything but documentation: *.md, docs/, notes/, LICENSE, .gitignore
    smoke    train/, synth/, data/, ci/, requirements-ci.*, .github/   (ci/train_smoke.py)
    runtime  the smoke set plus runtime/                                (ci/runtime.sh)
    cc       ["gcc","clang"] when runtime/ C code changed or this is not a pull request,
             else ["gcc"]: only the C sources can break one compiler and not the other
    vita     vita/, runtime/, .github/                                  (vita/scripts/build.sh)

A path in none of these sets (a new top-level directory, say) counts as code and runs pytest
only; add it to a set here when a job starts depending on it.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DOC_DIRS = ("docs/", "notes/")
DOC_FILES = {"LICENSE", ".gitignore"}
SMOKE_DIRS = ("train/", "synth/", "data/", "ci/", ".github/")
SMOKE_FILES = {"requirements-ci.in", "requirements-ci.lock"}
RUNTIME_DIRS = SMOKE_DIRS + ("runtime/",)
VITA_DIRS = ("vita/", "runtime/", ".github/")


def is_doc(path):
    return path.endswith(".md") or path.startswith(DOC_DIRS) or path in DOC_FILES


def classify(paths, event="pull_request", runtime_exists=True):
    """Job gates for a list of changed paths; paths=None means unknown (run everything)."""
    if paths is None:
        code = smoke = runtime = c_code = vita = True
    else:
        code_paths = [p for p in paths if not is_doc(p)]
        code = bool(code_paths)
        smoke = any(p.startswith(SMOKE_DIRS) or p in SMOKE_FILES for p in code_paths)
        runtime = any(p.startswith(RUNTIME_DIRS) or p in SMOKE_FILES for p in code_paths)
        c_code = any(p.startswith("runtime/") for p in code_paths)
        vita = any(p.startswith(VITA_DIRS) for p in code_paths)
    full_cc = c_code or event != "pull_request"
    return {
        "pytest": code,
        "smoke": smoke,
        "runtime": runtime and runtime_exists,
        "cc": ["gcc", "clang"] if full_cc else ["gcc"],
        "vita": vita and runtime_exists,
    }


def changed_paths(base, head):
    """Paths that differ between base and head, or None when that cannot be determined."""
    if not base or set(base) == {"0"}:
        return None
    try:
        if subprocess.run(["git", "cat-file", "-e", f"{base}^{{commit}}"], cwd=ROOT,
                          capture_output=True).returncode != 0:
            subprocess.run(["git", "fetch", "--no-tags", "--depth=1", "origin", base],
                           cwd=ROOT, check=True, capture_output=True, text=True)
        out = subprocess.run(["git", "diff", "--name-only", base, head], cwd=ROOT,
                             check=True, capture_output=True, text=True).stdout
    except subprocess.CalledProcessError as e:
        print(f"::warning title=ci/changed.py::could not diff {base[:12]}..{head}: "
              f"{(e.stderr or '').strip()[:200]}; running every job", file=sys.stderr)
        return None
    return [line for line in out.splitlines() if line]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default=os.environ.get("CI_BASE_SHA", ""))
    ap.add_argument("--head", default="HEAD")
    ap.add_argument("--event", default=os.environ.get("GITHUB_EVENT_NAME", "workflow_dispatch"))
    args = ap.parse_args()

    paths = changed_paths(args.base, args.head)
    gates = classify(paths, event=args.event, runtime_exists=(ROOT / "runtime").is_dir())

    if paths is None:
        print(f"{args.event}: no base to diff against, so every job runs")
    else:
        print(f"{args.event}: {len(paths)} changed path(s) since {args.base[:12]}")
        for p in paths[:200]:
            print(f"  {'doc ' if is_doc(p) else 'code'} {p}")
        if len(paths) > 200:
            print(f"  ... and {len(paths) - 200} more")
    lines = []
    for k, v in gates.items():
        v = json.dumps(v) if isinstance(v, list) else str(v).lower()
        lines.append(f"{k}={v}")
    print("gates: " + " ".join(lines))
    out = os.environ.get("GITHUB_OUTPUT")
    if out:
        with open(out, "a") as f:
            f.write("\n".join(lines) + "\n")
    skipped = [k for k, v in gates.items() if v is False]
    if skipped:
        print(f"::notice title=CI jobs skipped::{', '.join(skipped)}: nothing they check changed "
              "(ci/changed.py)")


if __name__ == "__main__":
    main()
