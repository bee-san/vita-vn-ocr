"""Repository policy (AGENTS.md, Data policy) on every tracked file, and the CI pinning rules."""
import os
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BLOCKED = {
    "fonts": {".ttf", ".otf", ".ttc", ".woff", ".woff2"},
    "screenshots or crops": {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"},
    "shards, datasets or archives": {".npz", ".npy", ".parquet", ".arrow", ".zip", ".tar", ".gz", ".xz", ".zst",
                                     ".7z"},
    "weights or checkpoints": {".pt", ".pth", ".ckpt", ".onnx", ".safetensors", ".tflite", ".bin", ".pdparams",
                               ".vocr"},
    "console binaries": {".vpk", ".self", ".velf", ".elf", ".suprx", ".skprx", ".prx", ".pbp", ".iso", ".cso"},
}
MAX_BYTES = 2 * 1024 * 1024
# Repo-relative paths that may break a rule above. Add one only with the reason in the PR body.
ALLOWED = set()


def tracked():
    try:
        out = subprocess.run(["git", "ls-files", "-z", "-s"], cwd=ROOT, check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    entries = []
    for rec in out.decode().split("\0"):
        if rec:
            meta, path = rec.split("\t", 1)
            entries.append((meta.split()[0], path))       # (mode, path)
    return [(mode, path) for mode, path in entries if path not in ALLOWED]


def test_no_copyrighted_or_bulky_files_are_tracked():
    bad = []
    for _, path in tracked():
        kind = next((k for k, exts in BLOCKED.items() if Path(path).suffix.lower() in exts), None)
        if kind:
            bad.append(f"{path}: {kind} are never committed")
        elif (ROOT / path).is_file() and (ROOT / path).stat().st_size > MAX_BYTES:
            bad.append(f"{path}: {(ROOT / path).stat().st_size:,} bytes (limit {MAX_BYTES:,})")
    assert not bad, ("AGENTS.md: commit code, manifests (URLs + hashes), small tables and results; "
                     "weights go on releases.\n" + "\n".join(bad))


def test_no_tracked_symlinks_out_of_the_repo():
    """A committed symlink into $HERD/shared or a home directory works for nobody else."""
    bad = []
    for mode, path in tracked():
        if mode == "120000":
            target = os.readlink(ROOT / path)
            resolved = (ROOT / path).parent.joinpath(target).resolve()
            if os.path.isabs(target) or ROOT.resolve() not in resolved.parents:
                bad.append(f"{path} -> {target}")
    assert not bad, bad


def workflow_files():
    gh = ROOT / ".github"
    return sorted([*gh.glob("workflows/*.y*ml"), *gh.glob("actions/*/action.y*ml")])


USES = re.compile(r"^\s*(?:-\s+)?uses:\s*['\"]?([^'\"\s#]+)['\"]?\s*(#.*)?$")
PINNED = re.compile(r"^[\w.-]+/[\w.-]+(/[\w./-]+)?@[0-9a-f]{40}$")
TAG = re.compile(r"^#\s*v\d+(\.\d+)*\b")


def test_actions_are_pinned_to_a_commit_with_the_tag_in_a_comment():
    files = workflow_files()
    assert files, "no workflows found"
    bad = []
    for f in files:
        for n, line in enumerate(f.read_text().splitlines(), 1):
            m = USES.match(line)
            if not m or m.group(1).startswith("./"):
                continue
            if not PINNED.match(m.group(1)) or not TAG.match(m.group(2) or ""):
                bad.append(f"{f.relative_to(ROOT)}:{n}: {line.strip()}")
    assert not bad, ("pin each action to a full commit SHA with its tag as a comment, "
                     "e.g. 'uses: actions/checkout@<40-hex sha> # v7.0.1':\n" + "\n".join(bad))


def test_jobs_run_on_ubuntu_24_04():
    runs_on = [(f, n, line.split(":", 1)[1].strip().strip("'\""))
               for f in (ROOT / ".github" / "workflows").glob("*.y*ml")
               for n, line in enumerate(f.read_text().splitlines(), 1) if line.strip().startswith("runs-on:")]
    assert runs_on
    bad = [f"{f.name}:{n}: {v}" for f, n, v in runs_on if v != "ubuntu-24.04"]
    assert not bad, bad
