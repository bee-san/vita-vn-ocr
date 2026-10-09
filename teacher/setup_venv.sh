#!/usr/bin/env bash
# Build the teacher labeller's env in teacher/.venv: owocr + MeikiOCR + the voters, CPU only.
# Idempotent: re-running re-resolves teacher/requirements.lock and re-syncs the venv to it.
#   teacher/setup_venv.sh            re-resolve the lock (uv keeps its pins unless requirements.in needs others) and sync
#   teacher/setup_venv.sh --frozen   sync to the committed lock without re-resolving (reproduction)
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$HERE")"
cd "$ROOT"

INDEX=(--index-url https://download.pytorch.org/whl/cpu
       --extra-index-url https://pypi.org/simple
       --index-strategy unsafe-best-match)
VENV=teacher/.venv
PY="$VENV/bin/python"

[ -x "$PY" ] || uv venv --python 3.11 "$VENV"
if [ "${1:-}" != "--frozen" ]; then
  # the repo lock is a constraint, so packages shared with the training env keep its versions
  uv pip compile teacher/requirements.in -o teacher/requirements.lock --python-version 3.11 \
    --emit-index-url -c requirements.lock "${INDEX[@]}"
fi
uv pip sync --python "$PY" teacher/requirements.lock "${INDEX[@]}"
uv pip install --python "$PY" --no-deps --require-hashes -r teacher/owocr.txt

"$PY" - <<'EOF'
import importlib.metadata as md
import onnxruntime, owocr.ocr  # noqa: F401  (owocr.ocr must import without its GUI deps)
for p in ("owocr", "meikiocr", "rapidocr", "onnxruntime", "manga-ocr", "torch", "transformers", "numpy"):
    print(p, md.version(p))
EOF
echo TEACHER_SETUP_DONE
