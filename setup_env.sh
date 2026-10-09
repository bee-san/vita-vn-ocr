#!/usr/bin/env bash
# Build the Python env for the Vita VN OCR work (training, labelling, benchmarks).
# Idempotent: re-running re-resolves requirements.lock and re-syncs .venv to it.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
mkdir -p logs

INDEX=(--index-url https://download.pytorch.org/whl/cpu
       --extra-index-url https://pypi.org/simple
       --index-strategy unsafe-best-match)

[ -x .venv/bin/python ] || uv venv --python 3.11 .venv
uv pip compile requirements.in -o requirements.lock --python-version 3.11 \
  --emit-index-url "${INDEX[@]}"
uv pip sync --python .venv/bin/python requirements.lock "${INDEX[@]}"

.venv/bin/python - <<'EOF'
import cv2, numpy, onnxruntime, PIL, torch, transformers
print("torch", torch.__version__, "threads", torch.get_num_threads())
print("numpy", numpy.__version__, "PIL", PIL.__version__, "cv2", cv2.__version__)
print("onnxruntime", onnxruntime.__version__, "transformers", transformers.__version__)
EOF
echo SETUP_DONE
