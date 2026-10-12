# Bounded FL10 baseline blends

`train/fl10_blend.py` produces a fixed follow-up hypothesis after the generic
development experiment rejected all ten checkpoints. It blends only the pinned
released-model initializer and scheduled step 2000; it cannot choose another
endpoint or expand its fraction grid. It loads weights and exports models,
without reading screenshots, scoring OCR, training, or promoting a runtime.

The seven fractions are `0, 1/16, 1/8, 1/4, 1/2, 3/4, 1`. Every named floating
model parameter uses three separate CPU float32 operations:
`delta=final-init`, `scaled=delta*float(alpha)`, `result=init+scaled`.
The endpoint tensors are cloned exactly. All 54 BatchNorm buffers must be
byte-identical between parents and are copied unchanged. Architecture, actual
tensor keys/types/shapes, finite values, ordered vocabulary and frozen-BN
metadata are validated before publication. No BatchNorm recalibration occurs.

Input checkpoint, export and original-selection SHA256s are fixed in the producer. The original
selection supplies the exact step-2000 checkpoint/export paths. Before creating
the destination, the producer verifies that alpha 0's exported payload equals
the released W8 payload and alpha 1's equals the previously scored step-2000
payload. Payload comparison excludes the 64-byte header, which contains the
source checkpoint SHA. Every export retains the 657,664-byte model and
124,832-byte W8I arena at maximum width 1024.

```sh
.venv-ci/bin/python train/fl10_blend.py \
  --initial /private/fl10_init.pt \
  --original-selection /private/completed-general-vn/selection.json \
  --baseline /private/FL10_w8.vocr \
  --out /private/new-general-vn-blends/derived
```

The destination must be new, even after an interrupted production. Seven
checkpoints have explicit fraction filenames; `inventory.json` publishes last
with all checkpoint/model-state/export hashes, parent/input hashes, producer
and dependency hashes, environment, arithmetic and endpoint checks. Checkpoints
retain actual endpoint training step 2000 plus derived provenance. They omit
optimizer, RNG, training-recipe and exposure state, and cannot resume the parent
training run. Keep these private weights and inventory ignored.

Freeze this complete inventory and new line-selector/textbox preparations before
new DEV forwards. Use the existing per-VN all/body full/hiragana and synthetic
guards and exact ranking; evaluate only the selected candidate through the
textbox guard. The completed failed experiment stays immutable. If every blend
fails, retain the released baseline. This is a DEV-informed follow-up frozen
before its own evaluation, and establishes no held-out or console results.

```sh
.venv-ci/bin/python -m pytest -q tests/test_fl10_blend.py
.venv-ci/bin/ruff check train/fl10_blend.py tests/test_fl10_blend.py
```
