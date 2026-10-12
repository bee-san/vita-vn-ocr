# Prospective generic FL10 training

`train/generic_fl10_train.py` prepares and executes a future source-balanced FL10
adaptation. It preserves the released architecture, vocabulary, BN statistics and
console memory contract. It does not score, select, export or promote a model.
Preparation makes no model forwards and never reads training crop arrays. Tiny
generated-fixture steps in the tests validate software behavior, not VN accuracy.

The released initializer is fixed to SHA256
`9fbcb3b2c620a582a003353abb9481d0ee05eb8a5709d4bb1481d4e733597b47`.
Its metadata must declare height 32, width 1.0, bottleneck 96, depth 1, context 2
and context kernel 5, frozen BN statistics and unknown class `〓`. The restored
fp32 source SHA is
`a42818bfa81361ac907ce79ec471c9e0778aa2d7dbe1b0d801207d27d63ecc7e`.
The ordered 3500-class vocabulary SHA is
`7f1cb2eb073ddae5f6c39bdd49562cd3bd3ac757f5ac8c0eca4bd98beda4fed8`;
matching the class count alone is insufficient. Height 24 and bottleneck 32 are
rejected. Existing trainers and historical recipes retain their geometry.

The selector preparation must compare against the exact released W8 baseline,
SHA256 `500e0b842e5f0cb326f55e7d991a52ab66dd24977ccc89bd188228aedc825e84`.
An alternative model with the same geometry cannot substitute for that baseline.
The standalone selector keeps its existing behavior.

Training requires the separately frozen real generic-development preparation from
`train/general_vn_select.py`. The driver uses its full `verify_lock` contract,
including source bytes, roles, reference metadata, code/dependencies, compiler,
runtime library and environment. A stored `ready: true` flag alone is insufficient.
All original readiness quotas remain required: at least three VNs per console,
five native frames, 500 body characters and 200 body hiragana per VN, plus an
explicit book-separated synthetic development guard. False readiness refuses
preparation before source metadata/crop arrays or initializer tensors and creates
no output directory. No real training should start until this cohort is ready.

Make the selector preparation from the integrated checkout that will run this
driver. Its runtime identity deliberately binds the source/compiler command and
checkout directory. A preparation from another worktree cannot be reused merely
because its source files have equal hashes. Keep existing preparations immutable;
prepare a new artifact using the original prospective protocol when needed.

The source policy must reserve every verified development VN and must have no
training artifact or asset byte identity overlapping the bound development
inventory. This check supplements independently reviewed source isolation;
omitted ownership, held-out book provenance or historical sessions cannot be
inferred from rendered pixels. Use the prospective producer and policy contracts
to prepare reviewed inputs. Development and protected references remain separate
from the training labels.

The driver also requires the unchanged Jiten policy at
`data/manifests/fl10_kanji_policy_20261011.json`, SHA256
`8fc9ea20804cc45876e25b60ed669446dc75fc7f0b6b383d65cc4688e050f763`.
It checks every frozen training label using metadata only, before crop arrays or
initializer tensors. A row outside the original ordered vocabulary or Jiten
allowlist rejects the input snapshot. It never removes characters from labels or
broadens the vocabulary; prepare safe whole rows upstream.

Keep the input config and generated artifacts in an ignored private directory.
The config declares exact file hashes, a complete optimizer recipe, and proposal
weights for named policy sources. It cannot pass arbitrary trainer arguments,
validation/selection inputs, teacher targets or a different architecture. Sources
are taken exactly from the policy inventory, with RAM/GT loading, KD disabled and
inversion zero. Augmentation is an explicit per-source boolean. These weights do
not certify consumed exposure ratios; the source-policy ledger enforces the
forwarded prefix caps.

```json
{
  "schema": 1,
  "kind": "general-vn-fl10-training-config",
  "init": {"file": "<restored initializer path>", "sha256": "<fixed initializer SHA>"},
  "vocab": {"file": "<released ordered vocabulary path>", "sha256": "<fixed vocabulary SHA>"},
  "kanji_policy": {"file": "<original Jiten policy path>", "sha256": "<fixed policy SHA>"},
  "source_policy": {"file": "<reviewed training policy path>", "sha256": "<measured policy SHA>"},
  "development": {"file": "<ready generic development preparation>", "sha256": "<measured preparation SHA>"},
  "run": {"name": "generic_all_s1", "out": "<new private output root>"},
  "recipe": {
    "steps": 2000, "batch": 32, "lr": 0.00002, "final_lr": 0.05,
    "warmup": 50, "wd": 0.01, "train_scope": "all", "seed": 1,
    "threads": 6, "eval_every": 200
  },
  "sources": [
    {"name": "broad", "weight": 0.8, "augmentation": false},
    {"name": "game_assets", "weight": 0.18, "augmentation": false},
    {"name": "kanji", "weight": 0.02, "augmentation": false}
  ]
}
```

These example recipe values are a prospective configuration, not selected
hyperparameters or accuracy evidence. Replace placeholders with reviewed inputs
and measured hashes. Paths can be absolute or relative to the config directory.
The output root is resolved once and frozen. Steps must be divisible by the
checkpoint interval so the entire retained cohort has an even spacing.

```sh
.venv-ci/bin/python train/generic_fl10_train.py prepare \
  --config build/general-training/config.json \
  --out build/general-training/prepared.json
.venv-ci/bin/python train/generic_fl10_train.py run \
  --prepared build/general-training/prepared.json
```

Preparation atomically publishes complete JSON without replacing any existing
artifact. Execution and resume reverify config, preparation, policy, initializer,
vocabulary, Jiten, code and environment fingerprints before accessing checkpoints
or crop arrays. The trainer execution identity and durable journal additionally
bind the wrapper, frozen configuration, development lock and fixed geometry.
The existing source-policy crash refusal and exclusive writer rules still apply.
Do not reset a failed run's journal or copy an arbitrary checkpoint over it.

Every scheduled checkpoint is retained using atomic replacement. On successful
completion the parent driver checks the exact complete `ckpt_step*.pt` inventory
and freezes `retained-checkpoints.json` with the step/path/byte SHA for every
candidate. Missing or unexpected checkpoints refuse publication. The historical
trainer convenience files `ckpt_best.pt` and `ckpt_last.pt` have no development
selection meaning here; no validation images or selector forwards run in this
driver.

Supply the complete retained cohort to the separately frozen selector:

```sh
.venv-ci/bin/python train/general_vn_select.py select \
  --lock build/general-development/prepared.json \
  --checkpoints 'build/general-training/runs/generic_all_s1/ckpt_step*.pt' \
  --out build/general-development/selection.json
```

Use the output destination already bound in that selector preparation. The
selector freezes all candidate bytes before scoring, retains its original real
and synthetic guards, and can keep released FL10 if no candidate qualifies.
Training completion does not establish improvement, console speed or memory
measurements, or readiness for model promotion.
