# Prospective end-to-end development guard

The existing generic selector measures hand-annotated individual line crops. It
does not measure whether the console finder reads a user's whole selected dialogue
box. This separate guard measures that pipeline after the unchanged selector has
locked one candidate. It cannot rank checkpoints, reopen selection, use test or
Google references, promote a runtime, or deploy a model. The old twenty-checkpoint
experiment and its gate remain unchanged.

Freeze this protocol and reviewed ROI geometry before future training. The
prospective generic training driver now requires the resulting textbox preparation
in its schema-2 configuration, verifies it before initializer tensors or training
crop arrays, and binds its hash into the run/checkpoints. Its previous preparation
artifacts cannot silently adopt the new requirement. Readiness remains exactly the
existing minimum: three VNs per console, five native frames, 500 body characters
and 200 body hiragana per VN/platform, plus the original synthetic guards.

## Reviewed input scope

Keep plans, sources, transcripts, generated manifests/builds and outputs ignored.
The plan contains `schema: 1`, `split: "dev"` and `sources`. Each source declares:

- Stable `id`, `split: "dev"`, `platform`, VNDB `vn`, `session`, `release`, native
  `dims`, source PNG `sha256`, and `image` path.
- `base_roi: [x, y, width, height]`, with native integer coordinates chosen from
  reviewed image geometry before any model output.
- Explicitly ordered `line_ids` from the same ready line-crop development lock.
  Every eligible dialogue/name line inside the ROI must be included; each bound
  line is covered exactly once across sources. There is no guessed spatial order.
- `hand_checked: true`, `complete_eligible_text: true`, `uncertain: false` and
  `scope_exclusions`, an explicit list, possibly empty. Assistant review is not
  independent human review. Existing line annotations retain their review flags,
  body/name kind, styles and provenance.

A predeclared ROI may exclude a vertical speaker sidebar when evaluating the
current horizontal reader. That limitation must be explicit in geometry and
`scope_exclusions`. Horizontal/inline names inside the ROI belong in the complete
all-text reference. No name pixel masking occurs. Incidental UI recognized as
extra output is an insertion, including in the conservative body score.

All sources receive the same three cases: the base ROI, symmetric expansion by
four native pixels, and symmetric expansion by eight native pixels. Expansion is
clamped to the native frame. No variant may clip any declared ink box. These are
input-selection sensitivity cases, not changes to runtime crop margins or
device/game-specific OCR rules. A case that clamps to identical pixels is still
reported honestly. The three cases never select a model or margin.

`materialize` rejects the entire wrong-role/malformed metadata cohort before any
source pixels, then binds decoded RGB and each ROI's pixels with SHA256. It runs
neither a detector nor a recognizer. `prepare` revalidates reviewed metadata,
source bytes, the unchanged ready selector lock and exact released baseline. It
does not decode pixels or execute a model. Test/train roles, partial references,
uncertainty, missing variants, changed source identities and ink clipping fail.

## One fixed comparison

The guard always builds its own source-consistent scalar runtime and C GRAD host
driver beside the private lock. Compiler executable hash/version, exact argv,
minimal build environment, C sources, output library/binary hashes, Python and
installed torch/NumPy/Pillow versions, dependency files and code are frozen.
Original cached bytes are verified before unconditional deterministic rebuilds;
a stale cache, compiler change or dependency drift fails. It uses no other
worktree's runtime cache, including the frozen CLANNAD long-run library.

Each native RGB ROI becomes RGBA8 with opaque alpha. The existing C finder returns
the boxes and dark flags once. Both models receive the identical returned RGBA
crops, unchanged vertical/horizontal geometry and the corresponding C inversion
flag; the C recognizer performs its own console preprocessing. There is no Python
polarity approximation or detector-to-GT matching. Zero detections produce an
empty transcript and charge missing reference text. C failure, truncation,
invalid boxes/polarity or inference errors abort; they cannot become a blank
successful prediction. The required W8I blob/arena are 657,664/124,832 bytes at
maximum width 1024, with the original FL10 topology and ordered vocabulary.

Predictions concatenate in C finder output order. References concatenate in the
reviewed explicit line order. Each reference line and each detected crop uses
`eval.metrics.norm` independently, preserving small kana and dakuten distinctions.
The joined strings are not normalized again: dots on adjacent lines/detections
must not collapse across a body/name boundary or change reference attribution.
Guard-local full-transcript Levenshtein alignment uses the existing deterministic
diagonal, then deletion, then insertion tie order. It is never
optimized for the body score. Full errors include every edit. Hiragana errors
charge reference hiragana substitutions/deletions and hiragana insertions, as in
the existing metrics. Body counters charge edits to reference body characters and
**every unmatched insertion**, even beside a name. A missed name remains an
all-text error; it cannot erase a missed body line.

For every ROI variant and every `(platform, vn)`, candidate full and hiragana
errors must not exceed baseline in either all or body scope. Each variant must
strictly lower the exact console-balanced hiragana rate in **both** scopes, and
at least two distinct VNs must gain in both scopes in each variant. Gains cannot be
pooled across variants. Each console receives half the
weight; VNs receive equal weight within their console. Integer errors/reference
counts and exact fractions drive the decision. No PSP aggregate gain can conceal
a Vita/title/body regression. Existing line-crop/synthetic gates remain mandatory.

## Commands

Run from the integrated checkout that will train/evaluate. First prepare the
unchanged generic line-crop selector and manually freeze the reviewed ROI plan.

```sh
.venv-ci/bin/python train/general_vn_textbox_guard.py materialize \
  --line-lock build/general-vn/protocol.json \
  --plan build/general-vn/roi-plan.json \
  --out build/general-vn/roi-manifest.json

.venv-ci/bin/python train/general_vn_textbox_guard.py prepare \
  --line-lock build/general-vn/protocol.json \
  --roi-manifest build/general-vn/roi-manifest.json \
  --out build/general-vn/textbox-prepared.json \
  --result-out build/general-vn/textbox-result.json
```

Pin `textbox-prepared.json` as the training configuration's
`textbox_development: {"file": "...", "sha256": "..."}` before preparing/running
training. After the original selector completes a strict development improvement,
bind exactly its chosen export. A baseline fallback has no eligible comparison.

```sh
.venv-ci/bin/python train/general_vn_textbox_guard.py bind \
  --lock build/general-vn/textbox-prepared.json \
  --selection build/general-vn/selection.json \
  --out build/general-vn/textbox-prepared-candidate.json

.venv-ci/bin/python train/general_vn_textbox_guard.py compare \
  --candidate-lock build/general-vn/textbox-prepared-candidate.json \
  --out build/general-vn/textbox-result.json
```

Every command refuses replacement of completed artifacts. Comparisons use an
exclusive writer lock. Interrupted runs may repeat the same frozen pair and
inputs; they cannot change candidate, references, runtime, destination or rules.
Results retain aggregate errors/counts, reference/source hashes, metadata and
actual detections; no dialogue text, model transcripts or pixels are persisted.
Only small derived aggregate results may later be committed.

This implementation has no real baseline/candidate result. Generated tests check
the software contract, including C finder behavior on generated pixels, and do
not establish VN accuracy or physical-console performance. A separate held-out
promotion protocol must be frozen before protected access and apply the same
fixed chosen model, full pipeline and per-VN/console checks. Failure rejects that
candidate without checkpoint/margin/source-mixture retuning from test results.
No existing held-out gate is loosened or replaced here.

```sh
.venv-ci/bin/python -m pytest -q tests/test_general_vn_textbox_guard.py \
  tests/test_generic_fl10_train.py tests/test_general_vn_select.py tests/test_linefind.py
.venv-ci/bin/ruff check train/general_vn_textbox_guard.py train/generic_fl10_train.py \
  eval/linefind/cfinder.py tests/test_general_vn_textbox_guard.py tests/test_generic_fl10_train.py
```
