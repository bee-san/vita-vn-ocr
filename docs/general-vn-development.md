# Prospective general-VN development protocol

The new selector applies the same recognizer, vocabulary and preprocessing to every game. It is independent of the original CLANNAD-led experiment: `hiragana_select.py`, `hiragana_gate.py`, the twenty frozen checkpoints/results, runtime and CI remain unchanged. The new protocol must be registered before a future experiment. Preparing it does not evaluate models or establish an accuracy improvement. Google long-run outputs and protected test results never enter development selection.

This selector measures hand-annotated individual line crops. It does not measure
the C finder's whole-textbox behavior. The separate [prospective textbox guard](general-vn-textbox-guard.md)
freezes reviewed input ROIs before training, then compares only the one selected
export with the released baseline through the actual C finder/W8I pipeline.
The prospective training driver requires that preparation; both checks must pass.

Real development strata are `(platform, vn)`. A game appearing on both consoles receives two separate nonregression checks and counts as one distinct VN for the required gains. Release/session, line IDs, styles, source-image hashes and geometry are preserved in aggregates. Native dimensions establish platform when old manifests lack an explicit platform field; contradictory metadata is rejected. Speaker-name lines are excluded from body strata using annotation tags, without inspecting the recognized words. No font-family identity is inferred from screenshot appearance, styles or game titles.

## Readiness before scoring

Each console needs at least three VNs. Each VN/platform needs at least five unique native frames, 500 normalized body characters and 200 reference hiragana. Repeated lines from one image do not manufacture more frames; names cannot satisfy body quotas. These conservative quotas are a preregistered design choice, not empirically optimal thresholds. A protocol change requires a new version/registration.

The existing development cohort is not ready:

| Stratum | Frames | Body characters | Body hiragana |
|---|---:|---:|---:|
| PSP v12775 | 5 | 90 | 46 |
| PSP v5297 | 4 | 142 | 86 |
| Vita v1015 | 2 | 47 | 26 |
| Vita v13766 | 1 | 25 | 17 |
| Vita v4 (CLANNAD) | 19 | 207 | 127 |

There are two PSP and three Vita VNs. All five fall short of the body-text quotas. These are label/provenance counts only, obtained without decoding screenshots or scoring a model. The additional development annotations are assistant reviewed; they are not independent human test annotations. The [small derived readiness record](../eval/general_vn_dev_readiness_20261011.json) pins its development manifest hashes.

## Qualification and exact ranking

Every real VN/platform must retain its own baseline full and hiragana error counts, both for all lines and for body lines. A gain in another VN or in speaker names cannot hide a regression. Every separately registered book-held-out synthetic guard must also retain full and hiragana errors. Reference IDs, provenance and character counts must match.

For qualified candidates, the primary score is exactly:

`H = (mean(per-VN PSP hiragana errors / characters) + mean(per-VN Vita hiragana errors / characters)) / 2`.

The two consoles have equal weight, and VNs have equal weight within each console. A large game or platform corpus cannot dominate the result. Ratios use exact rational arithmetic, never rounded CER display fields. Ties use analogous full-text score, total real full errors, total synthetic full errors, earliest training step and checkpoint path, in that order.

Replacement also requires strictly lower `H` and hiragana error reductions in at least two distinct VN IDs. The eligible candidates are ranked after this requirement is applied. If none qualifies, retain the released baseline without claiming an improvement. CLANNAD has the same conditions as every other VN.

## Frozen preparation and future selection

Keep source images/transcripts, protocol locks, checkpoints and exports ignored. The example JSON declares the complete rule. Preparation checks development roles before source-image access, binds source bytes and manifest/dependency/code hashes, and records readiness. It does not decode image pixels or evaluate candidates. Selection refuses an inadequate cohort before decoding pixels or constructing the recognizer.

```sh
.venv-ci/bin/python train/general_vn_select.py prepare \
  --protocol train/general_vn_protocol.example.json \
  --baseline /private/FL10_w8.vocr \
  --dev-manifest /private/general-vn-dev/labels.jsonl \
  --synthetic-dev broad=/private/book-heldout/synth_00000.npz \
  --synthetic-dev bitmap=/private/bitmap-dev/synth_00000.npz \
  --out build/general-vn/protocol.json \
  --selection-out build/general-vn/selection.json

.venv-ci/bin/python train/general_vn_select.py select \
  --lock build/general-vn/protocol.json \
  --checkpoints 'build/general-vn/runs/*/ckpt_step*.pt' \
  --out build/general-vn/selection.json
```

The second command is for a future ready cohort; it has not been run on existing checkpoints. It freezes the complete checkpoint-path/hash inventory before baseline or candidate scoring, uses scalar C W8I, preserves FL10 topology/vocabulary order and the 657,664-byte model / 124,832-byte arena at maximum width 1024, and locks the chosen export hash. Changed references, code, dependencies, candidates, output destination or a completed selection cannot silently reopen a decision. A lock can be reverified after an interruption; exports are ignored. Aggregates contain hashes and line metadata, without dialogue text or pixels.

Preparation compiles a dedicated library beside the ignored lock with an explicit compiler invocation, `-DVOCR_NO_NEON`, fixed scalar flags and a minimal fixed build environment. It records the compiler executable hash/version, complete argv, source hashes and resulting library SHA. It also records the actual Python executable/version, host identity and installed torch, NumPy and Pillow versions. Requirement-file pins remain separately bound. Selection verifies these original artifacts before rebuilding, then invokes the compiler again into a staging file and requires the complete identity to match before replacing the library. It passes that selected library explicitly to `Vocr`; the default `libvocr.so` mtime cache is bypassed. Identity checks run before pixels/scoring and before every candidate, and resume uses the same checks. Compiler or dependency changes require a new preregistration rather than silently reopening the old one. `prepare --compiler /path/to/gcc` can select another compiler before freezing.

## Prospective promotion, separate from the old gate

A future generic promotion protocol must be separately implemented and registered before test access. It must require identical references, scalar C W8I, fixed FL10 topology/vocabulary/memory limits, separate per-console full-text nonregression and per-VN/platform full plus hiragana nonregression on every shared and private test title. Require a strict console-balanced overall hiragana reduction and gains in at least two distinct shared VNs. CLANNAD must not uniquely be required to improve. Fixed general finder/pipeline measurements must also be preregistered, with no game-specific margins, script folding or detection matching exceptions.

This describes a prospective gate; it does not modify or apply `hiragana_gate.py`. There are no new test scores or promotion results. Lock one development-selected model before the held-out run. Reject failures without choosing another checkpoint, font mixture or margin using those test results. The long CLANNAD/Google comparison remains diagnostic and cannot select recipes or supply training labels. Physical-console performance and deployment remain separate validation steps.

A later training-source protocol should bound the share of any one game's native font/text assets across streams, retain broad multi-font text, keep corrective patterns and rare kanji bounded, and log actual consumed counts separately from prefetch draws. Native font-family/source-VN IDs and per-book provenance must be declared by producers; absent identities cannot be invented. This selector does not alter any current source mixture or sampler.

## Offline checks

```sh
.venv-ci/bin/python -m pytest -q tests/test_general_vn_select.py
.venv-ci/bin/ruff check eval/group_metrics.py train/general_vn_select.py tests/test_general_vn_select.py
```

Tests use generated references, aggregate records and fake readers; no protected screenshots or Google output are read.
