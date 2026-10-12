# Fresh native textbox evaluation

`eval/fresh_native_textbox.py` evaluates one prospectively fixed finder family on
a fresh private test cohort. It compares original GRAD (mode 0), TEXTBOX/C3
(mode 2) and optional recovery/C4 (mode 3), using the same released FL10 model.
It does not select weights, alter defaults or promote a release. Its fixed
runtime source is commit `cfa7ee6ce5c381ab57dcbca37bbde931d3282e54`, with C source
SHA256 `90c30a1c1e1707b818c2bd919243ee589700ff44ceac8b8d7a842b9d55d8d9ca`.

This harness has generated contract tests only until a separate real cohort
review, immutable preparation and execution authorization occur. No accuracy
result follows from the existence of the harness or its passing unit tests.

## Private input schema

Keep the manifest, annotations, source images and execution directory under
ignored `build/` or `data/`. Commit code, source URL/hash manifests and aggregate
results only. Source image paths and all reference/predicted text remain private.

The metadata manifest is JSON with these top-level fields:

```json
{
  "schema": 1,
  "split": "test",
  "purpose": "fresh-native-textbox",
  "original_training_history": "unknown",
  "exclusions": {
    "schema": 1,
    "evidence_file": "recorded-exposure-plan.json",
    "evidence_sha256": "<lowercase SHA256>",
    "vns": ["<excluded VNDB IDs, including known related families>"],
    "release_ids": ["<excluded native releases>"],
    "urls": ["<previously used source URLs>"],
    "sha256": ["<previously used source file hashes>"]
  },
  "sources": ["<source records below>"]
}
```

The evidence file's `excluded` object must contain exactly those four lists.
Its full file hash binds the recorded-use/family provenance, including its own
input pins. Exclusions must cover known training, calibration, development and
previous evaluation exposure according to the reviewed acquisition plan.
The original released model's initializer training snapshot is unavailable:
recorded-manifest exclusions cannot prove historical training independence.
The harness requires that limitation to remain explicit.

Each source record contains:

```json
{
  "id": "fresh-psp-v123-001",
  "split": "test",
  "platform": "psp",
  "vn": "v123",
  "release": "r456",
  "session": "r456-session1",
  "image": "images/source.png",
  "url": "https://example.test/source.png",
  "sha256": "<source file SHA256>",
  "pixel_sha256": "<decoded native RGB byte SHA256>",
  "dims": [480, 272],
  "base_roi": [12, 170, 456, 90],
  "variants": {
    "base": {"roi": [12, 170, 456, 90], "pixel_sha256": "<RGB ROI SHA256>"},
    "expanded-4": {"roi": [8, 166, 464, 98], "pixel_sha256": "<RGB ROI SHA256>"},
    "expanded-8": {"roi": [4, 162, 472, 106], "pixel_sha256": "<RGB ROI SHA256>"}
  },
  "line_ids": ["fresh-psp-v123-001-name", "fresh-psp-v123-001-body1"],
  "hand_checked": true,
  "reviewer_kind": "assistant",
  "complete_eligible_text": true,
  "uncertain": false,
  "scope_exclusions": ["supplementary ruby"],
  "eligibility": {
    "used_in_recorded_training": false,
    "used_in_calibration": false,
    "used_in_development": false,
    "related_to_used_title": false
  }
}
```

PSP dimensions must be exactly 480×272; `vita` must be exactly 960×544.
Coordinates are native `x,y,width,height`. ROI expansion is symmetric by 4/8
pixels and clamps to the native image bounds. Every main annotation box must
fit inside the base ROI. File and decoded RGB hashes are distinct pins; each
fixed ROI has its own RGB byte hash. The fixture values above are illustrative.

`hand_checked` denotes a reviewed transcription, not proof of independent human
review. `reviewer_kind` must identify `human` or `assistant` explicitly. Uncertain
or incompletely transcribed sources cannot enter this test. An approved source
must bind every eligible name/body line in its intended reading order.

The separate private annotation file is JSON:

```json
{
  "schema": 1,
  "split": "test",
  "lines": [
    {
      "id": "fresh-psp-v123-001-body1",
      "source_id": "fresh-psp-v123-001",
      "split": "test",
      "content_kind": "body",
      "box": [24, 198, 380, 20],
      "text": "<private transcription>",
      "uncertain": false
    }
  ],
  "supplementary": [
    {"source_id": "fresh-psp-v123-001", "kind": "ruby", "box": [30, 184, 40, 8]}
  ]
}
```

The only main kinds are `body` and `name`. Ruby stays supplementary and is never
used as a main reference or to filter detected crops. Annotation IDs must bind
exactly once; references outside the source/base ROI, role changes and unknown
main kinds fail before any source image or model is opened.

## Readiness and duplicate accounting

Both consoles need at least three fresh titles. Each title needs at least five
unique native RGB frames, 500 distinct body-text characters and 200 hiragana
characters. These minima cannot be lowered with a command-line option.

Before counting, sources are sorted by stable source ID. Duplicate native RGB images are checked globally; repeated normalized whole-body
transcripts are checked within each platform/title. Each retains the first source
and creates an exclusion receipt for subsequent duplicates. Identical RGB pixels
with contradictory title provenance or main text/ink geometry are rejected. Unique
normalized body lines count toward character quotas once per platform/title,
preventing recurring dialogue from inflating readiness while retaining the same
phrase rendered in different games and fonts. This affects admission and quotas before any
prediction exists; scoring preserves every main reference of every admitted
source and every returned crop.

## Prepare, review and execute once

Preparation validates every manifest role before opening the annotation file.
It then validates exclusions, main references, duplicates and minima before
decoding any source image or reading model/library bytes. It checks native and
ROI pixels, model SHA/header/CRC, dependency versions and fixed source hashes,
then compiles one explicit finder driver and writes immutable protocol and
preparation files. Preparation performs zero finder calls and zero model
forwards. Use a new execution directory:

```sh
.venv-ci/bin/python eval/fresh_native_textbox.py prepare \
  --manifest build/fresh-test/manifest.json \
  --annotations build/fresh-test/annotations.json \
  --model /path/to/FL10_w8.vocr \
  --library /path/to/pinned-scalar-libvocr.so \
  --directory build/fresh-test/audit --compiler gcc

.venv-ci/bin/python eval/fresh_native_textbox.py verify \
  --prepared build/fresh-test/audit/prepared.json
```

The model SHA is
`500e0b842e5f0cb326f55e7d991a52ab66dd24977ccc89bd188228aedc825e84`.
The explicit reviewed scalar library SHA is
`4a5635ba9a0721373f35e8957c0788b74e1725ea8cd96b8200cc5e427536b11b`.
There is no default library, cached finder or automatic model download.
Native construction additionally verifies CRC, W8I, maximum width 1024,
657,664 blob bytes, 124,832 arena bytes and 3,501 classes. Finder modes share the
same frozen reviewed source, rather than three independently changing builds.

After the coordinating reviewer checks the exact driver, source, protocol,
cohort provenance, reference completeness and preparation hash, it supplies an
authorization JSON matching `expected_authorization(prepared_path)`. The CLI
does not generate authorization. That receipt binds the preparation bytes,
protocol, cohort fingerprint, fixed model and candidate source.

```sh
.venv-ci/bin/python eval/fresh_native_textbox.py compare \
  --prepared build/fresh-test/audit/prepared.json \
  --authorization build/fresh-test/audit/execution-authorization.json
```

Run in a fresh process with the pinned environment. Changed roles, files,
code, protocol, compiler, interpreter, dependency versions or native binaries
invalidate preparation. A nonblocking writer lock and exclusive `attempt.json`
protect the one-shot run. A failure preserves its receipt and partial private
prediction log; another comparison of that preparation is rejected. An image,
finder, truncation or reader failure never becomes a blank successful sample.

## Scoring and fixed acceptance

Every finder crop is recognized in native C order using its C polarity.
Zero detections count as an empty prediction and incur the reference deletions.
Blank crops remain recorded. The existing `transcript_score` normalizes each
reference line and returned crop separately, concatenates them and uses a
single Levenshtein alignment with diagonal/deletion/insertion tie order. It
charges all unmatched insertions to body as well as all-text counters. It does
not match predictions to GT boxes or mask UI/name/ruby outputs after inference.
Small kana, dakuten and similar kanji remain distinct under normalization.

Acceptance requires all of the following in each base/+4/+8 variant:

- Every title's all/body full and hiragana errors do not exceed mode 0.
- All twelve pooled variant/scope/script metrics do not exceed mode 2.
- Pooled all-text full errors fall by at least 10% against the actual mode-2
  anchor, using exact integer `100 * C4 <= 90 * C3`. A zero-error anchor cannot
  demonstrate the required relative reduction.
- Body hiragana strictly improves with each console weighted one half and
  titles equally weighted within each console.
- At least two titles per console improve body hiragana.

No development-set count limit is substituted for the fresh anchor. Detection
counts may differ; GT/reference identities and denominators must match. There
is no model ranking, fallback candidate, teacher comparison, retry or automatic
release promotion.

The public aggregate result contains source URLs/hashes, review provenance,
cohort and code fingerprints, readiness/exclusion counts, all per-title counters,
blank/zero-detection accounting and the fixed decision. It contains no pixels,
private paths, reference text or raw predictions. `private-predictions.jsonl`
remains private and its SHA binds the raw audit. Host execution does not prove
physical-console latency, stack peak or resident memory.

Generated verification:

```sh
.venv-ci/bin/python -m pytest -q tests/test_fresh_native_textbox.py
.venv-ci/bin/ruff check eval/fresh_native_textbox.py tests/test_fresh_native_textbox.py
```
