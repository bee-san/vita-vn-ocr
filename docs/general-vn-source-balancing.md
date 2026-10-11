# Prospective general VN source balancing

This opt-in training policy limits how much any game or deliberate correction can
influence a future model. It does not change the historical twenty-checkpoint
experiment, its recipes, the existing hiragana selector/gate, the runtime, CI, or
the separate general-VN development thresholds. Metadata validation performs no
training and does not establish readiness. Authorized training uses a ready development
cohort and independently verified source isolation under its frozen recipe. CLANNAD has the same source
limit as every other VN; long-run Google comparisons cannot supply training labels
or select a recipe.

The limits apply across all streams to each prefix of samples actually forwarded
to the training model. They use integer arithmetic, without floating tolerances:

| Contribution | Maximum after N forwarded rows |
| --- | --- |
| Each VN's extracted font or text | floor(N / 20), 5% |
| Any game asset, including screenshot scenery | floor(N / 5), 20% |
| Each deliberately selected correction pattern | floor(N / 200), 0.5% |
| Deliberate similar-kanji corrective stream | floor(N / 50), 2% |

A real game crop contains game font/text and counts against that VN's 5% limit.
A row containing both text and font from the same VN counts once for that VN.
A row containing assets from several VNs counts once for each contributing VN,
and once for the 20% union. A game background alone counts against 20%, without
being classified as extracted font/text. Ordinary occurrences of a kana or kanji
in broadly sampled text are not deliberate correction patterns.

The width-bucketed candidate order is checked immediately before constructing the
model batch. A capped row or a row with unknown ownership is replaced by one row
from the verified broad fallback. For example, the first game-font row can enter
at forwarded row 20, and the first explicitly selected correction pattern at row
200. There is no retry loop that can hang when a candidate source has no allowance.
Fallback widths may increase the final padded batch width; correctness takes
priority over this host-side efficiency cost.

`train2` records exposure immediately after each successful training forward,
including a forward without an optimizer update. It records optimizer rows
separately. Validation forwards remain separate development measurements. An
exception during a model forward aborts the run; it cannot silently skip that
batch and continue using a stale ledger. The report distinguishes weighted
prefetch draws, extra fallback draws, rejected candidates, buffered rows,
forwarded rows, and optimizer rows. Existing `mix` fields retain their historical
prefetch meaning; `source_policy.consumed` is the actual exposure accounting.

An aggregate hash-chained forward intent is flushed and fsynced before invoking
the model. It carries the existing ledger and claims no successful rows. A
successful forward resolves that intent with a separate durable record; optimizer
boundaries are recorded separately. Duplicate intents and forward records without
an intent are rejected. An unresolved intent refuses resume even when a supplied
checkpoint tail matches it, covering partial forwards, process death and failures
between model return and durable forward recording. Each checkpoint binds the journal tail. A loss,
backward or optimizer failure after a successful forward leaves that exposure in
the journal. The driver refuses to resume if the journal is ahead of, missing
from, truncated relative to, or otherwise inconsistent with the model checkpoint.
Checkpoint publication uses a temporary file followed by atomic replacement.
Arbitrary crash recovery is refused in this first version; uninterrupted runs and
resumes at a completed checkpoint boundary are reproducible. An exclusive run
writer lock is acquired before writing args, metrics or checkpoints, including
non-policy invocations of this driver, so concurrent invocations cannot race a
policy run's artifacts.

## Producer contract

Private NPZ shards keep their existing arrays and add immutable row provenance to
each `meta` entry. Preserve `split: "train"` at the top level:

```json
{
  "split": "train",
  "source_provenance": {
    "schema": 1,
    "sample_id": "producer:source-snapshot:row-000001",
    "asset_ids": ["ofl-font", "book-chunk", "procedural-scene"],
    "correction_pattern_ids": [],
    "kanji_corrective": false,
    "broad_sampling": true
  }
}
```

Sample IDs must be globally unique in the frozen input inventory. Selected chunk,
font, metric and background references survive cropping, width bucketing and
fallback replacement. A targeted producer adds stable deliberate IDs such as
`kana:ずっ` or a similar-kanji pair ID. It does not tag the same sequence when it
occurs organically in an ordinary stream. The registry must bind the actual
selected assets rather than all assets that were available to the renderer.

The policy's `assets` object maps each ID to a record with a lowercase SHA256,
`kind` (`font`, `text`, `background`, `real_crop`) and `origin` (`game`, `ofl`,
`non_game`, `procedural`, `unknown`). A game-derived asset requires canonical
VNDB `vn: "vN"` and `release: "rN"` identities. Missing ownership is unknown.
An OFL font requires `license: "OFL-1.1"`; non-game text requires
`game_free_reviewed: true`. A procedural identity pins the reviewed generator
and recipe; it cannot hide game text or game-derived name/frequency inputs.
Procedural text also requires `game_free_reviewed: true`.
Identical asset bytes cannot have conflicting kind/ownership identities.
Text source identities may hash the canonical source-file SHA, chunk index and
raw UTF-8 chunk SHA together, retaining those components in the registry. This
keeps identical phrases from independent source files distinct without inventing
game ownership.

The fallback source is a separate, entirely verified broad source. Every row
must have `broad_sampling: true`, no deliberate patterns or kanji correction,
and at least one font, text and background reference. Every font reference must
be OFL, every text reference must be reviewed non-game or procedural, and every
background must be procedural. Game, unknown and real-crop references exclude
fallback eligibility. Usable fallback items also require valid CTC labels.
Every synthetic row also needs complete font/text/background coverage to have
known provenance. Partial kind coverage is unknown and uses fallback, even in a
non-fallback source. An explicitly declared game `real_crop` reference represents
the entire native input crop, including its font/text/background, and can supply
complete provenance by itself. Its ownership still counts against that VN's 5%
and the 20% union. Unknown rows in other streams may fall back; unknown fallback rows abort loading.

The policy JSON has this shape (hashes below describe fields, not measured assets):

```text
{
  schema: 1,
  kind: "general-vn-source-policy",
  assets: { assetID: {kind, origin, sha256, ...ownership/license evidence} },
  fallback_source: "broad",
  forbidden_vns: [ ...development and protected VN IDs... ],
  shards: [
    {file: "broad/synth_00000.npz", source: "broad", split: "train",
     sha256: <lowercase hex64>, bytes: <positive integer>, samples: <positive integer>}
  ]
}
```

Shard paths are relative to the policy directory and cannot escape it. The policy
must cover exactly the configured sources, without duplicate shards or sample
IDs. Every referenced forbidden VN is rejected. Pin this registry and the complete
development/protected inventory before producer decoding/rendering; the consumer
cannot infer completeness of an omitted game's historical source inventory.
Absent row roles/registry entries become unknown and cannot qualify as fallback;
explicit non-training roles, malformed hashes or malformed identities abort.
Declared shard roles must be explicitly training. All row roles and shard hashes
are validated before loading crop arrays. Old shards are not rewritten or
retroactively certified by a source name or font family.

## Validation and future use

Metadata/source-byte validation makes zero model forwards and does not load crop
arrays:

```sh
.venv-ci/bin/python train/source_policy.py \
  --policy build/general-training/policy.json \
  --out build/general-training/policy-validation.json
```

After producer, isolation and development readiness review, a future run can add
`--source-policy` to its separately frozen recipe:

```sh
.venv-ci/bin/python train/train2.py --run future-general-vn --steps 1000 \
  --source-policy build/general-training/policy.json \
  --src 'broad:0.80:build/general-training/broad/*.npz:stream=0' \
  --src 'game_assets:0.18:build/general-training/game-assets/*.npz:stream=0' \
  --src 'kanji:0.02:build/general-training/kanji/*.npz:stream=0' \
  --keep-evals --eval-every 200
```

These example weights are proposal probabilities; they do not certify consumed
ratios or establish a selected recipe. Counts in each checkpoint/report do.
This first implementation uses RAM sources only. `stream=1` is rejected because
the old asynchronous shard loader does not expose resumable queue/cursor state.
It requires the shards' GT labels and rejects external teacher target files,
teacher label modes and enabled KD/kdonly options: those auxiliary inputs need a
future exact byte/provenance inventory. Historical teacher recipes are unaffected.
This is a host-training limitation and adds no console runtime memory.

Policy mode rejects `--val` and `--select` before decoding any validation pixels.
Development roles, references and bytes belong to the separately frozen general-VN
selector, which scores the retained checkpoint inventory. Use `--keep-evals` when
the recipe requires every evaluation-step checkpoint. This first mode does not
introduce an unpinned trainer validation schema; legacy non-policy `--val` behavior
is unchanged.

Each policy checkpoint saves its policy SHA, actual ledger, draw/rejection counts,
source permutations/cursors and RNGs, mixer/augmentation RNG, buffered sample
references, and Torch RNG including dropout. Resuming reloads exact pinned shards
and validates initial checkpoint bytes (or seeded-random initial state), raw and
ordered vocabulary hashes, driver/model/helper source hashes and actual installed
Python/Torch/NumPy/OpenCV versions and dependency locks, plus the actual host,
platform/libc, Torch build/parallel configuration, MKLDNN/determinism/matmul/thread
settings and OpenCV build/optimization/OpenCL settings. The frozen-BN helper
`restore_blob.py` is pinned along with the other execution source files.
Supplying the original `--init` on resume also
checks its byte SHA; the stored initial identity persists when that option is
omitted. The ordered checkpoint vocabulary must still match.
Readable `source_execution.json` freezes this provenance before the first forward,
including runs that later abort without a model checkpoint.
Resume rejects changed policy, source options/weights, batch/bucket settings, invalid
ledger accounting or missing immutable row identities. A resume cannot add/remove
the policy, reset its caps, or change optimizer/dropout-affecting recipe parameters.
Forward-only batches count even without optimizer updates; optimizer accounting
cannot count a forwarded batch twice.

Generated tests exercise every exact prefix limit, multi-VN unions, scenery,
unknown provenance, fallback eligibility, role/hash guards before pixels,
prefetch-versus-forward accounting and resumed batch identities. A four-step tiny
generated-pixel training test interrupts after a checkpoint and verifies identical
final model/dropout RNG and exposure state after resume. It is a software check,
with additional generated checks for initial-model byte changes, durable loss
failure accounting, journal truncation/corruption and simultaneous writers,
not a measured VN accuracy experiment. No protected images, Google transcripts or
real candidate scoring are used by these tests.
