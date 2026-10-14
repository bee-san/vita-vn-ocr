# The owner pool: training lines from the owner's own console captures

The owner's Google Drive drop of 2026-10-13 (`vn-ocr-dataset-20261013.zip`, sorted into
`shared/private/owner-games/` by owner-drive-ingest) holds live PSP and PS Vita framebuffer captures.
The training titles among them are labelled here into a training pool. The pool is for the next
training round (VOCR4), not VOCR3 round 1.

**Private data.** The frames are the owner's copyrighted captures. Frames, readings, sheets, labels,
crops and the npz files stay under `shared/private/owner-games/_build/owner-pool/`. Nothing from them
went to Codex or any other model or third party: the labelling has no Luna vote. The repo holds only the
code and the aggregate counts in `realplus/manifests/owner/stats.json`.

## Folders and eligibility (`owner_frames.py inventory`)

`owner_frames.py inventory` re-runs the clean check at ingest against `reserved_vns` on the 2026-10-13
dump. It also records every file's hashes, size, duplicates and source resolution: a 960x544 frame that
is a 2x upscale of a PSP picture (an Adrenaline capture) is detected by `upscale_2x`.

| folder | VN | frames | role | eligibility at ingest |
|---|---|---:|---|---|
| psp-planetarian-v34 | v34 planetarian ～ちいさなほしのゆめ～ | 202 (480x272) | **training** | owner decision (below); `reserved_vns`: shared-release |
| psp-sharin-no-kuni-v57 | v57 車輪の国、向日葵の少女 | 217 (480x272) | **training** | not reserved |
| vita-chaos-child-v14018 | v14018 CHAOS;CHILD | 159 (960x544) | **training** | not reserved; 1 frame refused (below) |
| vita-clannad-v4-TEST | v4 CLANNAD | 2,206 | test only | protected (CLANNAD) |
| psp-amagami-v1281-TEST | v1281 Amagami | 113 | test only | development VN |
| psp-narcissu-portable-v10-TEST | v10 Narcissu | 200 | test only | shares a release with test-set v18738 |
| vita-danganronpa-v3-v18334-TEST | v18334 Danganronpa V3 | 84 | test only | shares a release with test-set v7014 |
| vita-iwaihime-matsuri-v17863-TEST | v17863 Iwaihime Matsuri | 240 | test only | development VN (teacher_val) |

- No file is a duplicate of another, across all folders (decoded-pixel sha256).
- Every frame is native: no 960x544 frame is a nearest-neighbour 2x upscale.
- 1 CHAOS;CHILD frame reads as a possible smooth 2x upscale (`upscale_2x`, the detector's known false-positive
  rate is about 1 in 262 native Vita frames). `pool-prep` refuses it, so 158 CHAOS;CHILD frames enter.
- The pool reads **577 training frames**: 419 PSP, 158 Vita.

The test-only folders were never read, planned or labelled. Their frames were used only as the
comparison side of the leak check below.

### planetarian (v34): the owner's decision

`reserved_vns` reserves v34 as `shared-release`: the PC Key 10th Memorial BOX (r3748) holds both it and
CLANNAD. The owner made it training data:

> Owner (2026-10-13 19:21Z/19:27Z, recorded by owner-drive-ingest in `prompts/_rules.md`): ... planetarian
> (v34, Key) is TRAINING data by owner decision, although reserved_vns reserves it as shared-release (the
> PC Key 10th Memorial BOX with CLANNAD): it is in no eval or development set (vita-test, psp-test,
> teacher_val, devnative: 0 rows; thumbnail max 0.94 < 0.95). Its PSP port and CLANNAD's Vita port are
> both Key/Prototype releases, so any CLANNAD score of a model trained on planetarian frames must disclose
> that exposure. CLANNAD (whole family) stays a protected test title.

> Owner 20:47Z: planetarian (v34, all releases) is TRAINING data, explicitly allowed even where it
> overlaps an eval set. Use all 202 owner frames for training. If planetarian lines are in teacher_val,
> devnative, vita-test or psp-test, remove them from those sets going forward ... Do not train on any
> other protected game.

- **The override.** `owner_frames.OWNER_DECISIONS` lets v34 through only for the reservation reason
  `shared-release`. A test, development or CLANNAD reservation of the same VN would still refuse it.
  `native.py` honours the decision only in an owner pool (`NATIVE_OWNER_POOL=1`).
- **The 0-row check**, re-run here:
  - v34 is not in eval's `test_games.json` "all", the union of vita-test's and psp-test's VNs.
  - `eval/manifests/vita-test.jsonl` and `psp-test.jsonl` name it 0 times.
  - teacher_val and all 12 devnative npz files (v1, v2, v3; strict and tv_native; ink and GRAD) hold 0
    rows of it. No devnative JSON names it.
  - So no evaluation or development set changes, and no score needs re-reporting.
- **Synthetic data.** v34 stays in `data/manifests/synth3_exclude_clannad.json`, as a VN sharing a
  release with CLANNAD. synth3 does not use its names or scenes; that is unchanged here.
- **Disclosure.** planetarian's PSP port and CLANNAD's Vita port are both Key/Prototype releases. Any
  CLANNAD score of a model trained on this pool must disclose that exposure.

### Leak check (`eval/tools/leakcheck.py`, thumbnail correlation >= 0.95)

The 577 training frames against every protected and development screenshot set, and against the
owner's own test folders:

| set | screenshots | pairs >= 0.95 |
|---|---:|---:|
| vita-test + psp-test (eval) | 157 | 0 |
| devnative cohort | 346 | 0 |
| teacher_val | 646 | 0 |
| linefind | 264 | 0 |
| the owner's -TEST folders | 2,843 | 28, all cross-VN |

The 28 pairs are 26 planetarian frames against one Iwaihime Matsuri frame and 2 against one CLANNAD
frame. They are white-out scenes with a dialogue box at the bottom: the thumbnails match on layout, and
the three games' frames and text differ (checked by eye). No frame was dropped.

## Reads and plan

`devnative/read.py` read the 577 frames offline at native size: MeikiOCR plus PP-OCRv6 small, four readings
per line, 39 s in one heavy.sh slot. `pool.py plan` found **946 candidate lines**:

| VN | frames | candidate lines | auto (all four readings agree) | for review |
|---|---:|---:|---:|---:|
| v34 planetarian | 202 | 325 | 190 | 135 |
| v57 Sharin no Kuni | 217 | 380 | 65 | 315 |
| v14018 CHAOS;CHILD | 158 | 241 | 143 | 98 |
| **all** | **577** | **946** | **398** | **548** (t1 368, t2 13, t3 167) |

None of these lines is a line teacher or realdata labelled: the frames are new to the repo.

## Labelling: devnative's training rule, no Codex

The labelling runs main's `realplus/pool.py`, realplus v2's driver of devnative's frozen pipeline, with
`REALPLUS` pointed at the owner pool. It has no Luna step. The rule is `decide.py final --mode train
--auto-consensus`:

- a line all four readings agree on is kept without review (auto);
- a reviewed label needs at least one engine family (MeikiOCR or PP-OCRv6 small); a label no engine read
  is dropped;
- here every label only one family supports also had a blind second Claude look (pass 2), which must
  agree.

| stage | who | lines | result |
|---|---|---:|---|
| pass 1 (`review/pass1/b0001`-`b0003`, 69 sheets) | review-nf-14 (Claude) | 548 | 310 picks; 234 skipped cut, 1 ui; 3 typed (no engine support, dropped) |
| pass 2, blind, single-family labels (13 sheets) | native-frames (Claude, another session) | 87 | 86 agree; 1 disagrees (a full-size あ read as small ぁ in pass 1), dropped |
| edge check, final crops (14 sheets) | native-frames | 181 | 178 hold the line; 3 cut (an opening 「 or a closing 」 just outside) |

The final labels: **707 kept** of 946 (398 auto, 223 both engine families, 86 one family plus pass 2).

Most of Sharin no Kuni's review lines were skipped as cut: 224 of 315. MeikiOCR's line boxes on its PSP
text stop a character or two short of the line, leaving the closing 。 or the last character outside
the frame (checked on sheets). The reviewers were right to skip them. A re-read with a frame grown to
the text box could recover them in a later round.

## Pool (`native.py build`, `NATIVE_OWNER_POOL=1`)

`native.py build --tag v1 --margins 4,8` cuts each kept line from its native frame. The npz files use
teacher_train's shard format, in console polarity, with `pool: "owner"` and `session: "owner/<folder>"`
in the meta.

| | |
|---|---:|
| **lines** | **595** from 3 VNs |
| per VN | v34 planetarian 323, v14018 CHAOS;CHILD 227, v57 Sharin no Kuni 45 |
| how | 348 auto, 164 both engine families, 83 one family plus pass 2 |
| dropped by the build | 104 repeated name plates (realplus's per-VN repeat cap), 3 edge cuts, 5 with no usable crop |

The crop files:

| file | crops | PSP / Vita | sha256 |
|---|---:|---|---|
| `native_train_ink_v1.npz` | 594 | 368 / 226 | `a2f621d910602cf70dbea623d06ca81fcda662fe44333e970f7452710d66a61a` |
| `native_train_grad_v1.npz` | 540 | 331 / 209 | `90f1675aef4c9203904329757934e2b4b07a5d46ac71999d9a445d969d476883` |
| `native_train_ink_m4_v1.npz` | 585 | 359 / 226 | `ffc638ef3ff860b00633d8bb277fac1f2b916de09b5f9a30b9f4eef126c46041` |
| `native_train_ink_m8_v1.npz` | 431 | 205 / 226 | `9c17464cc62376151bfc908c210ff3aa22e27a0cbe708464fa695a5dc574c895` |

- **ink**: eval's crop widened to the labelled characters, as in the native pool. One line has no ink crop:
  another detected line enters it.
- **grad**: the console GRAD finder's box, holding the line and nothing else.
- **ink_m4 / ink_m8**: the ink crop grown by 4 or 8 px on every side and clamped to the frame. This is the
  fresh-native harness's multi-margin idea, applied to one line. A variant is kept only when no other
  detected line enters it, by the base crop's own mixed test; 9 and 163 were dropped that way, mostly
  PSP lines with tight line spacing. The variants carry the same labels and are optional augmentation.

`train2.load_shard` loads every row of all four files (0 dropped). `realplus/manifests/owner/stats.json`
is the committed, aggregate-only record: per-folder eligibility, leak pair counts, label and build
counts per VN, file sha256 and the precision summary. It holds no file name, hash, box or text of a
private frame.

### Precision

The check is independent and blind (`native.py check`, `check-score`, `CHECK_STRATA` 40 auto + 120
reviewed, seed 20261013, drawn from the pre-build before v1). A Claude helper that labelled none of the
lines (review-nf-15) typed the framed text of each crop with no candidate shown. The crop for each line
was drawn at random from the files that hold it, so the margin variants are covered too.

| | agree | rate (Wilson 95%) |
|---|---:|---|
| **all** | **160 / 160** | **100% (97.7-100%)** |
| auto / reviewed | 40 / 40, 120 / 120 | 100% (91.2-100%), 100% (96.9-100%) |
| ink / grad / ink_m4 / ink_m8 | 38 / 38, 39 / 39, 55 / 55, 28 / 28 | |

v1 is the pre-build with the meta's `pool` set to `"owner"`: same lines, crops and labels.

## Limits

- **Sharin no Kuni** gives only 45 lines. 224 of its 315 review lines were cut by the reader's boxes (see
  Labelling), and the repeat cap drops its many repeated name plates.
- **Small and lossless.** 595 lines from three games, captured as lossless PNG. VNDB's frames are Q90
  JPEGs, so this pool has none of their artefacts.
- **One reviewer per stage.** The pass-2 and edge looks were made by this agent, in a session separate
  from pass 1. The precision check is the independent measurement.

## Reproduce

```sh
B=$HERD/shared/private/owner-games/_build; OP=$B/owner-pool
.venv-ci/bin/python realplus/owner_frames.py inventory            # -> $B/inventory.jsonl, summary.json
.venv-ci/bin/python realplus/owner_frames.py pool-prep --pool $OP # -> $OP/reads/shots.jsonl
HEAVY_SLOTS=12 $HERD/heavy.sh timeout 7200 unshare -rn teacher/.venv/bin/python devnative/read.py run \
    --shots $OP/reads/shots.jsonl --out $OP/reads --procs 5
export REALPLUS=$OP NATIVE_POOL=$OP NATIVE_OWNER_POOL=1
.venv-ci/bin/python realplus/pool.py plan
.venv-ci/bin/python realplus/native.py known && .venv-ci/bin/python realplus/native.py filter-plan
# pass 1 batches (pool.py batches, with the private-frames note), Claude helpers, then pass 2 of the
# single-family labels (decide.py queue, sheets.py make --mode pass2)
.venv-ci/bin/python realplus/pool.py final
.venv-ci/bin/python realplus/native.py edge-boxes && .venv-ci/bin/python realplus/pool.py edge-batches
.venv-ci/bin/python realplus/pool.py edge-apply
.venv-ci/bin/python realplus/native.py build --tag pre --margins 4,8
.venv-ci/bin/python realplus/native.py check --tag pre      # blind sheets; a Claude helper decides them
.venv-ci/bin/python realplus/native.py check-score --tag pre
.venv-ci/bin/python realplus/native.py build --tag v1 --margins 4,8
```
