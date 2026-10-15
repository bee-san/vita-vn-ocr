# The owner pool: training lines from the owner's own console captures

The owner's Google Drive drop of 2026-10-13 (`vn-ocr-dataset-20261013.zip`, sorted into
`shared/private/owner-games/` by owner-drive-ingest) holds live PSP and PS Vita framebuffer captures.
The training titles among them are labelled here into a training pool. The pool is for the next
training round (VOCR4), not VOCR3 round 1.

**Private data.** The frames are the owner's copyrighted captures. Frames, readings, sheets, labels,
crops and the npz files stay under `shared/private/owner-games/_build/owner-pool/`. Nothing from them
went to Codex or any other model or third party: the labelling has no Luna vote. The repo holds only the
code and the aggregate counts in `realplus/manifests/owner/stats.json`.

**Version 2** ([below](#version-2-the-2026-10-14-drop-and-the-first-drops-former-test-folders)) labels the
frames no owner pool had read: the owner's 2026-10-14 drop and the first drop's four former test folders, which
the 2026-10-14 13:25Z policy made training data. It lives in `shared/private/owner-games/_build2/owner-pool-2/`;
its aggregate record is `realplus/manifests/owner/stats_v2.json`.

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

`native.py build --margins 4,8` cuts each kept line from its native frame. The npz files use
teacher_train's shard format, in console polarity, with `pool: "owner"` in the meta.

`session` is null in every row, as in the native pool: devnative's `read.py` does not carry it over from
the read input. Every frame's session is `owner/<folder>`, one folder per VN.

| | |
|---|---:|
| **lines** | **595** from 3 VNs |
| per VN | v34 planetarian 323, v14018 CHAOS;CHILD 227, v57 Sharin no Kuni 45 |
| how | 348 auto, 164 both engine families, 83 one family plus pass 2 |
| dropped by the build | 104 repeated name plates (realplus's per-VN repeat cap), 3 edge cuts, 5 with no usable crop |

The crop files:

| file | crops | PSP / Vita | sha256 |
|---|---:|---|---|
| `native_train_ink_v1.npz` (= `_v2`) | 594 | 368 / 226 | `a2f621d910602cf70dbea623d06ca81fcda662fe44333e970f7452710d66a61a` |
| `native_train_grad_v1.npz` (= `_v2`) | 540 | 331 / 209 | `90f1675aef4c9203904329757934e2b4b07a5d46ac71999d9a445d969d476883` |
| `native_train_ink_m4_v2.npz` | 577 | 354 / 223 | `ab124a58b620fe68063aab4b9f2ed666409111331c980125a277267c2fc1a226` |
| `native_train_ink_m8_v2.npz` | 577 | 354 / 223 | `ec70cc3fffddd70b577627d96741c966b36e9a73bc368ca9a867a48e63611011` |

v1 and v2 hold the same lines. v2 rebuilt only the margin files, with the clamp below. Its ink and grad
files are byte-identical to v1's, the files VOCR4 pins.

- **ink**: eval's crop widened to the labelled characters, as in the native pool. One line has no ink crop:
  another detected line enters it.
- **grad**: the console GRAD finder's box, holding the line and nothing else.
- **ink_m4 / ink_m8**: the ink crop grown by 4 or 8 px on every side and clamped to the frame. This is the
  fresh-native harness's multi-margin idea, applied to one line. The variants carry the same labels and
  are optional augmentation.
  - The grown box is then shrunk on the side of every other read line's ink box or detected text box that
    enters it, so none enters by more than 1 px (`native.clamp_margin`).
  - A line whose base crop another box already overlaps gets no variant: 17 lines.
  - All 577 + 577 variants hold at most 1 px of another line's ink box.
  - v1's margin files used only the base crop's 25%-of-area mixed test. Under it, a short last line under
    a long one kept the line above: 6 of v1's 431 `ink_m8` crops held two whole lines, and 21 more held
    7 px or more of a neighbouring line (audit-owner-pool, B1). Those files are renamed
    `*_v1_superseded_neighbour_lines.npz` and must not be used.

`train2.load_shard` loads every row of all four files (0 dropped). `realplus/manifests/owner/stats.json`
is the committed, aggregate-only record: per-folder eligibility, leak pair counts, label and build
counts per VN, file sha256 and the precision summary. It holds no file name, hash, box or text of a
private frame.

### Precision

The check is independent and blind (`native.py check`, `check-score`, `CHECK_STRATA` 40 auto + 120
reviewed, seed 20261013, drawn from the pre-build before v1). Its margin crops came from the pre-build,
before the clamp. The labels it checks are the same in v2. A Claude helper that labelled none of the
lines (review-nf-15) typed the framed text of each crop with no candidate shown. The crop for each line
was drawn at random from the files that hold it, so the margin variants are covered too.

| | agree | rate (Wilson 95%) |
|---|---:|---|
| **all** | **160 / 160** | **100% (97.7-100%)** |
| auto / reviewed | 40 / 40, 120 / 120 | 100% (91.2-100%), 100% (96.9-100%) |
| ink / grad / ink_m4 / ink_m8 | 38 / 38, 39 / 39, 55 / 55, 28 / 28 | |

v1 is the pre-build with the meta's `pool` set to `"owner"`: same lines, crops and labels. v2 is v1 with
the clamped margin files.

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
.venv-ci/bin/python realplus/native.py build --tag v2 --margins 4,8     # ink, grad byte-equal to v1's
```

## Version 2: the 2026-10-14 drop and the first drop's former test folders

Owner pool v2 labels the owner's console frames that no owner pool had read. Under the 2026-10-14 13:25Z policy
([reservations](reservations.md)), only the CLANNAD family, the vita-test and psp-test VNs and their shared releases
are test; every other owner frame trains. The driver is `realplus/owner_pool.py`, around main's `realplus/pool.py`,
`native.py` and devnative's pipeline, with `REALPLUS=NATIVE_POOL=_build2/owner-pool-2` and `NATIVE_OWNER_POOL=1`.

The privacy rule is v1's. Frames, readings, sheets, decisions, labels, crops and npz files stay under
`shared/private/owner-games/_build2/owner-pool-2/`. Nothing went to Codex or any other model or third party.
Every reviewer was a Claude Opus 5.5 helper session (`reviewer_kind` "assistant"). The repo holds the code and the
aggregate `realplus/manifests/owner/stats_v2.json`.

**Names.** This pool's build tag is `v2` too, so its files are `_build2/owner-pool-2/native_train_<crop>_v2.npz`. They
are not owner pool v1's `_build/owner-pool/native_train_<crop>_v2.npz`, which is v1's rebuild with clamped margins.
The merged files (`owner_train_<crop>_v2.npz`, below) hold both pools.

### Frames

| source | VNs | frames read | PSP / Vita |
|---|---:|---:|---|
| the 2026-10-14 drop's new training frames ([owner-games](owner-games.md#the-drive-drop-of-2026-10-14)) | 77 | 3,777 | 2,210 / 1,567 |
| the 2026-10-13 drop's former -TEST folders: Amagami v1281 (113), Narcissu v10 (200), Danganronpa V3 v18334 (84), Iwaihime -Matsuri- v17863 (240) | 4 | 637 | 313 / 324 |
| **all** | **80** | **4,414** | **2,523 / 1,891** |

`owner_frames.py pool-prep --prefix p2f --skip-pool _build/owner-pool` reads every unique, native frame of the
training folders:
- It skips the 577 frames owner pool v1 read, matched by decoded-pixel sha256.
- It refuses 3 frames that `upscale_2x` reads as possible 2x upscales. One is the CHAOS;CHILD frame v1 refused too.
  The other two are new: a Mikagami Sumika frame (smooth) and VA-11 HALL-A's one frame (nearest-neighbour). So 3,777
  of the drop's 3,779 linked new training frames are read.

Iwaihime's 241 frames are the first drop's 240 plus the second drop's 1 new frame. The shots are named
`<vn>_p2f<n>`, apart from v1's `<vn>_owner<n>`.

81 of the 3,777 new frames are title or settings screens by the drop's own categories. The drop's other 6 such frames
are the test games' 5 and VA-11 HALL-A's one refused frame. Their lines are counted apart from dialogue
([below](#dialogue-only-count)).

The four test games of the 2026-10-14 drop (408 frames) were never read, planned or labelled. Neither was CLANNAD.
They serve only as the comparison side of the leak check.

### The labelling rule (`AGREEMENT.md`)

The rule was fixed in the pool's `AGREEMENT.md` (sha256 `0b1ca439760f1a8a22b84239d789d6221bd54fb24d2bd5d478f705f22e313b9a`)
at 2026-10-14T18:14:07Z, before the first read, plan or batch:

1. **auto**: a line whose four readings (`devnative/read.py`: MeikiOCR on its detected line and on the line's
   widened view, PP-OCRv6 small on the view and on the ink crop) give one key is kept without review
   (`decide.py --mode train --auto-consensus`), as in v1.
2. **pass 1**: a Claude helper decides every other candidate line on contact sheets. A label must be read by at
   least one engine family (MeikiOCR or PP-OCRv6 small), else it is dropped (`no-engine-support`).
3. **confirm look**: every label rule 2 keeps needs an independent second Claude read. It comes from another helper
   session, on a blind sheet (`sheets.py` mode blind: engines hidden, candidates shuffled and unattributed, pass 1's
   text not marked) framed on the line's final crop. The line is kept only when that read has pass 1's key
   (`teacher/vote.key`, eval's normalisation, so a missing or extra bracket or 。 is a disagreement). A different
   text, an unsure or a skip drops it, with no third look. A line without a confirm read never enters the pool.
4. **edge check**: a kept line whose final crop `edgecheck` flags needs an edge pick (the frame holds the whole line
   and exactly the label), as in v1.
5. **build**: `native.py build --margins 4,8`, realplus's repeat caps (2 per VN, 8 in all), out-of-vocab lines
   dropped.
6. **precision**: a blind transcription (`native.py check`, no candidates) of 60 auto and 140 reviewed lines from the
   pre-build (seed 20261014), each shown as one of its crops at random, by helpers that labelled none of the lines.
   A wrong label is dropped (`review/resolutions.jsonl`) before the final build.

The task asked for a Luna vote "within the Codex limits". Those limits admit only public VNDB crops, and these frames
are private, so Luna has no vote here. The confirm look (rule 3) takes its place as the independent second vote on
every reviewed label.

### Leak check (`eval/tools/leakcheck.py`, thumbnail correlation >= 0.95)

The 4,414 frames were compared with the three test sets only:

| against | screenshots | pairs >= 0.95 |
|---|---:|---:|
| vita-test + psp-test (eval) | 157 | 1 (0.952, cross-VN) |
| the owner's test frames: CLANNAD (2,206) and the 2026-10-14 drop's four test games (408) | 2,614 | 2 (0.957, cross-VN) |

All three pairs involve one Iwaihime -Matsuri- frame. It is dropped (`manifests/leak_drop.json`): its lines were
never shown on a sheet, and its one auto line is left out of the build (`excluded-late`).

The comparison side's 2,614 hardlinks are kept apart from the pool, in `_build2/leak-test-links/`, so that no tool
globbing the pool directory for images takes them in.

### Reads and plan

`devnative/read.py` read the 4,414 frames offline at native size in 371 s (5 workers, one heavy.sh slot):
MeikiOCR 0.3.5 and PP-OCRv6 small (pinned ONNX), four readings per line. `pool.py plan` gives **10,103 candidate
lines**:
- **4,238 auto**: all four readings agree.
- **5,865 for review**: tier t1 3,275, t2 394, t3 2,196.

One review line sits on the leak-dropped frame, so pass 1 saw 5,864. `native.py known` finds no line that teacher or
realdata labelled on these frames.

### Labelling

The first pass-1 helper was queued at 2026-10-14T18:34Z. From 19:13Z, `owner_pool.py cycle --loop` ran the pipeline
every 600 s until the last batch was decided at 2026-10-15T09:16Z. Each round it ran `final`, `native.py edge-boxes`,
cap-aware confirm and edge batches, and queued helpers (`review-o2-N`, at most 6 at a time). Each batch was decided by
one helper session; none needed a retry.

| stage | Claude sessions | batches, sheets | lines | result |
|---|---:|---|---:|---|
| pass 1 (`review/pass1/`) | 17 | 17, 713 | 5,864 | 4,547 picks and 367 typed: 4,914 labels. Skipped: 670 ui, 223 cut, 53 bad. 4 unsure. 671 flagged as speaker names |
| engine support (rule 2) | | | 4,914 | 366 typed labels no engine read: dropped |
| confirm look (`review/confirm/`) | 10 | 10, 481 | 3,793 | 3,775 have pass 1's key. Dropped: 12 another text, 2 unsure, 2 cut. 2 more looks were at lines pass 1 called cut |
| edge check (`review/edge/`) | 7 | 7, 223 | 2,408 | 2,317 hold the whole line and exactly the label. Dropped: 78 bad, 12 cut, 1 another text |

- **Confirm independence.** No confirm look came from the session that decided the line's pass 1. Agreement with pass
  1 was 3,775 of 3,791 confirm reads of a pass-1 label (99.6%).
- **The 2 extra looks.** They were at lines of pass-1 batch b0009 whose helper changed its decision to cut after the
  confirm batch was made. Pass 1's final decision stands, so both lines are dropped as cut.
- **Lines with no confirm look.** 757 pass-1 labels never got one; 561 of them are speaker names. The batches were
  cap-aware (`confirm_entries`): a copy was held back once its label's kept or batched copies reached the repeat caps,
  as `owner_pool.CapCount` counts them.
  - `CapCount` counts every copy toward the global cap of 8. The build's `pool.cap_repeats` counts a copy toward it
    only after the copy passes the per-VN cap of 2.
  - So a label repeated in one VN held back its copies in other VNs too. Some labels have 40 to 92 kept copies in one
    game.
  - For 687 of the 757, the final build had no room anyway. The other **70** had room: fewer than 2 built lines of
    their label in their VN, and fewer than 8 in all. 69 of the 70 are names.
  - All 757 stay out of the pool (`pending`, `confirm-missing`), as rule 3 says.
- **Lines with no edge look.** The same counting left 221 flagged kept crops without an edge look, and the build drops
  them (`edge-unchecked`). **12** of them had room: all auto lines, of 3 labels.

The final labels (`labels.jsonl`, after the precision check below):

| | lines |
|---|---:|
| **kept** | **8,012** |
| auto (four readings agree) | 4,237 |
| reviewed, both engine families, plus an agreeing confirm look | 2,324 |
| reviewed, one engine family, plus an agreeing confirm look | 1,451 |
| dropped | 1,334: 670 ui, 366 no engine support, 223 cut, 53 bad, 16 by the confirm look, 4 unsure, 1 on the leak-dropped frame, 1 by the precision check |
| pending: no confirm look (copies past the repeat caps) | 757 |

### Pool (`native.py build --tag v2 --margins 4,8`)

The build cuts each kept line from its native frame. It writes teacher_train's shard format in console polarity,
with `pool: "owner"` in the meta and `session` `null`, as in v1. Every frame's session is `owner/<folder>`.

| | |
|---:|---|
| **6,594 lines** | from 71 VNs: 3,817 PSP (32 VNs), 2,777 Vita (39 VNs) |
| how | 3,042 auto, 2,198 both engine families, 1,354 one family; every reviewed line had a confirm look |
| dialogue | **6,236** (below); 181 more are menu lines and 177 speaker names |
| dropped by the build | of the 8,012 kept labels: 958 repeated labels (repeat caps); 312 by the edge check (221 unchecked copies past the caps, 78 bad, 12 cut, 1 another text); 145 with no usable crop; 2 out of vocab; 1 on the leak-dropped frame |

| file (`_build2/owner-pool-2/`) | crops | PSP / Vita | sha256 |
|---|---:|---|---|
| `native_train_ink_v2.npz` | 6,587 | 3,814 / 2,773 | `fc6addb5d170018d5ce2b716f96cf5507c3f87a2180d0f2a426a32c120158466` |
| `native_train_grad_v2.npz` | 5,084 | 2,911 / 2,173 | `c639dc621dc346ce11b01ab2399458e07092e0a37b0c5fb035d0dba7385f2001` |
| `native_train_ink_m4_v2.npz` | 6,064 | 3,514 / 2,550 | `3366d882e92e7ac387deb7de10f7acd293e94ee280924e3ed27ad9c0272aca46` |
| `native_train_ink_m8_v2.npz` | 6,064 | 3,514 / 2,550 | `d698f1b431307f2970f4d673cad919aeeecd7098d85d573331601ec65535e27a` |

The crop kinds are v1's: ink (eval's crop widened to the labelled characters), grad (the console GRAD finder's box
where it holds the line and nothing else), and ink_m4 / ink_m8 (the ink crop grown by 4 or 8 px, clamped by
`native.clamp_margin`).
- A line whose ink crop another box already overlaps gets no margin variant: 523 lines.
- 7 lines have only a grad crop.
- Every margin crop holds at most 1 px of another read line's ink box.
- `train2.load_shard` loads every row of the four files (0 dropped).

`v2` is the pre-build (`pre`, the precision sample's frame) less the one line the check dropped. The other 6,594
lines' rows are identical in pixels, label and meta in all four files.

### Per game

Frames read (of them title or settings screens), candidate lines, kept labels, built lines, menu lines and dialogue
lines per VN (`stats_v2.json` "per_game"). Nine VNs give no line: each has one frame, a title or settings screen.

| VN | title (VNDB) | platform | frames read (menu) | candidate lines | kept labels | built lines | menu lines | dialogue lines |
|---|---|---|---:|---:|---:|---:|---:|---:|
| v2113 | ソルフェージュ | psp | 150 | 705 | 480 | 393 | 0 | 377 |
| v14909 | 銃声とダイヤモンド | psp | 149 (1) | 599 | 465 | 341 | 0 | 323 |
| v10623 | カスタムドライブ | psp | 150 (2) | 465 | 362 | 324 | 6 | 301 |
| v3278 | のーふぇいと！～only the power of will～ | psp | 150 | 299 | 265 | 238 | 0 | 227 |
| v3992 | 涼風のメルト -Where wishes are drawn to each other- | psp | 150 (2) | 327 | 301 | 236 | 4 | 228 |
| v689 | MagusTale ～世界樹と恋する魔法使い～ | psp | 149 (2) | 332 | 269 | 230 | 2 | 226 |
| v10 | narcissu | psp | 200 | 312 | 257 | 222 | 0 | 222 |
| v1322 | 天神乱漫 LUCKY or UNLUCKY!? | psp | 150 (2) | 310 | 279 | 222 | 1 | 208 |
| v1646 | メモリーズオフ6 ～T-wave～ | psp | 150 | 238 | 201 | 179 | 0 | 175 |
| v1281 | アマガミ | psp | 113 | 197 | 177 | 177 | 0 | 177 |
| v7018 | DUNAMIS15 | psp | 150 | 248 | 174 | 173 | 0 | 168 |
| v1171 | Memories Off ～それから～ | psp | 150 | 238 | 234 | 164 | 0 | 162 |
| v7301 | 白衣性恋愛症候群 | psp | 100 | 181 | 166 | 155 | 0 | 151 |
| v1080 | ぱすてるチャイムContinue | psp | 150 (2) | 267 | 238 | 138 | 7 | 131 |
| v264 | D.C.～ダ・カーポ～ | psp | 150 | 275 | 236 | 125 | 0 | 124 |
| v12369 | Fate/EXTRA CCC | psp | 44 (1) | 128 | 104 | 98 | 0 | 97 |
| v16386 | MISSING PARTS the TANTEI stories | psp | 49 | 112 | 90 | 76 | 0 | 72 |
| v7014 | ダンガンロンパ 希望の学園と絶望の高校生 | psp | 38 | 122 | 67 | 67 | 0 | 63 |
| v7602 | フェイト/エクストラ | psp | 24 (2) | 85 | 76 | 61 | 4 | 57 |
| v11713 | Trick x Logic Season 1 | psp | 48 | 98 | 83 | 53 | 0 | 49 |
| v266 | D.C.II ～ダ・カーポII～ | psp | 37 (2) | 53 | 39 | 37 | 2 | 34 |
| v6572 | BLAZBLUE CONTINUUM SHIFT | psp | 32 (2) | 52 | 35 | 35 | 1 | 30 |
| v728 | プリンセスメーカー5 | psp | 10 (2) | 41 | 35 | 27 | 1 | 26 |
| v19010 | 探偵 神宮寺三郎 灰とダイヤモンド | psp | 11 (2) | 33 | 17 | 17 | 1 | 16 |
| v1170 | 想い出にかわる君 ～メモリーズオフ～ | psp | 6 | 13 | 12 | 11 | 0 | 11 |
| v1325 | Memories Off #5 とぎれたフィルム | psp | 3 | 5 | 5 | 5 | 0 | 3 |
| v2063 | 流行り神 警視庁怪異事件ファイル | psp | 2 (2) | 15 | 6 | 5 | 5 | 0 |
| v727 | プリンセスメーカー4 | psp | 2 (2) | 9 | 3 | 3 | 3 | 0 |
| v22505 | CHAOS;HEAD NOAH | psp | 1 | 2 | 2 | 2 | 0 | 2 |
| v3333 | ディスガイア インフィニット | psp | 1 (1) | 4 | 2 | 1 | 1 | 0 |
| v4091 | セカンドノベル ～彼女の夏、15分の記憶～ | psp | 2 | 2 | 1 | 1 | 0 | 1 |
| v4185 | 密室のサクリファイス | psp | 1 (1) | 3 | 1 | 1 | 1 | 0 |
| v21259 | 流行り神2 警視庁怪異事件ファイル | psp | 1 (1) | 6 | 0 | 0 | 0 | 0 |
| v17863 | 祝姫 | vita | 241 (1) | 408 | 340 | 304 | 0 | 302 |
| v7803 | AMNESIA | vita | 119 | 371 | 369 | 279 | 0 | 279 |
| v10680 | 月に寄りそう乙女の作法 | vita | 116 (1) | 505 | 429 | 272 | 0 | 264 |
| v16610 | 白衣性愛情依存症 | vita | 150 | 363 | 297 | 250 | 0 | 246 |
| v2002 | STEINS;GATE | vita | 150 | 359 | 185 | 183 | 0 | 181 |
| v17285 | 帝国海軍恋慕情 ～明治横須賀行進曲～ | vita | 88 (2) | 207 | 192 | 151 | 7 | 142 |
| v15773 | 百華夜光 | vita | 79 (2) | 166 | 155 | 143 | 7 | 133 |
| v17936 | 東京喰種 JAIL | vita | 120 (2) | 176 | 141 | 140 | 1 | 139 |
| v18334 | NEWダンガンロンパV3 | vita | 84 | 179 | 121 | 121 | 0 | 118 |
| v6618 | STEINS;GATE 比翼恋理のだーりん | vita | 104 | 283 | 120 | 120 | 0 | 112 |
| v19743 | Side Kicks! | vita | 63 (2) | 122 | 110 | 110 | 11 | 97 |
| v7809 | 極限脱出ADV 善人シボウデス | vita | 102 (1) | 203 | 193 | 100 | 2 | 98 |
| v11108 | Princess Arthur | vita | 53 (2) | 123 | 107 | 86 | 10 | 74 |
| v7833 | OZMAFIA!! | vita | 76 (1) | 88 | 79 | 79 | 0 | 76 |
| v5883 | ROBOTICS;NOTES | vita | 86 (2) | 126 | 66 | 66 | 10 | 56 |
| v11173 | さくら荘のペットな彼女 | vita | 38 (1) | 86 | 65 | 65 | 5 | 56 |
| v22174 | 見鏡澄香の制服活動 | vita | 48 | 83 | 74 | 64 | 0 | 64 |
| v17102 | STEINS;GATE 0 | vita | 48 | 67 | 57 | 49 | 0 | 47 |
| v19100 | クロスアンジュ 天使と竜の輪舞tr. | vita | 13 | 38 | 37 | 29 | 0 | 29 |
| v13059 | XBLAZE CODE:EMBRYO | vita | 33 (2) | 60 | 37 | 27 | 1 | 19 |
| v11660 | STEINS;GATE 線形拘束のフェノグラム | vita | 27 (1) | 68 | 27 | 24 | 0 | 20 |
| v15231 | 逆転 吉原 | vita | 1 (1) | 27 | 19 | 19 | 19 | 0 |
| v17902 | クロガネ回姫譚 -一期一会- | vita | 2 (2) | 39 | 25 | 15 | 15 | 0 |
| v19851 | 蝶々事件ラブソディック | vita | 4 (2) | 31 | 15 | 14 | 11 | 3 |
| v12630 | やはりゲームでも俺の青春ラブコメはまちがっている。 | vita | 8 | 17 | 14 | 13 | 0 | 9 |
| v21899 | あざやかな彩りの中で、君らしく | vita | 2 (2) | 21 | 11 | 11 | 11 | 0 |
| v24957 | レイギガント | vita | 2 (2) | 21 | 12 | 11 | 11 | 0 |
| v3091 | カオスヘッド らぶChu☆Chu! | vita | 10 (1) | 29 | 10 | 10 | 1 | 9 |
| v10561 | 真かまいたちの夜 11人目の訪問者（サスペクト） | vita | 1 (1) | 7 | 5 | 5 | 5 | 0 |
| v2923 | Cure Mate Club | vita | 1 (1) | 5 | 4 | 3 | 3 | 0 |
| v2954 | Bitter Smile. | vita | 1 (1) | 5 | 2 | 2 | 2 | 0 |
| v4540 | 猛獣使いと王子様 | vita | 2 (2) | 8 | 2 | 2 | 2 | 0 |
| v5386 | Present for you ～わたしをあ・げ・る～ | vita | 1 (1) | 4 | 2 | 2 | 2 | 0 |
| v13865 | IS＜インフィニット・ストラトス＞2 イグニッション・ハーツ | vita | 2 (2) | 10 | 2 | 2 | 2 | 0 |
| v19441 | プラスティック・メモリーズ | vita | 3 (1) | 3 | 2 | 2 | 0 | 2 |
| v7967 | ホチキス | vita | 1 (1) | 3 | 1 | 1 | 1 | 0 |
| v10271 | 彼女はオレからはなれない | vita | 1 (1) | 4 | 1 | 1 | 1 | 0 |
| v10641 | キスベル | vita | 1 (1) | 4 | 1 | 1 | 1 | 0 |
| v16467 | アブナイ恋の捜査室 ~Eternal Happiness~ | vita | 2 (2) | 3 | 1 | 1 | 1 | 0 |
| v683 | パルフェ ～ショコラ second brew～ | vita | 1 (1) | 1 | 0 | 0 | 0 | 0 |
| v11677 | 花咲くまにまに | vita | 1 (1) | 0 | 0 | 0 | 0 | 0 |
| v14787 | Binary Star | vita | 1 (1) | 1 | 0 | 0 | 0 | 0 |
| v14919 | レンドフルール | vita | 1 (1) | 0 | 0 | 0 | 0 | 0 |
| v16468 | Vamwolf Cross† | vita | 1 (1) | 0 | 0 | 0 | 0 | 0 |
| v16607 | 5人の恋プリンス ~ヒミツの契約結婚~ | vita | 1 (1) | 1 | 0 | 0 | 0 | 0 |
| v18777 | イケメン幕末◆運命の恋 | vita | 1 (1) | 1 | 0 | 0 | 0 | 0 |
| v19119 | Fate/EXTELLA | vita | 1 (1) | 1 | 0 | 0 | 0 | 0 |
| **all** | | | **4,414 (81)** | **10,103** | **8,012** | **6,594** | **181** | **6,236** |

### Dialogue-only count

**6,236 of the 6,594 lines are dialogue**: 3,661 PSP, 2,575 Vita. The other 358 are menu lines (181) or speaker names
(177), with no line in both.

**Menu lines** (`owner_pool.MENU_RULE`). 181 lines (39 PSP, 142 Vita; 39 VNs) come from 81 of the frames read. The
drop's own manifest (`owner-games/<folder>/manifest_20261014.jsonl`, "category") calls those frames title or
settings screens: 52 Vita, 29 PSP.
- Pass 1 kept no reviewed line on them: it skipped 284 as ui and 6 as bad.
- All 181 lines are auto lines, kept by rule 1 because all four readings agreed.
- They are real console text with correct labels, so they stay in the pool. But they are menus, not dialogue.
- `manifests/menu_lines_v2.jsonl` in the pool lists their ids, for a trainer that wants to leave them out.

The first drop's 637 frames carry no category, so a menu frame among them would not be counted here.

**Speaker names** (`owner_pool.NAME_RULE`). A name is one of these:
- a line that a Claude look flagged (`"name": true` in pass 1, a confirm look, an edge check or the precision check);
- a line whose label has the key of a flagged line of the same VN;
- a label that is one `【…】` bracket, the review conventions' 【name】.

84 of the 177 carry pass 1's own flag, which is the npz meta's `name`. The other 93 are auto lines.

An auto name plate without brackets that no look ever flagged in its game still counts as dialogue. So 6,236 is an
upper bound. The repeat caps keep any one name to 2 lines per VN.

### Precision (`native.py check`, `check-score`)

The check followed rule 6, pre-registered in `AGREEMENT.md`: 60 auto and 140 reviewed lines, seed 20261014, drawn
once from the pre-build (`review/check_spec_pre.json`). Each line was shown as one of the crops the npz holds for it,
at random, on blind sheets with no candidate. Two fresh Claude sessions typed the framed text (review-o2-35: 100 lines,
12 sheets; review-o2-36: 100 lines, 11 sheets). Neither had labelled any line.

| | agree | rate (Wilson 95%) |
|---|---:|---|
| **all** | **199 / 200** | **99.5% (97.2-99.9%)** |
| auto / reviewed | 59 / 60, 140 / 140 | 98.3% (91.1-99.7%), 100% (97.3-100%) |
| ink / grad / ink_m4 / ink_m8 | 68 / 68, 40 / 41, 48 / 48, 43 / 43 | |

**The one disagreement.** It is an auto line of DUNAMIS15 (v7018), shown as its grad crop. All four readings gave
katakana ヘ where the line has the particle へ: `…ジェクトへシフトできる見込みです」`. The two glyphs look the same in
the game's font; the grammar settles it. The checker was right and the label wrong. The third look recorded that in
`review/resolutions.jsonl`, and `final` and the v2 build drop the line (`dropped:blind-wrong-label`).

A scan of the built lines for the same confusion, ヘ or へ next to katakana, found 16 more. They read correctly:
へ in `カバンへ`, `前ページへ`, `へー` and the like, ヘ in `ロングヘア` and `ヘン`.

### One set of files (`owner_pool.py merge --tag v2 --old _build/owner-pool:v2`)

`owner_train_<crop>_v2.npz` in the v2 pool holds owner pool v1's v2 files (595 lines; v1's clamped margins) and this
pool's v2 lines under one global repeat cap of 8 per label. Each row's meta carries `owner_pool` ("owner-pool" or
"owner-pool-2"). The two pools share no VN, so no line id is in both.

| file | crops | v1 / v2 | PSP / Vita | sha256 |
|---|---:|---|---|---|
| `owner_train_ink_v2.npz` | 7,179 | 594 / 6,585 | 4,182 / 2,997 | `b877056375ff4d5275414c740be2f68925de6778643818aa721354d44fc08679` |
| `owner_train_grad_v2.npz` | 5,624 | 540 / 5,084 | 3,242 / 2,382 | `a42db74342c459c732e1720e436869f2387efa2a132be75c8afaed29d28d343e` |
| `owner_train_ink_m4_v2.npz` | 6,639 | 577 / 6,062 | 3,868 / 2,771 | `659f0c82c69049ef9d0244107b7be7caff0bc9319d4d87e57a9c417e2cf9dc9a` |
| `owner_train_ink_m8_v2.npz` | 6,639 | 577 / 6,062 | 3,868 / 2,771 | `fcea192e6b248ce794d0878aab2f2533e94ea7c3d3bb13975ce61e3904ac696e` |

The global cap removes 2 v2 lines, labels that already had 8 lines across the two pools. Neither line has a grad
crop, so no line is left in one crop kind and missing from another. `train2.load_shard` loads every row (0 dropped).

**v1's frames against the new test frames.** v1's 577 frames were not leak-checked against this drop's test games
when v2 was built. `leakcheck.py` on them against the 2,614 owner test frames gives 7 pairs at 0.950-0.952, all
cross-VN, on 3 planetarian frames: 5 against Utawarerumono test frames and 2 against one CLANNAD frame. That is 0 of
v1's frames against vita-test + psp-test.
- In audit-owner-pool-2's look they are white-out scenes with a dialogue box, matching only in layout. v1's audit
  accepted the CLANNAD pair the same way.
- v2's own rule would drop frames like these. The merged files keep v1's 2 lines on them (8 rows), under v1's
  decision.

**Disclosure.** As with v1, the merged files hold v1's planetarian lines (v34, owner decision). Any CLANNAD score of
a model trained on them must disclose that exposure.

### Hygiene

`eval/tools/reserve_check.py --dump <2026-10-13 dump> --pixels --require-hashes --shots manifests/frames.jsonl` ran
over the four v2 files, the four merged files, the line manifest, the frame manifest and `labels.jsonl`. Every row
was hash-checked, decoded pixels included, against 4,726 protected images. Every file returned **RESERVE_OK**, with
no blocker.

Reported for information (training data under the policy), in the ink file:
- 1,017 rows of 7 VNs that were development VNs before 2026-10-14;
- 341 rows of one former fresh-native title;
- 733 rows of 8 VNs of the old `is_test_vn` split;
- 1,192 rows of 8 VNs of the old val bucket.

The counts are in `stats_v2.json` "hygiene". No VN of the development sample (`reserved.json` "dev_sample", re-carved
in #84) is in the pool. No frame of a test VN was read, and the test games' frames appear only on the leak check's
comparison side.

### Limits

- **Menu lines.** The pool holds 181 auto lines of title and settings screens (above). They are left in as console
  text and are excluded from the dialogue count; `manifests/menu_lines_v2.jsonl` lists them.
- **Names.** The dialogue count misses an unbracketed auto name plate that no look flagged; see above.
- **Look-alike characters.** The engines agreed on a wrong ヘ in one of 60 sampled auto lines. Other identical-glyph
  pairs (ロ/口, エ/工, ニ/二, カ/力, ー/一) were not scanned. The auto stratum's precision bound (91.1-99.7%) covers
  them only as a rate.
- **Repeat copies.** 757 reviewed labels and 221 flagged crops got no second look and are not in the pool. Most could
  not have entered the build. But `CapCount`'s stricter global count held back 82 that had room: 70 labels, 69 of them
  names, and 12 auto crops of 3 labels. That is at most 82 lines lost. A later pool should count like `cap_repeats`.
- **Neighbour slivers.** Tightly spaced lines (D.C., Pastel Chime and other PSP text) leave part of the next line in
  the crop.
  - 507 of the 6,587 ink crops (299 PSP, 208 Vita) hold another read line's ink box by 2 px or more where they
    overlap.
  - 226 of them hold at least a quarter of that line's ink height, and at most a half. None shows a whole second line.
  - This is eval's and v1's ink-crop convention. None of these 507 lines has a margin variant.
- **Skew.** PSP gives 58% of the lines. Five large games (Solfege, Juusei to Diamond, Custom Drive, Iwaihime, AMNESIA)
  give a quarter of them.
- **Lossless.** Like v1, the frames are lossless PNG framebuffers, without VNDB's JPEG artefacts.

### Reproduce (version 2)

```sh
B2=$HERD/shared/private/owner-games/_build2; OP2=$B2/owner-pool-2
OWNER_BUILD=$B2 .venv-ci/bin/python realplus/owner_frames.py inventory           # -> $B2/inventory.jsonl
OWNER_BUILD=$B2 .venv-ci/bin/python realplus/owner_frames.py pool-prep --pool $OP2 --prefix p2f \
    --skip-pool $HERD/shared/private/owner-games/_build/owner-pool                # 4,414 frames
HEAVY_SLOTS=12 $HERD/heavy.sh timeout 7200 unshare -rn teacher/.venv/bin/python devnative/read.py run \
    --shots $OP2/reads/shots.jsonl --out $OP2/reads --procs 5
export REALPLUS=$OP2 NATIVE_POOL=$OP2 NATIVE_OWNER_POOL=1
$OP2/leak.sh                        # leakcheck.py against eval's test screenshots and $B2/leak-test-links
.venv-ci/bin/python realplus/owner_pool.py leak-drop $OP2/logs/leak_test_sets.log $OP2/logs/leak_owner_test.log
.venv-ci/bin/python realplus/pool.py plan
.venv-ci/bin/python realplus/native.py known && .venv-ci/bin/python realplus/native.py filter-plan
.venv-ci/bin/python realplus/owner_pool.py batches --n 1 --size 280 && \
    .venv-ci/bin/python realplus/owner_pool.py batches --n 16 --size 360     # pass 1: 17 batches, 5,864 lines
.venv-ci/bin/python realplus/owner_pool.py cycle --loop --every 600 --size 400 --min 300
.venv-ci/bin/python realplus/pool.py edge-apply
.venv-ci/bin/python realplus/native.py build --tag pre --margins 4,8
.venv-ci/bin/python realplus/native.py check --tag pre --strata auto:60,reviewed:140 --seed 20261014 --size 100
.venv-ci/bin/python realplus/owner_pool.py helpers --kinds check --retry
.venv-ci/bin/python realplus/native.py check-score --tag pre
# review/resolutions.jsonl: the disagreement's third look; then final, edge-boxes and edge-apply again
.venv-ci/bin/python realplus/owner_pool.py final
.venv-ci/bin/python realplus/native.py edge-boxes && .venv-ci/bin/python realplus/pool.py edge-apply
.venv-ci/bin/python realplus/native.py build --tag v2 --margins 4,8
.venv-ci/bin/python realplus/owner_pool.py merge --tag v2 --old $HERD/shared/private/owner-games/_build/owner-pool:v2
DB=$HERD/shared/artifacts/native-frames/vndb/dump/db      # the pools first: --shots takes every argument after it
.venv-ci/bin/python eval/tools/reserve_check.py owner2_ink_v2=$OP2/native_train_ink_v2.npz ... \
    --dump $DB --pixels --require-hashes --shots $OP2/manifests/frames.jsonl  # every v2 and merged file, lines_v2, frames, labels
.venv-ci/bin/python realplus/owner_pool.py publish --tag v2 --check pre   # realplus/manifests/owner/stats_v2.json
```
