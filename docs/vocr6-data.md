# VOCR6 data: the owner's hooked PS Vita and PSP corpus

Issue #125, track T0 of the VOCR6 plan. This document covers the owner's hooked corpus: how it was imported into
this repository's pool formats, audited and split, and how the released VOCR5 models score on it. Those scores are
VOCR6's baseline. Every number here was measured on the host. Nothing ran on a console.

**Summary**

- **The corpus.** 49 titles (33 Vita, 16 PSP) with 55,493 native pages. It holds 99,071 machine-verified line
  records: 88,508 horizontal dialogue and narration lines, 1,611 vertical columns, 8,535 speaker names and 417 ruby
  readings. Each label is the game's own hooked text (below).
- **After cleaning:** 81,977 distinct dialogue lines (1,293,093 characters; 53,541 Vita, 28,436 PSP), 1,567
  vertical columns, 679 distinct names and 302 ruby readings.
- **Label audit.** I checked 320 lines by eye. 5 were wrong, all in two systematic classes. The importer now removes
  both classes (15 records). The 318 audited lines that remain are all right (0 of 318; the 95% upper bound is 1.2%).
- **Splits** (pages):
  - train: 50,272.
  - hval: 2,193 (3.95%), for screening only. Final models train on it.
  - guard: 2,338, the bands around hval.
  - test: 690 (1.24%), the test title Dance with Devils.
- **Baseline: VOCR5-N as released**, with the adopted head gate and two-level head (#120, #122), through Rubi's
  LM, counter re-decode and cascade. Rubi itself still pins v0.3.0's plain blobs, without the gate and the two-level
  head, whose dev CER is within +0.05 points of the plain blobs. On hval:
  - Clean line crops: 0.36% dialogue CER.
  - Whole dialogue through the console's finder (GRAD + refine): 5.18%. Without the one title whose dotted box border
    the finder reads as text, 2.57%.
  - The textspot3 spotter on whole frames: 4.25%.

  So the page path holds almost all the error; the recognizer holds very little.
- **Training throughput, CPU only.** This host has no GPU. A Normal fine-tune runs at 1.76 steps/s in one 5-thread
  slot on a partly loaded box (1.58 slot-hours per 10,000 steps), and at 0.90 when all 15 heavy slots are busy
  (3.10 slot-hours). Data loading is 2-5% of a step.

Everything downloaded is the owner's private, copyrighted game data. Pages, crops and texts stay under
`$HERD/shared/private/hooked-b2` and `$HERD/shared/artifacts/vocr6-ingest`. Nothing of it is sent to a third party.
The repository holds code, ids, hashes, geometry, counts and scores only.

## The corpus

The data comes from the private Backblaze B2 bucket `data-from-psp-and-vita`. The owner's capture farm filled it:
PPSSPP and Vita3K capture builds, with the text from a game-memory hook and the boxes from renderer geometry. An AI
reviewer approved every page.

- **The bucket.** On 2026-10-26 at 08:29Z it held 424 objects, 15,982,345,095 B. It was synced once, read only, into
  `$HERD/shared/private/hooked-b2/`.
- **Verification.** Every file each title's `metadata.json` pins matches its bytes and sha256: 370 of 370 files.
  `data/manifests/vocr6_hooked/index.json` `sources` records that check, with the sha256 of the five unpinned
  files (README.md, index.json and the three font-pack files).
- **More titles than index.json.** The bucket's `index.json` (04:36Z) lists 41 titles. The bucket held 49 by the
  time of the sync: 8 PSP titles were uploaded 05:27-08:24Z, and 5 of them say `in_progress`. One of those,
  NPJH50127, has 58 pages and no verified line yet.
- **Versions.** v1 is a snapshot of the sync. Later uploads and fixes go into a new version (`pools/v2`); v1 is
  frozen and read-only.

How a line is verified (the bucket's README):

- `glyph_cells` (86,096 records). The renderer's glyph quads are grouped into character cells. The hook text is
  assigned to a line only when every line's cell count equals its text exactly and the assignment is unique.
- `image_rows` (12,975 records). These are titles whose text is drawn on the CPU, or pages without a unique
  assignment. A row needs both of these:
  - its pitch-grid cell count equals the label's length;
  - a MeikiOCR read of the exact crop agrees with the hook text: at most 1 error per 10 characters, the first and
    last characters read.

  The OCR is a check only; its text is never stored.
- Neither path is a person reading the line: every record says `hand_checked: false`. This is a machine-verified
  farm source, not a hand-checked cohort. Hence the audit below.

## Import (`data/import_hooked.py`)

`extract` unpacks every title's pages and checks each PNG's sha256 against its page record and against every line
that cites it.

`lines` cuts each record into one or more pools:

| pool | what | crop |
|---|---|---|
| `line` | horizontal dialogue and narration | box plus the record's pad |
| `name` | speaker names drawn as glyphs in a nameplate | box plus pad |
| `ruby` | furigana readings, as lines of their own | box (pad 0) |
| `furimain` | lines with furigana, cut at `box_main` (the main text alone) | box_main plus pad |
| `vertical` | tategaki columns turned 90° anticlockwise, as VOCR5-V reads them | box plus pad, rotated |
| `vreflow` | the same columns re-flowed left to right from their `cell_boxes`, the label unchanged | each cell cut at the midpoints between neighbouring cells, 2 px apart |
| `line_x2`, `name_x2` | PSP lines and names cut from the page upscaled 2x bilinear (cv2): the Adrenaline view Rubi reads on a Vita | the native crop rectangle x2 |

Every pool uses the shard format of `train/train2.py`: `data`, `offsets`, `shapes`, `labels` and `meta`, one file
per split, pool and title.

- **Crops** are integer luma (`synth/common.to_gray`) at native size. Text that `synth/common.line_polarity` calls
  dark is inverted, so crops are stored as the console hands them over, the same as VOCR5's real pools
  (`meta.inverted`).
- **Labels** are `teacher/vote.canon` of the record's `text`, which the farm already writes in that form. Canon
  changed none of the 99,071 labels.
- **Meta** carries the provenance: title and VN, platform, page, image sha256, box, pad, crop box, source and its
  check, polarity, the per-character boxes (when the cells spell the label), cluster and duplicate status, and
  whether the label fits Small's vocab.

Filters, in this order (counts per record kind; the generated table below):

1. **Label in doubt** (`label_doubt`, 10 records): an `image_rows` label whose first character the crop may not
   hold. The audit found this class.
2. **Outside the vocab.** The label has a character that `data/vocab_vn_5631.txt` lacks: 84 lines and 8 ruby
   readings. That vocab is the Normal and Large one, with its 5,142 kanji. These records go to `drop/oov` and are
   kept for the synthesis track.
   - Small's 4,987-class vocab has an `<unk>` class. Its trainer maps a missing character there, so nothing is
     dropped for Small.
   - `meta.s_vocab` records whether a label fits Small's vocab: 99.7% or more of the rows of every title do.
3. **Geometry**: a crop side under 8 px (`data/import_native.py`'s floor). This removed 48 lines and 25 ruby
   readings.
4. **CTC feasibility** (`synth/render.crop_ctc_feasible`): a label that needs more stride-8 frames than its crop
   gives at height 32. It removed nothing.
5. **Duplicates**, within a title and pool:
   - *exact*: the same label and identical pixels;
   - *near*: the same label, crop height and width within 2 px, and a mean absolute grey difference of at most 8
     between the crops' 16-px-tall signatures.

   The threshold was set by eye on same-label pairs sampled across the whole range (0-13). Every pair showed the
   same rendered text, differing in its translucent backdrop or by a 1-px shift. 8 keeps the cut conservative, so
   some near-identical crops above it stay as separate lines.

   A cluster's representative is its first train copy if it has one. Derived pools (`line_x2`, `furimain`,
   `vreflow`) follow their base pool's decision. The exception: 17 `line_x2` rows whose native copy failed the
   8-px floor are their own representatives. Identical pixels under two labels are a conflict, and every member
   is dropped: 22 one-character name records of PCSG00469, one identical nameplate crop that the hook labels with
   two different names (20 and 2 records).

What survived each filter:

<!-- generated:filters -->
| kind | records | label in doubt | outside the vocab | under 8 px | CTC-infeasible | duplicates (exact / near) | pixel conflicts | hval copies of a guard line | usable (all splits) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dialogue line | 88,508 | 10 | 84 | 48 | 0 | 3,766 / 2,598 | 0 | 25 | 81,977 |
| vertical column | 1,611 | 0 | 0 | 0 | 0 | 3 / 41 | 0 | 0 | 1,567 |
| speaker name | 8,535 | 0 | 0 | 0 | 0 | 3,556 / 4,274 | 22 | 4 | 679 |
| ruby reading | 417 | 0 | 8 | 25 | 0 | 73 / 8 | 0 | 1 | 302 |
<!-- /generated:filters -->

Each row sums to its record count. "hval copies of a guard line" are hval lines whose duplicate cluster is
represented by a guard line: screening reads them, one per cluster, and a final model reads the guard copy.

Most speaker names are duplicates: one nameplate repeats hundreds of times. 679 distinct name crops remain, and the
names on every page stay in the page pool.

`pages` writes the page-level pool, one JSONL per title. It is for finder and spotter training and for page-path
scoring. A page row holds:

- its split, its renderer boxes and its verified lines (box, `box_main`, text, kind, cells, drop reason);
- `complete`, true when every renderer box belongs to a verified record and no label is in doubt (32,206 of
  55,493 pages);
- the whole-dialogue regions of `eval/finder_tighten.py`'s rule: verified horizontal lines grouped by eval's
  vertical-gap rule, their union plus 12 px (Vita) or 6 px (PSP), and +4 / +8 px variants. Every other box is kept
  out: names, unverified rows, columns, a furigana band that holds a line. A region whose text such a box overlaps
  is dropped.

Pages with regions: 48,384 train, 2,267 guard, 2,097 hval and 685 test. The other pages have no verified horizontal
dialogue line.

`manifests` writes `data/manifests/vocr6_hooked/`:

- one JSONL per title: each page's id, sha256, split, completeness, and each line record's outcome as a short code
  (`line:train`, `line:dup-near`, `line:o` for outside the vocab, `d` for a label in doubt, and so on);
- `index.json`: the split rule and counts, every pool file's sha256 and row count, and the import's counts.

No hooked text is committed. `tests/test_import_hooked.py` checks that on a synthetic mirror.

## Label audit (`data/hooked_audit.py sample / sheets / tally`)

**The sample.** 320 lines from 48 titles, with quotas: 100 `image_rows` (the weaker check), 40 furigana lines, 40
names, 20 ruby readings, 20 vertical columns, and 100 random `glyph_cells` lines spread round-robin over the
titles. It was drawn before the fixes, from the first import.

**The method.**

- 16 contact sheets of 20 lines each (`$HERD/shared/artifacts/vocr6-ingest/audit/sheets`, kept local): each crop
  in screen polarity at 28 px tall, its label under it in BIZ UDGothic.
- I (Claude, reviewer_kind assistant) read every sheet. I zoomed in, or looked at the page around the box, where
  unsure.
- The verdicts are in `audit/decisions.jsonl` and tallied in `data/manifests/vocr6_hooked/label_audit_v1.json`.

<!-- generated:audit -->
| label audit stratum | lines | ok | wrong | wrong-label rate, 95% interval (Wilson) |
|---|---:|---:|---:|---|
| glyph_cells | 100 | 100 | 0 | 0.0-3.7% |
| image_rows | 100 | 98 | 2 | 0.5-7.0% |
| furigana | 40 | 37 | 3 | 2.6-19.9% |
| name | 40 | 40 | 0 | 0.0-8.8% |
| ruby | 20 | 20 | 0 | 0.0-16.1% |
| vertical | 20 | 20 | 0 | 0.0-16.1% |
| platform:vita | 202 | 197 | 5 | 1.1-5.7% |
| platform:psp | 118 | 118 | 0 | 0.0-3.2% |
| all | 320 | 315 | 5 | 0.7-3.6% |
| after the fixes (pools v1) | 318 | 318 | 0 | 0.0-1.2% |
<!-- /generated:audit -->

The two wrong classes, both now handled by the importer:

1. **A bracket outside the crop** (2 audited lines, 10 records corpus-wide). An `image_rows` row whose grid count
   needed a lead cell outside its cited cells (`checks.count.uncited_lead_cells`) and whose label opens with 「: the
   bracket is drawn left of the row's box, so the crop lacks the label's first character.
   - 9 records corpus-wide, in PCSG00540, PCSG00628, NPJH50001 and NPJH50279.
   - One more `image_rows` label opens with a closing 』, which ends the row above (PCSG00751).
   - All 10 are dropped (`label_doubt`), and their pages count as in doubt.
2. **A tall furigana band** (3 audited lines, 5 records, all in the test title PCSG00807). The record marks the line
   as having furigana, but the band above `box_main` is 1.4-1.6x the text height and holds a short dialogue line
   that the label does not contain. Real ruby bands are 0.19-0.63x the text height in every other title.
   - A band at least 0.9x the text height is now cut off: the crop is `box_main`.
   - The page gets no page-path region.

The other strata were clean.

- **Readings that only looked wrong.** Outlined 「」 look hollow, like 『』, at contact-sheet size. Zooms showed the
  label right each time, and MeikiOCR had agreed.
- **The image_rows check.** Labels with an unread edge bracket (`edge_brackets_unread`) still hold the bracket. I
  checked the 51 such records without an uncited lead cell by their start and end windows; the only exception is
  the label that opens with 』 (class 1).
- **After the fixes.** Of the 320 audited records, 315 are in pools v1 with the same crop and label, 3 are re-cut
  and right (viewed again), and 2 are dropped.

## Splits (`data/import_hooked.py split`, `splits_v1.json`)

The split follows `prompts/vocr6-common.md` (owner, 2026-10-26). Every `train` title trains, and the only test
title is the one marked `test`: Dance with Devils (PCSG00807). CLANNAD is not in the corpus. No VNDB reservation
rule is re-applied to this data.

- **hval** gives within-title screening. A title of at least 200 pages has H = round(0.04 x pages) hval pages in
  k = ceil(H / 40) contiguous blocks of near-equal length.
  - The title is cut into k equal segments, and each block sits in its own segment.
  - The block's offset comes from `default_rng(sha256("20261026:" + title id))`. It keeps a guard band of 15 pages
    on either side inside the segment.
  - Pages are in capture order, so a block is a stretch of play, and the guard bands keep neighbouring
    near-duplicate frames out of training while hval is screened.
- **hval_fold.** hval lines that are their duplicate cluster's representative: what a final model adds when hval
  is folded back in.
- **The hval screening set** (`hval/`). hval lines whose cluster has no train copy, one per cluster. Of the 3,170,
  none duplicates a train line, so clean-crop screening never scores a line the model trained on.
  - 81 of them have the same label as a line of the title's train pool. Their crops differ by more than the
    near-duplicate threshold, but the audit found visual duplicates among them, and a model may have learnt the
    text. Normal reads the screening set without them at 0.357% (0.364% with them).
- **The hval page path** reads every hval page, so it is not deduplicated against train.
  - 118 of its 2,109 base regions hold one of the 128 hval lines whose duplicate cluster has a train copy: 916 of
    the 54,157 reference characters (1.7%).
  - 9 hval pages are byte-identical to a train or guard page, and 141 have the same page text as one.
  - The results file scores the page path on the regions without such a line too (`refine|console|base|unseen`):
    Normal reads them at 4.76%, against 5.18% for all regions. VOCR6's page-path screening should use that key, and
    finder or spotter work should also leave out hval pages identical to a train page.
- **The screening share.** While screening, hval + guard + test hold out 9.41% of the pages. When CLANNAD's hooked
  pages arrive as a test title, that share will pass 10%. Final models hold out the test titles only: 1.24% plus
  CLANNAD.
- **guard.** Out of screening runs, back in final models. Its lines are deduplicated against train.
- **Held out for good:** 690 test pages, 1.24% of all pages; the owner's cap is 10%. hval and guard are out only
  while screening.

<!-- generated:splits -->
| rows | train | guard | hval | hval_fold | test |
|---|---:|---:|---:|---:|---:|
| pages | 50,272 | 2,338 | 2,193 | | 690 |
| line | 74,274 | 3,406 | 3,170 | 3,145 | 1,152 |
| name | 647 | 22 | 14 | 10 | 0 |
| ruby | 273 | 19 | 11 | 10 | 0 |
| vertical | 1,461 | 56 | 50 | 50 | 0 |
| vreflow | 1,408 | 56 | 50 | 50 | 0 |
| furimain | 231 | 22 | 14 | 13 | 0 |
| line_x2 | 26,313 | 1,063 | 1,096 | 1,077 | 0 |
| name_x2 | 302 | 13 | 7 | 5 | 0 |
<!-- /generated:splits -->

## Titles, coverage and what VOCR5 has seen (`hooked_audit.py stats / overlap`)

Per title (`stats_v1.json`, `overlap_v1.json`). Text height is the median height of the line boxes in native pixels.
"seen" marks a title whose VN is in the protected family, in VOCR5's development sample, or in VOCR5's fine-tune
pools, with that pool's row count:

<!-- generated:titles -->
| title | platform | split | VN | pages (hval) | records | usable dialogue lines | dialogue chars | duplicates removed | text height (px, median) | dark-on-light | image_rows share | seen |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| NPJG00016 | psp | train | v14909 | 271 (11) | 566 | 446 | 3,660 | 119 | 18 | 0.2% | 17% | VOCR5 pools (1,224 rows) |
| NPJH00140 | psp | train | v7301 | 1,640 (66) | 3,425 | 2,885 | 38,165 | 540 | 16 | 0.0% | 100% | VOCR5 pools (762 rows) |
| NPJH50001 | psp | train | v2063 | 1,602 (64) | 2,913 | 1,418 | 24,932 | 1,475 | 19 | 0.0% | 1% | VOCR5 pools (35 rows) |
| NPJH50002 | psp | train | v21259 | 1,727 (69) | 2,660 | 2,174 | 36,398 | 442 | 19 | 0.0% | 0% |  |
| NPJH50127 | psp | train | v3360 | 58 (0) | 0 | 0 | 0 | 0 |  |  | 0% | VOCR5 pools (8 rows) |
| NPJH50132 | psp | train | v1171 | 1,851 (74) | 2,641 | 2,044 | 35,257 | 557 | 16 | 88.1% | 0% | VOCR5 pools (813 rows) |
| NPJH50279 | psp | train | v4091 | 1,004 (40) | 1,540 | 1,243 | 20,624 | 275 | 19 | 1.5% | 100% | VOCR5 pools (39 rows) |
| NPJH50297 | psp | train | v1322 | 1,177 (47) | 2,495 | 1,872 | 29,370 | 623 | 14 | 0.0% | 100% | VOCR5 pools (1,064 rows) |
| NPJH50356 | psp | train | v398 | 1,999 (80) | 2,597 | 2,530 | 36,182 | 64 | 18 | 0.1% | 0% | VOCR5 pools (46 rows) |
| NPJH50382 | psp | train | v1646 | 1,965 (79) | 3,230 | 2,254 | 37,230 | 840 | 20 | 0.0% | 0% | VOCR5 pools (840 rows) |
| NPJH50551 | psp | train | v22505 | 214 (9) | 369 | 358 | 7,289 | 5 | 21 | 0.0% | 2% | VOCR5 pools (36 rows) |
| NPJH50621 | psp | train | v57 | 1,504 (60) | 2,633 | 1,922 | 28,333 | 657 | 17 | 0.0% | 0% | VOCR5 pools (159 rows) |
| NPJH50641 | psp | train | v7018 | 1,130 (45) | 2,361 | 1,711 | 31,123 | 516 | 18 | 0.0% | 0% | VOCR5 pools (821 rows) |
| NPJH50649 | psp | train | v10623 | 1,064 (43) | 2,963 | 1,932 | 21,706 | 1,008 | 16 | 100.0% | 100% | VOCR5 pools (1,493 rows) |
| ULJM05383 | psp | train | v34 | 1,827 (73) | 2,877 | 2,712 | 48,104 | 159 | 18 | 0.0% | 1% | protected family; VOCR5 pools (1,564 rows) |
| ULJM05674 | psp | train | v10 | 1,997 (80) | 2,972 | 2,935 | 47,334 | 37 | 18 | 7.4% | 0% | VOCR5 pools (698 rows) |
| PCSG00007 | vita | train | v5 | 562 (22) | 956 | 657 | 10,333 | 265 | 37 | 0.0% | 0% | protected family |
| PCSG00146 | vita | train | v6618 | 1,498 (60) | 2,800 | 1,878 | 36,739 | 841 | 27 | 0.0% | 1% | VOCR5 pools (524 rows) |
| PCSG00352 | vita | train | v5883 | 585 (23) | 936 | 809 | 16,216 | 81 | 27 | 0.0% | 0% | VOCR5 pools (287 rows) |
| PCSG00367 | vita | train | v15118 | 1,165 (47) | 2,042 | 1,977 | 29,198 | 65 | 23 | 98.2% | 100% |  |
| PCSG00380 | vita | train | v18456 | 1,095 (44) | 2,087 | 2,071 | 35,430 | 16 | 28 | 0.0% | 0% |  |
| PCSG00468 | vita | train | v15771 | 412 (16) | 701 | 678 | 11,263 | 14 | 27 | 83.9% | 0% | VOCR5 pools (12 rows) |
| PCSG00469 | vita | train | v6723 | 2,047 (82) | 4,364 | 3,866 | 53,794 | 444 | 36 | 84.2% | 0% | VOCR5 pools (20 rows) |
| PCSG00509 | vita | train | v16049 | 1,299 (52) | 1,912 | 1,867 | 32,491 | 27 | 32 | 0.1% | 0% | VOCR5 pools (27 rows) |
| PCSG00540 | vita | train | v12260 | 1,037 (41) | 1,885 | 1,099 | 17,295 | 605 | 32 | 1.7% | 0% | VOCR5 pools (9 rows) |
| PCSG00547 | vita | train | v17285 | 1,021 (41) | 2,208 | 1,499 | 24,009 | 688 | 39 | 0.2% | 0% | VOCR5 pools (630 rows) |
| PCSG00553 | vita | train | v11472 | 1,129 (45) | 2,237 | 2,156 | 29,000 | 50 | 25 | 99.5% | 0% | VOCR5 pools (69 rows) |
| PCSG00555 | vita | train | v14018 | 394 (16) | 602 | 574 | 11,837 | 20 | 21 | 0.7% | 0% | VOCR5 pools (1,056 rows) |
| PCSG00576 | vita | train | v12188 | 1,219 (49) | 2,250 | 2,231 | 31,135 | 19 | 44 | 0.0% | 0% | dev sample |
| PCSG00611 | vita | train | v15248 | 1,382 (55) | 1,905 | 1,725 | 34,162 | 180 | 23 | 0.0% | 0% |  |
| PCSG00620 | vita | train | v14973 | 995 (40) | 1,823 | 1,795 | 31,512 | 28 | 37 | 0.0% | 0% | VOCR5 pools (31 rows) |
| PCSG00628 | vita | train | v11301 | 1,208 (48) | 2,402 | 1,860 | 29,036 | 419 | 31 | 9.6% | 0% | VOCR5 pools (15 rows) |
| PCSG00648 | vita | train | v10680 | 1,393 (56) | 3,307 | 2,041 | 46,051 | 1,216 | 28 | 0.0% | 1% | VOCR5 pools (1,259 rows) |
| PCSG00653 | vita | train | v17343 | 863 (35) | 1,611 | 0 | 0 | 44 |  |  | 0% | VOCR5 pools (2 rows) |
| PCSG00666 | vita | train | v14759 | 1,583 (63) | 2,600 | 2,561 | 33,002 | 39 | 20 | 99.8% | 0% | VOCR5 pools (33 rows) |
| PCSG00667 | vita | train | v17608 | 2,000 (80) | 3,236 | 3,074 | 46,608 | 129 | 33 | 0.0% | 0% | VOCR5 pools (30 rows) |
| PCSG00713 | vita | train | v17287 | 1,463 (59) | 3,979 | 2,699 | 36,319 | 1,253 | 24 | 0.0% | 0% | VOCR5 pools (24 rows) |
| PCSG00751 | vita | train | v8203 | 240 (10) | 221 | 210 | 1,926 | 10 | 26 | 99.5% | 100% | VOCR5 pools (24 rows) |
| PCSG00753 | vita | train | v8243 | 1,000 (40) | 1,379 | 1,310 | 25,545 | 69 | 23 | 100.0% | 0% | VOCR5 pools (33 rows) |
| PCSG00805 | vita | train | v18314 | 1,329 (53) | 2,421 | 2,365 | 32,935 | 55 | 32 | 0.0% | 0% | VOCR5 pools (37 rows) |
| PCSG00807 | vita | test | v18432 | 690 (0) | 1,174 | 1,152 | 13,911 | 22 | 23 | 0.0% | 0% | dev sample |
| PCSG00854 | vita | train | v18833 | 988 (40) | 1,667 | 1,643 | 26,760 | 24 | 31 | 74.9% | 0% |  |
| PCSG00866 | vita | train | v18160 | 513 (21) | 997 | 897 | 13,250 | 100 | 32 | 0.0% | 0% | protected family |
| PCSG00876 | vita | train | v18162 | 1,049 (42) | 1,756 | 1,747 | 28,057 | 9 | 32 | 0.0% | 0% | VOCR5 pools (31 rows) |
| PCSG00914 | vita | train | v4602 | 515 (21) | 1,089 | 1,064 | 13,611 | 24 | 28 | 100.0% | 0% |  |
| PCSG00917 | vita | train | v2023 | 1,564 (63) | 2,402 | 2,373 | 39,004 | 29 | 31 | 0.0% | 0% | VOCR5 pools (72 rows) |
| PCSG00918 | vita | train | v2024 | 1,500 (60) | 2,466 | 2,421 | 39,459 | 45 | 31 | 0.0% | 0% | VOCR5 pools (68 rows) |
| PCSG00935 | vita | train | v8095, v12775 | 382 (15) | 771 | 758 | 10,986 | 13 | 30 | 0.0% | 0% | VOCR5 pools (40 rows) |
| PCSG01110 | vita | train | v21521 | 343 (14) | 643 | 484 | 6,512 | 159 | 32 | 0.0% | 0% |  |
<!-- /generated:titles -->

**Mapping titles to VNs.** Six titles had no VNDB id in their metadata. I matched them in the VNDB dump of
2026-10-13 by release title and platform (`data/manifests/vocr6_hooked/vndb_map.json`).

**Overlap with the protected sets** (the owner trains on these anyway; every VOCR6 report flags them):

- **PCSG00007**, Little Busters! (v5), and **ULJM05383**, planetarian (v34). Both share release r3748, the Key 10th
  Memorial BOX, with CLANNAD (v4). A CLANNAD score of any model trained on them carries that exposure. planetarian's was
  already disclosed for VOCR5.
- **PCSG00866**, Collar x Malice (v18160). It shares four releases with vita-test's Collar x Malice -Unlimited-
  (v21789). A vita-test score of any model trained on it is not an unseen-game score for that VN.
- **The development sample.** PCSG00576 (v12188: its Vita port 幻燈ノスタルジィ is a release of the dev-sample VN
  黄昏ポウラスタ) trains, so VOCR6 models will have seen a dev-sample VN: it has 14 teacher_val and 18 devnative
  development lines. The test title PCSG00807 (v18432) is also a dev-sample VN, so selecting on its 4 dev lines
  couples selection to the test title.
  - **VOCR6's dev comparisons leave both VNs out** (the results' `without_v12188_v18432`; 11 tv, 23 dn and 42 wd
    dialogue rows).
  - The baseline's dev score is then 3.147 for Small, 2.915 for Normal and 2.891 for Large (rubi_cascade), against
    3.096, 2.864 and 2.838 with them.

**VOCR5's exposure.** 38 of the 49 titles' VNs had rows in VOCR5's fine-tune pools, as VNDB screenshots or owner
captures: 1,564 rows for planetarian, 1,493 for カスタムドライブ, 1,259 for 月に寄りそう乙女の作法. So VOCR5's hval
scores are not unseen-game scores either. The unseen-game signal for VOCR6 decisions is the development sample
without v12188 and v18432.

**Kanji coverage.** <!-- generated:coverage -->
Of the 5,142 kanji of `data/vocab_vn_5631.txt`, 2,587 occur on the training side (train + guard + hval): 727 at least 80 times, 967 10-79 times and 893 1-9 times. 2,555 never occur. The kept rows hold 2,873 distinct characters (2,592 kanji). 12 characters outside the vocab removed 92 records; 5 kept characters are outside Small's 4,987-class vocab, which trains them as `<unk>`.
<!-- /generated:coverage -->

So the real data covers about half of the output vocab's kanji, and about 900 of them fewer than 10 times. The rest
needs synthesis in the harvested game fonts (T2, synth6). `stats_v1.json` lists the absent and the rare kanji.

## Baseline: VOCR5 as shipped (`eval/vocr6_baseline.py`)

**The arms** are the released files, sha256-checked. They hold the head gate and the two-level head that #120 and
#122 adopted. Rubi_for_vita's main still pins v0.3.0's plain blobs, so these arms are VOCR5 as released, not yet as
Rubi ships it. Their adoption rules held the dev score within +0.05 points (95% upper bounds) of the plain blobs:

- Small: `VOCR5_S_w8g.vocr` (v0.3.5-headgate.1), with its head gate.
- Normal and Large: `VOCR5_N_w8g2.vocr` and `VOCR5_L_w8g2.vocr` (v0.3.6-head2level.1), with the head gate and the
  two-level head on.
- Vertical columns: VOCR5-V (v0.3.2-vertical.1), greedy.

**The reading path** is Rubi's, plus the two-level head's re-read:

1. W8I through the C runtime, top-8.
2. The counter re-decode {6000, 4000}, then beam 8 with the 1 MB character LM `vocr_lm_o3_1000k.vlm` at Rubi's
   settings.
3. The ungated re-read of a line that the re-decode finds a counter context in, or that holds a full-size カ/ケ
   before a counter character. It fired on none of these sets' crops, so `rubi` equals head2level's `rubi`.
4. The counter rule.
5. `rubi_cascade`: a line whose confidence is under 700 is read again by the next tier up.

**The measures**, all pre-registered in `vocr6/experiments.jsonl` before any read:

- *Clean*: the hval pools' crops as stored.
- *Page GRAD+refine*: the C GRAD finder with the refinement (VOCR_REFINE_DEFAULT; the driver matched its Python twin
  box for box on every region) on every whole-dialogue region, each line read with its dark flag, scored as whole
  dialogue (`train/general_vn_textbox_guard.transcript_score`).
  - The *console view* is Vita pages at native size and PSP pages upscaled 2x, the view Rubi has on a Vita.
  - PSP at native size is a separate column.
  - Regions are the base variant (+4 / +8 px in the results file).
- *Page textspot3*: the textspot3 pick (`T3_kd_t64n8c16_seed3`, det+refine, 51,008 B) on every page with verified
  text (PSP at 2x, as textspot's sets do), scored by `eval/textspot_metrics.py`.
  - The references are the verified lines' main-text boxes. Ruby and every renderer box that no verified record
    covers are ignore areas.
  - So the spotter's F1 counts UI it finds outside those boxes as false positives, and is lower than on textspot's
    fully labelled sets.
- *Dev*: VOCR5's development score (½ fixed lines, ½ whole dialogue; `eval/vocr5_select.py`). It reproduces
  `eval/head2level_dev_20261023.json` exactly for every tier and decode, e.g. Normal's rubi_cascade 2.86387.

The test title was read once.

**Relation to the shared page-path benchmark.** `eval/pagepath.py` (#129, merged in #131 after this baseline ran)
reads the same page pool's regions the way Rubi's main does today:
- Rubi's finder mode (TEXTBOX_RECOVERY with its short-line probe, the refinement on PSP titles only);
- v0.3.0's plain blobs;
- Rubi's text normalisation and its non-Japanese line filter.

This baseline's page path is the one the T0 task asked for: GRAD with the refinement everywhere, and the released
head-gate / two-level arms. So the two page-path figures are not the same measure. VOCR6 decisions on the page path
use #129's benchmark; this document's figures describe the corpus and VOCR5 on it.

**The intervals** are 95% bootstraps (10,000 resamples, seed 0):

- over titles for the pooled rates;
- over pages within a title, and for the one-title test set;
- over VNs for the dev score (seeds 0 / 1 / 2 per component, as VOCR5).

**The metric** is dialogue CER: `eval/metrics.norm` and Levenshtein over the reference characters. Speaker names are
scored separately (hval Normal: 0 errors on 40 characters).

<!-- generated:baseline -->
| tier | decode | hval clean, Vita | hval clean, PSP | hval clean, all | hval page GRAD+refine, console view (pooled; title macro) | the same, regions without a seen line | PSP page at native size | hval page textspot3, dialogue | textspot3 F1 | dev score [CI] | dev score without v12188, v18432 | test clean | test page GRAD+refine | test page textspot3 |
|---|---|---|---|---|---|---|---|---|---:|---|---|---|---|---|
| S | greedy | 0.69% [0.51, 0.89] | 1.29% [0.67, 2.08] | 0.89% [0.63, 1.22] | 5.34% [2.34, 10.56]; 5.48% | 4.91% [1.91, 10.32] | 6.51% [2.19, 12.35] | 4.44% [1.38, 10.63] | 0.726 | 3.798 [2.10, 5.79] | 3.851 [2.14, 5.83] | 0.27% [0.15, 0.42] | 1.76% [1.33, 2.24] | 1.75% [1.31, 2.23] |
| S | rubi | 0.46% [0.32, 0.62] | 0.84% [0.48, 1.27] | 0.59% [0.43, 0.78] | 4.56% [2.08, 8.76]; 4.76% | 4.10% [1.65, 8.44] | 5.91% [1.91, 11.22] | 3.68% [1.12, 8.77] | 0.726 | 3.407 [1.76, 5.35] | 3.464 [1.79, 5.41] | 0.18% [0.08, 0.30] | 1.71% [1.29, 2.18] | 1.61% [1.19, 2.08] |
| S | rubi_cascade | 0.38% [0.24, 0.55] | 0.68% [0.35, 1.08] | 0.48% [0.33, 0.65] | 5.29% [1.95, 11.44]; 5.43% | 4.87% [1.54, 11.22] | 5.56% [1.75, 10.62] | 4.70% [1.12, 12.16] | 0.726 | 3.096 [1.57, 4.87] | 3.147 [1.59, 4.97] | 0.14% [0.07, 0.24] | 1.66% [1.25, 2.12] | 1.48% [1.10, 1.90] |
| N | greedy | 0.51% [0.32, 0.76] | 0.84% [0.42, 1.36] | 0.62% [0.42, 0.87] | 5.35% [2.07, 11.29]; 5.51% | 4.93% [1.66, 11.04] | 5.95% [1.87, 11.44] | 4.70% [1.24, 11.93] | 0.726 | 3.357 [1.76, 5.23] | 3.419 [1.81, 5.31] | 0.17% [0.09, 0.26] | 1.81% [1.38, 2.29] | 1.63% [1.21, 2.10] |
| N | rubi | 0.34% [0.22, 0.49] | 0.61% [0.31, 0.98] | 0.43% [0.30, 0.59] | 5.24% [1.90, 11.40]; 5.38% | 4.83% [1.49, 11.17] | 5.52% [1.78, 10.58] | 4.58% [1.01, 12.07] | 0.726 | 3.075 [1.56, 4.85] | 3.130 [1.58, 4.92] | 0.12% [0.06, 0.20] | 1.71% [1.29, 2.19] | 1.41% [1.05, 1.84] |
| N | rubi_cascade | 0.27% [0.18, 0.39] | 0.54% [0.27, 0.87] | 0.36% [0.25, 0.50] | 5.18% [1.86, 11.31]; 5.32% | 4.76% [1.46, 11.07] | 5.32% [1.70, 10.10] | 4.25% [0.94, 11.11] | 0.726 | 2.864 [1.40, 4.54] | 2.915 [1.42, 4.64] | 0.12% [0.06, 0.20] | 1.71% [1.28, 2.18] | 1.31% [0.96, 1.70] |
| L | greedy | 0.36% [0.23, 0.50] | 0.67% [0.33, 1.09] | 0.46% [0.31, 0.64] | 5.24% [1.92, 11.32]; 5.36% | 4.83% [1.52, 11.10] | 5.71% [1.84, 10.87] | 4.50% [1.07, 11.66] | 0.726 | 2.974 [1.45, 4.73] | 3.031 [1.49, 4.83] | 0.11% [0.04, 0.19] | 1.69% [1.26, 2.15] | 1.40% [1.04, 1.80] |
| L | rubi | 0.26% [0.15, 0.37] | 0.46% [0.24, 0.74] | 0.33% [0.22, 0.45] | 5.16% [1.85, 11.25]; 5.30% | 4.73% [1.44, 11.03] | 5.26% [1.62, 10.08] | 4.23% [0.92, 11.10] | 0.726 | 2.838 [1.37, 4.52] | 2.891 [1.40, 4.63] | 0.07% [0.02, 0.13] | 1.67% [1.24, 2.15] | 1.30% [0.95, 1.69] |
| L | rubi_cascade | 0.26% [0.15, 0.37] | 0.46% [0.24, 0.74] | 0.33% [0.22, 0.45] | 5.16% [1.85, 11.25]; 5.30% | 4.73% [1.44, 11.03] | 5.26% [1.62, 10.08] | 4.23% [0.92, 11.10] | 0.726 | 2.838 [1.37, 4.52] | 2.891 [1.40, 4.63] | 0.07% [0.02, 0.13] | 1.67% [1.24, 2.15] | 1.30% [0.95, 1.69] |
<!-- /generated:baseline -->

**Findings:**

- **The recognizer reads clean native crops almost perfectly.** Normal's hval CER is 0.36%, and 95.4% of lines are
  exact. Six titles hold 46% of the errors: NPJH50621, NPJH50356, PCSG00648, PCSG00620, NPJH50279 and PCSG00914.
  PSP is about twice Vita (0.54% against 0.27%).
- **The page path is 14x the clean CER: 5.18% against 0.36%.** The finder makes most of that error. The textspot3
  spotter is about as bad on Vita: 6.02% against GRAD + refine's 6.50%.
  - PCSG00854 alone holds 51.5% of the console view's errors. Its dialogue box has a decorative dotted border a few
    pixels above the first line. GRAD returns it as a line, and Normal reads it as "1010…" (112% CER for that
    title; textspot3 127%). Without that title the console view reads at 2.57%.
  - The next largest are ULJM05674, PCSG00611, PCSG00753 and PCSG00866 (11-16%):
    - ULJM05674 (at 2x): two rows merged into one crop, which reads as noise. The 7 regions where the finder returns
      fewer lines than are labelled hold 198 of its 220 errors.
    - PCSG00753: rows of nothing but …… that the finder does not find. The 5 regions with such a row hold 100 of its
      122 errors.
    - PCSG00611 loses text in three ways:
      - two …… rows the finder does not find (40 errors);
      - 9 lines lose their head (64 characters) and 4 their tail (32). On lines with a wide inner gap of full-width
        spaces between two sentences, GRAD + refine keeps the part on one side of the gap and drops the other (viewed
        on page 001146);
      - 6 more whole lines lost or garbled (23 characters).
    - PCSG00866: 3 lines lose their head, 2 their tail and 2 whole lines are lost (17, 17 and 8 characters).
  - That is finder work for T3 and T3b. With the region given, the recognizer tier hardly matters: S, N and L read
    the console view at 5.29%, 5.18% and 5.16%.
- **PSP reads better at 2x.** PSP pages upscaled 2x (the console view) read at 2.81% through GRAD + refine against
  5.32% at native size. The clean `line_x2` crops read at 0.47% against 0.54% native.
- **The spotter is weak on Vita.** textspot3 finds PSP boxes well (F1 0.85, dialogue 1.00%) but not Vita boxes
  (F1 0.67, dialogue 6.02%). The test title gives F1 0.57.
- **The test title.** Normal reads its clean crops at 0.12% and its whole dialogue at 1.71% (GRAD + refine) or 1.31%
  (textspot3).
- **Vertical text.** The vertical title's columns (PCSG00653, hval) read at 1.98% with VOCR5-V on the turned crop.
  The re-flowed crops read at 15.85% with VOCR5-N, which never saw such strips.

Normal (`rubi_cascade`) per hval title. The page-path columns are the console view (base regions):

<!-- generated:baseline_titles -->
| title | hval clean dialogue CER [page CI] | page GRAD+refine, console view | page textspot3 dialogue | clean ref. chars | page-path pages |
|---|---|---|---|---:|---:|
| NPJG00016 | 4.94% [0.00, 17.78] | 4.98% [0.00, 16.67] | 3.46% [0.00, 12.44] | 162 | 10 |
| NPJH00140 | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | 0.20% [0.00, 0.46] | 1358 | 65 |
| NPJH50001 | 0.10% [0.00, 0.32] | 0.18% [0.04, 0.36] | 0.27% [0.09, 0.48] | 1014 | 64 |
| NPJH50002 | 0.56% [0.16, 1.03] | 0.68% [0.12, 1.58] | 3.05% [1.68, 4.83] | 1258 | 41 |
| NPJH50132 | 0.69% [0.16, 1.36] | 2.42% [0.66, 4.79] | 0.60% [0.15, 1.16] | 1305 | 73 |
| NPJH50279 | 1.69% [0.43, 3.11] | 3.63% [1.28, 6.61] | 2.84% [1.37, 4.66] | 593 | 33 |
| NPJH50297 | 0.10% [0.00, 0.32] | 5.49% [2.71, 8.59] | 1.65% [1.01, 2.33] | 1005 | 45 |
| NPJH50356 | 1.13% [0.52, 1.86] | 2.92% [1.12, 5.21] | 1.46% [0.52, 2.83] | 1503 | 80 |
| NPJH50382 | 0.40% [0.13, 0.69] | 0.78% [0.24, 1.45] | 0.45% [0.06, 0.95] | 1769 | 79 |
| NPJH50551 | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | 1.84% [0.00, 6.02] | 272 | 8 |
| NPJH50621 | 1.83% [0.87, 3.04] | 2.30% [0.96, 3.97] | 1.74% [0.69, 3.10] | 1255 | 60 |
| NPJH50641 | 0.15% [0.00, 0.39] | 4.57% [2.06, 7.66] | 0.88% [0.36, 1.62] | 1351 | 45 |
| NPJH50649 | 0.00% [0.00, 0.00] | 0.62% [0.00, 1.55] | 0.25% [0.00, 0.70] | 796 | 42 |
| PCSG00007 | 0.22% [0.00, 0.71] | 1.08% [0.00, 3.02] | 0.22% [0.00, 0.70] | 461 | 22 |
| PCSG00146 | 0.14% [0.00, 0.37] | 0.71% [0.07, 1.83] | 0.78% [0.15, 1.66] | 1386 | 58 |
| PCSG00352 | 0.00% [0.00, 0.00] | 0.92% [0.00, 2.09] | 0.39% [0.00, 0.92] | 764 | 23 |
| PCSG00367 | 0.17% [0.00, 0.47] | 0.29% [0.00, 1.07] | 0.14% [0.00, 0.40] | 1155 | 46 |
| PCSG00380 | 0.13% [0.00, 0.31] | 0.51% [0.07, 1.14] | 0.32% [0.06, 0.67] | 1572 | 42 |
| PCSG00468 | 0.95% [0.00, 2.80] | 7.81% [1.99, 16.18] | 2.19% [0.00, 6.90] | 316 | 16 |
| PCSG00469 | 0.22% [0.05, 0.41] | 2.89% [1.45, 4.72] | 0.88% [0.41, 1.49] | 2302 | 80 |
| PCSG00509 | 0.22% [0.00, 0.60] | 1.65% [0.35, 3.32] | 3.46% [1.80, 5.27] | 1373 | 51 |
| PCSG00540 | 0.13% [0.00, 0.45] | 0.13% [0.00, 0.45] | 1.05% [0.42, 1.88] | 751 | 41 |
| PCSG00547 | 0.13% [0.00, 0.41] | 1.80% [0.36, 3.90] | 0.64% [0.12, 1.34] | 757 | 41 |
| PCSG00553 | 0.17% [0.00, 0.44] | 2.01% [0.39, 4.35] | 1.02% [0.08, 2.54] | 1154 | 44 |
| PCSG00555 | 0.00% [0.00, 0.00] | 2.40% [0.00, 7.32] | 1.44% [0.00, 5.45] | 411 | 16 |
| PCSG00576 | 0.17% [0.00, 0.43] | 0.97% [0.00, 3.18] | 0.41% [0.00, 1.25] | 1196 | 47 |
| PCSG00611 | 0.22% [0.00, 0.46] | 11.08% [6.49, 16.59] | 4.00% [0.84, 8.75] | 1366 | 54 |
| PCSG00620 | 0.91% [0.40, 1.49] | 3.02% [1.71, 4.60] | 0.93% [0.40, 1.54] | 1326 | 40 |
| PCSG00628 | 0.54% [0.10, 1.15] | 0.18% [0.00, 0.48] | 1.09% [0.09, 2.40] | 1104 | 48 |
| PCSG00648 | 0.70% [0.32, 1.14] | 0.82% [0.18, 1.71] | 0.51% [0.21, 0.87] | 1851 | 53 |
| PCSG00653 |  |  | 11.14% [6.40, 18.49] |  |  |
| PCSG00666 | 0.38% [0.08, 0.73] | 3.44% [1.04, 6.72] | 1.72% [1.12, 2.34] | 1317 | 63 |
| PCSG00667 | 0.36% [0.06, 0.72] | 2.08% [1.18, 3.22] | 1.91% [1.01, 3.06] | 1667 | 80 |
| PCSG00713 | 0.13% [0.00, 0.34] | 0.26% [0.00, 0.68] | 0.38% [0.00, 0.90] | 1553 | 59 |
| PCSG00751 | 0.00% [0.00, 0.00] | 3.30% [0.00, 8.64] | 0.00% [0.00, 0.00] | 91 | 6 |
| PCSG00753 | 0.00% [0.00, 0.00] | 15.84% [6.10, 27.83] | 14.03% [4.30, 25.77] | 670 | 40 |
| PCSG00805 | 0.42% [0.07, 0.94] | 0.76% [0.07, 1.84] | 0.07% [0.00, 0.22] | 1439 | 53 |
| PCSG00854 | 0.08% [0.00, 0.25] | 112.20% [84.80, 141.48] | 126.81% [103.36, 151.21] | 1287 | 40 |
| PCSG00866 | 0.00% [0.00, 0.00] | 11.28% [4.41, 20.18] | 2.56% [0.00, 6.92] | 390 | 21 |
| PCSG00876 | 0.09% [0.00, 0.30] | 2.50% [0.65, 5.16] | 0.74% [0.16, 1.57] | 1074 | 42 |
| PCSG00914 | 1.69% [0.42, 3.20] | 3.69% [1.52, 6.64] | 2.35% [1.10, 3.99] | 592 | 21 |
| PCSG00917 | 0.00% [0.00, 0.00] | 2.01% [0.59, 3.81] | 0.06% [0.00, 0.20] | 1685 | 63 |
| PCSG00918 | 0.18% [0.00, 0.39] | 0.47% [0.06, 1.05] | 0.23% [0.06, 0.46] | 1699 | 60 |
| PCSG00935 | 0.00% [0.00, 0.00] | 5.41% [0.97, 11.90] | 0.18% [0.00, 0.55] | 550 | 15 |
| PCSG01110 | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | 0.00% [0.00, 0.00] | 218 | 14 |
| ULJM05383 | 0.04% [0.00, 0.14] | 1.38% [0.44, 2.68] | 0.17% [0.00, 0.40] | 2302 | 73 |
| ULJM05674 | 0.49% [0.19, 0.83] | 13.31% [5.59, 21.86] | 0.79% [0.35, 1.33] | 1624 | 80 |
<!-- /generated:baseline_titles -->

The machine-readable results: `eval/vocr6_baseline_20261026.json`, compact. It holds every tier, decode, pool, region
variant and scale, per title, with intervals, plus the dev components and the read counts.

## Training throughput on this host (`train/bench_train2.py`, `eval/vocr6_throughput_20261026.json`)

This host has no GPU (torch 2.5.1+cpu): 64 AMD EPYC 9R14 cores, 246 GB of RAM, 15 heavy slots of 5 threads since
10:18Z (12 before).

**The method.**

- `bench_train2.py` runs train2's step exactly: its sources, mixer, batching with augmentation and finder
  augmentation, forward, CTC, backward, clipping, AdamW and EMA. The batch is 96, in one heavy.sh slot.
- 30 warm-up steps, then 300 measured.
- The box carried other agents' jobs during the run (load average 31-41 of 64 cores), as real runs do. The table's
  rates are from that partly loaded box; the loaded reruns are below it.
- "Data" is the mixer and batching in train2's main thread. The streamed synthetic shards load in its background
  thread.

| recipe (VOCR5's, with the hooked line pool as the real half) | parameters | steps/s | lines/s | data share of a step | slot-hours per 10k steps |
|---|---:|---:|---:|---:|---:|
| Small fine-tune: recipe C synthetic half + hooked half | 0.73 M | 2.27 | 218 | 5.3% | 1.23 |
| Normal pretrain: recipe M, synthetic only | 1.07 M | 2.08 | 200 | 2.0% | 1.34 |
| **Normal fine-tune: recipe M synthetic half (finder augmentation) + hooked half** | 1.07 M | **1.76** | 169 | 4.1% | **1.58** |
| Normal fine-tune: hooked lines alone | 1.07 M | 1.49 | 143 | 1.6% | 1.87 |
| Large fine-tune: recipe M synthetic half + hooked half | 2.32 M | 1.20 | 116 | 2.7% | 2.31 |

**The rate falls as the box fills.** The same Normal fine-tune bench, with the same losses at every logged step:

| load average (64 cores) | steps/s | slot-hours per 10k steps |
|---|---:|---:|
| 31-41 (09:39Z) | 1.76 | 1.58 |
| 58-65 (the audit's rerun, 11:26Z) | 1.40 | 1.99 |
| 79-87 (12:35Z, 15 heavy slots busy) | 0.90 | 3.10 |

**VOCR5's own runs on this host,** measured over each run's whole log, evaluations included:

- the Normal pretrain ran at 1.75 steps/s for 23.1 h (1.59 h per 10k steps), its fine-tune at 1.75;
- Small at 2.49 and 2.37;
- Large at 1.31 and 1.25.

The fine-tune benches agree with VOCR5's fine-tunes. The Normal pretrain bench (2.08 steps/s) is 19% faster than
VOCR5's real pretrain (1.75), which ran on a busier box. The hooked-only arm is slower per step because its lines are
wider (mean batch width 452 px against 410 px for the
mix), and compute grows with width.

**What it means for the budget:**

- **Training is compute-bound.** Data loading takes 2-5% of a step, so faster loaders would buy little; more threads
  per run or a GPU would.
- **Run costs** (estimates from the rates above):
  - A 15k-step Normal fine-tune (VOCR5's length) costs about 2.4 slot-hours on a partly loaded box and 4.6 on a full
    one.
  - A 150k-step Normal pretrain costs about 24 slot-hours on a partly loaded box (VOCR5's took 23 h), and up to
    about 2x that on a full one.
  - Three seeds of a Normal fine-tune arm cost about 7-14 slot-hours.
- **The whole box** (an estimate): 15 slots make about 1.2 M Normal fine-tune steps a day at the full-box rate
  (0.90 steps/s), shared by every VOCR6 track (`vocr6/slots.txt`). The 1.40 rate at load 58-65 would give 1.8 M at
  15 slots and 1.45 M at 12; the box ran fuller than that today.

## Reproduce

```sh
H=$HERD/shared; M=$H/private/hooked-b2; A=$H/artifacts/vocr6-ingest; PY=.venv/bin/python
AWS_PROFILE=b2-kiro aws s3 sync s3://data-from-psp-and-vita/ $M --endpoint-url https://s3.eu-central-003.backblazeb2.com \
    --region eu-central-003                            # read only; then every metadata.json file's sha256 checked
$PY data/import_hooked.py extract --root $M --out $A/pages
$PY data/import_hooked.py split --root $M --out $A/splits_v1.json
$PY data/import_hooked.py lines --root $M --pages $A/pages --splits $A/splits_v1.json --out $A/pools/v1
$PY data/import_hooked.py pages --root $M --pages $A/pages --splits $A/splits_v1.json --pools $A/pools/v1 --out $A/pagepool/v1
$PY data/hooked_audit.py sample --pools $A/pools/v1 --out $A/audit/sample.jsonl     # drawn from the first import
$PY data/hooked_audit.py sheets --sample $A/audit/sample.jsonl --pools $A/pools/v1 --out $A/audit/sheets
$PY data/hooked_audit.py tally --sheets $A/audit/sheets --decisions $A/audit/decisions.jsonl --out $A/audit/audit_tally.json
$PY data/hooked_audit.py stats --pools $A/pools/v1 --pagepool $A/pagepool/v1 --splits $A/splits_v1.json --out $A/audit/stats.json
$PY data/hooked_audit.py overlap --stats $A/audit/stats.json --vndb-map data/manifests/vocr6_hooked/vndb_map.json \
    --vndb-dump $H/artifacts/native-frames/vndb/dump --out $A/audit/overlap.json
$PY data/import_hooked.py manifests --splits $A/splits_v1.json --pools $A/pools/v1 --pagepool $A/pagepool/v1 \
    --sources $A/sources_v1.json --out data/manifests/vocr6_hooked
for S in hval test; do $PY eval/vocr6_baseline.py build --set $S --pools $A/pools/v1 --pagepool $A/pagepool/v1 \
    --pages $A/pages --cache $A/baseline; done
$PY eval/vocr6_baseline.py dev-build --cache $A/baseline
for S in hval test dev; do for T in S N L V; do $PY eval/vocr6_baseline.py read --set $S --tier $T --cache $A/baseline; done; done
$PY eval/vocr6_baseline.py score --cache $A/baseline --pools $A/pools/v1 --pagepool $A/pagepool/v1 --out $A/baseline/results.json
$PY data/hooked_audit.py tables --index data/manifests/vocr6_hooked/index.json --stats data/manifests/vocr6_hooked/stats_v1.json \
    --overlap data/manifests/vocr6_hooked/overlap_v1.json --audit data/manifests/vocr6_hooked/label_audit_v1.json \
    --baseline eval/vocr6_baseline_20261026.json --doc docs/vocr6-data.md     # --check: what CI runs
```

The benchmark commands are in `eval/vocr6_throughput_20261026.json` `commands`. Run every heavy step through
`HEAVY_SLOTS=15 $HERD/heavy.sh`. The import took a few minutes per pass in one slot,
and the baseline's reads 0.3-8 minutes per tier and set. The pinned v1 files have these sha256:

- `pools/v1/manifest.json`: `ba95ff2d6c31d2ce…`
- `pagepool/v1/manifest.json`: `eab8f07bba1359bf…`
- `splits_v1.json`: `417df3d784cbd2cf…`

## Limits

- **Machine-verified labels.** The audit bounds the label error of the kinds it sampled: 0 of 318 after the fixes,
  under 1.2% at 95%. A rarer failure class could still exist; one reader looked at each line.
- **The 2x view is an emulation.** `line_x2` and the console view's PSP pages are a cv2 bilinear upscale, not real
  Adrenaline captures.
- **The page path does not cover every page.** Its regions come from verified lines, and boxes that no verified
  record covers are kept out. Pages whose dialogue is not fully verified count only their verified groups.
- **The bucket is live.** Titles marked `in_progress` may grow. v1 is the 08:29Z snapshot.
