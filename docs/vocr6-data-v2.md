# VOCR6 data v2: the hooked corpus at 138 titles

Issue #166. The owner's hooked PS Vita and PSP bucket grew from 49 titles to 138 between 2026-10-26 and 2026-10-31.
This document covers how the new state was downloaded and imported as pools v2.1, what the label audit found, and how
the shipped VOCR6.2 models read the new titles. It continues `docs/vocr6-data.md` (pools v1), whose rules apply unless
this document says otherwise. Every number was measured on the host. Nothing ran on a console.

**Summary**

- **The corpus.** 138 titles (119 Vita, 19 PSP) in 166 capture runs: 82,185 native pages and 147,424 machine-verified
  line records. Pools v1 had 49 titles and 55,493 pages.
- **After cleaning:** 116,987 usable dialogue lines (1,845,698 characters), up from 81,977 in v1 (+43%), plus 5,151
  vertical columns (v1 1,567), 872 names and 997 ruby readings.
- **A new format: several capture runs per title.** 28 titles have a second, shorter run. For 11 of them the bucket
  replaced the first run's files with the second run's. The first runs were recovered from the bucket's object
  versions, so pools v1 still rebuilds byte for byte.
- **Label audit.** 360 lines drawn from the records new in v2: 2 wrong. A scan of every title's rare characters then
  found wrong-label classes the random sample could not reach: an engine code that PCSG00122 never draws (505
  records), glyphs that six titles draw as other characters (113 records) and a nameplate drawn twice (1 record).
  The importer now fixes or drops them. After the fixes, 0 of the 358 audited records that remain are wrong (95%
  upper bound 1.1%).
- **Splits.** train 74,355 pages, guard 3,780, hval 3,010 (3.66%), test 1,040 (1.27%: Dance with Devils PCSG00807 and
  a new test title, うたわれるもの PORTABLE NPJH50882). A cross-run guard keeps a later run's copies of held-out pages
  out of training while hval is screened.
- **Baseline: VOCR6.2 as Rubi 0.10.5 runs it** (Normal with the cascade to Large). On every page of the 82 new train
  titles with dialogue (598,622 reference characters, never trained on), Rubi's default path reads 1.19% dialogue CER
  [0.98, 1.53]: Vita 1.16%, PSP 1.76%. Its 7,151 errors split three ways:
  - 2,860 come from Rubi's non-Japanese line filter, which drops rows of only punctuation (……, ――, ！？) that the
    spotter finds and the recognizer reads right;
  - 2,275 come from the spotter's boxes;
  - 2,016 (0.34%; Large 0.27%) come from the recognizer and its LM. On its own training titles' hval VOCR6.2 reads
    0.26%, so the new titles cost the recognizer about 0.08 points.

Everything downloaded is the owner's private, copyrighted game data. Pages, crops and texts stay under
`$HERD/shared/private/hooked-b2` and `$HERD/shared/artifacts/b2-ingest2`. The repository holds code, ids, hashes,
geometry, counts and scores only (`data/manifests/vocr6_hooked_v2/`).

## The bucket and the download

- **The sync.** `aws s3 sync` (read only, without `--delete`) into the v1 mirror, 2026-10-31 23:29:47-23:31:35Z: 1,337
  objects, 24,322,232,201 B. The bucket's listing was the same before and after the sync.
- **Verification.** Every file that the 155 title folders' `metadata.json` pin matches its bytes and sha256 (1,177 of
  1,177). The fonts are unchanged since v1. `README.md` and `index.json` changed; the README's format text did not,
  apart from its totals.
- **index.json** (2026-10-31T23:15:04Z) has 166 entries. They are capture runs, not titles: 138 distinct title ids.
  Its totals are 82,185 pages, 145,734 verified lines and names, and 1,690 ruby readings (147,424 records together).
  1,040 pages (1.27%) belong to the two `test` titles. Every other title is `train`.
- **v1 stays reproducible.** Before the sync the mirror kept v1's `README.md` and `index.json` as `README-20261026.md`
  and `index-20261026.json`. `v1-20261026/` holds the v1 mirror as hard links (49 title folders, the fonts, the two
  root files), checked against `sources_v1.json`: 370 pinned and 5 unpinned files. The record of the download is
  `$HERD/shared/artifacts/b2-ingest2/sources_v2.json` (in `index.json` `sources`).

### Capture runs

28 titles have two runs: the first, and a later run of about 100 pages (`paired-…-reloc-20261027`). The later run
mostly captures the same opening again: 97 of PCSG00367's 100 later page texts occur in its first run. Its frames are
not byte-identical to the first run's (other backgrounds, other animation frames).

- **Overwritten runs.** For 11 titles the later run went into the earlier run's folder (the same folder name) and
  replaced its files, while `index.json` lists both runs. The bucket keeps old object versions, so every earlier run
  was recovered into `versions/<platform>/<folder>/<timestamp>/`:
  - 3 runs of pools v1 (PCSG00367: 1,165 pages, PCSG00854: 988, PCSG00866: 513) were copied from the mirror;
  - 9 were fetched read only by version id (`s3api get-object --version-id`);
  - every file their `metadata.json` pins matches its sha256 (82 of 82).
- **One leftover export.** The 12th recovered run is a 2-page export of PCSG00062 that the later 17-page run replaced
  and `index.json` no longer lists. The importer leaves it out.

## Import (`data/import_hooked.py`, schema `vocr6-hooked/2`)

v1's filters, crops, pools and duplicate rules are unchanged. What is new:

- **One title, all its runs** (`title_runs`, `load_title_runs`). A title's runs are its folders plus every recovered
  run that `index.json` lists, in `packaged` order. Run 0 keeps its page ids. Run r ≥ 1's page NNNNNN becomes
  r × 100000 + NNNNNN, and its record ids `TITLE:PAGE:kindN` follow. Its pool rows keep their own record id and page
  (`src_id`, `src_page`) and name their run (`run`).
- **Splits per run, then a cross-run guard** (`split_runs`, `cross_run_guard`). Each run is split as a title of its
  own; run 0 keeps v1's seed, so every v1 title's first run has v1's split. Then a page of one run that repeats a
  held-out page of another run (the same page text of at least 6 characters, or for an hval page a shared verified
  line label of at least 6 characters) becomes guard, with a 15-page band around it when the page it repeats is hval.
  This moved 452 pages of 11 titles. No train page repeats the page text of another run's hval page (checked on the
  page pool).
- **VNDB ids** (`--vndb-map`, `data/manifests/vocr6_hooked_v2/vndb_map.json`). 23 titles carry no VNDB id in their
  metadata. 12 were matched by catalog number in the VNDB dump of 2026-10-13, 11 by hand by release title and
  platform. Three titles are two VNs each (meta `vns`): PCSG00883, PCSG00916 and PCSG00935.
- **The label rules of this audit** (`--label-rules v2`, the default; below).
- **Pool tag** `hooked-v2` (v1: `hooked-v1`).
- **v1, rebuilt.** `split` on `v1-20261026/`, then `lines --pool-tag hooked-v1 --label-rules v1` and `pages`, gives
  v1's files exactly: 370 of 370 pool files, 49 of 49 decision files, 49 of 49 page-pool files and all 49 titles'
  splits have v1's sha256 (`$HERD/shared/artifacts/b2-ingest2/v1check/report.json`).
- **pools v2 is superseded.** The first v2 import (pools manifest `22dbd259…`) predates the label rules. It is kept
  read only, marked `SUPERSEDED.txt`; use pools v2.1.

What the filters left:

<!-- generated:v2_filters -->
| kind | records | label in doubt | outside the vocab | under 8 px | CTC-infeasible | duplicates (exact / near) | pixel conflicts | hval copies of a guard line | usable (all splits) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| dialogue line | 129,903 | 483 | 13 | 52 | 0 | 7,888 / 4,375 | 9 | 96 | 116,987 |
| vertical column | 5,658 | 0 | 82 | 0 | 0 | 369 / 185 | 0 | 1 | 5,151 |
| speaker name | 10,173 | 0 | 0 | 0 | 0 | 4,438 / 4,837 | 22 | 4 | 872 |
| ruby reading | 1,560 | 0 | 8 | 25 | 0 | 300 / 229 | 0 | 1 | 997 |
<!-- /generated:v2_filters -->

The pools' rows per split:

<!-- generated:v2_splits -->
| rows | train | guard | hval | hval_fold | test |
|---|---:|---:|---:|---:|---:|
| pages | 74,355 | 3,780 | 3,010 | | 1,040 |
| line | 106,111 | 5,084 | 4,347 | 4,251 | 1,541 |
| name | 838 | 24 | 14 | 10 | 0 |
| ruby | 886 | 36 | 60 | 59 | 16 |
| vertical | 4,738 | 230 | 184 | 183 | 0 |
| vreflow | 4,669 | 228 | 181 | 181 | 0 |
| furimain | 817 | 42 | 53 | 52 | 19 |
| line_x2 | 29,085 | 1,199 | 1,204 | 1,185 | 389 |
| name_x2 | 322 | 14 | 7 | 5 | 0 |
<!-- /generated:v2_splits -->

## Label audit (`data/hooked_audit.py sample / sheets / tally`)

**The random sample.** 360 lines of the records that pools v1 did not hold (`sample --exclude-v1`), round robin over
106 title-and-run groups, with quotas: 70 image_rows lines, 40 furigana lines, 50 names, 40 ruby readings, 40
vertical columns (shown upright, re-flowed from their cells) and 120 glyph_cells lines. I (Claude, reviewer_kind
assistant) read 18 contact sheets of 20 lines, with zooms and page context where unsure. Only 19 lines are PSP: 86 of
the 89 new titles are Vita.

<!-- generated:v2_audit -->
| label audit stratum | lines | ok | wrong | wrong-label rate, 95% interval (Wilson) |
|---|---:|---:|---:|---|
| glyph_cells | 120 | 118 | 2 | 0.5-5.9% |
| image_rows | 70 | 70 | 0 | 0.0-5.2% |
| furigana | 40 | 40 | 0 | 0.0-8.8% |
| name | 50 | 50 | 0 | 0.0-7.1% |
| ruby | 40 | 40 | 0 | 0.0-8.8% |
| vertical | 40 | 40 | 0 | 0.0-8.8% |
| platform:vita | 341 | 339 | 2 | 0.2-2.1% |
| platform:psp | 19 | 19 | 0 | 0.0-16.8% |
| all | 360 | 358 | 2 | 0.2-2.0% |
| after the fixes (pools v2.1) | 358 | 358 | 0 | 0.0-1.1% |
<!-- /generated:v2_audit -->

The 2 wrong lines:

1. **Doubled cells** (PCSG00926, 1 record). The game draws a ？？？ nameplate twice under two atlas keys, so its 3
   glyphs gave 6 cells and the 6-character dialogue line below matched them. The crop is the nameplate.
2. **A remapped glyph** (PCSG00476, 3 records). The hook's iteration mark ゞ is drawn as a !? ligature.

**The rare-character scan.** Both classes are per-title quirks, so I listed every non-kanji character that occurs in
one title only and checked each on a sheet (`audit/sheets_odd`, `sheets_check2`, `sheets_check3`, `sheets_caret`).
That found the classes below. Each is now a rule in the importer:

- **An undrawn code** (PCSG00122, Fate/stay night: 505 records). The hook text carries `^`, an engine code the game
  never draws, and the cell matcher gave it a cell of its own.
  - Where its cell is blank (150 records), the label without the code is right: the code is removed and the record
    kept (`undrawn_stripped`).
  - Where a drawn glyph sits under it (355 records), the whole assignment is shifted: the record is dropped
    (`undrawn_shifted`).
  - The rule tells the two apart by ink in the code's cells and in every other cell (GRAD's contrast threshold). It
    agreed with all 20 verdicts of the caret sheet. A post-fix sheet of 20 kept rows found 0 wrong.
- **Remapped glyphs** (113 records): characters a title's font draws as something else.
  - NPJH50001 ㊤㊥㊦ and NPJH50382 ├┤≪≫ and NPJH50621 ├┤: drawn as ―.
  - NPJH50002 ㊨: drawn as 鬼.
  - NPJH50882, the new test title: Ⅵ drawn as a dakuten, Ⅷ as ―, ⑮-⑲ as ～ (31 records).
  - PCSG00476 ゞ: drawn as !?.
  - Every record of such a character is dropped (`remapped`).
- **Doubled cells** (1 record): a glyph_cells record of which at least half the cells coincide with the next one
  (overlap ≥ 0.85 along and across the line). Only PCSG00926's record is like that.
- **One image_rows record** (PCSG00367): the line of ＃＆＄％ symbols ends in a ／ drawn where the label has ！
  (`audited_wrong`).

A dropped record's page counts as in doubt and loses its page-path regions, as v1's `label_doubt` does:

<!-- generated:v2_rules -->
| title | split | undrawn code removed, kept | undrawn code, shifted | remapped glyph | doubled cells | audited wrong | dropped (label in doubt) | pages in doubt (no regions) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| NPJH50001 | train | 0 | 0 | 3 | 0 | 0 | 5 | 5 |
| NPJH50002 | train | 0 | 0 | 1 | 0 | 0 | 1 | 1 |
| NPJH50382 | train | 0 | 0 | 73 | 0 | 0 | 73 | 73 |
| NPJH50621 | train | 0 | 0 | 2 | 0 | 0 | 2 | 2 |
| NPJH50882 | test | 0 | 0 | 31 | 0 | 0 | 31 | 31 |
| PCSG00122 | train | 150 | 355 | 0 | 0 | 0 | 355 | 153 |
| PCSG00367 | train | 0 | 0 | 0 | 0 | 1 | 1 | 1 |
| PCSG00476 | train | 0 | 0 | 3 | 0 | 0 | 3 | 3 |
| PCSG00926 | train | 0 | 0 | 0 | 1 | 0 | 1 | 1 |
| all | | 150 | 355 | 113 | 1 | 1 | 472 | 270 |
<!-- /generated:v2_rules -->

- **What it meant for v1.** Five of these titles are v1 titles. 76 of their records were already outside v1's pools
  (outside the vocab). 4 were in pools v1's training pool with a wrong label: 3 NPJH50382 lines with ≪≫ and the
  PCSG00367 line. Every VOCR6 model so far trained on them.
- **Readings that only looked wrong.** PCSG00853 draws 『』 where most titles draw 「」; the labels are right
  (VOCR6.2 reads them as 「」: a recognizer error). A PCSG01202 line with a dashed outline reads like ま at sheet size;
  at 10x it is よ, as labelled.
- **After the fixes.** 358 of the 360 audited records are in pools v2.1 with the same crop and label, and all are
  right; the 2 wrong ones are dropped.

## Splits (`splits_v2.json`)

The rule is v1's, per run, plus the cross-run guard. Test titles are exactly the two that `index.json` marks `test`.

- **hval:** 3,010 pages (3.66%), for screening only; final models train on it.
- **guard:** 3,780 pages (4.60%), the bands around hval and the cross-run guard; back in final models.
- **test:** 1,040 pages (1.27%), the owner's cap is 10%.
- While screening, hval, guard and test hold out 9.53% of the pages. Final models hold out the test titles only.

## Corpus v1 against v2.1 (`hooked_audit.py stats`, `data/manifests/vocr6_hooked_v2/`)

<!-- generated:v2_compare -->
| | Vita, v1 | Vita, v2.1 | PSP, v1 | PSP, v2.1 | all, v1 | all, v2.1 | added |
|---|---:|---:|---:|---:|---:|---:|---:|
| titles | 33 | 119 | 16 | 19 | 49 | 138 | +89 |
| runs | 33 | 147 | 16 | 19 | 49 | 166 | +117 |
| pages | 34,463 | 58,791 | 21,030 | 23,394 | 55,493 | 82,185 | +26,692 |
| test pages | 690 | 690 | 0 | 350 | 690 | 1,040 | +350 |
| records: dialogue lines | 56,254 | 94,120 | 32,254 | 35,783 | 88,508 | 129,903 | +41,395 |
| records: vertical columns | 1,611 | 5,658 | 0 | 0 | 1,611 | 5,658 | +4,047 |
| records: speaker names | 4,612 | 6,078 | 3,923 | 4,095 | 8,535 | 10,173 | +1,638 |
| records: ruby readings | 352 | 1,599 | 65 | 91 | 417 | 1,690 | +1,273 |
| label in doubt (dropped) | 5 | 368 | 5 | 115 | 10 | 483 | +473 |
| usable: dialogue lines | 53,541 | 85,146 | 28,436 | 31,841 | 81,977 | 116,987 | +35,010 |
| usable dialogue characters | 847,386 | 1,356,323 | 445,707 | 489,375 | 1,293,093 | 1,845,698 | +552,605 |
| usable: vertical columns | 1,567 | 5,151 | 0 | 0 | 1,567 | 5,151 | +3,584 |
| usable: speaker names | 359 | 531 | 320 | 341 | 679 | 872 | +193 |
| usable: ruby readings | 254 | 933 | 48 | 64 | 302 | 997 | +695 |
| usable: PSP lines at 2x | 0 | 0 | 28,453 | 31,858 | 28,453 | 31,858 | +3,405 |
<!-- /generated:v2_compare -->

**Kanji coverage.** <!-- generated:v2_coverage -->
Of the 5,142 kanji of `data/vocab_vn_5631.txt`, 2,868 occur on the training side (train + guard + hval): 934 at least 80 times, 1,012 10-79 times and 922 1-9 times. 2,274 never occur. The kept rows hold 3,160 distinct characters (2,872 kanji). 15 characters outside the vocab removed 21 records; 10 kept characters are outside Small's 4,987-class vocab, which trains them as `<unk>`.
<!-- /generated:v2_coverage -->

v1's training side held 2,587 of the 5,142 vocab kanji; v2.1's holds 2,868 (+281).

Per title. "seen / overlaps" lists a title whose VN is in the protected family (vita-test, psp-test, the CLANNAD
family, their shared releases), in #119's development units, in textspot7/8's confirmation set, related to a hooked
test title, or in VOCR5's fine-tune pools:

<!-- generated:v2_titles -->
| title | platform | split | VN | runs | pages v1 → v2.1 (hval) | usable dialogue lines v1 → v2.1 | dialogue chars | dropped as in doubt | seen / overlaps |
|---|---|---|---|---:|---|---|---:|---:|---|
| NPJG00016 | psp | train | v14909 | 1 | 271 → 271 (11) | 446 → 446 | 3,660 | 0 | VOCR5 pools (1,224 rows) |
| NPJH00140 | psp | train | v7301 | 1 | 1,640 → 1,640 (66) | 2,885 → 2,885 | 38,165 | 0 | VOCR5 pools (762 rows) |
| NPJH50001 | psp | train | v2063 | 1 | 1,602 → 1,602 (64) | 1,418 → 1,418 | 24,932 | 5 | VOCR5 pools (35 rows) |
| NPJH50002 | psp | train | v21259 | 1 | 1,727 → 1,727 (69) | 2,174 → 2,174 | 36,398 | 1 |  |
| NPJH50127 | psp | train | v3360 | 1 | 58 → 58 (0) | 0 → 0 | 0 | 0 | VOCR5 pools (8 rows) |
| NPJH50132 | psp | train | v1171 | 1 | 1,851 → 1,851 (74) | 2,044 → 2,044 | 35,257 | 0 | VOCR5 pools (813 rows) |
| NPJH50279 | psp | train | v4091 | 1 | 1,004 → 1,004 (40) | 1,243 → 1,243 | 20,624 | 3 | VOCR5 pools (39 rows) |
| NPJH50297 | psp | train | v1322 | 1 | 1,177 → 1,177 (47) | 1,872 → 1,872 | 29,370 | 0 | VOCR5 pools (1,064 rows) |
| NPJH50356 | psp | train | v398 | 1 | 1,999 → 1,999 (80) | 2,530 → 2,530 | 36,182 | 0 | VOCR5 pools (46 rows) |
| NPJH50382 | psp | train | v1646 | 1 | 1,965 → 1,965 (79) | 2,254 → 2,251 | 37,201 | 73 | VOCR5 pools (840 rows) |
| NPJH50551 | psp | train | v22505 | 1 | 214 → 214 (9) | 358 → 358 | 7,289 | 0 | VOCR5 pools (36 rows) |
| NPJH50591 | psp | train | v6173 | 1 | new: 1,664 (67) | new: 2,313 | 29,738 | 0 | VOCR5 pools (54 rows) |
| NPJH50609 | psp | train | v1805 | 1 | new: 350 (14) | new: 706 | 9,293 | 0 | VOCR5 pools (24 rows) |
| NPJH50621 | psp | train | v57 | 1 | 1,504 → 1,504 (60) | 1,922 → 1,922 | 28,333 | 2 | VOCR5 pools (159 rows) |
| NPJH50641 | psp | train | v7018 | 1 | 1,130 → 1,130 (45) | 1,711 → 1,711 | 31,123 | 0 | VOCR5 pools (821 rows) |
| NPJH50649 | psp | train | v10623 | 1 | 1,064 → 1,064 (43) | 1,932 → 1,932 | 21,706 | 0 | VOCR5 pools (1,493 rows) |
| NPJH50882 | psp | test | v3 | 1 | new: 350 (0) | new: 389 | 4,666 | 31 | protected family |
| ULJM05383 | psp | train | v34 | 1 | 1,827 → 1,827 (73) | 2,712 → 2,712 | 48,104 | 0 | protected family; VOCR5 pools (1,564 rows) |
| ULJM05674 | psp | train | v10 | 1 | 1,997 → 1,997 (80) | 2,935 → 2,935 | 47,334 | 0 | VOCR5 pools (698 rows) |
| PCSG00007 | vita | train | v5 | 1 | 562 → 562 (22) | 657 → 657 | 10,333 | 0 | protected family |
| PCSG00020 | vita | train | v7809 | 1 | new: 51 (0) | new: 47 | 493 | 0 | VOCR5 pools (378 rows) |
| PCSG00062 | vita | train | v10856 | 1 | new: 17 (0) | new: 23 | 365 | 0 | protected family |
| PCSG00122 | vita | train | v11 | 1 | new: 922 (37) | new: 452 | 6,669 | 355 |  |
| PCSG00139 | vita | train | v11000 | 2 | new: 122 (0) | new: 207 | 3,109 | 0 | VOCR5 pools (13 rows) |
| PCSG00146 | vita | train | v6618 | 1 | 1,498 → 1,498 (60) | 1,878 → 1,878 | 36,739 | 0 | VOCR5 pools (524 rows) |
| PCSG00147 | vita | train | v2002 | 1 | new: 56 (0) | new: 74 | 1,486 | 0 | VOCR5 pools (812 rows) |
| PCSG00172 | vita | train | v5844 | 1 | new: 1,111 (44) | new: 0 | 0 | 0 |  |
| PCSG00183 | vita | train | v2089 | 1 | new: 12 (0) | new: 17 | 243 | 0 | VOCR5 pools (41 rows) |
| PCSG00206 | vita | train | v5154 | 1 | new: 3,292 (132) | new: 5,740 | 98,811 | 0 | VOCR5 pools (11 rows) |
| PCSG00250 | vita | train | v11660 | 1 | new: 60 (0) | new: 76 | 1,467 | 0 | VOCR5 pools (135 rows) |
| PCSG00272 | vita | train | v9846 | 1 | new: 60 (0) | new: 122 | 1,312 | 0 | VOCR5 pools (27 rows) |
| PCSG00291 | vita | train | v1281 | 1 | new: 53 (0) | new: 78 | 1,139 | 0 | VOCR5 pools (823 rows) |
| PCSG00352 | vita | train | v5883 | 1 | 585 → 585 (23) | 809 → 809 | 16,216 | 0 | VOCR5 pools (287 rows) |
| PCSG00367 | vita | train | v15118 | 2 | 1,165 → 1,265 (47) | 1,977 → 1,977 | 29,189 | 1 |  |
| PCSG00370 | vita | train | v5297 | 2 | new: 110 (0) | new: 140 | 2,075 | 0 | VOCR5 pools (36 rows) |
| PCSG00380 | vita | train | v18456 | 1 | 1,095 → 1,095 (44) | 2,071 → 2,071 | 35,430 | 0 |  |
| PCSG00382 | vita | train | v14813 | 1 | new: 26 (0) | new: 43 | 526 | 0 | VOCR5 pools (47 rows) |
| PCSG00410 | vita | train | v15253 | 2 | new: 784 (27) | new: 0 | 0 | 0 |  |
| PCSG00420 | vita | train | v5916 | 2 | new: 675 (23) | new: 904 | 14,846 | 0 | VOCR5 pools (32 rows) |
| PCSG00428 | vita | train | v14917 | 1 | new: 100 (0) | new: 145 | 2,085 | 0 | VOCR5 pools (46 rows) |
| PCSG00448 | vita | train | v14652 | 1 | new: 596 (24) | new: 1,183 | 13,239 | 0 | VOCR5 pools (78 rows) |
| PCSG00450 | vita | train | v17739 | 1 | new: 78 (0) | new: 133 | 2,537 | 0 |  |
| PCSG00451 | vita | train | v22613 | 1 | new: 64 (0) | new: 129 | 2,345 | 0 |  |
| PCSG00459 | vita | train | v14267 | 1 | new: 51 (0) | new: 85 | 1,430 | 0 | dev units |
| PCSG00468 | vita | train | v15771 | 1 | 412 → 412 (16) | 678 → 678 | 11,263 | 0 | VOCR5 pools (12 rows) |
| PCSG00469 | vita | train | v6723 | 1 | 2,047 → 2,047 (82) | 3,866 → 3,866 | 53,794 | 0 | VOCR5 pools (20 rows) |
| PCSG00472 | vita | train | v15448 | 1 | new: 20 (0) | new: 48 | 549 | 0 |  |
| PCSG00476 | vita | train | v12150 | 1 | new: 21 (0) | new: 42 | 560 | 3 | confirmation set |
| PCSG00477 | vita | train | v15773 | 2 | new: 787 (27) | new: 1,106 | 15,858 | 0 | VOCR5 pools (695 rows) |
| PCSG00505 | vita | train | v12480 | 1 | new: 100 (0) | new: 117 | 1,614 | 0 | VOCR5 pools (44 rows) |
| PCSG00509 | vita | train | v16049 | 1 | 1,299 → 1,299 (52) | 1,867 → 1,867 | 32,491 | 0 | VOCR5 pools (27 rows) |
| PCSG00510 | vita | train | v2021 | 2 | new: 552 (18) | new: 689 | 9,882 | 0 | confirmation set; VOCR5 pools (31 rows) |
| PCSG00530 | vita | train | v14649 | 1 | new: 8 (0) | new: 14 | 146 | 0 |  |
| PCSG00540 | vita | train | v12260 | 1 | 1,037 → 1,037 (41) | 1,099 → 1,099 | 17,295 | 1 | VOCR5 pools (9 rows) |
| PCSG00547 | vita | train | v17285 | 1 | 1,021 → 1,021 (41) | 1,499 → 1,499 | 24,009 | 0 | VOCR5 pools (630 rows) |
| PCSG00553 | vita | train | v11472 | 1 | 1,129 → 1,129 (45) | 2,156 → 2,156 | 29,000 | 0 | VOCR5 pools (69 rows) |
| PCSG00555 | vita | train | v14018 | 1 | 394 → 394 (16) | 574 → 574 | 11,837 | 0 | VOCR5 pools (1,056 rows) |
| PCSG00567 | vita | train | v1377 | 1 | new: 100 (0) | new: 106 | 1,606 | 0 | VOCR5 pools (18 rows) |
| PCSG00576 | vita | train | v12188 | 2 | 1,219 → 1,319 (49) | 2,231 → 2,231 | 31,135 | 0 | dev units |
| PCSG00591 | vita | train | v16467 | 1 | new: 100 (0) | new: 134 | 2,479 | 0 | VOCR5 pools (5 rows) |
| PCSG00602 | vita | train | v12751 | 1 | new: 33 (0) | new: 54 | 624 | 0 | VOCR5 pools (32 rows) |
| PCSG00611 | vita | train | v15248 | 2 | 1,382 → 1,482 (55) | 1,725 → 1,730 | 34,269 | 0 |  |
| PCSG00620 | vita | train | v14973 | 1 | 995 → 995 (40) | 1,795 → 1,795 | 31,512 | 0 | VOCR5 pools (31 rows) |
| PCSG00628 | vita | train | v11301 | 1 | 1,208 → 1,208 (48) | 1,860 → 1,860 | 29,036 | 3 | VOCR5 pools (15 rows) |
| PCSG00629 | vita | train | v1715 | 1 | new: 20 (0) | new: 31 | 489 | 0 |  |
| PCSG00638 | vita | train | v2835 | 1 | new: 19 (0) | new: 0 | 0 | 0 | dev units |
| PCSG00648 | vita | train | v10680 | 1 | 1,393 → 1,393 (56) | 2,041 → 2,041 | 46,051 | 0 | VOCR5 pools (1,259 rows) |
| PCSG00653 | vita | train | v17343 | 2 | 863 → 963 (35) | 0 → 0 | 0 | 0 | VOCR5 pools (2 rows) |
| PCSG00666 | vita | train | v14759 | 2 | 1,583 → 1,683 (63) | 2,561 → 2,715 | 35,020 | 0 | VOCR5 pools (33 rows) |
| PCSG00667 | vita | train | v17608 | 1 | 2,000 → 2,000 (80) | 3,074 → 3,074 | 46,608 | 0 | VOCR5 pools (30 rows) |
| PCSG00673 | vita | train | v17102 | 1 | new: 101 (0) | new: 126 | 2,413 | 0 | VOCR5 pools (259 rows) |
| PCSG00696 | vita | train | v28235 | 1 | new: 135 (0) | new: 20 | 245 | 0 | VOCR5 pools (57 rows) |
| PCSG00706 | vita | train | v16770 | 1 | new: 42 (0) | new: 64 | 765 | 0 | dev units |
| PCSG00713 | vita | train | v17287 | 1 | 1,463 → 1,463 (59) | 2,699 → 2,699 | 36,319 | 0 | VOCR5 pools (24 rows) |
| PCSG00739 | vita | train | v15395 | 1 | new: 101 (0) | new: 179 | 2,921 | 0 | VOCR5 pools (8 rows) |
| PCSG00745 | vita | train | v4719 | 1 | new: 503 (20) | new: 996 | 12,345 | 0 | VOCR5 pools (28 rows) |
| PCSG00751 | vita | train | v8203 | 2 | 240 → 340 (10) | 210 → 212 | 1,946 | 2 | VOCR5 pools (24 rows) |
| PCSG00753 | vita | train | v8243 | 2 | 1,000 → 1,094 (40) | 1,310 → 1,364 | 26,553 | 0 | VOCR5 pools (33 rows) |
| PCSG00755 | vita | train | v7557 | 1 | new: 44 (0) | new: 81 | 1,204 | 0 | VOCR5 pools (47 rows) |
| PCSG00766 | vita | train | v18163 | 1 | new: 31 (0) | new: 44 | 680 | 0 | dev units |
| PCSG00768 | vita | train | v13833 | 2 | new: 130 (0) | new: 0 | 0 | 0 | dev units |
| PCSG00769 | vita | train | v19140 | 1 | new: 37 (0) | new: 61 | 846 | 0 |  |
| PCSG00776 | vita | train | v93 | 2 | new: 468 (15) | new: 491 | 7,993 | 0 | VOCR5 pools (14 rows) |
| PCSG00777 | vita | train | v92 | 1 | new: 22 (0) | new: 25 | 319 | 0 | VOCR5 pools (17 rows) |
| PCSG00785 | vita | train | v12438 | 1 | new: 25 (0) | new: 35 | 545 | 0 |  |
| PCSG00787 | vita | train | v18159 | 1 | new: 667 (27) | new: 964 | 12,582 | 0 |  |
| PCSG00794 | vita | train | v11469 | 2 | new: 41 (0) | new: 51 | 813 | 0 | VOCR5 pools (41 rows) |
| PCSG00805 | vita | train | v18314 | 2 | 1,329 → 1,429 (53) | 2,365 → 2,397 | 33,397 | 0 | VOCR5 pools (37 rows) |
| PCSG00807 | vita | test | v18432 | 1 | 690 → 690 (0) | 1,152 → 1,152 | 13,911 | 0 | dev units |
| PCSG00808 | vita | train | v19012 | 1 | new: 488 (20) | new: 836 | 10,756 | 0 | VOCR5 pools (36 rows) |
| PCSG00817 | vita | train | v147 | 1 | new: 685 (27) | new: 643 | 11,258 | 0 | VOCR5 pools (10 rows) |
| PCSG00826 | vita | train | v17345 | 1 | new: 11 (0) | new: 18 | 210 | 0 |  |
| PCSG00829 | vita | train | v17750 | 1 | new: 100 (0) | new: 165 | 2,186 | 0 | protected family |
| PCSG00845 | vita | train | v16134 | 1 | new: 1,507 (60) | new: 3,203 | 70,598 | 0 | VOCR5 pools (66 rows) |
| PCSG00852 | vita | train | v18838 | 1 | new: 396 (16) | new: 584 | 10,454 | 0 | VOCR5 pools (49 rows) |
| PCSG00853 | vita | train | v17640 | 1 | new: 833 (33) | new: 1,227 | 16,741 | 0 |  |
| PCSG00854 | vita | train | v18833 | 2 | 988 → 1,088 (40) | 1,643 → 1,697 | 27,576 | 0 |  |
| PCSG00866 | vita | train | v18160 | 2 | 513 → 613 (21) | 897 → 899 | 13,283 | 0 | protected family |
| PCSG00876 | vita | train | v18162 | 1 | 1,049 → 1,049 (42) | 1,747 → 1,747 | 28,057 | 0 | VOCR5 pools (31 rows) |
| PCSG00883 | vita | train | v3112, v7809 | 1 | new: 32 (0) | new: 28 | 288 | 0 | VOCR5 pools (378 rows) |
| PCSG00903 | vita | train | v19717 | 1 | new: 207 (8) | new: 391 | 5,253 | 0 | VOCR5 pools (60 rows) |
| PCSG00911 | vita | train | v15070 | 2 | new: 692 (24) | new: 959 | 14,809 | 0 | VOCR5 pools (8 rows) |
| PCSG00912 | vita | train | v20117 | 1 | new: 33 (0) | new: 51 | 819 | 0 |  |
| PCSG00914 | vita | train | v4602 | 1 | 515 → 515 (21) | 1,064 → 1,064 | 13,611 | 0 |  |
| PCSG00916 | vita | train | v2022, v5854 | 1 | new: 1,001 (40) | new: 1,581 | 23,275 | 0 | VOCR5 pools (96 rows) |
| PCSG00917 | vita | train | v2023 | 2 | 1,564 → 1,664 (63) | 2,373 → 2,374 | 39,012 | 0 | VOCR5 pools (72 rows) |
| PCSG00918 | vita | train | v2024 | 2 | 1,500 → 1,600 (60) | 2,421 → 2,421 | 39,459 | 0 | VOCR5 pools (68 rows) |
| PCSG00926 | vita | train | v14265 | 1 | new: 57 (0) | new: 0 | 0 | 1 | VOCR5 pools (36 rows) |
| PCSG00935 | vita | train | v8095, v12775 | 1 | 382 → 382 (15) | 758 → 758 | 10,986 | 0 | VOCR5 pools (40 rows) |
| PCSG00938 | vita | train | v4542 | 2 | new: 431 (13) | new: 603 | 8,725 | 0 | confirmation set |
| PCSG00981 | vita | train | v18152 | 1 | new: 101 (0) | new: 163 | 2,584 | 0 | VOCR5 pools (46 rows) |
| PCSG00988 | vita | train | v15080 | 1 | new: 25 (0) | new: 41 | 678 | 0 | VOCR5 pools (11 rows) |
| PCSG00991 | vita | train | v12928 | 1 | new: 66 (0) | new: 98 | 1,473 | 0 | confirmation set |
| PCSG01000 | vita | train | v18498 | 1 | new: 39 (0) | new: 46 | 623 | 0 | VOCR5 pools (27 rows) |
| PCSG01008 | vita | train | v20747 | 2 | new: 313 (9) | new: 478 | 5,831 | 0 | VOCR5 pools (40 rows) |
| PCSG01019 | vita | train | v18295 | 1 | new: 644 (26) | new: 921 | 16,900 | 0 | VOCR5 pools (25 rows) |
| PCSG01023 | vita | train | v20577 | 1 | new: 963 (39) | new: 1,179 | 23,213 | 0 | VOCR5 pools (108 rows) |
| PCSG01046 | vita | train | v20634 | 2 | new: 461 (14) | new: 664 | 9,107 | 0 | VOCR5 pools (12 rows) |
| PCSG01065 | vita | train | v20958 | 1 | new: 22 (0) | new: 51 | 838 | 0 |  |
| PCSG01075 | vita | train | v21140 | 1 | new: 333 (13) | new: 662 | 9,463 | 0 | VOCR5 pools (11 rows) |
| PCSG01110 | vita | train | v21521 | 2 | 343 → 444 (14) | 484 → 510 | 6,838 | 0 |  |
| PCSG01127 | vita | train | v15708 | 2 | new: 138 (0) | new: 153 | 2,358 | 0 | dev units |
| PCSG01152 | vita | train | v22114 | 1 | new: 91 (0) | new: 196 | 2,688 | 0 | related to the test title |
| PCSG01167 | vita | train | v18291 | 1 | new: 108 (0) | new: 169 | 3,960 | 0 | VOCR5 pools (26 rows) |
| PCSG01180 | vita | train | v21784 | 1 | new: 8 (0) | new: 15 | 206 | 0 |  |
| PCSG01187 | vita | train | v19545 | 1 | new: 70 (0) | new: 120 | 1,943 | 0 | VOCR5 pools (4 rows) |
| PCSG01194 | vita | train | v12096 | 1 | new: 30 (0) | new: 45 | 707 | 0 | VOCR5 pools (32 rows) |
| PCSG01196 | vita | train | v1399 | 1 | new: 10 (0) | new: 10 | 134 | 0 | VOCR5 pools (6 rows) |
| PCSG01197 | vita | train | v18695 | 1 | new: 87 (0) | new: 126 | 1,972 | 0 | VOCR5 pools (44 rows) |
| PCSG01202 | vita | train | v18842 | 1 | new: 104 (0) | new: 113 | 1,858 | 1 | VOCR5 pools (44 rows) |
| PCSG01221 | vita | train | v1940 | 1 | new: 111 (0) | new: 159 | 2,521 | 0 | VOCR5 pools (51 rows) |
| PCSG01245 | vita | train | v24291 | 1 | new: 29 (0) | new: 11 | 106 | 0 | VOCR5 pools (42 rows) |
| PCSG01247 | vita | train | v1892 | 1 | new: 83 (0) | new: 107 | 1,846 | 0 | VOCR5 pools (28 rows) |
| PCSG01250 | vita | train | v19444 | 1 | new: 22 (0) | new: 26 | 486 | 1 | VOCR5 pools (19 rows) |
| PCSG01253 | vita | train | v22045 | 1 | new: 101 (0) | new: 122 | 2,083 | 0 | VOCR5 pools (12 rows) |
| PCSG01260 | vita | train | v20314 | 1 | new: 86 (0) | new: 0 | 0 | 0 | VOCR5 pools (20 rows) |
| PCSG01268 | vita | train | v22311 | 2 | new: 137 (0) | new: 191 | 2,781 | 0 | VOCR5 pools (10 rows) |
| PCSG01282 | vita | train | v6656 | 1 | new: 31 (0) | new: 44 | 762 | 0 | dev units |
<!-- /generated:v2_titles -->

## Overlaps with the protected and development sets (`hooked_audit.py overlap`, `overlap_v2.json`)

The split policy (`prompts/vocr6-common.md`) trains on every `train` title. These overlaps change no split; every
report of a model trained on v2.1 has to carry them.

- **The new test title** NPJH50882 (うたわれるもの PORTABLE, v3) shares 7 releases with vita-test's Utawarerumono:
  Itsuwari no Kamen (v7721). It is a test title, so nothing trains on it; this baseline does not read it.
- **New seen flags** (train titles whose VN is in a protected set):
  - PCSG00062 (タイムトラベラーズ, v10856) is a vita-test VN (2 lines) and a psp-test VN (13 lines).
  - PCSG00829 (喧嘩番長乙女, v17750) is a vita-test VN (16 lines).
  - v1's flags stand: PCSG00007 and ULJM05383 share CLANNAD's Key 10th Memorial BOX; PCSG00866 shares releases with
    vita-test's Collar x Malice -Unlimited-.
- **#119's development units** (the unseen-game dev sets): 7 new titles' VNs are in them (v14267, v2835, v16770,
  v18163, v13833, v15708, v6656: 41 base units, 218 fixed lines). Two of them (PCSG00638, PCSG00768) have pages but
  no verified line. A model trained on v2.1 has seen these VNs: VOCR6 dev comparisons have to leave them out, as they
  left out v12188 and v18432 for v1.
- **textspot7/8's confirmation set** (41 VNs): 4 are now hooked titles (v12150, v2021, v4542, v12928: 17 of its 172
  base regions).
- **The test title's fandisc.** PCSG01152 (Dance with Devils My Carol, v22114) is a VNDB fandisc of the test title's
  VN (v18432) and shares a release with it. It trains (another VN). Test-title scores of models trained on it are
  not same-series-unseen; the two titles share 1 distinct verified line label.
- **VOCR5's exposure.** 97 of the 138 titles' VNs had rows in VOCR5's fine-tune pools (59 of the 89 new ones).

## Baseline: VOCR6.2 as Rubi 0.10.5 runs it

**What was read** (`eval/pagepath.py run --heads`, pre-registered in `vocr6/experiments.jsonl` before the read):

- the shipped blobs `VOCR6.2_S_w8.vocr`, `VOCR6.2_N_w8g2.vocr` and `VOCR6.2_L_w8_2l.vocr`, W8I through the C runtime,
  each with its own head sections (the head gate and the two-level head where it has them) and Rubi's ungated re-read;
- Rubi main 3d415cd's line read: the counter re-decode, beam 8 with the 1 MB game-text LM `vocr_lm6_o3_1000k.vlm`
  (LM weight 0.6, bonus 2.5 nats, the start-of-line context), the cascade (a line under confidence 0.7 is read again
  by the next tier up), `core/textnorm.c` and the Shell's non-Japanese line filter;
- three ways of finding the lines in a region:
  - the textspot8 spotter's crop decode (`textspot8_t64n8c16_w8.vtsp`, Rubi's default finder);
  - Rubi's GRAD path (`rubi,fix=1`, its fallback);
  - each reference line's own box plus its pad (clean crops: the recognizer's floor on the page path).
- **Two sets.** hval: every hval page's regions (base, +4 and +8 px), all titles, split into the titles new in v2
  (VOCR6.2 never trained on them) and pools v1's titles (VOCR6.2 trained on their train, guard and hval pages). And
  all pages of the new train titles (train, guard and hval; base regions): 21,869 regions of 82 titles that no VOCR6
  model has seen.
- **Not read:** the test titles, vita-test, psp-test, CLANNAD and the dev sets.
- **Intervals:** 95% bootstraps over titles (10,000 resamples, seed 0); over pages for a group of fewer than 5 titles
  (the new PSP titles: 2).
- The `--heads` option is new in `eval/pagepath.py`: without it every run reads the blobs plain, as before.

hval (dialogue CER):

<!-- generated:v2_baseline_hval -->
| finder | tier | new titles, Vita | new titles, PSP | v1 titles, Vita | v1 titles, PSP | all hval |
|---|---|---|---|---|---|---|
| textspot8 spotter (Rubi's default) | S | 1.56% [0.96, 2.46] | 1.80% [0.78, 3.16] | 1.57% [0.89, 2.50] | 0.89% [0.57, 1.31] | 1.40% [1.02, 1.87] |
| textspot8 spotter (Rubi's default) | N | 1.38% [0.86, 2.08] | 1.56% [0.55, 2.90] | 1.69% [0.83, 2.93] | 0.76% [0.48, 1.10] | 1.37% [0.93, 1.96] |
| textspot8 spotter (Rubi's default) | L | 1.34% [0.83, 1.99] | 1.39% [0.40, 2.72] | 1.69% [0.83, 2.92] | 0.70% [0.43, 1.03] | 1.34% [0.90, 1.93] |
| textspot8, filter lines+punct | N | 0.83% [0.45, 1.17] | 0.57% [0.22, 0.98] | 1.08% [0.39, 2.31] | 0.55% [0.32, 0.85] | 0.87% [0.51, 1.44] |
| textspot8, line filter off | N | 0.83% [0.45, 1.17] | 0.57% [0.22, 0.98] | 2.42% [0.36, 6.81] | 0.55% [0.32, 0.85] | 1.48% [0.50, 3.48] |
| Rubi's GRAD path (`rubi,fix=1`) | S | 2.27% [1.17, 3.41] | 3.38% [1.91, 5.17] | 1.33% [0.74, 2.16] | 2.12% [0.81, 4.44] | 1.82% [1.22, 2.58] |
| Rubi's GRAD path (`rubi,fix=1`) | N | 2.09% [1.04, 3.16] | 2.74% [1.32, 4.50] | 1.40% [0.75, 2.28] | 1.81% [0.59, 3.97] | 1.71% [1.13, 2.44] |
| Rubi's GRAD path (`rubi,fix=1`) | L | 2.04% [0.99, 3.10] | 2.55% [1.16, 4.28] | 1.40% [0.75, 2.28] | 1.82% [0.56, 4.05] | 1.70% [1.10, 2.44] |
| clean crops (each line's own box) | S | 0.94% [0.47, 1.93] | 1.42% [0.48, 2.70] | 0.96% [0.46, 1.69] | 0.51% [0.24, 0.85] | 0.85% [0.55, 1.23] |
| clean crops (each line's own box) | N | 0.89% [0.41, 1.82] | 1.13% [0.21, 2.41] | 0.91% [0.41, 1.65] | 0.38% [0.18, 0.66] | 0.77% [0.48, 1.16] |
| clean crops (each line's own box) | L | 0.79% [0.35, 1.71] | 0.99% [0.07, 2.25] | 0.91% [0.40, 1.66] | 0.35% [0.15, 0.62] | 0.74% [0.45, 1.12] |
| clean crops, line filter off | S | 0.53% [0.25, 1.07] | 0.50% [0.17, 0.87] | 0.35% [0.10, 0.77] | 0.28% [0.15, 0.47] | 0.38% [0.23, 0.60] |
| clean crops, line filter off | N | 0.39% [0.16, 0.81] | 0.21% [0.00, 0.48] | 0.31% [0.06, 0.73] | 0.17% [0.08, 0.32] | 0.30% [0.15, 0.50] |
| clean crops, line filter off | L | 0.32% [0.12, 0.72] | 0.07% [0.00, 0.22] | 0.34% [0.05, 0.79] | 0.13% [0.05, 0.28] | 0.27% [0.12, 0.49] |
<!-- /generated:v2_baseline_hval -->

Every page of the new train titles:

<!-- generated:v2_baseline_new -->
| finder | tier | Vita | PSP | all |
|---|---|---|---|---|
| textspot8 spotter (Rubi's default) | S | 1.26% [1.04, 1.63] | 1.97% [1.66, 2.32] | 1.31% [1.08, 1.67] |
| textspot8 spotter (Rubi's default) | N | 1.16% [0.93, 1.50] | 1.76% [1.48, 2.07] | 1.19% [0.98, 1.53] |
| textspot8 spotter (Rubi's default) | L | 1.12% [0.90, 1.46] | 1.69% [1.41, 2.00] | 1.16% [0.94, 1.48] |
| textspot8, filter lines+punct | N | 0.73% [0.55, 0.94] | 1.07% [0.84, 1.32] | 0.75% [0.58, 0.94] |
| textspot8, line filter off | N | 0.69% [0.50, 0.90] | 1.04% [0.82, 1.30] | 0.72% [0.54, 0.91] |
| clean crops (each line's own box) | S | 0.86% [0.68, 1.20] | 1.57% [1.31, 1.86] | 0.91% [0.71, 1.25] |
| clean crops (each line's own box) | N | 0.77% [0.60, 1.08] | 1.27% [1.06, 1.50] | 0.81% [0.61, 1.11] |
| clean crops (each line's own box) | L | 0.70% [0.52, 1.02] | 1.22% [1.01, 1.45] | 0.74% [0.55, 1.05] |
| clean crops, line filter off | S | 0.42% [0.31, 0.56] | 0.82% [0.64, 1.04] | 0.45% [0.33, 0.59] |
| clean crops, line filter off | N | 0.32% [0.22, 0.43] | 0.54% [0.42, 0.67] | 0.34% [0.23, 0.44] |
| clean crops, line filter off | L | 0.25% [0.19, 0.34] | 0.49% [0.37, 0.63] | 0.27% [0.20, 0.36] |
<!-- /generated:v2_baseline_new -->

Per new title (Normal, cascade Large), sorted by the page path's errors:

<!-- generated:v2_baseline_titles -->
| title | page path, textspot8 (N) | clean crops (N) | reference chars | units |
|---|---:|---:|---:|---:|
| PCSG00845 | 1.15% | 0.32% | 95,785 | 1,591 |
| PCSG00206 | 0.68% | 0.61% | 99,032 | 3,291 |
| NPJH50591 | 1.71% | 1.45% | 29,863 | 1,648 |
| PCSG00122 | 9.19% | 7.02% | 4,875 | 230 |
| PCSG00853 | 2.01% | 1.64% | 17,054 | 835 |
| PCSG00808 | 2.10% | 1.83% | 10,851 | 485 |
| PCSG00477 | 1.23% | 1.02% | 18,477 | 787 |
| PCSG01202 | 11.18% | 7.92% | 1,843 | 98 |
| PCSG01023 | 0.86% | 0.64% | 23,480 | 939 |
| PCSG00776 | 1.15% | 0.77% | 16,243 | 466 |
| PCSG00448 | 1.38% | 0.99% | 13,379 | 596 |
| NPJH50609 | 1.92% | 0.70% | 9,326 | 345 |
| PCSG01046 | 1.46% | 0.62% | 12,109 | 477 |
| PCSG00938 | 1.28% | 0.50% | 11,320 | 431 |
| PCSG00911 | 0.78% | 0.67% | 17,392 | 692 |
| PCSG00567 | 8.07% | 6.45% | 1,660 | 96 |
| PCSG00787 | 1.02% | 0.54% | 12,870 | 667 |
| PCSG00817 | 1.02% | 0.86% | 12,364 | 675 |
| PCSG00916 | 0.49% | 0.38% | 23,596 | 1,001 |
| PCSG01075 | 1.10% | 0.70% | 9,463 | 333 |
| PCSG00852 | 0.90% | 0.59% | 10,462 | 395 |
| PCSG01247 | 4.79% | 4.74% | 1,877 | 83 |
| PCSG01019 | 0.46% | 0.47% | 17,561 | 644 |
| PCSG00510 | 0.65% | 0.63% | 12,102 | 552 |
| PCSG00420 | 0.43% | 0.46% | 17,301 | 657 |
| PCSG01253 | 3.46% | 3.28% | 2,137 | 95 |
| PCSG01268 | 1.99% | 1.34% | 3,420 | 137 |
| PCSG00991 | 4.34% | 4.07% | 1,499 | 66 |
| PCSG01127 | 1.89% | 1.28% | 3,125 | 130 |
| PCSG00020 | 10.07% | 9.53% | 556 | 48 |
| PCSG00745 | 0.37% | 0.28% | 12,417 | 490 |
| PCSG00602 | 6.97% | 6.81% | 646 | 33 |
| PCSG00903 | 0.83% | 0.34% | 5,276 | 207 |
| PCSG01152 | 1.60% | 0.15% | 2,753 | 93 |
| PCSG00981 | 1.51% | 0.54% | 2,586 | 101 |
| PCSG01197 | 1.92% | 1.72% | 1,978 | 87 |
| PCSG00673 | 1.49% | 1.07% | 2,419 | 101 |
| PCSG01008 | 0.44% | 0.34% | 8,244 | 313 |
| PCSG01187 | 1.75% | 0.15% | 1,945 | 70 |
| PCSG00428 | 1.48% | 0.91% | 2,089 | 100 |
| PCSG00451 | 1.28% | 0.47% | 2,345 | 64 |
| PCSG00450 | 1.10% | 1.06% | 2,545 | 77 |
| PCSG00739 | 0.96% | 0.03% | 2,921 | 101 |
| PCSG00769 | 3.58% | 2.62% | 726 | 30 |
| PCSG00988 | 3.71% | 4.00% | 700 | 25 |
| PCSG00250 | 1.48% | 0.47% | 1,491 | 66 |
| PCSG00706 | 2.75% | 0.65% | 765 | 42 |
| PCSG00883 | 6.09% | 6.09% | 312 | 31 |
| PCSG00291 | 1.57% | 1.05% | 1,143 | 53 |
| PCSG00459 | 1.26% | 0.42% | 1,430 | 51 |
| PCSG00476 | 1.65% | 0.87% | 1,030 | 18 |
| PCSG00766 | 2.48% | 2.48% | 686 | 31 |
| PCSG01167 | 0.40% | 0.23% | 3,960 | 108 |
| PCSG00370 | 0.58% | 0.09% | 2,240 | 94 |
| PCSG00382 | 2.25% | 2.25% | 534 | 26 |
| PCSG00777 | 3.76% | 3.76% | 319 | 22 |
| PCSG00794 | 1.02% | 0.59% | 1,180 | 40 |
| PCSG00829 | 0.55% | 0.14% | 2,186 | 99 |
| PCSG01000 | 1.90% | 2.22% | 632 | 39 |
| PCSG00139 | 0.35% | 0.18% | 2,854 | 93 |
| PCSG00912 | 1.07% | 0.64% | 936 | 33 |
| PCSG01250 | 1.88% | 2.51% | 479 | 17 |
| PCSG00272 | 0.64% | 0.64% | 1,253 | 51 |
| PCSG00505 | 0.62% | 0.00% | 1,285 | 66 |
| PCSG01221 | 0.42% | 0.05% | 1,892 | 81 |
| PCSG00826 | 3.33% | 2.86% | 210 | 11 |
| PCSG00591 | 0.24% | 0.24% | 2,479 | 100 |
| PCSG01194 | 0.85% | 0.42% | 707 | 30 |
| PCSG00696 | 1.63% | 0.00% | 245 | 8 |
| PCSG00755 | 0.33% | 0.50% | 1,209 | 44 |
| PCSG01282 | 0.29% | 0.07% | 1,388 | 31 |
| PCSG00062 | 0.82% | 0.27% | 365 | 17 |
| PCSG00147 | 0.20% | 0.00% | 1,486 | 56 |
| PCSG00183 | 1.23% | 1.23% | 243 | 12 |
| PCSG00785 | 0.55% | 2.02% | 545 | 25 |
| PCSG00530 | 1.13% | 0.00% | 177 | 8 |
| PCSG01245 | 1.89% | 0.00% | 106 | 4 |
| PCSG00472 | 0.17% | 0.00% | 576 | 20 |
| PCSG00629 | 0.00% | 0.00% | 489 | 20 |
| PCSG01065 | 0.00% | 0.00% | 838 | 22 |
| PCSG01180 | 0.00% | 0.49% | 206 | 8 |
| PCSG01196 | 0.00% | 0.75% | 134 | 10 |
<!-- /generated:v2_baseline_titles -->

**Findings** (Normal with the cascade to Large unless a row says otherwise):

- **Where the new titles' errors are.** On all their pages, Rubi's path reads 1.19%. With the line filter off it
  reads 0.72% (`--filter none`), and the clean crops with the filter off read 0.34%. So the filter costs 0.48
  points, the spotter's boxes 0.38 and the recognizer with its LM 0.34. Large's recognizer share is 0.27%.
- **The line filter is the largest share.** Rubi's Shell drops a line with no kana or kanji (`non_japanese_filter =
  lines`). The hooked games draw whole rows of …… , ―― or ！？, the spotter finds them, the recognizer reads them,
  and the filter deletes the reading.
  - Keeping a line that has CJK punctuation, a bracket or an ellipsis (`--filter lines+punct`) gives 0.75% on the new
    titles (from 1.19%) and 0.87% on hval (from 1.37%).
  - With no filter at all, hval reads worse (1.48%): PCSG00854's dotted box border, read as digits, comes back.
  - `docs/finder-fix.md` measured `lines+punct` on GRAD's path at −0.055 points on hval and did not adopt it. GRAD
    often does not find such rows (`docs/vocr6-data.md`: PCSG00753, PCSG00611), so the filter had less to delete
    there; the spotter, Rubi's default since #149, finds them.
  - The filter is Rubi's (the Shell's), so this is reported, not decided here: a change needs its own pre-registered
    rule with the unseen-game dev sets as the guard.
- **The finder.** The spotter's boxes cost 0.38 points (both sides with the filter off). One title holds a third of
  that (769 of 2,275 errors): PCSG00845 (Silverio Vendetta, 95,785 reference characters) reads 1.03% through the
  spotter and 0.23% on its clean crops.
- **PSP.** The new PSP titles read 1.76% on the page path, 1.27% on clean crops and 0.54% with the filter off, against
  1.16%, 0.77% and 0.32% for Vita. They are 2 titles (NPJH50591, NPJH50609); the corpus's usable PSP lines grew 12%
  (28,436 → 31,841), its Vita lines 59% (53,541 → 85,146).
- **New against v1 titles on hval.** VOCR6.2 trained on the v1 titles' hval pages (they were folded in) and on none
  of the new titles. Rubi's path reads the new titles at 1.39% and the v1 titles at 1.36% (1.07% without PCSG00854,
  whose border costs 504 errors on 120 units). On clean crops 0.90% against 0.72%; with the filter off 0.38% against
  0.26%.
- **The hardest new titles for the recognizer** (clean crops, filter off): PCSG01202 at 7.5% (1,843 reference
  characters), PCSG00776 0.65%, PCSG00853 0.63% (it draws 『』, which VOCR6.2 reads as 「」). PCSG00206 holds the
  most errors (497) at 0.50%.

## What the new data is worth for Rubi's models

These are measurements to decide on, not decisions (the conductor and the owner decide):

- **A VOCR6 S/N/L retrain on pools v2.1** adds 35,010 usable dialogue lines (+43%), 3,584 vertical columns, 89 titles
  and 281 more kanji on the training side. On the new titles the recognizer and its LM make 0.34 points of
  Normal's 1.19% (Large 0.27 of 1.16%). That share is the most a retrain can remove there; VOCR6.2's 0.26% on its
  own training titles' hval suggests what training on a title buys at the recognizer, about 0.08 points. Any such
  run has to leave out of its dev comparisons the 7 dev-unit VNs and the 4 confirmation-set VNs that v2.1 now trains
  on.
- **A spotter retrain** (textspot8's recipe on v2.1's pages: 69,183 train pages with regions against v1's 48,384) aims
  at the finder's 0.38 points, a third of them on one title.
- **Rubi's line filter** (not a model) holds the largest share, 0.48 points; `lines+punct` removed 0.44 of them here.

## Reproduce

```sh
H=$HERD/shared; M=$H/private/hooked-b2; A=$H/artifacts/b2-ingest2; V=$H/artifacts/vocr6-ingest; PY=.venv/bin/python
# the download (read only), after the v1 copies; the earlier runs from the bucket's object versions
AWS_PROFILE=b2-kiro aws s3 sync s3://data-from-psp-and-vita/ $M --endpoint-url https://s3.eu-central-003.backblazeb2.com \
    --region eu-central-003
$PY data/import_hooked.py extract --root $M --out $A/pages
$PY data/import_hooked.py split --root $M --out $A/splits_v2.json
$PY data/import_hooked.py lines --root $M --pages $A/pages --splits $A/splits_v2.json --out $A/pools/v2.1 \
    --vndb-map data/manifests/vocr6_hooked_v2/vndb_map.json
$PY data/import_hooked.py pages --root $M --pages $A/pages --splits $A/splits_v2.json --pools $A/pools/v2.1 \
    --out $A/pagepool/v2.1
$PY data/import_hooked.py manifests --splits $A/splits_v2.json --pools $A/pools/v2.1 --pagepool $A/pagepool/v2.1 \
    --sources $A/sources_v2.json --out data/manifests/vocr6_hooked_v2
# v1, rebuilt from the frozen v1 root: equal to vocr6-ingest's pinned files
$PY data/import_hooked.py split --root $M/v1-20261026 --out SPLITS1.json
$PY data/import_hooked.py lines --root $M/v1-20261026 --pages $V/pages --splits $V/splits_v1.json --out POOLS1 \
    --pool-tag hooked-v1 --label-rules v1
# the audit and the records
$PY data/hooked_audit.py sample --pools $A/pools/v2 --exclude-v1 $V/pools/v1 --n 360 --seed 20261101 \
    --quota '{"image_rows": 70, "furigana": 40, "name": 50, "ruby": 40, "vertical": 40}' --out $A/audit/sample_v2.jsonl
$PY data/hooked_audit.py stats --pools $A/pools/v2.1 --pagepool $A/pagepool/v2.1 --splits $A/splits_v2.json \
    --out data/manifests/vocr6_hooked_v2/stats_v2.json
$PY data/hooked_audit.py overlap --stats data/manifests/vocr6_hooked_v2/stats_v2.json \
    --vndb-map data/manifests/vocr6_hooked_v2/vndb_map.json --vndb-dump $H/artifacts/native-frames/vndb/dump \
    --vn-set dev_units=DEV_VNS --vn-set confirm=CONFIRM_VNS --v1-index data/manifests/vocr6_hooked/index.json \
    --hooked-test --relations --out data/manifests/vocr6_hooked_v2/overlap_v2.json
# the baseline: units, the spotter's boxes, the reads (one heavy slot each)
$PY eval/pagepath.py units --pagepool $A/pagepool/v2.1 --set hval --out hval_v21.jsonl
$PY eval/textspot6.py spot-crop --blob textspot8_t64n8c16_w8.vtsp --set hval --units hval_v21.jsonl --pages $A/pages \
    --out hval_v21_spot.jsonl
$PY eval/pagepath.py run --units hval_v21.jsonl --pages $A/pages --models MODELS --lm vocr_lm6_o3_1000k.vlm \
    --lm-alpha 600 --lm-beta 2500 --lm-flags 1 --heads --tier N --rubi RUBI --cache CACHE \
    --boxes hval_v21_spot.jsonl --out hval_v21_spot_N.json          # also --finder rubi,fix=1 | oracle; S, L
$PY eval/hooked_v2_baseline.py score --res RESULTS --overlap data/manifests/vocr6_hooked_v2/overlap_v2.json \
    --protocol PROTOCOL_LINE.json --out eval/hooked_v2_baseline_20261101.json
$PY data/hooked_audit.py tables2 --index data/manifests/vocr6_hooked_v2/index.json \
    --stats data/manifests/vocr6_hooked_v2/stats_v2.json --overlap data/manifests/vocr6_hooked_v2/overlap_v2.json \
    --audit data/manifests/vocr6_hooked_v2/label_audit_v2.json --index-v1 data/manifests/vocr6_hooked/index.json \
    --stats-v1 data/manifests/vocr6_hooked/stats_v1.json --baseline eval/hooked_v2_baseline_20261101.json \
    --doc docs/vocr6-data-v2.md                                     # --check: what CI runs
```

MODELS holds the three VOCR6.2 blobs under pagepath's tier names (`VOCR5_{S,N,L}_w8.vocr`). The full commands are in
`$HERD/shared/artifacts/b2-ingest2/tools/`. The pinned v2.1 files:

- `pools/v2.1/manifest.json`: `8577e62a94f65756…`
- `pagepool/v2.1/manifest.json`: `bbadfcbff2a28975…`
- `splits_v2.json`: `85670e6467ef320f…`

## Limits

- **Machine-verified labels.** The audit bounds the label error of what it sampled (0 of 358 after the fixes, under
  1.1% at 95%). The rare-character scan covers characters unique to one title. A remapping of a common character,
  or one shared by two titles, would not show in it.
- **PSP is thin among the new titles.** 3 of the 89 new titles are PSP (one of them the test title), so the new
  titles' PSP numbers rest on 2 titles.
- **The second runs repeat the first.** Their pages add little new text; the cross-run guard keeps them out of
  screening but the near-duplicate filter keeps most of them as separate rows (their frames differ).
- **The 2x view is an emulation**, as in v1.
- **The bucket is live.** 27 runs say `in_progress`; v2.1 is the 2026-10-31 23:30Z snapshot.
