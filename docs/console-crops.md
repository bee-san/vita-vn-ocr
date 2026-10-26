# Console crops: the finder's own line crops, labelled from the hooked corpus (#129, VOCR6 #125)

The recognizer is trained on clean line crops, but on the console it reads the crops that the line finder cuts.
Those crops are padded, cut at the top, wider than the line, or hold a neighbour's edge. #119 found that most of the
console's whole-dialogue errors come from finding and cutting lines, and part of that is the recognizer misreading
imperfect crops.

The owner's hooked corpus (#125) gives every line's text and its renderer cells on native frames. So the console's
crops can be labelled exactly: run the finder over every train page, keep each crop that holds exactly one verified
line, and label it with that line's text. The resulting pool is for crop-in-the-loop training, vocr6-lead's stage-2
arm S2cil.

`data/console_crops.py` builds the pool. It is private by design: the crops and labels are copyrighted game data and
stay under `$HERD/shared/artifacts`. Only the manifests (sha256 and counts, no text) are committed:
`data/manifests/vocr6_console_crops_v1.json` and `_v2.json`.

## How a crop is made

For every page of the chosen splits with dialogue regions, in each chosen region variant and view, the finder
returns its line boxes exactly as Rubi cuts them. The regions are the page pool's (`eval/finder_tighten`'s rule, as
in `eval/pagepath.py`). The two views are:

- **console**: Vita frames as they are, and PSP frames upscaled 2× bilinear, as Adrenaline shows them;
- **native**: PSP frames at 480×272.

The finder runs through `eval/pagepath.py`'s driver: the finder mode, Rubi's short-line probe, and the refinement on
PSP titles.

Each box is matched to the region's verified lines by their character cells. A cell is inked when it has
high-contrast pixels (`eval/pagepath.ink_points`, GRAD's own test), and an inked cell counts as held by a box when
the centroid of those pixels lies inside the box.

Some cells hold a character with no pixel at the line's threshold, such as a word in a dim colour. A box must also
cover at least half of each such cell (`required_cells`):
- when the line has one cell per character of its label (89% of lines), every inkless cell of a non-space
  character;
- otherwise, the inkless cells with weak ink (at half the threshold), but only when there are more of them than the
  line has extra cells. A line's extra cells, such as an indent or a blank cell before or after the text, are blank.
  Over a textured box or scenery a blank cell has weak "ink" too, and requiring it would drop correct crops
  (audit-vocr6-finder-d).

| box | rule | becomes |
|---|---|---|
| whole | holds every inked cell and covers every required cell of one line, and holds or covers none of another | a crop labelled with the line's text |
| partial | holds or covers some of one line's cells and none of another | left out: no exact label exists |
| merged | holds or covers cells of two lines | left out |
| empty | holds no inked cell, covers no required cell and overlaps no renderer text box (name plates, furigana and unverified rows are text) | a negative, labelled "" |

Each whole crop also gets `--jitter` variants (2 by default). Each one is the finder's box moved at random:
- top and bottom by −0.15 to +0.35 of its height;
- left and right by −0.3 to +0.8 of its height.

The moves are seeded by the crop. A variant is kept only if:
- it still holds the line's whole ink extent and covers its required cells;
- it holds no other line's inked cell and covers none of its required cells.

Its dark flag is `synth/common.line_dark`'s, as the finder would give it. The rule tests ink centroids, so a
jittered box can hold part of a neighbouring line's glyphs. In v1, 8.6% of console_jitter crops hold at least 30%
of some neighbour cell's ink and 1.6% hold at least half of it, against 0.10% and 0.02% of the finder's own crops
(audit-vocr6-finder-c, a 15,000-crop sample). The label is the main line's.

Crops are stored as the console hands them to the recognizer: integer luma, with a dark-text crop inverted (meta
`inverted`). Identical crops of one page are stored once.

The pools are written as `OUT/<split>/<pool>/<title>.npz` in `train/train2.py`'s shard format (data, offsets,
shapes, labels, meta). There are three pools, `console`, `console_jitter` and `console_empty`, each with a `_native`
twin. `train/train2.py` keeps no "" label, so the negatives need a blank-target option there before they can be
trained on. A crop whose label has a character outside `data/vocab_vn_5631.txt`, or which `synth/render.crop_ctc_feasible`
rejects, goes to `drop_<pool>`. `OUT/manifest.json` records the inputs' sha256, every file's sha256 and crop count,
and the counts per title and kind.

## v1 (2026-10-26)

- **Finder:** `rubi,fix=5`, COLGAP + JUNK (`runtime/vocr_lines_fix.c`, PR #135).
  - That was #129's default when v1 was built, before #129's amendment 3. Since then the default is COLGAP alone
    (`fix=1`), because JUNK drops text lines in regions drawn tight.
  - On the page pool's regions, which keep a margin, JUNK does not change the labelled crops.
    audit-vocr6-finder-f rebuilt 3 titles at `fix=1`:
    - PCSG00935: all files are identical.
    - PCSG00854, JUNK's ornament title: the same whole boxes and identical whole crops. There are more empty boxes
      (232 → 1,102, 996 of them on renderer text boxes, so not negatives) and more negatives (20 → 102), and other
      jitter draws where box indices shift.
    - NPJG00016: 1 console and 5 native whole crops differ.
  - [v2](#v2-2026-10-26) is cut with `fix=1`, Rubi's default. If #129's amendment 2 selects vocr6-spotter's
    spotter, a later pool is cut from its boxes.
- **Input:** page pool v1 (`eab8f07b…`, splits `417df3d7…`), sets train / hval / guard, variants base and
  expanded-8, both views.
- **Build:** 7 minutes for the 46 titles with dialogue regions, 5 processes.
- **Manifest:** sha256 `4adb00c98838fc0a2350c700ad4423ff9fd7212aa31d7e72cbcf0b79647b3895`, pinned by vocr6-lead's
  protocol (`data_pins.extra_pools`).

| pool | train | hval | guard |
|---|---:|---:|---:|
| console (whole lines) | 98,324 | 4,245 | 4,574 |
| console_jitter | 304,524 | 13,064 | 14,249 |
| console_empty (negatives) | 1,145 | 38 | 50 |
| console_native / _jitter_native / _empty_native (PSP at 480×272) | 38,500 / 111,152 / 689 | 1,679 / 4,700 / 25 | 1,671 / 4,779 / 29 |
| drop_* (outside the vocab or not CTC-feasible) | 150 / 321 / 142 / 299 | 2 / 4 / 2 / 4 | 0 |

Finder boxes in the console view, all splits:
- 168,846 whole;
- 2,821 partial;
- 526 merged, 402 of them Narcissu's touching rows, which #135's opt-in `VOCR_FIX_VALLEY` would split;
- 2,035 empty, of which 758 overlap a renderer text box and are not used as negatives.

Coverage: 77,596 of the train split's 79,733 region lines (97.3%) have at least one whole console crop.

Two titles have no crop. NPJH50127's pages have no verified line in a region, and PCSG00653's dialogue is vertical.
The test title (PCSG00807) is not in the pool, and CLANNAD is not in the corpus.

**Use:** train on `train/` only. hval and guard are vocr6's screening holdout and its guard bands; their crops exist
for a final model retrained with hval folded back in.

**Verification** (2026-10-26, a private script beside the pool, `verify_v1.py`):

- **Files:** all 461 listed files are present, with no extra ones. Every sha256, crop count and layout is right.
- **Re-cut:** 3,000 random crops were re-cut from their page in their view, after checking the page's sha256. All
  3,000 are pixel-exact. Each label equals the page pool's text of its line, or "" for an empty crop, and each page is
  in its file's split.
- **Loader:** `train/train2.load_shard` with the N vocab keeps all 98,324 train/console crops, all 304,524
  train/console_jitter crops and all native ones.
- **By eye:** 252 crops on 7 contact sheets, checked by the assistant (Claude):
  - 216 labelled crops, every label right;
  - 36 negatives, all text-free (frames, icons, scenery);
  - the 36 most extreme chars-per-width crops, labels right. They are very wide boxes over background, and some are
    cut at the top: real finder output.

**Independent audit** (audit-vocr6-finder-c, its own seeds): 18,012 more random crops re-cut pixel-exact, 280 of 280
labelled crops and 80 of 80 negatives right by eye, and 4 titles rebuilt byte for byte.

**Known issue: v1 has 63 wrong labels in its train pools**, 0.011% of their 552,500 whole and jittered crops, and none
in hval or guard. Two audits counted them:
- 12 in console and 44 in console_jitter, the pools S2cil reads (56 of 402,848, 0.014%);
- 3 in console_native and 4 in console_jitter_native.

v1 was built before the required-cell rule above existed. Each of these crops is a finder box that leaves out a
character in a dim colour, or more than half of its cell, labelled with the whole line all the same. They are in
PCSG00805, NPJG00016, PCSG00935 and NPJH50297. The lists are:
- audit-vocr6-finder-c's `missing_chars.json`: 44;
- audit-vocr6-finder-d's `v1_extra_wrong.json`: its 19 entries with reason `own_uncovered`. Its other 4 entries are
  jittered native crops whose labels are right: their own box covers the character, and only their finder box does
  not.

The rule now in the script makes such a box partial. Over the whole pool it removes 67 crops: the 63, and 4 jittered
crops of their finder boxes. By title that is PCSG00805 45, NPJG00016 12, NPJH50297 5 and PCSG00935 5
(audit-vocr6-finder-f's scan of every v1 crop). I rebuilt 8 titles with it, on #135's finder. Only the three among
them with dim words change, all in the train split:
- PCSG00805: 45 crops fewer;
- NPJG00016: 12 fewer;
- PCSG00935: 5 fewer.

By pool: console 12, console_jitter 44, console_native 2 and console_jitter_native 4. 77 of the 85 files are byte
for byte v1's. Unchanged are:
- PCSG00713, with scenery and portraits behind its box;
- NPJH50356, a patterned box;
- NPJH50641;
- PCSG00380, whose lines start with an inkless indent cell;
- PCSG00469, with centred lines.

A first version of the rule required every weak-ink extra cell. It would have dropped 2,581 correct crops pool-wide,
in 14 titles; PCSG00713 alone would have lost 12% (audit-vocr6-finder-d B1).

The rule's limit is a count. On a line with k more cells than characters, none of its n weak-ink inkless cells is
required unless n > k. So up to k dim characters on such a line are not required, wherever they sit, and k reaches
9 on centred lines. On v1's boxes this happens nowhere: audit-vocr6-finder-f viewed all 490 cells the count leaves
unrequired, and every one is blank (an indent, a centring cell, a trailing or a full-width-space cell, with box
texture or art as its weak "ink").

Other notes on v1:
- **The driver.** v1's driver (`e32cd2da…` in the manifest) was compiled from a `vocr_lines_fix.c` (`9fe5427c…`) that
  differs from #135's committed file only in a comment. A rebuild on #135 has another driver hash and writes the same
  shards.
- **Repeats.** There is no de-duplication across pages, unlike vocr6-ingest's line pools. train/console holds 98,324
  crops of 71,528 distinct (title, label) pairs: mean 1.37, at most 491 copies of one label. console_jitter's mean
  is 4.27.
- **The script.** The committed `data/console_crops.py` differs from the one that built v1 (sha256 `757f1ecc…`) in
  these ways:
  - the required-cell rule;
  - the driver is built once before the workers start (they raced on its file on a cold work directory);
  - `--sets test` is refused;
  - the de-duplication key hashes the crop's bytes (its shape is already in the key), which changes no output;
  - its units carry the page's sha256, so that the page check of #135's `eval/pagepath.py` applies;
  - its last print.

  v1 itself is rebuilt byte for byte by #136's script (`git show 7902aff:data/console_crops.py`) on #135's finder
  (audit-vocr6-finder-c §4).

## v2 (2026-10-26)

v2 is cut by Rubi's shipped finder and labelled with the required-cell rule. It is for vocr6-lead's stage-4b fold-in
retrain, on the train, hval and guard pages. v1 is unchanged and stays pinned for S2cil.

- **Finder:** `rubi,fix=1`, COLGAP alone. That is `VOCR_FIX_DEFAULT` since #129's amendment 3, and Rubi's
  `ocr_line_fix = on` (bee-san/Rubi_for_vita#330).
- **Labelling:** the required-cell rule above.
- **Everything else is v1's:** page pool v1, the three sets, both variants, both views and jitter 2.
- **Build:** `data/console_crops.py` at main `7766d4a` (#136's merge) with the command under Reproduce. It took 7.5
  minutes with 5 processes.
- **Manifest:** sha256 `45cbddaa2aa7d5226ea4bf5999b25aee313e977132d58b0f19f49833177e894e`, committed as
  `data/manifests/vocr6_console_crops_v2.json`.

| pool | train v1 → v2 | hval v1 → v2 | guard v1 → v2 |
|---|---:|---:|---:|
| console (whole lines) | 98,324 → 98,340 | 4,245 → 4,245 | 4,574 → 4,574 |
| console_jitter | 304,524 → 304,543 | 13,064 → 13,068 | 14,249 → 14,247 |
| console_empty (negatives) | 1,145 → **4,712** | 38 → 195 | 50 → 226 |
| console_native / _jitter_native | 38,500 / 111,152 → 38,542 / 111,284 | 1,679 / 4,700 → 1,680 / 4,705 | 1,671 / 4,779 → 1,672 / 4,782 |
| console_empty_native | 689 → **1,937** | 25 → 81 | 29 → 76 |
| drop_* | 150 / 321 / 142 / 299 → 150 / 323 / 142 / 299 | 2 / 4 / 2 / 4, unchanged | 0 |

The labelled pools barely change. The negatives grow about fourfold, because without JUNK the finder returns frame
strips, ornaments and slivers of art on a region's edge as lines, as Rubi's default does on the console. The
required-cell rule cannot add empty boxes: a box over a required cell counts as holding its line. So the growth is
the finder's (audit-vocr6-finder-f's PCSG00854 rebuild above shows the same).

Finder boxes in the console view, all splits:
- 168,858 whole;
- 2,919 partial;
- 526 merged;
- 6,969 empty, of which 1,792 overlap a renderer text box and are not used as negatives.

Coverage: 77,597 of the train split's 79,733 region lines (97.3%) have at least one whole console crop (v1: 77,596).

**Verification** (2026-10-26, `verify_v2.py`: v1's script on v2's directory):

- **Files:** all 476 listed files are present, with no extra ones. Every sha256, crop count and layout is right.
- **Re-cut:** 3,000 random crops were re-cut from their page, after checking the page's sha256. All 3,000 are
  pixel-exact, with the right label and split.
- **Loader:** `train/train2.load_shard` with the N vocab keeps every train crop.
- **v1's known wrong labels:** none of the 63 listed crops is in v2 (no v2 crop has their pixels).
- **By eye:** 48 random console_empty crops that are new in v2 (of 3,563), on one contact sheet, checked by the
  assistant (Claude). They are frame strips, ornament borders, gradient bars, slivers of character art and a clock
  icon, and none holds text.
- **Against v1:** of v2's whole-line crops, 99.94% of train/console and 99.69% of train/console_native have the
  pixels and the label of a v1 crop (hval and guard 99.88-100%). Of the jittered ones, 96.0-96.8% do in every split:
  a box's draws follow its index among the region's boxes, and that shifts where JUNK dropped a box before. Only
  17-37% of the negatives are v1's.

## Reproduce

```sh
A=$HERD/shared/artifacts
data/console_crops.py build --pagepool $A/vocr6-ingest/pagepool/v1 --pages $A/vocr6-ingest/pages \
  --finder rubi,fix=1 --sets train,hval,guard --variants base,expanded-8 --views console,native --jitter 2 \
  --procs 5 --out $A/vocr6-finder/crops/v2    # Rubi's default finder; a new pool: never write into the pinned v1
```

`tests/test_console_crops.py` builds a pool from procedural pages through the same driver. No game data is used.
