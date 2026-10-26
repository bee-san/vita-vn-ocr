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
stay under `$HERD/shared/artifacts`. Only the manifest (sha256 and counts, no text) is committed:
`data/manifests/vocr6_console_crops_v1.json`.

## How a crop is made

For every page of the chosen splits with dialogue regions, in each chosen region variant and view, the finder
returns its line boxes exactly as Rubi cuts them. The regions are the page pool's (`eval/finder_tighten`'s rule, as
in `eval/pagepath.py`). The two views are:

- **console**: Vita frames as they are, and PSP frames upscaled 2× bilinear, as Adrenaline shows them;
- **native**: PSP frames at 480×272.

The finder runs through `eval/pagepath.py`'s driver: the finder mode, Rubi's short-line probe, and the refinement on
PSP titles.

Each box is matched to the region's verified lines by their inked character cells. A cell counts as held by a box
when the centroid of its high-contrast pixels lies inside the box (`eval/pagepath.ink_points`, GRAD's own contrast
test).

| box | rule | becomes |
|---|---|---|
| whole | holds every inked cell of one line and none of another | a crop labelled with the line's text |
| partial | holds some of one line's cells and none of another | left out: no exact label exists |
| merged | holds cells of two lines | left out |
| empty | holds no inked cell and overlaps no renderer text box (name plates, furigana and unverified rows are text) | a negative, labelled "" |

Each whole crop also gets `--jitter` variants (2 by default). Each one is the finder's box moved at random:
- top and bottom by −0.15 to +0.35 of its height;
- left and right by −0.3 to +0.8 of its height.

The moves are seeded by the crop. A variant is kept only if it still holds the line's whole ink extent and no other
line's inked cell. Its dark flag is `synth/common.line_dark`'s, as the finder would give it.

Crops are stored as the console hands them to the recognizer: integer luma, with a dark-text crop inverted (meta
`inverted`). Identical crops of one page are stored once.

The pools are written as `OUT/<split>/<pool>/<title>.npz` in `train/train2.py`'s shard format (data, offsets,
shapes, labels, meta). There are three pools, `console`, `console_jitter` and `console_empty`, each with a `_native`
twin. A crop whose label has a character outside `data/vocab_vn_5631.txt`, or which `synth/render.crop_ctc_feasible`
rejects, goes to `drop_<pool>`. `OUT/manifest.json` records the inputs' sha256, every file's sha256 and crop count,
and the counts per title and kind.

## v1 (2026-10-26)

- **Finder:** `rubi,fix=5`, #129's adopted default (`VOCR_FIX_DEFAULT` = COLGAP | JUNK, `runtime/vocr_lines_fix.c`,
  PR #135).
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

The committed `data/console_crops.py` differs from the one that built v1 (sha256 `757f1ecc…`) in two ways only:
- its last print;
- its units carry the page's sha256, so that the page check of #135's `eval/pagepath.py` applies.

## Reproduce

```sh
A=$HERD/shared/artifacts
data/console_crops.py build --pagepool $A/vocr6-ingest/pagepool/v1 --pages $A/vocr6-ingest/pages \
  --finder rubi,fix=5 --sets train,hval,guard --variants base,expanded-8 --views console,native --jitter 2 \
  --procs 5 --out $A/vocr6-finder/crops/v1
```

`tests/test_console_crops.py` builds a pool from procedural pages through the same driver. No game data is used.
