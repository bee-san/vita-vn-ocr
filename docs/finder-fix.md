# finder-fix: three fixes to the console's line finder, measured on the owner's hooked pages (#129)

The console reads a dialogue region by cutting it into lines (`runtime/vocr_lines.c`, the GRAD finder in Rubi's
`VOCR_LINES_TEXTBOX_RECOVERY` mode) and reading each line. #119 showed that cutting is where most of the console's
errors are. The owner's hooked corpus (#125) gives 46,805 native pages with the renderer's own line boxes and the
game's text, so the finder can now be measured and tuned on real console frames of 47 train titles.

`runtime/vocr_lines_fix.c` is the finder with three fixes behind flags. With `fix = 0` it is the frozen finder bit for
bit. Rubi gets the default, `VOCR_FIX_DEFAULT` = COLGAP | JUNK, which the pre-registered rule adopted. VALLEY stays an
opt-in flag.

Every number is host-measured. The page path is `eval/pagepath.py` (#131): Rubi main `bd130a8`'s path from the finder
to the text the player sees, with VOCR5-N W8I, the 1 MB LM and the cascade to Large, in the console view (PSP frames
upscaled 2× as Adrenaline shows them). On sampled units it matches Rubi's own host tool exactly. Console times are
estimates; nothing ran on a console.

This PR also carries #131's audit fixes to the benchmark:
- `--boxes` is clipped;
- `compare` refuses runs of different units;
- the cache tag names the runtime library;
- the decoder settings follow each LM;
- pages are checked against their sha256;
- the probe port's licence note is corrected.

None of these changes what a `--finder` run measures: the numbers below come from the benchmark as merged in #131.

| | Rubi today | **default (COLGAP + JUNK)** | Δ points [95% interval] |
|---|---:|---:|---|
| hval (other pages of the train titles, 6,327 regions of 46 titles) | 3.875% | **1.898%** | −1.978 [−5.186, −0.377] |
| … Vita / PSP | 4.528% / 2.705% | 1.742% / 2.177% | −2.786 / −0.528 [−1.058, −0.152] |
| dev (#119's units: 93 other VNs) | 3.036% | **2.325%** | −0.712 [−1.279, −0.264] |
| … Vita / PSP | 2.948% / 3.113% | 1.932% / 2.665% | −1.016 / −0.448 [−0.947, −0.008] |
| test title, read once (Dance with Devils, 2,055 regions) | 1.241% | 0.978% | −0.263 [−0.454, −0.106] (page clusters) |

Dialogue CER is measured on whole regions: what a player who draws a box around the dialogue reads. Intervals come
from 10,000 cluster resamples: titles on hval, VNs on dev, pages on the single test title. Small and Large move the
same way: on hval and dev, Small goes 3.22% / 3.21% → 2.03% / 2.52%, and Large goes 3.92% / 3.01% → 1.89% / 2.28%
(every interval below 0). The cost is about 1.3 KB of Thumb-2 code. The arena, the stack and the finder's own time
do not change, and 4-6% fewer CTC frames are read.

## 1. What fails on the hooked pages

**The tune set.** For tuning I took a fixed sample of train pages, 60 per train title: 8,364 region units. These are
the page pool's regions at base / +4 / +8 px. Nothing from hval or the development units was used for tuning.

**The finder's cost.** Rubi today reads tune at 2.732%. With each reference line's own box (`--finder oracle`) it
reads 0.846%. So finding and cutting cost about 1.9 points. Looking at the units by failure (line statistics against
the renderer's cells, contact sheets of the worst ones) gave four main causes:

- **Lines cut at a full-width space** (#119's bug). GRAD keeps only the heaviest column cluster of a band. Clusters
  split at gaps over 2 band heights, and a full-width space after ！ or ？ is wider than that. The rest of the line is
  lost, or `TEXTBOX_RECOVERY` adds it back as a line of its own, often read first because its box starts higher.
  This happened in 31 titles.
- **Two touching rows read as one.** Heavy outlines or tight leading leave no row between two lines with an empty
  profile, so GRAD's band rule keeps them together and the merged crop reads as garbage. Narcissu (ULJM05674) read at
  20% on tune for this reason; PCSG00866 and the speech bubbles of NPJG00016 too.
- **Frame strips and ornaments read as text.**
  - PCSG00854's dialogue box has a dotted ornament along its top edge. The region a player draws includes it, GRAD
    makes a line of it, and the line reads as "で1001010101…". The title read at 23.6% on tune.
  - Thin strips of a box's frame at the region's top or bottom edge (NPJH50356, NPJH50641, PCSG00753) do the same
    in smaller numbers.
- **The long tail.**
  - grey or dimmer words inside a line, which fall under the region's contrast threshold;
  - trailing ！！；
  - `TEXTBOX`'s topology repair replacing a correct line around a long ー stroke;
  - a ruled-paper background.

  These are not addressed here.

## 2. The fixes (`runtime/vocr_lines_fix.c`, `vocr_lines_fix.h`)

`vocr_lines_fix.c` is `vocr_lines_rows.c` (the row form Rubi streams the region through) with a `fix` argument. The
frozen files `vocr_lines.c` and `vocr_lines_rows.c` are pinned by the protocols of #83, #96, #98, #100, #104 and #108,
so the fixes live beside them. The in-memory entry point reads the image through the same row accessor, so the two
forms cannot differ. The legacy finder ignores `fix`.

- **`VOCR_FIX_COLGAP`** (GRAD step 6): a band's column clusters split at gaps over 4 band heights instead of 2.
- **`VOCR_FIX_VALLEY`** (after the bands of step 5): a band of at least 2 × 8 + 1 rows is split at its first row of
  least kept-run length among the rows that leave 8 rows on each side. The split needs both of these:
  - that row's kept-run length is at most 1/4 of the largest on each side;
  - its on-pixel count is at most 1/3 of the largest on each side.

  The second test matters. Inside a single text row, the kept runs break between strokes but the on-pixels stay; in
  the gap between two rows both drop. The split row goes to neither part. Parts are examined again, so three touching
  rows split twice.
- **`VOCR_FIX_JUNK`** (on the finder's final set): two kinds of line are dropped.
  - A line on the region's top or bottom edge that is under 70% of the tallest other line's height.
  - A line from the region's left edge to its right edge whose on-mask repeats itself. The test is that, at some lag
    of 4 px to 3 line heights, under 40% of the pixels where either it or the mask shifted by the lag is on differ.
    It is computed 32 columns at a time.

  A region keeps at least one line.

## 3. Tuning on tune (train pages only)

| step | Δ points on tune |
|---|---|
| COLGAP 2.5× / 3× / 3.5× / **4×** / 5× / 6× band heights, against Rubi today | −0.371 / −0.479 / −0.541 / **−0.555** / −0.544 / −0.540 |
| VALLEY, on top of COLGAP, with the length test only (1/4) | −0.281: 26 titles worse (single rows split mid-glyph) |
| VALLEY with the on-pixel test, length factor 3 / **4** / 6, on-pixel factor **3** | −0.541 / **−0.543** / −0.543, 0 titles worse |
| VALLEY with on-pixel factor 2 | −0.504, 9 titles worse |
| JUNK on top of COLGAP + VALLEY: edge test at 0.6 / **0.7** | −0.163 / −0.243 |
| … plus the ornament test at **0.40** | −0.343 |

The chosen constants are in bold. On tune, Rubi today reads 2.732% and the default (COLGAP + JUNK) 1.724%: −1.008
[−1.983, −0.364]. The finder modes compared on top of COLGAP + VALLEY were GRAD +0.009 and TEXTBOX −0.024 against
`TEXTBOX_RECOVERY`, so Rubi's mode stays. Rubi's short-line probe still helps on top of all three fixes (+0.080
without it), so it stays as well.

## 4. The pre-registered comparison

The protocol is #129's comment of 2026-10-26T10:36Z. It was posted before any fix arm was scored on hval or dev; only
the two baselines had been. Each step is compared with the last adopted arm. A step is adopted when both of these hold:

1. on hval, Δ has a title-cluster 95% interval entirely below 0;
2. on dev, Δ has a VN-cluster 95% interval with an upper bound ≤ +0.05 points.

**Code check (passed).**
- With fix 0, `vocr_lines_fix.c` returns the frozen finder's boxes and flags on every unit: hval 6,327, dev 1,161,
  tune 8,364.
- The pipeline also matches Rubi's own host tool. `vjo-ocr`, built from Rubi `bd130a8` with vita-vn-ocr `817d203`,
  gives the same boxes and text on 400 / 400 sampled hval units and 200 / 200 dev units.

| step | hval: CER, Δ [interval] | dev: CER, Δ [interval] | verdict |
|---|---|---|---|
| Rubi today | 3.875% | 3.036% | |
| COLGAP vs Rubi today | 3.292%, −0.583 [−0.926, −0.309]; 25 titles better, 1 worse | 2.340%, −0.696 [−1.262, −0.249]; 9 VNs better, 0 worse | **adopted** |
| COLGAP + VALLEY vs COLGAP | 2.729%, −0.563 [−1.434, 0.000] | 1.574%, −0.766 [−2.862, +0.475] | not adopted |
| COLGAP + VALLEY + JUNK vs COLGAP | 1.505%, −1.787 [−5.163, −0.007] | 1.497%, −0.843 [−2.926, +0.293] | not adopted |
| COLGAP + JUNK vs COLGAP (amendment 1, registered before scoring) | 1.898%, −1.395 [−4.521, −0.004]; 5 better, 0 worse | 2.325%, −0.015 [−0.062, +0.011] | **adopted** |

**Why VALLEY is not in the default.**
- Its gain sits in a few titles. On hval it is Narcissu (677 → 137 errors) and PCSG00854. On dev it is v27449 (311 →
  72), #119's translucent box and its largest single error source, and v1315.
- A title-cluster bootstrap over 46 titles cannot resolve a two-title gain: about 13% of resamples hold neither title.
- On dev it also makes v10972 worse (38 → 86 errors). That VN's dialogue box has a dashed stitch line just under the
  text. It used to merge harmlessly into the text's box; VALLEY splits it off as a line of its own, read ニーニュー….
- So VALLEY is an opt-in flag for games with touching outlined rows. With JUNK already in the default, its remaining
  hval gain is Narcissu alone (13.65% → 2.76% on that title).

**JUNK's evidence and the remaining problems.**
- JUNK's numbers on top of VALLEY were seen before amendment 1 registered it on top of COLGAP alone; its constants
  were not changed.
- Per title on hval, the default helps 27 titles and makes one worse: PCSG00620, 61 → 70 errors.
- The largest remaining hval errors are Narcissu (13.65%), PCSG00753 (13.42%, a ruled-paper background) and
  PCSG00854 (8.68%, from 67.34%).

**Sibling titles (the audit's N10, a sensitivity check).** Some train titles share an engine and fonts. Resampling
them together (38 clusters instead of 46) leaves every verdict as it was. The groups are:
- Hayarigami 1, 2 and Shin;
- Memories Off Sorekara and 6;
- the four MAGES science-adventure titles;
- Code Realize 1 and 2;
- Starry Sky Autumn and Winter.

| comparison on hval | Δ points | 95% interval, title clusters | 95% interval, series clusters |
|---|---:|---|---|
| COLGAP vs Rubi today | −0.583 | [−0.926, −0.309] | [−0.956, −0.315] |
| COLGAP + VALLEY vs COLGAP | −0.563 | [−1.434, 0.000] | [−1.487, 0.000] |
| COLGAP + JUNK vs COLGAP | −1.395 | [−4.521, −0.004] | [−4.760, −0.004] |
| default vs Rubi today | −1.978 | [−5.186, −0.377] | [−5.422, −0.378] |

**Development empty regions (eval/linefind's 66, report only).** The same 10 regions read some text before and after
the fixes, but the characters read fall from 27 to 23 and the boxes from 112 to 100.

**The test title** (Dance with Devils, PCSG00807, a test-split title of the corpus), read once after the decision for
Rubi today and the default: 1.241% → 0.978%, −0.263 [−0.454, −0.106] over 685 pages.

`eval/finder_fix_20261026.json` holds every count, interval and per-title number above; it is built by
`eval/finder_fix_tables.py`.

## 5. Cost

- **Code.** `vocr_lines_fix.o`, Thumb-2 with the VitaSDK compiler (arm-vita-eabi-gcc 15.2.0, Cortex-A9,
  `-ffunction-sections`): 13,764 B at -O2 and 9,898 B at -Os. `vocr_lines_rows.o`, which it replaces in Rubi, is
  12,500 / 9,048 B, so the growth is **+1,264 B at -O2** (+850 B at -Os). VALLEY's code is included.
- **Memory.**
  - No arena change: `vocr_lines_arena_bytes()` is unchanged.
  - No heap and no recursion.
  - The deepest stack frame is the same as `vocr_lines_rows.c`'s (`find_grad` 1,440 B; the new `drop_junk` is
    136 B, called after `find_textbox` returns).
- **Finder time.** Host, gcc -O2, best of 5 per region over the 2,109 base hval regions in the console view, on a
  loaded 64-core box (load about 78). Two runs:

  | | fix 0 | default | all three |
  |---|---:|---:|---:|
  | run 1, µs per region | 433.3 | 429.8 | 429.9 |
  | run 2, µs per region | 439.3 | 436.0 | 436.0 |

  So there is no measurable cost: the default is 0.8% faster in both runs, because it has fewer lines to finish. The
  ornament test runs only on lines that span the whole region, 32 columns at a time.
- **Recognition time.** Fewer junk lines are read: hval CTC frames fall 5.7% (540,993 → 510,032) and dev 3.7%. That
  is an estimated saving at the console's per-frame rate, not a measurement.

## 6. In Rubi

`vocr_find_lines_fix_rows(&region, VOCR_LINES_TEXTBOX_RECOVERY, VOCR_FIX_DEFAULT, …)` replaces
`vocr_find_lines_rows(…)`, with the same arena. Rubi's short-line probe and its refinement on PSP titles stay. A
config key turns the fixes off and on (Rubi PR).

**To test on console:**
- a Narcissu or PCSG00854-style page;
- a region whose line has a full-width space after ！;
- time per job with the fixes on and off.

## 7. Limits

- The hooked titles are 47 train titles. hval holds other pages of the same titles, so a fix that fits a title's
  style shows there too. The development units (other games, mostly PC screenshots scaled to the console's size)
  are the unseen-game guard.
- COLGAP comes from #119's analysis of the development units, so its dev number is not a held-out estimate.
- The labels are machine-verified (renderer cells against hook text), not hand-checked. One label swap was seen: two
  speech bubbles of NPJG00016 with the same cell count. Labels affect both arms of every comparison alike.
- Rubi's `non_japanese_filter = lines` drops lines of only ellipses or brackets (「……」), which the references hold.
  This costs every arm alike (for example 1-2 points of PCSG00611) and is not a finder matter.
- A filter that also keeps lines with CJK punctuation, a bracket or an ellipsis (`--filter lines+punct`) was looked
  at after the decision. It was not pre-registered and is not adopted. With the default finder:
  - hval: 1.898% → 1.843%, −0.055 [−0.134, +0.017], not resolved (PSP −0.098 [−0.246, −0.011]).
  - dev: unchanged.
  - Keeping every line (`--filter none`) is worse on Vita: +0.158 [−0.127, +0.678].

  Rubi's Shell owns that filter, so a change to it is a separate decision.

## 8. Reproduce

The paths are under `$HERD/shared/artifacts/vocr6-finder` (private). One heavy slot each:

```sh
A=$HERD/shared/artifacts; P="--pages $A/vocr6-ingest/pages --models $A/vvo-public/v030 \
  --lm $A/vvo-public/v030/vocr_lm_o3_1000k.vlm --rubi RUBI_CHECKOUT --cache CACHE.sqlite"
eval/pagepath.py units --pagepool $A/vocr6-ingest/pagepool/v1 --set hval --out hval.jsonl     # also: tune --per-title 60
eval/pagepath.py run --units hval.jsonl $P --finder rubi --out base_hval.json
eval/pagepath.py run --units hval.jsonl $P --finder rubi,fix=5 --out F13_hval.json                # the default
eval/pagepath.py run --units $A/vocr5-analysis/work/units.pkl $P --finder rubi,fix=5 --out F13_dev.json
eval/pagepath.py compare --base base_hval.json --arm F13_hval.json
make -C runtime check-lines-fix
```

Sources of the measured runs:
- The decision runs (§4) used `vocr_lines_fix.c` sha256 `746aa003…`.
- The tiers, the test title and the console crops used `9fe5427c…`.
- On every hval and dev unit the two give the same boxes, dark flags and text (a re-run of the default).
- The committed file differs from `9fe5427c…` only in a comment that names its test, and `eval/pagepath_find.c`
  differs from the measured `3eed7809…` only in its licence note.
