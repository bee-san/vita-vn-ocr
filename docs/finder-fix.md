# finder-fix: three fixes to the console's line finder, measured on the owner's hooked pages (#129)

The console reads a dialogue region by cutting it into lines (`runtime/vocr_lines.c`, the GRAD finder in Rubi's
`VOCR_LINES_TEXTBOX_RECOVERY` mode) and reading each line. #119 showed that cutting is where most of the console's
errors are. The owner's hooked corpus (#125) gives 46,805 native pages with the renderer's own line boxes and the
game's text, so the finder can now be measured and tuned on real console frames: 46 train titles have dialogue
regions (48 are marked train; NPJH50127 and PCSG00653 have none).

`runtime/vocr_lines_fix.c` is the finder with three fixes behind flags. With `fix = 0` it is the frozen finder bit for
bit.
- **`VOCR_FIX_DEFAULT` is COLGAP**, which Rubi gets.
- **JUNK is opt-in.** The pre-registered rule adopted COLGAP + JUNK (amendment 1), but #135's audit then showed that
  JUNK deletes real dialogue lines in regions drawn tight around the text (§4, amendment 3). JUNK is for regions drawn
  with a margin.
- **VALLEY is opt-in** as well.

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

None of these changes what a `--finder` run measures (a cold re-run gives the decision runs unit for unit).

**The default (COLGAP) against Rubi today:**

| regions | Rubi today | **default (COLGAP)** | Δ points [95% interval] |
|---|---:|---:|---|
| hval: other pages of the train titles, 6,327 regions of 46 titles, as the page pool draws them (12-20 px margin) | 3.875% | **3.292%** | −0.583 [−0.926, −0.309]; 25 titles better, 1 worse |
| … Vita / PSP | 4.528% / 2.705% | 3.902% / 2.199% | −0.626 [−1.107, −0.286] / −0.506 [−1.038, −0.128] |
| dev: #119's units of 93 other VNs | 3.036% | **2.340%** | −0.696 [−1.262, −0.249]; 9 VNs better, 0 worse |
| … Vita / PSP | 2.948% / 3.113% | 1.965% / 2.665% | −0.983 [−2.074, −0.162] / −0.448 [−0.947, −0.008] |
| hval base units cut tight: the lines' union, 0 px / + 2 px | 2.570% / 2.243% | 1.933% / 1.621% | −0.637 [−1.022, −0.340] / −0.622 [−1.003, −0.330] |
| dev base units cut 1 px around the ink | 3.110% | 2.123% | −0.986 [−1.852, −0.362] |
| VOCR5-S, hval / dev | 3.221% / 3.210% | 2.639% / 2.541% | −0.583 [−0.920, −0.311] / −0.669 [−1.220, −0.235] |
| VOCR5-L, hval / dev | 3.918% / 3.009% | 3.342% / 2.313% | −0.576 [−0.916, −0.303] / −0.696 [−1.262, −0.249] |
| the test title (Dance with Devils, 2,055 regions; report only, §4) | 1.241% | 0.978% | −0.263 [−0.454, −0.106] (page clusters) |

**JUNK (opt-in) on top of COLGAP:**

| regions | COLGAP | COLGAP + JUNK | Δ points [95% interval] |
|---|---:|---:|---|
| hval, as the page pool draws them | 3.292% | 1.898% | −1.395 [−4.521, −0.004]; 5 titles better, 0 worse |
| dev, as #119 drew them | 2.340% | 2.325% | −0.015 [−0.062, +0.011] |
| hval cut tight (0 px) | 1.933% | 2.391% | **+0.458 [−0.007, +1.451]**; 1 better, 2 worse |
| hval cut tight + 2 px | 1.621% | 1.627% | +0.006 [0.000, +0.018] |
| dev cut 1 px around the ink | 2.123% | 2.727% | **+0.603 [0.000, +1.938]** |
| tune (train pages) cut tight (0 px) | 1.707% | 2.348% | **+0.641 [+0.022, +1.790]**; 2 better, 8 worse |

Dialogue CER is measured on whole regions: what a player who draws a box around the dialogue reads. Intervals come
from 10,000 cluster resamples: titles on hval and tune, VNs on dev, pages on the single test title.

The cost of the default:
- **Code:** about 1.3 KB of Thumb-2.
- **Memory:** the arena does not change, and the stack grows by 8 B.
- **Time:** the finder's own time does not measurably change, but 0.6-0.7% more CTC frames are read, because COLGAP
  keeps a line's parts together (§5).

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

  When every line of a region would be dropped, none is.

  JUNK keys on lines that reach the region's edges. In the page pool's regions (12-20 px margin), no text line's box
  reaches an edge, so only frame strips and ornaments are dropped. In a region drawn tight around the text, the first
  and last lines and the full-width lines reach the edges too. Then JUNK drops real dialogue lines (§4), so it is
  opt-in.

## 3. Tuning on tune (train pages only)

| step | Δ points on tune |
|---|---|
| COLGAP 2.5× / 3× / 3.5× / **4×** / 5× / 6× band heights, against Rubi today | −0.371 / −0.479 / −0.541 / **−0.555** / −0.544 / −0.540 |
| VALLEY, on top of COLGAP, with the length test only (1/4) | −0.281: 26 titles worse (single rows split mid-glyph) |
| VALLEY with the on-pixel test, length factor 3 / **4** / 6, on-pixel factor **3** | −0.541 / **−0.543** / −0.543, 0 titles worse |
| VALLEY with on-pixel factor 2 | −0.504, 9 titles worse |
| JUNK on top of COLGAP + VALLEY: edge test at 0.6 / **0.7** | −0.163 / −0.243 |
| … plus the ornament test at **0.40** | −0.343 |

The chosen constants are in bold. These sweeps ran on the tune units of the page pool as it was before
vocr6-ingest's label audit (8,358 units, where Rubi today reads 2.739%). The v1 pool's tune set (8,364 units, Rubi
today 2.732%) gives COLGAP + JUNK 1.724%, −1.008 [−1.983, −0.364]. The finder modes compared on top of COLGAP + VALLEY were GRAD +0.009 and TEXTBOX −0.024 against
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
| COLGAP + JUNK vs COLGAP (amendment 1, registered before scoring) | 1.898%, −1.395 [−4.521, −0.004]; 5 better, 0 worse | 2.325%, −0.015 [−0.062, +0.011] | adopted, then withdrawn by amendment 3 |

**Amendment 3: the default is COLGAP alone.** #135's audit checked robustness beyond the protocol.
- **Why the protocol could not see it.** Every region the protocol scores keeps a 12-20 px margin, so no text line's
  box ever reaches the region's edge. JUNK's two rules key on exactly that.
- **What Rubi users do.** Rubi's guide asks them to choose a tight dialogue region, and `vocr_lines.h` allows "a
  tight box around the dialogue".
- **What happens then.** On the same units cut tight, JUNK drops real dialogue lines:
  - PCSG00628: a textured box makes a full-width text line pass the repeat test; 2 → 240 errors.
  - NPJH50297: inline name lines and short last lines sit on an edge.
  - PCSG00648: a furigana line makes "the tallest other line" tall, so the normal last line under it is dropped.
  - Synthetic full-width lines of repeated kana are dropped in 16 of 42 tight cases.

  I replicated the audit's tight-region runs (table at the top; `tight` in the results JSON). JUNK is resolved worse
  on tight tune pages (+0.641 [+0.022, +1.790]) and is not resolved better on tight hval or dev. COLGAP is resolved
  better than Rubi today in both region styles.
- **The decision.** Applied to the tight variants, the registered rule would not adopt COLGAP + JUNK: the dev upper
  bound is +1.938, far above +0.05. So amendment 3 (#129, 2026-10-26) makes `VOCR_FIX_DEFAULT` = COLGAP, the last step
  that is robust. JUNK stays an opt-in flag for regions drawn with a margin, where it is −1.395 points.

  This is a deviation from amendment 1's adoption. It rests on the robustness check, not on a new selection score.

A guarded JUNK, which cannot remove text lines in tight regions, is future work with its own registration. It would
use an edge rule against the median line height and a repeat test only for strips thinner than the text lines.

**Why VALLEY is not in the default.**
- Its gain sits in a few titles. On hval it is Narcissu (677 → 137 errors) and PCSG00854. On dev it is v27449 (311 →
  72), #119's translucent box and its largest single error source, and v1315.
- A title-cluster bootstrap over 46 titles cannot resolve a two-title gain: about 13% of resamples hold neither title.
- On dev it also makes v10972 worse (38 → 86 errors). That VN's dialogue box has a dashed stitch line just under the
  text. It used to merge harmlessly into the text's box; VALLEY splits it off as a line of its own, read ニーニュー….
- So VALLEY is an opt-in flag for games with touching outlined rows. On Narcissu it takes hval from 13.65% to 2.76%.

**JUNK's evidence and the remaining problems.**
- JUNK's numbers on top of VALLEY were seen before amendment 1 registered it on top of COLGAP alone; its constants
  were not changed.
- On hval, COLGAP + JUNK helps 27 titles and makes one worse (PCSG00620, 61 → 70 errors). Most of its gain is in
  PCSG00854, whose dotted top ornament reads as text: 67.34% → 8.68%.
- With the default (COLGAP), the largest remaining hval errors are:
  - PCSG00854's ornament, which JUNK removes in a region with a margin;
  - Narcissu's touching rows (13.65%), which VALLEY splits;
  - PCSG00753's ruled paper.

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
| COLGAP + JUNK vs Rubi today | −1.978 | [−5.186, −0.377] | [−5.422, −0.378] |

**Development empty regions (eval/linefind's 66, report only).** Rubi today reads 27 characters in 10 of them, from 112
boxes.
- With COLGAP + JUNK, 10 regions read 23 characters from 100 boxes: 9 of Rubi today's 10, and `dev:38`, one of the
  two that COLGAP adds. JUNK empties `dev:26`, which read 6 characters.
- With the default, COLGAP, 12 regions read 31 characters from 112 boxes: two regions that read nothing before now
  read 2 characters each.

The effect is tiny either way. The text-presence gate (#100) is the tool for empty regions.

**The test title** (Dance with Devils, PCSG00807, the corpus's test-split title) was read four times, all after the
decisions they follow and only to report. Nothing was selected on any of them:
- Rubi today: 1.241%;
- COLGAP + JUNK: 0.978%;
- COLGAP + JUNK with the report-only `--filter lines+punct`: 0.978%;
- COLGAP, after amendment 3: 0.978%, −0.263 [−0.454, −0.106] over 685 page clusters.

JUNK changes nothing on this title: its regions keep a margin, as hval's do. The three later reads give the same 409
errors in 41,823 characters.

`eval/finder_fix_20261026.json` holds every count, interval and per-title number above, the tight-region runs included.
`eval/finder_fix_tables.py --runs $A/vocr6-finder/runs` rebuilds it byte for byte at this commit.

## 5. Cost

- **Code.** `vocr_lines_fix.o`, Thumb-2 with the VitaSDK compiler (arm-vita-eabi-gcc 15.2.0, Cortex-A9,
  `-ffunction-sections`), is 13,764 B at -O2 and 9,898 B at -Os. `vocr_lines_rows.o`, which it replaces in Rubi, is
  12,500 / 9,048 B. As Rubi links it (`--gc-sections`, without the in-memory entry point or `vocr_line_dark_rows`),
  the growth is **+1,256 B at -O2** and +830 B at -Os (#135's audit). JUNK's and VALLEY's code is included.
- **Memory.**
  - No arena change: `vocr_lines_arena_bytes()` is unchanged.
  - No heap and no recursion.
  - The deepest frame is `pix_dark` (2,112 B) in both files. The deepest call chain is 3,656 B against
    `vocr_lines_rows.c`'s 3,648 B at -O2 (+8 B; #135's audit).
- **Finder time.** Host, gcc -O2, best of 7 per region over the 2,109 base hval regions in the console view, on a
  loaded 64-core box:

  | | mean, µs per region | p50 | p90 | p99 | max |
  |---|---:|---:|---:|---:|---:|
  | fix 0 | 426.2 | | | | |
  | COLGAP (the default), per region against fix 0 | 424.1 | 0.995 | 1.009 | 1.029 | 1.083 |
  | COLGAP + JUNK, per region against fix 0 | 424.1 | 0.994 | 1.012 | 1.041 | 1.283 |

  So there is no measurable cost for the default. JUNK's ornament test makes a few regions slower: p99 +4%, max +28%
  here. The audit's run on its own load saw p99 +6.5% and max +42%.
- **Recognition time.** COLGAP keeps a line's parts together, so slightly more is read: hval CTC frames go 540,993 →
  544,697 (+0.7%) and dev 85,618 → 86,168 (+0.6%). JUNK, opt-in, reads 5.7% fewer on hval. These are estimates at the
  console's per-frame rate, not measurements.

## 6. In Rubi

`vocr_find_lines_fix_rows(&region, VOCR_LINES_TEXTBOX_RECOVERY, flags, …)` replaces `vocr_find_lines_rows(…)`, with
the same arena. Rubi's short-line probe and its refinement on PSP titles stay. A config key chooses the flags (the
Rubi PR):
- `on` (the default): `VOCR_FIX_DEFAULT`;
- `junk`: with JUNK, for regions drawn with a margin;
- `all`: with VALLEY too;
- `off`: the frozen finder.

**To test on console:**
- a region whose line has a full-width space after ！;
- with `junk`, a PCSG00854-style box with an ornament, drawn with a margin;
- with `all`, a Narcissu page;
- time per job with the fixes on and off.

## 7. Limits

- The hooked titles are 46 train titles with dialogue regions. hval holds other pages of the same titles, so a fix
  that fits a title's style shows there too. The development units (other games, mostly PC screenshots scaled to
  the console's size) are the unseen-game guard.
- The protocol's regions all keep a margin, and Rubi's users draw their own regions, often tight. Since amendment 3,
  every finder change is also scored on tight variants of the units: hval at 0 and +2 px, dev at 1 px around the
  ink, and tune at 0 and +2 px for tuning. How tight users really draw is unmeasured.
- COLGAP comes from #119's analysis of the development units, so its dev number is not a held-out estimate.
- The labels are machine-verified (renderer cells against hook text), not hand-checked. One label swap was seen: two
  speech bubbles of NPJG00016 with the same cell count. Labels affect both arms of every comparison alike.
- Rubi's `non_japanese_filter = lines` drops lines of only ellipses or brackets (「……」), which the references hold.
  This costs every arm alike (for example 1-2 points of PCSG00611) and is not a finder matter.
- A filter that also keeps lines with CJK punctuation, a bracket or an ellipsis (`--filter lines+punct`) was looked
  at after the decision. It was not pre-registered and is not adopted. With COLGAP + JUNK:
  - hval, with COLGAP + JUNK: 1.898% → 1.843%, −0.055 [−0.134, +0.017], not resolved (PSP −0.098 [−0.246,
    −0.011]).
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
eval/pagepath.py run --units hval.jsonl $P --finder rubi,fix=1 --out F1_hval.json                 # the default
eval/pagepath.py run --units $A/vocr5-analysis/work/units.pkl $P --finder rubi,fix=1 --out F1_dev.json
eval/pagepath.py run --units hval.jsonl $P --finder rubi,fix=5 --out F13_hval.json               # + JUNK
eval/pagepath.py compare --base base_hval.json --arm F1_hval.json
# tight variants: $A/vocr6-finder/scratch/make_tight_units.py (hval/tune 0 and +2 px, dev 1 px around the ink)
make -C runtime check-lines-fix
```

Sources of the measured runs:
- The decision runs (§4) used `vocr_lines_fix.c` sha256 `746aa003…`.
- The tiers, the test title and the console crops used `9fe5427c…`.
- On every hval and dev unit the two give the same boxes, dark flags and text (a re-run of COLGAP + JUNK).
- The committed `vocr_lines_fix.c` differs from `9fe5427c…` only in a comment that names its test.
  `vocr_lines_fix.h` differs from the measured one only in `VOCR_FIX_DEFAULT` and a comment. `eval/pagepath_find.c`
  differs from the measured `3eed7809…` only in its licence note.
- The pre-registered `746aa003…` / `30de25ba…` were not kept as files. Their equivalence rests on the re-run and on
  the compiled drivers (#135's audit §1).
