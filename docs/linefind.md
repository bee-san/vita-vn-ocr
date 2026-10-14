# Dialogue line finder (`runtime/vocr_lines.c`)

A [rejected horizontal-padding experiment](dialogue-padding.md) is archived with its frozen source revision and aggregate results. Its runtime change is excluded; the original preprocessing remains the default.

An optional post-finder refinement, `runtime/vocr_lines_refine.c` (separate from this frozen file), cuts each GRAD box back to its text band, drops frame rows, splits merged rows and trims cursors and portrait edges. It is a gain on PSP-sized text and on the development cohorts, not on native Vita text in the protected sets: see [finder-tighten](finder-tighten.md).

The recognizer reads one line at a time, so something has to cut the user's region into lines. The finder v1 was trained with is vita-jp-overlay's bright-pixel row projection (`vjo_ocr_find_lines`). It only sees near-white text. Dark text on a light box, light text on a light bar, outlined text and pale-yellow names make it reject the region or find nothing. On the research campaign's 16 real regions it rejects 4, and the pipeline CER is 28.4%.

`runtime/vocr_lines.c` replaces it with a polarity-agnostic finder. It is plain C99 with no allocation (the caller passes an arena), integer-only and bit-exact with a Python twin, `synth/common.py find_lines()`, so the training crops are the crops the console sees. The legacy finder is kept behind a flag (`VOCR_LINES_LEGACY`, `find_lines(img, "legacy")`) for comparison and to reproduce v1's data.

`VOCR_LINES_TEXTBOX` is a separate optional C finder for whole dialogue boxes. It computes GRAD first and keeps its ordinary crops, padding and polarity unchanged. Components can repair a crop only when they prove multiple separated text rows inside an oversized old crop, or supply a coherent missing row beside a wide, tall contaminated projection. This conservative operation avoids changing good crops merely because another detector found a different rectangle.

The component stage shares GRAD's contrast mask and 10-pixel joining gap, with an eight-pixel minimum run local to this mode. Four rows of connected runs form components; overlapping nearby fragments are coalesced. Components over 96 pixels tall cannot propose a row. Punctuation recovery and three-pixel vertical growth/padding happen before duplicate detection. Final recovered rectangles whose horizontal and vertical overlap is at least 90% of both rectangles are coalesced before any topology decision.

The largest consistent cluster of at least two old row heights anchors the main height: heights within 25% count as consistent, and ties prefer the larger height. Without a repeated-height anchor, a geometry-matched old row supplies it; incompatible single-row anchors return the original result. Otherwise the weighted median of broad component heights supplies a proposal, weighted by mean horizontal kept-run support. A new row is broad when this support is at least twice its core height. Accepted repair rows have core heights between 55% and twice the main height. Empty old output can also use a short/narrow component, without blanket edge or shape rejection.

An old crop is replaced only if it is taller than twice the main height plus six pixels and contains at least two coherent row centers separated by the main height. A missing row can be added when an old crop spans at least 90% of the ROI width and its height minus the nominal six-pixel padding is at least 1.5 times the main height. New rows overlapping at least half of a retained row's vertical span are discarded, preserving the old row exactly. Component-capacity failure or ambiguous evidence leaves the GRAD result unchanged. Reading order is top to bottom, then left to right.

This geometry cannot identify all UI art, repair missing words within an otherwise ordinary row, or recover pixels omitted by the user's crop. Glyph rows under eight pixels retain the existing finder limitation. Real-screenshot CER and console timings must be measured before promotion; generated geometry checks are not accuracy measurements.

`VOCR_LINES_TEXTBOX_RECOVERY` (mode 3, `Finder("textbox_recovery")`) optionally
extends TEXTBOX with independently supported missing rows. It retains existing
TEXTBOX crops and uses component core heights, compatible neighboring rows and
complete detached-mark spans to recover uncovered text. Its caller arena is
unchanged. The frozen six-title development comparison passes its prospective
gate; independent test accuracy and physical-console performance remain pending.
See [the algorithm and DEVELOPMENT ONLY results](native-textbox-recovery.md).

The optional stage uses the old output array in place and reuses dead GRAD scratch. For `R = floor((width + 10) / 18)` and `b = band_cap(height)`, its runs, live components and candidates need `160R + 20b` bytes, bounded by the existing `15*width + 4*height + 24b` scratch space for width/height at least eight. At 8×8 both are exactly 200 bytes, so `vocr_lines_arena_bytes()` remains 235 there and unchanged elsewhere. There is no heap allocation, additional output-array copy or recursion. Scalars and the existing four-row count array use bounded stack; target stack/code size and time require separate console builds or measurements. `eval/linefind/cfinder.py Finder("textbox")` calls this C mode; GRAD/legacy and their historical protocols remain unchanged.

Out of sample, on eval's hand-checked vita-test (PR #2: 59 regions from 12 VNs that share no VN or screenshot with the tuning sets), v1's pipeline CER goes from 40.6% with the legacy finder to 21.1% with grad and dark-text crops inverted. With the ground-truth boxes it is 15.4%. Line F1 on the four detector-built sets goes from 0.41–0.48 to 0.58–0.72. On research sets A and B, which the constants were tuned on, CER goes from 28.4% to 4.4% and from 38.5% to 5.3%; those two are in-sample numbers (details below).

## Method

All steps work on the region the user selected: a tight box or a whole text box, up to 960×544.

1. Grey: `g = (77 R + 150 G + 29 B) >> 8`, the recognizer's own luma.
2. Local contrast: `c = max − min` of `g` over each 3×3 window, with edges replicated. Strokes are high-contrast whatever their polarity, and an outline adds contrast instead of hiding the fill.
3. Threshold: `T = max(40, 2/5 · p99(c))`. A pixel is on when `c ≥ T`.
4. Runs: in each row, on-pixels at most 10 px apart join into runs, and runs of at least 16 px are kept. Text lines become long runs; specks, vertical borders and the edges of large shapes do not. `prof(y)` is the kept run length of row y.
5. Bands: rows with `prof ≥ max(16, max(prof)/16)` are active. Active rows separated by gaps of up to 2 rows form bands, and bands under 8 rows are dropped. A bridged gap that contains an empty row (`prof = 0`), with at least 8 active rows directly above and below it, ends the band instead. This keeps lines set 1–2 px apart, common on PSP, as separate lines.
6. Per band:
   - The line is the column cluster with the most kept pixels. Clusters split at gaps over twice the band height.
   - It is widened over on-pixels at most `max(4, h)` columns away, for punctuation and small kana.
   - It is grown vertically by up to `h/2` rows, never past half the gap to the next band, while rows hold 2 or more on-pixels.
7. Lines under 55% of the median line height are dropped (furigana, icons).
8. Boxes are padded by 3 px. After step 2's 1 px dilation this equals the legacy finder's 4 px around the ink. On 400 synthetic regions the median box edge difference against legacy is 0 on every side.
9. Polarity per box:
   - background = median of the box's border pixels;
   - `d = max(16, (p98 − p2)/4)`;
   - the box is dark when the count of `g < bg − d` exceeds 8 × the count of `g > bg + d`.

The constants were chosen on research sets A and B plus synthetic regions, before the detector-built sets existed, so A and B are tuning sets and their results are in-sample. Step 5's split was then added after analysing psp_dev merges, and was checked on the dev sets only (see "Change log").

## API and memory

```c
VocrLinesImage im = { pixels, width, height, stride, VOCR_LINES_RGBA };   /* or GREY */
size_t need = vocr_lines_arena_bytes(960, 544);                           /* 83,323 */
VocrLineSet out;
int rc = vocr_find_lines(&im, VOCR_LINES_GRAD, arena, need, &out);       /* 0 = OK */
for (unsigned i = 0; i < out.count; i++) { /* out.lines[i].x, y, w, h, dark */ }
```

- Return codes: `VOCR_LINES_E_ARG` for bad arguments or zero sizes, and `VOCR_LINES_E_ARENA` for an arena that is too small. `VOCR_LINES_E_REGION` is returned only by the legacy finder, for the overlay's rejects. The row-callback form below adds `VOCR_LINES_E_SOURCE`, when a row could not be read.
- `out.count` is cleared on error. At most 16 lines are returned, and `truncated` is set if there were more.
- Arena: 83,323 B covers any region up to 960×544, and 25,355 B any region up to 480×272. It needs no alignment and is scratch only. Most of it is a 1-bit on-mask.
- Stack: about 3.4 KB at peak (`find_grad` 1.2 KB plus `vocr_line_dark` 2.0 KB; clang armv7 frame sizes are 1,216 B and 2,112 B).
- Code size (freestanding compile only, clang 15): 6,728 B text for armv7-a Cortex-A9 at -O2, 5,796 B at -Os, and 7,908 B for mips32 at -O2.
- External symbols: `memset`, plus on 32-bit targets libgcc's 64-bit division helper (`__aeabi_uldivmod` on ARM, `__udivdi3` on MIPS) for the percentiles in `vocr_line_dark`. VitaSDK and PSPSDK link libgcc by default.
- `vocr_line_dark(im, x, y, w, h)` exposes the polarity rule for any box.
- Row-callback form, for a caller that streams the region (vita-jp-overlay reads it from the game's framebuffer in strips and holds no copy of the whole region): `runtime/vocr_lines_rows.h` / `vocr_lines_rows.c`, `vocr_find_lines_rows(&src, ...)` and `vocr_line_dark_rows(&src, ...)` with a `VocrLinesRows` `{width, height, bpp, row, user}` and the same arena (`vocr_lines_arena_bytes`). `row(user, y)` returns row `y`'s `width × bpp` bytes. The finders ask for rows in increasing order within a pass and make two passes over the region (one for the legacy finder, which may stop early), then one over each line's box; a returned row only has to stay valid until the next call. `NULL` ends the call with `VOCR_LINES_E_SOURCE` (−4) and no lines. `vocr_lines.c` itself is frozen (pre-registered protocols record its sha256, e.g. `eval/fresh_native_textbox.py` `FINDER_SHA`), so the row form is a separate file: `vocr_lines.c` with every pixel read routed through one row accessor and the in-memory entry points replaced. It names the `vocr_lines.c` it was derived from, and `tests/test_vocr_lines_rows_source.py` fails when that file changes without it. `runtime/tests/test_vocr_lines_rows.c` (in `make check-lines`) checks that both return identical lines, flags and return codes on 300 procedural regions plus edge sizes, for every finder and both pixel formats, with a callback that poisons each row once the next one is asked for.
- `runtime/vocr_lines_test.c` is a host driver, not library code. It has `--selftest`, `--arena W H`, a batch mode used by the parity test and `evaluate.py`, and `--bench BATCH REPS`. It has its own `main()`, so library builds must leave it out.

## Polarity contract (runtime, overlay, training, synth, eval)

v1 was trained on light text only and cannot read dark-on-light text. When `VocrLine.dark` is 1, invert the crop (`255 − g`) before recognition.

- The twin does this in `common.line_crops(img, finder="grad", invert_dark=True)`.
- The recognizer runtime does it without a copy of the crop: `img.format |= VOCR_INVERT` (`runtime/vocr.h`) inverts each source pixel before the resize, exactly as `line_crops` does.
- `eval/run.py --polarity console` scores a recognizer behind this step.
- On eval's vita-test (the 59 out-of-sample regions) this one step takes grad's CER from 32.6% to 21.1%; on set B, from 24.6% to 5.3%.
- `synth/render.py` (synth2) cuts crops with an ideal finder on the renderer's ink mask and keeps their rendered polarity by default, so a recognizer trained on its shards reads both polarities. `--invert-dark` stores crops the way the console hands them over. The rule is not a clean split on synthetic styles: it calls 84% of dark-on-light crops dark, but also 40% of outlined light text on light boxes and 21% of dark outlined text (`calibrate.py finders --n 1000 --seed 5`).
- Recognizers trained on synth2's default crops score within 0.3 points with and without the inversion (vita-test 5.38% and 5.11% CER for the 5,000-step A/B model), while light-only ones need it (the same recipe on v1's data: 29.25% and 13.18%). See `docs/synth2.md`.
- v1's data came from the renderer before synth2 (`render.py --finder legacy --keep-polarity` at b4c584e); its shards are in `$HERD/shared/data/synth`.

Keep the inversion in the console runtime: light-only recognizers need it, and recognizers trained on synth2's default crops are indifferent to it.

## Ground truth (`eval/linefind/`)

Every region has line boxes in screenshot pixels. The manifests store each screenshot's VNDB URL and sha256, never the image; `build_gt.py fetch` downloads them at most 2 requests a second and checks the hashes.

| set | regions | lines (incl. ignore) | source |
|---|---|---|---|
| research_a | 16 | – | tuning set: the research campaign's 16 tight regions, hand-transcribed (564 chars) |
| research_b | 19 | – | tuning set: the campaign's replication, transcribed independently (623 chars); 7 of its 19 screenshots are also in A, so A ∪ B has 28 |
| vita_dev | 206 | 455 | VNDB Vita screenshots, PP-OCRv5 server det boxes, reviewed by eye |
| vita_holdout | 48 | 95 | same, held-out VNs |
| psp_dev | 166 | 368 | VNDB PSP screenshots (480×272) of `manifests/psp_pool.jsonl`: 252 screenshots of 86 VNs |
| psp_holdout | 90 | 207 | same, held-out VNs |

Out-of-sample CER comes from eval's own hand-checked set instead (`evaluate.py --eval-set vita-test`, see Results).

How the detector-built sets were made:

1. `build_gt.py det vita|psp` takes the screenshot's main dialogue block from detector boxes.
2. Each screenshot gives a tight region (block + 4–12 px) and a loose one (+16–64 px sideways, as when a whole text box is selected). Every fourth screenshot also gives a text-free region from the upper screen, which tests false alarms.
3. Variants that would cut detected text are skipped. At most 2 screenshots are used per VN.
4. A VN is held out when `md5("linefind:" + vn) % 3 == 0`.
5. Every candidate set was checked by eye on contact sheets. `manifests/review.json` drops 123 unusable regions (title screens, menus, wrong boxes) and marks 11 regions' watermark or graphic lines as ignore.
6. `build_gt.py finalize` applies the review and the data rules below to the reviewed candidates (`SET.candidates.jsonl`).

Data rules:

- Screenshots with sexual < 0.3 and violence < 0.5 only.
- No region from one of eval's test games: `is_test_vn`, plus the "all" list of `eval/manifests/test_games.json` (until eval merges, its shared copy). That list grows as eval builds its sets, so `finalize` must be rerun when it does. `tests/test_linefind.py` checks the committed sets against the repo's copy. The `.candidates.jsonl` files are the reviewed snapshot: running `det` again today would propose different regions, because the list has changed since they were built. At the last `finalize` (172 listed games) it dropped 5 vita_dev regions (v3, v18717), 12 psp_holdout regions and 12 pool screenshots (v9915, v10606, v11709, v12159).
- No holdout region from a VN of research_a/b. vita_holdout lost 21 reviewed regions of 7 such VNs, 11 of them on the very screenshots A and B use. vita_dev keeps its A/B VNs, since dev is for tuning anyway.
- research_a/b are exempt because they are the tuning sets, not test sets. They contain eval test games (v3, v7721, v19850, v21782, v21789), and 7 of their regions sit on 6 screenshots of eval's vita-test: A_v7721_0, A_v19850_0, A_v19850_1 (= B_14_v19850), A_v21782_2, B_03_v21782 and B_18_v21789. So the finder's constants were partly chosen on those 6 vita-test screenshots. `evaluate.py --eval-set` leaves out every vita-test region of those VNs and screenshots, and pipeline numbers on the full vita-test should say that they include them.

Metrics (`metrics.py`):

- A prediction matches a GT line at IoU ≥ 0.5, one to one, greedy.
- An ignore line absorbs a prediction at IoU ≥ 0.3, or when ≥ 50% of the prediction lies inside it.
- `merged`: one prediction covers two or more GT lines. `split`: one GT line is covered by two or more predictions. `cut`: a matched prediction covers less than 95% of its GT line's width.
- `region exact`: no misses and no false alarms in the region.
- Pipeline CER: total edit distance divided by total reference length, comparing NFKC-normalised, whitespace-free concatenated line texts. A rejected region scores its full length.

## Results

All numbers come from `evaluate.py`. `c-grad`, the C finder through its driver, equals `grad`, the twin, box for box in every set below. `research` is the campaign's float gradient-energy prototype. `oracle` is the GT boxes themselves, unpadded. The recognizer is v1 (`$HERD/shared/models/v1/ckpt_best.pt`, also `tinyvnocr-v1-ckpt_best.pt` of release v0.1.0-baseline). Columns: line recall / precision / F1, region exact, merged / split / cut, rejected regions, CER with dark crops inverted (`--norm`), CER with crops as found.

### Out of sample: eval's vita-test

This is eval PR #2's hand-checked vita-test (`eval/manifests/vita-test.jsonl` at aac4170), regrouped into its 84 regions. 2 regions with an uncertain line are left out. "Clean" also leaves out the 23 regions whose VN or screenshot is in research A/B, which leaves 59 regions, 147 lines, 12 VNs and 1,850 characters. "All" keeps them (82 regions, 16 VNs, 2,592 characters). The CER is this tool's region-level CER (concatenated lines, above), not `eval/run.py`'s per-line pipeline CER.

| regions | finder | R | P | F1 | exact | m/s/c | rej | CER inv | CER raw |
|---|---|---|---|---|---|---|---|---|---|
| clean (59) | legacy | .537 | .823 | .650 | .373 | 0/0/0 | 17 | 40.6% | 40.6% |
| clean | grad | .810 | .804 | .807 | .593 | 0/0/5 | 0 | **21.1%** | 32.6% |
| clean | research | .463 | .531 | .495 | .169 | 7/0/0 | 0 | 34.2% | 43.8% |
| clean | oracle | 1 | 1 | 1 | 1 | – | – | 15.4% | 27.0% |
| all (82) | legacy | .535 | .806 | .643 | .378 | 1/0/0 | 24 | 41.8% | 41.8% |
| all | grad | .797 | .801 | .799 | .573 | 0/0/6 | 0 | 20.3% | 33.6% |
| all | oracle | 1 | 1 | 1 | 1 | – | – | 15.8% | 28.8% |

Clean regions by eval's style tag, legacy → grad, CER inverted (`--by-tag`; a region carries every tag of its lines):

| tag | regions | legacy rejected | legacy CER | grad CER | oracle CER |
|---|---|---|---|---|---|
| bright-on-dark | 46 | 6 | 27.4% | 18.0% | 13.8% |
| dark-on-light | 18 | 11 | 84.6% | 28.7% | 20.5% |
| outlined | 22 | 12 | 69.9% | 29.5% | 19.5% |
| name | 42 | 13 | 45.9% | 24.0% | 17.6% |
| mincho | 17 | 0 | 8.3% | 8.6% | 8.9% |

Grad is 5.7 points above the oracle here, against 0.5 on set A, where it was tuned. Most of the remaining CER is v1's own: it is 15.4% even on the ground-truth boxes.

### Tuning sets: research A and B (in-sample)

| set | finder | R | P | F1 | exact | m/s/c | rej | CER inv | CER raw |
|---|---|---|---|---|---|---|---|---|---|
| A (16) | legacy | .750 | .960 | .842 | .625 | 0/0/1 | 4 | 28.4% | 28.4% |
| A | grad | 1.000 | .941 | .970 | .938 | 0/0/2 | 0 | 4.4% | 21.1% |
| A | research | .906 | .853 | .879 | .812 | 0/1/0 | 0 | 6.2% | 23.9% |
| A | oracle | 1 | 1 | 1 | 1 | – | – | 3.9% | 20.0% |
| B (19) | legacy | .649 | .923 | .762 | .684 | 0/0/4 | 5 | 38.5% | 38.5% |
| B | grad | 1.000 | 1.000 | 1.000 | 1.000 | 0/0/7 | 0 | 5.3% | 24.6% |
| B | research | .568 | .583 | .575 | .368 | 1/1/0 | 0 | 18.6% | 34.8% |
| B | oracle | 1 | 1 | 1 | 1 | – | – | 18.5% | 35.0% |

The constants were chosen on these two sets, so these are not test results. The oracle's CER on B is high because v1 reads the unpadded detector boxes badly; the oracle row is an upper bound only for detection. Grad on B was 4.8% when its boxes were padded 4 px at the bottom. That 1 px extra was an off-by-one against legacy's crops and was fixed, so the training crops match. On 623 characters the difference is 3 errors.

### Detector-built sets

These are detection only (they have no transcriptions), on the sets as `finalize` now writes them. Each cell is R / P / F1 / region exact, followed by merged / split / cut, then rejected:

| set | legacy | grad | research |
|---|---|---|---|
| vita_dev (206) | .337/.639/.441/.272, 3/1/8, rej 103 | **.782/.662/.717/.442**, 5/1/16 | .500/.477/.488/.160, 13/1/0 |
| vita_holdout (48) | .304/.622/.409/.250, 0/0/0, rej 27 | **.598/.567/.582/.312**, 8/0/5 | .609/.549/.577/.250, 2/2/0 |
| psp_dev (166) | .398/.610/.482/.199, 21/0/10, rej 47 | **.646/.619/.632/.325**, 22/3/0 | .359/.402/.380/.108, 45/6/0 |
| psp_holdout (90) | .322/.606/.420/.189, 5/0/0, rej 37 | **.727/.716/.722/.444**, 12/5/4 | .361/.430/.393/.122, 24/1/0 |

Before the data rules were applied (211 / 69 / 166 / 102 regions), grad scored .713 / .643 / .632 / .724 F1. The drop on vita_holdout is the 21 regions from tuning-set VNs, which grad found well.

Grad by region type (`--by-tag`: region exact; false alarms; misses; regions):

| set | tight | loose | empty (no text) |
|---|---|---|---|
| vita_dev | 0.59; 50 fp; 44 fn (88) | 0.47; 68 fp; 51 fn (81) | 0.03; 56 fp (37) |
| vita_holdout | 0.45; 13; 17 (20) | 0.33; 15; 20 (18) | 0.00; 14 (10) |
| psp_dev | 0.47; 42; 54 (68) | 0.26; 69; 74 (68) | 0.13; 33 (30) |
| psp_holdout | 0.59; 18; 26 (39) | 0.44; 26; 30 (36) | 0.07; 15 (15) |

### Change log of what was measured

The band split (step 5) was designed from psp_dev merges and checked on the dev sets with the twin. The table compares the twin before it (commit 6f154f3) and after it, both on the sets as they are now:

| set | F1 before → after | exact | merged |
|---|---|---|---|
| psp_dev | 0.542 → 0.632 | 0.265 → 0.325 | 45 → 22 |
| vita_dev | 0.718 → 0.717 | 0.437 → 0.442 | 7 → 5 |
| research_a / b | 0.970 / 1.000, unchanged | unchanged | 0 |
| psp_holdout | 0.614 → 0.722 | 0.344 → 0.444 | 24 → 12 |
| vita_holdout | 0.587 → 0.582 | 0.312 → 0.312 | 8 → 8 |

Alternatives tried and not kept:

- A deeper valley split, cutting at rows under 1/4 of both neighbours' peaks: psp_dev 0.676, but it split single lines on research_a/b.
- Capping the widening at 1–3× the band height: ±0.006 on dev, and it lost a line on B.

The holdouts, in their first form, were scored once with the pre-split finder (psp_holdout F1 0.615, vita_holdout 0.642) and once after the split (0.724 and 0.643). Nothing was tuned on them.

## Speed (host only)

Console timings were not measured. Host: Xeon 8488C, gcc 11.5 -O2, shared box at load ≈ 10–11 on 16 vCPUs. `bench_c.py` reports the best of 20 runs per region, as RGBA. Each cell gives the medians of 3 repeats in µs:

| set | grad | legacy |
|---|---|---|
| research_a regions | 346 / 414 / 414 | 42 / 46 / 39 |
| research_b regions | 753 / 748 / 778 | 89 / 88 / 90 |
| vita_holdout regions | 550 / 548 / 756 | 46 / 48 / 56 |
| psp_holdout regions | 212 / 209 / 315 | 20 / 18 / 22 |
| full 480×272 screen (40) | 1419 / 1476 / 1842 | 160 / 229 / 192 |
| full 960×544 screen (40) | 5242 / 5350 / 5727 | 550 / 615 / 574 |

Instruction counts are steadier than time on a loaded host. With `perf stat -e instructions:u`, `--bench` at 11 minus 1 reps, over 20 screens, x86-64:

- Grad: 69.5M per 960×544 screen and 18.2M per 480×272 screen. Before the fused contrast pass (commit da0289f) these were 76.7M and 20.1M. The review's rerun on 20 other screens of each size, from the detector-built sets, gave 70.4M and 18.3M.
- Legacy: 4.6M and 1.3M (rerun: 4.2M and 1.2M).

Grad is about 10–15× the legacy finder's cost. About two thirds of its instructions are the two 3×3 contrast passes: `contrast_row`, `put3` and `contrast_load` account for 64.6% under `perf record`, in a build with inlining disabled so functions can be attributed. The contrast is computed twice, once for the threshold histogram and once for the mask, so that no contrast image has to be stored in the arena.

The following is an unmeasured estimate, not a measurement. If a 444 MHz Cortex-A9 ran a similar instruction count at 0.5–1 instructions per cycle, a whole Vita screen would take roughly 0.15–0.3 s, and a typical dialogue region about a tenth of that.

## Reproduce

```sh
V=python   # an env with numpy, Pillow, requests; torch only for --rec
$V eval/linefind/build_gt.py fetch                     # screenshots named by the manifests
M=eval/linefind/manifests
REC=$HERD/shared/models/v1/ckpt_best.pt                # or tinyvnocr-v1-ckpt_best.pt from release v0.1.0-baseline
# out of sample: eval's vita-test (screenshots from eval/fetch.py, in $VOCR_EVAL_DATA/vita/shots;
# before eval merges: git show aac4170:eval/manifests/vita-test.jsonl > vita-test.jsonl, and
# pass the file instead of the name). Add --with-tuning for all 82 regions, --by-tag per style.
VOCR_EVAL_DATA=$HERD/shared/artifacts/eval $V eval/linefind/evaluate.py --eval-set vita-test \
   --finders legacy,grad,c-grad,research,oracle --rec $REC --norm --by-tag
# the tuning sets, with v1 (c-grad builds build/vocr_lines_test on first use)
$V eval/linefind/evaluate.py --manifest $M/research_a.jsonl --manifest $M/research_b.jsonl \
   --finders legacy,grad,c-grad,research,oracle --rec $REC --norm
# detection on the detector-built sets (after eval's test-game list changes: build_gt.py finalize)
$V eval/linefind/evaluate.py --manifest $M/vita_dev.jsonl --manifest $M/vita_holdout.jsonl \
   --manifest $M/psp_dev.jsonl --manifest $M/psp_holdout.jsonl --finders legacy,grad,c-grad,research --by-tag
# what CI runs (ci/runtime.sh): strict compile, sanitized selftest and parity on 400 procedural regions
make -C runtime check PYTHON=$(command -v $V)          # SAN_ENV="env LD_LIBRARY_PATH=..." if libasan is not found
# C vs twin, sanitized, with every real region
cc -std=c99 -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all \
   runtime/vocr_lines.c runtime/vocr_lines_test.c -o build/vl_san
$V tests/linefind_parity.py --bin build/vl_san --cases 600 \
   $(for s in research_a research_b vita_dev vita_holdout psp_dev psp_holdout; do echo --regions $M/$s.jsonl; done)
# host timing
$V eval/linefind/bench_c.py --manifest $M/research_b.jsonl --manifest $M/psp_holdout.jsonl --full 40
```

On this branch the parity run gives `LINEFIND_PARITY_OK 2290 runs (1145 regions)` under gcc and clang ASan+UBSan and under gcc -O2. The 1,145 regions are 600 procedural and 545 real. In CI, `ci/runtime.sh` runs `make -C runtime check` under both compilers with ASan and UBSan, which is the strict compile, `--selftest` and parity on 400 procedural regions (no screenshots in CI), and `tests/test_linefind.py` runs a smaller unsanitized parity check.

## Limitations

- Adjacent lines still merge on PSP (psp_dev 22, psp_holdout 12). The lines there are 16–19 px tall. In the 12 psp_dev merges whose row profiles were inspected, no row between the two lines was empty, so step 5 cannot split them.
- Textured artwork gives false alarms: almost every "empty" region gets a line, and loose regions pick up decoration around the box. The finder assumes the region holds text, which is true when the user selects a dialogue box.
- In loose regions, a line next to textured background can widen into it, dropping its IoU below 0.5. This is the main source of `cut` and of misses in loose regions.
- A speaker name sitting on a decorated plate can merge with the decoration above it.
- Outlined light text on a light box is classified light, which is correct, but only while the fill is brighter than the box by more than `d`. The probe cases (250, 200, outline 20) and (255, 200, outline 20) come out dark.
- Vertical text is not handled. Neither finder handles it.
- Speed has been measured only on the host. It is about 10–15× the legacy finder.

## Not run

- Console or emulator timing, and ARM/MIPS instruction counts. Only freestanding compiles were done; there is no qemu or cross toolchain on the host.
- `-m32` builds: no 32-bit glibc on the host.
- An out-of-sample PSP pipeline number: eval's psp-test is not published yet. `evaluate.py --eval-set psp-test` will read it once it is.
- `eval/run.py`'s own pipeline CER for this finder. The vita-test numbers above use this tool's region-level CER on eval's manifest (PR #2, aac4170), with dark crops inverted as the console does.
- Recognition CER on the detector-built sets, which have no transcriptions.
