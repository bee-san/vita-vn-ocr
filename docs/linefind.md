# Dialogue line finder (`runtime/vocr_lines.c`)

The recognizer reads one line at a time, so something has to cut the user's region into lines. The finder v1 was trained with is vita-jp-overlay's bright-pixel row projection (`vjo_ocr_find_lines`). It only sees near-white text. Dark text on a light box, light text on a light bar, outlined text and pale-yellow names make it reject the region or find nothing. On the research campaign's 16 real regions it rejects 4, and the pipeline CER is 28.4%.

`runtime/vocr_lines.c` replaces it with a polarity-agnostic finder. It is plain C99 with no allocation (the caller passes an arena), integer-only and bit-exact with a Python twin, `synth/common.py find_lines()`, so the training crops are the crops the console sees. The legacy finder is kept behind a flag (`VOCR_LINES_LEGACY`, `find_lines(img, "legacy")`) for comparison and to reproduce v1's data.

On real screenshots, with v1 and dark-text crops inverted, the pipeline CER goes from 28.4% to 4.4% on research set A and from 38.5% to 5.3% on set B. Line F1 on four detector-built sets goes from 0.44–0.48 to 0.63–0.72 (details below).

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

The constants were chosen on research sets A and B plus synthetic regions, before the detector-built sets existed. Step 5's split was then added after analysing psp_dev merges, and was checked on the dev sets only (see "Change log").

## API and memory

```c
VocrLinesImage im = { pixels, width, height, stride, VOCR_LINES_RGBA };   /* or GREY */
size_t need = vocr_lines_arena_bytes(960, 544);                           /* 83,323 */
VocrLineSet out;
int rc = vocr_find_lines(&im, VOCR_LINES_GRAD, arena, need, &out);       /* 0 = OK */
for (unsigned i = 0; i < out.count; i++) { /* out.lines[i].x, y, w, h, dark */ }
```

- Return codes: `VOCR_LINES_E_ARG` for bad arguments or zero sizes, and `VOCR_LINES_E_ARENA` for an arena that is too small. `VOCR_LINES_E_REGION` is returned only by the legacy finder, for the overlay's rejects.
- `out.count` is cleared on error. At most 16 lines are returned, and `truncated` is set if there were more.
- Arena: 83,323 B covers any region up to 960×544, and 25,355 B any region up to 480×272. It needs no alignment and is scratch only. Most of it is a 1-bit on-mask.
- Stack: about 3.4 KB at peak (`find_grad` 1.2 KB plus `vocr_line_dark` 2.0 KB; clang armv7 frame sizes are 1,216 B and 2,112 B).
- Code size (freestanding compile only, clang 15): 6,728 B text for armv7-a Cortex-A9 at -O2, 5,796 B at -Os, and 7,908 B for mips32 at -O2.
- `vocr_line_dark(im, x, y, w, h)` exposes the polarity rule for any box.
- `runtime/vocr_lines_test.c` is a host driver, not library code. It has `--selftest`, `--arena W H`, a batch mode used by the parity test and `evaluate.py`, and `--bench BATCH REPS`. It has its own `main()`, so library builds must leave it out.

## Polarity contract (runtime, overlay, training, synth, eval)

v1 was trained on light text only and cannot read dark-on-light text. When `VocrLine.dark` is 1, invert the crop (`255 − g`) before recognition.

- The twin does this in `common.line_crops(img, finder="grad", invert_dark=True)`.
- `synth/render.py` uses grad plus inversion by default. `--finder legacy --keep-polarity` reproduces v1's data.
- On set B this one step takes the CER from 24.6% to 5.3%.

Any recognizer trained on these crops sees light-on-dark text only. Keep the inversion in the console runtime as long as that holds.

## Ground truth (`eval/linefind/`)

Every region has line boxes in screenshot pixels. The manifests store each screenshot's VNDB URL and sha256, never the image; `build_gt.py fetch` downloads them at most 2 requests a second and checks the hashes.

| set | regions | lines (incl. ignore) | source |
|---|---|---|---|
| research_a | 16 | – | the research campaign's 16 tight regions, hand-transcribed (564 chars) |
| research_b | 19 | – | the campaign's independent replication, hand-transcribed (623 chars) |
| vita_dev | 211 | 467 | VNDB Vita screenshots, PP-OCRv5 server det boxes, reviewed by eye |
| vita_holdout | 69 | 135 | same, held-out VNs |
| psp_dev | 166 | 368 | 264 VNDB PSP screenshots (480×272) from 90 VNs (`manifests/psp_pool.jsonl`) |
| psp_holdout | 102 | 237 | same, held-out VNs |

How the detector-built sets were made:

1. `build_gt.py det vita|psp` takes the screenshot's main dialogue block from detector boxes.
2. Each screenshot gives a tight region (block + 4–12 px) and a loose one (+16–64 px sideways, as when a whole text box is selected). Every fourth screenshot also gives a text-free region from the upper screen, which tests false alarms.
3. Variants that would cut detected text are skipped. At most 2 screenshots are used per VN.
4. A VN is held out when `md5("linefind:" + vn) % 3 == 0`.
5. Every candidate set was checked by eye on contact sheets. `manifests/review.json` drops 123 unusable regions (title screens, menus, wrong boxes) and marks 11 regions' watermark or graphic lines as ignore. `build_gt.py finalize` applies it.

Data rules: screenshots with sexual < 0.3 and violence < 0.5 only. Eval's test games (`test_games.json` "all" plus `is_test_vn`) are excluded from every set except research_a/b. Those two contain v19850, v21782, v7721 and v21789 and are used only for evaluation.

Metrics (`metrics.py`):

- A prediction matches a GT line at IoU ≥ 0.5, one to one, greedy.
- An ignore line absorbs a prediction at IoU ≥ 0.3, or when ≥ 50% of the prediction lies inside it.
- `merged`: one prediction covers two or more GT lines. `split`: one GT line is covered by two or more predictions. `cut`: a matched prediction covers less than 95% of its GT line's width.
- `region exact`: no misses and no false alarms in the region.
- Pipeline CER: total edit distance divided by total reference length, comparing NFKC-normalised, whitespace-free concatenated line texts. A rejected region scores its full length.

## Results

All numbers come from `evaluate.py` with `c-grad`, the C finder through its driver. `c-grad` equals `grad`, the twin, on every metric in every set. `research` is the campaign's float gradient-energy prototype. `oracle` is the GT boxes themselves, unpadded.

Research sets with v1 (`$HERD/shared/models/v1/ckpt_best.pt`). Columns: line recall / precision / F1, region exact, merged / split / cut, rejected regions, CER with dark crops inverted (`--norm`), CER with crops as found.

| set | finder | R | P | F1 | exact | m/s/c | rej | CER inv | CER raw |
|---|---|---|---|---|---|---|---|---|---|
| A (16) | legacy | .750 | .960 | .842 | .625 | 0/0/1 | 4 | 28.4% | 28.4% |
| A | grad | 1.000 | .941 | .970 | .938 | 0/0/2 | 0 | **4.4%** | 21.1% |
| A | research | .906 | .853 | .879 | .812 | 0/1/0 | 0 | 6.2% | 23.9% |
| A | oracle | 1 | 1 | 1 | 1 | – | – | 3.9% | 20.0% |
| B (19) | legacy | .649 | .923 | .762 | .684 | 0/0/4 | 5 | 38.5% | 38.5% |
| B | grad | 1.000 | 1.000 | 1.000 | 1.000 | 0/0/7 | 0 | **5.3%** | 24.6% |
| B | research | .568 | .583 | .575 | .368 | 1/1/0 | 0 | 18.6% | 34.8% |
| B | oracle | 1 | 1 | 1 | 1 | – | – | 18.5% | 35.0% |

The oracle's CER on B is high because v1 reads the unpadded detector boxes badly; the oracle row is an upper bound only for detection. Grad on B was 4.8% when its boxes were padded 4 px at the bottom. That 1 px extra was an off-by-one against legacy's crops and was fixed, so the training crops match. On 623 characters the difference is 3 errors.

Detector-built sets, detection only (they have no transcriptions). Each cell is R / P / F1 / region exact, followed by merged / split / cut, then rejected:

| set | legacy | grad | research |
|---|---|---|---|
| vita_dev | .333/.637/.437/.265, 5/1/8, rej 106 | **.775/.661/.713/.436**, 5/1/16 | .500/.479/.489/.156, 13/1/0 |
| vita_holdout | .371/.613/.462/.246, 0/0/0, rej 31 | **.674/.614/.643/.348**, 8/0/6 | .629/.576/.601/.217, 2/2/0 |
| psp_dev | .398/.610/.482/.199, 21/0/10, rej 47 | **.646/.619/.632/.325**, 22/3/0 | .359/.402/.380/.108, 45/6/0 |
| psp_holdout | .366/.610/.457/.186, 5/0/2, rej 37 | **.732/.717/.724/.431**, 12/5/4 | .349/.414/.379/.108, 28/1/0 |

Grad by region type (region exact; false alarms; misses):

| set | tight | loose | empty (no text) |
|---|---|---|---|
| vita_dev | 0.58; 52 fp; 47 fn (91) | 0.46; 70 fp; 54 fn (83) | 0.03; 56 fp (37) |
| vita_holdout | 0.55; 15; 19 (29) | 0.32; 21; 24 (25) | 0.00; 20 (15) |
| psp_dev | 0.47; 42; 54 (68) | 0.26; 69; 74 (68) | 0.13; 33 (30) |
| psp_holdout | 0.57; 21; 29 (44) | 0.41; 30; 34 (41) | 0.12; 17 (17) |

### Change log of what was measured

The band split (step 5) was designed from psp_dev merges and checked on the dev sets with the twin:

| set | F1 before → after | exact | merged |
|---|---|---|---|
| psp_dev | 0.542 → 0.632 | 0.265 → 0.325 | 45 → 22 |
| vita_dev | 0.714 → 0.713 | 0.431 → 0.436 | 7 → 5 |
| research_a / b | 0.970 / 1.000, unchanged | unchanged | 0 |

Alternatives tried and not kept:

- A deeper valley split, cutting at rows under 1/4 of both neighbours' peaks: psp_dev 0.676, but it split single lines on research_a/b.
- Capping the widening at 1–3× the band height: ±0.006 on dev, and it lost a line on B.

The holdouts had already been scored once with the pre-split finder (psp_holdout F1 0.615, vita_holdout 0.642). They were scored once more after the split: 0.724 and 0.643. Nothing was tuned on them.

## Speed (host only)

Console timings were not measured. Host: Xeon 8488C, gcc 11.5 -O2, shared box at load ≈ 18 on 16 vCPUs. `bench_c.py` reports the best of 20 runs per region, as RGBA. Each cell gives the medians of 3 repeats in µs:

| set | grad | legacy |
|---|---|---|
| research_a regions | 478 / 555 / 558 | 44 / 52 / 46 |
| research_b regions | 957 / 1063 / 1070 | 101 / 100 / 109 |
| vita_holdout regions | 784 / 769 / 769 | 71 / 72 / 73 |
| psp_holdout regions | 260 / 322 / 304 | 19 / 23 / 20 |
| full 480×272 screen (40) | 1890 / 1682 / 1772 | 184 / 194 / 186 |
| full 960×544 screen (40) | 7287 / 6977 / 8034 | 593 / 634 / 654 |

Instruction counts are steadier than time on a loaded host. With `perf stat -e instructions:u`, `--bench` at 11 minus 1 reps, over 20 screens, x86-64:

- Grad: 69.5M per 960×544 screen and 18.2M per 480×272 screen. Before the fused contrast pass (commit da0289f) these were 76.7M and 20.1M.
- Legacy: 4.6M and 1.3M.

Grad is about 10–15× the legacy finder's cost. About two thirds of its instructions are the two 3×3 contrast passes: `contrast_row`, `put3` and `contrast_load` account for 64.6% under `perf record`, in a build with inlining disabled so functions can be attributed. The contrast is computed twice, once for the threshold histogram and once for the mask, so that no contrast image has to be stored in the arena.

The following is an unmeasured estimate, not a measurement. If a 444 MHz Cortex-A9 ran a similar instruction count at 0.5–1 instructions per cycle, a whole Vita screen would take roughly 0.15–0.3 s, and a typical dialogue region about a tenth of that.

## Reproduce

```sh
V=python   # an env with numpy, Pillow, requests; torch only for --rec
$V eval/linefind/build_gt.py fetch                     # screenshots named by the manifests
M=eval/linefind/manifests
# detection + v1 CER on the research sets (c-grad builds build/vocr_lines_test on first use)
$V eval/linefind/evaluate.py --manifest $M/research_a.jsonl --manifest $M/research_b.jsonl \
   --finders legacy,grad,c-grad,research,oracle --rec models/v1/ckpt_best.pt --norm
# detection on the detector-built sets
$V eval/linefind/evaluate.py --manifest $M/vita_dev.jsonl --manifest $M/vita_holdout.jsonl \
   --manifest $M/psp_dev.jsonl --manifest $M/psp_holdout.jsonl --finders legacy,grad,c-grad,research
# C vs twin, sanitized
cc -std=c99 -O1 -g -fsanitize=address,undefined -fno-sanitize-recover=all \
   runtime/vocr_lines.c runtime/vocr_lines_test.c -o build/vl_san
$V tests/linefind_parity.py --bin build/vl_san --cases 600 \
   $(for s in research_a research_b vita_dev vita_holdout psp_dev psp_holdout; do echo --regions $M/$s.jsonl; done)
# host timing
$V eval/linefind/bench_c.py --manifest $M/research_b.jsonl --manifest $M/psp_holdout.jsonl --full 40
```

On this branch the parity run gives `LINEFIND_PARITY_OK 2366 runs (1183 regions)` under gcc and clang ASan+UBSan and under gcc -O2. The 1,183 regions are 600 procedural and 583 real. `tests/test_linefind.py` runs a smaller parity check in CI.

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
- The eval agent's benchmark sets (`eval/`, unmerged when this was written).
- Recognition CER on the detector-built sets, which have no transcriptions.
