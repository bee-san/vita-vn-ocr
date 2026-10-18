# textgate: the smallest text-presence gate in front of GRAD + refine (issue #100)

When a region holds no text (a hidden text box, a CG screen, a menu without dialogue) the console's finder still
returns lines and the recognizer reads junk from them: GRAD and GRAD + refine return no line for only 5 of the 66
text-free development regions of `eval/linefind` (`docs/textdet.md`). #97's detector used as a gate (no lines when
it finds none) fixed that for 0.92 of them at a cost of 0.02-0.03 points of development whole-dialogue CER, but
Rubi_for_vita #209 declined its 8,360 B of worker code and 5,700 B of weights. This work looks for the smallest gate
that does the same job: a yes/no per region, GRAD + refine unchanged behind it.

This document starts as the pre-registered protocol (`eval/textgate_protocol.json`, written by
`eval/textgate_make_protocol.py`), frozen before any development score of any gate. Results are added below it;
nothing above "Results" changes except through a dated amendment.

## Protocol

### What a gate is

A gate decides per region. "No text": the finder returns no line, so nothing is recognized and the recognizer's
weights are never opened. "Text": GRAD + refine run as they do today (`runtime/vocr_lines.c` `VOCR_LINES_GRAD`, then
`runtime/vocr_lines_refine.c` `VOCR_REFINE_DEFAULT`).

- **Learned gates** are int8 networks in #97's VTDT layout (`synth/textdet_twin.py`), run on the region's grey; the
  head's first map K is a text logit per output cell (stride 4 or 8). The rest of the head is unused. Two modes:
  - `region`: score = the k-th largest K code of the map (the smallest when it has fewer cells);
  - `boxes`: score = the k-th largest K code over the cells whose block meets a GRAD box (-129 when none does).
  The region holds text iff score ≥ t.
- **Non-learned baselines**, which the networks must beat: `blank`, Rubi's blank-line check (`core/vocr_ocr.c`
  `vjo_vocr_ink`: a line whose crop has fewer than 50 edges per 10,000 pixels is blank; the region holds text iff
  some GRAD + refine line is not blank), and `rule8`, the longest horizontal run of 8 x 8 blocks with at least 16
  edge pixels (|dx| or |dy| ≥ 24) and a contrast of at least 48 (text iff the run reaches t blocks).

### Operating points, fixed on training-side data

Every threshold comes from calibration cohorts of training-side VNs (`eval/textgate_data.py cal`): a fifth of the
training-side VNs by hash (`sha256("textgate-split:" + vn)`), never trained on, never a development, test or
eval/linefind VN or screenshot.

- `cal_dlg`: whole-dialogue regions built exactly as the development cohort is (`eval/finder_tighten._frame_units`:
  labelled non-name lines grouped at 1.6 line heights, their union plus 12 / 6 px, grown by 0 / 4 / 8 px, shrunk away
  from other text): 11,103 cases of 3,701 regions from 1,810 VNs (realplus PC screenshots at the Vita target, the
  teacher's Vita, PSP and scaled PC frames, the owner's console captures).
- `cal_empty`: text-free regions cut as eval/linefind's empty regions are (300-600 x 60-140 px, half on the PSP, in
  the upper 45% of the frame, at least 10 px from every detected text box): 3,344 regions from 1,805 VNs.

For each candidate, mode and target share τ ∈ {0.998, 0.999}: for each k ∈ {1, 2, 4}, t is the largest code that
keeps at least τ of the cal_dlg cases GRAD + refine returns a line for (the gate cannot lose the others); the chosen k is the one whose (k, t) leaves the most cal_empty regions with no
line (the gate says no, or GRAD + refine returns none), the smaller k on a tie. **Screen:** an operating point goes
on to development only if that cal_empty rate is at least 0.85.

### Candidates

- **New presence-only networks** (`train/textgate_train.py`): #97's architecture family (a 3x3 stride-2 stem,
  separable downsampling, separable blocks, a 1x1 head) at output stride 8 with 4-12 channels (`p8_c4`, `p8_c6`,
  `p8_c8`, `p8_c12`) and stride 4 with 4-8 (`p4_c4`, `p4_c8`); seed 1 for each, seed 2 for `p8_c6` and `p8_c8`.
  Trained for presence only: a cell is text when its block's centre lies inside a horizontal line's box; BCE + dice
  plus a term on each window's highest logit. #97's windows, masks, optimiser and QAT; 24,000 steps; half real, half
  synthetic.
- **#97's trained detectors read as presence gates**: its 22 runs under 4,100 B (`s4_c4` to `s8_c16n4`), unchanged.
- **The baselines** `blank` and `rule8`.
- **Report only**: #97's `s8_c24_seed4` (5,700 B), calibrated like the others, and as #97's own gate.

Training data of the new networks: #97's real shards (`data/manifests/textdet_shards_v1.json`: realplus, teacher,
devnative and the owner's) without the calibration split's VNs, and clean synthetic shards
(`data/manifests/textgate_shards_v1.json`, 24 shards, 48,000 canvases): #97's generator with synth3's scene pool cut
further, to no screenshot of a VN of eval/linefind's four detector-built sets, no screenshot any eval or
eval/linefind manifest lists, and no CHAOS;CHILD (vita scenes 980 → 120, PSP 1,382 → 1,035, scaled PC 3,181 →
3,137). **#97's own synthetic shards are not used**: their scene pool held 54 of the 66 development empty regions'
screenshots (#97's caveat). The gate's training data therefore hold no eval/linefind screenshot or VN at all.

### Development data (the rule reads only these)

- **wd**: VOCR5's development whole dialogue as #97 scored it (1,515 cases, 1,161 pooled units, 93 VNs). A case the
  gate passes keeps its GRAD + refine unit (#97's frozen baseline units, `eval/textdet_baseline_20261017.json`); a
  case it stops reads nothing, every character a deletion. Gate − refine in points, variants pooled, with 10,000
  VN-cluster bootstrap resamples (`eval/textdet.vn_diff`).
- **native_vita**: the same on the native PS Vita frames (150 units, 9 VNs), the CLANNAD-dev-style subset.
- **empties**: eval/linefind's 66 development empty regions (vita_dev + psp_dev, a reserved test VN's regions left
  out). The empty rate is the share returning no line. Reported both on the 54 regions whose screenshot is in
  synth3's default scene pool and on the 12 that are not (#97's split; here no training canvas drew from any of them).
- Report only: the 25 holdout empty regions; per platform and per variant.

### The selection rule (pre-registered)

- **Eligible**: a learned gate's C runtime (`runtime/vocr_textgate.c`) gives the twin's score on every development
  case and empty region.
- **Qualifies**: for each of VOCR5-S, -N and -L, gate − refine on wd has a VN-cluster 95% interval whose lower end is
  ≤ 0 (not resolved worse) **and** a point estimate of at most **+0.10 points** (the owner's "at almost no cost"; #97's
  gate cost 0.02-0.03); the native_vita interval's lower end is ≤ 0; and the development empty rate is at least
  **0.85** (57 of 66).
- **Size order**: total bytes = int8 blob bytes + the VitaSDK gcc 15.2 `-Os` `.text` bytes of the gate's code for its
  mode, linked with `--gc-sections` from the gate's entry points (`vita/textgate_size.py`); then blob bytes, then
  `region` before `boxes`, then the higher τ (the more conservative point), then the name.
- **Winner**: the first qualifying candidate in that order.
- **No winner**: the verdict is negative and GRAD + refine stay as they are; the protected sets are read once for
  the candidate with the most development empties returning no line among those meeting the wd conditions,
  labelled not promoted.

### The runtime

`runtime/vocr_textgate.c`: C99, no malloc, no recursion, integer only, scalar (no NEON, for size), a caller arena
that depends on the region's width. It loads VTDT blobs and computes the K map row by row from rows pushed to it in
order (`begin` / `row` / `end`), so a caller can feed it the rows the finder's first pass reads (Rubi's capture
streams the region in strips and never holds it whole) without reading the region again; plus the in-memory form, a
pull form over `vocr_lines_rows.h`, and a finder mode behind a flag: with no gate it returns `vocr_find_lines`'
result bit for bit, and `vocr_lines.c`, `vocr_lines_rows.c`, `vocr_lines_refine.c` and `vocr_textdet.c` are not
touched. `eval/textgate.py` (on `synth/textdet_twin.run_net`) is the bit-exact twin; a parity test compares them.

### Protected sets (once, after the selection record is pushed)

vita-test + psp-test whole dialogue (eval's regions at base / +4 / +8, #97's GRAD + refine units for VOCR5-S/N/L
with the gate's decisions on the same regions) and CLANNAD's long run (the gate on each page's ROI; GRAD + refine
read for every reference line as `eval/textdet_protected.py` does; dialogue CER with and without the gate; the lines
returned on pages the references hold no text for). Attempt records refuse a second run.

### Estimates

Console time from the MAC rate of VOCR3-H15 (1,564,992 MACs per CTC frame at 5.0-5.2 ms on one A9 core, 333 MHz
assumed), labelled as estimates; code and weight bytes and arenas are exact.

### Before the freeze (disclosed)

Training-side studies on the calibration cohorts, before this protocol: block-statistics designs (per 8 x 8 or
16 x 16 block: contrast, mean |dx|, mean |dy|, edge share, with tiny CNNs of 0.3-5 K parameters trained densely on
#97's real canvases) kept at most 0.47 of cal_empty at 0.998 recall of cal_dlg; the edge-run rule separated nothing
at that recall. #97's detectors read as presence gates kept 0.17-0.81 (1,004-1,884 B), 0.80-0.91 (2,228-3,276 B)
and 0.95 (`s8_c24_seed1`). So the block-statistics CNNs are not carried into the sweep (the rules are, as the
baseline). #97's `s8_c24_seed4` with its own decode on the calibration cohorts: 14 of 11,103 cases with no line,
0.904 of cal_empty with no line (on development: 3 of 1,515 and 0.92). The eight presence runs started at
2026-10-18T07:47Z with the pinned trainer, before this protocol was pushed; nothing of them was scored before it.


## Results

Every number here is a host measurement (the C finder and refinement and their twins, the gate's C runtime checked
against its twin, the VOCR5 recognizers' frozen units) or a host estimate of console time, labelled as such.
Nothing of the gate has run on a console.

### Answer first

- **The rule has a winner: `p8_c8_seed1` in `boxes` mode at the 0.998 calibration point (k = 4, t = -8): 1,540 B of
  int8 weights and 2,072 B of VitaSDK `-Os` code, 3,612 B in all** (`eval/textgate_dev_20261018.json`). #97's
  detector, which Rubi #209 declined, is 5,700 B of weights and 7,250 B of code by the same measurement (8,360 B
  at `-O2` in #209's).
- **Empty regions:** in front of GRAD + refine it leaves **59 of the 66** development empty regions with no line
  (0.894; GRAD + refine alone: 5 of 66): 10 of the 12 whose screenshot is outside the synthetic scene pool and 49 of
  the 54 inside it, none of which the gate's training data hold. The 25 holdout empty regions (report only): 23,
  6 of the 6 outside the pool.
- **Development whole dialogue:** +0.062 / +0.058 / +0.073 points for VOCR5-S / N / L (5.21 → 5.28%, 4.78 → 4.84%,
  4.46 → 4.53%; VN-cluster intervals 0 to +0.19 / +0.18 / +0.22): not resolved worse and inside the +0.10 cap. It
  stops 7 of the 1,515 cases: 3 that GRAD + refine read nothing from anyway, and 4 short single lines (8-20
  characters) in the +4 / +8 px variants of three regions. No base region and none of the 150 native-Vita units
  loses anything.
- **The protected sets, read once:** vita-test + psp-test whole dialogue **unchanged** (body text 9.51 / 9.16 /
  8.35%; the gate stops 3 of 525 cases, one vita-test region's three variants, which hold a 3-character speaker name
  only and from which GRAD + refine read nothing). **CLANNAD unchanged** (dialogue 1.43 / 0.92 / 0.63%: it stops
  none of the 2,044 pages the references score); on the 133 pages they hold no text for it stops 47, and GRAD +
  refine's 162 lines there become 114.
- **What it costs in time (estimate):** about 67-135 ms per 900×150 box on one Cortex-A9 core at 333 MHz (50-101 ms
  at 444), 18-36 ms per 450×80 PSP box and 256-517 ms for a whole 960×544 screen (a cycle model on the Vita
  compiler's code). The code is scalar for size; #97's NEON detector does its 10.6 M MACs in about 53 ms, the gate's
  2.7 M take longer.
- **Is it worth it?** It is the smallest thing measured here that stops junk on empty regions: a quarter of #97's
  weights and a quarter of its code, at no cost on the protected sets and 0.06-0.07 points on development, where the
  cost sits on short lines in loose regions. Its price is time, about 0.1 s per text box (more for a whole screen),
  in every job while it is on. So: an opt-in, off by default until a console A/B shows how often empty regions occur
  and what the time costs (the Rubi_for_vita PR does that).

### Calibration (training-side, before development)

The cal_empty share left with no line at each operating point (bold: through the screen, ≥ 0.85); every point keeps
at least its τ of the cal_dlg cases GRAD + refine finds a line in (`eval/textgate_calibration_20261018.json`).

| candidate | int8 B | region, τ 0.998 | boxes, τ 0.998 | region, τ 0.999 | boxes, τ 0.999 |
|---|---:|---:|---:|---:|---:|
| blank | 0 | 0.077 |  | 0.077 |  |
| rule8 | 0 | 0.083 |  | 0.083 |  |
| p4_c4_seed1 | 816 | 0.350 | 0.344 | 0.271 | 0.281 |
| p8_c4_seed1 | 816 | 0.420 | 0.442 | 0.325 | 0.330 |
| r97_s4_c4_seed1 | 1,004 | 0.262 | 0.232 | 0.122 | 0.129 |
| r97_s4_c4_seed2 | 1,004 | 0.598 | 0.609 | 0.445 | 0.398 |
| r97_s4p_c4_seed1 | 1,100 | 0.566 | 0.570 | 0.408 | 0.414 |
| p4_c8_seed1 | 1,196 | 0.722 | 0.725 | 0.699 | 0.668 |
| p8_c6_seed1 | 1,280 | 0.809 | 0.807 | 0.727 | 0.713 |
| p8_c6_seed2 | 1,280 | **0.876** | **0.876** | 0.837 | 0.793 |
| r97_s4_c6_seed1 | 1,280 | 0.618 | 0.603 | 0.532 | 0.356 |
| p8_c8_seed1 | 1,540 | **0.928** | **0.921** | **0.888** | **0.879** |
| p8_c8_seed2 | 1,540 | **0.852** | **0.851** | 0.757 | 0.746 |
| r97_s4_c8_seed1 | 1,540 | 0.680 | 0.654 | 0.495 | 0.470 |
| r97_s4_c8_seed2 | 1,540 | 0.607 | 0.607 | 0.474 | 0.465 |
| r97_s8_c8_seed1 | 1,884 | **0.867** | **0.851** | 0.810 | 0.676 |
| r97_s8_c8_seed2 | 1,884 | 0.725 | 0.706 | 0.638 | 0.617 |
| r97_s8_c8_seed3 | 1,884 | 0.767 | 0.726 | 0.621 | 0.563 |
| p8_c12_seed1 | 2,000 | **0.863** | **0.865** | 0.803 | 0.775 |
| r97_s4_c8n5_seed1 | 2,228 | 0.849 | 0.828 | 0.756 | 0.731 |
| r97_s4_c8n5_seed3 | 2,228 | 0.681 | 0.665 | 0.521 | 0.470 |
| r97_s4_c12_seed1 | 2,276 | 0.744 | 0.750 | 0.642 | 0.655 |
| r97_s4p_c8_seed1 | 2,700 | 0.826 | 0.822 | 0.765 | 0.731 |
| r97_s4_c12n4_seed3 | 2,808 | **0.904** | 0.843 | 0.768 | 0.681 |
| r97_s4_c16_seed1 | 3,108 | **0.862** | 0.826 | 0.783 | 0.702 |
| r97_s4_c16_seed2 | 3,108 | **0.928** | **0.904** | **0.865** | 0.814 |
| r97_s8_c16_seed1 | 3,276 | **0.913** | **0.906** | **0.860** | 0.755 |
| r97_s8_c16_seed2 | 3,276 | **0.876** | **0.864** | 0.790 | 0.674 |
| r97_s8_c16_seed3 | 3,276 | **0.853** | **0.856** | 0.752 | 0.712 |
| r97_s4_c16n4_seed1 | 3,860 | **0.947** | **0.931** | **0.882** | **0.876** |
| r97_s4_c16n4_seed3 | 3,860 | **0.959** | **0.939** | **0.912** | **0.904** |
| r97_s8_c16n4_seed3 | 4,028 | **0.891** | **0.862** | 0.810 | 0.762 |
| r97_s8_c24_seed4 | 5,700 | **0.972** | **0.957** | **0.928** | 0.812 |

The two baselines change nothing GRAD + refine would not (cal_empty: 0.077 and 0.083 against the finder's own
0.074): the networks beat them easily. Below about 1.3 KB no network gets through the screen; the presence-trained
`p8_c8_seed1` (1,540 B) matches #97's 3-4 KB detectors, and only #97's 5.7 KB `s8_c24_seed4` (report only) is clearly
better.

### Development (the rule's data)

`eval/textgate_dev_20261018.json`; the C runtime gives the twin's score on every case and empty region for every
candidate, in the push and in-memory forms (109,208 scores, driver at `ec4b980`). Total bytes = weights + the
mode's code (1,892 B region, 2,072 B boxes; `eval/textgate_sizes_20261018.json`).

| operating point | total B (weights + code) | wd cases stopped | gate − refine, S / N / L (VN CI) | native Vita, N | empty: no line (of 66) | out of the scene pool (of 12) | qualifies |
|---|---:|---:|---|---|---:|---:|---|
| p8_c6_seed2/region/0.998 | 3,172 (1,280 + 1,892) | 10 | +0.062 (+0.00 to +0.19) / +0.050 (+0.00 to +0.17) / +0.058 (+0.00 to +0.20) | +0.000 (+0.00 to +0.00) | 55 (0.833) | 10 | no |
| p8_c6_seed2/boxes/0.998 | 3,352 (1,280 + 2,072) | 10 | +0.062 (+0.00 to +0.19) / +0.050 (+0.00 to +0.17) / +0.058 (+0.00 to +0.20) | +0.000 (+0.00 to +0.00) | 56 (0.849) | 10 | no |
| p8_c8_seed1/region/0.999 | 3,432 (1,540 + 1,892) | 6 | +0.054 (+0.00 to +0.18) / +0.050 (+0.00 to +0.17) / +0.058 (+0.00 to +0.20) | +0.000 (+0.00 to +0.00) | 56 (0.849) | 10 | no |
| p8_c8_seed1/region/0.998 | 3,432 (1,540 + 1,892) | 8 | +0.089 (+0.00 to +0.28) / +0.081 (+0.00 to +0.26) / +0.104 (+0.00 to +0.32) | +0.000 (+0.00 to +0.00) | 59 (0.894) | 10 | no |
| p8_c8_seed2/region/0.998 | 3,432 (1,540 + 1,892) | 10 | +0.159 (+0.00 to +0.46) / +0.159 (+0.00 to +0.46) / +0.166 (+0.00 to +0.47) | +0.000 (+0.00 to +0.00) | 54 (0.818) | 10 | no |
| p8_c8_seed1/boxes/0.999 | 3,612 (1,540 + 2,072) | 5 | +0.050 (+0.00 to +0.17) / +0.050 (+0.00 to +0.17) / +0.058 (+0.00 to +0.20) | +0.000 (+0.00 to +0.00) | 52 (0.788) | 10 | no |
| p8_c8_seed1/boxes/0.998 | 3,612 (1,540 + 2,072) | 7 | +0.062 (+0.00 to +0.19) / +0.058 (+0.00 to +0.18) / +0.073 (+0.00 to +0.22) | +0.000 (+0.00 to +0.00) | 59 (0.894) | 10 | **yes** |
| p8_c8_seed2/boxes/0.998 | 3,612 (1,540 + 2,072) | 10 | +0.159 (+0.00 to +0.46) / +0.159 (+0.00 to +0.46) / +0.166 (+0.00 to +0.47) | +0.000 (+0.00 to +0.00) | 54 (0.818) | 10 | no |
| r97_s8_c8_seed1/region/0.998 | 3,776 (1,884 + 1,892) | 11 | +0.124 (+0.01 to +0.30) / +0.108 (+0.00 to +0.27) / +0.112 (+0.00 to +0.28) | +0.000 (+0.00 to +0.00) | 53 (0.803) | 11 | no |
| p8_c12_seed1/region/0.998 | 3,892 (2,000 + 1,892) | 12 | +0.217 (+0.00 to +0.64) / +0.213 (+0.00 to +0.64) / +0.221 (+0.00 to +0.65) | +0.000 (+0.00 to +0.00) | 51 (0.773) | 9 | no |
| r97_s8_c8_seed1/boxes/0.998 | 3,956 (1,884 + 2,072) | 10 | +0.120 (+0.01 to +0.30) / +0.108 (+0.00 to +0.27) / +0.112 (+0.00 to +0.28) | +0.000 (+0.00 to +0.00) | 51 (0.773) | 11 | no |
| p8_c12_seed1/boxes/0.998 | 4,072 (2,000 + 2,072) | 12 | +0.221 (+0.00 to +0.64) / +0.213 (+0.00 to +0.64) / +0.221 (+0.00 to +0.65) | +0.000 (+0.00 to +0.00) | 54 (0.818) | 9 | no |
| r97_s4_c12n4_seed3/region/0.998 | 4,700 (2,808 + 1,892) | 9 | +0.093 (+0.00 to +0.28) / +0.081 (+0.00 to +0.26) / +0.097 (+0.00 to +0.31) | +0.000 (+0.00 to +0.00) | 58 (0.879) | 11 | **yes** |
| r97_s4_c16_seed2/region/0.999 | 5,000 (3,108 + 1,892) | 8 | +0.085 (+0.00 to +0.28) / +0.073 (+0.00 to +0.25) / +0.089 (+0.00 to +0.30) | +0.000 (+0.00 to +0.00) | 55 (0.833) | 11 | no |
| r97_s4_c16_seed1/region/0.998 | 5,000 (3,108 + 1,892) | 7 | +0.058 (+0.00 to +0.18) / +0.050 (+0.00 to +0.17) / +0.058 (+0.00 to +0.20) | +0.000 (+0.00 to +0.00) | 56 (0.849) | 11 | no |
| r97_s4_c16_seed2/region/0.998 | 5,000 (3,108 + 1,892) | 10 | +0.097 (+0.00 to +0.29) / +0.081 (+0.00 to +0.26) / +0.097 (+0.00 to +0.31) | +0.000 (+0.00 to +0.00) | 62 (0.939) | 11 | **yes** |
| r97_s8_c16_seed1/region/0.999 | 5,168 (3,276 + 1,892) | 8 | +0.097 (+0.00 to +0.31) / +0.101 (+0.00 to +0.34) / +0.108 (+0.00 to +0.37) | +0.000 (+0.00 to +0.00) | 50 (0.758) | 10 | no |
| r97_s8_c16_seed1/region/0.998 | 5,168 (3,276 + 1,892) | 8 | +0.097 (+0.00 to +0.31) / +0.101 (+0.00 to +0.34) / +0.108 (+0.00 to +0.37) | +0.000 (+0.00 to +0.00) | 54 (0.818) | 10 | no |
| r97_s8_c16_seed2/region/0.998 | 5,168 (3,276 + 1,892) | 4 | +0.004 (+0.00 to +0.01) / +0.000 (+0.00 to +0.00) / +0.000 (+0.00 to +0.00) | +0.000 (+0.00 to +0.00) | 55 (0.833) | 10 | no |
| r97_s8_c16_seed3/region/0.998 | 5,168 (3,276 + 1,892) | 6 | +0.093 (+0.00 to +0.24) / +0.093 (+0.00 to +0.24) / +0.097 (+0.00 to +0.25) | +0.000 (+0.00 to +0.00) | 53 (0.803) | 11 | no |
| r97_s4_c16_seed2/boxes/0.998 | 5,180 (3,108 + 2,072) | 10 | +0.097 (+0.00 to +0.29) / +0.081 (+0.00 to +0.26) / +0.097 (+0.00 to +0.31) | +0.000 (+0.00 to +0.00) | 61 (0.924) | 11 | **yes** |
| r97_s8_c16_seed1/boxes/0.998 | 5,348 (3,276 + 2,072) | 8 | +0.097 (+0.00 to +0.31) / +0.101 (+0.00 to +0.34) / +0.108 (+0.00 to +0.37) | +0.000 (+0.00 to +0.00) | 54 (0.818) | 10 | no |
| r97_s8_c16_seed2/boxes/0.998 | 5,348 (3,276 + 2,072) | 4 | +0.004 (+0.00 to +0.01) / +0.000 (+0.00 to +0.00) / +0.000 (+0.00 to +0.00) | +0.000 (+0.00 to +0.00) | 56 (0.849) | 10 | no |
| r97_s8_c16_seed3/boxes/0.998 | 5,348 (3,276 + 2,072) | 6 | +0.093 (+0.00 to +0.24) / +0.093 (+0.00 to +0.24) / +0.097 (+0.00 to +0.25) | +0.000 (+0.00 to +0.00) | 53 (0.803) | 11 | no |
| r97_s4_c16n4_seed1/region/0.999 | 5,752 (3,860 + 1,892) | 6 | +0.019 (+0.00 to +0.06) / +0.015 (+0.00 to +0.05) / +0.015 (+0.00 to +0.05) | +0.000 (+0.00 to +0.00) | 56 (0.849) | 10 | no |
| r97_s4_c16n4_seed3/region/0.999 | 5,752 (3,860 + 1,892) | 8 | +0.043 (+0.00 to +0.11) / +0.054 (+0.00 to +0.14) / +0.054 (+0.00 to +0.14) | +0.000 (+0.00 to +0.00) | 60 (0.909) | 11 | **yes** |
| r97_s4_c16n4_seed1/region/0.998 | 5,752 (3,860 + 1,892) | 10 | +0.089 (+0.00 to +0.22) / +0.085 (+0.00 to +0.22) / +0.093 (+0.00 to +0.25) | +0.000 (+0.00 to +0.00) | 63 (0.955) | 10 | no |
| r97_s4_c16n4_seed3/region/0.998 | 5,752 (3,860 + 1,892) | 10 | +0.116 (+0.00 to +0.34) / +0.124 (+0.00 to +0.37) / +0.132 (+0.00 to +0.40) | +0.000 (+0.00 to +0.00) | 63 (0.955) | 11 | no |
| r97_s8_c16n4_seed3/region/0.998 | 5,920 (4,028 + 1,892) | 8 | +0.093 (+0.00 to +0.28) / +0.081 (+0.00 to +0.26) / +0.097 (+0.00 to +0.31) | +0.000 (+0.00 to +0.00) | 59 (0.894) | 10 | **yes** |
| r97_s4_c16n4_seed1/boxes/0.999 | 5,932 (3,860 + 2,072) | 6 | +0.019 (+0.00 to +0.06) / +0.015 (+0.00 to +0.05) / +0.015 (+0.00 to +0.05) | +0.000 (+0.00 to +0.00) | 54 (0.818) | 10 | no |
| r97_s4_c16n4_seed3/boxes/0.999 | 5,932 (3,860 + 2,072) | 8 | +0.043 (+0.00 to +0.11) / +0.054 (+0.00 to +0.14) / +0.054 (+0.00 to +0.14) | +0.000 (+0.00 to +0.00) | 59 (0.894) | 11 | **yes** |
| r97_s4_c16n4_seed1/boxes/0.998 | 5,932 (3,860 + 2,072) | 10 | +0.089 (+0.00 to +0.22) / +0.085 (+0.00 to +0.22) / +0.093 (+0.00 to +0.25) | +0.000 (+0.00 to +0.00) | 59 (0.894) | 10 | no |
| r97_s4_c16n4_seed3/boxes/0.998 | 5,932 (3,860 + 2,072) | 8 | +0.043 (+0.00 to +0.11) / +0.054 (+0.00 to +0.14) / +0.054 (+0.00 to +0.14) | +0.000 (+0.00 to +0.00) | 63 (0.955) | 11 | **yes** |
| r97_s8_c16n4_seed3/boxes/0.998 | 6,100 (4,028 + 2,072) | 8 | +0.085 (+0.00 to +0.28) / +0.073 (+0.00 to +0.25) / +0.089 (+0.00 to +0.30) | +0.000 (+0.00 to +0.00) | 57 (0.864) | 10 | **yes** |

Report only (`eval/textgate_dev_report_20261018.json`): #97's `s8_c24_seed4` read as a presence gate stops 16 cases at
its 0.998 point (+0.55 / +0.56 / +0.57 points, 65 of 66 empty) and 6 at its 0.999 point (+0.12 / +0.11 / +0.12, 63 of
66); as #97's own gate (its decode returns no line) it cost 0.02-0.03 points for 61 of 66 (`docs/textdet.md`). The
baselines `blank` and `rule8` leave 5 of 66, the finder's own.

The winner, per variant (VOCR5-N): base 4.37 → 4.37%, +4 px 4.82 → 4.90%, +8 px 5.15 → 5.24%; per platform: Vita
frames 3.49 → 3.60%, PSP frames 5.90 → 5.92%. The development empties it misses: five of vita_dev (four VNs) and
two of psp_dev (six VNs in all); two of the seven lie outside the scene pool.

### The protected sets, once (`eval/textgate_protected_20261018.json`, `eval/textgate_clannad_20261018.json`)

After the selection record was pushed (`69fc939`, 2026-10-18T09:19:56Z; tag `textgate-selection-69fc939`), the
winner was read once (attempt records 2026-10-18T09:20:50Z), the C runtime checked against the twin on every
protected region.

| set | cases / pages | stopped by the gate | VOCR5-S | VOCR5-N | VOCR5-L |
|---|---:|---:|---|---|---|
| vita-test + psp-test whole dialogue, body | 525 cases | 3 (0 characters read before) | 9.51 → 9.51% | 9.16 → 9.16% | 8.35 → 8.35% |
| the same, all text (names included) | | | 11.05 → 11.05% | 10.50 → 10.50% | 9.71 → 9.71% |
| CLANNAD long run, dialogue (2,052 lines) | 1,867 pages | 0 | 1.43 → 1.43% | 0.92 → 0.92% | 0.63 → 0.63% |
| CLANNAD, all lines (3,048) | 2,044 pages | 0 | 2.45 → 2.45% | 1.46 → 1.46% | 0.90 → 0.90% |
| CLANNAD pages the references hold no text for | 133 pages | 47 | lines returned: 162 → 114 | | |

Whether those 133 pages show text was not checked (#97 noted the same); the references exclude lines cut by the ROI,
UI and unreadable ones. Every VOCR5 model trained on the owner's planetarian PSP frames (Key / Prototype, like
CLANNAD's Vita port); the gate trained on none of the owner's planetarian frames (v34 falls in its calibration
split) and none of CLANNAD's.

### The runtime (`runtime/vocr_textgate.c`, `runtime/README.md`)

C99, no malloc, no recursion, no global state, integer only, scalar, self-contained (its own VTDT loader: Rubi pins
only `vocr_textgate.c` / `.h`). The push form (`vocr_textgate_begin` / `_push` / `_score`) takes the region's rows in
order, so a caller can feed it the rows its finder's first pass reads and read nothing more; the in-memory and
row-callback forms and the finder mode `vocr_find_lines_gated` (no gate: `vocr_find_lines`, bit for bit) are built on
it; `vocr_lines.c`, `vocr_lines_rows.c`, `vocr_lines_refine.c` and `vocr_textdet.c` are unchanged. Arena at 960×544:
27,489 B in region mode, 35,529 B in boxes mode (the K map), inside the finder's own 83,323 B; stack 264 B (push) on
the Vita. Code, VitaSDK gcc 15.2 with Rubi's runtime flags, `--gc-sections`: 1,892 B (region) / 2,072 B (boxes) at
`-Os`, 2,380 / 2,636 B at `-O2`; psp-gcc `-Os`: 3,532 / 3,864 B (`eval/textgate_sizes_20261018.json`). Parity:
`tests/textgate_parity.py` (random blobs of every gate and #97 shape, procedural regions, every form) in
`make check-textgate`, under ASan + UBSan with gcc and clang in CI.

### Console time (host estimates; `eval/textgate_a9_20261018.json`)

| `p8_c8`, boxes mode | 900×150 box | 450×80 PSP box | 960×544 screen | 480×272 screen |
|---|---:|---:|---:|---:|
| MACs | 2.74 M | 0.73 M | 10.5 M | 2.63 M |
| A9 cycle model @333 / @444 MHz (low to calibrated high) | 67-135 / 50-101 ms | 18-36 / 13-27 ms | 256-517 / 192-388 ms | 64-129 / 48-97 ms |
| MAC-rate estimate @333 MHz (VOCR3-H15's console rate, NEON code) | 8.8-9.1 ms | 2.3-2.4 ms | 33.6-34.9 ms | 8.4-8.7 ms |

The cycle model prices the Vita compiler's `-O2` code (the scalar gate) under plugin qemu-arm with `vita/a9_cycles.py`,
calibrated to VOCR3-H15's 5.0-5.2 ms per CTC frame on the console; the MAC-rate figures assume the recognizer's NEON
efficiency, which scalar code does not reach, and are shown for comparison with #97's tables. In Rubi the gate's work
is spread over the finder's first pass (its rows as they arrive), and on an empty region it saves the recognizer's
work and the model's load.

### What changed after the freeze (disclosed)

- `eval/textgate.py` is unchanged since the protocol. Three modules came after it and implement what the protocol
  describes: `eval/textgate_c.py` (the C runtime against the twin's recorded scores, for eligibility),
  `eval/textgate_protected.py` (the once-only protected scorer) and `eval/textgate_tables.py` (records and tables).
- The report-only `s8_c24_seed4` was calibrated with the candidates but left out of the rule's input
  (`calibration.json` without it), as the protocol marks it report only; it was scored apart.
- The C runtime changed during its size work after the development decisions (`93a5812` → `f57d9bc` → `ec4b980`,
  the faster of two small variants, +90 B); the parity was re-run on the final code before the selection. The order
  does not depend on which version measured the code: every learned candidate carries the same code, and `boxes`
  would lose its place only if it cost 1,268 B more than `region`.
- The presence runs started (07:48Z) before the protocol was pushed (07:52Z), with the pinned trainer, as the
  protocol says; nothing of them was scored before it.

### Limits

- 66 development empty regions (12 outside the scene pool) and a 93-VN development sample: the empty rate's interval
  is wide (59 of 66: about 0.80-0.95, a 95% Wilson interval); the 0.85 condition was met with two regions to spare.
- The empty regions are upper-screen scene crops of VNDB screenshots; menus, hidden text boxes and full CG screens
  as Rubi captures them were not scored as such.
- The cost sits on short lines in loose regions; a region holding only a short line of a few characters is where
  the gate is most likely to stop text. In boxes mode Rubi's finder (the textbox recovery, a superset of GRAD's
  boxes) can only raise the score, never stop more.
- Vertical text: the gate was not trained to keep it (no loss on vertical lines) and no vertical text is scored.
- Speed is an estimate, and scalar: about 0.1 s per text box at 333 MHz.
- Nothing ran on a console.

### Reproduce

```sh
H=$HERD; A=$H/shared/artifacts/textgate; P=$H/shared/private/textgate
.venv/bin/python eval/textgate_data.py cal --out $P/cohorts                      # calibration cohorts (private)
.venv/bin/python eval/textgate_data.py synth --out $A/shards/synth_clean --shards 24   # clean synthetic shards (pinned)
eval/reproducibility/textgate-20261018/sweep.sh $A/sweep                         # the 8 presence runs
.venv/bin/python eval/textgate.py calibrate --cand NAME=BLOB ... --cohorts $P/cohorts --out $A/cal/calibration.json
.venv/bin/python eval/textgate.py dev --calibration $A/cal/calibration_selection.json --out $A/dev/merged
.venv/bin/python eval/textgate_c.py parity --dev $A/dev/merged --cbin build/vocr_textgate_test --calibration ... --out PARITY
.venv/bin/python eval/textgate.py select --dev $A/dev/merged --sizes eval/textgate_sizes_20261018.json --parity PARITY --out SEL
.venv/bin/python eval/textgate_tables.py record ... ; .venv/bin/python eval/textgate_tables.py tables --cal ... --dev ...
.venv/bin/python eval/textgate_protected.py tests|clannad --gate p8_c8_seed1=BLOB --mode boxes --k 4 --t -8 ...   # once
.venv-ci/bin/python vita/textgate_size.py --out eval/textgate_sizes_20261018.json
.venv-ci/bin/python eval/textgate_a9.py measure ... ; eval/textgate_a9.py report ... --json eval/textgate_a9_20261018.json
```
