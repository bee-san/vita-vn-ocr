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
