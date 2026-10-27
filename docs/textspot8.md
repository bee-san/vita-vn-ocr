# textspot8: MeikiOCR's line detector as the reference, and a spotter distilled from it for Rubi (issue #149)

The owner (2026-10-27): "line dialogue spotter is exactly what i want, meiki text detector is really good at this. i
want to get close to that performance to prevent these dialogue box issues in the future". textspot6 (#128,
`docs/textspot6.md`) trained tagged spotters on the hooked pages. Through Rubi's page path they read the unseen dev
games at 1.39-2.07% dialogue CER, against GRAD + refine's 3.08% and Rubi's finder `rubi,fix=1`'s 2.38%, but its rule
selected none: no unseen-dev interval lies below 0. textspot8 is the MeikiOCR-teacher line of the follow-up
(vocr6-spotter2; vocr6-spotter3's textspot7 does a spotter + GRAD fallback, decode fixes and a fresh confirmation set,
`docs/textspot7.md`):

1. MeikiOCR's text detector, the version `teacher/` uses, measured as the reference: line boxes and page-path dialogue
   CER on hval and the unseen-game dev units, per VN, its size and host speed.
2. A crop-geometry diagnostic on hval.
3. A ≤ 500 KB int8 spotter for the plain-C runtime, trained on MeikiOCR's boxes on VNDB and owner screenshots and the
   hooked pages' exact boxes, with the changes the analysis below calls for.
4. The decision by textspot6's rule, with two guards.

This document starts as the pre-registered protocol (`eval/textspot8_protocol.json`). Results are added below
"Results" as they are measured. Nothing above "Results" changes except through a dated amendment.

## Why: where textspot6's spotters lose

Measured on textspot6's stored per-unit results (`$A/vocr6-spotter/eval/`). Unseen dev means #119's units without
v12188 (92 VNs, 1,143 units); every comparison is paired by unit.

- **One VN carries the average gain.** v27449 (a translucent box over a busy scene) holds 311 of GRAD + refine's 785
  errors; every spotter reads it with 20-33.
- **Without v27449, Rubi's finder is ahead.** `rubi,fix=1` makes 294 errors on the other 91 VNs; `base_t64n8c16`
  seed 1 (det) makes 334, `base_qS4` seed 2 (crop) 333, textspot3's pick (det) 310.
- **The losses, paired against `rubi,fix=1`** (base_t64n8c16 seed 1, det; without v27449):

  | the spotter's unit | `rubi,fix=1`'s unit | units | errors: spotter / fix=1 |
  |---|---|---:|---|
  | a line cut | every line whole | 28 | 80 / 12 |
  | a line missed | every line whole | 6 | 48 / 6 |
  | a line cut or merged | a junk box | 4 | 30 / 0 |
  | a line merged | every line whole | 1 | 7 / 0 |
  | every line whole | every line whole | 993 | 97 / 99 |
  | a junk box | a junk box | 4 | 7 / 10 |
  | every line whole | a junk box | 55 | 5 / 20 |
  | every line whole | a line cut | 7 | 0 / 23 |
  | every line whole | a line missed | 16 | 60 / 124 |
  | all 1,119 units of the 91 VNs (rows with equal errors not listed) | | 1,119 | 334 / 294 |

  The crop decode (base_qS4 seed 2) loses on fragmented lines instead: +99 errors in 114 units.
- **The cuts are line ends.** On hval, 231 of base_t64n8c16 seed 1's 306 cut lines (det) lack one end, mostly one
  character: a leading 「 (81) or ― (30), or a trailing ― (33), … (24), 。 (20), 」 (16), 、 (12) or ！ (12). These
  glyphs have little ink, and the kernel stops before them.
- **Crop geometry is not a loss.** The spotters' boxes are 2-3 px taller per side than GRAD's, on dev and on hval.
  But on the 993 units both read whole the errors are level (97 / 99), and step 2 shrinks the boxes on hval with no
  resolved change.

## Protocol

### Step 1: MeikiOCR's detector as the reference

- **The detector:** `teacher/engines.Meiki`'s pinned `meiki.text.detect.v0.1` (rtr46/meiki.text.detect.v0 at
  `a9cffa4f`, sha256 `40b6a016...`), MeikiOCR 0.3.5's own preprocessing (fit to 960 × 544, INTER_LINEAR), BGR input as
  `teacher/label.py` and textspot6's cache feed it, threshold 0.5 (MeikiOCR's default).
- **Frames** (`eval/textspot8.py meiki-spot`, in colour): dev (#119's units' frames in the console view: a PSP frame
  upscaled 2x bilinear), hval (the hval pages in the views vita, psp2x and psp), tsdev (textspot's 188 dev frames).
- **Decodes:** `raw` (Meiki's boxes), `pad` (grown by 4 px; 2 px on a native PSP view), `ink` (refined to ink as
  textspot2 refines meiki boxes, then the pad). Each is read as det and det+refine through eval/textspot6.py's
  `pagepath-boxes`, in textspot_twin's reading order.
- **The reference row of record** is the decode with the lowest hval page-path CER (hval only, never dev); every
  decode is reported.
- **Metrics:** page-path dialogue CER (dev headline, Vita / PSP, hval), line recall / box precision / F1 on the dev and
  hval units, hval box P / R / F1 against the verified lines, tsdev, per-VN errors, the detector's size and host ms per
  frame.

### Step 2: the crop-geometry diagnostic (hval only)

`rubi,fix=1`'s boxes grown by 1 and 2 px vertically and by 4 px horizontally; textspot6's base_t64n8c16 seed 1 (det)
and base_qS4 seed 2 (crop) boxes shrunk by 1 and 2 px vertically. Report only; it chooses the student's across shrink
(amendment 1).

### Step 3: the student (amendment 1)

One arm, `t8_t64n8c16`, seeds 1-3: textspot6's family-T trunk with the tagged head (VTSP version 4, 52,176 B, est.
264.2 ms @333 / 198.2 ms @444 per 960 × 544 frame, arena 244,607 B; the shape Rubi's integration branch runs),
`train/textspot8_train.py`.

- **Recipe:** textspot6's `base` exactly (30,000 steps, AdamW 4e-3, cosine, EMA, QAT from 80%).
- **Data:** textspot6's base mix exactly:
  - hooked pages 0.5 (`data/manifests/textspot6_pages_v1.json`, split train: exact renderer line boxes refined to ink,
    MeikiOCR's boxes on the text the hook does not cover as "not dialogue");
  - synthetic frames 0.25 (`data/manifests/textspot_shards_v1.json`);
  - MeikiOCR's detector boxes on 52,785 VNDB and owner screenshots of games outside the dev sample 0.25
    (`data/manifests/textspot2_shards_v1.json`): the distillation set.
- **The changes (T8):**
  - **edge kernel:** the K weight of the kernel cells at either end of a line ×4 (only ends inside the window);
  - **across shrink:** the across-line extent targets moved inward by the amount step 2's rule picks;
  - **icons:** with p 0.35 per window, 1-2 synthetic animated-icon marks (spinner dot rings and arcs, page cursors,
    a four-point star, a diamond, a ring, an hourglass) drawn as background after a line's end or on free background;
  - **tight regions:** with p 0.15 the window is cut to a block of lines plus 0-0.4 of the line height;
  - **zoom and blends:** zoom p 0.45 over 0.6-1.6 (textspot6: p 0.3, 0.8-1.25), translucent-box blends p 0.2 at
    alpha 0.45-0.9 (textspot6: p 0.1, 0.6-0.9).
- **Decodes:** textspot6's four (crop, crop+refine, det, det+refine), scored by
  `eval/reproducibility/textspot8-20261027/score_run8.sh` (textspot6's score_run.sh with textspot8's directories and
  textspot6's pins).
- **Control (report only):** textspot6's base_t64n8c16 seeds 1-3, the same shape, data and schedule without T8.

### The rule (textspot6's, with two guards)

- **Eligible:** VTSP ≤ 512,000 B, est. ≤ 300 ms @333 per 960 × 544 frame, arena ≤ 262,144 B, C == twin on every
  development frame.
- **Representative:** the median seed of 3 by unseen-dev page-path dialogue CER.
- **Beats GRAD + refine:** unseen dev (without v12188) with a VN-cluster 95% interval entirely below 0; hval not
  resolved above 0.
- **Guard:** not resolved worse than `rubi,fix=1` on vocr6-spotter3's fresh confirmation set (≥ 40 VNs no spotter
  trained on), read once per candidate set at the end.
- **Pick:** the cheapest qualifier (bytes, est. ms, arena, decode); a costlier one replaces it only when resolved
  better on unseen dev and not resolved worse on hval.
- **Rubi's default:** only if the pick also beats `rubi,fix=1` by the same test (and the guard holds); otherwise it
  ships behind a setting, off by default. Only one spotter is Rubi's default; if textspot7's also qualifies, the better
  one under this test, agreed with vocr6-spotter3 and vocr6-lead first.
- **No qualifier:** a negative verdict and a research pre-release of the eligible representative with the lowest
  unseen-dev CER.

The rule runs as code: `eval/textspot8_table.py` writes eval/textspot6_select.py's table, and textspot6_select.py
applies the rule unchanged.

### Disclosures registered in advance

- The student's design is informed by textspot6's per-VN analysis of the same unseen-dev units; the confirmation set
  guards against the optimism this brings.
- Step 2 reads hval only.
- Console times are textspot6's A9 cycle-model estimates; nothing runs on a console.

## Results

Every number is measured on the host unless it is labelled an estimate.

### Step 2: crop geometry (hval)

| boxes | hval page-path CER | against the stored boxes |
|---|---:|---|
| `rubi,fix=1` as stored | 3.292 | |
| `rubi,fix=1`, +1 px per side vertically | 1.900 | −1.392 [−4.460, −0.002] |
| `rubi,fix=1`, +2 px vertically | 2.729 | −0.563 [−1.977, +0.105] |
| `rubi,fix=1`, +4 px horizontally | 3.285 | −0.007 [−0.082, +0.062] |
| textspot6 base_t64n8c16 s1 det as stored | 1.302 | |
| the same, −1 / −2 px per side vertically | 1.292 / 1.291 | −0.011 [−0.061, +0.036] / −0.012 [−0.058, +0.034] |
| textspot6 base_qS4 s2 crop as stored | 1.614 | |
| the same, −1 / −2 px vertically | 1.607 / 1.976 | −0.007 [−0.045, +0.032] / +0.361 [−0.029, +1.172] |

- The spotters' margin does not matter; amendment 1's rule gives the across shrink 1.0 network px (G = 1.3024 /
  1.2919 / 1.2907 for d = 0 / 1 / 2).
- `rubi,fix=1`'s +1 px gain is one title, PCSG00854 (2,581 errors in 3,861 characters as stored, 383 at +1 px, 1,576 at
  +2 px): a knife-edge font, every other title within ±30 errors. Report only.

### Step 1: MeikiOCR's detector, the reference (measured 07:4xZ)

Page-path dialogue CER in %, through textspot6's pinned page path (VOCR5-N W8I, the counter re-decode, beam 8 + the 1 MB
LM, the cascade to L, Rubi's textnorm). Dev is the headline set: unseen games, without v12188. Line R / box P are
eval/pagepath.py's (whole lines / lines, boxes that are not junk / boxes). Intervals are VN clusters on dev and title
clusters on hval.

| finder | dev | dev Vita / PSP | hval | hval Vita / PSP | dev line R / box P / F1 | hval line R / box P / F1 | dev − GRAD + refine | dev − `rubi,fix=1` | hval − `rubi,fix=1` |
|---|---:|---|---:|---|---|---|---|---|---|
| GRAD + refine (`rubi`) | 3.08 | 2.95 / 3.21 | 3.88 | 4.53 / 2.70 | 0.970 / 0.950 / 0.960 | 0.960 / 0.959 / 0.960 | | +0.71 [+0.25, +1.29] | +0.58 [+0.31, +0.93] |
| `rubi,fix=1` (Rubi's finder) | 2.38 | 1.97 / 2.74 | 3.29 | 3.90 / 2.20 | 0.986 / 0.950 / 0.968 | 0.976 / 0.959 / 0.967 | −0.71 [−1.29, −0.25] | | |
| textspot6 `base_qS4` s2 crop (its research release) | 1.39 | 0.96 / 1.77 | 1.61 | 1.84 / 1.21 | 0.930 / 0.992 / 0.960 | 0.941 / 0.985 / 0.963 | −1.70 [−4.33, +0.08] | −0.99 [−3.65, +0.65] | −1.68 [−5.21, +0.31] |
| textspot6 `base_t64n8c16` s1 det | 1.41 | 1.47 / 1.35 | 1.30 | 1.37 / 1.18 | 0.979 / 0.995 / 0.987 | 0.970 / 0.990 / 0.980 | −1.68 [−4.31, +0.12] | −0.97 [−3.59, +0.71] | −1.99 [−5.64, −0.05] |
| MeikiOCR, raw boxes, det | 2.25 | 1.94 / 2.52 | 3.51 | 4.13 / 2.40 | 0.998 / 0.997 / 0.998 | 0.920 / 0.957 / 0.938 | −0.84 [−3.54, +0.89] | −0.13 [−2.83, +1.54] | +0.22 [−3.65, +2.59] |
| MeikiOCR, raw, det+refine | 2.46 | 2.12 / 2.77 | 3.60 | 4.25 / 2.43 | 0.993 / 0.995 / 0.994 | 0.917 / 0.956 / 0.936 | −0.62 [−3.35, +1.11] | +0.08 [−2.62, +1.73] | +0.30 [−3.51, +2.68] |
| **MeikiOCR, +4 px, det** | **0.72** | 0.50 / 0.92 | 2.22 | 3.00 / 0.83 | 0.998 / 0.997 / 0.998 | 0.972 / 0.957 / 0.965 | −2.36 [−4.96, −0.67] | −1.65 [−4.24, −0.09] | −1.07 [−5.01, +1.75] |
| **MeikiOCR, +4 px, det+refine (row of record)** | **0.91** | 0.62 / 1.17 | **1.65** | 2.08 / 0.88 | 0.997 / 0.997 / 0.997 | 0.969 / 0.957 / 0.963 | −2.17 [−4.84, −0.47] | −1.47 [−4.15, +0.10] | −1.65 [−5.41, +0.60] |
| MeikiOCR, ink + 4 px, det | 2.48 | 0.80 / 3.97 | 3.40 | 3.40 / 3.40 | 0.936 / 0.997 / 0.966 | 0.875 / 0.957 / 0.914 | −0.61 [−3.32, +1.32] | +0.10 [−2.61, +1.91] | +0.11 [−4.03, +3.13] |
| MeikiOCR, ink + 4 px, det+refine | 2.67 | 0.85 / 4.30 | 2.81 | 2.41 / 3.54 | 0.933 / 0.995 / 0.963 | 0.872 / 0.956 / 0.912 | −0.41 [−3.11, +1.44] | +0.29 [−2.44, +2.03] | −0.48 [−4.43, +2.06] |

- **The row of record** by the registered rule (the lowest hval CER) is +4 px det+refine: unseen dev **0.91%**, hval
  1.65%. Its interval against GRAD + refine lies below 0 (−2.17 [−4.84, −0.47]); against `rubi,fix=1` it reaches
  +0.10. The plain +4 px decode reads unseen dev at **0.72%**, and its interval is below 0 against both (`rubi,fix=1`:
  −1.65 [−4.24, −0.09]). So MeikiOCR's line boxes, given a margin, pass the rule this study applies to the student.
- **Meiki finds the lines; the margin makes the reading.** On dev it finds nearly every line whole and boxes almost no
  junk (line recall 0.998, box precision 0.997; textspot6 0.93-0.98 / 0.99, GRAD 0.97 / 0.95). Its boxes are tight to
  the ink, though, and without a margin the recognizer misreads their edges: raw boxes read dev at 2.25%, the same
  boxes grown by 4 px at 0.72%. Refining the box to ink first and then padding (ink) is worse on PSP (3.97% against
  0.92%); why was not checked.
- **Per VN** (+4 px det against `rubi,fix=1`): 26 VNs better, 9 worse, 57 level. Without v27449 the difference is
  −0.54 [−1.32, +0.04]. Its largest gains are v27449 (311 → 24 errors), v17972 (68 → 0) and v10972 (38 → 12). Its
  largest losses are v1971 (28 → 54: the side-by-side choice buttons every spotter loses), v6985 (6 → 21), v18163
  (8 → 15) and v25796 (0 → 8). Against textspot6's research release it is better on 44 VNs and worse on 6.
- **On hval** Meiki's errors are concentrated in 3 Vita titles: PCSG00917, PCSG00918 and PCSG00509 make 784 / 765 / 252
  errors with +4 px det against textspot6 t64's 24 / 6 / 15. The refinement brings the first two down to 279 / 241.
  Every other title is within a few dozen errors of textspot6.
  - In PCSG00917 and PCSG00918 every line is found whole, but 228 and 182 of Meiki's boxes are junk. In their base
    units 41-49% of the boxes are about 0.4 of the line's height, which fits ruby (furigana) boxed as lines of their
    own. The references and the hooked training targets leave ruby out.
  - PCSG00509 misses 12 lines.
- **Whole frames** (textspot's 188 development frames, +4 px det): box F1 0.895 (precision 0.948, recall 0.847),
  all-text CER 15.1%, dialogue CER 1.45%, UI recall 0.77. textspot6's runs: F1 0.57, all-text 40-42%, dialogue
  1.9-2.0%, UI recall 0.22.
- **hval boxes** against the verified lines (F1, +4 px): Vita 0.940, PSP 2x 0.979, PSP native 0.982 (textspot6 t64 s1:
  0.901 / 0.966 / 0.952).
- **Size and speed:** a DETR-style network (102 convolutions, 66 matrix products) of 3,547,383 float32 parameters,
  14,503,825 B of ONNX, reading a 960 × 544 colour frame. On this host it takes 137 ms per frame on 1 thread and 95 ms
  on 5 (median of 100 dev frames on the 64-core EPYC under the shared box's load). That is about 70 times the parameters
  of the student's 52 KB of int8 weights, and about 3.5 MB even in int8. It is the quality bar, not a console model.

<!-- STEP1 -->

<!-- STUDENT -->
