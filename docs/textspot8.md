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

### Amendments

Each is dated in `eval/textspot8_protocol.json` and precedes the reads it names (the experiment log
/local/home/skerraut/herd-synth3/vocr6/experiments.jsonl has a pre-registration line for each).

1. **The student** (06:51Z, before every training run and score): one arm, `t8_t64n8c16`, seeds 1-3, as above.
2. **The across shrink is 0** (07:48Z, before any student score). Step 1's raw-vs-padded MeikiOCR boxes showed on
   hval that boxes without a margin cost the recognizer, and the shrink amendment 1's rule had picked from step 2
   (1.0 network px) would have put the margins below what step 2 tested. The first seed-1 run (43 min, 6,300 steps,
   with the shrink) was stopped, never scored, and is not a candidate; the arm's runs restarted in `runs2/`.
3. **vocr6-spotter3's D, F and F+D as extra decodes** (07:55Z, before any student score): textspot7's post-pass
   (`join10` then `extend`, as its hval fit keeps them, `feat/textspot7` a2fa039 / 3781f2f) on the student's det boxes,
   and its spotter + GRAD fallback F and F+D fit on hval per seed, with textspot7's cost rules (F adds GRAD's est.
   122.4 ms on the largest dev region, so F and F+D are over 300 ms for a T trunk).
4. **The context decode `ctx`** (09:25Z, before any student score): the spotter on the region grown by 48 px (chosen
   without data), the boxes whose centre lies in the region; the crop decode's cost row.
5. **Rubi's default** (10:17Z, before any student score; disclosed: after vocr6-spotter3 recorded textspot7's dev and
   confirmation reads of other models): the pick becomes Rubi's default iff, against `rubi,fix=1`, hval is resolved
   below 0, unseen dev is not resolved above 0 and the confirmation set is not resolved above 0 (the task's wording
   and vocr6-common's adoption rule against the incumbent). vocr6-lead accepted it for this study and kept
   textspot7's own clause for textspot7.
6. **The VOCR6 guard** (12:50Z, after the dev / hval and confirmation records, before any VOCR6 read): the pick and
   `rubi,fix=1` read with VOCR6-N → L (v0.4.0-vocr6.1) and Rubi's VOCR6 LM settings; neither difference may be
   resolved above 0 (vocr6-lead's request: that is what Rubi will run).

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


### Step 3: the student, and the decision (measured 11:30-13:00Z)

The records are:
- `eval/textspot8_dev_20261027.json`: the dev / hval phase, written by `eval/textspot8_select.py`, which is textspot6's
  rule code. It was committed before the confirmation read;
- `eval/textspot8_confirm_20261027.json`: the confirmation read, made once.

Page-path dialogue CER is in %. Each row is a decode's representative, the median seed; the seeds' unseen-dev CER
(1 / 2 / 3) is in brackets.

| decode | representative (seeds) | unseen dev | dev Vita / PSP | hval | dev − GRAD + refine | dev − `rubi,fix=1` | hval − `rubi,fix=1` | est. ms @333 | qualifies |
|---|---|---:|---|---:|---|---|---|---:|---|
| **crop** (the region alone) | **s3** (0.94 / 1.09 / 0.99) | **0.99** | 0.71 / 1.23 | **1.31** | −2.10 [−4.66, −0.41] | −1.39 [−3.96, +0.16] | −1.98 [−5.38, −0.11] | 264.2 | **yes** |
| crop+refine | s1 (1.21 / 1.33 / 1.15) | 1.21 | 0.92 / 1.47 | 1.42 | −1.88 [−4.46, −0.24] | −1.17 [−3.76, +0.34] | −1.87 [−5.21, −0.01] | 265.6 | yes |
| ctx (the region + 48 px) | s1 (1.04 / 0.97 / 1.04) | 1.04 | 0.82 / 1.23 | 1.33 | −2.05 [−4.64, −0.31] | −1.34 [−3.94, +0.25] | −1.97 [−5.49, −0.05] | 264.2 | yes |
| det (the whole frame) | s1 (1.34 / 1.61 / 1.24) | 1.34 | 1.26 / 1.41 | 1.28 | −1.75 [−4.24, −0.06] | −1.04 [−3.51, +0.51] | −2.01 [−5.68, −0.02] | 264.2 | yes |
| det+refine | s1 (1.34 / 1.83 / 1.31) | 1.34 | 1.18 / 1.47 | 1.40 | −1.75 [−4.29, −0.06] | −1.04 [−3.59, +0.50] | −1.89 [−5.54, +0.09] | 271.2 | yes |
| det+D (textspot7's post-pass) | s2 (1.21 / 1.15 / 0.93) | 1.15 | 1.12 / 1.18 | 1.14 | −1.93 [−4.49, −0.19] | −1.23 [−3.79, +0.40] | −2.15 [−5.44, −0.33] | 297.2 | yes |
| F (spotter + GRAD fallback) | s2 (1.13 / 1.14 / 1.17) | 1.14 | 0.85 / 1.39 | 1.21 | −1.95 [−4.61, −0.12] | −1.24 [−3.90, +0.47] | −2.08 [−5.82, −0.05] | 386.6 | no: over 300 ms |
| F+D | s2 (1.02 / 0.87 / 0.82) | 0.87 | 0.72 / 1.00 | 1.14 | −2.22 [−4.76, −0.55] | −1.51 [−4.03, +0.04] | −2.15 [−5.44, −0.33] | 419.6 | no: over 300 ms |

- **Eligibility.**
  - Every eligible row is the same 52,176 B blob shape. Its arena is 244,607 B (textspot6's row; the refine decodes
    take the larger of it and the refinement's 72,391 B).
  - The C runtime equals the twin on every development frame and crop, and on every grown ctx region of dev and hval.
  - textspot7's C post-pass equals its twin on every dev and hval unit.
  - D's largest-region cost is est. 33.0-33.8 ms (textspot7's A9 method), so det+D is est. 297.2 ms, under 300.
- **The pick** is the cheapest qualifier: the crop decode, seed 3. No costlier qualifier is resolved better than it on
  unseen dev, so none replaces it.

**The confirmation set** is vocr6-spotter3's: 41 VNs that no spotter trained on, 516 units, 4,737 characters. It was
read once, for every qualifier, after the dev / hval record. The baselines are vocr6-spotter3's stored reads with the
same pins.

| finder | confirmation CER | − `rubi,fix=1` | VNs better / worse / level |
|---|---:|---|---|
| GRAD + refine | 5.01 | | |
| `rubi,fix=1` | 4.29 | | |
| **crop, s3 (the pick)** | **1.86** (PSP 1.99 / Vita 0.64) | **−2.43 [−5.69, −0.30]** | 17 / 12 / 12 |
| crop+refine, s1 | 1.90 | −2.40 [−5.40, −0.34] | 20 / 9 / 12 |
| ctx, s1 | 2.28 | −2.01 [−5.36, +0.21] | 13 / 15 / 13 |
| det, s1 | 3.10 | −1.19 [−4.01, +1.10] | 18 / 11 / 12 |
| det+refine, s1 | 3.33 | −0.97 [−3.82, +1.36] | 13 / 13 / 15 |
| det+D, s2 | 2.18 | −2.11 [−5.20, +0.13] | 18 / 6 / 17 |

**The decision**:
- **Every qualifier passes the guard** (not resolved worse than `rubi,fix=1`), so the pick stays the crop decode.
- **Amendment 5's Rubi-default test passes**, against `rubi,fix=1`:
  - hval −1.98 [−5.38, −0.11], resolved below 0;
  - unseen dev −1.39 [−3.96, +0.16], not resolved above 0;
  - confirmation −2.43 [−5.69, −0.30], resolved below 0.
- **So the spotter becomes Rubi's default line finder**, with `rubi,fix=1` as the fallback and a setting to turn it
  off (vocr6-lead's ruling: one spotter in Rubi; textspot7's pick is released in vita-vn-ocr but not pinned).
- **textspot6's own clause** (unseen dev resolved below `rubi,fix=1`) would not have made it the default.

**With the recognizer Rubi will run** (amendment 6, vocr6-lead's guard).
- **The setup:** the pick's boxes and `rubi,fix=1`'s are read by VOCR6-N with the cascade to VOCR6-L. The recognizers
  are v0.4.0-vocr6.1's plain `_w8` blobs: the pinned page path has no switch for the two-level head that the shipped
  `_w8_2l` blobs add, so the guard has not read it on the spotter's boxes. vocr6-final-gate measured it on its own
  VOCR6-path kit and line plans with Rubi's decode: dev +0.003 [0.000, +0.009] (N with the L cascade) and +0.000 (L),
  hval page +0.005 for both (its `result.json`; corrected 15:25Z, the protocol's `corrections`). Rubi's VOCR6 LM
  settings apply (alpha 600, beta 2500, BOS): on dev the shipped
  1 MB game-text LM, on hval its screening build, as vocr6-lead reads hval.
- **Unseen dev:** the pick reads 0.95% (Vita 0.77 / PSP 1.12) against `rubi,fix=1`'s 2.41%: −1.46 [−4.04, +0.11].
- **hval:** 1.03% against 1.52%: −0.49 [−1.31, −0.02].
- **Tier L alone** (no cascade): dev 0.95% against 2.37%, hval 1.01% against 1.51%.
- **The guard holds,** and on these 92 unseen-game VNs the page path reads dialogue at **0.95%: under the owner's 1%
  target, measured with VOCR6-N → L**.

### What the changes bought (report only)

- **Against textspot6's control** (base_t64n8c16, the same shape, data and schedule; paired by seed number):
  - **Crop decode:** seed 1 −1.06 [−1.51, −0.73], seed 2 −0.71 [−1.01, −0.45], seed 3 −0.88 [−1.47, −0.45]. Resolved
    for every seed. hval moves +0.25 / +0.04 / −0.23, none resolved.
  - **Det decode:** −0.07 / +0.14 / −0.13, none resolved.
  - So T8's settings made the region-only read as good as the whole-frame one, and better. Which of them did it was
    not separated: the arm changes them together. The tight-region windows train exactly that read.
- **Against `rubi,fix=1`, paired per unit** (the pick, dev without v27449, 91 VNs):

  | the spotter's unit | `rubi,fix=1`'s unit | units | errors: spotter / fix=1 |
  |---|---|---:|---|
  | every line whole | a line missed | 15 | 42 / 115 |
  | a line cut | every line whole | 17 | 23 / 9 |
  | a line merged | a junk box | 2 | 14 / 0 |
  | every line whole | every line whole | 996 | 96 / 108 |
  | every line whole | a line cut | 6 | 0 / 12 |
  | every line whole | a junk box | 54 | 6 / 15 |
  | a junk box | every line whole | 20 | 7 / 0 |
  | all 91 VNs | | 1,119 | 222 / 294 |

  textspot6's crop decode lost +99 errors on cut lines here; the student loses +14. On v27449 it makes 20-22 errors
  against GRAD's 311.
- **Where it still loses:**
  - v1971, the side-by-side choice buttons (28 → 58 errors), where every spotter and MeikiOCR lose too;
  - single regions of v3496, v6985 and v5000.
- **Whole frames** (textspot's 188 dev frames, det): box F1 0.56-0.58, UI recall 0.22-0.23, dialogue CER 1.61-1.83%.
  That is textspot6's level and below MeikiOCR's (0.895 / 0.77 / 1.45%). The student reads dialogue lines, not the
  rest of the screen. No dialogue box appears on any of the 25 text-free frames.
- **hval boxes** against the verified lines (F1, seed 3): Vita 0.860, PSP 2x 0.949, PSP native 0.935.

### Empty regions (report only)

eval/linefind's 66 development and 25 report text-free regions (textgate's set), read with the crop decode as Rubi
reads a region (`eval/textspot8_empty.py`). The share is the regions given no line at all:

| finder | development: no line | report: no line |
|---|---:|---:|
| GRAD + refine (textgate's record) | 5 / 66 | |
| textspot6 base_t64n8c16 s1 | 62 / 66 | 25 / 25 |
| textspot6 research release (qS4) | 57 / 66 | 25 / 25 |
| student s1 / s2 / **s3** | 39 / 30 / **39** of 66 | 19 / 21 / **18** of 25 |

- The student leaves fewer empty regions empty than textspot6 did: 59% against 86-94%. Its tight-region and zoom
  windows probably taught it to box more faint marks, but that was not checked.
- It still does far better than GRAD + refine (8%). That is why its Rubi path does not fall back to GRAD when it finds
  no line: on the measured sets with text it returns a line in every unit (0 of 1,161 dev, 6,327 hval and 516
  confirmation units without a box), so a fallback on "no line" would only bring GRAD's junk back to empty regions.
- The text gate (#100, opt-in) stays a GRAD-path option.

### Spinners and page cursors (report only)

276 dev units (the regions where an icon drawn 0.35 line heights after the last line's end lies inside the region),
with an icon of the trained kinds or of three unseen kinds.

| finder | the icon inside a line box: trained / unseen icons | junk boxes | CER: trained / unseen |
|---|---|---|---|
| `rubi,fix=1` | 97.8% / 96.4% | 24 / 21 | 2.33 / 2.98 |
| textspot6 base_t64n8c16 s1, crop | 75.7% / 82.3% | 9 / 8 | 1.31 / 1.80 |
| **student s3, crop (the pick)** | 81.2% / 93.8% | 16 / 23 | **0.75 / 1.36** |
| student s3, det | 74.3% / 88.8% | 10 / 20 | 0.80 / 1.47 |

- The student reads the text next to an icon best, with a third to a half of GRAD's errors.
- Its line boxes still cover most of an icon this close to the text: their end is the last character plus the
  decode's pad and cell. So a checksum over the line boxes still sees a spinner that close.
- rubi-spinner's PR covers that case with a timeout path. Its line check takes the active finder's lines, so the
  spotter drops in.

### Vertical text in Rubi (report only)

Rubi's default `ocr_vertical = auto` checks GRAD's rows for vertical text (API 47). With the spotter as the finder,
Rubi hands a region to GRAD's rows when the spotter's crop decode returns more vertical than horizontal boxes in it
(`eval/textspot8_orient.py`, seed 3): **0 of 1,161** development units and **0 of 516** confirmation units
(horizontal dialogue), and **172 of 181** vertical development regions (79 of 84 VN pages, 93 of 97 manga windows;
eval/manifests/vertical-dev-regions.jsonl). So horizontal dialogue reads with the spotter and vertical text goes to
auto's rule as before.

### Cost (estimates; nothing has run on a console)

The pick is textspot6's `t64n8c16` shape, so textspot6's A9 rows of record apply (`eval/textspot6_a9_20261026.json`,
the Vita compiler's `-O2` code with the dark pass, priced by the Cortex-A9 cycle model):

| read | est. ms @333 / @444 MHz | arena |
|---|---:|---:|
| crop decode (the pick), a 900×150 region | 82.1 / 61.6 | 235,119 B |
| crop decode, 900×200 | 104.7 / 78.5 | 235,119 B |
| crop decode, 900×300 | 150.5 / 112.9 | 235,119 B |
| crop decode, a whole 960×544 region (the rule's bound) | 264.2 / 198.2 | 244,607 B |

VTSP 52,176 B. MeikiOCR's detector, the reference: 3,547,383 float32 parameters, 14,503,825 B of ONNX, 137 ms per
960×544 frame on one host thread (95 ms on 5), the EPYC host under the shared box's load.

### Disclosures found after the protocol

- **A wrong first analysis, corrected before any student read:** amendment 1's "why" first blamed crop geometry. The
  paired check and step 2 refuted it; the protocol's `corrections` and #149 record the correction. No arm or rule
  changed.
- **The stopped run:** amendment 2 stopped the first seed-1 run before any score.
- **Amendment 5 came after textspot7's reads of other models** (above).
- **The branch was rebased onto main** after textspot6 merged (#147, 4674e54): the registered commits keep their
  times; the tag `textspot8-registered-20261027` holds the branch as registered and run. The training worktree ran
  at 7eb4484, the amendment-1 commit (its tree is the rebased 1e681d7's for every file the trainer reads).
- **Release partners of training titles:** as textspot6's (PCSG00007 and ULJM05383 share VNDB release r3748 with
  CLANNAD; PCSG00866 shares releases with v21789, a vita-test VN). The owner's 2026-10-25 policy allows them.

<!-- TESTS -->

### Reproduction

```sh
S=eval/reproducibility/textspot8-20261027
# step 1 (MeikiOCR) and step 2 (geometry), one heavy slot each
HEAVY_SLOTS=12 $HERD/heavy.sh bash $S/bench_meiki.sh            # spot decode pp tsdev boxes speed
HEAVY_SLOTS=12 $HERD/heavy.sh bash $S/bench_meiki.sh geom
.venv/bin/python $S/geom_rule.py --geom $A/vocr6-spotter2/eval/geom --base-eval $A/vocr6-spotter/eval --out GEOM.json
# step 3: the student, from a worktree at the amendment-1 commit (seeds 1-3; the final: train,guard,hval)
HEAVY_SLOTS=12 $HERD/heavy.sh bash $S/run_train8.sh t64n8c16 SEED $RUNS '{"across_shrink": 0.0}' [train,guard,hval]
# scoring per seed: textspot6's decodes, the context decode, textspot7's D / F / F+D, the spinner check
bash $S/score_run8.sh $RUNS/t8_t64n8c16_seedN t8_t64n8c16_seedN
bash $S/score_ctx8.sh $RUNS/t8_t64n8c16_seedN t8_t64n8c16_seedN
T7=<textspot7 worktree at 3781f2f> bash $S/score_d8.sh t8_t64n8c16_seedN
bash $S/spinner_runs.sh blob t8_t64n8c16_seedN $RUNS/t8_t64n8c16_seedN/final.vtsp "det crop"
# the rule (textspot6_select's code), then the confirmation read once, then the VOCR6 guard
.venv/bin/python eval/textspot8_table.py --eval $A/vocr6-spotter2/eval --runs $RUNS --base-eval $A/vocr6-spotter/eval \
    --out TABLE.json
.venv/bin/python eval/textspot8_select.py --table TABLE.json --out eval/textspot8_dev_20261027.json
bash $S/confirm_read.sh
bash $S/guard_vocr6.sh t8_t64n8c16_seed3 crop
# report only
.venv/bin/python eval/textspot8_empty.py --blob B.vtsp --out EMPTY.json
.venv/bin/python eval/textspot8_orient.py --blob B.vtsp --out ORIENT.json
```

