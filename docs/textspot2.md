# textspot2: a better text spotter (issue #106)

textspot (#104, `docs/textspot.md`) selected a full-frame text spotter of 5,732 B of int8 weights,
`B_h2_s4_c24n4_seed2` with det+refine (est. 139 ms per 960×544 frame on one 333 MHz A9 core), and published the
smallest candidate that cleared its bar beside it, `C_h2_s4p_c8_seed1` (2,732 B, est. 80 ms). The owner
(2026-10-20 07:53-07:58Z): "how can we improve the 2.7kb model?", "can we use [meiki.text.detect.v0] as the
teacher?", "we can maybe make it bigger too, i think 2.7kb is crazy small for this".

Weights are not the constraint: Rubi's recognizers are 0.8-1.6 MB, and the spotter's working memory (72-81 KB) is set
by the frame's width, not by the weights. Scan screen runs on a button press, so the constraints are **time and
arena**. textspot2 looks for the most accurate spotter within that budget, and for a fast pick beside it.

This document starts as the pre-registered protocol (`eval/textspot2_protocol.json`), frozen before any textspot2
run, label or data job and score. Results are added below "Results" as they are measured. Nothing above "Results"
changes except through a dated amendment listed in the protocol's `amendments`.

## Protocol

### What is reused unchanged

From textspot, as frozen and merged (`7a6dce0`):
- **the development set**: `eval/manifests/textspot-dev.jsonl` (188 frames of 81 dev_sample VNs, 885 instances,
  sha256 `0392a1d3…`), the only data the rule reads;
- **the metrics**: `eval/textspot_metrics.py` (P / R / F1 at IoU ≥ 0.5, the end-to-end CER all and dialogue, the
  VN-cluster bootstrap: 10,000 resamples, seed 0);
- **the harness**: `eval/textspot.py` (`det:` and `detref:` arms, the reading by the C runtime in W8I with greedy
  decoding: VOCR5-N for horizontal boxes, VOCR5-V for vertical ones, the four recognizers' sha256 as textspot pins
  them; C parity per frame with `--cbin`) and the frozen baseline `eval/textspot_baseline_20261019.json` with its
  eight baselines;
- **the spotter**: the VTSP blob, the twin `synth/textspot_twin.py` and the runtime `runtime/vocr_textspot.c`
  (unchanged unless the d2s extension below is needed by a selected model);
- **the estimates**: the A9 cycle model's high bound × 1.0007 at 333 MHz on the runtime's code
  (`eval/textspot_a9.py`), plus 1.4 ms per refined horizontal box at the arm's median on the development Vita
  frames for det+refine; arenas exact (`vocr_textspot_arena_bytes(m, 960, 544)`, and the refinement's 72,391 B);
- **the test set and CLANNAD**: `eval/manifests/textspot-test.jsonl` (textspot amendment 3) and the CLANNAD long run,
  read through `eval/textspot_protected.py`.

### The references

| role | arm | VTSP B | est. ms @333 | arena B | dev F1 | CER all | dialogue |
|---|---|---:|---:|---:|---:|---:|---:|
| `ref`: the textspot winner | `B_h2_s4_c24n4_seed2` det+refine | 5,732 | 139.09 | 80,799 | 0.5416 | 32.04 | 3.01 |
| `fast_ref`: textspot's fastest qualifier | `C_h2_s4p_c8_seed1` det+refine | 2,732 | 80.25 | 72,391 | 0.5327 | 34.92 | 3.64 |

Their development units (VOCR5-N) are textspot's, read, not re-run; the protocol pins their sha256. `ref` is the
better of its shape's two seeds, so comparisons against it are conservative.

### Budget

- est. ≤ **150 ms** per 960×544 frame on one A9 core at 333 MHz (444 MHz and the PSP's 480×272 are reported too);
- arena at 960×544 ≤ **131,072 B** (the always-safe tier; for det+refine the larger of the spotter's and the
  refinement's);
- VTSP blob ≤ **262,144 B** of int8 weights and header;
- the C runtime gives the twin's boxes on every development frame;
- the **fast pick**: the same at est. ≤ **80 ms**.

### Metrics

- **Primary** (UI-inclusive, as textspot's): detection F1 over every class, and the end-to-end all-text CER (every
  class, plus the characters read from boxes that match no reference).
- **Guard**: dialogue CER (classes dialogue and choice, without unassigned readings).
- **Secondary** (report only): CER over the dialogue, name and choice references only, without unassigned readings
  (`textspot2_rule.dnc_units`).
- **Report**: recall per class, tightness, text-free frames with a box, the dialogue pick, per part; VOCR5-S and
  VOCR5-L readings for the selected models.

### Seeds

Every textspot2 arm (a recipe, a shape, a scale) is trained with seeds 1, 2 and 3, and every seed is reported. For
each decode (det, det+refine) the arm is represented by its **median seed** by development all-text CER
(`textspot2_rule.median_seed`: sorted by all-text CER, dialogue CER, −F1, seed; index n // 2, so with two finished
seeds the worse one). Never the best seed. Eligibility is then checked on that seed's blob, with no substitution. A
crashed run restarts from scratch with the same seed, at most three attempts.

### The rule (pre-registered; `eval/textspot2_rule.apply_rule2`)

- **Eligible:** within the budget above (main: ≤ 150 ms).
- **Clears textspot's bar:** `textspot_metrics.beats` against every one of the eight frozen baselines (F1 resolved
  above, all-text CER resolved below, dialogue CER not resolved above).
- **Better than X** (`textspot2_rule.better`), on the paired development frames with VN-cluster 95% intervals:
  - resolved better on all-text CER (interval below 0) **or** on F1 (interval above 0);
  - and resolved worse on **none** of all-text CER, F1 and dialogue CER.
- **Cost order:** est. ms at 333 MHz, then VTSP bytes, then arena, then det before det+refine.
- **Main pick:** among the eligible candidates that clear the bar and are better than `ref`: the cheapest; a
  costlier one replaces it only when it is better than it (the cheapest such, repeated until none is). So a
  smaller or faster candidate wins when the bigger one is not resolved better.
- **Fast pick:** the same among the candidates at est. ≤ 80 ms (tier 1). If none of them is better than `ref`, the
  same with `fast_ref` as the comparator (tier 2, labelled so). If none is, there is no fast pick. When the main
  pick is itself at ≤ 80 ms, it can be the fast pick too.
- **No main pick:** the verdict is negative and textspot's winner stands.
- **Protected sets:** after the selection is committed and pushed, each selected model is read **once** on
  vita-test + psp-test as whole frames and on CLANNAD (`eval/textspot2_protected.py`, its own attempt records under
  `$HERD/shared/private/textspot2`), with paired comparisons against `ref`'s own protected units (textspot's
  record, read, not re-run). Nothing is read when nothing is selected. No model of this work runs on a test frame
  or a CLANNAD page before that.

### Data

**Excluded from all training** (as textspot): every `reserved.json` test and dev_sample VN and test session; every
eval and eval/linefind screenshot and every linefind VN; CHAOS;CHILD (v14018); every development and test frame's
screenshot. Every new pool passes `eval/tools/reserve_check.py`.

**The synthetic half** is textspot's (`data/manifests/textspot_shards_v1.json`, 80,000 whole frames), unchanged.

**The teacher.** meiki.text.detect v0.1 (`rtr46/meiki.text.detect.v0` at revision `a9cffa4f…`, the 960×544 file
pinned in `teacher/engines.py`): a D-FINE box detector, LGPL-3.0. The spotters train on its outputs (boxes and
scores, no feature or map distillation); its weights are never shipped.

**b. The textspot2 labels** of a real canvas (the same canvases as textspot's real half):
1. the canvas's colour source at the canvas's size, made as #97 made the canvas (checked: its grey equals the
   canvas; a canvas without its source is read as its grey in three channels, counted);
2. meiki's boxes and scores on it (score ≥ 0.2), each refined to ink with `teacher/geometry.refine`;
3. **positives**: the canvas's existing positives (the reading pipelines' lines, as textspot), plus every meiki box of
   score ≥ 0.5 that no existing positive covers (≥ 50% of its area), vertical when its height is at least 1.5× its
   width, kept when it is 6-96 px across (else an ignore area);
4. **ignore areas**: meiki boxes of score 0.2-0.5; the textspot winner's boxes (det decode, C runtime) that no meiki
   box ≥ 0.2 and no positive covers (≥ 50% of the box), so text meiki missed is neither positive nor negative; the
   canvas's existing ignore areas, except one that a new positive's meiki box overlaps at IoU ≥ 0.5 (#97 made
   uncovered meiki boxes ignore areas; they are now positives);
5. **weights**: 1 for existing positives and for a meiki positive the winner agrees with (≥ 50% of its box inside a
   winner box); else 0.5 + 0.5 min(1, (score − 0.5) / 0.3).

A positive's weight multiplies the K cross-entropy, O and A losses of its kernel cells; every other valid cell weighs
1; dice and the normalisations are #97's.

**c. More real data**: the same labels on more eligible screenshots of the local pools:
- VNDB PC screenshots (realplus's and the teacher's) at the Vita target, as #97 made its realplus canvases: up to 8
  per VN, existing canvases included;
- native Vita and PSP VNDB screenshots (the teacher's and devnative's) at their own size: up to 12 per VN;
- the owner's training captures: up to 200 per VN (the `-TEST` folders and every reserved VN out);
- within a VN, screenshots in sha256(`textspot2-pool:` + shot) order.

New canvases join their source's pool, so the mix per source stays textspot's (synthetic 0.5; realplus, teacher,
owner, devnative 0.175, 0.175, 0.125, 0.025).

The shards and their manifest are pinned by an amendment before any run reads them.

### Recipes

| recipe | arm | what |
|---|---|---|
| `r0` | control | textspot's recipe, data and trainer (`train/textspot_train.py`), unchanged: seed 3 of `s4_c24n4` and `s4p_c8` (seeds 1-2 are textspot's runs). Report only, never a candidate |
| `lab` | b | `r0` with the real half's labels replaced by the textspot2 labels (same canvases) |
| `data` | c | `lab` plus the enlarged real pool |
| `ema` | e | `data` + an exponential moving average of the weights (decay 0.999 from step 1,000; at the QAT start the averaged weights replace the live ones, BatchNorm statistics re-estimated on 32 batches; during QAT an average of the QAT weights, which is the run's model) |
| `cls` | f | `data` + class weights: 2 on the lines of class dialogue, name and choice (synthetic frames; the real half has no classes), 1 otherwise |
| `long` | e | `data` + `ema` + 60,000 steps, peak LR 2e-3 (1,000 warm-up steps, cosine to 1e-4, QAT from step 48,000) |
| `kd` | e | `data` + the kept extras + distillation from a teacher spotter (below): loss = labels + K BCE to σ(K_t) on valid cells + 0.5 O BCE to σ(O_t) and A1 / A2 smooth L1 to the teacher's (÷ 8 px) on cells where σ(K_t) > 0.5 |

Everything else is textspot's recipe: 16 windows of 256×512 at the network's scale, h2 (`half_above` 600), AdamW
4e-3 (except `long`), 500 warm-up steps, cosine to 1e-4, QAT from 80% of the steps, 30,000 steps (except `long`), the
last step is the run's model. New trainer: `train/textspot2_train.py` (textspot's trainer imported, its pinned files
unchanged).

### Shapes (all at h2; MACs exact, est. ms from a fit to textspot's A9 table until measured)

| shape | layers | int8 weights | est. ms @333 (det) | role |
|---|---|---:|---:|---|
| `s4p_c8` | textspot's: plain 3×3, 8 channels, dilations 1-2-4 | 2,084 | 71.9 (measured) | fast reference |
| `s4p_c6n4` | plain, 6 channels, 4 blocks, dilations 1-2-4-8 | 1,572 | ~62 | fast, receptive field 127 |
| `s4p_c8n4` | plain, 8 channels, 4 blocks, dilations 1-2-4-8 | 2,660 | ~85 | between |
| `s4p_c10n4` | plain, 10 channels, 4 blocks, dilations 1-2-4-8 | 4,036 | ~113 | big, deep |
| `s4p_c12` | plain, stem 6, 12 channels, 3 blocks, dilations 1-2-4 | 4,638 | ~125 | big, wide |
| `s4_c24n4` | textspot's winner shape: separable, 24 channels, 4 blocks | 3,272 | 130.7 (measured) | reference |
| `s4_c32n3` | separable, 32 channels, 3 blocks, dilations 1-2-4 | 3,680 | ~132 | wide separable |
| `s8_c24n4x2` | separable stride-8 trunk (#97's `s8_c24n4`), 2×2 sub-pixel head | ~4,350 | ~70 | d2s, fast (contingent) |
| `s8_c48n4x2` | separable stride-8 trunk, 48 channels, 4 blocks, 2×2 sub-pixel head | ~13,420 | ~111 | d2s, big (contingent) |

**d2s (contingent).** A stride-8 trunk is about four times cheaper per parameter than a stride-4 one, but textspot's
stride-8 spotters read dialogue at 8.5-13% CER: at half scale a cell is 16 frame pixels and the kernels of
neighbouring lines touch. A 2×2 sub-pixel head gives the trunk stride-4 maps: its 16 channels are K, O, A1, A2 for each
of a cell's four sub-cells, and the decode reads them as a stride-4 map (the textspot decode, unchanged). This needs a
VTSP version 2 (the twin, the runtime, the export and the trainer), flag-gated by the blob: a version-1 blob runs bit
for bit as before, and a runtime without the extension refuses version 2. The d2s arms run only if that extension,
its parity tests and its A9 rows are committed before stage 2 starts; else they are reported as not run. The
runtime change is merged only if a selected model needs it.

### Arms and stages

**Stage 1** (after the data amendment; one selection at the end reads every stage):
- the ladder on `s4_c24n4`: `lab`, `data`, `ema`, `cls` (3 seeds each), and `r0` seed 3;
- `long` on `s4_c24n4` and `s4p_c8` (3 seeds each);
- every shape above except the d2s ones under `data` (3 seeds each), and `r0` seed 3 of `s4p_c8`.

**Stage 2**, after every stage-1 run is scored, by fixed rules that read the development scores (so stage 2's
development intervals are optimistic; the protected sets are the check, as in textspot's family C):
- **kept extras**: `ema` is kept when its median seed's det+refine all-text CER is below `data`'s on `s4_c24n4`;
  `cls` when its is below `data`'s and its dialogue CER at most 0.5 points above;
- **the fast shape**: among the stage-1 `data` arms, the shape whose median-seed candidate (det or det+refine) has
  the lowest all-text CER among those eligible at ≤ 80 ms (`s4p_c8` when none is);
- **the main shape**: the same among those eligible at ≤ 150 ms;
- **`kd`**: the fast shape with the kept extras, distilled from the teacher spotter: the eligible stage-1 candidate
  with the lowest median-seed all-text CER (any recipe and decode), its median seed's final model (3 seeds);
- **`long`** on the main shape and the fast shape when they are not `s4_c24n4` / `s4p_c8`, with the kept extras
  (3 seeds each);
- **d2s**: `s8_c24n4x2` and `s8_c48n4x2` under `data` + the kept extras (3 seeds each), when the runtime extension
  is ready.

Every stage-2 choice is recorded by an amendment before its runs, with the figures it read.

**a. The teacher ceiling** (report only; host only; never a candidate): meiki.text.detect v0.1 at 960×544 and at
320×192 (`meiki.text.detect.v0.1.320x192.onnx`, the same repository and revision, sha256 `8cdc5daa…`) on the
development frames through the same reading and metrics: its boxes at score ≥ 0.5 as they are, padded by the
spotter's pad rule, and padded with det+refine.

**What each arm bought** (report): paired development differences of median seeds (det+refine and det): `lab` −
`r0`, `data` − `lab`, `ema` − `data`, `cls` − `data`, `long` − `ema` (`s4_c24n4`), each shape's `data` − `data`
`s4_c24n4`, `kd` − its shape's best non-`kd` arm, d2s − `data` `s4_c24n4`, and every pick and the ceiling against
`ref`.

### Disclosures registered in advance

- textspot's development references started from MeikiOCR's boxes refined to ink (reviewed by two looks; 96.7% of
  the boxes approved), and so did its test set outside eval's hand-checked lines. meiki's ceiling, and spotters
  trained on meiki's boxes, are favoured by that construction; the dialogue lines of the test set (eval's) and
  CLANNAD's references are independent of it.
- The development set has 81 VNs; with about 30 candidates, intervals near 0 can still be chance.

### Estimates

Binding: the A9 cycle model, as textspot (`eval/textspot_a9.py` on the shapes registered by textspot2's trainer;
`eval/textspot2_a9_<date>.json`, pinned by an amendment before any score). Report: 444 MHz, the PSP's 480×272 and
VitaSDK `-Os` code sizes. Every console time is a host estimate: nothing runs on a console (hardware work is paused).

### Amendments

Each amendment is dated in `eval/textspot2_protocol.json` and precedes the runs and scores it names.

1. **Label, data and ceiling precisions** (2026-10-20T08:38Z, before any label or data job, run or score).
   - **Coverage**: "covered (≥ 50%)" means at least half the box lies inside the *union* of the named boxes, so a
     line meiki splits in two still counts as covered.
   - **Furigana**: a meiki box is an ignore area, never a positive, when either:
     - the source record's furigana boxes cover half of it; or
     - it is furigana of a bigger box by `teacher/geometry.is_furigana`'s geometry, without its kana test (meiki's
       detector reads no text).
   - **Refinement**: a vertical box is refined to ink on the transposed crop; the 6-96 px gates read the refined
     box.
   - **The teacher call**: `teacher/engines.Meiki`'s detector, BGR input as `teacher/label.py` feeds it, threshold
     0.2, no recognition.
   - **Shards**: #97's format, lines `[x0, y0, x1, y1, vertical, class, weight]`. A lab shard's canvases may be
     symlinks to #97's. textspot's skip lists are applied at build time and carried in the manifest. Owner
     canvases stay private.
   - **New screenshots (c)**: those no #97 canvas holds. Every development and test frame's source is excluded, and
     the new pool must pass `reserve_check`.
   - **The ceiling (a)**: four report-only arms, `meiki960_det` / `_detref` and `meiki320_det` / `_detref`:
     - boxes of score ≥ 0.5, vertical when taller than wide (MeikiOCR's rule);
     - padded by the spotter's pad rule;
     - the `_detref` arms put their horizontal boxes through the refinement.
   - **Names**: runs `<recipe>_h2_<shape>_seed<n>`, arms `T2_<run>_det` / `_detref`. r0's seeds 1-2 are textspot's
     own runs, and their units are read, not re-run.
2. **Two label rules, from a look at training canvases** (2026-10-20T09:28Z, before any shard is built, any run and
   any score; the labels' cache of meiki and winner boxes existed, nothing had been built from it).
   - **Winner-only boxes** become ignore areas only at a winner confidence ≥ 40 (of 127); below it they stay
     negatives, as in textspot's labels.
     - Evidence: the QA sheet of 40 random winner-only boxes on training canvases. The conductor counted 21 text
       (UI buttons, name plates, choices) and 19 not text (icons, ornaments, patterns).
     - 19 of the 21 text boxes have confidence ≥ 40, and 15 of the 19 others are below it.
     - Without the threshold, the winner's own false positives would have lost their negative labels.
   - **Low-score boxes** (score 0.2-0.5) that the existing positives cover are dropped, so the negatives between
     labelled lines stay. Uncovered ones remain ignore areas: on their sheet (19 text, 17 not, 4 unsure) the score
     does not separate text from non-text.
   - The labels helper's ten readings of the rules (refinement order, caps, hygiene, the skip block) are recorded
     verbatim in the protocol. Shard directories are named `lab_<source>` / `more_<source>`.

## Results

Nothing is measured yet.
