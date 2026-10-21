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
     in the protocol's amendment 2. Shard directories are named `lab_<source>` / `more_<source>`.
3. **The scoring harness, the eligibility tables, the d2s extension and the data** (2026-10-20T10:07Z, before any
   score of a textspot2 arm and any run).
   - **Scoring**: `eval/textspot2.py` (run, ceiling, select, tables) and `eval/textspot2_protected.py`. They call
     textspot's harness, metrics and the frozen rule unchanged, with a reading cache of their own.
   - **Eligibility**: `eval/textspot2_a9_20261020.json` (the seven shapes; textspot's two re-measured, identical) and
     `eval/textspot2_a9_d2s_20261020.json` (the d2s shapes on the new runtime file). textspot's own table gives its
     two shapes' rows.
   - **d2s**: VTSP version 2: the stride-8 trunk's 16 head channels are a cell's four sub-cells, read as stride-4
     maps, then textspot's decode. `synth/textspot2_twin.py`, `train/textspot2_export.py` and
     `runtime/vocr_textspot2.c` implement it. A version-1 blob runs bit for bit as before, and parity holds on host,
     under sanitizers and under qemu.
   - **Data** (`data/manifests/textspot2_shards_v1.json`): 52,785 real canvases with 299,986 positives:
     - lab: textspot's 28,265 canvases, relabelled;
     - more: 24,520 new screenshots, mostly VNDB PC screenshots;
     - RESERVE_OK, and no leak against vita-test / psp-test.

     Known label noise: long thin vertical strips taken as vertical text, mostly at weights 0.53-0.60.
4. **The trainer and stage 1's run list** (2026-10-20T10:46Z, before any run).
   - `train/textspot2_train.py` is one file for both stages. Its d2s path changed no other path: logs, weights and
     blobs are identical before and after it was added.
   - `eval/reproducibility/textspot2-20261020/stage1_plan.jsonl` lists the 38 stage-1 runs, longest first,
     47.5 slot-hours estimated. At most 6 run at once, from a detached worktree at this commit.
   - The data-path probe read every pinned shard; the skip lists drop nothing, as expected.
5. **Stage 2's choices** (2026-10-20T20:07Z, after all 38 stage-1 runs were trained and scored, before any stage-2
   run and before the selection). The fixed rules, applied to stage 1's median seeds (point estimates):
   - **ema is kept** (det+refine all-text CER on `s4_c24n4`: 34.74 against `data`'s 35.33); **cls is not** (35.86).
   - **The fast shape is `s4p_c6n4`**: its det+refine (est. 73.1 ms) reads all text at 40.61%, against `s4p_c8`
     det's 40.86%. `s4p_c8` det+refine is est. 81.7 ms, so over 80 ms.
   - **The main shape is `s4_c24n4`** (`data` det+refine 35.33%, the lowest at ≤ 150 ms).
   - **The kd teacher** is `lab_h2_s4_c24n4_seed1` (`lab` det+refine, 34.36%, the lowest of any recipe).
   - **Runs**: `long` on `s4p_c6n4`, plus the two d2s shapes under `data` + ema. 12 runs in all, at most 7 at once,
     from a new worktree at this commit.
   - Disclosed: no stage-1 median seed reads all text below the reference's 32.04% (the closest is 34.36%).
6. **Merge scope of the d2s runtime** (2026-10-20T23:49Z, after the selection commit; it changes no score, rule or
   selection). A disclosed deviation from "d2s" above: no model was selected, yet `runtime/vocr_textspot2.c` and its
   driver, make target and tests stay, as a host and research tool.
   - The committed selection's d2s rows (C parity on every development frame, the A9 rows) were measured with them,
     and the pinned harness builds them.
   - Nothing on a console changes: `runtime/vocr_textspot.c` is unchanged, and no console build, no Rubi build and
     no release uses the new file (`runtime/README.md` says so).
   - The d2s direction gave this work's largest dialogue gain, so a follow-up would start from it.

## Results

Every number here is a host measurement on the development set, unless it says otherwise:
- 188 frames of 81 VNs, read by the C runtimes;
- horizontal boxes read by VOCR5-N, vertical ones by VOCR5-V;
- textspot's frozen metrics, with VN-cluster 95% intervals.

Every time is an estimate from the A9 cycle model; nothing of this work has run on a console.

### Answer first

- **The verdict is negative on both picks: textspot's winner stands.** That is `B_h2_s4_c24n4_seed2` det+refine:
  5,732 B, est. 139 ms per 960×544 frame at 333 MHz, F1 0.542, CER all 32.04%, dialogue 3.01%.
  - The selection (`eval/textspot2_dev_20261020.json`, commit `180cfaa`) found 32 candidates, all eligible. 9 of
    them clear textspot's bar against its eight baselines.
  - No candidate is better than the winner on the primary metrics, so there is no main pick.
  - No candidate at est. ≤ 80 ms is better than the winner, nor than textspot's fastest qualifier
    (`C_h2_s4p_c8_seed1`, 2,732 B, est. 80 ms), so there is no fast pick.
  - As registered, nothing was read on vita-test, psp-test or CLANNAD, nothing was released, and Rubi is not
    changed.
- **Why: no median seed reaches the winner's all-text CER.** Every candidate's median seed reads all text at
  34.36-43.83%, against the winner's 32.04%, and 29 of the 32 are resolved worse.
  - **For the closest candidates, most of that gap is the reference's seed.** The winner is the best of its
    recipe's three seeds. That recipe, r0, reads all text at 32.04 / 34.59 / 36.28% over seeds 2, 1 and 3, so its
    median seed is itself 2.56 points behind the winner (paired +2.56, −0.11 to +5.09).
    - That is more than half the gap of the 10 det+refine candidates with at least 10 channels (34.36-36.82%).
    - The other 22, every det decode and every 6-8-channel trunk, are 5.13-11.79 points behind, mostly on their own
      account (the seed is 22-50% of their gaps).
    - That includes the whole fast pool. Against its own comparator, `fast_ref` (itself the best of its recipe's three
      seeds, 34.92 / 36.71 / 38.66%), the fast pool is 3.93-8.91 points behind, all resolved worse.
  - **Against the matched control, the new labels are level on all-text CER and better on dialogue.** `lab` minus
    r0, both median seeds, is −0.24 points (−2.42 to +1.81) on all-text CER and −1.61 (−3.62 to −0.13) on dialogue.
    `lab`'s 2.32-point gap to the winner is smaller than that seed gap. For `data` and d2s `s8_c24n4x2`, the arm's
    own share of its gap is 0.74 of 3.29 and 1.15 of 3.71 points.
  - **F1 is level or worse**: within ±0.02 for 20 of the 32, 8 resolved worse (−0.025 to −0.069), none better. The
    best F1 is `s8_c48n4x2` det+refine at 0.583, +0.042 (−0.001 to +0.088), not resolved.
- **What did improve is dialogue** (report only; the rule reads all-text CER and F1 first). Three candidates are
  resolved better than the winner on dialogue CER:

  | candidate | VTSP B | est. ms @333 | F1 | CER all | dialogue (vs winner) |
  |---|---:|---:|---:|---:|---|
  | d2s `s8_c24n4x2` det+refine | 7,416 | 80.2 | 0.539 | 35.75 | **1.13%** (−1.88, −3.36 to −0.77) |
  | d2s `s8_c48n4x2` det+refine | 18,568 | 129.5 | 0.583 | 36.34 | 1.53% (−1.48, −2.93 to −0.35) |
  | `s4p_c10n4` det+refine (plain 3×3, 4 blocks) | 4,888 | 132.0 | 0.535 | 36.57 | 1.76% (−1.25, −2.74 to −0.16) |

  The d2s head (a stride-8 trunk whose 16 output channels are the maps of each cell's four stride-4 sub-cells) is
  this work's clearest gain. `s8_c24n4x2` reads dialogue at 1.13% against the winner's 3.01%, in about 58% of its
  estimated time. It misses the fast budget by 0.2 ms with refinement, and with or without refinement it is not
  resolved better than either reference on the primary metrics.
- **Where the all-text errors are** (character errors in the development references, by class, plus readings of
  boxes that match no reference):

  | arm (det+refine, median seed unless named) | dialogue | name | choice | UI | scene | unmatched readings | total | CER all |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | textspot's winner (r0's recipe, seed 2) | 114 | 74 | 6 | 1,557 | 22 | 396 | 2,169 | 32.04 |
  | r0's recipe, seed 1 (r0's median) | 161 | 83 | 6 | 1,551 | 23 | 518 | 2,342 | 34.59 |
  | r0's recipe, seed 3 | 211 | 85 | 5 | 1,579 | 22 | 554 | 2,456 | 36.28 |
  | `lab` `s4_c24n4` | 98 | 72 | 5 | 1,522 | 24 | 605 | 2,326 | 34.36 |
  | `data` `s4_c24n4` | 90 | 66 | 8 | 1,590 | 21 | 617 | 2,392 | 35.33 |
  | d2s `s8_c24n4x2` | 41 | 79 | 4 | 1,637 | 27 | 632 | 2,420 | 35.75 |

  Reference characters: dialogue 3,851, name 298, choice 136, UI 2,455, scene 30.
  - UI text is 72% of the winner's errors. These three arms make 1,522-1,637 UI errors, against r0's 1,551-1,579 and
    the winner's 1,557. Of the three, only `lab` is below the winner, and its UI recall is 0.364 against 0.324.
  - These three read dialogue better and read more characters from boxes that match no reference:
    - 605-632 of them, 50-60% more than the winner's 396;
    - 17-22% more than r0's median seed's 518.

    So the meiki labels add unmatched readings, but most of these three arms' gap to the winner is its own seed.
  - **Extra readings against the dialogue gain**:
    - Against the winner they cost more than the dialogue gains save: `lab` +209 against −16, `data` +221 against
      −24, d2s +236 against −73.
    - Against r0's median they outweigh the dialogue gain for `lab` (+87 against −63) and `data` (+99 against −71),
      not for d2s (+114 against −120).
    - In total errors `lab` is 16 below r0's median, because its UI and name errors fell too. `data` and d2s are 50
      and 78 above it, d2s mostly through UI (+86): against its matched control, d2s's problem is missed UI text,
      not stray readings.
- **The teacher's ceiling is far away.** meiki.text.detect v0.1 at 960×544 (14 MB, report only, never a candidate)
  scores F1 0.899, CER all 14.39%, dialogue 1.18%, and finds 78% of UI labels against the winner's 32%.
  - Caveat: the development references started from MeikiOCR's boxes (docs/textspot.md), so this ceiling is
    favoured by construction.
  - The 320×192 file: F1 0.759, 24.52%, 2.21%.

### What each arm bought (paired development differences, median seeds, det+refine)

| what | F1 Δ | CER all Δ, points | dialogue Δ, points |
|---|---:|---:|---:|
| `lab` − textspot's recipe (r0, `s4_c24n4`) | +0.004 (−0.019 to +0.025) | −0.24 (−2.42 to +1.81) | **−1.61 (−3.62 to −0.13)** |
| `data` − `lab` (24,520 more real screenshots) | +0.005 (−0.023 to +0.036) | +0.97 (−1.31 to +3.31) | −0.13 (−1.17 to +0.81) |
| `ema` − `data` | −0.002 (−0.028 to +0.022) | −0.59 (−3.15 to +1.88) | +1.35 (−0.10 to +3.08) |
| `cls` − `data` (class weights 2 on dialogue / name / choice) | −0.018 (−0.038 to +0.001) | +0.53 (−1.81 to +3.30) | **+2.78 (+0.25 to +6.77)** worse |
| `long` − `ema` (60,000 steps) | +0.012 (−0.012 to +0.034) | +0.61 (−2.04 to +3.56) | −0.08 (−2.44 to +3.32) |
| `s4p_c10n4` − `s4_c24n4` (`data`) | −0.009 (−0.038 to +0.022) | +1.24 (−1.12 to +3.72) | **−0.70 (−1.49 to −0.05)** |
| `s4p_c6n4` − `s4_c24n4` (`data`) | −0.017 (−0.045 to +0.011) | **+5.27 (+2.66 to +8.18)** worse | +0.00 (−0.77 to +0.70) |
| `s4p_c8` − `s4_c24n4` (`data`) | **−0.061 (−0.089 to −0.032)** worse | **+2.33 (+0.23 to +4.56)** worse | +1.20 (−0.08 to +2.83) |
| `kd` `s4p_c6n4` − `long` `s4p_c6n4` (its best other arm) | −0.011 (−0.031 to +0.008) | −0.43 (−3.24 to +2.12) | −1.35 (−5.18 to +0.86) |
| d2s `s8_c24n4x2` − `s4_c24n4` (`data`) | −0.005 (−0.028 to +0.022) | +0.41 (−2.79 to +3.54) | **−1.33 (−2.34 to −0.54)** |
| d2s `s8_c48n4x2` − `s4_c24n4` (`data`) | **+0.040 (+0.003 to +0.080)** | +1.00 (−1.68 to +3.80) | **−0.93 (−1.98 to −0.03)** |

In short:
- The new labels bought dialogue accuracy. With refinement, more real data, EMA, the long schedule and distillation
  bought nothing resolved (without it, the long schedule bought F1 +0.022, +0.002 to +0.041). Class weighting made
  dialogue worse.
- Plain 3×3 trunks at 6-8 channels lose all-text CER.
- The d2s head buys dialogue at either size, and F1 at 48 channels.
- `s4p_c12` and `s4_c32n3` resolved nothing against `s4_c24n4`. Every arm's other decode is in the record.

### Seeds

Seeds of one arm still differ by several points. The rule read the median seed, never the best:
- `ema` det+refine: 32.19 / 34.74 / 35.98% all-text CER (median 34.74);
- `lab` det+refine: 33.65 / 34.36 / 36.32;
- d2s `s8_c24n4x2` det+refine: 34.20 / 35.75 / 38.49, with dialogue 1.13-1.66%.

The winner is itself the better of textspot's two seeds of its shape; the other reads 34.59%. A third seed of
textspot's recipe (r0) reads 36.28%.

### The speed / size front (development; est. ms at 333 MHz)

24 of the 34 points (the eligible candidates and both references) are on the front of est. ms, VTSP bytes,
arena, F1, all-text CER and dialogue CER. The main ones:

| arm | VTSP B | est. ms @333 / @444 | arena B | F1 | CER all | dialogue |
|---|---:|---:|---:|---:|---:|---:|
| `data` `s4p_c6n4` det | 2,244 | 66.1 / 49.5 | 49,895 | 0.493 | 43.31 | 4.21 |
| `data` `s4p_c6n4` det+refine | 2,244 | 73.1 / 54.8 | 72,391 | 0.527 | 40.61 | 2.46 |
| d2s `s8_c24n4x2` det | 7,416 | 73.2 / 54.9 | 79,647 | 0.537 | 39.76 | 1.93 |
| d2s `s8_c24n4x2` det+refine | 7,416 | 80.2 / 60.1 | 79,647 | 0.539 | 35.75 | 1.13 |
| textspot's fastest qualifier (`fast_ref`) | 2,732 | 80.3 / 60.2 | 72,391 | 0.533 | 34.92 | 3.64 |
| d2s `s8_c48n4x2` det+refine | 18,568 | 129.5 / 97.1 | 125,983 | 0.583 | 36.34 | 1.53 |
| `data` `s4p_c10n4` det+refine | 4,888 | 132.0 / 99.0 | 73,279 | 0.535 | 36.57 | 1.76 |
| `lab` `s4_c24n4` det+refine | 5,732 | 139.1 / 104.3 | 80,799 | 0.538 | 34.36 | 2.58 |
| **textspot's winner (`ref`)** | **5,732** | **139.1 / 104.3** | **80,799** | **0.542** | **32.04** | **3.01** |

- Time binds long before weights: the largest blob, 18,568 B, is 7% of the 256 KB weight budget, at est. 130 ms.
- Every arena is within the always-safe 131,072 B.
- Code: the d2s runtime's own `.text` is 280-408 B larger than `vocr_textspot.c`'s (VitaSDK; `reports/textspot2-runtime.md`).

### Training and scoring

- **Labels and data** (`synth/textspot2_labels.py`, `data/manifests/textspot2_shards_v1.json`): meiki boxes on
  52,785 real canvases (amendments 2-3).
  - On textspot's 28,265 canvases it added 8,147 positives to the 149,370 kept.
  - The 24,520 new screenshots brought 142,469 positives.
  - Low-score, furigana and winner-only boxes became ignore areas.
- **Runs**: 38 in stage 1 (10:51-20:01Z, 6 at once) and 12 in stage 2 (20:11-23:34Z, 7 at once), every one to its
  last step, with 0 crashes and 0 restarts. Every run records its pinned commit (`1163bf2` / `b1d7cc0`) with no
  modified file (`eval/textspot2_train_runs_20261020.json`, `eval/textspot2_train_runs_stage2_20261020.json`).
- **Scoring**: all 50 runs, both decodes. The C runtime equals the twin on all 188 frames for every arm,
  including the d2s blobs through `runtime/vocr_textspot2.c`.
- **A box problem, without effect on any result**: by about 22:45Z the box's `/tmp` had no free inodes (another
  workload's files).
  - Three scoring rounds (22:51, 22:57 and 23:39Z) stopped in the reader build, before writing anything: the
    pinned build functions give gcc a fixed environment without `TMPDIR`. Later rounds scored their four runs, and
    an integrity check over all 104 arm directories passed.
  - The local CI ran inside a private tmpfs `/tmp` (`unshare -Urm`). With `TMPDIR` alone, gcc in
    `eval/vocr3_report.py`'s fixed build environment still used the shared `/tmp`, and 12 unrelated tests failed.
- **Arm c's pool was drawn before amendment 2.** The pool was made at 09:27Z and its cache jobs started at 09:28:35Z;
  amendment 2 is dated 09:28:44Z. Both steps are rule-independent and read no development frame. The pool's rules
  (caps, hygiene, the `-TEST` folders) are the labels helper's readings 7-9, written at 09:15Z and adopted unchanged
  by amendment 2.

### Measured and estimated

- **Measured on the host**: every F1, CER, interval, recall and count above (C runtimes, frozen metrics), and blob
  and arena sizes (exact).
- **Estimated**: every console time (the A9 cycle model's high bound × 1.0007 at 333 MHz, plus 1.4 ms per refined
  line; 444 MHz is that × 333/444).
- **Not measured**: nothing has run on a console, and the protected sets were not read.

### Limits

- **One development set** of 81 VNs, built from MeikiOCR's boxes; UI is 59% of its references (519 of 885).
  A spotter that drops UI text gains little on this set.
- **Stage 2's development intervals are optimistic**: its choices read stage 1's development scores (amendment
  5). The d2s dialogue gain has no test-set check, because nothing was selected.
- **Label noise**: in the meiki-labelled real half, long thin vertical strips (card edges, UI bars) were taken as
  vertical text, mostly at reduced weight.
- **Primary metrics**: the protocol keeps textspot's UI-inclusive metrics primary. A spotter for dialogue first
  (Rubi's scan screen starts on the dialogue block) would be chosen by another rule. That would need a new
  pre-registered protocol and the protected sets read for it.

### Reproduce

```sh
H=$HERD; T2=$H/shared/artifacts/textspot2
# labels and shards (arms b and c); every source of each arm, then the manifest
.venv/bin/python eval/textspot.py cbuild --out $T2/labels/cbuild                 # the C drivers cache needs (the winner's boxes)
.venv/bin/python synth/textspot2_labels.py cache --arm lab --source realplus      # and teacher, devnative, owner
.venv/bin/python synth/textspot2_labels.py build --arm lab --source realplus      # the same sources
.venv/bin/python synth/textspot2_labels.py pool                                   # arm c's new screenshots
.venv/bin/python synth/textspot2_labels.py cache --arm more --source realplus     # and teacher, owner; then build
.venv/bin/python synth/textspot2_labels.py reserve --pixels                       # reserve_check --require-hashes --pixels
.venv/bin/python synth/textspot2_labels.py leak
.venv/bin/python synth/textspot2_labels.py manifest                               # data/manifests/textspot2_shards_v1.json
# training, from detached worktrees at 1163bf2 (stage 1) and b1d7cc0 (stage 2)
MAX_PAR=6 eval/reproducibility/textspot2-20261020/sweep.sh $T2/runs eval/reproducibility/textspot2-20261020/stage1_plan.jsonl
MAX_PAR=7 eval/reproducibility/textspot2-20261020/sweep.sh $T2/runs eval/reproducibility/textspot2-20261020/stage2_plan.jsonl
# scoring (the reading cache is seeded inside run)
.venv/bin/python eval/textspot2.py cbuild --out $T2/eval/cbuild
.venv/bin/python eval/textspot2.py fetch-teacher                                  # the 320x192 file, sha256-checked
.venv/bin/python eval/textspot2.py ceiling --out $T2/eval/ceiling --cbin $T2/eval/cbuild
.venv/bin/python eval/textspot2.py run --runs $T2/runs --out $T2/eval/units --cbin $T2/eval/cbuild --models N
.venv/bin/python eval/textspot2.py select --units $T2/eval/units --a9 eval/textspot2_a9_20261020.json \
    --a9-d2s eval/textspot2_a9_d2s_20261020.json --cbin $T2/eval/cbuild --out eval/textspot2_dev_20261020.json
.venv/bin/python eval/textspot2.py tables --dev eval/textspot2_dev_20261020.json
```
