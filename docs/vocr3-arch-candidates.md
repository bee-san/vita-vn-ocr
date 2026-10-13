# VOCR3: recognizer design candidates inside the Vita budget

Written 2026-10-13 for the conductor of VOCR3 (`synth-dataset`), who owns training and selection.
This note supplies candidates and the evidence behind each; it selects nothing.

- **Ceiling** (`docs/vita-memory-budget.md`, `vita/budget_sweep.py check`): int8 W8I blob
  ≤ 1,600,000 B, W8I arena ≤ 262,144 B, and an estimated ≤ 550 ms per 640-px line on one
  Cortex-A9 core at 333 MHz (the high end of `vita/a9_cycles.py`'s range). Since #60 that is the
  budget's "preflight" tier; its "always" tier is ≤ 900,000 B, ≤ 131,072 B and ≤ 300 ms. Every
  candidate below targets the ceiling, for the largest size class; LWa4 is the one "always" idea.
- **Every console time below is an estimate** from that cycle model. Nothing here ran on a console.
- **Sizes are exact**: random-init blobs built by the runtime's exporter (`train/arch_candidates.py`),
  arenas from `export.plan_arena()`.
- **Proxy CERs are proxies**: 5,000-step runs from scratch, scored on development data only
  (teacher_val, synth2's and nvsynth's held-out shards). No protected set (vita-test, psp-test, the
  fresh native textbox test, CLANNAD) was read, trained on or scored.

## The ranked list

All fit the ceiling (main's "preflight" tier, `docs/vita-memory-budget.md` since #60) and need no
runtime change. Proxy numbers are 5,000-step runs from scratch, read through the C runtime (W8I) on
teacher_val, with raw weights (section 3). H15's two seeds read 6.13% and 5.95%. "1 MiB L2" is the
same estimate with the game holding half of the shared L2.

| # | candidate | train2.py flags | int8 blob B | W8I arena B | est. ms, 640-px line, 333 MHz: whole L2 / 1 MiB L2 | proxy, teacher_val | other evidence |
|---|---|---|---|---|---|---|---|
| 1 | **LW**, late-wide | `--chans 48,96,192,320,384` | 1,473,280 | 192,928 | 261-468 / 261-559 | 5.37% (2 seeds); LW - H15 -0.67 (VN CI -0.91 to -0.43, both seeds below 0); LW - F20 +0.10 (-0.12 to +0.31), but +0.59 (+0.39 to +0.79) with EMA weights | the conductor's VOCR3 choice (05:55Z); PaddleOCR's mobile recognizers widen the last stages (PP-LCNet; PP-OCRv6 LCNetV4 tiers) |
| 2 | **EW**, early-wide | `--chans 64,128,256,240,288` | 1,235,584 | 231,072 | 269-477 / 269-521 | 5.64% (1 seed); EW - LW s1 +0.09 (-0.23 to +0.42); EW - H15 s1 -0.49 (-0.83 to -0.14) | LW's mirror at the same MACs and the same result so far. It reads 23% fewer weight bytes per frame than LW (197 vs 257 KB), so it slows least when the game holds part of the L2; 238 KB smaller blob |
| 3 | **F20**, hard labels | `--width 2.0` | 1,558,016 | 241,056 | 311-548 / 311-688 | 5.28% (2 seeds); F20 - H15 -0.76 (-1.04 to -0.49) | sweep `w200_r` - `w150_r` -1.38 with real data; the only shape whose gain is resolved on native Vita screenshots (-0.91, -1.88 to -0.27); at the ceiling, and 25% slower with a 1 MiB L2 share |
| 4 | **--rep 2** on any shape | add `--rep 2` | same as its shape | same | same | H15 + rep 5.54% (2 seeds); - H15 -0.49 (-0.73 to -0.26, both seeds below 0) | training-only, exact fusion into the plain blob; MobileOne-S0 +1.8, PP-OCRv6 RepDWConv +0.06; trains about 1.6x slower |
| 5 | **W20b64** | `--width 2.0 --bottleneck 64` | 1,374,080 | 238,624 | 292-524 / not run | not run | 5% faster than F20; bottleneck 48 at width 1.0 measured +0.26 (CI -0.40 to +0.92, `b48_r`) |
| - | **LWa4**, FL10-class late-wide (not trained) | `--chans 32,64,128,192,256` with FL10's 3,500-entry vocab | 779,728 | 130,464 | 140-263 / not run | not run | the only shape here in main's "always" tier (≤ 900,000 B, ≤ 131,072 B, ≤ 300 ms); FL10 plus 16% MACs in the late stages. An idea for VOCR3-FL10's size class, untested |

**Dropped: `--guide nrtr`** (training-only NRTR/GTC decoder). In these runs it is harmful: GTC - H15
+3.08 points (CI +2.67 to +3.50), both seeds, every stratum. At the run's step-5,000 weights its
encoder gradient is 1.6x the CTC one and nearly orthogonal to it (cosine 0.11), and the shared
gradient clip at norm 5 scales the CTC gradient by 0.61, so CTC learns more slowly (CTC loss 0.85 vs
H15's 0.59 at step 5,000). PaddleOCR's reported +1.16 to +1.8 come from long schedules with tuned
loss weights; a lower guide weight, a separate clip or a warm start from a CTC-trained model might
change the sign, but none of that is measured here.

Not in the list, and why (section 5): a global-attention neck (needs new kernels and about 120-250 KB of
whole-line buffers, so H15 would pass the 256 KiB arena), h-swish (a new activation flag, about +2% time,
needs a retrain to measure), a blank-skipping head (approximate, needs a trained gate), int4 weights (time
binds, not bytes; ARMv7 has no int4 multiply-accumulate), anything wider than F20 (over 550 ms).

## 1. Where the budget leaves room

H15 (`--width 1.5`) uses 72% of the weight ceiling, 70% of the arena ceiling and 72% of the time
ceiling (398 of 550 ms). F20 (`--width 2.0`) is at 97%, 92% and 99.7% (548 ms). So:

- **Time binds first.** A shape wider than F20 or with more MACs than about 2.34M per frame does
  not fit, whatever its bytes. W2125b64 below (2.38M MACs, 1.50 MB) is over at 567 ms.
- **Bytes bind second.** Between H15 and F20 there are 400 KB of weights to spend.
- **The arena is not binding** below F20's width: it grows with the widest stage, not with depth or
  with the head.
- **The L2 share matters more for late-wide shapes.** With the game holding half of the 2 MiB L2
  (`budget_sweep.py a9 --l2 1024`), the high bound rises 10% for H15, 9% for EW, 19% for LW and 25%
  for F20 (section 2's ranked times). What decides it is the weight bytes the runtime reads per
  frame: 177 KB (H15), 197 KB (EW), 257 KB (LW), 278 KB (F20). Width in the 8-row and 4-row stages
  reuses each weight over 8 or 4 rows; width in the 2-row stage and the frame row does not.

What one more MAC costs depends little on where it sits (`docs/vita-memory-budget.md`: 0.55-1.06
cycles per MAC from width 1.5 to 2.0). The cost table in section 2 measures that for every shape.

## 2. Shapes inside the ceiling (sizes measured, times estimated)

All use only the runtime's existing ops (section 5) and input height 32; LWa4 has FL10's 3,501
classes, every other shape 5,355. Measured with `train/arch_candidates.py blobs` and
`vita/budget_sweep.py check` (main's runtime, unchanged since `eab1ab8`; 40 synth-val lines, 2 MiB
L2) at 2026-10-13T01:06-01:09Z, EW at 05:03-05:06Z and LWa4 at 07:30Z (with main's tiered `check`).
H15 and F20 reproduce the released blobs' sizes and `docs/vita-memory-budget.md`'s estimates
exactly (219-398 and 311-548 ms). "tier" is the tier main's `vita/budget_sweep.py check` prints for
each blob since #60 (`arch_candidates.py table` prints "within the ceiling"; the column was added
from `check`'s output). LWa4 is not in `arch_candidates.py`'s list: its blob was built with
`random_model` directly (Reproduce) and uses budget_sweep's 3,500-character stand-in vocab, so with
FL10's own vocab its blob differs by some tens of bytes (FL10 657,664 B against its random-init
shape's 657,616 B).

| candidate | train2.py flags | params | int8 blob B | of which head B | W8I arena B | MACs/frame | est. ms per 640-px line, 333 MHz | 444 MHz | tier |
|---|---|---|---|---|---|---|---|---|---|
| H15 | `--width 1.5` | 1,038,507 | 1,155,968 | 557,440 | 182,944 | 1,564,992 | 219-398 | 164-299 | preflight |
| F20 | `--width 2.0` | 1,423,499 | 1,558,016 | 557,440 | 241,056 | 2,339,232 | 311-548 | 234-411 | preflight |
| LW | `--chans 48,96,192,320,384` | 1,346,827 | 1,473,280 | 557,440 | 192,928 | 1,933,472 | 261-468 | 196-351 | preflight |
| LWb64 | `--chans 48,96,192,320,384 --bottleneck 64` | 1,163,179 | 1,289,344 | 385,920 | 190,496 | 1,749,824 | 242-444 | 181-333 | preflight |
| LW224 | `--chans 48,96,224,320,384` | 1,374,155 | 1,502,976 | 557,440 | 199,072 | 2,021,216 | 271-484 | 203-363 | preflight |
| LW256 | `--chans 48,96,256,320,384` | 1,403,531 | 1,534,720 | 557,440 | 205,216 | 2,117,152 | 282-500 | 211-375 | preflight |
| LW384b80 | `--chans 48,96,192,384,384 --bottleneck 80` | 1,362,267 | 1,491,392 | 471,680 | 194,784 | 2,006,640 | 269-485 | 202-364 | preflight |
| EW | `--chans 64,128,256,240,288` | 1,110,059 | 1,235,584 | 557,440 | 231,072 | 1,960,512 | 269-477 | 201-358 | preflight |
| W175 | `--width 1.75` | 1,217,755 | 1,343,744 | 557,440 | 212,000 | 1,925,616 | 265-474 | 199-356 | preflight |
| W20b64 | `--width 2.0 --bottleneck 64` | 1,239,851 | 1,374,080 | 385,920 | 238,624 | 2,155,584 | 292-524 | 219-393 | preflight |
| W2125b64 | `--chans 64,136,272,344,408 --bottleneck 64` | 1,358,403 | 1,496,832 | 385,920 | 252,576 | 2,380,896 | 318-567 | 239-425 | never |
| LWa4 | `--chans 32,64,128,192,256 --vocab FL10_vocab_3500.txt` | 696,717 | 779,728 | 364,416 | 130,464 | 957,024 | 140-263 | 105-197 | always |

`chans` is (stem, 8-row stage, 4-row stage, 2-row stage, frame row). Where the MACs go (share of
MACs per CTC frame; "h8" is the 8-row stage):

| shape | stem + depthwise | pointwise h8 | pointwise h4 | pointwise h2 + collapse | frame row (Ctx + bottleneck) | head |
|---|---|---|---|---|---|---|
| H15 | 4.4% | 14.1% | 14.1% | 22.1% | 12.4% | 32.8% |
| F20 | 4.0% | 16.8% | 16.8% | 26.3% | 14.2% | 22.0% |
| LW | 3.7% | 11.4% | 11.4% | 29.7% | 17.2% | 26.6% |
| EW | 4.6% | 20.1% | 20.1% | 19.2% | 9.9% | 26.2% |
| W175 | 4.2% | 15.6% | 15.6% | 24.4% | 13.4% | 26.7% |

- **Where to put the width.** LW, EW and W175 cost the same, about 82-84% of F20's MACs (1.93-1.96M
  per frame, 468-477 ms against F20's 548):
  - **LW** (late-wide) keeps H15's three early stages and takes F20's 2-row stage, collapse and frame
    row: all 368K extra MACs go to the low-resolution stages that feed the 5,355-way head. Most
    channels in the last stages is the usual rule for small CNNs and what PaddleOCR's mobile
    recognizers do (PP-LCNet, arXiv:2109.15099v1; PP-OCRv6's LCNetV4 rec tiers 48/48/96/160 and
    96/96/192/384, arXiv:2606.13108v1 Table 2).
  - **EW** (early-wide) is the opposite: F20's stem, 8-row and 4-row stages, H15's 2-row stage and
    frame row.
  - **W175** is uniform width 1.75.
  The architecture sweep only ever scaled every stage at once (`docs/arch-sweep.md`), so this was
  untested. In the proxies (section 3) LW and EW are level with each other and close to F20; W175
  was not trained.
- **Bottleneck 64** cuts 171K head MACs per frame and 175-184 KB of blob (LW 1,473,280 → 1,289,344 B),
  about 5% of the time. With real data, bottleneck 48 at width 1.0 measured +0.26 points (CI -0.40
  to +0.92, `b48_r`); PaddleOCR went the other way in PP-OCRv4's recognizer, raising the same width
  from 64 to 120 (docs at v2.7.0, `PP-OCRv4_introduction.md`). It buys time cheaply but is not
  shown to be free in accuracy.

## 3. Proxy trainings (development data only)

**These are proxies.** They rank shapes and training-only add-ons under one short recipe; they
do not predict release CERs. Recipe (`train/train2.py` from scratch, every run identical except
the flags in the table and the seed):

- 5,000 steps of 96 lines, AdamW at lr 2e-3 with 500 warm-up steps and a cosine decay to 5%,
  weight decay 0.01, EMA 0.999, hard labels only (no KD, no `ps` pseudo-labels).
- Training data, all training-side: nvsynth/full train sources 000-041 (0.375), synth2 shards
  0-38 (0.1875), synth3 v2 train shards (0.1875), teacher_train (`tt`, 0.175), realdata's
  real_train (`rd`, 0.075); polarity inversion 0.2 on synthetic and 0.5 on real crops.
- Round 1: five single-thread runs in one `heavy.sh` slot, 01:10-05:02Z, on the training code of this
  branch's first commit (started from the working tree just before it was committed; pre-rebase hash
  `d6bfef7`, later `36db2de`). Round 2: six runs in three slots at once (steered at 04:55Z), 2-3
  threads per run, 05:03-07:15Z, on pre-rebase commit `2e75bd6` (adds `--guide`). Both are
  reproducible with the merged code: the A/B of section 7 between each of those revisions and this
  PR's head is identical for every flag set the proxies used. Thread counts change float summation
  order, not the recipe.
- Scoring: each run's step-5,000 weights exported to an int8 blob and read through the C runtime
  in W8I (`eval/synth3_ablation.py export` / `score`) on teacher_val's 1,651 lines, twice (as
  stored and in screen polarity, `rv` and `rvn`); CER = edit distance over both views / 34,236
  reference characters; paired bootstrap of the difference, 10,000 resamples over the 181 VNs as
  clusters, and over lines (`synth3_ablation.py compare`).
- **Raw weights, not EMA, are the primary score.** At 5,000 steps EMA 0.999 still lags: 86% of its
  weight lies on steps 3,000-5,000, while the loss is still falling, and 0.7% on the random
  initialisation (0.999^5000). For H15 seed 1 the EMA blob reads teacher_val at 7.17% and the raw one at
  6.13% (EMA - raw +1.04 points, VN CI +0.77 to +1.32), and the lag differs by shape, which
  distorts comparisons (LW - F20: +0.70 with EMA, +0.11 with raw weights). Release-length runs
  (15,000-40,000 steps) are not affected the same way; the EMA column is listed for reference. The
  comparisons were listed before any score; scoring raw weights was decided after the EMA scores
  showed the lag, so both are reported.
- The H15 seed pair measures training noise: a bootstrap interval covers line sampling only.

All eleven runs (pooled two-seed arms are named without a seed suffix):

| run | shape (train2.py flags) | seed | threads | lines/s | teacher_val CER, raw | EMA | s2v / nvv CER (EMA, fp32) |
|---|---|---|---|---|---|---|---|
| P_EW_s1 | `--chans 64,128,256,240,288` | 1 | 2 | 72.0 | 5.64% | 6.20% | 10.36% / 12.70% |
| P_F20_s1 | `--width 2.0` | 1 | 1 | 44.0 | 5.44% | 5.85% | 9.31% / 11.54% |
| P_F20_s2 | `--width 2.0` | 2 | 2 | 67.0 | 5.12% | 5.59% | 9.15% / 11.39% |
| P_H15_s1 | `--width 1.5` | 1 | 1 | 56.0 | 6.13% | 7.16% | 11.68% / 14.07% |
| P_H15_s2 | `--width 1.5` | 2 | 1 | 56.0 | 5.95% | 6.64% | 10.79% / 13.40% |
| P_H15gtc_s1 | `--width 1.5 --guide nrtr` | 1 | 3 | 69.0 | 9.07% | 9.54% | 15.30% / 17.87% |
| P_H15gtc_s2 | `--width 1.5 --guide nrtr` | 2 | 3 | 69.0 | 9.17% | 10.03% | 15.03% / 17.87% |
| P_H15rep_s1 | `--width 1.5 --rep 2` | 1 | 1 | 35.0 | 5.53% | 6.31% | 9.88% / 13.59% |
| P_H15rep_s2 | `--width 1.5 --rep 2` | 2 | 3 | 70.0 | 5.56% | 6.48% | 10.78% / 13.82% |
| P_LW_s1 | `--chans 48,96,192,320,384` | 1 | 1 | 54.0 | 5.55% | 6.55% | 10.52% / 12.65% |
| P_LW_s2 | `--chans 48,96,192,320,384` | 2 | 2 | 86.0 | 5.20% | 6.06% | 9.88% / 12.34% |

| comparison (B - A) | raw: delta, points [VN-cluster 95% CI] | line CI | VNs better / worse | EMA: delta [VN CI] |
|---|---|---|---|---|
| H15_s2 - H15_s1 | -0.18 [-0.54, +0.18] | [-0.52, +0.17] | 67 / 63 | -0.53 [-0.88, -0.19] |
| F20_s1 - H15_s1 | -0.69 [-1.05, -0.35] | [-0.99, -0.39] | 79 / 45 | -1.31 [-1.69, -0.97] |
| LW_s1 - H15_s1 | -0.58 [-0.91, -0.26] | [-0.89, -0.27] | 75 / 47 | -0.61 [-0.99, -0.23] |
| LW_s1 - F20_s1 | +0.11 [-0.19, +0.42] | [-0.19, +0.41] | 63 / 59 | +0.70 [+0.41, +1.00] |
| H15rep_s1 - H15_s1 | -0.60 [-0.93, -0.26] | [-0.90, -0.30] | 81 / 43 | -0.85 [-1.24, -0.49] |
| GTC_s1 - H15_s1 | +2.94 [+2.47, +3.46] | [+2.56, +3.33] | 14 / 130 | +2.38 [+1.97, +2.80] |
| LW - H15 | -0.67 [-0.91, -0.43] | [-0.88, -0.45] | 90 / 43 | -0.59 [-0.85, -0.35] |
| GTC - H15 | +3.08 [+2.67, +3.50] | [+2.77, +3.39] | 6 / 152 | +2.89 [+2.53, +3.27] |
| F20 - H15 | -0.76 [-1.04, -0.49] | [-0.99, -0.54] | 90 / 45 | -1.18 [-1.49, -0.89] |
| H15rep - H15 | -0.49 [-0.73, -0.26] | [-0.71, -0.27] | 87 / 49 | -0.50 [-0.77, -0.24] |
| LW - F20 | +0.10 [-0.12, +0.31] | [-0.11, +0.30] | 67 / 74 | +0.59 [+0.39, +0.79] |
| EW_s1 - H15_s1 | -0.49 [-0.83, -0.14] | [-0.80, -0.19] | 71 / 58 | -0.97 [-1.32, -0.64] |
| EW_s1 - F20_s1 | +0.20 [-0.11, +0.51] | [-0.10, +0.51] | 58 / 68 | +0.35 [+0.05, +0.64] |
| EW_s1 - LW_s1 | +0.09 [-0.23, +0.41] | [-0.21, +0.40] | 55 / 68 | -0.36 [-0.68, -0.04] |

Reading the table:

- **Width pays wherever it goes, at this length.** LW (late-wide), EW (early-wide) and F20 all beat
  H15 by about half a point or more: LW - H15 -0.67 (two seeds, CI -0.91 to -0.43), EW - H15 s1
  -0.49 (-0.83 to -0.14), F20 - H15 -0.76 (-1.04 to -0.49). LW and EW, at the same MACs, are level
  (EW - LW s1 +0.09, CI -0.23 to +0.42; with EMA weights EW is ahead, -0.36, CI -0.68 to -0.04), so
  round 1's "width pays where the head reads it" did not survive EW: the proxies cannot say where
  width is best spent, only that LW and EW buy most of F20's gain at 82-85% of its time. LW - F20 is +0.10 (CI -0.12 to +0.31) over two seeds.
- **Strata.** Most of every shape's gain is on teacher_val's PC screenshots scaled to console size
  (`kind:hires`, 1,064 lines: LW -0.77, F20 -0.88 against H15). On native captures the intervals
  are wide: native Vita (161 lines) LW -0.73 (CI -1.63 to +0.02), F20 -0.91 (-1.88 to -0.27);
  native PSP (426 lines) LW -0.32 (-0.69 to +0.02), F20 -0.36 (-0.74 to +0.05). EW is level with H15
  on both native kinds in its one seed (+0.16 and +0.14) and better on name-like lines (EW - LW
  -1.33, CI -2.46 to -0.17), but worse than LW on native Vita (+0.69, +0.30 to +0.98). One seed
  each; these are leads, not results.
- **With EMA weights the order of LW and F20 changes** (LW - F20 +0.59, CI +0.39 to +0.79): at 5,000
  steps EMA lags F20's raw weights by 0.44 points and LW's by 0.93 (H15's by 0.86, EW's by 0.56). Which reading
  predicts a release-length run is not known from these runs, so LW needs F20 as its control at
  full length.
- **Re-parameterisation holds up in its second seed:** --rep 2 - H15 -0.49 (CI -0.73 to -0.26; seeds
  -0.60 and -0.39), with the same blob, so it can be stacked on any shape.
- **The NRTR guide hurts** (+3.08, both seeds, every stratum; see the ranked list for why).
- **Training noise:** the two H15 seeds differ by -0.18 (CI -0.54 to +0.18) with raw weights and
  -0.53 with EMA weights; single-seed differences under about half a point are not resolved.
- **Hard strata stay hard.** No shape's gain is resolved on the 116 rare-kanji lines. H15's styles
  above the bar are names, coloured text, furigana and text without a box (`docs/train2.md`), and
  data, not shape, has been the larger lever so far (`docs/arch-sweep.md`).
- **Blank frames.** In the C runtime's W8I argmax over teacher_val (52,095 frames), 64.7-65.3% of
  frames are blank for H15, LW and F20, and 2.1-2.6% repeat the previous character. A head that
  skipped blank frames could save at most about two thirds of the head's work (section 5).

## 4. What the literature says transfers

Each finding with its source. "Training-only" means the deployed blob and runtime are unchanged.
The full survey, with every table read, is `$HERD/shared/artifacts/arch-research/lit/literature.md`.

| technique | evidence | for this recognizer |
|---|---|---|
| Attention guidance of CTC (GTC; NRTR head dropped at export) | PP-OCRv6 rec: CTC only 79.08 → CTC + NRTR-384 80.24 (+1.16), arXiv:2606.13108v1 Table 14; PP-OCRv3: +1.8 (74.0 → 75.8), arXiv:2206.03001v2 Table 4; SVTRv2-B: GTC +0.47 on Union14M, arXiv:2411.15858v2 Table 2; GTC: +3.1 IC15, +5.5 SVTP, AAAI 2020, arXiv:2002.01276v1 Table 2 | training-only; `--guide nrtr` here; costs training time |
| Semantic guidance (SVTRv2 SGM) | +2.28 Union14M, +5.11 occluded text, against GTC's +0.47 on the same backbone, arXiv:2411.15858v2 Table 2 | training-only; not implemented here |
| Structural re-parameterisation | MobileOne-S0 71.4 vs 69.6 without branches, arXiv:2206.04040v2 Table 5; DBB on MobileNetV1 +0.99, arXiv:2103.13425v2 Table 2; ACNet +0.78 on ResNet-18, arXiv:1908.03930v3 Table 3; PP-OCRv6 RepDWConv (3x3 + 1x1 + identity on the depthwise conv only) +0.06, arXiv:2606.13108v1 Table 12 | training-only; `--rep N` here, on the stem and depthwise convs only, which the runtime keeps in float32 (merged RepVGG weights quantise badly: RepVGG-A0 int8 72.4 → 52.2, arXiv:2212.01593v2 Table 1) |
| Width in the late stages | PP-LCNet puts 5x5 depthwise and SE only in the last blocks (+0.91 and +0.82, arXiv:2109.15099v1 Table 9); LCNetV4 rec tiers widen mostly at stages 3-4 | same ops: LW |
| Temporal context on the frame row | PP-OCRv6: dw 1x7 before the neck's attention +0.47, no neck -4.83, arXiv:2606.13108v1 Table 14; SVTR: local-only mixing 70.2 vs local + global 74.8 on IC15, arXiv:2205.00159v2 Table 3; BiLSTM +2.5 / +4.5 for +2.7M params, arXiv:1904.01906v4 Table 2 | dw 1xk up to k = 7 is in the runtime; ctx 4 / k 7 did not separate from width at 5k steps here (`docs/arch-sweep.md`); global attention or LSTM needs new kernels (section 5) |
| Distillation | Frame-level KL from a teacher with different CTC spike timing made the student worse (UniLSTM 15.3 → 17.1 WER), arXiv:1904.08311v2 Table 2; PP-OCRv6 tiny + CTC KL from a vocabulary-matched medium teacher +2.69 (71.83 → 74.52), arXiv:2606.13108v1 Table 14; mutual learning (U-DML) +1.5 on PP-OCRv3, Table 4 | train2's frame KD from PP-OCRv6 small (other architecture, 48-px input, 188 classes without mass) hurt (+0.53, `docs/train2.md`), which the spike-timing result explains. A same-geometry, same-vocabulary teacher (a wide TinyVNOCR trained by the same recipe) avoids every listed cause (untested here) |
| Data filtering | drop lines a weak model reads at confidence > 0.95 and a strong one < 0.15: +1.2, training 2 weeks → 5 days, PP-OCRv4 doc at v2.7.0 | data, not architecture; for the conductor's pools |
| Low-rank output layer | factorised final layer saves 30-50% of parameters without significant loss (Sainath et al., ICASSP 2013, doi:10.1109/ICASSP.2013.6638949); PP-OCRv6 tiny: rank 80 before the head, Japanese dropped to keep its 1.1M-parameter head small | the head is already factorised (bottleneck 96); bottleneck 64 above |
| int4 / mixed precision | ARMv7 NEON has no int4 multiply-accumulate (a new instruction is proposed in arXiv:2008.00638v1); sub-8-bit packing beats int8 by 1.5x only at 3 bits with custom kernels (ULPPACK, MLSys 2022) | saves bytes only; bytes are not what binds (section 1) |
| Attention decoders | PARSeq-S 23.8M params, SVTR-T 6.03M, CPPD +2.3M over SVTR-T (arXiv:2207.06966v1, 2205.00159v2 Table 4, 2307.12270v2 Table VI) | 4-20x the whole budget; out |

## 5. The runtime: what it has and what new ops would cost

`runtime/vocr.c` (`parse_op`) accepts, without any change:
- a first conv with one input channel, any kernel up to 7x7, stride 1 or 2 (the stem);
- depthwise convs of any kernel up to 7x7 (not only 3x3 and 1xk), stride 1 or 2 per axis;
- pointwise convs with any channel counts (multiples of 8, at most 1,024; K up to 4,096), including
  the row collapse over all remaining rows, and pointwise ops back to back (a low-rank K → r → N
  pair);
- ReLU, a residual add only from the input of a stride-1 depthwise op to the pointwise op right
  after it, the factorised head; up to 48 ops, input height up to 64.

So non-uniform widths (LW), wider depthwise kernels (5x5 or 3x5 in the late stages, as PP-LCNet
uses), more or longer Ctx blocks (k ≤ 7) and low-rank pointwise pairs need **no runtime change**.
`tests/test_arch_candidates.py` checks the last two on op lists the exporter writes: the C logits
equal `runtime/tests/vocr_ref.py`'s bit for bit in f32, W8F and W8I over a 65-frame line.
`train/model.py` builds non-uniform widths and the Ctx variants today (`--chans`, `--ctx`,
`--ctx-k`); wider depthwise kernels and low-rank pairs need a model.py knob first.

Not in the runtime, with an estimated A9 cost (cycle-model arithmetic on the shapes above, not
measured):

| op | needs | est. cost on one A9 | verdict |
|---|---|---|---|
| h-swish (or GELU) | an activation flag in the blob, the kernel epilogues, `vocr_ref.py` and the exporter | about 4 extra 128-bit NEON float ops (2 cycles each) per 4 outputs: ~2 cycles per activation, 14,400 activations per frame at H15 → 1.7-3.2% of the time (about +7 ms per 640-px line at 333 MHz) | cheap; PP-LCNet's largest single gain (+2.60 top-1, Table 9) but needs a retrain to test |
| SE / global pooling | a whole-line pooling pass | breaks the 4-frame tile streaming (the scale depends on the whole line) | not compatible with the runtime's design |
| global self-attention neck at the frame row (LightSVTR-like, D = 96, 1 layer, FFN ratio 2) | matmul, softmax, LayerNorm kernels; the whole line's features before the first output | ~120-145K extra MACs per frame (about +20-35 ms per 640-px line at 333 MHz); the 128-frame feature row plus K and V: about 120 KB in int16 or 250 KB in float, on top of the arena, so H15 (183 KB) lands at about 300 KB, over the 256 KiB ceiling unless the max width or D shrinks | the largest possible gain (no neck -4.83 in PP-OCRv6), the largest runtime change; test in PyTorch first |
| BiLSTM | LSTM kernel; the backward direction needs the whole line | 4 gates x (in + hidden) x hidden x 2 per frame, sequential, plus sigmoid/tanh | dominated by the attention neck |
| dilated temporal dw | a dilation field and a larger halo | negligible MACs | small runtime change; untested |
| blank-skipping head | a 2-class gate per frame and frame compaction in `run_head` | if about 70% of frames are blank (ASR's figure, arXiv:2305.11558v1 Table 1; not measured on our lines), ~15-23% of the time at F20-H15 widths | approximate (outputs change), needs a trained gate; buys room for width |
| NEON block-max filter in the head's argmax | 20 lines in `run_head` | measured below | exact; small win |

**The head's argmax, measured.** `run_head` scans every one of the 5,355 logits of every frame in
scalar code: 108K of H15's 1.16M instructions per frame. A NEON filter that skips the scan of an
8-class block unless its maximum could beat a frame's best (`>=`, NaN-safe, so the scan still
decides every update) was tried in a scratch worktree (not part of this PR), measured with the same
`budget_sweep.py check`:

| shape | instructions per frame | est. ms, 640 px, 333 MHz (main) | with the filter | output text on the 40 lines |
|---|---|---|---|---|
| H15 | 1,161,598 → 1,081,556 | 219-398 | 217-387 | identical |
| F20 | 1,589,644 → 1,509,187 | 311-548 | 309-537 | identical |
| LW | 1,347,693 → 1,266,774 | 261-468 | 258-456 | identical |
| W20b64 | 1,505,752 → 1,425,133 | 292-524 | 289-513 | identical |

It removes 80K instructions per frame but adds NEON-to-ARM transfers and unit switches, so the high
bound falls only 2.1-2.9% (11-12 ms). In the high bound for H15, 70% is instructions, 19% L1 misses
and 11% NEON/VFP switches.

## 6. Suggested use in the VOCR3 protocol

For the conductor, who owns the protocol; nothing here is selected.

1. **Shape for the largest size class: LW (the conductor's choice), with F20 (hard labels) as its
   control.** Over two seeds LW is level with F20 at 85% of the estimated time; the order depends on
   raw vs EMA weights, which 5,000 steps cannot settle and a release-length pair can.
2. **EW is the same bet with a different cost profile:** level with LW in one seed, 238 KB smaller,
   and the least slowed when the game holds part of the L2 (521 ms high bound with a 1 MiB share,
   against LW's 559 and F20's 688). If a second slot opens, an EW run at full length is the most
   informative extra one.
3. **VOCR3-H15 stays width 1.5** (the owner's like-for-like comparison with VOCR_H15). `--rep 2` does
   not change its blob, so it can be added without breaking the comparison: -0.49 over two seeds
   here, about 1.6x the training time.
4. **Do not use `--guide nrtr` as implemented** (+3.08 here). Re-test only with a lower weight or its
   own gradient clip.
5. **If any short run is compared, score raw weights as well as EMA** (`arch_candidates.py rawckpt`):
   at 5,000 steps EMA 0.999 misorders shapes.
6. **For VOCR3-FL10's size class**, LWa4 (FL10's early stages, wider late ones, FL10's vocab) is in the
   "always" tier at FL10 + 16% MACs; untested.
7. **Not now:** width beyond F20 (over 550 ms), int4, an attention neck (needs runtime kernels and
   arena), h-swish (needs a blob flag; worth a PyTorch-only probe first).

## 7. What changed in the code, and why the defaults are unchanged

- `train/model.py`: `TinyVNOCR(..., chans=None, rep=0)` and `forward(x, return_features=False)`;
  `RepCBR`, `conv_bn`, `fuse_rep`, `load_plain_into_rep`. `train/guide.py`: the NRTR guidance
  decoder. `train/train2.py`: `--chans`, `--rep`, `--guide`, `--guide-weight`, `--guide-dim`,
  `--guide-layers`; checkpoints record `chans` and `rep` (and `guide` / `model_rep` only when used). `train/export.py` and `eval/backends.py`
  rebuild `chans` checkpoints. `train/arch_candidates.py`: blobs, tables, `rawckpt`.
- **Defaults give exactly main's model** (`tests/test_model_default_equivalence.py`, against
  `tests/model_pre_arch_candidates.py`, a verbatim copy of main a0b4d22's `train/model.py`, sha256
  pinned): for v1, FL10 (width 1.0, 3,501 classes), H15, F20 and a depth-2 / ctx-4 / k-7 /
  bottleneck-128 shape, the same modules, state_dict keys, shapes, dtypes and initial values for a
  seed; the same forward output in eval and train mode, gradients and BatchNorm updates; f32 and int8
  blobs byte for byte equal, and equal to hashes computed with main's `model.py` and `export.py`;
  old-format checkpoints rebuild and export unchanged.
- **The released blobs re-export byte for byte** from the v0.2.0 checkpoints with this branch's
  `export.load_checkpoint`: FL10 `500e0b84…`, H15 `316ebc37…` (VOCR_H15), F20 `f8c1abe4…` (the same
  test, where the release checkpoints are on disk; it skips elsewhere).
- **train2.py with default flags trains bit for bit as main's** (run once, not a test, because the
  old driver is not in the tree): main's and this branch's `train/train2.py`, same arguments, one
  thread, on a 48-line toy shard (a streamed and a RAM source, EMA, two validation sets), from
  scratch for 6 steps, resumed to 9, then a fine-tune from that checkpoint for 4: all 1,572
  checkpoint tensors (`model`, `model_raw`, optimizer state) equal, the same metrics and log lines,
  at widths 1.0, 1.5 and 2.0. The only difference is two new checkpoint keys, `chans: None` and
  `rep: 0`. The same A/B against this branch's head is identical for the code each proxy round ran:
  round 1's (`36db2de`) with `--chans 48,96,192,320,384` and with `--rep 2` (1,572 and 2,872
  tensors), and round 2's (`2e75bd6`) with EW's `--chans`, `--guide nrtr`, `--rep 2`, `--width 1.5`
  and `--width 2.0` (checkpoint `model`, `model_raw`, `model_rep`, `guide` and optimizer tensors), so
  the proxies above are reproducible with the merged code. The audit (`audit-arch`) also loaded one
  checkpoint of each of the 24 checkpoint formats under `$HERD/shared/artifacts` (480 files) with
  this branch and got main's int8 blob for every one.
- The random-init H15 and F20 blobs of section 2 are byte-identical to `docs/vita-memory-budget.md`'s
  `w150` and `w200` blobs, built from main's code (sha256 `87fb00b5…`, `d0aa13d5…`).

## Limitations

- **Console speed is an estimate** from the cycle model; so is every "fits" verdict.
- **Proxies are short and one or two seeds** (EW one): 5,000 steps is 6-9% of a release run. Gaps between
  shapes this early can grow or shrink with training (`docs/arch-sweep.md`: the width gain held from
  5k to 15k steps). A teacher_val interval covers line sampling, not training noise; the H15 seed
  pair measures that.
- **teacher_val is mostly PC screenshots scaled down** (64% of its lines; `docs/synth3.md`), so it
  under-represents native console rendering.
- **Literature gains are on other tasks and data** (ImageNet, Chinese and English scene text); they
  set expectations, not predictions.
- The NEON argmax filter was checked on 40 synth-val lines of output text only, not with
  `runtime/tests/parity.py`, the PSP build or Vita3K. Its patch is
  `$HERD/shared/artifacts/arch-research/neon_head_argmax_filter.patch`.
- The guide diagnosis (gradient norms, cosine, clip factor) is from four batches of teacher_train at
  one checkpoint; it explains the slower CTC learning but does not test a fix.
- **Where `--chans` checkpoints load.** `train/export.py`, `train/model.from_checkpoint`,
  `eval/backends.py` (so `eval/run.py`) and every blob-based tool read them. `eval/linefind/evaluate.py`,
  `synth/ab_eval.py` and `train/restore_blob.py` still rebuild a model from `width` alone, as they
  already did for bottleneck, depth and ctx checkpoints, so they cannot load an LW checkpoint without
  a change (audit-arch, N1).

## Reproduce

```sh
export HERD=/local/home/skerraut/herd-vita-ocr
export A=$HERD/shared/artifacts
export D=$A/arch-research/cands
export VITASDK=$A/vita/vitasdk VOCR_SYNTH_VAL=$HERD/shared/data/synth/synth_00039.npz
export ARM_CC=$A/runtime/tools/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-linux-gnueabihf/bin/arm-none-linux-gnueabihf-gcc
.venv-ci/bin/python train/arch_candidates.py blobs --out $D          # random-init blobs, sizes.json
HEAVY_SLOTS=12 $HERD/heavy.sh timeout 5400 .venv-ci/bin/python vita/budget_sweep.py check $D/blobs/*_w8.vocr \
    --out $D/check --qemu $A/vita/qemu --jobs 5 < /dev/null           # the ceiling, per candidate
.venv-ci/bin/python vita/budget_sweep.py a9 --dir $D/check --l2 1024 --only H15,F20,LW,EW \
    --qemu $A/vita/qemu --jobs 4                                       # the game holding half the L2 (heavy.sh)
.venv-ci/bin/python - <<'PY'                                          # LWa4 (FL10-class late-wide, 3,501 classes)
import os, sys; sys.path[:0] = ["train", "vita", "eval"]
import budget_sweep as B, export; from arch_candidates import random_model
m = random_model(dict(chans=(32, 64, 128, 192, 256)), n_classes=3501)
blob = export.build_blob(export.model_ops(m, 32), B.vocab_of(3501), 32, int8=True)[0]
open(os.environ["D"] + "/blobs/LWa4_w8.vocr", "wb").write(blob)
PY
.venv-ci/bin/python vita/budget_sweep.py check $D/blobs/LWa4_w8.vocr --out $D/check-tiers \
    --qemu $A/vita/qemu --jobs 2                                        # prints the tier (heavy.sh)
.venv-ci/bin/python train/arch_candidates.py table --dir $D            # the cost table
# proxies (one slot per run group; development data only; see section 3 for the mixture)
NV="$A/nvsynth/full/train/nv_train_0[0-3]?_b*.npz+$A/nvsynth/full/train/nv_train_04[01]_b*.npz"
S2="$A/synth2/shards/synth_000[0-2]?.npz+$A/synth2/shards/synth_0003[0-8].npz"
S3="$A/synth3/shards/train_v2/synth3_000??.npz"
SRC=(--src "nv:0.375:$NV:stream=1,pool=3,inv=0.2" --src "s2:0.1875:$S2:stream=1,pool=3,inv=0.2"
     --src "s3:0.1875:$S3:stream=1,pool=3,inv=0.2"
     --src "tt:0.175:$A/teacher/teacher_train.npz:inv=0.5" --src "rd:0.075:$A/realdata/real_train.npz:inv=0.5")
VALS=(--val "rv:$A/teacher/teacher_val.npz" --val "rvn:$A/teacher/teacher_val.npz:native=1"
      --val "s2v:$A/synth2/shards/synth_00039.npz:n=2000" --val "nvv:$A/nvsynth/shards/nv_val.npz:n=2000"
      --select rv,rvn)
P=$A/arch-research/proxy
HEAVY_SLOTS=12 $HERD/heavy.sh timeout 43200 .venv-ci/bin/python train/train2.py --run P_LW_s1 --seed 1 --threads 1 \
    --out $P/runs --steps 5000 --batch 96 --lr 2e-3 --warmup 500 --final-lr 0.05 --wd 0.01 --ema 0.999 \
    --eval-every 2500 --chans 48,96,192,320,384 "${SRC[@]}" "${VALS[@]}" < /dev/null
#   likewise P_H15_s{1,2} (--width 1.5), P_F20_s{1,2} (--width 2.0), P_H15rep_s{1,2} (--width 1.5 --rep 2),
#   P_H15gtc_s{1,2} (--width 1.5 --guide nrtr), P_LW_s2, P_EW_s1 (--chans 64,128,256,240,288)
.venv-ci/bin/python train/arch_candidates.py rawckpt $P/runs/P_LW_s1/ckpt_last.pt raw_LW_s1.pt
.venv-ci/bin/python eval/synth3_ablation.py export --ckpt raw_LW_s1.pt --out $P/blobs/R_LW_s1_w8.vocr
.venv-ci/bin/python eval/synth3_ablation.py score --blob R_LW_s1=$P/blobs/R_LW_s1_w8.vocr \
    --dev $A/teacher/teacher_val.npz --out $P/scores/R_LW_s1.json            # through heavy.sh
.venv-ci/bin/python eval/synth3_ablation.py compare --scores $P/scores/R_*.json \
    --protocol $P/compare_spec_raw.json --out $P/compare_raw.json            # arms and pairs: section 3
.venv-ci/bin/python train/arch_candidates.py proxies --dir $P                # the proxy table
```

The run logs, checkpoints, blobs, per-line scores and comparison specs are in
`$HERD/shared/artifacts/arch-research/` (`proxy/`, `cands/`); `docs/vocr3-arch-candidates.json` holds
the cost and proxy tables as data.

```sh
# default-path equivalence (section 7)
.venv-ci/bin/python -m pytest -q tests/test_model_default_equivalence.py tests/test_arch_candidates.py
```
