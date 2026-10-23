# head2level: the two-level output head (issue #117, item 2)

Two CTC frames in three are blank, and the rest mostly read a common character. The output head is a 96-wide
matrix product with one row per class: 4,987 (VOCR5-S) or 5,631 (VOCR5-N and VOCR5-L) rows on every frame, 29-40% of
Small's and Normal's frame time and 15% of Large's (text-stats). VN dialogue uses few characters: the 1,503 most
frequent classes of the fine-tune pools' labels cover 98.6% of their characters. A **two-level head** computes the
blank, those frequent rows and a small **escape** on every frame, and the other (rare) rows only on the frames where
the escape fires.

This document records the trial: its pre-registered protocol (`eval/head2level_protocol.json`, frozen by commit
`e1db2cd` at 2026-10-23T14:06:29Z, amendment 1 at 14:40Z), what was built, the development results and the verdict.
Every millisecond is an estimate calibrated to VOCR3-H15's measured 5.0-5.2 ms per CTC frame; nothing ran on a
console.

## What is built

- **Permuted blobs.** `train/head2level.py permute` reorders the head's rows (int8 weights, scales, biases) and the
  vocab together into the frequency order: class 0 stays the blank. Every class's logit is today's bit for bit
  (checked on every frame of real crops), and the blob is an ordinary version-1 blob. Class numbers change, class
  texts do not; the LM decoder, the counter re-decode and the character boxes map classes by text.
- **The VOC2 trailer** (`runtime/vocr.c` `parse_two`, `train/head2level.py write_trailer`): a versioned section after
  the blob (and after head-gate's `VOCG` section when there is one), with its own CRC-32, bound to the base blob's
  byte count and CRC. It holds level 1's block count, the escape, and for the r48 variant the 48-wide rare block.
  A runtime without two-level support ignores it.
- **Level 1**: the head's first ceil((K + 1) / 8) 8-row blocks (K = 1,500: 1,504 rows, the blank and 1,503 classes).
- **The escape**: a 98 → 16 (ReLU) → 1 MLP on the head's input as the W8I head sees it (vocr.c `quant16`), plus level
  1's largest non-blank logit and its blank logit. Layer 1's 96 head-input weights are int8 through the head's own
  integer kernel (`mm_i16`); the rest is float32, in a fixed order. 1,856 B of trailer arrays per tier.
- **Level 2** on the frames that fire, packed into one batch: the head's own rare rows (**r96**), or the rare rows on a
  48-wide projection of the head input (**r48**: P 96 → 48, re-quantised by `quant16`, then the rows; trained by a
  short head-only distillation, below). A frame that fires gets today's logits, argmax and top-K (r96); one that does
  not gets them over the level-1 classes only.
- **Runtime switch** `vocr_opts.two_level` (W8I only; 0 = the full head, today's output). Asking for full logits
  computes every row. `vocr_extra.esc_score` and `escaped` report the escape per frame and per line.
- **The exact twin** (`train/head2level.py twolevel_exact`) reproduces the C escape scores, fired frames, argmax and
  top-K bit for bit (`tests/test_head2level.py` on a random-init model; on every frame of 104 real console-cut crops
  per tier and arm).

## How the escape and r48 were trained (training data only)

- **Ranking**: character counts in the labels of VOCR5's fine-tune pools (127,156 distinct lines, 1.58 M characters,
  3,361 distinct; `eval/vocr5_ft_sources.json` "real"); ties and unseen classes in `data/vocab_vn_6459.txt` order.
- **Escape data**: 2,034,818 training-side frames: every row of the console-cut crops of the fine-tune pools (owner,
  native and devnative GRAD crops and their refined copies) and a seeded 40,000-row sample of realplus v3 GRAD crops,
  as stored. The label is the frozen model's own: its W8I argmax lies outside level 1. No text label, no development
  or test VN (asserted).
- **Fit**: 1 VN in 5 (by hash) is the calibration split; standardised inputs, BCE with a class weight, Adam, 12
  epochs, seed 0, one fit per tier, K and variant. **Threshold**: from the exact int8 escape so that at least 99.9% of
  the calibration positives fire.
- **K_alt**: of K ∈ {500, 750, 1,000, 1,250, 1,750, 2,000, 2,500}, the fewest expected head rows per frame on the
  calibration split, each K with its own escape: **500** for every tier.
- **r48**: P and the rare rows initialised from the best rank-48 factorisation of the released rare rows in the head
  inputs' metric, then distilled to the released full head (KL over all classes) on 2.5 M cached head inputs (the
  escape's fit split and 30,000 synth4 crops), 4 epochs; then exported int8 and given its own escape on its own
  labels.

## Verdict

The pre-registered rule adopts the two-level head for **Normal** and **Large**: K = 500 (504 level-1 rows), the
released rare rows (r96), with head-gate's gate (`2Lg_Kalt_r96`). **Small** keeps today's head with the gate alone.
Every 48-wide rare block (r48) is rejected.

| tier | adopted | vs the gate alone, est. ms per nominal box at 333 MHz (measured on the pack / projected to wd) | gate + two-level vs today | dev score vs the gate (rubi) | rare-character recall loss vs today (rubi) |
|---|---|---|---|---|---|
| S | no | (K = 1,500: −17 / −42 ms; K = 500: rejected) | | | |
| N | `2Lg_Kalt_r96` | **−32 / −62 ms** (3.5 / 6.8%); real box −12 / −23 ms | −160 ms (17.4%) | +0.000 [+0.000, +0.000] | +0.06 [0.00, 0.15] points |
| L | `2Lg_Kalt_r96` | **−66 / −94 ms** (3.2 / 4.5%); real box −25 / −35 ms | −284 ms (13.7%) | +0.006 [−0.006, +0.022] | +0.09 [0.00, 0.24] points |

- **Bytes.** The VOC2 trailer is 1,984 B per blob (after head-gate's 1,792 B VOCG section); no weight changes. The W8I
  arena grows by 1,456 B (N: 184,448 B with the gate, against 182,944 B today). `vocr.o` grows from 20,187 to 24,115 B (VitaSDK gcc `-O2`, NEON; scalar
  15,195 to 19,151 B).
- **Why not Small.** K = 500 is resolved worse than today on greedy dialogue (+0.020 points [+0.003, +0.038]: the
  gate's own +0.016 [0.000, +0.033] plus the escape's +0.004). K = 1,500 passes (a) but not (b) on the cascade
  (+0.014 [−0.020, +0.051] against today) and not (d): the pack's measured saving over the gate is 2.7% of the box
  (projected 6.7%).
- **Why not K = 1,500 for N and L.** Not (d): the measured savings over the gate are 3.0% (N) and 2.1% (L) of the
  box. The projected savings clear 3%; amendment 2 made the smaller of the two decide, before any 2L + gate number
  existed.
- **Why not r48.** On frames that need a rare row, the 48-wide approximation picks another class often enough to
  cost rare characters: recall losses of 0.6-2.5 points and dev scores resolved worse (for example N `2L_K1500_r48`
  +0.036 [+0.005, +0.073] on rubi), although it agrees with the full head's argmax on 99.85-99.96% of all frames.
- **Counter contexts (#109; report only).** On the synthetic counter set the escape sometimes misses the small ヵ or
  ヶ and level 1 reads the full-size カ or ケ (S K = 1,500: 250 of 3,969 lines worse; L K = 500: 158). A Rubi line
  whose reading holds a full-size カ or ケ before 月 所 年 国 条, or whose re-decode finds a counter context, is read
  again with the full head (`vocr_extra.ungated`). With that re-read the adopted arms add 3 (N) and 1 (L) lines
  beyond the gate's own (the gate with it: 76 and 18 lines worse than today; the two-level head alone: 3 and 1).

Nothing here ran on a console. Every millisecond is an estimate: `vita/a9_layers.py` instruction counts of the VitaSDK
build under qemu, on ocr-speed's 24 development lines (pack `sub24.bin`, 1,609 frames), ×1.000676 to VOCR3-H15's
measured 4,942 µs per CTC frame at an assumed 333 MHz, no prefetch credit.

## Results

**Development data only** (VOCR5's fixed development sample: tv and dn fixed lines in both views, wd whole-dialogue
regions through the C GRAD finder with and without the refinement at three region sizes). The 'today' arm reproduces
VOCR5's selection scores exactly (greedy dev score 3.78209 / 3.34196 / 2.95890 for S / N / L). The reads of the earlier
runtime build and of the integrated one (head-gate's runtime with run_head2) are bit for bit the same (N today and N
`2L_K1500_r96`: every text, LM text, confidence and argmax frame of the 8,233 reads).

Arms per tier: 2L (two-level, no gate) and 2Lg (two-level and head-gate's gate), K = 1,500 and K_alt = 500, r96 and
r48; references today and, for 2Lg, the gate. Verdicts: "rejected" fails (a), "not adopted" fails (b), (c) or (d),
"adoptable" passes all. Head-gate adopted every tier's gate, so only the 2Lg arms are candidates. The escape's fire
share is over the frames level 1 runs on (the gate skips about 60% before it).

### Escapes (training-side calibration split)

| tier | K | level-1 rows | calibration positives | fire share (0.999 recall) | head rows per frame W |
|---|---:|---:|---:|---:|---:|
| S | 500 (K_alt) | 504 | 14,205 | 7.89% | 875 |
| S | 750 | 752 | 8,809 | 6.96% | 1,063 |
| S | 1000 | 1,008 | 5,442 | 7.08% | 1,307 |
| S | 1250 | 1,256 | 3,389 | 9.01% | 1,609 |
| S | 1500 | 1,504 | 2,105 | 6.70% | 1,754 |
| S | 1750 | 1,752 | 1,323 | 6.38% | 1,975 |
| S | 2000 | 2,008 | 804 | 13.90% | 2,439 |
| S | 2500 | 2,504 | 303 | 13.70% | 2,861 |
| N | 500 (K_alt) | 504 | 14,310 | 8.95% | 979 |
| N | 750 | 752 | 8,931 | 9.03% | 1,209 |
| N | 1000 | 1,008 | 5,524 | 13.14% | 1,632 |
| N | 1250 | 1,256 | 3,419 | 5.70% | 1,522 |
| N | 1500 | 1,504 | 2,113 | 6.68% | 1,796 |
| N | 1750 | 1,752 | 1,318 | 10.44% | 2,174 |
| N | 2000 | 2,008 | 789 | 11.68% | 2,448 |
| N | 2500 | 2,504 | 277 | 8.04% | 2,772 |
| L | 500 (K_alt) | 504 | 14,493 | 6.06% | 831 |
| L | 750 | 752 | 9,033 | 6.39% | 1,080 |
| L | 1000 | 1,008 | 5,584 | 7.45% | 1,369 |
| L | 1250 | 1,256 | 3,453 | 7.42% | 1,597 |
| L | 1500 | 1,504 | 2,133 | 5.79% | 1,759 |
| L | 1750 | 1,752 | 1,329 | 6.92% | 2,037 |
| L | 2000 | 2,008 | 799 | 25.86% | 2,962 |
| L | 2500 | 2,504 | 271 | 18.32% | 3,094 |

### r48 (the 48-wide rare block, calibration split)

| tier | K | mean KL (float) | argmax = today's (exact) | non-blank frames | rare-block bytes: r96 -> r48 | escape fire share |
|---|---:|---:|---:|---:|---:|---:|
| L | 1500 | 0.00068 | 99.960% | 99.897% | 429,312 -> 236,160 | 5.61% |
| L | 500 | 0.00187 | 99.880% | 99.709% | 533,312 -> 292,160 | 5.95% |
| N | 1500 | 0.00068 | 99.961% | 99.899% | 429,312 -> 236,160 | 6.17% |
| N | 500 | 0.00183 | 99.881% | 99.701% | 533,312 -> 292,160 | 7.69% |
| S | 1500 | 0.00060 | 99.950% | 99.872% | 362,752 -> 200,320 | 6.33% |
| S | 500 | 0.00178 | 99.852% | 99.621% | 466,752 -> 256,320 | 6.59% |

### Development (dev score in CER points, 95% VN-cluster intervals; rare recall loss in points)

| tier | arm | verdict | vs | greedy | rubi | rubi_cascade | rare recall loss (rubi) | escape fires (wd) | head rows / frame (wd) |
|---|---|---|---|---|---|---|---|---:|---:|
| S | 2L_K1500_r48 | rejected | today | +0.005 [-0.023, +0.035] | +0.028 [+0.004, +0.057] | +0.035 [+0.001, +0.074] | +1.65 [-0.61, +4.32] | 6.01% | 1,713.7 |
| S | 2L_K1500_r96 | adoptable | today | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.001 [+0.000, +0.003] | +0.05 [+0.00, +0.19] | 6.31% | 1,724.2 |
| S | 2L_Kalt_r48 | rejected | today | +0.079 [+0.032, +0.127] | +0.074 [+0.032, +0.117] | +0.051 [-0.010, +0.113] | +0.77 [+0.34, +1.24] | 5.92% | 769.5 |
| S | 2L_Kalt_r96 | adoptable | today | +0.004 [+0.000, +0.012] | +0.003 [+0.000, +0.010] | +0.005 [+0.000, +0.013] | +0.04 [+0.00, +0.12] | 7.25% | 829.2 |
| S | 2Lg_K1500_r48 | rejected | today | +0.021 [-0.014, +0.055] | +0.026 [-0.019, +0.067] | +0.047 [+0.003, +0.095] | +1.65 [-0.61, +4.32] | 15.41% | 2,041.4 |
| S | 2Lg_K1500_r48 | rejected | gate | +0.005 [-0.023, +0.035] | +0.034 [+0.010, +0.062] | +0.034 [+0.003, +0.069] | +1.65 [-0.61, +4.32] | 15.41% | 2,041.4 |
| S | 2Lg_K1500_r96 | not adopted | today | +0.016 [+0.000, +0.033] | -0.007 [-0.061, +0.035] | +0.014 [-0.020, +0.051] | +0.05 [+0.00, +0.19] | 16.16% | 2,067.7 |
| S | 2Lg_K1500_r96 | not adopted | gate | +0.000 [+0.000, +0.000] | +0.001 [+0.000, +0.003] | +0.001 [+0.000, +0.003] | +0.05 [+0.00, +0.19] | 16.16% | 2,067.7 |
| S | 2Lg_Kalt_r48 | rejected | today | +0.095 [+0.043, +0.146] | +0.065 [+0.006, +0.121] | +0.059 [-0.003, +0.123] | +0.77 [+0.34, +1.24] | 15.16% | 1,184.6 |
| S | 2Lg_Kalt_r48 | rejected | gate | +0.079 [+0.032, +0.127] | +0.073 [+0.033, +0.114] | +0.046 [-0.009, +0.101] | +0.77 [+0.34, +1.24] | 15.16% | 1,184.6 |
| S | 2Lg_Kalt_r96 | rejected | today | +0.020 [+0.003, +0.038] | -0.004 [-0.058, +0.039] | +0.018 [-0.015, +0.055] | +0.04 [+0.00, +0.12] | 18.37% | 1,328.4 |
| S | 2Lg_Kalt_r96 | rejected | gate | +0.004 [+0.000, +0.012] | +0.004 [+0.000, +0.012] | +0.005 [+0.000, +0.013] | +0.04 [+0.00, +0.12] | 18.37% | 1,328.4 |
| N | 2L_K1500_r48 | rejected | today | +0.040 [+0.009, +0.073] | +0.036 [+0.005, +0.073] | +0.022 [-0.003, +0.047] | +2.32 [+0.39, +4.25] | 5.72% | 1,740.2 |
| N | 2L_K1500_r96 | adoptable | today | -0.004 [-0.012, +0.000] | +0.000 [+0.000, +0.000] | +0.004 [+0.000, +0.009] | +0.00 [+0.00, +0.00] | 6.38% | 1,767.4 |
| N | 2L_Kalt_r48 | rejected | today | +0.083 [+0.010, +0.161] | +0.071 [+0.012, +0.134] | +0.026 [-0.004, +0.057] | +0.74 [+0.16, +1.31] | 7.22% | 874.1 |
| N | 2L_Kalt_r96 | adoptable | today | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.003 [-0.010, +0.002] | +0.00 [+0.00, +0.00] | 8.37% | 933.1 |
| N | 2Lg_K1500_r48 | rejected | today | +0.055 [+0.014, +0.102] | +0.021 [-0.020, +0.067] | +0.009 [-0.021, +0.040] | +2.17 [+0.53, +3.98] | 14.58% | 2,106.0 |
| N | 2Lg_K1500_r48 | rejected | gate | +0.040 [+0.009, +0.073] | +0.031 [+0.005, +0.062] | +0.021 [-0.004, +0.046] | +1.88 [+0.31, +3.77] | 14.58% | 2,106.0 |
| N | 2Lg_K1500_r96 | not adopted | today | +0.011 [-0.008, +0.034] | -0.009 [-0.040, +0.023] | -0.008 [-0.028, +0.012] | +0.30 [+0.00, +0.68] | 16.13% | 2,169.7 |
| N | 2Lg_K1500_r96 | not adopted | gate | -0.004 [-0.012, +0.000] | +0.000 [+0.000, +0.000] | +0.004 [+0.000, +0.009] | +0.00 [+0.00, +0.00] | 16.13% | 2,169.7 |
| N | 2Lg_Kalt_r48 | rejected | today | +0.093 [+0.012, +0.182] | +0.053 [-0.005, +0.112] | +0.020 [-0.016, +0.055] | +0.75 [+0.17, +1.30] | 18.15% | 1,434.7 |
| N | 2Lg_Kalt_r48 | rejected | gate | +0.077 [+0.008, +0.148] | +0.062 [+0.008, +0.119] | +0.032 [-0.001, +0.064] | +0.69 [+0.14, +1.23] | 18.15% | 1,434.7 |
| N | 2Lg_Kalt_r96 | adoptable | today | +0.015 [-0.001, +0.037] | -0.009 [-0.040, +0.023] | -0.014 [-0.035, +0.007] | +0.06 [+0.00, +0.15] | 20.77% | 1,569.0 |
| N | 2Lg_Kalt_r96 | adoptable | gate | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | -0.002 [-0.009, +0.003] | +0.00 [+0.00, +0.00] | 20.77% | 1,569.0 |
| L | 2L_K1500_r48 | rejected | today | +0.058 [+0.025, +0.096] | +0.032 [+0.013, +0.054] | +0.032 [+0.013, +0.054] | +1.45 [+0.58, +2.40] | 5.36% | 1,725.4 |
| L | 2L_K1500_r96 | adoptable | today | +0.003 [+0.000, +0.007] | +0.004 [+0.000, +0.010] | +0.004 [+0.000, +0.010] | +0.10 [+0.00, +0.38] | 5.61% | 1,735.5 |
| L | 2L_Kalt_r48 | rejected | today | +0.142 [+0.089, +0.200] | +0.064 [+0.032, +0.098] | +0.064 [+0.032, +0.098] | +0.65 [+0.36, +0.95] | 5.45% | 783.4 |
| L | 2L_Kalt_r96 | adoptable | today | +0.011 [-0.004, +0.028] | +0.005 [-0.007, +0.021] | +0.005 [-0.007, +0.021] | +0.09 [+0.00, +0.24] | 5.56% | 789.2 |
| L | 2Lg_K1500_r48 | rejected | today | +0.063 [+0.028, +0.101] | +0.031 [+0.008, +0.056] | +0.031 [+0.008, +0.056] | +1.45 [+0.58, +2.40] | 12.53% | 2,021.4 |
| L | 2Lg_K1500_r48 | rejected | gate | +0.058 [+0.025, +0.096] | +0.032 [+0.013, +0.054] | +0.032 [+0.013, +0.054] | +1.45 [+0.58, +2.40] | 12.53% | 2,021.4 |
| L | 2Lg_K1500_r96 | not adopted | today | +0.007 [-0.001, +0.017] | +0.002 [-0.013, +0.016] | +0.002 [-0.013, +0.016] | +0.10 [+0.00, +0.38] | 13.06% | 2,043.2 |
| L | 2Lg_K1500_r96 | not adopted | gate | +0.003 [+0.000, +0.007] | +0.004 [+0.000, +0.010] | +0.004 [+0.000, +0.010] | +0.10 [+0.00, +0.38] | 13.06% | 2,043.2 |
| L | 2Lg_Kalt_r48 | rejected | today | +0.147 [+0.092, +0.205] | +0.069 [+0.036, +0.104] | +0.069 [+0.036, +0.104] | +0.65 [+0.36, +0.95] | 12.68% | 1,154.1 |
| L | 2Lg_Kalt_r48 | rejected | gate | +0.142 [+0.089, +0.200] | +0.071 [+0.036, +0.106] | +0.071 [+0.036, +0.106] | +0.65 [+0.36, +0.95] | 12.68% | 1,154.1 |
| L | 2Lg_Kalt_r96 | adoptable | today | +0.015 [-0.002, +0.034] | +0.004 [-0.016, +0.026] | +0.004 [-0.016, +0.026] | +0.09 [+0.00, +0.24] | 12.90% | 1,165.8 |
| L | 2Lg_Kalt_r96 | adoptable | gate | +0.011 [-0.004, +0.028] | +0.006 [-0.006, +0.022] | +0.006 [-0.006, +0.022] | +0.09 [+0.00, +0.24] | 12.90% | 1,165.8 |

### Time (est., one Cortex-A9 core at 333 MHz, no prefetch credit; nominal box 240 frames + 0.1 s)

| tier | arm | ms / frame | saved per nominal box vs its reference: measured / projected | % of box | real box (89 frames) | vs today | 444 MHz | (d) |
|---|---|---:|---|---:|---:|---:|---:|---|
| S | today | 2.192 | | | | | | |
| S | gate | 1.743 | 108 (vs today) | 17.2% | 40 | | 81 | |
| S | 2L_K1500_r48 | 1.758 | 104 / 128 (vs today) | 16.6% / 20.4% | 39 / 47 | 104 | 78 | yes |
| S | 2L_K1500_r96 | 1.865 | 79 / 124 (vs today) | 12.6% / 19.9% | 29 / 46 | 79 | 59 | yes |
| S | 2L_K500_r48 | 1.609 | 140 / 172 (vs today) | 22.4% / 27.5% | 52 / 64 | 140 | 105 | yes |
| S | 2L_K500_r96 | 1.739 | 109 / 166 (vs today) | 17.4% / 26.4% | 40 / 61 | 109 | 82 | yes |
| S | 2Lg_K1500_r48 | 1.583 | 38 / 45 (vs gate) | 6.1% / 7.2% | 14 / 17 | 146 | 29 | yes |
| S | 2Lg_K1500_r96 | 1.673 | 17 / 42 (vs gate) | 2.7% / 6.7% | 6 / 16 | 125 | 13 | no |
| S | 2Lg_K500_r48 | 1.524 | 53 / 61 (vs gate) | 8.4% / 9.8% | 20 / 23 | 160 | 39 | yes |
| S | 2Lg_K500_r96 | 1.637 | 26 / 55 (vs gate) | 4.1% / 8.8% | 9 / 20 | 133 | 19 | yes |
| N | today | 3.409 | | | | | | |
| N | gate | 2.878 | 127 (vs today) | 13.9% | 47 | | 96 | |
| N | 2L_K1500_r48 | 2.902 | 122 / 154 (vs today) | 13.2% / 16.7% | 45 / 57 | 122 | 91 | yes |
| N | 2L_K1500_r96 | 2.994 | 100 / 149 (vs today) | 10.8% / 16.2% | 37 / 55 | 100 | 75 | yes |
| N | 2L_K500_r48 | 2.768 | 154 / 196 (vs today) | 16.8% / 21.3% | 57 / 72 | 154 | 115 | yes |
| N | 2L_K500_r96 | 2.895 | 123 / 188 (vs today) | 13.4% / 20.4% | 46 / 70 | 123 | 92 | yes |
| N | 2Lg_K1500_r48 | 2.704 | 42 / 56 (vs gate) | 4.5% / 6.0% | 16 / 21 | 169 | 31 | yes |
| N | 2Lg_K1500_r96 | 2.765 | 27 / 51 (vs gate) | 3.0% / 5.6% | 10 / 19 | 154 | 20 | no |
| N | 2Lg_K500_r48 | 2.659 | 52 / 69 (vs gate) | 5.7% / 7.6% | 20 / 26 | 180 | 39 | yes |
| N | 2Lg_K500_r96 | 2.743 | 32 / 62 (vs gate) | 3.5% / 6.8% | 12 / 23 | 160 | 24 | yes |
| L | today | 8.220 | | | | | | |
| L | gate | 7.312 | 218 (vs today) | 10.5% | 81 | | 164 | |
| L | 2L_K1500_r48 | 7.546 | 162 / 197 (vs today) | 7.8% / 9.5% | 60 / 73 | 162 | 121 | yes |
| L | 2L_K1500_r96 | 7.658 | 135 / 192 (vs today) | 6.5% / 9.3% | 50 / 71 | 135 | 101 | yes |
| L | 2L_K500_r48 | 7.290 | 223 / 251 (vs today) | 10.8% / 12.1% | 83 / 93 | 223 | 167 | yes |
| L | 2L_K500_r96 | 7.405 | 196 / 245 (vs today) | 9.4% / 11.8% | 73 / 91 | 196 | 147 | yes |
| L | 2Lg_K1500_r48 | 7.090 | 53 / 78 (vs gate) | 2.6% / 3.8% | 20 / 29 | 271 | 40 | no |
| L | 2Lg_K1500_r96 | 7.130 | 44 / 74 (vs gate) | 2.1% / 3.5% | 16 / 27 | 262 | 33 | no |
| L | 2Lg_K500_r48 | 7.008 | 73 / 100 (vs gate) | 3.5% / 4.8% | 27 / 37 | 291 | 55 | yes |
| L | 2Lg_K500_r96 | 7.035 | 66 / 94 (vs gate) | 3.2% / 4.5% | 25 / 35 | 284 | 50 | yes |

### Selection

- S: candidates 2Lg_K1500_r48, 2Lg_K1500_r96, 2Lg_Kalt_r48, 2Lg_Kalt_r96; adoptable none; selected **none (today)**
- N: candidates 2Lg_K1500_r48, 2Lg_K1500_r96, 2Lg_Kalt_r48, 2Lg_Kalt_r96; adoptable 2Lg_Kalt_r96; selected **2Lg_Kalt_r96**
- L: candidates 2Lg_K1500_r48, 2Lg_K1500_r96, 2Lg_Kalt_r48, 2Lg_Kalt_r96; adoptable 2Lg_Kalt_r96; selected **2Lg_Kalt_r96**

### Counter-context set (#109, synthetic, report only)

| tier | arm | greedy CER (lines worse) | rubi CER (lines worse) | with the re-read (lines worse) | re-read lines |
|---|---|---:|---:|---:|---:|
| S | today | 4.81% (0) | 3.45% (0) | 3.45% (0) | 0 |
| S | gate | 4.82% (5) | 3.47% (14) | 3.47% (14) | 735 |
| S | 2L_K1500_r96 | 5.31% (250) | 4.01% (253) | 3.45% (2) | 948 |
| S | 2Lg_K1500_r96 | 5.32% (255) | 4.04% (267) | 3.48% (16) | 948 |
| S | 2L_K1500_r48 | 5.47% (344) | 4.13% (317) | 3.47% (16) | 1006 |
| S | 2Lg_K1500_r48 | 5.48% (348) | 4.16% (329) | 3.49% (28) | 1006 |
| S | 2L_Kalt_r96 | 4.81% (0) | 3.45% (2) | 3.45% (0) | 735 |
| S | 2Lg_Kalt_r96 | 4.82% (5) | 3.48% (16) | 3.47% (14) | 735 |
| S | 2L_Kalt_r48 | 5.01% (144) | 3.62% (116) | 3.46% (29) | 822 |
| S | 2Lg_Kalt_r48 | 5.02% (149) | 3.64% (129) | 3.48% (42) | 822 |
| N | today | 5.18% (0) | 3.60% (0) | 3.60% (0) | 0 |
| N | gate | 5.23% (27) | 3.78% (96) | 3.74% (76) | 666 |
| N | 2L_K1500_r96 | 5.20% (7) | 3.62% (7) | 3.60% (1) | 680 |
| N | 2Lg_K1500_r96 | 5.24% (34) | 3.79% (103) | 3.74% (77) | 669 |
| N | 2L_K1500_r48 | 5.44% (140) | 3.83% (116) | 3.64% (20) | 755 |
| N | 2Lg_K1500_r48 | 5.49% (166) | 4.01% (212) | 3.78% (96) | 744 |
| N | 2L_Kalt_r96 | 5.19% (2) | 3.61% (3) | 3.61% (3) | 677 |
| N | 2Lg_Kalt_r96 | 5.23% (29) | 3.79% (99) | 3.75% (79) | 666 |
| N | 2L_Kalt_r48 | 5.41% (124) | 3.77% (93) | 3.65% (31) | 734 |
| N | 2Lg_Kalt_r48 | 5.46% (151) | 3.95% (189) | 3.79% (107) | 724 |
| L | today | 4.29% (0) | 2.98% (0) | 2.98% (0) | 0 |
| L | gate | 4.30% (5) | 3.03% (32) | 3.00% (18) | 624 |
| L | 2L_K1500_r96 | 4.29% (1) | 2.98% (2) | 2.98% (1) | 625 |
| L | 2Lg_K1500_r96 | 4.30% (6) | 3.03% (34) | 3.01% (19) | 624 |
| L | 2L_K1500_r48 | 4.46% (89) | 3.11% (71) | 2.99% (14) | 658 |
| L | 2Lg_K1500_r48 | 4.47% (94) | 3.16% (102) | 3.02% (32) | 657 |
| L | 2L_Kalt_r96 | 4.60% (159) | 3.43% (158) | 2.98% (1) | 698 |
| L | 2Lg_Kalt_r96 | 4.61% (164) | 3.48% (189) | 3.01% (19) | 697 |
| L | 2L_Kalt_r48 | 4.59% (169) | 3.28% (140) | 3.01% (33) | 689 |
| L | 2Lg_Kalt_r48 | 4.60% (174) | 3.33% (168) | 3.03% (49) | 688 |

## Protected sets (once, after the verdict; report only)

Read once on 2026-10-23 (attempt records 17:48Z, after the verdict was pushed at `c5f7dae`), for the adopted arms only:
N and L, today (the release blob) against `2Lg_Kalt_r96` (the gate and the two-level head). They decide nothing.
Records: `eval/head2level_protected_20261023.json` (tests) and `eval/head2level_clannad_20261023.json` (counts, rates
and hashes only).

| set (dialogue) | decode | N: today → gate + two-level (lines changed) | the gate alone | L: today → gate + two-level | the gate alone |
|---|---|---|---|---|---|
| vita-test + psp-test (451 lines, console polarity) | greedy | 1.129% → 1.129% (0) | 1.129% (0) | 0.788% → 0.788% (0) | 0.788% (0) |
| | rubi | 0.725% → 0.682% (2) | 0.682% (2) | 0.533% → 0.511% (1) | 0.511% (1) |
| | rubi_cascade | 0.618% → 0.575% (2) | 0.575% (2) | 0.533% → 0.511% (1) | 0.511% (1) |
| CLANNAD (3,048 lines, 33,485 characters, page bootstrap) | greedy | 0.923% → 0.923% (0) | 0.923% (0) | 0.630% → 0.639% (3) | 0.630% (0) |
| | rubi | 0.765% → 0.768% (1) | 0.768% (1) | 0.547% → 0.559% (5), +0.012 [+0.003, +0.024] | 0.550% (1) |
| | rubi with the re-read | 0.765% → 0.768% (1) | 0.768% (1) | 0.547% → 0.553% (3), +0.006 [0.000, +0.015] | 0.550% (1) |
| | rubi_cascade | 0.654% → 0.663% (3) | 0.660% (2) | 0.547% → 0.559% (5) | 0.550% (1) |

On the test sets the two-level head changes no line beyond the gate's. On CLANNAD it changes none for N and adds 3-4
characters for L (3 to 5 dialogue lines; 2 with the re-read), +0.003 to +0.009 points beyond the gate's; the gate
itself improves CLANNAD's whole text (names included) a lot (N rubi 2.379% → 1.799%), which the two-level head leaves
as it is (N) or nearly (L 1.804% against the gate's 1.796%).

## Limits

- **Estimates.** Times are qemu instruction counts with a cache model, calibrated on one model (H15) in one game at an
  assumed clock. The pack's 24 lines fire the escape on 26-29% of their frames, against 5-8% of wd's (no gate), so
  the measured savings are the pessimistic ones; the projections use wd's rates.
- **Development data.** 1,098 tv + dn dialogue rows of 94 VNs and wd's 1,515 region cases; rare characters (outside
  level 1) are 1,618 / 1,144 / 5,142 reference characters at K = 500 and 332 / 166 / 1,008 at K = 1,500 (tv, dn, wd).
- **The escape's training frames** are VOCR5's own fine-tune pools (console-cut crops and a realplus sample), 2.0 M
  frames of 12,220 VNs; the development check is the guard against their being easier than unseen games.
- **The counter set** is synthetic (fonts over noisy boxes); no development line holds a counter context.
