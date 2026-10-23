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

<!-- RESULTS -->
