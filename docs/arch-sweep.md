# Architecture sweep: how small, and how big, should the recognizer be?

<!-- BEGIN summary -->
**Recommendation for train2 (reviewed and re-scored 2026-10-10, after the long runs finished): width 2.0, v1's layout otherwise, int8 weights. Width 1.5 is the lighter fallback if the console speed estimate (below) is too slow. PSP-lean: v1's width 1.0.**

| | train2 (width 2.0) | lighter (width 1.5) | PSP-lean (width 1.0) |
|---|---|---|---|
| config | `--width 2.0`, bottleneck 96, height 32, 2 Ctx (k 5), depth 1, full vocab | `--width 1.5`, the rest the same | v1's size, `--width 1.0` (optionally `--vocab data/vocab_3500.txt`) |
| params | 1,423,499 | 1,041,915 | 761,771 (581,933 with 3,500) |
| int8 blob (runtime's exporter) | 1,558,016 B | 1,155,968 B | 859,904 B (657,664 B) |
| arena, `vocr_arena_bytes()` W8F / W8I | 228,512 / 241,056 B | 173,472 / 182,944 B | 118,432 / 124,832 B |
| MMAC, median / p90 real Vita line (299 / 608 px) | 89 / 178 | 59 / 119 | 38 / 76 (31 / 63) |
| **est.** Cortex-A9 444 MHz, one core, p90 Vita line | **593-741 ms** (estimate) | 396-496 ms | 254-318 ms (209-261 ms) |
| real CER at 5,000 steps with real data, int8 (vita-test-inv + psp-test-inv) | **5.95%** | 7.33% | 8.99% |
| as-is vita-test / psp-test CER, int8 (exact lines) | 6.28% / 15.70% (63.6% / 44.1%) | 7.33% / 17.09% (54.9% / 42.0%) | 10.76% / 20.80% (44.2% / 33.5%) |

Width 2.0 was still training when this PR was first opened; its run (`w200_r`) and the decisive 15,000-step pair have since finished and been re-scored, so the recommendation moves from width 1.5 to width 2.0 (the PR's stated condition: width 2.0 beats width 1.5 by more than a point with its interval clear of zero, and is within the speed budget).

Why:

- **Width is the only knob that measurably helps once real data is in the mix, and it keeps paying up to width 2.0.** With realdata's 9,811 real lines mixed in (`--real real_train.npz --real-weight 10`, 5,000 steps):
  - width 1.5 beats width 1.0 by **-1.66 points** of pooled real CER (95% bootstrap CI -2.32 to -1.00);
  - width 2.0 beats width 1.5 by a further **-1.38 points** (CI -2.01 to -0.76), and width 1.0 by -3.04 (CI -3.79 to -2.33). As-is vita-test 6.28%, psp-test 15.70%. It is better than width 1.5 on every style of both sets except vita-test furigana (3 lines) and the two coloured-name ties.
  - Both gains are beyond the 0.98-point seed-to-seed spread of the baseline.
- **The 15,000-step pair confirms width 1.5 > 1.0 at 3× the budget**, so the width gain is not a short-run artefact: width 1.5 is -1.50 points (CI -2.10 to -0.89), and ahead at every checkpoint (pooled vita-test-inv + psp-test-inv: 7.80% vs 10.21% at 5k, 5.64% vs 7.11% at 10k, 4.59% vs 6.09% at 15k). I did not run a 15k width 2.0; the 5k gain plus this trend is the evidence for going wider.
- On synthetic data alone, no knob beats the v1 layout by more than seed noise: width 1.5 -0.45, bottleneck 128 -0.41, ctx 4 / k 7 -0.55, depth 2 +0.41, vocab 3,500 +0.10 points. Width 1.5 still fits synth-val much better (0.85% vs 1.28%), so the synthetic-to-real gap, not capacity, limited those runs.
- **depth and ctx add nothing over plain width.** With real data, width 1.5 + depth 2 + ctx 4 (`w150d2c4_r`) ties width 1.5 (-0.12 points, CI -0.72 to +0.49) at more MACs, so plain width 2.0 is the better way to spend the budget.
- **Shrinking hurts:** width 0.75 costs +2.81 points, height 24 +3.31, width 0.5 +5.52, all with intervals clear of zero. Vocab 2,500 costs +0.79 (CI +0.10 to +1.51) and can't emit 0.43% of vita-test's characters.
- **Data and budget matter more than any architecture knob.** Mixing in real data cuts the as-is vita-test CER of the v1-size model from 28.67% to 10.76% at the same 5,000 steps. Training 6.5× longer (v1's 32k steps) cuts the synthetic-only pooled real CER from 15.46% to 9.88%. The 2.30% bar has to come from train2's data and budget: nvsynth's NVIDIA line crops (17.0M every-line train crops built so far), synth2, and teacher-labelled real crops.
- **int8 needs no QAT:** post-training per-channel int8 (W8F) moves pooled real CER by at most 0.19 points in any run (width 0.5; 7.27% fp32 vs 7.33% int8 for width 1.5 with real data). QAT from half-way (`qat_r`) does not beat v1-size PTQ (+0.08 points vs `base_r`, CI -0.33 to +0.51). Simulated W8I matches W8F to within one character error in every run, and the C runtime gives the same CERs in f32, W8F and W8I.
- **bottleneck 48 does not help with real data** (`b48_r`: +0.26 points vs `base_r`, CI -0.40 to +0.92), so the head bottleneck stays at 96.
- **Everything fits the budget easily.** Across the whole grid the int8 blob is 0.55-2.7 MB (width 3.0 is the largest) and the arena 35-349 KiB, at most about 6% of the 50 MB budget. Every width runs in all three runtime modes, W8I included. Speed is the binding constraint, and all of it is an estimate (below): a p90 Vita line is about 593-741 ms at width 2.0, 396-496 ms at width 1.5, 0.84-1.05 s at width 2.5 and 1.15-1.43 s at width 3.0. The vita agent's cycle-level estimate (PR #12, qemu instruction mix plus a cache model) is lower than this GMAC/s figure for v1 (w8i about 117-285 ms for a 640-px line), so these ms numbers are likely pessimistic; a console run is still the only real measurement.
- **Keep the full vocab** for the release model. It misses 0.04% of vita-test's characters and none of psp-test's. The 3,500 vocab misses 0.12% / 0.04% and 0.49% of the NVIDIA text, for 18% fewer MACs. To keep NVIDIA lines that contain rare symbols, `train/vocab.py build --size 0 --extra <nvsynth's OOV list>` builds the full vocab plus those symbols and `<unk>`.
- **PSP-lean:** the PSP has no NEON, so MACs matter more there, and the psp agent (PR #13) estimates a width-2.0 line at about 2.2× v1's, with v1 already near 1 s. Width 1.0 saves 64% of width 2.0's MACs for +3.04 points; the 3,500 vocab saves another 18%. With real data the 3,500 vocab is level with the full one (`v3500_r`: +0.06 points, 95% CI -0.61 to +0.73, single seed). Height 24 isn't worth its +3.3 points. The psp agent should pick between v1's width 1.0 and the 3,500 vocab on measured PSP speed.
<!-- END summary -->

## Setup

Every run uses the v1 recipe: `train/train.py` on v1's synthetic shards, the same
augmentation, AdamW at lr 2e-3 with 1,000 warm-up steps and a cosine decay to 5%. Each
run changes **one knob** and trains for a **matched budget of 5,000 steps of 96 lines**
(480k lines, 1.2 epochs of the 390k-line synthetic set) with seed 1. Each `heavy.sh`
slot (5 threads) runs two or three configs at 2 + 2 (+ 1) threads, which changes only
the wall time. v1 itself trained for 32,496
steps (8 epochs); its row is shown for scale and isn't part of the Pareto
comparison. `base_s2` repeats the baseline with seed 2 to show the seed-to-seed
noise.

Knobs (the defaults rebuild v1 exactly; `train.py --help`):

| knob | trained (v1's value in bold) | cost only | what it changes |
|---|---|---|---|
| `--width` | 0.5, 0.75, **1.0**, 1.5, 2.0; 2.0 and the 1.0-vs-1.5 pair at 15k with real data | 0.25, 2.5, 3.0 | every channel count (32/64/128/160/192 at 1.0) |
| `--depth` | **1**, 2; 2 + ctx 4 at width 1.5 with real data | 3 | stride-1 DS blocks after each of the three downsampling blocks |
| `--ctx`, `--ctx-k` | **2** blocks k **5**; 4 blocks k 7 | | temporal context blocks (depthwise 1×k + pointwise, residual) on the CTC frame row |
| `--bottleneck` | **96**, 128; 48 with real data | 64 | width of the factorised head's 1×1 bottleneck |
| `--vocab` | 2,500 and 3,500 + `<unk>`, **5,354**; 3,500 with real data | | output classes (plus the CTC blank) |
| `--height` | 24, **32** | | input line height (24 collapses 24→12→6→3→2→1 with the same layers) |
| `--qat` | with real data | | int8 weight fake quantisation over the last half of training |
| `--real` | off; realdata's `real_train.npz` at weight 10 for `*_r` runs | | mixes real training lines into every batch |

Scores (`train/sweep_eval.py`, collected by `train/sweep_table.py`):

- **vita-test** and **psp-test** are eval's hand-checked sets, cropped and scored by
  eval's own `sets.py` and `metrics.py`: vita-test (206 lines, 2,564 characters, 16 Vita
  games) and psp-test (245 lines, 2,510 characters, 14 PSP VNs) as merged on main (eval
  #2 and #5; manifest sha256s are in each run's eval JSON). On v1 this gives eval's
  published numbers exactly: vita-test 23.75% CER / 52.43% exact, psp-test 46.45% / 30.2%.
- **vita-test-inv** and **psp-test-inv** are the same lines with the dark-on-light
  ones inverted by their style tag. The v1 recipe renders only light text, so every
  v1-recipe model reads dark text at 80–95% CER whatever its size. That's a data gap,
  which synth2, NVIDIA pretraining and the real fine-tuning close. Inverting by the tag
  (an oracle) keeps it from drowning the architecture differences. Inverted, v1 reads
  dark-on-light lines at 4.9% (vita) and 5.6% (psp) CER (int8).
- **real-hand-x** is 46 lines (727 characters) that the research campaign
  transcribed by hand from 17 Vita games outside vita-test's set games, with dark lines
  inverted by their tag. Two caveats: 2 of the lines are from v3, which eval's
  `test_games.json` reserves as family of vita-test's v7721, so the set is not strictly
  independent of vita-test; and the research cut `v17287_1_2` as a 99×380 scene crop
  with about 20 px of text at the bottom, which no model reads at height 32. Neither
  line is training data (only synthetic-only runs are scored on this pool).
- **real CER**, the decision metric, pools vita-test-inv, psp-test-inv and real-hand-x:
  5,801 characters from 45 games (vita-test and psp-test share v10856 and v13072), int8
  weights (the runtime's W8F mode). "Δ vs base" is the paired difference to `base_s1`,
  with a 95% bootstrap interval over lines (10,000 resamples), computed from the same
  per-line errors as the pooled CERs, so it equals the difference of the table's
  columns. That interval covers test-set sampling only. Training noise is a separate
  term, which `base_s2` measures.
- **synth-val** is the whole held-out synthetic shard (10,001 lines).
- Every set is predicted in batches of one padded width, so every line gets exactly the
  input of eval/run.py's (and the C runtime's) one-line inference. In a mixed-width
  batch the replicate padding of shorter lines changes their last frames.

Sizes come from runtime's exporter (`train/export.py`, merged in #4) and are the same
numbers `vocr_cli --arena` prints for the exported blobs.
The **int8 blob** has BN folded, per-output-channel int8 for every pointwise,
row-collapse, bottleneck and head conv, float32 stem, depthwise convs, biases and
scales, and the vocab embedded. The **arena** is the exact `vocr_arena_bytes()` at the
default tiling. The runtime streams each line in 4-frame tiles, so the arena doesn't
grow with line width. The **footprint** is the int8 blob plus the W8F arena.

**Speed is an estimate.** Nothing has run on a console. It is the MACs of a 640-px
line divided by 0.24–0.30 GMAC/s, the research's fp32 throughput for one 444 MHz
Cortex-A9 core, which runtime's PR uses too. 640 px is about the p90 of real lines once
resized to height 32: vita-test lines are 299 px at the median, 608 px at p90 and 726 px
at most; psp-test lines are 247 / 488 / 664 px. No test line reaches the 1024-px clamp.

Environment: the shared 16-core x86 host, Python 3.11.15, torch 2.5.1+cpu.

## Results

<!-- BEGIN results -->
| run | knobs | params | int8 blob KiB | W8F arena KiB | footprint KiB | MMAC (longest line) | est. A9 ms, 640-px line | synth-val CER fp32 / int8 | vita-test CER (as is) | of which bright-on-dark | vita-test-inv CER | psp-test CER (as is) | psp-test-inv CER | real-hand-x CER | vita+psp CER, console polarity | real CER fp32 / int8 | Δ vs base_s1 (95% CI) | Pareto |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2500 | 2500 chars | 484,933 | 536 | 116 | 651 | 93 | 194-243 | 1.85% / 1.85% | 30.54% | 13.64% | 12.68% | 54.10% | 16.77% | 27.10% | 15.71% | 16.26% / 16.26% | +0.79 (+0.10, +1.51) | memory / speed |
| w050 | w0.5 | 587,611 | 654 | 62 | 716 | 84 | 174-217 | 2.80% / 2.82% | 32.72% | 17.21% | 15.80% | 55.86% | 20.48% | 40.99% | 18.82% | 20.79% / 20.98% | +5.52 (+4.36, +6.73) | speed |
| v3500 | 3500 chars | 581,933 | 642 | 116 | 758 | 106 | 220-275 | 1.34% / 1.34% | 29.68% | 12.95% | 11.82% | 53.11% | 16.81% | 24.48% | 15.20% | 15.51% / 15.57% | +0.10 (-0.61, +0.81) | memory / speed |
| w075 | w0.75 | 661,443 | 734 | 89 | 823 | 103 | 214-267 | 1.73% / 1.73% | 31.55% | 14.88% | 13.77% | 55.10% | 18.68% | 32.74% | 16.75% | 18.20% / 18.27% | +2.81 (+1.97, +3.70) |  |
| h24 | h24 | 761,771 | 840 | 95 | 935 | 91 | 189-236 | 2.49% / 2.49% | 32.49% | 16.47% | 15.05% | 54.94% | 19.20% | 30.40% | 17.80% | 18.69% / 18.77% | +3.31 (+2.37, +4.25) | speed |
| base_s1 | v1 layout | 761,771 | 840 | 116 | 955 | 128 | 267-334 | 1.28% / 1.28% | 28.67% | 12.35% | 11.54% | 51.20% | 16.57% | 25.45% | 14.90% | 15.43% / 15.46% | ref | memory / speed |
| base_s2 | v1 layout | 761,771 | 840 | 116 | 955 | 128 | 267-334 | 1.24% / 1.24% | 30.07% | 14.04% | 12.68% | 53.71% | 17.09% | 27.51% | 15.51% | 16.46% / 16.45% | +0.98 (+0.25, +1.72) |  |
| v1 | v1 layout | 761,771 | 840 | 116 | 955 | 128 | 267-334 | 0.30% / 0.30% | 23.87% | 8.04% | 7.37% | 46.57% | 10.88% | 15.27% | 9.91% | 9.84% / 9.88% | -5.59 (-6.60, -4.63) | (32,496 steps) |
| d2 | d2 | 812,427 | 902 | 153 | 1,055 | 154 | 320-401 | 1.27% / 1.26% | 29.52% | 12.75% | 11.78% | 51.04% | 16.34% | 28.75% | 14.78% | 15.86% / 15.88% | +0.41 (-0.28, +1.13) |  |
| c4k7 | ctx4 k7 | 840,491 | 930 | 134 | 1,064 | 138 | 288-360 | 1.18% / 1.18% | 28.43% | 12.35% | 11.31% | 52.31% | 16.21% | 23.11% | 14.64% | 14.89% / 14.91% | -0.55 (-1.25, +0.14) | memory / speed |
| b128 | b128 | 939,275 | 1,013 | 118 | 1,131 | 151 | 315-393 | 1.18% / 1.18% | 28.32% | 12.20% | 11.12% | 53.47% | 16.53% | 23.80% | 14.76% | 15.07% / 15.05% | -0.41 (-1.14, +0.31) |  |
| w150 | w1.5 | 1,041,915 | 1,129 | 169 | 1,298 | 200 | 417-522 | 0.85% / 0.85% | 28.63% | 12.55% | 11.70% | 52.59% | 15.14% | 26.27% | 14.25% | 15.01% / 15.01% | -0.45 (-1.19, +0.29) |  |
<!-- END results -->

Per-style int8 CER as is (eval's tags). Inverting dark-on-light lines changes only the
dark-on-light column and the rows that share lines with it.

<!-- BEGIN styles -->
| run | set | bright-on-dark | dark-on-light | outlined | coloured | name | coloured-name | furigana | mincho | small | no-box |
|---|---|---|---|---|---|---|---|---|---|---|---|
| v2500 | vita-test | 13.64% | 92.70% | 35.79% | 56.41% | 32.56% | 52.38% | 21.88% | 4.14% | – | – |
| v2500 | psp-test | 22.50% | 94.55% | 29.73% | 70.09% | 75.60% | 95.83% | 65.53% | 49.09% | 58.68% | 72.88% |
| w050 | vita-test | 17.21% | 89.78% | 39.74% | 56.92% | 37.79% | 47.62% | 37.50% | 5.77% | – | – |
| w050 | psp-test | 26.83% | 93.01% | 34.59% | 72.90% | 78.47% | 100.00% | 79.13% | 49.09% | 58.79% | 83.05% |
| v3500 | vita-test | 12.95% | 91.24% | 34.88% | 56.41% | 30.81% | 47.62% | 18.75% | 3.25% | – | – |
| v3500 | psp-test | 21.93% | 93.01% | 29.73% | 71.50% | 72.73% | 83.33% | 67.96% | 47.46% | 57.76% | 77.97% |
| w075 | vita-test | 14.88% | 92.88% | 38.52% | 60.51% | 35.47% | 66.67% | 20.31% | 3.85% | – | – |
| w075 | psp-test | 24.70% | 94.01% | 32.57% | 71.50% | 76.08% | 95.83% | 72.82% | 49.80% | 59.20% | 91.53% |
| h24 | vita-test | 16.47% | 91.42% | 38.93% | 58.97% | 37.21% | 52.38% | 26.56% | 5.33% | – | – |
| h24 | psp-test | 25.48% | 92.64% | 32.84% | 71.96% | 76.08% | 87.50% | 74.27% | 49.39% | 59.51% | 81.36% |
| base_s1 | vita-test | 12.35% | 88.69% | 32.56% | 54.36% | 27.33% | 42.86% | 17.19% | 3.40% | – | – |
| base_s1 | psp-test | 22.14% | 88.37% | 29.32% | 68.22% | 71.77% | 91.67% | 65.05% | 45.53% | 57.35% | 74.58% |
| base_s2 | vita-test | 14.04% | 89.05% | 37.41% | 57.44% | 30.23% | 52.38% | 21.88% | 2.51% | – | – |
| base_s2 | psp-test | 22.78% | 93.28% | 31.08% | 71.03% | 72.73% | 100.00% | 67.96% | 48.37% | 58.07% | 83.05% |
| v1 | vita-test | 8.04% | 82.12% | 27.00% | 46.15% | 21.51% | 33.33% | 7.81% | 1.33% | – | – |
| v1 | psp-test | 14.97% | 87.01% | 20.00% | 64.95% | 63.64% | 87.50% | 45.15% | 44.21% | 56.32% | 59.32% |
| d2 | vita-test | 12.75% | 91.24% | 35.39% | 56.41% | 29.65% | 47.62% | 14.06% | 2.37% | – | – |
| d2 | psp-test | 21.50% | 88.83% | 28.78% | 71.50% | 72.73% | 100.00% | 66.02% | 44.61% | 57.04% | 79.66% |
| c4k7 | vita-test | 12.35% | 87.59% | 33.57% | 54.36% | 33.14% | 52.38% | 21.88% | 3.85% | – | – |
| c4k7 | psp-test | 20.79% | 92.64% | 27.43% | 69.63% | 72.73% | 95.83% | 65.05% | 47.56% | 58.38% | 74.58% |
| b128 | vita-test | 12.20% | 87.59% | 33.37% | 53.33% | 29.07% | 47.62% | 20.31% | 2.96% | – | – |
| b128 | psp-test | 22.29% | 93.37% | 29.87% | 69.16% | 73.21% | 87.50% | 67.96% | 48.07% | 57.97% | 79.66% |
| w150 | vita-test | 12.55% | 87.77% | 34.28% | 56.92% | 29.07% | 52.38% | 14.06% | 2.81% | – | – |
| w150 | psp-test | 20.94% | 93.10% | 28.11% | 68.69% | 71.29% | 91.67% | 60.68% | 48.48% | 57.97% | 72.88% |
<!-- END styles -->

### With real data mixed in

`--real real_train.npz --real-weight 10` adds realdata's 9,811 real Vita and PSP training
lines (dark text already inverted by the console's rule), at about 20% of each batch.
These runs are scored on vita-test-inv and psp-test-inv only, because real-hand-x shares
13 of its 17 games with realdata's training split (its column is blank for them). vita-test and psp-test share no game
and no capture session with realdata's training or validation split; I checked both
manifests against `real_train.npz` and `real_val.npz`.

<!-- BEGIN results-real -->
| run | knobs | params | int8 blob KiB | W8F arena KiB | footprint KiB | MMAC (longest line) | est. A9 ms, 640-px line | synth-val CER fp32 / int8 | vita-test CER (as is) | of which bright-on-dark | vita-test-inv CER | psp-test CER (as is) | psp-test-inv CER | real-hand-x CER | vita+psp CER, console polarity | real CER fp32 / int8 | Δ vs base_r (95% CI) | Pareto |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b48_r | b48 | 495,515 | 579 | 112 | 691 | 94 | 196-245 | 2.10% / 2.11% | 10.22% | 7.19% | 6.47% | 21.32% | 12.07% | – | 9.50% | 9.26% / 9.24% | +0.26 (-0.40, +0.92) | memory / speed |
| v3500_r | 3500 chars | 581,933 | 642 | 116 | 758 | 106 | 220-275 | 1.83% / 1.84% | 9.83% | 6.75% | 6.32% | 19.64% | 11.83% | – | 9.28% | 9.09% / 9.05% | +0.06 (-0.61, +0.73) | memory / speed |
| base_r | v1 layout | 761,771 | 840 | 116 | 955 | 128 | 267-334 | 1.88% / 1.90% | 10.76% | 7.54% | 6.63% | 20.80% | 11.39% | – | 9.18% | 9.01% / 8.99% | ref | memory / speed |
| base_r15 | v1 layout | 761,771 | 840 | 116 | 955 | 128 | 267-334 | 0.88% / 0.89% | 6.36% | 4.76% | 4.33% | 14.38% | 7.89% | – | 6.17% | 6.01% / 6.09% | -2.90 (-3.60, -2.23) | (15,000 steps) |
| qat_r | QAT | 761,771 | 840 | 116 | 955 | 128 | 267-334 | 1.89% / 1.89% | 10.72% | 7.74% | 6.83% | 20.64% | 11.36% | – | 9.26% | 9.12% / 9.07% | +0.08 (-0.33, +0.51) |  |
| v1 | v1 layout | 761,771 | 840 | 116 | 955 | 128 | 267-334 | 0.30% / 0.30% | 23.87% | 8.04% | 7.37% | 46.57% | 10.88% | 15.27% | 9.91% | 9.05% / 9.11% | +0.12 (-1.35, +1.65) | (32,496 steps) |
| w150_r | w1.5 | 1,041,915 | 1,129 | 169 | 1,298 | 200 | 417-522 | 1.33% / 1.32% | 7.33% | 4.96% | 4.68% | 17.09% | 10.04% | – | 7.53% | 7.27% / 7.33% | -1.66 (-2.32, -1.00) | memory / speed |
| w150_r15 | w1.5 | 1,041,915 | 1,129 | 169 | 1,298 | 200 | 417-522 | 0.56% / 0.56% | 4.29% | 3.08% | 2.85% | 12.27% | 6.38% | – | 4.79% | 4.65% / 4.59% | -4.39 (-5.31, -3.56) | (15,000 steps) |
| w150d2c4_r | w1.5 d2 ctx4 | 1,323,531 | 1,436 | 244 | 1,679 | 278 | 578-723 | 1.29% / 1.28% | 7.53% | 5.31% | 5.03% | 16.65% | 9.44% | – | 7.43% | 7.27% / 7.21% | -1.77 (-2.48, -1.08) | memory / speed |
| w200_r | w2 | 1,428,043 | 1,522 | 223 | 1,745 | 299 | 624-780 | 1.01% / 1.01% | 6.28% | 4.22% | 3.94% | 15.70% | 8.01% | – | 6.27% | 5.97% / 5.95% | -3.04 (-3.79, -2.33) | memory / speed |
<!-- END results-real -->

<!-- BEGIN styles-real -->
| run | set | bright-on-dark | dark-on-light | outlined | coloured | name | coloured-name | furigana | mincho | small | no-box |
|---|---|---|---|---|---|---|---|---|---|---|---|
| b48_r | vita-test | 7.19% | 21.35% | 11.22% | 22.05% | 22.09% | 19.05% | 15.62% | 3.40% | – | – |
| b48_r | psp-test | 14.76% | 29.70% | 14.73% | 32.71% | 51.68% | 75.00% | 57.28% | 16.77% | 24.15% | 47.46% |
| v3500_r | vita-test | 6.75% | 21.17% | 9.40% | 17.95% | 20.93% | 14.29% | 14.06% | 4.59% | – | – |
| v3500_r | psp-test | 13.41% | 27.61% | 11.49% | 31.31% | 52.63% | 70.83% | 58.25% | 15.35% | 22.61% | 52.54% |
| base_r | vita-test | 7.54% | 22.63% | 10.01% | 20.00% | 22.67% | 23.81% | 14.06% | 4.44% | – | – |
| base_r | psp-test | 14.34% | 29.06% | 12.84% | 35.98% | 54.07% | 70.83% | 57.28% | 17.68% | 23.12% | 47.46% |
| base_r15 | vita-test | 4.76% | 12.23% | 6.27% | 12.82% | 14.54% | 23.81% | 7.81% | 2.22% | – | – |
| base_r15 | psp-test | 9.65% | 20.44% | 7.43% | 23.83% | 42.10% | 58.33% | 44.17% | 11.58% | 16.96% | 33.90% |
| qat_r | vita-test | 7.74% | 21.71% | 10.62% | 20.51% | 22.09% | 23.81% | 10.94% | 4.44% | – | – |
| qat_r | psp-test | 14.12% | 28.97% | 12.70% | 30.84% | 51.68% | 62.50% | 55.34% | 16.77% | 23.33% | 47.46% |
| v1 | vita-test | 8.04% | 82.12% | 27.00% | 46.15% | 21.51% | 33.33% | 7.81% | 1.33% | – | – |
| v1 | psp-test | 14.97% | 87.01% | 20.00% | 64.95% | 63.64% | 87.50% | 45.15% | 44.21% | 56.32% | 59.32% |
| w150_r | vita-test | 4.96% | 16.06% | 7.08% | 15.38% | 16.86% | 23.81% | 7.81% | 2.22% | – | – |
| w150_r | psp-test | 11.85% | 23.80% | 10.95% | 26.17% | 45.93% | 58.33% | 50.00% | 13.11% | 19.32% | 42.37% |
| w150_r15 | vita-test | 3.08% | 8.76% | 3.84% | 9.74% | 9.30% | 14.29% | 6.25% | 1.18% | – | – |
| w150_r15 | psp-test | 7.45% | 18.44% | 6.22% | 17.29% | 39.23% | 54.17% | 36.41% | 10.06% | 14.08% | 23.73% |
| w150d2c4_r | vita-test | 5.31% | 15.69% | 5.87% | 12.31% | 19.77% | 14.29% | 12.50% | 2.96% | – | – |
| w150d2c4_r | psp-test | 11.07% | 23.80% | 9.32% | 28.04% | 44.50% | 70.83% | 47.09% | 14.02% | 19.22% | 37.29% |
| w200_r | vita-test | 4.22% | 13.87% | 5.26% | 12.31% | 16.86% | 19.05% | 9.38% | 1.63% | – | – |
| w200_r | psp-test | 9.51% | 23.61% | 8.11% | 22.43% | 42.10% | 58.33% | 46.60% | 12.91% | 18.40% | 40.68% |
<!-- END styles-real -->

<!-- BEGIN running -->
### The long runs (finished after the PR was opened)

These were still training when the PR was first opened. They have since finished and been
re-scored with the reviewed code (one padded width per batch); their numbers are folded into
the recommendation and the real-data results above. The arch agent's unattended job also
posted its own tables to PR #10 as comments, scored with the pre-review code (Δ via
`score_norm`, real-hand-x in mixed-width batches): those differ from the numbers here by at
most about 0.1 point.

| run | knobs | budget | pooled real CER int8 (vita-test-inv + psp-test-inv) | Δ |
|---|---|---|---|---|
| `base_r15`, `w150_r15` | width 1.0 vs 1.5, real data, checkpoints at 5k/10k/15k | 15,000 steps | 6.09% vs 4.59% | w1.5 -1.50 (-2.10, -0.89) |
| `w200_r` | width 2.0, real data | 5,000 steps | 5.95% | -1.38 vs w1.5 (-2.01, -0.76) |
| `w150d2c4_r` | width 1.5 + depth 2 + ctx 4, real data | 5,000 steps | 7.21% | -0.12 vs w1.5 (-0.72, +0.49) |
| `v3500_r` | the 3,500 vocab (pre-review version) with real data | 5,000 steps | 9.05% | +0.06 vs base_r (-0.61, +0.73) |
| `qat_r` | QAT from half-way, real data | 5,000 steps | 9.07% | +0.08 vs base_r (-0.33, +0.51) |
| `b48_r` | bottleneck 48, real data | 5,000 steps | 9.24% | +0.26 vs base_r (-0.40, +0.92) |

Learning curve, pooled vita-test-inv + psp-test-inv, int8 (5k/10k from the arch agent's
kept checkpoints, 15k re-scored here): base_r15 10.21% / 7.11% / 6.09%, w150_r15 7.80% /
5.64% / 4.59%.
<!-- END running -->

## What each knob costs

`train/cost.py --table`. The longest line is 1024 px at height 32 and 768 px at height 24,
the same physical text.

<!-- BEGIN cost -->
| config | params | head + bottleneck params | int8 blob B | fp32 blob B | W8F arena B | W8I arena B | MMAC 640 px / longest line | est. A9 ms, 640 px | est. A9 ms, longest line |
|---|---|---|---|---|---|---|---|---|---|
| v1 (w1.0 b96 h32 5,355) | 761,771 | 537,867 | 859,904 | 3,067,840 | 118,432 | 124,832 | 80 / 128 | 267-334 | 428-535 |
| width 0.25 | 540,275 | 524,043 | 614,528 | 2,188,672 | 35,872 | 37,664 | 45 / 71 | 148-186 | 238-297 |
| width 0.5 | 587,611 | 528,651 | 669,824 | 2,375,744 | 63,392 | 66,720 | 52 / 84 | 174-217 | 278-348 |
| width 0.75 | 661,443 | 533,259 | 751,616 | 2,668,800 | 90,912 | 95,776 | 64 / 103 | 214-267 | 342-427 |
| width 1.5 | 1,041,915 | 547,083 | 1,155,968 | 4,183,872 | 173,472 | 182,944 | 125 / 200 | 417-522 | 668-835 |
| width 2.0 | 1,428,043 | 556,299 | 1,558,016 | 5,723,840 | 228,512 | 241,056 | 187 / 299 | 624-780 | 998-1248 |
| width 2.5 | 1,920,155 | 565,515 | 2,066,048 | 7,687,744 | 283,552 | 299,168 | 266 / 426 | 887-1108 | 1419-1774 |
| width 3.0 | 2,518,251 | 574,731 | 2,680,064 | 10,075,584 | 338,592 | 357,280 | 362 / 579 | 1206-1508 | 1930-2413 |
| depth 2 | 812,427 | 537,867 | 923,264 | 3,268,032 | 156,832 | 163,232 | 96 / 154 | 320-401 | 513-641 |
| depth 3 | 863,083 | 537,867 | 986,624 | 3,468,224 | 199,328 | 205,728 | 112 / 179 | 374-467 | 598-747 |
| ctx 4 | 838,955 | 537,867 | 946,176 | 3,373,760 | 130,720 | 137,120 | 86 / 138 | 288-359 | 460-575 |
| ctx 4, k 7 | 840,491 | 537,867 | 952,320 | 3,379,904 | 136,864 | 143,264 | 86 / 138 | 288-360 | 461-576 |
| bottleneck 128 | 939,275 | 715,371 | 1,037,696 | 3,778,496 | 120,864 | 127,264 | 94 / 151 | 315-393 | 504-629 |
| bottleneck 64 | 584,267 | 360,363 | 682,112 | 2,357,184 | 116,000 | 122,400 | 66 / 106 | 220-275 | 352-440 |
| bottleneck 48 | 495,515 | 271,611 | 593,216 | 2,001,856 | 114,784 | 121,184 | 59 / 94 | 196-245 | 314-393 |
| vocab 3,500 | 581,933 | 358,029 | 657,664 | 2,338,496 | 118,432 | 124,832 | 66 / 106 | 220-275 | 352-440 |
| vocab 2,500 | 484,933 | 261,029 | 548,688 | 1,945,520 | 118,432 | 124,832 | 58 / 93 | 194-243 | 311-389 |
| height 24 | 761,771 | 537,867 | 859,904 | 3,067,840 | 97,216 | 100,480 | 57 / 91 | 189-236 | 302-378 |
| w1.5 depth 2 ctx 4 | 1,323,531 | 547,083 | 1,469,952 | 5,302,144 | 249,504 | 258,976 | 174 / 278 | 578-723 | 925-1157 |
| w2.0 height 24 | 1,428,043 | 556,299 | 1,558,016 | 5,723,840 | 186,304 | 192,640 | 127 / 204 | 425-531 | 679-849 |
<!-- END cost -->

The output head (bottleneck plus the 5,355-way 1×1 conv) holds 71% of v1's parameters.
So width barely moves the weights: halving it saves 22% of the blob, while it halves the
arena and cuts the backbone's MACs. The vocab and the bottleneck shrink the head itself.
The arena is 35–349 KiB for every config (W8F 35–331 KiB, W8I 37–349 KiB), so the
weights dominate the footprint, and every config is far below the 50 MB budget. Every
config runs in all three modes. At width ≥ 2.0 the row collapse has a reduction length
K = 640, and W8I still runs it: the runtime sums int8 × int16 products exactly in int32
chunks of 512 terms and adds the chunks in float (`runtime/vocr.c`, `W8I_CHUNK`). The
C runtime's W8I logits for random-init width 2.0 and 3.0 models (K = 640 and 960) are
within 2.2e-5 of PyTorch's int8-weight model, with the same argmax on every frame
(`tests/test_arch.py` checks width 2.0).

## Vocabulary

`train/vocab.py build --size N` keeps every kana, the VN punctuation set and the digits.
It ranks the other characters of v1's vocab by a VN-weighted frequency: 95% the
synthetic text pools at the renderer's weights, 5% the kanji of VNDB character names,
because VN scripts repeat their cast's names. The names are those of every character of
a Japanese-original VN with at least 100 votes (`vocab.py names`, 25,215 names), minus
the 5,134 that appear in one of eval's test games (`test_games.json` `all` or
`is_test_vn`), so the test sets' games never inform the ranking: 20,081 names. It keeps
the top N - 1 and appends `〓` as `<unk>`. `train.py --vocab` maps every other character
to `<unk>`, so the model learns to say "unknown" instead of guessing a lookalike. A
reduced vocab is a subset of v1's. `--extra FILE` adds characters, for example the
frequent ones nvsynth reports missing from the NVIDIA text.

The test-game exclusion came in at review. Before it, `vocab_2500.txt` had 坪碗蛍覇餌Ｂ
instead of x峯朧澪荻蓄 and `vocab_3500.txt` had 枷柾沫盧穗錫黛 instead of 柘櫂瓏籐耆蕾鴫.
The sweep's `v2500` and `v3500` runs (and the follow-up `v3500_r`) were trained with
those earlier files. None of the 26 characters occurs in vita-test or psp-test, and both
sets' coverage is the same with either version.

`train/vocab.py coverage` gives the share of characters, and of lines, that a vocab can't
emit:

| text | characters | 2,500 + unk | 3,500 + unk | v1 (5,354) |
|---|---|---|---|---|
| eval vita-test (16 games) | 2,564 | 0.43% (4.4% of lines) | 0.12% (1.5%) | 0.04% (0.5%) |
| eval psp-test (14 games) | 2,510 | 0.40% (3.7%) | 0.04% (0.4%) | 0 |
| research hand transcriptions (21 games) | 952 | 0.74% (8.5%) | 0.21% (1.7%) | 0 |
| agreed real pseudo-labels, all splits | 3,492 | 0.63% (5.7%) | 0.12% (0.9%) | 0.03% (0.3%) |
| synth-val labels | 181,264 | 0.77% (11.4%) | 0.20% (3.3%) | 0 |
| Aozora dialogue pool (every 20th chunk) | 1,236,590 | 0.67% (18.1%) | 0.20% (5.8%) | 0.06% (1.4%) |
| NVIDIA Japanese text (nvsynth's every-line build, 89 sources, 473M characters) | | 1.31% | 0.49% | 0.25% (2.71% of lines) |

On vita-test, the 2,500 vocab misses 浬唾縋咄嗟箕羨蓬瘴耽, the 3,500 vocab misses 浬瘴,
and v1's misses 瘴. On psp-test, 2,500 misses 霄錆蓬濯鞄絢爛虹漏鴎, 3,500 misses 霄, and v1's
misses nothing. The VNDB-name weighting is measured on text from games outside every test
set: on realdata's validation split (16,555 characters), ranking with the names misses
0.48% of the characters at 2,500 and 0.04% at 3,500. Ranking by the corpus alone misses
0.74% and 0.16%. The NVIDIA row counts every horizontal line of nvsynth's per-source
records (`full/<split>/nv_<split>_NNN.json` character histograms, 65 train and 24
validation sources as of 2026-10-09 23:40Z). v1's misses there are mostly web symbols
(– ^ • ¥ ▪ € £ ` ´ ») and 0.014% kanji (nvsynth PR #8's report).

## Quantisation

`model.int8_conv` picks exactly the convs runtime's `--int8` export quantises:
pointwise, row collapse, bottleneck and head. `fake_quant_weight` is symmetric
per-output-channel int8 with a straight-through gradient. Per output channel it commutes
with BatchNorm folding (`tests/test_arch.py`), so it simulates the exported weights
exactly. `sweep_eval.py` scores every checkpoint three ways: fp32; int8 weights with
float activations, the runtime's W8F mode; and int16 per-position activations into the
int8 convs, the runtime's W8I mode, simulated. Post-training W8F moves the pooled real
CER by at most 0.19 points (width 0.5: 1,206 → 1,217 errors in 5,801 characters), and
by 0.06 points for the recommended width 1.5 with real data. W8I is within one character
error of W8F in every run.

<!-- BEGIN quant -->
| run | QAT | real CER fp32 | real CER int8 W8F | real CER W8I (sim.) | synth-val CER fp32 | synth-val CER int8 |
|---|---|---|---|---|---|---|
| b128 |  | 15.07% | 15.05% | 15.05% | 1.18% | 1.18% |
| base_s1 |  | 15.43% | 15.46% | 15.46% | 1.28% | 1.28% |
| base_s2 |  | 16.46% | 16.45% | 16.45% | 1.24% | 1.24% |
| c4k7 |  | 14.89% | 14.91% | 14.91% | 1.18% | 1.18% |
| d2 |  | 15.86% | 15.88% | 15.88% | 1.27% | 1.26% |
| h24 |  | 18.69% | 18.77% | 18.77% | 2.49% | 2.49% |
| v1 |  | 9.84% | 9.88% | 9.88% | 0.30% | 0.30% |
| v2500 |  | 16.26% | 16.26% | 16.26% | 1.85% | 1.85% |
| v3500 |  | 15.51% | 15.57% | 15.57% | 1.34% | 1.34% |
| w050 |  | 20.79% | 20.98% | 20.98% | 2.80% | 2.82% |
| w075 |  | 18.20% | 18.27% | 18.27% | 1.73% | 1.73% |
| w150 |  | 15.01% | 15.01% | 15.03% | 0.85% | 0.85% |
<!-- END quant -->

## Runtime compatibility

Every config uses only the runtime's op set: 3×3 stem, depthwise 3×3, pointwise, 2×1
row collapse, depthwise 1×k temporal with k ≤ 7, ReLU, residual adds and the
factorised head. No op was added; a BiLSTM or attention mixer would need new kernels,
and the temporal Ctx blocks already are an attention-free mixer.

On main's runtime (merged in #4), re-checked at review: six random-init configs with
randomised BN statistics, 24 synth-val crops each, with the runtime's own preprocessing
as the network input. The configs are w1.0/b96/h32/5,355; w0.25/b48/h24/2,501;
w0.5/b64/h24/3,501; w0.75/b96/h32/3,501; w1.5 depth 2 ctx 4; and w2.0 (K = 640).
- f32: the C logits match this branch's PyTorch model to within 5.4e-7 of the largest
  logit.
- W8F: they match `set_weight_quant(model, 8)` to within 1.2e-6. That is the int8 export
  `fake_quant_weight` simulates, so QAT trains exactly the deployed weights.
- W8I: within 3.7e-5.
- Every frame has the same argmax in every mode.

The trained recommendation (`w150_r`) passes `runtime/tests/parity.py` on 40 lines each
of vita-test and psp-test, in f32, W8F and W8I:
- logits bit-identical to `tests/vocr_ref.py` and the same strings as PyTorch;
- tile-size invariant and `VOCR_INVERT` correct (checked on 10 of the lines);
- `vocr_arena_bytes()` equal to the plan.

Through eval's harness (`eval/run.py --model 'cmd:runtime/build/vocr_cli …'`), the C
runtime gives the PyTorch CERs on all 451 lines: 7.29% / 17.05% in f32, and 7.33% /
17.09% in W8F and W8I. `export.load_checkpoint()` and eval's `TinyTorch` backend rebuild
depth / ctx / ctx_k students (this PR). `tests/test_arch.py` exports a height-24 depth-2
ctx-3 k7 student through the C runtime.

## Limitations

- 5,000 steps is 15% of v1's budget, and train2 pretrains on far more data: nvsynth's
  every-line NVIDIA set (PR #8, not merged yet) had 17.0M train lines from 65 sources at
  2026-10-09 23:28Z, plus synth2, then real fine-tuning. Gaps between sizes measured this
  early can change with budget and data, usually in favour of larger models, so the
  large end of this sweep is a lower bound on what capacity buys.
- One seed per config, plus one repeat of the baseline, which differed by 0.98 points.
  Single-seed differences under about 1–2 points are within training noise. The
  width 1.5 vs 1.0 gain with real data (-1.66 points) is one seed pair. The 15,000-step
  pair, which repeats the comparison at three times the budget, was still training at
  merge (below).
- vita-test-inv and psp-test-inv use the style tags as a polarity oracle. The deployed
  pipeline needs the line finder or the training data to handle dark text. The
  vita-test-con / psp-test-con columns use the console's own polarity rule instead
  (`common.line_dark`, which agrees with the tags on 93.7% / 95.5% of lines).
- `v2500` and `v3500` were trained with the reduced vocabs from before the test-game
  exclusion (Vocabulary). The two versions differ in 6 and 7 characters, none of them
  in the test sets.
- Console speed isn't measured; MACs are the proxy. qemu-arm has no instruction-count
  plugin here.

## Reproduce

```sh
H=/local/home/skerraut/herd-vita-ocr; PY=$H/shared/venv/bin/python
export VOCR_EVAL_DATA=$H/shared/artifacts/eval      # eval's screenshots
$PY train/vocab.py names                            # VNDB character names + VN ids (local only)
$PY train/vocab.py build --size 2500; $PY train/vocab.py build --size 3500
$PY train/vocab.py coverage
# one knob per run; *_r runs add: --real $H/shared/artifacts/realdata/real_train.npz --real-weight 10
$H/heavy.sh $PY train/train.py --run base_s1 --out runs --steps 5000 --eval-every 2500 --seed 1 [knob]
mkdir -p runs/v1 && ln -s $H/shared/models/v1/ckpt_best.pt runs/v1/ckpt_last.pt   # v1, for scale
$H/heavy.sh $PY train/sweep_eval.py --run runs/base_s1        # every set, fp32 / W8F / W8I
$PY train/sweep_table.py --runs runs --ref base_s1 --json docs/arch-sweep.json --doc docs/arch-sweep.md
$PY train/sweep_table.py --runs runs --data real --ref base_r --json docs/arch-sweep-real.json --doc docs/arch-sweep.md
$PY train/cost.py --table                                   # the cost block (also written by --doc)
```

The review re-scored all 14 evaluated checkpoints this way at commit 777f74d. Its eval
JSONs, with every line's prediction and errors, are in
`$HERD/shared/artifacts/arch/review-777f74d/<run>/`.
