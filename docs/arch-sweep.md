# Architecture sweep: how small, and how big, should the recognizer be?

<!-- BEGIN summary -->
**Recommendation for train2 (provisional, 2026-10-09 21:45Z): width 1.5, v1's layout otherwise, int8 weights.**

| | train2 (release) | PSP-lean variant |
|---|---|---|
| config | `--width 1.5`, bottleneck 96, height 32, 2 Ctx blocks (k 5), depth 1, the full 5,354-character vocab | v1's size: `--width 1.0`, the rest the same (optionally `--vocab data/vocab_3500.txt`) |
| params | 1,041,915 | 761,771 (581,933 with the 3,500 vocab) |
| int8 blob (runtime's exporter) | 1,155,968 B | 859,904 B (657,664 B) |
| arena, `vocr_arena_bytes()` W8F / W8I | 173,472 / 182,944 B, at any line width | 118,432 / 124,832 B |
| MMAC, median / p90 real Vita line (299 / 608 px) | 59 / 119 | 38 / 76 (31 / 63) |
| est. Cortex-A9 444 MHz, one core, p90 Vita line | **396-496 ms** (estimate) | 254-318 ms (209-261 ms) |
| real CER at 5,000 steps with real data, int8 (vita-test-inv + psp-test-inv) | **7.33%** | 8.99% |
| as-is vita-test / psp-test CER at 5,000 steps, int8 (exact lines) | 7.33% / 17.09% (54.9% / 42.0%) | 10.76% / 20.80% (44.2% / 33.5%) |

Why:

- **Width is the only knob that measurably helps once real data is in the mix.** With realdata's 9,811 real lines mixed in (`--real real_train.npz --real-weight 10`), width 1.5 beats width 1.0 by **-1.69 points** of pooled real CER (95% bootstrap CI -2.35 to -1.05). That is beyond the 1.05-point seed-to-seed spread of the baseline. The as-is vita-test drops from 10.76% to 7.33% and psp-test from 20.80% to 17.09%. It improves on every style except vita-test's coloured names, where the two tie.
- On synthetic data alone, no knob beats the v1 layout by more than seed noise: width 1.5 -0.36, bottleneck 128 -0.38, ctx 4 / k 7 -0.45, depth 2 +0.47, vocab 3,500 +0.17 points. Width 1.5 still fits synth-val much better (0.84% vs 1.21%), so the synthetic-to-real gap, not capacity, limited those runs.
- **Shrinking hurts:** width 0.75 costs +2.84 points, height 24 +3.38, width 0.5 +5.52, all with intervals clear of zero. Vocab 2,500 costs +0.83 (CI +0.12 to +1.54) and can't emit 0.43% of vita-test's characters.
- **Data and budget matter more than any architecture knob.** Mixing in real data cuts the as-is vita-test CER of the v1-size model from 28.67% to 10.76% at the same 5,000 steps. Training 6.5× longer (v1's 32k steps) cuts the synthetic-only pooled real CER from 15.39% to 9.88%. The 2.30% bar has to come from train2's data and budget: about 23M NVIDIA lines, synth2, and teacher-labelled real crops.
- **int8 needs no QAT:** post-training per-channel int8 (W8F) moves pooled real CER by at most 0.16 points in any run (7.27% fp32 vs 7.33% int8 for width 1.5 with real data). Simulated W8I matches W8F to within one character error on every pooled set.
- **Everything fits the budget easily.** Across the whole grid the int8 blob is 0.55-2.7 MB (width 3.0 is the largest) and the arena 36-340 KiB, at most about 6% of the 50 MB budget. Speed is the binding constraint. The estimates for a p90 Vita line are 593-741 ms at width 2.0, 0.84-1.05 s at width 2.5 and 1.15-1.43 s at width 3.0.
- **Keep the full vocab** for the release model. It misses 0.04% of vita-test's characters and none of psp-test's. The 3,500 vocab misses 0.12% / 0.04% and 0.49% of the NVIDIA text, for 18% fewer MACs. To keep NVIDIA lines that contain rare symbols, `train/vocab.py build --size 0 --extra <nvsynth's OOV list>` builds the full vocab plus those symbols and `<unk>`.
- **PSP-lean:** the PSP has no NEON, so MACs matter more there. Width 1.0 saves 36% of width 1.5's MACs for +1.69 points; the 3,500 vocab saves 47%. Height 24 isn't worth its +3.3 points. The psp agent should pick between the two on measured PSP speed.

**Still running** (posted to this PR as comments when they finish; see "Still running" below): width 2.0 and width 1.5 + depth 2 + ctx 4 with real data, the 3,500 vocab with real data, QAT and bottleneck 48 with real data, and the decisive **15,000-step pair (width 1.0 vs 1.5, real data)**, with learning curves at 5k/10k/15k. If width 2.0 beats width 1.5 by more than about 1 point with its interval clear of zero, it is still within budget (W8F only: its row-collapse K = 640 exceeds W8I's 516).
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
| `--width` | 0.5, 0.75, **1.0**, 1.5; 2.0 with real data (running) | 0.25, 2.5, 3.0 | every channel count (32/64/128/160/192 at 1.0) |
| `--depth` | **1**, 2; 2 + ctx 4 at width 1.5 with real data (running) | 3 | stride-1 DS blocks after each of the three downsampling blocks |
| `--ctx`, `--ctx-k` | **2** blocks k **5**; 4 blocks k 7 | | temporal context blocks (depthwise 1×k + pointwise, residual) on the CTC frame row |
| `--bottleneck` | **96**, 128; 48 with real data (queued) | 64 | width of the factorised head's 1×1 bottleneck |
| `--vocab` | 2,500 and 3,500 + `<unk>`, **5,354**; 3,500 with real data (running) | | output classes (plus the CTC blank) |
| `--height` | 24, **32** | | input line height (24 collapses 24→12→6→3→2→1 with the same layers) |
| `--qat` | with real data (queued) | | int8 weight fake quantisation over the last half of training |
| `--real` | off; realdata's `real_train.npz` at weight 10 for `*_r` runs | | mixes real training lines into every batch |

Scores (`train/sweep_eval.py`, collected by `train/sweep_table.py`):

- **vita-test** and **psp-test** are eval's hand-checked sets, cropped and scored by
  eval's own `sets.py` and `metrics.py`. vita-test is on main (206 lines, 2,564
  characters, 16 Vita games) and psp-test (245 lines, 2,510 characters, 14 PSP VNs),
  both as merged on main (eval #2 and #5; manifest sha256s are in each run's eval
  JSON). On v1 this gives eval's published numbers exactly: vita-test 23.75% CER /
  52.43% exact, psp-test 46.45% / 30.2%.
- **vita-test-inv** and **psp-test-inv** are the same lines with the dark-on-light
  ones inverted by their style tag. The v1 recipe renders only light text, so every
  v1-recipe model reads dark text at 80–95% CER whatever its size. That's a data gap,
  which synth2, NVIDIA pretraining and the real fine-tuning close. Inverting by the tag
  (an oracle) keeps it from drowning the architecture differences. Inverted, v1 reads
  dark-on-light lines at 4.9% (vita) and 6.2% (psp) CER.
- **real-hand-x** is 46 lines (727 characters) that the research campaign
  transcribed by hand from 17 Vita games outside vita-test, with dark lines inverted by
  their tag.
- **real CER**, the decision metric, pools vita-test-inv, psp-test-inv and real-hand-x:
  5,801 characters, 47 games, int8 weights (the runtime's W8F mode). "Δ vs base" is the
  paired difference to `base_s1`, with a 95% bootstrap interval over lines (10,000
  resamples). That interval covers test-set sampling only. Training noise is a separate
  term, which `base_s2` measures.
- **synth-val** is the whole held-out synthetic shard (10,001 lines).

Sizes come from runtime's exporter (`train/export.py` on runtime's PR #4 at `d7460e3`).
The **int8 blob** has BN folded, per-output-channel int8 for every pointwise,
row-collapse, bottleneck and head conv, float32 stem, depthwise convs, biases and
scales, and the vocab embedded. The **arena** is the exact `vocr_arena_bytes()` at the
default tiling. The runtime streams each line in 4-frame tiles, so the arena doesn't
grow with line width. The **footprint** is the int8 blob plus the W8F arena.

**Speed is an estimate.** Nothing has run on a console. It is the MACs of a 640-px
line divided by 0.24–0.30 GMAC/s, the research's fp32 throughput for one 444 MHz
Cortex-A9 core, which runtime's PR uses too. 640 px is about the p90 of real lines once
resized to height 32: vita-test lines are 299 px at the median, 608 px at p90 and 726 px
at most; psp-test lines are 247 / 485 / 664 px. No test line reaches the 1024-px clamp.

Environment: the shared 16-core x86 host, Python 3.11.15, torch 2.5.1+cpu.

## Results

<!-- BEGIN results -->
| run | knobs | params | int8 blob KiB | W8F arena KiB | footprint KiB | MMAC (longest line) | est. A9 ms, 640-px line | synth-val CER fp32 / int8 | vita-test CER (as is) | of which bright-on-dark | vita-test-inv CER | psp-test CER (as is) | psp-test-inv CER | real-hand-x CER | vita+psp CER, console polarity | real CER fp32 / int8 | Δ vs base_s1 (95% CI) | Pareto |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v2500 | 2500 chars | 484,933 | 536 | 116 | 651 | 93 | 194-243 | 1.84% / 1.84% | 30.54% | 13.64% | 12.68% | 54.10% | 16.77% | 26.69% | 15.71% | 16.20% / 16.20% | +0.83 (+0.12, +1.54) | memory / speed |
| w050 | w0.5 | 587,611 | 654 | 62 | 716 | 84 | 174-217 | 2.77% / 2.79% | 32.72% | 17.21% | 15.80% | 55.86% | 20.48% | 40.44% | 18.82% | 20.76% / 20.91% | +5.52 (+4.35, +6.74) | speed |
| v3500 | 3500 chars | 581,933 | 642 | 116 | 758 | 106 | 220-275 | 1.32% / 1.33% | 29.68% | 12.95% | 11.82% | 53.11% | 16.81% | 24.35% | 15.20% | 15.50% / 15.55% | +0.17 (-0.53, +0.88) | memory / speed |
| w075 | w0.75 | 661,443 | 734 | 89 | 823 | 103 | 214-267 | 1.71% / 1.71% | 31.55% | 14.88% | 13.77% | 55.10% | 18.68% | 32.46% | 16.75% | 18.15% / 18.24% | +2.84 (+2.00, +3.73) |  |
| h24 | h24 | 761,771 | 840 | 95 | 935 | 91 | 189-236 | 2.46% / 2.46% | 32.49% | 16.47% | 15.05% | 54.94% | 19.20% | 30.40% | 17.80% | 18.69% / 18.77% | +3.38 (+2.46, +4.31) | speed |
| base_s1 | v1 layout | 761,771 | 840 | 116 | 955 | 128 | 267-334 | 1.21% / 1.21% | 28.67% | 12.35% | 11.54% | 51.20% | 16.57% | 24.90% | 14.90% | 15.36% / 15.39% | ref | memory / speed |
| base_s2 | v1 layout | 761,771 | 840 | 116 | 955 | 128 | 267-334 | 1.23% / 1.23% | 30.07% | 14.04% | 12.68% | 53.71% | 17.09% | 27.37% | 15.51% | 16.45% / 16.43% | +1.05 (+0.33, +1.78) |  |
| v1 | v1 layout | 761,771 | 840 | 116 | 955 | 128 | 267-334 | 0.29% / 0.30% | 23.87% | 8.04% | 7.37% | 46.57% | 10.88% | 15.27% | 9.91% | 9.84% / 9.88% | -5.52 (-6.53, -4.57) | (32,496 steps) |
| d2 | d2 | 812,427 | 902 | 153 | 1,055 | 154 | 320-401 | 1.26% / 1.25% | 29.52% | 12.75% | 11.78% | 51.04% | 16.34% | 28.61% | 14.78% | 15.82% / 15.86% | +0.47 (-0.23, +1.18) |  |
| c4k7 | ctx4 k7 | 840,491 | 930 | 134 | 1,064 | 138 | 288-360 | 1.17% / 1.17% | 28.43% | 12.35% | 11.31% | 52.31% | 16.21% | 23.25% | 14.64% | 14.91% / 14.93% | -0.45 (-1.15, +0.24) | memory / speed |
| b128 | b128 | 939,275 | 1,013 | 118 | 1,131 | 151 | 315-393 | 1.17% / 1.17% | 28.32% | 12.20% | 11.12% | 53.47% | 16.53% | 23.38% | 14.76% | 15.01% / 15.00% | -0.38 (-1.10, +0.34) |  |
| w150 | w1.5 | 1,041,915 | 1,129 | 169 | 1,298 | 200 | 417-522 | 0.84% / 0.84% | 28.63% | 12.55% | 11.70% | 52.59% | 15.14% | 26.00% | 14.25% | 14.98% / 14.98% | -0.36 (-1.10, +0.36) |  |
<!-- END results -->

Per-style int8 CER as is (eval's tags). Inverting dark-on-light lines changes only the
dark-on-light column and the rows that share lines with it.

<!-- BEGIN styles -->
| run | set | bright-on-dark | dark-on-light | outlined | coloured | name | furigana | mincho | small |
|---|---|---|---|---|---|---|---|---|---|
| v2500 | vita-test | 13.64% | 92.70% | 35.79% | 56.41% | 32.56% | 21.88% | 4.14% | – |
| v2500 | psp-test | 22.50% | 94.55% | 29.73% | 70.09% | 75.60% | 65.53% | 49.09% | 58.68% |
| w050 | vita-test | 17.21% | 89.78% | 39.74% | 56.92% | 37.79% | 37.50% | 5.77% | – |
| w050 | psp-test | 26.83% | 93.01% | 34.59% | 72.90% | 78.47% | 79.13% | 49.09% | 58.79% |
| v3500 | vita-test | 12.95% | 91.24% | 34.88% | 56.41% | 30.81% | 18.75% | 3.25% | – |
| v3500 | psp-test | 21.93% | 93.01% | 29.73% | 71.50% | 72.73% | 67.96% | 47.46% | 57.76% |
| w075 | vita-test | 14.88% | 92.88% | 38.52% | 60.51% | 35.47% | 20.31% | 3.85% | – |
| w075 | psp-test | 24.70% | 94.01% | 32.57% | 71.50% | 76.08% | 72.82% | 49.80% | 59.20% |
| h24 | vita-test | 16.47% | 91.42% | 38.93% | 58.97% | 37.21% | 26.56% | 5.33% | – |
| h24 | psp-test | 25.48% | 92.64% | 32.84% | 71.96% | 76.08% | 74.27% | 49.39% | 59.51% |
| base_s1 | vita-test | 12.35% | 88.69% | 32.56% | 54.36% | 27.33% | 17.19% | 3.40% | – |
| base_s1 | psp-test | 22.14% | 88.37% | 29.32% | 68.22% | 71.77% | 65.05% | 45.53% | 57.35% |
| base_s2 | vita-test | 14.04% | 89.05% | 37.41% | 57.44% | 30.23% | 21.88% | 2.51% | – |
| base_s2 | psp-test | 22.78% | 93.28% | 31.08% | 71.03% | 72.73% | 67.96% | 48.37% | 58.07% |
| v1 | vita-test | 8.04% | 82.12% | 27.00% | 46.15% | 21.51% | 7.81% | 1.33% | – |
| v1 | psp-test | 14.97% | 87.01% | 20.00% | 64.95% | 63.64% | 45.15% | 44.21% | 56.32% |
| d2 | vita-test | 12.75% | 91.24% | 35.39% | 56.41% | 29.65% | 14.06% | 2.37% | – |
| d2 | psp-test | 21.50% | 88.83% | 28.78% | 71.50% | 72.73% | 66.02% | 44.61% | 57.04% |
| c4k7 | vita-test | 12.35% | 87.59% | 33.57% | 54.36% | 33.14% | 21.88% | 3.85% | – |
| c4k7 | psp-test | 20.79% | 92.64% | 27.43% | 69.63% | 72.73% | 65.05% | 47.56% | 58.38% |
| b128 | vita-test | 12.20% | 87.59% | 33.37% | 53.33% | 29.07% | 20.31% | 2.96% | – |
| b128 | psp-test | 22.29% | 93.37% | 29.87% | 69.16% | 73.21% | 67.96% | 48.07% | 57.97% |
| w150 | vita-test | 12.55% | 87.77% | 34.28% | 56.92% | 29.07% | 14.06% | 2.81% | – |
| w150 | psp-test | 20.94% | 93.10% | 28.11% | 68.69% | 71.29% | 60.68% | 48.48% | 57.97% |
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
| base_r | v1 layout | 761,771 | 840 | 116 | 955 | 128 | 267-334 | 1.86% / 1.87% | 10.76% | 7.54% | 6.63% | 20.80% | 11.39% | – | 9.18% | 9.01% / 8.99% | ref | memory / speed |
| v1 | v1 layout | 761,771 | 840 | 116 | 955 | 128 | 267-334 | 0.29% / 0.30% | 23.87% | 8.04% | 7.37% | 46.57% | 10.88% | 15.27% | 9.91% | 9.05% / 9.11% | +0.04 (-1.44, +1.59) | (32,496 steps) |
| w150_r | w1.5 | 1,041,915 | 1,129 | 169 | 1,298 | 200 | 417-522 | 1.31% / 1.31% | 7.33% | 4.96% | 4.68% | 17.09% | 10.04% | – | 7.53% | 7.27% / 7.33% | -1.69 (-2.35, -1.05) | memory / speed |
<!-- END results-real -->

<!-- BEGIN styles-real -->
| run | set | bright-on-dark | dark-on-light | outlined | coloured | name | furigana | mincho | small |
|---|---|---|---|---|---|---|---|---|---|
| base_r | vita-test | 7.54% | 22.63% | 10.01% | 20.00% | 22.67% | 14.06% | 4.44% | – |
| base_r | psp-test | 14.34% | 29.06% | 12.84% | 35.98% | 54.07% | 57.28% | 17.68% | 23.12% |
| v1 | vita-test | 8.04% | 82.12% | 27.00% | 46.15% | 21.51% | 7.81% | 1.33% | – |
| v1 | psp-test | 14.97% | 87.01% | 20.00% | 64.95% | 63.64% | 45.15% | 44.21% | 56.32% |
| w150_r | vita-test | 4.96% | 16.06% | 7.08% | 15.38% | 16.86% | 7.81% | 2.22% | – |
| w150_r | psp-test | 11.85% | 23.80% | 10.95% | 26.17% | 45.93% | 50.00% | 13.11% | 19.32% |
<!-- END styles-real -->

<!-- BEGIN running -->
### Still running (2026-10-09 21:45Z)

These are in `heavy.sh` slots or queued behind them. An unattended job posts each group's tables to this PR as a comment when it finishes, and copies the checkpoints to `$HERD/shared/artifacts/arch/runs/`.

| run | knobs | budget | status at 21:45Z |
|---|---|---|---|
| `base_r15`, `w150_r15` | width 1.0 vs 1.5, real data mixed in, checkpoints kept at 5k/10k/15k | 15,000 steps | training (step ~3k of 15k), about 3-4 h left |
| `w200_r`, `w150d2c4_r`, `v3500_r` | width 2.0; width 1.5 + depth 2 + ctx 4; the 3,500 vocab; all with real data | 5,000 steps | training, about 3-4 h |
| `qat_r`, `b48_r` | QAT from half-way; bottleneck 48; with real data | 5,000 steps | queued after those |
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
| width 2.0 | 1,428,043 | 556,299 | 1,558,016 | 5,723,840 | 228,512 | n/a (K > 516) | 187 / 299 | 624-780 | 998-1248 |
| width 2.5 | 1,920,155 | 565,515 | 2,066,048 | 7,687,744 | 283,552 | n/a (K > 516) | 266 / 426 | 887-1108 | 1419-1774 |
| width 3.0 | 2,518,251 | 574,731 | 2,680,064 | 10,075,584 | 338,592 | n/a (K > 516) | 362 / 579 | 1206-1508 | 1930-2413 |
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
| w2.0 height 24 | 1,428,043 | 556,299 | 1,558,016 | 5,723,840 | 186,304 | n/a (K > 516) | 127 / 204 | 425-531 | 679-849 |
<!-- END cost -->

The output head (bottleneck plus the 5,355-way 1×1 conv) holds 71% of v1's parameters.
So width barely moves the weights: halving it saves 22% of the blob, while it halves the
arena and cuts the backbone's MACs. The vocab and the bottleneck shrink the head itself.
The arena is 36–340 KiB for every config, so the weights dominate the footprint, and
every config is far below the 50 MB budget. At width ≥ 2.0 the row collapse has a
reduction length K = 640, over the runtime's W8I limit of 516, so those models run in
f32 or W8F only.

## Vocabulary

`train/vocab.py build --size N` keeps every kana, the VN punctuation set and the digits.
It ranks the other characters of v1's vocab by a VN-weighted frequency: 95% the
synthetic text pools at the renderer's weights, 5% the kanji of 25,215 VNDB character
names, because VN scripts repeat their cast's names. It keeps the top N - 1 and appends
`〓` as `<unk>`. `train.py --vocab` maps every other character to `<unk>`, so the model
learns to say "unknown" instead of guessing a lookalike. A reduced vocab is a subset of
v1's. `--extra FILE` adds characters, for example the frequent ones nvsynth reports
missing from the NVIDIA text.

`train/vocab.py coverage` gives the share of characters, and of lines, that a vocab can't
emit:

| text | characters | 2,500 + unk | 3,500 + unk | v1 (5,354) |
|---|---|---|---|---|
| eval vita-test (16 games) | 2,564 | 0.43% (4.4% of lines) | 0.12% (1.5%) | 0.04% (0.5%) |
| research hand transcriptions (21 games) | 952 | 0.74% (8.5%) | 0.21% (1.7%) | 0 |
| agreed real pseudo-labels, all splits | 3,492 | 0.63% (5.7%) | 0.12% (0.9%) | 0.03% (0.3%) |
| synth-val labels | 181,264 | 0.77% (11.4%) | 0.20% (3.3%) | 0 |
| Aozora dialogue pool (every 20th chunk) | 1,236,590 | 0.66% (18.0%) | 0.20% (5.8%) | 0.06% (1.4%) |
| NVIDIA Japanese text (nvsynth PR #8, 305M chars) | | 1.33% | 0.49% | 0.25% (2.65% of lines) |

On vita-test, the 2,500 vocab misses 浬唾縋咄嗟箕羨蓬瘴耽, the 3,500 vocab misses 浬瘴,
and v1's misses 瘴. The VNDB-name weighting matters at 2,500: ranking by the corpus alone
misses 1.05% of the research lines' characters instead of 0.74%. v1's misses on the
NVIDIA text are mostly web symbols (– ^ • ¥ ▪ € £ ` ´ ») and 0.012% kanji.

## Quantisation

`model.int8_conv` picks exactly the convs runtime's `--int8` export quantises:
pointwise, row collapse, bottleneck and head. `fake_quant_weight` is symmetric
per-output-channel int8 with a straight-through gradient. Per output channel it commutes
with BatchNorm folding (`tests/test_arch.py`), so it simulates the exported weights
exactly. `sweep_eval.py` scores every checkpoint three ways: fp32; int8 weights with
float activations, the runtime's W8F mode; and int16 per-position activations into the
int8 convs, the runtime's W8I mode, simulated.

<!-- BEGIN quant -->
| run | QAT | real CER fp32 | real CER int8 W8F | real CER W8I (sim.) | synth-val CER fp32 | synth-val CER int8 |
|---|---|---|---|---|---|---|
| b128 |  | 15.01% | 15.00% | 15.00% | 1.17% | 1.17% |
| base_s1 |  | 15.36% | 15.39% | 15.39% | 1.21% | 1.21% |
| base_s2 |  | 16.45% | 16.43% | 16.43% | 1.23% | 1.23% |
| c4k7 |  | 14.91% | 14.93% | 14.93% | 1.17% | 1.17% |
| d2 |  | 15.82% | 15.86% | 15.86% | 1.26% | 1.25% |
| h24 |  | 18.69% | 18.77% | 18.77% | 2.46% | 2.46% |
| v1 |  | 9.84% | 9.88% | 9.88% | 0.29% | 0.30% |
| v2500 |  | 16.20% | 16.20% | 16.20% | 1.84% | 1.84% |
| v3500 |  | 15.50% | 15.55% | 15.55% | 1.32% | 1.33% |
| w050 |  | 20.76% | 20.91% | 20.91% | 2.77% | 2.79% |
| w075 |  | 18.15% | 18.24% | 18.24% | 1.71% | 1.71% |
| w150 |  | 14.98% | 14.98% | 15.00% | 0.84% | 0.84% |
<!-- END quant -->

## Runtime compatibility

Every config uses only the runtime's op set: 3×3 stem, depthwise 3×3, pointwise, 2×1
row collapse, depthwise 1×k temporal with k ≤ 7, ReLU, residual adds and the
factorised head. No op was added; a BiLSTM or attention mixer would need new kernels,
and the temporal Ctx blocks already are an attention-free mixer.

On runtime's code (export.py `cdcf220`, vocr.c/h `e5c4bb6`), four random-init configs
with randomised BN statistics were checked on 24 synth-val crops each:
w1.0/b96/h32/5,355; w0.25/b48/h24/2,501; w0.5/b64/h24/3,501; w0.75/b96/h32/3,501.
- f32: the C logits equal this branch's PyTorch model exactly (max relative difference 0).
- W8F: they equal `set_weight_quant(model, 8)` exactly (0).
- W8I: within 1.7e-5.
- Every frame has the same argmax.

The sizes are the same with runtime's PR head (`d7460e3`). Coordination is on PR #4:
`tests/parity.py` hard-codes height 32, and `export.load_checkpoint()` doesn't yet
pass `depth`/`ctx`/`ctx_k`. eval's `TinyTorch` backend passes them (this PR).

## Limitations

- 5,000 steps is 15% of v1's budget, and train2 pretrains on far more data (about 23M
  NVIDIA lines, synth2, then real fine-tuning). Gaps between sizes measured this early
  can change with budget and data, usually in favour of larger models, so the large end
  of this sweep is a lower bound on what capacity buys.
- One seed per config, plus one repeat of the baseline, which differed by 1.05 points.
  Single-seed differences under about 1–2 points are within training noise. The
  width 1.5 vs 1.0 gain with real data (-1.69 points) is one seed pair; the 15,000-step
  pair that is still running repeats the comparison at three times the budget.
- vita-test-inv and psp-test-inv use the style tags as a polarity oracle. The deployed
  pipeline needs the line finder or the training data to handle dark text. The
  vita-test-con / psp-test-con columns use the console's own polarity rule instead
  (`common.line_dark`, which agrees with the tags on 93.7% / 94.7% of lines).
- Console speed isn't measured; MACs are the proxy. qemu-arm has no instruction-count
  plugin here.

## Reproduce

```sh
H=/local/home/skerraut/herd-vita-ocr; PY=$H/shared/venv/bin/python
export VOCR_EVAL_DATA=$H/shared/artifacts/eval      # eval's screenshots
export VOCR_EVAL_MANIFESTS=DIR                      # optional: main's vita-test + PR #5's psp-test
$PY train/vocab.py names                            # VNDB character names (local only)
$PY train/vocab.py build --size 2500; $PY train/vocab.py build --size 3500
$PY train/vocab.py coverage
$H/heavy.sh $PY train/train.py --run base_s1 --out runs --steps 5000 --eval-every 2500 --seed 1 [knob]
$H/heavy.sh $PY train/sweep_eval.py --run runs/base_s1
$PY train/sweep_table.py --ref base_s1 --export RUNTIME/train/export.py --json docs/arch-sweep.json --doc docs/arch-sweep.md
```
