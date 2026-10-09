# Architecture sweep: how small, and how big, should the recognizer be?

<!-- BEGIN summary -->
STATUS: runs in progress; summary and recommendation follow when the sweep completes.
<!-- END summary -->

## Setup

Every run uses the v1 recipe: `train/train.py` on v1's synthetic shards, the same
augmentation, AdamW at lr 2e-3 with 1,000 warm-up steps and a cosine decay to 5%. Each
run changes **one knob** and trains for a **matched budget of 5,000 steps of 96 lines**
(480k lines, 1.2 epochs of the 390k-line synthetic set) with seed 1. Each `heavy.sh`
slot (5 threads) runs two configs at 2 threads each. v1 itself trained for 32,496
steps (8 epochs); its row is shown for scale and isn't part of the Pareto
comparison. `base_s2` repeats the baseline with seed 2 to show the seed-to-seed
noise.

Knobs (the defaults rebuild v1 exactly; `train.py --help`):

| knob | values | what it changes |
|---|---|---|
| `--width` | 0.25, 0.5, 0.75, **1.0**, 1.5, 2.0, 2.5 | every channel count (32/64/128/160/192 at 1.0) |
| `--depth` | **1**, 2, 3 | stride-1 DS blocks after each of the three downsampling blocks |
| `--ctx`, `--ctx-k` | **2**, 4 blocks; k **5**, 7 | temporal context blocks (depthwise 1×k + pointwise, residual) on the CTC frame row |
| `--bottleneck` | 48, 64, **96**, 128 | width of the factorised head's 1×1 bottleneck |
| `--vocab` | 2,500 / 3,500 / **5,354** + blank | output classes; the reduced vocabs end in an `<unk>` class |
| `--height` | 24, **32** | input line height (24 collapses 24→12→6→3→2→1 with the same layers) |
| `--qat` | off, 0.5 | int8 weight fake quantisation over the last half of training |

Scores (`train/sweep_eval.py`, collected by `train/sweep_table.py`):

- **vita-test** and **psp-test** are eval's hand-checked sets, cropped and scored by
  eval's own `sets.py` and `metrics.py`. vita-test is on main (206 lines, 2,564
  characters, 16 Vita games). psp-test comes from eval's PR #5 at `bebd4f9`
  (245 lines, 2,510 characters, 14 PSP VNs), not merged yet, so its numbers are
  provisional until that merges. On v1 this gives eval's published numbers exactly:
  vita-test 23.75% CER / 52.43% exact, psp-test 46.65%.
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
| run | knobs | params | int8 blob KiB | W8F arena KiB | footprint KiB | MMAC (longest line) | est. A9 ms, 640-px line | synth-val CER fp32 / int8 | vita-test CER (as is) | of which bright-on-dark | vita-test-inv CER | psp-test CER (as is) | psp-test-inv CER | real-hand-x CER | real CER fp32 / int8 | Δ vs base_s1 (95% CI) | Pareto |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| w050 | w0.5 | 587,611 | 654 | 62 | 716 | 84 | 174-217 | 2.77% / 2.79% | 32.72% | 17.21% | 15.80% | 56.06% | 21.83% | 40.44% | 21.34% / 21.50% | +5.55 (+4.39, +6.78) | memory / speed |
| v3500 | 3500 chars | 581,933 | 642 | 116 | 758 | 106 | 220-275 | 1.32% / 1.33% | 29.68% | 12.95% | 11.82% | 53.31% | 18.21% | 24.35% | 16.10% / 16.15% | +0.22 (-0.48, +0.93) | memory / speed |
| w075 | w0.75 | 661,443 | 734 | 89 | 823 | 103 | 214-267 | 1.71% / 1.71% | 31.55% | 14.88% | 13.77% | 55.30% | 20.00% | 32.46% | 18.74% / 18.81% | +2.86 (+2.02, +3.74) | speed |
| h24 | h24 | 761,771 | 840 | 95 | 935 | 91 | 189-236 | 2.46% / 2.46% | 32.49% | 16.47% | 15.05% | 55.14% | 20.36% | 30.40% | 19.19% / 19.27% | +3.33 (+2.40, +4.27) | speed |
| base_s1 | v1 layout | 761,771 | 840 | 116 | 955 | 128 | 267-334 | 1.21% / 1.21% | 28.67% | 12.35% | 11.54% | 51.31% | 17.85% | 24.90% | 15.91% / 15.95% | ref | memory / speed |
| base_s2 | v1 layout | 761,771 | 840 | 116 | 955 | 128 | 267-334 | 1.23% / 1.23% | 30.07% | 14.04% | 12.68% | 53.90% | 18.37% | 27.37% | 17.00% / 16.98% | +1.05 (+0.33, +1.78) |  |
| v1 | v1 layout | 761,771 | 840 | 116 | 955 | 128 | 267-334 | 0.29% / 0.30% | 23.87% | 8.04% | 7.37% | 46.77% | 12.27% | 15.27% | 10.45% / 10.48% | -5.46 (-6.47, -4.52) | (32,496 steps) |
<!-- END results -->

Per-style int8 CER as is (eval's tags). Inverting dark-on-light lines changes only the
dark-on-light column and the rows that share lines with it.

<!-- BEGIN styles -->
| run | set | bright-on-dark | dark-on-light | outlined | coloured | name | furigana | mincho | small |
|---|---|---|---|---|---|---|---|---|---|
| w050 | vita-test | 17.21% | 89.78% | 39.74% | 56.92% | 37.79% | 37.50% | 5.77% | – |
| w050 | psp-test | 28.50% | 92.84% | 34.59% | 72.90% | 78.47% | 79.13% | 49.59% | 58.89% |
| v3500 | vita-test | 12.95% | 91.24% | 34.88% | 56.41% | 30.81% | 18.75% | 3.25% | – |
| v3500 | psp-test | 23.83% | 92.65% | 29.73% | 71.50% | 72.73% | 67.96% | 47.97% | 57.86% |
| w075 | vita-test | 14.88% | 92.88% | 38.52% | 60.51% | 35.47% | 20.31% | 3.85% | – |
| w075 | psp-test | 26.34% | 93.95% | 32.57% | 71.50% | 76.08% | 72.82% | 50.30% | 59.30% |
| h24 | vita-test | 16.47% | 91.42% | 38.93% | 58.97% | 37.21% | 26.56% | 5.33% | – |
| h24 | psp-test | 26.90% | 92.84% | 32.84% | 72.43% | 76.55% | 74.27% | 49.80% | 59.51% |
| base_s1 | vita-test | 12.35% | 88.69% | 32.56% | 54.36% | 27.33% | 17.19% | 3.40% | – |
| base_s1 | psp-test | 23.76% | 88.09% | 29.32% | 68.22% | 71.77% | 65.05% | 45.83% | 57.35% |
| base_s2 | vita-test | 14.04% | 89.05% | 37.41% | 57.44% | 30.23% | 21.88% | 2.51% | – |
| base_s2 | psp-test | 24.39% | 93.30% | 31.08% | 71.03% | 72.73% | 67.96% | 48.88% | 58.17% |
| v1 | vita-test | 8.04% | 82.12% | 27.00% | 46.15% | 21.51% | 7.81% | 1.33% | – |
| v1 | psp-test | 16.79% | 86.79% | 20.00% | 64.95% | 63.64% | 45.15% | 44.71% | 56.42% |
<!-- END styles -->

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
| base_s1 |  | 15.91% | 15.95% | 15.95% | 1.21% | 1.21% |
| base_s2 |  | 17.00% | 16.98% | 16.98% | 1.23% | 1.23% |
| h24 |  | 19.19% | 19.27% | 19.27% | 2.46% | 2.46% |
| v1 |  | 10.45% | 10.48% | 10.48% | 0.29% | 0.30% |
| v3500 |  | 16.10% | 16.15% | 16.15% | 1.32% | 1.33% |
| w050 |  | 21.34% | 21.50% | 21.50% | 2.77% | 2.79% |
| w075 |  | 18.74% | 18.81% | 18.81% | 1.71% | 1.71% |
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
  Single-seed differences under about 1–2 points are within training noise.
- vita-test-inv and psp-test-inv use the style tags as a polarity oracle. The deployed
  pipeline needs the line finder or the training data to handle dark text. Two psp-test
  lines look mis-tagged (reported on PR #5); they count against every model equally.
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
