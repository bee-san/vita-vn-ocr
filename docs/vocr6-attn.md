# vocr6-attn: a light global-attention block in the recognizer (VOCR6 stage 3, #125)

VOCR6's stage 3 spends the time that the head gate (#120) and the two-level head (#122) freed on capacity. Until now
the recognizer saw only ±(ctx·(k−1)/2) frames of temporal context (±4 for VOCR5). This adds one light global
self-attention layer over the whole line's frames, behind a flag, end to end: the model and `train2.py --attn 1`, the
blob (a new op), the C runtime in all three modes, the twin and parity tests, and the cost.

Nothing here is adopted. This builds the capability and measures what it costs. The lead's pre-registered stage-3
arm decides whether VOCR6 uses it.

**Result.**

- **Exact.** The C runtime equals the bit-exact numpy twin (`runtime/tests/vocr_ref.py`) in f32, W8F and W8I, at every
  tile and head-batch size, scalar and NEON (Cortex-A9 under qemu-arm), under ASan and UBSan with gcc and clang. It
  agrees with PyTorch within `parity.py`'s tolerances: max normalised logit difference 5.7e-6 (f32), 8.3e-6 (W8F),
  6.2e-4 (W8I), with no text difference on the 40 fixture lines.
- **Off is today.** `--attn 0` (the default) leaves every model, checkpoint and blob as it was. The release blobs
  VOCR5-S/N/L re-export from their checkpoints byte for byte, and the pinned blob hashes of
  `tests/test_model_default_equivalence.py` hold. A VOCR5 checkpoint loaded into an attention model starts at gate 0
  and gives bit-identical logits, in PyTorch and through the C runtime.
- **Size.** +37,153 trained parameters and +40,000 B of int8 blob per tier. That is the same for every tier, because
  every tier's bottleneck is 96 wide. Large would go from 2,473,696 to 2,513,696 B, over its 2.5 MB cap ("Limits").
- **Arena.** +49,152 B for Normal at 1,024 px in W8I: the line of bottleneck outputs (128 frames × 96 floats). The
  attention's other buffers fit in memory the tiles leave idle.
- **Time (estimates).** The op costs about 0.15 ms a frame on a 640-px line, at 333 MHz on one Cortex-A9 without
  prefetch credit. That is **+35 ms per nominal 3-line box for Normal** on top of the adopted gate + two-level head
  (758 → 794 ms; +26 ms at 444 MHz), and +28 ms for Small. That is 22-23% of the 122-160 ms the speed work freed.
- **Proxy (a smoke test, not a decision).** A 15k-step re-fine-tune of VOCR5-N with the block against the same
  run without it, 1 seed each: the block trains (its gate moves from 0 to 0.039). teacher_val_dev reads 1.434 /
  1.508% against the control's 1.572 / 1.551%. The dev score (W8I, Rubi's decode) is level: 2.042 against 2.049,
  −0.007 points [−0.133, +0.117]. "Proxy" below.

Nothing ran on a console.

## Design

The block sits on the 96-wide bottleneck output, right before the head:

```
z  = bottleneck(features(x))                                   N x 96 x 1 x T
z' = z + g · O(concat_h softmax(Q_h(z) K_h(z)^T / sqrt(dh)) V_h(z))
logits = head(z')
```

- **Projections.** Q, K, V (96 → `--attn-dim`, default 96) and O (dim → 96) are 1×1 `QConv2d`, so the int8 export
  and QAT treat them like every other pointwise conv. The dim is split into `--attn-heads` heads (default 2, so
  dh = 48). dh must be a multiple of 8, the runtime's block width.
- **No positional embedding.** The depthwise temporal convolutions before the block give position, as conditional
  positional encodings do (Chu et al. 2021, arXiv:2102.10882).
- **No LayerNorm.** Instead g is a learnt scalar initialised to 0 (ReZero: Bachlechner et al. 2020, arXiv:2003.04887).
  A model fine-tuned from a VOCR5 checkpoint therefore starts as exactly VOCR5: `model.load_state` of a state_dict
  without the block gives bit-identical logits and gradients for every shared parameter (tested), and g's own
  gradient is non-zero from the first step. (At g = 0, z' = z + 0: an exact −0 in z would become +0, which no logit
  can see unless the head's own sum is exactly zero there; W8I's int16 grid maps both to 0. Every tested line and
  all 24 timing lines read identically.)
- **Padding.** The batch's right padding (frames ≥ the line's `ceil(w / 8)`) is masked out of the keys in training.
  `train2.py` passes `lengths` to the model. A line read alone (the runtime, train2's validation, `eval/run.py`) has no
  padding.
- **One deviation from the brief: K has no bias.** A K bias b adds q·b to every score of query q, the same for every
  key, so the softmax cancels it. Dropping it changes no function and saves 96 parameters. The blob still stores the
  QKV bias as 3 × dim floats, with K's rows 0.
- **Initialisation.** The block is built after the head, so every other module draws the same initial values for a
  seed as a model without it.

Checkpoints record `attn`, `attn_dim` and `attn_heads`. These all rebuild the block from them: `model.from_checkpoint`,
`export.load_checkpoint` (and so `pyvocr.Vocr.from_checkpoint`, `parity.py --ckpt`, `vita/`'s tools), `eval/run.py`'s
PyTorch backend and `train2.py --init` / resume. Checkpoints without the keys (every one before this) have no block.
`train2.py` refuses to resume across a change of `--attn`, and refuses to start an `--attn 0` run from an attention
checkpoint. `--rep` and `--qat` cover the block: it has no depthwise conv, and its four projections are int8 convs.

## The blob: op type 5 (ATTN)

The full layout is in `train/export.py`'s docstring. In short:

- **Position.** At most one ATTN op, right before the head, on one row, with cin = cout = d (96).
- **Record.** Its matrix fields hold QKV: `[3 dim / 8][d][8]`, cout_pad = 3 dim, f32 or int8 with per-row scales,
  bias always. 1 / sqrt(dh) is folded into Q's rows and bias.
- **O.** Bytes 36-55 hold the heads and dim, then O's weight, bias and scale offsets: `[d / 8][dim][8]`, with g folded
  into O's weights and bias.
- **int8.** Per-row int8 commutes with both folds, so `--qat` sees exactly the exported weights.
- **Compatibility.** `vocr_load` validates the op strictly (position, shape, flags, every offset and size, the reserved
  bytes) and refuses two of them. A runtime from before this refuses such a blob (an unknown op type). A blob without
  the op is byte for byte what the exporter wrote before.

`export.py --info` prints the op (`attn: {heads, dim}`), and its MACs per line include the T-dependent scores.

## The runtime

`runtime/vocr.c` streams the line in tiles up to the bottleneck, as before.

- **The line of z.** With an ATTN op, the bottleneck writes every frame's output into a line buffer of
  `t_max × d` floats, where t_max = max_width / stride. That buffer is the only part of the arena that grows with the
  width.
- **The attention.** After the last tile, `attn_line` runs over the whole line:
  1. K and V of every frame, in chunks of the head batch (16 frames by default). The QKV matrix's K and V rows run in
     the blob's mode, exactly as a pointwise layer runs in that mode: f32, W8F (dequantised blocks) or W8I (int16
     activations, exact int32 sums). K is stored per head in blocks of 8 keys, so that 8 scores are one
     `mm_f32` block.
  2. Per chunk of queries: Q; then per head the scores against all T keys (float32, in a fixed order), the softmax and
     the weighted sums of V. Then O and the residual (z' = O(A) + z), in the blob's mode, in place.
- **The softmax** is float32 in every mode, with the runtime's own `exp_neg` (float arithmetic only, no libm; relative
  error < 2e-5). It takes the first maximum, sums `exp_neg(s − max)` in order and multiplies by the sum's reciprocal.
- **The head.** It then streams over z' as it streamed over z: the head gate's per-tile step, its batches of 16, the
  two-level head. The head's input is z'.
- **Order.** Every value is computed in one fixed order, whatever the chunk, tile or build. So the logits are
  bit-identical across tile and batch sizes, scalar and NEON (and the PSP's VFPU path, which `mm_f32` already shares).
- **Arena.** K, V, a chunk's Q, scores and outputs, and the projections' scratch overlay the rings, the head's
  accumulator and the transients. All of those are idle between the tiles and the head. The line of z comes after
  everything. `vocr_arena_bytes()` and `train/export.plan_arena()` report the total exactly. No malloc, nothing new
  from libc.
- **Cancel.** The callback is also polled once per chunk of the attention and once per tile of the head phase after
  it.
- **Profile.** `VOCR_PROF` marks the projections under the op's index and the scores, softmax and weighted sums as
  phase 103.
- **API.** `vocr_model.attn` (heads, dim, O's arrays; dim 0 = none) and nine `vocr_ctx` plan fields are new. ARM
  32-bit: `sizeof(vocr_model)` goes from 1,848 to 1,872 B and `sizeof(vocr_ctx)` from 2,012 to 2,048 B. `vocr_opts`
  and `vocr_extra` are unchanged.
- **Code and stack.** `vocr.o` grows from 23,267 to 25,979 B (+2,712 B; VitaSDK gcc `-O2`, NEON). The scalar build
  goes from 18,295 to 21,079 B, and psp-gcc `-O2 -G0` from 28,091 to 32,603 B (VFPU build 28,607 to 33,027 B). `attn_line` is inlined into `vocr_run`, whose frame grows from 592 to 600 B (scalar 592 to 616 B).
  The deepest chain is still `vocr_arena_bytes` → `plan()` (2,736 → 2,800 B, a local `vocr_ctx`); `vocr_run`'s
  deepest (→ `run_head` → `mm_i16`) goes from 2,528 to 2,536 B.

### The head gate and the two-level head

Both read the head's input, which with attention is z'. They need no change: `gate_tile` and `run_head2` take their
frames from the head's batches, which the attention phase fills from z'.

- **Head gate.** `tests/test_vocr6_attn.py` fits a gate on an attention model's own z' frames. The C gate scores then
  equal `train/headgate.py`'s twin on `vocr_extra.head_in`, and head_gate on equals the ungated read with the skipped
  frames as blanks, at four tilings.
- **Two-level head.** A permuted attention blob with a random escape equals `train/head2level.py`'s twin: escape
  scores, fired frames, argmax and top-K.
- **C tests.** `runtime/tests/test_vocr.c` runs its gate and two-level suites again on attention blobs, including one
  whose head follows the stem.
- **Refitting.** A model trained with attention needs its own gate and escape. `headgate.py` and `head2level.py`
  extract their training frames through the C runtime's `head_in`, so they work on an attention blob unchanged.

## Sizes and arena

Measured on VOCR5-S/N/L's own checkpoints (the released runs), with the op at dim 96 and 2 heads; `eval/vocr6_attn_cost.py
sizes`.

| | S (width 1.0) | N (1.5) | L (late-wide 2.0 / 3.0) |
|---|---:|---:|---:|
| trained parameters | 726,075 → 763,228 | 1,068,687 → 1,105,840 | 2,324,287 → 2,361,440 |
| int8 blob (`_w8`) | 819,792 → 859,792 B | 1,185,632 → 1,225,632 B | 2,473,696 → 2,513,696 B |
| adopted speed blob (gate; N, L + two-level) | 821,584 → 861,584 B | 1,189,408 → 1,229,408 B | 2,477,472 → 2,517,472 B |
| MACs per frame (projections) | 967,392 → 1,004,256 | 1,591,488 → 1,628,352 | 3,402,720 → 3,439,584 |
| MACs per frame on an 80-frame line | → 1,019,616 (+5.4%) | → 1,643,712 (+3.3%) | → 3,454,944 (+1.5%) |

- **Per tier.** The op adds 37,153 trained parameters (37,248 stored, with K's zero bias) and 40,000 B to an int8 blob:
  the record, QKV 27,648 B, O 9,216 B, scales and biases 3,072 B.
- **MACs.** The projections are 4 × 96 × 96 = 36,864 MACs a frame. The scores and weighted sums add 2 × T × 96 a frame
  on a line of T frames: 15,360 at T = 80 (640 px), 24,576 at T = 128 (1,024 px).
- **Arena.** The table gives exact bytes (`vocr_arena_bytes` = `plan_arena`, tested), at the default tile 4 and head
  batch 16, as today → with the op.

| tier | mode | 256 px | 640 px | 1,024 px |
|---|---|---:|---:|---:|
| S | W8I | 124,832 → 137,120 (+12,288) | → 155,552 (+30,720) | → 173,984 (+49,152) |
| S | W8I + gate | 124,880 → 137,168 | → 155,600 | → 174,032 (+49,152) |
| S | W8F / F32 | 118,432 / 116,384 → 130,720 / 128,672 | → 149,152 / 147,104 | → 171,264 / 168,192 (+52,832 / +51,808) |
| N | W8I | 182,944 → 195,232 (+12,288) | → 213,664 (+30,720) | → 232,096 (+49,152) |
| N | W8I + gate + two-level | 184,448 → 196,736 | → 215,168 | → **233,600 (+49,152)** |
| N | W8F / F32 | 173,472 / 170,400 → +12,288 | +30,720 | +49,152 |
| L | W8I | 261,024 → 273,312 (+12,288) | → 291,744 (+30,720) | → 310,176 (+49,152) |
| L | W8I + gate + two-level | 262,528 → 274,816 | → 293,248 | → 311,680 (+49,152) |
| L | W8F / F32 | 257,696 / 244,384 → +12,288 | +30,720 | +49,152 |

- **What grows.** The delta is the line of z, `t_max × 96 × 4` B: 12,288 / 30,720 / 49,152 at 256 / 640 / 1,024 px.
  K and V of every frame (2 × t_max × 96 floats), a chunk's queries, scores and outputs fit in the memory the tiles
  leave idle, except for Small at 1,024 px in W8F and F32, whose rings are the smallest: +3,680 / +2,656 B beyond the
  line.
- **Against the target.** The brief's target was ≤ +100 KB for N at 1,024 px. The measured +49,152 B leaves room, so K
  and V stay float32: compacting them would cost precision and code for no arena gain at these shapes.
- **Batch size.** A larger head batch (`vocr_opts.head_frames`) grows the query chunk's buffers too (16 → 128 frames:
  scores 8 → 64 KB). The plan reports it exactly.

## Time (estimates)

ocr-speed's method, as docs/vocr5.md, docs/headgate.md and docs/head2level.md used it
(`eval/reproducibility/vocr6-attn-20261026/a9.sh`, `eval/vocr6_attn_cost.py time`):

- **Instruments.** `vita/a9_layers.py` instruction counts of the VitaSDK `-O2` NEON build under qemu-arm, with its cache
  model.
- **Lines.** ocr-speed's 24 development lines of 60-100 frames (1,609 frames), top-8.
- **Calibration.** The high bound × 1.000676 (VOCR3-H15's measured console rate at an assumed 333 MHz). 444 MHz is by
  the clock ratio.
- **What was compared.** Per tier, the adopted speed blob against the same blob with the op at gate 0. The adopted
  blobs are S with the head gate, and N and L with the gate and the two-level head (K = 500, r96). Both blobs were
  rebuilt from the checkpoints, and the rebuilt adopted blob equals the released one byte for byte.
- **Why gate 0.** At gate 0, z' = z exactly, so the gate and the escape skip and fire on the same frames (the texts of
  all 24 lines are identical). The difference is the op's own cost. Its arithmetic does not depend on its weights'
  values.
- **Runtime measured.** `vocr.c` `b2509902…` (the record's `vocr_c_sha256`). The later commits change only `plan()`'s
  overflow checks and a comment, which no frame runs (the auditor's re-run of one pair gave identical per-phase
  counts, apart from 64-187 instructions of integer division in the setup phase).
- **Projection.** The scores and weighted sums cost T per frame. Their phase is projected from the pack's lines
  (Σ T² = 108,719) to the nominal box's 3 lines of 80 frames. The nominal box is 240 frames plus 0.1 s for the finder.

| tier | basis | ms per frame: adopted → with the op | op: projections + scores at T = 80 | nominal box at 333 MHz | at 444 MHz |
|---|---|---:|---:|---:|---:|
| S | no prefetch credit | 1.746 → 1.852 | 0.066 + 0.080 | 519 → 547 ms (+28) | 389 → 411 ms (+21) |
| S | with prefetch credit | 1.528 → 1.630 | 0.061 + 0.080 | 467 → 494 ms (+27) | 350 → 371 ms (+21) |
| **N** | no prefetch credit | 2.743 → 2.877 | 0.066 + 0.080 | **758 → 794 ms (+35)** | 569 → 595 ms (+26) |
| N | with prefetch credit | 2.328 → 2.457 | 0.060 + 0.080 | 659 → 693 ms (+34) | 494 → 520 ms (+26) |
| L | no prefetch credit | 7.036 → 6.954 | 0.072 + 0.080 | 1,789 → 1,772 ms (−17) | 1,341 → 1,329 ms (−12) |
| L | with prefetch credit | 5.010 → 5.138 | 0.063 + 0.080 | 1,302 → 1,336 ms (+34) | 977 → 1,002 ms (+25) |

- **The op's own cost.** The same in every tier, since its shape is: about 0.066 ms a frame for the projections, plus
  0.0010 ms × T for the scores, softmax and weighted sums. On a line of T frames that is T × 0.066 + T² × 0.0010 ms
  (N, 333 MHz, no prefetch credit): 12 ms for an 80-frame line, 25 ms for a 128-frame one.
- **Against the speed work.** Normal's +35 ms per box is 22% of the 160 ms the gate + two-level head freed (est.,
  docs/head2level.md). Small's +28 ms is 23% of the gate's 122 ms.
- **Large without prefetch credit gets faster (−17 ms).** With the op, the head runs after all the tiles instead of
  between them. Large's weights (2.5 MB) are larger than the 2 MiB L2, so in the cache model the head's 540 KB then
  stops evicting the convolutions' weights every tile: the 2-row and frame-row pointwise layers' L2 misses drop by 21-34%, and
  the head's own cost falls 0.44 → 0.39 ms a frame. With prefetch credit the effect disappears (+34 ms).
- **This is not an attention gain.** It comes from the head running once per line. A runtime without the op could get
  the same by batching the head over the whole line (`head_frames ≥ t_max`). That is noted here and not measured
  further.

## Parity and tests

- **`runtime/tests/parity.py --fixture --attn 1`.** A random-init width-0.5 model with the op: gate 0.8, and Q and K ×
  4, so the softmax is peaked and the block changes the argmax on about half the frames. Its 40 fixture lines (grey and
  RGBA, 1-1,400 px) pass in all three modes:

  | mode | logits bit-exact to the twin | max normalised logit difference to PyTorch (tolerance) | same text as PyTorch |
  |---|---:|---:|---:|
  | f32 | 40 / 40 | 5.7e-6 (1e-3) | 40 / 40 |
  | W8F | 40 / 40 | 8.3e-6 (1e-3) | 40 / 40 |
  | W8I | 40 / 40 | 6.2e-4 (5e-3) | 40 / 40 |

  On those lines it also checks preprocessing, tile and batch invariance (6 sizes from 1/1 to 128/128), `VOCR_INVERT`
  and the arena plan at 6 widths. PyTorch here is the original model in f32, and a PyTorch forward of the int8 blob in
  W8F and W8I.
- **CI's `make -C runtime check`.** It runs the same parity, 24 lines, through the ASan + UBSan `vocr_cli` (gcc and
  clang, `ci/runtime.sh`). `make parity-arm` passes it through the Cortex-A9 NEON and scalar CLIs under qemu-arm.
- **`runtime/tests/test_vocr.c`.** The op is inserted into the C test blobs:
  - the loader's validation: 16 field edits, two ops, the op last, an int8 op in an f32 blob;
  - O = 0 gives the model without the op bit for bit;
  - logits invariant across 6 tilings and a narrower `max_width`; W8I close to W8F;
  - a cancel at every poll;
  - known-answer hashes (host gcc and clang and Cortex-A9 NEON and scalar under qemu give the same);
  - the gate and two-level suites on attention blobs.
- **`tests/test_vocr6_attn.py` (21 tests, CI's pytest).**
  - the model: off = no block, initialisation, refused shapes, gate 0 = old logits and gradients, masking;
  - `train2.py --attn` from an old checkpoint, with resume and the refused combinations;
  - the export's folds and sizes;
  - the C runtime against the twin and PyTorch at three shapes;
  - gate 0 through the C runtime equals the blob without the op;
  - the arena mirror over 7 tilings × widths × modes;
  - the head gate and the two-level head on z';
  - the checkpoint tools;
  - VOCR5-S/N/L's checkpoints still export to the v0.3.0 blobs (local only), and the committed cost records.
- **The tools.** `vita/a9_layers.py` names the op and phase 103. `eval/vocr6_attn_cost.py` checks that VOCR5-S/N/L
  re-export to the released blobs and that the adopted speed blobs rebuild byte for byte.

## Proxy (a smoke test, not a decision)

One short run so the lead knows the block trains. **This is not a selection, and nothing is decided on it.** It is
1 seed per arm, logged on teacher_val_dev as VOCR5 did. Never read: vita-test, psp-test, CLANNAD and the hooked test
titles.

Pre-registered in `vocr6/experiments.jsonl` (09:33Z). `eval/vocr6_attn_proxy.sh ATTN`:

- **Recipe.** A second 15,000-step fine-tune of VOCR5-N from `v5f_N_seed1/ckpt_step15000.pt`, with VOCR5-N's own
  fine-tune recipe and mix (`eval/vocr5_run.sh N F`: its 12 sources, the same `--src` list as v5f_N_seed1's
  `args.json`). lr 1.2e-3, warmup 500, cosine to 5%, batch 96, EMA 0.999, seed 1.
- **Arms.** `--attn 1` (dim 96, 2 heads, from gate 0) against the same run with `--attn 0`.

**Results** (measured; `eval/vocr6_attn_proxy_20261026.json`). Dev = the lead's `eval/vocr6_score.py`
(`feat/vocr6-recipe`), dev plan only: VOCR5's development score 0.25 tv + 0.25 dn + 0.5 wd in the console view, W8I,
Rubi's decode without the cascade, no head gate, the hooked titles' VNs (v12188, v18432) left out. The init's row is
the lead's own score of the same blob.

| run | teacher_val_dev CER, rv / rvn (train2's log: float, EMA, greedy) | dev score, W8I + Rubi's decode (tv / dn / wd) | dev score, W8I greedy |
|---|---:|---:|---:|
| init: `v5f_N_seed1` step 15000 (VOCR5-N, seed 1) | 1.561 / 1.561% | 2.098 (0.675 / 0.773 / 3.472) | 2.309 |
| a0: + 15k steps, `--attn 0` | 1.572 / 1.551% | 2.049 (0.612 / 0.836 / 3.373) | 2.305 |
| a1: + 15k steps, `--attn 1` | 1.434 / 1.508% | 2.042 (0.650 / 0.698 / 3.409) | 2.290 |

Paired differences in the dev score, in CER points, with 95% intervals from the lead's `eval/vocr6_select.py`
bootstrap over VNs (10,000 resamples):

- a0 → a1, the block (a1 against its control a0): dev score −0.007 points [−0.133, +0.117] with Rubi's decode (tv
  +0.038, dn −0.137, wd +0.035); greedy −0.015 [−0.100, +0.065]; teacher_val_dev rv −0.138, rvn −0.042
- init → a0, 15k more fine-tuning steps without it: dev score −0.050 points [−0.137, +0.036] with Rubi's decode (tv
  −0.064, dn +0.062, wd −0.098); greedy −0.004 [−0.078, +0.071]; teacher_val_dev rv +0.011, rvn −0.011
- init → a1, 15k more steps with it: dev score −0.057 points [−0.150, +0.036] with Rubi's decode (tv −0.025, dn
  −0.075, wd −0.063); greedy −0.019 [−0.108, +0.071]; teacher_val_dev rv −0.127, rvn −0.053

- **It trains.** The learnt gate moves from 0 to 0.039. On 40 teacher_val_dev lines the block's output g·O(·) is a
  median 15% of z's norm (8-21%). Its attention is sharply peaked (mean largest weight 0.97), on keys a mean 10.9
  frames from the query, beyond the ±4 frames of VOCR5's temporal convolutions.
- **The trained checkpoint is exact.** `parity.py --ckpt` on a1's final checkpoint, 120 lines of synth2's held-out
  shard: bit-exact to the twin in f32, W8F and W8I; max normalised logit difference to PyTorch 5.1e-6 / 3.7e-6 /
  3.3e-4; the same text as PyTorch on every line; tiles, invert and arena pass.
- **What it shows.** The block trains. On teacher_val_dev a1 reads 0.14 / 0.04 points better than a0. On the dev
  score, the shipped form, the two are level: −0.007 points [−0.133, +0.117] (tv +0.04 and wd +0.04 worse, dn −0.14
  better, none resolved). One seed of a 15k re-fine-tune cannot resolve gains of this size. The arms also do not share their
  dropout draws: the block's initialisation draws from torch's global RNG, so every later dropout mask differs (the
  batches are the same, from numpy's generator). The stage-3 arm decides, with its own seeds and the hval page path.
- **Cost of the runs.** a1 09:35-11:58Z (2.4 slot-hours), a0 12:34-15:29Z (2.9 slot-hours, on a busier machine).

## Limits

- **Large's blob size.** At dim 96, Large's blob is 2,513,696 B, over its 2.5 MB cap (2,500,000 B) by 13,696 B. Two
  ways to fit:
  - `--attn-dim 48` (1 head of 48, or 2 of 24): +20,416 B and 18,625 trained parameters, so 2,494,112 B fits.
    `--attn-dim 64` (+26,944 B, 2,500,640 B) does not.
  - Taking ~14 KB from elsewhere in L.

  Small (< 1 MB) and Normal (≤ 1.6 MB) have room.
- **Estimates.** Every millisecond is an estimate from qemu instruction counts with a cache model, calibrated on one
  model (H15) in one game at an assumed 333 MHz. Nothing of this has run on a console.
- **PSP.** The PSP build (psp-gcc, scalar and `VOCR_PSP_VFPU`) compiles with the op, and the op uses only kernels
  the PSP paths already run (`mm_f32`, `mm_i16`, `dequant8`, `quant16`). The PPSSPP parity of an attention blob
  (psp/README.md) was not run here; the PSP model's own work (VOCR6 model B) runs it if it uses the op.
- **Longer lines.** The op's cost grows with T² per line, the only part of the recognizer that does: a 1,024-px line
  (128 frames) costs about 25 ms in the op (est., N, 333 MHz), against about 12 ms for 640 px.
- **Turning it off.** The op is part of a blob, not a runtime option: a model trained with it reads z', so it cannot
  run without it. The off setting for a console is the model choice (a blob without the op), and a runtime without
  the op support refuses an attention blob instead of misreading it.
- **Memory.** The line buffer makes the arena depend on `max_width`. A caller that plans for narrower lines
  (`vocr_opts.max_width`) gets a smaller arena, and `VOCR_E_WIDTH` for a wider line, as today.
- **Gate and escape refit.** The released gate and escape were fit on VOCR5's z. A model trained with attention needs
  its own (the tools work unchanged). The time estimates above assume the refit gate skips as many frames as VOCR5's.

## Reproduce

```sh
H=/local/home/skerraut/herd-vita-ocr; D=$H/shared/artifacts/vocr6-s3-attn
# exactness (CI's runtime job does the first two; make parity-arm the third)
make -C runtime check CC=gcc        # or: PATH=$PWD/.venv-ci/bin:$PATH setarch -R ci/runtime.sh gcc (and clang)
(cd runtime && ../.venv-ci/bin/python tests/parity.py --fixture --attn 1 --n 40)
make -C runtime check-arm parity-arm PYTHON=../.venv-ci/bin/python PARITY_ARGS="--attn 1 --n 24 --tile-lines 12"
.venv-ci/bin/python -m pytest -q tests/test_vocr6_attn.py
# sizes, arena, the a9 blobs and time (one heavy slot)
.venv-ci/bin/python eval/vocr6_attn_cost.py sizes --out $D/cost/sizes.json
.venv-ci/bin/python eval/vocr6_attn_cost.py blobs --out $D/cost/blobs
HEAVY_SLOTS=12 $H/heavy.sh eval/reproducibility/vocr6-attn-20261026/a9.sh $D/cost/blobs $D/cost/a9
.venv-ci/bin/python eval/vocr6_attn_cost.py time --a9 $D/cost/a9 \
  --pack $H/shared/artifacts/ocr-speed-resume/packs/sub24.bin --out $D/cost/time.json
# the proxy (one heavy slot each)
HEAVY_SLOTS=12 $H/heavy.sh timeout 36000 eval/vocr6_attn_proxy.sh 1 1 $D/runs
HEAVY_SLOTS=12 $H/heavy.sh timeout 36000 eval/vocr6_attn_proxy.sh 0 1 $D/runs
# their dev scores: eval/vocr6_score.py and eval/vocr6_select.py from origin/feat/vocr6-recipe, the dev plan only
for a in a0 a1; do R=vocr6attn_N_${a}_s1; B=$D/blobs/${R}_w8.vocr; W=$H/shared/artifacts/vocr5-analysis/work
  .venv-ci/bin/python train/export.py --ckpt $D/runs/$R/ckpt_step15000.pt --out $B --int8
  HEAVY_SLOTS=12 $H/heavy.sh .venv-ci/bin/python eval/vocr6_score.py read --plan $W/plan.pkl --blob $B \
    --lm $H/shared/artifacts/ctc-lm-decode/release_pub2/vocr_lm_o3_1000k.vlm --out $D/scores/dump_dev_$a.json
  .venv-ci/bin/python eval/vocr6_score.py score --name $R --blob $B --rubi-norm $H/shared/artifacts/vocr6-lead/tools/rubi_norm \
    --dev-plan $W/plan.pkl --dev-units $W/units.pkl --dev-dump $D/scores/dump_dev_$a.json --exclude-vns v12188 v18432 \
    --out $D/scores/score_$a.json
done
```

The committed records are `eval/vocr6_attn_sizes_20261026.json` and `eval/vocr6_attn_time_20261026.json` (the two
`cost/` files above) and `eval/vocr6_attn_proxy_20261026.json` (the proxy: the runs' logs, the dev scores and their
paired intervals, the trained block's diagnostics and parity).
