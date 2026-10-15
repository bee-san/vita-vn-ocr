# OCR speed: faster W8I kernels, bit for bit (ocr-speed)

The owner's v0.8.0-test.1 console logs (vita-jp-overlay #81) are the ground truth: VOCR3-H15 ran its lines at
**4,726-5,174 µs per CTC frame (frame-weighted 4,942 µs) on the log's ten lines of 60-100 frames**, and its 11
jobs at 4,941-5,238 µs per frame (frame-weighted 5,068 µs: recognize time over frames, which in the one
five-line job includes 258 µs per frame of strip reads outside `vocr_run`). That is 0.2-0.4 s per textbox line.
This change makes the same arithmetic cheaper on the Cortex-A9. Every logit is bit-identical to `main`'s (W8I and
W8F, greedy and with the LM's top-K, five builds, three released models, 884 development lines), and the builds
without NEON (the PSP, scalar ARM, the hosts) compile to `main`'s exact machine code.

**Estimated effect (calibrated to the console, below), ms per CTC frame at 333 MHz, `main` → this PR, with the
LM's top 8 (the overlay's default decoding) and greedily:**

| model | top-8 (LM) | greedy | without the prefetch model (top-8) |
|---|---|---|---|
| VOCR3-FL10 | 3.01 → 1.71 (1.76x) | 2.93 → 1.64 (1.78x) | 1.94 (1.55x) |
| VOCR3-H15 | 5.07 → 2.90 (1.75x) | 4.95 → 2.81 (1.76x) | 3.36 (1.51x) |
| VOCR4-W20 | 6.94 → 4.08 (1.70x) | 6.82 → 3.99 (1.71x) | 4.83 (1.44x) |

A 3-line box of 640-px lines (240 frames) of H15 with the LM goes from 1.22 s to 0.70 s of recognition at 333 MHz
(0.81 s without the prefetch model), 0.52 s at 444 MHz. These are estimates; the console has not run this code.

## What changed (runtime/vocr.c)

| kernel | before | after | why the bits are the same |
|---|---|---|---|
| `mm_i16` 4 positions x 8 outputs (every pointwise layer and the head) | 4 k per pass, two 64-bit weight loads and a VMOVL per k, 8 int32 accumulators | 8 k per pass: one 128-bit load per position, the 8 weight rows in two 256-bit loads (`vld1q_s8_x2`), 64 VMLAL per pass; a `PLD` 128-160 B ahead on the weight stream; the single-chunk case (K ≤ 512, every VOCR layer but W20's collapse) skips the float chunk adder | integer sums are exact in any order (a chunk is at most 512 x 127 x 32,767 < 2^31); the float steps (convert, `+` chunk sums, `* (scale[j] * sx[p])`, `+ bias`, relu, `+ res`) are the same operations in the same order; for one chunk `0 + float(c)` is `float(c)` (an int's float is never -0 or denormal) |
| `mm_i16` 1-3 leftover positions (head batches of 17-19 frames, line ends) | plain C, scalar VFP | NEON, 4 accumulators per position, added exactly at the chunk's end | the same for finite normal values; NEON flushes a denormal to zero and `vmaxq_f32(NaN, 0)` is NaN where the scalar relu gives 0. `main`'s own 4-position NEON path already behaved so, and no development line produces either |
| `run_head` argmax, W8I with NEON, greedy | one VFP compare per class (5,355 per frame for H15), reloading the logits | `mm_i16` on 32 frames per call; the running maximum per lane in NEON with an integer key of the float (`fkey`: magnitude bits, negated for negatives, NaN below everything); the first-maximum tie rule finished per frame (`lanes_argmax`) | the key orders floats exactly as `v > best` under IEEE compares (FPSCR.FZ = 0; -0 equals +0, a NaN is never kept); ties go to the lower class, as the scalar scan; a partial last block (5,355 = 669 x 8 + 3) still runs the scalar scan after the lanes, in class order |
| `run_head` with the top K (the LM's input, vita-vn-ocr #82), W8I with NEON | the per-class insertion for every class | `mm_i16` on 32 frames per call; a block of 8 classes goes through the per-class insertion only when one of its logits is above the frame's K-th largest so far (`fkey` again) | that value only grows while classes are inserted in order, and the argmax is never below it (it is `[0]`): a block with no logit above it is one the insertion leaves as it is. The logits and the blank logit are copied out of every block as before |
| `k_dw` / `k_conv1` (stem and depthwise) | index arithmetic per tap, 128-bit loads | pointer stepping per tap row, 256-bit loads (`vld1q_f32_x2`), 3-wide taps unrolled, the stem's input value loaded into a lane (`vld1_dup_f32`) | the same multiply-adds in the same order per output (`vmlaq` lane by lane, `-ffp-contract=off`) |
| `ring_shift` | float by float | 4 floats at a time, still forwards | a copy |

Only the NEON builds change. `VOCR_NOINLINE` (the kernels kept out of `step_op`, whose loop state gcc 14/15 would
otherwise spill) applies to NEON builds only, and `run_head`'s lane and top-K paths are under `VOCR_NEON`, so every
other build compiles `main`'s code token for token: the `.text` of host gcc 11.5 and clang 18.1.8, Arm GNU 14.2
`-DVOCR_NO_NEON` and psp-gcc 15.2 (VFPU 0 and 1) objects is byte-identical to `main`'s (a540027). With the Vita
compiler (arm-vita-eabi-gcc 15.2, the flags of `vita/CMakeLists.txt`, with or without `-mthumb`), `vocr.o`'s
`.text` goes from 15,284 to 17,948 B. `run_head`'s frame grows from 376 to 1,328 B (the 32-frame logit block), so
the deepest chain is `vocr_run` 448 + `run_head` 1,328 + `mm_i16` 616 = 2,392 B (`-fstack-usage`). No arena or
API change: `vocr.h` is untouched and `vocr_arena_bytes()` is the same for every model and option.

## Before / after per layer (VOCR3-H15, greedy)

`vita/a9_layers.py` (new): the Vita compiler's code for `runtime/vocr.c` with `-DVOCR_PROF`, which makes
`vocr_run()` call a marker as each phase starts, under qemu-arm with `vita/qemu/insn_mix.c` counting instructions
per phase (`marker=`, `layers=`), its data cache model (`l1d=on`: 32 KiB 4-way L1, 2 MiB 8-way L2, LRU) and the
`PLD` sites (`pld=`, which qemu itself runs as no-ops; `--no-pld` leaves them out); then `vita/a9_cycles.py`'s
Cortex-A9 cost model per phase. Without the marker the same lines give the same texts (`--plain`). The lines: 24
development lines of 60-100 CTC frames (1,609 frames, sha256 `3162151a…`) from `vita/dev_pack.py` (below).

**Calibration.** The console's per-line rate on lines of that length, 4,942 µs per frame, is 1,645,586 cycles at
333 MHz. The code the console ran (vita-vn-ocr a5d5aad, the overlay's pin for v0.8.0-test.1, whose `vocr.c` is
`d1fe1f7b…`) gives 1,644,475 cycles per frame on these lines as the model's high bound (no dual issue, every miss
at full latency), so the console ran at **1.0007 x the high bound**. Every "ms" here is the high bound x 1.0007 at
that clock: an estimate. 333 MHz is an assumption (the overlay's default; the log does not record the clock),
supported by the console landing on the high bound at that clock. The factor comes from one model (H15) and one
game; FL10, W20 and the sweep's shapes below get it by assumption, which holds best where their code is H15's
kind (the released shapes, all inside the 2 MiB L2) and least for weights beyond the L2. The jobs' rate (5,068 µs)
would give about 1.026: a textbox job of one line spends nothing outside `vocr_run`, the five-line full screen 5 %.

| phase | shape | MACs/frame | before cycles/frame (low-high) | after | before ms/frame @333 | after | speed-up |
|---|---|---:|---:|---:|---:|---:|---:|
| 19 | head 96 → 5,355 | 514,080 | 293,068-538,696 | 235,489-247,402 | 1.619 | 0.743 | 2.18x |
| 4 | PW 96 → 96, h8, +res | 147,456 | 82,139-139,693 | 68,539-94,434 | 0.420 | 0.284 | 1.48x |
| 13 | PW 2x1 collapse 240 → 288 | 138,240 | 65,084-114,360 | 53,156-59,760 | 0.344 | 0.180 | 1.91x |
| 8 | PW 192 → 192, h4, +res | 147,456 | 71,706-112,596 | 60,359-74,537 | 0.338 | 0.224 | 1.51x |
| 2 | PW 48 → 96, h8 | 73,728 | 47,002-91,311 | 37,435-52,468 | 0.274 | 0.158 | 1.74x |
| 12 | PW 240 → 240, h2, +res | 115,200 | 54,273-84,843 | 45,715-51,019 | 0.255 | 0.153 | 1.66x |
| 15 | PW 288 → 288, +res | 82,944 | 43,100-77,988 | 33,252-39,964 | 0.234 | 0.120 | 1.95x |
| 10 | PW 192 → 240, h2 | 92,160 | 46,210-75,529 | 37,164-43,556 | 0.227 | 0.131 | 1.73x |
| 0 | stem 3x3/2, 1 → 48 | 27,648 | 43,733-73,302 | 33,464-45,137 | 0.220 | 0.136 | 1.62x |
| 17 | PW 288 → 288, +res | 82,944 | 39,785-70,447 | 32,575-35,931 | 0.212 | 0.108 | 1.96x |
| 6 | PW 96 → 192, h4 | 73,728 | 39,159-67,998 | 31,845-40,368 | 0.204 | 0.121 | 1.68x |
| 3 | DW 3x3, 96, h8 | 13,824 | 21,316-46,360 | 18,146-32,466 | 0.139 | 0.098 | 1.43x |
| 18 | PW 288 → 96 | 27,648 | 13,986-25,027 | 11,542-13,547 | 0.075 | 0.041 | 1.85x |
| 7, 1, 5, 9, 11, 14, 16 | the other depthwise layers | 27,936 | 41,364-107,455 | 35,571-81,690 | 0.323 | 0.245 | 1.32x |
| 101 | input columns (resize, normalise) | 0 | 8,422-18,595 | 8,346-18,461 | 0.056 | 0.055 | 1.01x |
| 100 | the rest of `vocr_run` | 0 | 580-2,810 | 580-2,810 | 0.008 | 0.008 | 1.00x |
| **all** | | 1,564,992 | **910,928-1,647,008** | **743,181-933,549** | **4.949** | **2.805** | **1.76x** |

"Before" is `main` a540027 (#82's top-K branch in `run_head` makes its greedy head 0.15 % dearer than the
console's code). Instructions per frame went from 1,156,544 to 772,544, NEON-VFP unit switches from 17,509 to 463
per frame (the VFP compare per class in the head), and the simulated L1 data misses halved (10,339 to 5,151 per
frame) because the weight stream is prefetched: 5,162 lines per frame are filled by `PLD`. With the top 8 (the
head's insertion per block) the head costs 0.835 instead of 0.743 ms; `main` pays 1.740 for it.

**Without the prefetch model.** The model's `PLD` is an instant fill into L1 and L2, the best case. A run
without it (`--no-pld`: the same binary, every L1 miss at the L2 latency, every L2 miss at the DRAM latency, the
bound if the A9's `PLD` did nothing) gives H15 3.27 ms greedy, 3.36 ms with the top 8: 1.51x. The low bound
(perfect overlap, NEON issue) moves 1.23x. So the range to expect on the console is **1.5-1.8x**, most likely near
the high bound, which is where the console sat before. For weights beyond the 2 MiB L2 the prefetch decides most
of the gain (the sweep below): `l_w325v5631` has 22,672 L2 misses per frame without it, 3,735 with it.

| model (int8 weights) | `main` ms/frame @333, top-8 | this PR @333 | without PLD @333 | this PR @444 | without PLD @444 | a 3-line box's recognition (240 frames) @333, `main` → this PR (without PLD) |
|---|---:|---:|---:|---:|---:|---:|
| VOCR3-FL10 (657,664 B) | 3.01 | 1.71 | 1.94 | 1.28 | 1.46 | 0.72 → 0.41 s (0.47) |
| VOCR3-H15 (1,155,968 B) | 5.07 | 2.90 | 3.36 | 2.17 | 2.52 | 1.22 → 0.70 s (0.81) |
| VOCR4-W20 (1,558,016 B) | 6.94 | 4.08 | 4.83 | 3.06 | 3.62 | 1.66 → 0.98 s (1.16) |

The 444 MHz column scales the cycles (the DRAM term in nanoseconds stays): a 444 MHz run is the overlay's opt-in
clock flag (vita-jp-overlay), off by default.

Larger tiles (`--tile 8 --head 32`, arena 318,528 B instead of 182,944 B for H15) give another 7 % (3.07 instead of
3.29 ms per frame at that point of the work); not taken: the arena grows by 74 % and the overlay pins the default
tiling.

## Bit-identity

`vita/speed_parity.py` builds `runtime/tools/vocr_cli.c` (with `vocr_lm.c`, which it links) with `main`'s `vocr.c`
and with this one, each five ways (host gcc 11.5 and clang 18.1.8 at `-O2`; Cortex-A9 NEON and scalar with the Arm
GNU 14.2 toolchain under qemu-arm; the Vita compiler's objects, arm-vita-eabi-gcc 15.2, linked into the ARM CLI),
runs every line of the pack with every blob and mode, hashes all logits as they stream (`--logits` through a pipe),
the texts, and with `--topk 8` every frame's top 8, blank logit and line confidence (`vocr_cli --topk-out`), and
requires every hash to equal `main`'s and every build's to equal every other build's. On the 884-line pack
(`6cff78dc…`, 31,479 frames):

| blob | mode | logit bytes per build | logits sha256 | texts sha256 | top-8 sha256 | 5 builds, new == `main` |
|---|---|---:|---|---|---|:-:|
| VOCR3-FL10 | w8i | 440,838,988 | `47142b9ba3d8c11e` | `57c3a62759da142d` | `e768e8cc84934f11` | yes |
| VOCR3-FL10 | w8f | 440,838,988 | `0991add51ba165b1` | `57c3a62759da142d` | `fa47833853a3fb8d` | yes |
| VOCR3-H15 | w8i | 674,287,252 | `ca014b98cc312609` | `6a616d6564d4375e` | `5c8ac73ceb471329` | yes |
| VOCR3-H15 | w8f | 674,287,252 | `8fbceb2b7020a88e` | `6a616d6564d4375e` | `d7ca458e1f419827` | yes |
| VOCR4-W20 | w8i | 674,287,252 | `6b1f13c548b9de14` | `fe96443343405d11` | `2c4e433246bb9cba` | yes |
| VOCR4-W20 | w8f | 674,287,252 | `70d3457cb6332a13` | `fe96443343405d11` | `08254c081c3e0bbb` | yes |

The logits and texts of the greedy and the top-8 runs are the same (the top-K changes neither); the table gives
each run's hashes once.

`runtime/tests/test_vocr.c` checks every NEON kernel against an independent answer:
- the logits and text of eight blob shapes in every mode, format and tiling hash to fixed known answers (`main`'s
  scalar code computed them, and host gcc and clang and Cortex-A9 NEON and scalar give the same): 8 or 32 stem and
  3x3 depthwise channels (32 runs their 16-channel NEON paths twice per position), a 16-channel 1x5 depthwise (the
  kw ≠ 3 path), pointwise and collapse matrices with K below and above the 512-term chunk;
- the head's argmax on a 21-class head (two full lane blocks and a partial one) on ties within and across blocks,
  -0 against +0 in both orders, a NaN logit, all -inf and NaN with -inf, in W8I, W8F and f32 at two head batch
  sizes; every logit is its class's bias; the top 1, 3, 8 and 16 equal the per-class insertion's.

`make check-arm` runs it as Cortex-A9 NEON and scalar builds under qemu-arm (CI is x86 only and never compiles the
NEON paths). Each of 25 mutations of the new kernels (audit-ocr-speed-1's: the tie rule, the NaN key, the key's
sign, the lane hand-offs and init, the accumulators, weight rows, positions, the scale lane, chunk sums, the bias,
`ring_shift`, `k_dw`'s taps and weight step, `k_conv1`'s input step and lane weights, the logits copy) and 8 of 9
mutations of the top-K block filter fail it; the ninth ("never skip a block") is equivalent. `make parity-arm` (the
PyTorch reference: f32, w8f and w8i, 40 lines, NEON and scalar) passes. The PyTorch parity test (`make check`) is
unchanged and passes. No sanitizer sees inside NEON intrinsic loads (gcc does not instrument them); their bounds
rest on the reasoning above and on the parity runs.

## The development pack

`vita/dev_pack.py` writes what the overlay's game worker hands the recognizer on the console: for every region of
`eval/linefind/manifests/vita_dev.jsonl` and `psp_dev.jsonl` (363 regions), the region's grey rows with the
display hook's `(77 R + 150 G + 29 B) >> 8`, the GRAD finder (`runtime/vocr_lines.c`, which gives the same lines as
the worker's row finder), and every found line's crop, inverted when the finder calls it dark (exactly
`VOCR_INVERT`). 884 lines, 31,479 CTC frames (36 per line on average; 60-100-frame lines are the console's textbox
lines). It refuses every row of a VN that the committed test reservation (`eval/manifests/reserved.json` "test": the
CLANNAD family, vita-test, psp-test and their shared releases) or `eval/manifests/test_games.json` names: on the
2026-10-14 dev manifests the 9 rows of v1267, v4060 and v36. The 24 timing lines are the same with and without
them (the subset's sha256 `3162151a…` either way; none of the 24 is from those VNs). The pack holds crops of
copyrighted screenshots: it stays in `$HERD/shared/artifacts/ocr-speed/`.

## What these speeds allow (model-size ceilings)

Random-init blobs of `vita/model_tiers.py`'s shapes (and three wider ones of the same family) on the same 24 lines
with the top 8, `main`'s runtime and this one, calibrated the same way; this PR with and without the prefetch
model. Box = 3 x 80 frames of recognition plus 0.1 s for the finder and the strips, one core.

| shape | int8 B | MMAC/frame | `main` ms/frame @333 | this PR @333 (without PLD) | this PR @444 (without PLD) | speed-up without PLD | box @333, this PR (without PLD) |
|---|---:|---:|---:|---:|---:|---:|---:|
| `s_w100v3500` | 657,616 | 0.82 | 3.02 | 1.72 (1.96) | 1.29 (1.47) | 1.54x | 0.51 s (0.57) |
| `s_w120v4987b64` | 761,168 | 1.01 | 3.88 | 2.21 (2.51) | 1.66 (1.88) | 1.54x | 0.63 s (0.70) |
| `s_w100v4987` | 819,792 | 0.97 | 3.50 | 1.95 (2.21) | 1.46 (1.66) | 1.58x | 0.57 s (0.63) |
| `n_w200v5631` | 1,587,680 | 2.37 | 7.03 | 4.14 (4.89) | 3.11 (3.67) | 1.44x | 1.09 s (1.27) |
| `n_w250v5631` | 2,095,712 | 3.35 | 10.08 | 5.67 (7.47) | 4.27 (5.71) | 1.35x | 1.46 s (1.89) |
| `n_lw300v5631` | 2,473,696 | 3.40 | 10.75 | 5.69 (8.24) | 4.29 (6.36) | 1.30x | 1.47 s (2.08) |
| `n_w300v5631` | 2,709,728 | 4.55 | 15.66 | 7.91 (12.56) | 6.01 (9.92) | 1.25x | 2.00 s (3.11) |
| `l_w300v6459b128` | 3,025,808 | 4.85 | 16.48 | 8.31 (13.18) | 6.32 (10.42) | 1.25x | 2.09 s (3.26) |
| `l_w325v5631` | 3,056,480 | 5.23 | 18.25 | 9.32 (14.89) | 7.10 (11.79) | 1.23x | 2.34 s (3.67) |
| `l_w350v5631` | 3,429,728 | 5.96 | 20.63 | 10.50 (16.95) | 8.00 (13.43) | 1.22x | 2.62 s (4.17) |
| `l_w300v6459b192` | 3,476,624 | 5.30 | 17.49 | 8.83 (14.05) | 6.71 (11.11) | 1.24x | 2.22 s (3.47) |
| `l_w375v5631` | 3,829,472 | 6.74 | 23.34 | 12.04 (19.37) | 9.16 (15.35) | 1.21x | 2.99 s (4.75) |
| `l_w400v5631` | 4,255,712 | 7.58 | 25.92 | 13.32 (21.61) | 10.13 (17.12) | 1.20x | 3.30 s (5.29) |
| `l_w450v5631` | 5,187,680 | 9.42 | 32.08 | 16.73 (27.13) | 12.71 (21.48) | 1.18x | 4.12 s (6.61) |
| `l_w500v5631` | 6,225,632 | 11.46 | 38.73 | 20.22 (33.02) | 15.35 (26.16) | 1.17x | 4.95 s (8.02) |

Inside the L2 this runtime takes 1.7-2.1 ms per frame per million MACs at 333 MHz (`main`: 3.0-3.8; the shapes with
few MACs per class at the high end), and 2.1-2.4 without the prefetch model. Beyond the 2 MiB L2 (`n_w250v5631` and up) the shapes are bound by the weight
stream's DRAM misses: with the prefetch model this PR halves their time, without it the gain falls to 1.17-1.35x.
Which of the two the console gives is unmeasured; the overlay's next test build logs ms per frame per line, which
settles it for H15 (inside the L2) but not for a shape beyond it.

The largest int8 weights of the width family (5,631 classes) that fit each time limit, interpolated on the
frontier of the measured shapes (box limit = 3 lines on one core, or 2 lines when a second core recognizes the
third; estimates):

| clock, cores | `main` 1 s / 2 s / 3 s box | this PR without PLD | this PR with PLD |
|---|---|---|---|
| 333 MHz, 1 core | 0.87 / 1.74 / 2.32 MB | **1.26 / 2.15 / 2.65 MB** | 1.45 / 2.71 / 3.84 MB |
| 444 MHz, 1 core | 1.14 / 2.13 / 2.69 MB | 1.61 / 2.42 / 3.12 MB | 1.87 / 3.40 / 4.96 MB |
| 333 MHz, 2 cores | 1.28 / 2.29 / 3.04 MB | 1.73 / 2.63 / 3.62 MB | 2.08 / 3.79 / 5.60 MB |
| 444 MHz, 2 cores | 1.66 / 2.66 / 3.79 MB | 2.07 / 3.08 / 4.47 MB | 2.57 / 4.89 / > 6.2 MB |

The safe reading (one core, 333 MHz, no prefetch credit): a 1 s box allows 1.26 MB, a 2 s box (Normal) 2.15 MB, a
3 s box (Large) 2.65 MB. Small stays under 1 MB (the owner's cap; time would allow 1.26 MB). The overlay's
`docs/ocr-speed.md` (vita-jp-overlay #93) has what its parallel lines and clock flag add. All estimates.

## Reproduce

```sh
export VITASDK=... ARM_CC=.../arm-none-linux-gnueabihf-gcc VOCR_A9_PLUGINS=<libinsn_mix.so of this checkout>
python vita/dev_pack.py --shots DIR [--shots DIR ...] --out dev.bin --index dev.json
python vita/a9_layers.py measure --blob VOCR3_H15_w8.vocr --pack dev.bin --out a9/new \
    --min-frames 60 --max-frames 100 --limit 24 --plain [--cli-opts '--topk 8'] [--no-pld]
python vita/a9_layers.py measure ... --out a9/main --runtime <main's runtime/>
python vita/a9_layers.py compare a9/main a9/new --scale 1.000676 --console-mhz 333
python vita/speed_parity.py --base <main's runtime/> --new runtime --pack dev.bin \
    --blob VOCR3_FL10_w8.vocr --blob VOCR3_H15_w8.vocr --blob VOCR4_W20_w8.vocr --modes w8i,w8f [--cli-opts '--topk 8']
make -C runtime check-arm parity-arm ARM_CC=... QEMU=.../qemu-arm
```

`vita/scripts/build_qemu_plugins.sh` builds the plugin with the new options.

## Not measured

- The console has not run this code; every time here is the cost model calibrated on the console's code and one
  console figure. The overlay's next test build logs ms per frame per line, which settles it for H15.
- Whether the A9's `PLD` behaves as the model's prefetch (an instant fill): the figures without it are the bound.
- CPU3: unmeasured (AGENTS.md).
