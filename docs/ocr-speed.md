# OCR speed: faster W8I kernels, bit for bit (ocr-speed)

The owner's v0.8.0-test.1 console logs (vita-jp-overlay #81) are the ground truth: VOCR3-H15 ran
**4,941-5,238 µs per CTC frame** (mean about 5.07 ms) on 70-96-frame CLANNAD lines, 0.2-0.4 s per
textbox line. This change makes the same arithmetic cheaper on the Cortex-A9. Every logit is
bit-identical to `main`'s (W8I and W8F, five builds, three released models, 900 development lines).

**Estimated effect (calibrated to the console, see below): VOCR3-H15 5.07 → 2.88 ms per frame at 333
MHz (1.76x), VOCR3-FL10 3.00 → 1.68 (1.78x), VOCR4-W20 6.98 → 4.09 (1.71x).** A 3-line box of 640-px
lines (240 frames) of H15 goes from 1.22 s to 0.69 s of recognition at 333 MHz, 0.52 s at 444 MHz.
These are estimates; the console has not run this code.

## What changed (runtime/vocr.c)

| kernel | before | after | why the bits are the same |
|---|---|---|---|
| `mm_i16` 4 positions x 8 outputs (every pointwise layer and the head) | 4 k per pass, two 64-bit weight loads and a VMOVL per k, 8 int32 accumulators | 8 k per pass: one 128-bit load per position, the 8 weight rows in two 256-bit loads (`vld1q_s8_x2`), 64 VMLAL per pass; a `PLD` 128-160 B ahead on the weight stream; the single-chunk case (K ≤ 512, every VOCR layer but W20's collapse) skips the float chunk adder | integer sums are exact in any order; the float steps (convert, `+` chunk sums, `* (scale[j] * sx[p])`, `+ bias`, relu, `+ res`) are the same operations in the same order; for one chunk `0 + float(c)` is `float(c)` (an int's float is never -0) |
| `mm_i16` 1-3 leftover positions (head batches of 17-19 frames, line ends) | plain C, 8 int32 sums through the stack | NEON, 4 accumulators per position, added exactly at the chunk's end | same |
| `run_head` argmax (W8I with NEON) | one VFP compare per class (5,355 per frame for H15), reloading the logits | `mm_i16` on 32 frames per call; the running maximum per lane in NEON with an integer key of the float (`fkey`: magnitude bits, negated for negatives, NaN below everything); the first-maximum tie rule finished per frame (`lanes_argmax`) | the key orders floats exactly as `v > best` (-0 equals +0, a NaN is never kept); ties go to the lower class, as the scalar scan; a partial last block (5,355 = 669 x 8 + 3) still runs the scalar scan after the lanes, in class order |
| `k_dw` / `k_conv1` (stem and depthwise) | index arithmetic per tap, 128-bit loads | pointer stepping per tap row, 256-bit loads (`vld1q_f32_x2`), 3-wide taps unrolled, the stem's input value loaded into a lane (`vld1_dup_f32`) | the same multiply-adds in the same order per output (`vmlaq` lane by lane, `-ffp-contract=off`) |
| `ring_shift` | float by float | 4 floats at a time, still forwards | a copy |

`VOCR_NO_NEON` (the PSP and scalar path) is untouched. Code size with the Vita compiler: `vocr.o`
14,583 → 16,095 B (`arm-vita-eabi-size`, text). The stack frame of `run_head` grows by 1 KiB (the
32-frame logit block). No arena change: `vocr_arena_bytes()` is the same for every model and option.

## Before / after per layer (VOCR3-H15)

`vita/a9_layers.py` (new): the Vita compiler's code for `runtime/vocr.c` with `-DVOCR_PROF`, which
makes `vocr_run()` call a marker as each phase starts, under qemu-arm with `vita/qemu/insn_mix.c`
counting instructions per phase (`marker=`, `layers=`), its data cache model (`l1d=on`: 32 KiB 4-way
L1, 2 MiB 8-way L2, LRU) and the `PLD` sites (`pld=`, which qemu itself runs as no-ops); then
`vita/a9_cycles.py`'s Cortex-A9 cost model per phase. Without the marker the same lines give the same
texts (`--plain` checks it). The lines: 24 development lines of 60-100 CTC frames (1,609 frames)
from `vita/dev_pack.py` (below), the length of the console's lines.

**Calibration.** The console's 5.07 ms per frame at 333 MHz is 1,688,310 cycles. `main`'s code gives
910,919-1,644,475 cycles per frame on these lines (low and high bound), so the console runs at
1.027 x the high bound (no dual issue, every miss at full latency). Every "ms" below is the high
bound x 1.027 at that clock: an estimate, calibrated on one model and one game.

| phase | shape | MACs/frame | before cycles/frame (low-high) | after | before ms/frame @333 | after | speed-up |
|---|---|---:|---:|---:|---:|---:|---:|
| 19 | head 96 → 5,355 | 514,080 | 293,068-536,061 | 235,489-246,522 | 1.653 | 0.760 | 2.17x |
| 4 | PW 96 → 96, h8, +res | 147,456 | 82,139-140,273 | 68,539-94,547 | 0.432 | 0.291 | 1.48x |
| 13 | PW 2x1 collapse 240 → 288 | 138,240 | 65,084-114,365 | 53,156-59,767 | 0.353 | 0.184 | 1.91x |
| 8 | PW 192 → 192, h4, +res | 147,456 | 71,706-112,579 | 60,359-74,571 | 0.347 | 0.230 | 1.51x |
| 2 | PW 48 → 96, h8 | 73,728 | 47,002-91,257 | 37,435-52,466 | 0.281 | 0.162 | 1.74x |
| 12 | PW 240 → 240, h2, +res | 115,200 | 54,273-84,924 | 45,715-51,033 | 0.262 | 0.157 | 1.66x |
| 15 | PW 288 → 288, +res | 82,944 | 43,100-77,881 | 33,252-39,933 | 0.240 | 0.123 | 1.95x |
| 10 | PW 192 → 240, h2 | 92,160 | 46,210-75,535 | 37,164-43,534 | 0.233 | 0.134 | 1.74x |
| 0 | stem 3x3/2, 1 → 48 | 27,648 | 43,733-73,266 | 33,464-45,080 | 0.226 | 0.139 | 1.63x |
| 17 | PW 288 → 288, +res | 82,944 | 39,785-70,419 | 32,575-35,945 | 0.217 | 0.111 | 1.96x |
| 6 | PW 96 → 192, h4 | 73,728 | 39,159-68,083 | 31,845-40,430 | 0.210 | 0.125 | 1.68x |
| 3 | DW 3x3, 96, h8 | 13,824 | 21,316-46,327 | 18,146-32,451 | 0.143 | 0.100 | 1.43x |
| 18 | PW 288 → 96 | 27,648 | 13,986-25,019 | 11,542-13,544 | 0.077 | 0.042 | 1.85x |
| 7, 1, 5, 9, 11, 14, 16 | the other depthwise layers | 27,936 | 41,364-107,116 | 35,566-81,509 | 0.330 | 0.251 | 1.31x |
| 101 | input columns (resize, normalise) | 0 | 8,412-18,553 | 8,336-18,414 | 0.057 | 0.057 | 1.01x |
| 100 | the rest of `vocr_run` | 0 | 582-2,816 | 582-2,815 | 0.009 | 0.009 | 1.00x |
| **all** | | 1,564,992 | **910,919-1,644,475** | **743,166-932,560** | **5.070** | **2.875** | **1.76x** |

Instructions per frame went from 1,153,879 to 771,630, NEON-VFP unit switches nearly vanished from
the head (the VFP compare per class), and the simulated L1 data misses halved (10,344 to 5,149 per
frame) because the weight stream is prefetched. If the 5,162 prefetched lines per frame were
charged as misses anyway (a prefetch the A9 issues too late), the estimate drops to 1.51x for H15
(1.56x FL10, 1.44x W20); the low bound (perfect overlap, NEON issue) moves 1.23x. So the range to
expect on the console is **1.5-1.8x**, most likely near the high bound, which is where the
console sat before.

| model | before ms/frame @333 | after @333 | after @444 | speed-up | 3-line box (240 frames) @333 before → after | @444 after |
|---|---:|---:|---:|---:|---:|---:|
| VOCR3-FL10 (657,664 B) | 3.00 | 1.68 | 1.26 | 1.78x | 0.72 → 0.40 s | 0.30 s |
| VOCR3-H15 (1,155,968 B) | 5.07 (console) | 2.88 | 2.16 | 1.76x | 1.22 → 0.69 s | 0.52 s |
| VOCR4-W20 (1,558,016 B) | 6.98 | 4.09 | 3.07 | 1.71x | 1.68 → 0.98 s | 0.74 s |

The 444 MHz column scales the cycles (the DRAM term in nanoseconds stays): a 444 MHz run is the
overlay's opt-in clock flag (vita-jp-overlay), off by default.

Larger tiles (`--tile 8 --head 32`, arena 318,528 B instead of 182,944 B for H15) give another 7%
(3.07 instead of 3.29 ms per frame at that point of the work); not taken: the arena grows by 74% and
the overlay pins the default tiling.

## Bit-identity

`vita/speed_parity.py` builds `runtime/tools/vocr_cli.c` with `main`'s `vocr.c` and with this one,
each five ways (host gcc 11.5 and clang 18.1.8 at `-O2`; Cortex-A9 NEON and scalar with the Arm GNU
14.2 toolchain under qemu-arm; the Vita compiler's object, arm-vita-eabi-gcc 15.2, linked into the
ARM CLI), runs every line of the pack with every blob and mode, hashes all logits as they stream
(`--logits` through a pipe) and the texts, and requires every hash to equal `main`'s and every
build's to equal every other build's.

| blob | mode | lines | frames | logit bytes per build | builds | equal to main | equal across builds |
|---|---|---:|---:|---:|---:|:-:|:-:|
| VOCR3-FL10 | w8i, w8f | 900 | 32,120 | 449,815,680 | 5 | yes | yes |
| VOCR3-H15 | w8i, w8f | 900 | 32,120 | 688,017,600 | 5 | yes | yes |
| VOCR4-W20 | w8i, w8f | 900 | 32,120 | 688,017,600 | 5 | yes | yes |

The hashes are in the PR body. `runtime/tests/test_vocr.c` gains a head-argmax test (a 21-class head:
two full lane blocks and a partial one) on ties within and across blocks, -0 against +0 in both
orders, a NaN logit, all -inf and NaN with -inf, in W8I, W8F and f32 at two head batch sizes; `make
check-arm` (new) runs the unit tests as Cortex-A9 NEON and scalar builds under qemu-arm. Six
mutations of the new argmax and of the leftover-position kernel each fail it. The PyTorch parity test
(`make check`) is unchanged and passes.

## The development pack

`vita/dev_pack.py` writes what the overlay's game worker hands the recognizer on the console: for
every region of `eval/linefind/manifests/vita_dev.jsonl` and `psp_dev.jsonl` (372 regions, never the
holdout or a test set), the region's grey rows with the display hook's `(77 R + 150 G + 29 B) >> 8`,
the GRAD finder (`runtime/vocr_lines.c`, which gives the same lines as the worker's row finder), and
every found line's crop, inverted when the finder calls it dark (exactly `VOCR_INVERT`). 900 lines,
32,120 CTC frames (36 per line on average; 60-100-frame lines are the console's textbox lines). It
refuses any row of a VN in `eval/manifests/test_games.json` or CLANNAD; the dev manifests hold none.
The pack holds crops of copyrighted screenshots: it stays in `$HERD/shared/artifacts/ocr-speed/`.

## What these speeds allow (model-size ceilings)

Random-init blobs of `vita/model_tiers.py`'s shapes on the same 24 lines with this runtime, calibrated
the same way; the time per 3-line box of 640-px lines (240 frames), recognition only, on one core.
Time follows MACs closely: 1.68-2.04 ms per frame per million MACs at 333 MHz (about 1.8; `main`:
2.98-3.55).

| shape | int8 B | MMAC/frame | ms/frame @333 | box @333 | ms/frame @444 | box @444 |
|---|---:|---:|---:|---:|---:|---:|
| `s_w100v4987` | 819,792 | 0.97 | 1.89 | 0.45 s | 1.42 | 0.34 s |
| `n_w200v5631` | 1,587,680 | 2.37 | 4.13 | 0.99 s | 3.10 | 0.74 s |
| `n_w250v5631` | 2,095,712 | 3.35 | 5.69 | 1.37 s | 4.28 | 1.03 s |
| `n_w300v5631` | 2,709,728 | 4.55 | 8.00 | 1.92 s | 6.08 | 1.46 s |
| `l_w300v6459b128` | 3,025,808 | 4.85 | 8.42 | 2.02 s | 6.40 | 1.54 s |
| `l_w325v5631` | 3,056,480 | 5.23 | 9.46 | 2.27 s | 7.20 | 1.73 s |
| `l_w300v6459b192` | 3,476,624 | 5.30 | 8.93 | 2.14 s | 6.79 | 1.63 s |
| `l_w350v5631` | 3,429,728 | 5.96 | 10.65 | 2.56 s | 8.11 | 1.95 s |
| `l_w400v5631` | 4,255,712 | 7.58 | 13.60 | 3.26 s | 10.34 | 2.48 s |

The ceilings per tier, with the overlay's further gains (parallel lines, the clock flag) and the
finder's time, are in the report to the training conductor (vita-jp-overlay `docs/ocr-speed.md`).
`main`'s `l_w325v5631` took 18.59 ms per frame: its 3 MB of weights miss the L2, which the prefetch
now hides (1.97x).

## Reproduce

```sh
export VITASDK=... ARM_CC=.../arm-none-linux-gnueabihf-gcc VOCR_A9_PLUGINS=<libinsn_mix.so of this checkout>
python vita/dev_pack.py --shots DIR [--shots DIR ...] --out dev.bin --index dev.json
python vita/a9_layers.py measure --blob VOCR3_H15_w8.vocr --pack dev.bin --out a9/new \
    --min-frames 60 --max-frames 100 --limit 24 --plain
python vita/a9_layers.py measure ... --out a9/main --runtime <main's runtime/>
python vita/a9_layers.py compare a9/main a9/new --console-ms 5.07 --console-mhz 333
python vita/speed_parity.py --base <main's runtime/> --new runtime --pack dev.bin \
    --blob VOCR3_FL10_w8.vocr --blob VOCR3_H15_w8.vocr --blob VOCR4_W20_w8.vocr --modes w8i,w8f
```

`vita/scripts/build_qemu_plugins.sh` builds the plugin with the new options.

## Not measured

- The console has not run this code; every time here is the cost model calibrated on `main`'s code
  and one console figure. The overlay's next test build logs ms per frame per line, which settles it.
- Whether the A9's `PLD` behaves as the model's prefetch (an instant fill): the 1.5x figure is the
  bound without it.
- CPU3: unmeasured (AGENTS.md).
