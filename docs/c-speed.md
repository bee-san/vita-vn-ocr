# c-speed: a faster plain-C runtime on the Vita, bit for bit (#165)

The owner asked whether VOCR could be improved with what outside researchers found (2026-10-31). astra
(GameSentenceMiner) converted VOCR6.2 and textspot8 to ONNX, kept the int8 arithmetic exact and found the conversions
2-10x faster than our C runtime on a desktop CPU. vocr-research reproduced that and explained it (#164,
`reports/vocr-research.md`): on a PC most of the gap is the C build (gcc `-O2` leaves the scalar loops unvectorised),
and little of ONNX Runtime's advantage transfers to the Vita, whose build already has hand-written NEON kernels. This
change takes what does transfer to the Cortex-A9 and applies it to the two files Rubi runs on the console:
`runtime/vocr.c` (the recognizer) and `runtime/vocr_textspot6.c` (textspot8, Rubi's line finder). Every output stays
the same bit for bit, the arenas do not change, and every build without NEON compiles to `main`'s exact machine code.

Thank you, astra: your report pointed at the kernels' arithmetic and at the skipped head work, and your exactness
checks set the bar this change is held to.

**Estimated effect** (the Vita compiler's code under qemu, the Cortex-A9 cost model, x1.000676 to VOCR3-H15's measured
console rate; one core; Rubi's options: top-8, Normal with its head gate and two-level head, Large with its two-level
head; nothing here has run on a console):

| | before → after, ms per CTC frame @333 MHz | 3-line box (240 frames + 0.1 s) @333 | @444 | with prefetch credit, box @333 |
|---|---:|---:|---:|---:|
| Small `VOCR6.2_S_w8` | 2.195 → **2.000** (−8.9%) | 0.627 → **0.580 s** | 0.495 → 0.460 s | 0.564 → 0.517 s |
| Normal `VOCR6.2_N_w8g2` (default) | 2.807 → **2.683** (−4.4%) | 0.774 → **0.744 s** | 0.605 → 0.583 s | 0.672 → 0.642 s |
| Large `VOCR6.2_L_w8_2l` | 7.415 → **7.146** (−3.6%) | 1.880 → **1.815 s** | 1.470 → 1.421 s | 1.335 → 1.271 s |

| textspot8 (`t64n8c16`, 4 tags), est. ms @333 / @444 | with the dark pass (the published figures) | without it (Rubi's `VOCR_TEXTSPOT_NO_DARK`) |
|---|---:|---:|
| a 900×150 region (the crop decode) | 81.9 → **61.0** / 45.7 (−25.5%) | 76.1 → **55.2** / 41.4 (−27.4%) |
| 900×200 | 104.7 → 77.5 | 98.4 → 71.1 |
| 900×300 | 150.6 → 110.6 | 143.3 → 103.2 |
| a whole 960×544 frame (the rule's bound) | 264.3 → **192.2** / 144.1 (−27.3%) | 258.4 → 186.3 / 139.7 |

The box column is the recognizer's nominal box as the earlier docs give it (`docs/vocr6.md`): its 0.1 s stands for
the line finder. With textspot8 as Rubi's finder, a Normal box with a 900×150 region is about 0.774 + 0.076 → 0.744 +
0.055 s (estimates, Rubi's flags). Rubi builds the recognizer with `-DVOCR_NO_ATTN` (below): 0.579 / 0.743 / 1.813 s,
the same code in every kernel.

## What changed

### `runtime/vocr.c` (NEON builds only)

| kernel | before | after | why the bits are the same |
|---|---|---|---|
| `mm_i16`'s 4 positions × 8 outputs pass (every pointwise layer, the head, the gate and the escape) | intrinsics: gcc kept the 8 sums, 4 inputs, 32 weight bytes and a widened row in the 16 Q registers only by rotating a sum through the weight registers (4 `VORR` a pass) and copied the weight pointer for the `PLD`s: 88 instructions per 256 multiply-adds | the same pass as an `asm` loop (`VOCR_ASM`; `-DVOCR_NO_ASM` keeps the intrinsics) with the registers fixed and `:128` alignment hints on the 6 loads: 82 instructions, and the loads issue in 6 cycles instead of 14. It runs when the rows are 16-byte aligned and K is a multiple of 8 (every VOCR matrix: the arena's scratch and the blob's sections are aligned); any other call takes the one-position loop, so there is one copy of the pass | the same `VMLAL.S16` in the same order; integer sums are exact in any order anyway. The `PLD`s go to the same addresses |
| the 3×3 stem (`k_conv1`, C a multiple of 16) | 16 channels per output position: per tap, two 32-byte weight loads, the input as a lane, 4 `VMLA` | `k_conv1_3x3`: 4 channels at a time with their 9 tap weights in 9 Q registers over every output of the call, and each column's 3 rows in one load | each output is still bias, then + w · x over the taps in order (dy, then dx) by `VMLA` by lane, the vertical taps outside the input skipped (the border rows take the taps one by one, as before); only the order of independent outputs changes. With C % 16 the old loop ran every channel on NEON too, so the flush of denormals is the same |
| the top-K filter in `run_head` (Small; any head with the top-K and without the two-level head) | per block of 8 classes and frame: `fkey` (6 operations) of the 8 logits and of the frame's K-th largest, then one NEON→ARM move, which the A9 pays as a NEON/VFP switch | a cheaper order key (`okey`: 3 operations, the magnitude bits flipped for negatives), the frames' thresholds kept as keys (updated after an insertion), 4 frames per test and the pass bits of the whole batch gathered in one register, so one move per block | the filter only decides which blocks go through the per-class insertion, which compares exactly as before; `okey` orders every two non-NaN floats as `>` does (only -0 < +0 more), the threshold is never NaN, so every block with a logit above the threshold still passes (a superset); the insertion itself is unchanged |
| `block_above` (the two-level head's block test) | `fkey` of both | `okey`, the same superset argument | as above |

Kept as they were after trying (measured on the same lines): a 4-channel form of the depthwise layers (slower: their
2-8 rows are mostly border rows, and the 16-channel loop already loads each input once per output), `-O3` (+23-42% per
frame: gcc 15.2 spills NEON registers; the research measured the same) and `-fno-tree-vectorize` (−0.1%).

**`-DVOCR_NO_ATTN`** (a new build option; the default build is unchanged, to the machine code): it leaves out the
attention op (#130, VOCR6's stage-3 candidate, not adopted; no released blob has one), and `vocr_load` then refuses a
blob with one (`VOCR_E_FORMAT`). Rubi loads only the blobs it pins, none of which has the op, so its build can drop
the code: `vocr.o` 26,723 → 24,083 B, which more than pays for this change's code in Rubi's worker (Memory, below).
`make check` and `make check-arm` run the unit tests in that build too (the attention tests become refusal checks).

### `runtime/vocr_textspot6.c` (NEON builds only)

| kernel | before | after | why the bits are the same |
|---|---|---|---|
| 1×1 layers (`k1_block`, 56% of a region's instructions before) | 2 output channels per pass: each input channel's 8 pixels loaded and widened twice per 4 output channels | 4 output channels per pass (8 sums, the 4 weight rows of 8 input channels, the inputs of 4 channels in one 32-byte load); the lone channels past a multiple of 4 one at a time | the int32 sums are exact (at most 64 products of two int8 codes) in any order |
| the requantisation (`RQ_STORE`, every output) | `VMULL.S32` into int64, `VRSHL.S64`, two narrowing steps | for a channel whose bias + sum fits int32 and whose shift is at least 32 (every channel of a trained blob): `VQDMULH.S32` then `VRSHL.S32` by 31 − shift, the multi-scale path's mode 0 (`rq4v`), whose proof the module comment gives; a 1×1 layer whose every channel has that form takes its 4 channels' constants with one `VLD4` and by lane; the int64 forms moved out of line (`rq_int64`), unchanged | floor((floor(v m / 2^31) + 2^(k − 1)) / 2^k) = floor((v m + 2^(shift − 1)) / 2^shift) for k = shift − 31 ≥ 1, the twin's value; then the same saturation and ReLU |
| depthwise 3×3 (`K3_DW`) | `K3_BLOCK`'s loop, inlined into `feed`, which spilled its row pointers per channel | its own loop, the row pointers and weights in registers | the same products and exact sums |
| the stem (one input channel, stride 2: `k3_c1_s2`) | `K3_BLOCK`: each of the 8 output channels loaded and widened the 9 inputs again | the 9 inputs loaded and widened once per block; a kernel row outside the input reads zeros instead of the centre row with masked weights | either way each such product is 0 and the sums are exact |

`chan_fast` and `layer_fast` (the form flags) exist only in the NEON build; the scalar setup keeps its 0 / 1, so the
PSP's and the hosts' objects are `main`'s. Kept as they were after trying: `-O3` (61.9 → 69.7 ms per region: NEON/VFP
switches), and the partial last block of a row in the lane form (gcc then spilled in the common path). Not done: an
int8-pair form of the 1×1 layers (`VMULL.S8` + `VMLAL.S8` + `VPADAL`, as the multi-scale path's `pw_sums`), est.
−12% of the 1×1 work, which needs a transposed input block and a check that no weight is −128 (two products of
−128 × −128 overflow int16).

## Per layer (no prefetch credit, ms per frame @333)

`vita/a9_layers.py compare` on ocr-speed's 24 development lines (60-100 CTC frames, 1,609 frames, `sub24.bin`
`3162151a…`), the texts the same before and after:

| phases | Small before | after | Normal before | after | Large before | after |
|---|---:|---:|---:|---:|---:|---:|
| the head (S: 96 → 4,987, full; N: 96 → 5,631 with the gate and two-level head; L: two-level) | 0.867 | **0.720** | 0.369 | 0.337 | 0.632 | 0.593 |
| the 3×3 stem | 0.094 | 0.083 | 0.136 | 0.123 | 0.185 | 0.171 |
| the 10 pointwise layers (collapse included) | 0.973 | 0.935 | 1.888 | 1.809 | 5.931 | 5.715 |
| the 8 depthwise layers | 0.200 | 0.200 | 0.343 | 0.342 | 0.596 | 0.595 |
| the input columns, the gate's step, the rest of `vocr_run` | 0.061 | 0.061 | 0.071 | 0.071 | 0.071 | 0.071 |
| **all** | **2.195** | **2.000** | **2.807** | **2.683** | **7.415** | **7.146** |

The per-phase tables are in the PR (`vita/a9_layers.py compare`).

The high bound the calibration uses is instruction-bound in most phases (instructions ≥ NEON issue cycles), plus the
data cache misses: 25-44% of it, unchanged here because the tiles and their buffers are (a larger tile would halve the
late layers' weight stream but grow the arena by 87-183 KB per context: the research's rank 3, opt-in only).
Instructions per frame: S 529,642 → 476,117, N 626,586 → 590,017, L 1,349,842 → 1,277,225 (with `-DVOCR_NO_ATTN`:
475,001, 588,588 and 1,275,031).

textspot8 on a 900×150 region: 23.89 M → 16.82 M instructions; the 1×1 layers 14.40 M → 9.44 M, the depthwise layers
and the rest of the stream 6.93 M → 3.97 M + 0.59 M (the stem's own function), the dark pass 1.75 M (unchanged).

## Bit identity

Every check compares `main` (205f47e) with this change on development data only (no protected set):

- **Recognizer, five builds** (`vita/speed_parity.py`: host gcc and clang `-O2`, Cortex-A9 NEON and scalar with the
  Arm GNU toolchain under qemu-arm, and the Vita compiler's objects linked into the ARM CLI), the 884 development lines
  of `vita/dev_pack.py` (31,479 frames, `dev_res.bin`), VOCR6.2 S, N and L:
  - W8I and W8F, every logit of every frame and every frame's top 8 (the full head, which a run asking for the logits
    computes): the same in every build, before and after;
  - W8I (and W8F for N and L) with Rubi's options and no logits, so that the head gate and the two-level head run as
    in Rubi: the texts and every frame's top 8, blank logit and line confidence the same (`--no-logits`, new).
- **`make check-arm`** (the C unit tests, known-answer logits of every NEON kernel, the head's ties, -0 / +0 and NaN in
  W8I, W8F and F32, the top 1, 3, 8 and 16 against the per-class insertion; NEON and scalar under qemu) and **`make
  parity-arm`** (the PyTorch reference, f32 / w8f / w8i, NEON and scalar) pass.
- **Spotter against its twin on NEON:** `tests/textspot6_parity.py` with the Vita compiler's objects (both forms)
  linked into ARM drivers under qemu: 1,214 runs, 139 blobs (115 of version 4, among them wild-requantisation networks
  with −128 weights and shifts below 32, op graphs, 16 probes on designed maps; 24 of versions 1-3 against
  `vocr_textspot3_test`), 216 map checks, 343 loader cases: equal.
- **Spotter, five builds × both forms** (`vita/ts_speed_parity.py`, new): the maps the decode reads and the boxes,
  classes, blocks, order, pick and presence of 1,194 records: the 363 development regions of `dev_regions_res.bin`
  (as Rubi's crop decode reads them, by the default call with the dark pass, and in grey) and 27 procedural frames
  (noise, odd sizes, padded strides, odd arenas, both scales). 2,388 map lines and 1,194 box lines per run, the same
  before and after in all 10 builds (sha256 `0ed9d754…` and `7150d714…`).
- **Vita3K** (`vita/vita3k_parity.py`, new; Vita3K build 4154): `vita/par`'s app runs the runtime's own host drivers
  on the Vita compiler's code. 11 jobs: every logit of the 24 timing lines in W8I and W8F for each tier (the full head,
  32-36 MB each), the texts and every frame's top 8 of the 884 development lines with Rubi's options for each tier,
  and textspot8's boxes and decode maps of the 363 development regions, as Rubi's crop decode reads them and by the
  default call (27 MB of map lines). All 20 output files are byte-identical in the four runs: `main` on the host and
  in Vita3K, this change on the host and in Vita3K (about 290 s of emulation each).
- **No other build changes:** the `.text` of host gcc 11.5 and clang 18.1.8 `-O2`, the Vita compiler and Arm GNU 14.2
  with `-DVOCR_NO_NEON` (also with `-DVOCR_TEXTSPOT6_NO_MS`) and psp-gcc 15.2 (VFPU 0 and 1) objects of both files is
  byte-identical to `main`'s. A host gcc `-O3 -march=native` build gives the `-O2` build's logits, top 8, texts, maps
  and boxes too.

## Records

- `eval/c_speed_20261101.json` (`eval/c_speed_record.py`): every number above from the measurement directories, with
  the source hashes before and after, the parity runs' hashes, the arenas and the builds whose machine code is
  unchanged. Hashes and counts only. `tests/test_c_speed_tools.py` checks it against the files and this page.
- `eval/runtime_repins.json`: the two files' changes after protocols and records had pinned them, with this record as
  the evidence that every read those pins stand for is unchanged. The tests of frozen pins accept a re-pinned file
  (`tests/repins.py`): `eval/textspot6_a9_20261026.json`'s sources (its times are the earlier code's) and
  `eval/vocr6_final_test_protocol.json`'s checkout. `eval/textspot3_protocol.json` re-pins `runtime/vocr.c` by its
  amendment 13, as amendments 8-12 did for earlier runtime changes. The final test's harness itself still accepts
  only the runtime pair its runtime check recorded, so it refuses to read again with this runtime.

## Memory

- **Arenas: unchanged** (neither plan changed). `vocr_arena_bytes` at 1,024 px with Rubi's options: 124,832 (S),
  184,448 (N), 262,480 B (L). `vocr_textspot6_arena_bytes`: 235,119 B for a 900×150 region at scale 2, 244,607 B
  for 960×544.
- **Code** (VitaSDK gcc 15.2 `-O2`, Thumb-2, NEON objects): `vocr.o` 26,435 → 26,723 B (+288: the stem kernel, the
  filter, the `asm` pass in place of the intrinsics one), 24,083 B with `-DVOCR_NO_ATTN`; `vocr_textspot6.o` with
  `-DVOCR_TEXTSPOT6_NO_MS` (Rubi's) 13,516 → 15,228 B (+1,712), its `--gc-sections` text 13,020 → 14,664 B. As Rubi
  builds the two (`-DVOCR_NO_ATTN`, `-DVOCR_TEXTSPOT6_NO_MS`, `-ffunction-sections`) they shrink: 26,423 + 13,532 →
  24,083 + 15,240 B (−632 B). The scalar and PSP objects do not change.
- **Stack** (`-fcallgraph-info=su`, the heaviest path): the recognizer's deepest chain 2,536 → 2,320 B. Before it was
  `vocr_run` → `run_head` → `mm_i16` (616 B); `mm_i16`'s frame is 336 B now, so the deepest path goes through
  `mm_f32` (W8F and F32 only), and W8I's (Rubi's mode) is 2,256 B. The spotter's peak stays the dark pass's (find
  2,720 → 2,728 B); its call without the dark pass (Rubi's) 1,136 → 1,384 B with `-DVOCR_TEXTSPOT6_NO_MS` (the 1×1
  kernel's 128-byte sum block and the new kernels' frames).

## What does not transfer from the ONNX findings

From astra's report and vocr-research's (§4-5 of `reports/vocr-research.md`):

- **FP16:** the A9's VFPv3 and NEON convert half precision but compute none (half-precision arithmetic came with
  Armv8.2-A); the weights are already int8.
- **NVFP4:** no FP4 or FP8 arithmetic on the A9, and a scale per 16 values would replace the exact int32 sums.
- **Int8 activations (W8A8, calibrated QDQ):** they change text (astra: 93 and 50 of 345; the research: 86-216 of
  1,729 per tier). The exact INT8 form (16-bit activations split into two bytes) helps x86's int8 dot-product units;
  on the A9 `VMLAL.S16` by scalar already does the exact int16 × int8 product at 4 per cycle, and a split would
  double the multiplies.
- **FP32 compute (W8F):** 1.8-2.5x slower than W8I on the A9 cost model.
- **Threads, im2col:** Rubi already reads a box's lines in parallel (`ocr_threads`); neither side uses im2col.
- **Fusion:** `mm_i16` already applies scale, bias, ReLU and the residual in its store, and the spotter requantises in
  the kernel; what this change fused is the spotter's per-channel requantisation constants (4 channels per `VLD4`).
- **Skipping head work** (astra's note that the ONNX graphs compute the whole head): the C runtime already does it for
  Normal and Large. Small has neither head section; a two-level head for VOCR6.2-S is the research's rank 1 (est.
  −0.10 s per box, −16%), but it changes the frames where the escape misses a rare class, so it needs a fit and #122's
  development rule, a model change outside this bit-exact one.

## Host speed (x86, unchanged)

The changes are NEON-only, so host builds run `main`'s code. One pinned core of the EPYC host (load 12-20), gcc 11.5,
median of 5 runs (`tools` in `$HERD/shared/artifacts/vocr-c-speed/`):

| | `-O2` (the Makefile's) | `-O3 -march=native` |
|---|---:|---:|
| recognizer, ms per line (24 lines of 60-100 frames, top-8, Rubi's options) S / N / L | 25.9 / 31.5 / 78.3 | 6.2 / 7.2 / 15.4 |
| textspot8, ms per region (60 development regions, Rubi's crop decode) | 7.13 | 2.96 |

Before and after agree to 0.4%. These lines average 67 CTC frames, about twice the 32.5 of the research's timing set,
so the per-line times are about twice its.

This checks astra's main desktop finding against our best gcc build rather than redoing the ONNX side, which
vocr-research reproduced (#164, its §2): astra's C reference was the MSVC `/O2 /fp:strict` build (12.30 / 13.77 /
34.05 ms per line, S / N / L, against 3.82 / 5.61 / 8.99 for the faithful ONNX graphs and 1.47 / 2.31 / 3.58 for the
fast ones). Built with `-O3 -march=native`, the same unchanged C source runs 4.2-5.1x faster here than gcc `-O2`, and
the spotter 2.4x, with the same logits, top 8, texts, maps and boxes (checked on these 24 lines and 726 region
records). So most of the gap astra saw is the desktop compiler's: `vocr.c` and `vocr_textspot6.c` have hand-written
NEON for the Vita and none for x86. For desktop builds, `-O3` with the target's SIMD (keeping `-ffp-contract=off`) is
the cheap win (the research's §6); the Vita's own build stays `-O2` (above).

## Reproduce

```sh
export VITASDK=... ARM_CC=.../arm-none-linux-gnueabihf-gcc VOCR_A9_PLUGINS=<plugins built from vita/qemu/insn_mix.c>
python vita/a9_layers.py measure --blob VOCR6.2_N_w8g2.vocr --pack sub24.bin --out a9/new --no-pld \
    --cli-opts '--topk 8 --head-gate on --two-level'          # --runtime <main's runtime/> for before
python vita/a9_layers.py compare a9/main a9/new --scale 1.000676 --console-mhz 333
    # measure --vocr-cflags '-O2 -DVOCR_NO_ATTN': Rubi's build
python eval/textspot6_a9.py measure --out ts/new --rows t64n8c16_t4 --sizes 900x150,960x544 \
    --define VOCR_TEXTSPOT6_NO_MS [--nodark]                 # from a checkout of main for before
python vita/speed_parity.py --base <main's runtime/> --new runtime --pack dev.bin --blob ... --modes w8i,w8f \
    --cli-opts '--topk 8' [--no-logits --cli-opts '--topk 8 --head-gate on --two-level' \
    --builds host-gcc,host-clang,arm-neon,arm-scalar,vita-neon,vita-neon-noattn]
python vita/ts_speed_parity.py --base <main's runtime/> --new runtime --blob textspot8_t64n8c16_w8.vtsp \
    --regions dev_regions.bin --frames 27
make -C runtime check-arm parity-arm ARM_CC=... QEMU=...
python vita/vita3k_parity.py --base <main's runtime/> --new runtime --sub sub24.bin --pack dev.bin \
    --regions dev_regions.bin --spot textspot8_t64n8c16_w8.vtsp --blob S=VOCR6.2_S_w8.vocr \
    --blob 'N=VOCR6.2_N_w8g2.vocr:--head-gate on --two-level' --blob 'L=VOCR6.2_L_w8_2l.vocr:--two-level' --work DIR
```

The packs hold crops of copyrighted screenshots (development data only, `vita/dev_pack.py`): keep them local.

## Not measured

- Nothing here has run on a console; every console time is the cost model calibrated on one console figure (VOCR3-H15).
  The no-prefetch and prefetch columns bound what the A9's `PLD` gives.
- The instruction cache: the cost model does not charge it, and the code grew by 2.0 KB in the two hot objects (as
  Rubi builds them, it shrank by 0.6 KB).
