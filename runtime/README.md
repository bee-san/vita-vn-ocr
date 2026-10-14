# runtime: the recognizer in plain C

`vocr.h` and `vocr.c` run `train/model.py` checkpoints on the console. `train/export.py` turns a checkpoint into a weights blob. The dialogue line finder that feeds it lives next to it (`vocr_lines.c`, `docs/linefind.md`); the two share nothing but naming. `make check` runs the checks of both.

- C99. The runtime uses no `malloc` and nothing from libc except `memcpy`, `memset` and `<math.h>` (on 32-bit ARM, also libgcc's integer-division helpers; see the porting notes).
- The blob is read in place. All working memory comes from one arena that the caller provides, sized exactly by `vocr_arena_bytes()`.
- The line is processed in tiles of a few CTC frames (8 input columns each). Every layer keeps its left context between tiles, so the arena stays the same size for any line width, and the output doesn't depend on the tile size (checked bit for bit).
- NEON kernels sit behind `__ARM_NEON` (Vita). The scalar fallback gives the same bits; it's the PSP path, and `-DVOCR_NO_NEON` forces it.

## Numbers (v1 checkpoint)

Memory, measured with `vocr_arena_bytes()` (equal to `export.plan_arena()`; the parity test checks this) at the default tile of 4 frames and head batch of 16:

| | fp32 | int8, float sums (`w8f`) | int8, int32 sums (`w8i`) |
|---|---|---|---|
| weights blob | 3,067,840 B | 859,904 B | 859,904 B |
| arena for lines up to 256 px | 116,384 B | 118,432 B | 124,832 B |
| arena for lines up to 640 px | 116,384 B | 118,432 B | 124,832 B |
| arena for lines up to 1024 px | 116,384 B | 118,432 B | 124,832 B |
| `vocr_model` + `vocr_ctx`, 32-bit ARM | 1,772 + 1,984 B | same | same |
| code (`vocr.o`, ARM `-O2`, NEON / scalar) | 13,843 / 11,699 B (VitaSDK gcc 15.2: 14,583 / 12,455 B) | same | same |
| total | ≈ 3.2 MB | ≈ 0.99 MB | ≈ 1.0 MB |

For comparison, the research measured the trimmed int8 PP-OCRv5 at a 13.4 MiB peak in the overlay.

Stack on ARM, from gcc's call graph (`-fcallgraph-info=su`; every frame is static):
- The deepest chain is `vocr_arena_bytes()` → `plan()`, 2,664 B, because it plans into a local `vocr_ctx`.
- `vocr_run()` → `step_op()` → `mm_i16()` (w8i) is 1,400 B with NEON and 1,000 B scalar. Through `mm_f32()` (f32, w8f) it is 1,224 / 1,168 B. (Before `psp`'s PR #13 register-blocked the scalar kernels `step_op` also compiles into, these were 1,264 / 1,104 and 1,128 / 960 B; this PR's own `mm_i16` NEON change adds 8 B on top of #13's base, 1,392 to 1,400.)
- On the Vita (VitaSDK gcc 15.2), the test app's probe thread (`vita/app/main.c`) measured 4,316 B of stack at `2873b98` (4,100 B at an earlier head) for `vocr_arena_bytes()`, `vocr_init()`, `vocr_run()` on a 1024-px line and both line finders, thread entry included (in Vita3K, which runs the Vita's own machine code).

Tiles trade arena for speed (bytes; host speed is in the next table):

| tile / head frames | fp32 | w8f | w8i |
|---|---|---|---|
| 1 / 1 | 51,744 | 59,168 | 53,856 |
| 4 / 16 (default) | 116,384 | 118,432 | 124,832 |
| 16 / 16 | 351,488 | 353,536 | 385,280 |
| 128 / 128, 1024 px (one tile, no streaming) | 2,589,696 | 2,591,744 | 2,860,032 |

Compute is 1,002,720 multiply-accumulates per CTC frame: 488,640 in the backbone and 514,080 in the head. A 640-px line is 80 frames, about 80 M MACs, against 1.52 G for PP-OCRv5 mobile at height 48 (research). The research puts one Cortex-A9 core at 444 MHz at 0.24-0.30 GMAC/s in fp32 with ncnn. By that figure, a 640-px line would take about 0.27-0.33 s. The later cycle model of the Vita build (`vita/README.md`) puts it at 113-212 ms in w8i. Both are **estimates**: the runtime has run on a console only inside vita-jp-overlay (FL10, `docs/vita-memory-budget.md`), without per-line timing.

### Larger students (owner goal of 2026-10-09: at most 50 MB, under about 1 s per line)

The runtime takes any `TinyVNOCR` width and bottleneck. Shapes only (random weights), computed with `export.py` and `plan_arena()`:

| width | bottleneck | params | int8 blob | fp32 blob | arena w8f / w8i | MACs per frame | 640-px line |
|---|---|---|---|---|---|---|---|
| 1.0 (v1) | 96 | 0.76M | 0.86 MB | 3.07 MB | 118 / 125 KB | 1.00M | 80M MACs |
| 1.5 | 96 | 1.04M | 1.16 MB | 4.18 MB | 173 / 183 KB | 1.56M | 125M |
| 2.0 | 96 | 1.42M | 1.56 MB | 5.72 MB | 229 / 241 KB | 2.34M | 187M |
| 2.0 | 192 | 1.97M | 2.11 MB | 7.93 MB | 236 / 248 KB | 2.89M | 231M |
| 3.0 | 192 | 3.08M | 3.25 MB | 12.36 MB | 346 / 365 KB | 5.09M | 407M |

Memory stays far under 50 MB at every size here, so speed is what limits a student. The cycle model of the Vita build (`vita/a9_cycles.py`, w8i, one 444 MHz A9 core) puts a 640-px line at about:
- 112-210 ms at width 1.0;
- 234-411 ms at width 2.0;
- 420-965 ms at width 3.0, whose 2.68 MB of weights no longer fit the 2 MiB L2.

The research's 0.24-0.30 GMAC/s (ncnn fp32) gave 0.3, 0.6-0.8 and 1.4-1.7 s. All of these are **estimates**; this runtime's NEON speed on a real A9 hasn't been measured. `docs/vita-memory-budget.md` has the full grid, the L2 effect and the size ceiling new models design to. `export.py --info` prints the MACs per frame and per line of any blob.

### Correctness (v1)

`tests/parity.py` (below) passed every check in all three modes on all of these runs:

| C build | lines, at this head | lines, at earlier heads |
|---|---|---|
| gcc 11.5 `-O2`, `libvocr.so` | eval's hand-checked **vita-test** (206 lines, 16 games; grey, and RGBA with `--set vita-test`) and **psp-test** (245 lines, 14 games; eval PR #5, not merged yet), as the GT-box crops of `eval/sets.py`; all 984 `data/real` crops (RGBA input); synth-val lines 1,001-4,000 | synth-val (the held-out shard) lines 1-1,000 and 9,001-10,000 |
| Arm GNU gcc 14.2, Cortex-A9 **NEON**, qemu-arm 10.0.13 | vita-test, psp-test | synth-val first 100; real first 50 |
| same, **scalar** (`-DVOCR_NO_NEON`) | vita-test, psp-test | synth-val first 100; real first 50 |
| gcc 11.5 / clang 15, ASan + UBSan (`vocr_cli`) | vita-test (gcc), psp-test (clang) | synth-val first 100; random fixture, 40 |
| CI's `ci/runtime.sh gcc` / `clang` (locally and on GitHub's ubuntu-24.04) | CI's random-init fixture on its 48-line OFL-font shard | same |

The earlier heads compute the same bits: the last arithmetic change, the w8i chunking, leaves every result for reductions up to 512 terms unchanged (every v1 layer), and `VOCR_INVERT` doesn't touch the default path. Only this head has the `VOCR_INVERT` check.

"Every check" means:
- preprocessing matches exactly;
- the logits are within tolerance of PyTorch;
- the strings equal PyTorch's, apart from near ties (the rule is under "Build and test"). Only w8i ever needed it: 1 line of psp-test, 1 of CI's fixture shard and 2 of the 40 `--fixture` crops;
- the logits are bit-identical to `tests/vocr_ref.py`;
- the logits are bit-identical across tile sizes;
- `VOCR_INVERT` equals inverting the crop before the resize, bit for bit;
- the arena plan matches.

Since the NEON, scalar and host builds all equal the same numpy model bit for bit, they equal each other.

The largest logit difference from PyTorch, relative to the largest logit, was 6.1e-6 in f32, 5.4e-6 in w8f and 4.4e-4 in w8i.

Through eval's own harness (`eval/run.py --model 'cmd:runtime/build/vocr_cli v1_f32.vocr --stdin' --set vita-test`), the f32 runtime gives exactly the PyTorch checkpoint's strings on all 206 lines: CER 23.75%, 52.43% exact, the same in every style row. w8i gives 23.87%.

What int8 costs:
- On synth-val lines 1-1,000, CER against the labels is 0.41% in f32, w8f and w8i alike (95.4% of lines exact); int8 changed 2 strings, both already wrong in f32. On lines 1,001-4,000 it is 0.33% in f32 and 0.34% in both int8 modes (9 of 3,000 strings change), and on lines 9,001-10,000 0.37% in every mode (1 string changes).
- On the 984 real crops, int8 changed 99 strings. Every one was a line where the f32 model's top-1/top-2 margin fell below 0.5 somewhere (median 0.04); on unchanged lines the median margin is 0.89.
- On the 336 crops with a pseudo-label (agreement of PP-OCRv5 server and manga-ocr, not hand-checked), CER was 9.74% for f32, 9.71% for w8f and 9.68% for w8i.
- On eval's hand-checked sets, f32 → int8 moves CER from 23.83% to 23.95% on vita-test (15 of 206 strings change) and from 46.69% to 46.81% on psp-test (27 of 245); w8f and w8i give the same CER. These are `parity.py`'s unnormalised CERs; eval's normalised vita-test CER goes from 23.75% to 23.87%.

So int8 costs about 0.1 point of CER on the real test sets, and nothing measurable on synthetic validation. The real-screenshot benchmark number comes from `eval/` (`docs/results.md`).

What `VOCR_INVERT` buys v1, on the same GT-box crops in w8i (inverting where `synth/common.line_polarity`, the line finder's polarity rule, says the text is dark, as the console would):

| set | without inversion | with the polarity rule |
|---|---|---|
| vita-test | 23.87% | 9.05% |
| psp-test | 46.57% | 10.80% |

This is CER with `eval/metrics.py`'s normalisation, on ground-truth boxes, so it is not eval's pipeline number. The psp-test row is re-measured on the merged psp-test manifest (eval #5) with `eval/run.py --model 'cmd:runtime/build/vocr_cli v1_w8.vocr --mode w8i --stdin' --set psp-test`, with and without `--polarity console`; before the merge it read 46.77% and 11.04%.

### Host speed

Setup: x86-64, one thread pinned to one core, the first 100 synth-val lines (5,767 CTC frames, 58 frames per line on average), `vocr_cli --pack ... --repeat 1`. Each cell is the median (range) of interleaved runs: 5 for the 4/16 rows, 3 for the others. The box is shared and was loaded (load average 20-30 on 16 cores), so only the ratios mean much. qemu timings say nothing about a Cortex-A9 and aren't reported.

| build | tile/head | f32 ms/line | w8f ms/line | w8i ms/line |
|---|---|---|---|---|
| gcc 11.5 `-O2` (the Makefile default) | 4/16 | 60.3 (53.3-63.8) | 67.0 (61.9-69.9) | 48.4 (47.1-48.8) |
| gcc 11.5 `-O2` | 1/1 | 65.2 (62.7-70.0) | 112.5 (111.5-115.5) | 50.0 (47.8-55.1) |
| gcc 11.5 `-O2` | 128/128 (no streaming) | 64.0 (62.7-68.3) | 64.1 (64.1-66.6) | 48.7 (48.3-52.8) |
| clang 15 `-O2` | 4/16 | 8.0 (7.0-9.2) | 10.8 (9.1-12.1) | 18.6 (16.3-19.9) |

- Streaming costs nothing measurable at the default tile: 4/16 against 128/128.
- In w8f, tiny tiles hurt, because each int8 weight block is dequantised again for every tile.
- gcc 11 at `-O2` doesn't auto-vectorise the scalar loops, and its speed depends on code layout. In a 7-round interleaved A/B, w8i took 51.4 ms before the w8i chunking (4d80728), 70.6 ms with it (77d5b85) and 53.9 ms with this head's unrelated `VOCR_INVERT` edit. Built with `-falign-functions=64 -falign-loops=64`, the same three ran at 48.4, 45.0 and 48.1 ms. So the chunking costs nothing measurable, and single gcc `-O2` numbers can be a third off either way.
- With `OPT=-O3`, gcc took 8.3-11.8 ms per line in f32 and 16 ms in w8i (the PR author's 2 runs, at an earlier head). The outputs stay bit-identical (parity passes on an `-O3` build), because vectorising runs across output channels, not along a sum.
- On the Vita the NEON kernels are explicit, so none of this decides speed there.
- The table predates the register-blocked scalar kernels (`psp/README.md`). With them, gcc 11.5 `-O2` on one core of the same loaded host, over psp-test, went from 36.9 to 12.6 ms per line in f32, 39.1 to 15.5 in w8f and 27.4 to 14.3 in w8i (median of 3 interleaved runs each), with identical logits.

## API

```c
#include "vocr.h"

static vocr_model model;   /* about 1.8 KB on 32-bit ARM; points into the blob */
static vocr_ctx ctx;       /* about 2 KB */

/* blob: the .vocr file in memory, 16-byte aligned, kept alive while in use */
int rc = vocr_load(&model, blob, blob_bytes);
vocr_opts opts = {0};                       /* defaults: tile 4, head 16, max width 1024,
                                               AUTO mode (f32 or w8f, from the blob) */
opts.mode = VOCR_MODE_W8I;                  /* int8 blob, int32 sums */
size_t need = vocr_arena_bytes(&model, &opts);
/* arena: need bytes, 16-byte aligned, private to ctx */
rc = vocr_init(&ctx, &model, &opts, arena, need);

vocr_image img = { pixels, width, height, stride_bytes, VOCR_RGBA8 };   /* or VOCR_GREY8 */
if (line.dark) img.format |= VOCR_INVERT;  /* dark text (vocr_lines.h VocrLine.dark) */
char text[1024];
rc = vocr_run(&ctx, &img, text, sizeof text, NULL);    /* NUL-terminated UTF-8 */
```

- One context runs one line at a time. Use one context per thread, each with its own arena; the model can be shared.
- `vocr_run()` takes one line crop at any size up to 65535 px, resizes it to height 32 and clamps the width to 1024, as `train/train.py` does. It returns `VOCR_E_WIDTH` if the resized width is over `opts.max_width`.
- `vocr_extra` is optional. It can return the logits and per-frame argmax, and takes a cancel callback that is polled once per tile.
- Errors are negative `VOCR_E_*` codes; `vocr_strerror()` describes them. On `VOCR_E_TEXT` the text holds whole characters only.
- `vocr_preprocess()` returns the exact grey network input, and `vocr_check_crc()` checks a blob's CRC-32.
- `opts.tail_conf` (1/1000, 0 = off) drops the trailing run of low-confidence characters: a character at the end of the line whose emitting CTC frame has softmax-max confidence below `tail_conf/1000` is dropped, and so is every character after it; interior characters are kept. `tail_conf = 0` reproduces the untrimmed output bit-for-bit and allocates no extra arena. It targets the finder's wide speaker-name crops, where the recognizer appends glyphs read off the plate art after the name (`docs/confidence-trim.md`).
- `VOCR_INVERT`, or'ed into `vocr_image.format`, reads every source pixel as `255 − grey` before the resize. Training does exactly this for dark text (`synth/common.line_crops`), so set it whenever the line finder reports `VocrLine.dark = 1` (the polarity contract in `docs/linefind.md`). It needs no copy of the crop. Inverting after the resize would not be the same: on 500 random crops, `cv2.resize(255 − g)` and `255 − cv2.resize(g)` differ on 25% of the pixels, by 1.

From Python (host tests, evaluation), `pyvocr.py` wraps `build/libvocr.so` with ctypes:

```python
from pyvocr import Vocr
r = Vocr("v1_w8.vocr", mode="w8i")                  # or Vocr.from_checkpoint("ckpt_best.pt")
text = r.text(crop)                                 # HxW uint8 grey or HxWx4 uint8 RGBA
text = r.text(crop, invert=True)                    # dark text (VOCR_INVERT)
text, logits = r.run(crop, logits=True)
```

`pyvocr` rebuilds `build/libvocr.so` whenever `vocr.c` or `vocr.h` is newer than it, and refuses crops that aren't `uint8`.

From eval's harness (PR #2), use its `cmd:` protocol. The CLI reads PGM or PPM paths on stdin and prints one line each:

```sh
python eval/run.py --model 'cmd:runtime/build/vocr_cli v1_w8.vocr --mode w8i --stdin' --set vita-test
```

`vocr_cli --invert` sets `VOCR_INVERT` for every crop it reads, and an unknown `--mode` is an error.

## Modes and exactness

| mode | weights | pointwise and head arithmetic |
|---|---|---|
| `f32` | float32 | float32 |
| `w8f` | int8, one float32 scale per output channel | weights dequantised to float32 per 8-channel block, then float32 |
| `w8i` | same | activations quantised to int16 per position (max-abs), exact int32 sums over chunks of 512 terms (added in float), one float rescale |

The stem, the depthwise layers, the biases and the scales stay float32 in every mode. In the 859,904-byte int8 blob that's 87 KB:
- stem and depthwise weights: 29,568 B;
- biases: 30,528 B;
- scales: 26,944 B.

The rest is 744,960 B of int8 matrices (the head alone is 514,560 B) and 27,904 B of vocabulary and headers.

Every output is computed in a fixed order: bias first, then each product added and rounded on its own. The NEON code only vectorises across output channels, so it gives the scalar code's bits. Build with `-ffp-contract=off` (the Makefile does), so that compilers don't fuse `a*b+c`. `tests/vocr_ref.py` is a numpy model of exactly this arithmetic, and the parity test requires the C logits to equal it bit for bit.

ARMv7 NEON flushes denormals to zero while VFP and x86 don't, so in principle the NEON build could differ wherever a denormal appears. On every line tested so far, the NEON logits were bit-identical to the scalar and host ones.

## Weights blob

`train/export.py` writes the blob:

```sh
python train/export.py --ckpt models/v1/ckpt_best.pt --out v1_f32.vocr
python train/export.py --ckpt models/v1/ckpt_best.pt --out v1_w8.vocr --int8
python train/export.py --info v1_w8.vocr          # sizes and arena bytes at 256/640/1024
```

It folds BatchNorm into the convolutions (in float64) and maps the model to five op types: one-input-channel conv (the stem), depthwise conv, pointwise conv (including the 2×1 height collapse as a pointwise over two rows), and the head. With `--int8`, every pointwise and head matrix becomes symmetric per-channel int8.

The blob is version 1:
- a 64-byte header: magic `VOCR`, version, sizes, a CRC-32 and the checkpoint's SHA-256 prefix;
- one 64-byte record per op;
- 16-byte-aligned weight sections, with the matrices packed `[cout/8][K][8]`;
- the vocabulary as UTF-8 with 16-bit offsets.

The full layout is in the docstring of `train/export.py`. `vocr_load()` rejects anything it doesn't run: any other version, layer, residual or shape. A model change that adds an op therefore fails at load time and in CI's op-set test.

## Build and test

```sh
make -C runtime                  # build/vocr_cli, build/libvocr.so
make -C runtime check            # line finder checks, then C unit tests + PyTorch parity on a
                                 # random-init fixture, built with ASan + UBSan (CI: ci/runtime.sh)
python runtime/tests/parity.py --ckpt models/v1/ckpt_best.pt               # synth val shard
python runtime/tests/parity.py --ckpt models/v1/ckpt_best.pt --set real    # RGBA crops
python runtime/tests/parity.py --ckpt models/v1/ckpt_best.pt --set vita-test   # eval's set, RGBA
```

`--set vita-test` (and `psp-test`, once its manifest is on main) reads the crops through `eval/sets.py`, with the screenshots from `--eval-data`, `$VOCR_EVAL_DATA` or `data/eval` (`eval/fetch.py` rebuilds them).

Per line and mode, `tests/parity.py` checks:
- the grey input equals `cv2.resize(INTER_LINEAR)` at the model's height exactly;
- the logits are within tolerance of PyTorch (the original model for f32; a PyTorch forward of the int8 blob for w8f and w8i);
- the decoded string equals PyTorch's, or differs only through near ties: every frame whose argmax differs has a PyTorch top-1/top-2 margin of at most twice the measured logit difference (reported as `text_diff_tie_only`);
- the logits equal `tests/vocr_ref.py` bit for bit, so the C string is exactly the reference model's even where it differs from PyTorch on a near tie;
- the logits are identical across several tile and head-batch sizes (six through the library, three through a CLI);
- `VOCR_INVERT` gives exactly the input and logits of the crop inverted before the resize (on the same lines as the tile check).

Near ties do occur, in w8i only so far. PyTorch's float layers sum in a different order from the C kernels, and w8i's int16 rounding amplifies those 1-ulp differences: the C logits differ from PyTorch's forward of the same arithmetic by up to about 4e-4 of the largest logit, so a frame whose top two classes are closer than that can flip. Examples: 1 line of CI's fixture shard (gcc and clang), and `psp-test:105`, where PyTorch's top-2 margin is 7.3e-5 against a logit difference of 6.6e-3.

It also checks `vocr_arena_bytes()` against `export.plan_arena()`. `--cli CMD` runs the C side through any `vocr_cli` build instead of the ctypes library, such as a sanitizer build or an ARM build under qemu.

ARM (Cortex-A9) under qemu-arm, with pinned tools fetched in user space:

```sh
runtime/scripts/fetch_arm_tools.sh                 # Arm GNU 14.2.rel1 + qemu-arm 10.0.13, sha256-checked
make -C runtime arm parity-arm ARM_CC=$PWD/runtime/scripts/tools/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-linux-gnueabihf/bin/arm-none-linux-gnueabihf-gcc
make -C runtime size ARM_CC=...                    # code size and stack frames
```

## Porting notes (vita, psp)

- Compile `vocr.c` alone with `-std=c99 -O2 -ffp-contract=off`. Vita: `-mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard`. PSP: the scalar path is picked automatically when `__ARM_NEON` is undefined.
- `mm_i16()` keeps its eight int32 accumulators in named variables. Written as an array (`int32x4_t a[8]`), gcc 14.2 and 15.2 stored all eight to the stack on every pass of the 32-`VMLAL` inner loop (16 `VSTR` there). Measured under qemu on the VitaSDK build, the change takes w8i from 973K to 841K instructions per CTC frame with bit-identical output (`vita/README.md`, "Cortex-A9 estimate").
- The Cortex-A9 has no integer divide, so on ARM `vocr.o` also needs libgcc's `__aeabi_idiv`, `__aeabi_uidiv` and `__aeabi_uidivmod` (VitaSDK links libgcc by default). Expect the same kind of helpers on MIPS. They are used per line and per tile, not per MAC.
- To match OpenCV bit for bit, preprocessing computes one `double` per output row and per resized column (`row_table`, `col_coef`). On the PSP, whose FPU is single precision, these are soft-float calls: a few thousand per line, against millions of MACs.
- The blob and the arena must be 16-byte aligned, and the blob must stay alive while in use. Both can live anywhere: a memblock, a static buffer, or the overlay's private heap.
- The blob format is little-endian, and `vocr_load()` refuses to run on a big-endian host. Vita and PSP are both little-endian.
- The runtime makes no system calls and has no global state, so it's safe from a plugin thread.
- PSP: `psp/` builds it with pspdev (`-O2 -G0`) into a test EBOOT and checks it in PPSSPPHeadless: the same bits as the host on vita-test, psp-test and synth-val in every mode (`psp/README.md`). The scalar kernels keep their sums in registers (2 positions × 8 outputs for f32, 1 × 8 for w8i, 8 channels for the stem and depthwise convs), which the PSP and the host's gcc `-O2` builds need to be fast. `-DVOCR_PSP_VFPU` (`make -C psp VFPU=1`) runs the float matrix products on the VFPU, with the same arithmetic in the same order; the thread needs `PSP_THREAD_ATTR_VFPU`, and `vocr_run()` overwrites VFPU matrices 0-4 without saving them (a caller that keeps values there across the call must save them).

## Not done here

- Only FL10 has run on a console (a PCH-1001, inside vita-jp-overlay), without a per-line timing or hardware CER (`docs/vita-memory-budget.md`), and nothing has run on a PSP; the console speed above is an estimate. `vita/` builds it with VitaSDK, checks it bit for bit in the Vita3K emulator and estimates Cortex-A9 cycles from the Vita build's instruction mix (`vita/README.md`); the `psp` agent covers the PSP.
- Parity covered 5,000 of the 10,001 synth-val lines (1-4,000 and 9,001-10,000), not the whole shard, because the `heavy.sh` slots stayed busy. The full shard is one command: `parity.py --ckpt ... --workers 5`.
- Real-screenshot CER belongs to `eval/`; the CER figures here only show that the runtime reproduces PyTorch, and what int8 and `VOCR_INVERT` change.
