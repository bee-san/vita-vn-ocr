# runtime: the recognizer in plain C

`vocr.h` and `vocr.c` run `train/model.py` checkpoints on the console. `train/export.py` turns a checkpoint into a weights blob. The dialogue line finder that feeds it lives next to it (`vocr_lines.c`, `docs/linefind.md`); the two share nothing but naming. `make check` runs the checks of both.

- C99. The runtime uses no `malloc` and nothing from libc except `memcpy`, `memset` and `<math.h>`.
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
| code (`vocr.o`, ARM `-O2`, NEON / scalar) | 12,287 / 9,779 B | same | same |
| total | ≈ 3.2 MB | ≈ 0.99 MB | ≈ 1.0 MB |

For comparison, the research measured the trimmed int8 PP-OCRv5 at a 13.4 MiB peak in the overlay.

Stack, from `-fstack-usage` on ARM:
- The deepest static chain is `vocr_arena_bytes()` → `plan()`, 2,664 B, because it plans into a local `vocr_ctx`.
- `vocr_run()` → `step_op()` → `mm_f32()` is 1,136 B with NEON and 968 B scalar.

Tiles trade arena for speed (bytes; host speed is in the next table):

| tile / head frames | fp32 | w8f | w8i |
|---|---|---|---|
| 1 / 1 | 51,744 | 59,168 | 53,856 |
| 4 / 16 (default) | 116,384 | 118,432 | 124,832 |
| 16 / 16 | 351,488 | 353,536 | 385,280 |
| 128 / 128, 1024 px (one tile, no streaming) | 2,589,696 | 2,591,744 | 2,860,032 |

Compute is 1,002,720 multiply-accumulates per CTC frame: 488,640 in the backbone and 514,080 in the head. A 640-px line is 80 frames, about 80 M MACs, against 1.52 G for PP-OCRv5 mobile at height 48 (research). The research puts one Cortex-A9 core at 444 MHz at 0.24-0.30 GMAC/s in fp32 with ncnn. By that figure, a 640-px line would take about 0.27-0.33 s. That's an **estimate**: nothing has run on a Vita yet.

### Correctness (v1)

`tests/parity.py` (below) passed every check in all three modes on all of these runs:

| C build | lines |
|---|---|
| gcc 11.5 `-O2`, `libvocr.so` | synth-val (the held-out shard), first 1,000 |
| gcc 11.5 `-O2`, `libvocr.so` | all 984 `data/real` crops, RGBA input |
| Arm GNU gcc 14.2, Cortex-A9 **NEON**, qemu-arm 10.0.13 | synth-val first 100; real first 50 |
| same, **scalar** (`-DVOCR_NO_NEON`) | synth-val first 100; real first 50 |
| gcc 11.5 and clang 15, ASan + UBSan (`vocr_cli`) | synth-val first 100; random fixture, 40 |
| CI's `ci/runtime.sh gcc` / `clang` (locally and on GitHub's ubuntu-24.04) | CI's random-init fixture on its 48-line OFL-font shard |

"Every check" means:
- preprocessing matches exactly;
- the logits are within tolerance of PyTorch;
- the strings are identical;
- the logits are bit-identical to `tests/vocr_ref.py`;
- the logits are bit-identical across tile sizes;
- the arena plan matches.

Since the NEON, scalar and host builds all equal the same numpy model bit for bit, they equal each other.

The largest logit difference from PyTorch, relative to the largest logit, was 5.1e-6 in f32, 3.0e-6 in w8f and 4.4e-4 in w8i.

What int8 costs:
- On the 1,000 synth-val lines, CER against the labels is 0.41% in f32, w8f and w8i alike (95.4% of lines exact). int8 changed 2 strings, both already wrong in f32.
- On the 984 real crops, int8 changed 99 strings. Every one was a line where the f32 model's top-1/top-2 margin fell below 0.5 somewhere (median 0.04); on unchanged lines the median margin is 0.89.
- On the 336 crops with a pseudo-label (agreement of PP-OCRv5 server and manga-ocr, not hand-checked), CER was 9.74% for f32, 9.71% for w8f and 9.68% for w8i.

So int8 shows no measurable loss. The absolute real CER is not the benchmark figure; that comes from `eval/`.

### Host speed

Setup: x86-64, one thread, the first 100 synth-val lines (5,767 CTC frames, 58 frames per line on average), median and range of 3 runs. The box is shared and was loaded (load average 13-16 on 16 cores), so only the ratios mean much. qemu timings say nothing about a Cortex-A9 and aren't reported.

| build | tile/head | f32 ms/line | w8f ms/line | w8i ms/line |
|---|---|---|---|---|
| gcc 11.5 `-O2` (the Makefile default) | 4/16 | 53.4 (36.9-56.2) | 56.0 (40.8-58.5) | 47.8 (30.7-48.8) |
| gcc 11.5 `-O2` | 1/1 | 57.4 (42.8-59.3) | 103.4 (74.1-109.4) | 55.9 (37.8-61.8) |
| gcc 11.5 `-O2` | 128/128 (no streaming) | 53.1 (37.4-56.6) | 53.2 (38.5-56.4) | 52.1 (30.5-55.9) |
| clang 15 `-O2` | 4/16 | 6.7 (5.4-8.0) | 10.6 (7.2-12.5) | 17.0 (13.3-19.6) |

- Streaming costs nothing measurable at the default tile: 4/16 against 128/128.
- In w8f, tiny tiles hurt, because each int8 weight block is dequantised again for every tile.
- gcc 11 at `-O2` doesn't auto-vectorise the scalar loops. With `OPT=-O3`, gcc took 8.3-11.8 ms per line in f32 and 16 ms in w8i (2 runs). The outputs stay bit-identical (parity passes on an `-O3` build), because vectorising runs across output channels, not along a sum.
- On the Vita the NEON kernels are explicit, so none of this decides speed there.

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
char text[1024];
rc = vocr_run(&ctx, &img, text, sizeof text, NULL);    /* NUL-terminated UTF-8 */
```

- One context runs one line at a time. Use one context per thread, each with its own arena; the model can be shared.
- `vocr_run()` takes one line crop at any size up to 65535 px, resizes it to height 32 and clamps the width to 1024, as `train/train.py` does. It returns `VOCR_E_WIDTH` if the resized width is over `opts.max_width`.
- `vocr_extra` is optional. It can return the logits and per-frame argmax, and takes a cancel callback that is polled once per tile.
- Errors are negative `VOCR_E_*` codes; `vocr_strerror()` describes them. On `VOCR_E_TEXT` the text holds whole characters only.
- `vocr_preprocess()` returns the exact grey network input, and `vocr_check_crc()` checks a blob's CRC-32.

From Python (host tests, evaluation), `pyvocr.py` wraps `build/libvocr.so` with ctypes:

```python
from pyvocr import Vocr
r = Vocr("v1_w8.vocr", mode="w8i")                  # or Vocr.from_checkpoint("ckpt_best.pt")
text = r.text(crop)                                 # HxW uint8 grey or HxWx4 uint8 RGBA
text, logits = r.run(crop, logits=True)
```

From eval's harness (PR #2), use its `cmd:` protocol. The CLI reads PGM or PPM paths on stdin and prints one line each:

```sh
python eval/run.py --model 'cmd:runtime/build/vocr_cli v1_w8.vocr --mode w8i --stdin' --set vita-test
```

## Modes and exactness

| mode | weights | pointwise and head arithmetic |
|---|---|---|
| `f32` | float32 | float32 |
| `w8f` | int8, one float32 scale per output channel | weights dequantised to float32 per 8-channel block, then float32 |
| `w8i` | same | activations quantised to int16 per position (max-abs), int32 sums, one float rescale |

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
```

Per line and mode, `tests/parity.py` checks five things:
- the grey input equals `cv2.resize(INTER_LINEAR)` exactly;
- the logits are within tolerance of PyTorch (the original model for f32; a PyTorch forward of the int8 blob for w8f and w8i);
- the decoded string is identical;
- the logits equal `tests/vocr_ref.py` bit for bit;
- the logits are identical across several tile and head-batch sizes (six through the library, three through a CLI).

It also checks `vocr_arena_bytes()` against `export.plan_arena()`. `--cli CMD` runs the C side through any `vocr_cli` build instead of the ctypes library, such as a sanitizer build or an ARM build under qemu.

ARM (Cortex-A9) under qemu-arm, with pinned tools fetched in user space:

```sh
runtime/scripts/fetch_arm_tools.sh                 # Arm GNU 14.2.rel1 + qemu-arm 10.0.13, sha256-checked
make -C runtime arm parity-arm ARM_CC=$PWD/runtime/scripts/tools/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-linux-gnueabihf/bin/arm-none-linux-gnueabihf-gcc
make -C runtime size ARM_CC=...                    # code size and stack frames
```

## Porting notes (vita, psp)

- Compile `vocr.c` alone with `-std=c99 -O2 -ffp-contract=off`. Vita: `-mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard`. PSP: the scalar path is picked automatically when `__ARM_NEON` is undefined.
- The blob and the arena must be 16-byte aligned, and the blob must stay alive while in use. Both can live anywhere: a memblock, a static buffer, or the overlay's private heap.
- The blob format is little-endian, and `vocr_load()` refuses to run on a big-endian host. Vita and PSP are both little-endian.
- The runtime makes no system calls and has no global state, so it's safe from a plugin thread.

## Not done here

- Nothing has run on a PCH-1000 or a PSP; the console speed above is an estimate. The `vita` and `psp` agents build and run it there.
- Parity covered the first 1,000 of the 10,001 synth-val lines, not the whole shard. The `heavy.sh` slots stayed busy for hours, so these sampled runs went single-threaded instead. The full shard is one command: `parity.py --ckpt ... --workers 5`.
- The eval test sets haven't merged yet, so their parity isn't measured. Once they exist it's `parity.py --shard <file>`, or eval's `cmd:` backend. Real-screenshot CER belongs to `eval/`.
