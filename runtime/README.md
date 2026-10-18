# runtime: the recognizer in plain C

`vocr.h` and `vocr.c` run `train/model.py` checkpoints on the console. `train/export.py` turns a checkpoint into a weights blob. The dialogue line finder that feeds it lives next to it (`vocr_lines.c`, `docs/linefind.md`); the two share nothing but naming. `make check` runs the checks of both. An optional refinement of the finder's boxes (`vocr_lines_refine.c`: `vocr_refine_lines` / `vocr_refine_lines_rows`, called after the finder with its arena; flags 0 leave the lines unchanged) is described with its measurements in `docs/finder-tighten.md`; `make check-lines` runs its self-test and its parity with `synth/lines_refine.py`. The learned text detector (`vocr_textdet.c`, issue #96, `docs/textdet.md`) is a finder mode behind a flag; it has its own section below, and `make check-textdet` runs its checks.

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

### Console speed (VOCR3-H15, 2026-10-14)

On the owner's console (vita-jp-overlay #81, v0.8.0-test.1) VOCR3-H15 ran its lines of 60-100 CTC frames at
4,942 µs per frame (frame-weighted), which is `vita/a9_cycles.py`'s high bound x 1.0007 at 333 MHz (the clock is
assumed). The W8I kernels have since been rewritten for the Cortex-A9 (8-k NEON passes in `mm_i16` with a
prefetched weight stream, the head's argmax and the LM's top-K block filter in NEON lanes, pointer-stepped
depthwise taps), bit for bit: the estimate is 2.90 ms per frame for H15 with the top 8 at 333 MHz (1.75x; 1.51x
if the prefetch does not help). Builds without NEON compile to the same machine code as before. Per layer,
calibration and parity: `docs/ocr-speed.md`. `make check-arm` runs the C unit tests, with known-answer logits for
every NEON kernel, as Cortex-A9 NEON and scalar builds under qemu-arm.

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
- `VOCR_INVERT`, or'ed into `vocr_image.format`, reads every source pixel as `255 − grey` before the resize. Training does exactly this for dark text (`synth/common.line_crops`), so set it whenever the line finder reports `VocrLine.dark = 1` (the polarity contract in `docs/linefind.md`). It needs no copy of the crop. Inverting after the resize would not be the same: on 500 random crops, `cv2.resize(255 − g)` and `255 − cv2.resize(g)` differ on 25% of the pixels, by 1.

From Python (host tests, evaluation), `pyvocr.py` wraps `build/libvocr.so` with ctypes:

```python
from pyvocr import Vocr
r = Vocr("v1_w8.vocr", mode="w8i")                  # or Vocr.from_checkpoint("ckpt_best.pt")
text = r.text(crop)                                 # HxW uint8 grey or HxWx4 uint8 RGBA
text = r.text(crop, invert=True)                    # dark text (VOCR_INVERT)
text, logits = r.run(crop, logits=True)
```

`pyvocr` rebuilds `build/libvocr.so` (`vocr.c` and `vocr_lm.c`) whenever one of `vocr.c`, `vocr.h`, `vocr_lm.c` or `vocr_lm.h` is newer than it, and refuses crops that aren't `uint8`.

From eval's harness (PR #2), use its `cmd:` protocol. The CLI reads PGM or PPM paths on stdin and prints one line each:

```sh
python eval/run.py --model 'cmd:runtime/build/vocr_cli v1_w8.vocr --mode w8i --stdin' --set vita-test
```

`vocr_cli --invert` sets `VOCR_INVERT` for every crop it reads, and an unknown `--mode` is an error.

### Top-K, confidence and the beam + LM decoder (`vocr_lm.h`)

`vocr_run()` decodes greedily. Two optional additions read the same network output differently; neither changes `vocr_arena_bytes()`, the logits or the greedy text (tested bit for bit). A zeroed `vocr_extra` keeps the top-K off, and the decoder runs only when called (a zeroed `vocr_lm_opts` is invalid: `beam` and `topk` must be at least 1). The measurements behind the recommended settings are in [docs/ctc-lm-decode.md](../docs/ctc-lm-decode.md).

**Per-frame top-K and a confidence, greedy path.** Set `vocr_extra.topk = K` (1..`VOCR_TOPK_MAX` = 16) and give it caller buffers of `cap_frames` frames: `topk_logit` (K floats per frame, decreasing; ties keep the lower class first, so `[f * K]` is the argmax), `topk_class` (K `uint16_t`) and optionally `blank_logit` (one float per frame). `cap_frames` must hold the line's frames (`vocr_resized_width(m, w, h) / m->stride + 1` always does), else `VOCR_E_ARG`. The run also writes `conf_min` and `conf_mean`: over the frames that emit a character of the greedy text, the smallest and the mean softmax-max probability, computed over that frame's top K (an upper bound of the full softmax's). For a 1,024-px line (128 frames) at K = 8 the buffers are 128 × 8 × 6 + 128 × 4 = 6,656 B.

**CTC prefix beam search with a character n-gram LM.** `runtime/vocr_lm.c` (C99, no malloc, no libm, integer scores in 1/1024 nat) decodes a line from those top-K extras with an LM blob written by `lm/build.py` (a `VOLM` file; the tiers are release assets, `docs/ctc-lm-decode.md`):

```c
#include "vocr_lm.h"

static vocr_lm lm;             /* 188 B on 32-bit ARM (368 on x86-64); points into the LM blob */
static vocr_lm_ctx dec;        /* 120 B (136) */
vocr_lm_opts lo = {0};
lo.beam = 8; lo.topk = 8;                         /* the recommended settings for each LM tier: */
lo.alpha = 500; lo.beta = 3500; lo.gate = 8000;   /* docs/ctc-lm-decode.md (1/1000 nat; alpha 1/1000) */
lo.flags = VOCR_LM_EOS; lo.bracket = 8000;
rc = vocr_lm_load(&lm, lm_blob, lm_bytes);        /* 16-byte aligned, used in place, any memory type */
size_t ws_need = vocr_lm_workspace_bytes(&model, &lm, &lo);   /* exact; ~35-41 KB at these settings */
rc = vocr_lm_init(&dec, &model, &lm, &lo, ws, ws_need);       /* ws 16-byte aligned; maps classes once */

vocr_extra ex = {0};
ex.cap_frames = cap; ex.topk = 8;
ex.topk_logit = tv; ex.topk_class = tc; ex.blank_logit = bl;  /* caller buffers, cap frames each */
ex.cancel = cancel_cb; ex.user = job;                         /* polled by both calls */
rc = vocr_run(&ctx, &img, greedy, sizeof greedy, &ex);         /* greedy text, top-K, conf_min/mean */
rc = vocr_lm_decode(&dec, &ex, text, sizeof text, &chars);     /* the beam + LM text */
```

- The search keeps `beam` prefixes per frame (Viterbi: max over alignments). A prefix scores its frames' logit margins to each frame's top-1, plus `alpha` × the LM's log-probability of each character, plus `beta` per character, plus (with `VOCR_LM_EOS`) `alpha` × log P(end of line). A frame's candidates are its top-1 and the other top-K classes and the blank within `gate` nat of it: a frame whose top-1/top-2 margin is at least `gate` keeps its top-1, the guard for confidently read names and rare words. A closing bracket that does not match the line's innermost open bracket costs `bracket`; the LM reads every bracket pair as 「」, so the picture picks the style.
- With `alpha = beta = bracket = 0` the output is `vocr_run()`'s greedy text bit for bit, for any beam, top-K and gate (every non-top-1 candidate costs at least 1/1024 nat). `lm` may then be NULL.
- `vocr_lm_decode()` needs `ex->frames <= max_frames` (default: the model's 1,024 px / 8 = 128 frames; at most 1,024) and `<= ex->cap_frames`, else `VOCR_E_WIDTH`. It polls `ex->cancel` every 8 frames (`VOCR_E_CANCELLED`, empty text) and writes whole characters only on `VOCR_E_TEXT`.
- One decoder context per thread; the LM blob and the model can be shared. `vocr_lm_check_crc()` checks the blob's CRC-32; `vocr_lm_load()` checks what keeps every read inside the blob (sections, child ranges, token ids), not the order of the tokens within a range: a damaged blob can score wrongly but never makes the decoder read outside it, and the CRC catches the damage.
- Code: `vocr_lm.o` is 4,997 B with `vita/CMakeLists.txt`'s flags (VitaSDK gcc 15.2, Thumb-2; `size.txt` of `vita/scripts/build.sh`) and 7,897 B with `psp/Makefile`'s (psp-gcc 15.2.0, `-O2 -G0`); it needs only `memcpy`, `memset`, `vocr_class_text` (and libgcc's `__aeabi_uidiv` on ARM, once per init). The top-K branch grows `vocr.o` from 14,583 to 15,555 B (NEON) and from 12,455 to 13,403 B (scalar). The decoder has run on the host (gcc, clang, ASan/UBSan) and under qemu-arm for the cycle estimate; not on a console or in PPSSPP.
- `lm/vlm.py` is the specification: its reference decoder and scorer give the C code's output bit for bit (`tests/test_ctclm.py`).

From Python: `Vocr.run_topk(crop, k)` returns the greedy text, `topv`, `topc`, `blank`, `conf_min` and `conf_mean`; `VocrLM(vocr, lm_path, beam=8, topk=8, alpha=..., beta=..., gate=..., flags=..., bracket=...)` wraps the decoder (`.decode(topv, topc, blank)`). `pyvocr` loads `$VOCR_LIB` as is when it is set (a frozen build for long runs). The CLI takes `--topk K`, `--conf` (conf_min and conf_mean per line on stderr) and `--lm LM.vlm --beam N --alpha A --beta B --gate G --lm-flags F --bracket C` (prints the beam text; `--lm none` runs the beam without an LM).

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
make -C runtime check            # line finder checks, then C unit tests (vocr, vocr_lm) + PyTorch parity on a
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

## Text detector (`vocr_textdet.c`)

The learned text detector of issue #96 (`docs/textdet.md`): an int8 network on the grey region whose four maps
decode into line boxes, a vertical flag per box and, by an empty set, "no text". `train/textdet_export.py` writes
its blob (`VTDT`); `synth/textdet_twin.py` is the specification, and this runtime computes exactly what it computes
for every region and every blob its loader accepts.

```c
#include "vocr_textdet.h"

static vocr_textdet det;                         /* about 1 KB on 32-bit ARM; points into the blob */
int rc = vocr_textdet_load(&det, blob, bytes);   /* any alignment; VOCR_TEXTDET_E_BLOB if refused */
size_t need = vocr_textdet_arena_bytes(&det, 960, 544);   /* exact; depends on the width only */
VocrTextdetSet set;
rc = vocr_textdet_find(&det, &im, arena, need, &set);          /* or vocr_textdet_find_rows(&det, &src, ...) */
for (unsigned i = 0; i < set.count; i++) { /* set.lines[i] (x, y, w, h, dark), set.vertical[i] */ }

/* as a finder mode: horizontal lines only, every other finder value is vocr_find_lines unchanged */
need = vocr_lines_det_arena_bytes(&det, 960, 544);            /* max of the finder's and the detector's */
rc = vocr_find_lines_det(&im, VOCR_LINES_TEXTDET, &det, arena, need, &lines);    /* or _rows */
```

- Images are `VOCR_LINES_GREY` or `VOCR_LINES_RGBA` (grey = (77 R + 150 G + 29 B) >> 8), at most
  `VOCR_TEXTDET_MAX_SIDE` = 2048 px a side, which keeps every sum of the decode in int32 (the proof heads
  `vocr_textdet.c`). Return codes: `VOCR_LINES_E_ARG` (NULL pointers, an unloaded model, a bad format or stride, a
  zero or oversized side), `VOCR_LINES_E_ARENA`, `VOCR_LINES_E_SOURCE` (a NULL row), `VOCR_TEXTDET_E_BLOB` (load);
  the set is empty on any error. At most 16 lines (horizontal ones in rows, then vertical ones right to left) from at
  most 64 candidates; `truncated` when either cap dropped one. Each line's dark flag is `vocr_line_dark` of its box.
- The region is read once, top to bottom: every 3x3 layer keeps a ring of the 2d + 1 input rows its kernel spans
  (3 for the stride-2 layers), channel-planar with zero padding and slack; 1x1 layers run on the 8-pixel blocks
  their producer just made; the head's map row feeds a streaming 4-connected component decode (union-find over
  slots, at most the map's width of them live). So the arena depends on the region's width, not its height, and a
  strip height changes no output.
- Row-callback form (`vocr_lines_rows.h`): rows are asked once each, in increasing order, in one pass over the
  region, then each line's box once for its dark flag; a returned row has to stay valid only until the next call;
  NULL ends the call with `VOCR_LINES_E_SOURCE` and no lines. Same lines as the in-memory form.
- C99, no allocation, no recursion, no global state, integer arithmetic only (int32 sums; int64 for the
  requantisation product). NEON kernels behind `__ARM_NEON` (Vita), the same bits as the scalar ones (PSP, hosts;
  `-DVOCR_NO_NEON` forces them). External symbols: `memset`, `memcpy`, libgcc's `__aeabi_idivmod`, `__aeabi_uidiv`
  and `__aeabi_uidivmod` on ARM, and `vocr_find_lines`, `vocr_find_lines_rows`, `vocr_line_dark`,
  `vocr_line_dark_rows`, `vocr_lines_arena_bytes`: link `vocr_lines.c` and `vocr_lines_rows.c`, which stay
  byte-for-byte as pinned (`tests/test_textdet.py` checks their sha256 against `eval/textdet_protocol.json`).
- Code: `vocr_textdet.o` is 8,912 B with VitaSDK gcc 15.2, Thumb-2, `-O2` (7,736 B at `-Os`; scalar 8,420 / 6,584 B)
  and 13,068 B with psp-gcc 15.2 `-O2 -G0` (10,816 B at `-Os`). Stack (gcc `-fcallgraph-info=su`, every frame
  static): 2,848 B on the Vita build at peak, 2,808 B on the PSP, of which `vocr_line_dark`'s two histograms are
  about 2 KB; the network and decode alone 904 / 840 B. With another finder, `vocr_find_lines_det` adds its own
  frame, 72 / 48 B, to GRAD's 3,640 / 3,360 B.

Arena (`vocr_textdet_arena_bytes`, exact) and Cortex-A9 cost per region, for random blobs of every
`train/textdet_model.SHAPES` shape. The times are **estimates**: the Vita compiler's code under plugin qemu-arm with
the Vita's caches, priced by `vita/a9_cycles.py`, ms = its high bound x 1.0007 (ocr-speed's console calibration of
VOCR3-H15, `docs/ocr-speed.md`) at 333 / 444 MHz; `eval/textdet_a9.py`, `eval/textdet_a9_20261018.json` (instruction
counts, low/high cycles, hashes). Nothing of the detector has run on a console.

| shape | blob | arena 960x544 / 480x272 | MACs 900x150 | 900x150 @333 / @444 | 450x80 @333 / @444 | 960x544 @333 | 480x272 @333 |
|---|---:|---:|---:|---:|---:|---:|---:|
| s4_c4 | 1,004 | 39,311 / 21,071 | 3.1 M | 29 / 22 ms | 8 / 6 ms | 108 ms | 27 ms |
| s4_c6 | 1,280 | 48,215 / 25,895 | 4.2 M | 37 / 28 ms | 10 / 7 ms | 135 ms | 33 ms |
| s4_c8 | 1,540 | 57,111 / 30,711 | 5.6 M | 41 / 31 ms | 11 / 8 ms | 150 ms | 36 ms |
| s4_c8n5 | 2,228 | 73,799 / 39,719 | 7.9 M | 55 / 42 ms | 15 / 11 ms | 204 ms | 50 ms |
| s4_c12 | 2,276 | 77,967 / 41,967 | 9.8 M | 63 / 47 ms | 17 / 13 ms | 231 ms | 56 ms |
| s4_c16 | 3,108 | 98,815 / 53,215 | 14.9 M | 76 / 57 ms | 21 / 16 ms | 283 ms | 70 ms |
| s4_c16n4 | 3,860 | 111,375 / 60,015 | 18.4 M | 92 / 69 ms | 25 / 19 ms | 338 ms | 85 ms |
| s4_c24 | 4,412 | 103,103 / 55,583 | 23.0 M | 99 / 75 ms | 27 / 21 ms | 366 ms | 92 ms |
| s4p_c4 | 1,100 | 38,959 / 20,719 | 6.3 M | 33 / 25 ms | 9 / 7 ms | 120 ms | 30 ms |
| s4p_c8 | 2,700 | 56,567 / 30,167 | 18.7 M | 70 / 53 ms | 19 / 14 ms | 261 ms | 64 ms |
| s8_c8 | 1,884 | 41,527 / 23,295 | 3.0 M | 25 / 19 ms | 7 / 5 ms | 89 ms | 23 ms |
| s8_c16 | 3,276 | 60,807 / 34,415 | 4.9 M | 32 / 24 ms | 9 / 7 ms | 115 ms | 30 ms |
| s8_c24 | 5,700 | 92,639 / 52,327 | 10.6 M | 53 / 40 ms | 15 / 11 ms | 191 ms | 50 ms |
| s8_c32 | 7,924 | 111,919 / 63,447 | 14.3 M | 63 / 47 ms | 18 / 13 ms | 227 ms | 60 ms |
| s8_c32n4 | 9,876 | 116,063 / 65,671 | 17.1 M | 72 / 54 ms | 21 / 16 ms | 260 ms | 69 ms |
| s8_c48 | 14,188 | 129,855 / 73,703 | 25.7 M | 92 / 69 ms | 26 / 20 ms | 333 ms | 88 ms |
| s4_c8_ko | 1,540 | 57,111 / 30,711 | 5.6 M | 41 / 31 ms | 11 / 8 ms | 150 ms | 36 ms |
| s8_c16_ko | 3,276 | 60,807 / 34,415 | 4.9 M | 32 / 24 ms | 9 / 7 ms | 115 ms | 30 ms |

Every shape fits the always-safe arena tier (131,072 B) at 960x544. Checks: `make check-textdet` (strict compile,
then `tests/textdet_parity.py` through the sanitized driver `vocr_textdet_test.c`: random blobs of every shape,
probe blobs that drive the decode into its corners, wild requantisation constants, edge sizes and whole screens,
GREY and RGBA, both forms, against the twin's maps and lines; plus the driver's `--selftest`), `tests/test_textdet.py`
(the twin on hand-made maps, the loader, the C parity). The Cortex-A9 builds run the same parity under qemu-arm:

```sh
G=runtime/scripts/tools/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-linux-gnueabihf/bin/arm-none-linux-gnueabihf-gcc
$G -std=c99 -O2 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -static runtime/vocr_lines.c runtime/vocr_lines_rows.c \
   runtime/vocr_lines_refine.c runtime/vocr_textdet.c runtime/vocr_textdet_test.c -o vtd_neon   # -DVOCR_NO_NEON: scalar
printf '#!/bin/sh\nexec runtime/scripts/tools/bin/qemu-arm -cpu cortex-a9 ./vtd_neon "$@"\n' > vtd_qemu && chmod +x vtd_qemu
python tests/textdet_parity.py --bin ./vtd_qemu --seeds 1 --selftest 3
```

det+refine (the protocol's round-2 finder) is the finder mode's lines, then `vocr_refine_lines(_rows)` with
`VOCR_REFINE_DEFAULT` in the same arena of max(`vocr_textdet_arena_bytes`, `vocr_refine_arena_bytes`) bytes;
`vocr_textdet_test --batch` runs it for record flag 2 and `--arena-refine` prints that arena, and the parity checks it
against `synth/lines_refine.refine_lines` on the twin's horizontal lines.

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
