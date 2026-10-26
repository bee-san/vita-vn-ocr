# runtime: the recognizer in plain C

`vocr.h` and `vocr.c` run `train/model.py` checkpoints on the console. `train/export.py` turns a checkpoint into a weights blob. The dialogue line finder that feeds it lives next to it (`vocr_lines.c`, `docs/linefind.md`); the two share nothing but naming. `make check` runs the checks of both. An optional refinement of the finder's boxes (`vocr_lines_refine.c`: `vocr_refine_lines` / `vocr_refine_lines_rows`, called after the finder with its arena; flags 0 leave the lines unchanged) is described with its measurements in `docs/finder-tighten.md`; `make check-lines` runs its self-test and its parity with `synth/lines_refine.py`. The learned text detector (`vocr_textdet.c`, issue #96, `docs/textdet.md`) is a finder mode behind a flag; it has its own section below, and `make check-textdet` runs its checks. So is the text-presence gate (`vocr_textgate.c`, issue #100, `docs/textgate.md`): `make check-textgate`. The stroke split (`vocr_lines_split.c`, issue #98, `docs/translucent-box.md`) is a second optional step between the finder and the refinement, for text over translucent boxes and busy scenes; flags 0 leave the lines unchanged, it is not a default (its pre-registered rule did not adopt it), and `make check-split` runs its checks.

- C99. The runtime uses no `malloc` and nothing from libc except `memcpy`, `memset` and `<math.h>` (on 32-bit ARM, also libgcc's integer-division helpers; see the porting notes).
- The blob is read in place. All working memory comes from one arena that the caller provides, sized exactly by `vocr_arena_bytes()`.
- The line is processed in tiles of a few CTC frames (8 input columns each; 12 or 16 for the coarser-frame models of `train2.py --stride`, which `docs/stride16.md` trains). Every layer keeps its left context between tiles, so the arena stays the same size for any line width, and the output doesn't depend on the tile size (checked bit for bit).
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
vocr_opts opts = {0};                       /* defaults: tile 4 frames (about 32 input columns
                                               at stride 12 or 16: 3 or 2), head 16, max width 1024,
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

### Skipping the output head on blank frames (`vocr_opts.head_gate`)

Issue #117 ([docs/headgate.md](../docs/headgate.md)): about three CTC frames in five are blank, and the output head
(96 → 4,987 or 5,631 classes) is 15-40% of a frame's time. A gated blob (`VOCR5_{S,N,L}_w8g.vocr`) carries a tiny
network after the unchanged version-1 blob, a `VOCG` section of 1,792 B (`train/headgate.py` writes it): a 96 → 16
(ReLU) → 1 layer on the head's input, its first layer int8 through the head's own integer kernel (`mm_i16`), and a
threshold. With the gate on, a frame scoring above the threshold skips the head and reads as the blank.

```c
opts.mode = VOCR_MODE_W8I;
opts.head_gate = vocr_has_gate(&model) ? VOCR_HEAD_GATE_ON : VOCR_HEAD_GATE_OFF;   /* ON needs W8I and a gate */
size_t need = vocr_arena_bytes(&model, &opts);      /* + 48 B for VOCR5 at the default tile and batch */
rc = vocr_init(&ctx, &model, &opts, arena, need);
rc = vocr_run(&ctx, &img, text, sizeof text, &ex);  /* ex.gated: frames whose head was skipped */
```

- A skipped frame's argmax is 0, its `blank_logit` is the head's own class-0 logit (its first 8-class block on the
  same int16 input, bit for bit), and its top-K is `[blank logit, -inf, ...]` with classes 0. The other frames' logits,
  top-K and confidence are exactly the ungated run's; the text is the CTC collapse of the result. The head's batches
  hold only the frames it runs on, so they stay full, and the output is the same for every tile and batch size.
- `VOCR_HEAD_GATE_OFF` (0, a zeroed `vocr_opts`) is today's run: the same output, arena and code path (on a gated
  blob too: the section is only parsed). `VOCR_HEAD_GATE_SCORE` computes the gate (`vocr_extra.gate_score`) and skips
  nothing. A run that asks for the logits, or sets `vocr_extra.ungated`, skips nothing. A caller of
  `vocr_post_redecode` should read a line again with `ungated = 1` when the re-decode finds a counter context in its
  gated top-K (`vocr_post_stats.contexts`): the re-decode looks at blank frames' top-K, which a skipped frame does not
  offer (docs/headgate.md, "Counters").
- `vocr_extra.head_in` (frames × K floats) returns the head's input per frame, for fitting a gate.
- A blob without the section loads as before; an older runtime reads a gated blob as the plain version-1 blob (it
  reads only the header's byte count). `vocr_load` rejects a damaged section's structure, `vocr_check_crc` its
  content; `export.py --info` and `headgate.py info` print it.
- Cost (VitaSDK gcc 15.2, Cortex-A9, estimates from qemu instruction counts calibrated to VOCR3-H15's console rate,
  docs/headgate.md): the gate is about 1,600 instructions a frame; VOCR5-N's head goes from 270,807 to 120,138
  instructions a frame at the measured lines' 57% skip share. `vocr.o` grows from 18,219 to 20,159 B at `-O2` with NEON
  (scalar 13,403 to 15,159 B; psp-gcc `-O2 -G0` 19,795 to 22,959 B). No malloc; the arena grows by 2 B per accumulated
  frame (the frames' line indices), and the gate's scratch fits the head's.
- Twin and tests: `train/headgate.py` (`gate_scores`, `simulate`: a gated run from an ungated one) is bit-exact;
  `tests/test_headgate.py` checks it against the C library at several tilings, and `tests/test_vocr.c` (ASan/UBSan,
  Cortex-A9 NEON and scalar under qemu) checks the section's validation and the runs against its own reference.
  CLI: `vocr_cli --head-gate on|score|off`; Python: `Vocr(blob, mode="w8i", head_gate="on")`, `run_topk(...,
  gate_score=True, ungated=False)`.

### The two-level output head (`vocr_opts.two_level`)

Issue #117 item 2 ([docs/head2level.md](../docs/head2level.md)): VN dialogue uses few characters, so most frames need
only the head's most frequent rows. A two-level blob's head rows and vocab are in frequency order (an ordinary
version-1 blob: every class's logit is the same, its class number changes), and a `VOC2` trailer of 1,984 B
(`train/head2level.py write_trailer`, after head-gate's `VOCG` section when there is one) holds level 1's block count
and an escape: a 98 → 16 (ReLU) → 1 network on the head's quantised input (layer 1 int8 through `mm_i16`) plus level
1's largest non-blank logit and its blank logit. Level 2 is the head's own rows: a trailer with the trial's 48-wide
rare block (`rare_k` and its offsets non-zero; rejected, docs/head2level.md) is refused (`VOCR_E_FORMAT`).

```c
opts.mode = VOCR_MODE_W8I;
opts.head_gate = vocr_has_gate(&model) ? VOCR_HEAD_GATE_ON : VOCR_HEAD_GATE_OFF;
opts.two_level = (unsigned)vocr_has_two_level(&model);   /* needs W8I and a trailer, else VOCR_E_MODE */
size_t need = vocr_arena_bytes(&model, &opts);      /* + 1,456 B for VOCR5 at the default tile and batch */
rc = vocr_run(&ctx, &img, text, sizeof text, &ex);  /* ex.escaped: frames that also ran level 2 */
```

- Level 1 (the head's first `two.l1_blocks` 8-row blocks: the blank and the most frequent classes) runs on every
  frame the gate does not skip; the escape decides which of them also run level 2, the other rows, packed into one
  batch. A frame that fires gets today's logits, argmax and top-K bit for bit; one that does not gets them over the
  level-1 classes. A block whose 8 logits cannot change a frame's state is not scanned per class.
- `two_level = 0` (a zeroed `vocr_opts`) runs the full head on the permuted rows: today's output. A run that asks for
  the logits, or sets `vocr_extra.ungated`, runs the full head too (and the gate skips nothing). A caller should read
  a line again with `ungated = 1` when its reading holds a full-size カ or ケ before 月 所 年 国 条, where the escape may
  have missed the small ヵ / ヶ, as it does for the gate's counter contexts (docs/head2level.md, "Counter-context set").
- `vocr_extra.esc_score` returns each frame's escape score; class texts never change, so `vocr_lm.c`, `vocr_post.c`
  and `vocr_charbox.c` read a permuted blob as they read today's, except where two classes tie exactly (the first
  maximum, the top-K and the decoders' ties follow class numbers).
- A runtime without two-level support reads the blob as its version-1 part (the permuted full head) and ignores the
  trailer. `vocr_load` rejects a damaged trailer's structure, `vocr_check_crc` its content; `head2level.py info`
  prints it. The arena mirror is `train/export.plan_arena(..., two_level=read_trailer(blob))`.
- Cost (VitaSDK gcc 15.2 `-O2`; times are estimates from qemu instruction counts calibrated to VOCR3-H15's console
  rate): `vocr.o` grows from 20,159 to 23,259 B with NEON (scalar, `-DVOCR_NO_NEON`: 15,159 to 18,283 B; psp-gcc
  `-O2 -G0` 22,959 to 27,979 B); no malloc.
  VOCR5-N with the gate: est. 2.74 ms per frame against the gate's 2.88 (24 development lines, 333 MHz, no prefetch
  credit).
- Twin and tests: `train/head2level.py twolevel_exact` is bit-exact; `tests/test_head2level.py` checks it against the
  C library (with the gate, several tilings; a rare-block trailer refused); `tests/test_vocr.c` (ASan/UBSan, Cortex-A9 NEON and scalar
  under qemu) checks the trailer's validation and the runs against its own reference. CLI: `vocr_cli --two-level`;
  Python: `Vocr(blob, mode="w8i", two_level=1)`, `run_topk(..., esc=True)`.

### The attention op (VOCR6 stage 3)

[docs/vocr6-attn.md](../docs/vocr6-attn.md) (#125): a blob from a `train2.py --attn 1` model carries one light global
self-attention layer (op type 5) between the bottleneck and the head: z' = z + O(concat_h softmax(Q_h K_h^T) V_h)
over the line's frames. Nothing changes for the caller: `vocr_load` parses it (`vocr_model.attn.dim != 0`),
`vocr_arena_bytes` plans for it and `vocr_run` runs it.

- The line is streamed in tiles up to the bottleneck, as before, but every frame's bottleneck output is kept (the
  arena grows by `max_width / stride × 96 × 4` B: +49,152 B at 1,024 px); then the attention runs over the whole line,
  its projections in the blob's mode (f32, W8F, W8I) and its scores, softmax (the runtime's own `exp_neg`) and weighted
  sums in float32; then the head streams over z' with the gate and the two-level head as usual (their input is z').
- K, V and a head batch's queries overlay the rings and the head's buffers, idle at that point. The output is the
  same for every tile and head batch size, scalar and NEON. The cancel callback is also polled per chunk of the
  attention and per tile of the head after it.
- A blob without the op runs exactly as before; a runtime from before refuses a blob with it (`VOCR_E_FORMAT`).
- Cost (estimates, VitaSDK gcc `-O2`, NEON, docs/vocr6-attn.md): about 0.066 ms a frame for the projections plus
  0.0010 ms × T for the scores on a line of T frames at 333 MHz (+35 ms per nominal 3-line box for VOCR5-N's shape);
  `vocr.o` grows from 23,267 to 25,971 B (scalar 18,295 to 21,075 B); no malloc.
- Twin and tests: `runtime/tests/vocr_ref.py attn_exact` is bit-exact; `parity.py --fixture --attn 1` (run by
  `make check` through the sanitized CLI, and by `make parity-arm`), `tests/test_vocr6_attn.py` and
  `tests/test_vocr.c` check it.

### Counters and cut lines (`vocr_post.h`, `vocr_lines_widen.h`)

Three optional steps for counter contexts and for lines a tight region cuts (issue #109; [docs/counters.md](../docs/counters.md) has the corpus counts and the development measurements). None runs unless its caller calls it, so a pipeline without them is bit for bit as before.

- `vocr_post_counters(text)`: in place, the text never grows. ッ becomes ヶ before 月 and 年, before 国 and 条 after a numeral or 何数幾, and before 所 after one of those other than 一. ッ月 and ッ年 occur nowhere in the 700 M characters of training-side text counted. Returns the changes.
- `vocr_post_redecode(model, extra, opts, stats)`: on a `vocr_run()`'s top-K extras. At X + ッ/ヶ/ヵ + 月/所/年/国/条, where X is a letter and not a numeral, the cheapest numeral within `opts.tau` nats replaces X; when the line lacks its opening 「 or 『 before X, that bracket is inserted within `opts.open`. A change is written into the extras as a class swap with the frame's top-1, so `vocr_post_greedy(model, extra, text, cap, chars)` gives the edited greedy text and `vocr_lm_decode()` reads it. The development choice is `{6000, 4000}` (1/1000 nat): 0 real development lines changed, and the synthetic counter set got 0 lines worse. Keep `tau` and `open` below the LM's gate and the LM's `topk` at least the extras' (`vocr_post.h`).
- `vocr_line_widen(strip, rx, rw, &x, &w)`: continues a found line's box past the region's edge through the ink the edge cuts, into a margin of the frame around the region, when that ink ends inside the margin. Not adopted as a default (docs/counters.md): a large gain when regions are drawn into the text, a small, unresolved loss on regions with a margin.

Code (VitaSDK gcc 15.2, Thumb-2; -O2 as `vita/CMakeLists.txt` builds it / -Os with `-ffunction-sections -fdata-sections` as Rubi's runtime library builds it): `vocr_post.o` 3,964 / 2,598 B, with an 896 / 832 B stack frame for the re-decode; `vocr_lines_widen.o` 756 / 560 B. No arena, no weights. Twins: `eval/vocr_post.py` and `synth/lines_widen.py`, checked by `make check-post` and `check-widen`. CLI: `--post`, `--redecode TAU[,OPEN]`. Python: `pyvocr.post_counters(text)`, `Vocr.redecode(out, tau, open)`.

### Per-character positions (`vocr_charbox.h`)

Issue #114 ([docs/char-boxes.md](../docs/char-boxes.md)): a box on the line for every character `vocr_run()` emits,
for an overlay drawn on the game's own text. It reads only what a run already returns (the per-frame classes) and,
optionally, the crop; it changes nothing in `vocr.c`, so a caller that never calls it is bit for bit as before.

```c
#include "vocr_charbox.h"

static const vocr_charbox_cal cal_n = {0, {0, -45, -32, 48, -23, -4, 0, -6}};   /* eval/charboxes_cal.json, N */
uint16_t best[129];                          /* vocr_extra.best (or topk_class with stride vocr_extra.topk) */
uint32_t prof[1024];                         /* the crop's column profile, w entries */
vocr_charbox box[VOCR_CHARBOX_MAX];
unsigned n;
ex.best = best; ex.cap_frames = 129;
rc = vocr_run(&ctx, &img, text, sizeof text, &ex);
rc = vocr_charbox_profile(&img, prof);       /* once per line: sum over rows of |g(y + 1) - g(y)| */
rc = vocr_charbox_find(&model, best, 1, ex.frames, img.width, img.height, VOCR_CHARBOX_INK, &cal_n,
                       NULL, prof, box, VOCR_CHARBOX_MAX, &n);
/* box[i].c, .x0/.x1 (cell), .i0/.i1 (ink): crop px x 16 along the line; screen x = crop x + v / 16.0,
   y = the crop's rows. A column read turned 90 degrees anticlockwise: positions down the column. */
```

- A character's centre is its run's first frame, `8 f0 + 4` resized px, plus the model's calibration (a shift and
  an offset per character group, `eval/charboxes_cal.json`), times `w / nw`. The cells tile the line (cut between
  neighbouring centres; the ends half a pitch out), so a point on the line picks one character.
  `VOCR_CHARBOX_INK` moves each cut to the emptiest columns between two centres and shrinks each ink box to its
  cell's ink; `VOCR_CHARBOX_PITCH` (no gain measured) snaps a regular line to a grid; `VOCR_CHARBOX_RUN` centres
  on the whole run.
- For a touch, hit-test the cells of a find **without** `VOCR_CHARBOX_INK` (the calibration only; no crop or
  profile needed): inside 「……」 INK's cuts fall in the gaps between the dots, and its cells take 0.75% of taps to a
  wrong cell or none, against 0.14% (another character: 0.41% against none; development, report only:
  `docs/char-boxes.md`, "Hit-test"). INK's centres and ink boxes are for drawing on the glyph.
- `vocr_charbox_align(src, n, dst, m, map)`: the characters of another text (the beam + LM text, the normalised
  text, a native hook's line) mapped onto a line's characters by their code points: `map[j]` is a character of
  `src` or -1 (an inserted character, with no box of its own). Bounded: at most `VOCR_CHARBOX_SYNC` (8) characters
  are skipped on either side to resynchronise.
- C99, no allocation, no recursion, no global state, integer arithmetic (int64 where a product needs it). The only
  memory is the caller's: the output (28 B a character) and, optionally, the profile (4 B a column;
  without it the profile is recomputed column by column, about 3.4x the instructions). External symbols:
  `vocr_class_text`, `vocr_resized_width`, `memset` and libgcc's `__aeabi_ldivmod` on ARM.
- Code: `vocr_charbox.o` 5,032 B with VitaSDK gcc 15.2, Thumb-2, `-O2`, 3,186 B at `-Os`; psp-gcc 15.2
  `-G0`: 8,038 / 5,722 B. Stack: at most 104 B a frame (`vocr_charbox_align`; `-O2`).
- Cost, an **estimate**: qemu-arm instruction counts of the `-O2` Thumb-2 code for a 900 × 36 grey line: 0.52 M
  instructions for the profile and INK (almost all of it the profile, about 16 a pixel), 10 K without INK; at one
  instruction a cycle, 1.6 ms at 333 MHz, against roughly 0.3-0.5 s to recognize such a line. Nothing of it has run
  on a console.
- Checks: `make check-charbox` (in `make check`, so in `ci/runtime.sh gcc|clang` with ASan + UBSan): strict
  compiles, the driver's self-test, then `tests/charbox_parity.py`: random lines (greedy runs with blanks and
  repeats over `data/vocab.txt`, every flag, calibrations to the int16 extremes, caps, profiles with ties, zeros
  and spikes, GREY8 and RGBA8 crops) and random text pairs against the twin `eval/charbox.py`, every number equal.
  Also run: 4,000 + 4,000 cases on a gcc build and 1,500 + 1,500 on clang (sanitized), 400 real development lines per
  model, each with four flag sets (VOCR5-S/N/L/V's own `best` arrays and crops, `--real-units`), and the Cortex-A9 NEON, scalar and Thumb-2 builds under qemu-arm
  (Arm GNU 14.2, 4,500 + 3,000 cases): no difference.

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

It folds BatchNorm into the convolutions (in float64) and maps the model to five op types: one-input-channel conv (the stem), depthwise conv, pointwise conv (including the 2×1 height collapse as a pointwise over two rows), and the head. With `--int8`, every pointwise and head matrix becomes symmetric per-channel int8. A `train2.py --attn 1` model adds a sixth, the attention op before the head (its Q, K, V and O matrices int8 too; "The attention op" above).

The blob is version 1:
- a 64-byte header: magic `VOCR`, version, sizes, a CRC-32 and the checkpoint's SHA-256 prefix;
- one 64-byte record per op;
- 16-byte-aligned weight sections, with the matrices packed `[cout/8][K][8]`;
- the vocabulary as UTF-8 with 16-bit offsets.

The full layout is in the docstring of `train/export.py`. `vocr_load()` rejects anything it doesn't run: any other version, layer, residual or shape. A model change that adds an op therefore fails at load time and in CI's op-set test.

An optional head-gate section (`VOCG`, its own version and CRC-32, bound to the blob's byte count and CRC) may follow the blob at its 16-byte aligned end; `train/headgate.py` documents and writes it, and "Skipping the output head on blank frames" above says what it does. The version-1 bytes before it are unchanged, so the blob's own CRC and every older runtime read it as before.

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

## Full-frame text spotter (`vocr_textspot.c`)

The spotter of issue #104 (`docs/textspot.md`): every text box of a whole frame (Vita 960x544, PSP 480x272,
Adrenaline's 2x picture, a PC screenshot scaled to the Vita), each with an orientation, a confidence and a dark flag,
grouped into blocks in reading order, and one block picked as the likely dialogue. A `VTSP` blob is a 32-byte header
(`half_above`: a frame wider than this is halved by an exact 2x2 mean first; `conf_min`) around #97's `VTDT`
network, unchanged. `synth/textspot_twin.py` is the specification; this runtime gives `spot()`'s boxes, blocks,
order, pick and truncated flag for every frame and every blob its loader accepts.

```c
#include "vocr_textspot.h"

static vocr_textspot ts;                          /* about 1 KB on 32-bit ARM; points into the blob */
int rc = vocr_textspot_load(&ts, blob, bytes);    /* any alignment; VOCR_TEXTSPOT_E_BLOB if refused */
size_t need = vocr_textspot_arena_bytes(&ts, 960, 544);   /* exact for every frame up to 960x544 */
static VocrTextspotSet set;                       /* 1,552 B: up to 64 boxes and 64 blocks */
rc = vocr_textspot_find(&ts, &im, 0, arena, need, &set);  /* or vocr_textspot_find_rows(&ts, &src, 0, ...) */
for (unsigned k = 0; k < set.block_count; k++)    /* blocks in reading order; set.pick: the dialogue, -1: none */
    for (unsigned i = set.blocks[k].first; i < set.blocks[k].first + set.blocks[k].count; i++) {
        /* set.boxes[i]: x, y, w, h, vertical, conf (1..127), dark, block */
    }
```

- Frames are `VOCR_LINES_GREY` or `VOCR_LINES_RGBA` (grey = (77 R + 150 G + 29 B) >> 8), at most
  `VOCR_TEXTSPOT_MAX_SIDE` = 2048 px a side, which keeps every sum of the decode in int32 (the proof heads
  `vocr_textspot.c`). Return codes: `VOCR_LINES_E_ARG` (NULL pointers, an unloaded model, a bad format, stride or
  flag, a zero or oversized side), `VOCR_LINES_E_ARENA`, `VOCR_LINES_E_SOURCE` (a NULL row), `VOCR_TEXTSPOT_E_BLOB`
  (load); on any error the set is empty (`pick` -1). At most 64 boxes, the most confident first
  (`VOCR_TEXTSPOT_MAX_BOXES`; `truncated` when more were found).
- Flags: 0 is the twin exactly. `VOCR_TEXTSPOT_NO_DARK` leaves every dark flag 0 and reads no box: the row-callback
  form then reads the frame exactly once, top to bottom (Rubi's scan-screen job; the read job computes
  `vocr_line_dark` itself, the same bit).
- Arena: `vocr_textspot_arena_bytes(m, W, H)` is exact for every frame up to W x H, and the number the protocol's
  rule reads. It depends on the width only. For a halving blob (`half_above` h > 0) and W > h it is the larger of the
  needs of an h-wide frame (read at full size) and a W-wide one (halved). `vocr_textspot_frame_arena_bytes(m, W, H)`
  is one frame's own need, which is what each call checks.
- The frame is read once, top to bottom. Halving keeps one row of ceil(W/2) pair sums. Every 3x3 layer keeps a ring
  of its 2d + 1 input rows, and 1x1 layers run on the 8-pixel blocks their producer just made (both as
  `vocr_textdet.c`). The head's map row feeds the streaming 4-connected component decode (sums of K, O, A1, A2,
  rows and columns; at most the map's width of live slots). The 64 best candidates are kept in a list sorted by the
  twin's cap key. Blocks, reading order and the pick follow the twin's stable sorts over at most 64 boxes, in
  scratch laid over the network's dead part of the arena.
- Row-callback form (`vocr_lines_rows.h`): the frame's rows are asked once each, in increasing order, in one pass,
  then (without `VOCR_TEXTSPOT_NO_DARK`) each box's rows once, boxes in output order, for its dark flag
  (`vocr_line_dark_rows`). A returned row has to stay valid only until the next call. NULL ends the call with
  `VOCR_LINES_E_SOURCE` and no boxes. Same set as the in-memory form.
- C99, no allocation, no recursion, no global state, integer arithmetic only (int32 sums; int64 for the
  requantisation product). The network and the decode are `vocr_textdet.c`'s, copied, not linked: the spotter is
  self-contained, and `vocr_textdet.c` stays frozen. NEON kernels, halving and input conversion behind
  `__ARM_NEON` (Vita) give the same bits as the scalar ones (`-DVOCR_NO_NEON`: hosts and the PSP build; the scalar
  path is parity-checked on hosts and under qemu-arm, and psp-gcc compiles it, but no MIPS build has run). External
  symbols:
  `memset`, `memcpy`, libgcc's `__aeabi_idiv`, `__aeabi_idivmod`, `__aeabi_uidiv`, `__aeabi_uidivmod` on ARM, and
  `vocr_line_dark` / `vocr_line_dark_rows`: link `vocr_lines.c` and `vocr_lines_rows.c`, byte-for-byte as pinned
  (`tests/test_textspot.py` and `tests/test_textspot_runtime.py` check the protocol's sha256). No data or bss.
- Code (`.text` of the spotter's own sections kept by a `--gc-sections` link of a caller of load, both arenas, find
  and find_rows): 11,440 B with VitaSDK gcc 15.2, Thumb-2, `-O2`, NEON; 9,466 B at `-Os`; 10,396 / 8,058 B scalar
  (`-DVOCR_NO_NEON`) at `-O2` / `-Os`. psp-gcc 15.2 (mips32 Allegrex) `vocr_textspot.o`: 15,786 B at `-O2 -G0`,
  12,890 B at `-Os -G0`. `vocr_line_dark` and `vocr_line_dark_rows` add 2,160 B. On the Vita, NEON `-Os` saves 17% of
  the code for 1-5% more time, and scalar `-Os` (`-Os -DVOCR_NO_NEON`, the smallest build) saves 30% for 2.7-3.8x the
  time (the JSON's report tables), so no further size-optimised engine is offered. Stack (gcc
  `-fcallgraph-info=su`, every frame static): 2,880 B at peak on the Vita `-O2` build and 2,832 B on the PSP,
  of which `vocr_line_dark`'s two histograms are about 2 KB. The network, decode and blocks alone use 880 / 832 B.
  `vocr_textspot_maps` (a test hook) adds its own 1.5 KB set.

Cortex-A9 estimates per frame of random blobs, for every family B shape of `eval/textspot_protocol.json`
(`eval/textspot_a9.py`, `eval/textspot_a9_20261019.json`; all 21 shapes of #97's runs at both scales are in the JSON).
These are **estimates**, made as `eval/textdet_a9.py` makes them: the Vita compiler's `-O2` code runs under plugin
qemu-arm with the Vita's caches on one pinned procedural grey frame, priced by `vita/a9_cycles.py`. ms @333 is the
high bound x 1.0007 (ocr-speed's console calibration of VOCR3-H15, `docs/ocr-speed.md`), and ms @444 is that x
333/444. The time is the default call's, dark pass included. Random blobs fill the 64-box cap, so the dark pass is
near its largest. The arenas are exact, and the rule reads the up-to one. Nothing of the spotter has run on a
console.

| scale | shape | VTSP B | MACs / 960x544 frame | 960x544 @333 / @444 | no dark @333 | 480x272 @333 | arena up to 960x544 | arena of the 960x544 frame | arena 480x272 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| h2 | s4_c8 | 1,572 | 5.3 M | 46.7 / 35.0 ms | 41.3 ms | 37.9 ms | 38,311 | 32,535 | 31,575 |
| h2 | s4_c12n4 | 2,840 | 11.4 M | 82.4 / 61.8 ms | 74.2 ms | 71.2 ms | 58,159 | 48,903 | 47,943 |
| h2 | s4_c16n4 | 3,892 | 17.6 M | 97.6 / 73.2 ms | 91.5 ms | 88.5 ms | 73,855 | 61,839 | 60,879 |
| h2 | s4_c24n4 | 5,732 | 28.5 M | 130.7 / 98.0 ms | 122.5 ms | 121.2 ms | 80,799 | 67,583 | 66,623 |
| h2 | s8_c16 | 3,308 | 4.7 M | 44.8 / 33.6 ms | 34.6 ms | 32.6 ms | 41,711 | 35,999 | 35,039 |
| h2 | s8_c24 | 5,732 | 10.1 M | 60.9 / 45.7 ms | 55.0 ms | 53.0 ms | 63,103 | 53,911 | 52,951 |
| h2 | s8_c32 | 7,956 | 13.6 M | 73.6 / 55.2 ms | 65.3 ms | 63.3 ms | 76,263 | 65,031 | 64,071 |
| f1 | s8_c8 | 1,916 | 11.6 M | 88.8 / 66.6 ms | 86.0 ms | 23.6 ms | 42,391 | 42,391 | 23,919 |
| f1 | s8_c16 | 3,308 | 18.9 M | 115.8 / 86.9 ms | 112.5 ms | 32.6 ms | 61,671 | 61,671 | 35,039 |
| f1 | s8_c16n4 | 4,060 | 22.1 M | 131.2 / 98.4 ms | 127.0 ms | 35.6 ms | 72,791 | 72,791 | 41,359 |
| f1 | s4_c6 | 1,312 | 16.3 M | 136.4 / 102.3 ms | 133.6 ms | 35.8 ms | 49,559 | 49,559 | 26,759 |

Every family B shape is within the protocol's limits at its scale (at most 150 ms at 333 MHz and 131,072 B at
960x544). A 480x272 frame is never halved at h2 (480 <= 600), so both scales cost the same on it. An RGBA frame
(report only, in the JSON) adds 18-25 ms at 960x544, mostly the model's DRAM time for 4-byte pixels (about 51,000
more L2 misses; no prefetch credit). With RGBA, s4_c24n4 h2, s8_c16n4 f1 and s4_c6 f1 pass 150 ms (153, 150 and 157
ms).

Checks: `make check-textspot` (strict compiles, then `tests/textspot_parity.py` through the sanitized driver
`vocr_textspot_test.c`). The parity runs random networks of every family B and `SHAPES` shape, wild requantisation
and probe networks (designed frames of more than 64 candidates), each at `half_above` 0 and 600 and some with
`conf_min` > 1. It covers whole screens (960x544, 480x272, 725x544, 870x544), edge and odd sizes, GREY and RGBA,
both forms and `VOCR_TEXTSPOT_NO_DARK`, and compares maps, boxes, blocks, pick and truncated with the twin. It also
runs the driver's `--selftest`. `tests/test_textspot_runtime.py` covers the loader against the twin, the contracts
and the arenas. The Cortex-A9 builds run the same parity under qemu-arm (`--bin` takes a prefixed command):

```sh
G=runtime/scripts/tools/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-linux-gnueabihf/bin/arm-none-linux-gnueabihf-gcc
$G -std=c99 -O2 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -static runtime/vocr_lines.c runtime/vocr_lines_rows.c \
   runtime/vocr_textspot.c runtime/vocr_textspot_test.c -o vts_neon          # -DVOCR_NO_NEON: scalar
python tests/textspot_parity.py --bin "runtime/scripts/tools/bin/qemu-arm -cpu cortex-a9 ./vts_neon" --selftest 4
```

## Spotter for VTSP versions 1 and 2 (`vocr_textspot2.c`, d2s)

**Research only.** No console build (`vita/`, `psp/`) and no Rubi build links this file, and no release ships a
version-2 blob. textspot2 selected no model (`docs/textspot2.md`, amendment 6); the file stays so that its
recorded d2s results reproduce. The console spotter is `vocr_textspot.c`, unchanged.

textspot2's spotter (issue #106, `docs/textspot2.md` "d2s"): the full-frame spotter above for VTSP version 1 blobs,
and for version 2 blobs, whose network is a stride-8 trunk with a 2 x 2 sub-pixel head. `synth/textspot2_twin.py`
specifies the format (its docstring) and is the bit-exact twin of both versions; `train/textspot2_export.py` writes
version-2 blobs (from the trainer's quantisation-aware checkpoint, or random ones for tests).

- **Version 1** is `vocr_textspot.c` exactly: the same boxes, blocks, order, pick, truncated and dark flags and the
  same arena sizes (`vocr_textspot.c` is copied, not linked, and stays frozen; the parity runs both drivers on every
  version-1 batch).
- **Version 2** (the smallest change of version 1): the VTSP header says version 2, and its network is #97's VTDT
  layout with outputs 16 at stride 8 (min cells 2), which no version-1 loader accepts (they need outputs 4).
  Channel 4 (2 dy + dx) + m of a stride-8 cell is map m (K, O, A1, A2) of its sub-cell (dy, dx), so the decode reads
  the four maps at stride 4, cropped to ceil(h / 4) x ceil(w / 4) of the network's input; the decode is textspot's.
  A version-1 runtime (`vocr_textspot.c`, `textspot_twin`) refuses version 2.

```c
#include "vocr_textspot2.h"

static vocr_textspot2 ts;                          /* points into the blob; .version 1 or 2 */
int rc = vocr_textspot2_load(&ts, blob, bytes);    /* VOCR_TEXTSPOT_E_BLOB if refused */
size_t need = vocr_textspot2_arena_bytes(&ts, 960, 544);
static VocrTextspotSet set;                        /* vocr_textspot.h's result, flags and limits */
rc = vocr_textspot2_find(&ts, &im, 0, arena, need, &set);   /* or vocr_textspot2_find_rows(&ts, &src, 0, ...) */
```

- The calls, flags (`VOCR_TEXTSPOT_NO_DARK`), return codes, arena forms, row-callback contract and limits are
  `vocr_textspot.h`'s; only the names carry a 2. `vocr_textspot2_maps` gives the maps the decode reads (version 2:
  the stride-4 rows of the d2s head). C99, no allocation, no recursion, no global state, integer only; NEON and
  scalar give the same bits.
- Arena: for a version-2 network the plan holds its 16-plane head row (16 ceil(w / 8) bytes) and a 4-plane map row
  (4 ceil(w / 4) bytes), and the decode's component slots and run lists follow the stride-4 maps (ceil(w / 4)
  cells a row); a version-1 plan is `vocr_textspot.c`'s, byte for byte. Stack (gcc `-fcallgraph-info=su`, every
  frame static): find 2,904 B at peak on the Vita `-O2` build (2,832 B at `-Os`), 2,848 B on the PSP `-O2 -G0`
  (2,768 B at `-Os`); the network, decode and blocks alone 904 B (Vita `-O2`). `vocr_textspot.c`: 2,880 / 880 B.
- Code (`.text` of the spotter's own sections kept by a `--gc-sections` link of a caller of load, both arenas, find
  and find_rows; `eval/textspot2_a9.py codesize --runtime v2`): 11,844 B with VitaSDK gcc 15.2, Thumb-2, `-O2`,
  NEON (`vocr_textspot.c`: 11,440 B); 9,746 B at `-Os` (9,466 B); 10,804 / 8,338 B scalar at `-O2` / `-Os`. psp-gcc
  15.2 `vocr_textspot2.o`: 16,530 B at `-O2 -G0`, 13,482 B at `-Os -G0`.

Cortex-A9 estimates (`eval/textspot2_a9.py measure --runtime v2`, `eval/textspot2_a9_d2s_20261020.json`: textspot's
method on this file's Vita `-O2` code, random blobs, the pinned frames; ms @333 = high bound x 1.0007, ms @444 =
that x 333/444; the time is the default call's, dark pass included). These are **estimates**: nothing of it has run
on a console.

| shape (h2) | VTSP B | MACs / 960x544 frame | 960x544 @333 / @444 | no dark @333 | 480x272 @333 | arena up to 960x544 | arena of the 960x544 frame |
|---|---:|---:|---:|---:|---:|---:|---:|
| `s8_c24n4x2` (d2s) | 7,416 | 12.3 M | 73.2 / 54.9 ms | 65.8 ms | 63.6 ms | 79,647 | 67,631 |
| `s8_c48n4x2` (d2s) | 18,568 | 31.2 M | 122.5 / 91.9 ms | 115.8 ms | 113.3 ms | 125,983 | 106,767 |
| `s4_c24n4` (version 1) | 5,732 | 28.5 M | 130.7 / 98.0 ms | 122.6 ms | 121.2 ms | 80,799 | 67,583 |

`s4_c24n4` through this file costs what it costs through `vocr_textspot.c` (130.69 ms there, +0.02 ms here: the
extra code moves the layout, not the work), with the same boxes and arenas.

Checks: `make check-textspot2` (in `make check`, so in `ci/runtime.sh gcc|clang` with ASan + UBSan): strict
compiles (also `-DVOCR_NO_NEON`), then `tests/textspot2_parity.py` through the sanitized driver
`vocr_textspot2_test.c` and textspot's `vocr_textspot_test.c`: random version-1 networks of the protocol's "kroa"
shapes, random d2s networks of its "kroa_x2" shapes, wild-requantisation and probe networks of both versions (the
d2s probe must give exactly the stride-4 probe's result; designed frames of more than 64 candidates), at
`half_above` 0 and 600, GREY and RGBA, both forms, `VOCR_TEXTSPOT_NO_DARK`; maps, boxes, blocks, pick and truncated
against the twin, version 1 also against `vocr_textspot_test`'s lines and arenas, the loader against the twin, and
the driver's `--selftest`. `tests/test_textspot2_runtime.py` and `tests/test_textspot2_twin.py` cover the loaders,
the contracts, the arenas and the decode on hand-made maps. Under qemu-arm:

```sh
G=runtime/scripts/tools/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-linux-gnueabihf/bin/arm-none-linux-gnueabihf-gcc
$G -std=c99 -O2 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -static runtime/vocr_lines.c runtime/vocr_lines_rows.c \
   runtime/vocr_textspot2.c runtime/vocr_textspot2_test.c -o vts2_neon       # -DVOCR_NO_NEON: scalar
python tests/textspot2_parity.py --bin "runtime/scripts/tools/bin/qemu-arm -cpu cortex-a9 ./vts2_neon" --selftest 4
```

## Spotter for VTSP versions 1, 2 and 3 (`vocr_textspot3.c`, multi-scale)

**Research only, for now.** No console build (`vita/`, `psp/`) and no Rubi build links this file, and no release
ships a version-3 blob. It is textspot3's runtime (issue #108, `docs/textspot3.md`): family M's networks run on it,
and it would only reach a console if a family-M model is selected. The console spotter is `vocr_textspot.c`.

`vocr_textspot2.c` copied (not linked) with VTSP version 3 added. `synth/textspot3_twin.py` specifies the format (its
docstring) and is the bit-exact twin of all three versions; `train/textspot3_model.py` writes version-3 blobs (from
the trainer's checkpoint, or random ones for tests and the A9 rows).

- **Versions 1 and 2** run exactly as `vocr_textspot2.c`: the same boxes, blocks, order, pick, truncated and dark
  flags, and the same arena sizes (the parity runs both drivers on every version-1 and version-2 batch).
- **Version 3** (family M): a VTSP header with `half_above` and `half_above_h` (a frame wider than 480 or taller
  than 272 px is halved, so the network reads at most 480 x 272), then a VTMS network:
  - a stride-8 stream of version 1's layers, run row by row on version 1's kernels into a whole stride-8 map;
  - then generic map ops on whole maps: 3x3 depthwise (stride 1 or 2, dilation 1..8) and 1x1 over up to three
    earlier maps, each read through a nearest upsampling by 2^u (u <= 3), up to 1,024 channels;
  - the last op is version 2's 16-channel d2s head, whose rows feed textspot's decode at stride 4.

  Weights are int8 in -127..127 (the loader refuses -128), with version 1's int32 bias and fixed-point
  requantisation. Earlier runtimes (`vocr_textspot.c`, `vocr_textspot2.c` and their twins) refuse version 3.

```c
#include "vocr_textspot3.h"

static vocr_textspot3 ts;                          /* points into the blob; .version 1, 2 or 3 */
int rc = vocr_textspot3_load(&ts, blob, bytes);    /* VOCR_TEXTSPOT_E_BLOB if refused */
size_t need = vocr_textspot3_arena_bytes(&ts, 960, 544);   /* every frame up to 960 x 544 */
static VocrTextspotSet set;                        /* vocr_textspot.h's result, flags and limits */
rc = vocr_textspot3_find(&ts, &im, 0, arena, need, &set);   /* or vocr_textspot3_find_rows(&ts, &src, 0, ...) */
```

- The calls, flags (`VOCR_TEXTSPOT_NO_DARK`), return codes, row-callback contract and limits are `vocr_textspot.h`'s;
  only the names carry a 3. C99, no allocation, no recursion, no global state, integer only; NEON and scalar give
  the same bits.
- Arena, version 3: the stream's state over a whole stride-8 map while the frame is read, then the maps in slots on
  a stack (in-place 1x1 and depthwise ops where a source dies, an FPN neck freeing its stages last in, first out),
  each op's scratch on top. It depends on both sides of the frame (`vocr_textspot3_arena_bytes(m, w, h)` is the most
  any frame up to w x h needs).
- Speed (version 3's map part, NEON):
  - 1x1 ops: VMULL.S8 + VMLAL.S8 put two int8 products in each int16 lane, VPADAL.S16 folds them into int32, 4
    output channels x 2 cells per pass: 32 instructions per 128 multiply-adds;
  - depthwise rows: two passes, with the 9 taps' weights in registers;
  - no NEON register spills (the A9 pays for extension-register loads between NEON instructions).
- Stack (gcc `-fcallgraph-info=su`, every frame static; `eval/textspot3_a9r.py stack`):

  | build | find | find_rows | find without the dark pass | load | `_maps` (test hook) |
  |---|---:|---:|---:|---:|---:|
  | Vita `-O2` | 2,712 | 2,704 | 1,960 | 160 | 4,272 |
  | Vita `-Os` | 2,688 | 2,680 | 1,920 | 124 | 4,240 |
  | PSP `-O2 -G0` | 2,616 | 2,616 | 1,800 | 64 | 4,168 |
  | PSP `-Os -G0` | 2,624 | 2,624 | 1,768 | 72 | 4,176 |

  The peak runs through `vocr_line_dark`'s histograms, as in textspot; `vocr_textspot2.c`, counted the same way, is
  2,904 / 960 B (Vita `-O2`).
- Code (`.text` of the spotter's own sections kept by a `--gc-sections` link of a caller of load, both arenas, find
  and find_rows; `eval/textspot3_a9m.py codesize`):
  - VitaSDK gcc 15.2, Thumb-2, NEON: 23,880 B at `-O2` and 19,452 B at `-Os`;
  - scalar (`-DVOCR_NO_NEON`): 20,252 B at `-O2` and 16,054 B at `-Os`;
  - psp-gcc 15.2, `vocr_textspot3.o`: 31,618 B at `-O2 -G0` and 26,706 B at `-Os -G0`;
  - `vocr_textspot2.c` is 11,844 / 9,746 B on the Vita; version 3 adds the map ops, their NEON kernels and the plan.

Cortex-A9 estimates of the registered family-M shapes (`eval/textspot3_a9_20261021.json`):
- the method is textspot's, on this file's Vita `-O2` code: random version-3 blobs and the pinned frames;
- ms @333 = high bound x 1.0007; ms @444 = that x 333/444; the default call (dark pass included);
- these are **estimates**: nothing of it has run on a console.

| shape (point) | VTSP B | MACs / frame | 960x544 @333 / @444 | no dark @333 | 480x272 @333 / @444 | arena up to 960x544 | split @333: stream / map / decode / dark / cache |
|---|---:|---:|---:|---:|---:|---:|---|
| `qS4` (S) | 117,184 | 75.1 M | 195.1 / 146.3 ms | 187.0 ms | 184.8 / 138.6 ms | 256,247 | 122.4 / 38.5 / 2.7 / 7.3 / 23.4 |
| `qM4` (M) | 247,392 | 96.1 M | 216.6 / 162.4 ms | 211.3 ms | 208.4 / 156.3 ms | 256,247 | 122.4 / 59.4 / 3.0 / 4.8 / 26.3 |
| `qL5` (L) | 447,392 | 107.3 M | 231.8 / 173.8 ms | 225.3 ms | 222.9 / 167.2 ms | 256,247 | 122.4 / 70.8 / 2.9 / 5.7 / 29.2 |

All three keep family T's `t64n4c16` trunk, whose stream costs the same 122.4 ms in each, and whose state sets the
arena (5,897 B under the 262,144 B tier).

Checks: `make check-textspot3` (in `make check`, so in `ci/runtime.sh gcc|clang` with ASan + UBSan):
- strict compiles (also `-DVOCR_NO_NEON`);
- `tests/textspot3_parity.py` through the sanitized driver (`TEXTSPOT3_PARITY_OK ...`), on:
  - random, graph, wild and probe version-3 networks; the probe must give #97's stride-4 probe's result;
  - version-1 and version-2 blobs, against `vocr_textspot2_test`;
- the C loader against the twin's refusals;
- the driver's self-test.

`tests/test_textspot3_runtime.py` adds the arenas and the `--batch` contract.

## Text-presence gate (`vocr_textgate.c`)

The gate of issue #100 (`docs/textgate.md`): does a region hold text at all? It runs a VTDT blob (#97's layout,
`train/textdet_export.py`) on the region's grey and computes only the head's first map K (a text logit per s x s
cell, s = 4 or 8). Region mode scores the k-th largest K code of the map, boxes mode the k-th largest over the cells
whose block meets a finder box (-129 when none does); the region holds text iff score >= t. `eval/textgate.py`
(`kmap`, `kth`, `box_mask`, `score`, on `synth/textdet_twin.run_net`) is the specification, and this runtime gives
exactly its scores and K maps for every region and every blob the twin's loader accepts. It is self-contained: link
`vocr_textgate.c` (plus `vocr_lines.c` / `vocr_lines_rows.c` for the finder mode), not `vocr_textdet.c`.

```c
#include "vocr_textgate.h"

vocr_textgate_model g;                           /* 4 bytes on 32-bit ARM: the checked blob pointer */
int rc = vocr_textgate_load(&g, blob, bytes);    /* any alignment; VOCR_TEXTGATE_E_BLOB if refused */
size_t need = vocr_textgate_arena_bytes(&g, VOCR_TEXTGATE_BOXES, 960, 544);   /* exact */

/* the push form: feed it the grey rows the finder's first pass reads, in order */
vocr_textgate_state st;                          /* 32 bytes on 32-bit ARM, the caller's */
rc = vocr_textgate_begin(&st, &g, VOCR_TEXTGATE_BOXES, k, w, h, arena, need);
for (y = 0; y < h; y++) rc = vocr_textgate_push(&st, grey_row_y);   /* copied: valid during the call only */
rc = vocr_textgate_score(&st, &lines, &score);   /* region mode: vocr_textgate_score_region(&st, &score) */
text = !rc && score >= t;

/* one-shot: vocr_textgate_image(&g, mode, k, &im, boxes, arena, need, &score), or _rows over VocrLinesRows */
/* the finder mode behind a flag: g NULL is vocr_find_lines unchanged */
need = vocr_lines_gated_arena_bytes(&g, mode, 960, 544);           /* max of the finder's and the gate's */
rc = vocr_find_lines_gated(&im, VOCR_LINES_GRAD, &g, mode, k, t, arena, need, &lines);   /* or _rows */
```

- Modes: `VOCR_TEXTGATE_REGION` (the gate first; no text: no lines and the finder never runs) and
  `VOCR_TEXTGATE_BOXES` (the finder, then the gate on its boxes in the same arena). k is 1..8. Boxes with a zero
  side meet nothing; any unsigned box is handled exactly (`box_mask`'s semantics, no sum can wrap). The caller
  refines the lines afterwards as today (`vocr_refine_lines` never adds a line to an empty set).
- Push form: rows 0..h-1 in order (grey; the one-shot forms also take `VOCR_LINES_RGBA`: grey = (77 R + 150 G +
  29 B) >> 8); a row is copied into the first layer's ring, so it only has to stay valid during the call. In boxes
  mode `score` may be asked again with other boxes. Errors of `begin` or `push` stick; `*score` is written only on
  success. Return codes: `VOCR_LINES_E_ARG` (NULL pointers, an unloaded model, a bad mode or k, a zero side or one
  over 2048, a row past the last, a score before the last row), `VOCR_LINES_E_ARENA`, `VOCR_LINES_E_SOURCE` (a NULL
  row from a row source), `VOCR_TEXTGATE_E_BLOB` (load, every check of `synth/textdet_twin.Model`). The finder mode
  returns the gate's or the finder's error with no lines, and checks its arena against
  `vocr_lines_gated_arena_bytes` up front.
- Streaming: every layer keeps a ring of the (k - 1) d + 1 input rows its kernel spans (one row for a 1x1 layer),
  a pixel's channels side by side, d zero pixels either side; the deepest layer that can produce a row goes first.
  The arena is 20 bytes per layer, the rings, then the K row (region mode) or the K map (boxes mode), plus 3 bytes of
  alignment slack: region mode depends on the width only. Exact bytes for the trained presence blobs, 960x544 region
  / boxes: p8_c4 15,721 / 23,761, p8_c8 27,489 / 35,529, p8_c12 32,873 / 40,913, p4_c8 31,361 / 63,761 (480x272:
  7,981 / 9,961 for p8_c4). In the finder mode the finder's own arena (83,323 B at 960x544) is the larger for every
  presence blob (#97's s8_c24 needs 101,145 B in boxes mode).
- C99, no allocation, no recursion, no global state, integer only (int32 sums; int64 for the requantisation
  product, formed so that every blob's constants are exact), scalar code built for size. External symbols:
  `memset`, libgcc's `__aeabi_uidivmod` on ARM (the ring index, a few calls per row), and for the finder mode
  `vocr_find_lines`, `vocr_find_lines_rows`, `vocr_lines_arena_bytes`. `vocr_lines.c`, `vocr_lines_rows.c`,
  `vocr_lines_refine.c` and `vocr_textdet.c` are not touched (`tests/test_textgate.py` checks their sha256 against
  `eval/textgate_protocol.json`).
- Code (`vita/textgate_size.py`: a program calling one way of using the gate, VitaSDK gcc 15.2 with Rubi's runtime
  flags, `-Wl,--gc-sections`, the `.text` the link map keeps from `vocr_textgate.o`; no `.rodata`):

  | path | Vita -Os | Vita -O2 | PSP -Os |
  |---|---:|---:|---:|
  | region: load, arena_bytes, begin, push, score_region | 1,892 | 2,380 | 3,532 |
  | boxes: load, arena_bytes, begin, push, score | 2,072 | 2,636 | 3,864 |
  | the finder mode (gated, gated_rows), without vocr_lines.c | 2,638 | 3,484 | 4,968 |
  | #97's vocr_textdet path, for comparison | 7,250 | 8,232 | 10,188 |

  Stack (gcc `-fcallgraph-info=su`, every frame static; `psp/stack.py`): `vocr_textgate_push` 264 B at peak on the
  Vita build at `-Os` (320 B at `-O2`, 208 B on the PSP), the one-shot forms 376 B, the finder mode 464 B plus the
  finder's own frames.
- Speed, Cortex-A9 **estimates** (`eval/textgate_a9.py`: the Vita compiler's `-O2` code under plugin qemu-arm with
  the Vita's caches, priced by `vita/a9_cycles.py`, ms = its high bound x 1.0007 at the clock;
  `eval/textgate_a9_20261018.json` has all 20 shapes, both modes and the protocol's MAC-rate figures). Scalar code:
  #97's NEON detector ran s8_c24 at an estimated 53 ms per 900x150 box. Boxes mode adds 0.1-0.6%.

  | shape | blob | 900x150 @333 / @444 | 450x80 @333 | 960x544 @333 | 480x272 @333 |
  |---|---:|---:|---:|---:|---:|
  | p8_c4 | 816 | 97 / 73 ms | 26 ms | 372 ms | 94 ms |
  | p8_c6 | 1,280 | 118 / 89 ms | 31 ms | 453 ms | 114 ms |
  | p8_c8 | 1,540 | 135 / 101 ms | 36 ms | 516 ms | 129 ms |
  | p8_c12 | 2,000 | 156 / 117 ms | 41 ms | 597 ms | 149 ms |
  | p4_c4 | 816 | 135 / 101 ms | 36 ms | 515 ms | 129 ms |
  | p4_c8 | 1,196 | 204 / 153 ms | 54 ms | 780 ms | 195 ms |
  | s8_c8 (#97) | 1,884 | 148 / 111 ms | 39 ms | 569 ms | 142 ms |

  The row kernel runs a row channel by channel and 1x1 layers as dot products: 90 B more than the smallest form
  (pixel by pixel), for about 26% fewer instructions. Nothing of the gate has run on a console.

Checks: `make check-textgate` (in `make check`; strict compiles of `vocr_textgate.c` and its driver
`vocr_textgate_test.c`, then `tests/textgate_parity.py` through the sanitized driver: random blobs of the presence
shapes and #97's, wild requantisation, probe and random-chain blobs, edge sizes, GREY and RGBA, the image, rows and
push forms, both modes with random boxes, k in {1, 2, 4, 8}, the finder mode and the K maps, against the twin; plus
the driver's `--selftest`), and `tests/test_textgate.py`. `tests/textgate_parity.py --blob NAME=FILE --cohort PKL`
runs trained blobs on real regions and reports only equal or differ. The Cortex-A9 builds run the same parity under
qemu-arm:

```sh
G=runtime/scripts/tools/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-linux-gnueabihf/bin/arm-none-linux-gnueabihf-gcc
$G -std=c99 -O2 -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard -static runtime/vocr_lines.c runtime/vocr_lines_rows.c \
   runtime/vocr_textgate.c runtime/vocr_textgate_test.c -o vtg_arm
printf '#!/bin/sh\nexec runtime/scripts/tools/bin/qemu-arm -cpu cortex-a9 ./vtg_arm "$@"\n' > vtg_qemu && chmod +x vtg_qemu
python tests/textgate_parity.py --bin ./vtg_qemu --seeds 1 --regions 3 --selftest 6
```

## Stroke split (`vocr_lines_split.c`)

Issue #98 (`docs/translucent-box.md`): over a translucent dialogue box with a busy scene behind it, the finder can
return one box for two lines. `vocr_split_lines` / `vocr_split_lines_rows` take the finder's `VocrLineSet` and split
a box at valleys of a row profile of thin strokes of both polarities; `synth/lines_split.py` is the bit-exact twin
and the specification.

```c
size_t need = vocr_lines_arena_bytes(w, h);              /* the finder's arena serves the split and the refinement */
vocr_find_lines_rows(&src, VOCR_LINES_GRAD, arena, need, &set);
vocr_split_lines_rows(&src, VOCR_SPLIT_DEFAULT, arena, need, &set, &rows);    /* 0 = off: set unchanged */
vocr_refine_lines_rows(&src, VOCR_REFINE_DEFAULT, arena, need, &set, &rows);  /* optional, as before */
```

- C99, no allocation or recursion, integer only. Arena `vocr_split_arena_bytes(w, h) = 3 + up4(3w) + 3 up4(2h)`
  (6,147 B at 960x544), never more than `vocr_lines_arena_bytes` (tested on a grid of sizes). Reads only each box's
  own pixels, twice, top to bottom (boxes under 16 rows are not read), so a caller can split each box on the strip
  it holds. Lines keep their box's columns and dark flag and stay in order; nothing is dropped.
- Code (VitaSDK gcc 15.2, Thumb-2): 3,092 B at `-O2`, 2,432 B at `-Os`; deepest frame 3,712 B (`split`, three
  256-entry histograms). Cost (estimate, A9 cycle model under qemu-arm): 194 instructions per pixel of the boxes it
  profiles, about 79 ms for a 900x150 region at 333 MHz read whole, 6.8 ms for the mean development region
  (`eval/translucent_box_a9_20261018.json`).
- Checks: `make check-split` (strict compile; the sanitized driver `vocr_lines_split_test.c`'s `--selftest`: flags 0
  is the identity, the in-memory, row-callback and per-box strip forms and the minimum misaligned arena agree, lines
  stay inside their box, errors leave the set unchanged; then `tests/lines_split_parity.py` against the twins, both
  forms, and `tests/lines_split_merged.py` on merged-line regions, where the split acts), and `tests/test_lines_split.py`.

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
