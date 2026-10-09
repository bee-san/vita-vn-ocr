# psp: the recognizer on a PSP

`runtime/vocr.c` built for the PSP's Allegrex CPU with pspdev (`-O2 -G0`, the scalar FPU
path), a test EBOOT that runs it on line crops from the memory stick, and a PPSSPPHeadless
harness that checks it against the host runtime. **Nothing here has run on a PSP console.**
PPSSPP's timing is emulated time, not console time.

## Results

- **Same output as the host.** In PPSSPPHeadless 1.20.4 the PSP build's logits are
  bit-identical to `runtime/tests/vocr_ref.py` (the numpy model of the C arithmetic that the
  host build also equals) on every line of psp-test (245), vita-test (206) and synth-val
  (first 300), in f32, w8f and w8i, for both the scalar and the VFPU build. Strings equal
  PyTorch's except the one known w8i near tie (psp-test:105, `runtime/README.md`). Through
  `eval/run.py`, the PSP build returns the host runtime's string on every line, so the CER is
  the host's: int8 v1 23.87% on vita-test, 46.57% on psp-test.
- **Memory: about 1 MB of the PSP-1000's 24 MB user partition.** int8 blob 859,904 B, arena
  118,432 B (w8f, any line width), worker stack 16 KB of which 2,992 B are used, code 18 KB.
  The fp32 blob is 3,067,840 B.
- **Speed in PPSSPP (emulated, about one cycle per instruction at 333 MHz):** v1, w8f, tile 16:
  12.1 ms per CTC frame, 0.38 s for the median psp-test line (31 frames) and 0.73 s at the
  90th percentile (61 frames). The opt-in VFPU build: 8.0 ms per frame, 0.25 / 0.47 s. My
  console estimate is 1.05-1.45× the emulated time for the scalar build and 0.98-1.55× for the
  VFPU build (assumptions below). The real figure is one EBOOT run on a PSP away.
- **`runtime/vocr.c` change:** the scalar kernels keep their sums in registers. This gives the
  same bits on every build (host x86, ARM NEON and scalar under qemu, PSP in PPSSPP) and
  2.1-2.7× fewer emulated PSP cycles (f32 30.1 → 11.1 ms per frame). Host gcc `-O2` builds
  get 1.9-2.9× faster.
- **VFPU:** worth about 1.5× (f32 1.59×, w8f 1.52× fewer emulated cycles) and bit-exact in
  PPSSPP. It is kept behind `make -C psp VFPU=1` (`-DVOCR_PSP_VFPU`) until a console run's
  built-in host check passes: the real VFPU flushes denormals to zero, and PPSSPP is the only
  place it has run.
- **PSP-lean variant:** not needed at v1's size, needed for anything bigger. v1's p90 psp-test
  line is already 0.77-1.06 s by the console estimate (scalar w8f), so a model wider or deeper
  than v1 (about 1.0 M MACs per CTC frame) needs its own PSP variant: the v1 layout with arch's
  3,500-character vocab (about 658 KB int8, 0.82× v1's PSP time, +0.22 points of real CER in
  arch's 5k-step sweep, 95% CI −0.48 to +0.93). Or the VFPU build, once a console has
  checked it. Details below.
- **Use w8f on the PSP.** Allegrex does every integer multiply through HI/LO (`mult` then
  `mflo`), so w8i is the slowest mode (16.4 ms per frame) and gets no VFPU path.

## Files

| file | what |
|---|---|
| `scripts/fetch_pspdev.sh` | pspdev v20261001 (psp-gcc 15.2.0, binutils 2.44, pspsdk 6f15c15), sha256-pinned, in user space. On hosts with glibc < 2.38 it also unpacks Ubuntu 24.04's glibc 2.39 next to it and repoints the toolchain's host programs with patchelf 0.18.0 (both sha256-pinned) |
| `scripts/build_ppsspp.sh` | PPSSPPHeadless from the PPSSPP 1.20.4 release tarball plus SDL2 2.32.10, both sha256-pinned. FFmpeg, X11 and Wayland are off |
| `Makefile`, `main.c` | the EBOOT (`vocr_test`, title "vocr OCR test"): `make -C psp`, `make -C psp VFPU=1`, `make -C psp size` (code size, stack chains) |
| `pack.py` | crop packs from eval's sets, the host runtime's expected outputs, memory-stick bundles |
| `ppsspp_cli.py` | runs the EBOOT in PPSSPPHeadless with `vocr_cli`'s command line, so it drops into `runtime/tests/parity.py --cli` and `eval/run.py --model cmd:...`; `--bundle DIR` runs a bundle as a console would |
| `run_all.sh`, `summarize.py` | every parity, self-check, timing and eval run in the tables below |
| `shapes.py` | PSP blob, arena and emulated time of arch's candidate model shapes |
| `cachesim.c` | host D-cache model of `vocr_run()` (`make -C psp cachesim`) |
| `stack.py` | worst stack chain per entry point from gcc's `-fcallgraph-info` |

The app reads `model.vocr`, `lines.bin` (vocr_cli's pack format), the optional `expect.txt`
(the host's CRC-32 of every line's logits, plus its text) and `vocr.cfg` from its own folder.
It writes `texts.txt`, `results.txt` (per line: frames, µs, host check) and `stats.json`. vocr
runs in its own thread, so its stack use can be read back. Free memory is measured by
allocating it: pspsdk's `pspSdkTotalFreeUserMemSize()` compiles to an endless loop in this
pspsdk.

## Run it on a PSP

You need a PSP with custom firmware that runs homebrew EBOOTs (ARK-4, PRO or LME; PSP-1000
included).

```sh
psp/scripts/fetch_pspdev.sh                                   # once
make -C psp                                                   # psp/EBOOT.PBP, scalar
python train/export.py --ckpt CKPT --out v1_w8.vocr --int8
python psp/pack.py build --set psp-test --out packs/psp-test  # eval's crops (copyrighted: keep private)
python psp/pack.py bundle --blob v1_w8.vocr --mode w8f --tile 16 --pack packs/psp-test --out stick
make -C psp clean && make -C psp VFPU=1                       # optional second app, the VFPU build
python psp/pack.py bundle --eboot psp/EBOOT.PBP --name VOCRV --blob v1_w8.vocr --mode w8f --tile 16 \
    --pack packs/psp-test --out stick
```

1. Copy `stick/PSP` to the root of the memory stick. That gives `ms0:/PSP/GAME/VOCR/`, and
   `VOCRV/` too if you built it.
2. Start "vocr OCR test" (and "vocr OCR test (VFPU)") from Game → Memory Stick. It sets
   333 MHz (`clock=` in `vocr.cfg`). It prints each line's time and whether it matches the
   host, then the totals, memory and stack. Press START to exit.
3. Send back `stats.json` and `results.txt` from each folder.
   `sceKernelGetSystemTimeWide()` is a 1 MHz hardware timer, so these are console times.
   `text_ok` and `logits_ok` equal to `lines` means the console computed the host's bits.

## Measurements

Environment: the shared 16-core x86 host (Amazon Linux 2023), PPSSPPHeadless 1.20.4 (JIT
core), the v1 checkpoint (`models/v1/ckpt_best.pt`) exported at this head. The crops are
eval's GT-box crops (RGBA). psp-test was taken from eval's PR #5 at 9ea9710, before it merged;
the manifest on main is byte-identical (sha256 2b8db084…). The model is the
EBOOT's (`MEMSIZE=0`: the 24 MB user partition of a PSP-1000). PPSSPP's emulated time is
deterministic, so one run per row. Command: `psp/run_all.sh OUT models/v1/ckpt_best.pt`.
Raw outputs are in `$HERD/shared/artifacts/psp/results`.

### Memory

Measured inside the app on the full psp-test run (`stats.json`):

| | f32 | w8f | w8i |
|---|---|---|---|
| blob | 3,067,840 B | 859,904 B | 859,904 B |
| arena, any width up to 1024 px (tile 4 / tile 16) | 116,384 / 351,488 B | 118,432 / 353,536 B | 124,832 / 385,280 B |
| `vocr_model` + `vocr_ctx` | 1,772 + 1,984 B | same | same |
| vocr thread stack used (of 16 KB) | 2,992 B | 2,992 B | 2,992 B |
| `vocr.o` code (scalar / VFPU build) | 18,219 / 18,759 B | same | same |
| user memory free at start | 24,408,832 B | 24,408,832 B | 24,408,832 B |

- The deepest stack chain by gcc's call graph is `vocr_arena_bytes()` at 2,616 B (it plans
  into a local `vocr_ctx`). `vocr_run()` is 920 B plus libgcc's soft-float helpers, which the
  resize tables use a few thousand times per line. A 4 KB thread stack is enough.
- With the w8f model, a tile-4 arena and the worker thread loaded, free memory fell from
  24,408,832 B to 20,640,768 B. 994,720 B of that is the OCR (blob + arena + stack). The rest
  is the self-check's 2,763,180-B logits buffer and `expect.txt`, which an integration
  doesn't need.
- The whole EBOOT (newlib, pspsdk, the app) is 218,298 B.

**Next to a running VN (a plugin) on a PSP-1000.** The OCR needs about 1.0 MB at once:

| item | v1 | v1 + 3,500 vocab | + bottleneck 64 |
|---|---|---|---|
| int8 blob | 859,904 B | 657,616 B | 539,216 B |
| W8F arena, tile 4 | 118,432 B | 118,432 B | 116,000 B |
| line finder arena (`vocr_lines`, a full 480×272 screen) | 25,355 B | same | same |
| code (`vocr.o` + `vocr_lines.o`) | 18,219 + 7,644 B | same | same |
| stack | 4 KB | 4 KB | 4 KB |
| total | 1.03 MB | 0.83 MB | 0.71 MB |

The crop can be read in place from the framebuffer (`vocr_image.stride`), or copied as grey
(at most 480 × 272 = 130 KB). A plugin can load the blob from the memory stick per request
and free it afterwards, but it still needs the full amount while it runs.

The VN owns the 24 MB user partition. A plugin has to fit into what the VN leaves free there
or in the kernel partition, and I measured neither: that needs a console, the VN and a plugin
build. On hardware, pspsdk documents `sceKernelTotalFreeMemSize()` (`kernel_total_free_start`
in `stats.json`) as the kernel partition's free memory. In PPSSPP it returns the user
partition. On a PSP-2000 or later, the second 32 MB lies outside a 24 MB game. Whether a
plugin can use it depends on the CFW, which I didn't check.

### Speed in PPSSPP, and a console estimate

Emulated ms per line (full sets; psp-test: 245 lines, 8,029 frames; vita-test: 206 lines,
8,355 frames). Every row passed the self-check on every line:

| build | mode | tile/head | ms per frame | psp-test mean / median / p90 / max | vita-test median / p90 |
|---|---|---|---|---|---|
| scalar | f32 | 4/16 | 11.03 | 361 / 347 / 670 / 915 | 417 / 824 |
| scalar | w8f | 16/16 | 12.07 | 395 / 381 / 731 / 990 | 447 / 897 |
| scalar | w8f | 4/16 | 13.02 | 427 / 405 / 782 / 1,070 | 491 / 966 |
| scalar | w8i | 4/16 | 16.44 | 539 / 511 / 1,004 / 1,367 | 625 / 1,250 |
| VFPU | f32 | 4/16 | 6.92 | 227 / 211 / 409 / 565 | 260 / 514 |
| VFPU | w8f | 16/16 | 7.96 | 261 / 245 / 472 / 641 | 294 / 588 |
| VFPU | w8f | 4/16 | 8.92 | 292 / 271 / 529 / 723 | 339 / 653 |

- Before this PR's kernel change, the same EBOOT took 30.1 (f32), 33.5 (w8f) and 34.1 (w8i)
  ms per frame (first 8 psp-test lines, tile 4).
- Tile 16 helps w8f, because each int8 weight block is dequantised once per tile. It costs
  235 KB more arena.

PPSSPP charges one cycle per instruction. The exceptions are `div.s` (29 cycles) and the
VFPU's `vadd`/`vsub` (2; the VFPU docs measure a throughput of 1). It models no cache and no
pipeline. Allegrex is a single-issue in-order core (at most one instruction per cycle), so the
emulated time is close to a lower bound. The console adds:

- **D-cache misses** (16 KB, 64-byte lines). `cachesim.c` replays every load and store of
  `vocr_run()` on the first 40 psp-test lines. Per frame, 2-way LRU, with 1-way / 4-way in
  brackets: f32 tile 4: 10,122 misses (17,345 / 9,608) and 1,040 writebacks. w8f tile 16:
  6,592 (9,863 / 6,305) and 1,434. w8f tile 4: 5,814 (10,252 / 4,900) and 1,658. w8i:
  4,224 (10,254 / 3,935) and 1,104. Reading the int8 weights alone takes at least about 1,400
  line fills per frame at tile 4 and 730 at tile 16 (the 230,400 B of backbone matrices once
  per tile, the 514,560-B head once per 16 frames). The rest are activation reads: each
  pointwise layer reads its whole input tile once per 8 output channels.
- **Stalls** that PPSSPP doesn't model. The f32 and w8f inner loops keep 16 independent sums,
  so FPU and VFPU latency (VFPU arithmetic: 5 cycles) is covered. w8i's `mult`/`mflo` pairs
  are back to back.

Estimate per frame, 2-way cache:
- low = the emulated time − PPSSPP's extra `vadd` cycle (0.72 ms per frame in the VFPU
  build), + 30 cycles per miss;
- high = 1.25 × the emulated time + 100 cycles per miss and per writeback.

The miss cost and the stall allowance are assumptions; I found no measured figure for either.

| build, mode | emulated ms per frame | console estimate | median psp-test line | p90 line |
|---|---|---|---|---|
| scalar w8f, tile 16 | 12.07 | 12.7-17.5 ms (1.05-1.45×) | 0.40-0.55 s | 0.77-1.06 s |
| VFPU w8f, tile 16 | 7.96 | 7.8-12.4 ms (0.98-1.55×) | 0.24-0.38 s | 0.46-0.73 s |
| scalar f32, tile 4 | 11.03 | 11.9-17.1 ms (1.08-1.55×) | 0.38-0.54 s | 0.73-1.04 s |

Other speed levers, not done:
- loop order that keeps the pointwise inputs in cache across output blocks (bit-identical);
- the Media Engine, a second Allegrex without a VFPU (kernel mode, CFW only);
- the 16 KB scratchpad.

### Model shapes on the PSP (`shapes.py`)

The shapes are random-initialised (the work doesn't depend on the weights), exported, and run
on the first 40 psp-test lines (34.4 frames per line at height 32, 26.0 at height 24). The
table gives emulated ms per line (f32 tile 4, w8f tile 16):

| shape | params | MACs per frame | int8 blob B | W8F arena B (tile 4 / 16) | scalar f32 | scalar w8f | VFPU f32 | VFPU w8f | vs v1 |
|---|---|---|---|---|---|---|---|---|---|
| v1 | 761,771 | 1,002,720 | 859,904 | 118,432 / 353,536 | 381 | 416 | 239 | 274 | 1.00 |
| width 0.5 | 587,611 | 652,416 | 669,824 | 63,392 / 183,296 | 254 | 280 | 161 | 187 | 0.67 |
| width 0.75 | 661,443 | 801,072 | 751,616 | 90,912 / 268,416 | 309 | 339 | 196 | 225 | 0.81 |
| width 1.5 | 1,041,915 | 1,564,992 | 1,155,968 | 173,472 / 523,776 | 576 | 627 | 354 | 405 | 1.51 |
| width 2.0 | 1,428,043 | 2,339,232 | 1,558,016 | 228,512 / 694,016 | 841 | 914 | 506 | 578 | 2.20 |
| bottleneck 64 | 584,267 | 825,216 | 682,112 | 116,000 / 349,568 | 321 | 349 | 206 | 233 | 0.84 |
| bottleneck 48 | 495,515 | 736,464 | 593,216 | 114,784 / 347,584 | 292 | 316 | 189 | 213 | 0.76 |
| vocab 3,500 | 581,933 | 824,736 | 657,616 | 118,432 / 353,536 | 316 | 343 | 200 | 227 | 0.82 |
| vocab 2,500 | 484,933 | 728,736 | 548,608 | 118,432 / 353,536 | 280 | 303 | 178 | 202 | 0.73 |
| height 24 | 761,771 | 943,776 | 859,904 | 97,216 / 282,144 | 270 | 299 | 171 | 199 | 0.72 |
| depth 2 | 812,427 | 1,201,696 | 923,264 | 156,832 / 481,024 | 455 | 494 | 289 | 327 | 1.19 |
| ctx 4 | 838,955 | 1,078,368 | 946,176 | 130,720 / 384,256 | 409 | 450 | 257 | 298 | 1.08 |
| vocab 3,500 + bottleneck 64 | 463,757 | 706,560 | 539,216 | 116,000 / 349,568 | 276 | 298 | 177 | 200 | 0.72 |
| the same at height 24 | 463,757 | 647,616 | 539,216 | 94,784 / 278,176 | 191 | 208 | 123 | 141 | 0.50 |

Across every shape the PSP build costs 3.5-3.8 emulated cycles per MAC (scalar f32), 3.8-4.2
(scalar w8f, tile 16), 2.1-2.5 (VFPU f32) and 2.4-2.8 (VFPU w8f). A 333 MHz PSP is therefore
at about MACs per line × 4 / 333 MHz in the default build.

**PSP-lean decision.** v1 runs a median psp-test line in 0.38 s emulated and fits the PSP's
memory, but its p90 line is 0.77-1.06 s by the console estimate, so v1 is about the largest
model the scalar PSP build carries at under about 1 s per line. arch's sweep (PR #10, a draft,
`docs/arch-sweep.md` at 3544c61) measured these real-CER costs against the v1 layout at
5,000 steps:
- vocab 3,500: +0.22 points (CI −0.48 to +0.93);
- width 0.75: +2.86;
- height 24: +3.33.

Its bottleneck and depth results weren't in yet. A release model wider or deeper than v1
(width 1.5: 1.51×, width 2.0: 2.20×, depth 2: 1.19×, ctx 4: 1.08× v1's PSP time) puts the p90
psp-test line at 0.8-1.6 s emulated, about 0.8-2.3 s on the console by the estimate above. In
that case train the PSP-lean variant with the same recipe:
- the v1 layout with the 3,500-character vocab;
- if a PSP-1000 plugin has to save more memory, add bottleneck 64 (539,216 B, 0.72×), once
  arch has measured its accuracy.

Height 24 is faster but costs accuracy.

### Host and ARM checks of the `vocr.c` change

- `make -C runtime check` passes with gcc (ASan + UBSan) and clang.
- On psp-test, the host logits' CRC-32s are identical to main's runtime in f32, w8f and w8i.
- `runtime/tests/parity.py --set psp-test` passes for the Cortex-A9 NEON and scalar builds
  under qemu-arm: bit-exact logits on all 245 lines in every mode. The PSP and x86 machine
  code of the final head is identical to the one measured above. Its last edit only removes
  NEON-unreachable code.
- ARM code size: NEON 12,479 → 13,683 B, scalar 9,855 → 11,699 B.
- Host speed: one core of the shared x86 host (load average about 20), psp-test, median of 3
  interleaved runs:
  - f32: 36.9 → 12.6 ms per line;
  - w8f: 39.1 → 15.5;
  - w8i: 27.4 → 14.3.

## Reproduce

```sh
psp/scripts/fetch_pspdev.sh /path/to/tools                  # or psp/tools; PSPDEV=.../pspdev
$HERD/heavy.sh psp/scripts/build_ppsspp.sh /path/to/tools   # PPSSPP=.../ppsspp/PPSSPPHeadless
export VOCR_EVAL_DATA=$HERD/shared/artifacts/eval PSPDEV=... PPSSPP=...
psp/run_all.sh OUT models/v1/ckpt_best.pt                   # parity, self-check, timing, eval: about 40 min
python psp/pack.py build --set psp-test --n 40 --out packs/psp40
python psp/shapes.py --out OUT/shapes --pack packs/psp40/lines.bin --model-py ARCH/train/model.py \
    --eboot scalar=OUT/EBOOT_scalar.PBP --eboot vfpu=OUT/EBOOT_vfpu.PBP
make -C psp cachesim && psp/build/cachesim v1_w8.vocr --mode w8f --tile 16 --pack packs/psp40/lines.bin
```

`--model-py` takes the arch branch's `train/model.py`, which has `depth` and `ctx`. Without
it, shapes.py skips those shapes. Tests: `python -m pytest tests/test_psp.py` (no PSP
toolchain needed).

## Not done

- Any run on a PSP console: speed, the VFPU's bits, and the free memory next to a VN.
- A PSP plugin (PRX) build. The runtime has no system calls, no globals and no malloc, so it
  drops into one.
- `vocr_lines.c` on the PSP: it compiles (7,644 B) but wasn't run in PPSSPP.
