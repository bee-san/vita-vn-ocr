# vita: the runtime on the PS Vita

`runtime/` built with VitaSDK, a test app that runs it on the console, a check in the Vita3K emulator, and a Cortex-A9 cycle estimate for the Vita's machine code.

What is known so far:
- The Vita build of `runtime/vocr.c` and `vocr_lines.c` compiles warning-free with the runtime's own `-Werror` flags.
- Inside Vita3K, every case of the test bundles gives the host build's text, logits (CRC-32 of every float) and line boxes, bit for bit: 728 of 728 cases of the full bundle in w8i, and 52 of 52 in each of w8i, w8f and f32 on the quick bundle.
- From the Vita build's instruction mix, a 640-px dialogue line (80 CTC frames) of the v1 model should take **117-285 ms in w8i** on one Cortex-A9 core at 444 MHz (149-341 ms in f32). This is an estimate, not a measurement.
- Nothing has run on a PCH-1000 yet. The test app is ready for that; see "On a PCH-1000".

## Files

| | |
|---|---|
| `scripts/fetch_vitasdk.sh` | installs the pinned VitaSDK in user space (no root) |
| `CMakeLists.txt`, `scripts/build.sh` | `libvocr.a` (the runtime and the line finder), `vocr_test.vpk` and a code-size report |
| `app/main.c`, `app/cases.c`, `app/screen.c` | the test app: runs a bundle, checks it against the host, times it, writes `results.tsv` |
| `bundle.py` | writes the bundle the app reads from `ux0:data/vocr/`, with the host runtime's expected results |
| `compare.py` | reads `results.tsv` back: agreement with the host, CER, time, memory |
| `scripts/fetch_vita3k.sh`, `scripts/run_vita3k.sh` | Vita3K in user space, and a headless run under Xvfb |
| `scripts/build_qemu_plugins.sh`, `qemu/insn_mix.c`, `a9_cycles.py` | the Cortex-A9 estimate |
| `mkicon.py`, `sce_sys/` | the LiveArea images (generated at build time) and template |
| `tests/cases_dump.c` | lets `tests/test_vita.py` check the app's bundle parser on the host |

Everything the scripts download goes to `vita/tools/` (gitignored) or a directory you name. SDK binaries, VPKs and bundles are never committed.

## Build

```sh
vita/scripts/fetch_vitasdk.sh                      # vita/tools/vitasdk, about 380 MB
export VITASDK=$PWD/vita/tools/vitasdk PATH=$PWD/vita/tools/vitasdk/bin:$PATH
vita/scripts/build.sh                              # build/vita/vocr_test.vpk, libvocr.a, size.txt
```

- The SDK is vitasdk/autobuilds release `sdk-snapshot-20260926.790.1`, SDK version 0.20260926.321, with arm-vita-eabi-gcc 15.2.0. The x86_64-linux tarball is checked against its sha256.
- The runtime is compiled as `runtime/README.md` asks: C99, `-O2 -ffp-contract=off -mcpu=cortex-a9 -mfpu=neon -mfloat-abi=hard`, with its full warning set and `-Werror`. The line finder also gets `vocr_lines.mk`'s `-Wconversion -Wsign-conversion`.
- VitaSDK's gcc defaults to Thumb-2 and to short enums. `vocr.h` and `vocr_lines.h` pass no enum across the API, so the enum size does not matter.
- Code size with the Vita compiler: `vocr.o` 13,143 B with NEON (10,543 B scalar) and `vocr_lines.o` 4,636 B. The whole test app, `eboot.bin`, is 76 KB.
- CI's `vita (VitaSDK build)` job runs the same build on every pull request and uploads `vocr_test.vpk` as the `vocr_test-vpk` artifact.

## The test bundle

```sh
python vita/bundle.py --out vocr-bundle --profile quick    # 52 cases; --profile full: 728 cases
```

Needs the v1 checkpoint (`--ckpt`, release v0.1.0-baseline) and eval's screenshots (`eval/fetch.py`; `--eval-data` or `$VOCR_EVAL_DATA`).

| | quick | full |
|---|---|---|
| synth-val line crops (grey) | 8 | 100 |
| vita-test line crops (RGBA, GT boxes) | 24, spread over the games | all 206 |
| psp-test line crops (RGBA, GT boxes, 480x272 screenshots) | 8 | all 245 |
| vita-test dialogue regions (line finder + recognizer) | 8 | all 84 |
| psp-test dialogue regions | 4 | all 93 |

`cases.bin` is 3.4 MB for quick and 38.9 MB for full; the two weights blobs add 3.9 MB. PSP screenshots are in the bundle because PSP VNs also run on the Vita (Adrenaline), and their small text is a harder test of the same code.

Each case carries the host build's result for every mode: the text, the CRC-32 of the logits, the frame count and, for regions, the line boxes. The host results come from `runtime/build/libvocr.so` and a host build of `vocr_lines.c`. Line crops of real screenshots take their polarity from `vocr_line_dark()` and run with `VOCR_INVERT` when the text is dark, which is what the line finder makes the overlay do. The bundle also has the weights blobs and a 16-px screen font (`font.bin`). The font is rasterised from the OFL font M+ 1p, and its licence goes with it as `font-OFL.txt`.

**The bundle holds crops of copyrighted screenshots. Keep it on your own machines; never commit or share it.**

## On a PCH-1000

You need a PS Vita that runs homebrew (HENkaku, h-encore or Ensō) and VitaShell.

1. **Get the VPK and a bundle.** Take `vocr_test.vpk` from CI's `vocr_test-vpk` artifact or build it (above). Write a bundle on a PC with `vita/bundle.py`.
2. **Copy the files.** In VitaShell, start FTP (SELECT) or USB. Put `vocr_test.vpk` anywhere on `ux0:`, and copy the bundle folder's contents to `ux0:data/vocr/`.
3. **Install.** Install the VPK in VitaShell with X. A bubble called "vita-vn-ocr test" (title ID VVOC00001) appears.
4. **Run it.** Close other apps first, then start it from the LiveArea.
   - The app asks for 444 MHz with `scePowerSetArmClockFrequency`, as homebrew emulators do, and logs the clock it got (`# clock arm_mhz=` in `results.tsv`). No overclocking plugin is needed, and 500 MHz is never used.
   - Each case runs once to check the logits, then three timed runs.
   - Going by the estimate below, the quick bundle takes 1-2 minutes and the full one 15-30. With `models=w8i` in `config.txt`, the full bundle takes 4-10 minutes.
   - The screen shows one line per case: the status per model, the median ms and the decoded text. CIRCLE stops early; START exits once it is done.
5. **Read the results.** Copy `ux0:data/vocr/results.tsv` back, then run:

   ```sh
   python vita/compare.py --bundle vocr-bundle results.tsv
   ```

   Please attach `results.tsv` and that output to issue #6.

`config.txt` in the bundle sets the run:
- `clock=333` measures at the default clock;
- `repeat=N` sets the number of timed runs;
- `models=w8i` runs one mode;
- `pmu=0` turns off the ARM performance counters;
- `cpu=` picks the core the worker runs on.

The app first tries the counters through ScePerf. If the firmware allows it, `results.tsv` also gets cycles, instructions, NEON instructions, L1 D-cache misses and stall cycles per run, and `compare.py` prints cycles per frame. That number replaces the estimate below. If ScePerf is not available, the app says so (`# pmu state=-1`) and goes on.

What the statuses mean:
- MATCH: the console reproduced the PC bit for bit.
- TEXT: same text, other logits. Possible only through denormals, which ARMv7 NEON flushes to zero (`runtime/README.md`). None were seen in Vita3K.
- DIFF: a different text.
- ERR: an error code.

## Vita3K (emulator check)

```sh
vita/scripts/fetch_vita3k.sh                                   # vita/tools/vita3k/4154
vita/scripts/run_vita3k.sh vocr-bundle build/vita/vocr_test.vpk out models=w8i repeat=1
```

- **Pin.** Build 4154 (v0.2.1, commit b75d5e62, 2026-10-09), by the AppImage's sha256, from that build's release in Vita3K/Vita3K-builds (the rolling "continuous" release of Vita3K/Vita3K is the fallback).
- **Running it here.** The binary needs glibc 2.43, so the script unpacks Debian's libc6 2.43-7 and libwayland-client0 1.26.0-1 beside it and starts Vita3K through that loader. It runs in a throwaway sandbox (config, `ux0:` and logs under `$TMPDIR`, never `$HOME`), with the SDL frontend and OpenGL on Mesa llvmpipe under Xvfb. No PS Vita firmware is needed: the app uses only HLE'd system calls.
- **Results.** Vita3K runs the Vita's own ARM machine code (`eboot.bin`) through a JIT. Every run reproduced the host exactly:

| bundle | modes | code | cases | MATCH | other |
|---|---|---|---|---|---|
| full | w8i | this branch | 728 | 728 | 0 |
| quick | w8i, w8f, f32 | this branch | 52 each | 52 each | 0 |
| full, before psp-test was on `main` | w8i | before and after the `mm_i16` change below | 390 | 390 each time | 0 |
| quick, before psp-test | w8i, w8f, f32 | before and after the `mm_i16` change | 40 each | 40 each time | 0 |

- **Emulator timing is not console timing.** Under Vita3K, w8i took 6.6-8.0 ms per CTC frame and f32 and w8f 74-100 ms, varying with the load on the shared host. That is a property of the x86 JIT, not of a Cortex-A9.

On the 728-case run, `compare.py` also scores the console's own pipeline against the hand-checked text (v1, w8i):

| | lines | CER | exact |
|---|---|---|---|
| vita-test GT-box crops, polarity rule | 206 | 9.05% (the figure `runtime/README.md` reports for that rule on the host) | 64.6% |
| psp-test GT-box crops, polarity rule | 245 | 10.80% | 60.8% |
| synth-val crops | 100 | 0.39% | 98.0% |

| dialogue regions (`vocr_find_lines`, `VOCR_INVERT` on dark lines) | regions | line recall | pipeline CER |
|---|---|---|---|
| vita-test | 84 | 95.1% | 20.16% |
| psp-test | 93 | 90.2% | 30.96% |

These numbers come from `eval/metrics.py`'s normalisation and eval's region matching. They are not eval's benchmark numbers, which feed crops without inversion (`docs/results.md`).

## Memory on the Vita

Reported by the app, Vita build (so far in Vita3K; a console run reports the same fields):

| | w8i | w8f | f32 |
|---|---|---|---|
| weights blob | 859,904 B | 859,904 B | 3,067,840 B |
| arena (`vocr_arena_bytes`, any line up to 1024 px) | 124,832 B | 118,432 B | 116,384 B |
| line finder arena (`vocr_lines_arena_bytes`) | 83,323 B for a whole 960x544 screen; 41,395 B for the largest test region | same | same |
| `vocr_model` + `vocr_ctx` | 1,772 + 1,984 B | same | same |
| stack, measured | 4,100 B | same | same |
| code (`vocr.o` + `vocr_lines.o`) | 17,779 B | same | same |
| total | about 1.09 MB | about 1.09 MB | about 3.29 MB |

- The arena sizes equal the host's `vocr_arena_bytes()`, so the console uses the bytes the runtime README reports.
- The stack figure comes from a probe thread that paints its 32 KiB stack. It runs `vocr_arena_bytes()`, `vocr_init()`, `vocr_run()` on a 1024-px line and both line finders, thread entry included.
- The blob and arena sizes are fixed by the code and the model. The stack figure was measured in Vita3K, which runs the same machine code; the console run will confirm it.

## Cortex-A9 estimate

```sh
vita/scripts/build_qemu_plugins.sh                 # qemu 10.0.13 arm-linux-user with TCG plugins
python vita/a9_cycles.py measure --out a9 --lines 40
python vita/a9_cycles.py report a9
```

How it works:
1. `a9_cycles.py` compiles `runtime/vocr.c` with the Vita's compiler and flags, and links that object into an ARM Linux `vocr_cli`. qemu-arm then runs the Vita's exact machine code for the runtime.
2. It runs 40 lines, 20 from synth-val and 20 from vita-test (1,828 CTC frames), with two plugins:
   - `qemu/insn_mix.c` counts every executed instruction and every switch between the NEON and VFP units. The A9 runs one at a time.
   - qemu's cache simulator is set up like the Vita: 32 KiB 4-way L1 caches with 32-byte lines, and the 2 MiB L2.
3. Each instruction gets its Cortex-A9 issue cost from the NEON MPE TRM (ARM DDI 0409I, tables 3-2 to 3-10). NEON loads and stores are costed as unaligned, because gcc emits no alignment hints.
4. The result is a range:
   - **Low bound:** perfect overlap. It is the largest of: NEON + VFP issue cycles, instructions / 2 (dual issue), load/store-pipe cycles, and L2-to-L1 refill bandwidth. Add 3 cycles per unit switch.
   - **High bound:** no dual issue, 10 cycles per unit switch (with `VLDR`/`VSTR` counted as VFP), 11 per NEON-to-ARM register transfer, and every miss paying full latency: 30 cycles for an L2 hit, plus 110 ns for DRAM.

v1, per CTC frame (1,002,720 MACs) and per 640-px line:

| mode | instructions / frame | NEON issue cycles | L1D misses | L2 misses | cycles / frame | 640-px line, 444 MHz | 333 MHz |
|---|---|---|---|---|---|---|---|
| w8i | 893,668 | 547,481 | 6,081 | 34 | 649K-1,580K | **117-285 ms** | 156-380 ms |
| w8i at 71bc277 (before the change below) | 1,023,601 | 553,425 | 6,074 | 34 | 754K-1,796K | 136-324 ms | 181-431 ms |
| w8f | 988,636 | 991,159 | 6,027 | 34 | 1,036K-1,265K | 187-228 ms | 249-304 ms |
| f32 | 815,657 | 782,135 | 15,291 | 7,113 | 827K-1,890K | 149-341 ms | 199-433 ms |

What the numbers say:
- **Every mode is NEON-bound at the low end.** The busiest code is `mm_i16`/`mm_f32` (the pointwise layers and the head). Then come `step_op` with the inlined stem and depthwise convolutions (119K instructions per frame), `run_head`'s argmax over 5,355 classes in VFP compares (108K), and preprocessing (19K).
- **w8i has the cheapest arithmetic.** One `VMLAL.S16` does 4 MACs per cycle, against 2 for `VMLA.F32`.
- **f32 has the widest range.** Its 3.07 MB blob does not fit the 2 MiB L2, so its weights stream from DRAM: 7,113 L2 misses per frame.
- **w8f is the slowest at the low end**, because it dequantises every weight block again on every tile (`runtime/README.md`).
- **These are estimates.** Unknowns include:
  - how well the A9 overlaps NEON with integer work;
  - whether VLDR/VSTR count as VFP for unit switches;
  - whether the L1 prefetcher is on;
  - how much of the L2 the game running beside the overlay takes.

  The PMU columns of a console run will settle the first three.
- The runtime README's figure for v1 (80M MACs at the research's 0.24-0.30 GMAC/s, i.e. 0.27-0.33 s per 640-px line) sits at or above the top of these ranges.

**The `mm_i16` change in this branch.**
- The estimate found that gcc 14.2 and 15.2 stored all eight w8i accumulators to the stack on every pass of the inner loop: 16 `VSTR` per 32 `VMLAL`, 179K VLDR/VSTR per frame.
- `mm_i16` now holds them in named variables. That cuts w8i from 1.02M to 0.89M instructions per frame, and the 640-px estimate from 136-324 ms to 117-285 ms.
- The output is bit-identical: parity-arm (NEON and scalar) on the fixture, NEON parity on all of vita-test in all three modes, and every Vita3K run above.
- What is left in w8i:
  - `mm_i16`'s plain-C path for leftover positions, which runs when a call's position count is not a multiple of 4 (head batches of 17-19 frames, the ends of a line). gcc vectorises it but keeps its accumulators in memory: 62K of the 80K VLDR/VSTR left per frame;
  - `run_head`'s per-class VFP argmax.

  Both are follow-ups for `runtime/`.

## Not done here

- No run on a PCH-1000. Console speed above is an estimate, and emulator times are not console times.
- ScePerf's counters are untested on retail firmware. The app falls back to timing only.
- The PSP itself (Allegrex, PSPSDK, PPSSPP) is the `psp` agent's job; here psp-test only runs on the Vita.
- Vita3K's build 4154 will disappear from its rolling release; `fetch_vita3k.sh` fetches it from Vita3K/Vita3K-builds, or takes `VITA3K_APPIMAGE=<kept copy>`.
