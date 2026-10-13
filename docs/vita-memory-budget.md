# How big can the recognizer be on a PS Vita PCH-1000?

Written 2026-10-12 for the agents training the next recognizers. Every console speed here is an
**estimate** from a cycle model. No controlled timing has run on a console, and no engine has a
hardware CER.

*Corrected 2026-10-13* on where the overlay's memory comes from (marked "correction" below), after
the SDK documentation and vita-jp-overlay PR [#60](https://github.com/bee-san/vita-jp-overlay/pull/60)
were read: the recognizer's budget is the game process's main memory only; SceShell's 50 MiB
partition, not its 5 MiB PAF heap, is the pool that runs out there; and `*ALL` loads the worker
into SceShell too. The short platform note is [`vita-platform-memory.md`](vita-platform-memory.md);
the full reference is vita-jp-overlay's
[`docs/vita-memory.md`](https://github.com/bee-san/vita-jp-overlay/blob/main/docs/vita-memory.md).

## The answer

The line other agents design to (the fleet's `vita-budget.status`):

```
BUDGET_READY weights<=1600000 arena<=262144 ms<=550
```

| | ceiling | F20 (width 2.0) | H15 (width 1.5) | FL10 (the overlay's device default) |
|---|---|---|---|---|
| int8 W8I blob, bytes | **≤ 1,600,000** | 1,558,016 | 1,155,968 | 657,664 |
| `vocr_arena_bytes()`, W8I, default tiling | **≤ 262,144** | 241,056 | 182,944 | 124,832 |
| est. ms per 640-px line, one A9 core at 333 MHz, high bound | **≤ 550** | 311-548 | 219-398 | 124-235 |
| the same at 444 MHz | ≤ 412 | 234-411 | 164-299 | 93-177 |
| MACs per CTC frame (a proxy; check the ms) | about 2.0-2.35M, depending on shape | 2,339,232 | 1,564,992 | 824,736 |

- **What the ms limit means.** It is the high end of `vita/a9_cycles.py`'s W8I range for an
  80-frame (640-px) line, which is about the p90 real Vita line. It is taken at 333 MHz, the clock
  games run at by default, with the whole 2 MiB L2.
- **Checking a model.** All three limits must hold. `vita/budget_sweep.py check BLOB` tests a
  trained blob against them, reading synth-val lines only. Run it through `heavy.sh`.
- **Margin.** The figure is per 640-px line. For a model at the ceiling, an estimated:
  - full-width 917-px CLANNAD line (110-115 frames): about 760-790 ms at 333 MHz;
  - three-line box of 640-px lines: about 1.65 s, before the model load (section 4).

  The published line mix (20 synth-val plus 20 vita-test lines) costs about 0.8% more per frame
  than `check`'s synth-val lines, so leave a few ms of margin rather than designing to 549 ms.
- **Headroom.**
  - Memory: about 25× (section 1).
  - L2: weights plus arena stay at 89% or less of it.
  - Speed: the ms limit is the pessimistic bound at the slower clock. The owner's earlier goal was
    under about 1 s per line (`runtime/README.md`).

What the numbers say:

- **RAM is not the limit.** Inside a game, one OCR job of a ceiling-sized model needs 2.0-2.4 MB of
  the game's own memory. Free memory seen: 87 MiB in CLANNAD (measured), about 56 MiB in Stardew
  Valley (a third-party screenshot).
- **Speed binds first, then the shared 2 MiB L2.** Up to F20's size, cost grows slightly less than
  linearly with MACs. The knee comes where weights plus arena outgrow the L2, past about 2.0 MB:
  - from there, the worst-case estimate grows faster than MACs: L2 misses add 8-10% to width 2.5's
    high bound and 30-36% to width 3.0's;
  - the best-case estimate stays linear;
  - a game that holds part of the L2 moves the knee down.
- **"More memory is slower" is true today, but because of the overlay's loader, not the cache.**
  - vita-jp-overlay reads and re-hashes the whole weights file on every OCR job, through 8 KiB
    kernel round trips with a 1 ms sleep each.
  - On the console that took 213-359 ms for FL10, more than the recognition itself (36-145 ms).
  - The load grows with the blob: an estimated 0.37-0.63 s for H15 and 0.50-0.85 s for F20.
  - It is an integration bug with cheap fixes. See section 5 and
    [vita-jp-overlay #53](https://github.com/bee-san/vita-jp-overlay/issues/53).
- **None of the overlay's OCR failures came from the recognizer's size.** The failures that stopped
  OCR on the console were integration problems (section 2): the wrong process, eager or full-frame
  buffers, the game's file namespace, an error-code ABI bug, or a pool leaked by a test tool. The
  rest were output quality (FL10, judged by eye) and speed (Meiki, about 8 s per line).

A slightly bigger model than H15 is affordable, up to F20's size and cost. Shapes from the sweep
that fit the ceiling, alongside plain width 2.0 (sizes measured, times estimated at 333 MHz):
- two blocks per stage at width 1.5 (`w150d2`: 1.29 MB, 240,544 B, est. 284-508 ms);
- a larger head bottleneck (`w150b128`: 1.34 MB, 185,376 B, est. 238-421 ms);
- four temporal blocks (`w150c4`: 1.34 MB, 201,376 B, est. 241-441 ms).

Width 2.5 and above do not fit (width 2.5: 2.07 MB, est. 425-789 ms). This ceiling is for the Vita; the
PSP has no NEON and its own budget (`psp/README.md`).

## 1. Where the OCR runs, and which memory it uses

vita-jp-overlay main is at `1a118ef`; `M` below stands for
`https://github.com/bee-san/vita-jp-overlay/blob/1a118ef8ca794bb5e2b5b2a96cd2cd07552d93eb`. It has
three parts:
- a kernel plugin (`*KERNEL`) that captures the screen into a region-sized `KERNEL_RW` buffer and
  brokers the model file;
- a SceShell plugin (`*main`) for the UI and the disk-backed dictionary;
- a worker plugin (`VitaJPOverlay_OCR.suprx`) registered under `*ALL`, which runs the line finder
  and the recognizer in the game's own process. Correction: `*ALL` loads it into SceShell and every
  system application as well, not only into enabled games; its `module_start` finds it is not in a
  game and starts nothing, but the module stays resident (45,056 B of SceShell's pages, PR #60).
  The expert's rule, binding in the overlay's `AGENTS.md`, is one `*TITLEID` section per game.

Facts, each with its source:

| | value | source |
|---|---|---|
| RAM | 512 MiB LPDDR2 main memory, 128 MiB CDRAM | [Physical Memory](https://wiki.henkaku.xyz/index.php?title=Physical_Memory&oldid=21616) |
| the game's partitions | 256 MiB main (+29, +77 or +109 MiB with `ATTRIBUTE2`), 112 MiB CDRAM, 26 MiB phycont | [Memory budget](https://wiki.henkaku.xyz/index.php?title=Memory_budget&oldid=18995); Vita3K [SceSysmem.cpp#L171-L211](https://github.com/Vita3K/Vita3K/blob/7e152b66e85345f175d3d287fcbd7f8abec7574a/vita3k/modules/SceSysmem/SceSysmem.cpp#L171-L211) |
| SceShell, kernel drivers | SceShell: 50 MiB main (`ScePhyMemPartShell`) + 16 MiB CDRAM, shared by all of SceShell and every `*main`/`*ALL` plugin; measured free beside a running game: 28,672 B and 98,304 B. Kernel drivers and kernel plugins: 36 MiB (`ScePhyMemPartKD`), free size unmeasured | same Memory budget page; vita-jp-overlay PR #60; `notes/08`, `notes/26` |
| a game plugin's memory | taiHEN loads the plugin into the game's process (`ksceKernelLoadStartModuleForPid`), before the game's own code runs. Correction: its module pages, thread stacks, TLS and memory blocks are charged to that process's budget. The SDK's Kernel Overview says the memory block service loads modules and allocates stacks, its Kernel Reference that stacks and TLS come from the application's budget, and PR #60 measured the overlay's Shell pages (540,672 B) | taiHEN [plugin.c#L165-L173](https://github.com/yifanlu/taiHEN/blob/309b3800bcb8ebbd5e4f5e5e920af3da3590b829/plugin.c#L165-L173), [hen.c#L345-L380](https://github.com/yifanlu/taiHEN/blob/309b3800bcb8ebbd5e4f5e5e920af3da3590b829/hen.c#L345-L380); vita-jp-overlay `docs/vita-memory.md` sections 3-4 |
| heap | none: the worker links with `-nostdlib`. It allocates one `USER_RW` memblock per job (4 KiB granularity) and frees it when the job ends | [M/vita/CMakeLists.txt#L176](https://github.com/bee-san/vita-jp-overlay/blob/1a118ef8ca794bb5e2b5b2a96cd2cd07552d93eb/vita/CMakeLists.txt#L176), [M/game_ocr/main.c#L146-L151](https://github.com/bee-san/vita-jp-overlay/blob/1a118ef8ca794bb5e2b5b2a96cd2cd07552d93eb/game_ocr/main.c#L146-L151), [#L118-L127](https://github.com/bee-san/vita-jp-overlay/blob/1a118ef8ca794bb5e2b5b2a96cd2cd07552d93eb/game_ocr/main.c#L118-L127) |
| preflight | `sceKernelGetFreeMemorySize` must report free `USER_RW` of at least the workspace plus a 4 MiB reserve, or the job is refused before anything is allocated | [M/game_ocr/main.c#L18](https://github.com/bee-san/vita-jp-overlay/blob/1a118ef8ca794bb5e2b5b2a96cd2cd07552d93eb/game_ocr/main.c#L18), [#L146-L150](https://github.com/bee-san/vita-jp-overlay/blob/1a118ef8ca794bb5e2b5b2a96cd2cd07552d93eb/game_ocr/main.c#L146-L150) |
| free memory seen in games | CLANNAD: 91,226,112 B of free `USER` before a job (one logged sample). Stardew Valley in +109 MiB mode: 309 of 365 MB used, about 56 MiB free (a third-party PSVshell screenshot, not our measurement) | `notes/27`; [PSVshell README](https://github.com/Electry/PSVshell#readme) |
| file mapping | VitaSDK has no `mmap`, so the weights must be read into RAM | the pinned SDK headers: no `sys/mman.h`, no map API |
| how the weights are read | on every job that finds a line. The read goes through the kernel broker, because the game cannot see `ux0:`, in 8 KiB round trips with a 1 ms sleep each. SHA-256 is checked every time | [M/core/vocr_ocr.c#L192-L196](https://github.com/bee-san/vita-jp-overlay/blob/1a118ef8ca794bb5e2b5b2a96cd2cd07552d93eb/core/vocr_ocr.c#L192-L196), [#L144-L156](https://github.com/bee-san/vita-jp-overlay/blob/1a118ef8ca794bb5e2b5b2a96cd2cd07552d93eb/core/vocr_ocr.c#L144-L156); [M/game_ocr/main.c#L50-L67](https://github.com/bee-san/vita-jp-overlay/blob/1a118ef8ca794bb5e2b5b2a96cd2cd07552d93eb/game_ocr/main.c#L50-L67); [M/kernel/game_ocr.c#L405-L436](https://github.com/bee-san/vita-jp-overlay/blob/1a118ef8ca794bb5e2b5b2a96cd2cd07552d93eb/kernel/game_ocr.c#L405-L436) |
| stack | the worker thread has 32 KiB. The runtime and both line finders use 4,316 B of it (measured in Vita3K, section 3) | [M/game_ocr/main.c#L7](https://github.com/bee-san/vita-jp-overlay/blob/1a118ef8ca794bb5e2b5b2a96cd2cd07552d93eb/game_ocr/main.c#L7) |
| cores | user threads run on cores 0-2; core 3 is reserved for the system. The worker inherits the game's affinity and runs at the process's default priority (`0x10000100`) | vita-headers [cpu.h#L17-L23](https://github.com/vitasdk/vita-headers/blob/e66ebe90b73d1fa4cce005a5b2072cec27322544/include/psp2/kernel/cpu.h#L17-L23); [M/game_ocr/main.c#L202](https://github.com/bee-san/vita-jp-overlay/blob/1a118ef8ca794bb5e2b5b2a96cd2cd07552d93eb/game_ocr/main.c#L202) |
| clock | 333 MHz by default (the shell's default; a game may ask for more). `scePowerSetArmClockFrequency` allows up to 444 MHz and works from a user plugin inside a game; oclockvita does it. 500 MHz needs kernel code. One clock drives all four cores. The overlay never changes it | [ScePower](https://wiki.henkaku.xyz/index.php?title=ScePower&oldid=21824), [Pervasive](https://wiki.henkaku.xyz/index.php?title=Pervasive&oldid=21701), [oclockvita.c](https://github.com/frangarcj/oclockvita/blob/63aa0442685a45c99bee433c6242032008aec740/oclockvita.c#L20-L21), PSVshell [main.c#L106-L123](https://github.com/Electry/PSVshell/blob/9a47b0eb22c60bdbddd8c7b5c9f1218c7e8a3037/src/main.c#L106-L123) |
| caches | 32 KiB L1 I and 32 KiB L1 D per core, 32-byte lines. One **2 MiB, 16-way PL310 L2 is shared by all four cores**, with pseudo-random replacement and no L2 data prefetch. The "1 MB L2" sometimes quoted is 7-cpu.com's Exynos 4210, not the Vita | [Main Processor](https://wiki.henkaku.xyz/index.php?title=Main_Processor&oldid=21601) ("a L2 cache of 2 MiB shared by all cores"); [Caches](https://wiki.henkaku.xyz/index.php?title=Caches&oldid=21009): AUX_CTRL `0x60090001` = 16 ways × 128 KB, bits 25 and 28 clear ([Linux cache-l2x0.h#L125-L129](https://github.com/torvalds/linux/blob/v6.7/arch/arm/include/asm/hardware/cache-l2x0.h#L125-L129)) |

**Why the game process is the right place.**
- **SceShell plugin.** Correction: the pool that runs out is SceShell's 50 MiB partition, measured
  at 28-98 KiB free beside a running game, from which every page of a `*main` plugin (module,
  stacks, TLS, `USER_RW` blocks) is taken. The 5 MiB PAF heap, with about 2.4 MiB free beside
  CLANNAD and as little as 0.19 MiB later, is a pre-mapped sub-pool inside it; exhausting the heap
  starves the UI, exhausting the partition starves firmware (#59). A Shell plugin's whole budget is
  genuinely under 2 MiB and SceShell wants that headroom itself (the expert's rule). That is why
  ncnn, Meiki and VOCR all moved out of it, and why nothing of the recognizer may return.
- **Kernel plugin.** It would share the 36 MiB kernel-driver pool (`ScePhyMemPartKD`), whose free
  space is unknown. A 2 MiB full-frame capture buffer has already failed there once (section 2).
- **Separate homebrew app.** It would get a game-sized budget, but only one game-budget app runs
  at a time, so it cannot OCR a game that is running. This is an inference; no source states it
  as a rule.

**What one job allocates.** The workspace is one block
([M/core/vocr_ocr.c#L64-L73](https://github.com/bee-san/vita-jp-overlay/blob/1a118ef8ca794bb5e2b5b2a96cd2cd07552d93eb/core/vocr_ocr.c#L64-L73)):

> weights + max(line-finder arena, recognizer arena) + 3,760 B of runtime structs + the grey region (width × height), each part rounded up to 16 B

The finder's arena (83,323 B at 960×544) is smaller than every recognizer arena, so it adds nothing.

| model | 917×125 region (CLANNAD's box) | 960×544, full screen | free `USER` the preflight needs at 917×125 |
|---|---|---|---|
| FL10 | 900,896 B (logged on the console) | 1,308,496 B | 5.1 MB |
| H15 | 1,457,312 B | 1,864,912 B | 5.7 MB |
| F20 | 1,917,472 B | 2,325,072 B | 6.1 MB |
| width 3.0 | 3,155,744 B | 3,563,344 B | 7.4 MB |

The kernel's capture buffer (region × 4 B: 458,500 B at 917×125, allocated as 458,752 B in 4 KiB
pages) comes from the kernel-driver pool (`ScePhyMemPartKD`, 36 MiB) and is the same for every
model.

## 2. Why the overlay's OCR kept failing

None of the failures was the recognizer's size. The full report, with every link and error code,
is `$HERD/shared/artifacts/vita-budget/research/overlay-failures.md`; the device evidence is in
`notes/`.

| when | engine, where it ran | what failed | class (memory pool) | fix |
|---|---|---|---|---|
| 10-08 | ncnn PP-OCRv5 mobile, SceShell | Needed a 64 MiB block (32-42 MiB used; 13.4 MiB even trimmed to int8) against a 5 MiB Shell PAF heap. Never ran on a console | out of memory by design (SceShell) | dropped ([#18](https://github.com/bee-san/vita-jp-overlay/pull/18), [#19](https://github.com/bee-san/vita-jp-overlay/pull/19)) |
| 10-10 | none (the overlay's result arenas), SceShell | An eager 786,432 B result allocation failed in CLANNAD with `0x80024302`, "no free physical page" | out of memory: SceShell's 50 MiB partition had no free page for a `USER_RW` block (correction: the partition, not the PAF heap) | lazy, smaller arenas ([#11](https://github.com/bee-san/vita-jp-overlay/pull/11)); checked on the console (E07, E31) |
| 10-10 | Google Lens, kernel capture | The 2,088,960 B full-frame `KERNEL_RW` capture buffer failed with `0x80024302`. The UI said "unexpected response (-108)" | out of memory (kernel-driver pool `ScePhyMemPartKD`), plus error mapping | region-sized capture ([#14](https://github.com/bee-san/vita-jp-overlay/pull/14)); error encoding ([#27](https://github.com/bee-san/vita-jp-overlay/pull/27)) |
| 10-10 | Meiki, SceShell | Its 5,392,856 B model alone is larger than the whole 5,242,880 B Shell PAF heap | out of memory (SceShell PAF) | inference moved into the game |
| 10-11 | VOCR FL10/H15, SceShell | Moved out before any console run: about 2.4 MiB PAF free minus the 1.5 MiB reserve is less than H15's 1.4 MiB workspace | out of memory, predicted | game-process worker ([#21](https://github.com/bee-san/vita-jp-overlay/pull/21)) |
| 10-11 | VOCR FL10, game | The game cannot open `ux0:` (`0x80010013`). The UI told the user to re-download the weights | integration (file namespace) | model read through the kernel ([#24](https://github.com/bee-san/vita-jp-overlay/pull/24), [#25](https://github.com/bee-san/vita-jp-overlay/pull/25)); checked |
| 10-11 | VOCR FL10, game | The kernel's BUSY reached the game as `0xBFFFFFFD`, with bit 30 cleared. The game kept ownership and blocked every later capture | ABI and lifetime | [#27](https://github.com/bee-san/vita-jp-overlay/pull/27); 15 jobs with rc=0 on the console (HW02) |
| 10-11 | VOCR FL10 in the game, dictionary in SceShell | Six OCR jobs succeeded, then dictionary arenas failed at 355-441 KB of PAF free. A test tool had left a 2,187,264 B JPEG pool allocated | out of memory (SceShell PAF, leaked by an outside tool) | reboot; one 64 KiB arena ([#30](https://github.com/bee-san/vita-jp-overlay/pull/30)); checked (HW15) |
| 10-11 | VOCR FL10, game | **Works**: 14 text-bearing jobs with rc=0. The user then reported poor OCR quality | poor output (qualitative) | led to the Meiki trial |
| 10-11 | Meiki, game | Fitted easily (21.4 MB requested, 91 MB free), but took 7.9-8.0 s per line and needed a copy of its engine in every game | too slow; packaging | removed ([#50](https://github.com/bee-san/vita-jp-overlay/pull/50)) |
| 10-11 | VOCR, dictionary in SceShell | Shell PAF free fell to 190 KiB, and a static 64 KiB buffer in the Shell plugin hung the boot | out of memory (SceShell PAF, cause unknown); hang: correction, the 65,544 B of `.bss` are 16 pages of SceShell's 50 MiB partition, the same pool that failed in #59 (PR #60) | [issue #42](https://github.com/bee-san/vita-jp-overlay/issues/42), closed unfixed |

VOCR2 and PP-OCRv6 small have never run on a console. PP-OCRv6 small
([#52](https://github.com/bee-san/vita-jp-overlay/pull/52), open) needs 23.7 MB of free game memory
for a full-screen job and an estimated 3-10 s per line.

**Still on main.** Problems that cost time or hide evidence (details and fixes in
[#53](https://github.com/bee-san/vita-jp-overlay/issues/53)):
1. **The weights are reloaded and re-hashed on every job.** This is the cost that grows with model
   size (section 5, item 1).
2. **A refused preflight hides its numbers.** It is logged as `workspace=0`, without the free
   `USER` bytes, which are exactly the per-game figure a memory budget needs. The CTC frame count
   is not logged either; with it, every console job would measure ms per frame.
3. **Every out-of-memory error at the OCR stage gives the same advice.** It suggests a tighter
   region and FL10, even when the kernel capture failed and the model is irrelevant
   ([M/shell/worker.c#L580](https://github.com/bee-san/vita-jp-overlay/blob/1a118ef8ca794bb5e2b5b2a96cd2cd07552d93eb/shell/worker.c#L580)).

**Two corrections to earlier statements:**
- **The overlay's device default is FL10, not H15.**
  - `VJO_VOCR_DEFAULT_MODEL "H15_w8.vocr"`
    ([M/core/vocr_ocr.h#L24](https://github.com/bee-san/vita-jp-overlay/blob/1a118ef8ca794bb5e2b5b2a96cd2cd07552d93eb/core/vocr_ocr.h#L24))
    is only the host tools' default.
  - The config default and the release package both set `FL10_w8.vocr`
    ([M/core/config.c#L93](https://github.com/bee-san/vita-jp-overlay/blob/1a118ef8ca794bb5e2b5b2a96cd2cd07552d93eb/core/config.c#L93)).
  - Every console log is FL10. **H15 has never run on a console.**
- **FL10 has run on a console:** a PCH-1001 on firmware 3.65, inside CLANNAD (`notes/26`, HW02 and
  HW15).
  - Per job: model load 213-359 ms and recognition 36-145 ms over 14 text-bearing jobs in two
    sessions; line finding 34-38 ms in the 7 jobs of the first session (HW02).
  - These are logged samples of varying dialogue at an unknown clock.
  - Line widths were not logged, so the logs cannot test the estimates below. They hint that the
    high bounds are pessimistic: if any of those lines filled the 917-px box, 145 ms would sit near
    the low bound at 444 MHz. A full line is about 110 frames, an estimated 128-243 ms at 444 MHz.

## 3. What a bigger model costs

Sizes are measured on the host; the A9 times are estimated.

**Method** (`vita/budget_sweep.py`; artifacts in `$HERD/shared/artifacts/vita-budget/`):
- **Models.** 23 configurations of `train/model.py`, with random weights and random BatchNorm
  statistics, exported as int8 blobs. The released FL10, H15 and F20 blobs are measured as well.
- **Sizes** are exact: the blob bytes and `export.plan_arena()`, which the parity tests hold equal
  to `vocr_arena_bytes()`.
- **The A9 time** is `vita/a9_cycles.py`'s estimate: the Vita compiler's machine code for
  `runtime/vocr.c` run under plugin qemu-arm, with qemu's cache simulator and the Cortex-A9 cost
  model.
  - Input: the first 40 synth-val lines (2,133 CTC frames). No protected test line was read.
  - Only the L2's size, associativity and replacement policy and the runtime's tiling are changed.

**Checks:**
- Unmodified `a9_cycles.py` on the same 40 lines gives FL10's range to the printed precision
  (517,567-979,622 cycles per frame; L2 misses 18.948 against 18.946 per frame).
- Random and trained weights of the same shape agree to within 0.005%: FL10, H15 and F20 against
  `w100v3500`, `w150` and `w200`. The kernels have no data-dependent branches, so only the shapes
  matter.
- The published figures (20 synth-val plus 20 vita-test lines) are 0.8% higher for FL10 (521,887
  cycles per frame), because the line mix differs.

**Conventions:**
- The range is the cost model's (`vita/README.md`). The low bound assumes perfect overlap. The
  high bound assumes no dual issue and every cache miss at full latency.
- Defaults unless stated: tile 4, head batch 16, a 2 MiB 8-way LRU L2 (`a9_cycles.py`'s setup).

| config | params | int8 blob B | of which head B | W8I arena B | MACs/frame | est. cycles/frame (444 MHz) | cycles/MAC | est. ms, 640-px line, 444 MHz | 333 MHz |
|---|---|---|---|---|---|---|---|---|---|
| w075 | 659,739 | 751,616 | 557,440 | 95,776 | 801,072 | 517,907-975,928 | 0.647-1.218 | 93-176 | 124-234 |
| w100 (v1) | 759,499 | 859,904 | 557,440 | 124,832 | 1,002,720 | 619,966-1,166,946 | 0.618-1.164 | 112-210 | 149-280 |
| w100v3500 (FL10's shape) | 579,661 | 657,616 | 364,416 | 124,832 | 824,736 | 517,566-979,582 | 0.628-1.188 | 93-177 | 124-235 |
| w150 (H15's shape) | 1,038,507 | 1,155,968 | 557,440 | 182,944 | 1,564,992 | 912,779-1,657,448 | 0.583-1.059 | 164-299 | 219-398 |
| w200 (F20's shape) | 1,423,499 | 1,558,016 | 557,440 | 241,056 | 2,339,232 | 1,296,080-2,282,260 | 0.554-0.976 | 234-411 | 311-548 |
| w250 | 1,914,475 | 2,066,048 | 557,440 | 299,168 | 3,325,440 | 1,767,981-3,331,051 | 0.532-1.002 | 319-600 | 425-789 |
| w300 | 2,511,435 | 2,680,064 | 557,440 | 357,280 | 4,523,616 | 2,332,886-5,357,762 | 0.516-1.184 | 420-965 | 560-1234 |
| w400 | 4,023,307 | 4,226,048 | 557,440 | 473,504 | 7,555,872 | 3,730,489-9,018,492 | 0.494-1.194 | 672-1625 | 896-2068 |
| w150b48 | 767,643 | 884,672 | 300,160 | 179,296 | 1,294,128 | 794,535-1,511,183 | 0.614-1.168 | 143-272 | 191-363 |
| w150b64 | 857,931 | 975,104 | 385,920 | 180,512 | 1,384,416 | 833,949-1,559,203 | 0.602-1.126 | 150-281 | 200-375 |
| w150b128 | 1,219,083 | 1,336,832 | 728,960 | 185,376 | 1,745,568 | 991,609-1,754,813 | 0.568-1.005 | 179-316 | 238-421 |
| w150b192 | 1,580,235 | 1,698,560 | 1,072,000 | 190,240 | 2,106,720 | 1,149,268-1,949,529 | 0.546-0.925 | 207-351 | 276-468 |
| w150v2500 | 761,669 | 844,672 | 260,416 | 182,944 | 1,291,008 | 755,201-1,369,092 | 0.585-1.060 | 136-247 | 181-329 |
| w150v3500 | 858,669 | 953,680 | 364,416 | 182,944 | 1,387,008 | 810,379-1,470,059 | 0.584-1.060 | 146-265 | 195-353 |
| w150v7000 (about JIS X 0208) | 1,198,169 | 1,335,600 | 728,832 | 182,944 | 1,723,008 | 1,003,702-1,823,790 | 0.583-1.058 | 181-329 | 241-438 |
| w150v11000 (about JIS X 0213) | 1,586,169 | 1,771,600 | 1,144,832 | 182,944 | 2,107,008 | 1,224,413-2,229,412 | 0.581-1.058 | 221-402 | 294-535 |
| w150d2 | 1,147,995 | 1,285,376 | 557,440 | 240,544 | 2,000,160 | 1,182,793-2,115,699 | 0.591-1.058 | 213-381 | 284-508 |
| w150d3 | 1,257,483 | 1,414,784 | 557,440 | 304,288 | 2,435,328 | 1,445,410-2,575,176 | 0.594-1.057 | 260-464 | 347-619 |
| w150c4 | 1,208,427 | 1,340,544 | 557,440 | 201,376 | 1,733,760 | 1,004,562-1,836,314 | 0.579-1.059 | 181-331 | 241-441 |
| w200b192 | 1,974,443 | 2,109,824 | 1,072,000 | 248,352 | 2,890,176 | 1,536,568-2,886,326 | 0.532-0.999 | 277-520 | 369-682 |
| w200h24 | 1,423,499 | 1,558,016 | 557,440 | 192,640 | 2,123,040 | 1,192,188-2,097,564 | 0.562-0.988 | 215-378 | 286-504 |
| w300b192 | 3,080,811 | 3,250,304 | 1,072,000 | 364,576 | 5,092,992 | 2,581,643-5,817,586 | 0.507-1.142 | 465-1048 | 620-1339 |
| w300v11000 | 3,059,097 | 3,295,696 | 1,144,832 | 357,280 | 5,065,632 | 2,644,521-6,034,276 | 0.522-1.191 | 476-1087 | 635-1393 |
| FL10 (released) | 579,661 | 657,664 | 364,416 | 124,832 | 824,736 | 517,567-979,622 | 0.628-1.188 | 93-177 | 124-235 |
| H15 (released) | 1,038,507 | 1,155,968 | 557,440 | 182,944 | 1,564,992 | 912,779-1,657,479 | 0.583-1.059 | 164-299 | 219-398 |
| F20 (released) | 1,423,499 | 1,558,016 | 557,440 | 241,056 | 2,339,232 | 1,296,081-2,282,290 | 0.554-0.976 | 234-411 | 311-548 |

How to read the names:
- `wW` is width W/100;
- `bB` is the head bottleneck (96 otherwise);
- `vV` is the number of classes (5,355 otherwise);
- `dD` is the number of stride-1 blocks per stage;
- `c4` has four temporal blocks;
- `h24` takes input height 24.

A 640-px line at height 32 is about the p90 real Vita line. The median is about 300 px, so a median
line costs about half the figures above.

What the table says:
- **Time follows MACs.** Below the L2 knee, cycles per MAC fall slowly as the model widens, from
  0.63-1.19 at FL10's size to 0.55-0.98 at F20's: NEON stays busier and the per-line overhead is
  spread over more work. Width, depth, temporal blocks and head size cost about the same per MAC, so
  `train/cost.py`'s MACs per frame is a good proxy up to F20's size.
- **The head is half of an H15 blob** (557,440 of 1,155,968 B) and a third of its MACs.
  - A bottleneck of 64 instead of 96 saves 181 KB, 12% of the MACs and 6-9% of the estimated time
    at width 1.5.
  - A vocab the size of JIS X 0208 (7,000 classes) costs 16% more blob and 10% more time.
  - One the size of JIS X 0213 (11,000 classes) costs 53% more blob and 34% more time.
- **The footprint grows slowly.** The W8I arena stays at 95-473 KB everywhere. Up to width 2.0 with
  the default 96-wide bottleneck, the blob stays at 0.66-1.77 MB.

### The L2: where the cost turns superlinear

All four cores share the L2, so a running game can hold part of it. Running the same estimate with
smaller simulated L2 sizes stands in for the share the game leaves:

| config | blob B | weight B read per frame | L2 2 MiB: L2 misses/frame; est. ms 333 MHz | 1 MiB | 512 KiB | 256 KiB |
|---|---|---|---|---|---|---|
| FL10 | 657,664 | 91,416 | 19; 124-235 | 19; 124-235 | 1,607; 124-261 | 3,940; 124-298 |
| w100 | 859,904 | 103,480 | 22; 149-280 | 39; 149-281 | 2,120; 149-314 | 4,375; 149-350 |
| H15 | 1,155,968 | 177,496 | 27; 219-398 | 2,514; 219-438 | 7,126; 219-512 | 7,628; 219-520 |
| w150b192 | 1,698,560 | 216,664 | 41; 276-468 | 4,099; 276-533 | 8,539; 276-604 | 9,020; 276-612 |
| w150v11000 | 1,771,600 | 214,208 | 58; 294-535 | 4,188; 294-602 | 8,405; 294-669 | 8,930; 294-677 |
| F20 | 1,558,016 | 278,008 | 34; 311-548 | 8,742; 311-688 | 11,000; 311-724 | 11,619; 311-734 |
| w250 | 2,066,048 | 405,016 | 3,832; 425-789 | 15,251; 425-972 | 16,097; 425-985 | 16,656; 425-994 |
| w300 | 2,680,064 | 558,520 | 17,958; 560-1234 | 20,791; 560-1280 | 21,837; 560-1297 | 22,979; 560-1315 |
| w300b192 | 3,250,304 | 604,600 | 19,837; 620-1339 | 22,475; 620-1382 | 23,497; 620-1398 | 24,632; 620-1416 |
| w400 | 4,226,048 | 945,016 | 33,505; 896-2068 | 35,340; 896-2098 | 36,299; 896-2113 | 38,771; 896-2153 |

"Weight B read per frame" counts each backbone op's weights once per 4-frame tile and the head's
once per 16-frame batch.

- **The knee sits where weights plus arena reach the L2's size.**
  - Up to 1.95 MB of blob plus W8I arena (`w150v11000`), a model misses the L2 at most 58 times per
    frame: after the first tile, the weights stay in the L2.
  - At 2.07 + 0.30 MB (`w250`) the misses jump to 3,832 per frame, and to 17,958 at width 3.0.
  - With a 1 MiB share of the L2, H15 (1.34 MB) already spills; with 512 KiB, every model does.
- **Superlinear, but mildly.**
  - A spill never moves the low bound, which assumes the DRAM traffic overlaps the NEON work.
  - It raises the high bound at 333 MHz by 10% (H15) and 26% (F20) with a 1 MiB share.
  - With the whole L2, L2 misses add 8-10% to width 2.5's high bound, 30-36% to width 3.0's and
    35-41% to width 4.0's (333-444 MHz). Without them, those bounds would follow the trend of the smaller
    models (0.84-0.91 cycles per MAC).
  - The weights stream in order and every tile reuses them, so a spill costs bandwidth, not a cliff.
- **The bandwidth bill.**
  - F20 with a 1 MiB share pulls 8,742 × 32 B per frame: about its 278,008 B of weight reads, every
    pass from DRAM. That is 22 MB per 640-px line.
  - At 0.5-1 GB/s it takes 22-45 ms, 7-14% of F20's low bound at 333 MHz. That bandwidth range is an
    unmeasured working figure for one A9 core: no Vita memory benchmark exists.
  - The cost model's DRAM latency (110 ns) and L2-to-L1 rate (8.7 cycles per line) are 7-cpu.com's
    Exynos 4210 figures.
- **L1 misses are a steady share of the high bound.**
  - Weights pass through the 32 KiB L1 once per tile, so L1 misses grow with the weight bytes read
    per frame: 5,836 per frame for FL10, 10,513 for H15, 15,968 for F20, 31,043 at width 3.0.
  - At 30 cycles each, they make up 12-22% of the high bound across the grid (18-21% for FL10, H15
    and F20). That is part of why the high bound sits about twice the low bound everywhere.

The cache model itself (default tiling):

| config | L2 | 8-way LRU (`a9_cycles.py`): L2 misses/frame; est. ms 333 MHz | 16-way LRU |
|---|---|---|---|
| H15 | 2 MiB | 27; 219-398 | 27; 219-398 |
| H15 | 1 MiB | 2,514; 219-438 | 2,530; 219-438 |
| F20 | 2 MiB | 34; 311-548 | 34; 311-548 |
| F20 | 1 MiB | 8,742; 311-688 | 9,744; 311-704 |
| w250 | 2 MiB | 3,832; 425-789 | 4,311; 425-797 |
| w250 | 1 MiB | 15,251; 425-972 | 15,300; 425-973 |
| w300 | 2 MiB | 17,958; 560-1234 | 18,290; 560-1240 |
| w300 | 1 MiB | 20,791; 560-1280 | 20,844; 560-1281 |

- **Associativity changes little.** The simulator's L2 has 8 ways and the Vita's PL310 has 16.
  With LRU replacement, 16 ways leave the misses within 1% of the 8-way figures for models that fit,
  and up to 12-13% more for models that spill (w250 with the whole L2: 4,311 against 3,832; F20
  with a 1 MiB share: 9,744 against 8,742). The estimated ms moves by at most about 3%. A weight stream larger than the cache thrashes LRU at any
  associativity.
- **Replacement is the bigger unknown, and is not simulated here.** The PL310 replaces lines
  pseudo-randomly (AUX_CTRL bit 25 clear), while every run above uses qemu's default, LRU.
  *Inference, not measured:* a cyclic stream slightly larger than the cache makes LRU miss on every
  line, while random replacement keeps a share of them, so LRU should be the pessimistic case for
  models just past the knee. `budget_sweep.py a9 --evict rand` runs the comparison.

**Bigger tiles keep more of the hot set in the L2.** The runtime's tile and head batch trade arena
for passes over the weights:

| config | L2 | tile / head batch | W8I arena B | weight B read per frame | L2 misses/frame | est. ms 333 MHz | at tile 4 / head 16 |
|---|---|---|---|---|---|---|---|
| H15 | 2 MiB | 4 / 64 | 201,760 | 151,366 | 27 | 219-392 | 27; 219-398 |
| H15 | 2 MiB | 16 / 64 | 589,696 | 44,374 | 33 | 218-377 | 27; 219-398 |
| H15 | 512 KiB | 4 / 64 | 201,760 | 151,366 | 6,230 | 219-491 | 7,126; 219-512 |
| H15 | 512 KiB | 16 / 64 | 589,696 | 44,374 | 3,307 | 218-429 | 7,126; 219-512 |
| F20 | 2 MiB | 4 / 64 | 259,872 | 251,878 | 34 | 311-543 | 34; 311-548 |
| F20 | 2 MiB | 16 / 64 | 775,296 | 69,502 | 1,036 | 309-537 | 34; 311-548 |
| F20 | 512 KiB | 4 / 64 | 259,872 | 251,878 | 10,132 | 311-704 | 11,000; 311-724 |
| F20 | 512 KiB | 16 / 64 | 775,296 | 69,502 | 5,123 | 309-603 | 11,000; 311-724 |
| w300 | 2 MiB | 16 / 64 | 1,146,496 | 139,630 | 7,847 | 557-1088 | 17,958; 560-1234 |
| w300 | 512 KiB | 16 / 64 | 1,146,496 | 139,630 | 9,837 | 557-1119 | 21,837; 560-1297 |
| w400 | 2 MiB | 16 / 64 | 1,517,696 | 236,254 | 13,030 | 891-1846 | 33,505; 896-2068 |
| w400 | 512 KiB | 16 / 64 | 1,517,696 | 236,254 | 16,703 | 891-1905 | 36,299; 896-2113 |

- **A model that fits the L2** gains 1-5% from bigger tiles.
- **A model that spills** gets 2.1-2.6× fewer L2 misses from 16-frame tiles and 64-frame head
  batches, and a 10-17% lower high bound. The cost is 0.4-1.0 MB more arena.
- **A bigger head batch alone** (4 / 64) costs 19 KB of arena and buys only 1-4%.
- The default tiling stays right for models under the ceiling.

### Vita3K and code size

The Vita build runs the larger blobs bit for bit:
- tool: `vita/scripts/run_vita3k.sh`, Vita3K build 4154;
- app: the VPK built from this PR's base (`vita/scripts/build.sh`);
- input: 100 synth-val crops per blob, `models=w8i`.

| blob | bytes | arena on the Vita (= host) | classes | result | runtime stack (probe thread) |
|---|---|---|---|---|---|
| `w150v11000` | 1,771,600 | 182,944 B | 11,001 | 100 of 100 MATCH | 4,316 B of 32,768 |
| `w300` | 2,680,064 | 357,280 B | 5,355 | 100 of 100 MATCH | 4,316 B |
| `w400` | 4,226,048 | 473,504 B | 5,355 | 100 of 100 MATCH | 4,316 B |

- **MATCH** means the same text and the same CRC-32 of every logit as the host build.
- The stack does not depend on the model.
- Emulator times are x86 JIT times; they say nothing about a console.
- This box has no Xvfb, so Vita3K ran under SDL's offscreen driver, with `libOpenGL.so.0` unpacked
  privately from Amazon Linux's `libglvnd-opengl` 1.7.0 package.

Code size, VitaSDK gcc 15.2 `-O2` Thumb-2 at `2873b98`: `vocr.o` is 14,583 B with NEON and
`vocr_lines.o` is 12,456 B. Neither depends on the model.

## 4. What one OCR job costs on the console

A job runs capture, line finding, model load, then recognition of each detected line, one after
another on one core.

| model | line finding (FL10, measured) | model load, today's loader | est. recognition, 3 lines of 640 px, 333 MHz | total today (est.) | est. without the reload, at 444 MHz |
|---|---|---|---|---|---|
| FL10 | 34-38 ms | 213-359 ms (measured) | 0.37-0.71 s | 0.62-1.10 s | 0.31-0.57 s |
| H15 | same | 0.37-0.63 s (est.) | 0.66-1.19 s | 1.07-1.86 s | 0.53-0.93 s |
| F20 | same | 0.50-0.85 s (est.) | 0.93-1.64 s | 1.47-2.53 s | 0.74-1.27 s |
| width 3.0 | same | 0.87-1.46 s (est.) | 1.68-3.70 s | 2.58-5.20 s | 1.29-2.93 s |

**Load-time estimate.** It scales FL10's measured 213-359 ms by blob bytes. That is plausible,
because every part of the load grows with the file. Estimated breakdown for FL10:

| part | for FL10 | how it was estimated |
|---|---|---|
| sleeps | at least 81 ms | 81 kernel round trips of 8 KiB, at least 1 ms each |
| card read | about 86 ms | the official memory card at about 7.6 MB/s, a Digital Foundry figure quoted second-hand |
| SHA-256 over the whole file | 40-60 ms | 20-30 cycles per byte at 333 MHz; not measured |
| **total** | **about 210-230 ms** | |

That total matches the low end of the measured 213-359 ms. A median three-line box takes about half
the recognition column.

## 5. Buying capacity cheaply

In order of value per effort. Items 1-3 are overlay changes; nothing in `runtime/` has to change.

1. **Stop reloading the weights on every job** (vita-jp-overlay
   [#53](https://github.com/bee-san/vita-jp-overlay/issues/53); done in #54: the verified weights
   now stay resident in their own `USER_RW` block in the game, 659,456 B for FL10 and 1,159,168 B
   for H15 page-rounded, and the kernel stage is 32 KiB with a wait instead of a 1 ms sleep).
   Any one of these:
   - wait on the kernel's completion instead of a 1 ms sleep loop, or raise the 8 KiB kernel stage
     to the game's 32 KiB read size;
   - hash once per session;
   - keep the verified weights in their own `USER_RW` block while the title is enabled and in the
     foreground (657,664 B for FL10, 1,155,968 B for H15), under the same preflight.

   This saves 0.2-0.36 s per FL10 job today and an estimated 0.50-0.85 s per F20 job. Resident
   weights hold 0.7-1.6 MB of the game's memory for the session instead of per job. That is small
   next to the 56-87 MiB seen free.
2. **Ask for 444 MHz during a job.** Call `scePowerSetArmClockFrequency(444)`, then restore the
   game's clock afterwards.
   - Compute-bound work gets 1.33× faster, which pays for a model a third bigger at the same
     latency. It costs battery while the job runs.
   - Unknown: whether the L2 then runs at 222 or 444 MHz (the Pervasive table lists both).
3. **Use a second core for multi-line boxes.** Run one line per thread on cores 1 and 2, below the
   game's priority.
   - The runtime allows this as it is: one context and arena per thread, one shared model.
   - A box of two or more lines then takes about half as long, for one more arena (183 KB for H15)
     and 32 KiB of stack.
   - The two threads share one copy of the weights in the L2, but both arenas count against it.
     H15 with two arenas is 1.52 MB and fits. F20 with two is 2.04 MB, 97% of the L2, leaving no
     room for the game.
   - Not measured. In PSVshell's README screenshots of Stardew Valley, cores 1 and 2 were 0-3%
     busy.
4. **Spend MACs where they pay.** The architecture sweep found width to be the knob that helps on
   real data. The cheapest place to find MACs and bytes is the head:
   - Bottleneck 64 instead of 96 saves 181 KB and an estimated 6-9% of the time at width 1.5.
     Accuracy: bottleneck 48 at width 1.0 measured +0.26 points with real data (CI -0.40 to +0.92),
     not significantly worse (`docs/arch-sweep.md`, `b48_r`).
   - FL10's 3,500-class vocab saves 18% of a width-1.0 model's MACs.
   - Growing the vocab past 5,355 classes is expensive (above).
5. **Low-rank pointwise layers.** Two pointwise ops (`K→r→N`), as the head already factorises, cut
   MACs and bytes within the current op set. This needs training, and a check that `vocr_load()`
   accepts the op sequence.
6. **Bigger tiles,** but only for a model that does not fit the L2 (above).
7. **Not worth it now:**
   - **int4 or mixed precision.** The A9 has no int4 multiply-accumulate, the estimate is
     NEON-bound, and unpacking costs instructions. int4 would save only bytes, which are not scarce.
     It becomes worth it only to keep a much bigger model inside the L2.
   - **A shared or hierarchical vocab head** beyond the existing bottleneck. It needs new runtime
     ops, and the bottleneck and vocab knobs above already reach the head's cost.
   - **Weight streaming from storage.** RAM is not short, and every tile needs every backbone
     weight. Streaming layer by layer would need whole-line activations: 2.86 MB of arena for v1
     (`runtime/README.md`, tile 128).

## 6. What would turn these estimates into measurements

- **Run `vita/`'s test app on a PCH-1000** (`vita/README.md`, "On a PCH-1000"), with `clock=333`
  and `clock=444`. If ScePerf works on retail firmware, its counters replace the cycle model.
- **Add two numbers to the overlay's log** (#53): line frames with each job's recognition time, and
  the free `USER` bytes on every refusal. That gives on-console ms per frame and per-title memory
  budgets.
- **Measure free memory in the Japanese VNs that matter**, with PSVshell or
  `sceKernelGetFreeMemorySize` (`sizeMain`, inside the game). Two samples (CLANNAD, Stardew Valley)
  are all there is. Adrenaline, which runs PSP VNs, is unmeasured. This is the only number that can
  make RAM the limit for a model; SceShell's figures cannot (`vita-platform-memory.md`).
- **Still unknown:**
  - the game's share of the L2;
  - DRAM bandwidth and latency on a Vita;
  - the L2 clock at 444 MHz;
  - CLANNAD's ARM clock;
  - the free memory in the 36 MiB kernel-driver pool.

`$HERD/shared/artifacts/vita-budget/research/platform-facts.md` lists the cited platform facts with
their sources and the open questions.

## Reproduce

```sh
A=$HERD/shared/artifacts; D=$A/vita-budget/sweep
export VITASDK=$A/vita/vitasdk VOCR_SYNTH_VAL=$HERD/shared/data/synth/synth_00039.npz
export ARM_CC=$A/runtime/tools/arm-gnu-toolchain-14.2.rel1-x86_64-arm-none-linux-gnueabihf/bin/arm-none-linux-gnueabihf-gcc
python vita/budget_sweep.py blobs --out $D                       # 23 random-init blobs + FL10/H15/F20, sizes.json
S="python vita/budget_sweep.py a9 --dir $D --qemu $A/vita/qemu --jobs 5"   # through heavy.sh
$S --l2 2048                                                     # every config, about 7 min
$S --l2 1024,512,256 --only FL10,H15,F20,w100,w250,w300,w400,w150v11000,w150b192,w300b192
$S --l2 2048,512 --tile 16 --head 64 --only H15,F20,w300,w400
$S --l2 2048,512 --tile 4 --head 64 --only H15,F20
$S --l2 2048,1024 --l2-ways 16 --only H15,F20,w250,w300
python vita/budget_sweep.py table --dir $D                       # the tables above
python vita/budget_sweep.py check my_model_w8.vocr               # a trained blob against the ceiling
python vita/budget_sweep.py ckpt --name w300 --out w300.pt       # for vita/bundle.py --ckpt w300.pt
                                                                 #   --sets synth-val --modes w8i, then run_vita3k.sh
```

The scripts that ran these are `$A/vita-budget/run_sweep.sh` and `run_sweep2.sh`.
