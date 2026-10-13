# How big can the recognizer be on a PS Vita PCH-1000?

Written 2026-10-12 for the agents training the next recognizers. Every console speed here is an
**estimate** from a cycle model. No controlled timing has run on a console, and no engine has a
hardware CER.

*Revised 2026-10-13*, after the SDK documentation and vita-jp-overlay PR
[#60](https://github.com/bee-san/vita-jp-overlay/pull/60) were read: the budget is stated in three
tiers for a recognizer in the game process only; section 1 gains the SDK's game-budget rules, the
per-title free-memory table with the preflight rule, and the scheduler facts; and where the
overlay's memory comes from is corrected (marked "correction"): the recognizer's budget is the game
process's main memory only; SceShell's 50 MiB partition, not its 5 MiB PAF heap, is the pool that
runs out there; and `*ALL` loads the worker into SceShell too. The short platform note is
[`vita-platform-memory.md`](vita-platform-memory.md); the full reference is vita-jp-overlay's
[`docs/vita-memory.md`](https://github.com/bee-san/vita-jp-overlay/blob/main/docs/vita-memory.md).

## The answer

The recognizer runs in the game's own process and nowhere else (section 1). Its budget comes in
three tiers. Sizes are exact; console times are estimates. The fleet's `vita-budget.status`
lines, which `vita/budget_sweep.py check` prints after naming a blob's tier:

```
BUDGET_READY weights<=1600000 arena<=262144 ms<=550
BUDGET_TIERS always: weights<=900000 arena<=131072 ms<=300; preflight: weights<=1600000 arena<=262144 ms<=550; never: beyond
```

| | always safe | safe with the preflight (the ceiling, unchanged since #53) | never |
|---|---|---|---|
| int8 W8I blob, bytes | **≤ 900,000** | **≤ 1,600,000** | more |
| `vocr_arena_bytes()`, W8I, default tiling | **≤ 131,072** | **≤ 262,144** | more |
| blob + arena against the 2 MiB L2 the game shares | ≤ 1,031,072 B: fits beside a game that holds half the L2 | ≤ 1,862,144 B (89%): fits an idle L2, spills when the game holds half | past about 2.0 MB it spills even an idle L2 |
| game memory held between jobs (resident weights, 4 KiB pages) | ≤ 901,120 B | ≤ 1,601,536 B | |
| game memory at a job's peak (weights + workspace), 917×125 box / 960×544 full screen | ≤ 1.10 / 1.49 MiB | ≤ 1.89 / 2.28 MiB | |
| free `USER` the first full-screen job needs (peak + the 4 MiB reserve) | 5.5 MiB | 6.3 MiB | |
| est. ms per 640-px line, one A9 core at 333 MHz, high bound, whole L2 (`check`'s number) | **≤ 300** | **≤ 550** | more |
| the same with the game holding half (1 MiB) / three quarters (512 KiB) of the L2 | +0-1% / +12% (`w100`: 281 / 314 ms) | +26% / +32% (F20: 688 / 724 ms) | |
| a full-width 917-px CLANNAD line (about 115 frames), game holding half the L2 | about 0.43 s | about 1.0 s (F20) | |
| three 640-px lines, same share | about 0.9 s | about 2.1 s (F20) | |
| at 444 MHz | three quarters of the above | three quarters | |
| models (section 3) | FL10 (657,664 B, 124,832 B, est. 124-235 ms), v1's shape `w100` (859,904, 124,832, 149-280), `w075` | H15 (1,155,968, 182,944, 219-398), F20 (1,558,016, 241,056, 311-548), `w150b128`, `w150c4`, `w150d2`, `w200h24`, every width-1.5 variant with a 5,355-class head, bottleneck ≤ 128 and one block per stage | width 2.5 and up; `w200b192`, `w150b192` and `w150v11000` (blob bytes); `w150d3` (arena and ms) |

- **What the tiers mean.**
  - *Always safe* is the class of the fallback model. A job holds at most about 1.5 MiB of the
    game's memory for a few hundred ms and at most 0.9 MB between jobs. Blob and arena fit half
    of the L2, so the estimate hardly moves when the game holds the other half (`w100`: 149-280
    to 149-281 ms; FL10 unchanged), and a 917-px line costs about 0.43 s at the high bound.
    It still passes through the preflight, which is one syscall, but no measured or inferred game
    refuses it: the smallest free figure seen in a game is 56 MiB, nearly 40× its peak. A default
    model from this tier needs no fallback model.
  - *Safe with the preflight* is the ceiling. It needs what the overlay has had since
    [#54](https://github.com/bee-san/vita-jp-overlay/pull/54)-[#57](https://github.com/bee-san/vita-jp-overlay/pull/57):
    free `USER` minus the 4 MiB reserve must cover the job plus the weights unless they are
    resident, a refusal falls back to an always-safe model, and the resident weights go back to
    the game when it is not the active one. Its cost is speed, not safety: with the game holding
    half or three quarters of the L2, F20's high bound is 26-32% above `check`'s figure, about
    1.0 s for a full-width line and 2.1 s for a three-line box, before any measurement. The
    owner's goal was under about 1 s per line (`runtime/README.md`).
  - *Never* is anything that fails a ceiling limit, in any process. Also, at any size: nothing of
    the recognizer in SceShell (weights, arenas, dictionaries, caches); a heap or newlib in the
    game; a job that would need more than the 4 MiB reserve itself.
- **What the ms limit means.** It is the high end of `vita/a9_cycles.py`'s W8I range for an
  80-frame (640-px) line, which is about the p90 real Vita line. It is taken at 333 MHz, the clock
  games run at by default, with the whole 2 MiB L2. The two share rows restate it for a game that
  keeps half or three quarters of the L2 (section 3); the game's real share is unknown.
- **Checking a model.** All three limits of a tier must hold. `vita/budget_sweep.py check BLOB`
  tests a trained blob against the ceiling, names its tier and prints both status lines, reading
  synth-val lines only. Run it through `heavy.sh`.
- **Margin.** The figure is per 640-px line. For a model at the ceiling, an estimated:
  - full-width 917-px CLANNAD line (110-115 frames): about 760-790 ms at 333 MHz with the whole
    L2, about 1.0 s with half of it;
  - three-line box of 640-px lines: about 1.65 s with the whole L2, 2.1 s with half, after the
    one-time model load (section 4).

  The published line mix (20 synth-val plus 20 vita-test lines) costs about 0.8% more per frame
  than `check`'s synth-val lines, so leave a few ms of margin rather than designing to 549 ms.
- **Headroom.**
  - Memory: no budget-derived worst case exists (section 1); against the free memory measured in
    games (56-87 MiB) a ceiling job is under 5%, and the preflight refuses what does not fit.
  - L2: an always-safe model keeps blob plus arena under half of it; a ceiling model under 89%.
  - Speed: the ms limit is the pessimistic bound at the slower clock.

What the numbers say:

- **The budget is a game-process budget. SceShell has none to give.** SceShell's 50 MiB partition
  runs within tens of KiB of empty beside a game (28,672 B and 98,304 B free in two console
  samples), and the overlay's `*main` plugin plus the `*ALL` worker, which taiHEN also loads into
  SceShell, already hold 540,672 B of it permanently. A 36,864 B growth of that footprint between
  the build that boots (API 13) and the one that does not (API 16) is the most economical
  explanation of overlay issue [#59](https://github.com/bee-san/vita-jp-overlay/issues/59):
  SceShell's own `SceLiveAreaDailyCronJob` died with `0x80024302`, no free page, before any OCR
  job ran ([PR #60](https://github.com/bee-san/vita-jp-overlay/pull/60)). So every byte above is
  the game's, and nothing of the recognizer may sit in SceShell at any size
  ([`vita-platform-memory.md`](vita-platform-memory.md) has the pools in one table).
- **In the game, RAM does not set the ceiling; the preflight does.** One job of a ceiling-sized
  model needs 2.0-2.4 MB of the game's memory. Free memory seen: 87 MiB in CLANNAD (measured, a
  plain 256 MiB title), about 56 MiB in Stardew Valley (a third-party screenshot, a +109 MiB
  title). No SDK rule reserves memory for a plugin, and some games come close to their whole
  budget, so the safe allowance per game is a rule rather than a number (section 1).
- **Speed binds first, then the shared 2 MiB L2.** Up to F20's size, cost grows slightly less than
  linearly with MACs. The knee comes where weights plus arena outgrow the L2, past about 2.0 MB:
  - from there, the worst-case estimate grows faster than MACs: L2 misses add 8-10% to width 2.5's
    high bound and 30-36% to width 3.0's;
  - the best-case estimate stays linear;
  - a game that holds part of the L2 moves the knee down: with half of it, H15 already spills
    (+10% on the high bound) and F20 pays +26%; an always-safe model does not spill.
- **"More memory is slower" was the overlay's loader, not the cache, and #54 fixed it.**
  - Until [#54](https://github.com/bee-san/vita-jp-overlay/pull/54), vita-jp-overlay read and
    re-hashed the whole weights file on every OCR job, through 8 KiB kernel round trips with a
    1 ms sleep each. On the console that took 213-359 ms for FL10, more than the recognition
    itself (36-145 ms), and it grew with the blob.
  - Since #54 the verified weights stay resident in their own `USER_RW` block while the title is
    the active game; the first job of a session reads them in 32 KiB pieces without sleeps (an
    estimated 0.13-0.15 s for FL10, 0.23-0.26 s for H15, not measured), later jobs pay an open,
    a stat and a close. The one-time load still grows with the blob (section 4).
- **None of the overlay's OCR failures came from the recognizer's size.** The failures that stopped
  OCR on the console were integration problems (section 2): the wrong process, eager or full-frame
  buffers, the game's file namespace, an error-code ABI bug, a pool leaked by a test tool, and
  now the Shell plugin's own footprint (#59). The rest were output quality (FL10, judged by eye)
  and speed (Meiki, about 8 s per line).

A slightly bigger model than H15 is affordable, up to F20's size and cost, as long as the FL10-class
fallback stays. Shapes from the sweep that fit the ceiling, alongside plain width 2.0 (sizes
measured, times estimated at 333 MHz with the whole L2):
- two blocks per stage at width 1.5 (`w150d2`: 1.29 MB, 240,544 B, est. 284-508 ms);
- a larger head bottleneck (`w150b128`: 1.34 MB, 185,376 B, est. 238-421 ms);
- four temporal blocks (`w150c4`: 1.34 MB, 201,376 B, est. 241-441 ms).

Width 2.5 and above do not fit (width 2.5: 2.07 MB, est. 425-789 ms). This ceiling is for the Vita; the
PSP has no NEON and its own budget (`psp/README.md`).

## 1. Where the OCR runs, and which memory it uses

vita-jp-overlay main is at `599dd9c` (2026-10-13; `#54`-`#58` merged). The line links below, `M`,
point at `1a118ef`, the main this section was first written against:
`https://github.com/bee-san/vita-jp-overlay/blob/1a118ef8ca794bb5e2b5b2a96cd2cd07552d93eb`; the
facts that changed since are marked with their PR. The overlay has three parts:
- a kernel plugin (`*KERNEL`) that captures the screen into a region-sized `KERNEL_RW` buffer and
  brokers the model file;
- a SceShell plugin (`*main`) for the UI and the disk-backed dictionary;
- a worker plugin (`VitaJPOverlay_OCR.suprx`) registered under `*ALL`, which runs the line finder
  and the recognizer in the game's own process. Correction: `*ALL` loads it into SceShell and every
  system application as well, not only into enabled games (taiHEN matches the `ALL` section for
  every section but `KERNEL`); there its `module_start` finds no candidate title and starts
  nothing, but the module stays resident (45,056 B of SceShell's pages,
  [PR #60](https://github.com/bee-san/vita-jp-overlay/pull/60), section 3.3). The expert's rule,
  binding in the overlay's `AGENTS.md`, is one `*TITLEID` section per game; #60 plans that move.

Facts, each with its source:

| | value | source |
|---|---|---|
| RAM | 512 MiB LPDDR2 main memory, 128 MiB CDRAM | [Physical Memory](https://wiki.henkaku.xyz/index.php?title=Physical_Memory&oldid=21616) |
| the game's partitions | 256 MiB main (+29, +77 or +109 MiB when the title's `param.sfo` asks for it; fixed per title), 112 MiB CDRAM, 26 MiB phycont | SDK `Programming-Startup_Guide` ("Resources that Can Be Used by an Application"); [Memory budget](https://wiki.henkaku.xyz/index.php?title=Memory_budget&oldid=18995); Vita3K [SceSysmem.cpp#L171-L211](https://github.com/Vita3K/Vita3K/blob/7e152b66e85345f175d3d287fcbd7f8abec7574a/vita3k/modules/SceSysmem/SceSysmem.cpp#L171-L211) (`ATTRIBUTE2 & 0x0C`: 4, 8, 12) |
| SceShell, kernel drivers | SceShell: 50 MiB main (`ScePhyMemPartShell`) + 16 MiB CDRAM, shared by all of SceShell and every `*main`/`*ALL` plugin; measured free beside a running game: 28,672 B and 98,304 B. Kernel drivers and kernel plugins: 36 MiB (`ScePhyMemPartKD`), free size unmeasured | same Memory budget page; vita-jp-overlay PR #60; `notes/08`, `notes/26` |
| a game plugin's memory | taiHEN loads the plugin into the game's process (`ksceKernelLoadStartModuleForPid`), before the game's own code runs. Correction: its module pages, its thread's stack, at least 4 KiB of TLS and its memory blocks are charged to that process's budget, like the game's own (the SDK's Kernel Overview: the memory block service loads modules and allocates stacks; its Kernel Reference: stacks and TLS come out of the application's budget at thread creation) | taiHEN [plugin.c#L165-L173](https://github.com/yifanlu/taiHEN/blob/309b3800bcb8ebbd5e4f5e5e920af3da3590b829/plugin.c#L165-L173), [hen.c#L345-L380](https://github.com/yifanlu/taiHEN/blob/309b3800bcb8ebbd5e4f5e5e920af3da3590b829/hen.c#L345-L380); SDK `Kernel-Overview` (thread context), `Kernel-Reference` (`sceKernelCreateThread`); the same accounting in SceShell, measured (540,672 B): PR #60; vita-jp-overlay `docs/vita-memory.md` sections 3-4 |
| heap | none: the worker links with `-nostdlib`. It allocates one `USER_RW` memblock per job (4 KiB granularity) and frees it when the job ends; since #54 the verified weights have a second `USER_RW` block that stays while the title is the active game | [M/vita/CMakeLists.txt#L176](https://github.com/bee-san/vita-jp-overlay/blob/1a118ef8ca794bb5e2b5b2a96cd2cd07552d93eb/vita/CMakeLists.txt#L176); [`game_ocr/main.c`](https://github.com/bee-san/vita-jp-overlay/blob/599dd9cbb6f5c452ceadb10363eea065219d435e/game_ocr/main.c) at `599dd9c` (`weights_alloc`, `check_memory`, `cleanup`) |
| preflight | `sceKernelGetFreeMemorySize` (`sizeMain`, the game's own main-memory budget, minus a system margin, per the SDK) must report free `USER_RW` of at least the page-rounded workspace plus the page-rounded weights, unless they are already resident, plus a 4 MiB reserve, or the job is refused before anything is allocated; a refusal tries the configured fallback model when it is pinned and needs less, and is logged with the free bytes ([#55](https://github.com/bee-san/vita-jp-overlay/pull/55)) | [`game_ocr/main.c#L19`](https://github.com/bee-san/vita-jp-overlay/blob/599dd9cbb6f5c452ceadb10363eea065219d435e/game_ocr/main.c#L19), [`#L196-L205`](https://github.com/bee-san/vita-jp-overlay/blob/599dd9cbb6f5c452ceadb10363eea065219d435e/game_ocr/main.c#L196-L205), [`#L241-L244`](https://github.com/bee-san/vita-jp-overlay/blob/599dd9cbb6f5c452ceadb10363eea065219d435e/game_ocr/main.c#L241-L244); SDK `Kernel-Reference` (`sceKernelGetFreeMemorySize`) |
| free memory seen in games | CLANNAD (PCSG00415, a plain 256 MiB title: its `param.sfo` has `ATTRIBUTE2=0`): 91,226,112 B of free `USER` before a job (one logged sample). Stardew Valley in +109 MiB mode: 309 of 365 MB used, about 56 MiB free (a third-party PSVshell screenshot, not our measurement) | `notes/27`, `notes/26`; [PSVshell README](https://github.com/Electry/PSVshell#readme), whose [perf.c](https://github.com/Electry/PSVshell/blob/master/src/perf.c) reads the running process's own partition totals |
| free memory in SceShell beside a game | 98,304 B and 28,672 B of its 50 MiB (two samples, different days). The overlay's `*main` plugin and the `*ALL` worker hold 540,672 B of that partition permanently at API 16 | `notes/26`, `notes/08`; [PR #60](https://github.com/bee-san/vita-jp-overlay/pull/60) |
| file mapping | VitaSDK has no `mmap`, so the weights must be read into RAM | the pinned SDK headers: no `sys/mman.h`, no map API |
| how the weights are read | once per game session since [#54](https://github.com/bee-san/vita-jp-overlay/pull/54): through the kernel broker, because the game cannot see `ux0:`, in 32 KiB pieces with a bounded in-kernel wait instead of a 1 ms sleep, SHA-256 checked once, then kept in their own `USER_RW` block while the title is the active game (re-read when the file's size or time changes). Before #54: on every job that found a line, in 8 KiB round trips with a 1 ms sleep each | [`include/vjo_game_ocr.h#L13`](https://github.com/bee-san/vita-jp-overlay/blob/599dd9cbb6f5c452ceadb10363eea065219d435e/include/vjo_game_ocr.h#L13); before: [M/core/vocr_ocr.c#L192-L196](https://github.com/bee-san/vita-jp-overlay/blob/1a118ef8ca794bb5e2b5b2a96cd2cd07552d93eb/core/vocr_ocr.c#L192-L196), [M/game_ocr/main.c#L50-L67](https://github.com/bee-san/vita-jp-overlay/blob/1a118ef8ca794bb5e2b5b2a96cd2cd07552d93eb/game_ocr/main.c#L50-L67) |
| stack | the worker thread has 32 KiB (plus at least 4 KiB of TLS, from the game's budget). The runtime and both line finders use 4,316 B of it (measured in Vita3K, section 3) | [`game_ocr/main.c#L8`](https://github.com/bee-san/vita-jp-overlay/blob/599dd9cbb6f5c452ceadb10363eea065219d435e/game_ocr/main.c#L8) |
| cores and priority | a game may use three of the four A9 cores; the fourth is meant for kernel and system threads. The worker is created with the default priority and the default affinity mask, which for a game process mean priority 160 in the common ready queue and any of the three cores. The scheduler is fixed-priority and not time-sharing (below). The kernel plugin's worker thread runs at priority 60, above every game thread, pinned to core 0 | SDK `Programming-Startup_Guide` (CPU), `Kernel-Overview` (thread priority, CPU affinity mask, thread scheduling); vita-headers [cpu.h#L17-L23](https://github.com/vitasdk/vita-headers/blob/e66ebe90b73d1fa4cce005a5b2072cec27322544/include/psp2/kernel/cpu.h#L17-L23); [`game_ocr/main.c#L345`](https://github.com/bee-san/vita-jp-overlay/blob/599dd9cbb6f5c452ceadb10363eea065219d435e/game_ocr/main.c#L345); [`kernel/main.c#L484`](https://github.com/bee-san/vita-jp-overlay/blob/599dd9cbb6f5c452ceadb10363eea065219d435e/kernel/main.c#L484) |
| clock | 333 MHz by default (the shell's default; a game may ask for more). `scePowerSetArmClockFrequency` allows up to 444 MHz and works from a user plugin inside a game; oclockvita does it. 500 MHz needs kernel code. One clock drives all four cores. The overlay never changes it | [ScePower](https://wiki.henkaku.xyz/index.php?title=ScePower&oldid=21824), [Pervasive](https://wiki.henkaku.xyz/index.php?title=Pervasive&oldid=21701), [oclockvita.c](https://github.com/frangarcj/oclockvita/blob/63aa0442685a45c99bee433c6242032008aec740/oclockvita.c#L20-L21), PSVshell [main.c#L106-L123](https://github.com/Electry/PSVshell/blob/9a47b0eb22c60bdbddd8c7b5c9f1218c7e8a3037/src/main.c#L106-L123) |
| caches | 32 KiB L1 I and 32 KiB L1 D per core, 32-byte lines. One **2 MiB, 16-way PL310 L2 is shared by all four cores**, with pseudo-random replacement and no L2 data prefetch. The "1 MB L2" sometimes quoted is 7-cpu.com's Exynos 4210, not the Vita | [Main Processor](https://wiki.henkaku.xyz/index.php?title=Main_Processor&oldid=21601) ("a L2 cache of 2 MiB shared by all cores"); [Caches](https://wiki.henkaku.xyz/index.php?title=Caches&oldid=21009): AUX_CTRL `0x60090001` = 16 ways × 128 KB, bits 25 and 28 clear ([Linux cache-l2x0.h#L125-L129](https://github.com/torvalds/linux/blob/v6.7/arch/arm/include/asm/hardware/cache-l2x0.h#L125-L129)) |

**Why the game process is the only place.**
- **SceShell plugin.** The expert's rule: a plugin that competes with SceShell has a budget of
  genuinely under 2 MiB, and SceShell wants that headroom itself (screenshots and the like). The
  console agrees. Correction: the pool that runs out is SceShell's 50 MiB partition
  (`ScePhyMemPartShell`), from which every page of a `*main` plugin (module, stacks, TLS,
  `USER_RW` blocks) is taken; it had 28,672 B and 98,304 B free beside a running game. The 5 MiB
  PAF heap, with about 2.4 MiB free beside CLANNAD and as little as 0.19 MiB later, is a pre-mapped
  sub-pool inside it: exhausting the heap starves the UI, exhausting the partition starves
  firmware, as in #59, where a SceShell thread of the firmware's own ran out of pages while our
  plugins held 540,672 B of the partition
  ([PR #60](https://github.com/bee-san/vita-jp-overlay/pull/60)). That is why ncnn, Meiki and VOCR
  all moved out of it, and why no weights, arena, dictionary or cache of the recognizer may go
  back, at any size.
- **Kernel plugin.** It would share the 36 MiB kernel-driver pool (`ScePhyMemPartKD`), whose free
  space is unknown. A 2 MiB full-frame capture buffer has already failed there once (section 2).
- **Separate homebrew app.** It would get a game-sized budget, but only one game-budget app runs
  at a time, so it cannot OCR a game that is running. This is an inference; no source states it
  as a rule.

**What one job allocates.** Since #54 there are two `USER_RW` blocks, each rounded up to 4 KiB
pages: the verified weights, which stay while the title is the active game, and the job
workspace ([`core/vocr_ocr.c`](https://github.com/bee-san/vita-jp-overlay/blob/599dd9cbb6f5c452ceadb10363eea065219d435e/core/vocr_ocr.c), `layout`):

> workspace = max(line-finder arena, recognizer arena) + 3,760 B of runtime structs + the grey region (width × height), each part rounded up to 16 B

The finder's arena (83,323 B at 960×544) is smaller than every recognizer arena, so it adds nothing.
The peak is the two blocks together; between jobs only the weights block remains.

| model | weights block (between jobs) | workspace, 917×125 (CLANNAD's box) / 960×544 | peak, 917×125 / full screen | free `USER` the first full-screen job needs (peak + 4 MiB) |
|---|---|---|---|---|
| FL10 | 659,456 B | 245,760 / 651,264 B | 905,216 / 1,310,720 B | 5.3 MiB |
| always-safe limit (900,000 B, 131,072 B) | 901,120 B | 249,856 / 659,456 B | 1,150,976 / 1,560,576 B | 5.5 MiB |
| H15 | 1,159,168 B | 303,104 / 712,704 B | 1,462,272 / 1,871,872 B | 5.8 MiB |
| F20 | 1,560,576 B | 360,448 / 770,048 B | 1,921,024 / 2,330,624 B | 6.2 MiB |
| ceiling (1,600,000 B, 262,144 B) | 1,601,536 B | 380,928 / 790,528 B | 1,982,464 / 2,392,064 B | 6.3 MiB |
| width 3.0 | 2,682,880 B | 479,232 / 884,736 B | 3,162,112 / 3,567,616 B | 7.4 MiB |

FL10's 905,216 B peak and 659,456 B resident block are the figures #54 computed for the console;
the rest follow from the same formula. Before #54, one block held weights and workspace together
(900,896 B for FL10 at 917×125, logged on the console). The kernel's capture buffer (region × 4 B:
458,500 B at 917×125, allocated as 458,752 B in 4 KiB pages) comes from the kernel-driver pool
(`ScePhyMemPartKD`, 36 MiB) and is the same for every model.

### What the SDK gives a game, and what it takes back

In our words, from the SDK's `Programming-Startup_Guide`, `Kernel-Overview`, `Kernel-Reference`,
`Memory_Management-Tutorial` and `Scream-Overview` (release 3.55 documents):

- A retail game gets 256 MiB of ordinary main memory plus 26 MiB of physically contiguous memory,
  and up to 112 MiB of CDRAM. A title's `param.sfo` can ask for +29, +77 or +109 MiB more (the
  "Large Memory Mode" of its application settings; Vita3K reads it as `ATTRIBUTE2 & 0x0C`). The
  mode is fixed per title and cannot change while it runs; the larger modes also stop other
  applications from running beside the game. CLANNAD's `param.sfo` has `ATTRIBUTE2=0` (`notes/26`):
  plain 256 MiB.
- Before `main()` runs, the system has already taken the game's own program, a 3 MiB kernel
  reserve, libc (312 KiB), libfios (160 KiB), apputil (44 KiB) and the libc heap the title asked
  for (256 KiB by default; titles set their own, and the SDK's audio samples set 16 MiB). After that, every
  thread's stack and at least 4 KiB of TLS, and every PRX loaded through libsysmodule, come out of
  the same budget. A plugin's module pages, its thread's 32 KiB stack and its TLS are charged to
  the game in the same way, because taiHEN loads it into the game's process.
- The SDK tells titles to allocate large blocks at startup and sub-allocate from them, because
  the kernel allocator is slow and 4 KiB-granular. *Inference:* a title that follows this leaves a
  free figure that is fixed once it has loaded, which is why one sample per title, at its first
  dialogue, is informative; a title with per-scene heaps can move during a session.
- `sceKernelGetFreeMemorySize` returns the free size minus a margin for the system, per pool
  (main, CDRAM, phycont). It is a total, not a largest block; `USER_RW` blocks are 4 KiB pages
  mapped into the game's address space, so a block of 1.6 MB fails only when the total is short
  (*inference*: the SDK requires contiguity only of the PHYCONT types).
- Limits a plugin shares with its game: 2,048 memory blocks and 128 threads per process. A title
  that has used them up cannot start the worker thread; the game is unaffected, OCR unavailable.

### What games leave free, and the rule that replaces a worst case

| title | memory mode | free `USER` | source |
|---|---|---|---|
| CLANNAD (PCSG00415) | plain 256 MiB (`ATTRIBUTE2=0`) | 91,226,112 B (87 MiB) before a job | `notes/27` (one logged sample), `notes/26` |
| Stardew Valley | +109 MiB (365 MB shown) | about 56 MiB (309 of 365 MB used) | PSVshell README screenshot; its `perf.c` reads the running process's partition |
| the 16 vita-test titles: Collar×Malice Unlimited, Sweet Clown, Saekano, BELIEVER!, Kenka Banchou Otome, Kin'iro no Corda 3 AnotherSky and 4, Utsusemi no Meguri, Utawarerumono Itsuwari no Kamen, Naderebo!, Piofiore, Akai Suna Ochiru Tsuki, VARIABLE BARRICADE, Gunka o Haita Neko, Reco Love Gold Beach, Time Travelers | unknown | **unmeasured** | `eval/manifests/vita-test.jsonl` (titles only) |
| SceShell beside a game, for contrast | 50 MiB partition | 28,672 B; 98,304 B | `notes/08`, `notes/26`, PR #60 |

- **No safe worst case follows from the budget.** Nothing in the SDK reserves memory for anyone
  but the title, and the expert's rule says that a few games come close to the whole 256 MiB
  (homebrew often does). Two games have been measured; both left more than 50 MiB. Visual novels
  are 2D and usually light, but that is an expectation, not a figure: the 16 titles above have
  not been measured, and the per-title number is now free, because since
  [#55](https://github.com/bee-san/vita-jp-overlay/pull/55) every job, refused or not, logs
  `game vocr memory: free USER=…`. The first session in each title fills the table.
- **So the allowance is a rule, not a number.** Per job: free `USER` − 4 MiB reserve ≥ page-rounded
  workspace + page-rounded weights (0 when they are already resident; another model's resident
  block counts as free because it is released first). It refuses before anything is allocated,
  logs the numbers, and tries the configured fallback when that is a pinned model needing less.
  An always-safe model needs 5.5 MiB free for a full-screen job; a game that cannot spare that
  gets no OCR, which is the right outcome.
- **What the rule cannot see** is the game's own later allocations. The harm is bounded: the game
  loses at most the resident block (≤ 1.6 MB at the ceiling), only while it is the active
  foreground title, and never the last 4 MiB (more than the kernel's own 3 MiB pre-`main()`
  reserve, and enough for the late allocations the SDK lists: threads and PRX loads). A title that
  then fails for want of those bytes had under 1.6 MB of margin at that moment. Releasing the
  weights after some minutes without a job would shrink the window further; the overlay releases
  them today when the game is suspended, behind another app or disabled, when the file changes,
  on a refused preflight, and at stop.
- **The right structure is the current one:** a default from the ceiling tier with a fallback from
  the always-safe tier (H15 with FL10 since
  [#57](https://github.com/bee-san/vita-jp-overlay/pull/57)). A default from the always-safe tier
  needs no fallback. Both still wait for their first console run (#59).

### Which cores and which priority

From the SDK's `Kernel-Overview` ("Thread Priority", "CPU Affinity Mask of Threads", "Thread
Scheduling") and `Programming-Startup_Guide` ("CPU"), in our words:

- A game may use three of the four Cortex-A9 cores. Kernel and system threads are meant to run on
  the fourth, but some system work (system threads, interrupt handling) still lands on the three,
  so a game may not assume its threads are never preempted.
- The scheduler is fixed-priority and **not time-sharing**. Priorities run from 64 (highest) to
  191; a game's default is 160. Priorities 64-127 go to a ready queue per core, 128-191 to one
  common queue shared by the cores. A running thread is never preempted by a thread of the same or
  lower priority; a higher-priority thread preempts it at once. The SDK recommends pairing a
  common-queue priority with the all-cores mask (`SCE_KERNEL_CPU_MASK_USER_ALL`).
- The worker is created with the defaults (`0x10000100`, mask 0): priority 160 in the common
  queue, any of the three cores. **Consequence:** while it recognizes a line (up to about 0.7 s
  for a ceiling model with the game holding half the L2), a game thread at priority 160 that
  becomes ready cannot take that core; it waits for another core or for the worker to finish. On a
  visual novel, which rarely keeps three cores busy (PSVshell's Stardew Valley screenshots show
  cores 1 and 2 at 0-3%), that is invisible; on a title that does, it is a stutter of up to one
  line.
- **Recommendation to the overlay, one line:** create the worker at the lowest user priority
  (`0x10000100 + 31`, which the kernel maps to 191 in a game) with `SCE_KERNEL_CPU_MASK_USER_ALL`,
  the SDK's recommended pairing. Then every game thread preempts it, the game never waits for OCR,
  and OCR runs only on cores the game leaves idle. The cost is a longer line on a busy title; the
  Shell already waits up to 60 s for a job.
- The kernel plugin's worker thread (`VjoWorker`, priority 60, mask `0x10000`; kernel masks use
  bits 16-19 for cores 0-3, so this is core 0) outranks every game thread and runs on a game core
  by the plugin's choice. It serves every 32 KiB model read (36 for H15, once per session since
  #54) and every capture. Whether it belongs on the system core instead, and what it costs the
  game, is a measurement question: the CPU3 impact cannot be computed from source. Until matched
  PSVshell per-core readings exist (baseline against plugin, same clock, same scene, same
  interval), it is **unmeasured**.
- The L2 is one PL310 shared by the four cores; the game's share is unknown, and the tier table
  assumes half. The SDK publishes no ARM clock figure (its power modes A, B and C set the GPU
  clock), so 333 MHz stays the design clock, and CLANNAD's clock is unknown.

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

**Fixed since, on the host only** (the three items of
[#53](https://github.com/bee-san/vita-jp-overlay/issues/53), none yet run on a console):
1. **The weights were reloaded and re-hashed on every job**, the cost that grew with model size.
   [#54](https://github.com/bee-san/vita-jp-overlay/pull/54) keeps the verified weights resident
   and reads them in 32 KiB pieces without sleeps.
2. **A refused preflight hid its numbers.** [#55](https://github.com/bee-san/vita-jp-overlay/pull/55)
   logs the free `USER` bytes, the needs and the reserve on every job, and the CTC frames per
   line, so each console job measures ms per frame and each title reports its budget.
3. **Every out-of-memory error gave the same advice.**
   [#56](https://github.com/bee-san/vita-jp-overlay/pull/56) names the stage (kernel capture, game,
   Shell).

**The first console test of that build failed in SceShell, not in the game**
([#59](https://github.com/bee-san/vita-jp-overlay/issues/59), 2026-10-13). The API 16 build did not
boot: SceShell's `SceLiveAreaDailyCronJob` died inside ScePaf with `0x80024302`, no free page in
SceShell's 50 MiB partition. No OCR job had run. The root-cause analysis
([PR #60](https://github.com/bee-san/vita-jp-overlay/pull/60)) attributes it to the overlay's own
permanent footprint in that partition: the `*main` plugin's module pages, its two thread stacks and
TLS, and the `*ALL` worker that taiHEN loads into SceShell as well, 540,672 B in all, 36,864 B more
than the API 13 build that boots, in a partition measured at 28,672-98,304 B free beside a game.
The same mechanism explains issue #42 (a 65,544 B static buffer hung the boot). It is the clearest
statement yet of the rule this document rests on: the recognizer's budget is the game's, and the
Shell plugin must shrink, not grow. The fix (`NO_RESIDENT` for the worker outside games,
`--gc-sections`, UI arenas allocated only while open, then the dictionary and the worker leaving
SceShell for `*TITLEID` sections) is the overlay's work, outside this repo.

**Corrections to earlier statements:**
- **The overlay's default model is H15 with an automatic FL10 fallback since
  [#57](https://github.com/bee-san/vita-jp-overlay/pull/57)** (`e759187`). Until then the config
  default and the release package set `FL10_w8.vocr`
  ([M/core/config.c#L93](https://github.com/bee-san/vita-jp-overlay/blob/1a118ef8ca794bb5e2b5b2a96cd2cd07552d93eb/core/config.c#L93));
  `VJO_VOCR_DEFAULT_MODEL "H15_w8.vocr"` was only the host tools' default. Every console log is
  still FL10: **H15 has never run on a console** (#59 is that test, and it has not reached a game).
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

A job runs capture, line finding, then recognition of each detected line, one after another on
one core. Since #54 the model load happens once per game session (the first job that finds a
line), not per job.

| model | line finding (FL10, measured) | first-job load, once per session (est.) | est. recognition, 3 lines of 640 px, 333 MHz, whole L2 | the same, game holding half the L2 | later jobs, 333 MHz, half the L2 (est.) | later jobs, 444 MHz, whole L2 (est.) |
|---|---|---|---|---|---|---|
| FL10 | 34-38 ms | 0.13-0.15 s | 0.37-0.71 s | 0.37-0.71 s | 0.41-0.74 s | 0.31-0.57 s |
| H15 | same | 0.23-0.26 s | 0.66-1.19 s | 0.66-1.31 s | 0.69-1.35 s | 0.53-0.93 s |
| F20 | same | 0.31-0.36 s | 0.93-1.64 s | 0.93-2.06 s | 0.97-2.10 s | 0.74-1.27 s |
| width 3.0 | same | 0.53-0.61 s | 1.68-3.70 s | 1.68-3.84 s | 1.71-3.88 s | 1.29-2.93 s |

**Load-time estimate.** Before #54 the load was measured at 213-359 ms per FL10 job and estimated
to be: at least 81 ms of sleeps (81 kernel round trips of 8 KiB, at least 1 ms each), about 86 ms
of card read (the official memory card at about 7.6 MB/s, a Digital Foundry figure quoted
second-hand) and 40-60 ms of SHA-256 over the whole file (20-30 cycles per byte at 333 MHz; not
measured): about 210-230 ms, the low end of the measurement. #54 removes the sleeps and the
repeat, so the first job of a session pays the read and the hash, about 0.13-0.15 s for FL10
(#54's own estimate), scaled by blob bytes for the others; later jobs pay an open, a stat and a
close, a few ms. None of this has been measured since #54 (#59). A median three-line box takes
about half the recognition columns.

## 5. Buying capacity cheaply

In order of value per effort. Items 1-3 are overlay changes; nothing in `runtime/` has to change.

1. **Stop reloading the weights on every job.** Done in
   [#54](https://github.com/bee-san/vita-jp-overlay/pull/54): a bounded in-kernel wait instead of
   the 1 ms sleep loop, a 32 KiB kernel stage, one hash per session, and the verified weights kept
   in their own `USER_RW` block while the title is the active game, under the same preflight. It
   saves 0.2-0.36 s per FL10 job and an estimated 0.5-0.85 s per F20 job, for 0.66-1.6 MB of the
   game's memory held for the session instead of per job. Not yet run on a console (#59).
2. **Ask for 444 MHz during a job.** Call `scePowerSetArmClockFrequency(444)`, then restore the
   game's clock afterwards.
   - Compute-bound work gets 1.33× faster, which pays for a model a third bigger at the same
     latency. It costs battery while the job runs.
   - Unknown: whether the L2 then runs at 222 or 444 MHz (the Pervasive table lists both).
3. **Use a second core for multi-line boxes.** Run one line per thread, below the game's priority
   (section 1: with the default priority a game thread of the same priority would wait behind it).
   - The runtime allows this as it is: one context and arena per thread, one shared model.
   - A box of two or more lines then takes about half as long, for one more arena (183 KB for H15)
     and 32 KiB of stack plus 4 KiB of TLS.
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

- **Run the overlay's console test** ([#59](https://github.com/bee-san/vita-jp-overlay/issues/59),
  `docs/console-test.md` there), once the boot failure's fix has merged behind its boot-safety
  guard. Since #55 every job logs `game vocr speed: … us per frame` and
  `game vocr memory: free USER=…`, so ten pages of CLANNAD give the first on-console ms per frame
  for H15 and the FL10 fallback, and the first per-title budget line.
- **Fill the per-title table** (section 1): one session per title of the library, reading the
  `memory:` line (`sceKernelGetFreeMemorySize().sizeMain` inside the game), with the title's
  memory mode from its `param.sfo` (`ATTRIBUTE2`). This is the only number that can make RAM the
  limit for a model; SceShell's figures cannot (`vita-platform-memory.md`). Adrenaline, which runs
  PSP VNs, is unmeasured too. The tiers change only if a title refuses an always-safe job.
- **Run `vita/`'s test app on a PCH-1000** (`vita/README.md`, "On a PCH-1000"), with `clock=333`
  and `clock=444`. If ScePerf works on retail firmware, its counters replace the cycle model.
- **Measure the cores**, with PSVshell: per-core busy percentages for a baseline (no overlay) and
  for the plugin set, at the same clock, scene and interval, during ten pages of auto OCR. That is
  the only admissible evidence on the kernel worker's cost on core 0 and on the system core; it
  cannot be computed from source.
- **Sample SceShell's free memory at boot and at game launch** in both plugin sets (#60's plan):
  it settles whether the 36,864 B growth was the trigger of #59, and it is the number any future
  Shell-side change must be checked against.
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
python vita/budget_sweep.py check my_model_w8.vocr               # a trained blob against the ceiling, with its tier
python vita/budget_sweep.py ckpt --name w300 --out w300.pt       # for vita/bundle.py --ckpt w300.pt
                                                                 #   --sets synth-val --modes w8i, then run_vita3k.sh
```

The scripts that ran these are `$A/vita-budget/run_sweep.sh` and `run_sweep2.sh`.
