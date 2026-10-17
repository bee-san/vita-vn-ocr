# The PS Vita memory the recognizer runs in

Written 2026-10-13. The full platform reference, with every pool, size and citation, is
vita-jp-overlay's [`docs/vita-memory.md`](https://github.com/bee-san/Rubi_for_vita/blob/main/docs/vita-memory.md)
(PR [#61](https://github.com/bee-san/Rubi_for_vita/pull/61)). This note states the part that
decides how big a model may be, and corrects what `docs/vita-memory-budget.md` got wrong about
where the overlay's memory comes from. Sources are the Sony SDK documentation (cited there by
file name), the henkaku wiki, the taiHEN sources and the overlay's console measurements, in
particular vita-jp-overlay PR [#60](https://github.com/bee-san/Rubi_for_vita/pull/60)
(`docs/issue-59-root-cause.md`).

## The pools, in one table

| Pool | Size | Who draws from it | What of ours lives there |
|---|---|---|---|
| the game's main-memory budget (`ScePhyMemPartGame`) | 256 MiB (+29, +77 or +109 MiB when the title's `param.sfo` selects a Large Memory Mode); 4 KiB pages | the game: its code, heap, stacks, GPU-mapped data; plus every plugin loaded into it | **the whole recognizer**: weights block, job workspace (arena, runtime structs, grey region), the worker's 32 KiB stack and 4 KiB TLS, the worker module's pages |
| the game's video memory (`ScePhyMemPartGameCdram`) | 112 MiB; 256 KiB pages | the game's framebuffers, parameter buffer, textures | nothing of ours |
| SceShell's budget (`ScePhyMemPartShell`) | 50 MiB; measured 28,672 B and 98,304 B free beside a running game | SceShell: LiveArea, screenshots, notifications, its PAF heaps (the global one is 5,242,880 B, pre-mapped); every `*main` plugin; every `*ALL` plugin's pages | the overlay's UI plugin (405,504 B of pages, 90,112 B of stacks and TLS), the dictionary result arena (64 KiB, from the PAF heap), and today the `*ALL` worker's 45,056 B of pages: **never the recognizer** |
| the kernel drivers' pool (`ScePhyMemPartKD`) | 36 MiB, shared by every driver and kernel plugin; free size unmeasured | kernel modules and `KERNEL_RW` blocks | the kernel plugin's pages and its region-sized capture buffer (width × height × 4 B, 458,752 B at 917×125), the same for every model |
| physically contiguous main memory (`ScePhyMemPartPhyCont`) | 26 MiB; 1 MiB pages | Codec Engine (JPEG, AVC), camera | nothing of ours |

Sources: wiki Memory_budget; SDK Kernel Reference (`sceKernelAllocMemBlock` granularity,
`sceKernelGetFreeMemorySize`), SDK Programming Startup Guide (memory modes); PR #60 and
vita-vn-ocr `notes/26`, `/27` for the measured figures. A process can only take pages from its own
budget. When they are gone the kernel returns `0x80024302`
(`SCE_KERNEL_ERROR_NO_FREE_PHYSICAL_PAGE`), whatever is free in the other pools.

## What this means for model size

1. **The recognizer's budget is the game's free main memory, and nothing else.** Weights, arena,
   grey region and the worker thread all come out of the game's 256 MiB, as `USER_RW` blocks
   rounded up to 4 KiB. SceShell's figures (the 5 MiB PAF heap, its 2.4 MiB free, the 1.5 MiB UI
   reserve) do not bound the model; they bound the overlay's UI and dictionary result arena only.
   The expert's rule, now binding in the overlay's `AGENTS.md`: no OCR work, weights, workspaces
   or dictionaries in SceShell, and the worker is to move from `*ALL` to one `*TITLEID` section
   per enabled game.
2. **How much a game leaves free is the number that matters, and it is barely measured.** CLANNAD
   (default mode, `ATTRIBUTE2 = 0`): 91,226,112 B before a job and about 87 MiB in another
   session [measured]. Stardew Valley in the +109 MiB mode: about 56 MiB (a third-party PSVshell
   screenshot). The expert's experience: few games take all 256 MiB unconditionally, some get
   close, homebrews often do. No other Japanese visual novel has been sampled. The overlay logs
   the free figure on every job and refusal (`game vocr memory: free USER=…`); the self-test
   planned in vita-jp-overlay will sample it per title.
3. **Preflight decides, per job, with a reserve.** The worker reads
   `sceKernelGetFreeMemorySize().sizeMain`, subtracts a 4 MiB reserve (an untuned figure; one
   title so far) and refuses or falls back to FL10 when the page-rounded weights plus workspace do
   not fit. A failed allocation after a passed check is normal: the game may take the memory
   first. So a model's memory cost is judged by its *worst-case per-job total* in the game:
   weights (resident between jobs since vita-jp-overlay #54, in their own block: 659,456 B for
   FL10, 1,159,168 B for H15, page-rounded) + the region-sized workspace (arena + 3,760 B of
   runtime structs + width × height grey pixels) + reserve. For the ceiling model (1.6 MB weights,
   256 KiB arena) at a full 960×544 region that is about 2.4 MB plus the 4 MiB reserve: well under
   the 56-91 MiB seen free, so **RAM is not what limits the recognizer** in the two games
   measured. The 2 MiB shared L2 and the time per line bind first, as `docs/vita-memory-budget.md`
   argues.
4. **What would make RAM the limit** is a game that leaves little free: below about 6 MiB free
   (the 4 MiB reserve plus about 1.8 MiB of resident H15 weights and a full-screen workspace) the
   H15 job already fails the check [estimate from the sizes above]. That is
   handled by preflight, the FL10 fallback and a tighter region, not by shrinking the ceiling.
   Larger models cost more only where the game has less to give; a tier scheme ("always safe",
   "safe with preflight", "never") should therefore be stated as per-job totals against measured
   free memory per title, with the SceShell constraint removed from the equation entirely.
5. **The kernel capture buffer is independent of the model** and comes from the 36 MiB driver pool:
   region × 4 B, allocated per job and freed when the worker has read the last row. A full-screen
   2,088,960 B buffer has failed there once (vita-jp-overlay #14); the region size, not the model,
   decides this.
6. **Vita3K says nothing about any of this.** It does not model the budgets, SceShell or taiHEN; a
   Vita3K pass is evidence of arithmetic parity, not of memory safety on a console.

## Corrections to `docs/vita-memory-budget.md`

Applied in the same change, inline:

- `*ALL` does not mean "every enabled game". taiHEN loads an `*ALL` plugin into SceShell and every
  system application too (taihen-parser `parser.c`: the `ALL` section is matched for every
  section but `KERNEL`), where its pages stay resident: 45,056 B of SceShell's budget for the
  worker today.
- A plugin's module pages, stacks, TLS and memory blocks are charged to the process it is loaded
  into. This is no longer an inference: the SDK's Kernel Overview and Reference state that the
  memory block service loads modules and allocates stacks, and that stacks and TLS come from the
  application's budget; PR #60 measured our Shell pages.
- The SceShell problem is the 50 MiB partition, not the 5 MiB PAF heap. The PAF heap is pre-mapped
  inside the partition; `sce_paf_memalign` failing returns NULL. `0x80024302` in SceShell (our
  786,432 B result allocation in #11; the firmware LiveArea thread in #59) means the partition had
  no free page. The two console samples beside a game: 28,672 B and 98,304 B free.
- Issue #42's hang came from 65,544 B of `.bss` added to the Shell module: 16 pages of the
  partition, not PAF heap. PR #60 attributes it.
- The "kernel pool" is `ScePhyMemPartKD`, 36 MiB for every kernel driver together.
