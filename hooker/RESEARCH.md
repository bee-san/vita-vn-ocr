# Japanese text extraction from game RAM on PS Vita

Investigated on 2026-10-10 for `ps-vita-ocr` and `vita-jp-overlay`.

## Decision

Reading Japanese dialogue directly from RAM on a homebrew-enabled PS Vita is
technically feasible. A generic finder could discover text buffers without a
handwritten parser for each game engine. A shared font hook could capture
character codes from games that use the Vita's font APIs.

These are promising optional text sources. This investigation did not establish
a reliable, automatic replacement for OCR across arbitrary Vita games. The
recommended direction is an extraction backend with OCR fallback, and a small
hardware experiment to establish coverage before committing to a larger system.

The evidence is upstream source inspection and documentation. No native text
extractor was implemented, no commercial game's RAM was inspected, and no
PCH-1000 or PSP extraction run was performed. There are no measured extraction
accuracy, memory, latency or compatibility figures.

## What "engine independent" can mean

| Requirement | Finding |
|---|---|
| Scan standard text encodings without recognising an engine | Feasible design; Textractor already demonstrates this on Windows |
| Select a text source once per game, without writing engine code | Plausible on Vita when a usable buffer exists; needs hardware validation |
| Hook one shared text/font API across different engines | Feasible API boundary for games that call it; coverage is unknown |
| Automatically return exactly the current visible dialogue in any game | Not established by this investigation |
| Reuse Luna's existing title-specific hook knowledge on hardware | Potentially useful, but requires native adaptation and version checks |

An engine-independent discovery algorithm can still produce a per-game
configuration: encoding, buffer locator, pointer chain, hook location or selected
text stream. Avoiding bespoke engine code does not eliminate source selection,
relocation or verification.

## What Textractor actually does

Textractor is broader than a collection of engine-specific RAM readers:

1. Its shared Windows API hooks intercept functions such as `TextOut`,
   `ExtTextOut`, `DrawText` and `GetGlyphOutline`. These offer common interception
   points when a game uses those APIs. See [the API hook implementation][tex-api].
2. `SearchForText` takes a known sentence, converts it to UTF-8 and a selected
   code page, searches writable memory for those and UTF-16, and creates
   `DIRECT_READ` sources at matching addresses. This is a concrete precedent for
   discovery without identifying an engine. See [the finder][tex-finder].
3. `SearchForHooks` instruments candidate functions and examines possible string
   arguments, then reports candidates. This discovers a useful hook rather than
   knowing the engine's layout in advance. See [the same source][tex-finder].

The Windows APIs, executable conventions, x86/x64 trampolines, MinHook and
Windows memory-query/exception machinery are platform-specific. Their discovery
ideas are transferable; the implementation cannot run unchanged on the Vita.

## What LunaTranslator's Vita support actually does

[Luna's emulator documentation][luna-emulators] identifies Vita3K for Vita games
and PPSSPP for PSP games. This is emulator support, not a native Vita plugin.

The inspected Vita3K backend finds the emulator's JIT registration code and
connects guest ARM addresses to generated host code. It matches the running title
ID before installing known hooks. Its version-dependent address adjustments
also show why emulator addresses need interpretation. See [the backend][luna-vita].

[The hook table][luna-table] contains title IDs, guest instruction addresses,
argument offsets, encoding flags and extraction/filter callbacks. It includes
UTF-8, UTF-16 and CP932/Shift-JIS paths, distinct dialogue/name/choice hooks and
version-specific entries. Automatic support is therefore informed by knowledge
of individual games.

Luna also has generic discovery. `SafeSendJitVeh` includes a
`JITTYPE::VITA3K` case, examines emulated registers as possible string pointers
and passes candidates to `DoSend`. Its JIT search instruments registered code
locations in a chosen address range. This does not require a prewritten hook for
the game's engine. See [the generic finder][luna-finder]. Its effectiveness on
unknown titles was not tested here.

The JIT gives the host convenient instrumentation points. A physical Vita runs
the ARM instructions directly; a native counterpart would need its own
ARM/Thumb hooks, register capture and memory-access handling. Broadly
instrumenting game code is a more complicated experiment than a bounded RAM
search or a small set of shared API hooks.

## Access to RAM on real Vita hardware

Hardware access is available in the homebrew environment:

- [rinCheat][rin-readme] exposes in-game stack and partial heap scanning.
  Its `scanMemory` compares bytes in a supplied range; `scanHeap` locates a
  memory block and scans its mapped range. See [the memory implementation][rin-memory].
- [taiHEN's usage examples][tai-usage] and [public API][tai-api] expose import,
  export and module-offset hooking. They provide the underlying interception
  facilities for a native text-capture plugin.

rinCheat is evidence of access, not an implementation recommendation: its own
documentation describes it as unfinished, and `scanHeap` has a comment about
random crashes and mapping uncertainty. A text finder must enumerate valid,
readable game mappings and handle their lifetimes rather than assume a linear,
permanently readable heap.

The inspected `bee-san/vita-jp-overlay` already copies framebuffer data in the
game's execution context using `ksceKernelMemcpyUserToKernel`. Its comments
report crashes from reading another process's framebuffer in a worker. See
[the capture implementation][overlay-capture]. That report is upstream evidence,
not a crash reproduced in this investigation.

Text capture needs an explicit game-process access path. A SceShell worker
cannot simply dereference a pointer obtained from a game. A game-side plugin or
a correctly scoped kernel capture path could pass bounded text events to the
existing shell overlay. The exact transport and mapping APIs remain design work.

## Why finding Japanese bytes does not identify visible dialogue

This is the central engineering problem, beyond obtaining RAM access.

A candidate string can be a current dialogue buffer, an earlier line, backlog,
speaker name, menu, choice, dictionary entry or a loaded script containing future
pages. All can be valid Japanese. Language scoring alone cannot prove which
candidate is visible.

Other possible obstacles are:

- The current line's pointer changes while the underlying script remains static.
- A typewriter effect displays only a prefix of a complete stored sentence.
- Characters are split across buffers, intermixed with commands, or represented
  by custom glyph IDs rather than a standard contiguous string.
- The text is transiently decoded, then only cached glyphs or an image remain.
- Allocation, scene changes, save/load or a restart invalidate saved addresses.
- Several copies of a sentence exist, and more than one changes during play.

These are failure modes to test, not a measured description of every Vita VN.
There is a concrete encoding example in Luna: `TPCSG00291` follows several
pointers and decodes swapped-byte Shift-JIS with special token handling. Other
callbacks remove game-specific commands. A search limited to ordinary UTF-8,
UTF-16LE and CP932 will not cover every representation. See [the title callbacks][luna-table].

Raw RAM strings also do not inherently supply screen coordinates. An overlay
can show extracted text and offer dictionary lookup in its own panel; locating
words on the original game image requires additional layout information.

## Option A: a generic RAM finder

This is the closest fit for avoiding handwritten engine support while accepting
one-time source selection for a game. The following is a proposed workflow,
not an implemented or benchmarked algorithm:

1. Obtain a short distinctive visible sentence, typed manually or recognised
   once with OCR. Manual input allows discovery without invoking OCR.
2. Search validated readable game mappings for UTF-8, UTF-16LE and CP932/Shift-JIS
   forms. Treat every hit as a candidate, with its encoding and mapping identity.
3. Advance multiple dialogue pages and compare candidate output with the visible
   text. Retain sources that follow the current dialogue rather than merely
   contain the original sentence somewhere in a script.
4. If text is selected through a changing pointer, investigate a stable locator
   or pointer chain. This may require more work than polling a fixed buffer.
5. Save a locator that can be resolved and validated on the next launch. Include
   game/build identity; a raw virtual address alone is insufficient evidence of
   a durable configuration.
6. During normal use, copy a bounded string from the selected source, decode it,
   debounce partial updates and suppress duplicate emissions.
7. Invalidate the source if its mapping, encoding or content checks fail. Rescan,
   ask for source selection or use OCR rather than return stale text silently.

Without a known sentence, a discovery mode could rank Japanese-looking strings
and track changes across pages. That is a weaker heuristic because it lacks an
initial exact anchor. A page change can also modify unrelated UI strings.

Repeated whole-memory scans are unnecessary once a valid source has been found.
A chunked discovery pass, bounded candidate list and small text buffer would fit
the project's memory priorities better than retaining complete RAM snapshots.
Hashing changed blocks may help investigation, but changed bytes alone do not
identify dialogue. Chunk size, candidate limits and scan scheduling require
hardware measurements.

## Option B: shared font API hooks

VitaSDK provides these concrete interception candidates:

| API | Available input |
|---|---|
| `scePvfGetCharGlyphImage` / `_Clip` | Font handle, character code and output image-buffer description |
| `sceFontGetCharGlyphImage` / `_Clip` | Font handle, character code and glyph-image description |
| `scePvfGetCharInfo` / `sceFontGetCharInfo` | Character codes used for metric queries |

See [PVF declarations][sdk-pvf] and [PGF declarations][sdk-pgf].

A common hook could log character-code events for games calling these APIs and
try to reconstruct text without understanding each engine's script format.
The existence of these APIs is verified; their use by representative commercial
VNs and the completeness of their event streams remain unmeasured.

Important limits of the proposed capture method:

- A game can rasterise glyphs into a cache or atlas long before displaying them;
  subsequent display may produce no new font calls.
- Metric queries can repeat, arrive out of reading order or concern invisible
  text. They are discovery evidence, not proof of display.
- Bitmap positions can be positions in an offscreen target or atlas, rather
  than final screen coordinates.
- Separate threads, font handles, layout passes, outlines and ruby text can
  complicate grouping and deduplication.
- A game can use its own renderer or prebuilt bitmap font and bypass the APIs.

A diagnostic logger should record the API, code, font handle, thread, ordering,
destination identity and available bitmap positions. It should copy events into
a bounded queue and defer decoding or UI work. This is a proposed logging design.

## Why a generic graphics hook is insufficient

The Vita's graphics API separates texture binding and geometry drawing.
`sceGxmDraw` takes primitive/index data, and `sceGxmSetFragmentTexture` binds
textures. Neither is a shared "draw Japanese sentence" call. See [GXM declarations][sdk-gxm].

Consequently, intercepting a draw call can reveal textured geometry without
revealing the original characters. Recovering text from a font atlas would also
need its character mapping, texture coordinates, transforms and reading order.
Font-rasterisation events might help build a mapping when available, but this
does not turn GPU interception into a demonstrated universal extractor.

## Option C: adapt known title hooks

Luna's guest addresses, argument layouts, encodings and filters are useful
research starting points for native capture. A selected hook could provide
more precise extraction than a heuristic scanner.

Porting requires checking the exact game version and region, translating guest
addresses to the native loaded module, preserving ARM/Thumb execution state,
and establishing which registers or pointers hold text. A recorded address may
be an instruction inside a function, so it must not automatically be treated as
a function-entry hook. Emulator-relative adjustments must not be copied blindly.

This route accepts game-specific knowledge. It offers a possible path for titles
where generic discovery fails, rather than satisfying the strictest
engine-independence requirement. No existing Luna hook was ported here.

## PSP and Adrenaline

Luna's PSP backend uses PPSSPP's JIT and matches game disc IDs; it is also
emulator instrumentation. See [the PPSSPP backend][luna-psp]. Its conceptual
discovery methods can inform a PSP implementation, but the code does not become
a native PSP plugin without adaptation.

Native Vita games execute ARM code, while PSP games use the PSP's MIPS/Allegrex
environment. PSP/Adrenaline therefore needs a separately investigated capture
backend. This investigation did not establish how the Vita overlay should access
PSP game RAM or forward text events from that environment.

## Suggested architecture and first experiment

Treat extraction as another source feeding the overlay's text/dictionary flow:

```text
game-side capture / validated game-memory reader
                  |
          bounded text/event queue
                  |
          decode and select source
                  |
          overlay dictionary lookup

screen capture -> OCR when extraction is unavailable or invalid
```

The capture mechanism and optional per-game locator should be separate from the
dictionary UI. Keep the console implementation dependency-free C where possible.
Avoid a blanket port of the Windows instrumentation stack.

For successful extraction, model weights are unnecessary. Loading OCR weights
only when falling back could reduce resident memory. This is an architectural
expectation, not a measured memory saving; code, conversion tables, queues and
discovery state must still be accounted for. A located string is also expected
to cost less work than OCR, but discovery may be expensive. No speed estimate
was established.

The first useful implementation would be a diagnostic font-call logger plus a
bounded known-sentence RAM search, tested on several representative Vita VNs.
Determine whether either route follows current dialogue through multiple pages
before building automated source scoring or broad native instrumentation.

### Hardware validation plan

For each title, record title ID, game/build version, firmware, plugins, console
model, CPU clock and extraction method. Keep hand-checked dialogue separate
from pages used to select or calibrate a source.

Measure and report:

- Exact current-page extraction and CER using the shared `eval/` normalisation
  and metric code where applicable; also report missed pages and extra text.
- Capture of punctuation, kana, kanji, names, choices, ruby and line breaks, with
  a declared policy for text not intended as dialogue.
- Partial typewriter output versus completed pages, stale lines, duplicates and
  preloaded future text. Choose and document the intended visibility policy.
- Behaviour during backlog, skip, scene changes, save/load, suspend/resume and
  process exit; revalidate locators after fresh launches.
- Peak additional memory, including code, stack, decoding tables, queues and
  scanner state. Distinguish reserved memory from measured peak use.
- Discovery latency, ordinary capture latency and game slowdown, with repeated
  samples and the distribution or median reported alongside the command/data.

Static screenshots alone cannot test RAM-source discovery or its time-dependent
correctness. Pair hand-checked visible pages with runtime extraction events.
Keep copyrighted screenshots, RAM dumps, extracted corpora, executables and
fonts local; commit only code, source manifests and small aggregate results.

## Evidence boundaries and unresolved questions

| Established from source | Still needs investigation or hardware measurement |
|---|---|
| Generic memory/text discovery exists in Textractor | Coverage of the corresponding method on real Vita VNs |
| Luna has Vita3K title hooks and a generic JIT finder | Native discovery cost and reliability without the emulator's JIT |
| Homebrew plugins can scan memory and install hooks | Complete readable-mapping enumeration and stable text locators |
| Vita font APIs expose character codes | Which games call them, when, and whether full sentences are recoverable |
| GXM draw calls do not accept text strings | Practical atlas mapping if a game bypasses shared font APIs |
| The existing overlay uses game-context memory copying | A robust text-event transport into its SceShell UI |

No general native Vita text hooker was identified in the sources checked. This
is a bounded search result, not proof that no such project exists anywhere.

The older claim in `docs/research-findings.md` that hooking works "only in
emulation" is too broad. Emulators provide existing implementations and easier
instrumentation; native Vita hooking itself is feasible. Generic, reliable
dialogue coverage on hardware is the unresolved part. This note does not change
the existing OCR measurements or imply that extraction has surpassed them.

## Source provenance and reproducing the inspection

Source revisions below identify the inspected code, not recommended dependency
versions for a future implementation. Mutable documentation links describe the
documentation read on the investigation date. No source checkout or third-party
binary is included in this folder.

| Repository | Inspected revision |
|---|---|
| Artikash/Textractor | `df5b830d1757a66aa162f8081e99d42c9bd50488` |
| HIllya51/LunaTranslator | `b0c0c2d2a03b330c388b42d7e8f0556b6d230980` |
| Rinnegatamante/rinCheat | `d17fd2a7a542346e503afd9cd1df8c5e8bd33626` |
| yifanlu/taiHEN | `309b3800bcb8ebbd5e4f5e5e920af3da3590b829` |
| vitasdk/vita-headers | `e66ebe90b73d1fa4cce005a5b2072cec27322544` |
| bee-san/vita-jp-overlay | `61a2a50ab93953753b7531ec408621f36618cf98` |

The inspection used GitHub repository/tree metadata and raw source downloads,
then local searches and reads. For example, on the Linux host:

```sh
rg -n 'SearchForText|SearchForHooks|DIRECT_READ' tex-hookfinder.cc
rg -n 'JITTYPE::VITA3K|SafeSendJitVeh|DoSend' luna-hookfinder.cc
rg -n 'MatchGameId|CheckEmAddrHOOKable' luna-vita3k.cpp
rg -n 'emfunctionhooks_1|CODEC_UTF|TPCSG00291' luna-vita3k_1.cpp
rg -n 'scePvfGetCharGlyphImage|sceFontGetCharGlyphImage' pvf.h pgf.h
rg -n 'sceGxmDraw|sceGxmSetFragmentTexture' gxm.h
rg -n 'scanMemory|scanHeap' rin-memory.c
rg -n 'copy_from_game|ksceKernelMemcpyUserToKernel' overlay-capture.c
sha256sum tex-hookfinder.cc luna-hookfinder.cc pvf.h pgf.h
```

To repeat the source review, download each file at its linked revision into a
scratch directory, verify its SHA-256 against the table below, and inspect the
named functions. Source inspection has no timed/repeated benchmark samples.

[tex-api]: https://github.com/Artikash/Textractor/blob/df5b830d1757a66aa162f8081e99d42c9bd50488/texthook/engine/native/pchooks.cc
[tex-finder]: https://github.com/Artikash/Textractor/blob/df5b830d1757a66aa162f8081e99d42c9bd50488/texthook/hookfinder.cc
[luna-emulators]: https://docs.lunatranslator.org/en/emugames.html
[luna-vita]: https://github.com/HIllya51/LunaTranslator/blob/b0c0c2d2a03b330c388b42d7e8f0556b6d230980/src/NativeImpl/LunaHook/LunaHook/emulators/vita3k.cpp
[luna-table]: https://github.com/HIllya51/LunaTranslator/blob/b0c0c2d2a03b330c388b42d7e8f0556b6d230980/src/NativeImpl/LunaHook/LunaHook/emulators/vita3k_1.cpp
[luna-finder]: https://github.com/HIllya51/LunaTranslator/blob/b0c0c2d2a03b330c388b42d7e8f0556b6d230980/src/NativeImpl/LunaHook/LunaHook/hookfinder.cc
[luna-psp]: https://github.com/HIllya51/LunaTranslator/blob/b0c0c2d2a03b330c388b42d7e8f0556b6d230980/src/NativeImpl/LunaHook/LunaHook/emulators/ppsspp.cpp
[rin-readme]: https://github.com/Rinnegatamante/rinCheat/blob/d17fd2a7a542346e503afd9cd1df8c5e8bd33626/README.md
[rin-memory]: https://github.com/Rinnegatamante/rinCheat/blob/d17fd2a7a542346e503afd9cd1df8c5e8bd33626/main_module/memory.c
[tai-usage]: https://github.com/yifanlu/taiHEN/blob/309b3800bcb8ebbd5e4f5e5e920af3da3590b829/USAGE.md
[tai-api]: https://github.com/yifanlu/taiHEN/blob/309b3800bcb8ebbd5e4f5e5e920af3da3590b829/taihen.h
[sdk-pvf]: https://github.com/vitasdk/vita-headers/blob/e66ebe90b73d1fa4cce005a5b2072cec27322544/include/psp2/pvf.h
[sdk-pgf]: https://github.com/vitasdk/vita-headers/blob/e66ebe90b73d1fa4cce005a5b2072cec27322544/include/psp2/pgf.h
[sdk-gxm]: https://github.com/vitasdk/vita-headers/blob/e66ebe90b73d1fa4cce005a5b2072cec27322544/include/psp2/gxm.h
[overlay-capture]: https://github.com/bee-san/Rubi_for_vita/blob/61a2a50ab93953753b7531ec408621f36618cf98/kernel/capture.c

### Inspected-file SHA-256

Scratch filenames correspond to the linked upstream files; digests cover the
downloaded bytes without newline or encoding conversion.

| Scratch file / upstream source | SHA-256 |
|---|---|
| [tex-pchooks.cc][tex-api] | `7bd853adfda42b1678b15b17136cfe7cd577590353d35b0ec1ba04c2e8454306` |
| [tex-hookfinder.cc][tex-finder] | `0bcf9e1dcff469145667640df3796de69b629981d5802b5f19f85eb84da46d53` |
| [luna-vita3k.cpp][luna-vita] | `9f8f34c304a025d0e5690f549660d8cb240db33e2a1aa5bb5da21c2a6ec26354` |
| [luna-vita3k_1.cpp][luna-table] | `680fe9da376939f273f81a5600ba16cce3ceb5f3b0177fbbb23673563d5e9948` |
| [luna-hookfinder.cc][luna-finder] | `cadc8e46c529b1c1798de096423461c14ee93ea7212e747293d8f862762ef691` |
| [luna-ppsspp.cpp][luna-psp] | `4bdcf89c404a7b10c02eb21e32847e5f13c5a8232fcbc5cd9e91594ce3543325` |
| [rin-readme.md][rin-readme] | `f665cbfe6be59264a658b2cbd75e429affb12a8226960d17a459fc30e226d0b5` |
| [rin-memory.c][rin-memory] | `16922105b8294c65fe581aff7847d544f883569115f6afd90bbb618daa5ccc3a` |
| [taihen-USAGE.md][tai-usage] | `d24cbb159513965ffb9cdfa3464d9e753fbcc1e2860abaaf553a75cdfa3307b1` |
| [taihen-taihen.h][tai-api] | `d8e6d351bc6a567a06d75503be20d10dba96933ad0b9662ecb8000b579a9282d` |
| [pvf.h][sdk-pvf] | `e3a45149cb3b251346e6069e98c562c16f9b4ec4bdf470baa36582d492ed78ba` |
| [pgf.h][sdk-pgf] | `7cd9d56b6bfa4dc1d2dcafc6cbac22c39d3fe52d790713deee6d6266b149984a` |
| [gxm.h][sdk-gxm] | `80b0b9dce238b8570fdd964e04d3c276d0f62e0ba293033c01c5a299ee0d296b` |
| [overlay-capture.c][overlay-capture] | `b6e37975e3688061a327901b07ae34979a99b13b1c429520c26ec2e8775e8295` |
