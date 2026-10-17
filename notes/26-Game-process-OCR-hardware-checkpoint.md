---
title: Game-process OCR hardware checkpoint
created: 2026-10-11
updated: 2026-10-11
status: historical-v075-hardware-confirmed-see-later-checkpoint
tags: [vita, ocr, hardware, debugging, input]
---

# Game-process OCR hardware checkpoint

**Later checkpoint:** [[27-Meiki-hardware-and-project-lessons]] records two
successful physical ABI3 Meiki jobs, the Square/Circle repair, verified release
installation and the user's working-region confirmation. It also records the
locally validated Settings engine picker, VOCR default, Japanese-only selection
default, Daijirin conversion/staging and pending installation/LiveArea checks.
The v0.7.5 record below remains historical; Meiki game-mount visibility is now
proved for CLANNAD.

**Historical v0.7.5 checkpoint:** v0.7.5 is released, installed with full native readback, booted and physically tested in CLANNAD. Local lookup selects characters manually, returns at most one entry and uses one 64 KiB result arena; background local OCR uses no dictionary-result arena. All frozen-source sanitizer and exact PR/main/tag checks pass. Seven fresh text-bearing FL10 jobs finish with `rc=0 io=0 cleanup=0`; the local arena peaks at 53,488 bytes. Physical cursor sessions reuse saved text without another OCR job, and the user reports "works great". The user also reports poor OCR quality and requests another Meiki trial. Recognition accuracy/CER, broad compatibility and total peak memory remain unmeasured; historical checkpoints retain their original boundaries.

## Exact release and native evidence

**Historical v0.7.3 observation:** the physical Vita boots API 13, loads the kernel/Shell/game OCR plugins, and completes text-bearing FL10 OCR jobs with `rc=0 io=0 cleanup=0`. More captures complete afterward. This resolves the observed API 12 model-polling/retained-ownership failure for that session. The user reports that opening the overlay with L+R also advances game dialogue. That input defect is separate from clean OCR cleanup and capture recovery.

[Overlay v0.7.3](https://github.com/bee-san/Rubi_for_vita/releases/tag/v0.7.3) is tagged at `24a3c711c24994e6b466930c2087f8dcdc010d32`. Its independent published-asset audit verifies API 13, all 38 manifest files, pinned FL10/H15 and JMdict, settings artwork, local defaults and no Anki queue. That package result is separate from the native observations below.

| ID | Evidence inspected | What it establishes |
|---|---|---|
| HW01 | Native plugin response collected at 09:43:58 UTC on 2026-10-11 | Complete 20-entry inventory, process-query error 0; overlay kernel under `*KERNEL`, Shell under `*main`, OCR worker under `*ALL` all loaded, individual query errors 0. |
| HW02 | `overlay-api13-boot-retry.log`, final API 13 section beginning at line 1841 | API 13 startup and worker-start marker, followed by 15 successful game-vocr records. Seven have one detected line and actual model load/inference; eight have zero lines. All 15 have `rc=0`, `io=0x00000000`, `cleanup=0`. No capture-request failure appears in this final section. Earlier failed API 11/12 records in the same cumulative file remain historical. |
| HW03 | User report during this physical test | L+R opening the overlay also advances dialogue. Keep this open even though OCR cleanup/capture recovery succeeds. |

HW02 is 115,209 bytes, SHA-256 `fe34c35029bc39d78a7b4539cbd308379d9203e314e972f6c2f0ae526998d9d7`. Private locator: `vita-agent-use/agent/api13-testing-20261011/overlay-api13-boot-retry.log`. Raw dialogue, screenshots and native response JSON stay private.

**Measured session values:** all records use FL10 and a 917×125 region with 900,896 bytes reported as USER workspace. The seven text-bearing records show line finding 34–38 ms, model load 217–288 ms and recognition 36–126 ms. These are logs from varying dialogue, not a controlled benchmark or a measured process/console peak. Zero-line jobs do not prove model loading or recognition. Successful nonempty recognition still needs comparison with the actual displayed text before claiming accuracy; no hand-checked CER or screenshot comparison is reported here.

The native agent command path now responds successfully with the saved pairing. No pairing reset is needed or credited. See [[25-Vita-debugging-tools]] and [[09-Device-operations-and-recovery]] for the supported client and exact evidence-collection syntax.

## v0.7.4 mapped input repair and deployment

**Implemented and published, HW05:** [overlay PR #28](https://github.com/bee-san/Rubi_for_vita/pull/28), tested head `da861f06e9ed3ee6430018be47d8db9b70df67b8`, is merged and released as [v0.7.4](https://github.com/bee-san/Rubi_for_vita/releases/tag/v0.7.4) at `3f7ae81971db7ca4913d881be9750cfaada4799f`. The tag has the same source tree as the tested head. API 13 and its exports are unchanged.

The physical v0.7.3 input trace records `ReadPositive2` hook 7 receiving `0xC00` while raw L+R is `0x300`. The [pinned VitaSDK contract](https://github.com/vitasdk/vita-headers/blob/e66ebe90b73d1fa4cce005a5b2072cec27322544/include/psp2/ctrl.h) maps the shoulders to L1/R1 in `*2`/`Ext2` exports. The old filter hid only legacy L/R and left pressure bytes untouched. The repair translates mapped shoulders into shared canonical hold/replay state before filtering, then restores their native encoding while preserving genuine external-controller L2/R2. It clears hidden buttons' matching pressure values and all 12 pressure fields during full blocking, preserving timestamps and the SDK's final four reserved bytes. Only an introduced short-tap replay receives pressure 255; ordinary held pressures remain unchanged. Contended reads reserve the trigger components without spinning.

**Measured host regression, HW06:** the production wrappers/filter pass all 14 controller cases, covering all 12 exports before Shell input ownership, positive/negative logic, 64-sample canaries, mixed APIs, real L2/R2, held suppression, short-tap replay, pressure preservation, contention/release, copy failures, raw Shell input and 256 repeated bounded diagnostic calls. The same final harness compiled against v0.7.3 `24a3c71` fails 9/14 cases. Linux x86_64/GCC 16.2.1 with the pinned OCR runtime and released FL10/H15 passes all 18 host suites. These are deterministic boundary tests, not physical input-latency or visible-dialogue measurements.

An independent Linux x86_64/Clang 22.1.8 Debug run uses `-fsanitize=address,undefined -fno-omit-frame-pointer`, leak detection and halt-on-error. All 18 suites, 190 C cases and 41 tooling tests pass with no failures, skips or sanitizer reports; both pinned models exercise the actual game/kernel broker. Three JPEG round trips and five error cases also pass; the JPEG exit-lifetime check sets `LSAN_OPTIONS=use_stacks=0:use_registers=0`. Reproduction commands are recorded in the private verification receipt; the test entrypoints are `ctest --test-dir build/host --output-on-failure` and `python3 tools/test_jpegsw.py` after the sanitizer build. Ordinary and sanitizer results are separate from console testing.

Private host receipts: `vita-release-2026-10-11/input-alias-v074/validation.json` and `vita-release-2026-10-11/sanitizers-input-alias-v074/verification.json`. The [published source manifest](https://github.com/bee-san/Rubi_for_vita/releases/download/v0.7.4/source.json) retains exact revisions, environment and validation counts.

**Package audit and installed bytes, HW07:** [exact-tag host/native CI](https://github.com/bee-san/Rubi_for_vita/actions/runs/38130337978) passes, including its kernel instruction gate: 7,720 ARM instructions and no FPU/NEON. Independent download verification checks all 38 manifest files, asset hashes and ZIP/VPK/gzip integrity; packaged native ABI audit finds 22 kernel exports, 14 Shell imports and 9 worker imports. Indexed, opaque settings artwork passes. The package retains local OCR enabled by default, FL10/H15, plain JMdict, per-game settings and disabled Anki/queue behavior. Private package receipt: `vita-release-2026-10-11/verify-v074-independent/verification.json`; the instruction count is attributed to tag CI and the published source manifest.

The device update changes only the 24,807-byte kernel. Full-file readback matches the actual tagged-build SHA-256 `9f6591373b15fabab21be4d151237fbec49968fa65133e22dd54ba952856e10a`; the existing Shell and game-worker payloads already match the release and remain unchanged, as do the settings VPK members. taiHEN configuration (1,983 bytes) and overlay settings (2,886 bytes) are byte-identical before/after. The install receipt predates the accepted reboot request at 10:05:43 UTC on October 11. Private receipt: `vita-agent-use/agent/api13-testing-20261011/native-v074-install-verification.json`. Upload/readback establish installed bytes, not executed hook behavior.

**Fresh native boot and attempted input check, HW08:** the supported native agent reconnects at 10:06:08 UTC; the 24.8-second-uptime snapshot reports API 13 and both kernel/Shell modules in native state 6. CLANNAD launches and the foreground title query matches it, but the system UI is reported as overlaid. A fresh screenshot attempt fails with `frame_unavailable`, code `-7`, at stage `jpeg_encode`; no JPEG is produced. The older `pool` failure belongs to the v0.7.3 session and does not explain this new encoding failure.

One title-bound L+R macro is attempted. `macro.acquire` is rejected with `0xBFFFFFFB` (`-1073741829`) before `input.submit`; **zero L+R input is submitted**. Pre/post-attempt kernel log bytes are identical and contain no fresh trigger or block record. **Inferred:** the return is consistent with the agent's `VAU_STALE(-5)` after bit 30 is cleared and a possible controller-routing mismatch; that attribution comes from the agent contract and routing context, not the native error database. It establishes no controller-neutralization or no-dialogue-advance result. No PS tap is sent: the supported input format/client whitelist excludes PS and system buttons. No screenshot retry, menu navigation or save/load test is credited.

One fresh automatic OCR observation has zero detected lines and `rc=0 io=0 cleanup=0`, with model load 0 ms. It establishes clean completion of that empty job, not fresh model inference or recognition accuracy. The seven text-bearing v0.7.3 jobs in HW02 remain the actual model-load/inference evidence. The native receipt at 10:08:45 UTC records no pending device effect. Private locator: `vita-agent-use/agent/api13-testing-20261011/native-v074-verification.json`.

**Measured offline error lookup, HW09:** `psp2-err` 0.1.0 at `5fef5fb55cebc82ce90185acb0b55d49cc0fdf18` decodes the rejected routing return as an error with `critical=false`, reserved facility `0xFFF`, code `0xFFFB`, and **no exact database match**. It does not supply a Sony/native symbolic name or prove a universal syscall transformation. From the agent repository checkout, the JSON-producing command is (exit 1 for this unmatched code):

```sh
agent/tools/psp2_err/psp2_err --json 0xBFFFFFFB
```

This local lookup makes no console call. Private derived receipt: `vita-agent-use/agent/api13-testing-20261011/v074-routing-error-decode.json`. [[25-Vita-debugging-tools]] retains the tool versions, earlier native error lookups and historical `psp2-core-parse` work; HW04 below retains the supplied-image limits. No new dump or screenshot is attributed to this failed input attempt.

## Resumed game, memory failure and recovery

**Observed physical input, HW10:** after the game resumes, two user-operated L+R openings produce fresh hook-7 records with `in=00000C00 out=00000000`, `LR=0/0`, `L1R1=0/0` and `write=00000000`. Full-block samples also show zero digital output and shoulder pressures, with successful reads/copies. The agent submits no button input. This supersedes HW08's lack of fresh hook evidence; it establishes containment at this game's hooked API, not a visual comparison or proof that every title behaves identically.

**Observed allocation failure, HW11:** six game-OCR records complete with `rc=0 io=0 cleanup=0`; three contain one detected line and actual FL10 model loading/inference. Five later result jobs fail with `-103` while Shell PAF free space is 355,256–441,080 bytes. The local pipeline requests 262,144 bytes for two result arenas while retaining a 1,572,864-byte UI reserve. The guard correctly refuses the request. This is a Shell result/display allocation failure after successful game OCR; the old UI incorrectly labels it a text-recognition failure and suggests shrinking the game region.

**Inferred cause with recovery evidence, HW12:** the installed-style testing agent's `frames_shell.c` allocates a 2,187,264-byte PAF JPEG buffer. A `jpeg_encode` failure leaves it ready/allocated; ordinary release clears the borrowed flag without freeing the output pool. The native wrapper's only active explicit cleanup is reboot. The raw framebuffer belongs to Shell and must not be freed by a repair. This identifies retained output-buffer ownership, not a new OCR model failure or a coredump diagnosis.

One native cold reboot is accepted at 10:14:06 UTC and reconnects at 10:14:36 UTC. With **zero screenshot requests afterward**, fresh PAF free space is 2,538,568 bytes; the 262,144-byte result allocation succeeds and an initial zero-line job finishes `rc=0`. This strongly supports the retained-pool explanation, though other allocations can vary during reboot. The bounded observation ends at 10:17:29 UTC with CLANNAD reported foreground but system UI still overlaid. No additional macro, screenshot, controlled dialogue lookup or visible UI result is credited.

A private testing-agent lifetime repair is prepared separately. It frees the JPEG output pool after failed encoding or after a completed response, while retaining ownership on failed cleanup and leaving the borrowed Shell framebuffer alone. All eight lifecycle fixtures pass under GCC and Clang ASan/UBSan; the changed translation unit builds for ARM EABI5. The complete native agent build remains blocked by an existing SDK prototype conflict in `capture_kernel.c`. The repair is **not deployed or published**, so the hardware checks avoid further screenshot requests. No new crash dump is generated for this investigation.

The current OCR memory contract is not 256 KiB: FL10 weights are 657,664 bytes. At 917×125, the reported complete game workspace is 900,896 bytes, rounded to a 901,120-byte USER allocation. The recognizer scratch alone is 124,832 bytes; model weights, region pixels, runtime structures, stacks and kernel staging are separate costs. Dictionary result arenas are separate Shell allocations. The whole disk-backed JMdict file is never loaded into RAM.

**Observed memory-mode boundary:** a native readback backs up CLANNAD's 1,328-byte `param.sfo` and finds integer `ATTRIBUTE2=0`. No metadata is modified. A single 100,083 µs native performance sample completes but samples Shell controller routing: USER_RW total 52,428,800 bytes and free 98,304 bytes, with metric error 0. These are Shell figures, not CLANNAD's budget or global free physical memory. The client reports an invalid PC-server reply, while the preserved native completion proves the watch ended; no duplicate measurement is issued. rePatch is loaded; no LowMemMode registration appears. A 333 MiB game mode is neither enabled nor verified by these checks.

Private derived evidence: `vita-agent-use/agent/api13-testing-20261011/native-v074-verification.json`, the fresh `overlay-v074-memory-error-status.txt` and `overlay-v074-recovery-first-*` logs, and `memory-mode-readonly/native-memory-measurement.json`. Raw dialogue, device IDs, native responses and SFO bytes remain private.

## Character selection and bounded dictionary results

[Overlay v0.7.5](https://github.com/bee-san/Rubi_for_vita/releases/tag/v0.7.5) follows manual selection rather than whole-sentence tokenization: left/right selects each Unicode character, including a position with no match. Lookup returns at most one entry beginning at that position, choosing the longest source prefix, exact hits before inflected hits, then configured dictionary order. Inflection remains useful without deciding all word boundaries in advance. Cursor-only work reuses the immutable recognized sentence; it does not submit another capture or run the OCR model.

The selected-result path is tested with both VJDICT formats, supplementary characters, newlines, invalid cursor boundaries, first-posting priority, repeated match/no-match movement and injected read failures. In the host sizing fixture, a maximum 8,192-byte entry needs 53,584 bytes with static saved text, or 61,776 bytes when the 4,095-byte input is copied twice. Both fit in a 65,536-byte arena. These are host arena measurements, not a total Vita process-memory measurement.

The Shell allocation is one 64 KiB local result arena instead of two 128 KiB arenas; local background/OCR-only jobs allocate no dictionary-result buffer. Bounded static sentence copies remain separate costs, and the PAF reserve stays at 1.5 MiB. Dictionary allocation failure preserves recognized text and applies dictionary backoff. This reduces the specific Shell result requirement rather than changing a game's memory mode. [PR30](https://github.com/bee-san/Rubi_for_vita/pull/30) merges tested head `41925bb443df2cf4400048770ecc74fe3c3f829f` into main/tag `89be9e761042419191cf8af1dfd0c6e44d1f18b9`; both have tree `e83b912a50835a4ea8534703d2c65225cb57f41b`. Two independent source reviews find no remaining blockers. Frozen Linux/Clang 22.1.8 ASan/UBSan/LSan checks pass all 19 suites, 216 C cases and 41 tooling cases with no failures, skips or reports. The sizing case prints its two measurements before its final pass marker; count it once when parsing the log.

Nine illustrative JMdict positions are repeated with the final sanitizer binary, one warmup plus five measured runs each: all output digests match, no position returns more than one entry, and the maximum arena peak is 54,000 bytes. The punctuation no-match performs zero file I/O. These are host lookups, not a screenshot CER test or Vita timing. Native ELF BSS grows 8,292 bytes (159,476 to 167,768), with data unchanged at 132 bytes. The requested dynamic local-result block shrinks by 196,608 bytes; code/static growth, alignment and other users prevent treating that as a measured total-free-memory increase.

Exact [PR CI](https://github.com/bee-san/Rubi_for_vita/actions/runs/38132855761), [main CI](https://github.com/bee-san/Rubi_for_vita/actions/runs/38133070224) and [tag CI](https://github.com/bee-san/Rubi_for_vita/actions/runs/38133085777) pass host/native gates. The tagged kernel retains API13, ABI22/14/9 and 7,720 ARM instructions without FPU/NEON. The complete package now has 40 manifest files, including both compact memory receipts; installed settings-app members are byte-identical to v0.7.4 despite the VPK archive timestamp changing. Native installation and fresh dialogue/cursor behavior are recorded separately below.

## v0.7.5 native installation and runtime boundary

**Verified deployment, HW13:** the 146,813-byte changed Shell plugin is uploaded through the existing paired native agent and fully read back at 10:51:27 UTC. Its SHA-256 is `d29236fcd9e64216f12178497df10bfaf4be0b3436f6342e26920982f734300c`, matching the independently downloaded release. Both taiHEN and overlay configuration files remain byte-identical to fresh backups. The kernel, game OCR worker, installed settings-app members, weights and dictionary remain unchanged. All five published assets, ZIP/VPK/gzip integrity, 40 manifest members, pinned models/plain JMdict/provenance/license, API13/ABI22/14/9 and local defaults/no queue pass the independent download audit.

A single native cold reboot is accepted at 10:51:58 UTC; the original paired server reconnects at 10:52:44 UTC. CLANNAD is launched once and confirmed at 10:53:02 UTC. Fresh API13 kernel/Shell startup and all three native plugin-state-6 records establish the new boot with `*KERNEL`, `*main` and `*ALL` registrations.

**Historical initial empty job, HW14:** one fresh FL10 background OCR job uses the reported 900,896-byte workspace, finds zero lines and finishes `rc=0 io=0 cleanup=0`. Its result arena peak is 0, consistent with the new OCR-only path allocating no dictionary-result arena. Zero lines mean no model weights/inference or dialogue lookup is proved by this job. The post-launch snapshot at 10:53:25 UTC still reports system UI over CLANNAD. No input lease, L+R/right macro, screenshot or performance measurement is attempted before the initial 10:54:05 UTC receipt. Fresh lookup and dialogue behavior are subsequently established by HW15.

Private receipts: `vita-agent-use/agent/api13-testing-20261011/native-v075-backups/verification.json`, `native-v075-install/installation-verification.json` and `native-v075-install/verification.json` (connected, pending command null at 10:54:05 UTC). The independent published audit is `vita-release-2026-10-11/verify-v075-independent/verification.json`; the immutable sanitizer receipt is `sanitizers-local-cursor-v075/verification.json`. Raw dialogue, native responses, pairing/device identifiers and binaries stay private.

**Physical dialogue and character selection, HW15:** after the user confirms dialogue is visible, the 10:56:43 UTC native snapshot reports CLANNAD foreground and system UI absent. Seven text-bearing FL10 jobs in the fresh boot use the 900,896-byte USER workspace and finish `rc=0 io=0 cleanup=0`, with model loads 213–359 ms and recognition 69–145 ms. These varying-dialogue observations establish actual inference, not a controlled latency benchmark or accuracy measurement. Four successful local result allocations request 65,536 bytes while retaining the 1,572,864-byte PAF reserve; observed lookup peaks are 53,488 bytes.

Six user-operated overlay windows record 4, 3, 7, 12, 7 and 9 cursor moves. Four windows contain zero OCR jobs throughout, establishing dictionary lookup from saved text during physical navigation. Fresh trigger/full-block records retain mapped shoulder neutralization. The user's "works great" report confirms the visible workflow; the subsequent poor-OCR report is qualitative and supplies no hand-checked CER or correspondence transcript.

One title-bound native L+R pulse submitted at 10:57:21 UTC finishes with two applied events, error 0 and a neutral final pad. Its append-only log delta contains an automatic text-bearing OCR job, but no new trigger/input-block or lookup record. API completion therefore does not establish that injected input opened the overlay. No right-tap or close macro is submitted. The runner releases its lease, the pending effect is null, and there are zero screenshots or performance queries. Private derived receipt: `vita-agent-use/agent/api13-testing-20261011/native-v075-install/physical-dialogue-verification.json`, recorded at 10:59:09 UTC. The working release is preserved while the earlier Meiki build is investigated; no Meiki model switch is credited here.

## What the supplied firmware ELFs add

**Measured, HW04:** `psp2-core-parse` 0.2.0 at `3b1219cf403c3a4a9ff7e8b1085ac0ead90ff52c` was used with the supplied SceShell and SceLibKernel images against the same three completed **October 10 Shell dumps** described in [[25-Vita-debugging-tools]]. Embedded runtime names, module NIDs (`0x5549BF1F` and `0x016A1B6F`), versions and load geometry match all three dumps. Matching geometry alone is not full byte identity:

| Historical dump | SceShell captured RX / differing bytes | SceLibKernel captured RX / differing bytes |
|---|---:|---:|
| First | 200,704 / 0 | 9,496 / 653 |
| Second | 200,704 / 88 | 8,192 / 604 |
| Third | 204,800 / 143 | 7,048 / 33 |

Differences require relocation/hook interpretation. Relevant caller windows are absent from the captured memory. The actual crashing modules are pngshot/ScePaf; neither corresponding image was supplied, so backtraces stop at frame 0 and provide **zero verified unwind transitions**. Supplying Shell/LibKernel images does not prove either historical crash's source-level cause.

Both supplied images have zero ELF sections and no symbol tables. Image-backed disassembly reported `available` but emitted no target instructions through the ordinary section-based route. Four bounded 64-byte raw PT_LOAD caller windows were disassembled privately with VitaSDK objdump, yielding 26, 23, 23 and 27 instruction lines. These are image-derived windows, not executed traces or current OCR RPC stubs; they establish no firmware exception-handler or syscall status-transformation mechanism.

Example analysis form, using the actual runtime names and matching private files:

```sh
agent/tools/psp2_core_parse/psp2-core-parse analyze agent/private-debug/old-shell.psp2dmp \
  --image SceShell=agent/private-debug/SceShell.elf \
  --image SceLibKernel=agent/private-debug/SceLibKernel.elf --json
```

Private derived evidence and commands: `vita-agent-use/agent/private-debug/elf-compat-20261011/derived-findings.json` and `commands.json`. Firmware images, extracted windows, raw dump data and disassembly are not published.

## Next discriminating checks

- Compare recognized text with the actual displayed dialogue; record a bounded result before any accuracy claim.
- Return physically to actual game dialogue and verify foreground/system-UI routing before any title-bound input. HW10 already establishes `0xC00` → `0` with zero pressures and a successful copy. Obtain a visual comparison or user confirmation before claiming that L+R no longer advances dialogue. Treat this separately from the earlier × selection leak in [[24-Overlay-selection-input-leak]].
- Preserve the physically confirmed character workflow and 64 KiB allocation from HW15. [[27-Meiki-hardware-and-project-lessons]] now records Meiki's game-process/API compatibility and model visibility in CLANNAD. The next check is the new Settings picker and Japanese-only navigation on hardware after verified installation; do not use the old Shell-based branch as a recognizer switch.
- Preserve the successful API 13 native log and installed revision while testing; verify further captures, cleanup and user-visible behavior with fresh evidence.

The older device snapshot remains history in [[02-Current-device-state]]. Tool diagnostics, host regression tests, package readback, native boot and visible dialogue correspondence remain distinct claims.
