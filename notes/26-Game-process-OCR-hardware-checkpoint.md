---
title: Game-process OCR hardware checkpoint
created: 2026-10-11
updated: 2026-10-11
status: v074-booted-input-check-blocked-before-submit
tags: [vita, ocr, hardware, debugging, input]
---

# Game-process OCR hardware checkpoint

**Latest checkpoint:** v0.7.4 repairs the mapped L+R controller filter, is released, and its tagged kernel bytes are installed and read back. Fresh native boot is observed. The attempted title-bound L+R test is rejected before any input is submitted, so neither controller neutralization nor visible dialogue containment is established. The earlier v0.7.3 OCR success and reported opening-chord leak remain recorded below. Recognition accuracy, CER, broad compatibility and peak process memory remain unmeasured.

## Exact release and native evidence

**Historical v0.7.3 observation:** the physical Vita boots API 13, loads the kernel/Shell/game OCR plugins, and completes text-bearing FL10 OCR jobs with `rc=0 io=0 cleanup=0`. More captures complete afterward. This resolves the observed API 12 model-polling/retained-ownership failure for that session. The user reports that opening the overlay with L+R also advances game dialogue. That input defect is separate from clean OCR cleanup and capture recovery.

[Overlay v0.7.3](https://github.com/bee-san/vita-jp-overlay/releases/tag/v0.7.3) is tagged at `24a3c711c24994e6b466930c2087f8dcdc010d32`. Its independent published-asset audit verifies API 13, all 38 manifest files, pinned FL10/H15 and JMdict, settings artwork, local defaults and no Anki queue. That package result is separate from the native observations below.

| ID | Evidence inspected | What it establishes |
|---|---|---|
| HW01 | Native plugin response collected at 09:43:58 UTC on 2026-10-11 | Complete 20-entry inventory, process-query error 0; overlay kernel under `*KERNEL`, Shell under `*main`, OCR worker under `*ALL` all loaded, individual query errors 0. |
| HW02 | `overlay-api13-boot-retry.log`, final API 13 section beginning at line 1841 | API 13 startup and worker-start marker, followed by 15 successful game-vocr records. Seven have one detected line and actual model load/inference; eight have zero lines. All 15 have `rc=0`, `io=0x00000000`, `cleanup=0`. No capture-request failure appears in this final section. Earlier failed API 11/12 records in the same cumulative file remain historical. |
| HW03 | User report during this physical test | L+R opening the overlay also advances dialogue. Keep this open even though OCR cleanup/capture recovery succeeds. |

HW02 is 115,209 bytes, SHA-256 `fe34c35029bc39d78a7b4539cbd308379d9203e314e972f6c2f0ae526998d9d7`. Private locator: `vita-agent-use/agent/api13-testing-20261011/overlay-api13-boot-retry.log`. Raw dialogue, screenshots and native response JSON stay private.

**Measured session values:** all records use FL10 and a 917×125 region with 900,896 bytes reported as USER workspace. The seven text-bearing records show line finding 34–38 ms, model load 217–288 ms and recognition 36–126 ms. These are logs from varying dialogue, not a controlled benchmark or a measured process/console peak. Zero-line jobs do not prove model loading or recognition. Successful nonempty recognition still needs comparison with the actual displayed text before claiming accuracy; no hand-checked CER or screenshot comparison is reported here.

The native agent command path now responds successfully with the saved pairing. No pairing reset is needed or credited. See [[25-Vita-debugging-tools]] and [[09-Device-operations-and-recovery]] for the supported client and exact evidence-collection syntax.

## v0.7.4 mapped input repair and deployment

**Implemented and published, HW05:** [overlay PR #28](https://github.com/bee-san/vita-jp-overlay/pull/28), tested head `da861f06e9ed3ee6430018be47d8db9b70df67b8`, is merged and released as [v0.7.4](https://github.com/bee-san/vita-jp-overlay/releases/tag/v0.7.4) at `3f7ae81971db7ca4913d881be9750cfaada4799f`. The tag has the same source tree as the tested head. API 13 and its exports are unchanged.

The physical v0.7.3 input trace records `ReadPositive2` hook 7 receiving `0xC00` while raw L+R is `0x300`. The [pinned VitaSDK contract](https://github.com/vitasdk/vita-headers/blob/e66ebe90b73d1fa4cce005a5b2072cec27322544/include/psp2/ctrl.h) maps the shoulders to L1/R1 in `*2`/`Ext2` exports. The old filter hid only legacy L/R and left pressure bytes untouched. The repair translates mapped shoulders into shared canonical hold/replay state before filtering, then restores their native encoding while preserving genuine external-controller L2/R2. It clears hidden buttons' matching pressure values and all 12 pressure fields during full blocking, preserving timestamps and the SDK's final four reserved bytes. Only an introduced short-tap replay receives pressure 255; ordinary held pressures remain unchanged. Contended reads reserve the trigger components without spinning.

**Measured host regression, HW06:** the production wrappers/filter pass all 14 controller cases, covering all 12 exports before Shell input ownership, positive/negative logic, 64-sample canaries, mixed APIs, real L2/R2, held suppression, short-tap replay, pressure preservation, contention/release, copy failures, raw Shell input and 256 repeated bounded diagnostic calls. The same final harness compiled against v0.7.3 `24a3c71` fails 9/14 cases. Linux x86_64/GCC 16.2.1 with the pinned OCR runtime and released FL10/H15 passes all 18 host suites. These are deterministic boundary tests, not physical input-latency or visible-dialogue measurements.

An independent Linux x86_64/Clang 22.1.8 Debug run uses `-fsanitize=address,undefined -fno-omit-frame-pointer`, leak detection and halt-on-error. All 18 suites, 190 C cases and 41 tooling tests pass with no failures, skips or sanitizer reports; both pinned models exercise the actual game/kernel broker. Three JPEG round trips and five error cases also pass; the JPEG exit-lifetime check sets `LSAN_OPTIONS=use_stacks=0:use_registers=0`. Reproduction commands are recorded in the private verification receipt; the test entrypoints are `ctest --test-dir build/host --output-on-failure` and `python3 tools/test_jpegsw.py` after the sanitizer build. Ordinary and sanitizer results are separate from console testing.

Private host receipts: `vita-release-2026-10-11/input-alias-v074/validation.json` and `vita-release-2026-10-11/sanitizers-input-alias-v074/verification.json`. The [published source manifest](https://github.com/bee-san/vita-jp-overlay/releases/download/v0.7.4/source.json) retains exact revisions, environment and validation counts.

**Package audit and installed bytes, HW07:** [exact-tag host/native CI](https://github.com/bee-san/vita-jp-overlay/actions/runs/38130337978) passes, including its kernel instruction gate: 7,720 ARM instructions and no FPU/NEON. Independent download verification checks all 38 manifest files, asset hashes and ZIP/VPK/gzip integrity; packaged native ABI audit finds 22 kernel exports, 14 Shell imports and 9 worker imports. Indexed, opaque settings artwork passes. The package retains local OCR enabled by default, FL10/H15, plain JMdict, per-game settings and disabled Anki/queue behavior. Private package receipt: `vita-release-2026-10-11/verify-v074-independent/verification.json`; the instruction count is attributed to tag CI and the published source manifest.

The device update changes only the 24,807-byte kernel. Full-file readback matches the actual tagged-build SHA-256 `9f6591373b15fabab21be4d151237fbec49968fa65133e22dd54ba952856e10a`; the existing Shell and game-worker payloads already match the release and remain unchanged, as do the settings VPK members. taiHEN configuration (1,983 bytes) and overlay settings (2,886 bytes) are byte-identical before/after. The install receipt predates the accepted reboot request at 10:05:43 UTC on October 11. Private receipt: `vita-agent-use/agent/api13-testing-20261011/native-v074-install-verification.json`. Upload/readback establish installed bytes, not executed hook behavior.

**Fresh native boot and attempted input check, HW08:** the supported native agent reconnects at 10:06:08 UTC; the 24.8-second-uptime snapshot reports API 13 and both kernel/Shell modules in native state 6. CLANNAD launches and the foreground title query matches it, but the system UI is reported as overlaid. A fresh screenshot attempt fails with `frame_unavailable`, code `-7`, at stage `jpeg_encode`; no JPEG is produced. The older `pool` failure belongs to the v0.7.3 session and does not explain this new encoding failure.

One title-bound L+R macro is attempted. `macro.acquire` is rejected with `0xBFFFFFFB` (`-1073741829`) before `input.submit`; **zero L+R input is submitted**. Pre/post-attempt kernel log bytes are identical and contain no fresh trigger or block record. **Inferred:** the return is consistent with the agent's `VAU_STALE(-5)` after bit 30 is cleared and a possible controller-routing mismatch; that attribution comes from the agent contract and routing context, not the native error database. It establishes no controller-neutralization or no-dialogue-advance result. No PS tap is sent: the supported input format/client whitelist excludes PS and system buttons. No screenshot retry, menu navigation or save/load test is credited.

One fresh automatic OCR observation has zero detected lines and `rc=0 io=0 cleanup=0`, with model load 0 ms. It establishes clean completion of that empty job, not fresh model inference or recognition accuracy. The seven text-bearing v0.7.3 jobs in HW02 remain the actual model-load/inference evidence. The native receipt at 10:08:45 UTC records no pending device effect. Private locator: `vita-agent-use/agent/api13-testing-20261011/native-v074-verification.json`.

**Measured offline error lookup, HW09:** `psp2-err` 0.1.0 at `5fef5fb55cebc82ce90185acb0b55d49cc0fdf18` decodes the rejected routing return as an error with `critical=false`, reserved facility `0xFFF`, code `0xFFFB`, and **no exact database match**. It does not supply a Sony/native symbolic name or prove a universal syscall transformation. From the tool checkout, the successful JSON command is:

```sh
agent/tools/psp2_err/psp2_err --json 0xBFFFFFFB
```

This local lookup makes no console call. Private derived receipt: `vita-agent-use/agent/api13-testing-20261011/v074-routing-error-decode.json`. [[25-Vita-debugging-tools]] retains the tool versions, earlier native error lookups and historical `psp2-core-parse` work; HW04 below retains the supplied-image limits. No new dump or screenshot is attributed to this failed input attempt.

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
- Return physically to actual game dialogue, verify the foreground/system-UI state, then retry the still-unsubmitted bounded L+R test or inspect a user's physical chord. Obtain a visual comparison or user confirmation before claiming that L+R no longer advances dialogue. Submitted `0xC00` → `0` with zero pressures and a successful copy would establish hook containment, not the visible result. Treat this separately from the earlier × selection leak in [[24-Overlay-selection-input-leak]].
- Preserve the successful API 13 native log and installed revision while testing; verify further captures, cleanup and user-visible behavior with fresh evidence.

The older device snapshot remains history in [[02-Current-device-state]]. Tool diagnostics, host regression tests, package readback, native boot and visible dialogue correspondence remain distinct claims.
