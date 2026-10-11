---
title: Game-process OCR hardware checkpoint
created: 2026-10-11
updated: 2026-10-11
status: native-ocr-completes-trigger-input-leak-open
tags: [vita, ocr, hardware, debugging, input]
---

# Game-process OCR hardware checkpoint

**Observed:** the physical Vita boots API 13, loads the kernel/Shell/game OCR plugins, and completes text-bearing FL10 OCR jobs with `rc=0 io=0 cleanup=0`. More captures complete afterward. This resolves the observed API 12 model-polling/retained-ownership failure for the tested session. **Remaining defect:** the user reports that opening the overlay with L+R advances game dialogue. Recognition accuracy, CER, broad compatibility and an input-leak repair are not established by this checkpoint.

## Exact release and native evidence

[Overlay v0.7.3](https://github.com/bee-san/vita-jp-overlay/releases/tag/v0.7.3) is tagged at `24a3c711c24994e6b466930c2087f8dcdc010d32`. Its independent published-asset audit verifies API 13, all 38 manifest files, pinned FL10/H15 and JMdict, settings artwork, local defaults and no Anki queue. That package result is separate from the native observations below.

| ID | Evidence inspected | What it establishes |
|---|---|---|
| HW01 | Native plugin response collected at 09:43:58 UTC on 2026-10-11 | Complete 20-entry inventory, process-query error 0; overlay kernel under `*KERNEL`, Shell under `*main`, OCR worker under `*ALL` all loaded, individual query errors 0. |
| HW02 | `overlay-api13-boot-retry.log`, final API 13 section beginning at line 1841 | API 13 startup and worker-start marker, followed by 15 successful game-vocr records. Seven have one detected line and actual model load/inference; eight have zero lines. All 15 have `rc=0`, `io=0x00000000`, `cleanup=0`. No capture-request failure appears in this final section. Earlier failed API 11/12 records in the same cumulative file remain historical. |
| HW03 | User report during this physical test | L+R opening the overlay also advances dialogue. Keep this open even though OCR cleanup/capture recovery succeeds. |

HW02 is 115,209 bytes, SHA-256 `fe34c35029bc39d78a7b4539cbd308379d9203e314e972f6c2f0ae526998d9d7`. Private locator: `vita-agent-use/agent/api13-testing-20261011/overlay-api13-boot-retry.log`. Raw dialogue, screenshots and native response JSON stay private.

**Measured session values:** all records use FL10 and a 917×125 region with 900,896 bytes reported as USER workspace. The seven text-bearing records show line finding 34–38 ms, model load 217–288 ms and recognition 36–126 ms. These are logs from varying dialogue, not a controlled benchmark or a measured process/console peak. Zero-line jobs do not prove model loading or recognition. Successful nonempty recognition still needs comparison with the actual displayed text before claiming accuracy; no hand-checked CER or screenshot comparison is reported here.

The native agent command path now responds successfully with the saved pairing. No pairing reset is needed or credited. See [[25-Vita-debugging-tools]] and [[09-Device-operations-and-recovery]] for the supported client and exact evidence-collection syntax.

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
- Repair and retest L+R input containment on the physical Vita. Treat this separately from the earlier × selection leak in [[24-Overlay-selection-input-leak]]. No cause or successful fix is attributed here.
- Preserve the successful API 13 native log and installed revision while testing; verify further captures, cleanup and user-visible behavior with fresh evidence.

The older device snapshot remains history in [[02-Current-device-state]]. Tool diagnostics, host regression tests, package readback, native boot and visible dialogue correspondence remain distinct claims.
