---
title: Evidence and claim ledger
created: 2026-10-10
updated: 2026-10-10
status: active
tags: [vita, evidence]
---

# Evidence and claim ledger

Use this ledger to prevent a partial result becoming a compatibility claim. Evidence IDs are stable: append corrections instead of silently changing what an earlier experiment established.

## Established observations

| ID | Observation | Evidence and scope |
|---|---|---|
| E01 | Physical firmware is 3.65; the device is a PCH-1001. | Recorded runtime inspection in the firmware and memory benchmark reports. A spoofed version is not the actual ABI. [[04-Firmware-365-boot-ABI]] |
| E02 | The failing kernel was `READY` (2); the shell reached its first custom API call with an unresolved import. | Runtime binding words and native module state, preserved in the boot benchmark. Presence alone did not establish successful startup. |
| E03 | The strong 3.60 `SceSysmemForKernel` memory-type import was incompatible with 3.65. | Runtime binding plus SDK 3.60/3.63 NID comparison. [[04-Firmware-365-boot-ABI]] |
| E04 | Corrected kernel and shell booted normally, API 8 returned, registration returned 0, modules reached `STARTED` (6). | Physical postboot logs and plugin inventory; full installed-file readback. [[02-Current-device-state]] |
| E05 | Original text-submit nested compiled frames totalled 4,840 B; changed frames totalled 2,784 B. | GCC `-fstack-usage`, excluding OS overhead. Static measurement, not device stack high-water. [[05-Kernel-syscall-stack]] |
| E06 | Opening the old overlay in CLANNAD failed its 786,432 B result allocation with `0x80024302`. | Shell status plus user-visible memory error; decoded as no free physical page. [[06-Low-memory-overlay]] |
| E07 | The changed shell booted and the user could open the picker without that memory error. | Installed shell readback, postboot log and user report. This does not prove Paf dictionary fallback ran. |
| E08 | The picker remained empty after dialogue advanced; live metadata reported zero candidates through repeated scan passes. | User report and `status-clannad-02.txt`; scanner addresses changed but successful game-memory reads were not separately established. [[07-CLANNAD-investigation]] |
| E09 | CLANNAD's inspected main module imports none of the four generic NIDs originally targeted. | Offline loaded ELF with 250 function imports; import metadata. Four equivalent copy/string functions are imported from `SceLibc`. |
| E10 | No shared PVF/font import appeared in that main-module import inventory. | Same inventory. This says nothing by itself about dynamically loaded modules or every rendering path. |
| E11 | The inspected font codepoint table has 7,120 entries and uses CP932-style values; `0x82A0` occurs at index 368 while Unicode `0x3042` is absent. | Root investigator's offline asset inspection. No font or game text copied here. This is encoding evidence, not a proven live dialogue representation. |
| E12 | Original installed `eboot.bin` is an encrypted SELF; the later research ELF contains decrypted, relocated loaded segments. | Distinct hashes, dump boundary and private research metadata. They must never be confused. [[07-CLANNAD-investigation]] |
| E13 | PRs 10 and 11 are merged. | GitHub API queried 2026-10-10; exact merge revisions in [[10-Repositories-and-revisions]]. |
| E14 | Corrected generic hooks pass 255 isolated Vita3K checks, including actual retail-library alias imports. | Production-plugin test result and benchmark at `0a660159...`; 43 portable + 192 observer + 20 plugin checks, zero failures. Draft PR #12, no physical deployment/capture claim. [[14-Retail-libc-hook-coverage]] |

## Inferences and hypotheses

| ID | Claim | Status | What would resolve it |
|---|---|---|---|
| H01 | Correct generic `SceLibc` hooks may expose useful CLANNAD text. | Implemented and synthetic tests pass; real-game capture remains pending. | Successful hook attachment, bounded observations, then two visible pages matching output. |
| H02 | CLANNAD requires a signed native register profile. | **Unproven.** Earlier suspicion was premature because the generic hooks targeted absent imports. | Reassess only after corrected generic capture and measured scanner diagnostics. |
| H03 | CP932 should be tried before assuming UTF-16 for this title. | Supported by offline font data, still an inference for live dialogue. | Compare captured bytes to visible pages with an explicitly recorded decoder. |
| H04 | Scanner zero candidates means the game stores no readable Japanese text. | Unsupported. | First establish that reads enter the right process and cover readable mappings, then inspect decoding/filter statistics. |
| H05 | The later reported power-off has the same cause as the fixed boot failure. | Unknown. | A new sequence of installed hashes, last markers, module state and crash evidence. Do not reuse the earlier diagnosis by resemblance. |
| H06 | Paf's existing heap will sustain dictionary lookup under physical-page exhaustion. | Host behavior validated; physical success not observed. | Actual local lookup on the device with allocation-path evidence. |

## Claims that are explicitly not established

- Generic native extraction works for arbitrary Vita visual novels.
- CLANNAD dialogue extraction works, or a correct picker candidate has been selected.
- A static matching script string follows the current page.
- Native PSP/Adrenaline text hooks work; that is a different CPU and backend.
- A single partition-memory sample belongs to CLANNAD. Process ownership was uncertain.
- A game launch accepted by the control API proves it remained running.
- Synthetic Vita3K success proves hardware kernel import compatibility or SceShell/Paf behavior.

Related: [[08-Validation-and-measurements]], [[2026-10-10-Investigation-log]], [[13-Note-maintenance]].
