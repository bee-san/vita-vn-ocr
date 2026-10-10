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
| E15 | Real CLANNAD execution reached imported SceLibc memmove at `0x810A9458`, caller LR `0x81033D6F`. | Private `runtime-trace.jsonl`: first observed call had `n=0` and source NULL, so this is call-path evidence, not dialogue capture. [[15-Vita3K-runtime-investigation]] |
| E16 | The inspected Vita3K debugger continuation path re-locks the same thread mutex. | `cmd_continue` holds `thread->mutex` then calls `ThreadState::resume`, which locks that nonrecursive mutex again; source revision `b75d5e6292d5caa6a8411bacc9db1eca1531beab`. This explains the observed breakpoint-resume hang in the investigation. |
| E17 | Uninstrumented CLANNAD rendered Japanese startup notices in isolated Vita3K/Xvfb with SDL X11 selected. | Active runtime investigator report; private captures stay local. The inherited Wayland setting had produced misleading black captures. Startup notices are not dialogue validation. |
| E18 | The first visible CLANNAD narration page matches five records emitted through the unchanged production text plugin and emulator test bridge, with no game-specific profile. | Root independently compared private `dialogue-3.png` and `captures-first-run.log` (81,920 B, 506 TEXT records). Four inspected matching call sites are BLX memcpy. This proves a real-page match, not yet a temporally current stream or hardware capture. [[15-Vita3K-runtime-investigation]] |
| E19 | Two different visible narration pages matched immediate source `0x8102CA7F` through generic SceLibc memcpy, without a title profile. | Benchmark at evidence commit `d1ee0dc`; pages 1 and 4 matched 17/13 input bytes and 24/18 UTF-8 bytes. The first helper filtered short/non-Japanese events before ranking, so it did not prove production picker selectability. |
| E20 | Full-feed production ranking hid initial dialogue outside the five displayed choices. | First-page snapshot sequence 425 retained the actual six-normalized-character line in three streams, each displayed rank 0 (outside top five). Update-heavy noise, mixed data and static strings ranked above it. Page 4 sequence 1684 put `0x8102CA7F` at rank 5 with update count 3. [[16-Full-feed-picker-ranking]] |
| E21 | Corrected full-feed ranking displays three matching sources on both checked pages. | Exact comparator revision `6eef6c0`, unchanged production Text SHA `84259631...`, test bridge SHA `106be9d4...`, no title profile. Page 1 seq424 ranks 2/3/4, updates 1; page 4 seq1680 ranks 3/4/5, updates 3. All snapshots complete; selected=0, so manual selection was not exercised. [[16-Full-feed-picker-ranking]] |
| E22 | The final kernel/text pair is installed with full readback and normal physical boot verified. | Final source head `69006af`, merged `a5e0c50`; readbacks kernel `6cc2dc86...` and text `84259631...`; fresh 706 B startup log confirms API 8/registration 0; native kernel/shell STARTED 6. The later physical CLANNAD user test failed the intended workflow (E23); boot success still stands. [[02-Current-device-state]] |
| E23 | The physical user test failed the intended workflow despite candidate traffic. | User saw changing 0% hooks and irrelevant text. Saved status shows 1→5 candidates, full discovery after opening, sequence up to 3,035, selected=0. No correct visible-page match or usable selected source is established. [[17-OCR-first-physical-workflow]] |
| E24 | The installed mode was manual `text_source=hooks`; no OCR reference existed for its 0% labels. | Root confirmed the deployed configuration and acknowledged choosing that mode. Baseline renderer always printed score as match percent; repository default remains `text_source=auto`, which is distinct from the installed override. |
| E25 | The OCR-first candidate uses one temporary 384 KiB arena and passes local ownership/lifecycle regressions. | PR #13 head `5826329`: 10 host suites in 4.18 s, 37 worker cases, local Vita release/ABI passed. Both exact-head hosted jobs passed; merged as `539c2888`. Shell/config readback and normal boot are now verified (E27); the physical OCR result remains pending. Offline memory probes are narrow host measurements, not device success; the separate live host check is E26. [[18-OCR-anchor-memory-measurements]] |
| E26 | Two live host Lens requests exactly matched the same previously verified short visible line. | Full 960×544 and saved 755×146 crop, production JPEG/OCR/BearSSL, HTTP 200, TLS 0, `rc=0`, three verified sources each score 100%. Arena peaks 131,128/93,432 B. Two single samples, one six-codepoint line; no CER, console performance or physical capture claim. [[18-OCR-anchor-memory-measurements]] |
| E27 | OCR-first shell/config are installed, read back and normally booted; CLANNAD foreground launch is confirmed. | Shell SHA `32baceeb...`, config SHA `b7c0dd2e...`, only hooks→auto changed. Fresh startup at 11:27:24 UTC has API 8/registration 0; complete inventory has kernel/shell STARTED 6 and zero errors. The user screenshot/match test is pending. [[02-Current-device-state]] |
| E28 | The physical OCR-first test failed with -108 and no game text. | Exact user report in [[19-Physical-OCR-source-failure]]. `VJO_E_SOURCE=-108` covers multiple screenshot/JPEG paths; generic “unexpected response” wording does not establish a Lens/HTTP error. Precise original stage/cause is unresolved. |
| E29 | Six physical OCR attempts fail before JPEG/Lens because the 2,040 KiB full-frame kernel capture buffer cannot be allocated. | Paired shell/kernel logs: request error `0xC0000002`, job -108/arena peak 0; kernel allocation `0x80024302`. Expected no-memory error propagation remains under review. A region-sized buffer repair is not yet deployed. [[19-Physical-OCR-source-failure]] |
| E30 | The user chose a fixed manual hook list; only auto→hooks changed on the device. | Full 3,083 B config readback SHA `3833843d...`, no reboot/plugin replacement. Physical selection is logged, but the user suspects wrong text. [[20-Fixed-manual-hook-workflow]] |
| E31 | The physical Paf fallback completes local dictionary jobs for the selected text. | Session 12 selected ID 1, kind 2/encoding 0, six bytes/one update; 256 KiB Paf, job rc=0, peak 110,080 B, observed 37/21/25 ms. This does not prove dialogue correspondence or visible dictionary usefulness. [[06-Low-memory-overlay]] |
| E32 | Region-sized capture repair passes local validation and is published in PR #14. | Head `5118c4a`: 10 host suites in 4.61 s; 41 worker/16 production-kernel cases; kernel ASAN/UBSAN; SDK Release/ABI; 6,055 ARM instructions, no FPU/NEON. Two reviews; hosted host passed, Vita pending. Not merged or installed. [[19-Physical-OCR-source-failure]] |

## Inferences and hypotheses

| ID | Claim | Status | What would resolve it |
|---|---|---|---|
| H01 | Correct generic `SceLibc` hooks may expose useful CLANNAD text. | Two visible-page matches and corrected top-five source ranking pass in the emulator; hardware capture remains pending. | Successful hook attachment, bounded observations, then two visible pages matching output. |
| H02 | CLANNAD requires a signed native register profile. | **Unproven.** Earlier suspicion was premature because the generic hooks targeted absent imports. | Reassess only after corrected generic capture and measured scanner diagnostics. |
| H03 | CP932 should be tried before assuming UTF-16 for this title. | Supported by offline font data, still an inference for live dialogue. | Compare captured bytes to visible pages with an explicitly recorded decoder. |
| H04 | Scanner zero candidates means the game stores no readable Japanese text. | Unsupported. | First establish that reads enter the right process and cover readable mappings, then inspect decoding/filter statistics. |
| H05 | The later reported power-off has the same cause as the fixed boot failure. | Unknown. | A new sequence of installed hashes, last markers, module state and crash evidence. Do not reuse the earlier diagnosis by resemblance. |
| H06 | Paf's existing heap will sustain dictionary lookup under physical-page exhaustion. | Physical 256 KiB Paf allocation and local dictionary rc=0 now observed; visible usefulness unverified. | User-visible correct result and dialogue correspondence, beyond allocation/job return code. |

## Claims that are explicitly not established

- Generic native extraction works for arbitrary Vita visual novels.
- The complete physical CLANNAD workflow works, or a correct picker candidate has been manually selected.
- A static matching script string follows the current page.
- Native PSP/Adrenaline text hooks work; that is a different CPU and backend.
- A single partition-memory sample belongs to CLANNAD. Process ownership was uncertain.
- A game launch accepted by the control API proves it remained running.
- Synthetic Vita3K success proves hardware kernel import compatibility or SceShell/Paf behavior.

Related: [[08-Validation-and-measurements]], [[2026-10-10-Investigation-log]], [[13-Note-maintenance]].
