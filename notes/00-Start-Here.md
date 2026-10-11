---
title: Vita text extraction investigation
created: 2026-10-10
updated: 2026-10-11
status: active
tags: [vita, native-text, moc]
---

# Vita text extraction investigation

This vault records the native text-hook investigation alongside the OCR project. Open this `notes/` directory as an Obsidian vault, or follow the ordinary Markdown files. It needs no Obsidian plugin. Findings are linked so a future session can recover the reasoning, evidence and limits without rereading the whole conversation.

**Latest 2026-10-11 checkpoint, 15:53:04 UTC:** overlay PR #38 adds default whole-word/optional character highlighting and Japanese-filtered taps on the recognized header, retaining character navigation without a tokenizer. Exact-head host/Vita CI passed, and all six downloaded stable v0.7.9 assets match their audited candidates. The promotion connection was lost while running at 15:34:40 UTC; its retained status remains uncertain. After the user powered the Vita on, the installed Settings executable, registered native foreground launch and four actual touch/controller highlight actions were verified independently. Full configuration readbacks preserve VOCR, Japanese-only ON and unrelated bytes, ending in Whole word mode. The new Shell file has complete readback after a new boot, but fresh loaded-module proof and game-header taps remain pending. A separate guarded native sorter has 12 actual-coordinator host sanitizer cases and a clean native build for the complete 59-icon plan. The sorter runtime has complete native readback and an accepted reboot request; its fresh activation and physical ACL approval remain pending, with no hardware move proof. See [[28-Touch-lookup-and-LiveArea-sort-checkpoint]] and [[02-Current-device-state]].

**Earlier proven October 11 checkpoint:** VOCR and Meiki have completed actual game-process OCR in CLANNAD, and the user confirms the repaired Square/Circle region workflow works. Daijirin v3 installation completed with native hash verification; the verified configuration selects VOCR, Japanese-only character selection ON and Daijirin first with JMdict fallback, preserving `ocr_mode=on_press`. The reboot accepted at 13:56 UTC has fresh API 13 startup evidence. The installed v0.7.7 Settings app subsequently crashed; two completed native dumps and linked-code analysis identify a libPNG/setjmp ABI mismatch. Repair PR #36 passed host/native CI, merged as `b0642493688c0f31bc33738d937d876ba0edbab6` and shipped as v0.7.8. All six downloaded release assets match their audited candidates. The installed repaired executable matches those bytes; retained promotion status after a disconnect remains uncertain, separately from successful native launch and controls. The same foreground process survived a measured 100.106 seconds. Complete configuration readbacks verify engine choices and a later Japanese-only ON → OFF → ON transition, ending byte-equal to the intended configuration. The initial SELECT/readback discrepancy is preserved separately. The native keep-awake worker is loaded and its first DEFAULT tick succeeded; a 374.066-second command-free connection check passed. Physical OLED visibility during that interval remains unmeasured. Physical font readability and new game OCR/Daijirin lookup remain unproved. The 59-app/seven-page LiveArea plan also remains pending. Hardware accuracy/CER remains unmeasured. See [[27-Meiki-hardware-and-project-lessons]], [[26-Game-process-OCR-hardware-checkpoint]] and [[25-Vita-debugging-tools]]. The native-hook narratives below retain their dated historical boundaries.

**Native-hook outcome (October 10):** the user reports **“omg it works”** on the physical Vita. Fresh logs confirm a selected native CALL source changes through five updates and completes dictionary jobs across reopens. The working source is kind 2 CALL, **not** the installed kind 4 REGISTER profile. Exact visible-page count is not separately stated; see [[23-Physical-native-hook-success]].

**Historical physical issue (October 10):** × selects a hook but also advances the visual novel; the user confirmed it happens on the press, not while idle. The selection path already retains input ownership, and Positive2 is hooked. A controller hardening/diagnostic candidate is published as overlay PR #17, prepared but not installed. The user explicitly stopped that deployment before any upload, app close or reboot; the fix was unverified at that checkpoint. Later controller and OCR releases are recorded in [[26-Game-process-OCR-hardware-checkpoint]]. See [[24-Overlay-selection-input-leak]].

At that October 10 checkpoint, the kernel, diagnostic shell and title profile were installed/read back, normal boot and CLANNAD launch passed, and the user's fixed manual mode was preserved. The new register profile has two-page emulator proof but no physical register-source match yet. OCR was still physically unverified after that capture repair; later actual runs are recorded in the latest checkpoint above. See [[02-Current-device-state]] and [[22-CLANNAD-register-profile-proof]].

Earlier wrong-text, memory and boot failures remain part of the record. User-reported usefulness plus logged selected-source follow/lookup is a stronger result than installation alone, while broader scenes and compatibility remain unmeasured.

The earlier repair established install/boot and a two-page emulator capture/top-five result; the later user report and selected CALL metadata now add a working physical experience. It is implementation `6eef6c0`, reviewed head `69006af`, merged as `a5e0c50`, with 264 passing synthetic Vita3K checks. The stronger physical UX failure is now part of the record.

## Start a session

1. Read [[02-Current-device-state]], then [[12-Open-questions-and-next-steps]]. These are time-sensitive snapshots, not a promise that the device is still in the same state.
2. Read [[10-Repositories-and-revisions]] before changing branches. Several worktrees belong to concurrent work.
3. Follow [[09-Device-operations-and-recovery]] for device access; preserve pairing and unfinished operation IDs.
4. Add experiments using [[Experiment-template]], then update [[01-Evidence-and-claims]] and the relevant topic. Follow [[13-Note-maintenance]].

## Map of content

| Topic | Note |
|---|---|
| What is established, inferred or still unknown | [[01-Evidence-and-claims]] |
| Installed artifacts, normal boot and remaining failure | [[02-Current-device-state]] |
| Capture path and memory bounds | [[03-Native-text-architecture]] |
| Firmware 3.65 unresolved import and boot shutdown | [[04-Firmware-365-boot-ABI]] |
| Kernel syscall stack correction | [[05-Kernel-syscall-stack]] |
| Picker/raw text under memory pressure | [[06-Low-memory-overlay]] |
| CLANNAD's actual imports and possible encodings | [[07-CLANNAD-investigation]] |
| Corrected generic SceLibc hooks and validation | [[14-Retail-libc-hook-coverage]] |
| Real-game startup trace and emulator tooling failures | [[15-Vita3K-runtime-investigation]] |
| Full-feed source-ranking failure and validation gate | [[16-Full-feed-picker-ranking]] |
| Physical UX failure and requested OCR-first workflow | [[17-OCR-first-physical-workflow]] |
| Measured Lens anchor workspace and its limits | [[18-OCR-anchor-memory-measurements]] |
| Physical OCR -108 failure and confirmed capture allocation failure | [[19-Physical-OCR-source-failure]] |
| Current fixed manual list, selection and wrong-text concern | [[20-Fixed-manual-hook-workflow]] |
| Merged picker diagnostics and screenshot-runtime uncertainty | [[21-Picker-diagnostics-and-screenshot-provenance]] |
| Two-page native register profile proof and rejected candidates | [[22-CLANNAD-register-profile-proof]] |
| User-reported physical success and actual CALL-source evidence | [[23-Physical-native-hook-success]] |
| × selection reaching the game and pending controller hardening | [[24-Overlay-selection-input-leak]] |
| What tests establish, and what they cannot | [[08-Validation-and-measurements]] |
| API 13 native OCR, repeated captures and character lookup | [[26-Game-process-OCR-hardware-checkpoint]] |
| Meiki native runs, verified dictionary/Settings repair and pending font/game/LiveArea checks | [[27-Meiki-hardware-and-project-lessons]] |
| Verified Settings highlight controls, pending game touch and guarded native sorting | [[28-Touch-lookup-and-LiveArea-sort-checkpoint]] |
| Error lookup, CPU dumps and live native debugging | [[25-Vita-debugging-tools]] |
| Safe operational commands and recovery | [[09-Device-operations-and-recovery]] |
| Worktrees, commits and merged PRs | [[10-Repositories-and-revisions]] |
| Source and private-artifact register | [[11-Sources-and-artifacts]] |
| Next decision points | [[12-Open-questions-and-next-steps]] |
| Editing this notebook without losing history | [[13-Note-maintenance]] |
| Investigation chronology | [[2026-10-10-Investigation-log]] |

## Evidence boundary

A working physical CLANNAD experience is user-reported and supported by native follow/lookup metadata; broad commercial-game compatibility, capture latency, console peak-memory and real-screenshot CER remain unmeasured. Host tests, synthetic Vita3K tests, correct installation and a started module are useful but separate pieces of evidence. The user did not separately enumerate page count, and no screenshot comparison was obtained; keep those limits distinct from the successful native metadata.

Only notes, small derived tables, hashes and links belong here. Game executables, screenshots, dialogue, fonts, RAM dumps, licenses, device identity, credentials and network details remain outside the repository. Private local links are deliberate references, not attachments.
