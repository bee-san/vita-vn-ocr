---
title: Vita text extraction investigation
created: 2026-10-10
updated: 2026-10-10
status: active
tags: [vita, native-text, moc]
---

# Vita text extraction investigation

This vault records the native text-hook investigation alongside the OCR project. Open this `notes/` directory as an Obsidian vault, or follow the ordinary Markdown files. It needs no Obsidian plugin. Findings are linked so a future session can recover the reasoning, evidence and limits without rereading the whole conversation.

**Current outcome:** the user chose a fixed manual hook list after the OCR capture failure. The installed shell remains normally bootable; only the configuration changed back to `text_source=hooks`, with full readback verified. A source was selected and physical Paf dictionary jobs completed, but the user thinks the chosen text is wrong. Correct visible dialogue and following the next page remain unverified. See [[20-Fixed-manual-hook-workflow]].

The earlier OCR-first path is implemented and installed, but its physical request fails when allocating the full-frame 2,040 KiB kernel capture buffer, before JPEG or Lens. The region-sized capture repair and bounded picker diagnostics are merged through `5e5bc25`, with combined artifacts prepared but not installed. Live production Lens on the host exactly matched one short verified page; that result bypassed the failing kernel allocation. See [[19-Physical-OCR-source-failure]] and [[18-OCR-anchor-memory-measurements]].

The previous repair remains a verified install/boot result and a two-page emulator capture/top-five result, not a completed physical text-hook experience. It is implementation `6eef6c0`, reviewed head `69006af`, merged as `a5e0c50`, with 264 passing synthetic Vita3K checks. The stronger physical UX failure is now part of the record.

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
| What tests establish, and what they cannot | [[08-Validation-and-measurements]] |
| Safe operational commands and recovery | [[09-Device-operations-and-recovery]] |
| Worktrees, commits and merged PRs | [[10-Repositories-and-revisions]] |
| Source and private-artifact register | [[11-Sources-and-artifacts]] |
| Next decision points | [[12-Open-questions-and-next-steps]] |
| Editing this notebook without losing history | [[13-Note-maintenance]] |
| Investigation chronology | [[2026-10-10-Investigation-log]] |

## Evidence boundary

No commercial-game compatibility, capture latency, console peak-memory or real-screenshot CER number is established here. Host tests, synthetic Vita3K tests, correct installation and a started module are useful but separate pieces of evidence. Success requires the selected source to follow at least two actual dialogue pages, then a broader validation pass.

Only notes, small derived tables, hashes and links belong here. Game executables, screenshots, dialogue, fonts, RAM dumps, licenses, device identity, credentials and network details remain outside the repository. Private local links are deliberate references, not attachments.
