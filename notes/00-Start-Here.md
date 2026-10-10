---
title: Vita text extraction investigation
created: 2026-10-10
updated: 2026-10-10
status: active
tags: [vita, native-text, moc]
---

# Vita text extraction investigation

This vault records the native text-hook investigation alongside the OCR project. Open this `notes/` directory as an Obsidian vault, or follow the ordinary Markdown files. It needs no Obsidian plugin. Findings are linked so a future session can recover the reasoning, evidence and limits without rereading the whole conversation.

**Current outcome:** the installed capture/ranking repair boots normally and supplies candidates, but the physical CLANNAD test failed the intended workflow. The user saw rapidly changing hooks labelled 0% and then irrelevant text. No correct selected source following visible dialogue has been verified on the Vita.

**New explicit preference:** opening the overlay should take one OCR screenshot first and show the matching source. The deployed configuration was manual `text_source=hooks`, so it had no OCR reference; its 0% labels were not evidence that the visible text had been compared and rejected. The implemented follow-up is merged in overlay PR #13 as `539c2888`, with exact-head host/Vita CI passing. Its shell upload is in progress; verified installation and boot remain pending. Live production Lens on the host now exactly matches one verified short page from full-frame and cropped inputs; physical OCR and the interaction remain unverified. See [[17-OCR-first-physical-workflow]] and [[02-Current-device-state]].

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
