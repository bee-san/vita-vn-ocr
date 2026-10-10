---
title: Vita text extraction investigation
created: 2026-10-10
updated: 2026-10-10
status: active
tags: [vita, native-text, moc]
---

# Vita text extraction investigation

This vault records the native text-hook investigation alongside the OCR project. Open this `notes/` directory as an Obsidian vault, or follow the ordinary Markdown files. It needs no Obsidian plugin. Findings are linked so a future session can recover the reasoning, evidence and limits without rereading the whole conversation.

**Current outcome:** the corrected generic SceLibc hooks and source ranking are merged, installed on the physical Vita, read back and normally booted. Kernel API 8, shell registration 0 and STARTED kernel/shell states are verified. The same code captures two visible CLANNAD narration pages in isolated Vita3K without a game-specific profile, with all three matching sources in the top five after the ranking repair.

**Still unresolved:** actual CLANNAD source selection, advancing dialogue and dictionary behavior on the physical Vita. The user has been asked to open CLANNAD, choose matching text with L+R / ▲▼ / ×, then advance and reopen twice; their result is pending. See [[02-Current-device-state]], [[07-CLANNAD-investigation]] and [[16-Full-feed-picker-ranking]].

The deployed implementation is `6eef6c0`; final reviewed evidence head is `69006af`; overlay PR #12 merged as `a5e0c50`. It passed 264 synthetic Vita3K checks and exact final-head hosted CI. These are distinct from the pending physical game test.

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
