---
title: Fixed manual hook workflow
created: 2026-10-10
updated: 2026-10-10
status: later-native-call-success-reported
tags: [vita, native-text, manual, user-preference]
---

# Fixed manual hook workflow

The user's latest choice is a **fixed manual hook list**. After the OCR capture failure, the user suggested optional downscaling/black-and-white, then explicitly said ordinary text hooks without OCR are acceptable if the list does not spam or constantly change. They selected the fixed manual-list option. This supersedes an assumption that OCR is mandatory for the next usable result; it does not invalidate the earlier requested OCR-first implementation or its measured capture failure.

**Later result:** after the combined install, the user reports the manual workflow works. A different selected CALL source (ID 18 in session 6) follows changing text and completes lookups; the earlier six-byte source below remains a failed/suspect historical choice. The successful source is not the installed register profile. See [[23-Physical-native-hook-success]].

## Verified configuration change

Root changed only `text_source = auto` to `text_source = hooks`, guarding the currently installed config hash, waiting for transfer completion and verifying the complete readback. The 3,083 B file has SHA-256 `3833843d30aa9bd2bc294149d6c63b90f2909ce679ea05a2b2b731e9c101eabd`. The notes author independently hashed that local readback and compared it with the pre-change config. Lens and the saved region remain available for future use, but manual opening does not run OCR.

This setting takes effect on the next overlay open; no reboot or plugin replacement was required. The config-only change initially kept PR #13 shell `32baceeb...`; the later combined install replaces it with `4f1f0650...`, preserving this manual mode and stable picker behavior. This is distinct from the old shell's constantly republished, scanning list that originally frustrated the user.

## Observed manual selection and dictionary job

Private `status.manual-start.txt` shows session 12 selecting source ID 1 at sequence 475 with kind 2, encoding 0, six bytes and update count 1. It reports `native dictionary workspace: Paf heap, 256 KiB`, then a manual-selection job completing `rc=0` in 37 ms, arena peak 110,080 B. Later reopens at sequences 1,787 and 3,046 complete native-text jobs in 21/25 ms with the same arena peak. These are isolated observed job durations, not a latency distribution or performance guarantee.

The selected source remains six bytes with update count 1 in the recorded samples. Other source-table sequences advance, but that alone does not show the selected source follows current dialogue. The user thinks the chosen text is wrong. Root cannot visually confirm it because the device screenshot tool returns `frame_unavailable`, code -7, stage `jpeg_encode`, and has asked for a short example. The installed runtime has since been identified as the stock private SceShell JPEG backend; code -7 still has several possible branches and does not diagnose a single encoder cause. See [[21-Picker-diagnostics-and-screenshot-provenance]]. Do not quote private game text into these notes when comparing it.

**Established:** a source can be selected, the manual list is no longer driving the earlier repeated OCR failure, and the physical Paf dictionary allocation/job completes successfully for that selected text. **Unverified:** the selected source is the visible dialogue, the dictionary result is useful on-screen, and the source follows advancing pages. Stable wrong text is still a failed dialogue workflow.

An earlier empty app inventory followed the user closing CLANNAD and was not a crash diagnosis. After the latest combined install/reboot, CLANNAD is again confirmed foreground with the new title profile installed. The earlier six-byte generic selection does not validate that profile; the user's two-page test is pending.

## Evidence and follow-up

Private directory: `/home/bee/Documents/src/ps-vita-ocr-lab/vita-agent-use/agent/overlay-ocr-picker-20261010/`.

| Artifact | Bytes | SHA-256 | Scope |
|---|---:|---|---|
| `config.manual-readback.ini` | 3,083 | `3833843d30aa9bd2bc294149d6c63b90f2909ce679ea05a2b2b731e9c101eabd` | Verified hooks mode; contents remain private. |
| `status.manual-start.txt` | 6,092 | `73032e8678ff51095fb6cd220ef93a964176924f58c396710965bc373f182be5` | Selected-source metadata and completed Paf dictionary jobs; no visible-dialogue match proof. |

`config.before-manual.ini`, `config.manual.ini` and transfer state remain local. Preserve operation IDs privately. Notes work made no device calls.

The later success supersedes the unresolved result for this earlier selection. Preserve source identity and observed updates; do not infer a screenshot-verified page count from the enthusiastic user reply alone. Keep the manual mode until the user chooses otherwise. The region-sized capture repair and picker diagnostics are now installed alongside the title profile, while manual mode is preserved. The current test still uses native hooks, not OCR.

Related: [[02-Current-device-state]], [[06-Low-memory-overlay]], [[17-OCR-first-physical-workflow]], [[19-Physical-OCR-source-failure]], [[12-Open-questions-and-next-steps]].
