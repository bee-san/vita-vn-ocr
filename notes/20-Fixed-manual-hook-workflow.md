---
title: Fixed manual hook workflow
created: 2026-10-10
updated: 2026-10-10
status: selected-source-does-not-yet-match-verified-dialogue
tags: [vita, native-text, manual, user-preference]
---

# Fixed manual hook workflow

The user's latest choice is a **fixed manual hook list**. After the OCR capture failure, the user suggested optional downscaling/black-and-white, then explicitly said ordinary text hooks without OCR are acceptable if the list does not spam or constantly change. They selected the fixed manual-list option. This supersedes an assumption that OCR is mandatory for the next usable result; it does not invalidate the earlier requested OCR-first implementation or its measured capture failure.

## Verified configuration change

Root changed only `text_source = auto` to `text_source = hooks`, guarding the currently installed config hash, waiting for transfer completion and verifying the complete readback. The 3,083 B file has SHA-256 `3833843d30aa9bd2bc294149d6c63b90f2909ce679ea05a2b2b731e9c101eabd`. The notes author independently hashed that local readback and compared it with the pre-change config. Lens and the saved region remain available for future use, but manual opening does not run OCR.

This setting takes effect on the next overlay open; no reboot or plugin replacement was required. The installed shell remains PR #13 `32baceeb...`, including its stable manual picker behavior. This is distinct from the old shell's constantly republished, scanning list that originally frustrated the user.

## Observed manual selection and dictionary job

Private `status.manual-start.txt` shows session 12 selecting source ID 1 at sequence 475 with kind 2, encoding 0, six bytes and update count 1. It reports `native dictionary workspace: Paf heap, 256 KiB`, then a manual-selection job completing `rc=0` in 37 ms, arena peak 110,080 B. Later reopens at sequences 1,787 and 3,046 complete native-text jobs in 21/25 ms with the same arena peak. These are isolated observed job durations, not a latency distribution or performance guarantee.

The selected source remains six bytes with update count 1 in the recorded samples. Other source-table sequences advance, but that alone does not show the selected source follows current dialogue. The user thinks the chosen text is wrong. Root cannot visually confirm it because the device screenshot tool returns `frame_unavailable`, code -7, stage `jpeg_encode`, and has asked for a short example. The installed runtime is not matched to the dirty local agent backend, so that stage label does not diagnose a particular encoder implementation. See [[21-Picker-diagnostics-and-screenshot-provenance]]. Do not quote private game text into these notes when comparing it.

**Established:** a source can be selected, the manual list is no longer driving the earlier repeated OCR failure, and the physical Paf dictionary allocation/job completes successfully for that selected text. **Unverified:** the selected source is the visible dialogue, the dictionary result is useful on-screen, and the source follows advancing pages. Stable wrong text is still a failed dialogue workflow.

The latest completed app inventory has no running entries after the user closed CLANNAD. Do not reinterpret that as an observed crash. Root is separately evaluating a possible title-specific register entry; its validity is not established by the current manual selection.

## Evidence and follow-up

Private directory: `/home/bee/Documents/src/ps-vita-ocr-lab/vita-agent-use/agent/overlay-ocr-picker-20261010/`.

| Artifact | Bytes | SHA-256 | Scope |
|---|---:|---|---|
| `config.manual-readback.ini` | 3,083 | `3833843d30aa9bd2bc294149d6c63b90f2909ce679ea05a2b2b731e9c101eabd` | Verified hooks mode; contents remain private. |
| `status.manual-start.txt` | 6,092 | `73032e8678ff51095fb6cd220ef93a964176924f58c396710965bc373f182be5` | Selected-source metadata and completed Paf dictionary jobs; no visible-dialogue match proof. |

`config.before-manual.ini`, `config.manual.ini` and transfer state remain local. Preserve operation IDs privately. Notes work made no device calls.

Next compare the short user-reported text with the visible line or obtain equivalent safe visual evidence. Choose a different source if required, then verify two changing pages rather than accepting a one-time selection or dictionary return code. Keep the manual mode until the user chooses otherwise. The optional region-sized capture repair is now merged as `c888b59`; its prepared artifacts remain uninstalled and do not gate this requested manual experiment.

Related: [[02-Current-device-state]], [[06-Low-memory-overlay]], [[17-OCR-first-physical-workflow]], [[19-Physical-OCR-source-failure]], [[12-Open-questions-and-next-steps]].
