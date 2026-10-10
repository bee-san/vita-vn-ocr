---
title: Physical native hook success
created: 2026-10-10
updated: 2026-10-10
status: user-reported-working-native-follow-and-lookup-logged
tags: [vita, clannad, physical, native-text, success]
---

# Physical native hook success

At approximately 12:15 UTC, after the combined install and requested physical test, the user replied **“omg it works.”** Fresh device logs independently show selection of a native source, changing captured text and successful dictionary jobs across reopens. This is the first user-reported working physical CLANNAD experience in this investigation.

The successful selected source is **CALL kind 2**, not REGISTER kind 4. The newly installed title profile exists and has emulator proof, but this hardware result does not establish that the profile supplied the working text or was the missing component. Keep that distinction explicit.

## Recorded physical source and progression

| Field | Observation |
|---|---|
| Session | 6 in this log; identifiers are run-specific. |
| Selected source | ID 18, kind 2 CALL, encoding 0 AUTO. |
| Bounded candidate address | `0x8105910F`. |
| Selection | Sequence 1,507, 24 UTF-8 bytes, update count 1. |
| Subsequent text sizes | 24 → 6 → 18 → 72 → 18 bytes. |
| Update counts | 1 → 2 → 3 → 4 → 5. |
| Native lookup jobs | `rc=0` in 111 ms, 166 ms and 1,290 ms, plus a 421 ms rerun after text changed. |
| Measured job arena peaks | 110,128 / 110,112 / 110,224 B in these jobs. |
| Picker | Five fixed displayed rows on open, scanning=0. |
| Final explicit refresh | Sequence 4,371 clears selection to 0; the source remains among displayed rows. |

These are observed individual job durations, not a benchmark or promised response time. Source-table activity alone was never enough; this log additionally shows the selected source's own update counter and text length changing, then lookup completion. The notes author inspected only bounded metadata and hashes, leaving actual excerpts private.

The user response supports visible usefulness, while the log proves native follow/lookup behavior. It does not enumerate the exact number of visible pages checked, and no independent device screenshot was obtained. Do not relabel it as a screenshot-verified two-page comparison. The explicit final refresh intentionally clears selection; future work should not assume ID 18 remains selected or that its numeric ID survives a new game session.

## Provenance

Private file: `/home/bee/Documents/src/ps-vita-ocr-lab/vita-agent-use/agent/overlay-region-capture-20261010/status.hardware-success.txt`, **4,952 B**, SHA-256 `c31bb2cc0692a3e3f049894950a8f3ec6fc8864c6c997541cbcfab92565656b1`; recorded device modification time 12:14:59 UTC. The notes author independently hashed it and checked selected-source/update/job markers without publishing dialogue.

The updated deployment plan status is `physical_native_hook_user_reported_working_logs_confirm_follow_and_lookup`. Installed files are kernel `ae3c0341...`, shell `4f1f0650...`, unchanged text `84259631...`, profile `99c73600...` and preserved manual config `3833843d...`; exact hashes are in [[02-Current-device-state]].

## What remains separate

- Physical REGISTER stream `0x70000000` correspondence is unproved. Its two-page proof remains emulator-only even though the profile is installed.
- Physical OCR after the region-sized capture repair remains untested; this working flow uses manual hooks without OCR.
- Screenshot code -7 has a likely pngshot/JPEG format conflict, but no returned PNG bytes or alternate screenshot retrieval are confirmed.
- Other scenes, speakers, game revisions, long sessions, suspend/resume and broader game compatibility are unmeasured.

The physical CALL address differs from the earlier emulator memcpy caller `0x8102CA7F` by `0x2C690`. Relocation is a possible explanation, but the native segment base has not been queried, so caller identity across the runs is not established.

Related: [[02-Current-device-state]], [[20-Fixed-manual-hook-workflow]], [[22-CLANNAD-register-profile-proof]], [[21-Picker-diagnostics-and-screenshot-provenance]], [[08-Validation-and-measurements]].
