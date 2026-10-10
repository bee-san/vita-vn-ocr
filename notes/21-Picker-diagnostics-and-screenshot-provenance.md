---
title: Picker diagnostics and screenshot provenance
created: 2026-10-10
updated: 2026-10-10
status: merged-prepared-not-installed
tags: [vita, diagnostics, provenance, picker]
---

# Picker diagnostics and screenshot provenance

The user suspects the selected manual text is wrong, while the device screenshot tool returns `frame_unavailable`, code -7, stage `jpeg_encode`. Existing logs identify selected IDs and byte counts but cannot show whether the displayed candidates match the intended text. The diagnostics change records what the picker actually presented on explicit user actions so the next investigation can compare bounded private evidence.

## Merged diagnostic change

[Overlay PR #15](https://github.com/bee-san/vita-jp-overlay/pull/15), “Log bounded picker candidates on explicit user actions,” has exact head `f58b1aa49dd43fcc3d6d78b0c9bcd48a1c0ca6d7`. It merged at 11:55:40 UTC as `5e5bc2557ee183e60c4afd9335b55666e8697cea`, on top of capture-repair merge `c888b59`.

- Open, refresh and selection actions record displayed rows/selected source, ID, kind, encoding, address and at most 160 UTF-8 bytes of text.
- OCR-triggered actions emit once when the result is published; closing or changing games clears a pending diagnostic.
- Newlines/control bytes become spaces, UTF-8 boundaries are preserved, and logging happens after releasing the view lock.
- Worst-case row/selection records are 237/240 bytes before the newline, within the existing 256-byte logger. Existing status/log sinks receive them.
- Background source updates do not emit excerpts. This avoids continuous dialogue logging and preserves the fixed manual interaction.

These excerpts can contain copyrighted game dialogue. They must remain private and ignored: publish only narrow derived metadata, hashes, counts and conclusions in this notebook. Neither the diagnostic code nor these notes include dialogue samples.

## Validation and prepared artifacts

Reported local Linux validation uses host GCC 16.2.1 and VitaSDK GCC 15.2.0. All 10 existing host suites passed in 5.31 s. The clean ncnn-OFF Vita Release build with `VJO_TEXT_TEST_APP=ON` and packaged syscall ABI validation passed; source review covered UTF-8/record bounds, explicit-action emission and view-lock ownership. Exact-head hosted run `38049801084` passed host in 1m06s and Vita in 3m07s; release was skipped. The notes author independently queried the PR's exact merge and check records.

The combined PR #14+#15 artifacts are prepared locally under `vita-agent-use/agent/overlay-region-capture-20261010/`. The notes author independently hashed all three files:

| Prepared artifact | Bytes | SHA-256 |
|---|---:|---|
| Kernel | 39,048 | `ae3c03412837ceb0770d078dbc4f7c5197accfefccd96a3e86292a0d67515cfd` |
| Shell with picker diagnostics | 159,528 | `4f1f0650816c5ae5d9073d3d92bb7afbacc020500497e474e29765daf9371b83` |
| Text, unchanged | 7,936 | `84259631e0efe87472750a04a944a0d2cb7673f218c85f316ae3a450e6c7565f` |

They are **not installed**. The device still has kernel `6cc2dc86...`, shell `32baceeb...` and text `84259631...` in manual hooks mode. Prepared files are not readbacks, and merged diagnostics do not establish physical dialogue capture. The stack-entry candidate was rejected for preloading future lines. Its replacement at `0x2CA70`/r6 now passes two-page emulator proof but remains uninstalled; see [[22-CLANNAD-register-profile-proof]].

## Screenshot failure is not a source-level diagnosis

The screenshot tool's returned stage label is an observation, not proof of which implementation failed. The installed agent runtime's provenance has not been matched to the dirty local `vita-agent-use` checkout. That checkout includes changes to its capture/backend files and uses a different backend; inspecting it cannot explain the installed runtime without a binary/revision match.

Do not infer a specific JPEG encoder bug, conflate this tool failure with the overlay's separately proved kernel capture-buffer OOM, or install/reset unrelated agent changes. No agent-code changes or device calls were made for this notes work. A future screenshot investigation needs the loaded runtime identity, exact invocation/result and matching source before causal claims.

Related: [[02-Current-device-state]], [[08-Validation-and-measurements]], [[19-Physical-OCR-source-failure]], [[20-Fixed-manual-hook-workflow]], [[11-Sources-and-artifacts]].
