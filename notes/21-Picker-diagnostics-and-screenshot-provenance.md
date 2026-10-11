---
title: Picker diagnostics and screenshot provenance
created: 2026-10-10
updated: 2026-10-10
status: installed-native-call-success-logged
tags: [vita, diagnostics, provenance, picker]
---

# Picker diagnostics and screenshot provenance

Earlier the user suspected the selected manual text was wrong, while the device screenshot tool returns `frame_unavailable`, code -7, stage `jpeg_encode`. Existing logs identify selected IDs and byte counts but cannot show whether the displayed candidates match the intended text. The diagnostics change records what the picker actually presented on explicit user actions so the next investigation can compare bounded private evidence.

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

The combined PR #14+#15 artifacts were prepared locally under `vita-agent-use/agent/overlay-region-capture-20261010/`. The notes author independently hashed all three files:

| Prepared artifact | Bytes | SHA-256 |
|---|---:|---|
| Kernel | 39,048 | `ae3c03412837ceb0770d078dbc4f7c5197accfefccd96a3e86292a0d67515cfd` |
| Shell with picker diagnostics | 159,528 | `4f1f0650816c5ae5d9073d3d92bb7afbacc020500497e474e29765daf9371b83` |
| Text, unchanged | 7,936 | `84259631e0efe87472750a04a944a0d2cb7673f218c85f316ae3a450e6c7565f` |

These artifacts are now **installed with complete device readbacks and normal boot verified**. Manual hooks mode and the unchanged text plugin are preserved. A separate 63 B profile from PR #16 is also installed and CLANNAD is foreground; the user now reports the native workflow works and logs show selected CALL-source changes/lookups. Physical REGISTER correspondence remains unproved. See [[23-Physical-native-hook-success]]. The earlier stack-entry candidate was rejected for preloading future lines, while its `0x2CA70`/r6 replacement passes two-page emulator proof. See [[22-CLANNAD-register-profile-proof]] and [[02-Current-device-state]].

## Screenshot runtime identified; error cause still ambiguous

The installed screenshot-agent runtime was read back and matched to the stock private SceShell JPEG backend: **310,212 B**, SHA-256 `d30181867cba4c8922b7f744796a4cddb6455b44702c121263d04b131e87704f`. The notes author independently hashed the local `vita_agent_shell.suprx.provenance` copy in the deployment directory. This resolves the earlier unmatched-runtime limitation; the dirty local `vita-agent-use` checkout still uses a different backend and is not the installed source.

Code -7/stage `jpeg_encode` remains ambiguous even with this provenance: invalid dimensions, Paf allocation failure, invalid JPEG output or a native -7 return may reach it. No specific encoder cause is proved. Do not conflate it with the overlay's separately demonstrated full-frame capture-buffer OOM or install/reset unrelated local agent changes. No agent source changes were made. Future diagnosis needs branch-specific runtime evidence, not the stage label alone.


## Likely PNG/JPEG conflict, not yet confirmed from returned bytes

The root investigator reports pngshot loaded on this 3.65 system. The notes author independently inspected [pngshot's pinned source](https://github.com/xyzz/pngshot/blob/113d471500e7428a1b3f890a8ff2e9517d8218b8/main.c): the 3.65 retail SceShell module-NID branch includes `0x5549BF1F`, hooks encoder offset `0x36638A`, changes the requested format to 2 and replaces that encoder with PNG output. The matched installed agent calls the same private encode entry and expects JPEG SOI/EOI markers, rejecting non-JPEG bytes with -7.

This overlap strongly supports a PNG-output/JPEG-validation conflict. It remains an inference: raw PNG output has not been observed from the failing live call. No encoder patch or plugin disabling was performed, and the other -7 branches are not ruled out merely by this source match.

The user was instructed to press physical PS+START for the existing screenshot route, which can save under `ux0:picture/SCREENSHOT` for later filesystem retrieval. The agent cannot inject PS. That folder currently returned ENOENT `0x80010002`, and no PS+START result or screenshot retrieval has been confirmed. Do not treat the alternate route as completed evidence.

Related: [[02-Current-device-state]], [[08-Validation-and-measurements]], [[19-Physical-OCR-source-failure]], [[20-Fixed-manual-hook-workflow]], [[11-Sources-and-artifacts]].
