---
title: Current device state snapshot
created: 2026-10-10
updated: 2026-10-10
snapshot_utc: "2026-10-10 12:33"
status: native-text-works-input-fix-deployment-stopped
tags: [vita, device, handoff]
---

# Current device state snapshot

**Latest verified state:** the region-sized capture kernel, bounded picker-diagnostics shell and revision-checked CLANNAD profile are installed with complete readbacks. The unchanged text plugin and manual config also match their recorded bytes. Normal boot at 12:10:49 UTC shows API 8, shell registration 0 and worker startup; the complete 27-entry inventory reports kernel/shell loaded in STARTED state 6, with zero query errors. Root confirmed CLANNAD `PCSG00415` foreground at 12:11:53 UTC.

**Physical result:** the user replied “omg it works.” Fresh logs show selected ID 18, **CALL kind 2**, encoding AUTO 0, address `0x8105910F`, changing from one to five updates with successful dictionary jobs across reopens. This is user-reported visible success plus independent native follow/lookup metadata; exact page count is not separately stated. The selected source is not the profile's REGISTER stream `0x70000000`, so physical profile capture remains unproved. Final explicit refresh clears selection to 0. See [[23-Physical-native-hook-success]].

**New input defect:** the user confirms pressing × to select a hook advances CLANNAD, with no idle advance. The existing selection path does not unblock input; Positive2 is already hooked. Overlay PR #17 head `464248a` hardens neutral writes/error propagation and adds bounded diagnostics; its kernel `fee142fa...` and shell `faea5914...` are prepared, not installed. The user explicitly stopped this deployment before upload, app close or reboot. The PR is open/unmerged, the installed setup is unchanged, and the actual leak is not yet fixed or causally localized on hardware. See [[24-Overlay-selection-input-leak]].

**Current mode:** the user's fixed manual-list choice is preserved, `text_source=hooks`. No OCR is needed to open this workflow; Google Lens and the saved region remain configured for optional future use. The previous manual selection returned dictionary jobs successfully but appeared wrong to the user. That was an earlier generic selection; the new successful physical selection is also CALL kind 2. The installed profile cannot be credited as the cause from that result.

## Installed artifacts

Custom API remains 8. Kernel/shell are under `ur0:tai/`; the title profile is `ux0:data/VitaJPOverlay/hooks/PCSG00415.vjhook`. Staged uploads completed and complete readbacks were byte-equal before reboot. The notes author independently hashed the local readbacks/preserved copies.

| Artifact | Bytes | SHA-256 | Evidence |
|---|---:|---|---|
| `VitaJPOverlay_Kernel.skprx` | 39,048 | `ae3c03412837ceb0770d078dbc4f7c5197accfefccd96a3e86292a0d67515cfd` | Fresh full readback; normal boot and loaded STARTED 6. |
| `VitaJPOverlay_Shell.suprx` | 159,528 | `4f1f0650816c5ae5d9073d3d92bb7afbacc020500497e474e29765daf9371b83` | Fresh full readback; normal boot and loaded STARTED 6. |
| `VitaJPOverlay_Text.suprx` | 7,936 | `84259631e0efe87472750a04a944a0d2cb7673f218c85f316ae3a450e6c7565f` | Unchanged full readback; native CALL follow/lookup verified; REGISTER-profile observations unproved. |
| `PCSG00415.vjhook` | 63 | `99c73600540f5c441939a06f6cc727ec21b284fa055bc936f7bb0359382f9f81` | Fresh full readback; two-page proof is emulator-only so far. |
| Preserved `config.ini` | 3,083 | `3833843d30aa9bd2bc294149d6c63b90f2909ce679ea05a2b2b731e9c101eabd` | Exact manual hooks config preserved; no mode/region change in this install. |

Kernel source: PR #14 head `5118c4a09248a3d8b7e8018d85fa0d0247b82ca4`, merge `c888b5936ca46729b0fbc7698b2348c5a4b3d387`. Shell with diagnostics: PR #15 head `f58b1aa49dd43fcc3d6d78b0c9bcd48a1c0ca6d7`, merge `5e5bc2557ee183e60c4afd9335b55666e8697cea`. Profile: PR #16 head `30a68d216243f74e772410a31b90ff17be6dddcb`, merged at 12:09:53 UTC as `264241eb0f6323f373049a458be214448f8320cb`; exact-head host (44 s) and Vita (4m26s) passed. Local main/origin are both `264241e`.

## Private evidence and rollback

The latest directory is `vita-agent-use/agent/overlay-region-capture-20261010/`. It contains staged artifacts, kernel/shell `.before` backups, complete `.readback` files, unchanged text/config copies, transfer journals, `deployment-plan.json`, the fresh startup log and the installed agent-runtime provenance copy. Keep operation IDs, pairing and config contents private.

Installation time in the plan is `2026-10-10T12:10:33.316736+00:00`. It records all uploads/readbacks, preserved manual default and verified normal boot. Its latest status is `physical_native_hook_user_reported_working_logs_confirm_follow_and_lookup`, with text-plugin loaded verification true; this supersedes earlier launch/test-pending labels. The complete postlaunch inventory reports the per-title text plugin loaded in STARTED state 6 with zero query error. Loading alone does not prove profile observation or dialogue correspondence.

`status.after-reboot.txt` is 706 B, SHA-256 `bc3577582f65bb2538c7e6bd424d1c65fe13db1aea74c9c8c75699bb8a71e6fb`, with fresh device time 12:10:49 UTC. Its contents/hash match earlier healthy startup logs; freshness is established by collection and device time, not a different digest. The notes author inspected API 8/registration 0/worker-start markers.

Preinstall kernel `6cc2dc86...` and shell `32baceeb...` are preserved in the installation-specific backups. Earlier boot/capture/auto/manual experiments remain under `agent/overlay-libc-20261010/` and `agent/overlay-ocr-picker-20261010/`; do not mix rollback sets or reset unrelated agent files. Firmware, bootloader, tai configuration and saved region were unchanged. The emulator test bridge was not installed.

## Earlier failures and remaining limits

- The previous manual/scanning shell produced changing 0% hooks and irrelevant text. Its source-table sequence reached 3,035 with no selected source. See [[17-OCR-first-physical-workflow]].
- The OCR-first physical test failed before JPEG/Lens: six paired logs identify the old 2,040 KiB kernel capture allocation failing with `0x80024302`. The now-installed region-sized replacement has not yet completed a verified physical OCR request. See [[19-Physical-OCR-source-failure]].
- The later fixed manual list selected a generic six-byte/one-update source and completed 256 KiB Paf dictionary jobs (`rc=0`, peak 110,080 B), but the user suspected wrong text. See [[20-Fixed-manual-hook-workflow]].
- The installed screenshot-agent runtime is now identified as the stock private SceShell JPEG backend, SHA `d3018186...`, 310,212 B. Its reported code -7/stage `jpeg_encode` still has multiple possible causes; the dirty local alternate backend was neither installed nor changed. See [[21-Picker-diagnostics-and-screenshot-provenance]].
- Two-page generic and register capture proofs, plus one short-page live Lens host result, remain separate from physical visible dialogue. No broad game compatibility or console peak-memory/performance claim is established.

This is a dated snapshot. PIDs, app state and connections can change. Read current state before another operation; see [[09-Device-operations-and-recovery]]. The latest success log is 4,952 B, SHA `c31bb2cc0692a3e3f049894950a8f3ec6fc8864c6c997541cbcfab92565656b1`, from device time 12:14:59 UTC. Preserve the working manual setup while addressing the × input leak; host hardening and a prepared build do not establish the physical press is contained. See [[12-Open-questions-and-next-steps]].
