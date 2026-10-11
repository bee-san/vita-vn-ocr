---
title: Current device state snapshot
created: 2026-10-10
updated: 2026-10-11
historical_snapshot_utc: "2026-10-10 12:33"
latest_checkpoint_date: 2026-10-11
status: settings-png-repair-controls-verified
tags: [vita, device, handoff]
---

# Current device state snapshot

**Current October 11 checkpoint:** [[27-Meiki-hardware-and-project-lessons]] records
verified meiki.2 installation, fresh API13 boot, two actual CLANNAD Meiki runs,
and the user-confirmed region controls. VOCR was separately confirmed in
[[26-Game-process-OCR-hardware-checkpoint]]. The new Settings engine picker,
VOCR default and Japanese-only selection default are implemented and locally
validated: 22 host suites, 22 sanitizer suites and the native Release build pass.
The PR #33 build is merged and v0.7.6 is published with all downloaded asset
checksums verified. The Settings-only readability follow-up is PR #34 at
`c9ec54f9a6562ad61cbb8d9e4876e86028e652f4`, packaged as v0.7.7: three larger
game cards, clearer tabs, wrapped titles and higher-contrast text. Its actual
Settings input/draw loop passes 12/12 host ASan/UBSan scenarios, and the native
Release build passes. These checks do not establish physical firmware-font
readability or touch behavior. PR #34 is merged as
`b6518d0b01cf5274eddfa036f7f5f65109121450`; exact-head host/Vita CI passed and
v0.7.7 is published. All six downloaded assets match the audited candidate,
including all five payload checksums. The native installation is now verified
as recorded below. The later v0.7.8 repair and native preference checks are
recorded separately; physical firmware-font readability remains pending.

**Native v0.7.7 installation checkpoint, 2026-10-11:** recovery using the original
Daijirin transfer identity completed in 72.143 seconds with `code=0`,
`verified=true`, `complete=true` and `readback_required=false`. A subsequent
native stat confirms the installed destination is a 156,827,663-byte file.
Settings promotion completed with `installed=true`, `running=false`,
`native_result=0`, title `VJOSET001` and audit sequence 1778. Full readback of
the installed executable is byte-equal to the release VPK's executable. The
Shell and configuration were also fully read back and verified.

| Installed artifact | Bytes | SHA-256 |
|---|---:|---|
| v0.7.7 Shell | 149,220 | `d120e548be9e24f36f5e9551cb6ab8792eb5e595ae29430241450f97b401e308` |
| v0.7.7 Settings `eboot.bin`, before PNG repair | 277,084 | `2a34d0c7b2a7e4a500b34be4626d365d3bd9b715d7b4fc85bba807706e1f3b11` |
| Private configuration | 2,951 | `312ce783379b86e0194c04750eb2fe7e120cd51f29dcb5085ed01dd2dab79c00` |
| Private Daijirin v3 dictionary | 156,827,663 | `e767f405bcd8021f7fb226b8ca6da70ba0b6250782e8afc998bf924ab7c4fb53` |

Only three configuration values changed: `ocr_backend=vocr`,
`japanese_only_selection=on` and the local dictionary list, now Daijirin first
with JMdict fallback. `dictionary=local` was already selected, and
`ocr_mode=on_press` and unrelated settings remain intact. This proves the saved
preferences at that checkpoint, not an actual lookup using that priority or a
physical toggle. Later native input checks are recorded below.
The installed SFO retains static `APP_VER=01.00`; the executable checksum and
release comparison establish the v0.7.7 package identity.

**Fresh boot and launch:** the reboot accepted at 13:56 UTC is followed by a
snapshot at about 30 seconds uptime with Kernel/Shell native state 6 and zero
query errors. Fresh Shell startup reports API 13, PAF loaded and worker started.
Settings launch at about 62 seconds uptime is confirmed with the expected
title and an empty list of closed apps. The native snapshot at that checkpoint
reports `system_ui_overlaid=true`. No buttons were sent to clear it; the user
was asked to open the app physically and subsequently confirms Settings is
visible, but reports an error. Two completed native dumps now identify the
same `_free_r` failure at module offset `0x442f0`, with invalid read address
`0x7f80dea9`. Qualified stack scanning points into PNG loading and jump-buffer
cleanup; linked-code/layout analysis identifies a libPNG/setjmp ABI mismatch.
The linked `setjmp` writes 104 bytes into the shipped libPNG's 92-byte inline
jump buffer, overwriting its following cleanup metadata. The repair rebuilds
pinned libPNG 1.6.58 against the current SDK's 160-byte `jmp_buf`; the newer
candidate's build and installed-file checkpoint is recorded below.
See NI04 in [[27-Meiki-hardware-and-project-lessons]]. At this original
checkpoint, installed bytes, normal startup, process launch and user-reported
app visibility were established; successful saves and preference controls
had not yet been measured. The repaired app's later checks follow below.

**PNG repair candidate checkpoint:** [overlay PR #36](https://github.com/bee-san/vita-jp-overlay/pull/36)
at `1c8e8db1d4df301342416b6d68c4998e9dddbd67` passes both host and native CI in
[run 38146423675](https://github.com/bee-san/vita-jp-overlay/actions/runs/38146423675).
The complete 283,190-byte candidate VPK was uploaded and read back with SHA-256
`c8d032acca58ea4ff586c9de7c505f9dc5ea068af6d408659c7d459e43f2bd04`.
The connection disconnected while waiting for promotion. Retained status is
`state=uncertain`, `phase=idle`, `running=false`, `installed=null`; it is not a
terminal installed receipt. Reconciliation finds Settings launchable in a
complete 88-entry native app inventory. Full installed executable readback is
277,212 bytes, SHA-256
`217628a7fa68d10d8ae9f3f194d37eda53a1e0e2da5bccd69a2c86a5e6cb9299`,
byte-equal to the repaired candidate. The first repaired process launch is
confirmed. Preserve the retained promotion identity and distinguish those file
and launch observations from its unresolved terminal operation status.

**Verified native controls:** the same repaired Settings process remained
foreground with `system_ui_overlaid=false` over a measured 100.106 seconds.
Six controller submissions, including the initial SELECT, and two page/focus
touch submissions finished with zero error and released input. Complete config
readback after each of the five subsequent controller actions verified Meiki
then VOCR, the Japanese-only OFF state and restoration to ON. The flag was
already OFF before its LEFT action; this does not measure an ON-to-OFF change.
The final 2,951-byte config is byte-equal to the intended configuration above,
preserving dictionary priority, `ocr_mode=on_press` and unrelated settings.
The sampled native event window, sequences 2635–2648, contains no coredump and
reports zero lost events; this applies only to the observed interval.

The initial SELECT/readback unexpectedly showed Japanese-only OFF. A baseline
assertion caught it; user/interleaving causation is unknown. Two explicit
OCR-tab/VOCR-badge touch actions normalized focus before the measured sequence.
Private receipt: `settings-native-check/verified-controls.json`, SHA-256
`8d8bd3905e0ff6304e949fdc8d75b19a2f618befd20763487c2794c6ec652d5d`.
A later check measured the missing ON → OFF → ON transition: OCR-tab touch
retained ON, tapping the Japanese-only card changed it to OFF, and RIGHT
restored ON. Each action completed and had a full config readback; the final
configuration again matches the intended 2,951-byte file exactly. Receipt:
`settings-native-check/verified-japanese-transition.json`, SHA-256
`2bd521fae593dbdb0e1b4638f91dcd822bb317650aa2b2e48b77abd90263b9b1`.
Physical firmware-font inspection, new game OCR, Japanese-character navigation
and actual Daijirin-first lookup remain pending.

**Verified publication:** PR #36 merged as
`b0642493688c0f31bc33738d937d876ba0edbab6`.
[Release v0.7.8](https://github.com/bee-san/vita-jp-overlay/releases/tag/v0.7.8)
was published at 14:23:11 UTC on October 11. All six downloaded assets are
byte-equal to the audited candidates, and every downloaded ZIP passes CRC.
The private download-audit receipt is
`vita-release-2026-10-11/published-v078-audit/verification.json`, SHA-256
`36aae9d2b0a275e1d460a30e39191a66d2142107012844e66cbebf40ce269d77`.
GitHub publication is separate from retained promotion status and the still
pending game/visual checks.

**Keep-awake worker installed and loaded:** the previously loaded
agent called PowerTick with argument 1 (`AUTO_SUSPEND`), leaving the display
timer available, and synchronous requests blocked its service-loop tick.
Local-only commit `1d8a438d51454c27508a42849e0674080f01e9cd`, based on
`87e9bc6`, adds an independent 8 KiB worker issuing DEFAULT ticks every five
seconds during authenticated command control or pending native
install/decrypt/content work. Eighteen actual-worker ASan/UBSan cases and 20
repeated lifecycles pass; imports remain 31 libraries and 143 functions.
Native upload completed with code 0 and full
readback of the 311,356-byte candidate matches SHA-256
`7b803d324963257fcdc0d88d7f141ea8ae3c753b7520ccd759253956341de5c4`.
After the ordinary reboot, fresh logs report worker initialization result 0
and the first DEFAULT tick result 0 with count 1. Settings relaunch is confirmed
foreground with `system_ui_overlaid=false`; the post-boot configuration remains
byte-identical to the intended 2,951-byte file. The six-minute quiet interval is
still pending. These checks do not prove OLED visibility, pairing-wait protection
or long-request hardware wakefulness. See NI09 in
[[27-Meiki-hardware-and-project-lessons]].

Private receipts are under `vita-agent-use/agent/ocr-choice-20261011/`:
`daijirin-recover-after-reconnect.jsonl`, `daijirin-installed-stat.jsonl`,
`settings-v077-install-status-02.jsonl`, the installed-executable and
configuration readbacks, post-reboot startup/snapshot records, and
`settings-v077-crash-symbols.json`/`settings-v077-two-crashes-compare.json`.
The repair-candidate installed-executable and retained install-status receipts
are in the same directory; raw operation IDs remain private.
Commercial dictionary contents, configuration contents and device identities
remain private. The earlier 13:23 UTC recovery-required/600-second timeout is
retained as history in [[27-Meiki-hardware-and-project-lessons]] and is
superseded by the completed recovery above. The 59-app/seven-page LiveArea plan
still awaits supported, observed moves. The device narrative below retains its
dated native-hook state.

**Historical early October 11 checkpoint:** API 13 booted and all three overlay plugins loaded. Fresh native game-process FL10 jobs completed with `rc=0 io=0 cleanup=0`, and subsequent captures succeeded. At that point the user reported L+R advancing dialogue when opening the overlay, and the input repair was unverified. Later hook-7 containment and physical character-workflow evidence are retained in [[26-Game-process-OCR-hardware-checkpoint]]; the repaired Square/Circle workflow and actual Meiki runs are in [[27-Meiki-hardware-and-project-lessons]]. Hardware accuracy/CER remains unmeasured.

## Historical snapshot — 2026-10-10 12:33 UTC

The remaining text below preserves the earlier native-hook setup and its then-current limits. It does not describe the installed API 13 release.

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
