---
title: Current device state snapshot
created: 2026-10-10
updated: 2026-10-10
snapshot_utc: "2026-10-10 11:00:20"
status: installed-boot-verified-game-test-pending
tags: [vita, device, handoff]
---

# Current device state snapshot

**Latest verified state:** the generic SceLibc and source-ranking correction is installed, fully read back, and normally booted on the physical Vita. The fresh 706 B status log (device modification time 11:00:20 UTC) shows API 8, shell registration 0 and `bootstrap: worker started`. The control server reconnected. Native inventory was complete (27 entries); kernel and shell were loaded in STARTED state 6, with zero query errors. The per-title text plugin was unloaded because the game was not running at that check, which is expected.

**Pending:** the user has been asked to open CLANNAD, show a page, press L+R, select matching text with ▲/▼ and ×, then advance dialogue and reopen twice. No physical game-capture result from this installed version has arrived at this snapshot. Do not infer success from the boot or the earlier emulator result.

## Installed plugin set

All use custom API 8, under the active `ur0:tai/` path. Kernel and text were uploaded sequentially with expected-old-hash guards, native completion and full readback before reboot. The low-memory shell remains unchanged.

| Plugin | Bytes | SHA-256 | Evidence |
|---|---:|---|---|
| `VitaJPOverlay_Kernel.skprx` | 38,620 | `6cc2dc868dc4403bcab252d689acd2f0779a55f9bd8069654b335eaffa571933` | Fresh installed-file readback and successful normal boot. |
| `VitaJPOverlay_Shell.suprx` | 156,196 | `45f86b1ea83c08bdf61c1dfd174cc5356151e746a3659a0299218436a8f10c8c` | Previous full readback; unchanged in this install; loaded and started after reboot. |
| `VitaJPOverlay_Text.suprx` | 7,936 | `84259631e0efe87472750a04a944a0d2cb7673f218c85f316ae3a450e6c7565f` | Fresh installed-file readback; game-process load/capture test pending. |

Implementation source: `6eef6c031aa1cc4bc162ebf31b8628d3098f0f5e`. Final reviewed evidence head: `69006af9e36347b75534c01900b48d43ed372981`. Overlay PR #12 merged at 10:59:00 UTC as `a5e0c50b1096b5ae7de450ae712e63b272dc7f17`; final-head host (46 s) and Vita (4 m 23 s) CI both passed. The installed artifacts come from the implementation revision; the evidence-only commit changes no binary.

## Preserved rollback and evidence

Fresh preinstall backups match the previous kernel SHA `447f2e640973743f80dbeb6099e62b6e7c2b72fe85936af4f8d9c24dfd54f217` and previous text SHA `c146238b6e6544ec9ba27f1b03d35db89072c2fc4f3a05d0e75d94c679a777a7`. They remain private under `vita-agent-use/agent/overlay-libc-20261010/`, alongside upload journals, readbacks and the installation report. Operation IDs remain in private state.

The notes author independently hashed both local readbacks and inspected the postboot log. Its SHA-256 is `bc3577582f65bb2538c7e6bd424d1c65fe13db1aea74c9c8c75699bb8a71e6fb`. Installation-report status is `installed_readback_verified_normal_boot_verified_awaiting_user_clannad_test`, installation time `2026-10-10T11:00:04.481678+00:00`.

No firmware, bootloader, tai configuration or region settings were changed in this deployment. The emulator test bridge was not installed. Experimental scanner/kernel diagnostic commit `2d56ce4` was not shipped.

## Environment and boundaries

- Physical device: PCH-1001, actual firmware 3.65.
- Title: CLANNAD, `PCSG00415`.
- Last inspected controls/configuration: L+R overlay, Select+R subtitles, hooks text source, local dictionary.
- The earlier installed low-memory shell already opened the picker, but found no usable sources with the old text backend.
- Latest offline proof: two visible narration pages captured through generic imports, with three matching sources in the top five on each page and no title profile. Physical kernel mapping and shell interaction are separate tests.
- A later pre-fix launch/power-off report remains a historical unresolved observation; normal boot of this new installed revision is now independently established.

This is still a dated snapshot. PIDs, app state, tokens and connection state can change. Obtain them live before the next operation. Preserve the existing paired configuration and exact transfer state; see [[09-Device-operations-and-recovery]].

Next update: record the user's actual picker/source-selection result, then metadata for two advancing pages and dictionary behavior. Keep a failure at any stage visible rather than replacing the verified boot/install record. See [[12-Open-questions-and-next-steps]].
