---
title: Current device state snapshot
created: 2026-10-10
updated: 2026-10-10
snapshot_utc: "2026-10-10 11:26"
status: boot-verified-physical-workflow-failed
tags: [vita, device, handoff]
---

# Current device state snapshot

**Latest verified state:** the generic SceLibc and source-ranking correction is installed, fully read back, and normally booted on the physical Vita. The fresh 706 B status log (device modification time 11:00:20 UTC) shows API 8, shell registration 0 and `bootstrap: worker started`. The control server reconnected. Native inventory was complete (27 entries); kernel and shell were loaded in STARTED state 6, with zero query errors. The per-title text plugin was unloaded because the game was not running at that check, which is expected.

**Physical test result:** the user opened CLANNAD and reported a flood of changing hooks labelled 0%, followed by irrelevant text. The workflow is not working for the user, and correct current-dialogue capture/selection remains unverified. Saved metadata shows candidates rising from 1 to 5 before opening; opening starts full memory discovery, with sequence advancing to 3,035, scanning active and selected source 0. See [[17-OCR-first-physical-workflow]].

**New user preference:** take one OCR screenshot first and present the match by default. Root acknowledged that the installed configuration was manual `text_source=hooks`, with no OCR reference. The follow-up from `a5e0c50` is merged through overlay PR #13 (`fix/ocr-first-hook-picker`); its deployment is not yet verified. Candidate head is `5826329`; shell is 158,564 B, SHA `32baceeb...`. Fresh shell/config/region backups are prepared, and Lens remains selected. The generated-config comment mismatch is fixed. PR #13 merged as `539c2888f61b28277bb7fd7bdd2b6a4bb5b330fd` after exact-head host (52 s) and Vita (4m25s) passed. Shell upload is in progress; readback/boot verification is still required. The last fully verified installed binary set is below; an in-progress upload is not yet treated as a successful replacement. A live production-Lens host check matches one short verified page; it does not establish physical OCR. See [[18-OCR-anchor-memory-measurements]].

## Last fully verified installed plugin set

All use custom API 8, under the active `ur0:tai/` path. Kernel and text were uploaded sequentially with expected-old-hash guards, native completion and full readback before reboot. The low-memory shell remains unchanged.

| Plugin | Bytes | SHA-256 | Evidence |
|---|---:|---|---|
| `VitaJPOverlay_Kernel.skprx` | 38,620 | `6cc2dc868dc4403bcab252d689acd2f0779a55f9bd8069654b335eaffa571933` | Fresh installed-file readback and successful normal boot. |
| `VitaJPOverlay_Shell.suprx` | 156,196 | `45f86b1ea83c08bdf61c1dfd174cc5356151e746a3659a0299218436a8f10c8c` | Previous full readback; unchanged in this install; loaded and started after reboot. |
| `VitaJPOverlay_Text.suprx` | 7,936 | `84259631e0efe87472750a04a944a0d2cb7673f218c85f316ae3a450e6c7565f` | Fresh installed-file readback; physical candidate traffic observed, but correct dialogue selection unverified. |

Implementation source: `6eef6c031aa1cc4bc162ebf31b8628d3098f0f5e`. Final reviewed evidence head: `69006af9e36347b75534c01900b48d43ed372981`. Overlay PR #12 merged at 10:59:00 UTC as `a5e0c50b1096b5ae7de450ae712e63b272dc7f17`; final-head host (46 s) and Vita (4 m 23 s) CI both passed. The installed artifacts come from the implementation revision; the evidence-only commit changes no binary.

## Preserved rollback and evidence

Fresh preinstall backups match the previous kernel SHA `447f2e640973743f80dbeb6099e62b6e7c2b72fe85936af4f8d9c24dfd54f217` and previous text SHA `c146238b6e6544ec9ba27f1b03d35db89072c2fc4f3a05d0e75d94c679a777a7`. They remain private under `vita-agent-use/agent/overlay-libc-20261010/`, alongside upload journals, readbacks and the installation report. Operation IDs remain in private state.

The notes author independently hashed both local readbacks and inspected the postboot log. Its SHA-256 is `bc3577582f65bb2538c7e6bd424d1c65fe13db1aea74c9c8c75699bb8a71e6fb`. At installation completion the report status was `installed_readback_verified_normal_boot_verified_awaiting_user_clannad_test`; the later user result below supersedes that pending-test state, installation time `2026-10-10T11:00:04.481678+00:00`.

No firmware, bootloader, tai configuration or region settings were changed in this deployment. The emulator test bridge was not installed. Experimental scanner/kernel diagnostic commit `2d56ce4` was not shipped.

## Environment and boundaries

- Physical device: PCH-1001, actual firmware 3.65.
- Title: CLANNAD, `PCSG00415`.
- Last inspected controls/configuration: L+R overlay, Select+R subtitles, hooks text source, local dictionary.
- The earlier low-memory shell opened an empty picker with the old backend. The new backend produces candidate traffic, but the manual/scanning UI is unstable and did not give the user relevant current text.
- Latest offline proof: two visible narration pages captured through generic imports, with three matching sources in the top five on each page and no title profile. Physical kernel mapping and shell interaction are separate tests.
- A later pre-fix launch/power-off report remains a historical unresolved observation; normal boot of this new installed revision is now independently established.

This is still a dated snapshot. PIDs, app state, tokens and connection state can change. Obtain them live before the next operation. Preserve the existing paired configuration and exact transfer state; see [[09-Device-operations-and-recovery]].

Next update: record the OCR-first change, exact deployment/configuration and whether one screenshot supplies a matching current-page source. Preserve this failed physical UX result alongside the successful install/boot record. See [[12-Open-questions-and-next-steps]].
