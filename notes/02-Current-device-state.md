---
title: Current device state snapshot
created: 2026-10-10
updated: 2026-10-10
snapshot_utc: "2026-10-10 11:48"
status: manual-mode-selected-dialogue-match-unverified
tags: [vita, device, handoff]
---

# Current device state snapshot

**Latest verified state:** the OCR-first shell from merged overlay PR #13 is installed, fully read back and normally booted. The user subsequently chose a fixed manual list; the active configuration is now `text_source=hooks`, verified by full readback without a reboot. Google Lens and the saved region are preserved for optional later use. A fresh 706 B status log (device modification time 11:27:24 UTC) shows API 8, shell registration 0 and `bootstrap: worker started`. The complete 27-entry native inventory has kernel and shell STARTED (6), with zero query errors. CLANNAD `PCSG00415` was subsequently confirmed foreground for testing; latest completed app inventory is empty after the user closed the game. That inventory is not a crash diagnosis.

**Earlier OCR test failed:** the user reports “text recognition failed. unexpected response (-108). no game text found yet.” Six paired status/kernel attempts identify a failed 2,040 KiB capture-buffer allocation (`0x80024302`) before JPEG or Lens; job arena peak is 0. The shell receives `0xC0000002` instead of the expected no-memory return, an additional propagation issue under review. Physical Lens completion, a correct visible-page match, source selection and following the next page remain unverified. See [[19-Physical-OCR-source-failure]].

**Prior physical failure:** the previous shell/manual `text_source=hooks` mode produced a flood of changing 0% hooks and irrelevant text. Its status reached source sequence 3,035, full discovery active and selected source 0. Root acknowledged configuring that unanchored manual mode. This failed workflow motivated the new explicit user preference: one screenshot first, then a matching source. The successful install/boot does not erase that failure or prove its resolution. See [[17-OCR-first-physical-workflow]].

**Latest manual result:** source ID 1 was selected (kind 2, encoding 0, six bytes, one update). A 256 KiB Paf dictionary workspace completed the selection job with `rc=0`, 37 ms and 110,080 B arena peak; two later reopens also completed. The user suspects the text is wrong. Device screenshots fail at JPEG encoding, so root requested a short example. No visible-page correspondence or changing selected dialogue is verified. See [[20-Fixed-manual-hook-workflow]].

## Installed plugin set

All use custom API 8 under the active `ur0:tai/` path. The kernel/text were installed with expected-old-hash guards, native completion and full readback for PR #12. PR #13 replaced only the shell; kernel/text remain unchanged. Its initial hooks→auto configuration was later changed back to hooks at the user's request, with no binary update.

| Plugin | Bytes | SHA-256 | Evidence |
|---|---:|---|---|
| `VitaJPOverlay_Kernel.skprx` | 38,620 | `6cc2dc868dc4403bcab252d689acd2f0779a55f9bd8069654b335eaffa571933` | PR #12 full installed-file readback; started after latest normal boot. |
| `VitaJPOverlay_Shell.suprx` | 158,564 | `32baceebd6b9af749b7b52a59336fe9225bef382f20eec10a60c41b930815d3c` | PR #13 full installed-file readback; started after latest normal boot. |
| `VitaJPOverlay_Text.suprx` | 7,936 | `84259631e0efe87472750a04a944a0d2cb7673f218c85f316ae3a450e6c7565f` | PR #12 full readback; prior physical candidate traffic observed, correct dialogue still unverified. |

Shell source: reviewed head `5826329ddcb1244b0e318f2c6c527181cc14d4d8`, merged at 11:26:11 UTC as `539c2888f61b28277bb7fd7bdd2b6a4bb5b330fd`. Both exact-head hosted jobs passed (host 52 s, Vita 4m25s). The initial 3,082 B auto config had SHA `b7c0dd2e5214c0d00612e832abeba2b5ea5dde67ad63875745764ce4ae4beeb0`. The current 3,083 B manual config is SHA `3833843d30aa9bd2bc294149d6c63b90f2909ce679ea05a2b2b731e9c101eabd`. The notes author independently hashed the readbacks and confirmed the later change only replaces `text_source = auto` with `text_source = hooks`.

Kernel/text implementation source: `6eef6c031aa1cc4bc162ebf31b8628d3098f0f5e`; final evidence head `69006af9e36347b75534c01900b48d43ed372981`; PR #12 merge `a5e0c50b1096b5ae7de450ae712e63b272dc7f17`. The evidence-only commit changed no binary.

## Preserved rollback and evidence

The latest private directory is `vita-agent-use/agent/overlay-ocr-picker-20261010/`. It contains shell/config/region backups, the deployment plan, transfer journals, full readbacks, `installation-report.json`, `status.after-reboot.txt` and `status.clannad-start.txt`. The report records upload completion and full readback verification; its physical OCR field at installation was `pending user dialogue test`, now superseded by the user failure above. Operation IDs and private configuration stay local.

The notes author inspected the fresh startup log. It is 706 B, SHA-256 `bc3577582f65bb2538c7e6bd424d1c65fe13db1aea74c9c8c75699bb8a71e6fb`. Its content/hash equals the earlier healthy startup, so freshness comes from the newly collected file and device modification time, not a different digest. The CLANNAD startup status is 2,056 B, SHA-256 `f88744dabecd69056a441fcbbeb0610975d4182834025f00992651a6a3549296`.

The previous low-memory shell `45f86b1ea83c08bdf61c1dfd174cc5356151e746a3659a0299218436a8f10c8c` remains backed up. Earlier kernel/text backups are under `vita-agent-use/agent/overlay-libc-20261010/`: kernel SHA `447f2e640973743f80dbeb6099e62b6e7c2b72fe85936af4f8d9c24dfd54f217`, text SHA `c146238b6e6544ec9ba27f1b03d35db89072c2fc4f3a05d0e75d94c679a777a7`. Preserve the installation-specific journals rather than mixing rollback sets.

No firmware, bootloader, tai configuration or region setting was changed. The overlay-mode changes were hooks→auto for the OCR test, then auto→hooks for the user's fixed manual-list choice. The emulator bridge and local scanner diagnostic commit `2d56ce4` were not installed.

## Environment and boundaries

- Physical device: PCH-1001, actual firmware 3.65.
- Title: CLANNAD, `PCSG00415`; tested after reboot, then closed by the user (latest completed app inventory empty).
- Controls/configuration: L+R overlay, Select+R subtitles, hooks text source, optional Google Lens OCR, existing local dictionary configuration.
- Offline generic capture matches two visible narration pages without a title profile. Corrected full-feed ranking displays matching sources in the top five, but those snapshots had no selected source.
- Live production Lens on the host exactly matches one short verified page from full-frame and cropped inputs. It does not establish physical capture/TLS or broad OCR accuracy. See [[18-OCR-anchor-memory-measurements]].
- An earlier launch/power-off report remains historical and unresolved; normal boot of the latest installed set is established separately.

This is a dated snapshot. PIDs, app state, tokens and connections can change. Read current state before another operation and preserve paired configuration/transfer state; see [[09-Device-operations-and-recovery]]. Next resolve the user's wrong-text concern and verify that a selected manual source follows dialogue. Region-sized capture PR #14 is merged and prepared separately, but no new binary is installed; it remains optional for the current manual experiment. See [[12-Open-questions-and-next-steps]].
