---
title: Current device state snapshot
created: 2026-10-10
updated: 2026-10-10
snapshot_utc: "2026-10-10 10:32"
status: needs-live-refresh
tags: [vita, device, handoff]
---

# Current device state snapshot

**This is a dated handoff, not live telemetry.** Latest user report: the console is on. Latest update from the active investigator: game closed, native running-app inventory empty, old text plugin restored with full readback. Experimental scanner/kernel diagnostics have **not** been installed. The generic-import correction is built and locally validated at `0a66015994dbb3f12a3a506f0c0e4cab8ec72fdd`, draft PR #12, and is not installed. Its new kernel/text artifacts are listed separately in [[14-Retail-libc-hook-coverage]]; do not confuse them with this installed set.

Normal boot was already physically verified for the firmware fix and again after installing the low-memory shell. The user later reported another power-off around a CLANNAD launch. Its cause is not established; do not turn the earlier boot success into a claim that all later shutdowns are resolved.

## Last verified installed plugin set

All three use custom API 8. `ur0:tai/` is the active plugin location in this investigation. Check the current runtime and readback before the next deployment.

| Plugin | Bytes | SHA-256 | Last evidence |
|---|---:|---|---|
| `VitaJPOverlay_Kernel.skprx` | 38,620 | `447f2e640973743f80dbeb6099e62b6e7c2b72fe85936af4f8d9c24dfd54f217` | Boot-fix upload, full readback and successful normal boot; unchanged by the low-memory shell install. |
| `VitaJPOverlay_Shell.suprx` | 156,196 | `45f86b1ea83c08bdf61c1dfd174cc5356151e746a3659a0299218436a8f10c8c` | Low-memory upload and full readback, followed by normal boot and user-visible picker. |
| `VitaJPOverlay_Text.suprx` | 7,680 | `c146238b6e6544ec9ba27f1b03d35db89072c2fc4f3a05d0e75d94c679a777a7` | Restored original production text plugin; investigator confirmed full readback. |

The preceding shell backup is 155,068 B, SHA-256 `e5a408f3b68aaf809a0e37ada54160fbaa91d1b615d0b12b2774ab32a58bde57`. It belongs to the fixed-boot but eager-allocation version; it is not the current low-memory shell.

## Confirmed behavior

- Device: physical PCH-1001, actual firmware 3.65.
- Boot markers: kernel API 8; shell registration 0; final `bootstrap: worker started`.
- Kernel and shell reached native `STARTED` state 6. The game text module also reached that state in an earlier CLANNAD run.
- Title under investigation: **CLANNAD, PCSG00415**.
- User could open the native picker after the low-memory install; no captured candidate followed dialogue.
- Input configuration at the last inspected point: L+R overlay, Select+R subtitles, hooks text source, local dictionary.
- No firmware, bootloader or broad tai configuration migration was part of these repairs.

## Do not reuse transient state

PIDs, app IDs, tokens, RPC IDs, transfer IDs and connection state are session-specific. Obtain them from current native inventory or saved private operation state. A PID from an earlier boot is not a valid target after reboot.

The existing paired control identity and its private directory are preserved. See [[09-Device-operations-and-recovery]]; the public notebook deliberately contains no network address, credential, certificate fingerprint or identity string.

## Next update checklist

Record the new source commit, full plugin hashes and exact readback result. Record whether normal boot, game launch, hook attachment, candidate discovery and two advancing pages each passed. Keep these as separate fields even if one fails. Link the new private log in [[11-Sources-and-artifacts]], then append the event in [[2026-10-10-Investigation-log]].
