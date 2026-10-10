---
title: Vita3K real-game runtime investigation
created: 2026-10-10
updated: 2026-10-10
status: two-page-capture-and-top-five-verified
tags: [vita3k, clannad, debugger, experiment]
---

# Vita3K real-game runtime investigation

This note separates problems in the isolated emulator/debugger setup from findings about CLANNAD and the native hook code. Synthetic plugin checks are in [[14-Retail-libc-hook-coverage]]; physical state remains in [[02-Current-device-state]].

## Real game executes SceLibc memmove

A bounded startup trace observed:

| Field | Value |
|---|---|
| Imported function | SceLibc memmove |
| PC/stub | `0x810A9458` |
| Caller LR | `0x81033D6F` |
| Source | NULL |
| Length | 0 |
| Elapsed at first recorded call | About 0.283 s; trace-relative, not a performance benchmark |

This establishes actual use of the import by this game, beyond the static import-table finding. It does **not** show dialogue, an encoding, useful positive-length copies, or the new Vita plugin capturing a real page. Zero-length capture should be ignored. The raw trace remains private at `build/clannad-offline/runtime-trace.jsonl` in the discovery worktree.

## Breakpoint continuation deadlocks in inspected debugger

Source revision: `b75d5e6292d5caa6a8411bacc9db1eca1531beab` of the local Vita3K source checkout.

- `vita3k/gdbstub/src/gdb.cpp`, `cmd_continue`, takes `thread->mutex` before calling `thread->resume(step)` for the inferior thread.
- `vita3k/kernel/src/thread.cpp`, `ThreadState::resume`, acquires the same mutex with `std::lock_guard<std::mutex>`.
- The mutex is nonrecursive, so this nested acquisition deadlocks on that path.

This matches the observed hang when continuing from the game's import breakpoint. It is an instrumentation limitation, not a conclusion that uninstrumented CLANNAD or the native plugin has the same deadlock. No corrected debugger build or resumed trace is claimed here.

Pinned source references: [continuation handler](https://github.com/Vita3K/Vita3K/blob/b75d5e6292d5caa6a8411bacc9db1eca1531beab/vita3k/gdbstub/src/gdb.cpp), [thread resume](https://github.com/Vita3K/Vita3K/blob/b75d5e6292d5caa6a8411bacc9db1eca1531beab/vita3k/kernel/src/thread.cpp).

## Black captures were a display-backend issue

The uninstrumented isolated game rendered Japanese startup notices when run in Xvfb with `SDL_VIDEODRIVER=x11`. An inherited Wayland setting had caused earlier black captures. Setting the backend explicitly was sufficient for the observed startup; no firmware installation was added for that result.

Do not generalize this into “all Vita3K games need no firmware,” or assume the game reached dialogue just because notices rendered. The investigator subsequently passed first-run UI and reached the narration matched below. Screenshots stay private and are not copied to this notebook.

## SDL keyboard input caveat

Another emulator-only block occurred at the first-run save-create OK dialog. At the same pinned Vita3K revision, `vita3k/sdl-frontend/src/session.cpp` lines 292–349 handles key-down via `handle_ime_keydown` without a key-up/controller-keyboard-state update in that event loop. `vita3k/app/src/app_init.cpp`, `poll_overlay_input`, reads `ctrl.keyboard_state`. The notes author independently inspected both source paths with `git show HEAD:<path>`; sparse checkout meant the SDL frontend file was not present as an ordinary working-tree file.

The isolated Qt frontend launched successfully. Its C key dismissed the save-create dialog and reached the PROTOTYPE logo, confirming this input route worked. A later run reached the first narration and produced matching records, described below. Pinned references: [SDL event loop](https://github.com/Vita3K/Vita3K/blob/b75d5e6292d5caa6a8411bacc9db1eca1531beab/vita3k/sdl-frontend/src/session.cpp), [controller polling](https://github.com/Vita3K/Vita3K/blob/b75d5e6292d5caa6a8411bacc9db1eca1531beab/vita3k/app/src/app_init.cpp).

Treat this as an input-delivery limitation in that frontend, not evidence that the native hook or physical controls fail. Avoid sending more identical keyboard events before establishing that they reach the game control state.

## Operating boundary for this experiment

One investigator owns the isolated game's debugger connection. Do not attach another client or resume/kill the game from a second agent while a trace is active. The private run configuration, memory dump and renderer state belong to this experiment; none changes the physical Vita's installed plugins.

## Test-only transport for the real game

A private `VjoClannadTestBridge.suprx` exports API 8 and `vjoTextSubmit` under `VitaJPOverlayForUser`, allowing the **unchanged production text plugin** to run inside real CLANNAD in isolated Vita3K. It uses the production decoder and source ranking. It is an emulator research helper and must never be deployed to the physical Vita.

Initial bridge SHA-256: `a2941cc98e157a82945f7d6307e0c201c6da57ac8bd780ffc57cdc4f3ecb5c17`. Production text artifact: `84259631e0efe87472750a04a944a0d2cb7673f218c85f316ae3a450e6c7565f`. Source, README and build live privately under `build/clannad-offline/bridge/`. The isolated emulator loads bridge then text plugin from its own per-title `ux0:tai/config.txt`; this does not edit physical-device configuration.

The bridge bounds each submission to 1,024 B and the remaining SDK-reported memory-block range, requires a terminator for length-zero events, and avoids unbounded strlen. These user-space checks are not atomic with a concurrent free. The production 24-source table filters/ranks observations; its private log is capped at 16 MiB and can contain copyrighted text, so publish only derived metadata.

This bridge can establish interception and decoding in the real game. It does not verify physical kernel mapping pinning, process isolation, syscall entry, memory ownership or startup. Its `map_rejects` may include emulator stack/module mappings absent from the SDK allocation inventory; they are not proof that the actual kernel would reject those addresses. The helper has now run with the production text plugin and emitted records matching the first visible narration page; later checks matched two pages through the same immediate source, then verified top-five ranking with the full feed after correcting a comparator defect.

## First visible narration matches capture records

Root independently inspected private `dialogue-3.png` and matched the first visible narration against five records in `captures-first-run.log`. The saved log is 81,920 B and contains 506 TEXT records. No `.vjhook` game profile was loaded; the text plugin bytes remain the production artifact listed above.

| Matching stream address | Call-site inspection |
|---|---|
| `0x81023C23` | BLX to memcpy stub `0x810A9388` |
| `0x8102CA7F` | Not included in the four disassembled sites at this update |
| `0x810A2393` | BLX to memcpy stub `0x810A9388` |
| `0x810A34EF` | BLX to memcpy stub `0x810A9388` |
| `0x81028773` | BLX to memcpy stub `0x810A9388` |

This is actual game text flowing through generic capture, not only a synthetic test. It remains possible that some records came from a script cache or prefetch rather than a current-page event. The log also contains false positives and mapping rejections; 506 TEXT records are not 506 correct dialogue pages. Establish a stable stream changing with subsequent visible pages before claiming the source follows dialogue. Nothing in this emulator result verifies physical kernel mapping or console installation.

## Two pages matched, then full-feed ranking exposed a defect

Evidence follow-up `d1ee0dc` records two manually checked visible pages through immediate SceLibc memcpy source `0x8102CA7F`:

| Visible page | Source updates in first helper | Input bytes | UTF-8 bytes |
|---|---:|---:|---:|
| 1 | 1 | 17 | 24 |
| 4 | 2 | 13 | 18 |

A delayed backlog source was excluded. Private screenshot/log hashes are recorded in the public benchmark; no dialogue is included there. The first helper filtered short/non-Japanese events **before** source ranking. That changes what competes for the top five, so it cannot prove the production picker will show the matching source.

The follow-up bridge fed all production observations into ranking and caught a real defect: the initial matching line was retained but hidden outside the displayed top five. On page 4, the full feed reported three updates, because the first helper had filtered out a one-kanji page. This explains the count difference rather than invalidating either measurement. See [[16-Full-feed-picker-ranking]] for exact baseline counters and the corrective experiment.

## Corrected full-feed result

With production comparator revision `6eef6c0`, the fresh full-feed run verified the same three matching streams on two visible pages. Page 1 sequence 424 shows ranks 2/3/4 with updates 1. Page 4 sequence 1680 shows ranks 3/4/5 with updates 3 and different current text. Both snapshots are complete and `selected=0`; no manual source selection occurred. The exact proof status is `generic_capture_and_top5_ranking_verified_for_two_visible_pages`, with no raw dialogue in the JSON.

The new helper SHA is `106be9d495ce17b622f67913d0f5cd8c09ce7dd767c5e647c59b85095364c316` (27,164 B); the production text plugin remains unchanged. The initial failing proof is retained as `offline-capture-proof-before-ranking.json`. See [[16-Full-feed-picker-ranking]] for before/after tables and hardware limits.

## Next useful evidence

The two-page generic-capture/top-five gate now passes. Next establish physical kernel transport, actual shell picker rendering, manual source selection, selected-source continuity and dictionary behavior. Broader scenes and performance remain separate measurements. Preserve the debugger/input failures and original traces as tooling evidence rather than mistaking them for native game incompatibility.

Related: [[07-CLANNAD-investigation]], [[08-Validation-and-measurements]], [[12-Open-questions-and-next-steps]], [[2026-10-10-Investigation-log]].
