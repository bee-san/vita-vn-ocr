---
title: Open questions and next steps
created: 2026-10-10
updated: 2026-10-10
status: active
tags: [vita, next-steps, questions]
---

# Open questions and next steps

The immediate goal is to make the user's CLANNAD dialogue available in the Vita overlay and dictionary. Boot and picker allocation repairs are necessary completed steps, not the goal itself.

## Current decision

The generic SceLibc capture and ranking repair is merged, installed, read back and normally booted. Corrected full-feed emulator proof matches two visible pages through the same streams and shows them in the top five without a game profile. The remaining immediate question is the user's physical CLANNAD experience. See [[02-Current-device-state]] and [[16-Full-feed-picker-ranking]].

## Pending user test and follow-up

1. **Wait for the requested CLANNAD result.** The user has been asked to open CLANNAD, show dialogue, press L+R, choose the matching text with ▲/▼ and ×, then advance dialogue and reopen twice. Avoid duplicate requests while that response is pending.
2. **Verify actual selection and page changes.** Emulator snapshots had `selected=0`; manual selection and selected-source continuity remain separate hardware evidence. Collect current metadata while respecting one-client device ownership.
3. **If the picker is empty or wrong, isolate the boundary.** Verify the current game's text module/attachment, process/mapping identity, successful reads and decoding/ranking counters. The known-correct emulator decoder/ranker does not prove physical kernel mapping.
4. **If the device stops or the game exits, preserve the exact state.** Keep installed hashes, last markers and any completed coredump; use the established recovery path and private backups as needed. Do not assume the original boot ABI failure has returned.
5. **Once text follows pages, verify dictionary and lifecycle.** Exercise the bounded allocation fallback, overlay reopen, scene change, restart and suspend/resume. Measure overhead/memory with correct process ownership.
6. **Revisit title-specific work only when evidence requires it.** The two tested pages use generic imports without a profile. Other scenes, features and revisions remain untested; any profile needs exact module/instruction validation.

Do not reinstall or reboot merely because the user has not answered yet. The concrete candidate is already on the device, and boot is verified; the next observation should resolve the actual game workflow.

## Questions requiring evidence

| Question | Evidence needed |
|---|---|
| Are corrected SceLibc hooks installed on physical CLANNAD? | Individual hook return IDs plus actual call observations. |
| Does the physical picker show the matching dialogue in its five choices? | The emulator full-feed result passes; confirm actual shell rendering and user selection on the console. |
| Does copied text represent the visible page or a script cache? | Two immediate-source matches are now observed; preserve timing/source association and extend scene/choice/typewriter checks. |
| Is the generic scanner reading the intended game's mappings? | Successful map/read counters tied to current process and allocation identity. |
| Is CP932 sufficient, or are command/glyph tokens interleaved? | Bounded private captured-byte inspection compared with visible output. |
| Is a special CLANNAD hook necessary? | Negative/insufficient generic evidence after the real imports are covered. |
| Why did the later launch/power-off happen? | Exact installed build, last startup/capture stage, native error or coredump if available. |
| Does Paf fallback actually satisfy a dictionary lookup? | Physical allocation-path log and visible lookup result. |
| What is capture cost on console? | Repeated measurements tied to the correct process, clock and page workload. |

## Completion criteria

The user can open CLANNAD, show a page, select a matching source, advance to a different page and see current Japanese text remain available for dictionary lookup without a shutdown or memory-error gate. Record the exact game/build, firmware, plugin hashes and evidence. Compatibility with other games and PSP remains a separate task.

Related: [[01-Evidence-and-claims]], [[08-Validation-and-measurements]], [[02-Current-device-state]].
