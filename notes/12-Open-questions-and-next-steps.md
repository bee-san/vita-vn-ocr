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

The user reports the physical manual-hook workflow works. Fresh logs verify a selected CALL source changes through five updates and completes dictionary jobs across reopens. The successful source is kind 2 CALL, not the installed kind 4 REGISTER profile. Exact visible-page count was not separately stated and no independent physical screenshot was obtained. See [[23-Physical-native-hook-success]].

The current kernel/shell/profile are installed/read back and normally booted; manual mode is preserved. The final refresh in the success log intentionally clears selection, so source ID 18 must not be assumed persistent. Do not request the same test again merely because broader compatibility remains unmeasured.

## Follow-up work when requested

1. **Preserve the working manual configuration and evidence.** Keep exact installed hashes, successful source metadata and private diagnostic logs. Avoid unrequested mode changes/reinstalls.
2. **Separate generic success from profile proof.** The title profile has strong emulator evidence, but the successful physical source was CALL. A later profile-specific check must identify actual REGISTER observations, not infer causation from installation timing.
3. **Broaden validation deliberately.** Other scenes, speakers, choices, restarts and suspend/resume remain untested; record their real outcomes without turning the current success into blanket compatibility.
4. **Keep OCR optional.** The smaller kernel buffer is installed, but no successful physical OCR request has been measured. Native hooking already supplies the requested current workflow.
5. **Diagnose screenshots only from new evidence.** pngshot's encoder overlap makes a PNG/JPEG mismatch likely, but no live returned PNG or alternate screenshot retrieval is confirmed. The dirty local agent backend remains untouched.

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
| Does the completed Paf dictionary job show the intended result? | Allocation/job rc=0 is now logged; verify useful visible output for matching dialogue. |
| What is capture cost on console? | Repeated measurements tied to the correct process, clock and page workload. |

## Completion criteria

The user can open CLANNAD, show a page, select a matching source, advance to a different page and see current Japanese text remain available for dictionary lookup without a shutdown or memory-error gate. Record the exact game/build, firmware, plugin hashes and evidence. Compatibility with other games and PSP remains a separate task.

Related: [[01-Evidence-and-claims]], [[08-Validation-and-measurements]], [[02-Current-device-state]].
