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

The corrected generic SceLibc hooks capture two visible narration pages in actual CLANNAD under the emulator bridge, without a game profile. A stronger full-feed check found a production ranking defect; commit `6eef6c0` fixes it and a fresh run now places all three matching sources in the top five on both pages. The remaining immediate gate is the physical Vita workflow. See [[07-CLANNAD-investigation]] and [[16-Full-feed-picker-ranking]].

## Ordered investigation

1. **Verify deployment preconditions and exact-head review/checks.** Root refreshed read-only backups of the installed kernel/text, confirmed connection and an empty app inventory, and updated the private plan with the new kernel hash. Refresh again if state changes; source snapshots are not live telemetry.
2. **Install the tested kernel/text pair when the primary investigator proceeds.** Preserve backups and perform full readback before reboot. The physical shell already matches the candidate build. Never install the emulator test bridge.
3. **Run the normal-boot and controlled physical test.** Establish startup/API state, launch CLANNAD, show a page and confirm matching picker choices. The user asked to be told when this real-Vita test is needed; make the concrete test request at that point.
4. **Select a source and advance dialogue.** Emulator snapshots had `selected=0`; actual manual selection and selected-source continuity remain untested. Verify at least two distinct visible pages and close/reopen behavior.
5. **If physical candidates are absent, isolate the transport boundary.** Check actual hook attachment, game PID/mapping ownership, successful reads and decoder/rejection counters. The emulator bridge does not implement kernel page pinning or process isolation.
6. **Verify dictionary and lifecycle.** Test bounded allocation fallback, overlay reopen, scene change, game restart and suspend/resume. Measure overhead/memory with correct process ownership.
7. **Revisit title-specific work only if evidence requires it.** The two tested pages use generic imports with no profile. Other scenes, engine features and revisions remain untested; any new profile needs exact module/instruction validation.

The current candidate passed 10 host suites, a Vita release build/ABI audit, 264 synthetic Vita3K checks, and the two-page corrected full-feed ranking experiment. These establish a concrete reviewed candidate for hardware testing, not success on the physical device. No physical install has occurred at this note snapshot.

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
