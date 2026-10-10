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

The inspected game imports `SceLibc` copy/string functions, while the first generic backend targeted different NIDs. The correction is implemented, built and validated in synthetic tests; draft PR #12 is awaiting real-game evidence. Test that corrected path before concluding CLANNAD requires a special instruction-site hook. See [[07-CLANNAD-investigation]] and [[14-Retail-libc-hook-coverage]].

## Ordered investigation

1. **Verify the restored baseline.** Refresh actual connected state, installed text-plugin hash and running-app inventory; preserve the working boot kernel/shell. The latest state note is a snapshot.
2. **Review the corrected import coverage and establish actual attachment.** Draft PR #12 covers the two C-library families with bounded capture. Synthetic tests passed; confirm actual game attachment and useful observations. Keep zero matching imports visible as a diagnostic outcome.
3. **Complete review before another hardware install.** The production plugin now passes 255 isolated Vita3K checks with retail alias imports; the release build and ABI audit pass. Review the exact head and match the tested kernel/text pair, because bounded automatic-encoding submission changes the kernel as well. Do not deploy unrelated diagnostic experiments.
4. **Run a controlled physical capture test.** Install exact reviewed bytes with backup/readback, establish successful hook attachment, then display two distinct dialogue pages. Record candidate count, selected-source behavior and user verification separately.
5. **If candidates stay empty, isolate why.** Confirm game PID/mapping ownership, readable memory coverage, decoder/filter rejections and hook observations. An advancing scan address is not enough.
6. **Only if generic capture cannot identify current dialogue, investigate a title profile.** Locate the actual decoder/display boundary, establish register/pointer/encoding behavior and verify instruction signatures for the exact revision. Avoid treating an emulator address as a native profile.
7. **After capture works, verify dictionary and lifecycle.** Test bounded fallback, overlay reopen, page advancement, scene change, game restart and suspend/resume. Measure overhead and memory with correct ownership.

## Questions requiring evidence

| Question | Evidence needed |
|---|---|
| Are corrected SceLibc hooks installed on physical CLANNAD? | Individual hook return IDs plus actual call observations. |
| Does copied text represent the visible page or a script cache? | Two advancing pages at minimum; broader scene/choice/typewriter checks later. |
| Is the generic scanner reading the intended game's mappings? | Successful map/read counters tied to current process and allocation identity. |
| Is CP932 sufficient, or are command/glyph tokens interleaved? | Bounded private captured-byte inspection compared with visible output. |
| Is a special CLANNAD hook necessary? | Negative/insufficient generic evidence after the real imports are covered. |
| Why did the later launch/power-off happen? | Exact installed build, last startup/capture stage, native error or coredump if available. |
| Does Paf fallback actually satisfy a dictionary lookup? | Physical allocation-path log and visible lookup result. |
| What is capture cost on console? | Repeated measurements tied to the correct process, clock and page workload. |

## Completion criteria

The user can open CLANNAD, show a page, select a matching source, advance to a different page and see current Japanese text remain available for dictionary lookup without a shutdown or memory-error gate. Record the exact game/build, firmware, plugin hashes and evidence. Compatibility with other games and PSP remains a separate task.

Related: [[01-Evidence-and-claims]], [[08-Validation-and-measurements]], [[02-Current-device-state]].
