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

The user now explicitly chooses a fixed manual hook list and accepts ordinary text hooking without OCR if the list stays stable. Root switched only auto→hooks with full config readback; no reboot or new binary was needed. A source can be selected and physical Paf dictionary jobs complete, but the user thinks the text is wrong. Root requested a short example because the screenshot tool fails at JPEG encoding. See [[20-Fixed-manual-hook-workflow]].

The OCR-first shell and its boot remain verified. Its earlier screenshot failure is a confirmed full-frame kernel capture allocation failure before Lens. Region-sized storage is now merged as PR #14 `c888b59`; it is optional for the current manual experiment and is not installed. See [[19-Physical-OCR-source-failure]].

## Follow-up work

1. **Resolve the current wrong-text concern.** Compare the requested short example to the visible line; do not publish dialogue in the notebook. Six selected bytes and one update do not establish a current dialogue stream.
2. **Validate manual continuity.** Keep the list stable, deliberately select a relevant source and verify two advancing pages. The source-table sequence changing does not mean the selected text changes.
3. **Confirm dictionary usefulness.** Paf allocation and `rc=0` are now physical evidence, but the on-screen result still needs to correspond to the intended text.
4. **Preserve the user's mode choice.** The current config is hooks, with Lens/region retained. Avoid silently changing back to auto or rebooting solely for a config change that takes effect on the next open.
5. **Keep the merged optional capture repair distinct from deployment.** Region-sized storage, frozen geometry/capacity, single-pass JPEG release, multipass preservation and cleanup passed host/build/review checks in PR #14. Prepared binaries are not installed; hardware allocation and capture remain unverified.
6. **Preserve each outcome separately.** Boot/readback, host Lens, failed physical capture, successful allocator/job returns and correct visible dialogue are distinct observations. A usable text hook requires the last one.

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
