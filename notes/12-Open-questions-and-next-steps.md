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

The physical test did not produce a usable workflow: the user saw changing 0% hooks and irrelevant text. The user explicitly wants one OCR screenshot first, then the matching source, and accepts Google Lens to reduce local memory demand. The built follow-up is overlay PR #13, merged as `539c2888` from exact reviewed head `5826329`. Shell/config readbacks and normal boot are verified; CLANNAD launches, but the user screenshot test failed with “unexpected response (-108)” and no game text. Paired logs identify failed 2,040 KiB kernel capture allocation before Lens. Validate a region-sized allocation repair and original-error propagation before another physical attempt; see [[19-Physical-OCR-source-failure]]. Do not ask the user to keep trying the same manual stream-picker flow as if the earlier hardware test were merely unanswered.

## Follow-up work

1. **Make the requested OCR-first path concrete.** Root owns worker/configuration and eventual physical deployment; delegated work covers OCR memory and picker rendering. The working tree now implements a single 384 KiB anchor, stable ≥50% matches, LISTEN without full scans, and explicit user selection. Current head `5826329` passes the final 10 host suites in 4.18 s, including 37 worker cases, and the local Vita release/ABI check. The config-comment mismatch is fixed; both exact-head hosted jobs passed (host 52 s, Vita 4m25s), and PR #13 merged as `539c2888`. Live production-Lens host checks pass for one short verified page; physical OCR remains pending.
2. **Make match state truthful.** The prior installed manual mode had no OCR reference. A 0% label there does not mean a calibrated match failed; distinguish uncalibrated discovery from a computed score.
3. **Stop unrelated source churn from driving the default UI.** The baseline starts full memory discovery on open and republishes the picker for every source-table sequence change. Verify the actual corrective behavior rather than assuming a comparator tweak alone fixes the interaction.
4. **Validate the constrained OCR path and failure handling.** CLANNAD already exhausted large result allocations. A one-time OCR anchor must be tested under the relevant memory budget and leave useful, stable feedback on failure.
5. **Preserve the completed deployment record.** Shell/config readbacks match reviewed head `5826329`; only hooks→auto changed. Normal boot is verified. Preserve this known bootable set while isolating the new source-stage failure.
6. **Resolve the source failure, then retest the actual workflow.** The -108 UI wording hid a confirmed full-frame kernel capture OOM; fix and validate region-sized storage and original-error handling. Then One screenshot should yield relevant current-page text/source matching, then a chosen source should follow another page. Verify dictionary behavior and reopens afterward. Candidate count, normal boot and emulator snapshots alone do not satisfy this.

The prior boot/install and two-page emulator results remain valid within their scopes. The physical experience remains unverified/failed until the requested OCR-first interaction works for the user. See [[17-OCR-first-physical-workflow]].

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
