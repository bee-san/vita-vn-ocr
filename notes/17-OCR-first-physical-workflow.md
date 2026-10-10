---
title: Physical UX failure and OCR-first workflow
created: 2026-10-10
updated: 2026-10-10
status: installed-boot-verified-physical-ocr-pending
tags: [vita, clannad, ocr, ux, requirements]
---

# Physical UX failure and OCR-first workflow

The user tested the installed generic capture/ranking repair and saw a flood of changing hooks labelled 0%, then irrelevant text. This is a failed user workflow. A booted plugin, nonzero candidate count and a two-page emulator match do not establish a usable physical text hook.

## Explicit user preference

The default interaction should **take one OCR screenshot first and show the matching source**. The user also explicitly accepted **Google Lens** to reduce local memory demand; it is already the configured/default backend and is retained. The user does not want to sort through an unstable stream of raw hooks as the normal starting experience. Record this as an accepted requirement, not an optional idea awaiting another preference question.

The deployed configuration was `text_source=hooks`, deliberately selected by root during earlier debugging. Root acknowledged that choice. In that mode no OCR reference existed, so 0% was an uncalibrated score, not evidence that the current visible page had been compared and rejected. The repository configuration default was already `text_source=auto`; the physical override matters more than that source default.

The status log's `ocr_mode auto` is a different setting from `text_source`. Do not read that line as proof that OCR calibration ran.

## Physical metadata

Saved evidence: `vita-agent-use/agent/overlay-libc-20261010/status-clannad-user-spam.txt`, 2,123 B, SHA-256 `5e5ee8dcc624af6b18fcb7f892f4b0ac8f215b57c2a30d87dfe8af547858e189`. It records:

| Point | Candidates | Scanning | Sequence / cursor | Selected |
|---|---:|---:|---|---:|
| Before opening | 1, then 3, 4, 5 | 0 | Sequence 6→12 | 0 |
| First recorded open | 5 | 0 | Sequence 293, cursor `0x80000000` | 0 |
| After opening | 5 | 1 | Sequence 479, cursor `0x80FC0000` | 0 |
| Later reopen | 5 | 1 | Sequence 1,065, cursor `0x818B8000` | 0 |
| Later reopen | 5 | 1 | Sequence 1,852, cursor `0x825C8000` | 0 |
| Last saved open | 5 | 1 | Sequence 3,035, cursor `0x8449C000` | 0 |

This shows candidate traffic and continued discovery. It does not identify any candidate as the visible dialogue, and the saved sequence never establishes a selected source. The log contains metadata rather than raw dialogue.

## Baseline mechanism inspected at merged revision a5e0c50

`open_overlay` starts `VJO_TEXT_DISCOVER` when no source is selected. This expands collection beyond import observations into the full memory scanner, mixing potential current text with static strings. `poll_hooks` republishes the picker whenever the overall source-table sequence changes, even if that change is not useful to the displayed choice.

The baseline renderer always formats `c->score` as a match percentage. Without an OCR reference, that produces misleading 0% labels. The default UX combines unanchored results with rapid republishing, which explains why an offline top-five snapshot was insufficient validation of the interactive experience.

The source-table ranking repair still matters: it prevents short/frequent noise from burying readable sources in a snapshot. It does not identify current visible text by itself or stabilize the shell interaction. Keep this later failure separate from the fixed comparator regression.

## Follow-up ownership and scope

Branch `fix/ocr-first-hook-picker` starts from merged `a5e0c50`. Root owns worker behavior, configuration and device work; another agent investigates OCR memory and another owns picker rendering. Notes work remains separate and makes no device calls.

The follow-up is published as [overlay PR #13](https://github.com/bee-san/vita-jp-overlay/pull/13). Implementation commit `4205be7034a2b48df45e65c7f253e7c7524b349a` and current follow-up head `5826329ddcb1244b0e318f2c6c527181cc14d4d8` are built, reviewed, merged and installed with full shell/config readback and normal boot verified. Exact-head hosted validation is discussed below; live production Lens has since matched one verified short page on the host, while device behavior remains unverified. The installed kernel/text remain the verified `6cc2dc86...` / `84259631...` pair, now accompanied by OCR-first shell `32baceeb...` replacing the backed-up low-memory shell `45f86b1e...`.

## Implemented follow-up

The notes author inspected the worker/view/renderer changes. Local validation establishes the behavior under test stubs and build tooling; physical rendering and physical OCR remain pending; the host endpoint check is documented separately in [[18-OCR-anchor-memory-measurements]].

- A dedicated one-time Lens OCR anchor job uses **one 384 KiB workspace**, with no dictionary lookup in that job. It attempts a physical memblock, then a bounded 384 KiB Paf allocation if necessary. The second result arena has no storage during anchoring.
- The anchor is released only after job completion. A text-ready event occurs before the worker has returned, so it cannot safely free the workspace. The reference is copied into bounded scratch/view storage before release; no displayed pointer retains anchor-owned memory.
- The default path uses `VJO_TEXT_LISTEN`, not full memory discovery. Because the reference syscall can start discovery for older clients, the worker explicitly returns to LISTEN after setting the OCR reference.
- Matching choices are stable snapshots filtered to at least **50% normalized Unicode similarity**. This is a string-match score, not a probability of OCR correctness. The follow-up removes automatic source selection; the user chooses the displayed match.
- A new match/retry occurs only through explicit open/reopen or Triangle action. Background source events do not repeatedly trigger OCR or continuously replace the displayed choices.
- View state distinguishes none/matching/ready/failed, includes a bounded 256 B UTF-8 screenshot excerpt and whether OCR retry is available. The renderer shows screenshot text, a primary choice and shorter alternatives, with percentages only when the reference is valid. Codepoint-safe previews remove display/control noise without altering capture text.

## Lifecycle defects caught during review

| Trigger | Defect caught | Intended corrected behavior |
|---|---|---|
| User clicks a row after the underlying source changed | A stale displayed choice could silently select different current text | Validate the displayed identity/content before selecting; keep the picker if it no longer matches. |
| Close overlay, advance the game, reopen while an old anchor is running | Old screenshot completion could calibrate the new screen | Cancel/invalidate that generation and discard stale completion; release only after completion, then request the current screenshot. |
| Return from a dictionary result to the picker | Reference could be lost or associated with the previous screen | Request a fresh screenshot and publish its reference with the new picker. |

## Validation and candidate artifact

Host coverage includes anchor heap ownership, explicit/retryable OOM, free-after-completion cancellation, reopen during an anchor, dictionary-to-picker refresh and stale clicked-row rejection. The final full **10-suite host run passed in 4.18 s**, including **37 production-worker cases**. The last case verifies that cancellation clears the subtitle busy indicator. The earlier 4.86 s full run and 36-case focused run were intermediate results, superseded by this final build. The Vita SDK Release build with `VJO_WITH_NCNN=OFF` and packaged syscall ABI check passed. Both independent reviewers reported no remaining blockers after the lifecycle fixes.

The UI preview formatter passed 650 ASAN/UBSAN cases and the renderer object compiled. These checks do not show the physical rendered UI. The separate Lens memory probes and their limitations are in [[18-OCR-anchor-memory-measurements]].

Current installed shell: **158,564 B**, SHA-256 `32baceebd6b9af749b7b52a59336fe9225bef382f20eec10a60c41b930815d3c`. This supersedes the earlier 158,516 B candidate `28097a40...`. The kernel/text binaries remain unchanged. Root backed up the device's existing shell/config/region; the shell/config uploads completed with full readbacks and normal boot verified. The only configuration change is `text_source=hooks` → `auto`, preserving Lens and other selected settings.

At head `a048faa`, hosted run `38047960757` failed the core configuration golden equality check: the updated explanatory comment in `docs/config.example.ini` was not synchronized with the generated default string in `core/config.c`. Hosted worker tests passed. Head `5826329` fixes the documentation/default-string mismatch and the cancelled subtitle state. The full host/Vita checks above were rerun afterward. Both hosted jobs at `5826329` passed in run `38048108519`: host 52 s and Vita 4m25s. PR #13 merged at 11:26:11 UTC as `539c2888f61b28277bb7fd7bdd2b6a4bb5b330fd`. The older mismatch is a superseded CI failure, not a physical OCR failure. Shell/config installation and normal boot are verified; CLANNAD is in the foreground. The user has been asked to show dialogue and press L+R, then report the excerpt/match or exact error. Physical OCR and source selection are still pending.

Live Lens OCR on the Vita, available device memory and the physical interaction remain unverified; the separate host endpoint result is established only within its narrow scope. The source-level 384 KiB workspace is a bounded allocation policy, not a measured total additional memory figure.

## Validation that now matters

Verify one screenshot supplies the initial OCR anchor and the UI presents a relevant, stable match. Distinguish “no OCR reference,” “OCR failed,” “matching in progress” and an actual computed percentage as appropriate to the implementation. Confirm the flow works under CLANNAD's memory constraints and does not re-run OCR continuously merely because hook observations arrive.

Then select/follow a source through another visible page and verify dictionary behavior. Test the rendered interaction, not only the candidate table. Preserve any failed OCR/matching result with its configuration and last stage; do not label arbitrary Japanese bytes as current dialogue.

Related: [[02-Current-device-state]], [[06-Low-memory-overlay]], [[07-CLANNAD-investigation]], [[16-Full-feed-picker-ranking]], [[12-Open-questions-and-next-steps]].
