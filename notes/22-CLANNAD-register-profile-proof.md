---
title: CLANNAD register profile proof
created: 2026-10-10
updated: 2026-10-10
status: two-page-emulator-proof-physical-validation-pending
tags: [vita, clannad, register-hook, evidence]
---

# CLANNAD register profile proof

A title-specific register hook now matches two different visible CLANNAD narration pages through the production ARM observer in isolated Vita3K. The same captured source stays unchanged during two idle periods and changes when the visible page advances. This is stronger timing evidence than finding the expected line somewhere in prefetched script text. The profile and new plugins are **not installed on the physical Vita**.

## Candidate evolution

| Candidate | Observation | Outcome |
|---|---|---|
| Function entry at segment 0 + `0x2C906`, r1 as CP932 | Entry events fire, but original r1=1 rather than a text pointer | Rejected wrong-register descriptor; never installed. |
| Same entry, fifth argument at `*(original SP)` (register D, one indirection) | Expected lines occur, but the entry also captures preloaded future lines | Rejected as the current-dialogue choice; expected-line occurrence alone is insufficient. |
| Segment 0 + `0x2CA70`, Thumb, r6 CP932, no dereference/padding | Current register source matches two visible pages and remains stable with no input | Current emulator-proved candidate; physical validation remains pending. |

The final profile targets native module `VITA`, module NID `0xC5187D60`. Its unique eight-byte instruction signature and placement were checked; the proof records no direct branch target entering the overwritten interior, and independent placement review found no blocker. This is revision/placement evidence, not proof of hardware kernel stack access or every instruction/runtime path.

The final profile is 63 B with SHA-256 `99c73600540f5c441939a06f6cc727ec21b284fa055bc936f7bb0359382f9f81`. Root is preparing a separate profile/docs PR from merge `5e5bc25` in `/home/bee/Documents/src/vita-jp-overlay-clannad-profile`, planning `profiles/PCSG00415.vjhook` as small derived revision-guarded configuration. The full instruction bytes are not reproduced in this notebook. Game executables, screenshots, dialogue and complete logs remain private.

## Complete two-page evidence

The test uses unchanged production Text plugin SHA `84259631...` and a test-only bridge. REGISTER stream `0x70000000`, kind 4, encoding 3 is the profile source. Emulator source ID 15 is run-specific, not an identifier to hard-code for the physical picker. Score 0 here reflects no OCR reference and is not a failed visible-text comparison.

| Checkpoint | Complete source sequence | Source rank | Update count | UTF-8 bytes | No-input interval | Result |
|---|---:|---:|---:|---:|---:|---|
| Visible page 1 | 906 | 2 | 1 | 24 | 10.330 s | Exact visible match; identical complete source row after idle. |
| Visible page 4 | 3,432 | 5 | 3 | 18 | 11.290 s | Different current visible match; identical complete source row after idle. |

The byte/character counts are small derived metadata; no dialogue is included here. Generic import hooks remain enabled, so competing sources still exist. The profile does not itself force picker selection or suppress other sources. No shell selection/follow interaction was exercised in this emulator proof.

## Test bridge boundary

Vita3K's SDK memory lookup does not register directly allocated thread stacks. The private bridge therefore obtains explicit bounds for the current thread through `GetThreadInfo` before its bounded copy. The emulator's process-ID field is unimplemented in this path. This workaround is confined to the research helper; it does not prove real kernel pin/copy syscalls, process isolation, `FindProcMemBlock` stack reads or physical safety.

The final bridge is 27,980 B, SHA-256 `ccde7a587ad1c284b7540688f6842ab58c63e84e31fda39ac641d2df5deea13e`; its source is 11,586 B, SHA `44aba5e6c792a67bcabc4510147930e2f1e74d5e4b03a007b3c9f44b3b629c9b`. It is never a device deployment artifact. The isolated emulator has been stopped after evidence capture.

## Provenance and remaining work

Private evidence is under `/home/bee/Documents/src/vita-jp-overlay-discovery/build/clannad-offline/specific-hook/`:

- `README.md` describes the experiment and limitations.
- `native-profile-proof.json` records status `native_register_current_dialogue_verified_two_pages`, timestamp `2026-10-10T12:03:01.739084+00:00`, profile/bridge/module hashes and complete page/idle checkpoints.
- Proof JSON is 7,194 B, SHA-256 `1b1ce2e79019aff2ce9b45d437da20bc2de3288f0f474b348532f66afa162eda`.

The notes author inspected the bounded proof metadata and independently verified all 17 referenced artifact hashes, without copying game content. Visible-line comparison is attributed to the capture investigator; this notes pass did not repeat visual inspection. Hash integrity does not substitute for physical execution.

Next publish/review the concrete revision-guarded profile, then verify actual physical attachment and bounded register capture. Confirm a displayed candidate matches two changing pages after deliberate selection, including reopen and dictionary behavior. Other scenes, speakers, choices, releases and long-running stability remain untested.

Related: [[07-CLANNAD-investigation]], [[20-Fixed-manual-hook-workflow]], [[21-Picker-diagnostics-and-screenshot-provenance]], [[02-Current-device-state]], [[12-Open-questions-and-next-steps]].
