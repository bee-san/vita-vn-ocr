---
title: Repositories, worktrees and revisions
created: 2026-10-10
updated: 2026-10-10
status: snapshot
tags: [vita, repositories, revisions, handoff]
---

# Repositories, worktrees and revisions

Multiple sessions share these directories. Inspect `git status --short --branch` before editing; a worktree name is not proof of its current branch. Never reset unrelated work or push directly to main. All changes in these projects go through their own focused branch and PR.

## Workspace map

Paths below are local research locations. Private artifact directories remain untracked/ignored; links do not grant permission to publish their contents.

| Local path under `/home/bee/Documents/src/` | Purpose and last observed ownership |
|---|---|
| `ps-vita-ocr/` | User's requested current folder. Branch `feat/existing-ocr-vita`; another session's OCR work. Existing untracked `hooker/RESEARCH.md` preserved. Notes are mirrored here without changing its branch. |
| `ps-vita-ocr-notes/` | Isolated notes worktree, branch `docs/vita-hook-investigation-notes`, based on OCR origin/main `7964b43f022e51bd26da84f9ff1fc0a26c9dd1aa`. Only `notes/` belongs to this change. |
| `vita-jp-overlay-hooks/` | Firmware fix branch `fix/native-hooks-fw365`. |
| `vita-jp-overlay-low-memory/` | Picker-memory branch `fix/native-text-low-memory`. |
| `vita-jp-overlay-clannad-profile/` | Root's separate profile/docs worktree, based on merge `5e5bc25`; prepares the final `0x2CA70`/r6 profile. PR #16 merged and profile installed with complete readback; physical user-reported CALL success is logged; REGISTER-profile correspondence remains unproved. |
| `vita-jp-overlay-input-ownership/` | Root's branch `fix/overlay-input-ownership`; overlay PR #17 head `464248a`, controller blocking hardening and ownership regressions. Prepared, uninstalled; user stopped deployment before writes. |
| `vita-jp-overlay-discovery/` | Current branch `fix/region-sized-ocr-capture`, based on PR #13 merge `539c2888`; capture-buffer changes are PR #14 head `5118c4a`, merged as `c888b59`; now installed with PR #15 shell and PR #16 profile. Earlier capture branch `fix/native-libc-capture` remains separate. Earlier diagnostic experiment preserved separately in local-only `fix/native-text-discovery` commit `2d56ce4`, not deployed or submitted as a PR. |
| `ps-vita-ocr-lab/vita-jp-overlay/` | Shared overlay worktree; last observed on `fix/local-ocr-memory`, owned by the OCR session. Preserve it. |
| `ps-vita-ocr-lab/vita-agent-use/` | Device control client and private ignored `agent/` evidence. |
| `ps-vita-ocr-lab/tools/vitasdk/` | Local VitaSDK; arm-vita-eabi-gcc 15.2.0 in the recorded boot-fix build. |
| `ps-vita-ocr-lab/tools/paf/psp2cxml-tool/mbuild/` | Paf resource compiler tooling used by local builds. |

## Merged overlay work

| Change | Key commit(s) | PR and merge |
|---|---|---|
| Native hook 3.65 boot ABI, guards, stack correction | `8486c97`; evidence `8e7a6e8`; SDK pin `c7eed5066fe2550ddb17748929db620951209418` | [#10](https://github.com/bee-san/vita-jp-overlay/pull/10), merged `bfb1ada4f3f6995100cb7fa04d4b8240c868ddcf` |
| Picker/raw text without eager result arenas, optional bounded heap fallback | `e2ed3b7`; measured evidence `ca46033`; exact PR head `5b1310328a102aba915a58db9413a0b3ac475c9f` | [#11](https://github.com/bee-san/vita-jp-overlay/pull/11), merged `13bcf816ab4e45fb8f59da0f52407a621b07284e` |

GitHub PR states above were independently queried on 2026-10-10. The active investigator reports overlay local main subsequently synchronized through PR #14 to latest merge `264241e`. Do not switch or fast-forward another session's feature worktree to achieve that again.

The earlier installed shell came from the low-memory implementation; it is now replaced by the OCR-first shell. Later docs/merge commits alone do not imply a different installed binary. Artifact hashes are the deployment evidence in [[02-Current-device-state]]. Source, built artifact, remote branch, merged revision and installed bytes are separate states.

## Notes publication

This notebook is published separately as [OCR PR #17](https://github.com/bee-san/vita-vn-ocr/pull/17), branch `docs/vita-hook-investigation-notes`. Its initial head `b2295008c646405a02da7325c86cef1b86a7121e` passed all six hosted gates. Follow-up notes capture ongoing investigation; confirm the latest PR head/checks before merge. The user-facing current folder contains a byte-verified mirror of the owned note files, not a branch switch or staged OCR-code change.

## Capture correction merged and installed

[Overlay PR #12](https://github.com/bee-san/vita-jp-overlay/pull/12), “Capture CLANNAD retail libc text and surface readable hooks,” merged at 10:59:00 UTC as `a5e0c50b1096b5ae7de450ae712e63b272dc7f17`. Final reviewed head `69006af9e36347b75534c01900b48d43ed372981` passed hosted host (46 s) and Vita (4 m 23 s) CI. The unattached local main reference was fast-forwarded to that merge; concurrent worktree branches were preserved.

Implementation `6eef6c031aa1cc4bc162ebf31b8628d3098f0f5e` combines initial import correction `0a660159...` with the production comparator repair. It passes 10 host suites, 264 synthetic Vita3K checks, the release build/ABI audit and a corrected full-feed two-page CLANNAD ranking experiment. Intermediate evidence-only commit `d1ee0dc` records the first two-page capture; final evidence head `69006af` preserves the stronger snapshot proof. Artifact hashes do not change across evidence-only commits.

The new kernel/text pair is installed and read back, with normal boot verified; the user's physical test found changing 0% hooks and irrelevant text. Correct dialogue selection/capture is still unverified. The follow-up `fix/ocr-first-hook-picker` is active from that merged baseline. See [[02-Current-device-state]], [[14-Retail-libc-hook-coverage]] and [[16-Full-feed-picker-ranking]].

## OCR-first follow-up merged and installed

[Overlay PR #13](https://github.com/bee-san/vita-jp-overlay/pull/13), “Read screenshots first and keep native text matches stable,” is on `fix/ocr-first-hook-picker` from merged `a5e0c50`. Code `4205be7034a2b48df45e65c7f253e7c7524b349a` introduced the 384 KiB anchor, stable screenshot matches and explicit user selection. Current head `5826329ddcb1244b0e318f2c6c527181cc14d4d8` also synchronizes the generated default config and clears cancelled subtitle busy state. See [[17-OCR-first-physical-workflow]].

Final local validation passed all 10 host suites in 4.18 s, including 37 worker cases, and the Vita release/packaged ABI check. Two reviews found no remaining blockers. The older hosted `a048faa` failed only generated-default/config-example equality after a comment change; this is fixed in `5826329`. Both exact-head hosted jobs passed in run `38048108519`: host 52 s, Vita 4m25s. PR #13 merged at 11:26:11 UTC as `539c2888f61b28277bb7fd7bdd2b6a4bb5b330fd`; local main was synchronized. Shell/config installation, complete readbacks and normal reboot are verified. The installed shell is 158,564 B, SHA `32baceebd6b9af749b7b52a59336fe9225bef382f20eec10a60c41b930815d3c`. Live production-Lens host checks subsequently matched one short verified page from full and cropped inputs; the physical test subsequently failed with source error -108 and no game text. Paired logs now show the full-frame capture allocation fails before Lens; the region-sized allocation repair has since merged as PR #14, with bounded diagnostics following in PR #15. The user subsequently chose fixed manual hooks; the current config is hooks, with Lens and its saved region preserved. See [[19-Physical-OCR-source-failure]].

## Capture repair merged and installed

[Overlay PR #14](https://github.com/bee-san/vita-jp-overlay/pull/14), head `5118c4a09248a3d8b7e8018d85fa0d0247b82ca4`, uses selected-region capture capacity and explicit single-pass JPEG ownership. Final local checks: 10 host suites in 4.61 s, 41 worker/16 production-kernel cases, kernel ASAN/UBSAN, ncnn-OFF SDK Release and packaged ABI; 6,055 kernel ARM instructions, no FPU/NEON. Both independent reviews completed. Both hosted jobs passed (host 56 s, Vita 3m41s), and PR #14 merged at 11:46:20 UTC as `c888b5936ca46729b0fbc7698b2348c5a4b3d387`. Local main/origin are synchronized. The kernel and subsequent PR #15 shell are now installed/read back and normally booted; the device remains in user-selected manual hooks mode. See [[19-Physical-OCR-source-failure]].

## Bounded picker diagnostics merged and installed

[Overlay PR #15](https://github.com/bee-san/vita-jp-overlay/pull/15), exact head `f58b1aa49dd43fcc3d6d78b0c9bcd48a1c0ca6d7`, merged at 11:55:40 UTC as `5e5bc2557ee183e60c4afd9335b55666e8697cea`. It adds bounded displayed-row/selection excerpts only on explicit actions. Local 10-suite host validation took 5.31 s; clean Vita Release/ABI passed. Exact hosted run `38049801084` passed host 1m06s and Vita 3m07s. Combined PR #14+#15 artifacts are installed with complete readbacks and normal boot: kernel `ae3c0341...`, shell `4f1f0650...`, unchanged text `84259631...`. See [[21-Picker-diagnostics-and-screenshot-provenance]].

## CLANNAD profile merged and installed

[Overlay PR #16](https://github.com/bee-san/vita-jp-overlay/pull/16), exact head `30a68d216243f74e772410a31b90ff17be6dddcb`, merged at 12:09:53 UTC as `264241eb0f6323f373049a458be214448f8320cb`; local main/origin both match. Hosted run `38050663785` passed host 44 s and Vita 4m26s. The 63 B profile SHA `99c73600...` is installed with byte-equal readback. Normal boot and foreground launch pass; the user now reports success and logs show CALL-source follow/lookup. Physical REGISTER correspondence and exact visible-page count remain unproved. See [[22-CLANNAD-register-profile-proof]].

## Input ownership follow-up published, uninstalled

[Overlay PR #17](https://github.com/bee-san/vita-jp-overlay/pull/17), head `464248a8606e38b45c0cd35677acb3d37cd6879f`, combines implementation `cda7de4` and production controller/worker regressions. It addresses the user's × selection reaching CLANNAD without assuming the exact hardware cause. Local 11 suites/4.26 s, 7 controller/44 worker cases and Vita release/ABI pass; 6,492 kernel ARM instructions, no FPU/NEON. Both exact-head hosted jobs passed: host 1m03s, Vita 2m46s. Prepared hashes are kernel `fee142fa...`, shell `faea5914...`; neither is installed. The user stopped deployment before upload/app close/reboot; PR stays open/unmerged and must not be merged/deployed merely because CI completes. See [[24-Overlay-selection-input-leak]].

## Adjacent work to preserve

- [OCR PR #16](https://github.com/bee-san/vita-vn-ocr/pull/16): existing Meiki OCR port, branch `feat/existing-ocr-vita`.
- [OCR PR #15](https://github.com/bee-san/vita-vn-ocr/pull/15): initial native-text feasibility research, branch `docs/vita-text-hook-research`.
- Overlay PR #9: bounded Meiki OCR module; separate ownership and validation.

Do not fold these changes into the notes or native-hook PRs. Initial `hooker/RESEARCH.md` describes an earlier source-only investigation and should not be silently rewritten into a claim of current device success.

## Remote discipline

OCR remote: `bee-san/vita-vn-ocr`. Overlay remote: `bee-san/vita-jp-overlay`. The overlay fork is independent; upstream `Dartv/vita-jp-overlay` is pull-only. Always use explicit `gh -R bee-san/vita-jp-overlay` or `gh -R bee-san/vita-vn-ocr` as appropriate so a fork-default cannot publish to the wrong project.

Before merge, inspect exact-head required checks and current review state. Documentation-only validation is link/integrity/scope checking; device success must not be inferred from a documentation PR merge.
