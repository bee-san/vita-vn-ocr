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
| `vita-jp-overlay-discovery/` | Current branch `fix/ocr-first-hook-picker`, based on merged PR #12 `a5e0c50`; latest PR #13 merged `539c2888`. Earlier capture branch `fix/native-libc-capture` remains separate. Earlier diagnostic experiment preserved separately in local-only `fix/native-text-discovery` commit `2d56ce4`, not deployed or submitted as a PR. |
| `ps-vita-ocr-lab/vita-jp-overlay/` | Shared overlay worktree; last observed on `fix/local-ocr-memory`, owned by the OCR session. Preserve it. |
| `ps-vita-ocr-lab/vita-agent-use/` | Device control client and private ignored `agent/` evidence. |
| `ps-vita-ocr-lab/tools/vitasdk/` | Local VitaSDK; arm-vita-eabi-gcc 15.2.0 in the recorded boot-fix build. |
| `ps-vita-ocr-lab/tools/paf/psp2cxml-tool/mbuild/` | Paf resource compiler tooling used by local builds. |

## Merged overlay work

| Change | Key commit(s) | PR and merge |
|---|---|---|
| Native hook 3.65 boot ABI, guards, stack correction | `8486c97`; evidence `8e7a6e8`; SDK pin `c7eed5066fe2550ddb17748929db620951209418` | [#10](https://github.com/bee-san/vita-jp-overlay/pull/10), merged `bfb1ada4f3f6995100cb7fa04d4b8240c868ddcf` |
| Picker/raw text without eager result arenas, optional bounded heap fallback | `e2ed3b7`; measured evidence `ca46033`; exact PR head `5b1310328a102aba915a58db9413a0b3ac475c9f` | [#11](https://github.com/bee-san/vita-jp-overlay/pull/11), merged `13bcf816ab4e45fb8f59da0f52407a621b07284e` |

GitHub PR states above were independently queried on 2026-10-10. The active investigator reports overlay local main subsequently synchronized through PR #12 to latest merge `539c2888`. Do not switch or fast-forward another session's feature worktree to achieve that again.

The earlier installed shell came from the low-memory implementation; it is now replaced by the OCR-first shell. Later docs/merge commits alone do not imply a different installed binary. Artifact hashes are the deployment evidence in [[02-Current-device-state]]. Source, built artifact, remote branch, merged revision and installed bytes are separate states.

## Notes publication

This notebook is published separately as [OCR PR #17](https://github.com/bee-san/vita-vn-ocr/pull/17), branch `docs/vita-hook-investigation-notes`. Its initial head `b2295008c646405a02da7325c86cef1b86a7121e` passed all six hosted gates. Follow-up notes capture ongoing investigation; confirm the latest PR head/checks before merge. The user-facing current folder contains a byte-verified mirror of the owned note files, not a branch switch or staged OCR-code change.

## Capture correction merged and installed

[Overlay PR #12](https://github.com/bee-san/vita-jp-overlay/pull/12), “Capture CLANNAD retail libc text and surface readable hooks,” merged at 10:59:00 UTC as `a5e0c50b1096b5ae7de450ae712e63b272dc7f17`. Final reviewed head `69006af9e36347b75534c01900b48d43ed372981` passed hosted host (46 s) and Vita (4 m 23 s) CI. The unattached local main reference was fast-forwarded to that merge; concurrent worktree branches were preserved.

Implementation `6eef6c031aa1cc4bc162ebf31b8628d3098f0f5e` combines initial import correction `0a660159...` with the production comparator repair. It passes 10 host suites, 264 synthetic Vita3K checks, the release build/ABI audit and a corrected full-feed two-page CLANNAD ranking experiment. Intermediate evidence-only commit `d1ee0dc` records the first two-page capture; final evidence head `69006af` preserves the stronger snapshot proof. Artifact hashes do not change across evidence-only commits.

The new kernel/text pair is installed and read back, with normal boot verified; the user's physical test found changing 0% hooks and irrelevant text. Correct dialogue selection/capture is still unverified. The follow-up `fix/ocr-first-hook-picker` is active from that merged baseline. See [[02-Current-device-state]], [[14-Retail-libc-hook-coverage]] and [[16-Full-feed-picker-ranking]].

## OCR-first follow-up merged and installed

[Overlay PR #13](https://github.com/bee-san/vita-jp-overlay/pull/13), “Read screenshots first and keep native text matches stable,” is on `fix/ocr-first-hook-picker` from merged `a5e0c50`. Code `4205be7034a2b48df45e65c7f253e7c7524b349a` introduced the 384 KiB anchor, stable screenshot matches and explicit user selection. Current head `5826329ddcb1244b0e318f2c6c527181cc14d4d8` also synchronizes the generated default config and clears cancelled subtitle busy state. See [[17-OCR-first-physical-workflow]].

Final local validation passed all 10 host suites in 4.18 s, including 37 worker cases, and the Vita release/packaged ABI check. Two reviews found no remaining blockers. The older hosted `a048faa` failed only generated-default/config-example equality after a comment change; this is fixed in `5826329`. Both exact-head hosted jobs passed in run `38048108519`: host 52 s, Vita 4m25s. PR #13 merged at 11:26:11 UTC as `539c2888f61b28277bb7fd7bdd2b6a4bb5b330fd`; local main was synchronized. Shell/config installation, complete readbacks and normal reboot are verified. The installed shell is 158,564 B, SHA `32baceebd6b9af749b7b52a59336fe9225bef382f20eec10a60c41b930815d3c`. Live production-Lens host checks subsequently matched one short verified page from full and cropped inputs; the physical test subsequently failed with source error -108 and no game text. Paired logs now show the full-frame capture allocation fails before Lens; a region-sized allocation repair is being developed. the config uses auto with Lens and its saved region preserved. See [[19-Physical-OCR-source-failure]].

## Adjacent work to preserve

- [OCR PR #16](https://github.com/bee-san/vita-vn-ocr/pull/16): existing Meiki OCR port, branch `feat/existing-ocr-vita`.
- [OCR PR #15](https://github.com/bee-san/vita-vn-ocr/pull/15): initial native-text feasibility research, branch `docs/vita-text-hook-research`.
- Overlay PR #9: bounded Meiki OCR module; separate ownership and validation.

Do not fold these changes into the notes or native-hook PRs. Initial `hooker/RESEARCH.md` describes an earlier source-only investigation and should not be silently rewritten into a claim of current device success.

## Remote discipline

OCR remote: `bee-san/vita-vn-ocr`. Overlay remote: `bee-san/vita-jp-overlay`. The overlay fork is independent; upstream `Dartv/vita-jp-overlay` is pull-only. Always use explicit `gh -R bee-san/vita-jp-overlay` or `gh -R bee-san/vita-vn-ocr` as appropriate so a fork-default cannot publish to the wrong project.

Before merge, inspect exact-head required checks and current review state. Documentation-only validation is link/integrity/scope checking; device success must not be inferred from a documentation PR merge.
