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
| `vita-jp-overlay-discovery/` | Current branch `fix/native-libc-capture`, based on merged `13bcf816...`; draft PR #12. Earlier diagnostic experiment preserved separately in local-only `fix/native-text-discovery` commit `2d56ce4`, not deployed or submitted as a PR. |
| `ps-vita-ocr-lab/vita-jp-overlay/` | Shared overlay worktree; last observed on `fix/local-ocr-memory`, owned by the OCR session. Preserve it. |
| `ps-vita-ocr-lab/vita-agent-use/` | Device control client and private ignored `agent/` evidence. |
| `ps-vita-ocr-lab/tools/vitasdk/` | Local VitaSDK; arm-vita-eabi-gcc 15.2.0 in the recorded boot-fix build. |
| `ps-vita-ocr-lab/tools/paf/psp2cxml-tool/mbuild/` | Paf resource compiler tooling used by local builds. |

## Merged overlay work

| Change | Key commit(s) | PR and merge |
|---|---|---|
| Native hook 3.65 boot ABI, guards, stack correction | `8486c97`; evidence `8e7a6e8`; SDK pin `c7eed5066fe2550ddb17748929db620951209418` | [#10](https://github.com/bee-san/vita-jp-overlay/pull/10), merged `bfb1ada4f3f6995100cb7fa04d4b8240c868ddcf` |
| Picker/raw text without eager result arenas, optional bounded heap fallback | `e2ed3b7`; measured evidence `ca46033`; exact PR head `5b1310328a102aba915a58db9413a0b3ac475c9f` | [#11](https://github.com/bee-san/vita-jp-overlay/pull/11), merged `13bcf816ab4e45fb8f59da0f52407a621b07284e` |

GitHub PR states above were independently queried on 2026-10-10. The active investigator reports overlay local main synchronized to merge `13bcf816...`. Do not switch or fast-forward another session's feature worktree to achieve that again.

The installed shell was built from the low-memory implementation; later docs/merge commits do not imply a different installed binary. Artifact hashes are the deployment evidence in [[02-Current-device-state]]. Source, built artifact, remote branch, merged revision and installed bytes are separate states.

## Capture correction in review

[Overlay draft PR #12](https://github.com/bee-san/vita-jp-overlay/pull/12), head `0a66015994dbb3f12a3a506f0c0e4cab8ec72fdd`, adds the missing SceLibc imports. It has 10 passing host suites, 255 passing synthetic Vita3K checks, and a passing Vita release build/ABI audit. It is deliberately draft pending real-game evidence; it is not installed or a CLANNAD compatibility result. See [[14-Retail-libc-hook-coverage]].

## Adjacent work to preserve

- [OCR PR #16](https://github.com/bee-san/vita-vn-ocr/pull/16): existing Meiki OCR port, branch `feat/existing-ocr-vita`.
- [OCR PR #15](https://github.com/bee-san/vita-vn-ocr/pull/15): initial native-text feasibility research, branch `docs/vita-text-hook-research`.
- Overlay PR #9: bounded Meiki OCR module; separate ownership and validation.

Do not fold these changes into the notes or native-hook PRs. Initial `hooker/RESEARCH.md` describes an earlier source-only investigation and should not be silently rewritten into a claim of current device success.

## Remote discipline

OCR remote: `bee-san/vita-vn-ocr`. Overlay remote: `bee-san/vita-jp-overlay`. The overlay fork is independent; upstream `Dartv/vita-jp-overlay` is pull-only. Always use explicit `gh -R bee-san/vita-jp-overlay` or `gh -R bee-san/vita-vn-ocr` as appropriate so a fork-default cannot publish to the wrong project.

Before merge, inspect exact-head required checks and current review state. Documentation-only validation is link/integrity/scope checking; device success must not be inferred from a documentation PR merge.
