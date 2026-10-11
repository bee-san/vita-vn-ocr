---
title: Overlay selection input leak
created: 2026-10-10
updated: 2026-10-10
status: deployment-stopped-by-user-prepared-fix-uninstalled
tags: [vita, input, controller, overlay, regression]
---

# Overlay selection input leak

After confirming native text hooks worked, the user reported that pressing **× to select a hook also advances CLANNAD**. The game advances on that press, not while idle. This is a separate overlay input-ownership defect; the earlier selected CALL text updates and dictionary-job evidence remain valid.

## User stopped deployment

The user then said **“dont put it on the vita, i wanna try something else.”** Root stopped the input-fix deployment before any new upload, app close or reboot. Only read-only backup downloads had started: kernel backup completed and shell backup read was in flight at that report. No new controller build reached the device.

The overlay PR remains open and unmerged. CI completion is not authorization to merge or deploy this fix. Preserve the prepared artifacts and evidence while awaiting the user's next direction; do not resume the proposed hardware test automatically. The installed working-text set remains kernel `ae3c0341...`, shell `4f1f0650...`, profile `99c73600...`, Text `84259631...` and manual config `3833843d...`.

## What inspection establishes

The picker selection path already keeps the input block enabled; selection does not call close/unblock. CLANNAD's Positive2 controller import is already covered, and recorded input-hook installation results are all 0. Therefore neither an intentional selection-time unblock nor a simply missing Positive2 hook is established as the cause. The actual physical leakage path remains unconfirmed.

The real-worker test flow checks open → select → dictionary completion → refresh retains blocking; close releases it once. Stale session and changed-row rejection retain ownership. A denied-block opening originally failed five assertions; the checked-open correction now passes, including visible failure and retry cleanup. A separate regression caught the error strip surviving game exit/replacement (two failures); the `clear_input_block_error` correction passes. These are discriminating host failures and repairs, not physical input proof.

## Published hardening and bounded diagnostics

[Overlay PR #17](https://github.com/bee-san/vita-jp-overlay/pull/17), “Harden overlay input blocking and trace selection button leakage,” is on `fix/overlay-input-ownership` in `/home/bee/Documents/src/vita-jp-overlay-input-ownership`. Implementation `cda7de4` and validation follow-up `464248a8606e38b45c0cd35677acb3d37cd6879f` are published, with hardware validation pending. This is the **overlay repository's** PR #17; the separate **notes repository's** [PR #17](https://github.com/bee-san/vita-vn-ocr/pull/17) only maintains documentation.

- While blocked, write neutral buttons/sticks directly without depending on reading the old controller sample. Preserve timestamps/reserved fields, negative-button semantics and raw/SceShell input.
- Check every write, propagate native copy errors and reject an oversized blocked batch instead of returning an unfiltered tail as successful input.
- Check the block request before opening an interactive overlay. If refused, keep that UI closed and show an error strip; clear it on retry or game change.
- Log block transitions and at most five controller-condition records per opening: first blocked read, ×, copy failure, foreign caller and empty native result. Records contain hook/PID/read/write metadata, no game text, and allocate no heap.

There is no speculative syscall-context/ENTER change or close-latch rewrite. These changes strengthen known failure paths and make the remaining hardware path observable; they do not prove which path caused the user's × press to leak.

## Validation and candidate files

Linux local validation passes **11 host suites in 4.26 s**, including seven tests of the actual production controller wrapper/filter and 44 actual worker cases. Four initial controller regressions fail against the original filter. Cases cover Positive2 × suppression, raw/Shell preservation, negative 64-sample batches/canaries, read/write and partial-batch failures, original native errors and diagnostic bounds over 256 calls. The 44-case worker suite independently passed its focused run in 0.03 s; these durations are host test runtime, not console performance.

VitaSDK GCC 15.2.0 Release with ncnn OFF and Text test app ON passes packaged syscall ABI/import checks. The kernel audit counts 6,492 ARM instructions and zero FPU/NEON. Hosted run `38052215727` at exact head `464248a` passed host in 1m03s and Vita in 2m46s; release was skipped. The PR remains open/unmerged under the user's stop instruction.

The notes author independently hashed the private prepared files under `vita-agent-use/agent/overlay-input-ownership-20261010/`:

| Prepared artifact | Bytes | SHA-256 |
|---|---:|---|
| Kernel | 40,056 | `fee142fa6a90c87bfe8d481c0a5df11850da443b1548b875b1aec3132085174a` |
| Shell | 159,752 | `faea59148d1f23119ea9b6f00f07941addbc67d59984f261c4a0c7b653965144` |

`deployment-plan.json` says `prepared_not_deployed`. These binaries are **not installed**. Current installed kernel/shell remain `ae3c0341...` / `4f1f0650...`; manual config, title profile and Text plugin are preserved. Root initially told the user to save before a planned install, but the later stop instruction above supersedes that plan.

## Physical check only if the user later resumes this fix

If the user later resumes this work and the concrete build is installed/read back and booted, select a hook with × and establish whether CLANNAD stays on the same page. Then close the overlay and confirm ordinary game controls work. If × still leaks, use the bounded records to identify the real blocked flag, caller and copy result; do not declare the leak fixed from host ownership tests or hook installation alone.

Related: [[02-Current-device-state]], [[23-Physical-native-hook-success]], [[08-Validation-and-measurements]], [[10-Repositories-and-revisions]], [[12-Open-questions-and-next-steps]].
