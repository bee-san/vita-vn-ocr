---
title: 2026-10-10 native text investigation log
created: 2026-10-10
updated: 2026-10-10
status: ongoing
tags: [vita, experiment, chronology]
---

# 2026-10-10 native text investigation log

Ordered reconstruction from recorded measurements, local artifacts and the active conversation. Exact timestamps are omitted where not established. This log distinguishes an earlier result from later changes; it is not a replacement for [[02-Current-device-state]].

## 1. Feasibility and initial implementation

Initial source research concluded native RAM reading and taiHEN hooks are feasible, while universal engine-independent dialogue extraction was unproven. The research file intentionally had no native-game measurements. A later experimental implementation added portable decoding/ranking, bounded kernel scanning, a game-side text plugin and shell source selection. See [[03-Native-text-architecture]].

Isolated Vita3K synthetic validation eventually passed 249 checks: 43 portable, 192 observer/register preservation and 14 production plugin. This did not exercise the physical kernel loader or commercial CLANNAD text.

## 2. Repeated physical boot shutdowns

The initial kernel contained FPU/NEON instructions. An earlier correction removed them, but the physical Vita still shut down. A diagnostic shell pausing before the first custom call allowed LiveArea to remain running. Runtime inspection found kernel state READY (2) and the unresolved import trap.

Comparison with actual firmware 3.65 identified a strong 3.60 `SceSysmemForKernel` import. Replacing it with the driver memory-info API and guarding every custom call corrected startup. Static stack inspection also found and corrected a 4,840 B nested chain. See [[04-Firmware-365-boot-ABI]] and [[05-Kernel-syscall-stack]].

## 3. Physical boot repaired

The corrected kernel/shell/text artifacts were uploaded and fully read back. Normal reboot succeeded without holding L. API 8, registration 0 and STARTED module states were observed. CLANNAD's text plugin could start. The firmware benchmark preserved exact hashes and limitations.

CI initially failed because the August SDK lacked the new header/stub. Official Vita headers were pinned and Sysmem stubs rebuilt; the focused fix was merged as overlay PR #10 at `bfb1ada4f3f6995100cb7fa04d4b8240c868ddcf`.

## 4. CLANNAD overlay memory failure

The user pressed L+R and saw an overlay-memory error. Status showed the 786,432 B result allocation failing with `0x80024302`. Screenshot encoding failed separately at `jpeg_encode`, so no captured frame was available for visual inspection.

A performance attempt returned `Invalid PC server reply`; pushed samples had memory figures but uncertain process ownership. These were not accepted as CLANNAD memory measurements. A native decrypt attempt failed during authentication with `0x800F0616`; its output was not treated as decrypted.

## 5. Low-memory picker repair

The shell was changed to keep picker/raw text/subtitles in resident storage and allocate only for lookup/OCR. A bounded Paf fallback can provide two 128 KiB arenas for native dictionary lookup. Ten host suites and 30 worker cases passed; Vita build/ABI audit passed.

The new shell, SHA-256 `45f86b1ea83c08bdf61c1dfd174cc5356151e746a3659a0299218436a8f10c8c`, matched full readback. The device rebooted successfully and CLANNAD opened. The user confirmed the picker appeared. This was merged as overlay PR #11 at `13bcf816ab4e45fb8f59da0f52407a621b07284e`.

## 6. Picker still finds nothing

The user advanced dialogue but the picker kept searching. Metadata showed zero candidates through repeated scanner passes. Opening the UI was a real improvement, but capture remained unverified. Module STARTED and changing scan addresses were insufficient evidence of hook attachment or successful game-memory reads.

During subsequent investigation a game launch was accepted but sustained execution was not established. The user reported the console turned off, then reported it on. The active investigator restored the original text plugin (`c146238b...`) with readback and closed the game; native running-app inventory was empty. Experimental kernel/scanner diagnostic code was not installed. This later power-off remains a separate unresolved observation.

## 7. Offline executable and import breakthrough

An isolated Vita3K instance produced actual decrypted, relocated loaded segments before game execution. The research ELF hash is `23611e880fb573f85246756073edeb20cdfa02c3f7062cd0427cb3b0f5d499d9`, distinct from the original encrypted SELF `76b2b2af...`.

Its main module has 250 function imports. None of the four generic hook NIDs appears. It instead imports SceLibc memcpy/strcpy/strncpy/memmove, and no PVF/font API appears in that main-module inventory. Offline font-table inspection supports CP932 as an encoding to investigate. See exact rows and limitations in [[07-CLANNAD-investigation]].

This supersedes the premature inference that empty capture already implied a required CLANNAD-specific engine hook. Correct generic import coverage is the next discriminating test.

## 8. Notebook created alongside active debugging

The user explicitly requested substantial Obsidian-style notes in the current folder and delegated their creation to a subagent while the primary investigator continued capture work. An isolated docs branch/worktree preserves concurrent OCR/native-code edits. Notes capture evidence boundaries, deployment hashes, failed attempts, decisions, commands and remaining questions. No private game data or identity was copied.

The next experiment should append here or use a dated child note, then update [[01-Evidence-and-claims]], [[02-Current-device-state]] and [[12-Open-questions-and-next-steps]].

## 9. Corrected retail libc hooks validated, real capture pending

Commit `0a66015994dbb3f12a3a506f0c0e4cab8ec72fdd` adds SceLibc coverage alongside the original clib family. `strcpy` uses a kernel-bounded NUL read with automatic UTF-8/CP932 decoding, avoiding a user-side unbounded `strlen`. Ten host suites passed (2.14 s), including 10 production-kernel cases. The actual ARM production-plugin harness passed 255 checks: 43 portable, 192 observer and 20 plugin; both families, return values, overlapping memmove, zero-length suppression and unload were exercised.

Version `0.6-hooks.5` built and passed packaged ABI checks. It was pushed as [draft overlay PR #12](https://github.com/bee-san/vita-jp-overlay/pull/12), deliberately pending real commercial-game evidence. Physical plugins remain unchanged after rollback. Earlier scanner diagnostics are preserved only in local commit `2d56ce4` on `fix/native-text-discovery`; they are not part of this deployment. See [[14-Retail-libc-hook-coverage]].
