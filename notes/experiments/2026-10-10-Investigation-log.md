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

## 10. Real startup call, debugger deadlock and rendering correction

The runtime investigator captured real CLANNAD execution at SceLibc memmove stub `0x810A9458`, caller LR `0x81033D6F`. The call occurred about 0.283 s into the recorded trace with NULL source and length zero. It proves that the game uses the imported path, but supplies no dialogue.

Continuing after the breakpoint hung. Source inspection at Vita3K revision `b75d5e6292d5caa6a8411bacc9db1eca1531beab` showed `cmd_continue` holding `thread->mutex` while calling `ThreadState::resume`, which acquires the same nonrecursive mutex. The experiment therefore exposed a debugger tooling defect, not proof that ordinary game execution or the new capture code hangs.

Separately, uninstrumented game startup rendered Japanese notices in isolated Xvfb with `SDL_VIDEODRIVER=x11`; an inherited Wayland setting explained earlier black captures. No added firmware install was required for that observed startup. Advancing first-run save UI toward dialogue is ongoing. A source view, initial notices and a zero-byte copy are all short of the two-page capture criterion.

Independent read-only review of PR #12 head `0a660159...` found no blockers, with 2,784 B compiled stack subtotal and zero FPU/NEON in 5,922 decoded instructions. It preserved the distinction between the emulator's test transport and the production-kernel bound tests. The user wants notice when physical testing is required; none has been requested for this follow-up yet.

## 11. Input routing and prepared deployment are still pending work

The first-run save-create confirmation could not accept SDL frontend keyboard input. The runtime investigator identified a frontend route that forwards keys to IME without updating the controller keyboard state and is trying the isolated Qt frontend next. This is a separate emulator-input uncertainty; gameplay dialogue still has not been reached.

The primary investigator prepared a private deployment plan under `vita-agent-use/agent/overlay-libc-20261010/`, pinning old/new kernel and text hashes with backup/readback steps. The plan is **not executed**; a fresh device-state check is required. No physical device commands occurred during this resumed portion of work. A prepared plan does not change [[02-Current-device-state]].

## 12. Qt input succeeds; real-game transport prepared

Qt's C key dismissed the first-run save-create dialog and reached the PROTOTYPE logo. A private bridge plugin was built to export the overlay transport API and run the unchanged production text plugin in actual CLANNAD under isolated Vita3K. Initial bridge hash `a2941cc9...` and production text hash `84259631...` are recorded in [[15-Vita3K-runtime-investigation]]. The bridge uses bounded user-space reads and the production decoder/ranker, but does not model real kernel mapping ownership or syscall behavior. It must never be installed on the physical console.

Root confirmed all six hosted notes-PR gates passed at initial notes head `b229500`. PR #12's host/Vita gates also passed at `0a660159...`. These CI outcomes do not resolve the remaining actual-game text or physical-device validation.

## 13. First visible narration matched through generic hooks

Root independently viewed private `dialogue-3.png` and matched its first narration to five records in `captures-first-run.log` (81,920 B, 506 TEXT records). The production text plugin SHA remains `84259631...`, using the test bridge and no `.vjhook` profile. Four inspected matching caller sites branch to SceLibc memcpy stub `0x810A9388`; five stream addresses are retained as small derived metadata in [[15-Vita3K-runtime-investigation]].

This establishes useful real-game Japanese bytes through generic imports. It does not yet distinguish current-page events from script prefetch or prove that one selected stream follows subsequent visible pages. False positives and bridge mapping rejections remain visible limitations. Root is collecting the next-page evidence before treating the offline test as complete; physical plugins are unchanged.

## 14. Two visible pages match; full-feed validation catches ranking defect

The immediate stream `0x8102CA7F` matched two distinct rendered narration pages without a game profile. Root independently inspected the later `live-page-4b.png` and capture log. Evidence-only commit `d1ee0dc` records pages 1/4 with first-helper update counts 1/2, input lengths 17/13 B and decoded lengths 24/18 B. The implementation binaries are unchanged; new-head CI was started as run `38046046267`.

The helper had filtered short/non-Japanese events before ranking. A stronger full-feed snapshot at sequence 425 retained the actual six-character initial line in three streams, but none made the displayed top five. Frequent noise and static data buried dialogue because the production comparator prioritized update count before Japanese fraction/length. A later full-feed snapshot at sequence 1684 put immediate source `0x8102CA7F` at display rank 5 with update count 3; the earlier helper had filtered a one-kanji page, explaining the additional update.

This is a production ranking bug caught by stronger validation, not a bridge failure. A focused `core/text_source.c` correction and synthetic regression coverage are being developed in the same PR. The physical deployment gate remains closed until initial-picker behavior is proven with the full feed. See [[16-Full-feed-picker-ranking]].

## 15. Ranking correction passes discriminating regression and ARM checks

Commit `6eef6c031aa1cc4bc162ebf31b8628d3098f0f5e` moves Japanese fraction and normalized length ahead of update count, without adding allocation or changing the 26,480 B source table. Its synthetic fixture fails four checks with the old comparator. Ten host suites passed (5.44 s), Vita release/ABI passed, and the new kernel has 5,921 ARM instructions with zero FPU/NEON. Vita3K passed 264 checks (52 portable, 192 observer, 20 plugin).

New kernel SHA-256 `6cc2dc868dc4403bcab252d689acd2f0779a55f9bd8069654b335eaffa571933`; shell/text bytes unchanged. The corrected full-feed bridge is `106be9d495ce17b622f67913d0f5cd8c09ce7dd767c5e647c59b85095364c316` and a fresh real-game run is in progress. The failing baseline is preserved. No physical install or corrected full-feed success is claimed yet.

## 16. Corrected full-feed capture and displayed ranks verified

The fresh run with comparator `6eef6c0`, unchanged production Text SHA `84259631...`, and bridge SHA `106be9d4...` matched two rendered pages without a title profile. First-page sequence 424 displays the immediate matching streams at ranks 2/3/4, update count 1. Later-page sequence 1680 displays those same streams at ranks 3/4/5, update count 3 and different current text. Both snapshots are complete; `selected=0`, so manual selection remains outside this proof. The failing baseline is preserved separately.

Root refreshed read-only physical backups: installed kernel remains `447f2e64...`, text remains `c146238b...`, device connected, app inventory empty. The private deployment plan now pins candidate kernel `6cc2dc86...` plus production text `84259631...`. No upload/reboot/install has occurred for this candidate. Offline proof is now concrete enough to proceed toward the requested physical test after exact-head checks/review.

## 17. Final offline evidence published; physical candidate still uninstalled

Final evidence-only overlay head `69006af9e36347b75534c01900b48d43ed372981` publishes the complete unfiltered checkpoints and hashes. Root independently compared both rendered pages with the corresponding source text and checked all six evidence hashes plus complete snapshot headers/footers. Implementation artifacts are unchanged from `6eef6c0`. PR #12 is now ready for review, no longer draft; exact final-head CI is starting. Earlier passing CI belongs to earlier heads and is recorded separately.

Fresh physical read-only backups still show the old kernel/text; the new candidate has not been installed. The next dependent action waits for final-head checks/merge, then follows the existing authorized backup/readback/install workflow.

## 18. Final revision merged, installed and normally booted

Final head `69006af` passed hosted host (46 s) and Vita (4 m 23 s) CI, then merged at 10:59:00 UTC as `a5e0c50b1096b5ae7de450ae712e63b272dc7f17`. The standalone local main reference was fast-forwarded without changing concurrent worktree branches.

Kernel SHA `6cc2dc86...` (38,620 B) and text SHA `84259631...` (7,936 B) were installed sequentially with expected-old-hash guards, native completion and whole-file readback comparisons. The low-memory shell SHA `45f86b1e...` remained unchanged. No config, region or firmware changes accompanied this deployment; the emulator bridge was not installed.

Reboot was first accepted, then independently confirmed by reconnection and a fresh 706 B status log at 11:00:20 UTC: API 8, shell registration 0, worker started. Complete native inventory (27 entries) showed kernel/shell STARTED 6 with zero errors. The per-title text plugin was unloaded with the game closed, as expected. The notes author independently hashed both saved readbacks and inspected the startup log.

The installation report now states `installed_readback_verified_normal_boot_verified_awaiting_user_clannad_test`. The user was asked to open CLANNAD, select matching text in the L+R picker, then advance/reopen twice; their result is pending. This records successful installation and boot, not physical dialogue capture. Exact hashes and private evidence locations are in [[02-Current-device-state]].

## 19. Physical test reports changing 0% hooks and irrelevant text

The user opened the installed overlay and reported a flood of changing 0% hooks, then irrelevant text. This fails the intended physical workflow despite successful boot and nonzero candidate traffic. Saved status metadata shows candidates 1→5 before opening, then full discovery and sequence growth to 3,035 with selected=0. No correct current-page match or selected-source continuity was established.

The user explicitly requested the default flow take one OCR screenshot and show the match. Root acknowledged that the device was configured as manual `text_source=hooks`, so no OCR reference existed for the 0% labels. The repository default `text_source=auto` and logged `ocr_mode auto` do not override that physical setting.

Baseline worker/renderer inspection explains additional churn: opening starts full memory discovery, picker publication follows every table sequence change, and the renderer prints match percentages even without a reference. A new `fix/ocr-first-hook-picker` branch starts from `a5e0c50`, with worker/config/device, OCR-memory and UI work split among agents. The prior physical capture result remains unverified; this is a UX failure to fix, not a completed text-hook success. See [[17-OCR-first-physical-workflow]].

## 20. OCR-first worker and explicit picker states under development

Root is implementing one OCR anchor job with one 384 KiB workspace, no dictionary lookup during anchoring, and release only after job completion. The renderer work adds explicit none/matching/ready/failed states, a bounded screenshot excerpt, readable previews and scores only when a valid OCR reference exists. These are development changes, with no live OCR or new physical deployment validated yet.

The notes author inspected the UI/view-model diff and preserved the distinction between inspected work in progress and shipped behavior. Publication of this notes follow-up is held until implementation results arrive, while the current-folder mirror already records the user's failed physical UX and requested OCR-first default.

## 21. Lens accepted; OCR-first lifecycle fixes and intermediate validation

The user explicitly accepted Google Lens for lower local memory demand. Root confirmed it was already configured/default and kept it. The working tree now uses one temporary 384 KiB anchor workspace with no dictionary lookup, LISTEN rather than full memory scanning, a screenshot excerpt, stable candidates at ≥50% normalized string similarity, explicit source selection and retries only on explicit open/reopen or Triangle.

Reviews caught stale clicked rows, close/advance/reopen consuming an old screenshot, and dictionary-to-picker transitions losing the reference. Focused production-worker regressions cover these lifetimes; cancellation cannot free an anchor still used by the worker. An intermediate full 10-suite host run passed in 4.86 s, plus Vita OCR-disabled build/ABI, before the newest stale-choice regression. Final rebuild is pending. No new physical install or live Lens success is claimed.

## 22. OCR-first candidate published; local validation and hosted config mismatch

Overlay PR #13 publishes code `4205be7` and docs head `a048faa`. The final focused production-worker run passed 36 cases in 0.02 s; the prior full 10-suite run passed in 4.86 s. Vita release/packaged ABI passed and both independent reviews found no remaining blockers. UI preview sanitizers passed 650 cases. The new shell is 158,516 B, SHA `28097a40d824fb0a97abe69d8ea851e046ef9477e2f3e50a85ffaa05564cce10`; no kernel/text changes are needed.

Host Lens probes measured 117,336/133,720 B arena peaks for two private screenshots and up to 391,769 B with padded responses, supporting the chosen 384 KiB allocation for those tested cases. These are deterministic host allocation results, not live OCR, console speed or a universal response-memory bound. See [[18-OCR-anchor-memory-measurements]].

Hosted head `a048faa` failed the core config golden comparison because its explanatory config-example comment was not synchronized with the generated default string; worker tests passed. Root is correcting that mismatch and rerunning full host/Vita checks. Device shell/config/region backups are prepared; Lens remains selected with only hooks→auto planned. No new install or physical OCR result has occurred.


## 23. Config equality corrected and final worker regression added

Head `5826329ddcb1244b0e318f2c6c527181cc14d4d8` synchronizes the generated-default comment with the example config and clears the subtitle busy indicator when cancellation completes. The final 10 host suites pass in 4.18 s, including 37 worker cases; the Vita release/packaged ABI check passes. The new shell is 158,564 B, SHA `32baceebd6b9af749b7b52a59336fe9225bef382f20eec10a60c41b930815d3c`, superseding the earlier candidate. Hosted host passed at this exact head; Vita remained in progress in run `38048108519` when checked. Device shell/config/region backups are prepared, but the candidate is uninstalled and live OCR remains unverified.


## 24. Live host Lens recognizes the previously verified page

Two production JPEG/OCR/BearSSL requests at 11:23–11:24 UTC completed with HTTP 200, TLS 0 and `rc=0` in a 384 KiB arena. The full 960×544 page used a 131,128 B arena peak; the saved 755×146 crop used 93,432 B. Both exactly matched the same verified six-codepoint current line and produced 100% normalized similarity against all three established streams. These single host samples establish a narrow real endpoint result; they are not shared-benchmark CER, repeated timing, console memory measurements or a physical UI test. Evidence and reproduction paths are in [[18-OCR-anchor-memory-measurements]].


## 25. OCR-first change merged; deployment verification pending

Exact head `5826329` passed hosted host (52 s) and Vita (4m25s) in run `38048108519`. PR #13 merged at 11:26:11 UTC as `539c2888f61b28277bb7fd7bdd2b6a4bb5b330fd`, and local main was synchronized from `a5e0c50`. Root began the shell upload. No successful new readback/normal boot or physical OCR interaction has yet been recorded at this checkpoint.


## 26. OCR-first shell installed and normally booted; user test pending

Full readback verified shell 158,564 B SHA `32baceebd6b9af749b7b52a59336fe9225bef382f20eec10a60c41b930815d3c` and config 3,082 B SHA `b7c0dd2e5214c0d00612e832abeba2b5ea5dde67ad63875745764ce4ae4beeb0`. The notes author independently checked those local readbacks and confirmed only hooks→auto changed; Lens and the region remain preserved. The fresh 11:27:24 UTC startup log records API 8, registration 0 and worker startup. Root reports complete 27-entry native inventory, kernel/shell STARTED 6, zero errors, and CLANNAD confirmed foreground after launch. The user was asked to show dialogue, press L+R and report the OCR excerpt/stable match or exact error. Physical OCR remains pending. Narrow evidence paths and current hashes are in [[02-Current-device-state]].


## 27. Physical OCR fails with source error -108

The user reports “text recognition failed. unexpected response (-108). no game text found yet.” This supersedes the pending physical test, while verified installation/boot and host Lens results retain their narrower scopes. `core/conn.h` labels -108 `VJO_E_SOURCE`; inspection finds capture request, timeout, capture result and non-OOM JPEG failures all mapped there. The UI has no dedicated source-error case and uses generic “unexpected response” wording. No original branch/cause is yet confirmed. See [[19-Physical-OCR-source-failure]] for the stage table and evidence needed.


## 28. Paired physical logs identify full-frame capture allocation failure

Six attempts in `status.ocr-failed.txt` each return capture-request error -1073741826 (`0xC0000002`), job -108 and arena peak 0. Six corresponding `kernel.ocr-failed.txt` entries report the 2,040 KiB capture-buffer allocation failed with `0x80024302`. No JPEG/Lens request began. The notes author inspected these markers and recorded file hashes in [[19-Physical-OCR-source-failure]]. Root is developing storage sized to the selected 755×146 region; that correction is not yet installed. The expected -2 no-memory return becoming `0xC0000002` is a separate unresolved propagation issue under review.


## 29. User chooses fixed manual hooks; selection works but text is suspect

After suggesting optional downscale/black-and-white, the user accepted ordinary hooks without OCR if the list stays stable and selected the fixed manual-list option. Root changed only auto→hooks, guarded the actual config hash and verified its complete 3,083 B readback (SHA `3833843d...`), without reboot. Session 12 then selected ID 1, kind 2/encoding 0, six bytes and one update. A 256 KiB Paf dictionary job returned rc=0 in 37 ms with 110,080 B arena peak; later reopens returned rc=0 in 21/25 ms. The user thinks the text is wrong; screenshot capture fails at JPEG encoding, so root requested a short example. This proves allocation/selection/job completion, not the dialogue stream. See [[20-Fixed-manual-hook-workflow]].

## 30. Region-sized capture and explicit JPEG ownership under review

The uncommitted `fix/region-sized-ocr-capture` patch requests 432 KiB for the saved crop rather than 2,040 KiB for every frame. It freezes crop geometry, checks capacity, publishes pending state atomically, preserves multipass local OCR and frees single-pass JPEG raw storage after the final copied row before Lens. A DISCARD flag handles an idle failed encoder without freeing pending or multipass ownership. Reviewers cover lifetimes/regressions. No completed final build/PR/deployment is claimed at this checkpoint; manual hooks remain the chosen physical mode.


## 31. Capture repair published with final local validation

PR #14 head `5118c4a09248a3d8b7e8018d85fa0d0247b82ca4` passes 10 host suites in 4.61 s (41 actual worker cases, 16 production-kernel cases), kernel ASAN/UBSAN, SDK ncnn-OFF Release and packaged ABI. The kernel audit counts 6,055 ARM instructions and zero FPU/NEON. Two independent reviews completed; hosted host passed in 56 s, Vita remained running. The repair is not merged or deployed. User-selected manual hooks and installed binaries remain unchanged.

## 32. Physical wrong-text concern keeps title-specific investigation open

The user asks whether real Vita hooking is possible and whether to give up. Root preserves the difference between the verified two-page emulator capture and the unverified/wrong physical selection. A private candidate at module-relative `0x2C906`, r1/CP932, is awaiting validated register captures; it is not an installed or proven profile. A completed native app inventory has no running entries because the user closed CLANNAD, not evidence of a new crash.


## 33. Capture repair merged; title-profile proof still absent

PR #14 final head `5118c4a` passed hosted host 56 s and Vita 3m41s, then merged at 11:46:20 UTC as `c888b5936ca46729b0fbc7698b2348c5a4b3d387`; local main/origin are synchronized. Prepared private artifacts are kernel 39,048 B SHA `ae3c0341...`, shell 158,848 B SHA `83dac94f...`, unchanged text `84259631...`. The notes author independently hashed them. No release or physical install occurred.

The first candidate-profile emulator run has no kind-4 register stream, so attachment/capture is still being diagnosed. A separate small bounded picker-diagnostics branch is in development for explicit-action observations, with private game text excluded from public notes. The user has not been asked to test the prepared capture build; manual hooks and the wrong-text concern remain current.
