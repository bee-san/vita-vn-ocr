---
title: Meiki hardware and project lessons
created: 2026-10-11
updated: 2026-10-11
status: settings-readability-published-native-installation-pending
tags: [vita, ocr, meiki, memory, deployment, lessons]
---

# Meiki hardware and project lessons

**Observed:** the user confirmed Meiki and the repaired region controls work on
the physical Vita. This supersedes the earlier Meiki visibility and input
uncertainty in [[26-Game-process-OCR-hardware-checkpoint]]. It does not establish
OCR accuracy, broad game compatibility or a complete residency measurement.

## Exact build and runtime observations

[Overlay PR32](https://github.com/bee-san/vita-jp-overlay/pull/32) restored Square
and Circle in the ordinary overlay input handler. Tested head
`996642e5da5e3c87ae72bd748329e4d568539ce1` and merged main
`8cf11bb0eb8b3884f4299fe08ac6239ebb0bec7b` have tree
`a60400ce7fb3204910fdaeb306a95ca767e05cc8`.
[Release meiki.2](https://github.com/bee-san/vita-jp-overlay/releases/tag/v0.7.6-meiki.2)
contains the changed 147,544-byte Shell with SHA-256
`7be08ef1e6993be63d521d165a0e169b290d3deced16a2c7eb7e8cce64e70295`.
Native staged upload and full readback match. The API13 kernel, game worker and
ABI3 engine were retained. One reboot completed and the existing pairing
reconnected; no screenshot, injected input, tai configuration change, theme
write or database replacement was performed in that test.

Two actual CLANNAD jobs ran Meiki from the retail game's process through the
game-visible rePatch `app0:/VitaJPOverlay/meiki/meiki-engine.suprx` path. Both
used 864×44 pixels, returned one line and finished `rc=0 io=0 cleanup=0`.

| Measurement | First job | Second job |
|---|---:|---:|
| End-to-end job | 7,904 ms | 8,043 ms |
| Model load | 2,153 ms | 2,149 ms |
| Recognition | 3,483 ms | 3,478 ms |
| Shell lookup arena peak | 53,488 bytes | 53,488 bytes |

The first job reported 91,226,112 free game USER bytes before the 27,648,000-byte
preflight. Requested private buffers were 21,356,544 bytes; measured neural
and metadata peaks were 13,090,176 and 2,764,800 bytes. The reader returned
5,392,856 bytes without error. These are logged samples from varying dialogue,
not a controlled speed benchmark or a total memory peak. Hardware CER remains
unmeasured. Host reader/path results remain separately attributed to the pinned
runtime in the overlay's documentation.

Fresh touch-region enable/disable and physical drag records prove Square enters
region mode. The user reports the repaired workflow works. A ten-case harness
extracting the actual input functions passes under ASan/UBSan; the previous
handler fails nine cases. The harness also covers Cross save, reverse drag,
Circle cancel/close, held Square and Unicode cursor preservation. The native
log did not independently show a Cross-save marker, so preserve that distinction.

Private evidence locator: `vita-agent-use/agent/meiki-region-controls-20261011/installation-verification.json`.
Raw dialogue, native JSON, config, model files, screenshots and identities stay
private.

## Lessons to keep

- Run neural inference in the enabled game process. `*main` is SceShell;
  it cannot spend the game's unused USER budget. Keep only UI and bounded
  disk-backed dictionary lookup in Shell. An `*ALL` worker must gate itself
  to enabled foreground titles; per-title entries are a narrower option.
- A game's advertised 256 MiB allowance is not a free-memory measurement.
  Check actual USER headroom, preserve a game reserve and handle allocation
  failures. Never infer a largest free chunk or full physical availability.
- Plain-text dictionary output reduces rendering and result memory; it does
  not make an entire dictionary resident. Keep disk indexes, bounded reads,
  one returned entry and manual character navigation. Models stay on storage
  and only the selected recognizer allocates its transient workspace.
- Meiki needs one horizontal line, at most 96 pixels high. Tap/release Square,
  drag a line and Cross to save. Holding Square resets full-screen geometry.
  Navigation through saved OCR text should never recapture or reinfer.
- Poll input in error views. Missing Square/Circle handlers caused the dead
  controls; error rendering itself did not disable them. Check every native
  controller encoding and pressure field when hiding trigger buttons.
- Use the saved native agent identity and one server/operator. Preserve IDs
  for uncertain effects; accepted is not complete. Back up, check expected
  hashes, stage sequentially and read back before reboot. Avoid overwriting
  settings, device registries or newer theme entries with old backups.
- Keep secrets and copyrighted data out of tool output and public artifacts.
  Config edits preserve unrelated fields. Commercial dictionaries can be
  converted and installed privately while the converter remains public.
- Check settings VPK artwork format, not just whether PNGs decode on a host.
  Indexed opaque images repaired the earlier VitaShell install error.
- Design the standalone Settings app for the Vita's 960×544 screen: larger
  text, fewer visible cards, distinct focus and state, useful touch targets
  and bounded wrapped messages. Share drawing and touch geometry. Measure
  full-width PVF advances for em targets; its nominal `text_height` is not a
  firmware glyph-bitmap bound. A substitute-font host preview cannot prove
  native crispness, physical readability or touch behavior.
- Use [[25-Vita-debugging-tools]]: `vita-agent-use` for native logs/events,
  `psp2-err` for the exact error code and `psp2-core-parse` for completed dumps
  with matching images. A Shell/LibKernel image is not automatically a matching
  source-level backtrace, and absent dump memory cannot be reconstructed.
- Host tests and Vita3K validate useful interfaces but do not prove taiHEN
  retail behavior. Record build, publication, readback, boot, real inference
  and user-visible confirmation separately. Mark unresolved accuracy clearly.
- Land small changes through the user's fork PRs, never push main or publish
  upstream. Pin source/dependencies, freeze test revisions and independently
  download and verify release manifests.

## Current follow-up: built and locally validated

**Implemented, MC01:** JP Overlay Settings retains its Games page. Press
**Select** to open the OCR/lookup page, then use Cross or touch to switch
VOCR/Meiki. Left/Right explicitly choose VOCR/Meiki on the model card and OFF/ON
on the selection card. VOCR is the default; switching to Meiki preserves its
manual, single-horizontal-line requirement. Both recognizers remain on storage,
and only the chosen one requests an inference workspace. The app reports model
file presence; that check does not validate the model or run inference.

`japanese_only_selection = on` is the separate, default-ON character setting.
The initial selection and Left/Right use Han/kana characters, skipping Latin,
decimal digits, punctuation and Hangul. OFF preserves the original all-character
cursor. This is script selection, not language identification or word
tokenization. Text without Japanese characters retains its OCR header and
performs zero dictionary lookups or result-arena allocations. Cursor movement
still reuses saved text without another OCR request.

Settings rewrites only the engine and selection keys, preserving unrelated
API keys, model/dictionary paths, mode and comments. The overlay reloads changes
when it next opens. Publication and the actual Shell loader share a 24,575-byte
text limit, fitting the unchanged 24 KiB Shell scratch including the terminator.
Fault tests cover partial I/O, failed sync/rename, backup-only recovery and
truthful save errors. The Japanese selector rejects malformed overlong UTF-8
without changing the older decoder's compatibility behavior.

**Measured local validation, MC02:** [overlay PR #33](https://github.com/bee-san/vita-jp-overlay/pull/33)
passed exact-head hosted host/native gates at
`e6c11ac688b8115202064a609500df2c0f2b1988` and merged as
`45c6f11fa1d32459d749ba106dcbcaf3c92fade6`; both trees are
`6e8cd308dba7c50f0ec56e4c6779714ee2d0bfda`.
[Release v0.7.6](https://github.com/bee-san/vita-jp-overlay/releases/tag/v0.7.6)
is published and all downloaded assets match SHA256SUMS. Its 57 unique ZIP
members comprise 56 checksum-manifested payloads and one manifest. Both model
families, JMdict and the complete dependency notices are present; commercial
Daijirin is absent. Native installation remains pending. Linux x86_64 results are:

| Gate | Result | Boundary |
|---|---|---|
| Full normal host CTest | 22/22 suites pass, 17.21 s | Actual FL10/H15 and pinned Meiki inputs; not console timing. |
| Independent Clang 22 ASan/UBSan | 22/22 suites pass, 26.26 s | 256 C cases and 42 Python tooling tests; no skips or sanitizer reports. |
| JPEG lifetime checks | 3 round trips and 5 error cases pass | Additional host tests, not screenshots from this device. |
| VitaSDK native Release build | Pass with VOCR and Meiki enabled, API 13 retained | Native build evidence; no new hardware settings test yet. |

Reproduction from the overlay checkout, after configuring/building with both
backends and pinned models:

```sh
ctest --test-dir build/host-ocr-choice --output-on-failure
ctest --test-dir build/host-ocr-choice-sanitize --output-on-failure
cmake --build build/vita-ocr-choice --parallel 8
```

The sanitizer build uses `-fsanitize=address,undefined -fno-omit-frame-pointer`
with leak detection and halt-on-error. Private locators:
`vita-release-2026-10-11/ocr-choice-owner-review/owner-checkpoint.json`,
`vita-jp-overlay-ocr-choice/build/host-ocr-choice/final-tests.xml` and
`vita-jp-overlay-ocr-choice/build/host-ocr-choice-sanitize/verification.json`.
The source hashes remain unchanged through the independent validation.

**Measured conversion and staged bytes, MC03:** the user's private Daijirin v3
export converts to a 156,827,663-byte disk-backed `.vjdict`, SHA-256
`e767f405bcd8021f7fb226b8ca6da70ba0b6250782e8afc998bf924ab7c4fb53`.
Conversion produces 334,741 entries; 53 long definitions are truncated to the
7,000-byte definition limit. The production selected-position API is measured
with 25 host cases, six samples each, returning at most one entry. Its maximum
arena peak is 53,584 bytes inside the 64 KiB caller arena. This is lookup memory,
not dictionary residency, OCR memory or native storage latency. The whole
dictionary remains on disk.

The derived report is `docs/benchmarks/daijirin-v3-selected-20261011.json` in the
overlay candidate. All converted bytes have been privately staged. Native
verification of the complete staged file passed in 53.27 seconds; the subsequent
commit and recovery remain unresolved as recorded below. Completed verification
of the installed destination is pending. The dictionary's contents
and source archive are not redistributed. Staging alone does not establish a
usable dictionary or an active priority order on the Vita.

**Pending organization, MC04:** a read-only inventory contains 59 existing apps,
and a private seven-page LiveArea plan is prepared. Applying the plan awaits
supported, observed Edit Mode moves and native page/slot verification. No
completed move, rewritten AppDB/layout file or unseen drag is credited here.

Native installation, installed-file readback, physical engine switching, Japanese-only
navigation and LiveArea placement remain separate follow-up checks. Preserve
the working meiki.2 rollback and the earlier hardware observations above while
performing them.

## Settings readability follow-up: published; native checks pending

**Implemented, SR01:** [overlay PR #34](https://github.com/bee-san/vita-jp-overlay/pull/34)
at exact head `c9ec54f9a6562ad61cbb8d9e4876e86028e652f4` packages the
Settings-only redesign as v0.7.7. The in-game overlay and OCR worker layouts
are unchanged. Settings shows three 80-pixel game cards instead of five
56-pixel rows, two wrapped title lines, a separate title-ID column, clear
ON/OFF badges and focused-card borders. Games and OCR/lookup tabs and direct
VOCR/Meiki choices have 48-pixel touch heights. Existing controller actions
and configuration persistence remain intact; only three icon textures are
retained, with the existing unload-on-scroll ownership.

**Measured host validation, SR02:** the exact `settings/main.c` SHA-256 is
`032e7bfd0689c7b69403794bb5d999cac88b1606af28c6704b3f7a81efaf4abc`.
A private harness compiles its complete input/draw loop with the real games,
UTF-8 and arena code, and bounded controller, touch, storage, PVF and GPU
substitutes. Twelve scenarios pass under Clang ASan/UBSan with no reports:
navigation and clamps, third-card touch and gaps, page switching, direct engine
choices, character-selection toggles, failed saves, unreadable preferences,
empty filters and UTF-8 wrapping. The local native VitaSDK Release build also
passes. These are build and host-interface results, not new hardware UI proof.

Three private 960×544 previews replay the actual Games, OCR and failed-save draw
calls with synthetic titles and a local substitute Noto CJK font. Independent
bounding-box checks find zero viewport overflow or intersections with their
scissor boundaries under those substitute metrics. Source colors give a
theoretical 7.68:1 sRGB contrast for secondary text on the selected card; this
is a color calculation, not a measurement of the Vita display.

**Font lesson and limits, SR03:** the pinned libvita2d sets PVF resolution to
128 dpi and character size to 10.125, then scales cached glyph textures when
drawing. Its `text_height` uses a nominal 10.125×scale; drawing uses actual
glyph dimensions and bearings. Calibrating desired pixel text from that
nominal height was an incorrect assumption and was removed before freeze.
The app now uses the largest positive scale-1 full-width advance of `あ`, `漢`
and `Ｗ`, with an 18-pixel em fallback, for 24-pixel body, 26-pixel title and
32-pixel header em targets. These are measured advance targets, not measured
firmware bitmap heights. Enlarging the cached system-font raster can soften
text; native visual review and physical input checks are still required.

Private evidence locator:
`vita-release-2026-10-11/settings-readability-qa/verification.json`.
The public [Settings readability note](https://github.com/bee-san/vita-jp-overlay/blob/c9ec54f9a6562ad61cbb8d9e4876e86028e652f4/docs/settings-readability.md)
records the harness command shape and limitations. No game-library data,
screenshots, fonts or device identity are included in the notes or release.
PR #34 merged as `b6518d0b01cf5274eddfa036f7f5f65109121450` after exact-head
host/Vita CI passed. Release v0.7.7 is published, and all six downloaded assets
match the independently audited candidate; all five payload checksums pass.
The complete ZIP has 58 unique members and 57 manifested payloads, with SHA-256
`161c8ae340550a54958213c39ff7dd66d050f77244e457a17bd72a0531138497`.
Native installation is pending; no actual visual-readability success is claimed.

See [[09-Device-operations-and-recovery]], [[08-Validation-and-measurements]],
[[06-Low-memory-overlay]] and [[13-Note-maintenance]].

## Native-agent verification boundary

A large dictionary may exceed the ordinary 20-second host response budget
during full-file SHA verification; commit and recovery can repeat full reads.
An isolated host-only client adaptation passed 13 recovery/timeout/security
tests and independent review. It allows a bounded 600-second response for
large-file hash steps with truthful five-second waiting messages; ordinary RPC
and chunk timeouts remain unchanged. Original client files, device code, pairing,
ACLs, stop monitoring and the dictionary transfer identity remain intact.

Earlier native diagnostics showed `stop_buttons_masked`: mask `0x603F0000` had
PS but lacked SELECT, with no physical stop reported. That correctly prevented
verification and input; it did not justify changing masks or skipping checks.
The ordinary Settings launch was accepted but not confirmed foreground, so
physical app opening was requested. UDP diagnostics intentionally ignore a query
within one second of the last reply, so status and input observations must be
separated by more than one second.

**Latest recorded transfer state, 2026-10-11 13:23 UTC:** full staged-file native
verification passed in 53.27 seconds. Commit returned recovery-required, and
recovery with the bounded 600-second host response budget timed out while the
Vita went dark.
The user has since woken the device; recovery of the same recorded operation is
pending. A host response timeout does not prove cancellation or a completed
remote effect. Preserve that transfer identity and its native journal; resolve
its effect before a replacement write. Installed destination verification,
dictionary activation and the new Settings installation are unconfirmed.

Saved-session reconnect then timed out after the wake; native diagnostics
reported the command service returning to pairing with stop monitoring ready,
no held/stopped chord and no guard error. A physical reboot/open-Settings step
is requested. The transfer journal and identities remain intact; this does not
establish destination installation. Host waiting messages are not native hash
progress. Power ticks outside synchronous upload work are a source-level gap,
but causation and the exact installed runtime implementation are unproved.
