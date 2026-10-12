---
title: Meiki hardware and project lessons
created: 2026-10-11
updated: 2026-10-11
status: settings-png-repair-controls-verified
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
- Seek a terminal promotion result; if its status remains uncertain, reconcile
  supported native registry state and installed bytes separately. Compare the
  installed executable with the audited release. Static SFO `APP_VER=01.00`
  and a versioned upload filename do not identify the actual installed build.
- Pin library artifacts and check their ABI against the linked console runtime.
  A rebuilt app can still inherit an incompatible prebuilt dependency; host
  substitutes do not exercise its native allocation and cleanup paths. Use
  completed dumps, matching native images and instruction/layout evidence.
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

## Published follow-up and measured host validation

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
Daijirin is absent. Native installation was pending at that release checkpoint;
the later v0.7.7 installation is recorded below. Linux x86_64 results are:

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

**Measured conversion and verified installation, MC03:** the user's private Daijirin v3
export converts to a 156,827,663-byte disk-backed `.vjdict`, SHA-256
`e767f405bcd8021f7fb226b8ca6da70ba0b6250782e8afc998bf924ab7c4fb53`.
Conversion produces 334,741 entries; 53 long definitions are truncated to the
7,000-byte definition limit. The production selected-position API is measured
with 25 host cases, six samples each, returning at most one entry. Its maximum
arena peak is 53,584 bytes inside the 64 KiB caller arena. This is lookup memory,
not dictionary residency, OCR memory or native storage latency. The whole
dictionary remains on disk.

The derived report is `docs/benchmarks/daijirin-v3-selected-20261011.json` in the
overlay candidate. Native verification of the complete staged file passed in
53.27 seconds. Commit initially required recovery, and a bounded recovery wait
timed out; the original operation later completed native verified recovery in
72.143 seconds. A subsequent native stat confirms the installed size. The
read-back configuration places Daijirin before JMdict, but an actual lookup
under that saved priority remains unproved. See the native installation
checkpoint below. The dictionary's contents and source archive are not
redistributed; public release assets contain JMdict and omit Daijirin.

**Pending organization, MC04:** a read-only inventory contains 59 existing apps,
and a private seven-page LiveArea plan is prepared. Applying the plan awaits
supported, observed Edit Mode moves and native page/slot verification. No
completed move, rewritten AppDB/layout file or unseen drag is credited here.

Native installation and readback are established; later native Settings engine
controls are measured in NI07. Japanese-only navigation in a game, actual
lookup using the new dictionary priority and LiveArea placement remain separate
checks. Preserve the working rollback and earlier hardware observations.

## Settings readability follow-up: controls verified; font inspection pending

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
text; native visual review remains required. The repaired app's later native
preference-input checks are recorded in NI07.

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
The package is now installed and verified below; no actual visual-readability
success is claimed.

See [[09-Device-operations-and-recovery]], [[08-Validation-and-measurements]],
[[06-Low-memory-overlay]] and [[13-Note-maintenance]].

## Native installation checkpoint

**Observed completed installation, NI01:** native recovery of the original
Daijirin transfer finished in 72.143 seconds with `code=0`, `verified=true`,
`complete=true` and `readback_required=false`. Native destination stat confirms
156,827,663 bytes. This supersedes the recovery uncertainty retained below;
the transfer identity, native journal and existing pairing were preserved.
No commercial dictionary bytes or raw native receipts are published.

The original v0.7.7 Settings promotion reports `installed=true`, `running=false`,
`native_result=0`, title `VJOSET001` and audit sequence 1778. Full native readback
of the installed 277,084-byte `eboot.bin` has SHA-256
`2a34d0c7b2a7e4a500b34be4626d365d3bd9b715d7b4fc85bba807706e1f3b11`
and is byte-equal to the release VPK's executable. Installed SFO
`APP_VER=01.00` is static package metadata, not the release version. The
149,220-byte Shell readback matches v0.7.7 with SHA-256
`d120e548be9e24f36f5e9551cb6ab8792eb5e595ae29430241450f97b401e308`.
These are the published binaries; this checkpoint adds no runtime code changes.

**Observed saved preferences, NI02:** the complete 2,951-byte configuration
readback has SHA-256
`312ce783379b86e0194c04750eb2fe7e120cd51f29dcb5085ed01dd2dab79c00`.
Comparing parsed values before and after finds exactly three semantic changes:
`ocr_backend=vocr`, `japanese_only_selection=on`, and Daijirin first with
JMdict fallback in the local dictionary list. `dictionary=local` was already
selected. `ocr_mode=on_press`, unrelated fields and private credentials remain
unchanged. Saved configuration proves the requested defaults and order; it
did not prove a physical Settings toggle or actual dictionary lookup. Later
native preference-input proof is recorded in NI07; game lookup remains pending.

**Observed boot and process launch, NI03:** after the reboot accepted at
13:56 UTC on October 11, a fresh snapshot at about 30 seconds uptime reports
Kernel/Shell native state 6 with zero query errors. Fresh Shell startup reports
API 13, PAF loaded and worker started. Settings launch at about 62 seconds
uptime is confirmed for `VJOSET001`, with no apps closed by that launch. The
native Settings snapshot at that checkpoint reports `system_ui_overlaid=true`,
and no buttons were sent to dismiss it. The user was asked to open the app
physically and subsequently confirms Settings is visible, but reports an
error. The completed native dumps and linked-code diagnosis are recorded in
NI04; no automated UI inputs or successful save/toggle were credited at this
original checkpoint. NI07 records the repaired app's later measured inputs.

**Diagnosed native crash, NI04:** two completed native Settings dumps compare
with `same_bucket=true`. Their three reported differences are process,
register and relocated-PC metadata; both resolve with the matching native ELF
to `_free_r` at Settings module offset `0x442f0`, with invalid read address
`DFAR=0x7f80dea9` (the invalid pointer `0x7f80dead` minus four). Qualified stack
scanning points to the PNG loader and
`png_free_jmpbuf`. A scanned address is supporting evidence, not a complete
source-level unwind.

Actual linked `setjmp` instructions save ten general-purpose registers
(40 bytes) plus `d8`–`d15` (64 bytes), writing 104 bytes. The shipped libPNG
inline jump buffer is 92 bytes, with its cleanup function, buffer pointer and
size metadata at offsets 92, 96 and 100. The runtime save therefore overwrites
that metadata; later cleanup follows the corrupted state into the invalid
free observed in both dumps. The current SDK defines a 160-byte `jmp_buf` with
eight-byte alignment. The substitute-font host harness does not exercise this
linked dependency/runtime ABI.

A separate repair branch rebuilds pinned libPNG 1.6.58 against the
current SDK headers, making its inline buffer 160 bytes while retaining the
Settings design and OCR source. Independent analysis finds exactly one
92-byte request in the existing libvita2d caller; the rebuilt libPNG's first
request returns its full 160-byte inline buffer, and constructor/destructor
locals also use 160 bytes. The build and reconciled installed-file observations
are recorded in NI05; NI07 records native controls after the repair. No
physical firmware-font readability success is claimed.

Private metadata locators: `settings-v077-crash-symbols.json`,
`settings-v077-two-crashes-compare.json` and
`settings-v077-crash-events-01.jsonl` and the independent analysis directory
`independent-crash-audit/` in the checkpoint directory. Native events
2568 and 2595 report completed dumps. The second private dump is 90,096 bytes,
SHA-256 `8019ab73ca390adc7fd117a97869072f666ad07fd8e82e4fedf91bca33477f7c`;
only small derived metadata is recorded here. Raw dumps, firmware/game images
and native identities remain private.

The completed independent report is
`independent-crash-audit/verification.json`, SHA-256
`39c6975942b066da58abc25084c655668e5a9378402202058172c72da51faf2c`.
It records `d14` and `d15` each containing `0x7ff8dead7f80dead`, predicting the
overwritten buffer pointer at offset 96 as `0x7f80dead` and the fault read at
pointer minus four. The host-side native-image gate rejects the old 92-byte
layout and accepts the repaired 104-byte save within 160-byte storage; six
adversarial cases pass. These are ABI/image checks, not a physical launch test.

**Measured repair and installed-file reconciliation, NI05:**
[overlay PR #36](https://github.com/bee-san/vita-jp-overlay/pull/36) at exact head
`1c8e8db1d4df301342416b6d68c4998e9dddbd67` passes host and native CI in
[run 38146423675](https://github.com/bee-san/vita-jp-overlay/actions/runs/38146423675).
The repair retains the existing libvita2d, Settings design and OCR/plugin
source, rebuilding the pinned PNG dependency and checking its actual linked ABI.
The complete 283,190-byte candidate VPK was uploaded and fully read back with
SHA-256 `c8d032acca58ea4ff586c9de7c505f9dc5ea068af6d408659c7d459e43f2bd04`;
its ZIP CRC check passes.

After a disconnection, retained native promotion status reports
`state=uncertain`, `phase=idle`, `running=false`, `installed=null`. This is not a
terminal installed receipt. Reconciliation obtains a complete 88-entry native
app inventory with Settings launchable and a complete installed `eboot.bin`
readback of 277,212 bytes, SHA-256
`217628a7fa68d10d8ae9f3f194d37eda53a1e0e2da5bccd69a2c86a5e6cb9299`,
byte-equal to the repaired candidate. The first repaired process launch is
confirmed, with scoped native controls recorded in NI07. Preserve the retained
operation identity and unresolved terminal promotion status separately from
the verified installed bytes.

PR #36 merged as `b0642493688c0f31bc33738d937d876ba0edbab6`, and
[release v0.7.8](https://github.com/bee-san/vita-jp-overlay/releases/tag/v0.7.8)
was published at 14:23:11 UTC on October 11. All six downloaded assets are
byte-equal to the independently audited candidates; all downloaded ZIP CRCs
pass. The private receipt is
`vita-release-2026-10-11/published-v078-audit/verification.json`, SHA-256
`36aae9d2b0a275e1d460a30e39191a66d2142107012844e66cbebf40ce269d77`.
This establishes GitHub publication separately from native promotion status
and actual runtime checks.

Private receipts: `settings-v078-candidate-readback.jsonl`,
`settings-v078-candidate-install-status-03.jsonl`,
`settings-v078-installed-eboot-download.jsonl` and
`settings-v078-first-launch.jsonl` in the checkpoint directory. No raw registry
inventory, native identities or operation IDs are published.

**Observed earlier power-management gap, NI06:** the previously loaded agent called
PowerTick with argument 1 (`AUTO_SUSPEND`), leaving the screen timer available.
Synchronous native work blocked the existing service-loop tick. NI09 records
the later independent-worker candidate and native file readback; physical
long-request wakefulness remains a separate measurement. Pairing, stop policy
and retained operation state must remain intact.

**Measured native controls, NI07:** the repaired Settings process remained
foreground with `system_ui_overlaid=false` over 100.106 seconds measured from
its confirmed launch. Six controller submissions, including initial SELECT,
and two page/focus touch submissions finished with zero error and released
input. Five subsequent controller actions each had a complete config readback:
RIGHT selected Meiki, LEFT restored VOCR, DOWN selected the Japanese-only row,
LEFT retained the already-OFF value, and RIGHT restored ON. This measures
engine switching and OFF-state selection/restoration to ON, not an ON-to-OFF
transition. The final 2,951-byte config is byte-equal to NI02, preserving local
dictionary order, `ocr_mode=on_press` and unrelated keys.

The initial SELECT/readback unexpectedly showed Japanese-only OFF. The
baseline assertion caught this; user/interleaving causation is unknown.
Two explicit OCR-tab/VOCR-badge touch actions normalized focus before the
measured sequence. The sampled native event window, sequences 2635–2648,
contains no coredump and reports zero lost events; it does not establish
indefinite crash freedom. Private receipt:
`settings-native-check/verified-controls.json`, SHA-256
`8d8bd3905e0ff6304e949fdc8d75b19a2f618befd20763487c2794c6ec652d5d`.

**Measured Japanese-only transition, NI08:** a later scoped check resolves
NI07's idempotent OFF limitation. Touching the OCR tab retained ON; tapping the
Japanese-only card changed the saved value to OFF; RIGHT restored ON. Every
action completed and had a full config readback. The final configuration is
again byte-equal to the intended 2,951-byte file in NI02. Private receipt:
`settings-native-check/verified-japanese-transition.json`, SHA-256
`2bd521fae593dbdb0e1b4638f91dcd822bb317650aa2b2e48b77abd90263b9b1`.
This proves the Settings ON → OFF → ON save path while preserving the original
discrepancy, not Japanese-character navigation within a game.

**Measured keep-awake build, native readback and activation, NI09:** local-only agent
commit `1d8a438d51454c27508a42849e0674080f01e9cd` is based on
`87e9bc6feea3401d16844ee7d9b5d89c74ab3316`; the clean baseline build is
byte-identical to the previous installed Shell, SHA-256
`d30181867cba4c8922b7f744796a4cddb6455b44702c121263d04b131e87704f`.
The candidate adds an independent 8 KiB worker issuing DEFAULT ticks every five
seconds while authenticated command control or pending native
install/decrypt/content work is active. It performs
no worker I/O and changes no trust, pinning, ACL, stop/input or operation policy.
Eighteen tests compile the actual worker under ASan/UBSan, including blocked
owner execution and 20 repeated lifecycles; no sanitizer report is recorded.
Host wait acceleration validates scheduling and the five-second native timeout,
not physical elapsed timing. Imports remain 31 libraries and 143 functions.

Native upload completed with code 0, complete/verified state and audit sequence
1799. The 311,356-byte complete native readback matches SHA-256
`7b803d324963257fcdc0d88d7f141ea8ae3c753b7520ccd759253956341de5c4`.
An ordinary reboot loaded the worker. Fresh runtime log records initialization
at monotonic 7,174,029 microseconds with result 0 and the first DEFAULT tick at
21,416,363 microseconds with result 0/count 1. A fresh snapshot follows at
89,680,721 microseconds; Settings relaunch is confirmed at 145,801,860
microseconds and its snapshot reports foreground `VJOSET001` with
`system_ui_overlaid=false`. The post-boot config is byte-identical to the
intended 2,951-byte file in NI02. These establish native loading and a successful
tick, not physical display inspection. Protected active control is the scope;
no claim covers pairing waits, physical OLED visibility or hardware wakefulness
during a long request. Kernel and loader were not rebuilt or deployed.

**Measured command-free connection check:** an interval starting at 14:34:32 UTC
lasted 374.066 seconds. Only PC-side status reads occurred; no native, wake or
input request was issued before the end UDP diagnostic. That diagnostic reports
phase 8 and stop guard ready 1/held 0/stopped 0/error 0. A fresh ordinary snapshot
responds in the same boot with the same Settings process foreground and
`system_ui_overlaid=false`. The 14-event post-boot window reports zero lost
events and no disconnect, stop or coredump. This establishes a responsive native
connection after the interval. The final snapshot uses the normal screen-on
path, so the physical OLED's visibility throughout the interval was not
measured. Worker ticking during a blocked owner is covered only by host tests.

Private host receipt: `vita-agent-use-keepawake/agent/keepawake-review/verification.json`,
SHA-256 `e3b46b052ac6155994125e8c6accc7a52bae178c0270cf968708ffeff763cfad`.
Private native receipt: `keepawake-audit/native-activation-proof.json`, SHA-256
`46393c45a75acea2d430b65f82b1c6fc3a84729e57468947482a6d29fe011224`.
Private quiet-interval receipt: `keepawake-audit/verified-quiet-interval.json`,
SHA-256 `f7b3bb144f5436e13c27764515bde9b2d33ed88b10be2fa7ed53435ffd65c7bd`.
This source revision remains local-only; no upstream publication is credited.

Native installation, readback, boot, repaired process execution and saved
preference controls are established separately. Physical firmware-font
readability, Japanese-only navigation in a game, new game inference and
actual Daijirin-first lookup remain pending.
Earlier Meiki/VOCR game results do not establish those new-release checks;
hardware accuracy/CER remains unmeasured. LiveArea organization also remains
pending and is unaffected by this checkpoint.

Private evidence locator: `vita-agent-use/agent/ocr-choice-20261011/`, including
`daijirin-recover-after-reconnect.jsonl`, `daijirin-installed-stat.jsonl`,
`settings-v077-install-status-02.jsonl`, installed-executable/SFO and Shell/config
readback receipts, and fresh startup/launch observations. Configuration,
dictionary contents, native identities and operation IDs stay private.

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

**Historical transfer state, 2026-10-11 13:23 UTC:** full staged-file native
verification passed in 53.27 seconds. Commit returned recovery-required, and
recovery with the bounded 600-second host response budget timed out while the
Vita went dark.
At that checkpoint the user had woken the device, but recovery of the same
recorded operation was pending. A host response timeout did not prove
cancellation or a completed remote effect. Preserving the transfer identity
and native journal allowed the later completed recovery recorded in NI01;
installed destination verification and Settings promotion are now established.
Actual dictionary lookup and physical firmware-font inspection remain pending;
later native Settings preference controls are measured in NI07.

Saved-session reconnect then timed out after the wake; native diagnostics
reported the command service returning to pairing with stop monitoring ready,
no held/stopped chord and no guard error. A physical reboot/open-Settings step
was requested. The transfer journal and identities remained intact; that
observation alone did not establish destination installation. Host waiting
messages are not native hash progress. At that historical checkpoint, the
source-level power-tick gap had not yet been tied to the installed runtime.
NI06 records the later installed-image audit; causation of each disconnect
remains unproved.

## Later touch/highlight and sorting candidate

**Superseded checkpoint:** the initial pending activation and sorting below are
resolved by [[28-Touch-lookup-and-LiveArea-sort-checkpoint]]. Fresh Shell loading
and physical scoped approval were subsequently established. A guarded sort
failed and all 59 coordinates remained unchanged; the user canceled sorting.
Do not resume it from this historical candidate description. Game-header touch
and physical font readability remain unmeasured.

[[28-Touch-lookup-and-LiveArea-sort-checkpoint]] records the next merged overlay
implementation, verified stable v0.7.9 publication and native file readbacks.
Promotion lost its connection while running at 15:34:40 UTC; its retained status remains uncertain. Later complete
installed-executable readback, registered native launch and four actual Settings
touch/controller highlight actions establish the new Settings controls
independently of that status. Whole-word highlighting retains character
navigation without a tokenizer. Fresh Shell loaded-module proof, game-header
taps and physical font readability remain pending. The separate guarded native
sorter uses native AppDB operations and a SHA-bound complete plan, with 12
actual-coordinator host sanitizer cases and a clean native build. Complete
runtime readback and an accepted reboot do not establish fresh activation;
physical ACL approval and hardware moves remain pending. These later
stages do not replace the dated native successes above.
