---
title: Touch lookup and native LiveArea sorting checkpoint
created: 2026-10-11
updated: 2026-10-11
checkpoint_utc: "2026-10-11 15:53:04"
interruption_checkpoint_utc: "2026-10-11 15:34:40"
settings_controls_verified_by_utc: "2026-10-11 15:43:54"
status: settings-highlight-controls-verified-game-touch-pending
tags: [vita, overlay, touch, dictionary, livearea, evidence]
---

# Touch lookup and native LiveArea sorting checkpoint

The user chose whole-word highlighting with a character option, while retaining
character navigation and one dictionary result. The overlay implementation is
merged, v0.7.9 is published with verified downloaded assets, and its new files
have native readbacks. After the promotion connection loss, the installed
Settings executable and four actual highlight controls were
verified independently of the retained uncertain promoter status. The new Shell
file was read back after a new boot; fresh native module inventory now confirms
that the Shell plugin is loaded. Actual game-header taps remain pending. The
revised local native sorting helper has booted and advertises its operation.
Its verified 59-bubble plan is uploaded; physical approval and moves remain
pending. This checkpoint was refreshed at 16:47 UTC on October 11.

Earlier working VOCR/Meiki, region controls, repaired Settings preferences and
the 374.066-second quiet connection result remain dated evidence in
[[27-Meiki-hardware-and-project-lessons]] and
[[26-Game-process-OCR-hardware-checkpoint]]. They do not validate this new build.

## Highlighting and touch implementation

**Implemented and checked, TL01:** [overlay PR #38](https://github.com/bee-san/vita-jp-overlay/pull/38)
at `ee796f1ea683d84a668bbc52961bf146a9f9986c` passed host and Vita CI and
merged as `dc0bea53a6a8d075d4598cea27708953ed467666` at 15:22:53 UTC.
[Exact-head CI run](https://github.com/bee-san/vita-jp-overlay/actions/runs/38150696126)
reports both jobs successful. The local full host gate passed 22 suites;
native Release and the existing ABI, instruction and Settings PNG gates passed.
These are build/host checks, not native touch or font measurements.

- `highlight_mode = word | character` defaults to `word`, including when an
  older configuration has no key. Settings → OCR & lookup → Highlight offers
  **Whole word** and **Character** through controller and touch controls.
- Whole word highlights the dictionary's matched **source span**, including
  an inflected ending. It does not highlight a normalized spelling that differs
  from the source. Pending or unsuccessful lookup keeps the selected character
  highlighted. Character mode always uses that character.
- Arrows still move by Unicode character; there is no sentence tokenizer.
  Switching presentation preserves recognized text, and selected lookup reuses
  it without a new OCR capture. Transactional saves retain backend selection,
  Japanese-only selection, dictionary priority, paths, API keys and unknown
  configuration lines.
- A new tap on the recognized header chooses the character under the finger
  and queues the existing asynchronous lookup. It does not read an unrecognized
  glyph directly from the game framebuffer. Japanese-only ON rejects taps on
  Latin text, digits, punctuation and Hangul instead of jumping to a neighbour.
  One held touch sends one request; touch already held on opening is ignored.
- Shell owns its front touch region throughout the open overlay and crop
  editor. Closing releases it; a failed activation is not treated as active.
  The existing game touch filtering remains in place. Its actual routing on
  hardware still requires a separate check.

**Implementation geometry, TL02:** hit testing queries PAF `GetCharInfo`
character starts and line numbers, the header's line count/content height and
its current scroll. The uniformly styled header uses content height divided by
line count for line pitch. The last cell's width is bounded by the configured
Japanese point width; it is not a measured glyph bitmap. Blank trailing areas,
clipped rows and unavailable layout produce no selection. Undocumented
`LineInfo.frame` fields and `GetNextDownHit` are not used.

The saved UTF-8 text is bounded to 4,095 bytes plus NUL, and the styled header
budget is 4,096 UTF-16 units. Mapping omits CR, retains newline/wrapped-line
coordinates and maps a supplementary character's surrogate pair to its original
UTF-8 start. The touch helper allocates no memory and borrows no dictionary
arena pointers outside the view lock. Host tests cover proportional advances,
wrapping/scrolling/clipping, Japanese-only rejection, CR/newlines, supplementary
Han and unavailable layout. Runtime PAF queries are the implementation's
geometry source; this checkpoint contains no measured hardware finger accuracy
or new firmware glyph bounds.

[Overlay touch contract and test boundaries](https://github.com/bee-san/vita-jp-overlay/blob/dc0bea53a6a8d075d4598cea27708953ed467666/docs/touch-lookup.md)

## Native transfer, interrupted promotion and verified Settings controls

**Measured file readbacks, TL03:** native upload and complete local readback
establish these bytes. Their activation stages remain separate.

| Artifact | Bytes | SHA-256 | Established stage |
|---|---:|---|---|
| New Shell plugin | 150,500 | `2e477d8d2616e3b9f72a5cc2da354d2af42719c9e7c594b4a583312329770716` | Replaced at the installed plugin path and fully read back after a new boot; fresh native inventory confirms loaded after restart |
| New Settings VPK | 283,521 | `3e9795fbfd833379d090b90482783cb05cf99f622e7c6addd3625f4d07e0ac00` | Staged and fully read back; original promoter status retained as uncertain |
| Installed Settings executable | 277,608 | `b5a6e579536085215c1a6464bbdd41926242b9905fbf153e863a7dff35c5be16` | Complete installed-file readback, registered title and native foreground launch verified in TL05 |

**Observed interruption, TL04:** Settings promotion returned native result 0
with `state=running`, `phase=extracting`, `running=true` and `installed=null`.
The connection was lost at 15:34:40 UTC, followed by a no-route diagnostic.
An accepted/running operation is not a terminal installed receipt. At this
historical interruption no cause of the loss, successful reboot, new Settings
execution or in-game touch result was inferred. The saved operation was retained
for supported recovery rather than replaced merely because the connection was
absent. TL05 records the later independent installed-file and control proof; it
does not relabel the original promoter response as terminal success.

Private evidence locator: `vita-agent-use/agent/touch-lookup-20261011/`, with
`shell-upload.json`, `overlay-shell-installed.suprx`, `settings-upload.json`,
`settings-v079-readback.vpk`, `settings-install-request.jsonl` and
`promotion-events.jsonl`. These are private evidence locators; publication and
installed-executable proof are established separately in TL05/TL06. Do not copy
those raw files, transfer identities, configuration or network information into
the notebook.

**Verified Settings controls, TL05 (October 11, complete by 15:43:54 UTC):**
The user powered the Vita on after the disconnect. The old promoter status
remained uncertain and not running. Complete readback of the installed 277,608-byte executable
matched the TL03 hash; title `VJOSET001` was registered and a native foreground
launch was confirmed, with the snapshot reporting `system_ui_overlaid=false`.
These independently establish installation and execution without claiming a
terminal successful promoter receipt.

Four actual actions passed full configuration readbacks: touch **Character**,
Left to **Whole word**, Right to **Character**, then touch **Whole word**. Every
readback retained `ocr_backend=vocr`, Japanese-only ON and all unrelated bytes.
The final whole-word configuration is 2,973 bytes, SHA-256
`162280dcbf85bd43b7152d4d8d01899fc77112ff83494ac95c2a224a474c37be`.
Private receipt: `touch-lookup-20261011/settings-controls-verification.json`,
SHA-256 `ab88dcfca340ccf0df048d6e8e86527feeb9cc29dd8fab35ccffa6bf046c9743`.
The first Character action was at 15:41:20 UTC; a clock check immediately after
all four actions bounds completion to no later than 15:43:54 UTC. The aggregate
receipt has no own timestamp. Settings-card taps are separate from recognized
header taps in a game. Physical font readability, fresh Shell loaded-module
proof was subsequently obtained; game-touch behavior remains pending.

**Verified publication, TL06:** [stable v0.7.9](https://github.com/bee-san/vita-jp-overlay/releases/tag/v0.7.9)
was published at 15:44:16 UTC, targeting merge
`dc0bea53a6a8d075d4598cea27708953ed467666`. A fresh independent download audit
verified all six assets against the candidate bytes and GitHub asset digests,
plus all five `SHA256SUMS` entries. The main ZIP has 61 unique members and 60
manifested payloads; payload sizes, SHA-256 values and CRCs pass. The six-member
Settings VPK and 44-member corresponding-source ZIP also pass their checks.

All 45 packaged source files match overlay head
`ee796f1ea683d84a668bbc52961bf146a9f9986c`; all 35 Git-origin OCR source files
match `39a3a9ae069180675c145f6df74c843dc2061967`. Model pins, notices, JMdict and
public configuration pass; private Daijirin and device artifacts are absent.
Private receipt: `published-v079-audit/verification.json`, SHA-256
`797d9071312c67126ef2f638a9784b969617be3876e5e914e65fe2c000214be4`.
Downloaded package proof is separate from fresh Shell loading and game touch.

## Separate native sorting candidate

**Implemented locally, SORT01:** the private sorter uses a SHA-bound complete
59-icon plan and retains existing page records. It calls native `sceLsdbMoveIcons`
through the process's existing AppDB connection; its independent SQL queries
are read-only. It does not edit `app.db`, replace the database, delete apps or
create pages. The parser bounds the plan to 100 icons, validates unique title,
source and target coordinates, restricts coordinates to 0–9 and requires all
normal-page icons to match the reviewed complete source or complete target.
Current native page count is checked before moving.

The protocol requires authenticated CONTROL, explicit `yes=true`, no pending
approval/content work or active run, and a healthy physical stop guard. The
native coordinator additionally checks the live session/subject and an
owner-approved write ACL scoped to `ur0:shell`; ordinary writes to protected
`app.db` remain forbidden. The coordinator repeats stop/ACL and foreground
checks before the native effect. Its final full layout check rejects a changed
source, or skips the move if the complete target is already present.

The private journal binds the exact operation ID to the authenticated subject
and plan digest before the effect. A completed target can be reconciled without
another move; a completed journal paired with the old source is rejected. A
partial/unexpected post-move state is reported as a failure rather than being
labelled complete. The final precheck reduces the source-coordinate race but
is not a native transaction spanning the precheck and move.

**Measured host coordinator, SORT02:** the actual coordinator, parser, SHA and
title validation passed 12 private Clang ASan/UBSan cases with bounded SDK,
read-only SQLite and in-memory journal substitutes. Cases include normal move,
target replay, source→target reconciliation, changed-layout rejection,
completed-journal/source rejection, subject/digest binding, final ACL revocation,
native failure and partial/failed readback. Both pre-effect layout interleavings
fail a private copy without the final recheck and pass the repair. File and
wrapper ownership balance in every passing case; no sanitizer report occurred.

The reviewed coordinator SHA-256 is
`6864680ee75d3b5cc26f6ab9651b63b065ff1f708ccfdbfe246912b16e637c70`.
Private receipt: `vita-agent-use-livearea-sort/agent/tests/sort-coordinator-verification.json`,
SHA-256 `b12d1673f1979e0900621c2820c43c90c24ef017a3b41fd1a2c59d46723d9da5`.
The SDK and database calls are mocked, and this source is a local candidate.
Actual native moves, persisted hardware layout and physical page organization
remain unconfirmed. See [[25-Vita-debugging-tools]] and
[[09-Device-operations-and-recovery]] for the evidence and recovery rules.

**Clean native candidate, SORT03:** local revision
`d5b9221602aca0d3242de5bf003fa7b50c84348e` produced a 315,544-byte runtime,
SHA-256 `81d7235f4a5d6dba9eee33099fe436d42cb8d27d4d59760712dcd2f06f961245`.
Complete native readback matched those bytes; an ordinary reboot was accepted
at 15:45 UTC. Fresh runtime activation remains unconfirmed: native command
connections were refused, the UDP diagnostic timed out and a later no-route
diagnostic followed. At 15:53:04 UTC, activation remained unconfirmed. The saved
pending subscription was preserved. These symptoms do not prove a boot
cause or successful loading. Recovery/log inspection and any narrowly approved
rollback were pending at that earlier checkpoint. The recovery and revised
runtime below supersede that activation status. This is a local helper, not
an upstream agent release.

The build audit caught stale callback offsets after a fast CMake path bypassed
header-dependency refresh. Removing all 110 objects and rebuilding restored
byte-identical transport, UI, keep-awake and Shell objects. The final import audit
retained the prior 143 imports and added five documented SDK imports. A successful
link alone was insufficient: callback-layout and object-identity checks caught
the stale build before deployment.

## Recovery and revised native boot

The fresh loader tail identified the first candidate's failure as
`-2147299325`, or `0x8002D003`. The bundled `psp2_err` decoded it as
`SCE_KERNEL_ERROR_MODULEMGR_NO_LIB`. The code identifies a library-binding
failure; it does not name the library. SceLsdb was the only newly imported
library, making it the leading compatibility suspect rather than a proven
firmware export diagnosis. Do not attribute this result to sleep or OCR.

The user explicitly authorized a narrow FTP recovery. Only the loader log and
agent runtime were read, and the previous runtime was restored through a
separate staging file, full readbacks and a preserved failed-trial backup.
The restored 311,356-byte file matched
`7b803d324963257fcdc0d88d7f141ea8ae3c753b7520ccd759253956341de5c4`.
No tai configuration, database, identity or ACL files were changed. A subsequent
16:09:33 log was byte-identical to the failed-boot log; this did not establish a
new failure of the restored runtime.

At about 16:22 the host `serve` process was found absent despite its socket
remaining. Restarting ordinary `serve` with the saved identity and state
recovered the native connection. Fresh snapshots then confirmed the overlay
Shell/kernel and agent kernel/loader loaded. Check process and connection
health together; do not delete trust, counters or pending state to repair a
stale socket.

Local runtime revision `90c494a9fe3622523262629e39751b486fe3f093` resolves exactly
four SceLsdb exports per guarded command, before journal creation, rather than
adding a mandatory startup import. An unavailable or partial resolution fails
without an icon effect. It retains Shell's shared provider and database; it
neither loads/unloads the provider nor opens/closes/rebuilds Shell's database.
The journal uses the existing full-width seek wrapper. An independent binary
check matched all 31 prior library records, 143 function and three variable
imports, and verified unchanged transport, approval UI, runtime, keep-awake and
diagnostic objects. The actual coordinator passed 41 ASan/UBSan cases,
including missing/null exports, fresh retry, guards and recovery.

The revised 315,584-byte runtime has SHA-256
`10ad086ffeb5cc2382ab183e2051f2eae88e1b5b30ff50c7e92b667ffaa468bd`.
Its native upload completed with code zero, verified/complete flags and an
identical full readback. After an accepted ordinary native reboot, the saved
session renewed and fresh capabilities advertised `livearea.sort` among 62
operations. This establishes actual runtime activation. Companion local client
revision `ce74602aaf00798668f9587e269a22683a237a00` validates only the exact known
operation, a 32-digit ID, SHA-bound plan path and literal `yes=true`; 17 actual
validator cases passed on each of two clients. Existing long-hash behavior,
request recovery and security checks were retained. These local tool changes
were not published upstream.

A fresh complete inventory still contained 59 bubbles, nine page records
(including the special and empty tail pages) and zero orphans. All source
coordinates matched the 1,904-byte plan, SHA-256
`81b343f422a7772f5bc87da97217bd2949f5b291257ca04c1271766c4f21b6b1`.
The uploaded plan has an identical full readback. Genre placement remains the
same: visual novels, Japanese games, other games, then tools.

No grant or bubble move is established. The scoped access request returned
agent `VAU_BUSY` (`-2`) while system UI covered Settings; status showed no active
request. A guarded sorting call returned `VAU_DENIED` (`-4`) without the ACL.
These small negative codes are defined by the agent header, not Sony error
codes. Extra SQLite column diagnostics in that response are stale metadata,
not evidence of a dictionary/memory failure. A JPEG capture returned
`frame_unavailable`, `jpeg_encode`, code `-7`; it supplied no usable image.
Settings launch is confirmed, but the app must be made physically visible
before the approval dialog can appear. The agent cannot inject PS or approve
that dialog. No input is sent while a physical approval is pending.

Private evidence remains under `vita-agent-use/agent/touch-lookup-20261011/`
and `vita-agent-use-livearea-sort/agent/lazy-lsdb-review/`. Keep captures,
firmware, identity data and private dictionaries out of public artifacts.

## Remaining checks

- Retain the original uncertain promoter status alongside the independently
  verified installed executable and actual Settings controls.
- Retain the fresh native Shell-loaded proof alongside installed byte readbacks.
- Verify both highlight choices and Japanese-only header taps in a game,
  including held-touch/release and region-editor ownership.
- Inspect physical typography separately from successful saves or snapshots.
- Obtain physical scoped approval, then sort only through the supported guarded
  operation and verify
  complete persisted layout and physical page order.
- Retain pending native Daijirin lookup and hardware OCR CER; earlier host and
  game runs do not establish these new-release checks.
