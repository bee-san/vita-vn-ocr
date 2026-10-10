---
title: OCR anchor memory measurements
created: 2026-10-10
updated: 2026-10-10
status: host-live-verified-device-unverified
tags: [vita, ocr, lens, memory, measurements]
---

# OCR anchor memory measurements

The OCR-first flow needs a screenshot reference under CLANNAD's tight shell-memory conditions. The user explicitly accepted Google Lens to reduce local memory demand. Lens is the existing configured/default backend; no local OCR model is being deployed here.

**Current evidence:** production Lens OCR on the host recognized one previously verified short dialogue line exactly from both its full screenshot and saved crop. Both completed in a 384 KiB arena. Physical Lens completion and correct dialogue matching remain unverified. Later logs do confirm the 384 KiB Paf anchor allocation succeeds before the separate kernel capture allocation fails.

## Allocation policy

The anchor job uses one temporary **384 KiB (393,216 B)** result arena, first through USER_RW and then an equal-sized Paf fallback if the memblock fails. No dictionary lookup runs in this job. Result slot 0 owns the arena; slot 1 is NULL/0. After the network job finishes and the reference is copied into bounded resident storage, the temporary allocation is released. The earlier text-ready callback is too soon to free it.

The former worker requested two 384 KiB arenas in one 768 KiB USER_RW allocation even for the first screenshot. Its existing 256 KiB Paf fallback applied only after a native source was selected. Selected native text still uses its separate dictionary path. Arena capacity is not the complete resident cost of the plugin, capture buffers or operating system.

The raw capture separately requests **2,088,960 B (2,040 KiB) KERNEL_RW** for 960×544×4 pixels. The earlier shell USER_RW failure did not prove this different allocation failed. The later physical OCR test now independently confirms this KERNEL_RW allocation fails with `0x80024302`; see [[19-Physical-OCR-source-failure]]. Keep these separate failures and allocations distinct.

## JPEG and TLS measurements

Production `vjo-jpegsw`, quality 80, was measured on two private 960×544 images:

| Input | JPEG bytes | Encoder arena peak |
|---|---:|---:|
| Page 1 full image | 13,980 | 81,552 B |
| Page 4 full image | 21,834 | 97,936 B |
| Page 1 lower 192 pixels | 8,420 | 81,552 B |
| Page 4 lower 192 pixels | 10,236 | 81,552 B |

The fixed 16-row encoder band depends on width, so shortening an image need not reduce that fixed workspace. The actual ARM compiler type-size probe gives `sizeof(VjoTls)=6,496 B` and `VJO_TLS_MIN_BUF=16,709 B`. Both are allocated inside the caller's arena by `core/net.c`; this plugin does not create a separate TLS or network-pool memblock. SceShell already owns SceNet, although native socket/resolver operations can still fail internally.

## Deterministic offline probe

`memory-probe.c` executes production JPEG, Lens request building, HTTP reception, protobuf parsing and text filtering. It reserves the exact ARM TLS state/buffer sizes instead of performing a TLS handshake. The input response is the existing 2,321 B `host/fixtures/vn_textbox/lens.pb`, **not** a live Lens result for these screenshots. Parser/reader structures retain host sizes.

| Case | Measured arena peak | Tested capacities/result |
|---|---:|---|
| Page 1 | 117,336 B | 128/192/256/384 KiB pass |
| Page 4 | 133,720 B | 192/256/384 KiB pass; 128 KiB fails during HTTP receive (`rc=-103`) |
| Valid unknown-field padding to 65,536 B response | 260,696 B | 384 KiB passes |
| Padding to 131,072 B response | 391,768 B | 384 KiB passes |
| Padding to the 262,144 B response limit | 391,769 B | 384 KiB passes, 1,447 B remaining |

Unknown-field padding stresses HTTP storage, not arbitrary parsed-word expansion. This does not establish a universal response bound. The 128 KiB failure had reached 129,624 B before failing; that partial peak is not the amount required to complete. Larger or more complex inputs must still return bounded errors.

From the discovery worktree, repeat a passing production-code fixture probe with:

```sh
build/clannad-offline/ocr-memory/memory-probe \
  build/clannad-offline/ocr-memory/ranked-page-4-full.rgba \
  host/fixtures/vn_textbox/lens.pb 544 384
```

Replace the final `384` with `128` to reproduce the receive OOM. Exact build/run commands and outputs are in the private measurement files listed below. Do not infer live OCR accuracy or console performance from this fixture probe.

## Live production Lens check on the host

At 11:23–11:24 UTC on 2026-10-10, `live-lens-probe.c` ran production JPEG, `vjo_overlay_ocr`, BearSSL, Lens request/parser and Japanese filtering against the actual endpoint. It used one 384 KiB arena and no dictionary job. POSIX sockets and private raw screenshot input replace Vita transport/capture. These were **two single host requests**, not repeated timing samples.

| Input | JPEG | HTTP / TLS / production result | Response | Arena peak | Host elapsed |
|---|---:|---|---:|---:|---:|
| Page 4 full 960×544 | 21,834 B | 200 / 0 / `rc=0` | 638 B | 131,128 B | 459.737 ms |
| Same page, saved 755×146 crop | 7,154 B | 200 / 0 / `rc=0` | 839 B | 93,432 B | 560.435 ms |

Both results exactly matched the entire previously screenshot-verified six-codepoint line (18 UTF-8 bytes). Production normalization leaves four Japanese codepoints; `vjo_text_similarity` returns 100% for each of the three verified sources `8102CA7F`, `810A2393` and `810A34EF`. These are similarity scores, not confidence probabilities. No dialogue is reproduced here.

The saved normalized region `(2457,47826,51541,17588)` became pixel rectangle `(34,396,755,146)` through the exact kernel integer crop formulas and even-x alignment, without resampling. Recognizing one short page twice is not a shared-benchmark CER result or reliable-accuracy claim for other scenes/games. Host BearSSL sizes also differ from ARM, so the live peak does not replace a console measurement.

## Capture source and lifecycle boundaries

Source inspection shows `kernel/display.c:display_hook` admits only primary-display/index-0 submissions from `g.game_pid`. `kernel/capture.c:capture_copy` copies from that game-owned pointer, before final display composition. The Paf picker belongs to SceShell and fails the game-PID filter. Opening it blocks controller input but does not suspend the game. A game that stops submitting frames reaches the existing 1.5-second capture timeout; no stale composited framebuffer fallback is used. This is a source-code finding, not a new physical capture validation.

Production-worker tests with Vita I/O substitutes cover Paf fallback, allocation failure, delayed completion, close/reopen, exit cancellation, stable choices and stale-source confirmation. The final PR #13 head `5826329` passes 10 host suites in 4.18 s, including 37 worker cases, plus local Vita release/ABI checks. The original host tests do not measure actual Paf availability. Later physical logs show the anchor obtaining 384 KiB from Paf on all six failed capture attempts; those requests still stop before JPEG or Lens. The shell is installed and normally booted, but its physical OCR test failed with source error -108. Paired logs confirm the full-frame kernel buffer allocation fails before JPEG or Lens; see [[19-Physical-OCR-source-failure]].

## Evidence locations and next measurement

All probe code and private artifacts are ignored under `/home/bee/Documents/src/vita-jp-overlay-discovery/build/clannad-offline/ocr-memory/`:

- `README.md`, `memory-probe.c` and `type-sizes.c` explain the offline scope and commands.
- `jpeg-measurements.json`, `arena-measurements.json` and `response-budget-measurements.json` hold exact deterministic measurements.
- `live-lens-probe.c` is the live helper; the README contains its complete build command.
- `live-page-4-full/measurement.json` and `live-page-4-user-region/measurement.json` contain exact live commands, timestamps and bounded status metadata.
- `comparison.json` beside each records equality and scores without dialogue; private `recognized.txt`, `lens.pb` and image inputs must not be copied into notes or public commits.

The notes author inspected the README and both live measurement/comparison pairs. Next verify a real Vita screenshot request: allocation path, completion/failure stage, displayed reference and stable matching choices. Then verify native updates avoid repeated screenshots and cancellation cannot apply a stale reference. Keep host peaks, source capacities and future physical measurements separately labelled.

Related: [[06-Low-memory-overlay]], [[17-OCR-first-physical-workflow]], [[08-Validation-and-measurements]], [[11-Sources-and-artifacts]].
