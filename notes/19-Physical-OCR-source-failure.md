---
title: Physical OCR source failure
created: 2026-10-10
updated: 2026-10-10
status: kernel-capture-allocation-failure-verified
tags: [vita, ocr, capture, failure]
---

# Physical OCR source failure

After the OCR-first shell was installed and normally booted, the user pressed L+R in CLANNAD and reported:

> text recognition failed. unexpected response (-108). no game text found yet.

This is a failed physical OCR/matching test. Successful installation, normal boot and the two live Lens host requests remain valid within their scopes; they do not establish the Vita capture path. The device has shell `32baceeb...`, kernel `6cc2dc86...`, text `84259631...`, and auto/Lens configuration from [[02-Current-device-state]].

## Physical logs localize the failure

The downloaded status and kernel logs now identify the failing stage. Six attempts each show `capture request failed -1073741826` (`0xC0000002`), job `rc=-108` and arena peak 0. Six corresponding kernel entries show `alloc capture buffer (2040 KiB) failed 0x80024302`. The 2,088,960 B full-frame KERNEL_RW allocation fails before JPEG encoding or a Lens request. This is a physical capture allocation failure, not an unexpected Lens response.

The notes author independently inspected the narrow markers and hashed the local logs under `vita-agent-use/agent/overlay-ocr-picker-20261010/`:

| File | Bytes | SHA-256 |
|---|---:|---|
| `status.ocr-failed.txt` | 4,511 | `c6b006688880a53528fa629d456485d0420730477504acf4728e1c6cee237d03` |
| `kernel.ocr-failed.txt` | 527 | `18dc3510e020bd9a26e24bc0c1180a3bcd85b3d1f31d33213bc4db53fc6ff5c2` |

The source has a separate unresolved error-propagation issue: the expected kernel no-memory return is -2, but the shell receives `0xC0000002`. This may explain why the ordinary OOM UI mapping was bypassed; its mechanism is being reviewed and must not be guessed. The underlying allocation failure is independently established by the kernel log.

Root is preparing capture storage sized to the selected region instead of always allocating a full 960×544 frame. The saved 755×146 region contains 440,920 pixel bytes before allocation alignment. This is the intended size reduction, not a deployed fix or measured successful physical allocation. Ownership, capacity and region changes require validation before another install.

## What the code established before logs arrived

At merged source `539c2888` / reviewed head `5826329`, `core/conn.h` defines `VJO_E_SOURCE=-108` with the comment “JPEG source read failed.” That comment is broader in actual use: `shell/worker.c` maps several capture-stage failures to this value. The notes author inspected these call sites rather than inferring a cause from the numeric value alone.

| Path in `shell/worker.c` | Condition mapped to -108 | Evidence needed to identify it |
|---|---|---|
| `wait_capture` request | `vjoRequestCapture` returns a negative non-memory error | `capture request failed` marker and original return value |
| `wait_capture` wait loop | Matching done sequence not received before timeout | `capture N timed out` marker and capture/event state |
| `wait_capture` completed result | Nonzero non-memory `capture_result` | `capture failed` marker and original kernel result |
| `vjo_capture_jpeg` | JPEG encoder fails without its OOM flag | Raw-row return/geometry and encoder-stage evidence |

The memory-specific request/result error is separately mapped to `VJO_E_OOM=-103`. This does not prove that every possible memory-related source failure is distinguishable from -108. The user report alone could not identify a branch; the later paired logs above identify the request-stage allocation failure.

`core/client.c` has no dedicated `VJO_E_SOURCE` presentation case, so it falls through to “unexpected response (%d).” The wording does not establish an unexpected HTTP response or Lens rejection. `core/client.c` can also produce a source error when reading its JPEG input; preserve the original stage and logs before choosing a cause.

## Region-sized repair under development

At published [overlay PR #14](https://github.com/bee-san/vita-jp-overlay/pull/14), head `5118c4a09248a3d8b7e8018d85fa0d0247b82ca4` on `fix/region-sized-ocr-capture` from merge `539c2888`, root implements 4 KiB-aligned allocation for the requested pixel region. The saved 755×146 crop needs **442,368 B (432 KiB)** after alignment, compared with the old 2,040 KiB frame allocation. This is code-derived requested capacity, not a successful physical allocation measurement.

The patch freezes requested framebuffer/crop geometry before allocation, checks raw capacity before copying, and publishes pending state atomically. A `VJO_CAPTURE_ONCE` flag distinguishes a single JPEG pass from local ncnn OCR's multiple row passes: the kernel frees the JPEG raw buffer after the final copied row, before Lens transmission, while preserving a multipass image. `VJO_CAPTURE_DISCARD` releases an idle single-pass image after encoder failure and refuses pending/multipass ownership. Both independent reviewers checked these lifetimes/regressions. Final local validation passes all 10 host suites in 4.61 s, including 41 actual worker cases and 16 production-kernel cases; kernel ASAN/UBSAN also passes. The ncnn-OFF SDK Release build and packaged ABI pass; the kernel has 6,055 decoded ARM instructions with zero FPU/NEON. Hosted host passed at exact head `5118c4a` (56 s), with Vita CI still running in run `38049340699` when checked. PR #14 is open, unmerged and uninstalled.

The user subsequently chose a fixed manual list, so this repair is an optional OCR path rather than a prerequisite for the current requested manual experiment. See [[20-Fixed-manual-hook-workflow]].

## Next evidence

Validate the region-sized allocation and bounded reuse before deployment. Preserve the current full-frame OOM baseline, fix the error reporting/propagation as evidence permits, then repeat the real screenshot request. Confirm actual kernel capture, JPEG completion, Lens entry/result and the stable match separately. The shell anchor's 384 KiB budget and the kernel raw buffer are different allocations; the host Lens result bypassed the failing physical kernel allocation.

Do not treat “no game text found” as proof that generic hooks never fire. Keep this capture failure separate from the earlier unanchored source-picker UX and boot ABI bug. Append corrective measurements and device outcomes here and in [[2026-10-10-Investigation-log]].

Related: [[17-OCR-first-physical-workflow]], [[18-OCR-anchor-memory-measurements]], [[08-Validation-and-measurements]], [[12-Open-questions-and-next-steps]].
