---
title: Retail SceLibc hook coverage correction
created: 2026-10-10
updated: 2026-10-10
status: draft-locally-validated-not-installed
tags: [vita, native-text, libc, validation]
---

# Retail SceLibc hook coverage correction

The first generic text plugin observed SceLibKernel clib functions. Offline CLANNAD inspection found none of those four imports and found equivalent functions in SceLibc. The fix supports **both** families instead of treating this as proof that the game needs an engine-specific profile.

Initial import-fix source: `0a66015994dbb3f12a3a506f0c0e4cab8ec72fdd`, version `0.6-hooks.5`. The later ranking repair is implementation `6eef6c0`; final evidence head `69006af9e36347b75534c01900b48d43ed372981` merged in [PR #12](https://github.com/bee-san/vita-jp-overlay/pull/12) as `a5e0c50b1096b5ae7de450ae712e63b272dc7f17`.

**The final kernel/text pair is installed, fully read back and normally booted on the physical Vita.** Actual CLANNAD selection/capture there remains pending. The emulator proof matches two pages through the same immediate sources without a title profile, and the full-feed ranking repair puts all three matching sources in the displayed five on both pages. See [[02-Current-device-state]] and [[16-Full-feed-picker-ranking]].

## Capture behavior

Four additional import hooks cover SceLibc memcpy, strcpy, strncpy and memmove, with separate taiHEN references because both families can coexist in a module. Each calls the original function and preserves its return value before submitting a bounded observation of the copied destination.

`strcpy` has no length argument. Its hook does not call an unsafe user-side `strlen`; it asks the kernel to copy a bounded NUL-terminated string while holding the allocation mapping. Automatic UTF-8/CP932 decoding is accepted for call events. Automatic encoding for register events remains rejected. Other copy hooks retain explicit length bounds and ignore zero-length operations.

This is a kernel-and-text-plugin change. The shell artifact hash is unchanged. Match the tested pair if installing; do not assume a newer text plugin's bounded submission semantics are covered by the old installed kernel.

## Validation at the initial import-fix head

| Check | Result |
|---|---|
| Host CTest | 10 suites passed, recorded elapsed 2.14 s |
| Production kernel cases | 10 passed, including CP932 at allocation end, absent NUL, ASCII and rejection of automatic-encoding register events |
| Vita release build | Passed, OCR OFF, text test app ON |
| Packaged syscall ABI | Passed |
| Hosted exact-head CI | Host and Vita both SUCCESS on run `38045205714`, completed 2026-10-10 10:36:03 UTC; PR was draft at that stage |
| Isolated Vita3K | **255 checks, zero failures**: 43 portable + 192 observer + 20 production-plugin |
| Synthetic production plugin events | 8 copy/string call events, 2 register-profile events, 10 matching sources |

The emulator harness uses actual retail alias import stubs. It checks both C libraries, original return values, overlapping memmove, ignored zero-length copies and restoration after unload. These are synthetic app sources; the “10 matching sources” count is not a CLANNAD result.

Private result: `/home/bee/Documents/src/vita-jp-overlay-discovery/build/vita3k-libc-test/pref/ux0/data/VitaJPOverlay-hook-test/results.txt`.

```sh
python3 tools/test_vita3k_hooks.py --vpk build/vita/vjo-hook-test.vpk \
  --workdir build/vita3k-libc-test
```

## Independent read-only review

A separate reviewer found no blockers at this exact head. Confirmed checks include the explicit zero-length copy gate, bounded strcpy handling, allocation-end/NUL behavior and correct mapping release. Compiled submit/offer/similarity frames still total 2,784 B excluding OS overhead; the new kernel has 5,922 decoded ARM instructions and zero FPU/NEON.

The review calls out an important test boundary: the Vita3K fake transport uses a strlen substitute and does not itself prove production kernel bounds. Separate host tests of the actual kernel path provide that coverage. Real-game capture and extra syscall overhead remain unmeasured. A real CLANNAD startup memmove was observed, but with zero bytes; it is not text evidence. See [[15-Vita3K-runtime-investigation]].

## Built artifacts at the original import-fix commit

The following hashes belong to implementation `0a660159...` before the ranking correction. Current candidate `6eef6c0` changes the kernel to SHA-256 `6cc2dc868dc4403bcab252d689acd2f0779a55f9bd8069654b335eaffa571933` (38,620 B), with the same shell/text bytes. See [[16-Full-feed-picker-ranking]] for its 264-check validation and full-feed rerun status. The original pre-ranking kernel was not installed; the final `6cc2dc86...` kernel is now installed and normally booted.


| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Kernel | 38,604 | `97d4901e8de5e58bcd9c1b2c926e7fb3fa7532b9d74bd93997203cc6ee90a1c7` |
| Shell | 156,196 | `45f86b1ea83c08bdf61c1dfd174cc5356151e746a3659a0299218436a8f10c8c` |
| Text | 7,936 | `84259631e0efe87472750a04a944a0d2cb7673f218c85f316ae3a450e6c7565f` |

See [[02-Current-device-state]] for the separately verified installed set. Further code changes require new head/artifact records; do not overwrite these historical hashes.

## Remaining decisions

Physical startup now passes. Actual game capture/selection, runtime overhead, broader-scene dialogue continuity and whether later scenes need a title profile remain unresolved. Two visible CLANNAD pages now match the same immediate generic source in the emulator. The full-feed experiment exposed and verified the correction of an initial-picker ranking defect; [[16-Full-feed-picker-ranking]] records the preserved failing baseline and successful new snapshot ranks. Review the exact head before deployment, then establish actual attachment and two advancing visible pages. See [[12-Open-questions-and-next-steps]].
