---
title: Retail SceLibc hook coverage correction
created: 2026-10-10
updated: 2026-10-10
status: draft-locally-validated-not-installed
tags: [vita, native-text, libc, validation]
---

# Retail SceLibc hook coverage correction

The first generic text plugin observed SceLibKernel clib functions. Offline CLANNAD inspection found none of those four imports and found equivalent functions in SceLibc. The fix supports **both** families instead of treating this as proof that the game needs an engine-specific profile.

Source commit: `0a66015994dbb3f12a3a506f0c0e4cab8ec72fdd`, branch `fix/native-libc-capture`, version `0.6-hooks.5`. [Draft PR #12](https://github.com/bee-san/vita-jp-overlay/pull/12) is published for review. **This revision is not installed on the physical Vita and real CLANNAD dialogue remains unverified.**

## Capture behavior

Four additional import hooks cover SceLibc memcpy, strcpy, strncpy and memmove, with separate taiHEN references because both families can coexist in a module. Each calls the original function and preserves its return value before submitting a bounded observation of the copied destination.

`strcpy` has no length argument. Its hook does not call an unsafe user-side `strlen`; it asks the kernel to copy a bounded NUL-terminated string while holding the allocation mapping. Automatic UTF-8/CP932 decoding is accepted for call events. Automatic encoding for register events remains rejected. Other copy hooks retain explicit length bounds and ignore zero-length operations.

This is a kernel-and-text-plugin change. The shell artifact hash is unchanged. Match the tested pair if installing; do not assume a newer text plugin's bounded submission semantics are covered by the old installed kernel.

## Validation at this head

| Check | Result |
|---|---|
| Host CTest | 10 suites passed, recorded elapsed 2.14 s |
| Production kernel cases | 10 passed, including CP932 at allocation end, absent NUL, ASCII and rejection of automatic-encoding register events |
| Vita release build | Passed, OCR OFF, text test app ON |
| Packaged syscall ABI | Passed |
| Isolated Vita3K | **255 checks, zero failures**: 43 portable + 192 observer + 20 production-plugin |
| Synthetic production plugin events | 8 copy/string call events, 2 register-profile events, 10 matching sources |

The emulator harness uses actual retail alias import stubs. It checks both C libraries, original return values, overlapping memmove, ignored zero-length copies and restoration after unload. These are synthetic app sources; the “10 matching sources” count is not a CLANNAD result.

Private result: `/home/bee/Documents/src/vita-jp-overlay-discovery/build/vita3k-libc-test/pref/ux0/data/VitaJPOverlay-hook-test/results.txt`.

```sh
python3 tools/test_vita3k_hooks.py --vpk build/vita/vjo-hook-test.vpk \
  --workdir build/vita3k-libc-test
```

## Built artifacts, not installed artifacts

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Kernel | 38,604 | `97d4901e8de5e58bcd9c1b2c926e7fb3fa7532b9d74bd93997203cc6ee90a1c7` |
| Shell | 156,196 | `45f86b1ea83c08bdf61c1dfd174cc5356151e746a3659a0299218436a8f10c8c` |
| Text | 7,936 | `84259631e0efe87472750a04a944a0d2cb7673f218c85f316ae3a450e6c7565f` |

See [[02-Current-device-state]] for the separately verified installed set. Further code changes require new head/artifact records; do not overwrite these historical hashes.

## Remaining decisions

Physical startup/capture, runtime overhead, real dialogue continuity and whether a title profile is necessary are unresolved. A separate real-CLANNAD runtime experiment is in progress. Review the exact head before deployment, then establish actual attachment and two advancing visible pages. See [[12-Open-questions-and-next-steps]].
