---
title: Kernel syscall stack budget
created: 2026-10-10
updated: 2026-10-10
status: fixed-static-measurement
tags: [vita, kernel, stack, memory]
---

# Kernel syscall stack budget

The native text submission path had another concrete risk: nested compiled frames exceeded taiHEN's documented 4 KiB syscall stack before allowing for OS overhead. This was found during the boot investigation but was not the firmware import failure itself.

## Measurement

GCC `-fstack-usage` on the Vita kernel reported:

| Frame or chain | Before | After |
|---|---:|---:|
| Text submit | 2,120 B | 64 B |
| Offer | 2,152 B | 2,152 B |
| Similarity | 568 B | 568 B |
| Nested sum | **4,840 B** | **2,784 B** |

The sum is a static compiled-code measurement. It excludes system call overhead and is **not a physical high-water measurement**. Do not relabel the remaining difference as a measured safe stack margin.

## Change

`kernel/text.c` reuses disjoint raw-byte and decoded-text portions of the existing 5,122 B scan scratch buffer while holding the text mutex. It avoids adding another resident buffer. Raw data must be copied/decoded before scratch reuse can invalidate it; the lock owns the shared scratch for the full operation.

Coverage includes maximum 1,024 B UTF-16 input, 256 Japanese codepoints, scoring against a reference, oversized-input rejection and persistence after scratch reuse. The host kernel policy suite had nine cases in the recorded firmware-fix run.

```sh
cmake -S vita -B build/vita-hooks-stack -G Ninja   -DVJO_WITH_NCNN=OFF -DCMAKE_C_FLAGS=-fstack-usage
cmake --build build/vita-hooks-stack --target VitaJPOverlay_Kernel -j4
```

Run from the relevant overlay worktree with its pinned VitaSDK on PATH. Preserve compiler version and optimization flags when comparing. Adding automatic arrays or deeper calls to this path requires a new compiled-frame audit.

Source commit `8486c97`; machine-readable measurements are in `docs/benchmarks/native-hooks-fw365-20261010.json` in the overlay repository. See [[11-Sources-and-artifacts]].
