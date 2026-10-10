---
title: Validation and measurement boundaries
created: 2026-10-10
updated: 2026-10-10
status: partial-hardware-validation
tags: [vita, validation, measurements]
---

# Validation and measurement boundaries

Each validation layer answers a different question. A passing lower layer does not establish the higher one.

| Layer | Recorded result | Establishes | Does not establish |
|---|---|---|---|
| Host CTest, boot fix | 10 suites passed; 9 kernel policy cases | Portable policy, selected mapping/ownership behavior with SDK substitutes | Actual firmware bindings or commercial-game text |
| Vita release builds, boot fix | OCR OFF and ON passed | Compilation and linking under recorded SDK | Hardware boot |
| Packaged ABI/disassembly | Audit passed; 5,923 ARM instructions, zero FPU/NEON | Checked packaged imports/exports/relocations and forbidden-instruction absence | Every runtime path or memory mapping |
| Isolated Vita3K self-test | 249 checks, zero failures | 43 portable checks + 192 observer checks + 14 production-plugin checks | Physical kernel loading, SceShell/Paf behavior, commercial-title compatibility |
| Host CTest, low-memory fix | 10 suites; 30 worker cases; 1.59 s recorded run | Policy and bounded fallback ownership under host substitutes | Device Paf fallback success or console performance |
| SceLibc correction, host and isolated Vita3K | 10 host suites (2.14 s), 10 production-kernel cases, 255 emulator checks; zero emulator failures | Both C-library hook families, bounded strcpy, original returns, overlapping memmove, zero-length suppression and hook removal | Real CLANNAD dialogue or physical startup of this new revision |
| Installed binary readback | Full hashes matched | Intended bytes reached installed paths | Loaded modules or functional capture |
| Physical normal boot | API 8, registration 0, STARTED 6 | Corrected set initialized on this 3.65 device | Later game stability on all paths |
| Physical picker | User observed it open after low-memory fix | Previous eager-allocation gate removed in this scenario | Useful text capture |
| Physical CLANNAD capture | Not verified; picker empty | Current failure reproduced | A compatibility success |

## Reproduction commands

From the relevant overlay worktree:

```sh
cmake -S . -B build/host -G Ninja -DVJO_WITH_NCNN=OFF
cmake --build build/host -j4
ctest --test-dir build/host --output-on-failure
```

With the pinned VitaSDK and Paf tooling on PATH:

```sh
cmake -S vita -B build/vita-hooks-device -G Ninja   -DCMAKE_BUILD_TYPE=Release -DVJO_WITH_NCNN=OFF -DVJO_TEXT_TEST_APP=ON
cmake --build build/vita-hooks-device -j4
python3 tools/check_kernel_instructions.py   --objdump "$VITASDK/bin/arm-vita-eabi-objdump"   build/vita-hooks-device/VitaJPOverlay_Kernel
python3 tools/check_vita_syscall_abi.py   --kernel build/vita-hooks-device/VitaJPOverlay_Kernel.skprx   --shell build/vita-hooks-device/VitaJPOverlay_Shell.suprx --require-hooks
python3 tools/test_vita3k_hooks.py   --vpk build/vita-hooks-device/vjo-hook-test.vpk
```

Record the commit, compiler, configuration and exact command for each new result. The 249-check emulator run belongs to the recorded boot-fix source; do not imply it reran after every shell edit. Low-memory source had its own host/Vita validation.

## Important diagnostic limits

Vita-agent screenshot attempts failed with `frame_unavailable`, code `-7`, stage `jpeg_encode`. No successful image file from those attempts exists. User reports and metadata were therefore needed for visual behavior; a screenshot request alone is not image evidence.

A performance command returned a client error, while pushed native events contained roughly 59.992 FPS and partition free-memory fields (USER_RW 28,672 B free of 50 MiB; CDRAM 0 free of 16 MiB). Address-space ownership was not verified; these must not be reported as confirmed CLANNAD memory measurements. One sample is also not a latency distribution or stability benchmark.

No console peak-memory, capture-latency, scanner-duration, broad-title-coverage or real-screenshot CER number is established. Real text validation should keep hand-checked evaluation pages separate from pages used for source selection/calibration, and preserve test text privately. Static screenshots cannot alone validate a moving RAM source.

Related: [[01-Evidence-and-claims]], [[Experiment-template]], [[2026-10-10-Investigation-log]].
