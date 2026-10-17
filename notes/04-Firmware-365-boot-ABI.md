---
title: Firmware 3.65 boot ABI failure
created: 2026-10-10
updated: 2026-10-10
status: fixed-and-device-verified
tags: [vita, kernel, abi, recovery]
---

# Firmware 3.65 boot ABI failure

The repeated boot shutdown was traced to an unresolved custom kernel API call after the overlay kernel failed to start. The physical device's actual firmware is 3.65. Its spoofed version did not determine which kernel-library NIDs were available.

## Observed chain

1. The overlay kernel existed in native module inventory, but state was **READY (2)**, not STARTED.
2. The shell treated module presence as enough to call the custom API.
3. The last marker was `worker startup: checking kernel API`.
4. The `vjoGetVersion` runtime import contained the unresolved trap words below and branched to address zero.

```text
E24FC008 E3A0F000 E1A00000 00000000
```

The unresolved state followed a strong import of a firmware-specific memory-type function:

| ABI | Library | Library NID | Function | Function NID |
|---|---|---|---|---|
| Built for 3.60 | `SceSysmemForKernel` | `0x63A519E5` | `ksceKernelGetMemBlockType` | `0x289BE3EC` |
| Available on 3.63+ family, including this 3.65 device | `SceSysmemForKernel` | `0x02451F0F` | `ksceKernelGetMemBlockType` | `0xD44FE44B` |
| Replacement | `SceSysmemForDriver` | Stable driver import | `ksceKernelMemBlockGetInfoEx` | `0x24A99FFF` |

The replacement uses the exact 184 B (`0xB8`) structure, sets its size, zeroes it and reads `core_info.type`. The kernel no longer links the firmware-specific ForKernel stub.

## Defenses added

- `include/vjo_import.h` checks custom import stubs before calls. Both user plugins fail closed if the kernel binding is missing.
- Shell guards cover all 14 custom API calls; the text plugin guards version/submission calls.
- Known unresolved-return patterns, the fatal zero branch and untouched weak stubs are rejected.
- `tools/check_vita_syscall_abi.py` audits packaged SELF imports, exports and relocation targets; it rejects strong firmware-specific `SceSysmemForKernel` imports.
- Kernel code is built with `-mgeneral-regs-only` and its disassembly is checked for FPU/NEON instructions.

The earlier FPU-only repair removed 85 FPU/NEON instructions but **did not fix the 3.65 import problem**. The later checked kernel has 5,923 decoded ARM instructions and zero FPU/NEON instructions. Keep the two findings separate.

## Verified outcome

The physical Vita booted normally; kernel API returned 8, shell registration returned 0, and startup reached `bootstrap: worker started`. Kernel and shell were STARTED (6), as was the text plugin after launching CLANNAD. Installed binary hashes and readback are in [[02-Current-device-state]]. CLANNAD then exposed a separate result-arena allocation failure, covered in [[06-Low-memory-overlay]].

Fix source commit: `8486c97`; boot evidence: `8e7a6e8`; merged PR: [overlay #10](https://github.com/bee-san/Rubi_for_vita/pull/10), merge `bfb1ada4f3f6995100cb7fa04d4b8240c868ddcf`.

## CI header mismatch

The August 2026 VitaSDK image lacked `SceKernelMemBlockInfoEx` and the new driver stub used by the implementation. Local builds passed because the local SDK was newer. CI failed at compile time, not on the device.

`tools/setup_toolchain.sh` now pins official `vita-headers` revision `e66ebe90b73d1fa4cce005a5b2072cec27322544`, installs its headers and builds the Sysmem stubs from `db/360/SceSysmem.yml`. Pin commit: `c7eed5066fe2550ddb17748929db620951209418`. A separate local validation prefix successfully built the kernel with those headers/stubs. No claim is made that the old container was run locally; Docker socket access was unavailable. The resulting local validation kernel hash differed from the installed artifact and was not deployed merely because it built.

Related: [[05-Kernel-syscall-stack]], [[08-Validation-and-measurements]], [[09-Device-operations-and-recovery]].
