---
title: Sources and private artifact register
created: 2026-10-10
updated: 2026-10-10
status: active
tags: [vita, sources, provenance]
---

# Sources and private artifact register

This is an evidence locator. Public references can be shared; private artifacts must stay local. Existence of a filename is not enough: each claim should identify what was actually inspected and its limitations. No credentials, game bytes or raw dialogue are copied here.

## Repository evidence inspected for these notes

| ID | Source | Use |
|---|---|---|
| S01 | [Firmware benchmark at pinned overlay commit](https://github.com/bee-san/vita-jp-overlay/blob/c7eed5066fe2550ddb17748929db620951209418/docs/benchmarks/native-hooks-fw365-20261010.json) | Runtime binding, physical boot, installed hashes, host/Vita/Vita3K results, stack frames. |
| S02 | [Native-text architecture at the same commit](https://github.com/bee-san/vita-jp-overlay/blob/c7eed5066fe2550ddb17748929db620951209418/docs/native-text.md) | Modes, source selection, limits, profiles and validation boundaries. |
| S03 | [Low-memory benchmark at merged revision](https://github.com/bee-san/vita-jp-overlay/blob/13bcf816ab4e45fb8f59da0f52407a621b07284e/docs/benchmarks/native-text-low-memory-20261010.json) | Host case count, changed allocation sizes, picker success and remaining empty-source failure. |
| S04 | [Toolchain pin](https://github.com/bee-san/vita-jp-overlay/commit/c7eed5066fe2550ddb17748929db620951209418) | Headers/stubs needed by CI. |
| S05 | [Initial source-only research PR](https://github.com/bee-san/vita-vn-ocr/pull/15) | Feasibility, generic-vs-engine-specific distinction, PSP separation. Local `hooker/RESEARCH.md` is earlier research, not a current experiment log. |
| S06 | Local `vita-agent-use/AGENTS.md` and `docs/agent-reference/` | Actual client invocation, recovery and concurrency rules. Read fresh before operations. |
| S14 | [Bounded picker diagnostics PR #15](https://github.com/bee-san/vita-jp-overlay/pull/15) and [exact head](https://github.com/bee-san/vita-jp-overlay/commit/f58b1aa49dd43fcc3d6d78b0c9bcd48a1c0ca6d7) | Explicit-action diagnostic bounds, source review and local/hosted validation; merged, prepared, not installed. |
| S13 | [Region-sized capture PR #14](https://github.com/bee-san/vita-jp-overlay/pull/14) and [exact implementation](https://github.com/bee-san/vita-jp-overlay/commit/5118c4a09248a3d8b7e8018d85fa0d0247b82ca4) | Region-sized capacity, frozen geometry, JPEG single-pass release/discard, multipass preservation and regression tests; uninstalled. |
| S12 | [OCR-first PR #13](https://github.com/bee-san/vita-jp-overlay/pull/13) and [implementation](https://github.com/bee-san/vita-jp-overlay/commit/4205be7034a2b48df45e65c7f253e7c7524b349a) | Worker/UI/config changes, lifecycle tests, host allocation probes and unverified physical boundaries. |
| S11 | [Final full-feed evidence at exact review head](https://github.com/bee-san/vita-jp-overlay/blob/69006af9e36347b75534c01900b48d43ed372981/docs/benchmarks/native-libc-hooks-20261010.json) | Complete two-page snapshot ranks and private artifact hashes; root independently checked both screenshots and all six evidence hashes. |
| S10 | [Ranking correction at exact head](https://github.com/bee-san/vita-jp-overlay/commit/6eef6c031aa1cc4bc162ebf31b8628d3098f0f5e) | Comparator fix, discriminating synthetic fixture, 264-check validation and new kernel hash. |
| S09 | [Two-page emulator evidence follow-up](https://github.com/bee-san/vita-jp-overlay/commit/d1ee0dc) | Same immediate source matches two visible pages; records helper prefilter limitation and private evidence hashes. |
| S08 | [Retail-libc benchmark at exact draft head](https://github.com/bee-san/vita-jp-overlay/blob/0a66015994dbb3f12a3a506f0c0e4cab8ec72fdd/docs/benchmarks/native-libc-hooks-20261010.json) | Corrected import coverage, host/emulator validation, built artifact hashes and explicit hardware limits. |
| S07 | Active-investigator updates in the 2026-10-10 conversation | Later text-plugin rollback, user screen reports, import-table and font-codepoint findings. Separately marked where direct artifact inspection was not repeated. |

## Primary external references

These are references recorded by the investigation; the notebook does not imply every moving upstream page was revalidated while writing it.

- [VitaSDK 3.60 SceSysmem NIDs, pinned headers revision](https://github.com/vitasdk/vita-headers/blob/e66ebe90b73d1fa4cce005a5b2072cec27322544/db/360/SceSysmem.yml) and [3.63 SceSysmem NIDs](https://github.com/vitasdk/vita-headers/blob/e66ebe90b73d1fa4cce005a5b2072cec27322544/db/363/SceSysmem.yml): firmware import comparison.
- [taiHEN usage](https://github.com/yifanlu/taiHEN/blob/master/USAGE.md): native hooking and syscall stack context.
- [VitaSDK kernel module management](https://docs.vitasdk.org/group__SceModulemgrKernel.html): module-state interpretation.
- [VitaSDK Paf user APIs](https://docs.vitasdk.org/group__ScePafUser.html): Paf allocator interface.
- [Pinned Luna Vita hook table](https://github.com/HIllya51/LunaTranslator/blob/b0c0c2d2a03b330c388b42d7e8f0556b6d230980/src/NativeImpl/LunaHook/LunaHook/emulators/vita3k_1.cpp): examples of title-specific emulator descriptions; not drop-in native profiles.
- [PSVshellPlus memory information source](https://github.com/GrapheneCt/PSVshellPlus/blob/main/PSVshellPlus_Kernel/info.c): reason to verify process/address-space ownership before attributing memory metrics.

## Local private evidence

Artifact paths are intentionally local and may be machine-specific. Read only the narrow file required. Emulator logs and game directories can contain license/decryption material; never dump them wholesale into tool output or a commit.

| Local directory or file | What it contains | Status/limits |
|---|---|---|
| `/home/bee/Documents/src/ps-vita-ocr-lab/vita-agent-use/agent/overlay-firmware-fix-20261010/` | Plugin backups/readbacks, installation report, `status-after-boot.txt`, postboot/game plugin inventories, initial memory-failure log | Physical boot evidence; no successful commercial-text capture. |
| `/home/bee/Documents/src/ps-vita-ocr-lab/vita-agent-use/agent/overlay-low-memory-20261010/` | Shell backup/readback, upload state, `status-after-boot.txt`, `status-clannad-01.txt`, `status-clannad-02.txt` | Low-memory deployment and empty-picker metadata. Do not publish transfer-state JSON. |
| `/home/bee/Documents/src/ps-vita-ocr-lab/vita-agent-use/agent/overlay-region-capture-20261010/` | Prepared combined PR #14+#15 kernel/shell/text artifacts | Independently hashed local build outputs; not device readbacks and not installed. Current sizes/hashes in [[21-Picker-diagnostics-and-screenshot-provenance]]; prior PR #14-only shell is retained historically. |
| `/home/bee/Documents/src/ps-vita-ocr-lab/vita-agent-use/agent/overlay-ocr-picker-20261010/status.manual-start.txt` | Current manual selection and Paf dictionary-job metadata; manual config readback beside it | 6,092 B, SHA `73032e86...`; user suspects selected text is wrong. Exact hashes and limits in [[20-Fixed-manual-hook-workflow]]. |
| `/home/bee/Documents/src/ps-vita-ocr-lab/vita-agent-use/agent/overlay-ocr-picker-20261010/` | PR #13 shell/config/region backups, readbacks, installation report, `status.after-reboot.txt`, `status.clannad-start.txt` | Shell/config hashes independently checked; normal boot and foreground CLANNAD launch reported. Later `status.ocr-failed.txt`/`kernel.ocr-failed.txt` localize six OCR failures to full-frame kernel buffer allocation before Lens. Do not publish private config/transfer state. |
| `/home/bee/Documents/src/ps-vita-ocr-lab/vita-agent-use/agent/overlay-libc-20261010/` | Preinstall backups, guarded upload journals, full readbacks, installation report and `status-after-reboot.txt` | Installation/readback/normal boot verified; later physical user test failed UX. Do not publish private operation state. |
| `/home/bee/Documents/src/ps-vita-ocr-lab/vita-agent-use/agent/overlay-libc-20261010/status-clannad-user-spam.txt` | Physical candidate-count/discovery metadata after the user reported changing 0% hooks | 2,123 B, SHA-256 `5e5ee8dcc624af6b18fcb7f892f4b0ac8f215b57c2a30d87dfe8af547858e189`; selected=0, no correct dialogue proof. |
| `/home/bee/Documents/src/vita-jp-overlay-discovery/build/clannad-offline/ocr-memory/` | `README.md`, `memory-probe.c`, `type-sizes.c`, JPEG/arena/response-budget JSON measurements | Host production-code probe with fixture response and target TLS reservations; no live handshake. Private image/input artifacts remain ignored. See [[18-OCR-anchor-memory-measurements]]. |
| `/home/bee/Documents/src/vita-jp-overlay-discovery/build/clannad-offline/research-metadata.json` | Loaded-image hash, boundary and 250-import count | Small metadata inspected locally; original path fields are private and not blindly copied. |
| `/home/bee/Documents/src/vita-jp-overlay-discovery/build/clannad-offline/imports.tsv` and `imports.json` | Main-module imported library/NID/stub records | Derived metadata; this notebook includes only the four relevant rows. |
| `/home/bee/Documents/src/vita-jp-overlay-discovery/build/clannad-offline/eboot.loaded.elf` | Loaded/relocated decrypted game segments, SHA-256 recorded in [[07-CLANNAD-investigation]] | Private copyrighted artifact; never install this research reconstruction on hardware. |
| `/home/bee/Documents/src/vita-jp-overlay-discovery/build/clannad-offline/offline-capture-proof.json` | Corrected full-feed two-page proof, artifact/screenshot/log hashes, ranks and update counts, no raw dialogue | Status `generic_capture_and_top5_ranking_verified_for_two_visible_pages`; manual selection and hardware kernel transport untested. |
| `/home/bee/Documents/src/vita-jp-overlay-discovery/build/clannad-offline/offline-capture-proof-before-ranking.json` | Preserved failed comparator baseline | SHA-256 `5e707f06762778b18a3e7c1d2eb1913ec00abc8b0463407dce41816d56426579`, 5,962 B. |
| `/home/bee/Documents/src/vita-jp-overlay-discovery/build/clannad-offline/bridge/README.md` | Test-only real-game capture transport, source/build references and limitations | Never install the bridge on physical hardware; logs can contain game text. |
| `/home/bee/Documents/src/vita-jp-overlay-discovery/build/clannad-offline/runtime-trace.jsonl` | Bounded private startup call trace | First verified call is zero-length SceLibc memmove, not dialogue. Avoid printing future payload fields wholesale. |
| `/home/bee/Documents/src/vita-jp-overlay-discovery/build/clannad-offline/vita3k-source/` | Vita3K source at `b75d5e6292d5caa6a8411bacc9db1eca1531beab` | Source inspection of debugger double-lock; no emulator source copied into notes. |
| `/home/bee/Documents/src/vita-jp-overlay-discovery/build/vita3k-libc-test/pref/ux0/data/VitaJPOverlay-hook-test/results.txt` | Synthetic production-plugin result: 255 checks, 8 call events, 2 register events, 10 matching sources | Test-app sources, not commercial CLANNAD dialogue. |
| `/home/bee/Documents/src/vita-jp-overlay-low-memory/build/host/low-memory-build.log` | Host build/test output | Verify its revision before reusing a result. |
| `/home/bee/Documents/src/vita-jp-overlay-low-memory/build/vita-hooks-device/low-memory-build.log` | Device-build output | Installed shell hash in [[02-Current-device-state]]. |
| `/home/bee/Documents/src/ps-vita-ocr-lab/device/agent/` | Existing private control configuration and paired identity | Reference location only; do not print, copy or commit its contents. |

## What to preserve for the next finding

Keep source revision, build settings, tool versions, exact artifact hash, command, completion status, observations, and a narrow evidence path. Preserve failed results too, including whether a failure occurred before execution, during authentication, while reading memory, or after a launch was merely accepted. [[Experiment-template]] provides the fields.
