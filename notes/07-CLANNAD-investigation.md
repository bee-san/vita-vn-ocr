---
title: CLANNAD native text investigation
created: 2026-10-10
updated: 2026-10-10
status: capture-unverified
tags: [vita, clannad, native-text, investigation]
---

# CLANNAD native text investigation

**Title:** CLANNAD, `PCSG00415`. **Outcome so far:** the low-memory picker opens, but no advancing dialogue capture is verified. **Newest finding:** the four generic hooks targeted imports absent from this game's inspected main module. The module instead imports equivalent functions from `SceLibc`.

Native extraction is technically feasible in this homebrew environment. The current observations do not establish working CLANNAD extraction, and they do not yet establish that CLANNAD needs a bespoke engine hook.

## What happened on the device

After [[04-Firmware-365-boot-ABI]] was corrected, all three modules could reach STARTED. Opening the overlay initially failed memory allocation. After [[06-Low-memory-overlay]], the user saw the picker and advanced dialogue, but it kept searching.

`status-clannad-02.txt` records candidate count 0 and discovery advancing through addresses including `0x8187A000`, `0x81EDA000`, `0x83C4E000`, then beginning further passes at `0x80000000`. The increasing cursor establishes scanner progress through its loop; it does **not** establish successful reads of the game's actual mappings. Separate map/read diagnostics are needed.

A later launch was accepted by the control API but did not establish a sustained running game. The user reported power-off, then reported the Vita on again. The investigator restored the original text plugin and verified readback; the later failure cause is unresolved.

## Main-module import discovery

Private analysis of an isolated Vita3K loaded-memory ELF enumerated **250 function imports** from module `VITA`. The original four targeted NIDs were all absent:

```text
0xAC595E68  0x2CDFCD1C  0x14E9DBD7  0xC458D60A
```

The module imports these equivalents from **SceLibc, library NID `0xBE43BB07`**:

| Function | Function NID | Observed import-stub address |
|---|---|---|
| `memcpy` | `0x7205BFDB` | `0x810A9388` |
| `strcpy` | `0x85B924B7` | `0x810A93D8` |
| `strncpy` | `0x9F87712D` | `0x810A9428` |
| `memmove` | `0xAF5C218D` | `0x810A9458` |

These are addresses in the inspected loaded image, not a ready-made hardware register profile. Prefer correct library/function import matching over blindly patching the listed address. Confirm attachment results on the actual process.

No PVF/font import was present in this main-module inventory. A shared-font-hook plan is therefore not supported by this inventory alone; inspect dynamic modules or actual call paths before extending that conclusion.

**Decision:** correct the generic library coverage and observe attachment/capture before deciding a CLANNAD-specific native instruction hook is necessary. A STARTED plugin with four absent imports can still produce zero useful observations.

The correction is now implemented as `0.6-hooks.5`, commit `0a66015994dbb3f12a3a506f0c0e4cab8ec72fdd`, and passes 255 synthetic Vita3K checks. It is published in draft PR #12, not installed or proven against real CLANNAD dialogue. See [[14-Retail-libc-hook-coverage]].

## Executable provenance: two different artifacts

| Artifact | SHA-256 | What it is |
|---|---|---|
| Original installed `eboot.bin` | `76b2b2af4ba367ada9dfcd9d464c713d18976f14cd5a4d117bb188b1a9c61b5b` | Encrypted SELF, 516,688 B. Its five segments were marked encrypted. It did not expose readable imports by simple decompression. |
| `eboot.loaded.elf` | `23611e880fb573f85246756073edeb20cdfa02c3f7062cd0427cb3b0f5d499d9` | Decrypted, relocated PT_LOAD bytes dumped after isolated Vita3K load and before game execution. It omits original relocation sections and is for analysis, not installation. |

An earlier device decryption attempt failed in authentication with native result `0x800F0616`; no decrypted output from that attempt was established. The later loaded image is separate evidence, not success retroactively assigned to the failed operation. A patch executable path was absent in the inspected install.

Private metadata lives under `vita-jp-overlay-discovery/build/clannad-offline/`. See [[11-Sources-and-artifacts]]. Do not commit the executable, segment dumps, game install, decryption cache, license or full runtime dump.

## Encoding evidence

Offline font-table inspection reported 7,120 codepoints; CP932 value `0x82A0` was present at index 368, while Unicode `0x3042` was absent. This supports testing CP932 paths. It does not prove that the current visible dialogue is stored as one NUL-terminated CP932 string, or that generic copy hooks will see that buffer at the useful time.

Potential remaining causes of empty/wrong candidates include failed game-memory mapping, an unsuitable decoder/filter, internal copies rather than imported functions, transient decoded text, custom control bytes, static script storage, or a changing pointer not followed by the current source. These are hypotheses, not observed engine facts.

## Minimum successful test

1. Record exact plugin hashes and actual hook attachment results in the game process.
2. Open a dialogue page; establish at least one bounded observation and candidate.
3. Select a candidate matching that page without copying dialogue into public notes.
4. Advance to another distinct page; establish that the selected source follows it rather than retaining a static script line.
5. Close/reopen the overlay and verify current text remains correct.
6. Only then test dictionary lookup, title restart, scene changes and stability.

Do not infer success from a hook ID, a candidate count, one matching buffer, a host test or an emulator launch. See [[08-Validation-and-measurements]] and [[12-Open-questions-and-next-steps]].
