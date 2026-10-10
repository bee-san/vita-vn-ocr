---
title: Low-memory native text overlay
created: 2026-10-10
updated: 2026-10-10
status: picker-device-verified
tags: [vita, memory, overlay, ownership]
---

# Low-memory native text overlay

CLANNAD exposed an unnecessary dependency: opening a native source picker tried to allocate the same 768 KiB result storage used by richer lookup/OCR jobs. On the physical device that allocation failed with `0x80024302`, `SCE_KERNEL_ERROR_NO_FREE_PHYSICAL_PAGE` (C1-2569-2), and the user saw “not enough memory for the overlay.”

The fix separates viewing and selecting native text from optional result storage. It has been installed, read back and booted. The user can now open the picker. This fixes that observed allocation gate; it does not fix empty capture sources by itself.

## Ownership and allocation policy

- `activate_game`, `open_overlay` and subtitle activation no longer allocate result arenas eagerly.
- The picker uses resident view storage.
- Selected native text is filtered in the existing 24 KiB control-thread scratch and copied under `view_lock` into `g_view.hooks[0].text`.
- A small static `VjoEntryList` exposes the raw header with no allocated dictionary entries. Picker/raw-header views are mutually exclusive; publishing picker data clears the old list before overwriting its storage.
- Subtitles copy to existing strip storage without starting a job or requiring a result arena.
- `start_job` publishes native text first, returns immediately when lookup is disabled, and allocates only for actual dictionary/OCR work.
- Separate allocation backoff avoids retrying a failed physical allocation every 50 ms.
- Allocation failure leaves raw text available with a dictionary-memory message.
- Stale asynchronous work cannot replace a newer native-text result or changed pipeline.

## Optional Paf fallback

| Path | Total result storage | Arena size | Evidence |
|---|---:|---:|---|
| Normal physical allocation | 786,432 B | Two × 393,216 B | Existing full lookup/OCR capacity |
| Native selected text fallback | 262,144 B | Two × 131,072 B | Host validated; later physical 256 KiB Paf dictionary jobs complete with rc=0 |
| Picker/raw text only | No result arenas | None | Host validated and picker opened on device |

When the full allocation fails for a current selected native source, `vjo_shell_heap_alloc` uses `sce_paf_memalign(16, bytes)` and releases with `sce_paf_free`. Existing Paf heap pages may satisfy it without acquiring a new physical memory block. The later manual-mode log now confirms this allocation path succeeded for selected text in CLANNAD; it does not prove the selected bytes are the visible dialogue.

`mem_heap` tracks ownership independently of a negative `mem_uid`. Freeing storage clears the view list under lock, resets arena/cache ownership and uses the matching allocator. A native-to-OCR transition retires the small heap allocation before requesting full capacity, with no running job using it. Deferred exit cleanup checks both ownership kinds.

## Validation and physical result

Source: `e2ed3b7`; shell version `0.6-hooks.4`. Ten host CTest suites and 30 production-worker cases passed in the recorded run (1.59 s total host time). The cases cover real production worker logic with NULL arenas, simulated physical allocator failure, raw/subtitle continuity, allocation backoff, stale completion, teardown and pipeline changes. Two cases execute real relay lookup inside the 128 KiB arenas and verify correct release and migration to OCR capacity.

The Vita release build and packaged ABI check passed. Shell SHA-256 `45f86b1ea83c08bdf61c1dfd174cc5356151e746a3659a0299218436a8f10c8c` matched full device readback. Normal boot passed. User confirmed the picker appeared, but it remained in “finding Japanese text” after advancing dialogue. See [[07-CLANNAD-investigation]].

Metadata logging records session, sequence, candidate count, scanning state, address, selected ID/kind/encoding, byte count and update count without raw dialogue. It throttles to at most one update per second while retaining a pending transition.

Merged as [overlay PR #11](https://github.com/bee-san/vita-jp-overlay/pull/11), merge `13bcf816ab4e45fb8f59da0f52407a621b07284e`. Later physical manual selection uses the 256 KiB Paf fallback and returns dictionary `rc=0` (37 ms, arena peak 110,080 B; reopens 21/25 ms). These are isolated job observations, not total memory or performance measurements. The user suspects the selected text is wrong, and correct captured pages remain unverified. See [[20-Fixed-manual-hook-workflow]].
