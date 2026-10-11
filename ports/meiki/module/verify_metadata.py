#!/usr/bin/env python3
"""Compile the actual private sbrk implementation with minimal host SDK shims.

This exercises caller-buffer bounds and owner-thread rejection under ASan and
UBSan. It does not emulate SDK malloc, module loading or Vita memory partitions.
"""

import argparse
import subprocess
from pathlib import Path

DRIVER = r"""
#include <assert.h>
#include <errno.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "internal.h"
#include <reent.h>
static int current_thread = 37;
static unsigned recorded_failures;
int sceKernelGetThreadId(void) { return current_thread; }
void MNNEmbeddingRecordAllocationFailure(void) { ++recorded_failures; }
void *_sbrk_r(struct _reent *, ptrdiff_t);
unsigned int _get_vita_heap_size(void);
static unsigned char backing[MEIKI_MODULE_METADATA_BYTES + 128] __attribute__((aligned(64)));
int main(void) {
    unsigned char *base = backing + 64;
    struct _reent reent = {0};
    memset(backing, 0xa5, sizeof(backing));
    meiki_metadata_prepare(base, MEIKI_MODULE_METADATA_BYTES, 37);
    assert(_get_vita_heap_size() == MEIKI_MODULE_METADATA_BYTES);
    assert(_sbrk_r(&reent, 0) == base);
    assert(_sbrk_r(&reent, MEIKI_MODULE_METADATA_BYTES) == base);
    memset(base, 0x5a, MEIKI_MODULE_METADATA_BYTES);
    assert(meiki_metadata_brk_peak() == MEIKI_MODULE_METADATA_BYTES);
    assert(_sbrk_r(&reent, 1) == (void *)-1 && reent._errno == ENOMEM);
    assert(meiki_metadata_tainted() && recorded_failures == 1);
    assert(_sbrk_r(&reent, 0) == base + MEIKI_MODULE_METADATA_BYTES);
    assert(_sbrk_r(&reent, -(ptrdiff_t)MEIKI_MODULE_METADATA_BYTES) == base + MEIKI_MODULE_METADATA_BYTES);
    assert(_sbrk_r(&reent, -1) == (void *)-1);
    for (unsigned i = 0; i < 64; ++i) {
        assert(backing[i] == 0xa5);
        assert(backing[MEIKI_MODULE_METADATA_BYTES + 64 + i] == 0xa5);
    }
    meiki_metadata_prepare(base, MEIKI_MODULE_METADATA_BYTES, 37);
    assert(!meiki_metadata_tainted());
    assert(_sbrk_r(&reent, PTRDIFF_MAX) == (void *)-1);
    assert(_sbrk_r(&reent, PTRDIFF_MIN) == (void *)-1);
    assert(_sbrk_r(NULL, -1) == (void *)-1);
    meiki_metadata_prepare(base, MEIKI_MODULE_METADATA_BYTES, 37);
    current_thread = 99;
    assert(_sbrk_r(&reent, 1) == (void *)-1 && meiki_metadata_tainted());
    current_thread = 37;
    meiki_metadata_prepare(base, MEIKI_MODULE_METADATA_BYTES, 37);
    size_t used = 0, peak = 0;
    uint32_t rng = 0x887123u;
    for (unsigned i = 0; i < 4000; ++i) {
        rng = rng * 1664525u + 1013904223u;
        ptrdiff_t delta = (ptrdiff_t)(rng % 32768u);
        if (rng & 0x80000000u) delta = -delta;
        int valid = delta < 0 ? (size_t)(-delta) <= used : (size_t)delta <= MEIKI_MODULE_METADATA_BYTES - used;
        void *previous = _sbrk_r(&reent, delta);
        if (!valid) assert(previous == (void *)-1);
        else {
            assert(previous == base + used);
            used = delta < 0 ? used - (size_t)(-delta) : used + (size_t)delta;
            if (used > peak) peak = used;
        }
        assert(_sbrk_r(&reent, 0) == base + used);
        assert(meiki_metadata_brk_peak() == peak);
    }
    puts("Private sbrk: exact capacity, guard bytes, exhaustion taint, signed overflow, "
         "owner-thread rejection and 4000 adjustments passed under ASan/UBSan");
    return 0;
}
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    shim = args.out / "shims"
    (shim / "psp2/kernel").mkdir(parents=True, exist_ok=True)
    (shim / "reent.h").write_text("struct _reent { int _errno; };\n")
    (shim / "psp2/kernel/threadmgr.h").write_text("int sceKernelGetThreadId(void);\n")
    source = args.out / "driver.c"
    source.write_text(DRIVER)
    module = Path(__file__).resolve().parent
    binary = args.out / "verify-metadata"
    command = [
        "cc",
        "-std=c99",
        "-O1",
        "-g",
        "-Wall",
        "-Wextra",
        "-Werror",
        "-fsanitize=address,undefined",
        "-fno-omit-frame-pointer",
        "-I",
        str(shim),
        "-I",
        str(module),
        str(module / "metadata.c"),
        str(source),
        "-o",
        str(binary),
    ]
    subprocess.run(command, check=True)
    subprocess.run([str(binary)], check=True)


if __name__ == "__main__":
    main()
