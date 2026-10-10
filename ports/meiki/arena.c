/* SPDX-License-Identifier: GPL-3.0-or-later
 * Adapted from bee-san/vita-jp-overlay core/ocr_heap.c. One serialized job.
 * MNN's model and tensor allocations use this pool; STL metadata is separate.
 */
#include "arena.h"
#include <stdint.h>
#define ALIGN 64u
typedef struct Block {
    size_t bytes;
    struct Block *prev, *next;
    int free;
} Block;
static Block *head;
static size_t used, peak;
static int failed;

int meiki_arena_begin(void *memory, size_t bytes) {
    if (head || !memory || ((uintptr_t)memory & (ALIGN - 1)) ||
        bytes < 2 * ALIGN)
        return -1;
    head = memory;
    head->bytes = (bytes & ~(size_t)(ALIGN - 1)) - ALIGN;
    head->prev = head->next = NULL;
    head->free = 1;
    used = peak = 0;
    failed = 0;
    return 0;
}

void *meiki_arena_alloc(size_t bytes) {
    if (!head || bytes > SIZE_MAX - ALIGN) {
        failed = 1;
        return NULL;
    }
    bytes = bytes ? (bytes + ALIGN - 1) & ~(size_t)(ALIGN - 1) : ALIGN;
    for (Block *b = head; b; b = b->next) {
        if (!b->free || b->bytes < bytes)
            continue;
        if (b->bytes >= bytes + 2 * ALIGN) {
            Block *tail = (Block *)((unsigned char *)b + ALIGN + bytes);
            tail->bytes = b->bytes - bytes - ALIGN;
            tail->prev = b;
            tail->next = b->next;
            tail->free = 1;
            if (tail->next)
                tail->next->prev = tail;
            b->next = tail;
            b->bytes = bytes;
        }
        b->free = 0;
        used += b->bytes + ALIGN;
        if (used > peak)
            peak = used;
        return (unsigned char *)b + ALIGN;
    }
    failed = 1;
    return NULL;
}

void meiki_arena_free(void *pointer) {
    if (!pointer)
        return;
    Block *b = (Block *)((unsigned char *)pointer - ALIGN);
    used -= b->bytes + ALIGN;
    b->free = 1;
    if (b->next && b->next->free) {
        b->bytes += ALIGN + b->next->bytes;
        b->next = b->next->next;
        if (b->next)
            b->next->prev = b;
    }
    if (b->prev && b->prev->free) {
        b->prev->bytes += ALIGN + b->bytes;
        b->prev->next = b->next;
        if (b->next)
            b->next->prev = b->prev;
    }
}
size_t meiki_arena_used(void) { return used; }
size_t meiki_arena_peak(void) { return peak; }
int meiki_arena_failed(void) { return failed; }
void meiki_arena_end(void) { head = NULL; }
