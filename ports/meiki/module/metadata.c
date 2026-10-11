/* Private SDK newlib heap. Does not link/call newlib's USER memblock allocator.
 * All users execute on the module engine thread, never on Shell/Paf threads.
 */
#include "internal.h"
#include <errno.h>
#include <reent.h>
#include <psp2/kernel/threadmgr.h>

static unsigned char *heap_base;
static size_t heap_capacity, heap_break, heap_peak;
static int engine_thread;
static uint32_t tainted;
/* Nonallocating marker in the pinned MNN embedding patch. Set before libc/new
 * unwinding so destructors cannot reuse partially mutated allocator metadata. */
extern void MNNEmbeddingRecordAllocationFailure(void);

void meiki_metadata_prepare(void *base, size_t bytes, int thread)
{
    heap_base = (unsigned char *)base;
    heap_capacity = bytes;
    heap_break = heap_peak = 0;
    tainted = 0;
    engine_thread = thread;
}

void *_sbrk_r(struct _reent *reent, ptrdiff_t increment)
{
    size_t next;
    if (!heap_base || sceKernelGetThreadId() != engine_thread) goto failure;
    if (increment < 0) {
        size_t shrink = (size_t)(-(increment + 1)) + 1u;
        if (shrink > heap_break) goto failure;
        next = heap_break - shrink;
    } else {
        if ((size_t)increment > heap_capacity - heap_break) goto failure;
        next = heap_break + (size_t)increment;
    }
    void *previous = heap_base + heap_break;
    heap_break = next;
    if (next > heap_peak) heap_peak = next;
    return previous;
failure:
    tainted = 1;
    MNNEmbeddingRecordAllocationFailure();
    if (reent) reent->_errno = ENOMEM;
    return (void *)-1;
}

uint32_t meiki_metadata_brk_peak(void) { return (uint32_t)heap_peak; }
uint32_t meiki_metadata_tainted(void) { return tainted; }
unsigned int _get_vita_heap_size(void) { return (unsigned int)heap_capacity; }
