/* Private newlib and C++ initialization lives only on this module's thread.
 * The fixed SDK TLS key 0x89 therefore never replaces Shell's own reent.
 */
#include "internal.h"
#include <malloc.h>
#include <new>
#include <reent.h>
#include <psp2/kernel/threadmgr.h>

extern "C" {
void _init_vita_reent(void);
void _init_vita_malloc(void);
void _init_vita_io(void);
void _free_vita_io(void);
void _free_vita_malloc(void);
void _free_vita_reent(void);
void __libc_init_array(void);
void __libc_fini_array(void);
void __cxa_finalize(void *);
struct _reent *__getreent(void);
extern void *__dso_handle;
/* -nostartfiles deliberately omits the SDK _start / default USER heap. */
void _init(void) {}
void _fini(void) {}
// libc's _exit references this hook. It must never reach the SDK's default
// heap teardown, which owns a USER memblock this module did not allocate.
void _free_vita_newlib(void) {
    _free_vita_io();
    _free_vita_malloc();
    _free_vita_reent();
}
}
static bool initialized, runtime_tainted;
static uint32_t metadata_capacity, observed_used_peak;

static uint32_t used_bytes() { return (uint32_t)mallinfo().uordblks; }
static void observe(MeikiModuleStats *stats)
{
    uint32_t used = used_bytes();
    if (used > observed_used_peak) observed_used_peak = used;
    stats->size = sizeof(*stats);
    stats->metadata_capacity = metadata_capacity;
    stats->metadata_brk_peak = meiki_metadata_brk_peak();
    stats->metadata_used_after = used;
    stats->metadata_used_peak = observed_used_peak;
    stats->tainted = meiki_metadata_tainted() || runtime_tainted;
    stats->engine_thread_id = sceKernelGetThreadId();
}

extern "C" int meiki_engine_initialize(void *metadata, size_t bytes,
                                      MeikiModuleStats *stats)
{
    meiki_metadata_prepare(metadata, bytes, sceKernelGetThreadId());
    metadata_capacity = (uint32_t)bytes;
    observed_used_peak = 0;
    runtime_tainted = false;
    _init_vita_reent();
    _init_vita_malloc();
    _init_vita_io();
    try {
        __libc_init_array();
        initialized = true;
    } catch (...) {
        runtime_tainted = true;
        observe(stats);
        stats->tainted = 1;
        return MEIKI_MODULE_INITIALIZATION;
    }
    observe(stats);
    stats->metadata_used_before = stats->metadata_used_after;
    return stats->tainted ? MEIKI_MODULE_TAINTED : 0;
}

extern "C" int meiki_engine_run(const MeikiModuleRequest *request,
                                MeikiModuleStats *stats)
{
    if (!initialized) return MEIKI_MODULE_STOPPED;
    if (meiki_metadata_tainted()) { observe(stats); return MEIKI_MODULE_TAINTED; }
    stats->metadata_used_before = used_bytes();
    int rc = meiki_run(request->model_path, request->input, request->workspace,
                       request->workspace_bytes, request->output, &stats->runtime);
    if (stats->runtime.heap_remaining || stats->runtime.allocation_failed || rc == -11)
        runtime_tainted = true;
    observe(stats);
    return stats->tainted ? MEIKI_MODULE_TAINTED : rc;
}

extern "C" int meiki_engine_detect(const MeikiModuleDetectRequest *request,
                                MeikiModuleStats *stats)
{
    if (!initialized) return MEIKI_MODULE_STOPPED;
    if (meiki_metadata_tainted()) { observe(stats); return MEIKI_MODULE_TAINTED; }
    stats->metadata_used_before = used_bytes();
    int rc = meiki_detect_run(request->model_path, request->input,
                              request->target_width, request->target_height,
                              request->workspace,
                       request->workspace_bytes, request->output, &stats->runtime);
    if (stats->runtime.heap_remaining || stats->runtime.allocation_failed || rc == -11)
        runtime_tainted = true;
    observe(stats);
    return stats->tainted ? MEIKI_MODULE_TAINTED : rc;
}

extern "C" int meiki_engine_shutdown(MeikiModuleStats *stats)
{
    int rc = 0;
    stats->metadata_used_before = used_bytes();
    try {
        // GCC bare-metal crtbegin's fini array does not run registered atexit
        // destructors. The private newlib registry contains only this module.
        __cxa_finalize(__dso_handle);
        if (initialized) __libc_fini_array();
    } catch (...) { runtime_tainted = true; rc = MEIKI_MODULE_TAINTED; }
    initialized = false;
    _reclaim_reent(__getreent());
    _free_vita_io();
    observe(stats);
    if (stats->tainted) rc = MEIKI_MODULE_TAINTED;
    _free_vita_malloc();
    _free_vita_reent();
    *(void **)sceKernelGetTLSAddr(0x89) = nullptr;
    return rc;
}
