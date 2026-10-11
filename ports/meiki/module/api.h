/* Versioned C ABI for an isolated Vita user-module OCR engine. */
#ifndef MEIKI_MODULE_API_H
#define MEIKI_MODULE_API_H
#include "../runtime.h"
#include "../detect_runtime.h"
#ifdef __cplusplus
extern "C" {
#endif
#define MEIKI_MODULE_ABI 3u
#define MEIKI_MODULE_MAGIC 0x4d4b494du
#define MEIKI_MODULE_METADATA_BYTES (4u * 1024u * 1024u)
enum {
    MEIKI_MODULE_INVALID = -100,
    MEIKI_MODULE_BUSY = -101,
    MEIKI_MODULE_STOPPED = -102,
    MEIKI_MODULE_TAINTED = -103,
    MEIKI_MODULE_KERNEL = -104,
    MEIKI_MODULE_INITIALIZATION = -105
};
typedef struct {
    uint32_t size;
    const char *model_path;
    const float *input; /* MEIKI_INPUT_ELEMENTS BGR NCHW floats */
    void *workspace; /* >= MEIKI_WORKSPACE_BYTES, aligned to 64 bytes */
    size_t workspace_bytes;
    MeikiOutput *output;
    const MeikiModelReader *reader; /* Optional; when present model_path is unused. */
} MeikiModuleRequest;
typedef struct {
    uint32_t size;
    const char *model_path;
    const float *input; /* MEIKI_DETECT_ELEMENTS BGR NCHW floats */
    int32_t target_width, target_height;
    void *workspace;
    size_t workspace_bytes;
    MeikiDetectOutput *output;
} MeikiModuleDetectRequest;
typedef struct {
    uint32_t size;
    MeikiStats runtime;
    uint32_t metadata_capacity, metadata_brk_peak; /* actual sbrk highwater */
    /* mallinfo snapshots before/after calls and their maximum; this maximum
     * does not measure every transient malloc live-allocation peak. */
    uint32_t metadata_used_before, metadata_used_after, metadata_used_peak;
    uint32_t tainted;
    int32_t engine_thread_id;
} MeikiModuleStats;
typedef struct {
    uint32_t size, abi;
    /* Synchronous, serial: run returns only after the dedicated engine thread
     * has released every request pointer. Concurrent run/stop returns BUSY.
     * All pointers and both buffers remain caller-owned until return.
     * Never call run from a display hook or Shell's UI thread.
     */
    int (*run)(const MeikiModuleRequest *request, MeikiModuleStats *stats);
    /* Quiesce and join the engine before calling sceKernelStopUnloadModule.
     * Refuses a busy engine. The API becomes invalid after module unload.
     * Metadata OOM taints this instance; stop/unload then load a fresh image.
     * Returns 0 after a successful join even if stats.tainted is set. Kernel
     * join/delete failure is retryable; no second STOP command is submitted.
     */
    int (*stop)(MeikiModuleStats *stats);
    /* Same synchronous ownership and serialization contract as run. Detection
     * releases its model/session before recognition reuses the same arena. */
    int (*detect)(const MeikiModuleDetectRequest *request, MeikiModuleStats *stats);
} MeikiModuleApi;
typedef struct {
    uint32_t size, magic, abi;
    void *metadata_heap; /* >=4 MiB, aligned to 64 bytes, owned by caller */
    size_t metadata_bytes;
    MeikiModuleApi *api_out;
    MeikiModuleStats *initial_stats;
} MeikiModuleStart;
/* Pass a MeikiModuleStart to sceKernelLoadStartModule. On success api_out
 * receives these entry points after constructors finish on the engine thread.
 * Keep metadata_heap alive until stop plus successful module unload. Model
 * workspace, metadata, input, output and optional stats must not overlap.
 * api_out and initial_stats must lie outside metadata and each other. Optional
 * stats pointers have natural 8-byte alignment. Start refuses malformed args.
 */
#ifdef __cplusplus
}
#endif
#endif
