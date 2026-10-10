/* Caller-facing code uses only kernel APIs and sceClib: no newlib TLS access. */
#include "internal.h"
#include <psp2/kernel/clib.h>
#include <psp2/kernel/modulemgr.h>
#include <psp2/kernel/threadmgr.h>

enum { COMMAND_RUN = 1, COMMAND_STOP = 2, COMMAND_DETECT = 3 };
static SceUID gate = -1, command_event = -1, done_event = -1, engine = -1;
enum { ENGINE_STOPPED = 0, ENGINE_RUNNING = 1, ENGINE_QUIESCED = 2 };
static int running, command, result;
static volatile int done;
static void *metadata;
static size_t metadata_bytes;
static char model_path[256];
static MeikiModuleRequest request;
static MeikiModuleDetectRequest detect_request;
static MeikiModuleStats stats;

static int overlaps(const void *a, size_t an, const void *b, size_t bn)
{
    uintptr_t ap = (uintptr_t)a, bp = (uintptr_t)b;
    if (ap > UINTPTR_MAX - an || bp > UINTPTR_MAX - bn) return 1;
    return ap < bp + bn && bp < ap + an;
}

static void complete(int rc)
{
    result = rc;
    __atomic_store_n(&done, 1, __ATOMIC_RELEASE);
    sceKernelSetEventFlag(done_event, 1);
}

static void wait_done(void)
{
    while (!__atomic_load_n(&done, __ATOMIC_ACQUIRE)) {
        unsigned bits;
        SceUInt timeout = 100000;
        int rc = sceKernelWaitEventFlag(done_event, 1,
                                        SCE_EVENT_WAITAND | SCE_EVENT_WAITCLEAR_PAT,
                                        &bits, &timeout);
        // A transient canceled wait must not release caller pointers while
        // inference is still using them. The ownership handshake remains live.
        if (rc < 0) sceKernelDelayThread(1000);
    }
}

static int engine_main(SceSize unused, void *arg)
{
    (void)unused; (void)arg;
    sceClibMemset(&stats, 0, sizeof(stats));
    complete(meiki_engine_initialize(metadata, metadata_bytes, &stats));
    for (;;) {
        unsigned bits;
        SceUInt timeout = 100000;
        int rc = sceKernelWaitEventFlag(command_event, 1,
                                        SCE_EVENT_WAITAND | SCE_EVENT_WAITCLEAR_PAT,
                                        &bits, &timeout);
        (void)rc;
        // The atomic slot is also a fallback if a kernel event is canceled.
        // Timeout polling preserves the request ownership handshake.
        int next = __atomic_exchange_n(&command, 0, __ATOMIC_ACQ_REL);
        if (!next) continue;
        if (next == COMMAND_STOP) {
            sceClibMemset(&stats.runtime, 0, sizeof(stats.runtime));
            complete(meiki_engine_shutdown(&stats));
            return 0;
        }
        sceClibMemset(&stats, 0, sizeof(stats));
        complete(next == COMMAND_DETECT ? meiki_engine_detect(&detect_request, &stats) :
                                         meiki_engine_run(&request, &stats));
    }
}

static int submit(int requested_command, MeikiModuleStats *out)
{
    __atomic_store_n(&done, 0, __ATOMIC_RELEASE);
    sceKernelClearEventFlag(done_event, 0);
    __atomic_store_n(&command, requested_command, __ATOMIC_RELEASE);
    sceKernelSetEventFlag(command_event, 1);
    wait_done();
    if (out) sceClibMemcpy(out, &stats, sizeof(stats));
    return result;
}

static int api_run(const MeikiModuleRequest *in, MeikiModuleStats *out)
{
    if (!in || in->size != sizeof(*in) || !in->model_path || !in->input ||
        !in->workspace || !in->output || in->workspace_bytes < MEIKI_WORKSPACE_BYTES ||
        ((uintptr_t)in->workspace & 63u) || ((uintptr_t)in->input & 3u) ||
        ((uintptr_t)in->output & 3u) ||
        overlaps(metadata, metadata_bytes, in->workspace, in->workspace_bytes) ||
        overlaps(in->input, MEIKI_INPUT_ELEMENTS * sizeof(float), in->workspace, in->workspace_bytes) ||
        overlaps(in->output, sizeof(*in->output), in->workspace, in->workspace_bytes) ||
        overlaps(in->input, MEIKI_INPUT_ELEMENTS * sizeof(float), metadata, metadata_bytes) ||
        overlaps(in->output, sizeof(*in->output), metadata, metadata_bytes) ||
        overlaps(in->input, MEIKI_INPUT_ELEMENTS * sizeof(float), in->output, sizeof(*in->output)) ||
        (out && (((uintptr_t)out & 7u) ||
                 overlaps(out, sizeof(*out), metadata, metadata_bytes) ||
                 overlaps(out, sizeof(*out), in->workspace, in->workspace_bytes) ||
                 overlaps(out, sizeof(*out), in->input, MEIKI_INPUT_ELEMENTS * sizeof(float)) ||
                 overlaps(out, sizeof(*out), in->output, sizeof(*in->output)))))
        return MEIKI_MODULE_INVALID;
    if (__atomic_load_n(&running, __ATOMIC_ACQUIRE) != ENGINE_RUNNING || gate < 0)
        return MEIKI_MODULE_STOPPED;
    if (sceKernelTryLockMutex(gate, 1) < 0) return MEIKI_MODULE_BUSY;
    int rc;
    if (running != ENGINE_RUNNING) rc = MEIKI_MODULE_STOPPED;
    else if (stats.tainted) rc = MEIKI_MODULE_TAINTED;
    else {
        size_t length = sceClibStrnlen(in->model_path, sizeof(model_path));
        if (!length || length == sizeof(model_path)) rc = MEIKI_MODULE_INVALID;
        else {
            sceClibMemcpy(&request, in, sizeof(request));
            sceClibMemcpy(model_path, in->model_path, (SceSize)(length + 1));
            request.model_path = model_path;
            rc = submit(COMMAND_RUN, out);
            // DONE acknowledges the engine has released every request pointer.
            sceClibMemset(&request, 0, sizeof(request));
        }
    }
    sceKernelUnlockMutex(gate, 1);
    return rc;
}

static int api_detect(const MeikiModuleDetectRequest *in, MeikiModuleStats *out)
{
    if (!in || in->size != sizeof(*in) || !in->model_path || !in->input ||
        !in->workspace || !in->output || in->target_width <= 0 || in->target_height <= 0 || in->workspace_bytes < MEIKI_WORKSPACE_BYTES ||
        ((uintptr_t)in->workspace & 63u) || ((uintptr_t)in->input & 3u) ||
        ((uintptr_t)in->output & 3u) ||
        overlaps(metadata, metadata_bytes, in->workspace, in->workspace_bytes) ||
        overlaps(in->input, MEIKI_DETECT_ELEMENTS * sizeof(float), in->workspace, in->workspace_bytes) ||
        overlaps(in->output, sizeof(*in->output), in->workspace, in->workspace_bytes) ||
        overlaps(in->input, MEIKI_DETECT_ELEMENTS * sizeof(float), metadata, metadata_bytes) ||
        overlaps(in->output, sizeof(*in->output), metadata, metadata_bytes) ||
        overlaps(in->input, MEIKI_DETECT_ELEMENTS * sizeof(float), in->output, sizeof(*in->output)) ||
        (out && (((uintptr_t)out & 7u) ||
                 overlaps(out, sizeof(*out), metadata, metadata_bytes) ||
                 overlaps(out, sizeof(*out), in->workspace, in->workspace_bytes) ||
                 overlaps(out, sizeof(*out), in->input, MEIKI_DETECT_ELEMENTS * sizeof(float)) ||
                 overlaps(out, sizeof(*out), in->output, sizeof(*in->output)))))
        return MEIKI_MODULE_INVALID;
    if (__atomic_load_n(&running, __ATOMIC_ACQUIRE) != ENGINE_RUNNING || gate < 0)
        return MEIKI_MODULE_STOPPED;
    if (sceKernelTryLockMutex(gate, 1) < 0) return MEIKI_MODULE_BUSY;
    int rc;
    if (running != ENGINE_RUNNING) rc = MEIKI_MODULE_STOPPED;
    else if (stats.tainted) rc = MEIKI_MODULE_TAINTED;
    else {
        size_t length = sceClibStrnlen(in->model_path, sizeof(model_path));
        if (!length || length == sizeof(model_path)) rc = MEIKI_MODULE_INVALID;
        else {
            sceClibMemcpy(&detect_request, in, sizeof(detect_request));
            sceClibMemcpy(model_path, in->model_path, (SceSize)(length + 1));
            detect_request.model_path = model_path;
            rc = submit(COMMAND_DETECT, out);
            // DONE acknowledges the engine has released every request pointer.
            sceClibMemset(&detect_request, 0, sizeof(detect_request));
        }
    }
    sceKernelUnlockMutex(gate, 1);
    return rc;
}

static int api_stop(MeikiModuleStats *out)
{
    if (out && (((uintptr_t)out & 7u) || overlaps(out, sizeof(*out), metadata, metadata_bytes)))
        return MEIKI_MODULE_INVALID;
    if (__atomic_load_n(&running, __ATOMIC_ACQUIRE) == ENGINE_STOPPED || gate < 0)
        return MEIKI_MODULE_STOPPED;
    if (sceKernelTryLockMutex(gate, 1) < 0) return MEIKI_MODULE_BUSY;
    if (running == ENGINE_STOPPED) {
        sceKernelUnlockMutex(gate, 1);
        return MEIKI_MODULE_STOPPED;
    }
    int rc = 0, status;
    if (running == ENGINE_RUNNING) {
        submit(COMMAND_STOP, out);
        __atomic_store_n(&running, ENGINE_QUIESCED, __ATOMIC_RELEASE);
    } else if (out) sceClibMemcpy(out, &stats, sizeof(stats));
    // A retry after failed join/delete only retries that kernel cleanup. It
    // never submits another command to a worker which has already stopped.
    if (sceKernelWaitThreadEnd(engine, &status, NULL) < 0 ||
        sceKernelDeleteThread(engine) < 0) rc = MEIKI_MODULE_KERNEL;
    else { engine = -1; __atomic_store_n(&running, ENGINE_STOPPED, __ATOMIC_RELEASE); }
    sceKernelUnlockMutex(gate, 1);
    return rc;
}

static void delete_handles(void)
{
    if (done_event >= 0) sceKernelDeleteEventFlag(done_event);
    if (command_event >= 0) sceKernelDeleteEventFlag(command_event);
    if (gate >= 0) sceKernelDeleteMutex(gate);
    done_event = command_event = gate = -1;
}

int module_start(SceSize args, void *argp)
{
    const MeikiModuleStart *config = (const MeikiModuleStart *)argp;
    if (args != sizeof(*config) || !config || config->size != sizeof(*config) ||
        config->magic != MEIKI_MODULE_MAGIC || config->abi != MEIKI_MODULE_ABI ||
        !config->metadata_heap || ((uintptr_t)config->metadata_heap & 63u) ||
        config->metadata_bytes < MEIKI_MODULE_METADATA_BYTES || !config->api_out ||
        ((uintptr_t)config->api_out & 3u) ||
        (uintptr_t)config->metadata_heap > UINTPTR_MAX - config->metadata_bytes ||
        overlaps(config->api_out, sizeof(*config->api_out), config->metadata_heap, config->metadata_bytes) ||
        (config->initial_stats && (((uintptr_t)config->initial_stats & 7u) ||
            overlaps(config->initial_stats, sizeof(*config->initial_stats), config->metadata_heap, config->metadata_bytes) ||
            overlaps(config->initial_stats, sizeof(*config->initial_stats), config->api_out, sizeof(*config->api_out)))))
        return SCE_KERNEL_START_FAILED;
    sceClibMemset(config->api_out, 0, sizeof(*config->api_out));
    metadata = config->metadata_heap;
    metadata_bytes = config->metadata_bytes;
    gate = sceKernelCreateMutex("MeikiModuleGate", 0, 0, NULL);
    command_event = sceKernelCreateEventFlag("MeikiModuleCommand", 0, 0, NULL);
    done_event = sceKernelCreateEventFlag("MeikiModuleDone", 0, 0, NULL);
    if (gate < 0 || command_event < 0 || done_event < 0) { delete_handles(); return SCE_KERNEL_START_FAILED; }
    engine = sceKernelCreateThread("MeikiModuleEngine", engine_main, 0x10000100,
                                   128u * 1024u, 0, 0, NULL);
    done = 0;
    if (engine < 0 || sceKernelStartThread(engine, 0, NULL) < 0) {
        if (engine >= 0) sceKernelDeleteThread(engine);
        engine = -1;
        delete_handles();
        return SCE_KERNEL_START_FAILED;
    }
    wait_done();
    if (config->initial_stats) sceClibMemcpy(config->initial_stats, &stats, sizeof(stats));
    __atomic_store_n(&running, ENGINE_RUNNING, __ATOMIC_RELEASE);
    if (result < 0) {
        api_stop(NULL);
        // Start failure must leave no executing code or metadata users, even
        // if the first kernel join/delete call failed transiently.
        if (engine >= 0) {
            int status;
            while (sceKernelWaitThreadEnd(engine, &status, NULL) < 0)
                sceKernelDelayThread(1000);
            sceKernelDeleteThread(engine);
            engine = -1;
            __atomic_store_n(&running, ENGINE_STOPPED, __ATOMIC_RELEASE);
        }
        delete_handles();
        return SCE_KERNEL_START_FAILED;
    }
    config->api_out->size = sizeof(*config->api_out);
    config->api_out->abi = MEIKI_MODULE_ABI;
    config->api_out->run = api_run;
    config->api_out->stop = api_stop;
    config->api_out->detect = api_detect;
    return SCE_KERNEL_START_SUCCESS;
}

int module_stop(SceSize args, void *argp)
{
    (void)args; (void)argp;
    if (__atomic_load_n(&running, __ATOMIC_ACQUIRE) != ENGINE_STOPPED || engine >= 0)
        return SCE_KERNEL_STOP_FAIL;
    delete_handles();
    metadata = NULL;
    metadata_bytes = 0;
    return SCE_KERNEL_STOP_SUCCESS;
}
