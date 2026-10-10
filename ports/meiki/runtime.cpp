#include "runtime.h"
#include "arena.h"
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <cstring>
#include <initializer_list>
#include <new>
#if defined(MEIKI_TRACE)
#include <cstdio>
#define TRACE(stage)                                                           \
    std::fprintf(stdout, "meiki: %s; pool_used=%u\n", stage,                   \
                 (unsigned)meiki_arena_used())
#else
#define TRACE(stage) ((void)0)
#endif
#include <memory>
#if defined(__vita__)
#include <psp2/kernel/threadmgr.h>
static uint64_t microseconds() { return sceKernelGetSystemTimeWide(); }
#else
#include <chrono>
static uint64_t microseconds() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}
#endif

extern "C" int MNNEmbeddingExceptionSupport();
extern "C" void MNNEmbeddingResetFailures();
extern "C" int MNNEmbeddingAllocationFailed();

// Link MNN's static library with --wrap for these three functions. This
// covers its source model, packed weights, activations and host tensors.
extern "C" void *__wrap_MNNMemoryAllocAlign(size_t bytes, size_t alignment) {
    if (MNNEmbeddingAllocationFailed() || !alignment || alignment > 64 ||
        (alignment & (alignment - 1)))
        throw std::bad_alloc();
    void *memory = meiki_arena_alloc(bytes);
    // Several MNN CPU operations dereference allocation results unchecked.
    // These static-library callers are C++; unwind before they see NULL.
    if (!memory)
        throw std::bad_alloc();
    return memory;
}
extern "C" void *__wrap_MNNMemoryCallocAlign(size_t bytes, size_t alignment) {
    void *p = __wrap_MNNMemoryAllocAlign(bytes, alignment);
    if (p)
        std::memset(p, 0, bytes);
    return p;
}
extern "C" void __wrap_MNNMemoryFreeAlign(void *pointer) {
    meiki_arena_free(pointer);
}

static bool tensor_matches(const MNN::Tensor *tensor, halide_type_code_t code,
                           unsigned bits, std::initializer_list<int> shape) {
    if (!tensor || tensor->getType().code != code ||
        tensor->getType().bits != bits || tensor->getType().lanes != 1 ||
        tensor->dimensions() != (int)shape.size())
        return false;
    int dimension = 0;
    for (int extent : shape)
        if (tensor->length(dimension++) != extent)
            return false;
    return true;
}

// Keep the arena active until execute's interpreter and tensors have unwound.
// A C++ allocation failure must not leave the next serialized job locked out.
class ArenaScope {
public:
    explicit ArenaScope(MeikiStats *stats) : stats_(stats) {}
    ~ArenaScope() {
        stats_->heap_peak = meiki_arena_peak();
        stats_->heap_remaining = meiki_arena_used();
        stats_->allocation_failed |= meiki_arena_failed() ||
                                     MNNEmbeddingAllocationFailed();
        meiki_arena_end();
    }
private:
    MeikiStats *stats_;
};

static int execute(const char *path, const float *data, MeikiOutput *out,
                   MeikiStats *stats) {
    TRACE("loading model");
    uint64_t started = microseconds();
    std::unique_ptr<MNN::Interpreter> net(
        MNN::Interpreter::createFromFile(path));
    if (!net)
        return -2;
    TRACE("creating session");
    MNN::BackendConfig backend;
    backend.memory = MNN::BackendConfig::Memory_Low;
    backend.precision = MNN::BackendConfig::Precision_Normal;
    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 1;
    config.backendConfig = &backend;
    auto *session = net->createSession(config);
    if (meiki_arena_failed() || MNNEmbeddingAllocationFailed())
        return -10;
    if (!session)
        return -3;
    int resize_status = -1;
    if (!net->getSessionInfo(session, MNN::Interpreter::RESIZE_STATUS,
                            &resize_status) || resize_status != 0)
        return -3;
    TRACE("session ready");
    auto *input = net->getSessionInput(session, "images");
    auto *codes = net->getSessionOutput(session, "char_codes");
    auto *boxes = net->getSessionOutput(session, "boxes");
    auto *scores = net->getSessionOutput(session, "scores");
    if (!tensor_matches(input, halide_type_float, 32, {1, 3, 32, 960}) ||
        !tensor_matches(codes, halide_type_int, 32, {1, 48}) ||
        !tensor_matches(boxes, halide_type_float, 32, {1, 48, 4}) ||
        !tensor_matches(scores, halide_type_float, 32, {1, 48}) ||
        !input->host<float>())
        return -4;
    net->releaseModel();
    TRACE("source model released");
    net->getSessionInfo(session, MNN::Interpreter::MEMORY,
                        &stats->session_memory_mib);
    MNN::Tensor host_input(input, MNN::Tensor::CAFFE);
    if (!host_input.host<float>())
        return -5;
    std::memcpy(host_input.host<float>(), data,
                MEIKI_INPUT_ELEMENTS * sizeof(float));
    if (!input->copyFromHostTensor(&host_input))
        return -6;
    stats->load_us = microseconds() - started;
    started = microseconds();
    TRACE("running inference");
    if (net->runSession(session) != MNN::NO_ERROR)
        return -7;
    stats->inference_us = microseconds() - started;
    TRACE("inference complete");
    MNN::Tensor hc(codes, codes->getDimensionType());
    MNN::Tensor hb(boxes, boxes->getDimensionType());
    MNN::Tensor hs(scores, scores->getDimensionType());
    if (!hc.host<int32_t>() || !hb.host<float>() || !hs.host<float>() ||
        !codes->host<int32_t>() || !boxes->host<float>() ||
        !scores->host<float>())
        return -8;
    if (!codes->copyToHostTensor(&hc) || !boxes->copyToHostTensor(&hb) ||
        !scores->copyToHostTensor(&hs))
        return -8;
    if (!tensor_matches(&hc, halide_type_int, 32, {1, 48}) ||
        !tensor_matches(&hb, halide_type_float, 32, {1, 48, 4}) ||
        !tensor_matches(&hs, halide_type_float, 32, {1, 48}))
        return -9;
    std::memcpy(out->codes, hc.host<int32_t>(), sizeof(out->codes));
    std::memcpy(out->boxes, hb.host<float>(), sizeof(out->boxes));
    std::memcpy(out->scores, hs.host<float>(), sizeof(out->scores));
    return 0;
}

extern "C" int meiki_run(const char *path, const float *data, void *workspace,
                         size_t bytes, MeikiOutput *out, MeikiStats *stats) {
    if (!path || !data || !out || !stats)
        return -1;
    std::memset(out, 0, sizeof(*out));
    std::memset(stats, 0, sizeof(*stats));
    if (!MNNEmbeddingExceptionSupport())
        return -13;
    if (bytes < MEIKI_WORKSPACE_BYTES)
        return -1;
    if (meiki_arena_begin(workspace, bytes) < 0)
        return -1;
    MNNEmbeddingResetFailures();
    int rc;
    {
        ArenaScope arena_scope(stats);
        try {
            rc = execute(path, data, out, stats);
        } catch (const std::bad_alloc &) {
            stats->allocation_failed = 1;
            rc = -10;
        } catch (...) {
            rc = -11;
        }
    }
    if (stats->allocation_failed)
        return -10;
    return stats->heap_remaining ? -12 : rc;
}
