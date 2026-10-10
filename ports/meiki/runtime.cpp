#include "runtime.h"
#include "arena.h"
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
#include <cstring>
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

// Link MNN's static library with --wrap for these three functions. This
// covers its source model, packed weights, activations and host tensors.
extern "C" void *__wrap_MNNMemoryAllocAlign(size_t bytes, size_t alignment) {
    if (!alignment || alignment > 64 || (alignment & (alignment - 1)))
        return nullptr;
    return meiki_arena_alloc(bytes);
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
    if (!session)
        return -3;
    TRACE("session ready");
    auto *input = net->getSessionInput(session, "images");
    auto *codes = net->getSessionOutput(session, "char_codes");
    auto *boxes = net->getSessionOutput(session, "boxes");
    auto *scores = net->getSessionOutput(session, "scores");
    if (!input || !codes || !boxes || !scores ||
        input->elementSize() != MEIKI_INPUT_ELEMENTS)
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
    if (!codes->copyToHostTensor(&hc) || !boxes->copyToHostTensor(&hb) ||
        !scores->copyToHostTensor(&hs))
        return -8;
    if (hc.elementSize() != 48 || hb.elementSize() != 192 ||
        hs.elementSize() != 48 || hc.getType().code != halide_type_int ||
        hc.getType().bits != 32 || hb.getType().code != halide_type_float ||
        hb.getType().bits != 32 || hs.getType().code != halide_type_float ||
        hs.getType().bits != 32)
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
    if (bytes < MEIKI_WORKSPACE_BYTES)
        return -1;
    if (meiki_arena_begin(workspace, bytes) < 0)
        return -1;
    int rc = execute(path, data, out, stats);
    stats->heap_peak = meiki_arena_peak();
    stats->heap_remaining = meiki_arena_used();
    stats->allocation_failed = meiki_arena_failed();
    meiki_arena_end();
    return stats->allocation_failed ? -10 : rc;
}
