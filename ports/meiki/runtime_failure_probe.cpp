// Failure injection against the actual static MNN/C binding; host only.
// Link with --wrap=meiki_arena_alloc,--wrap=_Znwm,--wrap=_Znam.
#include "runtime.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

enum Fault { NONE, ARENA, BAD_ALLOC, OTHER_EXCEPTION, METADATA };
static Fault fault;
static size_t ordinal, aligned_calls, metadata_calls;
static bool injected;
static bool unsupported_backend;
struct InjectedException {};

extern "C" int __real_MNNEmbeddingExceptionSupport();
extern "C" int __wrap_MNNEmbeddingExceptionSupport() {
    return unsupported_backend ? 0 : __real_MNNEmbeddingExceptionSupport();
}

extern "C" void *__real_meiki_arena_alloc(size_t);
extern "C" void *__wrap_meiki_arena_alloc(size_t bytes) {
    ++aligned_calls;
    if (!injected && aligned_calls == ordinal) {
        if (fault == ARENA) {
            injected = true;
            return __real_meiki_arena_alloc(std::numeric_limits<size_t>::max());
        }
        if (fault == BAD_ALLOC) {
            injected = true;
            throw std::bad_alloc();
        }
        if (fault == OTHER_EXCEPTION) {
            injected = true;
            throw InjectedException();
        }
    }
    return __real_meiki_arena_alloc(bytes);
}

extern "C" void *__real__Znwm(size_t);
extern "C" void *__real__Znam(size_t);
static void inject_metadata() {
    ++metadata_calls;
    if (!injected && fault == METADATA && metadata_calls == ordinal) {
        injected = true;
        throw std::bad_alloc();
    }
}
extern "C" void *__wrap__Znwm(size_t bytes) {
    inject_metadata();
    return __real__Znwm(bytes);
}
extern "C" void *__wrap__Znam(size_t bytes) {
    inject_metadata();
    return __real__Znam(bytes);
}

static float input[MEIKI_INPUT_ELEMENTS];
static MeikiOutput baseline, output;

int main(int argc, char **argv) {
    // MODE GOOD_MODEL INPUT_F32 BAD_MODEL ORDINAL_OR_REPETITIONS
    if (argc != 6)
        return 2;
    FILE *source = std::fopen(argv[3], "rb");
    if (!source)
        return 3;
    bool exact = std::fread(input, 1, sizeof(input), source) == sizeof(input) &&
                 std::fgetc(source) == EOF;
    std::fclose(source);
    if (!exact)
        return 3;
    void *pool = nullptr;
    if (posix_memalign(&pool, 64, MEIKI_WORKSPACE_BYTES))
        return 3;
    MeikiStats stats;
    int rc = meiki_run(argv[2], input, pool, MEIKI_WORKSPACE_BYTES, &baseline,
                       &stats);
    if (rc || stats.heap_remaining)
        return 4;
    char *end;
    ordinal = std::strtoul(argv[5], &end, 10);
    if (*end || !ordinal)
        return 2;
    if (!std::strcmp(argv[1], "arena"))
        fault = ARENA;
    else if (!std::strcmp(argv[1], "exception"))
        fault = BAD_ALLOC;
    else if (!std::strcmp(argv[1], "other"))
        fault = OTHER_EXCEPTION;
    else if (!std::strcmp(argv[1], "metadata"))
        fault = METADATA;
    else if (std::strcmp(argv[1], "baseline") &&
             std::strcmp(argv[1], "missing") &&
             std::strcmp(argv[1], "invalid") &&
             std::strcmp(argv[1], "no-exceptions"))
        return 2;
    unsupported_backend = !std::strcmp(argv[1], "no-exceptions");
    aligned_calls = metadata_calls = 0;
    size_t repetitions = fault != NONE || unsupported_backend ||
                         !std::strcmp(argv[1], "baseline")
                             ? 1 : ordinal;
    int expected = fault == OTHER_EXCEPTION ? -11 : fault != NONE ? -10 : -2;
    if (!std::strcmp(argv[1], "baseline"))
        expected = 0;
    if (unsupported_backend)
        expected = -13;
    size_t maximum_retained = 0;
    int checked = 0;
    for (size_t i = 0; i < repetitions; ++i) {
        const char *path = expected == -2 ? argv[4] : argv[2];
        rc = meiki_run(path, input, pool, MEIKI_WORKSPACE_BYTES, &output, &stats);
        if (stats.heap_remaining > maximum_retained)
            maximum_retained = stats.heap_remaining;
        bool triggered = fault == NONE || injected;
        if (!triggered)
            expected = 0;
        if (rc != expected || (!rc && std::memcmp(&baseline, &output, sizeof(output)))) {
            std::fprintf(stderr, "unexpected failure: mode=%s ordinal=%zu "
                         "rc=%d expected=%d remaining=%zu\n", argv[1], ordinal,
                         rc, expected, stats.heap_remaining);
            return 5;
        }
        ++checked;
    }
    size_t measured_aligned = aligned_calls, measured_metadata = metadata_calls;
    fault = NONE;
    unsupported_backend = false;
    int retry = meiki_run(argv[2], input, pool, MEIKI_WORKSPACE_BYTES, &output,
                          &stats);
    bool recovered = !retry && !stats.heap_remaining &&
                     !std::memcmp(&baseline, &output, sizeof(output));
    std::printf("{\"mode\":\"%s\",\"ordinal\":%zu,\"checked\":%d,"
                "\"injected\":%s,\"rc\":%d,\"maximum_retained\":%zu,"
                "\"aligned_calls\":%zu,\"metadata_calls\":%zu,"
                "\"recovered\":%s}\n", argv[1], ordinal, checked,
                injected ? "true" : "false", rc, maximum_retained,
                measured_aligned, measured_metadata,
                recovered ? "true" : "false");
    std::free(pool);
    return recovered ? 0 : 6;
}
