// Actual pinned static MNN path/reader equivalence and failure recovery; host only.
#include "runtime.h"
#include "decode.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

static bool reject_path;
static unsigned path_calls;
extern "C" void *__real__ZN3MNN11Interpreter14createFromFileEPKc(const char *);
extern "C" void *__wrap__ZN3MNN11Interpreter14createFromFileEPKc(const char *path) {
    ++path_calls;
    if (reject_path) std::abort();
    return __real__ZN3MNN11Interpreter14createFromFileEPKc(path);
}

struct Source {
    FILE *file;
    uint32_t fail_offset;
    int error, exception;
    size_t calls, bytes;
};
static int read_model(void *context, uint32_t offset, void *target, uint32_t bytes) {
    auto *source = static_cast<Source *>(context);
    if (!bytes || bytes > MEIKI_MODEL_READ_CHUNK || offset > MEIKI_MODEL_BYTES ||
        bytes > MEIKI_MODEL_BYTES - offset || offset != source->bytes) std::abort();
    ++source->calls;
    if (offset == source->fail_offset) {
        if (source->exception == 1) throw std::bad_alloc();
        if (source->exception == 2) throw 17;
        return source->error;
    }
    if (std::fseek(source->file, offset, SEEK_SET) ||
        std::fread(target, 1, bytes, source->file) != bytes) return -19;
    source->bytes += bytes;
    return 0;
}
static float input[MEIKI_INPUT_ELEMENTS];
int main(int argc, char **argv) {
    if (argc < 3) return 2; // MODEL INPUT_F32...
    bool matrix = !std::strcmp(argv[1], "--matrix");
    int first_fixture = matrix ? 3 : 2;
    const char *model_path = argv[matrix ? 2 : 1];
    FILE *model = std::fopen(model_path, "rb");
    if (!model || std::fseek(model, 0, SEEK_END) || std::ftell(model) != MEIKI_MODEL_BYTES) return 3;
    void *pool = nullptr;
    if (posix_memalign(&pool, 64, MEIKI_WORKSPACE_BYTES)) return 3;
    size_t path_peak = 0, reader_peak = 0, retained = 0, successes = 0, errors = 0;
    unsigned raw_differences = 0, decoded_differences = 0;
    for (int fixture = first_fixture; fixture < argc; ++fixture) {
        unsigned crop_width = 960, effective_width = 960;
        char *dimensions = std::strchr(argv[fixture], '\t');
        if (dimensions) {
            *dimensions++ = 0;
            if (std::sscanf(dimensions, "%u\t%u", &crop_width, &effective_width) != 2 ||
                !crop_width || !effective_width) return 2;
        }
        FILE *file = std::fopen(argv[fixture], "rb");
        if (!file || std::fread(input, 1, sizeof(input), file) != sizeof(input) || std::fgetc(file) != EOF) return 3;
        std::fclose(file);
        MeikiOutput baseline, output;
        MeikiStats stats;
        reject_path = false;
        if (meiki_run(model_path, input, pool, MEIKI_WORKSPACE_BYTES, &baseline, &stats) || stats.heap_remaining) return 4;
        path_peak = std::max(path_peak, stats.heap_peak);
        if (matrix) {
            if (meiki_run(model_path, input, pool, MEIKI_WORKSPACE_BYTES, &output, &stats) ||
                stats.heap_remaining || std::memcmp(&baseline, &output, sizeof(output))) return 4;
            path_peak = std::max(path_peak, stats.heap_peak);
        }
        reject_path = true;
        for (unsigned repetition = 0; repetition < (matrix ? 2u : 3u); ++repetition) {
            Source source = {model, UINT32_MAX, 0, 0, 0, 0};
            MeikiModelReader reader = {sizeof(reader), MEIKI_MODEL_BYTES, &source, read_model};
            if (meiki_run_reader(&reader, input, pool, MEIKI_WORKSPACE_BYTES, &output, &stats) ||
                stats.heap_remaining || stats.model_read_bytes != MEIKI_MODEL_BYTES ||
                stats.model_read_error || source.calls != 659) return 5;
            reader_peak = std::max(reader_peak, stats.heap_peak);
            raw_differences += std::memcmp(&baseline, &output, sizeof(output)) != 0;
            char before[193], after[193];
            if (meiki_decode(&baseline, crop_width, effective_width, before, sizeof(before)) < 0 ||
                meiki_decode(&output, crop_width, effective_width, after, sizeof(after)) < 0) return 5;
            decoded_differences += std::strcmp(before, after) != 0;
            ++successes;
        }
        for (unsigned failure = 0; failure < (matrix ? 0u : 6u); ++failure) {
            uint32_t offsets[] = {0, 8192, 5390336, 8192, 8192, 8192};
            int errors_[] = {-19, -17, -5, 37, -1, -1};
            Source source = {model, offsets[failure], errors_[failure], failure == 4 ? 1 : failure == 5 ? 2 : 0, 0, 0};
            MeikiModelReader reader = {sizeof(reader), MEIKI_MODEL_BYTES, &source, read_model};
            int rc = meiki_run_reader(&reader, input, pool, MEIKI_WORKSPACE_BYTES, &output, &stats);
            int expected = failure == 4 ? -10 : failure == 5 ? -11 : -14;
            if (rc != expected || stats.heap_remaining || source.bytes != offsets[failure] ||
                (failure < 4 && stats.model_read_error != errors_[failure])) return 6;
            retained = std::max(retained, stats.heap_remaining);
            ++errors;
            source = {model, UINT32_MAX, 0, 0, 0, 0};
            if (meiki_run_reader(&reader, input, pool, MEIKI_WORKSPACE_BYTES, &output, &stats) ||
                stats.heap_remaining || std::memcmp(&baseline, &output, sizeof(output))) return 7;
            ++successes;
        }
        if (matrix) {
            char decoded[193];
            if (meiki_decode(&output, crop_width, effective_width, decoded, sizeof(decoded)) < 0) return 5;
            const char *name = std::strrchr(argv[fixture], '/');
            std::printf("{\"fixture\":\"%s\",\"text_hex\":\"", name ? name + 1 : argv[fixture]);
            for (const unsigned char *p = (const unsigned char *)decoded; *p; ++p) std::printf("%02x", *p);
            std::printf("\"}\n");
            std::fflush(stdout);
        }
    }
    std::printf("{\"fixtures\":%d,\"successful_reader_calls\":%zu,\"failure_cases\":%zu,"
                "\"path_calls\":%u,\"path_heap_peak\":%zu,\"reader_heap_peak\":%zu,"
                "\"maximum_retained\":%zu,\"raw_output_differences\":%u,"
                "\"decoded_output_differences\":%u}\n", argc - first_fixture, successes, errors,
                path_calls, path_peak, reader_peak, retained, raw_differences, decoded_differences);
    std::fclose(model);
    std::free(pool);
    return raw_differences || decoded_differences ? 8 : 0;
}
