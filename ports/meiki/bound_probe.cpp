// Host build of the same bounded C binding linked into the Vita app.
#include "decode.h"
#include "runtime.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char **argv) {
    if (argc != 3)
        return 2;
    char *end = nullptr;
    unsigned long mib = std::strtoul(argv[2], &end, 10);
    if (!end || *end || !mib || mib > 64)
        return 2;
    size_t bytes = mib * 1024u * 1024u;
    void *pool = nullptr;
    if (posix_memalign(&pool, 64, bytes))
        return 3;
    float input[MEIKI_INPUT_ELEMENTS];
    MeikiOutput out;
    char line[4096];
    while (std::fgets(line, sizeof(line), stdin)) {
        char *newline = std::strchr(line, '\n');
        if (newline)
            *newline = 0;
        unsigned crop_width = 0, effective_width = 0;
        char *dimensions = std::strchr(line, '\t');
        if (dimensions) {
            *dimensions++ = 0;
            if (std::sscanf(dimensions, "%u\t%u", &crop_width,
                            &effective_width) != 2)
                return 4;
        }
        FILE *file = std::fopen(line, "rb");
        if (!file)
            return 5;
        bool exact =
            std::fread(input, 1, sizeof(input), file) == sizeof(input) &&
            std::fgetc(file) == EOF;
        std::fclose(file);
        if (!exact)
            return 6;
        MeikiStats stats;
        int rc = meiki_run(argv[1], input, pool, bytes, &out, &stats);
        char text[193] = {};
        int decoded = dimensions && !rc
                          ? meiki_decode(&out, crop_width, effective_width,
                                         text, sizeof(text))
                          : 0;
        std::printf("{\"ms\":%.6f,\"load_us\":%llu,\"rc\":%d,\"pool_bytes\":%"
                    "zu,\"heap_peak\":%zu,\"heap_remaining\":%zu,\"allocation_"
                    "failed\":%d,\"decoded\":%d,\"text_hex\":\"",
                    stats.inference_us / 1000.,
                    (unsigned long long)stats.load_us, rc, bytes,
                    stats.heap_peak, stats.heap_remaining,
                    stats.allocation_failed, decoded);
        for (const unsigned char *p = (const unsigned char *)text; *p; ++p)
            std::printf("%02x", *p);
        std::printf("\",\"codes\":[");
        for (int i = 0; i < 48; ++i)
            std::printf("%s%d", i ? "," : "", out.codes[i]);
        std::printf("],\"boxes\":[");
        for (int i = 0; i < 192; ++i)
            std::printf("%s%.9g", i ? "," : "", out.boxes[i]);
        std::printf("],\"scores\":[");
        for (int i = 0; i < 48; ++i)
            std::printf("%s%.9g", i ? "," : "", out.scores[i]);
        std::printf("]}\n");
        std::fflush(stdout);
        if (rc || decoded < 0 || stats.heap_remaining) {
            std::free(pool);
            return 7;
        }
    }
    std::free(pool);
    return std::ferror(stdin) ? 8 : 0;
}
