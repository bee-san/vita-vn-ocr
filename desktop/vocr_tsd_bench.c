/* vocr_tsd_bench.c - time textspot-desktop's C detector (vocr_tsd.h) on frames in memory, for profiling.
 *
 *   vocr_tsd_bench BLOB.vtsd [--w 960] [--h 544] [--ch 3] [--raw FRAMES.rgb] [--runs 200] [--warmup 20]
 *                  [--threads 1] [--tasks-per-thread 1] [--size fit:960|long:N|native] [--seed 1] [--refine]
 *
 * The frames: --raw holds consecutive w x h x ch byte frames (as many as the file has), else one seeded random frame
 * (smoothed noise). Prints one JSON line: the detector's p50 / p95 / mean in us over the runs (each run one frame,
 * cycling), the boxes of the last run, and, in a -DVOCR_TSD_PROF build, the mean us of each profiled step. --refine: the
 * box refinement on (vocr_tsdr_defaults). */
#define _POSIX_C_SOURCE 200112L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "vocr_tsd.h"

#ifdef VOCR_TSD_PROF
extern double vocr_tsd_prof_us[];
extern unsigned vocr_tsd_prof_runs;
const char *vocr_tsd_prof_name(unsigned i);
void vocr_tsd_prof_reset(void);
#endif

static double now_us(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1e6 * (double)ts.tv_sec + 1e-3 * (double)ts.tv_nsec;
}

static int cmp_d(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static unsigned char *read_file(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    unsigned char *buf;
    long sz;
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf = (unsigned char *)malloc((size_t)sz + 64);
    if (buf && fread(buf, 1, (size_t)sz, f) != (size_t)sz) {
        free(buf);
        buf = NULL;
    }
    fclose(f);
    *n = (size_t)sz;
    return buf;
}

int main(int argc, char **argv)
{
    const char *blob_path = NULL, *raw = NULL, *size = "fit:960";
    unsigned w = 960, h = 544, ch = 3, runs = 200, warmup = 20, threads = 1, seed = 1, kind, n = 0, i, nframes;
    unsigned tasks_per = 1, ntasks, refine = 0;
    vocr_tsdr_opts ropts;
    size_t blob_n = 0, raw_n = 0, frame_bytes, arena_n;
    unsigned char *blob_raw, *blob, *frames;
    vocr_tsd_model m;
    vocr_tsd_set *set;
    vocr_tsd_pool *pool = NULL;
    void *arena;
    double *t;
    int rc, a;
    for (a = 1; a < argc; a++) {
        if (!strcmp(argv[a], "--w") && a + 1 < argc) w = (unsigned)atoi(argv[++a]);
        else if (!strcmp(argv[a], "--h") && a + 1 < argc) h = (unsigned)atoi(argv[++a]);
        else if (!strcmp(argv[a], "--ch") && a + 1 < argc) ch = (unsigned)atoi(argv[++a]);
        else if (!strcmp(argv[a], "--raw") && a + 1 < argc) raw = argv[++a];
        else if (!strcmp(argv[a], "--runs") && a + 1 < argc) runs = (unsigned)atoi(argv[++a]);
        else if (!strcmp(argv[a], "--warmup") && a + 1 < argc) warmup = (unsigned)atoi(argv[++a]);
        else if (!strcmp(argv[a], "--threads") && a + 1 < argc) threads = (unsigned)atoi(argv[++a]);
        else if (!strcmp(argv[a], "--size") && a + 1 < argc) size = argv[++a];
        else if (!strcmp(argv[a], "--seed") && a + 1 < argc) seed = (unsigned)atoi(argv[++a]);
        else if (!strcmp(argv[a], "--tasks-per-thread") && a + 1 < argc) tasks_per = (unsigned)atoi(argv[++a]);
        else if (!strcmp(argv[a], "--refine")) refine = 1;
        else if (!blob_path) blob_path = argv[a];
        else {
            fprintf(stderr, "unknown argument %s\n", argv[a]);
            return 2;
        }
    }
    if (!blob_path || !runs) {
        fprintf(stderr, "usage: vocr_tsd_bench BLOB.vtsd [--w W --h H --ch C --raw F --runs N --threads T --size S]\n");
        return 2;
    }
    if (!strcmp(size, "native")) {
        kind = VOCR_TSD_SIZE_NATIVE;
    } else if (!strncmp(size, "fit:", 4)) {
        kind = VOCR_TSD_SIZE_FIT;
        n = (unsigned)atoi(size + 4);
    } else if (!strncmp(size, "long:", 5)) {
        kind = VOCR_TSD_SIZE_LONG;
        n = (unsigned)atoi(size + 5);
    } else {
        fprintf(stderr, "size %s: fit:N, long:N or native\n", size);
        return 2;
    }
    blob_raw = read_file(blob_path, &blob_n);
    if (!blob_raw) {
        fprintf(stderr, "%s: cannot read\n", blob_path);
        return 1;
    }
    blob = (unsigned char *)(((size_t)blob_raw + 63u) & ~(size_t)63u);
    memmove(blob, blob_raw, blob_n);
    rc = vocr_tsd_load(&m, blob, blob_n);
    if (rc) {
        fprintf(stderr, "vocr_tsd_load: %d\n", rc);
        return 1;
    }
    frame_bytes = (size_t)w * h * ch;
    if (raw) {
        frames = read_file(raw, &raw_n);
        if (!frames || raw_n < frame_bytes) {
            fprintf(stderr, "%s: no %ux%ux%u frame\n", raw, w, h, ch);
            return 1;
        }
        nframes = (unsigned)(raw_n / frame_bytes);
    } else {
        unsigned x, y, c, s = seed * 2654435761u + 1u;
        frames = (unsigned char *)malloc(frame_bytes);
        nframes = 1;
        for (y = 0; y < h; y++)
            for (x = 0; x < w; x++)
                for (c = 0; c < ch; c++) {
                    s = s * 1103515245u + 12345u;
                    frames[((size_t)y * w + x) * ch + c] = (unsigned char)(((s >> 16) & 255u) / 2u + 64u * ((x / 37 + y / 23) & 1u));
                }
    }
    ntasks = threads > 1 ? tasks_per * threads : 1;
    vocr_tsdr_defaults(&ropts);
    arena_n = vocr_tsd_arena_bytes_ex(&m, kind, n, w, h, ch, ntasks, (int)refine);
    arena = malloc(arena_n);
    set = (vocr_tsd_set *)malloc(sizeof *set);
    t = (double *)malloc(sizeof(double) * runs);
    if (!arena || !set || !t) return 1;
    memset(arena, 0, arena_n);
    if (threads > 1) {
        pool = vocr_tsd_pool_new(threads);
        if (!pool) {
            fprintf(stderr, "vocr_tsd_pool_new failed\n");
            return 1;
        }
    }
    for (i = 0; i < warmup + runs; i++) {
        const unsigned char *fr = frames + (size_t)(i % nframes) * frame_bytes;
        double t0;
#ifdef VOCR_TSD_PROF
        if (i == warmup) vocr_tsd_prof_reset();
#endif
        t0 = now_us();
        rc = vocr_tsd_detect_ex(&m, fr, w, h, w * ch, ch, kind, n, refine ? &ropts : NULL, arena, arena_n, set,
                                pool ? vocr_tsd_pool_run : NULL, pool, ntasks);
        if (rc) {
            fprintf(stderr, "vocr_tsd_detect: %d\n", rc);
            return 1;
        }
        if (i >= warmup) t[i - warmup] = now_us() - t0;
    }
    {
        double mean = 0.0;
        for (i = 0; i < runs; i++) mean += t[i];
        mean /= runs;
        qsort(t, runs, sizeof(double), cmp_d);
        printf("{\"blob\": \"%s\", \"w\": %u, \"h\": %u, \"ch\": %u, \"size\": \"%s\", \"threads\": %u, \"tasks\": %u, \"simd\": %d, "
               "\"frames\": %u, \"runs\": %u, \"p50_us\": %.1f, \"p95_us\": %.1f, \"mean_us\": %.1f, \"min_us\": %.1f, "
               "\"boxes\": %u, \"refine\": %u, \"arena_bytes\": %zu",
               blob_path, w, h, ch, size, threads, ntasks, vocr_tsd_simd(), nframes, runs, t[runs / 2], t[runs * 95 / 100],
               mean, t[0], set->count, refine, arena_n);
#ifdef VOCR_TSD_PROF
        {
            unsigned k, first = 1;
            printf(", \"prof_us\": {");
            for (k = 0; k < 64; k++) {
                const char *nm = vocr_tsd_prof_name(k);
                if (!nm) continue;
                printf("%s\"%s\": %.2f", first ? "" : ", ", nm, vocr_tsd_prof_us[k] / (vocr_tsd_prof_runs ? vocr_tsd_prof_runs : 1));
                first = 0;
            }
            printf("}");
        }
#endif
        printf("}\n");
    }
    vocr_tsd_pool_free(pool);
    free(t);
    free(set);
    free(arena);
    free(frames);
    free(blob_raw);
    return 0;
}
