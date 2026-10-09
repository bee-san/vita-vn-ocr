/* cachesim - how many Allegrex D-cache misses does vocr_run() cause per CTC frame?
 *
 *   make -C psp cachesim            (host gcc; build/cachesim)
 *   build/cachesim MODEL.vocr --mode f32|w8f|w8i [--tile N --head N] --pack LINES.bin [--n N]
 *
 * runtime/vocr.c is compiled for the host with gcc's KASAN instrumentation in callback mode
 * (-fsanitize=kernel-address --param asan-instrumentation-with-call-threshold=0), which calls
 * __asan_loadN/__asan_storeN with the address of every load and store the optimised code does
 * to memory (not register spills, which hit the stack's cached lines). This file implements
 * those callbacks as a cache model, and memcpy / memset (wrapped with -Wl,--wrap) as one access
 * per cache line they touch. Only vocr_run() is measured. The PSP build runs the same scalar
 * kernels, so this is the algorithmic data traffic of the PSP scalar build.
 *
 * The cache: 16 KB, 64-byte lines (the Allegrex D-cache: 16 KB data cache, 64-byte lines),
 * write-allocate, write-back, LRU, simulated direct-mapped, 2-way and 4-way at once because
 * the associativity is not documented anywhere I could read. Every miss is one 64-byte line
 * fill from main memory; writebacks of dirty lines are counted too. What a miss costs in CPU
 * cycles is not modelled here (psp/README.md works with a range).
 */
#define _POSIX_C_SOURCE 200112L
#include "../runtime/vocr.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define LINE 64u
#define CACHE_BYTES 16384u
#define N_CFG 3

typedef struct {
    unsigned ways, sets;
    uint64_t *tag;  /* [sets][ways]: line address + 1 (0 = empty) */
    uint8_t *dirty; /* [sets][ways] */
    uint32_t *age;  /* [sets][ways]: LRU stamp */
    uint32_t clock;
    unsigned long long access, miss, wback;
} cache_t;

static cache_t g_cache[N_CFG];
static int g_on;
static unsigned long long g_bytes_load, g_bytes_store;

static void cache_init(cache_t *c, unsigned ways)
{
    memset(c, 0, sizeof *c);
    c->ways = ways;
    c->sets = CACHE_BYTES / LINE / ways;
    c->tag = (uint64_t *)calloc((size_t)c->sets * ways, sizeof *c->tag);
    c->dirty = (uint8_t *)calloc((size_t)c->sets * ways, 1);
    c->age = (uint32_t *)calloc((size_t)c->sets * ways, sizeof *c->age);
    if (!c->tag || !c->dirty || !c->age) {
        fprintf(stderr, "out of memory\n");
        exit(1);
    }
}

static void cache_touch(cache_t *c, uint64_t line, int store)
{
    const unsigned set = (unsigned)(line % c->sets);
    uint64_t *t = c->tag + (size_t)set * c->ways;
    uint8_t *d = c->dirty + (size_t)set * c->ways;
    uint32_t *a = c->age + (size_t)set * c->ways;
    unsigned w, victim = 0;
    c->access++;
    c->clock++;
    for (w = 0; w < c->ways; w++)
        if (t[w] == line + 1) {
            a[w] = c->clock;
            d[w] |= (uint8_t)store;
            return;
        }
    c->miss++;
    for (w = 0; w < c->ways; w++) { /* an empty way, else the least recently used */
        if (!t[w]) {
            victim = w;
            break;
        }
        if (a[w] < a[victim]) victim = w;
    }
    if (t[victim] && d[victim]) c->wback++;
    t[victim] = line + 1;
    d[victim] = (uint8_t)store;
    a[victim] = c->clock;
}

static void touch(uintptr_t addr, size_t n, int store)
{
    uint64_t l0, l1, l;
    int i;
    if (!g_on || !n) return;
    if (store) g_bytes_store += n;
    else g_bytes_load += n;
    l0 = (uint64_t)addr / LINE;
    l1 = (uint64_t)(addr + n - 1) / LINE;
    for (l = l0; l <= l1; l++)
        for (i = 0; i < N_CFG; i++) cache_touch(&g_cache[i], l, store);
}

/* gcc's KASAN callbacks (outline instrumentation) */
#define CB(N)                                                                        \
    void __asan_load##N##_noabort(uintptr_t a);                                      \
    void __asan_store##N##_noabort(uintptr_t a);                                     \
    void __asan_load##N##_noabort(uintptr_t a) { touch(a, N, 0); }                   \
    void __asan_store##N##_noabort(uintptr_t a) { touch(a, N, 1); }
CB(1)
CB(2)
CB(4)
CB(8)
CB(16)
void __asan_loadN_noabort(uintptr_t a, size_t n);
void __asan_storeN_noabort(uintptr_t a, size_t n);
void __asan_loadN_noabort(uintptr_t a, size_t n) { touch(a, n, 0); }
void __asan_storeN_noabort(uintptr_t a, size_t n) { touch(a, n, 1); }
void __asan_handle_no_return(void);
void __asan_handle_no_return(void) {}

/* memcpy / memset from vocr.c (and from here: only counted while g_on) */
void *__real_memcpy(void *d, const void *s, size_t n);
void *__real_memset(void *d, int v, size_t n);
void *__wrap_memcpy(void *d, const void *s, size_t n);
void *__wrap_memset(void *d, int v, size_t n);
void *__wrap_memcpy(void *d, const void *s, size_t n)
{
    touch((uintptr_t)s, n, 0);
    touch((uintptr_t)d, n, 1);
    return __real_memcpy(d, s, n);
}
void *__wrap_memset(void *d, int v, size_t n)
{
    touch((uintptr_t)d, n, 1);
    return __real_memset(d, v, n);
}

static void *read_file(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    void *buf = NULL;
    long len;
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) || (len = ftell(f)) < 0 || fseek(f, 0, SEEK_SET) ||
        posix_memalign(&buf, 64, (size_t)len + 64) || fread(buf, 1, (size_t)len, f) != (size_t)len) {
        fclose(f);
        free(buf);
        return NULL;
    }
    fclose(f);
    *n = (size_t)len;
    return buf;
}

static unsigned rd32(const unsigned char *p)
{
    return (unsigned)p[0] | (unsigned)p[1] << 8 | (unsigned)p[2] << 16 | (unsigned)p[3] << 24;
}

int main(int argc, char **argv)
{
    vocr_model m;
    vocr_ctx ctx;
    vocr_opts o;
    size_t blob_n = 0, pack_n = 0, off = 4, need;
    unsigned char *blob, *pack;
    void *arena = NULL;
    const char *mode = "auto", *pack_path = NULL;
    unsigned count, j, limit = 0xFFFFFFFFu, ways[N_CFG] = {1, 2, 4};
    unsigned long long frames = 0;
    char text[4096];
    int i, rc;
    if (argc < 2) {
        fprintf(stderr, "usage: %s MODEL.vocr --mode M [--tile N --head N] --pack LINES.bin [--n N]\n", argv[0]);
        return 2;
    }
    memset(&o, 0, sizeof o);
    for (i = 2; i + 1 < argc; i += 2) {
        if (!strcmp(argv[i], "--mode")) mode = argv[i + 1];
        else if (!strcmp(argv[i], "--tile")) o.tile_frames = (unsigned)atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--head")) o.head_frames = (unsigned)atoi(argv[i + 1]);
        else if (!strcmp(argv[i], "--pack")) pack_path = argv[i + 1];
        else if (!strcmp(argv[i], "--n")) limit = (unsigned)atoi(argv[i + 1]);
        else {
            fprintf(stderr, "unknown option %s\n", argv[i]);
            return 2;
        }
    }
    o.mode = !strcmp(mode, "f32") ? VOCR_MODE_F32 : !strcmp(mode, "w8f") ? VOCR_MODE_W8F
           : !strcmp(mode, "w8i") ? VOCR_MODE_W8I : VOCR_MODE_AUTO;
    blob = (unsigned char *)read_file(argv[1], &blob_n);
    pack = pack_path ? (unsigned char *)read_file(pack_path, &pack_n) : NULL;
    if (!blob || !pack || pack_n < 4) {
        fprintf(stderr, "cannot read the model or the pack\n");
        return 1;
    }
    if ((rc = vocr_load(&m, blob, blob_n)) || !(need = vocr_arena_bytes(&m, &o)) ||
        posix_memalign(&arena, 64, need) || (rc = vocr_init(&ctx, &m, &o, arena, need))) {
        fprintf(stderr, "vocr: %s\n", vocr_strerror(rc));
        return 1;
    }
    for (i = 0; i < N_CFG; i++) cache_init(&g_cache[i], ways[i]);
    count = rd32(pack);
    for (j = 0; j < count && j < limit; j++) {
        vocr_image img;
        vocr_extra ex;
        unsigned h = rd32(pack + off), w = rd32(pack + off + 4), ch = rd32(pack + off + 8);
        off += 12;
        img.pixels = pack + off;
        img.width = w;
        img.height = h;
        img.stride = (size_t)w * ch;
        img.format = ch == 4 ? VOCR_RGBA8 : VOCR_GREY8;
        memset(&ex, 0, sizeof ex);
        g_on = 1;
        rc = vocr_run(&ctx, &img, text, sizeof text, &ex);
        g_on = 0;
        if (rc && rc != VOCR_E_TEXT) {
            fprintf(stderr, "vocr_run: %s\n", vocr_strerror(rc));
            return 1;
        }
        frames += ex.frames;
        off += (size_t)h * w * ch;
    }
    printf("{\"model\": \"%s\", \"mode\": \"%s\", \"tile\": %u, \"head\": %u, \"arena\": %lu, \"lines\": %u, "
           "\"frames\": %llu, \"load_bytes\": %llu, \"store_bytes\": %llu, \"line_bytes\": %u, \"cache_bytes\": %u,\n"
           " \"configs\": [",
           argv[1], mode, ctx.tile, ctx.head, (unsigned long)need, j, frames, g_bytes_load, g_bytes_store, LINE,
           CACHE_BYTES);
    for (i = 0; i < N_CFG; i++)
        printf("%s{\"ways\": %u, \"accesses\": %llu, \"misses\": %llu, \"writebacks\": %llu, "
               "\"misses_per_frame\": %.1f, \"writebacks_per_frame\": %.1f}",
               i ? ", " : "", g_cache[i].ways, g_cache[i].access, g_cache[i].miss, g_cache[i].wback,
               (double)g_cache[i].miss / (double)(frames ? frames : 1),
               (double)g_cache[i].wback / (double)(frames ? frames : 1));
    printf("]}\n");
    free(arena);
    free(pack);
    free(blob);
    return 0;
}
