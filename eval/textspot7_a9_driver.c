/* A9 driver of textspot7's cost rows (eval/textspot7_a9.py): Rubi's finder as the pinned page path reads a region
 * (eval/pagepath_find.c of textspot6's pinned snapshot, included here with its main renamed), and textspot7's
 * post-pass and fallback (runtime/vocr_textspot7.c).
 *
 *   textspot7_a9_driver --empty|--grad BATCH       BATCH: pagepath_find.c's records (8 x u32, then the grey region)
 *   textspot7_a9_driver --empty|--t7 BATCH         BATCH: vocr_textspot7_test.c's records (choose / post)
 *
 * --empty reads every record and runs nothing (the same length of input, so the counts of the I/O cancel);
 * --grad runs pagepath_find.c's main on the batch; --t7 runs each record through vocr_textspot7.c.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* The driver is built by the Vita compiler (so the pinned page path's own code, Rubi's short-line probe among it, is
 * the Vita's machine code) and linked against the ARM Linux toolchain's C library: newlib's stderr macro reads a
 * newlib global, so stderr comes from a host-built shim (eval/textspot7_a9_shim.c). */
#undef stderr
extern void *t7_host_stderr(void);
#define stderr ((FILE *)t7_host_stderr())

#define main pagepath_find_main
#include "pagepath_find.c"
#undef main

#include "vocr_textspot7.h"

static unsigned char *slurp(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    unsigned char *b;
    long sz;
    if (!f)
        return NULL;
    fseek(f, 0, SEEK_END);
    sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    b = (unsigned char *)malloc((size_t)sz + 1u);
    if (b && fread(b, 1, (size_t)sz, f) != (size_t)sz) {
        free(b);
        b = NULL;
    }
    fclose(f);
    *n = (size_t)sz;
    return b;
}

static int i32at(const unsigned char *p)
{
    return (int)((unsigned)p[0] | ((unsigned)p[1] << 8) | ((unsigned)p[2] << 16) | ((unsigned)p[3] << 24));
}

static unsigned char t7_arena[1u << 21];

/* Walks vocr_textspot7_test.c's records; run = 0 only reads them. */
static int t7_batch(const unsigned char *b, size_t n, int run)
{
    size_t off = 0;
    unsigned recs = 0;
    while (off + 4u <= n) {
        int kind = i32at(b + off);
        off += 4u;
        if (kind == 1) {
            VocrT7Box s[VOCR_T7_MAX_BOXES], g[VOCR_T7_MAX_BOXES];
            VocrT7Params p;
            int ns, ng, i, rc = 0;
            if (off + 24u > n)
                return 3;
            p.mode = (unsigned)i32at(b + off);
            p.tau = i32at(b + off + 4);
            p.lam = i32at(b + off + 8);
            p.kappa = i32at(b + off + 12);
            ns = i32at(b + off + 16);
            ng = i32at(b + off + 20);
            off += 24u;
            if (ns < 0 || ng < 0 || ns > (int)VOCR_T7_MAX_BOXES || ng > (int)VOCR_T7_MAX_BOXES ||
                off + 20u * (size_t)ns + 16u * (size_t)ng > n)
                return 3;
            for (i = 0; i < ns; i++, off += 20u) {
                s[i].x = i32at(b + off); s[i].y = i32at(b + off + 4); s[i].w = i32at(b + off + 8);
                s[i].h = i32at(b + off + 12); s[i].conf = i32at(b + off + 16); s[i].dark = 0;
            }
            for (i = 0; i < ng; i++, off += 16u) {
                g[i].x = i32at(b + off); g[i].y = i32at(b + off + 4); g[i].w = i32at(b + off + 8);
                g[i].h = i32at(b + off + 12); g[i].conf = 0; g[i].dark = 0;
            }
            if (run)
                rc = vocr_textspot7_choose(s, (unsigned)ns, g, (unsigned)ng, &p);
            printf("C %d\n", rc);
        } else if (kind == 2) {
            VocrT7Box fb[VOCR_T7_MAX_BOXES], out[VOCR_T7_MAX_BOXES];
            VocrLinesImage im;
            unsigned steps, m = 0;
            int x0, y0, x1, y1, nb, i, rc = 0;
            if (off + 24u > n)
                return 3;
            steps = (unsigned)i32at(b + off);
            x0 = i32at(b + off + 4); y0 = i32at(b + off + 8); x1 = i32at(b + off + 12); y1 = i32at(b + off + 16);
            nb = i32at(b + off + 20);
            off += 24u;
            if (nb < 0 || nb > (int)VOCR_T7_MAX_BOXES || off + 24u * (size_t)nb > n)
                return 3;
            for (i = 0; i < nb; i++, off += 24u) {
                fb[i].x = i32at(b + off); fb[i].y = i32at(b + off + 4); fb[i].w = i32at(b + off + 8);
                fb[i].h = i32at(b + off + 12); fb[i].conf = i32at(b + off + 16);
                fb[i].dark = (unsigned)i32at(b + off + 20);
            }
            memset(&im, 0, sizeof im);
            if (steps & (VOCR_T7_EXTEND | VOCR_T7_TIGHTEN)) {
                size_t px = (size_t)(x1 - x0) * (size_t)(y1 - y0), padded = (px + 3u) & ~(size_t)3u;
                if (off + padded > n)
                    return 3;
                im.pixels = b + off;
                im.width = (unsigned)(x1 - x0);
                im.height = (unsigned)(y1 - y0);
                im.stride = im.width;
                im.bpp = VOCR_LINES_GREY;
                off += padded;
            }
            if (run)
                rc = vocr_textspot7_post(fb, (unsigned)nb, x0, y0, x1, y1,
                                         (steps & (VOCR_T7_EXTEND | VOCR_T7_TIGHTEN)) ? &im : NULL, steps,
                                         t7_arena, sizeof t7_arena, out, &m);
            printf("P %d %u\n", rc, m);
        } else {
            return 4;
        }
        recs++;
    }
    return recs ? 0 : 5;
}

/* Walks pagepath_find.c's records without running the finder (--empty for --grad). */
static int grad_empty(const unsigned char *b, size_t n)
{
    size_t off = 0;
    while (off + 32u <= n) {
        unsigned w = rd32(b + off), h = rd32(b + off + 4);
        off += 32u;
        if (!w || !h || off + (size_t)w * h > n)
            return 2;
        printf("0 0 0 0\n");
        off += (size_t)w * h;
    }
    return 0;
}

int main(int argc, char **argv)
{
    size_t n = 0;
    unsigned char *b;
    int rc;
    if (argc != 3) {
        fprintf(stderr, "usage: %s --empty|--grad|--t7 BATCH (--empty-t7 for the t7 records)\n", argv[0]);
        return 2;
    }
    if (!strcmp(argv[1], "--grad")) {
        char *a[2];
        a[0] = argv[0];
        a[1] = argv[2];
        return pagepath_find_main(2, a);
    }
    b = slurp(argv[2], &n);
    if (!b)
        return 2;
    if (!strcmp(argv[1], "--empty"))
        rc = grad_empty(b, n);
    else if (!strcmp(argv[1], "--t7"))
        rc = t7_batch(b, n, 1);
    else if (!strcmp(argv[1], "--empty-t7"))
        rc = t7_batch(b, n, 0);
    else
        rc = 2;
    free(b);
    return rc;
}
