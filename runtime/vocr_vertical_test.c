/* vocr_vertical_test.c - host driver of vocr_vertical.c (not part of the console build).
 *
 *   vocr_vertical_test REGION.pgm|ppm [--flags F] [--rows]   columns: "x y w h dark" per line, then "truncated T"
 *   vocr_vertical_test --rotate IN.pgm OUT.pgm               the crop turned 90 degrees anticlockwise
 *   vocr_vertical_test --arena W H                           vocr_columns_arena_bytes(W, H)
 *   vocr_vertical_test --selftest N                          N procedural regions: the in-memory and row forms
 *                                                            agree for both flag sets, the arena is exact (one
 *                                                            byte less is refused), a failing row callback gives
 *                                                            VOCR_LINES_E_SOURCE, bad arguments are refused, and
 *                                                            the rotation matches a direct transpose
 * Binary PGM (P5) and PPM (P6, read through the RGBA path), maxval 255.
 */
#include "vocr_vertical.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned char *read_pnm(const char *path, unsigned *w, unsigned *h, unsigned *bpp)
{
    FILE *f = fopen(path, "rb");
    char magic[3] = {0};
    unsigned maxv = 0, x;
    unsigned char *raw, *out;
    size_t n;
    if (!f) return NULL;
    if (fscanf(f, "%2s %u %u %u", magic, w, h, &maxv) != 4 || maxv != 255 || (magic[1] != '5' && magic[1] != '6')) {
        fclose(f);
        return NULL;
    }
    fgetc(f);
    *bpp = magic[1] == '5' ? 1u : 4u;
    n = (size_t)*w * *h * (magic[1] == '5' ? 1u : 3u);
    raw = (unsigned char *)malloc(n ? n : 1u);
    if (!raw || fread(raw, 1, n, f) != n) {
        free(raw);
        fclose(f);
        return NULL;
    }
    fclose(f);
    if (*bpp == 1u) return raw;
    out = (unsigned char *)malloc((size_t)*w * *h * 4u);
    if (!out) {
        free(raw);
        return NULL;
    }
    for (x = 0; x < *w * *h; x++) {
        out[4u * x] = raw[3u * x];
        out[4u * x + 1u] = raw[3u * x + 1u];
        out[4u * x + 2u] = raw[3u * x + 2u];
        out[4u * x + 3u] = 255u;
    }
    free(raw);
    return out;
}

typedef struct {
    const unsigned char *pixels;
    size_t stride;
    unsigned fail_at;          /* row index whose read fails (0: never) */
    unsigned calls;
} RowsUser;

static const unsigned char *row_cb(void *user, unsigned y)
{
    RowsUser *u = (RowsUser *)user;
    u->calls++;
    if (u->fail_at && y == u->fail_at) return NULL;
    return u->pixels + (size_t)y * u->stride;
}

static int run_one(const unsigned char *px, unsigned w, unsigned h, unsigned bpp, unsigned flags, int rows,
                   VocrLineSet *set)
{
    size_t need = vocr_columns_arena_bytes(w, h);
    unsigned char *arena = (unsigned char *)malloc(need);
    int rc;
    if (!arena) return -100;
    if (rows) {
        RowsUser u;
        VocrLinesRows src;
        u.pixels = px;
        u.stride = (size_t)w * bpp;
        u.fail_at = 0;
        u.calls = 0;
        src.width = w;
        src.height = h;
        src.bpp = bpp;
        src.row = row_cb;
        src.user = &u;
        rc = vocr_find_columns_rows(&src, flags, arena, need, set);
    } else {
        VocrLinesImage im;
        im.pixels = px;
        im.width = w;
        im.height = h;
        im.stride = (size_t)w * bpp;
        im.bpp = bpp;
        rc = vocr_find_columns(&im, flags, arena, need, set);
    }
    free(arena);
    return rc;
}

static unsigned long long rng_state = 88172645463325252ull;

static unsigned rnd(unsigned n)
{
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return n ? (unsigned)(rng_state % n) : 0u;
}

/* A procedural region: a background (flat, gradient or noise), then columns of glyph-like blobs (random strokes
 * in cells), sometimes furigana beside them and a horizontal line of strokes, in either polarity. */
static void procedural(unsigned char *g, unsigned w, unsigned h)
{
    unsigned x, y, kind = rnd(3), bg = rnd(256), fg, ncol, c;
    for (y = 0; y < h; y++)
        for (x = 0; x < w; x++) {
            unsigned v = kind == 0 ? bg : kind == 1 ? (bg + x / 4u + y / 3u) % 256u : (bg + rnd(40)) % 256u;
            g[(size_t)y * w + x] = (unsigned char)v;
        }
    fg = bg > 128u ? rnd(60) : 200u + rnd(56);
    ncol = 1u + rnd(6);
    for (c = 0; c < ncol; c++) {
        unsigned cw = 8u + rnd(24), pitch = cw + rnd(cw / 3u + 1u), x0 = rnd(w), y0 = rnd(h / 2u + 1u);
        unsigned nch = 1u + rnd(14), k;
        for (k = 0; k < nch; k++) {
            unsigned cy = y0 + k * pitch, s, strokes = 2u + rnd(5);
            for (s = 0; s < strokes; s++) {
                unsigned vert = rnd(2), len = 2u + rnd(cw), ox = rnd(cw), oy = rnd(cw), t;
                for (t = 0; t < len; t++) {
                    unsigned px = x0 + (vert ? ox : ox + t), py = cy + (vert ? oy + t : oy);
                    if (px < w && py < h) g[(size_t)py * w + px] = (unsigned char)fg;
                    if (px + 1u < w && py < h && rnd(2)) g[(size_t)py * w + px + 1u] = (unsigned char)fg;
                }
            }
        }
    }
}

static int same(const VocrLineSet *a, const VocrLineSet *b)
{
    unsigned i;
    if (a->count != b->count || a->truncated != b->truncated) return 0;
    for (i = 0; i < a->count; i++)
        if (a->lines[i].x != b->lines[i].x || a->lines[i].y != b->lines[i].y || a->lines[i].w != b->lines[i].w ||
            a->lines[i].h != b->lines[i].h || a->lines[i].dark != b->lines[i].dark)
            return 0;
    return 1;
}

static int selftest(unsigned cases)
{
    unsigned k, found = 0, fails = 0;
    for (k = 0; k < cases; k++) {
        unsigned w = 16u + rnd(500), h = 16u + rnd(400), f;
        unsigned char *g = (unsigned char *)malloc((size_t)w * h), *rgba = (unsigned char *)malloc((size_t)w * h * 4u);
        VocrLineSet a, b, c;
        size_t need = vocr_columns_arena_bytes(w, h), i;
        unsigned char *arena = (unsigned char *)malloc(need);
        VocrLinesImage im;
        if (!g || !rgba || !arena) return 2;
        procedural(g, w, h);
        for (i = 0; i < (size_t)w * h; i++) {
            rgba[4u * i] = rgba[4u * i + 1u] = rgba[4u * i + 2u] = g[i];
            rgba[4u * i + 3u] = 255u;
        }
        for (f = 0; f <= VOCR_COLUMNS_FLAGS; f++) {
            if (run_one(g, w, h, 1u, f, 0, &a) || run_one(g, w, h, 1u, f, 1, &b) || run_one(rgba, w, h, 4u, f, 0, &c) ||
                !same(&a, &b) || !same(&a, &c)) {
                fprintf(stderr, "case %u (%ux%u, flags %u): forms differ\n", k, w, h, f);
                fails++;
            }
            found += a.count;
        }
        im.pixels = g;
        im.width = w;
        im.height = h;
        im.stride = w;
        im.bpp = 1u;
        if (vocr_find_columns(&im, 0, arena, need - 1u, &a) != VOCR_LINES_E_ARENA || a.count) {
            fprintf(stderr, "case %u: a short arena was not refused\n", k);
            fails++;
        }
        if (vocr_find_columns(&im, 0x80u, arena, need, &a) != VOCR_LINES_E_ARG) {
            fprintf(stderr, "case %u: unknown flags were not refused\n", k);
            fails++;
        }
        {
            RowsUser u;
            VocrLinesRows src;
            u.pixels = g;
            u.stride = w;
            u.fail_at = 1u + rnd(h - 1u);
            u.calls = 0;
            src.width = w;
            src.height = h;
            src.bpp = 1u;
            src.row = row_cb;
            src.user = &u;
            if (vocr_find_columns_rows(&src, 0, arena, need, &a) != VOCR_LINES_E_SOURCE || a.count) {
                fprintf(stderr, "case %u: a failing row was not reported\n", k);
                fails++;
            }
        }
        {   /* the rotation against a direct transpose, grey and RGBA */
            unsigned cw = 1u + rnd(40), chh = 1u + rnd(80), x, y;
            unsigned char *src = (unsigned char *)malloc((size_t)cw * chh * 4u), *d1 = (unsigned char *)malloc((size_t)cw * chh);
            unsigned char *d2 = (unsigned char *)malloc((size_t)cw * chh);
            if (!src || !d1 || !d2) return 2;
            for (i = 0; i < (size_t)cw * chh * 4u; i++) src[i] = (unsigned char)rnd(256);
            if (vocr_column_rotate(src, cw, 1u, cw, chh, d1, chh) || vocr_column_rotate(src, (size_t)cw * 4u, 4u, cw, chh, d2, chh)) {
                fprintf(stderr, "case %u: rotate refused\n", k);
                fails++;
            }
            for (y = 0; y < cw; y++)
                for (x = 0; x < chh; x++) {
                    const unsigned char *q = src + (size_t)x * cw * 4u + (size_t)(cw - 1u - y) * 4u;
                    unsigned gry = (77u * q[0] + 150u * q[1] + 29u * q[2]) >> 8;
                    if (d1[(size_t)y * chh + x] != src[(size_t)x * cw + (cw - 1u - y)] || d2[(size_t)y * chh + x] != gry) {
                        fprintf(stderr, "case %u: rotation differs at %u,%u\n", k, x, y);
                        fails++;
                        y = cw;
                        break;
                    }
                }
            free(src);
            free(d1);
            free(d2);
        }
        free(g);
        free(rgba);
        free(arena);
    }
    if (vocr_columns_arena_bytes(0, 10) || vocr_columns_arena_bytes(10, 0)) fails++;
    printf("selftest: %u cases, %u columns, %u failures; arena 960x544 %lu B, 480x272 %lu B\n", cases, found, fails,
           (unsigned long)vocr_columns_arena_bytes(960, 544), (unsigned long)vocr_columns_arena_bytes(480, 272));
    return fails ? 1 : 0;
}

int main(int argc, char **argv)
{
    unsigned w, h, bpp, flags = 0, i;
    int rows = 0, rc;
    unsigned char *px;
    VocrLineSet set;
    if (argc >= 3 && !strcmp(argv[1], "--selftest")) return selftest((unsigned)strtoul(argv[2], NULL, 10));
    if (argc == 4 && !strcmp(argv[1], "--arena")) {
        printf("%lu\n", (unsigned long)vocr_columns_arena_bytes((unsigned)strtoul(argv[2], NULL, 10),
                                                                (unsigned)strtoul(argv[3], NULL, 10)));
        return 0;
    }
    if (argc == 4 && !strcmp(argv[1], "--rotate")) {
        unsigned char *dst;
        FILE *f;
        px = read_pnm(argv[2], &w, &h, &bpp);
        if (!px) return 2;
        dst = (unsigned char *)malloc((size_t)w * h);
        if (!dst || vocr_column_rotate(px, (size_t)w * bpp, bpp, w, h, dst, h)) return 2;
        f = fopen(argv[3], "wb");
        if (!f) return 2;
        fprintf(f, "P5\n%u %u\n255\n", h, w);
        fwrite(dst, 1, (size_t)w * h, f);
        fclose(f);
        free(px);
        free(dst);
        return 0;
    }
    if (argc < 2) {
        fprintf(stderr, "usage: %s REGION.pgm|ppm [--flags F] [--rows] | --rotate IN OUT | --arena W H | --selftest N\n",
                argv[0]);
        return 2;
    }
    for (i = 2; i < (unsigned)argc; i++) {
        if (!strcmp(argv[i], "--rows")) rows = 1;
        else if (!strcmp(argv[i], "--flags") && i + 1 < (unsigned)argc) flags = (unsigned)strtoul(argv[++i], NULL, 0);
    }
    px = read_pnm(argv[1], &w, &h, &bpp);
    if (!px) {
        fprintf(stderr, "cannot read %s\n", argv[1]);
        return 2;
    }
    rc = run_one(px, w, h, bpp, flags, rows, &set);
    free(px);
    if (rc) {
        fprintf(stderr, "vocr_find_columns: %d\n", rc);
        return 1;
    }
    for (i = 0; i < set.count; i++)
        printf("%u %u %u %u %u\n", set.lines[i].x, set.lines[i].y, set.lines[i].w, set.lines[i].h, set.lines[i].dark);
    printf("truncated %u\n", set.truncated);
    return 0;
}
