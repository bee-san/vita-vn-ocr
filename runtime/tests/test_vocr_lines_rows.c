/* vocr_lines_rows.c (vocr_find_lines_rows / vocr_line_dark_rows) against vocr_lines.c
 * (vocr_find_lines / vocr_line_dark): the same
 * lines, flags and return codes for the same pixels, on procedural regions of every
 * finder, both pixel formats and padded strides, with a row callback that hands out
 * each row in a buffer it poisons on the next call (a finder that kept a row pointer
 * past the next call would read the poison). Also: the access pattern the header
 * promises, and NULL rows. No screenshots, fonts or models. */
#include "../vocr_lines.h"
#include "../vocr_lines_rows.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "rows: line %d: %s (case %u)\n", __LINE__, #x, current); return 1; } } while (0)

static unsigned current;
static uint32_t rng = 0x2c1b3c6du;
static uint32_t random_word(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}
static unsigned below(unsigned n) { return n ? random_word() % n : 0u; }

typedef struct {
    const VocrLinesImage *im;
    unsigned char *slot[2];   /* rows handed out alternately; the other one is poisoned */
    unsigned calls, turn, scans, last, fail_at, order_errors;
} Source;

static const unsigned char *source_row(void *user, unsigned y)
{
    Source *s = (Source *)user;
    size_t n = (size_t)s->im->width * s->im->bpp;
    unsigned char *out;
    if (y >= s->im->height) { s->order_errors++; return NULL; }
    /* within a pass rows increase; any smaller row starts a new pass */
    if (!s->calls || y <= s->last) s->scans++;
    s->last = y;
    if (s->fail_at && ++s->calls >= s->fail_at) return NULL;
    if (!s->fail_at) s->calls++;
    memset(s->slot[s->turn ^ 1u], 0xA5, n); /* the row handed out last time */
    out = s->slot[s->turn];
    memcpy(out, s->im->pixels + (size_t)y * s->im->stride, n);
    s->turn ^= 1u;
    return out;
}

static void draw(unsigned char *p, unsigned w, unsigned h, size_t stride, unsigned bpp)
{
    unsigned x, y, k, i, lines = below(7), boxes = below(3);
    unsigned paper = below(256), ink = (paper + 96u + below(64)) & 255u;
    int gradient = (int)below(3) == 0, noise = (int)below(4) == 0;
    for (y = 0; y < h; y++)
        for (x = 0; x < w; x++) {
            unsigned v = gradient ? (paper + x * 64u / (w ? w : 1u) + y) & 255u : paper;
            if (noise) v = (v + below(9)) & 255u;
            for (k = 0; k < bpp; k++) p[(size_t)y * stride + (size_t)x * bpp + k] =
                (unsigned char)(bpp == 4 && k < 3 ? (v + k * below(20)) & 255u : v);
        }
    for (i = 0; i < boxes; i++) {               /* box borders and filled panels */
        unsigned x0 = below(w), y0 = below(h), bw = below(w - x0) + 1u, bh = below(h - y0) + 1u;
        unsigned v = below(256), fill = below(2);
        for (y = y0; y < y0 + bh; y++)
            for (x = x0; x < x0 + bw; x++) {
                if (!fill && y != y0 && y + 1u != y0 + bh && x != x0 && x + 1u != x0 + bw) continue;
                for (k = 0; k < bpp; k++) p[(size_t)y * stride + (size_t)x * bpp + k] = (unsigned char)v;
            }
    }
    for (i = 0; i < lines; i++) {               /* rows of glyph-like strokes */
        unsigned lh = 8u + below(40), y0 = below(h), x0 = below(w), x1, pitch = 4u + below(20);
        unsigned v = below(3) ? ink : below(256);
        if (y0 + lh > h) lh = h - y0;
        x1 = x0 + below(w - x0 + 1u);
        for (y = y0; y < y0 + lh; y++)
            for (x = x0; x < x1; x++) {
                unsigned cx = (x - x0) % pitch, cy = y - y0;
                int on = cx < pitch * 3u / 4u && (cx % 4u < 2u || cy % 7u < 2u || (cx + cy) % 9u == 0u);
                if (!on) continue;
                for (k = 0; k < bpp; k++) p[(size_t)y * stride + (size_t)x * bpp + k] = (unsigned char)v;
            }
    }
}

static int same_set(const VocrLineSet *a, const VocrLineSet *b)
{
    unsigned i;
    if (a->count != b->count || a->truncated != b->truncated) return 0;
    for (i = 0; i < a->count; i++)
        if (a->lines[i].x != b->lines[i].x || a->lines[i].y != b->lines[i].y ||
            a->lines[i].w != b->lines[i].w || a->lines[i].h != b->lines[i].h ||
            a->lines[i].dark != b->lines[i].dark) return 0;
    return 1;
}

static const int finders[] = {VOCR_LINES_GRAD, VOCR_LINES_LEGACY, VOCR_LINES_TEXTBOX,
                              VOCR_LINES_TEXTBOX_RECOVERY};

/* One region: every finder both ways, a failing callback, and line_dark on a few boxes. */
static int region(unsigned w, unsigned h, unsigned bpp, unsigned pad, unsigned long long *rows_read,
                  unsigned long long *passes)
{
    size_t stride = (size_t)w * bpp + pad, need = vocr_lines_arena_bytes(w, h);
    unsigned char *pixels = (unsigned char *)malloc(stride * h + 1u);
    unsigned char *arena = (unsigned char *)malloc(need + 3u);
    unsigned char *slots = (unsigned char *)malloc(2u * (size_t)w * bpp);
    VocrLinesImage im;
    VocrLinesRows src;
    Source s;
    unsigned f, i;
    REQUIRE(pixels && arena && slots);
    draw(pixels, w, h, stride, bpp);
    im.pixels = pixels; im.width = w; im.height = h; im.stride = stride; im.bpp = bpp;
    src.width = w; src.height = h; src.bpp = bpp; src.row = source_row; src.user = &s;
    for (f = 0; f < sizeof(finders) / sizeof(finders[0]); f++) {
        VocrLineSet a, b;
        int ra, rb;
        unsigned at = below(3) + 1u;
        memset(&s, 0, sizeof(s));
        s.im = &im; s.slot[0] = slots; s.slot[1] = slots + (size_t)w * bpp;
        ra = vocr_find_lines(&im, finders[f], arena + at, need, &a);
        rb = vocr_find_lines_rows(&src, finders[f], arena + 3u - at, need, &b);
        REQUIRE(ra == rb);
        REQUIRE(same_set(&a, &b));
        REQUIRE(!s.order_errors);
        /* the passes the header names: two over the region (one for the legacy finder,
         * which may also stop early), then one over each line's box */
        if (finders[f] != VOCR_LINES_LEGACY && rb == VOCR_LINES_OK) REQUIRE(s.calls >= 2u * h);
        *rows_read += s.calls;
        *passes += s.scans;
        /* a row that cannot be read: an error and no lines, whichever row it is */
        if (s.calls) {
            unsigned fail = 1u + below(s.calls);
            memset(&s, 0, sizeof(s));
            s.im = &im; s.slot[0] = slots; s.slot[1] = slots + (size_t)w * bpp; s.fail_at = fail;
            rb = vocr_find_lines_rows(&src, finders[f], arena, need, &b);
            REQUIRE(rb < 0 && b.count == 0);
            if (finders[f] != VOCR_LINES_LEGACY) REQUIRE(rb == VOCR_LINES_E_SOURCE);
        }
    }
    for (i = 0; i < 6; i++) {
        unsigned x = below(w), y = below(h), bw = 1u + below(w - x), bh = 1u + below(h - y);
        memset(&s, 0, sizeof(s));
        s.im = &im; s.slot[0] = slots; s.slot[1] = slots + (size_t)w * bpp;
        REQUIRE(vocr_line_dark(&im, x, y, bw, bh) == vocr_line_dark_rows(&src, x, y, bw, bh));
        REQUIRE(s.calls == bh && s.scans == 1u); /* the box's rows once, top to bottom */
    }
    free(pixels);
    free(arena);
    free(slots);
    return 0;
}

int main(int argc, char **argv)
{
    static const unsigned edge[][2] = {{1, 1}, {1, 9}, {9, 1}, {8, 8}, {16, 16}, {31, 40}, {33, 7},
                                       {300, 80}, {480, 272}, {961, 30}, {960, 545}, {960, 544}};
    unsigned cases = argc > 1 ? (unsigned)strtoul(argv[1], NULL, 10) : 300u, i;
    unsigned long long rows_read = 0, passes = 0;
    VocrLinesRows bad;
    VocrLineSet out;
    unsigned char arena[64];
    for (i = 0; i < sizeof(edge) / sizeof(edge[0]); i++) {
        current = i;
        if (region(edge[i][0], edge[i][1], VOCR_LINES_RGBA, 4u * (i % 3u), &rows_read, &passes)) return 1;
        if (region(edge[i][0], edge[i][1], VOCR_LINES_GREY, i % 5u, &rows_read, &passes)) return 1;
    }
    for (i = 0; i < cases; i++) {
        /* one draw per statement: the sequence must not depend on argument evaluation order */
        unsigned wide = below(4), w, tall, h, bpp, pad;
        w = 1u + below(wide ? 400u : 960u);
        tall = below(4);
        h = 1u + below(tall ? 160u : 544u);
        bpp = below(2) ? VOCR_LINES_RGBA : VOCR_LINES_GREY;
        pad = below(3);
        pad = pad ? 0u : below(64);
        current = 1000u + i;
        if (region(w, h, bpp, pad, &rows_read, &passes)) return 1;
    }
    /* argument checks */
    current = 0;
    memset(&bad, 0, sizeof(bad));
    bad.width = bad.height = 4; bad.bpp = VOCR_LINES_GREY;
    REQUIRE(vocr_find_lines_rows(&bad, VOCR_LINES_GRAD, arena, sizeof(arena), &out) == VOCR_LINES_E_ARG);
    bad.row = source_row;
    REQUIRE(vocr_find_lines_rows(&bad, 9, arena, sizeof(arena), &out) == VOCR_LINES_E_ARG);
    REQUIRE(vocr_find_lines_rows(NULL, VOCR_LINES_GRAD, arena, sizeof(arena), &out) == VOCR_LINES_E_ARG);
    REQUIRE(vocr_find_lines_rows(&bad, VOCR_LINES_GRAD, arena, sizeof(arena), NULL) == VOCR_LINES_E_ARG);
    REQUIRE(vocr_find_lines_rows(&bad, VOCR_LINES_GRAD, arena, 1, &out) == VOCR_LINES_E_ARENA);
    bad.bpp = 3;
    REQUIRE(vocr_find_lines_rows(&bad, VOCR_LINES_GRAD, arena, sizeof(arena), &out) == VOCR_LINES_E_ARG);
    REQUIRE(vocr_line_dark_rows(&bad, 0, 0, 1, 1) == VOCR_LINES_E_ARG);
    bad.bpp = VOCR_LINES_GREY;
    REQUIRE(vocr_line_dark_rows(&bad, 3, 0, 2, 1) == VOCR_LINES_E_ARG);
    REQUIRE(vocr_line_dark_rows(&bad, 0, 0, 0, 1) == VOCR_LINES_E_ARG);
    printf("rows: %u random + %u edge regions, 4 finders: identical lines; %llu rows in %llu passes\n",
           cases, (unsigned)(2u * (sizeof(edge) / sizeof(edge[0]))), rows_read, passes);
    return 0;
}
