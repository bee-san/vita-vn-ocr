/* Host driver and self-test of the text-presence gate (vocr_textgate.c). Not library code: it has its
 * own main() and uses stdio and malloc.
 *
 *   vocr_textgate_test --selftest CASES BLOB...  invariants on procedural regions, per blob: push ==
 *                                                image == rows (a callback that poisons each row once
 *                                                the next is asked, and the promised access pattern),
 *                                                GREY == RGBA of the same grey, padded strides, every
 *                                                arena alignment, the exact arena (one byte less:
 *                                                VOCR_LINES_E_ARENA), a dirty arena, NULL rows, edge
 *                                                sizes, region == boxes over one box covering the region,
 *                                                the finder mode (g NULL == vocr_find_lines for finders
 *                                                0..3; lines iff score >= t), argument and blob checks
 *   vocr_textgate_test --batch BLOB FILE         one output line per record (below; with --maps after
 *   vocr_textgate_test --maps BLOB FILE          --batch's line, boxes mode: "M hm wm" + the K map)
 *   vocr_textgate_test --dry BLOB FILE           reads the records, runs nothing ("0 0" each): the
 *                                                baseline of the A9 instruction counts
 *   vocr_textgate_test --load BLOB...            vocr_textgate_load's return code per file
 *   vocr_textgate_test --arena BLOB MODE W H     vocr_textgate_arena_bytes (one integer)
 *   vocr_textgate_test --arenas BLOB MODE W H    that, vocr_lines_arena_bytes, vocr_lines_gated_arena_bytes
 *   vocr_textgate_test --bench BLOB FILE REPS    host time per record (best of REPS), the record's form
 *
 * FILE records: struct.pack("<4I", width, height, bpp, flags), width * height * bpp pixel bytes (bpp 1 =
 * GREY, 4 = RGBA); then, with FL_BOXES, struct.pack("<I", n) and n boxes struct.pack("<4I", x, y, w, h)
 * (n <= 16); then, with FL_FINDER, struct.pack("<2i", finder, t). Flags: bits 0-1 the form (0 image, 1
 * rows: a callback that poisons each row once the next is asked, 2 push: begin, push, score); 4 boxes
 * mode; 8 a padded stride (image) and an arena at an odd address; bits 4-7 k (1..8); 256 the finder mode
 * (vocr_find_lines_gated, or _rows with form 1; the gate's mode from bit 2, k and t as given; the arena is
 * vocr_lines_gated_arena_bytes; 512 with it: no gate, g NULL). Output per record: "rc score" (score 99999
 * where it was not written),
 * or in the finder mode "rc count truncated" then "x y w h dark" per line.
 */
#include "vocr_textgate.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define FL_FORM 3u
#define FL_ROWS 1u
#define FL_PUSH 2u
#define FL_BOXES 4u
#define FL_ODD 8u
#define FL_FINDER 256u
#define FL_NOGATE 512u
#define UNSET 99999

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "textgate: line %d: %s (blob %s, case %u, %ux%u bpp %u)\n", \
                                            __LINE__, #x, blob_name, current, cur_w, cur_h, cur_bpp); return 1; } } while (0)

static const char *blob_name = "";
static unsigned current, cur_w, cur_h, cur_bpp;
static uint32_t rng = 0x7e57ca7eu;

static uint32_t random_word(void)
{
    rng ^= rng << 13;
    rng ^= rng >> 17;
    rng ^= rng << 5;
    return rng;
}

static unsigned below(unsigned n)
{
    return n ? random_word() % n : 0u;
}

/* ---------------------------------------------------------------- a row source that poisons */

typedef struct {
    const VocrLinesImage *im;
    unsigned char *slot[2];       /* rows handed out alternately; the other one is poisoned */
    unsigned calls, turn, fail_at;
    unsigned *log, cap;           /* the rows asked, in order (optional) */
} Source;

static const unsigned char *source_row(void *user, unsigned y)
{
    Source *s = (Source *)user;
    size_t n = (size_t)s->im->width * s->im->bpp;
    unsigned char *out;
    if (s->log && s->calls < s->cap) s->log[s->calls] = y;
    s->calls++;
    if (y >= s->im->height) return NULL;
    if (s->fail_at && s->calls >= s->fail_at) return NULL;
    memset(s->slot[s->turn ^ 1u], 0xA5, n);   /* the row handed out last time */
    out = s->slot[s->turn];
    memcpy(out, s->im->pixels + (size_t)y * s->im->stride, n);
    s->turn ^= 1u;
    return out;
}

static void source_init(Source *s, const VocrLinesImage *im, unsigned char *slots, unsigned *log, unsigned cap)
{
    memset(s, 0, sizeof(*s));
    s->im = im;
    s->slot[0] = slots;
    s->slot[1] = slots + (size_t)im->width * im->bpp;
    s->log = log;
    s->cap = cap;
}

static void rows_of(VocrLinesRows *r, const VocrLinesImage *im, Source *s)
{
    r->width = im->width;
    r->height = im->height;
    r->bpp = im->bpp;
    r->row = source_row;
    r->user = s;
}

/* ---------------------------------------------------------------- procedural regions */

static void put(unsigned char *p, unsigned bpp, unsigned v)
{
    unsigned k;
    for (k = 0; k < bpp; k++) p[k] = (unsigned char)(bpp == 4u && k < 3u ? (v + k * 7u) & 255u : v);
}

/* A plain, gradient or noisy ground, boxes, rows of text-like strokes and blobs. */
static void draw(unsigned char *p, unsigned w, unsigned h, size_t stride, unsigned bpp)
{
    unsigned x, y, i, lines = below(7), boxes = below(3), blobs = below(3);
    unsigned paper = below(256), ink = (paper + 96u + below(64)) & 255u;
    int gradient = below(3) == 0, noise = below(3) == 0;
    for (y = 0; y < h; y++)
        for (x = 0; x < w; x++) {
            unsigned v = gradient ? (paper + x * 64u / w + y) & 255u : paper;
            if (noise) v = (v + below(13)) & 255u;
            put(p + (size_t)y * stride + (size_t)x * bpp, bpp, v);
            if (bpp == 4u) p[(size_t)y * stride + (size_t)x * bpp + 3u] = (unsigned char)below(256);
        }
    for (i = 0; i < boxes; i++) {
        unsigned x0 = below(w), y0 = below(h), bw = below(w - x0) + 1u, bh = below(h - y0) + 1u, v = below(256);
        int fill = below(2) == 0;
        for (y = y0; y < y0 + bh; y++)
            for (x = x0; x < x0 + bw; x++)
                if (fill || y == y0 || y + 1u == y0 + bh || x == x0 || x + 1u == x0 + bw)
                    put(p + (size_t)y * stride + (size_t)x * bpp, bpp, v);
    }
    for (i = 0; i < lines; i++) {
        unsigned lh = 6u + below(28), pitch = 4u + below(20), v = below(3) ? ink : below(256);
        unsigned a0 = below(h), b0 = below(w), b1 = b0 + below(w - b0 + 1u);
        for (y = 0; y < lh && a0 + y < h; y++)
            for (x = b0; x < b1; x++) {
                unsigned cx = (x - b0) % pitch;
                if (cx < pitch * 3u / 4u && (cx % 4u < 2u || y % 7u < 2u || (cx + y) % 9u == 0u))
                    put(p + (size_t)(a0 + y) * stride + (size_t)x * bpp, bpp, v);
            }
    }
    for (i = 0; i < blobs; i++) {
        unsigned bw = 2u + below(30), bh = 2u + below(30), x0 = below(w), y0 = below(h), v = below(256);
        for (y = y0; y < y0 + bh && y < h; y++)
            for (x = x0; x < x0 + bw && x < w; x++) put(p + (size_t)y * stride + (size_t)x * bpp, bpp, v);
    }
}

static unsigned stride_of(const vocr_textgate_model *m)   /* the output stride: header byte 18 */
{
    return m->blob[18];
}

static unsigned cells_of(unsigned n, const vocr_textgate_model *m)
{
    return (n + stride_of(m) - 1u) / stride_of(m);
}

static unsigned grey_at(const VocrLinesImage *im, unsigned x, unsigned y)
{
    const unsigned char *p = im->pixels + (size_t)y * im->stride + (size_t)x * im->bpp;
    return im->bpp == 1u ? p[0] : (77u * p[0] + 150u * p[1] + 29u * p[2]) >> 8;
}

static int same_set(const VocrLineSet *a, const VocrLineSet *b)
{
    unsigned i;
    if (a->count != b->count || a->truncated != b->truncated) return 0;
    for (i = 0; i < a->count; i++)
        if (a->lines[i].x != b->lines[i].x || a->lines[i].y != b->lines[i].y || a->lines[i].w != b->lines[i].w ||
            a->lines[i].h != b->lines[i].h || a->lines[i].dark != b->lines[i].dark) return 0;
    return 1;
}

/* The gate through the push form, the grey of im pushed row by row (a copy, then poisoned). */
static int by_push(const vocr_textgate_model *m, int mode, unsigned k, const VocrLinesImage *im,
                   const VocrLineSet *boxes, void *arena, size_t bytes, int *score)
{
    vocr_textgate_state st;
    unsigned char *row = (unsigned char *)malloc(im->width ? im->width : 1u);
    unsigned x, y;
    int rc;
    if (!row) return -100;
    rc = vocr_textgate_begin(&st, m, mode, k, im->width, im->height, arena, bytes);
    for (y = 0; !rc && y < im->height; y++) {
        for (x = 0; x < im->width; x++) row[x] = (unsigned char)grey_at(im, x, y);
        rc = vocr_textgate_push(&st, row);
        memset(row, 0x5A, im->width);
    }
    if (!rc) rc = vocr_textgate_score(&st, boxes, score);
    free(row);
    return rc;
}

/* The k-th largest of n codes (the smallest when n < k), VOCR_TEXTGATE_NO_CELLS when n is 0. */
static int kth_of(const signed char *v, unsigned n, unsigned k)
{
    unsigned hist[256], i, seen = 0;
    int c;
    if (!n) return VOCR_TEXTGATE_NO_CELLS;
    if (k > n) k = n;
    memset(hist, 0, sizeof(hist));
    for (i = 0; i < n; i++) hist[v[i] + 128]++;
    for (c = 255; c >= 0; c--) {
        seen += hist[c];
        if (seen >= k) return c - 128;
    }
    return VOCR_TEXTGATE_NO_CELLS;
}

/* ---------------------------------------------------------------- self-test */

static void random_boxes(VocrLineSet *b, unsigned w, unsigned h)
{
    unsigned i;
    b->count = below(5);
    b->truncated = 0;
    for (i = 0; i < b->count; i++) {
        VocrLine *l = &b->lines[i];
        l->x = below(w + 8u);
        l->y = below(h + 8u);
        l->w = below(3) ? 1u + below(w) : below(4);
        l->h = below(3) ? 1u + below(h) : below(4);
        l->dark = below(2);
    }
}

static int region(const vocr_textgate_model *m, unsigned w, unsigned h, unsigned bpp)
{
    size_t stride = (size_t)w * bpp, pad = 1u + below(9);
    size_t need0 = vocr_textgate_arena_bytes(m, VOCR_TEXTGATE_REGION, w, h);
    size_t need1 = vocr_textgate_arena_bytes(m, VOCR_TEXTGATE_BOXES, w, h);
    size_t lneed = vocr_lines_gated_arena_bytes(m, VOCR_TEXTGATE_BOXES, w, h), big = need1 + 64u;
    unsigned char *px = (unsigned char *)malloc(stride * h);
    unsigned char *other = (unsigned char *)malloc((size_t)w * h * (bpp == 1u ? 4u : 1u));
    unsigned char *padded = (unsigned char *)malloc((stride + pad) * h);
    unsigned char *exact = (unsigned char *)malloc(need1);
    unsigned char *arena = (unsigned char *)malloc(big + 8u);
    unsigned char *larena = (unsigned char *)malloc(lneed + 8u);
    unsigned char *slots = (unsigned char *)malloc(2u * stride);
    unsigned cap = 8u * h + 8u;
    unsigned *log = (unsigned *)malloc(cap * sizeof(unsigned));
    VocrLinesImage im, im2, imp;
    VocrLinesRows src;
    Source s;
    VocrLineSet boxes, all, la, lb;
    unsigned i, off, k, mode;
    cur_w = w;
    cur_h = h;
    cur_bpp = bpp;
    REQUIRE(px && other && padded && exact && arena && larena && slots && log);
    REQUIRE(need0 > 3u && need1 >= need0 && lneed >= need1 && lneed >= vocr_lines_arena_bytes(w, h));
    REQUIRE(need1 - need0 == (size_t)cells_of(w, m) * (cells_of(h, m) - 1u));
    draw(px, w, h, stride, bpp);
    im.pixels = px; im.width = w; im.height = h; im.stride = stride; im.bpp = bpp;
    random_boxes(&boxes, w, h);
    all.count = 1;
    all.truncated = 0;
    all.lines[0].x = all.lines[0].y = 0;
    all.lines[0].w = w;
    all.lines[0].h = h;
    all.lines[0].dark = 0;
    /* the same grey in the other pixel format */
    im2 = im;
    if (bpp == 1u) {
        for (i = 0; i < w * h; i++) {
            other[4u * i] = other[4u * i + 1u] = other[4u * i + 2u] = px[i];   /* (77 + 150 + 29) g >> 8 = g */
            other[4u * i + 3u] = (unsigned char)below(256);
        }
        im2.bpp = 4u;
    } else {
        for (i = 0; i < w * h; i++)
            other[i] = (unsigned char)((77u * px[4u * i] + 150u * px[4u * i + 1u] + 29u * px[4u * i + 2u]) >> 8);
        im2.bpp = 1u;
    }
    im2.pixels = other;
    im2.stride = (size_t)w * im2.bpp;
    for (i = 0; i < h; i++) {     /* a padded stride with junk past each row */
        memcpy(padded + i * (stride + pad), px + i * stride, stride);
        memset(padded + i * (stride + pad) + stride, 0xA5, pad);
    }
    imp = im;
    imp.pixels = padded;
    imp.stride = stride + pad;

    for (mode = 0; mode < 2u; mode++) {
        size_t need = mode ? need1 : need0;
        const VocrLineSet *bx = mode ? &boxes : NULL;
        for (k = 1; k <= VOCR_TEXTGATE_MAX_K; k += 1u + below(3)) {
            int a = UNSET, b = UNSET;
            /* the exact arena, malloc'd to its size (ASan sees any byte past it) */
            REQUIRE(vocr_textgate_image(m, (int)mode, k, &im, bx, exact, need, &a) == VOCR_LINES_OK && a != UNSET);
            REQUIRE(a >= (mode ? VOCR_TEXTGATE_NO_CELLS : -128) && a <= 127);
            /* determinism on an arena full of junk, at every alignment */
            for (off = 0; off < 8u; off++) {
                memset(arena, (int)(0x31u + 17u * off), big + 8u);
                b = UNSET;
                REQUIRE(vocr_textgate_image(m, (int)mode, k, &im, bx, arena + off, need, &b) == VOCR_LINES_OK && a == b);
            }
            /* one byte short: the arena error, the score not written */
            b = UNSET;
            REQUIRE(vocr_textgate_image(m, (int)mode, k, &im, bx, arena, need - 1u, &b) == VOCR_LINES_E_ARENA && b == UNSET);
            /* push and rows: the same score; rows 0..h-1 asked once each, in order */
            b = UNSET;
            REQUIRE(by_push(m, (int)mode, k, &im, bx, arena + below(8), need, &b) == VOCR_LINES_OK && a == b);
            source_init(&s, &im, slots, log, cap);
            rows_of(&src, &im, &s);
            b = UNSET;
            REQUIRE(vocr_textgate_rows(m, (int)mode, k, &src, bx, arena + below(8), need, &b) == VOCR_LINES_OK && a == b);
            REQUIRE(s.calls == h);
            for (i = 0; i < h; i++) REQUIRE(log[i] == i);
            /* a row that cannot be read */
            source_init(&s, &im, slots, NULL, 0);
            s.fail_at = 1u + below(h);
            rows_of(&src, &im, &s);
            b = UNSET;
            REQUIRE(vocr_textgate_rows(m, (int)mode, k, &src, bx, arena, need, &b) == VOCR_LINES_E_SOURCE && b == UNSET);
            /* the other pixel format, a padded stride */
            b = UNSET;
            REQUIRE(vocr_textgate_image(m, (int)mode, k, &im2, bx, arena, big, &b) == VOCR_LINES_OK && a == b);
            b = UNSET;
            REQUIRE(vocr_textgate_image(m, (int)mode, k, &imp, bx, arena, big, &b) == VOCR_LINES_OK && a == b);
            if (!mode) {   /* region == boxes over one box covering the region */
                b = UNSET;
                REQUIRE(vocr_textgate_image(m, VOCR_TEXTGATE_BOXES, k, &im, &all, arena, big, &b) == VOCR_LINES_OK && a == b);
            }
        }
    }
    imp.stride = stride - 1u;                              /* too short */
    {
        int b = UNSET;
        REQUIRE(vocr_textgate_image(m, 0, 1, &imp, NULL, arena, big, &b) == VOCR_LINES_E_ARG && b == UNSET);
    }
    {   /* the K map: the region score is its k-th largest code; a re-score with other boxes */
        vocr_textgate_state st;
        unsigned wm = 0, hm = 0, y, x;
        const signed char *km;
        int a = UNSET, b = UNSET;
        unsigned char *row = (unsigned char *)malloc(w);
        REQUIRE(row != NULL);
        REQUIRE(vocr_textgate_begin(&st, m, VOCR_TEXTGATE_BOXES, 3, w, h, arena, need1) == VOCR_LINES_OK);
        REQUIRE(vocr_textgate_kmap(&st, &wm, &hm) == NULL);
        for (y = 0; y < h; y++) {
            for (x = 0; x < w; x++) row[x] = (unsigned char)grey_at(&im, x, y);
            REQUIRE(vocr_textgate_score(&st, &all, &a) == VOCR_LINES_E_ARG && a == UNSET);   /* not all rows yet */
            REQUIRE(vocr_textgate_push(&st, row) == VOCR_LINES_OK);
        }
        REQUIRE(vocr_textgate_push(&st, row) == VOCR_LINES_E_ARG);    /* a row past the last: sticks */
        REQUIRE(vocr_textgate_score(&st, &all, &a) == VOCR_LINES_E_ARG && a == UNSET);
        REQUIRE(vocr_textgate_begin(&st, m, VOCR_TEXTGATE_BOXES, 3, w, h, arena, need1) == VOCR_LINES_OK);
        for (y = 0; y < h; y++) {
            for (x = 0; x < w; x++) row[x] = (unsigned char)grey_at(&im, x, y);
            REQUIRE(vocr_textgate_push(&st, row) == VOCR_LINES_OK);
        }
        km = vocr_textgate_kmap(&st, &wm, &hm);
        REQUIRE(km && wm == cells_of(w, m) && hm == cells_of(h, m));
        for (k = 1; k <= VOCR_TEXTGATE_MAX_K; k++) {
            REQUIRE(vocr_textgate_image(m, VOCR_TEXTGATE_REGION, k, &im, NULL, exact, need0, &a) == VOCR_LINES_OK);
            REQUIRE(a == kth_of(km, wm * hm, k));
        }
        REQUIRE(vocr_textgate_score(&st, &all, &a) == VOCR_LINES_OK && a == kth_of(km, wm * hm, 3));
        REQUIRE(vocr_textgate_score(&st, &boxes, &b) == VOCR_LINES_OK);
        REQUIRE(vocr_textgate_score(&st, &all, &b) == VOCR_LINES_OK && a == b);      /* again: the same */
        all.count = 0;
        REQUIRE(vocr_textgate_score(&st, &all, &b) == VOCR_LINES_OK && b == VOCR_TEXTGATE_NO_CELLS);
        REQUIRE(vocr_textgate_score(&st, NULL, &b) == VOCR_LINES_E_ARG);
        all.count = VOCR_LINES_MAX + 1u;
        REQUIRE(vocr_textgate_score(&st, &all, &b) == VOCR_LINES_E_ARG);
        all.count = 1;
        free(row);
    }
    {   /* the finder mode: g NULL is vocr_find_lines; with a gate, lines iff score >= t */
        int f;
        for (f = VOCR_LINES_GRAD; f <= VOCR_LINES_TEXTBOX_RECOVERY; f++) {
            int r1 = vocr_find_lines(&im, f, larena, lneed, &la), r2, sc = UNSET;
            r2 = vocr_find_lines_gated(&im, f, NULL, (int)below(2), 1u + below(8), (int)below(300) - 150, larena + 1,
                                       lneed, &lb);
            REQUIRE(r1 == r2 && same_set(&la, &lb));
            source_init(&s, &im, slots, NULL, 0);
            rows_of(&src, &im, &s);
            r2 = vocr_find_lines_gated_rows(&src, f, NULL, 0, 1, 0, larena, lneed, &lb);
            REQUIRE(r1 == r2 && same_set(&la, &lb));
            if (r1) continue;   /* the legacy finder refuses some regions */
            for (mode = 0; mode < 2u; mode++) {
                unsigned kk = 1u + below(8);
                int t;
                REQUIRE(vocr_textgate_image(m, (int)mode, kk, &im, mode ? &la : NULL, arena, big, &sc) == VOCR_LINES_OK);
                for (t = sc - 1; t <= sc + 1; t++) {
                    VocrLineSet want = la;
                    if (sc < t) want.count = want.truncated = 0;
                    REQUIRE(vocr_find_lines_gated(&im, f, m, (int)mode, kk, t, larena, lneed, &lb) == VOCR_LINES_OK);
                    REQUIRE(same_set(&want, &lb));
                    source_init(&s, &im, slots, log, cap);
                    rows_of(&src, &im, &s);
                    REQUIRE(vocr_find_lines_gated_rows(&src, f, m, (int)mode, kk, t, larena + 3, lneed, &lb) ==
                            VOCR_LINES_OK);
                    REQUIRE(same_set(&want, &lb));
                    if (!mode)   /* the gate's pass comes first: rows 0..h-1 */
                        for (i = 0; i < h && i < cap; i++) REQUIRE(log[i] == i);
                    if (!mode && sc < t) REQUIRE(s.calls == h);   /* the finder never ran */
                }
                REQUIRE(vocr_find_lines_gated(&im, f, m, (int)mode, kk, sc, larena,
                                              vocr_lines_gated_arena_bytes(m, (int)mode, w, h) - 1u, &lb) ==
                        VOCR_LINES_E_ARENA && !lb.count && !lb.truncated);
            }
        }
        REQUIRE(vocr_find_lines_gated(&im, 7, m, 0, 1, 0, larena, lneed, &lb) == VOCR_LINES_E_ARG && !lb.count);
        REQUIRE(vocr_find_lines_gated(&im, 0, m, 0, 0, 0, larena, lneed, &lb) == VOCR_LINES_E_ARG && !lb.count);
        REQUIRE(vocr_find_lines_gated(&im, 0, m, 1, 9, 0, larena, lneed, &lb) == VOCR_LINES_E_ARG && !lb.count);
        REQUIRE(vocr_find_lines_gated(&im, 0, m, 2, 1, 0, larena, lneed, &lb) == VOCR_LINES_E_ARG && !lb.count);
        REQUIRE(vocr_find_lines_gated(NULL, 0, m, 0, 1, 0, larena, lneed, &lb) == VOCR_LINES_E_ARG && !lb.count);
        REQUIRE(vocr_find_lines_gated_rows(NULL, 0, m, 0, 1, 0, larena, lneed, &lb) == VOCR_LINES_E_ARG);
        source_init(&s, &im, slots, NULL, 0);
        s.fail_at = 1u + below(h);
        rows_of(&src, &im, &s);
        REQUIRE(vocr_find_lines_gated_rows(&src, 0, m, (int)below(2), 1, -200, larena, lneed, &lb) ==
                VOCR_LINES_E_SOURCE && !lb.count && !lb.truncated);
    }
    free(px); free(other); free(padded); free(exact); free(arena); free(larena); free(slots); free(log);
    return 0;
}

static int arena_checks(const vocr_textgate_model *m)
{
    unsigned w, h;
    size_t prev = 0;
    for (w = 1; w <= VOCR_TEXTGATE_MAX_SIDE; w += 1u + w / 8u) {
        size_t a = vocr_textgate_arena_bytes(m, VOCR_TEXTGATE_REGION, w, 1);
        REQUIRE(a >= prev);                                               /* never shrinks with the width */
        for (h = 2; h <= VOCR_TEXTGATE_MAX_SIDE; h = h * 3u + 1u) {
            REQUIRE(vocr_textgate_arena_bytes(m, VOCR_TEXTGATE_REGION, w, h) == a);
            REQUIRE(vocr_textgate_arena_bytes(m, VOCR_TEXTGATE_BOXES, w, h) ==
                    a + (size_t)cells_of(w, m) * (cells_of(h, m) - 1u));
        }
        prev = a;
    }
    REQUIRE(vocr_textgate_arena_bytes(m, 0, 0, 5) == 0 && vocr_textgate_arena_bytes(m, 0, 5, 0) == 0);
    REQUIRE(vocr_textgate_arena_bytes(m, 0, VOCR_TEXTGATE_MAX_SIDE + 1u, 5) == 0);
    REQUIRE(vocr_textgate_arena_bytes(m, 1, 5, VOCR_TEXTGATE_MAX_SIDE + 1u) == 0);
    REQUIRE(vocr_textgate_arena_bytes(m, 2, 5, 5) == 0 && vocr_textgate_arena_bytes(m, -1, 5, 5) == 0);
    REQUIRE(vocr_textgate_arena_bytes(NULL, 0, 5, 5) == 0);
    REQUIRE(vocr_lines_gated_arena_bytes(NULL, 0, 960, 544) == vocr_lines_arena_bytes(960, 544));
    return 0;
}

static int arg_checks(const vocr_textgate_model *m)
{
    unsigned char px[16] = {0}, arena[64];
    VocrLinesImage im;
    vocr_textgate_model bad;
    vocr_textgate_state st;
    size_t need;
    unsigned char *big;
    int sc = UNSET;
    im.pixels = px; im.width = 4; im.height = 4; im.stride = 4; im.bpp = 1;
    memset(&bad, 0, sizeof(bad));
    need = vocr_textgate_arena_bytes(m, 1, VOCR_TEXTGATE_MAX_SIDE, 2);
    big = (unsigned char *)malloc(need + 8u);
    REQUIRE(big != NULL);
    REQUIRE(vocr_textgate_image(NULL, 0, 1, &im, NULL, big, need, &sc) == VOCR_LINES_E_ARG);
    REQUIRE(vocr_textgate_image(&bad, 0, 1, &im, NULL, big, need, &sc) == VOCR_LINES_E_ARG);
    REQUIRE(vocr_textgate_image(m, 0, 1, NULL, NULL, big, need, &sc) == VOCR_LINES_E_ARG);
    REQUIRE(vocr_textgate_image(m, 0, 1, &im, NULL, NULL, need, &sc) == VOCR_LINES_E_ARG);
    REQUIRE(vocr_textgate_image(m, 0, 1, &im, NULL, big, need, NULL) == VOCR_LINES_E_ARG);
    REQUIRE(vocr_textgate_image(m, 0, 0, &im, NULL, big, need, &sc) == VOCR_LINES_E_ARG);
    REQUIRE(vocr_textgate_image(m, 0, 9, &im, NULL, big, need, &sc) == VOCR_LINES_E_ARG);
    REQUIRE(vocr_textgate_image(m, 2, 1, &im, NULL, big, need, &sc) == VOCR_LINES_E_ARG);
    REQUIRE(vocr_textgate_image(m, 1, 1, &im, NULL, big, need, &sc) == VOCR_LINES_E_ARG);   /* boxes mode, no boxes */
    REQUIRE(sc == UNSET);
    REQUIRE(vocr_textgate_image(m, 0, 8, &im, NULL, arena, sizeof(arena), &sc) == VOCR_LINES_E_ARENA);
    im.bpp = 3;
    im.stride = 12;
    REQUIRE(vocr_textgate_image(m, 0, 1, &im, NULL, big, need, &sc) == VOCR_LINES_E_ARG);
    im.bpp = 1;
    im.stride = 4;
    im.width = 0;
    REQUIRE(vocr_textgate_image(m, 0, 1, &im, NULL, big, need, &sc) == VOCR_LINES_E_ARG);
    im.width = 4;
    im.height = 0;
    REQUIRE(vocr_textgate_image(m, 0, 1, &im, NULL, big, need, &sc) == VOCR_LINES_E_ARG);
    REQUIRE(sc == UNSET);
    {   /* the largest side is fine, one more is not */
        unsigned char *row = (unsigned char *)calloc(VOCR_TEXTGATE_MAX_SIDE + 1u, 2);
        VocrLineSet one;
        REQUIRE(row != NULL);
        one.count = 1;
        one.lines[0].x = one.lines[0].y = 0;
        one.lines[0].w = one.lines[0].h = 1;
        im.pixels = row;
        im.height = 2;
        im.width = VOCR_TEXTGATE_MAX_SIDE;
        im.stride = VOCR_TEXTGATE_MAX_SIDE + 1u;
        REQUIRE(vocr_textgate_image(m, 1, 1, &im, &one, big, need, &sc) == VOCR_LINES_OK && sc != UNSET);
        im.width = VOCR_TEXTGATE_MAX_SIDE + 1u;
        sc = UNSET;
        REQUIRE(vocr_textgate_image(m, 1, 1, &im, &one, big, need, &sc) == VOCR_LINES_E_ARG && sc == UNSET);
        free(row);
    }
    {
        VocrLinesRows r;
        memset(&r, 0, sizeof(r));
        r.width = 4; r.height = 4; r.bpp = 1;
        REQUIRE(vocr_textgate_rows(m, 0, 1, &r, NULL, big, need, &sc) == VOCR_LINES_E_ARG);   /* no callback */
        REQUIRE(vocr_textgate_rows(m, 0, 1, NULL, NULL, big, need, &sc) == VOCR_LINES_E_ARG);
    }
    /* the push form's own checks; errors stick */
    REQUIRE(vocr_textgate_begin(NULL, m, 0, 1, 4, 4, big, need) == VOCR_LINES_E_ARG);
    REQUIRE(vocr_textgate_begin(&st, m, 0, 1, 4, 4, big, 3) == VOCR_LINES_E_ARENA);
    REQUIRE(vocr_textgate_push(&st, px) == VOCR_LINES_E_ARENA && vocr_textgate_score(&st, NULL, &sc) == VOCR_LINES_E_ARENA);
    REQUIRE(vocr_textgate_begin(&st, m, 0, 1, 4, 4, big, need) == VOCR_LINES_OK);
    REQUIRE(vocr_textgate_push(&st, NULL) == VOCR_LINES_E_ARG && vocr_textgate_push(&st, px) == VOCR_LINES_E_ARG);
    REQUIRE(vocr_textgate_push(NULL, px) == VOCR_LINES_E_ARG && vocr_textgate_score(NULL, NULL, &sc) == VOCR_LINES_E_ARG);
    REQUIRE(vocr_textgate_begin(&st, m, 0, 1, 4, 4, big, need) == VOCR_LINES_OK);
    REQUIRE(vocr_textgate_score(&st, NULL, NULL) == VOCR_LINES_E_ARG && sc == UNSET);
    REQUIRE(vocr_textgate_kmap(&st, NULL, NULL) == NULL && vocr_textgate_kmap(NULL, NULL, NULL) == NULL);
    free(big);
    REQUIRE(vocr_textgate_load(NULL, px, sizeof(px)) == VOCR_LINES_E_ARG);
    REQUIRE(vocr_textgate_load(&bad, NULL, 0) == VOCR_LINES_E_ARG);
    REQUIRE(vocr_textgate_load(&bad, px, sizeof(px)) == VOCR_TEXTGATE_E_BLOB && !bad.blob);
    return 0;
}

/* Damaged copies of a good blob are refused: a flipped byte anywhere fails the CRC, a cut one the size. */
static int damage_checks(const unsigned char *blob, size_t n)
{
    unsigned char *c = (unsigned char *)malloc(n);
    vocr_textgate_model t;
    unsigned i;
    REQUIRE(c != NULL);
    for (i = 0; i < 64u; i++) {
        size_t at = below((unsigned)n);
        memcpy(c, blob, n);
        c[at] ^= (unsigned char)(1u + below(255));
        REQUIRE(vocr_textgate_load(&t, c, n) == VOCR_TEXTGATE_E_BLOB && !t.blob);
    }
    memcpy(c, blob, n);
    REQUIRE(vocr_textgate_load(&t, c, n - 1u) == VOCR_TEXTGATE_E_BLOB);
    REQUIRE(vocr_textgate_load(&t, c, n) == VOCR_LINES_OK);
    free(c);
    return 0;
}

static unsigned char *load_file(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    unsigned char *buf;
    long len;
    if (!f) return NULL;
    if (fseek(f, 0, SEEK_END) || (len = ftell(f)) < 0 || fseek(f, 0, SEEK_SET)) { fclose(f); return NULL; }
    buf = (unsigned char *)malloc(len > 0 ? (size_t)len : 1u);
    if (!buf || fread(buf, 1, (size_t)len, f) != (size_t)len) { fclose(f); free(buf); return NULL; }
    fclose(f);
    *n = (size_t)len;
    return buf;
}

static int selftest(unsigned cases, int nblobs, char **blobs)
{
    /* edge sizes, the small ones first: CASES below the list's length runs only its head */
    static const unsigned edge[][2] = {{1, 1}, {1, 2}, {2, 1}, {2, 2}, {3, 3}, {1, 37}, {37, 1}, {5, 7}, {7, 5},
                                       {9, 9}, {15, 17}, {31, 33}, {33, 31}, {63, 65}, {127, 9}, {3, 300},
                                       {300, 3}, {480, 272}, {2048, 3}, {3, 2048}, {961, 545}, {960, 544}};
    int bi;
    unsigned total = 0;
    if (nblobs < 1) { fprintf(stderr, "--selftest needs blob files\n"); return 2; }
    for (bi = 0; bi < nblobs; bi++) {
        size_t n = 0;
        unsigned char *blob = load_file(blobs[bi], &n);
        vocr_textgate_model m;
        unsigned e;
        blob_name = blobs[bi];
        current = 0;
        REQUIRE(blob != NULL);
        REQUIRE(vocr_textgate_load(&m, blob, n) == VOCR_LINES_OK && m.blob == blob);
        if (arena_checks(&m) || arg_checks(&m) || damage_checks(blob, n)) return 1;
        for (e = 0; e < sizeof(edge) / sizeof(edge[0]) && current < cases; e++, current++)
            if (region(&m, edge[e][0], edge[e][1], below(2) ? 4u : 1u)) return 1;
        for (; current < cases; current++) {
            /* one draw per statement: the cases must not depend on the order of evaluation */
            unsigned w = 1u + below(current % 5u == 0 ? 960u : 400u);
            unsigned h = 1u + below(current % 7u == 0 ? 544u : 160u);
            unsigned bpp = below(2) ? 4u : 1u;
            if (region(&m, w, h, bpp)) return 1;
        }
        total += current;
        free(blob);
    }
    printf("TEXTGATE_SELFTEST_OK %d blobs, %u regions\n", nblobs, total);
    return 0;
}

/* ---------------------------------------------------------------- batch modes */

static unsigned rd32(const unsigned char *p)
{
    return (unsigned)p[0] | (unsigned)p[1] << 8 | (unsigned)p[2] << 16 | (unsigned)p[3] << 24;
}

enum { RUN, MAPS, DRY, BENCH };

static int read_u32s(FILE *f, unsigned *v, unsigned n)
{
    unsigned char b[64];
    unsigned i;
    if (n > 16u || fread(b, 4, n, f) != n) return 0;
    for (i = 0; i < n; i++) v[i] = rd32(b + 4u * i);
    return 1;
}

static int batch(const char *blob_path, const char *path, int mode, unsigned reps)
{
    size_t n = 0;
    unsigned char *blob = load_file(blob_path, &n);
    FILE *f = fopen(path, "rb");
    vocr_textgate_model m;
    unsigned head[4];
    int rc;
    double total = 0;
    if (!blob || !f) { fprintf(stderr, "cannot read %s or %s\n", blob_path, path); return 2; }
    rc = vocr_textgate_load(&m, blob, n);
    if (rc) { fprintf(stderr, "%s: vocr_textgate_load %d\n", blob_path, rc); return 2; }
    while (read_u32s(f, head, 4)) {
        unsigned w = head[0], h = head[1], bpp = head[2], flags = head[3], i, form = flags & FL_FORM;
        unsigned k = (flags >> 4) & 15u, gmode = (flags & FL_BOXES) ? 1u : 0u, ft[2] = {0, 0};
        size_t bytes = (size_t)w * h * bpp, stride = (size_t)w * bpp, need;
        unsigned char *px = (unsigned char *)malloc(bytes ? bytes : 1u), *buf = px, *arena, *base;
        VocrLinesImage im;
        VocrLineSet boxes;
        int score = UNSET;
        memset(&boxes, 0, sizeof(boxes));
        if (!px || fread(px, 1, bytes, f) != bytes) { fprintf(stderr, "truncated batch\n"); return 2; }
        if (flags & FL_BOXES) {
            unsigned nb, v[4];
            if (!read_u32s(f, &nb, 1) || nb > VOCR_LINES_MAX) { fprintf(stderr, "bad boxes\n"); return 2; }
            boxes.count = nb;
            for (i = 0; i < nb; i++) {
                if (!read_u32s(f, v, 4)) { fprintf(stderr, "truncated boxes\n"); return 2; }
                boxes.lines[i].x = v[0]; boxes.lines[i].y = v[1]; boxes.lines[i].w = v[2]; boxes.lines[i].h = v[3];
            }
        }
        if ((flags & FL_FINDER) && !read_u32s(f, ft, 2)) { fprintf(stderr, "truncated finder record\n"); return 2; }
        if ((flags & FL_ODD) && form == 0u) {   /* a padded stride, junk past each row */
            size_t ps = stride + 3u + w % 5u;
            buf = (unsigned char *)malloc(ps * h + 1u);
            if (!buf) return 2;
            for (i = 0; i < h; i++) {
                memcpy(buf + i * ps, px + i * stride, stride);
                memset(buf + i * ps + stride, 0xA5, ps - stride);
            }
            stride = ps;
        }
        im.pixels = buf; im.width = w; im.height = h; im.stride = stride; im.bpp = bpp;
        need = (flags & FL_FINDER) ? vocr_lines_gated_arena_bytes(&m, (int)gmode, w, h)
                                   : vocr_textgate_arena_bytes(&m, (int)gmode, w, h);
        arena = (unsigned char *)malloc(need + 8u);
        if (!arena) return 2;
        base = arena + ((flags & FL_ODD) ? 1u + w % 7u : 0u);
        if (mode == DRY) {
            printf("0 0\n");
        } else {
            unsigned char *slots = (unsigned char *)malloc(2u * (size_t)w * bpp + 1u);
            Source s;
            VocrLinesRows src;
            double best = 1e30;
            unsigned r, nrep = mode == BENCH ? reps : 1u;
            if (!slots) return 2;
            for (r = 0; r < nrep; r++) {
                clock_t t0 = clock();
                double dt;
                source_init(&s, &im, slots, NULL, 0);
                rows_of(&src, &im, &s);
                if (flags & FL_FINDER) {
                    VocrLineSet ls;
                    const vocr_textgate_model *g = (flags & FL_NOGATE) ? NULL : &m;
                    rc = form == FL_ROWS ? vocr_find_lines_gated_rows(&src, (int)ft[0], g, (int)gmode, k, (int)ft[1],
                                                                      base, need, &ls)
                                         : vocr_find_lines_gated(&im, (int)ft[0], g, (int)gmode, k, (int)ft[1], base,
                                                                 need, &ls);
                    if (mode != BENCH) {
                        printf("%d %u %u", rc, ls.count, ls.truncated);
                        for (i = 0; i < ls.count; i++)
                            printf(" %u %u %u %u %u", ls.lines[i].x, ls.lines[i].y, ls.lines[i].w, ls.lines[i].h,
                                   ls.lines[i].dark);
                        printf("\n");
                    }
                } else if (form == FL_PUSH || mode == MAPS) {
                    vocr_textgate_state st;
                    unsigned char *row = (unsigned char *)malloc(w ? w : 1u);
                    unsigned x, y;
                    if (!row) return 2;
                    rc = vocr_textgate_begin(&st, &m, (int)gmode, k, w, h, base, need);
                    for (y = 0; !rc && y < h; y++) {
                        for (x = 0; x < w; x++) {
                            const unsigned char *p = buf + (size_t)y * stride + (size_t)x * bpp;
                            row[x] = (unsigned char)(bpp == 1u ? p[0] : (77u * p[0] + 150u * p[1] + 29u * p[2]) >> 8);
                        }
                        rc = vocr_textgate_push(&st, row);
                    }
                    if (!rc) rc = vocr_textgate_score(&st, gmode ? &boxes : NULL, &score);
                    if (mode != BENCH) printf("%d %d\n", rc, score);
                    if (mode == MAPS) {
                        unsigned wm = 0, hm = 0, j;
                        const signed char *km = vocr_textgate_kmap(&st, &wm, &hm);
                        printf("M %u %u", km ? hm : 0u, km ? wm : 0u);
                        for (j = 0; km && j < wm * hm; j++) printf(" %d", km[j]);
                        printf("\n");
                    }
                    free(row);
                } else {
                    rc = form == FL_ROWS ? vocr_textgate_rows(&m, (int)gmode, k, &src, gmode ? &boxes : NULL, base,
                                                              need, &score)
                                         : vocr_textgate_image(&m, (int)gmode, k, &im, gmode ? &boxes : NULL, base,
                                                               need, &score);
                    if (mode != BENCH) printf("%d %d\n", rc, score);
                }
                dt = (double)(clock() - t0) / CLOCKS_PER_SEC;
                if (dt < best) best = dt;
            }
            if (mode == BENCH) {
                total += best;
                printf("%d %.1f\n", rc, best * 1e6);
            }
            free(slots);
        }
        if (buf != px) free(buf);
        free(px);
        free(arena);
    }
    if (mode == BENCH) fprintf(stderr, "%.1f us total (best of %u per record)\n", total * 1e6, reps);
    fclose(f);
    free(blob);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc >= 3 && !strcmp(argv[1], "--selftest"))
        return selftest((unsigned)strtoul(argv[2], NULL, 10), argc - 3, argv + 3);
    if (argc == 4 && !strcmp(argv[1], "--batch")) return batch(argv[2], argv[3], RUN, 0);
    if (argc == 4 && !strcmp(argv[1], "--maps")) return batch(argv[2], argv[3], MAPS, 0);
    if (argc == 4 && !strcmp(argv[1], "--dry")) return batch(argv[2], argv[3], DRY, 0);
    if (argc == 5 && !strcmp(argv[1], "--bench"))
        return batch(argv[2], argv[3], BENCH, (unsigned)strtoul(argv[4], NULL, 10));
    if (argc >= 3 && !strcmp(argv[1], "--load")) {
        int i;
        for (i = 2; i < argc; i++) {
            size_t n = 0;
            unsigned char *b = load_file(argv[i], &n);
            vocr_textgate_model m;
            printf("%d\n", b ? vocr_textgate_load(&m, b, n) : -100);
            free(b);
        }
        return 0;
    }
    if (argc == 6 && (!strcmp(argv[1], "--arena") || !strcmp(argv[1], "--arenas"))) {
        size_t n = 0;
        unsigned char *b = load_file(argv[2], &n);
        vocr_textgate_model m;
        int mode = atoi(argv[3]);
        unsigned w = (unsigned)strtoul(argv[4], NULL, 10), h = (unsigned)strtoul(argv[5], NULL, 10);
        if (!b || vocr_textgate_load(&m, b, n)) { fprintf(stderr, "cannot load %s\n", argv[2]); return 2; }
        if (!strcmp(argv[1], "--arena"))
            printf("%lu\n", (unsigned long)vocr_textgate_arena_bytes(&m, mode, w, h));
        else
            printf("%lu %lu %lu\n", (unsigned long)vocr_textgate_arena_bytes(&m, mode, w, h),
                   (unsigned long)vocr_lines_arena_bytes(w, h), (unsigned long)vocr_lines_gated_arena_bytes(&m, mode, w, h));
        free(b);
        return 0;
    }
    fprintf(stderr, "usage: %s --selftest CASES BLOB... | --batch BLOB FILE | --maps BLOB FILE | --dry BLOB FILE |\n"
                    "       --load BLOB... | --arena BLOB MODE W H | --arenas BLOB MODE W H | --bench BLOB FILE REPS\n",
            argv[0]);
    return 2;
}
