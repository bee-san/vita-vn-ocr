/* Host driver and self-test of the text detector (vocr_textdet.c). Not library code: it has its own
 * main() and uses stdio and malloc.
 *
 *   vocr_textdet_test --selftest CASES BLOB...   invariants on procedural regions, per blob: the
 *                                                row-callback form equals the in-memory form (with a
 *                                                callback that poisons each row once the next is asked,
 *                                                and the promised access pattern), GREY equals RGBA of
 *                                                the same grey, padded strides, every arena alignment,
 *                                                the exact arena (one byte less: VOCR_LINES_E_ARENA),
 *                                                a dirty arena, NULL rows, the finder mode, edge sizes
 *   vocr_textdet_test --batch BLOB FILE          one output line per record (below; eval/textdet.py's
 *                                                c_detect reads this)
 *   vocr_textdet_test --maps BLOB FILE           the head's maps per record: a line "X" + hm * wm cells of
 *                                                K O A1 A2 (row-major), then a line "rc hm wm"
 *   vocr_textdet_test --dry BLOB FILE            reads the records, runs nothing ("0 0 0" each): the
 *                                                baseline of the A9 instruction counts
 *   vocr_textdet_test --load BLOB...             vocr_textdet_load's return code per file
 *   vocr_textdet_test --arena BLOB W H           vocr_textdet_arena_bytes (one integer)
 *   vocr_textdet_test --arenas BLOB W H          that, vocr_lines_arena_bytes and vocr_lines_det_arena_bytes
 *   vocr_textdet_test --bench BLOB FILE REPS     host time per record (best of REPS), in-memory form
 *
 * FILE records: struct.pack("<4I", width, height, bpp, flags), then width * height * bpp pixel bytes
 * (bpp 1 = GREY, 4 = RGBA). Flags: 0 the in-memory form, 1 the row-callback form (a callback that
 * poisons each row once the next is asked); 2 the finder mode (vocr_find_lines_det with
 * VOCR_LINES_TEXTDET: horizontal lines only, vertical printed 0), 4 a padded stride, 8 an arena at an
 * odd address, all with exactly the arena the call needs. Output per record, one line:
 * rc count truncated, then x y w h dark vertical per line, space separated.
 */
#include "vocr_textdet.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "textdet: line %d: %s (blob %s, case %u, %ux%u bpp %u)\n", \
                                            __LINE__, #x, blob_name, current, cur_w, cur_h, cur_bpp); return 1; } } while (0)

static const char *blob_name = "";
static unsigned current, cur_w, cur_h, cur_bpp;
static uint32_t rng = 0x7e57d37au;

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

/* A plain, gradient or noisy ground, boxes, rows of text-like strokes (horizontal and vertical, some
 * close together) and blobs. */
static void draw(unsigned char *p, unsigned w, unsigned h, size_t stride, unsigned bpp)
{
    unsigned x, y, i, lines = below(7), cols = below(3), boxes = below(3), blobs = below(3);
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
    for (i = 0; i < lines + cols; i++) {
        int vert = i >= lines;
        unsigned lh = 6u + below(28), pitch = 4u + below(20), v = below(3) ? ink : below(256);
        unsigned a0 = below(vert ? w : h), b0 = below(vert ? h : w), b1, span = vert ? h : w;
        b1 = b0 + below(span - b0 + 1u);
        for (y = 0; y < lh; y++)
            for (x = b0; x < b1; x++) {
                unsigned cx = (x - b0) % pitch, px, py;
                int on = cx < pitch * 3u / 4u && (cx % 4u < 2u || y % 7u < 2u || (cx + y) % 9u == 0u);
                if (!on || a0 + y >= (vert ? w : h)) continue;
                px = vert ? a0 + y : x;
                py = vert ? x : a0 + y;
                put(p + (size_t)py * stride + (size_t)px * bpp, bpp, v);
            }
    }
    for (i = 0; i < blobs; i++) {
        unsigned bw = 2u + below(30), bh = 2u + below(30), x0 = below(w), y0 = below(h), v = below(256);
        for (y = y0; y < y0 + bh && y < h; y++)
            for (x = x0; x < x0 + bw && x < w; x++) put(p + (size_t)y * stride + (size_t)x * bpp, bpp, v);
    }
}

/* ---------------------------------------------------------------- comparisons */

static int same_det(const VocrTextdetSet *a, const VocrTextdetSet *b)
{
    unsigned i;
    if (a->count != b->count || a->truncated != b->truncated) return 0;
    for (i = 0; i < a->count; i++)
        if (a->lines[i].x != b->lines[i].x || a->lines[i].y != b->lines[i].y || a->lines[i].w != b->lines[i].w ||
            a->lines[i].h != b->lines[i].h || a->lines[i].dark != b->lines[i].dark || a->vertical[i] != b->vertical[i])
            return 0;
    return 1;
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

/* The finder mode's set: the detector's horizontal lines, its truncated flag. */
static int is_horizontal_part(const VocrTextdetSet *d, const VocrLineSet *s)
{
    unsigned i, k = 0;
    for (i = 0; i < d->count; i++) {
        if (d->vertical[i]) continue;
        if (k >= s->count) return 0;
        if (d->lines[i].x != s->lines[k].x || d->lines[i].y != s->lines[k].y || d->lines[i].w != s->lines[k].w ||
            d->lines[i].h != s->lines[k].h || d->lines[i].dark != s->lines[k].dark) return 0;
        k++;
    }
    return k == s->count && s->truncated == d->truncated;
}

/* Lines inside the region, non-empty, flags 0/1, horizontal lines first. */
static int well_formed(const VocrTextdetSet *d, unsigned w, unsigned h)
{
    unsigned i, seen_v = 0;
    if (d->count > VOCR_LINES_MAX || d->truncated > 1u) return 0;
    for (i = 0; i < d->count; i++) {
        const VocrLine *l = &d->lines[i];
        if (!l->w || !l->h || l->x >= w || l->y >= h || l->w > w - l->x || l->h > h - l->y) return 0;
        if (l->dark > 1u || d->vertical[i] > 1u) return 0;
        if (d->vertical[i]) seen_v = 1;
        else if (seen_v) return 0;
    }
    return 1;
}

/* ---------------------------------------------------------------- self-test */

static int region(const vocr_textdet *m, unsigned w, unsigned h, unsigned bpp)
{
    size_t stride = (size_t)w * bpp, need = vocr_textdet_arena_bytes(m, w, h);
    size_t big = need + 64u, lines_need = vocr_lines_det_arena_bytes(m, w, h);
    unsigned pad = 1u + below(9), i, off;
    unsigned char *px = (unsigned char *)malloc(stride * h);
    unsigned char *other = (unsigned char *)malloc((size_t)w * h * (bpp == 1u ? 4u : 1u));
    unsigned char *padded = (unsigned char *)malloc((stride + pad) * h);
    unsigned char *exact = (unsigned char *)malloc(need);
    unsigned char *arena = (unsigned char *)malloc(big + 8u);
    unsigned char *larena = (unsigned char *)malloc(lines_need + 8u);
    unsigned char *slots = (unsigned char *)malloc(2u * stride);
    unsigned cap = h + VOCR_LINES_MAX * h + 8u;
    unsigned *log = (unsigned *)malloc(cap * sizeof(unsigned));
    VocrLinesImage im, im2, imp;
    VocrLinesRows src;
    Source s;
    VocrTextdetSet a, b;
    VocrLineSet la, lb;
    int rc;
    cur_w = w;
    cur_h = h;
    cur_bpp = bpp;
    REQUIRE(px && other && padded && exact && arena && larena && slots && log && need > 7u);
    REQUIRE(lines_need >= need && lines_need >= vocr_lines_arena_bytes(w, h));
    draw(px, w, h, stride, bpp);
    im.pixels = px; im.width = w; im.height = h; im.stride = stride; im.bpp = bpp;

    /* the exact arena, malloc'd to its size (ASan sees any byte past it) */
    rc = vocr_textdet_find(m, &im, exact, need, &a);
    REQUIRE(rc == VOCR_LINES_OK && well_formed(&a, w, h));
    /* determinism: again, on an arena full of junk, at every alignment */
    for (off = 0; off < 8u; off++) {
        memset(arena, (int)(0x31u + 17u * off), big + 8u);
        REQUIRE(vocr_textdet_find(m, &im, arena + off, need, &b) == VOCR_LINES_OK && same_det(&a, &b));
    }
    /* one byte short: the arena error, no lines */
    REQUIRE(vocr_textdet_find(m, &im, arena, need - 1u, &b) == VOCR_LINES_E_ARENA && !b.count && !b.truncated);
    /* the row-callback form: the same lines; rows 0..h-1 once each in order, then each line's box once */
    source_init(&s, &im, slots, log, cap);
    rows_of(&src, &im, &s);
    REQUIRE(vocr_textdet_find_rows(m, &src, arena + below(8), need, &b) == VOCR_LINES_OK && same_det(&a, &b));
    {
        unsigned k = 0, y;
        for (y = 0; y < h; y++, k++) REQUIRE(k < s.calls && log[k] == y);
        for (i = 0; i < a.count; i++)
            for (y = a.lines[i].y; y < a.lines[i].y + a.lines[i].h; y++, k++) REQUIRE(k < s.calls && log[k] == y);
        REQUIRE(k == s.calls);
    }
    /* a row that cannot be read: VOCR_LINES_E_SOURCE and no lines */
    {
        unsigned total = s.calls;
        source_init(&s, &im, slots, NULL, 0);
        s.fail_at = 1u + below(total);
        rows_of(&src, &im, &s);
        REQUIRE(vocr_textdet_find_rows(m, &src, arena, need, &b) == VOCR_LINES_E_SOURCE && !b.count && !b.truncated);
    }
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
    REQUIRE(vocr_textdet_find(m, &im2, arena, big, &b) == VOCR_LINES_OK && same_det(&a, &b));
    /* a padded stride with junk past each row */
    for (i = 0; i < h; i++) {
        memcpy(padded + i * (stride + pad), px + i * stride, stride);
        memset(padded + i * (stride + pad) + stride, 0xA5, pad);
    }
    imp = im;
    imp.pixels = padded;
    imp.stride = stride + pad;
    REQUIRE(vocr_textdet_find(m, &imp, arena, big, &b) == VOCR_LINES_OK && same_det(&a, &b));
    imp.stride = stride - 1u;                              /* too short */
    REQUIRE(vocr_textdet_find(m, &imp, arena, big, &b) == VOCR_LINES_E_ARG && !b.count);
    /* the finder mode: TEXTDET gives the horizontal lines; the other finders are vocr_find_lines' */
    REQUIRE(vocr_find_lines_det(&im, VOCR_LINES_TEXTDET, m, larena, lines_need, &la) == VOCR_LINES_OK);
    REQUIRE(is_horizontal_part(&a, &la));
    source_init(&s, &im, slots, NULL, 0);
    rows_of(&src, &im, &s);
    REQUIRE(vocr_find_lines_det_rows(&src, VOCR_LINES_TEXTDET, m, larena, lines_need, &lb) == VOCR_LINES_OK);
    REQUIRE(same_set(&la, &lb));
    REQUIRE(vocr_find_lines_det(&im, VOCR_LINES_TEXTDET, NULL, larena, lines_need, &lb) == VOCR_LINES_E_ARG && !lb.count);
    {
        int f;
        for (f = VOCR_LINES_GRAD; f <= VOCR_LINES_TEXTBOX_RECOVERY; f++) {
            int r1 = vocr_find_lines(&im, f, larena, lines_need, &la), r2;
            r2 = vocr_find_lines_det(&im, f, below(2) ? m : NULL, larena + 1, lines_need, &lb);
            REQUIRE(r1 == r2 && same_set(&la, &lb));
            source_init(&s, &im, slots, NULL, 0);
            rows_of(&src, &im, &s);
            r2 = vocr_find_lines_det_rows(&src, f, m, larena, lines_need, &lb);
            REQUIRE(r1 == r2 && same_set(&la, &lb));
        }
    }
    free(px); free(other); free(padded); free(exact); free(arena); free(larena); free(slots); free(log);
    return 0;
}

static int arena_checks(const vocr_textdet *m)
{
    unsigned w, h;
    size_t prev = 0;
    for (w = 1; w <= VOCR_TEXTDET_MAX_SIDE; w += 1u + w / 8u) {
        size_t a = vocr_textdet_arena_bytes(m, w, 1);
        REQUIRE(a >= prev);                                               /* never shrinks with the width */
        for (h = 2; h <= VOCR_TEXTDET_MAX_SIDE; h = h * 3u + 1u) REQUIRE(vocr_textdet_arena_bytes(m, w, h) == a);
        REQUIRE(vocr_textdet_arena_bytes(m, w, VOCR_TEXTDET_MAX_SIDE) == a);
        prev = a;
    }
    REQUIRE(vocr_textdet_arena_bytes(m, 0, 5) == 0 && vocr_textdet_arena_bytes(m, 5, 0) == 0);
    REQUIRE(vocr_textdet_arena_bytes(m, VOCR_TEXTDET_MAX_SIDE + 1u, 5) == 0);
    REQUIRE(vocr_textdet_arena_bytes(m, 5, VOCR_TEXTDET_MAX_SIDE + 1u) == 0);
    REQUIRE(vocr_textdet_arena_bytes(NULL, 5, 5) == 0);
    REQUIRE(vocr_lines_det_arena_bytes(NULL, 960, 544) == vocr_lines_arena_bytes(960, 544));
    return 0;
}

static int arg_checks(const vocr_textdet *m)
{
    unsigned char px[16] = {0}, arena[64];
    VocrLinesImage im;
    VocrTextdetSet d;
    vocr_textdet bad;
    size_t need;
    unsigned char *big;
    im.pixels = px; im.width = 4; im.height = 4; im.stride = 4; im.bpp = 1;
    d.count = 7;
    REQUIRE(vocr_textdet_find(NULL, &im, arena, sizeof(arena), &d) == VOCR_LINES_E_ARG && !d.count);
    memset(&bad, 0, sizeof(bad));
    REQUIRE(vocr_textdet_find(&bad, &im, arena, sizeof(arena), &d) == VOCR_LINES_E_ARG);
    REQUIRE(vocr_textdet_find(m, NULL, arena, sizeof(arena), &d) == VOCR_LINES_E_ARG);
    REQUIRE(vocr_textdet_find(m, &im, arena, sizeof(arena), NULL) == VOCR_LINES_E_ARG);
    need = vocr_textdet_arena_bytes(m, VOCR_TEXTDET_MAX_SIDE, 2);
    big = (unsigned char *)malloc(need + 8u);
    REQUIRE(big != NULL);
    REQUIRE(vocr_textdet_find(m, &im, NULL, need, &d) == VOCR_LINES_E_ARG);
    im.bpp = 3;
    im.stride = 12;
    REQUIRE(vocr_textdet_find(m, &im, big, need, &d) == VOCR_LINES_E_ARG);
    im.bpp = 1;
    im.stride = 4;
    im.width = 0;
    REQUIRE(vocr_textdet_find(m, &im, big, need, &d) == VOCR_LINES_E_ARG);
    im.width = 4;
    im.height = 0;
    REQUIRE(vocr_textdet_find(m, &im, big, need, &d) == VOCR_LINES_E_ARG);
    {
        /* the largest side is fine, one more is not */
        unsigned char *row = (unsigned char *)calloc(VOCR_TEXTDET_MAX_SIDE + 1u, 2);
        REQUIRE(row != NULL);
        im.pixels = row;
        im.height = 2;
        im.width = VOCR_TEXTDET_MAX_SIDE;
        im.stride = VOCR_TEXTDET_MAX_SIDE + 1u;
        REQUIRE(vocr_textdet_find(m, &im, big, need, &d) == VOCR_LINES_OK);
        im.width = VOCR_TEXTDET_MAX_SIDE + 1u;
        REQUIRE(vocr_textdet_find(m, &im, big, need, &d) == VOCR_LINES_E_ARG);
        free(row);
    }
    {
        VocrLinesRows r;
        memset(&r, 0, sizeof(r));
        r.width = 4; r.height = 4; r.bpp = 1;
        REQUIRE(vocr_textdet_find_rows(m, &r, big, need, &d) == VOCR_LINES_E_ARG);   /* no callback */
        REQUIRE(vocr_textdet_find_rows(m, NULL, big, need, &d) == VOCR_LINES_E_ARG);
    }
    free(big);
    REQUIRE(vocr_textdet_load(NULL, px, sizeof(px)) == VOCR_LINES_E_ARG);
    REQUIRE(vocr_textdet_load(&bad, NULL, 0) == VOCR_LINES_E_ARG);
    REQUIRE(vocr_textdet_load(&bad, px, sizeof(px)) == VOCR_TEXTDET_E_BLOB && !bad.loaded);
    return 0;
}

/* Damaged copies of a good blob are refused: a flipped byte anywhere fails the CRC, a cut one the size. */
static int damage_checks(const unsigned char *blob, size_t n)
{
    unsigned char *c = (unsigned char *)malloc(n);
    vocr_textdet t;
    unsigned i;
    REQUIRE(c != NULL);
    for (i = 0; i < 64u; i++) {
        size_t at = below((unsigned)n);
        memcpy(c, blob, n);
        c[at] ^= (unsigned char)(1u + below(255));
        REQUIRE(vocr_textdet_load(&t, c, n) == VOCR_TEXTDET_E_BLOB && !t.loaded);
    }
    memcpy(c, blob, n);
    REQUIRE(vocr_textdet_load(&t, c, n - 1u) == VOCR_TEXTDET_E_BLOB);
    REQUIRE(vocr_textdet_load(&t, c, n) == VOCR_LINES_OK);
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
    static const unsigned edge[][2] = {{1, 1}, {1, 2}, {2, 1}, {2, 2}, {3, 3}, {1, 37}, {37, 1}, {5, 7}, {7, 5},
                                       {9, 9}, {15, 17}, {31, 33}, {33, 31}, {63, 65}, {127, 9}, {3, 300},
                                       {300, 3}, {2048, 3}, {3, 2048}, {961, 545}, {960, 544}, {480, 272}};
    int bi;
    unsigned total = 0;
    if (nblobs < 1) { fprintf(stderr, "--selftest needs blob files\n"); return 2; }
    for (bi = 0; bi < nblobs; bi++) {
        size_t n = 0;
        unsigned char *blob = load_file(blobs[bi], &n);
        vocr_textdet m;
        unsigned e;
        blob_name = blobs[bi];
        current = 0;
        REQUIRE(blob != NULL);
        REQUIRE(vocr_textdet_load(&m, blob, n) == VOCR_LINES_OK && m.blob_bytes == n);
        if (arena_checks(&m) || arg_checks(&m) || damage_checks(blob, n)) return 1;
        for (e = 0; e < sizeof(edge) / sizeof(edge[0]); e++, current++)
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
    printf("TEXTDET_SELFTEST_OK %d blobs, %u regions\n", nblobs, total);
    return 0;
}

/* ---------------------------------------------------------------- batch modes */

static unsigned rd32(const unsigned char *p)
{
    return (unsigned)p[0] | (unsigned)p[1] << 8 | (unsigned)p[2] << 16 | (unsigned)p[3] << 24;
}

typedef struct {
    unsigned rows, wm;
} MapPrint;

static void print_map_row(void *user, unsigned i, const signed char *cells, unsigned width)
{
    MapPrint *p = (MapPrint *)user;
    unsigned j;
    (void)i;
    for (j = 0; j < 4u * width; j++) printf(" %d", cells[j]);
    p->rows++;
    p->wm = width;
}

enum { RUN, MAPS, DRY, BENCH };

static int batch(const char *blob_path, const char *path, int mode, unsigned reps)
{
    size_t n = 0;
    unsigned char *blob = load_file(blob_path, &n), head[16];
    FILE *f = fopen(path, "rb");
    vocr_textdet m;
    int rc;
    double total = 0;
    if (!blob || !f) { fprintf(stderr, "cannot read %s or %s\n", blob_path, path); return 2; }
    rc = vocr_textdet_load(&m, blob, n);
    if (rc) { fprintf(stderr, "%s: vocr_textdet_load %d\n", blob_path, rc); return 2; }
    while (fread(head, 1, 16, f) == 16) {
        unsigned w = rd32(head), h = rd32(head + 4), bpp = rd32(head + 8), flags = rd32(head + 12), i;
        size_t bytes = (size_t)w * h * bpp, stride = (size_t)w * bpp, need, lneed;
        unsigned char *px = (unsigned char *)malloc(bytes ? bytes : 1u), *buf = px, *arena, *base;
        VocrLinesImage im;
        VocrTextdetSet d;
        if (!px || fread(px, 1, bytes, f) != bytes) { fprintf(stderr, "truncated batch\n"); return 2; }
        if (flags & 4u) {                       /* a padded stride, junk past each row */
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
        need = vocr_textdet_arena_bytes(&m, w, h);
        lneed = vocr_lines_det_arena_bytes(&m, w, h);
        if (flags & 2u) need = lneed;
        arena = (unsigned char *)malloc(need + 8u);
        if (!arena) return 2;
        base = arena + ((flags & 8u) ? 1u + w % 7u : 0u);
        if (mode == DRY) {
            printf("0 0 0\n");
        } else if (mode == MAPS) {
            MapPrint mp = {0, 0};
            printf("X");
            rc = vocr_textdet_maps(&m, &im, base, need, print_map_row, &mp);
            printf("\n%d %u %u\n", rc, mp.rows, mp.wm);
        } else if (mode == BENCH) {
            double best = 1e30;
            unsigned r;
            for (r = 0; r < reps; r++) {
                clock_t t0 = clock();
                double dt;
                rc = vocr_textdet_find(&m, &im, base, need, &d);
                dt = (double)(clock() - t0) / CLOCKS_PER_SEC;
                if (dt < best) best = dt;
            }
            total += best;
            printf("%d %u %.1f\n", rc, d.count, best * 1e6);
        } else {
            unsigned char *slots = (unsigned char *)malloc(2u * (size_t)w * bpp + 1u);
            Source s;
            VocrLinesRows src;
            if (!slots) return 2;
            source_init(&s, &im, slots, NULL, 0);
            rows_of(&src, &im, &s);
            if (flags & 2u) {
                VocrLineSet ls;
                rc = (flags & 1u) ? vocr_find_lines_det_rows(&src, VOCR_LINES_TEXTDET, &m, base, need, &ls)
                                  : vocr_find_lines_det(&im, VOCR_LINES_TEXTDET, &m, base, need, &ls);
                printf("%d %u %u", rc, ls.count, ls.truncated);
                for (i = 0; i < ls.count; i++)
                    printf(" %u %u %u %u %u 0", ls.lines[i].x, ls.lines[i].y, ls.lines[i].w, ls.lines[i].h,
                           ls.lines[i].dark);
            } else {
                rc = (flags & 1u) ? vocr_textdet_find_rows(&m, &src, base, need, &d)
                                  : vocr_textdet_find(&m, &im, base, need, &d);
                printf("%d %u %u", rc, d.count, d.truncated);
                for (i = 0; i < d.count; i++)
                    printf(" %u %u %u %u %u %u", d.lines[i].x, d.lines[i].y, d.lines[i].w, d.lines[i].h,
                           d.lines[i].dark, d.vertical[i]);
            }
            printf("\n");
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
    if (argc == 4 && !strcmp(argv[1], "--maps")) return batch(argv[2], argv[3], MAPS, 0);
    if (argc == 4 && !strcmp(argv[1], "--dry")) return batch(argv[2], argv[3], DRY, 0);
    if (argc == 5 && !strcmp(argv[1], "--bench")) return batch(argv[2], argv[3], BENCH, (unsigned)strtoul(argv[4], NULL, 10));
    if (argc >= 3 && !strcmp(argv[1], "--load")) {
        int i;
        for (i = 2; i < argc; i++) {
            size_t n = 0;
            unsigned char *b = load_file(argv[i], &n);
            vocr_textdet m;
            printf("%d\n", b ? vocr_textdet_load(&m, b, n) : -100);
            free(b);
        }
        return 0;
    }
    if (argc == 5 && (!strcmp(argv[1], "--arena") || !strcmp(argv[1], "--arenas"))) {
        size_t n = 0;
        unsigned char *b = load_file(argv[2], &n);
        vocr_textdet m;
        unsigned w = (unsigned)strtoul(argv[3], NULL, 10), h = (unsigned)strtoul(argv[4], NULL, 10);
        if (!b || vocr_textdet_load(&m, b, n)) { fprintf(stderr, "cannot load %s\n", argv[2]); return 2; }
        if (!strcmp(argv[1], "--arena"))
            printf("%lu\n", (unsigned long)vocr_textdet_arena_bytes(&m, w, h));
        else
            printf("%lu %lu %lu\n", (unsigned long)vocr_textdet_arena_bytes(&m, w, h),
                   (unsigned long)vocr_lines_arena_bytes(w, h), (unsigned long)vocr_lines_det_arena_bytes(&m, w, h));
        free(b);
        return 0;
    }
    if (argc == 4 && !strcmp(argv[1], "--batch")) return batch(argv[2], argv[3], RUN, 0);
    fprintf(stderr, "usage: %s --selftest CASES BLOB... | --batch BLOB FILE | --maps BLOB FILE | --dry BLOB FILE |\n"
                    "       --load BLOB... | --arena BLOB W H | --arenas BLOB W H | --bench BLOB FILE REPS\n", argv[0]);
    return 2;
}
