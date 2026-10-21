/* Host driver and self-test of the full-frame text spotter for VTSP version 1, 2 and 3 blobs (vocr_textspot3.c;
 * issue #108). Not library code: it has its own main() and uses stdio and malloc. It is runtime/vocr_textspot2_test.c
 * on the vocr_textspot3_* calls, with the same record format and output lines (so eval/textspot.py's C parity and
 * eval/textspot_a9.py's batches read it as they read that driver; for a version-1 or version-2 blob it prints what
 * vocr_textspot2_test prints), plus --info:
 *
 *   vocr_textspot3_test --selftest CASES BLOB...  invariants on procedural frames, per blob: the row-callback
 *                                                 form equals the in-memory form (with a callback that poisons
 *                                                 each row once the next is asked, and the promised access
 *                                                 pattern: the frame's rows once each in order, then each box's
 *                                                 rows in output order; with VOCR_TEXTSPOT_NO_DARK the frame's
 *                                                 rows only), GREY equals RGBA of the same grey, padded strides,
 *                                                 every arena alignment and a junk-filled arena (determinism),
 *                                                 the exact arena malloc'd to size at every alignment
 *                                                 of its start (one byte less: VOCR_LINES_E_ARENA), NULL rows, the flag, a well-formed set
 *                                                 (boxes in the frame, blocks partitioning them in order, the pick
 *                                                 the twin's rule), edge sizes 1x1 .. 2048 and odd sizes,
 *                                                 damaged blobs, argument errors; the arena functions (version 3:
 *                                                 over a grid of widths and heights, every frame up to a size fits
 *                                                 its up-to arena, which is one frame's need, and both grow with
 *                                                 each side)
 *   vocr_textspot3_test --batch BLOB FILE         one output line per record (below)
 *   vocr_textspot3_test --maps BLOB FILE          the maps the decode reads per record (version 1: the head's;
 *                                                 version 2: the stride-4 maps of the d2s head): a line "X" +
 *                                                 hm * wm cells of K O A1 A2 (row-major), then a line "rc hm wm"
 *   vocr_textspot3_test --dry BLOB FILE           reads the records, runs nothing ("0 0 0 -1 0" each): the
 *                                                 baseline of the A9 instruction counts (--empty: the same, spelt
 *                                                 as long as --batch)
 *   vocr_textspot3_test --load BLOB...            vocr_textspot3_load's return code per file
 *   vocr_textspot3_test --info BLOB...            per file: rc version outputs net_stride stride min_cells
 *                                                 half_above conf_min layers blob_bytes net_bytes half_above_h
 *                                                 ops slots (layers: version 3's stream layers; ops and slots 0
 *                                                 for versions 1 and 2)
 *   vocr_textspot3_test --arena BLOB W H          vocr_textspot3_arena_bytes (one integer)
 *   vocr_textspot3_test --frame-arena BLOB W H    vocr_textspot3_frame_arena_bytes (one integer)
 *   vocr_textspot3_test --bench BLOB FILE REPS    host time per record (best of REPS), in-memory form
 *
 * FILE records: struct.pack("<4I", width, height, bpp, flags), then width * height * bpp pixel bytes (bpp 1 =
 * GREY, 4 = RGBA). Flags: 1 the row-callback form (a callback that poisons each row once the next is asked),
 * 2 VOCR_TEXTSPOT_NO_DARK, 4 a padded stride, 8 an arena at an odd address, 16 an arena of
 * vocr_textspot3_arena_bytes(w, h) (default: exactly vocr_textspot3_frame_arena_bytes(w, h), malloc'd to size).
 * Output per record, one line: rc count block_count pick truncated, then x y w h vertical conf dark block per
 * box, then x y w h vertical first count per block, space separated.
 */
#include "vocr_textspot3.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "textspot3: line %d: %s (blob %s, case %u, %ux%u bpp %u)\n", \
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

/* ---------------------------------------------------------------- procedural frames */

static void put(unsigned char *p, unsigned bpp, unsigned v)
{
    unsigned k;
    for (k = 0; k < bpp; k++) p[k] = (unsigned char)(bpp == 4u && k < 3u ? (v + k * 7u) & 255u : v);
}

/* A plain, gradient or noisy ground, boxes, rows of text-like strokes (horizontal and vertical, some close
 * together, some in stacks like a dialogue box) and blobs. */
static void draw(unsigned char *p, unsigned w, unsigned h, size_t stride, unsigned bpp)
{
    unsigned x, y, i, lines = below(12), cols = below(4), boxes = below(3), blobs = below(4);
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
        unsigned stack = below(3) ? 1u : 2u + below(3), r;
        b1 = b0 + below(span - b0 + 1u);
        for (r = 0; r < stack; r++, a0 += lh + 2u + below(lh))
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

static int same_box(const VocrTextspotBox *a, const VocrTextspotBox *b, int dark)
{
    return a->x == b->x && a->y == b->y && a->w == b->w && a->h == b->h && a->vertical == b->vertical &&
           a->conf == b->conf && a->block == b->block && (!dark || a->dark == b->dark);
}

static int same_set(const VocrTextspotSet *a, const VocrTextspotSet *b, int dark)
{
    unsigned i;
    if (a->count != b->count || a->block_count != b->block_count || a->pick != b->pick ||
        a->truncated != b->truncated)
        return 0;
    for (i = 0; i < a->count; i++)
        if (!same_box(&a->boxes[i], &b->boxes[i], dark)) return 0;
    for (i = 0; i < a->block_count; i++) {
        const VocrTextspotBlock *p = &a->blocks[i], *q = &b->blocks[i];
        if (p->x != q->x || p->y != q->y || p->w != q->w || p->h != q->h || p->vertical != q->vertical ||
            p->first != q->first || p->count != q->count || p->reserved != q->reserved)
            return 0;
    }
    return 1;
}

static int is_empty(const VocrTextspotSet *s)
{
    return !s->count && !s->block_count && s->pick == -1 && !s->truncated;
}

/* Boxes inside the frame and non-empty, flags 0/1 (dark 0 with nodark), conf 1..127; blocks partitioning the boxes
 * in order with their union boxes, one orientation each; the pick the largest doubled-in-the-lower-half area. */
static int well_formed(const VocrTextspotSet *d, unsigned w, unsigned h, int nodark)
{
    unsigned i, k, next = 0;
    uint32_t best = 0;
    int pick = -1;
    unsigned best_y = 0;
    if (d->count > VOCR_TEXTSPOT_MAX_BOXES || d->block_count > d->count || d->truncated > 1u) return 0;
    if (d->truncated && d->count != VOCR_TEXTSPOT_MAX_BOXES) return 0;
    if (!d->count) return d->block_count == 0 && d->pick == -1;
    for (i = 0; i < d->count; i++) {
        const VocrTextspotBox *b = &d->boxes[i];
        if (!b->w || !b->h || b->x >= w || b->y >= h || b->w > w - b->x || b->h > h - b->y) return 0;
        if (b->vertical > 1u || b->dark > 1u || (nodark && b->dark) || !b->conf || b->conf > 127u) return 0;
        if (b->block >= d->block_count) return 0;
    }
    for (k = 0; k < d->block_count; k++) {
        const VocrTextspotBlock *bk = &d->blocks[k];
        unsigned x0 = 0xFFFFu, y0 = 0xFFFFu, x1 = 0, y1 = 0;
        uint32_t score = 0;
        if (bk->first != next || !bk->count || bk->first + bk->count > d->count || bk->reserved) return 0;
        for (i = bk->first; i < (unsigned)bk->first + bk->count; i++) {
            const VocrTextspotBox *b = &d->boxes[i];
            if (b->block != k || b->vertical != bk->vertical) return 0;
            if (b->x < x0) x0 = b->x;
            if (b->y < y0) y0 = b->y;
            if ((unsigned)b->x + b->w > x1) x1 = (unsigned)b->x + b->w;
            if ((unsigned)b->y + b->h > y1) y1 = (unsigned)b->y + b->h;
            score += (uint32_t)b->w * b->h;
        }
        if (bk->x != x0 || bk->y != y0 || bk->w != x1 - x0 || bk->h != y1 - y0) return 0;
        if (2u * bk->y + bk->h >= h) score *= 2u;
        if (pick < 0 || score > best || (score == best && bk->y > best_y)) {
            pick = (int)k;
            best = score;
            best_y = bk->y;
        }
        next += bk->count;
    }
    return next == d->count && d->pick == pick;
}

/* ---------------------------------------------------------------- self-test */

static unsigned char *xmalloc(size_t n)
{
    return (unsigned char *)malloc(n ? n : 1u);
}

static int frame(const vocr_textspot3 *m, unsigned w, unsigned h, unsigned bpp)
{
    size_t stride = (size_t)w * bpp, need = vocr_textspot3_frame_arena_bytes(m, w, h);
    size_t most = vocr_textspot3_arena_bytes(m, w, h), big = most + 64u;
    unsigned pad = 1u + below(9), i, off;
    unsigned char *px = xmalloc(stride * h);
    unsigned char *other = xmalloc((size_t)w * h * (bpp == 1u ? 4u : 1u));
    unsigned char *padded = xmalloc((stride + pad) * h);
    unsigned char *exact = xmalloc(need);
    unsigned char *arena = xmalloc(big + 8u);
    unsigned char *slots = xmalloc(2u * stride);
    unsigned cap = h + VOCR_TEXTSPOT_MAX_BOXES * h + 8u;
    unsigned *log = (unsigned *)malloc(cap * sizeof(unsigned));
    VocrLinesImage im, im2, imp;
    VocrLinesRows src;
    Source s;
    VocrTextspotSet a, b;
    cur_w = w;
    cur_h = h;
    cur_bpp = bpp;
    REQUIRE(px && other && padded && exact && arena && slots && log && need > 7u && most >= need);
    draw(px, w, h, stride, bpp);
    im.pixels = px; im.width = w; im.height = h; im.stride = stride; im.bpp = bpp;

    /* the exact arena, malloc'd to its size (ASan sees any byte past it), at every alignment of its start: the
     * arena's bytes need no alignment, and the aligned base the runtime takes must stay inside them */
    REQUIRE(vocr_textspot3_find(m, &im, 0, exact, need, &a) == VOCR_LINES_OK && well_formed(&a, w, h, 0));
    for (off = 1; off < 8u; off++) {
        unsigned char *ex = xmalloc(need + off);
        REQUIRE(ex != NULL);
        memset(ex, (int)(0x5Au + 29u * off), need + off);
        REQUIRE(vocr_textspot3_find(m, &im, 0, ex + off, need, &b) == VOCR_LINES_OK && same_set(&a, &b, 1));
        free(ex);
    }
    /* determinism: again, on an arena full of junk, at every alignment, and with the arena for every frame up to
     * this size */
    for (off = 0; off < 8u; off++) {
        memset(arena, (int)(0x31u + 17u * off), big + 8u);
        REQUIRE(vocr_textspot3_find(m, &im, 0, arena + off, off & 1u ? most : need, &b) == VOCR_LINES_OK &&
                same_set(&a, &b, 1));
    }
    /* one byte short: the arena error, an empty set */
    REQUIRE(vocr_textspot3_find(m, &im, 0, arena, need - 1u, &b) == VOCR_LINES_E_ARENA && is_empty(&b));
    /* without dark flags: the same boxes, every dark flag 0 */
    REQUIRE(vocr_textspot3_find(m, &im, VOCR_TEXTSPOT_NO_DARK, arena + 3, need, &b) == VOCR_LINES_OK &&
            same_set(&a, &b, 0) && well_formed(&b, w, h, 1));
    REQUIRE(vocr_textspot3_find(m, &im, 2u, arena, need, &b) == VOCR_LINES_E_ARG && is_empty(&b));
    /* the row-callback form: the same set; rows 0..h-1 once each in order, then each box's rows once */
    source_init(&s, &im, slots, log, cap);
    rows_of(&src, &im, &s);
    REQUIRE(vocr_textspot3_find_rows(m, &src, 0, arena + below(8), need, &b) == VOCR_LINES_OK && same_set(&a, &b, 1));
    {
        unsigned k = 0, y;
        for (y = 0; y < h; y++, k++) REQUIRE(k < s.calls && log[k] == y);
        for (i = 0; i < a.count; i++)
            for (y = a.boxes[i].y; y < (unsigned)a.boxes[i].y + a.boxes[i].h; y++, k++)
                REQUIRE(k < s.calls && log[k] == y);
        REQUIRE(k == s.calls);
    }
    /* the flag: the frame's rows once each, nothing more */
    source_init(&s, &im, slots, log, cap);
    rows_of(&src, &im, &s);
    REQUIRE(vocr_textspot3_find_rows(m, &src, VOCR_TEXTSPOT_NO_DARK, arena, need, &b) == VOCR_LINES_OK &&
            same_set(&a, &b, 0) && s.calls == h);
    for (i = 0; i < h; i++) REQUIRE(log[i] == i);
    /* a row that cannot be read: VOCR_LINES_E_SOURCE and no boxes */
    {
        unsigned total = h + 1u;
        for (i = 0; i < a.count; i++) total += a.boxes[i].h;
        source_init(&s, &im, slots, NULL, 0);
        s.fail_at = 1u + below(total - 1u);
        rows_of(&src, &im, &s);
        REQUIRE(vocr_textspot3_find_rows(m, &src, 0, arena, need, &b) == VOCR_LINES_E_SOURCE && is_empty(&b));
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
    REQUIRE(vocr_textspot3_find(m, &im2, 0, arena, big, &b) == VOCR_LINES_OK && same_set(&a, &b, 1));
    /* a padded stride with junk past each row */
    for (i = 0; i < h; i++) {
        memcpy(padded + i * (stride + pad), px + i * stride, stride);
        memset(padded + i * (stride + pad) + stride, 0xA5, pad);
    }
    imp = im;
    imp.pixels = padded;
    imp.stride = stride + pad;
    REQUIRE(vocr_textspot3_find(m, &imp, 0, arena, big, &b) == VOCR_LINES_OK && same_set(&a, &b, 1));
    imp.stride = stride - 1u;                              /* too short */
    REQUIRE(vocr_textspot3_find(m, &imp, 0, arena, big, &b) == VOCR_LINES_E_ARG && is_empty(&b));
    free(px); free(other); free(padded); free(exact); free(arena); free(slots); free(log);
    return 0;
}

static int arena_checks(const vocr_textspot3 *m)
{
    unsigned w, h;
    size_t prev = 0;
    for (w = 1; w <= VOCR_TEXTSPOT_MAX_SIDE; w += 1u + w / 8u) {
        size_t a = vocr_textspot3_arena_bytes(m, w, 1), f = vocr_textspot3_frame_arena_bytes(m, w, 1);
        REQUIRE(a >= prev && a >= f && f > 7u);                     /* never shrinks with the width */
        for (h = 2; h <= VOCR_TEXTSPOT_MAX_SIDE; h = h * 3u + 1u)
            REQUIRE(vocr_textspot3_arena_bytes(m, w, h) == a && vocr_textspot3_frame_arena_bytes(m, w, h) == f);
        REQUIRE(vocr_textspot3_arena_bytes(m, w, VOCR_TEXTSPOT_MAX_SIDE) == a);
        /* every frame up to w wide fits the arena of w */
        if (m->half_above && w > m->half_above)
            REQUIRE(a >= vocr_textspot3_frame_arena_bytes(m, m->half_above, 1));
        if (w > 1u) REQUIRE(a >= vocr_textspot3_frame_arena_bytes(m, w - 1u, 1));
        prev = a;
    }
    REQUIRE(vocr_textspot3_arena_bytes(m, 0, 5) == 0 && vocr_textspot3_arena_bytes(m, 5, 0) == 0);
    REQUIRE(vocr_textspot3_arena_bytes(m, VOCR_TEXTSPOT_MAX_SIDE + 1u, 5) == 0);
    REQUIRE(vocr_textspot3_arena_bytes(m, 5, VOCR_TEXTSPOT_MAX_SIDE + 1u) == 0);
    REQUIRE(vocr_textspot3_frame_arena_bytes(m, VOCR_TEXTSPOT_MAX_SIDE + 1u, 5) == 0);
    REQUIRE(vocr_textspot3_arena_bytes(NULL, 5, 5) == 0 && vocr_textspot3_frame_arena_bytes(NULL, 5, 5) == 0);
    return 0;
}

/* Version 3: the arena depends on both sides. Over a grid of sides: the up-to arena covers every frame up to the
 * size (each smaller side of the grid), grows with each side, and is the need of one frame: the largest frame read at
 * full size, or the frame itself when it is halved. */
static int arena_checks3(const vocr_textspot3 *m)
{
    static const unsigned side[] = {1, 2, 3, 8, 9, 31, 64, 271, 272, 273, 479, 480, 481, 544, 725, 961, 2048};
    unsigned n = sizeof(side) / sizeof(side[0]), a, b, c, d;
    for (a = 0; a < n; a++)
        for (b = 0; b < n; b++) {
            unsigned w = side[a], h = side[b];
            unsigned w1 = m->half_above && w > m->half_above ? m->half_above : w;
            unsigned h1 = m->half_above_h && h > m->half_above_h ? m->half_above_h : h;
            int halved = (m->half_above && w > m->half_above) || (m->half_above_h && h > m->half_above_h);
            size_t up = vocr_textspot3_arena_bytes(m, w, h), fr = vocr_textspot3_frame_arena_bytes(m, w, h);
            size_t full = vocr_textspot3_frame_arena_bytes(m, w1, h1);
            REQUIRE(fr > 7u && up >= fr && full > 7u);
            REQUIRE(up == (halved && fr > full ? fr : full));
            for (c = 0; c <= a; c++)
                for (d = 0; d <= b; d++) REQUIRE(up >= vocr_textspot3_frame_arena_bytes(m, side[c], side[d]));
            if (a) REQUIRE(up >= vocr_textspot3_arena_bytes(m, side[a - 1u], h));
            if (b) REQUIRE(up >= vocr_textspot3_arena_bytes(m, w, side[b - 1u]));
            if (!halved && a && b)                /* a frame read at full size: its need grows with each side */
                REQUIRE(fr >= vocr_textspot3_frame_arena_bytes(m, side[a - 1u], h) &&
                        fr >= vocr_textspot3_frame_arena_bytes(m, w, side[b - 1u]));
        }
    REQUIRE(vocr_textspot3_arena_bytes(m, 0, 5) == 0 && vocr_textspot3_arena_bytes(m, 5, 0) == 0);
    REQUIRE(vocr_textspot3_arena_bytes(m, VOCR_TEXTSPOT_MAX_SIDE + 1u, 5) == 0);
    REQUIRE(vocr_textspot3_arena_bytes(m, 5, VOCR_TEXTSPOT_MAX_SIDE + 1u) == 0);
    REQUIRE(vocr_textspot3_frame_arena_bytes(m, VOCR_TEXTSPOT_MAX_SIDE + 1u, 5) == 0);
    REQUIRE(vocr_textspot3_frame_arena_bytes(m, 5, 0) == 0);
    REQUIRE(vocr_textspot3_arena_bytes(NULL, 5, 5) == 0 && vocr_textspot3_frame_arena_bytes(NULL, 5, 5) == 0);
    return 0;
}

static int arg_checks(const vocr_textspot3 *m)
{
    unsigned char px[16] = {0}, arena[64];
    VocrLinesImage im;
    VocrTextspotSet d;
    vocr_textspot3 bad;
    size_t need;
    unsigned char *big;
    im.pixels = px; im.width = 4; im.height = 4; im.stride = 4; im.bpp = 1;
    d.count = 7;
    d.pick = 3;
    REQUIRE(vocr_textspot3_find(NULL, &im, 0, arena, sizeof(arena), &d) == VOCR_LINES_E_ARG && is_empty(&d));
    memset(&bad, 0, sizeof(bad));
    REQUIRE(vocr_textspot3_find(&bad, &im, 0, arena, sizeof(arena), &d) == VOCR_LINES_E_ARG);
    REQUIRE(vocr_textspot3_find(m, NULL, 0, arena, sizeof(arena), &d) == VOCR_LINES_E_ARG && is_empty(&d));
    REQUIRE(vocr_textspot3_find(m, &im, 0, arena, sizeof(arena), NULL) == VOCR_LINES_E_ARG);
    need = vocr_textspot3_arena_bytes(m, VOCR_TEXTSPOT_MAX_SIDE, 2);
    big = (unsigned char *)malloc(need + 8u);
    REQUIRE(big != NULL);
    REQUIRE(vocr_textspot3_find(m, &im, 0, NULL, need, &d) == VOCR_LINES_E_ARG);
    REQUIRE(vocr_textspot3_find(m, &im, 0x80u, big, need, &d) == VOCR_LINES_E_ARG);
    im.bpp = 3;
    im.stride = 12;
    REQUIRE(vocr_textspot3_find(m, &im, 0, big, need, &d) == VOCR_LINES_E_ARG);
    im.bpp = 1;
    im.stride = 4;
    im.width = 0;
    REQUIRE(vocr_textspot3_find(m, &im, 0, big, need, &d) == VOCR_LINES_E_ARG);
    im.width = 4;
    im.height = 0;
    REQUIRE(vocr_textspot3_find(m, &im, 0, big, need, &d) == VOCR_LINES_E_ARG);
    {
        /* the largest side is fine, one more is not */
        unsigned char *row = (unsigned char *)calloc(VOCR_TEXTSPOT_MAX_SIDE + 1u, 2);
        REQUIRE(row != NULL);
        im.pixels = row;
        im.height = 2;
        im.width = VOCR_TEXTSPOT_MAX_SIDE;
        im.stride = VOCR_TEXTSPOT_MAX_SIDE + 1u;
        REQUIRE(vocr_textspot3_find(m, &im, 0, big, need, &d) == VOCR_LINES_OK);
        im.width = VOCR_TEXTSPOT_MAX_SIDE + 1u;
        REQUIRE(vocr_textspot3_find(m, &im, 0, big, need, &d) == VOCR_LINES_E_ARG && is_empty(&d));
        free(row);
    }
    {
        VocrLinesRows r;
        memset(&r, 0, sizeof(r));
        r.width = 4; r.height = 4; r.bpp = 1;
        REQUIRE(vocr_textspot3_find_rows(m, &r, 0, big, need, &d) == VOCR_LINES_E_ARG && is_empty(&d));
        REQUIRE(vocr_textspot3_find_rows(m, NULL, 0, big, need, &d) == VOCR_LINES_E_ARG);
        REQUIRE(vocr_textspot3_find_rows(m, &r, 0, big, need, NULL) == VOCR_LINES_E_ARG);
    }
    im.pixels = px; im.width = 4; im.height = 4; im.stride = 4; im.bpp = 1;
    REQUIRE(vocr_textspot3_maps(m, &im, big, need, NULL, NULL) == VOCR_LINES_E_ARG);
    free(big);
    REQUIRE(vocr_textspot3_load(NULL, px, sizeof(px)) == VOCR_LINES_E_ARG);
    REQUIRE(vocr_textspot3_load(&bad, NULL, 0) == VOCR_LINES_E_ARG);
    REQUIRE(vocr_textspot3_load(&bad, px, sizeof(px)) == VOCR_TEXTSPOT_E_BLOB && !bad.loaded);
    return 0;
}

/* Damaged copies of a good blob are refused: a flipped byte anywhere fails the CRC (of the VTSP blob, or first of
 * the VTDT one when the outer CRC is fixed), a cut one the size. */
static uint32_t crc32_of(const unsigned char *p, size_t n)
{
    uint32_t c = 0xFFFFFFFFu;
    while (n--) {
        int k;
        c ^= *p++;
        for (k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
    }
    return ~c;
}

static void put32(unsigned char *p, uint32_t v)
{
    p[0] = (unsigned char)v;
    p[1] = (unsigned char)(v >> 8);
    p[2] = (unsigned char)(v >> 16);
    p[3] = (unsigned char)(v >> 24);
}

static int damage_checks(const unsigned char *blob, size_t n)
{
    unsigned char *c = (unsigned char *)malloc(n);
    vocr_textspot3 t;
    unsigned i;
    REQUIRE(c != NULL);
    for (i = 0; i < 64u; i++) {
        size_t at = below((unsigned)n);
        memcpy(c, blob, n);
        c[at] ^= (unsigned char)(1u + below(255));
        REQUIRE(vocr_textspot3_load(&t, c, n) == VOCR_TEXTSPOT_E_BLOB && !t.loaded);
        if (at >= 48u) {                      /* inside the VTDT blob's CRC'd part: its own CRC refuses it */
            put32(c + 12, crc32_of(c + 16, n - 16u));
            REQUIRE(vocr_textspot3_load(&t, c, n) == VOCR_TEXTSPOT_E_BLOB && !t.loaded);
        }
    }
    memcpy(c, blob, n);
    REQUIRE(vocr_textspot3_load(&t, c, n - 1u) == VOCR_TEXTSPOT_E_BLOB);
    REQUIRE(vocr_textspot3_load(&t, c, n) == VOCR_LINES_OK && t.blob_bytes == n && t.net_bytes == n - 32u);
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
                                       {300, 3}, {599, 7}, {600, 7}, {601, 7}, {602, 9}, {2048, 3}, {3, 2048},
                                       {961, 545}, {960, 544}, {480, 272}, {725, 544}, {870, 544}};
    int bi;
    unsigned total = 0, v2 = 0, v3 = 0;
    if (nblobs < 1) { fprintf(stderr, "--selftest needs blob files\n"); return 2; }
    for (bi = 0; bi < nblobs; bi++) {
        size_t n = 0;
        unsigned char *blob = load_file(blobs[bi], &n);
        vocr_textspot3 m;
        unsigned e;
        blob_name = blobs[bi];
        current = 0;
        REQUIRE(blob != NULL);
        REQUIRE(vocr_textspot3_load(&m, blob, n) == VOCR_LINES_OK && m.blob_bytes == n);
        /* version 1: 4 maps at the network's stride; versions 2 and 3: 16 channels at stride 8, maps at stride 4 */
        REQUIRE((m.version == 1u && m.outputs == 4u && m.stride == m.net_stride && !m.n_ops) ||
                ((m.version == VOCR_TEXTSPOT3_VERSION_D2S || m.version == VOCR_TEXTSPOT3_VERSION_MS) &&
                 m.outputs == 16u && m.net_stride == 8u && m.stride == 4u &&
                 (m.version == VOCR_TEXTSPOT3_VERSION_MS) == (m.n_ops > 0u)));
        v2 += m.version == VOCR_TEXTSPOT3_VERSION_D2S;
        v3 += m.version == VOCR_TEXTSPOT3_VERSION_MS;
        if ((m.version == VOCR_TEXTSPOT3_VERSION_MS ? arena_checks3(&m) : arena_checks(&m)) || arg_checks(&m) ||
            damage_checks(blob, n))
            return 1;
        for (e = 0; e < sizeof(edge) / sizeof(edge[0]) && current < cases; e++, current++)
            if (frame(&m, edge[e][0], edge[e][1], below(2) ? 4u : 1u)) return 1;
        for (; current < cases; current++) {
            /* one draw per statement: the cases must not depend on the order of evaluation */
            unsigned w = 1u + below(current % 5u == 0 ? 960u : 400u);
            unsigned h = 1u + below(current % 7u == 0 ? 544u : 160u);
            unsigned bpp = below(2) ? 4u : 1u;
            if (frame(&m, w, h, bpp)) return 1;
        }
        total += current;
        free(blob);
    }
    printf("TEXTSPOT3_SELFTEST_OK %d blobs (%u of version 2, %u of version 3), %u frames\n", nblobs, v2, v3, total);
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

static void print_map_row(void *user, unsigned i, const signed char *maps, unsigned width)
{
    MapPrint *p = (MapPrint *)user;
    unsigned j, c;
    (void)i;
    for (j = 0; j < width; j++)
        for (c = 0; c < 4u; c++) printf(" %d", maps[c * width + j]);
    p->rows++;
    p->wm = width;
}

static void print_set(int rc, const VocrTextspotSet *d)
{
    unsigned i;
    printf("%d %u %u %d %u", rc, d->count, d->block_count, d->pick, d->truncated);
    for (i = 0; i < d->count; i++) {
        const VocrTextspotBox *b = &d->boxes[i];
        printf(" %u %u %u %u %u %u %u %u", b->x, b->y, b->w, b->h, b->vertical, b->conf, b->dark, b->block);
    }
    for (i = 0; i < d->block_count; i++) {
        const VocrTextspotBlock *b = &d->blocks[i];
        printf(" %u %u %u %u %u %u %u", b->x, b->y, b->w, b->h, b->vertical, b->first, b->count);
    }
    printf("\n");
}

enum { RUN, MAPS, DRY, BENCH };

static int batch(const char *blob_path, const char *path, int mode, unsigned reps)
{
    size_t n = 0;
    unsigned char *blob = load_file(blob_path, &n), head[16];
    FILE *f = fopen(path, "rb");
    vocr_textspot3 m;
    int rc;
    double total = 0;
    if (!blob || !f) { fprintf(stderr, "cannot read %s or %s\n", blob_path, path); return 2; }
    rc = vocr_textspot3_load(&m, blob, n);
    if (rc) { fprintf(stderr, "%s: vocr_textspot_load %d\n", blob_path, rc); return 2; }
    while (fread(head, 1, 16, f) == 16) {
        unsigned w = rd32(head), h = rd32(head + 4), bpp = rd32(head + 8), flags = rd32(head + 12), i;
        size_t bytes = (size_t)w * h * bpp, stride = (size_t)w * bpp, need;
        unsigned char *px = xmalloc(bytes), *buf = px, *arena, *base;
        unsigned sflags = (flags & 2u) ? VOCR_TEXTSPOT_NO_DARK : 0u;
        VocrLinesImage im;
        VocrTextspotSet d;
        if (!px || fread(px, 1, bytes, f) != bytes) { fprintf(stderr, "truncated batch\n"); return 2; }
        if (flags & 4u) {                       /* a padded stride, junk past each row */
            size_t ps = stride + 3u + w % 5u;
            buf = xmalloc(ps * h + 1u);
            if (!buf) return 2;
            for (i = 0; i < h; i++) {
                memcpy(buf + i * ps, px + i * stride, stride);
                memset(buf + i * ps + stride, 0xA5, ps - stride);
            }
            stride = ps;
        }
        im.pixels = buf; im.width = w; im.height = h; im.stride = stride; im.bpp = bpp;
        need = (flags & 16u) ? vocr_textspot3_arena_bytes(&m, w, h) : vocr_textspot3_frame_arena_bytes(&m, w, h);
        arena = xmalloc(need + ((flags & 8u) ? 8u : 0u));
        if (!arena) return 2;
        base = arena + ((flags & 8u) ? 1u + w % 7u : 0u);
        if (mode == DRY) {
            printf("0 0 0 -1 0\n");
        } else if (mode == MAPS) {
            MapPrint mp = {0, 0};
            printf("X");
            rc = vocr_textspot3_maps(&m, &im, base, need, print_map_row, &mp);
            printf("\n%d %u %u\n", rc, mp.rows, mp.wm);
        } else if (mode == BENCH) {
            double best = 1e30;
            unsigned r;
            for (r = 0; r < reps; r++) {
                clock_t t0 = clock();
                double dt;
                rc = vocr_textspot3_find(&m, &im, sflags, base, need, &d);
                dt = (double)(clock() - t0) / CLOCKS_PER_SEC;
                if (dt < best) best = dt;
            }
            total += best;
            printf("%d %u %.1f\n", rc, d.count, best * 1e6);
        } else {
            unsigned char *slots = xmalloc(2u * (size_t)w * bpp + 1u);
            Source s;
            VocrLinesRows src;
            if (!slots) return 2;
            source_init(&s, &im, slots, NULL, 0);
            rows_of(&src, &im, &s);
            rc = (flags & 1u) ? vocr_textspot3_find_rows(&m, &src, sflags, base, need, &d)
                              : vocr_textspot3_find(&m, &im, sflags, base, need, &d);
            print_set(rc, &d);
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
    if (argc == 4 && (!strcmp(argv[1], "--dry") || !strcmp(argv[1], "--empty")))
        return batch(argv[2], argv[3], DRY, 0);
    if (argc == 5 && !strcmp(argv[1], "--bench")) return batch(argv[2], argv[3], BENCH, (unsigned)strtoul(argv[4], NULL, 10));
    if (argc >= 3 && !strcmp(argv[1], "--info")) {
        int i;
        for (i = 2; i < argc; i++) {
            size_t n = 0;
            unsigned char *b = load_file(argv[i], &n);
            vocr_textspot3 m;
            int rc = b ? vocr_textspot3_load(&m, b, n) : -100;
            if (rc)
                printf("%d\n", rc);
            else
                printf("0 %u %u %u %u %u %u %u %u %lu %lu %u %u %u\n", m.version, m.outputs, m.net_stride, m.stride,
                       m.min_cells, m.half_above, m.conf_min, m.n_layers, m.blob_bytes, m.net_bytes, m.half_above_h,
                       m.n_ops, m.n_ops ? m.n_slots : 0u);
            free(b);
        }
        return 0;
    }
    if (argc >= 3 && !strcmp(argv[1], "--load")) {
        int i;
        for (i = 2; i < argc; i++) {
            size_t n = 0;
            unsigned char *b = load_file(argv[i], &n);
            vocr_textspot3 m;
            printf("%d\n", b ? vocr_textspot3_load(&m, b, n) : -100);
            free(b);
        }
        return 0;
    }
    if (argc == 5 && (!strcmp(argv[1], "--arena") || !strcmp(argv[1], "--frame-arena"))) {
        size_t n = 0;
        unsigned char *b = load_file(argv[2], &n);
        vocr_textspot3 m;
        unsigned w = (unsigned)strtoul(argv[3], NULL, 10), h = (unsigned)strtoul(argv[4], NULL, 10);
        if (!b || vocr_textspot3_load(&m, b, n)) { fprintf(stderr, "cannot load %s\n", argv[2]); return 2; }
        printf("%lu\n", (unsigned long)(!strcmp(argv[1], "--arena") ? vocr_textspot3_arena_bytes(&m, w, h)
                                                                      : vocr_textspot3_frame_arena_bytes(&m, w, h)));
        free(b);
        return 0;
    }
    if (argc == 4 && !strcmp(argv[1], "--batch")) return batch(argv[2], argv[3], RUN, 0);
    fprintf(stderr, "usage: %s --selftest CASES BLOB... | --batch BLOB FILE | --maps BLOB FILE | --dry BLOB FILE |\n"
                    "       --load BLOB... | --info BLOB... | --arena BLOB W H | --frame-arena BLOB W H |\n"
                    "       --bench BLOB FILE REPS\n",
            argv[0]);
    return 2;
}
