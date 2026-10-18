/* Host driver and self-test of the stroke split (vocr_lines_split.c). Not library code.
 *
 *   vocr_lines_split_test --selftest [CASES]   procedural regions (plain, noisy and line-art grounds behind
 *                                              rows of strokes): flags 0 is the identity, the in-memory and
 *                                              row-callback forms agree, splitting each box on its own strip
 *                                              equals splitting the region, lines stay inside their box, in
 *                                              order, with its columns and flag, the minimum misaligned arena
 *                                              gives the same lines, errors leave the set unchanged
 *   vocr_lines_split_test [--rows] BATCH       for each record: the finder, the split, then the refinement
 *   vocr_lines_split_test --bench BATCH REPS   host time of the split alone (best of REPS per record)
 *   vocr_lines_split_test --arena W H          vocr_split_arena_bytes and vocr_lines_arena_bytes
 *
 * BATCH records: uint32 little-endian width, height, bpp (1 or 4), finder, stride, split flags, refine flags,
 * then stride * height pixel bytes. Output per record, one line: rc found split_count count truncated
 * rows_read, then x y w h dark per line of the final set (found = the finder's line count, split_count = the
 * count after the split, rows_read = the split's).
 */
#include "vocr_lines_refine.h"
#include "vocr_lines_split.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "split: line %d: %s (case %u)\n", __LINE__, #x, current); return 1; } } while (0)

static unsigned current;
static uint32_t rng = 0x5eed9898u;
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
    unsigned calls, last, turn, fail_at, order_errors, passes;
} Source;

static const unsigned char *source_row(void *user, unsigned y)
{
    Source *s = (Source *)user;
    size_t n = (size_t)s->im->width * s->im->bpp;
    unsigned char *out;
    if (y >= s->im->height) { s->order_errors++; return NULL; }
    if (!s->calls || y <= s->last) s->passes++;
    s->last = y;
    s->calls++;
    if (s->fail_at && s->calls >= s->fail_at) return NULL;
    memset(s->slot[s->turn ^ 1u], 0xA5, n);
    out = s->slot[s->turn];
    memcpy(out, s->im->pixels + (size_t)y * s->im->stride, n);
    s->turn ^= 1u;
    return out;
}

static void put(unsigned char *p, size_t stride, unsigned bpp, unsigned x, unsigned y, unsigned v)
{
    unsigned k;
    for (k = 0; k < bpp; k++) p[(size_t)y * stride + (size_t)x * bpp + k] = (unsigned char)(k < 3 ? v : 255u);
}

/* A procedural region: a plain, gradient or noisy ground, or line art and blobs as a scene behind a translucent
 * box; rows of outlined or plain strokes, some set close together, some with a short last row; frames. */
static void draw(unsigned char *p, unsigned w, unsigned h, size_t stride, unsigned bpp)
{
    unsigned x, y, k, i, rows = 1u + below(4), ground = below(4), paper = below(256);
    unsigned ink = (paper + 96u + below(64)) & 255u, outline = below(2), lh = 8u + below(30);
    unsigned pitch = lh + below(lh), y0 = below(h / 2u + 1u), x0 = below(w / 4u + 1u);
    for (y = 0; y < h; y++)
        for (x = 0; x < w; x++) {
            unsigned v = paper;
            if (ground == 1) v = (paper + x * 64u / (w ? w : 1u) + y) & 255u;
            if (ground == 2) v = (paper + below(40)) & 255u;
            for (k = 0; k < bpp; k++) p[(size_t)y * stride + (size_t)x * bpp + k] = (unsigned char)v;
        }
    if (ground == 3) {                        /* line art: random strokes and blobs at reduced contrast */
        for (i = 0; i < 6u + below(30); i++) {
            unsigned ax = below(w), ay = below(h), len = 4u + below(60), dir = below(4), v = (paper + 40u + below(80)) & 255u;
            for (k = 0; k < len; k++) {
                unsigned px = ax + (dir == 0 || dir == 2 ? k : 0u), py = ay + (dir == 1 || dir == 2 ? k : 0u);
                if (dir == 3) { px = ax + k; py = ay + k / 2u; }
                if (px < w && py < h) put(p, stride, bpp, px, py, v);
            }
        }
        for (i = 0; i < below(4); i++) {
            unsigned bx = below(w), by = below(h), bw = 3u + below(40), bh = 3u + below(30), v = below(256);
            for (y = by; y < by + bh && y < h; y++)
                for (x = bx; x < bx + bw && x < w; x++) put(p, stride, bpp, x, y, (v + (x ^ y)) & 255u);
        }
    }
    for (i = 0; i < rows; i++) {
        unsigned ry = y0 + i * pitch, rx1 = w - below(w / 3u + 1u), cw = 4u + below(lh);
        if (i + 1u == rows && below(2)) rx1 = x0 + (rx1 - x0) / (2u + below(4));
        if (ry + lh > h || rx1 <= x0) break;
        for (y = ry; y < ry + lh; y++)
            for (x = x0; x < rx1; x++) {
                unsigned cx = (x - x0) % (cw + 2u), cy = y - ry;
                int on = cx < cw && (cx % 4u < 2u || cy % 6u < 2u || (cx + cy) % 7u == 0u);
                if (on) put(p, stride, bpp, x, y, ink);
                else if (outline && cx < cw + 1u && (cx % 4u == 2u || cy % 6u == 2u)) put(p, stride, bpp, x, y, 255u - ink);
            }
    }
    if (below(3) == 0) {                      /* a frame */
        unsigned v = below(256);
        for (x = 0; x < w; x++) { put(p, stride, bpp, x, 0, v); put(p, stride, bpp, x, h - 1u, v); }
    }
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

static int r_same(const VocrLine *a, const VocrLine *b)
{
    return a->x == b->x && a->y == b->y && a->w == b->w && a->h == b->h && a->dark == b->dark;
}

/* The lines one box became: inside it, with its columns and flag, top to bottom and not overlapping. */
static int box_ok(const VocrLine *f, const VocrLineSet *one)
{
    unsigned i;
    if (!one->count || one->count > VOCR_LINES_MAX) return 0;
    for (i = 0; i < one->count; i++) {
        const VocrLine *r = &one->lines[i];
        if (r->x != f->x || r->w != f->w || r->dark != f->dark || !r->h || r->y < f->y || r->y + r->h > f->y + f->h)
            return 0;
        if (i && one->lines[i - 1u].y + one->lines[i - 1u].h > r->y) return 0;
    }
    return one->count == 1u ? r_same(&one->lines[0], f) : 1;   /* a box not split stays exactly as it is */
}

static int region(unsigned w, unsigned h, unsigned bpp, unsigned pad)
{
    size_t stride = (size_t)w * bpp + pad, need = vocr_lines_arena_bytes(w, h), small = vocr_split_arena_bytes(w, h);
    unsigned char *pixels = (unsigned char *)malloc(stride * h + 1u);
    unsigned char *arena = (unsigned char *)malloc(need + 3u), *arena2 = (unsigned char *)malloc(small + 3u);
    unsigned char *slots = (unsigned char *)malloc(2u * (size_t)w * bpp);
    VocrLinesImage im;
    VocrLinesRows src;
    Source s;
    VocrLineSet found, a, b, c, d, cat;
    unsigned i, j, at = below(4);
    unsigned long ra = 0, rb = 0, rc_rows = 0, expect = 0;
    int rc;
    REQUIRE(pixels && arena && arena2 && slots);
    REQUIRE(small <= need);
    draw(pixels, w, h, stride, bpp);
    im.pixels = pixels; im.width = w; im.height = h; im.stride = stride; im.bpp = bpp;
    src.width = w; src.height = h; src.bpp = bpp; src.row = source_row; src.user = &s;
    rc = vocr_find_lines(&im, below(2) ? VOCR_LINES_GRAD : VOCR_LINES_TEXTBOX, arena, need, &found);
    REQUIRE(rc == VOCR_LINES_OK);
    for (i = 0; i < found.count; i++)
        if (found.lines[i].h >= 16u) expect += 2u * found.lines[i].h;
    /* flags 0: untouched, nothing read */
    a = found;
    REQUIRE(vocr_split_lines(&im, 0, arena, need, &a, &ra) == VOCR_LINES_OK && same_set(&a, &found) && !ra);
    a = found;
    REQUIRE(vocr_split_lines(&im, VOCR_SPLIT_DEFAULT, arena, need, &a, &ra) == VOCR_LINES_OK && ra == expect);
    /* each box alone, on the whole region and on its own strip (the box made strip-relative): the same lines,
     * inside the box; their concatenation is the set's split when no box had to be capped */
    cat.count = 0;
    cat.truncated = found.truncated;
    for (i = 0; i < found.count; i++) {
        VocrLinesImage strip = im;
        VocrLineSet one, st;
        one.count = 1;
        one.truncated = 0;
        one.lines[0] = found.lines[i];
        st = one;
        st.lines[0].y = 0;
        strip.pixels = im.pixels + (size_t)found.lines[i].y * im.stride;
        strip.height = found.lines[i].h;
        REQUIRE(vocr_split_lines(&im, VOCR_SPLIT_DEFAULT, arena, need, &one, NULL) == VOCR_LINES_OK);
        REQUIRE(box_ok(&found.lines[i], &one));
        REQUIRE(vocr_split_lines(&strip, VOCR_SPLIT_DEFAULT, arena, need, &st, NULL) == VOCR_LINES_OK);
        REQUIRE(st.count == one.count);
        for (j = 0; j < st.count; j++) {
            st.lines[j].y += found.lines[i].y;
            if (cat.count < VOCR_LINES_MAX) cat.lines[cat.count] = one.lines[j];
            cat.count++;
        }
        REQUIRE(same_set(&st, &one));
    }
    if (cat.count <= VOCR_LINES_MAX) REQUIRE(same_set(&a, &cat));
    REQUIRE(a.count <= VOCR_LINES_MAX && a.count >= found.count);
    /* the minimum arena, misaligned: the same lines */
    b = found;
    REQUIRE(vocr_split_lines(&im, VOCR_SPLIT_DEFAULT, arena2 + 3u - at, small, &b, &rc_rows) == VOCR_LINES_OK);
    REQUIRE(same_set(&a, &b) && rc_rows == ra);
    /* the row-callback form: the same lines, each box read twice in increasing order */
    memset(&s, 0, sizeof(s));
    s.im = &im; s.slot[0] = slots; s.slot[1] = slots + (size_t)w * bpp;
    c = found;
    REQUIRE(vocr_split_lines_rows(&src, VOCR_SPLIT_DEFAULT, arena2 + at, small + 3u - at, &c, &rb) == VOCR_LINES_OK);
    REQUIRE(same_set(&a, &c) && !s.order_errors && rb == ra && s.calls == rb && s.passes <= 2u * found.count);
    /* the refinement after the split */
    d = a;
    REQUIRE(vocr_refine_lines(&im, VOCR_REFINE_DEFAULT, arena, need, &d, NULL) == VOCR_LINES_OK);
    /* a row that cannot be read: an error, the set unchanged */
    if (rb) {
        memset(&s, 0, sizeof(s));
        s.im = &im; s.slot[0] = slots; s.slot[1] = slots + (size_t)w * bpp; s.fail_at = 1u + below((unsigned)rb);
        c = found;
        rc = vocr_split_lines_rows(&src, VOCR_SPLIT_DEFAULT, arena, need, &c, NULL);
        REQUIRE(rc == VOCR_LINES_E_SOURCE && same_set(&c, &found));
    }
    /* errors leave the set unchanged */
    c = found;
    REQUIRE(vocr_split_lines(&im, 0x02u, arena, need, &c, NULL) == VOCR_LINES_E_ARG && same_set(&c, &found));
    REQUIRE(vocr_split_lines(&im, VOCR_SPLIT_DEFAULT, arena, small - 1u, &c, NULL) == VOCR_LINES_E_ARENA &&
            same_set(&c, &found));
    REQUIRE(vocr_split_lines(&im, VOCR_SPLIT_DEFAULT, NULL, need, &c, NULL) == VOCR_LINES_E_ARENA);
    if (found.count) {
        c = found;
        c.lines[0].h = h - c.lines[0].y + 1u;
        d = c;
        REQUIRE(vocr_split_lines(&im, VOCR_SPLIT_DEFAULT, arena, need, &c, NULL) == VOCR_LINES_E_ARG && same_set(&c, &d));
    }
    free(pixels); free(arena); free(arena2); free(slots);
    return 0;
}

static int selftest(unsigned cases)
{
    unsigned w, h;
    for (w = 1; w <= 1000; w += 37)
        for (h = 1; h <= 600; h += 23) REQUIRE(vocr_split_arena_bytes(w, h) <= vocr_lines_arena_bytes(w, h));
    for (w = 1; w <= 40; w++)
        for (h = 1; h <= 6000; h += 97) {
            REQUIRE(vocr_split_arena_bytes(w, h) <= vocr_lines_arena_bytes(w, h));
            REQUIRE(vocr_split_arena_bytes(h, w) <= vocr_lines_arena_bytes(h, w));
        }
    REQUIRE(vocr_split_arena_bytes(960, 544) <= vocr_lines_arena_bytes(960, 544));
    REQUIRE(vocr_split_arena_bytes(0, 5) == 0 && vocr_split_arena_bytes(5, 0) == 0);
    REQUIRE(vocr_split_lines(NULL, 1, NULL, 0, NULL, NULL) == VOCR_LINES_E_ARG);
    REQUIRE(vocr_split_lines_rows(NULL, 1, NULL, 0, NULL, NULL) == VOCR_LINES_E_ARG);
    for (current = 0; current < cases; current++) {
        unsigned rw = 8u + below(current % 5u == 0 ? 953u : 500u), rh = 8u + below(current % 7u == 0 ? 537u : 140u);
        unsigned bpp = below(2) ? VOCR_LINES_RGBA : VOCR_LINES_GREY;
        unsigned pad = below(3) * 4u;
        if (region(rw, rh, bpp, pad)) return 1;
    }
    printf("SPLIT_SELFTEST_OK %u regions\n", cases);
    return 0;
}

static unsigned rd32(const unsigned char *p)
{
    return (unsigned)p[0] | (unsigned)p[1] << 8 | (unsigned)p[2] << 16 | (unsigned)p[3] << 24;
}

static unsigned char *load(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    unsigned char *buf;
    long len;
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    len = ftell(f);
    fseek(f, 0, SEEK_SET);
    buf = (unsigned char *)malloc(len > 0 ? (size_t)len : 1u);
    if (!buf || fread(buf, 1, (size_t)len, f) != (size_t)len) { fclose(f); free(buf); return NULL; }
    fclose(f);
    *n = (size_t)len;
    return buf;
}

typedef struct {
    const unsigned char *p;
    size_t stride;
} MemRows;

static const unsigned char *mem_row(void *user, unsigned y)
{
    const MemRows *m = (const MemRows *)user;
    return m->p + (size_t)y * m->stride;
}

static int batch(const char *path, int use_rows, unsigned reps)
{
    size_t n = 0, off = 0;
    unsigned char *buf = load(path, &n);
    double total = 0;
    if (!buf) { fprintf(stderr, "cannot read %s\n", path); return 2; }
    while (off + 28u <= n) {
        unsigned w = rd32(buf + off), h = rd32(buf + off + 4), bpp = rd32(buf + off + 8);
        unsigned finder = rd32(buf + off + 12), sflags = rd32(buf + off + 20), rflags = rd32(buf + off + 24);
        size_t stride = rd32(buf + off + 16), need, i;
        unsigned char *arena;
        VocrLinesImage im;
        VocrLineSet found, set;
        unsigned long rows = 0;
        unsigned nsplit = 0;
        int rc;
        off += 28u;
        if (off + stride * h > n) { fprintf(stderr, "truncated batch\n"); free(buf); return 2; }
        im.pixels = buf + off; im.width = w; im.height = h; im.stride = stride; im.bpp = bpp;
        need = vocr_lines_arena_bytes(w, h);
        arena = (unsigned char *)malloc(need + 3u);
        if (!arena) { free(buf); return 2; }
        rc = vocr_find_lines(&im, (int)finder, arena, need, &found);
        set = found;
        if (!rc) {
            if (reps) {
                double best = 1e30;
                unsigned r;
                for (r = 0; r < reps; r++) {
                    clock_t t0 = clock();
                    set = found;
                    rc = vocr_split_lines(&im, sflags, arena, need, &set, &rows);
                    {
                        double dt = (double)(clock() - t0) / CLOCKS_PER_SEC;
                        if (dt < best) best = dt;
                    }
                }
                total += best;
            } else if (use_rows) {
                MemRows mr;
                VocrLinesRows src;
                mr.p = im.pixels; mr.stride = stride;
                src.width = w; src.height = h; src.bpp = bpp; src.row = mem_row; src.user = &mr;
                rc = vocr_split_lines_rows(&src, sflags, arena, need, &set, &rows);
            } else {
                rc = vocr_split_lines(&im, sflags, arena, need, &set, &rows);
            }
            nsplit = set.count;
            if (!rc && !reps) rc = vocr_refine_lines(&im, rflags, arena, need, &set, NULL);
        }
        if (!reps) {
            printf("%d %u %u %u %u %lu", rc, found.count, rc ? 0u : nsplit, rc ? 0u : set.count, set.truncated, rows);
            for (i = 0; !rc && i < set.count; i++)
                printf(" %u %u %u %u %u", set.lines[i].x, set.lines[i].y, set.lines[i].w, set.lines[i].h,
                       set.lines[i].dark);
            printf("\n");
        }
        free(arena);
        off += stride * h;
    }
    if (reps) printf("%.3f us total (best of %u per record)\n", total * 1e6, reps);
    free(buf);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc >= 2 && !strcmp(argv[1], "--selftest")) return selftest(argc > 2 ? (unsigned)strtoul(argv[2], NULL, 10) : 300u);
    if (argc == 4 && !strcmp(argv[1], "--arena")) {
        unsigned w = (unsigned)strtoul(argv[2], NULL, 10), h = (unsigned)strtoul(argv[3], NULL, 10);
        printf("%lu %lu\n", (unsigned long)vocr_split_arena_bytes(w, h), (unsigned long)vocr_lines_arena_bytes(w, h));
        return 0;
    }
    if (argc == 4 && !strcmp(argv[1], "--bench")) return batch(argv[2], 0, (unsigned)strtoul(argv[3], NULL, 10));
    if (argc == 3 && !strcmp(argv[1], "--rows")) return batch(argv[2], 1, 0);
    if (argc == 2) return batch(argv[1], 0, 0);
    fprintf(stderr, "usage: %s --selftest [CASES] | [--rows] BATCH | --bench BATCH REPS | --arena W H\n", argv[0]);
    return 2;
}
