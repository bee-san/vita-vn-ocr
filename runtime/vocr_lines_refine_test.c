/* Host driver and self-test of the line refinement (vocr_lines_refine.c). Not library code.
 *
 *   vocr_lines_refine_test --selftest [CASES]   procedural regions: flags 0 is the identity, the
 *                                               in-memory and row-callback forms agree, the one- and
 *                                               two-pass arenas agree, boxes only shrink, errors
 *   vocr_lines_refine_test [--rows] BATCH       for each record: the finder, then the refinement
 *   vocr_lines_refine_test --bench BATCH REPS   host time of the refinement alone (best of REPS)
 *   vocr_lines_refine_test --arena W H          vocr_refine_arena_bytes and vocr_lines_arena_bytes
 *
 * BATCH records: uint32 little-endian width, height, bpp (1 or 4), finder, stride, flags, then
 * stride * height pixel bytes. Output per record, one line: rc found count truncated rows_read,
 * then x y w h dark per line (the refined set; found = the finder's line count).
 */
#include "vocr_lines_refine.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "refine: line %d: %s (case %u)\n", __LINE__, #x, current); return 1; } } while (0)

static unsigned current;
static uint32_t rng = 0x5eed1234u;
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
    unsigned calls, turn, last, fail_at, order_errors, passes;
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

/* A procedural region: a plain, gradient or noisy ground, frames and panels, glyph-like rows of
 * strokes (some close together), solid blobs and tall slabs at the ends. */
static void draw(unsigned char *p, unsigned w, unsigned h, size_t stride, unsigned bpp)
{
    unsigned x, y, k, i, lines = 1u + below(5), boxes = below(3), blobs = below(3);
    unsigned paper = below(256), ink = (paper + 96u + below(64)) & 255u;
    int gradient = (int)below(3) == 0, noise = (int)below(4) == 0;
    for (y = 0; y < h; y++)
        for (x = 0; x < w; x++) {
            unsigned v = gradient ? (paper + x * 64u / (w ? w : 1u) + y) & 255u : paper;
            if (noise) v = (v + below(9)) & 255u;
            for (k = 0; k < bpp; k++) p[(size_t)y * stride + (size_t)x * bpp + k] =
                (unsigned char)(bpp == 4 && k < 3 ? (v + k * below(20)) & 255u : v);
        }
    for (i = 0; i < boxes; i++) {
        unsigned x0 = below(w), y0 = below(h), bw = below(w - x0) + 1u, bh = below(h - y0) + 1u;
        unsigned v = below(256), fill = below(2);
        for (y = y0; y < y0 + bh; y++)
            for (x = x0; x < x0 + bw; x++) {
                if (!fill && y != y0 && y + 1u != y0 + bh && x != x0 && x + 1u != x0 + bw) continue;
                for (k = 0; k < bpp; k++) p[(size_t)y * stride + (size_t)x * bpp + k] = (unsigned char)v;
            }
    }
    for (i = 0; i < lines; i++) {
        unsigned lh = 8u + below(30), y0 = below(h), x0 = below(w), x1, pitch = 4u + below(20);
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
    for (i = 0; i < blobs; i++) {               /* cursor-like blobs and portrait-like slabs */
        unsigned bw = 3u + below(30), bh = 3u + below(h), x0 = below(w), y0 = below(h), v = below(256);
        for (y = y0; y < y0 + bh && y < h; y++)
            for (x = x0; x < x0 + bw && x < w; x++)
                for (k = 0; k < bpp; k++) p[(size_t)y * stride + (size_t)x * bpp + k] = (unsigned char)v;
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

/* Every refined line lies inside a finder line, they follow the finder's order, and keep its flag. */
static int inside(const VocrLineSet *found, const VocrLineSet *ref)
{
    unsigned i, j = 0;
    if (ref->count > VOCR_LINES_MAX || ref->count < found->count) return 0;
    for (i = 0; i < ref->count; i++) {
        const VocrLine *r = &ref->lines[i];
        while (j < found->count) {
            const VocrLine *f = &found->lines[j];
            if (r->x >= f->x && r->y >= f->y && r->x + r->w <= f->x + f->w && r->y + r->h <= f->y + f->h &&
                r->dark == f->dark && r->w && r->h) break;
            j++;
        }
        if (j == found->count) return 0;
    }
    return 1;
}

static const unsigned flag_sets[] = {VOCR_REFINE_DEFAULT, VOCR_REFINE_ALL, VOCR_REFINE_TIGHTEN,
                                     VOCR_REFINE_SPLIT, VOCR_REFINE_FRAMES | VOCR_REFINE_TIGHTEN,
                                     VOCR_REFINE_CURSOR, VOCR_REFINE_PORTRAIT,
                                     VOCR_REFINE_CURSOR | VOCR_REFINE_PORTRAIT | VOCR_REFINE_TIGHTEN};

static int region(unsigned w, unsigned h, unsigned bpp, unsigned pad)
{
    size_t stride = (size_t)w * bpp + pad, need = vocr_lines_arena_bytes(w, h), small = vocr_refine_arena_bytes(w, h);
    size_t big = need + (size_t)w * h + 64u;
    unsigned char *pixels = (unsigned char *)malloc(stride * h + 1u);
    unsigned char *arena = (unsigned char *)malloc(big + 3u), *arena2 = (unsigned char *)malloc(big + 3u);
    unsigned char *slots = (unsigned char *)malloc(2u * (size_t)w * bpp);
    VocrLinesImage im;
    VocrLinesRows src;
    Source s;
    VocrLineSet found, a, b, c, d;
    unsigned f, i;
    unsigned long ra_rows = 0, rb_rows = 0, rc_rows = 0;
    int rc;
    REQUIRE(pixels && arena && arena2 && slots);
    REQUIRE(small <= need);
    draw(pixels, w, h, stride, bpp);
    im.pixels = pixels; im.width = w; im.height = h; im.stride = stride; im.bpp = bpp;
    src.width = w; src.height = h; src.bpp = bpp; src.row = source_row; src.user = &s;
    rc = vocr_find_lines(&im, below(2) ? VOCR_LINES_GRAD : VOCR_LINES_TEXTBOX, arena, need, &found);
    REQUIRE(rc == VOCR_LINES_OK);
    /* flags 0: untouched, and nothing read */
    a = found;
    REQUIRE(vocr_refine_lines(&im, 0, arena, need, &a, &ra_rows) == VOCR_LINES_OK && same_set(&a, &found) && !ra_rows);
    for (f = 0; f < sizeof(flag_sets) / sizeof(flag_sets[0]); f++) {
        unsigned flags = flag_sets[f], at = below(4);
        a = b = c = d = found;
        REQUIRE(vocr_refine_lines(&im, flags, arena + at, big, &a, &ra_rows) == VOCR_LINES_OK);
        REQUIRE(inside(&found, &a));
        /* the two-pass arena (the documented minimum, misaligned) gives the same lines */
        REQUIRE(vocr_refine_lines(&im, flags, arena2 + 3u - at, small + (at ? 0u : 3u), &c, &rc_rows) == VOCR_LINES_OK);
        REQUIRE(same_set(&a, &c));
        /* the row-callback form: the same lines, rows in increasing order within each pass */
        memset(&s, 0, sizeof(s));
        s.im = &im; s.slot[0] = slots; s.slot[1] = slots + (size_t)w * bpp;
        REQUIRE(vocr_refine_lines_rows(&src, flags, arena + at, big, &b, &rb_rows) == VOCR_LINES_OK);
        REQUIRE(same_set(&a, &b) && !s.order_errors && rb_rows == ra_rows && s.calls == rb_rows);
        {   /* one pass per box with the big arena: exactly its rows */
            unsigned long sum = 0;
            for (i = 0; i < found.count; i++) sum += found.lines[i].h;
            REQUIRE(ra_rows == sum && s.passes <= found.count);
        }
        memset(&s, 0, sizeof(s));
        s.im = &im; s.slot[0] = slots; s.slot[1] = slots + (size_t)w * bpp;
        REQUIRE(vocr_refine_lines_rows(&src, flags, arena2, small, &d, &rb_rows) == VOCR_LINES_OK);
        REQUIRE(same_set(&a, &d) && !s.order_errors && rb_rows == rc_rows && s.calls == rb_rows);
        REQUIRE(s.passes <= 2u * found.count);
        /* a row that cannot be read: an error, the set unchanged */
        if (s.calls) {
            d = found;
            memset(&s, 0, sizeof(s));
            s.im = &im; s.slot[0] = slots; s.slot[1] = slots + (size_t)w * bpp; s.fail_at = 1u + below(s.calls + 1u);
            rc = vocr_refine_lines_rows(&src, flags, arena2, small, &d, NULL);
            REQUIRE((rc == VOCR_LINES_E_SOURCE && same_set(&d, &found)) || rc == VOCR_LINES_OK);
        }
    }
    /* errors leave the set unchanged */
    a = found;
    REQUIRE(vocr_refine_lines(&im, 0x40u, arena, need, &a, NULL) == VOCR_LINES_E_ARG && same_set(&a, &found));
    if (small > 1u) REQUIRE(vocr_refine_lines(&im, VOCR_REFINE_DEFAULT, arena, small - 1u, &a, NULL) == VOCR_LINES_E_ARENA &&
                            same_set(&a, &found));
    if (found.count) {
        a = found;
        a.lines[0].w = w - a.lines[0].x + 1u;
        b = a;
        REQUIRE(vocr_refine_lines(&im, VOCR_REFINE_DEFAULT, arena, need, &a, NULL) == VOCR_LINES_E_ARG && same_set(&a, &b));
    }
    free(pixels); free(arena); free(arena2); free(slots);
    return 0;
}

static int selftest(unsigned cases)
{
    unsigned w, h;
    for (w = 1; w <= 1000; w += 37)
        for (h = 1; h <= 600; h += 23) REQUIRE(vocr_refine_arena_bytes(w, h) <= vocr_lines_arena_bytes(w, h));
    for (w = 1; w <= 40; w++)                    /* thin and tall, wide and flat */
        for (h = 1; h <= 6000; h += 97) {
            REQUIRE(vocr_refine_arena_bytes(w, h) <= vocr_lines_arena_bytes(w, h));
            REQUIRE(vocr_refine_arena_bytes(h, w) <= vocr_lines_arena_bytes(h, w));
        }
    REQUIRE(vocr_refine_arena_bytes(960, 544) <= vocr_lines_arena_bytes(960, 544));
    REQUIRE(vocr_refine_arena_bytes(0, 5) == 0 && vocr_refine_arena_bytes(5, 0) == 0);
    REQUIRE(vocr_refine_lines(NULL, 1, NULL, 0, NULL, NULL) == VOCR_LINES_E_ARG);
    REQUIRE(vocr_refine_lines_rows(NULL, 1, NULL, 0, NULL, NULL) == VOCR_LINES_E_ARG);
    for (current = 0; current < cases; current++) {
        unsigned rw = 8u + below(current % 5u == 0 ? 953u : 400u), rh = 8u + below(current % 7u == 0 ? 537u : 120u);
        if (region(rw, rh, below(2) ? VOCR_LINES_RGBA : VOCR_LINES_GREY, below(3) * 4u)) return 1;
    }
    printf("REFINE_SELFTEST_OK %u regions\n", cases);
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
    unsigned w;
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
    while (off + 24u <= n) {
        unsigned w = rd32(buf + off), h = rd32(buf + off + 4), bpp = rd32(buf + off + 8);
        unsigned finder = rd32(buf + off + 12), flags = rd32(buf + off + 20);
        size_t stride = rd32(buf + off + 16), need, i;
        unsigned char *arena;
        VocrLinesImage im;
        VocrLineSet found, set;
        unsigned long rows = 0;
        int rc;
        off += 24u;
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
                    rc = vocr_refine_lines(&im, flags, arena, need, &set, &rows);
                    {
                        double dt = (double)(clock() - t0) / CLOCKS_PER_SEC;
                        if (dt < best) best = dt;
                    }
                }
                total += best;
            } else if (use_rows) {
                MemRows mr;
                VocrLinesRows src;
                mr.p = im.pixels; mr.stride = stride; mr.w = w;
                src.width = w; src.height = h; src.bpp = bpp; src.row = mem_row; src.user = &mr;
                rc = vocr_refine_lines_rows(&src, flags, arena, need, &set, &rows);
            } else {
                rc = vocr_refine_lines(&im, flags, arena, need, &set, &rows);
            }
        }
        if (!reps) {
            printf("%d %u %u %u %lu", rc, found.count, rc ? 0u : set.count, set.truncated, rows);
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
        printf("%lu %lu\n", (unsigned long)vocr_refine_arena_bytes(w, h), (unsigned long)vocr_lines_arena_bytes(w, h));
        return 0;
    }
    if (argc == 4 && !strcmp(argv[1], "--bench")) return batch(argv[2], 0, (unsigned)strtoul(argv[3], NULL, 10));
    if (argc == 3 && !strcmp(argv[1], "--rows")) return batch(argv[2], 1, 0);
    if (argc == 2) return batch(argv[1], 0, 0);
    fprintf(stderr, "usage: %s --selftest [CASES] | [--rows] BATCH | --bench BATCH REPS | --arena W H\n", argv[0]);
    return 2;
}
