/* vocr_lines_fix.c against the frozen finder and itself (issue #129):
 *   - fix = 0: vocr_find_lines_fix() returns vocr_find_lines()'s lines, flags and return codes for every finder;
 *   - any fix: the row form (vocr_find_lines_fix_rows, through a callback that poisons each row it handed out on
 *     the next call) equals the in-memory form; every line lies inside the region; a row that cannot be read is
 *     VOCR_LINES_E_SOURCE and no lines;
 *   - each fix on a region drawn for it: VOCR_FIX_COLGAP keeps a line whole across a 3-band-height gap,
 *     VOCR_FIX_VALLEY splits two touching rows, VOCR_FIX_JUNK drops an edge strip and an ornament, and none of
 *     them changes a plain two-line region;
 *   - the ornament test's word-parallel counts against the pixel-by-pixel definition on random masks;
 *   - argument checks.
 * Procedural regions only: no screenshots, fonts or models. The file includes vocr_lines_fix.c to reach its static
 * helpers, so the test is built from vocr_lines.c and this file alone. */
#include "../vocr_lines.h"
#include "../vocr_lines_fix.c"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr, "fix: line %d: %s (case %u)\n", __LINE__, #x, current); return 1; } } while (0)

static unsigned current;
static uint32_t rng = 0x6c8e9cf5u;
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
    unsigned char *slot[2];
    unsigned calls, turn, fail_at;
} Source;

static const unsigned char *source_row(void *user, unsigned y)
{
    Source *s = (Source *)user;
    size_t n = (size_t)s->im->width * s->im->bpp;
    unsigned char *out;
    if (y >= s->im->height) return NULL;
    if (s->fail_at && ++s->calls >= s->fail_at) return NULL;
    if (!s->fail_at) s->calls++;
    memset(s->slot[s->turn ^ 1u], 0xA5, n);
    out = s->slot[s->turn];
    memcpy(out, s->im->pixels + (size_t)y * s->im->stride, n);
    s->turn ^= 1u;
    return out;
}

/* tests/test_vocr_lines_rows.c's regions: paper, boxes and panels, rows of glyph-like strokes */
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

static int inside(const VocrLineSet *a, unsigned w, unsigned h)
{
    unsigned i;
    if (a->count > VOCR_LINES_MAX) return 0;
    for (i = 0; i < a->count; i++)
        if (!a->lines[i].w || !a->lines[i].h || a->lines[i].x + a->lines[i].w > w || a->lines[i].y + a->lines[i].h > h)
            return 0;
    return 1;
}

static const int finders[] = {VOCR_LINES_GRAD, VOCR_LINES_LEGACY, VOCR_LINES_TEXTBOX, VOCR_LINES_TEXTBOX_RECOVERY};

static int region(unsigned w, unsigned h, unsigned bpp, unsigned pad)
{
    size_t stride = (size_t)w * bpp + pad, need = vocr_lines_arena_bytes(w, h);
    unsigned char *pixels = (unsigned char *)malloc(stride * h + 1u);
    unsigned char *arena = (unsigned char *)malloc(need + 3u);
    unsigned char *slots = (unsigned char *)malloc(2u * (size_t)w * bpp);
    VocrLinesImage im;
    VocrLinesRows src;
    Source s;
    unsigned f;
    REQUIRE(pixels && arena && slots);
    draw(pixels, w, h, stride, bpp);
    im.pixels = pixels; im.width = w; im.height = h; im.stride = stride; im.bpp = bpp;
    src.width = w; src.height = h; src.bpp = bpp; src.row = source_row; src.user = &s;
    for (f = 0; f < sizeof(finders) / sizeof(finders[0]); f++) {
        VocrLineSet a, b, c;
        unsigned fix = below(VOCR_FIX_ALL + 1u), at = below(3) + 1u;
        int ra, rb, rc;
        /* fix = 0 is the frozen finder */
        ra = vocr_find_lines(&im, finders[f], arena + at, need, &a);
        rb = vocr_find_lines_fix(&im, finders[f], 0u, arena + 3u - at, need, &b);
        REQUIRE(ra == rb);
        REQUIRE(same_set(&a, &b));
        /* any fix: both forms agree */
        memset(&s, 0, sizeof(s));
        s.im = &im; s.slot[0] = slots; s.slot[1] = slots + (size_t)w * bpp;
        rb = vocr_find_lines_fix(&im, finders[f], fix, arena + at, need, &b);
        rc = vocr_find_lines_fix_rows(&src, finders[f], fix, arena + 3u - at, need, &c);
        REQUIRE(rb == rc);
        REQUIRE(same_set(&b, &c));
        REQUIRE(rb != VOCR_LINES_OK || inside(&b, w, h));
        /* the legacy finder takes no fix */
        if (finders[f] == VOCR_LINES_LEGACY) REQUIRE(ra == rb && same_set(&a, &b));
        if (s.calls) {
            unsigned fail = 1u + below(s.calls);
            memset(&s, 0, sizeof(s));
            s.im = &im; s.slot[0] = slots; s.slot[1] = slots + (size_t)w * bpp; s.fail_at = fail;
            rc = vocr_find_lines_fix_rows(&src, finders[f], fix, arena, need, &c);
            REQUIRE(rc < 0 && c.count == 0);
            if (finders[f] != VOCR_LINES_LEGACY) REQUIRE(rc == VOCR_LINES_E_SOURCE);
        }
    }
    free(pixels);
    free(arena);
    free(slots);
    return 0;
}

/* ---- regions drawn for one fix each: bright glyph blocks on a dark paper ---- */

#define PAPER 30u
#define INK 220u

static void fill(unsigned char *p, unsigned w, unsigned h, unsigned v)
{
    memset(p, (int)v, (size_t)w * h);
}

/* n glyphs of size g at (x0, y0), pitch g + g / 5: vertical strokes every 4 px, a stroke row every 6 */
static void glyphs(unsigned char *p, unsigned w, unsigned x0, unsigned y0, unsigned n, unsigned g)
{
    unsigned i, x, y, pitch = g + g / 5u;
    for (i = 0; i < n; i++)
        for (y = 0; y < g; y++)
            for (x = 0; x < g; x++)
                if (x % 4u < 2u || y % 6u == 0u || y + 1u == g) p[(size_t)(y0 + y) * w + x0 + i * pitch + x] = INK;
}

static int find(const unsigned char *p, unsigned w, unsigned h, unsigned fix, VocrLineSet *out)
{
    VocrLinesImage im;
    size_t need = vocr_lines_arena_bytes(w, h);
    unsigned char *arena = (unsigned char *)malloc(need);
    int rc;
    im.pixels = p; im.width = w; im.height = h; im.stride = w; im.bpp = VOCR_LINES_GREY;
    if (!arena) return -100;
    rc = vocr_find_lines_fix(&im, VOCR_LINES_TEXTBOX_RECOVERY, fix, arena, need, out);
    free(arena);
    return rc;
}

static int targeted(void)
{
    enum { W = 640, H = 120 };
    static unsigned char p[W * H];
    VocrLineSet a, b;
    unsigned x, y, i;
    current = 9001;
    /* COLGAP: 6 glyphs, a gap of 3 glyph widths (a full-width space after ！ is 1-2), then 3 glyphs */
    fill(p, W, H, PAPER);
    glyphs(p, W, 20, 40, 6, 20);
    glyphs(p, W, 20 + 6 * 24 + 60, 40, 3, 20);
    REQUIRE(find(p, W, H, 0u, &a) == VOCR_LINES_OK && a.count == 1);
    REQUIRE(a.lines[0].x + a.lines[0].w < 20 + 6 * 24 + 60);       /* the frozen finder keeps the heavier part */
    REQUIRE(find(p, W, H, VOCR_FIX_COLGAP, &b) == VOCR_LINES_OK && b.count == 1);
    REQUIRE(b.lines[0].x <= 20 && b.lines[0].x + b.lines[0].w >= 20 + 6 * 24 + 60 + 2 * 24 + 20);
    /* VALLEY: two rows of glyphs three rows apart, joined by short ink runs in the gap (outlines touching) */
    current = 9002;
    fill(p, W, H, PAPER);
    glyphs(p, W, 20, 20, 20, 20);
    glyphs(p, W, 20, 43, 20, 20);
    for (i = 0; i < 4; i++)                     /* striped, so the gap's middle row has kept runs */
        for (x = 0; x < 20; x += 3)
            for (y = 40; y < 43; y++) p[(size_t)y * W + 60 + i * 120 + x] = INK;
    REQUIRE(find(p, W, H, 0u, &a) == VOCR_LINES_OK && a.count == 1);
    REQUIRE(find(p, W, H, VOCR_FIX_VALLEY, &b) == VOCR_LINES_OK && b.count == 2);
    REQUIRE(b.lines[0].y + b.lines[0].h <= 43 + 3 && b.lines[1].y >= 40 - 3);
    /* JUNK, edge: an ornament strip on the region's top edge above a text row */
    current = 9003;
    fill(p, W, H, PAPER);
    for (y = 0; y < 12; y++)
        for (x = 0; x < W; x++)
            if (x % 8u < 3u || y % 5u == 0u) p[(size_t)y * W + x] = INK;
    glyphs(p, W, 20, 40, 20, 20);
    REQUIRE(find(p, W, H, 0u, &a) == VOCR_LINES_OK && a.count == 2 && a.lines[0].y == 0);
    REQUIRE(find(p, W, H, VOCR_FIX_JUNK, &b) == VOCR_LINES_OK && b.count == 1 && b.lines[0].y > 20);
    /* JUNK, ornament: the same strip away from the edges, as tall as the text and as wide as the region */
    current = 9004;
    fill(p, W, H, PAPER);
    for (y = 8; y < 28; y++)
        for (x = 0; x < W; x++)
            if (x % 8u < 3u || y % 5u == 0u) p[(size_t)y * W + x] = INK;
    glyphs(p, W, 20, 60, 20, 20);
    REQUIRE(find(p, W, H, 0u, &a) == VOCR_LINES_OK && a.count == 2);
    REQUIRE(find(p, W, H, VOCR_FIX_JUNK, &b) == VOCR_LINES_OK && b.count == 1 && b.lines[0].y > 40);
    /* none of the fixes changes a plain two-line region */
    current = 9005;
    fill(p, W, H, PAPER);
    glyphs(p, W, 20, 20, 20, 20);
    glyphs(p, W, 20, 60, 12, 20);
    REQUIRE(find(p, W, H, 0u, &a) == VOCR_LINES_OK && a.count == 2);
    REQUIRE(find(p, W, H, VOCR_FIX_ALL, &b) == VOCR_LINES_OK && same_set(&a, &b));
    /* a region keeps a line: two edge strips alone stay */
    current = 9006;
    fill(p, W, H, PAPER);
    glyphs(p, W, 20, 0, 20, 20);
    glyphs(p, W, 20, H - 20, 20, 20);
    REQUIRE(find(p, W, H, 0u, &a) == VOCR_LINES_OK);
    REQUIRE(find(p, W, H, VOCR_FIX_JUNK, &b) == VOCR_LINES_OK && b.count >= 1 && b.count <= a.count);
    return 0;
}

/* junk_periodic's definition, pixel by pixel */
static int periodic_reference(const uint32_t *mask, unsigned words, const VocrLine *ln)
{
    unsigned p, y, x, pmax = 3u * ln->h;
    if (pmax > ln->w / 4u) pmax = ln->w / 4u;
    for (p = 4u; p <= pmax; p++) {
        unsigned long xo = 0, ou = 0;
        for (y = ln->y; y < ln->y + ln->h; y++)
            for (x = ln->x; x + p < ln->x + ln->w; x++) {
                int s0 = bit(mask + (size_t)y * words, x), s1 = bit(mask + (size_t)y * words, x + p);
                xo += (unsigned long)(s0 ^ s1);
                ou += (unsigned long)(s0 | s1);
            }
        if (ou && 100u * xo < FIX_JUNK_PERIOD * ou) return 1;
    }
    return 0;
}

static int period_counts(void)
{
    enum { MW = 200, MH = 30 };
    static uint32_t mask[((MW + 31) / 32) * MH];
    unsigned t, words = (MW + 31u) / 32u, agree = 0, periodic = 0;
    for (t = 0; t < 3000u; t++) {
        unsigned x, y, period = 4u + below(40), duty = 1u + below(period), noise = below(4) ? below(30) : 0u;
        VocrLine ln;
        memset(mask, 0, sizeof(mask));
        ln.w = 8u + below(MW - 8u);
        ln.x = below(MW - ln.w + 1u);
        ln.h = 1u + below(MH);
        ln.y = below(MH - ln.h + 1u);
        for (y = 0; y < MH; y++)
            for (x = 0; x < MW; x++) {
                int on = (x % period) < duty;
                if (below(100) < noise) on = !on;
                if (below(3) == 0 && t % 2u) on = (int)below(2);   /* half the masks: random texture */
                if (on) mask[(size_t)y * words + (x >> 5)] |= (uint32_t)1 << (x & 31u);
            }
        current = 20000u + t;
        REQUIRE(junk_periodic(mask, words, &ln) == periodic_reference(mask, words, &ln));
        agree++;
        periodic += (unsigned)junk_periodic(mask, words, &ln);
    }
    REQUIRE(periodic > 100u && periodic < agree - 100u);  /* both outcomes exercised */
    return 0;
}

int main(int argc, char **argv)
{
    static const unsigned edge[][2] = {{1, 1}, {1, 9}, {9, 1}, {8, 8}, {16, 16}, {31, 40}, {33, 7},
                                       {300, 80}, {480, 272}, {961, 30}, {960, 545}, {960, 544}};
    unsigned cases = argc > 1 ? (unsigned)strtoul(argv[1], NULL, 10) : 300u, i;
    VocrLinesRows bad;
    VocrLinesImage im;
    VocrLineSet out;
    unsigned char arena[64], px[16] = {0};
    for (i = 0; i < sizeof(edge) / sizeof(edge[0]); i++) {
        current = i;
        if (region(edge[i][0], edge[i][1], VOCR_LINES_RGBA, 4u * (i % 3u))) return 1;
        if (region(edge[i][0], edge[i][1], VOCR_LINES_GREY, i % 5u)) return 1;
    }
    for (i = 0; i < cases; i++) {
        unsigned wide = below(4), w, tall, h, bpp, pad;
        w = 1u + below(wide ? 400u : 960u);
        tall = below(4);
        h = 1u + below(tall ? 160u : 544u);
        bpp = below(2) ? VOCR_LINES_RGBA : VOCR_LINES_GREY;
        pad = below(3);
        pad = pad ? 0u : below(64);
        current = 1000u + i;
        if (region(w, h, bpp, pad)) return 1;
    }
    if (targeted()) return 1;
    if (period_counts()) return 1;
    /* argument checks */
    current = 0;
    memset(&bad, 0, sizeof(bad));
    bad.width = bad.height = 4; bad.bpp = VOCR_LINES_GREY; bad.row = source_row;
    REQUIRE(vocr_find_lines_fix_rows(&bad, VOCR_LINES_GRAD, VOCR_FIX_ALL + 1u, arena, sizeof(arena), &out) ==
            VOCR_LINES_E_ARG);
    REQUIRE(vocr_find_lines_fix_rows(&bad, 9, 0u, arena, sizeof(arena), &out) == VOCR_LINES_E_ARG);
    REQUIRE(vocr_find_lines_fix_rows(NULL, VOCR_LINES_GRAD, 0u, arena, sizeof(arena), &out) == VOCR_LINES_E_ARG);
    REQUIRE(vocr_find_lines_fix_rows(&bad, VOCR_LINES_GRAD, 0u, arena, sizeof(arena), NULL) == VOCR_LINES_E_ARG);
    im.pixels = px; im.width = 4; im.height = 4; im.stride = 4; im.bpp = VOCR_LINES_GREY;
    REQUIRE(vocr_find_lines_fix(&im, VOCR_LINES_GRAD, 0u, arena, 1, &out) == VOCR_LINES_E_ARENA);
    REQUIRE(vocr_find_lines_fix(&im, VOCR_LINES_GRAD, 8u, arena, sizeof(arena), &out) == VOCR_LINES_E_ARG);
    im.stride = 3;
    REQUIRE(vocr_find_lines_fix(&im, VOCR_LINES_GRAD, 0u, arena, sizeof(arena), &out) == VOCR_LINES_E_ARG);
    REQUIRE(vocr_find_lines_fix(NULL, VOCR_LINES_GRAD, 0u, arena, sizeof(arena), &out) == VOCR_LINES_E_ARG);
    printf("fix: %u random + %u edge regions, 4 finders: fix 0 = the frozen finder, both forms agree; "
           "6 targeted cases; ornament counts = their definition on 3,000 masks\n", cases, (unsigned)(2u * (sizeof(edge) / sizeof(edge[0]))));
    return 0;
}
