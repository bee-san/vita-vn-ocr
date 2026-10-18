/* Stroke split of the line finder's boxes: see vocr_lines_split.h and docs/translucent-box.md.
 *
 * Per line box (x, y, w, h) of the set, in set order, from the box's own pixels only (indices clamped at the
 * box's edges):
 *  1. Grey g = (77 R + 150 G + 29 B) >> 8 (the finder's and the recognizer's grey).
 *  2. Stroke maps, along the box's rows. B = g minus the brighter of the darkest of the S_R pixels on the left
 *     and the darkest of the S_R pixels on the right, 0 when negative (a bright ridge: darker pixels close by on
 *     both sides); D the same the other way round (a dark ridge). C = max - min of the pixel and its two
 *     neighbours.
 *  3. Levels from a first pass over the box: tb, td = max(S_TFLOOR, 2/5 of the 99th percentile of B, D),
 *     tc = max(S_CFLOOR, 2/5 of C's) (GRAD's threshold rule). A second pass counts per row
 *     S(y) = min(#B >= tb, #D >= td) and On(y) = #C >= tc.
 *  4. Segments: the box's rows, split at the deepest valley row y of a segment (S[y] against the smaller of the
 *     peaks of S above and below it within the segment; the first on ties) while S[y] * S_VDEN <=
 *     S_VNUM * that peak, in passes over the segments in order, at most S_SEGS segments. Each segment's band:
 *     its rows from the first to the last with S * S_BDEN >= S_BNUM * the segment's peak.
 *  5. Text bands: at least S_BAND_H % as tall as the tallest band, S_BAND_M % of the heaviest band's mass and
 *     S_BAND_MINH rows tall. Each one after the first is a line of its own when the smallest On between it and
 *     the previous kept band is at most S_CONT % of the upper median of On over both bands' rows; otherwise the
 *     two are one band (strokes run through the gap: a band boundary inside large glyphs).
 *  6. Two or more text bands, and room in the set: one line per band, its rows the band plus S_MARGIN % of its
 *     height above and below, never past the midpoint to the next band; the box's columns and dark flag.
 *     Otherwise, and for a box under S_MIN_H rows or over S_MAXW columns, the box as it is.
 *
 * Memory: one grey row of the box with its 3-pixel minima and maxima, and three per-row arrays (S, On and a
 * scratch row of 16-bit values) from the caller's arena; three 256-entry histograms on the stack. Each box is read
 * twice, top to bottom. No allocation, no recursion, integer arithmetic only.
 */
#include "vocr_lines_split.h"

#include <stdint.h>
#include <string.h>

#define S_MIN_H 16u        /* boxes under this many rows are kept */
#define S_MAXW 65535u      /* row counts are 16-bit: boxes wider than this are kept */
#define S_R 3u             /* reach of the ridge test on each side: the 3-pixel window centred S_R - 1 away */
#define S_TFLOOR 24u       /* the ridge levels' floor */
#define S_CFLOOR 40u       /* the contrast level's floor (GRAD's T_MIN) */
#define S_VNUM 1u          /* a valley: S[y] * S_VDEN <= S_VNUM * the smaller neighbouring peak */
#define S_VDEN 3u
#define S_BNUM 1u          /* a band's rows: S * S_BDEN >= S_BNUM * its segment's peak */
#define S_BDEN 4u
#define S_BAND_H 70u       /* % of the tallest band's height a text band needs */
#define S_BAND_M 10u       /* % of the heaviest band's mass */
#define S_BAND_MINH 6u     /* rows */
#define S_CONT 60u         /* % : the gap's smallest On against the bands' upper median */
#define S_MARGIN 16u       /* % of the band height above and below */
#define S_SEGS 16u

typedef struct {
    const VocrLinesImage *im;  /* in-memory image, or */
    const VocrLinesRows *rows; /* a row source */
} Src;

typedef struct {
    unsigned a, b;             /* rows [a, b) of the box */
    unsigned long m;           /* mass: the sum of S over them */
} Band;

static size_t up4(size_t n)
{
    return (n + 3u) & ~(size_t)3u;
}

static unsigned char *align4(void *p)
{
    uintptr_t v = (uintptr_t)p;
    return (unsigned char *)p + ((4u - (v & 3u)) & 3u);
}

size_t vocr_split_arena_bytes(unsigned max_width, unsigned max_height)
{
    if (!max_width || !max_height) return 0;
    return 3u + up4((size_t)3u * max_width) + 3u * up4((size_t)max_height * 2u);
}

static unsigned src_width(const Src *s)
{
    return s->im ? s->im->width : s->rows->width;
}

static unsigned src_height(const Src *s)
{
    return s->im ? s->im->height : s->rows->height;
}

static unsigned src_bpp(const Src *s)
{
    return s->im ? s->im->bpp : s->rows->bpp;
}

static const unsigned char *src_row(const Src *s, unsigned y)
{
    if (s->im) return s->im->pixels + (size_t)y * s->im->stride;
    return s->rows->row(s->rows->user, y);
}

static unsigned grey_px(const unsigned char *row, unsigned x, unsigned bpp)
{
    const unsigned char *p = row + (size_t)x * bpp;
    if (bpp == VOCR_LINES_GREY) return p[0];
    return (77u * p[0] + 150u * p[1] + 29u * p[2]) >> 8;
}

/* max(floor, 2/5 of the 99th percentile): the smallest v whose cumulative count times 100 reaches n times 99. */
static unsigned level(const uint32_t *hist, unsigned long long n, unsigned floor)
{
    unsigned long long cum = 0;
    unsigned v, p99 = 255u;
    for (v = 0; v < 256u; v++) {
        cum += hist[v];
        if (cum * 100u >= n * 99u) {
            p99 = v;
            break;
        }
    }
    v = 2u * p99 / 5u;
    return v < floor ? floor : v;
}

/* One box row of the maps from its grey g (w pixels); mn3 / mx3 receive the minimum and maximum of each pixel's
 * 3-pixel window (edges replicated). Pass 0 adds B, D and C to the histograms; pass 1 counts the pixels at or
 * above the levels lv into cnt. */
static void row_maps(const unsigned char *g, unsigned w, unsigned char *mn3, unsigned char *mx3, unsigned pass,
                     uint32_t *hb, uint32_t *hd, uint32_t *hc, const unsigned *lv, unsigned *cnt)
{
    unsigned x;
    for (x = 0; x < w; x++) {
        unsigned a = g[x ? x - 1u : 0], b = g[x], c = g[x + 1u < w ? x + 1u : w - 1u];
        unsigned lo = a < b ? a : b, hi = a > b ? a : b;
        mn3[x] = (unsigned char)(c < lo ? c : lo);
        mx3[x] = (unsigned char)(c > hi ? c : hi);
    }
    cnt[0] = cnt[1] = cnt[2] = 0;
    for (x = 0; x < w; x++) {
        /* the S_R pixels on the left are the window centred on x - 2 (only g[0] near the edge), those on the
         * right the window centred on x + 2 (only g[w - 1]) */
        unsigned v = g[x], mnl, mnr, mxl, mxr, hi, lo, b = 0, d = 0, c = (unsigned)mx3[x] - mn3[x];
        if (x >= S_R - 1u) {
            mnl = mn3[x - (S_R - 1u)];
            mxl = mx3[x - (S_R - 1u)];
        } else {
            mnl = mxl = g[0];
        }
        if (x + (S_R - 1u) < w) {
            mnr = mn3[x + (S_R - 1u)];
            mxr = mx3[x + (S_R - 1u)];
        } else {
            mnr = mxr = g[w - 1u];
        }
        hi = mnl > mnr ? mnl : mnr;
        lo = mxl < mxr ? mxl : mxr;
        if (v > hi) b = v - hi;
        if (lo > v) d = lo - v;
        if (!pass) {
            hb[b]++;
            hd[d]++;
            hc[c]++;
        } else {
            cnt[0] += b >= lv[0];
            cnt[1] += d >= lv[1];
            cnt[2] += c >= lv[2];
        }
    }
}

/* S and On of a box: two passes over its rows. buf: three rows of w bytes. */
static int box_profiles(const Src *src, const VocrLine *ln, unsigned char *buf, uint16_t *S, uint16_t *On,
                        unsigned long *rows_read)
{
    uint32_t hb[256], hd[256], hc[256];
    unsigned w = ln->w, h = ln->h, bpp = src_bpp(src), pass, y, x, lv[3] = {0, 0, 0}, cnt[3];
    unsigned long long n = (unsigned long long)w * h;
    unsigned char *g = buf, *mn3 = buf + w, *mx3 = buf + 2u * (size_t)w;
    memset(hb, 0, sizeof(hb));
    memset(hd, 0, sizeof(hd));
    memset(hc, 0, sizeof(hc));
    for (pass = 0; pass < 2u; pass++) {
        for (y = 0; y < h; y++) {
            const unsigned char *row = src_row(src, ln->y + y);
            if (!row) return VOCR_LINES_E_SOURCE;
            (*rows_read)++;
            for (x = 0; x < w; x++) g[x] = (unsigned char)grey_px(row, ln->x + x, bpp);
            row_maps(g, w, mn3, mx3, pass, hb, hd, hc, lv, cnt);
            if (pass) {
                S[y] = (uint16_t)(cnt[0] < cnt[1] ? cnt[0] : cnt[1]);
                On[y] = (uint16_t)cnt[2];
            }
        }
        if (!pass) {
            lv[0] = level(hb, n, S_TFLOOR);
            lv[1] = level(hd, n, S_TFLOOR);
            lv[2] = level(hc, n, S_CFLOOR);
        }
    }
    return VOCR_LINES_OK;
}

/* The segments of rows [0, h) (step 4) into sa / sb; returns their count. suf: h scratch entries. */
static unsigned segments(const uint16_t *S, unsigned h, uint16_t *suf, unsigned *sa, unsigned *sb)
{
    unsigned ta[S_SEGS], tb[S_SEGS], ns = 1, i, nt, count, split;
    sa[0] = 0;
    sb[0] = h;
    do {
        split = 0;
        nt = 0;
        count = ns;
        for (i = 0; i < ns; i++) {
            unsigned a = sa[i], b = sb[i], y, by = 0, have = 0, left;
            unsigned long best = 0;
            if (b - a >= 3u) {
                suf[b - 1u] = S[b - 1u];               /* suf[y] = max S[y .. b) */
                for (y = b - 1u; y > a; y--) suf[y - 1u] = S[y - 1u] > suf[y] ? S[y - 1u] : suf[y];
                left = S[a];                           /* max S[a .. y) */
                for (y = a + 1u; y + 1u < b; y++) {
                    unsigned m = left < suf[y + 1u] ? left : suf[y + 1u];
                    if (m && (unsigned long)S[y] * S_VDEN <= (unsigned long)S_VNUM * m) {
                        unsigned long q = ((unsigned long)S[y] * 4096u) / m;
                        if (!have || q < best) {
                            best = q;
                            by = y;
                            have = 1;
                        }
                    }
                    if (S[y] > left) left = S[y];
                }
            }
            if (have && count < S_SEGS) {
                ta[nt] = a;
                tb[nt++] = by;
                ta[nt] = by + 1u;
                tb[nt++] = b;
                count++;
                split = 1;
            } else {
                ta[nt] = a;
                tb[nt++] = b;
            }
        }
        for (i = 0; i < nt; i++) {
            sa[i] = ta[i];
            sb[i] = tb[i];
        }
        ns = nt;
    } while (split);
    return ns;
}

/* The upper median of On over rows [a0, b0) and [a1, b1): their values sorted in tmp, the middle one. */
static unsigned upper_median(const uint16_t *On, unsigned a0, unsigned b0, unsigned a1, unsigned b1, uint16_t *tmp)
{
    unsigned n = 0, i, j, y;
    for (y = a0; y < b0; y++) tmp[n++] = On[y];
    for (y = a1; y < b1; y++) tmp[n++] = On[y];
    for (i = 1; i < n; i++) {                          /* insertion sort */
        uint16_t v = tmp[i];
        for (j = i; j && tmp[j - 1u] > v; j--) tmp[j] = tmp[j - 1u];
        tmp[j] = v;
    }
    return tmp[n / 2u];
}

/* The text bands of a box (steps 4-5) into bd; returns their count. */
static unsigned text_bands(const uint16_t *S, const uint16_t *On, unsigned h, uint16_t *tmp, Band *bd)
{
    unsigned sa[S_SEGS], sb[S_SEGS], ns, nb = 0, nsel = 0, nk, i, y, hmax = 0;
    unsigned long mmax = 0;
    Band all[S_SEGS], sel[S_SEGS];
    ns = segments(S, h, tmp, sa, sb);
    for (i = 0; i < ns; i++) {
        unsigned a = sa[i], b = sb[i], pk = 0, first = 0, last = 0, have = 0;
        unsigned long m = 0;
        for (y = a; y < b; y++)
            if (S[y] > pk) pk = S[y];
        if (!pk) continue;
        for (y = a; y < b; y++)
            if ((unsigned long)S[y] * S_BDEN >= (unsigned long)S_BNUM * pk) {
                if (!have) first = y;
                last = y;
                have = 1;
            }
        for (y = first; y <= last; y++) m += S[y];
        all[nb].a = first;
        all[nb].b = last + 1u;
        all[nb].m = m;
        nb++;
    }
    if (nb < 2u) return 0;
    for (i = 0; i < nb; i++) {
        if (all[i].b - all[i].a > hmax) hmax = all[i].b - all[i].a;
        if (all[i].m > mmax) mmax = all[i].m;
    }
    for (i = 0; i < nb; i++) {
        unsigned bh = all[i].b - all[i].a;
        if (100ull * bh >= (unsigned long long)S_BAND_H * hmax &&
            100ull * all[i].m >= (unsigned long long)S_BAND_M * mmax &&
            bh >= S_BAND_MINH)
            sel[nsel++] = all[i];
    }
    if (nsel < 2u) return 0;
    bd[0] = sel[0];
    nk = 1;
    for (i = 1; i < nsel; i++) {
        Band *p = &bd[nk - 1u];
        unsigned med = upper_median(On, p->a, p->b, sel[i].a, sel[i].b, tmp), gmin = On[p->b];
        for (y = p->b; y < sel[i].a; y++)
            if (On[y] < gmin) gmin = On[y];
        if (100u * gmin <= S_CONT * med) {
            bd[nk++] = sel[i];
        } else {
            p->b = sel[i].b;
            p->m += sel[i].m;
        }
    }
    return nk;
}

/* Splits one box into out (appending); room is how many lines it may become. Returns 0, or
 * VOCR_LINES_E_SOURCE. */
static int split_box(const Src *src, const VocrLine *ln, unsigned char *base, unsigned H, unsigned room,
                     VocrLineSet *out, unsigned long *rows_read)
{
    unsigned w = ln->w, h = ln->h, W = src_width(src), nk, i;
    uint16_t *S = (uint16_t *)(void *)(base + up4((size_t)3u * W));
    uint16_t *On = S + up4((size_t)H * 2u) / 2u;
    uint16_t *tmp = On + up4((size_t)H * 2u) / 2u;
    Band bd[S_SEGS];
    int rc;
    if (h < S_MIN_H || w > S_MAXW) {
        out->lines[out->count++] = *ln;
        return VOCR_LINES_OK;
    }
    rc = box_profiles(src, ln, base, S, On, rows_read);
    if (rc) return rc;
    nk = text_bands(S, On, h, tmp, bd);
    if (nk < 2u || nk > room) {
        out->lines[out->count++] = *ln;
        return VOCR_LINES_OK;
    }
    for (i = 0; i < nk; i++) {
        unsigned a = bd[i].a, b = bd[i].b, mg = (S_MARGIN * (b - a) + 50u) / 100u;
        unsigned lo = i ? (bd[i - 1u].b + a + 1u) / 2u : 0, hi = i + 1u < nk ? (b + bd[i + 1u].a) / 2u : h;
        unsigned t0 = a > mg ? a - mg : 0, t1 = b + mg;
        VocrLine *o = &out->lines[out->count++];
        if (t0 < lo) t0 = lo;
        if (t1 > hi) t1 = hi;
        o->x = ln->x;
        o->y = ln->y + t0;
        o->w = w;
        o->h = t1 - t0;
        o->dark = ln->dark;
    }
    return VOCR_LINES_OK;
}

static int split(const Src *src, unsigned flags, void *arena, size_t arena_bytes, VocrLineSet *set,
                 unsigned long *rows_read)
{
    unsigned i, W, H, bpp;
    unsigned long dummy = 0;
    unsigned char *base;
    VocrLineSet out;
    if (!rows_read) rows_read = &dummy;
    *rows_read = 0;
    if (!set || (flags & ~VOCR_SPLIT_ALL)) return VOCR_LINES_E_ARG;
    if (!flags) return VOCR_LINES_OK;           /* bit-identical when off */
    W = src_width(src);
    H = src_height(src);
    bpp = src_bpp(src);
    if (!W || !H || (bpp != VOCR_LINES_GREY && bpp != VOCR_LINES_RGBA) || set->count > VOCR_LINES_MAX)
        return VOCR_LINES_E_ARG;
    for (i = 0; i < set->count; i++) {
        const VocrLine *ln = &set->lines[i];
        if (!ln->w || !ln->h || ln->x >= W || ln->y >= H || ln->w > W - ln->x || ln->h > H - ln->y)
            return VOCR_LINES_E_ARG;
    }
    if (!arena || arena_bytes < vocr_split_arena_bytes(W, H)) return VOCR_LINES_E_ARENA;
    base = align4(arena);
    out.count = 0;
    out.truncated = set->truncated;
    for (i = 0; i < set->count; i++) {
        int rc = split_box(src, &set->lines[i], base, H, VOCR_LINES_MAX - out.count - (set->count - i - 1u), &out,
                           rows_read);
        if (rc) return rc;                      /* the set is untouched until here */
    }
    *set = out;
    return VOCR_LINES_OK;
}

int vocr_split_lines(const VocrLinesImage *im, unsigned flags, void *arena, size_t arena_bytes, VocrLineSet *set,
                     unsigned long *rows_read)
{
    Src s;
    if (!im || !im->pixels || (im->bpp != VOCR_LINES_GREY && im->bpp != VOCR_LINES_RGBA) ||
        im->stride < (size_t)im->width * im->bpp)
        return VOCR_LINES_E_ARG;
    s.im = im;
    s.rows = NULL;
    return split(&s, flags, arena, arena_bytes, set, rows_read);
}

int vocr_split_lines_rows(const VocrLinesRows *src, unsigned flags, void *arena, size_t arena_bytes, VocrLineSet *set,
                          unsigned long *rows_read)
{
    Src s;
    if (!src || !src->row) return VOCR_LINES_E_ARG;
    s.im = NULL;
    s.rows = src;
    return split(&s, flags, arena, arena_bytes, set, rows_read);
}
