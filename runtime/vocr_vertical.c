/* vocr_vertical.c - the column finder and the column rotation of vocr_vertical.h (issue #99, docs/vertical.md).
 *
 * The finder is the GRAD finder of vocr_lines.c on the region's transpose (synth/vertical_twin.py find_columns is the
 * bit-exact twin; the step numbers below are its docstring's):
 *  1. grey and its 3x3 contrast (max - min, edges replicated), streamed from rings of three rows; the threshold
 *     T = max(40, 2/5 of the 99th percentile) from a first pass, the on-mask (c >= T) from a second. The mask is
 *     stored column-major (one bit string per region column), so every step after it reads a "row" of the
 *     transpose as one contiguous bit string
 *  2. per column, on-pixels at most GAP rows apart join into runs; runs of at least LMIN rows are kept;
 *     prof[x] = their total length
 *  3. bands of active columns (prof >= max(LMIN, max(prof) / ROW_DIV), gaps of up to MERGE columns bridged,
 *     at least MIN_H wide; a bridged gap holding an empty column between MIN_H-wide runs ends the band);
 *     VOCR_COLUMNS_VALLEYS then splits each band at its columns of little ink
 *  4. per band: the row cluster with the most kept pixels (split at gaps over GAPC x the band width),
 *     lengthened over rows with ink at most max(4, width) apart and widened by up to half the band width (never
 *     past half the gap to the neighbour band) while columns hold 2+ on-pixels
 *  5. columns under FURI_PCT % of the median width dropped; 6. boxes wider than 5/4 of their height dropped
 *  7. a PAD px pad and vocr_line_dark's flag (VOCR_COLUMNS_NO_DARK: 0, the caller computes it); 8. right to
 *     left (a column's right edge, larger first; then top)
 * Memory: the column-major on-mask, per-column and per-row counters, three row rings and the candidates, all
 * from the caller's arena (vocr_columns_arena_bytes). Integer arithmetic only.
 */
#include "vocr_vertical.h"

#include <stdint.h>
#include <string.h>

/* GRAD's constants (vocr_lines.c, synth/common.py): keep in step */
#define T_MIN 40u
#define GAP 10u
#define LMIN 16u
#define ROW_DIV 16u
#define MIN_H 8u
#define MERGE 2u
#define GAPC 2u
#define PAD 3u
#define FURI_PCT 55u
/* the columns' own (synth/vertical_twin.py) */
#define ASPECT_NUM 5u
#define ASPECT_DEN 4u
#define VALLEY_DIV 4u
#define DENSE_DIV 8u

typedef struct {
    unsigned a, b;      /* band columns [a, b) */
} Band;

typedef struct {
    unsigned y1, x1, y2, x2;   /* rows [y1, y2), columns [x1, x2) of the region */
} Cand;

static unsigned band_cap(unsigned w)
{
    return (w + 1u) / (MIN_H + 1u) + 1u;
}

static size_t up4(size_t n)
{
    return (n + 3u) & ~(size_t)3u;
}

static size_t col_bytes(unsigned w, unsigned h)
{
    size_t words = ((size_t)h + 31u) / 32u;
    return up4(words * w * 4u)               /* on-mask, column-major */
         + up4((size_t)w * 4u)               /* prof (then the columns' on-pixel counts) */
         + up4((size_t)w) * 7u               /* row max / min rings, contrast row */
         + up4((size_t)h * 4u) * 2u          /* kc, oc */
         + up4(sizeof(Band) * band_cap(w)) * 2u
         + up4(sizeof(Cand) * band_cap(w));
}

size_t vocr_columns_arena_bytes(unsigned max_width, unsigned max_height)
{
    if (!max_width || !max_height) return 0;
    return 3u + col_bytes(max_width, max_height);
}

static unsigned char *align4(void *p)
{
    uintptr_t v = (uintptr_t)p;
    return (unsigned char *)p + ((4u - (v & 3u)) & 3u);
}

/* ---------------------------------------------------------------- the region's rows: memory or callback */

typedef struct {
    const VocrLinesImage *im;
    const VocrLinesRows *src;
    unsigned w, h, bpp;
    int failed;
} Source;

static const unsigned char *source_row(Source *s, unsigned y)
{
    const unsigned char *p;
    if (s->im) return s->im->pixels + (size_t)y * s->im->stride;
    p = s->src->row(s->src->user, y);
    if (!p) s->failed = 1;
    return p;
}

static void put3(unsigned char *mx, unsigned char *mn, unsigned x, unsigned a, unsigned b, unsigned c)
{
    unsigned hi = a > b ? a : b, lo = a < b ? a : b;
    mx[x] = (unsigned char)(c > hi ? c : hi);
    mn[x] = (unsigned char)(c < lo ? c : lo);
}

typedef struct {
    Source *s;
    unsigned char *mx[3], *mn[3];
    unsigned next;
} Contrast;

static int contrast_load(Contrast *cs, unsigned y)
{
    const unsigned char *p = source_row(cs->s, y);
    unsigned char *mx = cs->mx[y % 3u], *mn = cs->mn[y % 3u];
    unsigned w = cs->s->w, x, a, b, c;
    if (!p) return -1;
    if (cs->s->bpp == VOCR_LINES_GREY) {
        a = b = p[0];
        for (x = 0; x + 1 < w; x++, a = b, b = c) {
            c = p[x + 1];
            put3(mx, mn, x, a, b, c);
        }
    } else {
        a = b = (77u * p[0] + 150u * p[1] + 29u * p[2]) >> 8;
        for (x = 0; x + 1 < w; x++, a = b, b = c) {
            const unsigned char *q = p + 4u * (x + 1u);
            c = (77u * q[0] + 150u * q[1] + 29u * q[2]) >> 8;
            put3(mx, mn, x, a, b, c);
        }
    }
    put3(mx, mn, w - 1u, a, b, b);
    return 0;
}

static int contrast_begin(Contrast *cs)
{
    if (contrast_load(cs, 0)) return -1;
    if (cs->s->h > 1 && contrast_load(cs, 1)) return -1;
    cs->next = 2;
    return 0;
}

static int contrast_row(Contrast *cs, unsigned y, unsigned char *out)
{
    unsigned h = cs->s->h, w = cs->s->w, x;
    unsigned ya = y ? y - 1 : 0, yc = y + 1 < h ? y + 1 : h - 1;
    const unsigned char *ma, *mb, *mc, *na, *nb, *nc;
    if (yc >= cs->next && cs->next < h) {
        if (contrast_load(cs, cs->next)) return -1;
        cs->next++;
    }
    ma = cs->mx[ya % 3u]; mb = cs->mx[y % 3u]; mc = cs->mx[yc % 3u];
    na = cs->mn[ya % 3u]; nb = cs->mn[y % 3u]; nc = cs->mn[yc % 3u];
    for (x = 0; x < w; x++) {
        unsigned hi = ma[x] > mb[x] ? ma[x] : mb[x], lo = na[x] < nb[x] ? na[x] : nb[x];
        hi = mc[x] > hi ? mc[x] : hi;
        lo = nc[x] < lo ? nc[x] : lo;
        out[x] = (unsigned char)(hi - lo);
    }
    return 0;
}

/* ---------------------------------------------------------------- bit strings (one per region column) */

static int bit(const uint32_t *col, unsigned y)
{
    return (int)((col[y >> 5] >> (y & 31u)) & 1u);
}

static unsigned close_run(unsigned s, unsigned e, uint32_t *kc)
{
    unsigned y;
    if (e - s < LMIN) return 0;
    if (kc)
        for (y = s; y < e; y++) kc[y]++;
    return e - s;
}

/* vocr_lines.c's row_kept along one column: on-pixels at most GAP apart join; runs >= LMIN count. */
static unsigned col_kept(const uint32_t *col, unsigned h, uint32_t *kc)
{
    unsigned y, s = 0, last = 0, total = 0;
    int have = 0;
    for (y = 0; y < h; y++) {
        if (!col[y >> 5]) {
            y |= 31u;
            continue;
        }
        if (!bit(col, y)) continue;
        if (have && y - last <= GAP) {
            last = y;
            continue;
        }
        if (have) total += close_run(s, last + 1, kc);
        s = last = y;
        have = 1;
    }
    if (have) total += close_run(s, last + 1, kc);
    return total;
}

static unsigned col_on(const uint32_t *col, unsigned y1, unsigned y2)
{
    unsigned y, n = 0;
    for (y = y1; y < y2; y++) n += (unsigned)bit(col, y);
    return n;
}

/* sorted(v[0 .. n))[n / 2] by rank (n is small: a band's width) */
static unsigned upper_median(const uint32_t *v, unsigned n)
{
    unsigned k, j;
    for (k = 0; k < n; k++) {
        unsigned below = 0, equal = 0;
        for (j = 0; j < n; j++) {
            below += v[j] < v[k];
            equal += v[j] == v[k];
        }
        if (below <= n / 2u && n / 2u < below + equal) return v[k];
    }
    return 0;
}

/* ---------------------------------------------------------------- the finder */

static int find_columns(Source *s, unsigned flags, unsigned char *base, VocrLineSet *out)
{
    unsigned w = s->w, h = s->h, words = (h + 31u) / 32u, cap = band_cap(w);
    unsigned x, y, i, nb = 0, nc = 0, peak = 0, th, T, p99 = 0, k;
    uint32_t hist[256];
    unsigned long long n, cum;
    uint32_t *bits, *prof, *kc, *oc;
    unsigned char *crow;
    Band *bands, *split;
    Cand *cand;
    Contrast cs;
    unsigned char *p = base;

    bits = (uint32_t *)(void *)p; p += up4((size_t)words * w * 4u);
    prof = (uint32_t *)(void *)p; p += up4((size_t)w * 4u);
    for (i = 0; i < 3; i++) { cs.mx[i] = p; p += up4(w); }
    for (i = 0; i < 3; i++) { cs.mn[i] = p; p += up4(w); }
    crow = p; p += up4(w);
    kc = (uint32_t *)(void *)p; p += up4((size_t)h * 4u);
    oc = (uint32_t *)(void *)p; p += up4((size_t)h * 4u);
    bands = (Band *)(void *)p; p += up4(sizeof(Band) * cap);
    split = (Band *)(void *)p; p += up4(sizeof(Band) * cap);
    cand = (Cand *)(void *)p;
    cs.s = s;
    cs.next = 0;

    /* 1. pass 1: the contrast histogram and T; pass 2: the column-major on-mask */
    memset(hist, 0, sizeof(hist));
    if (contrast_begin(&cs)) return VOCR_LINES_E_SOURCE;
    for (y = 0; y < h; y++) {
        if (contrast_row(&cs, y, crow)) return VOCR_LINES_E_SOURCE;
        for (x = 0; x < w; x++) hist[crow[x]]++;
    }
    n = (unsigned long long)w * h;
    for (cum = 0, i = 0; i < 256; i++) {
        cum += hist[i];
        if (cum * 100u >= n * 99u) { p99 = i; break; }
    }
    T = 2u * p99 / 5u;
    if (T < T_MIN) T = T_MIN;
    memset(bits, 0, (size_t)words * w * 4u);
    if (contrast_begin(&cs)) return VOCR_LINES_E_SOURCE;
    for (y = 0; y < h; y++) {
        uint32_t m = (uint32_t)1u << (y & 31u);
        uint32_t *word = bits + (y >> 5);
        if (contrast_row(&cs, y, crow)) return VOCR_LINES_E_SOURCE;
        for (x = 0; x < w; x++)
            if (crow[x] >= T) word[(size_t)x * words] |= m;
    }

    /* 2. kept vertical runs per column */
    for (x = 0; x < w; x++) {
        prof[x] = col_kept(bits + (size_t)x * words, h, NULL);
        if (prof[x] > peak) peak = prof[x];
    }
    th = peak / ROW_DIV;
    if (th < LMIN) th = LMIN;

    /* 3. bands of active columns */
    for (x = 0; x < w;) {
        unsigned s0, rs, last;
        if (prof[x] < th) { x++; continue; }
        s0 = rs = last = x++;
        while (x < w && (prof[x] >= th || x - last <= MERGE)) {
            if (prof[x] >= th) {
                if (x - last > 1u) {
                    unsigned q = x, g, empty = 0;
                    while (q < w && prof[q] >= th) q++;
                    for (g = last + 1u; g < x; g++)
                        if (!prof[g]) empty = 1u;
                    if (empty && last + 1u - rs >= MIN_H && q - x >= MIN_H) {
                        bands[nb].a = s0;
                        bands[nb].b = last + 1u;
                        nb++;
                        s0 = x;
                    }
                    rs = x;
                }
                last = x;
            }
            x++;
        }
        if (last + 1u - s0 >= MIN_H) {
            bands[nb].a = s0;
            bands[nb].b = last + 1u;
            nb++;
        }
    }
    if (flags & VOCR_COLUMNS_VALLEYS) {
        /* prof is done with: it now holds each column's on-pixels; each band splits at columns under
         * max(1, median / VALLEY_DIV), keeping pieces at least MIN_H wide (none: the band as it was) */
        unsigned ns = 0;
        for (x = 0; x < w; x++) {
            const uint32_t *col = bits + (size_t)x * words;
            unsigned c = 0, q;
            for (q = 0; q < words; q++) {
                uint32_t v = col[q];
                while (v) { v &= v - 1u; c++; }
            }
            prof[x] = c;
        }
        for (i = 0; i < nb; i++) {
            unsigned a = bands[i].a, b = bands[i].b, med = upper_median(prof + a, b - a), lim, s0 = 0, have = 0;
            unsigned first = ns;
            lim = med / VALLEY_DIV;
            if (lim < 1u) lim = 1u;
            for (x = a; x <= b; x++) {
                int hi = x < b && prof[x] >= lim;
                if (hi && !have) { s0 = x; have = 1; }
                else if (!hi && have) {
                    if (x - s0 >= MIN_H) { split[ns].a = s0; split[ns].b = x; ns++; }
                    have = 0;
                }
            }
            if (ns == first) { split[ns].a = a; split[ns].b = b; ns++; }
        }
        memcpy(bands, split, sizeof(Band) * ns);
        nb = ns;
    }

    /* 4. per band: its column */
    for (i = 0; i < nb; i++) {
        unsigned a = bands[i].a, b = bands[i].b, bh = b - a, d, y1 = 0, y2 = 0, last, dense;
        unsigned long best = 0, mass = 0;
        unsigned c0 = 0, prevc = 0, have = 0, up, dn, left, right;
        memset(kc, 0, (size_t)h * 4u);
        memset(oc, 0, (size_t)h * 4u);
        for (x = a; x < b; x++) {
            const uint32_t *col = bits + (size_t)x * words;
            col_kept(col, h, kc);
            for (y = 0; y < h; y++) {
                if (!col[y >> 5]) { y |= 31u; continue; }
                oc[y] += (uint32_t)bit(col, y);
            }
        }
        if (flags & VOCR_COLUMNS_VALLEYS) {
            dense = bh / DENSE_DIV;
            if (dense < 2u) dense = 2u;
            for (y = 0; y < h; y++) {
                if (kc[y] < dense) kc[y] = 0;
                if (oc[y] < dense) oc[y] = 0;
            }
        }
        for (y = 0; y <= h; y++) {
            if (y < h && !kc[y]) continue;
            if (have && (y == h || y - prevc > GAPC * bh)) {
                if (mass > best) { best = mass; y1 = c0; y2 = prevc + 1; }
                have = 0;
            }
            if (y == h) break;
            if (!have) { c0 = y; mass = 0; have = 1; }
            mass += kc[y];
            prevc = y;
        }
        if (!best) continue;
        d = bh > 4u ? bh : 4u;
        for (last = y2 - 1, y = y2; y < h && y - last <= d; y++)
            if (oc[y]) last = y;
        y2 = last + 1;
        for (last = y1, y = y1; y > 0 && last - (y - 1) <= d; y--)
            if (oc[y - 1]) last = y - 1;
        y1 = last;
        up = bh / 2u;
        if (i) { if ((a - bands[i - 1].b) / 2u < up) up = (a - bands[i - 1].b) / 2u; }
        else if (a < up) up = a;
        dn = bh / 2u;
        if (i + 1 < nb) { if ((bands[i + 1].a - b) / 2u < dn) dn = (bands[i + 1].a - b) / 2u; }
        else if (w - b < dn) dn = w - b;
        for (left = a; a - left < up && col_on(bits + (size_t)(left - 1) * words, y1, y2) >= 2; left--) {}
        for (right = b; right - b < dn && col_on(bits + (size_t)right * words, y1, y2) >= 2; right++) {}
        cand[nc].y1 = y1; cand[nc].x1 = left; cand[nc].y2 = y2; cand[nc].x2 = right;
        nc++;
    }

    /* 5-7. drop narrow columns and horizontal text, pad (compacting the kept ones in place) */
    out->count = 0;
    out->truncated = 0;
    if (nc) {
        unsigned med = 0, j, kept = 0;
        for (k = 0; k < nc; k++) {
            unsigned wk = cand[k].x2 - cand[k].x1, below = 0, equal = 0;
            for (j = 0; j < nc; j++) {
                unsigned wj = cand[j].x2 - cand[j].x1;
                below += wj < wk;
                equal += wj == wk;
            }
            if (below <= nc / 2u && nc / 2u < below + equal) { med = wk; break; }
        }
        for (k = 0; k < nc; k++) {
            unsigned cw = cand[k].x2 - cand[k].x1, ch = cand[k].y2 - cand[k].y1;
            Cand c = cand[k];
            if (100u * cw < FURI_PCT * med) continue;
            if (ASPECT_DEN * cw > ASPECT_NUM * ch) continue;
            c.x1 = c.x1 > PAD ? c.x1 - PAD : 0;
            c.y1 = c.y1 > PAD ? c.y1 - PAD : 0;
            c.x2 = c.x2 + PAD < w ? c.x2 + PAD : w;
            c.y2 = c.y2 + PAD < h ? c.y2 + PAD : h;
            cand[kept++] = c;
        }
        /* 8. reading order: right edge descending, then top ascending (insertion sort: stable) */
        for (k = 1; k < kept; k++) {
            Cand c = cand[k];
            j = k;
            while (j > 0 && (cand[j - 1].x2 < c.x2 || (cand[j - 1].x2 == c.x2 && cand[j - 1].y1 > c.y1))) {
                cand[j] = cand[j - 1];
                j--;
            }
            cand[j] = c;
        }
        for (k = 0; k < kept; k++) {
            VocrLine *ln;
            int dark;
            if (out->count == VOCR_LINES_MAX) { out->truncated = 1; break; }
            ln = &out->lines[out->count++];
            ln->x = cand[k].x1; ln->y = cand[k].y1;
            ln->w = cand[k].x2 - cand[k].x1; ln->h = cand[k].y2 - cand[k].y1;
            if (flags & VOCR_COLUMNS_NO_DARK) {
                ln->dark = 0;
                continue;
            }
            dark = s->im ? vocr_line_dark(s->im, ln->x, ln->y, ln->w, ln->h)
                         : vocr_line_dark_rows(s->src, ln->x, ln->y, ln->w, ln->h);
            if (dark < 0) return dark;
            ln->dark = (unsigned)dark;
        }
    }
    return VOCR_LINES_OK;
}

int vocr_find_columns(const VocrLinesImage *im, unsigned flags, void *arena, size_t arena_bytes, VocrLineSet *out)
{
    Source s;
    int rc;
    if (!out) return VOCR_LINES_E_ARG;
    out->count = 0;
    out->truncated = 0;
    if (!im || !im->pixels || !im->width || !im->height || !arena || (flags & ~VOCR_COLUMNS_FLAGS) ||
        (im->bpp != VOCR_LINES_GREY && im->bpp != VOCR_LINES_RGBA) || im->stride < (size_t)im->width * im->bpp)
        return VOCR_LINES_E_ARG;
    if (arena_bytes < vocr_columns_arena_bytes(im->width, im->height)) return VOCR_LINES_E_ARENA;
    s.im = im;
    s.src = NULL;
    s.w = im->width;
    s.h = im->height;
    s.bpp = im->bpp;
    s.failed = 0;
    rc = find_columns(&s, flags, align4(arena), out);
    if (rc) out->count = 0;
    return rc;
}

int vocr_find_columns_rows(const VocrLinesRows *src, unsigned flags, void *arena, size_t arena_bytes,
                           VocrLineSet *out)
{
    Source s;
    int rc;
    if (!out) return VOCR_LINES_E_ARG;
    out->count = 0;
    out->truncated = 0;
    if (!src || !src->row || !src->width || !src->height || !arena || (flags & ~VOCR_COLUMNS_FLAGS) ||
        (src->bpp != VOCR_LINES_GREY && src->bpp != VOCR_LINES_RGBA))
        return VOCR_LINES_E_ARG;
    if (arena_bytes < vocr_columns_arena_bytes(src->width, src->height)) return VOCR_LINES_E_ARENA;
    s.im = NULL;
    s.src = src;
    s.w = src->width;
    s.h = src->height;
    s.bpp = src->bpp;
    s.failed = 0;
    rc = find_columns(&s, flags, align4(arena), out);
    if (rc) out->count = 0;
    return rc;
}

int vocr_column_rotate(const unsigned char *src, size_t src_stride, unsigned bpp, unsigned w, unsigned h,
                       unsigned char *dst, size_t dst_stride)
{
    unsigned i, j;
    if (!src || !dst || !w || !h || (bpp != VOCR_LINES_GREY && bpp != VOCR_LINES_RGBA) ||
        src_stride < (size_t)w * bpp || dst_stride < (size_t)h)
        return VOCR_LINES_E_ARG;
    for (i = 0; i < w; i++) {
        unsigned char *d = dst + (size_t)i * dst_stride;
        unsigned x = w - 1u - i;
        if (bpp == VOCR_LINES_GREY) {
            for (j = 0; j < h; j++) d[j] = src[(size_t)j * src_stride + x];
        } else {
            for (j = 0; j < h; j++) {
                const unsigned char *q = src + (size_t)j * src_stride + (size_t)x * 4u;
                d[j] = (unsigned char)((77u * q[0] + 150u * q[1] + 29u * q[2]) >> 8);
            }
        }
    }
    return VOCR_LINES_OK;
}
