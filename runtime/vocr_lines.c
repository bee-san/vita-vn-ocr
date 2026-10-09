/* Dialogue line finder: see vocr_lines.h and docs/linefind.md.
 *
 * GRAD finder, in order:
 *  1. grey  g = (77 R + 150 G + 29 B) >> 8 (the recognizer's own grey)
 *  2. local contrast c = max - min of g over each 3x3 window (edges replicated):
 *     text strokes are high-contrast whether they are bright, dark or outlined
 *  3. threshold T = max(40, 2/5 of the 99th percentile of c); on = c >= T
 *  4. per row, join on-pixels at most 10 px apart into runs and keep runs of at least
 *     16 px (a text line becomes long runs; specks, vertical borders and the edges of
 *     big shapes do not); prof(y) = kept run length in row y
 *  5. rows with prof >= max(16, max(prof) / 16) are active; active rows with gaps of
 *     at most 2 rows form bands; bands under 8 rows are dropped
 *  6. per band: the column cluster with the most kept pixels (clusters split at gaps
 *     over twice the band height) is the line; it is widened over on-pixels at most
 *     max(4, h) columns away (punctuation, small kana) and grown by up to h/2 rows
 *     (never past half the gap to the next band) while rows hold 2+ on-pixels
 *  7. lines under 55% of the median line height are dropped (furigana, icons)
 *  8. boxes are padded by 3 px, which after the 1 px dilation of step 2 matches the
 *     legacy finder's 4 px around bright ink, and each gets a polarity flag
 *
 * Memory: a 1-bit on-mask of the region plus a few rows and per-row/column counters,
 * all carved from the caller's arena. Integer arithmetic only.
 */
#include "vocr_lines.h"

#include <stdint.h>
#include <string.h>

#define T_MIN 40u
#define GAP 10u
#define LMIN 16u
#define ROW_DIV 16u
#define MIN_H 8u
#define MERGE 2u
#define GAPC 2u
#define PAD 3u
#define FURI_PCT 55u

/* legacy (vjo_ocr_find_lines) limits */
#define LEG_MAX_W 960u
#define LEG_MAX_H 544u
#define LEG_MAX_LINES 8u
#define LEG_MAX_LINE_H 96u

typedef struct {
    unsigned a, b;          /* band rows [a, b) */
} Band;

typedef struct {
    unsigned x1, top, x2, bot;
} Cand;

static unsigned band_cap(unsigned h)
{
    /* bands are separated by at least MERGE + 1 inactive rows */
    return h / (MERGE + 2u) + 2u;
}

static size_t up4(size_t n)
{
    return (n + 3u) & ~(size_t)3u;
}

static size_t grad_bytes(unsigned w, unsigned h)
{
    size_t words = ((size_t)w + 31u) / 32u;
    return up4(words * h * 4u)              /* on-mask */
         + up4((size_t)h * 4u)              /* prof */
         + up4((size_t)w) * 10u             /* grey, row max, row min rings; contrast row */
         + up4((size_t)w * 4u) * 2u         /* kc, oc */
         + up4(sizeof(Band) * band_cap(h))
         + up4(sizeof(Cand) * band_cap(h));
}

static size_t legacy_bytes(unsigned w, unsigned h)
{
    (void)w;
    return up4((size_t)h * 4u) * 2u;        /* left, right */
}

size_t vocr_lines_arena_bytes(unsigned max_width, unsigned max_height)
{
    size_t g, l;
    if (!max_width || !max_height) return 0;
    g = grad_bytes(max_width, max_height);
    l = legacy_bytes(max_width, max_height);
    return 3u + (g > l ? g : l);            /* 3: worst-case alignment of the base */
}

static unsigned char *align4(void *p)
{
    uintptr_t v = (uintptr_t)p;
    return (unsigned char *)p + ((4u - (v & 3u)) & 3u);
}

static unsigned grey_at(const VocrLinesImage *im, unsigned x, unsigned y)
{
    const unsigned char *p = im->pixels + (size_t)y * im->stride + (size_t)x * im->bpp;
    if (im->bpp == VOCR_LINES_GREY) return p[0];
    return (77u * p[0] + 150u * p[1] + 29u * p[2]) >> 8;
}

static void grey_row(const VocrLinesImage *im, unsigned y, unsigned char *out)
{
    unsigned x, w = im->width;
    const unsigned char *p = im->pixels + (size_t)y * im->stride;
    if (im->bpp == VOCR_LINES_GREY) {
        memcpy(out, p, w);
        return;
    }
    for (x = 0; x < w; x++, p += 4)
        out[x] = (unsigned char)((77u * p[0] + 150u * p[1] + 29u * p[2]) >> 8);
}

/* Horizontal 3-max and 3-min of one grey row (edges replicated). */
static void row_maxmin(const unsigned char *g, unsigned w, unsigned char *mx, unsigned char *mn)
{
    unsigned x;
    for (x = 0; x < w; x++) {
        unsigned a = g[x ? x - 1 : 0], b = g[x], c = g[x + 1 < w ? x + 1 : w - 1];
        unsigned hi = a > b ? a : b, lo = a < b ? a : b;
        mx[x] = (unsigned char)(c > hi ? c : hi);
        mn[x] = (unsigned char)(c < lo ? c : lo);
    }
}

/* Streams the 3x3 contrast of every row: calls back with row y's contrast in out. */
typedef struct {
    const VocrLinesImage *im;
    unsigned char *g[3], *mx[3], *mn[3];
    unsigned next;  /* next source row to load into the ring */
} Contrast;

static void contrast_load(Contrast *cs, unsigned y)
{
    unsigned s = y % 3u;
    grey_row(cs->im, y, cs->g[s]);
    row_maxmin(cs->g[s], cs->im->width, cs->mx[s], cs->mn[s]);
}

static void contrast_begin(Contrast *cs)
{
    contrast_load(cs, 0);
    if (cs->im->height > 1) contrast_load(cs, 1);
    cs->next = 2;
}

/* Contrast of row y (rows must be requested in order 0, 1, 2, ...). */
static void contrast_row(Contrast *cs, unsigned y, unsigned char *out)
{
    unsigned h = cs->im->height, w = cs->im->width, x;
    unsigned ya = y ? y - 1 : 0, yc = y + 1 < h ? y + 1 : h - 1;
    const unsigned char *ma, *mb, *mc, *na, *nb, *nc;
    if (yc >= cs->next && cs->next < h) {
        contrast_load(cs, cs->next);
        cs->next++;
    }
    ma = cs->mx[ya % 3u]; mb = cs->mx[y % 3u]; mc = cs->mx[yc % 3u];
    na = cs->mn[ya % 3u]; nb = cs->mn[y % 3u]; nc = cs->mn[yc % 3u];
    for (x = 0; x < w; x++) {
        unsigned hi = ma[x], lo = na[x];
        if (mb[x] > hi) hi = mb[x];
        if (mc[x] > hi) hi = mc[x];
        if (nb[x] < lo) lo = nb[x];
        if (nc[x] < lo) lo = nc[x];
        out[x] = (unsigned char)(hi - lo);
    }
}

static int bit(const uint32_t *row, unsigned x)
{
    return (int)((row[x >> 5] >> (x & 31u)) & 1u);
}

static unsigned close_run(unsigned s, unsigned e, uint32_t *kc)
{
    unsigned x;
    if (e - s < LMIN) return 0;
    if (kc)
        for (x = s; x < e; x++) kc[x]++;
    return e - s;
}

/* Joins on-pixels at most GAP apart; returns the length of runs >= LMIN and, if kc
 * is given, counts their pixels per column. */
static unsigned row_kept(const uint32_t *row, unsigned w, uint32_t *kc)
{
    unsigned x, s = 0, last = 0, total = 0;
    int have = 0;
    for (x = 0; x < w; x++) {
        if (!row[x >> 5]) {             /* skip empty words */
            x |= 31u;
            continue;
        }
        if (!bit(row, x)) continue;
        if (have && x - last <= GAP) {
            last = x;
            continue;
        }
        if (have) total += close_run(s, last + 1, kc);
        s = last = x;
        have = 1;
    }
    if (have) total += close_run(s, last + 1, kc);
    return total;
}

static unsigned row_on(const uint32_t *row, unsigned x1, unsigned x2)
{
    unsigned x, n = 0;
    for (x = x1; x < x2; x++) n += (unsigned)bit(row, x);
    return n;
}

static int find_grad(const VocrLinesImage *im, unsigned char *base, VocrLineSet *out)
{
    unsigned w = im->width, h = im->height, words = (w + 31u) / 32u;
    unsigned x, y, i, nb = 0, nc = 0, peak = 0, th, T, p99 = 0;
    uint32_t hist[256];
    unsigned long long n, cum;
    uint32_t *bits, *prof, *kc, *oc;
    unsigned char *crow;
    Band *bands;
    Cand *cand;
    Contrast cs;
    unsigned char *p = base;

    bits = (uint32_t *)(void *)p; p += up4((size_t)words * h * 4u);
    prof = (uint32_t *)(void *)p; p += up4((size_t)h * 4u);
    for (i = 0; i < 3; i++) { cs.g[i] = p; p += up4(w); }
    for (i = 0; i < 3; i++) { cs.mx[i] = p; p += up4(w); }
    for (i = 0; i < 3; i++) { cs.mn[i] = p; p += up4(w); }
    crow = p; p += up4(w);
    kc = (uint32_t *)(void *)p; p += up4((size_t)w * 4u);
    oc = (uint32_t *)(void *)p; p += up4((size_t)w * 4u);
    bands = (Band *)(void *)p; p += up4(sizeof(Band) * band_cap(h));
    cand = (Cand *)(void *)p;
    cs.im = im;
    cs.next = 0;

    /* Pass 1: contrast histogram -> threshold. */
    memset(hist, 0, sizeof(hist));
    contrast_begin(&cs);
    for (y = 0; y < h; y++) {
        contrast_row(&cs, y, crow);
        for (x = 0; x < w; x++) hist[crow[x]]++;
    }
    n = (unsigned long long)w * h;
    for (cum = 0, i = 0; i < 256; i++) {
        cum += hist[i];
        if (cum * 100u >= n * 99u) { p99 = i; break; }
    }
    T = 2u * p99 / 5u;
    if (T < T_MIN) T = T_MIN;

    /* Pass 2: on-mask and row profile of kept runs. */
    memset(bits, 0, (size_t)words * h * 4u);
    contrast_begin(&cs);
    for (y = 0; y < h; y++) {
        uint32_t *row = bits + (size_t)y * words;
        contrast_row(&cs, y, crow);
        for (x = 0; x < w; x++)
            if (crow[x] >= T) row[x >> 5] |= (uint32_t)1u << (x & 31u);
        prof[y] = row_kept(row, w, NULL);
        if (prof[y] > peak) peak = prof[y];
    }
    th = peak / ROW_DIV;
    if (th < LMIN) th = LMIN;

    /* Bands of active rows (gaps of up to MERGE rows bridged), at least MIN_H tall. */
    for (y = 0; y < h;) {
        unsigned s, last;
        if (prof[y] < th) { y++; continue; }
        s = last = y++;
        while (y < h && (prof[y] >= th || y - last <= MERGE)) {
            if (prof[y] >= th) last = y;
            y++;
        }
        if (last + 1 - s >= MIN_H) {
            bands[nb].a = s;
            bands[nb].b = last + 1;
            nb++;
        }
    }

    for (i = 0; i < nb; i++) {
        unsigned a = bands[i].a, b = bands[i].b, bh = b - a, d, x1 = 0, x2 = 0, last;
        unsigned long best = 0, mass = 0;
        unsigned cs0 = 0, prevc = 0, have = 0, up, dn, top, bot;
        memset(kc, 0, (size_t)w * 4u);
        memset(oc, 0, (size_t)w * 4u);
        for (y = a; y < b; y++) {
            const uint32_t *row = bits + (size_t)y * words;
            row_kept(row, w, kc);
            for (x = 0; x < w; x++) {
                if (!row[x >> 5]) { x |= 31u; continue; }
                oc[x] += (uint32_t)bit(row, x);
            }
        }
        /* column clusters of kept pixels; keep the heaviest (first on ties) */
        for (x = 0; x <= w; x++) {
            if (x < w && !kc[x]) continue;
            if (have && (x == w || x - prevc > GAPC * bh)) {
                if (mass > best) { best = mass; x1 = cs0; x2 = prevc + 1; }
                have = 0;
            }
            if (x == w) break;
            if (!have) { cs0 = x; mass = 0; have = 1; }
            mass += kc[x];
            prevc = x;
        }
        if (!best) continue;
        /* widen over nearby on-pixels */
        d = bh > 4u ? bh : 4u;
        for (last = x2 - 1, x = x2; x < w && x - last <= d; x++)
            if (oc[x]) last = x;
        x2 = last + 1;
        for (last = x1, x = x1; x > 0 && last - (x - 1) <= d; x--)
            if (oc[x - 1]) last = x - 1;
        x1 = last;
        /* grow vertically into the gaps, at most half way to the neighbour band */
        up = bh / 2u;
        if (i) { if ((a - bands[i - 1].b) / 2u < up) up = (a - bands[i - 1].b) / 2u; }
        else if (a < up) up = a;
        dn = bh / 2u;
        if (i + 1 < nb) { if ((bands[i + 1].a - b) / 2u < dn) dn = (bands[i + 1].a - b) / 2u; }
        else if (h - b < dn) dn = h - b;
        for (top = a; a - top < up && row_on(bits + (size_t)(top - 1) * words, x1, x2) >= 2; top--) {}
        for (bot = b; bot - b < dn && row_on(bits + (size_t)bot * words, x1, x2) >= 2; bot++) {}
        cand[nc].x1 = x1; cand[nc].top = top; cand[nc].x2 = x2; cand[nc].bot = bot;
        nc++;
    }

    /* drop lines under FURI_PCT of the median height (upper median) */
    out->count = 0;
    out->truncated = 0;
    if (nc) {
        unsigned med = 0, k, j;
        /* med = sorted(heights)[nc / 2] */
        for (k = 0; k < nc; k++) {
            unsigned hk = cand[k].bot - cand[k].top, below = 0, equal = 0;
            for (j = 0; j < nc; j++) {
                unsigned hj = cand[j].bot - cand[j].top;
                below += hj < hk;
                equal += hj == hk;
            }
            if (below <= nc / 2u && nc / 2u < below + equal) { med = hk; break; }
        }
        for (k = 0; k < nc; k++) {
            unsigned hk = cand[k].bot - cand[k].top, lx, ly, ex, ey;
            VocrLine *ln;
            if (100u * hk < FURI_PCT * med) continue;
            if (out->count == VOCR_LINES_MAX) { out->truncated = 1; break; }
            lx = cand[k].x1 > PAD ? cand[k].x1 - PAD : 0;
            ly = cand[k].top > PAD ? cand[k].top - PAD : 0;
            ex = cand[k].x2 + PAD < w ? cand[k].x2 + PAD : w;
            ey = cand[k].bot + PAD < h ? cand[k].bot + PAD : h;
            ln = &out->lines[out->count++];
            ln->x = lx; ln->y = ly; ln->w = ex - lx; ln->h = ey - ly;
            ln->dark = (unsigned)vocr_line_dark(im, lx, ly, ln->w, ln->h);
        }
    }
    return VOCR_LINES_OK;
}

/* Exact port of vita-jp-overlay's vjo_ocr_find_lines (core/ocr_lines.c). */
static int find_legacy(const VocrLinesImage *im, unsigned char *base, VocrLineSet *out)
{
    unsigned w = im->width, h = im->height, y, start = 0, last = 0, active = 0;
    uint32_t *left = (uint32_t *)(void *)base, *right = left + up4((size_t)h * 4u) / 4u;
    out->count = 0;
    out->truncated = 0;
    if (w > LEG_MAX_W || h > LEG_MAX_H) return VOCR_LINES_E_REGION;
    for (y = 0; y <= h; y++) {
        unsigned count = 0;
        if (y < h) {
            const unsigned char *p = im->pixels + (size_t)y * im->stride;
            unsigned x;
            left[y] = w; right[y] = 0;
            for (x = 0; x < w; x++, p += im->bpp) {
                int bright = im->bpp == VOCR_LINES_GREY ? p[0] >= 180
                                                        : p[0] >= 180 && p[1] >= 180 && p[2] >= 180;
                if (bright) {
                    if (x < left[y]) left[y] = x;
                    right[y] = x + 1;
                    count++;
                }
            }
            if (count > w * 3u / 4u) return VOCR_LINES_E_REGION;
        }
        if (count >= 2) {
            if (!active) { start = y; active = 1; }
            last = y;
        }
        if (active && (y == h || y - last >= 3)) {
            unsigned lh = last - start + 1;
            if (lh >= 8) {
                unsigned x1 = w, x2 = 0, yy;
                if (lh > LEG_MAX_LINE_H - 8 || out->count == LEG_MAX_LINES) return VOCR_LINES_E_REGION;
                for (yy = start; yy <= last; yy++) {
                    if (left[yy] < x1) x1 = left[yy];
                    if (right[yy] > x2) x2 = right[yy];
                }
                if (x2 > x1) {
                    VocrLine *line = &out->lines[out->count++];
                    unsigned endx = x2 + 4 < w ? x2 + 4 : w, endy = last + 5 < h ? last + 5 : h;
                    line->x = x1 > 4 ? x1 - 4 : 0;
                    line->y = start > 4 ? start - 4 : 0;
                    line->w = endx - line->x;
                    line->h = endy - line->y;
                    line->dark = 0;
                }
            }
            active = 0;
        }
    }
    return VOCR_LINES_OK;
}

int vocr_find_lines(const VocrLinesImage *im, int finder, void *arena, size_t arena_bytes,
                    VocrLineSet *out)
{
    unsigned char *base;
    int rc;
    if (!out) return VOCR_LINES_E_ARG;
    out->count = 0;
    out->truncated = 0;
    if (!im || !im->pixels || !im->width || !im->height || !arena ||
        (im->bpp != VOCR_LINES_GREY && im->bpp != VOCR_LINES_RGBA) ||
        im->stride < (size_t)im->width * im->bpp ||
        (finder != VOCR_LINES_GRAD && finder != VOCR_LINES_LEGACY))
        return VOCR_LINES_E_ARG;
    if (arena_bytes < vocr_lines_arena_bytes(im->width, im->height)) return VOCR_LINES_E_ARENA;
    base = align4(arena);
    rc = finder == VOCR_LINES_GRAD ? find_grad(im, base, out) : find_legacy(im, base, out);
    if (rc) out->count = 0; /* never report a partial result */
    return rc;
}

int vocr_line_dark(const VocrLinesImage *im, unsigned x, unsigned y, unsigned w, unsigned h)
{
    uint32_t hist[256], bh[256];
    unsigned long long n, nb_, k, cum, nd = 0, nbr = 0;
    unsigned i, j, bg = 0, lo = 0, hi = 0, d;
    if (!im || !im->pixels || (im->bpp != VOCR_LINES_GREY && im->bpp != VOCR_LINES_RGBA) ||
        !w || !h || x >= im->width || y >= im->height || w > im->width - x || h > im->height - y)
        return VOCR_LINES_E_ARG;
    memset(hist, 0, sizeof(hist));
    memset(bh, 0, sizeof(bh));
    for (j = 0; j < h; j++)
        for (i = 0; i < w; i++) hist[grey_at(im, x + i, y + j)]++;
    /* border: top row, bottom row, left column, right column (corners twice) */
    for (i = 0; i < w; i++) {
        bh[grey_at(im, x + i, y)]++;
        bh[grey_at(im, x + i, y + h - 1)]++;
    }
    for (j = 0; j < h; j++) {
        bh[grey_at(im, x, y + j)]++;
        bh[grey_at(im, x + w - 1, y + j)]++;
    }
    nb_ = 2ul * w + 2ul * h;
    for (cum = 0, i = 0; i < 256; i++) {        /* sorted(border)[nb_ / 2] */
        cum += bh[i];
        if (cum > nb_ / 2u) { bg = i; break; }
    }
    n = (unsigned long)w * h;
    k = n * 2u / 100u;
    for (cum = 0, i = 0; i < 256; i++) {
        cum += hist[i];
        if (cum >= k) { lo = i; break; }
    }
    k = n * 98u / 100u;
    for (cum = 0, i = 0; i < 256; i++) {
        cum += hist[i];
        if (cum >= k) { hi = i; break; }
    }
    d = (hi - lo) / 4u;
    if (d < 16u) d = 16u;
    for (i = 0; i < 256; i++) {
        if (i + d < bg) nd += hist[i];
        if (i > bg + d) nbr += hist[i];
    }
    return nd > 8u * nbr ? 1 : 0;
}
