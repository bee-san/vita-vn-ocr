/* vocr_tsd_post.c - textspot-desktop's box post-pass and reading filter (vocr_tsd_post.h): the C twins of
 * tools/tsdesk.py's merge_overlaps, split_rows, extend_ends and drop_set, bit for bit. */
#include "vocr_tsd_post.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#if defined(__AVX2__) && !defined(VOCR_TSD_SCALAR) && !defined(VOCR_TSDR_SCALAR)
#include <immintrin.h>
#define TSDP_AVX2 1
#else
#define TSDP_AVX2 0
#endif

void vocr_tsdp_defaults(vocr_tsdp_opts *o)
{
    if (!o) return;
    memset(o, 0, sizeof *o);
    o->valley_pct = 10;
    o->rows_len = 300;
    o->gap_weak = 3;
    o->ext_gap_pct = 70;
    o->ext_run_pct = 15;
    o->ext_dash_pct = 80;
    o->ext_join = 1;
    o->vflip_join_pct = 100;
    o->vflip_min = 3;
    o->vflip_h_pct = 50;
}

void vocr_tsdd_defaults(vocr_tsdd_opts *o)
{
    if (!o) return;
    memset(o, 0, sizeof *o);
    o->nojp_gap_pct = 100;
    o->rep_len = 4;
}

int vocr_tsdp_nopts(void) { return VOCR_TSDP_NOPTS; }
int vocr_tsdd_nopts(void) { return VOCR_TSDD_NOPTS; }

int vocr_tsdp_pre_on(const vocr_tsdp_opts *o)
{
    return o && (o->vflip_pct || o->merge_pct || o->rows_pct || o->gap_pct);
}
int vocr_tsdp_post_on(const vocr_tsdp_opts *o) { return o && o->ext_pct; }

static int r100(long v, int pct) { return (int)((v * (long)pct + 50) / 100); }
static int imax(int a, int b) { return a > b ? a : b; }
static int imin(int a, int b) { return a < b ? a : b; }
static double dmin(double a, double b) { return b < a ? b : a; }    /* Python's min(a, b): a unless b < a */
static double dmax(double a, double b) { return b > a ? b : a; }    /* Python's max(a, b) */

/* ----------------------------------------------------------------------------- scratch */

typedef struct {
    unsigned char *d, *c3, *m0, *m;   /* window planes (at most the frame) */
    unsigned char *vmax, *vmin;       /* one row each (W + 34) */
    int *rc, *cc;                     /* W + H ints each */
    unsigned char *vcol;              /* per window column: 1 when it has ink in the line's rows (W) */
    int *lims;                        /* 4 ints per box (cap) */
    unsigned char *alive;             /* cap */
    unsigned char *reach;             /* 2 per box (cap) */
    vocr_tsdp_box *tmp;               /* cap boxes */
} tsdp_scr;

static size_t plane_bytes(unsigned w, unsigned h) { return ((size_t)w + 2) * ((size_t)h + 2); }

size_t vocr_tsdp_scratch_bytes(unsigned w, unsigned h, unsigned cap)
{
    size_t P = (plane_bytes(w, h) + 15) & ~(size_t)15;
    return 4 * P + 2 * ((size_t)h + (size_t)w) * sizeof(int) + ((size_t)w + 16) + (size_t)cap * (4 * sizeof(int) + 3) +
           (size_t)cap * sizeof(vocr_tsdp_box) + 2 * ((size_t)w + 48) + 64;
}

static void scr_map(tsdp_scr *s, void *buf, unsigned w, unsigned h, unsigned cap)
{
    unsigned char *p = (unsigned char *)(((uintptr_t)buf + 7u) & ~(uintptr_t)7u);
    size_t P = (plane_bytes(w, h) + 15) & ~(size_t)15;
    s->tmp = (vocr_tsdp_box *)(void *)p;
    p += (size_t)cap * sizeof(vocr_tsdp_box);
    s->rc = (int *)(void *)p;                /* H rows of a box */
    p += ((size_t)h + (size_t)w) * sizeof(int);
    s->cc = (int *)(void *)p;                /* a sort buffer (H) or the columns of a search window (W) */
    p += ((size_t)h + (size_t)w) * sizeof(int);
    s->lims = (int *)(void *)p;
    p += (size_t)cap * 4 * sizeof(int);
    s->d = p;
    p += P;
    s->c3 = p;
    p += P;
    s->m0 = p;
    p += P;
    s->m = p;
    p += P;
    s->vcol = p;
    p += (size_t)w + 16;
    s->alive = p;
    p += cap;
    s->reach = p;
    p += 2 * (size_t)cap;
    s->vmax = p;
    p += (size_t)w + 48;
    s->vmin = p;
}

/* ----------------------------------------------------------------------------- the ink (tools/tsdesk.ink_mask) */

/* the element ceil(pct n / 100) - 1 (clipped) of the values counted in hist[256] (n of them): tools/tsdesk_refine.
 * quantile */
static int hist_quantile(const unsigned *hist, unsigned long n, int pct)
{
    long idx;
    unsigned long cum = 0;
    int v;
    if (n == 0) return 0;
    idx = ((long)pct * (long)n + 99) / 100 - 1;
    if (idx < 0) idx = 0;
    if (idx > (long)n - 1) idx = (long)n - 1;
    for (v = 0; v < 256; v++) {
        cum += hist[v];
        if (cum > (unsigned long)idx) return v;
    }
    return 255;
}

/* the lower median (element (n - 1) div 2 of the sorted values) of the window's border ring as numpy concatenates it:
 * its first row, its last row, then its first and last columns without those rows (a one-row or one-column window
 * counts its pixels twice, as numpy does) */
static int ring_median(const unsigned char *g, unsigned stride, int wx0, int wy0, int mw, int mh)
{
    unsigned hist[256];
    unsigned long n = 0;
    int x, y;
    memset(hist, 0, sizeof hist);
    for (x = 0; x < mw; x++) hist[g[(size_t)wy0 * stride + (size_t)(wx0 + x)]]++;
    for (x = 0; x < mw; x++) hist[g[(size_t)(wy0 + mh - 1) * stride + (size_t)(wx0 + x)]]++;
    n += 2 * (unsigned long)mw;
    for (y = 1; y < mh - 1; y++) {
        hist[g[(size_t)(wy0 + y) * stride + (size_t)wx0]]++;
        hist[g[(size_t)(wy0 + y) * stride + (size_t)(wx0 + mw - 1)]]++;
        n += 2;
    }
    {
        unsigned long idx = (n - 1) / 2, cum = 0;
        int v;
        for (v = 0; v < 256; v++) {
            cum += hist[v];
            if (cum > idx) return v;
        }
    }
    return 255;
}

/* c3 (tools/tsdesk_refine._c3: the 3 x 3 max - min, the frame's edge pixels repeated) of one row y over the columns
 * [x0, x0 + mw): the refinement's row kernel (desktop/vocr_tsd_refine.c row_c3), AVX2 when built for it */
static void row_c3(const unsigned char *g, unsigned W, unsigned H, unsigned stride, int x0, int mw, int y,
                   unsigned char *vmax, unsigned char *vmin, unsigned char *out)
{
    int ya = y > 0 ? y - 1 : 0, yc = y + 1 < (int)H ? y + 1 : (int)H - 1;
    const unsigned char *a = g + (size_t)ya * stride, *b = g + (size_t)y * stride, *c = g + (size_t)yc * stride;
    int i, n = mw + 2, lo = 0, hi = n;
    if (x0 - 1 < 0) lo = 1;
    if (x0 + mw > (int)W - 1) hi = n - 1;
    i = lo;
#if TSDP_AVX2
    for (; i + 32 <= hi; i += 32) {
        int x = x0 - 1 + i;
        __m256i va = _mm256_loadu_si256((const __m256i *)(const void *)(a + x));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(const void *)(b + x));
        __m256i vc = _mm256_loadu_si256((const __m256i *)(const void *)(c + x));
        _mm256_storeu_si256((__m256i *)(void *)(vmax + i), _mm256_max_epu8(_mm256_max_epu8(va, vb), vc));
        _mm256_storeu_si256((__m256i *)(void *)(vmin + i), _mm256_min_epu8(_mm256_min_epu8(va, vb), vc));
    }
#endif
    for (; i < hi; i++) {
        int x = x0 - 1 + i;
        unsigned char p = a[x], q = b[x], r = c[x];
        unsigned char mx = p > q ? p : q, mn = p < q ? p : q;
        vmax[i] = mx > r ? mx : r;
        vmin[i] = mn < r ? mn : r;
    }
    if (lo == 1) {
        vmax[0] = vmax[1];
        vmin[0] = vmin[1];
    }
    if (hi == n - 1) {
        vmax[n - 1] = vmax[n - 2];
        vmin[n - 1] = vmin[n - 2];
    }
    i = 0;
#if TSDP_AVX2
    for (; i + 32 <= mw; i += 32) {
        __m256i m0 = _mm256_loadu_si256((const __m256i *)(const void *)(vmax + i));
        __m256i m1 = _mm256_loadu_si256((const __m256i *)(const void *)(vmax + i + 1));
        __m256i m2 = _mm256_loadu_si256((const __m256i *)(const void *)(vmax + i + 2));
        __m256i n0 = _mm256_loadu_si256((const __m256i *)(const void *)(vmin + i));
        __m256i n1 = _mm256_loadu_si256((const __m256i *)(const void *)(vmin + i + 1));
        __m256i n2 = _mm256_loadu_si256((const __m256i *)(const void *)(vmin + i + 2));
        __m256i mx = _mm256_max_epu8(_mm256_max_epu8(m0, m1), m2);
        __m256i mn = _mm256_min_epu8(_mm256_min_epu8(n0, n1), n2);
        _mm256_storeu_si256((__m256i *)(void *)(out + i), _mm256_subs_epu8(mx, mn));
    }
#endif
    for (; i < mw; i++) {
        unsigned char mx = vmax[i], mn = vmin[i];
        if (vmax[i + 1] > mx) mx = vmax[i + 1];
        if (vmax[i + 2] > mx) mx = vmax[i + 2];
        if (vmin[i + 1] < mn) mn = vmin[i + 1];
        if (vmin[i + 2] < mn) mn = vmin[i + 2];
        out[i] = (unsigned char)(mx - mn);
    }
}

/* d = |g - bg| of mw pixels */
static void row_d(const unsigned char *src, int mw, int bg, unsigned char *out)
{
    int i = 0;
#if TSDP_AVX2
    __m256i vb = _mm256_set1_epi8((char)bg);
    for (; i + 32 <= mw; i += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)(src + i));
        _mm256_storeu_si256((__m256i *)(void *)(out + i),
                            _mm256_or_si256(_mm256_subs_epu8(v, vb), _mm256_subs_epu8(vb, v)));
    }
#endif
    for (; i < mw; i++) {
        int v = (int)src[i] - bg;
        out[i] = (unsigned char)(v < 0 ? -v : v);
    }
}

/* m0 = (d > t) & (c3 > tc) as 0 / 1 bytes of n pixels */
static void plane_mask(const unsigned char *d, const unsigned char *c3, unsigned char *m0, size_t n, int t, int tc)
{
    size_t i = 0;
#if TSDP_AVX2
    if (t < 255 && tc < 255) {
        __m256i vt = _mm256_set1_epi8((char)(t + 1)), vtc = _mm256_set1_epi8((char)(tc + 1));
        __m256i one = _mm256_set1_epi8(1);
        for (; i + 32 <= n; i += 32) {
            __m256i a = _mm256_loadu_si256((const __m256i *)(const void *)(d + i));
            __m256i b = _mm256_loadu_si256((const __m256i *)(const void *)(c3 + i));
            __m256i ga = _mm256_cmpeq_epi8(_mm256_max_epu8(a, vt), a);
            __m256i gb = _mm256_cmpeq_epi8(_mm256_max_epu8(b, vtc), b);
            _mm256_storeu_si256((__m256i *)(void *)(m0 + i), _mm256_and_si256(_mm256_and_si256(ga, gb), one));
        }
    }
#endif
    for (; i < n; i++) m0[i] = (unsigned char)(d[i] > t && c3[i] > tc);
}

/* m = m0 & (a 4-neighbour of m0 inside the mw x mh plane) */
static void plane_filter(const unsigned char *src, unsigned char *dst, int mw, int mh)
{
    int x, y;
    for (y = 0; y < mh; y++) {
        const unsigned char *r = src + (size_t)y * mw;
        const unsigned char *up = y > 0 ? r - mw : NULL, *dn = y + 1 < mh ? r + mw : NULL;
        unsigned char *o = dst + (size_t)y * mw;
        x = 0;
#if TSDP_AVX2
        if (mw >= 34) {
            __m256i z = _mm256_setzero_si256();
            o[0] = (unsigned char)(r[0] & ((up ? up[0] : 0) | (dn ? dn[0] : 0) | r[1]));
            for (x = 1; x + 32 <= mw - 1; x += 32) {
                __m256i c = _mm256_loadu_si256((const __m256i *)(const void *)(r + x));
                __m256i l = _mm256_loadu_si256((const __m256i *)(const void *)(r + x - 1));
                __m256i rr = _mm256_loadu_si256((const __m256i *)(const void *)(r + x + 1));
                __m256i u = up ? _mm256_loadu_si256((const __m256i *)(const void *)(up + x)) : z;
                __m256i dd = dn ? _mm256_loadu_si256((const __m256i *)(const void *)(dn + x)) : z;
                __m256i nb = _mm256_or_si256(_mm256_or_si256(l, rr), _mm256_or_si256(u, dd));
                _mm256_storeu_si256((__m256i *)(void *)(o + x), _mm256_and_si256(c, nb));
            }
        }
#endif
        for (; x < mw; x++) {
            unsigned char nb = 0;
            if (x > 0) nb |= r[x - 1];
            if (x + 1 < mw) nb |= r[x + 1];
            if (up) nb |= up[x];
            if (dn) nb |= dn[x];
            o[x] = (unsigned char)(r[x] & nb);
        }
    }
}

/* the histograms of the grey and of c3 over the pixel box [ix0, ix1) x [iy0, iy1) (c3: the frame's own, so these do
 * not depend on any window) */
static void box_hists(const unsigned char *g, unsigned W, unsigned H, unsigned stride, int ix0, int iy0, int ix1,
                      int iy1, tsdp_scr *s, unsigned *hg, unsigned *hc)
{
    const int w = ix1 - ix0;
    int x, y;
    memset(hg, 0, 256 * sizeof *hg);
    memset(hc, 0, 256 * sizeof *hc);
    for (y = iy0; y < iy1; y++) {
        const unsigned char *r = g + (size_t)y * stride + (size_t)ix0;
        row_c3(g, W, H, stride, ix0, w, y, s->vmax, s->vmin, s->c3);
        for (x = 0; x < w; x++) {
            hg[r[x]]++;
            hc[s->c3[x]]++;
        }
    }
}

/* t and tc of a window whose background is bg: max(floor, r(q99, k_pct)) of d = |g - bg| (from the box's grey
 * histogram) and of c3 over the box (n pixels) */
static void thresholds(const unsigned *hg, const unsigned *hc, unsigned long n, int bg, int *t, int *tc)
{
    unsigned hd[256];
    int v;
    memset(hd, 0, sizeof hd);
    for (v = 0; v < 256; v++) hd[v < bg ? bg - v : v - bg] += hg[v];
    *t = imax(28, r100(hist_quantile(hd, n, 99), 40));
    *tc = imax(28, r100(hist_quantile(hc, n, 99), 40));
}

/* the mask m (m0 with a 4-neighbour, inside the plane) of the columns [px0, px0 + pw) and rows [py0, py0 + ph): the
 * plane's own borders bound the neighbour rule, so a caller takes a plane one column (row) past what it reads where the
 * window goes on */
static void plane_ink(const unsigned char *g, unsigned W, unsigned H, unsigned stride, int px0, int py0, int pw, int ph,
                      int bg, int t, int tc, tsdp_scr *s)
{
    int y;
    for (y = 0; y < ph; y++) {
        row_c3(g, W, H, stride, px0, pw, py0 + y, s->vmax, s->vmin, s->c3 + (size_t)y * pw);
        row_d(g + (size_t)(py0 + y) * stride + (size_t)px0, pw, bg, s->d + (size_t)y * pw);
    }
    plane_mask(s->d, s->c3, s->m0, (size_t)pw * ph, t, tc);
    plane_filter(s->m0, s->m, pw, ph);
}

/* tools/tsdesk.ink_mask of the pixel box [ix0, ix1) x [iy0, iy1): the window is the box grown by max(2, r(h, 25)), its
 * background the lower median of the window's ring, the thresholds from the box; s->m holds the mask of the plane
 * [*px0, *px0 + *pw) x [*py0, *py0 + *ph): the box grown by one pixel inside the window (enough for the neighbour rule) */
static void ink_box(const unsigned char *g, unsigned W, unsigned H, unsigned stride, int ix0, int iy0, int ix1,
                    int iy1, tsdp_scr *s, int *pw_, int *bx0, int *by0)
{
    const int h = iy1 - iy0, A = imax(2, r100(h, 25));
    const int wx0 = imax(0, ix0 - A), wy0 = imax(0, iy0 - A), wx1 = imin((int)W, ix1 + A), wy1 = imin((int)H, iy1 + A);
    const int px0 = imax(wx0, ix0 - 1), py0 = imax(wy0, iy0 - 1), px1 = imin(wx1, ix1 + 1), py1 = imin(wy1, iy1 + 1);
    unsigned hg[256], hc[256];
    int t, tc, bg;
    bg = ring_median(g, stride, wx0, wy0, wx1 - wx0, wy1 - wy0);
    box_hists(g, W, H, stride, ix0, iy0, ix1, iy1, s, hg, hc);
    thresholds(hg, hc, (unsigned long)(ix1 - ix0) * (unsigned long)(iy1 - iy0), bg, &t, &tc);
    plane_ink(g, W, H, stride, px0, py0, px1 - px0, py1 - py0, bg, t, tc, s);
    *pw_ = px1 - px0;
    *bx0 = ix0 - px0;
    *by0 = iy0 - py0;
}

/* tools/tsdesk._pixel_box: floor / ceil of the edges, clipped to the frame */
static int pix_floor(double v, unsigned lim)
{
    double f = floor(v);
    return f < 0 ? 0 : (f > lim ? (int)lim : (int)f);
}

static int pix_ceil(double v, unsigned lim)
{
    double f = ceil(v);
    return f < 0 ? 0 : (f > lim ? (int)lim : (int)f);
}

/* ----------------------------------------------------------------------------- merge_pct */

static void merge_overlaps(vocr_tsdp_box *b, unsigned *n_, int pct)
{
    unsigned n = *n_;
    for (;;) {
        double best = -1.0;
        unsigned bi = 0, bj = 0, i, j;
        int found = 0;
        for (i = 0; i < n; i++) {
            const vocr_tsdp_box *a = &b[i];
            for (j = i + 1; j < n; j++) {
                const vocr_tsdp_box *c = &b[j];
                double dx, dy, across, size, small, share;
                if (a->vertical != c->vertical) continue;
                dx = dmin(a->x + a->w, c->x + c->w) - dmax(a->x, c->x);
                dy = dmin(a->y + a->h, c->y + c->h) - dmax(a->y, c->y);
                if (dx <= 0 || dy <= 0) continue;
                if (a->vertical) {
                    across = dx;
                    size = dmin(a->w, c->w);
                } else {
                    across = dy;
                    size = dmin(a->h, c->h);
                }
                if (across < 0.5 * size) continue;
                small = dmin(a->w * a->h, c->w * c->h);
                if (small <= 0 || dx * dy * 100 < (double)pct * small) continue;
                share = dx * dy / small;
                if (!found || share > best) {
                    best = share;
                    bi = i;
                    bj = j;
                    found = 1;
                }
            }
        }
        if (!found) break;
        {
            const vocr_tsdp_box a = b[bi], c = b[bj];
            const vocr_tsdp_box *big = a.w * a.h >= c.w * c.h ? &a : &c;
            const double x0 = dmin(a.x, c.x), y0 = dmin(a.y, c.y);
            const double x1 = dmax(a.x + a.w, c.x + c.w), y1 = dmax(a.y + a.h, c.y + c.h);
            vocr_tsdp_box u = *big;
            u.x = x0;
            u.y = y0;
            u.w = x1 - x0;
            u.h = y1 - y0;
            u.cells = a.cells + c.cells;
            b[bi] = u;
            memmove(&b[bj], &b[bj + 1], (n - bj - 1) * sizeof *b);
            n--;
        }
    }
    *n_ = n;
}

/* ----------------------------------------------------------------------------- rows_pct */

/* the lower median of the nonzero row counts rc[0 .. h) */
static int lower_median_pos(const int *rc, int h, int *sortbuf)
{
    int i, k = 0;
    for (i = 0; i < h; i++)
        if (rc[i] > 0) sortbuf[k++] = rc[i];
    /* insertion sort: a box is a few dozen rows */
    for (i = 1; i < k; i++) {
        int v = sortbuf[i], j = i;
        while (j > 0 && sortbuf[j - 1] > v) {
            sortbuf[j] = sortbuf[j - 1];
            j--;
        }
        sortbuf[j] = v;
    }
    return k ? sortbuf[(k - 1) / 2] : 0;
}

/* tools/tsdesk.split_rows of box b: its parts appended to out (at most cap - *k of them) */
static void split_rows(const unsigned char *g, unsigned W, unsigned H, unsigned stride, const vocr_tsdp_box *b,
                       int rows_pct, int valley_pct, int rows_len, int depth, tsdp_scr *s, vocr_tsdp_box *out,
                       unsigned *k, unsigned cap)
{
    const int ix0 = pix_floor(b->x, W), iy0 = pix_floor(b->y, H);
    const int ix1 = pix_ceil(b->x + b->w, W), iy1 = pix_ceil(b->y + b->h, H);
    const int h = iy1 - iy0, w = ix1 - ix0;
    int mw, bx0, by0, x, y, ninks = 0, med, low, lo = -1, hi = -1, min_band, min_len, found = 0, bv0 = 0, bv1 = 0;
    int best_mn = 0, best_mx = 1, best_len = 0;
    if (*k >= cap) return;
    if (b->vertical || h < 8 || w < 8 || depth >= 2) {
        out[(*k)++] = *b;
        return;
    }
    ink_box(g, W, H, stride, ix0, iy0, ix1, iy1, s, &mw, &bx0, &by0);
    for (y = 0; y < h; y++) {
        const unsigned char *r = s->m + (size_t)(by0 + y) * mw + bx0;
        int c = 0;
        for (x = 0; x < w; x++) c += r[x];
        s->rc[y] = c;
        if (c > 0) ninks++;
    }
    if (ninks < 2) {
        out[(*k)++] = *b;
        return;
    }
    med = lower_median_pos(s->rc, h, s->cc);
    low = r100(med, valley_pct);
    for (y = 0; y < h; y++)
        if (s->rc[y] > low) {
            if (lo < 0) lo = y;
            hi = y + 1;
        }
    {
        int ndense = 0;
        for (y = 0; y < h; y++) ndense += s->rc[y] > low;
        if (ndense < 2) {
            out[(*k)++] = *b;
            return;
        }
    }
    min_band = imax(4, r100(h, 20));
    min_len = r100(h, rows_len);
    y = lo;
    while (y < hi) {
        int v0, v1, ha, hb, mn, mx, ca0 = -1, ca1 = -1, cb0 = -1, cb1 = -1, yy;
        if (s->rc[y] > low) {
            y++;
            continue;
        }
        v0 = y;
        while (y < hi && s->rc[y] <= low) y++;
        v1 = y;
        ha = v0 - lo;
        hb = hi - v1;
        mn = imin(ha, hb);
        mx = imax(ha, hb);
        if (mn < min_band || mn * 100 < rows_pct * mx) continue;
        for (x = 0; x < w; x++) {
            int a = 0, c = 0;
            for (yy = lo; yy < v0 && !a; yy++) a = s->m[(size_t)(by0 + yy) * mw + bx0 + x];
            for (yy = v1; yy < hi && !c; yy++) c = s->m[(size_t)(by0 + yy) * mw + bx0 + x];
            if (a) {
                if (ca0 < 0) ca0 = x;
                ca1 = x;
            }
            if (c) {
                if (cb0 < 0) cb0 = x;
                cb1 = x;
            }
        }
        if (ca0 < 0 || cb0 < 0 || ca1 - ca0 + 1 < min_len || cb1 - cb0 + 1 < min_len) continue;
        /* key (min / max, valley length), compared as Python tuples */
        {
            const double r_new = (double)mn / (double)mx, r_best = (double)best_mn / (double)best_mx;
            if (!found || r_new > r_best || (r_new == r_best && v1 - v0 > best_len)) {
                found = 1;
                best_mn = mn;
                best_mx = mx;
                best_len = v1 - v0;
                bv0 = v0;
                bv1 = v1;
            }
        }
    }
    if (!found) {
        out[(*k)++] = *b;
        return;
    }
    {
        vocr_tsdp_box top = *b, bot = *b;
        top.h = (double)(iy0 + bv0) - b->y;
        bot.y = (double)(iy0 + bv1);
        bot.h = b->y + b->h - (double)(iy0 + bv1);
        split_rows(g, W, H, stride, &top, rows_pct, valley_pct, rows_len, depth + 1, s, out, k, cap);
        split_rows(g, W, H, stride, &bot, rows_pct, valley_pct, rows_len, depth + 1, s, out, k, cap);
    }
}

/* ----------------------------------------------------------------------------- vflip_pct */

/* insertion sort of a few ints (the bands of a box) */
static void isort(int *v, int n)
{
    int i;
    for (i = 1; i < n; i++) {
        int x = v[i], j = i;
        while (j > 0 && v[j - 1] > x) {
            v[j] = v[j - 1];
            j--;
        }
        v[j] = x;
    }
}

/* tools/tsdesk.vflip on boxes[0 .. *n_): the boxes left as they were stay first (in order), the pieces of the turned
 * ones (joined) follow, at most cap in all */
static void vflip(const unsigned char *g, unsigned W, unsigned H, unsigned stride, vocr_tsdp_box *boxes,
                  unsigned *n_, unsigned cap, const vocr_tsdp_opts *o, tsdp_scr *s)
{
    vocr_tsdp_box *pc = s->tmp;
    unsigned n = *n_, i, nout = 0, np = 0, room;
    /* first the kept boxes, compacted in place; the turned boxes' pieces go to s->tmp in box order. A piece is made
     * only while kept + pieces stay under cap (tools/tsdesk.vflip caps the same way) */
    for (i = 0; i < n; i++) {
        const vocr_tsdp_box b = boxes[i];
        const int ix0 = pix_floor(b.x, W), iy0 = pix_floor(b.y, H);
        const int ix1 = pix_ceil(b.x + b.w, W), iy1 = pix_ceil(b.y + b.h, H);
        const int bw = ix1 - ix0, bh = iy1 - iy0;
        int mw, bx0, by0, x, y, nb = 0, k, hm, gm, minb;
        int *ba = s->cc, *be;                /* band starts and ends in cc (2 (bh / 2 + 1) <= H + W ints); the row
                                                counts in rc, which then sorts the band heights and gaps */
        if (!b.vertical || bw < 4 || bh < 8) {
            boxes[nout++] = b;
            continue;
        }
        ink_box(g, W, H, stride, ix0, iy0, ix1, iy1, s, &mw, &bx0, &by0);
        for (y = 0; y < bh; y++) {
            const unsigned char *r = s->m + (size_t)(by0 + y) * mw + bx0;
            int c = 0;
            for (x = 0; x < bw; x++) c += r[x];
            s->rc[y] = c;
        }
        minb = imax(2, r100(bw, 15));
        be = ba + (bh / 2 + 1);
        y = 0;
        while (y < bh) {
            int y0;
            if (!s->rc[y]) {
                y++;
                continue;
            }
            y0 = y;
            while (y < bh && s->rc[y]) y++;
            if (y - y0 >= minb) {
                ba[nb] = y0;
                be[nb] = y;
                nb++;
            }
        }
        if (nb < imax(2, o->vflip_min)) {
            boxes[nout++] = b;
            continue;
        }
        for (k = 0; k < nb; k++) s->rc[k] = be[k] - ba[k];
        isort(s->rc, nb);
        hm = s->rc[(nb - 1) / 2];
        for (k = 0; k + 1 < nb; k++) s->rc[k] = ba[k + 1] - be[k];
        isort(s->rc, nb - 1);
        gm = s->rc[(nb - 2) / 2];
        if ((long)gm * 100 < (long)o->vflip_pct * hm || (long)hm * 100 < (long)o->vflip_h_pct * bw) {
            boxes[nout++] = b;
            continue;
        }
        for (k = 0; k < nb; k++) {
            int c0 = -1, c1 = -1;
            for (x = 0; x < bw; x++) {
                int any = 0;
                for (y = ba[k]; y < be[k] && !any; y++) any = s->m[(size_t)(by0 + y) * mw + bx0 + x];
                if (any) {
                    if (c0 < 0) c0 = x;
                    c1 = x;
                }
            }
            if (np < cap) {
                vocr_tsdp_box q = b;
                q.x = (double)(ix0 + c0);
                q.y = (double)(iy0 + ba[k]);
                q.w = (double)(c1 - c0 + 1);
                q.h = (double)(be[k] - ba[k]);
                q.vertical = 0;
                q.cells = 1;
                pc[np++] = q;
            }
        }
    }
    room = cap > nout ? cap - nout : 0;
    if (np > room) np = room;
    for (;;) {                              /* the joins: the first pair in order, until none */
        int changed = 0;
        unsigned a, c;
        for (a = 0; a < np && !changed; a++) {
            for (c = a + 1; c < np; c++) {
                const vocr_tsdp_box A = pc[a], C = pc[c];
                const vocr_tsdp_box *big;
                vocr_tsdp_box u;
                double x0, y0, x1, y1;
                if (dmin(A.y + A.h, C.y + C.h) - dmax(A.y, C.y) < 0.5 * dmin(A.h, C.h)) continue;
                if (dmax(C.x - (A.x + A.w), A.x - (C.x + C.w)) * 100 > (double)o->vflip_join_pct * dmax(A.h, C.h))
                    continue;
                big = A.w * A.h >= C.w * C.h ? &A : &C;
                x0 = dmin(A.x, C.x);
                y0 = dmin(A.y, C.y);
                x1 = dmax(A.x + A.w, C.x + C.w);
                y1 = dmax(A.y + A.h, C.y + C.h);
                u = *big;
                u.x = x0;
                u.y = y0;
                u.w = x1 - x0;
                u.h = y1 - y0;
                u.cells = A.cells + C.cells;
                pc[a] = u;
                memmove(&pc[c], &pc[c + 1], (np - c - 1) * sizeof *pc);
                np--;
                changed = 1;
                break;
            }
        }
        if (!changed) break;
    }
    memcpy(boxes + nout, pc, np * sizeof *pc);
    *n_ = nout + np;
}

int vocr_tsdp_pre(const unsigned char *grey, unsigned w, unsigned h, unsigned stride, vocr_tsdp_box *boxes,
                  unsigned *n, unsigned cap, const vocr_tsdp_opts *o, void *scratch, size_t bytes)
{
    tsdp_scr s;
    unsigned i, k = 0, m;
    if (!n || !o || (*n && !boxes) || *n > cap) return VOCR_TSDP_E_ARG;
    if (o->gap_pct) return VOCR_TSDP_E_ARG;
    if (!o->vflip_pct && !o->merge_pct && !o->rows_pct) return VOCR_TSDP_OK;
    if (!grey || !w || !h || stride < w || !scratch) return VOCR_TSDP_E_ARG;
    if (bytes < vocr_tsdp_scratch_bytes(w, h, cap)) return VOCR_TSDP_E_SCRATCH;
    scr_map(&s, scratch, w, h, cap);
    if (o->vflip_pct) vflip(grey, w, h, stride, boxes, n, cap, o, &s);
    if (o->merge_pct) merge_overlaps(boxes, n, o->merge_pct);
    if (o->rows_pct) {
        m = *n;
        for (i = 0; i < m && k < cap; i++)
            split_rows(grey, w, h, stride, &boxes[i], o->rows_pct, o->valley_pct, o->rows_len, 0, &s, s.tmp, &k, cap);
        memcpy(boxes, s.tmp, k * sizeof *boxes);
        *n = k;
    }
    return VOCR_TSDP_OK;
}

/* ----------------------------------------------------------------------------- ext_pct */

static int finite4(double a, double b, double c, double d)
{
    return isfinite(a) && isfinite(b) && isfinite(c) && isfinite(d);
}

/* tools/tsdesk_refine.neighbour_limits for box i of the n boxes (vertical ones too) */
static void neighbour_limits(const vocr_tsdp_box *bx, unsigned n, unsigned i, unsigned W, unsigned H, int *out)
{
    const double a0 = bx[i].x, a1 = bx[i].y, a2 = bx[i].x + bx[i].w, a3 = bx[i].y + bx[i].h;
    int ix0, iy0, ix1, iy1, nl = 0, nt = 0, nr = (int)W, nb = (int)H;
    unsigned j;
    out[0] = 0;
    out[1] = 0;
    out[2] = (int)W;
    out[3] = (int)H;
    if (!finite4(a0, a1, a2, a3)) return;
    ix0 = pix_floor(a0, W);
    iy0 = pix_floor(a1, H);
    ix1 = pix_ceil(a2, W);
    iy1 = pix_ceil(a3, H);
    for (j = 0; j < n; j++) {
        const double b0 = bx[j].x, b1 = bx[j].y, b2 = bx[j].x + bx[j].w, b3 = bx[j].y + bx[j].h;
        double oy, ox, ha, hb, wa, wb;
        if (j == i || !finite4(b0, b1, b2, b3)) continue;
        oy = (a3 < b3 ? a3 : b3) - (a1 > b1 ? a1 : b1);
        ox = (a2 < b2 ? a2 : b2) - (a0 > b0 ? a0 : b0);
        ha = a3 - a1;
        hb = b3 - b1;
        wa = a2 - a0;
        wb = b2 - b0;
        if (oy >= 0.5 * (ha < hb ? ha : hb)) {
            if (b0 > a0) nr = imin(nr, imax(ix1, pix_floor(b0, W)));
            if (b2 < a2) nl = imax(nl, imin(ix0, pix_ceil(b2, W)));
        }
        if (ox >= 0.5 * (wa < wb ? wa : wb)) {
            if (b1 < a1) nt = imax(nt, imin(iy0, pix_ceil(b3, H)));
            if (b3 > a3) nb = imin(nb, imax(iy1, pix_floor(b1, H)));
        }
    }
    out[0] = nl;
    out[1] = nt;
    out[2] = nr;
    out[3] = nb;
}

/* tools/tsdesk._ink_beyond for a search window [wx0, wx1): the union window [ux0, ux1) x [uy0, uy1) of the line's
 * pixel box and the search window gives the background (its ring's lower median), the line's box the thresholds (its
 * histograms hg, hc, n pixels); the mask is the union window's, read over the search window's columns, so the plane
 * is those columns and the one beside them on either side that lies in the union window (the neighbour rule) */
static void ink_beyond(const unsigned char *g, unsigned W, unsigned H, unsigned stride, int iy0, int iy1, int bx0,
                       int bx1, int wx0, int wx1, const unsigned *hg, const unsigned *hc, unsigned long nbox,
                       tsdp_scr *s, int *pw_, int *off_, int *nr_, int *ry0_)
{
    const int h = iy1 - iy0, A = imax(2, r100(h, 25));
    const int ux0 = imax(0, imin(bx0, wx0) - 1), ux1 = imin((int)W, imax(bx1, wx1) + 1);
    const int uy0 = imax(0, iy0 - A), uy1 = imin((int)H, iy1 + A);
    const int px0 = imax(ux0, wx0 - 1), px1 = imin(ux1, wx1 + 1);
    int t, tc, bg;
    (void)h;
    bg = ring_median(g, stride, ux0, uy0, ux1 - ux0, uy1 - uy0);
    thresholds(hg, hc, nbox, bg, &t, &tc);
    plane_ink(g, W, H, stride, px0, uy0, px1 - px0, uy1 - uy0, bg, t, tc, s);
    *pw_ = px1 - px0;
    *off_ = wx0 - px0;
    *nr_ = uy1 - uy0;
    *ry0_ = iy0 - uy0;
}

/* the column at window column `wc` (an index into the search window [wx0, wx1)) of the line's rows: any ink / ink in
 * the middle rows [cy0, cy1) only */
typedef struct {
    const unsigned char *m;
    int uw, off, r0, h, cy0, cy1;     /* off: the search window's first column in the union window */
} tsdp_cols;

static int col_any(const tsdp_cols *c, int wc)
{
    int y;
    for (y = 0; y < c->h; y++)
        if (c->m[(size_t)(c->r0 + y) * c->uw + c->off + wc]) return 1;
    return 0;
}

static int col_dash(const tsdp_cols *c, int wc)
{
    int y, mid = 0;
    for (y = 0; y < c->h; y++) {
        if (!c->m[(size_t)(c->r0 + y) * c->uw + c->off + wc]) continue;
        if (y < c->cy0 || y >= c->cy1) return 0;
        mid = 1;
    }
    return mid;
}

/* tools/tsdesk._walk_dash over the n columns cols[] (window columns in outward order): the index into cols of the new
 * edge or -1, *ran_out set when the walk reached the window's end on the run */
static int walk_dash(const tsdp_cols *c, const int *cols, int n, int G, int Gr, int small, int low_small, int D,
                     int *ran_out)
{
    const int mid = c->h / 2;
    int i = 0, gap = 0, dash = 0, last = -1;
    *ran_out = 0;
    while (i < imin(n, small) && col_any(c, cols[i]) && !col_dash(c, cols[i])) i++;
    while (i < n) {
        int j, y, first_row = -1;
        if (!col_any(c, cols[i])) {
            gap++;
            if (gap > (dash ? Gr : G)) return dash >= D ? last : -1;
            i++;
            continue;
        }
        gap = 0;
        if (col_dash(c, cols[i])) {
            dash++;
            last = i;
            i++;
            continue;
        }
        j = i;
        while (j < n && col_any(c, cols[j])) j++;
        if (dash >= D) {
            if (j - i <= small) last = j - 1;
            return last;
        }
        for (y = 0; y < c->h && first_row < 0; y++) {
            int q;
            for (q = i; q < j; q++)
                if (c->m[(size_t)(c->r0 + y) * c->uw + c->off + cols[q]]) {
                    first_row = y;
                    break;
                }
        }
        if (!dash && j - i <= low_small && first_row >= 0 && first_row >= mid) {
            i = j;
            continue;
        }
        return dash >= D ? last : -1;
    }
    *ran_out = dash >= D && gap == 0;
    return dash >= D ? last : -1;
}

int vocr_tsdp_post(const unsigned char *grey, unsigned w, unsigned h, unsigned stride, vocr_tsdp_box *boxes,
                   unsigned *n_, const vocr_tsdp_opts *o, void *scratch, size_t bytes)
{
    tsdp_scr s;
    unsigned i, j, n;
    int any_reach = 0;
    if (!n_ || !o || (*n_ && !boxes)) return VOCR_TSDP_E_ARG;
    n = *n_;
    if (!o->ext_pct || !n) return VOCR_TSDP_OK;
    if (!grey || !w || !h || stride < w || !scratch) return VOCR_TSDP_E_ARG;
    if (bytes < vocr_tsdp_scratch_bytes(w, h, n)) return VOCR_TSDP_E_SCRATCH;
    scr_map(&s, scratch, w, h, n);
    for (i = 0; i < n; i++) neighbour_limits(boxes, n, i, w, h, s.lims + 4 * (size_t)i);
    for (i = 0; i < n; i++) {
        vocr_tsdp_box *b = &boxes[i];
        const int ix0 = pix_floor(b->x, w), iy0 = pix_floor(b->y, h);
        const int ix1 = pix_ceil(b->x + b->w, w), iy1 = pix_ceil(b->y + b->h, h);
        const int bh = iy1 - iy0, *lim = s.lims + 4 * (size_t)i;
        double x0 = b->x, x1 = b->x + b->w;
        int side, L, G, Gr, D, q, have_hists = 0;
        unsigned hg[256], hc[256];
        s.reach[2 * i] = s.reach[2 * i + 1] = 0;
        if (b->vertical || bh < 6 || ix1 - ix0 < 4) continue;
        L = r100(bh, o->ext_pct);
        G = imax(2, r100(bh, o->ext_gap_pct));
        Gr = imax(2, r100(bh, o->ext_run_pct));
        D = imax(2, r100(bh, o->ext_dash_pct));
        q = r100(bh, 25);
        for (side = -1; side <= 1; side += 2) {
            int wx0, wx1, pw, off, nr, ry0, k, ran_out, c, nwc, limit, colbuf_ok = 1;
            tsdp_cols cl;
            if (side < 0) {
                wx0 = imax(imax(0, lim[0]), ix0 - L);
                wx1 = ix0;
            } else {
                wx0 = ix1;
                wx1 = imin(imin((int)w, lim[2]), ix1 + L);
            }
            if (wx1 - wx0 < D) continue;
            if (!have_hists) {      /* the line box's histograms: the same for both sides */
                box_hists(grey, w, h, stride, ix0, iy0, ix1, iy1, &s, hg, hc);
                have_hists = 1;
            }
            ink_beyond(grey, w, h, stride, iy0, iy1, ix0, ix1, wx0, wx1, hg, hc,
                       (unsigned long)(ix1 - ix0) * (unsigned long)bh, &s, &pw, &off, &nr, &ry0);
            nwc = wx1 - wx0;
            cl.m = s.m;
            cl.uw = pw;
            cl.off = off;
            cl.r0 = ry0;
            cl.h = bh;
            cl.cy0 = q;
            cl.cy1 = bh - q;
            /* the columns in outward order, in s.rc / s.cc (int buffers of H and W entries: nwc <= W) */
            {
                int *cols = s.cc, t;
                for (t = 0; t < nwc; t++) cols[t] = side < 0 ? nwc - 1 - t : t;
                k = walk_dash(&cl, cols, nwc, G, Gr, r100(bh, 80), r100(bh, 60), D, &ran_out);
                limit = side < 0 ? (wx0 > 0 && wx0 == ix0 - L) : (wx1 < (int)w && wx1 == ix1 + L);
                if (k < 0 || (ran_out && limit)) colbuf_ok = 0;
                c = colbuf_ok ? cols[k] : 0;
                if (colbuf_ok) {
                    /* the rows just above and below the line over the span from the edge to the new end */
                    const int a = imin(c, cols[0]), z = imax(c, cols[0]), span = z - a + 1;
                    if (nr > ry0 + bh || ry0 > 0) {
                        int outer_rows = ry0 + (nr - (ry0 + bh)), cnt = 0, xx, yy;
                        if (outer_rows > 0) {
                            for (xx = a; xx <= z; xx++) {
                                int ink = 0;
                                for (yy = 0; yy < ry0 && !ink; yy++) ink = s.m[(size_t)yy * pw + cl.off + xx];
                                for (yy = ry0 + bh; yy < nr && !ink; yy++) ink = s.m[(size_t)yy * pw + cl.off + xx];
                                cnt += ink;
                            }
                            if (cnt * 10 > span) colbuf_ok = 0;
                        }
                    }
                }
            }
            if (!colbuf_ok) continue;
            if (side < 0) {
                x0 = (double)(wx0 + c);
                s.reach[2 * i] = (unsigned char)(wx0 > 0 && wx0 == lim[0] && c <= 2);
            } else {
                x1 = (double)(wx0 + c + 1);
                s.reach[2 * i + 1] = (unsigned char)(wx1 < (int)w && wx1 == lim[2] && c >= wx1 - wx0 - 3);
            }
        }
        if (x0 != b->x || x1 != b->x + b->w) {
            b->x = x0;
            b->w = x1 - x0;
        }
        any_reach |= s.reach[2 * i] | s.reach[2 * i + 1];
    }
    if (!o->ext_join || !any_reach) return VOCR_TSDP_OK;
    for (i = 0; i < n; i++) s.alive[i] = 1;
    for (i = 0; i < n; i++) {
        const int a = s.reach[2 * i], z = s.reach[2 * i + 1];
        vocr_tsdp_box bi;
        if (!(a || z) || !s.alive[i]) continue;
        bi = boxes[i];
        for (j = 0; j < n; j++) {
            const vocr_tsdp_box *bj = &boxes[j];
            double dy;
            int touch_r, touch_l;
            if (j == i || !s.alive[j] || bj->vertical) continue;
            dy = dmin(bi.y + bi.h, bj->y + bj->h) - dmax(bi.y, bj->y);
            if (dy < 0.5 * dmin(bi.h, bj->h)) continue;
            touch_r = z && bj->x >= bi.x && bj->x - (bi.x + bi.w) <= 2;
            touch_l = a && bj->x + bj->w <= bi.x + bi.w && bi.x - (bj->x + bj->w) <= 2;
            if (touch_r || touch_l) {
                const double x0 = dmin(bi.x, bj->x), y0 = dmin(bi.y, bj->y);
                const double x1 = dmax(bi.x + bi.w, bj->x + bj->w), y1 = dmax(bi.y + bi.h, bj->y + bj->h);
                vocr_tsdp_box u = bi.w >= bj->w ? bi : *bj;
                u.x = x0;
                u.y = y0;
                u.w = x1 - x0;
                u.h = y1 - y0;
                u.cells = bi.cells + bj->cells;
                bi = u;
                boxes[i] = bi;
                s.alive[j] = 0;
            }
        }
    }
    for (i = 0, j = 0; i < n; i++)
        if (s.alive[i]) boxes[j++] = boxes[i];
    *n_ = j;
    return VOCR_TSDP_OK;
}

/* ----------------------------------------------------------------------------- the reading filter */

/* the next code point of a NUL-terminated UTF-8 string (0 at its end, -1 for bad UTF-8); *p advances */
static long utf8_next(const unsigned char **p)
{
    const unsigned char *s = *p;
    long c;
    int k, i;
    if (!*s) return 0;
    if (s[0] < 0x80) {
        *p = s + 1;
        return s[0];
    }
    if ((s[0] & 0xE0) == 0xC0) {
        c = s[0] & 0x1F;
        k = 1;
    } else if ((s[0] & 0xF0) == 0xE0) {
        c = s[0] & 0x0F;
        k = 2;
    } else if ((s[0] & 0xF8) == 0xF0) {
        c = s[0] & 0x07;
        k = 3;
    } else {
        return -1;
    }
    for (i = 1; i <= k; i++) {
        if ((s[i] & 0xC0) != 0x80) return -1;
        c = (c << 6) | (s[i] & 0x3F);
    }
    *p = s + k + 1;
    return c;
}

/* tools/tsdesk.JAPANESE */
static int is_jp(long c)
{
    return (c >= 0x3000 && c <= 0x303F) || (c >= 0x3040 && c <= 0x30FF) || (c >= 0x3400 && c <= 0x4DBF) ||
           (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0xF900 && c <= 0xFAFF) || (c >= 0xFF01 && c <= 0xFF60) ||
           c == 0x2025 || c == 0x2026 || c == 0x2015 || c == 0x2014;
}

/* a reading's code points: count, Japanese ones, and the most frequent code point (ties: the one seen first, as
 * Counter.most_common(1)) with its count; -1 for bad UTF-8 */
typedef struct {
    long n, jp, top_cp, top_n;
} reading_stats;

#define MAX_CP 1024

static int stats_of(const char *text, reading_stats *r)
{
    long cps[MAX_CP];
    long cnt[MAX_CP];
    long ndist = 0, i;
    const unsigned char *p = (const unsigned char *)text;
    memset(r, 0, sizeof *r);
    r->top_cp = -1;
    for (;;) {
        long c = utf8_next(&p);
        if (c == 0) break;
        if (c < 0) return -1;
        r->n++;
        r->jp += is_jp(c);
        for (i = 0; i < ndist; i++)
            if (cps[i] == c) break;
        if (i == ndist) {
            if (ndist == MAX_CP) return -1;
            cps[ndist] = c;
            cnt[ndist] = 0;
            ndist++;
        }
        cnt[i]++;
    }
    for (i = 0; i < ndist; i++)
        if (cnt[i] > r->top_n) {
            r->top_n = cnt[i];
            r->top_cp = cps[i];
        }
    return 0;
}

int vocr_tsdd_has_japanese(const char *text)
{
    reading_stats r;
    if (!text || stats_of(text, &r)) return -1;
    return r.jp > 0;
}

double vocr_tsdd_japanese_share(const char *text)
{
    reading_stats r;
    if (!text || stats_of(text, &r)) return -1.0;
    return r.n ? (double)r.jp / (double)r.n : 0.0;
}

int vocr_tsdd_drop(const double *xywh, const char *const *text, const float *conf_mean, unsigned n,
                   const vocr_tsdd_opts *o, unsigned char *drop)
{
    unsigned i, j;
    int dropped = 0;
    if (!o || (n && (!xywh || !text || !drop)) || (o->conf_milli && n && !conf_mean)) return VOCR_TSDP_E_ARG;
    for (i = 0; i < n; i++) {
        reading_stats r;
        drop[i] = 0;
        if (!text[i] || stats_of(text[i], &r)) return VOCR_TSDP_E_ARG;
    }
    for (i = 0; i < n; i++) {
        reading_stats r;
        const double x = xywh[4 * i], y = xywh[4 * i + 1], w = xywh[4 * i + 2], h = xywh[4 * i + 3];
        stats_of(text[i], &r);
        if (o->conf_milli && (double)conf_mean[i] * 1000.0 < (double)o->conf_milli) {
            drop[i] = 1;
            continue;
        }
        if (o->rep_pct && r.n >= (o->rep_len > 1 ? o->rep_len : 1) && !is_jp(r.top_cp) &&
            r.top_n * 100 >= (long)o->rep_pct * r.n) {
            drop[i] = 1;
            continue;
        }
        if (!o->nojp || r.jp) continue;
        if (o->nojp >= 2 && r.n) {
            int keep = 0;
            for (j = 0; j < n && !keep; j++) {
                reading_stats q;
                double x2, y2, w2, h2, gap;
                if (j == i) continue;
                stats_of(text[j], &q);
                if (!q.jp || (double)q.jp / (double)q.n * 100.0 < (double)o->nojp_anchor_pct) continue;
                x2 = xywh[4 * j];
                y2 = xywh[4 * j + 1];
                w2 = xywh[4 * j + 2];
                h2 = xywh[4 * j + 3];
                if (dmin(y + h, y2 + h2) - dmax(y, y2) < 0.5 * dmin(h, h2)) continue;
                gap = dmax(x2 - (x + w), x - (x2 + w2));
                if (gap * 100 <= (double)o->nojp_gap_pct * dmax(h, h2)) keep = 1;
            }
            if (keep) continue;
        }
        drop[i] = 1;
    }
    for (i = 0; i < n; i++) dropped += drop[i];
    return dropped;
}
