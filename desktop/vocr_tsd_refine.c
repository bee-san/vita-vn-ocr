/* vocr_tsd_refine.c - textspot-desktop's box refinement (vocr_tsd_refine.h; the rule: tools/tsdesk_refine.py).
 *
 * Every step mirrors tools/tsdesk_refine.refine_box line for line in integers, so the boxes are the same bits. The
 * per-box work runs in the caller's scratch: the window's grey difference d and its 3 x 3 local contrast c3 (one
 * byte a pixel each), then the ink mask written over them, a few histograms and the row / column counts. The window
 * passes (c3, d, the mask, its 4-neighbour filter, the counts) have AVX2 versions; the histograms and the walks are
 * scalar in both builds.
 */
#include "vocr_tsd_refine.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX2__) && !defined(VOCR_TSDR_SCALAR)
#include <immintrin.h>
#define TSDR_AVX2 1
#else
#define TSDR_AVX2 0
#endif

typedef char tsdr_opts_are_ints[(sizeof(vocr_tsdr_opts) == VOCR_TSDR_NOPTS * sizeof(int)) ? 1 : -1];

void vocr_tsdr_defaults(vocr_tsdr_opts *o)
{
    o->min_h = 4; o->min_w = 4;
    o->band_pct = 25; o->band_min = 2;
    o->start_pct = 180; o->end_pct = 400; o->along_min = 3;
    o->core_pct = 25; o->along_core_pct = 50;
    o->row_gap = 0; o->valley_pct = 8; o->low_run_pct = 20; o->rise_pct = 50; o->beyond_pct = 15;
    o->gap_pct = 70; o->gap_min = 2;
    o->bar_pct = 15;
    o->shrink_a_pct = 10; o->shrink_l_pct = 25;
    o->floor_ = 28; o->k_pct = 40;
    o->margin_pct = 90; o->margin_gap_pct = 50; o->margin_min = 3; o->noise_add = 12; o->texture_pct = 0;
    o->busy_pct = 25;
}

int vocr_tsdr_nopts(void) { return VOCR_TSDR_NOPTS; }

int vocr_tsdr_simd(void) { return TSDR_AVX2; }

static int r100(long v, int pct) { return (int)((v * (long)pct + 50) / 100); }
static int imax(int a, int b) { return a > b ? a : b; }
static int imin(int a, int b) { return a < b ? a : b; }

/* scratch layout */
typedef struct {
    unsigned char *d, *c3;            /* window planes (mw x mh), then the masks over them */
    unsigned char *vmax, *vmin;       /* one row (W + 2) */
    int *rc, *cc, *sortbuf;           /* H, W, H */
    unsigned char *barrier;           /* W */
} tsdr_scr;

size_t vocr_tsdr_scratch_bytes(unsigned w, unsigned h)
{
    size_t W = w, H = h;
    return 2 * W * H + 2 * (W + 2) + (H + W + H) * sizeof(int) + W + 64;
}

static void scr_map(tsdr_scr *s, void *buf, unsigned w, unsigned h)
{
    unsigned char *p = (unsigned char *)buf;
    size_t W = w, H = h;
    s->rc = (int *)p; p += H * sizeof(int);
    s->cc = (int *)p; p += W * sizeof(int);
    s->sortbuf = (int *)p; p += H * sizeof(int);
    s->d = p; p += W * H;
    s->c3 = p; p += W * H;
    s->vmax = p; p += W + 2;
    s->vmin = p; p += W + 2;
    s->barrier = p;
}

/* the element ceil(pct n / 100) - 1 (clipped) of the values counted in hist[256] (n of them) */
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

static void hist_rect(unsigned *hist, const unsigned char *plane, int mw, int x0, int y0, int x1, int y1)
{
    int x, y;
    for (y = y0; y < y1; y++) {
        const unsigned char *r = plane + (size_t)y * mw;
        for (x = x0; x < x1; x++) hist[r[x]]++;
    }
}

static int cmp_int(const void *a, const void *b)
{
    int x = *(const int *)a, y = *(const int *)b;
    return (x > y) - (x < y);
}

/* ---- window passes ---- */

/* c3 (3 x 3 max - min, the frame's edge pixels repeated) and d = |g - bg| of one window row */
static void row_c3(const unsigned char *g, unsigned W, unsigned H, unsigned stride, int wx0, int mw, int y,
                   unsigned char *vmax, unsigned char *vmin, unsigned char *out)
{
    int ya = y > 0 ? y - 1 : 0, yc = y + 1 < (int)H ? y + 1 : (int)H - 1;
    const unsigned char *a = g + (size_t)ya * stride, *b = g + (size_t)y * stride, *c = g + (size_t)yc * stride;
    int i, n = mw + 2;
    /* vmax / vmin over columns wx0 - 1 .. wx0 + mw (clamped), index i = column - (wx0 - 1) */
    int lo = 0, hi = n;     /* the part whose columns lie inside the frame without clamping */
    if (wx0 - 1 < 0) lo = 1;
    if (wx0 + mw > (int)W - 1) hi = n - 1;
#if TSDR_AVX2
    i = lo;
    for (; i + 32 <= hi; i += 32) {
        int x = wx0 - 1 + i;
        __m256i va = _mm256_loadu_si256((const __m256i *)(a + x));
        __m256i vb = _mm256_loadu_si256((const __m256i *)(b + x));
        __m256i vc = _mm256_loadu_si256((const __m256i *)(c + x));
        _mm256_storeu_si256((__m256i *)(vmax + i), _mm256_max_epu8(_mm256_max_epu8(va, vb), vc));
        _mm256_storeu_si256((__m256i *)(vmin + i), _mm256_min_epu8(_mm256_min_epu8(va, vb), vc));
    }
#else
    i = lo;
#endif
    for (; i < hi; i++) {
        int x = wx0 - 1 + i;
        unsigned char p = a[x], q = b[x], r = c[x];
        unsigned char mx = p > q ? p : q, mn = p < q ? p : q;
        vmax[i] = mx > r ? mx : r;
        vmin[i] = mn < r ? mn : r;
    }
    if (lo == 1) { vmax[0] = vmax[1]; vmin[0] = vmin[1]; }            /* column -1 repeats column 0 */
    if (hi == n - 1) { vmax[n - 1] = vmax[n - 2]; vmin[n - 1] = vmin[n - 2]; }   /* column W repeats W - 1 */
    i = 0;
#if TSDR_AVX2
    for (; i + 32 <= mw; i += 32) {
        __m256i m0 = _mm256_loadu_si256((const __m256i *)(vmax + i));
        __m256i m1 = _mm256_loadu_si256((const __m256i *)(vmax + i + 1));
        __m256i m2 = _mm256_loadu_si256((const __m256i *)(vmax + i + 2));
        __m256i n0 = _mm256_loadu_si256((const __m256i *)(vmin + i));
        __m256i n1 = _mm256_loadu_si256((const __m256i *)(vmin + i + 1));
        __m256i n2 = _mm256_loadu_si256((const __m256i *)(vmin + i + 2));
        __m256i mx = _mm256_max_epu8(_mm256_max_epu8(m0, m1), m2);
        __m256i mn = _mm256_min_epu8(_mm256_min_epu8(n0, n1), n2);
        _mm256_storeu_si256((__m256i *)(out + i), _mm256_subs_epu8(mx, mn));
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

static void row_d(const unsigned char *src, int mw, int bg, unsigned char *out)
{
    int i = 0;
#if TSDR_AVX2
    __m256i vb = _mm256_set1_epi8((char)bg);
    for (; i + 32 <= mw; i += 32) {
        __m256i v = _mm256_loadu_si256((const __m256i *)(src + i));
        _mm256_storeu_si256((__m256i *)(out + i), _mm256_or_si256(_mm256_subs_epu8(v, vb), _mm256_subs_epu8(vb, v)));
    }
#endif
    for (; i < mw; i++) {
        int v = (int)src[i] - bg;
        out[i] = (unsigned char)(v < 0 ? -v : v);
    }
}

/* m0 = (d > t) & (c3 > tc) as 0 / 1 bytes, written over d */
static void plane_mask(unsigned char *d, const unsigned char *c3, size_t n, int t, int tc)
{
    size_t i = 0;
#if TSDR_AVX2
    /* unsigned x > t  <=>  max(x, t + 1) == x, for t < 255 */
    if (t < 255 && tc < 255) {
        __m256i vt = _mm256_set1_epi8((char)(t + 1)), vtc = _mm256_set1_epi8((char)(tc + 1));
        __m256i one = _mm256_set1_epi8(1);
        for (; i + 32 <= n; i += 32) {
            __m256i a = _mm256_loadu_si256((const __m256i *)(d + i));
            __m256i b = _mm256_loadu_si256((const __m256i *)(c3 + i));
            __m256i ga = _mm256_cmpeq_epi8(_mm256_max_epu8(a, vt), a);
            __m256i gb = _mm256_cmpeq_epi8(_mm256_max_epu8(b, vtc), b);
            _mm256_storeu_si256((__m256i *)(d + i), _mm256_and_si256(_mm256_and_si256(ga, gb), one));
        }
    }
#endif
    for (; i < n; i++) d[i] = (unsigned char)(d[i] > t && c3[i] > tc);
}

/* m = m0 & (a 4-neighbour of m0 inside the window), m0 in src, m in dst (both mw x mh) */
static void plane_filter(const unsigned char *src, unsigned char *dst, int mw, int mh)
{
    int x, y;
    for (y = 0; y < mh; y++) {
        const unsigned char *r = src + (size_t)y * mw;
        const unsigned char *up = y > 0 ? r - mw : NULL, *dn = y + 1 < mh ? r + mw : NULL;
        unsigned char *o = dst + (size_t)y * mw;
        x = 0;
#if TSDR_AVX2
        if (mw >= 34) {
            /* interior columns 1 .. mw - 2 in blocks of 32 */
            __m256i z = _mm256_setzero_si256();
            o[0] = (unsigned char)(r[0] & ((up ? up[0] : 0) | (dn ? dn[0] : 0) | r[1]));
            for (x = 1; x + 32 <= mw - 1; x += 32) {
                __m256i c = _mm256_loadu_si256((const __m256i *)(r + x));
                __m256i l = _mm256_loadu_si256((const __m256i *)(r + x - 1));
                __m256i rr = _mm256_loadu_si256((const __m256i *)(r + x + 1));
                __m256i u = up ? _mm256_loadu_si256((const __m256i *)(up + x)) : z;
                __m256i dd = dn ? _mm256_loadu_si256((const __m256i *)(dn + x)) : z;
                __m256i nb = _mm256_or_si256(_mm256_or_si256(l, rr), _mm256_or_si256(u, dd));
                _mm256_storeu_si256((__m256i *)(o + x), _mm256_and_si256(c, nb));
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

static int row_sum(const unsigned char *r, int x0, int x1)
{
    int s = 0, x = x0;
#if TSDR_AVX2
    __m256i acc = _mm256_setzero_si256(), z = _mm256_setzero_si256();
    for (; x + 32 <= x1; x += 32)
        acc = _mm256_add_epi64(acc, _mm256_sad_epu8(_mm256_loadu_si256((const __m256i *)(r + x)), z));
    {
        __m128i q = _mm_add_epi64(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
        s = (int)(_mm_cvtsi128_si64(q) + _mm_extract_epi64(q, 1));
    }
#endif
    for (; x < x1; x++) s += r[x];
    return s;
}

/* rc[y] = the ink of row y over columns [x0, x1), every window row */
static void row_counts(const unsigned char *m, int mw, int mh, int x0, int x1, int *rc)
{
    int y;
    for (y = 0; y < mh; y++) rc[y] = row_sum(m + (size_t)y * mw, x0, x1);
}

/* cc[x] = the ink of column x over rows [y0, y1) */
static void col_counts(const unsigned char *m, int mw, int y0, int y1, int *cc)
{
    int x, y;
    for (x = 0; x < mw; x++) cc[x] = 0;
    for (y = y0; y < y1; y++) {
        const unsigned char *r = m + (size_t)y * mw;
        for (x = 0; x < mw; x++) cc[x] += r[x];
    }
}

/* ---- the walks (tools/tsdesk_refine._walk / _walk_rows) ---- */

static int walk(const int *cnt, int start, int step, int n, int gap, const unsigned char *stop, int *more)
{
    int last = start, i = start + step, run = 0;
    while (i >= 0 && i < n) {
        if (stop && stop[i]) { *more = 0; return last; }
        if (cnt[i] > 0) { last = i; run = 0; }
        else if (++run > gap) { *more = 0; return last; }
        i += step;
    }
    *more = 1;
    return last;
}

static int walk_rows(const int *cnt, int start, int step, int n, int gap, int low, int rise, int low_max,
                     int beyond_max, int *more)
{
    int last = start, i = start + step, run = 0, lows = 0, valley = -1;
    while (i >= 0 && i < n) {
        int c = cnt[i];
        if (c > 0) {
            if (c <= low) {
                if (lows == 0) valley = i;
                if (++lows > low_max) { *more = 0; return valley; }
            } else {
                lows = 0;
                if (valley >= 0 && c >= rise) { *more = 0; return valley; }
            }
            if (valley >= 0 && (i - valley) * step > beyond_max) { *more = 0; return valley; }
            last = i;
            run = 0;
        } else if (++run > gap) {
            *more = 0;
            return last;
        }
        i += step;
    }
    *more = 1;
    return last;
}

/* ---- one box ---- */

typedef struct {
    const vocr_tsdr_opts *o;
    const unsigned char *m;
    int mw, mh, bx0, by0, w, h, lim_t, lim_b, S_a;
    int *rc, *sortbuf;
} tsdr_box;

/* the across pass over window columns [cx0, cx1): 0 when the core has no ink; top / bot = -1 when kept */
static int across(const tsdr_box *B, int cx0, int cx1, int *top, int *bot, unsigned *st)
{
    const vocr_tsdr_opts *o = B->o;
    int h = B->h, by0 = B->by0, q, k0, k1, k, first = -1, last = -1, n = 0, med, low, rise, low_max, beyond, more, t,
        b;
    row_counts(B->m, B->mw, B->mh, cx0, cx1, B->rc);
    q = imin(r100(h, o->core_pct), (h - 1) / 2);
    k0 = by0 + q;
    k1 = by0 + h - q;
    for (k = k0; k < k1; k++) {
        if (B->rc[k] > 0) {
            if (first < 0) first = k;
            last = k;
            B->sortbuf[n++] = B->rc[k];
        }
    }
    if (first < 0) return 0;
    qsort(B->sortbuf, (size_t)n, sizeof(int), cmp_int);
    med = B->sortbuf[(n - 1) / 2];
    low = o->valley_pct > 0 ? r100(med, o->valley_pct) : 0;
    rise = r100(med, o->rise_pct);
    low_max = imax(2, r100(h, o->low_run_pct));
    beyond = imax(2, r100(h, o->beyond_pct));
    *st = 0;
    t = walk_rows(B->rc, first, -1, B->mh, o->row_gap, low, rise, low_max, beyond, &more);
    if (more && B->lim_t) { t = -1; *st |= VOCR_TSDR_LIMIT; }
    else if (t - by0 > B->S_a) { t = -1; *st |= VOCR_TSDR_SHRINK; }
    b = walk_rows(B->rc, last, +1, B->mh, o->row_gap, low, rise, low_max, beyond, &more);
    if (more && B->lim_b) { b = -1; *st |= VOCR_TSDR_LIMIT; }
    else if ((by0 + h - 1) - b > B->S_a) { b = -1; *st |= VOCR_TSDR_SHRINK; }
    *top = t;
    *bot = b;
    return 1;
}

static unsigned refine_one(const unsigned char *g, unsigned W, unsigned H, unsigned stride, double *bx,
                           const vocr_tsdr_opts *o, tsdr_scr *s)
{
    double x0 = bx[0], y0 = bx[1], x1 = bx[2], y1 = bx[3], f;
    int ix0, iy0, ix1, iy1, h, w, A, Ls, Le, wx0, wx1, wy0, wy1, mw, mh, bx0, by0, x, y, bg, mg, nq = 0, qv[2], noise, t,
        tc, top, bot, left, right, rt, rb, hh, bar, have_bar = 0, c, k0, k1, first = -1, last = -1, G, S_l, more, cl,
        cr, bt, bb;
    unsigned st = 0, st2 = 0, hist[256], hist2[256];
    unsigned long n_ring;
    long outside, ink_all = 0, ink_in = 0;
    tsdr_box B;

    if (!isfinite(x0) || !isfinite(y0) || !isfinite(x1) || !isfinite(y1)) return VOCR_TSDR_SMALL;
    f = floor(x0); ix0 = f < 0 ? 0 : (f > W ? (int)W : (int)f);
    f = floor(y0); iy0 = f < 0 ? 0 : (f > H ? (int)H : (int)f);
    f = ceil(x1); ix1 = f < 0 ? 0 : (f > W ? (int)W : (int)f);
    f = ceil(y1); iy1 = f < 0 ? 0 : (f > H ? (int)H : (int)f);
    h = iy1 - iy0;
    w = ix1 - ix0;
    if (h < o->min_h || w < o->min_w) return VOCR_TSDR_SMALL;
    A = imax(o->band_min, r100(h, o->band_pct));
    Ls = imax(o->along_min, r100(h, o->start_pct));
    Le = imax(o->along_min, r100(h, o->end_pct));
    wx0 = imax(0, ix0 - Ls); wx1 = imin((int)W, ix1 + Le);
    wy0 = imax(0, iy0 - A); wy1 = imin((int)H, iy1 + A);
    mw = wx1 - wx0; mh = wy1 - wy0;
    bx0 = ix0 - wx0; by0 = iy0 - wy0;

    /* the background: the lower median of the window's border ring (rows 0 and mh - 1 whole, then columns 0 and
       mw - 1 of the rows between, as the twin concatenates them) */
    memset(hist, 0, sizeof hist);
    {
        const unsigned char *r0 = g + (size_t)wy0 * stride + wx0, *r1 = g + (size_t)(wy1 - 1) * stride + wx0;
        for (x = 0; x < mw; x++) { hist[r0[x]]++; hist[r1[x]]++; }
        for (y = 1; y < mh - 1; y++) {
            const unsigned char *r = g + (size_t)(wy0 + y) * stride + wx0;
            hist[r[0]]++;
            hist[r[mw - 1]]++;
        }
        n_ring = 2ul * (unsigned long)mw + 2ul * (unsigned long)(mh > 2 ? mh - 2 : 0);
    }
    {
        unsigned long cum = 0, idx = (n_ring - 1) / 2;
        for (bg = 0; bg < 256; bg++) { cum += hist[bg]; if (cum > idx) break; }
    }
    /* the window's d and c3 */
    for (y = 0; y < mh; y++) {
        row_d(g + (size_t)(wy0 + y) * stride + wx0, mw, bg, s->d + (size_t)y * mw);
        row_c3(g, W, H, stride, wx0, mw, wy0 + y, s->vmax, s->vmin, s->c3 + (size_t)y * mw);
    }
    /* the thresholds: tc from the box's c3; the noise from the along margins (the lower of the two margins'
       quantiles of d, over the pixels with c3 > tc when texture_pct > 0, 0 for an untextured margin); t */
    memset(hist, 0, sizeof hist);
    memset(hist2, 0, sizeof hist2);
    hist_rect(hist, s->d, mw, bx0, by0, bx0 + w, by0 + h);
    hist_rect(hist2, s->c3, mw, bx0, by0, bx0 + w, by0 + h);
    tc = imax(o->floor_, r100(hist_quantile(hist2, (unsigned long)w * (unsigned long)h, 99), o->k_pct));
    mg = imax(2, r100(h, o->margin_gap_pct));
    {
        int side, sx0, sx1;
        for (side = 0; side < 2; side++) {
            unsigned mh_[256];
            unsigned long ntex = 0, nall;
            sx0 = side == 0 ? 0 : bx0 + w + mg;
            sx1 = side == 0 ? bx0 - mg : mw;
            if (sx1 - sx0 < o->margin_min) continue;
            nall = (unsigned long)(sx1 - sx0) * (unsigned long)h;
            memset(mh_, 0, sizeof mh_);
            if (o->texture_pct > 0) {
                for (y = by0; y < by0 + h; y++) {
                    const unsigned char *rd = s->d + (size_t)y * mw, *rc3 = s->c3 + (size_t)y * mw;
                    for (x = sx0; x < sx1; x++)
                        if (rc3[x] > tc) { mh_[rd[x]]++; ntex++; }
                }
                if (ntex * 100ul < (unsigned long)o->texture_pct * nall) { qv[nq++] = 0; continue; }
                qv[nq++] = hist_quantile(mh_, ntex, o->margin_pct) + o->noise_add;
            } else {
                hist_rect(mh_, s->d, mw, sx0, by0, sx1, by0 + h);
                qv[nq++] = hist_quantile(mh_, nall, o->margin_pct) + o->noise_add;
            }
        }
    }
    noise = nq == 0 ? 0 : (nq == 1 ? qv[0] : imin(qv[0], qv[1]));
    t = imax(imax(o->floor_, r100(hist_quantile(hist, (unsigned long)w * (unsigned long)h, 99), o->k_pct)), noise);
    /* the ink: m0 over d, then m = m0 with a 4-neighbour, over c3 */
    plane_mask(s->d, s->c3, (size_t)mw * (size_t)mh, t, tc);
    plane_filter(s->d, s->c3, mw, mh);

    B.o = o; B.m = s->c3; B.mw = mw; B.mh = mh; B.bx0 = bx0; B.by0 = by0; B.w = w; B.h = h;
    B.lim_t = wy0 > 0; B.lim_b = wy1 < (int)H;
    B.S_a = r100(h, o->shrink_a_pct);
    B.rc = s->rc; B.sortbuf = s->sortbuf;
    if (!across(&B, bx0, bx0 + w, &top, &bot, &st)) return VOCR_TSDR_NOINK;
    rt = top >= 0 ? top : by0;
    rb = bot >= 0 ? bot + 1 : by0 + h;
    hh = rb - rt;
    col_counts(B.m, mw, rt, rb, s->cc);
    bar = imax(1, r100(hh, o->bar_pct));
    if (o->bar_pct > 0 && rt - bar >= 0 && rb + bar <= mh) {
        have_bar = 1;
        for (x = 0; x < mw; x++) {
            int all = 1;
            for (y = rt - bar; y < rb + bar && all; y++) {
                const unsigned char *r = B.m + (size_t)y * mw;
                all = r[x] | (x > 0 ? r[x - 1] : 0) | (x + 1 < mw ? r[x + 1] : 0);
            }
            s->barrier[x] = (unsigned char)all;
            if (all) s->cc[x] = 0;
        }
    }
    c = imin(r100(hh, o->along_core_pct), (w - 1) / 2);
    k0 = bx0 + c;
    k1 = bx0 + w - c;
    for (x = k0; x < k1; x++) {
        if (s->cc[x] > 0) {
            if (first < 0) first = x;
            last = x;
        }
    }
    if (first < 0) return st | VOCR_TSDR_NOINK;
    G = imax(o->gap_min, r100(hh, o->gap_pct));
    S_l = r100(hh, o->shrink_l_pct);
    left = walk(s->cc, first, -1, mw, G, have_bar ? s->barrier : NULL, &more);
    if (more && wx0 > 0) { left = -1; st |= VOCR_TSDR_LIMIT; }
    else if (left - bx0 > S_l) { left = -1; st |= VOCR_TSDR_SHRINK; }
    right = walk(s->cc, last, +1, mw, G, have_bar ? s->barrier : NULL, &more);
    if (more && wx1 < (int)W) { right = -1; st |= VOCR_TSDR_LIMIT; }
    else if ((bx0 + w - 1) - right > S_l) { right = -1; st |= VOCR_TSDR_SHRINK; }
    cl = left >= 0 ? left : bx0;
    cr = right >= 0 ? right + 1 : bx0 + w;
    {
        int t2, b2;
        if (across(&B, cl, cr, &t2, &b2, &st2)) {
            top = t2;
            bot = b2;
            st |= st2;
        }
    }
    bt = top >= 0 ? top : by0;
    bb = bot >= 0 ? bot + 1 : by0 + h;
    outside = (long)mh * mw - (long)(bb - bt) * (cr - cl);
    for (y = 0; y < mh; y++) {
        const unsigned char *r = B.m + (size_t)y * mw;
        int sall = row_sum(r, 0, mw);
        ink_all += sall;
        if (y >= bt && y < bb) ink_in += row_sum(r, cl, cr);
    }
    if (outside > 0 && (ink_all - ink_in) * 100 > (long)o->busy_pct * outside) return st | VOCR_TSDR_BUSY;
    if (left >= 0) { bx[0] = (double)(wx0 + left); st |= VOCR_TSDR_SNAP_L; }
    if (top >= 0) { bx[1] = (double)(wy0 + top); st |= VOCR_TSDR_SNAP_T; }
    if (right >= 0) { bx[2] = (double)(wx0 + right + 1); st |= VOCR_TSDR_SNAP_R; }
    if (bot >= 0) { bx[3] = (double)(wy0 + bot + 1); st |= VOCR_TSDR_SNAP_B; }
    return st;
}

int vocr_tsdr_refine(const unsigned char *grey, unsigned w, unsigned h, unsigned stride, double *xyxy,
                     const unsigned char *vertical, unsigned n, const vocr_tsdr_opts *opts, unsigned *status,
                     void *scratch, size_t scratch_bytes)
{
    vocr_tsdr_opts def;
    tsdr_scr s;
    unsigned i;
    if (!grey || !w || !h || stride < w || (n && !xyxy) || !scratch) return VOCR_TSDR_E_ARG;
    if (scratch_bytes < vocr_tsdr_scratch_bytes(w, h)) return VOCR_TSDR_E_SCRATCH;
    if (!opts) {
        vocr_tsdr_defaults(&def);
        opts = &def;
    }
    scr_map(&s, scratch, w, h);
    for (i = 0; i < n; i++) {
        unsigned st;
        if (vertical && vertical[i]) st = VOCR_TSDR_VERT;
        else st = refine_one(grey, w, h, stride, xyxy + 4 * (size_t)i, opts, &s);
        if (status) status[i] = st;
    }
    return VOCR_TSDR_OK;
}
