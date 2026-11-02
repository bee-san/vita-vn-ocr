/* vocr_tsd_post.c - textspot-desktop's box post-pass and reading filter (vocr_tsd_post.h): the C twins of
 * tools/tsdesk.py's merge_overlaps, split_rows, extend_ends and drop_set, bit for bit. */
#include "vocr_tsd_post.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

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
}

void vocr_tsdd_defaults(vocr_tsdd_opts *o)
{
    if (!o) return;
    memset(o, 0, sizeof *o);
    o->nojp_gap_pct = 100;
}

int vocr_tsdp_nopts(void) { return VOCR_TSDP_NOPTS; }
int vocr_tsdd_nopts(void) { return VOCR_TSDD_NOPTS; }

int vocr_tsdp_pre_on(const vocr_tsdp_opts *o) { return o && (o->merge_pct || o->rows_pct || o->gap_pct); }
int vocr_tsdp_post_on(const vocr_tsdp_opts *o) { return o && o->ext_pct; }

static int r100(long v, int pct) { return (int)((v * (long)pct + 50) / 100); }
static int imax(int a, int b) { return a > b ? a : b; }
static int imin(int a, int b) { return a < b ? a : b; }
static double dmin(double a, double b) { return b < a ? b : a; }    /* Python's min(a, b): a unless b < a */
static double dmax(double a, double b) { return b > a ? b : a; }    /* Python's max(a, b) */

/* ----------------------------------------------------------------------------- scratch */

typedef struct {
    unsigned char *d, *c3, *m0, *m;   /* window planes (at most the frame) */
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
           (size_t)cap * sizeof(vocr_tsdp_box) + 64;
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

/* c3 (tools/tsdesk_refine._c3: the 3 x 3 max - min, the frame's edge pixels repeated) and d = |g - bg| over the window
 * [wx0, wx0 + mw) x [wy0, wy0 + mh), row-major mw wide */
static void window_planes(const unsigned char *g, unsigned W, unsigned H, unsigned stride, int wx0, int wy0, int mw,
                          int mh, int bg, unsigned char *d, unsigned char *c3)
{
    int x, y, i, j;
    for (y = 0; y < mh; y++) {
        const int gy = wy0 + y;
        for (x = 0; x < mw; x++) {
            const int gx = wx0 + x;
            int mx = 0, mn = 255, v = (int)g[(size_t)gy * stride + (size_t)gx] - bg;
            for (j = -1; j <= 1; j++) {
                int yy = gy + j;
                yy = yy < 0 ? 0 : (yy > (int)H - 1 ? (int)H - 1 : yy);
                for (i = -1; i <= 1; i++) {
                    int xx = gx + i, p;
                    xx = xx < 0 ? 0 : (xx > (int)W - 1 ? (int)W - 1 : xx);
                    p = g[(size_t)yy * stride + (size_t)xx];
                    if (p > mx) mx = p;
                    if (p < mn) mn = p;
                }
            }
            c3[(size_t)y * mw + x] = (unsigned char)(mx - mn);
            d[(size_t)y * mw + x] = (unsigned char)(v < 0 ? -v : v);
        }
    }
}

/* t and tc: max(floor, r(q99 over the region, k_pct)) of d and c3 over [rx0, rx1) x [ry0, ry1) of the window */
static void thresholds(const unsigned char *d, const unsigned char *c3, int mw, int rx0, int ry0, int rx1, int ry1,
                       int *t, int *tc)
{
    unsigned hd[256], hc[256];
    unsigned long n = 0;
    int x, y;
    memset(hd, 0, sizeof hd);
    memset(hc, 0, sizeof hc);
    for (y = ry0; y < ry1; y++)
        for (x = rx0; x < rx1; x++) {
            hd[d[(size_t)y * mw + x]]++;
            hc[c3[(size_t)y * mw + x]]++;
            n++;
        }
    *t = imax(28, r100(hist_quantile(hd, n, 99), 40));
    *tc = imax(28, r100(hist_quantile(hc, n, 99), 40));
}

/* m = (d > t) & (c3 > tc) & (a 4-neighbour of that inside the window), over the window */
static void window_mask(const unsigned char *d, const unsigned char *c3, int mw, int mh, int t, int tc,
                        unsigned char *m0, unsigned char *m)
{
    int x, y;
    const size_t n = (size_t)mw * mh;
    size_t i;
    for (i = 0; i < n; i++) m0[i] = (unsigned char)(d[i] > t && c3[i] > tc);
    for (y = 0; y < mh; y++)
        for (x = 0; x < mw; x++) {
            const size_t q = (size_t)y * mw + x;
            unsigned char nb = 0;
            if (y > 0) nb |= m0[q - mw];
            if (y + 1 < mh) nb |= m0[q + mw];
            if (x > 0) nb |= m0[q - 1];
            if (x + 1 < mw) nb |= m0[q + 1];
            m[q] = (unsigned char)(m0[q] & nb);
        }
}

/* tools/tsdesk.ink_mask of the pixel box [ix0, ix1) x [iy0, iy1): the window (the box grown by max(2, r(h, 25))),
 * its mask in s->m (mw x mh), and the box's offset in it */
static void ink_box(const unsigned char *g, unsigned W, unsigned H, unsigned stride, int ix0, int iy0, int ix1,
                    int iy1, tsdp_scr *s, int *mw_, int *mh_, int *bx0, int *by0)
{
    const int h = iy1 - iy0, A = imax(2, r100(h, 25));
    const int wx0 = imax(0, ix0 - A), wy0 = imax(0, iy0 - A), wx1 = imin((int)W, ix1 + A), wy1 = imin((int)H, iy1 + A);
    const int mw = wx1 - wx0, mh = wy1 - wy0;
    int t, tc, bg;
    bg = ring_median(g, stride, wx0, wy0, mw, mh);
    window_planes(g, W, H, stride, wx0, wy0, mw, mh, bg, s->d, s->c3);
    thresholds(s->d, s->c3, mw, ix0 - wx0, iy0 - wy0, ix1 - wx0, iy1 - wy0, &t, &tc);
    window_mask(s->d, s->c3, mw, mh, t, tc, s->m0, s->m);
    *mw_ = mw;
    *mh_ = mh;
    *bx0 = ix0 - wx0;
    *by0 = iy0 - wy0;
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
    int mw, mh, bx0, by0, x, y, ninks = 0, med, low, lo = -1, hi = -1, min_band, min_len, found = 0, bv0 = 0, bv1 = 0;
    int best_mn = 0, best_mx = 1, best_len = 0;
    if (*k >= cap) return;
    if (b->vertical || h < 8 || w < 8 || depth >= 2) {
        out[(*k)++] = *b;
        return;
    }
    ink_box(g, W, H, stride, ix0, iy0, ix1, iy1, s, &mw, &mh, &bx0, &by0);
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

int vocr_tsdp_pre(const unsigned char *grey, unsigned w, unsigned h, unsigned stride, vocr_tsdp_box *boxes,
                  unsigned *n, unsigned cap, const vocr_tsdp_opts *o, void *scratch, size_t bytes)
{
    tsdp_scr s;
    unsigned i, k = 0, m;
    if (!n || !o || (*n && !boxes) || *n > cap) return VOCR_TSDP_E_ARG;
    if (o->gap_pct) return VOCR_TSDP_E_ARG;
    if (!o->merge_pct && !o->rows_pct) return VOCR_TSDP_OK;
    if (!grey || !w || !h || stride < w || !scratch) return VOCR_TSDP_E_ARG;
    if (bytes < vocr_tsdp_scratch_bytes(w, h, cap)) return VOCR_TSDP_E_SCRATCH;
    scr_map(&s, scratch, w, h, cap);
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

/* tools/tsdesk._ink_beyond: the mask of the union window [ux0, ux1) x [uy0, uy1) of the line's pixel box and the search
 * window, in s->m (uw wide), the background and thresholds from the line's box; *ry0 its first row in the window */
static void ink_beyond(const unsigned char *g, unsigned W, unsigned H, unsigned stride, int iy0, int iy1, int bx0,
                       int bx1, int wx0, int wx1, tsdp_scr *s, int *ux0_, int *uw_, int *nr_, int *ry0_)
{
    const int h = iy1 - iy0, A = imax(2, r100(h, 25));
    const int ux0 = imax(0, imin(bx0, wx0) - 1), ux1 = imin((int)W, imax(bx1, wx1) + 1);
    const int uy0 = imax(0, iy0 - A), uy1 = imin((int)H, iy1 + A);
    const int uw = ux1 - ux0, uh = uy1 - uy0, ry0 = iy0 - uy0;
    int t, tc, bg;
    bg = ring_median(g, stride, ux0, uy0, uw, uh);
    window_planes(g, W, H, stride, ux0, uy0, uw, uh, bg, s->d, s->c3);
    thresholds(s->d, s->c3, uw, bx0 - ux0, ry0, bx1 - ux0, ry0 + h, &t, &tc);
    window_mask(s->d, s->c3, uw, uh, t, tc, s->m0, s->m);
    *ux0_ = ux0;
    *uw_ = uw;
    *nr_ = uh;
    *ry0_ = ry0;
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
        int side, L, G, Gr, D, q;
        s.reach[2 * i] = s.reach[2 * i + 1] = 0;
        if (b->vertical || bh < 6 || ix1 - ix0 < 4) continue;
        L = r100(bh, o->ext_pct);
        G = imax(2, r100(bh, o->ext_gap_pct));
        Gr = imax(2, r100(bh, o->ext_run_pct));
        D = imax(2, r100(bh, o->ext_dash_pct));
        q = r100(bh, 25);
        for (side = -1; side <= 1; side += 2) {
            int wx0, wx1, ux0, uw, nr, ry0, k, ran_out, c, nwc, limit, colbuf_ok = 1;
            tsdp_cols cl;
            if (side < 0) {
                wx0 = imax(imax(0, lim[0]), ix0 - L);
                wx1 = ix0;
            } else {
                wx0 = ix1;
                wx1 = imin(imin((int)w, lim[2]), ix1 + L);
            }
            if (wx1 - wx0 < D) continue;
            ink_beyond(grey, w, h, stride, iy0, iy1, ix0, ix1, wx0, wx1, &s, &ux0, &uw, &nr, &ry0);
            nwc = wx1 - wx0;
            cl.m = s.m;
            cl.uw = uw;
            cl.off = wx0 - ux0;
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
                                for (yy = 0; yy < ry0 && !ink; yy++) ink = s.m[(size_t)yy * uw + cl.off + xx];
                                for (yy = ry0 + bh; yy < nr && !ink; yy++) ink = s.m[(size_t)yy * uw + cl.off + xx];
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
        if (o->rep_pct && r.n >= 4 && !is_jp(r.top_cp) && r.top_n * 100 >= (long)o->rep_pct * r.n) {
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
