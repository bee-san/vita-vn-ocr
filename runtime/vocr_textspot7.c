/* textspot7's fallback (F) and post-pass (D): see vocr_textspot7.h; synth/textspot7_twin.py is the specification. */
#include "vocr_textspot7.h"

#include <string.h>

#include "vocr_lines_refine.h"

#define T7_CONTRAST_MIN 40

static int imin(int a, int b) { return a < b ? a : b; }
static int imax(int a, int b) { return a > b ? a : b; }

static int vov(const VocrT7Box *a, const VocrT7Box *b)
{
    int t = imax(a->y, b->y), e = imin(a->y + a->h, b->y + b->h);
    return e > t ? e - t : 0;
}

static int hov(const VocrT7Box *a, const VocrT7Box *b)
{
    int t = imax(a->x, b->x), e = imin(a->x + a->w, b->x + b->w);
    return e > t ? e - t : 0;
}

/* The lower middle of v[0..n) (n >= 1), sorting v in place. */
static int lower_median(int *v, unsigned n)
{
    unsigned i, j;
    for (i = 1; i < n; i++) {
        int k = v[i];
        for (j = i; j > 0 && v[j - 1] > k; j--)
            v[j] = v[j - 1];
        v[j] = k;
    }
    return v[(n - 1u) / 2u];
}

static unsigned find(unsigned char *parent, unsigned i)
{
    while (parent[i] != i) {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    return i;
}

/* Row id per box: the smallest index of its component under 2 vov >= the smaller height. */
static void rows_of(const VocrT7Box *b, unsigned n, unsigned char *row)
{
    unsigned char parent[VOCR_T7_MAX_BOXES];
    unsigned i, j;
    for (i = 0; i < n; i++)
        parent[i] = (unsigned char)i;
    for (i = 0; i < n; i++)
        for (j = i + 1; j < n; j++)
            if (2 * vov(&b[i], &b[j]) >= imin(b[i].h, b[j].h)) {
                unsigned ri = find(parent, i), rj = find(parent, j);
                if (ri != rj) {
                    if (ri < rj)
                        parent[rj] = (unsigned char)ri;
                    else
                        parent[ri] = (unsigned char)rj;
                }
            }
    for (i = 0; i < n; i++)
        row[i] = (unsigned char)find(parent, i);
}

static int centre_in(const VocrT7Box *s, const VocrT7Box *b)
{
    int cx = 2 * s->x + s->w, cy = 2 * s->y + s->h;
    return 2 * b->x <= cx && cx < 2 * (b->x + b->w) && 2 * b->y <= cy && cy < 2 * (b->y + b->h);
}

static int s_lacks(const VocrT7Box *b, const VocrT7Box *s, unsigned ns, int lam, int hmed)
{
    int a[VOCR_T7_MAX_BOXES], e[VOCR_T7_MAX_BOXES];
    unsigned m = 0, i, j;
    int cov = 0, end = b->y;
    if (!lam || 1000 * b->h < lam * hmed)
        return 0;
    for (i = 0; i < ns; i++) {
        int ka, ke;
        if (hov(&s[i], b) <= 0)
            continue;
        ka = imax(s[i].y, b->y);
        ke = imin(s[i].y + s[i].h, b->y + b->h);
        for (j = m; j > 0 && (a[j - 1] > ka || (a[j - 1] == ka && e[j - 1] > ke)); j--) {
            a[j] = a[j - 1];
            e[j] = e[j - 1];
        }
        a[j] = ka;
        e[j] = ke;
        m++;
    }
    for (i = 0; i < m; i++) {
        int lo = a[i];
        if (e[i] <= lo)
            continue;
        lo = imax(lo, end);
        if (e[i] > lo) {
            cov += e[i] - lo;
            end = e[i];
        }
    }
    return 2 * cov < b->h;
}

static int s_short(const VocrT7Box *b, const VocrT7Box *s, unsigned ns, int kappa)
{
    unsigned i, k = 0;
    int u0 = 0, u1 = 0, ex;
    if (!kappa)
        return 0;
    for (i = 0; i < ns; i++) {
        if (2 * vov(b, &s[i]) < imin(b->h, s[i].h) || hov(b, &s[i]) <= 0)
            continue;
        if (!k++) {
            u0 = s[i].x;
            u1 = s[i].x + s[i].w;
        } else {
            u0 = imin(u0, s[i].x);
            u1 = imax(u1, s[i].x + s[i].w);
        }
    }
    if (!k)
        return 0;
    ex = imax(u0 - b->x, b->x + b->w - u1);
    return 1000 * ex > kappa * b->h;
}

static int g_merge(const VocrT7Box *s, unsigned ns, const VocrT7Box *g, unsigned ng)
{
    unsigned char row[VOCR_T7_MAX_BOXES];
    unsigned i, j;
    rows_of(s, ns, row);
    for (i = 0; i < ng; i++) {
        int first = -1;
        for (j = 0; j < ns; j++) {
            if (!centre_in(&s[j], &g[i]))
                continue;
            if (first < 0)
                first = row[j];
            else if (row[j] != (unsigned)first)
                return 1;
        }
    }
    return 0;
}

int vocr_textspot7_choose(const VocrT7Box *s, unsigned ns, const VocrT7Box *g, unsigned ng, const VocrT7Params *p)
{
    int v[VOCR_T7_MAX_BOXES];
    unsigned i;
    int cmed, hmed;
    if (!p || ns > VOCR_T7_MAX_BOXES || ng > VOCR_T7_MAX_BOXES || (ns && !s) || (ng && !g) ||
        p->lam < 0 || p->kappa < 0)
        return VOCR_T7_E_ARG;
    if (p->mode == VOCR_T7_MODE_S) {
        if (!ns)
            return ng ? VOCR_T7_SHOW_GRAD : VOCR_T7_SHOW_SPOTTER;
        for (i = 0; i < ns; i++)
            v[i] = s[i].conf;
        cmed = lower_median(v, ns);
        if (cmed < p->tau)
            return VOCR_T7_SHOW_GRAD;
        for (i = 0; i < ns; i++)
            v[i] = s[i].h;
        hmed = lower_median(v, ns);
        for (i = 0; i < ng; i++)
            if (s_lacks(&g[i], s, ns, p->lam, hmed) || s_short(&g[i], s, ns, p->kappa))
                return VOCR_T7_SHOW_GRAD;
        return VOCR_T7_SHOW_SPOTTER;
    }
    if (p->mode == VOCR_T7_MODE_G) {
        if (!ng)
            return ns ? VOCR_T7_SHOW_SPOTTER : VOCR_T7_SHOW_GRAD;
        if (!ns)
            return VOCR_T7_SHOW_GRAD;
        for (i = 0; i < ns; i++)
            v[i] = s[i].conf;
        cmed = lower_median(v, ns);
        return (cmed >= p->tau && g_merge(s, ns, g, ng)) ? VOCR_T7_SHOW_SPOTTER : VOCR_T7_SHOW_GRAD;
    }
    return VOCR_T7_E_ARG;
}

/* ----------------------------------------------------------------------------- D */

/* EXTEND's row buffers: the 3-row column max and min of a row's columns (with one column each side). */
static size_t extend_bytes(unsigned max_width)
{
    return 2u * ((size_t)max_width + 2u);
}

size_t vocr_textspot7_arena_bytes_steps(unsigned max_width, unsigned max_height, unsigned steps)
{
    size_t r = 0, e = 0;
    if (!max_width || !max_height)
        return 0;
    if (steps & VOCR_T7_TIGHTEN)
        r = vocr_refine_arena_bytes(max_width, max_height);
    if (steps & VOCR_T7_EXTEND)
        e = extend_bytes(max_width);
    return r > e ? r : e;
}

size_t vocr_textspot7_arena_bytes(unsigned max_width, unsigned max_height)
{
    return vocr_textspot7_arena_bytes_steps(max_width, max_height, VOCR_T7_STEPS);
}

static int same_row6(const VocrT7Box *a, const VocrT7Box *b)
{
    return 10 * vov(a, b) >= 6 * imin(a->h, b->h);
}

static void join(VocrT7Box *b, unsigned *n, int g10)
{
    for (;;) {
        unsigned i, j, hi = 0, hj = 0;
        int hit = 0;
        for (i = 0; i < *n && !hit; i++)
            for (j = i + 1; j < *n; j++) {
                int gap;
                if (!same_row6(&b[i], &b[j]))
                    continue;
                gap = imax(b[i].x, b[j].x) - imin(b[i].x + b[i].w, b[j].x + b[j].w);
                if (10 * gap <= g10 * imax(b[i].h, b[j].h)) {
                    hi = i;
                    hj = j;
                    hit = 1;
                    break;
                }
            }
        if (!hit)
            return;
        {
            VocrT7Box *a = &b[hi], *c = &b[hj];
            int x0 = imin(a->x, c->x), y0 = imin(a->y, c->y);
            int x1 = imax(a->x + a->w, c->x + c->w), y1 = imax(a->y + a->h, c->y + c->h);
            unsigned dark = a->w >= c->w ? a->dark : c->dark;
            a->conf = imin(a->conf, c->conf);
            a->x = x0;
            a->y = y0;
            a->w = x1 - x0;
            a->h = y1 - y0;
            a->dark = dark;
            memmove(&b[hj], &b[hj + 1], (*n - hj - 1u) * sizeof *b);
            (*n)--;
        }
    }
}

/* max - min of the 3 x 3 window at (x, y), edges replicated. */
static int contrast_at(const VocrLinesImage *im, int x, int y)
{
    int W = (int)im->width, H = (int)im->height, mx = 0, mn = 255, dy, dx;
    for (dy = -1; dy <= 1; dy++) {
        int yy = imin(imax(y + dy, 0), H - 1);
        const unsigned char *row = im->pixels + (size_t)yy * im->stride;
        for (dx = -1; dx <= 1; dx++) {
            int v = row[imin(imax(x + dx, 0), W - 1)];
            if (v > mx)
                mx = v;
            if (v < mn)
                mn = v;
        }
    }
    return mx - mn;
}

/* The 3 x 3 contrast of row y at columns [x0, x1) into the histogram, edges replicated: the column max and min
 * over rows y - 1 .. y + 1 once per column (x0 - 1 .. x1, clamped), then each pixel's 3 columns of them. The same
 * values as contrast_at, for a fraction of the reads. buf holds 2 (x1 - x0 + 2) bytes. */
static void row_hist(const VocrLinesImage *im, int y, int x0, int x1, unsigned char *buf, unsigned long *hist)
{
    int W = (int)im->width, H = (int)im->height, n = x1 - x0 + 2, c, x;
    const unsigned char *r0 = im->pixels + (size_t)imax(y - 1, 0) * im->stride;
    const unsigned char *r1 = im->pixels + (size_t)y * im->stride;
    const unsigned char *r2 = im->pixels + (size_t)imin(y + 1, H - 1) * im->stride;
    unsigned char *vmx = buf, *vmn = buf + n;
    for (c = 0; c < n; c++) {
        int xx = imin(imax(x0 - 1 + c, 0), W - 1);
        unsigned a = r0[xx], b = r1[xx], d = r2[xx];
        unsigned mx = a > b ? a : b, mn = a < b ? a : b;
        vmx[c] = (unsigned char)(mx > d ? mx : d);
        vmn[c] = (unsigned char)(mn < d ? mn : d);
    }
    for (x = 0; x < x1 - x0; x++) {
        unsigned mx = vmx[x], mn = vmn[x];
        if (vmx[x + 1] > mx) mx = vmx[x + 1];
        if (vmx[x + 2] > mx) mx = vmx[x + 2];
        if (vmn[x + 1] < mn) mn = vmn[x + 1];
        if (vmn[x + 2] < mn) mn = vmn[x + 2];
        hist[mx - mn]++;
    }
}

static int ink_threshold(const VocrLinesImage *im, const VocrT7Box *b, unsigned char *buf)
{
    unsigned long hist[256], k, cum = 0;
    unsigned long n = (unsigned long)b->w * (unsigned long)b->h;
    int y, v;
    if (!n)
        return 255;
    memset(hist, 0, sizeof hist);
    for (y = b->y; y < b->y + b->h; y++)
        row_hist(im, y, b->x, b->x + b->w, buf, hist);
    k = (99ul * (n - 1ul)) / 100ul;
    for (v = 0; v < 256; v++) {
        cum += hist[v];
        if (cum > k)
            break;
    }
    return imax(T7_CONTRAST_MIN, (2 * v) / 5);
}

static int inked(const VocrLinesImage *im, const VocrT7Box *b, int x, int t)
{
    int y, c = 0;
    for (y = b->y; y < b->y + b->h; y++)
        if (contrast_at(im, x, y) >= t && ++c >= 2)
            return 1;
    return 0;
}

static void extend(const VocrLinesImage *im, VocrT7Box *b, unsigned n, unsigned char *buf)
{
    VocrT7Box orig[VOCR_T7_MAX_BOXES];
    unsigned i, k;
    int W = (int)im->width;
    memcpy(orig, b, n * sizeof *b);
    for (i = 0; i < n; i++) {
        const VocrT7Box *o = &orig[i];
        int x = o->x, w = o->w, h = o->h;
        int t = ink_threshold(im, o, buf), reach = (3 * h) / 2, gapmax = h / 2;
        int pad = imax(2, (15 * h + 50) / 100);
        int stop = W, stopl = 0, last, gap, X, x0 = x, x1 = x + w;
        for (k = 0; k < n; k++) {
            if (k == i || !same_row6(o, &orig[k]))
                continue;
            if (orig[k].x >= x + w)
                stop = imin(stop, orig[k].x);
            if (orig[k].x + orig[k].w <= x)
                stopl = imax(stopl, orig[k].x + orig[k].w);
        }
        last = -1;
        gap = 0;
        for (X = x + w; X < imin(stop, x + w + reach); X++) {
            if (inked(im, o, X, t)) {
                last = X;
                gap = 0;
            } else if (++gap > gapmax) {
                break;
            }
        }
        if (last >= 0)
            x1 = imax(x1, imin(stop, last + 1 + pad));
        last = -1;
        gap = 0;
        for (X = x - 1; X >= imax(stopl, x - reach); X--) {
            if (inked(im, o, X, t)) {
                last = X;
                gap = 0;
            } else if (++gap > gapmax) {
                break;
            }
        }
        if (last >= 0)
            x0 = imin(x0, imax(stopl, last - pad));
        b[i].x = x0;
        b[i].w = x1 - x0;
    }
}

static int tighten(const VocrLinesImage *im, VocrT7Box *b, unsigned n, void *arena, size_t arena_bytes)
{
    unsigned base = 0;
    while (base < n) {
        VocrLineSet set;
        unsigned m = n - base < VOCR_LINES_MAX ? n - base : VOCR_LINES_MAX, i;
        int rc;
        memset(&set, 0, sizeof set);
        for (i = 0; i < m; i++) {
            set.lines[i].x = (unsigned)b[base + i].x;
            set.lines[i].y = (unsigned)b[base + i].y;
            set.lines[i].w = (unsigned)b[base + i].w;
            set.lines[i].h = (unsigned)b[base + i].h;
            set.lines[i].dark = b[base + i].dark;
        }
        set.count = m;
        rc = vocr_refine_lines(im, VOCR_REFINE_TIGHTEN, arena, arena_bytes, &set, NULL);
        if (rc == VOCR_LINES_E_ARENA)
            return VOCR_T7_E_ARENA;
        if (rc != VOCR_LINES_OK || set.count != m)
            return VOCR_T7_E_ARG;
        for (i = 0; i < m; i++) {
            b[base + i].x = (int)set.lines[i].x;
            b[base + i].y = (int)set.lines[i].y;
            b[base + i].w = (int)set.lines[i].w;
            b[base + i].h = (int)set.lines[i].h;
            b[base + i].dark = set.lines[i].dark;
        }
        base += m;
    }
    return VOCR_T7_OK;
}

/* assign (the centre rule, or VOCR_T7_ASSIGN's) and join: the steps both forms share */
static void assign_join(const VocrT7Box *fb, unsigned n, int x0, int y0, int x1, int y1, unsigned steps,
                        VocrT7Box *out, unsigned *n_out)
{
    int W = x1 - x0, H = y1 - y0;
    unsigned i, m = 0;
    for (i = 0; i < n; i++) {
        const VocrT7Box *b = &fb[i];
        int cx = 2 * b->x + b->w, cy = 2 * b->y + b->h;
        int keep = 2 * x0 <= cx && cx < 2 * x1 && 2 * y0 <= cy && cy < 2 * y1;
        int a0, b0, a1, b1;
        if (!keep && (steps & VOCR_T7_ASSIGN)) {
            int ih = imin(b->y + b->h, y1) - imax(b->y, y0);
            int iw = imin(b->x + b->w, x1) - imax(b->x, x0);
            keep = ih > 0 && iw > 0 && 10 * ih >= 6 * b->h && iw >= b->h;
        }
        if (!keep)
            continue;
        a0 = imax(0, b->x - x0);
        b0 = imax(0, b->y - y0);
        a1 = imin(W, b->x + b->w - x0);
        b1 = imin(H, b->y + b->h - y0);
        if (a1 <= a0 || b1 <= b0)
            continue;
        out[m].x = a0;
        out[m].y = b0;
        out[m].w = a1 - a0;
        out[m].h = b1 - b0;
        out[m].conf = b->conf;
        out[m].dark = b->dark;
        m++;
    }
    if (steps & VOCR_T7_JOIN10)
        join(out, &m, 10);
    else if (steps & VOCR_T7_JOIN5)
        join(out, &m, 5);
    *n_out = m;
}

int vocr_textspot7_post(const VocrT7Box *fb, unsigned n, int x0, int y0, int x1, int y1,
                        const VocrLinesImage *region, unsigned steps, void *arena, size_t arena_bytes,
                        VocrT7Box *out, unsigned *n_out)
{
    int W = x1 - x0, H = y1 - y0;
    unsigned m = 0;
    if (!out || !n_out || (n && !fb) || n > VOCR_T7_MAX_BOXES || (steps & ~VOCR_T7_STEPS) || W <= 0 || H <= 0)
        return VOCR_T7_E_ARG;
    if (steps & (VOCR_T7_EXTEND | VOCR_T7_TIGHTEN)) {
        if (!region || !region->pixels || region->bpp != VOCR_LINES_GREY || (int)region->width != W ||
            (int)region->height != H || region->stride < region->width)
            return VOCR_T7_E_ARG;
        if (!arena || arena_bytes < vocr_textspot7_arena_bytes_steps(region->width, region->height, steps))
            return VOCR_T7_E_ARENA;
    }
    assign_join(fb, n, x0, y0, x1, y1, steps, out, &m);
    if (steps & VOCR_T7_EXTEND)
        extend(region, out, m, (unsigned char *)arena);
    if (steps & VOCR_T7_TIGHTEN) {
        int rc = tighten(region, out, m, arena, arena_bytes);
        if (rc != VOCR_T7_OK)
            return rc;
    }
    *n_out = m;
    return VOCR_T7_OK;
}

/* ----------------------------------------------------------------------------- the row-streamed form */

typedef struct {
    const VocrLinesRows *src;
    unsigned char *ring;    /* 3 rows of width bytes, row y in slot y % 3 */
    int W, H, last;         /* last: the newest row read in this pass, -1 before the pass's first */
} RowWin;

static void win_begin(RowWin *w)
{
    w->last = -1;
}

/* Rows y - 1, y, y + 1 (clamped) into r, reading the rows the pass has not read yet, in increasing order, each
 * once (vocr_lines_rows.h's contract): a pass asks for y = ya, ya + 1, ... */
static int win_at(RowWin *w, int y, const unsigned char **r)
{
    int need = imin(y + 1, w->H - 1), lo = imax(y - 1, 0), yy;
    if (w->last < lo - 1)
        w->last = lo - 1;
    for (yy = w->last + 1; yy <= need; yy++) {
        const unsigned char *p = w->src->row(w->src->user, (unsigned)yy);
        if (!p)
            return -1;
        memcpy(w->ring + (size_t)(yy % 3) * (size_t)w->W, p, (size_t)w->W);
        w->last = yy;
    }
    r[0] = w->ring + (size_t)(lo % 3) * (size_t)w->W;
    r[1] = w->ring + (size_t)(y % 3) * (size_t)w->W;
    r[2] = w->ring + (size_t)(need % 3) * (size_t)w->W;
    return 0;
}

static int contrast_rows(const unsigned char *const *r, int W, int x)
{
    int mx = 0, mn = 255, dy, dx;
    for (dy = 0; dy < 3; dy++)
        for (dx = -1; dx <= 1; dx++) {
            int v = r[dy][imin(imax(x + dx, 0), W - 1)];
            if (v > mx)
                mx = v;
            if (v < mn)
                mn = v;
        }
    return mx - mn;
}

/* EXTEND for one box from the rows: the threshold's histogram (pass 1), then each scanned column's inked count
 * (pass 2, only the columns the scan can reach). The same result as extend() on the image. */
static int extend_one_rows(RowWin *w, const VocrT7Box *o, int stop, int stopl, VocrT7Box *b)
{
    unsigned long hist[256], k, cum = 0, n = (unsigned long)o->w * (unsigned long)o->h;
    int reach = (3 * o->h) / 2, gapmax = o->h / 2, pad = imax(2, (15 * o->h + 50) / 100);
    int rlo = o->x + o->w, rhi = imin(stop, o->x + o->w + reach);      /* right scan columns [rlo, rhi) */
    int llo = imax(stopl, o->x - reach), lhi = o->x;                    /* left scan columns [llo, lhi) */
    unsigned short cnt[2 * 1024];
    int y, x, t, v, last, gap, X, x0 = o->x, x1 = o->x + o->w;
    const unsigned char *r[3];
    if (rhi - rlo > 1024 || lhi - llo > 1024)
        return VOCR_T7_E_ARG;
    memset(hist, 0, sizeof hist);
    win_begin(w);
    for (y = o->y; y < o->y + o->h; y++) {
        if (win_at(w, y, r))
            return VOCR_LINES_E_SOURCE;
        for (x = o->x; x < o->x + o->w; x++)
            hist[contrast_rows(r, w->W, x)]++;
    }
    if (!n) {
        t = 255;
    } else {
        k = (99ul * (n - 1ul)) / 100ul;
        for (v = 0; v < 256; v++) {
            cum += hist[v];
            if (cum > k)
                break;
        }
        t = imax(T7_CONTRAST_MIN, (2 * v) / 5);
    }
    memset(cnt, 0, sizeof cnt);
    if (rhi > rlo || lhi > llo) {
        win_begin(w);
        for (y = o->y; y < o->y + o->h; y++) {
            if (win_at(w, y, r))
                return VOCR_LINES_E_SOURCE;
            for (x = rlo; x < rhi; x++)
                if (contrast_rows(r, w->W, x) >= t)
                    cnt[x - rlo]++;
            for (x = llo; x < lhi; x++)
                if (contrast_rows(r, w->W, x) >= t)
                    cnt[1024 + (x - llo)]++;
        }
    }
    last = -1;
    gap = 0;
    for (X = rlo; X < rhi; X++) {
        if (cnt[X - rlo] >= 2) {
            last = X;
            gap = 0;
        } else if (++gap > gapmax) {
            break;
        }
    }
    if (last >= 0)
        x1 = imax(x1, imin(stop, last + 1 + pad));
    last = -1;
    gap = 0;
    for (X = lhi - 1; X >= llo; X--) {
        if (cnt[1024 + (X - llo)] >= 2) {
            last = X;
            gap = 0;
        } else if (++gap > gapmax) {
            break;
        }
    }
    if (last >= 0)
        x0 = imin(x0, imax(stopl, last - pad));
    b->x = x0;
    b->w = x1 - x0;
    return VOCR_T7_OK;
}

size_t vocr_textspot7_rows_arena_bytes_steps(unsigned max_width, unsigned max_height, unsigned steps)
{
    size_t r = 0, e = 0;
    if (!max_width || !max_height)
        return 0;
    if (steps & VOCR_T7_TIGHTEN)
        r = vocr_refine_arena_bytes(max_width, max_height);
    if (steps & VOCR_T7_EXTEND)
        e = 3u * (size_t)max_width;
    return r > e ? r : e;
}

size_t vocr_textspot7_rows_arena_bytes(unsigned max_width, unsigned max_height)
{
    return vocr_textspot7_rows_arena_bytes_steps(max_width, max_height, VOCR_T7_STEPS);
}

int vocr_textspot7_post_rows(const VocrT7Box *fb, unsigned n, int x0, int y0, int x1, int y1,
                             const VocrLinesRows *region, unsigned steps, void *arena, size_t arena_bytes,
                             VocrT7Box *out, unsigned *n_out)
{
    int W = x1 - x0, H = y1 - y0, rc;
    unsigned m = 0, i, k;
    if (!out || !n_out || (n && !fb) || n > VOCR_T7_MAX_BOXES || (steps & ~VOCR_T7_STEPS) || W <= 0 || H <= 0)
        return VOCR_T7_E_ARG;
    if (steps & (VOCR_T7_EXTEND | VOCR_T7_TIGHTEN)) {
        if (!region || !region->row || region->bpp != VOCR_LINES_GREY || (int)region->width != W ||
            (int)region->height != H)
            return VOCR_T7_E_ARG;
        if (!arena || arena_bytes < vocr_textspot7_rows_arena_bytes_steps(region->width, region->height, steps))
            return VOCR_T7_E_ARENA;
    }
    assign_join(fb, n, x0, y0, x1, y1, steps, out, &m);
    if (steps & VOCR_T7_EXTEND) {
        VocrT7Box orig[VOCR_T7_MAX_BOXES];
        RowWin w;
        w.src = region;
        w.ring = (unsigned char *)arena;
        w.W = W;
        w.H = H;
        w.last = -1;
        memcpy(orig, out, m * sizeof *out);
        for (i = 0; i < m; i++) {
            int stop = W, stopl = 0;
            for (k = 0; k < m; k++) {
                if (k == i || !same_row6(&orig[i], &orig[k]))
                    continue;
                if (orig[k].x >= orig[i].x + orig[i].w)
                    stop = imin(stop, orig[k].x);
                if (orig[k].x + orig[k].w <= orig[i].x)
                    stopl = imax(stopl, orig[k].x + orig[k].w);
            }
            rc = extend_one_rows(&w, &orig[i], stop, stopl, &out[i]);
            if (rc != VOCR_T7_OK)
                return rc;
        }
    }
    if (steps & VOCR_T7_TIGHTEN) {
        unsigned base = 0;
        while (base < m) {
            VocrLineSet set;
            unsigned c = m - base < VOCR_LINES_MAX ? m - base : VOCR_LINES_MAX;
            memset(&set, 0, sizeof set);
            for (i = 0; i < c; i++) {
                set.lines[i].x = (unsigned)out[base + i].x;
                set.lines[i].y = (unsigned)out[base + i].y;
                set.lines[i].w = (unsigned)out[base + i].w;
                set.lines[i].h = (unsigned)out[base + i].h;
                set.lines[i].dark = out[base + i].dark;
            }
            set.count = c;
            rc = vocr_refine_lines_rows(region, VOCR_REFINE_TIGHTEN, arena, arena_bytes, &set, NULL);
            if (rc == VOCR_LINES_E_SOURCE)
                return rc;
            if (rc == VOCR_LINES_E_ARENA)
                return VOCR_T7_E_ARENA;
            if (rc != VOCR_LINES_OK || set.count != c)
                return VOCR_T7_E_ARG;
            for (i = 0; i < c; i++) {
                out[base + i].x = (int)set.lines[i].x;
                out[base + i].y = (int)set.lines[i].y;
                out[base + i].w = (int)set.lines[i].w;
                out[base + i].h = (int)set.lines[i].h;
                out[base + i].dark = set.lines[i].dark;
            }
            base += c;
        }
    }
    *n_out = m;
    return VOCR_T7_OK;
}
