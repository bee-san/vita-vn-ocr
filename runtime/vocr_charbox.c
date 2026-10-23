/* vocr_charbox.c - per-character positions from the recognizer's CTC frames (vocr_charbox.h, issue #114).
 * eval/charbox.py is the specification; every number below is computed as it computes it. */
#include "vocr_charbox.h"

#include <string.h>

#define Q ((int64_t)VOCR_CHARBOX_Q)
#define PITCH_MIN 4
#define PITCH_TOL 4
#define INK_FRAC 4

/* round(a / b), half away from zero, b > 0 (eval/charbox.py _div_round) */
static int64_t div_round(int64_t a, int64_t b)
{
    return a >= 0 ? (a + b / 2) / b : -((-a + b / 2) / b);
}

static int64_t clamp64(int64_t v, int64_t lo, int64_t hi) { return v < lo ? lo : v > hi ? hi : v; }

/* the one code point of a class's UTF-8 text, or 0 (empty, several, or not valid UTF-8) */
static uint32_t one_cp(const unsigned char *s, unsigned n)
{
    uint32_t cp;
    unsigned need, i;
    if (!n) return 0;
    if (s[0] < 0x80) cp = s[0], need = 0;
    else if ((s[0] & 0xE0) == 0xC0) cp = s[0] & 0x1Fu, need = 1;
    else if ((s[0] & 0xF0) == 0xE0) cp = s[0] & 0x0Fu, need = 2;
    else if ((s[0] & 0xF8) == 0xF0) cp = s[0] & 0x07u, need = 3;
    else return 0;
    if (n != need + 1) return 0;
    for (i = 1; i <= need; i++) {
        if ((s[i] & 0xC0) != 0x80) return 0;
        cp = cp << 6 | (s[i] & 0x3Fu);
    }
    return cp;
}

static int in_list(uint32_t cp, const uint16_t *list, unsigned n)
{
    unsigned i;
    for (i = 0; i < n; i++)
        if (list[i] == cp) return 1;
    return 0;
}

unsigned vocr_charbox_group(uint32_t cp)
{
    /* eval/charbox.py _CLOSE, _OPEN, _DOTS, _SMALL, _LONG, _MARK */
    static const uint16_t close_[] = {0x3001, 0x3002, 0xFF0C, 0xFF0E, 0x300D, 0x300F, 0xFF09, 0x3011, 0x3009,
                                      0x300B, 0x3015, 0xFF5D, 0xFF3D, 0x3019, 0x3017, 0x201D, 0x2019};
    static const uint16_t open_[] = {0x300C, 0x300E, 0xFF08, 0x3010, 0x3008, 0x300A, 0x3014, 0xFF5B, 0xFF3B,
                                     0x3018, 0x3016, 0x201C, 0x2018};
    static const uint16_t dots[] = {0x2026, 0x2025, 0x30FB};
    static const uint16_t small[] = {0x3041, 0x3043, 0x3045, 0x3047, 0x3049, 0x3063, 0x3083, 0x3085, 0x3087,
                                     0x308E, 0x3095, 0x3096, 0x30A1, 0x30A3, 0x30A5, 0x30A7, 0x30A9, 0x30C3,
                                     0x30E3, 0x30E5, 0x30E7, 0x30EE, 0x30F5, 0x30F6, 0x31F0, 0x31F1, 0x31F2,
                                     0x31F3, 0x31F4, 0x31F5, 0x31F6, 0x31F7, 0x31F8, 0x31F9, 0x31FA, 0x31FB,
                                     0x31FC, 0x31FD, 0x31FE, 0x31FF};
    static const uint16_t long_[] = {0x30FC, 0x2015, 0x2500, 0xFF5E, 0x301C, 0xFF0D};
    static const uint16_t mark[] = {0xFF01, 0xFF1F, 0x266A, 0x2606, 0x2605, 0x2665, 0x2661, 0x203B};
    if (!cp) return VOCR_CHARBOX_G_OTHER;
    if (in_list(cp, close_, sizeof close_ / sizeof close_[0])) return VOCR_CHARBOX_G_CLOSE;
    if (in_list(cp, open_, sizeof open_ / sizeof open_[0])) return VOCR_CHARBOX_G_OPEN;
    if (in_list(cp, dots, sizeof dots / sizeof dots[0])) return VOCR_CHARBOX_G_DOTS;
    if (in_list(cp, small, sizeof small / sizeof small[0])) return VOCR_CHARBOX_G_SMALL;
    if (in_list(cp, long_, sizeof long_ / sizeof long_[0])) return VOCR_CHARBOX_G_LONG;
    if (in_list(cp, mark, sizeof mark / sizeof mark[0])) return VOCR_CHARBOX_G_MARK;
    if (cp < 0x80 || (cp >= 0xFF61 && cp <= 0xFF9F)) return VOCR_CHARBOX_G_HALF;
    return VOCR_CHARBOX_G_OTHER;
}

static int check_img(const vocr_image *im, unsigned w, unsigned h)
{
    int fmt;
    if (!im || !im->pixels || im->width != w || im->height != h) return VOCR_E_ARG;
    fmt = im->format & ~VOCR_INVERT;
    if (fmt != VOCR_GREY8 && fmt != VOCR_RGBA8) return VOCR_E_ARG;
    if (im->stride < (size_t)w * (fmt == VOCR_GREY8 ? 1u : 4u)) return VOCR_E_ARG;
    return VOCR_OK;
}

static int grey_at(const vocr_image *im, unsigned x, unsigned y)
{
    const unsigned char *p = im->pixels + (size_t)y * im->stride;
    if ((im->format & ~VOCR_INVERT) == VOCR_RGBA8) {
        p += 4u * x;
        return (77 * p[0] + 150 * p[1] + 29 * p[2]) >> 8;
    }
    return p[x];
}

static uint32_t column_ink(const vocr_image *im, unsigned x)
{
    uint32_t s = 0;
    unsigned y;
    int prev = grey_at(im, x, 0);
    for (y = 1; y < im->height; y++) {
        const int g = grey_at(im, x, y);
        s += (uint32_t)(g > prev ? g - prev : prev - g);
        prev = g;
    }
    return s;
}

int vocr_charbox_profile(const vocr_image *img, uint32_t *prof)
{
    unsigned x, y;
    if (!img || !prof || check_img(img, img->width, img->height)) return VOCR_E_ARG;
    memset(prof, 0, (size_t)img->width * sizeof *prof);
    if ((img->format & ~VOCR_INVERT) == VOCR_GREY8) {   /* row by row: the rows are contiguous */
        for (y = 1; y < img->height; y++) {
            const unsigned char *a = img->pixels + (size_t)(y - 1) * img->stride, *b = a + img->stride;
            for (x = 0; x < img->width; x++) prof[x] += (uint32_t)(b[x] > a[x] ? b[x] - a[x] : a[x] - b[x]);
        }
        return VOCR_OK;
    }
    for (y = 1; y < img->height; y++)
        for (x = 0; x < img->width; x++) {
            const int g0 = grey_at(img, x, y - 1), g1 = grey_at(img, x, y);
            prof[x] += (uint32_t)(g1 > g0 ? g1 - g0 : g0 - g1);
        }
    return VOCR_OK;
}

typedef struct {
    const vocr_image *img;
    const uint32_t *prof;
} ink_t;

static uint32_t ink_at(const ink_t *k, unsigned x) { return k->prof ? k->prof[x] : column_ink(k->img, x); }

/* the k-th smallest (0-based) of the n - 1 spacings c[i + 1] - c[i], without a copy: O(n^2) */
static int64_t kth_spacing(const vocr_charbox *o, unsigned n, unsigned k)
{
    unsigned i, j;
    for (i = 0; i + 1 < n; i++) {
        const int64_t v = (int64_t)o[i + 1].c - o[i].c;
        unsigned less = 0, leq = 0;
        for (j = 0; j + 1 < n; j++) {
            const int64_t u = (int64_t)o[j + 1].c - o[j].c;
            less += u < v;
            leq += u <= v;
        }
        if (less <= k && k < leq) return v;
    }
    return 0; /* not reached for n >= 2 */
}

static void ink_refine(vocr_charbox *o, unsigned n, unsigned w, const ink_t *k)
{
    uint32_t peak = 0, thr;
    unsigned i, x;
    for (x = 0; x < w; x++) {
        const uint32_t v = ink_at(k, x);
        if (v > peak) peak = v;
    }
    if (!peak) return;
    thr = peak / INK_FRAC;
    for (i = 0; i + 1 < n; i++) {
        const int64_t a = (o[i].c + Q - 1) / Q, b = o[i + 1].c / Q; /* whole columns between the two centres */
        uint32_t mn = 0;
        int64_t best_len = 0, best_s = a, run_s = -1, cut;
        if (b - a < 1) continue;
        for (x = (unsigned)a; x < (unsigned)b; x++) {
            const uint32_t v = ink_at(k, x);
            if (x == (unsigned)a || v < mn) mn = v;
        }
        for (x = (unsigned)a; x <= (unsigned)b; x++) { /* the first longest run of minimal columns */
            const int on = x < (unsigned)b && ink_at(k, x) == mn;
            if (on && run_s < 0) run_s = x;
            if (!on && run_s >= 0) {
                if ((int64_t)x - run_s > best_len) {
                    best_len = (int64_t)x - run_s;
                    best_s = run_s;
                }
                run_s = -1;
            }
        }
        cut = best_s * Q + best_len * Q / 2;
        o[i].x1 = o[i].i1 = (int32_t)cut;
        o[i + 1].x0 = o[i + 1].i0 = (int32_t)cut;
    }
    for (i = 0; i < n; i++) {
        const int64_t a = o[i].x0 / Q;
        const int64_t b0 = (o[i].x1 + Q - 1) / Q, b = b0 < (int64_t)w ? b0 : (int64_t)w;
        int64_t first = -1, last = -1, xx;
        for (xx = a; xx < b; xx++)
            if (ink_at(k, (unsigned)xx) > thr) {
                if (first < 0) first = xx;
                last = xx;
            }
        if (first >= 0) {
            const int64_t i0 = o[i].x0 > first * Q ? o[i].x0 : first * Q;
            const int64_t i1 = o[i].x1 < (last + 1) * Q ? o[i].x1 : (last + 1) * Q;
            if (i1 > i0) {
                o[i].i0 = (int32_t)i0;
                o[i].i1 = (int32_t)i1;
                o[i].c = (int32_t)((i0 + i1) / 2);
            }
        }
    }
}

int vocr_charbox_find(const vocr_model *m, const uint16_t *cls, unsigned cls_stride, unsigned frames, unsigned w,
                      unsigned h, unsigned flags, const vocr_charbox_cal *cal, const vocr_image *img,
                      const uint32_t *prof, vocr_charbox *out, unsigned cap, unsigned *n)
{
    unsigned f, i, cnt = 0, nw, last = 0;
    int64_t pitch, wq;
    ink_t ink;
    if (n) *n = 0;
    if (!m || !m->blob || !cls || !cls_stride || !frames || !w || !h || !out || !n || !cap ||
        cap > VOCR_CHARBOX_MAX || (flags & ~(unsigned)(VOCR_CHARBOX_RUN | VOCR_CHARBOX_PITCH | VOCR_CHARBOX_INK)))
        return VOCR_E_ARG;
    if ((flags & VOCR_CHARBOX_INK) && !prof && check_img(img, w, h)) return VOCR_E_ARG;
    nw = vocr_resized_width(m, w, h);
    if (!nw) return VOCR_E_ARG;
    wq = (int64_t)w * Q;
    for (f = 0; f < frames; f++) {           /* greedy CTC: the runs of the emitted characters */
        const unsigned c = cls[(size_t)f * cls_stride];
        if (c >= m->n_classes) return VOCR_E_ARG;
        if (c && c != last) {
            if (cnt == cap) break;
            out[cnt].cls = (uint16_t)c;
            out[cnt].f0 = out[cnt].f1 = (uint16_t)f;
            cnt++;
        } else if (c && c == last) {
            out[cnt - 1].f1 = (uint16_t)f;
        }
        last = c;
    }
    *n = cnt;   /* at a break the last kept character's run has ended: frame f starts a new character */
    if (!cnt) return VOCR_OK;
    for (i = 0; i < cnt; i++) {
        unsigned len;
        const char *t = vocr_class_text(m, out[i].cls, &len);
        const unsigned g = vocr_charbox_group(one_cp((const unsigned char *)t, len));
        const int64_t s = (int64_t)m->stride;
        int64_t r = flags & VOCR_CHARBOX_RUN ? Q * s * ((int64_t)out[i].f0 + out[i].f1 + 1) / 2
                                             : Q * (s * out[i].f0 + s / 2);
        out[i].group = (uint16_t)g;
        if (cal) r += cal->shift + cal->group[g];
        out[i].c = (int32_t)clamp64(div_round(r * (int64_t)w, (int64_t)nw), 0, wq);
    }
    pitch = cnt > 1 ? kth_spacing(out, cnt, (cnt - 2) / 2) : (int64_t)h * Q;
    if (pitch <= 0) pitch = (int64_t)h * Q;
    if ((flags & VOCR_CHARBOX_PITCH) && cnt >= PITCH_MIN) {
        int regular = 1;
        for (i = 0; i + 1 < cnt; i++) {
            const int64_t d = (int64_t)out[i + 1].c - out[i].c, e = d > pitch ? d - pitch : pitch - d;
            if (e * PITCH_TOL > pitch) regular = 0;
        }
        if (regular) {                       /* least squares of c_k = a + p k, k = 0 .. n - 1 */
            const int64_t nn = cnt, sk = nn * (nn - 1) / 2, skk = (nn - 1) * nn * (2 * nn - 1) / 6;
            int64_t sc = 0, skc = 0, den, p_num, a_num;
            for (i = 0; i < cnt; i++) {
                sc += out[i].c;
                skc += (int64_t)i * out[i].c;
            }
            den = nn * skk - sk * sk;
            p_num = nn * skc - sk * sc;
            a_num = sc * skk - sk * skc;
            for (i = 0; i < cnt; i++) out[i].c = (int32_t)clamp64(div_round(a_num + (int64_t)i * p_num, den), 0, wq);
            pitch = div_round(p_num, den);
            if (pitch < 1) pitch = 1;
        }
    }
    for (i = 0; i < cnt; i++) {               /* cells: midpoints, half a pitch past the ends */
        const int64_t lo = i ? div_round((int64_t)out[i - 1].c + out[i].c, 2) : clamp64(out[0].c - pitch / 2, 0, wq);
        const int64_t hi = i + 1 < cnt ? div_round((int64_t)out[i].c + out[i + 1].c, 2)
                                       : (out[i].c + pitch / 2 < wq ? out[i].c + pitch / 2 : wq);
        out[i].x0 = out[i].i0 = (int32_t)lo;
        out[i].x1 = out[i].i1 = (int32_t)(hi > lo ? hi : lo);
    }
    if (flags & VOCR_CHARBOX_INK) {
        ink.img = img;
        ink.prof = prof;
        ink_refine(out, cnt, w, &ink);
    }
    return VOCR_OK;
}

int vocr_charbox_align(const uint32_t *src, unsigned n, const uint32_t *dst, unsigned m, int16_t *map)
{
    unsigned i = 0, j = 0, t;
    if ((!src && n) || (!dst && m) || (!map && m) || n > 32767u || m > 32767u) return VOCR_E_ARG;
    for (t = 0; t < m; t++) map[t] = -1;
    while (i < n && j < m) {
        unsigned s, bdi = 0, bdj = 0, found = 0;
        if (src[i] == dst[j]) {
            map[j++] = (int16_t)i++;
            continue;
        }
        for (s = 1; s <= 2u * VOCR_CHARBOX_SYNC && !found; s++) {
            unsigned di, bkey = 0;
            const unsigned lo = s > VOCR_CHARBOX_SYNC ? s - VOCR_CHARBOX_SYNC : 0u;
            const unsigned hi = s < VOCR_CHARBOX_SYNC ? s : VOCR_CHARBOX_SYNC;
            for (di = lo; di <= hi; di++) {
                const unsigned dj = s - di, key = (di > dj ? di - dj : dj - di) * 64u + dj;
                if (i + di >= n || j + dj >= m || src[i + di] != dst[j + dj]) continue;
                if (!(i + di + 1 >= n || j + dj + 1 >= m || src[i + di + 1] == dst[j + dj + 1])) continue;
                if (!found || key < bkey) {
                    found = 1;
                    bkey = key;
                    bdi = di;
                    bdj = dj;
                }
            }
        }
        if (!found) {
            map[j++] = (int16_t)i++;
            continue;
        }
        for (t = 0; t < bdi && t < bdj; t++) map[j + t] = (int16_t)(i + t);
        i += bdi;
        j += bdj;
    }
    return VOCR_OK;
}
