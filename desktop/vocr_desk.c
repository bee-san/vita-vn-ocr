/* vocr_desk.c - the desktop line detector in C (vocr_desk.h). C99 + libm; AVX2 + FMA kernels when the compiler
 * targets them, plain C otherwise. */
#ifdef VOCR_DESK_PROF
#define _POSIX_C_SOURCE 200112L
#endif
#include "vocr_desk.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#if defined(__AVX2__) && defined(__FMA__) && !defined(VOCR_DESK_SCALAR)
#include <immintrin.h>
#define DESK_AVX2 1
#else
#define DESK_AVX2 0
#endif

#define Q14 16384
#define ALIGN_UP(x, a) (((x) + (size_t)(a) - 1u) / (size_t)(a) * (size_t)(a))

/* size arithmetic that saturates at SIZE_MAX instead of wrapping: an arena that large fails to allocate or is
 * refused, never undersized (the sides are bounded first, so this only matters where size_t is 32 bits) */
static size_t sz_mul(size_t a, size_t b)
{
    return a && b > SIZE_MAX / a ? SIZE_MAX : a * b;
}

static size_t sz_add(size_t a, size_t b)
{
    return b > SIZE_MAX - a ? SIZE_MAX : a + b;
}

static size_t sz_align(size_t x, size_t a)
{
    size_t y = sz_add(x, a - 1);
    return y == SIZE_MAX ? SIZE_MAX : y / a * a;
}

/* ----------------------------------------------------------------------------- the blob */

static uint32_t rd32(const unsigned char *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static double rdf64(const unsigned char *p)
{
    uint64_t v = 0;
    double d;
    int i;
    for (i = 7; i >= 0; i--) v = v << 8 | p[i];
    memcpy(&d, &v, sizeof d);
    return d;
}

static uint32_t crc32_of(const unsigned char *p, size_t n)
{
    uint32_t c = 0xFFFFFFFFu;
    size_t i;
    int k;
    for (i = 0; i < n; i++) {
        c ^= p[i];
        for (k = 0; k < 8; k++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
    }
    return c ^ 0xFFFFFFFFu;
}

static int little_endian(void)
{
    const uint32_t one = 1;
    unsigned char b;
    memcpy(&b, &one, 1);
    return b == 1;
}

/* take n floats at *off (4-byte aligned in an aligned blob); 0 when the blob is too short */
static const float *take(const unsigned char *base, size_t bytes, size_t *off, size_t n)
{
    const float *p;
    if (*off + 4 * n > bytes) return NULL;
    p = (const float *)(const void *)(base + *off);
    *off += 4 * n;
    return p;
}

static int set_layer(vocr_desk_layer *L, const unsigned char *base, size_t bytes, size_t *off, size_t nw, unsigned cin,
                     unsigned cout, unsigned stride, unsigned dil, size_t *params)
{
    L->w = take(base, bytes, off, nw);
    L->b = take(base, bytes, off, cout);
    L->cin = cin;
    L->cout = cout;
    L->stride = stride;
    L->dil = dil;
    *params += nw + cout;
    return L->w && L->b ? 0 : VOCR_DESK_E_BLOB;
}

int vocr_desk_load(vocr_desk_model *m, const void *blob, size_t bytes)
{
    const unsigned char *b = (const unsigned char *)blob;
    size_t off = VOCR_DESK_HEADER, total;
    unsigned s, i, cin, cout, ch;
    if (!m || !b) return VOCR_DESK_E_ARG;
    memset(m, 0, sizeof *m);
    if (!little_endian() || ((uintptr_t)b & 3u) || bytes < VOCR_DESK_HEADER + 4 || memcmp(b, "VDSK", 4) ||
        rd32(b + 4) != VOCR_DESK_VERSION || rd32(b + 8) != VOCR_DESK_HEADER)
        return VOCR_DESK_E_BLOB;
    total = rd32(b + 12);
    if (total != bytes) return VOCR_DESK_E_BLOB;
    if (crc32_of(b, bytes - 4) != rd32(b + bytes - 4)) return VOCR_DESK_E_CRC;
    for (i = 0; i < 5; i++) m->c[i] = rd32(b + 16 + 4 * i);
    for (i = 0; i < 4; i++) m->n[i] = rd32(b + 36 + 4 * i);
    m->F = rd32(b + 52);
    m->tags = rd32(b + 56);
    m->min_cells = rd32(b + 60);
    m->size_kind = rd32(b + 64);
    m->size_n = rd32(b + 68);
    for (i = 0; i < 4; i++) {
        unsigned k;
        m->dil_len[i] = rd32(b + 72 + 4 * i);
        if (m->dil_len[i] < 1 || m->dil_len[i] > 4) return VOCR_DESK_E_BLOB;
        for (k = 0; k < 4; k++) m->dil[i][k] = b[88 + 4 * i + k];
    }
    m->conf_min_logit = rdf64(b + 104);
    m->pad_pct = rdf64(b + 112);
    m->pad_min = rdf64(b + 120);
    m->min_across = rdf64(b + 128);
    m->pad_frame = rd32(b + 136);
    m->pad_frame_min = rd32(b + 140);
    if (m->tags > VOCR_DESK_MAX_TAGS || !vocr_desk_policy_ok(m->size_kind, m->size_n) || m->F % 8 || !m->F ||
        m->F > VOCR_DESK_MAX_CHANNELS)
        return VOCR_DESK_E_BLOB;
    for (i = 0; i < 5; i++)
        if (!m->c[i] || m->c[i] % 8 || m->c[i] > VOCR_DESK_MAX_CHANNELS) return VOCR_DESK_E_BLOB;
    for (i = 0; i < 4; i++)
        if (m->n[i] > VOCR_DESK_MAX_BLOCKS) return VOCR_DESK_E_BLOB;
    if (set_layer(&m->stem, b, bytes, &off, 9u * m->c[0], 1, m->c[0], 2, 1, &m->params)) return VOCR_DESK_E_BLOB;
    for (s = 0; s < 4; s++) {
        cin = m->c[s];
        cout = m->c[s + 1];
        if (set_layer(&m->dw[s][0], b, bytes, &off, 9u * cin, cin, cin, 2, 1, &m->params) ||
            set_layer(&m->pw[s][0], b, bytes, &off, (size_t)cin * cout, cin, cout, 1, 1, &m->params))
            return VOCR_DESK_E_BLOB;
        for (i = 0; i < m->n[s]; i++) {
            unsigned d = m->dil[s][i % m->dil_len[s]];
            if (d < 1 || d > 16) return VOCR_DESK_E_BLOB;
            if (set_layer(&m->dw[s][1 + i], b, bytes, &off, 9u * cout, cout, cout, 1, d, &m->params) ||
                set_layer(&m->pw[s][1 + i], b, bytes, &off, (size_t)cout * cout, cout, cout, 1, 1, &m->params))
                return VOCR_DESK_E_BLOB;
        }
    }
    for (s = 0; s < 4; s++)
        if (set_layer(&m->lat[s], b, bytes, &off, (size_t)m->c[s + 1] * m->F, m->c[s + 1], m->F, 1, 1, &m->params))
            return VOCR_DESK_E_BLOB;
    if (set_layer(&m->sdw, b, bytes, &off, 9u * m->F, m->F, m->F, 1, 1, &m->params) ||
        set_layer(&m->spw, b, bytes, &off, (size_t)m->F * m->F, m->F, m->F, 1, 1, &m->params))
        return VOCR_DESK_E_BLOB;
    ch = 4 + m->tags;
    if (set_layer(&m->head, b, bytes, &off, (size_t)m->F * ch, m->F, ch, 1, 1, &m->params)) return VOCR_DESK_E_BLOB;
    if (off + 4 != bytes) return VOCR_DESK_E_BLOB;
    return VOCR_DESK_OK;
}

/* ----------------------------------------------------------------------------- the input */

int vocr_desk_policy_ok(unsigned kind, unsigned n)
{
    if (kind == VOCR_DESK_SIZE_NATIVE) return 1;
    if (kind != VOCR_DESK_SIZE_FIT && kind != VOCR_DESK_SIZE_LONG) return 0;
    return n >= 1 && n <= VOCR_DESK_MAX_SIDE;
}

static int frame_ok(unsigned w, unsigned h)
{
    return w && h && w <= VOCR_DESK_MAX_SIDE && h <= VOCR_DESK_MAX_SIDE;
}

void vocr_desk_input_size(unsigned kind, unsigned n, unsigned w, unsigned h, unsigned *iw, unsigned *ih,
                          unsigned *pw, unsigned *ph)
{
    double f = 1.0, lng = (double)(w > h ? w : h);
    unsigned a = 0, c = 0, pa = 0, pc = 0;
    if (vocr_desk_policy_ok(kind, n) && frame_ok(w, h)) {
        /* each side ends up at most max(n, the frame's side) <= 8192, a multiple of 32: the padding cannot wrap */
        if (kind == VOCR_DESK_SIZE_FIT) {
            f = (double)n / lng;
            if (f > 1.0) f = 1.0;
        } else if (kind == VOCR_DESK_SIZE_LONG) {
            f = (double)n / lng;
        }
        if (f == 1.0) {
            a = w;
            c = h;
        } else {
            double x = nearbyint((double)w * f), y = nearbyint((double)h * f);
            a = x < 1.0 ? 1u : (unsigned)x;
            c = y < 1.0 ? 1u : (unsigned)y;
        }
        pa = (a + VOCR_DESK_ALIGN - 1) / VOCR_DESK_ALIGN * VOCR_DESK_ALIGN;
        pc = (c + VOCR_DESK_ALIGN - 1) / VOCR_DESK_ALIGN * VOCR_DESK_ALIGN;
    }
    if (iw) *iw = a;
    if (ih) *ih = c;
    if (pw) *pw = pa;
    if (ph) *ph = pc;
}

size_t vocr_desk_model_bytes(void)
{
    return sizeof(vocr_desk_model);
}

size_t vocr_desk_set_bytes(void)
{
    return sizeof(vocr_desk_set);
}

void vocr_desk_grey(const unsigned char *rgb, unsigned w, unsigned h, unsigned stride, unsigned channels,
                    unsigned char *out)
{
    unsigned x, y;
    for (y = 0; y < h; y++) {
        const unsigned char *r = rgb + (size_t)y * stride;
        unsigned char *o = out + (size_t)y * w;
        for (x = 0; x < w; x++, r += channels)
            o[x] = (unsigned char)((77u * r[0] + 150u * r[1] + 29u * r[2]) >> 8);
    }
}

/* the taps of one axis: K per output, the source index and its Q14 weight; the weights of an output sum to 16384 */
static unsigned axis_taps(unsigned n, unsigned m)
{
    if (m == n) return 1;
    if (m < n) return n / m + 2;
    return 2;
}

static void axis_weights(unsigned n, unsigned m, unsigned K, int *idx, int *wt)
{
    unsigned i, k;
    for (i = 0; i < m; i++) {
        int *ix = idx + (size_t)i * K, *wv = wt + (size_t)i * K;
        for (k = 0; k < K; k++) {
            ix[k] = 0;
            wv[k] = 0;
        }
        if (m == n) {
            ix[0] = (int)i;
            wv[0] = Q14;
        } else if (m < n) {
            /* area: output i covers [i n, (i + 1) n) in units of 1 / m, source j covers [j m, (j + 1) m) */
            int64_t lo = (int64_t)i * n, hi = lo + n, cum = 0, prev = 0;
            int64_t j = lo / m;
            k = 0;
            for (; j * (int64_t)m < hi && k < K; j++, k++) {
                int64_t a = lo > j * (int64_t)m ? lo : j * (int64_t)m;
                int64_t b = hi < (j + 1) * (int64_t)m ? hi : (j + 1) * (int64_t)m;
                int64_t q;
                cum += b - a;
                q = (cum * 2 * Q14 + n) / (2 * (int64_t)n);
                ix[k] = (int)j;
                wv[k] = (int)(q - prev);
                prev = q;
            }
            for (; k < K; k++) ix[k] = ix[k - 1];
        } else {
            /* bilinear on pixel centres: x = ((2 i + 1) n - m) / (2 m) */
            int64_t num = (int64_t)(2 * i + 1) * n - m;
            if (num <= 0) {
                ix[0] = ix[1] = 0;
                wv[0] = Q14;
            } else {
                int64_t x0 = num / (2 * (int64_t)m), r = num - x0 * 2 * (int64_t)m;
                if (x0 >= (int64_t)n - 1) {
                    ix[0] = ix[1] = (int)n - 1;
                    wv[0] = Q14;
                } else {
                    int64_t w1 = (r * Q14 + m) / (2 * (int64_t)m);
                    ix[0] = (int)x0;
                    ix[1] = (int)x0 + 1;
                    wv[0] = (int)(Q14 - w1);
                    wv[1] = (int)w1;
                }
            }
        }
    }
}

size_t vocr_desk_resize_scratch(unsigned w, unsigned h, unsigned dw, unsigned dh)
{
    size_t kx, ky;
    if (!frame_ok(w, h) || !frame_ok(dw, dh)) return 0;
    kx = axis_taps(w, dw);
    ky = axis_taps(h, dh);
    return sizeof(int) * (2 * kx * dw + 2 * ky * dh + (size_t)h * dw);
}

int vocr_desk_resize(const unsigned char *src, unsigned w, unsigned h, unsigned stride, unsigned char *dst,
                     unsigned dw, unsigned dh, int *scratch)
{
    unsigned kx, ky, x, y, k;
    int *ixx, *wx, *iyy, *wy, *tmp;
    if (!src || !dst || !scratch || !frame_ok(w, h) || !frame_ok(dw, dh) || stride < w) return VOCR_DESK_E_ARG;
    kx = axis_taps(w, dw);
    ky = axis_taps(h, dh);
    ixx = scratch;
    wx = ixx + (size_t)kx * dw;
    iyy = wx + (size_t)kx * dw;
    wy = iyy + (size_t)ky * dh;
    tmp = wy + (size_t)ky * dh;
    axis_weights(w, dw, kx, ixx, wx);
    axis_weights(h, dh, ky, iyy, wy);
    for (y = 0; y < h; y++) {
        const unsigned char *r = src + (size_t)y * stride;
        int *t = tmp + (size_t)y * dw;
        for (x = 0; x < dw; x++) {
            const int *ii = ixx + (size_t)x * kx, *ww = wx + (size_t)x * kx;
            int acc = 0;
            for (k = 0; k < kx; k++) acc += r[ii[k]] * ww[k];
            t[x] = acc;
        }
    }
    for (y = 0; y < dh; y++) {
        const int *ii = iyy + (size_t)y * ky, *ww = wy + (size_t)y * ky;
        unsigned char *o = dst + (size_t)y * dw;
        for (x = 0; x < dw; x++) {
            int64_t acc = 0;
            for (k = 0; k < ky; k++) acc += (int64_t)tmp[(size_t)ii[k] * dw + x] * ww[k];
            acc = (acc + ((int64_t)1 << 27)) >> 28;
            o[x] = (unsigned char)(acc > 255 ? 255 : acc < 0 ? 0 : acc);
        }
    }
    return VOCR_DESK_OK;
}

/* ----------------------------------------------------------------------------- kernels */

/* out[p][co] = act(b[co] + sum_ci in[p][ci] w[ci][co] (+ res[p][co])); res may be out (read before written) */
static void pw_layer(const float *in, size_t P, const vocr_desk_layer *L, const float *res, int relu, float *out)
{
    const unsigned Cin = L->cin, Cout = L->cout;
    const float *w = L->w, *b = L->b;
    size_t p = 0;
#if DESK_AVX2
    if (Cout % 8 == 0) {
        const __m256 z = _mm256_setzero_ps();
        for (; p + 6 <= P; p += 6) {
            unsigned co = 0;
            const float *x0 = in + p * Cin;
            for (; co + 16 <= Cout; co += 16) {
                __m256 b0 = _mm256_loadu_ps(b + co), b1 = _mm256_loadu_ps(b + co + 8);
                __m256 a00 = b0, a01 = b1, a10 = b0, a11 = b1, a20 = b0, a21 = b1;
                __m256 a30 = b0, a31 = b1, a40 = b0, a41 = b1, a50 = b0, a51 = b1;
                unsigned ci, r;
                for (ci = 0; ci < Cin; ci++) {
                    const __m256 w0 = _mm256_loadu_ps(w + (size_t)ci * Cout + co);
                    const __m256 w1 = _mm256_loadu_ps(w + (size_t)ci * Cout + co + 8);
                    __m256 v;
                    v = _mm256_broadcast_ss(x0 + ci);
                    a00 = _mm256_fmadd_ps(v, w0, a00);
                    a01 = _mm256_fmadd_ps(v, w1, a01);
                    v = _mm256_broadcast_ss(x0 + Cin + ci);
                    a10 = _mm256_fmadd_ps(v, w0, a10);
                    a11 = _mm256_fmadd_ps(v, w1, a11);
                    v = _mm256_broadcast_ss(x0 + 2 * Cin + ci);
                    a20 = _mm256_fmadd_ps(v, w0, a20);
                    a21 = _mm256_fmadd_ps(v, w1, a21);
                    v = _mm256_broadcast_ss(x0 + 3 * Cin + ci);
                    a30 = _mm256_fmadd_ps(v, w0, a30);
                    a31 = _mm256_fmadd_ps(v, w1, a31);
                    v = _mm256_broadcast_ss(x0 + 4 * Cin + ci);
                    a40 = _mm256_fmadd_ps(v, w0, a40);
                    a41 = _mm256_fmadd_ps(v, w1, a41);
                    v = _mm256_broadcast_ss(x0 + 5 * Cin + ci);
                    a50 = _mm256_fmadd_ps(v, w0, a50);
                    a51 = _mm256_fmadd_ps(v, w1, a51);
                }
                {
                    __m256 acc[12];
                    acc[0] = a00, acc[1] = a01, acc[2] = a10, acc[3] = a11, acc[4] = a20, acc[5] = a21;
                    acc[6] = a30, acc[7] = a31, acc[8] = a40, acc[9] = a41, acc[10] = a50, acc[11] = a51;
                    for (r = 0; r < 6; r++) {
                        float *o = out + (p + r) * Cout + co;
                        __m256 u = acc[2 * r], v = acc[2 * r + 1];
                        if (res) {
                            u = _mm256_add_ps(u, _mm256_loadu_ps(res + (p + r) * Cout + co));
                            v = _mm256_add_ps(v, _mm256_loadu_ps(res + (p + r) * Cout + co + 8));
                        }
                        if (relu) {
                            u = _mm256_max_ps(u, z);
                            v = _mm256_max_ps(v, z);
                        }
                        _mm256_storeu_ps(o, u);
                        _mm256_storeu_ps(o + 8, v);
                    }
                }
            }
            for (; co < Cout; co += 8) {
                __m256 b0 = _mm256_loadu_ps(b + co), acc[6];
                unsigned ci, r;
                for (r = 0; r < 6; r++) acc[r] = b0;
                for (ci = 0; ci < Cin; ci++) {
                    const __m256 w0 = _mm256_loadu_ps(w + (size_t)ci * Cout + co);
                    for (r = 0; r < 6; r++)
                        acc[r] = _mm256_fmadd_ps(_mm256_broadcast_ss(x0 + r * Cin + ci), w0, acc[r]);
                }
                for (r = 0; r < 6; r++) {
                    __m256 u = acc[r];
                    if (res) u = _mm256_add_ps(u, _mm256_loadu_ps(res + (p + r) * Cout + co));
                    if (relu) u = _mm256_max_ps(u, z);
                    _mm256_storeu_ps(out + (p + r) * Cout + co, u);
                }
            }
        }
        for (; p < P; p++) {
            const float *x0 = in + p * Cin;
            unsigned co;
            for (co = 0; co < Cout; co += 8) {
                __m256 acc = _mm256_loadu_ps(b + co);
                unsigned ci;
                for (ci = 0; ci < Cin; ci++)
                    acc = _mm256_fmadd_ps(_mm256_broadcast_ss(x0 + ci), _mm256_loadu_ps(w + (size_t)ci * Cout + co), acc);
                if (res) acc = _mm256_add_ps(acc, _mm256_loadu_ps(res + p * Cout + co));
                if (relu) acc = _mm256_max_ps(acc, z);
                _mm256_storeu_ps(out + p * Cout + co, acc);
            }
        }
        return;
    }
#endif
    for (; p < P; p++) {
        float acc[VOCR_DESK_MAX_CHANNELS];
        const float *x0 = in + p * Cin;
        unsigned ci, co;
        for (co = 0; co < Cout; co++) acc[co] = b[co];
        for (ci = 0; ci < Cin; ci++) {
            const float v = x0[ci], *wr = w + (size_t)ci * Cout;
            for (co = 0; co < Cout; co++) acc[co] += v * wr[co];
        }
        for (co = 0; co < Cout; co++) {
            float u = acc[co];
            if (res) u += res[p * Cout + co];
            if (relu && u < 0.0f) u = 0.0f;
            out[p * Cout + co] = u;
        }
    }
}

/* one output pixel of a 3x3 depthwise layer, every tap checked against the map's borders (the border pixels) */
static void dw_pixel_checked(const float *in, unsigned H, unsigned W, unsigned C, const float *w, const float *b,
                             const int *iy, int ox0, unsigned d, float *o)
{
    unsigned c, ky, kx;
    for (c = 0; c < C; c++) o[c] = b[c];
    for (ky = 0; ky < 3; ky++) {
        const float *row;
        if (iy[ky] < 0 || iy[ky] >= (int)H) continue;
        row = in + (size_t)iy[ky] * W * C;
        for (kx = 0; kx < 3; kx++) {
            int ix = ox0 + (int)(kx * d);
            const float *px, *wk;
            if (ix < 0 || ix >= (int)W) continue;
            px = row + (size_t)ix * C;
            wk = w + (ky * 3 + kx) * C;
            for (c = 0; c < C; c++) o[c] += px[c] * wk[c];
        }
    }
    for (c = 0; c < C; c++)
        if (o[c] < 0.0f) o[c] = 0.0f;
}

/* 3x3 depthwise on an H x W x C map, stride s, dilation d, zero padding d, ReLU: out Ho x Wo x C. The pixels whose
 * three columns are inside the map take the unchecked path (32 channels in registers at a time with AVX2). */
static void dw_layer(const float *in, unsigned H, unsigned W, const vocr_desk_layer *L, float *out, unsigned Ho,
                     unsigned Wo, unsigned oy0, unsigned oy1)
{
    const unsigned C = L->cin, s = L->stride, d = L->dil;
    const float *w = L->w, *b = L->b;
    unsigned oy, ox, ky, lo, hi;
    (void)Ho;
    /* interior columns: ox s - d >= 0 and ox s + d <= W - 1 */
    lo = (d + s - 1) / s;
    hi = W > d ? (W - 1 - d) / s + 1 : 0;
    if (hi > Wo) hi = Wo;
    if (lo > hi) lo = hi;
    for (oy = oy0; oy < oy1; oy++) {
        int iy[3];
        const float *rows[3];
        unsigned nk = 0, kys[3];
        for (ky = 0; ky < 3; ky++) {
            iy[ky] = (int)(oy * s + ky * d) - (int)d;
            if (iy[ky] >= 0 && iy[ky] < (int)H) {
                kys[nk] = ky;
                rows[nk++] = in + (size_t)iy[ky] * W * C;
            }
        }
        for (ox = 0; ox < lo; ox++)
            dw_pixel_checked(in, H, W, C, w, b, iy, (int)(ox * s) - (int)d, d, out + ((size_t)oy * Wo + ox) * C);
        for (ox = hi; ox < Wo; ox++)
            dw_pixel_checked(in, H, W, C, w, b, iy, (int)(ox * s) - (int)d, d, out + ((size_t)oy * Wo + ox) * C);
        for (ox = lo; ox < hi; ox++) {
            float *o = out + ((size_t)oy * Wo + ox) * C;
            const size_t x0 = ((size_t)ox * s - d) * C, step = (size_t)d * C;
            unsigned c = 0, t;
#if DESK_AVX2
            const __m256 z = _mm256_setzero_ps();
            for (; c + 32 <= C; c += 32) {
                __m256 a0 = _mm256_loadu_ps(b + c), a1 = _mm256_loadu_ps(b + c + 8);
                __m256 a2 = _mm256_loadu_ps(b + c + 16), a3 = _mm256_loadu_ps(b + c + 24);
                for (t = 0; t < nk; t++) {
                    const float *r = rows[t] + x0 + c, *wk = w + kys[t] * 3 * C + c;
                    unsigned kx;
                    for (kx = 0; kx < 3; kx++, r += step, wk += C) {
                        a0 = _mm256_fmadd_ps(_mm256_loadu_ps(r), _mm256_loadu_ps(wk), a0);
                        a1 = _mm256_fmadd_ps(_mm256_loadu_ps(r + 8), _mm256_loadu_ps(wk + 8), a1);
                        a2 = _mm256_fmadd_ps(_mm256_loadu_ps(r + 16), _mm256_loadu_ps(wk + 16), a2);
                        a3 = _mm256_fmadd_ps(_mm256_loadu_ps(r + 24), _mm256_loadu_ps(wk + 24), a3);
                    }
                }
                _mm256_storeu_ps(o + c, _mm256_max_ps(a0, z));
                _mm256_storeu_ps(o + c + 8, _mm256_max_ps(a1, z));
                _mm256_storeu_ps(o + c + 16, _mm256_max_ps(a2, z));
                _mm256_storeu_ps(o + c + 24, _mm256_max_ps(a3, z));
            }
            for (; c + 8 <= C; c += 8) {
                __m256 a0 = _mm256_loadu_ps(b + c);
                for (t = 0; t < nk; t++) {
                    const float *r = rows[t] + x0 + c, *wk = w + kys[t] * 3 * C + c;
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(r), _mm256_loadu_ps(wk), a0);
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(r + step), _mm256_loadu_ps(wk + C), a0);
                    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(r + 2 * step), _mm256_loadu_ps(wk + 2 * C), a0);
                }
                _mm256_storeu_ps(o + c, _mm256_max_ps(a0, z));
            }
#endif
            for (; c < C; c++) {
                float acc = b[c];
                for (t = 0; t < nk; t++) {
                    const float *r = rows[t] + x0 + c, *wk = w + kys[t] * 3 * C + c;
                    acc += r[0] * wk[0];
                    acc += r[step] * wk[C];
                    acc += r[2 * step] * wk[2 * C];
                }
                o[c] = acc > 0.0f ? acc : 0.0f;
            }
        }
    }
}

/* the stem: 3x3 stride 2 padding 1 on the normalised grey (g - 128) / 128, 1 -> C, ReLU. xp: the normalised input
 * with a zero border, (H + 2) x (W + 2) floats */
static void stem_prep(const unsigned char *g, unsigned H, unsigned W, float *xp, unsigned y0, unsigned y1)
{
    /* rows y0 .. y1 - 1 of the (H + 2) x (W + 2) padded input */
    const unsigned XW = W + 2;
    unsigned x, y;
    for (y = y0; y < y1; y++) {
        float *r = xp + (size_t)y * XW;
        if (y == 0 || y == H + 1) {
            for (x = 0; x < XW; x++) r[x] = 0.0f;
            continue;
        }
        {
            const unsigned char *gr = g + (size_t)(y - 1) * W;
            r[0] = r[W + 1] = 0.0f;
            for (x = 0; x < W; x++) r[x + 1] = ((float)gr[x] - 128.0f) * 0.0078125f;
        }
    }
}

static void stem_layer(unsigned W, const vocr_desk_layer *L, float *out, unsigned Wo, const float *xp, unsigned oy0,
                       unsigned oy1)
{
    const unsigned C = L->cout, XW = W + 2;
    const float *w = L->w, *b = L->b;
    unsigned oy, ox, c;
    for (oy = oy0; oy < oy1; oy++) {
        const float *r0 = xp + (size_t)(2 * oy) * XW, *r1 = r0 + XW, *r2 = r1 + XW;
        for (ox = 0; ox < Wo; ox++) {
            const float *p0 = r0 + 2 * ox, *p1 = r1 + 2 * ox, *p2 = r2 + 2 * ox;
            float *o = out + ((size_t)oy * Wo + ox) * C;
            c = 0;
#if DESK_AVX2
            {
                const __m256 z = _mm256_setzero_ps();
                const __m256 v0 = _mm256_set1_ps(p0[0]), v1 = _mm256_set1_ps(p0[1]), v2 = _mm256_set1_ps(p0[2]);
                const __m256 v3 = _mm256_set1_ps(p1[0]), v4 = _mm256_set1_ps(p1[1]), v5 = _mm256_set1_ps(p1[2]);
                const __m256 v6 = _mm256_set1_ps(p2[0]), v7 = _mm256_set1_ps(p2[1]), v8 = _mm256_set1_ps(p2[2]);
                for (; c + 8 <= C; c += 8) {
                    __m256 a = _mm256_loadu_ps(b + c);
                    a = _mm256_fmadd_ps(v0, _mm256_loadu_ps(w + c), a);
                    a = _mm256_fmadd_ps(v1, _mm256_loadu_ps(w + C + c), a);
                    a = _mm256_fmadd_ps(v2, _mm256_loadu_ps(w + 2 * C + c), a);
                    a = _mm256_fmadd_ps(v3, _mm256_loadu_ps(w + 3 * C + c), a);
                    a = _mm256_fmadd_ps(v4, _mm256_loadu_ps(w + 4 * C + c), a);
                    a = _mm256_fmadd_ps(v5, _mm256_loadu_ps(w + 5 * C + c), a);
                    a = _mm256_fmadd_ps(v6, _mm256_loadu_ps(w + 6 * C + c), a);
                    a = _mm256_fmadd_ps(v7, _mm256_loadu_ps(w + 7 * C + c), a);
                    a = _mm256_fmadd_ps(v8, _mm256_loadu_ps(w + 8 * C + c), a);
                    _mm256_storeu_ps(o + c, _mm256_max_ps(a, z));
                }
            }
#endif
            for (; c < C; c++) {
                float a = b[c];
                a += p0[0] * w[c];
                a += p0[1] * w[C + c];
                a += p0[2] * w[2 * C + c];
                a += p1[0] * w[3 * C + c];
                a += p1[1] * w[4 * C + c];
                a += p1[2] * w[5 * C + c];
                a += p2[0] * w[6 * C + c];
                a += p2[1] * w[7 * C + c];
                a += p2[2] * w[8 * C + c];
                o[c] = a > 0.0f ? a : 0.0f;
            }
        }
    }
}

/* fine[y][x][c] += coarse[y / 2][x / 2][c] (nearest 2x up-sampling and addition) */
static void up_add(float *fine, unsigned W, const float *coarse, unsigned C, unsigned y0, unsigned y1)
{
    unsigned y, x, c, W2 = W / 2;
    for (y = y0; y < y1; y++) {
        const float *cr = coarse + (size_t)(y / 2) * W2 * C;
        float *fr = fine + (size_t)y * W * C;
        for (x = 0; x < W; x++) {
            const float *cp = cr + (size_t)(x / 2) * C;
            float *fp = fr + (size_t)x * C;
            for (c = 0; c < C; c++) fp[c] += cp[c];
        }
    }
}

/* ----------------------------------------------------------------------------- the network */

#ifdef VOCR_DESK_PROF
/* per-layer host timing for profiling builds only (-DVOCR_DESK_PROF): not thread safe, not in a release build */
#include <time.h>
double vocr_desk_prof_us[64];
static double prof_t0;
static unsigned prof_i;
static double prof_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1e6 * (double)ts.tv_sec + 1e-3 * (double)ts.tv_nsec;
}
#define PROF_START() (prof_i = 0, prof_t0 = prof_now())
#define PROF_MARK()                                                   \
    do {                                                              \
        double t_ = prof_now();                                       \
        if (prof_i < 64) vocr_desk_prof_us[prof_i++] += t_ - prof_t0; \
        prof_t0 = t_;                                                 \
    } while (0)
#else
#define PROF_START() ((void)0)
#define PROF_MARK() ((void)0)
#endif

typedef struct {
    size_t S, T, f[4], p[3], maps, xp, end;   /* float offsets */
} desk_plan;

static unsigned max3(unsigned a, unsigned b, unsigned c)
{
    unsigned m = a > b ? a : b;
    return m > c ? m : c;
}

/* every size saturates (sz_*): an input too large for size_t gives P->end == SIZE_MAX, never a wrapped plan */
static void plan(const vocr_desk_model *m, unsigned pw, unsigned ph, desk_plan *P)
{
    const size_t A = 8;   /* floats: 32 bytes */
    size_t H2 = ph / 2, W2 = pw / 2, H4 = ph / 4, W4 = pw / 4, off = 0, s0, p4;
    unsigned s, cmax = 0;
    s0 = sz_mul(sz_mul(H2, W2), m->c[0]);
    p4 = sz_mul(sz_mul(H4, W4), m->F);
    P->S = off;
    off = sz_add(off, sz_align(s0 > p4 ? s0 : p4, A));
    for (s = 0; s <= 4; s++)
        if (m->c[s] > cmax) cmax = m->c[s];
    P->T = off;
    off = sz_add(off, sz_align(sz_mul(sz_mul(H4, W4), max3(cmax, m->F, 1)), A));
    for (s = 0; s < 4; s++) {
        size_t hs = ph >> (2 + s), ws = pw >> (2 + s);
        P->f[s] = off;
        off = sz_add(off, sz_align(sz_mul(sz_mul(hs, ws), m->c[s + 1]), A));
    }
    for (s = 0; s < 3; s++) {   /* p8, p16, p32 */
        size_t hs = ph >> (3 + s), ws = pw >> (3 + s);
        P->p[s] = off;
        off = sz_add(off, sz_align(sz_mul(sz_mul(hs, ws), m->F), A));
    }
    P->maps = off;
    off = sz_add(off, sz_align(sz_mul(sz_mul(H4, W4), 4 + m->tags), A));
    P->xp = off;
    off = sz_add(off, sz_align(sz_mul((size_t)ph + 2, (size_t)pw + 2), A));
    P->end = off;
}

static size_t maps_bytes(const vocr_desk_model *m, unsigned pw, unsigned ph)
{
    desk_plan P;
    plan(m, pw, ph, &P);
    return sz_add(sz_mul(P.end, sizeof(float)), 32);
}

static float *aligned_floats(void *arena)
{
    uintptr_t a = ((uintptr_t)arena + 31u) & ~(uintptr_t)31u;
    return (float *)(void *)a;
}

/* one step of the network over a range of its rows (or pixels): the steps run one after the other, each split into
 * tasks that write disjoint outputs, so any number of tasks gives the same bits */
enum { ST_PREP, ST_STEM, ST_DW, ST_PW, ST_UP };

typedef struct {
    int kind, relu;
    const vocr_desk_layer *L;
    const unsigned char *g;
    const float *in, *res;
    float *out, *xp;
    unsigned H, W, Ho, Wo, C;
    size_t P;
} desk_step;

#define MAX_STEPS (8 + 4 * 2 * (1 + VOCR_DESK_MAX_BLOCKS) + 16)

typedef struct {
    desk_step st[MAX_STEPS];
    unsigned n, cur;
} desk_steps;

static void range_of(size_t n, unsigned task, unsigned ntasks, size_t *a, size_t *b)
{
    *a = n * task / ntasks;
    *b = n * (task + 1) / ntasks;
}

static void step_task(void *arg, unsigned task, unsigned ntasks)
{
    const desk_steps *S = (const desk_steps *)arg;
    const desk_step *t = &S->st[S->cur];
    size_t a, b;
    switch (t->kind) {
    case ST_PREP:
        range_of(t->H + 2, task, ntasks, &a, &b);
        stem_prep(t->g, t->H, t->W, t->xp, (unsigned)a, (unsigned)b);
        break;
    case ST_STEM:
        range_of(t->Ho, task, ntasks, &a, &b);
        stem_layer(t->W, t->L, t->out, t->Wo, t->xp, (unsigned)a, (unsigned)b);
        break;
    case ST_DW:
        range_of(t->Ho, task, ntasks, &a, &b);
        dw_layer(t->in, t->H, t->W, t->L, t->out, t->Ho, t->Wo, (unsigned)a, (unsigned)b);
        break;
    case ST_PW:
        range_of(t->P, task, ntasks, &a, &b);
        if (b > a)
            pw_layer(t->in + a * t->L->cin, b - a, t->L, t->res ? t->res + a * t->L->cout : NULL, t->relu,
                     t->out + a * t->L->cout);
        break;
    default:
        range_of(t->Ho, task, ntasks, &a, &b);
        up_add(t->out, t->Wo, t->in, t->C, (unsigned)a, (unsigned)b);
        break;
    }
}

static desk_step *add_step(desk_steps *S, int kind)
{
    desk_step *t = &S->st[S->n++];
    memset(t, 0, sizeof *t);
    t->kind = kind;
    return t;
}

static void add_pw(desk_steps *S, const float *in, size_t P, const vocr_desk_layer *L, const float *res, int relu,
                   float *out)
{
    desk_step *t = add_step(S, ST_PW);
    t->in = in, t->P = P, t->L = L, t->res = res, t->relu = relu, t->out = out;
}

static void add_dw(desk_steps *S, const float *in, unsigned H, unsigned W, const vocr_desk_layer *L, float *out,
                   unsigned Ho, unsigned Wo)
{
    desk_step *t = add_step(S, ST_DW);
    t->in = in, t->H = H, t->W = W, t->L = L, t->out = out, t->Ho = Ho, t->Wo = Wo;
}

static void add_up(desk_steps *S, float *fine, unsigned H, unsigned W, const float *coarse, unsigned C)
{
    desk_step *t = add_step(S, ST_UP);
    t->out = fine, t->Ho = H, t->Wo = W, t->in = coarse, t->C = C;
}

int vocr_desk_maps_mt(const vocr_desk_model *m, const unsigned char *in, unsigned pw, unsigned ph, void *arena,
                      size_t bytes, const float **maps, vocr_desk_runner run, void *ctx, unsigned ntasks)
{
    desk_plan P;
    desk_steps S;
    float *A, *Sb, *T, *f[4], *p[3], *P4, *out;
    unsigned s, i, Hs, Ws, Hin, Win;
    const float *x;
    desk_step *t;
    if (!m || !in || !arena || !maps || !frame_ok(pw, ph) || pw % VOCR_DESK_ALIGN || ph % VOCR_DESK_ALIGN)
        return VOCR_DESK_E_ARG;
    if (bytes < maps_bytes(m, pw, ph)) return VOCR_DESK_E_ARENA;
    if (!run || ntasks < 1) ntasks = 1;
    plan(m, pw, ph, &P);
    A = aligned_floats(arena);
    Sb = A + P.S;
    T = A + P.T;
    for (s = 0; s < 4; s++) f[s] = A + P.f[s];
    for (s = 0; s < 3; s++) p[s] = A + P.p[s];
    out = A + P.maps;
    S.n = 0;
    /* stem: H / 2 x W / 2 x c0 */
    t = add_step(&S, ST_PREP);
    t->g = in, t->H = ph, t->W = pw, t->xp = A + P.xp;
    t = add_step(&S, ST_STEM);
    t->W = pw, t->L = &m->stem, t->out = Sb, t->Ho = ph / 2, t->Wo = pw / 2, t->xp = A + P.xp;
    x = Sb;
    Hin = ph / 2;
    Win = pw / 2;
    for (s = 0; s < 4; s++) {
        Hs = ph >> (2 + s);
        Ws = pw >> (2 + s);
        add_dw(&S, x, Hin, Win, &m->dw[s][0], T, Hs, Ws);
        add_pw(&S, T, (size_t)Hs * Ws, &m->pw[s][0], NULL, 1, f[s]);
        for (i = 0; i < m->n[s]; i++) {
            add_dw(&S, f[s], Hs, Ws, &m->dw[s][1 + i], T, Hs, Ws);
            add_pw(&S, T, (size_t)Hs * Ws, &m->pw[s][1 + i], f[s], 1, f[s]);
        }
        x = f[s];
        Hin = Hs;
        Win = Ws;
    }
    /* the pyramid: p32 = lat3(f32); p16 = lat2(f16) + up(p32); p8 = lat1(f8) + up(p16); P4 = lat0(f4) + up(p8) */
    add_pw(&S, f[3], (size_t)(ph >> 5) * (pw >> 5), &m->lat[3], NULL, 1, p[2]);
    add_pw(&S, f[2], (size_t)(ph >> 4) * (pw >> 4), &m->lat[2], NULL, 1, p[1]);
    add_up(&S, p[1], ph >> 4, pw >> 4, p[2], m->F);
    add_pw(&S, f[1], (size_t)(ph >> 3) * (pw >> 3), &m->lat[1], NULL, 1, p[0]);
    add_up(&S, p[0], ph >> 3, pw >> 3, p[1], m->F);
    P4 = Sb;   /* the stem's output is no longer needed */
    add_pw(&S, f[0], (size_t)(ph >> 2) * (pw >> 2), &m->lat[0], NULL, 1, P4);
    add_up(&S, P4, ph >> 2, pw >> 2, p[0], m->F);
    add_dw(&S, P4, ph >> 2, pw >> 2, &m->sdw, T, ph >> 2, pw >> 2);
    add_pw(&S, T, (size_t)(ph >> 2) * (pw >> 2), &m->spw, P4, 1, P4);
    add_pw(&S, P4, (size_t)(ph >> 2) * (pw >> 2), &m->head, NULL, 0, out);
    PROF_START();
    for (S.cur = 0; S.cur < S.n; S.cur++) {
        if (ntasks > 1)
            run(ctx, step_task, &S, ntasks);
        else
            step_task(&S, 0, 1);
        PROF_MARK();
    }
    *maps = out;
    return VOCR_DESK_OK;
}

int vocr_desk_maps(const vocr_desk_model *m, const unsigned char *in, unsigned pw, unsigned ph, void *arena,
                   size_t bytes, const float **maps)
{
    return vocr_desk_maps_mt(m, in, pw, ph, arena, bytes, maps, NULL, NULL, 1);
}

/* ----------------------------------------------------------------------------- the decode */

#define MAX_MAP_SIDE (VOCR_DESK_MAX_SIDE / VOCR_DESK_STRIDE)

size_t vocr_desk_decode_bytes(unsigned mw, unsigned mh, unsigned tags)
{
    size_t cells = (size_t)mw * mh, labels = cells / 2 + 2;
    if (mw > MAX_MAP_SIDE || mh > MAX_MAP_SIDE || tags > VOCR_DESK_MAX_TAGS) return 0;
    /* labels per cell, the union-find parents, the final ids; per label 7 + tags doubles and 4 ints; candidates */
    return sizeof(int) * (cells + 2 * labels) + sizeof(double) * labels * (7 + tags) + sizeof(int) * labels * 4 +
           sizeof(vocr_desk_box) * labels + sizeof(int) * labels + 64;
}

static int uf_find(int *par, int a)
{
    while (par[a] != a) {
        par[a] = par[par[a]];
        a = par[a];
    }
    return a;
}

static int box_before(const vocr_desk_box *a, int ia, const vocr_desk_box *b, int ib)
{
    if (a->logit != b->logit) return a->logit > b->logit;
    if (a->y != b->y) return a->y < b->y;
    if (a->x != b->x) return a->x < b->x;
    return ia < ib;
}

int vocr_desk_decode(const vocr_desk_model *m, const float *maps, unsigned mw, unsigned mh, unsigned iw, unsigned ih,
                     void *scratch, size_t bytes, vocr_desk_set *out)
{
    const unsigned ch = 4 + m->tags, S = VOCR_DESK_STRIDE, T = m->tags;
    unsigned hm = (ih + S - 1) / S, wm = (iw + S - 1) / S, r, c, t;
    size_t cells, labels, i;
    unsigned char *base;
    int *lab, *par, *fin, *imin, *imax, *jmin, *jmax, *order, next = 1, n = 1;
    double *cnt, *sk, *so, *sn, *sf, *st;
    vocr_desk_box *cand;
    unsigned ncand = 0, k;
    if (!m || !maps || !scratch || !out || mw > MAX_MAP_SIDE || mh > MAX_MAP_SIDE || iw > VOCR_DESK_MAX_SIDE ||
        ih > VOCR_DESK_MAX_SIDE || hm > mh || wm > mw)
        return VOCR_DESK_E_ARG;
    if (bytes < vocr_desk_decode_bytes(mw, mh, T)) return VOCR_DESK_E_ARENA;
    cells = (size_t)hm * wm;
    labels = (size_t)mw * mh / 2 + 2;
    base = (unsigned char *)(((uintptr_t)scratch + 7u) & ~(uintptr_t)7u);
    cnt = (double *)(void *)base;
    sk = cnt + labels;
    so = sk + labels;
    sn = so + labels;
    sf = sn + labels;
    st = sf + labels;   /* labels x T */
    lab = (int *)(void *)(st + labels * (T ? T : 1));
    par = lab + (size_t)mw * mh;
    fin = par + labels;
    imin = fin + labels;
    imax = imin + labels;
    jmin = imax + labels;
    jmax = jmin + labels;
    cand = (vocr_desk_box *)(void *)(((uintptr_t)(jmax + labels) + 7u) & ~(uintptr_t)7u);
    order = (int *)(void *)(cand + labels);
    memset(out, 0, sizeof *out);
    out->in_w = iw;
    out->in_h = ih;
    /* pass 1: 4-connected components of K > 0, provisional labels in raster order, union by the smaller root */
    par[0] = 0;
    for (r = 0; r < hm; r++)
        for (c = 0; c < wm; c++) {
            size_t q = (size_t)r * wm + c;
            int up = 0, left = 0;
            if (!(maps[((size_t)r * mw + c) * ch] > 0.0f)) {
                lab[q] = 0;
                continue;
            }
            if (r && lab[q - wm]) up = lab[q - wm];
            if (c && lab[q - 1]) left = lab[q - 1];
            if (!up && !left) {
                par[next] = next;
                lab[q] = next++;
            } else if (up && left) {
                int a = uf_find(par, up), b = uf_find(par, left);
                lab[q] = a < b ? a : b;
                if (a != b) par[a > b ? a : b] = a < b ? a : b;
            } else {
                lab[q] = uf_find(par, up ? up : left);
            }
        }
    /* final labels in the order of each component's first cell (its smallest provisional label) */
    fin[0] = 0;
    for (k = 1; k < (unsigned)next; k++) {
        int root = uf_find(par, (int)k);
        fin[k] = root == (int)k ? n++ : 0;
    }
    for (k = 1; k < (unsigned)next; k++) fin[k] = fin[uf_find(par, (int)k)];
    for (k = 0; k < (unsigned)n; k++) {
        cnt[k] = sk[k] = so[k] = sn[k] = sf[k] = 0.0;
        for (t = 0; t < T; t++) st[(size_t)k * T + t] = 0.0;
        imin[k] = jmin[k] = 1 << 30;
        imax[k] = jmax[k] = -1;
    }
    for (i = 0; i < cells; i++)
        if (lab[i]) lab[i] = fin[lab[i]];
    /* pass 2: per-component sums over the cells in raster order (tools/vocr_desktop.decode's bincounts) */
    for (r = 0; r < hm; r++)
        for (c = 0; c < wm; c++) {
            int L = lab[(size_t)r * wm + c];
            const float *v = maps + ((size_t)r * mw + c) * ch;
            if (!L) continue;
            cnt[L] += 1.0;
            sk[L] += (double)v[0];
            so[L] += (double)v[1];
            for (t = 0; t < T; t++) st[(size_t)L * T + t] += (double)v[4 + t];
            if ((int)r < imin[L]) imin[L] = (int)r;
            if ((int)r > imax[L]) imax[L] = (int)r;
            if ((int)c < jmin[L]) jmin[L] = (int)c;
            if ((int)c > jmax[L]) jmax[L] = (int)c;
        }
    /* pass 3: the across extent, from the orientation of each cell's component */
    for (r = 0; r < hm; r++)
        for (c = 0; c < wm; c++) {
            int L = lab[(size_t)r * wm + c];
            const float *v = maps + ((size_t)r * mw + c) * ch;
            double cc;
            if (!L) continue;
            cc = (double)(S * ((so[L] / cnt[L]) > 0.0 ? c : r)) + (double)(S - 1) / 2.0;
            sn[L] += cc - (double)v[2];
            sf[L] += cc + (double)v[3];
        }
    for (k = 1; k < (unsigned)n; k++) {
        double logit = sk[k] / cnt[k], near = sn[k] / cnt[k], far = sf[k] / cnt[k], size, pad, lo, hi;
        double a0, a1, b0, b1, x0, x1, y0, y1, best = 0.0, second = -INFINITY;
        unsigned vertical = (so[k] / cnt[k]) > 0.0, tag = 0;
        vocr_desk_box *bx;
        if (cnt[k] < (double)m->min_cells || logit < m->conf_min_logit) continue;
        size = far - near;
        if (size < m->min_across) continue;
        if (vertical) {
            lo = (double)(S * (unsigned)imin[k]);
            hi = (double)(S * (unsigned)(imax[k] + 1));
        } else {
            lo = (double)(S * (unsigned)jmin[k]);
            hi = (double)(S * (unsigned)(jmax[k] + 1));
        }
        pad = size * m->pad_pct;
        if (m->pad_min > pad) pad = m->pad_min;
        a0 = near - pad;
        a1 = far + pad;
        b0 = lo - pad;
        b1 = hi + pad;
        if (vertical) {
            x0 = a0, x1 = a1, y0 = b0, y1 = b1;
        } else {
            x0 = b0, x1 = b1, y0 = a0, y1 = a1;
        }
        if (x0 < 0.0) x0 = 0.0;
        if (y0 < 0.0) y0 = 0.0;
        if (x1 > (double)iw) x1 = (double)iw;
        if (y1 > (double)ih) y1 = (double)ih;
        if (x1 <= x0 || y1 <= y0) continue;
        /* the tag: class 0 scores 0, class t + 1 the sum of tag map t; the first largest wins */
        for (t = 0; t < T; t++) {
            double sc = st[(size_t)k * T + t];
            if (sc > best) {
                best = sc;
                tag = t + 1;
            }
        }
        if (T) {
            if (tag != 0) second = 0.0;
            for (t = 0; t < T; t++) {
                double sc = st[(size_t)k * T + t];
                if (t + 1 != tag && sc > second) second = sc;
            }
        } else {
            second = 0.0;
        }
        bx = &cand[ncand];
        bx->x = x0;
        bx->y = y0;
        bx->w = x1 - x0;
        bx->h = y1 - y0;
        bx->logit = logit;
        bx->tconf = (best - second) / cnt[k];
        bx->vertical = vertical;
        bx->tag = tag;
        bx->cells = (unsigned)cnt[k];
        ncand++;
    }
    /* the first VOCR_DESK_MAX_BOXES in the stable order logit descending, then y, then x (then the candidate's index):
     * a sorted insertion into at most VOCR_DESK_MAX_BOXES slots, so linear in the candidates; it keeps the same boxes
     * in the same order as sorting every candidate and truncating (tools/vocr_desktop.decode) */
    {
        unsigned kept = 0;
        for (k = 0; k < ncand; k++) {
            unsigned j;
            if (kept == VOCR_DESK_MAX_BOXES &&
                !box_before(&cand[k], (int)k, &cand[order[kept - 1]], order[kept - 1]))
                continue;
            j = kept < VOCR_DESK_MAX_BOXES ? kept++ : kept - 1;
            while (j > 0 && box_before(&cand[k], (int)k, &cand[order[j - 1]], order[j - 1])) {
                order[j] = order[j - 1];
                j--;
            }
            order[j] = (int)k;
        }
        out->truncated = ncand > VOCR_DESK_MAX_BOXES;
        out->count = kept;
    }
    for (k = 0; k < out->count; k++) out->boxes[k] = cand[order[k]];
    return VOCR_DESK_OK;
}

/* ----------------------------------------------------------------------------- the detector */

typedef struct {
    size_t net, input, resized, rscratch, decode, end;
} detect_plan;

/* for a policy and frame the callers have checked (vocr_desk_policy_ok, frame_ok); the sizes saturate (sz_*) */
static void dplan(const vocr_desk_model *m, unsigned kind, unsigned n, unsigned w, unsigned h, detect_plan *D)
{
    unsigned iw, ih, pw, ph;
    size_t off = 0;
    vocr_desk_input_size(kind, n, w, h, &iw, &ih, &pw, &ph);
    D->net = off;
    off = sz_add(off, sz_align(maps_bytes(m, pw, ph), 64));
    D->input = off;
    off = sz_add(off, sz_align((size_t)pw * ph, 64));
    D->resized = off;
    off = sz_add(off, (iw != w || ih != h) ? sz_align((size_t)iw * ih, 64) : 0);
    D->rscratch = off;
    off = sz_add(off, (iw != w || ih != h) ? sz_align(vocr_desk_resize_scratch(w, h, iw, ih), 64) : 0);
    D->decode = off;
    off = sz_add(off, sz_align(vocr_desk_decode_bytes(pw / VOCR_DESK_STRIDE, ph / VOCR_DESK_STRIDE, m->tags), 64));
    D->end = sz_add(off, 64);
}

size_t vocr_desk_arena_bytes(const vocr_desk_model *m, unsigned kind, unsigned n, unsigned w, unsigned h)
{
    detect_plan D;
    if (!m || !frame_ok(w, h) || !vocr_desk_policy_ok(kind, n)) return 0;
    dplan(m, kind, n, w, h, &D);
    return D.end == SIZE_MAX ? 0 : D.end;
}

int vocr_desk_detect(const vocr_desk_model *m, const unsigned char *grey, unsigned w, unsigned h, unsigned stride,
                     unsigned kind, unsigned n, void *arena, size_t bytes, vocr_desk_set *out)
{
    return vocr_desk_detect_mt(m, grey, w, h, stride, kind, n, arena, bytes, out, NULL, NULL, 1);
}

int vocr_desk_detect_mt(const vocr_desk_model *m, const unsigned char *grey, unsigned w, unsigned h, unsigned stride,
                        unsigned kind, unsigned n, void *arena, size_t bytes, vocr_desk_set *out, vocr_desk_runner run,
                        void *ctx, unsigned ntasks)
{
    detect_plan D;
    unsigned iw, ih, pw, ph, y, x, k, keep = 0;
    unsigned char *base, *in, *src;
    const float *maps;
    double fx, fy;
    int rc;
    if (!m || !grey || !arena || !out || !frame_ok(w, h) || stride < w || !vocr_desk_policy_ok(kind, n))
        return VOCR_DESK_E_ARG;
    dplan(m, kind, n, w, h, &D);
    if (D.end == SIZE_MAX) return VOCR_DESK_E_ARG;
    if (bytes < D.end) return VOCR_DESK_E_ARENA;
    vocr_desk_input_size(kind, n, w, h, &iw, &ih, &pw, &ph);
    base = (unsigned char *)(((uintptr_t)arena + 63u) & ~(uintptr_t)63u);
    in = base + D.input;
    if (iw != w || ih != h) {
        src = base + D.resized;
        rc = vocr_desk_resize(grey, w, h, stride, src, iw, ih, (int *)(void *)(base + D.rscratch));
        if (rc) return rc;
        stride = iw;
    } else {
        src = (unsigned char *)(uintptr_t)grey;
    }
    for (y = 0; y < ih; y++) {
        const unsigned char *r = src + (size_t)y * stride;
        unsigned char *o = in + (size_t)y * pw;
        memcpy(o, r, iw);
        for (x = iw; x < pw; x++) o[x] = r[iw - 1];
    }
    for (y = ih; y < ph; y++) memcpy(in + (size_t)y * pw, in + (size_t)(ih - 1) * pw, pw);
    rc = vocr_desk_maps_mt(m, in, pw, ph, base + D.net, D.input - D.net, &maps, run, ctx, ntasks);
    if (rc) return rc;
    rc = vocr_desk_decode(m, maps, pw / VOCR_DESK_STRIDE, ph / VOCR_DESK_STRIDE, iw, ih, base + D.decode,
                          D.end - 64 - D.decode, out);
    if (rc) return rc;
    /* to frame px (tools/vocr_desktop.Detector.detect), then the frame pad when the blob asks for it */
    fx = (double)iw / w;
    fy = (double)ih / h;
    {
        double fp = 0.0;
        if (m->pad_frame) {
            fp = nearbyint((double)m->pad_frame * (double)h / 544.0);
            if (fp < (double)m->pad_frame_min) fp = (double)m->pad_frame_min;
        }
        for (k = 0; k < out->count; k++) {
            vocr_desk_box b = out->boxes[k];
            double x0 = b.x / fx - fp, y0 = b.y / fy - fp, x1 = (b.x + b.w) / fx + fp, y1 = (b.y + b.h) / fy + fp;
            if (x0 < 0.0) x0 = 0.0;
            if (y0 < 0.0) y0 = 0.0;
            if (x1 > (double)w) x1 = (double)w;
            if (y1 > (double)h) y1 = (double)h;
            if (!(x1 > x0 && y1 > y0)) continue;
            b.x = x0;
            b.y = y0;
            b.w = x1 - x0;
            b.h = y1 - y0;
            out->boxes[keep++] = b;
        }
    }
    out->count = keep;
    return VOCR_DESK_OK;
}
