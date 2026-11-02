/* vocr_tsd_resize.c - textspot-desktop's frame luma, input policy and integer resize (vocr_tsd_resize.h).
 *
 * The rules are desktop/vocr_desk.c's (vocr_desk_grey, vocr_desk_input_size, vocr_desk_resize) and tools/tsdesk.resize's
 * bit for bit: the luma (77 R + 150 G + 29 B) >> 8, Q14 area weights when a side shrinks, bilinear on pixel centres
 * when it grows, the horizontal sums then the vertical ones, (acc + 2^27) >> 28, clamped. This file adds SIMD paths
 * (AVX2, when the compiler targets it; -DVOCR_TSD_SCALAR forces the plain-C ones) that compute the same integers: a
 * pmaddubsw luma, the identity / 2:1 / polyphase horizontal passes, exact 1- / 2-tap and int32-split vertical sums, the
 * exact 2x enlargement and the quarter-weight shrinks in 16-bit lanes, a ring of resized rows, and the same resize in
 * bands of output rows for threads. tests/test_tsdesk_c.py checks them against a numpy model of the rules and against
 * vocr_desk_resize / vocr_desk_grey. C99, no allocation, no dependency.
 */
#include "vocr_tsd_resize.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "vocr_tsd.h"

#if defined(__AVX2__) && defined(__FMA__) && !defined(VOCR_TSD_SCALAR)
#include <immintrin.h>
#define RS_AVX2 1
#else
#define RS_AVX2 0
#endif

#define Q14 16384
#define ALIGN_UP(x, a) (((x) + (size_t)(a) - 1u) / (size_t)(a) * (size_t)(a))

void vocr_tsd_input_size(unsigned kind, unsigned n, unsigned w, unsigned h, unsigned *iw, unsigned *ih,
                          unsigned *pw, unsigned *ph)
{
    double f = 1.0, lng = (double)(w > h ? w : h);
    unsigned a, c;
    if (kind == VOCR_TSD_SIZE_FIT && n && lng > 0) {
        f = (double)n / lng;
        if (f > 1.0) f = 1.0;
    } else if (kind == VOCR_TSD_SIZE_LONG && n && lng > 0) {
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
    if (iw) *iw = a;
    if (ih) *ih = c;
    if (pw) *pw = (a + VOCR_TSD_ALIGN - 1) / VOCR_TSD_ALIGN * VOCR_TSD_ALIGN;
    if (ph) *ph = (c + VOCR_TSD_ALIGN - 1) / VOCR_TSD_ALIGN * VOCR_TSD_ALIGN;
}

void vocr_tsd_grey(const unsigned char *rgb, unsigned w, unsigned h, unsigned stride, unsigned channels,
                    unsigned char *out)
{
    unsigned x, y;
    for (y = 0; y < h; y++) {
        const unsigned char *r = rgb + (size_t)y * stride;
        unsigned char *o = out + (size_t)y * w;
        x = 0;
#if RS_AVX2
        /* 16 pixels a step, the same integer formula: 77 R + 150 G + 29 B <= 65280 is exact in 16-bit lanes */
        if (channels == 3) {
            /* 16 pixels a step: four 4-pixel groups (12 bytes) into the 128-bit lanes, one byte shuffle to [R B] x 4
             * [G 0] x 4 per lane, pmaddubsw by [77 29] / [75 0] (77 R + 29 B <= 26520, 75 G <= 19125: no saturation),
             * then (77 R + 29 B) + 2 (75 G) <= 64770 as unsigned 16 bits, >> 8: the formula's bits (all 2^24 colours
             * checked) */
            const __m256i sh = _mm256_setr_epi8(0, 2, 3, 5, 6, 8, 9, 11, 1, -1, 4, -1, 7, -1, 10, -1,
                                                0, 2, 3, 5, 6, 8, 9, 11, 1, -1, 4, -1, 7, -1, 10, -1);
            const __m256i wt = _mm256_setr_epi8(77, 29, 77, 29, 77, 29, 77, 29, 75, 0, 75, 0, 75, 0, 75, 0,
                                                77, 29, 77, 29, 77, 29, 77, 29, 75, 0, 75, 0, 75, 0, 75, 0);
            for (; x + 16 <= w && 3 * (size_t)x + 52 <= 3 * (size_t)w; x += 16) {   /* reads 52 bytes */
                const unsigned char *p = r + 3 * (size_t)x;
                const __m256i u = _mm256_loadu2_m128i((const __m128i *)(const void *)(p + 12),
                                                      (const __m128i *)(const void *)p);
                const __m256i v = _mm256_loadu2_m128i((const __m128i *)(const void *)(p + 36),
                                                      (const __m128i *)(const void *)(p + 24));
                __m256i tu = _mm256_maddubs_epi16(_mm256_shuffle_epi8(u, sh), wt);   /* [rb0..3 g0..3 | rb4..7 g4..7] */
                __m256i tv = _mm256_maddubs_epi16(_mm256_shuffle_epi8(v, sh), wt);
                __m256i q;
                tu = _mm256_srli_epi16(_mm256_add_epi16(tu, _mm256_bsrli_epi128(_mm256_slli_epi16(tu, 1), 8)), 8);
                tv = _mm256_srli_epi16(_mm256_add_epi16(tv, _mm256_bsrli_epi128(_mm256_slli_epi16(tv, 1), 8)), 8);
                q = _mm256_permute4x64_epi64(_mm256_unpacklo_epi64(tu, tv), 0xD8);   /* y0 .. y15 as 16-bit */
                _mm_storeu_si128((__m128i *)(void *)(o + x),
                                 _mm_packus_epi16(_mm256_castsi256_si128(q), _mm256_extracti128_si256(q, 1)));
            }
        } else if (channels == 4) {
            const __m256i lo8 = _mm256_set1_epi32(0x00FF00FF), wrb = _mm256_set1_epi32(29 << 16 | 77);
            const __m256i wg = _mm256_set1_epi32(150);
            for (; x + 16 <= w; x += 16) {
                const unsigned char *p = r + 4 * (size_t)x;
                const __m256i v0 = _mm256_loadu_si256((const __m256i *)(const void *)p);
                const __m256i v1 = _mm256_loadu_si256((const __m256i *)(const void *)(p + 32));
                __m256i g0 = _mm256_add_epi32(_mm256_madd_epi16(_mm256_and_si256(v0, lo8), wrb),
                                              _mm256_madd_epi16(_mm256_and_si256(_mm256_srli_epi16(v0, 8), lo8), wg));
                __m256i g1 = _mm256_add_epi32(_mm256_madd_epi16(_mm256_and_si256(v1, lo8), wrb),
                                              _mm256_madd_epi16(_mm256_and_si256(_mm256_srli_epi16(v1, 8), lo8), wg));
                __m256i s = _mm256_permute4x64_epi64(_mm256_packus_epi32(_mm256_srli_epi32(g0, 8),
                                                                         _mm256_srli_epi32(g1, 8)), 0xD8);
                _mm_storeu_si128((__m128i *)(void *)(o + x),
                                 _mm_packus_epi16(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1)));
            }
        }
#endif
        for (r += (size_t)x * channels; x < w; x++, r += channels)
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

/* The horizontal pass's polyphase blocks (resize mode 4): output block b is outputs 4b .. 4b + 3, each with up to 4
 * taps inside the 16 source bytes from start[b]: a byte shuffle puts the 16 tap bytes in output order, then 16-bit
 * products summed in pairs (pmaddwd) give the same exact int32 sums as the table loop (a padded tap has weight 0). */
#define HB_TAPS 4u
static size_t hb_bytes(unsigned dw)
{
    const size_t nb = (size_t)dw / 4 + 1;
    return ALIGN_UP(nb * (4 + 16 + 32) + 64, 64);
}

/* fill the blocks of an axis's tables (kx <= 4 consecutive taps); returns how many leading blocks the SIMD path may
 * take (their 16 source bytes inside the row), 0 if none */
static unsigned hb_build(unsigned w, unsigned dw, unsigned kx, const int *ixx, const int *wx, void *buf)
{
    const unsigned nb = dw / 4;
    int32_t *start = (int32_t *)buf;
    uint8_t *shuf = (uint8_t *)(start + nb + 1);
    int16_t *wt = (int16_t *)(void *)(shuf + 16 * (size_t)(nb + 1));
    unsigned b, o, k, ok_blocks = 0;
    if (kx > HB_TAPS || ((uintptr_t)wt & 1u)) return 0;
    for (b = 0; b < nb; b++) {
        const int s0 = ixx[(size_t)4 * b * kx];
        int fits = s0 >= 0 && (unsigned)s0 + 16 <= w;
        start[b] = s0;
        for (o = 0; o < 4; o++)
            for (k = 0; k < HB_TAPS; k++) {
                const size_t x = 4 * (size_t)b + o;
                int idx = s0, wv = 0;
                if (k < kx) {
                    idx = ixx[x * kx + k];
                    wv = wx[x * kx + k];
                }
                if (!wv) idx = s0;
                if (idx < s0 || idx > s0 + 15 || wv < 0 || wv > Q14) fits = 0, idx = s0, wv = 0;
                shuf[16 * b + 4 * o + k] = (uint8_t)(idx - s0);
                wt[16 * b + 4 * o + k] = (int16_t)wv;
            }
        if (!fits) break;   /* a block past the row's end or with a tap outside its window: the rest scalar */
        ok_blocks = b + 1;
    }
    return ok_blocks;
}

size_t vocr_tsd_resize_scratch(unsigned w, unsigned h, unsigned dw, unsigned dh)
{
    size_t kx = axis_taps(w, dw), ky = axis_taps(h, dh);
    /* the axis tables, the row ring (or the whole-frame intermediate), a uint16 row of w (coarse_rows) and the
     * polyphase blocks (hb_build) */
    return sizeof(int) * (2 * kx * dw + 2 * ky * dh + (size_t)h * dw) + ALIGN_UP(2 * (size_t)w + 64, 64) + 64 +
           hb_bytes(dw);
}

/* one source row through the horizontal taps: t[x] = sum_k r[ix[x][k]] wx[x][k] (exact int32: at most 255 x 16384).
 * mode (resize_hmode): 0 the table (scalar), 1 the identity, 2 the 2:1 pairs, 3 consecutive taps (kx <= 8). An
 * output's taps are consecutive source pixels from its first (axis_weights: area runs of j, bilinear x0, x0 + 1; a
 * padded or clamped tap repeats an index with weight 0), which resize_hmode checks, so the SIMD paths read
 * r[s0 + k]: the same exact sums. */
static void resize_row_h(const unsigned char *r, unsigned w, unsigned dw, unsigned kx, const int *ixx, const int *wx,
                         int *t, int mode, const void *hb, unsigned nbs)
{
    unsigned x = 0, k;
#if RS_AVX2
    if (mode == 4 && hb && nbs) {                   /* the polyphase blocks (hb_build), two a step */
        const unsigned nb = dw / 4;
        const int32_t *start = (const int32_t *)hb;
        const uint8_t *shuf = (const uint8_t *)(start + nb + 1);
        const int16_t *wt = (const int16_t *)(const void *)(shuf + 16 * (size_t)(nb + 1));
        unsigned b = 0;
        for (; b + 2 <= nbs; b += 2) {
            const __m128i s0 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(const void *)(r + start[b])),
                                                _mm_loadu_si128((const __m128i *)(const void *)(shuf + 16 * b)));
            const __m128i s1 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i *)(const void *)(r + start[b + 1])),
                                                _mm_loadu_si128((const __m128i *)(const void *)(shuf + 16 * b + 16)));
            const __m256i p0 = _mm256_madd_epi16(_mm256_cvtepu8_epi16(s0),
                                                 _mm256_loadu_si256((const __m256i *)(const void *)(wt + 16 * b)));
            const __m256i p1 = _mm256_madd_epi16(_mm256_cvtepu8_epi16(s1),
                                                 _mm256_loadu_si256((const __m256i *)(const void *)(wt + 16 * b + 16)));
            /* p0 = [o0a o0b o1a o1b | o2a o2b o3a o3b]: the pair sums of block b, p1 of block b + 1 */
            const __m256i q = _mm256_permute4x64_epi64(_mm256_hadd_epi32(p0, p1), 0xD8);
            _mm256_storeu_si256((__m256i *)(void *)(t + 4 * b), q);
        }
        x = 4 * b;
    } else if (mode == 1) {                         /* identity: r[x] x 16384 */
        for (; x + 8 <= dw; x += 8) {
            const __m256i v = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(const void *)(r + x)));
            _mm256_storeu_si256((__m256i *)(void *)(t + x), _mm256_slli_epi32(v, 14));
        }
    } else if (mode == 2) {                         /* 2:1 area: taps 2x, 2x + 1 at 8192 each */
        const __m256i one = _mm256_set1_epi8(1);
        for (; x + 16 <= dw && 2 * x + 32 <= w; x += 16) {
            const __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)(r + 2 * x));
            const __m256i s = _mm256_maddubs_epi16(v, one);           /* r[2x] + r[2x + 1], 16-bit */
            const __m256i lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(s));
            const __m256i hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(s, 1));
            _mm256_storeu_si256((__m256i *)(void *)(t + x), _mm256_slli_epi32(lo, 13));
            _mm256_storeu_si256((__m256i *)(void *)(t + x + 8), _mm256_slli_epi32(hi, 13));
        }
    } else if (mode == 3) {                         /* 8 source bytes x 8 int16 weights per output */
        static const int lanes[16] = {-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0};
        const __m256i mask = _mm256_loadu_si256((const __m256i *)(const void *)(lanes + 8 - kx));
        for (; x < dw; x++) {
            const int *ww = wx + (size_t)x * kx;
            const unsigned s0 = (unsigned)ixx[(size_t)x * kx];
            __m256i w32;
            __m128i w16, p16, m;
            if (s0 + 8 > w) break;                  /* the right edge: the scalar loop below */
            w32 = _mm256_maskload_epi32(ww, mask);
            w16 = _mm_packs_epi32(_mm256_castsi256_si128(w32), _mm256_extracti128_si256(w32, 1));
            p16 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)(const void *)(r + s0)));
            m = _mm_madd_epi16(p16, w16);
            m = _mm_add_epi32(m, _mm_shuffle_epi32(m, 0x4E));
            m = _mm_add_epi32(m, _mm_shuffle_epi32(m, 0xB1));
            t[x] = _mm_cvtsi128_si32(m);
        }
    }
#else
    (void)w;
    (void)mode;
    (void)hb;
    (void)nbs;
#endif
    for (; x < dw; x++) {
        const int *ii = ixx + (size_t)x * kx, *ww = wx + (size_t)x * kx;
        int acc = 0;
        for (k = 0; k < kx; k++) acc += r[ii[k]] * ww[k];
        t[x] = acc;
    }
}

/* the horizontal path for an axis's taps (resize_row_h), each SIMD path's assumption checked on the tables */
static int resize_hmode(unsigned w, unsigned dw, unsigned kx, const int *idx, const int *wt)
{
    unsigned i, k;
    if (kx == 1 && w == dw) return 1;
    if (w == 2 * dw && kx == 4) {
        for (i = 0; i < dw; i++) {
            const int *ii = idx + (size_t)i * 4, *ww = wt + (size_t)i * 4;
            if (ii[0] != (int)(2 * i) || ii[1] != (int)(2 * i + 1) || ww[0] != Q14 / 2 || ww[1] != Q14 / 2 ||
                ww[2] || ww[3])
                return 0;
        }
        return 2;
    }
    if (kx > 8) return 0;
    for (i = 0; i < dw; i++)
        for (k = 0; k < kx; k++)
            if (wt[(size_t)i * kx + k] < 0 || wt[(size_t)i * kx + k] > Q14 ||
                (k && wt[(size_t)i * kx + k] && idx[(size_t)i * kx + k] != idx[(size_t)i * kx] + (int)k))
                return 0;
    return 3;
}

/* output row y from the ring of horizontally resized rows: (sum_k t_k[x] wy_k + 2^27) >> 28, clamped (int64 exact;
 * the SIMD forms are the same integer: one tap of 16384 gives (t + 2^13) >> 14, two of 8192 (t0 + t1 + 2^14) >> 15,
 * and the general sum is exact in double, every term below 2^36 and the sum below 2^40) */
#if RS_AVX2
/* 1 if n vertical weights are >= 0 and sum to 16384 (axis_weights' do): the int32 split path's bounds */
static int wsum_q14(const int *w, unsigned n)
{
    int sum = 0;
    unsigned k;
    for (k = 0; k < n; k++) {
        if (w[k] < 0 || w[k] > Q14) return 0;
        sum += w[k];
    }
    return sum == Q14;
}
#endif

static void resize_row_v(const int *const *rows, const int *wy, unsigned ky, unsigned dw, unsigned char *o)
{
    const int *rr[64];
    int ww[64];
    unsigned n = 0, k, x = 0;
    for (k = 0; k < ky && n < 64; k++)
        if (wy[k]) {
            rr[n] = rows[k];
            ww[n++] = wy[k];
        }
#if RS_AVX2
    if (n == 1 && ww[0] == Q14) {
        const __m256i h = _mm256_set1_epi32(1 << 13);
        for (; x + 16 <= dw; x += 16) {
            __m256i a = _mm256_srai_epi32(_mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(const void *)(rr[0] + x)), h), 14);
            __m256i b = _mm256_srai_epi32(_mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(const void *)(rr[0] + x + 8)), h), 14);
            __m256i s = _mm256_permute4x64_epi64(_mm256_packs_epi32(a, b), 0xD8);
            _mm_storeu_si128((__m128i *)(void *)(o + x), _mm_packus_epi16(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1)));
        }
    } else if (n == 2 && ww[0] == Q14 / 2 && ww[1] == Q14 / 2) {
        const __m256i h = _mm256_set1_epi32(1 << 14);
        for (; x + 16 <= dw; x += 16) {
            __m256i a = _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(const void *)(rr[0] + x)),
                                         _mm256_loadu_si256((const __m256i *)(const void *)(rr[1] + x)));
            __m256i b = _mm256_add_epi32(_mm256_loadu_si256((const __m256i *)(const void *)(rr[0] + x + 8)),
                                         _mm256_loadu_si256((const __m256i *)(const void *)(rr[1] + x + 8)));
            __m256i s;
            a = _mm256_srai_epi32(_mm256_add_epi32(a, h), 15);
            b = _mm256_srai_epi32(_mm256_add_epi32(b, h), 15);
            s = _mm256_permute4x64_epi64(_mm256_packs_epi32(a, b), 0xD8);
            _mm_storeu_si128((__m128i *)(void *)(o + x), _mm_packus_epi16(_mm256_castsi256_si128(s), _mm256_extracti128_si256(s, 1)));
        }
    } else if (n >= 1 && n <= 16 && wsum_q14(ww, n)) {
        /* exact in int32: t = 2^11 t_hi + t_lo (0 <= t < 2^22), A = sum t_hi w, B = sum t_lo w (each < 2^25 with the
         * weights summing to 16384), and (2^11 A + B + 2^27) >> 28 = (A + ((B + 2^27) >> 11)) >> 17 (the low 11 bits
         * of B + 2^27 cannot carry into bit 28) */
        const __m256i m11 = _mm256_set1_epi32(2047), h27 = _mm256_set1_epi32(1 << 27);
        __m256i wv[16];
        for (k = 0; k < n; k++) wv[k] = _mm256_set1_epi32(ww[k]);
        for (; x + 8 <= dw; x += 8) {
            __m256i A = _mm256_setzero_si256(), Bs = _mm256_setzero_si256(), q;
            __m128i q16;
            for (k = 0; k < n; k++) {
                const __m256i t = _mm256_loadu_si256((const __m256i *)(const void *)(rr[k] + x));
                A = _mm256_add_epi32(A, _mm256_mullo_epi32(_mm256_srli_epi32(t, 11), wv[k]));
                Bs = _mm256_add_epi32(Bs, _mm256_mullo_epi32(_mm256_and_si256(t, m11), wv[k]));
            }
            q = _mm256_srli_epi32(_mm256_add_epi32(A, _mm256_srli_epi32(_mm256_add_epi32(Bs, h27), 11)), 17);
            q = _mm256_permute4x64_epi64(_mm256_packus_epi32(q, _mm256_setzero_si256()), 0x08);
            q16 = _mm256_castsi256_si128(q);
            _mm_storel_epi64((__m128i *)(void *)(o + x), _mm_packus_epi16(q16, q16));
        }
    }
    if (n >= 1 && n <= 16) {
        const __m256d half = _mm256_set1_pd(134217728.0), scale = _mm256_set1_pd(1.0 / 268435456.0);
        const __m256d hi = _mm256_set1_pd(255.0), zero = _mm256_setzero_pd();
        for (; x + 4 <= dw; x += 4) {
            __m256d acc = half;
            __m128i q;
            for (k = 0; k < n; k++)
                acc = _mm256_fmadd_pd(_mm256_cvtepi32_pd(_mm_loadu_si128((const __m128i *)(const void *)(rr[k] + x))),
                                      _mm256_set1_pd((double)ww[k]), acc);
            acc = _mm256_min_pd(_mm256_max_pd(_mm256_floor_pd(_mm256_mul_pd(acc, scale)), zero), hi);
            q = _mm256_cvttpd_epi32(acc);
            q = _mm_packus_epi16(_mm_packs_epi32(q, q), q);
            o[x] = (unsigned char)_mm_extract_epi8(q, 0);
            o[x + 1] = (unsigned char)_mm_extract_epi8(q, 1);
            o[x + 2] = (unsigned char)_mm_extract_epi8(q, 2);
            o[x + 3] = (unsigned char)_mm_extract_epi8(q, 3);
        }
    }
#endif
    for (; x < dw; x++) {
        int64_t acc = 0;
        for (k = 0; k < n; k++) acc += (int64_t)rr[k][x] * ww[k];
        acc = (acc + ((int64_t)1 << 27)) >> 28;
        o[x] = (unsigned char)(acc > 255 ? 255 : acc < 0 ? 0 : acc);
    }
}

/* source row j of the resize: the grey row itself, or (channels 3 / 4) its integer luma (vocr_tsd_grey) in rowbuf */
static const unsigned char *src_row(const unsigned char *src, unsigned w, unsigned stride, unsigned channels, unsigned j,
                                    unsigned char *rowbuf)
{
    if (channels == 1) return src + (size_t)j * stride;
    vocr_tsd_grey(src + (size_t)j * stride, w, 1, stride, channels, rowbuf);
    return rowbuf;
}

/* output rows [y0, y1) through a ring of R horizontally resized source rows: output row y reads the source rows
 * iyy[y][0 ..] (consecutive, at most ky), so each source row of the band is resized once and the ring holds every row
 * an output needs. Each output row is the same integer sums whatever the band (the ring is only a cache). */
static void resize_rows(const unsigned char *src, unsigned w, unsigned stride, unsigned channels, unsigned char *dst,
                        unsigned dw, unsigned kx, unsigned ky, unsigned R, int mode, const int *ixx, const int *wx,
                        const int *iyy, const int *wy, unsigned y0, unsigned y1, int *ring, unsigned char *rowbuf,
                        const void *hb, unsigned nbs)
{
    int slot_row[64];
    const int *rows[64];
    unsigned y, k;
    for (k = 0; k < R; k++) slot_row[k] = -1;
    for (y = y0; y < y1; y++) {
        const int *ii = iyy + (size_t)y * ky;
        for (k = 0; k < ky; k++) {
            const int j = ii[k];
            const unsigned s = (unsigned)j % R;
            if (slot_row[s] != j) {
                resize_row_h(src_row(src, w, stride, channels, (unsigned)j, rowbuf), w, dw, kx, ixx, wx,
                             ring + (size_t)s * dw, mode, hb, nbs);
                slot_row[s] = j;
            }
            rows[k] = ring + (size_t)s * dw;
        }
        resize_row_v(rows, wy + (size_t)y * ky, ky, dw, dst + (size_t)y * dw);
    }
}

/* 1 if the ring applies: ky <= 64 and each output's source rows span fewer than R = min(ky, h) rows */
static int ring_fits(const int *iyy, unsigned ky, unsigned h, unsigned dh)
{
    const unsigned R = ky < h ? ky : h;
    unsigned y, k;
    if (ky > 64) return 0;
    for (y = 0; y < dh; y++) {
        int lo = iyy[(size_t)y * ky], hi = lo;
        for (k = 1; k < ky; k++) {
            const int j = iyy[(size_t)y * ky + k];
            lo = j < lo ? j : lo;
            hi = j > hi ? j : hi;
        }
        if ((unsigned)(hi - lo) >= R) return 0;
    }
    return 1;
}

/* The exact 2x enlargement (dw = 2 w, dh = 2 h). Bilinear on pixel centres makes every weight a multiple of 4096:
 * output 0 is 16384 r[0], output 2j (j >= 1) 4096 r[j - 1] + 12288 r[j], output 2j + 1 (j <= w - 2) 12288 r[j] +
 * 4096 r[j + 1], the last 16384 r[w - 1] (axis_weights). So a horizontal sum is t = 4096 t' with t' = r[j - 1] + 3 r[j]
 * (and the like) <= 1020, and a vertical sum is 2^24 S with S = the same 1 / 3 / 4 combination of t' rows <= 4080:
 * (2^24 S + 2^27) >> 28 = (S + 8) >> 4, the rule's bits computed in 16-bit lanes. */
static void up2_row_h(const unsigned char *r, unsigned w, uint16_t *t)
{
    unsigned j = 1;
    t[0] = (uint16_t)(4u * r[0]);
    t[2 * w - 1] = (uint16_t)(4u * r[w - 1]);
#if RS_AVX2
    for (; j + 17 <= w; j += 16) {   /* outputs 2j .. 2j + 31 from r[j - 1 .. j + 16] */
        const __m256i c = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(const void *)(r + j)));
        const __m256i l = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(const void *)(r + j - 1)));
        const __m256i n = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(const void *)(r + j + 1)));
        const __m256i c3 = _mm256_add_epi16(_mm256_add_epi16(c, c), c);
        const __m256i ev = _mm256_add_epi16(l, c3), od = _mm256_add_epi16(c3, n);
        const __m256i lo = _mm256_unpacklo_epi16(ev, od), hi = _mm256_unpackhi_epi16(ev, od);
        _mm256_storeu_si256((__m256i *)(void *)(t + 2 * j), _mm256_permute2x128_si256(lo, hi, 0x20));
        _mm256_storeu_si256((__m256i *)(void *)(t + 2 * j + 16), _mm256_permute2x128_si256(lo, hi, 0x31));
    }
#endif
    for (; j < w; j++) t[2 * j] = (uint16_t)(r[j - 1] + 3u * r[j]);
    for (j = 0; j + 1 < w; j++) t[2 * j + 1] = (uint16_t)(3u * r[j] + r[j + 1]);
}

/* output rows [y0, y1) of the 2x enlargement through a ring of three t' rows (ring: 3 x 2w uint16) */
static void up2_rows(const unsigned char *src, unsigned w, unsigned h, unsigned stride, unsigned channels,
                     unsigned char *dst, unsigned y0, unsigned y1, uint16_t *ring, unsigned char *rowbuf)
{
    const unsigned dw = 2 * w;
    int slot_row[3] = {-1, -1, -1};
    unsigned y;
    for (y = y0; y < y1; y++) {
        unsigned ja, jb, wa, wb, k, x = 0;
        const uint16_t *ta, *tb;
        unsigned char *o = dst + (size_t)y * dw;
        if (y == 0) {
            ja = jb = 0, wa = 4, wb = 0;
        } else if (y == 2 * h - 1) {
            ja = jb = h - 1, wa = 4, wb = 0;
        } else if (y % 2 == 0) {
            ja = y / 2 - 1, jb = y / 2, wa = 1, wb = 3;
        } else {
            ja = y / 2, jb = y / 2 + 1, wa = 3, wb = 1;
        }
        for (k = 0; k < 2; k++) {
            const unsigned j = k ? jb : ja;
            if (slot_row[j % 3] != (int)j) {
                up2_row_h(src_row(src, w, stride, channels, j, rowbuf), w, ring + (size_t)(j % 3) * dw);
                slot_row[j % 3] = (int)j;
            }
        }
        ta = ring + (size_t)(ja % 3) * dw;
        tb = ring + (size_t)(jb % 3) * dw;
#if RS_AVX2
        {
            const __m256i k8 = _mm256_set1_epi16(8), va = _mm256_set1_epi16((short)wa), vb = _mm256_set1_epi16((short)wb);
            for (; x + 16 <= dw; x += 16) {
                const __m256i a = _mm256_loadu_si256((const __m256i *)(const void *)(ta + x));
                const __m256i b = _mm256_loadu_si256((const __m256i *)(const void *)(tb + x));
                __m256i v = _mm256_add_epi16(_mm256_add_epi16(_mm256_mullo_epi16(a, va), _mm256_mullo_epi16(b, vb)), k8);
                v = _mm256_srli_epi16(v, 4);
                v = _mm256_permute4x64_epi64(_mm256_packus_epi16(v, v), 0x08);
                _mm_storeu_si128((__m128i *)(void *)(o + x), _mm256_castsi256_si128(v));
            }
        }
#endif
        for (; x < dw; x++) o[x] = (unsigned char)((wa * ta[x] + wb * tb[x] + 8u) >> 4);
    }
}

/* 1 if every weight of both axes is a multiple of 4096 (quarters: 2:1, 4:3 and 4:1 shrinks, the identity): then the
 * rule's sum is 2^24 S with S = sum r wx' wy' (w' = w / 4096, the w' of an output summing to 4) <= 4080, exact in 16-bit
 * lanes, and (2^24 S + 2^27) >> 28 = (S + 8) >> 4. The sum is exact in any order, so the vertical taps can come
 * first: V'[x] = sum r[iy][x] wy' (<= 1020), then S = sum V'[ix] wx'. */
static int coarse_axis(const int *wt, unsigned m, unsigned K)
{
    size_t i;
    for (i = 0; i < (size_t)m * K; i++)
        if (wt[i] < 0 || wt[i] % 4096) return 0;
    return 1;
}

/* output rows [y0, y1) of a coarse resize (coarse_axis on both): V (w uint16) the vertical sums of one row */
static void coarse_rows(const unsigned char *src, unsigned w, unsigned stride, unsigned channels, unsigned char *dst,
                        unsigned dw, unsigned kx, unsigned ky, const int *ixx, const int *wx, const int *iyy,
                        const int *wy, unsigned y0, unsigned y1, uint16_t *V, unsigned char *rowbuf)
{
    unsigned y, k, x;
    int pat43 = 0;   /* 4:3: outputs 3q, 3q + 1, 3q + 2 from V[4q .. 4q + 3] as 3a + b, 2b + 2c, c + 3d (the tables say so) */
    if (kx >= 2 && 4 * dw == 3 * w) {
        pat43 = 1;
        for (k = 0; k < dw && pat43; k++) {
            const int *ik = ixx + (size_t)k * kx, *wk = wx + (size_t)k * kx;
            const unsigned q = k / 3, rr = k % 3, j0 = 4 * q + rr, w0 = rr == 0 ? 3 : rr == 1 ? 2 : 1;
            unsigned j;
            pat43 = ik[0] == (int)j0 && wk[0] == (int)(4096 * w0) && ik[1] == (int)j0 + 1 &&
                    wk[1] == (int)(4096 * (4 - w0));
            for (j = 2; j < kx; j++) pat43 = pat43 && !wk[j];
        }
    }
    int pat41 = 0, pat11 = 0;   /* 4:1: output q from V[4q .. 4q + 3] at 1 each; 1:1: output x = V[x] at 4 */
    if (kx >= 4 && dw * 4 == w) {
        pat41 = 1;
        for (k = 0; k < dw && pat41; k++) {
            const int *ik = ixx + (size_t)k * kx, *wk = wx + (size_t)k * kx;
            unsigned j;
            for (j = 0; j < 4; j++) pat41 = pat41 && ik[j] == (int)(4 * k + j) && wk[j] == 4096;
            for (j = 4; j < kx; j++) pat41 = pat41 && !wk[j];
        }
    }
    if (kx == 1 && dw == w) pat11 = 1;   /* axis_weights: index x, weight 16384 */
#if !RS_AVX2
    (void)pat43;
    (void)pat41;
    (void)pat11;
#endif
    for (y = y0; y < y1; y++) {
        const int *ii = iyy + (size_t)y * ky, *ww = wy + (size_t)y * ky;
        unsigned char *o = dst + (size_t)y * dw;
        int first = 1;
        for (k = 0; k < ky; k++) {
            const unsigned wk = (unsigned)ww[k] >> 12;
            const unsigned char *r;
            if (!wk) continue;
            r = src_row(src, w, stride, channels, (unsigned)ii[k], rowbuf);
            x = 0;
#if RS_AVX2
            {
                const __m256i vw = _mm256_set1_epi16((short)wk);
                for (; x + 16 <= w; x += 16) {
                    __m256i v = _mm256_mullo_epi16(_mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(const void *)(r + x))), vw);
                    if (!first) v = _mm256_add_epi16(v, _mm256_loadu_si256((const __m256i *)(const void *)(V + x)));
                    _mm256_storeu_si256((__m256i *)(void *)(V + x), v);
                }
            }
#endif
            for (; x < w; x++) V[x] = (uint16_t)((first ? 0u : V[x]) + wk * r[x]);
            first = 0;
        }
        x = 0;
#if RS_AVX2
        if (pat43) {
            static const signed char sx[16] = {0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, -1, -1, -1, -1};
            static const signed char sy[16] = {2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, -1, -1, -1, -1};
            {
                const __m128i mx = _mm_loadu_si128((const __m128i *)(const void *)sx);
                const __m128i my = _mm_loadu_si128((const __m128i *)(const void *)sy);
                const __m128i wa = _mm_setr_epi16(3, 2, 1, 3, 2, 1, 0, 0), wb = _mm_setr_epi16(1, 2, 3, 1, 2, 3, 0, 0);
                const __m128i k8 = _mm_set1_epi16(8);
                /* 8 inputs (2 periods) -> 6 outputs; an 8-byte store, the next group's overwriting its last 2 */
                for (; x + 8 <= dw && (x / 3) * 4 + 8 <= w; x += 6) {
                    const __m128i v = _mm_loadu_si128((const __m128i *)(const void *)(V + (x / 3) * 4));
                    __m128i sum = _mm_add_epi16(_mm_mullo_epi16(_mm_shuffle_epi8(v, mx), wa),
                                                _mm_mullo_epi16(_mm_shuffle_epi8(v, my), wb));
                    sum = _mm_srli_epi16(_mm_add_epi16(sum, k8), 4);
                    _mm_storel_epi64((__m128i *)(void *)(o + x), _mm_packus_epi16(sum, sum));
                }
            }
        } else if (pat41) {
            const __m256i one = _mm256_set1_epi16(1), k8 = _mm256_set1_epi32(8);
            for (; x + 8 <= dw; x += 8) {   /* 32 inputs -> 8 outputs */
                const __m256i a = _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *)(const void *)(V + 4 * x)), one);
                const __m256i b = _mm256_madd_epi16(_mm256_loadu_si256((const __m256i *)(const void *)(V + 4 * x + 16)), one);
                __m256i q = _mm256_permute4x64_epi64(_mm256_hadd_epi32(a, b), 0xD8);   /* outputs x .. x + 7 */
                __m128i q16;
                q = _mm256_srli_epi32(_mm256_add_epi32(q, k8), 4);
                q = _mm256_permute4x64_epi64(_mm256_packus_epi32(q, _mm256_setzero_si256()), 0x08);
                q16 = _mm256_castsi256_si128(q);                                        /* 8 uint16, in order */
                _mm_storel_epi64((__m128i *)(void *)(o + x), _mm_packus_epi16(q16, q16));
            }
        } else if (pat11) {
            const __m256i k8 = _mm256_set1_epi16(8);
            for (; x + 16 <= dw; x += 16) {
                __m256i v = _mm256_slli_epi16(_mm256_loadu_si256((const __m256i *)(const void *)(V + x)), 2);
                v = _mm256_srli_epi16(_mm256_add_epi16(v, k8), 4);
                v = _mm256_permute4x64_epi64(_mm256_packus_epi16(v, v), 0x08);
                _mm_storeu_si128((__m128i *)(void *)(o + x), _mm256_castsi256_si128(v));
            }
        }
#endif
        for (; x < dw; x++) {
            const int *ik = ixx + (size_t)x * kx, *wk = wx + (size_t)x * kx;
            unsigned acc = 8;
            for (k = 0; k < kx; k++) acc += ((unsigned)wk[k] >> 12) * V[ik[k]];
            o[x] = (unsigned char)(acc >> 4);
        }
    }
}

static void resize_core(const unsigned char *src, unsigned w, unsigned h, unsigned stride, unsigned channels,
                        unsigned char *dst, unsigned dw, unsigned dh, int *scratch, unsigned char *rowbuf)
{
    unsigned kx = axis_taps(w, dw), ky = axis_taps(h, dh), y, k, nbs = 0;
    int *ixx = scratch, *wx = ixx + (size_t)kx * dw, *iyy = wx + (size_t)kx * dw, *wy = iyy + (size_t)ky * dh;
    int *ring = wy + (size_t)ky * dh, mode;
    unsigned char *hb;
    if (dw == 2 * w && dh == 2 * h) {               /* the exact 2x enlargement (up2_rows: the same bits) */
        up2_rows(src, w, h, stride, channels, dst, 0, dh, (uint16_t *)(void *)scratch, rowbuf);   /* 6 dw bytes */
        return;
    }
    axis_weights(w, dw, kx, ixx, wx);
    axis_weights(h, dh, ky, iyy, wy);
    if (!(w == 2 * dw && kx == 4) && coarse_axis(wx, dw, kx) && coarse_axis(wy, dh, ky)) {
        /* quarter weights (but not the 2:1 shrink, which resize_rows' paths do faster): coarse_rows, the same bits */
        coarse_rows(src, w, stride, channels, dst, dw, kx, ky, ixx, wx, iyy, wy, 0, dh,
                    (uint16_t *)(void *)(ring + (size_t)h * dw), rowbuf);   /* the uint16 row after the ring */
        return;
    }
    mode = resize_hmode(w, dw, kx, ixx, wx);
    /* the polyphase blocks after the ring and the uint16 row (vocr_tsd_resize_scratch) */
    hb = (unsigned char *)(void *)(ring + (size_t)h * dw) + ALIGN_UP(2 * (size_t)w + 64, 64);
    hb = (unsigned char *)(((uintptr_t)hb + 63u) & ~(uintptr_t)63u);
    if (mode == 3 && (nbs = hb_build(w, dw, kx, ixx, wx, hb)) > 0) mode = 4;
    if (!ring_fits(iyy, ky, h, dh)) {               /* a shrink by more than 62: the whole-frame intermediate */
        int *tmp = ring;
        for (y = 0; y < h; y++)
            resize_row_h(src_row(src, w, stride, channels, y, rowbuf), w, dw, kx, ixx, wx, tmp + (size_t)y * dw, mode,
                         hb, nbs);
        for (y = 0; y < dh; y++) {
            const int *ii = iyy + (size_t)y * ky, *ww = wy + (size_t)y * ky;
            unsigned char *o = dst + (size_t)y * dw;
            unsigned x;
            for (x = 0; x < dw; x++) {
                int64_t acc = 0;
                for (k = 0; k < ky; k++) acc += (int64_t)tmp[(size_t)ii[k] * dw + x] * ww[k];
                acc = (acc + ((int64_t)1 << 27)) >> 28;
                o[x] = (unsigned char)(acc > 255 ? 255 : acc < 0 ? 0 : acc);
            }
        }
        return;
    }
    resize_rows(src, w, stride, channels, dst, dw, kx, ky, ky < h ? ky : h, mode, ixx, wx, iyy, wy, 0, dh, ring, rowbuf,
                hb, nbs);
}

/* the banded resize's tables: a header, then the four axis tables */
typedef struct {
    unsigned w, h, dw, dh, kx, ky, R, nbs;
    int mode, ring;
} resize_plan_hdr;

#define PLAN_HDR ALIGN_UP(sizeof(resize_plan_hdr), 64)

size_t vocr_tsd_resize_plan_bytes(unsigned w, unsigned h, unsigned dw, unsigned dh)
{
    size_t kx = axis_taps(w, dw), ky = axis_taps(h, dh);
    return PLAN_HDR + ALIGN_UP(sizeof(int) * (2 * kx * dw + 2 * ky * dh), 64) + hb_bytes(dw);
}

/* the plan's polyphase blocks, after its axis tables */
static unsigned char *plan_hb(const void *plan, unsigned kx, unsigned ky, unsigned dw, unsigned dh)
{
    return (unsigned char *)(uintptr_t)plan + PLAN_HDR +
           ALIGN_UP(sizeof(int) * (2 * (size_t)kx * dw + 2 * (size_t)ky * dh), 64);
}

int vocr_tsd_resize_plan(unsigned w, unsigned h, unsigned dw, unsigned dh, void *plan)
{
    resize_plan_hdr *P = (resize_plan_hdr *)plan;
    int *ixx = (int *)(void *)((unsigned char *)plan + PLAN_HDR);
    unsigned kx = axis_taps(w, dw), ky = axis_taps(h, dh);
    int *wx = ixx + (size_t)kx * dw, *iyy = wx + (size_t)kx * dw, *wy = iyy + (size_t)ky * dh;
    axis_weights(w, dw, kx, ixx, wx);
    axis_weights(h, dh, ky, iyy, wy);
    P->w = w;
    P->h = h;
    P->dw = dw;
    P->dh = dh;
    P->kx = kx;
    P->ky = ky;
    P->R = ky < h ? ky : h;
    P->mode = resize_hmode(w, dw, kx, ixx, wx);
    P->nbs = 0;
    if (P->mode == 3 && (P->nbs = hb_build(w, dw, kx, ixx, wx, plan_hb(plan, kx, ky, dw, dh))) > 0) P->mode = 4;
    P->ring = ring_fits(iyy, ky, h, dh);
    if (!(w == 2 * dw && kx == 4) && coarse_axis(wx, dw, kx) && coarse_axis(wy, dh, ky))
        P->mode = -3, P->ring = 1;                                   /* quarter weights (coarse_rows) */
    if (dw == 2 * w && dh == 2 * h) P->mode = -2, P->ring = 1;   /* the exact 2x enlargement (up2_rows) */
    return P->ring ? 0 : -1;
}

static size_t band_ring_bytes(unsigned w, unsigned R, unsigned dw)
{
    size_t ring = sizeof(int) * R * dw;
    if (ring < 6 * (size_t)dw) ring = 6 * (size_t)dw;   /* the 2x enlargement's three uint16 rows */
    if (ring < 2 * (size_t)w) ring = 2 * (size_t)w;     /* coarse_rows' uint16 row */
    return ALIGN_UP(ring, 64);
}

size_t vocr_tsd_resize_band_bytes(unsigned w, unsigned h, unsigned dw, unsigned dh)
{
    size_t ky = axis_taps(h, dh), R = ky < h ? ky : h;
    return band_ring_bytes(w, (unsigned)R, dw) + ALIGN_UP((size_t)w + 64, 64);
}

void vocr_tsd_resize_band(const void *plan, const unsigned char *src, unsigned stride, unsigned channels,
                           unsigned char *dst, unsigned y0, unsigned y1, void *work)
{
    const resize_plan_hdr *P = (const resize_plan_hdr *)plan;
    const int *ixx = (const int *)(const void *)((const unsigned char *)plan + PLAN_HDR);
    const int *wx = ixx + (size_t)P->kx * P->dw, *iyy = wx + (size_t)P->kx * P->dw, *wy = iyy + (size_t)P->ky * P->dh;
    int *ring = (int *)work;
    unsigned char *rowbuf = (unsigned char *)work + band_ring_bytes(P->w, P->R, P->dw);
    if (y1 > P->dh) y1 = P->dh;
    if (y0 >= y1 || !P->ring) return;
    if (P->mode == -2) {   /* the exact 2x enlargement: three uint16 rows of 2w in the ring area */
        up2_rows(src, P->w, P->h, stride, channels, dst, y0, y1, (uint16_t *)(void *)ring, rowbuf);
        return;
    }
    if (P->mode == -3) {   /* quarter weights: a uint16 row of w in the ring area */
        coarse_rows(src, P->w, stride, channels, dst, P->dw, P->kx, P->ky, ixx, wx, iyy, wy, y0, y1,
                    (uint16_t *)(void *)ring, rowbuf);
        return;
    }
    resize_rows(src, P->w, stride, channels, dst, P->dw, P->kx, P->ky, P->R, P->mode, ixx, wx, iyy, wy, y0, y1, ring,
                rowbuf, plan_hb(plan, P->kx, P->ky, P->dw, P->dh), P->nbs);
}

void vocr_tsd_resize(const unsigned char *src, unsigned w, unsigned h, unsigned stride, unsigned char *dst,
                      unsigned dw, unsigned dh, int *scratch)
{
    resize_core(src, w, h, stride, 1, dst, dw, dh, scratch, NULL);
}

size_t vocr_tsd_resize_rgb_scratch(unsigned w, unsigned h, unsigned dw, unsigned dh)
{
    return vocr_tsd_resize_scratch(w, h, dw, dh) + ((size_t)w + 64) / 64 * 64;
}

void vocr_tsd_resize_rgb(const unsigned char *rgb, unsigned w, unsigned h, unsigned stride, unsigned channels,
                          unsigned char *dst, unsigned dw, unsigned dh, int *scratch)
{
    unsigned char *rowbuf = (unsigned char *)(void *)((char *)(void *)scratch + vocr_tsd_resize_scratch(w, h, dw, dh));
    resize_core(rgb, w, h, stride, channels, dst, dw, dh, scratch, rowbuf);
}

