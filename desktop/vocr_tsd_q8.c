/* vocr_tsd_q8.c - textspot-desktop's int8 network (issue #173, round 3; docs/textspot-desktop.md): the network of a
 * VTSD version-2 blob (train/tsdesk_blob.py, written by train/tsdesk_quant.py from a version-1 blob and a calibration).
 *
 * The maps between layers are uint8 with a scale per tensor and zero point 0, the weights int8 per output channel.
 * Every layer sums its products exactly in int32 (acc), then out = q(f32(acc + bq) * m), adding f32(res) * r in a
 * residual block or max(., 0) + f32(up) * r in a lower lateral, each float operation rounded to float32, and
 * q(y) = rint(clip(y, 0, 255)); the head gives float32 maps f32(acc + bq) * m. gc sums exactly, takes its context
 * vector in float64 in a fixed order and adds it in float32. So every build (plain C, AVX2 with exact int16 products,
 * AVX-512 VNNI), every thread count and the numpy twin
 * (train/tsdesk_quant.q8_maps) give the same maps bit for bit, whatever order the kernels sum in.
 *
 * The kernels: the 1x1 layers, the stem (its 4x4 patches as 16 input channels) and the head on VNNI (vpdpbusd: four
 * uint8 x int8 products a lane) or AVX2 (vpmaddwd on int16) or plain C; the depthwise layers on VNNI from expanded
 * rows (each cell's three taps of a row and a zero side by side per channel, so a row of taps is one vpdpbusd per 16
 * channels; the expanded rows of a column strip are kept in a ring per task and shared by the output rows that read
 * them) or plain C. Channel counts are multiples of 8; the SIMD paths take what their vectors cover and plain C the
 * rest (the same integers). C99, no allocation (the rings are in the arena).
 */
#ifdef VOCR_TSD_PROF
#define _POSIX_C_SOURCE 200112L
#endif
#include "vocr_tsd_q8.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#if defined(__AVX2__) && defined(__FMA__) && !defined(VOCR_TSD_SCALAR)
#include <immintrin.h>
#define Q_AVX2 1
#else
#define Q_AVX2 0
#endif
#if Q_AVX2 && defined(__AVX512F__) && defined(__AVX512BW__) && defined(__AVX512VL__) && defined(__AVX512VNNI__) && \
    !defined(VOCR_TSD_NO_AVX512)
#define Q_VNNI 1
#else
#define Q_VNNI 0
#endif

#if defined(__GNUC__) || defined(__clang__)
#define Q_INLINE static inline __attribute__((always_inline))
#else
#define Q_INLINE static inline
#endif

#define ALIGN_UP(x, a) (((x) + (size_t)(a) - 1u) / (size_t)(a) * (size_t)(a))
#ifndef Q_CHUNK
#define Q_CHUNK 48u            /* output columns a chunk (a multiple of 12: every tile's pixel count) */
#endif
#define Q_MAX_STEPS 64u
#define Q_RING_ROWS 5u         /* the AVX2 path's ring of float rows: 2 d + 1 rows for dilation d <= 2 */
/* the VNNI path's vertically expanded row: up to (Q_CHUNK - 1) s + 2 d + 1 input columns (stride <= 2,
 * dilation <= 16) of 4 bytes per channel */
#define Q_RING_BYTES ((2u * Q_CHUNK + 2u * 16u + 1u) * VOCR_TSD_MAX_CHANNELS * 4u)
#define Q_BIAS_MAX (1 << 30)   /* |bq|: acc + bq stays in int32 (|acc| < 2^23: 256 channels of 255 x 128) */
#define PROF_STEP0 8u

int vocr_tsd_q8_simd(void)
{
    return Q_VNNI ? 2 : Q_AVX2;
}

#ifdef VOCR_TSD_PROF
#include <stdio.h>
#define QPROF_START() vocr_tsd_prof_start_()
#define QPROF_MARK(i, name) vocr_tsd_prof_mark_((i), (name))
/* profiling builds (one thread): the depthwise part of each step, the row expansion and the head, in us */
static double qprof_dw[Q_MAX_STEPS], qprof_expand, qprof_head;
static char qprof_names[Q_MAX_STEPS][32];
#define QPROF_T(v) double v = vocr_tsd_prof_now_()
#define QPROF_ADD(acc, t0) ((acc) += vocr_tsd_prof_now_() - (t0))
#else
#define QPROF_START() ((void)0)
#define QPROF_MARK(i, name) ((void)0)
#define QPROF_T(v) ((void)0)
#define QPROF_ADD(acc, t0) ((void)0)
#endif

/* ----------------------------------------------------------------------------- the blob */

static const unsigned char *q_take(const unsigned char *b, size_t bytes, size_t *off, size_t n)
{
    const unsigned char *p;
    if (n > bytes || *off > bytes - n) return NULL;
    p = b + *off;
    *off += n;
    return p;
}

static int finite_nonneg(float v)
{
    return v == v && v >= 0.0f && v <= 3.0e38f;
}

/* a layer's int8 weights (nw bytes), int32 bq[cout], float32 m[cout] and (with_r) float32 r */
static int q_set(vocr_tsd_qlayer *L, const unsigned char *b, size_t bytes, size_t *off, size_t nw, unsigned cin,
                 unsigned cout, unsigned stride, unsigned dil, int with_r, size_t *params)
{
    const unsigned char *w, *bq, *mm, *r = NULL;
    unsigned i;
    if (nw % 4) return VOCR_TSD_E_BLOB;
    w = q_take(b, bytes, off, nw);
    bq = q_take(b, bytes, off, 4u * cout);
    mm = q_take(b, bytes, off, 4u * cout);
    if (with_r) r = q_take(b, bytes, off, 4u);
    if (!w || !bq || !mm || (with_r && !r)) return VOCR_TSD_E_BLOB;
    L->w = (const int8_t *)(const void *)w;
    L->bq = (const int32_t *)(const void *)bq;
    L->m = (const float *)(const void *)mm;
    L->r = 0.0f;
    if (r) memcpy(&L->r, r, sizeof L->r);
    L->cin = cin;
    L->cout = cout;
    L->stride = stride;
    L->dil = dil;
    for (i = 0; i < cout; i++)
        if (L->bq[i] > Q_BIAS_MAX || L->bq[i] < -Q_BIAS_MAX || !finite_nonneg(L->m[i])) return VOCR_TSD_E_BLOB;
    if (!finite_nonneg(L->r)) return VOCR_TSD_E_BLOB;
    *params += nw + cout;
    return VOCR_TSD_OK;
}

int vocr_tsdq_load(vocr_tsd_model *m, const unsigned char *b, size_t bytes)
{
    size_t off = VOCR_TSD_HEADER;
    unsigned s, i;
    if (q_set(&m->qstem, b, bytes, &off, 16u * m->c[0], 16, m->c[0], 4, 1, 1, &m->params)) return VOCR_TSD_E_BLOB;
    for (s = 0; s < 4; s++) {
        const unsigned C = m->c[s];
        if (s > 0) {
            const unsigned Ci = m->c[s - 1];
            if (q_set(&m->qdw[s][0], b, bytes, &off, 9u * Ci, Ci, Ci, 2, 1, 0, &m->params) ||
                q_set(&m->qpw[s][0], b, bytes, &off, (size_t)Ci * C, Ci, C, 1, 1, 1, &m->params))
                return VOCR_TSD_E_BLOB;
        }
        for (i = 0; i < m->n[s]; i++)
            if (q_set(&m->qdw[s][1 + i], b, bytes, &off, 9u * C, C, C, 1, m->dil[s][i], 0, &m->params) ||
                q_set(&m->qpw[s][1 + i], b, bytes, &off, (size_t)C * C, C, C, 1, 1, 1, &m->params))
                return VOCR_TSD_E_BLOB;
    }
    if (m->gc) {
        const unsigned C = m->c[3];
        const unsigned char *w = q_take(b, bytes, &off, 4u * C * C), *bb = q_take(b, bytes, &off, 4u * C);
        const unsigned char *sc = q_take(b, bytes, &off, 8u);
        if (!w || !bb || !sc) return VOCR_TSD_E_BLOB;
        m->gcl.w = (const float *)(const void *)w;
        m->gcl.b = (const float *)(const void *)bb;
        m->gcl.cin = m->gcl.cout = C;
        m->gcl.stride = m->gcl.dil = 1;
        memcpy(&m->gc_sx, sc, 4);
        memcpy(&m->gc_so, sc + 4, 4);
        if (!finite_nonneg(m->gc_sx) || !(m->gc_so > 0.0f) || !finite_nonneg(m->gc_so)) return VOCR_TSD_E_BLOB;
        for (i = 0; i < C * C + C; i++) {
            const float v = i < C * C ? m->gcl.w[i] : m->gcl.b[i - C * C];
            if (v != v || v > 3.0e38f || v < -3.0e38f) return VOCR_TSD_E_BLOB;
        }
        m->params += (size_t)C * C + C;
    }
    for (s = m->hs == 4 ? 0u : 1u; s < 4; s++)
        if (q_set(&m->qlat[s], b, bytes, &off, (size_t)m->c[s] * m->F, m->c[s], m->F, 1, 1, 1, &m->params))
            return VOCR_TSD_E_BLOB;
    for (i = 0; i < m->smooth; i++)
        if (q_set(&m->qsdw[i], b, bytes, &off, 9u * m->F, m->F, m->F, 1, 1, 0, &m->params) ||
            q_set(&m->qspw[i], b, bytes, &off, (size_t)m->F * m->F, m->F, m->F, 1, 1, 1, &m->params))
            return VOCR_TSD_E_BLOB;
    if (q_set(&m->qhead, b, bytes, &off, (size_t)m->F * m->outp, m->F, m->outp, 1, 1, 1, &m->params))
        return VOCR_TSD_E_BLOB;
    if (off + 4 != bytes) return VOCR_TSD_E_BLOB;
    return VOCR_TSD_OK;
}

/* ----------------------------------------------------------------------------- requantization */

enum { Q_RELU = 0, Q_RES = 1, Q_UP = 2, Q_FLOAT = 3 };

typedef struct {
    int kind;
    const uint8_t *res;   /* Q_RES: the block's input at the chunk's cells, [P][Cout] */
    const uint8_t *up;    /* Q_UP: the coarser level's row, [.][Cout] */
    unsigned x0;          /* Q_UP: the chunk's first column */
    float *f;             /* Q_FLOAT: the maps of the chunk's cells, [P][fstride] */
    unsigned fstride;     /* Q_FLOAT: maps a cell (the first fstride outputs are stored) */
} q_epi;

static inline uint8_t q_round(float y)
{
    y = y > 0.0f ? y : 0.0f;
    y = y < 255.0f ? y : 255.0f;
    return (uint8_t)lrintf(y);
}

/* output co of cell p from its sum (the module comment's rule) */
static inline void q_out1(int32_t acc, const vocr_tsd_qlayer *L, const q_epi *e, unsigned p, unsigned co, uint8_t *out)
{
    const unsigned C = L->cout;
    float y = (float)(acc + L->bq[co]) * L->m[co];
    if (e->kind == Q_FLOAT) {
        if (co < e->fstride) e->f[(size_t)p * e->fstride + co] = y;
        return;
    }
    if (e->kind == Q_RES) {
        const float t = (float)e->res[(size_t)p * C + co] * L->r;
        y = y + t;
    } else if (e->kind == Q_UP) {
        const float t = (float)e->up[(size_t)((e->x0 + p) >> 1) * C + co] * L->r;
        y = (y > 0.0f ? y : 0.0f) + t;
    }
    out[(size_t)p * C + co] = q_round(y);
}

/* ----------------------------------------------------------------------------- 1x1 layers (and the stem, the head) */

/* out[p] = epi(in[p] W) for p < P, channels co in [c0, c1) (in: [P][Cin] uint8, W: [Cin / 4][Cout][4] int8) */
static void q_pw_c(const uint8_t *in, unsigned P, const vocr_tsd_qlayer *L, const q_epi *e, unsigned c0, unsigned c1,
                   uint8_t *out)
{
    const unsigned Cin = L->cin, Cout = L->cout;
    unsigned p, co, j;
    for (p = 0; p < P; p++) {
        const uint8_t *x = in + (size_t)p * Cin;
        for (co = c0; co < c1; co++) {
            int32_t acc = 0;
            for (j = 0; j < Cin / 4; j++) {
                const int8_t *w = L->w + ((size_t)j * Cout + co) * 4;
                acc += (int32_t)x[4 * j] * w[0] + (int32_t)x[4 * j + 1] * w[1] + (int32_t)x[4 * j + 2] * w[2] +
                       (int32_t)x[4 * j + 3] * w[3];
            }
            q_out1(acc, L, e, p, co, out);
        }
    }
}

#if Q_VNNI
static inline int32_t rd_i32(const uint8_t *p)
{
    int32_t v;
    memcpy(&v, p, 4);
    return v;
}
#endif

#if Q_AVX2
/* 8 outputs from their exact sums (AVX2): the scalar rule (min then rint, then max 0: rint(clip(y, 0, 255)) as the
 * bounds are integers) */
Q_INLINE void qa_store(__m256i acc, const vocr_tsd_qlayer *L, const q_epi *e, unsigned p, unsigned co, uint8_t *out)
{
    const unsigned C = L->cout;
    __m256 y = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_add_epi32(acc, _mm256_loadu_si256((const __m256i *)(const void *)(L->bq + co)))),
                             _mm256_loadu_ps(L->m + co));
    __m256i q;
    __m128i q16;
    if (e->kind == Q_FLOAT) {
        if (co + 8 <= e->fstride) {
            _mm256_storeu_ps(e->f + (size_t)p * e->fstride + co, y);
        } else if (co < e->fstride) {
            float t[8];
            unsigned k;
            _mm256_storeu_ps(t, y);
            for (k = 0; co + k < e->fstride; k++) e->f[(size_t)p * e->fstride + co + k] = t[k];
        }
        return;
    }
    if (e->kind == Q_RES) {
        const __m256 r = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(const void *)(e->res + (size_t)p * C + co))));
        y = _mm256_add_ps(y, _mm256_mul_ps(r, _mm256_set1_ps(L->r)));
    } else if (e->kind == Q_UP) {
        const uint8_t *u = e->up + (size_t)((e->x0 + p) >> 1) * C + co;
        const __m256 r = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(const void *)u)));
        y = _mm256_add_ps(_mm256_max_ps(y, _mm256_setzero_ps()), _mm256_mul_ps(r, _mm256_set1_ps(L->r)));
    }
    q = _mm256_max_epi32(_mm256_cvtps_epi32(_mm256_min_ps(y, _mm256_set1_ps(255.0f))), _mm256_setzero_si256());
    q16 = _mm_packs_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256(q, 1));
    _mm_storel_epi64((__m128i *)(void *)(out + (size_t)p * C + co), _mm_packus_epi16(q16, q16));
}

#if !Q_VNNI
/* 8 outputs [co, co + 8) of 6 cells, or of 1 (AVX2, no VNNI): int16 inputs (x16 [.][Cin]) times the int8 weights
 * widened to int16, vpmaddwd (two exact products a lane), so each output's sum sits in two lanes until the end */
#define QA_ROW(r, LO, HI)                                                \
    do {                                                                 \
        long long v_;                                                    \
        __m256i a_;                                                      \
        memcpy(&v_, x16 + (size_t)(r) * Cin + 4 * j, 8);                 \
        a_ = _mm256_set1_epi64x(v_);                                     \
        LO = _mm256_add_epi32(LO, _mm256_madd_epi16(wl, a_));            \
        HI = _mm256_add_epi32(HI, _mm256_madd_epi16(wh, a_));            \
    } while (0)
#define QA_W                                                                                                   \
    const int8_t *wj = L->w + ((size_t)j * Cout + co) * 4;                                                     \
    const __m256i wl = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(const void *)wj));               \
    const __m256i wh = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(const void *)(wj + 16)))
#define QA_ST(r, LO, HI) qa_store(_mm256_permute4x64_epi64(_mm256_hadd_epi32(LO, HI), 0xD8), L, e, p + (r), co, out)

static void qa_6(const int16_t *x16, unsigned Cin, const vocr_tsd_qlayer *L, unsigned co, const q_epi *e, unsigned p,
                 uint8_t *out)
{
    const unsigned Cout = L->cout;
    const __m256i z = _mm256_setzero_si256();
    __m256i l0 = z, h0 = z, l1 = z, h1 = z, l2 = z, h2 = z, l3 = z, h3 = z, l4 = z, h4 = z, l5 = z, h5 = z;
    unsigned j;
    for (j = 0; j < Cin / 4; j++) {
        QA_W;
        QA_ROW(0, l0, h0);
        QA_ROW(1, l1, h1);
        QA_ROW(2, l2, h2);
        QA_ROW(3, l3, h3);
        QA_ROW(4, l4, h4);
        QA_ROW(5, l5, h5);
    }
    QA_ST(0, l0, h0);
    QA_ST(1, l1, h1);
    QA_ST(2, l2, h2);
    QA_ST(3, l3, h3);
    QA_ST(4, l4, h4);
    QA_ST(5, l5, h5);
}

static void qa_1(const int16_t *x16, unsigned Cin, const vocr_tsd_qlayer *L, unsigned co, const q_epi *e, unsigned p,
                 uint8_t *out)
{
    const unsigned Cout = L->cout;
    __m256i l0 = _mm256_setzero_si256(), h0 = _mm256_setzero_si256();
    unsigned j;
    for (j = 0; j < Cin / 4; j++) {
        QA_W;
        QA_ROW(0, l0, h0);
    }
    QA_ST(0, l0, h0);
}
#undef QA_ST
#undef QA_W
#undef QA_ROW
#endif
#endif

#if Q_VNNI
/* acc += a (uint8) . w (int8) four products a lane, in place: GCC 11 copies the accumulator of the intrinsic
 * (_mm512_dpbusd_epi32) to another register and back around every instruction, halving the kernels' rate; the
 * instruction is the same */
#define Q_DP(acc, a, w) __asm__("vpdpbusd %2, %1, %0" : "+v"(acc) : "v"(a), "v"(w))

/* The epilogue's value y of 16 outputs from their exact sums acc + bq (AVX-512; the VNNI sums start at bq: the same
 * int32); Q_FLOAT stores the maps here and returns 1 */
Q_INLINE int qz_y(__m512i acc, const vocr_tsd_qlayer *L, const q_epi *e, unsigned p, unsigned co, __m512 *yo)
{
    const unsigned C = L->cout;
    __m512 y = _mm512_mul_ps(_mm512_cvtepi32_ps(acc), _mm512_loadu_ps(L->m + co));
    if (e->kind == Q_FLOAT) {
        if (co < e->fstride) {
            const unsigned n = e->fstride - co >= 16 ? 16u : e->fstride - co;
            _mm512_mask_storeu_ps(e->f + (size_t)p * e->fstride + co, (__mmask16)((1u << n) - 1u), y);
        }
        return 1;
    }
    if (e->kind == Q_RES) {
        const __m512 r = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(const void *)(e->res + (size_t)p * C + co))));
        y = _mm512_add_ps(y, _mm512_mul_ps(r, _mm512_set1_ps(L->r)));
    } else if (e->kind == Q_UP) {
        const uint8_t *u = e->up + (size_t)((e->x0 + p) >> 1) * C + co;
        const __m512 r = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i *)(const void *)u)));
        y = _mm512_add_ps(_mm512_max_ps(y, _mm512_setzero_ps()), _mm512_mul_ps(r, _mm512_set1_ps(L->r)));
    }
    *yo = y;
    return 0;
}

/* rint(min(y, 255)) as int32; the packs below clamp what is negative to 0 (rint(clip(y, 0, 255)): the bounds are
 * integers) */
#define QZ_I(y) _mm512_cvtps_epi32(_mm512_min_ps(y, _mm512_set1_ps(255.0f)))

/* 16 outputs of cell p */
Q_INLINE void qz_store(__m512i acc, const vocr_tsd_qlayer *L, const q_epi *e, unsigned p, unsigned co, uint8_t *out)
{
    __m512 y;
    if (qz_y(acc, L, e, p, co, &y)) return;
    _mm_storeu_si128((__m128i *)(void *)(out + (size_t)p * L->cout + co),
                     _mm512_cvtusepi32_epi8(_mm512_max_epi32(QZ_I(y), _mm512_setzero_si512())));
}

/* 32, 48 or 64 outputs of cell p (n vectors, n >= 2; a3 unused below 4): two unsigned-saturating packs per pair
 * leave the bytes of 128-bit lane l as [a0 4 l.., a1 4 l.., a2 4 l.., a3 4 l..]; one dword permute restores the
 * channel order */
Q_INLINE void qz_storen(__m512i a0, __m512i a1, __m512i a2, __m512i a3, unsigned n, const vocr_tsd_qlayer *L,
                        const q_epi *e, unsigned p, unsigned co, uint8_t *out)
{
    __m512 y0, y1, y2 = _mm512_setzero_ps(), y3 = _mm512_setzero_ps();
    __m512i b;
    uint8_t *o = out + (size_t)p * L->cout + co;
    if (qz_y(a0, L, e, p, co, &y0)) {
        qz_y(a1, L, e, p, co + 16, &y1);
        if (n > 2) qz_y(a2, L, e, p, co + 32, &y2);
        if (n > 3) qz_y(a3, L, e, p, co + 48, &y3);
        return;
    }
    qz_y(a1, L, e, p, co + 16, &y1);
    if (n > 2) qz_y(a2, L, e, p, co + 32, &y2);
    if (n > 3) qz_y(a3, L, e, p, co + 48, &y3);
    b = _mm512_packus_epi16(_mm512_packus_epi32(QZ_I(y0), QZ_I(y1)), _mm512_packus_epi32(QZ_I(y2), QZ_I(y3)));
    b = _mm512_permutexvar_epi32(_mm512_setr_epi32(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15), b);
    if (n == 4)
        _mm512_storeu_si512((void *)o, b);
    else
        _mm512_mask_storeu_epi8((void *)o, n == 3 ? (__mmask64)0xFFFFFFFFFFFFull : (__mmask64)0xFFFFFFFFull, b);
}

/* NJ x 16 outputs from co of one cell (VNNI; the tails): per four input channels, one broadcast of the cell's four
 * input bytes and a vpdpbusd per 16 outputs (the weights' [ci / 4][co][4] order is vpdpbusd's) */
static void qz_1(const uint8_t *x, unsigned Cin, const vocr_tsd_qlayer *L, unsigned co, unsigned NJ, const q_epi *e,
                 unsigned p, uint8_t *out)
{
    const unsigned Cout = L->cout;
    __m512i acc[4];
    unsigned j, k;
    for (k = 0; k < NJ; k++) acc[k] = _mm512_loadu_si512((const void *)(L->bq + co + 16 * k));
    for (j = 0; j < Cin / 4; j++) {
        const int8_t *wj = L->w + ((size_t)j * Cout + co) * 4;
        const __m512i a = _mm512_set1_epi32(rd_i32(x + 4 * j));
        for (k = 0; k < NJ; k++) {
            const __m512i w = _mm512_loadu_si512((const void *)(wj + 64 * k));
            Q_DP(acc[k], a, w);
        }
    }
    for (k = 0; k < NJ; k++) qz_store(acc[k], L, e, p, co + 16 * k, out);
}

/* the tiles of NJ x 16 outputs and NR cells, the sums in named registers (no spills): QZ_ROWn(r, ...) adds cell r's
 * four input channels from 4 j to every sum of its row */
#define QZ_B(r) _mm512_set1_epi32(rd_i32(xr[r] + 4 * j))
#define QZ_ROW4(r, A, B, C_, D)                    \
    do {                                           \
        const __m512i v_ = QZ_B(r);                \
        Q_DP(A, v_, w0);        \
        Q_DP(B, v_, w1);        \
        Q_DP(C_, v_, w2);      \
        Q_DP(D, v_, w3);        \
    } while (0)
#define QZ_ROW3(r, A, B, C_)                       \
    do {                                           \
        const __m512i v_ = QZ_B(r);                \
        Q_DP(A, v_, w0);        \
        Q_DP(B, v_, w1);        \
        Q_DP(C_, v_, w2);      \
    } while (0)
#define QZ_ROW2(r, A, B)                           \
    do {                                           \
        const __m512i v_ = QZ_B(r);                \
        Q_DP(A, v_, w0);        \
        Q_DP(B, v_, w1);        \
    } while (0)
#define QZ_ROW1(r, A)                \
    do {                             \
        const __m512i v1_ = QZ_B(r); \
        Q_DP(A, v1_, w0);            \
    } while (0)
#define QZ_W(k) const __m512i w##k = _mm512_loadu_si512((const void *)(wj + 64 * (k)))
#define QZ_ST(A, r, k)                                                  \
    do {                                                                \
        if ((r) < nr) qz_store(A, L, e, p + (r), co + 16 * (k), out);   \
    } while (0)
#define QZ_ST4(A, B, C_, D, r)                                          \
    do {                                                                \
        if ((r) < nr) qz_storen(A, B, C_, D, 4, L, e, p + (r), co, out); \
    } while (0)
#define QZ_ST3(A, B, C_, r)                                             \
    do {                                                                \
        if ((r) < nr) qz_storen(A, B, C_, C_, 3, L, e, p + (r), co, out); \
    } while (0)
#define QZ_ST2(A, B, r)                                                 \
    do {                                                                \
        if ((r) < nr) qz_storen(A, B, B, B, 2, L, e, p + (r), co, out); \
    } while (0)
#define QZ_BQ(k) _mm512_loadu_si512((const void *)(L->bq + co + 16 * (k)))

/* 64 outputs x 6 cells */
static void qz_4x6(const uint8_t *x, unsigned Cin, const vocr_tsd_qlayer *L, unsigned co, const q_epi *e, unsigned p,
                   uint8_t *out, unsigned nr)
{
    const unsigned Cout = L->cout;
    const uint8_t *xr[6];
    unsigned r_;
    for (r_ = 0; r_ < 6; r_++) xr[r_] = x + (size_t)(r_ < nr ? r_ : nr - 1) * Cin;
    const __m512i z0 = QZ_BQ(0), z1 = QZ_BQ(1), z2 = QZ_BQ(2), z3 = QZ_BQ(3);
    __m512i a00 = z0, a01 = z1, a02 = z2, a03 = z3, a10 = z0, a11 = z1, a12 = z2, a13 = z3;
    __m512i a20 = z0, a21 = z1, a22 = z2, a23 = z3, a30 = z0, a31 = z1, a32 = z2, a33 = z3;
    __m512i a40 = z0, a41 = z1, a42 = z2, a43 = z3, a50 = z0, a51 = z1, a52 = z2, a53 = z3;
    unsigned j;
    for (j = 0; j < Cin / 4; j++) {
        const int8_t *wj = L->w + ((size_t)j * Cout + co) * 4;
        QZ_W(0);
        QZ_W(1);
        QZ_W(2);
        QZ_W(3);
        QZ_ROW4(0, a00, a01, a02, a03);
        QZ_ROW4(1, a10, a11, a12, a13);
        QZ_ROW4(2, a20, a21, a22, a23);
        QZ_ROW4(3, a30, a31, a32, a33);
        QZ_ROW4(4, a40, a41, a42, a43);
        QZ_ROW4(5, a50, a51, a52, a53);
    }
    QZ_ST4(a00, a01, a02, a03, 0);
    QZ_ST4(a10, a11, a12, a13, 1);
    QZ_ST4(a20, a21, a22, a23, 2);
    QZ_ST4(a30, a31, a32, a33, 3);
    QZ_ST4(a40, a41, a42, a43, 4);
    QZ_ST4(a50, a51, a52, a53, 5);
}

/* 48 outputs x 8 cells */
static void qz_3x8(const uint8_t *x, unsigned Cin, const vocr_tsd_qlayer *L, unsigned co, const q_epi *e, unsigned p,
                   uint8_t *out, unsigned nr)
{
    const unsigned Cout = L->cout;
    const uint8_t *xr[8];
    unsigned r_;
    for (r_ = 0; r_ < 8; r_++) xr[r_] = x + (size_t)(r_ < nr ? r_ : nr - 1) * Cin;
    const __m512i z0 = QZ_BQ(0), z1 = QZ_BQ(1), z2 = QZ_BQ(2);
    __m512i a00 = z0, a01 = z1, a02 = z2, a10 = z0, a11 = z1, a12 = z2, a20 = z0, a21 = z1, a22 = z2;
    __m512i a30 = z0, a31 = z1, a32 = z2, a40 = z0, a41 = z1, a42 = z2, a50 = z0, a51 = z1, a52 = z2;
    __m512i a60 = z0, a61 = z1, a62 = z2, a70 = z0, a71 = z1, a72 = z2;
    unsigned j;
    for (j = 0; j < Cin / 4; j++) {
        const int8_t *wj = L->w + ((size_t)j * Cout + co) * 4;
        QZ_W(0);
        QZ_W(1);
        QZ_W(2);
        QZ_ROW3(0, a00, a01, a02);
        QZ_ROW3(1, a10, a11, a12);
        QZ_ROW3(2, a20, a21, a22);
        QZ_ROW3(3, a30, a31, a32);
        QZ_ROW3(4, a40, a41, a42);
        QZ_ROW3(5, a50, a51, a52);
        QZ_ROW3(6, a60, a61, a62);
        QZ_ROW3(7, a70, a71, a72);
    }
    QZ_ST3(a00, a01, a02, 0);
    QZ_ST3(a10, a11, a12, 1);
    QZ_ST3(a20, a21, a22, 2);
    QZ_ST3(a30, a31, a32, 3);
    QZ_ST3(a40, a41, a42, 4);
    QZ_ST3(a50, a51, a52, 5);
    QZ_ST3(a60, a61, a62, 6);
    QZ_ST3(a70, a71, a72, 7);
}

/* 32 outputs x 12 cells */
static void qz_2x12(const uint8_t *x, unsigned Cin, const vocr_tsd_qlayer *L, unsigned co, const q_epi *e, unsigned p,
                    uint8_t *out, unsigned nr)
{
    const unsigned Cout = L->cout;
    const uint8_t *xr[12];
    unsigned r_;
    for (r_ = 0; r_ < 12; r_++) xr[r_] = x + (size_t)(r_ < nr ? r_ : nr - 1) * Cin;
    const __m512i z0 = QZ_BQ(0), z1 = QZ_BQ(1);
    __m512i a00 = z0, a01 = z1, a10 = z0, a11 = z1, a20 = z0, a21 = z1, a30 = z0, a31 = z1, a40 = z0, a41 = z1;
    __m512i a50 = z0, a51 = z1, a60 = z0, a61 = z1, a70 = z0, a71 = z1, a80 = z0, a81 = z1, a90 = z0, a91 = z1;
    __m512i aa0 = z0, aa1 = z1, ab0 = z0, ab1 = z1;
    unsigned j;
    for (j = 0; j < Cin / 4; j++) {
        const int8_t *wj = L->w + ((size_t)j * Cout + co) * 4;
        QZ_W(0);
        QZ_W(1);
        QZ_ROW2(0, a00, a01);
        QZ_ROW2(1, a10, a11);
        QZ_ROW2(2, a20, a21);
        QZ_ROW2(3, a30, a31);
        QZ_ROW2(4, a40, a41);
        QZ_ROW2(5, a50, a51);
        QZ_ROW2(6, a60, a61);
        QZ_ROW2(7, a70, a71);
        QZ_ROW2(8, a80, a81);
        QZ_ROW2(9, a90, a91);
        QZ_ROW2(10, aa0, aa1);
        QZ_ROW2(11, ab0, ab1);
    }
    QZ_ST2(a00, a01, 0);
    QZ_ST2(a10, a11, 1);
    QZ_ST2(a20, a21, 2);
    QZ_ST2(a30, a31, 3);
    QZ_ST2(a40, a41, 4);
    QZ_ST2(a50, a51, 5);
    QZ_ST2(a60, a61, 6);
    QZ_ST2(a70, a71, 7);
    QZ_ST2(a80, a81, 8);
    QZ_ST2(a90, a91, 9);
    QZ_ST2(aa0, aa1, 10);
    QZ_ST2(ab0, ab1, 11);
}

/* 16 outputs x 12 cells */
static void qz_1x12(const uint8_t *x, unsigned Cin, const vocr_tsd_qlayer *L, unsigned co, const q_epi *e, unsigned p,
                    uint8_t *out, unsigned nr)
{
    const unsigned Cout = L->cout;
    const uint8_t *xr[12];
    unsigned r_;
    for (r_ = 0; r_ < 12; r_++) xr[r_] = x + (size_t)(r_ < nr ? r_ : nr - 1) * Cin;
    const __m512i z = QZ_BQ(0);
    __m512i a0 = z, a1 = z, a2 = z, a3 = z, a4 = z, a5 = z, a6 = z, a7 = z, a8 = z, a9 = z, aa = z, ab = z;
    unsigned j;
    for (j = 0; j < Cin / 4; j++) {
        const int8_t *wj = L->w + ((size_t)j * Cout + co) * 4;
        QZ_W(0);
        QZ_ROW1(0, a0);
        QZ_ROW1(1, a1);
        QZ_ROW1(2, a2);
        QZ_ROW1(3, a3);
        QZ_ROW1(4, a4);
        QZ_ROW1(5, a5);
        QZ_ROW1(6, a6);
        QZ_ROW1(7, a7);
        QZ_ROW1(8, a8);
        QZ_ROW1(9, a9);
        QZ_ROW1(10, aa);
        QZ_ROW1(11, ab);
    }
    QZ_ST(a0, 0, 0); QZ_ST(a1, 1, 0); QZ_ST(a2, 2, 0); QZ_ST(a3, 3, 0); QZ_ST(a4, 4, 0); QZ_ST(a5, 5, 0);
    QZ_ST(a6, 6, 0); QZ_ST(a7, 7, 0); QZ_ST(a8, 8, 0); QZ_ST(a9, 9, 0); QZ_ST(aa, 10, 0); QZ_ST(ab, 11, 0);
}
#undef QZ_BQ
#undef QZ_ST4
#undef QZ_ST3
#undef QZ_ST2
#undef QZ_ST
#undef QZ_W
#undef QZ_ROW1
#undef QZ_ROW2
#undef QZ_ROW3
#undef QZ_ROW4
#undef QZ_B

/* 8 outputs from co (an 8-channel remainder) of NR cells: vpdpbusd on 256-bit vectors */
Q_INLINE void qy_tile(const uint8_t *x, unsigned Cin, const vocr_tsd_qlayer *L, unsigned co, unsigned NR,
                      const q_epi *e, unsigned p, uint8_t *out)
{
    const unsigned Cout = L->cout;
    __m256i acc[12];
    unsigned j, r;
    for (r = 0; r < NR; r++) acc[r] = _mm256_setzero_si256();
    for (j = 0; j < Cin / 4; j++) {
        const __m256i w = _mm256_loadu_si256((const __m256i *)(const void *)(L->w + ((size_t)j * Cout + co) * 4));
        for (r = 0; r < NR; r++) {
            const __m256i a = _mm256_set1_epi32(rd_i32(x + (size_t)r * Cin + 4 * j));
            Q_DP(acc[r], a, w);
        }
    }
    for (r = 0; r < NR; r++) qa_store(acc[r], L, e, p + r, co, out);
}
#endif

/* out = epi(in W) for P cells (in: [P][Cin] uint8 contiguous, P <= Q_CHUNK) */
static void q_pw(const uint8_t *in, unsigned P, const vocr_tsd_qlayer *L, const q_epi *e, uint8_t *out)
{
    const unsigned Cout = L->cout;
    unsigned co = 0;
#if Q_AVX2
    const unsigned Cin = L->cin;
    unsigned p;
#endif
#if Q_VNNI
    while (co + 16 <= Cout) {
        const unsigned nj = (Cout - co) / 16 >= 4 ? 4u : (Cout - co) / 16;
        p = 0;
        /* the last tile of a chunk padded: its rows past the chunk read the last cell's input and are not stored
         * (one cell alone: qz_1) */
        if (nj == 4)
            for (; p + 1 < P; p += 6) qz_4x6(in + (size_t)p * Cin, Cin, L, co, e, p, out, P - p < 6 ? P - p : 6);
        else if (nj == 3)
            for (; p + 1 < P; p += 8) qz_3x8(in + (size_t)p * Cin, Cin, L, co, e, p, out, P - p < 8 ? P - p : 8);
        else if (nj == 2)
            for (; p + 1 < P; p += 12) qz_2x12(in + (size_t)p * Cin, Cin, L, co, e, p, out, P - p < 12 ? P - p : 12);
        else
            for (; p + 1 < P; p += 12) qz_1x12(in + (size_t)p * Cin, Cin, L, co, e, p, out, P - p < 12 ? P - p : 12);
        for (; p < P; p++) qz_1(in + (size_t)p * Cin, Cin, L, co, nj, e, p, out);
        co += 16 * nj;
    }
    if (co + 8 <= Cout) {
        for (p = 0; p + 12 <= P; p += 12) qy_tile(in + (size_t)p * Cin, Cin, L, co, 12, e, p, out);
        for (; p < P; p++) qy_tile(in + (size_t)p * Cin, Cin, L, co, 1, e, p, out);
        co += 8;
    }
#elif Q_AVX2
    if (co + 8 <= Cout) {
        int16_t x16[Q_CHUNK * VOCR_TSD_MAX_CHANNELS];
        size_t i, n = (size_t)P * Cin;
        for (i = 0; i + 16 <= n; i += 16)
            _mm256_storeu_si256((__m256i *)(void *)(x16 + i),
                                _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i *)(const void *)(in + i))));
        for (; i < n; i++) x16[i] = in[i];
        for (; co + 8 <= Cout; co += 8) {
            for (p = 0; p + 6 <= P; p += 6) qa_6(x16 + (size_t)p * Cin, Cin, L, co, e, p, out);
            for (; p < P; p++) qa_1(x16 + (size_t)p * Cin, Cin, L, co, e, p, out);
        }
    }
#endif
    if (co < Cout) q_pw_c(in, P, L, e, co, Cout, out);
}

/* ----------------------------------------------------------------------------- depthwise layers */

#if !(Q_AVX2 && !Q_VNNI)
/* plain C: output columns [x0, x1) of one row into out ([x1 - x0][C]), channels [c0, c1): rows[t] is the input row of
 * tap row kys[t] (t < nk: the rows inside the map), zero padding at the sides */
static void q_dw_c(const uint8_t *const *rows, const unsigned *kys, unsigned nk, unsigned W, const vocr_tsd_qlayer *L,
                   unsigned x0, unsigned x1, unsigned c0, unsigned c1, uint8_t *out)
{
    const unsigned C = L->cin, s = L->stride, d = L->dil;
    int32_t acc[VOCR_TSD_MAX_CHANNELS];
    unsigned ox, c, t, kx;
    for (ox = x0; ox < x1; ox++) {
        for (c = c0; c < c1; c++) acc[c] = 0;
        for (t = 0; t < nk; t++)
            for (kx = 0; kx < 3; kx++) {
                const int ix = (int)(ox * s + kx * d) - (int)d;
                const uint8_t *px;
                const int8_t *wk;
                if (ix < 0 || ix >= (int)W) continue;
                px = rows[t] + (size_t)ix * C;
                wk = L->w + (size_t)(kys[t] * 3 + kx) * C;
                for (c = c0; c < c1; c++) acc[c] += (int32_t)px[c] * wk[c];
            }
        for (c = c0; c < c1; c++)
            out[(size_t)(ox - x0) * C + c] = q_round((float)(acc[c] + L->bq[c]) * L->m[c]);
    }
}
#endif

#if Q_VNNI
/* The VNNI depthwise works on a vertically expanded row: for output row y and input column ix, channel c's four
 * bytes (in[y s - d][ix][c], in[y s][ix][c], in[y s + d][ix][c], 0) (0 outside the map), so tap column kx of a cell
 * is one vpdpbusd against (w[0][kx][c], w[1][kx][c], w[2][kx][c], 0) at ix = ox s + (kx - 1) d. The bytes are interleaved by unpacks within 128-bit lanes, so a
 * group of 64 (32, 16) channels holds four vectors, vector k's lane i of 128-bit lane l being channel 16 l + 4 k + i;
 * the weights, bq and m of the group take the same order, and two packs bring the outputs back to channel order. */
typedef struct {
    unsigned off, width;              /* the group's first channel and width (64, 32 or 16) */
} dw_group;

typedef struct {
    unsigned ngroups, simd_c;         /* the channels [0, simd_c) in groups, [simd_c, C) plain C */
    dw_group g[VOCR_TSD_MAX_CHANNELS / 16];
    int8_t w[3][VOCR_TSD_MAX_CHANNELS * 4];        /* per tap column kx: (w[0][kx], w[1][kx], w[2][kx], 0) */
    int32_t bq[VOCR_TSD_MAX_CHANNELS];             /* in the groups' order */
    float m[VOCR_TSD_MAX_CHANNELS];
} dw_prep;

/* interleave four byte vectors' channels (a, b, c, zero) as the expanded order: e[k] = vector k */
#define DW_EXPAND(T, U8LO, U8HI, U16LO, U16HI, a, b, c, z, e)            \
    do {                                                                \
        const T ab_lo_ = U8LO(a, b), ab_hi_ = U8HI(a, b);               \
        const T c_lo_ = U8LO(c, z), c_hi_ = U8HI(c, z);                 \
        (e)[0] = U16LO(ab_lo_, c_lo_);                                  \
        (e)[1] = U16HI(ab_lo_, c_lo_);                                  \
        (e)[2] = U16LO(ab_hi_, c_hi_);                                  \
        (e)[3] = U16HI(ab_hi_, c_hi_);                                  \
    } while (0)

/* the position of channel `local` (0 .. width - 1) of a group in the expanded order: vector k, lane l, element i */
static unsigned dw_slot(unsigned local, unsigned width)
{
    const unsigned l = local / 16, k = (local % 16) / 4, i = local % 4, lanes = width / 16;
    return k * (4 * lanes) + l * 4 + i;   /* element index within the group's four vectors of `width / 4` int32 */
}

static void dw_prepare(const vocr_tsd_qlayer *L, dw_prep *D)
{
    const unsigned C = L->cin;
    unsigned c = 0, ky, j;
    D->ngroups = 0;
    while (c + 16 <= C) {
        const unsigned width = C - c >= 64 ? 64u : C - c >= 32 ? 32u : 16u;
        D->g[D->ngroups].off = c;
        D->g[D->ngroups].width = width;
        D->ngroups++;
        for (j = 0; j < width; j++) {
            const unsigned slot = dw_slot(j, width);
            D->bq[c + slot] = L->bq[c + j];
            D->m[c + slot] = L->m[c + j];
            for (ky = 0; ky < 3; ky++) {      /* here ky runs over the tap columns kx */
                int8_t *o = D->w[ky] + 4 * (size_t)(c + slot);
                o[0] = L->w[(size_t)ky * C + c + j];
                o[1] = L->w[(size_t)(3 + ky) * C + c + j];
                o[2] = L->w[(size_t)(6 + ky) * C + c + j];
                o[3] = 0;
            }
        }
        c += width;
    }
    D->simd_c = c;
}

/* a group's vertically expanded bytes for the input columns ia0 .. ia0 + ncols - 1 (V: a cell every rowb bytes, the
 * group at its offset): top, mid, bot the three input rows at the group's channel (NULL: outside the map), the
 * columns outside [0, W) zero */
#define VEXP_GROUP(NAME, T, LD, ST, U8LO, U8HI, U16LO, U16HI, SETZ, B)                                             \
    static void NAME(const uint8_t *top, const uint8_t *mid, const uint8_t *bot, unsigned W, unsigned C, int ia0,  \
                     unsigned ncols, size_t rowb, uint8_t *V)                                                     \
    {                                                                                                             \
        const T z = SETZ();                                                                                       \
        const int ia1 = ia0 + (int)ncols;                                                                         \
        const int lo = ia0 > 0 ? ia0 : 0, hi = ia1 < (int)W ? ia1 : (int)W;                                        \
        int ix;                                                                                                   \
        uint8_t *o = V;                                                                                           \
        for (ix = ia0; ix < (lo < ia1 ? lo : ia1); ix++, o += rowb) {                                              \
            ST(o, z);                                                                                             \
            ST(o + (B), z);                                                                                       \
            ST(o + 2 * (B), z);                                                                                   \
            ST(o + 3 * (B), z);                                                                                   \
        }                                                                                                         \
        for (ix = lo; ix < hi; ix++, o += rowb) {                                                                 \
            const size_t off = (size_t)ix * C;                                                                    \
            const T a_ = top ? LD(top + off) : z, b_ = LD(mid + off), c_ = bot ? LD(bot + off) : z;               \
            T e_[4];                                                                                              \
            DW_EXPAND(T, U8LO, U8HI, U16LO, U16HI, a_, b_, c_, z, e_);                                            \
            ST(o, e_[0]);                                                                                         \
            ST(o + (B), e_[1]);                                                                                   \
            ST(o + 2 * (B), e_[2]);                                                                               \
            ST(o + 3 * (B), e_[3]);                                                                               \
        }                                                                                                         \
        for (ix = hi > lo ? hi : (lo > ia0 ? lo : ia0); ix < ia1; ix++, o += rowb) {                               \
            ST(o, z);                                                                                             \
            ST(o + (B), z);                                                                                       \
            ST(o + 2 * (B), z);                                                                                   \
            ST(o + 3 * (B), z);                                                                                   \
        }                                                                                                         \
    }

#define XLD512(p) _mm512_loadu_si512((const void *)(p))
#define XST512(p, v) _mm512_storeu_si512((void *)(p), v)
#define XLD256(p) _mm256_loadu_si256((const __m256i *)(const void *)(p))
#define XST256(p, v) _mm256_storeu_si256((__m256i *)(void *)(p), v)
#define XLD128(p) _mm_loadu_si128((const __m128i *)(const void *)(p))
#define XST128(p, v) _mm_storeu_si128((__m128i *)(void *)(p), v)
VEXP_GROUP(vexp_g64, __m512i, XLD512, XST512, _mm512_unpacklo_epi8, _mm512_unpackhi_epi8, _mm512_unpacklo_epi16,
           _mm512_unpackhi_epi16, _mm512_setzero_si512, 64)
VEXP_GROUP(vexp_g32, __m256i, XLD256, XST256, _mm256_unpacklo_epi8, _mm256_unpackhi_epi8, _mm256_unpacklo_epi16,
           _mm256_unpackhi_epi16, _mm256_setzero_si256, 32)
VEXP_GROUP(vexp_g16, __m128i, XLD128, XST128, _mm_unpacklo_epi8, _mm_unpackhi_epi8, _mm_unpacklo_epi16,
           _mm_unpackhi_epi16, _mm_setzero_si128, 16)
#undef VEXP_GROUP

/* the vertically expanded row of output row y's three input rows (rows[ky]: in[y s + (ky - 1) d], NULL outside the
 * map) for the input columns ia0 .. ia0 + ncols - 1 into V ([ncols][simd_c][4]) */
static void dw_vexpand(const uint8_t *const *rows, unsigned W, const vocr_tsd_qlayer *L, const dw_prep *D, int ia0,
                       unsigned ncols, uint8_t *V)
{
    const unsigned C = L->cin;
    const size_t rowb = (size_t)D->simd_c * 4;
    unsigned gi;
    for (gi = 0; gi < D->ngroups; gi++) {
        const unsigned c = D->g[gi].off, wd = D->g[gi].width;
        const uint8_t *t0 = rows[0] ? rows[0] + c : NULL, *t1 = rows[1] + c, *t2 = rows[2] ? rows[2] + c : NULL;
        if (wd == 64)
            vexp_g64(t0, t1, t2, W, C, ia0, ncols, rowb, V + 4 * (size_t)c);
        else if (wd == 32)
            vexp_g32(t0, t1, t2, W, C, ia0, ncols, rowb, V + 4 * (size_t)c);
        else
            vexp_g16(t0, t1, t2, W, C, ia0, ncols, rowb, V + 4 * (size_t)c);
    }
}

/* One group of channels over the n cells of a strip: the vertically expanded row at the three tap columns e0, e1, e2
 * (rowb bytes from one output cell to the next: s cells of V) against the group's weights w0, w1, w2 (D->w[kx] at the
 * group), the group's bq (the sums start there) and m in the expanded order; the weights, bq and m stay in
 * registers across the cells.
 * VT / VF: the integer / float vector type, B its bytes; the outputs to o (C bytes a cell) in channel order. */
#define DW_GROUP(NAME, VT, VF, B, LD, LDF, ST, SETZ, CVT, MUL, MIN, SET1F, CVTI, MAXI, PACKS, PACKUS)        \
    static void NAME(const uint8_t *e0, const uint8_t *e1, const uint8_t *e2, size_t rowb, const int8_t *w0,      \
                     const int8_t *w1, const int8_t *w2, const int32_t *bq, const float *mm, unsigned n,          \
                     uint8_t *o, size_t C)  /* rowb: the bytes from one output cell's taps to the next's */       \
    {                                                                                                              \
        const VT W00 = LD(w0), W01 = LD(w0 + B), W02 = LD(w0 + 2 * B), W03 = LD(w0 + 3 * B);                      \
        const VT W10 = LD(w1), W11 = LD(w1 + B), W12 = LD(w1 + 2 * B), W13 = LD(w1 + 3 * B);                      \
        const VT W20 = LD(w2), W21 = LD(w2 + B), W22 = LD(w2 + 2 * B), W23 = LD(w2 + 3 * B);                      \
        const VT B0 = LD(bq), B1 = LD(bq + B / 4), B2 = LD(bq + B / 2), B3 = LD(bq + 3 * B / 4);                  \
        const VF M0 = LDF(mm), M1 = LDF(mm + B / 4), M2 = LDF(mm + B / 2), M3 = LDF(mm + 3 * B / 4);              \
        const VF K255 = SET1F(255.0f);                                                                             \
        const VT Z = SETZ();                                                                                       \
        unsigned p = 0;                                                                                            \
        /* two cells a step (8 independent sums); the requantization: rint(min(y, 255)) then the saturating packs, \
         * which clamp a negative value to 0 (rint(clip(y, 0, 255))) */                                             \
        for (; p + 2 <= n; p += 2, e0 += 2 * rowb, e1 += 2 * rowb, e2 += 2 * rowb, o += 2 * C) {                   \
            VT a0 = B0, a1 = B1, a2 = B2, a3 = B3, c0 = B0, c1 = B1, c2 = B2, c3 = B3, t, u;                          \
            t = LD(e0); u = LD(e0 + rowb); Q_DP(a0, t, W00); Q_DP(c0, u, W00);                                       \
            t = LD(e0 + B); u = LD(e0 + rowb + B); Q_DP(a1, t, W01); Q_DP(c1, u, W01);                               \
            t = LD(e0 + 2 * B); u = LD(e0 + rowb + 2 * B); Q_DP(a2, t, W02); Q_DP(c2, u, W02);                       \
            t = LD(e0 + 3 * B); u = LD(e0 + rowb + 3 * B); Q_DP(a3, t, W03); Q_DP(c3, u, W03);                       \
            t = LD(e1); u = LD(e1 + rowb); Q_DP(a0, t, W10); Q_DP(c0, u, W10);                                       \
            t = LD(e1 + B); u = LD(e1 + rowb + B); Q_DP(a1, t, W11); Q_DP(c1, u, W11);                               \
            t = LD(e1 + 2 * B); u = LD(e1 + rowb + 2 * B); Q_DP(a2, t, W12); Q_DP(c2, u, W12);                       \
            t = LD(e1 + 3 * B); u = LD(e1 + rowb + 3 * B); Q_DP(a3, t, W13); Q_DP(c3, u, W13);                       \
            t = LD(e2); u = LD(e2 + rowb); Q_DP(a0, t, W20); Q_DP(c0, u, W20);                                       \
            t = LD(e2 + B); u = LD(e2 + rowb + B); Q_DP(a1, t, W21); Q_DP(c1, u, W21);                               \
            t = LD(e2 + 2 * B); u = LD(e2 + rowb + 2 * B); Q_DP(a2, t, W22); Q_DP(c2, u, W22);                       \
            t = LD(e2 + 3 * B); u = LD(e2 + rowb + 3 * B); Q_DP(a3, t, W23); Q_DP(c3, u, W23);                       \
            a0 = CVTI(MIN(MUL(CVT(a0), M0), K255)); c0 = CVTI(MIN(MUL(CVT(c0), M0), K255));                         \
            a1 = CVTI(MIN(MUL(CVT(a1), M1), K255)); c1 = CVTI(MIN(MUL(CVT(c1), M1), K255));                         \
            a2 = CVTI(MIN(MUL(CVT(a2), M2), K255)); c2 = CVTI(MIN(MUL(CVT(c2), M2), K255));                         \
            a3 = CVTI(MIN(MUL(CVT(a3), M3), K255)); c3 = CVTI(MIN(MUL(CVT(c3), M3), K255));                         \
            ST(o, PACKUS(PACKS(a0, a1), PACKS(a2, a3)));                                                           \
            ST(o + C, PACKUS(PACKS(c0, c1), PACKS(c2, c3)));                                                       \
        }                                                                                                          \
        for (; p < n; p++, e0 += rowb, e1 += rowb, e2 += rowb, o += C) {                                          \
            VT a0 = B0, a1 = B1, a2 = B2, a3 = B3, t;                                                              \
            t = LD(e0); Q_DP(a0, t, W00);                                                                          \
            t = LD(e0 + B); Q_DP(a1, t, W01);                                                                      \
            t = LD(e0 + 2 * B); Q_DP(a2, t, W02);                                                                  \
            t = LD(e0 + 3 * B); Q_DP(a3, t, W03);                                                                  \
            t = LD(e1); Q_DP(a0, t, W10);                                                                          \
            t = LD(e1 + B); Q_DP(a1, t, W11);                                                                      \
            t = LD(e1 + 2 * B); Q_DP(a2, t, W12);                                                                  \
            t = LD(e1 + 3 * B); Q_DP(a3, t, W13);                                                                  \
            t = LD(e2); Q_DP(a0, t, W20);                                                                          \
            t = LD(e2 + B); Q_DP(a1, t, W21);                                                                      \
            t = LD(e2 + 2 * B); Q_DP(a2, t, W22);                                                                  \
            t = LD(e2 + 3 * B); Q_DP(a3, t, W23);                                                                  \
            a0 = CVTI(MIN(MUL(CVT(a0), M0), K255));                                                                \
            a1 = CVTI(MIN(MUL(CVT(a1), M1), K255));                                                                \
            a2 = CVTI(MIN(MUL(CVT(a2), M2), K255));                                                                \
            a3 = CVTI(MIN(MUL(CVT(a3), M3), K255));                                                                \
            ST(o, PACKUS(PACKS(a0, a1), PACKS(a2, a3)));                                                           \
        }                                                                                                          \
        (void)Z;                                                                                                   \
    }

#define LD512(p) _mm512_loadu_si512((const void *)(p))
#define LDF512(p) _mm512_loadu_ps(p)
#define ST512(p, v) _mm512_storeu_si512((void *)(p), v)
#define LD256(p) _mm256_loadu_si256((const __m256i *)(const void *)(p))
#define LDF256(p) _mm256_loadu_ps(p)
#define ST256(p, v) _mm256_storeu_si256((__m256i *)(void *)(p), v)
#define LD128(p) _mm_loadu_si128((const __m128i *)(const void *)(p))
#define LDF128(p) _mm_loadu_ps(p)
#define ST128(p, v) _mm_storeu_si128((__m128i *)(void *)(p), v)
DW_GROUP(dw_g64, __m512i, __m512, 64, LD512, LDF512, ST512, _mm512_setzero_si512, _mm512_cvtepi32_ps,
         _mm512_mul_ps, _mm512_min_ps, _mm512_set1_ps, _mm512_cvtps_epi32, _mm512_max_epi32, _mm512_packs_epi32,
         _mm512_packus_epi16)
DW_GROUP(dw_g32, __m256i, __m256, 32, LD256, LDF256, ST256, _mm256_setzero_si256, _mm256_cvtepi32_ps,
         _mm256_mul_ps, _mm256_min_ps, _mm256_set1_ps, _mm256_cvtps_epi32, _mm256_max_epi32, _mm256_packs_epi32,
         _mm256_packus_epi16)
DW_GROUP(dw_g16, __m128i, __m128, 16, LD128, LDF128, ST128, _mm_setzero_si128, _mm_cvtepi32_ps,
         _mm_mul_ps, _mm_min_ps, _mm_set1_ps, _mm_cvtps_epi32, _mm_max_epi32, _mm_packs_epi32, _mm_packus_epi16)
#undef DW_GROUP

/* depthwise output columns of one row of a strip (n cells) into out ([n][C]) from its vertically expanded row V (the
 * strip's input columns from x0 s - d, V's cell 0) */
static void dw_vnni(const uint8_t *V, const vocr_tsd_qlayer *L, const dw_prep *D, unsigned n, uint8_t *out)
{
    const unsigned C = L->cin, s = L->stride, d = L->dil;
    const size_t rowb = (size_t)D->simd_c * 4, step = (size_t)s * rowb;
    unsigned gi;
    for (gi = 0; gi < D->ngroups; gi++) {
        const unsigned c = D->g[gi].off, wd = D->g[gi].width;
        const size_t b = 4 * (size_t)c;
        const uint8_t *e0 = V + b, *e1 = V + (size_t)d * rowb + b, *e2 = V + 2 * (size_t)d * rowb + b;
        if (wd == 64)
            dw_g64(e0, e1, e2, step, D->w[0] + b, D->w[1] + b, D->w[2] + b, D->bq + c, D->m + c, n, out + c, C);
        else if (wd == 32)
            dw_g32(e0, e1, e2, step, D->w[0] + b, D->w[1] + b, D->w[2] + b, D->bq + c, D->m + c, n, out + c, C);
        else
            dw_g16(e0, e1, e2, step, D->w[0] + b, D->w[1] + b, D->w[2] + b, D->bq + c, D->m + c, n, out + c, C);
    }
}
#endif

#if Q_AVX2 && !Q_VNNI
/* The AVX2 depthwise (no VNNI): float fused multiply-adds of integer values are exact here (uint8 inputs times int8
 * weights, every partial sum an integer of magnitude below 2^24), so the float sums are the int32 sums whatever the
 * order; the strip's input rows are converted to float once per strip and row (a ring per task, like the VNNI path's
 * expanded rows), the weights once per step. Then the module comment's requantization. */
#define DWF_RING_ROWS 5u
#define DWF_ROW_FLOATS ((2u * Q_CHUNK + 2u * 16u) * VOCR_TSD_MAX_CHANNELS)
#define Q_FRING_BYTES (DWF_RING_ROWS * DWF_ROW_FLOATS * 4u)

/* input columns [ia0, ia1) of a row (W x C uint8) as floats */
static void dwf_row(const uint8_t *row, unsigned C, unsigned ia0, unsigned ia1, float *o)
{
    const uint8_t *r = row + (size_t)ia0 * C;
    const size_t n = (size_t)(ia1 - ia0) * C;
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(o + i, _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(const void *)(r + i)))));
    for (; i < n; i++) o[i] = (float)r[i];
}

/* 8 channels' exact sums (float) to bytes */
static inline void dwf_store(__m256 acc, const int32_t *bq, const float *mm, uint8_t *o)
{
    const __m256i a = _mm256_add_epi32(_mm256_cvtps_epi32(acc), _mm256_loadu_si256((const __m256i *)(const void *)bq));
    const __m256 y = _mm256_mul_ps(_mm256_cvtepi32_ps(a), _mm256_loadu_ps(mm));
    const __m256i q = _mm256_max_epi32(_mm256_cvtps_epi32(_mm256_min_ps(y, _mm256_set1_ps(255.0f))),
                                       _mm256_setzero_si256());
    const __m128i q16 = _mm_packs_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256(q, 1));
    _mm_storel_epi64((__m128i *)(void *)o, _mm_packus_epi16(q16, q16));
}

/* depthwise output columns [x0, x1) of one row into out ([x1 - x0][C]) from float rows of input columns [ia0, .)
 * (rows[t] for t < nk, tap row kys[t]) and float weights wf ([9][C]): the columns whose taps are all inside the map
 * unchecked, the others checked (a tap outside adds nothing) */
static void dwf_avx2(const float *const *rows, const unsigned *kys, unsigned nk, unsigned W, unsigned ia0,
                     const vocr_tsd_qlayer *L, const float *wf, unsigned x0, unsigned x1, uint8_t *out)
{
    const unsigned C = L->cin, s = L->stride, d = L->dil;
    unsigned lo, hi, ilo, ihi, ox, c, t;
    ilo = (d + s - 1) / s;
    ihi = W > d ? (W - 1 - d) / s + 1 : 0;
    lo = ilo > x0 ? ilo : x0;
    hi = ihi < x1 ? ihi : x1;
    if (hi < lo) lo = hi = x0;
    for (c = 0; c < C; c += 8) {
        unsigned side;
        for (side = 0; side < 2; side++) {
            const unsigned a0 = side ? hi : x0, a1 = side ? x1 : lo;
            for (ox = a0; ox < a1; ox++) {
                __m256 a = _mm256_setzero_ps();
                for (t = 0; t < nk; t++) {
                    unsigned kx;
                    for (kx = 0; kx < 3; kx++) {
                        const int ix = (int)(ox * s + kx * d) - (int)d;
                        if (ix < 0 || ix >= (int)W) continue;
                        a = _mm256_fmadd_ps(_mm256_loadu_ps(rows[t] + (size_t)(ix - (int)ia0) * C + c),
                                            _mm256_loadu_ps(wf + (size_t)(kys[t] * 3 + kx) * C + c), a);
                    }
                }
                dwf_store(a, L->bq + c, L->m + c, out + (size_t)(ox - x0) * C + c);
            }
        }
        if (nk == 3) {
            const __m256 w00 = _mm256_loadu_ps(wf + c), w01 = _mm256_loadu_ps(wf + C + c);
            const __m256 w02 = _mm256_loadu_ps(wf + 2 * C + c), w10 = _mm256_loadu_ps(wf + 3 * C + c);
            const __m256 w11 = _mm256_loadu_ps(wf + 4 * C + c), w12 = _mm256_loadu_ps(wf + 5 * C + c);
            const __m256 w20 = _mm256_loadu_ps(wf + 6 * C + c), w21 = _mm256_loadu_ps(wf + 7 * C + c);
            const __m256 w22 = _mm256_loadu_ps(wf + 8 * C + c);
            const size_t step = (size_t)d * C, xs = (size_t)s * C;
            const size_t base = (size_t)(lo * s - d - ia0) * C + c;
            const float *r0 = rows[0] + base, *r1 = rows[1] + base, *r2 = rows[2] + base;
            for (ox = lo; ox < hi; ox++, r0 += xs, r1 += xs, r2 += xs) {
                __m256 a = _mm256_mul_ps(_mm256_loadu_ps(r0), w00);
                a = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + step), w01, a);
                a = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + 2 * step), w02, a);
                a = _mm256_fmadd_ps(_mm256_loadu_ps(r1), w10, a);
                a = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + step), w11, a);
                a = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + 2 * step), w12, a);
                a = _mm256_fmadd_ps(_mm256_loadu_ps(r2), w20, a);
                a = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + step), w21, a);
                a = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + 2 * step), w22, a);
                dwf_store(a, L->bq + c, L->m + c, out + (size_t)(ox - x0) * C + c);
            }
        } else {
            const size_t step = (size_t)d * C;
            for (ox = lo; ox < hi; ox++) {
                __m256 a = _mm256_setzero_ps();
                for (t = 0; t < nk; t++) {
                    const float *r = rows[t] + (size_t)(ox * s - d - ia0) * C + c;
                    const float *wk = wf + (size_t)kys[t] * 3 * C + c;
                    a = _mm256_fmadd_ps(_mm256_loadu_ps(r), _mm256_loadu_ps(wk), a);
                    a = _mm256_fmadd_ps(_mm256_loadu_ps(r + step), _mm256_loadu_ps(wk + C), a);
                    a = _mm256_fmadd_ps(_mm256_loadu_ps(r + 2 * step), _mm256_loadu_ps(wk + 2 * C), a);
                }
                dwf_store(a, L->bq + c, L->m + c, out + (size_t)(ox - x0) * C + c);
            }
        }
    }
}
#endif

/* ----------------------------------------------------------------------------- the network */

enum { QS_STEM, QS_SEP, QS_LAT, QS_GC };

typedef struct {
    int kind, res, head;              /* res: a residual block; head: the head after this step (into the maps) */
    const vocr_tsd_qlayer *a, *b;     /* SEP: depthwise, pointwise; LAT: the lateral; STEM: the stem */
    const uint8_t *in;                /* SEP / LAT: the input map, H x W */
    unsigned H, W, Ho, Wo;
    uint8_t *out;                     /* the output map (unused when the head follows) */
    const uint8_t *up;                /* LAT: the coarser level (Ho / 2 x Wo / 2 x F), or NULL */
    const char *name;
} q_step;

typedef struct {
    const vocr_tsd_model *m;
    const unsigned char *px;          /* the stem's grey source, w x h, row stride `stride` */
    unsigned w, h, stride;
    float *maps;
    uint8_t *rings;                   /* the tasks' rings of expanded rows (Q_RING_BYTES each; VNNI) */
    q_step st[Q_MAX_STEPS];
    unsigned n, cur;
} q_net;

typedef struct {
    unsigned H[4], W[4];
    size_t X[4][2], P[4], Q, maps, rings, end;   /* bytes from the 64-byte aligned arena start */
} q_plan;

static void q_plan_of(const vocr_tsd_model *m, unsigned pw, unsigned ph, unsigned ntasks, q_plan *P)
{
    const unsigned h = m->hs == 4 ? 0u : 1u;
    size_t off = 0;
    unsigned s;
    memset(P, 0, sizeof *P);
    for (s = 0; s < 4; s++) {
        P->H[s] = ph >> (2 + s);
        P->W[s] = pw >> (2 + s);
    }
    for (s = 0; s < 4; s++) {
        const size_t sz = ALIGN_UP((size_t)P->H[s] * P->W[s] * m->c[s], 64);
        P->X[s][0] = off;
        off += sz;
        P->X[s][1] = off;
        off += m->n[s] ? sz : 0;    /* the residual blocks' second map */
    }
    for (s = h; s < 4; s++) {
        P->P[s] = off;
        off += ALIGN_UP((size_t)P->H[s] * P->W[s] * m->F, 64);
    }
    P->Q = off;
    off += m->smooth >= 2 ? ALIGN_UP((size_t)P->H[h] * P->W[h] * m->F, 64) : 0;
    P->maps = off;
    off += ALIGN_UP((size_t)P->H[h] * P->W[h] * m->out * sizeof(float), 64);
    P->rings = off;
#if Q_VNNI
    off += (size_t)Q_RING_BYTES * (ntasks < 1 ? 1u : ntasks);
#elif Q_AVX2
    off += (size_t)Q_FRING_BYTES * (ntasks < 1 ? 1u : ntasks);
#else
    (void)ntasks;
#endif
    P->end = off;
}

size_t vocr_tsdq_maps_bytes(const vocr_tsd_model *m, unsigned pw, unsigned ph, unsigned ntasks)
{
    q_plan P;
    if (!m || !pw || !ph) return 0;
    q_plan_of(m, pw, ph, ntasks, &P);
    return P.end + 64;
}

/* the stem's 16 bytes of each output column of row oy, [x0, x1): patch[ox - x0][4 ky + kx] = the grey byte at
 * (4 ox + kx, 4 oy + ky), the edge padding (columns past the source repeat its last, rows its last row) */
static void q_patches(const q_net *N, unsigned oy, unsigned x0, unsigned x1, uint8_t *patch)
{
    unsigned ky, ox, kx, xs = x0;    /* xs: the first cell the plain loop below takes */
#if Q_AVX2
    {   /* four cells a step inside the source: a 4 x 4 transpose of the rows' dwords (cell ox: r0..r3 at 4 ox) */
        const uint8_t *r[4];
        unsigned xin = N->w / 4 > x0 ? N->w / 4 : x0;     /* the cells whose 4 columns are all inside the source */
        for (ky = 0; ky < 4; ky++) r[ky] = N->px + (size_t)(4 * oy + ky < N->h ? 4 * oy + ky : N->h - 1) * N->stride;
        xin = xin < x1 ? xin : x1;
        for (ox = x0; ox + 4 <= xin; ox += 4) {
            const __m128i a = _mm_loadu_si128((const __m128i *)(const void *)(r[0] + 4 * ox));
            const __m128i b = _mm_loadu_si128((const __m128i *)(const void *)(r[1] + 4 * ox));
            const __m128i c = _mm_loadu_si128((const __m128i *)(const void *)(r[2] + 4 * ox));
            const __m128i d = _mm_loadu_si128((const __m128i *)(const void *)(r[3] + 4 * ox));
            const __m128i t0 = _mm_unpacklo_epi32(a, b), t1 = _mm_unpacklo_epi32(c, d);
            const __m128i t2 = _mm_unpackhi_epi32(a, b), t3 = _mm_unpackhi_epi32(c, d);
            uint8_t *o = patch + (size_t)(ox - x0) * 16;
            _mm_storeu_si128((__m128i *)(void *)o, _mm_unpacklo_epi64(t0, t1));
            _mm_storeu_si128((__m128i *)(void *)(o + 16), _mm_unpackhi_epi64(t0, t1));
            _mm_storeu_si128((__m128i *)(void *)(o + 32), _mm_unpacklo_epi64(t2, t3));
            _mm_storeu_si128((__m128i *)(void *)(o + 48), _mm_unpackhi_epi64(t2, t3));
        }
        xs = ox;          /* the rest below, cell by cell */
    }
#endif
    for (ky = 0; ky < 4; ky++) {
        const unsigned sy = 4 * oy + ky < N->h ? 4 * oy + ky : N->h - 1;
        const uint8_t *r = N->px + (size_t)sy * N->stride;
        for (ox = xs; ox < x1; ox++) {
            uint8_t *o = patch + (size_t)(ox - x0) * 16 + 4 * ky;
            if (4 * ox + 3 < N->w) {
                memcpy(o, r + 4 * ox, 4);
            } else {
                for (kx = 0; kx < 4; kx++) o[kx] = r[4 * ox + kx < N->w ? 4 * ox + kx : N->w - 1];
            }
        }
    }
}

static void q_head(const q_net *N, const uint8_t *U, unsigned n, float *maps)
{
    q_epi e;
    QPROF_T(t0);
    memset(&e, 0, sizeof e);
    e.kind = Q_FLOAT;
    e.f = maps;
    e.fstride = N->m->out;
    q_pw(U, n, &N->m->qhead, &e, NULL);
    QPROF_ADD(qprof_head, t0);
}

static void q_rows(const q_net *N, const q_step *t, unsigned y0, unsigned y1, unsigned task)
{
    const vocr_tsd_model *m = N->m;
    uint8_t T[Q_CHUNK * VOCR_TSD_MAX_CHANNELS], U[Q_CHUNK * VOCR_TSD_MAX_CHANNELS];
    q_epi e;
    unsigned y, x0;
    memset(&e, 0, sizeof e);
    (void)task;
    if (t->kind == QS_STEM) {
        const unsigned C = m->c[0];
        e.kind = Q_RELU;
        for (y = y0; y < y1; y++)
            for (x0 = 0; x0 < t->Wo; x0 += Q_CHUNK) {
                const unsigned x1 = x0 + Q_CHUNK < t->Wo ? x0 + Q_CHUNK : t->Wo;
                q_patches(N, y, x0, x1, T);
                q_pw(T, x1 - x0, t->a, &e, t->out + ((size_t)y * t->Wo + x0) * C);
            }
        return;
    }
    if (t->kind == QS_SEP) {
        const unsigned Ci = t->a->cin, Co = t->b->cout, s = t->a->stride, d = t->a->dil;
#if Q_VNNI
        /* the rows in order, each in strips: output row y's three input rows interleaved per channel over the strip's
         * input columns (dw_vexpand: one vertically expanded row, zero outside the map), then the depthwise from it
         * and the pointwise */
        dw_prep D;
        uint8_t *V = N->rings + (size_t)Q_RING_BYTES * task;
        dw_prepare(t->a, &D);
        for (y = y0; y < y1; y++) {
            const uint8_t *rows3[3], *rows[3];
            unsigned kys[3], nk = 0, ky;
            for (ky = 0; ky < 3; ky++) {
                const int iy = (int)(y * s + ky * d) - (int)d;
                rows3[ky] = iy >= 0 && iy < (int)t->H ? t->in + (size_t)iy * t->W * Ci : NULL;
                if (rows3[ky]) {
                    rows[nk] = rows3[ky];
                    kys[nk++] = ky;
                }
            }
            for (x0 = 0; x0 < t->Wo; x0 += Q_CHUNK) {
                const unsigned x1 = x0 + Q_CHUNK < t->Wo ? x0 + Q_CHUNK : t->Wo, n = x1 - x0;
                uint8_t *o = t->head ? U : t->out + ((size_t)y * t->Wo + x0) * Co;
                QPROF_T(t_dw);
                if (D.simd_c) {
                    QPROF_T(t_ex);
                    dw_vexpand(rows3, t->W, t->a, &D, (int)(x0 * s) - (int)d, (n - 1) * s + 2 * d + 1, V);
                    QPROF_ADD(qprof_expand, t_ex);
                    dw_vnni(V, t->a, &D, n, T);
                }
                if (D.simd_c < Ci) q_dw_c(rows, kys, nk, t->W, t->a, x0, x1, D.simd_c, Ci, T);
                QPROF_ADD(qprof_dw[N->cur], t_dw);
                e.kind = t->res ? Q_RES : Q_RELU;
                e.res = t->res ? t->in + ((size_t)y * t->W + x0) * Ci : NULL;
                q_pw(T, n, t->b, &e, o);
                if (t->head) q_head(N, U, n, N->maps + ((size_t)y * t->Wo + x0) * m->out);
            }
        }
        return;
#else
        /* (SIMD builds) the strip's input rows, expanded (VNNI) or as floats (AVX2), in a ring: input row r in slot
         * r % R when the rows are shared (stride 1 and dilation <= 2: 2 d + 1 slots; stride 2 and dilation 1: 3), else
         * slot ky (computed for every output row) */
        const int shared = (s == 1 && d <= 2) || (s == 2 && d == 1);
        const unsigned R = s == 1 && d <= 2 ? 2 * d + 1 : 3;
#if Q_AVX2
        float wf[9 * VOCR_TSD_MAX_CHANNELS];
        float *fring = (float *)(void *)(N->rings + (size_t)Q_FRING_BYTES * task);
        unsigned i;
        for (i = 0; i < 9 * Ci; i++) wf[i] = (float)t->a->w[i];
#else
        (void)shared;
        (void)R;
#endif
        for (x0 = 0; x0 < t->Wo; x0 += Q_CHUNK) {
            const unsigned x1 = x0 + Q_CHUNK < t->Wo ? x0 + Q_CHUNK : t->Wo, n = x1 - x0;
#if Q_AVX2
            int slot_row[Q_RING_ROWS];
            unsigned k;
#endif
#if Q_AVX2
            /* the input columns the strip reads */
            const unsigned ia0 = x0 * s >= d ? x0 * s - d : 0;
            const unsigned ia1 = (x1 - 1) * s + d + 1 < t->W ? (x1 - 1) * s + d + 1 : t->W;
            const float *F[3];
#endif
#if Q_AVX2
            for (k = 0; k < Q_RING_ROWS; k++) slot_row[k] = -1;
#endif
            for (y = y0; y < y1; y++) {
                const uint8_t *rows[3];
                unsigned kys[3], nk = 0, ky;
                uint8_t *o = t->head ? U : t->out + ((size_t)y * t->Wo + x0) * Co;
                QPROF_T(t_dw);
                for (ky = 0; ky < 3; ky++) {
                    const int iy = (int)(y * s + ky * d) - (int)d;
                    if (iy < 0 || iy >= (int)t->H) continue;
                    rows[nk] = t->in + (size_t)iy * t->W * Ci;
#if Q_AVX2
                    {
                        const unsigned sl = shared ? (unsigned)iy % R : ky;
                        float *Fr = fring + (size_t)sl * DWF_ROW_FLOATS;
                        if (!shared || slot_row[sl] != iy) {
                            QPROF_T(t_ex);
                            dwf_row(rows[nk], Ci, ia0, ia1, Fr);
                            QPROF_ADD(qprof_expand, t_ex);
                            slot_row[sl] = shared ? iy : -1;
                        }
                        F[nk] = Fr;
                    }
#endif
                    kys[nk++] = ky;
                }
#if Q_AVX2
                dwf_avx2(F, kys, nk, t->W, ia0, t->a, wf, x0, x1, T);
#else
                q_dw_c(rows, kys, nk, t->W, t->a, x0, x1, 0, Ci, T);
#endif
                QPROF_ADD(qprof_dw[N->cur], t_dw);
                e.kind = t->res ? Q_RES : Q_RELU;
                e.res = t->res ? t->in + ((size_t)y * t->W + x0) * Ci : NULL;
                q_pw(T, n, t->b, &e, o);
                if (t->head) q_head(N, U, n, N->maps + ((size_t)y * t->Wo + x0) * m->out);
            }
        }
        return;
#endif
    }
    if (t->kind == QS_LAT) {
        const unsigned Ci = t->a->cin, F = m->F;
        for (y = y0; y < y1; y++)
            for (x0 = 0; x0 < t->Wo; x0 += Q_CHUNK) {
                const unsigned x1 = x0 + Q_CHUNK < t->Wo ? x0 + Q_CHUNK : t->Wo;
                uint8_t *o = t->head ? U : t->out + ((size_t)y * t->Wo + x0) * F;
                e.kind = t->up ? Q_UP : Q_RELU;
                e.up = t->up ? t->up + (size_t)(y >> 1) * (t->Wo >> 1) * F : NULL;
                e.x0 = x0;
                q_pw(t->in + ((size_t)y * t->W + x0) * Ci, x1 - x0, t->a, &e, o);
                if (t->head) q_head(N, U, x1 - x0, N->maps + ((size_t)y * t->Wo + x0) * m->out);
            }
        return;
    }
}

/* gc in place: x = q(f32(x) A + G[c]) with A = f32(sx / so) and G[c] = f32(max(b + mean W, 0) / so), mean = sum(x)
 * sx / P over every position (the sums exact, the rest float64 in a fixed order) */
static void q_gc(const vocr_tsd_model *m, uint8_t *x, size_t P)
{
    const unsigned C = m->c[3];
    const double sx = (double)m->gc_sx, so = (double)m->gc_so;
    const float A = (float)(sx / so);
    uint32_t sum[VOCR_TSD_MAX_CHANNELS];
    double mean[VOCR_TSD_MAX_CHANNELS];
    float G[VOCR_TSD_MAX_CHANNELS];
    size_t p;
    unsigned c, ci;
    for (c = 0; c < C; c++) sum[c] = 0;
    for (p = 0; p < P; p++)
        for (c = 0; c < C; c++) sum[c] += x[p * C + c];
    for (c = 0; c < C; c++) mean[c] = (double)sum[c] * sx / (double)P;
    for (c = 0; c < C; c++) {
        double a = (double)m->gcl.b[c];
        for (ci = 0; ci < C; ci++) a = a + mean[ci] * (double)m->gcl.w[(size_t)ci * C + c];
        G[c] = (float)((a > 0.0 ? a : 0.0) / so);
    }
    for (p = 0; p < P; p++) {
        uint8_t *r = x + p * C;
        c = 0;
#if Q_AVX2
        for (; c + 8 <= C; c += 8) {
            const __m256 v = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(const void *)(r + c))));
            const __m256 y = _mm256_add_ps(_mm256_mul_ps(v, _mm256_set1_ps(A)), _mm256_loadu_ps(G + c));
            const __m256i q = _mm256_max_epi32(_mm256_cvtps_epi32(_mm256_min_ps(y, _mm256_set1_ps(255.0f))),
                                               _mm256_setzero_si256());
            const __m128i q16 = _mm_packs_epi32(_mm256_castsi256_si128(q), _mm256_extracti128_si256(q, 1));
            _mm_storel_epi64((__m128i *)(void *)(r + c), _mm_packus_epi16(q16, q16));
        }
#endif
        for (; c < C; c++) {
            const float t = (float)r[c] * A;
            r[c] = q_round(t + G[c]);
        }
    }
}

static void q_task(void *arg, unsigned task, unsigned ntasks)
{
    const q_net *N = (const q_net *)arg;
    const q_step *t = &N->st[N->cur];
    const unsigned rows = t->Ho;
    q_rows(N, t, (unsigned)((size_t)rows * task / ntasks), (unsigned)((size_t)rows * (task + 1) / ntasks), task);
}

static q_step *q_add(q_net *N, int kind, const char *name)
{
    q_step *t = &N->st[N->n++];
    memset(t, 0, sizeof *t);
    t->kind = kind;
    t->name = name;
    return t;
}

static const char *const Q_STAGE_NAMES[4][1 + VOCR_TSD_MAX_BLOCKS] = {
    {"s4.-", "s4.b0", "s4.b1", "s4.b2", "s4.b3", "s4.b4", "s4.b5", "s4.b6", "s4.b7"},
    {"s8.down", "s8.b0", "s8.b1", "s8.b2", "s8.b3", "s8.b4", "s8.b5", "s8.b6", "s8.b7"},
    {"s16.down", "s16.b0", "s16.b1", "s16.b2", "s16.b3", "s16.b4", "s16.b5", "s16.b6", "s16.b7"},
    {"s32.down", "s32.b0", "s32.b1", "s32.b2", "s32.b3", "s32.b4", "s32.b5", "s32.b6", "s32.b7"}};
static const char *const Q_LAT_NAMES[4] = {"lat4", "lat8", "lat16", "lat32"};
static const char *const Q_SMOOTH_NAMES[VOCR_TSD_MAX_SMOOTH] = {"smooth0", "smooth1", "smooth2", "smooth3"};

static void q_build(q_net *N, const vocr_tsd_model *m, unsigned char *A, const q_plan *P)
{
    const unsigned h = m->hs == 4 ? 0u : 1u;
    unsigned s, i;
    const uint8_t *coarse = NULL, *last;
    uint8_t *fin[4];
    q_step *t;
    N->m = m;
    N->n = 0;
    N->maps = (float *)(void *)(A + P->maps);
    N->rings = A + P->rings;
    t = q_add(N, QS_STEM, "stem");
    t->a = &m->qstem;
    t->Ho = P->H[0];
    t->Wo = P->W[0];
    t->out = A + P->X[0][0];
    for (s = 0; s < 4; s++) {
        unsigned cur = 0;
        if (s > 0) {
            t = q_add(N, QS_SEP, Q_STAGE_NAMES[s][0]);
            t->a = &m->qdw[s][0];
            t->b = &m->qpw[s][0];
            t->in = fin[s - 1];
            t->H = P->H[s - 1];
            t->W = P->W[s - 1];
            t->Ho = P->H[s];
            t->Wo = P->W[s];
            t->out = A + P->X[s][0];
        }
        for (i = 0; i < m->n[s]; i++) {
            t = q_add(N, QS_SEP, Q_STAGE_NAMES[s][1 + i]);
            t->a = &m->qdw[s][1 + i];
            t->b = &m->qpw[s][1 + i];
            t->in = A + P->X[s][cur];
            t->H = t->Ho = P->H[s];
            t->W = t->Wo = P->W[s];
            t->res = 1;
            cur ^= 1u;
            t->out = A + P->X[s][cur];
        }
        fin[s] = A + P->X[s][cur];
    }
    if (m->gc) {
        t = q_add(N, QS_GC, "gc");
        t->out = fin[3];
        t->H = P->H[3];
        t->W = P->W[3];
        t->Ho = 1;
    }
    for (s = 3; s + 1 > h; s--) {
        t = q_add(N, QS_LAT, Q_LAT_NAMES[s]);
        t->a = &m->qlat[s];
        t->in = fin[s];
        t->H = t->Ho = P->H[s];
        t->W = t->Wo = P->W[s];
        t->up = coarse;
        if (s == h && !m->smooth) {
            t->head = 1;
        } else {
            t->out = A + P->P[s];
            coarse = t->out;
        }
        if (s == 0) break;
    }
    last = coarse;
    for (i = 0; i < m->smooth; i++) {
        t = q_add(N, QS_SEP, Q_SMOOTH_NAMES[i]);
        t->a = &m->qsdw[i];
        t->b = &m->qspw[i];
        t->in = last;
        t->H = t->Ho = P->H[h];
        t->W = t->Wo = P->W[h];
        t->res = 1;
        if (i + 1 == m->smooth) {
            t->head = 1;
        } else {
            t->out = last == A + P->P[h] ? A + P->Q : A + P->P[h];
            last = t->out;
        }
    }
}

int vocr_tsdq_run(const vocr_tsd_model *m, const unsigned char *px, unsigned w, unsigned h, unsigned stride,
                  unsigned pw, unsigned ph, void *arena, size_t bytes, const float **maps, vocr_tsd_runner run,
                  void *ctx, unsigned ntasks)
{
    q_plan P;
    q_net N;
    unsigned char *A;
    if (!m || !px || !arena || !maps || !w || !h || !pw || !ph || pw % VOCR_TSD_ALIGN || ph % VOCR_TSD_ALIGN ||
        w > pw || h > ph)
        return VOCR_TSD_E_ARG;
    if (!run || ntasks < 1) ntasks = 1;
    if (bytes < vocr_tsdq_maps_bytes(m, pw, ph, ntasks)) return VOCR_TSD_E_ARENA;
    q_plan_of(m, pw, ph, ntasks, &P);
    A = (unsigned char *)(((uintptr_t)arena + 63u) & ~(uintptr_t)63u);
    N.px = px;
    N.w = w;
    N.h = h;
    N.stride = stride;
    q_build(&N, m, A, &P);
    QPROF_START();
    for (N.cur = 0; N.cur < N.n; N.cur++) {
        const q_step *t = &N.st[N.cur];
        const unsigned nt = ntasks < t->Ho ? ntasks : t->Ho;
        if (t->kind == QS_GC) {
            q_gc(m, t->out, (size_t)t->H * t->W);
        } else if (nt > 1 && run) {
            run(ctx, q_task, &N, nt);
        } else {
            q_task(&N, 0, 1);
        }
        QPROF_MARK(PROF_STEP0 + N.cur, t->name);
#ifdef VOCR_TSD_PROF
        if (t->kind == QS_SEP && 40u + N.cur < 62u) {
            snprintf(qprof_names[N.cur], sizeof qprof_names[N.cur], "%s.dw", t->name);
            vocr_tsd_prof_add_(40u + N.cur, qprof_names[N.cur], qprof_dw[N.cur]);
            qprof_dw[N.cur] = 0.0;
        }
#endif
    }
#ifdef VOCR_TSD_PROF
    vocr_tsd_prof_add_(62u, "expand", qprof_expand);
    vocr_tsd_prof_add_(63u, "head", qprof_head);
    qprof_expand = qprof_head = 0.0;
#endif
    *maps = N.maps;
    return VOCR_TSD_OK;
}
