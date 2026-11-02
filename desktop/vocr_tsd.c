/* vocr_tsd.c - textspot-desktop's line detector in C (vocr_tsd.h). C99 + libm + desktop/vocr_desk.c (its resize,
 * luma and input policy); AVX2 + FMA kernels when the compiler targets them, plain C otherwise. */
#ifdef VOCR_TSD_PROF
#define _POSIX_C_SOURCE 200112L
#endif
#include "vocr_tsd.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "vocr_desk.h"

#if defined(__AVX2__) && defined(__FMA__) && !defined(VOCR_TSD_SCALAR)
#include <immintrin.h>
#define TSD_AVX2 1
#else
#define TSD_AVX2 0
#endif

#define ALIGN_UP(x, a) (((x) + (size_t)(a) - 1u) / (size_t)(a) * (size_t)(a))
#define CHUNK_FLOATS 6144u   /* a row chunk's scratch (floats): at most 48 px x 128 channels or 24 px x 256 */
#define CHUNK_MAX 48u
#define MAX_STEPS 64u
#define RESIZE_MAX_TASKS 64u
#define STEM_PF_ROWS 2u   /* the stem prefetches its source this many output rows ahead */

/* ----------------------------------------------------------------------------- profiling (-DVOCR_TSD_PROF) */

#ifdef VOCR_TSD_PROF
#include <time.h>
/* per-step host timing of profiling builds only: not thread safe (call from one thread), not in a release build */
double vocr_tsd_prof_us[MAX_STEPS];
unsigned vocr_tsd_prof_runs;
static const char *prof_names[MAX_STEPS];
static double prof_t0;
static double prof_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1e6 * (double)ts.tv_sec + 1e-3 * (double)ts.tv_nsec;
}
const char *vocr_tsd_prof_name(unsigned i);
const char *vocr_tsd_prof_name(unsigned i)
{
    return i < MAX_STEPS ? prof_names[i] : NULL;
}
void vocr_tsd_prof_reset(void);
void vocr_tsd_prof_reset(void)
{
    memset(vocr_tsd_prof_us, 0, sizeof vocr_tsd_prof_us);
    vocr_tsd_prof_runs = 0;
}
#define PROF_START() (prof_t0 = prof_now())
#define PROF_MARK(i, name)                     \
    do {                                       \
        double t_ = prof_now();                \
        if ((i) < MAX_STEPS) {                 \
            vocr_tsd_prof_us[i] += t_ - prof_t0; \
            prof_names[i] = (name);            \
        }                                      \
        prof_t0 = t_;                          \
    } while (0)
#else
#define PROF_START() ((void)0)
#define PROF_MARK(i, name) ((void)0)
#endif

/* profile slots: 0 luma, 1 resize, 2 decode, 3 boxes, 8 + k the network's step k */
#define PROF_STEP0 8u

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
    uint32_t tab[256], c = 0xFFFFFFFFu;
    unsigned i, k;
    size_t j;
    for (i = 0; i < 256; i++) {
        uint32_t v = i;
        for (k = 0; k < 8; k++) v = (v >> 1) ^ (0xEDB88320u & (0u - (v & 1u)));
        tab[i] = v;
    }
    for (j = 0; j < n; j++) c = tab[(c ^ p[j]) & 0xFFu] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

static int little_endian(void)
{
    const uint32_t one = 1;
    unsigned char b;
    memcpy(&b, &one, 1);
    return b == 1;
}

static const float *take(const unsigned char *base, size_t bytes, size_t *off, size_t n)
{
    const float *p;
    if (*off + 4 * n > bytes) return NULL;
    p = (const float *)(const void *)(base + *off);
    *off += 4 * n;
    return p;
}

static int set_layer(vocr_tsd_layer *L, const unsigned char *base, size_t bytes, size_t *off, size_t nw, size_t nb,
                     unsigned cin, unsigned cout, unsigned stride, unsigned dil, size_t *params)
{
    L->w = take(base, bytes, off, nw);
    L->b = take(base, bytes, off, nb);
    L->cin = cin;
    L->cout = cout;
    L->stride = stride;
    L->dil = dil;
    *params += nw + nb;
    return L->w && L->b ? 0 : VOCR_TSD_E_BLOB;
}

static int chan_ok(unsigned c)
{
    return c >= 8 && c <= VOCR_TSD_MAX_CHANNELS && c % 8 == 0;
}

int vocr_tsd_load(vocr_tsd_model *m, const void *blob, size_t bytes)
{
    const unsigned char *b = (const unsigned char *)blob;
    size_t off = VOCR_TSD_HEADER;
    unsigned s, i;
    if (!m || !b) return VOCR_TSD_E_ARG;
    memset(m, 0, sizeof *m);
    if (!little_endian() || ((uintptr_t)b & 3u) || bytes < VOCR_TSD_HEADER + 4 || memcmp(b, "VTSD", 4) ||
        rd32(b + 4) != VOCR_TSD_VERSION || rd32(b + 8) != VOCR_TSD_HEADER || rd32(b + 12) != bytes)
        return VOCR_TSD_E_BLOB;
    if (crc32_of(b, bytes - 4) != rd32(b + bytes - 4)) return VOCR_TSD_E_CRC;
    m->patch = rd32(b + 16);
    m->hs = rd32(b + 20);
    for (i = 0; i < 4; i++) {
        m->c[i] = rd32(b + 24 + 4 * i);
        m->n[i] = rd32(b + 40 + 4 * i);
    }
    m->F = rd32(b + 56);
    m->smooth = rd32(b + 60);
    m->gc = rd32(b + 64);
    m->tags = rd32(b + 68);
    m->out = rd32(b + 72);
    m->outp = rd32(b + 76);
    m->min_cells = rd32(b + 80);
    m->max_boxes = rd32(b + 84);
    m->size_kind = rd32(b + 88);
    m->size_n = rd32(b + 92);
    m->conf_min_logit = rdf64(b + 128);
    m->min_across = rdf64(b + 136);
    m->along_reach = rdf64(b + 144);
    if (m->patch != 4 || (m->hs != 4 && m->hs != 8) || !chan_ok(m->F) || m->smooth > VOCR_TSD_MAX_SMOOTH ||
        m->gc > 1 || m->tags > VOCR_TSD_MAX_TAGS || m->out != 6 + m->tags || m->outp != (m->out + 7) / 8 * 8 ||
        m->min_cells < 1 || m->max_boxes < 1 || m->max_boxes > VOCR_TSD_MAX_BOXES ||
        m->size_kind > VOCR_TSD_SIZE_LONG || rd32(b + 152) != VOCR_TSD_ALIGN || m->conf_min_logit != m->conf_min_logit ||
        m->min_across != m->min_across || m->along_reach != m->along_reach)
        return VOCR_TSD_E_BLOB;
    for (s = 0; s < 4; s++) {
        if (!chan_ok(m->c[s]) || m->n[s] > VOCR_TSD_MAX_BLOCKS) return VOCR_TSD_E_BLOB;
        for (i = 0; i < VOCR_TSD_MAX_BLOCKS; i++) {
            unsigned d = b[96 + 8 * s + i];
            if (i < m->n[s] ? (d < 1 || d > 16) : d != 0) return VOCR_TSD_E_BLOB;
            m->dil[s][i] = d;
        }
    }
    if (set_layer(&m->stem, b, bytes, &off, 16u * m->c[0], m->c[0], 16, m->c[0], 4, 1, &m->params))
        return VOCR_TSD_E_BLOB;
    for (s = 0; s < 4; s++) {
        const unsigned C = m->c[s];
        if (s > 0) {
            const unsigned Ci = m->c[s - 1];
            if (set_layer(&m->dw[s][0], b, bytes, &off, 9u * Ci, Ci, Ci, Ci, 2, 1, &m->params) ||
                set_layer(&m->pw[s][0], b, bytes, &off, (size_t)Ci * C, C, Ci, C, 1, 1, &m->params))
                return VOCR_TSD_E_BLOB;
        }
        for (i = 0; i < m->n[s]; i++)
            if (set_layer(&m->dw[s][1 + i], b, bytes, &off, 9u * C, C, C, C, 1, m->dil[s][i], &m->params) ||
                set_layer(&m->pw[s][1 + i], b, bytes, &off, (size_t)C * C, C, C, C, 1, 1, &m->params))
                return VOCR_TSD_E_BLOB;
    }
    if (m->gc && set_layer(&m->gcl, b, bytes, &off, (size_t)m->c[3] * m->c[3], m->c[3], m->c[3], m->c[3], 1, 1,
                           &m->params))
        return VOCR_TSD_E_BLOB;
    for (s = m->hs == 4 ? 0u : 1u; s < 4; s++)
        if (set_layer(&m->lat[s], b, bytes, &off, (size_t)m->c[s] * m->F, m->F, m->c[s], m->F, 1, 1, &m->params))
            return VOCR_TSD_E_BLOB;
    for (i = 0; i < m->smooth; i++)
        if (set_layer(&m->sdw[i], b, bytes, &off, 9u * m->F, m->F, m->F, m->F, 1, 1, &m->params) ||
            set_layer(&m->spw[i], b, bytes, &off, (size_t)m->F * m->F, m->F, m->F, m->F, 1, 1, &m->params))
            return VOCR_TSD_E_BLOB;
    if (set_layer(&m->head, b, bytes, &off, (size_t)m->F * m->outp, m->outp, m->F, m->outp, 1, 1, &m->params))
        return VOCR_TSD_E_BLOB;
    if (off + 4 != bytes) return VOCR_TSD_E_BLOB;
    return VOCR_TSD_OK;
}

void vocr_tsd_decode_opts_of(const vocr_tsd_model *m, vocr_tsd_decode_opts *o)
{
    o->stride = m->hs;
    o->tags = m->tags;
    o->min_cells = m->min_cells;
    o->max_boxes = m->max_boxes;
    o->conf_min_logit = m->conf_min_logit;
    o->min_across = m->min_across;
    o->along_reach = m->along_reach;
}

void vocr_tsd_input_size(unsigned kind, unsigned n, unsigned w, unsigned h, unsigned *iw, unsigned *ih,
                         unsigned *pw, unsigned *ph)
{
    vocr_desk_input_size(kind, n, w, h, iw, ih, pw, ph);
}

size_t vocr_tsd_model_bytes(void)
{
    return sizeof(vocr_tsd_model);
}

size_t vocr_tsd_set_bytes(void)
{
    return sizeof(vocr_tsd_set);
}

size_t vocr_tsd_box_bytes(void)
{
    return sizeof(vocr_tsd_box);
}

int vocr_tsd_simd(void)
{
    return TSD_AVX2;
}

/* ----------------------------------------------------------------------------- kernels */

/* the pointwise epilogues: relu(acc); relu(acc + res[p]); relu(acc) + up[(x0 + p) >> 1] (the pyramid's nearest 2x
 * up-sampling and addition); acc (the head) */
enum { EPI_RELU = 0, EPI_RES = 1, EPI_UP = 2, EPI_NONE = 3 };

typedef struct {
    int kind;
    const float *res;   /* EPI_RES: [P][Cout] */
    const float *up;    /* EPI_UP: the coarse row, [.][Cout] */
    unsigned x0;        /* EPI_UP: the chunk's first output column */
} epi_t;

#if TSD_AVX2
static inline void epi_store(__m256 u, const epi_t *e, unsigned p, unsigned co, unsigned Cout, float *out)
{
    const __m256 z = _mm256_setzero_ps();
    if (e->kind == EPI_RES)
        u = _mm256_max_ps(_mm256_add_ps(u, _mm256_loadu_ps(e->res + (size_t)p * Cout + co)), z);
    else if (e->kind == EPI_UP)
        u = _mm256_add_ps(_mm256_max_ps(u, z), _mm256_loadu_ps(e->up + (size_t)((e->x0 + p) >> 1) * Cout + co));
    else if (e->kind == EPI_RELU)
        u = _mm256_max_ps(u, z);
    _mm256_storeu_ps(out + (size_t)p * Cout + co, u);
}

#define BC(r) _mm256_broadcast_ss(x + (size_t)(r) * Cin + ci)
#define F2(r, a, b)                    \
    do {                               \
        const __m256 v_ = BC(r);       \
        a = _mm256_fmadd_ps(v_, w0, a); \
        b = _mm256_fmadd_ps(v_, w1, b); \
    } while (0)

/* 6 pixels x 16 channels: out[p + r][co .. co + 16) for r < 6; x: pixel p's input (stride Cin) */
static void k16x6(const float *x, unsigned Cin, const float *w, unsigned Cout, const float *b, unsigned co,
                  const epi_t *e, unsigned p, float *out)
{
    const __m256 b0 = _mm256_loadu_ps(b + co), b1 = _mm256_loadu_ps(b + co + 8);
    __m256 a00 = b0, a01 = b1, a10 = b0, a11 = b1, a20 = b0, a21 = b1;
    __m256 a30 = b0, a31 = b1, a40 = b0, a41 = b1, a50 = b0, a51 = b1;
    const float *wc = w + co;
    unsigned ci;
    for (ci = 0; ci < Cin; ci++, wc += Cout) {
        const __m256 w0 = _mm256_loadu_ps(wc), w1 = _mm256_loadu_ps(wc + 8);
        F2(0, a00, a01);
        F2(1, a10, a11);
        F2(2, a20, a21);
        F2(3, a30, a31);
        F2(4, a40, a41);
        F2(5, a50, a51);
    }
    epi_store(a00, e, p, co, Cout, out);
    epi_store(a01, e, p, co + 8, Cout, out);
    epi_store(a10, e, p + 1, co, Cout, out);
    epi_store(a11, e, p + 1, co + 8, Cout, out);
    epi_store(a20, e, p + 2, co, Cout, out);
    epi_store(a21, e, p + 2, co + 8, Cout, out);
    epi_store(a30, e, p + 3, co, Cout, out);
    epi_store(a31, e, p + 3, co + 8, Cout, out);
    epi_store(a40, e, p + 4, co, Cout, out);
    epi_store(a41, e, p + 4, co + 8, Cout, out);
    epi_store(a50, e, p + 5, co, Cout, out);
    epi_store(a51, e, p + 5, co + 8, Cout, out);
}

#define F3(r, a, b, c)                 \
    do {                               \
        const __m256 v_ = BC(r);       \
        a = _mm256_fmadd_ps(v_, w0, a); \
        b = _mm256_fmadd_ps(v_, w1, b); \
        c = _mm256_fmadd_ps(v_, w2, c); \
    } while (0)

/* 4 pixels x 24 channels */
static void k24x4(const float *x, unsigned Cin, const float *w, unsigned Cout, const float *b, unsigned co,
                  const epi_t *e, unsigned p, float *out)
{
    const __m256 b0 = _mm256_loadu_ps(b + co), b1 = _mm256_loadu_ps(b + co + 8), b2 = _mm256_loadu_ps(b + co + 16);
    __m256 a00 = b0, a01 = b1, a02 = b2, a10 = b0, a11 = b1, a12 = b2;
    __m256 a20 = b0, a21 = b1, a22 = b2, a30 = b0, a31 = b1, a32 = b2;
    const float *wc = w + co;
    unsigned ci;
    for (ci = 0; ci < Cin; ci++, wc += Cout) {
        const __m256 w0 = _mm256_loadu_ps(wc), w1 = _mm256_loadu_ps(wc + 8), w2 = _mm256_loadu_ps(wc + 16);
        F3(0, a00, a01, a02);
        F3(1, a10, a11, a12);
        F3(2, a20, a21, a22);
        F3(3, a30, a31, a32);
    }
    epi_store(a00, e, p, co, Cout, out);
    epi_store(a01, e, p, co + 8, Cout, out);
    epi_store(a02, e, p, co + 16, Cout, out);
    epi_store(a10, e, p + 1, co, Cout, out);
    epi_store(a11, e, p + 1, co + 8, Cout, out);
    epi_store(a12, e, p + 1, co + 16, Cout, out);
    epi_store(a20, e, p + 2, co, Cout, out);
    epi_store(a21, e, p + 2, co + 8, Cout, out);
    epi_store(a22, e, p + 2, co + 16, Cout, out);
    epi_store(a30, e, p + 3, co, Cout, out);
    epi_store(a31, e, p + 3, co + 8, Cout, out);
    epi_store(a32, e, p + 3, co + 16, Cout, out);
}

#define F1(r, a) a = _mm256_fmadd_ps(BC(r), w0, a)

/* 12 pixels x 8 channels */
static void k8x12(const float *x, unsigned Cin, const float *w, unsigned Cout, const float *b, unsigned co,
                  const epi_t *e, unsigned p, float *out)
{
    const __m256 b0 = _mm256_loadu_ps(b + co);
    __m256 a0 = b0, a1 = b0, a2 = b0, a3 = b0, a4 = b0, a5 = b0, a6 = b0, a7 = b0, a8 = b0, a9 = b0, aa = b0, ab = b0;
    const float *wc = w + co;
    unsigned ci;
    for (ci = 0; ci < Cin; ci++, wc += Cout) {
        const __m256 w0 = _mm256_loadu_ps(wc);
        F1(0, a0);
        F1(1, a1);
        F1(2, a2);
        F1(3, a3);
        F1(4, a4);
        F1(5, a5);
        F1(6, a6);
        F1(7, a7);
        F1(8, a8);
        F1(9, a9);
        F1(10, aa);
        F1(11, ab);
    }
    epi_store(a0, e, p, co, Cout, out);
    epi_store(a1, e, p + 1, co, Cout, out);
    epi_store(a2, e, p + 2, co, Cout, out);
    epi_store(a3, e, p + 3, co, Cout, out);
    epi_store(a4, e, p + 4, co, Cout, out);
    epi_store(a5, e, p + 5, co, Cout, out);
    epi_store(a6, e, p + 6, co, Cout, out);
    epi_store(a7, e, p + 7, co, Cout, out);
    epi_store(a8, e, p + 8, co, Cout, out);
    epi_store(a9, e, p + 9, co, Cout, out);
    epi_store(aa, e, p + 10, co, Cout, out);
    epi_store(ab, e, p + 11, co, Cout, out);
}

/* one pixel x nv vectors of 8 channels (the tails): the same FMA order per output as the blocked kernels */
static void k1(const float *x, unsigned Cin, const float *w, unsigned Cout, const float *b, unsigned co, unsigned nv,
               const epi_t *e, unsigned p, float *out)
{
    unsigned v;
    for (v = 0; v < nv; v++) {
        const float *wc = w + co + 8 * v;
        __m256 a = _mm256_loadu_ps(b + co + 8 * v);
        unsigned ci;
        for (ci = 0; ci < Cin; ci++, wc += Cout) a = _mm256_fmadd_ps(_mm256_broadcast_ss(x + ci), _mm256_loadu_ps(wc), a);
        epi_store(a, e, p, co + 8 * v, Cout, out);
    }
}
#undef BC
#undef F1
#undef F2
#undef F3
#else
static inline float epi_scalar(float u, const epi_t *e, unsigned p, unsigned co, unsigned Cout)
{
    if (e->kind == EPI_RES) {
        u += e->res[(size_t)p * Cout + co];
        return u > 0.0f ? u : 0.0f;
    }
    if (e->kind == EPI_UP) {
        u = u > 0.0f ? u : 0.0f;
        return u + e->up[(size_t)((e->x0 + p) >> 1) * Cout + co];
    }
    if (e->kind == EPI_RELU) return u > 0.0f ? u : 0.0f;
    return u;
}
#endif

/* out[p][co] = epi(b[co] + sum_ci in[p][ci] w[ci][co]) for p < P (in: [P][Cin], out: [P][Cout], contiguous) */
static void pw_chunk(const float *in, unsigned P, const vocr_tsd_layer *L, const epi_t *e, float *out)
{
    const unsigned Cin = L->cin, Cout = L->cout;
    const float *w = L->w, *b = L->b;
#if TSD_AVX2
    unsigned co = 0, p;
    if (Cout % 24 == 0 && Cout % 16 != 0) {
        for (; co < Cout; co += 24) {
            for (p = 0; p + 4 <= P; p += 4) k24x4(in + (size_t)p * Cin, Cin, w, Cout, b, co, e, p, out);
            for (; p < P; p++) k1(in + (size_t)p * Cin, Cin, w, Cout, b, co, 3, e, p, out);
        }
        return;
    }
    for (; co + 16 <= Cout; co += 16) {
        for (p = 0; p + 6 <= P; p += 6) k16x6(in + (size_t)p * Cin, Cin, w, Cout, b, co, e, p, out);
        for (; p < P; p++) k1(in + (size_t)p * Cin, Cin, w, Cout, b, co, 2, e, p, out);
    }
    if (co < Cout) {
        for (p = 0; p + 12 <= P; p += 12) k8x12(in + (size_t)p * Cin, Cin, w, Cout, b, co, e, p, out);
        for (; p < P; p++) k1(in + (size_t)p * Cin, Cin, w, Cout, b, co, 1, e, p, out);
    }
#else
    unsigned p;
    for (p = 0; p < P; p++) {
        float acc[VOCR_TSD_MAX_CHANNELS];
        const float *x = in + (size_t)p * Cin;
        unsigned ci, co;
        for (co = 0; co < Cout; co++) acc[co] = b[co];
        for (ci = 0; ci < Cin; ci++) {
            const float v = x[ci], *wr = w + (size_t)ci * Cout;
            for (co = 0; co < Cout; co++) acc[co] += v * wr[co];
        }
        for (co = 0; co < Cout; co++) out[(size_t)p * Cout + co] = epi_scalar(acc[co], e, p, co, Cout);
    }
#endif
}

/* 3x3 depthwise with zero padding d, stride s, dilation d, ReLU, on an H x W x C map: output row oy, columns
 * [x0, x1), into out ([x1 - x0][C]). The taps run in (ky, kx) order, a tap outside the map skipped; the columns whose
 * three taps are inside the map take the unchecked path, the others the checked one (a column's path depends on its
 * position only, so any split of the rows gives the same bits). */
static void dw_chunk(const float *in, unsigned H, unsigned W, const vocr_tsd_layer *L, unsigned oy, unsigned x0,
                     unsigned x1, float *out)
{
    const unsigned C = L->cin, s = L->stride, d = L->dil;
    const float *wt = L->w, *bs = L->b;
    const float *rows[3];
    unsigned kys[3], nk = 0, ky, lo, hi, ilo, ihi, ox, c;
    for (ky = 0; ky < 3; ky++) {
        int iy = (int)(oy * s + ky * d) - (int)d;
        if (iy >= 0 && iy < (int)H) {
            rows[nk] = in + (size_t)iy * W * C;
            kys[nk++] = ky;
        }
    }
    /* the interior output columns [ilo, ihi): ox s - d >= 0 and ox s + d <= W - 1; then within [x0, x1) */
    ilo = (d + s - 1) / s;
    ihi = W > d ? (W - 1 - d) / s + 1 : 0;
    lo = ilo > x0 ? ilo : x0;
    hi = ihi < x1 ? ihi : x1;
    if (hi < lo) lo = hi = x0;
    for (c = 0; c < C; c += 8) {
#if TSD_AVX2
        const __m256 z = _mm256_setzero_ps(), bias = _mm256_loadu_ps(bs + c);
        unsigned t, side;
        /* the checked columns: [x0, lo), then [hi, x1) */
        for (side = 0; side < 2; side++) {
            const unsigned a0 = side ? hi : x0, a1 = side ? x1 : lo;
            for (ox = a0; ox < a1; ox++) {
                __m256 a = bias;
                for (t = 0; t < nk; t++) {
                    unsigned kx;
                    for (kx = 0; kx < 3; kx++) {
                        int ix = (int)(ox * s + kx * d) - (int)d;
                        if (ix < 0 || ix >= (int)W) continue;
                        a = _mm256_fmadd_ps(_mm256_loadu_ps(rows[t] + (size_t)ix * C + c),
                                            _mm256_loadu_ps(wt + (kys[t] * 3 + kx) * C + c), a);
                    }
                }
                _mm256_storeu_ps(out + (size_t)(ox - x0) * C + c, _mm256_max_ps(a, z));
            }
        }
        if (nk == 3) {
            const __m256 w00 = _mm256_loadu_ps(wt + c), w01 = _mm256_loadu_ps(wt + C + c);
            const __m256 w02 = _mm256_loadu_ps(wt + 2 * C + c), w10 = _mm256_loadu_ps(wt + 3 * C + c);
            const __m256 w11 = _mm256_loadu_ps(wt + 4 * C + c), w12 = _mm256_loadu_ps(wt + 5 * C + c);
            const __m256 w20 = _mm256_loadu_ps(wt + 6 * C + c), w21 = _mm256_loadu_ps(wt + 7 * C + c);
            const __m256 w22 = _mm256_loadu_ps(wt + 8 * C + c);
            const size_t step = (size_t)d * C, xs = (size_t)s * C;
            const float *r0 = rows[0] + (size_t)(lo * s - d) * C + c, *r1 = rows[1] + (size_t)(lo * s - d) * C + c;
            const float *r2 = rows[2] + (size_t)(lo * s - d) * C + c;
            float *o = out + (size_t)(lo - x0) * C + c;
            for (ox = lo; ox < hi; ox++, r0 += xs, r1 += xs, r2 += xs, o += C) {
                __m256 a = bias;
                a = _mm256_fmadd_ps(_mm256_loadu_ps(r0), w00, a);
                a = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + step), w01, a);
                a = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + 2 * step), w02, a);
                a = _mm256_fmadd_ps(_mm256_loadu_ps(r1), w10, a);
                a = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + step), w11, a);
                a = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + 2 * step), w12, a);
                a = _mm256_fmadd_ps(_mm256_loadu_ps(r2), w20, a);
                a = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + step), w21, a);
                a = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + 2 * step), w22, a);
                _mm256_storeu_ps(o, _mm256_max_ps(a, z));
            }
        } else {
            const size_t step = (size_t)d * C;
            for (ox = lo; ox < hi; ox++) {
                __m256 a = bias;
                for (t = 0; t < nk; t++) {
                    const float *r = rows[t] + (size_t)(ox * s - d) * C + c, *wk = wt + kys[t] * 3 * C + c;
                    a = _mm256_fmadd_ps(_mm256_loadu_ps(r), _mm256_loadu_ps(wk), a);
                    a = _mm256_fmadd_ps(_mm256_loadu_ps(r + step), _mm256_loadu_ps(wk + C), a);
                    a = _mm256_fmadd_ps(_mm256_loadu_ps(r + 2 * step), _mm256_loadu_ps(wk + 2 * C), a);
                }
                _mm256_storeu_ps(out + (size_t)(ox - x0) * C + c, _mm256_max_ps(a, z));
            }
        }
#else
        unsigned t, k;
        for (ox = x0; ox < x1; ox++) {
            float a[8];
            const int interior = ox >= lo && ox < hi;
            for (k = 0; k < 8; k++) a[k] = bs[c + k];
            for (t = 0; t < nk; t++) {
                unsigned kx;
                for (kx = 0; kx < 3; kx++) {
                    int ix = (int)(ox * s + kx * d) - (int)d;
                    const float *px, *wk;
                    if (!interior && (ix < 0 || ix >= (int)W)) continue;
                    px = rows[t] + (size_t)ix * C + c;
                    wk = wt + (kys[t] * 3 + kx) * C + c;
                    for (k = 0; k < 8; k++) a[k] += px[k] * wk[k];
                }
            }
            for (k = 0; k < 8; k++) out[(size_t)(ox - x0) * C + c + k] = a[k] > 0.0f ? a[k] : 0.0f;
        }
#endif
    }
}

/* ----------------------------------------------------------------------------- the stem's input */

typedef struct {
    const unsigned char *px;   /* the source: grey (1), RGB (3) or RGBA (4) bytes per pixel */
    unsigned w, h, stride, ch; /* the network reads its pw x ph input with x >= w as x = w - 1, y >= h as h - 1 */
} tsd_src;

/* n source pixels from (x, y) to normalised grey floats (g - 128) / 128, g the integer luma (77 R + 150 G + 29 B) >> 8
 * of an RGB(A) pixel (vocr_desk_grey's) or the grey byte; every pixel inside the source */
static void src_floats(const tsd_src *S, unsigned x, unsigned y, unsigned n, float *o)
{
    const unsigned char *r = S->px + (size_t)y * S->stride + (size_t)x * S->ch;
    unsigned i = 0;
#if TSD_AVX2
    const __m256 k128 = _mm256_set1_ps(128.0f), inv = _mm256_set1_ps(0.0078125f);
    if (S->ch == 1) {
        for (; i + 8 <= n; i += 8) {
            __m256i g = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i *)(const void *)(r + i)));
            _mm256_storeu_ps(o + i, _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(g), k128), inv));
        }
    } else if (S->ch == 4) {
        const __m256i lo8 = _mm256_set1_epi32(0x00FF00FF), wrb = _mm256_set1_epi32(29 << 16 | 77);
        const __m256i wg = _mm256_set1_epi32(150);
        for (; i + 8 <= n; i += 8) {
            const __m256i v = _mm256_loadu_si256((const __m256i *)(const void *)(r + 4 * i));
            const __m256i rb = _mm256_and_si256(v, lo8), ga = _mm256_and_si256(_mm256_srli_epi16(v, 8), lo8);
            __m256i g = _mm256_add_epi32(_mm256_madd_epi16(rb, wrb), _mm256_madd_epi16(ga, wg));
            g = _mm256_srli_epi32(g, 8);
            _mm256_storeu_ps(o + i, _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(g), k128), inv));
        }
    } else {
        /* 16 RGB pixels (48 bytes) per step: the R, G and B bytes gathered by byte shuffles */
        const __m128i ra = _mm_setr_epi8(0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i rb_ = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14, -1, -1, -1, -1, -1);
        const __m128i rc = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 1, 4, 7, 10, 13);
        const __m128i ga = _mm_setr_epi8(1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i gb = _mm_setr_epi8(-1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15, -1, -1, -1, -1, -1);
        const __m128i gc = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 5, 8, 11, 14);
        const __m128i ba = _mm_setr_epi8(2, 5, 8, 11, 14, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
        const __m128i bb = _mm_setr_epi8(-1, -1, -1, -1, -1, 1, 4, 7, 10, 13, -1, -1, -1, -1, -1, -1);
        const __m128i bc = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 3, 6, 9, 12, 15);
        const __m256i w77 = _mm256_set1_epi16(77), w150 = _mm256_set1_epi16(150), w29 = _mm256_set1_epi16(29);
        for (; i + 16 <= n; i += 16) {
            const __m128i a = _mm_loadu_si128((const __m128i *)(const void *)(r + 3 * i));
            const __m128i b = _mm_loadu_si128((const __m128i *)(const void *)(r + 3 * i + 16));
            const __m128i c = _mm_loadu_si128((const __m128i *)(const void *)(r + 3 * i + 32));
            const __m128i R = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(a, ra), _mm_shuffle_epi8(b, rb_)),
                                           _mm_shuffle_epi8(c, rc));
            const __m128i G = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(a, ga), _mm_shuffle_epi8(b, gb)),
                                           _mm_shuffle_epi8(c, gc));
            const __m128i B = _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(a, ba), _mm_shuffle_epi8(b, bb)),
                                           _mm_shuffle_epi8(c, bc));
            /* 77 R + 150 G + 29 B <= 65280: exact in unsigned 16-bit lanes */
            __m256i y16 = _mm256_add_epi16(_mm256_mullo_epi16(_mm256_cvtepu8_epi16(R), w77),
                                           _mm256_mullo_epi16(_mm256_cvtepu8_epi16(G), w150));
            y16 = _mm256_srli_epi16(_mm256_add_epi16(y16, _mm256_mullo_epi16(_mm256_cvtepu8_epi16(B), w29)), 8);
            {
                const __m256i g0 = _mm256_cvtepu16_epi32(_mm256_castsi256_si128(y16));
                const __m256i g1 = _mm256_cvtepu16_epi32(_mm256_extracti128_si256(y16, 1));
                _mm256_storeu_ps(o + i, _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(g0), k128), inv));
                _mm256_storeu_ps(o + i + 8, _mm256_mul_ps(_mm256_sub_ps(_mm256_cvtepi32_ps(g1), k128), inv));
            }
        }
    }
#endif
    for (; i < n; i++) {
        const unsigned char *p = r + (size_t)i * S->ch;
        unsigned g = S->ch == 1 ? p[0] : (77u * p[0] + 150u * p[1] + 29u * p[2]) >> 8;
        o[i] = ((float)g - 128.0f) * 0.0078125f;
    }
}

#if defined(__GNUC__) || defined(__clang__)
#define PREFETCH(p) __builtin_prefetch(p)
#else
#define PREFETCH(p) ((void)(p))
#endif

/* prefetch the source bytes of output row oy's columns [x0, x1) (its four source rows): a frame fresh from a capture
 * is in DRAM, and the stem reads it once */
static void stem_prefetch(const tsd_src *S, unsigned oy, unsigned x0, unsigned x1)
{
    unsigned ky, sx = 4 * x0, ex = 4 * x1;
    if (sx >= S->w) return;
    if (ex > S->w) ex = S->w;
    for (ky = 0; ky < 4; ky++) {
        const unsigned sy = 4 * oy + ky;
        const unsigned char *r;
        size_t b, n;
        if (sy >= S->h) return;
        r = S->px + (size_t)sy * S->stride + (size_t)sx * S->ch;
        n = (size_t)(ex - sx) * S->ch;
        for (b = 0; b < n; b += 64) PREFETCH(r + b);
        PREFETCH(r + n - 1);
    }
}

/* the stem's four source rows of output row oy, columns [x0, x1): R[ky][0 .. 4 (x1 - x0)) floats (the edge padding:
 * columns past the source repeat its last column, rows past it its last row) */
static void stem_rows(const tsd_src *S, unsigned oy, unsigned x0, unsigned x1, float (*R)[4 * CHUNK_MAX])
{
    const unsigned n = 4 * (x1 - x0), sx = 4 * x0;
    unsigned ky, inside = sx < S->w ? S->w - sx : 0, i;
    if (inside > n) inside = n;
    for (ky = 0; ky < 4; ky++) {
        unsigned sy = 4 * oy + ky;
        if (sy >= S->h) sy = S->h - 1;
        if (inside) src_floats(S, sx, sy, inside, R[ky]);
        if (inside < n) {
            float v;
            src_floats(S, S->w - 1, sy, 1, &v);
            for (i = inside; i < n; i++) R[ky][i] = v;
        }
    }
}

/* the stem (4x4 at stride 4, 1 -> C, ReLU) on n output columns from their four source rows: out[p][co] = relu(b[co] +
 * sum_{ky, kx} R[ky][4 p + kx] w[4 ky + kx][co]), the taps in ci = 4 ky + kx order (pw_chunk's order on the patches:
 * the same bits as the patch form) */
static void stem_chunk(const float (*R)[4 * CHUNK_MAX], unsigned n, const vocr_tsd_layer *L, float *out)
{
    const unsigned C = L->cout;
    const float *w = L->w, *b = L->b;
    unsigned co = 0, p;
#if TSD_AVX2
    const __m256 z = _mm256_setzero_ps();
    /* the taps in ci = 4 ky + kx order: ky outermost (a source row pointer), kx unrolled (constant offsets) */
#define ST6(kx)                                                                       \
    do {                                                                              \
        const __m256 w0 = _mm256_loadu_ps(wk + (kx) * C), w1 = _mm256_loadu_ps(wk + (kx) * C + 8); \
        __m256 v;                                                                     \
        v = _mm256_broadcast_ss(rk + (kx));                                           \
        a00 = _mm256_fmadd_ps(v, w0, a00);                                            \
        a01 = _mm256_fmadd_ps(v, w1, a01);                                            \
        v = _mm256_broadcast_ss(rk + 4 + (kx));                                       \
        a10 = _mm256_fmadd_ps(v, w0, a10);                                            \
        a11 = _mm256_fmadd_ps(v, w1, a11);                                            \
        v = _mm256_broadcast_ss(rk + 8 + (kx));                                       \
        a20 = _mm256_fmadd_ps(v, w0, a20);                                            \
        a21 = _mm256_fmadd_ps(v, w1, a21);                                            \
        v = _mm256_broadcast_ss(rk + 12 + (kx));                                      \
        a30 = _mm256_fmadd_ps(v, w0, a30);                                            \
        a31 = _mm256_fmadd_ps(v, w1, a31);                                            \
        v = _mm256_broadcast_ss(rk + 16 + (kx));                                      \
        a40 = _mm256_fmadd_ps(v, w0, a40);                                            \
        a41 = _mm256_fmadd_ps(v, w1, a41);                                            \
        v = _mm256_broadcast_ss(rk + 20 + (kx));                                      \
        a50 = _mm256_fmadd_ps(v, w0, a50);                                            \
        a51 = _mm256_fmadd_ps(v, w1, a51);                                            \
    } while (0)
    for (; co + 16 <= C; co += 16) {
        const __m256 b0 = _mm256_loadu_ps(b + co), b1 = _mm256_loadu_ps(b + co + 8);
        for (p = 0; p + 6 <= n; p += 6) {
            __m256 a00 = b0, a01 = b1, a10 = b0, a11 = b1, a20 = b0, a21 = b1;
            __m256 a30 = b0, a31 = b1, a40 = b0, a41 = b1, a50 = b0, a51 = b1;
            unsigned ky;
            for (ky = 0; ky < 4; ky++) {
                const float *rk = R[ky] + 4 * p, *wk = w + (size_t)(4 * ky) * C + co;
                ST6(0);
                ST6(1);
                ST6(2);
                ST6(3);
            }
            _mm256_storeu_ps(out + (size_t)p * C + co, _mm256_max_ps(a00, z));
            _mm256_storeu_ps(out + (size_t)p * C + co + 8, _mm256_max_ps(a01, z));
            _mm256_storeu_ps(out + (size_t)(p + 1) * C + co, _mm256_max_ps(a10, z));
            _mm256_storeu_ps(out + (size_t)(p + 1) * C + co + 8, _mm256_max_ps(a11, z));
            _mm256_storeu_ps(out + (size_t)(p + 2) * C + co, _mm256_max_ps(a20, z));
            _mm256_storeu_ps(out + (size_t)(p + 2) * C + co + 8, _mm256_max_ps(a21, z));
            _mm256_storeu_ps(out + (size_t)(p + 3) * C + co, _mm256_max_ps(a30, z));
            _mm256_storeu_ps(out + (size_t)(p + 3) * C + co + 8, _mm256_max_ps(a31, z));
            _mm256_storeu_ps(out + (size_t)(p + 4) * C + co, _mm256_max_ps(a40, z));
            _mm256_storeu_ps(out + (size_t)(p + 4) * C + co + 8, _mm256_max_ps(a41, z));
            _mm256_storeu_ps(out + (size_t)(p + 5) * C + co, _mm256_max_ps(a50, z));
            _mm256_storeu_ps(out + (size_t)(p + 5) * C + co + 8, _mm256_max_ps(a51, z));
        }
        for (; p < n; p++) {
            __m256 a0 = b0, a1 = b1;
            unsigned ci;
            for (ci = 0; ci < 16; ci++) {
                const __m256 v = _mm256_broadcast_ss(R[ci >> 2] + 4 * p + (ci & 3));
                a0 = _mm256_fmadd_ps(v, _mm256_loadu_ps(w + (size_t)ci * C + co), a0);
                a1 = _mm256_fmadd_ps(v, _mm256_loadu_ps(w + (size_t)ci * C + co + 8), a1);
            }
            _mm256_storeu_ps(out + (size_t)p * C + co, _mm256_max_ps(a0, z));
            _mm256_storeu_ps(out + (size_t)p * C + co + 8, _mm256_max_ps(a1, z));
        }
    }
#undef ST6
#define ST12(kx)                                                                      \
    do {                                                                              \
        const __m256 w0 = _mm256_loadu_ps(wk + (kx) * C);                             \
        a[0] = _mm256_fmadd_ps(_mm256_broadcast_ss(rk + (kx)), w0, a[0]);             \
        a[1] = _mm256_fmadd_ps(_mm256_broadcast_ss(rk + 4 + (kx)), w0, a[1]);         \
        a[2] = _mm256_fmadd_ps(_mm256_broadcast_ss(rk + 8 + (kx)), w0, a[2]);         \
        a[3] = _mm256_fmadd_ps(_mm256_broadcast_ss(rk + 12 + (kx)), w0, a[3]);        \
        a[4] = _mm256_fmadd_ps(_mm256_broadcast_ss(rk + 16 + (kx)), w0, a[4]);        \
        a[5] = _mm256_fmadd_ps(_mm256_broadcast_ss(rk + 20 + (kx)), w0, a[5]);        \
        a[6] = _mm256_fmadd_ps(_mm256_broadcast_ss(rk + 24 + (kx)), w0, a[6]);        \
        a[7] = _mm256_fmadd_ps(_mm256_broadcast_ss(rk + 28 + (kx)), w0, a[7]);        \
        a[8] = _mm256_fmadd_ps(_mm256_broadcast_ss(rk + 32 + (kx)), w0, a[8]);        \
        a[9] = _mm256_fmadd_ps(_mm256_broadcast_ss(rk + 36 + (kx)), w0, a[9]);        \
        a[10] = _mm256_fmadd_ps(_mm256_broadcast_ss(rk + 40 + (kx)), w0, a[10]);      \
        a[11] = _mm256_fmadd_ps(_mm256_broadcast_ss(rk + 44 + (kx)), w0, a[11]);      \
    } while (0)
    if (co < C) {   /* 8 channels left: 12 columns a step */
        const __m256 b0 = _mm256_loadu_ps(b + co);
        for (p = 0; p + 12 <= n; p += 12) {
            __m256 a[12];
            unsigned ky, r;
            for (r = 0; r < 12; r++) a[r] = b0;
            for (ky = 0; ky < 4; ky++) {
                const float *rk = R[ky] + 4 * p, *wk = w + (size_t)(4 * ky) * C + co;
                ST12(0);
                ST12(1);
                ST12(2);
                ST12(3);
            }
            for (r = 0; r < 12; r++) _mm256_storeu_ps(out + (size_t)(p + r) * C + co, _mm256_max_ps(a[r], z));
        }
        for (; p < n; p++) {
            __m256 a0 = b0;
            unsigned ci;
            for (ci = 0; ci < 16; ci++)
                a0 = _mm256_fmadd_ps(_mm256_broadcast_ss(R[ci >> 2] + 4 * p + (ci & 3)),
                                     _mm256_loadu_ps(w + (size_t)ci * C + co), a0);
            _mm256_storeu_ps(out + (size_t)p * C + co, _mm256_max_ps(a0, z));
        }
    }
#undef ST12
#else
    for (p = 0; p < n; p++) {
        float acc[VOCR_TSD_MAX_CHANNELS];
        unsigned ci;
        for (co = 0; co < C; co++) acc[co] = b[co];
        for (ci = 0; ci < 16; ci++) {
            const float v = R[ci >> 2][4 * p + (ci & 3)], *wr = w + (size_t)ci * C;
            for (co = 0; co < C; co++) acc[co] += v * wr[co];
        }
        for (co = 0; co < C; co++) out[(size_t)p * C + co] = acc[co] > 0.0f ? acc[co] : 0.0f;
    }
#endif
}

/* ----------------------------------------------------------------------------- the network */

typedef struct {
    unsigned H[4], W[4];       /* map sides at strides 4, 8, 16, 32 */
    size_t X[4][2];            /* each stage's maps (floats from the aligned arena start) */
    size_t P[4];               /* the pyramid's levels (those that are kept) */
    size_t Q;                  /* the smoothing blocks' second map (smooth >= 2) */
    size_t maps, end;
} tsd_plan;

static void plan(const vocr_tsd_model *m, unsigned pw, unsigned ph, tsd_plan *P)
{
    const size_t A = 16;   /* floats: 64 bytes */
    unsigned s, h = m->hs == 4 ? 0u : 1u;
    size_t off = 0;
    memset(P, 0, sizeof *P);
    for (s = 0; s < 4; s++) {
        P->H[s] = ph >> (2 + s);
        P->W[s] = pw >> (2 + s);
    }
    for (s = 0; s < 4; s++) {
        const size_t n = (size_t)P->H[s] * P->W[s] * m->c[s];
        P->X[s][0] = off;
        off += ALIGN_UP(n, A);
        P->X[s][1] = off;
        if (m->n[s]) off += ALIGN_UP(n, A);
    }
    for (s = h; s < 4; s++) {
        P->P[s] = off;
        if (s > h || m->smooth) off += ALIGN_UP((size_t)P->H[s] * P->W[s] * m->F, A);
    }
    P->Q = off;
    if (m->smooth >= 2) off += ALIGN_UP((size_t)P->H[h] * P->W[h] * m->F, A);
    P->maps = off;
    off += ALIGN_UP((size_t)P->H[h] * P->W[h] * m->out, A);
    P->end = off;
}

size_t vocr_tsd_maps_bytes(const vocr_tsd_model *m, unsigned pw, unsigned ph, unsigned ntasks)
{
    tsd_plan P;
    (void)ntasks;
    if (!m || !pw || !ph) return 0;
    plan(m, pw, ph, &P);
    return P.end * sizeof(float) + 64;
}

static float *aligned_floats(void *arena)
{
    uintptr_t a = ((uintptr_t)arena + 63u) & ~(uintptr_t)63u;
    return (float *)(void *)a;
}

enum { ST_STEM, ST_SEP, ST_LAT, ST_GC };

typedef struct {
    int kind, res, head;          /* res: a residual block; head: the head fused after this step (into maps) */
    const vocr_tsd_layer *a, *b;  /* SEP: depthwise, pointwise; LAT: the lateral; STEM: the stem */
    const float *in;              /* SEP / LAT: the input map, H x W */
    unsigned H, W, Ho, Wo;        /* input and output sides */
    float *out;                   /* the output map (NULL when the head is fused) */
    const float *up;              /* LAT: the coarser level (Ho / 2 x Wo / 2 x F), or NULL */
    const char *name;
} tsd_step;

typedef struct {
    const vocr_tsd_model *m;
    tsd_src src;
    float *maps;
    unsigned lazy;             /* the head's maps past K only on kernel cells (K > 0): what the decode reads */
    tsd_step st[MAX_STEPS];
    unsigned n, cur;
} tsd_net;

static unsigned chunk_of(unsigned cmax)
{
    unsigned c = CHUNK_FLOATS / cmax / 12 * 12;
    return c > CHUNK_MAX ? CHUNK_MAX : c;
}

/* the head on a chunk ([P][F]) into the maps (stride out per cell) */
/* acc + x w as the head's kernels compute it: one fused multiply-add (AVX2 + FMA builds) or a product and a sum */
static inline float madd(float acc, float x, float w)
{
#if TSD_AVX2
    return _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(x), _mm_set_ss(w), _mm_set_ss(acc)));
#else
    return acc + x * w;
#endif
}

/* the head's map co of one cell: b[co], then + x[ci] w[ci][co] for ci in order (pw_chunk's chain for that output) */
static inline float head_one(const vocr_tsd_layer *L, const float *x, unsigned co)
{
    const unsigned F = L->cin, ld = L->cout;
    float acc = L->b[co];
    unsigned ci;
    for (ci = 0; ci < F; ci++) acc = madd(acc, x[ci], L->w[(size_t)ci * ld + co]);
    return acc;
}

/* the K map of P cells (in: [P][F]) into maps[p * out]: eight cells a step, the cells in the lanes (an 8 x 8 transpose
 * of the input per 8 channels), each lane the same chain as pw_chunk's for K: the same bits */
static void head_k(const vocr_tsd_layer *L, const float *in, unsigned P, unsigned out, float *maps)
{
    const unsigned F = L->cin;
    unsigned p = 0;
#if TSD_AVX2
    const unsigned ld = L->cout;
    for (; p + 8 <= P; p += 8) {
        const float *x = in + (size_t)p * F;
        __m256 acc = _mm256_set1_ps(L->b[0]);
        float o[8];
        unsigned cb, j;
        for (cb = 0; cb < F; cb += 8) {
            __m256 r0 = _mm256_loadu_ps(x + cb), r1 = _mm256_loadu_ps(x + F + cb);
            __m256 r2 = _mm256_loadu_ps(x + 2 * F + cb), r3 = _mm256_loadu_ps(x + 3 * F + cb);
            __m256 r4 = _mm256_loadu_ps(x + 4 * F + cb), r5 = _mm256_loadu_ps(x + 5 * F + cb);
            __m256 r6 = _mm256_loadu_ps(x + 6 * F + cb), r7 = _mm256_loadu_ps(x + 7 * F + cb);
            __m256 t0 = _mm256_unpacklo_ps(r0, r1), t1 = _mm256_unpackhi_ps(r0, r1);
            __m256 t2 = _mm256_unpacklo_ps(r2, r3), t3 = _mm256_unpackhi_ps(r2, r3);
            __m256 t4 = _mm256_unpacklo_ps(r4, r5), t5 = _mm256_unpackhi_ps(r4, r5);
            __m256 t6 = _mm256_unpacklo_ps(r6, r7), t7 = _mm256_unpackhi_ps(r6, r7);
            __m256 u0 = _mm256_shuffle_ps(t0, t2, 0x44), u1 = _mm256_shuffle_ps(t0, t2, 0xEE);
            __m256 u2 = _mm256_shuffle_ps(t1, t3, 0x44), u3 = _mm256_shuffle_ps(t1, t3, 0xEE);
            __m256 u4 = _mm256_shuffle_ps(t4, t6, 0x44), u5 = _mm256_shuffle_ps(t4, t6, 0xEE);
            __m256 u6 = _mm256_shuffle_ps(t5, t7, 0x44), u7 = _mm256_shuffle_ps(t5, t7, 0xEE);
            __m256 c[8];
            c[0] = _mm256_permute2f128_ps(u0, u4, 0x20);   /* channel cb + 0 of the 8 cells */
            c[1] = _mm256_permute2f128_ps(u1, u5, 0x20);
            c[2] = _mm256_permute2f128_ps(u2, u6, 0x20);
            c[3] = _mm256_permute2f128_ps(u3, u7, 0x20);
            c[4] = _mm256_permute2f128_ps(u0, u4, 0x31);
            c[5] = _mm256_permute2f128_ps(u1, u5, 0x31);
            c[6] = _mm256_permute2f128_ps(u2, u6, 0x31);
            c[7] = _mm256_permute2f128_ps(u3, u7, 0x31);
            for (j = 0; j < 8; j++) acc = _mm256_fmadd_ps(c[j], _mm256_set1_ps(L->w[(size_t)(cb + j) * ld]), acc);
        }
        _mm256_storeu_ps(o, acc);
        for (j = 0; j < 8; j++) maps[(size_t)(p + j) * out] = o[j];
    }
#endif
    for (; p < P; p++) maps[(size_t)p * out] = head_one(L, in + (size_t)p * F, 0);
}

static void head_chunk(const vocr_tsd_model *m, const float *in, unsigned P, float *maps, unsigned lazy)
{
    float tmp[CHUNK_MAX * (6 + VOCR_TSD_MAX_TAGS + 2)];
    epi_t e;
    unsigned p, co;
    if (lazy) {   /* K everywhere; the other maps where K > 0 (the decode reads them nowhere else) */
        float xk[CHUNK_FLOATS];   /* P x F <= CHUNK_FLOATS (chunk_of) */
        unsigned idx[CHUNK_MAX], nk = 0, F = m->F, k;
        head_k(&m->head, in, P, m->out, maps);
        for (p = 0; p < P; p++)
            if (maps[(size_t)p * m->out] > 0.0f) idx[nk++] = p;
        if (!nk) return;
        /* the kernel cells' inputs side by side, then the whole head on them (pw_chunk: the same chains) */
        for (k = 0; k < nk; k++) memcpy(xk + (size_t)k * F, in + (size_t)idx[k] * F, F * sizeof(float));
        e.kind = EPI_NONE;
        e.res = e.up = NULL;
        e.x0 = 0;
        pw_chunk(xk, nk, &m->head, &e, tmp);
        for (k = 0; k < nk; k++)
            for (co = 1; co < m->out; co++) maps[(size_t)idx[k] * m->out + co] = tmp[(size_t)k * m->outp + co];
        return;
    }
    e.kind = EPI_NONE;
    e.res = e.up = NULL;
    e.x0 = 0;
    pw_chunk(in, P, &m->head, &e, tmp);
    for (p = 0; p < P; p++) memcpy(maps + (size_t)p * m->out, tmp + (size_t)p * m->outp, m->out * sizeof(float));
}

static void run_rows(const tsd_net *N, const tsd_step *t, unsigned y0, unsigned y1)
{
    const vocr_tsd_model *m = N->m;
    float T[CHUNK_FLOATS], U[CHUNK_FLOATS];
    unsigned y, x0;
    epi_t e;
    memset(&e, 0, sizeof e);
    if (t->kind == ST_STEM) {
        float R[4][4 * CHUNK_MAX];
        const unsigned C = m->c[0], ch = CHUNK_MAX;
        for (y = y0; y < y1; y++)
            for (x0 = 0; x0 < t->Wo; x0 += ch) {
                const unsigned x1 = x0 + ch < t->Wo ? x0 + ch : t->Wo;
                stem_prefetch(&N->src, y + STEM_PF_ROWS, x0, x1);
                stem_rows(&N->src, y, x0, x1, R);
                stem_chunk((const float (*)[4 * CHUNK_MAX])R, x1 - x0, &m->stem, t->out + ((size_t)y * t->Wo + x0) * C);
            }
        return;
    }
    if (t->kind == ST_SEP) {
        const unsigned Ci = t->a->cin, Co = t->b->cout, ch = chunk_of(Ci > Co ? Ci : Co);
        for (x0 = 0; x0 < t->Wo; x0 += ch)
            for (y = y0; y < y1; y++) {
                const unsigned x1 = x0 + ch < t->Wo ? x0 + ch : t->Wo;
                float *o = t->head ? U : t->out + ((size_t)y * t->Wo + x0) * Co;
                dw_chunk(t->in, t->H, t->W, t->a, y, x0, x1, T);
                e.kind = t->res ? EPI_RES : EPI_RELU;
                e.res = t->res ? t->in + ((size_t)y * t->W + x0) * Ci : NULL;
                pw_chunk(T, x1 - x0, t->b, &e, o);
                if (t->head) head_chunk(m, U, x1 - x0, N->maps + ((size_t)y * t->Wo + x0) * m->out, N->lazy);
            }
        return;
    }
    if (t->kind == ST_LAT) {
        const unsigned Ci = t->a->cin, F = m->F, ch = chunk_of(Ci > F ? Ci : F);
        for (y = y0; y < y1; y++)
            for (x0 = 0; x0 < t->Wo; x0 += ch) {
                const unsigned x1 = x0 + ch < t->Wo ? x0 + ch : t->Wo;
                float *o = t->head ? U : t->out + ((size_t)y * t->Wo + x0) * F;
                e.kind = t->up ? EPI_UP : EPI_RELU;
                e.up = t->up ? t->up + (size_t)(y >> 1) * (t->Wo >> 1) * F : NULL;
                e.x0 = x0;
                pw_chunk(t->in + ((size_t)y * t->W + x0) * Ci, x1 - x0, t->a, &e, o);
                if (t->head) head_chunk(m, U, x1 - x0, N->maps + ((size_t)y * t->Wo + x0) * m->out, N->lazy);
            }
        return;
    }
}

/* the global context: x += relu(W mean(x) + b), the mean over every position of the (padded) stride-32 map */
static void gc_apply(const vocr_tsd_model *m, float *x, size_t P)
{
    const unsigned C = m->c[3];
    double sum[VOCR_TSD_MAX_CHANNELS];
    float mean[VOCR_TSD_MAX_CHANNELS], g[VOCR_TSD_MAX_CHANNELS];
    size_t p;
    unsigned c, ci;
    for (c = 0; c < C; c++) sum[c] = 0.0;
    for (p = 0; p < P; p++)
        for (c = 0; c < C; c++) sum[c] += (double)x[p * C + c];
    for (c = 0; c < C; c++) mean[c] = (float)(sum[c] / (double)P);
    for (c = 0; c < C; c++) {
        float a = m->gcl.b[c];
        for (ci = 0; ci < C; ci++) a += mean[ci] * m->gcl.w[(size_t)ci * C + c];
        g[c] = a > 0.0f ? a : 0.0f;
    }
    for (p = 0; p < P; p++)
        for (c = 0; c < C; c++) x[p * C + c] += g[c];
}

static void step_task(void *arg, unsigned task, unsigned ntasks)
{
    const tsd_net *N = (const tsd_net *)arg;
    const tsd_step *t = &N->st[N->cur];
    const unsigned rows = t->Ho;
    run_rows(N, t, (unsigned)((size_t)rows * task / ntasks), (unsigned)((size_t)rows * (task + 1) / ntasks));
}

static tsd_step *add_step(tsd_net *N, int kind, const char *name)
{
    tsd_step *t = &N->st[N->n++];
    memset(t, 0, sizeof *t);
    t->kind = kind;
    t->name = name;
    return t;
}

static const char *const STAGE_NAMES[4][1 + VOCR_TSD_MAX_BLOCKS] = {
    {"s4.-", "s4.b0", "s4.b1", "s4.b2", "s4.b3", "s4.b4", "s4.b5", "s4.b6", "s4.b7"},
    {"s8.down", "s8.b0", "s8.b1", "s8.b2", "s8.b3", "s8.b4", "s8.b5", "s8.b6", "s8.b7"},
    {"s16.down", "s16.b0", "s16.b1", "s16.b2", "s16.b3", "s16.b4", "s16.b5", "s16.b6", "s16.b7"},
    {"s32.down", "s32.b0", "s32.b1", "s32.b2", "s32.b3", "s32.b4", "s32.b5", "s32.b6", "s32.b7"}};
static const char *const LAT_NAMES[4] = {"lat4", "lat8", "lat16", "lat32"};
static const char *const SMOOTH_NAMES[VOCR_TSD_MAX_SMOOTH] = {"smooth0", "smooth1", "smooth2", "smooth3"};

/* build the steps of the network on a source (the stem reads it with the edge padding to pw x ph) */
static void build(tsd_net *N, const vocr_tsd_model *m, float *A, const tsd_plan *P)
{
    const unsigned h = m->hs == 4 ? 0u : 1u;
    unsigned s, i;
    const float *coarse;
    float *fin[4];
    tsd_step *t;
    N->m = m;
    N->n = 0;
    N->maps = A + P->maps;
    t = add_step(N, ST_STEM, "stem");
    t->a = &m->stem;
    t->Ho = P->H[0];
    t->Wo = P->W[0];
    t->out = A + P->X[0][0];
    for (s = 0; s < 4; s++) {
        unsigned cur = 0;
        if (s > 0) {
            t = add_step(N, ST_SEP, STAGE_NAMES[s][0]);
            t->a = &m->dw[s][0];
            t->b = &m->pw[s][0];
            t->in = fin[s - 1];
            t->H = P->H[s - 1];
            t->W = P->W[s - 1];
            t->Ho = P->H[s];
            t->Wo = P->W[s];
            t->out = A + P->X[s][0];
        }
        for (i = 0; i < m->n[s]; i++) {
            t = add_step(N, ST_SEP, STAGE_NAMES[s][1 + i]);
            t->a = &m->dw[s][1 + i];
            t->b = &m->pw[s][1 + i];
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
        t = add_step(N, ST_GC, "gc");
        t->out = fin[3];
        t->H = P->H[3];
        t->W = P->W[3];
        t->Ho = 1;
    }
    coarse = NULL;
    for (s = 3; s + 1 > h; s--) {
        t = add_step(N, ST_LAT, LAT_NAMES[s]);
        t->a = &m->lat[s];
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
    for (i = 0; i < m->smooth; i++) {
        const float *src = (i % 2 == 0) ? A + P->P[h] : A + P->Q;
        t = add_step(N, ST_SEP, SMOOTH_NAMES[i]);
        t->a = &m->sdw[i];
        t->b = &m->spw[i];
        t->in = src;
        t->H = t->Ho = P->H[h];
        t->W = t->Wo = P->W[h];
        t->res = 1;
        if (i + 1 == m->smooth)
            t->head = 1;
        else
            t->out = (i % 2 == 0) ? A + P->Q : A + P->P[h];
    }
}

static void net_run(tsd_net *N, vocr_tsd_runner run, void *ctx, unsigned ntasks)
{
    PROF_START();
    for (N->cur = 0; N->cur < N->n; N->cur++) {
        const tsd_step *t = &N->st[N->cur];
        unsigned nt = ntasks < t->Ho ? ntasks : t->Ho;
        if (t->kind == ST_GC) {
            gc_apply(N->m, t->out, (size_t)t->H * t->W);
        } else if (nt > 1 && run) {
            run(ctx, step_task, N, nt);
        } else {
            step_task(N, 0, 1);
        }
        PROF_MARK(PROF_STEP0 + N->cur, t->name);
    }
}

int vocr_tsd_maps_mt(const vocr_tsd_model *m, const unsigned char *in, unsigned pw, unsigned ph, void *arena,
                     size_t bytes, const float **maps, vocr_tsd_runner run, void *ctx, unsigned ntasks)
{
    tsd_plan P;
    tsd_net N;
    float *A;
    if (!m || !in || !arena || !maps || !pw || !ph || pw % VOCR_TSD_ALIGN || ph % VOCR_TSD_ALIGN ||
        pw > VOCR_TSD_MAX_SIDE + VOCR_TSD_ALIGN || ph > VOCR_TSD_MAX_SIDE + VOCR_TSD_ALIGN)
        return VOCR_TSD_E_ARG;
    if (bytes < vocr_tsd_maps_bytes(m, pw, ph, ntasks)) return VOCR_TSD_E_ARENA;
    if (!run || ntasks < 1) ntasks = 1;
    plan(m, pw, ph, &P);
    A = aligned_floats(arena);
    N.src.px = in;
    N.src.w = pw;
    N.src.h = ph;
    N.src.stride = pw;
    N.src.ch = 1;
    N.lazy = 0;   /* every map of every cell */
    build(&N, m, A, &P);
    net_run(&N, run, ctx, ntasks);
    *maps = N.maps;
    return VOCR_TSD_OK;
}

int vocr_tsd_maps(const vocr_tsd_model *m, const unsigned char *in, unsigned pw, unsigned ph, void *arena,
                  size_t bytes, const float **maps)
{
    return vocr_tsd_maps_mt(m, in, pw, ph, arena, bytes, maps, NULL, NULL, 1);
}

/* ----------------------------------------------------------------------------- the decode */

size_t vocr_tsd_decode_bytes(unsigned mw, unsigned mh, unsigned tags)
{
    size_t cells = (size_t)mw * mh, nl = cells / 2 + 2;
    return sizeof(int) * cells + sizeof(int) * nl * 8 + sizeof(double) * nl * (9 + (size_t)tags) +
           sizeof(unsigned char) * nl + sizeof(vocr_tsd_box) * nl + 64;
}

static int uf_find(int *par, int a)
{
    while (par[a] != a) {
        par[a] = par[par[a]];
        a = par[a];
    }
    return a;
}

/* a before b: logit descending, then y, then x (the stable sort keeps the rest in order) */
static int box_before(const vocr_tsd_box *a, const vocr_tsd_box *b)
{
    if (a->logit != b->logit) return a->logit > b->logit;
    if (a->y != b->y) return a->y < b->y;
    return a->x < b->x;
}

/* stable bottom-up merge sort of idx[0 .. n) by box_before */
static void sort_boxes(const vocr_tsd_box *c, int *idx, int *tmp, unsigned n)
{
    unsigned width, i;
    int *a = idx, *b = tmp;
    for (width = 1; width < n; width *= 2) {
        for (i = 0; i < n; i += 2 * width) {
            unsigned lo = i, mid = i + width < n ? i + width : n, hi = i + 2 * width < n ? i + 2 * width : n;
            unsigned p = lo, q = mid, k = lo;
            while (p < mid && q < hi) b[k++] = box_before(&c[a[q]], &c[a[p]]) ? a[q++] : a[p++];
            while (p < mid) b[k++] = a[p++];
            while (q < hi) b[k++] = a[q++];
        }
        {
            int *sw = a;
            a = b;
            b = sw;
        }
    }
    if (a != idx) memcpy(idx, a, n * sizeof(int));
}

int vocr_tsd_decode_ex(const vocr_tsd_decode_opts *o, const float *maps, unsigned mw, unsigned mh, unsigned iw,
                       unsigned ih, void *scratch, size_t bytes, vocr_tsd_set *out)
{
    unsigned s, T, ch, hm, wm, r, c, t, k, ncand = 0;
    size_t cells, nl;
    unsigned char *base, *vert;
    int *lab, *par, *fin, *imin, *imax, *jmin, *jmax, *order, *tmp, next = 1, n = 1;
    double *cnt, *sk, *so, *sn, *sf, *ss, *se, *n1, *n2, *st, half;
    vocr_tsd_box *cand;
    if (!o || !maps || !scratch || !out || !o->stride || o->tags > VOCR_TSD_MAX_TAGS ||
        o->max_boxes > VOCR_TSD_MAX_BOXES)
        return VOCR_TSD_E_ARG;
    s = o->stride;
    T = o->tags;
    ch = 6 + T;
    hm = (ih + s - 1) / s;
    wm = (iw + s - 1) / s;
    if (hm > mh || wm > mw) return VOCR_TSD_E_ARG;
    if (bytes < vocr_tsd_decode_bytes(mw, mh, T)) return VOCR_TSD_E_ARENA;
    out->count = out->truncated = 0;
    out->in_w = iw;
    out->in_h = ih;
    cells = (size_t)hm * wm;
    if (!cells) return VOCR_TSD_OK;
    nl = (size_t)mw * mh / 2 + 2;
    base = (unsigned char *)(((uintptr_t)scratch + 7u) & ~(uintptr_t)7u);
    cnt = (double *)(void *)base;
    sk = cnt + nl;
    so = sk + nl;
    sn = so + nl;
    sf = sn + nl;
    ss = sf + nl;
    se = ss + nl;
    n1 = se + nl;
    n2 = n1 + nl;
    st = n2 + nl;   /* nl x T */
    cand = (vocr_tsd_box *)(void *)(st + nl * T);
    lab = (int *)(void *)(cand + nl);
    par = lab + (size_t)mw * mh;
    fin = par + nl;
    imin = fin + nl;
    imax = imin + nl;
    jmin = imax + nl;
    jmax = jmin + nl;
    order = jmax + nl;
    tmp = order + nl;
    vert = (unsigned char *)(void *)(tmp + nl);
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
    if (next == 1) return VOCR_TSD_OK;
    /* final labels in the order of each component's first cell (its smallest provisional label, its root) */
    fin[0] = 0;
    for (k = 1; k < (unsigned)next; k++) fin[k] = uf_find(par, (int)k) == (int)k ? n++ : 0;
    for (k = 1; k < (unsigned)next; k++) fin[k] = fin[uf_find(par, (int)k)];
    for (k = 0; k < (unsigned)n; k++) {
        cnt[k] = sk[k] = so[k] = sn[k] = sf[k] = ss[k] = se[k] = n1[k] = n2[k] = 0.0;
        for (t = 0; t < T; t++) st[(size_t)k * T + t] = 0.0;
        imin[k] = jmin[k] = 1 << 30;
        imax[k] = jmax[k] = -1;
    }
    for (r = 0; r < cells; r++)
        if (lab[r]) lab[r] = fin[lab[r]];
    /* pass 2: per-component sums over the cells in raster order (tools/tsdesk.decode's bincounts) */
    for (r = 0; r < hm; r++)
        for (c = 0; c < wm; c++) {
            const int L = lab[(size_t)r * wm + c];
            const float *v = maps + ((size_t)r * mw + c) * ch;
            if (!L) continue;
            cnt[L] += 1.0;
            sk[L] += (double)v[0];
            so[L] += (double)v[1];
            for (t = 0; t < T; t++) st[(size_t)L * T + t] += (double)v[6 + t];
            if ((int)r < imin[L]) imin[L] = (int)r;
            if ((int)r > imax[L]) imax[L] = (int)r;
            if ((int)c < jmin[L]) jmin[L] = (int)c;
            if ((int)c > jmax[L]) jmax[L] = (int)c;
        }
    for (k = 1; k < (unsigned)n; k++) vert[k] = (so[k] / cnt[k]) > 0.0;
    /* pass 3: across (cell centre -/+ A1 / A2) and the along ends (first / last column, or row, -/+ E1 / E2) */
    half = (double)(s - 1) / 2.0;
    for (r = 0; r < hm; r++)
        for (c = 0; c < wm; c++) {
            const int L = lab[(size_t)r * wm + c];
            const float *v = maps + ((size_t)r * mw + c) * ch;
            unsigned across, along;
            double cc, ca;
            if (!L) continue;
            across = vert[L] ? c : r;
            along = vert[L] ? r : c;
            cc = (double)s * (double)across + half;
            sn[L] += cc - (double)v[2];
            sf[L] += cc + (double)v[3];
            ca = (double)s * (double)along + half;
            if ((int)along == (vert[L] ? imin[L] : jmin[L])) {
                ss[L] += ca - (double)v[4];
                n1[L] += 1.0;
            }
            if ((int)along == (vert[L] ? imax[L] : jmax[L])) {
                se[L] += ca + (double)v[5];
                n2[L] += 1.0;
            }
        }
    for (k = 1; k < (unsigned)n; k++) {
        const double logit = sk[k] / cnt[k], near = sn[k] / cnt[k], far = sf[k] / cnt[k];
        const unsigned v = vert[k];
        const int first = v ? imin[k] : jmin[k], last = v ? imax[k] : jmax[k];
        double size, lo, hi, b0, b1, x0, x1, y0, y1, best = 0.0, second = -INFINITY;
        unsigned tag = 0;
        vocr_tsd_box *bx;
        if (cnt[k] < (double)o->min_cells || logit < o->conf_min_logit) continue;
        size = far - near;
        if (size < o->min_across) continue;
        lo = (double)(s * (unsigned)first);
        hi = (double)(s * ((unsigned)last + 1u));
        b0 = ss[k] / n1[k];
        b1 = se[k] / n2[k];
        if (b0 > lo + (double)s) b0 = lo + (double)s;
        if (b0 < lo - o->along_reach * size) b0 = lo - o->along_reach * size;
        if (b1 < hi - (double)s) b1 = hi - (double)s;
        if (b1 > hi + o->along_reach * size) b1 = hi + o->along_reach * size;
        if (v) {
            x0 = near, x1 = far, y0 = b0, y1 = b1;
        } else {
            x0 = b0, x1 = b1, y0 = near, y1 = far;
        }
        if (x0 < 0.0) x0 = 0.0;
        if (y0 < 0.0) y0 = 0.0;
        if (x1 > (double)iw) x1 = (double)iw;
        if (y1 > (double)ih) y1 = (double)ih;
        if (x1 <= x0 || y1 <= y0) continue;
        /* the tag: class 0 scores 0, class t + 1 the sum of tag map t; the first largest wins */
        for (t = 0; t < T; t++) {
            const double sc = st[(size_t)k * T + t];
            if (sc > best) {
                best = sc;
                tag = t + 1;
            }
        }
        if (T) {
            if (tag != 0) second = 0.0;
            for (t = 0; t < T; t++) {
                const double sc = st[(size_t)k * T + t];
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
        bx->vertical = v;
        bx->tag = tag;
        bx->cells = (unsigned)cnt[k];
        order[ncand] = (int)ncand;
        ncand++;
    }
    sort_boxes(cand, order, tmp, ncand);
    out->truncated = ncand > o->max_boxes;
    out->count = ncand > o->max_boxes ? o->max_boxes : ncand;
    for (k = 0; k < out->count; k++) out->boxes[k] = cand[order[k]];
    return VOCR_TSD_OK;
}

int vocr_tsd_decode(const vocr_tsd_model *m, const float *maps, unsigned mw, unsigned mh, unsigned iw, unsigned ih,
                    void *scratch, size_t bytes, vocr_tsd_set *out)
{
    vocr_tsd_decode_opts o;
    if (!m) return VOCR_TSD_E_ARG;
    vocr_tsd_decode_opts_of(m, &o);
    return vocr_tsd_decode_ex(&o, maps, mw, mh, iw, ih, scratch, bytes, out);
}

/* ----------------------------------------------------------------------------- the detector */

typedef struct {
    size_t net, grey, resized, rscratch, rplan, rwork, rwork_bytes, decode, end;   /* byte offsets (aligned arena) */
    unsigned iw, ih, pw, ph, resize, ntasks;
} detect_plan;

static void dplan(const vocr_tsd_model *m, unsigned kind, unsigned n, unsigned w, unsigned h, unsigned channels,
                  unsigned ntasks, detect_plan *D)
{
    size_t off = 0;
    vocr_tsd_input_size(kind, n, w, h, &D->iw, &D->ih, &D->pw, &D->ph);
    D->resize = D->iw != w || D->ih != h;
    D->net = off;
    off += ALIGN_UP(vocr_tsd_maps_bytes(m, D->pw, D->ph, ntasks), 64);
    D->grey = off;   /* unused: an RGB(A) frame's luma is taken row by row inside the resize */
    D->resized = off;
    off += D->resize ? ALIGN_UP((size_t)D->iw * D->ih, 64) : 0;
    D->rscratch = off;
    off += D->resize ? ALIGN_UP(channels == 1 ? vocr_desk_resize_scratch(w, h, D->iw, D->ih)
                                              : vocr_desk_resize_rgb_scratch(w, h, D->iw, D->ih), 64) : 0;
    /* the banded resize for threads: its plan and a work area per task */
    D->ntasks = ntasks < 1 ? 1 : ntasks > RESIZE_MAX_TASKS ? RESIZE_MAX_TASKS : ntasks;
    D->rplan = off;
    off += D->resize && D->ntasks > 1 ? ALIGN_UP(vocr_desk_resize_plan_bytes(w, h, D->iw, D->ih), 64) : 0;
    D->rwork = off;
    D->rwork_bytes = D->resize && D->ntasks > 1 ? ALIGN_UP(vocr_desk_resize_band_bytes(w, h, D->iw, D->ih), 64) : 0;
    off += D->rwork_bytes * D->ntasks;
    D->decode = off;
    off += ALIGN_UP(vocr_tsd_decode_bytes(D->pw / m->hs, D->ph / m->hs, m->tags), 64);
    D->end = off + 64;
}

size_t vocr_tsd_arena_bytes(const vocr_tsd_model *m, unsigned kind, unsigned n, unsigned w, unsigned h,
                            unsigned channels, unsigned ntasks)
{
    detect_plan D;
    if (!m || !w || !h || w > VOCR_TSD_MAX_SIDE || h > VOCR_TSD_MAX_SIDE) return 0;
    dplan(m, kind, n, w, h, channels, ntasks, &D);
    return D.end;
}

typedef struct {               /* the banded resize on the runner */
    const void *plan;
    const unsigned char *src;
    unsigned stride, channels, dh;
    unsigned char *dst, *work;
    size_t work_bytes;
} resize_job;

static void resize_task(void *arg, unsigned task, unsigned ntasks)
{
    const resize_job *j = (const resize_job *)arg;
    vocr_desk_resize_band(j->plan, j->src, j->stride, j->channels, j->dst, (unsigned)((size_t)j->dh * task / ntasks),
                          (unsigned)((size_t)j->dh * (task + 1) / ntasks), j->work + j->work_bytes * task);
}

int vocr_tsd_detect(const vocr_tsd_model *m, const unsigned char *img, unsigned w, unsigned h, unsigned stride,
                    unsigned channels, unsigned kind, unsigned n, void *arena, size_t bytes, vocr_tsd_set *out)
{
    return vocr_tsd_detect_mt(m, img, w, h, stride, channels, kind, n, arena, bytes, out, NULL, NULL, 1);
}

int vocr_tsd_detect_mt(const vocr_tsd_model *m, const unsigned char *img, unsigned w, unsigned h, unsigned stride,
                       unsigned channels, unsigned kind, unsigned n, void *arena, size_t bytes, vocr_tsd_set *out,
                       vocr_tsd_runner run, void *ctx, unsigned ntasks)
{
    detect_plan D;
    tsd_plan P;
    tsd_net N;
    unsigned char *base;
    float *A;
    double fx, fy;
    unsigned k, keep = 0;
    int rc;
    if (!m || !img || !arena || !out || !w || !h || w > VOCR_TSD_MAX_SIDE || h > VOCR_TSD_MAX_SIDE ||
        (channels != 1 && channels != 3 && channels != 4) || stride < (size_t)w * channels ||
        kind > VOCR_TSD_SIZE_LONG)
        return VOCR_TSD_E_ARG;
    if (!run || ntasks < 1) ntasks = 1;
    dplan(m, kind, n, w, h, channels, ntasks, &D);
    if (bytes < D.end) return VOCR_TSD_E_ARENA;
    base = (unsigned char *)(((uintptr_t)arena + 63u) & ~(uintptr_t)63u);
    PROF_START();
    N.src.px = img;
    N.src.w = w;
    N.src.h = h;
    N.src.stride = stride;
    N.src.ch = channels;
    if (D.resize) {
        /* the policy's resize of the frame's luma (vocr_desk_resize; an RGB(A) frame's luma row by row inside it:
         * vocr_desk_resize_rgb, the same bits as vocr_desk_grey then vocr_desk_resize) */
        if (D.ntasks > 1 && run && !vocr_desk_resize_plan(w, h, D.iw, D.ih, base + D.rplan)) {
            resize_job j;   /* bands of output rows on the runner (vocr_desk_resize_band): the same bits */
            j.plan = base + D.rplan;
            j.src = img;
            j.stride = stride;
            j.channels = channels;
            j.dh = D.ih;
            j.dst = base + D.resized;
            j.work = base + D.rwork;
            j.work_bytes = D.rwork_bytes;
            run(ctx, resize_task, &j, D.ntasks < D.ih ? D.ntasks : D.ih);
        } else if (channels == 1) {
            vocr_desk_resize(img, w, h, stride, base + D.resized, D.iw, D.ih, (int *)(void *)(base + D.rscratch));
        } else {
            vocr_desk_resize_rgb(img, w, h, stride, channels, base + D.resized, D.iw, D.ih,
                                 (int *)(void *)(base + D.rscratch));
        }
        PROF_MARK(1u, "resize");
        N.src.px = base + D.resized;
        N.src.w = D.iw;
        N.src.h = D.ih;
        N.src.stride = D.iw;
        N.src.ch = 1;
    }
    plan(m, D.pw, D.ph, &P);
    A = aligned_floats(base + D.net);
    N.lazy = 1;   /* the maps the decode reads: K, and the rest on kernel cells */
    build(&N, m, A, &P);
    net_run(&N, run, ctx, ntasks);
    PROF_START();
    rc = vocr_tsd_decode(m, N.maps, D.pw / m->hs, D.ph / m->hs, D.iw, D.ih, base + D.decode, D.end - 64 - D.decode,
                         out);
    PROF_MARK(2u, "decode");
    if (rc) return rc;
    /* to frame px (tools/tsdesk.Detector.detect): max(0, x / fx), min(W, (x + w) / fx) */
    fx = (double)D.iw / (double)w;
    fy = (double)D.ih / (double)h;
    for (k = 0; k < out->count; k++) {
        vocr_tsd_box b = out->boxes[k];
        double x0 = b.x / fx, y0 = b.y / fy, x1 = (b.x + b.w) / fx, y1 = (b.y + b.h) / fy;
        x0 = x0 > 0.0 ? x0 : 0.0;
        y0 = y0 > 0.0 ? y0 : 0.0;
        x1 = x1 < (double)w ? x1 : (double)w;
        y1 = y1 < (double)h ? y1 : (double)h;
        if (!(x1 > x0 && y1 > y0)) continue;
        b.x = x0;
        b.y = y0;
        b.w = x1 - x0;
        b.h = y1 - y0;
        out->boxes[keep++] = b;
    }
    out->count = keep;
    PROF_MARK(3u, "boxes");
#ifdef VOCR_TSD_PROF
    vocr_tsd_prof_runs++;
#endif
    return VOCR_TSD_OK;
}
