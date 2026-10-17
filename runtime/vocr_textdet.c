/* vocr_textdet.c - the learned text detector's runtime (vocr_textdet.h, docs/textdet.md).
 *
 * synth/textdet_twin.py is the specification; this file computes exactly what it computes, for every
 * region and every blob its Model accepts (tests/textdet_parity.py checks the head's maps and the
 * lines). In short:
 *
 * Network. x0 = grey - 128 (int8, grey = (77 R + 150 G + 29 B) >> 8). Each layer (dense or depthwise,
 * 1x1 or 3x3 with a stride of 1 or 2 and a dilation d, zero padding of d) sums int8 weights times int8
 * codes into an int32 accumulator, adds the int32 bias and requantises:
 *     out = clamp(floor(((bias + sum) * mult + 2^(shift - 1)) / 2^shift), relu ? 0 : -128, 127).
 * The last layer gives four int8 maps at the output stride s: K (kernel), O (vertical), A1, A2 (near
 * and far edge distances across the line, in 1/2 px).
 *
 * Streaming. The region is read once, top to bottom. Every 3x3 layer keeps a ring of the
 * (k - 1) * d + 1 = 2 d + 1 input rows its kernel spans (3 for the stride-2 layers, whose d is 1), each
 * row stored with d zero columns on both sides and the channels of a pixel side by side. An output row
 * is computed once the last input row it reads has arrived (or lies below the input), and the layers
 * are drained deepest first, so a ring slot is overwritten only when no later output row reads it.
 * 1x1 layers keep no rows: they run per pixel on the vector the layer before them (or the input) has
 * just produced, and their result goes straight into the next 3x3 layer's ring or into the head's map
 * row. The arena therefore holds sum(R_l * (w_l + 2 d_l) * cin_l) ring bytes, one map row and the
 * decode below, for any height.
 *
 * Decode. In each map row the kernel cells (K > 0) form runs; a run joins the components of the runs
 * of the row above it overlaps (4-connectivity), by union-find over component slots, and adds its
 * cells' sums. After a row, a component that no run of the row continues is complete: its box is
 * computed (twin: decode) and kept in a list of at most 64 candidates sorted by (vertical, y, x, h, w).
 * Live components are bounded by the runs of two rows, which is at most the map's width: a slot is
 * needed per distinct component of the row above plus per run of this row that touches none, and their
 * first columns are distinct. After the last row: horizontal lines in rows, vertical lines right to
 * left, at most VOCR_LINES_MAX, then each line's dark flag (vocr_line_dark of its box).
 *
 * Integer ranges (with VOCR_TEXTDET_MAX_SIDE = 2048):
 *  - an accumulator sums at most 9 * 64 = 576 products of two int8 codes, |w x| <= 2^14, so
 *    |sum| <= 576 * 2^14 < 2^24 fits int32. The bias enters through bq = bias * mult + 2^(shift - 1)
 *    (int64, per channel): |bias * mult| <= 2^31 * (2^31 - 1) < 2^62, 2^(shift - 1) <= 2^61 and
 *    |sum * mult| < 2^24 * 2^31 = 2^55, so bq + sum * mult is the exact value and below 2^63.
 *  - the map has at most ceil(2048 / 4)^2 = 2^18 cells, so a component has n <= 2^18 cells. Across
 *    the line a cell's c2 = 2 s i + s - 1 <= 2 (2048 - 1) + s - 1 <= 4101 (s <= 8), and |O|, |A1|,
 *    |A2| <= 128, so every component sum (n; O; A1; A2; rows i or columns j, each below 512;
 *    2 s sum(i) + n (s - 1) -+ sum(A)) is at most 2^18 * (4101 + 128) = 1,108,606,976 < 2^31 in
 *    magnitude, and 2 n <= 2^19. A box side is at most 2048 + 2 * 1,172 (the pad), so the box sums fit.
 *
 * Memory: no allocation; one arena (vocr_textdet_arena_bytes) holds the per-layer state, the
 * per-channel requantisation constants, the rings, the map row, the decode's component slots and run
 * lists and the candidates. libc: memset and memcpy only (and libgcc's division helpers on 32-bit ARM
 * and MIPS, a few per row and per component).
 */
#include "vocr_textdet.h"

#include <stdint.h>
#include <string.h>

/* NEON kernels on the Vita (-mfpu=neon), the same arithmetic as the scalar ones; -DVOCR_NO_NEON forces
 * the scalar code (as for vocr.c). */
#if defined(__ARM_NEON) && !defined(VOCR_NO_NEON)
#include <arm_neon.h>
#define TD_NEON 1
#else
#define TD_NEON 0
#endif

#define HEADER 64u
#define RECORD 32u
#define LOADED 0x31445456u  /* "VTD1" */
#define CANDS 64u           /* candidates kept before the line cap */
#define PAD_PCT 18
#define PAD_MIN 2
#define MIN_ACROSS 4
#define MAXCH 64u           /* channels of a layer */
#define NONE 0xFFFFu        /* no component slot / a free slot */
#define BLK 8u              /* output pixels computed together (a block) */

/* ---------------------------------------------------------------- the blob */

static uint32_t rd16(const unsigned char *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8;
}

static uint32_t rd32(const unsigned char *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}

static int32_t rds32(const unsigned char *p)
{
    uint32_t v = rd32(p);
    return v < 0x80000000u ? (int32_t)v : -(int32_t)(0xFFFFFFFFu - v) - 1;
}

static uint32_t crc32_ieee(const unsigned char *p, size_t n)
{
    uint32_t c = 0xFFFFFFFFu;
    while (n--) {
        int b;
        c ^= *p++;
        for (b = 0; b < 8; b++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
    }
    return ~c;
}

/* [off, off + size) inside the data part of the blob, off 4-byte aligned (twin: Model). */
static int section_ok(uint32_t off, uint32_t size, uint32_t data, uint32_t total)
{
    return !(off % 4u) && off >= data && off <= total && size <= total - off;
}

/* Every check of the twin's Model; fills m (not yet marked loaded). */
static int parse(vocr_textdet *m, const unsigned char *b, size_t bytes)
{
    uint32_t total, n, data, cin = 1, stride = 1, len, i;
    if (bytes < HEADER || b[0] != 'V' || b[1] != 'T' || b[2] != 'D' || b[3] != 'T') return VOCR_TEXTDET_E_BLOB;
    total = rd32(b + 8);
    if (rd32(b + 4) != 1u || (size_t)total != bytes) return VOCR_TEXTDET_E_BLOB;
    if (crc32_ieee(b + 16, bytes - 16u) != rd32(b + 12)) return VOCR_TEXTDET_E_BLOB;
    n = rd16(b + 16);
    if ((b[18] != 4u && b[18] != 8u) || b[19] != 4u || b[20] != 2u || b[21] < 1u || b[21] > 4u)
        return VOCR_TEXTDET_E_BLOB;
    if (n < 2u || n > VOCR_TEXTDET_MAX_LAYERS || HEADER + n * RECORD > total) return VOCR_TEXTDET_E_BLOB;
    for (len = 32; len && !b[32 + len - 1u]; len--) {}   /* the name: ASCII once trailing NULs go */
    for (i = 0; i < len; i++)
        if (b[32 + i] >= 0x80u) return VOCR_TEXTDET_E_BLOB;
    data = HEADER + n * RECORD;
    for (i = 0; i < n; i++) {
        const unsigned char *r = b + HEADER + i * RECORD;
        vocr_textdet_layer *L = &m->layers[i];
        uint32_t kind = r[0], k = r[1], st = r[2], dil = r[3], relu = r[4];
        uint32_t lc = rd16(r + 8), cout = rd16(r + 10), nw, c;
        uint32_t wo = rd32(r + 12), bo = rd32(r + 16), mo = rd32(r + 20), so = rd32(r + 24);
        if (kind > 1u || (k != 1u && k != 3u) || (st != 1u && st != 2u) || dil < 1u || dil > 8u || relu > 1u ||
            lc != cin || cout < 1u || cout > MAXCH)
            return VOCR_TEXTDET_E_BLOB;
        if (kind == 1u && (cout != lc || k != 3u)) return VOCR_TEXTDET_E_BLOB;
        if (st == 2u && (k != 3u || dil != 1u)) return VOCR_TEXTDET_E_BLOB;
        nw = cout * (kind == 1u ? 1u : lc) * k * k;
        if (!section_ok(wo, nw, data, total) || !section_ok(bo, 4u * cout, data, total) ||
            !section_ok(mo, 4u * cout, data, total) || !section_ok(so, cout, data, total))
            return VOCR_TEXTDET_E_BLOB;
        for (c = 0; c < cout; c++)
            if (rds32(b + mo + 4u * c) <= 0 || b[so + c] < 1u || b[so + c] > 62u) return VOCR_TEXTDET_E_BLOB;
        L->w = b + wo;
        L->b = b + bo;
        L->m = b + mo;
        L->sh = b + so;
        L->cin = (unsigned short)lc;
        L->cout = (unsigned short)cout;
        L->kind = (unsigned char)kind;
        L->k = (unsigned char)k;
        L->stride = (unsigned char)st;
        L->dil = (unsigned char)dil;
        L->relu = (unsigned char)relu;
        m->channels += cout;
        stride *= st;
        cin = cout;
    }
    if (stride != b[18] || cin != 4u || m->layers[n - 1u].relu) return VOCR_TEXTDET_E_BLOB;
    m->n_layers = n;
    m->stride = b[18];
    m->min_cells = b[21];
    m->blob_bytes = (unsigned long)total;
    memcpy(m->ckpt, b + 24, 8);
    memcpy(m->name, b + 32, 32);
    m->name[32] = '\0';
    return VOCR_LINES_OK;
}

int vocr_textdet_load(vocr_textdet *m, const void *blob, size_t bytes)
{
    int rc;
    if (!m) return VOCR_LINES_E_ARG;
    memset(m, 0, sizeof(*m));
    if (!blob) return VOCR_LINES_E_ARG;
    rc = parse(m, (const unsigned char *)blob, bytes);
    if (rc) {
        memset(m, 0, sizeof(*m));
        return rc;
    }
    m->loaded = LOADED;
    return VOCR_LINES_OK;
}

/* ---------------------------------------------------------------- arena */

typedef struct {
    int64_t bq;   /* bias * multiplier + 2^(shift - 1) */
    int32_t m;    /* multiplier */
    int32_t sh;   /* shift */
} Chan;

typedef struct {
    uint32_t ring;       /* arena offset of a 3x3 layer's input ring */
    uint32_t plane;      /* bytes per channel plane of a ring row: w_in + 2 dil + (BLK - 1) stride */
    uint16_t w_in, h_in, w_out, h_out;
    uint16_t received;   /* input rows in the ring so far (3x3 layers) */
    uint16_t next_out;   /* next output row */
    uint16_t chan;       /* the layer's first channel in the Chan table */
    uint16_t next;       /* the next 3x3 layer, or n_layers: the head's map row */
} Run;

typedef struct {
    uint16_t a, b, comp; /* map columns [a, b) of a kernel run and its component slot */
} Seg;

typedef struct {
    int32_t n, so, sa1, sa2, sr, sc;  /* cells; sums of O, A1, A2, map rows, map columns */
    uint16_t r0, r1, c0, c1;          /* first and last row, first and last column */
    uint16_t parent;                  /* union-find parent; NONE when the slot is free */
    uint16_t stamp;                   /* map row + 1 of the last row whose runs joined it */
} Comp;

typedef struct {
    uint16_t v, y, x, h, w;           /* vertical flag and the clipped box */
} Cand;

typedef struct {
    size_t run, chan, rings, tmp, head, comp, freel, prev, cur, cand, total;
    unsigned wm, nc, nr;
} Plan;

static size_t up8(size_t n)
{
    return (n + 7u) & ~(size_t)7u;
}

static int loaded(const vocr_textdet *m)
{
    return m && m->loaded == LOADED && m->n_layers >= 2u && m->n_layers <= VOCR_TEXTDET_MAX_LAYERS;
}

/* Bytes per channel plane of a 3x3 layer's ring row: the d zero columns either side of the w input
 * pixels, then BLK * stride zero columns that a last, partial block of BLK output pixels reads past them
 * (BLK - 1 strides for the scalar code, one byte more for NEON's 16-byte de-interleaving load). */
static size_t plane_bytes(unsigned w, const vocr_textdet_layer *L)
{
    return (size_t)w + 2u * L->dil + BLK * L->stride;
}

/* The arena layout for regions of the given width (sections 8-byte aligned from an aligned base). */
static void plan(const vocr_textdet *m, unsigned width, Plan *p)
{
    unsigned i, w = width, maxch = 1;
    size_t rings = 0;
    for (i = 0; i < m->n_layers; i++) {
        const vocr_textdet_layer *L = &m->layers[i];
        if (L->k == 3u) rings += (size_t)(2u * L->dil + 1u) * L->cin * plane_bytes(w, L);
        if (L->cout > maxch) maxch = L->cout;
        w = (w + L->stride - 1u) / L->stride;
    }
    p->wm = w;
    p->nc = w;
    p->nr = (w + 1u) / 2u;
    p->run = 0;
    p->chan = p->run + up8((size_t)m->n_layers * sizeof(Run));
    p->rings = p->chan + up8((size_t)m->channels * sizeof(Chan));
    p->tmp = p->rings + up8(rings);
    p->head = p->tmp + up8((size_t)3u * BLK * maxch);   /* three blocks: a 3x3 output, two 1x1 ones */
    p->comp = p->head + up8((size_t)4u * w);
    p->freel = p->comp + up8((size_t)p->nc * sizeof(Comp));
    p->prev = p->freel + up8((size_t)p->nc * sizeof(uint16_t));
    p->cur = p->prev + up8((size_t)p->nr * sizeof(Seg));
    p->cand = p->cur + up8((size_t)p->nr * sizeof(Seg));
    p->total = p->cand + up8((size_t)CANDS * sizeof(Cand));
}

size_t vocr_textdet_arena_bytes(const vocr_textdet *m, unsigned max_width, unsigned max_height)
{
    Plan p;
    if (!loaded(m) || !max_width || !max_height || max_width > VOCR_TEXTDET_MAX_SIDE ||
        max_height > VOCR_TEXTDET_MAX_SIDE)
        return 0;
    plan(m, max_width, &p);
    return p.total + 7u;
}

/* ---------------------------------------------------------------- network */

typedef struct {
    const VocrLinesImage *im;   /* an image in memory, or */
    const VocrLinesRows *rows;  /* a row source */
} Src;

typedef struct {
    const vocr_textdet *m;
    unsigned char *base;        /* the aligned arena */
    Run *run;
    Chan *chan;
    int8_t *head;               /* the head's map row: 4 codes per cell */
    int8_t *tmp;                /* three blocks of BLK x maxch codes: a 3x3 output, two 1x1 ones */
    unsigned maxch;             /* the most output channels of a layer */
    unsigned rl[VOCR_TEXTDET_MAX_LAYERS], nrl;   /* the 3x3 layers, in order */
    unsigned width, height, wm, hm;
    vocr_textdet_map_row hook;
    void *user;
    /* decode */
    Comp *comp;
    uint16_t *freel;
    unsigned nfree, nc;
    Seg *prev, *cur;
    unsigned np;
    Cand *cand;
    unsigned ncand, ctrunc;
} Net;

static const unsigned char *src_row(const Src *s, unsigned y)
{
    if (s->im) return s->im->pixels + (size_t)y * s->im->stride;
    return s->rows->row(s->rows->user, y);
}

#if !TD_NEON
/* floor(v / 2^s) without shifting a negative value */
static int64_t shr64(int64_t v, unsigned s)
{
    return v < 0 ? ~(~v >> s) : v >> s;
}

static int32_t shr32(int32_t v, unsigned s)
{
    return v < 0 ? ~(~v >> s) : v >> s;
}

static int8_t requant(int32_t acc, const Chan *c, int32_t lo)
{
    int64_t p = c->bq + (int64_t)acc * c->m;
    int32_t q;
    if (c->sh >= 32) {
        q = shr32((int32_t)shr64(p, 32u), (unsigned)(c->sh - 32));   /* floor(floor(p / 2^32) / 2^(sh - 32)) */
    } else {
        int64_t r = shr64(p, (unsigned)c->sh);
        q = r < lo ? lo : r > 127 ? 127 : (int32_t)r;
    }
    return (int8_t)(q < lo ? lo : q > 127 ? 127 : q);
}
#endif

#if TD_NEON
/* ---- NEON (the Vita): a block's 8 int32 sums in two registers; the arithmetic of requant() */

/* floor((bq + acc m) / 2^sh) per lane in int64 (vshl by -sh: an arithmetic shift), saturated to int8,
 * then at least lo; n lanes stored at out. */
#define RQ_STORE(LO, HI, C, LOW, OUT, N) do { \
        const Chan *c_ = (C); \
        int64x2_t b_ = vdupq_n_s64(c_->bq), s_ = vdupq_n_s64(-(int64_t)c_->sh); \
        int32x2_t m_ = vdup_n_s32(c_->m); \
        int64x2_t p0_ = vshlq_s64(vmlal_s32(b_, vget_low_s32(LO), m_), s_); \
        int64x2_t p1_ = vshlq_s64(vmlal_s32(b_, vget_high_s32(LO), m_), s_); \
        int64x2_t p2_ = vshlq_s64(vmlal_s32(b_, vget_low_s32(HI), m_), s_); \
        int64x2_t p3_ = vshlq_s64(vmlal_s32(b_, vget_high_s32(HI), m_), s_); \
        int16x4_t h0_ = vqmovn_s32(vcombine_s32(vqmovn_s64(p0_), vqmovn_s64(p1_))); \
        int16x4_t h1_ = vqmovn_s32(vcombine_s32(vqmovn_s64(p2_), vqmovn_s64(p3_))); \
        int8x8_t v_ = vmax_s8(vqmovn_s16(vcombine_s16(h0_, h1_)), vdup_n_s8((int8_t)(LOW))); \
        if ((N) == BLK) { \
            vst1_s8((OUT), v_); \
        } else { \
            int8_t t_[BLK]; \
            unsigned i_; \
            vst1_s8(t_, v_); \
            for (i_ = 0; i_ < (N); i_++) (OUT)[i_] = t_[i_]; \
        } \
    } while (0)

/* 8 pixels of one input channel at stride S, widened (vld2 reads 16 bytes: the plane's slack covers it) */
#define LOAD8(x, S) vmovl_s8((S) == 1u ? vld1_s8(x) : vld2_s8(x).val[0])
/* LO/HI += X * lane L of the int16x4 weights W, or * the scalar K */
#define MACL(LO, HI, X, W, L) do { \
        int16x8_t x_ = (X); \
        (LO) = vmlal_lane_s16((LO), vget_low_s16(x_), (W), L); \
        (HI) = vmlal_lane_s16((HI), vget_high_s16(x_), (W), L); \
    } while (0)
#define MACN(LO, HI, X, K) do { \
        int16x8_t x_ = (X); \
        (LO) = vmlal_n_s16((LO), vget_low_s16(x_), (int16_t)(K)); \
        (HI) = vmlal_n_s16((HI), vget_high_s16(x_), (int16_t)(K)); \
    } while (0)

/* A block of BLK output pixels from pixel j0 of a 3x3 layer with input stride S, output channel by output
 * channel into out[c * ostride + p] (p < n). rows[ky] is the padded input row of kernel row ky (the centre
 * row where valid[ky] is 0, its weights then masked to 0), plane bytes from one input channel to the next.
 * Depthwise: output channel c reads input channel c; otherwise every input channel (cin of them). */
#define K3_BLOCK(name, S) \
static void name(const int8_t *w, const Chan *ch, int32_t lo, unsigned cin, unsigned cout, int depthwise, \
                 const int8_t *const *rows, const int *valid, size_t plane, unsigned d, unsigned j0, int8_t *out, \
                 size_t ostride, unsigned n) \
{ \
    static const int16_t top_[8] = {0, 0, 0, -1, -1, -1, -1, -1}, mid_[8] = {-1, -1, -1, -1, -1, -1, 0, 0}; \
    int16x8_t keep = vdupq_n_s16(-1); \
    int last = valid[2]; \
    unsigned c, q; \
    const int8_t *r0 = rows[0] + (size_t)j0 * (S), *r1 = rows[1] + (size_t)j0 * (S); \
    const int8_t *r2 = rows[2] + (size_t)j0 * (S); \
    if (!valid[0]) keep = vandq_s16(keep, vld1q_s16(top_)); \
    if (!valid[2]) keep = vandq_s16(keep, vld1q_s16(mid_)); \
    for (c = 0; c < cout; c++) { \
        int32x4_t lo4 = vdupq_n_s32(0), hi4 = lo4; \
        unsigned q0 = depthwise ? c : 0, q1 = depthwise ? c + 1u : cin; \
        for (q = q0; q < q1; q++) { \
            const int8_t *wq = w + 9u * (depthwise ? c : c * cin + q); \
            int16x8_t wv = vandq_s16(vmovl_s8(vld1_s8(wq)), keep); \
            int16x4_t wa = vget_low_s16(wv), wb = vget_high_s16(wv); \
            const int8_t *x0 = r0 + q * plane, *x1 = r1 + q * plane, *x2 = r2 + q * plane; \
            MACL(lo4, hi4, LOAD8(x0, S), wa, 0); \
            MACL(lo4, hi4, LOAD8(x0 + d, S), wa, 1); \
            MACL(lo4, hi4, LOAD8(x0 + 2u * d, S), wa, 2); \
            MACL(lo4, hi4, LOAD8(x1, S), wa, 3); \
            MACL(lo4, hi4, LOAD8(x1 + d, S), wb, 0); \
            MACL(lo4, hi4, LOAD8(x1 + 2u * d, S), wb, 1); \
            MACL(lo4, hi4, LOAD8(x2, S), wb, 2); \
            MACL(lo4, hi4, LOAD8(x2 + d, S), wb, 3); \
            if (last) MACN(lo4, hi4, LOAD8(x2 + 2u * d, S), wq[8]); \
        } \
        RQ_STORE(lo4, hi4, &ch[c], lo, out + c * ostride, n); \
    } \
}
K3_BLOCK(k3_block_s1, 1u)
K3_BLOCK(k3_block_s2, 2u)

/* A block of a 1x1 layer: in[q * BLK + p] -> out[c * ostride + p] (p < n). */
static void k1_block(const int8_t *w, const Chan *ch, int32_t lo, unsigned cin, unsigned cout, const int8_t *in,
                     int8_t *out, size_t ostride, unsigned n)
{
    unsigned c, q;
    for (c = 0; c < cout; c++) {
        int32x4_t lo4 = vdupq_n_s32(0), hi4 = lo4;
        const int8_t *wc = w + (size_t)c * cin;
        for (q = 0; q + 8u <= cin; q += 8u) {
            int16x8_t wv = vmovl_s8(vld1_s8(wc + q));
            int16x4_t wa = vget_low_s16(wv), wb = vget_high_s16(wv);
            const int8_t *x = in + q * BLK;
            MACL(lo4, hi4, vmovl_s8(vld1_s8(x)), wa, 0);
            MACL(lo4, hi4, vmovl_s8(vld1_s8(x + BLK)), wa, 1);
            MACL(lo4, hi4, vmovl_s8(vld1_s8(x + 2u * BLK)), wa, 2);
            MACL(lo4, hi4, vmovl_s8(vld1_s8(x + 3u * BLK)), wa, 3);
            MACL(lo4, hi4, vmovl_s8(vld1_s8(x + 4u * BLK)), wb, 0);
            MACL(lo4, hi4, vmovl_s8(vld1_s8(x + 5u * BLK)), wb, 1);
            MACL(lo4, hi4, vmovl_s8(vld1_s8(x + 6u * BLK)), wb, 2);
            MACL(lo4, hi4, vmovl_s8(vld1_s8(x + 7u * BLK)), wb, 3);
        }
        for (; q < cin; q++) MACN(lo4, hi4, vmovl_s8(vld1_s8(in + q * BLK)), wc[q]);
        RQ_STORE(lo4, hi4, &ch[c], lo, out + c * ostride, n);
    }
}
#else
/* ---- scalar (the PSP, hosts): eight named accumulators, so they stay in registers */

typedef struct {
    int32_t v[BLK];
} Acc;

static Acc acc_zero(void)
{
    Acc a = {{0, 0, 0, 0, 0, 0, 0, 0}};
    return a;
}

#define MAC(a, k, x, S) do { \
        const int8_t *x_ = (x); \
        int32_t k_ = (k); \
        (a).v[0] += k_ * x_[0]; (a).v[1] += k_ * x_[1 * (S)]; (a).v[2] += k_ * x_[2 * (S)]; \
        (a).v[3] += k_ * x_[3 * (S)]; (a).v[4] += k_ * x_[4 * (S)]; (a).v[5] += k_ * x_[5 * (S)]; \
        (a).v[6] += k_ * x_[6 * (S)]; (a).v[7] += k_ * x_[7 * (S)]; \
    } while (0)

static void store_acc(Acc a, const Chan *c, int32_t lo, int8_t *out, unsigned n)
{
    unsigned p;
    if (c->sh >= 32) {                        /* the usual case: the high word, then a 32-bit shift */
        int64_t bq = c->bq;
        int32_t m = c->m;
        unsigned s = (unsigned)(c->sh - 32);
        for (p = 0; p < n; p++) {
            int32_t q = shr32((int32_t)shr64(bq + (int64_t)a.v[p] * m, 32u), s);
            out[p] = (int8_t)(q < lo ? lo : q > 127 ? 127 : q);
        }
    } else {
        for (p = 0; p < n; p++) out[p] = requant(a.v[p], c, lo);
    }
}

/* A block of BLK output pixels from pixel j0 of a 3x3 layer with input stride S, output channel by output
 * channel into out[c * ostride + p] (p < n). rows[ky] is the padded input row of kernel row ky (valid[ky]
 * when it lies inside the input), plane bytes from one input channel to the next. Depthwise: output
 * channel c reads input channel c; otherwise every input channel (cin of them). */
#define K3_BLOCK(name, S) \
static void name(const int8_t *w, const Chan *ch, int32_t lo, unsigned cin, unsigned cout, int depthwise, \
                 const int8_t *const *rows, const int *valid, size_t plane, unsigned d, unsigned j0, int8_t *out, \
                 size_t ostride, unsigned n) \
{ \
    unsigned c, q, ky; \
    for (c = 0; c < cout; c++) { \
        Acc a = acc_zero(); \
        unsigned q0 = depthwise ? c : 0, q1 = depthwise ? c + 1u : cin; \
        for (q = q0; q < q1; q++) { \
            const int8_t *wq = w + 9u * (depthwise ? c : c * cin + q); \
            for (ky = 0; ky < 3u; ky++) { \
                const int8_t *x = rows[ky] + q * plane + (size_t)j0 * (S), *wk = wq + 3u * ky; \
                if (!valid[ky]) continue; \
                MAC(a, wk[0], x, S); \
                MAC(a, wk[1], x + d, S); \
                MAC(a, wk[2], x + 2u * d, S); \
            } \
        } \
        store_acc(a, &ch[c], lo, out + c * ostride, n); \
    } \
}
K3_BLOCK(k3_block_s1, 1u)
K3_BLOCK(k3_block_s2, 2u)

/* A block of a 1x1 layer: in[q * BLK + p] -> out[c * ostride + p] (p < n). */
static void k1_block(const int8_t *w, const Chan *ch, int32_t lo, unsigned cin, unsigned cout, const int8_t *in,
                     int8_t *out, size_t ostride, unsigned n)
{
    unsigned c, q;
    for (c = 0; c < cout; c++) {
        Acc a = acc_zero();
        const int8_t *wc = w + (size_t)c * cin;
        for (q = 0; q < cin; q++) MAC(a, wc[q], in + q * BLK, 1u);
        store_acc(a, &ch[c], lo, out + c * ostride, n);
    }
}
#endif

static const int8_t *weights(const vocr_textdet_layer *L)
{
    return (const int8_t *)(const void *)L->w;
}

/* The 1x1 layers [from, to) on a block in[q * BLK + p]; the last writes n pixels to dst (plane dstride
 * bytes apart); a and b are BLK * maxch scratch blocks. */
static void chain(const Net *t, unsigned from, unsigned to, const int8_t *in, int8_t *a, int8_t *b, int8_t *dst,
                  size_t dstride, unsigned n)
{
    unsigned l;
    for (l = from; l < to; l++) {
        const vocr_textdet_layer *L = &t->m->layers[l];
        int last = l + 1u == to;
        int8_t *o = last ? dst : a;
        k1_block(weights(L), t->chan + t->run[l].chan, L->relu ? 0 : -128, L->cin, L->cout, in, o,
                 last ? dstride : BLK, last ? n : BLK);
        in = o;
        a = b;
        b = o;
    }
}

static unsigned min_u(unsigned a, unsigned b)
{
    return a < b ? a : b;
}

static unsigned max_u(unsigned a, unsigned b)
{
    return a > b ? a : b;
}

/* ---------------------------------------------------------------- decode */

static unsigned root_of(Comp *c, unsigned x)
{
    while (c[x].parent != x) {
        c[x].parent = c[c[x].parent].parent;
        x = c[x].parent;
    }
    return x;
}

static void unite(Comp *c, unsigned a, unsigned b)   /* b joins a */
{
    Comp *p = &c[a], *q = &c[b];
    p->n += q->n;
    p->so += q->so;
    p->sa1 += q->sa1;
    p->sa2 += q->sa2;
    p->sr += q->sr;
    p->sc += q->sc;
    if (q->r0 < p->r0) p->r0 = q->r0;
    if (q->r1 > p->r1) p->r1 = q->r1;
    if (q->c0 < p->c0) p->c0 = q->c0;
    if (q->c1 > p->c1) p->c1 = q->c1;
    q->parent = (uint16_t)a;
}

static int32_t floor_div(int32_t a, int32_t b)   /* b > 0 */
{
    int32_t q = a / b;
    return (a % b != 0 && a < 0) ? q - 1 : q;
}

static int32_t ceil_div(int32_t a, int32_t b)    /* b > 0 */
{
    int32_t q = a / b;
    return (a % b != 0 && a > 0) ? q + 1 : q;
}

static int cand_less(const Cand *a, const Cand *b)   /* (vertical, y, x, h, w) */
{
    if (a->v != b->v) return a->v < b->v;
    if (a->y != b->y) return a->y < b->y;
    if (a->x != b->x) return a->x < b->x;
    if (a->h != b->h) return a->h < b->h;
    return a->w < b->w;
}

static int row_less(const Cand *a, const Cand *b)    /* a row of horizontal lines: (x, y, h, w) */
{
    if (a->x != b->x) return a->x < b->x;
    if (a->y != b->y) return a->y < b->y;
    if (a->h != b->h) return a->h < b->h;
    return a->w < b->w;
}

static int vert_less(const Cand *a, const Cand *b)   /* vertical lines: (-(x + w), y, x, w, h) */
{
    unsigned ra = (unsigned)a->x + a->w, rb = (unsigned)b->x + b->w;
    if (ra != rb) return ra > rb;
    if (a->y != b->y) return a->y < b->y;
    if (a->x != b->x) return a->x < b->x;
    if (a->w != b->w) return a->w < b->w;
    return a->h < b->h;
}

static void sort_cands(Cand *c, unsigned n, int (*less)(const Cand *, const Cand *))
{
    unsigned i, j;
    for (i = 1; i < n; i++) {
        Cand v = c[i];
        for (j = i; j > 0 && less(&v, &c[j - 1u]); j--) c[j] = c[j - 1u];
        c[j] = v;
    }
}

/* A candidate into the sorted list; the 64 smallest are kept. */
static void keep(Net *t, const Cand *v)
{
    unsigned j;
    if (t->ncand == CANDS) {
        t->ctrunc = 1;
        if (!cand_less(v, &t->cand[CANDS - 1u])) return;
        t->ncand--;
    }
    for (j = t->ncand; j > 0 && cand_less(v, &t->cand[j - 1u]); j--) t->cand[j] = t->cand[j - 1u];
    t->cand[j] = *v;
    t->ncand++;
}

/* A complete component: its box (twin: decode), if it has one. */
static void finish(Net *t, const Comp *c)
{
    int32_t s = (int32_t)t->m->stride, n = c->n, base, near_, far_, across, pad, lo, hi;
    int32_t x0, x1, y0, y1;
    int vertical;
    Cand v;
    if (n < (int32_t)t->m->min_cells) return;
    vertical = c->so > 0;
    base = 2 * s * (vertical ? c->sc : c->sr) + n * (s - 1);   /* sum of c2 = 2 s i + s - 1 */
    near_ = floor_div(base - c->sa1, 2 * n);
    far_ = ceil_div(base + c->sa2, 2 * n);
    across = far_ - near_;
    if (across < MIN_ACROSS) return;
    pad = (across * PAD_PCT + 50) / 100;
    if (pad < PAD_MIN) pad = PAD_MIN;
    if (vertical) {
        lo = (int32_t)c->r0 * s;
        hi = ((int32_t)c->r1 + 1) * s;
        x0 = near_ - pad;
        x1 = far_ + pad;
        y0 = lo - pad;
        y1 = hi + pad;
    } else {
        lo = (int32_t)c->c0 * s;
        hi = ((int32_t)c->c1 + 1) * s;
        y0 = near_ - pad;
        y1 = far_ + pad;
        x0 = lo - pad;
        x1 = hi + pad;
    }
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > (int32_t)t->width) x1 = (int32_t)t->width;
    if (y1 > (int32_t)t->height) y1 = (int32_t)t->height;
    if (x1 <= x0 || y1 <= y0) return;
    v.v = (uint16_t)vertical;
    v.y = (uint16_t)y0;
    v.x = (uint16_t)x0;
    v.h = (uint16_t)(y1 - y0);
    v.w = (uint16_t)(x1 - x0);
    keep(t, &v);
}

/* Frees every slot but the live roots (stamp); a root no run of the row continued is complete. */
static void sweep(Net *t, unsigned stamp)
{
    unsigned s;
    for (s = 0; s < t->nc; s++) {
        Comp *c = &t->comp[s];
        if (c->parent == NONE) continue;
        if (c->parent == s && c->stamp == stamp) continue;
        if (c->parent == s) finish(t, c);
        c->parent = NONE;
        t->freel[t->nfree++] = (uint16_t)s;
    }
}

/* Map row i (in t->head: K, O, A1, A2 planes of wm cells) into the streaming components. */
static void decode_row(Net *t, unsigned i)
{
    const int8_t *kk = t->head, *oo = kk + t->wm, *a1 = oo + t->wm, *a2 = a1 + t->wm;
    Comp *cp = t->comp;
    Seg *prev = t->prev, *cur = t->cur;
    unsigned wm = t->wm, j = 0, k = 0, q, nc = 0, np = t->np;
    while (j < wm) {
        unsigned a, root = NONE;
        int32_t n = 0, so = 0, sa1 = 0, sa2 = 0, sc = 0;
        Comp *c;
        if (kk[j] <= 0) {
            j++;
            continue;
        }
        a = j;
        for (; j < wm && kk[j] > 0; j++) {
            so += oo[j];
            sa1 += a1[j];
            sa2 += a2[j];
            sc += (int32_t)j;
            n++;
        }
        while (k < np && prev[k].b <= a) k++;
        for (q = k; q < np && prev[q].a < j; q++) {   /* the runs above that this one touches */
            unsigned r = root_of(cp, prev[q].comp);
            if (root == NONE) root = r;
            else if (r != root) unite(cp, root, r);
        }
        if (root == NONE) {
            root = t->freel[--t->nfree];
            c = &cp[root];
            c->n = c->so = c->sa1 = c->sa2 = c->sr = c->sc = 0;
            c->r0 = (uint16_t)i;
            c->c0 = (uint16_t)a;
            c->c1 = (uint16_t)(j - 1u);
            c->parent = (uint16_t)root;
            c->stamp = 0;
        }
        c = &cp[root];
        c->n += n;
        c->so += so;
        c->sa1 += sa1;
        c->sa2 += sa2;
        c->sr += (int32_t)i * n;
        c->sc += sc;
        c->r1 = (uint16_t)i;
        if (a < c->c0) c->c0 = (uint16_t)a;
        if (j - 1u > c->c1) c->c1 = (uint16_t)(j - 1u);
        cur[nc].a = (uint16_t)a;
        cur[nc].b = (uint16_t)j;
        cur[nc].comp = (uint16_t)root;
        nc++;
    }
    for (q = 0; q < nc; q++) {
        unsigned r = root_of(cp, cur[q].comp);
        cur[q].comp = (uint16_t)r;
        cp[r].stamp = (uint16_t)(i + 1u);
    }
    sweep(t, i + 1u);
    t->prev = cur;
    t->cur = prev;
    t->np = nc;
}

/* ---------------------------------------------------------------- the row pipeline */

static int8_t *ring_row(const Net *t, unsigned l, unsigned y)
{
    const vocr_textdet_layer *L = &t->m->layers[l];
    const Run *r = &t->run[l];
    return (int8_t *)(void *)(t->base + r->ring + (size_t)(y % (2u * L->dil + 1u)) * L->cin * r->plane);
}

/* Where the segment ending before layer `next` writes pixel 0 of channel 0 of its row (channel c at
 * + c * dest_stride): the next 3x3 layer's newest ring row, after its left padding, or the map row. */
static int8_t *dest_row(const Net *t, unsigned next)
{
    if (next < t->m->n_layers) return ring_row(t, next, t->run[next].received) + t->m->layers[next].dil;
    return t->head;
}

static size_t dest_stride(const Net *t, unsigned next)
{
    return next < t->m->n_layers ? t->run[next].plane : t->wm;
}

/* The row has gone into layer `next` (a ring) or is the map row i. */
static void delivered(Net *t, unsigned next, unsigned i)
{
    if (next < t->m->n_layers) {
        t->run[next].received++;
        return;
    }
    if (t->hook) t->hook(t->user, i, (const signed char *)(const void *)t->head, t->wm);
    decode_row(t, i);
}

/* Output row next_out of the 3x3 layer l, through the 1x1 layers after it, BLK pixels at a time. */
static void produce(Net *t, unsigned l)
{
    const vocr_textdet_layer *L = &t->m->layers[l];
    Run *r = &t->run[l];
    unsigned s = L->stride, d = L->dil, nx = r->next, i = r->next_out, ky, j0;
    const int8_t *rows[3];
    int valid[3];
    const Chan *ch = t->chan + r->chan;
    int32_t lo = L->relu ? 0 : -128;
    int8_t *dst = dest_row(t, nx), *blk = t->tmp;
    size_t ds = dest_stride(t, nx), bs = (size_t)BLK * t->maxch;
    for (ky = 0; ky < 3u; ky++) {             /* input row i s + (ky - 1) d; the centre one always exists */
        unsigned y = i * s + ky * d;
        valid[ky] = y >= d && y - d < r->h_in;
        rows[ky] = ring_row(t, l, valid[ky] ? y - d : i * s);
    }
    for (j0 = 0; j0 < r->w_out; j0 += BLK) {
        unsigned n = min_u(BLK, r->w_out - j0);
        int8_t *o = nx == l + 1u ? dst + j0 : blk;
        if (s == 1u)
            k3_block_s1(weights(L), ch, lo, L->cin, L->cout, L->kind == 1u, rows, valid, r->plane, d, j0, o,
                        nx == l + 1u ? ds : BLK, nx == l + 1u ? n : BLK);
        else
            k3_block_s2(weights(L), ch, lo, L->cin, L->cout, L->kind == 1u, rows, valid, r->plane, d, j0, o,
                        nx == l + 1u ? ds : BLK, nx == l + 1u ? n : BLK);
        if (nx != l + 1u) chain(t, l + 1u, nx, blk, blk + bs, blk + 2u * bs, dst + j0, ds, n);
    }
    r->next_out++;
    delivered(t, nx, i);
}

static int producible(const Net *t, unsigned l)
{
    const Run *r = &t->run[l];
    unsigned need = (unsigned)r->next_out * t->m->layers[l].stride + t->m->layers[l].dil + 1u;
    if (r->next_out >= r->h_out) return 0;
    return r->received >= min_u(need, r->h_in);
}

/* Computes every row that can be computed, deepest layer first. */
static void drain(Net *t)
{
    int p = 0;
    while (p >= 0) {
        unsigned l = t->rl[p];
        if (producible(t, l)) {
            produce(t, l);
            if ((unsigned)p + 1u < t->nrl) p++;
        } else {
            p--;
        }
    }
}

static unsigned grey_px(const unsigned char *row, unsigned x, unsigned bpp)
{
    const unsigned char *p = row + (size_t)x * bpp;
    if (bpp == VOCR_LINES_GREY) return p[0];
    return (77u * p[0] + 150u * p[1] + 29u * p[2]) >> 8;
}

/* Region row y into the first 3x3 layer, through the 1x1 layers before it. */
static int feed(Net *t, const Src *s, unsigned y, unsigned bpp)
{
    const unsigned char *row = src_row(s, y);
    unsigned first = t->rl[0], x, p;
    int8_t *dst = dest_row(t, first);
    if (!row) return VOCR_LINES_E_SOURCE;
    if (!first) {
        for (x = 0; x < t->width; x++) dst[x] = (int8_t)((int)grey_px(row, x, bpp) - 128);
    } else {
        size_t bs = (size_t)BLK * t->maxch;
        for (x = 0; x < t->width; x += BLK) {
            unsigned n = min_u(BLK, t->width - x);
            for (p = 0; p < BLK; p++) t->tmp[p] = (int8_t)(p < n ? (int)grey_px(row, x + p, bpp) - 128 : 0);
            chain(t, 0, first, t->tmp, t->tmp + bs, t->tmp + 2u * bs, dst + x, t->run[first].plane, n);
        }
    }
    delivered(t, first, y);
    drain(t);
    return VOCR_LINES_OK;
}

/* The per-layer state and the requantisation constants of one call. */
static void setup(Net *t, const Plan *p)
{
    const vocr_textdet *m = t->m;
    unsigned i, c, ci = 0, w = t->width, h = t->height, next = m->n_layers;
    uint32_t roff = (uint32_t)p->rings;
    t->run = (Run *)(void *)(t->base + p->run);
    t->chan = (Chan *)(void *)(t->base + p->chan);
    t->head = (int8_t *)(void *)(t->base + p->head);
    t->tmp = (int8_t *)(void *)(t->base + p->tmp);
    t->comp = (Comp *)(void *)(t->base + p->comp);
    t->freel = (uint16_t *)(void *)(t->base + p->freel);
    t->prev = (Seg *)(void *)(t->base + p->prev);
    t->cur = (Seg *)(void *)(t->base + p->cur);
    t->cand = (Cand *)(void *)(t->base + p->cand);
    t->nrl = 0;
    t->maxch = 1;
    for (i = 0; i < m->n_layers; i++) {
        const vocr_textdet_layer *L = &m->layers[i];
        Run *r = &t->run[i];
        r->w_in = (uint16_t)w;
        r->h_in = (uint16_t)h;
        w = (w + L->stride - 1u) / L->stride;
        h = (h + L->stride - 1u) / L->stride;
        r->w_out = (uint16_t)w;
        r->h_out = (uint16_t)h;
        r->received = r->next_out = 0;
        r->chan = (uint16_t)ci;
        for (c = 0; c < L->cout; c++, ci++) {
            Chan *q = &t->chan[ci];
            int32_t sh = L->sh[c];
            q->m = rds32(L->m + 4u * c);
            q->sh = sh;
            q->bq = (int64_t)rds32(L->b + 4u * c) * q->m + ((int64_t)1 << (sh - 1));
        }
        r->ring = r->plane = 0;
        if (L->cout > t->maxch) t->maxch = L->cout;
        if (L->k == 3u) {
            size_t bytes;
            r->ring = roff;
            r->plane = (uint32_t)plane_bytes(r->w_in, L);
            bytes = (size_t)(2u * L->dil + 1u) * L->cin * r->plane;
            memset(t->base + roff, 0, bytes);   /* the padding and slack columns stay zero */
            roff += (uint32_t)bytes;
            t->rl[t->nrl++] = i;
        }
    }
    for (i = m->n_layers; i-- > 0;) {
        t->run[i].next = (uint16_t)next;
        if (m->layers[i].k == 3u) next = i;
    }
    t->wm = w;
    t->hm = h;
    t->nc = p->nc;
    for (i = 0; i < p->nc; i++) {
        t->comp[i].parent = NONE;
        t->freel[i] = (uint16_t)(p->nc - 1u - i);
    }
    t->nfree = p->nc;
    t->np = 0;
    t->ncand = 0;
    t->ctrunc = 0;
}

/* Where the lines go: a VocrTextdetSet's arrays, or (vertical NULL: the finder mode) a VocrLineSet's
 * lines, which receives the horizontal ones only. */
typedef struct {
    VocrLine *lines;
    unsigned *vertical;
    unsigned count, truncated;
} Out;

/* Horizontal lines in rows (a line joins a row when it overlaps the row's first line by half the
 * smaller height; a row goes left to right), then vertical lines right to left; at most
 * VOCR_LINES_MAX of them in all. Without out->vertical only the horizontal ones among those. */
static void order(Net *t, Out *out)
{
    Cand *c = t->cand;
    unsigned n = t->ncand, nh = 0, i, j, k = 0;
    while (nh < n && !c[nh].v) nh++;
    for (i = 0; i < nh; i = j) {
        unsigned fy = c[i].y, fh = c[i].h;
        for (j = i + 1u; j < nh; j++) {
            int ov = (int)min_u((unsigned)c[j].y + c[j].h, fy + fh) - (int)max_u(c[j].y, fy);
            if (2 * ov < (int)min_u(c[j].h, fh)) break;
        }
        sort_cands(c + i, j - i, row_less);
    }
    if (out->vertical) sort_cands(c + nh, n - nh, vert_less);
    for (i = 0; i < (out->vertical ? n : nh) && k < VOCR_LINES_MAX; i++, k++) {
        out->lines[k].x = c[i].x;
        out->lines[k].y = c[i].y;
        out->lines[k].w = c[i].w;
        out->lines[k].h = c[i].h;
        out->lines[k].dark = 0;
        if (out->vertical) out->vertical[k] = c[i].v;
    }
    out->count = k;
    out->truncated = (t->ctrunc || n > VOCR_LINES_MAX) ? 1u : 0u;
}

static int detect(const vocr_textdet *m, const Src *s, unsigned width, unsigned height, unsigned bpp, void *arena,
                  size_t arena_bytes, Out *out, vocr_textdet_map_row hook, void *user)
{
    Plan p;
    Net t;
    unsigned y, k;
    uintptr_t a;
    int rc;
    out->count = out->truncated = 0;
    if (!loaded(m) || !width || !height || width > VOCR_TEXTDET_MAX_SIDE || height > VOCR_TEXTDET_MAX_SIDE ||
        (bpp != VOCR_LINES_GREY && bpp != VOCR_LINES_RGBA) || !arena)
        return VOCR_LINES_E_ARG;
    plan(m, width, &p);
    if (arena_bytes < p.total + 7u) return VOCR_LINES_E_ARENA;
    a = (uintptr_t)arena;
    t.m = m;
    t.base = (unsigned char *)arena + ((8u - (unsigned)(a & 7u)) & 7u);
    t.width = width;
    t.height = height;
    t.hook = hook;
    t.user = user;
    setup(&t, &p);
    for (y = 0; y < height; y++) {
        rc = feed(&t, s, y, bpp);
        if (rc) return rc;
    }
    sweep(&t, NONE);   /* the components still open after the last map row */
    order(&t, out);
    for (k = 0; k < out->count; k++) {
        VocrLine *ln = &out->lines[k];
        int d = s->im ? vocr_line_dark(s->im, ln->x, ln->y, ln->w, ln->h)
                      : vocr_line_dark_rows(s->rows, ln->x, ln->y, ln->w, ln->h);
        if (d < 0) {
            out->count = out->truncated = 0;
            return d;
        }
        ln->dark = (unsigned)d;
    }
    return VOCR_LINES_OK;
}

static int image_ok(const VocrLinesImage *im)
{
    return im && im->pixels && (im->bpp == VOCR_LINES_GREY || im->bpp == VOCR_LINES_RGBA) &&
           im->stride >= (size_t)im->width * im->bpp;
}

/* The detector into a set (vertical flags) or a finder-mode set (vertical NULL). */
static int run_image(const vocr_textdet *m, const VocrLinesImage *im, void *arena, size_t arena_bytes, VocrLine *lines,
                     unsigned *vertical, unsigned *count, unsigned *truncated)
{
    Src s;
    Out o;
    int rc = VOCR_LINES_E_ARG;
    o.lines = lines;
    o.vertical = vertical;
    o.count = o.truncated = 0;
    if (image_ok(im)) {
        s.im = im;
        s.rows = NULL;
        rc = detect(m, &s, im->width, im->height, im->bpp, arena, arena_bytes, &o, NULL, NULL);
    }
    *count = o.count;
    *truncated = o.truncated;
    return rc;
}

static int run_rows(const vocr_textdet *m, const VocrLinesRows *src, void *arena, size_t arena_bytes, VocrLine *lines,
                    unsigned *vertical, unsigned *count, unsigned *truncated)
{
    Src s;
    Out o;
    int rc = VOCR_LINES_E_ARG;
    o.lines = lines;
    o.vertical = vertical;
    o.count = o.truncated = 0;
    if (src && src->row) {
        s.im = NULL;
        s.rows = src;
        rc = detect(m, &s, src->width, src->height, src->bpp, arena, arena_bytes, &o, NULL, NULL);
    }
    *count = o.count;
    *truncated = o.truncated;
    return rc;
}

int vocr_textdet_find(const vocr_textdet *m, const VocrLinesImage *im, void *arena, size_t arena_bytes,
                      VocrTextdetSet *out)
{
    if (!out) return VOCR_LINES_E_ARG;
    return run_image(m, im, arena, arena_bytes, out->lines, out->vertical, &out->count, &out->truncated);
}

int vocr_textdet_find_rows(const vocr_textdet *m, const VocrLinesRows *src, void *arena, size_t arena_bytes,
                           VocrTextdetSet *out)
{
    if (!out) return VOCR_LINES_E_ARG;
    return run_rows(m, src, arena, arena_bytes, out->lines, out->vertical, &out->count, &out->truncated);
}

int vocr_textdet_maps(const vocr_textdet *m, const VocrLinesImage *im, void *arena, size_t arena_bytes,
                      vocr_textdet_map_row row, void *user)
{
    Src s;
    Out o;
    VocrLine lines[VOCR_LINES_MAX];
    unsigned vertical[VOCR_LINES_MAX];
    if (!image_ok(im) || !row) return VOCR_LINES_E_ARG;
    s.im = im;
    s.rows = NULL;
    o.lines = lines;
    o.vertical = vertical;
    return detect(m, &s, im->width, im->height, im->bpp, arena, arena_bytes, &o, row, user);
}

/* ---------------------------------------------------------------- the finder mode */

size_t vocr_lines_det_arena_bytes(const vocr_textdet *m, unsigned max_width, unsigned max_height)
{
    size_t a = vocr_lines_arena_bytes(max_width, max_height), b = m ? vocr_textdet_arena_bytes(m, max_width, max_height) : 0;
    return a > b ? a : b;
}

int vocr_find_lines_det(const VocrLinesImage *im, int finder, const vocr_textdet *m, void *arena,
                        size_t arena_bytes, VocrLineSet *out)
{
    if (finder != VOCR_LINES_TEXTDET) return vocr_find_lines(im, finder, arena, arena_bytes, out);
    if (!out) return VOCR_LINES_E_ARG;
    return run_image(m, im, arena, arena_bytes, out->lines, NULL, &out->count, &out->truncated);
}

int vocr_find_lines_det_rows(const VocrLinesRows *src, int finder, const vocr_textdet *m, void *arena,
                             size_t arena_bytes, VocrLineSet *out)
{
    if (finder != VOCR_LINES_TEXTDET) return vocr_find_lines_rows(src, finder, arena, arena_bytes, out);
    if (!out) return VOCR_LINES_E_ARG;
    return run_rows(m, src, arena, arena_bytes, out->lines, NULL, &out->count, &out->truncated);
}
