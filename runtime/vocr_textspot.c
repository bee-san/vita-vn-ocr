/* vocr_textspot.c - the full-frame text spotter's runtime (vocr_textspot.h, docs/textspot.md).
 *
 * synth/textspot_twin.py is the specification; this file computes exactly what it computes, for every frame and
 * every blob its Model accepts (tests/textspot_parity.py checks the head's maps, the boxes, blocks, order, pick
 * and truncated flag). The network and its streaming are vocr_textdet.c's (copied, not linked: that file is
 * frozen). In short:
 *
 * Input. grey = (77 R + 150 G + 29 B) >> 8. A frame W x H wider than the blob's half_above (> 0) is halved:
 * f = 2 and the network reads x[i][j] = (g[2i][2j] + g[2i][2j'] + g[2i'][2j] + g[2i'][2j'] + 2) >> 2 with
 * 2i' = min(2i + 1, H - 1), 2j' = min(2j + 1, W - 1) (ceil(H/2) x ceil(W/2)); else f = 1 and it reads g. The
 * halving keeps one row of ceil(W/2) pair sums: an even frame row writes g[y][2j] + g[y][2j'], the odd row
 * after it adds its own and the input row (sum + 2) >> 2 goes into the network; the last row of an odd height
 * adds its pairs twice (the twin repeats it).
 *
 * Network (vocr_textdet.c, textdet_twin.run_net). x0 = x - 128 (int8). Each layer (dense or depthwise, 1x1 or
 * 3x3 with a stride of 1 or 2 and a dilation d, zero padding of d) sums int8 weights times int8 codes into an
 * int32 accumulator, adds the int32 bias and requantises:
 *     out = clamp(floor(((bias + sum) * mult + 2^(shift - 1)) / 2^shift), relu ? 0 : -128, 127).
 * The last layer gives four int8 maps at the output stride s: K (kernel), O (vertical), A1, A2 (near and far
 * edge distances across the line, in 1/2 network-input px).
 *
 * Streaming. Every 3x3 layer keeps a ring of the (k - 1) d + 1 = 2 d + 1 input rows its kernel spans, each row
 * one plane per channel with d zero columns either side and BLK * stride zero columns of slack; an output row is
 * computed, BLK = 8 pixels at a time, once the last input row it reads has arrived (or lies below the input), and
 * the layers are drained deepest first, so a ring slot is overwritten only when no later output row reads it.
 * 1x1 layers keep no rows: they run on the block of 8 pixels the layer before them has just produced, into the
 * next 3x3 layer's ring or the head's map row. The arena holds the rings, three blocks, one map row, the halving
 * row and the decode below, for any height.
 *
 * Decode (textspot_twin.candidates). In each map row the kernel cells (K > 0) form runs; a run joins the
 * components of the runs of the row above it overlaps (4-connectivity), by union-find over component slots, and
 * adds its cells' sums (n, K, O, A1, A2, rows, columns). After a row, a component no run of the row continues
 * is complete: n >= min cells; conf = floor(sum K / n) >= conf_min; vertical = sum O > 0; across the line
 * near = floor(f (2 s sum i + n (s - 1) - sum A1) / 2n), far = ceil(f (2 s sum i + n (s - 1) + sum A2) / 2n);
 * along it [f s first, f s (last + 1)); dropped when far - near < 4 f; padded by max(2, ((far - near) 18 + 50)
 * / 100), clipped to the frame, dropped when empty. Live components are bounded by the runs of two rows, at most
 * the map's width (a slot per distinct component of the row above plus per run of this row touching none, and
 * their first columns are distinct). The 64 best candidates by the cap key (-conf, -n, vertical, y, x, h, w) are
 * kept in a sorted list (truncated when one more arrives).
 *
 * Blocks (textspot_twin.group_and_order) on the kept boxes sorted by (x, y, w, h, vertical, conf): union-find
 * over every pair that joins (one orientation, overlap along, gap across <= the larger size across, the larger
 * size across <= 3/2 of the smaller), blocks in the order of their first box; each block's boxes in rows (or
 * column groups) by stable sorts with the twin's keys; blocks in rows top to bottom by stable sorts; the pick.
 * Then each box's dark flag, boxes in output order (vocr_line_dark of its box on the frame).
 *
 * Integer ranges (with VOCR_TEXTSPOT_MAX_SIDE = 2048; the network's input side is at most 2048 at f = 1 and
 * 1024 at f = 2):
 *  - an accumulator sums at most 9 * 64 = 576 products of two int8 codes, |w x| <= 2^14, so |sum| <= 576 * 2^14
 *    < 2^24 fits int32. The bias enters through bq = bias * mult + 2^(shift - 1) (int64, per channel):
 *    |bias * mult| <= 2^31 * (2^31 - 1) < 2^62, 2^(shift - 1) <= 2^61 and |sum * mult| < 2^24 * 2^31 = 2^55, so
 *    bq + sum * mult is the exact value and below 2^63. The NEON code, for a channel whose |bias| <= 2^31 - 1 -
 *    2^24 (so bias + sum fits int32), forms (bias + sum) * mult < 2^62 instead and lets VRSHL add 2^(shift - 1).
 *  - a halving pair sum is at most 2 * 255 and the row sum 4 * 255 + 2, in uint16.
 *  - the map has at most ceil(2048 / 4)^2 = 2^18 cells at f = 1 (2^16 at f = 2), so a component has n <= 2^18
 *    cells, sum K <= 127 n < 2^25, and |sum O|, |sum A1|, |sum A2| <= 128 n. Across the line a cell's
 *    c2 = 2 s i + s - 1 <= 4091 at f = 1 (map side <= 512 at s = 4, 256 at s = 8) and <= 2043 at f = 2, so
 *    |2 s sum i + n (s - 1) -+ sum A| <= n (4091 + 128) <= 2^18 * 4219 < 2^31 at f = 1 and f n (2043 + 128)
 *    <= 2 * 2^16 * 2171 < 2^29 at f = 2; 2 n <= 2^19. Then near >= -124 and far <= 2170, so across <= 2294, pad
 *    <= 413 and an edge lies in [-537, 2583] before the clip; after it every coordinate fits uint16.
 *  - a block's score, the sum of its boxes' w h doubled, is at most 2 * 64 * 2048^2 = 2^29 (uint32).
 *
 * Memory: no allocation; one arena (vocr_textspot_frame_arena_bytes for the frame, vocr_textspot_arena_bytes for
 * every frame up to a size) holds the per-layer state, the requantisation constants, the rings, the map row, the
 * halving row, the decode's component slots and run lists, the candidates and (over the network's part, dead by
 * then) the blocks' scratch. libc: memset and memcpy only (and libgcc's division helpers on 32-bit ARM and MIPS,
 * a few per row and per component).
 */
#include "vocr_textspot.h"

#include <stdint.h>
#include <string.h>

/* NEON kernels on the Vita (-mfpu=neon), the same arithmetic as the scalar ones; -DVOCR_NO_NEON forces the
 * scalar code (as for vocr.c and vocr_textdet.c). */
#if defined(__ARM_NEON) && !defined(VOCR_NO_NEON)
#include <arm_neon.h>
#define TS_NEON 1
#else
#define TS_NEON 0
#endif

#define VTSP_HEADER 32u
#define HEADER 64u          /* the VTDT header */
#define RECORD 32u
#define LOADED 0x31505354u  /* "TSP1" */
#define CANDS VOCR_TEXTSPOT_MAX_BOXES
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

/* [off, off + size) inside the data part of the blob, off 4-byte aligned (twin: textdet_twin.Model). */
static int section_ok(uint32_t off, uint32_t size, uint32_t data, uint32_t total)
{
    return !(off % 4u) && off >= data && off <= total && size <= total - off;
}

/* Every check of textdet_twin.Model on the VTDT blob b (bytes long); fills m's network fields. */
static int parse_net(vocr_textspot *m, const unsigned char *b, size_t bytes)
{
    uint32_t total, n, data, cin = 1, stride = 1, len, i;
    if (bytes < HEADER || b[0] != 'V' || b[1] != 'T' || b[2] != 'D' || b[3] != 'T') return VOCR_TEXTSPOT_E_BLOB;
    total = rd32(b + 8);
    if (rd32(b + 4) != 1u || (size_t)total != bytes) return VOCR_TEXTSPOT_E_BLOB;
    if (crc32_ieee(b + 16, bytes - 16u) != rd32(b + 12)) return VOCR_TEXTSPOT_E_BLOB;
    n = rd16(b + 16);
    if ((b[18] != 4u && b[18] != 8u) || b[19] != 4u || b[20] != 2u || b[21] < 1u || b[21] > 4u)
        return VOCR_TEXTSPOT_E_BLOB;
    if (n < 2u || n > VOCR_TEXTSPOT_MAX_LAYERS || HEADER + n * RECORD > total) return VOCR_TEXTSPOT_E_BLOB;
    for (len = 32; len && !b[32 + len - 1u]; len--) {}   /* the name: ASCII once trailing NULs go */
    for (i = 0; i < len; i++)
        if (b[32 + i] >= 0x80u) return VOCR_TEXTSPOT_E_BLOB;
    data = HEADER + n * RECORD;
    for (i = 0; i < n; i++) {
        const unsigned char *r = b + HEADER + i * RECORD;
        vocr_textspot_layer *L = &m->layers[i];
        uint32_t kind = r[0], k = r[1], st = r[2], dil = r[3], relu = r[4];
        uint32_t lc = rd16(r + 8), cout = rd16(r + 10), nw, c;
        uint32_t wo = rd32(r + 12), bo = rd32(r + 16), mo = rd32(r + 20), so = rd32(r + 24);
        if (kind > 1u || (k != 1u && k != 3u) || (st != 1u && st != 2u) || dil < 1u || dil > 8u || relu > 1u ||
            lc != cin || cout < 1u || cout > MAXCH)
            return VOCR_TEXTSPOT_E_BLOB;
        if (kind == 1u && (cout != lc || k != 3u)) return VOCR_TEXTSPOT_E_BLOB;
        if (st == 2u && (k != 3u || dil != 1u)) return VOCR_TEXTSPOT_E_BLOB;
        nw = cout * (kind == 1u ? 1u : lc) * k * k;
        if (!section_ok(wo, nw, data, total) || !section_ok(bo, 4u * cout, data, total) ||
            !section_ok(mo, 4u * cout, data, total) || !section_ok(so, cout, data, total))
            return VOCR_TEXTSPOT_E_BLOB;
        for (c = 0; c < cout; c++)
            if (rds32(b + mo + 4u * c) <= 0 || b[so + c] < 1u || b[so + c] > 62u) return VOCR_TEXTSPOT_E_BLOB;
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
    if (stride != b[18] || cin != 4u || m->layers[n - 1u].relu) return VOCR_TEXTSPOT_E_BLOB;
    m->n_layers = n;
    m->stride = b[18];
    m->min_cells = b[21];
    m->net_bytes = (unsigned long)total;
    memcpy(m->ckpt, b + 24, 8);
    memcpy(m->name, b + 32, 32);
    m->name[32] = '\0';
    return VOCR_LINES_OK;
}

/* Every check of textspot_twin.Model (the VTSP header), then the inner VTDT blob's. */
static int parse(vocr_textspot *m, const unsigned char *b, size_t bytes)
{
    uint32_t total;
    if (bytes < VTSP_HEADER || b[0] != 'V' || b[1] != 'T' || b[2] != 'S' || b[3] != 'P') return VOCR_TEXTSPOT_E_BLOB;
    total = rd32(b + 8);
    if (rd32(b + 4) != 1u || (size_t)total != bytes) return VOCR_TEXTSPOT_E_BLOB;
    if (crc32_ieee(b + 16, bytes - 16u) != rd32(b + 12)) return VOCR_TEXTSPOT_E_BLOB;
    /* u16 half_above, u8 conf_min, u8 0, u32 offset 32, u32 size total - 32, u32 0 */
    if (b[18] < 1u || b[18] > 127u || b[19] || rd32(b + 20) != VTSP_HEADER || rd32(b + 24) != total - VTSP_HEADER ||
        rd32(b + 28))
        return VOCR_TEXTSPOT_E_BLOB;
    m->half_above = rd16(b + 16);
    m->conf_min = b[18];
    m->blob_bytes = (unsigned long)total;
    return parse_net(m, b + VTSP_HEADER, (size_t)(total - VTSP_HEADER));
}

int vocr_textspot_load(vocr_textspot *m, const void *blob, size_t bytes)
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
    int32_t b;    /* bias */
    int32_t m;    /* multiplier */
    int32_t sh;   /* shift */
    int32_t fast; /* |b| <= 2^31 - 1 - 2^24: bias + sum fits int32 (NEON's 32-bit form) */
} Chan;

typedef struct {
    uint32_t ring;       /* arena offset of a 3x3 layer's input ring */
    uint32_t plane;      /* bytes per channel plane of a ring row: w_in + 2 dil + BLK stride */
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
    int32_t n, sk, so, sa1, sa2, sr, sc;  /* cells; sums of K, O, A1, A2, map rows, map columns */
    uint16_t r0, r1, c0, c1;              /* first and last row, first and last column */
    uint16_t parent;                      /* union-find parent; NONE when the slot is free */
    uint16_t stamp;                       /* map row + 1 of the last row whose runs joined it */
} Comp;

/* A kept box (the cap's candidates) or, in the blocks' scratch, a block. */
typedef struct {
    uint16_t x, y, w, h;      /* the box, clipped to the frame */
    uint8_t v, conf;          /* vertical flag, confidence (boxes) */
    uint8_t first, count;     /* blocks: their boxes' indices are member[first .. first + count) */
    uint32_t n;               /* boxes: the component's cells (in the cap key) */
} Cand;

/* The blocks' scratch, over the network's part of the arena once the frame has been read. */
typedef struct {
    uint8_t parent[CANDS];    /* union-find over the boxes in sorted order (by position) */
    uint8_t sorted[CANDS];    /* the boxes (indices into the candidates) by (x, y, w, h, vertical, conf) */
    uint8_t member[CANDS];    /* the boxes block by block, each block in reading order */
    uint8_t order[CANDS];     /* the blocks in reading order */
    Cand blk[CANDS];          /* the blocks in the order of their first box */
} Group;

typedef struct {
    size_t run, chan, rings, tmp, head, comp, freel, prev, cur, psum, cand, total;
    unsigned f, win, wm, nc, nr;
} Plan;

static size_t up8(size_t n)
{
    return (n + 7u) & ~(size_t)7u;
}

static int loaded(const vocr_textspot *m)
{
    return m && m->loaded == LOADED && m->n_layers >= 2u && m->n_layers <= VOCR_TEXTSPOT_MAX_LAYERS;
}

/* 2 when a frame this wide is halved before the network (textspot_twin.scale_of), else 1. */
static unsigned scale_of(const vocr_textspot *m, unsigned width)
{
    return m->half_above && width > m->half_above ? 2u : 1u;
}

/* Bytes per channel plane of a 3x3 layer's ring row: the d zero columns either side of the w input pixels, then
 * BLK * stride zero columns that a last, partial block of BLK output pixels reads past them (BLK - 1 strides for
 * the scalar code, one byte more for NEON's 16-byte de-interleaving load). */
static size_t plane_bytes(unsigned w, const vocr_textspot_layer *L)
{
    return (size_t)w + 2u * L->dil + BLK * L->stride;
}

/* The arena layout for frames of the given width (sections 8-byte aligned from an aligned base). The blocks'
 * scratch (Group) lies over the sections before the candidates, so those span at least its size. */
static void plan(const vocr_textspot *m, unsigned width, Plan *p)
{
    unsigned i, f = scale_of(m, width), w = f == 2u ? (width + 1u) / 2u : width, maxch = 1;
    size_t rings = 0;
    p->f = f;
    p->win = w;
    for (i = 0; i < m->n_layers; i++) {
        const vocr_textspot_layer *L = &m->layers[i];
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
    p->psum = p->cur + up8((size_t)p->nr * sizeof(Seg));
    p->cand = p->psum + up8(f == 2u ? (size_t)p->win * sizeof(uint16_t) : 0u);
    if (p->cand < up8(sizeof(Group))) p->cand = up8(sizeof(Group));
    p->total = p->cand + up8((size_t)CANDS * sizeof(Cand));
}

static int side_ok(unsigned w, unsigned h)
{
    return w && h && w <= VOCR_TEXTSPOT_MAX_SIDE && h <= VOCR_TEXTSPOT_MAX_SIDE;
}

size_t vocr_textspot_frame_arena_bytes(const vocr_textspot *m, unsigned width, unsigned height)
{
    Plan p;
    if (!loaded(m) || !side_ok(width, height)) return 0;
    plan(m, width, &p);
    return p.total + 7u;
}

size_t vocr_textspot_arena_bytes(const vocr_textspot *m, unsigned max_width, unsigned max_height)
{
    size_t a = vocr_textspot_frame_arena_bytes(m, max_width, max_height), b;
    if (a && scale_of(m, max_width) == 2u) {   /* a frame of half_above px is read at full size */
        b = vocr_textspot_frame_arena_bytes(m, m->half_above, max_height);
        if (b > a) a = b;
    }
    return a;
}

/* ---------------------------------------------------------------- network */

typedef struct {
    const VocrLinesImage *im;   /* a frame in memory, or */
    const VocrLinesRows *rows;  /* a row source */
} Src;

typedef struct {
    const vocr_textspot *m;
    unsigned char *base;        /* the aligned arena */
    Run *run;
    Chan *chan;
    int8_t *head;               /* the head's map row: the K, O, A1, A2 planes of wm cells each */
    int8_t *tmp;                /* three blocks of BLK x maxch codes: a 3x3 output, two 1x1 ones */
    uint16_t *psum;             /* the halving's pair sums (f = 2) */
    unsigned maxch;             /* the most output channels of a layer */
    unsigned rl[VOCR_TEXTSPOT_MAX_LAYERS], nrl;   /* the 3x3 layers, in order */
    unsigned f, fw, fh;         /* the scale and the frame's size */
    unsigned width, height, wm, hm;   /* the network's input and output sizes */
    vocr_textspot_map_row hook;
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

/* bias * m + 2^(sh - 1): the requantisation's constant term (int64, see the ranges above) */
static int64_t bq_of(const Chan *c)
{
    return (int64_t)c->b * c->m + ((int64_t)1 << (c->sh - 1));
}

#if !TS_NEON
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
    int64_t p = bq_of(c) + (int64_t)acc * c->m;
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

#if TS_NEON
/* ---- NEON (the Vita): a block's 8 int32 sums in two registers; the arithmetic of requant() */

/* (bias + sum) * m + 2^(sh - 1), floor-divided by 2^sh, per lane, saturated to int8, then at least lo; n lanes
 * stored at out. Where bias + sum fits int32 (fast: every channel of a trained blob) that is VRSHL by -sh of the
 * exact int64 product: (x + 2^(sh-1)) >> sh, round half up, the twin's rule. Otherwise the int64 constant
 * bias * m + 2^(sh - 1) is added by VMLAL and VSHL shifts (floor). VSHL / VRSHL read the shift from each 64-bit
 * lane's lowest byte, so a dup of the int32 -sh serves. */
#define RQ_STORE(LO, HI, C, LOW, OUT, N) do { \
        const Chan *c_ = (C); \
        int64x2_t s_ = vreinterpretq_s64_s32(vdupq_n_s32(-c_->sh)), p0_, p1_, p2_, p3_; \
        int32x2_t m_ = vdup_n_s32(c_->m); \
        if (c_->fast) { \
            int32x4_t b_ = vdupq_n_s32(c_->b), l_ = vaddq_s32((LO), b_), h_ = vaddq_s32((HI), b_); \
            p0_ = vrshlq_s64(vmull_s32(vget_low_s32(l_), m_), s_); \
            p1_ = vrshlq_s64(vmull_s32(vget_high_s32(l_), m_), s_); \
            p2_ = vrshlq_s64(vmull_s32(vget_low_s32(h_), m_), s_); \
            p3_ = vrshlq_s64(vmull_s32(vget_high_s32(h_), m_), s_); \
        } else { \
            int64x2_t b_ = vdupq_n_s64(bq_of(c_)); \
            p0_ = vshlq_s64(vmlal_s32(b_, vget_low_s32(LO), m_), s_); \
            p1_ = vshlq_s64(vmlal_s32(b_, vget_high_s32(LO), m_), s_); \
            p2_ = vshlq_s64(vmlal_s32(b_, vget_low_s32(HI), m_), s_); \
            p3_ = vshlq_s64(vmlal_s32(b_, vget_high_s32(HI), m_), s_); \
        } \
        { \
            int16x4_t h0_ = vqmovn_s32(vcombine_s32(vqmovn_s64(p0_), vqmovn_s64(p1_))); \
            int16x4_t h1_ = vqmovn_s32(vcombine_s32(vqmovn_s64(p2_), vqmovn_s64(p3_))); \
            int8x8_t v_ = vqmovn_s16(vcombine_s16(h0_, h1_)); \
            if (!(LOW)) v_ = vmax_s8(v_, vmov_n_s8(0));   /* ReLU; -128 is the saturation's own floor */ \
            if ((N) == BLK) { \
                vst1_s8((OUT), v_); \
            } else { \
                int8_t t_[BLK]; \
                unsigned i_; \
                vst1_s8(t_, v_); \
                for (i_ = 0; i_ < (N); i_++) (OUT)[i_] = t_[i_]; \
            } \
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

/* A block of BLK output pixels from pixel j0 of a 3x3 layer with input stride S, output channel by output channel
 * into out[c * ostride + p] (p < n). rows[ky] is the padded input row of kernel row ky (the centre row where
 * valid[ky] is 0, its weights then masked to 0), plane bytes from one input channel to the next. Depthwise:
 * output channel c reads input channel c; otherwise every input channel (cin of them). */
#define K3_BLOCK(name, S) \
static void name(const int8_t *w, const Chan *ch, int32_t lo, unsigned cin, unsigned cout, int depthwise, \
                 const int8_t *const *rows, const int *valid, size_t plane, unsigned d, unsigned j0, int8_t *out, \
                 size_t ostride, unsigned n) \
{ \
    static const int16_t top_[8] = {0, 0, 0, -1, -1, -1, -1, -1}, mid_[8] = {-1, -1, -1, -1, -1, -1, 0, 0}; \
    int16x8_t keep = vdupq_n_s16(-1); \
    int last = valid[2]; \
    unsigned c, q, nq = depthwise ? 1u : cin; \
    const int8_t *r0 = rows[0] + (size_t)j0 * (S), *r1 = rows[1] + (size_t)j0 * (S); \
    const int8_t *r2 = rows[2] + (size_t)j0 * (S), *wq = w; \
    if (!valid[0]) keep = vandq_s16(keep, vld1q_s16(top_)); \
    if (!valid[2]) keep = vandq_s16(keep, vld1q_s16(mid_)); \
    for (c = 0; c < cout; c++) { \
        int32x4_t lo4 = vdupq_n_s32(0), hi4 = lo4; \
        const int8_t *x0 = r0, *x1 = r1, *x2 = r2; \
        if (depthwise) { \
            x0 += c * plane; \
            x1 += c * plane; \
            x2 += c * plane; \
        } \
        for (q = 0; q < nq; q++, wq += 9, x0 += plane, x1 += plane, x2 += plane) { \
            int16x8_t wv = vandq_s16(vmovl_s8(vld1_s8(wq)), keep); \
            int16x4_t wa = vget_low_s16(wv), wb = vget_high_s16(wv); \
            const int8_t *x = x0; \
            MACL(lo4, hi4, LOAD8(x, S), wa, 0); \
            x += d; \
            MACL(lo4, hi4, LOAD8(x, S), wa, 1); \
            x += d; \
            MACL(lo4, hi4, LOAD8(x, S), wa, 2); \
            x = x1; \
            MACL(lo4, hi4, LOAD8(x, S), wa, 3); \
            x += d; \
            MACL(lo4, hi4, LOAD8(x, S), wb, 0); \
            x += d; \
            MACL(lo4, hi4, LOAD8(x, S), wb, 1); \
            x = x2; \
            MACL(lo4, hi4, LOAD8(x, S), wb, 2); \
            x += d; \
            MACL(lo4, hi4, LOAD8(x, S), wb, 3); \
            if (last) MACN(lo4, hi4, LOAD8(x + d, S), wq[8]); \
        } \
        RQ_STORE(lo4, hi4, &ch[c], lo, out + c * ostride, n); \
    } \
}
K3_BLOCK(k3_block_s1, 1u)
K3_BLOCK(k3_block_s2, 2u)

/* A block of a 1x1 layer: in[q * BLK + p] -> out[c * ostride + p] (p < n). Two output channels per pass (four
 * sums, two weight vectors and the inputs stay in the 16 Q registers), weights 8 input channels at a time (never
 * read past a weight row), then one at a time. */
#define K1_STEP(U, H, L) do { \
        int16x8_t x_ = vmovl_s8(vld1_s8(in + (q + (U)) * BLK)); \
        int16x4_t xl_ = vget_low_s16(x_), xh_ = vget_high_s16(x_); \
        a0 = vmlal_lane_s16(a0, xl_, H(k0), L); b0 = vmlal_lane_s16(b0, xh_, H(k0), L); \
        a1 = vmlal_lane_s16(a1, xl_, H(k1), L); b1 = vmlal_lane_s16(b1, xh_, H(k1), L); \
    } while (0)

static void k1_block(const int8_t *w, const Chan *ch, int32_t lo, unsigned cin, unsigned cout, const int8_t *in,
                     int8_t *out, size_t ostride, unsigned n)
{
    unsigned c = 0, q;
    for (; c + 2u <= cout; c += 2u) {
        int32x4_t a0 = vdupq_n_s32(0), b0 = a0, a1 = a0, b1 = a0;
        const int8_t *w0 = w + (size_t)c * cin, *w1 = w0 + cin;
        for (q = 0; q + 8u <= cin; q += 8u) {
            int16x8_t k0 = vmovl_s8(vld1_s8(w0 + q)), k1 = vmovl_s8(vld1_s8(w1 + q));
            K1_STEP(0, vget_low_s16, 0);
            K1_STEP(1, vget_low_s16, 1);
            K1_STEP(2, vget_low_s16, 2);
            K1_STEP(3, vget_low_s16, 3);
            K1_STEP(4, vget_high_s16, 0);
            K1_STEP(5, vget_high_s16, 1);
            K1_STEP(6, vget_high_s16, 2);
            K1_STEP(7, vget_high_s16, 3);
        }
        for (; q < cin; q++) {
            int16x8_t x = vmovl_s8(vld1_s8(in + q * BLK));
            MACN(a0, b0, x, w0[q]);
            MACN(a1, b1, x, w1[q]);
        }
        RQ_STORE(a0, b0, &ch[c], lo, out + c * ostride, n);
        RQ_STORE(a1, b1, &ch[c + 1u], lo, out + (c + 1u) * ostride, n);
    }
    for (; c < cout; c++) {
        int32x4_t lo4 = vdupq_n_s32(0), hi4 = lo4;
        const int8_t *wc = w + (size_t)c * cin;
        for (q = 0; q < cin; q++) MACN(lo4, hi4, vmovl_s8(vld1_s8(in + q * BLK)), wc[q]);
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
        int64_t bq = bq_of(c);
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

/* A block of BLK output pixels from pixel j0 of a 3x3 layer with input stride S, output channel by output channel
 * into out[c * ostride + p] (p < n). rows[ky] is the padded input row of kernel row ky (valid[ky] when it lies
 * inside the input), plane bytes from one input channel to the next. Depthwise: output channel c reads input
 * channel c; otherwise every input channel (cin of them). */
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

static const int8_t *weights(const vocr_textspot_layer *L)
{
    return (const int8_t *)(const void *)L->w;
}

/* The 1x1 layers [from, to) on a block in[q * BLK + p]; the last writes n pixels to dst (plane dstride bytes
 * apart); a and b are BLK * maxch scratch blocks. */
static void chain(const Net *t, unsigned from, unsigned to, const int8_t *in, int8_t *a, int8_t *b, int8_t *dst,
                  size_t dstride, unsigned n)
{
    unsigned l;
    for (l = from; l < to; l++) {
        const vocr_textspot_layer *L = &t->m->layers[l];
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
    p->sk += q->sk;
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

/* The cap's order: the twin's key (-conf, -n, vertical, y, x, h, w), smallest first. */
static int cap_less(const Cand *a, const Cand *b)
{
    if (a->conf != b->conf) return a->conf > b->conf;
    if (a->n != b->n) return a->n > b->n;
    if (a->v != b->v) return a->v < b->v;
    if (a->y != b->y) return a->y < b->y;
    if (a->x != b->x) return a->x < b->x;
    if (a->h != b->h) return a->h < b->h;
    return a->w < b->w;
}

/* A candidate into the sorted list; the CANDS smallest by the cap key are kept (a tie with the last one is the
 * same box, so either may go). */
static void keep(Net *t, const Cand *v)
{
    unsigned j;
    if (t->ncand == CANDS) {
        t->ctrunc = 1;
        if (!cap_less(v, &t->cand[CANDS - 1u])) return;
        t->ncand--;
    }
    for (j = t->ncand; j > 0 && cap_less(v, &t->cand[j - 1u]); j--) t->cand[j] = t->cand[j - 1u];
    t->cand[j] = *v;
    t->ncand++;
}

/* A complete component: its box (twin: candidates), if it has one. */
static void finish(Net *t, const Comp *c)
{
    int32_t s = (int32_t)t->m->stride, n = c->n, f = (int32_t)t->f, base, near_, far_, across, pad, lo, hi;
    int32_t x0, x1, y0, y1, conf;
    int vertical;
    Cand v;
    if (n < (int32_t)t->m->min_cells) return;
    conf = c->sk / n;                                          /* sum K >= n > 0: the floor */
    if (conf < (int32_t)t->m->conf_min) return;
    vertical = c->so > 0;
    base = 2 * s * (vertical ? c->sc : c->sr) + n * (s - 1);   /* sum of c2 = 2 s i + s - 1 */
    near_ = floor_div(f * (base - c->sa1), 2 * n);
    far_ = ceil_div(f * (base + c->sa2), 2 * n);
    across = far_ - near_;
    if (across < MIN_ACROSS * f) return;
    pad = (across * PAD_PCT + 50) / 100;
    if (pad < PAD_MIN) pad = PAD_MIN;
    if (vertical) {
        lo = f * s * (int32_t)c->r0;
        hi = f * s * ((int32_t)c->r1 + 1);
        x0 = near_ - pad;
        x1 = far_ + pad;
        y0 = lo - pad;
        y1 = hi + pad;
    } else {
        lo = f * s * (int32_t)c->c0;
        hi = f * s * ((int32_t)c->c1 + 1);
        y0 = near_ - pad;
        y1 = far_ + pad;
        x0 = lo - pad;
        x1 = hi + pad;
    }
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > (int32_t)t->fw) x1 = (int32_t)t->fw;
    if (y1 > (int32_t)t->fh) y1 = (int32_t)t->fh;
    if (x1 <= x0 || y1 <= y0) return;
    v.x = (uint16_t)x0;
    v.y = (uint16_t)y0;
    v.w = (uint16_t)(x1 - x0);
    v.h = (uint16_t)(y1 - y0);
    v.v = (uint8_t)vertical;
    v.conf = (uint8_t)conf;
    v.first = v.count = 0;
    v.n = (uint32_t)n;
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
        int32_t n = 0, sk = 0, so = 0, sa1 = 0, sa2 = 0, sc = 0;
        Comp *c;
        if (kk[j] <= 0) {
            j++;
            continue;
        }
        a = j;
        for (; j < wm && kk[j] > 0; j++) {
            sk += kk[j];
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
            c->n = c->sk = c->so = c->sa1 = c->sa2 = c->sr = c->sc = 0;
            c->r0 = (uint16_t)i;
            c->c0 = (uint16_t)a;
            c->c1 = (uint16_t)(j - 1u);
            c->parent = (uint16_t)root;
            c->stamp = 0;
        }
        c = &cp[root];
        c->n += n;
        c->sk += sk;
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
    const vocr_textspot_layer *L = &t->m->layers[l];
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
    const vocr_textspot_layer *L = &t->m->layers[l];
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

#if TS_NEON
/* The grey of 8 RGBA pixels, (77 R + 150 G + 29 B) >> 8 (at most 256 * 255: no u16 overflow). */
static uint8x8_t grey8(const unsigned char *p)
{
    uint8x8x4_t a = vld4_u8(p);
    uint16x8_t s = vmull_u8(a.val[0], vdup_n_u8(77));
    s = vmlal_u8(s, a.val[1], vdup_n_u8(150));
    s = vmlal_u8(s, a.val[2], vdup_n_u8(29));
    return vshrn_n_u16(s, 8);
}
#endif

/* The halving's pair sums of a frame row: ps[j] (+)= g[2j] + g[min(2j + 1, W - 1)], j < ceil(W / 2). */
static void pair_sums(uint16_t *ps, const unsigned char *row, unsigned fw, unsigned bpp, int add)
{
    unsigned j = 0, n = fw / 2u;
#if TS_NEON
    if (bpp == VOCR_LINES_GREY) {
        for (; j + 8u <= n; j += 8u) {        /* 16 pixels: 8 even, 8 odd */
            uint8x8x2_t p = vld2_u8(row + 2u * j);
            uint16x8_t s = vaddl_u8(p.val[0], p.val[1]);
            if (add) s = vaddq_u16(s, vld1q_u16(ps + j));
            vst1q_u16(ps + j, s);
        }
    } else {
        for (; j + 8u <= n; j += 8u) {        /* 16 pixels, two grey8 of 8; pairwise sums */
            uint16x8_t s = vcombine_u16(vpaddl_u8(grey8(row + 8u * j)), vpaddl_u8(grey8(row + 8u * j + 32u)));
            if (add) s = vaddq_u16(s, vld1q_u16(ps + j));
            vst1q_u16(ps + j, s);
        }
    }
#endif
    if (bpp == VOCR_LINES_GREY) {
        if (add)
            for (; j < n; j++) ps[j] = (uint16_t)(ps[j] + row[2u * j] + row[2u * j + 1u]);
        else
            for (; j < n; j++) ps[j] = (uint16_t)(row[2u * j] + row[2u * j + 1u]);
    } else {
        for (; j < n; j++) {
            unsigned v = grey_px(row, 2u * j, bpp) + grey_px(row, 2u * j + 1u, bpp);
            ps[j] = (uint16_t)(add ? ps[j] + v : v);
        }
    }
    if (fw & 1u) {                            /* the last column repeats */
        unsigned v = 2u * grey_px(row, fw - 1u, bpp);
        ps[n] = (uint16_t)(add ? ps[n] + v : v);
    }
}

/* n network-input codes: x - 128, x = (ps + 2) >> 2 (sums) or the row's grey (row). Only [0, n) is written: the
 * ring's padding past the row stays zero. */
static void input_codes(int8_t *dst, const uint16_t *ps, const unsigned char *row, unsigned n, unsigned bpp)
{
    unsigned x = 0;
#if TS_NEON
    if (ps) {
        for (; x + 8u <= n; x += 8u)
            vst1_s8(dst + x, vreinterpret_s8_u8(veor_u8(vrshrn_n_u16(vld1q_u16(ps + x), 2), vdup_n_u8(0x80))));
    } else if (bpp == VOCR_LINES_GREY) {
        for (; x + 16u <= n; x += 16u)
            vst1q_s8(dst + x, vreinterpretq_s8_u8(veorq_u8(vld1q_u8(row + x), vdupq_n_u8(0x80))));
    } else {
        for (; x + 8u <= n; x += 8u) vst1_s8(dst + x, vreinterpret_s8_u8(veor_u8(grey8(row + 4u * x), vdup_n_u8(0x80))));
    }
#endif
    if (ps)
        for (; x < n; x++) dst[x] = (int8_t)((int)((ps[x] + 2u) >> 2) - 128);
    else
        for (; x < n; x++) dst[x] = (int8_t)((int)grey_px(row, x, bpp) - 128);
}

/* Frame row y into the network: at f = 1 the row itself; at f = 2 every second row (and the last) completes an
 * input row of 2x2 means. The input row goes into the first 3x3 layer, through the 1x1 layers before it. */
static int feed(Net *t, const Src *s, unsigned y, unsigned bpp)
{
    const unsigned char *row = src_row(s, y);
    unsigned first = t->rl[0], x, p;
    int8_t *dst;
    if (!row) return VOCR_LINES_E_SOURCE;
    if (t->f == 2u) {
        pair_sums(t->psum, row, t->fw, bpp, (int)(y & 1u));
        if (!(y & 1u)) {
            if (y + 1u < t->fh) return VOCR_LINES_OK;   /* the pair's second row comes next */
            pair_sums(t->psum, row, t->fw, bpp, 1);      /* an odd height: the last row repeats */
        }
    }
    dst = dest_row(t, first);
    if (!first) {
        input_codes(dst, t->f == 2u ? t->psum : NULL, row, t->width, bpp);
    } else {
        size_t bs = (size_t)BLK * t->maxch;
        for (x = 0; x < t->width; x += BLK) {
            unsigned n = min_u(BLK, t->width - x);
            for (p = 0; p < BLK; p++) {
                unsigned g = p >= n ? 128u : t->f == 2u ? (t->psum[x + p] + 2u) >> 2 : grey_px(row, x + p, bpp);
                t->tmp[p] = (int8_t)((int)g - 128);
            }
            chain(t, 0, first, t->tmp, t->tmp + bs, t->tmp + 2u * bs, dst + x, t->run[first].plane, n);
        }
    }
    delivered(t, first, y / t->f);
    drain(t);
    return VOCR_LINES_OK;
}

/* The per-layer state and the requantisation constants of one call. */
static void setup(Net *t, const Plan *p)
{
    const vocr_textspot *m = t->m;
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
    t->psum = (uint16_t *)(void *)(t->base + p->psum);
    t->cand = (Cand *)(void *)(t->base + p->cand);
    t->nrl = 0;
    t->maxch = 1;
    for (i = 0; i < m->n_layers; i++) {
        const vocr_textspot_layer *L = &m->layers[i];
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
            q->b = rds32(L->b + 4u * c);
            q->m = rds32(L->m + 4u * c);
            q->sh = L->sh[c];
            /* |sum| < 2^24 (576 products of 2^14), so bias + sum fits int32 when |bias| <= 2^31 - 1 - 2^24 */
            q->fast = q->b >= -(INT32_MAX - (1 << 24)) && q->b <= INT32_MAX - (1 << 24);
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

/* ---------------------------------------------------------------- blocks, reading order, pick */

/* The sort keys of textspot_twin.group_and_order, as field codes: 0 x, 1 y, 2 w, 3 h, 4 -(x + w), 5 vertical,
 * 6 conf; 7 ends a key. Python's sorts are stable, so each is an insertion sort from the order before it. */
enum { K_FULL, K_HSORT, K_HROW, K_VSORT, K_VROW, K_BVROW };
static const unsigned char KEYS[6][7] = {
    {0, 1, 2, 3, 5, 6, 7},   /* sorted(boxes): (x, y, w, h, vertical, conf); dark follows from the box */
    {1, 0, 3, 2, 7, 7, 7},   /* horizontal members, and blocks: (y, x, h, w) */
    {0, 1, 3, 2, 7, 7, 7},   /* a row of horizontal members, a row of blocks: (x, y, h, w) */
    {4, 1, 0, 2, 3, 7, 7},   /* vertical members: (-(x + w), y, x, w, h) */
    {1, 4, 3, 2, 7, 7, 7},   /* a group of vertical members: (y, -(x + w), h, w) */
    {4, 1, 3, 2, 7, 7, 7},   /* a row of vertical blocks: (-(x + w), y, h, w) */
};

static int32_t field(const Cand *c, unsigned f)
{
    switch (f) {
    case 0: return c->x;
    case 1: return c->y;
    case 2: return c->w;
    case 3: return c->h;
    case 4: return -((int32_t)c->x + c->w);
    case 5: return c->v;
    default: return c->conf;
    }
}

static int key_less(const Cand *a, const Cand *b, unsigned kind)
{
    const unsigned char *k = KEYS[kind];
    for (; *k != 7u; k++) {
        int32_t p = field(a, *k), q = field(b, *k);
        if (p != q) return p < q;
    }
    return 0;
}

/* A stable sort of ix[0 .. n) (indices into c) by the key. */
static void isort(uint8_t *ix, unsigned n, const Cand *c, unsigned kind)
{
    unsigned i, j;
    for (i = 1; i < n; i++) {
        uint8_t v = ix[i];
        for (j = i; j > 0 && key_less(&c[v], &c[ix[j - 1u]], kind); j--) ix[j] = ix[j - 1u];
        ix[j] = v;
    }
}

/* textspot_twin._joins: one orientation, overlap along, gap across <= the larger size across, and the larger
 * size across <= 3/2 of the smaller. */
static int joins(const Cand *a, const Cand *b)
{
    int32_t ov, gap, sa, sb, mx, mn;
    if (a->v != b->v) return 0;
    if (a->v) {
        ov = (int32_t)min_u((unsigned)a->y + a->h, (unsigned)b->y + b->h) - (a->y > b->y ? a->y : b->y);
        gap = (a->x > b->x ? a->x : b->x) - (int32_t)min_u((unsigned)a->x + a->w, (unsigned)b->x + b->w);
        sa = a->w;
        sb = b->w;
    } else {
        ov = (int32_t)min_u((unsigned)a->x + a->w, (unsigned)b->x + b->w) - (a->x > b->x ? a->x : b->x);
        gap = (a->y > b->y ? a->y : b->y) - (int32_t)min_u((unsigned)a->y + a->h, (unsigned)b->y + b->h);
        sa = a->h;
        sb = b->h;
    }
    mx = sa > sb ? sa : sb;
    mn = sa > sb ? sb : sa;
    return ov > 0 && gap <= mx && 2 * mx <= 3 * mn;
}

/* Overlap of [a, a + n) and [b, b + m), at least half the smaller of n and m (the twin's row rule). */
static int same_row(unsigned a, unsigned n, unsigned b, unsigned m)
{
    int32_t ov = (int32_t)min_u(a + n, b + m) - (int32_t)(a > b ? a : b);
    return 2 * ov >= (int32_t)min_u(n, m);
}

/* Gathers ix[0 .. n) (sorted already) into rows (an item joins the row when it overlaps the row's first item by
 * half the smaller size, along y and h or, vertical, x and w) and sorts each row by the key. */
static void rows_of(uint8_t *ix, unsigned n, const Cand *c, int vertical, unsigned kind)
{
    unsigned a = 0, b;
    for (b = 1; b <= n; b++) {
        if (b < n) {
            const Cand *f = &c[ix[a]], *it = &c[ix[b]];
            if (vertical ? same_row(it->x, it->w, f->x, f->w) : same_row(it->y, it->h, f->y, f->h)) continue;
        }
        isort(ix + a, b - a, c, kind);
        a = b;
    }
}

static uint8_t find_root(uint8_t *parent, uint8_t a)
{
    while (parent[a] != a) {
        parent[a] = parent[parent[a]];
        a = parent[a];
    }
    return a;
}

/* The kept candidates into out: blocks, reading order and the pick (dark flags 0; textspot_twin.group_and_order). */
static void group(const Net *t, VocrTextspotSet *out)
{
    Group *g = (Group *)(void *)t->base;
    const Cand *c = t->cand;
    unsigned n = t->ncand, i, j, nb = 0, nm = 0, k, b, nbox = 0;
    uint32_t best = 0;
    unsigned best_y = 0;
    for (i = 0; i < n; i++) {
        g->sorted[i] = (uint8_t)i;
        g->parent[i] = (uint8_t)i;
    }
    isort(g->sorted, n, c, K_FULL);
    for (i = 0; i < n; i++)                   /* union-find: a root is its group's first box */
        for (j = i + 1u; j < n; j++)
            if (joins(&c[g->sorted[i]], &c[g->sorted[j]])) {
                uint8_t ri = find_root(g->parent, (uint8_t)i), rj = find_root(g->parent, (uint8_t)j);
                if (ri != rj) g->parent[ri > rj ? ri : rj] = ri > rj ? rj : ri;
            }
    for (i = 0; i < n; i++) {                 /* blocks in the order of their first box */
        Cand *bk;
        unsigned x1 = 0, y1 = 0;
        if (find_root(g->parent, (uint8_t)i) != i) continue;
        bk = &g->blk[nb];
        bk->x = bk->y = 0xFFFFu;
        bk->first = (uint8_t)nm;
        for (j = i; j < n; j++) {
            const Cand *m;
            if (find_root(g->parent, (uint8_t)j) != i) continue;
            m = &c[g->sorted[j]];
            g->member[nm++] = g->sorted[j];
            if (m->x < bk->x) bk->x = m->x;
            if (m->y < bk->y) bk->y = m->y;
            if ((unsigned)m->x + m->w > x1) x1 = (unsigned)m->x + m->w;
            if ((unsigned)m->y + m->h > y1) y1 = (unsigned)m->y + m->h;
        }
        bk->w = (uint16_t)(x1 - bk->x);
        bk->h = (uint16_t)(y1 - bk->y);
        bk->v = c[g->sorted[i]].v;
        bk->conf = 0;
        bk->count = (uint8_t)(nm - bk->first);
        bk->n = 0;
        /* the block's boxes in reading order: rows top to bottom (left to right), or columns right to left */
        isort(g->member + bk->first, bk->count, c, bk->v ? K_VSORT : K_HSORT);
        rows_of(g->member + bk->first, bk->count, c, bk->v, bk->v ? K_VROW : K_HROW);
        g->order[nb] = (uint8_t)nb;
        nb++;
    }
    /* blocks: rows of blocks top to bottom; a row left to right, or right to left when all its blocks are vertical */
    isort(g->order, nb, g->blk, K_HSORT);
    for (i = 0, j = 1; j <= nb; j++) {
        int all_v = 1;
        if (j < nb) {
            const Cand *f = &g->blk[g->order[i]], *it = &g->blk[g->order[j]];
            if (same_row(it->y, it->h, f->y, f->h)) continue;
        }
        for (k = i; k < j; k++) all_v &= g->blk[g->order[k]].v;
        isort(g->order + i, j - i, g->blk, all_v ? K_BVROW : K_HROW);
        i = j;
    }
    out->pick = -1;
    for (k = 0; k < nb; k++) {
        const Cand *bk = &g->blk[g->order[k]];
        VocrTextspotBlock *ob = &out->blocks[k];
        uint32_t score = 0;
        ob->x = bk->x;
        ob->y = bk->y;
        ob->w = bk->w;
        ob->h = bk->h;
        ob->vertical = bk->v;
        ob->first = (unsigned char)nbox;
        ob->count = bk->count;
        ob->reserved = 0;
        for (b = 0; b < bk->count; b++) {
            const Cand *m = &c[g->member[bk->first + b]];
            VocrTextspotBox *ox = &out->boxes[nbox++];
            ox->x = m->x;
            ox->y = m->y;
            ox->w = m->w;
            ox->h = m->h;
            ox->vertical = m->v;
            ox->conf = m->conf;
            ox->dark = 0;
            ox->block = (unsigned char)k;
            score += (uint32_t)m->w * m->h;
        }
        if (2u * bk->y + bk->h >= t->fh) score *= 2u;   /* the block's centre in the lower half */
        if (out->pick < 0 || score > best || (score == best && bk->y > best_y)) {
            out->pick = (int)k;
            best = score;
            best_y = bk->y;
        }
    }
    out->count = nbox;
    out->block_count = nb;
    out->truncated = t->ctrunc;
}

/* ---------------------------------------------------------------- the calls */

static void clear(VocrTextspotSet *out)
{
    out->count = out->block_count = out->truncated = 0;
    out->pick = -1;
}

static int spot(const vocr_textspot *m, const Src *s, unsigned fw, unsigned fh, unsigned bpp, unsigned flags,
                void *arena, size_t arena_bytes, VocrTextspotSet *out, vocr_textspot_map_row hook, void *user)
{
    Plan p;
    Net t;
    unsigned y, k;
    int rc;
    clear(out);
    if (!loaded(m) || !side_ok(fw, fh) || (bpp != VOCR_LINES_GREY && bpp != VOCR_LINES_RGBA) || !arena ||
        (flags & ~VOCR_TEXTSPOT_FLAGS))
        return VOCR_LINES_E_ARG;
    plan(m, fw, &p);
    if (arena_bytes < p.total + 7u) return VOCR_LINES_E_ARENA;
    t.m = m;
    t.base = (unsigned char *)arena + ((8u - (unsigned)((uintptr_t)arena & 7u)) & 7u);
    t.f = p.f;
    t.fw = fw;
    t.fh = fh;
    t.width = p.win;
    t.height = p.f == 2u ? (fh + 1u) / 2u : fh;
    t.hook = hook;
    t.user = user;
    setup(&t, &p);
    for (y = 0; y < fh; y++) {
        rc = feed(&t, s, y, bpp);
        if (rc) return rc;
    }
    sweep(&t, NONE);   /* the components still open after the last map row */
    group(&t, out);
    if (flags & VOCR_TEXTSPOT_NO_DARK) return VOCR_LINES_OK;
    for (k = 0; k < out->count; k++) {
        VocrTextspotBox *b = &out->boxes[k];
        int d = s->im ? vocr_line_dark(s->im, b->x, b->y, b->w, b->h)
                      : vocr_line_dark_rows(s->rows, b->x, b->y, b->w, b->h);
        if (d < 0) {
            clear(out);
            return d;
        }
        b->dark = (unsigned char)d;
    }
    return VOCR_LINES_OK;
}

static int image_ok(const VocrLinesImage *im)
{
    return im && im->pixels && (im->bpp == VOCR_LINES_GREY || im->bpp == VOCR_LINES_RGBA) &&
           im->stride >= (size_t)im->width * im->bpp;
}

int vocr_textspot_find(const vocr_textspot *m, const VocrLinesImage *im, unsigned flags, void *arena,
                       size_t arena_bytes, VocrTextspotSet *out)
{
    Src s;
    if (!out) return VOCR_LINES_E_ARG;
    if (!image_ok(im)) {
        clear(out);
        return VOCR_LINES_E_ARG;
    }
    s.im = im;
    s.rows = NULL;
    return spot(m, &s, im->width, im->height, im->bpp, flags, arena, arena_bytes, out, NULL, NULL);
}

int vocr_textspot_find_rows(const vocr_textspot *m, const VocrLinesRows *src, unsigned flags, void *arena,
                            size_t arena_bytes, VocrTextspotSet *out)
{
    Src s;
    if (!out) return VOCR_LINES_E_ARG;
    if (!src || !src->row) {
        clear(out);
        return VOCR_LINES_E_ARG;
    }
    s.im = NULL;
    s.rows = src;
    return spot(m, &s, src->width, src->height, src->bpp, flags, arena, arena_bytes, out, NULL, NULL);
}

int vocr_textspot_maps(const vocr_textspot *m, const VocrLinesImage *im, void *arena, size_t arena_bytes,
                       vocr_textspot_map_row row, void *user)
{
    Src s;
    VocrTextspotSet out;
    if (!image_ok(im) || !row) return VOCR_LINES_E_ARG;
    s.im = im;
    s.rows = NULL;
    return spot(m, &s, im->width, im->height, im->bpp, VOCR_TEXTSPOT_NO_DARK, arena, arena_bytes, &out, row, user);
}
