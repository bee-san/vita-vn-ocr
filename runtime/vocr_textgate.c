/* vocr_textgate.c - the text-presence gate (vocr_textgate.h, docs/textgate.md).
 *
 * eval/textgate.py is the specification (kmap = synth/textdet_twin.run_net(model, grey)[0]; kth,
 * box_mask, score); this file computes exactly what it computes, for every region and every blob that
 * textdet_twin.Model accepts (tests/textgate_parity.py compares K maps and scores). In short:
 *
 * Network (textdet_twin.run_layer). x0 = grey - 128. A layer (dense or depthwise; 1x1, or 3x3 with a
 * stride s of 1 or 2 and a dilation d, zero padding of d) sums int8 weights times int8 codes into an
 * int32 sum and requantises with its per-channel int32 bias and multiplier and u8 shift:
 *     out = clamp(floor(((bias + sum) * mult + 2^(shift - 1)) / 2^shift), relu ? 0 : -128, 127).
 * Only channel 0 of the last layer, K, is computed.
 *
 * Streaming. Rows arrive top to bottom. Every layer keeps a ring of the (k - 1) d + 1 input rows its
 * kernel spans (one row for a 1x1 layer), a pixel's channels side by side, with p = (k - 1) d / 2 zero
 * columns on either side (zeroed once). A layer computes its next output row once the last input row
 * it reads is in (or lies below the input), into the next layer's newest ring row or, for the last
 * layer, into the K row (region mode; its codes then go into the list of the k largest) or the K map
 * (boxes mode). The deepest layer that can produce a row always goes first, so a ring row is
 * overwritten only when no output row still reads it: when a layer receives input row y, it has
 * produced every row whose kernel ends above y, and those left read rows >= y + 1 - R (R the ring).
 * Layer parameters are read from the blob's records; the arena holds 20 bytes of run state per layer.
 *
 * Integer ranges (VOCR_TEXTGATE_MAX_SIDE = 2048):
 *  - a sum has at most 9 * 64 = 576 products of two int8 codes, |w x| <= 2^14, so |sum| < 2^24 fits
 *    int32. bias + sum is formed in int64: below 2^31 + 2^24 in magnitude, times a multiplier below 2^31
 *    gives less than 2^62.01, and 2^(shift - 1) <= 2^61 keeps the sum below 2^63: exact. floor(v / 2^sh)
 *    is ~(~v >> sh) for v < 0, so no negative value is shifted.
 *  - a ring is at most 17 rows x 64 channels x (2048 + 16) bytes < 2^22; 32 of them, 20 bytes per layer
 *    and a K map of at most 512 x 512 cells stay below 2^28 bytes, so every size and offset fits a
 *    32-bit size_t and uint32; widths, heights and row counts fit uint16.
 *  - a box [x, x + w) meets a cell's block [a, a + s) iff a >= x ? a - x < w : x - a < s and w > 0, for
 *    every unsigned box (no sum can wrap); so a box with w or h 0 meets nothing (eval/textgate.box_mask).
 *
 * Memory: no allocation; the caller's arena holds the run state, the K row or map and the rings; a
 * run's state (vocr_textgate_state) is the caller's. libc: memset only (and the compiler's division
 * helper for the ring index on 32-bit ARM).
 */
#include "vocr_textgate.h"

#include <stdint.h>
#include <string.h>

#define HEADER 64u
#define RECORD 32u
#define MAGIC 0x54445456u   /* "VTDT" */

/* The kernel stays a function of its own: inlined into the row pipeline it runs out of registers (gcc
 * -Os, Thumb-2), which costs bytes and time. */
#if defined(__GNUC__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif

typedef struct {
    uint32_t ring, row;     /* arena offset of the layer's input ring, bytes per ring row */
    uint16_t iw, ih;        /* input width and height */
    uint16_t got, done;     /* input rows received, output rows produced */
    uint16_t p, pad;        /* zero columns either side, and their bytes */
} Run;

static uint32_t rd16(const unsigned char *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8;
}

static uint32_t rd32(const unsigned char *p)
{
    return rd16(p) | rd16(p + 2) << 16;
}

static int32_t rds32(const unsigned char *p)
{
    uint32_t v = rd32(p);
    return v < 0x80000000u ? (int32_t)v : -(int32_t)(0xFFFFFFFFu - v) - 1;
}

/* Layer records: u8 kind (1 depthwise), k, stride, dilation, relu, 3 reserved; u16 cin, cout; u32 offsets
 * of the weights, biases, multipliers, shifts; u32 reserved (synth/textdet_twin.py). */
static const unsigned char *rec(const unsigned char *blob, unsigned l)
{
    return blob + HEADER + RECORD * l;
}

/* ---------------------------------------------------------------- the blob */

/* Every check of synth/textdet_twin.Model. */
static int bad_blob(const unsigned char *b, size_t bytes)
{
    uint32_t total, n, l, c, cin = 1, stride = 1, data, last = 1;
    if (bytes < HEADER || rd32(b) != MAGIC || rd32(b + 4) != 1u) return 1;
    total = rd32(b + 8);
    if ((size_t)total != bytes) return 1;
    {   /* CRC-32 of [16, total) */
        uint32_t crc = 0xFFFFFFFFu;
        const unsigned char *p = b + 16;
        while (p < b + total) {
            int k;
            crc ^= *p++;
            for (k = 0; k < 8; k++) crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
        }
        if (~crc != rd32(b + 12)) return 1;
    }
    n = rd16(b + 16);
    /* stride 4 or 8, 4 outputs; A scale 2, min cells 1..4; 2..32 layers */
    if (((rd16(b + 18) - 0x0404u) & ~4u) || ((rd16(b + 20) - 0x0102u) & ~0x300u) || n - 2u > 30u ||
        HEADER + n * RECORD > total)
        return 1;
    for (c = 32; c < HEADER; c++)   /* the name: ASCII (NUL padding included) */
        if (b[c] & 0x80u) return 1;
    data = HEADER + n * RECORD;
    for (l = 0; l < n; l++) {
        const unsigned char *r = rec(b, l);
        uint32_t dw = r[0], k = r[1], s = r[2], cout = rd16(r + 10);
        if ((dw | r[4]) > 1u || (k | 2u) != 3u || s - 1u > 1u || r[3] - 1u > 7u || rd16(r + 8) != cin ||
            cout - 1u > 63u || (dw && (cout != cin || k != 3u)) || (s == 2u && (k != 3u || r[3] != 1u)))
            return 1;
        for (c = 0; c < 4u; c++) {   /* weights, biases, multipliers, shifts: aligned, after the records */
            uint32_t off = rd32(r + 12u + 4u * c);
            uint32_t size = c ? (c < 3u ? 4u : 1u) * cout : cout * (dw ? 1u : cin) * k * k;
            if ((off & 3u) || off < data || off > total || size > total - off) return 1;
        }
        for (c = 0; c < cout; c++)
            if (rds32(b + rd32(r + 20) + 4u * c) <= 0 || b[rd32(r + 24) + c] - 1u > 61u) return 1;
        stride *= s;
        cin = cout;
        last = r[4];
    }
    return stride != b[18] || cin != 4u || last;
}

int vocr_textgate_load(vocr_textgate_model *m, const void *blob, size_t bytes)
{
    if (!m) return VOCR_LINES_E_ARG;
    m->blob = NULL;
    if (!blob) return VOCR_LINES_E_ARG;
    if (bad_blob((const unsigned char *)blob, bytes)) return VOCR_TEXTGATE_E_BLOB;
    m->blob = (const unsigned char *)blob;
    return VOCR_LINES_OK;
}

/* ---------------------------------------------------------------- arena */

static unsigned shift_of(const unsigned char *blob)   /* log2 of the output stride */
{
    return blob[18] == 8u ? 3u : 2u;
}

static unsigned cells(unsigned n, unsigned shift)   /* ceil(n / 2^shift), n >= 1 */
{
    return ((n - 1u) >> shift) + 1u;
}

/* The arena layout from the aligned base: the run state, the K row (region mode) or map (boxes mode), the
 * rings. Fills the run state when run is given. */
static size_t plan(const vocr_textgate_model *g, int mode, unsigned w, unsigned h, Run *run)
{
    unsigned l, n = rd16(g->blob + 16);
    size_t off = n * sizeof(Run);
    for (l = 0; l < n; l++) {
        const unsigned char *r = rec(g->blob, l);
        unsigned s = r[2], p = (r[1] >> 1) * r[3], row = rd16(r + 8) * (w + 2u * p);
        if (run) {
            run[l].ring = (uint32_t)off;
            run[l].row = row;
            run[l].iw = (uint16_t)w;
            run[l].ih = (uint16_t)h;
            run[l].p = (uint16_t)p;
            run[l].pad = (uint16_t)(p * rd16(r + 8));
        }
        off += (size_t)(2u * p + 1u) * row;
        w = (w + s - 1u) >> (s - 1u);
        h = (h + s - 1u) >> (s - 1u);
    }
    return off + (size_t)w * (mode ? h : 1u);   /* the K row or map, after the rings */
}

size_t vocr_textgate_arena_bytes(const vocr_textgate_model *m, int mode, unsigned max_width,
                                 unsigned max_height)
{
    if (!m || !m->blob || (unsigned)mode > 1u || !max_width || !max_height ||
        max_width > VOCR_TEXTGATE_MAX_SIDE || max_height > VOCR_TEXTGATE_MAX_SIDE)
        return 0;
    return plan(m, mode, max_width, max_height, NULL) + 3u;
}

/* ---------------------------------------------------------------- network */

static Run *run_of(const vocr_textgate_state *st)
{
    return (Run *)(void *)st->base;
}

static signed char *kbuf(const vocr_textgate_state *st)   /* the K row or map */
{
    const Run *u = run_of(st) + st->n - 1u;   /* after the last layer's ring */
    return (signed char *)(void *)(st->base + u->ring + (2u * u->p + 1u) * u->row);
}

/* Layer l's ring row holding its input row y, from its left padding on; with fill, the newest row (y is
 * then the rows received so far) past its left padding, and counted as received. */
static signed char *ring_row(vocr_textgate_state *st, unsigned l, unsigned y, int fill)
{
    Run *u = run_of(st) + l;
    signed char *row = (signed char *)(void *)(st->base + u->ring) + (size_t)(y % (2u * u->p + 1u)) * u->row;
    if (!fill) return row;
    u->got++;
    return row + u->pad;
}

/* One output pixel of layer r into out (nc channels). tap[t]: the input the t-th kernel tap reads for the
 * row's first pixel (NULL where its row lies outside the input), a pixel's channels side by side; off: the
 * bytes on to this pixel. */
NOINLINE static void pixel(const unsigned char *blob, const unsigned char *r, const signed char *const *tap, size_t off,
                  signed char *out, unsigned nc)
{
    const signed char *w = (const signed char *)(const void *)(blob + rd32(r + 12));
    unsigned c, q, t, kk = r[1] * r[1];
    for (c = 0; c < nc; c++) {
        int32_t sum = 0, m = rds32(blob + rd32(r + 20) + 4u * c), lo;
        unsigned sh = blob[rd32(r + 24) + c] - 1u;
        int64_t v;
        for (q = 0; q < (r[0] ? 1u : rd16(r + 8)); q++) {
            size_t o = off + (r[0] ? c : q);
            for (t = 0; t < kk; t++, w++)
                if (tap[t]) sum += *w * tap[t][o];
        }
        /* v = floor((bias + sum) m / 2^(shift - 1)); floor((v + 1) / 2) is then the twin's
         * floor(((bias + sum) m + 2^(shift - 1)) / 2^shift), clamped through v in [2 lo - 1, 254] */
        v = (int64_t)rds32(blob + rd32(r + 16) + 4u * c) * m + (int64_t)sum * m;
        v = v < 0 ? ~(~v >> sh) : v >> sh;
        lo = 256 * r[4] - 257;   /* 2 lo - 1: -1 with ReLU, else -257 */
        if (v > 254) v = 254;
        if (v < lo) v = lo;
        out[c] = (signed char)((int)((v + 257) >> 1) - 128);
    }
}

/* The k largest codes so far, largest first. */
static void put(vocr_textgate_state *st, signed char v)
{
    unsigned n = st->count;
    if (n == st->k) {
        if (v <= st->top[n - 1u]) return;
        n--;
    } else {
        st->count = (unsigned char)(n + 1u);
    }
    for (; n && st->top[n - 1u] < v; n--) st->top[n] = st->top[n - 1u];
    st->top[n] = v;
}

/* The next output row of layer l: into layer l + 1's newest ring row, or the K row / map. */
static void produce(vocr_textgate_state *st, unsigned l)
{
    const unsigned char *blob = st->blob, *r = rec(blob, l);
    Run *u = run_of(st) + l;
    const signed char *tap[9];
    signed char *dst;
    unsigned k = r[1], s = r[2], d = r[3], p = u->p, cin = rd16(r + 8), nc = rd16(r + 10);
    unsigned i = u->done, wo = (u->iw + s - 1u) >> (s - 1u), ky, kx, j, t = 0;
    for (ky = 0; ky < k; ky++) {
        unsigned y = i * s + ky * d - p;   /* wraps above row 0 */
        const signed char *row = y < u->ih ? ring_row(st, l, y, 0) : NULL;
        for (kx = 0; kx < k; kx++) tap[t++] = row ? row + kx * d * cin : NULL;
    }
    if (l + 1u < st->n) {
        dst = ring_row(st, l + 1u, u[1].got, 1);
    } else {
        nc = 1;
        dst = kbuf(st) + (st->mode ? (size_t)i * wo : 0u);
    }
    for (j = 0; j < wo; j++) pixel(blob, r, tap, (size_t)j * s * cin, dst + (size_t)j * nc, nc);
    if (l + 1u == st->n && !st->mode)
        for (j = 0; j < wo; j++) put(st, dst[j]);
    u->done++;
}

/* Every row that can be computed, the deepest layer first. */
static void drain(vocr_textgate_state *st)
{
    const Run *u = run_of(st);
    unsigned l;
    for (;;) {
        for (l = st->n; l-- > 0;) {   /* output row i reads input rows up to i s + p: is the last one in? */
            unsigned t = (unsigned)u[l].done * rec(st->blob, l)[2];
            if (t < u[l].ih && (u[l].got == u[l].ih || u[l].got > t + u[l].p)) break;
        }
        if (l >= st->n) return;
        produce(st, l);
    }
}

/* The first layer's ring row for the region's next row (counted in), or NULL after an error or the last row
 * (VOCR_LINES_E_ARG then sticks). The caller writes the row's codes into it, then drains. */
static signed char *next_in(vocr_textgate_state *st)
{
    if (st->rows >= st->height && !st->rc) st->rc = VOCR_LINES_E_ARG;
    return st->rc ? NULL : ring_row(st, 0, st->rows++, 1);
}

/* ---------------------------------------------------------------- the push form */

int vocr_textgate_begin(vocr_textgate_state *st, const vocr_textgate_model *m, int mode, unsigned k,
                        unsigned width, unsigned height, void *arena, size_t arena_bytes)
{
    size_t need;
    if (!st) return VOCR_LINES_E_ARG;
    memset(st, 0, sizeof(*st));
    st->rc = VOCR_LINES_E_ARG;
    need = vocr_textgate_arena_bytes(m, mode, width, height);
    if (!need || !k || k > VOCR_TEXTGATE_MAX_K || !arena) return st->rc;
    if (arena_bytes < need) return st->rc = VOCR_LINES_E_ARENA;
    st->base = (unsigned char *)arena + ((0u - (uintptr_t)arena) & 3u);
    memset(st->base, 0, need - 3u);
    plan(m, mode, width, height, run_of(st));
    st->blob = m->blob;
    st->n = (unsigned char)rd16(m->blob + 16);
    st->mode = (unsigned char)mode;
    st->k = (unsigned char)k;
    st->width = (unsigned short)width;
    st->height = (unsigned short)height;
    return st->rc = VOCR_LINES_OK;
}

int vocr_textgate_push(vocr_textgate_state *st, const unsigned char *grey_row)
{
    signed char *dst;
    unsigned x;
    if (!st) return VOCR_LINES_E_ARG;
    if (!grey_row && !st->rc) st->rc = VOCR_LINES_E_ARG;
    dst = next_in(st);
    if (!dst) return st->rc;
    for (x = 0; x < st->width; x++) dst[x] = (signed char)((int)grey_row[x] - 128);
    drain(st);
    return VOCR_LINES_OK;
}

/* The block [a, a + s) meets [x, x + w): a >= x ? a - x < w : x - a < s and w > 0 (no sum can wrap). */
static int meets(unsigned a, unsigned s, unsigned x, unsigned w)
{
    return a >= x ? a - x < w : x - a < s && w;
}

int vocr_textgate_score_region(vocr_textgate_state *st, int *score)
{
    if (!st || !score || st->mode) return VOCR_LINES_E_ARG;
    if (st->rc) return st->rc;
    if (st->rows < st->height) return VOCR_LINES_E_ARG;
    *score = st->top[st->count - 1u];
    return VOCR_LINES_OK;
}

int vocr_textgate_score(vocr_textgate_state *st, const VocrLineSet *boxes, int *score)
{
    const signed char *km;
    const VocrLine *ln, *end;
    unsigned i, j, s;
    if (!st || !score) return VOCR_LINES_E_ARG;
    if (st->rc) return st->rc;
    if (st->rows < st->height) return VOCR_LINES_E_ARG;
    if (st->mode) {
        if (!boxes || boxes->count > VOCR_LINES_MAX) return VOCR_LINES_E_ARG;
        km = kbuf(st);
        s = 1u << shift_of(st->blob);
        end = boxes->lines + boxes->count;
        st->count = 0;
        for (i = 0; i < st->height; i += s)
            for (j = 0; j < st->width; j += s, km++)
                for (ln = boxes->lines; ln < end; ln++)
                    if (meets(i, s, ln->y, ln->h) && meets(j, s, ln->x, ln->w)) {
                        put(st, *km);
                        break;
                    }
    }
    *score = st->count ? st->top[st->count - 1u] : VOCR_TEXTGATE_NO_CELLS;
    return VOCR_LINES_OK;
}

const signed char *vocr_textgate_kmap(const vocr_textgate_state *st, unsigned *wm, unsigned *hm)
{
    if (!st || st->rc || st->mode != VOCR_TEXTGATE_BOXES || st->rows < st->height) return NULL;
    if (wm) *wm = cells(st->width, shift_of(st->blob));
    if (hm) *hm = cells(st->height, shift_of(st->blob));
    return kbuf(st);
}

/* ---------------------------------------------------------------- one-shot forms */

/* The gate on an image in memory (src NULL) or through a row source. */
static int gate(const vocr_textgate_model *m, int mode, unsigned k, const VocrLinesImage *im,
                const VocrLinesRows *src, const VocrLineSet *boxes, void *arena, size_t arena_bytes, int *score)
{
    vocr_textgate_state st;
    unsigned y, w, h, bpp;
    int rc;
    if (src) {
        if (!src->row) return VOCR_LINES_E_ARG;
        w = src->width;
        h = src->height;
        bpp = src->bpp;
    } else {
        if (!im || !im->pixels) return VOCR_LINES_E_ARG;
        w = im->width;
        h = im->height;
        bpp = im->bpp;
        if (im->stride < (size_t)w * bpp) return VOCR_LINES_E_ARG;
    }
    if (bpp != VOCR_LINES_GREY && bpp != VOCR_LINES_RGBA) return VOCR_LINES_E_ARG;
    rc = vocr_textgate_begin(&st, m, mode, k, w, h, arena, arena_bytes);
    for (y = 0; !rc && y < h; y++) {
        const unsigned char *row = src ? src->row(src->user, y) : im->pixels + (size_t)y * im->stride;
        signed char *dst;
        unsigned x;
        if (!row) return VOCR_LINES_E_SOURCE;
        dst = next_in(&st);
        for (x = 0; x < w; x++, row += bpp)   /* grey = (77 R + 150 G + 29 B) >> 8 */
            dst[x] = (signed char)((int)(bpp == 1u ? row[0] : (77u * row[0] + 150u * row[1] + 29u * row[2]) >> 8) - 128);
        drain(&st);
    }
    return rc ? rc : vocr_textgate_score(&st, boxes, score);
}

int vocr_textgate_image(const vocr_textgate_model *m, int mode, unsigned k, const VocrLinesImage *im,
                        const VocrLineSet *boxes, void *arena, size_t arena_bytes, int *score)
{
    return gate(m, mode, k, im, NULL, boxes, arena, arena_bytes, score);
}

int vocr_textgate_rows(const vocr_textgate_model *m, int mode, unsigned k, const VocrLinesRows *src,
                       const VocrLineSet *boxes, void *arena, size_t arena_bytes, int *score)
{
    return gate(m, mode, k, NULL, src, boxes, arena, arena_bytes, score);
}

/* ---------------------------------------------------------------- the finder mode */

size_t vocr_lines_gated_arena_bytes(const vocr_textgate_model *g, int mode, unsigned max_width,
                                    unsigned max_height)
{
    size_t a = vocr_lines_arena_bytes(max_width, max_height);
    size_t b = g ? vocr_textgate_arena_bytes(g, mode, max_width, max_height) : 0;
    return a > b ? a : b;
}

static int gated(const VocrLinesImage *im, const VocrLinesRows *src, int finder, const vocr_textgate_model *g,
                 int mode, unsigned k, int t, void *arena, size_t arena_bytes, VocrLineSet *out)
{
    int rc, s;
    if (!out) return VOCR_LINES_E_ARG;
    out->count = out->truncated = 0;
    if (!(src || im) || (unsigned)finder > VOCR_LINES_TEXTBOX_RECOVERY) return VOCR_LINES_E_ARG;
    if (arena_bytes < vocr_lines_gated_arena_bytes(g, mode, src ? src->width : im->width,
                                                   src ? src->height : im->height))
        return VOCR_LINES_E_ARENA;
    if (mode == VOCR_TEXTGATE_REGION) {
        rc = gate(g, mode, k, im, src, NULL, arena, arena_bytes, &s);
        if (rc || s < t) return rc;
    }
    rc = src ? vocr_find_lines_rows(src, finder, arena, arena_bytes, out)
             : vocr_find_lines(im, finder, arena, arena_bytes, out);
    if (!rc && mode != VOCR_TEXTGATE_REGION) {
        rc = gate(g, mode, k, im, src, out, arena, arena_bytes, &s);
        if (!rc && s < t) out->count = 0;
    }
    if (rc) out->count = 0;
    if (!out->count) out->truncated = 0;
    return rc;
}

int vocr_find_lines_gated(const VocrLinesImage *im, int finder, const vocr_textgate_model *g, int mode,
                          unsigned k, int t, void *arena, size_t arena_bytes, VocrLineSet *out)
{
    if (!g) return vocr_find_lines(im, finder, arena, arena_bytes, out);
    return gated(im, NULL, finder, g, mode, k, t, arena, arena_bytes, out);
}

int vocr_find_lines_gated_rows(const VocrLinesRows *src, int finder, const vocr_textgate_model *g,
                               int mode, unsigned k, int t, void *arena, size_t arena_bytes,
                               VocrLineSet *out)
{
    if (!g) return vocr_find_lines_rows(src, finder, arena, arena_bytes, out);
    return gated(NULL, src, finder, g, mode, k, t, arena, arena_bytes, out);
}
