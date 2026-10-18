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
 * Streaming. Rows arrive top to bottom. Every 3x3 layer keeps a ring of the 2 d + 1 input rows its
 * kernel spans, a pixel's channels side by side, with d zero columns on either side (zeroed once). A
 * 3x3 layer computes an output row once the last input row it reads is in (or lies below the input),
 * one pixel at a time: the pixel's channels, then the 1x1 layers after it on that pixel, into the next
 * 3x3 layer's newest ring row or, after the last layer, into the gate's sink (region mode: the k
 * largest codes so far; boxes mode: the stored K map). The deepest layer that can produce a row always
 * goes first, so a ring row is overwritten only when no output row still reads it: when a layer
 * receives input row y, it has produced every row whose kernel ends above y, and those left read rows
 * >= y - 2 d, inside the ring's 2 d + 1 newest rows.
 *
 * Integer ranges (VOCR_TEXTGATE_MAX_SIDE = 2048):
 *  - a sum has at most 9 * 64 = 576 products of two int8 codes, |w x| <= 2^14, so |sum| < 2^24 fits
 *    int32. bias + sum is formed in int64: below 2^31 + 2^24 in magnitude, times a multiplier below 2^31
 *    gives less than 2^62.01, and 2^(shift - 1) <= 2^61 keeps the sum below 2^63: exact. floor(v / 2^sh)
 *    is ~(~v >> sh) for v < 0, so no negative value is shifted.
 *  - a ring is at most 17 rows x 64 channels x (2048 + 16) bytes < 2^22; 32 of them, 40 bytes per layer
 *    and a K map of at most 512 x 512 cells stay below 2^28 bytes, so every size and offset fits a
 *    32-bit size_t and uint32; widths, heights and row counts fit uint16.
 *  - a box [x, x + w) meets a cell's block [a, a + s) iff a >= x ? a - x < w : x - a < s, for every
 *    unsigned box (no sum can wrap); a box with w or h 0 meets nothing (eval/textgate.box_mask).
 *
 * Memory: no allocation; the caller's arena holds a 40-byte record per layer, the K map (boxes mode)
 * and the rings; a run's state (vocr_textgate_state) is the caller's. libc: memset and memcpy only
 * (and the compiler's division helper for the ring index on 32-bit ARM).
 */
#include "vocr_textgate.h"

#include <stdint.h>
#include <string.h>

#define HEADER 64u
#define RECORD 32u
#define MAGIC 0x54445456u   /* "VTDT" */

/* A layer as a run uses it (decoded from its record), in the arena: fixed-width fields only, so the
 * arena has the same size on every platform. */
typedef struct {
    uint32_t w, b, m, sh;   /* blob offsets of the weights, biases, multipliers and shifts */
    uint32_t ring;          /* 3x3 layers: arena offset of the input ring */
    uint16_t iw, ih;        /* input width and height */
    uint16_t got, done;     /* 3x3 layers: input rows received, output rows produced */
    uint16_t cin, cout;     /* cout: the channels computed (1 for the last layer) */
    uint8_t dw, k, s, d, relu;
} Lay;

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

static void decode(const unsigned char *blob, unsigned l, Lay *L)
{
    const unsigned char *r = blob + HEADER + RECORD * l;
    memset(L, 0, sizeof(*L));
    L->dw = r[0];
    L->k = r[1];
    L->s = r[2];
    L->d = r[3];
    L->relu = r[4];
    L->cin = (uint16_t)rd16(r + 8);
    L->cout = (uint16_t)rd16(r + 10);
    L->w = rd32(r + 12);
    L->b = rd32(r + 16);
    L->m = rd32(r + 20);
    L->sh = rd32(r + 24);
}

/* ---------------------------------------------------------------- the blob */

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

/* [off, off + size) inside the data part, off 4-byte aligned */
static int bad_section(uint32_t off, uint32_t size, uint32_t data, uint32_t total)
{
    return (off & 3u) || off < data || off > total || size > total - off;
}

/* Every check of synth/textdet_twin.Model. */
static int parse(const unsigned char *b, size_t bytes)
{
    uint32_t total, n, l, c, cin = 1, stride = 1, data, relu = 1;
    if (bytes < HEADER || rd32(b) != MAGIC || rd32(b + 4) != 1u) return 1;
    total = rd32(b + 8);
    if ((size_t)total != bytes || crc32_ieee(b + 16, bytes - 16u) != rd32(b + 12)) return 1;
    n = rd16(b + 16);
    if ((b[18] != 4u && b[18] != 8u) || b[19] != 4u || b[20] != 2u || b[21] - 1u > 3u || n - 2u > 30u ||
        HEADER + n * RECORD > total)
        return 1;
    for (c = 32; c < HEADER; c++)   /* the name: ASCII (NUL padding included) */
        if (b[c] & 0x80u) return 1;
    data = HEADER + n * RECORD;
    for (l = 0; l < n; l++) {
        Lay L;
        decode(b, l, &L);
        if (L.dw > 1u || (L.k != 1u && L.k != 3u) || L.s - 1u > 1u || L.d - 1u > 7u || L.relu > 1u ||
            L.cin != cin || L.cout - 1u > 63u || (L.dw && (L.cout != L.cin || L.k != 3u)) ||
            (L.s == 2u && (L.k != 3u || L.d != 1u)) ||
            bad_section(L.w, (uint32_t)L.cout * (L.dw ? 1u : L.cin) * L.k * L.k, data, total) ||
            bad_section(L.b, 4u * L.cout, data, total) || bad_section(L.m, 4u * L.cout, data, total) ||
            bad_section(L.sh, L.cout, data, total))
            return 1;
        for (c = 0; c < L.cout; c++)
            if (rds32(b + L.m + 4u * c) <= 0 || b[L.sh + c] - 1u > 61u) return 1;
        stride *= L.s;
        cin = L.cout;
        relu = L.relu;
    }
    return stride != b[18] || cin != 4u || relu;
}

int vocr_textgate_load(vocr_textgate_model *m, const void *blob, size_t bytes)
{
    const unsigned char *b = (const unsigned char *)blob;
    if (!m) return VOCR_LINES_E_ARG;
    memset(m, 0, sizeof(*m));
    if (!b) return VOCR_LINES_E_ARG;
    if (parse(b, bytes)) return VOCR_TEXTGATE_E_BLOB;
    m->blob = b;
    m->layers = rd16(b + 16);
    m->shift = b[18] == 8u ? 3u : 2u;
    m->bytes = (unsigned long)bytes;
    return VOCR_LINES_OK;
}

/* ---------------------------------------------------------------- arena */

static unsigned cells(unsigned n, unsigned shift)   /* ceil(n / 2^shift), n >= 1 */
{
    return ((n - 1u) >> shift) + 1u;
}

/* The arena layout from the aligned base: the layer records, the K map (boxes mode), the rings. Fills
 * the records when lays is given. */
static size_t plan(const vocr_textgate_model *g, int mode, unsigned w, unsigned h, Lay *lays)
{
    unsigned l, n = g->layers;
    size_t off = n * sizeof(Lay);
    if (mode) off += (size_t)cells(w, g->shift) * cells(h, g->shift);
    for (l = 0; l < n; l++) {
        Lay L;
        decode(g->blob, l, &L);
        L.iw = (uint16_t)w;
        L.ih = (uint16_t)h;
        if (L.k == 3u) {
            L.ring = (uint32_t)off;
            off += (size_t)(2u * L.d + 1u) * L.cin * (w + 2u * L.d);
        }
        if (l + 1u == n) L.cout = 1;
        if (lays) lays[l] = L;
        w = (w + L.s - 1u) >> (L.s - 1u);
        h = (h + L.s - 1u) >> (L.s - 1u);
    }
    return off;
}

size_t vocr_textgate_arena_bytes(const vocr_textgate_model *m, int mode, unsigned max_width,
                                 unsigned max_height)
{
    if (!m || !m->layers || (unsigned)mode > 1u || !max_width || !max_height ||
        max_width > VOCR_TEXTGATE_MAX_SIDE || max_height > VOCR_TEXTGATE_MAX_SIDE)
        return 0;
    return plan(m, mode, max_width, max_height, NULL) + 3u;
}

/* ---------------------------------------------------------------- network */

static Lay *lays_of(const vocr_textgate_state *st)
{
    return (Lay *)(void *)st->base;
}

static signed char requant(const unsigned char *blob, const Lay *L, unsigned c, int32_t sum)
{
    unsigned sh = blob[L->sh + c];
    int64_t v = ((int64_t)rds32(blob + L->b + 4u * c) + sum) * rds32(blob + L->m + 4u * c) +
                ((int64_t)1 << (sh - 1u));
    int lo = L->relu ? 0 : -128;
    v = v < 0 ? ~(~v >> sh) : v >> sh;
    return (signed char)(v > 127 ? 127 : v < lo ? lo : v);
}

/* One output pixel of layer L into out (its computed channels). at[ky]: the input row of kernel row ky
 * (NULL where it lies outside the input), off bytes on to the pixel its first tap reads; a pixel's
 * channels are side by side, the next tap d pixels on. A 1x1 layer reads at[0] + off. */
static void pixel(const unsigned char *blob, const Lay *L, const signed char *const *at, size_t off,
                  signed char *out)
{
    const signed char *w = (const signed char *)(const void *)(blob + L->w);
    unsigned c, q, ky, kx, k = L->k, cin = L->cin, step = L->d * cin;
    for (c = 0; c < L->cout; c++) {
        int32_t sum = 0;
        for (q = L->dw ? c : 0; q < (L->dw ? c + 1u : cin); q++) {
            const signed char *wq = w + k * k * (L->dw ? c : c * cin + q);
            for (ky = 0; ky < k; ky++, wq += k) {
                const signed char *x = at[ky];
                if (!x) continue;
                x += off + q;
                for (kx = 0; kx < k; kx++, x += step) sum += wq[kx] * *x;
            }
        }
        out[c] = requant(blob, L, c, sum);
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

/* The layer after the 1x1 chain starting at L (the next 3x3 layer, or the end) and, for a 3x3 layer, its
 * newest ring row. */
static signed char *dest(const vocr_textgate_state *st, Lay *L, Lay **end)
{
    Lay *e = lays_of(st) + st->m->layers;
    while (L < e && L->k == 1u) L++;
    *end = L;
    if (L == e) return NULL;
    return (signed char *)(void *)(st->base + L->ring +
                                   (size_t)(L->got % (2u * L->d + 1u)) * L->cin * (L->iw + 2u * L->d));
}

/* Pixel j of a row (v: its codes, the input of layer L) through the 1x1 layers from L to e, then into
 * e's ring row dst, or (dst NULL) the sink, map row i. */
static void emit(vocr_textgate_state *st, const Lay *L, const Lay *e, signed char *dst, unsigned i,
                 unsigned j, signed char *v)
{
    const unsigned char *blob = st->m->blob;
    for (; L < e; L++) {
        signed char t[64];
        const signed char *p = v;
        pixel(blob, L, &p, 0, t);
        memcpy(v, t, L->cout);
    }
    if (dst)
        memcpy(dst + (size_t)(e->d + j) * e->cin, v, e->cin);
    else if (st->mode)
        st->base[sizeof(Lay) * st->m->layers + (size_t)i * st->wm + j] = (unsigned char)v[0];
    else
        put(st, v[0]);
}

/* The next output row of the 3x3 layer L. */
static void produce(vocr_textgate_state *st, Lay *L)
{
    const signed char *at[3];
    Lay *e;
    signed char *dst = dest(st, L + 1, &e);
    unsigned ky, j, i = L->done, s = L->s, d = L->d, wo = (L->iw + s - 1u) >> (s - 1u);
    for (ky = 0; ky < 3u; ky++) {
        unsigned y = i * s + ky * d - d;   /* wraps above row 0 */
        at[ky] = y < L->ih ? (const signed char *)(const void *)(st->base + L->ring + (size_t)(y % (2u * d + 1u)) *
                                                                  L->cin * (L->iw + 2u * d))
                           : NULL;
    }
    for (j = 0; j < wo; j++) {
        signed char v[64];
        pixel(st->m->blob, L, at, (size_t)j * s * L->cin, v);
        emit(st, L + 1, e, dst, i, j, v);
    }
    L->done++;
    if (dst) e->got++;
}

/* Every row that can be computed, the deepest layer first. */
static void drain(vocr_textgate_state *st)
{
    Lay *a = lays_of(st);
    unsigned l;
    for (;;) {
        for (l = st->m->layers; l-- > 0;) {
            Lay *L = a + l;
            unsigned need = (unsigned)L->done * L->s + L->d + 1u;
            if (L->k == 3u && L->done < (L->ih + L->s - 1u) >> (L->s - 1u) && L->got >= (need < L->ih ? need : L->ih))
                break;
        }
        if (l >= st->m->layers) return;
        produce(st, a + l);
    }
}

/* Region row y (st->bpp bytes a pixel) into the network. */
static int feed(vocr_textgate_state *st, const unsigned char *row)
{
    Lay *e;
    signed char *dst;
    unsigned x;
    if (st->rc) return st->rc;
    if (!row || st->rows >= st->height) return st->rc = VOCR_LINES_E_ARG;
    dst = dest(st, lays_of(st), &e);
    for (x = 0; x < st->width; x++) {
        const unsigned char *p = row + (size_t)x * st->bpp;
        unsigned g = st->bpp == 1u ? p[0] : (77u * p[0] + 150u * p[1] + 29u * p[2]) >> 8;
        signed char v[64];
        v[0] = (signed char)((int)g - 128);
        emit(st, lays_of(st), e, dst, 0, x, v);
    }
    e->got++;
    st->rows++;
    drain(st);
    return VOCR_LINES_OK;
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
    plan(m, mode, width, height, lays_of(st));
    st->m = m;
    st->mode = (unsigned char)mode;
    st->k = (unsigned char)k;
    st->bpp = 1;
    st->width = (unsigned short)width;
    st->height = (unsigned short)height;
    st->wm = (unsigned short)cells(width, m->shift);
    st->hm = (unsigned short)cells(height, m->shift);
    return st->rc = VOCR_LINES_OK;
}

int vocr_textgate_push(vocr_textgate_state *st, const unsigned char *grey_row)
{
    if (!st) return VOCR_LINES_E_ARG;
    return feed(st, grey_row);
}

/* a >= x ? a - x < w : x - a < s: the block [a, a + s) meets [x, x + w) (w >= 1) */
static int meets(unsigned a, unsigned s, unsigned x, unsigned w)
{
    return a >= x ? a - x < w : x - a < s;
}

int vocr_textgate_score(vocr_textgate_state *st, const VocrLineSet *boxes, int *score)
{
    if (!st || !score) return VOCR_LINES_E_ARG;
    if (st->rc) return st->rc;
    if (st->rows < st->height) return VOCR_LINES_E_ARG;
    if (st->mode) {
        const signed char *km = vocr_textgate_kmap(st, NULL, NULL);
        unsigned i, j, b, s = 1u << st->m->shift;
        if (!boxes || boxes->count > VOCR_LINES_MAX) return VOCR_LINES_E_ARG;
        st->count = 0;
        for (i = 0; i < st->hm; i++)
            for (j = 0; j < st->wm; j++)
                for (b = 0; b < boxes->count; b++) {
                    const VocrLine *ln = &boxes->lines[b];
                    if (ln->w && ln->h && meets(i * s, s, ln->y, ln->h) && meets(j * s, s, ln->x, ln->w)) {
                        put(st, km[(size_t)i * st->wm + j]);
                        break;
                    }
                }
    }
    *score = st->count ? st->top[st->count - 1u] : VOCR_TEXTGATE_NO_CELLS;
    return VOCR_LINES_OK;
}

const signed char *vocr_textgate_kmap(const vocr_textgate_state *st, unsigned *wm, unsigned *hm)
{
    if (!st || st->rc || st->mode != VOCR_TEXTGATE_BOXES || st->rows < st->height) return NULL;
    if (wm) *wm = st->wm;
    if (hm) *hm = st->hm;
    return (const signed char *)(const void *)(st->base + sizeof(Lay) * st->m->layers);
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
    st.bpp = (unsigned char)bpp;
    for (y = 0; !rc && y < h; y++) {
        const unsigned char *row = src ? src->row(src->user, y) : im->pixels + (size_t)y * im->stride;
        rc = row ? feed(&st, row) : VOCR_LINES_E_SOURCE;
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
