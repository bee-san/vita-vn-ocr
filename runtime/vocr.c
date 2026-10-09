/* vocr.c - see vocr.h. C99; libc use is limited to memcpy, memset and math.h. */
#include "vocr.h"

#include <math.h>
#include <string.h>

#if defined(__ARM_NEON) && !defined(VOCR_NO_NEON)
#include <arm_neon.h>
#define VOCR_NEON 1
#endif
#if defined(__clang__)
#pragma STDC FP_CONTRACT OFF
#endif

enum { T_CONV = 1, T_DW = 2, T_PW = 3, T_HEAD = 4 };
enum { F_RELU = 1, F_RES = 2, F_BIAS = 4, F_INT8 = 8, F_COLLAPSE = 16 };
enum { DST_RING = 1, DST_TRANS = 2, DST_ACC = 3 };
#define HEADER_BYTES 64u
#define OP_BYTES 64u
#define MIN_WIDTH 8u
#define W8I_MAX_K 516u /* 516 * 127 * 32767 < 2^31: int32 accumulators cannot overflow */

static unsigned rd16(const unsigned char *p) { return (unsigned)p[0] | (unsigned)p[1] << 8; }
static uint32_t rd32(const unsigned char *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static size_t al16(size_t n) { return (n + 15u) & ~(size_t)15u; }
static int is_magic(const unsigned char *b)
{
    return b[0] == 'V' && b[1] == 'O' && b[2] == 'C' && b[3] == 'R';
}
static unsigned umin(unsigned a, unsigned b) { return a < b ? a : b; }

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

int vocr_check_crc(const void *blob, size_t bytes)
{
    const unsigned char *b = (const unsigned char *)blob;
    uint32_t total;
    if (!b || bytes < HEADER_BYTES || !is_magic(b)) return VOCR_E_FORMAT;
    total = rd32(b + 8);
    if (total < HEADER_BYTES || total > bytes) return VOCR_E_FORMAT;
    return crc32_ieee(b + HEADER_BYTES, total - HEADER_BYTES) == rd32(b + 40) ? VOCR_OK
                                                                               : VOCR_E_FORMAT;
}

/* [off, off + n) lies inside the blob and off is 16-byte aligned */
static int in_blob(uint32_t total, uint32_t off, uint32_t n)
{
    return off >= HEADER_BYTES && !(off & 15u) && off <= total && n <= total - off;
}

static int parse_op(vocr_model *m, unsigned i, const unsigned char *r, unsigned *rows,
                    unsigned *chans, uint32_t total)
{
    vocr_op *op = &m->op[i];
    const unsigned char *b = m->blob;
    uint32_t w_off = rd32(r + 16), w_bytes = rd32(r + 20), b_off = rd32(r + 24),
             s_off = rd32(r + 28);
    unsigned j, last = i + 1 == m->n_ops;
    op->type = r[0];
    op->flags = r[1];
    op->kh = r[2];
    op->kw = r[3];
    op->sh = r[4];
    op->sw = r[5];
    op->ph = r[6];
    op->pw = r[7];
    op->cin = (uint16_t)rd16(r + 8);
    op->cout = (uint16_t)rd16(r + 10);
    op->h_in = (uint16_t)rd16(r + 12);
    op->h_out = (uint16_t)rd16(r + 14);
    op->k = (uint16_t)rd16(r + 32);
    op->cout_pad = (uint16_t)rd16(r + 34);
    for (j = 36; j < OP_BYTES; j++)
        if (r[j]) return VOCR_E_FORMAT;
    if (op->type < T_CONV || op->type > T_HEAD || (op->flags & ~31u)) return VOCR_E_FORMAT;
    if ((op->type == T_HEAD) != (int)last) return VOCR_E_FORMAT;
    if (op->kh < 1 || op->kh > 7 || op->kw < 1 || op->kw > 7 || op->sh < 1 || op->sh > 2 ||
        op->sw < 1 || op->sw > 2 || op->ph >= op->kh || op->pw >= op->kw)
        return VOCR_E_FORMAT;
    if (op->h_in != *rows || op->h_in + 2u * op->ph < op->kh ||
        op->h_out != (op->h_in + 2u * op->ph - op->kh) / op->sh + 1u)
        return VOCR_E_FORMAT;
    if (!op->cout || (op->type != T_HEAD && op->cout > 1024) || op->cin != *chans)
        return VOCR_E_FORMAT;
    if (op->flags & F_RES) {
        const vocr_op *d = i ? &m->op[i - 1] : NULL;
        if (!d || d->type != T_DW || d->sh != 1 || d->sw != 1 || d->h_in != d->h_out ||
            op->type != T_PW || op->cin != op->cout || (op->flags & F_COLLAPSE))
            return VOCR_E_FORMAT;
    }
    if (op->type == T_CONV || op->type == T_DW) {
        if ((op->type == T_CONV) != (i == 0) || (op->type == T_CONV && op->cin != 1) ||
            (op->type == T_DW && op->cin != op->cout) || (op->flags & (F_INT8 | F_COLLAPSE)) ||
            op->k != op->kh * op->kw || op->cout_pad != op->cout ||
            w_bytes != 4u * op->k * op->cout)
            return VOCR_E_FORMAT;
    } else {
        unsigned collapse = (op->flags & F_COLLAPSE) != 0;
        if (op->sh != 1 || op->sw != 1 || op->ph || op->pw || op->kw != 1 ||
            op->kh != (collapse ? op->h_in : 1u) || (collapse && op->h_out != 1) ||
            op->k != (collapse ? op->h_in * op->cin : op->cin) || !op->k || op->k > 4096)
            return VOCR_E_FORMAT;
        if (op->type == T_HEAD ? (op->h_in != 1 || (op->flags & (F_RELU | F_RES | F_COLLAPSE)) ||
                                  op->cout != m->n_classes || op->cout_pad != ((op->cout + 7u) & ~7u))
                               : ((op->cout & 7u) || op->cout_pad != op->cout))
            return VOCR_E_FORMAT;
        if (w_bytes != (op->flags & F_INT8 ? 1u : 4u) * op->cout_pad * op->k) return VOCR_E_FORMAT;
        if (i == 0) return VOCR_E_FORMAT;
    }
    if (!in_blob(total, w_off, w_bytes)) return VOCR_E_FORMAT;
    if (op->flags & F_INT8) {
        if (!in_blob(total, s_off, 4u * op->cout_pad)) return VOCR_E_FORMAT;
        op->q = (const int8_t *)(const void *)(b + w_off);
        op->scale = (const float *)(const void *)(b + s_off);
    } else {
        if (s_off) return VOCR_E_FORMAT;
        op->w = (const float *)(const void *)(b + w_off);
    }
    if (op->flags & F_BIAS) {
        if (!in_blob(total, b_off, 4u * op->cout_pad)) return VOCR_E_FORMAT;
        op->bias = (const float *)(const void *)(b + b_off);
    } else if (b_off) {
        return VOCR_E_FORMAT;
    }
    *rows = op->h_out;
    *chans = op->cout;
    return VOCR_OK;
}

int vocr_load(vocr_model *m, const void *blob, size_t bytes)
{
    const unsigned char *b = (const unsigned char *)blob;
    const uint16_t one = 1;
    uint32_t total, ops_off, v_off, v_bytes;
    unsigned i, rows, chans = 1, stride = 1, n_int8 = 0, n_mat = 0;
    int rc;
    if (!m || !b) return VOCR_E_ARG;
    memset(m, 0, sizeof *m);
    if (*(const unsigned char *)&one != 1) return VOCR_E_FORMAT; /* the blob is little-endian */
    if ((uintptr_t)b & 15u) return VOCR_E_ALIGN;
    if (bytes < HEADER_BYTES || !is_magic(b) || rd16(b + 4) != VOCR_BLOB_VERSION ||
        rd16(b + 6) != HEADER_BYTES)
        return VOCR_E_FORMAT;
    total = rd32(b + 8);
    if (total < HEADER_BYTES || total > bytes || (rd32(b + 12) & ~1u)) return VOCR_E_FORMAT;
    m->blob = b;
    m->bytes = total;
    m->height = rd16(b + 16);
    m->max_width = rd16(b + 18);
    m->stride = rd16(b + 20);
    m->n_ops = rd16(b + 22);
    m->n_classes = rd32(b + 24);
    ops_off = rd32(b + 28);
    v_off = rd32(b + 32);
    v_bytes = rd32(b + 36);
    m->params = rd32(b + 44);
    if (!m->height || m->height > VOCR_MAX_HEIGHT || m->n_ops < 2 || m->n_ops > VOCR_MAX_OPS ||
        m->n_classes < 2 || m->n_classes > 65535u || !m->stride || m->stride > 64 ||
        m->max_width < MIN_WIDTH || m->max_width > 8192 || m->max_width % m->stride ||
        !in_blob(total, ops_off, OP_BYTES * m->n_ops))
        return VOCR_E_FORMAT;
    rows = m->height;
    for (i = 0; i < m->n_ops; i++) {
        rc = parse_op(m, i, b + ops_off + OP_BYTES * i, &rows, &chans, total);
        if (rc) return rc;
        stride *= m->op[i].sw;
        if (m->op[i].type >= T_PW) {
            n_mat++;
            n_int8 += (m->op[i].flags & F_INT8) != 0;
        }
    }
    if (rows != 1 || stride != m->stride || (n_int8 && n_int8 != n_mat) ||
        (n_int8 != 0) != ((rd32(b + 12) & 1u) != 0))
        return VOCR_E_FORMAT;
    m->int8 = n_int8 != 0;
    if (!in_blob(total, v_off, 2u * (m->n_classes + 1u)) ||
        v_bytes > total - v_off - 2u * (m->n_classes + 1u))
        return VOCR_E_FORMAT;
    m->vocab_off = (const uint16_t *)(const void *)(b + v_off);
    m->vocab = (const char *)(b + v_off + 2u * (m->n_classes + 1u));
    if (m->vocab_off[0] || m->vocab_off[1] || m->vocab_off[m->n_classes] != v_bytes)
        return VOCR_E_FORMAT;
    for (i = 0; i < m->n_classes; i++)
        if (m->vocab_off[i] > m->vocab_off[i + 1]) return VOCR_E_FORMAT;
    return VOCR_OK;
}

const char *vocr_class_text(const vocr_model *m, unsigned c, unsigned *len)
{
    if (!m || !m->vocab_off || c >= m->n_classes) {
        if (len) *len = 0;
        return "";
    }
    if (len) *len = (unsigned)(m->vocab_off[c + 1] - m->vocab_off[c]);
    return m->vocab_off[c + 1] == m->vocab_off[c] ? "" : m->vocab + m->vocab_off[c];
}

const char *vocr_strerror(int rc)
{
    switch (rc) {
    case VOCR_OK: return "ok";
    case VOCR_E_ARG: return "bad argument";
    case VOCR_E_FORMAT: return "bad or unsupported model blob";
    case VOCR_E_ALIGN: return "blob or arena not 16-byte aligned";
    case VOCR_E_ARENA: return "arena too small";
    case VOCR_E_WIDTH: return "line wider than the planned max width";
    case VOCR_E_TEXT: return "text buffer too small";
    case VOCR_E_MODE: return "mode not available for this blob";
    case VOCR_E_CANCELLED: return "cancelled";
    default: return "unknown error";
    }
}

/* ---------------------------------------------------------------- plan
 *
 * Every op computes a fixed range of its output columns per tile, a constant `lag`
 * behind the source, so each layer's input is a sliding window (a ring) holding the
 * new columns plus `halo` columns carried over from the previous tile. Columns
 * outside the line are written as zeros, which is exactly the convolution padding.
 * The leading CONV/DW ops (the "front") run one output column of the last front op
 * at a time, so their large high-resolution rings stay a few columns wide.
 * PW inputs are transient: produced and consumed inside one tile. The op feeding the
 * head appends its columns to an accumulator; the head runs once it holds
 * head_frames frames. Every byte the plan places is reported by vocr_arena_bytes().
 */
static int resolve_mode(const vocr_model *m, int mode)
{
    if (mode == VOCR_MODE_AUTO) mode = m->int8 ? VOCR_MODE_W8F : VOCR_MODE_F32;
    if (mode == VOCR_MODE_F32 && !m->int8) return mode;
    if ((mode == VOCR_MODE_W8F || mode == VOCR_MODE_W8I) && m->int8) return mode;
    return VOCR_E_MODE;
}

static size_t op_scratch(const vocr_ctx *c, const vocr_op *op, unsigned n_tile)
{
    size_t pos;
    if (op->type < T_PW) return 0;
    pos = op->type == T_HEAD ? c->acc_frames
                             : (size_t)n_tile * (op->flags & F_COLLAPSE ? 1u : op->h_in);
    if (c->mode == VOCR_MODE_W8F) return al16((size_t)op->k * 8u * sizeof(float));
    if (c->mode == VOCR_MODE_W8I) return al16(pos * op->k * 2u) + al16(pos * sizeof(float));
    return 0;
}

static int plan(vocr_ctx *c, const vocr_model *m, const vocr_opts *o)
{
    static const vocr_opts defaults = {0, 0, 0, VOCR_MODE_AUTO};
    unsigned L, S, i, f, t_max, F, hb, s = 1, s_out[VOCR_MAX_OPS];
    size_t off, base, R = 0, tin[VOCR_MAX_OPS], tout[VOCR_MAX_OPS], scr[VOCR_MAX_OPS];
    int prev = 0, mode, in_top = 0;
    if (!c || !m || !m->blob) return VOCR_E_ARG;
    if (!o) o = &defaults;
    memset(c, 0, sizeof *c);
    L = m->n_ops;
    S = m->stride;
    mode = resolve_mode(m, o->mode);
    if (mode < 0) return mode;
    c->model = m;
    c->mode = mode;
    for (i = 0; i < L; i++)
        if (mode == VOCR_MODE_W8I && m->op[i].type >= T_PW && m->op[i].k > W8I_MAX_K)
            return VOCR_E_MODE;
    c->max_width = o->max_width && o->max_width < m->max_width ? o->max_width : m->max_width;
    c->max_width = (c->max_width + S - 1) / S * S;
    t_max = c->max_width / S;
    F = umin(o->tile_frames ? o->tile_frames : 4u, t_max);
    hb = umin(o->head_frames ? o->head_frames : 16u, t_max);
    c->tile = F;
    c->head = hb;
    c->acc_frames = hb + F - 1;
    for (f = 0; f < L && (m->op[f].type == T_CONV || m->op[f].type == T_DW); f++) {
    }
    if (f < L && (m->op[f].flags & F_RES)) f--;
    if (f < 1) return VOCR_E_FORMAT;
    f--;
    c->front = f;
    for (i = 0; i < L; i++) {
        const vocr_op *op = &m->op[i];
        vocr_plan_op *p = &c->op[i];
        int r = (int)op->kw - 1 - (int)op->pw;
        s *= op->sw;
        s_out[i] = s;
        p->lag = (prev + r + 1 + (int)op->sw - 1) / (int)op->sw - 1;
        prev = p->lag;
        p->in_col = (uint32_t)op->h_in * op->cin;
        p->out_col = (uint32_t)op->h_out * op->cout;
        p->dst = i + 1 == L ? 0
                 : m->op[i + 1].type == T_PW   ? DST_TRANS
                 : m->op[i + 1].type == T_HEAD ? DST_ACC
                                               : DST_RING;
    }
    if ((F * S) % s_out[L - 1]) return VOCR_E_FORMAT;
    c->n_src = s_out[f];
    for (i = 0; i < L; i++) {
        c->op[i].n_tile = F * S / s_out[i];
        c->op[i].n_step = i <= f ? s_out[f] / s_out[i] : c->op[i].n_tile;
    }
    off = al16(8u * m->height); /* resize row table */
    c->rows_off = 0;
    c->ring_off = (uint32_t)off;
    for (i = 0; i < L; i++) {
        const vocr_op *op = &m->op[i];
        vocr_plan_op *p = &c->op[i];
        unsigned n_prev;
        if (op->type != T_CONV && op->type != T_DW) continue;
        n_prev = i == 0 ? c->n_src : i <= f ? c->op[i - 1].n_step : c->op[i - 1].n_tile;
        p->halo = (uint32_t)(p->lag * (int)op->sw + (int)op->pw - (i ? c->op[i - 1].lag : 0));
        p->cap = n_prev + p->halo;
        p->in_off = (uint32_t)off;
        off += al16((size_t)p->cap * p->in_col * sizeof(float));
    }
    c->ring_end = (uint32_t)off;
    c->acc_off = (uint32_t)off;
    c->op[L - 1].in_off = (uint32_t)off;
    off += al16((size_t)c->acc_frames * m->op[L - 1].k * sizeof(float));
    c->bestv_off = (uint32_t)off;
    off += al16((size_t)c->acc_frames * sizeof(float));
    c->besti_off = (uint32_t)off;
    off += al16((size_t)c->acc_frames * sizeof(uint32_t));
    base = off;
    for (i = 0; i < L; i++) {
        const vocr_op *op = &m->op[i];
        size_t sum;
        tin[i] = op->type == T_PW ? al16((size_t)c->op[i - 1].n_tile * c->op[i - 1].out_col * 4u) : 0;
        tout[i] = c->op[i].dst == DST_TRANS ? al16((size_t)c->op[i].n_tile * c->op[i].out_col * 4u) : 0;
        scr[i] = op_scratch(c, op, c->op[i].n_tile);
        sum = tin[i] + tout[i] + scr[i];
        if (sum > R) R = sum;
    }
    /* transients alternate between the bottom and the top of the region, scratch
     * sits above the bottom one */
    for (i = 0; i < L; i++) {
        int out_top = 0;
        size_t bottom = 0;
        if (tin[i] && !in_top) bottom = tin[i];
        if (tout[i]) {
            out_top = tin[i] && !in_top;
            c->op[i + 1].in_off = (uint32_t)(base + (out_top ? R - tout[i] : 0));
            if (!out_top) bottom = tout[i];
        }
        c->op[i].scr_off = (uint32_t)(base + bottom);
        in_top = out_top;
    }
    off = base + R;
    if (off > 0xFFFFFFF0u) return VOCR_E_ARG;
    c->arena_bytes = off;
    return VOCR_OK;
}

size_t vocr_arena_bytes(const vocr_model *m, const vocr_opts *o)
{
    vocr_ctx c;
    return plan(&c, m, o) == VOCR_OK ? c.arena_bytes : 0;
}

int vocr_init(vocr_ctx *c, const vocr_model *m, const vocr_opts *o, void *arena, size_t bytes)
{
    int rc;
    if (!c || !arena) return VOCR_E_ARG;
    if ((uintptr_t)arena & 15u) return VOCR_E_ALIGN;
    rc = plan(c, m, o);
    if (rc) return rc;
    if (bytes < c->arena_bytes) return VOCR_E_ARENA;
    c->arena = (unsigned char *)arena;
    return VOCR_OK;
}

/* ---------------------------------------------------------------- preprocessing
 *
 * Bit-exact with cv2.resize(grey, (nw, H), interpolation=cv2.INTER_LINEAR) on uint8:
 * 11-bit fixed-point coefficients from the same float/double arithmetic, horizontal
 * pass into int, vertical pass with OpenCV's (b * (v >> 4)) >> 16 rounding. The
 * vertical coefficients are not clamped at the borders, the rows are.
 */
typedef struct {
    const vocr_image *img;
    unsigned nw, padded;
    double scale_x;
    const int16_t *rows; /* per output row: y0, y1, b0, b1 */
} line_t;

static int floor_i(float v)
{
    int s = (int)v;
    return (float)s > v ? s - 1 : s;
}

static int round_even(float v) /* v >= 0, like cvRound */
{
    int n = (int)v;
    float fr = v - (float)n;
    if (fr > 0.5f || (fr == 0.5f && (n & 1))) n++;
    return n;
}

unsigned vocr_resized_width(const vocr_model *m, unsigned w, unsigned h)
{
    uint32_t num, q;
    if (!m || !m->blob || !w || !h || w > 65535u || h > 65535u) return 0;
    num = (uint32_t)w * m->height; /* round(w * H / h), half to even like Python */
    q = (2u * num + h) / (2u * h);
    if ((2u * num + h) % (2u * h) == 0 && (q & 1u)) q--;
    if (q < MIN_WIDTH) q = MIN_WIDTH;
    if (q > m->max_width) q = m->max_width;
    return q;
}

static void row_table(int16_t *rows, unsigned H, unsigned h)
{
    double scale = 1.0 / ((double)H / (double)h);
    unsigned y;
    for (y = 0; y < H; y++) {
        volatile double t = ((double)y + 0.5) * scale; /* volatile: never fused */
        float fy = (float)(t - 0.5);
        int sy = floor_i(fy);
        fy -= (float)sy;
        rows[4 * y + 0] = (int16_t)(sy < 0 ? 0 : sy > (int)h - 1 ? (int)h - 1 : sy);
        rows[4 * y + 1] = (int16_t)(sy + 1 < 0 ? 0 : sy + 1 > (int)h - 1 ? (int)h - 1 : sy + 1);
        rows[4 * y + 2] = (int16_t)round_even((1.0f - fy) * 2048.0f);
        rows[4 * y + 3] = (int16_t)round_even(fy * 2048.0f);
    }
}

static int check_image(const vocr_image *im)
{
    unsigned bpp;
    if (!im || !im->pixels || !im->width || !im->height || im->width > 65535u ||
        im->height > 65535u)
        return VOCR_E_ARG;
    if (im->format != VOCR_GREY8 && im->format != VOCR_RGBA8) return VOCR_E_ARG;
    bpp = im->format == VOCR_GREY8 ? 1u : 4u;
    if (im->stride < (size_t)im->width * bpp) return VOCR_E_ARG;
    return VOCR_OK;
}

static void line_setup(line_t *ln, const vocr_model *m, const vocr_image *im, int16_t *rows)
{
    ln->img = im;
    ln->nw = vocr_resized_width(m, im->width, im->height);
    ln->padded = (ln->nw + m->stride - 1) / m->stride * m->stride;
    ln->scale_x = 1.0 / ((double)ln->nw / (double)im->width);
    ln->rows = rows;
    row_table(rows, m->height, im->height);
}

static int grey_px(int format, const unsigned char *row, unsigned x)
{
    if (format == VOCR_GREY8) return row[x];
    row += 4u * x;
    return (77 * row[0] + 150 * row[1] + 29 * row[2]) >> 8;
}

/* One resized column x (x < nw): its source columns and coefficients. */
typedef struct {
    unsigned s0, s1;
    int a0, a1;
} xcoef_t;

static xcoef_t col_coef(const line_t *ln, unsigned x)
{
    xcoef_t k;
    int w = (int)ln->img->width, sx;
    volatile double t = ((double)x + 0.5) * ln->scale_x;
    float fx = (float)(t - 0.5);
    sx = floor_i(fx);
    fx -= (float)sx;
    if (sx < 0) {
        fx = 0.0f;
        sx = 0;
    }
    if (sx >= w - 1) {
        fx = 0.0f;
        sx = w - 1;
    }
    k.a0 = round_even((1.0f - fx) * 2048.0f);
    k.a1 = round_even(fx * 2048.0f);
    k.s0 = (unsigned)sx;
    k.s1 = sx + 1 < w ? (unsigned)sx + 1u : (unsigned)sx;
    return k;
}

static unsigned resized_px(const line_t *ln, unsigned y, xcoef_t k)
{
    const vocr_image *im = ln->img;
    const int16_t *r = ln->rows + 4u * y;
    const unsigned char *p0 = im->pixels + (size_t)r[0] * im->stride;
    const unsigned char *p1 = im->pixels + (size_t)r[1] * im->stride;
    int h0 = grey_px(im->format, p0, k.s0) * k.a0 + grey_px(im->format, p0, k.s1) * k.a1;
    int h1 = grey_px(im->format, p1, k.s0) * k.a0 + grey_px(im->format, p1, k.s1) * k.a1;
    return (unsigned)((((r[2] * (h0 >> 4)) >> 16) + ((r[3] * (h1 >> 4)) >> 16) + 2) >> 2);
}

/* Source columns [j0, j0 + n) of the normalised network input, [col][row] floats. */
static void src_cols(const line_t *ln, unsigned H, long j0, unsigned n, float *dst)
{
    unsigned t, y;
    for (t = 0; t < n; t++) {
        long j = j0 + (long)t;
        float *col = dst + (size_t)t * H;
        xcoef_t k;
        if (j < 0 || j >= (long)ln->padded) {
            memset(col, 0, H * sizeof(float));
            continue;
        }
        k = col_coef(ln, j < (long)ln->nw ? (unsigned)j : ln->nw - 1u);
        for (y = 0; y < H; y++) col[y] = (float)resized_px(ln, y, k) / 127.5f - 1.0f;
    }
}

int vocr_preprocess(const vocr_model *m, const vocr_image *img, unsigned char *out,
                    size_t cap, unsigned *width, unsigned *padded_width)
{
    int16_t rows[4 * VOCR_MAX_HEIGHT];
    line_t ln;
    unsigned x, y;
    int rc = check_image(img);
    if (rc) return rc;
    if (!m || !m->blob) return VOCR_E_ARG;
    line_setup(&ln, m, img, rows);
    if (width) *width = ln.nw;
    if (padded_width) *padded_width = ln.padded;
    if (!out) return VOCR_OK;
    if (cap < (size_t)ln.padded * m->height) return VOCR_E_ARG;
    for (x = 0; x < ln.padded; x++) {
        xcoef_t k = col_coef(&ln, x < ln.nw ? x : ln.nw - 1u);
        for (y = 0; y < m->height; y++) out[(size_t)y * ln.padded + x] = (unsigned char)resized_px(&ln, y, k);
    }
    return VOCR_OK;
}

/* ---------------------------------------------------------------- kernels
 *
 * Activations are [column][row][channel], channels innermost. Every output element
 * is computed as acc = bias, then acc = acc + w * x over the taps (dy, then dx) or
 * over k in ascending order, each product and sum rounded separately; the NEON
 * paths vectorise across output channels only, so they give the same bits as the
 * scalar code. Taps that fall in the top/bottom padding are skipped; columns
 * outside the line are zero-filled by the producer, so they need no checks here.
 */
static float relu_f(float v) { return v > 0.0f ? v : 0.0f; }

/* CONV with one input channel (the stem). win: [col][h_in]; output column t reads
 * window columns t * sw + dx. Writes output columns [t0, t1) to out. */
static void k_conv1(const vocr_op *op, const float *win, float *out, unsigned t0, unsigned t1)
{
    const unsigned C = op->cout, kw = op->kw, Ho = op->h_out, Hi = op->h_in;
    const int relu = (op->flags & F_RELU) != 0;
    unsigned t, y, c0, c, dy, dx;
    for (t = t0; t < t1; t++)
        for (y = 0; y < Ho; y++) {
            const int r0 = (int)(y * op->sh) - (int)op->ph;
            const unsigned dy0 = r0 < 0 ? (unsigned)-r0 : 0u;
            const unsigned dy1 = umin(op->kh, (unsigned)((int)Hi - r0));
            float *o = out + ((size_t)(t - t0) * Ho + y) * C;
            c0 = 0;
#ifdef VOCR_NEON
            for (; c0 + 16 <= C; c0 += 16) {
                float32x4_t a0, a1, a2, a3, z = vdupq_n_f32(0.0f);
                if (op->bias) {
                    a0 = vld1q_f32(op->bias + c0);
                    a1 = vld1q_f32(op->bias + c0 + 4);
                    a2 = vld1q_f32(op->bias + c0 + 8);
                    a3 = vld1q_f32(op->bias + c0 + 12);
                } else {
                    a0 = a1 = a2 = a3 = z;
                }
                for (dy = dy0; dy < dy1; dy++)
                    for (dx = 0; dx < kw; dx++) {
                        const float x = win[(size_t)(t * op->sw + dx) * Hi + (unsigned)(r0 + (int)dy)];
                        const float *wt = op->w + (size_t)(dy * kw + dx) * C + c0;
                        a0 = vmlaq_n_f32(a0, vld1q_f32(wt), x);
                        a1 = vmlaq_n_f32(a1, vld1q_f32(wt + 4), x);
                        a2 = vmlaq_n_f32(a2, vld1q_f32(wt + 8), x);
                        a3 = vmlaq_n_f32(a3, vld1q_f32(wt + 12), x);
                    }
                if (relu) {
                    a0 = vmaxq_f32(a0, z);
                    a1 = vmaxq_f32(a1, z);
                    a2 = vmaxq_f32(a2, z);
                    a3 = vmaxq_f32(a3, z);
                }
                vst1q_f32(o + c0, a0);
                vst1q_f32(o + c0 + 4, a1);
                vst1q_f32(o + c0 + 8, a2);
                vst1q_f32(o + c0 + 12, a3);
            }
#endif
            for (; c0 < C; c0 += 32) {
                const unsigned cn = umin(32u, C - c0);
                float acc[32];
                for (c = 0; c < cn; c++) acc[c] = op->bias ? op->bias[c0 + c] : 0.0f;
                for (dy = dy0; dy < dy1; dy++)
                    for (dx = 0; dx < kw; dx++) {
                        const float x = win[(size_t)(t * op->sw + dx) * Hi + (unsigned)(r0 + (int)dy)];
                        const float *wt = op->w + (size_t)(dy * kw + dx) * C + c0;
                        for (c = 0; c < cn; c++) acc[c] = acc[c] + wt[c] * x;
                    }
                for (c = 0; c < cn; c++) o[c0 + c] = relu ? relu_f(acc[c]) : acc[c];
            }
        }
}

/* Depthwise kh x kw. win: [col][h_in][C]. */
static void k_dw(const vocr_op *op, const float *win, float *out, unsigned t0, unsigned t1)
{
    const unsigned C = op->cout, kw = op->kw, Ho = op->h_out, Hi = op->h_in;
    const int relu = (op->flags & F_RELU) != 0;
    unsigned t, y, c0, c, dy, dx;
    for (t = t0; t < t1; t++)
        for (y = 0; y < Ho; y++) {
            const int r0 = (int)(y * op->sh) - (int)op->ph;
            const unsigned dy0 = r0 < 0 ? (unsigned)-r0 : 0u;
            const unsigned dy1 = umin(op->kh, (unsigned)((int)Hi - r0));
            float *o = out + ((size_t)(t - t0) * Ho + y) * C;
            c0 = 0;
#ifdef VOCR_NEON
            for (; c0 + 16 <= C; c0 += 16) {
                float32x4_t a0, a1, a2, a3, z = vdupq_n_f32(0.0f);
                if (op->bias) {
                    a0 = vld1q_f32(op->bias + c0);
                    a1 = vld1q_f32(op->bias + c0 + 4);
                    a2 = vld1q_f32(op->bias + c0 + 8);
                    a3 = vld1q_f32(op->bias + c0 + 12);
                } else {
                    a0 = a1 = a2 = a3 = z;
                }
                for (dy = dy0; dy < dy1; dy++)
                    for (dx = 0; dx < kw; dx++) {
                        const float *x = win + ((size_t)(t * op->sw + dx) * Hi + (unsigned)(r0 + (int)dy)) * C + c0;
                        const float *wt = op->w + (size_t)(dy * kw + dx) * C + c0;
                        a0 = vmlaq_f32(a0, vld1q_f32(wt), vld1q_f32(x));
                        a1 = vmlaq_f32(a1, vld1q_f32(wt + 4), vld1q_f32(x + 4));
                        a2 = vmlaq_f32(a2, vld1q_f32(wt + 8), vld1q_f32(x + 8));
                        a3 = vmlaq_f32(a3, vld1q_f32(wt + 12), vld1q_f32(x + 12));
                    }
                if (relu) {
                    a0 = vmaxq_f32(a0, z);
                    a1 = vmaxq_f32(a1, z);
                    a2 = vmaxq_f32(a2, z);
                    a3 = vmaxq_f32(a3, z);
                }
                vst1q_f32(o + c0, a0);
                vst1q_f32(o + c0 + 4, a1);
                vst1q_f32(o + c0 + 8, a2);
                vst1q_f32(o + c0 + 12, a3);
            }
#endif
            for (; c0 < C; c0 += 32) {
                const unsigned cn = umin(32u, C - c0);
                float acc[32];
                for (c = 0; c < cn; c++) acc[c] = op->bias ? op->bias[c0 + c] : 0.0f;
                for (dy = dy0; dy < dy1; dy++)
                    for (dx = 0; dx < kw; dx++) {
                        const float *x = win + ((size_t)(t * op->sw + dx) * Hi + (unsigned)(r0 + (int)dy)) * C + c0;
                        const float *wt = op->w + (size_t)(dy * kw + dx) * C + c0;
                        for (c = 0; c < cn; c++) acc[c] = acc[c] + wt[c] * x[c];
                    }
                for (c = 0; c < cn; c++) o[c0 + c] = relu ? relu_f(acc[c]) : acc[c];
            }
        }
}

/* Float block: y[p][j] = act(bias[j] + sum_k w[k][j] x[p][k]) (+ res[p][j]) for the
 * 8 outputs of one column block w ([K][8]) and positions p < P. */
static void mm_f32(const float *w, const float *bias, const float *x, size_t xs, unsigned K,
                   unsigned P, float *y, size_t ys, const float *res, size_t rs, int relu)
{
    unsigned p = 0, q, j, k;
#ifdef VOCR_NEON
    const float32x4_t z = vdupq_n_f32(0.0f);
    const float32x4_t bl = bias ? vld1q_f32(bias) : z, bh = bias ? vld1q_f32(bias + 4) : z;
    for (; p + 4 <= P; p += 4) {
        const float *x0 = x + p * xs, *x1 = x0 + xs, *x2 = x1 + xs, *x3 = x2 + xs;
        float32x4_t a0l = bl, a0h = bh, a1l = bl, a1h = bh, a2l = bl, a2h = bh, a3l = bl, a3h = bh;
        float32x4_t r[8];
        for (k = 0; k + 4 <= K; k += 4) {
            const float32x4_t v0 = vld1q_f32(x0 + k), v1 = vld1q_f32(x1 + k);
            const float32x4_t v2 = vld1q_f32(x2 + k), v3 = vld1q_f32(x3 + k);
            const float *wk = w + (size_t)k * 8;
#define VOCR_STEP(LANEV, LANE, OFF)                                                         \
    do {                                                                                     \
        const float32x4_t wl = vld1q_f32(wk + (OFF)), wh = vld1q_f32(wk + (OFF) + 4);       \
        a0l = vmlaq_lane_f32(a0l, wl, LANEV(v0), LANE);                                      \
        a0h = vmlaq_lane_f32(a0h, wh, LANEV(v0), LANE);                                      \
        a1l = vmlaq_lane_f32(a1l, wl, LANEV(v1), LANE);                                      \
        a1h = vmlaq_lane_f32(a1h, wh, LANEV(v1), LANE);                                      \
        a2l = vmlaq_lane_f32(a2l, wl, LANEV(v2), LANE);                                      \
        a2h = vmlaq_lane_f32(a2h, wh, LANEV(v2), LANE);                                      \
        a3l = vmlaq_lane_f32(a3l, wl, LANEV(v3), LANE);                                      \
        a3h = vmlaq_lane_f32(a3h, wh, LANEV(v3), LANE);                                      \
    } while (0)
            VOCR_STEP(vget_low_f32, 0, 0);
            VOCR_STEP(vget_low_f32, 1, 8);
            VOCR_STEP(vget_high_f32, 0, 16);
            VOCR_STEP(vget_high_f32, 1, 24);
#undef VOCR_STEP
        }
        for (; k < K; k++) {
            const float32x4_t wl = vld1q_f32(w + (size_t)k * 8), wh = vld1q_f32(w + (size_t)k * 8 + 4);
            a0l = vmlaq_n_f32(a0l, wl, x0[k]);
            a0h = vmlaq_n_f32(a0h, wh, x0[k]);
            a1l = vmlaq_n_f32(a1l, wl, x1[k]);
            a1h = vmlaq_n_f32(a1h, wh, x1[k]);
            a2l = vmlaq_n_f32(a2l, wl, x2[k]);
            a2h = vmlaq_n_f32(a2h, wh, x2[k]);
            a3l = vmlaq_n_f32(a3l, wl, x3[k]);
            a3h = vmlaq_n_f32(a3h, wh, x3[k]);
        }
        r[0] = a0l, r[1] = a0h, r[2] = a1l, r[3] = a1h, r[4] = a2l, r[5] = a2h, r[6] = a3l, r[7] = a3h;
        for (q = 0; q < 4; q++) {
            float32x4_t lo = r[2 * q], hi = r[2 * q + 1];
            if (relu) {
                lo = vmaxq_f32(lo, z);
                hi = vmaxq_f32(hi, z);
            }
            if (res) {
                lo = vaddq_f32(lo, vld1q_f32(res + (p + q) * rs));
                hi = vaddq_f32(hi, vld1q_f32(res + (p + q) * rs + 4));
            }
            vst1q_f32(y + (p + q) * ys, lo);
            vst1q_f32(y + (p + q) * ys + 4, hi);
        }
    }
    for (; p < P; p++) {
        const float *x0 = x + p * xs;
        float32x4_t lo = bl, hi = bh;
        for (k = 0; k < K; k++) {
            lo = vmlaq_n_f32(lo, vld1q_f32(w + (size_t)k * 8), x0[k]);
            hi = vmlaq_n_f32(hi, vld1q_f32(w + (size_t)k * 8 + 4), x0[k]);
        }
        if (relu) {
            lo = vmaxq_f32(lo, z);
            hi = vmaxq_f32(hi, z);
        }
        if (res) {
            lo = vaddq_f32(lo, vld1q_f32(res + p * rs));
            hi = vaddq_f32(hi, vld1q_f32(res + p * rs + 4));
        }
        vst1q_f32(y + p * ys, lo);
        vst1q_f32(y + p * ys + 4, hi);
    }
#endif
    for (; p < P; p += 4) {
        const unsigned pn = umin(4u, P - p);
        float acc[4][8];
        for (q = 0; q < pn; q++)
            for (j = 0; j < 8; j++) acc[q][j] = bias ? bias[j] : 0.0f;
        for (k = 0; k < K; k++) {
            const float *wk = w + (size_t)k * 8;
            for (q = 0; q < pn; q++) {
                const float xv = x[(p + q) * xs + k];
                for (j = 0; j < 8; j++) acc[q][j] = acc[q][j] + wk[j] * xv;
            }
        }
        for (q = 0; q < pn; q++)
            for (j = 0; j < 8; j++) {
                float v = relu ? relu_f(acc[q][j]) : acc[q][j];
                if (res) v = v + res[(p + q) * rs + j];
                y[(p + q) * ys + j] = v;
            }
    }
}

/* int8 column block -> float: wf[k][j] = q[k][j] * scale[j]. */
static void dequant8(const int8_t *qw, const float *scale, unsigned K, float *wf)
{
    unsigned k = 0, j;
#ifdef VOCR_NEON
    const float32x4_t sl = vld1q_f32(scale), sh = vld1q_f32(scale + 4);
    for (; k < K; k++) {
        const int16x8_t v = vmovl_s8(vld1_s8(qw + (size_t)k * 8));
        vst1q_f32(wf + (size_t)k * 8, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v))), sl));
        vst1q_f32(wf + (size_t)k * 8 + 4, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v))), sh));
    }
#endif
    for (; k < K; k++)
        for (j = 0; j < 8; j++) wf[(size_t)k * 8 + j] = (float)qw[(size_t)k * 8 + j] * scale[j];
}

/* Per-position int16 quantisation: q = trunc(x * (32767 / max|x|) +- 0.5), and the
 * dequantisation scale max|x| / 32767 (0 for an all-zero position). */
static void quant16(const float *x, size_t xs, unsigned K, unsigned P, int16_t *qx, float *sx)
{
    unsigned p, k;
    for (p = 0; p < P; p++) {
        const float *xp = x + p * xs;
        int16_t *qp = qx + (size_t)p * K;
        float mx = 0.0f, inv;
        k = 0;
#ifdef VOCR_NEON
        if (K >= 4) {
            float32x4_t m4 = vabsq_f32(vld1q_f32(xp));
            float32x2_t m2;
            for (k = 4; k + 4 <= K; k += 4) m4 = vmaxq_f32(m4, vabsq_f32(vld1q_f32(xp + k)));
            m2 = vpmax_f32(vget_low_f32(m4), vget_high_f32(m4));
            m2 = vpmax_f32(m2, m2);
            mx = vget_lane_f32(m2, 0);
        }
#endif
        for (; k < K; k++) {
            const float a = fabsf(xp[k]);
            if (a > mx) mx = a;
        }
        if (!(mx > 0.0f)) {
            memset(qp, 0, K * sizeof(int16_t));
            sx[p] = 0.0f;
            continue;
        }
        inv = 32767.0f / mx;
        sx[p] = mx / 32767.0f;
        k = 0;
#ifdef VOCR_NEON
        {
            const float32x4_t vinv = vdupq_n_f32(inv);
            const uint32x4_t sign = vdupq_n_u32(0x80000000u), half = vreinterpretq_u32_f32(vdupq_n_f32(0.5f));
            for (; k + 4 <= K; k += 4) {
                const float32x4_t t = vmulq_f32(vld1q_f32(xp + k), vinv);
                const float32x4_t h = vreinterpretq_f32_u32(vorrq_u32(vandq_u32(vreinterpretq_u32_f32(t), sign), half));
                vst1_s16(qp + k, vqmovn_s32(vcvtq_s32_f32(vaddq_f32(t, h))));
            }
        }
#endif
        for (; k < K; k++) {
            const float t = xp[k] * inv;
            int v = (int)(t + (signbit(t) ? -0.5f : 0.5f));
            qp[k] = (int16_t)(v > 32767 ? 32767 : v < -32767 ? -32767 : v);
        }
    }
}

/* Integer block: y[p][j] = act(float(sum_k q[k][j] qx[p][k]) * (scale[j] * sx[p]) + bias[j]). */
static void mm_i16(const int8_t *qw, const float *scale, const float *bias, const int16_t *qx,
                   const float *sx, unsigned K, unsigned P, float *y, size_t ys, const float *res,
                   size_t rs, int relu)
{
    unsigned p = 0, q, j, k;
#ifdef VOCR_NEON
    const float32x4_t z = vdupq_n_f32(0.0f), sl = vld1q_f32(scale), sh = vld1q_f32(scale + 4);
    const float32x4_t bl = bias ? vld1q_f32(bias) : z, bh = bias ? vld1q_f32(bias + 4) : z;
    for (; p + 4 <= P; p += 4) {
        const int16_t *x0 = qx + (size_t)p * K, *x1 = x0 + K, *x2 = x1 + K, *x3 = x2 + K;
        int32x4_t a[8];
        for (q = 0; q < 8; q++) a[q] = vdupq_n_s32(0);
        for (k = 0; k + 4 <= K; k += 4) {
            const int16x4_t v0 = vld1_s16(x0 + k), v1 = vld1_s16(x1 + k);
            const int16x4_t v2 = vld1_s16(x2 + k), v3 = vld1_s16(x3 + k);
#define VOCR_ISTEP(LANE)                                                                     \
    do {                                                                                     \
        const int16x8_t wv = vmovl_s8(vld1_s8(qw + (size_t)(k + (LANE)) * 8));               \
        const int16x4_t wl = vget_low_s16(wv), wh = vget_high_s16(wv);                       \
        a[0] = vmlal_lane_s16(a[0], wl, v0, LANE);                                           \
        a[1] = vmlal_lane_s16(a[1], wh, v0, LANE);                                           \
        a[2] = vmlal_lane_s16(a[2], wl, v1, LANE);                                           \
        a[3] = vmlal_lane_s16(a[3], wh, v1, LANE);                                           \
        a[4] = vmlal_lane_s16(a[4], wl, v2, LANE);                                           \
        a[5] = vmlal_lane_s16(a[5], wh, v2, LANE);                                           \
        a[6] = vmlal_lane_s16(a[6], wl, v3, LANE);                                           \
        a[7] = vmlal_lane_s16(a[7], wh, v3, LANE);                                           \
    } while (0)
            VOCR_ISTEP(0);
            VOCR_ISTEP(1);
            VOCR_ISTEP(2);
            VOCR_ISTEP(3);
#undef VOCR_ISTEP
        }
        for (; k < K; k++) {
            const int16x8_t wv = vmovl_s8(vld1_s8(qw + (size_t)k * 8));
            const int16x4_t wl = vget_low_s16(wv), wh = vget_high_s16(wv);
            a[0] = vmlal_n_s16(a[0], wl, x0[k]);
            a[1] = vmlal_n_s16(a[1], wh, x0[k]);
            a[2] = vmlal_n_s16(a[2], wl, x1[k]);
            a[3] = vmlal_n_s16(a[3], wh, x1[k]);
            a[4] = vmlal_n_s16(a[4], wl, x2[k]);
            a[5] = vmlal_n_s16(a[5], wh, x2[k]);
            a[6] = vmlal_n_s16(a[6], wl, x3[k]);
            a[7] = vmlal_n_s16(a[7], wh, x3[k]);
        }
        for (q = 0; q < 4; q++) {
            float32x4_t lo = vmulq_f32(vcvtq_f32_s32(a[2 * q]), vmulq_n_f32(sl, sx[p + q]));
            float32x4_t hi = vmulq_f32(vcvtq_f32_s32(a[2 * q + 1]), vmulq_n_f32(sh, sx[p + q]));
            if (bias) {
                lo = vaddq_f32(lo, bl);
                hi = vaddq_f32(hi, bh);
            }
            if (relu) {
                lo = vmaxq_f32(lo, z);
                hi = vmaxq_f32(hi, z);
            }
            if (res) {
                lo = vaddq_f32(lo, vld1q_f32(res + (p + q) * rs));
                hi = vaddq_f32(hi, vld1q_f32(res + (p + q) * rs + 4));
            }
            vst1q_f32(y + (p + q) * ys, lo);
            vst1q_f32(y + (p + q) * ys + 4, hi);
        }
    }
#endif
    for (; p < P; p += 4) {
        const unsigned pn = umin(4u, P - p);
        int32_t acc[4][8];
        for (q = 0; q < pn; q++)
            for (j = 0; j < 8; j++) acc[q][j] = 0;
        for (k = 0; k < K; k++) {
            const int8_t *wk = qw + (size_t)k * 8;
            for (q = 0; q < pn; q++) {
                const int32_t xv = qx[(size_t)(p + q) * K + k];
                for (j = 0; j < 8; j++) acc[q][j] += (int32_t)wk[j] * xv;
            }
        }
        for (q = 0; q < pn; q++)
            for (j = 0; j < 8; j++) {
                float v = (float)acc[q][j] * (scale[j] * sx[p + q]);
                if (bias) v = v + bias[j];
                if (relu) v = relu_f(v);
                if (res) v = v + res[(p + q) * rs + j];
                y[(p + q) * ys + j] = v;
            }
    }
}

/* ---------------------------------------------------------------- run */

typedef struct {
    char *text;
    size_t cap, used;
    int overflow;
    unsigned last, chars, frame;
    vocr_extra *ex;
} out_t;

static float *at(const vocr_ctx *c, uint32_t off) { return (float *)(void *)(c->arena + off); }

/* PW over P positions of x (row stride K) into y (row stride cout). */
static void run_pw(const vocr_ctx *c, unsigned i, const float *x, unsigned P, float *y, const float *res)
{
    const vocr_op *op = &c->model->op[i];
    const unsigned K = op->k, N = op->cout;
    const int relu = (op->flags & F_RELU) != 0;
    unsigned char *scr = c->arena + c->op[i].scr_off;
    int16_t *qx = (int16_t *)(void *)scr;
    float *sx = (float *)(void *)(scr + al16((size_t)P * K * 2u));
    float *wf = (float *)(void *)scr;
    unsigned nb;
    if (!P) return;
    if (c->mode == VOCR_MODE_W8I) quant16(x, K, K, P, qx, sx);
    for (nb = 0; nb < N / 8; nb++) {
        const float *b = op->bias ? op->bias + nb * 8 : NULL;
        const float *rb = res ? res + nb * 8 : NULL;
        const size_t wo = (size_t)nb * K * 8;
        if (c->mode == VOCR_MODE_F32) {
            mm_f32(op->w + wo, b, x, K, K, P, y + nb * 8, N, rb, N, relu);
        } else if (c->mode == VOCR_MODE_W8F) {
            dequant8(op->q + wo, op->scale + nb * 8, K, wf);
            mm_f32(wf, b, x, K, K, P, y + nb * 8, N, rb, N, relu);
        } else {
            mm_i16(op->q + wo, op->scale + nb * 8, b, qx, sx, K, P, y + nb * 8, N, rb, N, relu);
        }
    }
}

static void ctc_push(const vocr_model *m, out_t *o, unsigned tok)
{
    if (tok && tok != o->last) {
        unsigned n;
        const char *s = vocr_class_text(m, tok, &n);
        if (!o->overflow && o->used + n < o->cap) {
            memcpy(o->text + o->used, s, n);
            o->used += n;
            o->text[o->used] = 0;
            o->chars++;
        } else {
            o->overflow = 1;
        }
    }
    o->last = tok;
}

/* Output head over the nf accumulated frames: argmax (first maximum) and CTC. */
static void run_head(const vocr_ctx *c, unsigned nf, out_t *o)
{
    const vocr_model *m = c->model;
    const unsigned L = m->n_ops, N = m->n_classes;
    const vocr_op *op = &m->op[L - 1];
    const unsigned K = op->k;
    const float *x = at(c, c->acc_off);
    float *bestv = at(c, c->bestv_off);
    uint32_t *besti = (uint32_t *)(void *)(c->arena + c->besti_off);
    unsigned char *scr = c->arena + c->op[L - 1].scr_off;
    int16_t *qx = (int16_t *)(void *)scr;
    float *sx = (float *)(void *)(scr + al16((size_t)nf * K * 2u));
    float *wf = (float *)(void *)scr;
    vocr_extra *ex = o->ex;
    unsigned f, nb, f0, q, j;
    for (f = 0; f < nf; f++) {
        bestv[f] = -INFINITY;
        besti[f] = 0;
    }
    if (c->mode == VOCR_MODE_W8I) quant16(x, K, K, nf, qx, sx);
    for (nb = 0; nb < op->cout_pad / 8u; nb++) {
        const float *b = op->bias ? op->bias + nb * 8 : NULL;
        const size_t wo = (size_t)nb * K * 8;
        if (c->mode == VOCR_MODE_W8F) dequant8(op->q + wo, op->scale + nb * 8, K, wf);
        for (f0 = 0; f0 < nf; f0 += 4) {
            const unsigned pn = umin(4u, nf - f0);
            float tmp[4 * 8];
            if (c->mode == VOCR_MODE_F32)
                mm_f32(op->w + wo, b, x + (size_t)f0 * K, K, K, pn, tmp, 8, NULL, 0, 0);
            else if (c->mode == VOCR_MODE_W8F)
                mm_f32(wf, b, x + (size_t)f0 * K, K, K, pn, tmp, 8, NULL, 0, 0);
            else
                mm_i16(op->q + wo, op->scale + nb * 8, b, qx + (size_t)f0 * K, sx + f0, K, pn, tmp,
                       8, NULL, 0, 0);
            for (q = 0; q < pn; q++) {
                const unsigned frame = o->frame + f0 + q;
                for (j = 0; j < 8 && nb * 8 + j < N; j++) {
                    const float v = tmp[q * 8 + j];
                    if (ex && ex->logits && frame < ex->cap_frames)
                        ex->logits[(size_t)frame * N + nb * 8 + j] = v;
                    if (v > bestv[f0 + q]) {
                        bestv[f0 + q] = v;
                        besti[f0 + q] = nb * 8 + j;
                    }
                }
            }
        }
    }
    for (f = 0; f < nf; f++, o->frame++) {
        if (ex && ex->best && o->frame < ex->cap_frames) ex->best[o->frame] = (uint16_t)besti[f];
        ctc_push(m, o, besti[f]);
    }
}

/* Keep the last `halo` columns of a ring as its first ones. The ranges can overlap
 * (fewer new columns than halo), so copy forwards: the destination is below the source. */
static void ring_shift(float *ring, unsigned cap, unsigned halo, unsigned col)
{
    const float *src = ring + (size_t)(cap - halo) * col;
    size_t i, n = (size_t)halo * col;
    if (!halo || src == ring) return;
    for (i = 0; i < n; i++) ring[i] = src[i];
}

/* Compute output columns [j0, j0 + n) of op i (wout of them exist) into its
 * destination; u is the mini-tile index for the last front op, else 0. */
static void step_op(const vocr_ctx *c, unsigned i, long j0, unsigned n, unsigned wout,
                    unsigned u, unsigned *acc_fill)
{
    const vocr_model *m = c->model;
    const vocr_op *op = &m->op[i];
    const vocr_plan_op *p = &c->op[i];
    const unsigned oc = p->out_col;
    unsigned t0 = 0, t1 = n, t;
    float *dst, *out;
    if (j0 < 0) t0 = (unsigned)(-j0) < n ? (unsigned)(-j0) : n;
    if (j0 + (long)n > (long)wout) t1 = (long)wout > j0 ? (unsigned)((long)wout - j0) : 0u;
    if (t1 < t0) t1 = t0;
    if (p->dst == DST_ACC) {
        out = at(c, c->acc_off) + (size_t)*acc_fill * oc;
        dst = NULL;
        *acc_fill += t1 - t0;
    } else {
        const vocr_plan_op *nx = &c->op[i + 1];
        dst = at(c, nx->in_off);
        if (p->dst == DST_RING) {
            if (!u) ring_shift(dst, nx->cap, nx->halo, nx->in_col);
            dst += (size_t)(nx->halo + u) * nx->in_col;
        } else {
            dst += (size_t)u * oc;
        }
        out = dst + (size_t)t0 * oc;
    }
    if (t1 > t0) {
        if (op->type == T_CONV) {
            k_conv1(op, at(c, p->in_off), out, t0, t1);
        } else if (op->type == T_DW) {
            k_dw(op, at(c, p->in_off), out, t0, t1);
        } else {
            const unsigned rows = op->flags & F_COLLAPSE ? 1u : op->h_in;
            const float *res = NULL;
            if (op->flags & F_RES)
                res = at(c, c->op[i - 1].in_off) + (size_t)(t0 + m->op[i - 1].pw) * oc;
            run_pw(c, i, at(c, p->in_off) + (size_t)t0 * p->in_col, (t1 - t0) * rows, out, res);
        }
    }
    if (dst) {
        for (t = 0; t < t0; t++) memset(dst + (size_t)t * oc, 0, oc * sizeof(float));
        for (t = t1; t < n; t++) memset(dst + (size_t)t * oc, 0, oc * sizeof(float));
    }
}

int vocr_run(vocr_ctx *c, const vocr_image *img, char *text, size_t text_cap, vocr_extra *ex)
{
    const vocr_model *m;
    line_t ln;
    out_t o;
    unsigned L, S, f, i, u, T, wout[VOCR_MAX_OPS], acc_fill = 0, n_front;
    long k, ntiles;
    int rc;
    if (!c || !c->arena || !c->model || !text || !text_cap) return VOCR_E_ARG;
    text[0] = 0;
    rc = check_image(img);
    if (rc) return rc;
    m = c->model;
    L = m->n_ops;
    S = m->stride;
    f = c->front;
    line_setup(&ln, m, img, (int16_t *)(void *)(c->arena + c->rows_off));
    if (ln.nw > c->max_width) return VOCR_E_WIDTH;
    T = ln.padded / S;
    memset(&o, 0, sizeof o);
    o.text = text;
    o.cap = text_cap;
    o.ex = ex;
    if (ex) {
        ex->width = ln.nw;
        ex->frames = T;
        ex->chars = 0;
    }
    for (i = 0, u = ln.padded; i < L; i++) wout[i] = u /= m->op[i].sw;
    memset(c->arena + c->ring_off, 0, c->ring_end - c->ring_off);
    n_front = c->op[f].n_tile;
    ntiles = ((long)T + c->op[L - 1].lag + (long)c->tile - 1) / (long)c->tile;
    for (k = 0; k < ntiles; k++) {
        if (ex && ex->cancel && ex->cancel(ex->user)) {
            text[0] = 0;
            return VOCR_E_CANCELLED;
        }
        for (u = 0; u < n_front; u++) {
            const long g = k * (long)n_front + (long)u;
            const vocr_plan_op *p0 = &c->op[0];
            float *ring0 = at(c, p0->in_off);
            ring_shift(ring0, p0->cap, p0->halo, p0->in_col);
            src_cols(&ln, m->height, g * (long)c->n_src, c->n_src, ring0 + (size_t)p0->halo * p0->in_col);
            for (i = 0; i <= f; i++)
                step_op(c, i, g * (long)c->op[i].n_step - c->op[i].lag, c->op[i].n_step, wout[i],
                        i == f ? u : 0u, &acc_fill);
        }
        for (i = f + 1; i + 1 < L; i++)
            step_op(c, i, k * (long)c->op[i].n_tile - c->op[i].lag, c->op[i].n_tile, wout[i], 0u,
                    &acc_fill);
        if (acc_fill >= c->head) {
            run_head(c, acc_fill, &o);
            acc_fill = 0;
        }
    }
    if (acc_fill) run_head(c, acc_fill, &o);
    if (ex) ex->chars = o.chars;
    if (o.frame != T) {
        text[0] = 0;
        return VOCR_E_FORMAT; /* cannot happen for a validated blob */
    }
    return o.overflow ? VOCR_E_TEXT : VOCR_OK;
}

void vocr_sizes(size_t *model_bytes, size_t *ctx_bytes)
{
    if (model_bytes) *model_bytes = sizeof(vocr_model);
    if (ctx_bytes) *ctx_bytes = sizeof(vocr_ctx);
}
