/* test_vocr.c - unit tests of the vocr runtime that need no model file.
 *
 * Builds a tiny valid blob in memory (stem, depthwise, pointwise, 2x1 collapse,
 * 1x5 temporal depthwise with a residual pointwise, head), then checks loading and
 * validation, arena sizing, tile invariance, text overflow, cancellation and error
 * paths, the head's argmax on ties, signed zeros, NaN and -inf (a 21-class head:
 * two full 8-class blocks and a partial one), and the head gate (docs/headgate.md): its
 * section's validation and CRC, and its runs against an independent reference of its
 * score and of the skipped frames, at every tiling, on that blob and on one whose head follows the stem. Run under ASan/UBSan by `make check`, and
 * as Cortex-A9 NEON and scalar builds under qemu-arm by `make check-arm`.
 */
#include "../vocr.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);       \
            failures++;                                                                    \
        }                                                                                  \
    } while (0)

/* ------------------------------------------------------------ tiny blob builder */

static unsigned char blob_buf[1 << 17] __attribute__((aligned(16)));
static size_t blob_pos;
static uint32_t lcg = 12345u;

/* every write into blob_buf asks first: a blob that does not fit stops the test */
static void room(size_t n)
{
    if (blob_pos + n > sizeof blob_buf) {
        fprintf(stderr, "test blob too large: %zu + %zu bytes\n", blob_pos, n);
        exit(2);
    }
}

static float frand(void)
{
    lcg = lcg * 1664525u + 1013904223u;
    return (float)((lcg >> 8) & 0xFFFF) / 32768.0f - 1.0f;
}

static void wr16(unsigned char *p, unsigned v) { p[0] = (unsigned char)v; p[1] = (unsigned char)(v >> 8); }
static void wr32(unsigned char *p, uint32_t v)
{
    p[0] = (unsigned char)v;
    p[1] = (unsigned char)(v >> 8);
    p[2] = (unsigned char)(v >> 16);
    p[3] = (unsigned char)(v >> 24);
}

static uint32_t put(const void *data, size_t n)
{
    uint32_t off = (uint32_t)blob_pos;
    room(n);
    memcpy(blob_buf + blob_pos, data, n);
    blob_pos = (blob_pos + n + 15u) & ~(size_t)15u;
    return off;
}

static uint32_t put_floats(unsigned n, float scale)
{
    uint32_t off = (uint32_t)blob_pos;
    unsigned i;
    room(4u * (size_t)n);
    for (i = 0; i < n; i++) {
        float v = frand() * scale;
        memcpy(blob_buf + blob_pos + 4u * i, &v, sizeof v);
    }
    blob_pos = (blob_pos + 4u * n + 15u) & ~(size_t)15u;
    return off;
}

static uint32_t crc32_ref(const unsigned char *p, size_t n)
{
    uint32_t c = 0xFFFFFFFFu;
    while (n--) {
        int b;
        c ^= *p++;
        for (b = 0; b < 8; b++) c = (c >> 1) ^ (0xEDB88320u & (0u - (c & 1u)));
    }
    return ~c;
}

enum { CLASSES = 6 };

/* op record */
static void op_rec(unsigned i, unsigned type, unsigned flags, unsigned kh, unsigned kw, unsigned sh,
                   unsigned sw, unsigned ph, unsigned pw, unsigned cin, unsigned cout, unsigned hin,
                   unsigned hout, uint32_t w_off, uint32_t w_bytes, uint32_t b_off, uint32_t s_off,
                   unsigned k, unsigned cout_pad)
{
    unsigned char *r = blob_buf + 64 + 64 * i;
    memset(r, 0, 64);
    r[0] = (unsigned char)type;
    r[1] = (unsigned char)flags;
    r[2] = (unsigned char)kh;
    r[3] = (unsigned char)kw;
    r[4] = (unsigned char)sh;
    r[5] = (unsigned char)sw;
    r[6] = (unsigned char)ph;
    r[7] = (unsigned char)pw;
    wr16(r + 8, cin);
    wr16(r + 10, cout);
    wr16(r + 12, hin);
    wr16(r + 14, hout);
    wr32(r + 16, w_off);
    wr32(r + 20, w_bytes);
    wr32(r + 24, b_off);
    wr32(r + 28, s_off);
    wr16(r + 32, k);
    wr16(r + 34, cout_pad);
}

/* matrix op (PW/HEAD), packed [cout_pad/8][K][8] */
static void mat_op(unsigned i, unsigned type, unsigned flags, unsigned kh, unsigned cin, unsigned cout,
                   unsigned hin, unsigned hout, int int8)
{
    unsigned k = kh * cin, cout_pad = (cout + 7u) & ~7u, n = cout_pad * k, j;
    uint32_t w_off, s_off = 0, b_off = 0;
    if (int8) {
        w_off = (uint32_t)blob_pos;
        room(n);
        for (j = 0; j < n; j++) blob_buf[blob_pos + j] = (unsigned char)(signed char)(int)(frand() * 127.0f);
        blob_pos = (blob_pos + n + 15u) & ~(size_t)15u;
        s_off = (uint32_t)blob_pos;
        room(4u * (size_t)cout_pad);
        for (j = 0; j < cout_pad; j++) {
            float sc = 0.01f + 0.004f * (float)(j % 5);
            memcpy(blob_buf + blob_pos + 4u * j, &sc, sizeof sc);
        }
        blob_pos = (blob_pos + 4u * cout_pad + 15u) & ~(size_t)15u;
        flags |= 8;
    } else {
        w_off = put_floats(n, 0.5f);
    }
    if (flags & 4) b_off = put_floats(cout_pad, 0.1f);
    op_rec(i, type, flags, kh, 1, 1, 1, 0, 0, cin, cout, hin, hout, w_off, n * (int8 ? 1u : 4u),
           b_off, s_off, k, cout_pad);
}

/* height 8, stride 4: stem 3x3/2 (1->8, 8->4 rows), dw 3x3/2 (4->2), pw 8->C1,
 * collapse 2x1 (C1->16, 2->1), dw 1x5, pw residual, head 16->6. C1 = 16, or 272 when
 * wide, which makes the collapse's reduction length K = 544 > 512 (the w8i chunk). */
static size_t build_blob_sc(int int8, unsigned s, unsigned c1);
static size_t build_blob_c1(int int8, unsigned c1) { return build_blob_sc(int8, 8, c1); }
static size_t build_blob(int int8) { return build_blob_c1(int8, 16); }

/* s: the stem's and the first depthwise layer's channels: 8, or 32 so that their NEON paths
 * (16 channels a pass) run twice per position */
static size_t build_blob_sc(int int8, unsigned s, unsigned c1)
{
    unsigned char vocab[64];
    const char *chars[CLASSES - 1] = {"a", "\xe3\x81\x82", "b", "\xe3\x83\xbc", "!"};
    unsigned i, pos = 0, n_ops = 7;
    uint32_t off;
    memset(blob_buf, 0, sizeof blob_buf);
    blob_pos = 64 + 64 * n_ops;
    lcg = 12345u;
    off = put_floats(9 * s, 0.6f);
    op_rec(0, 1, 1 | 4, 3, 3, 2, 2, 1, 1, 1, s, 8, 4, off, 9 * s * 4, put_floats(s, 0.1f), 0, 9, s);
    off = put_floats(9 * s, 0.6f);
    op_rec(1, 2, 1 | 4, 3, 3, 2, 2, 1, 1, s, s, 4, 2, off, 9 * s * 4, put_floats(s, 0.1f), 0, 9, s);
    mat_op(2, 3, 1 | 4, 1, s, c1, 2, 2, int8);
    mat_op(3, 3, 1 | 4 | 16, 2, c1, 16, 2, 1, int8);
    off = put_floats(5 * 16, 0.6f);
    op_rec(4, 2, 1 | 4, 1, 5, 1, 1, 0, 2, 16, 16, 1, 1, off, 5 * 16 * 4, put_floats(16, 0.1f), 0, 5, 16);
    mat_op(5, 3, 1 | 2 | 4, 1, 16, 16, 1, 1, int8);
    mat_op(6, 4, 4, 1, 16, CLASSES, 1, 1, int8);
    /* vocab: offsets then bytes */
    wr16(vocab, 0);
    wr16(vocab + 2, 0);
    for (i = 0; i < CLASSES - 1; i++) {
        size_t n = strlen(chars[i]);
        memcpy(vocab + 2 * (CLASSES + 1) + pos, chars[i], n);
        pos += (unsigned)n;
        wr16(vocab + 2 * (i + 2), pos);
    }
    off = put(vocab, 2 * (CLASSES + 1) + pos);
    memcpy(blob_buf, "VOCR", 4);
    wr16(blob_buf + 4, 1);
    wr16(blob_buf + 6, 64);
    wr32(blob_buf + 8, (uint32_t)blob_pos);
    wr32(blob_buf + 12, int8 ? 1u : 0u);
    wr16(blob_buf + 16, 8);
    wr16(blob_buf + 18, 256);
    wr16(blob_buf + 20, 4);
    wr16(blob_buf + 22, n_ops);
    wr32(blob_buf + 24, CLASSES);
    wr32(blob_buf + 28, 64);
    wr32(blob_buf + 32, off);
    wr32(blob_buf + 36, pos);
    wr32(blob_buf + 40, crc32_ref(blob_buf + 64, blob_pos - 64));
    return blob_pos;
}

/* The head right after the stem, no frame-row layer between them (audit-head-gate F1): height 4, a 4x3 conv
 * of stride 2 (1->16 channels, 4->1 rows), head 16->6. The front's last op appends the frames to the head's
 * accumulator itself. */
static size_t build_stem_head_blob(int int8)
{
    unsigned char vocab[64];
    const char *chars[CLASSES - 1] = {"a", "\xe3\x81\x82", "b", "\xe3\x83\xbc", "!"};
    unsigned i, pos = 0, n_ops = 2;
    uint32_t off;
    memset(blob_buf, 0, sizeof blob_buf);
    blob_pos = 64 + 64 * n_ops;
    lcg = 4242u;
    off = put_floats(12 * 16, 0.6f);
    op_rec(0, 1, 1 | 4, 4, 3, 1, 2, 0, 1, 1, 16, 4, 1, off, 12 * 16 * 4, put_floats(16, 0.1f), 0, 12, 16);
    mat_op(1, 4, 4, 1, 16, CLASSES, 1, 1, int8);
    wr16(vocab, 0);
    wr16(vocab + 2, 0);
    for (i = 0; i < CLASSES - 1; i++) {
        size_t n = strlen(chars[i]);
        memcpy(vocab + 2 * (CLASSES + 1) + pos, chars[i], n);
        pos += (unsigned)n;
        wr16(vocab + 2 * (i + 2), pos);
    }
    off = put(vocab, 2 * (CLASSES + 1) + pos);
    memcpy(blob_buf, "VOCR", 4);
    wr16(blob_buf + 4, 1);
    wr16(blob_buf + 6, 64);
    wr32(blob_buf + 8, (uint32_t)blob_pos);
    wr32(blob_buf + 12, int8 ? 1u : 0u);
    wr16(blob_buf + 16, 4);
    wr16(blob_buf + 18, 256);
    wr16(blob_buf + 20, 2);
    wr16(blob_buf + 22, n_ops);
    wr32(blob_buf + 24, CLASSES);
    wr32(blob_buf + 28, 64);
    wr32(blob_buf + 32, off);
    wr32(blob_buf + 36, pos);
    wr32(blob_buf + 40, crc32_ref(blob_buf + 64, blob_pos - 64));
    return blob_pos;
}

/* ------------------------------------------------------------ helpers */

static unsigned char arena_a[1 << 20] __attribute__((aligned(16)));
static unsigned char arena_b[1 << 20] __attribute__((aligned(16)));
static unsigned char img_px[40 * 300 * 4];

static void make_image(unsigned w, unsigned h, int rgba)
{
    unsigned x, y, c, bpp = rgba ? 4u : 1u;
    for (y = 0; y < h; y++)
        for (x = 0; x < w; x++)
            for (c = 0; c < bpp; c++)
                img_px[(y * w + x) * bpp + c] =
                    (unsigned char)((x * 7u + y * 13u + c * 31u + ((x / 5u) % 3u) * 60u) & 255u);
}

static int run_once(const vocr_model *m, vocr_opts o, unsigned char *arena, unsigned w, unsigned h,
                    int fmt, char *text, size_t cap, float *logits, unsigned cap_frames,
                    vocr_extra *ex_out)
{
    vocr_ctx c;
    vocr_image im;
    vocr_extra ex;
    size_t need = vocr_arena_bytes(m, &o);
    int rc;
    if (!need) return VOCR_E_ARG;
    rc = vocr_init(&c, m, &o, arena, need);
    if (rc) return rc;
    im.pixels = img_px;
    im.width = w;
    im.height = h;
    im.stride = (size_t)w * ((fmt & ~VOCR_INVERT) == VOCR_RGBA8 ? 4u : 1u);
    im.format = fmt;
    memset(&ex, 0, sizeof ex);
    ex.logits = logits;
    ex.cap_frames = cap_frames;
    rc = vocr_run(&c, &im, text, cap, &ex);
    if (ex_out) *ex_out = ex;
    return rc;
}

static int stop_now(void *user)
{
    int *calls = (int *)user;
    return ++*calls >= 2;
}

/* vocr_run with vocr_extra.topk = k (cap frames of buffers) */
static int run_topk(const vocr_model *m, vocr_opts o, unsigned w, unsigned h, int fmt, char *text, size_t cap,
                    unsigned k, float *tv, uint16_t *tc, float *bl, unsigned cap_frames, vocr_extra *ex_out)
{
    vocr_ctx c;
    vocr_image im;
    vocr_extra ex;
    size_t need = vocr_arena_bytes(m, &o);
    int rc;
    if (!need) return VOCR_E_ARG;
    rc = vocr_init(&c, m, &o, arena_b, need);
    if (rc) return rc;
    im.pixels = img_px;
    im.width = w;
    im.height = h;
    im.stride = (size_t)w * (fmt == VOCR_RGBA8 ? 4u : 1u);
    im.format = fmt;
    memset(&ex, 0, sizeof ex);
    ex.cap_frames = cap_frames;
    ex.topk = k;
    ex.topk_logit = tv;
    ex.topk_class = tc;
    ex.blank_logit = bl;
    rc = vocr_run(&c, &im, text, cap, &ex);
    if (ex_out) *ex_out = ex;
    return rc;
}

/* The per-frame top-K equals the K largest logits (ties: the lower class first), the blank logit is
 * class 0's, the text and the arena do not change, and the confidence lies in (0, 1]. */
static void test_topk(int int8)
{
    vocr_model m;
    vocr_opts o;
    char ref[256], text[256];
    float lref[80 * CLASSES], tv[80 * CLASSES], bl[80];
    uint16_t tc[80 * CLASSES];
    vocr_extra ex, eg;
    const size_t n = build_blob(int8);
    const unsigned ks[4] = {1, 2, 3, CLASSES}, tiles[3][2] = {{0, 0}, {1, 1}, {5, 7}};
    unsigned ki, ti, f, j, w = 157u, h = 19u;
    CHECK(vocr_load(&m, blob_buf, n) == VOCR_OK);
    make_image(w, h, 0);
    memset(&o, 0, sizeof o);
    o.mode = int8 ? VOCR_MODE_W8I : VOCR_MODE_F32;
    memset(lref, 0, sizeof lref);
    CHECK(run_once(&m, o, arena_a, w, h, VOCR_GREY8, ref, sizeof ref, lref, 80, &eg) == VOCR_OK);
    for (ki = 0; ki < 4; ki++)
        for (ti = 0; ti < 3; ti++) {
            const unsigned k = ks[ki];
            vocr_opts ot = o;
            ot.tile_frames = tiles[ti][0];
            ot.head_frames = tiles[ti][1];
            CHECK(run_topk(&m, ot, w, h, VOCR_GREY8, text, sizeof text, k, tv, tc, bl, 80, &ex) == VOCR_OK);
            CHECK(strcmp(text, ref) == 0);
            CHECK(ex.frames == eg.frames && ex.chars == eg.chars);
            CHECK(ex.conf_min > 0.0f && ex.conf_min <= ex.conf_mean && ex.conf_mean <= 1.0f);
            if (k == 1) CHECK(ex.conf_min == 1.0f && ex.conf_mean == 1.0f);
            for (f = 0; f < ex.frames; f++) {
                const float *row = lref + (size_t)f * CLASSES;
                unsigned used = 0;
                CHECK(memcmp(&bl[f], &row[0], sizeof(float)) == 0);
                for (j = 0; j < k; j++) {
                    unsigned c, bc = CLASSES;
                    for (c = 0; c < CLASSES; c++) /* the reference: the largest unused, first on ties */
                        if (!(used & (1u << c)) && (bc == CLASSES || row[c] > row[bc])) bc = c;
                    used |= 1u << bc;
                    CHECK(tc[f * k + j] == bc);
                    CHECK(memcmp(&tv[f * k + j], &row[bc], sizeof(float)) == 0);
                }
            }
        }
    /* errors: buffers too short, K too large, a missing buffer */
    CHECK(run_topk(&m, o, w, h, VOCR_GREY8, text, sizeof text, 2, tv, tc, bl, eg.frames - 1, NULL) == VOCR_E_ARG);
    CHECK(run_topk(&m, o, w, h, VOCR_GREY8, text, sizeof text, CLASSES + 1, tv, tc, bl, 80, NULL) == VOCR_E_ARG);
    CHECK(run_topk(&m, o, w, h, VOCR_GREY8, text, sizeof text, VOCR_TOPK_MAX + 1, tv, tc, bl, 80, NULL) ==
          VOCR_E_ARG);
    CHECK(run_topk(&m, o, w, h, VOCR_GREY8, text, sizeof text, 2, NULL, tc, bl, 80, NULL) == VOCR_E_ARG);
    CHECK(run_topk(&m, o, w, h, VOCR_GREY8, text, sizeof text, 2, tv, tc, NULL, 80, NULL) == VOCR_OK);
}

/* ------------------------------------------------------------ tests */

static void test_load(int int8)
{
    vocr_model m;
    size_t n = build_blob(int8);
    unsigned len;
    CHECK(vocr_check_crc(blob_buf, n) == VOCR_OK);
    CHECK(vocr_load(&m, blob_buf, n) == VOCR_OK);
    CHECK(m.n_ops == 7 && m.n_classes == CLASSES && m.height == 8 && m.stride == 4);
    CHECK(m.int8 == (unsigned)(int8 != 0));
    CHECK(strncmp(vocr_class_text(&m, 2, &len), "\xe3\x81\x82", 3) == 0 && len == 3);
    CHECK(vocr_class_text(&m, 0, &len)[0] == 0 && len == 0);
    CHECK(vocr_class_text(&m, CLASSES, &len)[0] == 0 && len == 0);
    /* truncated, misaligned, bad magic, bad version, bad op field */
    CHECK(vocr_load(&m, blob_buf, n - 1) == VOCR_E_FORMAT);
    CHECK(vocr_load(&m, blob_buf, 10) == VOCR_E_FORMAT);
    CHECK(vocr_load(&m, blob_buf + 4, n - 4) == VOCR_E_ALIGN);
    CHECK(vocr_load(NULL, blob_buf, n) == VOCR_E_ARG);
    blob_buf[0] = 'X';
    CHECK(vocr_load(&m, blob_buf, n) == VOCR_E_FORMAT);
    CHECK(vocr_check_crc(blob_buf, n) == VOCR_E_FORMAT);
    blob_buf[0] = 'V';
    blob_buf[4] = 2;
    CHECK(vocr_load(&m, blob_buf, n) == VOCR_E_FORMAT);
    blob_buf[4] = 1;
    blob_buf[64 + 64 * 2 + 10] = 17; /* pw cout 16 -> 17 */
    CHECK(vocr_load(&m, blob_buf, n) == VOCR_E_FORMAT);
    blob_buf[64 + 64 * 2 + 10] = 16;
    blob_buf[64 + 64 * 3 + 40] = 1; /* reserved byte */
    CHECK(vocr_load(&m, blob_buf, n) == VOCR_E_FORMAT);
    CHECK(vocr_check_crc(blob_buf, n) == VOCR_E_FORMAT);
    blob_buf[64 + 64 * 3 + 40] = 0;
    blob_buf[64 + 64 * 5 + 1] &= (unsigned char)~2u; /* toggling residual keeps it loadable */
    CHECK(vocr_load(&m, blob_buf, n) == VOCR_OK);
    blob_buf[64 + 64 * 5 + 1] |= 2u;
    blob_buf[64 + 64 * 2 + 1] |= 2u; /* residual after a strided dw: invalid */
    CHECK(vocr_load(&m, blob_buf, n) == VOCR_E_FORMAT);
    blob_buf[64 + 64 * 2 + 1] &= (unsigned char)~2u;
    CHECK(vocr_load(&m, blob_buf, n) == VOCR_OK);
    CHECK(vocr_check_crc(blob_buf, n) == VOCR_OK);
}

/* FNV-1a 64 over bytes */
static uint64_t fnv64(uint64_t h, const void *p, size_t n)
{
    const unsigned char *b = (const unsigned char *)p;
    while (n--) {
        h ^= *b++;
        h *= 0x100000001b3ull;
    }
    return h;
}

/* Runs the blob of build_blob_sc(int8, sc, c1) in every mode and format, at every tiling. `want` is
 * the known answer: the FNV-1a 64 of every reference run's logits and text, as main's scalar code
 * computes them (any build gives the same bits). The modes and tilings are only compared with each
 * other, and the stem and depthwise kernels are the same code in every mode, so a wrong tap there
 * would pass every other check: this hash is their independent reference. With sc = 32 the stem's
 * and the 3x3 depthwise's NEON paths (16 channels a pass) run twice per position, and the 1x5
 * depthwise (16 channels) takes the kw != 3 path. */
static void test_run_sc(int int8, unsigned sc, unsigned c1, uint64_t want);
static void test_run(int int8, unsigned c1, uint64_t want) { test_run_sc(int8, 8, c1, want); }

static void test_run_sc(int int8, unsigned sc, unsigned c1, uint64_t want)
{
    vocr_model m;
    vocr_opts o;
    char ref[256], text[256];
    float lref[80 * CLASSES], lcur[80 * CLASSES], lw8f[80 * CLASSES];
    vocr_extra ex;
    size_t n = build_blob_sc(int8, sc, c1);
    const int modes_i8[2] = {VOCR_MODE_W8F, VOCR_MODE_W8I};
    int mi, nmodes = int8 ? 2 : 1, fmt;
    unsigned tiles[5][2] = {{1, 1}, {2, 3}, {3, 1}, {5, 7}, {64, 64}};
    uint64_t kh = 0xcbf29ce484222325ull;
    CHECK(vocr_load(&m, blob_buf, n) == VOCR_OK);
    memset(&o, 0, sizeof o);
    o.mode = int8 ? VOCR_MODE_F32 : VOCR_MODE_W8F; /* wrong mode for this blob */
    CHECK(vocr_arena_bytes(&m, &o) == 0);
    for (fmt = VOCR_GREY8; fmt <= VOCR_RGBA8; fmt += 3)
        for (mi = 0; mi < nmodes; mi++) {
            unsigned t, w = fmt == VOCR_GREY8 ? 257u : 93u, h = 19u;
            int rc;
            make_image(w, h, fmt == VOCR_RGBA8);
            memset(&o, 0, sizeof o);
            o.mode = int8 ? modes_i8[mi] : VOCR_MODE_F32;
            memset(lref, 0, sizeof lref);
            rc = run_once(&m, o, arena_a, w, h, fmt, ref, sizeof ref, lref, 80, &ex);
            CHECK(rc == VOCR_OK);
            CHECK(ex.width == vocr_resized_width(&m, w, h));
            CHECK(ex.frames == (ex.width + 3u) / 4u);
            CHECK(ex.frames <= 80);
            for (t = 0; t < ex.frames * CLASSES; t++) CHECK(isfinite(lref[t]));
            kh = fnv64(kh, lref, (size_t)ex.frames * CLASSES * sizeof(float));
            kh = fnv64(kh, ref, strlen(ref) + 1);
            if (o.mode == VOCR_MODE_W8F) memcpy(lw8f, lref, sizeof lw8f);
            if (o.mode == VOCR_MODE_W8I) { /* int16 activations stay close to float ones */
                float mx = 0.0f, d = 0.0f;
                for (t = 0; t < ex.frames * CLASSES; t++) {
                    float a = fabsf(lw8f[t]), e = fabsf(lref[t] - lw8f[t]);
                    if (a > mx) mx = a;
                    if (e > d) d = e;
                }
                CHECK(mx > 0.1f);
                CHECK(d <= 1e-3f * (mx > 1.0f ? mx : 1.0f));
            }
            for (t = 0; t < 5; t++) {
                vocr_opts ot = o;
                ot.tile_frames = tiles[t][0];
                ot.head_frames = tiles[t][1];
                memset(lcur, 0, sizeof lcur);
                CHECK(run_once(&m, ot, arena_b, w, h, fmt, text, sizeof text, lcur, 80, NULL) == VOCR_OK);
                CHECK(strcmp(text, ref) == 0);
                CHECK(memcmp(lcur, lref, sizeof lcur) == 0);
            }
            /* arena one byte short */
            {
                vocr_ctx c;
                size_t need = vocr_arena_bytes(&m, &o);
                CHECK(need > 0);
                CHECK(vocr_init(&c, &m, &o, arena_a, need - 1) == VOCR_E_ARENA);
                CHECK(vocr_init(&c, &m, &o, arena_a + 8, need) == VOCR_E_ALIGN);
            }
            /* text buffer too small: whole characters only, NUL-terminated */
            if (strlen(ref) > 2) {
                size_t cap = 3;
                rc = run_once(&m, o, arena_a, w, h, fmt, text, cap, NULL, 0, NULL);
                CHECK(rc == VOCR_E_TEXT);
                CHECK(strlen(text) < cap);
                CHECK(strncmp(text, ref, strlen(text)) == 0);
            }
        }
    if (kh != want) {
        fprintf(stderr, "test_run_sc(%d, %u, %u): logits hash %016llx, want %016llx\n", int8, sc, c1,
                (unsigned long long)kh, (unsigned long long)want);
        failures++;
    }
}

/* VOCR_INVERT reads every source pixel as 255 - grey before the resize (what
 * synth/common.line_crops does for dark text): the same input, text and logits as an
 * inverted grey copy of the crop, for grey and RGBA input. */
static void test_invert(void)
{
    static unsigned char inv[40 * 300];
    static unsigned char in_a[8 * 64], in_b[8 * 64];
    vocr_model m;
    vocr_opts o;
    vocr_ctx c;
    vocr_image im, iv;
    char t_inv[256], t_copy[256], t_plain[256];
    float l_inv[80 * CLASSES], l_copy[80 * CLASSES], l_plain[80 * CLASSES];
    vocr_extra ex;
    size_t n = build_blob(0), need;
    unsigned w = 93, h = 19, x, y, w1 = 0, w2 = 0, p1 = 0, p2 = 0;
    int rgba;
    CHECK(vocr_load(&m, blob_buf, n) == VOCR_OK);
    memset(&o, 0, sizeof o);
    need = vocr_arena_bytes(&m, &o);
    CHECK(vocr_init(&c, &m, &o, arena_a, need) == VOCR_OK);
    for (rgba = 0; rgba <= 1; rgba++) {
        const unsigned bpp = rgba ? 4u : 1u;
        make_image(w, h, rgba);
        for (y = 0; y < h; y++)
            for (x = 0; x < w; x++) {
                const unsigned char *p = img_px + (y * w + x) * bpp;
                const int g = rgba ? (77 * p[0] + 150 * p[1] + 29 * p[2]) >> 8 : p[0];
                inv[y * w + x] = (unsigned char)(255 - g);
            }
        im.pixels = img_px;
        im.width = w;
        im.height = h;
        im.stride = (size_t)w * bpp;
        im.format = (rgba ? VOCR_RGBA8 : VOCR_GREY8) | VOCR_INVERT;
        iv = im;
        iv.pixels = inv;
        iv.stride = w;
        iv.format = VOCR_GREY8;
        memset(&ex, 0, sizeof ex);
        ex.cap_frames = 80;
        memset(l_inv, 0, sizeof l_inv);
        ex.logits = l_inv;
        CHECK(vocr_run(&c, &im, t_inv, sizeof t_inv, &ex) == VOCR_OK);
        memset(l_copy, 0, sizeof l_copy);
        ex.logits = l_copy;
        CHECK(vocr_run(&c, &iv, t_copy, sizeof t_copy, &ex) == VOCR_OK);
        CHECK(strcmp(t_inv, t_copy) == 0);
        CHECK(memcmp(l_inv, l_copy, sizeof l_inv) == 0);
        im.format &= ~VOCR_INVERT; /* and the flag does change the result */
        memset(l_plain, 0, sizeof l_plain);
        ex.logits = l_plain;
        CHECK(vocr_run(&c, &im, t_plain, sizeof t_plain, &ex) == VOCR_OK);
        CHECK(memcmp(l_plain, l_inv, sizeof l_inv) != 0);
        im.format |= VOCR_INVERT;
        CHECK(vocr_preprocess(&m, &im, in_a, sizeof in_a, &w1, &p1) == VOCR_OK);
        CHECK(vocr_preprocess(&m, &iv, in_b, sizeof in_b, &w2, &p2) == VOCR_OK);
        CHECK(w1 == w2 && p1 == p2 && p1 * 8u <= sizeof in_a);
        CHECK(memcmp(in_a, in_b, (size_t)p1 * 8u) == 0);
    }
    im.format = VOCR_INVERT; /* the flag alone is no pixel format */
    CHECK(vocr_run(&c, &im, t_inv, sizeof t_inv, NULL) == VOCR_E_ARG);
    im.format = VOCR_GREY8 | 0x200; /* unknown flag bits */
    CHECK(vocr_run(&c, &im, t_inv, sizeof t_inv, NULL) == VOCR_E_ARG);
}

static void test_limits(void)
{
    vocr_model m;
    vocr_opts o;
    vocr_ctx c;
    vocr_image im;
    vocr_extra ex;
    char text[64];
    int calls = 0;
    size_t n = build_blob(0), need;
    CHECK(vocr_load(&m, blob_buf, n) == VOCR_OK);
    memset(&o, 0, sizeof o);
    /* streaming: the arena does not grow with the max width once it covers a tile and
     * a head batch (16 frames = 64 px here); below that the batches shrink with it */
    o.max_width = 64;
    need = vocr_arena_bytes(&m, &o);
    o.max_width = 256;
    CHECK(vocr_arena_bytes(&m, &o) == need);
    o.max_width = 32;
    CHECK(vocr_arena_bytes(&m, &o) < need);
    need = vocr_arena_bytes(&m, &o);
    /* a line wider than the planned max width */
    o.max_width = 32;
    CHECK(vocr_init(&c, &m, &o, arena_a, need) == VOCR_OK);
    make_image(300, 8, 0);
    im.pixels = img_px;
    im.width = 300;
    im.height = 8;
    im.stride = 300;
    im.format = VOCR_GREY8;
    CHECK(vocr_run(&c, &im, text, sizeof text, NULL) == VOCR_E_WIDTH);
    /* resized width: round half to even, clamped */
    CHECK(vocr_resized_width(&m, 3, 8) == 8);
    CHECK(vocr_resized_width(&m, 1000, 8) == 256);
    CHECK(vocr_resized_width(&m, 25, 2) == 100);
    CHECK(vocr_resized_width(&m, 5, 16) == 8);   /* 2.5 -> 2 -> clamp 8 */
    CHECK(vocr_resized_width(&m, 21, 16) == 10); /* 10.5 -> 10 */
    CHECK(vocr_resized_width(&m, 23, 16) == 12); /* 11.5 -> 12 */
    /* bad images */
    o.max_width = 0;
    CHECK(vocr_init(&c, &m, &o, arena_a, vocr_arena_bytes(&m, &o)) == VOCR_OK);
    im.stride = 299;
    CHECK(vocr_run(&c, &im, text, sizeof text, NULL) == VOCR_E_ARG);
    im.stride = 300;
    im.format = 3;
    CHECK(vocr_run(&c, &im, text, sizeof text, NULL) == VOCR_E_ARG);
    im.format = VOCR_GREY8;
    im.height = 0;
    CHECK(vocr_run(&c, &im, text, sizeof text, NULL) == VOCR_E_ARG);
    im.height = 8;
    CHECK(vocr_run(&c, &im, text, 0, NULL) == VOCR_E_ARG);
    /* tiny crops work */
    im.width = 1;
    im.height = 1;
    CHECK(vocr_run(&c, &im, text, sizeof text, NULL) == VOCR_OK);
    /* a crop taller than 32767 rows (one column): row indices must not wrap */
    {
        static unsigned char tall[40000];
        unsigned char col[8 * 8];
        unsigned y, w = 0, pw = 0;
        for (y = 0; y < 40000; y++) tall[y] = (unsigned char)(y >= 35000 ? 255 : 0);
        im.pixels = tall;
        im.width = 1;
        im.height = 40000;
        im.stride = 1;
        CHECK(vocr_run(&c, &im, text, sizeof text, NULL) == VOCR_OK);
        CHECK(vocr_preprocess(&m, &im, col, sizeof col, &w, &pw) == VOCR_OK && w == 8 && pw == 8);
        CHECK(col[0] == 0 && col[7 * 8] == 255); /* output row 7 samples source rows 37499-37500 */
        im.pixels = img_px;
        im.stride = 300;
    }
    /* cancellation, polled per tile */
    im.width = 300;
    im.height = 8;
    memset(&ex, 0, sizeof ex);
    ex.cancel = stop_now;
    ex.user = &calls;
    CHECK(vocr_run(&c, &im, text, sizeof text, &ex) == VOCR_E_CANCELLED);
    CHECK(text[0] == 0 && calls == 2);
    CHECK(strcmp(vocr_strerror(VOCR_E_ARENA), "arena too small") == 0);
}


/* The head's argmax is the first maximum of a frame's logits, as the scalar scan
 * `if (v > best) best = v` gives it: on ties the lower class, -0 equal to +0, a NaN never
 * chosen, all -inf (or NaN) the blank. A 21-class head with zero weights makes every
 * frame's logits its biases, so each case sets them directly; in W8I exactly, sign of zero
 * included: 0 * (scale * sx) is -0 with a negative scale, and -0 + bias is bias. NEON's W8I
 * path keeps that scan in vector lanes (vocr.c lanes_argmax) and must agree. */
enum { HCLASSES = 21 };

static size_t build_head_blob(const float *bias, int int8)
{
    unsigned char vocab[2 * (HCLASSES + 1) + HCLASSES];
    unsigned i, pos = 0, n_ops = 7, cout_pad = (HCLASSES + 7u) & ~7u, n;
    uint32_t off, w_off, s_off = 0, b_off;
    float bpad[24];
    memset(blob_buf, 0, sizeof blob_buf);
    blob_pos = 64 + 64 * n_ops;
    lcg = 12345u;
    off = put_floats(9 * 8, 0.6f);
    op_rec(0, 1, 1 | 4, 3, 3, 2, 2, 1, 1, 1, 8, 8, 4, off, 9 * 8 * 4, put_floats(8, 0.1f), 0, 9, 8);
    off = put_floats(9 * 8, 0.6f);
    op_rec(1, 2, 1 | 4, 3, 3, 2, 2, 1, 1, 8, 8, 4, 2, off, 9 * 8 * 4, put_floats(8, 0.1f), 0, 9, 8);
    mat_op(2, 3, 1 | 4, 1, 8, 16, 2, 2, int8);
    mat_op(3, 3, 1 | 4 | 16, 2, 16, 16, 2, 1, int8);
    off = put_floats(5 * 16, 0.6f);
    op_rec(4, 2, 1 | 4, 1, 5, 1, 1, 0, 2, 16, 16, 1, 1, off, 5 * 16 * 4, put_floats(16, 0.1f), 0, 5, 16);
    mat_op(5, 3, 1 | 2 | 4, 1, 16, 16, 1, 1, int8);
    /* the head: zero weights, the given biases */
    n = cout_pad * 16u;
    w_off = (uint32_t)blob_pos;
    room((size_t)n * (int8 ? 1u : 4u));
    memset(blob_buf + blob_pos, 0, (size_t)n * (int8 ? 1u : 4u));
    blob_pos = (blob_pos + (size_t)n * (int8 ? 1u : 4u) + 15u) & ~(size_t)15u;
    if (int8) {
        s_off = (uint32_t)blob_pos;
        room(4u * (size_t)cout_pad);
        for (i = 0; i < cout_pad; i++) {
            const float sc = -0.01f;
            memcpy(blob_buf + blob_pos + 4u * i, &sc, sizeof sc);
        }
        blob_pos = (blob_pos + 4u * cout_pad + 15u) & ~(size_t)15u;
    }
    memset(bpad, 0, sizeof bpad);
    memcpy(bpad, bias, HCLASSES * sizeof(float));
    b_off = put(bpad, 4u * cout_pad);
    op_rec(6, 4, 4 | (int8 ? 8 : 0), 1, 1, 1, 1, 0, 0, 16, HCLASSES, 1, 1, w_off, n * (int8 ? 1u : 4u), b_off,
           s_off, 16, cout_pad);
    wr16(vocab, 0);
    wr16(vocab + 2, 0);
    for (i = 1; i < HCLASSES; i++) {
        vocab[2 * (HCLASSES + 1) + pos++] = (unsigned char)('A' + i - 1);
        wr16(vocab + 2 * (i + 1), pos);
    }
    off = put(vocab, 2 * (HCLASSES + 1) + pos);
    memcpy(blob_buf, "VOCR", 4);
    wr16(blob_buf + 4, 1);
    wr16(blob_buf + 6, 64);
    wr32(blob_buf + 8, (uint32_t)blob_pos);
    wr32(blob_buf + 12, int8 ? 1u : 0u);
    wr16(blob_buf + 16, 8);
    wr16(blob_buf + 18, 256);
    wr16(blob_buf + 20, 4);
    wr16(blob_buf + 22, n_ops);
    wr32(blob_buf + 24, HCLASSES);
    wr32(blob_buf + 28, 64);
    wr32(blob_buf + 32, off);
    wr32(blob_buf + 36, pos);
    wr32(blob_buf + 40, crc32_ref(blob_buf + 64, blob_pos - 64));
    return blob_pos;
}

static float hlogits[80 * HCLASSES];
static float htv[80 * HCLASSES], hbl[80];
static uint16_t htc[80 * HCLASSES];

/* main's top-K: the classes in order, each inserted where it is strictly larger (vocr.c run_head) */
static void topk_ref(const float *v, unsigned n, unsigned k, float *tv, uint16_t *tc)
{
    unsigned j, p;
    for (j = 0; j < k; j++) {
        tv[j] = -INFINITY;
        tc[j] = 0;
    }
    for (j = 0; j < n; j++)
        if (v[j] > tv[k - 1]) {
            for (p = k - 1; p && v[j] > tv[p - 1]; p--) {
                tv[p] = tv[p - 1];
                tc[p] = tc[p - 1];
            }
            tv[p] = v[j];
            tc[p] = (uint16_t)j;
        }
}

static int same_float(float a, float b) { return (isnan(a) && isnan(b)) || !memcmp(&a, &b, sizeof a); }

static void test_head_argmax(void)
{
    const float ninf = -INFINITY, nan_ = NAN, low = -5.0f;
    struct {
        unsigned expect;
        unsigned n;            /* (class, value) pairs; every other class is `low` */
        unsigned cls[4];
        float val[4];
        int all_ninf;          /* every other class -inf instead */
    } cases[] = {
        {9, 2, {9, 17}, {3.0f, 3.0f}, 0},            /* a tie across a full and the partial block */
        {3, 2, {3, 5}, {2.0f, 2.0f}, 0},             /* a tie inside a block */
        {2, 3, {12, 4, 2}, {1.0f, 1.0f, 1.0f}, 0},   /* lanes 4 and 2 of blocks 0 and 1 */
        {12, 3, {12, 20, 4}, {1.5f, 1.5f, 1.0f}, 0}, /* the first of two maxima, in block 1 */
        {20, 1, {20}, {4.0f}, 0},                    /* the partial block's last class */
        {1, 2, {1, 3}, {0.0f, -0.0f}, 0},            /* +0 then -0: equal, the first */
        {1, 2, {1, 3}, {-0.0f, 0.0f}, 0},            /* -0 then +0: equal, the first */
        {10, 2, {5, 10}, {0.0f, 1.0f}, 0},           /* a NaN is never chosen (set below) */
        {0, 1, {0}, {0.0f}, 1},                      /* all -inf: the blank */
        {0, 2, {0, 7}, {0.0f, 0.0f}, 1},             /* NaN and -inf only: the blank (set below) */
        {16, 1, {16}, {0.5f}, 0},                    /* the partial block's first class */
    };
    const int modes[3] = {VOCR_MODE_W8I, VOCR_MODE_W8F, VOCR_MODE_F32};
    unsigned ci, f;
    int mi;
    make_image(201, 19, 0);
    for (ci = 0; ci < sizeof cases / sizeof cases[0]; ci++)
        for (mi = 0; mi < 3; mi++) {
            float bias[HCLASSES];
            uint16_t best[80];
            vocr_model m;
            vocr_opts o;
            vocr_ctx c;
            vocr_image im;
            vocr_extra ex;
            char text[64], want[2] = {0, 0};
            unsigned j;
            for (j = 0; j < HCLASSES; j++) bias[j] = cases[ci].all_ninf ? ninf : low;
            for (j = 0; j < cases[ci].n; j++) bias[cases[ci].cls[j]] = cases[ci].val[j];
            if (ci == 7) bias[5] = nan_;
            if (ci == 9) {
                bias[0] = nan_;
                bias[7] = nan_;
            }
            CHECK(vocr_load(&m, blob_buf, build_head_blob(bias, mi < 2)) == VOCR_OK);
            memset(&o, 0, sizeof o);
            o.mode = modes[mi];
            o.head_frames = ci % 2 ? 16u : 5u;  /* lane and partial batches of 4 and of 1-3 frames */
            CHECK(vocr_init(&c, &m, &o, arena_a, vocr_arena_bytes(&m, &o)) == VOCR_OK);
            im.pixels = img_px;
            im.width = 201;
            im.height = 19;
            im.stride = 201;
            im.format = VOCR_GREY8;
            memset(&ex, 0, sizeof ex);
            memset(best, 0xFF, sizeof best);
            for (j = 0; j < 80 * HCLASSES; j++) hlogits[j] = 12345.0f;
            ex.best = best;
            ex.logits = hlogits;
            ex.cap_frames = 80;
            CHECK(vocr_run(&c, &im, text, sizeof text, &ex) == VOCR_OK);
            CHECK(ex.frames > 20 && ex.frames <= 80);
            /* every logit is its class's bias (NEON W8I copies them out of its lane path) */
            for (f = 0; f < ex.frames; f++)
                for (j = 0; j < HCLASSES; j++) {
                    const float v = hlogits[f * HCLASSES + j], b = bias[j];
                    if (!(v == b || (isnan(v) && isnan(b)))) {
                        fprintf(stderr, "head case %u mode %d frame %u class %u: logit %g, bias %g\n", ci,
                                modes[mi], f, j, (double)v, (double)b);
                        failures++;
                        f = ex.frames;
                        break;
                    }
                }
            for (f = 0; f < ex.frames; f++)
                if (best[f] != cases[ci].expect) {
                    fprintf(stderr, "head case %u mode %d frame %u: class %u, want %u\n", ci, modes[mi], f,
                            best[f], cases[ci].expect);
                    failures++;
                    break;
                }
            if (cases[ci].expect) want[0] = (char)('A' + cases[ci].expect - 1);
            CHECK(strcmp(text, want) == 0);
            /* the top K of every frame, as main's per-class insertion gives it (NEON W8I skips the blocks
             * with no logit above a frame's K-th largest): K 1, 3, 8 and the largest allowed */
            {
                static const unsigned ks[4] = {1, 3, 8, VOCR_TOPK_MAX < HCLASSES ? VOCR_TOPK_MAX : HCLASSES};
                unsigned ki;
                for (ki = 0; ki < 4; ki++) {
                    const unsigned k = ks[ki];
                    float rtv[HCLASSES];
                    uint16_t rtc[HCLASSES];
                    char ktext[64];
                    memset(&ex, 0, sizeof ex);
                    memset(best, 0xFF, sizeof best);
                    for (j = 0; j < 80 * HCLASSES; j++) {
                        hlogits[j] = 12345.0f;
                        htv[j] = 12345.0f;
                        htc[j] = 0xFFFF;
                    }
                    ex.best = best;
                    ex.logits = hlogits;
                    ex.cap_frames = 80;
                    ex.topk = k;
                    ex.topk_logit = htv;
                    ex.topk_class = htc;
                    ex.blank_logit = hbl;
                    CHECK(vocr_run(&c, &im, ktext, sizeof ktext, &ex) == VOCR_OK);
                    CHECK(strcmp(ktext, text) == 0);
                    for (f = 0; f < ex.frames; f++) {
                        const float *row = hlogits + f * HCLASSES;
                        int bad = best[f] != cases[ci].expect || !same_float(hbl[f], row[0]);
                        for (j = 0; j < HCLASSES; j++) /* the biases, -0 and +0 alike (W8F's sums give +0) */
                            bad |= !(row[j] == bias[j] || (isnan(row[j]) && isnan(bias[j])));
                        topk_ref(row, HCLASSES, k, rtv, rtc);
                        for (j = 0; j < k; j++) bad |= htc[f * k + j] != rtc[j] || !same_float(htv[f * k + j], rtv[j]);
                        if (bad) {
                            fprintf(stderr, "head case %u mode %d top-%u frame %u: wrong top-K, argmax, blank or "
                                    "logits\n", ci, modes[mi], k, f);
                            failures++;
                            break;
                        }
                    }
                }
            }
        }
}

/* ------------------------------------------------------------ the head gate (docs/headgate.md) */

/* Appends a VOCG section with random int8 layer-1 weights (hid x 16, the test head's K) to the int8 blob
 * of n bytes in blob_buf; returns the gated blob's bytes. */
static size_t append_gate(size_t n, unsigned hid, float t, float b2)
{
    const unsigned k = 16, s_w1 = 64, s_s1 = (64 + hid * k + 15u) & ~15u, s_b1 = s_s1 + ((4 * hid + 15u) & ~15u),
                   s_w2 = s_b1 + ((4 * hid + 15u) & ~15u), total = s_w2 + ((4 * hid + 15u) & ~15u);
    unsigned char *g = blob_buf + n;
    unsigned j;
    room(total);
    memset(g, 0, total);
    for (j = 0; j < hid * k; j++) g[s_w1 + j] = (unsigned char)(signed char)(int)(frand() * 127.0f);
    for (j = 0; j < hid; j++) {
        const float sc = 0.002f + 0.0005f * (float)(j % 7), b = frand() * 0.2f, w = frand();
        memcpy(g + s_s1 + 4 * j, &sc, 4);
        memcpy(g + s_b1 + 4 * j, &b, 4);
        memcpy(g + s_w2 + 4 * j, &w, 4);
    }
    memcpy(g, "VOCG", 4);
    wr16(g + 4, 1);
    wr16(g + 6, 64);
    wr32(g + 8, total);
    wr16(g + 16, k);
    wr16(g + 18, hid);
    wr32(g + 20, (uint32_t)n);
    memcpy(g + 24, blob_buf + 40, 4);
    memcpy(g + 28, &t, 4);
    memcpy(g + 32, &b2, 4);
    wr32(g + 36, s_w1);
    wr32(g + 40, s_s1);
    wr32(g + 44, s_b1);
    wr32(g + 48, s_w2);
    wr32(g + 52, crc32_ref(g + 64, total - 64));
    blob_pos = n + total;
    return n + total;
}

/* The gate's score of one frame's head input x (16 floats), written independently of vocr.c: quant16's
 * int16 grid, exact integer sums, then its float steps in order. */
static float gate_ref(const vocr_model *m, const float *x)
{
    const vocr_gate *g = &m->gate;
    const unsigned K = g->k;
    float mx = 0.0f, inv, sx, s = g->b2;
    int qx[64];
    unsigned j, k;
    for (k = 0; k < K; k++)
        if (fabsf(x[k]) > mx) mx = fabsf(x[k]);
    inv = mx > 0.0f ? 32767.0f / mx : 0.0f;
    sx = mx > 0.0f ? mx / 32767.0f : 0.0f;
    for (k = 0; k < K; k++) {
        const float t = x[k] * inv;
        int v = (int)(t + (signbit(t) ? -0.5f : 0.5f));
        qx[k] = v > 32767 ? 32767 : v < -32767 ? -32767 : v;
    }
    for (j = 0; j < g->hidden; j++) {
        const int8_t *w = g->q + (size_t)(j / 8u) * K * 8u + j % 8u;
        int32_t acc = 0;
        float h;
        for (k = 0; k < K; k++) acc += (int32_t)w[(size_t)k * 8u] * qx[k];
        h = (float)acc * (g->scale[j] * sx) + g->bias[j];
        if (!(h > 0.0f)) h = 0.0f;
        s = s + g->w2[j] * h;
    }
    return s;
}

static float g_logits[80 * CLASSES], g_hin[80 * 16], g_score[80], g_tv[80 * CLASSES], g_bl[80];
static uint16_t g_tc[80 * CLASSES], g_best[80];

/* One W8I run of the blob in blob_buf (n bytes) with head_gate hg and these tiles; topk k (0: none). */
static int gate_run(size_t n, unsigned hg, unsigned tile, unsigned head, unsigned w, unsigned h, unsigned k,
                    int logits, char *text, size_t cap, vocr_extra *out)
{
    vocr_model m;
    vocr_opts o;
    vocr_ctx c;
    vocr_image im;
    vocr_extra ex;
    size_t need;
    int rc = vocr_load(&m, blob_buf, n);
    if (rc) return rc;
    memset(&o, 0, sizeof o);
    o.mode = VOCR_MODE_W8I;
    o.tile_frames = tile;
    o.head_frames = head;
    o.head_gate = hg;
    need = vocr_arena_bytes(&m, &o);
    if (!need) return VOCR_E_MODE;
    rc = vocr_init(&c, &m, &o, arena_a, need);
    if (rc) return rc;
    im.pixels = img_px;
    im.width = w;
    im.height = h;
    im.stride = w;
    im.format = VOCR_GREY8;
    memset(&ex, 0, sizeof ex);
    ex.cap_frames = 80;
    ex.best = g_best;
    ex.logits = logits ? g_logits : NULL;
    ex.head_in = g_hin;
    ex.gate_score = hg ? g_score : NULL;
    if (k) {
        ex.topk = k;
        ex.topk_logit = g_tv;
        ex.topk_class = g_tc;
        ex.blank_logit = g_bl;
    }
    rc = vocr_run(&c, &im, text, cap, &ex);
    if (out) *out = ex;
    return rc;
}

static void test_gate_load(void)
{
    vocr_model m;
    const size_t n = build_blob(1), gn = append_gate(n, 8, 0.25f, 0.5f);
    const unsigned char save8 = blob_buf[n + 8];
    CHECK(vocr_load(&m, blob_buf, gn) == VOCR_OK && vocr_has_gate(&m) == 1);
    CHECK(m.bytes == n && m.gate.bytes == gn - n && m.gate.k == 16 && m.gate.hidden == 8);
    CHECK(m.gate.t == 0.25f && m.gate.b2 == 0.5f);
    CHECK(vocr_check_crc(blob_buf, gn) == VOCR_OK);
    CHECK(vocr_load(&m, blob_buf, n) == VOCR_OK && vocr_has_gate(&m) == 0); /* the base alone: no gate */
    CHECK(vocr_has_gate(NULL) == 0);
    CHECK(vocr_load(&m, blob_buf, gn - 16) == VOCR_E_FORMAT);             /* a cut section */
    CHECK(vocr_check_crc(blob_buf, gn - 16) == VOCR_E_FORMAT);
    blob_buf[n + 100] ^= 1u;                                                 /* content: only the CRC sees it */
    CHECK(vocr_load(&m, blob_buf, gn) == VOCR_OK);
    CHECK(vocr_check_crc(blob_buf, gn) == VOCR_E_FORMAT);
    blob_buf[n + 100] ^= 1u;
    blob_buf[n + 3] = 'X';                                                   /* not a gate: ignored, as before */
    CHECK(vocr_load(&m, blob_buf, gn) == VOCR_OK && !vocr_has_gate(&m));
    CHECK(vocr_check_crc(blob_buf, gn) == VOCR_OK);
    blob_buf[n + 3] = 'G';
    blob_buf[n + 4] = 2;                                                     /* version */
    CHECK(vocr_load(&m, blob_buf, gn) == VOCR_E_FORMAT);
    blob_buf[n + 4] = 1;
    blob_buf[n + 16] = 15;                                                   /* k != the head's K */
    CHECK(vocr_load(&m, blob_buf, gn) == VOCR_E_FORMAT);
    blob_buf[n + 16] = 16;
    blob_buf[n + 18] = 12;                                                   /* hidden not a multiple of 8 */
    CHECK(vocr_load(&m, blob_buf, gn) == VOCR_E_FORMAT);
    blob_buf[n + 18] = 72;                                                   /* hidden above 64 */
    CHECK(vocr_load(&m, blob_buf, gn) == VOCR_E_FORMAT);
    blob_buf[n + 18] = 8;
    blob_buf[n + 20] ^= 16u;                                                 /* the base blob's bytes */
    CHECK(vocr_load(&m, blob_buf, gn) == VOCR_E_FORMAT);
    blob_buf[n + 20] ^= 16u;
    blob_buf[n + 24] ^= 1u;                                                  /* the base blob's CRC */
    CHECK(vocr_load(&m, blob_buf, gn) == VOCR_E_FORMAT);
    blob_buf[n + 24] ^= 1u;
    blob_buf[n + 8] = (unsigned char)(save8 + 8u);                           /* not a multiple of 16 */
    CHECK(vocr_load(&m, blob_buf, gn) == VOCR_E_FORMAT);
    blob_buf[n + 8] = save8;
    blob_buf[n + 37] = 0x40;                                                 /* w1 outside the section */
    CHECK(vocr_load(&m, blob_buf, gn) == VOCR_E_FORMAT);
    blob_buf[n + 37] = 0;
    blob_buf[n + 36] = 72;                                                   /* w1 misaligned */
    CHECK(vocr_load(&m, blob_buf, gn) == VOCR_E_FORMAT);
    blob_buf[n + 36] = 64;
    blob_buf[n + 60] = 1;                                                    /* reserved byte */
    CHECK(vocr_load(&m, blob_buf, gn) == VOCR_E_FORMAT);
    blob_buf[n + 60] = 0;
    blob_buf[n + 12] = 1;                                                    /* flags */
    CHECK(vocr_load(&m, blob_buf, gn) == VOCR_E_FORMAT);
    blob_buf[n + 12] = 0;
    {
        const float nan_ = NAN;
        float t;
        memcpy(&t, blob_buf + n + 28, 4);
        memcpy(blob_buf + n + 28, &nan_, 4);                                 /* a NaN threshold */
        CHECK(vocr_load(&m, blob_buf, gn) == VOCR_E_FORMAT);
        memcpy(blob_buf + n + 28, &t, 4);
    }
    CHECK(vocr_load(&m, blob_buf, gn) == VOCR_OK && vocr_has_gate(&m));
    CHECK(vocr_check_crc(blob_buf, gn) == VOCR_OK);
}

/* head_gate OFF on a gated blob is the blob without the section exactly; SCORE computes the reference's scores
 * and changes nothing; ON gives what the ungated run gives with the skipped frames (score > t) read as blanks,
 * at every tiling; options and modes it needs. build: the base blob (int8); want: the known answer of its W8I
 * run without the gate, the FNV-1a 64 of its logits and text as main's runtime computes them (0: not checked). */
static void test_gate_run_on(size_t (*build)(int), uint64_t want_hash)
{
    const unsigned w = 257, hgt = 10, tiles[6][2] = {{0, 0}, {1, 1}, {2, 3}, {3, 1}, {5, 7}, {64, 64}};
    const unsigned ks[3] = {0, 1, 3};
    char ref[256], text[256], want[256];
    vocr_extra ex, e0;
    vocr_model m;
    float logits0[80 * CLASSES], hin0[80 * 16], tv0[80 * CLASSES], bl0[80];
    uint16_t tc0[80 * CLASSES], best0[80];
    size_t n, gn;
    unsigned f, j, ti, ki, frames, skipped = 0;
    float t;
    make_image(w, hgt, 0);
    n = build(1);
    /* the base blob, W8I, top-3: the reference */
    CHECK(gate_run(n, 0, 0, 0, w, hgt, 3, 1, ref, sizeof ref, &e0) == VOCR_OK);
    frames = e0.frames;
    if (want_hash) {
        uint64_t kh = fnv64(0xcbf29ce484222325ull, g_logits, (size_t)frames * CLASSES * sizeof(float));
        kh = fnv64(kh, ref, strlen(ref) + 1);
        if (kh != want_hash) {
            fprintf(stderr, "test_gate_run_on: logits hash %016llx, want %016llx\n", (unsigned long long)kh,
                    (unsigned long long)want_hash);
            failures++;
        }
    }
    memcpy(logits0, g_logits, sizeof logits0);
    memcpy(hin0, g_hin, sizeof hin0);
    memcpy(tv0, g_tv, sizeof tv0);
    memcpy(tc0, g_tc, sizeof tc0);
    memcpy(bl0, g_bl, sizeof bl0);
    memcpy(best0, g_best, sizeof best0);
    CHECK(frames > 40 && frames <= 80);
    CHECK(gate_run(n, VOCR_HEAD_GATE_ON, 0, 0, w, hgt, 0, 0, text, sizeof text, NULL) == VOCR_E_MODE);
    gn = append_gate(n, 16, 0.0f, 0.1f);
    CHECK(vocr_load(&m, blob_buf, gn) == VOCR_OK && vocr_has_gate(&m));
    /* the threshold: the median score, so about half the frames are skipped */
    CHECK(gate_run(gn, VOCR_HEAD_GATE_SCORE, 0, 0, w, hgt, 3, 1, text, sizeof text, &ex) == VOCR_OK);
    CHECK(strcmp(text, ref) == 0 && ex.gated == 0);
    CHECK(memcmp(g_logits, logits0, sizeof logits0) == 0 && memcmp(g_tv, tv0, sizeof tv0) == 0);
    CHECK(memcmp(g_hin, hin0, (size_t)frames * 16 * sizeof(float)) == 0);
    for (f = 0; f < frames; f++) {
        const float r = gate_ref(&m, hin0 + f * 16);
        CHECK(memcmp(&r, &g_score[f], sizeof r) == 0);
    }
    {
        float sorted[80];
        memcpy(sorted, g_score, frames * sizeof(float));
        for (f = 1; f < frames; f++) /* insertion sort */
            for (j = f; j > 0 && sorted[j - 1] > sorted[j]; j--) {
                const float tmp = sorted[j];
                sorted[j] = sorted[j - 1];
                sorted[j - 1] = tmp;
            }
        t = sorted[frames / 2];
    }
    memcpy(blob_buf + n + 28, &t, 4);
    wr32(blob_buf + n + 52, crc32_ref(blob_buf + n + 64, (uint32_t)(gn - n - 64)));
    CHECK(vocr_check_crc(blob_buf, gn) == VOCR_OK);
    /* OFF: the base exactly, arena too */
    {
        vocr_model mb, mg;
        vocr_opts oo;
        size_t base_w8i;
        CHECK(vocr_load(&mb, blob_buf, n) == VOCR_OK && vocr_load(&mg, blob_buf, gn) == VOCR_OK);
        memset(&oo, 0, sizeof oo);
        oo.mode = VOCR_MODE_W8I;
        base_w8i = vocr_arena_bytes(&mb, &oo);
        CHECK(base_w8i > 0 && vocr_arena_bytes(&mg, &oo) == base_w8i);
        oo.head_gate = VOCR_HEAD_GATE_SCORE;
        CHECK(vocr_arena_bytes(&mg, &oo) >= base_w8i);
        CHECK(vocr_arena_bytes(&mb, &oo) == 0); /* no gate in the blob */
        oo.head_gate = 3;
        CHECK(vocr_arena_bytes(&mg, &oo) == 0);
        oo.head_gate = VOCR_HEAD_GATE_ON;
        oo.mode = VOCR_MODE_W8F;
        CHECK(vocr_arena_bytes(&mg, &oo) == 0);
        oo.mode = VOCR_MODE_W8I;
        CHECK(vocr_arena_bytes(&mg, &oo) > base_w8i); /* the frame indices */
    }
    CHECK(gate_run(gn, VOCR_HEAD_GATE_OFF, 0, 0, w, hgt, 3, 1, text, sizeof text, &ex) == VOCR_OK);
    CHECK(strcmp(text, ref) == 0 && ex.gated == 0 && memcmp(g_logits, logits0, sizeof logits0) == 0);
    CHECK(memcmp(g_tv, tv0, sizeof tv0) == 0 && memcmp(g_tc, tc0, sizeof tc0) == 0);
    /* the expected gated output, from the reference run and the reference scores */
    {
        unsigned last = 0, used = 0;
        want[0] = 0;
        for (f = 0; f < frames; f++) {
            const unsigned c = gate_ref(&m, hin0 + f * 16) > t ? 0u : best0[f];
            skipped += gate_ref(&m, hin0 + f * 16) > t;
            if (c && c != last) {
                unsigned len;
                const char *s = vocr_class_text(&m, c, &len);
                memcpy(want + used, s, len);
                used += len;
                want[used] = 0;
            }
            last = c;
        }
    }
    CHECK(skipped > frames / 4 && skipped < frames);
    for (ti = 0; ti < 6; ti++)
        for (ki = 0; ki < 3; ki++) {
            const unsigned k = ks[ki];
            memset(g_tv, 0x7F, sizeof g_tv);
            memset(g_tc, 0x7F, sizeof g_tc);
            memset(g_bl, 0x7F, sizeof g_bl);
            memset(g_best, 0x7F, sizeof g_best);
            CHECK(gate_run(gn, VOCR_HEAD_GATE_ON, tiles[ti][0], tiles[ti][1], w, hgt, k, 0, text, sizeof text, &ex) ==
                  VOCR_OK);
            CHECK(strcmp(text, want) == 0);
            CHECK(ex.gated == skipped && ex.frames == frames);
            for (f = 0; f < frames; f++) {
                const int skip = gate_ref(&m, hin0 + f * 16) > t;
                CHECK(g_best[f] == (skip ? 0 : best0[f]));
                if (!k) continue;
                CHECK(memcmp(&g_bl[f], &bl0[f], sizeof(float)) == 0); /* the head's own class-0 logit */
                for (j = 0; j < k; j++) {
                    if (skip) {
                        CHECK(g_tc[f * k + j] == 0);
                        if (j) CHECK(g_tv[f * k + j] == -INFINITY);
                        else CHECK(memcmp(&g_tv[f * k], &bl0[f], sizeof(float)) == 0);
                    } else { /* the reference's top-k (its top-3's first k) */
                        CHECK(g_tc[f * k + j] == tc0[f * 3 + j]);
                        CHECK(memcmp(&g_tv[f * k + j], &tv0[f * 3 + j], sizeof(float)) == 0);
                    }
                }
            }
        }
    /* the logits asked for: nothing skipped, the base exactly */
    CHECK(gate_run(gn, VOCR_HEAD_GATE_ON, 0, 0, w, hgt, 3, 1, text, sizeof text, &ex) == VOCR_OK);
    CHECK(strcmp(text, ref) == 0 && ex.gated == 0 && memcmp(g_logits, logits0, sizeof logits0) == 0);
    /* thresholds: +inf skips nothing (the base's text), -inf every frame (no text, every best 0) */
    t = INFINITY;
    memcpy(blob_buf + n + 28, &t, 4);
    CHECK(gate_run(gn, VOCR_HEAD_GATE_ON, 0, 0, w, hgt, 3, 0, text, sizeof text, &ex) == VOCR_OK);
    CHECK(strcmp(text, ref) == 0 && ex.gated == 0 && memcmp(g_tv, tv0, (size_t)frames * 3 * sizeof(float)) == 0);
    t = -INFINITY;
    memcpy(blob_buf + n + 28, &t, 4);
    CHECK(gate_run(gn, VOCR_HEAD_GATE_ON, 2, 3, w, hgt, 3, 0, text, sizeof text, &ex) == VOCR_OK);
    CHECK(text[0] == 0 && ex.gated == frames && ex.chars == 0 && ex.conf_min == 1.0f);
    for (f = 0; f < frames; f++) CHECK(g_best[f] == 0 && g_tc[f * 3] == 0 && g_tv[f * 3 + 2] == -INFINITY);
    /* a gated run with no extras at all, and one cancelled */
    {
        vocr_model mg;
        vocr_opts oo;
        vocr_ctx c;
        vocr_image im;
        int calls = 0;
        CHECK(vocr_load(&mg, blob_buf, gn) == VOCR_OK);
        memset(&oo, 0, sizeof oo);
        oo.mode = VOCR_MODE_W8I;
        oo.head_gate = VOCR_HEAD_GATE_ON;
        CHECK(vocr_init(&c, &mg, &oo, arena_a, vocr_arena_bytes(&mg, &oo)) == VOCR_OK);
        im.pixels = img_px;
        im.width = w;
        im.height = hgt;
        im.stride = w;
        im.format = VOCR_GREY8;
        CHECK(vocr_run(&c, &im, text, sizeof text, NULL) == VOCR_OK && text[0] == 0);
        /* ungated: this line as head_gate OFF reads it, with the gate on (t = -inf would skip every frame) */
        memset(&ex, 0, sizeof ex);
        ex.ungated = 1;
        ex.cap_frames = 80;
        ex.best = g_best;
        ex.topk = 3;
        ex.topk_logit = g_tv;
        ex.topk_class = g_tc;
        ex.blank_logit = g_bl;
        CHECK(vocr_run(&c, &im, text, sizeof text, &ex) == VOCR_OK && strcmp(text, ref) == 0 && ex.gated == 0);
        CHECK(memcmp(g_tv, tv0, (size_t)frames * 3 * sizeof(float)) == 0);
        CHECK(memcmp(g_tc, tc0, (size_t)frames * 3 * sizeof(uint16_t)) == 0);
        CHECK(memcmp(g_best, best0, (size_t)frames * sizeof(uint16_t)) == 0);
        memset(&ex, 0, sizeof ex);
        ex.cancel = stop_now;
        ex.user = &calls;
        CHECK(vocr_run(&c, &im, text, sizeof text, &ex) == VOCR_E_CANCELLED && text[0] == 0);
    }
}

/* the full model shape; and the head right after the stem (audit-head-gate F1: its frames come from the front's
 * last op, so they must be counted and gated there too). The stem-head blob's known answer is main 6f24967's
 * (host gcc and clang give the same); before the fix this branch read that blob as VOCR_E_FORMAT. */
#define STEM_HEAD_W8I_HASH 0x3be72ad7e834d8caull
static void test_gate_run(void)
{
    test_gate_run_on(build_blob, 0);
    test_gate_run_on(build_stem_head_blob, STEM_HEAD_W8I_HASH);
}

/* ------------------------------------------------------------ the two-level head (docs/head2level.md) */

enum { TCLASSES = 21 }; /* three head blocks: level 1 = the first (the blank and 7 classes), or the first two */

/* build_blob(1)'s model with a 21-class random int8 head */
static size_t build_blob_2l(void)
{
    unsigned char vocab[2 * (TCLASSES + 1) + TCLASSES];
    unsigned i, pos = 0, n_ops = 7;
    uint32_t off;
    memset(blob_buf, 0, sizeof blob_buf);
    blob_pos = 64 + 64 * n_ops;
    lcg = 4321u;
    off = put_floats(9 * 8, 0.6f);
    op_rec(0, 1, 1 | 4, 3, 3, 2, 2, 1, 1, 1, 8, 8, 4, off, 9 * 8 * 4, put_floats(8, 0.1f), 0, 9, 8);
    off = put_floats(9 * 8, 0.6f);
    op_rec(1, 2, 1 | 4, 3, 3, 2, 2, 1, 1, 8, 8, 4, 2, off, 9 * 8 * 4, put_floats(8, 0.1f), 0, 9, 8);
    mat_op(2, 3, 1 | 4, 1, 8, 16, 2, 2, 1);
    mat_op(3, 3, 1 | 4 | 16, 2, 16, 16, 2, 1, 1);
    off = put_floats(5 * 16, 0.6f);
    op_rec(4, 2, 1 | 4, 1, 5, 1, 1, 0, 2, 16, 16, 1, 1, off, 5 * 16 * 4, put_floats(16, 0.1f), 0, 5, 16);
    mat_op(5, 3, 1 | 2 | 4, 1, 16, 16, 1, 1, 1);
    mat_op(6, 4, 4, 1, 16, TCLASSES, 1, 1, 1);
    wr16(vocab, 0);
    wr16(vocab + 2, 0);
    for (i = 1; i < TCLASSES; i++) {
        vocab[2 * (TCLASSES + 1) + pos++] = (unsigned char)('A' + i - 1);
        wr16(vocab + 2 * (i + 1), pos);
    }
    off = put(vocab, 2 * (TCLASSES + 1) + pos);
    memcpy(blob_buf, "VOCR", 4);
    wr16(blob_buf + 4, 1);
    wr16(blob_buf + 6, 64);
    wr32(blob_buf + 8, (uint32_t)blob_pos);
    wr32(blob_buf + 12, 1u);
    wr16(blob_buf + 16, 8);
    wr16(blob_buf + 18, 256);
    wr16(blob_buf + 20, 4);
    wr16(blob_buf + 22, n_ops);
    wr32(blob_buf + 24, TCLASSES);
    wr32(blob_buf + 28, 64);
    wr32(blob_buf + 32, off);
    wr32(blob_buf + 36, pos);
    wr32(blob_buf + 40, crc32_ref(blob_buf + 64, blob_pos - 64));
    return blob_pos;
}

/* Appends a VOC2 trailer at blob_buf + al16(n) for the base blob of `base` bytes: l1 level-1 blocks, an 8-unit
 * escape with weights from the seed and threshold t, and rk = 0 (the head's own rare rows) or 8 (a random 16 -> 8
 * projection and rare rows over it). Returns the blob's new bytes. */
static size_t append_two(size_t base, size_t n, unsigned l1, float t, unsigned rk, uint32_t seed)
{
    const unsigned K = 16, H = 8, R = ((TCLASSES + 7u) / 8u - l1) * 8u;
    const size_t at = (n + 15u) & ~(size_t)15u;
    unsigned char *p = blob_buf + at;
    uint32_t o[12], pos = 128, i, j;
    const uint32_t sz[12] = {H * K, 4 * H, 4 * H, 4 * H, 4 * H, 4 * H, rk * K, 4 * rk, 4 * rk, R * rk, 4 * R, 4 * R};
    float b2 = 0.05f;
    room(at - n + 4096);
    memset(blob_buf + n, 0, at - n);
    lcg = seed;
    for (i = 0; i < 12; i++) {
        o[i] = (i >= 6 && !rk) ? 0u : pos;
        if (o[i]) pos = (pos + sz[i] + 15u) & ~15u;
    }
    memset(p, 0, pos);
    for (j = 0; j < H * K; j++) p[o[0] + j] = (unsigned char)(signed char)(int)(frand() * 127.0f);
    for (j = 0; j < H; j++) {
        const float sc = 0.002f + 0.0005f * (float)(j % 5), b = frand() * 0.3f, wl = frand() * 0.05f,
                    wb = frand() * 0.05f, w2 = frand();
        memcpy(p + o[1] + 4 * j, &sc, 4);
        memcpy(p + o[2] + 4 * j, &b, 4);
        memcpy(p + o[3] + 4 * j, &wl, 4);
        memcpy(p + o[4] + 4 * j, &wb, 4);
        memcpy(p + o[5] + 4 * j, &w2, 4);
    }
    if (rk) {
        for (j = 0; j < rk * K; j++) p[o[6] + j] = (unsigned char)(signed char)(int)(frand() * 127.0f);
        for (j = 0; j < rk; j++) {
            const float sc = 0.01f, b = frand() * 0.1f;
            memcpy(p + o[7] + 4 * j, &sc, 4);
            memcpy(p + o[8] + 4 * j, &b, 4);
        }
        for (j = 0; j < R * rk; j++) p[o[9] + j] = (unsigned char)(signed char)(int)(frand() * 127.0f);
        for (j = 0; j < R; j++) {
            const float sc = 0.02f, b = frand() * 0.5f;
            memcpy(p + o[10] + 4 * j, &sc, 4);
            memcpy(p + o[11] + 4 * j, &b, 4);
        }
    }
    memcpy(p, "VOC2", 4);
    wr16(p + 4, 1);
    wr16(p + 6, 128);
    wr32(p + 8, pos);
    wr32(p + 16, (uint32_t)base);
    memcpy(p + 20, blob_buf + 40, 4);
    wr16(p + 24, l1);
    wr16(p + 26, H);
    wr16(p + 28, rk);
    memcpy(p + 32, &t, 4);
    memcpy(p + 36, &b2, 4);
    for (i = 0; i < 12; i++) wr32(p + 40 + 4 * i, o[i]);
    wr32(p + 12, crc32_ref(p + 16, pos - 16));
    blob_pos = at + pos;
    return at + pos;
}

/* The escape's score of one frame, written independently of vocr.c: quant16's int16 grid, exact integer sums,
 * then the float steps in order; l1 / b0 = level 1's largest non-blank logit and its blank logit. */
static float esc_ref(const vocr_2level *tw, const float *x, float l1, float b0)
{
    const unsigned K = 16;
    float mx = 0.0f, inv, sx, s = tw->e_b2;
    int qx[16];
    unsigned j, k;
    for (k = 0; k < K; k++)
        if (fabsf(x[k]) > mx) mx = fabsf(x[k]);
    inv = mx > 0.0f ? 32767.0f / mx : 0.0f;
    sx = mx > 0.0f ? mx / 32767.0f : 0.0f;
    for (k = 0; k < K; k++) {
        const float t = x[k] * inv;
        int v = (int)(t + (signbit(t) ? -0.5f : 0.5f));
        qx[k] = v > 32767 ? 32767 : v < -32767 ? -32767 : v;
    }
    for (j = 0; j < tw->hidden; j++) {
        const int8_t *w = tw->e_q + (size_t)(j / 8u) * K * 8u + j % 8u;
        int32_t acc = 0;
        float a;
        for (k = 0; k < K; k++) acc += (int32_t)w[(size_t)k * 8u] * qx[k];
        a = (float)acc * (tw->e_scale[j] * sx) + tw->e_bias[j];
        a = a + tw->e_wl1[j] * l1;
        a = a + tw->e_wb0[j] * b0;
        if (!(a > 0.0f)) a = 0.0f;
        s = s + tw->e_w2[j] * a;
    }
    return s;
}

static float t_logits[80 * TCLASSES], t_hin[80 * 16], t_esc[80], t_tv[80 * TCLASSES], t_bl[80], t_gs[80];
static uint16_t t_tc[80 * TCLASSES], t_best[80];

/* One W8I run of the blob in blob_buf (n bytes), two_level tl, head_gate hg, these tiles, top-k (k >= 1). */
static int two_run(size_t n, unsigned tl, unsigned hg, unsigned tile, unsigned head, unsigned w, unsigned h,
                   unsigned k, int logits, char *text, size_t cap, vocr_extra *out)
{
    vocr_model m;
    vocr_opts o;
    vocr_ctx c;
    vocr_image im;
    vocr_extra ex;
    size_t need;
    unsigned f;
    int rc = vocr_load(&m, blob_buf, n);
    if (rc) return rc;
    memset(&o, 0, sizeof o);
    o.mode = VOCR_MODE_W8I;
    o.tile_frames = tile;
    o.head_frames = head;
    o.head_gate = hg;
    o.two_level = tl;
    need = vocr_arena_bytes(&m, &o);
    if (!need) return VOCR_E_MODE;
    rc = vocr_init(&c, &m, &o, arena_a, need);
    if (rc) return rc;
    im.pixels = img_px;
    im.width = w;
    im.height = h;
    im.stride = w;
    im.format = VOCR_GREY8;
    memset(&ex, 0, sizeof ex);
    for (f = 0; f < 80; f++) t_esc[f] = NAN;
    ex.cap_frames = 80;
    ex.best = t_best;
    ex.logits = logits ? t_logits : NULL;
    ex.head_in = t_hin;
    ex.esc_score = t_esc;
    ex.gate_score = hg ? t_gs : NULL;
    ex.topk = k;
    ex.topk_logit = t_tv;
    ex.topk_class = t_tc;
    ex.blank_logit = t_bl;
    rc = vocr_run(&c, &im, text, cap, &ex);
    if (out) *out = ex;
    return rc;
}

static void test_two_load(void)
{
    vocr_model m;
    vocr_opts o;
    const size_t n = build_blob_2l(), tn = append_two(n, n, 1, 0.0f, 0, 7u);
    const size_t at = (n + 15u) & ~(size_t)15u;
    CHECK(vocr_check_crc(blob_buf, tn) == VOCR_OK && vocr_load(&m, blob_buf, tn) == VOCR_OK);
    CHECK(m.bytes == n && m.two.l1_blocks == 1 && m.two.hidden == 8 && m.two.rare_k == 0 && m.two.r_blocks == 2);
    CHECK(vocr_has_two_level(&m) == 1 && vocr_has_two_level(NULL) == 0);
    CHECK(vocr_load(&m, blob_buf, n) == VOCR_OK && m.two.l1_blocks == 0 && !vocr_has_two_level(&m)); /* none */
    memset(&o, 0, sizeof o);
    o.mode = VOCR_MODE_W8I;
    o.two_level = 1;
    CHECK(vocr_arena_bytes(&m, &o) == 0);                                  /* two_level needs a trailer */
    CHECK(vocr_load(&m, blob_buf, tn) == VOCR_OK);
    o.mode = VOCR_MODE_W8F;
    CHECK(vocr_arena_bytes(&m, &o) == 0);                                  /* and W8I */
    blob_buf[tn - 1] ^= 1u;
    CHECK(vocr_check_crc(blob_buf, tn) == VOCR_E_FORMAT && vocr_load(&m, blob_buf, tn) == VOCR_OK);
    blob_buf[tn - 1] ^= 1u;
    {   /* fields the loader refuses (each with a fresh CRC) */
        const unsigned offs[6] = {24, 24, 26, 28, 16, 88};
        const unsigned vals[6] = {0, 3, 12, 4, 7, 1};
        unsigned i;
        for (i = 0; i < 6; i++) {
            unsigned char save[4];
            memcpy(save, blob_buf + at + offs[i], 4);
            if (offs[i] == 16) wr32(blob_buf + at + offs[i], vals[i]);
            else if (offs[i] == 88) blob_buf[at + 88] = (unsigned char)vals[i];
            else wr16(blob_buf + at + offs[i], vals[i]);
            wr32(blob_buf + at + 12, crc32_ref(blob_buf + at + 16, (uint32_t)(tn - at - 16)));
            CHECK(vocr_load(&m, blob_buf, tn) == VOCR_E_FORMAT);
            memcpy(blob_buf + at + offs[i], save, 4);
        }
        wr32(blob_buf + at + 12, crc32_ref(blob_buf + at + 16, (uint32_t)(tn - at - 16)));
        CHECK(vocr_load(&m, blob_buf, tn) == VOCR_OK);
    }
}

/* run_head2 against a reference built from the full head's logits and head inputs: each frame's escape score,
 * then its argmax and top-K over every class (fired) or level 1's (not); at several tilings and thresholds. */
static void test_two_run(void)
{
    static float full[80 * TCLASSES], hin[80 * 16];
    char ref[256], text[256];
    vocr_extra e0, ex;
    vocr_model m;
    const unsigned w = 260, hgt = 21, k = 4;
    const unsigned tiles[3][2] = {{0, 0}, {1, 1}, {5, 7}};
    const float ts[4] = {-INFINITY, -0.2f, 0.3f, INFINITY};
    size_t n, tn;
    unsigned ti, i, f, c, fired_total = 0, frames_total = 0;
    make_image(w, hgt, 0);
    n = build_blob_2l();
    CHECK(two_run(n, 0, 0, 0, 0, w, hgt, k, 1, ref, sizeof ref, &e0) == VOCR_OK);
    memcpy(full, t_logits, sizeof full);
    memcpy(hin, t_hin, sizeof hin);
    for (i = 0; i < 4; i++) {
        for (ti = 0; ti < 3; ti++) {
            unsigned l1b = 1 + (i & 1u), L1 = 8 * l1b, fired = 0;
            tn = append_two(n, n, l1b, ts[i], 0, 99u);
            CHECK(vocr_load(&m, blob_buf, tn) == VOCR_OK);
            CHECK(two_run(tn, 1, 0, tiles[ti][0], tiles[ti][1], w, hgt, k, 0, text, sizeof text, &ex) == VOCR_OK);
            CHECK(ex.frames == e0.frames);
            for (f = 0; f < ex.frames; f++) {
                const float *lg = full + (size_t)f * TCLASSES;
                float l1 = -INFINITY, s, tv[TCLASSES];
                uint16_t tc[TCLASSES];
                unsigned nc, j, best = 0;
                for (c = 1; c < L1; c++)
                    if (lg[c] > l1) l1 = lg[c];
                s = esc_ref(&m.two, hin + (size_t)f * 16, l1, lg[0]);
                CHECK(same_float(s, t_esc[f]));
                nc = s > m.two.t ? TCLASSES : L1;
                fired += nc == TCLASSES;
                for (c = 1; c < nc; c++)
                    if (lg[c] > lg[best]) best = c;
                CHECK(t_best[f] == best);
                topk_ref(lg, nc, k, tv, tc);
                for (j = 0; j < k; j++) CHECK(same_float(tv[j], t_tv[f * k + j]) && tc[j] == t_tc[f * k + j]);
                CHECK(same_float(t_bl[f], lg[0]));
            }
            CHECK(ex.escaped == fired);
            if (ts[i] == -INFINITY) CHECK(!strcmp(text, ref) && ex.conf_min == e0.conf_min);
            if (ts[i] == INFINITY) CHECK(fired == 0);
            fired_total += fired;
            frames_total += ex.frames;
        }
    }
    CHECK(fired_total > 0 && fired_total < frames_total);
    /* the logits asked for: the full head on every frame; ungated: the full head too */
    tn = append_two(n, n, 1, 0.3f, 0, 99u);
    CHECK(two_run(tn, 1, 0, 0, 0, w, hgt, k, 1, text, sizeof text, &ex) == VOCR_OK);
    CHECK(!strcmp(text, ref) && ex.escaped == 0 && !memcmp(t_logits, full, (size_t)ex.frames * TCLASSES * 4));
    /* r48: with nothing fired the rare variant changes nothing (level 1 and the escape are the same) */
    {
        static float tv0[80 * TCLASSES], es0[80];
        static uint16_t tc0[80 * TCLASSES];
        size_t rn = append_two(n, n, 1, INFINITY, 0, 5u);
        CHECK(two_run(rn, 1, 0, 0, 0, w, hgt, k, 0, ref, sizeof ref, &e0) == VOCR_OK);
        memcpy(tv0, t_tv, sizeof tv0);
        memcpy(tc0, t_tc, sizeof tc0);
        memcpy(es0, t_esc, sizeof es0);
        rn = append_two(n, n, 1, INFINITY, 8, 5u);
        CHECK(vocr_load(&m, blob_buf, rn) == VOCR_OK && m.two.rare_k == 8);
        CHECK(two_run(rn, 1, 0, 0, 0, w, hgt, k, 0, text, sizeof text, &ex) == VOCR_OK);
        CHECK(!strcmp(text, ref) && !memcmp(tv0, t_tv, (size_t)ex.frames * k * 4) &&
              !memcmp(tc0, t_tc, (size_t)ex.frames * k * 2) && !memcmp(es0, t_esc, (size_t)ex.frames * 4));
        for (ti = 0; ti < 3; ti++) { /* and fired: it runs (sanitizers), the same at every tiling */
            rn = append_two(n, n, 2, -0.2f, 8, 5u);
            CHECK(two_run(rn, 1, 0, tiles[ti][0], tiles[ti][1], w, hgt, k, 0, text, sizeof text, &ex) == VOCR_OK);
            if (!ti) {
                memcpy(tv0, t_tv, sizeof tv0);
                strcpy(ref, text);
            } else {
                CHECK(!strcmp(text, ref) && !memcmp(tv0, t_tv, (size_t)ex.frames * k * 4));
            }
            CHECK(ex.escaped > 0);
        }
    }
}

/* head_gate ON with two_level: a frame the gate skips reads as the blank, the others as two_level alone */
static void test_two_gate(void)
{
    static float tv2[80 * TCLASSES], es2[80], bl2[80];
    static uint16_t tc2[80 * TCLASSES], b2[80];
    char t2[256], text[256];
    vocr_extra e2, ex;
    vocr_model m;
    const unsigned w = 260, hgt = 21, k = 4;
    size_t n = build_blob_2l(), gn, tn;
    unsigned f, j, skipped = 0;
    make_image(w, hgt, 0);
    gn = append_gate(n, 8, 0.0f, 0.1f);
    tn = append_two(n, gn, 1, 0.0f, 0, 77u);
    CHECK(vocr_load(&m, blob_buf, tn) == VOCR_OK && vocr_has_gate(&m) && m.two.l1_blocks == 1);
    CHECK(vocr_check_crc(blob_buf, tn) == VOCR_OK);
    CHECK(two_run(tn, 1, 0, 0, 0, w, hgt, k, 0, t2, sizeof t2, &e2) == VOCR_OK);
    memcpy(tv2, t_tv, sizeof tv2);
    memcpy(tc2, t_tc, sizeof tc2);
    memcpy(es2, t_esc, sizeof es2);
    memcpy(bl2, t_bl, sizeof bl2);
    memcpy(b2, t_best, sizeof b2);
    CHECK(two_run(tn, 1, VOCR_HEAD_GATE_ON, 1, 3, w, hgt, k, 0, text, sizeof text, &ex) == VOCR_OK);
    for (f = 0; f < ex.frames; f++) {
        const int skip = gate_ref(&m, t_hin + (size_t)f * 16) > m.gate.t;
        skipped += (unsigned)skip;
        if (skip) {
            CHECK(t_best[f] == 0 && t_tc[f * k] == 0 && same_float(t_tv[f * k], bl2[f]) && isnan(t_esc[f]));
            for (j = 1; j < k; j++) CHECK(t_tv[f * k + j] == -INFINITY);
        } else {
            CHECK(t_best[f] == b2[f] && same_float(t_esc[f], es2[f]));
            for (j = 0; j < k; j++) CHECK(same_float(t_tv[f * k + j], tv2[f * k + j]) && t_tc[f * k + j] == tc2[f * k + j]);
        }
    }
    CHECK(ex.gated == skipped && skipped > 0 && skipped < ex.frames);
}

int main(void)
{
    test_load(0);
    test_load(1);
    /* known answers: main a540027's scalar code (host gcc and clang, Cortex-A9 NEON and scalar agree) */
    test_run(0, 16, 0x119943e1dc1cc516ull);
    test_run(1, 16, 0x807bd1f142cf0a33ull);
    test_run(0, 272, 0x6b8b79592b793218ull);
    test_run(1, 272, 0x0e9062f159f32b9eull);
    test_run_sc(0, 32, 16, 0x59e8de63c50ea85full);
    test_run_sc(1, 32, 16, 0xc2246e1d797ae5c0ull);
    test_run_sc(0, 32, 272, 0xb905ef3abe85868aull);
    test_run_sc(1, 32, 272, 0x30951002cbd4e0e1ull);
    test_topk(0);
    test_topk(1);
    test_invert();
    test_limits();
    test_head_argmax();
    test_gate_load();
    test_gate_run();
    test_two_load();
    test_two_run();
    test_two_gate();
    if (failures) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    printf("test_vocr: all checks passed\n");
    return 0;
}
