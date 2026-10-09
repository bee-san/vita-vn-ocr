/* test_vocr.c - unit tests of the vocr runtime that need no model file.
 *
 * Builds a tiny valid blob in memory (stem, depthwise, pointwise, 2x1 collapse,
 * 1x5 temporal depthwise with a residual pointwise, head), then checks loading and
 * validation, arena sizing, tile invariance, text overflow, cancellation and error
 * paths. Run under ASan/UBSan by `make check`.
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

static unsigned char blob_buf[1 << 16] __attribute__((aligned(16)));
static size_t blob_pos;
static uint32_t lcg = 12345u;

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
    memcpy(blob_buf + blob_pos, data, n);
    blob_pos = (blob_pos + n + 15u) & ~(size_t)15u;
    return off;
}

static uint32_t put_floats(unsigned n, float scale)
{
    uint32_t off = (uint32_t)blob_pos;
    unsigned i;
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
        for (j = 0; j < n; j++) blob_buf[blob_pos + j] = (unsigned char)(signed char)(int)(frand() * 127.0f);
        blob_pos = (blob_pos + n + 15u) & ~(size_t)15u;
        s_off = (uint32_t)blob_pos;
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
static size_t build_blob_c1(int int8, unsigned c1);
static size_t build_blob(int int8) { return build_blob_c1(int8, 16); }

static size_t build_blob_c1(int int8, unsigned c1)
{
    unsigned char vocab[64];
    const char *chars[CLASSES - 1] = {"a", "\xe3\x81\x82", "b", "\xe3\x83\xbc", "!"};
    unsigned i, pos = 0, n_ops = 7;
    uint32_t off;
    memset(blob_buf, 0, sizeof blob_buf);
    blob_pos = 64 + 64 * n_ops;
    lcg = 12345u;
    off = put_floats(9 * 8, 0.6f);
    op_rec(0, 1, 1 | 4, 3, 3, 2, 2, 1, 1, 1, 8, 8, 4, off, 9 * 8 * 4, put_floats(8, 0.1f), 0, 9, 8);
    off = put_floats(9 * 8, 0.6f);
    op_rec(1, 2, 1 | 4, 3, 3, 2, 2, 1, 1, 8, 8, 4, 2, off, 9 * 8 * 4, put_floats(8, 0.1f), 0, 9, 8);
    mat_op(2, 3, 1 | 4, 1, 8, c1, 2, 2, int8);
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
    im.stride = (size_t)w * (fmt == VOCR_RGBA8 ? 4u : 1u);
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

static void test_run(int int8, unsigned c1)
{
    vocr_model m;
    vocr_opts o;
    char ref[256], text[256];
    float lref[80 * CLASSES], lcur[80 * CLASSES], lw8f[80 * CLASSES];
    vocr_extra ex;
    size_t n = build_blob_c1(int8, c1);
    const int modes_i8[2] = {VOCR_MODE_W8F, VOCR_MODE_W8I};
    int mi, nmodes = int8 ? 2 : 1, fmt;
    unsigned tiles[5][2] = {{1, 1}, {2, 3}, {3, 1}, {5, 7}, {64, 64}};
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

int main(void)
{
    test_load(0);
    test_load(1);
    test_run(0, 16);
    test_run(1, 16);
    test_run(0, 272);
    test_run(1, 272);
    test_limits();
    if (failures) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    printf("test_vocr: all checks passed\n");
    return 0;
}
