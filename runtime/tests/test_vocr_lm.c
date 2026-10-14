/* test_vocr_lm.c - unit tests of vocr_lm.c that need no model or LM file.
 *
 * Builds a tiny VOLM blob in memory (order 3, four characters) with hand-checkable scores, a fake
 * vocr_model holding only a vocabulary, and synthetic per-frame top-K logits. Checks loading and
 * validation, the backoff scorer against hand-computed values, that alpha = beta = 0 reproduces the
 * greedy text for every beam / top-K / gate on random lines, that the LM changes an uncertain frame and
 * the gate keeps a confident one, the workspace size, cancellation, text overflow and error paths.
 * Run under ASan/UBSan by `make check`. The bit-exact comparison with lm/vlm.py on real development
 * logits is tests/test_ctclm.py's.
 */
#include "../vocr_lm.h"

#include <stdio.h>
#include <string.h>

static int failures;
#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);       \
            failures++;                                                                    \
        }                                                                                  \
    } while (0)

/* ------------------------------------------------------------ a tiny VOLM blob */

static unsigned char lmb[4096] __attribute__((aligned(16)));
static size_t lpos;

static void wr32(unsigned char *p, uint32_t v)
{
    p[0] = (unsigned char)v;
    p[1] = (unsigned char)(v >> 8);
    p[2] = (unsigned char)(v >> 16);
    p[3] = (unsigned char)(v >> 24);
}

static uint32_t lput(const void *d, size_t n)
{
    uint32_t off = (uint32_t)lpos;
    memcpy(lmb + lpos, d, n);
    lpos = (lpos + n + 15u) & ~(size_t)15u;
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

static void row(unsigned k, uint32_t n, uint32_t ot, uint32_t op, uint32_t ob, uint32_t oc, uint32_t ocp, uint32_t ocb)
{
    unsigned char *r = lmb + 64 + 32 * (k - 1);
    wr32(r, n);
    wr32(r + 4, ot);
    wr32(r + 8, op);
    wr32(r + 12, ob);
    wr32(r + 16, oc);
    wr32(r + 20, ocp);
    wr32(r + 24, ocb);
}

static void fix_crc(size_t n) { wr32(lmb + 24, crc32_ref(lmb + 64, n - 64)); }

/* tokens: 0 <unk>, 1 <s>, 2 </s>, 3 a, 4 b, 5 c, 6 d.  bigrams (<s> a) (a b) (a c) (b c); trigram (a b c).
 * codebooks: order-1 prob code i = -64 i, order-2 prob -50 i, order-3 prob -10 i (Q10);
 * order-1 backoff {0, -100, -200, +50}, order-2 backoff {0, -300} */
static size_t build_lm(void)
{
    static const uint32_t cps[4] = {0x61, 0x62, 0x63, 0x64};
    static const uint8_t p1[7] = {10, 20, 4, 2, 3, 3, 5}, b1[7] = {0, 3, 0, 1, 2, 0, 0};
    static const uint32_t c1[8] = {0, 0, 1, 1, 3, 4, 4, 4};
    static const uint16_t t2[4] = {3, 4, 5, 5};
    static const uint8_t p2[4] = {1, 2, 3, 1}, b2[4] = {0, 1, 0, 0};
    static const uint32_t c2[5] = {0, 0, 1, 1, 1};
    static const uint16_t t3[1] = {5};
    static const uint8_t p3[1] = {7};
    int16_t cb1[256], cbb1[256], cb2[256], cbb2[256], cb3[256];
    uint32_t o[16];
    unsigned i;
    memset(lmb, 0, sizeof lmb);
    for (i = 0; i < 256; i++) {
        cb1[i] = (int16_t)(-64 * (int)i);
        cb2[i] = (int16_t)(-50 * (int)i);
        cb3[i] = (int16_t)(-10 * (int)i);
        cbb1[i] = cbb2[i] = 0;
    }
    cbb1[1] = -100;
    cbb1[2] = -200;
    cbb1[3] = 50;
    cbb2[1] = -300;
    lpos = 64 + 6 * 32;
    o[0] = lput(p1, 7);
    o[1] = lput(b1, 7);
    o[2] = lput(c1, sizeof c1);
    o[3] = lput(cb1, sizeof cb1);
    o[4] = lput(cbb1, sizeof cbb1);
    o[5] = lput(t2, sizeof t2);
    o[6] = lput(p2, 4);
    o[7] = lput(b2, 4);
    o[8] = lput(c2, sizeof c2);
    o[9] = lput(cb2, sizeof cb2);
    o[10] = lput(cbb2, sizeof cbb2);
    o[11] = lput(t3, sizeof t3);
    o[12] = lput(p3, 1);
    o[13] = lput(cb3, sizeof cb3);
    o[14] = lput(cps, sizeof cps);
    row(1, 7, 0, o[0], o[1], o[2], o[3], o[4]);
    row(2, 4, o[5], o[6], o[7], o[8], o[9], o[10]);
    row(3, 1, o[11], o[12], 0, 0, o[13], 0);
    memcpy(lmb, "VOLM", 4);
    wr32(lmb + 4, 1);
    wr32(lmb + 8, (uint32_t)lpos);
    wr32(lmb + 12, 3);
    wr32(lmb + 16, 7);
    wr32(lmb + 28, o[14]);
    fix_crc(lpos);
    return lpos;
}

/* ------------------------------------------------------------ a fake model: a vocabulary only */

/* classes: 0 blank, 1-4 a b c d, 5 あ, 6 「, 7 」, 8 『, 9 』 */
static const char vocab_txt[] = "abcd\xe3\x81\x82\xe3\x80\x8c\xe3\x80\x8d\xe3\x80\x8e\xe3\x80\x8f";
static const uint16_t vocab_off[11] = {0, 0, 1, 2, 3, 4, 7, 10, 13, 16, 19};

static void fake_model(vocr_model *m)
{
    memset(m, 0, sizeof *m);
    m->blob = (const unsigned char *)vocab_txt;
    m->n_classes = 10;
    m->vocab_off = vocab_off;
    m->vocab = vocab_txt;
    m->max_width = 64;
    m->stride = 8;
    m->height = 32;
}

static unsigned char ws[1 << 18] __attribute__((aligned(16)));

static void test_load(void)
{
    vocr_lm lm;
    const size_t n = build_lm();
    CHECK(vocr_lm_check_crc(lmb, n) == VOCR_OK);
    CHECK(vocr_lm_load(&lm, lmb, n) == VOCR_OK);
    CHECK(lm.order == 3 && lm.n_tokens == 7 && lm.o[0].n == 7 && lm.o[1].n == 4 && lm.o[2].n == 1);
    CHECK(vocr_lm_token(&lm, 'a') == 3 && vocr_lm_token(&lm, 'd') == 6 && vocr_lm_token(&lm, 'e') == 0);
    CHECK(vocr_lm_token(&lm, 0x3042) == 0 && vocr_lm_token(&lm, 0) == 0);
    CHECK(vocr_lm_load(&lm, lmb, n - 1) == VOCR_E_FORMAT);
    CHECK(vocr_lm_load(&lm, lmb, 100) == VOCR_E_FORMAT);
    CHECK(vocr_lm_load(NULL, lmb, n) == VOCR_E_ARG);
    CHECK(vocr_lm_load(&lm, lmb + 4, n - 4) == VOCR_E_ALIGN);
    lmb[0] = 'X';
    CHECK(vocr_lm_load(&lm, lmb, n) == VOCR_E_FORMAT);
    lmb[0] = 'V';
    lmb[4] = 2; /* version */
    CHECK(vocr_lm_load(&lm, lmb, n) == VOCR_E_FORMAT);
    lmb[4] = 1;
    CHECK(vocr_lm_load(&lm, lmb, n) == VOCR_OK);
    lmb[100] ^= 1; /* anywhere after the header: the CRC sees it */
    CHECK(vocr_lm_check_crc(lmb, n) == VOCR_E_FORMAT);
    lmb[100] ^= 1;
    CHECK(vocr_lm_check_crc(lmb, n) == VOCR_OK);
    /* a child range past the next order, a decreasing range, a token >= V, a nonzero backoff code 0 */
    {
        const uint32_t oc = (uint32_t)lmb[64 + 16] | (uint32_t)lmb[64 + 17] << 8;
        const uint32_t ot = (uint32_t)lmb[96 + 4] | (uint32_t)lmb[96 + 5] << 8;
        const uint32_t ocb = (uint32_t)lmb[64 + 24] | (uint32_t)lmb[64 + 25] << 8;
        wr32(lmb + oc + 4 * 7, 5);
        CHECK(vocr_lm_load(&lm, lmb, n) == VOCR_E_FORMAT);
        wr32(lmb + oc + 4 * 7, 4);
        wr32(lmb + oc + 4 * 4, 0);
        CHECK(vocr_lm_load(&lm, lmb, n) == VOCR_E_FORMAT);
        wr32(lmb + oc + 4 * 4, 3);
        CHECK(vocr_lm_load(&lm, lmb, n) == VOCR_OK);
        lmb[ot] = 7;
        CHECK(vocr_lm_load(&lm, lmb, n) == VOCR_E_FORMAT);
        lmb[ot] = 3;
        lmb[ocb] = 1;
        CHECK(vocr_lm_load(&lm, lmb, n) == VOCR_E_FORMAT);
        lmb[ocb] = 0;
        CHECK(vocr_lm_load(&lm, lmb, n) == VOCR_OK);
    }
}

static void test_score(void)
{
    vocr_lm lm;
    int32_t st[2], nx[2], s2[2];
    CHECK(vocr_lm_load(&lm, lmb, build_lm()) == VOCR_OK);
    vocr_lm_initial(&lm, 0, st);
    CHECK(st[0] == -1 && st[1] == -1);
    CHECK(vocr_lm_score(&lm, st, 3, nx) == -128); /* P(a): unigram code 2 */
    CHECK(nx[0] == 3 && nx[1] == -1);
    CHECK(vocr_lm_score(&lm, nx, 4, s2) == -100); /* P(b | a): bigram (a b), code 2 */
    CHECK(s2[0] == 4 && s2[1] == 1);
    CHECK(vocr_lm_score(&lm, s2, 5, nx) == -70);  /* P(c | a b): the trigram */
    CHECK(nx[0] == 5 && nx[1] == 3);              /* (b c) is bigram 3 */
    CHECK(vocr_lm_score(&lm, s2, 6, nx) == -300 - 200 - 320); /* P(d | a b): bo(a b) + bo(b) + P(d) */
    CHECK(nx[0] == 6 && nx[1] == -1);
    st[0] = 3;
    st[1] = -1;
    CHECK(vocr_lm_score(&lm, st, 5, NULL) == -150);       /* P(c | a): bigram (a c), code 3 */
    CHECK(vocr_lm_score(&lm, st, 6, NULL) == -100 - 320); /* P(d | a): bo(a) + P(d) */
    vocr_lm_initial(&lm, VOCR_LM_BOS, st);
    CHECK(st[0] == 1 && st[1] == -1);
    CHECK(vocr_lm_score(&lm, st, 3, NULL) == -50);        /* P(a | <s>) */
    CHECK(vocr_lm_score(&lm, st, 4, NULL) == 50 - 192);   /* P(b | <s>) = bo(<s>) + P(b) */
    CHECK(vocr_lm_score(&lm, st, 2, NULL) == 50 - 256);   /* P(</s> | <s>) */
}

/* ------------------------------------------------------------ the decoder */

static uint32_t lcg = 2026u;
static unsigned urand(unsigned n)
{
    lcg = lcg * 1664525u + 1013904223u;
    return (lcg >> 8) % n;
}

#define FR 8
#define TK 4
static float tv[FR * TK], bl[FR];
static uint16_t tcl[FR * TK];

static void make_line(vocr_extra *ex, unsigned frames, unsigned k)
{
    unsigned f, j;
    memset(ex, 0, sizeof *ex);
    for (f = 0; f < frames; f++) {
        unsigned used = 0;
        float v = 8.0f - (float)urand(40) / 10.0f;
        for (j = 0; j < k; j++) {
            unsigned c;
            do c = urand(6); while (used & (1u << c));
            used |= 1u << c;
            tcl[f * k + j] = (uint16_t)c;
            tv[f * k + j] = v;
            if (urand(4)) v -= (float)urand(3000) / 1000.0f; /* sometimes an exact tie */
        }
        bl[f] = (used & 1u) ? tv[f * k + 0] - 50.0f : v - 1.0f; /* unused when the blank is in the top-K */
        for (j = 0; j < k; j++)
            if (!tcl[f * k + j]) bl[f] = tv[f * k + j];
    }
    ex->frames = frames;
    ex->cap_frames = FR;
    ex->topk = k;
    ex->topk_logit = tv;
    ex->topk_class = tcl;
    ex->blank_logit = bl;
}

static void greedy_text(const vocr_model *m, const vocr_extra *ex, char *out)
{
    unsigned f, last = 0, n;
    size_t used = 0;
    out[0] = 0;
    for (f = 0; f < ex->frames; f++) {
        const unsigned c = ex->topk_class[f * ex->topk];
        if (c && c != last) {
            const char *s = vocr_class_text(m, c, &n);
            memcpy(out + used, s, n);
            used += n;
            out[used] = 0;
        }
        last = c;
    }
}

static int stop_now(void *user)
{
    (void)user;
    return 1;
}

static void test_greedy_equivalence(void)
{
    vocr_model m;
    vocr_lm lm;
    vocr_lm_ctx c;
    unsigned trial;
    fake_model(&m);
    CHECK(vocr_lm_load(&lm, lmb, build_lm()) == VOCR_OK);
    for (trial = 0; trial < 3000; trial++) {
        vocr_lm_opts o;
        vocr_extra ex;
        char g[64], t[64];
        unsigned chars = 99, k = 1 + urand(TK);
        size_t need;
        make_line(&ex, 1 + urand(FR), k);
        if (urand(3) == 0) ex.blank_logit = NULL;
        memset(&o, 0, sizeof o);
        o.beam = 1 + urand(VOCR_LM_MAX_BEAM);
        o.topk = 1 + urand(TK + 2);
        o.gate = (int)urand(VOCR_LM_MAX_GATE + 1);
        o.flags = urand(4);
        need = vocr_lm_workspace_bytes(&m, urand(2) ? &lm : NULL, &o);
        CHECK(need > 0 && need <= sizeof ws);
        CHECK(vocr_lm_init(&c, &m, urand(2) ? &lm : NULL, &o, ws, need) == VOCR_OK);
        CHECK(vocr_lm_decode(&c, &ex, t, sizeof t, &chars) == VOCR_OK);
        greedy_text(&m, &ex, g);
        CHECK(strcmp(t, g) == 0);
        CHECK(chars <= ex.frames);
    }
}

static void test_lm_and_gate(void)
{
    /* frames: a, blank, then d (5.0) over c (4.8) over blank (0): greedy "ad". P(c | a) = -150,
     * P(d | a) = -420 (Q10): with alpha 1.0 the 270 LM gap beats the 205 logit margin -> "ac". */
    static const float v[4 * 3] = {9.0f, 1.0f, 0.0f, 7.0f, 2.0f, 1.0f, 5.0f, 4.8f, 0.0f, 6.0f, 0.5f, 0.2f};
    static const uint16_t cl[4 * 3] = {1, 0, 2, 0, 1, 3, 4, 3, 0, 0, 4, 2};
    static const float b[4] = {0.0f, 7.0f, 0.0f, 6.0f};
    vocr_model m;
    vocr_lm lm;
    vocr_lm_ctx c;
    vocr_lm_opts o = {8, 3, 0, 1000, 0, 3000, 0, 0};
    vocr_extra ex;
    char t[64];
    unsigned chars = 0;
    fake_model(&m);
    CHECK(vocr_lm_load(&lm, lmb, build_lm()) == VOCR_OK);
    memset(&ex, 0, sizeof ex);
    ex.frames = 4;
    ex.cap_frames = 4;
    ex.topk = 3;
    ex.topk_logit = (float *)(void *)v;
    ex.topk_class = (uint16_t *)(void *)cl;
    ex.blank_logit = (float *)(void *)b;
    CHECK(vocr_lm_init(&c, &m, &lm, &o, ws, sizeof ws) == VOCR_OK);
    CHECK(vocr_lm_decode(&c, &ex, t, sizeof t, &chars) == VOCR_OK);
    CHECK(strcmp(t, "ac") == 0 && chars == 2);
    o.gate = 150; /* 0.15 nat: the 0.2 nat frame is confident, the LM may not touch it */
    CHECK(vocr_lm_init(&c, &m, &lm, &o, ws, sizeof ws) == VOCR_OK);
    CHECK(vocr_lm_decode(&c, &ex, t, sizeof t, NULL) == VOCR_OK);
    CHECK(strcmp(t, "ad") == 0);
    o.gate = 3000;
    o.alpha = 500; /* 0.5 * 270 = 135 < 205 */
    CHECK(vocr_lm_init(&c, &m, &lm, &o, ws, sizeof ws) == VOCR_OK);
    CHECK(vocr_lm_decode(&c, &ex, t, sizeof t, NULL) == VOCR_OK);
    CHECK(strcmp(t, "ad") == 0);
    o.alpha = 1000;
    o.topk = 1; /* no alternatives at all */
    CHECK(vocr_lm_init(&c, &m, &lm, &o, ws, sizeof ws) == VOCR_OK);
    CHECK(vocr_lm_decode(&c, &ex, t, sizeof t, NULL) == VOCR_OK);
    CHECK(strcmp(t, "ad") == 0);
    /* the same input twice gives the same text: nothing carries over between lines */
    o.topk = 3;
    CHECK(vocr_lm_init(&c, &m, &lm, &o, ws, sizeof ws) == VOCR_OK);
    CHECK(vocr_lm_decode(&c, &ex, t, sizeof t, NULL) == VOCR_OK && strcmp(t, "ac") == 0);
    CHECK(vocr_lm_decode(&c, &ex, t, sizeof t, NULL) == VOCR_OK && strcmp(t, "ac") == 0);
    /* text overflow keeps whole characters only; cancellation empties the text */
    CHECK(vocr_lm_decode(&c, &ex, t, 2, &chars) == VOCR_E_TEXT && strcmp(t, "a") == 0 && chars == 1);
    ex.cancel = stop_now;
    CHECK(vocr_lm_decode(&c, &ex, t, sizeof t, NULL) == VOCR_E_CANCELLED && t[0] == 0);
}

/* A line of single-character frames: per frame its top-1 class, an alternative and their margin; a blank
 * frame (top-1 blank, margin 5 nats) separates characters. */
static float bv[64 * 3];
static uint16_t bc[64 * 3];
static unsigned bracket_line(vocr_extra *ex, const unsigned *top, const unsigned *alt, const float *margin, unsigned n)
{
    unsigned f = 0, i;
    for (i = 0; i < n; i++) {
        bc[3 * f] = (uint16_t)top[i];
        bc[3 * f + 1] = (uint16_t)alt[i];
        bc[3 * f + 2] = 0;
        bv[3 * f] = 5.0f;
        bv[3 * f + 1] = 5.0f - margin[i];
        bv[3 * f + 2] = 0.0f;
        f++;
        bc[3 * f] = 0;
        bc[3 * f + 1] = (uint16_t)top[i];
        bc[3 * f + 2] = (uint16_t)alt[i];
        bv[3 * f] = 5.0f;
        bv[3 * f + 1] = 0.0f;
        bv[3 * f + 2] = -1.0f;
        f++;
    }
    memset(ex, 0, sizeof *ex);
    ex->frames = f;
    ex->cap_frames = f;
    ex->topk = 3;
    ex->topk_logit = bv;
    ex->topk_class = bc;
    return f;
}

static void test_brackets(void)
{
    /* 『 a 」, the 」 frame 0.5 nat over 』: greedy keeps 」; a 2-nat bracket cost pairs it with 『 */
    static const unsigned t1[3] = {8, 1, 7}, a1[3] = {6, 2, 9};
    static const float m1[3] = {3.0f, 3.0f, 0.5f};
    /* a 」 alone (its opener on an earlier line): free */
    static const unsigned t2[2] = {1, 7}, a2[2] = {2, 9};
    static const float m2[2] = {3.0f, 0.5f};
    /* 「 『 a 』 」 nested and matched: unchanged */
    static const unsigned t3[5] = {6, 8, 1, 9, 7}, a3[5] = {8, 6, 2, 7, 9};
    static const float m3[5] = {0.5f, 0.5f, 3.0f, 0.5f, 0.5f};
    vocr_model m;
    vocr_lm_ctx c;
    vocr_lm_opts o = {8, 3, 64, 0, 0, 3000, 0, 0}; /* 64 frames: longer than the fake model's 8 */
    vocr_extra ex;
    char t[64];
    fake_model(&m);
    bracket_line(&ex, t1, a1, m1, 3);
    CHECK(vocr_lm_init(&c, &m, NULL, &o, ws, sizeof ws) == VOCR_OK);
    CHECK(vocr_lm_decode(&c, &ex, t, sizeof t, NULL) == VOCR_OK && strcmp(t, "\xe3\x80\x8e" "a\xe3\x80\x8d") == 0);
    o.bracket = 2000;
    CHECK(vocr_lm_workspace_bytes(&m, NULL, &o) > 0);
    CHECK(vocr_lm_init(&c, &m, NULL, &o, ws, sizeof ws) == VOCR_OK);
    CHECK(vocr_lm_decode(&c, &ex, t, sizeof t, NULL) == VOCR_OK && strcmp(t, "\xe3\x80\x8e" "a\xe3\x80\x8f") == 0);
    o.bracket = 400; /* below the 0.5-nat margin: the picture wins */
    CHECK(vocr_lm_init(&c, &m, NULL, &o, ws, sizeof ws) == VOCR_OK);
    CHECK(vocr_lm_decode(&c, &ex, t, sizeof t, NULL) == VOCR_OK && strcmp(t, "\xe3\x80\x8e" "a\xe3\x80\x8d") == 0);
    o.bracket = 2000;
    CHECK(vocr_lm_init(&c, &m, NULL, &o, ws, sizeof ws) == VOCR_OK);
    bracket_line(&ex, t2, a2, m2, 2);
    CHECK(vocr_lm_decode(&c, &ex, t, sizeof t, NULL) == VOCR_OK && strcmp(t, "a\xe3\x80\x8d") == 0);
    bracket_line(&ex, t3, a3, m3, 5);
    CHECK(vocr_lm_decode(&c, &ex, t, sizeof t, NULL) == VOCR_OK &&
          strcmp(t, "\xe3\x80\x8c\xe3\x80\x8e" "a\xe3\x80\x8f\xe3\x80\x8d") == 0);
}

static void test_errors(void)
{
    vocr_model m;
    vocr_lm lm;
    vocr_lm_ctx c;
    vocr_lm_opts o = {8, 4, 0, 300, 0, 3000, 0, 0}, bad;
    vocr_extra ex;
    char t[32];
    size_t need;
    fake_model(&m);
    CHECK(vocr_lm_load(&lm, lmb, build_lm()) == VOCR_OK);
    need = vocr_lm_workspace_bytes(&m, &lm, &o);
    CHECK(need > 0);
    CHECK(vocr_lm_init(&c, &m, &lm, &o, ws, need - 1) == VOCR_E_ARENA);
    CHECK(vocr_lm_init(&c, &m, &lm, &o, ws + 8, need) == VOCR_E_ALIGN);
    CHECK(vocr_lm_init(&c, &m, &lm, &o, ws, need) == VOCR_OK);
    CHECK(c.max_frames == 8);
    bad = o;
    bad.beam = 0;
    CHECK(vocr_lm_workspace_bytes(&m, &lm, &bad) == 0);
    bad.beam = VOCR_LM_MAX_BEAM + 1;
    CHECK(vocr_lm_workspace_bytes(&m, &lm, &bad) == 0);
    bad = o;
    bad.topk = VOCR_TOPK_MAX + 1;
    CHECK(vocr_lm_workspace_bytes(&m, &lm, &bad) == 0);
    bad = o;
    bad.alpha = VOCR_LM_MAX_ALPHA + 1;
    CHECK(vocr_lm_workspace_bytes(&m, &lm, &bad) == 0);
    bad = o;
    bad.beta = -VOCR_LM_MAX_BETA - 1;
    CHECK(vocr_lm_workspace_bytes(&m, &lm, &bad) == 0);
    bad = o;
    bad.gate = -1;
    CHECK(vocr_lm_workspace_bytes(&m, &lm, &bad) == 0);
    bad = o;
    bad.flags = 4;
    CHECK(vocr_lm_workspace_bytes(&m, &lm, &bad) == 0);
    bad = o;
    bad.bracket = VOCR_LM_MAX_BRACKET + 1;
    CHECK(vocr_lm_workspace_bytes(&m, &lm, &bad) == 0);
    bad.bracket = -1;
    CHECK(vocr_lm_workspace_bytes(&m, &lm, &bad) == 0);
    bad.bracket = 5000;   /* the bracket state needs room */
    CHECK(vocr_lm_workspace_bytes(&m, &lm, &bad) > need);
    CHECK(vocr_lm_workspace_bytes(&m, NULL, &o) == 0); /* alpha > 0 needs an LM */
    bad = o;
    bad.alpha = 0;
    CHECK(vocr_lm_workspace_bytes(&m, NULL, &bad) > 0 && vocr_lm_workspace_bytes(&m, NULL, &bad) < need);
    make_line(&ex, 8, 4);
    ex.frames = 9;
    ex.cap_frames = 9;
    CHECK(vocr_lm_decode(&c, &ex, t, sizeof t, NULL) == VOCR_E_WIDTH);
    ex.frames = 8;
    ex.cap_frames = 8;
    ex.topk = 0;
    CHECK(vocr_lm_decode(&c, &ex, t, sizeof t, NULL) == VOCR_E_ARG);
    ex.topk = 4;
    ex.topk_class = NULL;
    CHECK(vocr_lm_decode(&c, &ex, t, sizeof t, NULL) == VOCR_E_ARG);
    CHECK(vocr_lm_decode(NULL, &ex, t, sizeof t, NULL) == VOCR_E_ARG);
    CHECK(vocr_lm_decode(&c, NULL, t, sizeof t, NULL) == VOCR_E_ARG);
    CHECK(vocr_lm_decode(&c, &ex, t, 0, NULL) == VOCR_E_ARG);
    {
        size_t a, b;
        vocr_lm_sizes(&a, &b);
        CHECK(a == sizeof(vocr_lm) && b == sizeof(vocr_lm_ctx));
    }
}

int main(void)
{
    test_load();
    test_score();
    test_greedy_equivalence();
    test_lm_and_gate();
    test_brackets();
    test_errors();
    if (failures) {
        fprintf(stderr, "test_vocr_lm: %d failure(s)\n", failures);
        return 1;
    }
    printf("test_vocr_lm: all checks passed\n");
    return 0;
}
