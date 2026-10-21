/* vocr_post_test - self-test and parity driver of vocr_post.c (not part of the console build).
 *
 *   vocr_post_test                      the self-test (unit cases of the rule, the re-decode and the greedy text)
 *   vocr_post_test --parity VOCAB CASES one result line per case of CASES (tests/post_parity.py writes them)
 *
 * VOCAB: one class per line, UTF-8; class 0 is the blank and is not in the file, so line i is class i + 1.
 * CASES: u32 count, then per case u32 kind: 0 (rule) u32 bytes + the text; 1 (re-decode) u32 frames,
 * u32 K, i32 tau, i32 open, frames x K float32 logits, frames x K u16 classes.
 * Rule result: "R <changes> <hex of the text>". Re-decode result: "D <rc> <contexts> <numerals> <brackets>
 * <hex of the edited greedy text> <the edited classes, frame by frame>".
 */
#include "vocr_post.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) {                                                           \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #c); \
            failures++;                                                       \
        }                                                                     \
    } while (0)

/* ------------------------------------------------------------- a model of a few classes, for the self-test */

static const char *const tiny[] = {"", "「", "」", "九", "ル", "ッ", "ヶ", "月", "で", "す", "一", "ー",
                                   "所", "二", "ニ", "し", "ヵ", "国", "『", "』", "八"};
enum { B = 0, OP, CL, KU, RU, TSU, KE, TSUKI, DE, SU, ICHI, CHOON, SHO, NI, NIK, SHI, KA, KUNI, OP2, CL2, HACHI,
       NTINY };

static uint16_t tiny_off[NTINY + 1];
static char tiny_voc[256];

static void tiny_model(vocr_model *m)
{
    size_t used = 0;
    unsigned i;
    memset(m, 0, sizeof *m);
    for (i = 0; i < NTINY; i++) {
        size_t n = strlen(tiny[i]);
        tiny_off[i] = (uint16_t)used;
        memcpy(tiny_voc + used, tiny[i], n);
        used += n;
    }
    tiny_off[NTINY] = (uint16_t)used;
    m->n_classes = NTINY;
    m->vocab_off = tiny_off;
    m->vocab = tiny_voc;
}

static char *rule(const char *in, unsigned *n)
{
    static char buf[256];
    strcpy(buf, in);
    *n = vocr_post_counters(buf);
    return buf;
}

static void test_rule(void)
{
    unsigned n;
    /* the owner's line: ッ before 月 whatever its left */
    CHECK(!strcmp(rule("ルッ月です」", &n), "ルヶ月です」") && n == 1);
    CHECK(!strcmp(rule("九ッ月", &n), "九ヶ月") && n == 1);
    CHECK(!strcmp(rule("3ッ月と５ッ年", &n), "3ヶ月と５ヶ年") && n == 2);
    CHECK(!strcmp(rule("ッ月", &n), "ヶ月") && n == 1);          /* at the start: no left */
    /* 国 条 after a numeral or 何 数 幾 only */
    CHECK(!strcmp(rule("十ッ国、数ッ条、何ッ国", &n), "十ヶ国、数ヶ条、何ヶ国") && n == 3);
    CHECK(!strcmp(rule("えッ国上山", &n), "えッ国上山") && n == 0); /* the corpora's own */
    CHECK(!strcmp(rule("為ッ国の為ッて", &n), "為ッ国の為ッて") && n == 0);
    CHECK(!strcmp(rule("命に別ッ条ない", &n), "命に別ッ条ない") && n == 0);
    /* 所: not after 一 (Aozora's 一ッ所, ひとつところ) */
    CHECK(!strcmp(rule("一ッ所に立つ", &n), "一ッ所に立つ") && n == 0);
    CHECK(!strcmp(rule("三ッ所、幾ッ所", &n), "三ヶ所、幾ヶ所") && n == 2);
    /* not covered: 日 村 町, place names, interjections, other small kana */
    CHECK(!strcmp(rule("三ッ日月", &n), "三ッ日月") && n == 0);
    CHECK(!strcmp(rule("イマドキッ日曜日", &n), "イマドキッ日曜日") && n == 0);
    CHECK(!strcmp(rule("四ッ谷、三ッ村、二ッ町", &n), "四ッ谷、三ッ村、二ッ町") && n == 0);
    CHECK(!strcmp(rule("九っ月、九ツ月、九ヵ月、九ケ月", &n), "九っ月、九ツ月、九ヵ月、九ケ月") && n == 0);
    CHECK(!strcmp(rule("ベッド", &n), "ベッド") && n == 0);
    /* a line break is never a numeral, and ッ at a line's end has nothing on its right */
    CHECK(!strcmp(rule("三\nッ国", &n), "三\nッ国") && n == 0);
    CHECK(!strcmp(rule("九ッ\n月", &n), "九ッ\n月") && n == 0);
    CHECK(!strcmp(rule("九ッッ月", &n), "九ッヶ月") && n == 1);
    /* invalid UTF-8 is kept, and a cut ッ is not read */
    {
        char b[16] = {'\xff', '\xe3', '\x83', '\x83', '\xe6', '\x9c', '\x88', 0};
        CHECK(vocr_post_counters(b) == 1 && !memcmp(b, "\xff\xe3\x83\xb6\xe6\x9c\x88", 8));
        strcpy(b, "\xe3\x83\x83\xe6\x9c");                         /* ッ then a cut 月 */
        CHECK(vocr_post_counters(b) == 0 && !strcmp(b, "\xe3\x83\x83\xe6\x9c"));
        strcpy(b, "\xe3\x83");
        CHECK(vocr_post_counters(b) == 0);
    }
    CHECK(vocr_post_counters(NULL) == 0);
    {
        char e[1] = {0};
        CHECK(vocr_post_counters(e) == 0);
    }
}

/* ------------------------------------------------------------- re-decode */

#define F 24
#define KK 4
static float tv[F * KK];
static uint16_t tcl[F * KK];

/* frame f: classes c[0..3] with logits v[0..3] (decreasing) */
static void frame(unsigned f, unsigned c0, float v0, unsigned c1, float v1, unsigned c2, float v2, unsigned c3,
                  float v3)
{
    tcl[f * KK] = (uint16_t)c0, tcl[f * KK + 1] = (uint16_t)c1, tcl[f * KK + 2] = (uint16_t)c2;
    tcl[f * KK + 3] = (uint16_t)c3;
    tv[f * KK] = v0, tv[f * KK + 1] = v1, tv[f * KK + 2] = v2, tv[f * KK + 3] = v3;
}

/* blank, ル (九 3 nats lower), ッ, 月, で, す, 」 with blanks between: greedy "ルッ月です」"; 「 at frame 1
 * 5 nats under the blank */
static unsigned owner_line(void)
{
    unsigned f = 0;
    frame(f++, B, 10, OP, 2, CL, 1, KU, 0);
    frame(f++, B, 10, OP, 5, KU, 4, RU, 1);
    frame(f++, RU, 9, KU, 6, B, 5, SHI, 1);
    frame(f++, RU, 9, B, 7, KU, 5.5f, SHI, 1);
    frame(f++, B, 10, RU, 3, KU, 2, TSU, 1);
    frame(f++, TSU, 8, KE, 6, B, 5, KA, 1);
    frame(f++, B, 10, TSU, 2, KE, 1, DE, 0);
    frame(f++, TSUKI, 9, B, 4, SU, 1, DE, 0);
    frame(f++, B, 10, TSUKI, 3, DE, 1, SU, 0);
    frame(f++, DE, 9, B, 4, SU, 1, CL, 0);
    frame(f++, B, 10, DE, 2, SU, 1, CL, 0);
    frame(f++, SU, 9, B, 4, DE, 1, CL, 0);
    frame(f++, B, 10, SU, 2, CL, 1, DE, 0);
    frame(f++, CL, 9, B, 3, SU, 1, DE, 0);
    frame(f++, B, 10, CL, 2, SU, 1, DE, 0);
    return f;
}

static void extras(vocr_extra *ex, unsigned frames)
{
    memset(ex, 0, sizeof *ex);
    ex->frames = frames;
    ex->cap_frames = F;
    ex->topk = KK;
    ex->topk_logit = tv;
    ex->topk_class = tcl;
}

static void test_redecode(void)
{
    vocr_model m;
    vocr_extra ex;
    vocr_post_opts o;
    vocr_post_stats st;
    char text[128];
    uint16_t keep[F * KK];
    unsigned T;
    tiny_model(&m);
    T = owner_line();
    extras(&ex, T);
    CHECK(vocr_post_greedy(&m, &ex, text, sizeof text, NULL) == VOCR_OK && !strcmp(text, "ルッ月です」"));
    memcpy(keep, tcl, sizeof keep);
    /* off: nothing changes, nothing is written */
    o.tau = o.open = 0;
    CHECK(vocr_post_redecode(&m, &ex, &o, &st) == 0 && !memcmp(keep, tcl, sizeof keep) && !st.contexts);
    /* ル -> 九: the cheapest path reads 九 at frame 2 (9 - 6 = 3 nats) and the blank at frame 3 (9 - 7 = 2):
     * 5 nats, 5120 in Q10; both 九 frames would cost 6.5 */
    o.tau = 4999; /* 5119 in Q10 */
    CHECK(vocr_post_redecode(&m, &ex, &o, &st) == 0 && st.contexts == 1 && !st.numerals);
    CHECK(!memcmp(keep, tcl, sizeof keep));
    o.tau = 5000;
    CHECK(vocr_post_redecode(&m, &ex, &o, &st) == 1 && st.numerals == 1 && !st.brackets);
    CHECK(vocr_post_greedy(&m, &ex, text, sizeof text, NULL) == VOCR_OK && !strcmp(text, "九ッ月です」"));
    vocr_post_counters(text);
    CHECK(!strcmp(text, "九ヶ月です」"));
    /* the bracket: 5 nats under the blank at frame 1 */
    memcpy(tcl, keep, sizeof keep);
    o.tau = 6000;
    o.open = 4999;
    CHECK(vocr_post_redecode(&m, &ex, &o, &st) == 1 && st.numerals == 1 && !st.brackets);
    memcpy(tcl, keep, sizeof keep);
    o.open = 5000;
    CHECK(vocr_post_redecode(&m, &ex, &o, &st) == 2 && st.numerals == 1 && st.brackets == 1);
    CHECK(vocr_post_greedy(&m, &ex, text, sizeof text, NULL) == VOCR_OK && !strcmp(text, "「九ッ月です」"));
    /* the bracket alone, X kept */
    memcpy(tcl, keep, sizeof keep);
    o.tau = 0;
    CHECK(vocr_post_redecode(&m, &ex, &o, &st) == 1 && !st.numerals && st.brackets == 1);
    CHECK(vocr_post_greedy(&m, &ex, text, sizeof text, NULL) == VOCR_OK && !strcmp(text, "「ルッ月です」"));
    /* no closer in the line: no bracket */
    memcpy(tcl, keep, sizeof keep);
    extras(&ex, 13); /* up to す */
    CHECK(vocr_post_redecode(&m, &ex, &o, &st) == 0 && st.contexts == 1);
    /* a numeral X is no context; neither is a bracket X */
    memcpy(tcl, keep, sizeof keep);
    tcl[2 * KK] = KU, tcl[2 * KK + 1] = RU;
    tcl[3 * KK] = KU, tcl[3 * KK + 2] = RU;
    extras(&ex, T);
    o.tau = o.open = 64000;
    CHECK(vocr_post_redecode(&m, &ex, &o, &st) == 0 && !st.contexts);
    memcpy(tcl, keep, sizeof keep);
    /* a frame lacking both the numeral and the blank: no path */
    {
        unsigned f = 0;
        frame(f++, CHOON, 9, ICHI, 8, CL, 1, OP, 0);
        frame(f++, CHOON, 9, SHI, 2, CL, 1, OP, 0);
        frame(f++, KE, 8, TSU, 6, B, 5, KA, 1);
        frame(f++, TSUKI, 9, B, 4, SU, 1, DE, 0);
        extras(&ex, f);
        o.tau = 64000;
        o.open = 0;
        CHECK(vocr_post_redecode(&m, &ex, &o, &st) == 0 && st.contexts == 1);
        /* with the blank in the second frame: 一 then the blank, 1 + 3 nats */
        frame(1, CHOON, 9, SHI, 2, B, 6, OP, 0);
        CHECK(vocr_post_redecode(&m, &ex, &o, &st) == 1);
        CHECK(vocr_post_greedy(&m, &ex, text, sizeof text, NULL) == VOCR_OK && !strcmp(text, "一ヶ月"));
    }
    /* a numeral run cannot merge with the same numeral right before it: 二 + ニ + ッ + 所 */
    {
        unsigned f = 0;
        frame(f++, NI, 9, B, 5, NIK, 1, OP, 0);
        frame(f++, NIK, 9, NI, 8.5f, B, 5, OP, 0);
        frame(f++, B, 9, NIK, 3, NI, 2, OP, 0);
        frame(f++, TSU, 8, KE, 6, B, 5, KA, 1);
        frame(f++, SHO, 9, B, 4, SU, 1, DE, 0);
        extras(&ex, f);
        o.tau = 64000;
        CHECK(vocr_post_redecode(&m, &ex, &o, &st) == 1);
        CHECK(vocr_post_greedy(&m, &ex, text, sizeof text, NULL) == VOCR_OK && !strcmp(text, "二二ッ所"));
        CHECK(tcl[1 * KK] == B && tcl[2 * KK] == NI); /* 二 at frame 2 after the blank, not at frame 1 */
    }
    /* errors */
    extras(&ex, T);
    o.tau = 1000;
    CHECK(vocr_post_redecode(NULL, &ex, &o, NULL) == VOCR_E_ARG);
    CHECK(vocr_post_redecode(&m, &ex, NULL, NULL) == VOCR_E_ARG);
    o.tau = -1;
    CHECK(vocr_post_redecode(&m, &ex, &o, NULL) == VOCR_E_ARG);
    o.tau = VOCR_POST_MAX_TAU + 1;
    CHECK(vocr_post_redecode(&m, &ex, &o, NULL) == VOCR_E_ARG);
    o.tau = 1000;
    ex.cap_frames = T - 1;
    CHECK(vocr_post_redecode(&m, &ex, &o, NULL) == VOCR_E_WIDTH);
    ex.cap_frames = F;
    ex.topk = 0;
    CHECK(vocr_post_redecode(&m, &ex, &o, NULL) == VOCR_E_ARG);
    ex.topk = KK;
    tcl[5] = NTINY;
    CHECK(vocr_post_redecode(&m, &ex, &o, NULL) == VOCR_E_ARG);
    CHECK(vocr_post_greedy(&m, &ex, text, sizeof text, NULL) == VOCR_OK); /* only the top-1 is read */
    tcl[4] = NTINY;
    CHECK(vocr_post_greedy(&m, &ex, text, sizeof text, NULL) == VOCR_E_ARG && !text[0]);
}

static void test_greedy(void)
{
    vocr_model m;
    vocr_extra ex;
    char text[8];
    unsigned n;
    tiny_model(&m);
    owner_line();
    extras(&ex, owner_line());
    /* whole characters only: ル ッ (6 bytes) fit in 8 with the NUL, 月 does not */
    CHECK(vocr_post_greedy(&m, &ex, text, sizeof text, &n) == VOCR_E_TEXT && !strcmp(text, "ルッ") && n == 2);
    CHECK(vocr_post_greedy(&m, &ex, text, 0, &n) == VOCR_E_ARG);
    CHECK(vocr_post_greedy(&m, NULL, text, sizeof text, &n) == VOCR_E_ARG);
}

/* ------------------------------------------------------------- parity driver */

static unsigned char *read_all(const char *path, size_t *n)
{
    FILE *f = fopen(path, "rb");
    unsigned char *b = NULL;
    long len;
    if (!f) return NULL;
    if (!fseek(f, 0, SEEK_END) && (len = ftell(f)) >= 0 && !fseek(f, 0, SEEK_SET)) {
        b = (unsigned char *)malloc((size_t)len + 1u);
        if (b && fread(b, 1, (size_t)len, f) != (size_t)len) {
            free(b);
            b = NULL;
        }
        if (b) {
            b[len] = 0;
            *n = (size_t)len;
        }
    }
    fclose(f);
    return b;
}

static void hex(const char *s)
{
    if (!*s) fputs("-", stdout);
    for (; *s; s++) printf("%02x", (unsigned)(unsigned char)*s);
}

static uint32_t rd(const unsigned char **p)
{
    const unsigned char *q = *p;
    *p += 4;
    return (uint32_t)q[0] | (uint32_t)q[1] << 8 | (uint32_t)q[2] << 16 | (uint32_t)q[3] << 24;
}

static int parity(const char *vocab_path, const char *cases_path)
{
    size_t nv = 0, nc = 0, i, used = 0;
    unsigned char *voc = read_all(vocab_path, &nv), *cs = read_all(cases_path, &nc);
    const unsigned char *p;
    static uint16_t off[65536];
    static char text[1 << 16];
    vocr_model m;
    uint32_t count, k;
    unsigned ncls = 1;
    if (!voc || !cs) return 2;
    /* classes: 0 the blank, then one per line */
    off[0] = off[1] = 0;
    for (i = 0; i < nv; i++) {
        if (voc[i] == '\n') {
            if (ncls >= 65535) return 2;
            off[++ncls] = (uint16_t)used;
        } else {
            voc[used++] = voc[i];
        }
    }
    if (used != off[ncls]) off[++ncls] = (uint16_t)used; /* a last line without its newline */
    memset(&m, 0, sizeof m);
    m.n_classes = ncls;
    m.vocab_off = off;
    m.vocab = (const char *)voc;
    p = cs;
    count = rd(&p);
    for (k = 0; k < count; k++) {
        const uint32_t kind = rd(&p);
        if (kind == 0) {
            const uint32_t n = rd(&p);
            unsigned ch;
            memcpy(text, p, n);
            text[n] = 0;
            p += n;
            ch = vocr_post_counters(text);
            printf("R %u ", ch);
            hex(text);
            putchar('\n');
        } else {
            const uint32_t T = rd(&p), K = rd(&p);
            vocr_post_opts o;
            vocr_post_stats st;
            vocr_extra ex;
            float *lv = (float *)malloc((size_t)T * K * sizeof(float) + 4u);
            uint16_t *lc = (uint16_t *)malloc((size_t)T * K * sizeof(uint16_t) + 2u);
            int rc;
            unsigned f;
            o.tau = (int)rd(&p);
            o.open = (int)rd(&p);
            if (!lv || !lc) return 2;
            memcpy(lv, p, (size_t)T * K * 4u);
            p += (size_t)T * K * 4u;
            memcpy(lc, p, (size_t)T * K * 2u);
            p += (size_t)T * K * 2u;
            memset(&ex, 0, sizeof ex);
            ex.frames = T;
            ex.cap_frames = T;
            ex.topk = K;
            ex.topk_logit = lv;
            ex.topk_class = lc;
            rc = vocr_post_redecode(&m, &ex, &o, &st);
            printf("D %d %u %u %u ", rc, st.contexts, st.numerals, st.brackets);
            if (vocr_post_greedy(&m, &ex, text, sizeof text, NULL) != VOCR_OK) text[0] = 0;
            hex(text);
            for (f = 0; f < T * K; f++) printf(" %u", (unsigned)lc[f]);
            putchar('\n');
            free(lv);
            free(lc);
        }
    }
    free(voc);
    free(cs);
    return 0;
}

int main(int argc, char **argv)
{
    if (argc == 4 && !strcmp(argv[1], "--parity")) return parity(argv[2], argv[3]);
    if (argc != 1) {
        fprintf(stderr, "usage: vocr_post_test [--parity VOCAB CASES]\n");
        return 2;
    }
    test_rule();
    test_redecode();
    test_greedy();
    if (failures) {
        fprintf(stderr, "vocr_post_test: %d failures\n", failures);
        return 1;
    }
    printf("vocr_post_test: all checks passed\n");
    return 0;
}
