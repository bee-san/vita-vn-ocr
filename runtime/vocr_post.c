/* vocr_post.c - post-processing of recognized text for counters (vocr_post.h, issue #109). */
#include "vocr_post.h"

#include <string.h>

#define CP_NONE 0xFFFFFFFFu
#define CP_TSU 0x30C3u     /* ッ */
#define CP_KE 0x30F6u      /* ヶ */
#define CP_KA 0x30F5u      /* ヵ */
#define CP_OPEN 0x300Cu    /* 「 */
#define CP_CLOSE 0x300Du   /* 」 */
#define CP_OPEN2 0x300Eu   /* 『 */
#define CP_CLOSE2 0x300Fu  /* 』 */
#define SIDE 4u            /* blank frames on each side of X's frames a numeral may take */
#define MAX_SPAN 16u       /* X's frames the re-decode reads at most (a glyph is a few) */
#define MAX_WIN (MAX_SPAN + 2u * SIDE)
#define MAX_CAND 32u       /* distinct numerals considered per context */

/* One code point of UTF-8 at s, at most n bytes; *k gets its bytes (1 for an invalid byte, CP_NONE). */
static uint32_t dec(const unsigned char *s, size_t n, unsigned *k)
{
    uint32_t c;
    unsigned len, i;
    *k = 1;
    if (!n) return CP_NONE;
    c = s[0];
    if (c < 0x80) return c;
    if ((c & 0xE0) == 0xC0) len = 2, c &= 0x1F;
    else if ((c & 0xF0) == 0xE0) len = 3, c &= 0x0F;
    else if ((c & 0xF8) == 0xF0) len = 4, c &= 0x07;
    else return CP_NONE;
    if (n < len) return CP_NONE;
    for (i = 1; i < len; i++) {
        if ((s[i] & 0xC0) != 0x80) return CP_NONE;
        c = c << 6 | (s[i] & 0x3Fu);
    }
    *k = len;
    return c;
}

static int is_numeral(uint32_t c)
{
    switch (c) {
    case 0x3007: case 0x4E00: case 0x4E8C: case 0x4E09: case 0x56DB: case 0x4E94: case 0x516D: /* 〇一二三四五六 */
    case 0x4E03: case 0x516B: case 0x4E5D: case 0x5341: case 0x767E: case 0x5343: case 0x4E07: /* 七八九十百千万 */
        return 1;
    default:
        return (c >= '0' && c <= '9') || (c >= 0xFF10 && c <= 0xFF19);
    }
}

static int is_quant(uint32_t c) { return c == 0x4F55 || c == 0x6570 || c == 0x5E7E; } /* 何 数 幾 */

/* ッ before right, after left, is a misread ヶ (vocr_post.h, rule 1). */
static int counter_tsu(uint32_t left, uint32_t right)
{
    switch (right) {
    case 0x6708: case 0x5E74:                       /* 月 年 */
        return 1;
    case 0x56FD: case 0x6761:                       /* 国 条 */
        return is_numeral(left) || is_quant(left);
    case 0x6240:                                    /* 所 */
        return (is_numeral(left) && left != 0x4E00) || is_quant(left);
    default:
        return 0;
    }
}

unsigned vocr_post_counters(char *text)
{
    unsigned char *s = (unsigned char *)text;
    uint32_t prev = 0;
    unsigned n = 0, k, k2;
    size_t left;
    if (!s) return 0;
    left = strlen(text);
    while (left) {
        uint32_t c = dec(s, left, &k);
        if (c == CP_TSU && counter_tsu(prev, dec(s + 3, left - 3, &k2))) {
            s[2] = 0xB6; /* E3 83 83 -> E3 83 B6 */
            c = CP_KE;
            n++;
        }
        prev = c;
        s += k;
        left -= k;
    }
    return n;
}

/* ---------------------------------------------------------------- re-decode */

static int32_t milli_q10(int x) /* 1/1000 -> Q10, half away from zero (vocr_lm.c's) */
{
    const int32_t a = x < 0 ? -(int32_t)x : (int32_t)x, q = (a * 1024 + 500) / 1000;
    return x < 0 ? -q : q;
}

/* The cost in Q10 nats of reading v instead of the frame's top-1 top (vocr_lm.c's rounding and clamp). */
static int32_t cost(float top, float v)
{
    float d = top - v;
    if (!(d >= 0.0f)) d = 0.0f;
    else if (d > 64.0f) d = 64.0f;
    return (int32_t)(d * 1024.0f + 0.5f);
}

static uint32_t class_cp(const vocr_model *m, unsigned c)
{
    unsigned len, k;
    const char *t = vocr_class_text(m, c, &len);
    uint32_t cp;
    if (!len) return 0;
    cp = dec((const unsigned char *)t, len, &k);
    return k == len ? cp : CP_NONE; /* a class of several characters matches nothing */
}

static int is_letter(uint32_t c)
{
    return (c >= 0x30A1 && c <= 0x30FA) || c == 0x30FC ||          /* katakana, ー */
           (c >= 0x3041 && c <= 0x3096) ||                         /* hiragana */
           (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF) || c == 0x3005 || /* kanji, 々 */
           (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
           (c >= 0xFF21 && c <= 0xFF3A) || (c >= 0xFF41 && c <= 0xFF5A);
}

static int is_counter(uint32_t c)
{
    return c == 0x6708 || c == 0x6240 || c == 0x5E74 || c == 0x56FD || c == 0x6761; /* 月 所 年 国 条 */
}

typedef struct {
    unsigned cls, s, e; /* class, first and last frame */
    uint32_t cp;
} run_t;

/* Position of class c in frame f's top-K, or -1. */
static int pos(const uint16_t *tc, unsigned K, unsigned f, unsigned c)
{
    unsigned j;
    for (j = 0; j < K; j++)
        if (tc[(size_t)f * K + j] == c) return (int)j;
    return -1;
}

static void swap0(uint16_t *tc, unsigned K, unsigned f, int j)
{
    uint16_t t;
    if (j <= 0) return;
    t = tc[(size_t)f * K];
    tc[(size_t)f * K] = tc[(size_t)f * K + (unsigned)j];
    tc[(size_t)f * K + (unsigned)j] = t;
}

/* X (run x, after the run ending at frame prev_end, -1 for none) before the run starting at frame next:
 * the cheapest numeral path over the window around x, and its edit when it costs at most tau. Returns the
 * first frame of the (possibly new) run X's characters take. */
static unsigned try_numeral(const vocr_model *m, vocr_extra *ex, const run_t *x, long prev_end, unsigned next,
                            int32_t tau, int *done)
{
    const unsigned K = ex->topk;
    const float *tv = ex->topk_logit;
    uint16_t *tc = ex->topk_class;
    unsigned w0, w1, n, f, a, b, i, ci, nc = 0, cand[MAX_CAND], best_a = 0, best_b = 0, best_c = 0;
    int32_t bc[MAX_WIN], dc[MAX_WIN], pb[MAX_WIN + 1], sb[MAX_WIN + 1], best = -1;
    unsigned char bok[MAX_WIN], dok[MAX_WIN], pok[MAX_WIN + 1], sok[MAX_WIN + 1];
    *done = 0;
    if (x->e - x->s + 1u > MAX_SPAN) return x->s;
    /* the window: X's frames and up to SIDE blank frames on each side, between the runs around X */
    w0 = x->s > SIDE ? x->s - SIDE : 0u;
    if ((long)w0 <= prev_end) w0 = (unsigned)(prev_end + 1);
    w1 = x->e + SIDE < next ? x->e + SIDE : next - 1u;
    n = w1 - w0 + 1u;
    /* the blank's cost per frame of the window (0 where it is the top-1), with prefix and suffix sums */
    for (i = 0; i < n; i++) {
        f = w0 + i;
        if (tc[(size_t)f * K] == 0) {
            bc[i] = 0;
            bok[i] = 1;
        } else {
            const int j = pos(tc, K, f, 0);
            bok[i] = (unsigned char)(j > 0);
            bc[i] = j > 0 ? cost(tv[(size_t)f * K], tv[(size_t)f * K + (unsigned)j]) : 0;
        }
    }
    pb[0] = 0;
    pok[0] = 1;
    for (i = 0; i < n; i++) {
        pb[i + 1] = pb[i] + bc[i];
        pok[i + 1] = (unsigned char)(pok[i] && bok[i]);
    }
    sb[n] = 0;
    sok[n] = 1;
    for (i = n; i-- > 0;) {
        sb[i] = sb[i + 1] + bc[i];
        sok[i] = (unsigned char)(sok[i + 1] && bok[i]);
    }
    for (f = x->s; f <= x->e && nc < MAX_CAND; f++) { /* the numerals in X's frames' top-K */
        unsigned j;
        for (j = 1; j < K && nc < MAX_CAND; j++) {
            const unsigned c = tc[(size_t)f * K + j];
            uint32_t cp;
            unsigned q;
            if (!c || c == x->cls) continue;
            for (q = 0; q < nc && cand[q] != c; q++) {
            }
            if (q < nc) continue;
            cp = class_cp(m, c);
            if (is_numeral(cp) || is_quant(cp)) cand[nc++] = c;
        }
    }
    /* each numeral d: the cheapest window path with one run [a, b] of d and the blank elsewhere */
    for (ci = 0; ci < nc; ci++) {
        const unsigned d = cand[ci];
        for (i = 0; i < n; i++) {
            const int j = pos(tc, K, w0 + i, d);
            dok[i] = (unsigned char)(j >= 0);
            dc[i] = j >= 0 ? cost(tv[(size_t)(w0 + i) * K], tv[(size_t)(w0 + i) * K + (unsigned)j]) : 0;
        }
        for (a = 0; a < n && pok[a]; a++) {
            int32_t run = 0;
            /* a run of d right after a run of d would merge with it (CTC) */
            if (prev_end >= 0 && (long)(w0 + a) == prev_end + 1 && tc[(size_t)prev_end * K] == d) continue;
            for (b = a; b < n && dok[b]; b++) {
                int32_t tot;
                run += dc[b];
                if (!sok[b + 1]) continue;
                tot = pb[a] + run + sb[b + 1];
                if (best < 0 || tot < best || (tot == best && d < best_c)) {
                    best = tot;
                    best_a = w0 + a;
                    best_b = w0 + b;
                    best_c = d;
                }
            }
        }
    }
    if (best < 0 || best > tau) return x->s;
    for (f = w0; f <= w1; f++) {
        if (f >= best_a && f <= best_b) swap0(tc, K, f, pos(tc, K, f, best_c));
        else if (tc[(size_t)f * K] != 0) swap0(tc, K, f, pos(tc, K, f, 0));
    }
    *done = 1;
    return best_a;
}

/* The opening bracket cp at one of the blank frames [0, end), when its cost is at most open. */
static int try_open(const vocr_model *m, vocr_extra *ex, uint32_t cp, unsigned end, int32_t open)
{
    const unsigned K = ex->topk;
    const float *tv = ex->topk_logit;
    uint16_t *tc = ex->topk_class;
    unsigned f, j, best_f = 0, best_j = 0;
    int32_t best = -1;
    for (f = 0; f < end; f++) {
        if (tc[(size_t)f * K] != 0) continue;
        for (j = 1; j < K; j++) {
            const unsigned c = tc[(size_t)f * K + j];
            if (c && class_cp(m, c) == cp) {
                const int32_t q = cost(tv[(size_t)f * K], tv[(size_t)f * K + j]);
                if (best < 0 || q < best) {
                    best = q;
                    best_f = f;
                    best_j = j;
                }
                break;
            }
        }
    }
    if (best < 0 || best > open) return 0;
    swap0(tc, K, best_f, (int)best_j);
    return 1;
}

int vocr_post_redecode(const vocr_model *m, vocr_extra *ex, const vocr_post_opts *o, vocr_post_stats *st)
{
    unsigned K, T, f, nrun = 0, cls, start = 0;
    uint32_t want = 0;
    int32_t tau, open;
    run_t r[4];
    int edits = 0;
    if (st) memset(st, 0, sizeof *st);
    if (!m || !ex || !o || !ex->topk || ex->topk > VOCR_TOPK_MAX || !ex->topk_logit || !ex->topk_class ||
        o->tau < 0 || o->tau > VOCR_POST_MAX_TAU || o->open < 0 || o->open > VOCR_POST_MAX_TAU)
        return VOCR_E_ARG;
    K = ex->topk;
    T = ex->frames;
    if (T > ex->cap_frames || T > VOCR_POST_MAX_FRAMES) return VOCR_E_WIDTH;
    for (f = 0; f < T * K; f++)
        if (ex->topk_class[f] >= m->n_classes) return VOCR_E_ARG;
    if (!o->tau && !o->open) return 0;
    tau = milli_q10(o->tau);
    open = milli_q10(o->open);
    /* the line's first bracket: a closer means the opener is missing */
    for (f = 0, cls = 0; f < T; f++) {
        const unsigned c = ex->topk_class[(size_t)f * K];
        if (c && c != cls) {
            const uint32_t cp = class_cp(m, c);
            if (cp == CP_OPEN || cp == CP_OPEN2) break;
            if (cp == CP_CLOSE || cp == CP_CLOSE2) {
                want = cp == CP_CLOSE ? CP_OPEN : CP_OPEN2;
                break;
            }
        }
        cls = c;
    }
    /* the greedy runs, three at a time: X, then ッ/ヶ/ヵ, then the counter */
    for (f = 0, cls = 0; f <= T; f++) {
        const unsigned c = f < T ? ex->topk_class[(size_t)f * K] : 0u;
        if (c == cls) continue;
        if (cls) { /* the run of cls [start, f) ends */
            run_t *q;
            if (nrun >= 4) memmove(r, r + 1, 3 * sizeof r[0]);
            q = &r[nrun < 4 ? nrun : 3];
            q->cls = cls;
            q->s = start;
            q->e = f - 1u;
            q->cp = class_cp(m, cls);
            if (nrun < 4) nrun++;
            if (nrun >= 3) {
                const run_t *x = &r[nrun - 3], *mid = &r[nrun - 2], *ctr = &r[nrun - 1];
                if ((mid->cp == CP_TSU || mid->cp == CP_KE || mid->cp == CP_KA) && is_counter(ctr->cp) &&
                    is_letter(x->cp) && !is_numeral(x->cp) && !is_quant(x->cp)) {
                    const long prev_end = nrun == 4 ? (long)r[0].e : -1L;
                    unsigned first = x->s;
                    int done = 0;
                    if (st) st->contexts++;
                    if (tau) first = try_numeral(m, ex, x, prev_end, mid->s, tau, &done);
                    if (done) {
                        edits++;
                        if (st) st->numerals++;
                    }
                    if (open && want && prev_end < 0 && try_open(m, ex, want, first, open)) {
                        want = 0;
                        edits++;
                        if (st) st->brackets++;
                    }
                }
            }
        }
        cls = c;
        start = f;
    }
    return edits;
}

int vocr_post_greedy(const vocr_model *m, const vocr_extra *ex, char *text, size_t text_cap, unsigned *chars)
{
    unsigned K, T, f, last = 0, n = 0, len;
    size_t used = 0;
    int overflow = 0;
    if (chars) *chars = 0;
    if (!m || !ex || !text || !text_cap || !ex->topk || ex->topk > VOCR_TOPK_MAX || !ex->topk_class)
        return VOCR_E_ARG;
    text[0] = 0;
    K = ex->topk;
    T = ex->frames;
    if (T > ex->cap_frames) return VOCR_E_WIDTH;
    for (f = 0; f < T; f++) {
        const unsigned c = ex->topk_class[(size_t)f * K];
        if (c >= m->n_classes) {
            text[0] = 0;
            return VOCR_E_ARG;
        }
        if (c && c != last) {
            const char *s = vocr_class_text(m, c, &len);
            if (!overflow && used + len < text_cap) {
                memcpy(text + used, s, len);
                used += len;
                text[used] = 0;
                n++;
            } else {
                overflow = 1;
            }
        }
        last = c;
    }
    if (chars) *chars = n;
    return overflow ? VOCR_E_TEXT : VOCR_OK;
}
