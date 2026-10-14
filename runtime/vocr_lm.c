#include "vocr_lm.h"

#include <string.h>

/* The specification is lm/vlm.py (the blob layout, LM.score, decode); tests/test_ctclm.py requires this
 * file's output to equal it bit for bit. */

#define HEADER 64u
#define ROW 32u
#define TABLE (HEADER + (unsigned)VOCR_LM_MAX_ORDER * ROW)
#define FIRST 3u
#define BOS 1u
#define EOS 2u
#define NONE (-1)
#define NEG (-(1 << 30))
#define CANCEL_EVERY 8u
#define LP_FLOOR (-65536)   /* -64 nats: the LM term per character is clamped to [LP_FLOOR, 0] */
#define MAX_FRAMES 1024u    /* with the clamps, 1024 frames of the worst scores stay above NEG */

static uint32_t rd32(const unsigned char *p)
{
    return (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16 | (uint32_t)p[3] << 24;
}
static size_t al16(size_t n) { return (n + 15u) & ~(size_t)15u; }
static int32_t imax(int32_t a, int32_t b) { return a > b ? a : b; }

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

int vocr_lm_check_crc(const void *blob, size_t bytes)
{
    const unsigned char *b = (const unsigned char *)blob;
    uint32_t total;
    if (!b || bytes < TABLE || memcmp(b, "VOLM", 4)) return VOCR_E_FORMAT;
    total = rd32(b + 8);
    if (total < TABLE || total > bytes) return VOCR_E_FORMAT;
    return crc32_ieee(b + HEADER, total - HEADER) == rd32(b + 24) ? VOCR_OK : VOCR_E_FORMAT;
}

/* [off, off + n) lies inside the blob, after the order table, 16-byte aligned */
static int in_blob(uint32_t total, uint32_t off, uint32_t n)
{
    return off >= TABLE && !(off & 15u) && off <= total && n <= total - off;
}

int vocr_lm_load(vocr_lm *lm, const void *blob, size_t bytes)
{
    const unsigned char *b = (const unsigned char *)blob;
    const uint16_t one = 1;
    uint32_t total, order, V, voc, k, e;
    if (!lm || !b) return VOCR_E_ARG;
    memset(lm, 0, sizeof *lm);
    if ((uintptr_t)b & 15u) return VOCR_E_ALIGN;
    if (*(const unsigned char *)&one != 1) return VOCR_E_FORMAT; /* the blob is little-endian */
    if (bytes < TABLE || memcmp(b, "VOLM", 4) || rd32(b + 4) != VOCR_LM_VERSION) return VOCR_E_FORMAT;
    total = rd32(b + 8);
    order = rd32(b + 12);
    V = rd32(b + 16);
    voc = rd32(b + 28);
    if (total < TABLE || total > bytes || order < 2 || order > VOCR_LM_MAX_ORDER || V <= FIRST ||
        V > 65535u || rd32(b + 20) || !in_blob(total, voc, 4u * (V - FIRST)))
        return VOCR_E_FORMAT;
    lm->blob = b;
    lm->bytes = total;
    lm->order = order;
    lm->n_tokens = V;
    lm->cps = (const uint32_t *)(const void *)(b + voc);
    for (e = 1; e < V - FIRST; e++)
        if (lm->cps[e] <= lm->cps[e - 1]) return VOCR_E_FORMAT;
    for (k = 0; k < VOCR_LM_MAX_ORDER; k++) {
        const unsigned char *r = b + HEADER + k * ROW;
        vocr_lm_order *o = &lm->o[k];
        uint32_t n = rd32(r), ot = rd32(r + 4), op = rd32(r + 8), ob = rd32(r + 12), oc = rd32(r + 16),
                 ocp = rd32(r + 20), ocb = rd32(r + 24);
        if (rd32(r + 28)) return VOCR_E_FORMAT;
        if (k >= order) {
            if (n || ot || op || ob || oc || ocp || ocb) return VOCR_E_FORMAT;
            continue;
        }
        if ((k == 0 && (n != V || ot)) || n > 0x7FFFFFF0u) return VOCR_E_FORMAT;
        if (k > 0) {
            if (!in_blob(total, ot, 2u * n)) return VOCR_E_FORMAT;
            o->tok = (const uint16_t *)(const void *)(b + ot);
        }
        if (!in_blob(total, op, n) || !in_blob(total, ocp, 512u)) return VOCR_E_FORMAT;
        o->n = n;
        o->prob = b + op;
        o->cb_prob = (const int16_t *)(const void *)(b + ocp);
        if (k + 1 < order) {
            if (n >= 0x3FFFFFFFu || !in_blob(total, ob, n) || !in_blob(total, oc, 4u * (n + 1u)) ||
                !in_blob(total, ocb, 512u))
                return VOCR_E_FORMAT;
            o->bo = b + ob;
            o->child = (const uint32_t *)(const void *)(b + oc);
            o->cb_bo = (const int16_t *)(const void *)(b + ocb);
            if (o->cb_bo[0] != 0) return VOCR_E_FORMAT;
        } else if (ob || oc || ocb) {
            return VOCR_E_FORMAT;
        }
    }
    /* child ranges: non-decreasing, from 0 to the next order's entries; every token below V */
    for (k = 0; k + 1 < order; k++) {
        const vocr_lm_order *o = &lm->o[k];
        if (o->child[0] != 0 || o->child[o->n] != lm->o[k + 1].n) return VOCR_E_FORMAT;
        for (e = 0; e < o->n; e++)
            if (o->child[e + 1] < o->child[e]) return VOCR_E_FORMAT;
    }
    for (k = 1; k < order; k++)
        for (e = 0; e < lm->o[k].n; e++)
            if (lm->o[k].tok[e] >= V) return VOCR_E_FORMAT;
    return VOCR_OK;
}

unsigned vocr_lm_token(const vocr_lm *lm, uint32_t cp)
{
    uint32_t lo = 0, hi;
    if (!lm || !lm->cps) return 0;
    hi = lm->n_tokens - FIRST;
    while (lo < hi) {
        const uint32_t mid = (lo + hi) >> 1;
        if (lm->cps[mid] < cp) lo = mid + 1;
        else if (lm->cps[mid] > cp) hi = mid;
        else return FIRST + mid;
    }
    return 0;
}

void vocr_lm_initial(const vocr_lm *lm, unsigned flags, int32_t *state)
{
    unsigned k;
    for (k = 0; k + 1 < lm->order; k++) state[k] = NONE;
    if (flags & VOCR_LM_BOS) state[0] = (int32_t)BOS;
}

/* order-(k+1) entry under order-k entry `parent` with last token t, or NONE */
static int32_t lm_find(const vocr_lm *lm, unsigned k, int32_t parent, unsigned t)
{
    const uint16_t *tok = lm->o[k].tok;
    uint32_t lo = lm->o[k - 1].child[parent], hi = lm->o[k - 1].child[parent + 1];
    while (lo < hi) {
        const uint32_t mid = (lo + hi) >> 1;
        const unsigned v = tok[mid];
        if (v < t) lo = mid + 1;
        else if (v > t) hi = mid;
        else return (int32_t)mid;
    }
    return NONE;
}

int32_t vocr_lm_score(const vocr_lm *lm, const int32_t *state, unsigned t, int32_t *next)
{
    const unsigned N = lm->order;
    int32_t e[VOCR_LM_MAX_ORDER], lp = 0;
    unsigned k;
    for (k = 1; k < N; k++) e[k] = state[k - 1] != NONE ? lm_find(lm, k, state[k - 1], t) : NONE;
    for (k = N - 1; k > 0; k--) {
        if (e[k] != NONE) {
            lp += lm->o[k].cb_prob[lm->o[k].prob[e[k]]];
            break;
        }
        if (state[k - 1] != NONE) lp += lm->o[k - 1].cb_bo[lm->o[k - 1].bo[state[k - 1]]];
    }
    if (!k) lp += lm->o[0].cb_prob[lm->o[0].prob[t]];
    if (next) {
        next[0] = (int32_t)t;
        for (k = 1; k + 1 < N; k++) next[k] = e[k];
    }
    return lp;
}

/* ---------------------------------------------------------------- decoder plan */

static int32_t milli_q10(int x) /* 1/1000 -> Q10, half away from zero */
{
    const int32_t a = x < 0 ? -(int32_t)x : (int32_t)x, q = (a * 1024 + 500) / 1000;
    return x < 0 ? -q : q;
}

static int plan(vocr_lm_ctx *c, const vocr_model *m, const vocr_lm *lm, const vocr_lm_opts *o)
{
    size_t off = 0, nn, nw;
    if (!c || !m || !o) return VOCR_E_ARG;
    memset(c, 0, sizeof *c);
    if (o->beam < 1 || o->beam > VOCR_LM_MAX_BEAM || o->topk < 1 || o->topk > VOCR_TOPK_MAX ||
        o->alpha < 0 || o->alpha > VOCR_LM_MAX_ALPHA || o->beta < -VOCR_LM_MAX_BETA ||
        o->beta > VOCR_LM_MAX_BETA || o->gate < 0 || o->gate > VOCR_LM_MAX_GATE || o->bracket < 0 ||
        o->bracket > VOCR_LM_MAX_BRACKET ||
        (o->flags & ~(unsigned)(VOCR_LM_BOS | VOCR_LM_EOS)) || (o->alpha && !lm) || !m->stride)
        return VOCR_E_ARG;
    c->model = m;
    c->lm = o->alpha ? lm : NULL;
    c->beam = o->beam;
    c->topk = o->topk;
    c->max_frames = o->max_frames ? o->max_frames : m->max_width / m->stride;
    if (!c->max_frames || c->max_frames > MAX_FRAMES) return VOCR_E_ARG;
    c->n_state = c->lm ? c->lm->order - 1 : 0;
    c->flags = o->flags;
    c->alpha = milli_q10(o->alpha);
    c->beta = milli_q10(o->beta);
    c->gate = milli_q10(o->gate);
    c->bracket = milli_q10(o->bracket);
    nn = 1u + (size_t)c->max_frames * c->beam;
    nw = (size_t)c->beam * c->topk;
    c->max_nodes = (uint32_t)nn;
    c->max_new = (uint32_t)nw;
    c->cmap_off = (uint32_t)off;
    off = al16(off + (c->lm ? 2u * (size_t)m->n_classes : 0u));
    c->role_off = (uint32_t)off;
    off = al16(off + (c->bracket ? (size_t)m->n_classes : 0u));
    c->par_off = (uint32_t)off;
    off = al16(off + 4u * nn);
    c->cls_off = (uint32_t)off;
    off = al16(off + 2u * nn);
    c->len_off = (uint32_t)off;
    off = al16(off + 2u * nn);
    c->hash_off = (uint32_t)off;
    off = al16(off + 4u * nn);
    c->brk_off = (uint32_t)off;
    off = al16(off + (c->bracket ? 2u * nn : 0u));
    c->st_off = (uint32_t)off;
    off = al16(off + 4u * nn * c->n_state);
    c->hyp_off = (uint32_t)off;
    off = al16(off + 12u * c->beam);
    c->nhyp_off = (uint32_t)off;
    off = al16(off + 12u * c->beam);
    c->stay_off = (uint32_t)off;
    off = al16(off + 8u * c->beam);
    c->new_off = (uint32_t)off;
    off = al16(off + 16u * nw);
    c->newst_off = (uint32_t)off;
    off = al16(off + 4u * nw * c->n_state);
    c->rank_off = (uint32_t)off;
    off = al16(off + 8u * c->beam);
    c->out_off = (uint32_t)off;
    off = al16(off + 2u * (size_t)c->max_frames);
    if (off > 0x7FFFFFF0u) return VOCR_E_ARG;
    c->ws_bytes = off;
    return VOCR_OK;
}

size_t vocr_lm_workspace_bytes(const vocr_model *m, const vocr_lm *lm, const vocr_lm_opts *o)
{
    vocr_lm_ctx c;
    return plan(&c, m, lm, o) == VOCR_OK ? c.ws_bytes : 0;
}

/* the one code point of a class's UTF-8 text, or -1 (empty, several, or not valid UTF-8) */
static long one_code_point(const unsigned char *s, unsigned n)
{
    unsigned need, i;
    uint32_t cp;
    if (!n) return -1;
    if (s[0] < 0x80) cp = s[0], need = 0;
    else if ((s[0] & 0xE0) == 0xC0) cp = s[0] & 0x1Fu, need = 1;
    else if ((s[0] & 0xF0) == 0xE0) cp = s[0] & 0x0Fu, need = 2;
    else if ((s[0] & 0xF8) == 0xF0) cp = s[0] & 0x07u, need = 3;
    else return -1;
    if (n != need + 1) return -1;
    for (i = 1; i <= need; i++) {
        if ((s[i] & 0xC0) != 0x80) return -1;
        cp = cp << 6 | (s[i] & 0x3Fu);
    }
    if ((need == 1 && cp < 0x80) || (need == 2 && cp < 0x800) || (need == 3 && cp < 0x10000) ||
        cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF))
        return -1;
    return (long)cp;
}

/* Bracket pairs: role of a code point, 0 none, k (1..8) the opener of pair k, 0x80 | k its closer */
static unsigned bracket_role(long cp)
{
    static const uint16_t open[8] = {0x300C, 0x300E, 0xFF08, 0x3010, 0x3008, 0x300A, 0x3014, 0xFF3B};
    static const uint16_t close[8] = {0x300D, 0x300F, 0xFF09, 0x3011, 0x3009, 0x300B, 0x3015, 0xFF3D};
    unsigned k;
    if (cp == '(') return 3;
    if (cp == ')') return 0x80u | 3u;
    for (k = 0; k < 8; k++) {
        if (cp == open[k]) return k + 1;
        if (cp == close[k]) return 0x80u | (k + 1);
    }
    return 0;
}

/* The bracket state after a character of role r: a stack of the last two unclosed openers (low byte the
 * innermost); *mismatch is set when r closes a pair other than the innermost open one. */
static unsigned bracket_next(unsigned st, unsigned r, int *mismatch)
{
    const unsigned s0 = st & 0xFFu, s1 = st >> 8;
    *mismatch = 0;
    if (!r) return st;
    if (!(r & 0x80u)) return r | s0 << 8;
    r &= 0x7Fu;
    if (s0 == r) return s1;
    if (!s0) return st;
    *mismatch = 1;
    return s1 == r ? 0u : st;
}

int vocr_lm_init(vocr_lm_ctx *c, const vocr_model *m, const vocr_lm *lm, const vocr_lm_opts *o, void *ws,
                 size_t bytes)
{
    int rc;
    unsigned i;
    if (!c || !ws) return VOCR_E_ARG;
    if ((uintptr_t)ws & 15u) return VOCR_E_ALIGN;
    rc = plan(c, m, lm, o);
    if (rc) return rc;
    if (bytes < c->ws_bytes) return VOCR_E_ARENA;
    c->ws = (unsigned char *)ws;
    if (c->bracket) {
        uint8_t *role = c->ws + c->role_off;
        role[0] = 0;
        for (i = 1; i < m->n_classes; i++) {
            unsigned n;
            const char *s = vocr_class_text(m, i, &n);
            role[i] = (uint8_t)bracket_role(one_code_point((const unsigned char *)s, n));
        }
    }
    if (c->lm) {
        /* every bracket pair reads as 「」 to the LM: two or three characters of context cannot tell a
         * pair's style, so the picture decides it (and the bracket state keeps a pair consistent) */
        uint16_t *cmap = (uint16_t *)(void *)(c->ws + c->cmap_off);
        cmap[0] = 0;
        for (i = 1; i < m->n_classes; i++) {
            unsigned n, r;
            const char *s = vocr_class_text(m, i, &n);
            long cp = one_code_point((const unsigned char *)s, n);
            r = bracket_role(cp);
            if (r) cp = r & 0x80u ? 0x300D : 0x300C;
            cmap[i] = (uint16_t)(cp < 0 ? 0u : vocr_lm_token(c->lm, (uint32_t)cp));
        }
    }
    return VOCR_OK;
}

/* ---------------------------------------------------------------- decoder */

/* log P in Q10 clamped to [LP_FLOOR, 0]: a probability's log is never positive (quantisation can make the
 * sum of a backoff and a probability code slightly so) */
static int32_t lp_clamp(int32_t lp) { return lp > 0 ? 0 : lp < LP_FLOOR ? LP_FLOOR : lp; }

/* minus the logit margin to the frame's top-1, Q10, rounded half up, clamped to 64 nats, at most -1 */
static int32_t cand_score(float top, float v)
{
    float d = top - v;
    int32_t q;
    if (!(d >= 0.0f)) d = 0.0f;
    else if (d > 64.0f) d = 64.0f;
    q = (int32_t)(d * 1024.0f + 0.5f);
    return q < 1 ? -1 : -q;
}

static uint32_t prefix_hash(uint32_t h, unsigned c) { return h * 0x9E3779B1u + c + 1u; }

typedef struct {
    int32_t *par;
    uint16_t *cls, *len, *brk;
    uint32_t *hash;
    int32_t *st;
    unsigned ns;
} nodes_t;

static int same_prefix(const nodes_t *nd, int32_t x, int32_t y)
{
    while (x != y) {
        if (nd->cls[x] != nd->cls[y]) return 0;
        x = nd->par[x];
        y = nd->par[y];
    }
    return 1;
}

int vocr_lm_decode(vocr_lm_ctx *c, const vocr_extra *ex, char *text, size_t text_cap, unsigned *chars)
{
    const vocr_model *m;
    const vocr_lm *lm;
    unsigned char *ws;
    nodes_t nd;
    int32_t *hyp, *nhyp, *stay, *nw, *nwst, *rank;
    uint16_t *cmap, *outs;
    const uint8_t *role;
    unsigned K, T, t, i, j, nh, nn, nsel, n_out, cc[VOCR_TOPK_MAX + 1];
    int32_t cq[VOCR_TOPK_MAX + 1], tmp[VOCR_LM_MAX_ORDER];
    size_t used = 0;
    int32_t best, best_s = 0;
    if (chars) *chars = 0;
    if (!c || !c->ws || !ex || !text || !text_cap) return VOCR_E_ARG;
    text[0] = 0;
    if (!ex->topk || ex->topk > VOCR_TOPK_MAX || !ex->topk_logit || !ex->topk_class) return VOCR_E_ARG;
    T = ex->frames;
    if (T > c->max_frames || T > ex->cap_frames) return VOCR_E_WIDTH;
    m = c->model;
    lm = c->lm;
    ws = c->ws;
    K = c->topk < ex->topk ? c->topk : ex->topk;
    cmap = (uint16_t *)(void *)(ws + c->cmap_off);
    role = c->bracket ? ws + c->role_off : NULL;
    nd.brk = (uint16_t *)(void *)(ws + c->brk_off);
    nd.par = (int32_t *)(void *)(ws + c->par_off);
    nd.cls = (uint16_t *)(void *)(ws + c->cls_off);
    nd.len = (uint16_t *)(void *)(ws + c->len_off);
    nd.hash = (uint32_t *)(void *)(ws + c->hash_off);
    nd.st = (int32_t *)(void *)(ws + c->st_off);
    nd.ns = c->n_state;
    hyp = (int32_t *)(void *)(ws + c->hyp_off);
    nhyp = (int32_t *)(void *)(ws + c->nhyp_off);
    stay = (int32_t *)(void *)(ws + c->stay_off);
    nw = (int32_t *)(void *)(ws + c->new_off);
    nwst = (int32_t *)(void *)(ws + c->newst_off);
    rank = (int32_t *)(void *)(ws + c->rank_off);
    outs = (uint16_t *)(void *)(ws + c->out_off);
    nd.par[0] = NONE;
    nd.cls[0] = 0;
    nd.len[0] = 0;
    nd.hash[0] = 0;
    if (role) nd.brk[0] = 0;
    if (lm) vocr_lm_initial(lm, c->flags, nd.st);
    nn = 1;
    hyp[0] = 0;
    hyp[1] = 0;
    hyp[2] = NEG;
    nh = 1;
    for (t = 0; t < T; t++) {
        const float *tv = ex->topk_logit + (size_t)t * ex->topk;
        const uint16_t *tc = ex->topk_class + (size_t)t * ex->topk;
        unsigned nc = 1, n_new = 0, has_blank;
        int32_t *sw;
        if (!(t % CANCEL_EVERY) && ex->cancel && ex->cancel(ex->user)) {
            text[0] = 0;
            return VOCR_E_CANCELLED;
        }
        /* candidates: the top-1 at 0, then the top-K classes and the blank within the gate */
        for (j = 0; j < K; j++)
            if (tc[j] >= m->n_classes) { /* not from this model's vocr_run() */
                text[0] = 0;
                return VOCR_E_ARG;
            }
        cc[0] = tc[0];
        cq[0] = 0;
        has_blank = tc[0] == 0;
        for (j = 1; j < K; j++) {
            const int32_t s = cand_score(tv[0], tv[j]);
            if (-s >= c->gate) break;
            cc[nc] = tc[j];
            cq[nc++] = s;
            if (!tc[j]) has_blank = 1;
        }
        if (ex->blank_logit && !has_blank) {
            const int32_t s = cand_score(tv[0], ex->blank_logit[t]);
            if (-s < c->gate) {
                cc[nc] = 0;
                cq[nc++] = s;
            }
        }
        for (i = 0; i < nh; i++) stay[2 * i] = stay[2 * i + 1] = NEG;
        for (i = 0; i < nh; i++) {
            const int32_t n = hyp[3 * i], sb = hyp[3 * i + 1], snb = hyp[3 * i + 2], s = imax(sb, snb);
            const unsigned last = n ? nd.cls[n] : 0u;
            for (j = 0; j < nc; j++) {
                const unsigned cl = cc[j];
                const int32_t q = cq[j];
                int32_t src, lmq = 0, val;
                unsigned h, mj, bst = 0;
                int mis = 0;
                const int32_t *nst = NULL;
                if (!cl) {
                    stay[2 * i] = imax(stay[2 * i], s + q);
                    continue;
                }
                if (cl == last) {
                    if (snb > NEG) stay[2 * i + 1] = imax(stay[2 * i + 1], snb + q);
                    src = sb;
                } else {
                    src = s;
                }
                if (src <= NEG) continue;
                h = prefix_hash(nd.hash[n], cl);
                for (mj = 0; mj < nh; mj++) {
                    const int32_t mm = hyp[3 * mj];
                    if (mm && nd.cls[mm] == cl && (unsigned)nd.len[mm] == nd.len[n] + 1u && nd.hash[mm] == h &&
                        same_prefix(&nd, nd.par[mm], n))
                        break;
                }
                if (lm) {
                    const int32_t lp = lp_clamp(vocr_lm_score(lm, nd.st + (size_t)n * nd.ns, cmap[cl], tmp));
                    lmq = c->alpha * lp / 1024;
                    nst = tmp;
                }
                val = src + q + lmq + c->beta;
                if (role) {
                    bst = bracket_next(nd.brk[n], role[cl], &mis);
                    if (mis) val -= c->bracket;
                }
                if (mj < nh) {
                    stay[2 * mj + 1] = imax(stay[2 * mj + 1], val);
                } else {
                    int32_t *e = nw + 4 * n_new;
                    e[0] = n;
                    e[1] = (int32_t)cl;
                    e[2] = val;
                    e[3] = (int32_t)bst;
                    if (nst) memcpy(nwst + (size_t)n_new * nd.ns, nst, nd.ns * sizeof(int32_t));
                    n_new++;
                }
            }
        }
        /* the beam best entries by score, ties to the lower index (old prefixes, then the new ones) */
        nsel = 0;
        for (i = 0; i < nh + n_new; i++) {
            const int32_t s = i < nh ? imax(stay[2 * i], stay[2 * i + 1]) : nw[4 * (i - nh) + 2];
            unsigned p;
            if (s <= NEG || (nsel == c->beam && s <= rank[2 * (nsel - 1)])) continue;
            p = nsel < c->beam ? nsel++ : nsel - 1;
            while (p && s > rank[2 * (p - 1)]) {
                rank[2 * p] = rank[2 * (p - 1)];
                rank[2 * p + 1] = rank[2 * (p - 1) + 1];
                p--;
            }
            rank[2 * p] = s;
            rank[2 * p + 1] = (int32_t)i;
        }
        for (j = 0; j < nsel; j++) {
            const unsigned idx = (unsigned)rank[2 * j + 1];
            if (idx < nh) {
                nhyp[3 * j] = hyp[3 * idx];
                nhyp[3 * j + 1] = stay[2 * idx];
                nhyp[3 * j + 2] = stay[2 * idx + 1];
            } else {
                const int32_t *e = nw + 4 * (idx - nh), p = e[0];
                nd.par[nn] = p;
                nd.cls[nn] = (uint16_t)e[1];
                if (role) nd.brk[nn] = (uint16_t)e[3];
                nd.len[nn] = (uint16_t)(nd.len[p] + 1u);
                nd.hash[nn] = prefix_hash(nd.hash[p], (unsigned)e[1]);
                if (nd.ns)
                    memcpy(nd.st + (size_t)nn * nd.ns, nwst + (size_t)(idx - nh) * nd.ns, nd.ns * sizeof(int32_t));
                nhyp[3 * j] = (int32_t)nn++;
                nhyp[3 * j + 1] = NEG;
                nhyp[3 * j + 2] = e[2];
            }
        }
        sw = hyp;
        hyp = nhyp;
        nhyp = sw;
        nh = nsel;
    }
    best = hyp[0];
    for (i = 0; i < nh; i++) {
        int32_t s = imax(hyp[3 * i + 1], hyp[3 * i + 2]);
        if (lm && (c->flags & VOCR_LM_EOS))
            s += c->alpha * lp_clamp(vocr_lm_score(lm, nd.st + (size_t)hyp[3 * i] * nd.ns, EOS, NULL)) / 1024;
        if (!i || s > best_s) {
            best = hyp[3 * i];
            best_s = s;
        }
    }
    n_out = 0;
    while (best) {
        outs[n_out++] = nd.cls[best];
        best = nd.par[best];
    }
    while (n_out) {
        unsigned n;
        const char *s = vocr_class_text(m, outs[--n_out], &n);
        if (used + n >= text_cap) {
            text[used] = 0;
            return VOCR_E_TEXT;
        }
        memcpy(text + used, s, n);
        used += n;
        text[used] = 0;
        if (chars) ++*chars;
    }
    return VOCR_OK;
}

void vocr_lm_sizes(size_t *lm_bytes, size_t *ctx_bytes)
{
    if (lm_bytes) *lm_bytes = sizeof(vocr_lm);
    if (ctx_bytes) *ctx_bytes = sizeof(vocr_lm_ctx);
}
