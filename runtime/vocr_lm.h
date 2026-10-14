/* vocr_lm.h - a character n-gram LM and CTC prefix beam search over vocr_run()'s per-frame top-K.
 *
 * Plain C99 like vocr.h: no malloc, no libm, no system calls, no global state. The LM blob (a VOLM file
 * written by lm/build.py) is used in place; the decoder's memory is one caller-provided workspace whose
 * exact size vocr_lm_workspace_bytes() reports. Every score is an int32 in Q10 nats (1/1024 nat), so the
 * result is the same on every platform, bit for bit, and equals lm/vlm.py's reference decoder.
 *
 *   vocr_lm lm;  vocr_lm_ctx dc;
 *   vocr_lm_opts lo = {8, 8, 0, 500, 3500, 8000, VOCR_LM_EOS, 8000};   // docs/ctc-lm-decode.md, per LM tier
 *   vocr_lm_load(&lm, lm_blob, lm_bytes);                          // blob 16-byte aligned, kept alive
 *   size_t ws_need = vocr_lm_workspace_bytes(&model, &lm, &lo);
 *   vocr_lm_init(&dc, &model, &lm, &lo, ws, ws_need);              // ws 16-byte aligned
 *   ...per line: ex.topk = 8 (>= lo.topk), ex.topk_logit/topk_class/blank_logit/cap_frames set...
 *   vocr_run(&ctx, &img, greedy, sizeof greedy, &ex);              // greedy text + top-K + confidence
 *   vocr_lm_decode(&dc, &ex, text, sizeof text, &chars);           // the beam + LM text
 *
 * The search is Viterbi (max over alignments, not sum): a prefix's score is its best path's
 *   sum over frames of (the frame's logit of the path's class - the frame's top-1 logit)
 *   + alpha * sum over its characters of log P_LM(character | the previous N-1)  +  beta * characters.
 * Logit differences are log-probability differences, so no softmax is needed. A frame's candidates are
 * its top-1 and every other top-K class (and the blank) within `gate` of it: a frame whose top-2 margin
 * is >= gate keeps only its top-1, so the LM cannot change a confidently read character (the guard for
 * names and rare words). Margins are rounded to Q10 and every non-top-1 candidate costs at least 1/1024
 * nat, so with alpha = beta = 0 the greedy path is the unique best path: the output is vocr_run()'s
 * greedy text, bit for bit, for every beam, topk and gate.
 *
 * Brackets: an n-gram LM sees two or three characters back, so it cannot pair a closing bracket with its
 * opener and pulls 』, 】 and ） towards the common 」. Each prefix therefore carries its last two unclosed
 * opening brackets (「『（【〈《〔［ and ASCII "("), and a closing bracket that does not match the innermost
 * open one costs `bracket`. A closer with nothing open on the line is free (its opener may be on an
 * earlier line). bracket = 0 turns this off; alpha = beta = bracket = 0 is the greedy text. The LM itself
 * reads every pair as 「」 (vocr_lm_init's class map), so the picture alone picks a pair's style.
 */
#ifndef VOCR_LM_H
#define VOCR_LM_H

#include "vocr.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VOCR_LM_VERSION 1
#define VOCR_LM_MAX_ORDER 6
#define VOCR_LM_MAX_BEAM 16
#define VOCR_LM_MAX_ALPHA 8000   /* 1/1000 */
#define VOCR_LM_MAX_BETA 8000    /* |beta|, 1/1000 nat */
#define VOCR_LM_MAX_GATE 64000   /* 1/1000 nat */
#define VOCR_LM_MAX_BRACKET 64000   /* 1/1000 nat */

/* vocr_lm_opts.flags */
enum {
    VOCR_LM_BOS = 1,   /* score a line's first characters after the LM's <s> (else: no context) */
    VOCR_LM_EOS = 2    /* add alpha * log P(</s> | the last characters) at the end of the line */
};

typedef struct {
    uint32_t n;                    /* entries (order 1: tokens) */
    const uint16_t *tok;           /* order >= 2: last token of each entry */
    const uint8_t *prob;           /* codes into cb_prob */
    const uint8_t *bo;             /* order < N: codes into cb_bo (0 = no backoff weight) */
    const uint32_t *child;         /* order < N: children of entry e are [child[e], child[e + 1]) */
    const int16_t *cb_prob, *cb_bo;    /* 256 Q10 log values each */
} vocr_lm_order;

typedef struct {               /* a parsed VOLM blob; points into it, so keep the blob alive */
    const unsigned char *blob;
    size_t bytes;
    unsigned order, n_tokens;
    const uint32_t *cps;           /* code points of tokens 3..n_tokens-1, increasing */
    vocr_lm_order o[VOCR_LM_MAX_ORDER];
} vocr_lm;

typedef struct {
    unsigned beam;         /* prefixes kept per frame, 1..VOCR_LM_MAX_BEAM */
    unsigned topk;         /* top-K classes considered per frame, 1..VOCR_TOPK_MAX; vocr_lm_decode()
                              uses min(topk, extra->topk) */
    unsigned max_frames;   /* longest line the workspace is sized for, <= 1024 (0: the model's max_width / stride) */
    int alpha;             /* LM weight, 1/1000, 0..VOCR_LM_MAX_ALPHA (0: no LM, needs no blob) */
    int beta;              /* per-character insertion bonus, 1/1000 nat */
    int gate;              /* confidence gate, 1/1000 nat (0: every frame keeps its top-1 only) */
    unsigned flags;        /* VOCR_LM_BOS, VOCR_LM_EOS */
    int bracket;           /* cost of a closing bracket that does not match the open one, 1/1000 nat (0: off) */
} vocr_lm_opts;

typedef struct {               /* a planned decoder; private */
    const vocr_model *model;
    const vocr_lm *lm;
    unsigned char *ws;
    size_t ws_bytes;
    unsigned beam, topk, max_frames, n_state, flags;
    int32_t alpha, beta, gate, bracket;    /* Q10 */
    uint32_t cmap_off, role_off, par_off, cls_off, len_off, hash_off, brk_off, st_off, hyp_off, nhyp_off,
             stay_off, new_off, newst_off, rank_off, out_off, max_nodes, max_new;
} vocr_lm_ctx;

/* Check a VOLM blob's CRC-32 (optional; vocr_lm_load checks structure, not content). */
int vocr_lm_check_crc(const void *blob, size_t bytes);
/* Parse and validate a VOLM blob already in memory (16-byte aligned). Nothing is copied. It checks what keeps
 * every later read inside the blob: the header, each section's bounds and alignment, the child ranges (non-
 * decreasing, ending at the next order's size) and every token id below n_tokens. It does not check the order
 * of the tokens within a range: a damaged blob can score wrongly but cannot make the decoder read outside it.
 * vocr_lm_check_crc() detects the damage. */
int vocr_lm_load(vocr_lm *lm, const void *blob, size_t bytes);
/* Exact workspace bytes for vocr_lm_init (0 if the options are invalid). lm may be NULL when alpha = 0. */
size_t vocr_lm_workspace_bytes(const vocr_model *m, const vocr_lm *lm, const vocr_lm_opts *o);
/* Plan the workspace and map the model's classes to LM tokens. ws: 16-byte aligned, >= the size above. */
int vocr_lm_init(vocr_lm_ctx *c, const vocr_model *m, const vocr_lm *lm, const vocr_lm_opts *o, void *ws,
                 size_t bytes);
/* Decode one line from the extras of a vocr_run() call with ex->topk >= 1: ex->frames, ex->topk,
 * ex->topk_logit, ex->topk_class and ex->blank_logit (NULL: the blank is a candidate only when it is in
 * the top-K). Polls ex->cancel(ex->user) every 8 frames (VOCR_E_CANCELLED). text gets NUL-terminated
 * UTF-8 (VOCR_E_TEXT if it does not fit: whole characters only); *chars (may be NULL) its characters.
 * VOCR_E_WIDTH if ex->frames is above the planned max_frames or above ex->cap_frames; VOCR_E_ARG if a top-K
 * class is not below the model's n_classes (extras from another model). */
int vocr_lm_decode(vocr_lm_ctx *c, const vocr_extra *ex, char *text, size_t text_cap, unsigned *chars);
/* LM token of one code point (0 = <unk>). */
unsigned vocr_lm_token(const vocr_lm *lm, uint32_t cp);
/* log P(token | state) in Q10. state: order - 1 entries from vocr_lm_initial() or a previous vocr_lm_score();
 * token < lm->n_tokens. Neither is checked (the decoder only passes valid ones). next (may be NULL): the state
 * after the token. */
int32_t vocr_lm_score(const vocr_lm *lm, const int32_t *state, unsigned token, int32_t *next);
/* The context state of a line's start (flags & VOCR_LM_BOS: after <s>). state: order - 1 entries. */
void vocr_lm_initial(const vocr_lm *lm, unsigned flags, int32_t *state);
/* sizeof(vocr_lm) and sizeof(vocr_lm_ctx), for foreign-function bindings. */
void vocr_lm_sizes(size_t *lm_bytes, size_t *ctx_bytes);

#ifdef __cplusplus
}
#endif
#endif
