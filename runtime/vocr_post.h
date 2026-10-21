/* vocr_post.h - post-processing of recognized text for counters: 九ヶ月, 三ヶ所, 五ヶ年 (issue #109).
 *
 * Plain C99 like vocr.h: no malloc, no libm, no system calls, no global state; the re-decode's stack frame is
 * about 1 KiB (896 B on the Vita at -O2, 1,024 B with x86-64 gcc -O2). Each step runs only when its caller calls
 * it, so a pipeline that does not call one is bit for bit what it was. eval/vocr_post.py is the Python twin of both (tests/post_parity.py checks them against each
 * other), and docs/counters.md has the corpus counts behind every context.
 *
 * 1. vocr_post_counters(text): a text rule for the small ke ヶ (U+30F6) the recognizer reads as the small
 *    tsu ッ (U+30C3). In place; the text never grows (both are 3 bytes of UTF-8). ッ becomes ヶ
 *      - before 月 or 年, whatever stands on its left (ッ月 and ッ年 occur nowhere in the 700 M characters
 *        of training-side text counted, against 14,883 and 1,522 counter forms),
 *      - before 国 or 条 when its left character is a numeral (〇一二三四五六七八九十百千万, 0-9, ０-９) or
 *        何 数 幾 (ッ国 and ッ条 occur only as えッ国上山, 為ッ国, 別ッ条: valid text with other lefts),
 *      - before 所 when its left character is one of those other than 一 (Aozora's 一ッ所, ひとつところ,
 *        is valid text and stays).
 *    Nothing else changes: ッ before 日, 村, 町 and place names (三ッ日月, 四ッ谷, イマドキッ日曜日) stays, and so
 *    do ヵ, ケ, 箇, か and カ, which are valid counters. A line break is never a numeral, so no context
 *    reaches across one.
 *
 * 2. vocr_post_redecode(m, ex, o, st): a counter-context re-decode over the per-frame top-K of one
 *    vocr_run() (vocr_extra.topk >= 1). It reads the extras' greedy path (each frame's top-1, CTC collapsed)
 *    and, at every X + (ッ | ヶ | ヵ) + (月 | 所 | 年 | 国 | 条) where X is a letter (kana, kanji, Latin) and not
 *    a numeral or 何 数 幾, scores X's frames again with the classes restricted to those: the cost of a
 *    numeral d is the greedy path's score over X's frames minus the best path's that reads d there instead
 *    (one run of frames holding d, the blank on the others), in Q10 nats from the logits as vocr_lm.c rounds
 *    them. The cheapest d is accepted when its cost is at most o->tau. When X is the line's first character
 *    and the line's first bracket is a closing 」 or 』, the matching opening bracket is also scored at the
 *    blank frames before X, and accepted when the cheapest frame's cost is at most o->open.
 *    An accepted change is written into ex itself: the chosen class trades places with the top-1 class in
 *    that frame's top-K (classes only; the logits stay where they are, so every list stays sorted). So
 *    vocr_post_greedy(m, ex) gives the edited greedy text, and vocr_lm_decode() reads the edited extras as
 *    the frames' top-1. o->tau = o->open = 0 changes nothing (the extras are only checked). The blank counts
 *    where it is in a frame's top-K: the re-decode does not read vocr_extra.blank_logit, and an edit does not
 *    change it. vocr_lm_decode() reads blank_logit only for a frame whose blank is outside its candidates:
 *    the top-K slots it scans, min(its topk, ex->topk), whose cost is below its gate. An edit moves a frame's
 *    old top-1 (the blank, say) to the slot of the class it took, at a cost of at most tau (or open). So keep
 *    tau and open strictly below the LM's gate, and the LM's topk at least ex->topk: then that blank stays a
 *    candidate and blank_logit (still the old top-1's logit) is never read for it. vita-vn-ocr's choice, as
 *    Rubi runs it: tau 6 and open 4 nats against a gate of 8, both topk 8.
 *
 * In a pipeline:
 *    vocr_run(&ctx, &img, text, cap, &ex);               // ex.topk >= 1
 *    if (vocr_post_redecode(&model, &ex, &po, NULL) > 0)
 *        vocr_post_greedy(&model, &ex, text, cap, NULL); // or vocr_lm_decode(&dec, &ex, text, cap, NULL)
 *    vocr_post_counters(text);
 */
#ifndef VOCR_POST_H
#define VOCR_POST_H

#include "vocr.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VOCR_POST_MAX_TAU 64000      /* 1/1000 nat */
#define VOCR_POST_MAX_FRAMES 1024u   /* frames vocr_post_redecode reads (vocr_run's: at most 1,024 / stride + 1) */

typedef struct {
    int tau;    /* the numeral's threshold, 1/1000 nat, 0..VOCR_POST_MAX_TAU (0: X is never replaced) */
    int open;   /* the opening bracket's threshold, 1/1000 nat, 0..VOCR_POST_MAX_TAU (0: never inserted) */
} vocr_post_opts;

typedef struct {
    unsigned contexts;   /* counter contexts with a letter X that is not a numeral */
    unsigned numerals;   /* of them, X replaced by a numeral */
    unsigned brackets;   /* opening brackets inserted */
} vocr_post_stats;

/* The text rule (1) on the NUL-terminated UTF-8 text, in place. Returns the number of ッ made ヶ.
 * Invalid UTF-8 is kept as it is; NULL is 0. */
unsigned vocr_post_counters(char *text);

/* The re-decode (2). Returns the number of changes written into ex (0: ex is untouched), or VOCR_E_ARG
 * (no model, no top-K, a top-K class outside the model, o's thresholds out of range) or VOCR_E_WIDTH
 * (ex->frames above ex->cap_frames or VOCR_POST_MAX_FRAMES). st (may be NULL) gets this line's counts. */
int vocr_post_redecode(const vocr_model *m, vocr_extra *ex, const vocr_post_opts *o, vocr_post_stats *st);

/* The greedy text of ex's top-1 classes (CTC: repeats merged, blanks dropped), as vocr_run() writes it for
 * the extras it filled: NUL-terminated UTF-8, VOCR_E_TEXT when it does not fit (whole characters only).
 * *chars (may be NULL) gets its characters. */
int vocr_post_greedy(const vocr_model *m, const vocr_extra *ex, char *text, size_t text_cap, unsigned *chars);

#ifdef __cplusplus
}
#endif
#endif
