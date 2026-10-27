/* textspot7: a per-region fallback from a text spotter's lines to the GRAD finder's (F) and a post-pass on the
 * spotter's boxes (D) (issue #128, docs/textspot7.md, eval/textspot7_protocol.json). synth/textspot7_twin.py is the
 * specification and the bit-exact Python twin of both.
 *
 * F, vocr_textspot7_choose: given the spotter's horizontal boxes of a region (with their confidences, 0..127) and
 * the GRAD finder's boxes of the same region, it says which set to show, never a mix:
 *   - mode S (the spotter by default): GRAD's when the spotter found nothing and GRAD found a line, when the median
 *     spotter confidence is under tau, when GRAD has a line the spotter's boxes do not cover (S_lacks, lam per
 *     mille of the spotter's median line height) or a line that reaches beyond the spotter's boxes of its row by
 *     more than kappa per mille of its height at either end (S_short); else the spotter's;
 *   - mode G (GRAD by default): the spotter's when GRAD found nothing and the spotter found a line, or when one
 *     GRAD box holds the centres of spotter boxes of two or more rows (GRAD merged lines) and the median spotter
 *     confidence is at least tau; else GRAD's.
 * D, vocr_textspot7_post: the spotter's whole-frame boxes into one region's line boxes: the region's boxes (the
 * box's centre inside it, or with VOCR_T7_ASSIGN a large enough part inside it), clipped; then optionally boxes of
 * one row joined (VOCR_T7_JOIN5 / _JOIN10: a gap of at most 0.5 / 1.0 line heights), boxes grown over the ink at
 * their ends (VOCR_T7_EXTEND), and their rows cut to the text band (VOCR_T7_TIGHTEN: vocr_lines_refine.c's
 * VOCR_REFINE_TIGHTEN).
 *
 * C99, no allocation, no recursion, no global state, integer arithmetic only; link vocr_lines.c and
 * vocr_lines_refine.c (tighten). A region image for vocr_textspot7_post is grey (VOCR_LINES_GREY).
 */
#ifndef VOCR_TEXTSPOT7_H
#define VOCR_TEXTSPOT7_H

#include <stddef.h>

#include "vocr_lines.h"
#include "vocr_lines_rows.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VOCR_T7_MAX_BOXES 64u   /* boxes per call, either set */

#define VOCR_T7_OK 0
#define VOCR_T7_E_ARG (-1)      /* null pointer, too many boxes, unknown mode or step, a bad region or image */
#define VOCR_T7_E_ARENA (-2)    /* arena smaller than vocr_textspot7_arena_bytes(w, h) */

/* vocr_textspot7_choose's modes and answers */
#define VOCR_T7_MODE_S 0u
#define VOCR_T7_MODE_G 1u
#define VOCR_T7_SHOW_SPOTTER 0
#define VOCR_T7_SHOW_GRAD 1

/* vocr_textspot7_post's steps, run in this order (JOIN10 wins over JOIN5) */
#define VOCR_T7_ASSIGN 0x01u
#define VOCR_T7_JOIN5 0x02u
#define VOCR_T7_JOIN10 0x04u
#define VOCR_T7_EXTEND 0x08u
#define VOCR_T7_TIGHTEN 0x10u
#define VOCR_T7_STEPS 0x1Fu

/* A box: x, y, w, h in the region's (choose, post's output) or the frame's (post's input) pixels; conf the
 * spotter's box confidence (0..127; unused for GRAD's boxes); dark 1 when the text is darker than its background. */
typedef struct {
    int x, y, w, h;
    int conf;
    unsigned dark;
} VocrT7Box;

/* F's parameters: lam and kappa in per mille of a line height, 0 = that test off. */
typedef struct {
    unsigned mode;   /* VOCR_T7_MODE_S or VOCR_T7_MODE_G */
    int tau;         /* the median spotter confidence that counts as sure */
    int lam;         /* S_lacks: a GRAD box at least lam / 1000 of the spotter's median height tall counts */
    int kappa;       /* S_short: the margin beyond the spotter's boxes, in per mille of the GRAD box's height */
} VocrT7Params;

/* VOCR_T7_SHOW_SPOTTER or VOCR_T7_SHOW_GRAD (s: ns spotter boxes, g: ng GRAD boxes, both in region pixels), or
 * VOCR_T7_E_ARG. */
int vocr_textspot7_choose(const VocrT7Box *s, unsigned ns, const VocrT7Box *g, unsigned ng, const VocrT7Params *p);

/* Arena bytes vocr_textspot7_post needs for a region up to max_width x max_height with EXTEND or TIGHTEN (the
 * larger of the refinement's arena and EXTEND's two row buffers); without either step no arena is read. */
size_t vocr_textspot7_arena_bytes(unsigned max_width, unsigned max_height);

/* The same for one step set: TIGHTEN needs the refinement's arena, EXTEND 2 (max_width + 2) bytes, the other steps
 * none (textspot7's pick, JOIN10 | EXTEND: 1,924 B at 960 px). */
size_t vocr_textspot7_arena_bytes_steps(unsigned max_width, unsigned max_height, unsigned steps);

/* The region (x0, y0) - (x1, y1) (frame pixels, x1 > x0, y1 > y0) of the frame boxes fb[0..n) as line boxes in
 * region pixels, out[0..*n_out) (out holds VOCR_T7_MAX_BOXES). region: its pixels, x1 - x0 by y1 - y0, grey (read
 * by EXTEND and TIGHTEN only; may be NULL without them). Returns VOCR_T7_OK, VOCR_T7_E_ARG or VOCR_T7_E_ARENA. */
int vocr_textspot7_post(const VocrT7Box *fb, unsigned n, int x0, int y0, int x1, int y1,
                        const VocrLinesImage *region, unsigned steps, void *arena, size_t arena_bytes,
                        VocrT7Box *out, unsigned *n_out);

/* Arena bytes vocr_textspot7_post_rows needs for a region max_width wide (any height) with EXTEND or TIGHTEN: the
 * larger of the refinement's arena (vocr_refine_arena_bytes(max_width, max_height)) and EXTEND's three rows. */
size_t vocr_textspot7_rows_arena_bytes(unsigned max_width, unsigned max_height);

/* The same for one step set: TIGHTEN needs the refinement's arena, EXTEND 3 max_width bytes (2,880 B at 960 px). */
size_t vocr_textspot7_rows_arena_bytes_steps(unsigned max_width, unsigned max_height, unsigned steps);

/* vocr_textspot7_post for a region read row by row (vocr_lines_rows.h), as a caller that streams the frame (Rubi's
 * game worker) has it: the same boxes for the same pixels, or VOCR_LINES_E_SOURCE when a row could not be read.
 * EXTEND reads each box's rows (one above and one below included) twice, top to bottom; TIGHTEN is
 * vocr_refine_lines_rows. A box whose EXTEND scan would pass 1,024 columns on a side (a box of 684 rows or more, in a
 * region wider than 1,024 px, since the scan stops at the region's edge) is refused with VOCR_T7_E_ARG; no box of a
 * 960 x 544 frame comes near it. */
int vocr_textspot7_post_rows(const VocrT7Box *fb, unsigned n, int x0, int y0, int x1, int y1,
                             const VocrLinesRows *region, unsigned steps, void *arena, size_t arena_bytes,
                             VocrT7Box *out, unsigned *n_out);

#ifdef __cplusplus
}
#endif

#endif
