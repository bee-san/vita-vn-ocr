/* vocr_tsd_post.h - textspot-desktop's box post-pass and reading filter (issue #173 round 2, protocol v2 amendment 4:
 * the C6 options), the C twins of tools/tsdesk.py's POST_DEFAULTS (post_pre, post_post) and DROP_DEFAULTS (drop_set).
 *
 * tools/tsdesk.py is the reference; this C99 file gives its boxes and its dropped readings bit for bit: the ink tests
 * are integer arithmetic on the grey frame, and the box arithmetic is the same double operations in the same order
 * (build with -ffp-contract=off, as desktop/Makefile does). No allocation: the caller passes a scratch buffer of
 * vocr_tsdp_scratch_bytes(w, h) bytes, reused for every box. Every option is an integer and 0 turns it off; with
 * every option 0 the boxes are untouched (the decode's and the refinement's bits).
 *
 *   before the refinement (vocr_tsdp_pre), in this order:
 *     merge_pct   two boxes of one orientation that overlap across by at least half the smaller size across and whose
 *                 intersection is at least merge_pct % of the smaller one's area become their union (tools/tsdesk.
 *                 merge_overlaps: the pair with the largest share first, the larger box's fields, its place the
 *                 earlier of the two);
 *     rows_pct    a horizontal box split across at an ink valley between two bands of similar height (tools/tsdesk.
 *                 split_rows, valley_pct and rows_len), at most two levels;
 *     gap_pct     the along split (tools/tsdesk.split_along): not in C (not adopted); a nonzero value is refused.
 *   after the refinement (vocr_tsdp_post):
 *     ext_pct     a horizontal box's ends extended over the solid dash run beyond them (tools/tsdesk.extend_ends,
 *                 ext_gap_pct, ext_run_pct, ext_dash_pct, ext_join).
 * At most `cap` boxes come out of vocr_tsdp_pre (tools/tsdesk.post_pre keeps the first MAX_BOXES).
 *
 * The reading filter (vocr_tsdd_drop: tools/tsdesk.drop_set) decides which of a page's read horizontal lines are left
 * out, from their tight boxes and UTF-8 readings: nojp (no Japanese character; 2: unless a mostly Japanese reading
 * sits in the row within nojp_gap_pct % of the larger height, nojp_anchor_pct), rep_pct (a reading dominated by one
 * non-Japanese character), conf_milli (the mean character confidence under conf_milli / 1000).
 */
#ifndef VOCR_TSD_POST_H
#define VOCR_TSD_POST_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The post-pass options, in tools/tsdesk.POST_DEFAULTS order. */
typedef struct {
    int merge_pct, rows_pct, valley_pct, rows_len, gap_pct, gap_weak;
    int ext_pct, ext_gap_pct, ext_run_pct, ext_dash_pct, ext_join;
} vocr_tsdp_opts;

#define VOCR_TSDP_NOPTS 11

/* The reading filter's options, in tools/tsdesk.DROP_DEFAULTS order. */
typedef struct {
    int nojp, nojp_gap_pct, nojp_anchor_pct, rep_pct, conf_milli;
} vocr_tsdd_opts;

#define VOCR_TSDD_NOPTS 5

enum { VOCR_TSDP_OK = 0, VOCR_TSDP_E_ARG = -1, VOCR_TSDP_E_SCRATCH = -2 };

/* A box as the post-pass sees it (vocr_tsd_box's fields that it reads and writes, frame px). */
typedef struct {
    double x, y, w, h;
    double logit, tconf;
    unsigned vertical, tag, cells;
} vocr_tsdp_box;

/* The defaults: tools/tsdesk.POST_DEFAULTS and DROP_DEFAULTS (every option off). */
void vocr_tsdp_defaults(vocr_tsdp_opts *o);
void vocr_tsdd_defaults(vocr_tsdd_opts *o);
int vocr_tsdp_nopts(void);
int vocr_tsdd_nopts(void);
/* 1 when any option of vocr_tsdp_pre (merge, rows) or of vocr_tsdp_post (ext) is on. */
int vocr_tsdp_pre_on(const vocr_tsdp_opts *o);
int vocr_tsdp_post_on(const vocr_tsdp_opts *o);

/* Scratch bytes for vocr_tsdp_pre / vocr_tsdp_post on a w x h frame (any box, any count up to cap); the scratch
 * must be aligned for double (malloc's alignment). */
size_t vocr_tsdp_scratch_bytes(unsigned w, unsigned h, unsigned cap);

/* The pre-refinement options on n boxes in place (boxes holds room for cap of them): *n is updated. grey: the frame's
 * luma (w x h, row stride `stride`). VOCR_TSDP_E_ARG for a nonzero gap_pct (no C twin) or a bad argument. */
int vocr_tsdp_pre(const unsigned char *grey, unsigned w, unsigned h, unsigned stride, vocr_tsdp_box *boxes,
                  unsigned *n, unsigned cap, const vocr_tsdp_opts *o, void *scratch, size_t bytes);
/* The post-refinement options (ext) on n boxes in place; *n may shrink (ext_join). */
int vocr_tsdp_post(const unsigned char *grey, unsigned w, unsigned h, unsigned stride, vocr_tsdp_box *boxes,
                   unsigned *n, const vocr_tsdp_opts *o, void *scratch, size_t bytes);

/* The reading filter on n read horizontal lines: xywh (n x 4 doubles, the tight boxes, frame px), text (n UTF-8
 * readings, NUL-terminated), conf_mean (n values, or NULL when conf_milli is 0); drop[i] = 1 for a line the filter
 * leaves out, else 0. Returns the count dropped, or VOCR_TSDP_E_ARG. */
int vocr_tsdd_drop(const double *xywh, const char *const *text, const float *conf_mean, unsigned n,
                   const vocr_tsdd_opts *o, unsigned char *drop);
/* tools/tsdesk.has_japanese / japanese_share of one UTF-8 reading (tests): the share in [0, 1], -1 for bad UTF-8. */
int vocr_tsdd_has_japanese(const char *text);
double vocr_tsdd_japanese_share(const char *text);

#ifdef __cplusplus
}
#endif

#endif
