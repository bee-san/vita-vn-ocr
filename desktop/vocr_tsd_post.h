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
 *     vflip_pct   a vertical box whose ink is a stack of at least vflip_min glyph bands (runs of ink rows at least
 *                 max(2, 15 % of its width) tall), their median height at least vflip_h_pct % of its width and their
 *                 median gap at least vflip_pct % of that height, becomes one horizontal box per band (a kana grid, a
 *                 menu; tools/tsdesk.vflip): the pieces of the frame's turned boxes that share a row (vertical
 *                 overlap at least half the smaller height) within vflip_join_pct % of the larger height along it
 *                 join, pair by pair in order; they follow the boxes left as they were;
 *     merge_pct   two boxes of one orientation that overlap across by at least half the smaller size across and whose
 *                 intersection is at least merge_pct % of the smaller one's area become their union (tools/tsdesk.
 *                 merge_overlaps: the pair with the largest share first, the larger box's fields, its place the
 *                 earlier of the two);
 *     join_pct    two horizontal boxes the detector tags as dialogue (tag 0) on one row (vertical overlap at least
 *                 join_ov_pct % of the smaller height), the smaller height at least join_h_pct % of the larger, with a
 *                 gap along of at most join_pct % of the larger height become their union (tools/tsdesk.join_row: the
 *                 first pair in order, the larger box's fields, its place the earlier's), until none qualifies;
 *     rows_pct    a horizontal box split across at an ink valley between two bands of similar height (tools/tsdesk.
 *                 split_rows, valley_pct and rows_len), at most two levels;
 *     gap_pct     the along split (tools/tsdesk.split_along): not in C (not adopted); a nonzero value is refused.
 *   after the refinement (vocr_tsdp_post):
 *     ext_pct     a horizontal box's ends extended over the solid dash run beyond them (tools/tsdesk.extend_ends,
 *                 ext_gap_pct, ext_run_pct, ext_dash_pct, ext_join).
 * At most `cap` boxes come out of vocr_tsdp_pre (tools/tsdesk.post_pre keeps the first MAX_BOXES).
 *
 * The reading stage's cut (vocr_tsdc_cut: tools/tsdesk.cut_reading) cuts a read line into items; the page runs it.
 * The reading filter (vocr_tsdd_drop: tools/tsdesk.drop_set) decides which of a page's read horizontal lines are left
 * out, from their tight boxes and UTF-8 readings: nojp (no Japanese character; 2: unless a mostly Japanese reading
 * sits in the row within nojp_gap_pct % of the larger height, nojp_anchor_pct), rep_pct (a reading dominated by one
 * non-Japanese character, rep_len characters or more), conf_milli (the mean character confidence under
 * conf_milli / 1000), garb_drop (a reading of at least garb_len code points, spaces aside, at least garb_pct % of them
 * digits or Latin letters, with a kana, no kanji and no two kana side by side: tools/tsdesk.is_garbage, any tag).
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
    int vflip_pct, vflip_join_pct, vflip_min, vflip_h_pct;
    int join_pct, join_h_pct, join_ov_pct;
} vocr_tsdp_opts;

#define VOCR_TSDP_NOPTS 18

/* The reading filter's options, in tools/tsdesk.DROP_DEFAULTS order. */
typedef struct {
    int nojp, nojp_gap_pct, nojp_anchor_pct, rep_pct, conf_milli, rep_len;
    int cut_pct, cut_rel_pct, cut_dlg;      /* the reading stage's cut (vocr_tsdc_cut; the page runs it) */
    int garb_drop, garb_len, garb_pct;      /* a garbage reading dropped (tools/tsdesk.is_garbage) */
} vocr_tsdd_opts;

#define VOCR_TSDD_NOPTS 12

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
/* 1 when any option of vocr_tsdp_pre (vflip, merge, join, rows) or of vocr_tsdp_post (ext) is on. */
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

/* The reading stage's cut (tools/tsdesk.cut_reading; the C page runs it on each read horizontal line before the
 * reading filter, tools/tsdesk.read_frame): a line cut into items at its wide ink gaps by its characters' positions.
 * box: the tight box x0, y0, x1, y1 (frame px); cx: the n characters' centres in frame px, in order; mark: 1 for a
 * character that is a punctuation mark (vocr_tsdc_is_mark). A gap is the longest run of empty ink columns (the
 * post-pass's ink over the box) between two consecutive centres; it cuts when it is at least max(2, pct % of the
 * box's pixel height) and at least rel_pct % of the line's median such gap (rounded up), unless either character is
 * a mark; a run as long between the box's first (last) ink and the first (last) centre trims ink that read as no
 * character. Returns 0: no cut (tools/tsdesk's None); 1: *np pieces in x0 / x1 (frame px), each holding the
 * characters [first, end), at most n of them; VOCR_TSDP_E_ARG (also for n above VOCR_TSDC_MAX) or
 * VOCR_TSDP_E_SCRATCH. scratch: vocr_tsdp_scratch_bytes(w, h, 1) bytes or more. */
#define VOCR_TSDC_MAX 256
int vocr_tsdc_cut(const unsigned char *grey, unsigned w, unsigned h, unsigned stride, const double *box,
                  const double *cx, const unsigned char *mark, unsigned n, int pct, int rel_pct, void *scratch,
                  size_t bytes, double *x0, double *x1, unsigned *first, unsigned *end, unsigned *np);
/* tools/tsdesk.is_garbage on a NUL-terminated UTF-8 reading (garb_drop's test). */
int vocr_tsdd_is_garbage(const char *text, int min_len, int pct);
/* 1 when the UTF-8 text of len bytes is exactly one code point of tools/tsdesk.CUT_MARKS. */
int vocr_tsdc_is_mark(const char *text, unsigned len);

#ifdef __cplusplus
}
#endif

#endif
