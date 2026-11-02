/* vocr_tsd.h - textspot-desktop's line detector in C (issue #173, docs/textspot-desktop.md).
 *
 * The detector (train/tsdesk_net.py TsdNet) is a small float32 fully convolutional network: a 4x4-patch stem at stride
 * 4, separable stages at strides 8, 16 and 32 (optional global context at 32), a top-down pyramid of 1x1 laterals back
 * to the head's stride (8 or 4), optional separable smoothing and a 1x1 head with 6 + T maps (K and O logits, A1, A2,
 * E1, E2 in input px, T tag logits). This file runs it from a VTSD blob (train/tsdesk_blob.py) with no dependency but
 * libm and desktop/vocr_tsd_resize.c (vocr-desktop's integer resize, luma and input policy, bit for bit, with SIMD paths
 * and row bands): channel-innermost float32 activations,
 * AVX2 + FMA kernels when the compiler targets them (-mavx2 -mfma, or -march=x86-64-v3) and plain C otherwise
 * (-DVOCR_TSD_SCALAR forces the plain-C kernels). The ONNX file of the same weights (tools/tsdesk.py) computes the same
 * network in another order of float operations, so the two give maps equal within float rounding; given the same maps,
 * vocr_tsd_decode gives tools/tsdesk.decode's boxes bit for bit (the same sums in the same order, in double).
 *
 * Threads: vocr_tsd_maps_mt and vocr_tsd_detect_mt split each step of the network into row bands on a caller's runner
 * (vocr_tsd_pool_* is one); every output value is computed by the same operations whatever the split, so the maps
 * and the boxes are the same bits at any thread count. The page pipeline (vocr_tsd_page_*) adds VOCR's recognizer
 * (runtime/vocr.c W8I, greedy, the blob's head sections on, as Rubi reads a line), the lines read on the same pool.
 * This is desktop code: it allocates (malloc) in the pool and the page pipeline and uses POSIX threads
 * (-DVOCR_DESK_NO_THREADS leaves them out). Nothing here is console code.
 */
#ifndef VOCR_TSD_H
#define VOCR_TSD_H

#include <stddef.h>
#include <stdint.h>

#include "vocr_tsd_post.h"
#include "vocr_tsd_refine.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VOCR_TSD_VERSION 1u
#define VOCR_TSD_HEADER 160u            /* header bytes of a VTSD blob */
#define VOCR_TSD_ALIGN 32u              /* the network's input sides are multiples of this */
#define VOCR_TSD_MAX_TAGS 8u
#define VOCR_TSD_MAX_BLOCKS 8u          /* residual separable blocks per stage */
#define VOCR_TSD_MAX_SMOOTH 4u
#define VOCR_TSD_MAX_CHANNELS 256u
#define VOCR_TSD_MAX_BOXES 256u         /* boxes kept per frame (the highest mean kernel logits) */
#define VOCR_TSD_MAX_SIDE 8192u         /* a frame side the detector accepts */
#define VOCR_TSD_TEXT 1024u             /* bytes of a line's text, NUL included */

enum {
    VOCR_TSD_OK = 0,
    VOCR_TSD_E_ARG = -1,      /* a bad argument: a NULL pointer, a frame of 0 or more than 8192 px a side, a policy
                                 vocr_tsd_policy_ok refuses */
    VOCR_TSD_E_BLOB = -2,     /* not a VTSD version-1 blob, or one this build cannot read */
    VOCR_TSD_E_CRC = -3,      /* the blob's CRC-32 does not match */
    VOCR_TSD_E_ARENA = -4,    /* the arena is too small */
    VOCR_TSD_E_MEM = -5,      /* an allocation failed (the pool, the page pipeline) */
    VOCR_TSD_E_REC = -6       /* the recognizer refused its blob or a line */
};

/* input policies (vocr_desk_input_size's, tools/tsdesk.input_size's): native, fit:n (the longer side down to n px, never
 * up), long:n */
enum { VOCR_TSD_SIZE_NATIVE = 0, VOCR_TSD_SIZE_FIT = 1, VOCR_TSD_SIZE_LONG = 2 };

typedef struct {
    const float *w, *b;       /* weights in the kernel's layout, bias */
    unsigned cin, cout;       /* a depthwise layer: cin == cout */
    unsigned stride, dil;
} vocr_tsd_layer;

typedef struct {               /* a parsed blob: points into it, so keep the blob alive */
    unsigned patch, hs;                    /* 4; the head's stride, 4 or 8 */
    unsigned c[4], n[4], F, smooth, gc, tags, out, outp;   /* out = 6 + tags maps; outp: out rounded up to 8 */
    unsigned min_cells, max_boxes;         /* the decode's constants (tools/tsdesk.DEFAULTS) */
    double conf_min_logit, min_across, along_reach;
    unsigned size_kind, size_n;            /* the default input policy: fit:960 */
    unsigned dil[4][VOCR_TSD_MAX_BLOCKS];
    vocr_tsd_layer stem;                   /* 4x4 at stride 4, 1 -> c0 (w: [16][c0]) */
    /* [s][0]: stage s's down block (s >= 1); [s][1 + i]: its residual block i */
    vocr_tsd_layer dw[4][1 + VOCR_TSD_MAX_BLOCKS], pw[4][1 + VOCR_TSD_MAX_BLOCKS];
    vocr_tsd_layer gcl;                    /* gc: c3 -> c3 */
    vocr_tsd_layer lat[4];                 /* by stride: [0] stride 4 (hs == 4 only), [1] 8, [2] 16, [3] 32 */
    vocr_tsd_layer sdw[VOCR_TSD_MAX_SMOOTH], spw[VOCR_TSD_MAX_SMOOTH];
    vocr_tsd_layer head;                   /* 1x1 F -> outp (A1, A2, E1, E2 already in input px) */
    size_t params;
} vocr_tsd_model;

typedef struct {
    double x, y, w, h;          /* frame px (vocr_tsd_detect) or input px (vocr_tsd_decode) */
    double logit;               /* the mean K logit over the box's kernel cells */
    double tconf;               /* (best tag score - second) / cells */
    unsigned vertical, tag, cells;
} vocr_tsd_box;

typedef struct {
    unsigned count, truncated;  /* truncated: more boxes than max_boxes passed the decode */
    unsigned in_w, in_h;        /* the network input's image part (before padding) */
    vocr_tsd_box boxes[VOCR_TSD_MAX_BOXES];    /* sorted: logit descending, then y, then x (stable) */
} vocr_tsd_set;

typedef struct {               /* the decode's constants (the blob's: vocr_tsd_decode_opts_of) */
    unsigned stride, tags, min_cells, max_boxes;
    double conf_min_logit, min_across, along_reach;
    int vote_pct;              /* tools/tsdesk.DECODE_EXTRA's vote_pct (0: off, the blob's default): each component
                                  split across where its cells' A1 / A2 edge votes are bimodal (vote_split) */
} vocr_tsd_decode_opts;

/* Parse and check a blob (magic, version, sizes, CRC-32); the blob must be 4-byte aligned. */
int vocr_tsd_load(vocr_tsd_model *m, const void *blob, size_t bytes);
void vocr_tsd_decode_opts_of(const vocr_tsd_model *m, vocr_tsd_decode_opts *o);

/* 1 when (kind, n) is a policy the detector takes (vocr_desk_policy_ok's rule): native (n ignored), or fit:N / long:N
 * with 1 <= N <= VOCR_TSD_MAX_SIDE. */
int vocr_tsd_policy_ok(unsigned kind, unsigned n);
/* The network input for a w x h frame under a policy (vocr_desk_input_size's rule): the image part iw x ih and the padded
 * input pw x ph (multiples of 32). All four are 0 for a policy vocr_tsd_policy_ok refuses or a frame of 0 or more than
 * VOCR_TSD_MAX_SIDE px a side; otherwise every side is at most VOCR_TSD_MAX_SIDE. */
void vocr_tsd_input_size(unsigned kind, unsigned n, unsigned w, unsigned h, unsigned *iw, unsigned *ih,
                         unsigned *pw, unsigned *ph);

/* A parallel runner: run fn(arg, task, ntasks) for task = 0 .. ntasks - 1 (in any order, on any threads) and return
 * when every task is done. The same type as vocr_desk_runner. */
typedef void (*vocr_tsd_task_fn)(void *arg, unsigned task, unsigned ntasks);
typedef void (*vocr_tsd_runner)(void *ctx, vocr_tsd_task_fn fn, void *arg, unsigned ntasks);

/* Arena bytes for vocr_tsd_maps_mt on a pw x ph padded input with ntasks tasks (1: vocr_tsd_maps). */
size_t vocr_tsd_maps_bytes(const vocr_tsd_model *m, unsigned pw, unsigned ph, unsigned ntasks);
/* The network on a padded grey input (pw x ph, multiples of 32, row-major, stride pw): *maps points into the arena,
 * channel-innermost float32 [ph / hs][pw / hs][out]. */
int vocr_tsd_maps(const vocr_tsd_model *m, const unsigned char *in, unsigned pw, unsigned ph, void *arena,
                  size_t bytes, const float **maps);
int vocr_tsd_maps_mt(const vocr_tsd_model *m, const unsigned char *in, unsigned pw, unsigned ph, void *arena,
                     size_t bytes, const float **maps, vocr_tsd_runner run, void *ctx, unsigned ntasks);

/* The decode (tools/tsdesk.decode) on maps as vocr_tsd_maps gives them (mw x mh cells of `out` = 6 + tags channels,
 * channel-innermost): the boxes of the iw x ih image part, in input px. scratch: vocr_tsd_decode_bytes(mw, mh, tags). */
size_t vocr_tsd_decode_bytes(unsigned mw, unsigned mh, unsigned tags);
/* The scratch for options o (vote_pct needs more; o NULL: vocr_tsd_decode_bytes). */
size_t vocr_tsd_decode_bytes_ex(unsigned mw, unsigned mh, unsigned tags, const vocr_tsd_decode_opts *o);
int vocr_tsd_decode(const vocr_tsd_model *m, const float *maps, unsigned mw, unsigned mh, unsigned iw, unsigned ih,
                    void *scratch, size_t bytes, vocr_tsd_set *out);
int vocr_tsd_decode_ex(const vocr_tsd_decode_opts *o, const float *maps, unsigned mw, unsigned mh, unsigned iw,
                       unsigned ih, void *scratch, size_t bytes, vocr_tsd_set *out);

/* Arena bytes for vocr_tsd_detect(_mt) on a w x h frame of `channels` (1 grey, 3 RGB, 4 RGBA) under a policy. */
size_t vocr_tsd_arena_bytes(const vocr_tsd_model *m, unsigned kind, unsigned n, unsigned w, unsigned h,
                            unsigned channels, unsigned ntasks);
/* The whole detector on a frame in memory (w x h, `channels` bytes per pixel, row stride `stride` bytes): the integer
 * luma of an RGB(A) frame (vocr_tsd_grey: vocr_desk_grey's), the policy's resize (vocr_tsd_resize: vocr_desk_resize's
 * rule, Q14 area weights down, bilinear
 * up), edge padding to multiples of 32, the network and the decode; boxes in frame px (tools/tsdesk.Detector.detect). */
int vocr_tsd_detect(const vocr_tsd_model *m, const unsigned char *img, unsigned w, unsigned h, unsigned stride,
                    unsigned channels, unsigned kind, unsigned n, void *arena, size_t bytes, vocr_tsd_set *out);
int vocr_tsd_detect_mt(const vocr_tsd_model *m, const unsigned char *img, unsigned w, unsigned h, unsigned stride,
                       unsigned channels, unsigned kind, unsigned n, void *arena, size_t bytes, vocr_tsd_set *out,
                       vocr_tsd_runner run, void *ctx, unsigned ntasks);

/* The same with the box refinement (vocr_tsd_refine.h; tools/tsdesk.Detector's refine) when refine is not NULL: after
 * the boxes are mapped to frame px, every horizontal box's edges are snapped to its line's ink on the frame's
 * full-resolution luma (vocr_tsdr_refine, the boxes in the decode's order). The arena for it:
 * vocr_tsd_arena_bytes_ex(..., refine = 1). With refine NULL these are vocr_tsd_arena_bytes / vocr_tsd_detect_mt. */
size_t vocr_tsd_arena_bytes_ex(const vocr_tsd_model *m, unsigned kind, unsigned n, unsigned w, unsigned h,
                               unsigned channels, unsigned ntasks, int refine);
int vocr_tsd_detect_ex(const vocr_tsd_model *m, const unsigned char *img, unsigned w, unsigned h, unsigned stride,
                       unsigned channels, unsigned kind, unsigned n, const vocr_tsdr_opts *refine, void *arena,
                       size_t bytes, vocr_tsd_set *out, vocr_tsd_runner run, void *ctx, unsigned ntasks);

/* The detector with the round-2 options (protocol v2 amendment 4's C6; tools/tsdesk.Detector's decode_opts, refine and
 * post): decode NULL, the blob's decode options (vote_pct 0); refine NULL, no refinement; post NULL, no post-pass
 * (vocr_tsd_post.h: merge and rows on the frame-px boxes before the refinement, ext after it). With every pointer NULL
 * this is vocr_tsd_detect_mt bit for bit, and with only refine set vocr_tsd_detect_ex. The arena:
 * vocr_tsd_arena_bytes_opt with the same options. */
typedef struct {
    const vocr_tsd_decode_opts *decode;
    const vocr_tsdr_opts *refine;
    const vocr_tsdp_opts *post;
} vocr_tsd_opts;
size_t vocr_tsd_arena_bytes_opt(const vocr_tsd_model *m, unsigned kind, unsigned n, unsigned w, unsigned h,
                                unsigned channels, unsigned ntasks, const vocr_tsd_opts *opt);
int vocr_tsd_detect_opt(const vocr_tsd_model *m, const unsigned char *img, unsigned w, unsigned h, unsigned stride,
                        unsigned channels, unsigned kind, unsigned n, const vocr_tsd_opts *opt, void *arena,
                        size_t bytes, vocr_tsd_set *out, vocr_tsd_runner run, void *ctx, unsigned ntasks);

/* sizeof(vocr_tsd_model), sizeof(vocr_tsd_set), sizeof(vocr_tsd_box), for callers that allocate them (ctypes). */
size_t vocr_tsd_model_bytes(void);
size_t vocr_tsd_set_bytes(void);
size_t vocr_tsd_box_bytes(void);
/* 1 if this build runs the AVX2 + FMA kernels, 2 the AVX-512 ones, 0 the plain-C ones. */
int vocr_tsd_simd(void);

/* ---- a thread pool (POSIX threads; one thread with -DVOCR_DESK_NO_THREADS) and its vocr_tsd_runner ---- */

typedef struct vocr_tsd_pool vocr_tsd_pool;
vocr_tsd_pool *vocr_tsd_pool_new(unsigned threads);
unsigned vocr_tsd_pool_threads(const vocr_tsd_pool *p);
/* the runner: pass vocr_tsd_pool_run as `run` and the pool as `ctx` */
void vocr_tsd_pool_run(void *pool, vocr_tsd_task_fn fn, void *arg, unsigned ntasks);
void vocr_tsd_pool_free(vocr_tsd_pool *p);

/* ---- the page pipeline: the detector + VOCR's recognizer (runtime/vocr.c), lines on the pool ---- */

typedef struct vocr_tsd_page vocr_tsd_page;

typedef struct {
    int x, y, w, h;             /* the crop read: the tight box grown by the pad, rounded outward, frame px */
    double bx, by, bw, bh;      /* the detector's tight box, frame px */
    double logit;
    unsigned tag, dark, vertical;
    float conf_min;             /* the recognizer's lowest character confidence (0..1) */
    float conf_mean;            /* the mean character confidence (runtime/vocr.c's conf_mean; 1 with no character) */
    char text[VOCR_TSD_TEXT];   /* greedy UTF-8; empty for a vertical box (not read) */
} vocr_tsd_line;

/* det/rec: the blobs (kept alive by the caller); threads >= 1; kind < 0: the blob's input policy. The reading pad:
 * pad_h / pad_v frame px horizontally / vertically when pad_px is nonzero, else the rule max(2, nearbyint(k H / 544))
 * frame px for an H px tall frame with k = pad_h horizontally and k = pad_v vertically (4, 4: eval's pad "auto"). */
vocr_tsd_page *vocr_tsd_page_new(const void *det, size_t det_bytes, const void *rec, size_t rec_bytes,
                                 unsigned threads, int kind, unsigned n, unsigned pad_h, unsigned pad_v,
                                 unsigned pad_px, int *err);
/* The reading rule in full (eval/tsdesk_eval.margin_hv: protocol v2 amendments 1 and 2), in place of the pad given to
 * vocr_tsd_page_new. Each horizontal box (x, y, w, h) is grown by (ph, pv) frame px and rounded outward:
 *   VOCR_TSD_RULE_FRAME  "kAvBfC": ph = max(c, nearbyint(a H / 544)), pv = max(2, nearbyint(b H / 544)) for an H px
 *                        tall frame (kN and kAvB are c = 2: vocr_tsd_page_new's rule);
 *   VOCR_TSD_RULE_LINE   "hXvY": with bw = max(1, (x + w) - x) and bh = max(1, (y + h) - y), ph = max(1, a bh) and
 *                        pv = max(1, b bh) when bw >= bh, else ph = max(1, b bw) and pv = max(1, a bw);
 *   VOCR_TSD_RULE_PX     ph = a, pv = b;
 *   VOCR_TSD_RULE_CLIP   "cXvY" (amendment 3: eval/tsdesk_eval.clip_sides): the line rule's ph / pv, then each side cut
 *                        to half the gap to the nearest other box of the frame (every box the detector returned, any
 *                        tag or orientation, a box with the same corners skipped) on that side: for the left and right
 *                        a box in the same row (overlapping the box across by at least half the smaller height, and
 *                        not along it), for the top and bottom a box overlapping it along (and not across).
 * VOCR_TSD_E_ARG (the rule unchanged) for an unknown kind, or a value that is negative, NaN or above 4096. */
enum { VOCR_TSD_RULE_FRAME = 0, VOCR_TSD_RULE_LINE = 1, VOCR_TSD_RULE_PX = 2, VOCR_TSD_RULE_CLIP = 3 };
int vocr_tsd_page_set_rule(vocr_tsd_page *p, unsigned kind, double a, double b, double c);
/* Only boxes whose tag bit is set in mask are read (bit t: tag t; the default ~0u: all). */
void vocr_tsd_page_set_tags(vocr_tsd_page *p, unsigned mask);
/* The box refinement on the detector's boxes before they are grown and read (opts copied; NULL: off, the default). */
void vocr_tsd_page_set_refine(vocr_tsd_page *p, const vocr_tsdr_opts *opts);
/* The round-2 options (copied; NULL: off, the default): the decode's (vote_pct; the other fields are the blob's when
 * opts is the value vocr_tsd_decode_opts_of gave), the post-pass (vocr_tsd_post.h), and the reading filter (after every
 * line is read, the lines tools/tsdesk.drop_set rejects are left out of the page: vocr_tsdd_drop). */
void vocr_tsd_page_set_decode(vocr_tsd_page *p, const vocr_tsd_decode_opts *opts);
void vocr_tsd_page_set_post(vocr_tsd_page *p, const vocr_tsdp_opts *opts);
void vocr_tsd_page_set_drop(vocr_tsd_page *p, const vocr_tsdd_opts *opts);
/* The reading stage's cut (the reading filter's cut_pct, cut_rel_pct, cut_dlg: tools/tsdesk.cut_and_filter): after
 * every line is read and before the reading filter, each horizontal line is cut into items by its greedy
 * characters' positions (runtime/vocr_charbox.c, flags 0) at its wide ink gaps (vocr_tsdc_cut), each piece a line in
 * its place: the line's fields with its own tight box, the reading rule's crop of it and its characters' text; a
 * line whose pieces would take the page past VOCR_TSD_MAX_BOXES lines stays whole. The positions need the
 * recognizer's calibration: cal = shift, then groups 0..7 (tools/vocr_onnx.Recognizer.cal, eval/charboxes_cal.json),
 * resized px x 16; NULL clears it (zeros). Setting it also keeps every read line's characters when the cut is off
 * (vocr_tsd_page_line_chars). VOCR_TSD_E_ARG for a value outside int16 or no memory. */
int vocr_tsd_page_set_charbox_cal(vocr_tsd_page *p, const int *cal);
/* Line i's characters (a piece's own) after vocr_tsd_page_run, when they are kept: their centres in frame px x 16 and
 * their byte offsets in the line's text, at most cap of them; returns their count (0: none kept, or a vertical line). */
unsigned vocr_tsd_page_line_chars(const vocr_tsd_page *p, unsigned i, int32_t *c16, unsigned *off, unsigned cap);
/* Read a frame (`channels` 1, 3 or 4; RGB(A) is taken to its integer luma first, timed as the page): lines in reading
 * order (rows top to bottom by vertical overlap, left to right in a row; tools/tsdesk.reading_order). */
int vocr_tsd_page_run(vocr_tsd_page *p, const unsigned char *img, unsigned w, unsigned h, unsigned stride,
                      unsigned channels);
/* The detector alone (vocr_tsd_page_boxes gives its boxes; no line is read). */
int vocr_tsd_page_detect(vocr_tsd_page *p, const unsigned char *img, unsigned w, unsigned h, unsigned stride,
                         unsigned channels);
unsigned vocr_tsd_page_count(const vocr_tsd_page *p);
const vocr_tsd_line *vocr_tsd_page_line(const vocr_tsd_page *p, unsigned i);
size_t vocr_tsd_line_bytes(void);
const vocr_tsd_set *vocr_tsd_page_boxes(const vocr_tsd_page *p);
/* The pad in frame px of the last run (horizontal, vertical), truncated to integers; (-1, -1) under a line rule
 * (VOCR_TSD_RULE_LINE, VOCR_TSD_RULE_CLIP: the pad is each line's own, its crop says it). */
void vocr_tsd_page_pad(const vocr_tsd_page *p, int *ph, int *pv);
/* The detector's last run alone, in us, and the whole page (detector + lines), in us. */
double vocr_tsd_page_detect_us(const vocr_tsd_page *p);
double vocr_tsd_page_total_us(const vocr_tsd_page *p);
void vocr_tsd_page_free(vocr_tsd_page *p);

#ifdef __cplusplus
}
#endif

#endif
