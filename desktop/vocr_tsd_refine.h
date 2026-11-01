/* vocr_tsd_refine.h - textspot-desktop's box refinement (issue #173): each horizontal box's four edges snapped to its
 * line's ink on the full-resolution grey frame.
 *
 * tools/tsdesk_refine.py is the reference (its module docstring states the rule step by step); this C99 file gives
 * its boxes bit for bit: every step is integer arithmetic on the grey frame (the input box's floor / ceil aside), so
 * the two cannot drift by rounding. No allocation: the caller passes a scratch buffer of vocr_tsdr_scratch_bytes(w,
 * h) bytes, reused for every box. AVX2 kernels when the compiler targets them (-mavx2, -march=x86-64-v3), plain C
 * otherwise (-DVOCR_TSDR_SCALAR forces it); both give the same bits. Vertical boxes are left alone.
 */
#ifndef VOCR_TSD_REFINE_H
#define VOCR_TSD_REFINE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* The options, in tools/tsdesk_refine.KEYS order (integers; *_pct are percentages of the line's height). */
typedef struct {
    int min_h, min_w;
    int band_pct, band_min;
    int start_pct, end_pct, along_min;
    int core_pct, along_core_pct;
    int row_gap, valley_pct, low_run_pct, rise_pct, beyond_pct;
    int gap_pct, gap_min;
    int bar_pct;
    int shrink_a_pct, shrink_l_pct;
    int floor_, k_pct;
    int margin_pct, margin_gap_pct, margin_min, noise_add, texture_pct;
    int busy_pct;
    int neighbours;           /* 1: an edge never grows into another box (neighbour limits) */
} vocr_tsdr_opts;

#define VOCR_TSDR_NOPTS 28

/* status bits, one word per box (tools/tsdesk_refine.ST_*) */
enum {
    VOCR_TSDR_SNAP_L = 1, VOCR_TSDR_SNAP_T = 2, VOCR_TSDR_SNAP_R = 4, VOCR_TSDR_SNAP_B = 8,
    VOCR_TSDR_SMALL = 16,     /* under min_h x min_w px (or not finite): kept */
    VOCR_TSDR_NOINK = 32,     /* no ink in the body: kept */
    VOCR_TSDR_LIMIT = 64,     /* an edge's walk ran out of the search band: that edge kept */
    VOCR_TSDR_SHRINK = 128,   /* an edge would move in past the shrink bound: that edge kept */
    VOCR_TSDR_BUSY = 512,     /* ink everywhere around the box: kept */
    VOCR_TSDR_VERT = 1024     /* vertical: left alone */
};

enum { VOCR_TSDR_OK = 0, VOCR_TSDR_E_ARG = -1, VOCR_TSDR_E_SCRATCH = -2 };

/* The defaults (tools/tsdesk_refine.DEFAULTS). */
void vocr_tsdr_defaults(vocr_tsdr_opts *o);
int vocr_tsdr_nopts(void);

/* Scratch bytes for vocr_tsdr_refine of n boxes on a w x h frame (enough for any box on it); the scratch must be
 * aligned for int (malloc's alignment). */
size_t vocr_tsdr_scratch_bytes(unsigned w, unsigned h, unsigned n);

/* Refine n boxes in place on a grey frame (w x h, row stride `stride` bytes): xyxy holds n x 4 doubles [x0, y0, x1,
 * y1] in frame px; vertical (n flags, or NULL: all horizontal); opts NULL: the defaults; status (n words, or NULL)
 * receives each box's bits. An edge never grows into another of the n boxes (opts->neighbours; the limits come from
 * the input boxes, so the order of the boxes does not matter). Returns VOCR_TSDR_OK, or an error before any box is
 * touched. */
int vocr_tsdr_refine(const unsigned char *grey, unsigned w, unsigned h, unsigned stride, double *xyxy,
                     const unsigned char *vertical, unsigned n, const vocr_tsdr_opts *opts, unsigned *status,
                     void *scratch, size_t scratch_bytes);

/* 1 when this build uses the AVX2 kernels. */
int vocr_tsdr_simd(void);

#ifdef __cplusplus
}
#endif

#endif
