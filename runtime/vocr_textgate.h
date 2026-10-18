/* The text-presence gate in front of the dialogue line finder (issue #100, docs/textgate.md): does a
 * region hold text at all? An int8 network in #97's VTDT layout (train/textdet_export.py writes it,
 * synth/textdet_twin.py documents it) runs on the region's grey; only the first map of its head, K (a
 * text logit per s x s cell, s = 4 or 8), is computed. The score is
 *
 *     VOCR_TEXTGATE_REGION   the k-th largest K code of the region's map (the smallest when it has
 *                            fewer than k cells)
 *     VOCR_TEXTGATE_BOXES    the k-th largest K code over the cells whose s x s block meets a box of
 *                            the finder's (VOCR_TEXTGATE_NO_CELLS when no cell does)
 *
 * and the region holds text iff score >= t. eval/textgate.py (kmap, kth, box_mask, score) is the
 * specification and a bit-exact twin.
 *
 * C99, no allocation, no recursion, no global state, integer arithmetic only, scalar code (built for
 * size); nothing from libc but memset. Self-contained: link vocr_textgate.c (and
 * vocr_lines.c / vocr_lines_rows.c for the finder mode) without vocr_textdet.c. The caller passes an
 * arena of vocr_textgate_arena_bytes(); the region is read once, top to bottom, so in region mode the
 * arena depends on the region's width only.
 */
#ifndef VOCR_TEXTGATE_H
#define VOCR_TEXTGATE_H

#include <stddef.h>

#include "vocr_lines.h"
#include "vocr_lines_rows.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VOCR_TEXTGATE_REGION 0
#define VOCR_TEXTGATE_BOXES 1

#define VOCR_TEXTGATE_E_BLOB (-5)     /* a blob vocr_textgate_load refuses (= VOCR_TEXTDET_E_BLOB) */
#define VOCR_TEXTGATE_NO_CELLS (-129) /* boxes mode: no cell meets a box */
#define VOCR_TEXTGATE_MAX_K 8u        /* k is 1..8 */
#define VOCR_TEXTGATE_MAX_SIDE 2048u  /* larger regions are VOCR_LINES_E_ARG (vocr_textgate.c: the ranges) */

/* A loaded blob: the blob itself (checked), which must stay alive and unchanged while the model is
 * used; it needs no alignment. NULL: not loaded. Read-only after vocr_textgate_load. */
typedef struct {
    const unsigned char *blob;
} vocr_textgate_model;

/* One gate run (the push form): 32 bytes on 32-bit ARM, the caller's; everything else is in the
 * arena. Fields are private. */
typedef struct {
    const unsigned char *blob;  /* the model's blob */
    unsigned char *base;        /* the arena, 4-byte aligned */
    unsigned short width, height, rows;
    unsigned char n, mode, k, count;
    signed char top[VOCR_TEXTGATE_MAX_K];   /* the k largest codes so far, largest first */
    int rc;                     /* VOCR_LINES_OK while the run is good, else the error */
} vocr_textgate_state;

/* Checks the blob as synth/textdet_twin.Model does (magic, version, size, CRC-32, header fields, the
 * layer chain, section bounds and alignment, multipliers > 0, shifts 1..62, the 4-map head at the
 * output stride). VOCR_LINES_OK, VOCR_LINES_E_ARG (a NULL pointer) or VOCR_TEXTGATE_E_BLOB; on an
 * error m is cleared (unusable). */
int vocr_textgate_load(vocr_textgate_model *m, const void *blob, size_t bytes);

/* Exact arena bytes of a gate run on any region up to max_width x max_height: 20 bytes per layer, every
 * layer's ring of input rows ((k - 1) d + 1 rows of cin codes a pixel, with d zero pixels either side for
 * a 3x3 layer; one row for a 1x1 layer), then the K row (region mode: ceil(w / s) bytes) or the K map
 * (boxes mode: ceil(h / s) x ceil(w / s) bytes), plus 3 bytes of alignment slack (the arena needs no
 * alignment). Region mode depends on the width only. 0 for a NULL or unloaded model, a bad mode, a
 * zero side or one over VOCR_TEXTGATE_MAX_SIDE. */
size_t vocr_textgate_arena_bytes(const vocr_textgate_model *m, int mode, unsigned max_width,
                                 unsigned max_height);

/* The push form. begin starts a run over a width x height region (k is 1..VOCR_TEXTGATE_MAX_K); push
 * takes its grey rows 0..height-1 in order (width bytes each, copied: a row only has to stay valid
 * during the call); score gives the decision's score once every row is in. boxes: the finder's lines
 * in boxes mode (VOCR_LINES_MAX at most), ignored (may be NULL) in region mode; in boxes mode score may
 * be asked again with other boxes. Return codes: VOCR_LINES_OK, VOCR_LINES_E_ARG (NULL pointers, an
 * unloaded model, a bad mode or k, a zero or oversized side, a row past the last, a score asked before
 * the last row), VOCR_LINES_E_ARENA (begin: arena_bytes below vocr_textgate_arena_bytes). An error of
 * begin or push sticks: every later call returns it. *score is written only on success. */
int vocr_textgate_begin(vocr_textgate_state *st, const vocr_textgate_model *m, int mode, unsigned k,
                        unsigned width, unsigned height, void *arena, size_t arena_bytes);
int vocr_textgate_push(vocr_textgate_state *st, const unsigned char *grey_row);
int vocr_textgate_score(vocr_textgate_state *st, const VocrLineSet *boxes, int *score);

/* Region mode only: vocr_textgate_score(st, NULL, score) without the boxes code, so a region-mode
 * caller links less (VOCR_LINES_E_ARG in boxes mode). */
int vocr_textgate_score_region(vocr_textgate_state *st, int *score);

/* The same in one call on an image in memory (VOCR_LINES_GREY or VOCR_LINES_RGBA: grey = (77 R +
 * 150 G + 29 B) >> 8), or through a row callback (vocr_lines_rows.h: rows asked once each, in
 * increasing order; a NULL row ends the call with VOCR_LINES_E_SOURCE). */
int vocr_textgate_image(const vocr_textgate_model *m, int mode, unsigned k, const VocrLinesImage *im,
                        const VocrLineSet *boxes, void *arena, size_t arena_bytes, int *score);
int vocr_textgate_rows(const vocr_textgate_model *m, int mode, unsigned k, const VocrLinesRows *src,
                       const VocrLineSet *boxes, void *arena, size_t arena_bytes, int *score);

/* Debug and test hook: in boxes mode, after the last row, the K map (hm rows of wm codes); NULL
 * otherwise. */
const signed char *vocr_textgate_kmap(const vocr_textgate_state *st, unsigned *wm, unsigned *hm);

/* ---- the finder mode behind a flag */

/* Arena bytes of vocr_find_lines_gated(_rows): the larger of vocr_lines_arena_bytes and the gate's
 * (g NULL: vocr_lines_arena_bytes alone). The finder and the gate run one after the other in it. */
size_t vocr_lines_gated_arena_bytes(const vocr_textgate_model *g, int mode, unsigned max_width,
                                    unsigned max_height);

/* g NULL: exactly vocr_find_lines(im, finder, arena, arena_bytes, out) (mode, k and t unread).
 * Region mode: the gate first; score < t gives no lines (count 0, truncated 0) and the finder never
 * runs, else the finder's lines. Boxes mode: the finder, then the gate on the finder's boxes in the
 * same arena; score < t clears the lines. The caller refines the lines afterwards as before. On any
 * error (the gate's or the finder's) the error is returned with no lines. The arena is checked
 * against vocr_lines_gated_arena_bytes up front (VOCR_LINES_E_ARENA). */
int vocr_find_lines_gated(const VocrLinesImage *im, int finder, const vocr_textgate_model *g, int mode,
                          unsigned k, int t, void *arena, size_t arena_bytes, VocrLineSet *out);

/* The row-callback form: g NULL is exactly vocr_find_lines_rows; the gate's pass reads every row once
 * in order, before (region mode) or after (boxes mode) the finder's passes. */
int vocr_find_lines_gated_rows(const VocrLinesRows *src, int finder, const vocr_textgate_model *g,
                               int mode, unsigned k, int t, void *arena, size_t arena_bytes,
                               VocrLineSet *out);

#ifdef __cplusplus
}
#endif

#endif
