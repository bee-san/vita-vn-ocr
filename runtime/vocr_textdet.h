/* The learned text detector for the console finder (issue #96, docs/textdet.md): an int8
 * network on the grey region whose four output maps (text kernel, vertical text, the two edge
 * distances across a line) decode into line boxes, a vertical flag per box and, by an empty
 * set, "no text here".
 *
 * C99, no allocation, no recursion, no global state, integer arithmetic only. The caller passes
 * an arena of vocr_textdet_arena_bytes() bytes. The region is read row by row in one pass, so the
 * arena depends on the region's width, not its height, and the row-callback form
 * (vocr_lines_rows.h) needs no copy of the region. synth/textdet_twin.py is the specification and
 * a bit-exact Python twin: the same boxes, flags, order and truncation for every region and blob.
 *
 * The detector is also a finder mode of vocr_lines.h behind a flag (VOCR_LINES_TEXTDET with
 * vocr_find_lines_det); every other finder value runs vocr_find_lines unchanged.
 * vocr_lines.c, vocr_lines_rows.c and vocr_lines_refine.c are not touched: this file calls
 * vocr_find_lines / vocr_line_dark (vocr_lines.c) and vocr_find_lines_rows / vocr_line_dark_rows
 * (vocr_lines_rows.c), so link both.
 */
#ifndef VOCR_TEXTDET_H
#define VOCR_TEXTDET_H

#include <stddef.h>

#include "vocr_lines.h"
#include "vocr_lines_rows.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The finder mode of vocr_find_lines_det (vocr_lines.h's finders are 0..3). */
#define VOCR_LINES_TEXTDET 4

/* A blob vocr_textdet_load refuses (every check of synth/textdet_twin.Model). */
#define VOCR_TEXTDET_E_BLOB (-5)

/* Largest region side (width and height). It keeps every sum of the decode within int32; the
 * proof is at the top of vocr_textdet.c. Larger regions are VOCR_LINES_E_ARG. */
#define VOCR_TEXTDET_MAX_SIDE 2048u

#define VOCR_TEXTDET_MAX_LAYERS 32u

typedef struct {
    const unsigned char *w;     /* int8 weights [cout][cin, or 1 if depthwise][k][k], in the blob */
    const unsigned char *b;     /* little-endian int32 bias[cout], in the blob */
    const unsigned char *m;     /* little-endian int32 multiplier[cout] (> 0), in the blob */
    const unsigned char *sh;    /* u8 shift[cout] (1..62), in the blob */
    unsigned short cin, cout;
    unsigned char kind;         /* 0: dense conv, 1: depthwise */
    unsigned char k, stride, dil, relu;
} vocr_textdet_layer;

/* A loaded blob: about 1 KB on 32-bit ARM. It points into the blob, which must stay alive and
 * unchanged while the model is used; the blob needs no alignment. Read-only after
 * vocr_textdet_load, so threads can share it (each with its own arena). */
typedef struct {
    vocr_textdet_layer layers[VOCR_TEXTDET_MAX_LAYERS];
    unsigned n_layers;
    unsigned stride;            /* output stride: 4 or 8 region pixels per map cell */
    unsigned min_cells;         /* the decode drops kernel components of fewer cells */
    unsigned channels;          /* output channels of all layers */
    unsigned long blob_bytes;
    unsigned char ckpt[8];      /* sha256 prefix of the source checkpoint */
    char name[33];              /* shape name, NUL-terminated */
    unsigned loaded;
} vocr_textdet;

/* The detector's lines: the boxes and dark flags of vocr_lines.h (top to bottom; lines whose rows
 * overlap by half the smaller height left to right; then vertical lines right to left) plus a
 * vertical flag per line. */
typedef struct {
    VocrLine lines[VOCR_LINES_MAX];
    unsigned vertical[VOCR_LINES_MAX]; /* 1: vertical text (never handed to the horizontal recognizers) */
    unsigned count;                    /* 0: no text in the region */
    unsigned truncated;                /* 1 when lines were dropped: over 64 candidates, or over VOCR_LINES_MAX */
} VocrTextdetSet;

/* Checks the blob (magic, version, size, CRC-32, header fields, the layer chain, section bounds and
 * alignment, multipliers > 0, shifts 1..62) and points m into it. VOCR_LINES_OK, VOCR_LINES_E_ARG
 * (a NULL pointer) or VOCR_TEXTDET_E_BLOB; on an error m is cleared (unusable). */
int vocr_textdet_load(vocr_textdet *m, const void *blob, size_t bytes);

/* Exact arena bytes vocr_textdet_find / _find_rows / _maps need for any region up to max_width x
 * max_height with this model (it depends on the width only): the network's row rings, the map row,
 * the streaming decode and the 64 candidates, plus 7 bytes of alignment slack (the arena needs no
 * alignment). 0 for a NULL or unloaded model, a zero side or a side over VOCR_TEXTDET_MAX_SIDE. */
size_t vocr_textdet_arena_bytes(const vocr_textdet *m, unsigned max_width, unsigned max_height);

/* The detector on an image in memory (VOCR_LINES_GREY or VOCR_LINES_RGBA: grey = (77 R + 150 G +
 * 29 B) >> 8). Returns VOCR_LINES_OK, VOCR_LINES_E_ARG (NULL pointer, unloaded model, bad format or
 * stride, a zero side or one over VOCR_TEXTDET_MAX_SIDE) or VOCR_LINES_E_ARENA; out->count is 0 on
 * an error. The arena is scratch only. Each line's dark flag is vocr_line_dark of its box. */
int vocr_textdet_find(const vocr_textdet *m, const VocrLinesImage *im, void *arena, size_t arena_bytes,
                      VocrTextdetSet *out);

/* The same through a row callback (vocr_lines_rows.h): rows are asked once each in increasing
 * order in one pass over the region, then each line's box once, top to bottom, for its dark flag
 * (vocr_line_dark_rows); a returned row only needs to stay valid until the next call. A NULL row
 * ends the call with VOCR_LINES_E_SOURCE and no lines. The same lines as vocr_textdet_find for the
 * same pixels. */
int vocr_textdet_find_rows(const vocr_textdet *m, const VocrLinesRows *src, void *arena, size_t arena_bytes,
                           VocrTextdetSet *out);

/* Debug and test hook: runs the network on im and calls row(user, i, cells, width) for each row i
 * of the head's output, in increasing order; cells holds width cells of 4 int8 codes (K, O, A1, A2)
 * and is valid only during the call. Same arena and return codes as vocr_textdet_find. */
typedef void (*vocr_textdet_map_row)(void *user, unsigned i, const signed char *cells, unsigned width);
int vocr_textdet_maps(const vocr_textdet *m, const VocrLinesImage *im, void *arena, size_t arena_bytes,
                      vocr_textdet_map_row row, void *user);

/* ---- the finder mode behind a flag */

/* Arena bytes for vocr_find_lines_det(_rows) with any finder: the larger of
 * vocr_lines_arena_bytes(max_width, max_height) and vocr_textdet_arena_bytes(m, ...) (m may be
 * NULL when the detector is not used). */
size_t vocr_lines_det_arena_bytes(const vocr_textdet *m, unsigned max_width, unsigned max_height);

/* finder == VOCR_LINES_TEXTDET: the detector's horizontal lines in its order (vertical lines are
 * dropped; count 0 means no text), its truncated flag and return codes, m required. Any other
 * finder: vocr_find_lines(im, finder, arena, arena_bytes, out), unchanged (m is not read). */
int vocr_find_lines_det(const VocrLinesImage *im, int finder, const vocr_textdet *m, void *arena,
                        size_t arena_bytes, VocrLineSet *out);

/* The row-callback form: vocr_textdet_find_rows or vocr_find_lines_rows. */
int vocr_find_lines_det_rows(const VocrLinesRows *src, int finder, const vocr_textdet *m, void *arena,
                             size_t arena_bytes, VocrLineSet *out);

#ifdef __cplusplus
}
#endif

#endif
