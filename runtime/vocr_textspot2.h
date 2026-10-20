/* The full-frame text spotter for VTSP version 1 and 2 blobs (issue #106, docs/textspot2.md "d2s").
 *
 * Version 1 is textspot's spotter (issue #104, vocr_textspot.h, docs/textspot.md): #97's int8 detector network with
 * four maps per cell (K, O, A1, A2) inside a VTSP blob; for a version-1 blob every call here gives what
 * vocr_textspot.c gives, bit for bit (boxes, blocks, order, pick, truncated flag, dark flags, the arena sizes).
 * Version 2 (d2s) is a stride-8 trunk whose head has 16 channels, K, O, A1, A2 for each of a cell's 2 x 2 sub-cells
 * (channel 4 (2 dy + dx) + m), which the decode reads as the four maps at stride 4, cropped to ceil(h / 4) x
 * ceil(w / 4) of the network's input; the decode is textspot's, unchanged. synth/textspot2_twin.py specifies the
 * format and is the bit-exact Python twin of both versions.
 *
 * The same results, flags, limits and error codes as vocr_textspot.h (its types are used here), so a caller can
 * switch from vocr_textspot_* to vocr_textspot2_* without other changes. vocr_textspot.c is not linked: this file is
 * self-contained (link vocr_lines.c and vocr_lines_rows.c for the dark flags).
 *
 * C99, no allocation, no recursion, no global state, integer arithmetic only. The caller passes an arena of
 * vocr_textspot2_arena_bytes() bytes (no alignment needed); the frame is read row by row in one pass, so the arena
 * depends on the frame's width, not its height.
 */
#ifndef VOCR_TEXTSPOT2_H
#define VOCR_TEXTSPOT2_H

#include <stddef.h>

#include "vocr_textspot.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VOCR_TEXTSPOT2_VERSION_D2S 2u   /* the VTSP version of a 2 x 2 sub-pixel head */

/* A loaded blob: about 1 KB on 32-bit ARM. It points into the blob, which must stay alive and unchanged while
 * the model is used; the blob needs no alignment. Read-only after vocr_textspot2_load, so threads can share it
 * (each with its own arena). */
typedef struct {
    vocr_textspot_layer layers[VOCR_TEXTSPOT_MAX_LAYERS];
    unsigned n_layers;
    unsigned version;           /* the VTSP version: 1, or 2 (VOCR_TEXTSPOT2_VERSION_D2S) */
    unsigned outputs;           /* the head's channels: 4, or 16 for version 2 */
    unsigned net_stride;        /* the network's output stride: 4 or 8 (version 2: 8) */
    unsigned stride;            /* the stride of the maps the decode reads: net_stride, or 4 for version 2 */
    unsigned min_cells;         /* the decode drops kernel components of fewer cells */
    unsigned channels;          /* output channels of all layers */
    unsigned half_above;        /* a frame wider than this is halved before the network (0: never) */
    unsigned conf_min;          /* candidates of a lower confidence are dropped (1..127) */
    unsigned long blob_bytes;   /* the VTSP blob */
    unsigned long net_bytes;    /* its VTDT network */
    unsigned char ckpt[8];      /* sha256 prefix of the network's source checkpoint */
    char name[33];              /* the network's shape name, NUL-terminated */
    unsigned loaded;
} vocr_textspot2;

/* Checks the blob: a version-1 blob exactly as vocr_textspot_load does; a version-2 blob the same header checks
 * (magic, version 2, size, CRC-32, the reserved fields, the network's offset and size, conf_min 1..127) and every
 * check of the inner VTDT blob with outputs 16 and output stride 8 (header fields, the layer chain to stride 8 and
 * 16 channels without ReLU, section bounds and alignment, multipliers > 0, shifts 1..62). Points m into it.
 * VOCR_LINES_OK, VOCR_LINES_E_ARG (a NULL pointer) or VOCR_TEXTSPOT_E_BLOB; on an error m is cleared. */
int vocr_textspot2_load(vocr_textspot2 *m, const void *blob, size_t bytes);

/* Exact arena bytes for every frame up to max_width x max_height (the width matters only), as
 * vocr_textspot_arena_bytes: for a version-1 blob the same number. 0 for a NULL or unloaded model, a zero side
 * or a side over VOCR_TEXTSPOT_MAX_SIDE. */
size_t vocr_textspot2_arena_bytes(const vocr_textspot2 *m, unsigned max_width, unsigned max_height);

/* Exact arena bytes for a frame of exactly width x height (what the calls check), as
 * vocr_textspot_frame_arena_bytes. */
size_t vocr_textspot2_frame_arena_bytes(const vocr_textspot2 *m, unsigned width, unsigned height);

/* The spotter on a frame in memory (VOCR_LINES_GREY or VOCR_LINES_RGBA), as vocr_textspot_find: flags 0 is
 * synth/textspot2_twin.spot exactly; VOCR_TEXTSPOT_NO_DARK leaves every dark flag 0 and reads no box. */
int vocr_textspot2_find(const vocr_textspot2 *m, const VocrLinesImage *im, unsigned flags, void *arena,
                        size_t arena_bytes, VocrTextspotSet *out);

/* The same through a row callback, as vocr_textspot_find_rows: the frame's rows once each in increasing order in
 * one pass, then (without VOCR_TEXTSPOT_NO_DARK) each box's rows once, top to bottom, boxes in output order. */
int vocr_textspot2_find_rows(const vocr_textspot2 *m, const VocrLinesRows *src, unsigned flags, void *arena,
                             size_t arena_bytes, VocrTextspotSet *out);

/* Debug and test hook: runs the network on the frame (halved when the model says so) and calls
 * row(user, i, maps, width) for each row i of the maps the decode reads, in increasing order: the head's own rows
 * for version 1 (as vocr_textspot_maps), the stride-4 rows of the d2s head for version 2; maps holds the row's K,
 * O, A1 and A2 one after the other ([0, width), ..., [3 width, 4 width)), valid only during the call. */
int vocr_textspot2_maps(const vocr_textspot2 *m, const VocrLinesImage *im, void *arena, size_t arena_bytes,
                        vocr_textspot_map_row row, void *user);

#ifdef __cplusplus
}
#endif

#endif
