/* The full-frame text spotter for VTSP version 1, 2 and 3 blobs (issue #108, docs/textspot3.md "Two families").
 *
 * Versions 1 and 2 are textspot's and textspot2's spotters (vocr_textspot.h, vocr_textspot2.h): for a version-1 or
 * version-2 blob every call here gives what vocr_textspot2.c gives, bit for bit (boxes, blocks, order, pick,
 * truncated flag, dark flags, the arena sizes). Version 3 is the multi-scale family M: a stride-8 trunk streamed
 * by rows into a whole stride-8 map, then layers on whole maps (3x3 depthwise layers of stride 1 or 2 with a
 * dilation; 1x1 layers over the concatenation of up to three earlier maps, each read through a nearest
 * upsampling by 2^u), ending in version 2's 16-channel d2s head, whose rows the decode reads as stride-4 maps. A
 * version-3 frame is halved when it is wider than half_above or taller than half_above_h. synth/textspot3_twin.py
 * specifies the format and is the bit-exact Python twin of all three versions.
 *
 * The same results, flags, limits and error codes as vocr_textspot.h (its types are used here), so a caller can
 * switch from vocr_textspot_* or vocr_textspot2_* to vocr_textspot3_* without other changes. vocr_textspot.c and
 * vocr_textspot2.c are not linked: this file is self-contained (link vocr_lines.c and vocr_lines_rows.c for the
 * dark flags).
 *
 * C99, no allocation, no recursion, no global state, integer arithmetic only. The caller passes an arena of
 * vocr_textspot3_arena_bytes() bytes (no alignment needed). Versions 1 and 2 read the frame row by row in one pass,
 * so their arena depends on the frame's width only; a version-3 arena also holds whole maps, so it depends on
 * both sides.
 */
#ifndef VOCR_TEXTSPOT3_H
#define VOCR_TEXTSPOT3_H

#include <stddef.h>

#include "vocr_textspot.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VOCR_TEXTSPOT3_VERSION_D2S 2u   /* the VTSP version of a 2 x 2 sub-pixel head on a stride-8 trunk */
#define VOCR_TEXTSPOT3_VERSION_MS 3u    /* the VTSP version of the multi-scale family M */
#define VOCR_TEXTSPOT3_MAX_OPS 64u      /* map ops of a version-3 network (the head is the last) */
#define VOCR_TEXTSPOT3_MAX_MAPS 65u     /* maps: the stream's output, then one per op */

/* A map op of a version-3 network, as loaded (pointers into the blob), with the buffer plan of its output. */
typedef struct {
    const unsigned char *w;     /* int8 weights [cout][cin, or 1 if depthwise][k][k], in the blob */
    const unsigned char *b;     /* little-endian int32 bias[cout] */
    const unsigned char *m;     /* little-endian int32 multiplier[cout] (> 0) */
    const unsigned char *sh;    /* u8 shift[cout] (1..62) */
    unsigned short cin, cout;   /* cin: the sum of its sources' channels (a depthwise op: cout) */
    unsigned short out_stride;  /* network-input px per cell of its output map */
    unsigned char kind;         /* 0: a 1x1 layer over 1..3 sources, 1: a 3x3 depthwise layer */
    unsigned char stride, dil, relu, nsrc;
    unsigned char src[3], up[3];   /* its source maps and their upsampling exponents */
    unsigned char inplace;      /* 1: its output overwrites its dying source's buffer */
    unsigned char slot;         /* its output's buffer (0xFF: the head, whose rows go to the decode) */
} vocr_textspot3_op;

/* A loaded blob: about 4 KB on 32-bit ARM. It points into the blob, which must stay alive and unchanged while
 * the model is used; the blob needs no alignment. Read-only after vocr_textspot3_load, so threads can share it
 * (each with its own arena). */
typedef struct {
    vocr_textspot_layer layers[VOCR_TEXTSPOT_MAX_LAYERS];   /* versions 1, 2: the network; 3: the stream */
    unsigned n_layers;
    unsigned version;           /* the VTSP version: 1, 2 or 3 */
    unsigned outputs;           /* the head's channels: 4, or 16 for versions 2 and 3 */
    unsigned net_stride;        /* the network's output stride: 4 or 8 (versions 2 and 3: 8) */
    unsigned stride;            /* the stride of the maps the decode reads: net_stride, or 4 for versions 2, 3 */
    unsigned min_cells;         /* the decode drops kernel components of fewer cells */
    unsigned channels;          /* output channels of all layers (version 3: of the stream's layers) */
    unsigned half_above;        /* a frame wider than this is halved before the network (0: never by its width) */
    unsigned half_above_h;      /* version 3: a frame taller than this is halved (0: never by its height); else 0 */
    unsigned conf_min;          /* candidates of a lower confidence are dropped (1..127) */
    unsigned long blob_bytes;   /* the VTSP blob */
    unsigned long net_bytes;    /* its network (a VTDT blob, or version 3's VTMS section) */
    unsigned char ckpt[8];      /* sha256 prefix of the network's source checkpoint */
    char name[33];              /* the network's shape name, NUL-terminated */
    /* version 3 */
    unsigned n_ops, n_slots;
    vocr_textspot3_op ops[VOCR_TEXTSPOT3_MAX_OPS];
    unsigned short map_stride[VOCR_TEXTSPOT3_MAX_MAPS];  /* map 0: the stream's output, map k + 1: op k's */
    unsigned short map_chans[VOCR_TEXTSPOT3_MAX_MAPS];
    signed char last_use[VOCR_TEXTSPOT3_MAX_MAPS];       /* the last op reading the map; -1: none */
    unsigned char map_slot[VOCR_TEXTSPOT3_MAX_MAPS];     /* the buffer holding it (0xFF: the head's output) */
    unsigned short slot_cap[VOCR_TEXTSPOT3_MAX_MAPS];    /* a buffer's channels per cell: its maps' most */
    unsigned short slot_stride[VOCR_TEXTSPOT3_MAX_MAPS]; /* the stride of its maps */
    unsigned loaded;
} vocr_textspot3;

/* Checks the blob: a version-1 or version-2 blob exactly as vocr_textspot2_load does; a version-3 blob every check
 * of synth/textspot3_twin.Model (the VTSP header with half_above_h; the VTMS section's header, CRC, counts and
 * name; the stream's layer records, chained to stride 8 and a multiple of 8 channels; every map op's record: its
 * kind, stride, dilation, sources (earlier maps meeting at one stride), channels, sections; weights in -127..127,
 * multipliers > 0, shifts 1..62; the last op the 16-channel head at stride 8 without ReLU). Points m into it.
 * VOCR_LINES_OK, VOCR_LINES_E_ARG (a NULL pointer) or VOCR_TEXTSPOT_E_BLOB; on an error m is cleared. */
int vocr_textspot3_load(vocr_textspot3 *m, const void *blob, size_t bytes);

/* Exact arena bytes for every frame up to max_width x max_height: for a version-1 or version-2 blob
 * vocr_textspot2_arena_bytes' number (the width matters only); for version 3 the most any frame up to that size
 * needs (the frame of min(max_width, half_above) x min(max_height, half_above_h), read at full size, or the
 * max_width x max_height frame itself, halved). 0 for a NULL or unloaded model, a zero side, a side over
 * VOCR_TEXTSPOT_MAX_SIDE, or (version 3) a need over 2^31 - 256 bytes. */
size_t vocr_textspot3_arena_bytes(const vocr_textspot3 *m, unsigned max_width, unsigned max_height);

/* Exact arena bytes for a frame of exactly width x height (what the calls check). */
size_t vocr_textspot3_frame_arena_bytes(const vocr_textspot3 *m, unsigned width, unsigned height);

/* The spotter on a frame in memory (VOCR_LINES_GREY or VOCR_LINES_RGBA), as vocr_textspot_find: flags 0 is
 * synth/textspot3_twin.spot exactly; VOCR_TEXTSPOT_NO_DARK leaves every dark flag 0 and reads no box. */
int vocr_textspot3_find(const vocr_textspot3 *m, const VocrLinesImage *im, unsigned flags, void *arena,
                        size_t arena_bytes, VocrTextspotSet *out);

/* The same through a row callback, as vocr_textspot_find_rows: the frame's rows once each in increasing order in
 * one pass, then (without VOCR_TEXTSPOT_NO_DARK) each box's rows once, top to bottom, boxes in output order. */
int vocr_textspot3_find_rows(const vocr_textspot3 *m, const VocrLinesRows *src, unsigned flags, void *arena,
                             size_t arena_bytes, VocrTextspotSet *out);

/* Debug and test hook: runs the network on the frame (halved when the model says so) and calls
 * row(user, i, maps, width) for each row i of the maps the decode reads, in increasing order: the head's own rows
 * for version 1, the stride-4 rows of the d2s head for versions 2 and 3; maps holds the row's K, O, A1 and A2 one
 * after the other ([0, width), ..., [3 width, 4 width)), valid only during the call. */
int vocr_textspot3_maps(const vocr_textspot3 *m, const VocrLinesImage *im, void *arena, size_t arena_bytes,
                        vocr_textspot_map_row row, void *user);

#ifdef __cplusplus
}
#endif

#endif
