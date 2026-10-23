/* vocr_charbox.h - per-character positions from the recognizer's CTC frames (vita-vn-ocr #114,
 * docs/char-boxes.md).
 *
 * vocr_run() emits one CTC frame per stride (8) resized columns; greedy decoding emits a character at the
 * first frame of each run of its class. vocr_charbox_find() turns a line's per-frame classes (vocr_extra.best,
 * or topk_class with stride K) into one box per greedy character along the reading direction, in the crop's
 * continuous pixel coordinates (pixel i spans [i, i + 1)), fixed point VOCR_CHARBOX_Q (1/16 px):
 *
 *   c        the centre: of the first frame of the run, stride * f0 + stride / 2 resized px (VOCR_CHARBOX_RUN:
 *            of the whole run), plus the model's calibration (a shift, and an offset for the character's group),
 *            times w / nw (INTER_LINEAR's map from the resized line to the crop), clipped to [0, w]
 *   x0, x1   the cell: the line cut between neighbouring centres; the first and last cells reach half a pitch (the
 *            median spacing of the centres) beyond their centres. The cells tile the line, for hit-testing
 *   i0, i1   the ink box: the cell, or with VOCR_CHARBOX_INK its columns that hold ink
 *
 * VOCR_CHARBOX_INK moves each cut to the emptiest run of columns between the two centres (the column profile
 * is sum over rows of |g(y + 1) - g(y)|, polarity-free), shrinks each ink box to its cell's columns above a
 * quarter of the line's peak, and puts the centre in the middle of that ink. VOCR_CHARBOX_PITCH moves the
 * centres of a regular line (at least 4 characters, every spacing within 25% of the median) to their
 * least-squares grid before the cells are cut. A vertical column read turned 90 degrees anticlockwise
 * (vocr_column_rotate) gives positions down the column.
 *
 * eval/charbox.py is the specification: this file computes its numbers bit for bit (tests/test_charbox.py).
 * C99, no allocation, no recursion, no global state, integer arithmetic (int64 where a product needs it);
 * the caller's output array and, optionally, a column profile of w entries are the only memory. Nothing in
 * vocr.c changes: a caller that never calls this file gets exactly what it got before.
 */
#ifndef VOCR_CHARBOX_H
#define VOCR_CHARBOX_H

#include <stddef.h>
#include <stdint.h>
#include "vocr.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VOCR_CHARBOX_MAX 256u   /* characters per line (a 1024-px line has at most 128 frames) */
#define VOCR_CHARBOX_Q 16       /* 1/16 px */
#define VOCR_CHARBOX_GROUPS 8

enum { VOCR_CHARBOX_RUN = 1, VOCR_CHARBOX_PITCH = 2, VOCR_CHARBOX_INK = 4 };

/* Calibration groups (eval/charbox.py group()). */
enum {
    VOCR_CHARBOX_G_OTHER = 0, /* every other character */
    VOCR_CHARBOX_G_CLOSE,     /* 、。，．」』）】〉》〕｝］〙〗”’ */
    VOCR_CHARBOX_G_OPEN,      /* 「『（【〈《〔｛［〘〖“‘ */
    VOCR_CHARBOX_G_DOTS,      /* …‥・ */
    VOCR_CHARBOX_G_SMALL,     /* small kana */
    VOCR_CHARBOX_G_LONG,      /* ー―─～〜－ */
    VOCR_CHARBOX_G_MARK,      /* ！？♪☆★♥♡※ */
    VOCR_CHARBOX_G_HALF       /* ASCII and half-width katakana */
};

typedef struct {               /* a model's calibration (eval/charboxes_cal.json), resized px x 16 */
    int16_t shift;             /* added to every centre */
    int16_t group[VOCR_CHARBOX_GROUPS]; /* added to the centre of a character of the group */
} vocr_charbox_cal;

typedef struct {               /* one greedy character of a line */
    uint16_t cls;              /* its class (vocr_class_text) */
    uint16_t f0, f1;           /* the frames of its run (frames above 65535 do not fit: vocr_run gives at most 129) */
    uint16_t group;            /* VOCR_CHARBOX_G_* */
    int32_t c, x0, x1, i0, i1; /* crop px x VOCR_CHARBOX_Q along the reading direction */
} vocr_charbox;

/* The calibration group of a code point. */
unsigned vocr_charbox_group(uint32_t cp);

/* The column profile of a line crop (GREY8 or RGBA8, VOCR_INVERT ignored): prof[x] = sum over y < h - 1 of
 * |g(y + 1, x) - g(y, x)|, w entries. 0 or VOCR_E_ARG. */
int vocr_charbox_profile(const vocr_image *img, uint32_t *prof);

/* The boxes of the greedy characters of one line crop of w x h px read by model m.
 *   cls        the frames' classes: cls[f * cls_stride] for f < frames (vocr_extra.best with stride 1, or
 *              topk_class with stride vocr_extra.topk: its top-1)
 *   flags      VOCR_CHARBOX_RUN | VOCR_CHARBOX_PITCH | VOCR_CHARBOX_INK
 *   cal        the model's calibration, or NULL (all zeros)
 *   img, prof  for VOCR_CHARBOX_INK: the crop (w x h, as read: GREY8 or RGBA8; for a column the turned strip)
 *              or its profile (w entries, vocr_charbox_profile); prof wins when both are given. Unused without
 *              the flag
 *   out, cap   at most cap (<= VOCR_CHARBOX_MAX) characters, the line's first ones; *n gets their count
 * Returns 0, or VOCR_E_ARG (NULL pointers, w, h or frames 0, cap 0 or over VOCR_CHARBOX_MAX, INK without an
 * image or profile, an image not w x h, a class outside the model). */
int vocr_charbox_find(const vocr_model *m, const uint16_t *cls, unsigned cls_stride, unsigned frames, unsigned w,
                      unsigned h, unsigned flags, const vocr_charbox_cal *cal, const vocr_image *img,
                      const uint32_t *prof, vocr_charbox *out, unsigned cap, unsigned *n);

/* Another text's characters placed on a line's characters: dst[j] gets the index of the src character it is
 * aligned to, or -1 (an inserted character, which has no box of its own). src and dst are code points, n and m
 * of them (at most 32767 each: the map is int16). Equal characters align; at a difference the shortest skip of at most
 * VOCR_CHARBOX_SYNC characters on either side after which two characters agree (or a text ends) is taken,
 * substitutions first, else the two characters are taken as a substitution (eval/charbox.py align_map).
 * Returns 0, or VOCR_E_ARG. */
#define VOCR_CHARBOX_SYNC 8u
int vocr_charbox_align(const uint32_t *src, unsigned n, const uint32_t *dst, unsigned m, int16_t *map);

#ifdef __cplusplus
}
#endif
#endif
